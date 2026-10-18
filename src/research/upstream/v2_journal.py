"""V2 journal, metric ordering, and debug-branch semantics adapted for V3.

Derived from SakanaAI/AI-Scientist-v2, commit
96bd51617cfdbb494a9fc283af00fe090edfae48, ai_scientist/treesearch/journal.py
(Node.stage_name/debug_depth, Journal node filters/metric selection) and
ai_scientist/treesearch/utils/metric.py (MetricValue ordering).
Copyright Sakana AI. AI Scientist Source Code License v1.0, December 2025.
Full license: third_party/ai_scientist_v2/LICENSE.

Modifications: use host-validated V3 records; remove provider, NumPy, rich and
dataclasses_json dependencies; reject non-finite/incomparable metrics; never
average different metrics; exclude replication nodes from promotion; validate
the graph; use deterministic ties; retain all analysis when rendering evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering
import math
from statistics import fmean
from typing import Any, Mapping


@total_ordering
@dataclass(frozen=True)
class MetricValue:
    """Compare which value is better, including a loss where smaller is better.

    A mapping contains one metric measured on the same named datasets. It is
    never a collection of unrelated quantities such as accuracy and loss.
    """

    value: float | Mapping[str, float] | None
    maximize: bool = field(kw_only=True)
    name: str = field(kw_only=True)

    def __post_init__(self) -> None:
        if type(self.maximize) is not bool or not isinstance(self.name, str) or not self.name:
            raise ValueError("Metric name and explicit optimization direction are required")
        values = self.value.values() if isinstance(self.value, Mapping) else [self.value]
        if isinstance(self.value, Mapping):
            if not self.value or any(not isinstance(key, str) or not key for key in self.value):
                raise ValueError("A multi-dataset metric needs named datasets")
            # Freeze a private copy: caller mutations cannot change an ordering.
            object.__setattr__(self, "value", dict(self.value))
        for item in values:
            if item is None and self.value is None:
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError("Metric values must be finite numbers")

    def _check_comparable(self, other: MetricValue) -> None:
        if self.name != other.name or self.maximize != other.maximize:
            raise ValueError("Cannot compare metrics with different names or directions")
        if self.value is not None and other.value is not None:
            left_map, right_map = isinstance(self.value, Mapping), isinstance(other.value, Mapping)
            if left_map != right_map:
                raise ValueError("Cannot compare metrics evaluated on different dataset sets")
            if isinstance(self.value, Mapping) and isinstance(other.value, Mapping) and set(self.value) != set(other.value):
                raise ValueError("Cannot compare metrics evaluated on different dataset sets")

    def get_mean_value(self) -> float | None:
        if self.value is None:
            return None
        return fmean(self.value.values()) if isinstance(self.value, Mapping) else float(self.value)

    @property
    def is_worst(self) -> bool:
        return self.value is None

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        self._check_comparable(other)
        if self.value is None:
            return other.value is not None
        if other.value is None:
            return False
        left, right = self.get_mean_value(), other.get_mean_value()
        if left is None or right is None:
            raise ValueError("Metric changed during comparison")
        return left < right if self.maximize else left > right

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricValue):
            return NotImplemented
        self._check_comparable(other)
        return self.get_mean_value() == other.get_mean_value()


@dataclass(eq=False)
class Node:
    """The V2 solution-node semantics without execution authority."""

    id: str
    metric: MetricValue
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    is_buggy: bool = False
    is_buggy_plots: bool = False
    is_seed_node: bool = False
    analysis: str = ""
    vlm_feedback_summary: list[str] = field(default_factory=list)

    @property
    def stage_name(self) -> str:
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def debug_depth(self) -> int:
        depth, current = 0, self
        while current.parent is not None and current.parent.is_buggy:
            depth += 1
            current = current.parent
        return depth


@dataclass
class Journal:
    """A read-only selection view over the host's complete experiment graph."""

    nodes: list[Node] = field(default_factory=list)

    @classmethod
    def from_search_records(cls, records: Mapping[str, Mapping[str, Any]]) -> Journal:
        nodes: dict[str, Node] = {}
        for node_id, row in records.items():
            if not isinstance(node_id, str) or not node_id or row.get("node_id") != node_id:
                raise ValueError("Journal node IDs must match their host records")
            score = row.get("score")
            metric = MetricValue(score, maximize=True, name="combined_score")
            nodes[node_id] = Node(
                id=node_id, metric=metric,
                is_buggy=row.get("correct") is not True or metric.is_worst,
                is_buggy_plots=row.get("plot_check") == "failed",
                is_seed_node=row.get("is_replication") is True,
                analysis=str(row.get("analysis") or row.get("hypothesis") or ""),
                vlm_feedback_summary=list(row.get("visual_feedback") or []),
            )
        for node_id, row in records.items():
            parent_id = row.get("parent_id")
            if parent_id is not None:
                if parent_id not in nodes or parent_id == node_id:
                    raise ValueError("Journal parent is missing or self-referential")
                nodes[node_id].parent = nodes[parent_id]
                nodes[parent_id].children.append(nodes[node_id])
        for node in nodes.values():
            seen: set[str] = set()
            current: Node | None = node
            while current is not None:
                if current.id in seen:
                    raise ValueError("Journal contains a parent cycle")
                seen.add(current.id)
                current = current.parent
        return cls(list(nodes.values()))

    @property
    def draft_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        return [node for node in self.nodes if node.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        return [node for node in self.nodes if not node.is_buggy and not node.is_buggy_plots]

    def get_node_by_id(self, node_id: str) -> Node | None:
        return next((node for node in self.nodes if node.id == node_id), None)

    def ranked_nodes(self) -> list[Node]:
        # Stable first sort makes equal metrics deterministic, as required by V3
        # leases. V2's metric ordering still determines the scientific ranking.
        candidates = sorted((node for node in self.good_nodes if not node.is_seed_node), key=lambda node: node.id)
        return sorted(candidates, key=lambda node: node.metric, reverse=True)

    def get_best_node(self) -> Node | None:
        ranked = self.ranked_nodes()
        return ranked[0] if ranked else None

    def debug_candidates(self, *, max_depth: int = 3) -> list[Node]:
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("Debug depth must be a positive integer")
        return sorted((node for node in self.buggy_nodes
                       if node.is_leaf and node.debug_depth < max_depth and not node.is_seed_node),
                      key=lambda node: (node.debug_depth, node.id))

    def selection_evidence(self) -> list[dict[str, Any]]:
        """Keep analysis and visual evidence even when a metric exists (V2 fix)."""
        return [{"node_id": node.id, "metric_name": node.metric.name,
                 "metric_value": node.metric.value, "maximize": node.metric.maximize,
                 "analysis": node.analysis, "visual_feedback": node.vlm_feedback_summary,
                 "stage": node.stage_name, "debug_depth": node.debug_depth,
                 "eligible": not node.is_buggy and not node.is_buggy_plots and not node.is_seed_node}
                for node in sorted(self.nodes, key=lambda item: item.id)]
