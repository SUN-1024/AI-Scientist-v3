"""Host-owned, bounded branching search over evaluator-checked programs.

The global incumbent remains a separate object. A lower-scoring but correct
candidate can seed another hypothesis without becoming the reported winner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.orchestrator import state as state_store
from src.research.upstream.v2_journal import Journal
from .harness_incumbent import _program_source_records

TREE_FILE = "search_tree.json"
PARENT_FILE = "search_parent.json"


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fingerprint(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Search parent is not a regular program directory: {path}")
    records = _program_source_records(path)
    if not records:
        raise ValueError(f"Search parent has no sources or contains symlinks: {path}")
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _source_group(node: dict[str, Any]) -> tuple[str, str]:
    """Only valid source digests may join otherwise separate experiment nodes."""
    digest = node.get("source_sha256")
    if (isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdefABCDEF" for char in digest)):
        return "source", digest.lower()
    # Failed attempts have no source. Do not merge those or malformed records.
    # A chosen program still has to pass _program_path's fingerprint check.
    return "node", node["node_id"]


class SearchTree:
    """One tree per worker; persistent cycle leases make retries deterministic."""

    def __init__(
        self, results_dir: Path, task_dir: Path, *, width: int = 4, explore_every: int = 3,
    ) -> None:
        if type(width) is not int or width < 2:
            raise ValueError("search_width must be an integer >= 2")
        if type(explore_every) is not int or explore_every < 2:
            raise ValueError("search_explore_every must be an integer >= 2")
        self.results_dir = results_dir.resolve()
        self.task_dir = task_dir.resolve()
        self.width = width
        self.explore_every = explore_every
        self.path = results_dir / TREE_FILE
        self.data: dict[str, Any] = {
            "schema_version": 1, "width": width, "explore_every": explore_every,
            "selection_policy_version": 2,
            "nodes": {}, "leases": {},
        }
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink() or self.path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Invalid search tree state file")
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(loaded, dict) or loaded.get("schema_version") != 1
                or not isinstance(loaded.get("nodes"), dict)
                or not isinstance(loaded.get("leases"), dict)
                or loaded.get("width") != width
                or loaded.get("explore_every") != explore_every
            ):
                raise ValueError("Search tree state/config mismatch; resume with the original policy")
            self.data = loaded
            self._validate_state()

    def _validate_state(self) -> None:
        version = self.data.get("selection_policy_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Unsupported search tree selection_policy_version")
        nodes = self.data["nodes"]
        if nodes and "root" not in nodes:
            raise ValueError("Search tree has no root")
        for key, node in nodes.items():
            if not isinstance(node, dict) or node.get("node_id") != key:
                raise ValueError("Invalid search tree node")
            if type(node.get("depth")) is not int or node["depth"] < 0:
                raise ValueError("Invalid search tree depth")
            parent = node.get("parent_id")
            if parent is not None and parent not in nodes:
                raise ValueError("Search tree parent is missing")
            if node.get("correct") is True and _score(node.get("score")) is None:
                raise ValueError("Correct search node has no finite score")
        if any(parent not in nodes for parent in self.data["leases"].values()):
            raise ValueError("Search tree lease points to a missing parent")

    def _exploration_pool(self, eligible: list[dict[str, Any]]) -> list[dict[str, Any]]:
        root = self.data["nodes"]["root"]
        if self.data.get("selection_policy_version", 1) == 1:
            # A missing version is the historical policy. Never upgrade a resume.
            pool = eligible[: self.width]
            if root not in pool:
                pool.append(root)
            return pool

        # ranked_nodes orders by score then node ID, so the first member is a
        # deterministic representative. The full graph and hypotheses stay intact.
        representatives: dict[tuple[str, str], dict[str, Any]] = {}
        for node in eligible:
            representatives.setdefault(_source_group(node), node)
        pool = list(representatives.values())[: self.width]
        if _source_group(root) not in {_source_group(node) for node in pool}:
            pool.append(root)
        return pool

    def _program_path(self, node: dict[str, Any]) -> Path:
        raw = node.get("program_dir")
        if not isinstance(raw, str):
            raise ValueError("Search tree node has no program path")
        path = Path(raw)
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in (self.results_dir, self.task_dir)):
            raise ValueError("Search tree program lies outside this worker's task/results")
        if _fingerprint(path) != node.get("source_sha256"):
            raise ValueError(f"Search parent sources changed after evaluation: {path}")
        return path

    def _save(self) -> None:
        if self.path.is_symlink():
            raise ValueError("Search tree state must not be a symlink")
        state_store.write_private_text(self.path, json.dumps(self.data, sort_keys=True, indent=2))

    def _portable_path(self, path: Path) -> dict[str, str]:
        resolved = path.resolve()
        if resolved.is_relative_to(self.results_dir):
            return {"path_scope": "results", "relative_program_dir": str(resolved.relative_to(self.results_dir))}
        if resolved.is_relative_to(self.task_dir):
            return {"path_scope": "task", "relative_program_dir": str(resolved.relative_to(self.task_dir))}
        raise ValueError("Search tree program lies outside this worker's task/results")

    def choose(
        self, cycle: int, incumbent: dict[str, Any], incumbent_dir: Path,
        *, selection_policy: str = "scheduled",
    ) -> dict[str, Any]:
        nodes = self.data["nodes"]
        if not nodes:
            nodes["root"] = {
                "node_id": "root", "parent_id": None, "depth": 0,
                "program_dir": str(incumbent_dir.resolve()),
                **self._portable_path(incumbent_dir),
                "source_sha256": _fingerprint(incumbent_dir),
                "score": _score(incumbent.get("combined_score")),
                "correct": incumbent.get("correct") is True,
                "hypothesis": "Starting incumbent", "cycle": 0,
            }
            # A task can legitimately begin with an unqualified baseline.
            if nodes["root"]["score"] is None:
                nodes["root"]["correct"] = False
        journal = Journal.from_search_records(nodes)
        leased = self.data["leases"].get(str(cycle))
        if leased is not None:
            chosen = nodes[leased]
        else:
            eligible = [nodes[node.id] for node in journal.ranked_nodes()]
            pool = self._exploration_pool(eligible)
            root = nodes["root"]
            if selection_policy == "incumbent":
                chosen = next((node for node in eligible if Path(node["program_dir"]).resolve() == incumbent_dir.resolve()), None)
                if chosen is None:
                    raise ValueError("Adaptive incumbent is missing from the validated search tree")
            elif selection_policy == "explore" or (selection_policy == "scheduled" and cycle % self.explore_every == 0):
                visits = list(self.data["leases"].values())
                if self.data.get("selection_policy_version", 1) == 2:
                    # Include leases to every member, even nodes now outside the
                    # eligible beam. A new copy cannot reset a source's visits.
                    group_visits = Counter(_source_group(nodes[node_id]) for node_id in visits)
                    visit_count = lambda node: group_visits[_source_group(node)]
                else:
                    visit_count = lambda node: visits.count(node["node_id"])
                chosen = min(
                    pool,
                    key=lambda node: (
                        visit_count(node), node["depth"],
                        -(node["score"] if node["score"] is not None else -math.inf),
                        node["node_id"],
                    ),
                )
            else:
                debug = [nodes[node.id] for node in journal.debug_candidates()
                         if nodes[node.id].get("program_dir")]
                chosen = eligible[0] if eligible else (debug[0] if debug else root)
            self.data["leases"][str(cycle)] = chosen["node_id"]
        self._program_path(chosen)
        self._save()
        selected_node = journal.get_node_by_id(chosen["node_id"])
        if selected_node is None:
            raise ValueError("Selected search node is missing from its journal")
        selection = {**chosen, "strategy": "tree", "selected_for_cycle": cycle,
                     "journal": {"upstream": "AI-Scientist-v2",
                                 "stage": selected_node.stage_name,
                                 "debug_depth": selected_node.debug_depth,
                                 "candidates": journal.selection_evidence()}}
        parent_file = self.results_dir / PARENT_FILE
        if parent_file.is_symlink():
            raise ValueError("Search parent record must not be a symlink")
        state_store.write_private_text(parent_file, json.dumps(selection, sort_keys=True, indent=2))
        return selection

    def record(
        self, *, cycle: int, idea: dict[str, Any],
        candidate: tuple[Path, str, float, bool] | None, exit_code: int,
    ) -> None:
        parent_id = self.data["leases"].get(str(cycle))
        if parent_id is None:
            raise ValueError("Search outcome has no parent lease")
        parent = self.data["nodes"][parent_id]
        node_id = f"cycle_{cycle:04d}"
        node: dict[str, Any] = {
            "node_id": node_id, "parent_id": parent_id, "depth": parent["depth"] + 1,
            "cycle": cycle, "idea_id": idea["idea_id"],
            "hypothesis": idea["hypothesis"], "exit_code": exit_code,
            "correct": False, "score": None, "program_dir": None, "source_sha256": None,
        }
        if candidate is not None:
            program, _, score, correct = candidate
            node.update(
                program_dir=str(program.resolve()), source_sha256=_fingerprint(program),
                score=_score(score), correct=correct is True and _score(score) is not None,
                **self._portable_path(program),
            )
            self._program_path(node)
        existing = self.data["nodes"].get(node_id)
        if existing is not None and existing != node:
            raise ValueError("A completed search cycle cannot be overwritten with a different result")
        self.data["nodes"][node_id] = node
        self._save()


def render_parent_context(results_dir: Path, cycle: int) -> str:
    """Give ideation the actual branch seed without overwriting the incumbent."""
    path = results_dir / PARENT_FILE
    if not path.is_file() or path.is_symlink():
        return "Use the best-known program as the starting implementation."
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("selected_for_cycle") != cycle:
        return "Use the best-known program as the starting implementation."
    scope = results_dir / "task" if selection.get("path_scope") == "task" else results_dir
    if selection.get("relative_program_dir"):
        selection["program_dir"] = os.path.relpath(scope / selection["relative_program_dir"], Path.cwd())
    return (
        "The host selected this branch as the starting implementation for this cycle. "
        "Design your hypothesis for this parent; the global best remains the comparison target. "
        "A lower search score does not by itself invalidate a scientific hypothesis.\n"
        + json.dumps(selection, ensure_ascii=False, indent=2)
    )
