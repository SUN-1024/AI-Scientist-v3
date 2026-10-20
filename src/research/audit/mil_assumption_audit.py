"""Finite, checkable diagnostics for max-pooling MIL assumption audits.

The diagnostics answer four questions about a supplied finite system:

1. which Boolean instance maps are consistent with the observed OR bags;
2. which points are forced positive, forced negative, or unresolved;
3. whether the bag-by-instance-type composition matrix is full rank;
4. what an exact target-blind adaptive query policy costs on the construction.

Everything is exact (integer arithmetic and ``fractions.Fraction``); there are no
third-party dependencies and no random draws.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

Map = tuple[int, ...]
Bag = tuple[int, ...]


def normalize_bags(bags: Iterable[Sequence[int]]) -> list[Bag]:
    normalized = []
    for bag in bags:
        indices = tuple(int(index) for index in bag)
        if not indices:
            raise ValueError("A bag must contain at least one instance")
        normalized.append(indices)
    return normalized


def bag_label(instance_map: Sequence[int], bag: Sequence[int]) -> int:
    return max(instance_map[index] for index in bag)


def compatible_maps(point_count: int, labelled_bags: Iterable[tuple[Sequence[int], int]]) -> list[Map]:
    """Enumerate maps consistent with OR-labelled bags, propagating negatives first."""
    forced: dict[int, int] = {}
    observations = [(normalize_bags([bag])[0], int(label)) for bag, label in labelled_bags]
    for bag, label in observations:
        if label == 0:
            for index in bag:
                if forced.get(index) == 1:
                    return []
                forced[index] = 0
    free = [index for index in range(point_count) if index not in forced]
    solutions: list[Map] = []
    for bits in product((0, 1), repeat=len(free)):
        instance_map = [forced.get(index, 0) for index in range(point_count)]
        for index, bit in zip(free, bits):
            instance_map[index] = bit
        if all(bag_label(instance_map, bag) == label for bag, label in observations):
            solutions.append(tuple(instance_map))
    return solutions


def certificate(point_count: int, labelled_bags: Iterable[tuple[Sequence[int], int]]) -> dict[str, Any]:
    """Monotone observed-negative / singleton-residual certificate."""
    observations = [(tuple(int(index) for index in bag), int(label)) for bag, label in labelled_bags]
    domain = set(range(point_count))
    negative_union = {index for bag, label in observations if label == 0 for index in bag}
    free = domain - negative_union
    residuals = [{index for index in bag} - negative_union for bag, label in observations if label == 1]
    forced_positive = {next(iter(residual)) for residual in residuals if len(residual) == 1}
    unresolved = free - forced_positive
    return {
        "negative_union": sorted(negative_union),
        "free": sorted(free),
        "positive_residuals": sorted(sorted(residual) for residual in residuals),
        "forced_positive": sorted(forced_positive),
        "forced_negative": sorted(negative_union),
        "unresolved": sorted(unresolved),
        "unique": bool(residuals) and not unresolved,
        "empty_positive_residual": any(not residual for residual in residuals),
    }


def version_space_labels(maps: Sequence[Map], point_count: int) -> dict[str, list[int]]:
    if not maps:
        return {"forced_one": [], "forced_zero": [], "unresolved": []}
    return {
        "forced_one": [index for index in range(point_count) if all(m[index] for m in maps)],
        "forced_zero": [index for index in range(point_count) if all(not m[index] for m in maps)],
        "unresolved": [index for index in range(point_count)
                       if any(m[index] for m in maps) and not all(m[index] for m in maps)],
    }


def composition_matrix(point_count: int, bags: Iterable[Sequence[int]]) -> list[list[int]]:
    rows = []
    for bag in normalize_bags(bags):
        row = [0] * point_count
        for index in bag:
            if not 0 <= index < point_count:
                raise ValueError("Bag index outside the point domain")
            row[index] += 1
        rows.append(row)
    return rows


def matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    """Exact rank over the rationals via fraction-free elimination."""
    rows = [[Fraction(int(value)) for value in row] for row in matrix if any(row)]
    if not rows:
        return 0
    width = max(len(row) for row in rows)
    for row in rows:
        row.extend([Fraction(0)] * (width - len(row)))
    rank = 0
    for column in range(width):
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for index in range(rank + 1, len(rows)):
            if rows[index][column] == 0:
                continue
            factor = rows[index][column] / rows[rank][column]
            rows[index] = [current - factor * base for current, base in zip(rows[index], rows[rank])]
        rank += 1
    return rank


def a4_a5_sets(points: Sequence[Sequence[Any]], labelled_bags: Iterable[tuple[Sequence[int], int]],
               causality: str = "T") -> dict[str, Any]:
    """A4 environment sets and A5 non-causal content sets, as printed by FocusMIL.

    ``points[i]`` is ``(content, environment)``. A4(i) requires at least two positive
    environments; A4(ii) requires positive environments to be contained in negative ones;
    A5 requires the positive-bag non-causal content set to be contained in the negative one.
    """
    observations = [(tuple(bag), int(label)) for bag, label in labelled_bags]
    positive = [bag for bag, label in observations if label == 1]
    negative = [bag for bag, label in observations if label == 0]

    def contents(bags):
        return {points[index][0] for bag in bags for index in bag}

    def environments(bags):
        return {points[index][1] for bag in bags for index in bag}

    positive_content = contents(positive) - {causality}
    negative_content = contents(negative) - {causality}
    positive_environment = environments(positive)
    negative_environment = environments(negative)
    return {
        "positive_environments": sorted(positive_environment),
        "negative_environments": sorted(negative_environment),
        "A4i_two_environments": len(positive_environment) >= 2,
        "A4ii_environment_subset": positive_environment <= negative_environment,
        "positive_noncausal_contents": sorted(positive_content),
        "negative_noncausal_contents": sorted(negative_content),
        "A5_content_subset": positive_content <= negative_content,
    }


def query_multisets(point_count: int, *, size: int = 3, shared_environment: bool = False,
                    points: Sequence[Sequence[Any]] | None = None) -> list[Bag]:
    queries = list(combinations_with_replacement(range(point_count), size))
    if not shared_environment:
        return queries
    if points is None:
        raise ValueError("Shared-environment queries require the point table")
    return [query for query in queries
            if len({points[index][1] for index in query}) == 1]


def _partition(maps: Sequence[Map], query: Bag) -> dict[int, frozenset[int]]:
    parts: dict[int, set[int]] = {}
    for index, instance_map in enumerate(maps):
        parts.setdefault(bag_label(instance_map, query), set()).add(index)
    return {label: frozenset(members) for label, members in parts.items()}


def adaptive_optimum(maps: Sequence[Map], queries: Sequence[Bag], *,
                     max_states: int = 200_000) -> dict[str, Any]:
    """Exact optimal worst-case and uniform-prior expected response counts.

    The returned means are exact fractions rendered as strings so the result is
    reproducible without floating point.
    """
    from functools import lru_cache

    full = frozenset(range(len(maps)))
    seen: set[frozenset[int]] = set()

    @lru_cache(maxsize=None)
    def minimax(state: frozenset[int]) -> int:
        seen.add(state)
        if len(seen) > max_states:
            raise ValueError("Version space too large for exact dynamic programming")
        if len(state) <= 1:
            return 0
        best = None
        for query in queries:
            parts = _partition([maps[index] for index in sorted(state)], query)
            if len(parts) == 1:
                continue
            options = []
            for members in parts.values():
                local = frozenset(sorted(state)[position] for position in members)
                options.append(minimax(local))
            candidate = 1 + max(options)
            best = candidate if best is None else min(best, candidate)
        return 0 if best is None else best

    @lru_cache(maxsize=None)
    def expected(state: frozenset[int]) -> Fraction:
        if len(state) <= 1:
            return Fraction(0)
        best = None
        ordered = sorted(state)
        for query in queries:
            parts = _partition([maps[index] for index in ordered], query)
            if len(parts) == 1:
                continue
            total = Fraction(1) + sum(
                Fraction(len(members), len(state)) * expected(frozenset(ordered[position] for position in members))
                for members in parts.values()
            )
            best = total if best is None else min(best, total)
        return Fraction(0) if best is None else best

    worst = minimax(full)
    mean = expected(full)
    return {"worst_case_responses": worst, "uniform_prior_mean": f"{mean.numerator}/{mean.denominator}",
            "uniform_prior_mean_float": float(mean), "queries": len(queries), "maps": len(maps)}


def _minimum_cover_size(elimination_masks: Sequence[int], full: int) -> int | None:
    """Exact minimum set-cover size over at most 2**competitors mask states."""
    if full == 0:
        return 0
    distance = {0: 0}
    frontier = [0]
    while frontier:
        next_frontier = []
        for mask in frontier:
            for query_mask in elimination_masks:
                combined = mask | query_mask
                if combined == full:
                    return distance[mask] + 1
                if combined not in distance:
                    distance[combined] = distance[mask] + 1
                    next_frontier.append(combined)
        frontier = next_frontier
    return None


def teaching_costs(maps: Sequence[Map], queries: Sequence[Bag], *,
                   target_index: int | None = None) -> dict[str, Any]:
    """Target-aware teaching cost: how many target-labelled bags identify a known target.

    A cover is a set of added bags such that every competing map disagrees with the target on at
    least one of their labels. Costs are reported per compatible target because they are
    target-dependent; ``worst_case`` and ``uniform_prior_mean`` aggregate over the version space.
    """
    if not maps:
        raise ValueError("Teaching costs need a non-empty version space")
    targets = range(len(maps)) if target_index is None else [target_index]
    costs: dict[int, int | None] = {}
    for position in targets:
        target = maps[position]
        competitors = [candidate for index, candidate in enumerate(maps) if index != position]
        if not competitors:
            costs[position] = 0
            continue
        masks = []
        for query in queries:
            mask = 0
            for bit, candidate in enumerate(competitors):
                if bag_label(candidate, query) != bag_label(target, query):
                    mask |= 1 << bit
            masks.append(mask)
        costs[position] = _minimum_cover_size(masks, (1 << len(competitors)) - 1)
    values = [cost for cost in costs.values() if cost is not None]
    mean = Fraction(sum(values), len(values)) if values else None
    return {
        "per_target": {str(position): cost for position, cost in sorted(costs.items())},
        "per_target_list": [costs[position] for position in sorted(costs)],
        "unresolved_targets": sum(1 for cost in costs.values() if cost is None),
        "worst_case": max(values) if values else None,
        "uniform_prior_mean": f"{mean.numerator}/{mean.denominator}" if mean is not None else None,
        "uniform_prior_mean_float": float(mean) if mean is not None else None,
    }


def diagnostic(points: Sequence[Sequence[Any]], labelled_bags: Iterable[tuple[Sequence[int], int]],
               *, causality: str = "T", permutation: Sequence[int] | None = None,
               query_size: int = 3, compute_queries: bool = True) -> dict[str, Any]:
    """Run every finite diagnostic and return one JSON-serialisable report."""
    observations = [(tuple(int(index) for index in bag), int(label)) for bag, label in labelled_bags]
    point_count = len(points)
    maps = compatible_maps(point_count, observations)
    matrix = composition_matrix(point_count, [bag for bag, _ in observations])
    report: dict[str, Any] = {
        "schema_version": 1,
        "point_count": point_count,
        "bag_count": len(observations),
        "compatible_maps": len(maps),
        "positive_supports": sorted(sorted(index for index, bit in enumerate(instance_map) if bit)
                                    for instance_map in maps),
        "certificate": certificate(point_count, observations),
        "version_space_labels": version_space_labels(maps, point_count),
        "composition_matrix": matrix,
        "composition_rank": matrix_rank(matrix),
        "composition_full_rank": matrix_rank(matrix) == point_count,
        "a4_a5": a4_a5_sets(points, observations, causality=causality),
    }
    if permutation is not None:
        if sorted(permutation) != list(range(point_count)):
            raise ValueError("Permutation must be a bijection on the point domain")
        permuted_observations = [(tuple(permutation[index] for index in bag), label)
                                 for bag, label in observations]
        report["permutation"] = {
            "map": list(permutation),
            "preserves_bag_multisets": sorted(tuple(sorted(bag)) for bag, _ in permuted_observations) ==
                                       sorted(tuple(sorted(bag)) for bag, _ in observations),
            "permuted_a4_a5": a4_a5_sets(points, permuted_observations, causality=causality),
        }
    if compute_queries and maps:
        for name, shared in (("shared_environment", True), ("mixed_environments", False)):
            queries = query_multisets(point_count, size=query_size, shared_environment=shared, points=points)
            try:
                report.setdefault("query_optima", {})[name] = adaptive_optimum(maps, queries)
                report.setdefault("teaching_costs", {})[name] = teaching_costs(maps, queries)
            except ValueError as exc:
                report.setdefault("query_optima", {})[name] = {"status": "skipped", "reason": str(exc)}
        report["query_optima_target_note"] = (
            "Queries are labelled by the target instance map; a target-blind policy conditions only on "
            "previous responses. The optimum is independent of which consistent map is chosen as target "
            "because the version space is symmetric under relabelling of the consistent maps."
        )
    return report


def _load_task(path: Path) -> dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(task, dict) or "points" not in task or "bags" not in task:
        raise ValueError("Task JSON needs 'points' and 'bags'")
    bags = []
    for row in task["bags"]:
        if isinstance(row, dict):
            bags.append((row["x"], row["y"]))
        else:
            bags.append((row[0], row[1]))
    return {"points": task["points"], "bags": bags,
            "causality": task.get("causality", "T"),
            "permutation": task.get("permutation"),
            "query_size": int(task.get("query_size", 3)),
            "compute_queries": bool(task.get("compute_queries", True))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path, help="JSON task with points, bags, optional permutation")
    parser.add_argument("--output", type=Path, help="Write the report here instead of stdout")
    args = parser.parse_args(argv)
    task = _load_task(args.task)
    report = diagnostic(task["points"], task["bags"], causality=task["causality"],
                        permutation=task["permutation"], query_size=task["query_size"],
                        compute_queries=task["compute_queries"])
    body = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
