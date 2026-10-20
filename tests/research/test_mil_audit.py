"""Finite diagnostics for max-pooling MIL assumption audits."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.audit.mil_assumption_audit import (
    a4_a5_sets, adaptive_optimum, bag_label, certificate, compatible_maps,
    composition_matrix, diagnostic, main, matrix_rank, normalize_bags, query_multisets,
)

POINTS = [["T", 0], ["T", 1], ["X", 0], ["X", 1], ["Y", 0], ["Y", 1], ["Z", 0], ["Z", 1]]
BAGS = [((0, 2, 6), 1), ((1, 5, 7), 1), ((4, 6, 6), 0), ((3, 7, 7), 0)]
REPAIR = [((0, 2, 6), 1), ((1, 5, 7), 1), ((2, 4, 6), 0), ((3, 5, 7), 0)]
PERMUTATION = [2, 5, 0, 3, 4, 1, 6, 7]


def test_bag_label_and_normalization() -> None:
    assert bag_label((1, 0, 1), (0, 2)) == 1
    assert bag_label((0, 1, 0), (0, 2)) == 0
    assert normalize_bags([[3, 1], [2]]) == [(3, 1), (2,)]
    with pytest.raises(ValueError):
        normalize_bags([[]])


def test_compatible_maps_reproduce_the_nine_map_version_space() -> None:
    maps = compatible_maps(len(POINTS), BAGS)
    assert len(maps) == 9
    supports = sorted(sorted(index for index, bit in enumerate(row) if bit) for row in maps)
    assert supports == [[0, 1], [0, 1, 2], [0, 1, 2, 5], [0, 1, 5], [0, 2, 5],
                        [0, 5], [1, 2], [1, 2, 5], [2, 5]]


def test_certificate_matches_the_version_space() -> None:
    cert = certificate(len(POINTS), BAGS)
    assert cert["negative_union"] == [3, 4, 6, 7]
    assert cert["free"] == [0, 1, 2, 5]
    assert cert["positive_residuals"] == [[0, 2], [1, 5]]
    assert cert["forced_positive"] == []
    assert cert["unresolved"] == [0, 1, 2, 5]
    assert cert["unique"] is False
    assert cert["empty_positive_residual"] is False


def test_repair_is_unique() -> None:
    maps = compatible_maps(len(POINTS), REPAIR)
    assert len(maps) == 1
    assert [index for index, bit in enumerate(maps[0]) if bit] == [0, 1]
    cert = certificate(len(POINTS), REPAIR)
    assert cert["forced_positive"] == [0, 1]
    assert cert["unresolved"] == []
    assert cert["unique"] is True


def test_composition_rank_is_four_for_both_systems() -> None:
    original = composition_matrix(len(POINTS), [bag for bag, _ in BAGS])
    repaired = composition_matrix(len(POINTS), [bag for bag, _ in REPAIR])
    assert matrix_rank(original) == 4
    assert matrix_rank(repaired) == 4
    assert original[0] == [1, 0, 1, 0, 0, 0, 1, 0]


def test_matrix_rank_edge_cases() -> None:
    assert matrix_rank([[0, 0], [0, 0]]) == 0
    assert matrix_rank([[1, 0], [0, 1]]) == 2
    assert matrix_rank([[1, 2], [2, 4]]) == 1
    assert matrix_rank([]) == 0


def test_a4_a5_hold_in_both_worlds() -> None:
    first = a4_a5_sets(POINTS, BAGS)
    assert first["A4i_two_environments"] is True
    assert first["A4ii_environment_subset"] is True
    assert first["A5_content_subset"] is True
    permuted = [((PERMUTATION[index] for index in bag), label) for bag, label in BAGS]
    second = a4_a5_sets(POINTS, permuted)
    assert second["A4i_two_environments"] is True
    assert second["A4ii_environment_subset"] is True
    assert second["A5_content_subset"] is True
    assert first["positive_environments"] == second["positive_environments"] == [0, 1]
    assert first["positive_noncausal_contents"] == second["positive_noncausal_contents"] == ["X", "Y", "Z"]


def test_permutation_preserves_bag_multisets() -> None:
    permuted = [tuple(PERMUTATION[index] for index in bag) for bag, _ in BAGS]
    assert sorted(map(sorted, permuted)) == sorted(map(sorted, [bag for bag, _ in BAGS]))


def test_query_families_have_the_declared_sizes() -> None:
    mixed = query_multisets(len(POINTS), points=POINTS)
    shared = query_multisets(len(POINTS), shared_environment=True, points=POINTS)
    assert len(mixed) == 120
    assert len(shared) == 40
    with pytest.raises(ValueError):
        query_multisets(len(POINTS), shared_environment=True)


def test_adaptive_optimum_is_four_and_ten_thirds() -> None:
    maps = compatible_maps(len(POINTS), BAGS)
    for shared in (True, False):
        queries = query_multisets(len(POINTS), shared_environment=shared, points=POINTS)
        result = adaptive_optimum(maps, queries)
        assert result["worst_case_responses"] == 4
        assert result["uniform_prior_mean"] == "10/3"


def test_diagnostic_report_is_json_serialisable_and_matches_records() -> None:
    report = diagnostic(POINTS, BAGS, permutation=PERMUTATION, compute_queries=False)
    assert json.loads(json.dumps(report)) == report
    assert report["compatible_maps"] == 9
    assert report["composition_rank"] == 4
    assert report["composition_full_rank"] is False
    assert report["permutation"]["preserves_bag_multisets"] is True
    assert report["certificate"]["unresolved"] == [0, 1, 2, 5]


def test_cli_writes_a_report(tmp_path: Path) -> None:
    task = tmp_path / "task.json"
    task.write_text(json.dumps({
        "points": POINTS,
        "bags": [{"x": list(bag), "y": label} for bag, label in BAGS],
        "permutation": PERMUTATION,
        "compute_queries": False,
    }))
    output = tmp_path / "report.json"
    assert main([str(task), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["compatible_maps"] == 9
    assert report["a4_a5"]["A5_content_subset"] is True


def test_rejects_bad_permutation_and_bag_index() -> None:
    with pytest.raises(ValueError):
        diagnostic(POINTS, BAGS, permutation=[0, 1, 2, 3, 4, 5, 6, 6], compute_queries=False)
    with pytest.raises(ValueError):
        composition_matrix(3, [[0, 5]])
