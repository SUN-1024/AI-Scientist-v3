"""Replay the host's comparison plan and evaluated control without new work."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.investigation.files import evaluation_results_complete
from src.variants_pipeline.execute import _inspect_program_results
from src.variants_pipeline.models import VariantRunResult, VariantSpec
from src.variants_pipeline.pipeline import (
    _load_variant_specs_from_disk,
    _validate_pipeline_summary,
    _write_pipeline_summary,
)
from src.variants_pipeline.rubric_contract import build_rubric_contract


def write_json(path: Path, value):
    path.write_text(json.dumps(value))


def comparison(tmp_path):
    root = tmp_path / "comparison with spaces"
    specs = []
    for name, kind in [("control_seed", "control"), ("z_baseline", "baseline"), ("a_baseline", "baseline")]:
        directory = root / "variants" / name
        program = directory / "task_bundle" / ("program_0" if kind == "control" else "program_1")
        program.mkdir(parents=True)
        spec = VariantSpec(name, name, kind, "best", directory,
                           description_body="Preserve the registered scientific comparison.")
        write_json(directory / "variant_spec.json", spec.to_dict())
        write_json(program / "metrics.json", {"combined_score": 0.5})
        write_json(program / "correct.json", {"correct": True})
        write_json(program / "evaluation_manifest.json", {
            "producer": "sandbox_host_eval", "execution_boundary": "brokered_search",
            "program_id": program.name, "command": ["trusted-host-evaluator"], "returncode": 0,
            "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:01Z",
        })
        rubrics = build_rubric_contract(name, kind)["rubrics"]
        write_json(program / "rubric_status.json", {
            "variant_id": name, "idea_id": name,
            "attempt_id": name + ("_attempt_00" if kind == "control" else "_attempt_01"),
            "overall_pass": True, "metrics": {"combined_score": 0.5, "correct": True},
            "rubrics": [{"id": r["id"], "pass": True, "evidence": "host checked"} for r in rubrics],
        })
        specs.append(spec)
    results = [VariantRunResult(s.variant_id, "completed", 1, True, 0, "", {}) for s in specs]
    _write_pipeline_summary(root, specs, results)
    return root, specs


def test_comparison_replays_plan_order_sources_and_brokered_control(tmp_path):
    root, original = comparison(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    loaded = _load_variant_specs_from_disk(root)
    assert [s.variant_id for s in loaded] != [s.variant_id for s in original]
    assert all(s.description_body == original[0].description_body for s in loaded)
    assert _validate_pipeline_summary(root, loaded) is None
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("field", ["title", "kind", "base_program", "description_body", "implementation_plan"])
def test_changed_materialized_science_remains_rejected(tmp_path, field):
    root, specs = comparison(tmp_path)
    path = specs[1].variant_dir / "variant_spec.json"
    data = json.loads(path.read_text())
    data[field] = ["changed"] if field == "implementation_plan" else "changed"
    write_json(path, data)
    assert "does not match" in _validate_pipeline_summary(root, _load_variant_specs_from_disk(root))


@pytest.mark.parametrize("change", ["duplicate", "missing", "renamed", "reordered"])
def test_unbound_summary_entries_remain_rejected(tmp_path, change):
    root, _ = comparison(tmp_path)
    path = root / "pipeline_summary.json"
    data = json.loads(path.read_text())
    if change == "duplicate":
        data["variants"].append(data["variants"][0])
    elif change == "missing":
        data["variants"].pop()
    elif change == "renamed":
        data["variants"][1]["variant_id"] = "unregistered"
    else:
        data["variants"].reverse()
    write_json(path, data)
    assert _validate_pipeline_summary(root, _load_variant_specs_from_disk(root)) is not None


@pytest.mark.parametrize(("field", "value"), [
    ("producer", "agent"), ("producer", None), ("execution_boundary", "unknown"),
    ("execution_boundary", ["brokered_search"]), ("returncode", 1), ("program_id", "program_9"),
    ("command", []), ("started_at", None), ("finished_at", None),
])
def test_brokered_completion_keeps_receipt_checks(tmp_path, field, value):
    root, specs = comparison(tmp_path)
    program = specs[0].variant_dir / "task_bundle/program_0"
    path = program / "evaluation_manifest.json"
    data = json.loads(path.read_text()); data[field] = value
    write_json(path, data)
    assert evaluation_results_complete(program) is False
    assert _validate_pipeline_summary(root, _load_variant_specs_from_disk(root)) is not None


@pytest.mark.parametrize("pending", [False, True])
def test_control_seed_needs_a_completed_host_evaluation(tmp_path, pending):
    root, specs = comparison(tmp_path)
    program = specs[0].variant_dir / "task_bundle/program_0"
    if pending:
        (program / ".evaluation_pending").touch()
    else:
        (program / "evaluation_manifest.json").unlink()
    assert _inspect_program_results(program.parent, specs[0].variant_id, include_evaluated_seed=True) == (0, False)
    assert _validate_pipeline_summary(root, _load_variant_specs_from_disk(root)) is not None


def test_baseline_seed_is_not_promoted_into_a_new_comparison(tmp_path):
    root, specs = comparison(tmp_path)
    baseline = specs[1]
    program = baseline.variant_dir / "task_bundle/program_1"
    program.rename(program.with_name("program_0"))
    seed = program.with_name("program_0")
    manifest = json.loads((seed / "evaluation_manifest.json").read_text()); manifest["program_id"] = "program_0"
    write_json(seed / "evaluation_manifest.json", manifest)
    assert _inspect_program_results(seed.parent, baseline.variant_id) == (0, False)
    assert _validate_pipeline_summary(root, _load_variant_specs_from_disk(root)) is not None


@pytest.mark.parametrize("identity", ["idea_id", "attempt_id", "overall_pass", "correct"])
def test_control_seed_keeps_canonical_identity_and_correctness(tmp_path, identity):
    root, specs = comparison(tmp_path)
    program = specs[0].variant_dir / "task_bundle/program_0"
    if identity == "correct":
        write_json(program / "correct.json", {"correct": False})
    else:
        path = program / "rubric_status.json"
        data = json.loads(path.read_text()); data[identity] = False if identity == "overall_pass" else "unbound"
        write_json(path, data)
    assert _validate_pipeline_summary(root, _load_variant_specs_from_disk(root)) is not None


def test_duplicate_materialized_ids_are_rejected(tmp_path):
    root, specs = comparison(tmp_path)
    duplicated = [*specs, replace(specs[0])]
    assert "duplicate" in _validate_pipeline_summary(root, duplicated)
