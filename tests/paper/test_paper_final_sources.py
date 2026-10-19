"""The final program stays distinct from development winners and remains inspectable."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from src.final_evaluation.runner import _tree_digest
from src.orchestrator.paper_final_sources import _final_tree_digest
from src.orchestrator.paper_study_sources import (
    _read_tree, stage_study_sources, verify_study_sources,
)
from src.pipeline.safe_fs import open_directory

ROOT = Path(__file__).resolve().parents[2]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _seal(final, program, *, correct=True):
    metrics = {"heldout_score": 0.45}
    _json(final / "results/metrics.json", metrics)
    _json(final / "results/correct.json", {"correct": correct})
    invocation = {
        "schema_version": 1, "status": "started", "started_at": "2026-10-19T12:00:00Z",
        "program_tree_sha256": _tree_digest(program),
        "task_tree_sha256": "sha256:" + "a" * 64,
        "evaluator_sha256": "sha256:" + "b" * 64,
    }
    _json(final / "invocation.json", invocation)
    manifest = {
        **invocation, "schema_version": 2, "status": "completed",
        "completed_at": "2026-10-19T12:01:00Z", "correct": correct,
        "outcome": "qualified" if correct else "not_qualified",
        "metrics": metrics, "declared_final_metrics": list(metrics),
        "results_tree_sha256": _tree_digest(final / "results"),
    }
    _json(final / "final_manifest.json", manifest)
    return {**manifest, "manifest_path": str(final / "final_manifest.json")}


@pytest.fixture
def case(tmp_path):
    investigation = tmp_path / "investigation"
    program = investigation / "trial/program_1"
    (program / "src").mkdir(parents=True)
    (program / "src/program.py").write_text("pooling = 'lme'\n")
    (program / "src/metrics.json").write_text('{"implementation_resource": true}')
    (program / "program_spec.toml").write_text('entrypoint = "src/program.py"\n')
    _json(program / "metrics.json", {"combined_score": 0.51})
    (program / "analysis").mkdir()
    (program / "analysis/run.csv").write_text("score\n0.51\n")
    _json(investigation / "best_program_manifest.json", {
        "program_dir": str(program), "entrypoint_path": "src/program.py",
    })
    final = tmp_path / "custom-final"
    data = {"final_evaluation": _seal(final, program), "best_correct": {
        "variant_id": "baseline_max", "program_dir": "unrelated_development_winner",
        "combined_score": 0.55,
    }}
    paper = tmp_path / "paper"
    paper.mkdir()
    return paper, data, investigation, program, final


def _stage(case):
    paper, data, investigation, _, _ = case
    stage_study_sources(data, paper, investigation)
    _json(paper / "paper_data.json", data)


def _verify(case):
    paper, _, investigation, _, final = case
    verify_study_sources(paper, investigation, final_evaluation_dir=final)


def test_digest_matches_final_runner_for_bounded_bytes_and_ignores_modes(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "données").mkdir()
    path = tmp_path / "données/binary.bin"
    path.write_bytes(b"\0file\0dir\0\xff")
    (tmp_path / "a.py").write_text("example = True\n")
    with open_directory(tmp_path, role="test sources") as fd:
        tree = _read_tree(fd)
    assert _final_tree_digest(tree) == _tree_digest(tmp_path)
    path.chmod(0o400)
    with open_directory(tmp_path, role="test sources") as fd:
        assert _final_tree_digest(_read_tree(fd)) == _tree_digest(tmp_path)


@pytest.mark.parametrize("correct", [True, False])
def test_final_source_uses_frozen_program_independently_of_best_development_row(case, correct):
    paper, data, _, program, final = case
    data["final_evaluation"] = _seal(final, program, correct=correct)
    original = copy.deepcopy(data)
    _stage(case)
    context = data["final_program"]
    assert context["program_dir"] == str(program)
    assert context["program_tree_sha256"] == data["final_evaluation"]["program_tree_sha256"]
    assert context["evaluation_scope"] == "sealed_final"
    from src.orchestrator.paper_evidence_audit import _paper_data_summary as audit_summary
    from src.orchestrator.paper_review import _paper_data_summary as review_summary

    for summarize in (audit_summary, review_summary):
        assert summarize(data)["final_program"] == context
    source = paper / context["source_evidence"]["program"]
    assert (source / context["entrypoint_path"]).read_text() == "pooling = 'lme'\n"
    assert (source / "src/metrics.json").is_file()
    assert not (source / "metrics.json").exists()
    assert not (source / "analysis").exists()
    assert {key: data[key] for key in original} == original
    before = (source / "src/program.py").stat()
    _stage(case)
    assert (source / "src/program.py").stat() == before
    _verify(case)


@pytest.mark.parametrize("change", [
    "code", "missing", "extra", "context", "entrypoint", "path", "both_contexts",
])
def test_audit_rejects_changes_without_recreating_missing_code(case, change):
    paper, data, _, _, _ = case
    _stage(case)
    source = paper / data["final_program"]["source_evidence"]["program"] / "src/program.py"
    if change == "code":
        source.write_text("invented_method = True\n")
    elif change == "missing":
        source.unlink()
    elif change == "extra":
        (source.parent / "invented.py").write_text("invented_method = True\n")
    elif change == "context":
        del data["final_program"]
    elif change == "entrypoint":
        data["final_program"]["entrypoint_path"] = "invented.py"
    elif change == "both_contexts":
        del data["final_program"]
        del data["final_evaluation"]
    else:
        data["final_program"]["source_evidence"]["program"] = "other"
    _json(paper / "paper_data.json", data)
    with pytest.raises(ValueError):
        _verify(case)
    if change == "missing":
        assert not source.exists()


@pytest.mark.parametrize("relative", [
    "source_evidence", "source_evidence/final_program", "source_evidence/final_program/program",
])
def test_symlinked_source_destination_cannot_write_outside_paper(case, relative):
    paper, _, _, _, _ = case
    outside = paper.parent / "outside"
    outside.mkdir()
    target = paper / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        _stage(case)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("change", ["program", "paper_digest", "results", "invocation"])
def test_changed_frozen_inputs_or_final_records_are_rejected_before_publication(case, change):
    paper, data, _, program, final = case
    if change == "program":
        (program / "src/program.py").write_text("changed = True\n")
    elif change == "paper_digest":
        data["final_evaluation"]["program_tree_sha256"] = "sha256:" + "c" * 64
    elif change == "results":
        _json(final / "results/metrics.json", {"heldout_score": 0.99})
    else:
        _json(final / "invocation.json", {"schema_version": 1, "status": "started"})
    with pytest.raises(ValueError):
        _stage(case)
    assert not (paper / "source_evidence/final_program").exists()


def test_configured_final_location_prevents_author_substitution(case):
    paper, data, _, program, _ = case
    _stage(case)
    forged = paper / "other-final"
    data["final_evaluation"] = _seal(forged, program)
    _json(paper / "paper_data.json", data)
    with pytest.raises(ValueError, match="configured final evaluation"):
        _verify(case)


def test_sources_survive_readonly_audit_and_revision_copy(case):
    from src.investigation.paper_revision_workspace import (
        seed_revision_source_evidence, snapshot_previous_paper,
    )
    from src.orchestrator.paper_evidence_audit import prepare_readonly_workspace

    paper, data, investigation, _, final = case
    _stage(case)
    readonly = paper.parent / "readonly"
    prepare_readonly_workspace(paper, readonly)
    revision = paper.parent / "revision"
    revision.mkdir()
    snapshot = revision / "previous_paper"
    snapshot_previous_paper(paper, snapshot)
    seed_revision_source_evidence(snapshot, revision)
    _json(revision / "paper_data.json", data)
    for root in (readonly, snapshot, revision):
        verify_study_sources(root, investigation, final_evaluation_dir=final)


def test_source_limits_and_nonexclusive_publication(case, monkeypatch):
    from src.orchestrator import paper_study_sources as sources
    from src.pipeline import safe_fs

    paper, _, _, _, _ = case
    with monkeypatch.context() as limited:
        limited.setattr(sources, "MAX_SOURCE_BYTES", 1)
        with pytest.raises(ValueError, match="limit"):
            _stage(case)
    assert not (paper / "source_evidence/final_program").exists()
    monkeypatch.setattr(safe_fs, "_renameat2_no_replace", lambda *a, **k: False)
    monkeypatch.setattr(safe_fs, "_renameatx_np_no_replace", lambda *a, **k: False)
    _stage(case)
    _verify(case)


@pytest.mark.parametrize("script", ["paper.sh", "paper_edit.sh"])
def test_production_audit_hook_verifies_final_source_before_readonly_copy(case, monkeypatch, script):
    paper, data, investigation, _, final = case
    _stage(case)
    source = paper / data["final_program"]["source_evidence"]["program"] / "src/program.py"
    source.write_text("invented_method = True\n")
    text = (ROOT / "src/scripts" / script).read_text()
    hook = text.index("\nprepare_readonly_paper_audit_workspace()")
    start = text.index("<<'PY'\n", hook) + len("<<'PY'\n")
    end = text.index("\nPY\n", start)
    assert end < text.index("prepare-audit-workspace", hook)
    monkeypatch.setattr(sys, "argv", ["-", str(ROOT), str(paper), str(investigation),
                                    str(final), "discover", "", "qualified_only"])
    with pytest.raises(ValueError, match="differ"):
        exec(compile(text[start:end], script, "exec"), {})


def test_missing_final_evaluation_does_not_invent_a_final_program(case):
    paper, data, investigation, _, _ = case
    del data["final_evaluation"]
    _stage(case)
    assert "final_program" not in data
    assert not (paper / "source_evidence").exists()
    verify_study_sources(paper, investigation)  # Explicitly omitted final evidence stays omitted.
    data["final_program"] = {"invented": True}
    with pytest.raises(ValueError, match="without final evaluation"):
        stage_study_sources(data, paper, investigation)
