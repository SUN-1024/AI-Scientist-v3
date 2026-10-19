"""Resume preserves completed negative reviews without weakening their bindings."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.research import harness
from src.research.config import load_config
from src.research.novelty_review import load_completed_novelty_report, review_novelty
from tests.research import test_novelty_gate as fixtures
from tests.research import test_selected_novelty_paper as selected_fixtures

study = fixtures.study
selected_study = selected_fixtures.selected_study


def hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


def assessment_for(status, review, packet, agenda):
    result = fixtures.candidate_review(review, packet)
    if status == "needs_evidence":
        result["candidate_assessment"]["checks"]["implementation_matches_claim"] = "uncertain"
    elif status == "overlap":
        result["verdict"] = "overlap"
        result["checks"]["difference_meaningful"] = "fail"
        result["overlap_assessment"] = {
            "core_problem_same": True, "key_method_same": True,
            "main_contribution_same": True, "remaining_difference_meaningful": False,
        }
    elif status == "no_novelty_claim":
        agenda["intent"] = "replication"
    return result


@pytest.fixture
def saved_review(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study

    def create(status="needs_evidence", *, missing_candidate=False):
        packet = fixtures.candidate_packet()
        actual = assessment_for(status, review, packet, agenda)
        if missing_candidate:
            actual.pop("candidate_assessment")
        calls = fixtures.install_provider(monkeypatch, [actual])
        cycle = tmp_path / "selected_review"
        report = review_novelty(task, agenda, cfg, cycle, candidate_packet=packet)
        assert report["status"] == status
        assert len(calls) == 2
        return task, cfg, cycle, packet, calls

    return create


@pytest.mark.parametrize("status", ["supported", "needs_evidence", "overlap", "no_novelty_claim"])
def test_complete_review_is_reused_without_changing_verdict_or_artifacts(saved_review, status):
    task, cfg, cycle, packet, calls = saved_review(status)
    before = hashes(cycle)
    for _ in range(2):
        report = load_completed_novelty_report(task, cycle, cfg, reviewer_count=2, candidate_packet=packet)
        assert report is not None and report["status"] == status
        assert len(report["model_judgments"]) == 2
        assert report["acceptance_guaranteed"] is False
    assert hashes(cycle) == before
    assert len(calls) == 2


@pytest.mark.parametrize("change", [
    "source", "candidate", "agenda", "response", "receipt", "missing_reviewer", "duplicate_session",
])
def test_stale_or_incomplete_reviews_cannot_be_reused(saved_review, change):
    task, cfg, cycle, packet, calls = saved_review()
    if change == "source":
        path = task / "research_context/text/prior.txt"
        path.write_text(path.read_text() + " changed source")
    elif change == "candidate":
        packet["evaluation"]["metrics"]["combined_score"] = 0.91
    elif change == "agenda":
        path = cycle / "research_agenda.json"
        agenda = json.loads(path.read_text())
        agenda["scope"] += " Changed scope."
        path.write_text(json.dumps(agenda))
    elif change == "response":
        path = cycle / "reviewer_1/01_novelty_review.json"
        path.write_text(path.read_text() + " ")
    elif change == "missing_reviewer":
        (cycle / "reviewer_2/receipt.json").unlink()
    else:
        path = cycle / "reviewer_2/receipt.json"
        receipt = json.loads(path.read_text())
        if change == "duplicate_session":
            receipt["session_id"] = json.loads((cycle / "reviewer_1/receipt.json").read_text())["session_id"]
        else:
            receipt["status"] = "failed"
        path.write_text(json.dumps(receipt))
    assert load_completed_novelty_report(task, cycle, cfg, reviewer_count=2, candidate_packet=packet) is None
    assert len(calls) == 2


def test_reviewer_configuration_and_required_count_still_apply(saved_review):
    task, cfg, cycle, packet, calls = saved_review()
    assert load_completed_novelty_report(task, cycle, cfg, reviewer_count=3, candidate_packet=packet) is None
    changed = copy.deepcopy(cfg)
    changed.model_search.model = "different-reviewer-model"
    with pytest.raises(ValueError, match="configuration.*changed"):
        load_completed_novelty_report(task, cycle, changed, reviewer_count=2, candidate_packet=packet)
    assert len(calls) == 2


def test_missing_candidate_assessment_is_not_a_completed_candidate_review(saved_review):
    task, cfg, cycle, packet, calls = saved_review(missing_candidate=True)
    assert load_completed_novelty_report(task, cycle, cfg, reviewer_count=2, candidate_packet=packet) is None
    assert len(calls) == 2


def test_saved_positive_label_cannot_replace_actual_negative_review(saved_review):
    task, cfg, cycle, packet, calls = saved_review("overlap")
    path = cycle / "novelty_report.json"
    saved = json.loads(path.read_text())
    saved["status"] = "supported"
    path.write_text(json.dumps(saved))
    report = load_completed_novelty_report(task, cycle, cfg, reviewer_count=2, candidate_packet=packet)
    assert report is not None and report["status"] == "overlap"
    assert len(calls) == 2


def pipeline_config():
    return SimpleNamespace(
        agent=SimpleNamespace(name="codex", model="fixture-model", profile="fixture", reasoning_effort="low"),
        review=SimpleNamespace(backend="model_search", timeout_seconds=60),
    )


@pytest.mark.parametrize("status", ["needs_evidence", "overlap", "no_novelty_claim"])
def test_native_selected_hook_keeps_negative_review_and_original_science(selected_study, study, monkeypatch, status):
    workspace, _program, _cycle, packet, _paper = selected_study
    _task, _literature, agenda, review = study
    cfg = load_config(workspace / "input_task")
    actual = assessment_for(status, review, packet, agenda)
    calls = fixtures.install_provider(monkeypatch, [actual])
    cycle = workspace / "novelty/selected_negative"
    report = review_novelty(workspace / "input_task", agenda, cfg.literature, cycle, candidate_packet=packet)
    assert report["status"] == status
    pointer_path = workspace / "novelty/latest.json"
    pointer = json.loads(pointer_path.read_text())
    pointer.update(cycle="novelty/selected_negative", status=status)
    pointer_path.write_text(json.dumps(pointer))
    before = hashes(workspace)
    new_cycle = Mock(side_effect=AssertionError("Unchanged completed review must not be rerolled"))
    monkeypatch.setattr(harness, "_novelty_cycle", new_cycle)
    harness._selected_novelty(workspace, cfg, pipeline_config(), offline=False)
    harness._selected_novelty(workspace, cfg, pipeline_config(), offline=False)
    new_cycle.assert_not_called()
    assert len(calls) == 2 and hashes(workspace) == before
    assert json.loads(pointer_path.read_text())["status"] == status


def test_native_selected_hook_does_not_reuse_an_insufficient_review_count(selected_study, monkeypatch):
    workspace, _program, cycle, _packet, _paper = selected_study
    cfg = load_config(workspace / "input_task")
    cfg.harness.novelty_reviewers = 3
    before = hashes(cycle)
    new_cycle = Mock(return_value={"status": "needs_evidence"})
    monkeypatch.setattr(harness, "_novelty_cycle", new_cycle)
    harness._selected_novelty(workspace, cfg, pipeline_config(), offline=False)
    new_cycle.assert_called_once()
    assert hashes(cycle) == before
    assert json.loads((workspace / "novelty/latest.json").read_text())["status"] == "needs_evidence"
