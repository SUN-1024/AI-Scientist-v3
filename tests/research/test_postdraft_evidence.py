"""Postdraft evidence retains its full derivation and a separate design scope."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.research import postdraft_plan as post
from src.research.campaign_plan import CAMPAIGN_PLAN_PATH, HOST_CONTEXT_PATH, load_campaign_plan
from src.research.evidence import render_context, verify_methodology_for_paper, verify_postdraft_execution_agenda
from src.research.harness_state import sha256
from src.research.paper_readiness import write_readiness
from src.research.portfolio_evidence import _planned_evidence, verify_portfolio_for_paper
from src.research.research_agenda import load_agenda
from src.research.workspace import resume
from tests.research import test_campaign_plan_evidence as planned_fixtures
from tests.research import test_novelty_gate as novelty_fixtures
from tests.research.test_portfolio_paper_evidence import read, write
from tests.research.test_postdraft_plan import make_postdraft

study = novelty_fixtures.study
planned_study = planned_fixtures.planned_study


@pytest.fixture
def postdraft_study(study, tmp_path, monkeypatch):
    source, literature, initial, review = study
    initial = deepcopy(initial)
    initial["study_design"] = {
        "observation_unit": "Optimization run", "split_unit": "Task instance",
        "intervention": "Step-size strategy", "fixed_factors": ["Data", "Update count"],
        "matched_budget": "Same number of updates", "outcomes": ["Final objective"],
        "falsifier": "No paired improvement", "minimal_test": "One paired task",
        "stop_or_pivot": "Stop if the matched effect disappears",
    }
    workspace, paths, cfg, pipeline, calls = make_postdraft(
        (source, literature, initial, review), tmp_path, monkeypatch,
    )
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    plan = load_campaign_plan(workspace / post.PLAN_TASK, required=True)
    context = read(workspace / post.PLAN_TASK / HOST_CONTEXT_PATH)
    # The plan projection reports unused allocations even before trials exist.
    # Full receipt-ledger verification is exercised by the portfolio fixtures.
    state = {
        "policy": context["portfolio_policy"], "campaign_plan": plan, "plan_sha256": plan["plan_sha256"],
        "failures": [], "leases": {}, "frozen_candidate": {"root_idea_id": "baseline"},
        "baseline": {"score": .6},
        "ideas": {root["root_idea_id"]: {"promoted": False, "best": None} for root in plan["roots"]},
    }
    handoff = {
        "campaign_plan": deepcopy(plan), "plan_sha256": plan["plan_sha256"],
        "baseline_evidence_role": "bootstrap", "baseline_tuned_comparator": False,
        "planned_root_execution": [{
            "root_idea_id": root["root_idea_id"], "card_sha256": root["card_sha256"], "role": root["role"],
            "promoted": False, "own_best": None,
            "actual_evaluations": {"screening": 0, "tuning": 0, "replication": 0},
        } for root in plan["roots"]],
    }
    root = workspace / "run/investigation"
    root.mkdir(parents=True)
    return SimpleNamespace(workspace=workspace, paths=paths, cfg=cfg, pipeline=pipeline, calls=calls,
        plan=plan, state=state, handoff=handoff, initial=initial, root=root,
        data={"results_dir": str(root), "all_programs": []})


def test_postdraft_plan_artifacts_are_resolved_from_the_verified_snapshot(postdraft_study):
    study = postdraft_study
    projection = _planned_evidence(study.workspace, study.paths.task_dir, study.state, [], study.handoff)
    assert projection["verification_status"] == "host_verified"
    assert projection["assessment_scope"] == "postdraft_execution_agenda"
    assert projection["actual_total_evaluations"] == 0
    assert projection["unexecuted_total_allocation"] == projection["planned_total_evaluations"]
    for artifact in projection["source_artifacts"].values():
        assert artifact["path"].startswith("campaign/postdraft/planned_task/")
        assert sha256(study.workspace / artifact["path"]) == artifact["sha256"]
    provenance = projection["postdraft_campaign"]["provenance"]
    assert provenance["drafted_task"]["path"] == post.DRAFT_TASK.as_posix()
    assert provenance["planned_task"]["path"] == post.PLAN_TASK.as_posix()
    assert provenance["publication"]["snapshot_sha256"] == provenance["planned_task"]["sha256"]
    assert "novelty/initial/novelty_report.json" in provenance["initial_artifacts"]
    assert "novelty/postdraft/agenda_draft/receipt.json" in provenance["review_artifacts"]
    assert "novelty/postdraft/reviewer_1/receipt.json" in provenance["review_artifacts"]
    assert "execution_plan" not in load_agenda(study.workspace / "input_task")


def test_initial_and_postdraft_methodology_are_distinct_and_neither_certifies_selected_design(postdraft_study):
    study = postdraft_study
    projected = verify_methodology_for_paper(study.data, investigation_results_dir=study.root)
    execution = projected["postdraft_execution_agenda"]
    assert projected["assessment_scope"] == "frozen_initial_agenda"
    assert projected["declared_agenda"]["research_question"] == study.initial["research_question"]
    assert execution["assessment_scope"] == "postdraft_execution_agenda"
    assert execution["declared_agenda"]["research_question"] != study.initial["research_question"]
    assert execution["rules_sha256"] == projected["rules_sha256"]
    for item in (projected, execution):
        assert item["status"] == "declared", item
        assert item["verification_status"] == "host_verified"
        assert item["selected_candidate_design_assessed"] is False
        assert item["scientific_validity_certified"] is False
    study.data["research_methodology"] = projected
    assert verify_methodology_for_paper(study.data) == projected


def test_resume_preserves_postdraft_projection_and_all_frozen_artifacts(postdraft_study):
    study = postdraft_study
    before = {path: sha256(path) for root in ("input_task", "novelty/initial", "novelty/postdraft", "campaign/postdraft")
              for path in (study.workspace / root).rglob("*") if path.is_file()}
    projected = verify_postdraft_execution_agenda(study.workspace)
    resume(study.workspace)
    assert post.ensure_postdraft_campaign_plan(study.workspace, study.paths, study.cfg, study.pipeline, recovery=True)
    assert verify_postdraft_execution_agenda(study.workspace) == projected
    assert before == {path: sha256(path) for path in before}
    assert len(study.calls) == 4


def test_readiness_keeps_execution_coverage_and_provenance_separate(postdraft_study):
    study = postdraft_study
    report = write_readiness(study.workspace, study.cfg, {})
    initial, execution = report["checks"]["methodology"], report["checks"]["postdraft_execution_agenda"]
    assert initial["assessment_scope"] == "frozen_initial_agenda"
    assert execution["assessment_scope"] == "postdraft_execution_agenda"
    assert initial["status"] == execution["status"] == "declared"
    assert initial["selected_candidate_design_assessed"] is execution["selected_candidate_design_assessed"] is False
    saved = read(study.workspace / "research_next_actions.json")["postdraft_execution_agenda"]
    assert saved["provenance"] == execution["provenance"]
    assert saved["scientific_validity_certified"] is False
    assert report["ready_for_human_review"] is False


@pytest.mark.parametrize("target", [
    "input_task/task.md", "input_task/research_agenda.json", "novelty/initial/novelty_report.json",
    "campaign/postdraft/drafted_task/program_0/src/program.py",
    "campaign/postdraft/planned_task/research_agenda.json",
    "novelty/postdraft/agenda_draft/receipt.json", "novelty/postdraft/reviewer_1/receipt.json",
    "runtime_plan", "publication",
])
def test_changed_derivation_or_publication_cannot_be_reported_as_verified(postdraft_study, target):
    study = postdraft_study
    if target == "publication":
        path = study.workspace / post.JOURNAL
        changed = read(path)
        changed["runtime_task_sha256"] = "0" * 64
        write(path, changed)
    elif target == "runtime_plan":
        path = study.paths.task_dir / CAMPAIGN_PLAN_PATH
        changed = read(path)
        changed["roots"][0]["complete_implementation"] = "Replace the reviewed control."
        write(path, changed)
    else:
        path = study.workspace / target
        path.write_bytes(path.read_bytes() + b" ")
    execution = verify_postdraft_execution_agenda(study.workspace)
    assert execution["verification_status"] == "invalid_or_missing", execution
    assert execution["selected_candidate_design_assessed"] is False
    with pytest.raises((ValueError, OSError, RuntimeError)):
        _planned_evidence(study.workspace, study.paths.task_dir, study.state, [], study.handoff)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        render_context(study.paths.task_dir)
    assert len(study.calls) == 4


@pytest.mark.parametrize("target", ["ledger_plan", "ledger_hash", "handoff_plan", "handoff_hash"])
def test_postdraft_plan_must_match_ledger_and_handoff(postdraft_study, target):
    study = postdraft_study
    changed = study.state if target.startswith("ledger") else study.handoff
    if target.endswith("_hash"):
        changed["plan_sha256"] = "0" * 64
    else:
        changed["campaign_plan"]["roots"][0]["complete_implementation"] = "A replacement declaration."
    with pytest.raises(ValueError, match="same campaign plan|changed the approved plan"):
        _planned_evidence(study.workspace, study.paths.task_dir, study.state, [], study.handoff)


def test_saved_postdraft_design_claim_is_rechecked(postdraft_study):
    study = postdraft_study
    study.data["research_methodology"] = verify_methodology_for_paper(study.data, investigation_results_dir=study.root)
    study.data["research_methodology"]["postdraft_execution_agenda"]["selected_candidate_design_assessed"] = True
    result = verify_methodology_for_paper(study.data)
    assert result["verification_status"] == "invalid_or_missing"
    assert result["selected_candidate_design_assessed"] is False


def test_marked_pending_workspace_reports_execution_evidence_gap(study, tmp_path, monkeypatch):
    workspace, _paths, _cfg, _pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    result = verify_postdraft_execution_agenda(workspace)
    assert result["assessment_scope"] == "postdraft_execution_agenda"
    assert result["verification_status"] == "invalid_or_missing"
    assert "awaits" in result["detail"]
    assert not calls


def test_legacy_projection_still_uses_input_task_without_postdraft_fields(planned_study):
    study = planned_study
    projected = verify_portfolio_for_paper(study.data, investigation_results_dir=study.root)
    assert projected["verification_status"] == "host_verified", projected
    assert "postdraft_campaign" not in projected["campaign_plan"]
    assert all(artifact["path"].startswith("input_task/")
               for artifact in projected["campaign_plan"]["source_artifacts"].values())
    method = verify_methodology_for_paper(study.data, investigation_results_dir=study.root)
    assert "postdraft_execution_agenda" not in method
    assert verify_postdraft_execution_agenda(study.workspace) is None
