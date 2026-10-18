"""Initial methodology declarations remain host-bound, scoped, and inspectable."""
from __future__ import annotations

import copy
import json

import pytest

from src.research.evidence import merge_assessments, verify_methodology_for_paper
from src.research.methodology import MARKER_PATH, load_methodology
from src.research.paper_readiness import assess_readiness
from src.research.workspace import prepare
from tests.research.test_unified_research import make_task


@pytest.fixture
def methodology_study(tmp_path):
    task = make_task(tmp_path / "task")
    agenda = {
        "schema_version": 1, "research_question": "Does adaptive step size improve convergence?",
        "scope": "The declared development optimization tasks and fixed update budget.",
        "study_design": {
            "observation_unit": "An optimization run", "split_unit": "Task instance",
            "intervention": "Step-size rule", "fixed_factors": ["Initialization", "Update count"],
            "matched_budget": "Same update count", "outcomes": ["Final objective"],
            "falsifier": "No effect under the matched control",
            "minimal_test": "One paired development task", "stop_or_pivot": "Stop on a null paired effect",
        },
        "alternatives": [{
            "explanation": "Extra compute explains the change",
            "discriminating_experiment": {
                "intervention": "Replace only the step-size rule", "controls": "Match update count",
                "outcome_if_claim": "The paired objective improves",
                "outcome_if_alternative": "The matched paired effect disappears",
                "decision_rule": "Inspect the paired effect before scaling",
            },
        }],
        "evidence_plan": {"strong_baselines": ["Tuned fixed step size"],
                          "robustness_checks": ["Declared development seeds"],
                          "independent_confirmation": ["Sealed final task instances"]},
    }
    (task / "research_agenda.json").write_text(json.dumps(agenda))
    workspace = tmp_path / "managed"
    manifest = prepare(task, workspace, offline=True, overrides=[])
    (workspace / "harness_state.json").write_text(json.dumps({
        "schema_version": 1, "input_task_sha256": manifest["input_task_sha256"],
    }))
    investigation = workspace / "run/investigation"
    investigation.mkdir(parents=True)
    data = {"results_dir": str(investigation), "all_programs": [],
            "final_evaluation": {"outcome": "not_qualified"}}
    return workspace, investigation, data, agenda


def test_paper_projection_rebuilds_frozen_initial_design_and_never_certifies_selected_method(methodology_study):
    workspace, investigation, data, agenda = methodology_study
    merge_assessments(data, investigation)
    method = data["research_methodology"]
    marker = load_methodology(workspace / "input_task")
    assert method["status"] == "declared"
    assert method["verification_status"] == "host_verified"
    assert method["assessment_scope"] == "frozen_initial_agenda"
    assert method["study_design"] == agenda["study_design"]
    assert method["selected_candidate_design_assessed"] is False
    assert method["scientific_validity_certified"] is False
    assert method["rules_sha256"] == marker["rules_sha256"]
    assert all(rule in method["review_guidance"] for rule in marker["stage_rules"]["review"])
    assert data["final_evaluation"] == {"outcome": "not_qualified"}
    assert verify_methodology_for_paper(data) == method
    assert verify_methodology_for_paper(data, investigation_results_dir=workspace / "run") == method
    assert all(len(method["provenance"][key]) == 64 for key in (
        "input_task_sha256", "research_manifest_sha256", "methodology_marker_sha256",
        "agenda_sha256", "paper_context_sha256",
    ))
    from src.research.config import load_config
    ready = assess_readiness(workspace, load_config(workspace / "input_task"), {})
    assert ready["checks"]["methodology"]["status"] == "declared"
    assert ready["checks"]["methodology"]["selected_candidate_design_assessed"] is False
    assert ready["ready_for_human_review"] is False


@pytest.mark.parametrize("change", ["projection", "agenda", "marker_missing", "paper_context", "foreign_results"])
def test_stale_methodology_projection_fails_closed_and_cannot_be_silently_repaired(methodology_study, change):
    workspace, investigation, data, _agenda = methodology_study
    data["research_methodology"] = verify_methodology_for_paper(data, investigation_results_dir=investigation)
    if change == "projection":
        data["research_methodology"]["scientific_validity_certified"] = True
    elif change == "agenda":
        path = workspace / "input_task/research_agenda.json"
        altered = json.loads(path.read_text())
        altered["study_design"]["matched_budget"] = "An altered budget after freezing"
        path.write_text(json.dumps(altered))
    elif change == "marker_missing":
        (workspace / "input_task" / MARKER_PATH).unlink()
    elif change == "paper_context":
        data["all_programs"] = [{"program_dir": str(investigation / "changed_candidate")}]
    else:
        data["results_dir"] = str(workspace.parent / "foreign/run")
    before = copy.deepcopy(data)
    method = verify_methodology_for_paper(data, investigation_results_dir=investigation)
    assert method["status"] == "needs_evidence"
    assert method["verification_status"] == "invalid_or_missing"
    assert method["scientific_validity_certified"] is False
    assert method["study_design"] == {}
    assert data == before
    data["research_methodology"] = method
    assert verify_methodology_for_paper(data, investigation_results_dir=investigation) == method


def test_standalone_metadata_never_establishes_methodology_verification(tmp_path):
    assert verify_methodology_for_paper({"results_dir": str(tmp_path)}) is None
    forged = {"results_dir": str(tmp_path), "research_methodology": {
        "status": "declared", "scientific_validity_certified": True,
    }}
    result = verify_methodology_for_paper(forged)
    assert result["status"] == "needs_evidence"
    assert result["scientific_validity_certified"] is False


def test_removing_marker_and_saved_projection_cannot_turn_enabled_study_into_legacy(methodology_study):
    workspace, investigation, data, _agenda = methodology_study
    assert "research_methodology" not in data
    (workspace / "input_task" / MARKER_PATH).unlink()
    result = verify_methodology_for_paper(data, investigation_results_dir=investigation)
    assert result["status"] == "needs_evidence"
    assert result["verification_status"] == "invalid_or_missing"
    assert result["scientific_validity_certified"] is False


def test_readiness_rejects_structurally_complete_but_changed_frozen_agenda(methodology_study):
    from src.research.config import load_config

    workspace, _investigation, _data, _agenda = methodology_study
    cfg = load_config(workspace / "input_task")
    path = workspace / "input_task/research_agenda.json"
    altered = json.loads(path.read_text())
    altered["study_design"]["matched_budget"] = "A different complete declaration"
    path.write_text(json.dumps(altered))
    report = assess_readiness(workspace, cfg, {})
    assert report["checks"]["methodology"]["status"] == "needs_evidence"
    assert report["checks"]["methodology"]["verification_status"] == "invalid_or_missing"


def test_disabled_managed_task_keeps_methodology_projection_absent(tmp_path):
    from src.research.config import load_config, write_config

    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.methodology.enabled = False
    write_config(task / "research.yaml", cfg)
    workspace = tmp_path / "managed"
    manifest = prepare(task, workspace, offline=True, overrides=[])
    (workspace / "harness_state.json").write_text(json.dumps({
        "schema_version": 1, "input_task_sha256": manifest["input_task_sha256"],
    }))
    investigation = workspace / "run/investigation"
    investigation.mkdir(parents=True)
    assert verify_methodology_for_paper({"results_dir": str(investigation)},
                                        investigation_results_dir=investigation) is None
