"""The paper receives the selected implementation review, rechecked on the host."""
from __future__ import annotations

import copy
import json
import shutil
import sys

import pytest

from src.investigation.native_harness import run_native_evaluation
from src.investigation.paper_prompt_context import paper_workspace_context_replacements
from src.orchestrator.paper_data import FinalEvaluationMode, build_paper_data
from src.research.config import ResearchConfig, write_config
from src.research.evidence import merge_assessments, verify_selected_novelty_for_paper
from src.research.harness_evidence import build_candidate_packet
from src.research.novelty import content_hash, evaluate_novelty
from src.research.novelty_review import review_novelty
from src.research.workspace import prepare
from tests.program_testutils import make_program_dir
from tests.research import test_novelty_gate as fixtures

study = fixtures.study


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def selected_study(study, tmp_path, monkeypatch):
    task, literature_cfg, agenda, review = study
    write_config(task / "research.yaml", ResearchConfig(literature=literature_cfg))
    (task / "evaluate.py").write_text('''import argparse, json, pathlib, runpy
p=argparse.ArgumentParser()
p.add_argument('--program_path');p.add_argument('--results_dir')
a=p.parse_args();out=pathlib.Path(a.results_dir)
score=runpy.run_path(a.program_path)['SCORE']
(out/'metrics.json').write_text(json.dumps({'combined_score':score}))
(out/'correct.json').write_text(json.dumps({'correct':True}))
''')
    workspace = tmp_path / "managed"

    def initial_context(staged, _cfg):
        return {"research_agenda.json": json.dumps(agenda),
                "research_context/harness_novelty.json": json.dumps(evaluate_novelty(staged, agenda))}

    manifest = prepare(task, workspace, offline=True, overrides=[], after_literature=initial_context)
    write(workspace / "harness_state.json", {"schema_version": 1,
          "input_task_sha256": manifest["input_task_sha256"], "quality_target": "development"})
    shutil.copytree(workspace / "input_task", workspace / "run/task")
    investigation = workspace / "run/investigation"
    idea = investigation / "ideas/idea_1"
    program = idea / "program_1"
    make_program_dir(program, content="SCORE = 0.7\n")
    write(idea / "idea_bundle.json", {"idea_id": "idea_1", "research_basis": {
        "status": "needs_verification", "contribution": "A bounded development study",
        "alternative_explanation": "A changed budget could explain the observation",
        "discriminating_test": "Match the budget and compare paired seeds", "closest_work": [],
    }})
    exit_code = run_native_evaluation(program_dir=program, evaluator=workspace / "run/task/evaluate.py",
                                     python_bin=sys.executable, timeout_seconds=10, environment={},
                                     attempt_id="idea_1_attempt_1", idea_id="idea_1")
    assert exit_code == 0, (program / "harness_evaluate_stderr.log").read_text()
    write(investigation / "best_program_manifest.json", {
        "program_dir": str(program), "entrypoint_path": "src/program.py",
        "source_attempt_id": "idea_1_attempt_1", "combined_score": 0.7, "correct": True,
    })
    packet = build_candidate_packet(workspace)
    actual = fixtures.candidate_review(review, packet)
    actual["rationale"] = "Selected implementation feedback: development-only evidence remains provisional."
    actual["candidate_assessment"]["claim_code_evidence_map"][0]["finding"] = (
        "The selected source declares SCORE=0.7 and the host development receipt records 0.7; "
        "this does not establish independent final confirmation."
    )
    fixtures.install_provider(monkeypatch, [actual])
    cycle = workspace / "novelty/selected_fixture"
    assert review_novelty(workspace / "input_task", agenda, literature_cfg, cycle,
                          candidate_packet=packet)["status"] == "supported"
    write(workspace / "novelty/latest.json", {
        "schema_version": 1, "status": "supported", "cycle": "novelty/selected_fixture",
        "candidate_program": packet["selection"]["program_relative_path"],
        "candidate_source_sha256": packet["candidate_source_sha256"],
        "candidate_packet_sha256": content_hash(packet),
    })
    data = build_paper_data(investigation, final_evaluation_mode=FinalEvaluationMode.OMIT)
    assert data["selected_novelty_review"]["status"] == "supported", data["selected_novelty_review"]
    return workspace, program, cycle, packet, data


def stage_paper(tmp_path, workspace, data):
    paper = tmp_path / "temporary_paper_workspace"
    paper.mkdir()
    (paper / "task").symlink_to(workspace / "run/task", target_is_directory=True)
    write(paper / "paper_data.json", data)
    return paper


def test_selected_review_is_in_real_paper_data_and_prompt(selected_study, tmp_path):
    workspace, _program, _cycle, packet, data = selected_study
    selected = data["selected_novelty_review"]
    assert selected["verification_status"] == "host_verified"
    assert selected["assessment_scope"] == "selected_candidate_development"
    assert selected["candidate_packet_sha256"] == content_hash(packet)
    assert selected["candidate_source_sha256"] == packet["candidate_source_sha256"]
    provenance = selected["provenance"]
    assert provenance["managed_workspace"] == str(workspace)
    assert provenance["cycle"] == "novelty/selected_fixture"
    assert len(provenance["review_receipts"]) == 2
    assert all(len(row["sha256"]) == 64 for row in provenance["review_receipts"])
    assert all(len(provenance[key]) == 64 for key in (
        "input_task_sha256", "research_manifest_sha256", "pointer_sha256", "report_sha256",
        "replayed_report_sha256", "paper_context_sha256", "evaluation_receipt_sha256",
    ))
    assert verify_selected_novelty_for_paper(data) == selected
    assert selected["human_approval"] is False and selected["publication_acceptance"] is False
    paper = stage_paper(tmp_path, workspace, data)
    context = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    assert "HARNESS AGENDA REVIEW FEEDBACK" in context
    assert "SELECTED-METHOD NOVELTY REVIEW FEEDBACK" in context
    assert "Selected implementation feedback: development-only evidence remains provisional" in context
    assert "Do not tune on the sealed final holdout" in context
    assert "human approval, or publication acceptance" in context


def test_real_pipeline_envelope_and_investigation_paths_have_same_projection(selected_study):
    from src.pipeline.cli import load_pipeline_config
    from src.pipeline.paths import derive_paths

    workspace, _program, _cycle, _packet, data = selected_study
    cfg = load_pipeline_config([f"task.dir={workspace / 'input_task'}", f"paths.run_dir={workspace / 'run'}"])
    paths = derive_paths(cfg)
    assert paths.investigation_results_dir == workspace / "run"
    assert paths.investigation_dir == workspace / "run/investigation"
    fresh = copy.deepcopy(data)
    fresh.pop("selected_novelty_review")
    merge_assessments(fresh, paths.investigation_results_dir)
    assert fresh["selected_novelty_review"] == data["selected_novelty_review"]
    envelope = build_paper_data(paths.investigation_results_dir,
                               investigation_results_dir=paths.investigation_results_dir,
                               final_evaluation_mode=FinalEvaluationMode.OMIT)
    assert envelope["selected_novelty_review"]["status"] == "supported"
    assert verify_selected_novelty_for_paper(envelope, investigation_results_dir=paths.investigation_dir) == envelope["selected_novelty_review"]


@pytest.mark.parametrize("change", ["missing_pointer", "missing_report", "source", "metrics", "frozen_source",
                                     "report", "raw_response", "assessment", "paper_candidate", "paper_projection"])
def test_stale_or_missing_evidence_downgrades_and_failure_is_stable(selected_study, tmp_path, change):
    workspace, program, cycle, _packet, data = selected_study
    if change == "missing_pointer":
        (workspace / "novelty/latest.json").unlink()
    elif change == "missing_report":
        (cycle / "novelty_report.json").unlink()
    elif change == "source":
        (program / "src/program.py").write_text("SCORE = 9\n")
    elif change == "metrics":
        write(program / "metrics.json", {"combined_score": 9})
    elif change == "frozen_source":
        source = workspace / "input_task/research_context/text/prior.txt"
        source.write_text(source.read_text() + " changed source")
    elif change in {"report", "raw_response", "assessment"}:
        path = cycle / {"report": "novelty_report.json", "raw_response": "reviewer_1/01_novelty_review.json",
                        "assessment": "reviewer_1/assessment.json"}[change]
        path.write_text(path.read_text() + " ")
    elif change == "paper_candidate":
        data["best_correct"]["program_dir"] = str(workspace / "run/investigation/other/program_2")
    else:
        data["selected_novelty_review"]["model_judgments"][0]["assessment"]["rationale"] = "Forged approval"
    failed = verify_selected_novelty_for_paper(data)
    assert failed["status"] == "needs_evidence" and failed["gaps"]
    assert failed["verification_status"] == "invalid_or_missing"
    assert failed["human_approval"] is False
    data["selected_novelty_review"] = failed
    assert verify_selected_novelty_for_paper(data) == failed
    assert verify_selected_novelty_for_paper(data, investigation_results_dir=workspace / "run") == failed
    paper = stage_paper(tmp_path, workspace, data)
    # Frozen-input corruption is independent of the staged task copy.
    context = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    selected_context = context.split("SELECTED-METHOD NOVELTY REVIEW FEEDBACK:", 1)[1]
    assert '"status": "needs_evidence"' in selected_context
    assert '"status": "supported"' not in selected_context


@pytest.mark.parametrize("field,value", [
    ("cycle", "../selected_elsewhere"), ("cycle", "/tmp/selected_elsewhere"),
    ("cycle", "run/paper"), ("cycle", "novelty/../novelty/selected_fixture"),
    ("candidate_program", "run/investigation/another/program"),
    ("candidate_source_sha256", "a" * 64), ("candidate_packet_sha256", "b" * 64),
])
def test_wrong_pointer_is_never_projected(selected_study, field, value):
    workspace, _program, _cycle, _packet, data = selected_study
    pointer_path = workspace / "novelty/latest.json"
    pointer = read(pointer_path)
    pointer[field] = value
    write(pointer_path, pointer)
    result = verify_selected_novelty_for_paper(data)
    assert result["status"] == "needs_evidence" and result["model_judgments"] == []


@pytest.mark.parametrize("relative", ["novelty/latest.json", "novelty/selected_fixture", "novelty", "run/investigation"])
def test_symlinked_pointer_or_evidence_roots_are_rejected(selected_study, tmp_path, relative):
    workspace, _program, _cycle, _packet, data = selected_study
    source = workspace / relative
    destination = tmp_path / "outside_evidence"
    source.rename(destination)
    source.symlink_to(destination, target_is_directory=destination.is_dir())
    result = verify_selected_novelty_for_paper(data)
    assert result["status"] == "needs_evidence" and result["model_judgments"] == []


@pytest.mark.parametrize("declaration", [None, "supported", {"status": "supported"},
                                          {"status": "supported", "provenance": {"managed_workspace": "/tmp/fake"}}])
def test_standalone_forged_selected_field_is_explicitly_unverified(tmp_path, declaration):
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    write(standalone / "novelty/latest.json", {"status": "supported", "cycle": "selected_fixture"})
    data = {"results_dir": str(standalone), "selected_novelty_review": declaration}
    result = verify_selected_novelty_for_paper(data)
    assert result["status"] == "needs_evidence"
    assert result["verification_status"] == "invalid_or_missing"
    assert result["model_judgments"] == []
    write(standalone / "paper_data.json", data)
    context = paper_workspace_context_replacements(standalone)["RESEARCH_CONTEXT"]
    assert "SELECTED-METHOD NOVELTY REVIEW FEEDBACK" in context
    assert '"status": "needs_evidence"' in context


def test_nonharness_paper_behavior_is_unchanged(tmp_path):
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    write(standalone / "novelty/latest.json", {"status": "supported"})
    data = {"results_dir": str(standalone)}
    merge_assessments(data, standalone)
    assert "selected_novelty_review" not in data
    assert verify_selected_novelty_for_paper(data) is None
    write(standalone / "paper_data.json", data)
    context = paper_workspace_context_replacements(standalone)["RESEARCH_CONTEXT"]
    assert "SELECTED-METHOD NOVELTY REVIEW FEEDBACK" not in context


def test_managed_paper_missing_projection_needs_evidence(selected_study):
    _workspace, _program, _cycle, _packet, data = selected_study
    data.pop("selected_novelty_review")
    result = verify_selected_novelty_for_paper(data)
    assert result["status"] == "needs_evidence"
    assert "missing its selected novelty projection" in result["detail"]


def test_copied_projection_cannot_certify_an_unrelated_paper(selected_study, tmp_path):
    _workspace, _program, _cycle, _packet, data = selected_study
    data["results_dir"] = str(tmp_path / "unrelated")
    result = verify_selected_novelty_for_paper(data)
    assert result["status"] == "needs_evidence"
    assert "do not belong" in result["detail"]
