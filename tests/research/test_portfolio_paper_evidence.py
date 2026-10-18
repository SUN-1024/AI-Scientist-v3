"""Whole-idea paper claims stay bound to the complete host development ledger."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.investigation.adaptive_runtime import validated_development_outcome
from src.investigation.harness_incumbent import _program_source_records
from src.investigation.paper_prompt_context import paper_workspace_context_replacements
from src.investigation.portfolio_controller import PortfolioConfig, PortfolioController, STATE_FILE
from src.investigation.program_clone import clone_program_dir
from src.investigation.native_harness import _hash
from src.research.portfolio_evidence import FIELD, MAX_PREVIEW_TRIALS, _preview, verify_portfolio_for_paper
from src.research.workspace import prepare
from tests.program_testutils import make_program_dir
from tests.research.test_unified_research import make_task


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path: Path):
    return json.loads(path.read_text())


@pytest.fixture
def portfolio_study(tmp_path, request):
    settings = getattr(request, "param", {})
    task = make_task(tmp_path / "source")
    (task / "evaluate.py").write_text("# deterministic sealed-host fixture, no external execution\n")
    policy = PortfolioConfig(enabled=True, initial_ideas=4, promote_ideas=2,
                             tuning_trials_per_idea=1, replication_evals=1, max_valid_evals=7)
    overrides = [f"investigation.portfolio.{key}={str(value).lower()}" for key, value in asdict(policy).items()]
    workspace = tmp_path / "managed"
    manifest = prepare(task, workspace, offline=True, overrides=overrides)
    write(workspace / "harness_state.json", {"schema_version": 1, "input_task_sha256": manifest["input_task_sha256"]})
    shutil.copytree(workspace / "input_task", workspace / "run/task")
    runtime = workspace / "run/task"
    root = workspace / "run/investigation"
    baseline = root / "baseline/program_0"
    make_program_dir(baseline, content="SCORE = 0.6\n")
    controller = PortfolioController(root, runtime, policy)
    controller.qualify_baseline({"combined_score": 0.6, "correct": True}, baseline)
    while controller.status == "running":
        lease = controller.reserve_next("0", "3")
        assert lease is not None
        result = Path(lease["result_dir"])
        write(result / "portfolio_lease.json", {"root_results_dir": str(root), "lease": lease})
        idea = result / "ideas" / lease["root_idea_id"]
        program = idea / "01_attempt/program_1"
        index = int(lease["root_idea_id"].split("_")[-1])
        if lease["stage"] == "replication":
            clone_program_dir(src_dir=Path(lease["parent_program_dir"]), dst_dir=program)
            card = deepcopy(controller.data["outcomes"][lease["parent"]["lease_id"]]["idea_bundle"]["whole_idea"])
            score, correct = settings.get("repeat_score", .78), True
        else:
            make_program_dir(program, content=f"TRIAL = {lease['lease_id']!r}\n")
            score = [.7, .5, .4, .65][index - 1] if lease["stage"] == "screening" else (.8 if index == 1 else .55)
            correct = index != 3
            card = {"schema_version": 1, "innovation_axis": lease["innovation_axis"],
                "problem_statement": f"Root {index} whole research question.", "contribution": f"Root {index} contribution.",
                "minimal_validation": {"protocol_id": "development-v1", "decision_rule": "Choose by development score",
                    "stop_or_pivot": "Stop if the whole idea does not improve", "cost_metric": "Steps", "quality_guardrail": "Correctness"},
                "parameter_space": [{"name": "learning_rate", "type": "float", "bounds": [.01, .2],
                    "default": .1 if lease["stage"] == "screening" else .02, "owner": "candidate"}],
                "selection_reason": "Jointly tune the whole idea under the fixed protocol.",
                "comparison_kind": "configuration_sensitivity"}
        bundle = {"idea_id": lease["root_idea_id"], "whole_idea": card}
        write(idea / "idea_bundle.json", bundle)
        write(program / "metrics.json", {"combined_score": score})
        write(program / "correct.json", {"correct": correct})
        write(program / "evaluation_evidence.json", {"schema_version": 1, "episodes": [{"solution": {
            "host_authored": True, "configuration": {item["name"]: item["default"] for item in card["parameter_space"]},
        }}]})
        attempt_id = lease["root_idea_id"] + "_attempt_01"
        write(program / "evaluation_manifest.json", {"producer": "sandbox_host_eval", "execution_boundary": "sandbox",
            "program_id": "program_1", "exit_code": 0, "returncode": 0, "attempt_id": attempt_id,
            "idea_id": lease["root_idea_id"], "adaptive_source_sha256": _program_source_records(program),
            "adaptive_evaluator_sha256": _hash(runtime / "evaluate.py"),
            "adaptive_result_sha256": {name: _hash(program / name) for name in (
                ("metrics.json", "correct.json", "evaluation_evidence.json") if settings.get("bind_configuration", True)
                else ("metrics.json", "correct.json"))}})
        observed = validated_development_outcome(program, task_dir=runtime, attempt_id=attempt_id,
                                                 idea_id=lease["root_idea_id"], native=False)
        assert observed is not None
        if not controller.data["outcomes"]:
            controller.record_failure(lease["lease_id"], "transient_transport_timeout")
        controller.record_receipt(lease["lease_id"], {**observed, "idea_bundle": bundle, "idea_file": str(idea / "idea_bundle.json"),
                                                   "lease_id": lease["lease_id"], "root_idea_id": lease["root_idea_id"]})
    candidate = controller.data["frozen_candidate"]
    write(root / "best_program_manifest.json", {"program_dir": candidate["program_dir"], "combined_score": candidate["score"],
        "correct": True, "root_idea_id": candidate["root_idea_id"], "portfolio_lease_id": candidate["lease_id"],
        "source_hash": candidate["source_hash"], "source_attempt_id": candidate["attempt_id"]})
    write(root / "outer_state.json", {"best_program_dir": candidate["program_dir"], "best_combined_score": candidate["score"],
        "best_correct": True, "portfolio_enabled": True, "portfolio_status": controller.data["status"]})
    paper_data = {"results_dir": str(root), "all_programs": [], "best_correct": {"program_dir": candidate["program_dir"]}}
    return workspace, root, controller, paper_data


def test_complete_projection_keeps_all_history_and_distinguishes_joint_trials(portfolio_study):
    workspace, root, controller, data = portfolio_study
    projected = verify_portfolio_for_paper(data, investigation_results_dir=root)
    assert projected["verification_status"] == "host_verified", projected
    assert projected["status"] == "development_complete"
    assert projected["observed_counts"] == {"root_ideas": 4, "trials": 7,
        "stages": {"screening": 4, "tuning": 2, "replication": 1}, "incorrect": 1,
        "not_selected": 6, "at_or_below_baseline": 3, "transport_failures": 1}
    assert len(projected["trial_preview"]) == 7
    assert projected["preview_complete"]
    assert any(row["correct"] is False for row in projected["trial_preview"])
    assert any(row["score"] < .6 for row in projected["trial_preview"])
    assert "efficiency" in projected["innovation_axes"] and "problem" in projected["innovation_axes"]
    selected = projected["selected_candidate"]["trial_evidence"]
    assert selected["root_idea_id"] == "idea_0001"
    assert selected["stage"] == "tuning"
    assert selected["configuration_evidence"]["configuration"] == {"learning_rate": .02}
    assert selected["configuration_evidence"]["status"] == "host_artifact_bound"
    assert projected["comparison_guidance"]["within_root_tuning"] == "configuration_sensitivity"
    assert projected["comparison_guidance"]["selection_is_component_ablation"] is False
    assert projected["independent_final_confirmation"] is False
    assert projected["development_comparison"]["replication_score_delta"] == pytest.approx(.18)
    index = projected["full_history_index"]
    assert index["sha256"] == _hash(workspace / index["path"])
    assert set(index["sections"].values()) >= {"/outcomes", "/failures", "/leases"}
    assert index["outcome_count"] == len(controller.data["outcomes"])
    data[FIELD] = projected
    assert verify_portfolio_for_paper(data) == projected
    assert verify_portfolio_for_paper(data, investigation_results_dir=workspace / "run") == projected


@pytest.mark.parametrize("portfolio_study", [{"bind_configuration": False}], indirect=True)
def test_legacy_receipt_does_not_certify_unbound_configuration_even_when_file_claims_host_authorship(portfolio_study):
    _workspace, root, controller, data = portfolio_study
    winner = Path(controller.data["frozen_candidate"]["program_dir"])
    payload = read(winner / "evaluation_evidence.json")
    payload["episodes"][0]["solution"]["configuration"]["learning_rate"] = .199
    write(winner / "evaluation_evidence.json", payload)
    projected = verify_portfolio_for_paper(data, investigation_results_dir=root)
    assert projected["verification_status"] == "host_verified", projected
    assert projected["status"] == "needs_evidence"
    configuration = projected["selected_candidate"]["trial_evidence"]["configuration_evidence"]
    assert configuration["status"] == "source_card_declaration_only"
    assert configuration["configuration"] is None
    assert configuration["declared_configuration"] == {"learning_rate": .02}
    assert projected["gaps"][0]["code"] == "executed_configuration_unverified"


@pytest.mark.parametrize("portfolio_study", [{"repeat_score": .4}], indirect=True)
def test_selected_peak_cannot_override_unfavorable_frozen_development_repeats(portfolio_study):
    _workspace, root, _controller, data = portfolio_study
    projected = verify_portfolio_for_paper(data, investigation_results_dir=root)
    comparison = projected["development_comparison"]
    assert comparison["development_score_delta"] > 0
    assert comparison["replication_score_delta"] < 0
    assert comparison["claim_supported"] is False
    assert projected["independent_final_confirmation"] is False


@pytest.mark.parametrize("damage", ["ledger", "ledger_missing", "source", "receipt", "weak_receipt", "negative_metrics", "configuration", "source_card", "lease",
                                     "handoff", "frozen_input", "policy_marker", "saved_projection", "paper_candidate"])
def test_tampering_fails_closed_without_mutating_paper_data(portfolio_study, damage):
    workspace, root, controller, data = portfolio_study
    data[FIELD] = verify_portfolio_for_paper(data, investigation_results_dir=root)
    winner = Path(controller.data["frozen_candidate"]["program_dir"])
    if damage == "ledger":
        state = read(root / STATE_FILE)
        state["selection_reason"] = "Hide lower results"
        write(root / STATE_FILE, state)
    elif damage == "ledger_missing":
        (root / STATE_FILE).unlink()
    elif damage == "source":
        (winner / "src/program.py").write_text("CHANGED = True\n")
    elif damage == "receipt":
        receipt = read(winner / "evaluation_manifest.json")
        receipt["returncode"] = 1
        write(winner / "evaluation_manifest.json", receipt)
    elif damage in {"weak_receipt", "negative_metrics"}:
        outcomes = controller.data["outcomes"].values()
        chosen = next(row for row in outcomes if row["correct"] is (damage == "weak_receipt") and row["score"] < .6)
        path = Path(chosen["program_dir"]) / ("evaluation_manifest.json" if damage == "weak_receipt" else "metrics.json")
        path.write_text("{}")
    elif damage == "configuration":
        payload = read(winner / "evaluation_evidence.json")
        payload["episodes"][0]["solution"]["configuration"]["learning_rate"] = .2
        write(winner / "evaluation_evidence.json", payload)
    elif damage == "source_card":
        card = read(winner.parent.parent / "idea_bundle.json")
        card["whole_idea"]["selection_reason"] = "Changed after selection"
        write(winner.parent.parent / "idea_bundle.json", card)
    elif damage == "lease":
        (winner.parents[3] / "portfolio_lease.json").unlink()
    elif damage == "handoff":
        handoff = read(root / "validation_handoff.json")
        handoff["independent_confirmation_complete"] = True
        write(root / "validation_handoff.json", handoff)
    elif damage == "frozen_input":
        (workspace / "input_task/task.md").write_text("Changed research question")
    elif damage == "policy_marker":
        (workspace / "input_task/research_context/whole_idea_policy.json").unlink()
    elif damage == "saved_projection":
        data[FIELD]["independent_final_confirmation"] = True
    else:
        data["best_correct"]["program_dir"] = str(winner.parent / "program_2")
    before = deepcopy(data)
    result = verify_portfolio_for_paper(data)
    assert result["status"] == "needs_evidence", result
    assert result["verification_status"] == "invalid_or_missing"
    assert result["independent_final_confirmation"] is False
    assert data == before


def test_undeclared_or_missing_managed_state_cannot_downgrade_to_legacy(portfolio_study):
    _workspace, root, _controller, data = portfolio_study
    assert verify_portfolio_for_paper(data)["verification_status"] == "invalid_or_missing"
    (root / STATE_FILE).unlink()
    (root / "outer_state.json").unlink()
    (root / "best_program_manifest.json").unlink()
    assert verify_portfolio_for_paper(data, investigation_results_dir=root)["verification_status"] == "invalid_or_missing"


def test_legacy_absence_and_same_basename_do_not_establish_managed_provenance(tmp_path):
    assert verify_portfolio_for_paper({"results_dir": str(tmp_path)}) is None
    fake = tmp_path / "arbitrary/run/investigation"
    write(fake / STATE_FILE, {"status": "ready_for_validation"})
    assert verify_portfolio_for_paper({"results_dir": str(fake)}) is None
    asserted = {"results_dir": str(fake), FIELD: {"status": "development_complete",
        "provenance": {"investigation_results_dir": str(fake)}}}
    assert verify_portfolio_for_paper(asserted)["verification_status"] == "invalid_or_missing"
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "legacy_managed"
    manifest = prepare(task, workspace, offline=True, overrides=[])
    write(workspace / "harness_state.json", {"schema_version": 1, "input_task_sha256": manifest["input_task_sha256"]})
    results = workspace / "run/investigation"
    results.mkdir(parents=True)
    assert verify_portfolio_for_paper({"results_dir": str(results)}, investigation_results_dir=results) is None


def test_paper_prompt_revalidates_saved_projection_before_each_use(portfolio_study, tmp_path):
    workspace, root, controller, data = portfolio_study
    data[FIELD] = verify_portfolio_for_paper(data, investigation_results_dir=root)
    paper = tmp_path / "paper"
    write(paper / "paper_data.json", data)
    prompt = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    assert "WHOLE-IDEA PORTFOLIO DEVELOPMENT EVIDENCE" in prompt
    assert '"verification_status": "host_verified"' in prompt
    assert "Different hyperparameter settings do not isolate a component effect" in prompt
    assert "testing every module is not required" in prompt
    assert '"transport_failures": 1' in prompt
    winner = Path(controller.data["frozen_candidate"]["program_dir"])
    (winner / "src/program.py").write_text("CHANGED = True\n")
    next_prompt = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    assert '"verification_status": "invalid_or_missing"' in next_prompt
    assert '"status": "development_complete"' not in next_prompt


def test_bounded_preview_preserves_weak_and_incorrect_cases_and_full_index_identity(portfolio_study):
    _workspace, root, _controller, data = portfolio_study
    projection = verify_portfolio_for_paper(data, investigation_results_dir=root)
    template = projection["selected_candidate"]["trial_evidence"]
    rows = [{**template, "lease_id": f"trial_{index:04d}", "correct": index != 90,
             "score": .1 if index == 91 else .7} for index in range(100)]
    preview = _preview(rows, "trial_0099", .6)
    assert len(preview) == MAX_PREVIEW_TRIALS
    assert {"trial_0099", "trial_0090", "trial_0091"} <= {row["lease_id"] for row in preview}
    assert projection["full_history_index"]["all_records_retained"]


def test_cross_directory_receipt_with_matching_basename_cannot_replace_trial(portfolio_study):
    workspace, root, controller, data = portfolio_study
    data[FIELD] = verify_portfolio_for_paper(data, investigation_results_dir=root)
    winner = Path(controller.data["frozen_candidate"]["program_dir"])
    elsewhere = workspace / "copied/program_1"
    shutil.copytree(winner, elsewhere)
    (winner / "evaluation_manifest.json").unlink()
    (winner / "evaluation_manifest.json").symlink_to(elsewhere / "evaluation_manifest.json")
    assert verify_portfolio_for_paper(data)["verification_status"] == "invalid_or_missing"


@pytest.mark.parametrize("missing_receipt_input", [False, True])
def test_review_packet_consumes_exact_portfolio_projection_and_requires_receipt_binding(
    portfolio_study, tmp_path, missing_receipt_input,
):
    from src.pipeline.review import _selected_novelty_evidence
    from tests.pipeline.test_final_ai_review import _Env
    from tests.pipeline.test_review_evidence_binding import _review_for_packet, _return_review, _supplied_evidence

    workspace, root, _controller, data = portfolio_study
    projection = verify_portfolio_for_paper(data, investigation_results_dir=root)
    assert projection["verification_status"] == "host_verified", projection
    data[FIELD] = projection
    paper = workspace / "run/paper"
    write(paper / "paper_data.json", data)
    paths = SimpleNamespace(paper_dir=paper, investigation_dir=root, run_dir=workspace / "run")
    packet_path = _selected_novelty_evidence(paths)
    assert read(packet_path)[FIELD] == projection
    assert len(packet_path.read_bytes()) < 512 * 1024
    env = _Env(tmp_path / "review_consumer")

    def execute(command, *_args):
        supplied = _supplied_evidence(command)
        assert len(supplied) == 1
        assert supplied[0].read_bytes() == packet_path.read_bytes()
        assert read(supplied[0])[FIELD]["full_history_index"]["sha256"] == _hash(root / STATE_FILE)
        review = _review_for_packet(command)
        if missing_receipt_input:
            del review["Provenance"]["inputs"][1:]
        else:
            review["Evidence References"][0]["source"] = "evidence_1"
        return _return_review(command, review)

    result = env.run(execute, reviewer_options={"backend": "agent", "evidence_paths": [packet_path]})
    assert result.status == ("failed" if missing_receipt_input else "completed")
    if missing_receipt_input:
        assert "configured evidence" in result.error
    ledger = root / STATE_FILE
    before = read(ledger)
    before["selection_reason"] = "Tampered after review"
    write(ledger, before)
    changed_packet = _selected_novelty_evidence(paths)
    assert changed_packet != packet_path
    assert read(changed_packet)[FIELD]["verification_status"] == "invalid_or_missing"
