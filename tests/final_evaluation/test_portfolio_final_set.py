"""Planned final comparisons freeze before use and reuse only verified outputs."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.final_evaluation.comparison_set import (
    SET_MANIFEST, freeze_portfolio_final_set, portfolio_final_set_required,
    run_portfolio_final_evaluation, verify_portfolio_final_set,
)
from src.final_evaluation.runner import FinalEvaluationError, run_final_evaluation
from src.investigation.adaptive_controller import _digest
from src.investigation.adaptive_runtime import validated_development_outcome
from src.investigation.harness_incumbent import _program_source_records
from src.investigation.native_harness import _hash
from src.investigation.portfolio_campaign import bind_portfolio_idea
from src.investigation.portfolio_controller import PortfolioConfig, PortfolioController
from src.investigation.program_clone import clone_program_dir
from src.research.campaign_plan import build_host_context, compile_campaign_plan, write_campaign_artifacts
from tests.final_evaluation.test_final_evaluation import _write_contract_task, _write_investigation
from tests.investigation.test_portfolio_external_lease import whole_card
from tests.program_testutils import make_program_dir


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def receipt(program, task, *, score, idea_id, configuration=True):
    write(program / "metrics.json", {"combined_score": score})
    write(program / "correct.json", {"correct": True})
    names = ["metrics.json", "correct.json"]
    if configuration:
        write(program / "evaluation_evidence.json", {"configuration": {"learning_rate": .1}})
        names.append("evaluation_evidence.json")
    attempt_id = idea_id + "_attempt_01"
    write(program / "evaluation_manifest.json", {
        "producer": "sandbox_host_eval", "execution_boundary": "sandbox", "program_id": program.name,
        "exit_code": 0, "returncode": 0, "attempt_id": attempt_id, "idea_id": idea_id,
        "fixture_invocation": str(program),
        "adaptive_source_sha256": _program_source_records(program),
        "adaptive_evaluator_sha256": _hash(task / "evaluate.py"),
        "adaptive_result_sha256": {name: _hash(program / name) for name in names},
    })
    return validated_development_outcome(program, task_dir=task, attempt_id=attempt_id, idea_id=idea_id, native=False)


def study(tmp_path, *, winner="proposal", configuration=True, promote=3, final_configuration=.1):
    task = _write_contract_task(tmp_path)
    if final_configuration is not None:
        with (task / "evaluate_final.py").open("a") as handle:
            evidence = {"schema_version": 1, "episodes": [{"seed": 2, "score": .75,
                "correct": True, "duration_seconds": 0., "solution": {
                    "configuration": {"learning_rate": final_configuration}}}]}
            handle.write("\n(out / 'evaluation_evidence.json').write_text(" + repr(json.dumps(evidence)) + ")\n")
    make_program_dir(task / "program_0", content="BASELINE = True\n")
    policy = PortfolioConfig(enabled=True, initial_ideas=3, promote_ideas=promote,
                             tuning_trials_per_idea=1, replication_evals=1,
                             max_valid_evals=3 + promote + 1)
    context = build_host_context(task, asdict(policy), workers=1, downstream_limits={})
    roots = [{"title": f"Method {index}", "role": role, "hypothesis": f"Compare family {index}.",
              "complete_implementation": f"Implement family {index} as a complete method.",
              "falsifier": "Reject an incorrect method.", "decision_rule": "Use the fixed development score.",
              "whole_idea": whole_card("optimization")}
             for index, role in enumerate(("proposal", "control", "control"))]
    for index, root in enumerate(roots):
        root["whole_idea"]["contribution"] = f"Whole method {index} contribution."
    agenda = {"execution_plan": {"schema_version": 1, "roots": roots, "allocation": context["allocation"]}}
    plan = compile_campaign_plan(agenda, context)
    write(task / "research_agenda.json", agenda)
    write_campaign_artifacts(task, context, plan)
    root = tmp_path / "investigation"
    baseline = root / "baseline/program_0"
    make_program_dir(baseline, content="BASELINE = True\n")
    baseline_score = .99 if winner == "bootstrap" else .1
    baseline_outcome = receipt(baseline, task, score=baseline_score, idea_id="baseline", configuration=configuration)
    controller = PortfolioController(root, task, policy)
    controller.qualify_baseline(baseline_outcome, baseline)
    while controller.status == "running":
        lease = controller.reserve_next(1)
        result = Path(lease["result_dir"])
        write(result / "portfolio_lease.json", {"root_results_dir": str(root), "lease": lease})
        idea = result / "ideas" / lease["root_idea_id"]
        program = idea / "01_attempt/program_1"
        if lease["stage"] == "replication":
            clone_program_dir(src_dir=Path(lease["parent_program_dir"]), dst_dir=program)
            score = lease["parent"]["score"]
        else:
            make_program_dir(program, content=f"TRIAL = {lease['lease_id']!r}\n")
            index = int(lease["root_idea_id"].split("_")[-1])
            score = .8 if (index == (2 if winner == "control" else 1)) else .3 + index / 100
            if lease["stage"] == "tuning":
                score += .01
        lease_context = controller.context(lease)
        bundle = bind_portfolio_idea({"idea_id": lease["root_idea_id"],
                                     "whole_idea": deepcopy(lease_context["source_card"])}, lease_context)
        write(idea / "idea_bundle.json", bundle)
        observed = receipt(program, task, score=score, idea_id=lease["root_idea_id"], configuration=configuration)
        assert observed
        controller.record_receipt(lease["lease_id"], {**observed, "idea_bundle": bundle,
            "idea_file": str(idea / "idea_bundle.json"), "lease_id": lease["lease_id"],
            "root_idea_id": lease["root_idea_id"]})
    best = controller.data["frozen_candidate"]
    write(root / "best_program_manifest.json", {"program_dir": best["program_dir"],
        "entrypoint_path": "src/program.py", "combined_score": best["score"], "correct": True,
        "root_idea_id": best["root_idea_id"], "portfolio_lease_id": best["lease_id"], "source_hash": best["source_hash"]})
    return task, root, tmp_path / "final", controller


def evaluate(task, root, output, **options):
    return run_portfolio_final_evaluation(task_dir=task, investigation_results_dir=root, output_dir=output,
                                         python_bin=Path(sys.executable), _allow_native_for_tests=True, **options)


def test_freezes_selected_and_two_controls_before_first_evaluator_then_reuses_zero_calls(tmp_path):
    task, root, output, controller = study(tmp_path)
    calls, protocols = [], []
    def run(command, **kwargs):
        frozen = json.loads((output / SET_MANIFEST).read_text())
        assert len(frozen["methods"]) == 3
        assert len({row["root_idea_id"] for row in frozen["methods"]}) == 3
        calls.append(command)
        protocols.append(frozen["protocol"])
        return subprocess.run(command, **kwargs)
    assert portfolio_final_set_required(task_dir=task, investigation_results_dir=root)
    result = evaluate(task, root, output, run=run)
    assert len(calls) == 3 and all(protocol == protocols[0] for protocol in protocols)
    assert result["status"] == "completed"
    assert result["selected_proposal_root_id"] == controller.data["frozen_candidate"]["root_idea_id"]
    assert (output / "final_manifest.json").is_file()
    assert all(row["metrics"] == {"final_accuracy": .75} for row in result["methods"])
    assert all(row["actual_evaluations"]["tuning"] == 1 and row["tuned"] for row in result["methods"])
    assert all(row["configuration_evidence"]["status"] == "host_artifact_bound" for row in result["methods"])
    assert all(row["final_configuration_status"] == "matches_frozen_development_configuration" for row in result["methods"])
    before = (output / SET_MANIFEST).read_bytes()
    repeated = evaluate(task, root, output, run=lambda *a, **kw: pytest.fail("reran completed method"))
    assert repeated == result and (output / SET_MANIFEST).read_bytes() == before


@pytest.mark.parametrize("damage", ["card", "source", "receipt", "config", "set", "set_rehashed", "plan"])
def test_frozen_comparison_rejects_tampering_before_first_call(tmp_path, damage):
    task, root, output, _ = study(tmp_path)
    frozen = freeze_portfolio_final_set(task_dir=task, investigation_results_dir=root, output_dir=output)
    source = Path(frozen["methods"][-1]["source_program_dir"])
    if damage == "source":
        (source / "src/program.py").write_text("CHANGED = True\n")
    elif damage in {"receipt", "config"}:
        (source / ("evaluation_manifest.json" if damage == "receipt" else "evaluation_evidence.json")).write_text("{}")
    elif damage == "card":
        (source.parent.parent / "idea_bundle.json").write_text("{}")
    elif damage == "plan":
        (task / "research_agenda.json").write_text("{}")
    else:
        frozen["methods"].pop()
        if damage == "set_rehashed":
            from src.final_evaluation.comparison_set import _digest as set_digest
            frozen["set_sha256"] = set_digest({key: value for key, value in frozen.items() if key != "set_sha256"})
        (output / SET_MANIFEST).chmod(0o600)
        write(output / SET_MANIFEST, frozen)
    with pytest.raises(FinalEvaluationError):
        evaluate(task, root, output, run=lambda *a, **kw: pytest.fail("read final data after tampering"))


def test_partial_failure_keeps_set_and_never_reruns_successful_or_failed_method(tmp_path):
    task, root, output, _ = study(tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            return subprocess.CompletedProcess(command, 1, "", "fixture failure")
        return subprocess.run(command, **kwargs)
    with pytest.raises(FinalEvaluationError, match="return code 1"):
        evaluate(task, root, output, run=run)
    frozen = (output / SET_MANIFEST).read_bytes()
    partial = verify_portfolio_final_set(task_dir=task, investigation_results_dir=root, output_dir=output,
                                         require_complete=False)
    assert partial["status"] == "incomplete" and sum(row["final_manifest"] is not None for row in partial["methods"]) == 1
    with pytest.raises(FinalEvaluationError, match="already started"):
        evaluate(task, root, output, run=lambda *a, **kw: pytest.fail("rerun after partial failure"))
    assert (output / SET_MANIFEST).read_bytes() == frozen


def test_existing_final_invocation_without_freeze_is_rejected(tmp_path):
    task, root, output, _ = study(tmp_path)
    write(output / "invocation.json", {"status": "started"})
    with pytest.raises(FinalEvaluationError, match="without a prior frozen"):
        evaluate(task, root, output, run=lambda *a, **kw: pytest.fail("final read before freeze"))
    assert not (output / SET_MANIFEST).exists()


@pytest.mark.parametrize("damage", [None, "missing_state", "missing_plan_and_state", "agenda_only"])
def test_public_single_method_api_cannot_bypass_planned_collection(tmp_path, damage):
    task, root, output, controller = study(tmp_path)
    if damage:
        controller.path.unlink()
    if damage in {"missing_plan_and_state", "agenda_only"}:
        (task / "research_context/campaign_plan.json").unlink()
    if damage == "agenda_only":
        (task / "research_context/host_context.json").unlink()
    with pytest.raises(FinalEvaluationError):
        run_final_evaluation(task_dir=task, investigation_results_dir=root, output_dir=output,
                             python_bin=Path(sys.executable), _allow_native_for_tests=True,
                             run=lambda *a, **kw: pytest.fail("single-method bypass read final data"))
    assert not output.exists()


def test_changed_execution_protocol_or_deleted_set_cannot_resume(tmp_path):
    task, root, output, _ = study(tmp_path)
    evaluate(task, root, output)
    with pytest.raises(FinalEvaluationError, match="execution protocol changed"):
        evaluate(task, root, output, timeout_seconds=119,
                 run=lambda *a, **kw: pytest.fail("changed final protocol executed"))
    (output / SET_MANIFEST).unlink()
    with pytest.raises(FinalEvaluationError, match="without a prior frozen"):
        evaluate(task, root, output, run=lambda *a, **kw: pytest.fail("refroze after final access"))


@pytest.mark.parametrize("damage", ["execution", "missing_method_binding"])
def test_completed_methods_cross_bind_frozen_execution(tmp_path, damage):
    task, root, output, _ = study(tmp_path)
    result = evaluate(task, root, output)
    if damage == "execution":
        from src.final_evaluation.comparison_set import _digest as set_digest
        path = output / "comparison_execution.json"
        value = json.loads(path.read_text())
        value["settings"].update(container_image="sha256:" + "f" * 64, native_for_tests=False)
        value["execution_sha256"] = set_digest({key: item for key, item in value.items() if key != "execution_sha256"})
        path.chmod(0o600)
        write(path, value)
    else:
        (output / result["methods"][-1]["output_relative_path"] / "comparison_method.json").unlink()
    with pytest.raises(FinalEvaluationError, match="binding|execution protocol"):
        verify_portfolio_final_set(task_dir=task, investigation_results_dir=root, output_dir=output)


def test_final_configuration_change_stops_before_next_method(tmp_path):
    task, root, output, _ = study(tmp_path, final_configuration=.2)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.run(command, **kwargs)
    with pytest.raises(FinalEvaluationError, match="changed its frozen configuration"):
        evaluate(task, root, output, run=run)
    assert len(calls) == 1
    with pytest.raises(FinalEvaluationError, match="changed its frozen configuration"):
        evaluate(task, root, output, run=lambda *a, **kw: pytest.fail("continued after configuration drift"))


def test_missing_final_configuration_is_an_explicit_evidence_gap(tmp_path):
    task, root, output, _ = study(tmp_path, final_configuration=None)
    result = evaluate(task, root, output)
    assert sum(gap["code"] == "final_configuration_unverified" for gap in result["gaps"]) == 3
    assert all(row["final_configuration_status"] == "final_configuration_unavailable" for row in result["methods"])


def test_same_final_seed_evaluator_and_resource_protocol_for_every_method(tmp_path):
    task, root, output, _ = study(tmp_path)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.run(command, **kwargs)
    result = evaluate(task, root, output, timeout_seconds=110, run=run)
    manifests = [row["final_manifest"] for row in result["methods"]]
    for field in ("task_tree_sha256", "evaluator_sha256", "declared_final_metrics"):
        assert all(row[field] == manifests[0][field] for row in manifests)
    assert all(row["compute_usage"]["wall_clock_limit_seconds"] == 110 for row in manifests)
    assert result["protocol"]["protocol"]["final_reset_sequence"] == [2]
    assert all(command[:2] == commands[0][:2] for command in commands)


@pytest.mark.parametrize("winner", ["control", "bootstrap"])
def test_control_or_bootstrap_winner_has_no_selected_proposal(tmp_path, winner):
    task, root, output, controller = study(tmp_path, winner=winner)
    result = evaluate(task, root, output)
    assert result["selected_proposal_root_id"] is None
    assert result["selected_method_id"] == controller.data["frozen_candidate"]["root_idea_id"]
    assert len(result["methods"]) == (2 if winner == "control" else 3)
    assert any(row["code"] == "no_selected_proposal" for row in result["gaps"])


def test_unpromoted_control_is_screened_not_tuned_and_missing_config_is_explicit(tmp_path):
    task, root, output, _ = study(tmp_path, promote=1, configuration=False)
    result = evaluate(task, root, output)
    assert all(row["actual_evaluations"]["tuning"] == 0 and not row["tuned"]
               for row in result["methods"] if row["is_control"])
    assert len([gap for gap in result["gaps"] if gap["code"] == "executed_configuration_unverified"]) == 3


def test_missing_control_best_and_arbitrary_method_override_fail_closed(tmp_path):
    task, root, output, controller = study(tmp_path)
    frozen = freeze_portfolio_final_set(task_dir=task, investigation_results_dir=root, output_dir=output)
    with pytest.raises(FinalEvaluationError, match="outside its frozen"):
        run_final_evaluation(task_dir=task, investigation_results_dir=root, output_dir=output / "other",
                             python_bin=Path(sys.executable), _allow_native_for_tests=True,
                             _portfolio_method=(output, frozen["methods"][0]["method_id"]))
    data = json.loads(controller.path.read_text())
    control_id = next(key for key, row in data["ideas"].items() if row["role"] == "control")
    data["ideas"][control_id]["best"] = None
    data["state_hash"] = _digest({key: value for key, value in data.items() if key != "state_hash"})
    write(controller.path, data)
    with pytest.raises(FinalEvaluationError, match="no valid own-best"):
        freeze_portfolio_final_set(task_dir=task, investigation_results_dir=root, output_dir=tmp_path / "new_final")


def test_legacy_single_method_keeps_existing_output_and_once_only_semantics(tmp_path):
    task = _write_contract_task(tmp_path)
    root, _ = _write_investigation(tmp_path)
    output = tmp_path / "final"
    assert not portfolio_final_set_required(task_dir=task, investigation_results_dir=root)
    options = dict(task_dir=task, investigation_results_dir=root, output_dir=output,
                   python_bin=Path(sys.executable), _allow_native_for_tests=True)
    first = run_final_evaluation(**options)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert run_final_evaluation(**options, run=lambda *a, **kw: pytest.fail("legacy reran")) == first
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert not (output / SET_MANIFEST).exists()
