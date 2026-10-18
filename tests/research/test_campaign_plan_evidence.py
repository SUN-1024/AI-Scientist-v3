"""Approved cards, actual budgets and paper claims retain the same host binding."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.investigation.adaptive_runtime import validated_development_outcome
from src.investigation.campaign_budget import render_campaign_budget, write_campaign_budget
from src.investigation.config import InvestigationConfig
from src.investigation.harness_incumbent import _program_source_records
from src.investigation.native_harness import _hash
from src.investigation.paper_prompt_context import paper_workspace_context_replacements
from src.investigation.portfolio_campaign import bind_portfolio_idea, load_external_portfolio_context
from src.investigation.portfolio_controller import HANDOFF_FILE, STATE_FILE, PortfolioConfig, PortfolioController
from src.investigation.program_clone import clone_program_dir
from src.research.campaign_plan import (
    CAMPAIGN_PLAN_PATH, HOST_CONTEXT_PATH, build_host_context, compile_campaign_plan,
)
from src.research.evidence import render_context
from src.research.portfolio_evidence import FIELD, verify_portfolio_for_paper
from src.research.research_agenda import normalize_agenda
from src.research.workspace import prepare
from tests.program_testutils import make_program_dir
from tests.research import test_portfolio_paper_evidence as portfolio_fixtures
from tests.research.test_portfolio_paper_evidence import read, write
from tests.research.test_unified_research import make_task

portfolio_study = portfolio_fixtures.portfolio_study


def _card(index):
    return {
        "title": f"Method {index}", "role": "proposal" if index == 1 else "control",
        "hypothesis": f"Method {index} has a distinct complete contribution.",
        "complete_implementation": f"IMPLEMENTATION_{index}: preserve the whole approved method and jointly tune learning rate.",
        "falsifier": "The gain disappears under the permitted development protocol.",
        "decision_rule": "Select by correct development outcomes before frozen repeats.",
        "whole_idea": {"schema_version": 1, "innovation_axis": "objective",
            "problem_statement": f"Root {index} whole research question.",
            "contribution": f"Root {index} contribution.",
            "minimal_validation": {"protocol_id": "development-v1", "decision_rule": "Choose by development score",
                "stop_or_pivot": "Stop if the whole idea does not improve", "cost_metric": "Steps", "quality_guardrail": "Correctness"},
            "parameter_space": [{"name": "learning_rate", "type": "float", "bounds": [.01, .2],
                "default": .1, "owner": "candidate"}],
            "selection_reason": "Jointly tune the whole idea under the fixed protocol.",
            "comparison_kind": "configuration_sensitivity"},
    }


@pytest.fixture
def planned_study(tmp_path, request):
    settings = getattr(request, "param", {})
    task = make_task(tmp_path / "source")
    (task / "evaluate.py").write_text("# Host receipt fixture; no external evaluation.\n")
    promote = settings.get("promote", 2)
    policy = PortfolioConfig(enabled=True, initial_ideas=3, promote_ideas=promote,
                             tuning_trials_per_idea=1, replication_evals=1, max_valid_evals=4 + promote)
    overrides = [f"investigation.portfolio.{key}={str(value).lower()}" for key, value in asdict(policy).items()]
    holder = {}

    def planned_outputs(staged, _config):
        context = build_host_context(staged, asdict(policy), 1, {"max_evals_per_variant": 2})
        agenda = normalize_agenda({"schema_version": 1, "research_question": "Compare full methods.",
            "execution_plan": {"schema_version": 1, "roots": [_card(index) for index in range(1, 4)],
                               "allocation": context["allocation"]}})
        plan = compile_campaign_plan(agenda, context)
        holder.update(plan=plan, agenda=agenda)
        return {"research_agenda.json": json.dumps(agenda), HOST_CONTEXT_PATH.as_posix(): json.dumps(context),
                CAMPAIGN_PLAN_PATH.as_posix(): json.dumps(plan)}

    workspace = tmp_path / "managed"
    manifest = prepare(task, workspace, offline=True, overrides=overrides, after_literature=planned_outputs)
    write(workspace / "harness_state.json", {"schema_version": 1, "input_task_sha256": manifest["input_task_sha256"]})
    runtime, root = workspace / "run/task", workspace / "run/investigation"
    shutil.copytree(workspace / "input_task", runtime)
    baseline = root / "baseline/program_0"
    make_program_dir(baseline, content="SCORE = 0.6\n")
    controller = PortfolioController(root, runtime, policy)
    controller.qualify_baseline({"combined_score": .6, "correct": True}, baseline)
    first_context = None
    while controller.status == "running":
        lease = controller.reserve_next("0", "3")
        assert lease is not None
        context = controller.context(lease)
        result = Path(lease["result_dir"])
        write(result / "portfolio_lease.json", {"root_results_dir": str(root), "lease": lease})
        if first_context is None:
            worker = InvestigationConfig()
            worker.execution.portfolio_lease_file = str(result / "portfolio_lease.json")
            worker.execution.max_idea_cycles = 1
            worker.execution.max_program_evals_per_attempt = 1
            worker.execution.harness_evaluated_attempts = True
            worker.paper.enabled = False
            first_context = load_external_portfolio_context(
                cfg=worker, task_dir=runtime, results_dir=result, seed_program_dir=Path(lease["parent_program_dir"]),
            )
            write_campaign_budget(result, worker, cycle=1, max_program_evals_per_attempt=1,
                                  outer_state={}, portfolio_context=first_context)
            holder["screen_prompt"] = render_context(runtime) + render_campaign_budget(result, cycle_index=1)
        idea = result / "ideas" / lease["root_idea_id"]
        program = idea / "01_attempt/program_1"
        index = int(lease["root_idea_id"].split("_")[-1])
        if lease["stage"] == "replication":
            clone_program_dir(src_dir=Path(lease["parent_program_dir"]), dst_dir=program)
            card = deepcopy(controller.data["outcomes"][lease["parent"]["lease_id"]]["idea_bundle"]["whole_idea"])
            score, correct = .78, True
        else:
            make_program_dir(program, content=f"TRIAL = {lease['lease_id']!r}\n")
            card = deepcopy(context["planned_card"]["whole_idea"])
            score = [.7, .65, .4][index - 1] if lease["stage"] == "screening" else (.8 if index == 1 else .55)
            correct = index != 3 or settings.get("all_correct", False)
            if lease["stage"] == "tuning":
                card["parameter_space"][0]["default"] = .02
        bundle = bind_portfolio_idea({"idea_id": lease["root_idea_id"], "whole_idea": card}, context)
        write(idea / "idea_bundle.json", bundle)
        write(program / "metrics.json", {"combined_score": score})
        write(program / "correct.json", {"correct": correct})
        write(program / "evaluation_evidence.json", {"configuration": {item["name"]: item["default"]
                                                                     for item in card["parameter_space"]}})
        attempt_id = lease["root_idea_id"] + "_attempt_01"
        write(program / "evaluation_manifest.json", {"producer": "sandbox_host_eval", "execution_boundary": "sandbox",
            "program_id": "program_1", "exit_code": 0, "returncode": 0, "attempt_id": attempt_id,
            "idea_id": lease["root_idea_id"], "adaptive_source_sha256": _program_source_records(program),
            "adaptive_evaluator_sha256": _hash(runtime / "evaluate.py"),
            "adaptive_result_sha256": {name: _hash(program / name) for name in (
                "metrics.json", "correct.json", "evaluation_evidence.json")}})
        observed = validated_development_outcome(program, task_dir=runtime, attempt_id=attempt_id,
                                                 idea_id=lease["root_idea_id"], native=False)
        assert observed is not None
        if not controller.data["outcomes"]:
            controller.record_failure(lease["lease_id"], "transient_transport_timeout")
        controller.record_receipt(lease["lease_id"], {**observed, "idea_bundle": bundle,
            "idea_file": str(idea / "idea_bundle.json"), "lease_id": lease["lease_id"], "root_idea_id": lease["root_idea_id"]})
    candidate = controller.data["frozen_candidate"]
    write(root / "best_program_manifest.json", {"program_dir": candidate["program_dir"], "combined_score": candidate["score"],
        "correct": True, "root_idea_id": candidate["root_idea_id"], "portfolio_lease_id": candidate["lease_id"],
        "source_hash": candidate["source_hash"], "source_attempt_id": candidate["attempt_id"]})
    write(root / "outer_state.json", {"best_program_dir": candidate["program_dir"], "best_combined_score": candidate["score"],
        "best_correct": True, "portfolio_enabled": True, "portfolio_status": controller.data["status"]})
    data = {"results_dir": str(root), "all_programs": [], "best_correct": {"program_dir": candidate["program_dir"]}}
    return SimpleNamespace(workspace=workspace, root=root, runtime=runtime, controller=controller, data=data, **holder)


def test_plan_prompt_keeps_verified_overview_and_current_complete_card_once(planned_study):
    study = planned_study
    prompt = render_context(study.runtime)
    assert study.plan["plan_sha256"] in prompt
    assert study.plan["agenda_sha256"] in prompt
    assert '"role": "control"' in prompt
    for index in range(1, 4):
        assert f"IMPLEMENTATION_{index}" not in prompt
    assert study.screen_prompt.count("IMPLEMENTATION_1") == 1
    assert "IMPLEMENTATION_2" not in study.screen_prompt
    assert "IMPLEMENTATION_3" not in study.screen_prompt


def test_plan_projection_retains_roles_cards_actual_counts_and_all_failures(planned_study):
    study = planned_study
    projection = verify_portfolio_for_paper(study.data, investigation_results_dir=study.root)
    assert projection["verification_status"] == "host_verified", projection
    planned = projection["campaign_plan"]
    first, second, third = planned["roots"]
    assert first["planned_card"] == study.agenda["execution_plan"]["roots"][0]
    assert [row["role"] for row in planned["roots"]] == ["proposal", "control", "control"]
    assert first["actual_counts"] == {"screening": 1, "tuning": 1, "replication": 1, "correct": 3, "incorrect": 0, "failed": 1}
    assert second["actual_counts"] == {"screening": 1, "tuning": 1, "replication": 0, "correct": 2, "incorrect": 0, "failed": 0}
    assert third["actual_counts"] == {"screening": 1, "tuning": 0, "replication": 0, "correct": 0, "incorrect": 1, "failed": 0}
    assert first["planned_allocation"] == {"screening": 1, "guaranteed_tuning": 0, "tuning_if_promoted": 1, "replication_if_selected": 1}
    assert first["best_configuration_evidence"]["configuration"] == {"learning_rate": .02}
    assert second["best_configuration_evidence"]["configuration"] == {"learning_rate": .1}
    assert third["own_best"] is None
    assert planned["budget_comparison"]["equal_completed_screening_and_tuning_counts"] is False
    assert planned["budget_comparison"]["matched_hpo_verified"] is False
    assert planned["bootstrap_baseline"]["tuned_comparator"] is False
    assert projection["comparison_guidance"]["baseline_tuning_budget_matched"] is False
    assert projection["full_history_index"]["all_records_retained"] is True
    assert projection["full_history_index"]["failure_count"] == 1
    assert projection["full_history_index"]["outcome_count"] == 6
    for row in planned["source_artifacts"].values():
        assert _hash(study.workspace / row["path"]) == row["sha256"]
    study.data[FIELD] = projection
    assert verify_portfolio_for_paper(study.data) == projection


@pytest.mark.parametrize("planned_study", [{"promote": 3}, {"promote": 3, "all_correct": True}], indirect=True)
def test_unused_guaranteed_budget_is_visible_and_equal_counts_never_certify_fair_hpo(planned_study):
    study = planned_study
    projection = verify_portfolio_for_paper(study.data, investigation_results_dir=study.root)
    assert projection["verification_status"] == "host_verified", projection
    assert projection["status"] == "development_complete"
    plan = projection["campaign_plan"]
    third = plan["roots"][2]
    expected_unused = int(third["actual_counts"]["incorrect"] > 0)
    assert plan["unexecuted_total_allocation"] == expected_unused
    assert third["unexecuted_candidate_allocation"]["guaranteed_tuning"] == expected_unused
    assert plan["budget_comparison"]["equal_completed_screening_and_tuning_counts"] is (not expected_unused)
    assert plan["budget_comparison"]["matched_hpo_verified"] is False


@pytest.mark.parametrize("target,missing", [
    ("runtime_plan", False), ("runtime_plan", True), ("runtime_context", False), ("runtime_context", True),
    ("runtime_agenda", False), ("runtime_agenda", True), ("frozen_plan", True),
    ("handoff_counts", False), ("handoff_baseline", False), ("handoff_best", False),
])
def test_plan_tampering_or_missing_bindings_fail_closed(planned_study, target, missing):
    study = planned_study
    study.data[FIELD] = verify_portfolio_for_paper(study.data, investigation_results_dir=study.root)
    paths = {"runtime_plan": study.runtime / CAMPAIGN_PLAN_PATH, "runtime_context": study.runtime / HOST_CONTEXT_PATH,
             "runtime_agenda": study.runtime / "research_agenda.json", "frozen_plan": study.workspace / "input_task" / CAMPAIGN_PLAN_PATH,
             "handoff_counts": study.root / HANDOFF_FILE, "handoff_baseline": study.root / HANDOFF_FILE,
             "handoff_best": study.root / HANDOFF_FILE}
    path = paths[target]
    if missing:
        path.unlink()
    else:
        raw = read(path)
        if target == "handoff_counts":
            raw["planned_root_execution"][0]["actual_evaluations"]["tuning"] += 1
        elif target == "handoff_baseline":
            raw["baseline_tuned_comparator"] = True
        elif target == "handoff_best":
            raw["planned_root_execution"][1]["own_best"]["score"] = .99
        elif target == "runtime_plan":
            raw["roots"][0]["complete_implementation"] = "Replace the approved implementation."
        elif target == "runtime_context":
            raw["sources"][0]["text"] += " Contract changed."
        elif target == "runtime_agenda":
            raw["research_question"] = "A different agenda."
        else:
            raw["tampered"] = True
        write(path, raw)
    assert verify_portfolio_for_paper(study.data)["verification_status"] == "invalid_or_missing"
    if target.startswith("runtime_"):
        with pytest.raises(ValueError):
            render_context(study.runtime)


def test_paper_and_review_receive_plan_without_reselecting_frozen_candidate(planned_study):
    from src.pipeline.review import _selected_novelty_evidence

    study = planned_study
    frozen = deepcopy(study.controller.data["frozen_candidate"])
    projection = verify_portfolio_for_paper(study.data, investigation_results_dir=study.root)
    study.data[FIELD] = projection
    paper = study.workspace / "run/paper"
    write(paper / "paper_data.json", study.data)
    shutil.copytree(study.runtime, paper / "task")
    prompt = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    assert prompt.count("IMPLEMENTATION_1") == 1 and prompt.count("IMPLEMENTATION_2") == 1
    assert '"matched_hpo_verified": false' in prompt
    packet = _selected_novelty_evidence(SimpleNamespace(paper_dir=paper, investigation_dir=study.root, run_dir=study.workspace / "run"))
    assert read(packet)[FIELD] == projection
    assert study.controller.data["frozen_candidate"] == frozen
    assert projection["selected_candidate"]["program_dir"] == frozen["program_dir"]
    study.data["best_correct"]["program_dir"] = str(study.root / "baseline/program_0")
    assert verify_portfolio_for_paper(study.data)["verification_status"] == "invalid_or_missing"


def test_control_best_claim_is_rechecked_against_host_receipt_not_just_ledger_hash(planned_study):
    from src.investigation.adaptive_controller import _digest

    study = planned_study
    state = read(study.root / STATE_FILE)
    state["ideas"]["idea_0002"]["best"]["score"] = .99
    state["state_hash"] = _digest({key: value for key, value in state.items() if key != "state_hash"})
    write(study.root / STATE_FILE, state)
    handoff = read(study.root / HANDOFF_FILE)
    handoff["planned_root_execution"][1]["own_best"]["score"] = .99
    write(study.root / HANDOFF_FILE, handoff)
    projection = verify_portfolio_for_paper(study.data, investigation_results_dir=study.root)
    assert projection["verification_status"] == "invalid_or_missing"
    assert "root best candidate differs from its host receipt" in projection["detail"]


def test_legacy_projection_has_no_new_plan_fields_and_replays_identically(portfolio_study):
    _workspace, root, _controller, data = portfolio_study
    projection = verify_portfolio_for_paper(data, investigation_results_dir=root)
    assert projection["verification_status"] == "host_verified"
    assert "campaign_plan" not in projection
    assert "plan_sha256" not in projection["provenance"]
    assert "plan_artifacts" not in projection["full_history_index"]
    assert "matched_hpo_verified" not in projection["comparison_guidance"]
    before = json.dumps(projection, ensure_ascii=False, sort_keys=True)
    data[FIELD] = deepcopy(projection)
    assert json.dumps(verify_portfolio_for_paper(data), ensure_ascii=False, sort_keys=True) == before


def test_legacy_no_catalog_retains_exact_context_bytes(tmp_path):
    (tmp_path / "research_agenda.json").write_text("not JSON")
    expected = "No structured literature dossier is attached. Do not infer that novelty has been verified."
    assert render_context(tmp_path) == expected
    assert render_context(tmp_path, for_paper=True) == expected
