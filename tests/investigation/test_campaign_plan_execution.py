"""Approved plans reach the real durable scheduler without model or GPU calls."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.investigation.adaptive_controller import _digest
from src.investigation.campaign_budget import render_campaign_budget, write_campaign_budget
from src.investigation.config import InvestigationConfig
from src.investigation.portfolio_campaign import bind_portfolio_idea
from src.investigation.portfolio_controller import (
    STATE_FILE, PortfolioConfig, PortfolioController, read_portfolio_state, validate_external_lease,
)
from src.investigation.search_tree import _fingerprint
from src.research.campaign_plan import (
    CAMPAIGN_PLAN_PATH, HOST_CONTEXT_PATH, build_host_context, compile_campaign_plan,
    write_campaign_artifacts,
)
from tests.investigation.test_portfolio_external_lease import ExternalFixture, whole_card
from tests.program_testutils import make_program_dir


def planned_task(tmp_path, *, publish=True):
    task = tmp_path / "task"
    make_program_dir(task / "program_0", content="print('bootstrap')\n")
    (task / "task.md").write_text("Compare candidate methods under one fixed development protocol.\n")
    (task / "evaluate.py").write_text("# Fixed evaluator fixture.\n")
    policy = PortfolioConfig(enabled=True, initial_ideas=2, promote_ideas=2,
                             tuning_trials_per_idea=2, replication_evals=1, max_valid_evals=7)
    context = build_host_context(task, asdict(policy), workers=2, downstream_limits={})
    roots = []
    for index, role in enumerate(("proposal", "control")):
        whole = whole_card("optimization")
        whole["contribution"] = f"Compare complete configuration family {index}."
        roots.append({"title": f"Complete method {index}", "role": role,
                      "hypothesis": f"Family {index} produces a useful development comparison.",
                      "complete_implementation": f"Implement family {index} in src/program.py.\nKeep the fixed evaluator unchanged.",
                      "falsifier": "Reject an incorrect or weaker candidate.",
                      "decision_rule": "Use the same development score and trial allocation.",
                      "whole_idea": whole})
    agenda = {"execution_plan": {"schema_version": 1, "roots": roots, "allocation": context["allocation"]}}
    plan = compile_campaign_plan(agenda, context)
    if publish:
        (task / "research_agenda.json").write_text(json.dumps(agenda))
        write_campaign_artifacts(task, context, plan)
    return task, policy, context, agenda, plan


def controller_for(tmp_path):
    task, policy, _, _, plan = planned_task(tmp_path)
    controller = PortfolioController(tmp_path / "results", task, policy)
    controller.qualify_baseline({"score": .5, "correct": True}, task / "program_0")
    return controller, plan


def outcome_for(controller, lease, score, *, default=None, correct=True):
    context = controller.context(lease)
    bundle = {"whole_idea": deepcopy(context["source_card"]), "target_rubrics": []}
    if default is not None:
        bundle["whole_idea"]["parameter_space"][0]["default"] = default
    bundle = bind_portfolio_idea(bundle, context)
    if lease["stage"] == "replication":
        program = Path(lease["parent_program_dir"])
    else:
        program = controller.results_dir / "programs" / lease["lease_id"]
        make_program_dir(program, content=f"print({lease['lease_id']!r})\n")
    return {"score": score, "correct": correct, "receipt_id": "receipt:" + lease["lease_id"],
            "program_dir": str(program), "source_hash": _fingerprint(program), "idea_bundle": bundle}


def test_complete_same_axis_cards_screen_then_each_root_tunes_own_best(tmp_path):
    controller, plan = controller_for(tmp_path)
    screens = [controller.reserve_next(index) for index in (1, 2)]
    assert [lease["innovation_axis"] for lease in screens] == ["optimization", "optimization"]
    for index, lease in enumerate(screens):
        context = controller.context(lease)
        approved = plan["roots"][index]
        assert context["source_card"] == approved["whole_idea"]
        assert context["planned_card"]["complete_implementation"] == approved["complete_implementation"]
        assert not {"root_idea_id", "card_sha256", "idea_key"} & context["planned_card"].keys()
        assert lease["plan_sha256"] == plan["plan_sha256"]
        assert lease["card_sha256"] == approved["card_sha256"]
        result = outcome_for(controller, lease, .6 + index * .05)
        assert result["idea_bundle"]["implementation_plan"] == [approved["complete_implementation"]]
        controller.record_receipt(lease["lease_id"], result)
        assert not controller.record_receipt(lease["lease_id"], result)
    assert controller.data["stage"] == "tuning"
    tuned_parents = {}
    first_tunes = [controller.reserve_next(index) for index in (1, 2)]
    for lease in first_tunes:
        assert lease["parent"]["root_idea_id"] == lease["root_idea_id"]
        result = outcome_for(controller, lease, .8, default=.7)
        controller.record_receipt(lease["lease_id"], result)
        tuned_parents[lease["root_idea_id"]] = result["source_hash"]
    second_tunes = [controller.reserve_next(index) for index in (1, 2)]
    for lease in second_tunes:
        assert lease["parent_source_hash"] == tuned_parents[lease["root_idea_id"]]
        assert controller.context(lease)["planned_card"] == controller.data["ideas"][lease["root_idea_id"]]["planned_card"]
        controller.record_receipt(lease["lease_id"], outcome_for(controller, lease, .7, default=.4))
    repeat = controller.reserve_next(1)
    controller.record_receipt(repeat["lease_id"], outcome_for(controller, repeat, .8))
    handoff = controller.handoff()
    assert controller.data["valid_evaluations"] == 7
    assert handoff["plan_sha256"] == plan["plan_sha256"]
    assert handoff["baseline_evidence_role"] == "bootstrap"
    assert handoff["baseline_tuned_comparator"] is False
    assert {row["role"] for row in handoff["planned_root_execution"]} == {"proposal", "control"}
    assert all(row["actual_evaluations"]["screening"] == 1 and row["actual_evaluations"]["tuning"] == 2
               for row in handoff["planned_root_execution"])
    assert all(row["own_best"]["source_hash"] == tuned_parents[row["root_idea_id"]]
               for row in handoff["planned_root_execution"])
    resumed = PortfolioController(controller.results_dir, controller.task_dir, controller.policy)
    resumed.qualify_baseline({"score": .5, "correct": True}, controller.task_dir / "program_0")
    assert resumed.handoff() == handoff


@pytest.mark.parametrize("stage", ["screening", "tuning"])
@pytest.mark.parametrize("mutation", ["problem_statement", "contribution", "protocol", "parameter_space", "axis"])
def test_bundle_cannot_replace_approved_core(tmp_path, stage, mutation):
    controller, _ = controller_for(tmp_path)
    lease = controller.reserve_next(1)
    context = controller.context(lease)
    context["stage"] = stage
    bundle = {"whole_idea": deepcopy(context["source_card"])}
    whole = bundle["whole_idea"]
    if mutation in {"problem_statement", "contribution"}:
        whole[mutation] = "A replacement root."
    elif mutation == "protocol":
        whole["minimal_validation"]["protocol_id"] = "other-protocol"
    elif mutation == "parameter_space":
        whole["parameter_space"][0]["bounds"] = [0., 2.]
    else:
        whole["innovation_axis"] = "structure"
    with pytest.raises(ValueError):
        bind_portfolio_idea(bundle, context)


def test_planned_budget_carries_full_implementation_before_first_screen(tmp_path):
    controller, _ = controller_for(tmp_path)
    lease = controller.reserve_next(1)
    context = {**controller.context(lease), "host_verified": True}
    cfg = InvestigationConfig()
    budget = write_campaign_budget(controller.results_dir, cfg, cycle=1,
        max_program_evals_per_attempt=1, outer_state={}, portfolio_context=context)
    assert budget["portfolio"]["planned_card"] == context["planned_card"]
    assert budget["campaign_plan"]["plan_sha256"] == lease["plan_sha256"]
    for index in (None, 1):
        prompt = render_campaign_budget(controller.results_dir, cycle_index=index)
        assert prompt.count("Implement family 0 in src/program.py.") == 1
        assert "Do not invent a replacement idea" in prompt
        assert "bootstrap evidence" in prompt
    controller.record_receipt(lease["lease_id"], outcome_for(controller, lease, .7))
    other = controller.reserve_next(2)
    controller.record_receipt(other["lease_id"], outcome_for(controller, other, .6))
    tune = controller.reserve_next(1)
    context = {**controller.context(tune), "host_verified": True}
    write_campaign_budget(controller.results_dir, cfg, cycle=1,
        max_program_evals_per_attempt=1, outer_state={}, portfolio_context=context)
    assert render_campaign_budget(controller.results_dir).count("Implement family 0 in src/program.py.") == 1
    saved = json.loads((controller.results_dir / "effective_campaign_budget.json").read_text())
    assert saved["portfolio"]["idea_history"][0]["outcome"]["idea_bundle"]["planned_card"] == context["planned_card"]


@pytest.mark.parametrize("damage", ["missing_plan", "missing_host_context", "agenda", "plan", "replacement_plan", "public_source"])
def test_planned_recovery_and_pending_lease_fail_closed_after_artifact_damage(tmp_path, damage):
    controller, _ = controller_for(tmp_path)
    lease = controller.reserve_next(1)
    if damage == "missing_plan":
        (controller.task_dir / CAMPAIGN_PLAN_PATH).unlink()
    elif damage == "missing_host_context":
        (controller.task_dir / HOST_CONTEXT_PATH).unlink()
    elif damage == "agenda":
        (controller.task_dir / "research_agenda.json").write_text("{}")
    elif damage == "plan":
        path = controller.task_dir / CAMPAIGN_PLAN_PATH
        raw = json.loads(path.read_text())
        raw["roots"][0]["complete_implementation"] = "Replace the implementation."
        path.write_text(json.dumps(raw))
    elif damage == "replacement_plan":
        agenda_path = controller.task_dir / "research_agenda.json"
        agenda = json.loads(agenda_path.read_text())
        agenda["execution_plan"]["roots"][0]["complete_implementation"] = "A valid replacement implementation."
        host_context = json.loads((controller.task_dir / HOST_CONTEXT_PATH).read_text())
        replacement = compile_campaign_plan(agenda, host_context)
        agenda_path.write_text(json.dumps(agenda))
        (controller.task_dir / CAMPAIGN_PLAN_PATH).write_text(json.dumps(replacement))
    else:
        (controller.task_dir / "program_0/src/program.py").write_text("print('replacement seed')\n")
    for operation in (
        lambda: controller.reserve_next(1),
        lambda: PortfolioController(controller.results_dir, controller.task_dir, controller.policy),
        lambda: read_portfolio_state(controller.results_dir, controller.task_dir),
        lambda: validate_external_lease({"root_results_dir": str(controller.results_dir), "lease": lease},
            controller.task_dir, Path(lease["result_dir"]), Path(lease["parent_program_dir"])),
    ):
        with pytest.raises(ValueError):
            operation()


def test_missing_planned_state_cannot_reissue_work_from_memory_or_existing_trial_directory(tmp_path):
    controller, _ = controller_for(tmp_path)
    lease = controller.reserve_next(1)
    Path(lease["result_dir"]).mkdir(parents=True)
    controller.path.unlink()
    with pytest.raises(ValueError, match="state is missing"):
        controller.reserve_next(1)
    with pytest.raises(ValueError, match="state is missing"):
        PortfolioController(controller.results_dir, controller.task_dir, controller.policy)


def test_actual_policy_mismatch_rejected_before_new_state(tmp_path):
    task, policy, _, _, _ = planned_task(tmp_path)
    with pytest.raises(ValueError, match="policy"):
        PortfolioController(tmp_path / "results", task, replace(policy, max_transport_failures=7))
    assert not (tmp_path / "results" / STATE_FILE).exists()


def test_declared_plan_cannot_silently_fall_back_to_axis_rotation(tmp_path):
    task, policy, _, agenda, _ = planned_task(tmp_path, publish=False)
    (task / "research_agenda.json").write_text(json.dumps(agenda))
    with pytest.raises(ValueError, match="missing"):
        PortfolioController(tmp_path / "results", task, policy)


@pytest.mark.parametrize("target", ["root", "lease"])
def test_rehashed_state_cannot_change_bound_root_card_or_pending_axis(tmp_path, target):
    controller, _ = controller_for(tmp_path)
    lease = controller.reserve_next(1)
    data = json.loads(controller.path.read_text())
    if target == "root":
        data["ideas"][lease["root_idea_id"]]["planned_card"]["complete_implementation"] = "Changed implementation."
    else:
        data["leases"][lease["lease_id"]]["innovation_axis"] = "structure"
    data["state_hash"] = _digest({key: value for key, value in data.items() if key != "state_hash"})
    controller.path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PortfolioController(controller.results_dir, controller.task_dir, controller.policy)


def test_legacy_pending_lease_recovers_without_adding_plan_or_rewriting_axes(tmp_path):
    task, policy, context, agenda, plan = planned_task(tmp_path, publish=False)
    controller = PortfolioController(tmp_path / "results", task, policy)
    controller.qualify_baseline({"score": .5, "correct": True}, task / "program_0")
    lease = controller.reserve_next(1)
    before = controller.path.read_bytes()
    assert lease["innovation_axis"] == "problem"
    (task / "research_agenda.json").write_text(json.dumps(agenda))
    write_campaign_artifacts(task, context, plan)
    resumed = PortfolioController(controller.results_dir, task, policy)
    assert resumed.reserve_next(1) == lease
    assert resumed.path.read_bytes() == before
    assert "plan_sha256" not in resumed.data
    assert "planned_card" not in resumed.context(lease)
    assert resumed.context(lease)["source_card"] is None


def test_complete_plan_reaches_existing_idea_coding_and_host_receipt_boundary(tmp_path):
    fixture = ExternalFixture(tmp_path)
    # This fixture has not reserved or executed any work. Recreate its empty
    # campaign after publishing the input plan, as normal setup would do.
    fixture.controller.path.unlink()
    context = build_host_context(fixture.task, asdict(fixture.policy), workers=1, downstream_limits={})
    card = {"title": "Approved full method", "role": "control", "hypothesis": "Test the full declared method.",
            "complete_implementation": "Keep the candidate architecture. Jointly tune alpha and width in src/program.py.",
            "falsifier": "No correct development candidate improves the bootstrap.",
            "decision_rule": "Keep the highest correct development score.", "whole_idea": whole_card()}
    agenda = {"execution_plan": {"schema_version": 1, "roots": [card], "allocation": context["allocation"]}}
    plan = compile_campaign_plan(agenda, context)
    (fixture.task / "research_agenda.json").write_text(json.dumps(agenda))
    write_campaign_artifacts(fixture.task, context, plan)
    fixture.controller = PortfolioController(fixture.results, fixture.task, fixture.policy)
    fixture.controller.qualify_baseline({"combined_score": .5, "correct": True}, fixture.seed)
    for stage in ("screening", "tuning", "replication"):
        fixture.reserve()
        assert fixture.lease["stage"] == stage
        assert fixture.run_child() == 0
        result = fixture.commit_child()
        assert result["idea_bundle"]["planned_card"] == card
        if stage != "replication":
            assert result["idea_bundle"]["implementation_plan"] == [card["complete_implementation"]]
        assert result["idea_bundle"]["card_sha256"] == plan["roots"][0]["card_sha256"]
    assert len(fixture.generated) == len(fixture.coded) == 2
    assert len(fixture.host_calls) == 3
    assert fixture.controller.handoff()["planned_root_execution"][0]["actual_evaluations"] == {
        "screening": 1, "tuning": 1, "replication": 1,
    }


def cli_config(task, results, policy, *, container):
    cfg = InvestigationConfig()
    cfg.paths.task_dir, cfg.paths.results_dir = str(task), str(results)
    cfg.paths.instruct = str(task / "task.md")
    cfg.container.enabled, cfg.container.gpu_count = container, 0
    cfg.container.image, cfg.container.auth_root = "sha256:fixture", str(task)
    cfg.portfolio = policy
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.max_program_evals_per_attempt = 1
    cfg.execution.resume = False
    cfg.paper.enabled = False
    return cfg


@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("damage", ["missing_plan", "tampered_plan", "policy", "pending_plan_missing"])
def test_cli_rejects_invalid_plan_before_runtime_baseline_or_agent(tmp_path, container, damage):
    from src.investigation import cli

    task, policy, _, _, _ = planned_task(tmp_path)
    results = tmp_path / "results"
    if damage == "pending_plan_missing":
        controller = PortfolioController(results, task, policy)
        controller.qualify_baseline({"score": .5, "correct": True}, task / "program_0")
        controller.reserve_next(1)
        previous_state = controller.path.read_bytes()
        (task / CAMPAIGN_PLAN_PATH).unlink()
    elif damage == "missing_plan":
        (task / CAMPAIGN_PLAN_PATH).unlink()
    elif damage == "tampered_plan":
        (task / CAMPAIGN_PLAN_PATH).write_text("{}")
    else:
        policy = replace(policy, max_transport_failures=policy.max_transport_failures + 1)
    cfg = cli_config(task, results, policy, container=container)
    with (
        patch.object(cli, "load_config", return_value=cfg),
        patch.object(cli, "run_task_setup") as setup,
        patch.object(cli, "prepare_task_runtime") as runtime,
        patch("src.investigation.native_harness.ensure_native_baseline") as baseline,
        patch.object(cli, "run_task_drafting") as drafting,
        patch.object(cli, "run_idea_generation") as idea,
        patch.object(cli, "run_single_attempt") as coding,
        patch.object(cli, "log_event") as events,
    ):
        assert cli.main() == 2
        for operation in (setup, runtime, baseline, drafting, idea, coding):
            operation.assert_not_called()
        assert any(call.kwargs.get("reason") in {"portfolio_coordinator_error", "invalid_campaign_plan"}
                   for call in events.call_args_list)
    if damage == "pending_plan_missing":
        assert (results / STATE_FILE).read_bytes() == previous_state
    else:
        assert not (results / STATE_FILE).exists()


@pytest.mark.parametrize("container", [False, True])
def test_cli_rechecks_plan_after_setup_before_any_runtime_evaluation(tmp_path, container):
    from src.investigation import cli

    task, policy, _, _, _ = planned_task(tmp_path)
    (task / "capture.py").write_text("# capture fixture\n")
    cfg = cli_config(task, tmp_path / "results", policy, container=container)

    def change_setup(*args):
        (task / "program_0/src/program.py").write_text("print('source changed during setup')\n")
        return True

    with (
        patch.object(cli, "load_config", return_value=cfg),
        patch.object(cli, "run_task_setup", side_effect=change_setup) as setup,
        patch.object(cli, "prepare_task_runtime") as runtime,
        patch("src.investigation.native_harness.ensure_native_baseline") as baseline,
        patch.object(cli, "run_task_drafting") as drafting,
        patch.object(cli, "run_idea_generation") as idea,
        patch.object(cli, "run_single_attempt") as coding,
        patch.object(cli, "_probe_docker_available", return_value=(True, "fixture")),
        patch.object(cli, "_pin_container_image", return_value="sha256:fixture"),
        patch("src.investigation.main_entry.shutil.which", return_value="/fixture/docker"),
        patch.object(cli, "log_event") as events,
    ):
        assert cli.main() == 2
        setup.assert_called_once()
        for operation in (runtime, baseline, drafting, idea, coding):
            operation.assert_not_called()
        assert any(call.kwargs.get("reason") == "invalid_campaign_plan" for call in events.call_args_list)


def test_cli_legacy_state_keeps_legacy_path_before_bootstrap(tmp_path):
    from src.investigation import cli

    task, policy, _, agenda, _ = planned_task(tmp_path, publish=False)
    controller = PortfolioController(tmp_path / "results", task, policy)
    controller.qualify_baseline({"score": .5, "correct": True}, task / "program_0")
    controller.reserve_next(1)
    before = controller.path.read_bytes()
    (task / "research_agenda.json").write_text(json.dumps(agenda))
    cfg = cli_config(task, controller.results_dir, policy, container=False)
    with patch.object(cli, "load_config", return_value=cfg), patch.object(cli, "run_task_setup", return_value=False) as setup:
        assert cli.main() == 2
        setup.assert_called_once()
    assert controller.path.read_bytes() == before


@pytest.mark.parametrize("artifact", ["plan", "host_context_only", "planned_state_only", "declared_agenda_only"])
def test_cli_cannot_disable_portfolio_to_bypass_an_existing_plan(tmp_path, artifact):
    from src.investigation import cli

    task, policy, _, _, _ = planned_task(tmp_path)
    results = tmp_path / "results"
    if artifact == "planned_state_only":
        controller = PortfolioController(results, task, policy)
        controller.qualify_baseline({"score": .5, "correct": True}, task / "program_0")
        controller.reserve_next(1)
        (task / CAMPAIGN_PLAN_PATH).unlink()
        (task / HOST_CONTEXT_PATH).unlink()
        (task / "research_agenda.json").unlink()
    elif artifact == "host_context_only":
        (task / CAMPAIGN_PLAN_PATH).unlink()
        (task / "research_agenda.json").unlink()
    elif artifact == "declared_agenda_only":
        (task / CAMPAIGN_PLAN_PATH).unlink()
        (task / HOST_CONTEXT_PATH).unlink()
    cfg = cli_config(task, results, replace(policy, enabled=False), container=False)
    with (
        patch.object(cli, "load_config", return_value=cfg),
        patch.object(cli, "run_task_setup") as setup,
        patch.object(cli, "prepare_task_runtime") as runtime,
        patch("src.investigation.native_harness.ensure_native_baseline") as baseline,
        patch.object(cli, "run_idea_generation") as idea,
        patch.object(cli, "run_single_attempt") as coding,
        patch.object(cli, "log_event") as events,
    ):
        assert cli.main() == 2
        for operation in (setup, runtime, baseline, idea, coding):
            operation.assert_not_called()
        assert any(call.kwargs.get("reason") == "invalid_campaign_plan" for call in events.call_args_list)


def test_cli_legacy_whole_idea_marker_does_not_force_portfolio(tmp_path):
    from src.investigation import cli
    from src.research.whole_idea_policy import freeze_whole_idea_policy

    task, policy, _, _, _ = planned_task(tmp_path, publish=False)
    freeze_whole_idea_policy(task)
    cfg = cli_config(task, tmp_path / "results", replace(policy, enabled=False), container=False)
    with patch.object(cli, "load_config", return_value=cfg), patch.object(cli, "run_task_setup", return_value=False) as setup:
        assert cli.main() == 2
        setup.assert_called_once()
