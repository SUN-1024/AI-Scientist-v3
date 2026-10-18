"""Durable scientific reservations, whole-idea ownership and honest handoffs."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.investigation.portfolio_controller import (
    HANDOFF_FILE, STATE_FILE, PortfolioConfig, PortfolioController,
    read_portfolio_state, validate_external_lease,
)
from src.investigation.search_tree import _fingerprint
from tests.program_testutils import make_program_dir


@pytest.fixture
def campaign(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.md").write_text("Find useful new whole ideas.\n")
    (task / "evaluate.py").write_text("# No model execution in this test.\n")
    (task / "protocol.json").write_text('{"seed": 1}')
    (task / "native_evaluation_inputs.json").write_text(json.dumps({"schema_version": 1, "required": ["protocol.json"]}))
    make_program_dir(task / "program_0", content="SCORE = 0.5\n")
    seed = task / "program_0"
    policy = PortfolioConfig(enabled=True, tuning_trials_per_idea=2, max_valid_evals=10)

    def create(*, qualify=True, results_name="results", **changes):
        controller = PortfolioController(tmp_path / results_name, task, replace(policy, **changes))
        if qualify:
            controller.qualify_baseline({"combined_score": 0.5, "correct": True}, seed)
        return controller
    return create


def receipt(controller, lease, *, score=0.6, correct=True, receipt_id=None, bundle=None):
    if lease["stage"] == "replication":
        program = Path(lease["parent_program_dir"])
    else:
        program = controller.results_dir / "programs" / lease["lease_id"]
        make_program_dir(program, content=f"SCORE = {score!r}\n# {lease['lease_id']}\n")
    result = {"score": score, "correct": correct, "receipt_id": receipt_id or f"receipt:{lease['lease_id']}",
              "source_hash": _fingerprint(program), "program_dir": str(program)}
    if bundle is not None:
        result["idea_bundle"] = bundle
    return result


def screen(controller, *, scores=None, correct=True, reverse=False):
    leases = [controller.reserve_next(f"worker-{index}", str(index)) for index in range(controller.policy.initial_ideas)]
    for lease in reversed(leases) if reverse else leases:
        score = scores[lease["root_idea_id"]] if scores else 0.6
        controller.record_receipt(lease["lease_id"], receipt(controller, lease, score=score, correct=correct))
    return leases


def complete(controller, *, tuning_score=0.7, repeat_correct=True):
    while controller.status == "running":
        leases = [controller.reserve_next(f"worker-{index}", str(index)) for index in range(4)]
        leases = [lease for lease in leases if lease]
        assert leases, controller.data
        for lease in reversed(leases):
            controller.record_receipt(lease["lease_id"], receipt(controller, lease, score=tuning_score,
                correct=repeat_correct if lease["stage"] == "replication" else True))


def test_concurrent_reservations_share_cohort_and_reserve_repeat_capacity(campaign):
    controller = campaign()
    def reserve(index):
        other = campaign()
        return other.reserve_next(f"worker-{index}", str(index))
    with ThreadPoolExecutor(max_workers=8) as pool:
        leases = [lease for lease in pool.map(reserve, range(8)) if lease]
    assert len(leases) == 4
    assert len({lease["root_idea_id"] for lease in leases}) == 4
    assert len({lease["lease_id"] for lease in leases}) == 4
    data = read_portfolio_state(controller.results_dir, controller.task_dir)
    assert data["valid_evaluations"] == 0
    assert len(data["pending"]) == 4
    assert set(lease["innovation_axis"] for lease in leases) == {"problem", "efficiency", "data", "objective"}
    assert controller.reserve_next("extra", "9") is None


def test_screen_barrier_then_own_best_joint_tuning(campaign):
    controller = campaign()
    leases = [controller.reserve_next(str(index), str(index)) for index in range(4)]
    for lease in leases[:3]:
        controller.record_receipt(lease["lease_id"], receipt(controller, lease, score=0.6))
    assert controller.reserve_next("free", "9") is None
    assert controller.data["stage"] == "screening"
    controller.record_receipt(leases[3]["lease_id"], receipt(controller, leases[3], score=0.9))
    first = controller.reserve_next("a", "0")
    second = controller.reserve_next("b", "1")
    assert first["stage"] == second["stage"] == "tuning"
    assert first["root_idea_id"] != second["root_idea_id"]
    assert first["parent"]["root_idea_id"] == first["root_idea_id"]
    assert second["parent"]["root_idea_id"] == second["root_idea_id"]
    assert controller.reserve_next("c", "2") is None
    root_id = first["root_idea_id"]
    controller.record_receipt(first["lease_id"], receipt(controller, first, score=0.8))
    next_trial = controller.reserve_next("a", "0")
    assert next_trial["root_idea_id"] == root_id
    assert next_trial["trial_index"] == 2
    assert next_trial["parent_source_hash"] == controller.data["ideas"][root_id]["best"]["source_hash"]


def test_completion_order_does_not_change_ties_or_promotion(campaign):
    left, right = campaign(results_name="left"), campaign(results_name="right")
    screen(left)
    screen(right, reverse=True)
    assert left.data["promoted_ideas"] == right.data["promoted_ideas"] == ["idea_0001", "idea_0002"]
    complete(left)
    complete(right)
    for key in ("root_idea_id", "lease_id", "score"):
        assert left.data["frozen_candidate"][key] == right.data["frozen_candidate"][key]


def test_pending_lease_and_gpu_survive_crash_and_caller_mutation(campaign):
    controller = campaign()
    lease = controller.reserve_next("worker", "3")
    expected = deepcopy(lease)
    lease["parent"]["score"] = 999
    lease["root_idea_id"] = "wrong"
    resumed = campaign()
    assert resumed.reserve_next("worker", "3") == expected
    assert resumed.pending_for_worker("worker") == expected
    with pytest.raises(ValueError, match="GPU assignment"):
        resumed.reserve_next("worker", "4")


def test_idempotence_conflicts_and_receipt_reuse_are_checked(campaign):
    controller = campaign()
    first = controller.reserve_next("a")
    result = receipt(controller, first)
    assert controller.record_receipt(first["lease_id"], result)
    assert not campaign().record_receipt(first["lease_id"], result)
    with pytest.raises(ValueError, match="cannot change"):
        controller.record_receipt(first["lease_id"], {**result, "score": 0.99})
    second = controller.reserve_next("a")
    with pytest.raises(ValueError, match="counted twice"):
        controller.record_receipt(second["lease_id"], receipt(controller, second, receipt_id=result["receipt_id"]))
    assert controller.data["valid_evaluations"] == 1
    assert second["lease_id"] in controller.data["pending"]


@pytest.mark.parametrize("change", ["task", "protocol", "evaluator", "policy"])
def test_policy_and_protocol_changes_fail_closed_even_in_live_instance(campaign, change):
    controller = campaign()
    if change == "policy":
        with pytest.raises(ValueError, match="policy/task/protocol changed"):
            campaign(max_valid_evals=11)
    else:
        path = controller.task_dir / {"task": "task.md", "protocol": "protocol.json", "evaluator": "evaluate.py"}[change]
        path.write_text(path.read_text() + "\nchanged\n")
        with pytest.raises(ValueError, match="policy/task/protocol changed"):
            controller.reserve_next("a")


def test_negative_valid_receipts_are_retained_without_promoting(campaign):
    controller = campaign()
    screen(controller, correct=False)
    assert controller.data["valid_evaluations"] == 4
    assert controller.data["promoted_ideas"] == []
    assert controller.data["stage"] == "replication"
    assert controller.data["frozen_candidate"]["root_idea_id"] == "baseline"
    assert all(row["correct"] is False for row in controller.data["outcomes"].values())
    complete(controller)
    handoff = controller.handoff()
    assert handoff["selection_reason"] == "no_qualified_new_idea"
    assert handoff["claim_supported"] is False
    assert handoff["development_score_delta"] == 0


def test_joint_tuning_finishes_before_freeze_and_repeat_score_never_reselects(campaign):
    controller = campaign()
    screen(controller)
    while controller.data["stage"] == "tuning":
        lease = controller.reserve_next("worker")
        assert lease["action"] == "tune_whole_idea"
        controller.record_receipt(lease["lease_id"], receipt(controller, lease, score=0.8))
    frozen = deepcopy(controller.data["frozen_candidate"])
    assert controller.data["valid_evaluations"] == 8
    lease = controller.reserve_next("worker")
    bad = receipt(controller, lease)
    bad["source_hash"] = "changed"
    with pytest.raises(ValueError, match="changed the frozen candidate"):
        controller.record_receipt(lease["lease_id"], bad)
    controller.record_receipt(lease["lease_id"], receipt(controller, lease, score=0.2))
    assert not (controller.results_dir / HANDOFF_FILE).exists()
    lease = controller.reserve_next("worker")
    controller.record_receipt(lease["lease_id"], receipt(controller, lease, score=0.3))
    assert controller.data["frozen_candidate"] == frozen
    assert controller.data["valid_evaluations"] == 10
    assert controller.status == "ready_for_validation"
    handoff = json.loads((controller.results_dir / HANDOFF_FILE).read_text())
    assert handoff == controller.handoff()
    assert handoff["development_aggregates"]["frozen_candidate"]["mean"] == pytest.approx(0.25)
    assert handoff["development_score_delta"] == pytest.approx(0.3)
    assert handoff["replication_score_delta"] == pytest.approx(-0.25)
    assert handoff["claim_supported"] is False
    assert handoff["independent_confirmation_complete"] is False
    assert handoff["submission_ready"] is False
    assert controller.reserve_next("worker") is None


def test_repeat_supported_improvement_remains_development_only(campaign):
    controller = campaign()
    screen(controller)
    complete(controller, tuning_score=0.7)
    handoff = controller.handoff()
    assert handoff["claim_supported"] is True
    assert handoff["replication_score_delta"] == pytest.approx(0.2)
    assert handoff["claim_scope"] == "development_repeat_mean_improvement_over_qualified_baseline"
    assert handoff["independent_confirmation_complete"] is False


def test_failed_repeat_and_transport_cap_never_emit_handoff(campaign):
    controller = campaign()
    screen(controller)
    complete(controller, repeat_correct=False)
    assert controller.status == "incomplete"
    assert controller.handoff() is None
    assert not (controller.results_dir / HANDOFF_FILE).exists()
    other = campaign(results_name="transport", max_transport_failures=2)
    lease = other.reserve_next("a")
    other.record_failure(lease["lease_id"], "network_timeout")
    assert other.reserve_next("a") == lease
    assert other.data["valid_evaluations"] == 0
    other.record_failure(lease["lease_id"], "network_timeout")
    assert other.status == "incomplete"
    assert other.data["transport_failures"] == 2
    assert other.reserve_next("a") is None
    assert other.pending_for_worker("a") == lease
    assert other.handoff() is None


def test_impossible_budget_is_rejected_before_state_creation(campaign):
    with pytest.raises(ValueError, match="reserve screening, tuning and repeats"):
        campaign(max_valid_evals=9)


def test_baseline_beats_correct_but_weaker_whole_ideas(campaign):
    controller = campaign()
    screen(controller, scores={f"idea_{index:04d}": 0.4 for index in range(1, 5)})
    assert controller.data["promoted_ideas"] == ["idea_0001", "idea_0002"]
    complete(controller, tuning_score=0.45)
    handoff = controller.handoff()
    assert handoff["candidate"]["root_idea_id"] == "baseline"
    assert handoff["candidate"]["score"] == 0.5
    assert handoff["selection_reason"] == "no_improvement"
    assert handoff["claim_supported"] is False


def test_innovation_axes_rotate_beyond_module_changes(campaign):
    controller = campaign(initial_ideas=8, max_valid_evals=14)
    axes = [controller.reserve_next(str(index))["innovation_axis"] for index in range(8)]
    assert axes == ["problem", "efficiency", "data", "objective", "optimization", "structure", "problem", "efficiency"]


@pytest.mark.parametrize("damage", ["contents", "symlink", "hardlink", "permissions", "directory_symlink"])
def test_unsafe_or_tampered_state_is_rejected(campaign, damage, tmp_path):
    controller = campaign()
    state = controller.results_dir / STATE_FILE
    if damage == "contents":
        data = json.loads(state.read_text())
        data["valid_evaluations"] = 100
        state.write_text(json.dumps(data))
    elif damage == "symlink":
        saved = tmp_path / "saved.json"
        state.rename(saved)
        state.symlink_to(saved)
    elif damage == "hardlink":
        (tmp_path / "linked.json").hardlink_to(state)
    elif damage == "permissions":
        state.chmod(0o666)
    else:
        saved = tmp_path / "saved_results"
        controller.results_dir.rename(saved)
        controller.results_dir.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError):
        campaign()


def test_parent_source_changes_are_rejected_and_handoff_recovers_after_crash(campaign):
    controller = campaign()
    lease = controller.reserve_next("a")
    parent = Path(lease["parent_program_dir"])
    make_program_dir(parent, content="SCORE = 99\n")
    with pytest.raises(ValueError, match="sources changed"):
        controller.reserve_next("a")
    other = campaign(results_name="complete", qualify=False)
    other.qualify_baseline({"combined_score": 0.5, "correct": True}, parent)
    screen(other)
    complete(other)
    handoff = other.results_dir / HANDOFF_FILE
    expected = handoff.read_text()
    handoff.unlink()
    PortfolioController(other.results_dir, other.task_dir, other.policy)
    assert handoff.read_text() == expected


def test_child_readonly_verification_and_full_idea_context(campaign):
    controller = campaign()
    screen_leases = [controller.reserve_next(str(index)) for index in range(4)]
    for index, lease in enumerate(screen_leases):
        controller.record_receipt(lease["lease_id"], receipt(controller, lease,
            bundle={"whole_idea": {"title": f"Whole idea {index}", "parameters": {"lr": 0.01}}}))
    lease = controller.reserve_next("a", "0")
    context = controller.context(lease)
    assert context["source_card"]["title"] == "Whole idea 0"
    assert context["idea_history"][0]["outcome"]["idea_bundle"]["whole_idea"]["parameters"] == {"lr": 0.01}
    assert len(context["idea_archive"]) == 3
    before = controller.path.read_bytes()
    payload = {"lease": lease, "root_results_dir": str(controller.results_dir)}
    verified = validate_external_lease(payload, controller.task_dir, Path(lease["result_dir"]), Path(lease["parent_program_dir"]))
    assert verified == context
    assert controller.path.read_bytes() == before
    payload["lease"] = {**lease, "innovation_axis": "forged"}
    with pytest.raises(ValueError, match="differs from its pending host record"):
        validate_external_lease(payload, controller.task_dir, Path(lease["result_dir"]), Path(lease["parent_program_dir"]))


@pytest.mark.parametrize("changes", [{"correct": 1}, {"score": float("nan")}, {"score": True}, {"program_dir": "relative"}])
def test_invalid_outcome_does_not_consume_reservation(campaign, changes):
    controller = campaign()
    lease = controller.reserve_next("a")
    result = {**receipt(controller, lease), **changes}
    with pytest.raises(ValueError):
        controller.record_receipt(lease["lease_id"], result)
    assert controller.data["valid_evaluations"] == 0
    assert controller.data["pending"] == [lease["lease_id"]]
