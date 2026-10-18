"""Evidence-driven stages resume safely and stop without overstating completion."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import yaml

from src.investigation.adaptive_controller import (
    AdaptiveController,
    AdaptiveInvestigationConfig,
    HANDOFF_FILE,
)
from src.investigation.config import ConfigValidationError, config_to_flat_dict
from src.investigation.config_loader import load_config, write_hydra_snapshot
from tests.program_testutils import make_program_dir


@pytest.fixture
def campaign(tmp_path: Path):
    task = tmp_path / "task"
    task.mkdir()
    seed = task / "program_0"
    make_program_dir(seed, content="SCORE = 0.5\n")
    (task / "task.md").write_text("Compare candidate mechanisms on development data.\n")
    (task / "evaluate.py").write_text("# Unit-test fixture: no model or API execution.\n")
    (task / "protocol.json").write_text('{"seeds": [1, 2]}')
    (task / "native_evaluation_inputs.json").write_text(json.dumps({
        "schema_version": 1, "required": ["protocol.json"],
    }))
    results = tmp_path / "results"
    policy = AdaptiveInvestigationConfig(
        enabled=True, min_tuning_evals=1, min_creative_evals=1,
        min_refinement_evals=1, stagnation_evals=1, replication_evals=1,
        max_valid_evals=20,
    )

    def create(*, qualify: bool = True, **changes) -> AdaptiveController:
        controller = AdaptiveController(results, task, replace(policy, **changes))
        if qualify:
            controller.qualify_baseline({"combined_score": 0.5, "correct": True}, seed)
        return controller

    return create


def outcome(controller: AdaptiveController, index: int, *, score: float = 0.6,
            correct: bool = True, source_hash: str | None = None) -> dict:
    best = controller.data["best"]
    return {
        "receipt_id": f"receipt-{index}", "score": score, "correct": correct,
        "program_dir": best["program_dir"],
        "source_hash": source_hash or best["source_hash"],
    }


def complete_cycle(controller: AdaptiveController, cycle: int, *, score: float = 0.6,
                   correct: bool = True) -> dict:
    lease = controller.plan_cycle(cycle)
    assert lease is not None
    assert controller.record_outcome(
        f"outcome-{cycle}", outcome(controller, cycle, score=score, correct=correct),
    )
    controller.finish_cycle(cycle)
    return lease


def test_negative_experiments_count_but_transport_failures_do_not(campaign):
    controller = campaign(min_tuning_evals=2)
    controller.plan_cycle(1)
    controller.record_transport_failure("request-1")
    controller.record_transport_failure("request-1")
    assert controller.record_outcome("infra-1", None, failure_kind="transport")
    controller.finish_cycle(1)
    assert controller.data["transport_failures"] == 1
    assert controller.data["valid_evaluations"] == 0
    assert controller.data["stage_counts"] == {}

    complete_cycle(controller, 2, score=0.99, correct=False)
    assert controller.data["valid_evaluations"] == 1
    assert controller.data["stage_counts"] == {"baseline_tuning/optimization": 1}
    assert controller.data["best"]["score"] == 0.5
    assert controller.data["substage"] == "optimization"
    complete_cycle(controller, 3, score=0.6)
    assert controller.data["valid_evaluations"] == 2
    assert controller.data["stage_counts"]["baseline_tuning/optimization"] == 2
    assert controller.data["best"]["score"] == 0.6
    assert controller.data["substage"] == "replication"


def test_pending_work_and_recorded_outcome_survive_resume(campaign):
    controller = campaign()
    assert controller.reconcile({}) == 1
    lease = controller.plan_cycle(1)
    controller.update_pending(phase="agent_started", attempt_id="attempt-1")
    resumed = campaign()
    assert resumed.reconcile({"current_cycle": 99}) == 1
    assert resumed.plan_cycle(1)["attempt_id"] == "attempt-1"
    assert resumed.plan_cycle(1)["lease_id"] == lease["lease_id"]
    with pytest.raises(ValueError, match="pending cycle"):
        resumed.plan_cycle(2)
    with pytest.raises(ValueError, match="before recording"):
        resumed.finish_cycle(1)

    observation = outcome(resumed, 1)
    assert resumed.record_outcome("result-1", observation)
    recorded = campaign()
    assert recorded.reconcile({"current_cycle": 99}) == 1
    assert recorded.data["pending"]["phase"] == "outcome_recorded"
    assert recorded.record_outcome("result-1", observation) is False
    assert recorded.data["valid_evaluations"] == 1
    recorded.finish_cycle(1)
    finished = campaign()
    assert finished.reconcile({"current_cycle": 1}) == 2
    assert finished.data["completed_cycles"] == [1]
    assert finished.record_outcome("result-1", observation) is False


def test_completed_outcome_conflict_is_rejected_without_mutating_counts(campaign):
    controller = campaign()
    controller.plan_cycle(1)
    observation = outcome(controller, 1)
    controller.record_outcome("result-1", observation)
    before = json.dumps(controller.data, sort_keys=True)
    with pytest.raises(ValueError, match="cannot change"):
        controller.record_outcome("result-1", {**observation, "score": 0.9})
    assert json.dumps(controller.data, sort_keys=True) == before


def test_one_pending_lease_cannot_accept_two_different_outcomes(campaign):
    controller = campaign(min_tuning_evals=3)
    controller.plan_cycle(1)
    controller.record_outcome("result-1", outcome(controller, 1))
    before = json.dumps(controller.data, sort_keys=True)
    with pytest.raises(ValueError):
        controller.record_outcome("result-2", outcome(controller, 2))
    assert json.dumps(controller.data, sort_keys=True) == before


def test_one_receipt_cannot_be_counted_in_two_cycles(campaign):
    controller = campaign()
    controller.plan_cycle(1)
    observation = outcome(controller, 1)
    controller.record_outcome("result-1", observation)
    controller.finish_cycle(1)
    controller.plan_cycle(2)
    with pytest.raises(ValueError, match="counted twice"):
        controller.record_outcome("result-2", observation)
    assert controller.data["valid_evaluations"] == 1
    assert controller.data["pending"]["phase"] == "planned"


@pytest.mark.parametrize("change", ["policy", "task", "protocol", "evaluator"])
def test_resume_rejects_changed_policy_or_task_protocol(campaign, change):
    controller = campaign()
    if change == "policy":
        with pytest.raises(ValueError, match="policy/task/protocol changed"):
            campaign(max_valid_evals=21)
        return
    filename = {"task": "task.md", "protocol": "protocol.json", "evaluator": "evaluate.py"}[change]
    path = controller.task_dir / filename
    path.write_text(path.read_text() + "\nchanged\n")
    with pytest.raises(ValueError, match="policy/task/protocol changed"):
        campaign()


def test_replication_rejects_changed_candidate_before_counting(campaign):
    controller = campaign()
    complete_cycle(controller, 1)
    lease = controller.plan_cycle(2)
    assert lease["parent_policy"] == "frozen"
    before = json.dumps(controller.data, sort_keys=True)
    with pytest.raises(ValueError, match="changed the frozen candidate"):
        controller.record_outcome("result-2", outcome(controller, 2, source_hash="changed-source"))
    assert json.dumps(controller.data, sort_keys=True) == before


def test_replication_aggregate_keeps_repeatability_separate_from_confirmation(campaign):
    controller = campaign(replication_evals=2)
    complete_cycle(controller, 1)
    complete_cycle(controller, 2, score=0.5)
    assert controller.data["substage"] == "replication"
    complete_cycle(controller, 3, score=0.7)
    aggregate = controller.data["aggregates"]["baseline_tuning"]
    assert aggregate["n"] == 2
    assert aggregate["mean"] == pytest.approx(0.6)
    assert aggregate["std"] == pytest.approx(0.1)
    assert aggregate["all_correct"] is True
    assert aggregate["independent_confirmation"] is False
    assert aggregate["evidence_scope"] == "same_frozen_development_protocol_repeatability"
    assert aggregate["source_hash"] == controller.data["best"]["source_hash"]
    assert controller.data["main_stage"] == "creative_research"
    assert not (controller.results_dir / HANDOFF_FILE).exists()


def test_failed_replication_stops_without_validation_handoff(campaign):
    controller = campaign()
    complete_cycle(controller, 1)
    complete_cycle(controller, 2, correct=False)
    assert controller.data["status"] == "incomplete"
    assert controller.data["transition_reason"] == "frozen_candidate_replication_failed"
    assert controller.data["aggregates"]["baseline_tuning"]["all_correct"] is False
    assert controller.plan_cycle(3) is None
    assert not (controller.results_dir / HANDOFF_FILE).exists()


def test_complete_transition_wins_at_exact_budget_and_assigns_parent_policies(campaign):
    controller = campaign(max_valid_evals=5)
    expected = [
        ("baseline_tuning", "optimization", "tune_baseline_training", "incumbent"),
        ("baseline_tuning", "replication", "replicate_frozen_candidate", "frozen"),
        ("creative_research", "exploration", "test_distinct_mechanism", "explore"),
        ("creative_research", "refinement", "refine_supported_mechanism", "incumbent"),
        ("creative_research", "replication", "replicate_frozen_candidate", "frozen"),
    ]
    for cycle, specification in enumerate(expected, 1):
        lease = complete_cycle(controller, cycle, score=0.6 if cycle < 3 else 0.7)
        assert tuple(lease[key] for key in ("main_stage", "substage", "action", "parent_policy")) == specification
    assert controller.data["valid_evaluations"] == 5
    assert controller.data["status"] == "ready_for_validation"
    assert controller.plan_cycle(6) is None
    assert controller.data["status"] == "ready_for_validation"
    handoff = json.loads((controller.results_dir / HANDOFF_FILE).read_text())
    assert handoff["status"] == "ready_for_validation"
    assert handoff["candidate"]["score"] == 0.7
    assert handoff["required_phases"] == ["ablations", "final_evaluation", "posthoc", "paper"]
    assert handoff["independent_confirmation_complete"] is False
    assert handoff["submission_ready"] is False
    for key in ("policy_hash", "task_hash", "protocol_hash"):
        assert handoff[key] == controller.data[key]
    assert set(handoff["development_aggregates"]) == {"baseline_tuning", "creative_research"}
    context = controller.context(lease)
    assert context["remaining_valid_evaluations"] == 0
    assert context["independent_confirmation"] is False


def test_scientific_budget_exhaustion_keeps_campaign_incomplete(campaign):
    controller = campaign(max_valid_evals=4)
    for cycle in range(1, 5):
        complete_cycle(controller, cycle, score=0.6 if cycle < 3 else 0.7)
    assert controller.plan_cycle(5) is None
    assert controller.data["status"] == "incomplete"
    assert controller.data["transition_reason"] == "scientific_evaluation_budget_exhausted"
    assert not (controller.results_dir / HANDOFF_FILE).exists()


def test_transport_budget_exhaustion_does_not_create_scientific_progress(campaign):
    controller = campaign(max_transport_failures=1)
    controller.plan_cycle(1)
    controller.record_transport_failure("request-failed")
    controller.record_outcome("request-failed", None, failure_kind="transport")
    controller.finish_cycle(1)
    assert controller.plan_cycle(2) is None
    assert controller.data["status"] == "incomplete"
    assert controller.data["transition_reason"] == "transport_failure_budget_exhausted"
    assert controller.data["valid_evaluations"] == 0


def test_adaptive_config_defaults_overrides_and_snapshot(tmp_path):
    defaults = load_config([])
    assert asdict(defaults.adaptive_investigation) == asdict(AdaptiveInvestigationConfig())
    cfg = load_config([
        "adaptive_investigation.enabled=true", "adaptive_investigation.replication_evals=3",
        "adaptive_investigation.max_valid_evals=17", "adaptive_investigation.min_improvement=0.02",
        "execution.harness_evaluated_attempts=true", "execution.prune_every_n_cycles=0",
    ])
    assert cfg.adaptive_investigation.enabled is True
    assert cfg.adaptive_investigation.replication_evals == 3
    assert cfg.adaptive_investigation.max_valid_evals == 17
    assert cfg.adaptive_investigation.min_improvement == 0.02
    write_hydra_snapshot(cfg, tmp_path)
    saved = yaml.safe_load((tmp_path / ".hydra/config.yaml").read_text())
    assert saved["adaptive_investigation"] == asdict(cfg.adaptive_investigation)
    flat = config_to_flat_dict(cfg)
    assert flat["adaptive_investigation.replication_evals"] == "3"


@pytest.mark.parametrize("field,value", [
    ("enabled", "1"), ("policy_version", "2"), ("policy_version", "true"),
    ("policy_version", "1.0"), ("min_tuning_evals", "0"),
    ("min_creative_evals", "true"), ("min_refinement_evals", "-1"),
    ("stagnation_evals", "1.5"), ("replication_evals", "null"),
    ("max_valid_evals", "0"), ("max_transport_failures", "0"),
    ("min_improvement", "-0.1"), ("min_improvement", ".nan"),
    ("min_improvement", ".inf"), ("min_improvement", "true"),
])
def test_adaptive_config_rejects_invalid_policy_values(field, value):
    with pytest.raises(ConfigValidationError):
        load_config([f"adaptive_investigation.{field}={value}"])


@pytest.mark.parametrize("override", [
    "execution.workers=2", "execution.campaign_workers=2",
    "execution.harness_evaluated_attempts=false", "execution.prune_every_n_cycles=1",
])
def test_adaptive_config_rejects_unmanaged_parallel_or_untrusted_execution(override):
    message = "workers=1" if "workers=" in override else "host evaluation and pruning disabled"
    with pytest.raises(ConfigValidationError) as error:
        load_config([
            "adaptive_investigation.enabled=true", "execution.harness_evaluated_attempts=true",
            "execution.prune_every_n_cycles=0", override,
        ])
    assert message in error.value.message


@pytest.mark.parametrize("reason", [
    "scientific_evaluation_budget_exhausted", "transport_failure_budget_exhausted",
    "frozen_candidate_replication_failed", "idea_generation_attempts_exhausted",
])
def test_schedule_extension_never_reopens_hard_stops(campaign, reason):
    controller = campaign()
    controller.stop(reason)
    assert controller.resume_schedule(100) is False
    assert controller.data["status"] == "incomplete"
    assert controller.data["transition_reason"] == reason


def test_schedule_extension_retains_stagnation_and_still_respects_scientific_cap(campaign):
    controller = campaign(min_tuning_evals=10, max_valid_evals=1)
    complete_cycle(controller, 1, score=0.4)
    assert controller.data["stagnation"] == 1
    controller.stop("scheduled_cycle_budget_exhausted", cycle_limit=1)
    assert controller.data["stagnation"] == 1
    assert controller.resume_schedule(100) is False
    assert controller.data["status"] == "incomplete"
