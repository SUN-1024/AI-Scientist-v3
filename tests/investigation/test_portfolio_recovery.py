"""A paid candidate can recover host transport without another coder or trial."""
import json
import os
from pathlib import Path
import shutil

import pytest
import yaml

from src.investigation import portfolio_recovery as recovery
from src.investigation.evaluation_usage import compute_usage
from src.investigation.harness_incumbent import _program_source_records
from src.investigation.native_harness import _hash
from src.investigation.portfolio_campaign import COORDINATOR_LOCK, WORKER_LOCK, _exclusive_lock
from src.investigation.portfolio_controller import PortfolioConfig, PortfolioController, read_portfolio_state
from src.orchestrator import state as state_store
from tests.program_testutils import make_program_dir


class RecoveryFixture:
    def __init__(self, root: Path, *, planned=False):
        self.task = root / "task"
        self.seed = self.task / "program_0"
        make_program_dir(self.seed, content="print('baseline')\n")
        (self.task / "task.md").write_text("A fixed development research task.\n")
        (self.task / "evaluate.py").write_text("# Immutable fixture evaluator\n")
        python = self.task / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("container-only runtime marker\n")
        self.results = root / "run"
        self.policy = PortfolioConfig(enabled=True, initial_ideas=2, promote_ideas=1,
            tuning_trials_per_idea=1, replication_evals=1, max_valid_evals=4, max_transport_failures=6)
        if planned:
            from tests.investigation.test_campaign_plan_execution import planned_task
            self.task, self.policy, _, _, self.plan = planned_task(root)
        self.controller = PortfolioController(self.results, self.task, self.policy)
        self.controller.qualify_baseline({"combined_score": .4, "correct": True}, self.seed)
        self.lease = self.controller.reserve_next(1, "device=0")
        self.worker = Path(self.lease["result_dir"])
        self.program = self.worker / "ideas/root/01_attempt/program_1"
        make_program_dir(self.program, content="print('already paid candidate')\n")
        shutil.copytree(self.seed, self.program.parent / "program_0")
        shutil.copyfile(self.task / "evaluate.py", self.program.parent / "evaluate.py")
        self.bundle = {"idea_id": self.lease["root_idea_id"], "title": "A whole idea",
            "hypothesis": "A fixed mechanism improves quality.", "target_rubrics": [],
            "whole_idea": {"schema_version": 1, "innovation_axis": self.lease["innovation_axis"],
                "problem_statement": "One bounded research question.", "contribution": "A coherent contribution.",
                "minimal_validation": {"protocol_id": "fixed-v1", "decision_rule": "Retain stronger correct candidate.",
                    "stop_or_pivot": "Stop at budget.", "cost_metric": "runtime", "quality_guardrail": "correct is true"},
                "parameter_space": [], "selection_reason": "Smallest permitted screen."}}
        if planned:
            context = self.controller.context(self.lease)
            self.bundle["whole_idea"] = context["source_card"]
            self.bundle = recovery.bind_portfolio_idea(self.bundle, context)
        self.write(self.program.parent.parent / "idea_bundle.json", self.bundle)
        self.write(self.program.parent / "active_rubrics.json", [])
        self.attempt_id = self.lease["root_idea_id"] + "_attempt_01"
        self.write(self.worker / "portfolio_lease.json", {"root_results_dir": str(self.results), "lease": self.lease})
        self.write(self.worker / "portfolio_process.json", {"pid": 99999999, "status": "exited"})
        self.write(self.worker / ".hydra/config.yaml", {})
        (self.worker / ".hydra/config.yaml").write_text(yaml.safe_dump({
            "paths": {"task_dir": str(self.task), "results_dir": str(self.worker), "seed_program": str(self.seed)},
            "container": {"enabled": True, "image": "sha256:fixture-image"},
            "execution": {"harness_evaluation_timeout_seconds": 900, "gpu_evaluation": True,
                          "harness_evaluated_attempts": True}}))
        self.usage = compute_usage(wall_clock_limit_seconds=900, elapsed_seconds=100,
                                  max_rpc_calls=3000, rpc_calls=1979, outcome="failed")
        self.write(self.program / "evaluation_failure.json", self.usage)
        self.history = {"attempt_id": self.attempt_id, "idea_id": self.lease["root_idea_id"],
                        "run_exit_code": 1, "evaluation_performed": False,
                        "output_tail": "CandidateContainerCleanupError: candidate runtime could not remove container"}
        (self.worker / "attempts_history.jsonl").write_text(json.dumps(self.history) + "\n")
        (self.program.parent / "wrapper_output.log").write_text(self.history["output_tail"])
        if planned:
            from tests.investigation.test_campaign_plan_execution import outcome_for
            other = self.controller.reserve_next(2, "device=1")
            self.controller.record_receipt(other["lease_id"], outcome_for(self.controller, other, .8))
        for _ in range(6):
            self.controller.record_failure(self.lease["lease_id"], "worker_exit_1")
        self.write(self.results / state_store.OUTER_STATE_FILE, {"current_cycle": 0})
        self.write(self.results / state_store.BEST_MANIFEST_FILE,
                   {"program_dir": str(self.seed), "combined_score": .4, "correct": True})
        self.evidence = root / "retained_evidence.json"
        self.capture()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def capture(self):
        files = [self.worker / "portfolio_lease.json", self.worker / "attempts_history.jsonl",
                 self.program / "evaluation_failure.json"]
        self.write(self.evidence, {"captured_at": "2026-10-18T11:05:07+00:00",
            "files": [{"source": str(p), "text": p.read_text(), "sha256": _hash(p)} for p in files],
            "candidate_source_files": [{"path": name, "sha256": digest}
                                       for name, digest in _program_source_records(self.program).items()]})

    def receipt(self, correct=True):
        self.write(self.program / "metrics.json", {"combined_score": .6})
        self.write(self.program / "correct.json", {"correct": correct})
        self.write(self.program / "evaluation_manifest.json", {
            "producer": "sandbox_host_eval", "execution_boundary": "sandbox", "program_id": "program_1",
            "attempt_id": self.attempt_id, "idea_id": self.lease["root_idea_id"], "exit_code": 0, "returncode": 0,
            "adaptive_source_sha256": _program_source_records(self.program),
            "adaptive_evaluator_sha256": _hash(self.task / "evaluate.py"),
            "adaptive_result_sha256": {name: _hash(self.program / name) for name in ("metrics.json", "correct.json")},
            "compute_usage": {**self.usage, "outcome": "completed"}})

    def recover(self, **kwargs):
        return recovery.recover_lease(results_dir=self.results, task_dir=self.task,
            lease_id=self.lease["lease_id"], evidence_file=kwargs.pop("evidence_file", self.evidence), **kwargs)

    def state(self):
        return read_portfolio_state(self.results, self.task)

    def ledger(self):
        path = self.results / recovery.RECOVERY_DIRECTORY / self.lease["lease_id"] / "ledger.json"
        return json.loads(path.read_text())


@pytest.fixture
def case(tmp_path):
    return RecoveryFixture(tmp_path)


def test_cleanup_recovery_runs_only_host_evaluation_preserves_failures_and_replays(case, monkeypatch):
    before = case.state()
    original = _program_source_records(case.program)
    called = []
    def evaluate(**kwargs):
        called.append(kwargs)
        assert kwargs["experiment_dir"] == case.program.parent
        assert kwargs["timeout_seconds"] == 900
        assert kwargs["sealed_config"].sandbox_gpus == "device=0"
        assert not (case.program / "evaluation_failure.json").exists()
        case.receipt()
        return 0
    monkeypatch.setattr(recovery, "run_harness_evaluation", evaluate)
    assert case.recover() == {"lease_id": case.lease["lease_id"], "evaluated": True, "committed": True, "status": "running"}
    after = case.state()
    assert after["valid_evaluations"] == 1
    assert after["transport_failures"] == 6
    assert after["failures"] == before["failures"]
    assert after["policy"] == before["policy"] and after["policy_hash"] == before["policy_hash"]
    assert after["leases"] == before["leases"]
    assert _program_source_records(case.program) == original
    assert case.recover(evidence_file=None)["evaluated"] is False
    assert len(called) == 1 and case.state()["valid_evaluations"] == 1
    assert len(case.state()["transport_recoveries"]) == 1
    ledger = case.ledger()
    run = ledger["runs"][0]
    assert run["status"] == "receipt_committed" and run["finished_at"]
    assert run["evidence"]["compute_usage"] == case.usage
    assert run["compute_usage"]["rpc_calls"] == 1979
    assert run["retained_source_evidence"]["original_evaluation_prelaunch_binding"] == "unavailable"
    assert len(state_store.tail_jsonl(state_store.attempts_history_path(case.results))) == 1
    # No budget refund: the very next transport error is still a hard stop.
    lease = case.controller.reserve_next(1, "device=0")
    case.controller.record_failure(lease["lease_id"], "worker_exit_1")
    assert case.state()["status"] == "incomplete"
    assert case.state()["transport_failures"] == 7


@pytest.mark.parametrize("correct", [True, False])
def test_existing_valid_receipt_never_evaluates_even_without_source_evidence(case, monkeypatch, correct):
    case.receipt(correct)
    monkeypatch.setattr(recovery, "run_harness_evaluation", lambda **kw: pytest.fail("receipt was evaluated twice"))
    assert case.recover(evidence_file=None)["evaluated"] is False
    assert case.state()["valid_evaluations"] == 1
    assert case.state()["outcomes"][case.lease["lease_id"]]["correct"] is correct
    assert (case.program / "evaluation_failure.json").exists()
    run = case.ledger()["runs"][0]
    assert run["evaluation_performed"] is False
    assert run["archived_files"]["run_0000/program/evaluation_failure.json"] == _hash(case.program / "evaluation_failure.json")


def test_recovery_error_never_fabricates_candidate_failure_or_refunds_budget(case, monkeypatch):
    before = case.state()
    def fail(**kwargs):
        case.write(case.program / "evaluation_failure.json", {**case.usage, "elapsed_seconds": 20})
        raise RuntimeError("host cleanup failed again")
    monkeypatch.setattr(recovery, "run_harness_evaluation", fail)
    with pytest.raises(RuntimeError, match="cleanup failed again"):
        case.recover()
    after = case.state()
    assert after["valid_evaluations"] == 0 and not after["outcomes"]
    assert after["failures"] == before["failures"] and after["transport_failures"] == 6
    assert after["status"] == "incomplete"
    assert not (case.program / "metrics.json").exists() and not (case.program / "correct.json").exists()
    run = case.ledger()["runs"][0]
    assert run["status"] == "failed" and run["error"]["type"] == "RuntimeError"
    assert run["compute_usage"]["elapsed_seconds"] == 20 and run["finished_at"]
    with pytest.raises(ValueError, match="already ran"):
        case.recover()


@pytest.mark.parametrize("lock_name", [COORDINATOR_LOCK, WORKER_LOCK])
def test_recovery_refuses_live_coordinator_or_worker_lock(case, monkeypatch, lock_name):
    monkeypatch.setattr(recovery, "run_harness_evaluation", lambda **kw: pytest.fail("locked lease evaluated"))
    root = case.results if lock_name == COORDINATOR_LOCK else case.worker
    with _exclusive_lock(root / lock_name):
        with pytest.raises(ValueError, match="already active"):
            case.recover()


def test_recovery_refuses_live_recorded_process(case, monkeypatch):
    case.write(case.worker / "portfolio_process.json", {"pid": os.getpid(), "status": "running",
               "process_identity": recovery.portfolio_process_identity(os.getpid())})
    with pytest.raises(ValueError, match="worker is live"):
        case.recover()
    assert case.state()["valid_evaluations"] == 0


@pytest.mark.parametrize("change", ["source", "parent", "lease", "task", "card", "receipt", "rubrics"])
def test_source_identity_and_invalid_receipt_fail_closed(case, monkeypatch, change):
    monkeypatch.setattr(recovery, "run_harness_evaluation", lambda **kw: pytest.fail("changed evidence evaluated"))
    if change == "source":
        (case.program / "src/program.py").write_text("print('different candidate')\n")
    elif change == "parent":
        (case.seed / "src/program.py").write_text("print('changed parent')\n")
    elif change == "lease":
        envelope = json.loads((case.worker / "portfolio_lease.json").read_text())
        envelope["lease"]["gpu_selector"] = "device=1"
        case.write(case.worker / "portfolio_lease.json", envelope)
    elif change == "task":
        (case.task / "evaluate.py").write_text("# changed evaluator\n")
    elif change == "card":
        case.bundle["idea_id"] = "different-root"
        case.write(case.program.parent.parent / "idea_bundle.json", case.bundle)
    elif change == "receipt":
        case.receipt()
        case.write(case.program / "metrics.json", {"combined_score": 999})
    elif change == "rubrics":
        case.write(case.program.parent / "active_rubrics.json", [{"id": "changed"}])
    with pytest.raises(ValueError):
        case.recover()
    assert not (case.results / recovery.RECOVERY_DIRECTORY).exists()


def test_cleanup_diagnostics_without_trusted_exception_do_not_authorize_recovery(case, monkeypatch):
    case.history["output_tail"] = "ValueError: candidate implementation is incorrect"
    (case.worker / "attempts_history.jsonl").write_text(json.dumps(case.history) + "\n")
    case.capture()
    monkeypatch.setattr(recovery, "run_harness_evaluation", lambda **kw: pytest.fail("untrusted failure evaluated"))
    with pytest.raises(ValueError, match="recoverable transport failure"):
        case.recover()
    assert case.state()["valid_evaluations"] == 0


def test_controller_never_resumes_with_an_unresolved_failed_lease(case):
    assert case.controller.resume_after_transport_recovery() is False
    assert case.controller.status == "incomplete"
    assert case.controller.reserve_next(1, "device=0") is None


def test_controller_requires_receipt_revalidation_and_transport_stop_reason(case):
    case.receipt()
    outcome = recovery._receipt_for_lease(case.lease, case.task, case.controller.context(case.lease))
    case.controller.record_receipt(case.lease["lease_id"], outcome)
    (case.program / "correct.json").write_text('{"correct":false}')
    with pytest.raises(ValueError, match="revalidated receipts"):
        case.controller.resume_after_transport_recovery()
    assert case.controller.status == "incomplete"
    case.controller.stop("manual_scientific_review_required")
    with pytest.raises(ValueError, match="recoverable transport failure"):
        case.controller.resume_after_transport_recovery()


def test_recovery_preserves_complete_frozen_plan_cards_and_prior_outcomes(tmp_path, monkeypatch):
    case = RecoveryFixture(tmp_path, planned=True)
    before = case.state()
    def evaluate(**kwargs):
        case.receipt()
        return 0
    monkeypatch.setattr(recovery, "run_harness_evaluation", evaluate)
    assert case.recover()["committed"] is True
    after = case.state()
    assert before["valid_evaluations"] == 1 and after["valid_evaluations"] == 2
    assert all(after["outcomes"][key] == value for key, value in before["outcomes"].items())
    for key in ("campaign_plan", "plan_sha256", "policy", "policy_hash", "failures", "transport_failures", "leases"):
        assert after[key] == before[key]
    outcome = after["outcomes"][case.lease["lease_id"]]
    assert outcome["plan_sha256"] == before["plan_sha256"]
    assert outcome["card_sha256"] == case.lease["card_sha256"]
    assert case.recover()["committed"] is False
    assert case.state()["valid_evaluations"] == 2


def test_explicit_evaluation_only_stop_can_resume_without_refunding_its_failure(case, monkeypatch):
    case.controller.stop("nonretryable_failure:evaluation_only_recovery_required")
    case.receipt()
    monkeypatch.setattr(recovery, "run_harness_evaluation", lambda **kw: pytest.fail("receipt was rerun"))
    assert case.recover(evidence_file=None)["status"] == "running"
    assert case.state()["transport_failures"] == 6
