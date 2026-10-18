"""Run native adaptive campaigns and recover durable work after abrupt interruption."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from src.investigation import campaign_loop
from src.investigation.adaptive_controller import AdaptiveController, AdaptiveInvestigationConfig
from src.investigation.config import InvestigationConfig
from src.investigation.files import find_best_program_in_experiment
from src.investigation.harness_eval import run_harness_evaluation
from src.investigation.models import IdeaGenerationResult, RunAttemptResult
from src.orchestrator import state as state_store
from tests.program_testutils import make_program_dir

ROOT = Path(__file__).resolve().parents[2]


class SimulatedCrash(RuntimeError):
    """An abrupt stop after durable writes, with no graceful campaign shutdown."""


class NativeCampaign:
    def __init__(self, root: Path):
        self.root = root
        self.task = root / "task"
        self.seed = self.task / "program_0"
        make_program_dir(self.seed, content="print('baseline')\n")
        (self.task / "task.md").write_text("# Goal\nCompare development candidates.\n")
        (self.task / "protocol.json").write_text('{"fixed_score": 0.75}')
        (self.task / "native_evaluation_inputs.json").write_text(json.dumps({
            "schema_version": 1, "required": ["protocol.json"],
        }))
        (self.task / "evaluate.py").write_text(
            """import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--program_path')
parser.add_argument('--results_dir')
args = parser.parse_args()
task = Path(os.environ['TASK_SOURCE_DIR'])
counter = task / 'execution_count.txt'
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
score = json.loads((task / 'protocol.json').read_text())['fixed_score']
correct = 'FAIL_CORRECTNESS = True' not in Path(args.program_path).read_text()
out = Path(args.results_dir)
(out / 'metrics.json').write_text(json.dumps({'combined_score': score}))
(out / 'correct.json').write_text(json.dumps({'correct': correct}))
"""
        )
        self.results = root / "run"
        self.results.mkdir()
        state_store.ensure_histories(self.results)
        self.cfg = InvestigationConfig()
        self.cfg.agent.name = "codex"
        self.cfg.agent.model = "fixture-model"
        self.cfg.execution.max_idea_cycles = 8
        self.cfg.execution.harness_evaluated_attempts = True
        self.cfg.execution.prune_every_n_cycles = 0
        self.cfg.execution.agent_transient_backoff_seconds = 0
        self.cfg.execution.harness_evaluation_timeout_seconds = 10
        self.cfg.adaptive_investigation = AdaptiveInvestigationConfig(
            enabled=True, min_tuning_evals=1, min_creative_evals=1,
            min_refinement_evals=1, stagnation_evals=1, replication_evals=1,
            max_valid_evals=5,
        )
        self.cfg.paper.enabled = False
        self.generated: list[int] = []
        self.coding_calls: list[str] = []
        self.events: list[tuple[str, dict]] = []
        self.crash_after_evaluation = False
        self.fail_first_candidate = False
        self.transport_fail_once = False
        self.transport_snapshot: dict | None = None

    def generate(self, **kwargs):
        cycle = kwargs["cycle_index"]
        self.generated.append(cycle)
        workspace = kwargs["output_dir"]
        workspace.mkdir(parents=True)
        bundle = {
            "idea_id": f"idea_{cycle}", "title": f"Candidate {cycle}",
            "hypothesis": "A candidate can pass the fixed development protocol.",
            "implementation_plan": ["Write the candidate source."],
            "target_rubrics": [{
                "id": "candidate_source", "description": "Candidate source prints a value.",
                "type": "implementation", "verification": {
                    "type": "source_contains", "patterns": ["print("],
                },
            }],
            "notes_for_agent": "Unit-test fixture; no model dispatch.",
        }
        (workspace / "idea_bundle.json").write_text(json.dumps(bundle))
        (workspace / "target_rubrics.json").write_text(json.dumps(bundle["target_rubrics"]))
        return IdeaGenerationResult("Fixture idea drafted.", 0, workspace)

    def attempt(self, **kwargs):
        self.coding_calls.append(kwargs["attempt_id"])
        workspace = self.results / "ideas" / kwargs["idea_dir_name"] / kwargs["attempt_dir_name"]
        if (workspace / "program_1").exists():
            # Emulate the real wrapper's occupied-workspace refusal. This fixture
            # must not silently overwrite a candidate whose receipt is invalid.
            return RunAttemptResult("Existing candidate requires explicit recovery.", 1, workspace)
        if self.transport_fail_once:
            self.transport_fail_once = False
            workspace.mkdir(parents=True)
            return RunAttemptResult("Provider unavailable.", 1, workspace,
                                    agent_failure_kind="transient_provider")
        if kwargs["attempt_id"].endswith("_02"):
            state = json.loads((self.results / "adaptive_investigation.json").read_text())
            self.transport_snapshot = {
                "transport_failures": state["transport_failures"],
                "valid_evaluations": state["valid_evaluations"],
            }
        source = "print('candidate')\n"
        if self.fail_first_candidate:
            self.fail_first_candidate = False
            source += "FAIL_CORRECTNESS = True\n"
        make_program_dir(workspace / "program_1", content=source)
        shutil.copyfile(self.task / "evaluate.py", workspace / "evaluate.py")
        idea_id = json.loads(kwargs["idea_file"].read_text())["idea_id"]
        code = run_harness_evaluation(
            experiment_dir=workspace, attempt_id=kwargs["attempt_id"], idea_id=idea_id,
            rubrics_path=kwargs["rubrics_file"], python_bin=sys.executable,
            global_memory_md=self.results / "memory.md",
            global_memory_jsonl=self.results / "memory.jsonl", timeout_seconds=10,
            environment=dict(os.environ, TASK_SOURCE_DIR=str(self.task)),
        )
        if self.crash_after_evaluation:
            self.crash_after_evaluation = False
            raise SimulatedCrash("after native evaluation before returning attempt")
        return RunAttemptResult("Fixture candidate and host evaluation completed.", code, workspace)

    def run(self, *, resume=False, checkpoint=lambda **_kwargs: None):
        self.cfg.execution.resume = resume
        outer = state_store.load_or_init_outer_state(self.results, self.seed, resume)
        best = state_store.load_json(self.results / state_store.BEST_MANIFEST_FILE) if resume else None
        best = best or {"program_dir": str(self.seed), "combined_score": 0.5, "correct": True}
        return campaign_loop.run_cycles_and_finalize(
            cfg=self.cfg, task_dir=self.task, results_dir=self.results, global_memory_dir=self.results,
            instruct_path=ROOT / "src/prompts/harness_edit.md",
            idea_instruct_path=ROOT / "src/prompts/idea.md", paper_instruct_path=self.task / "task.md",
            paper_contributions_path=None, containerize=False, container_image="", container_network="",
            container_auth_root=self.root, container_env_file=None, container_session_name=None,
            container_mount_root=None, outer_state=outer, best_manifest=best,
            best_program_dir=Path(best["program_dir"]), start_cycle=1,
            max_program_evals_per_attempt=1, main_started_at=0,
            log_event_fn=lambda name, **data: self.events.append((name, data)),
            log_step_start_fn=lambda *args, **kwargs: 0,
            log_step_end_fn=lambda *args, **kwargs: None, format_elapsed_fn=lambda _: "fixture",
            run_idea_generation_fn=self.generate, run_single_attempt_fn=self.attempt,
            run_paper_generation_fn=lambda **kwargs: None,
            reconcile_legacy_idea_local_memory_fn=lambda **kwargs: None,
            find_best_program_in_experiment_fn=find_best_program_in_experiment,
            read_json_fn=state_store.load_json, utc_now_fn=lambda: "2026-10-18T00:00:00Z",
            checkpoint_fn=checkpoint,
        )

    def assert_complete_once(self, *, transport_retry=False):
        assert self.generated == [1, 3, 4]
        assert self.coding_calls == [
            "idea_1_attempt_01", *(["idea_1_attempt_02"] if transport_retry else []),
            "idea_3_attempt_01", "idea_4_attempt_01",
        ]
        assert (self.task / "execution_count.txt").read_text() == "5"
        adaptive = json.loads((self.results / "adaptive_investigation.json").read_text())
        assert adaptive["status"] == "ready_for_validation"
        assert adaptive["completed_cycles"] == [1, 2, 3, 4, 5]
        assert adaptive["valid_evaluations"] == 5
        assert adaptive["transport_failures"] == int(transport_retry)
        assert len(adaptive["completed_outcome_ids"]) == 5
        assert adaptive["pending"] is None
        rows = state_store.tail_jsonl(state_store.attempts_history_path(self.results), max_lines=20)
        assert len(rows) == 5 + int(transport_retry)
        rows = [row for row in rows if not row.get("retry_scheduled")]
        assert [row["cycle_index"] for row in rows] == [1, 2, 3, 4, 5]
        assert all(row["scientific_evaluation"] for row in rows)
        assert all(row["evaluation_performed"] for row in rows)
        assert [row["adaptive_action"] for row in rows].count("replicate_frozen_candidate") == 2
        assert all(row["attempt_status"] == "pass" for row in rows if row["adaptive_action"] == "replicate_frozen_candidate")
        assert all(row["agent"] == "host" and row["agent_model"] is None for row in rows if row["adaptive_action"] == "replicate_frozen_candidate")
        ideas = state_store.tail_jsonl(state_store.ideas_history_path(self.results), max_lines=20)
        assert len(ideas) == 5  # Three generated ideas and two host-owned replication plans.
        outer = state_store.load_json(self.results / state_store.OUTER_STATE_FILE)
        assert outer["current_cycle"] == 5
        assert "pending_cycle" not in outer
        handoff = json.loads((self.results / "validation_handoff.json").read_text())
        assert handoff["status"] == "ready_for_validation"
        assert handoff["submission_ready"] is False
        assert handoff["independent_confirmation_complete"] is False
        assert set(handoff["development_aggregates"]) == {"baseline_tuning", "creative_research"}
        for aggregate in handoff["development_aggregates"].values():
            assert aggregate["n"] == 1
            assert aggregate["mean"] == 0.75
            assert aggregate["std"] == 0
            assert aggregate["all_correct"] is True
            assert aggregate["source_hash"]
            assert aggregate["independent_confirmation"] is False


def test_full_native_campaign_uses_three_coding_actions_and_two_host_replications(tmp_path):
    campaign = NativeCampaign(tmp_path)
    assert campaign.run() == 0
    campaign.assert_complete_once()


def test_resume_after_idea_checkpoint_does_not_generate_idea_again(tmp_path):
    campaign = NativeCampaign(tmp_path)

    def crash(**checkpoint):
        if checkpoint.get("event") == "idea_drafted":
            raise SimulatedCrash("after durable idea checkpoint")

    with pytest.raises(SimulatedCrash, match="idea checkpoint"):
        campaign.run(checkpoint=crash)
    assert campaign.generated == [1]
    assert campaign.coding_calls == []
    assert campaign.run(resume=True) == 0
    campaign.assert_complete_once()
    assert any(name == "idea_resumed" for name, _ in campaign.events)


def test_resume_after_native_evaluation_does_not_repeat_coding_or_evaluation(tmp_path):
    campaign = NativeCampaign(tmp_path)
    campaign.crash_after_evaluation = True
    with pytest.raises(SimulatedCrash, match="native evaluation"):
        campaign.run()
    assert (campaign.task / "execution_count.txt").read_text() == "1"
    assert campaign.run(resume=True) == 0
    campaign.assert_complete_once()
    assert any(name == "host_evaluation_recovered" for name, _ in campaign.events)


def test_resume_after_controller_outcome_repairs_outer_journal_once(tmp_path, monkeypatch):
    campaign = NativeCampaign(tmp_path)
    record = AdaptiveController.record_outcome

    def record_then_crash(controller, *args, **kwargs):
        recorded = record(controller, *args, **kwargs)
        if recorded:
            raise SimulatedCrash("after controller outcome before outer journal")
        return recorded

    with monkeypatch.context() as patch:
        patch.setattr(AdaptiveController, "record_outcome", record_then_crash)
        with pytest.raises(SimulatedCrash, match="before outer journal"):
            campaign.run()
    adaptive = json.loads((campaign.results / "adaptive_investigation.json").read_text())
    assert adaptive["valid_evaluations"] == 1
    assert adaptive["pending"]["phase"] == "outcome_recorded"
    assert state_store.tail_jsonl(state_store.attempts_history_path(campaign.results), max_lines=10) == []
    assert campaign.run(resume=True) == 0
    campaign.assert_complete_once()


def test_valid_negative_result_counts_without_promoting_wrong_candidate(tmp_path):
    campaign = NativeCampaign(tmp_path)
    campaign.fail_first_candidate = True

    def stop_after_first_result(**checkpoint):
        if checkpoint.get("event") == "evaluation_complete" and checkpoint.get("cycle") == 1:
            raise SimulatedCrash("inspect negative result")

    with pytest.raises(SimulatedCrash, match="negative result"):
        campaign.run(checkpoint=stop_after_first_result)
    adaptive = json.loads((campaign.results / "adaptive_investigation.json").read_text())
    assert adaptive["valid_evaluations"] == 1
    assert adaptive["stage_counts"] == {"baseline_tuning/optimization": 1}
    assert adaptive["best"]["score"] == 0.5
    assert adaptive["best"]["program_dir"] == str(campaign.seed)
    assert adaptive["outcomes"]["idea_1_attempt_01"]["outcome"]["correct"] is False
    assert not (campaign.results / state_store.BEST_MANIFEST_FILE).exists()
    first = state_store.tail_jsonl(state_store.attempts_history_path(campaign.results), max_lines=10)[0]
    assert first["scientific_evaluation"] is True
    assert first["best_updated"] is False
    assert first["attempt_status"] == "fail"
    assert campaign.run(resume=True) == 0
    campaign.assert_complete_once()


def test_transport_retry_spends_no_scientific_evaluation_budget(tmp_path):
    campaign = NativeCampaign(tmp_path)
    campaign.transport_fail_once = True
    assert campaign.run() == 0
    campaign.assert_complete_once(transport_retry=True)
    assert campaign.transport_snapshot == {"transport_failures": 1, "valid_evaluations": 0}
    rows = state_store.tail_jsonl(state_store.attempts_history_path(campaign.results), max_lines=20)
    assert rows[0]["retry_scheduled"] is True
    assert rows[0]["evaluation_performed"] is False
    assert rows[0]["best_updated"] is False


@pytest.mark.parametrize("tamper", ["manifest", "metrics"])
def test_invalid_receipt_is_not_recovered_or_promoted_after_crash(tmp_path, tamper):
    campaign = NativeCampaign(tmp_path)
    campaign.cfg.execution.max_idea_cycles = 1
    campaign.crash_after_evaluation = True
    with pytest.raises(SimulatedCrash, match="native evaluation"):
        campaign.run()
    candidate = next((campaign.results / "ideas").glob("*/01_attempt/program_1"))
    before_source = (candidate / "src/program.py").read_bytes()
    if tamper == "manifest":
        (candidate / "evaluation_manifest.json").write_text("{malformed")
    else:
        (candidate / "metrics.json").write_text('{"combined_score": 999}')
    assert campaign.run(resume=True) == 2
    assert campaign.generated == [1]
    assert (campaign.task / "execution_count.txt").read_text() == "1"
    assert (candidate / "src/program.py").read_bytes() == before_source
    adaptive = json.loads((campaign.results / "adaptive_investigation.json").read_text())
    assert adaptive["valid_evaluations"] == 0
    assert adaptive["status"] == "incomplete"
    assert adaptive["best"]["score"] == 0.5
    assert not (campaign.results / state_store.BEST_MANIFEST_FILE).exists()
    rows = state_store.tail_jsonl(state_store.attempts_history_path(campaign.results), max_lines=10)
    assert len(rows) == 1
    assert rows[0]["best_updated"] is False
    assert rows[0]["scientific_evaluation"] is False
    assert rows[0]["evaluation_performed"] is False
    assert not any(name == "host_evaluation_recovered" for name, _ in campaign.events)


def test_resume_cannot_silently_disable_persisted_adaptive_policy(tmp_path):
    campaign = NativeCampaign(tmp_path)

    def crash(**checkpoint):
        if checkpoint.get("event") == "idea_drafted":
            raise SimulatedCrash("after idea")

    with pytest.raises(SimulatedCrash):
        campaign.run(checkpoint=crash)
    campaign.cfg.adaptive_investigation.enabled = False
    assert campaign.run(resume=True) == 2
    assert campaign.generated == [1]
    assert campaign.coding_calls == []
    assert campaign.events[-1] == ("campaign_failed", {"reason": "adaptive_state_requires_original_policy"})


def test_larger_schedule_resumes_incomplete_campaign_without_repeating_work(tmp_path):
    campaign = NativeCampaign(tmp_path)
    campaign.cfg.execution.max_idea_cycles = 1
    assert campaign.run() == 2
    adaptive = json.loads((campaign.results / "adaptive_investigation.json").read_text())
    assert adaptive["status"] == "incomplete"
    assert adaptive["transition_reason"] == "scheduled_cycle_budget_exhausted"
    assert adaptive["exhausted_cycle_limit"] == 1
    assert adaptive["valid_evaluations"] == 1
    # The same schedule remains incomplete. Only an explicit increase reopens it.
    assert campaign.run(resume=True) == 2
    assert campaign.coding_calls == ["idea_1_attempt_01"]
    campaign.cfg.execution.max_idea_cycles = 8
    assert campaign.run(resume=True) == 0
    campaign.assert_complete_once()
    adaptive = json.loads((campaign.results / "adaptive_investigation.json").read_text())
    assert any(row["reason"] == "scheduled_cycle_budget_extended" for row in adaptive["transition_journal"])
