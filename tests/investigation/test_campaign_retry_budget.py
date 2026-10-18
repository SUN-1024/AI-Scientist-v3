"""Transport recovery and resolved prompt budgets, without external API calls."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from src.investigation import cli, campaign_loop
from src.investigation.campaign_budget import BUDGET_FILE, render_campaign_budget, write_campaign_budget
from src.investigation.campaign_retry import read_agent_failure, report_agent_failure
from src.investigation.config import InvestigationConfig
from src.investigation.files import find_best_program_in_experiment
from src.investigation.models import IdeaGenerationResult, RunAttemptResult
from src.orchestrator import state as state_store
from src.orchestrator.paper_data import build_paper_data
from tests.e2e.test_agent_execution_pipeline import _write_fake_agent, _write_task_dir
from tests.investigation.test_campaign_ideas import _IDEA_BUNDLE

ROOT = Path(__file__).resolve().parents[2]


def _campaign(root, attempt, *, cycles=1, start=1, resume=False):
    task = _write_task_dir(root)
    (task / "task.md").write_text("# Goal\nFirst try two rounds. Keep exactly 160 training steps and never read hidden test data.\n")
    (task / "evaluate.py").write_text("""import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--program_path'); p.add_argument('--results_dir')
a = p.parse_args(); out = Path(a.results_dir)
(out / 'metrics.json').write_text(json.dumps({'combined_score': 2.71}))
(out / 'correct.json').write_text(json.dumps({'correct': True}))
""")
    results = root / "run"
    results.mkdir()
    cfg = InvestigationConfig()
    cfg.agent.name = "codex"
    cfg.agent.model = "fixture-model"
    cfg.agent.reasoning_effort = "high"
    cfg.execution.max_idea_cycles = cycles
    cfg.execution.search_strategy = "tree"
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.agent_transient_backoff_seconds = 0
    cfg.execution.resume = resume
    cfg.paper.enabled = False
    generated = []
    events = []

    def generate(**kwargs):
        generated.append(kwargs["cycle_index"])
        workspace = kwargs["output_dir"]
        workspace.mkdir(parents=True)
        bundle = dict(_IDEA_BUNDLE, idea_id=f"idea_{kwargs['cycle_index']}")
        bundle["target_rubrics"] = [{"id": "does_the_thing", "description": "Uses candidate print", "type": "implementation", "verification": {"type": "source_contains", "patterns": ["print("]}}]
        (workspace / "idea_bundle.json").write_text(json.dumps(bundle))
        (workspace / "target_rubrics.json").write_text(json.dumps(bundle["target_rubrics"]))
        return IdeaGenerationResult("fixture", 0, workspace)

    code = campaign_loop.run_cycles_and_finalize(
        cfg=cfg, task_dir=task, results_dir=results, global_memory_dir=results,
        instruct_path=ROOT / "src/prompts/harness_edit.md", idea_instruct_path=ROOT / "src/prompts/idea.md",
        paper_instruct_path=task / "task.md", paper_contributions_path=None,
        containerize=False, container_image="", container_network="", container_auth_root=root,
        container_env_file=None, container_session_name=None, container_mount_root=None,
        outer_state={"current_cycle": start - 1},
        best_manifest={"program_dir": str(task / "program_0"), "combined_score": 1, "correct": True},
        best_program_dir=task / "program_0", start_cycle=start, max_program_evals_per_attempt=1,
        main_started_at=0, log_event_fn=lambda name, **data: events.append((name, data)),
        log_step_start_fn=lambda *a, **k: 0, log_step_end_fn=lambda *a, **k: None,
        format_elapsed_fn=lambda _: "fixture", run_idea_generation_fn=generate,
        run_single_attempt_fn=attempt, run_paper_generation_fn=lambda **k: None,
        reconcile_legacy_idea_local_memory_fn=lambda **k: None,
        find_best_program_in_experiment_fn=find_best_program_in_experiment,
        read_json_fn=state_store.load_json, utc_now_fn=lambda: "2026-10-18T00:00:00Z",
    )
    return code, results, generated, events


def test_real_wrapper_retries_same_idea_and_preserves_tree_and_paper_evidence(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_agent(bin_dir, "headless")
    script = bin_dir / "headless"
    original = script.read_text().split("cat > program_1/metrics.json", 1)[0] + 'echo "fixture wrote candidate source"\n'
    # The first invocation fails in transport. The second fixture writes one candidate.
    prefix = '''
if [ -n "${AGENTIC_CODING_FAILURE_FD:-}" ]; then exit 99; fi
if [ ! -f "$FIXTURE_COUNTER" ]; then
  touch "$FIXTURE_COUNTER"
  echo '{"type":"turn.failed","error":{"message":"provider returned 502 Bad Gateway"}}'
  exit 1
fi
'''
    script.write_text(original.replace('set -euo pipefail\n', 'set -euo pipefail\n' + prefix, 1))
    monkeypatch.setenv("HEADLESS_BIN", str(script))
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FIXTURE_COUNTER", str(tmp_path / "counter"))
    monkeypatch.setenv("WRAPPER_PYTHON", sys.executable)
    monkeypatch.setenv("RUBRIC_ID", "does_the_thing")
    code, results, generated, events = _campaign(tmp_path, cli.run_single_attempt)
    assert code == 0
    assert generated == [1]
    rows = state_store.tail_jsonl(state_store.attempts_history_path(results), max_lines=10)
    assert [row["attempt_number"] for row in rows] == [1, 2]
    assert rows[0]["failure_category"] == "transient_provider"
    assert rows[0]["evaluation_performed"] is False
    assert rows[1]["attempt_status"] == "pass"
    assert rows[0]["idea_id"] == rows[1]["idea_id"]
    for row, folder in zip(rows, ["01_attempt", "02_attempt"]):
        assert Path(row["experiment_dir"]).name == folder
        assert Path(row["wrapper_output_log"]).is_file()
    tree = json.loads((results / "search_tree.json").read_text())
    assert set(tree["nodes"]) == {"root", "cycle_0001"}
    best = state_store.load_json(results / state_store.BEST_MANIFEST_FILE)
    assert "02_attempt/program_1" in best["program_dir"]
    assert len(build_paper_data(results)["all_programs"]) == 1
    from src.investigation.parallel_helpers import _best_manifest_from_worker_round_dir
    recovered = _best_manifest_from_worker_round_dir(results, seed_program_dir=tmp_path / "task/program_0")
    assert recovered is not None
    assert recovered[0]["program_dir"] == best["program_dir"]
    assert len([name for name, _ in events if name == "coding_agent_retry_scheduled"]) == 1
    prompt = (Path(rows[1]["experiment_dir"]) / "prompt.md").read_text()
    assert '"global_total": 1' in prompt
    assert "overrides stale campaign round counts" in prompt


@pytest.mark.parametrize("failure,receipt,evaluated,expected", [
    ("invalid API key", None, False, 1),
    ("provider 502 service unavailable", None, False, 1),  # untrusted stdout alone is insufficient
    ("provider 502 service unavailable", "transient_provider", True, 1),
    ("provider 502 service unavailable", "transient_provider", False, 3),
])
def test_no_retry_of_permanent_or_scientific_failures_and_bounded_transient_exhaustion(
    tmp_path, failure, receipt, evaluated, expected,
):
    calls = []

    def attempt(**kwargs):
        calls.append(kwargs)
        folder = kwargs["results_dir"] / "ideas" / kwargs["idea_dir_name"] / kwargs["attempt_dir_name"]
        folder.mkdir(parents=True)
        if evaluated:
            (folder / "program_1").mkdir()
            (folder / "program_1/correct.json").write_text('{"correct": false}')
        return RunAttemptResult(failure, 1, folder, agent_failure_kind=receipt)

    code, results, generated, _ = _campaign(tmp_path, attempt)
    assert code == 0  # Existing campaign policy retains failed experiments and continues.
    assert len(calls) == expected
    assert generated == [1]
    assert len(state_store.tail_jsonl(state_store.attempts_history_path(results), max_lines=10)) == expected
    tree = json.loads((results / "search_tree.json").read_text())
    assert set(tree["nodes"]) == {"root", "cycle_0001"}


def test_classifier_private_receipt_rejects_evaluation_and_permanent_failure(tmp_path):
    log = tmp_path / "agent.log"
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    for message, evaluated, expected in [
        ('{"type":"turn.failed","error":{"message":"502 Bad Gateway"}}', False, "transient_provider"),
        ('{"type":"turn.failed","error":{"message":"unauthorized 502 Bad Gateway"}}', False, None),
        ('{"type":"turn.failed","error":{"message":"502 Bad Gateway"}}', True, None),
    ]:
        log.write_text(message)
        if evaluated:
            (experiment / "rubric_status.json").write_text("{}")
        with tempfile.TemporaryFile() as channel:
            report_agent_failure(channel.fileno(), "attempt_1", log, experiment)
            assert read_agent_failure(channel.fileno(), "attempt_1") == expected


def test_resolved_budget_refreshes_on_resume_and_does_not_multiply_parallel_cycles(tmp_path):
    cfg = InvestigationConfig()
    cfg.execution.max_idea_cycles = 12
    cfg.execution.resume = True
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.search_strategy = "tree"
    cfg.execution.search_width = 6
    cfg.agent.reasoning_effort = "high"
    (tmp_path / BUDGET_FILE).write_text('{"stale": "two rounds"}')
    state_store.append_jsonl(state_store.attempts_history_path(tmp_path), {"evaluation_performed": True})
    state_store.append_jsonl(state_store.attempts_history_path(tmp_path), {"retry_scheduled": True, "evaluation_performed": False})
    budget = write_campaign_budget(tmp_path, cfg, cycle=5, max_program_evals_per_attempt=4, outer_state={"current_cycle": 4})
    assert budget["cycle"]["local_remaining_including_current"] == 8
    assert budget["cycle"]["persisted_current_cycle"] == 4
    assert budget["observed_local_history"]["evaluated_attempts"] == 1
    assert budget["evaluation"]["max_evaluator_calls_per_attempt"] == 1
    text = render_campaign_budget(tmp_path, 5)
    assert "training, compute" in text
    assert "not an adaptive experiment manager" in text
    with pytest.raises(ValueError, match="does not match"):
        render_campaign_budget(tmp_path, 4)
    cfg.execution.max_idea_cycles = 6
    cfg.execution.total_cycles = 24
    cfg.execution.cycle_offset = 12
    cfg.execution.campaign_workers = 4
    budget = write_campaign_budget(tmp_path, cfg, cycle=2, max_program_evals_per_attempt=4, outer_state={"current_cycle": 1})
    assert budget["cycle"]["global_total"] == 24
    assert budget["cycle"]["global_position"] == 14
    assert budget["cycle"]["global_remaining_including_current"] is None
    assert budget["workers"]["campaign_workers"] == 4


def test_resumed_campaign_generates_current_budget_before_next_idea(tmp_path):
    code, results, generated, _ = _campaign(
        tmp_path, lambda **kwargs: RunAttemptResult("fixture", 0, None),
        cycles=5, start=5, resume=True,
    )
    assert code == 0
    assert generated == [5]
    budget = json.loads((results / BUDGET_FILE).read_text())
    assert budget["cycle"]["global_total"] == 5
    assert budget["cycle"]["local_remaining_including_current"] == 1
    assert budget["cycle"]["resume"] is True
    assert budget["cycle"]["persisted_current_cycle"] == 4


@pytest.mark.parametrize("exit_code", [0, 1])
def test_harness_campaign_does_not_promote_agent_metrics_without_receipt(tmp_path, exit_code):
    def forged_attempt(**kwargs):
        folder = kwargs["results_dir"] / "ideas" / kwargs["idea_dir_name"] / kwargs["attempt_dir_name"]
        from tests.program_testutils import make_program_dir
        make_program_dir(folder / "program_1", content='print("ok")\n')
        (folder / "program_1/metrics.json").write_text('{"combined_score": 999}')
        (folder / "program_1/correct.json").write_text('{"correct": true}')
        (folder / "rubric_status.json").write_text(json.dumps({
            "attempt_id": kwargs["attempt_id"], "idea_id": "idea_1",
            "overall_pass": True,
            "rubrics": [{"id": "does_the_thing", "pass": True}, {"id": "incumbent_score_improvement", "pass": True}],
            "metrics": {"combined_score": 999, "correct": True},
        }))
        return RunAttemptResult("agent supplied fake metrics", exit_code, folder)

    code, results, _, events = _campaign(tmp_path, forged_attempt)
    assert code == 0
    history = state_store.tail_jsonl(state_store.attempts_history_path(results), max_lines=10)
    assert history[0]["best_updated"] is False
    assert history[0]["evaluation_performed"] is False
    assert "metrics" not in history[0]
    assert not (results / state_store.BEST_MANIFEST_FILE).exists()
    tree = json.loads((results / "search_tree.json").read_text())
    assert tree["nodes"]["cycle_0001"]["program_dir"] is None
    assert any(name == "candidate_score_rejected" for name, _ in events)


def test_nested_attempt_guard_checks_real_parent_directory(tmp_path):
    from src.investigation.evaluation_artifact_cleanup import ExperimentDirectoryGuard
    results = tmp_path / "results"
    guard = ExperimentDirectoryGuard.prepare(results, "idea", "01_attempt")
    try:
        guard.experiment_dir.mkdir()
        descriptor = guard.open_experiment()
        assert descriptor is not None
        os.close(descriptor)
        parent = results / "ideas/idea"
        parent.rename(results / "ideas/moved")
        parent.symlink_to(results / "ideas/moved", target_is_directory=True)
        with pytest.raises(RuntimeError, match="Directory changed"):
            guard.open_experiment()
    finally:
        guard.close()
