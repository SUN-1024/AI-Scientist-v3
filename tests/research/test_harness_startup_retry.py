from __future__ import annotations

import fcntl
import json
import os
from unittest.mock import Mock

import pytest

from src.investigation.task_alignment import task_tree_fingerprint
from src.pipeline.cli import _prepare_investigation_workspace, load_pipeline_config
from src.pipeline.glue import prepare_run_task
from src.pipeline.paths import derive_paths
from src.pipeline.resume_config import write_resume_policy
from src.pipeline.task_snapshot import task_snapshot_is_complete
from src.research import cli, harness
from src.research.harness_state import read_json, sha256
from tests.research.test_unified_research import make_task


def _failed_startup(tmp_path, monkeypatch):
    task = make_task(tmp_path / "source")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())

    def fail(argv, *, event_callback):
        cfg = load_pipeline_config(argv)
        paths = derive_paths(cfg)
        prepare_run_task(paths)
        _prepare_investigation_workspace(paths)
        write_resume_policy(paths.run_dir, max_idea_cycles=cfg.investigation.max_idea_cycles)
        (paths.run_dir / ".investigation_completion.json").write_text('{"schema_version":1,"status":"running"}')
        (paths.run_dir / ".task.runtime-build.lock").touch(mode=0o600)
        (paths.investigation_workspace_dir / ".task-seal-empty").mkdir()
        event_callback("phase_start", phase_name="investigation")
        event = {"event": "task_runtime_failed", "source": "wrapper", "worker_index": 0,
                 "session_id": "startup", "phase": "task", "timestamp": harness.now(), "fields": {
                     "task_dir": str(paths.task_dir), "result_kind": "infrastructure_failure",
                     "failure_stage": "runtime_preparation", "failure_category": "runtime_bootstrap",
                     "task_fingerprint": task_tree_fingerprint(paths.task_dir)}}
        (paths.investigation_workspace_dir / "events.jsonl").write_text(json.dumps(event) + "\n")
        event_callback("phase_end", phase_name="investigation", status="failed", return_code=2)
        terminal = {"event": "pipeline_end", "source": "pipeline", "fields": {"status": "failed", "return_code": 2}}
        (paths.run_dir / "pipeline_events.jsonl").write_text(json.dumps(terminal) + "\n")
        event_callback("pipeline_end", status="failed", return_code=2)
        return 2

    monkeypatch.setattr("src.pipeline.cli.main", fail)
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--offline",
                     "--quality-target", "development", "execution.phases=investigation",
                     "container.enabled=false"]) == 2
    return workspace


def _plan(workspace):
    from src.pipeline.resume_config import read_run_idea_cycle_budget
    return harness._startup_retry_plan(workspace, read_json(workspace / "harness_state.json"),
                                       cycle_budget=read_run_idea_cycle_budget(workspace / "run"))


def _baseline_initialization_failure(workspace):
    path = workspace / "run/investigation/events.jsonl"
    failure = json.loads(path.read_text())
    failure["fields"].update(failure_stage="baseline_validation", failure_category="evaluation_failed")
    steps = []
    for step in ("runtime_capacity", "runtime_create", "tooling_build", "tooling_install", "dependency_install", "baseline_evaluation"):
        steps.append({**failure, "event": "preparation_step", "phase": None,
                      "fields": {"step": step, "state": "started"}})
        if step == "dependency_install":
            steps.append({**failure, "event": "heartbeat", "phase": None,
                          "fields": {"step": step, "state": "started"}})
        steps.append({**failure, "event": "preparation_step", "phase": None,
                      "fields": {"step": step, "state": "failed" if step == "baseline_evaluation" else "completed"}})
    events = [*steps, failure]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path, events


def test_failed_initialization_baseline_retries_without_claiming_zero_execution(tmp_path, monkeypatch, capsys):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path, _ = _baseline_initialization_failure(workspace)
    plan = _plan(workspace)
    assert plan["failure_classification"] == "baseline_validation_infrastructure_failure"
    assert plan["scientific_result_status"] == "not_retained_or_verified"
    original = path.read_bytes()
    capsys.readouterr()
    assert cli.main(["harness", "--resume", str(workspace), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["initial_runtime_retry"] is True
    state = read_json(workspace / "harness_state.json")
    harness._archive_startup_failure(workspace, state, plan)
    archive = workspace / state["startup_recoveries"][-1]["archive"]
    assert (archive / "investigation/events.jsonl").read_bytes() == original
    assert read_json(archive / "recovery.json")["scientific_result_status"] == "not_retained_or_verified"
    assert task_snapshot_is_complete(workspace / "run/task")


@pytest.mark.parametrize("mutation", ["authored_failure", "baseline_completed", "no_baseline_start", "missing_build_step", "reordered_steps", "unbound_heartbeat", "repeat_baseline", "final_step"])
def test_baseline_initialization_retry_requires_exact_host_step_sequence(tmp_path, monkeypatch, mutation):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path, events = _baseline_initialization_failure(workspace)
    if mutation == "authored_failure":
        events[-1]["fields"]["result_kind"] = "authored_failure"
    elif mutation == "baseline_completed":
        events[-2]["fields"]["state"] = "completed"
    elif mutation == "no_baseline_start":
        del events[-3]
    elif mutation == "missing_build_step":
        del events[2:4]
    elif mutation == "reordered_steps":
        events[2:6] = [*events[4:6], *events[2:4]]
    elif mutation == "unbound_heartbeat":
        next(event for event in events if event["event"] == "heartbeat")["fields"]["step"] = "tooling_install"
    elif mutation == "repeat_baseline":
        events[-1:-1] = events[-3:-1]
    else:
        events[-3]["fields"]["step"] = "final_evaluation"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="unsafe"):
        _plan(workspace)
    assert not (workspace / "startup_recovery").exists()


@pytest.mark.parametrize("artifact", [".venv", ".agentic_runtime.json", "program_0/metrics.json", "program_0/evaluation_manifest.json"])
def test_existing_baseline_runtime_or_receipts_are_never_discarded(tmp_path, monkeypatch, artifact):
    workspace = _failed_startup(tmp_path, monkeypatch)
    _baseline_initialization_failure(workspace)
    path = workspace / "run/task" / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("retained runtime or result")
    with pytest.raises(ValueError, match="published task runtime|baseline result evidence"):
        _plan(workspace)
    assert path.read_text() == "retained runtime or result"
    assert not (workspace / "startup_recovery").exists()


def test_resume_archives_only_failed_startup_and_reuses_preparation(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    initial = sha256(workspace / "novelty/initial/novelty_report.json")
    frozen = sha256(workspace / "research_manifest.json")
    run = workspace / "run"
    old_events = (run / "investigation/events.jsonl").read_bytes()
    markers = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    monkeypatch.setattr(harness, "prepare", Mock(side_effect=AssertionError("must retain prepared inputs")))
    monkeypatch.setattr(harness, "_novelty_cycle", Mock(side_effect=AssertionError("must retain initial review")))

    def fresh(argv, *, event_callback):
        cfg = load_pipeline_config(argv)
        paths = derive_paths(cfg)
        assert cfg.investigation.resume is False
        assert task_snapshot_is_complete(paths.task_dir)
        assert not paths.investigation_workspace_dir.exists()
        assert {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()} == markers
        _prepare_investigation_workspace(paths)
        event_callback("phase_start", phase_name="investigation")
        event_callback("phase_end", phase_name="investigation", status="failed", return_code=1)
        return 1

    monkeypatch.setattr("src.pipeline.cli.main", fresh)
    assert cli.main(["harness", "--resume", str(workspace)]) == 1
    state = read_json(workspace / "harness_state.json")
    recovery = state["startup_recoveries"][0]
    assert recovery["status"] == "archived"
    archive = workspace / recovery["archive"]
    assert (archive / "investigation/events.jsonl").read_bytes() == old_events
    assert read_json(archive / "recovery.json")["failure"]["event"] == "task_runtime_failed"
    assert sha256(workspace / "research_manifest.json") == frozen
    assert sha256(workspace / "novelty/initial/novelty_report.json") == initial


def test_startup_retry_preview_is_read_only(tmp_path, monkeypatch, capsys):
    workspace = _failed_startup(tmp_path, monkeypatch)
    capsys.readouterr()
    before = sha256(workspace / "harness_state.json")
    assert cli.main(["harness", "--resume", str(workspace), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["initial_runtime_retry"] is True
    assert "investigation.execution.resume=false" in preview["pipeline_argv"]
    assert sha256(workspace / "harness_state.json") == before
    assert not (workspace / "startup_recovery").exists()


def test_pipeline_terminal_checkpoint_is_allowed_only_in_same_session(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path = workspace / "run/pipeline_events.jsonl"
    terminal = json.loads(path.read_text()) | {"session_id": "pipeline"}
    checkpoint = {"event": "run_checkpoint", "source": "pipeline", "session_id": "pipeline", "fields": {}}
    path.write_text(json.dumps(terminal) + "\n" + json.dumps(checkpoint) + "\n")
    assert _plan(workspace) is not None
    checkpoint["session_id"] = "new-pipeline"
    path.write_text(json.dumps(terminal) + "\n" + json.dumps(checkpoint) + "\n")
    with pytest.raises(ValueError, match="activity continued"):
        _plan(workspace)


def test_special_file_lock_is_rejected_without_blocking(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path = workspace / "run/.task.runtime-build.lock"
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="unsafe run artifact"):
        _plan(workspace)


def test_preparation_after_failure_is_not_a_stopped_bootstrap(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path = workspace / "run/investigation/events.jsonl"
    failure = json.loads(path.read_text())
    event = {**failure, "event": "preparation_step", "fields": {"step": "runtime_capacity"}}
    path.write_text(json.dumps(failure) + "\n" + json.dumps(event) + "\n")
    with pytest.raises(ValueError, match="activity continued"):
        _plan(workspace)


@pytest.mark.parametrize("artifact", ["ideas/attempt/program.py", ".hydra/config.yaml", "history.jsonl", ".task-seal-empty/unexpected"])
def test_unknown_or_scientific_outputs_prevent_startup_retry(tmp_path, monkeypatch, artifact):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path = workspace / "run/investigation" / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("retained evidence")
    with pytest.raises(ValueError, match="unsafe"):
        _plan(workspace)
    assert path.read_text() == "retained evidence"
    assert not (workspace / "startup_recovery").exists()


@pytest.mark.parametrize("name", ["outer_state.json", "best_program_manifest.json"])
def test_any_checkpoint_uses_normal_resume_without_archival(tmp_path, monkeypatch, name):
    workspace = _failed_startup(tmp_path, monkeypatch)
    (workspace / "run/investigation" / name).write_text("{}")
    assert _plan(workspace) is None
    assert not (workspace / "startup_recovery").exists()


@pytest.mark.parametrize("mutation", ["later_event", "new_session", "forged_path", "stale_time", "task_tamper", "final_access", "active_state", "extra_run_artifact"])
def test_ambiguous_failure_is_rejected(tmp_path, monkeypatch, mutation):
    workspace = _failed_startup(tmp_path, monkeypatch)
    path = workspace / "run/investigation/events.jsonl"
    event = json.loads(path.read_text())
    if mutation == "later_event":
        path.write_text(json.dumps(event) + "\n" + json.dumps({**event, "event": "task_runtime_complete"}) + "\n")
    elif mutation == "new_session":
        path.write_text(json.dumps(event) + "\n" + json.dumps({**event, "event": "preparation_step", "session_id": "new"}) + "\n")
    elif mutation == "forged_path":
        event["fields"]["task_dir"] = str(tmp_path / "elsewhere")
        path.write_text(json.dumps(event) + "\n")
    elif mutation == "stale_time":
        event["timestamp"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(event) + "\n")
    elif mutation == "task_tamper":
        (workspace / "run/task/task.md").write_text("Changed")
    elif mutation == "final_access":
        (workspace / "run/final_evaluation").mkdir()
    elif mutation == "active_state":
        state = read_json(workspace / "harness_state.json")
        state["status"] = "running"
        (workspace / "harness_state.json").write_text(json.dumps(state))
    else:
        (workspace / "run/.task.runtime-build-leftover").mkdir()
    with pytest.raises((ValueError, RuntimeError), match="unsafe|snapshot"):
        _plan(workspace)
    assert path.exists()


def test_live_runtime_lock_prevents_retry(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    descriptor = os.open(workspace / "run/.task.runtime-build.lock", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="still active"):
            _plan(workspace)
    finally:
        os.close(descriptor)


def test_live_coordinator_lock_prevents_resume(tmp_path, monkeypatch, capsys):
    workspace = _failed_startup(tmp_path, monkeypatch)
    descriptor = os.open(workspace, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert cli.main(["harness", "--resume", str(workspace), "--dry-run"]) == 2
        assert "Another harness resume is active" in capsys.readouterr().err
    finally:
        os.close(descriptor)


def test_preflight_failure_keeps_original_startup(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock(side_effect=ValueError("runtime unavailable")))
    assert cli.main(["harness", "--resume", str(workspace)]) == 2
    assert (workspace / "run/investigation/events.jsonl").is_file()
    assert not (workspace / "startup_recovery").exists()


def test_fresh_run_keeps_coordinator_lock_through_final_readiness(tmp_path, monkeypatch):
    original = harness.write_readiness

    def checked(workspace, *args):
        descriptor = os.open(workspace, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        return original(workspace, *args)

    monkeypatch.setattr(harness, "write_readiness", checked)
    workspace = _failed_startup(tmp_path, monkeypatch)
    descriptor = os.open(workspace, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(descriptor)


def test_managed_harness_failure_hint_points_to_safe_resume(tmp_path, monkeypatch, caplog):
    from src.pipeline.cli import _log_resume_hint
    workspace = _failed_startup(tmp_path, monkeypatch)
    cfg = load_pipeline_config([f"paths.run_dir={workspace / 'run'}", "execution.phases=investigation"])
    _log_resume_hint("investigation", cfg, derive_paths(cfg))
    assert "agsci research harness --resume" in caplog.text
    assert "verified initial runtime failures" in caplog.text


def test_changed_evidence_is_rejected_before_archive(tmp_path, monkeypatch):
    workspace = _failed_startup(tmp_path, monkeypatch)
    plan = _plan(workspace)
    path = workspace / "run/investigation/events.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after preflight"):
        harness._archive_startup_failure(workspace, read_json(workspace / "harness_state.json"), plan)
    assert path.exists()
    assert not (workspace / "startup_recovery").exists()
