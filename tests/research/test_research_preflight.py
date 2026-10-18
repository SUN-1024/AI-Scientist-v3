from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agsci import doctor
from src.investigation.program_spec import render_default_program_spec_toml
from src.pipeline.cli import load_pipeline_config
from src.research import cli, preflight
from tests.research.test_unified_research import make_task


def runtime_host(tmp_path: Path, **changes) -> doctor.Host:
    return replace(doctor.Host(
        which=lambda name: f"/fixture/bin/{name}",
        run=Mock(side_effect=AssertionError("unexpected subprocess")),
        probe_docker=Mock(return_value=(True, "")),
        image_exists=Mock(return_value=(True, "")),
        environ={}, repo_root=tmp_path,
    ), **changes)


def complete_task(tmp_path: Path) -> Path:
    task = make_task(tmp_path / "task")
    (task / "capture.py").write_text("# Fixture capture\n")
    (task / "evaluate.py").write_text("# Fixture evaluator\n")
    (task / "program_0" / "src").mkdir(parents=True)
    (task / "program_0" / "src" / "program.py").write_text("# Fixture baseline\n")
    (task / "program_0" / "program_spec.toml").write_text(render_default_program_spec_toml())
    return task


def test_native_failure_stops_before_prepare_or_online_retrieval(tmp_path, monkeypatch, capsys):
    task = complete_task(tmp_path)
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path, which=lambda _name: None)
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    prepare = Mock(side_effect=AssertionError("must not prepare or spend on retrieval"))
    monkeypatch.setattr(cli, "prepare", prepare)
    assert cli.main([
        "run", str(task), "--workspace", str(workspace), "container.enabled=false",
        "execution.phases=investigation", "agent.name=codex",
    ]) == 2
    prepare.assert_not_called()
    host.probe_docker.assert_not_called()
    assert not workspace.exists()
    assert not (task / "research_context").exists()
    assert "codex" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["docker_cli", "daemon", "image", "auth_root", "env_file"])
def test_container_failures_are_actionable_before_prepare(tmp_path, monkeypatch, capsys, failure):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path)
    overrides = [f"container.auth_root={tmp_path}", "container.env_file=''"]
    expected = "docker"
    if failure == "docker_cli":
        host.which = lambda name: None if name == "docker" else f"/fixture/bin/{name}"
    elif failure == "daemon":
        host.probe_docker.return_value = (False, "daemon unavailable")
    elif failure == "image":
        host.image_exists.return_value = (False, "not installed")
        expected = "agent image"
    elif failure == "auth_root":
        overrides[0] = f"container.auth_root={tmp_path / 'missing_auth'}"
        expected = "auth root"
    else:
        overrides[1] = "container.env_file=missing.env"
        expected = "env file"
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    prepare = Mock(side_effect=AssertionError("must not prepare or retrieve"))
    monkeypatch.setattr(cli, "prepare", prepare)
    assert cli.main(["run", str(task), "--workspace", str(workspace), *overrides]) == 2
    prepare.assert_not_called()
    assert not workspace.exists()
    assert expected in capsys.readouterr().err
    host.run.assert_not_called()


def test_native_profile_uses_source_and_resume_keeps_execution_path(tmp_path, monkeypatch, capsys):
    task = complete_task(tmp_path)
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path)
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    pipeline = Mock(return_value=0)
    monkeypatch.setattr("src.pipeline.cli.main", pipeline)
    overrides = [
        "container.enabled=false", "execution.phases=investigation", "agent.name=codex",
        "agent.profile=local_magpie", "agent.model=custom/model",
        f"container.auth_root={tmp_path / 'no_standard_login'}",
    ]
    assert cli.main(["run", str(task), "--workspace", str(workspace), "--offline", *overrides]) == 0
    actual_args = pipeline.call_args.args[0]
    assert f"task.dir={json.dumps(str(workspace / 'input_task'))}" in actual_args
    assert "agent.profile=local_magpie" in actual_args
    assert "preflight warning: credentials:" in capsys.readouterr().err
    shutil.rmtree(task)
    monkeypatch.setattr(cli, "ensure_execution_ready", Mock(side_effect=AssertionError("no retrieval on resume")))
    assert cli.main(["run", "--resume", str(workspace)]) == 0
    assert pipeline.call_count == 2
    host.probe_docker.assert_not_called()
    host.image_exists.assert_not_called()
    host.run.assert_not_called()


@pytest.mark.parametrize("command,flags", [("run", ["--dry-run"]), ("prepare", ["--offline"])])
def test_prepare_and_dry_run_do_not_probe_execution_runtime(tmp_path, monkeypatch, command, flags):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    probe = Mock(side_effect=AssertionError("must not probe execution runtime"))
    monkeypatch.setattr(cli, "ensure_execution_ready", probe)
    assert cli.main([command, str(task), "--workspace", str(workspace), *flags]) == 0
    probe.assert_not_called()
    assert workspace.exists() is (command == "prepare")


def test_scaffold_failure_and_native_drafting_are_detected(tmp_path):
    task = make_task(tmp_path / "task")
    host = runtime_host(tmp_path)
    cfg = load_pipeline_config([
        "container.enabled=false", "execution.phases=investigation", "agent.name=codex",
    ])
    results = preflight.execution_checks(cfg, task, host=host)
    assert any(result.name == "task" and result.status == doctor.FAIL for result in results)
    cfg.execution.phases = "task_drafting,investigation"
    results = preflight.execution_checks(cfg, task, host=host)
    assert any(result.name == "task drafting" and result.status == doctor.FAIL for result in results)
    # A Docker drafting run can start from a brief; it need not have an evaluator yet.
    cfg.container.enabled = True
    cfg.container.auth_root = str(tmp_path)
    cfg.container.env_file = ""
    results = preflight.execution_checks(cfg, task, host=host)
    assert not any(result.status == doctor.FAIL for result in results)


def test_source_task_check_does_not_mutate_resolved_destination(tmp_path):
    task = complete_task(tmp_path)
    destination = tmp_path / "not_created" / "input_task"
    cfg = load_pipeline_config([
        f"task.dir={destination}", "container.enabled=false",
        "execution.phases=investigation", "agent.name=codex",
    ])
    results = preflight.execution_checks(cfg, task, host=runtime_host(tmp_path))
    assert not any(result.status == doctor.FAIL for result in results)
    assert cfg.task.dir == str(destination)
    assert not destination.exists()


def test_missing_task_brief_is_a_preflight_failure(tmp_path):
    task = complete_task(tmp_path)
    (task / "task.md").unlink()
    cfg = load_pipeline_config(["container.enabled=false", "execution.phases=investigation"])
    results = preflight.execution_checks(cfg, task, host=runtime_host(tmp_path))
    assert any(result.status == doctor.FAIL and "task.md" in result.detail for result in results)
