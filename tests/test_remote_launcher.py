"""Deployment command boundaries; no SSH or Docker calls in these tests."""
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import shlex
import subprocess

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/remote-agsci"
    loader = SourceFileLoader("remote_launcher", str(path))
    module = module_from_spec(spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def test_sync_does_not_upload_credentials_or_delete_remote(launcher, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "remote", lambda *a, **kw: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda argv, **kw: calls.append(argv))
    launcher.sync("/home/test/jobs/unique")
    command = calls[0]
    assert "--exclude=.env*" in command
    assert "--exclude=.codex" in command
    assert "--exclude=.venv" in command
    assert "--delete" not in command


def test_container_and_auth_cannot_be_disabled_by_extra_overrides(launcher):
    argv = launcher.args_for("run", ["container.enabled=false", "task.dir=examples/grpo"], "/tmp/job")
    assert argv.index("container.enabled=true") > argv.index("container.enabled=false")
    assert argv[-1] == "container.env_file=" + launcher.REMOTE + "/.env"
    assert "task.dir=examples/grpo" in argv
    assert launcher.CFG["task"] not in argv  # no conflicting positional task


def test_gpu_probe_excludes_action_flags_and_fails_before_launch(launcher):
    cmd = launcher.args_for("run", ["preset=grpo"], "/tmp/job") + ["--dry-run"]
    setup = launcher.gpu_setup(cmd)
    assert "--dry-run" not in setup
    assert setup.startswith("agsci_gpu_count=$(")
    assert 'if [ "$agsci_gpu_count" -gt 0 ]' in setup


def test_run_with_no_credentials_does_not_start_job(launcher, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "sync", lambda *a: None)
    monkeypatch.setattr(launcher.sys, "argv", ["remote-agsci", "run"])
    def remote(script, **kw):
        calls.append(script)
        raise subprocess.CalledProcessError(1, "ssh")
    monkeypatch.setattr(launcher, "remote", remote)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main()
    assert len(calls) == 1 and "test -s" in calls[0]
    assert "nohup" not in calls[0]


def test_background_run_has_snapshot_and_exit_receipt(launcher, monkeypatch):
    scripts, destinations = [], []
    monkeypatch.setattr(launcher, "sync", destinations.append)
    monkeypatch.setattr(launcher, "remote", lambda script, **kw: scripts.append(script))
    monkeypatch.setattr(launcher.sys, "argv", ["remote-agsci", "run", "agent.model=model with spaces"])
    launcher.main()
    assert "/jobs/" in destinations[0]
    assert "nohup bash run.sh" in scripts[-1]
    assert "exit-code" in scripts[-1]
    assert "paths.run_dir=" in scripts[-1]
    assert shlex.quote("agent.model=model with spaces") in shlex.split(scripts[-1])[2]
