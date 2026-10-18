"""A postdraft finalizer is a hard execution boundary, not an event observer."""
import pytest

from src.pipeline.cli import main
from tests.pipeline.test_pipeline_cli_dryrun import _source_task, _mock_successful_investigation

pytestmark = pytest.mark.usefixtures("stub_image_leases")


@pytest.fixture(autouse=True)
def canonical_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTIC_TEMP_ROOT", str(tmp_path.resolve()))


def test_finalizer_failure_stops_before_investigation_even_without_fail_fast(tmp_path, monkeypatch):
    source = _source_task(tmp_path)
    root = tmp_path / "run"
    commands = []
    monkeypatch.setattr("src.pipeline.cli.run_cmd", lambda command, **_: commands.append(command) or 0)
    def finalize(action, _cfg, _paths):
        if action == "finalize":
            raise ValueError("postdraft review incomplete")
        return False
    with pytest.raises(ValueError, match="review incomplete"):
        main([f"task.dir={source}", f"paths.run_dir={root}", "run_id=test",
              "execution.phases=task_drafting,investigation", "execution.fail_fast=false"], task_finalizer=finalize)
    assert len(commands) == 1 and "execution.task_only=true" in commands[0]
    assert not (root / "investigation").exists()


def test_recovered_draft_skips_child_command_and_verifies_before_investigation(tmp_path, monkeypatch):
    from src.pipeline.task_snapshot import prepare_run_task
    source = _source_task(tmp_path)
    root = tmp_path / "run"
    actions, commands = [], []
    _mock_successful_investigation(monkeypatch)
    def finalize(action, _cfg, paths):
        actions.append(action)
        if action == "recover":
            prepare_run_task(paths)
        if action == "verify":
            assert not paths.investigation_workspace_dir.exists()
        return True
    monkeypatch.setattr("src.pipeline.cli.run_cmd", lambda command, **_: commands.append(command) or 0)
    assert main([f"task.dir={source}", f"paths.run_dir={root}", "run_id=test",
                 "execution.phases=task_drafting,investigation"], task_finalizer=finalize) == 0
    assert actions == ["recover", "draft_complete", "finalize", "verify"]
    assert len(commands) == 1 and "execution.task_only=true" not in commands[0]


def test_dry_run_does_not_invoke_finalizer(tmp_path, monkeypatch):
    source = _source_task(tmp_path)
    calls = []
    assert main([f"task.dir={source}", f"paths.run_dir={tmp_path / 'run'}", "run_id=test",
                 "execution.phases=task_drafting,investigation", "--dry-run"],
                task_finalizer=lambda *args: calls.append(args)) == 0
    assert not calls
