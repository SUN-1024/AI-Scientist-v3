"""Resume binds skipped paper outputs without bypassing their host evidence."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.pipeline.cli import load_pipeline_config
from src.pipeline.config import parse_pipeline_phases, validate_pipeline_config
from src.research import cli, harness
from tests.research.test_unified_research import make_task
from tests.revision_test_support import review_payload, write_workspace


def completed_paper_workspace(tmp_path, monkeypatch, *, review_completed=True):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace with spaces"
    calls = []
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())

    def interrupted(argv, *, event_callback):
        calls.append(argv)
        paper = workspace / "run/paper"
        write_workspace(paper, title="Recorded manuscript")
        event_callback("phase_end", phase_name="paper", status="completed", return_code=0)
        if review_completed:
            review = paper / "ai_review.json"
            review.write_text(json.dumps(review_payload()))
            manifest_path = paper / "latest_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["ai_review"] = {"status": "completed", "review_json": str(review)}
            manifest_path.write_text(json.dumps(manifest))
            event_callback("phase_end", phase_name="review", status="completed", return_code=0)
            event_callback("phase_end", phase_name="revision", status="degraded", reason="fixture interruption")
        else:
            event_callback("phase_end", phase_name="review", status="degraded", reason="fixture interruption")
        return 0

    monkeypatch.setattr("src.pipeline.cli.main", interrupted)
    assert cli.main([
        "harness", str(task), "--workspace", str(workspace), "--offline",
        "--quality-target", "development", "execution.phases=paper,review,revision",
        "container.enabled=true",
    ]) == 0
    return workspace, calls


def frozen_files(workspace):
    return {name: (workspace / name).read_bytes() for name in [
        "harness_state.json", "research_manifest.json", "run/paper/paper.pdf", "run/paper/paper.tex",
    ]}


@pytest.mark.parametrize("review_completed", [False, True])
def test_downstream_resume_preview_has_executable_managed_paper_input(
    tmp_path, monkeypatch, capsys, review_completed,
):
    workspace, calls = completed_paper_workspace(tmp_path, monkeypatch, review_completed=review_completed)
    capsys.readouterr()
    before = frozen_files(workspace)
    monkeypatch.setattr(harness, "prepare", Mock(side_effect=AssertionError("no preparation on preview")))
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock(side_effect=AssertionError("no runtime probes on preview")))
    assert cli.main(["harness", "--resume", str(workspace), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["remaining_phases"] == (["revision"] if review_completed else ["review", "revision"])
    assert preview["requested_phases"] == ["paper", "review", "revision"]
    cfg = load_pipeline_config(preview["pipeline_argv"])
    assert Path(cfg.paths.paper_dir) == workspace / "run/paper"
    assert validate_pipeline_config(cfg) == []
    assert frozen_files(workspace) == before
    assert len(calls) == 1


def test_online_resume_uses_the_same_completed_manuscript(tmp_path, monkeypatch):
    workspace, calls = completed_paper_workspace(tmp_path, monkeypatch)
    before_pdf = (workspace / "run/paper/paper.pdf").read_bytes()

    def check(cfg, task):
        assert Path(cfg.paths.paper_dir) == workspace / "run/paper"
        assert validate_pipeline_config(replace(cfg, task=replace(cfg.task, dir=str(task)))) == []

    def resumed(argv, *, event_callback):
        calls.append(argv)
        cfg = load_pipeline_config(argv)
        assert parse_pipeline_phases(cfg.execution.phases) == {"revision"}
        assert Path(cfg.paths.paper_dir) == workspace / "run/paper"
        event_callback("phase_end", phase_name="revision", status="completed", return_code=0)
        return 0

    monkeypatch.setattr(harness, "ensure_execution_ready", check)
    monkeypatch.setattr("src.pipeline.cli.main", resumed)
    assert cli.main(["harness", "--resume", str(workspace)]) == 0
    assert len(calls) == 2
    assert (workspace / "run/paper/paper.pdf").read_bytes() == before_pdf


def test_resume_preview_rejects_incomplete_initial_review_manifest(tmp_path, monkeypatch, capsys):
    workspace, calls = completed_paper_workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    path = workspace / "run/paper/latest_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["ai_review"]["status"] = "failed"
    path.write_text(json.dumps(manifest))
    before = frozen_files(workspace)
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock(side_effect=AssertionError("no runtime probes")))
    assert cli.main(["harness", "--resume", str(workspace), "--dry-run"]) == 2
    assert "completed initial AI review" in capsys.readouterr().err
    assert frozen_files(workspace) == before
    assert len(calls) == 1


def test_unsealed_paper_files_cannot_supply_resume_input(tmp_path, monkeypatch):
    workspace, _ = completed_paper_workspace(tmp_path, monkeypatch)
    prior = deepcopy(json.loads((workspace / "harness_state.json").read_text()))
    prior["stages"]["paper"]["status"] = "pending"
    with pytest.raises(ValueError, match="Saved paper evidence changed or is incomplete"):
        harness._resume_paper_overrides(workspace, prior, {"revision"})


def test_resume_binding_preserves_manuscript_tamper_rejection(tmp_path, monkeypatch, capsys):
    workspace, calls = completed_paper_workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    (workspace / "run/paper/paper.tex").write_text("Unrecorded manuscript edit")
    before = frozen_files(workspace)
    assert cli.main(["harness", "--resume", str(workspace), "--dry-run"]) == 2
    assert "Saved paper evidence changed" in capsys.readouterr().err
    assert frozen_files(workspace) == before
    assert len(calls) == 1


def test_user_stage_path_override_is_still_rejected(tmp_path, monkeypatch, capsys):
    workspace, calls = completed_paper_workspace(tmp_path, monkeypatch)
    capsys.readouterr()
    before = frozen_files(workspace)
    assert cli.main([
        "harness", "--resume", str(workspace), "--dry-run",
        "paths.paper_dir=" + str(tmp_path / "foreign-paper"),
    ]) == 2
    assert "Harness stage paths are managed" in capsys.readouterr().err
    assert frozen_files(workspace) == before
    assert len(calls) == 1
