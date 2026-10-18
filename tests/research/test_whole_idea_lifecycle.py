"""Whole-idea policy is frozen before model calls and stays absent on legacy resume."""
from __future__ import annotations

from dataclasses import asdict
import json

import pytest
import yaml

from src.research import workspace
from src.research.config import ResearchConfig, WholeIdeaPolicyConfig, load_config
from src.research.evidence import render_context
from src.research.novelty import content_hash, freeze_evidence
from src.research.novelty_review import InvalidReviewResume, draft_research_agenda, review_novelty
from src.research.whole_idea_policy import (
    MARKER_PATH, freeze_whole_idea_policy, load_whole_idea_policy,
)
from tests.research import test_novelty_gate as fixtures
from tests.research.test_unified_research import make_task

study = fixtures.study


def test_prepare_freezes_policy_before_author_calls(tmp_path):
    task = make_task(tmp_path / "task")
    observed = []

    def first_call(staged, cfg):
        assert cfg.whole_idea_policy.enabled
        marker = load_whole_idea_policy(staged)
        assert marker is not None
        observed.append(content_hash(marker))
        assert "whole idea" in render_context(staged)

    location = tmp_path / "workspace"
    workspace.prepare(task, location, offline=True, overrides=[], after_literature=first_call)
    assert len(observed) == 1
    assert not (task / MARKER_PATH).exists()
    assert workspace.resume(location)[0].whole_idea_policy.enabled
    assert freeze_evidence(location / "input_task")["whole_idea_policy_sha256"] == observed[0]
    (location / "input_task" / MARKER_PATH).unlink()
    with pytest.raises(ValueError):
        workspace.resume(location)


def test_legacy_manifest_resume_does_not_enable_new_policy(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    raw = yaml.safe_load((task / "research.yaml").read_text())
    raw.pop("whole_idea_policy")
    (task / "research.yaml").write_text(yaml.safe_dump(raw))
    cfg = load_config(task)
    cfg.whole_idea_policy.enabled = False
    monkeypatch.setattr(workspace, "load_config", lambda _path: cfg)
    location = tmp_path / "workspace"
    manifest = workspace.prepare(task, location, offline=True, overrides=[])
    manifest["research_config"].pop("whole_idea_policy")
    manifest_path = location / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(workspace, "load_config", load_config)
    before = manifest_path.read_bytes()
    frozen = workspace.task_digest(location / "input_task")
    restored, _ = workspace.resume(location)
    assert restored.whole_idea_policy == WholeIdeaPolicyConfig(enabled=False)
    assert not (location / "input_task" / MARKER_PATH).exists()
    assert manifest_path.read_bytes() == before
    assert workspace.task_digest(location / "input_task") == frozen


@pytest.mark.parametrize("value", [{"enabled": "yes"}, {"policy_version": True}, {"policy_version": 2}])
def test_policy_config_rejects_ambiguous_or_unknown_versions(tmp_path, value):
    task = make_task(tmp_path / "task")
    raw = yaml.safe_load((task / "research.yaml").read_text())
    raw["whole_idea_policy"] = value
    (task / "research.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="whole_idea_policy"):
        load_config(task)


def test_legacy_normalization_does_not_hide_nondefault_policy():
    cfg = ResearchConfig()
    saved = asdict(cfg)
    saved.pop("whole_idea_policy")
    assert workspace._matches_frozen_config(cfg, saved)
    assert not cfg.whole_idea_policy.enabled
    changed = ResearchConfig(whole_idea_policy=WholeIdeaPolicyConfig(policy_version=2))
    assert not workspace._matches_frozen_config(changed, saved)


def test_author_and_review_receipts_bind_policy_before_calls(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    marker = freeze_whole_idea_policy(task)
    calls = fixtures.install_provider(monkeypatch, [agenda, review, review])
    author = tmp_path / "author"
    review_dir = tmp_path / "review"
    draft_research_agenda(task, cfg, author, num_reflections=1)
    review_novelty(task, agenda, cfg, review_dir)
    assert len(calls) == 3
    for directory in (author / "agenda_draft", review_dir / "reviewer_1", review_dir / "reviewer_2"):
        prompt = (directory / "prompt.txt").read_text()
        receipt = json.loads((directory / "receipt.json").read_text())
        assert "whole idea" in prompt
        assert receipt["frozen_evidence"]["whole_idea_policy_sha256"] == content_hash(marker)
    (task / MARKER_PATH).unlink()
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, review_dir, resume=True)
    assert len(calls) == 3
