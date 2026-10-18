"""Method guidance is frozen before paid calls and stays disabled on old studies."""
from __future__ import annotations

from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest
import yaml

from src.research import workspace
from src.research.config import MethodologyConfig, load_config, write_config
from src.research.methodology import MARKER_PATH, load_methodology, write_methodology
from src.research.novelty import freeze_evidence
from src.research.novelty_review import InvalidReviewResume, draft_research_agenda, review_novelty
from tests.research import test_novelty_gate as fixtures
from tests.research.test_unified_research import make_task

study = fixtures.study


def test_new_study_freezes_medical_guidance_before_first_call(tmp_path):
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.methodology.profile = "medical"
    write_config(task / "research.yaml", cfg)
    seen = []
    def after_literature(staged, frozen_cfg):
        marker = load_methodology(staged)
        assert marker["profile"] == frozen_cfg.methodology.profile == "medical"
        seen.append(marker["rules_sha256"])
        return None
    location = tmp_path / "workspace"
    workspace.prepare(task, location, offline=True, overrides=[], after_literature=after_literature)
    assert len(seen) == 1
    cfg, _manifest = workspace.resume(location)
    assert cfg.methodology.enabled
    assert freeze_evidence(location / "input_task")["methodology"]["rules_sha256"] == seen[0]
    assert not (task / MARKER_PATH).exists()


def _legacy_task(path):
    task = make_task(path)
    data = yaml.safe_load((task / "research.yaml").read_text())
    data.pop("methodology")
    (task / "research.yaml").write_text(yaml.safe_dump(data))
    return task


def test_legacy_preparation_never_enables_new_prompts_on_resume(tmp_path, monkeypatch):
    task = _legacy_task(tmp_path / "task")
    location = tmp_path / "workspace"
    original = workspace.load_config
    def old_loader(path):
        cfg = original(path)
        cfg.methodology.enabled = False
        return cfg
    monkeypatch.setattr(workspace, "load_config", old_loader)
    with pytest.raises(KeyboardInterrupt):
        workspace.prepare(task, location, offline=True, overrides=[],
                          after_literature=Mock(side_effect=KeyboardInterrupt))
    journal_path = location / workspace.PREPARATION_FILE
    journal = json.loads(journal_path.read_text())
    journal["research_config"].pop("methodology")
    journal_path.write_text(json.dumps(journal))
    monkeypatch.setattr(workspace, "load_config", original)
    before = workspace.task_digest(location / "input_task")
    cfg, _journal = workspace.load_preparation(location)
    assert cfg.methodology == MethodologyConfig(enabled=False)
    workspace.prepare(task, location, offline=True, overrides=[], resume_preparation=True)
    assert workspace.task_digest(location / "input_task") == before
    assert not (location / "input_task" / MARKER_PATH).exists()
    assert not workspace.resume(location)[0].methodology.enabled


def test_legacy_manifest_keeps_config_and_bytes_unchanged(tmp_path, monkeypatch):
    task = _legacy_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.methodology.enabled = False
    monkeypatch.setattr(workspace, "load_config", lambda _path: cfg)
    location = tmp_path / "workspace"
    manifest = workspace.prepare(task, location, offline=True, overrides=[])
    manifest["research_config"].pop("methodology")
    manifest_path = location / "research_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(workspace, "load_config", load_config)
    before = manifest_path.read_bytes()
    restored, _ = workspace.resume(location)
    assert not restored.methodology.enabled
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("value", [{"enabled": "yes"}, {"profile": "clinical-certified"}, {"profile": []}])
def test_methodology_config_rejects_invalid_values(tmp_path, value):
    task = make_task(tmp_path / "task")
    data = yaml.safe_load((task / "research.yaml").read_text())
    data["methodology"] = value
    (task / "research.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="methodology"):
        load_config(task)


def test_author_and_independent_review_receive_frozen_rules(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    marker = write_methodology(task, "medical")
    calls = fixtures.install_provider(monkeypatch, [agenda, review, review])
    author = tmp_path / "author"
    draft_research_agenda(task, cfg, author, num_reflections=1)
    reviews = tmp_path / "review"
    review_novelty(task, agenda, cfg, reviews)
    assert len(calls) == 3
    for directory in (author / "agenda_draft", reviews / "reviewer_1", reviews / "reviewer_2"):
        prompt = (directory / "prompt.txt").read_text()
        receipt = json.loads((directory / "receipt.json").read_text())
        assert marker["rules_sha256"] in prompt
        assert receipt["frozen_evidence"]["methodology"]["profile"] == "medical"
    (task / MARKER_PATH).unlink()
    write_methodology(task, "general")
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, reviews, resume=True)
    assert len(calls) == 3


def test_old_config_normalization_is_consistent_and_does_not_mask_changed_profiles():
    from src.research.config import ResearchConfig
    cfg = ResearchConfig()
    saved = asdict(cfg)
    saved.pop("methodology")
    assert workspace._matches_frozen_config(cfg, saved)
    assert not cfg.methodology.enabled
    changed = ResearchConfig(methodology=MethodologyConfig(profile="medical"))
    assert not workspace._matches_frozen_config(changed, saved)
