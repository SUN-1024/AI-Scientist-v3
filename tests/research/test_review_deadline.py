"""A shared deadline bounds new calls without invalidating completed receipts."""
from __future__ import annotations

import json

import pytest

from src.research.model_search import SearchSession
from src.research.novelty_review import draft_research_agenda, review_novelty
from tests.research import test_novelty_gate as fixtures
from tests.research.test_novelty_resume import install_agent

study = fixtures.study


def test_deadline_stops_next_reflection_and_preserves_completed_draft(study, tmp_path, monkeypatch):
    task, cfg, agenda, _ = study
    current = [100.0]
    calls = []
    monkeypatch.setenv("AGSCI_LITERATURE_API_KEY", "fixture-key")
    monkeypatch.setattr("time.time", lambda: current[0])

    def provider(*args, **kwargs):
        calls.append(kwargs["timeout"])
        current[0] = 111.0
        return {"choices": [{"message": {"content": json.dumps(agenda)}}]}

    monkeypatch.setattr("src.research.model_search._http_json", provider)
    output = tmp_path / "author"
    with pytest.raises(ValueError, match="deadline exhausted"):
        draft_research_agenda(task, cfg, output, deadline_unix=110.0)
    assert calls == [10]
    receipt = (output / "agenda_draft/receipt.json").read_bytes()
    assert not (output / "agenda_reflection_2").exists()
    with pytest.raises(ValueError, match="deadline exhausted"):
        draft_research_agenda(task, cfg, output, deadline_unix=110.0, resume=True)
    assert calls == [10] and (output / "agenda_draft/receipt.json").read_bytes() == receipt


def test_expired_deadline_still_replays_complete_author_and_review(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    current = [100.0]
    monkeypatch.setattr("time.time", lambda: current[0])
    calls = fixtures.install_provider(monkeypatch, [agenda])
    author = tmp_path / "author"
    draft_research_agenda(task, cfg, author, deadline_unix=110.0, num_reflections=1)
    current[0] = 500.0
    assert draft_research_agenda(task, cfg, author, deadline_unix=110.0,
                                 num_reflections=1, resume=True) == agenda
    assert len(calls) == 1
    current[0] = 100.0
    reviewer_calls = fixtures.install_provider(monkeypatch, [review])
    output = tmp_path / "review"
    assert review_novelty(task, agenda, cfg, output, deadline_unix=110.0)["status"] == "supported"
    current[0] = 500.0
    assert review_novelty(task, agenda, cfg, output, deadline_unix=110.0, resume=True)["status"] == "supported"
    assert len(reviewer_calls) == 2


def test_agent_timeout_is_clamped_but_original_configuration_replays(study, tmp_path, monkeypatch):
    task, cfg, agenda, _ = study
    current = [100.0]
    monkeypatch.setattr("time.time", lambda: current[0])
    calls = install_agent(monkeypatch, agenda)
    settings = {"agent": "codex", "model": "fixture", "timeout_seconds": 900}
    output = tmp_path / "author"
    draft_research_agenda(task, cfg, output, reviewer_settings=settings, num_reflections=1, deadline_unix=108.5)
    assert calls[0]["timeout_seconds"] == 8
    assert settings["timeout_seconds"] == 900
    current[0] = 200.0
    assert draft_research_agenda(task, cfg, output, reviewer_settings=settings, num_reflections=1,
                                 deadline_unix=108.5, resume=True) == agenda
    assert len(calls) == 1


def test_http_continuation_checks_remaining_time_again(study, tmp_path, monkeypatch):
    _task, cfg, _agenda, _ = study
    cfg.model_search.api_format = "anthropic"
    current = [100.0]
    calls = []
    monkeypatch.setenv("AGSCI_LITERATURE_API_KEY", "fixture-key")
    monkeypatch.setattr("time.time", lambda: current[0])

    def provider(*args, **kwargs):
        calls.append(kwargs["timeout"])
        current[0] = 111.0
        return {"stop_reason": "pause_turn", "content": [{"type": "text", "text": "continued"}]}

    monkeypatch.setattr("src.research.model_search._http_json", provider)
    session = SearchSession(cfg, tmp_path / "session", deadline_unix=110.0)
    with pytest.raises(ValueError, match="deadline exhausted"):
        session.ask("Inspect the supplied evidence", "research_agenda", native=False)
    assert calls == [10]
