"""Review feedback is attributed author input, never independent evidence."""
from __future__ import annotations

import copy
import json

import pytest

from src.research.novelty_review import InvalidReviewResume, draft_research_agenda
from tests.research import test_novelty_gate as fixtures

study = fixtures.study


def feedback():
    return {"schema_version": 1, "scope": "initial_agenda", "parent_review_sha256": "a" * 64,
            "unresolved_requests": [{"kind": "retrieve_literature", "reason": "Inspect the closest control."}]}


def test_author_feedback_is_frozen_and_completed_calls_replay(study, tmp_path, monkeypatch):
    task, cfg, agenda, _ = study
    calls = fixtures.install_provider(monkeypatch, [agenda])
    output = tmp_path / "review_cycle"
    original = feedback()
    result = draft_research_agenda(task, cfg, output, review_feedback=original)
    assert result == agenda
    assert len(calls) == 3
    assert json.loads((output / "review_feedback.json").read_text()) == original
    receipt = json.loads((output / "agenda_draft/receipt.json").read_text())
    assert receipt["frozen_evidence"]["review_feedback_sha256"]
    prompt = (output / "agenda_draft/prompt.txt").read_text()
    assert "Preserve unresolved concerns" in prompt
    assert draft_research_agenda(task, cfg, output, review_feedback=original, resume=True) == result
    assert len(calls) == 3
    changed = copy.deepcopy(original)
    changed["unresolved_requests"][0]["reason"] = "A new request"
    with pytest.raises(InvalidReviewResume):
        draft_research_agenda(task, cfg, output, review_feedback=changed, resume=True)
    assert len(calls) == 3


@pytest.mark.parametrize("change", ["omit", "feedback_file", "prompt_hash"])
def test_completed_feedback_tamper_never_reissues(study, tmp_path, monkeypatch, change):
    from src.research.novelty import content_hash

    task, cfg, agenda, _ = study
    calls = fixtures.install_provider(monkeypatch, [agenda])
    output = tmp_path / "review_cycle"
    original = feedback()
    draft_research_agenda(task, cfg, output, review_feedback=original)
    if change == "feedback_file":
        altered = {**original, "scope": "selected_interpretation"}
        (output / "review_feedback.json").write_text(json.dumps(altered))
    elif change == "prompt_hash":
        path = output / "agenda_draft/receipt.json"
        receipt = json.loads(path.read_text())
        receipt["frozen_evidence"]["review_feedback_sha256"] = content_hash({"changed": True})
        path.write_text(json.dumps(receipt))
    with pytest.raises(InvalidReviewResume):
        draft_research_agenda(task, cfg, output, review_feedback=None if change == "omit" else original,
                             resume=True)
    assert len(calls) == 3


def test_selected_feedback_without_candidate_stops_before_model(study, tmp_path, monkeypatch):
    task, cfg, agenda, _ = study
    calls = fixtures.install_provider(monkeypatch, [agenda])
    with pytest.raises(ValueError, match="candidate packet"):
        draft_research_agenda(task, cfg, tmp_path / "review_cycle",
                             review_feedback={**feedback(), "scope": "selected_interpretation"})
    assert calls == []


def test_legacy_author_has_no_feedback_artifacts(study, tmp_path, monkeypatch):
    task, cfg, agenda, _ = study
    fixtures.install_provider(monkeypatch, [agenda])
    output = tmp_path / "legacy"
    draft_research_agenda(task, cfg, output)
    assert not (output / "review_feedback.json").exists()
    receipt = json.loads((output / "agenda_draft/receipt.json").read_text())
    assert "review_feedback_sha256" not in receipt["frozen_evidence"]
