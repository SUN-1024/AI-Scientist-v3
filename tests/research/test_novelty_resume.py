"""Completed paid calls replay only when their original inputs and bytes agree."""
from __future__ import annotations

import copy
import hashlib
import json
import threading

import pytest

from src.research.novelty_review import (
    InvalidReviewResume, draft_research_agenda, load_review_records, review_novelty,
)
from tests.research import test_novelty_gate as fixtures

study = fixtures.study
install_provider = fixtures.install_provider


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("interruption", ["failed", "started"])
def test_agenda_resume_reuses_draft_and_first_reflection(study, tmp_path, monkeypatch, interruption):
    task, cfg, agenda, _review = study
    changed = {**agenda, "scope": agenda["scope"] + " Development evidence only."}
    calls = install_provider(monkeypatch, [agenda, changed, {"error": "interrupted"}])
    output = tmp_path / "agenda"
    with pytest.raises(ValueError):
        draft_research_agenda(task, cfg, output)
    assert len(calls) == 3
    completed = [(output / name / "receipt.json").read_bytes()
                 for name in ("agenda_draft", "agenda_reflection_2")]
    receipt_path = output / "agenda_reflection_3/receipt.json"
    if interruption == "started":
        receipt = read(receipt_path)
        receipt["status"] = "started"
        write(receipt_path, receipt)
    incomplete_bytes = {p.name: p.read_bytes() for p in receipt_path.parent.iterdir()}
    retry_calls = install_provider(monkeypatch, [{**changed, "_reflection": {"done": True}}])
    assert draft_research_agenda(task, cfg, output, resume=True) == changed
    assert len(retry_calls) == 1
    assert completed == [(output / name / "receipt.json").read_bytes()
                         for name in ("agenda_draft", "agenda_reflection_2")]
    archives = list((output / "retry_attempts/agenda_reflection_3").iterdir())
    assert len(archives) == 1
    assert {p.name: p.read_bytes() for p in archives[0].iterdir()} == incomplete_bytes
    assert read(output / "agenda_reflection_trace.json")["status"] == "converged"
    assert draft_research_agenda(task, cfg, output, resume=True) == changed
    assert len(retry_calls) == 1


def test_review_resume_only_retries_interrupted_second_reviewer(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [review, {"error": "interrupted"}])
    output = tmp_path / "reviews"
    assert review_novelty(task, agenda, cfg, output)["status"] == "needs_evidence"
    first = (output / "reviewer_1/receipt.json").read_bytes()
    old_raw = (output / "reviewer_2/01_novelty_review.json").read_bytes()
    retry_calls = install_provider(monkeypatch, [review])
    assert review_novelty(task, agenda, cfg, output, resume=True)["status"] == "supported"
    assert len(calls) == 2 and len(retry_calls) == 1
    assert (output / "reviewer_1/receipt.json").read_bytes() == first
    assert next((output / "retry_attempts/reviewer_2").glob("*/01_novelty_review.json")).read_bytes() == old_raw
    records, errors = load_review_records(task, agenda, output)
    assert len(records) == 2 and not errors
    assert len({record["receipt"]["session_id"] for record in records}) == 2
    assert review_novelty(task, agenda, cfg, output, resume=True)["status"] == "supported"
    assert len(retry_calls) == 1
    with pytest.raises(ValueError, match="already contains"):
        review_novelty(task, agenda, cfg, output)


@pytest.mark.parametrize("change", [
    "prompt_bytes", "assessment_bytes", "response_bytes", "assessment_and_hash",
    "receipt_status", "receipt_phase", "receipt_session", "receipt_model", "legacy_receipt",
    "catalog", "source", "agenda", "model", "provider", "settings", "proxy", "prompt_template",
])
def test_completed_review_tamper_or_changed_inputs_never_reissues(study, tmp_path, monkeypatch, change):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [review])
    output = tmp_path / "reviews"
    review_novelty(task, agenda, cfg, output)
    directory = output / "reviewer_1"
    receipt_path = directory / "receipt.json"
    if change.endswith("_bytes"):
        path = directory / {"prompt_bytes": "prompt.txt", "assessment_bytes": "assessment.json",
                            "response_bytes": "01_novelty_review.json"}[change]
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "assessment_and_hash":
        assessment = read(directory / "assessment.json")
        assessment["rationale"] = "A replacement claim, absent from the saved provider response."
        write(directory / "assessment.json", assessment)
        receipt = read(receipt_path)
        receipt["assessment_sha256"] = hashlib.sha256((directory / "assessment.json").read_bytes()).hexdigest()
        write(receipt_path, receipt)
    elif change.startswith("receipt_") or change == "legacy_receipt":
        receipt = read(receipt_path)
        key, value = {"receipt_status": ("status", "failed"), "receipt_phase": ("phase", "other"),
                      "receipt_session": ("session_id", None), "receipt_model": ("model", "other"),
                      "legacy_receipt": ("schema_version", 1)}[change]
        receipt[key] = value
        write(receipt_path, receipt)
    elif change == "catalog":
        path = task / "research_context/catalog.json"
        data = read(path)
        data["sources"][0]["title"] += " changed"
        write(path, data)
    elif change == "source":
        path = task / "research_context/text/prior.txt"
        path.write_text(path.read_text() + " Changed source.")
    elif change == "agenda":
        agenda["scope"] += " Changed scope."
    elif change in {"model", "provider"}:
        setattr(cfg.model_search, change, "other")
    elif change == "settings":
        cfg.model_search.request_overrides = {"temperature": 0.25}
    elif change == "proxy":
        cfg.proxy_fallback = "http://127.0.0.1:7891"
    else:
        from src.research import novelty_review
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "novelty_review.md").write_text("Changed instructions")
        monkeypatch.setattr(novelty_review, "PROMPTS", prompt_dir)
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, output, resume=True)
    assert len(calls) == 2
    assert not (output / "retry_attempts").exists()


@pytest.mark.parametrize("change", ["task", "research_basis", "reflection", "config"])
def test_completed_agenda_resume_checks_prompts_and_configuration(study, tmp_path, monkeypatch, change):
    task, cfg, agenda, _review = study
    calls = install_provider(monkeypatch, [agenda])
    output = tmp_path / "agenda"
    draft_research_agenda(task, cfg, output)
    kwargs = {}
    if change == "task":
        (task / "task.md").write_text("A different research task")
    elif change == "research_basis":
        kwargs["research_basis"] = {"contribution": "A changed premise"}
    elif change == "reflection":
        kwargs["num_reflections"] = 4
    else:
        cfg.model_search.max_output_tokens += 1
    with pytest.raises(InvalidReviewResume):
        draft_research_agenda(task, cfg, output, resume=True, **kwargs)
    assert len(calls) == 3


def install_agent(monkeypatch, review, *, fail_second=False, same_session=False):
    calls = []
    attempts = {}
    lock = threading.Lock()

    def run(prompt, *, work_dir, cancel_event=None, **settings):
        name = work_dir.parent.name
        with lock:
            calls.append(copy.deepcopy(settings))
            attempt = attempts[name] = attempts.get(name, 0) + 1
        work_dir.mkdir()
        if fail_second and name == "reviewer_2" and attempt == 1:
            (work_dir / "stdout.json").write_text("partial provider output")
            raise ValueError("interrupted reviewer")
        native = "session-1" if same_session else f"session-{name}-{attempt}"
        data = {"protocolVersion": 1, "type": "result", "exitCode": 0,
                "data": {"finalMessage": json.dumps(review), "nativeSessionId": native}}
        raw = json.dumps(data)
        (work_dir / "stdout.json").write_text(raw)
        transport = {"status": "completed", "independent_context": True,
                     "native_session_id": native, "observed_provider": "fixture",
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "stdout_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        write(work_dir / "receipt.json", transport)
        return review, transport

    monkeypatch.setattr("src.review_backend.run_fresh_agent_json", run)
    return calls


@pytest.mark.parametrize("change", ["settings", "independent_context", "native_session", "duplicate_session"])
def test_agent_resume_requires_exact_settings_and_independent_transport(study, tmp_path, monkeypatch, change):
    task, cfg, agenda, review = study
    calls = install_agent(monkeypatch, review)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture", "reasoning_effort": "high"}
    review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    if change == "settings":
        settings["reasoning_effort"] = "medium"
    elif change == "duplicate_session":
        path = output / "reviewer_2/receipt.json"
        receipt = read(path)
        receipt["session_id"] = read(output / "reviewer_1/receipt.json")["session_id"]
        write(path, receipt)
    else:
        path = output / "reviewer_1/transport/receipt.json"
        transport = read(path)
        transport["independent_context" if change == "independent_context" else "native_session_id"] = (
            False if change == "independent_context" else "changed-native-session")
        write(path, transport)
        # Even updating the transport hash cannot make contradictory proof valid.
        receipt_path = output / "reviewer_1/receipt.json"
        receipt = read(receipt_path)
        receipt["transport_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write(receipt_path, receipt)
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)
    assert len(calls) == 2


def test_agent_resume_checks_new_reviewer_native_session(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = install_agent(monkeypatch, review, fail_second=True, same_session=True)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture"}
    assert review_novelty(task, agenda, cfg, output, reviewer_settings=settings)["status"] == "needs_evidence"
    with pytest.raises(InvalidReviewResume, match="Duplicate"):
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)
    assert len(calls) == 3
    archived = next((output / "retry_attempts/reviewer_2").glob("*/transport/stdout.json"))
    assert archived.read_text() == "partial provider output"


def test_agent_resume_replays_success_and_preserves_raw_interruption(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = install_agent(monkeypatch, review, fail_second=True)
    output = tmp_path / "reviews"
    settings = {"agent": "codex", "model": "fixture", "profile": "research", "timeout_seconds": 60,
                "environment": {"EXAMPLE_API_KEY": "secret-must-not-be-saved"}}
    review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    first_bytes = (output / "reviewer_1/receipt.json").read_bytes()
    result = review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)
    assert result["status"] == "supported" and len(calls) == 3
    assert (output / "reviewer_1/receipt.json").read_bytes() == first_bytes
    assert b"secret-must-not-be-saved" not in first_bytes
    result = review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)
    assert result["status"] == "supported" and len(calls) == 3


@pytest.mark.parametrize("agent_backend", [False, True])
@pytest.mark.parametrize("published", [False, True])
def test_interruption_at_receipt_publication_never_downgrades_a_completed_call(
    study, tmp_path, monkeypatch, agent_backend, published,
):
    from src.research import novelty_review

    task, cfg, agenda, _review = study
    calls = install_agent(monkeypatch, agenda) if agent_backend else install_provider(monkeypatch, [agenda])
    settings = {"agent": "codex", "model": "fixture"} if agent_backend else None
    output = tmp_path / "agenda"
    original = novelty_review._dump
    interrupted = False

    def dump(path, value):
        nonlocal interrupted
        if path.name == "receipt.json" and value.get("status") == "complete" and not interrupted:
            interrupted = True
            if published:
                original(path, value)
            raise KeyboardInterrupt
        original(path, value)

    monkeypatch.setattr(novelty_review, "_dump", dump)
    with pytest.raises(KeyboardInterrupt):
        draft_research_agenda(task, cfg, output, reviewer_settings=settings, num_reflections=1)
    receipt_path = output / "agenda_draft/receipt.json"
    before = receipt_path.read_bytes()
    assert read(receipt_path)["status"] == ("complete" if published else "started")
    assert len(calls) == 1
    assert draft_research_agenda(task, cfg, output, reviewer_settings=settings,
                                 num_reflections=1, resume=True) == agenda
    assert len(calls) == (1 if published else 2)
    if published:
        assert receipt_path.read_bytes() == before
        assert not (output / "retry_attempts").exists()
    else:
        assert next((output / "retry_attempts/agenda_draft").glob("*/receipt.json")).read_bytes() == before


def test_resume_preflights_later_completed_calls_before_retry(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [{"error": "interrupted"}, review])
    output = tmp_path / "reviews"
    review_novelty(task, agenda, cfg, output)
    response = output / "reviewer_2/01_novelty_review.json"
    response.write_bytes(response.read_bytes() + b" ")
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, output, resume=True)
    assert len(calls) == 2
    assert not (output / "retry_attempts").exists()


@pytest.mark.parametrize("artifact", ["receipt", "response_link", "response_path"])
def test_resume_rejects_missing_receipts_and_unsafe_responses(study, tmp_path, monkeypatch, artifact):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [review])
    output = tmp_path / "reviews"
    review_novelty(task, agenda, cfg, output)
    directory = output / "reviewer_1"
    if artifact == "receipt":
        (directory / "receipt.json").unlink()
    elif artifact == "response_link":
        path = directory / "01_novelty_review.json"
        outside = tmp_path / "response.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        receipt = read(directory / "receipt.json")
        receipt["response"] = "../reviewer_2/01_novelty_review.json"
        write(directory / "receipt.json", receipt)
    with pytest.raises(InvalidReviewResume):
        review_novelty(task, agenda, cfg, output, resume=True)
    assert len(calls) == 2


@pytest.mark.parametrize("linked_directory", ["retry_attempts", "retry_attempts/reviewer_2"])
def test_resume_rejects_linked_retry_archives(study, tmp_path, monkeypatch, linked_directory):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [review, {"error": "interrupted"}])
    output = tmp_path / "reviews"
    review_novelty(task, agenda, cfg, output)
    failed_receipt = (output / "reviewer_2/receipt.json").read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = output / linked_directory
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(InvalidReviewResume, match="archive"):
        review_novelty(task, agenda, cfg, output, resume=True)
    assert len(calls) == 2 and list(outside.iterdir()) == []
    assert (output / "reviewer_2/receipt.json").read_bytes() == failed_receipt


@pytest.mark.parametrize("agent_backend", [False, True])
def test_legacy_completed_receipts_remain_readable_but_cannot_resume(study, tmp_path, monkeypatch, agent_backend):
    task, cfg, agenda, review = study
    calls = install_agent(monkeypatch, review) if agent_backend else install_provider(monkeypatch, [review])
    settings = {"agent": "codex", "model": "fixture"} if agent_backend else None
    output = tmp_path / "reviews"
    review_novelty(task, agenda, cfg, output, reviewer_settings=settings)
    for directory in output.glob("reviewer_*"):
        receipt = read(directory / "receipt.json")
        receipt["schema_version"] = 1
        receipt.pop("configuration_sha256")
        if agent_backend:
            # Old transport records allowed an absent native ID. They remain
            # readable for existing reports, but cannot authorize call resume.
            receipt.pop("native_session_id")
            transport = read(directory / "transport/receipt.json")
            transport.pop("native_session_id")
            response = read(directory / "transport/stdout.json")
            response["data"].pop("nativeSessionId")
            write(directory / "transport/stdout.json", response)
            receipt["response_sha256"] = hashlib.sha256((directory / "transport/stdout.json").read_bytes()).hexdigest()
            transport["stdout_sha256"] = receipt["response_sha256"]
            write(directory / "transport/receipt.json", transport)
            receipt["transport_receipt_sha256"] = hashlib.sha256((directory / "transport/receipt.json").read_bytes()).hexdigest()
        write(directory / "receipt.json", receipt)
    records, errors = load_review_records(task, agenda, output)
    assert len(records) == 2 and not errors
    with pytest.raises(InvalidReviewResume, match="configuration"):
        review_novelty(task, agenda, cfg, output, reviewer_settings=settings, resume=True)
    assert len(calls) == 2
