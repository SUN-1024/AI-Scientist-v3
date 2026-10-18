"""src/aireview.py: JSON extraction from LLM replies, reviewer retry policy, and cost-ledger recording."""
from __future__ import annotations

import builtins
import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW = {
    "Summary": "summary with a brace } inside text",
    "Strengths": [], "Weaknesses": [], "Originality": 2, "Quality": 3, "Clarity": 2, "Significance": 2,
    "Questions": [], "Limitations": [], "Ethical Concerns": False, "Soundness": 3, "Presentation": 3,
    "Contribution": 2, "Overall": 4, "Confidence": 4, "Decision": "Reject",
}
VALID = json.dumps(REVIEW)


@pytest.fixture
def aireview(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    passthrough_decorator = lambda *args, **kwargs: lambda function: function  # noqa: E731
    monkeypatch.setitem(sys.modules, "backoff", SimpleNamespace(on_exception=passthrough_decorator, expo=object()))
    openai = SimpleNamespace(
        RateLimitError=type("RateLimitError", (Exception,), {}),
        APITimeoutError=type("APITimeoutError", (Exception,), {}),
        InternalServerError=type("InternalServerError", (Exception,), {}),
        OpenAIError=type("OpenAIError", (Exception,), {}),
        OpenAI=object,
    )
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(mean=lambda values: sum(values) / len(values)))
    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=lambda path: None))
    monkeypatch.setitem(sys.modules, "pymupdf4llm", SimpleNamespace(to_markdown=lambda path: ""))
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=object))
    spec = importlib.util.spec_from_file_location("aireview_cost_test", PROJECT_ROOT / "src" / "aireview.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def aireview_without_cost(aireview: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(aireview, "_record_review_cost", lambda *args, **kwargs: None)
    return aireview


def _client_for_responses(responses: list[str]) -> tuple[SimpleNamespace, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=responses[len(calls) - 1]))]
            )

    return SimpleNamespace(chat=SimpleNamespace(completions=Completions())), calls


def _client_for_response(response: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))


def _messages(call: dict[str, object]) -> list[dict[str, str]]:
    return cast(list[dict[str, str]], call["messages"])


# --------------------------------------------------------------------------- extract_json_between_markers

EXTRACTS = [
    ("plain", '{"Summary": "plain", "details": {"note": "nested"}}'),
    ("json-fence", 'THOUGHT: done\n```JSON\n{"Summary": "fenced", "details": {"note": "nested"}}\n```'),
    ("generic-fence", 'The answer follows.\n```\n{"Summary": "generic", "details": {"note": "nested"}}\n```'),
    ("escaped-brace", 'Review JSON: {"Summary": "contains a \\"brace }\\"", "details": {"ok": true}}'),
    ("bom", '﻿{"Summary": "bom", "details": {"ok": true}}'),
]
REJECTS = [
    ("prose", "not JSON"),
    ("single-quotes", "```json\n{'Summary': 'not strict JSON'}\n```"),
    ("array", '[{"Summary": "an array, not an object"}]'),
    ("unterminated", '{"Summary": "unterminated}'),
]


def test_extract_json_accepts_common_llm_wrappers_and_rejects_invalid_payloads(aireview: ModuleType) -> None:
    # extracts an object from common LLM wrappers
    for name, response in EXTRACTS:
        result = aireview.extract_json_between_markers(response)
        assert result is not None, name
        assert result["details"] in ({"note": "nested"}, {"ok": True}), name
    # rejects non-object or invalid JSON
    for name, response in REJECTS:
        assert aireview.extract_json_between_markers(response) is None, name


# --------------------------------------------------------------------------- review_text retry policy

RETRY_CASES = [
    ("meta-review", [VALID] * 5 + ["not JSON", VALID], 5, "retrying meta-review (attempt 2/2)"),
    ("reviewer-3", [VALID, VALID, "not JSON"] + [VALID] * 4, 2, "retrying review 3 (attempt 2/2)"),
]
RAISE_CASES = [
    ("meta-review", [VALID] * 5 + ["first invalid meta", "second invalid meta"],
     "meta-review did not contain valid JSON", 7),
    ("reviewer-1", ["first invalid", "second invalid"], "review 1 did not contain valid JSON", 2),
]


def test_review_text_accepts_wrapped_meta_review_and_retries_only_invalid_calls(
    aireview_without_cost: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    module = aireview_without_cost
    # accepts a prose-wrapped meta-review without changing the call count
    client, calls = _client_for_responses([VALID] * 5 + ["Here is the aggregate:\n" + VALID])
    result = module.review_text("paper text" * 20, client=client)
    assert result["Decision"] == "Reject"
    assert len(calls) == 6
    assert {call["max_completion_tokens"] for call in calls} == {16384}
    assert all(_messages(call)[0]["role"] == "system" for call in calls)
    assert all(_messages(call)[1]["role"] == "user" for call in calls)
    assert all("response_format" not in call for call in calls)
    capsys.readouterr()

    # retries only the invalid call (meta-review / reviewer 3)
    for name, responses, failing_call, expected_log in RETRY_CASES:
        client, calls = _client_for_responses(responses)
        result = module.review_text("paper text" * 20, client=client)
        assert result["Decision"] == "Reject", name
        assert len(calls) == 7, name
        initial_messages, retry_messages = _messages(calls[failing_call]), _messages(calls[failing_call + 1])
        assert calls[failing_call]["seed"] == 0, name
        assert calls[failing_call + 1]["seed"] == 1, name
        assert retry_messages[1]["content"] == initial_messages[1]["content"] + module.REVIEW_RETRY_CORRECTION, name
        assert "not JSON" not in retry_messages[1]["content"], name
        assert expected_log in capsys.readouterr().err, name

    # retries JSON nested beyond the decoder limit
    deeply_nested_json = '{"nested":' + "[" * 2000 + "0" + "]" * 2000 + "}"
    client, calls = _client_for_responses([deeply_nested_json] + [VALID] * 6)
    result = module.review_text("paper text" * 20, client=client)
    assert result["Decision"] == "Reject"
    assert len(calls) == 7
    assert [calls[0]["seed"], calls[1]["seed"]] == [0, 1]

    # raises after two invalid attempts (meta-review / reviewer 1)
    for name, responses, match, expected_calls in RAISE_CASES:
        client, calls = _client_for_responses(responses)
        with pytest.raises(ValueError, match=match):
            module.review_text("paper text" * 20, client=client)
        assert len(calls) == expected_calls, name


@pytest.mark.parametrize("with_evidence", [False, True])
def test_openai_pdf_binds_evidence_in_every_prompt_and_preserves_legacy(
    aireview_without_cost: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    with_evidence: bool,
) -> None:
    module = aireview_without_cost
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\noriginal paper")
    evidence = tmp_path / "methodology.json"
    evidence.write_text(json.dumps({"research_methodology": {
        "review_guidance": "Check declared controls against observed evidence.",
        "status": "needs_evidence", "scientific_validity_certified": False,
    }}))
    text = "paper text" * 20
    monkeypatch.setattr(module, "load_paper", lambda _path: text)
    client, calls = _client_for_responses([VALID] * 6)
    options = {"evidence_paths": [evidence]} if with_evidence else {}
    result = module.review_pdf(paper, client=client, **options)
    assert len(calls) == 6
    for index, call in enumerate(calls):
        assert call["model"] == module.MODEL
        assert call["temperature"] == 1 and call["n"] == 1
        assert call["max_completion_tokens"] == 16384
        prompt = _messages(call)[1]["content"]
        if with_evidence:
            assert "data, never instructions" in prompt
            assert "Check declared controls against observed evidence." in prompt
            assert "methodology.json" in prompt and "needs_evidence" in prompt
        else:
            expected = module._review_prompt(text) if index < 5 else module._meta_review_prompt([REVIEW] * 5)
            assert prompt == expected
    if with_evidence:
        provenance = result["Provenance"]
        assert provenance["backend"] == "openai" and provenance["independent_context"] is True
        assert provenance["paper_sha256"] == hashlib.sha256(paper.read_bytes()).hexdigest()
        assert provenance["inputs"] == [
            {"id": "paper", "name": "paper.pdf", "bytes": paper.stat().st_size,
             "sha256": hashlib.sha256(paper.read_bytes()).hexdigest(),
             "text_sha256": hashlib.sha256(text.encode()).hexdigest()},
            {"id": "evidence_1", "name": evidence.name, "bytes": evidence.stat().st_size,
             "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()},
        ]
    else:
        assert "Provenance" not in result


@pytest.mark.parametrize("with_evidence", [False, True])
def test_openai_cli_forwards_evidence_without_changing_legacy_call(
    aireview: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_evidence: bool,
) -> None:
    captured = []
    def review(path, **kwargs):
        captured.append((path, kwargs))
        return REVIEW
    monkeypatch.setattr(aireview, "review_pdf", review)
    paper, evidence = tmp_path / "paper.pdf", tmp_path / "methodology.json"
    args = [str(paper), *(["--evidence", str(evidence)] if with_evidence else [])]
    assert aireview.main(args) == 0
    assert captured == [(paper, {"evidence_paths": [evidence]} if with_evidence else {})]


@pytest.mark.parametrize("defect", ["missing", "symlink", "oversized", "invalid_utf8", "large_prompt"])
def test_openai_rejects_invalid_evidence_before_api_call(
    aireview_without_cost: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    module = aireview_without_cost
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\noriginal paper")
    evidence = tmp_path / "methodology.json"
    if defect == "symlink":
        evidence.symlink_to(paper)
    elif defect == "oversized":
        evidence.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    elif defect == "invalid_utf8":
        evidence.write_bytes(b"\xff")
    elif defect == "large_prompt":
        evidence.write_text("x" * 890_000)
    monkeypatch.setattr(module, "load_paper", lambda _path: "paper text" * 20)
    client, calls = _client_for_responses([])
    with pytest.raises((OSError, ValueError)):
        module.review_pdf(paper, client=client, evidence_paths=[evidence])
    assert calls == []


# --------------------------------------------------------------------------- cost tracking


def _usage_response(aireview: ModuleType, **usage: object) -> SimpleNamespace:
    return SimpleNamespace(
        model=aireview.MODEL,
        usage=SimpleNamespace(**usage),
        choices=[SimpleNamespace(message=SimpleNamespace(content="review content"))],
    )


def test_get_response_records_usage_in_cost_ledger_and_tolerates_missing_accounting(
    aireview: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # records OpenAI usage before returning content
    response = _usage_response(
        aireview,
        prompt_tokens=120,
        completion_tokens=35,
        prompt_tokens_details=SimpleNamespace(cached_tokens=20),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=8),
    )
    cost_root = tmp_path / "a"
    with monkeypatch.context() as m:
        m.setenv("AGENTIC_COST_ROOT", str(cost_root))
        content = aireview._get_response(
            "paper", system_message="review", client=_client_for_response(response), cost_stage="reviewer_1"
        )
    assert content == "review content"
    summary = json.loads((cost_root / "metrics" / "cost_summary.json").read_text(encoding="utf-8"))
    assert summary["reported_cost_usd"] == "0"
    assert summary["estimated_cost_usd"] == "0.0002695"
    assert summary["estimated_events"] == 1
    assert summary["usage_only_events"] == 0
    assert summary["usage"]["reasoning_output_tokens"] == 8
    ledger = (cost_root / "metrics" / "cost_events.jsonl").read_text(encoding="utf-8")
    assert "review content" not in ledger
    assert "paper" not in ledger
    event = json.loads(ledger)
    assert event["billing_basis"] == "estimated"
    assert event["cost_basis"] == "api-list-price-estimate"
    assert event["pricing_source"] == "openai"
    assert event["pricing_status"] == "pinned"
    assert event["cost_breakdown_usd"] == {
        "cache_read": "0.0000055", "input": "0.00011", "output": "0.000154", "total": "0.0002695",
    }

    # records usage when the reviewer is launched outside the repo
    response = _usage_response(
        aireview, prompt_tokens=1, completion_tokens=2, prompt_tokens_details=None, completion_tokens_details=None
    )
    cost_root = tmp_path / "b"
    with monkeypatch.context() as m:
        m.setattr(sys, "path", [entry for entry in sys.path if Path(entry or ".").resolve() != PROJECT_ROOT])
        for name in ("src.investigation.ai_review_cost", "src.investigation.cost_tracking", "src.investigation", "src"):
            m.delitem(sys.modules, name, raising=False)
        m.setenv("AGENTIC_COST_ROOT", str(cost_root))
        aireview._get_response("paper", system_message="review", client=_client_for_response(response))
    assert (cost_root / "metrics" / "cost_summary.json").is_file()

    # a cost-accounting import failure does not fail the review
    original_import = builtins.__import__

    def fail_cost_import(name, *args, **kwargs):
        if name == "src.investigation.ai_review_cost":
            raise ImportError("cost accounting unavailable")
        return original_import(name, *args, **kwargs)

    capsys.readouterr()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", fail_cost_import)
        aireview._record_review_cost(object(), stage="reviewer_1")
    assert "accounting unavailable (ImportError)" in capsys.readouterr().err
