"""CLI discovery is an auditable suggestion source, never a search receipt."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.research.agent_literature import AgentLiteratureResumeError, build_agent_literature_backend
from src.research.config import LiteratureConfig, ModelSearchConfig


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def config():
    return LiteratureConfig(queries=["Closest optimization methods and matched controls"], max_papers=4,
                            model_search=ModelSearchConfig(timeout_seconds=17))


@pytest.fixture
def settings():
    return {"agent": "codex", "model": "fixture/model", "profile": "fixture-profile",
            "reasoning_effort": "low", "timeout_seconds": 120,
            "environment": {"EXAMPLE_API_KEY": "private-test-value"}}


def install_agent(monkeypatch, payload):
    calls = []

    def run(prompt, *, work_dir, **settings):
        calls.append({"prompt": prompt, "settings": deepcopy(settings)})
        work_dir.mkdir()
        (work_dir / "prompt.txt").write_text(prompt)
        data = {"finalMessage": json.dumps(payload), "nativeSessionId": "fixture-session",
                "model": settings["model"], "usage": {"inputTokens": 10, "outputTokens": 20}}
        raw = json.dumps({"protocolVersion": 1, "type": "result", "exitCode": 0, "data": data})
        (work_dir / "stdout.json").write_text(raw)
        (work_dir / "stderr.log").write_text("")
        transport = {"status": "completed", "agent": settings.get("agent", "codex"),
                     "model": settings.get("model"), "independent_context": True,
                     "native_session_id": data["nativeSessionId"], "observed_model": data["model"],
                     "prompt_sha256": digest(prompt.encode()), "stdout_sha256": digest(raw.encode()),
                     "stderr_sha256": digest(b""), "response_sha256": digest(data["finalMessage"].encode()),
                     "usage": data["usage"]}
        write(work_dir / "receipt.json", transport)
        return deepcopy(payload), transport

    monkeypatch.setattr("src.review_backend.run_fresh_agent_json", run)
    return calls


def paper(**updates):
    return {"title": "A fixed-budget optimization method", "url": "https://arxiv.org/abs/2401.12345",
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.12345.pdf"}, "year": 2024,
            "venue": "arXiv", **updates}


def test_one_call_uses_author_settings_and_remaining_deadline_but_no_downloads(config, settings, tmp_path, monkeypatch):
    calls = install_agent(monkeypatch, {"papers": [paper()], "limitations": []})
    monkeypatch.setattr("src.research.literature.download", lambda *_a, **_kw: pytest.fail("discovery must not download"))
    backend, backend_id = build_agent_literature_backend(settings)
    records, errors = backend(config, tmp_path)
    assert not errors and len(records) == len(calls) == 1
    assert records[0]["metadata_status"] == "agent_suggested_unverified"
    assert "search_sources" not in records[0]
    assert calls[0]["settings"] == {**settings, "timeout_seconds": 17}
    assert config.queries[0] in calls[0]["prompt"]
    assert '"max_papers": 4' in calls[0]["prompt"]
    assert "structured web-tool records" in calls[0]["prompt"]
    assert backend_id.startswith("agent-proposals-v1:") and len(backend_id) < 256
    assert "private-test-value" not in backend_id
    directory = tmp_path / "agent_literature"
    receipt = read(directory / "receipt.json")
    assert receipt["status"] == "complete" and receipt["structured_search_evidence"] is False
    assert read(directory / "proposed.json")["papers"][0] == paper()
    assert not any("private-test-value" in path.read_text() for path in directory.rglob("*") if path.is_file())


@pytest.mark.parametrize("url,pdf", [
    ("https://arxiv.org/abs/2401.12345v2", "https://arxiv.org/pdf/2401.12345v2.pdf"),
    ("https://arxiv.org/abs/hep-th/9901001", "https://arxiv.org/pdf/hep-th/9901001.pdf"),
    ("https://proceedings.mlr.press/v100/example.html", None),
])
def test_missing_pdf_has_only_stable_arxiv_mapping(config, settings, tmp_path, monkeypatch, url, pdf):
    install_agent(monkeypatch, {"papers": [paper(url=url, openAccessPdf=None)]})
    backend, _ = build_agent_literature_backend(settings)
    records, errors = backend(config, tmp_path)
    assert records[0]["openAccessPdf"] == ({"url": pdf} if pdf else None)
    assert bool(errors) is (pdf is None)
    assert records[0]["metadata_status"] == "agent_suggested_unverified"


@pytest.mark.parametrize("url", ["file:///tmp/paper.pdf", "http://localhost/paper", "http://127.0.0.1/paper",
    "http://[::1]/paper", "http://169.254.169.254/paper", "https://user:password@arxiv.org/abs/2401.12345",
    "https://internal.local/paper", "https://arxiv.org:bad/paper", "https://arxiv.org/paper\nnext"])
def test_invalid_or_private_urls_are_not_records(config, settings, tmp_path, monkeypatch, url):
    calls = install_agent(monkeypatch, {"papers": [paper(url=url)]})
    backend, _ = build_agent_literature_backend(settings)
    records, errors = backend(config, tmp_path)
    assert not records and errors and len(calls) == 1
    assert "invalid or non-public" in errors[0]["error"]


def test_prose_and_provider_claims_never_upgrade_metadata_or_fabricate_search_sources(config, settings, tmp_path, monkeypatch):
    untrusted = paper(metadata_status="publisher_verified", search_sources=[{"url": "https://arxiv.org"}],
                      availability="full_text", full_text="Claimed paper body", download_sha256="a" * 64)
    install_agent(monkeypatch, {"papers": [untrusted], "verified": True,
                               "limitations": ["No web search tool was available"]})
    backend, _ = build_agent_literature_backend(settings)
    records, errors = backend(config, tmp_path)
    assert records[0] == {**paper(), "metadata_status": "agent_suggested_unverified"}
    assert errors[0]["error"].startswith("Agent-reported limitation (unverified)")
    assert read(tmp_path / "agent_literature/proposed.json")["verified"] is True


def test_completed_transport_replays_zero_calls_and_preserves_all_bytes(config, settings, tmp_path, monkeypatch):
    calls = install_agent(monkeypatch, {"papers": [paper()]})
    backend, _ = build_agent_literature_backend(settings)
    first = backend(config, tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert backend(config, tmp_path) == first
    assert len(calls) == 1 and all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("artifact", ["prompt.txt", "proposed.json", "normalized.json", "transport/receipt.json",
                                     "transport/stdout.json", "transport/prompt.txt", "transport/stderr.log"])
def test_tampered_artifacts_fail_before_another_call(config, settings, tmp_path, monkeypatch, artifact):
    calls = install_agent(monkeypatch, {"papers": [paper()]})
    backend, _ = build_agent_literature_backend(settings)
    backend(config, tmp_path)
    path = tmp_path / "agent_literature" / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AgentLiteratureResumeError):
        backend(config, tmp_path)
    assert len(calls) == 1


def test_rehashed_proposals_must_still_match_raw_sdk_response(config, settings, tmp_path, monkeypatch):
    calls = install_agent(monkeypatch, {"papers": [paper()]})
    backend, _ = build_agent_literature_backend(settings)
    backend(config, tmp_path)
    directory = tmp_path / "agent_literature"
    write(directory / "proposed.json", {"papers": [paper(title="An invented replacement")]})
    receipt = read(directory / "receipt.json")
    receipt["files_sha256"]["proposed.json"] = digest((directory / "proposed.json").read_bytes())
    write(directory / "receipt.json", receipt)
    with pytest.raises(AgentLiteratureResumeError, match="differ from the SDK"):
        backend(config, tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["model", "environment", "query", "timeout"])
def test_configuration_changes_are_rejected_and_settings_have_distinct_backend_ids(config, settings, tmp_path, monkeypatch, change):
    calls = install_agent(monkeypatch, {"papers": [paper()]})
    backend, first_id = build_agent_literature_backend(settings)
    backend(config, tmp_path)
    if change == "model":
        settings["model"] = "changed/model"
    elif change == "environment":
        settings["environment"]["EXAMPLE_API_KEY"] = "changed-private-value"
    elif change == "query":
        config.queries = ["Different question"]
    else:
        config.model_search.timeout_seconds = 10
    updated, updated_id = build_agent_literature_backend(settings)
    assert (updated_id != first_id) is (change in {"model", "environment"})
    with pytest.raises(AgentLiteratureResumeError, match="configuration or prompt changed"):
        updated(config, tmp_path)
    assert len(calls) == 1


def test_failed_or_incomplete_invocations_never_silently_repeat(config, settings, tmp_path, monkeypatch):
    calls = []

    def failed(*_args, **_kwargs):
        calls.append(1)
        raise ValueError("provider failed with private-test-value")

    monkeypatch.setattr("src.review_backend.run_fresh_agent_json", failed)
    backend, _ = build_agent_literature_backend(settings)
    records, errors = backend(config, tmp_path)
    assert not records and "[REDACTED]" in errors[0]["error"]
    path = tmp_path / "agent_literature/receipt.json"
    assert read(path)["status"] == "failed"
    with pytest.raises(AgentLiteratureResumeError, match="incomplete or failed"):
        backend(config, tmp_path)
    receipt = read(path)
    receipt["status"] = "started"
    write(path, receipt)
    with pytest.raises(AgentLiteratureResumeError, match="incomplete or failed"):
        backend(config, tmp_path)
    assert len(calls) == 1


def test_max_papers_bounds_suggestions_without_extra_calls(config, settings, tmp_path, monkeypatch):
    config.max_papers = 1
    calls = install_agent(monkeypatch, {"papers": [paper(), paper(url="https://arxiv.org/abs/2401.54321")]})
    backend, _ = build_agent_literature_backend(settings)
    records, errors = backend(config, tmp_path)
    assert len(records) == len(calls) == 1 and "exceeded max_papers=1" in errors[0]["error"]
