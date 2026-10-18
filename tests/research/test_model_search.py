from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
from types import SimpleNamespace

import pytest
import yaml

from src.research import cli, literature, model_search
from src.research.config import ExternalSearchConfig, LiteratureConfig, ModelSearchConfig, validate_model_config
from src.research.providers import PROVIDERS

URL = "https://arxiv.org/abs/2408.06292"
SECRET = "fixture-api-secret-do-not-save"
PAPERS = {"papers": [{"title": "The AI Scientist", "year": 2024, "url": URL,
                      "summary": "A model summary, not original paper text.", "source_urls": [URL]}]}


@contextmanager
def server(respond):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.do_POST()

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            calls.append({"path": self.path, "body": json.loads(raw) if raw else None, "headers": dict(self.headers)})
            status, payload, headers = respond(calls)
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}", calls
    finally:
        service.shutdown()
        service.server_close()
        thread.join(timeout=2)


def config(base, provider="openai", **kwargs):
    return LiteratureConfig(
        queries=["automated scientific discovery"], fetch_full_text=False, proxy_fallback="",
        model_search=ModelSearchConfig(provider=provider, model="fixture-model", base_url=base,
                                       api_key_env="TEST_RESEARCH_KEY", **kwargs),
    )


def native_payload(provider, text):
    citation = {"type": "url_citation", "url": URL, "title": "The AI Scientist"}
    if provider in {"openai", "doubao", "xai", "perplexity"}:
        output = [{"type": "message", "content": [{"type": "output_text", "text": text, "annotations": [citation]}]}]
        if provider == "perplexity":
            output[0]["content"][0]["annotations"] = []
            output.insert(0, {"type": "search_results", "results": [{"url": URL, "title": "The AI Scientist"}]})
        else:
            output.insert(0, {"type": "web_search_call", "status": "completed", "action": {"sources": [{"url": URL}]}})
        return {"status": "completed", "output": output, "usage": {"input_tokens": 55, "output_tokens": 10}}
    if provider == "anthropic":
        return {"stop_reason": "end_turn", "content": [
            {"type": "web_search_tool_result", "content": [{"type": "web_search_result", "url": URL, "title": "The AI Scientist"}]},
            {"type": "text", "text": text}], "usage": {"input_tokens": 55}}
    if provider == "gemini":
        return {"steps": [{"type": "google_search_call", "arguments": {"queries": ["AI scientist"]}},
                          {"type": "model_output", "content": [{"type": "text", "text": text, "annotations": [citation]}]}]}
    if provider == "gemini_generate_content":
        return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP",
                                "groundingMetadata": {"groundingChunks": [{"web": {"uri": URL}}]}}]}
    if provider == "dashscope":
        return {"output": {"choices": [{"message": {"content": text}}],
                           "search_info": {"search_results": [{"url": URL, "title": "The AI Scientist"}]}}, "usage": {"total_tokens": 65}}
    if provider == "zhipu":
        return {"choices": [{"message": {"content": text}}], "web_search": [{"link": URL}]}
    return {"choices": [{"message": {"content": text, "annotations": [{"type": "url_citation", "url_citation": {"url": URL}}]}}]}


@pytest.mark.parametrize("provider", [name for name, spec in PROVIDERS.items() if spec.mode == "native"])
def test_native_api_http_and_grounded_paper_extraction(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    payload = native_payload(provider, "```json\n" + json.dumps(PAPERS) + "\n```")
    with server(lambda _calls: (200, payload, {})) as (base, calls):
        task = tmp_path / "task"
        task.mkdir()
        catalog = literature.prepare_literature(task, config(base, provider))
    assert len(calls) == 1
    source = catalog["sources"][0]
    assert source["url"] == URL
    assert source["availability"] == "metadata"
    assert source["text_path"] is None
    assert source["passages"] == []
    assert source["metadata_status"] == "model_extracted_unverified"
    assert source["model_summary"].startswith("A model summary")
    trace = json.loads((task / "research_context/model_search/trace.json").read_text())
    assert trace["sources"][0]["url"] == URL
    assert trace["status"] == "ready"
    raw_files = list((task / "research_context/model_search").glob("*"))
    assert all(SECRET not in path.read_text() for path in raw_files)
    body = calls[0]["body"]
    if provider == "openrouter":
        assert body["tools"] == [{"type": "openrouter:web_search"}]
    elif provider == "dashscope":
        assert body["parameters"]["search_options"]["enable_source"] is True
    elif provider == "anthropic":
        assert body["tools"][0]["max_uses"] == 4
        assert {key.lower(): value for key, value in calls[0]["headers"].items()}["x-api-key"] == SECRET
    elif provider == "gemini":
        assert calls[0]["path"] == "/interactions"
        assert body["tools"] == [{"type": "google_search"}]


@pytest.mark.parametrize("backend", ["searxng", "tavily", "brave", "zhipu", "deepseek"])
def test_plain_chat_model_plans_real_search_then_extracts(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    monkeypatch.setenv("TEST_SEARCH_KEY", "fixture-search-secret")
    def respond(calls):
        if len(calls) == 1:
            assert "tools" not in calls[-1]["body"]
            return 200, {"choices": [{"message": {"content": '{"queries":["AI Scientist original paper"]}'}}]}, {}
        if len(calls) == 2:
            assert "AI Scientist" in json.dumps(calls[-1]["body"]) or "AI+Scientist" in calls[-1]["path"]
            if backend == "deepseek":
                assert calls[-1]["path"] == "/v1/tools/search"
                assert calls[-1]["body"] == {"query": "AI Scientist original paper", "count": 5, "need_content": False}
                assert calls[-1]["headers"]["Authorization"] == "Bearer fixture-search-secret"
            rows = [{"url": URL, "title": "The AI Scientist", "content": "Original publication"}]
            payload = {"web": {"results": rows}} if backend == "brave" else {"search_result" if backend == "zhipu" else "results": rows}
            return 200, payload, {}
        assert URL in calls[-1]["body"]["messages"][0]["content"]
        return 200, {"choices": [{"message": {"content": json.dumps(PAPERS)}}]}, {}
    with server(respond) as (base, calls):
        cfg = config(base, "deepseek", external_search=ExternalSearchConfig(provider=backend, base_url=base, api_key_env="TEST_SEARCH_KEY"))
        papers, errors = model_search.search_literature(cfg, tmp_path)
    assert len(calls) == 3
    assert not errors
    assert papers[0]["url"] == URL
    assert papers[0]["openAccessPdf"]["url"] == "https://arxiv.org/pdf/2408.06292"
    assert "abstract" not in papers[0]


def test_native_prose_is_parsed_without_starting_another_search(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    def respond(calls):
        if len(calls) == 1:
            return 200, native_payload("openai", f"The AI Scientist (2024), source: {URL}"), {}
        assert "tools" not in calls[-1]["body"]
        return 200, {"output_text": json.dumps(PAPERS)}, {}
    with server(respond) as (base, calls):
        papers, errors = model_search.search_literature(config(base), tmp_path)
    assert len(calls) == 2 and not errors and len(papers) == 1
    assert (tmp_path / "model_search/01_native_search.json").is_file()
    assert (tmp_path / "model_search/02_parse_answer.json").is_file()


def test_plausible_model_text_without_search_receipts_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    with server(lambda _calls: (200, {"output_text": json.dumps(PAPERS)}, {})) as (base, calls):
        papers, errors = model_search.search_literature(config(base), tmp_path)
    assert papers == [] and len(calls) == 1
    assert "No structured search sources" in errors[0]["error"]


def test_fabricated_urls_and_unverified_pdf_are_not_accepted():
    papers, errors = model_search.normalize_papers({"papers": [{
        "title": "Invented", "url": "https://example.org/invented", "source_urls": [],
    }]}, [{"url": URL}], 5)
    assert not papers and errors
    row = {**PAPERS["papers"][0], "pdf_url": "https://example.org/invented.pdf"}
    papers, errors = model_search.normalize_papers({"papers": [row]}, [{"url": URL}], 5)
    assert papers[0]["openAccessPdf"]["url"] != row["pdf_url"] and errors


@pytest.mark.parametrize("status", [302, 401, 429])
def test_failed_requests_are_not_replayed_or_leak_keys(tmp_path, monkeypatch, status):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    with server(lambda _calls: (status, {"error": SECRET}, {"Location": "/do-not-follow"})) as (base, calls):
        papers, errors = model_search.search_literature(config(base), tmp_path)
    assert not papers and len(calls) == 1
    assert str(status) in errors[0]["error"]
    assert SECRET not in json.dumps(errors)
    assert SECRET not in (tmp_path / "model_search/trace.json").read_text()


def test_missing_key_is_inspectable_and_never_calls_network(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_RESEARCH_KEY", raising=False)
    monkeypatch.setattr(model_search, "_http_json", lambda *_args, **_kw: pytest.fail("must not call network"))
    papers, errors = model_search.search_literature(config("https://example.org/v1"), tmp_path)
    assert not papers and "TEST_RESEARCH_KEY" in errors[0]["error"]
    assert json.loads((tmp_path / "model_search/trace.json").read_text())["status"] == "unavailable"


def test_anthropic_pause_preserves_search_results_and_has_a_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    first = native_payload("anthropic", "Working")
    first["stop_reason"] = "pause_turn"
    second = {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(PAPERS)}]}
    def respond(calls):
        if len(calls) == 2:
            assert calls[-1]["body"]["messages"][-1]["content"] == first["content"]
        return 200, first if len(calls) == 1 else second, {}
    with server(respond) as (base, calls):
        papers, errors = model_search.search_literature(config(base, "anthropic", max_continuations=1), tmp_path)
    assert len(calls) == 2 and papers and not errors


def test_all_provider_templates_are_valid_and_cli_does_not_write_secrets(capsys, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    assert len(PROVIDERS) == 28
    for name in PROVIDERS:
        assert cli.main(["providers", "--config", name]) == 0
        text = capsys.readouterr().out
        assert SECRET not in text
        raw = yaml.safe_load(text)["literature"]["model_search"]
        ext = raw.pop("external_search")
        if name == "deepseek":
            assert ext["provider"] == "deepseek"
        validate_model_config(ModelSearchConfig(**raw, external_search=ExternalSearchConfig(**ext)))
    assert cli.main(["providers"]) == 0
    assert "deepseek" in capsys.readouterr().out


@pytest.mark.parametrize("override", [{"api_key": "wrong"}, {"messages": []}, {"stream": True}, {"extra": {"authorization": "secret"}}])
def test_config_rejects_inline_auth_and_input_replacement(override):
    with pytest.raises(ValueError, match="cannot replace"):
        validate_model_config(ModelSearchConfig(request_overrides=override))


def test_source_urls_in_tool_arguments_are_not_search_proof():
    assert model_search.source_receipts({"output": [{"type": "function_call", "arguments": {
        "search_results": [{"url": URL}],
    }}]}) == []
    assert model_search.source_receipts({"output_text": json.dumps({"search_results": [{"url": URL}]})}) == []


def test_missing_external_backend_fails_before_spending_model_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    monkeypatch.setattr(model_search, "_http_json", lambda *_args, **_kw: pytest.fail("must not call model before backend configuration"))
    papers, errors = model_search.search_literature(config("https://example.org", "deepseek"), tmp_path)
    assert not papers and "external_search.base_url" in errors[0]["error"]


def test_response_key_echo_is_removed_from_catalog_too(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    content = {"papers": [{**PAPERS["papers"][0], "summary": SECRET}]}
    with server(lambda _calls: (200, native_payload("openai", json.dumps(content)), {})) as (base, _calls):
        task = tmp_path / "task"
        task.mkdir()
        catalog = literature.prepare_literature(task, config(base))
    assert catalog["sources"][0]["model_summary"] == "[REDACTED]"
    assert all(SECRET not in path.read_text() for path in task.rglob("*") if path.is_file())


def test_reasoning_blocks_do_not_override_final_json():
    text = '<think>{"queries":["wrong"]}</think>{"queries":["actual final query"]}'
    assert model_search.parse_object(text, "queries")["queries"] == ["actual final query"]


def test_custom_gateway_path_auth_and_parameter_removal(monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    cfg = config("https://example.org", "custom", api_format="responses", mode="native",
                 api_path="/openai/v1/responses?api-version=preview", api_key_header="api-key",
                 request_overrides={"max_output_tokens": None, "temperature": 0.2}).model_search
    validate_model_config(cfg)
    url, body, headers = model_search.build_request(cfg, "A query", native=True)
    assert url.endswith("responses?api-version=preview")
    assert headers == {"api-key": SECRET}
    assert "max_output_tokens" not in body and body["temperature"] == 0.2
    cfg.api_path = "/responses?api_key=inline-secret"
    with pytest.raises(ValueError, match="environment|api_key_env"):
        validate_model_config(cfg)


@pytest.mark.parametrize("host", ["host.docker.internal", "gateway.docker.internal"])
def test_docker_desktop_host_api_allows_local_http(monkeypatch, host):
    called = []
    def open_request(request, timeout):
        called.append(request.full_url)
        return BytesIO(b'{"status":"completed"}')
    monkeypatch.setattr(model_search.urllib.request, "build_opener",
                        lambda *_: SimpleNamespace(open=open_request))
    url = f"http://{host}:3425/v1/responses"
    assert model_search._http_json(url, {}, {}, timeout=5, proxy="", fallback="") == {"status": "completed"}
    assert called == [url]


def test_interrupted_response_is_recordable_and_never_replayed(monkeypatch):
    import http.client

    calls = []

    class InterruptedBody(BytesIO):
        def read(self, *_args):
            raise http.client.IncompleteRead(b'partial', 100)

    def open_request(request, timeout):
        calls.append(request.full_url)
        return InterruptedBody()

    monkeypatch.setattr(model_search.urllib.request, "build_opener",
                        lambda *_: SimpleNamespace(open=open_request))
    with pytest.raises(ValueError, match="interrupted; request was not replayed"):
        model_search._http_json("https://example.org/v1/responses", {}, {},
                                timeout=5, proxy="", fallback="http://127.0.0.1:7890")
    assert calls == ["https://example.org/v1/responses"]


@pytest.mark.parametrize("host", ["api.example.org", "192.168.1.10", "10.0.0.1",
                                  "host.docker.internal.example.org", "anything.internal"])
def test_remote_or_lookalike_http_api_is_rejected_before_network(monkeypatch, host):
    monkeypatch.setattr(model_search.urllib.request, "build_opener",
                        lambda *_: pytest.fail("must reject before opening a connection"))
    with pytest.raises(ValueError, match="require HTTPS"):
        model_search._http_json(f"http://{host}/v1/responses", {}, {}, timeout=5, proxy="", fallback="")


@pytest.mark.parametrize("model", ["deepseek/deepseek-flash", "codex/gpt-6.1-sol"])
def test_magpie_model_can_switch_without_provider_code_changes(model):
    cfg = ModelSearchConfig(provider="magpie", model=model, mode="native")
    validate_model_config(cfg)
    url, body, headers = model_search.build_request(cfg, "Search original papers", native=True)
    assert url == "http://127.0.0.1:3425/v1/responses"
    assert body["model"] == model
    assert body["tools"] == [{"type": "web_search"}]
    assert headers == {}


def test_deepseek_search_default_endpoint_and_key_are_separate_from_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    cfg = LiteratureConfig(results_per_query=2, model_search=ModelSearchConfig(
        provider="magpie", model="deepseek/deepseek-flash",
        external_search=ExternalSearchConfig(provider="deepseek", max_results=5),
    ))
    session = model_search.SearchSession(cfg, tmp_path / "search")
    calls = []
    def request(url, body, headers, **_kwargs):
        calls.append((url, body, headers))
        return {"results": [{"url": URL, "title": "The AI Scientist", "summary": "Original paper result"}]}
    monkeypatch.setattr(model_search, "_http_json", request)
    sources = session.external("文" * 128)
    assert calls == [("https://api.deepseek.com/v1/tools/search",
                      {"query": "文" * 128, "count": 2, "need_content": False},
                      {"Authorization": "Bearer " + SECRET})]
    assert sources == [{"url": URL, "title": "The AI Scientist", "snippet": "Original paper result"}]
    assert session.calls[0]["search_provider"] == "deepseek"
    assert session.calls[0]["search_query"] == "文" * 128
    assert (tmp_path / "search/01_external_search.json").is_file()


@pytest.mark.parametrize("query", ["", "  ", "文" * 129])
def test_deepseek_search_rejects_invalid_query_without_truncation_or_network(tmp_path, monkeypatch, query):
    session = model_search.SearchSession(LiteratureConfig(model_search=ModelSearchConfig(
        external_search=ExternalSearchConfig(provider="deepseek"))), tmp_path / "search")
    monkeypatch.setattr(model_search, "_http_json", lambda *_args, **_kwargs: pytest.fail("must not call network"))
    with pytest.raises(ValueError, match="1 to 128 Unicode"):
        session.external(query)


def test_deepseek_model_plan_is_bounded_before_search(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RESEARCH_KEY", SECRET)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-search-secret")
    def respond(calls):
        assert len(calls) == 1
        assert "128 Unicode characters" in calls[0]["body"]["messages"][0]["content"]
        return 200, {"choices": [{"message": {"content": json.dumps({"queries": ["文" * 129]})}}]}, {}
    with server(respond) as (base, calls):
        cfg = config(base, "deepseek", external_search=ExternalSearchConfig(provider="deepseek", base_url=base))
        papers, errors = model_search.search_literature(cfg, tmp_path)
    assert not papers and len(calls) == 1
    assert "at most 128 characters" in errors[0]["error"]
