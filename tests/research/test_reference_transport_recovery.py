"""A dropped model response uses remaining search budget without weakening citations."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from src.research import model_search
from src.research import paper_references as refs
from src.research.config import ModelSearchConfig, load_config, write_config
from tests.research.test_paper_reference_bridge import online_task, write


@contextmanager
def model_server(*, permanent_status: int | None = None):
    requests = []
    url = "https://arxiv.org/abs/2408.06292"
    payload = {"output": [
        {"type": "web_search_call", "status": "completed", "action": {"sources": [{"url": url}]}},
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps({
            "papers": [{"title": "Unverified model title", "url": url, "source_urls": [url]}],
        })}]},
    ]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if permanent_status is not None:
                self.send_error(permanent_status)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # The first accepted POST loses its response, just as the real case
            # did when its model connection disconnected during literature search.
            self.wfile.write(body[:12] if len(requests) == 1 else body)
            self.wfile.flush()
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def configured_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str) -> Path:
    task = online_task(tmp_path / "task", monkeypatch)
    cfg = load_config(task)
    cfg.literature.model_search = ModelSearchConfig(
        provider="magpie", model="transport-test-model", base_url=endpoint,
        mode="native", api_format="responses", timeout_seconds=2,
    )
    cfg.literature.proxy_fallback = ""
    write_config(task / "research.yaml", cfg)
    monkeypatch.setattr(refs, "search_literature", model_search.search_literature)
    return task


def test_interrupted_response_keeps_failed_trace_and_recovers_in_next_round(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(refs.time, "sleep", waits.append)
    with model_server() as (endpoint, requests):
        task = configured_task(tmp_path, monkeypatch, endpoint)
        frozen = (task / "research.yaml").read_bytes()
        output = tmp_path / "paper"
        assert write(task, output) == 1
    assert len(requests) == 2
    assert waits == [15]
    assert (task / "research.yaml").read_bytes() == frozen
    summary = json.loads((output / "references_provenance.json").read_text())
    assert summary["status"] == "ready" and summary["count"] == 1
    assert len(summary["search_rounds"]) == 2
    first = json.loads((output / summary["search_rounds"][0]["search_trace"]).read_text())
    assert first["status"] == "unavailable" and len(first["calls"]) == 1
    assert first["errors"][0]["error_kind"] == "transport"
    assert summary["search_rounds"][0]["transport_recovery"]["prior_request_usage"] == "unknown"
    bib = (output / "references.bib").read_text()
    assert "A source title" in bib and "Doe, Jane and Smith, Alex" in bib
    assert "Unverified model title" not in bib


def test_transport_recovery_stops_at_existing_round_budget(tmp_path, monkeypatch):
    task = configured_task(tmp_path, monkeypatch, "http://127.0.0.1:9/v1")
    request = Mock(side_effect=model_search.ModelTransportError("test connection lost"))
    monkeypatch.setattr(model_search, "_http_json", request)
    waits = []
    monkeypatch.setattr(refs.time, "sleep", waits.append)
    output = tmp_path / "paper"
    output.mkdir()
    (output / "references.bib").write_text("preserved bibliography")
    with pytest.raises(ValueError, match="Only 0/1"):
        write(task, output)
    assert request.call_count == refs._MAX_REFERENCE_SEARCH_ROUNDS == 3
    assert waits == [15, 30]
    assert (output / "references.bib").read_text() == "preserved bibliography"
    summary = json.loads(next(output.glob("reference_search/*/attempt_*/result.json")).read_text())
    assert summary["status"] == "exhausted" and len(summary["search_rounds"]) == 3
    assert all((output / row["search_trace"]).is_file() for row in summary["search_rounds"])


@pytest.mark.parametrize("status", [401, 403, 429])
def test_http_policy_errors_do_not_trigger_recovery_rounds(tmp_path, monkeypatch, status):
    waits = []
    monkeypatch.setattr(refs.time, "sleep", waits.append)
    with model_server(permanent_status=status) as (endpoint, requests):
        task = configured_task(tmp_path, monkeypatch, endpoint)
        with pytest.raises(ValueError, match="Only 0/1"):
            write(task, tmp_path / "paper")
    assert len(requests) == 1 and waits == []


def test_error_text_alone_cannot_authorize_transport_recovery(tmp_path, monkeypatch):
    task = configured_task(tmp_path, monkeypatch, "http://127.0.0.1:9/v1")
    request = Mock(side_effect=ValueError("API response was interrupted; request was not replayed"))
    monkeypatch.setattr(model_search, "_http_json", request)
    waits = []
    monkeypatch.setattr(refs.time, "sleep", waits.append)
    with pytest.raises(ValueError, match="Only 0/1"):
        write(task, tmp_path / "paper")
    assert request.call_count == 1 and waits == []
