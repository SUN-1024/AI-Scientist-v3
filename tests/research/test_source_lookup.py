"""Frozen-byte navigation and real MCP stdio, without a model invocation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.research.evidence import _passage_span
from src.research.source_lookup import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    SourceLookup,
    _read_packet,
    serve,
    tools,
)


def packet(text: str = "α🙂\nA\x00B\r\nsame same") -> dict:
    return {"sources": [{"id": "primary", "title": "Primary text", "availability": "full_text",
                         "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "full_text": text}]}


def write_packet(tmp_path: Path, value: dict) -> tuple[Path, str]:
    body = json.dumps(value, ensure_ascii=False).encode()
    path = tmp_path / "packet.json"
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def test_exact_unicode_offsets_roundtrip_through_existing_citation_validator() -> None:
    data = packet()
    lookup = SourceLookup(data)
    result = lookup.call("locate_source_quote", {"source_id": "primary", "quote": "A\x00B\r\n"})
    assert result["status"] == "unique"
    span = result["matches"][0]
    assert span["start"] == 3  # Unicode code points, not UTF-8 byte offsets.
    assert _passage_span(span["passage_id"], "primary", data["sources"][0]["full_text"]) == span["quote"]
    assert lookup.call("read_source_span", {"source_id": "primary", "start": 3, "end": 8}) == span
    assert result["meaning_verified"] is False


def test_multiple_and_overlapping_matches_are_explicit_and_not_auto_selected() -> None:
    lookup = SourceLookup(packet("aaa"))
    result = lookup.call("locate_source_quote", {"source_id": "primary", "quote": "aa", "limit": 1})
    assert result["status"] == "ambiguous"
    assert result["total_matches"] == 2 and result["returned_matches"] == 1
    assert result["matches"][0]["quote"] == "aa"
    assert lookup.call("locate_source_quote", {"source_id": "primary", "quote": "a a"})["status"] == "not_found"


def test_no_whitespace_case_or_unicode_normalization() -> None:
    lookup = SourceLookup(packet("Line\r\né"))
    for absent in ("line", "Line\né", "e\u0301"):
        assert lookup.call("locate_source_quote", {"source_id": "primary", "quote": absent})["status"] == "not_found"


def test_search_window_preserves_long_query_and_validates_as_an_original_span() -> None:
    body = "prefix " + "x" * 3990 + " suffix"
    result = SourceLookup(packet(body)).call(
        "find_source_passages", {"source_id": "primary", "query": "x" * 3990, "limit": 1},
    )
    span = result["matches"][0]
    assert len(span["quote"]) <= 4000
    assert "x" * 3990 in span["quote"]
    assert _passage_span(span["passage_id"], "primary", body) == span["quote"]


@pytest.mark.parametrize("name,args", [
    ("read_source_span", {"source_id": "../../outside", "start": 0, "end": 3}),
    ("read_source_span", {"source_id": "primary", "start": False, "end": 3}),
    ("read_source_span", {"source_id": "primary", "start": 0, "end": 4001}),
    ("read_source_span", {"source_id": "primary", "start": 2, "end": 1}),
    ("find_source_passages", {"source_id": "primary", "query": ""}),
    ("find_source_passages", {"source_id": "primary", "query": "same", "limit": True}),
    ("locate_source_quote", {"source_id": "primary", "quote": "same", "path": "/outside"}),
    ("execute", {"command": "true"}),
    ("list_sources", {"path": "/outside"}),
])
def test_tool_arguments_cannot_expand_the_frozen_source_capability(name: str, args: dict) -> None:
    with pytest.raises(ValueError):
        SourceLookup(packet()).call(name, args)


def test_duplicate_source_ids_changed_text_and_missing_complete_body_are_rejected() -> None:
    data = packet()
    with pytest.raises(ValueError, match="unique"):
        SourceLookup({"sources": data["sources"] * 2})
    data["sources"][0]["full_text"] += "changed"
    with pytest.raises(ValueError, match="text/hash"):
        SourceLookup(data)
    del data["sources"][0]["full_text"]
    with pytest.raises(ValueError, match="missing"):
        SourceLookup(data)


def test_unavailable_or_partial_source_is_listed_but_not_detailed_citation_evidence() -> None:
    data = packet()
    data["sources"][0]["availability"] = "partial_text"
    lookup = SourceLookup(data)
    assert lookup.call("list_sources", {})["sources"][0]["availability"] == "partial_text"
    with pytest.raises(ValueError, match="complete"):
        lookup.call("locate_source_quote", {"source_id": "primary", "quote": "same"})


def test_packet_file_hash_and_path_are_checked_before_loading(tmp_path: Path) -> None:
    path, digest = write_packet(tmp_path, packet())
    assert _read_packet(path, digest) == path.read_bytes()
    with pytest.raises(ValueError, match="mismatch"):
        _read_packet(path, "0" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        _read_packet(link, digest)
    directory = tmp_path / "linked-directory"
    directory.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _read_packet(directory / path.name, digest)
    path.write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        _read_packet(path, digest)


def test_mcp_stdio_lifecycle_errors_and_tool_results_have_only_protocol_output() -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 0, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": "init", "method": "initialize",
         "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "locate_source_quote", "arguments": {"source_id": "primary", "quote": "same"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "read_source_span", "arguments": {"source_id": "/outside", "start": 0, "end": 1}}},
        {"jsonrpc": "2.0", "id": 5, "method": "run"},
        {"jsonrpc": "2.0", "id": 6, "method": "ping"},
    ]
    incoming = io.BytesIO(("\n".join(json.dumps(r) for r in requests) + "\n").encode())
    outgoing = io.BytesIO()
    events = []
    serve(SourceLookup(packet()), incoming, outgoing, packet_sha256="frozen", audit=events.append)
    responses = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert len(responses) == 7
    assert "error" in responses[0]
    assert responses[1]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert responses[2]["result"]["tools"] == tools()
    assert json.loads(responses[3]["result"]["content"][0]["text"])["status"] == "ambiguous"
    assert responses[4]["result"]["isError"] is True
    assert responses[5]["error"]["code"] == -32601
    assert responses[6]["result"] == {}
    assert [e["tool"] for e in events if e["event"] == "source_lookup_call"] == ["locate_source_quote", "read_source_span"]
    assert any(e["event"] == "source_lookup_initialized" for e in events)
    assert any(e["event"] == "source_lookup_tools_listed" for e in events)
    assert all(t["annotations"]["readOnlyHint"] for t in tools())


def test_real_stdio_process_preserves_packet_and_records_actual_tool_call(tmp_path: Path) -> None:
    path, digest = write_packet(tmp_path, packet())
    log = tmp_path / "audit.jsonl"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "locate_source_quote", "arguments": {"source_id": "primary", "quote": "A\x00B"}}},
    ]
    run = subprocess.run(
        [sys.executable, "-m", "src.research.source_lookup", "--packet", str(path),
         "--packet-sha256", digest, "--audit-log", str(log)],
        input="".join(json.dumps(r) + "\n" for r in requests), capture_output=True, text=True, timeout=10,
    )
    assert run.returncode == 0, run.stderr
    assert run.stderr == ""
    results = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(results) == 2 and results[1]["result"]["isError"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["event"] == "source_lookup_started"
    calls = [e for e in events if e["event"] == "source_lookup_call"]
    assert calls[0]["tool"] == "locate_source_quote" and calls[0]["packet_sha256"] == digest


def test_oversized_or_duplicate_field_protocol_messages_fail_without_running_a_tool() -> None:
    with pytest.raises(ValueError, match="bound"):
        serve(SourceLookup(packet()), io.BytesIO(b"x" * (MAX_MESSAGE_BYTES + 1)),
              io.BytesIO(), packet_sha256="frozen")
    output = io.BytesIO()
    serve(SourceLookup(packet()), io.BytesIO(b'{"jsonrpc":"2.0","id":1,"id":2,"method":"initialize"}\n'),
          output, packet_sha256="frozen")
    assert "Duplicate" in json.loads(output.getvalue())["error"]["message"]
