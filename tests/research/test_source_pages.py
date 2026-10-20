"""Frozen PDF page tools: hash binding, MCP image delivery, replay and tamper rejection."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib

import pytest
from pypdf import PdfWriter

from src.research import source_lookup, source_pages
from src.research.agent_source_pages import (
    pdf_files_for_task, prepare_source_pages, verify_source_pages,
)
from src.research.agent_tool_profile import prepare_tool_profile


def packet_for(source_ids: tuple[str, ...] = ("primary",)) -> dict:
    rows = []
    for source_id in source_ids:
        text = f"Frozen full text for {source_id}."
        rows.append({"id": source_id, "title": f"Paper {source_id}", "availability": "full_text",
                     "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "full_text": text})
    return {"sources": rows}


def packet_sha(packet: dict) -> str:
    return source_lookup._sha(source_lookup._json(packet))


def write_pdf(path: Path, pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=240, height=320)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer.write(stream)
    return path.read_bytes()


def staged_environment(tmp_path: Path, name: str = "original-codex") -> dict[str, str]:
    home = tmp_path / name
    home.mkdir()
    (home / "fixture.config.toml").write_text(
        'model_provider = "fixture"\nmodel_catalog_json = "catalog.json"\nmcp_servers = {}\n'
        '[model_providers.fixture]\nbase_url = "http://127.0.0.1:12345/v1"\n',
    )
    (home / "catalog.json").write_text('{"models":[]}')
    (home / "auth.json").write_text('{"fixture_token":"private-fixture-token"}')
    return {"HOME": str(tmp_path), "CODEX_HOME": str(home), "PATH": os.environ["PATH"],
            "SECRET_UNRELATED": "do-not-copy"}


def prepare(tmp_path: Path, packet: dict, pdfs: dict, name: str = "call"):
    environment = staged_environment(tmp_path)
    artifacts = tmp_path / name
    artifacts.mkdir()
    (tmp_path / "tool-profile").mkdir()
    profile = prepare_tool_profile(temporary=tmp_path / "tool-profile", environment=environment,
                                   agent="codex", model="fixture-model", profile="fixture")
    prepared = prepare_source_pages(packet, pdfs, artifacts=artifacts, profile=profile)
    return artifacts, prepared


def run_tools(prepared, requests: list[tuple[str, dict]], *, environment=None) -> list[dict]:
    config = tomllib.loads(prepared.profile.path.read_text(encoding="utf-8"))
    specification = config["mcp_servers"]["agsci_pages"]
    process = subprocess.Popen(
        [specification["command"], *specification["args"]], cwd=specification["cwd"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
        env=dict(prepared.profile.environment if environment is None else environment),
    )
    assert process.stdin is not None and process.stdout is not None

    def rpc(number: int, method: str, params: dict | None = None) -> dict:
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": number, "method": method,
                                        "params": params or {}}) + "\n")
        process.stdin.flush()
        return json.loads(process.stdout.readline())

    assert "result" in rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                            "clientInfo": {"name": "fixture-client", "version": "1"}})
    assert "result" in rpc(2, "tools/list")
    results = []
    for number, (tool, arguments) in enumerate(requests, start=3):
        results.append(rpc(number, "tools/call", {"name": tool, "arguments": arguments})["result"])
    process.stdin.close()
    process.wait(timeout=30)
    assert process.returncode == 0, process.stderr.read() if process.stderr else ""
    return results


def one_page_prepared(tmp_path: Path, *, pages: int = 2, name: str = "call"):
    packet = packet_for()
    pdf = tmp_path / "source.pdf"
    body = write_pdf(pdf, pages)
    pdfs = {"primary": (pdf, hashlib.sha256(body).hexdigest())}
    artifacts, prepared = prepare(tmp_path, packet, pdfs, name)
    return packet, pdfs, artifacts, prepared


def test_page_tool_delivers_png_image_and_text_metadata(tmp_path: Path) -> None:
    packet, pdfs, artifacts, prepared = one_page_prepared(tmp_path)
    listed, page = run_tools(prepared, [
        ("list_source_pdfs", {}),
        ("read_source_pdf_page", {"source_id": "primary", "page": 1}),
    ])
    assert listed["isError"] is False
    catalogue = json.loads(listed["content"][0]["text"])
    assert catalogue["page_numbering"] == "one-based"
    assert catalogue["sources"][0]["pages"] == 2
    assert catalogue["semantic_fidelity_verified"] is False

    assert page["isError"] is False
    assert [block["type"] for block in page["content"]] == ["text", "image"]
    metadata = json.loads(page["content"][0]["text"])
    image = base64.b64decode(page["content"][1]["data"])
    assert page["content"][1]["mimeType"] == "image/png"
    assert metadata["image_sha256"] == hashlib.sha256(image).hexdigest()
    assert metadata["pdf_sha256"] == pdfs["primary"][1]
    assert metadata["semantic_fidelity_verified"] is False
    assert (metadata["width"], metadata["height"]) == source_pages.image_dimensions(image)

    receipt = prepared.finish()
    binding = verify_source_pages(artifacts, receipt, expected_packet_sha256=packet_sha(packet),
                                  expected_pdf_hashes={key: value[1] for key, value in pdfs.items()})
    assert binding["tool_calls"] == 2 and binding["tool_errors"] == 0
    assert binding["rendered_pages"] == [hashlib.sha256(b"primary").hexdigest() + "-0001"]
    assert binding["credential_files_exported"] is False
    assert binding["permission_mode"] == "read-only"
    assert binding["semantic_fidelity_verified"] is False


def test_frozen_pdf_bytes_must_match_the_manifest(tmp_path: Path) -> None:
    packet = packet_for()
    pdf = tmp_path / "source.pdf"
    write_pdf(pdf, 1)
    with pytest.raises(ValueError, match="frozen catalog"):
        prepare(tmp_path, packet, {"primary": (pdf, "0" * 64)})


def test_unknown_source_and_out_of_range_pages_are_rejected(tmp_path: Path) -> None:
    _, _, _, prepared = one_page_prepared(tmp_path)
    unknown, low, high = run_tools(prepared, [
        ("read_source_pdf_page", {"source_id": "other", "page": 1}),
        ("read_source_pdf_page", {"source_id": "primary", "page": 0}),
        ("read_source_pdf_page", {"source_id": "primary", "page": 3}),
    ])
    assert [result["isError"] for result in (unknown, low, high)] == [True, True, True]
    assert "Unknown frozen PDF source" in unknown["content"][0]["text"]
    assert "outside its supported integer range" in low["content"][0]["text"]


def test_image_policy_rejects_non_png_and_oversized_pages() -> None:
    with pytest.raises(ValueError, match="PNG"):
        source_pages.image_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    with pytest.raises(ValueError, match="PNG"):
        source_pages.image_dimensions(b"not an image at all, but long enough to read the header" * 2)
    for width, height in ((0, 10), (10, 0), (source_pages.MAX_DIMENSION + 1, 10)):
        header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")
        with pytest.raises(ValueError, match="dimensions exceed policy"):
            source_pages.image_dimensions(header + b"\x00" * 16)


def test_missing_renderer_is_recorded_and_replayed_without_rendering(tmp_path: Path) -> None:
    _, _, _, prepared = one_page_prepared(tmp_path)
    empty_path = tmp_path / "empty-bin"
    empty_path.mkdir()
    environment = {**prepared.profile.environment, "PATH": str(empty_path)}
    assert environment["PATH"] != os.environ.get("PATH")
    (page,) = run_tools(prepared, [("read_source_pdf_page", {"source_id": "primary", "page": 1})],
                        environment=environment)
    assert page["isError"] is True
    assert page["content"][0]["text"] == "PDF page rendering requires pdftoppm"
    # Replay must reproduce the observed failure text without launching a renderer.
    receipt = prepared.finish()
    assert receipt["policy"] == source_pages.POLICY


@pytest.mark.parametrize("tamper", ["page", "page_receipt", "manifest", "binding", "audit", "server_snapshot", "pdf"])
def test_tampered_page_evidence_is_rejected(tamper: str, tmp_path: Path) -> None:
    packet, pdfs, artifacts, prepared = one_page_prepared(tmp_path, name=f"tamper-{tamper}")
    run_tools(prepared, [("read_source_pdf_page", {"source_id": "primary", "page": 1})])
    receipt = prepared.finish()
    root = artifacts / "source_pages"
    targets = {
        "page": root / "pages" / (hashlib.sha256(b"primary").hexdigest() + "-0001") / "page.png",
        "page_receipt": root / "pages" / (hashlib.sha256(b"primary").hexdigest() + "-0001") / "receipt.json",
        "manifest": root / "manifest.json",
        "binding": root / "binding.json",
        "audit": root / "calls.jsonl",
        "server_snapshot": root / "server.py",
        "pdf": root / source_pages.pdf_name("primary"),
    }
    target = targets[tamper]
    if tamper == "page_receipt":
        metadata = json.loads(target.read_text())
        metadata["width"] = 1
        target.write_text(json.dumps(metadata))
    else:
        target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        verify_source_pages(artifacts, receipt, expected_packet_sha256=packet_sha(packet),
                            expected_pdf_hashes={key: value[1] for key, value in pdfs.items()})


def test_replayed_lookup_refuses_missing_rendered_page(tmp_path: Path) -> None:
    packet, _, _, prepared = one_page_prepared(tmp_path, name="replay")
    run_tools(prepared, [("read_source_pdf_page", {"source_id": "primary", "page": 1})])
    prepared.finish()
    root = prepared.artifacts / "source_pages"
    manifest = json.loads((root / "manifest.json").read_text())
    lookup = source_pages.SourcePageLookup(manifest, root, replay=True)
    page_directory = lookup.page_directory("primary", 2)
    assert not page_directory.exists()
    with pytest.raises(ValueError, match="no rendered PDF page"):
        lookup.call("read_source_pdf_page", {"source_id": "primary", "page": 2})


def test_page_artifacts_do_not_export_credentials(tmp_path: Path) -> None:
    _, _, artifacts, prepared = one_page_prepared(tmp_path, name="credentials")
    run_tools(prepared, [("read_source_pdf_page", {"source_id": "primary", "page": 1})])
    prepared.finish()
    root = artifacts / "source_pages"
    assert "auth.json" not in {path.name for path in artifacts.rglob("*")}
    for path in root.iterdir():
        if path.is_file():
            assert b"private-fixture-token" not in path.read_bytes()
            assert stat.S_IMODE(path.stat().st_mode) == 0o600


def catalog(task_dir: Path, *, raw_path: str | None) -> dict:
    root = task_dir / "research_context"
    (root / "text").mkdir(parents=True)
    (root / "text" / "paper.txt").write_text("Abstract text.")
    pdf = root / "raw" / "paper.pdf"
    body = write_pdf(pdf, 1)
    source = {"id": "paper", "title": "Paper", "availability": "abstract", "passages": [],
              "text_path": "text/paper.txt", "text_sha256": hashlib.sha256(b"Abstract text.").hexdigest(),
              "download_sha256": hashlib.sha256(body).hexdigest()}
    if raw_path is not None:
        source["raw_path"] = raw_path
    document = {"schema_version": 1, "status": "ready", "queries": [], "sources": [source]}
    (root / "catalog.json").write_text(json.dumps(document))
    return document


def test_catalog_without_raw_path_stays_compatible(tmp_path: Path) -> None:
    task = tmp_path / "task"
    catalog(task, raw_path=None)
    assert pdf_files_for_task(task) == {}


def test_catalog_raw_path_binds_the_original_pdf(tmp_path: Path) -> None:
    task = tmp_path / "task"
    document = catalog(task, raw_path="raw/paper.pdf")
    mapping = pdf_files_for_task(task)
    assert set(mapping) == {"paper"}
    path, digest = mapping["paper"]
    assert path == task / "research_context" / "raw" / "paper.pdf"
    assert digest == document["sources"][0]["download_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_page_tools_require_the_frozen_source_packet(tmp_path: Path) -> None:
    from src.review_backend import AgentReviewError, run_fresh_agent_json

    pdf = tmp_path / "source.pdf"
    body = write_pdf(pdf, 1)
    with pytest.raises(AgentReviewError, match="PDF page tools require the frozen source lookup packet"):
        run_fresh_agent_json("Read pages.", work_dir=tmp_path / "call", agent="codex", model="fixture",
                             profile="fixture", environment=staged_environment(tmp_path), timeout_seconds=5,
                             source_pdf_files={"primary": (pdf, hashlib.sha256(body).hexdigest())})


def test_renderer_timeout_is_not_replayed_as_a_render(tmp_path: Path, monkeypatch) -> None:
    from src.investigation.bounded_process import BoundedProcessError

    packet = packet_for()
    pdf = tmp_path / "source.pdf"
    body = write_pdf(pdf, 1)
    _, prepared = prepare(tmp_path, packet, {"primary": (pdf, hashlib.sha256(body).hexdigest())},
                          name="render-timeout")
    root = prepared.artifacts / "source_pages"
    manifest = json.loads((root / "manifest.json").read_text())

    def renderer(*args, **kwargs):
        raise BoundedProcessError("Renderer timed out after 20 seconds")

    monkeypatch.setattr(source_pages, "run_bounded_process", renderer)
    lookup = source_pages.SourcePageLookup(manifest, root)
    with pytest.raises(ValueError, match="rendering did not complete within policy"):
        lookup.call("read_source_pdf_page", {"source_id": "primary", "page": 1})

    replay = source_pages.SourcePageLookup(manifest, root, replay=True)
    with pytest.raises(ValueError, match="no rendered PDF page"):
        replay.call("read_source_pdf_page", {"source_id": "primary", "page": 1})


def test_invalid_pdf_manifest_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "pages"
    (root / "pages").mkdir(parents=True)
    pdf = write_pdf(tmp_path / "source.pdf", 1)
    row = {"id": "primary", "title": "Paper", "file": source_pages.pdf_name("primary"),
           "pdf_sha256": hashlib.sha256(pdf).hexdigest(), "bytes": len(pdf), "pages": 1}
    (root / row["file"]).write_bytes(pdf)
    valid = {"schema_version": 1, "policy": source_pages.POLICY,
             "source_packet_sha256": "0" * 64, "sources": [row]}
    source_pages.SourcePageLookup(valid, root)
    for mutation in ({"policy": "other"}, {"schema_version": 2},
                     {"sources": [{**row, "pages": 0}]},
                     {"sources": [{**row, "file": "elsewhere.pdf"}]}):
        with pytest.raises(ValueError):
            source_pages.SourcePageLookup({**valid, **mutation}, root)


def test_catalog_rejects_changed_raw_pdf(tmp_path: Path) -> None:
    from src.research.evidence import load_catalog

    task = tmp_path / "task"
    catalog(task, raw_path="raw/paper.pdf")
    assert load_catalog(task) is not None
    (task / "research_context" / "raw" / "paper.pdf").write_bytes(b"%PDF-1.4 changed bytes")
    with pytest.raises(ValueError, match="PDF changed"):
        load_catalog(task)
