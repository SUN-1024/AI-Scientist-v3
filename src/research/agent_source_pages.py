"""Per-call frozen PDF page tools and offline verification of delivered images."""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import sys
from typing import Any, Mapping

from src.investigation import bounded_process
from . import agent_tool_profile, source_lookup, source_pages
from .agent_tool_profile import PrivateToolProfile, _private, _regular
from .source_lookup import _json, _parse, _sha

DIRECTORY = "source_pages"
PDFs = Mapping[str, tuple[Path, str]]


def pdf_files_for_task(task_dir: Path) -> dict[str, tuple[Path, str]]:
    from .evidence import load_catalog
    from .literature import CONTEXT_DIRECTORY

    catalog = load_catalog(task_dir)
    return {row["id"]: (task_dir / CONTEXT_DIRECTORY / row["raw_path"], row["download_sha256"])
            for row in (catalog or {}).get("sources", []) if row.get("raw_path")}


def _modules() -> dict[str, Path]:
    return {"server.py": Path(source_pages.__file__), "protocol.py": Path(source_lookup.__file__),
            "bounded_process.py": Path(bounded_process.__file__), "file_io.py": Path(agent_tool_profile.__file__)}


def _error(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _events(root: Path, manifest: dict[str, Any], audit: bytes, manifest_sha256: str) -> dict[str, Any]:
    lookup = source_pages.SourcePageLookup(manifest, root, replay=True)
    started = initialized = listed = False
    calls, errors = 0, 0
    seen = set()
    for line in audit.decode().splitlines():
        event = _parse(line)
        if event.get("packet_sha256") != manifest_sha256:
            raise ValueError("PDF tool audit is bound to a different manifest")
        kind = event.get("event")
        if kind == "source_lookup_started":
            if event.get("source_ids") != list(lookup.sources):
                raise ValueError("PDF tool startup changed its sources")
            started = True
        elif kind == "source_lookup_initialized":
            if not started or event.get("protocol_version") != source_lookup.PROTOCOL_VERSION:
                raise ValueError("PDF MCP initialization changed")
            initialized = True
        elif kind == "source_lookup_tools_listed":
            if not initialized or event.get("result_sha256") != _sha(_json({"tools": source_pages.tools()})):
                raise ValueError("PDF MCP tool catalog changed")
            listed = True
        elif kind == "source_lookup_call":
            if not listed:
                raise ValueError("PDF tool called before its catalog was delivered")
            try:
                if not isinstance(event.get("tool"), str):
                    raise ValueError("Tool name must be a string")
                data = lookup.call(event["tool"], event.get("arguments", {}))
                result = {"content": source_pages.content(data), "isError": False}
            except ValueError as exc:
                result = _error(str(exc))
                # Rendering failures are observed results, not deterministic
                # properties of a PDF. Preserve their exact delivered text;
                # replay must never launch a renderer to reproduce failure.
                if str(exc) == "Saved tool call has no rendered PDF page":
                    observed = event.get("error_result")
                    allowed = [_error("PDF page rendering requires pdftoppm"),
                               _error("PDF page rendering did not complete within policy")]
                    if len(seen) >= source_pages.MAX_PAGES:
                        allowed.append(_error("Distinct PDF page budget exhausted"))
                    if observed in allowed:
                        result = observed
            if (event.get("result_sha256") != _sha(_json(result)) or event.get("is_error") != result["isError"]
                    or (result["isError"] and event.get("error_result") != result)
                    or (not result["isError"] and "error_result" in event)):
                raise ValueError("Delivered PDF tool result differs from saved source/image replay")
            calls += 1
            errors += int(result["isError"])
            if not result["isError"] and event["tool"] == "read_source_pdf_page":
                seen.add(lookup.page_directory(data["source_id"], data["page"]).name)
        else:
            raise ValueError("Unknown PDF tool audit event")
    actual = {p.name for p in (root / "pages").iterdir()}
    if not (started and initialized and listed) or actual != seen or len(actual) > source_pages.MAX_PAGES:
        raise ValueError("PDF page evidence is incomplete or contains unrequested pages")
    return {"server_initialized": True, "tool_catalog_delivered": True, "tool_calls": calls,
            "tool_errors": errors, "rendered_pages": sorted(seen), "delivered_result_hashes_replayed": True}


@dataclass
class PreparedSourcePages:
    artifacts: Path
    profile: PrivateToolProfile
    binding: dict[str, Any]
    modules: dict[str, Path]

    def finish(self) -> dict[str, Any]:
        root = self.artifacts / DIRECTORY
        for name, path in self.modules.items():
            if _sha(_regular(path, source_pages.MAX_AUDIT_BYTES)) != self.binding["module_sha256"][name]:
                raise ValueError("Installed PDF tool module changed during the call")
        self.binding["private_profile_sha256"] = self.profile.verify()
        manifest_bytes = _regular(root / "manifest.json", 1024 * 1024)
        if _sha(manifest_bytes) != self.binding["manifest_sha256"]:
            raise ValueError("PDF manifest changed during the call")
        audit = _regular(root / "calls.jsonl", source_pages.MAX_AUDIT_BYTES)
        result = {**self.binding, **_events(root, _parse(manifest_bytes.decode()), audit, _sha(manifest_bytes)),
                  "audit_sha256": _sha(audit)}
        body = (_json(result) + "\n").encode()
        _private(root / "binding.json", body)
        receipt = {"policy": source_pages.POLICY, "binding_path": DIRECTORY + "/binding.json", "binding_sha256": _sha(body)}
        verify_source_pages(self.artifacts, receipt, expected_packet_sha256=self.binding["source_packet_sha256"],
                            expected_pdf_hashes=self.binding["pdf_sha256"])
        return receipt


def prepare_source_pages(packet: dict[str, Any], pdf_files: PDFs, *, artifacts: Path,
                         profile: PrivateToolProfile) -> PreparedSourcePages:
    from pypdf import PdfReader

    source_lookup.SourceLookup(packet)
    if not isinstance(pdf_files, Mapping) or not pdf_files or len(pdf_files) > 128:
        raise ValueError("PDF tools require a bounded nonempty PDF mapping")
    rows = {row["id"]: row for row in packet["sources"]}
    root = artifacts / DIRECTORY
    root.mkdir(mode=0o700)
    (root / "pages").mkdir(mode=0o700)
    manifest = {"schema_version": 1, "policy": source_pages.POLICY,
                "source_packet_sha256": _sha(_json(packet)), "sources": []}
    total = 0
    for source_id, specification in pdf_files.items():
        if source_id not in rows or not isinstance(specification, tuple) or len(specification) != 2:
            raise ValueError("PDF files must name existing frozen source IDs")
        path, expected = specification
        body = _regular(Path(path), source_pages.MAX_PDF_BYTES)
        total += len(body)
        if total > source_pages.MAX_TOTAL_PDF_BYTES or _sha(body) != expected:
            raise ValueError("PDF bytes/hash or total size differs from the frozen catalog")
        try:
            pages = len(PdfReader(io.BytesIO(body)).pages)
        except Exception as exc:
            raise ValueError("Frozen PDF pages could not be inspected") from exc
        filename = source_pages.pdf_name(source_id)
        _private(root / filename, body)
        manifest["sources"].append({"id": source_id, "title": rows[source_id].get("title"), "file": filename,
                                    "pdf_sha256": expected, "bytes": len(body), "pages": pages})
    source_pages.SourcePageLookup(manifest, root)
    raw = (_json(manifest) + "\n").encode()
    _private(root / "manifest.json", raw)
    installed = _modules()
    digests = {}
    for name, path in installed.items():
        snapshot = _regular(path, source_pages.MAX_AUDIT_BYTES)
        _private(root / name, snapshot)
        digests[name] = _sha(snapshot)
    names = [tool["name"] for tool in source_pages.tools()]
    profile.add_server("agsci_pages", {
        "command": sys.executable,
        "args": [str(installed["server.py"]), "--manifest", str(root / "manifest.json"),
                 "--manifest-sha256", _sha(raw), "--audit-log", str(root / "calls.jsonl")],
        "cwd": str(installed["server.py"].parent), "startup_timeout_sec": 20, "tool_timeout_sec": 30,
        "enabled_tools": names, "default_tools_approval_mode": "prompt",
        "tools": {name: {"approval_mode": "approve"} for name in names},
    })
    binding = {"schema_version": 1, "policy": source_pages.POLICY,
               "source_packet_sha256": manifest["source_packet_sha256"], "manifest_sha256": _sha(raw),
               "pdf_sha256": {row["id"]: row["pdf_sha256"] for row in manifest["sources"]},
               "module_sha256": digests, "profile_source_sha256": profile.source_sha256,
               "credential_files_exported": False, "permission_mode": "read-only",
               "semantic_fidelity_verified": False}
    return PreparedSourcePages(artifacts, profile, binding, installed)


def verify_source_pages(artifacts: Path, receipt: dict[str, Any], *, expected_packet_sha256: str,
                        expected_pdf_hashes: Mapping[str, str]) -> dict[str, Any]:
    if (not isinstance(receipt, dict) or receipt.get("policy") != source_pages.POLICY
            or receipt.get("binding_path") != DIRECTORY + "/binding.json"):
        raise ValueError("Invalid PDF page tool receipt")
    root = artifacts / DIRECTORY
    raw = _regular(root / "binding.json", source_pages.MAX_AUDIT_BYTES)
    if _sha(raw) != receipt.get("binding_sha256"):
        raise ValueError("PDF tool binding changed")
    binding = _parse(raw.decode())
    if (binding.get("schema_version") != 1 or binding.get("policy") != source_pages.POLICY
            or binding.get("source_packet_sha256") != expected_packet_sha256
            or binding.get("pdf_sha256") != dict(expected_pdf_hashes)
            or binding.get("credential_files_exported") is not False
            or binding.get("permission_mode") != "read-only"
            or binding.get("semantic_fidelity_verified") is not False):
        raise ValueError("PDF tool binding differs from the frozen evidence")
    if set(binding.get("module_sha256", {})) != set(_modules()):
        raise ValueError("PDF tool module list changed")
    for name, digest in binding["module_sha256"].items():
        if _sha(_regular(root / name, source_pages.MAX_AUDIT_BYTES)) != digest:
            raise ValueError("PDF tool module snapshot changed")
    raw = _regular(root / "manifest.json", 1024 * 1024)
    if _sha(raw) != binding.get("manifest_sha256"):
        raise ValueError("PDF tool manifest changed")
    manifest = _parse(raw.decode())
    if (manifest.get("source_packet_sha256") != expected_packet_sha256
            or {row["id"]: row["pdf_sha256"] for row in manifest["sources"]} != dict(expected_pdf_hashes)):
        raise ValueError("PDF tool manifest differs from frozen source hashes")
    audit = _regular(root / "calls.jsonl", source_pages.MAX_AUDIT_BYTES)
    if _sha(audit) != binding.get("audit_sha256"):
        raise ValueError("PDF tool audit changed")
    observed = _events(root, manifest, audit, binding["manifest_sha256"])
    if any(binding.get(k) != v for k, v in observed.items()):
        raise ValueError("PDF tool evidence does not replay")
    return binding
