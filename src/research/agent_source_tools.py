"""Per-call Codex source tools with replayable, credential-free provenance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

from .agent_tool_profile import (
    PrivateToolProfile, _private, _regular, _toml_value as _toml_value, prepare_tool_profile,
)
from . import source_lookup
from .source_lookup import SourceLookup, _json, _parse, _read_packet, _sha

POLICY = "frozen_source_lookup_v1"
DIRECTORY = "source_lookup"


def _events(body: bytes, lookup: SourceLookup, packet_sha256: str) -> dict[str, Any]:
    initialized = False
    listed = False
    started = False
    calls = []
    for line in body.decode("utf-8").splitlines():
        event = _parse(line)
        if event.get("packet_sha256") != packet_sha256:
            raise ValueError("Source-tool audit is bound to a different packet")
        name = event.get("event")
        if name == "source_lookup_started":
            if event.get("source_ids") != list(lookup.sources):
                raise ValueError("Source-tool startup omits or changes frozen sources")
            started = True
        elif name == "source_lookup_initialized":
            if not started or event.get("protocol_version") != source_lookup.PROTOCOL_VERSION:
                raise ValueError("Source-tool protocol initialization changed")
            initialized = True
        elif name == "source_lookup_tools_listed":
            expected = {"tools": source_lookup.tools()}
            if not initialized or event.get("result_sha256") != _sha(_json(expected)):
                raise ValueError("Source-tool catalog does not replay")
            listed = True
        elif name == "source_lookup_call":
            if not initialized or not listed:
                raise ValueError("Source tool was called before its catalog was delivered")
            try:
                if not isinstance(event.get("tool"), str):
                    raise ValueError("Tool name must be a string")
                data = lookup.call(event.get("tool"), event.get("arguments", {}))
                result = {"content": [{"type": "text", "text": _json(data)}], "isError": False}
            except ValueError as exc:
                result = {"content": [{"type": "text", "text": str(exc)}], "isError": True}
            if event.get("result_sha256") != _sha(_json(result)) or event.get("is_error") != result["isError"]:
                raise ValueError("Source-tool result differs from frozen source replay")
            calls.append({"tool": event["tool"], "is_error": event["is_error"]})
        else:
            raise ValueError("Unknown source-tool audit event")
    if not (started and initialized and listed):
        raise ValueError("Source tools were not initialized and delivered to the native client")
    return {"server_initialized": True, "tool_catalog_delivered": True,
            "tool_calls": len(calls), "tool_errors": sum(c["is_error"] for c in calls),
            "tools_called": sorted({c["tool"] for c in calls}), "result_hashes_replayed": True}


@dataclass
class PreparedSourceTools:
    artifacts: Path
    environment: dict[str, str]
    binding: dict[str, Any]
    installed_module: Path
    profile: PrivateToolProfile

    def finish(self) -> dict[str, Any]:
        if _sha(_regular(self.installed_module, source_lookup.MAX_PACKET_BYTES)) != self.binding["server_sha256"]:
            raise ValueError("Source-tool executable changed during the call")
        self.binding["private_profile_sha256"] = self.profile.verify()
        root = self.artifacts / DIRECTORY
        packet_bytes = _read_packet(root / "source-packet.json", self.binding["packet_sha256"])
        snapshot = _regular(root / "server.py", source_lookup.MAX_PACKET_BYTES)
        if _sha(snapshot) != self.binding["server_sha256"]:
            raise ValueError("Source-tool executable snapshot changed")
        audit = _regular(root / "calls.jsonl", source_lookup.MAX_PACKET_BYTES)
        observed = _events(audit, SourceLookup(_parse(packet_bytes.decode())), self.binding["packet_sha256"])
        result = {**self.binding, **observed, "audit_sha256": _sha(audit)}
        body = (_json(result) + "\n").encode()
        _private(root / "binding.json", body)
        return {"policy": POLICY, "binding_path": DIRECTORY + "/binding.json",
                "binding_sha256": _sha(body)}


def prepare_source_tools(packet: dict[str, Any], *, artifacts: Path, temporary: Path,
                         environment: Mapping[str, str], agent: str, model: str | None,
                         profile: str | None,
                         tool_profile: PrivateToolProfile | None = None) -> PreparedSourceTools:
    """Stage one selected profile, its auth, and its catalog into a fresh home."""
    selected = tool_profile or prepare_tool_profile(
        temporary=temporary, environment=environment, agent=agent, model=model, profile=profile)
    SourceLookup(packet)
    packet_bytes = (_json(packet) + "\n").encode()
    if len(packet_bytes) > source_lookup.MAX_PACKET_BYTES:
        raise ValueError("Source-tool packet exceeds its size bound")
    root = artifacts / DIRECTORY
    root.mkdir()
    installed = Path(source_lookup.__file__).resolve()
    server_bytes = _regular(installed, source_lookup.MAX_PACKET_BYTES)
    _private(root / "server.py", server_bytes)
    _private(root / "source-packet.json", packet_bytes)
    selected.add_server("agsci_sources", {
        "command": sys.executable,
        "args": [str(installed), "--packet", str(root / "source-packet.json"),
                 "--packet-sha256", _sha(packet_bytes),
                 "--audit-log", str(root / "calls.jsonl")],
        "cwd": str(installed.parent), "startup_timeout_sec": 20, "tool_timeout_sec": 30,
    })
    binding = {
        "schema_version": 1, "policy": POLICY,
        "packet_sha256": _sha(packet_bytes), "packet_content_sha256": _sha(_json(packet)),
        "server_sha256": _sha(server_bytes), "profile_source_sha256": selected.source_sha256,
        "private_profile_sha256": selected.verify(),
        "source_ids": list(SourceLookup(packet).sources),
        "offset_units": "Python Unicode code points",
        "credential_files_exported": False, "permission_mode": "read-only",
    }
    return PreparedSourceTools(
        artifacts, selected.environment, binding, installed, selected,
    )


def verify_source_tools(artifacts: Path, receipt: dict[str, Any], *,
                        expected_packet: dict[str, Any]) -> dict[str, Any]:
    """Replay the immutable tool evidence without a model or credential files."""
    if (not isinstance(receipt, dict) or receipt.get("policy") != POLICY
            or receipt.get("binding_path") != DIRECTORY + "/binding.json"):
        raise ValueError("Invalid source-tool receipt")
    root = artifacts / DIRECTORY
    binding_bytes = _read_packet(root / "binding.json", receipt.get("binding_sha256", ""))
    binding = _parse(binding_bytes.decode())
    if binding.get("policy") != POLICY or binding.get("schema_version") != 1:
        raise ValueError("Unsupported source-tool binding")
    packet_bytes = _read_packet(root / "source-packet.json", binding.get("packet_sha256", ""))
    packet = _parse(packet_bytes.decode())
    if _json(packet) != _json(expected_packet) or binding.get("packet_content_sha256") != _sha(_json(packet)):
        raise ValueError("Source tools do not contain the exact prompt's complete source packet")
    if _sha(_regular(root / "server.py", source_lookup.MAX_PACKET_BYTES)) != binding.get("server_sha256"):
        raise ValueError("Source-tool executable snapshot changed")
    audit = _regular(root / "calls.jsonl", source_lookup.MAX_PACKET_BYTES)
    if _sha(audit) != binding.get("audit_sha256"):
        raise ValueError("Source-tool audit changed")
    observed = _events(audit, SourceLookup(packet), binding["packet_sha256"])
    if (binding.get("source_ids") != list(SourceLookup(packet).sources)
            or binding.get("credential_files_exported") is not False
            or binding.get("permission_mode") != "read-only"
            or any(binding.get(k) != v for k, v in observed.items())):
        raise ValueError("Source-tool evidence differs from its binding")
    return binding
