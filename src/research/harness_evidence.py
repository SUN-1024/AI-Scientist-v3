"""Bind a selected-method review to actual code and host-scored observations."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from src.final_evaluation.runner import resolve_frozen_incumbent
from src.investigation.adaptive_runtime import validated_development_outcome
from src.investigation.native_harness import source_fingerprint
from .harness_state import read_json, sha256

MAX_CANDIDATE_PACKET_BYTES = 2 * 1024 * 1024


def build_candidate_packet(workspace: Path) -> dict[str, Any]:
    investigation = workspace / "run/investigation"
    task = workspace / "run/task"
    program, _ = resolve_frozen_incumbent(investigation)
    fingerprints = source_fingerprint(program)
    manifest = read_json(program / "evaluation_manifest.json")
    attempt_id, receipt_idea_id = manifest.get("attempt_id"), manifest.get("idea_id")
    if not isinstance(attempt_id, str) or not attempt_id or not isinstance(receipt_idea_id, str) or not receipt_idea_id:
        raise ValueError("Selected host evaluation receipt requires non-empty attempt and idea identifiers")
    observed = validated_development_outcome(
        program, task_dir=task, attempt_id=attempt_id, idea_id=receipt_idea_id,
        native=manifest.get("execution_boundary") == "host",
    )
    if observed is None:
        raise ValueError("Selected-method novelty review requires a current host receipt bound to code, evaluator, and metrics")
    sources = []
    for name, digest in sorted(fingerprints.items()):
        path = program / name
        if path.stat().st_size > MAX_CANDIDATE_PACKET_BYTES:
            raise ValueError("Selected candidate source exceeds the review packet bound")
        sources.append({"path": name, "sha256": digest, "text": path.read_text(encoding="utf-8")})
    best = read_json(investigation / "best_program_manifest.json")
    idea_id = observed.get("idea_id")
    if not isinstance(idea_id, str) or not idea_id:
        raise ValueError("Selected host evaluation receipt has no idea identifier")
    if best.get("source_attempt_id") and best["source_attempt_id"] != observed.get("attempt_id"):
        raise ValueError("Selected best program and host evaluation receipt identify different attempts")
    basis = None
    # Prefer the selected attempt's ancestors. Only then inspect a matching
    # source-grounded idea bundle; generation scratch copies are not evidence.
    ancestors = [parent / "idea_bundle.json" for parent in program.parents if parent.is_relative_to(investigation)]
    alternatives = sorted(investigation.rglob("idea_bundle.json"))
    for path in dict.fromkeys([*ancestors, *alternatives]):
        if not path.is_file() or "_idea_generation" in path.parts:
            continue
        bundle = read_json(path)
        if bundle.get("idea_id") == idea_id and isinstance(bundle.get("research_basis"), dict):
            basis = bundle["research_basis"]
            break
    if basis is None:
        raise ValueError("Selected candidate has no source-grounded research basis")
    protocol = {}
    for name in ("task.md", "task_contract.json", "native_evaluation_inputs.json", "study_plan.json", "evaluate.py"):
        path = task / name
        if path.is_file():
            if path.stat().st_size > MAX_CANDIDATE_PACKET_BYTES:
                raise ValueError("Selected protocol source exceeds the review packet bound")
            protocol[name] = {"sha256": sha256(path), "text": path.read_text(encoding="utf-8")}
    packet = {
        "schema_version": 1,
        "selection": {"idea_id": idea_id, "source_attempt_id": observed.get("attempt_id"), "program_relative_path": str(program.relative_to(workspace))},
        "candidate_source_sha256": hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_file_sha256": fingerprints, "source_files": sources,
        "research_basis": basis,
        "evaluation": {"source_receipt_validated": True, "receipt_sha256": sha256(program / "evaluation_manifest.json"),
                            "receipt": {key: manifest.get(key) for key in (
                                "producer", "execution_boundary", "attempt_id", "idea_id", "program_id", "started_at", "finished_at",
                                "exit_code", "returncode", "source_sha256", "evaluator_sha256", "evaluation_inputs_sha256",
                                "adaptive_source_sha256", "adaptive_evaluator_sha256", "adaptive_result_sha256")},
                            "metrics": read_json(program / "metrics.json"), "correct": read_json(program / "correct.json"),
                            "metrics_sha256": sha256(program / "metrics.json"), "correct_sha256": sha256(program / "correct.json")},
        "protocol": protocol,
        "claim_code_evidence_map": {"status": "not_independently_verified",
                                    "instruction": "Compare the proposed mechanism with the actual source files and measured outcomes. Identify absent or untested mechanisms."},
        "limits": "These are development observations. They do not establish independent final confirmation or causal attribution.",
    }
    for filename in ("evaluation_evidence.json", "claim_code_evidence_map.json"):
        path = program / filename
        if path.is_file():
            packet[filename.removesuffix(".json")] = {"sha256": sha256(path), "payload": read_json(path),
                                                     "status": "supporting_observations_not_independent_proof"}
    if len(json.dumps(packet, ensure_ascii=False).encode()) > MAX_CANDIDATE_PACKET_BYTES:
        raise ValueError("Selected-method review packet exceeds 2 MiB; use a bounded study artifact")
    return packet
