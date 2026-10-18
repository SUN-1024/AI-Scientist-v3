"""A claim-first research agenda. Completeness is not a scientific novelty score."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

CLAIM_TYPES = {"method", "empirical_finding", "theory", "benchmark", "system"}
INTENTS = {"novel_contribution", "replication", "negative_result", "exploratory"}
EVIDENCE_ASPECTS = {"problem_setting", "method", "evaluation", "contribution_boundary"}


def normalize_agenda(raw: object) -> dict[str, Any]:
    """Reject malformed data, while retaining incomplete agendas for actionable review."""
    if not isinstance(raw, dict) or raw.get("schema_version", 1) != 1:
        raise ValueError("Research agenda must be a schema_version=1 object")
    if len(json.dumps(raw, ensure_ascii=False)) > 200_000:
        raise ValueError("Research agenda exceeds 200,000 characters")
    result = copy.deepcopy(raw)
    result["schema_version"] = 1
    result.setdefault("intent", "exploratory")
    result.setdefault("claim_type", "empirical_finding")
    if result["intent"] not in INTENTS or result["claim_type"] not in CLAIM_TYPES:
        raise ValueError("Unknown research intent or claim_type")
    for field in ("research_question", "proposed_contribution", "why_meaningful", "scope"):
        result.setdefault(field, "")
        if not isinstance(result[field], str):
            raise ValueError(f"agenda.{field} must be text")
    for field in ("closest_work", "alternatives", "source_selection"):
        result.setdefault(field, [])
        if not isinstance(result[field], list) or not all(isinstance(x, dict) for x in result[field]):
            raise ValueError(f"agenda.{field} must be a list of objects")
    for row in result["closest_work"]:
        for field in ("source_id", "problem_setting", "core_method", "contribution_boundary",
                      "overlap", "difference", "why_meaningful"):
            row.setdefault(field, "")
            if not isinstance(row[field], str):
                raise ValueError(f"closest_work.{field} must be text")
        row.setdefault("evidence", [])
        if not isinstance(row["evidence"], list) or not all(isinstance(x, dict) for x in row["evidence"]):
            raise ValueError("closest_work.evidence must contain passage citation objects")
    for field in ("evidence_plan", "source_scan"):
        result.setdefault(field, {})
        if not isinstance(result[field], dict):
            raise ValueError(f"agenda.{field} must be an object")
    return result


def load_agenda(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / "research_agenda.json"
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid research_agenda.json artifact")
    return normalize_agenda(json.loads(path.read_text(encoding="utf-8")))


def agenda_from_basis(basis: dict[str, Any]) -> dict[str, Any]:
    """Migrate an old idea without inventing missing evidence or upgrading its status."""
    rows = [{"source_id": row.get("source_id", ""), "overlap": row.get("overlap", ""),
             "difference": row.get("difference", ""), "evidence": [],
             "legacy_evidence_ids": row.get("evidence_ids", [])}
            for row in basis.get("closest_work", []) if isinstance(row, dict)]
    return normalize_agenda({
        "schema_version": 1,
        "intent": "replication" if basis.get("status") == "replication" else "novel_contribution",
        "proposed_contribution": basis.get("contribution", ""), "closest_work": rows,
        "alternatives": [{"explanation": basis.get("alternative_explanation", ""),
                          "discriminating_experiment": {"intervention": basis.get("discriminating_test", "")}}],
        "legacy_basis": copy.deepcopy(basis),
    })
