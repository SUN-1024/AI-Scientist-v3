"""A claim-first research agenda. Completeness is not a scientific novelty score."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

CLAIM_TYPES = {"method", "empirical_finding", "theory", "benchmark", "system"}
INTENTS = {"novel_contribution", "replication", "negative_result", "exploratory"}
EVIDENCE_ASPECTS = {"problem_setting", "method", "evaluation", "contribution_boundary"}
SEED_AGENDA_PATH = Path("research_seed_agenda.json")


def load_seed_agenda(task_dir: Path) -> dict[str, Any] | None:
    """Read a previous proposal as an attributed input, never as a final agenda."""
    path = task_dir / SEED_AGENDA_PATH
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid research_seed_agenda.json artifact")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return normalize_seed_agenda(raw)


def normalize_seed_agenda(raw: object) -> dict[str, Any]:
    if (not isinstance(raw, dict) or raw.get("schema_version") != 1
            or set(raw) - {"schema_version", "agenda", "derived_from", "provenance"}
            or not isinstance(raw.get("derived_from"), dict) or not raw["derived_from"]):
        raise ValueError("Seed agenda needs schema_version=1, agenda, and nonempty derived_from provenance")
    if len(json.dumps(raw, ensure_ascii=False, allow_nan=False)) > 500_000:
        raise ValueError("Seed agenda exceeds 500,000 characters")
    result = copy.deepcopy(raw)
    result["agenda"] = normalize_agenda(raw.get("agenda"))
    return result


def _optional_methodology_fields(result: dict[str, Any]) -> None:
    """Validate declarations without creating missing fields or asserting facts."""
    # Keep legacy alternatives/evidence_plan values intact for receipt replay.
    # Their completeness is assessed downstream, not upgraded into schema errors.
    if "study_design" in result:
        design = result["study_design"]
        if not isinstance(design, dict):
            raise ValueError("agenda.study_design must be an object")
        for field in ("observation_unit", "split_unit", "intervention", "fixed_factors", "matched_budget",
                      "outcomes", "falsifier", "minimal_test", "stop_or_pivot"):
            if field not in design:
                continue
            value = design[field]
            if isinstance(value, str):
                continue
            if field in ("fixed_factors", "outcomes") and isinstance(value, list) and all(
                isinstance(item, str) for item in value
            ):
                continue
            raise ValueError(f"study_design.{field} must be text" +
                             (" or a list of text" if field in ("fixed_factors", "outcomes") else ""))
    if "medical_context" in result:
        medical = result["medical_context"]
        if not isinstance(medical, dict):
            raise ValueError("agenda.medical_context must be an object")
        for field in ("data_use_basis", "subject_isolation", "bias_and_population"):
            if field not in medical:
                continue
            fact = medical[field]
            if isinstance(fact, str):
                continue
            if not isinstance(fact, dict) or fact.get("status") not in ("known", "unknown", "not_applicable"):
                raise ValueError(f"medical_context.{field} needs text or a fact with a valid status")
            if "value" in fact and not isinstance(fact["value"], str):
                raise ValueError(f"medical_context.{field}.value must be text")
            if "evidence" in fact and (not isinstance(fact["evidence"], list) or not all(
                isinstance(item, str) for item in fact["evidence"]
            )):
                raise ValueError(f"medical_context.{field}.evidence must be a list of source references")


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
    _optional_methodology_fields(result)
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
