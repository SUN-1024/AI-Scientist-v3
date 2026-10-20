"""Independent multi-reviewer critique of a manuscript draft.

The audit note needed four separate review rounds, each performed by hand. This module
makes that loop reproducible: give it a LaTeX or PDF source, it asks N independent model
contexts to apply the same rubric, validates each response against a fixed schema, and
writes per-reviewer receipts plus one consolidated report.

It never edits the manuscript and never claims human review; the report records the
independence limit explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from src.review_backend import run_fresh_agent_json, strict_json_object

SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 2 * 1024 * 1024
VERDICTS = {"accept_as_note", "revise", "reject"}
REQUIRED_KEYS = ("verdict", "blocking", "non_blocking", "prior_art_gaps",
                 "venue_fit", "required_extra_evidence", "verdict_reasoning")

RUBRIC = """You are an independent reviewer for a short research note. Review the manuscript below.

Context you must take as given:
- the manuscript is a finite, assumption-and-implementation audit, not a method paper;
- every numerical claim was reproduced by bounded container runs;
- the authors do not claim any published theorem is false, nor any training failure.

Judge only:
1. Does any sentence claim more than its stated evidence supports?
2. Is the prior-art boundary accurate and complete?
3. Are there internal inconsistencies (numbers, definitions, notation)?
4. What is the minimum required to make this defensible as a comment / replication / diagnostic note?
5. Is the title and abstract calibrated to the content?
6. Would a venue reviewer ask for evidence beyond the current set? Name it.

Return ONLY one JSON object, no prose and no code fence:
{"verdict": "accept_as_note" | "revise" | "reject",
 "blocking": [{"quote": "...", "problem": "...", "fix": "..."}],
 "non_blocking": [{"quote": "...", "problem": "...", "fix": "..."}],
 "prior_art_gaps": ["..."],
 "venue_fit": {"comment": "...", "replication": "...", "diagnostic_note": "..."},
 "required_extra_evidence": ["..."],
 "verdict_reasoning": "at most four sentences"}

MANUSCRIPT:
"""


def sha256(payload: bytes | str) -> str:
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(body).hexdigest()


def load_source(path: Path) -> str:
    """Read a .tex source directly or extract text from a PDF."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Manuscript does not exist: {path}")
    if path.suffix.lower() == ".tex":
        body = path.read_text(encoding="utf-8")
    elif path.suffix.lower() == ".pdf":
        result = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                                capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ValueError("pdftotext could not read the manuscript PDF")
        body = result.stdout
    elif path.suffix.lower() == ".md":
        body = path.read_text(encoding="utf-8")
    else:
        raise ValueError("Manuscript must be .tex, .pdf or .md")
    if len(body.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Manuscript exceeds the critique input bound")
    return body


def _validate_critique(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Critique is not a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError("Critique is missing keys: " + ", ".join(missing))
    if payload["verdict"] not in VERDICTS:
        raise ValueError("Critique verdict is not one of " + ", ".join(sorted(VERDICTS)))
    for key in ("blocking", "non_blocking", "prior_art_gaps", "required_extra_evidence"):
        if not isinstance(payload[key], list):
            raise ValueError(f"Critique {key} must be a list")
    for row in payload["blocking"] + payload["non_blocking"]:
        if not isinstance(row, dict) or not all(isinstance(row.get(field), str)
                                                for field in ("quote", "problem", "fix")):
            raise ValueError("Each critique item needs text quote, problem and fix")
    if not isinstance(payload["venue_fit"], dict):
        raise ValueError("Critique venue_fit must be an object")
    if not isinstance(payload["verdict_reasoning"], str):
        raise ValueError("Critique verdict_reasoning must be text")
    return payload


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def critique(manuscript: Path, output: Path, *, settings: Mapping[str, Any],
             reviewers: int = 2, rubric: str = RUBRIC) -> dict[str, Any]:
    """Run ``reviewers`` independent model critiques and write a consolidated report."""
    if type(reviewers) is not int or not 2 <= reviewers <= 4:
        raise ValueError("Use two to four independent review contexts")
    if settings.get("agent") != "codex" or not settings.get("model") or not settings.get("profile"):
        raise ValueError("Critique requires an explicit codex model and profile")
    source = load_source(manuscript)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("reviewer_*")):
        raise ValueError("Critique output already contains reviewer receipts")
    prompt = rubric + source
    if len(prompt.encode("utf-8")) > MAX_SOURCE_BYTES + 100_000:
        raise ValueError("Critique prompt exceeds its bound")
    records = []
    started = datetime.now(timezone.utc).isoformat()
    for index in range(1, reviewers + 1):
        directory = output / f"reviewer_{index}"
        work = directory / "call"
        directory.mkdir(parents=True)
        (directory / "source.txt").write_text(source, encoding="utf-8")
        payload, receipt = run_fresh_agent_json(
            prompt, work_dir=work, agent=settings["agent"], model=settings["model"],
            profile=settings["profile"], reasoning_effort=settings.get("reasoning_effort"),
            timeout_seconds=settings.get("timeout_seconds", 1800),
            environment=settings.get("environment"),
        )
        assessment = _validate_critique(strict_json_object(payload if isinstance(payload, str)
                                                          else json.dumps(payload)))
        _dump(directory / "assessment.json", assessment)
        record = {
            "directory": directory.name,
            "native_session_id": receipt.get("native_session_id"),
            "observed_model": receipt.get("observed_model"),
            "status": receipt.get("status"),
            "verdict": assessment["verdict"],
            "blocking": len(assessment["blocking"]),
            "non_blocking": len(assessment["non_blocking"]),
            "assessment_sha256": sha256((directory / "assessment.json").read_bytes()),
            "receipt_sha256": sha256((work / "receipt.json").read_bytes())
            if (work / "receipt.json").is_file() else None,
        }
        records.append(record)
    report = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "manuscript": str(manuscript.resolve()),
        "manuscript_sha256": sha256(source),
        "reviewers": records,
        "verdicts": [record["verdict"] for record in records],
        "consensus": ("accept_as_note" if records and all(record["verdict"] == "accept_as_note" for record in records)
                      else "reject" if records and all(record["verdict"] == "reject" for record in records)
                      else "revise"),
        "blocking_total": sum(record["blocking"] for record in records),
        "independence_limit": "separate stateless model calls, possibly the same model; not human peer review",
    }
    _dump(output / "critique_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path, help="Path to .tex, .pdf or .md")
    parser.add_argument("--output", type=Path, required=True, help="Directory for receipts and report")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--reviewers", type=int, default=2)
    args = parser.parse_args(argv)
    settings = {"agent": args.agent, "model": args.model, "profile": args.profile,
                "reasoning_effort": args.reasoning_effort, "timeout_seconds": args.timeout_seconds}
    report = critique(args.manuscript, args.output, settings=settings, reviewers=args.reviewers)
    print(json.dumps({"consensus": report["consensus"], "verdicts": report["verdicts"],
                      "blocking_total": report["blocking_total"],
                      "report": str(args.output / "critique_report.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
