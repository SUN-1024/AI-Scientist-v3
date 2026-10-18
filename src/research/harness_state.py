"""Host-side lifecycle receipts for the reusable research harness.

A phase event is an execution observation. Scientific qualification is assessed
separately from evaluator receipts and manuscript audits.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.orchestrator.state import write_private_text
from src.pipeline.config import PIPELINE_PHASES

IGNORED = {".venv", ".git", "__pycache__", ".pytest_cache", "node_modules"}
SUFFIXES = {".json", ".jsonl", ".csv", ".py", ".toml", ".tex", ".bib", ".pdf", ".yaml"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"Missing or unsafe harness artifact: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular evidence file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_records(workspace: Path, phase: str) -> dict[str, str]:
    roots = {
        "task_drafting": ("run/task",), "investigation": ("run/investigation",),
        "ablations": ("run/ablations",), "final_evaluation": ("run/final_evaluation",),
        "posthoc": ("run/posthoc",), "paper": ("run/paper",),
        "review": ("run/paper",), "revision": ("run/paper", "run/revision"),
    }.get(phase, ())
    records = {}
    for name in roots:
        root = workspace / name
        if root.is_symlink():
            raise ValueError(f"Stage root must not be a symlink: {root}")
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(workspace)
            if any(part in IGNORED for part in relative.parts) or path.suffix not in SUFFIXES:
                continue
            if phase == "paper" and (path.name == "latest_manifest.json" or path.name.startswith("ai_review")):
                # These aliases belong to the later review/revision stage. Failed
                # review may update its status but must not unseal manuscript bytes.
                continue
            if phase == "review" and not (path.name == "paper.pdf" or (path.name.startswith("ai_review") and path.suffix == ".json")):
                continue
            if phase == "task_drafting" and (path.name.startswith(".") or ("program_0" in path.parts and path.name in {
                "metrics.json", "correct.json", "evaluation_manifest.json", "evaluation_evidence.json", "rubric_status.json"})):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(workspace.resolve()):
                raise ValueError(f"Stage evidence contains unsafe path: {relative}")
            if path.is_file():
                records[relative.as_posix()] = sha256(path)
            if len(records) > 20000:
                raise ValueError("Stage evidence exceeds 20000 auditable files")
    return records


def _append_only_memory_matches(path: Path, saved_digest: str) -> bool:
    """Later phases append workflow memory; its saved prefix stays immutable."""
    if path.stat().st_size > 32 * 1024 * 1024:
        return False
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        return False
    digest = hashlib.sha256()
    matched = digest.hexdigest() == saved_digest
    for line in raw.splitlines(keepends=True):
        digest.update(line)
        if digest.hexdigest() == saved_digest:
            matched = True
        if line.strip() and not isinstance(json.loads(line.decode("utf-8")), dict):
            return False
    return matched


def verify_stage_records(workspace: Path, stage: dict[str, Any]) -> bool:
    records = stage.get("artifacts", {})
    if stage.get("status") != "completed" or not isinstance(records, dict) or not records:
        return False
    try:
        for name, digest in records.items():
            path = workspace / name
            if Path(name).is_absolute() or ".." in Path(name).parts or not path.resolve().is_relative_to(workspace.resolve()):
                return False
            if sha256(path) != digest:
                if name != "run/investigation/memory.jsonl" or not _append_only_memory_matches(path, digest):
                    return False
    except (OSError, ValueError):
        return False
    return True


class HarnessState:
    def __init__(self, workspace: Path, *, phases: list[str], quality_target: str,
                 input_sha256: str | None, prior: dict[str, Any] | None = None):
        self.workspace = workspace
        self.payload = prior or {
            "schema_version": 1, "created_at": now(), "quality_target": quality_target,
            "input_task_sha256": input_sha256, "requested_phases": phases,
            "stages": {phase: {"status": "pending"} for phase in phases}, "events": [],
            "human_approval": {"status": "not_requested", "submitted": False},
        }
        if prior and (prior.get("input_task_sha256") != input_sha256 or prior.get("quality_target") != quality_target):
            raise ValueError("Harness resume does not match frozen inputs or quality target")
        self.payload["status"] = "running"

    def save(self) -> None:
        self.payload["updated_at"] = now()
        write_private_text(self.workspace / "harness_state.json", json.dumps(self.payload, ensure_ascii=False, indent=2) + "\n")

    def observe(self, event: str, **fields: Any) -> None:
        if event not in {"phase_start", "phase_end", "pipeline_end"}:
            return
        entry = {"event": event, "time": now(), **fields}
        self.payload["events"].append(json.loads(json.dumps(entry, default=str)))
        phase = fields.get("phase_name")
        if phase in PIPELINE_PHASES:
            stage = self.payload["stages"].setdefault(phase, {})
            stage.update({"status": "running" if event == "phase_start" else fields.get("status", "unknown"),
                          "updated_at": now()})
            for key in ("return_code", "reason", "outcome"):
                if key in fields:
                    stage[key] = fields[key]
            if event == "phase_end" and stage["status"] == "completed":
                stage["artifacts"] = stage_records(self.workspace, phase)
                # Review/revision legitimately update the published paper manifest.
                if phase == "revision":
                    for previous in ("paper", "review"):
                        earlier = self.payload["stages"].get(previous, {})
                        if earlier.get("status") == "completed":
                            earlier["artifacts"] = stage_records(self.workspace, previous)
        self.save()

    def finish(self, exit_code: int, *, error: str | None = None) -> None:
        self.payload.update(status="pipeline_returned" if exit_code == 0 else "interrupted" if exit_code == 130 else "failed",
                            pipeline_exit_code=exit_code)
        if error:
            self.payload["error"] = error
        elif exit_code == 0:
            self.payload.pop("error", None)
        for stage in self.payload["stages"].values():
            if stage.get("status") == "running":
                stage.update(status="interrupted" if exit_code == 130 else "failed", reason=error or "pipeline stopped")
        self.save()
