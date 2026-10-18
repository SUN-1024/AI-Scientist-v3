"""Classify coding transport failures before host evaluation, on a private channel."""
from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from src.task_drafting.agent_failure import AgentFailureKind, classify_agent_failure_log

AGENT_FAILURE_FD_ENV = "AGENTIC_CODING_FAILURE_FD"
_MAX_RECEIPT_BYTES = 4096


def has_evaluation_failure(experiment: Path) -> bool:
    """A retained host failure requires evaluation recovery, never another coder run."""
    return any(
        path.name != "program_0" and (path / "evaluation_failure.json").exists()
        for path in experiment.glob("program_*")
    )


def has_evaluation_artifacts(experiment: Path) -> bool:
    """Any candidate evaluation evidence disables transport retry, including failures."""
    if (experiment / "rubric_status.json").exists():
        return True
    for path in experiment.glob("program_*"):
        if path.name == "program_0":
            continue
        if any((path / name).exists() for name in ("metrics.json", "correct.json", "rubric_status.json", "evaluation.json", "evaluation_receipt.json", "evaluation_manifest.json", "evaluation_failure.json", "runs")):
            return True
    # Agent-managed evaluation may have failed before producing final metrics.
    return any(experiment.glob(".dev_eval_program_*"))


def report_agent_failure(descriptor: int, attempt_id: str, log: Path, experiment: Path) -> None:
    if has_evaluation_artifacts(experiment):
        return
    try:
        kind = classify_agent_failure_log(log)
    except ValueError:
        return
    if kind is not AgentFailureKind.TRANSIENT_PROVIDER:
        return
    raw = json.dumps({"attempt_id": attempt_id, "kind": kind.value}).encode()
    if len(raw) > _MAX_RECEIPT_BYTES or not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise ValueError("invalid private coding failure channel")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.write(descriptor, raw)


def read_agent_failure(descriptor: int, attempt_id: str) -> str | None:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("invalid private coding failure receipt")
    raw = os.pread(descriptor, _MAX_RECEIPT_BYTES + 1, 0)
    if not raw:
        return None
    expected = {"attempt_id": attempt_id, "kind": AgentFailureKind.TRANSIENT_PROVIDER.value}
    if json.loads(raw) != expected:
        raise ValueError("invalid private coding failure receipt")
    return AgentFailureKind.TRANSIENT_PROVIDER.value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    if args.fd < 3:
        parser.error("requires a private descriptor")
    report_agent_failure(args.fd, args.attempt_id, args.log, args.experiment)


if __name__ == "__main__":
    main()
