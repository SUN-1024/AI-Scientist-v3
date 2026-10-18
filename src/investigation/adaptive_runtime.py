"""Host-only actions and receipt recovery for adaptive investigation."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .harness_incumbent import _finite_score, _program_source_records, _read_harness_json
from .models import RunAttemptResult
from .native_harness import _hash, validated_native_outcome
from .program_clone import clone_program_dir
from .search_tree import _fingerprint


def validated_development_outcome(program_dir: Path, *, task_dir: Path, attempt_id: str, idea_id: str, native: bool) -> dict[str, Any] | None:
    if native:
        outcome = validated_native_outcome(program_dir, task_dir / "evaluate.py")
        if outcome and outcome.get("attempt_id") == attempt_id and outcome.get("idea_id") == idea_id:
            return {**outcome, "source_hash": _fingerprint(program_dir)}
        return None
    manifest = _read_harness_json(program_dir / "evaluation_manifest.json")
    if not manifest or not (
        manifest.get("producer") == "sandbox_host_eval" and manifest.get("execution_boundary") == "sandbox"
        and manifest.get("attempt_id") == attempt_id and manifest.get("idea_id") == idea_id
        and manifest.get("program_id") == program_dir.name
        and manifest.get("exit_code") == 0 and manifest.get("returncode") == 0
    ):
        return None
    try:
        if manifest.get("adaptive_source_sha256") != _program_source_records(program_dir) or manifest.get("adaptive_evaluator_sha256") != _hash(task_dir / "evaluate.py"):
            return None
        if manifest.get("adaptive_result_sha256") != {name: _hash(program_dir / name) for name in ("metrics.json", "correct.json")}:
            return None
        metrics, correct = _read_harness_json(program_dir / "metrics.json"), _read_harness_json(program_dir / "correct.json")
        score = _finite_score(metrics.get("combined_score")) if metrics else None
        if score is None or not correct or type(correct.get("correct")) is not bool:
            return None
        return {"score": score, "correct": correct["correct"], "receipt_id": _hash(program_dir / "evaluation_manifest.json"),
                "program_dir": str(program_dir.resolve()), "source_hash": _fingerprint(program_dir),
                "attempt_id": attempt_id, "idea_id": idea_id}
    except (OSError, ValueError, TypeError):
        return None


def recovered_attempt(experiment_dir: Path, *, task_dir: Path, attempt_id: str, idea_id: str, native: bool) -> RunAttemptResult | None:
    outcome = validated_development_outcome(experiment_dir / "program_1", task_dir=task_dir, attempt_id=attempt_id, idea_id=idea_id, native=native)
    if outcome is None:
        return None
    log = experiment_dir / "wrapper_output.log"
    output = log.read_text() if log.is_file() and not log.is_symlink() else "Recovered completed host evaluation; no evaluator rerun."
    return RunAttemptResult(output=output, exit_code=0, experiment_dir=experiment_dir)


def run_frozen_replication(*, task_dir: Path, experiment_dir: Path, candidate_dir: Path,
                          source_hash: str, attempt_id: str, idea_id: str, rubrics_file: Path,
                          global_memory_dir: Path, timeout_seconds: int, containerize: bool,
                          container_image: str, gpu_evaluation: bool) -> RunAttemptResult:
    """Run exact inherited code through the existing native/sealed host evaluator.

    No model creates code for this action. The evaluator's fixed seed/data protocol
    is unchanged; independent confirmation remains an outer-pipeline responsibility.
    """
    from .harness_eval import SealedHarnessConfig, run_harness_evaluation
    if (task_dir / "task_contract.json").exists() and not containerize:
        raise ValueError("Contracted replication requires the existing sealed evaluation boundary")
    native = not containerize and not (task_dir / "task_contract.json").exists()
    recovered = recovered_attempt(experiment_dir, task_dir=task_dir, attempt_id=attempt_id, idea_id=idea_id, native=native)
    if recovered:
        return recovered
    if _fingerprint(candidate_dir) != source_hash:
        raise ValueError("Frozen replication source changed after selection")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate = experiment_dir / "program_1"
    if candidate.exists():
        if _fingerprint(candidate) != source_hash:
            raise ValueError("Interrupted replication candidate differs from frozen source")
    else:
        clone_program_dir(src_dir=candidate_dir, dst_dir=candidate)
    if _fingerprint(candidate) != source_hash:
        raise ValueError("Replication clone does not preserve candidate sources")
    shutil.copyfile(task_dir / "evaluate.py", experiment_dir / "evaluate.py")
    # The sealed evaluator receives the canonical task and its existing contract.
    environment = dict(os.environ, TASK_SOURCE_DIR=str(task_dir))
    sealed = SealedHarnessConfig(task_source_dir=task_dir, sandbox_image=container_image,
                                sandbox_gpus=os.environ.get("AGENT_CONTAINER_GPUS"),
                                environment=environment, gpu_evaluation=gpu_evaluation) if containerize else None
    task_python = task_dir / ".venv" / "bin" / "python"
    if containerize:
        if task_python.is_symlink() or not task_python.is_file():
            raise ValueError("Sealed replication requires the canonical managed task Python")
        python_bin = str(task_python)
    else:
        python_bin = str(task_python) if task_python.is_file() and os.access(task_python, os.X_OK) else sys.executable
    code = run_harness_evaluation(experiment_dir=experiment_dir, attempt_id=attempt_id, idea_id=idea_id,
                                 rubrics_path=rubrics_file, python_bin=python_bin,
                                 global_memory_md=global_memory_dir / "memory.md",
                                 global_memory_jsonl=global_memory_dir / "memory.jsonl",
                                 timeout_seconds=timeout_seconds, sealed_config=sealed,
                                 environment=environment, gpu_evaluation=gpu_evaluation)
    return RunAttemptResult(output="Host repeated frozen candidate under the unchanged development protocol.\n", exit_code=code, experiment_dir=experiment_dir)


def resume_cycle(results_dir: Path, outer_state: dict[str, Any]) -> int:
    """Recover an idea drafted before the old current_cycle write was completed."""
    from src.orchestrator import state as state_store
    pending = outer_state.get("pending_cycle")
    if type(pending) is int and pending > 0:
        return pending
    current = int(outer_state.get("current_cycle", 0))
    ideas = state_store.tail_jsonl(state_store.ideas_history_path(results_dir), max_lines=100000)
    attempts = state_store.tail_jsonl(state_store.attempts_history_path(results_dir), max_lines=100000)
    completed = {row.get("cycle_index") for row in attempts if isinstance(row, dict) and not row.get("retry_scheduled")}
    drafts = [row for row in ideas if isinstance(row, dict) and row.get("cycle_index") == current]
    if current > 0 and drafts and current not in completed:
        outer_state["pending_cycle"] = current
        outer_state["pending_idea_dir"] = str(state_store.idea_dir(results_dir, drafts[-1]["idea_dir_name"]))
        return current
    return current + 1
