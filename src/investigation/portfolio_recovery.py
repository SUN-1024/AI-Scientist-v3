"""Explicit host-evaluation recovery for an unchanged, failed portfolio lease.

This command never starts an idea or coding agent. It retains the paid candidate,
archives host diagnostics outside that candidate, and counts only validated host
receipts. Ordinary portfolio resume does not call this recovery path.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

from src.orchestrator import state as state_store
from .adaptive_controller import _digest
from .evaluation_usage import validate_compute_usage
from .harness_eval import SealedHarnessConfig, run_harness_evaluation
from .harness_incumbent import _program_source_records
from .parallel_helpers import portfolio_process_identity
from .portfolio_campaign import (
    LEASE_FILE, PROCESS_FILE, WORKER_LOCK, _assert_no_lease_processes,
    _exclusive_lock, _publish_outcome, _read_object, _receipt_for_lease,
    _recorded_process_is_live, bind_portfolio_idea, portfolio_coordinator_lock,
)
from .portfolio_controller import (
    PortfolioConfig, PortfolioController, _open_root, read_portfolio_state,
    validate_external_lease,
)
from .search_tree import _fingerprint

RECOVERY_DIRECTORY = "portfolio_recovery"
_MAX_FILE_BYTES = 32 * 1024 * 1024
_DIAGNOSTICS = (
    "evaluation_failure.json", "stdout.log", "stderr.log",
    "harness_evaluate_stdout.log", "harness_evaluate_stderr.log",
)
_RESULTS = ("evaluation_manifest.json", "metrics.json", "correct.json", "evaluation_evidence.json")
_RECOVERABLE_REASONS = {
    "transport_failure_budget_exhausted",
    "nonretryable_failure:evaluation_only_recovery_required",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bytes(path: Path) -> bytes:
    parent_fd = _open_root(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"Recovery input is not a bounded regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read(_MAX_FILE_BYTES + 1)
        if len(value) > _MAX_FILE_BYTES:
            raise ValueError(f"Recovery input is too large: {path}")
        return value
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def _sha(path: Path) -> str:
    return hashlib.sha256(_bytes(path)).hexdigest()


def _save(path: Path, payload: dict[str, Any]) -> None:
    payload["ledger_sha256"] = _digest({key: value for key, value in payload.items() if key != "ledger_sha256"})
    state_store.write_private_text(path, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    fd = _open_root(path.parent)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_ledger(path: Path) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    value = json.loads(_bytes(path))
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("ledger_sha256") != _digest({k: v for k, v in value.items() if k != "ledger_sha256"})):
        raise ValueError("Recovery ledger integrity mismatch")
    for run in value["runs"]:
        for relative, expected in run.get("archived_files", {}).items():
            archived = path.parent / relative
            if not archived.is_relative_to(path.parent) or ".." in Path(relative).parts or _sha(archived) != expected:
                raise ValueError("Archived recovery evidence changed")
    return value


def _assert_stopped(worker: Path, ledger: dict[str, Any] | None) -> None:
    marker = worker / PROCESS_FILE
    if marker.exists() or marker.is_symlink():
        recorded = _read_object(marker)
        if _recorded_process_is_live(recorded):
            raise ValueError("Cannot recover while the prior portfolio worker is live")
    _assert_no_lease_processes(worker)
    if ledger and ledger["runs"]:
        previous = ledger["runs"][-1]
        if previous.get("status") in {"prepared", "evaluating"} and _recorded_process_is_live(previous):
            raise ValueError("Cannot recover while the prior host recovery process is live")


def _candidate(worker: Path, lease: dict[str, Any], context: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    programs = list(worker.glob("ideas/*/*_attempt/program_1"))
    if len(programs) != 1:
        raise ValueError("Recovery requires exactly one retained candidate for this lease")
    program = programs[0]
    descriptor = _open_root(program)
    os.close(descriptor)
    if not program.is_relative_to(worker):
        raise ValueError("Recovery candidate escaped its lease")
    bundle = _read_object(program.parent.parent / "idea_bundle.json")
    if bundle.get("idea_id") != lease["root_idea_id"]:
        raise ValueError("Recovery candidate belongs to a different root idea")
    bundle = bind_portfolio_idea(bundle, context)
    number = int(program.parent.name.split("_", 1)[0])
    if number < 1 or program.parent.name != f"{number:02d}_attempt":
        raise ValueError("Recovery requires the existing one-attempt layout")
    attempt_id = f"{lease['root_idea_id']}_attempt_{number:02d}"
    if _fingerprint(program.parent / "program_0") != lease["parent_source_hash"]:
        raise ValueError("Recovery attempt parent differs from its leased source")
    return program, bundle, attempt_id


def _sealed_settings(worker: Path, task: Path, lease: dict[str, Any]) -> tuple[SealedHarnessConfig, int, Path]:
    # Read the saved host configuration; do not recompute a new run configuration.
    import yaml

    saved = yaml.safe_load(_bytes(worker / ".hydra" / "config.yaml"))
    if not isinstance(saved, dict):
        raise ValueError("Recovery requires the original worker configuration")
    container, execution, paths = (saved.get(key, {}) for key in ("container", "execution", "paths"))
    image = container.get("image")
    timeout = execution.get("harness_evaluation_timeout_seconds")
    gpu = execution.get("gpu_evaluation")
    if (container.get("enabled") is not True or not isinstance(image, str) or not image
            or type(timeout) is not int or timeout <= 0 or type(gpu) is not bool
            or execution.get("harness_evaluated_attempts") is not True
            or Path(paths.get("task_dir", "")).absolute() != task
            or Path(paths.get("results_dir", "")).absolute() != worker
            or Path(paths.get("seed_program", "")).absolute() != Path(lease["parent_program_dir"])):
        raise ValueError("Recovery configuration differs from its sealed host lease")
    selector = lease.get("gpu_selector")
    if gpu and not selector:
        raise ValueError("Recovery cannot change or guess its leased GPU selector")
    environment = dict(os.environ, TASK_SOURCE_DIR=str(task))
    if selector:
        environment["AGENT_CONTAINER_GPUS"] = selector
    python = task / ".venv" / "bin" / "python"
    _bytes(python)  # The canonical container runtime marker must be a regular file.
    return SealedHarnessConfig(task, image, selector, environment, gpu), timeout, python


def _transport_evidence(worker: Path, program: Path, attempt_id: str, lease: dict[str, Any]) -> dict[str, Any]:
    if any((program / name).exists() or (program / name).is_symlink() for name in _RESULTS):
        raise ValueError("Invalid or partial host receipt; refusing to erase evaluation results")
    failure = json.loads(_bytes(program / "evaluation_failure.json"))
    validate_compute_usage(failure)
    # Legacy failure.json records measured compute only. The host attempt record
    # must independently name the infrastructure exception and the exact attempt.
    history = _bytes(state_store.attempts_history_path(worker)).decode()
    rows = [json.loads(line) for line in history.splitlines() if line.strip()]
    trusted = [row for row in rows if isinstance(row, dict)
               and row.get("attempt_id") == attempt_id
               and row.get("idea_id") == lease["root_idea_id"]
               and row.get("run_exit_code") not in (None, 0)
               and "CandidateContainerCleanupError:" in str(row.get("output_tail", ""))]
    if not trusted:
        raise ValueError("Host attempt history does not establish a recoverable transport failure")
    return {"exception_type": "CandidateContainerCleanupError", "compute_usage": failure,
            "attempt_history_sha256": hashlib.sha256(history.encode()).hexdigest()}


def _archive(path: Path, run: dict[str, Any], files: list[tuple[Path, str]]) -> None:
    run["archived_files"] = {}
    for source, name in files:
        if not source.exists() and not source.is_symlink():
            continue
        payload = _bytes(source)
        relative = f"run_{run['sequence']:04d}/{name}"
        target = path.parent / relative
        # A fresh run has an unused directory. Never overwrite archived bytes.
        if target.exists() or target.is_symlink():
            raise ValueError("Recovery archive already exists")
        state_store.write_private_text(target, payload.decode("utf-8"))
        descriptor = _open_root(target.parent)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        run["archived_files"][relative] = hashlib.sha256(payload).hexdigest()


def _source_evidence(path: Path | None, worker: Path, program: Path) -> dict[str, Any]:
    if path is None:
        raise ValueError("Evaluation-only recovery requires --evidence-file with retained source hashes")
    payload = _bytes(path)
    evidence = json.loads(payload)
    if not isinstance(evidence, dict) or not isinstance(evidence.get("captured_at"), str):
        raise ValueError("Recovery evidence needs its observation time")
    records = evidence.get("candidate_source_files")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Recovery evidence lacks observed candidate source hashes")
    expected = {row.get("path"): row.get("sha256") for row in records}
    if len(expected) != len(records) or expected != _program_source_records(program):
        raise ValueError("Recovery source differs from the retained candidate evidence")
    files = evidence.get("files")
    if not isinstance(files, list):
        raise ValueError("Recovery evidence lacks host failure records")
    wanted = {str(worker / LEASE_FILE): False,
              str(worker / "attempts_history.jsonl"): False,
              str(program / "evaluation_failure.json"): False}
    for row in files:
        if not isinstance(row, dict) or row.get("source") not in wanted:
            continue
        original = row.get("text")
        if not isinstance(original, str) or hashlib.sha256(original.encode()).hexdigest() != row.get("sha256"):
            raise ValueError("Retained host failure evidence has an invalid file hash")
        current = _bytes(Path(row["source"]))
        if current != original.encode():
            raise ValueError("Host failure evidence changed since it was retained")
        wanted[row["source"]] = True
    if not all(wanted.values()):
        raise ValueError("Recovery evidence must contain the original lease, failure, and attempt history")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "captured_at": evidence["captured_at"],
            "source_binding_scope": "observed_after_failure_before_recovery",
            "original_evaluation_prelaunch_binding": "unavailable"}


def _publication(controller: PortfolioController, lease: dict[str, Any], outcome: dict[str, Any]) -> bool:
    root = controller.results_dir
    outer = _read_object(root / state_store.OUTER_STATE_FILE)
    best = _read_object(root / state_store.BEST_MANIFEST_FILE)
    _publish_outcome(results_dir=root, outer_state=outer, best_manifest=best, lease=lease, outcome=outcome)
    committed = controller.record_receipt(lease["lease_id"], outcome)
    controller.resume_after_transport_recovery()
    outer["portfolio_status"] = controller.status
    state_store.save_outer_state(root, outer)
    return committed


def recover_lease(*, results_dir: Path, task_dir: Path, lease_id: str,
                  evidence_file: Path | None = None) -> dict[str, Any]:
    """Recover one failed reservation without granting a new scientific trial."""
    root, task = Path(results_dir).expanduser().absolute(), Path(task_dir).expanduser().absolute()
    # The root must already exist and must not resolve through symlinks.
    for directory in (root, task):
        descriptor = _open_root(directory)
        os.close(descriptor)
    with portfolio_coordinator_lock(root):
        state = read_portfolio_state(root, task)
        lease = state["leases"].get(lease_id)
        if not lease or not any(row["lease_id"] == lease_id for row in state["failures"]):
            raise ValueError("Recovery requires an existing failed portfolio lease")
        worker = Path(lease["result_dir"])
        if worker != root / "portfolio_trials" / lease_id:
            raise ValueError("Recovery worker path differs from its host lease")
        ledger_path = root / RECOVERY_DIRECTORY / lease_id / "ledger.json"
        with _exclusive_lock(worker / WORKER_LOCK):
            ledger = _load_ledger(ledger_path)
            _assert_stopped(worker, ledger)
            envelope = _read_object(worker / LEASE_FILE)
            if envelope != {"root_results_dir": str(root), "lease": lease}:
                raise ValueError("Recovery lease envelope differs from the host record")
            controller = PortfolioController(root, task, PortfolioConfig(**state["policy"]))
            context = controller.context(lease)
            if lease_id in state["pending"]:
                validate_external_lease(envelope, task, worker, Path(lease["parent_program_dir"]))
            else:
                controller._check_parent(lease["parent"])
            program, bundle, attempt_id = _candidate(worker, lease, context)
            identity = {"lease_sha256": _digest(lease), "task_hash": state["task_hash"],
                        "protocol_hash": state["protocol_hash"], "policy_hash": state["policy_hash"],
                        "plan_sha256": state.get("plan_sha256"), "card_sha256": lease.get("card_sha256"),
                        "parent_source_hash": lease["parent_source_hash"], "source_hash": _fingerprint(program),
                        "source_files_sha256": _program_source_records(program),
                        "idea_bundle_sha256": _sha(program.parent.parent / "idea_bundle.json"),
                        "program_dir": str(program), "attempt_id": attempt_id}
            if ledger is not None and ledger.get("identity") != identity:
                raise ValueError("Recovery source, card, parent, task, or lease changed")
            # A valid receipt always wins over a stale diagnostic. Never pay twice.
            outcome = _receipt_for_lease(lease, task, context)
            if lease_id in state["outcomes"]:
                expected = deepcopy(outcome)
                if expected is not None and "plan_sha256" in state:
                    expected.update({field: lease.get(field) for field in ("plan_sha256", "card_sha256", "role")})
                if expected is None or expected != state["outcomes"][lease_id]:
                    raise ValueError("Completed recovery receipt no longer validates")
            elif state["status"] != "incomplete" or state["transition_reason"] not in _RECOVERABLE_REASONS:
                raise ValueError("Evaluation recovery requires an explicit transport hard stop")
            if ledger is None:
                ledger = {"schema_version": 1, "lease_id": lease_id, "identity": identity,
                          "created_at": _now(), "runs": []}
            if outcome is not None:
                if ledger.get("status") == "receipt_committed" and lease_id in state["outcomes"]:
                    return {"lease_id": lease_id, "evaluated": False, "committed": False, "status": controller.status}
                receipt_run = {"sequence": len(ledger["runs"]), "started_at": _now(),
                               "status": "validating_receipt", "evaluation_performed": False,
                               "source_hash": identity["source_hash"], "receipt_id": outcome["receipt_id"]}
                _archive(ledger_path, receipt_run,
                    [(program / name, "program/" + name) for name in (*_DIAGNOSTICS, *_RESULTS)])
                ledger["runs"].append(receipt_run)
                _save(ledger_path, ledger)
                try:
                    committed = _publication(controller, lease, outcome)
                    ledger.update(status="receipt_committed", receipt_id=outcome["receipt_id"], finished_at=_now())
                    receipt_run.update(status="receipt_committed", compute_usage=_read_object(
                        program / "evaluation_manifest.json").get("compute_usage"))
                    for interrupted in ledger["runs"][:-1]:
                        if interrupted["status"] in {"prepared", "evaluating"}:
                            interrupted.update(status="receipt_recovered", completion_observed_at=_now(),
                                               receipt_id=outcome["receipt_id"])
                except BaseException as exc:
                    receipt_run.update(status="publication_failed", error={"type": type(exc).__name__, "message": str(exc)})
                    raise
                finally:
                    receipt_run["finished_at"] = _now()
                    _save(ledger_path, ledger)
                return {"lease_id": lease_id, "evaluated": False, "committed": committed, "status": controller.status}
            if ledger["runs"] and ledger["runs"][-1]["status"] in {"evaluating", "failed"}:
                raise ValueError("A recovery evaluation already ran without a valid receipt; retained for inspection")
            evidence = _transport_evidence(worker, program, attempt_id, lease)
            source_evidence = _source_evidence(evidence_file, worker, program)
            assert evidence_file is not None
            sealed, timeout, python = _sealed_settings(worker, task, lease)
            rubrics = program.parent / "active_rubrics.json"
            if json.loads(_bytes(rubrics)) != bundle.get("target_rubrics", []):
                raise ValueError("Recovery rubrics differ from the original idea card")
            if _sha(program.parent / "evaluate.py") != _sha(task / "evaluate.py"):
                raise ValueError("Recovery evaluator differs from the frozen task")
            run = {"sequence": len(ledger["runs"]), "started_at": _now(), "status": "prepared", "evaluation_invoked": False,
                   "pid": os.getpid(), "process_identity": portfolio_process_identity(os.getpid()),
                   "source_hash": identity["source_hash"], "evidence": evidence,
                   "retained_source_evidence": source_evidence,
                   "configuration_sha256": _sha(worker / ".hydra" / "config.yaml"),
                   "timeout_seconds": timeout, "sandbox_image": sealed.sandbox_image,
                   "gpu_selector": lease.get("gpu_selector"), "error": None}
            files = [(program / name, "program/" + name) for name in _DIAGNOSTICS]
            files += [(program.parent / "wrapper_output.log", "wrapper_output.log"),
                      (state_store.attempts_history_path(worker), "attempts_history.jsonl"),
                      (worker / PROCESS_FILE, PROCESS_FILE), (worker / LEASE_FILE, LEASE_FILE),
                      (worker / ".hydra" / "config.yaml", "config.yaml"),
                      (Path(evidence_file), "retained_failure_evidence.json")]
            _archive(ledger_path, run, files)
            ledger["runs"].append(run)
            _save(ledger_path, ledger)
            started = time.monotonic()
            try:
                # Only host diagnostics are moved out of the unchanged source tree.
                for name in _DIAGNOSTICS:
                    file = program / name
                    relative = f"run_{run['sequence']:04d}/program/{name}"
                    if relative in run["archived_files"]:
                        if _sha(file) != run["archived_files"][relative]:
                            raise ValueError("Host diagnostics changed while archiving")
                        file.unlink()
                if _fingerprint(program) != identity["source_hash"]:
                    raise ValueError("Recovery source changed before host evaluation")
                run["status"] = "evaluating"
                run["evaluation_invoked"] = True
                _save(ledger_path, ledger)
                returncode = run_harness_evaluation(
                    experiment_dir=program.parent, attempt_id=attempt_id, idea_id=lease["root_idea_id"],
                    rubrics_path=rubrics, python_bin=str(python),
                    global_memory_md=worker / "memory.md", global_memory_jsonl=worker / "memory.jsonl",
                    timeout_seconds=timeout, sealed_config=sealed, environment=sealed.environment,
                    gpu_evaluation=sealed.gpu_evaluation)
                run["returncode"] = returncode
                if _fingerprint(program) != identity["source_hash"]:
                    raise ValueError("Recovery source changed during host evaluation")
                outcome = _receipt_for_lease(lease, task, context)
                if outcome is None:
                    raise ValueError("Recovery evaluation did not produce a valid host receipt")
                committed = _publication(controller, lease, outcome)
                run.update(status="receipt_committed", receipt_id=outcome["receipt_id"],
                           compute_usage=_read_object(program / "evaluation_manifest.json").get("compute_usage"))
                ledger.update(status="receipt_committed", receipt_id=outcome["receipt_id"])
                return {"lease_id": lease_id, "evaluated": True, "committed": committed, "status": controller.status}
            except BaseException as exc:
                run.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                failure = program / "evaluation_failure.json"
                if failure.exists() and not failure.is_symlink():
                    try:
                        run["compute_usage"] = _read_object(failure)
                    except (OSError, ValueError) as diagnostic_error:
                        run["compute_usage_error"] = str(diagnostic_error)
                ledger["status"] = "failed"
                raise
            finally:
                run.update(finished_at=_now(), elapsed_seconds=max(0.0, time.monotonic() - started))
                run["produced_files_sha256"] = {}
                for name in (*_RESULTS, "evaluation_failure.json"):
                    output = program / name
                    if output.exists() or output.is_symlink():
                        try:
                            run["produced_files_sha256"][name] = _sha(output)
                            if name == "evaluation_manifest.json" and "compute_usage" not in run:
                                run["compute_usage"] = _read_object(output).get("compute_usage")
                        except (OSError, ValueError) as diagnostic_error:
                            run.setdefault("output_inspection_errors", {})[name] = str(diagnostic_error)
                _save(ledger_path, ledger)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--lease-id", required=True)
    parser.add_argument("--evidence-file", type=Path,
                        help="Retained post-failure source and host evidence; required only for a new evaluation")
    args = parser.parse_args(argv)
    result = recover_lease(results_dir=args.results_dir, task_dir=args.task_dir, lease_id=args.lease_id,
                           evidence_file=args.evidence_file)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
