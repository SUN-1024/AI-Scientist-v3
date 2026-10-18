"""Execute durable whole-idea leases through the existing V3 worker boundary.

Only this coordinator writes portfolio decisions. Workers use one isolated result
folder and a fixed GPU slot. A completed host receipt is recovered before any
new process is started, including receipts whose correctness result is false.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Callable, Iterator

from src.orchestrator import state as state_store
from .adaptive_runtime import validated_development_outcome
from .campaign_retry import has_evaluation_artifacts
from .container_gpus import GPU_SELECTOR_ENV
from .models import ParallelWorkerPlan, ParallelWorkerResult
from .parallel_helpers import portfolio_process_identity, worker_gpu_environment
from .program_spec import entrypoint_relpath_from_spec
from .search_tree import _fingerprint
from .task_failure import HOST_FAILURES

LEASE_FILE = "portfolio_lease.json"
PROCESS_FILE = "portfolio_process.json"
WORKER_LOCK = ".portfolio_worker.lock"
COORDINATOR_LOCK = ".portfolio_coordinator.lock"
_COORDINATOR_LOCKS = threading.local()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Invalid portfolio file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Portfolio file is not an object: {path}")
    return value


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Portfolio work is already active: {path.parent}") from exc
        yield
    finally:
        os.close(descriptor)



@contextmanager
def portfolio_coordinator_lock(results_dir: Path) -> Iterator[None]:
    """Lock before bootstrap; nested calls in the owning thread reuse it.

    Thread-local ownership is deliberate. Another thread must obtain its own
    OS lock and cannot borrow this thread's right to write campaign state.
    """
    key = str(results_dir.resolve())
    held = getattr(_COORDINATOR_LOCKS, "held", None)
    if held is None or getattr(_COORDINATOR_LOCKS, "owner_pid", None) != os.getpid():
        held = set()
        _COORDINATOR_LOCKS.held = held
        _COORDINATOR_LOCKS.owner_pid = os.getpid()
    if key in held:
        yield
        return
    with _exclusive_lock(Path(key) / COORDINATOR_LOCK):
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)


def load_external_portfolio_context(*, cfg: Any, task_dir: Path,
                                    results_dir: Path, seed_program_dir: Path) -> dict[str, Any] | None:
    """Read and validate a host lease without changing coordinator state."""
    raw = getattr(cfg.execution, "portfolio_lease_file", None)
    if not raw:
        return None
    from .portfolio_controller import validate_external_lease
    path = Path(raw).expanduser().resolve()
    if path.parent != results_dir.resolve() or path.name != LEASE_FILE:
        raise ValueError("External portfolio lease must belong to this worker result directory")
    context = validate_external_lease(_read_object(path), task_dir=task_dir,
                                      results_dir=results_dir, seed_program_dir=seed_program_dir)
    if cfg.adaptive_investigation.enabled or cfg.portfolio.enabled or cfg.execution.workers != 1:
        raise ValueError("Portfolio worker cannot schedule its own portfolio or adaptive actions")
    if cfg.execution.max_idea_cycles != 1 or cfg.execution.max_program_evals_per_attempt != 1 or cfg.paper.enabled:
        raise ValueError("Portfolio lease must run exactly one development evaluation and no paper generation")
    if not cfg.execution.harness_evaluated_attempts:
        raise ValueError("Portfolio lease requires host evaluation receipts")
    return {**context, "host_verified": True}


@contextmanager
def external_portfolio_worker_lock(*, cfg: Any, task_dir: Path, results_dir: Path) -> Iterator[None]:
    """Hold this before bootstrap, so orphan starts cannot modify the same lease."""
    seed = getattr(cfg.paths, "seed_program", None)
    if not seed:
        raise ValueError("Portfolio child requires the exact leased parent source")
    with _exclusive_lock(results_dir / WORKER_LOCK):
        load_external_portfolio_context(cfg=cfg, task_dir=task_dir, results_dir=results_dir,
                                        seed_program_dir=Path(seed).expanduser().resolve())
        yield



def bind_portfolio_idea(bundle: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Bind an agent proposal to its host root without changing its claim.

    Several candidate parameter defaults may change together. The parameter
    space and task/evaluator defaults remain fixed within the same root.
    """
    from copy import deepcopy
    from src.research.whole_idea_policy import normalize_whole_idea
    result = deepcopy(bundle)
    result["idea_id"] = context["root_idea_id"]
    result["root_idea_id"] = context["root_idea_id"]
    result["portfolio_lease_id"] = context["lease_id"]
    if context["stage"] == "replication":
        if context.get("source_card") is not None:
            result["whole_idea"] = deepcopy(context["source_card"])
        # Report the frozen candidate's actual selected settings, which may
        # differ from the original screen's defaults after joint tuning.
        frozen_lease_id = context.get("parent", {}).get("lease_id")
        for row in context.get("idea_history", []):
            if row.get("lease_id") == frozen_lease_id:
                frozen_bundle = row.get("outcome", {}).get("idea_bundle", {})
                if "whole_idea" in frozen_bundle:
                    result["whole_idea"] = deepcopy(frozen_bundle["whole_idea"])
                break
        return result
    proposed = normalize_whole_idea(result.get("whole_idea"))
    if proposed["innovation_axis"] != context["innovation_axis"]:
        raise ValueError("Whole idea does not use its host-assigned innovation axis")
    if context["stage"] == "tuning":
        source = normalize_whole_idea(context.get("source_card"))
        for key in ("problem_statement", "contribution", "innovation_axis", "minimal_validation"):
            if proposed[key] != source[key]:
                raise ValueError(f"Tuning cannot change root whole_idea.{key}; propose a separate root")
        original_parameters = {item["name"]: item for item in source["parameter_space"]}
        chosen_parameters = {item["name"]: item for item in proposed["parameter_space"]}
        if original_parameters.keys() != chosen_parameters.keys():
            raise ValueError("Tuning cannot replace the root parameter space")
        for name, original in original_parameters.items():
            chosen = chosen_parameters[name]
            if {key: value for key, value in original.items() if key != "default"} != {
                    key: value for key, value in chosen.items() if key != "default"}:
                raise ValueError(f"Tuning cannot change parameter bounds or ownership: {name}")
            if original["owner"] != "candidate" and chosen["default"] != original["default"]:
                raise ValueError(f"Tuning cannot change task/evaluator parameter: {name}")
        for row in context.get("idea_history", []):
            if row.get("stage") == "screening":
                original_bundle = row.get("outcome", {}).get("idea_bundle", {})
                for key in ("title", "hypothesis", "research_basis"):
                    if key in original_bundle:
                        result[key] = deepcopy(original_bundle[key])
                break
    result["whole_idea"] = proposed
    return result


def _set_override(command: list[str], name: str, value: Any) -> None:
    prefix = name + "="
    command[:] = [item for item in command if not item.startswith(prefix)]
    rendered = str(value).lower() if type(value) is bool else str(value)
    command.append(prefix + rendered)


def _slot_gpu_selector(cfg: Any, worker_id: int) -> str | None:
    selector = worker_gpu_environment(cfg, worker_id).get(GPU_SELECTOR_ENV)
    if selector is None and cfg.container.gpu_count > 0 and cfg.execution.workers == 1:
        selector = os.environ.get(GPU_SELECTOR_ENV)
    if cfg.container.gpu_count > 0 and not selector:
        raise ValueError("Portfolio GPU slots must be allocated before dispatch")
    return selector


def build_portfolio_worker_plan(*, cfg: Any, task_dir: Path, results_dir: Path,
                                instruct_path: Path, lease: dict[str, Any],
                                context: dict[str, Any], build_parallel_worker_command_fn: Callable[..., list[str]],
                                recover_corrupt_lease: bool = False) -> ParallelWorkerPlan:
    worker_dir = Path(lease["result_dir"]).resolve()
    if not worker_dir.is_relative_to(results_dir.resolve()) or worker_dir == results_dir.resolve():
        raise ValueError("Portfolio child result directory escapes the campaign")
    worker_dir.mkdir(parents=True, exist_ok=True)
    lease_path = worker_dir / LEASE_FILE
    envelope = {"root_results_dir": str(results_dir.resolve()), "lease": lease}
    publish_envelope = not lease_path.exists()
    if lease_path.exists():
        try:
            previous = _read_object(lease_path)
        except json.JSONDecodeError:
            if not recover_corrupt_lease:
                raise
            # The coordinator has re-read the exact durable root lease and
            # verified old work stopped. Restore only that authoritative lease.
            publish_envelope = True
        else:
            if previous != envelope:
                raise ValueError("Previously dispatched portfolio lease changed")
    if publish_envelope:
        state_store.write_private_text(lease_path, json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    worker_index = int(lease["worker_id"])
    command = build_parallel_worker_command_fn(
        cfg=cfg, task_dir=task_dir, instruct_path=instruct_path, results_dir=worker_dir,
        global_memory_dir=worker_dir, conclusions_dir=worker_dir, shared_memory_dir=None,
        assigned_cycles=1, seed_program=Path(lease["parent_program_dir"]),
        worker_index=worker_index, resume=(worker_dir / state_store.OUTER_STATE_FILE).exists(),
        global_cycle=lease["cycle"],
    )
    for key, value in {
        "execution.portfolio_lease_file": lease_path,
        "portfolio.enabled": False, "adaptive_investigation.enabled": False,
        "execution.workers": 1, "execution.max_idea_cycles": 1,
        "execution.max_program_evals_per_attempt": 1,
        "execution.harness_evaluated_attempts": True,
        "execution.search_strategy": "best", "execution.prune_every_n_cycles": 0,
        "paper.enabled": False, "container.gpu_count": cfg.container.gpu_count,
    }.items():
        _set_override(command, key, value)
    environment = worker_gpu_environment(cfg, worker_index)
    selector = lease.get("gpu_selector")
    if selector:
        environment[GPU_SELECTOR_ENV] = selector
    return ParallelWorkerPlan(
        worker_index=worker_index, assigned_cycles=1, cycle_start=lease["cycle"], cycle_end=lease["cycle"],
        results_dir=worker_dir, command=command, cycle_map={1: lease["cycle"]},
        lease_index=lease["cycle"], log_path=worker_dir / "parallel_worker.log", environment=environment,
    )


def _pid_exists(pid: Any) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorded_process_is_live(process: dict[str, Any]) -> bool:
    pid = process.get("pid")
    if not _pid_exists(pid):
        return False
    expected = process.get("process_identity")
    if not isinstance(expected, dict):
        # A legacy live PID alone cannot prove either continued ownership or
        # PID reuse. Do not treat a missing start token as proof of shutdown.
        return True
    if expected.get("pid") != pid:
        raise ValueError("Portfolio process identity does not match its PID")
    if expected.get("platform") == "linux":
        from .gpu_owner_process import ProcessIdentity, ProcessIdentityError
        try:
            ProcessIdentity.from_fields({key: value for key, value in expected.items() if key != "platform"})
        except ProcessIdentityError as exc:
            raise ValueError("Invalid portfolio Linux process identity") from exc
    elif expected.get("platform") == "posix-ps":
        import re
        if (type(expected.get("uid")) is not int or expected["uid"] < 0
                or not isinstance(expected.get("start_time"), str) or len(expected["start_time"].split()) != 5
                or not isinstance(expected.get("command_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", expected["command_sha256"]) is None):
            raise ValueError("Invalid portfolio POSIX process identity")
    else:
        raise ValueError("Unknown portfolio process identity platform")
    observed = portfolio_process_identity(pid)
    if observed is None:
        raise ValueError("Cannot establish identity of a live portfolio worker PID")
    if observed.keys() != expected.keys() or expected.get("platform") != observed.get("platform"):
        raise ValueError("Invalid portfolio process identity")
    return expected == observed


def _assert_no_lease_processes(worker_dir: Path) -> None:
    """Inspect live argv when a legacy marker is absent, stale, or truncated.

    Every real portfolio child receives this exact lease argument before it can
    enter bootstrap. Query errors stay errors, rather than assuming no process.
    """
    try:
        result = subprocess.run(["ps", "-ww", "-axo", "pid=,args="], capture_output=True,
                                text=True, timeout=5, check=False, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Cannot inspect prior portfolio worker processes") from exc
    if result.returncode or not result.stdout.strip() or len(result.stdout) > 8 * 1024 * 1024:
        raise ValueError("Prior portfolio process inventory is unavailable")
    identity_argument = "execution.portfolio_lease_file=" + str((worker_dir / LEASE_FILE).resolve())
    if identity_argument in result.stdout:
        raise ValueError("Cannot resume while a process for this portfolio lease is live")


def _confirm_old_work_stopped(*, cfg: Any, worker_dir: Path, stop_container_session_fn: Callable[..., bool],
                              build_container_session_name_fn: Callable[..., str], log_event_fn: Callable[..., None]) -> None:
    marker = worker_dir / PROCESS_FILE
    # Keep the lock through process and container inspection. A real child must
    # hold this lock before it can perform bootstrap, coding or evaluation.
    with _exclusive_lock(worker_dir / WORKER_LOCK):
        malformed_marker = False
        if marker.exists():
            try:
                process = _read_object(marker)
            except json.JSONDecodeError:
                malformed_marker = True
            else:
                if process.get("status") != "exited" and _recorded_process_is_live(process):
                    raise ValueError(f"Cannot resume while prior worker PID {process['pid']} is live")
        _assert_no_lease_processes(worker_dir)
        if cfg.container.enabled:
            # A resumed campaign may have changed its backend pool. All these
            # bounded names belong to this exact isolated lease directory.
            from .config import VALID_AGENTS
            for agent in VALID_AGENTS:
                name = build_container_session_name_fn(agent=agent, results_dir=worker_dir)
                stopped = stop_container_session_fn(name, log_event_fn=log_event_fn)
                if not stopped:
                    try:
                        inspection = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", name],
                                                    capture_output=True, text=True, timeout=15, check=False)
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        raise ValueError(f"Cannot inspect previous container: {name}") from exc
                    absent = inspection.returncode != 0 and "No such" in inspection.stderr
                    inactive = inspection.returncode == 0 and inspection.stdout.strip() == "false"
                    if not (absent or inactive):
                        raise ValueError(f"Cannot confirm previous container stopped: {name}")
        if malformed_marker:
            log_event_fn("portfolio_process_marker_recovered", results_dir=str(worker_dir),
                         evidence="exclusive_worker_lock_and_no_matching_process_or_container")


def _receipt_for_lease(lease: dict[str, Any], task_dir: Path, context: dict[str, Any] | None = None) -> dict[str, Any] | None:
    worker_dir = Path(lease["result_dir"])
    candidates: list[dict[str, Any]] = []
    # A host receipt may exist before the child wrote its history row. Scan the
    # fixed one-cycle attempt layout, never trust a child best manifest.
    for program in sorted(worker_dir.glob("ideas/*/*_attempt/program_1")):
        ancestry = [program, program.parent, program.parent.parent, program.parent.parent.parent]
        if (any(path.is_symlink() for path in ancestry) or not program.is_dir()
                or not program.resolve().is_relative_to(worker_dir.resolve())):
            continue
        idea_file = program.parent.parent / "idea_bundle.json"
        try:
            bundle = _read_object(idea_file)
            idea_id = bundle.get("idea_id")
            if idea_id != lease["root_idea_id"]:
                continue
            if context is not None:
                bundle = bind_portfolio_idea(bundle, context)
            number = int(program.parent.name.split("_", 1)[0])
            attempt_id = f"{idea_id}_attempt_{number:02d}"
            outcome = validated_development_outcome(program, task_dir=task_dir, attempt_id=attempt_id,
                                                    idea_id=idea_id, native=False)
            if outcome:
                candidates.append({**outcome, "idea_bundle": bundle, "idea_file": str(idea_file),
                                   "lease_id": lease["lease_id"], "root_idea_id": lease["root_idea_id"]})
        except (ValueError, OSError, TypeError):
            continue
    if len(candidates) > 1:
        raise ValueError("One portfolio lease produced more than one valid evaluation receipt")
    return candidates[0] if candidates else None


def _publish_outcome(*, results_dir: Path, outer_state: dict[str, Any], best_manifest: dict[str, Any],
                     lease: dict[str, Any], outcome: dict[str, Any]) -> None:
    """Publish serial, append-only root histories after host validation."""
    root_id, cycle = lease["root_idea_id"], lease["cycle"]
    bundle = outcome["idea_bundle"]
    root_idea_dir = state_store.idea_dir(results_dir, root_id)
    root_idea_dir.mkdir(parents=True, exist_ok=True)
    card = root_idea_dir / "idea_bundle.json"
    if not card.exists():
        state_store.write_json(card, bundle)
    state_store.append_jsonl_if_absent(state_store.ideas_history_path(results_dir), {
        "idea_id": root_id, "idea_dir_name": root_id, "title": bundle.get("title"),
        "cycle_index": cycle, "target_rubrics": bundle.get("target_rubrics", []),
        "portfolio_root_idea": True, "innovation_axis": lease.get("innovation_axis"),
    }, matching={"idea_id": root_id})
    best_score = best_manifest.get("combined_score")
    improved = (outcome["correct"] is True and lease["stage"] != "replication" and
                (best_manifest.get("correct") is not True or best_score is None or outcome["score"] > best_score))
    if improved:
        best_manifest.clear()
        best_manifest.update({"program_dir": outcome["program_dir"],
            "entrypoint_path": entrypoint_relpath_from_spec(Path(outcome["program_dir"]), required=False) or "src/program.py",
            "combined_score": outcome["score"], "correct": True, "source_attempt_id": outcome["attempt_id"],
            "portfolio_lease_id": lease["lease_id"], "root_idea_id": root_id, "source_hash": outcome["source_hash"],
            "updated_at": datetime.now(timezone.utc).isoformat()})
        state_store.save_best_manifest(results_dir, best_manifest)
    state_store.append_jsonl_if_absent(state_store.attempts_history_path(results_dir), {
        "cycle_index": cycle, "idea_id": root_id, "idea_dir_name": root_id,
        "attempt_id": f"{lease['lease_id']}:{outcome['attempt_id']}", "worker_attempt_id": outcome["attempt_id"],
        "portfolio_lease_id": lease["lease_id"], "portfolio_stage": lease["stage"],
        "trial_index": lease["trial_index"], "experiment_dir": str(Path(outcome["program_dir"]).parent),
        "run_exit_code": 0, "evaluation_performed": True, "scientific_evaluation": True,
        "retry_scheduled": False, "attempt_status": "pass" if outcome["correct"] else "fail",
        "metrics": {"combined_score": outcome["score"], "correct": outcome["correct"]},
        "receipt_id": outcome["receipt_id"], "source_hash": outcome["source_hash"], "best_updated": improved,
    }, matching={"portfolio_lease_id": lease["lease_id"]})
    outer_state.update(current_cycle=max(int(outer_state.get("current_cycle", 0)), cycle),
                       best_program_dir=best_manifest["program_dir"], best_combined_score=best_manifest.get("combined_score"),
                       best_correct=best_manifest.get("correct"), failure_streak=0)
    state_store.save_outer_state(results_dir, outer_state)


def run_portfolio_campaign(*, cfg: Any, task_dir: Path, results_dir: Path, global_memory_dir: Path,
                           instruct_path: Path, outer_state: dict[str, Any], best_manifest: dict[str, Any],
                           best_program_dir: Path, build_parallel_worker_command_fn: Callable[..., list[str]],
                           run_parallel_worker_fn: Callable[[ParallelWorkerPlan], ParallelWorkerResult],
                           log_event_fn: Callable[..., None], terminate_all_active_processes_fn: Callable[..., Any],
                           stop_container_session_fn: Callable[..., bool], build_container_session_name_fn: Callable[..., str],
                           interrupt_grace_seconds: int = 10, **_unused: Any) -> int:
    from .portfolio_controller import PortfolioController
    with portfolio_coordinator_lock(results_dir):
        controller = PortfolioController(results_dir, task_dir, cfg.portfolio)
        if controller.data.get("baseline") is None:
            controller.qualify_baseline(best_manifest, best_program_dir)
        state_store.ensure_histories(results_dir)
        if any(int(lease["worker_id"]) > cfg.execution.workers for lease in controller.pending_leases()):
            raise ValueError("Resume needs every GPU worker slot with a pending portfolio lease")
        pool = ThreadPoolExecutor(max_workers=cfg.execution.workers)
        running: dict[Future, tuple[dict[str, Any], ParallelWorkerPlan]] = {}
        active_dirs: set[Path] = set()

        def cleanup() -> None:
            terminate_all_active_processes_fn(grace_seconds=interrupt_grace_seconds)
            if cfg.container.enabled:
                for worker_dir in sorted(active_dirs):
                    for agent in sorted(set([cfg.agent.name, *cfg.agent.pool])):
                        stop_container_session_fn(build_container_session_name_fn(agent=agent, results_dir=worker_dir),
                                                  log_event_fn=log_event_fn)

        def publish(lease: dict[str, Any], outcome: dict[str, Any]) -> None:
            # Publish history first: a crash here leaves the lease pending and
            # replays this idempotent write. Controller commit is the final step.
            _publish_outcome(results_dir=results_dir, outer_state=outer_state, best_manifest=best_manifest,
                             lease=lease, outcome=outcome)
            controller.record_receipt(lease["lease_id"], outcome)
            log_event_fn("portfolio_receipt_committed", lease_id=lease["lease_id"], root_idea_id=lease["root_idea_id"],
                         stage=lease["stage"], correct=outcome["correct"], combined_score=outcome["score"])

        try:
            while True:
                busy = {plan.worker_index for _, plan in running.values()}
                for worker_id in range(1, cfg.execution.workers + 1):
                    if worker_id in busy or controller.status != "running":
                        continue
                    selector = _slot_gpu_selector(cfg, worker_id)
                    lease = controller.reserve_next(worker_id, gpu_selector=selector)
                    if lease is None:
                        continue
                    worker_dir = Path(lease["result_dir"])
                    if worker_dir.exists():
                        _confirm_old_work_stopped(cfg=cfg, worker_dir=worker_dir,
                            stop_container_session_fn=stop_container_session_fn,
                            build_container_session_name_fn=build_container_session_name_fn, log_event_fn=log_event_fn)
                    outcome = _receipt_for_lease(lease, task_dir, controller.context(lease))
                    if outcome is not None:
                        publish(lease, outcome)
                        continue
                    if any(has_evaluation_artifacts(path) for path in worker_dir.glob("ideas/*/*_attempt")):
                        controller.record_failure(lease["lease_id"], "unrecoverable_evaluation_receipt", retryable=False)
                        log_event_fn("portfolio_worker_failed", lease_id=lease["lease_id"], reason="unrecoverable_evaluation_receipt")
                        continue
                    plan = build_portfolio_worker_plan(cfg=cfg, task_dir=task_dir, results_dir=results_dir,
                        instruct_path=instruct_path, lease=lease, context=controller.context(lease),
                        build_parallel_worker_command_fn=build_parallel_worker_command_fn,
                        recover_corrupt_lease=True)
                    active_dirs.add(worker_dir)
                    # reserve_next and the immutable lease file are durable
                    # before a thread can create the subprocess.
                    future = pool.submit(run_parallel_worker_fn, plan)
                    running[future] = (lease, plan)
                    log_event_fn("portfolio_worker_lease", worker=worker_id, lease_id=lease["lease_id"],
                                 root_idea_id=lease["root_idea_id"], stage=lease["stage"], gpu_selector=selector)
                if not running:
                    # Recovery may have committed a whole stage without a spawn.
                    if controller.status == "running" and controller.pending_leases():
                        continue
                    # A commit can unlock the next stage; try one further round
                    # only when the controller can actually reserve new work.
                    if controller.status == "running":
                        probe = controller.reserve_next(1, gpu_selector=_slot_gpu_selector(cfg, 1))
                        if probe is not None:
                            continue
                    break
                completed, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: running[item][0]["cycle"]):
                    lease, plan = running.pop(future)
                    result = future.result()
                    outcome = _receipt_for_lease(lease, task_dir, controller.context(lease))
                    if outcome is not None:
                        publish(lease, outcome)
                    elif result.exit_code == 130:
                        cleanup()
                        return 130
                    else:
                        category = "worker_exit_" + str(result.exit_code) if result.exit_code else "missing_host_receipt"
                        evidence_exists = any(has_evaluation_artifacts(path) for path in plan.results_dir.glob("ideas/*/*_attempt"))
                        controller.record_failure(lease["lease_id"], category,
                                                  retryable=result.exit_code not in HOST_FAILURES and not evidence_exists)
                        log_event_fn("portfolio_worker_failed", lease_id=lease["lease_id"], reason=category)
                        if result.exit_code in HOST_FAILURES:
                            cleanup()
                            return result.exit_code
            handoff = controller.handoff()
            if handoff is not None:
                candidate = handoff["candidate"]
                program_dir = Path(candidate["program_dir"])
                if _fingerprint(program_dir) != candidate["source_hash"]:
                    raise ValueError("Portfolio final candidate changed after its host receipt")
                best_manifest.clear()
                best_manifest.update({"program_dir": str(program_dir),
                    "entrypoint_path": entrypoint_relpath_from_spec(program_dir, required=False) or "src/program.py",
                    "combined_score": candidate["score"], "correct": candidate["correct"],
                    "source_attempt_id": "bootstrap" if candidate["root_idea_id"] == "baseline" else candidate.get("attempt_id"),
                    "portfolio_lease_id": candidate["lease_id"],
                    "root_idea_id": candidate["root_idea_id"], "source_hash": candidate["source_hash"],
                    "updated_at": datetime.now(timezone.utc).isoformat()})
                state_store.save_best_manifest(results_dir, best_manifest)
                outer_state.update(best_program_dir=str(program_dir), best_combined_score=candidate["score"], best_correct=True)
            outer_state["portfolio_status"] = controller.status
            state_store.save_outer_state(results_dir, outer_state)
            return 0 if handoff is not None else 2
        except BaseException:
            # Stop children before waiting for executor shutdown. Waiting first
            # would leave a Ctrl-C blocked on a long model or GPU invocation.
            cleanup()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
