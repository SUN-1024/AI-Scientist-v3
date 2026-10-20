"""One reusable research harness over the V3 pipeline, not a second executor."""
from __future__ import annotations

import json
import fcntl
import os
import secrets
import stat
import sys
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.investigation.files import missing_task_scaffold_files
from src.final_evaluation.runner import FinalEvaluationError
from src.orchestrator.state import write_private_text
from src.pipeline.config import PIPELINE_PHASES, parse_pipeline_phases, validate_pipeline_config
from .config import ResearchConfig, load_config
from .harness_state import HarnessState, read_json, sha256, verify_stage_records, now
from .paper_readiness import write_readiness
from .preflight import ensure_execution_ready
from .workspace import prepare, resume, load_preparation, execution_overrides, record_execution, write_report

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PHASES = set(PIPELINE_PHASES) - {"task_drafting"}
RESUME_MUTABLE = {
    "execution.phases", "investigation.execution.max_idea_cycles", "investigation.execution.run_timeout_seconds",
    "investigation.execution.idea_timeout_seconds", "review.timeout_seconds", "logging.verbose",
    "logging.heartbeat_seconds", "execution.fail_fast", "posthoc.container.network",
}


def _merge(*groups: list[str]) -> list[str]:
    merged = {}
    for group in groups:
        for item in group:
            merged[item.split("=", 1)[0]] = item
    return list(merged.values())


def _dump(path: Path, payload: dict[str, Any]) -> None:
    write_private_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _source_lookup_settings(settings: dict[str, Any] | None, enabled: bool,
                            python_execution_image: str | None = None) -> dict[str, Any] | None:
    if not enabled and python_execution_image is None:
        return settings
    if settings is None:
        if python_execution_image is not None:
            raise ValueError("Python tools require the native Codex review backend")
        return None
    if settings.get("agent") != "codex" or not settings.get("model") or not settings.get("profile"):
        raise ValueError("Research tools require an explicit Codex model and profile")
    selected = {**settings, **({"source_lookup": True} if enabled else {})}
    if python_execution_image is not None:
        from .finite_python import execution_policy
        execution_policy(python_execution_image)
        selected["python_execution_image"] = python_execution_image
    return selected


def _author_settings(pipeline_cfg, *, source_lookup: bool = False,
                     python_execution_image: str | None = None) -> dict[str, Any]:
    """Use the configured research agent for author work, independent of review overrides."""
    settings = _source_lookup_settings({"agent": pipeline_cfg.agent.name, "model": pipeline_cfg.agent.model,
            "profile": pipeline_cfg.agent.profile, "reasoning_effort": pipeline_cfg.agent.reasoning_effort,
            "timeout_seconds": pipeline_cfg.review.timeout_seconds}, source_lookup, python_execution_image)
    assert settings is not None
    return settings


def _reviewer_settings(pipeline_cfg, *, source_lookup: bool = False,
                       python_execution_image: str | None = None) -> dict[str, Any] | None:
    review = pipeline_cfg.review
    if review.backend != "agent":
        if python_execution_image is not None:
            raise ValueError("Python tools require the native Codex review backend")
        return None
    return _source_lookup_settings({"agent": review.agent_name or pipeline_cfg.agent.name,
            "model": review.agent_model or pipeline_cfg.agent.model,
            "profile": review.agent_profile or pipeline_cfg.agent.profile,
            "reasoning_effort": review.agent_reasoning_effort or pipeline_cfg.agent.reasoning_effort,
            "timeout_seconds": review.timeout_seconds}, source_lookup, python_execution_image)


def _resolved_campaign_context(task: Path, pipeline_cfg) -> dict[str, Any]:
    """Use the final resolved execution policy, after all command overrides."""
    from dataclasses import fields
    from src.investigation.portfolio_controller import PortfolioConfig
    from .campaign_plan import build_host_context

    policy = {field.name: pipeline_cfg.investigation.get("portfolio." + field.name)
              for field in fields(PortfolioConfig)}
    return build_host_context(
        task, portfolio_policy=policy, workers=pipeline_cfg.investigation.workers,
        downstream_limits={"ablations": {"num_replications": pipeline_cfg.ablations.num_replications,
                       "max_evals_per_variant": pipeline_cfg.ablations.max_evals_per_variant},
         "confirmation": {"timeout_seconds": pipeline_cfg.final.timeout_seconds},
         "posthoc": {"max_selected_analyses": pipeline_cfg.posthoc.max_selected_analyses}},
    )


def _novelty_cycle(task: Path, cfg: ResearchConfig, output: Path, *, offline: bool,
                   reviewer_settings: dict[str, Any] | None, author_settings: dict[str, Any] | None = None,
                   research_basis: dict[str, Any] | None = None,
                   candidate_packet: dict[str, Any] | None = None, resume_calls: bool = False,
                   host_context: dict[str, Any] | None = None,
                   seed_agenda: dict[str, Any] | None = None) -> dict[str, Any]:
    from .research_agenda import load_agenda, normalize_agenda, agenda_from_basis
    from .novelty import content_hash, evaluate_novelty
    from .novelty_review import InvalidReviewResume, draft_research_agenda, review_novelty
    from .campaign_plan import PlanValidationError, compile_campaign_plan
    output.mkdir(parents=True, exist_ok=True)
    agenda = agenda_from_basis(research_basis) if research_basis is not None else load_agenda(task)
    permitted = cfg.harness.novelty_review and not offline
    if permitted and (cfg.harness.agent_source_lookup or cfg.harness.agent_python_image):
        author_settings = _source_lookup_settings(author_settings, cfg.harness.agent_source_lookup,
                                                 cfg.harness.agent_python_image or None)
        reviewer_settings = _source_lookup_settings(reviewer_settings, cfg.harness.agent_source_lookup,
                                                   cfg.harness.agent_python_image or None)
    errors = []
    resume_kwargs: dict[str, Any] = {"resume": True} if resume_calls else {}
    campaign_kwargs: dict[str, Any] = {"host_context": host_context} if host_context is not None else {}
    author_kwargs: dict[str, Any] = {**campaign_kwargs, **({"seed_agenda": seed_agenda} if seed_agenda is not None else {})}
    try:
        if permitted and (agenda is None or research_basis is not None or host_context is not None or seed_agenda is not None):
            agenda = draft_research_agenda(task, cfg.literature, output,
                                           research_basis=research_basis, reviewer_settings=author_settings,
                                           candidate_packet=candidate_packet, **resume_kwargs, **author_kwargs)
        agenda = agenda or normalize_agenda({"intent": "exploratory"})
        if host_context is not None:
            campaign_kwargs["compiled_plan"] = compile_campaign_plan(agenda, host_context)
        if permitted:
            report = review_novelty(task, agenda, cfg.literature, output,
                                    reviewer_count=cfg.harness.novelty_reviewers,
                                    reviewer_settings=reviewer_settings, candidate_packet=candidate_packet, **resume_kwargs, **campaign_kwargs)
        else:
            report = evaluate_novelty(task, agenda, candidate_packet=candidate_packet)
    except (InvalidReviewResume, PlanValidationError):
        raise
    except (ValueError, OSError, TypeError, KeyError) as exc:
        errors.append(str(exc))
        recovery_path = output / "agenda_reflection_recovery.json"
        recovery = None
        if recovery_path.is_file() and not recovery_path.is_symlink():
            try:
                recovery = read_json(recovery_path)
                if (recovery.get("schema_version") != 1
                        or recovery.get("status") != "verified_last_valid_author_agenda"
                        or recovery.get("frozen_evidence") is None):
                    raise ValueError("Author agenda recovery record is invalid")
                recovered_agenda = normalize_agenda(recovery.get("agenda"))
                if content_hash(recovered_agenda) != recovery.get("agenda_sha256"):
                    raise ValueError("Recovered author agenda hash changed")
                agenda = recovered_agenda
            except (ValueError, OSError, TypeError, KeyError, AttributeError) as recovery_error:
                errors.append("Author agenda recovery failed: " + str(recovery_error))
                recovery = None
        agenda = agenda or normalize_agenda({"intent": "exploratory"})
        report = evaluate_novelty(task, agenda, candidate_packet=candidate_packet)
        if recovery is not None:
            report["agenda_recovery"] = {
                "status": recovery["status"],
                "source_directory": recovery.get("source_directory"),
                "source_receipt_sha256": recovery.get("source_receipt_sha256"),
                "source_assessment_sha256": recovery.get("source_assessment_sha256"),
                "agenda_sha256": recovery["agenda_sha256"],
                "scientific_status": "needs_evidence",
            }
    if errors:
        report.setdefault("gaps", []).append({"code": "review_execution_failed", "detail": "; ".join(errors)})
        report["status"] = "needs_evidence"
    _dump(output / "research_agenda.json", agenda)
    if candidate_packet is not None:
        _dump(output / "candidate_packet.json", candidate_packet)
    _dump(output / "novelty_report.json", report)
    return report


def _selected_novelty(workspace: Path, cfg: ResearchConfig, pipeline_cfg, *, offline: bool) -> None:
    from .harness_evidence import build_candidate_packet
    from .novelty import content_hash
    from .novelty_review import load_completed_novelty_report
    packet = build_candidate_packet(workspace)
    selection_path = workspace / "novelty/latest.json"
    state_path, preparation_path = workspace / "harness_state.json", workspace / "research_preparation.json"
    state_version = read_json(state_path).get("selected_review_enrichment_version") if state_path.exists() else None
    prepared = read_json(preparation_path).get("initial_harness_state", {}) if preparation_path.exists() else {}
    prepared_version = prepared.get("selected_review_enrichment_version") if isinstance(prepared, dict) else None
    if state_version is not None or prepared_version is not None:
        if type(state_version) is not int or state_version != 1 or type(prepared_version) is not int or prepared_version != 1:
            raise ValueError("Selected review enrichment marker differs from fresh preparation")
        from .review_enrichment import ensure_selected_novelty, resolve_selected_evidence_task
        pointer = ensure_selected_novelty(workspace, cfg, packet=packet,
            author_settings=_author_settings(pipeline_cfg, source_lookup=cfg.harness.agent_source_lookup, python_execution_image=cfg.harness.agent_python_image or None), reviewer_settings=_reviewer_settings(pipeline_cfg, source_lookup=cfg.harness.agent_source_lookup, python_execution_image=cfg.harness.agent_python_image or None), offline=offline)
        resolve_selected_evidence_task(workspace, pointer, candidate_packet=packet)
        _dump(selection_path, pointer)
        return
    if selection_path.is_file() and not selection_path.is_symlink():
        previous = read_json(selection_path)
        cycle_value = previous.get("cycle", "")
        cycle_path = workspace / cycle_value if isinstance(cycle_value, str) else workspace
        if (previous.get("candidate_packet_sha256") == content_hash(packet)
                and previous.get("candidate_source_sha256") == packet["candidate_source_sha256"]
                and previous.get("candidate_program") == packet["selection"]["program_relative_path"]
                and isinstance(cycle_value, str) and cycle_value
                and not Path(cycle_value).is_absolute()
                and cycle_value == Path(cycle_value).as_posix() and ".." not in Path(cycle_value).parts
                and not cycle_path.is_symlink() and not (workspace / "novelty").is_symlink()
                and cycle_path.resolve().is_relative_to((workspace / "novelty").resolve())):
            try:
                current = load_completed_novelty_report(workspace / "input_task", cycle_path, cfg.literature,
                    reviewer_count=cfg.harness.novelty_reviewers, reviewer_settings=_reviewer_settings(pipeline_cfg, source_lookup=cfg.harness.agent_source_lookup, python_execution_image=cfg.harness.agent_python_image or None),
                    candidate_packet=packet)
                if current is not None:
                    return
            except (OSError, ValueError, TypeError, KeyError):
                pass
    cycle = workspace / "novelty" / ("selected_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2))
    report = _novelty_cycle(workspace / "input_task", cfg, cycle, offline=offline,
                           reviewer_settings=_reviewer_settings(pipeline_cfg, source_lookup=cfg.harness.agent_source_lookup, python_execution_image=cfg.harness.agent_python_image or None), author_settings=_author_settings(pipeline_cfg, source_lookup=cfg.harness.agent_source_lookup, python_execution_image=cfg.harness.agent_python_image or None),
                           research_basis=packet["research_basis"],
                           candidate_packet=packet)
    _dump(selection_path, {"schema_version": 1, "status": report["status"],
                           "cycle": str(cycle.relative_to(workspace)),
                           "candidate_source_sha256": packet["candidate_source_sha256"],
                           "candidate_packet_sha256": content_hash(packet),
                           "candidate_program": packet["selection"]["program_relative_path"], "assessed_at": now()})


def _resume_plan(workspace: Path, prior: dict[str, Any], phases: set[str]) -> set[str]:
    remaining = set(phases)
    for phase in phases:
        stage = prior.get("stages", {}).get(phase, {})
        if stage.get("status") == "completed":
            if not verify_stage_records(workspace, stage):
                raise ValueError(f"Saved {phase} evidence changed or is incomplete; inspect artifacts before resuming")
            remaining.discard(phase)
    final = workspace / "run/final_evaluation"
    if "investigation" in remaining and any((final / name).exists() or (final / name).is_symlink()
                                            for name in ("invocation.json", "comparison_set.json")):
        raise ValueError("Final confirmation has already been invoked or its method set has been frozen. Do not tune this study on confirmation results; create a new study with a new frozen confirmation protocol")
    return remaining


def _resume_paper_overrides(workspace: Path, prior: dict[str, Any], remaining: set[str]) -> list[str]:
    """Bind downstream stages to the unchanged, host-recorded manuscript."""
    if "paper" in remaining or not remaining.intersection({"review", "revision"}):
        return []
    if not verify_stage_records(workspace, prior.get("stages", {}).get("paper", {})):
        raise ValueError("Saved paper evidence changed or is incomplete; inspect artifacts before resuming")
    return ["paths.paper_dir=" + json.dumps(str(workspace / "run/paper"))]


def _startup_events(path: Path) -> list[dict[str, Any]]:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1 or metadata.st_size > 8 * 1024 * 1024):
        raise ValueError(f"Unsafe startup event journal: {path}")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not events or not all(isinstance(event, dict) and isinstance(event.get("fields", {}), dict) for event in events):
        raise ValueError(f"Invalid startup event journal: {path}")
    return events


def _startup_retry_plan(workspace: Path, prior: dict[str, Any], *, cycle_budget: int,
                        _runtime_lock_held: bool = False) -> dict[str, Any] | None:
    """Recognize sealed initialization failures before an investigation checkpoint."""
    from src.investigation.task_alignment import task_tree_fingerprint
    from src.pipeline.resume_config import read_run_idea_cycle_budget
    from src.pipeline.safe_fs import open_directory
    from src.pipeline.task_snapshot import task_snapshot_is_complete

    run, investigation = workspace / "run", workspace / "run/investigation"
    if not investigation.exists() and not investigation.is_symlink():
        return None
    # Any checkpoint, even an incomplete one, belongs to ordinary resume. Never
    # turn an ambiguous coding or evaluation state into a fresh investigation.
    if any((investigation / name).exists() or (investigation / name).is_symlink()
           for name in ("outer_state.json", "best_program_manifest.json")):
        return None
    reason = "Initial runtime retry is unsafe; inspect the retained startup evidence"
    stage = prior.get("stages", {}).get("investigation", {})
    lifecycle = prior.get("events", [])
    if not isinstance(stage, dict) or not isinstance(lifecycle, list) or not all(isinstance(event, dict) for event in lifecycle):
        raise ValueError(reason + ": invalid host failure record")
    phase_events = [event for event in lifecycle if event.get("phase_name") == "investigation"]
    if (prior.get("status") != "failed" or prior.get("pipeline_exit_code") != 2
            or stage.get("status") != "failed" or stage.get("return_code") != 2
            or len(phase_events) < 2 or phase_events[-2].get("event") != "phase_start"
            or phase_events[-1].get("event") != "phase_end" or phase_events[-1].get("status") != "failed"
            or phase_events[-1].get("return_code") != 2
            or not lifecycle or lifecycle[-1].get("event") != "pipeline_end"
            or lifecycle[-1].get("status") != "failed" or lifecycle[-1].get("return_code") != 2):
        raise ValueError(reason + ": no completed host failure record")
    for phase in PIPELINE_PHASES:
        if phase in {"task_drafting", "investigation"}:
            continue
        path = run / phase
        if (prior.get("stages", {}).get(phase, {}).get("status", "pending") != "pending"
                or path.exists() or path.is_symlink()):
            raise ValueError(reason + ": later research or final-confirmation activity exists")
    if (workspace / "novelty/latest.json").exists() or (workspace / "novelty/latest.json").is_symlink():
        raise ValueError(reason + ": selected-candidate evidence exists")
    with open_directory(run, role="startup run envelope") as run_fd, open_directory(investigation, role="startup investigation") as inv_fd:
        identities = {}
        for name, descriptor in (("run", run_fd), ("investigation", inv_fd)):
            metadata = os.fstat(descriptor)
            if metadata.st_uid != os.getuid():
                raise ValueError(reason + ": startup directories are not owned by this user")
            identities[name] = [metadata.st_dev, metadata.st_ino]
        allowed = {"task", "investigation", "pipeline_events.jsonl", ".task.runtime-build.lock",
                   ".agentic_resume_policy.json", ".investigation_completion.json"}
        if prior.get("stages", {}).get("task_drafting", {}).get("status") == "completed":
            allowed.add("task_drafting")
        for name in os.listdir(run_fd):
            ready_marker = name.startswith(".task.agentic_task_ready.") and len(name.rsplit(".", 1)[-1]) == 64
            if name not in allowed and not ready_marker:
                raise ValueError(reason + f": unexpected run artifact {name}")
            metadata = os.stat(name, dir_fd=run_fd, follow_symlinks=False)
            expected_directory = name in {"task", "investigation", "task_drafting"}
            valid_type = stat.S_ISDIR(metadata.st_mode) if expected_directory else stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            if metadata.st_uid != os.getuid() or not valid_type:
                raise ValueError(reason + f": unsafe run artifact {name}")
        if set(os.listdir(inv_fd)) - {"events.jsonl", ".task-seal-empty"}:
            raise ValueError(reason + ": investigation contains more than startup outputs")
        seal = investigation / ".task-seal-empty"
        if seal.exists() or seal.is_symlink():
            with open_directory(seal, role="empty task seal") as seal_fd:
                if os.fstat(seal_fd).st_uid != os.getuid() or os.listdir(seal_fd):
                    raise ValueError(reason + ": task seal is not empty and owned")
    events = _startup_events(investigation / "events.jsonl")
    failures = [event for event in events if event.get("event") == "task_runtime_failed"]
    if len(failures) != 1:
        raise ValueError(reason + ": missing or ambiguous runtime failure")
    failure = failures[0]
    session = failure.get("session_id")
    fields = failure.get("fields", {})
    baseline_retry = (fields.get("failure_stage") == "baseline_validation"
                      and fields.get("failure_category") == "evaluation_failed")
    build_steps = ("runtime_capacity", "runtime_create", "tooling_build", "tooling_install", "dependency_install")
    allowed_steps = {*build_steps, "baseline_evaluation"} if baseline_retry else {"runtime_capacity"}
    active_step = None
    completed_steps: list[str] = []
    failed_steps: list[str] = []
    if any(event.get("event") != "container_session_stopped" for event in events[events.index(failure) + 1:]):
        raise ValueError(reason + ": activity continued after the runtime failure")
    for event in events:
        event_name = event.get("event")
        progress = event.get("fields", {})
        if (event.get("source") != "wrapper" or not session or event.get("session_id") != session
                or event.get("worker_index") != 0
                or event_name not in {"container_session_stopped", "preparation_step", "heartbeat", "task_runtime_failed"}):
            raise ValueError(reason + ": journal is not limited to runtime startup")
        if event_name in {"preparation_step", "heartbeat"}:
            step, status = progress.get("step"), progress.get("state")
            if not isinstance(step, str) or step not in allowed_steps:
                raise ValueError(reason + ": unknown runtime preparation step")
            if event_name == "heartbeat":
                if step != active_step or status != "started":
                    raise ValueError(reason + ": heartbeat is not bound to an active preparation step")
            elif status == "started":
                if active_step is not None or step in completed_steps or failed_steps:
                    raise ValueError(reason + ": preparation steps overlap or repeat")
                active_step = step
            elif status in {"completed", "failed"} and active_step == step:
                (completed_steps if status == "completed" else failed_steps).append(step)
                active_step = None
            else:
                raise ValueError(reason + ": invalid runtime preparation step sequence")
    if (failure.get("phase") != "task" or fields.get("task_dir") != str(run / "task")
            or fields.get("result_kind") != "infrastructure_failure"
            or not (baseline_retry or (fields.get("failure_stage") == "runtime_preparation"
                                       and fields.get("failure_category") == "runtime_bootstrap"))):
        raise ValueError(reason + ": failure is not a verified runtime bootstrap error")
    if baseline_retry:
        # A failed initialization baseline is a development evaluation. Its
        # temporary result directory may already have been cleaned, so absence
        # of output must never be described as proof of zero execution.
        if active_step is not None or completed_steps != list(build_steps) or failed_steps != ["baseline_evaluation"]:
            raise ValueError(reason + ": baseline failure lacks a complete initialization step sequence")
        for name in (".venv", ".agentic_runtime.json"):
            path = run / "task" / name
            if path.exists() or path.is_symlink():
                raise ValueError(reason + ": a published task runtime must use its existing recovery path")
        for name in ("metrics.json", "correct.json", "evaluation_manifest.json", "evaluation_evidence.json", "rubric_status.json"):
            path = run / "task/program_0" / name
            if path.exists() or path.is_symlink():
                raise ValueError(reason + ": baseline result evidence must be inspected and retained in place")
    try:
        start = datetime.fromisoformat(phase_events[-2]["time"])
        end = datetime.fromisoformat(phase_events[-1]["time"])
        failed = datetime.fromisoformat(failure["timestamp"].replace("Z", "+00:00"))
        if not start <= failed <= end:
            raise ValueError("runtime failure falls outside the failed host phase")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(reason + ": runtime and host failure times do not match") from exc
    pipeline_events = _startup_events(run / "pipeline_events.jsonl")
    terminals = [index for index, event in enumerate(pipeline_events) if event.get("event") == "pipeline_end"]
    if not terminals:
        raise ValueError(reason + ": pipeline has no terminal failure")
    terminal_index = terminals[-1]
    terminal = pipeline_events[terminal_index]
    if (terminal.get("source") != "pipeline" or terminal.get("event") != "pipeline_end"
            or terminal.get("fields", {}).get("status") != "failed" or terminal.get("fields", {}).get("return_code") != 2):
        raise ValueError(reason + ": pipeline has not recorded a stopped failure")
    if any(event.get("source") != "pipeline" or event.get("event") != "run_checkpoint"
           or event.get("session_id") != terminal.get("session_id")
           for event in pipeline_events[terminal_index + 1:]):
        raise ValueError(reason + ": pipeline activity continued after failure")
    if read_json(run / ".investigation_completion.json") != {"schema_version": 1, "status": "running"}:
        raise ValueError(reason + ": investigation completion is not an unused startup marker")
    if read_run_idea_cycle_budget(run) != cycle_budget:
        raise ValueError(reason + ": retain the recorded idea-cycle budget for this startup retry")
    if not task_snapshot_is_complete(run / "task") or fields.get("task_fingerprint") != task_tree_fingerprint(run / "task"):
        raise ValueError(reason + ": run task changed after the failure")
    # This is the existing runtime builder's lock. Do not create a replacement
    # inode: a live builder must retain its authority over the run task.
    if not _runtime_lock_held:
        os.close(_startup_runtime_lock(workspace))
    return {"schema_version": 1, "kind": "initial_runtime_failure", "identities": identities,
            "failure_classification": "baseline_validation_infrastructure_failure" if baseline_retry else "runtime_bootstrap_failure",
            "scientific_result_status": "not_retained_or_verified" if baseline_retry else "not_assessed",
            "failure": failure, "host_phase": phase_events[-2:], "cycle_budget": cycle_budget,
            "artifacts": {str(path.relative_to(workspace)): sha256(path) for path in (
                investigation / "events.jsonl", run / "pipeline_events.jsonl",
                run / ".investigation_completion.json", run / ".agentic_resume_policy.json")}}


def _startup_runtime_lock(workspace: Path) -> int:
    lock_fd = os.open(workspace / "run/.task.runtime-build.lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(lock_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
            raise ValueError("Initial runtime retry has an unsafe runtime lock")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Initial runtime initialization is still active") from exc
        return lock_fd
    except BaseException:
        os.close(lock_fd)
        raise


def _archive_startup_failure(workspace: Path, prior: dict[str, Any], plan: dict[str, Any]) -> None:
    descriptor = _startup_runtime_lock(workspace)
    try:
        _archive_startup_failure_locked(workspace, prior, plan)
    finally:
        os.close(descriptor)


def _archive_startup_failure_locked(workspace: Path, prior: dict[str, Any], plan: dict[str, Any]) -> None:
    from src.pipeline.safe_fs import ensure_directory, open_directory, rename_directory_between_no_replace
    if _startup_retry_plan(workspace, prior, cycle_budget=plan["cycle_budget"], _runtime_lock_held=True) != plan:
        raise ValueError("Startup evidence changed after preflight; no retry was started")
    archive = workspace / "startup_recovery" / ("initial_runtime_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4))
    with open_directory(workspace / "run", role="startup run envelope") as run_fd, ensure_directory(archive, role="startup recovery archive") as archive_fd:
        _dump(archive / "recovery.json", {**plan, "archived_at": now(), "source": "run/investigation"})
        # Record intent first. A crash after the atomic rename remains resumable
        # as fresh initialization, and its old logs remain discoverable.
        entry = {"kind": plan["kind"], "archive": str(archive.relative_to(workspace)), "status": "authorized", "time": now()}
        prior.setdefault("startup_recoveries", []).append(entry)
        _dump(workspace / "harness_state.json", prior)
        current = os.stat("investigation", dir_fd=run_fd, follow_symlinks=False)
        if [current.st_dev, current.st_ino] != plan["identities"]["investigation"]:
            raise ValueError("Startup directory identity changed before archival")
        rename_directory_between_no_replace(run_fd, "investigation", archive_fd, "investigation", role="failed startup archive")
        os.fsync(run_fd)
        os.fsync(archive_fd)
        entry["status"] = "archived"
        _dump(workspace / "harness_state.json", prior)


def run_harness(args: Namespace, overrides: list[str]) -> int:
    """Serialize resume operations without adding mutable lock-file artifacts."""
    from src.pipeline.resume_config import ResumeConfigError
    from src.pipeline.safe_fs import UnsafeGluePathError, open_directory
    try:
        if args.resume:
            with open_directory(args.resume.expanduser(), role="harness resume workspace") as descriptor:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ValueError("Another harness resume is active for this workspace") from exc
                return _run_harness(args, overrides)
        return _run_harness(args, overrides)
    except (UnsafeGluePathError, ResumeConfigError) as exc:
        raise ValueError(f"Harness workspace safety check failed: {exc}") from exc


def _run_harness(args: Namespace, overrides: list[str]) -> int:
    from .cli import pipeline_args
    from src.pipeline.cli import load_pipeline_config, config_error_message, main as pipeline_main
    managed_paths = {"paths.paper_dir", "paths.ablations_dir", "paths.posthoc_dir", "paths.final_evaluation_dir"}
    for item in overrides:
        if item.split("=", 1)[0] in managed_paths:
            raise ValueError("Harness stage paths are managed inside its workspace; do not override " + item.split("=", 1)[0])
    prior = None
    manifest = None
    preparation = None
    if args.resume:
        if args.task or args.workspace or args.offline:
            raise ValueError("--resume uses frozen task/workspace/literature; omit task/--workspace/--offline")
        workspace = args.resume.expanduser().resolve()
        if (workspace / "research_manifest.json").exists() or (workspace / "research_manifest.json").is_symlink():
            cfg, manifest = resume(workspace)
            saved = execution_overrides(workspace, manifest)
        else:
            cfg, preparation = load_preparation(workspace)
            saved = list(preparation["pipeline_overrides"])
        state_path = workspace / "harness_state.json"
        prior = read_json(state_path) if state_path.exists() or state_path.is_symlink() else (preparation or {}).get("initial_harness_state")
        if not isinstance(prior, dict) or prior.get("schema_version") != 1:
            raise ValueError("Unsupported or missing harness state")
        if manifest is not None and prior.get("input_task_sha256") is None:
            # A crash may follow manifest publication but precede the next
            # lifecycle write. Accept only the matching sealed transaction.
            _, completed_preparation = load_preparation(workspace)
            if completed_preparation["state"] not in {"finalizing", "prepared"} or completed_preparation["pipeline_overrides"] != manifest["pipeline_overrides"]:
                raise ValueError("Harness preparation does not match its published manifest")
            prior["input_task_sha256"] = manifest["input_task_sha256"]
        old = {item.split("=", 1)[0]: item for item in saved}
        for item in overrides:
            key = item.split("=", 1)[0]
            if (preparation is not None or key not in RESUME_MUTABLE) and old.get(key) != item:
                raise ValueError(f"Cannot change frozen harness setting {key} on resume; create a new workspace")
        overrides = _merge(saved, overrides)
        target = args.quality_target or prior["quality_target"]
        task = Path(preparation["source_task"]) if preparation and preparation["state"] == "copying" else workspace / "input_task"
        offline = bool(preparation["offline"] if preparation else prior.get("offline", False))
    else:
        if args.task is None:
            raise ValueError("Provide a task directory or --resume WORKSPACE")
        task = args.task.expanduser().resolve()
        cfg = load_config(task)
        target = args.quality_target or cfg.harness.quality_target
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
        workspace = (args.workspace or ROOT / "results/harness" / task.name / run_id).expanduser().absolute()
        if workspace.exists() or workspace.is_symlink():
            raise ValueError("Workspace exists; use --resume or a new workspace")
        if workspace.resolve().is_relative_to(task):
            raise ValueError("The research workspace must be outside its source task directory")
        offline = args.offline
    overrides = _merge(["preset=research_harness"], overrides)
    try:
        pipeline_cfg = load_pipeline_config(pipeline_args(workspace, cfg, overrides))
    except Exception as exc:
        raise ValueError(config_error_message(exc)) from exc
    if any(getattr(pipeline_cfg.paths, name) is not None for name in ("paper_dir", "ablations_dir", "posthoc_dir", "final_evaluation_dir")):
        raise ValueError("Harness presets must keep all stage paths inside the managed workspace")
    phases = parse_pipeline_phases(pipeline_cfg.execution.phases)
    if not any(item.startswith("execution.phases=") for item in overrides) and not missing_task_scaffold_files(task):
        if target == "development" or (task / "task_contract.json").is_file():
            phases.discard("task_drafting")
    if target == "high_impact":
        if REQUIRED_PHASES - phases:
            raise ValueError("High-impact harness requires all research phases; use --quality-target development for a partial pilot")
        if not (pipeline_cfg.investigation.adaptive_enabled or pipeline_cfg.investigation.portfolio_enabled):
            raise ValueError("High-impact harness requires portfolio.enabled=true or adaptive_investigation.enabled=true")
    requested = [phase for phase in PIPELINE_PHASES if phase in phases]
    remaining = _resume_plan(workspace, prior, phases) if prior else phases
    if prior and target != prior["quality_target"]:
        raise ValueError("Quality target is frozen; create a new workspace to change it")
    effective = _merge(overrides, ["execution.phases=" + ",".join(phase for phase in PIPELINE_PHASES if phase in remaining)])
    if prior:
        effective = _merge(effective, _resume_paper_overrides(workspace, prior, remaining))
    if prior and "posthoc" in remaining:
        # Retain failed analysis output and retry in a fresh directory.
        effective = _merge(effective, ["execution.restart_posthoc=true"])
    startup_retry = None
    if prior and "investigation" in remaining and (workspace / "run/investigation").is_dir():
        startup_retry = _startup_retry_plan(workspace, prior, cycle_budget=pipeline_cfg.investigation.max_idea_cycles)
        effective = _merge(effective, ["investigation.execution.resume=" + ("false" if startup_retry else "true")])
    command = pipeline_args(workspace, cfg, effective) if remaining else []
    pipeline_cfg = load_pipeline_config(command) if command else pipeline_cfg
    if args.dry_run:
        if prior and "paper" not in remaining and remaining.intersection({"review", "revision"}):
            execution_task = workspace / "run/task" if (workspace / "run/task").is_dir() else task
            preview_cfg = replace(pipeline_cfg, task=replace(pipeline_cfg.task, dir=str(execution_task)))
            errors = validate_pipeline_config(preview_cfg)
            if errors:
                raise ValueError("Resumed pipeline configuration is invalid: " + "; ".join(errors))
        print(json.dumps({"dry_run": True, "mode": "research_harness", "workspace": str(workspace),
                          "quality_target": target, "requested_phases": requested,
                          "remaining_phases": [p for p in PIPELINE_PHASES if p in remaining],
                          "pipeline_argv": command, "novelty_review": cfg.harness.novelty_review and not offline,
                          "initial_runtime_retry": startup_retry is not None,
                          "human_submission_approval_required": True,
                          "note": "No files, API requests, agents, or training were created."}, ensure_ascii=False, indent=2))
        return 0
    # Even resume checks every remaining requested phase before any provider call.
    if remaining:
        ensure_execution_ready(pipeline_cfg, workspace / "run/task" if prior and (workspace / "run/task").is_dir() else task)
    if startup_retry is not None:
        assert prior is not None
        _archive_startup_failure(workspace, prior, startup_retry)
    state = HarnessState(workspace, phases=requested, quality_target=target,
                         input_sha256=manifest["input_task_sha256"] if manifest else None, prior=prior)
    state.payload["offline"] = offline
    if not args.resume and not offline and cfg.harness.novelty_review:
        state.payload["selected_review_enrichment_version"] = 1
    if not args.resume and not offline and cfg.harness.novelty_review and pipeline_cfg.investigation.portfolio_enabled:
        # This opt-in travels through the normal preparation transaction. Old
        # interrupted workspaces have no marker and keep their original calls.
        if ("task_drafting" not in remaining and not missing_task_scaffold_files(task)
                and (task / "task_contract.json").is_file()):
            state.payload["campaign_plan_version"] = 1
        else:
            state.payload["postdraft_campaign_plan_version"] = 1
            state.payload["campaign_plan"] = {
                "status": "awaiting_draft", "reason": "The task scaffold and fixed contract must exist before plan compilation.",
                "execution_mode": "postdraft_plan",
            }
    state.payload["preflight"] = {"status": "passed", "checked_at": now(), "remaining_phases": sorted(remaining)}
    exit_code = 1
    workspace_lock_fd = None
    try:
        if manifest is None:
            def after_literature(staged: Path, frozen_cfg: ResearchConfig) -> dict[str, str]:
                from .upstream.disclosure import integration_marker, MARKER_PATH
                from .campaign_plan import HOST_CONTEXT_PATH, CAMPAIGN_PLAN_PATH, PlanValidationError
                from .research_agenda import load_seed_agenda
                state.payload["literature"] = {"status": "prepared", "checked_at": now()}
                state.payload["preparation"] = {"status": "initial_review", "journal": "research_preparation.json"}
                state.save()
                host_context = (_resolved_campaign_context(staged, pipeline_cfg)
                                if state.payload.get("campaign_plan_version") == 1 else None)
                campaign_kwargs = ({"host_context": host_context, "seed_agenda": load_seed_agenda(staged)}
                                   if host_context is not None else {})
                try:
                    report = _novelty_cycle(staged, frozen_cfg, workspace / "novelty/initial", offline=offline,
                                            reviewer_settings=_reviewer_settings(pipeline_cfg, source_lookup=frozen_cfg.harness.agent_source_lookup, python_execution_image=frozen_cfg.harness.agent_python_image or None),
                                            author_settings=_author_settings(pipeline_cfg, source_lookup=frozen_cfg.harness.agent_source_lookup, python_execution_image=frozen_cfg.harness.agent_python_image or None), resume_calls=preparation is not None,
                                            **campaign_kwargs)
                except PlanValidationError:
                    state.payload["preparation"] = {
                        "status": "plan_validation_failed", "journal": "research_preparation.json",
                        "checkpoint": "novelty/initial/plan_validation.json",
                    }
                    state.save()
                    raise
                state.payload["novelty_initial"] = {"status": report["status"], "report": "novelty/initial/novelty_report.json"}
                state.save()
                if frozen_cfg.harness.novelty_review and not offline and len(report.get("model_judgments", [])) < frozen_cfg.harness.novelty_reviewers:
                    raise ValueError("Initial novelty review is incomplete; resume this workspace to continue its saved calls")
                agenda = read_json(workspace / "novelty/initial/research_agenda.json")
                outputs = {
                    "research_context/harness_novelty.json": report,
                    "research_agenda.json": agenda, str(MARKER_PATH): integration_marker(),
                }
                if host_context is not None:
                    from .campaign_plan import compile_campaign_plan
                    outputs[str(HOST_CONTEXT_PATH)] = host_context
                    outputs[str(CAMPAIGN_PLAN_PATH)] = compile_campaign_plan(agenda, host_context)
                return {name: json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                        for name, value in outputs.items()}
            manifest = prepare(task, workspace, offline=offline, overrides=overrides, after_literature=after_literature,
                               resume_preparation=preparation is not None, initial_state=state.payload)
            state.payload["preparation"] = {"status": "prepared", "journal": "research_preparation.json"}
            state.payload["input_task_sha256"] = manifest["input_task_sha256"]
        if not args.resume:
            # The fresh workspace was created exclusively by prepare(). Keep
            # resume out through pipeline execution and the final audit writes.
            workspace_lock_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            fcntl.flock(workspace_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.save()
        record_execution(workspace, manifest, overrides)
        print(f"Research harness workspace: {workspace}", flush=True)
        def refresh_selected_novelty() -> None:
            try:
                _selected_novelty(workspace, cfg, pipeline_cfg, offline=offline)
                state.payload["novelty_selected"] = read_json(workspace / "novelty/latest.json")
                state.payload.pop("novelty_review_error", None)
                state.save()
            except (OSError, ValueError, TypeError, KeyError, FinalEvaluationError) as exc:
                state.payload["novelty_review_error"] = str(exc)
                state.save()
        def observe(event: str, **fields: Any) -> None:
            state.observe(event, **fields)
            if event == "phase_end" and fields.get("phase_name") == "investigation" and fields.get("status") == "completed":
                refresh_selected_novelty()
        # A failed novelty provider must be retryable without spending another
        # training budget. A supported cache is replay-verified inside the helper.
        if prior and state.payload.get("stages", {}).get("investigation", {}).get("status") == "completed":
            refresh_selected_novelty()
        finalizer_kwargs: dict[str, Any] = {}
        if state.payload.get("postdraft_campaign_plan_version") == 1:
            from .postdraft_plan import ensure_postdraft_campaign_plan, verify_postdraft_campaign, record_postdraft_success

            def finalize_task(action, resolved, paths):
                if action == "draft_complete":
                    return record_postdraft_success(workspace, paths, cfg, resolved)
                if action == "verify":
                    verify_postdraft_campaign(workspace, require_frozen=True, runtime_task=paths.task_dir)
                    return True
                try:
                    return ensure_postdraft_campaign_plan(workspace, paths, cfg, resolved, recovery=action == "recover")
                finally:
                    try:
                        state.payload["campaign_plan"] = verify_postdraft_campaign(workspace) or {}
                    except (OSError, ValueError) as exc:
                        state.payload["campaign_plan"] = {"status": "invalid", "error": str(exc)}
                    state.save()

            finalizer_kwargs["task_finalizer"] = finalize_task
        exit_code = pipeline_main(command, event_callback=observe, **finalizer_kwargs) if command else 0
        state.finish(exit_code)
    except KeyboardInterrupt:
        exit_code = 130
        if workspace.is_dir():
            state.finish(exit_code, error="Interrupted by user")
    except Exception as exc:
        exit_code = 1
        if workspace.is_dir():
            state.finish(exit_code, error=f"{type(exc).__name__}: {exc}")
        print(f"Research harness failed: {exc}", file=sys.stderr)
    finally:
        if manifest is not None:
            try:
                write_report(workspace, exit_code=exit_code)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state.payload["research_report_error"] = str(exc)
        if workspace.is_dir():
            try:
                readiness = write_readiness(workspace, cfg, state.payload)
                state.payload["readiness"] = {"status": readiness["status"], "submission_ready": False,
                                             "report": "paper_readiness.json"}
                state.save()
                print(f"Evidence readiness: {readiness['status']}; inspect {workspace / 'paper_readiness.md'}", flush=True)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                state.payload["readiness_error"] = str(exc)
                state.save()
                print(f"Readiness audit could not complete: {exc}", file=sys.stderr)
                if exit_code == 0:
                    exit_code = 1
                    state.finish(exit_code, error=f"Readiness audit failed: {exc}")
        if workspace_lock_fd is not None:
            os.close(workspace_lock_fd)
    return exit_code
