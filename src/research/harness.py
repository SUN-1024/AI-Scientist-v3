"""One reusable research harness over the V3 pipeline, not a second executor."""
from __future__ import annotations

import json
import secrets
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.investigation.files import missing_task_scaffold_files
from src.final_evaluation.runner import FinalEvaluationError
from src.orchestrator.state import write_private_text
from src.pipeline.config import PIPELINE_PHASES, parse_pipeline_phases
from .config import ResearchConfig, load_config
from .harness_state import HarnessState, read_json, verify_stage_records, now
from .paper_readiness import write_readiness
from .preflight import ensure_execution_ready
from .workspace import prepare, resume, execution_overrides, record_execution, write_report

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PHASES = set(PIPELINE_PHASES) - {"task_drafting"}
RESUME_MUTABLE = {
    "execution.phases", "investigation.execution.max_idea_cycles", "investigation.execution.run_timeout_seconds",
    "investigation.execution.idea_timeout_seconds", "review.timeout_seconds", "logging.verbose",
    "logging.heartbeat_seconds", "execution.fail_fast",
}


def _merge(*groups: list[str]) -> list[str]:
    merged = {}
    for group in groups:
        for item in group:
            merged[item.split("=", 1)[0]] = item
    return list(merged.values())


def _dump(path: Path, payload: dict[str, Any]) -> None:
    write_private_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _author_settings(pipeline_cfg) -> dict[str, Any]:
    """Use the configured research agent for author work, independent of review overrides."""
    return {"agent": pipeline_cfg.agent.name, "model": pipeline_cfg.agent.model,
            "profile": pipeline_cfg.agent.profile, "reasoning_effort": pipeline_cfg.agent.reasoning_effort,
            "timeout_seconds": pipeline_cfg.review.timeout_seconds}


def _reviewer_settings(pipeline_cfg) -> dict[str, Any] | None:
    review = pipeline_cfg.review
    if review.backend != "agent":
        return None
    return {"agent": review.agent_name or pipeline_cfg.agent.name,
            "model": review.agent_model or pipeline_cfg.agent.model,
            "profile": review.agent_profile or pipeline_cfg.agent.profile,
            "reasoning_effort": review.agent_reasoning_effort or pipeline_cfg.agent.reasoning_effort,
            "timeout_seconds": review.timeout_seconds}


def _novelty_cycle(task: Path, cfg: ResearchConfig, output: Path, *, offline: bool,
                   reviewer_settings: dict[str, Any] | None, author_settings: dict[str, Any] | None = None,
                   research_basis: dict[str, Any] | None = None,
                   candidate_packet: dict[str, Any] | None = None) -> dict[str, Any]:
    from .research_agenda import load_agenda, normalize_agenda, agenda_from_basis
    from .novelty import evaluate_novelty
    from .novelty_review import draft_research_agenda, review_novelty
    output.mkdir(parents=True, exist_ok=True)
    agenda = agenda_from_basis(research_basis) if research_basis is not None else load_agenda(task)
    permitted = cfg.harness.novelty_review and not offline
    errors = []
    try:
        if permitted and (agenda is None or research_basis is not None):
            agenda = draft_research_agenda(task, cfg.literature, output,
                                           research_basis=research_basis, reviewer_settings=author_settings,
                                           candidate_packet=candidate_packet)
        agenda = agenda or normalize_agenda({"intent": "exploratory"})
        if permitted:
            report = review_novelty(task, agenda, cfg.literature, output,
                                    reviewer_count=cfg.harness.novelty_reviewers,
                                    reviewer_settings=reviewer_settings, candidate_packet=candidate_packet)
        else:
            report = evaluate_novelty(task, agenda, candidate_packet=candidate_packet)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        errors.append(str(exc))
        agenda = agenda or normalize_agenda({"intent": "exploratory"})
        report = evaluate_novelty(task, agenda, candidate_packet=candidate_packet)
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
    from .novelty_review import load_frozen_novelty_report
    packet = build_candidate_packet(workspace)
    selection_path = workspace / "novelty/latest.json"
    if selection_path.is_file():
        previous = read_json(selection_path)
        cycle_value = previous.get("cycle", "")
        cycle_path = workspace / cycle_value
        if (previous.get("candidate_packet_sha256") == content_hash(packet) and cycle_value
                and not Path(cycle_value).is_absolute()
                and cycle_path.resolve().is_relative_to((workspace / "novelty").resolve())):
            try:
                current = load_frozen_novelty_report(workspace / "input_task", cycle_path, candidate_packet=packet)
                if current.get("status") == "supported":
                    return
            except (OSError, ValueError, TypeError, KeyError):
                pass
    cycle = workspace / "novelty" / ("selected_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2))
    report = _novelty_cycle(workspace / "input_task", cfg, cycle, offline=offline,
                           reviewer_settings=_reviewer_settings(pipeline_cfg), author_settings=_author_settings(pipeline_cfg),
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
    if "investigation" in remaining and (workspace / "run/final_evaluation/invocation.json").exists():
        raise ValueError("Final confirmation has already been invoked. Do not tune this study on confirmation results; create a new study with a new frozen confirmation protocol")
    return remaining


def run_harness(args: Namespace, overrides: list[str]) -> int:
    from .cli import pipeline_args
    from src.pipeline.cli import load_pipeline_config, config_error_message, main as pipeline_main
    managed_paths = {"paths.paper_dir", "paths.ablations_dir", "paths.posthoc_dir", "paths.final_evaluation_dir"}
    for item in overrides:
        if item.split("=", 1)[0] in managed_paths:
            raise ValueError("Harness stage paths are managed inside its workspace; do not override " + item.split("=", 1)[0])
    prior = None
    manifest = None
    if args.resume:
        if args.task or args.workspace or args.offline:
            raise ValueError("--resume uses frozen task/workspace/literature; omit task/--workspace/--offline")
        workspace = args.resume.expanduser().resolve()
        cfg, manifest = resume(workspace)
        prior = read_json(workspace / "harness_state.json")
        if prior.get("schema_version") != 1:
            raise ValueError("Unsupported harness state")
        saved = execution_overrides(workspace, manifest)
        old = {item.split("=", 1)[0]: item for item in saved}
        for item in overrides:
            key = item.split("=", 1)[0]
            if key not in RESUME_MUTABLE and old.get(key) != item:
                raise ValueError(f"Cannot change frozen harness setting {key} on resume; create a new workspace")
        overrides = _merge(saved, overrides)
        target = args.quality_target or prior["quality_target"]
        task = workspace / "input_task"
        offline = bool(prior.get("offline", False))
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
        if not pipeline_cfg.investigation.adaptive_enabled:
            raise ValueError("High-impact harness requires adaptive_investigation.enabled=true")
    requested = [phase for phase in PIPELINE_PHASES if phase in phases]
    remaining = _resume_plan(workspace, prior, phases) if prior else phases
    if prior and target != prior["quality_target"]:
        raise ValueError("Quality target is frozen; create a new workspace to change it")
    effective = _merge(overrides, ["execution.phases=" + ",".join(phase for phase in PIPELINE_PHASES if phase in remaining)])
    if prior and "investigation" in remaining and (workspace / "run/investigation").is_dir():
        effective = _merge(effective, ["investigation.execution.resume=true"])
    command = pipeline_args(workspace, cfg, effective) if remaining else []
    pipeline_cfg = load_pipeline_config(command) if command else pipeline_cfg
    if args.dry_run:
        print(json.dumps({"dry_run": True, "mode": "research_harness", "workspace": str(workspace),
                          "quality_target": target, "requested_phases": requested,
                          "remaining_phases": [p for p in PIPELINE_PHASES if p in remaining],
                          "pipeline_argv": command, "novelty_review": cfg.harness.novelty_review and not offline,
                          "human_submission_approval_required": True,
                          "note": "No files, API requests, agents, or training were created."}, ensure_ascii=False, indent=2))
        return 0
    # Even resume checks every remaining requested phase before any provider call.
    if remaining:
        ensure_execution_ready(pipeline_cfg, workspace / "run/task" if prior and (workspace / "run/task").is_dir() else task)
    state = HarnessState(workspace, phases=requested, quality_target=target,
                         input_sha256=manifest["input_task_sha256"] if manifest else None, prior=prior)
    state.payload["offline"] = offline
    state.payload["preflight"] = {"status": "passed", "checked_at": now(), "remaining_phases": sorted(remaining)}
    exit_code = 1
    try:
        if manifest is None:
            def after_literature(staged: Path, frozen_cfg: ResearchConfig) -> None:
                from .upstream.disclosure import write_integration_marker
                write_integration_marker(staged)
                state.payload["literature"] = {"status": "prepared", "checked_at": now()}
                state.save()
                report = _novelty_cycle(staged, frozen_cfg, workspace / "novelty/initial", offline=offline,
                                        reviewer_settings=_reviewer_settings(pipeline_cfg),
                                        author_settings=_author_settings(pipeline_cfg))
                state.payload["novelty_initial"] = {"status": report["status"], "report": "novelty/initial/novelty_report.json"}
                state.save()
                _dump(staged / "research_context/harness_novelty.json", report)
                agenda = read_json(workspace / "novelty/initial/research_agenda.json")
                _dump(staged / "research_agenda.json", agenda)
            manifest = prepare(task, workspace, offline=offline, overrides=overrides, after_literature=after_literature)
            state.payload["input_task_sha256"] = manifest["input_task_sha256"]
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
        exit_code = pipeline_main(command, event_callback=observe) if command else 0
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
    return exit_code
