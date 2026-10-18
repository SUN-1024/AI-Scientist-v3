"""Read-only runtime checks before research retrieval can spend API credits."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from src.agsci import doctor
from src.investigation.files import missing_task_scaffold_files
from src.pipeline.config import PipelineConfig, parse_pipeline_phases


def execution_checks(
    cfg: PipelineConfig, task_dir: Path, *, host: doctor.Host | None = None,
) -> list[doctor.CheckResult]:
    """Check local prerequisites without agents, network requests, or containers.

    A new workspace has no input_task yet. Inspect its source scaffold while
    retaining the resolved execution settings and leaving the config unchanged.
    Credentials can come from a CLI profile or login, so absence of familiar API
    key variables remains a warning. These checks do not verify remote access.
    """
    host = host or doctor.Host()
    task_dir = task_dir.expanduser()
    if not task_dir.is_absolute():
        task_dir = host.repo_root / task_dir
    source_cfg = replace(cfg, task=replace(cfg.task, dir=str(task_dir)))
    results = [doctor.check_run_summary(cfg), doctor.check_python(host)]
    try:
        phases = parse_pipeline_phases(cfg.execution.phases)
    except ValueError as exc:
        return [*results, doctor.CheckResult("phases", doctor.FAIL, str(exc))]

    drafting = "task_drafting" in phases or (
        cfg.investigation.draft_task_if_missing and not cfg.investigation.resume
    )
    if drafting and not cfg.container.enabled:
        results.append(doctor.CheckResult(
            "task drafting", doctor.FAIL, "task_drafting requires container.enabled=true",
            hint="enable Docker, or use execution.phases=investigation with a complete task scaffold",
        ))

    task_check = doctor.check_task_scaffold(host, source_cfg)
    if (
        task_check.status != doctor.FAIL and "investigation" in phases
        and not drafting and not cfg.investigation.resume
    ):
        missing = missing_task_scaffold_files(task_dir)
        if missing:
            task_check = doctor.CheckResult(
                "task", doctor.FAIL,
                "missing scaffold: " + ", ".join(str(path.relative_to(task_dir)) for path in missing),
                hint="provide the task scaffold, or select task_drafting with container.enabled=true",
            )
    results.append(task_check)

    if cfg.container.enabled:
        daemon = doctor.check_docker_daemon(host)
        results.append(daemon)
        if daemon.status == doctor.OK:
            results.append(doctor.check_agent_image(
                host, cfg.container.image, network=cfg.container.network, required=True,
            ))
        results.append(doctor.check_auth_root(cfg))
        env_file = doctor.check_env_file(host, cfg)
        # Doctor treats a missing optional env file as advisory. Once a run
        # explicitly names it (including the default .env), the runtime needs it.
        if env_file.status == doctor.WARN:
            env_file = replace(env_file, status=doctor.FAIL)
        results.append(env_file)
    results.append(doctor.check_selected_agent_cli(host, cfg))
    results.append(doctor.check_selected_credentials(host, cfg))
    results.append(doctor.check_required_gpus(host, cfg))
    # Do not use doctor's container GPU probe: that check starts a container.
    return results


def ensure_execution_ready(cfg: PipelineConfig, task_dir: Path) -> None:
    """Stop on known local failures; report uncertain credentials as warnings."""
    results = execution_checks(cfg, task_dir)
    failures = [result for result in results if result.status == doctor.FAIL]
    if failures:
        details = [
            f"  {result.name}: {result.detail}" + (f"; {result.hint}" if result.hint else "")
            for result in failures
        ]
        raise ValueError("Execution preflight failed:\n" + "\n".join(details))
    for result in results:
        if result.status == doctor.WARN:
            print(f"agsci research preflight warning: {result.name}: {result.detail}", file=sys.stderr)
