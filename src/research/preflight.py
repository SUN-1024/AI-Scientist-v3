"""Bounded runtime checks before research retrieval can spend API credits."""

from __future__ import annotations

import sys
import os
import stat
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from src.agsci import doctor
from src.investigation.files import missing_task_scaffold_files
from src.pipeline.config import PipelineConfig, parse_pipeline_phases, validate_pipeline_config
from src.pipeline.descriptor_paths import DescriptorPathError, directory_descriptor_path


def _probe_descriptor_pinning(directory: Path) -> None:
    """Verify a traversable, identity-bound directory alias without writing."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    try:
        alias = directory_descriptor_path(descriptor)
        # stat(/dev/fd/N) alone is not a portable child-traversal guarantee.
        reopened = os.open(os.fspath(alias) + "/.", flags)
        try:
            expected, actual = os.fstat(descriptor), os.fstat(reopened)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise DescriptorPathError("directory descriptor alias changed identity")
        finally:
            os.close(reopened)
    finally:
        os.close(descriptor)


def _existing_runtime_directory(path: Path, repo_root: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    candidate = Path(os.path.abspath(candidate))
    while not candidate.exists() and not candidate.is_symlink():
        candidate = candidate.parent
    if not candidate.is_dir() or candidate.resolve(strict=True) != candidate:
        raise ValueError(f"Runtime probe parent must be a directory without symlink components: {candidate}")
    return candidate


def _runtime_filesystem_roots(cfg: PipelineConfig, host: doctor.Host) -> list[Path]:
    """Probe the planned output filesystem even before its workspace exists."""
    locations = [Path(cfg.paths.run_dir or cfg.paths.results_root)]
    if cfg.versioning.enabled and cfg.versioning.run_root:
        locations.append(Path(cfg.versioning.run_root))
    for value in (cfg.paths.paper_dir, cfg.paths.ablations_dir):
        if value:
            locations.append(Path(value))
    temporary = host.environ.get("AGENTIC_TEMP_ROOT") or host.environ.get("TMPDIR")
    if temporary:
        locations.append(Path(temporary))
    else:
        from src.investigation.temp_storage import _select_base
        locations.append(_select_base())
    return list(dict.fromkeys(_existing_runtime_directory(path, host.repo_root) for path in locations))


def _probe_transaction_filesystem(directory: Path) -> None:
    """Exercise the real journal/rename primitives inside one disposable tree."""
    from src.pipeline.portable_publication import (
        atomic_exchange_prepared, exchange_is_unsupported, finish_directory_replacement,
        prepare_directory_replacement,
    )
    with tempfile.TemporaryDirectory(prefix=".agsci-transaction-preflight-", dir=directory) as temporary:
        root = Path(temporary)
        for name, exchange in (("preferred", True), ("ordinary", False)):
            parent = root / name
            original, replacement = parent / "paper", parent / "replacement"
            original.mkdir(parents=True)
            replacement.mkdir()
            (original / "evidence.json").write_bytes(b'{"value":"original"}\n')
            (replacement / "evidence.json").write_bytes(b'{"value":"replacement"}\n')
            record = prepare_directory_replacement(original, replacement)
            if exchange:
                try:
                    atomic_exchange_prepared(record, parent=parent)
                except OSError as exc:
                    if not exchange_is_unsupported(exc):
                        raise
            finish_directory_replacement(original, record)
            if (original / "evidence.json").read_bytes() != b'{"value":"replacement"}\n' or set(parent.iterdir()) != {original}:
                raise ValueError("Publication transaction failed its content or cleanup check")
        restricted = root / "permission-probe"
        restricted.write_bytes(b"probe\n")
        restricted.chmod(0)
        try:
            if stat.S_IMODE(restricted.stat().st_mode) != 0:
                raise ValueError("Runtime filesystem does not preserve private file permissions")
            if os.geteuid() != 0:
                try:
                    descriptor = os.open(restricted, os.O_RDONLY | os.O_NOFOLLOW)
                except PermissionError:
                    pass
                else:
                    os.close(descriptor)
                    raise ValueError("Runtime filesystem does not enforce private file permissions")
        finally:
            restricted.chmod(0o600)


def execution_checks(
    cfg: PipelineConfig, task_dir: Path, *, host: doctor.Host | None = None,
    descriptor_probe: Callable[[Path], None] | None = None,
    transaction_probe: Callable[[Path], None] | None = None,
) -> list[doctor.CheckResult]:
    """Check prerequisites without agents, network requests, or containers.

    A new workspace has no input_task yet. Inspect its source scaffold while
    retaining the resolved execution settings and leaving the config unchanged.
    Credentials can come from a CLI profile or login, so absence of familiar API
    key variables remains a warning. Filesystem probes use disposable private
    directories on the actual runtime filesystems. No task files are changed.
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

    if cfg.versioning.enabled or phases.intersection({"review", "revision"}):
        try:
            (descriptor_probe or _probe_descriptor_pinning)(task_dir)
        except (OSError, DescriptorPathError) as exc:
            results.append(doctor.CheckResult(
                "host descriptor pinning", doctor.FAIL,
                f"Coordinator cannot safely pin directories for checkpoints/recovery: {exc}",
                hint="run the Linux coordinator with scripts/run-research-harness-container.sh; "
                     "Docker experiment workers do not change the coordinator's filesystem behavior",
            ))
        else:
            results.append(doctor.CheckResult(
                "host descriptor pinning", doctor.OK,
                "Coordinator directory aliases preserve identity and support traversal",
            ))
            location = "configured runtime locations"
            try:
                roots = _runtime_filesystem_roots(cfg, host)
                for root in roots:
                    location = str(root)
                    (transaction_probe or _probe_transaction_filesystem)(root)
            except (OSError, RuntimeError, ValueError) as exc:
                results.append(doctor.CheckResult(
                    "runtime filesystem transactions", doctor.FAIL,
                    f"Checkpoint/revision filesystem primitives failed on {location}: {exc}",
                    hint="place both the workspace and AGENTIC_TEMP_ROOT on the Linux coordinator's native Docker volume; "
                         "macOS shared mounts can pass descriptor checks but fail journaled publication",
                ))
            else:
                results.append(doctor.CheckResult(
                    "runtime filesystem transactions", doctor.OK,
                    f"Journaled rename, inode identity, cleanup and private permissions passed on {len(roots)} runtime location(s)",
                ))

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
    contract_path = task_dir / "task_contract.json"
    if contract_path.exists() or contract_path.is_symlink():
        from src.investigation.task_contract import load_task_contract
        try:
            contract = load_task_contract(contract_path)
            for relative in (contract.evaluators.search, contract.evaluators.final):
                if not (task_dir / relative).is_file():
                    raise ValueError(f"declared evaluator is missing: {relative}")
        except (OSError, ValueError) as exc:
            results.append(doctor.CheckResult("task contract", doctor.FAIL, str(exc)))
    elif "final_evaluation" in phases and not drafting:
        results.append(doctor.CheckResult("final contract", doctor.FAIL,
                                         "final_evaluation requires a frozen task_contract.json",
                                         hint="include task_drafting or provide a complete contracted task"))
    for phase in sorted(phases.intersection({"ablations", "final_evaluation", "revision"})):
        if not cfg.container.enabled:
            results.append(doctor.CheckResult(phase, doctor.FAIL, f"{phase} requires container.enabled=true"))
    if not cfg.container.enabled and phases.intersection({"paper", "revision"}):
        for binary in ("pdflatex", "bibtex"):
            if host.which(binary) is None:
                results.append(doctor.CheckResult("paper compiler", doctor.FAIL, f"{binary} is unavailable",
                                                 hint="install TeX Live or use the configured Docker image"))
    if phases.intersection({"review", "revision"}):
        if cfg.review.backend == "agent":
            headless = host.environ.get("HEADLESS_BIN")
            if (headless and not (Path(headless).expanduser().is_file() and os.access(Path(headless).expanduser(), os.X_OK))) or (not headless and host.which("headless") is None):
                results.append(doctor.CheckResult("review transport", doctor.FAIL,
                                                 "headless is unavailable; configure HEADLESS_BIN or install the agent SDK CLI"))
            reviewer_cfg = replace(cfg, container=replace(cfg.container, enabled=False),
                                   agent=replace(cfg.agent, name=cfg.review.agent_name or cfg.agent.name,
                                                 model=cfg.review.agent_model or cfg.agent.model,
                                                 profile=cfg.review.agent_profile or cfg.agent.profile))
            review_cli = doctor.check_selected_agent_cli(host, reviewer_cfg)
            results.append(replace(review_cli, name="review agent cli"))
        elif not host.environ.get("OPENAI_API_KEY") and not cfg.container.env_file:
            results.append(doctor.CheckResult("review credentials", doctor.FAIL,
                                             "OpenAI review requires OPENAI_API_KEY or a configured env file",
                                             hint="set credentials or review.backend=agent"))
    # Compose checks happen before retrieval. Validate every selected stage,
    # including optional phases that the outer pipeline may otherwise degrade.
    for error in validate_pipeline_config(source_cfg):
        results.append(doctor.CheckResult("pipeline configuration", doctor.FAIL, error))

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
    if phases.intersection({"task_drafting", "investigation", "ablations", "final_evaluation"}):
        results.append(doctor.check_required_gpus(host, cfg))
    else:
        # Pipeline posthoc consumes saved artifacts and forbids retraining.
        results.append(doctor.CheckResult("required gpus", doctor.OK,
                                         "remaining analysis/writing phases do not execute task evaluators"))
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


@contextmanager
def runtime_tool_path():
    """Discover standard macOS installs only for this Python invocation."""
    original = os.environ.get("PATH", "")
    extras = [str(path) for path in (Path("/Applications/Docker.app/Contents/Resources/bin"), Path("/Library/TeX/texbin"))
              if path.is_dir() and str(path) not in original.split(os.pathsep)]
    if extras:
        os.environ["PATH"] = os.pathsep.join([*extras, original])
    try:
        yield
    finally:
        os.environ["PATH"] = original
