"""Adapter for the end-to-end pipeline's final AI review phase."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.investigation import script_runners
from src.investigation.final_ai_review import FinalAiReviewResult, run_final_ai_review
from src.investigation.runtime import (
    ensure_text,
    register_active_subprocess,
    terminate_active_process_group,
    unregister_active_subprocess,
)

from .config import REPO_ROOT, PipelineConfig
from .paths import PipelinePaths


def describe_pipeline_review(
    *, cfg: PipelineConfig, paths: PipelinePaths, repo_root: Path = REPO_ROOT
) -> str:
    """Describe the managed review without presenting a bypass command."""
    return (
        f"managed src/aireview.py phase ({repo_root / 'src' / 'aireview.py'}): "
        f"{paths.paper_dir / 'paper.pdf'} -> "
        f"{paths.paper_dir / 'ai_review.json'} "
        f"(backend={cfg.review.backend}, restricted environment, timeout={cfg.review.timeout_seconds}s, "
        "manifest publication enabled)"
    )


def _review_env_file(cfg: PipelineConfig, *, repo_root: Path) -> Path | None:
    configured = str(cfg.container.env_file or "")
    if not configured:
        return None
    path = Path(configured).expanduser()
    return path if path.is_absolute() else repo_root / path


def _selected_novelty_evidence(paths: PipelinePaths) -> Path | None:
    """Rebuild the selected review on the host before exposing it to a reviewer."""
    from src.research.evidence import verify_methodology_for_paper, verify_selected_novelty_for_paper
    from src.research.portfolio_evidence import verify_portfolio_for_paper
    from src.research.portfolio_final_evidence import verify_portfolio_final_for_paper

    data_path = paths.paper_dir / "paper_data.json"
    if any(path.is_symlink() for path in (data_path, *data_path.parents)):
        raise ValueError("Paper data for the independent review must not use symlinks")
    paper_data = {}
    if data_path.exists():
        if not data_path.is_file() or data_path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Paper data for the independent review is invalid or oversized")
        paper_data = json.loads(data_path.read_text(encoding="utf-8"))
        if not isinstance(paper_data, dict):
            raise ValueError("Paper data for the independent review must be an object")
    selected = verify_selected_novelty_for_paper(
        paper_data, investigation_results_dir=paths.investigation_dir,
    )
    methodology = verify_methodology_for_paper(
        paper_data, investigation_results_dir=paths.investigation_dir,
    )
    portfolio = verify_portfolio_for_paper(
        paper_data, investigation_results_dir=paths.investigation_dir,
    )
    portfolio_final = verify_portfolio_final_for_paper(
        paper_data, investigation_results_dir=paths.investigation_dir,
    )
    if selected is None and methodology is None and portfolio is None and portfolio_final is None:
        return None
    packet = {
        "schema_version": 1,
        "scope": "host_revalidated_research_evidence",
        "interpretation": (
            "This packet binds the current novelty assessment, frozen methodology, and "
            "whole-idea development ledger where available. Its gaps and next actions "
            "limit the manuscript's claims. Supported does not establish final experimental "
            "success, external validity, human approval, or publication acceptance."
        ),
        "selected_novelty_review": selected,
    }
    if methodology is not None:
        packet["research_methodology"] = methodology
    if portfolio is not None:
        packet["research_portfolio"] = portfolio
    if portfolio_final is not None:
        packet["research_portfolio_final"] = portfolio_final
    body = (json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(body) > 512 * 1024:
        raise ValueError("Selected novelty evidence exceeds the independent review packet limit")
    output_dir = paths.run_dir / "harness_review_evidence"
    if any(path.is_symlink() for path in (output_dir, *output_dir.parents)):
        raise ValueError("Independent review evidence directory must not use symlinks")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"selected_novelty_{hashlib.sha256(body).hexdigest()}.json"
    fd, temporary = tempfile.mkstemp(prefix=".selected-novelty-", dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def run_pipeline_review(
    *,
    cfg: PipelineConfig,
    paths: PipelinePaths,
    repo_root: Path = REPO_ROOT,
    artifact_stem: str = "ai_review",
    manifest_key: str = "ai_review",
) -> FinalAiReviewResult:
    """Run the shared secure reviewer against the pipeline's published paper."""

    def execute_review(
        command: list[str], log_path: Path, environment: dict[str, str]
    ) -> tuple[str, int]:
        return script_runners.execute_managed_command(
            cmd=command,
            timeout_seconds=cfg.review.timeout_seconds,
            timeout_grace_seconds=cfg.review.timeout_grace_seconds,
            script_label="src/aireview.py",
            output_log_path=log_path,
            popen_factory=subprocess.Popen,
            register_subprocess=register_active_subprocess,
            unregister_subprocess=unregister_active_subprocess,
            terminate_process_group=terminate_active_process_group,
            ensure_text=ensure_text,
            env=environment,
        )

    evidence_paths = []
    for item in cfg.review.evidence_paths:
        path = Path(item).expanduser()
        evidence_paths.append(path if path.is_absolute() else paths.run_dir / path)
    reviewer_options: dict[str, Any] | None = {"evidence_paths": evidence_paths} if evidence_paths else None
    if cfg.review.backend == "agent":
        reviewer_options = {
            "backend": "agent",
            "agent": cfg.review.agent_name or cfg.agent.name,
            "model": cfg.review.agent_model or cfg.agent.model,
            "profile": cfg.review.agent_profile or cfg.agent.profile,
            "reasoning_effort": cfg.review.agent_reasoning_effort or cfg.agent.reasoning_effort,
            "reviewer_count": cfg.review.reviewer_count,
            "timeout_seconds": cfg.review.timeout_seconds,
            "evidence_paths": evidence_paths,
        }
    return run_final_ai_review(
        repo_root=repo_root,
        paper_dir=paths.paper_dir,
        paper_root=paths.paper_dir,
        results_dir=paths.run_dir,
        execute_command=execute_review,
        env_file=_review_env_file(cfg, repo_root=repo_root),
        artifact_stem=artifact_stem,
        manifest_key=manifest_key,
        reviewer_options=reviewer_options,
        reviewer_evidence_builder=lambda: _selected_novelty_evidence(paths),
    )
