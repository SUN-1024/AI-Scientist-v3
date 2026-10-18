"""Adapter for the end-to-end pipeline's final AI review phase."""

from __future__ import annotations

import subprocess
from pathlib import Path

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

    reviewer_options = None
    if cfg.review.backend == "agent":
        evidence_paths = []
        for item in cfg.review.evidence_paths:
            path = Path(item).expanduser()
            evidence_paths.append(path if path.is_absolute() else paths.run_dir / path)
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
        **({"reviewer_options": reviewer_options} if reviewer_options else {}),
    )
