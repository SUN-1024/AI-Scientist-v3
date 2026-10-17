"""Prepare immutable research inputs and report observed campaign state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.investigation.files import load_task_copy_ignore_patterns, task_copy_path_is_ignored
from src.orchestrator.state import write_private_text
from src.pipeline.safe_fs import ensure_directory, open_directory, validate_tree_symlinks_fd
from src.pipeline.task_tree import copy_task_tree_at, task_tree_manifest_at, verify_copied_task_tree
from .config import ResearchConfig, load_config
from .evidence import load_catalog
from .literature import prepare_literature

OMIT = (
    ".git", ".venv", "__pycache__", ".pytest_cache", ".env", ".env.*",
    ".agentic_runtime.json", ".agentic_task_snapshot.json", "research_context",
)


def copy_task(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.absolute()
    if destination.resolve().is_relative_to(source):
        raise ValueError("The research workspace must be outside its source task directory")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Refusing to overwrite an existing task: {destination}")
    if not (source / "task.md").is_file():
        raise ValueError(f"Task source has no task.md: {source}")
    patterns = (*load_task_copy_ignore_patterns(source), *OMIT)
    with open_directory(source, role="research task source") as source_fd:
        validate_tree_symlinks_fd(source_fd, relative_dir=Path(), role="research task", ignore_patterns=patterns)
        with ensure_directory(destination, role="research task copy") as destination_fd:
            copy_task_tree_at(source_fd, destination_fd, relative_dir=Path(), ignore_patterns=patterns)
            verify_copied_task_tree(source_fd, destination_fd, ignore_patterns=patterns)


def task_digest(task_dir: Path) -> str:
    with open_directory(task_dir, role="research task") as fd:
        records = task_tree_manifest_at(fd, relative_dir=Path(), ignore_patterns=())
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def prepare(task_dir: Path, workspace: Path, *, offline: bool, overrides: list[str]) -> dict[str, Any]:
    cfg = load_config(task_dir)
    if workspace.exists() or workspace.is_symlink():
        raise ValueError("Workspace already exists; use --resume or select a new workspace")
    if workspace.resolve().is_relative_to(task_dir.resolve()):
        raise ValueError("The research workspace must be outside its source task directory")
    workspace.mkdir(parents=True)
    staged = workspace / "input_task"
    copy_task(task_dir, staged)
    for relative in ("research_context", "research_context/catalog.json"):
        if task_copy_path_is_ignored(
            relative, is_dir="." not in Path(relative).name,
            patterns=load_task_copy_ignore_patterns(staged),
        ):
            raise ValueError(".agentic_task_ignore must include the generated research_context in the pipeline snapshot")
    catalog = prepare_literature(staged, cfg.literature, offline=offline)
    load_catalog(staged)
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_task": str(task_dir.resolve()), "input_task_sha256": task_digest(staged),
        "research_config": asdict(cfg), "pipeline_overrides": overrides,
        "literature_status": catalog["status"], "state": "prepared",
    }
    write_private_text(workspace / "research_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def resume(workspace: Path) -> tuple[ResearchConfig, dict[str, Any]]:
    path = workspace / "research_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Workspace has no valid research_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Invalid research workspace manifest")
    staged = workspace / "input_task"
    if task_digest(staged) != manifest.get("input_task_sha256"):
        raise ValueError("Prepared research inputs changed; create a new workspace to change the study")
    cfg = load_config(staged)
    if asdict(cfg) != manifest.get("research_config"):
        raise ValueError("Prepared research configuration does not match its manifest")
    if not isinstance(manifest.get("pipeline_overrides"), list) or not all(
        isinstance(value, str) for value in manifest["pipeline_overrides"]
    ):
        raise ValueError("Invalid saved pipeline overrides")
    load_catalog(staged)
    return cfg, manifest


def execution_overrides(workspace: Path, manifest: dict[str, Any]) -> list[str]:
    """Resume the latest effective budgets/model, rather than only preparation options."""
    path = workspace / "research_execution.json"
    if not path.exists() and not path.is_symlink():
        return list(manifest["pipeline_overrides"])
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Invalid research execution receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, dict) or receipt.get("schema_version") != 1
        or receipt.get("input_task_sha256") != manifest["input_task_sha256"]
        or not isinstance(receipt.get("pipeline_overrides"), list)
        or not all(isinstance(value, str) for value in receipt["pipeline_overrides"])
    ):
        raise ValueError("Research execution receipt does not match the frozen inputs")
    return receipt["pipeline_overrides"]


def record_execution(workspace: Path, manifest: dict[str, Any], overrides: list[str]) -> None:
    receipt = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "input_task_sha256": manifest["input_task_sha256"], "pipeline_overrides": overrides,
    }
    write_private_text(workspace / "research_execution.json", json.dumps(receipt, ensure_ascii=False, indent=2))


def write_report(workspace: Path, *, exit_code: int | None = None) -> dict[str, Any]:
    cfg, manifest = resume(workspace)
    catalog = load_catalog(workspace / "input_task") or {"sources": [], "status": "unavailable"}
    run_dir = workspace / "run"
    trees = []
    for path in sorted(run_dir.rglob("search_tree.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()):
            continue
        tree = json.loads(path.read_text(encoding="utf-8"))
        trees.append({"path": str(path.relative_to(workspace)), "nodes": len(tree.get("nodes", {}))})
    decisions = []
    seen = set()
    for path in sorted(run_dir.rglob("idea_bundle.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(run_dir.resolve()) or "_idea_generation" in path.parts:
            continue
        bundle = json.loads(path.read_text(encoding="utf-8"))
        basis = bundle.get("research_basis")
        if isinstance(basis, dict):
            key = (bundle.get("idea_id"), json.dumps(basis, sort_keys=True))
            if key not in seen:
                decisions.append({"idea_id": bundle.get("idea_id"), "research_basis": basis})
                seen.add(key)
    report = {
        "schema_version": 1,
        "state": "prepared" if exit_code is None else "pipeline_returned" if exit_code == 0 else "pipeline_failed",
        "pipeline_exit_code": exit_code, "research_validity": "requires_evidence_review",
        "literature_status": catalog["status"], "source_count": len(catalog["sources"]),
        "literature_model_provider": cfg.literature.model_search.provider,
        "literature_model": cfg.literature.model_search.model,
        "model_search_trace": catalog.get("model_search_trace"),
        "full_text_count": sum(source["availability"] == "full_text" for source in catalog["sources"]),
        "search_strategy": cfg.search.strategy, "trees": trees, "idea_assessments": decisions,
        "input_task_sha256": manifest["input_task_sha256"],
        "pipeline_overrides": execution_overrides(workspace, manifest),
        "note": "A successful pipeline exit is not evidence of scientific novelty, correctness, or publication acceptance.",
    }
    write_private_text(workspace / "research_report.json", json.dumps(report, ensure_ascii=False, indent=2))
    lines = [
        "# Unified research run", "",
        f"- State: {report['state']}", f"- Pipeline exit: {exit_code}",
        f"- Sources: {report['source_count']} ({report['full_text_count']} full text)",
        f"- Retrieval: {report['literature_status']}", f"- Search policy: {cfg.search.strategy}",
        "", report["note"], "", "See input_task/research_context/ for literature and run/ for experimental evidence.",
    ]
    for decision in decisions:
        basis = decision["research_basis"]
        lines += ["", f"## {decision['idea_id']}: {basis['status']}", "", basis["contribution"],
                  "", "Discriminating test: " + basis["discriminating_test"]]
    write_private_text(workspace / "research_report.md", "\n".join(lines) + "\n")
    return report
