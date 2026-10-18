"""Prepare immutable research inputs and report observed campaign state."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.investigation.files import load_task_copy_ignore_patterns, task_copy_path_is_ignored
from src.orchestrator.state import write_private_text
from src.pipeline.safe_fs import ensure_directory, open_directory, validate_tree_symlinks_fd
from src.pipeline.task_tree import copy_task_tree_at, task_tree_manifest_at, verify_copied_task_tree
from .config import MethodologyConfig, ResearchConfig, load_config
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


PREPARATION_FILE = "research_preparation.json"
PREPARATION_OUTPUTS = {
    "research_agenda.json", "research_context/harness_novelty.json",
    "research_context/upstream_integration.json",
}


def _matches_frozen_config(cfg: ResearchConfig, saved: object) -> bool:
    """Normalize legacy defaults without enabling new guidance during resume."""
    if not isinstance(saved, dict):
        return False
    saved = dict(saved)
    if "harness" not in saved:
        saved["harness"] = asdict(ResearchConfig().harness)
    if "methodology" not in saved:
        if cfg.methodology != MethodologyConfig():
            return False
        cfg.methodology = MethodologyConfig(enabled=False)
        saved["methodology"] = asdict(cfg.methodology)
    return asdict(cfg) == saved


def _verify_methodology(task_dir: Path, cfg: ResearchConfig) -> None:
    from .methodology import load_methodology

    marker = load_methodology(task_dir)
    if cfg.methodology.enabled:
        if marker is None or marker["profile"] != cfg.methodology.profile:
            raise ValueError("Frozen methodology does not match the research configuration")
    elif marker is not None:
        raise ValueError("Disabled methodology cannot contain a frozen guidance marker")


def _records(path: Path, *, source: bool = False) -> dict[str, Any]:
    patterns = (*load_task_copy_ignore_patterns(path), *OMIT) if source else ()
    with open_directory(path, role="research preparation input") as fd:
        records = task_tree_manifest_at(fd, relative_dir=Path(), ignore_patterns=patterns)
    return json.loads(json.dumps(records))


def _save_preparation(workspace: Path, journal: dict[str, Any]) -> None:
    journal["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_private_text(workspace / PREPARATION_FILE, json.dumps(journal, ensure_ascii=False, indent=2) + "\n")


def _read_preparation(workspace: Path) -> dict[str, Any]:
    path = workspace / PREPARATION_FILE
    if workspace.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Workspace has no valid preparation journal")
    journal = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(journal, dict) or journal.get("schema_version") != 1
            or journal.get("state") not in {"copying", "literature", "initial_review", "finalizing", "prepared"}
            or not isinstance(journal.get("source_records"), dict)
            or not isinstance(journal.get("pipeline_overrides"), list)
            or not all(isinstance(value, str) for value in journal["pipeline_overrides"])
            or type(journal.get("offline")) is not bool):
        raise ValueError("Invalid research preparation journal")
    return journal


def _verify_preparation_inputs(workspace: Path, journal: dict[str, Any]) -> list[Path]:
    staged = workspace / "input_task"
    current = _records(staged)
    expected = journal["source_records"]
    outputs = journal.get("generated_outputs", {})
    if not isinstance(outputs, dict) or set(outputs) - PREPARATION_OUTPUTS or not all(isinstance(v, str) for v in outputs.values()):
        raise ValueError("Invalid preparation output transaction")
    temporary_outputs = []
    # A hard kill can leave the atomic writer's temporary file. It is safe to
    # archive only a known output's regular, private, matching byte prefix.
    if journal["state"] == "finalizing":
        for name, text in outputs.items():
            output_path = Path(name)
            pattern = re.compile(r"\." + re.escape(output_path.name) + r"\.[0-9a-f]{16}\.tmp")
            for temporary, record in list(current.items()):
                path = Path(temporary)
                if temporary in expected or path.parent != output_path.parent or not pattern.fullmatch(path.name):
                    continue
                saved = staged / path
                content = text.encode()
                if record[0] != "file" or record[2] != "0o600" or saved.stat().st_size > len(content) or not content.startswith(saved.read_bytes()):
                    raise ValueError(f"Interrupted preparation output changed: {temporary}")
                temporary_outputs.append(saved)
                current.pop(temporary)
    # Output bytes are sealed before the first generated input is written. After
    # a crash each destination must still be the original or the sealed output.
    for name, text in outputs.items():
        record = current.get(name)
        generated = ["file", hashlib.sha256(text.encode()).hexdigest(), "0o600"]
        if record not in (expected.get(name), generated):
            raise ValueError(f"Prepared research output changed: {name}")
        current.pop(name, None)
    base = {name: value for name, value in current.items()
            if name != "research_context" and not name.startswith("research_context/")}
    original = {name: value for name, value in expected.items() if name not in outputs}
    if base != original:
        raise ValueError("Prepared research source inputs changed; create a new workspace")
    frozen = journal.get("literature_records")
    if frozen is not None:
        context = {name: value for name, value in current.items()
                   if name == "research_context" or name.startswith("research_context/")}
        if context != frozen:
            raise ValueError("Prepared literature changed; inspect the frozen evidence before resuming")
        load_catalog(staged)
    return temporary_outputs


def load_preparation(workspace: Path) -> tuple[ResearchConfig, dict[str, Any]]:
    """Read and validate an interrupted preparation without network or writes."""
    journal = _read_preparation(workspace)
    if journal["state"] == "copying":
        source = Path(journal["source_task"])
        if _records(source, source=True) != journal["source_records"]:
            raise ValueError("Source task changed during interrupted copy; create a new workspace")
        cfg = load_config(source)
    else:
        _verify_preparation_inputs(workspace, journal)
        cfg = load_config(workspace / "input_task")
    if not _matches_frozen_config(cfg, journal.get("research_config")):
        raise ValueError("Research preparation configuration changed")
    if journal["state"] in {"initial_review", "finalizing", "prepared"}:
        _verify_methodology(workspace / "input_task", cfg)
    return cfg, journal


def _archive_incomplete(workspace: Path, path: Path, label: str) -> str | None:
    if path.is_symlink():
        raise ValueError(f"Interrupted preparation path cannot be a symlink: {path.name}")
    if path.exists():
        archive = workspace / "preparation_attempts"
        archive.mkdir(exist_ok=True)
        if archive.is_symlink():
            raise ValueError("Preparation attempts cannot follow a symlink")
        destination = archive / f"{label}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        path.rename(destination)
        return str(destination.relative_to(workspace))
    return None


def prepare(task_dir: Path, workspace: Path, *, offline: bool, overrides: list[str],
            after_literature: Callable[[Path, ResearchConfig], dict[str, str] | None] | None = None,
            resume_preparation: bool = False, initial_state: dict[str, Any] | None = None) -> dict[str, Any]:
    if resume_preparation:
        cfg, journal = load_preparation(workspace)
        if offline != journal["offline"] or overrides != journal["pipeline_overrides"]:
            raise ValueError("Preparation settings are frozen; create a new workspace to change them")
        task_dir = Path(journal["source_task"])
    else:
        cfg = load_config(task_dir)
        if workspace.exists() or workspace.is_symlink():
            raise ValueError("Workspace already exists; use --resume or select a new workspace")
        if workspace.resolve().is_relative_to(task_dir.resolve()):
            raise ValueError("The research workspace must be outside its source task directory")
        journal = {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "state": "copying",
            "source_task": str(task_dir.resolve()), "source_records": _records(task_dir, source=True),
            "research_config": asdict(cfg), "pipeline_overrides": overrides, "offline": offline,
            "initial_harness_state": initial_state,
        }
        workspace.mkdir(parents=True)
        _save_preparation(workspace, journal)
    staged = workspace / "input_task"
    if journal["state"] == "copying":
        _archive_incomplete(workspace, staged, "input_task")
        copy_task(task_dir, staged)
        if _records(staged) != journal["source_records"]:
            raise ValueError("Source task changed during preparation; create a new workspace")
        journal["state"] = "literature"
        _save_preparation(workspace, journal)
    staged_cfg = load_config(staged)
    if not _matches_frozen_config(staged_cfg, journal["research_config"]):
        raise ValueError("Research configuration changed during preparation")
    for relative in ("research_context", "research_context/catalog.json"):
        if task_copy_path_is_ignored(
            relative, is_dir="." not in Path(relative).name,
            patterns=load_task_copy_ignore_patterns(staged),
        ):
            raise ValueError(".agentic_task_ignore must include the generated research_context in the pipeline snapshot")
    if journal["state"] == "literature":
        # Preserve failed/unfinished attempts. A completed, sealed catalog is
        # always reused, so resuming initial review never repeats retrieval.
        archived = _archive_incomplete(workspace, staged / "research_context", "literature")
        if archived is not None:
            journal["literature_retry"] = {
                "archived_attempt": archived, "explicit_resume": resume_preparation,
                "reason": "Previous literature stage was not sealed", "stage_requests_may_repeat": True,
            }
            _save_preparation(workspace, journal)
        catalog = prepare_literature(staged, cfg.literature, offline=offline)
        load_catalog(staged)
        if not offline and not catalog["sources"] and catalog.get("retrieval_errors"):
            journal["literature_error"] = "Literature retrieval failed with no usable sources; resume to retry preparation"
            _save_preparation(workspace, journal)
            raise ValueError(journal["literature_error"])
        if cfg.methodology.enabled:
            from .methodology import MARKER_PATH, write_methodology

            if task_copy_path_is_ignored(str(MARKER_PATH), is_dir=False,
                                         patterns=load_task_copy_ignore_patterns(staged)):
                raise ValueError(".agentic_task_ignore must retain the frozen methodology")
            write_methodology(staged, cfg.methodology.profile)
        _verify_methodology(staged, cfg)
        journal.pop("literature_error", None)
        journal["literature_records"] = {name: value for name, value in _records(staged).items()
                                         if name == "research_context" or name.startswith("research_context/")}
        journal["literature_status"] = catalog["status"]
        journal["state"] = "initial_review"
        _save_preparation(workspace, journal)
    if journal["state"] == "initial_review":
        outputs = after_literature(staged, cfg) if after_literature is not None else None
        _verify_preparation_inputs(workspace, journal)
        for relative in outputs or {}:
            if task_copy_path_is_ignored(relative, is_dir=False, patterns=load_task_copy_ignore_patterns(staged)):
                raise ValueError(f".agentic_task_ignore must retain generated research input: {relative}")
        journal["generated_outputs"] = outputs or {}
        journal["state"] = "finalizing"
        _save_preparation(workspace, journal)
    temporary_outputs = _verify_preparation_inputs(workspace, journal)
    for temporary in temporary_outputs:
        archived = _archive_incomplete(workspace, temporary, "generated_output")
        journal.setdefault("interrupted_output_files", []).append(archived)
    if temporary_outputs:
        _save_preparation(workspace, journal)
    for name, text in journal.get("generated_outputs", {}).items():
        write_private_text(staged / name, text)
    manifest = {
        "schema_version": 1, "created_at": journal["created_at"],
        "source_task": journal["source_task"], "input_task_sha256": task_digest(staged),
        "research_config": journal["research_config"], "pipeline_overrides": overrides,
        "literature_status": journal["literature_status"], "state": "prepared",
    }
    write_private_text(workspace / "research_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    journal["state"] = "prepared"
    journal["input_task_sha256"] = manifest["input_task_sha256"]
    _save_preparation(workspace, journal)
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
    if not _matches_frozen_config(cfg, manifest.get("research_config")):
        raise ValueError("Prepared research configuration does not match its manifest")
    if not isinstance(manifest.get("pipeline_overrides"), list) or not all(
        isinstance(value, str) for value in manifest["pipeline_overrides"]
    ):
        raise ValueError("Invalid saved pipeline overrides")
    load_catalog(staged)
    _verify_methodology(staged, cfg)
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
