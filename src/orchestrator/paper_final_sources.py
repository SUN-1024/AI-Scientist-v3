"""Carry the actual sealed-final program into writing, audit and revision."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.final_evaluation.runner import (
    FinalEvaluationError, resolve_frozen_incumbent, verify_final_manifest,
)
from src.orchestrator.paper_study_sources import (
    MAX_SOURCE_ENTRIES, SourceTree, _code_only, _publish_sources, _read_tree, _require_same_sources,
)
from src.pipeline.safe_fs import UnsafeGluePathError, open_directory

_DESTINATION = "final_program"
_SOURCE_PATH = f"source_evidence/{_DESTINATION}/program"


def _final_tree_digest(tree: SourceTree) -> str:
    """Apply the final runner's content framing to one bounded, no-follow snapshot.

    This differs from the fallback selection digest, which also includes modes.
    Hash the bytes that will be copied, so a later source change cannot authenticate
    a different snapshot.
    """
    digest = hashlib.sha256()
    for relative, (content, _) in sorted(tree.items()):
        encoded = relative.encode()
        if content is None:
            digest.update(b"dir\0" + encoded + b"\0")
        else:
            digest.update(b"file\0" + encoded + b"\0" + str(len(content)).encode() + b"\0")
            digest.update(content)
    return "sha256:" + digest.hexdigest()


def _sources(
    paper_data: dict[str, Any], investigation_dir: Path,
    *, final_evaluation_dir: Path | None = None,
) -> tuple[SourceTree, dict[str, Any]] | None:
    final = paper_data.get("final_evaluation")
    if final is None:
        if "final_program" in paper_data:
            raise ValueError("paper declares a final program without final evaluation")
        if (final_evaluation_dir is not None
                and (final_evaluation_dir / "final_manifest.json").exists()):
            raise ValueError("paper is missing the configured final evaluation and program sources")
        return None
    if not isinstance(final, dict):
        raise ValueError("paper final evaluation must contain an object")
    raw_manifest = final.get("manifest_path")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise ValueError("paper final evaluation has no manifest path")
    manifest_path = Path(raw_manifest).expanduser()
    if manifest_path.name != "final_manifest.json":
        raise ValueError("paper final manifest path has an unexpected filename")
    final_dir = final_evaluation_dir or manifest_path.parent
    if manifest_path.absolute() != (final_dir / "final_manifest.json").expanduser().absolute():
        raise ValueError("paper final manifest path differs from the configured final evaluation")
    try:
        manifest = verify_final_manifest(final_dir)
        program_dir, entrypoint = resolve_frozen_incumbent(investigation_dir)
        with open_directory(program_dir, role="final paper program") as source_fd:
            tree = _read_tree(source_fd)
    except (FinalEvaluationError, OSError, UnsafeGluePathError) as exc:
        raise ValueError(f"could not verify final paper program: {exc}") from exc
    expected_digest = manifest["program_tree_sha256"]
    if final.get("program_tree_sha256") != expected_digest:
        raise ValueError("paper final program binding differs from the final manifest")
    if _final_tree_digest(tree) != expected_digest:
        raise ValueError("paper program source digest differs from the final manifest")
    code = _code_only(tree)
    entrypoint_path = entrypoint.relative_to(program_dir).as_posix()
    if entrypoint_path not in code or code[entrypoint_path][0] is None:
        raise ValueError("final program entrypoint is excluded from the readable source inventory")
    combined = {"program": (None, 0o700)}
    combined.update({f"program/{path}": value for path, value in code.items()})
    if len(combined) > MAX_SOURCE_ENTRIES:
        raise ValueError("combined paper source entry limit exceeded")
    context = {
        "evaluation_scope": "sealed_final",
        "program_dir": str(program_dir),
        "entrypoint_path": entrypoint_path,
        "program_tree_sha256": expected_digest,
        "source_evidence": {"program": _SOURCE_PATH},
    }
    return combined, context


def stage_final_sources(
    paper_data: dict[str, Any], output_dir: Path, investigation_dir: Path,
) -> None:
    """Use the frozen incumbent, independently of the highest development row."""
    result = _sources(paper_data, investigation_dir)
    if result is None:
        return
    tree, context = result
    try:
        _publish_sources(output_dir, tree, destination_name=_DESTINATION)
    except (OSError, UnsafeGluePathError) as exc:
        raise ValueError(f"could not safely stage final paper sources: {exc}") from exc
    paper_data["final_program"] = context


def verify_final_sources(
    paper_data: dict[str, Any], output_dir: Path, investigation_dir: Path,
    *, final_evaluation_dir: Path | None = None,
) -> None:
    """Verify the whole copied source inventory without recreating missing files."""
    result = _sources(
        paper_data, investigation_dir, final_evaluation_dir=final_evaluation_dir,
    )
    if result is None:
        return
    expected, context = result
    if paper_data.get("final_program") != context:
        raise ValueError("paper final program context differs from the frozen evaluated program")
    try:
        with open_directory(output_dir / f"source_evidence/{_DESTINATION}",
                            role="staged final paper sources") as source_fd:
            actual = _read_tree(source_fd)
        _require_same_sources(actual, expected)
    except (OSError, UnsafeGluePathError) as exc:
        raise ValueError(f"could not verify staged final paper sources: {exc}") from exc
