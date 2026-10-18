"""Retain recorded inputs when orchestration updates manuscript artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any

from src.pipeline.safe_fs import open_directory

_MAX_BYTES = 4 * 1024 * 1024
_MAX_PDF_BYTES = 64 * 1024 * 1024


def _read(directory_fd: int, name: str, *, optional: bool = False, limit: int = _MAX_BYTES) -> bytes | None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        if optional:
            return None
        raise
    except OSError as exc:
        raise RuntimeError(f"Unsafe paper provenance artifact: {name}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError(f"Paper provenance artifact is not a regular file: {name}")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"Paper provenance artifact exceeds {limit} bytes: {name}")
    return data


def _json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _publish(directory_fd: int, name: str, data: bytes, *, immutable: bool = False) -> None:
    if Path(name).name != name:
        raise RuntimeError("Paper provenance publication requires a direct filename")
    if immutable:
        existing = _read(directory_fd, name, optional=True, limit=len(data))
        if existing is not None:
            if existing != data:
                raise RuntimeError(f"Archived paper provenance artifact changed: {name}")
            return
    else:
        try:
            target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(target.st_mode):
                raise RuntimeError(f"Paper provenance publication target is not a regular file: {name}")
    temporary = ".paper_provenance.write." + secrets.token_hex(16)
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o400 if immutable else 0o600, dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
            except FileExistsError:
                if _read(directory_fd, name, limit=len(data)) != data:
                    raise RuntimeError(f"Archived paper provenance artifact changed: {name}") from None
        else:
            os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _reference(name: str) -> str:
    # The writer source is saved as evidence; do not put a template token in it.
    return "@evidence" + "[[" + name + "]]"


def _checked_provenance(directory_fd: int, name: str, original: bytes) -> tuple[dict[str, Any] | None, bytes | None]:
    raw = _read(directory_fd, name + ".prov.json", optional=True)
    if raw is None:
        return None, None
    payload = json.loads(raw)
    artifact = payload.get("artifact") if isinstance(payload, dict) else None
    if (not isinstance(artifact, dict) or artifact.get("path") not in {name, "./" + name}
            or artifact.get("sha256") != hashlib.sha256(original).hexdigest()
            or payload.get("schema_name") != "scientific-evidence.artifact-manifest"
            or not isinstance(payload.get("derived_from"), list)):
        raise RuntimeError(f"Recorded paper artifact hash or identity does not match its input: {name}")
    return payload, raw


def _archive(directory_fd: int, name: str, original: bytes, provenance: dict[str, Any], raw: bytes) -> dict[str, str]:
    prefix = name.removesuffix(".json")
    archive_id = hashlib.sha256(original + b"\0" + raw).hexdigest()
    prior_name = f".{prefix}.before.{archive_id}{Path(name).suffix}"
    provenance_sha = hashlib.sha256(raw).hexdigest()
    provenance_name = f".{prefix}.provenance.{provenance_sha}.json"
    writer = Path(__file__).read_bytes()
    writer_sha = hashlib.sha256(writer).hexdigest()
    writer_name = f".paper_provenance.writer.{writer_sha}.py"
    prior_record = deepcopy(provenance)
    prior_record["artifact"]["path"] = prior_name
    prior_record["derived_from"] = [*provenance["derived_from"], _reference(provenance_name)]
    for filename, body in [(prior_name, original), (provenance_name, raw),
                           (prior_name + ".prov.json", _json(prior_record)), (writer_name, writer)]:
        _publish(directory_fd, filename, body, immutable=True)
    return {"prior": prior_name, "writer": writer_name, "writer_sha256": writer_sha,
            "prior_sha256": hashlib.sha256(original).hexdigest(), "provenance_sha256": provenance_sha}


def _record(
    name: str, updated: bytes, provenance: dict[str, Any], lineage: dict[str, str], *,
    operation: str, references: list[str], details: dict[str, Any] | None = None,
) -> bytes:
    return _json({
        "schema_name": "scientific-evidence.artifact-manifest",
        "artifact": {**provenance["artifact"], "path": name, "sha256": hashlib.sha256(updated).hexdigest()},
        "derived_from": [_reference(lineage["prior"]), *[_reference(p) for p in references],
                         _reference(lineage["writer"])],
        "generated_by": {"script_path": lineage["writer"], "script_sha256": lineage["writer_sha256"]},
        "named_outputs": {"operation": operation, "prior_artifact_sha256": lineage["prior_sha256"],
                          "prior_provenance_sha256": lineage["provenance_sha256"], **(details or {})},
        "notes": "Orchestrator artifact publication; the prior bytes and original provenance are retained. "
                 "The recorded source publishes provenance for the named operation; it does not claim authorship "
                 "of the underlying scientific results or figures.",
    })


def write_recorded_figure_manifest(
    paper_dir: Path, before: dict[str, Any], after: dict[str, Any], *, figure_id: str,
) -> None:
    """Record a metadata edit without re-stamping stale inputs or experiment receipts."""
    name = "figure_manifest.json"
    with open_directory(paper_dir, role="paper figure provenance") as directory_fd:
        original = _read(directory_fd, name)
        assert original is not None
        if json.loads(original) != before:
            raise RuntimeError("Figure manifest changed before metadata publication")
        provenance, raw = _checked_provenance(directory_fd, name, original)
        if before == after:
            return
        updated = _json(after)
        if provenance is None:
            _publish(directory_fd, name, updated)
            return
        assert raw is not None
        lineage = _archive(directory_fd, name, original, provenance, raw)
        record = _record(name, updated, provenance, lineage, operation="concept_render_metadata_update",
                         references=[f"figures/{figure_id}.prompts.json"], details={"figure_id": figure_id})
        try:
            _publish(directory_fd, name, updated)
            _publish(directory_fd, name + ".prov.json", record)
        except Exception:
            _publish(directory_fd, name, original)
            _publish(directory_fd, name + ".prov.json", raw)
            raise


@contextmanager
def recorded_pdf_compilation(paper_dir: Path, tex_name: str) -> Iterator[None]:
    """Retain a registered PDF before compilation and bind the successful replacement."""
    name = Path(tex_name).stem + ".pdf"
    with open_directory(paper_dir, role="paper PDF provenance") as directory_fd:
        if _read(directory_fd, name + ".prov.json", optional=True) is None:
            yield
            return
        original = _read(directory_fd, name, limit=_MAX_PDF_BYTES)
        assert original is not None
        provenance, raw = _checked_provenance(directory_fd, name, original)
        assert provenance is not None and raw is not None
        lineage = _archive(directory_fd, name, original, provenance, raw)
        try:
            yield
            updated = _read(directory_fd, name, limit=_MAX_PDF_BYTES)
            assert updated is not None
            if _read(directory_fd, name + ".prov.json") != raw:
                raise RuntimeError("PDF provenance changed during compilation")
            if updated == original:
                return
            inputs = [tex_name, Path(tex_name).stem + ".fls", Path(tex_name).stem + ".bbl",
                      "references.bib", "figure_manifest.json"]
            references = [p for p in inputs if (paper_dir / p).is_file() and not (paper_dir / p).is_symlink()]
            record = _record(name, updated, provenance, lineage, operation="latex_compile", references=references)
            _publish(directory_fd, name + ".prov.json", record)
        except Exception:
            _publish(directory_fd, name, original)
            _publish(directory_fd, name + ".prov.json", raw)
            raise
