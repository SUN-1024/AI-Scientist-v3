"""Host-owned receipts for explicitly native, non-contracted development runs.

These receipts bind scores to code, evaluator bytes, and declared task inputs. They do not provide a
container boundary and are never accepted by the sealed evaluation path.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .program_clone import clone_program_dir
from .program_spec import entrypoint_relpath_from_spec, load_program_spec

EVALUATION_INPUTS_DECLARATION = "native_evaluation_inputs.json"


def _hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular evaluation input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint(program_dir: Path) -> dict[str, str]:
    spec = load_program_spec(program_dir, required=True)
    if spec is None:
        raise ValueError("Missing program specification")
    root = program_dir / spec.code_root
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Candidate code root must be a real directory")
    records = {"program_spec.toml": _hash(program_dir / "program_spec.toml")}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise ValueError("Native candidate sources must not contain symlinks")
        if path.is_file():
            records[path.relative_to(program_dir).as_posix()] = _hash(path)
    return records


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate native evaluation input declaration key: {key}")
        result[key] = value
    return result


def _json(path: Path, *, unique_keys: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError(f"Invalid native evaluation artifact: {path.name}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object if unique_keys else None,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def evaluation_inputs_fingerprint(evaluator: Path) -> dict[str, str | None] | None:
    """Hash only task inputs declared as required or optional, including absence.

    The declaration lives beside evaluate.py. Without it, legacy tasks keep the
    evaluator-only receipt format. Declared paths are portable, task-relative
    regular files; symlink components are rejected, even when their targets are
    inside the task. Missing optional files are recorded so later creation also
    invalidates the receipt.
    """
    task_root = evaluator.parent
    declaration_path = task_root / EVALUATION_INPUTS_DECLARATION
    if not declaration_path.exists() and not declaration_path.is_symlink():
        return None
    declaration_hash = _hash(declaration_path)
    declaration = _json(declaration_path, unique_keys=True)
    if (
        set(declaration) - {"schema_version", "required", "optional"}
        or type(declaration.get("schema_version")) is not int
        or declaration["schema_version"] != 1
    ):
        raise ValueError("Invalid native evaluation input declaration schema")
    records: dict[str, str | None] = {EVALUATION_INPUTS_DECLARATION: declaration_hash}
    for kind in ("required", "optional"):
        names = declaration.get(kind, [])
        if not isinstance(names, list):
            raise ValueError(f"Native evaluation inputs {kind} must be a list")
        for name in names:
            if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
                raise ValueError("Native evaluation inputs must be relative file paths")
            relative = PurePosixPath(name)
            if (
                relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts
                or not relative.parts or relative.as_posix() != name or name in records
            ):
                raise ValueError(f"Invalid or duplicate native evaluation input path: {name}")
            path = task_root
            for part in relative.parts:
                path = path / part
                if path.is_symlink():
                    raise ValueError(f"Native evaluation input must not contain symlinks: {name}")
            if kind == "optional" and not path.exists():
                records[name] = None
            else:
                records[name] = _hash(path)
    if _hash(declaration_path) != declaration_hash:
        raise ValueError("Native evaluation input declaration changed while reading")
    return records


def validated_native_score(program_dir: Path, evaluator: Path) -> float | None:
    """Read a successful receipt only when code, evaluation inputs, and results match."""
    try:
        manifest = _json(program_dir / "evaluation_manifest.json")
        if not (
            manifest.get("producer") == "harness_eval"
            and manifest.get("execution_boundary") == "host"
            and manifest.get("program_id") == program_dir.name
            and manifest.get("exit_code") == 0
            and manifest.get("returncode") == 0
            and manifest.get("source_sha256") == source_fingerprint(program_dir)
            and manifest.get("evaluator_sha256") == _hash(evaluator)
            and manifest.get("evaluation_inputs_sha256") == evaluation_inputs_fingerprint(evaluator)
            and manifest.get("result_sha256") == {
                name: _hash(program_dir / name) for name in ("metrics.json", "correct.json")
            }
        ):
            return None
        metrics, correct = _json(program_dir / "metrics.json"), _json(program_dir / "correct.json")
        score = metrics.get("combined_score")
        if correct.get("correct") is not True or isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        return float(score) if math.isfinite(float(score)) else None
    except (OSError, ValueError, TypeError):
        return None


def run_native_evaluation(
    *, program_dir: Path, evaluator: Path, python_bin: str,
    timeout_seconds: int, environment: Mapping[str, str],
    attempt_id: str, idea_id: str,
) -> int:
    """Execute once and retain a receipt distinct from sandbox evaluation."""
    entrypoint_rel = entrypoint_relpath_from_spec(program_dir)
    if entrypoint_rel is None:
        raise ValueError("Candidate has no entrypoint")
    source_before, evaluator_before = source_fingerprint(program_dir), _hash(evaluator)
    evaluation_inputs_before = evaluation_inputs_fingerprint(evaluator)
    command = [python_bin, str(evaluator), "--program_path", str(program_dir / entrypoint_rel),
               "--results_dir", str(program_dir)]
    manifest: dict[str, Any] = {
        "producer": "harness_eval", "execution_boundary": "host",
        "attempt_id": attempt_id, "idea_id": idea_id, "program_id": program_dir.name,
        "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_before, "evaluator_sha256": evaluator_before,
        "evaluation_inputs_sha256": evaluation_inputs_before,
    }
    manifest_path = program_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        result = subprocess.run(command, cwd=evaluator.parent, env=dict(environment),
                                capture_output=True, text=True, timeout=timeout_seconds, check=False)
        returncode = result.returncode
        stdout, stderr = result.stdout or "", result.stderr or ""
        try:
            inputs_changed = (
                source_fingerprint(program_dir) != source_before
                or _hash(evaluator) != evaluator_before
                or evaluation_inputs_fingerprint(evaluator) != evaluation_inputs_before
            )
        except (OSError, ValueError, TypeError):
            inputs_changed = True
        if inputs_changed:
            returncode = 1
            stderr += "\nNative evaluation inputs changed during execution.\n"
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        stderr += f"\nNative evaluation timed out after {timeout_seconds}s.\n"
    (program_dir / "harness_evaluate_stdout.log").write_text(stdout)
    (program_dir / "harness_evaluate_stderr.log").write_text(stderr)
    manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),
                    exit_code=returncode, returncode=returncode)
    if returncode == 0:
        try:
            manifest["result_sha256"] = {name: _hash(program_dir / name)
                                         for name in ("metrics.json", "correct.json")}
        except (OSError, ValueError) as exc:
            manifest.update(exit_code=1, returncode=1, error=str(exc))
            returncode = 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return returncode


def ensure_native_baseline(
    *, task_dir: Path, seed_program_dir: Path, results_dir: Path, timeout_seconds: int,
) -> float:
    """Evaluate an isolated baseline copy once, and validate it before reuse."""
    if (task_dir / "task_contract.json").exists():
        raise ValueError("Contracted tasks require the sealed runtime")
    evaluator = task_dir / "evaluate.py"
    baseline = results_dir / ".native-baseline" / "program_0"
    if baseline.exists():
        score = validated_native_score(baseline, evaluator)
        if score is None or source_fingerprint(baseline) != source_fingerprint(seed_program_dir):
            raise ValueError("Native baseline inputs or receipt changed; use a new run directory")
        return score
    baseline.parent.mkdir(parents=True, exist_ok=True)
    clone_program_dir(src_dir=seed_program_dir, dst_dir=baseline)
    environment = dict(os.environ, TASK_SOURCE_DIR=str(task_dir), CUDA_VISIBLE_DEVICES="")
    returncode = run_native_evaluation(
        program_dir=baseline, evaluator=evaluator, python_bin=sys.executable,
        timeout_seconds=timeout_seconds, environment=environment,
        attempt_id="bootstrap", idea_id="baseline",
    )
    score = validated_native_score(baseline, evaluator)
    if returncode != 0 or score is None:
        raise ValueError(f"Native baseline evaluation failed; inspect {baseline}")
    return score
