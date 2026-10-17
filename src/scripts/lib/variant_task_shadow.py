"""Prepare the public task view used by credential-bearing variant agents."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path


def prepare_variant_task_shadow(
    source: Path, private_root: Path, workspace: Path,
) -> tuple[str, ...]:
    from src.investigation.task_contract import load_task_contract
    from src.investigation.task_shadow import create_sealed_task_shadow

    source = source.resolve(strict=True)
    private_root = private_root.resolve(strict=True)
    workspace = workspace.resolve(strict=True)
    metadata = private_root.stat()
    if (metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or private_root.is_relative_to(source)
            or private_root.is_relative_to(workspace)):
        raise ValueError("variant task shadow needs a private directory outside agent mounts")
    contract_path = source / "task_contract.json"
    contract = load_task_contract(contract_path)
    shadow = private_root / "public-task"
    shadow.mkdir(mode=0o700)
    create_sealed_task_shadow(
        source, shadow,
        final_evaluator=contract.evaluators.final,
        final_resources=contract.resources.final_root,
        contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    )
    empty = private_root / "sealed-empty"
    empty.mkdir(mode=0o700)
    runtime = source / ".venv"
    runtime_mount = ""
    if runtime.exists() or runtime.is_symlink():
        python = runtime / "bin/python"
        if (runtime.is_symlink() or not runtime.is_dir()
                or python.is_symlink() or not python.is_file()):
            raise ValueError("variant task runtime must be a managed, non-symlink directory")
        runtime_mount = str(runtime)
        (shadow / ".venv").mkdir(mode=0o700)
    files = "\n".join((
        contract.evaluators.final, "task_contract.json",
        ".agentic_task_validation.json", "manual_validation.json",
    ))
    return str(shadow), files, contract.resources.final_root, str(empty), runtime_mount


def main() -> None:
    # The script is called directly from the shell wrapper, before an agent runs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    values = prepare_variant_task_shadow(*(Path(x) for x in sys.argv[1:]))
    for value in values:
        sys.stdout.buffer.write(value.encode("utf-8") + b"\0")


if __name__ == "__main__":
    main()
