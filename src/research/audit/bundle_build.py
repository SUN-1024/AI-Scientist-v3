"""Build a clean, deterministic reproduction bundle.

The bundle format is the one `evidence_bundle.verify_bundle` checks: a ``MANIFEST.json`` mapping
relative paths to SHA-256 values, plus the listed files. Python's ``tarfile`` is used directly so
macOS AppleDouble ``._*`` files never enter the archive, and all member metadata is normalised so
two builds of the same inputs produce byte-identical archives.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Iterable, Sequence

MANIFEST_NAME = "MANIFEST.json"
FIXED_MTIME = 0
FIXED_MODE = 0o644


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def collect_files(root: Path, entries: Sequence[str]) -> list[tuple[str, Path]]:
    root = root.expanduser().resolve()
    collected: dict[str, Path] = {}
    for entry in entries:
        relative = PurePosixPath(entry)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Bundle entry is unsafe: {entry!r}")
        candidate = root / relative
        if candidate.is_symlink():
            raise ValueError(f"Bundle entry is a symlink: {entry!r}")
        if candidate.is_dir():
            for path in sorted(candidate.rglob("*")):
                if path.is_symlink():
                    raise ValueError(f"Bundle contains a symlink: {path}")
                if path.is_file() and not path.name.startswith("._"):
                    collected[path.relative_to(root).as_posix()] = path
        elif candidate.is_file():
            collected[relative.as_posix()] = candidate
        else:
            raise ValueError(f"Bundle entry does not exist: {entry!r}")
    return sorted(collected.items())


def build_bundle(root: Path, entries: Sequence[str], output: Path) -> dict:
    """Write ``output`` as a tar.gz bundle and return its hashes."""
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    files = collect_files(root, entries)
    if not files:
        raise ValueError("Bundle would be empty")
    manifest = {
        "schema_version": 1,
        "files": {name: _sha256_bytes(path.read_bytes()) for name, path in files},
    }
    manifest_bytes = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        # mtime=0 in the gzip header keeps two builds byte-identical.
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                members = [(MANIFEST_NAME, manifest_bytes)]
                members += [(name, path.read_bytes()) for name, path in files]
                for name, body in members:
                    info = tarfile.TarInfo(name=name)
                    info.size = len(body)
                    info.mtime = FIXED_MTIME
                    info.mode = FIXED_MODE
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(body))
    return {
        "archive": str(output),
        "archive_sha256": _sha256_bytes(output.read_bytes()),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "files": len(manifest["files"]),
        "bytes": output.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory the entries are relative to")
    parser.add_argument("--output", type=Path, required=True, help="Destination .tar.gz")
    parser.add_argument("entries", nargs="+", help="Files or directories to include")
    args = parser.parse_args(argv)
    report = build_bundle(args.root, args.entries, args.output)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
