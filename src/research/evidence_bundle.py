"""Verify a reproduction bundle against its manifest and against a manuscript citation.

A bundle is a directory or ``.tar.gz`` archive containing ``MANIFEST.json`` of the form
``{"files": {relative_path: sha256}}``. Verification is exact: every listed file must exist with
the recorded bytes, no unlisted file may be present, and symlinks are rejected. If a manuscript is
supplied, the bundle's manifest hash must also appear in the manuscript text, so a manuscript
cannot silently cite a stale bundle after a rebuild.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
from typing import Any

MANIFEST_NAME = "MANIFEST.json"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
HEX = set("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError(f"Bundle manifest path is unsafe: {name!r}")
    return name


def load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Bundle is missing a regular MANIFEST.json")
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Bundle manifest is oversized")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("Bundle manifest needs a nonempty 'files' mapping")
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64 or set(digest) - HEX:
            raise ValueError("Bundle manifest entries need a path and a lowercase SHA-256")
        _safe_relative(name)
    return manifest


def verify_directory(root: Path, *, expect_hash: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(root)
    manifest_bytes = (root / MANIFEST_NAME).read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    listed = dict(manifest["files"])
    missing, mismatched, unexpected = [], [], []
    for name, digest in listed.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            missing.append(name)
            continue
        if _sha256(path) != digest:
            mismatched.append(name)
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix()
            if relative not in listed and relative != MANIFEST_NAME:
                unexpected.append(relative)
        if path.is_symlink():
            unexpected.append(path.relative_to(root).as_posix() + " (symlink)")
    status = "verified" if not missing and not mismatched and not unexpected else "failed"
    if expect_hash is not None and expect_hash != manifest_sha:
        status = "failed"
    return {
        "schema_version": 1,
        "status": status,
        "root": str(root),
        "manifest_sha256": manifest_sha,
        "expected_manifest_sha256": expect_hash,
        "checked_files": len(listed),
        "missing": sorted(missing),
        "mismatched": sorted(mismatched),
        "unexpected": sorted(unexpected),
    }


def verify_bundle(bundle: Path, *, manuscript: Path | None = None,
                  expect_hash: str | None = None) -> dict[str, Any]:
    bundle = bundle.expanduser().resolve()
    if bundle.is_dir():
        report = verify_directory(bundle, expect_hash=expect_hash)
    elif bundle.is_file() and bundle.suffix == ".gz" and bundle.name.endswith(".tar.gz"):
        with tempfile.TemporaryDirectory(prefix="bundle-verify-") as directory:
            destination = Path(directory)
            with tarfile.open(bundle) as archive:
                if any(member.name.startswith("/") or ".." in PurePosixPath(member.name).parts
                       for member in archive.getmembers()):
                    raise ValueError("Bundle archive contains unsafe paths")
                archive.extractall(destination, filter="data")
            report = verify_directory(destination, expect_hash=expect_hash)
        report["archive"] = str(bundle)
    else:
        raise ValueError("Bundle must be a directory or a .tar.gz archive")
    if manuscript is not None:
        text = manuscript.expanduser().read_text(encoding="utf-8")
        cited = report["manifest_sha256"] in text
        expected = expect_hash is not None and expected_hash_in(text, expect_hash)
        mentions = bundle_hash_mentions(text)
        stale = sorted(token for token in mentions
                       if token != report["manifest_sha256"]
                       and (expect_hash is None or token != expect_hash))
        report["manuscript"] = str(manuscript.expanduser().resolve())
        report["manuscript_cites_manifest"] = cited
        report["manuscript_cites_expected_hash"] = expected if expect_hash is not None else None
        report["manuscript_bundle_hash_mentions"] = sorted(mentions)
        report["manuscript_stale_bundle_hashes"] = stale
        if not cited or (expect_hash is not None and not expected) or stale:
            report["status"] = "failed"
    return report


HASH = re.compile(r"[0-9a-f]{64}")


def bundle_hash_mentions(text: str, window: int = 240) -> set[str]:
    """SHA-256 tokens whose preceding text attributes them to the bundle or its manifest."""
    tokens = set()
    for match in HASH.finditer(text):
        context = text[max(0, match.start() - window):match.start()].lower()
        if "bundle" in context or "manifest" in context:
            tokens.add(match.group(0))
    return tokens


def expected_hash_in(text: str, expected: str) -> bool:
    return expected in text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Bundle directory or .tar.gz archive")
    parser.add_argument("--manuscript", type=Path, help="Require the manifest hash to appear here")
    parser.add_argument("--expect-hash", help="Require this exact manifest SHA-256")
    args = parser.parse_args(argv)
    report = verify_bundle(args.bundle, manuscript=args.manuscript, expect_hash=args.expect_hash)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
