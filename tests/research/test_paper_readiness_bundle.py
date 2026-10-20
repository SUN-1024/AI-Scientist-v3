"""Paper readiness rejects a reproduction bundle that no longer matches its manuscript."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

from src.research.paper_readiness import _evidence_bundle


def _bundle(paper: Path, *, tamper: bool = False) -> str:
    paper.mkdir(parents=True, exist_ok=True)
    staging = paper.parent / "bundle-staging"
    staging.mkdir(exist_ok=True)
    (staging / "solver.py").write_text("print('ok')\n")
    files = {"solver.py": hashlib.sha256((staging / "solver.py").read_bytes()).hexdigest()}
    (staging / "MANIFEST.json").write_text(json.dumps({"files": files}) + "\n")
    manifest_hash = hashlib.sha256((staging / "MANIFEST.json").read_bytes()).hexdigest()
    (paper / "paper.tex").write_text(f"Bundle \\texttt{{{manifest_hash}}}.\n")
    if tamper:
        (staging / "solver.py").write_text("raise SystemExit(1)\n")
    archive = paper / "reproduction-bundle-test.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(staging.iterdir()):
            stream.add(path, arcname=path.name)
    return manifest_hash


def test_absent_bundle_is_not_a_gap(tmp_path: Path) -> None:
    assert _evidence_bundle(tmp_path) is None


def test_consistent_bundle_is_verified(tmp_path: Path) -> None:
    _bundle(tmp_path / "run/paper")
    result = _evidence_bundle(tmp_path)
    assert result["status"] == "verified"
    assert result["checked_files"] == 1
    assert result["manuscript_cites_manifest"] is True
    assert result["archive"] == "run/paper/reproduction-bundle-test.tar.gz"


def test_tampered_bundle_fails(tmp_path: Path) -> None:
    _bundle(tmp_path / "run/paper", tamper=True)
    result = _evidence_bundle(tmp_path)
    assert result["status"] == "failed"


def test_stale_manuscript_citation_fails(tmp_path: Path) -> None:
    _bundle(tmp_path / "run/paper")
    (tmp_path / "run/paper/paper.tex").write_text("Bundle \\texttt{" + "0" * 64 + "}.\n")
    result = _evidence_bundle(tmp_path)
    assert result["status"] == "failed"
    assert result["manuscript_cites_manifest"] is False


def test_two_bundles_are_rejected(tmp_path: Path) -> None:
    _bundle(tmp_path / "run/paper")
    (tmp_path / "run/paper/reproduction-bundle-second.tar.gz").write_bytes(b"not an archive")
    result = _evidence_bundle(tmp_path)
    assert result["status"] == "missing_or_invalid"
    assert "exactly one" in result["detail"]


def test_bundle_without_manuscript_is_rejected(tmp_path: Path) -> None:
    _bundle(tmp_path / "run/paper")
    (tmp_path / "run/paper/paper.tex").unlink()
    result = _evidence_bundle(tmp_path)
    assert result["status"] == "missing_or_invalid"
    assert "paper.tex" in result["detail"]
