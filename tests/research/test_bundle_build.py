"""Build deterministic reproduction bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.research.audit.bundle_build import build_bundle, collect_files, main
from src.research.evidence_bundle import verify_bundle


def make_tree(root: Path) -> None:
    (root / "runs").mkdir(parents=True)
    (root / "note.tex").write_text("\\documentclass{article}\n")
    (root / "runs/receipt.json").write_text('{"status": "completed"}\n')
    (root / "runs/data").mkdir()
    (root / "runs/data/table.csv").write_text("a,b\n1,2\n")
    (root / "runs/data/._junk").write_text("appledouble")


def test_build_and_verify(tmp_path: Path) -> None:
    make_tree(tmp_path)
    archive = tmp_path / "out" / "bundle.tar.gz"
    report = build_bundle(tmp_path, ["note.tex", "runs"], archive)
    assert report["files"] == 3
    assert archive.is_file()
    result = verify_bundle(archive)
    assert result["status"] == "verified"
    assert result["manifest_sha256"] == report["manifest_sha256"]


def test_build_is_byte_deterministic(tmp_path: Path) -> None:
    make_tree(tmp_path)
    first = build_bundle(tmp_path, ["note.tex", "runs"], tmp_path / "a.tar.gz")
    second = build_bundle(tmp_path, ["note.tex", "runs"], tmp_path / "b.tar.gz")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_appledouble_files_are_skipped(tmp_path: Path) -> None:
    make_tree(tmp_path)
    files = collect_files(tmp_path, ["runs"])
    names = [name for name, _path in files]
    assert "runs/data/._junk" not in names
    assert "runs/data/table.csv" in names


def test_unsafe_or_missing_entries_are_rejected(tmp_path: Path) -> None:
    make_tree(tmp_path)
    for entry in ("../escape.txt", "/etc/passwd", "missing.txt"):
        with pytest.raises(ValueError):
            build_bundle(tmp_path, [entry], tmp_path / "x.tar.gz")


def test_symlink_is_rejected(tmp_path: Path) -> None:
    make_tree(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "note.tex")
    with pytest.raises(ValueError, match="symlink"):
        build_bundle(tmp_path, ["link"], tmp_path / "x.tar.gz")


def test_dot_entry_is_rejected(tmp_path: Path) -> None:
    make_tree(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        build_bundle(tmp_path, ["."], tmp_path / "x.tar.gz")


def test_cli_reports_hashes(tmp_path: Path, capsys) -> None:
    make_tree(tmp_path)
    output = tmp_path / "cli.tar.gz"
    assert main(["--root", str(tmp_path), "--output", str(output), "note.tex"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["files"] == 1
    assert hashlib.sha256(output.read_bytes()).hexdigest() == report["archive_sha256"]
