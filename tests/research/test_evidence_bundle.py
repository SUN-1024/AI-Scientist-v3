"""Reproduction-bundle verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from src.research.evidence_bundle import bundle_hash_mentions, main as verify_main
from src.research.evidence_bundle import verify_bundle, verify_directory


def make_bundle(root: Path) -> Path:
    root.mkdir()
    (root / "solver.py").write_text("print('ok')\n")
    (root / "receipt.json").write_text('{"status": "completed"}\n')
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in ("solver.py", "receipt.json")}
    (root / "MANIFEST.json").write_text(json.dumps({"files": files}) + "\n")
    return root


def manifest_hash(root: Path) -> str:
    return hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()


def test_valid_directory_bundle_verifies(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    report = verify_bundle(bundle)
    assert report["status"] == "verified"
    assert report["checked_files"] == 2
    assert report["manifest_sha256"] == manifest_hash(bundle)
    assert report["missing"] == [] and report["mismatched"] == [] and report["unexpected"] == []


def test_tampered_file_fails(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / "solver.py").write_text("raise SystemExit(1)\n")
    report = verify_directory(bundle)
    assert report["status"] == "failed"
    assert report["mismatched"] == ["solver.py"]


def test_missing_and_unexpected_files_fail(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / "receipt.json").unlink()
    (bundle / "extra.txt").write_text("x")
    report = verify_directory(bundle)
    assert report["status"] == "failed"
    assert report["missing"] == ["receipt.json"]
    assert report["unexpected"] == ["extra.txt"]


def test_symlink_inside_bundle_fails(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / "link.py").symlink_to(bundle / "solver.py")
    report = verify_directory(bundle)
    assert report["status"] == "failed"
    assert "link.py (symlink)" in report["unexpected"]


def test_manifest_paths_must_stay_relative_and_simple(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    for name in ("../escape.py", "/abs.py", "./dot.py"):
        manifest = {"files": {name: "0" * 64}}
        (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="unsafe"):
            verify_directory(bundle)


def test_manifest_requires_files_and_lowercase_hash(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    (bundle / "MANIFEST.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(ValueError, match="nonempty"):
        verify_directory(bundle)
    (bundle / "MANIFEST.json").write_text(json.dumps({"files": {"solver.py": "A" * 64}}))
    with pytest.raises(ValueError, match="SHA-256"):
        verify_directory(bundle)


def test_expected_hash_is_enforced(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    good = manifest_hash(bundle)
    assert verify_bundle(bundle, expect_hash=good)["status"] == "verified"
    assert verify_bundle(bundle, expect_hash="0" * 64)["status"] == "failed"


def test_manuscript_must_cite_the_manifest_hash(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    digest = manifest_hash(bundle)
    manuscript = tmp_path / "NOTE.tex"
    manuscript.write_text(f"Bundle SHA-256 \\texttt{{{digest}}}.\n")
    assert verify_bundle(bundle, manuscript=manuscript)["status"] == "verified"
    manuscript.write_text("Bundle SHA-256 \\texttt{deadbeef}.\n")
    report = verify_bundle(bundle, manuscript=manuscript)
    assert report["status"] == "failed"
    assert report["manuscript_cites_manifest"] is False


def test_tar_archive_bundle_verifies(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in sorted(bundle.iterdir()):
            stream.add(path, arcname=path.name)
    report = verify_bundle(archive)
    assert report["status"] == "verified"
    assert report["archive"] == str(archive.resolve())


def test_archive_with_escaping_path_is_rejected(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(bundle / "MANIFEST.json", arcname="MANIFEST.json")
        stream.add(bundle / "solver.py", arcname="../solver.py")
    with pytest.raises(ValueError, match="unsafe"):
        verify_bundle(archive)


def test_bundle_hash_mentions_ignores_unrelated_hashes() -> None:
    manifest = "a" * 64
    unrelated = "b" * 64
    filler = "unrelated prose. " * 30
    text = (f"The bundle has manifest SHA-256\n{manifest}.\n{filler}"
            f"The image hash is {unrelated}.\n")
    assert bundle_hash_mentions(text) == {manifest}


def test_stale_second_bundle_hash_fails_verification(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    digest = manifest_hash(bundle)
    stale = "c" * 64
    manuscript = tmp_path / "NOTE.tex"
    manuscript.write_text(
        f"The bundle manifest SHA-256 is {digest}.\n"
        f"An older copy of the bundle cited {stale}.\n")
    report = verify_bundle(bundle, manuscript=manuscript)
    assert report["status"] == "failed"
    assert report["manuscript_stale_bundle_hashes"] == [stale]
    assert set(report["manuscript_bundle_hash_mentions"]) == {digest, stale}


def test_cli_verify_bundle_subcommand(tmp_path: Path) -> None:
    from src.research.cli import main as cli_main

    bundle = make_bundle(tmp_path / "bundle")
    assert cli_main(["verify-bundle", str(bundle)]) == 0
    (bundle / "receipt.json").write_text("tampered\n")
    assert cli_main(["verify-bundle", str(bundle)]) == 1


def test_cli_exit_codes(tmp_path: Path, capsys) -> None:
    bundle = make_bundle(tmp_path / "bundle")
    assert verify_main([str(bundle)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "verified"
    (bundle / "solver.py").write_text("tampered\n")
    assert verify_main([str(bundle)]) == 1
