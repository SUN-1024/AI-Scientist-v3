"""Exercise the same standalone helper processes used by the paper shell script."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(script: str, *arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.defpath, "PYTHONNOUSERSITE": "1"}
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    # Keep third-party dependencies, but do not process an editable-install .pth
    # that would hide missing repository imports in the native helper process.
    dependency_paths = list(dict.fromkeys(
        sysconfig.get_path(name) for name in ("purelib", "platlib")
    ))
    bootstrap = (
        "import runpy, sys\nfrom pathlib import Path\n"
        "sys.path[0] = str(Path(sys.argv[1]).parent)\n"
        f"sys.path.extend({dependency_paths!r})\n"
        "sys.argv = sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", bootstrap,
         str(REPO_ROOT / "src/orchestrator" / script), *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_audit_workspace_lifecycle_from_outside_repo(tmp_path: Path) -> None:
    paper_dir = tmp_path / "project/run/paper"
    paper_dir.mkdir(parents=True)
    paper_tex = paper_dir / "paper.tex"
    paper_tex.write_text("An unchanged manuscript.\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()

    prepared = _run_cli(
        "paper_evidence_audit.py", "prepare-audit-workspace",
        "--source-dir", str(paper_dir), "--workspace-tag", "standalone-cli", cwd=outside,
    )
    assert prepared.returncode == 0, prepared.stderr
    readonly = Path(prepared.stdout.strip())
    try:
        assert (readonly / "paper_evidence_audit_packet.md").is_file()
        assert (readonly / "paper.tex").read_bytes() == paper_tex.read_bytes()
        assert not stat.S_IMODE((readonly / "paper.tex").stat().st_mode) & 0o222
    finally:
        cleaned = _run_cli(
            "paper_evidence_audit.py", "cleanup-readonly-workspace", "--path", str(readonly),
            "--source-dir", str(paper_dir), cwd=outside,
        )
        assert cleaned.returncode == 0, cleaned.stderr
    assert not readonly.exists()
    assert paper_tex.read_text(encoding="utf-8") == "An unchanged manuscript.\n"


def test_review_packet_from_outside_repo(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "paper.tex").write_text("An unchanged manuscript.\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    packet = tmp_path / "review-packet.md"

    result = _run_cli(
        "paper_review.py", "build-packet", "--paper-dir", str(paper_dir),
        "--output", str(packet), cwd=outside,
    )
    assert result.returncode == 0, result.stderr
    assert packet.is_file()
    assert "An unchanged manuscript." in packet.read_text(encoding="utf-8")
    assert (paper_dir / "paper.tex").read_text(encoding="utf-8") == "An unchanged manuscript.\n"
