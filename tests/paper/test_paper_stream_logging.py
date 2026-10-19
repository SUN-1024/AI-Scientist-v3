"""Paper log streams survive crashes without contaminating publication inputs."""

from __future__ import annotations

import os
import shlex
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from src.pipeline.revision_publication import _copy_directory_contents


LIBRARY = Path(__file__).resolve().parents[2] / "src/scripts/lib/paper_stream_logging.sh"


def _shell(output: Path) -> str:
    return (
        "set -euo pipefail\n"
        f"source {shlex.quote(str(LIBRARY))}\n"
        f"PYTHON_BIN={shlex.quote(sys.executable)}\n"
        f"OUTPUT_DIR={shlex.quote(str(output))}\n"
    )


def _run(output: Path, script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", _shell(output) + script],
        env=env, capture_output=True, text=True, timeout=10,
    )


def _cleanup(output: Path, pipe: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _run(
        output,
        f"LOG_PIPE_DIR={shlex.quote(str(pipe.parent))}\n"
        f"LOG_PIPE={shlex.quote(str(pipe))}\ncleanup_paper_stage_log_pipe\n",
        env,
    )


@pytest.mark.parametrize("location", ["external", "inside_output", "symlink_into_output"])
def test_log_pipe_is_private_unique_and_outside_publication_tree(tmp_path: Path, location: str) -> None:
    output = tmp_path / "paper"
    output.mkdir()
    temporary = (tmp_path if location == "external" else output) / "runtime with spaces"
    temporary.mkdir()
    if location == "symlink_into_output":
        alias = tmp_path / "runtime-alias"
        alias.symlink_to(temporary, target_is_directory=True)
        temporary = alias
    env = {**os.environ, "TMPDIR": str(temporary)}
    pipes = []
    try:
        for _ in range(2):
            result = _run(output, 'create_paper_stage_log_pipe\nprintf "%s\\n" "$LOG_PIPE"\n', env)
            assert result.returncode == 0, result.stderr
            pipe = Path(result.stdout.strip())
            pipes.append(pipe)
            assert not pipe.is_relative_to(output.resolve())
            assert stat.S_ISFIFO(pipe.lstat().st_mode)
            assert stat.S_IMODE(pipe.lstat().st_mode) == 0o600
            assert stat.S_IMODE(pipe.parent.stat().st_mode) == 0o700
        assert pipes[0] != pipes[1]
    finally:
        for pipe in pipes:
            result = _cleanup(output, pipe, env)
            assert result.returncode == 0, result.stderr
            assert not pipe.parent.exists()


def test_log_pipe_transports_exact_bytes_and_cleans_up_after_failure(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    runtime = tmp_path / "runtime"
    output.mkdir()
    runtime.mkdir()
    env = {**os.environ, "TMPDIR": str(runtime)}
    result = _run(output, '''
trap cleanup_paper_stage_log_pipe EXIT
create_paper_stage_log_pipe
printf '%s\n' "$LOG_PIPE"
cat "$LOG_PIPE" > "$OUTPUT_DIR/agent.log" &
reader=$!
printf 'first line\nsecond line\n' > "$LOG_PIPE"
wait "$reader"
exit 7
''', env)
    assert result.returncode == 7, result.stderr
    pipe = Path(result.stdout.strip())
    assert (output / "agent.log").read_bytes() == b"first line\nsecond line\n"
    assert not pipe.parent.exists()


def test_abrupt_exit_leaves_only_external_runtime_pipe_and_publication_still_copies(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    runtime = tmp_path / "runtime"
    target = tmp_path / "published"
    for directory in (output, runtime, target):
        directory.mkdir()
    (output / "paper.tex").write_text("unchanged manuscript\n")
    env = {**os.environ, "TMPDIR": str(runtime)}
    process = subprocess.Popen(
        ["bash", "-c", _shell(output) + '''
trap cleanup_paper_stage_log_pipe EXIT
create_paper_stage_log_pipe
printf '%s\n' "$LOG_PIPE"
sleep 60
'''], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    pipe = None
    try:
        assert process.stdout is not None
        pipe = Path(process.stdout.readline().strip())
        assert pipe.is_fifo()
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        assert pipe.is_fifo()
        _copy_directory_contents(output, target)
        assert (target / "paper.tex").read_text() == "unchanged manuscript\n"
        assert all(path.is_file() for path in output.iterdir())
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if pipe is not None:
            result = _cleanup(output, pipe, env)
            assert result.returncode == 0, result.stderr


def test_cleanup_rejects_directory_symlink_and_preserves_external_file(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    output.mkdir()
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    result = _run(output, 'create_paper_stage_log_pipe\nprintf "%s\\n" "$LOG_PIPE"\n', env)
    assert result.returncode == 0, result.stderr
    pipe = Path(result.stdout.strip())
    original = pipe.parent
    moved = tmp_path / "moved-runtime"
    victim = tmp_path / "external"
    victim.mkdir()
    protected = victim / "stream.fifo"
    protected.write_text("preserved data\n")
    original.rename(moved)
    original.symlink_to(victim, target_is_directory=True)
    try:
        assert _cleanup(output, pipe, env).returncode != 0
        assert protected.read_text() == "preserved data\n"
    finally:
        original.unlink()
        moved.rename(original)
        assert _cleanup(output, pipe, env).returncode == 0


def test_cleanup_preserves_unexpected_regular_file(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    output.mkdir()
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    result = _run(output, 'create_paper_stage_log_pipe\nprintf "%s\\n" "$LOG_PIPE"\n', env)
    assert result.returncode == 0, result.stderr
    pipe = Path(result.stdout.strip())
    pipe.unlink()
    pipe.write_text("unexpected data\n")
    try:
        assert _cleanup(output, pipe, env).returncode != 0
        assert pipe.read_text() == "unexpected data\n"
    finally:
        pipe.unlink()
        assert _cleanup(output, pipe, env).returncode == 0


def test_publication_still_rejects_old_in_tree_stream_fifo(tmp_path: Path) -> None:
    source = tmp_path / "paper"
    target = tmp_path / "staged"
    source.mkdir()
    target.mkdir()
    os.mkfifo(source / ".paper_evidence_audit_round_5_stream_20261018_230222.fifo")
    with pytest.raises(ValueError, match="special file"):
        _copy_directory_contents(source, target)
