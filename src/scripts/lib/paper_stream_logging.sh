#!/usr/bin/env bash
# Keep live log streams outside the manuscript publication tree.

create_paper_stage_log_pipe() {
  if ! LOG_PIPE_DIR="$("$PYTHON_BIN" - "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import os
import sys
import tempfile

output = Path(sys.argv[1]).resolve()
root = Path(tempfile.gettempdir()).resolve()
if root == output or output in root.parents:
    root = output.parent
if root == output or output in root.parents:
    raise ValueError("paper log streams need a runtime directory outside the output")
directory = Path(tempfile.mkdtemp(prefix=".agsci-paper-stream-", dir=root))
try:
    os.mkfifo(directory / "stream.fifo", 0o600)
except BaseException:
    directory.rmdir()
    raise
print(directory)
PY
  )"; then
    LOG_PIPE_DIR=""
    LOG_PIPE=""
    return 1
  fi
  LOG_PIPE="$LOG_PIPE_DIR/stream.fifo"
}

cleanup_paper_stage_log_pipe() {
  local cleanup_status=0
  if [ -n "${LOG_PIPE_DIR:-}" ] \
    && [ "${LOG_PIPE:-}" = "$LOG_PIPE_DIR/stream.fifo" ]; then
    "$PYTHON_BIN" - "$LOG_PIPE_DIR" <<'PY' || cleanup_status=$?
from pathlib import Path
import os
import stat
import sys

directory = Path(sys.argv[1])
if not directory.is_absolute() or not directory.name.startswith(".agsci-paper-stream-"):
    raise ValueError("invalid paper log runtime directory")
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
parent_fd = os.open(directory.anchor, flags)
try:
    for part in directory.parent.parts[1:]:
        next_fd = os.open(part, flags, dir_fd=parent_fd)
        os.close(parent_fd)
        parent_fd = next_fd
    try:
        leaf_fd = os.open(directory.name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        leaf_fd = None
    if leaf_fd is not None:
        try:
            original = os.fstat(leaf_fd)
            if original.st_uid != os.getuid() or stat.S_IMODE(original.st_mode) != 0o700:
                raise ValueError("paper log runtime directory is not private and owned")
            try:
                pipe = os.stat("stream.fifo", dir_fd=leaf_fd, follow_symlinks=False)
            except FileNotFoundError:
                pipe = None
            if pipe is not None:
                if not stat.S_ISFIFO(pipe.st_mode):
                    raise ValueError("paper log runtime entry is not a FIFO")
                os.unlink("stream.fifo", dir_fd=leaf_fd)
            current = os.stat(directory.name, dir_fd=parent_fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
                raise ValueError("paper log runtime directory changed during cleanup")
            os.rmdir(directory.name, dir_fd=parent_fd)
        finally:
            os.close(leaf_fd)
finally:
    os.close(parent_fd)
PY
  fi
  LOG_PIPE=""
  LOG_PIPE_DIR=""
  return "$cleanup_status"
}
