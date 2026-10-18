"""Own the process tree of one trusted review CLI, including detached children.

Headless can create a second process group for its model CLI. Group signals alone
do not stop that child after Headless exits. Keep observed descendants across
reparenting, and check their start identity before each signal. This is lifecycle
cleanup for a cooperative CLI, not containment of hostile, rapidly daemonizing code.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from .bounded_process import BoundedProcessError, run_bounded_process

_OBSERVE_SECONDS = 0.05
_TERM_GRACE_SECONDS = 0.15
_CLEANUP_SECONDS = 2.0
_MAX_ENV_BYTES = 256 * 1024
OWNER_TOKEN_ENV = "AGSCI_REVIEW_OWNER_TOKEN"
_Result = TypeVar("_Result")


def _retry_inspection(action: Callable[[], _Result], deadline: float) -> _Result:
    while True:
        try:
            return action()
        except (OSError, ValueError, subprocess.SubprocessError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


@dataclass(frozen=True)
class _Process:
    pid: int
    parent: int
    started: str
    state: str
    uid: int


def _process_table() -> dict[int, _Process]:
    if sys.platform == "linux":
        result = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                pid = int(entry.name)
                result[pid] = _Process(pid, int(fields[1]), fields[19], fields[0], entry.stat().st_uid)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return result
    # macOS and other POSIX hosts expose a stable start timestamp through ps.
    output = subprocess.check_output(
        ["/bin/ps", "-axo", "pid=,ppid=,uid=,stat=,lstart="],
        stderr=subprocess.DEVNULL, env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        timeout=1, text=True,
    )
    result = {}
    for line in output.splitlines():
        fields = line.split(None, 4)
        if len(fields) == 5:
            pid, parent = int(fields[0]), int(fields[1])
            result[pid] = _Process(pid, parent, fields[4], fields[3], int(fields[2]))
    return result


def process_baseline() -> set[tuple[int, str]]:
    """Record existing processes before launch, without reading environments."""
    return {(item.pid, item.started) for item in _process_table().values()}


def _has_owner_token(item: _Process, token: str, deadline: float) -> bool:
    """Read only a new same-user process; retain only its marker match result."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OSError("review process ownership inspection exceeded its deadline")
    marker = f"{OWNER_TOKEN_ENV}={token}".encode("ascii")
    if sys.platform == "linux":
        try:
            with Path(f"/proc/{item.pid}/environ").open("rb") as stream:
                data = stream.read(_MAX_ENV_BYTES + 1)
        except (FileNotFoundError, ProcessLookupError):
            return False
        if len(data) > _MAX_ENV_BYTES:
            raise OSError("review process ownership inspection exceeded its size limit")
        return marker in data.split(b"\0")
    try:
        result = run_bounded_process(
            ["/bin/ps", "eww", "-p", str(item.pid), "-o", "command="],
            cwd="/", env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            timeout_seconds=min(0.25, remaining), max_output_bytes=_MAX_ENV_BYTES,
        )
    except (OSError, BoundedProcessError):
        # Never expose captured process environments as reviewer diagnostics.
        raise OSError("review process ownership inspection failed") from None
    return marker in result.stdout.split()


class ReviewProcessTree:
    """Track descendants of one Popen child without selecting unrelated jobs."""

    def __init__(
        self, process: subprocess.Popen[bytes], *, owner_token: str,
        baseline: set[tuple[int, str]],
    ) -> None:
        self.process = process
        self._owner_token = owner_token
        self._baseline = baseline
        self._known: dict[int, _Process] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root_observed = False
        self._ownership_incomplete = False

    def _observe(self, *, recover_deadline: float | None = None) -> dict[int, _Process]:
        table = _process_table()
        if recover_deadline is not None:
            # A fast successful CLI can exit before the first periodic sample.
            # Recover its reparented descendants through a random inherited
            # marker. Do not inspect preexisting or different-user processes.
            self._ownership_incomplete = False
            for item in table.values():
                if (
                    item.uid != os.getuid()
                    or item.pid in self._known
                    or (item.pid, item.started) in self._baseline
                    or item.state.startswith(("Z", "X"))
                ):
                    continue
                try:
                    matches = _has_owner_token(item, self._owner_token, recover_deadline)
                except (OSError, ValueError, subprocess.SubprocessError):
                    # An unreadable unrelated process must not hide a later
                    # owned orphan. Check the rest, then report incomplete
                    # inspection only after stopping all confirmed descendants.
                    self._ownership_incomplete = True
                    continue
                if matches:
                    self._known[item.pid] = item
            # Signaling always uses a new identity check, including PID reuse
            # during the bounded environment read above.
            table = _process_table()
        if not self._root_observed:
            root = table.get(self.process.pid)
            # A child that has not been reaped cannot have its PID reused.
            if root is not None and self.process.returncode is None:
                self._known[root.pid] = root
                self._root_observed = True
        owned = {
            pid for pid, previous in self._known.items()
            if pid in table and table[pid].started == previous.started
        }
        while True:
            added = {pid for pid, item in table.items() if item.parent in owned} - owned
            if not added:
                break
            owned.update(added)
            self._known.update((pid, table[pid]) for pid in added)
        return {pid: table[pid] for pid in owned if not table[pid].state.startswith(("Z", "X"))}

    def start(self) -> None:
        self._observe()

        def observe() -> None:
            while not self._stop.wait(_OBSERVE_SECONDS):
                try:
                    self._observe()
                except (OSError, ValueError, subprocess.SubprocessError):
                    # Retry transient inspection errors. close() requires a
                    # successful final inspection before reporting completion.
                    continue

        self._thread = threading.Thread(target=observe, name="review-process-tree", daemon=True)
        self._thread.start()

    def _signal(self, owned: dict[int, _Process], signum: int) -> None:
        # Children first gives a living parent a chance to reap them. Never send
        # a signal to a different process that reused a recorded PID.
        current = _process_table()
        for pid in sorted(owned, key=lambda value: value == self.process.pid):
            observed = current.get(pid)
            if observed is None or observed.started != owned[pid].started:
                continue
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        """Stop owned descendants and reap the direct child before returning."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.2)
            if self._thread.is_alive():
                raise RuntimeError("review process inspection did not stop")
        deadline = time.monotonic() + _CLEANUP_SECONDS
        recovery_failure: Exception | None = None
        try:
            owned = _retry_inspection(lambda: self._observe(recover_deadline=deadline), deadline)
            if self._ownership_incomplete:
                recovery_failure = OSError("review process ownership inspection did not complete")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            # An unreadable new process must not prevent cleanup of descendants
            # already observed. Keep the incomplete ownership check as a failure.
            recovery_failure = exc
            deadline = time.monotonic() + _CLEANUP_SECONDS
            owned = _retry_inspection(self._observe, deadline)
        if owned:
            _retry_inspection(lambda: self._signal(owned, signal.SIGTERM), deadline)
            grace = min(deadline, time.monotonic() + _TERM_GRACE_SECONDS)
            while owned and time.monotonic() < grace:
                time.sleep(0.01)
                owned = _retry_inspection(self._observe, deadline)
            while owned:
                # Always attempt escalation once, even if inspection consumed
                # the grace/deadline. Do not leave a known paid child running.
                _retry_inspection(lambda: self._signal(owned, signal.SIGKILL), deadline)
                time.sleep(0.01)
                owned = _retry_inspection(self._observe, deadline)
                if time.monotonic() >= deadline:
                    break
        # wait() is needed even when inspection already reports a zombie.
        self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
        if owned:
            raise RuntimeError("review process tree did not stop before cleanup deadline")
        if recovery_failure is not None:
            raise OSError("review process ownership inspection did not complete") from recovery_failure
