"""Offline lifecycle checks for a Headless-like detached reviewer process tree."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from src import review_backend as backend
from src.investigation import bounded_process, review_process


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process-tree signals")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _alive(pid: int) -> bool:
    """An orphan zombie is terminated, even before the OS has reaped it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    observed = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=2,
    )
    status = observed.stdout.strip()
    return bool(status) and not status.startswith("Z")


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "timed out waiting for test process state"


@dataclass
class _ReviewTree:
    directory: Path
    executable: Path

    @property
    def pids(self) -> list[int]:
        return [int(path.read_text()) for path in self.directory.glob("*.pid")]

    @property
    def environment(self) -> dict[str, str]:
        return {"HEADLESS_BIN": str(self.executable), "PATH": os.environ["PATH"],
                "HOME": str(self.directory)}

    @property
    def artifacts(self) -> Path:
        return self.directory / "artifacts"

    def wait_ready(self) -> None:
        _wait_until(lambda: (self.directory / "ready").exists())
        assert len(self.pids) == 3

    def live_pids(self) -> list[int]:
        return [pid for pid in self.pids if _alive(pid)]

    def release(self) -> None:
        (self.directory / "release").touch()

    def assert_stopped(self) -> None:
        _wait_until(lambda: not self.live_pids(), timeout=3)

    def cleanup(self) -> None:
        # Kill the detached group as well as known nodes if an assertion failed.
        child_file = self.directory / "child.pid"
        if child_file.exists():
            try:
                os.killpg(int(child_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def tree_factory(tmp_path: Path):
    trees: list[_ReviewTree] = []

    def create(name: str = "review", *, output_flood: bool = False) -> _ReviewTree:
        directory = tmp_path / name
        directory.mkdir()
        executable = directory / "headless"
        # Headless creates a new-session agent. Its own SIGTERM handler forwards
        # to that detached group, then exits before the stubborn child does.
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"root = Path({str(directory)!r})\n"
            "role = sys.argv[1] if len(sys.argv) > 1 else ''\n"
            "if role == '__grandchild':\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
            "    (root / 'grandchild.pid').write_text(str(os.getpid()))\n"
            "    while True: time.sleep(0.02)\n"
            "if role == '__child':\n"
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "    (root / 'child.pid').write_text(str(os.getpid()))\n"
            "    subprocess.Popen([sys.executable, __file__, '__grandchild'])\n"
            "    while True: time.sleep(0.02)\n"
            "(root / 'headless.pid').write_text(str(os.getpid()))\n"
            "child = subprocess.Popen([sys.executable, __file__, '__child'],\n"
            "    start_new_session=True, stdin=subprocess.DEVNULL,\n"
            "    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "def terminate(signum, frame):\n"
            "    (root / 'forwarded').touch()\n"
            "    try: os.killpg(child.pid, signal.SIGTERM)\n"
            "    except ProcessLookupError: pass\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, terminate)\n"
            "while not (root / 'grandchild.pid').exists(): time.sleep(0.01)\n"
            "(root / 'ready').touch()\n"
            f"if {output_flood!r}:\n"
            "    os.write(1, b'o' * 600)\n"
            "    os.write(2, b'e' * 600)\n"
            "while not (root / 'release').exists(): time.sleep(0.01)\n"
            "print(json.dumps({'protocolVersion': 1, 'type': 'result', 'exitCode': 0,\n"
            "    'data': {'finalMessage': '{\"ok\": true}', 'model': 'offline-test'}}))\n"
        )
        executable.chmod(0o700)
        tree = _ReviewTree(directory, executable)
        trees.append(tree)
        return tree

    yield create
    for tree in trees:
        tree.cleanup()


def _invoke(tree: _ReviewTree, **kwargs: Any):
    return backend.run_fresh_agent_json(
        "Return a JSON object.", work_dir=tree.artifacts, environment=tree.environment, **kwargs,
    )


def _worker(call: Callable[[], Any]) -> tuple[threading.Thread, dict[str, Any]]:
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = call()
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def _join(thread: threading.Thread) -> None:
    thread.join(timeout=8)
    assert not thread.is_alive(), "review worker did not finish after cleanup"


@pytest.mark.parametrize("in_worker", [False, True], ids=["main", "worker"])
def test_timeout_cleans_detached_tree_before_exit_callback(tree_factory, in_worker: bool) -> None:
    tree = tree_factory()
    exited: list[tuple[int, list[int]]] = []

    def run():
        return _invoke(tree, timeout_seconds=2,
                       on_process_exit=lambda process: exited.append((process.pid, tree.live_pids())))

    started = time.monotonic()
    if in_worker:
        thread, outcome = _worker(run)
        tree.wait_ready()
        _join(thread)
        assert isinstance(outcome.get("error"), backend.AgentReviewError)
        assert "timed out" in str(outcome["error"])
    else:
        with pytest.raises(backend.AgentReviewError, match="timed out"):
            run()
        tree.wait_ready()

    assert time.monotonic() - started < 8
    assert exited == [(int((tree.directory / "headless.pid").read_text()), [])]
    tree.assert_stopped()
    assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"


def test_worker_cancel_is_scoped_to_owned_tree_and_other_review_completes(tree_factory) -> None:
    cancelled_tree = tree_factory("cancelled")
    independent_tree = tree_factory("independent")
    cancel = threading.Event()
    callbacks: dict[str, list[list[int]]] = {"cancelled": [], "independent": []}
    first, first_outcome = _worker(lambda: _invoke(
        cancelled_tree, timeout_seconds=20, cancel_event=cancel,
        on_process_exit=lambda process: callbacks["cancelled"].append(cancelled_tree.live_pids()),
    ))
    second, second_outcome = _worker(lambda: _invoke(
        independent_tree, timeout_seconds=20,
        on_process_exit=lambda process: callbacks["independent"].append(independent_tree.live_pids()),
    ))
    try:
        cancelled_tree.wait_ready()
        independent_tree.wait_ready()
        cancel.set()
        _join(first)
        assert isinstance(first_outcome.get("error"), backend.AgentReviewError)
        assert "cancel" in str(first_outcome["error"])
        cancelled_tree.assert_stopped()
        assert callbacks["cancelled"] == [[]]
        assert second.is_alive()
        assert len(independent_tree.live_pids()) == 3
        assert callbacks["independent"] == []
        independent_tree.release()
        _join(second)
        assert "error" not in second_outcome
        payload, receipt = second_outcome["result"]
        assert payload == {"ok": True}
        assert receipt["status"] == "completed"
        assert callbacks["independent"] == [[]]
        independent_tree.assert_stopped()
    finally:
        cancel.set()
        independent_tree.release()
        cancelled_tree.cleanup()
        independent_tree.cleanup()
        first.join(timeout=3)
        second.join(timeout=3)


@pytest.mark.parametrize("signum,exception_name,exit_code", [
    (signal.SIGTERM, "SystemExit", 143),
    (signal.SIGINT, "KeyboardInterrupt", None),
], ids=["sigterm", "keyboard-interrupt"])
def test_main_thread_signals_preserve_semantics_and_clean_tree(
    tree_factory, signum: int, exception_name: str, exit_code: int | None,
) -> None:
    tree = tree_factory()
    result_path = tree.directory / "signal-result.json"
    script = tree.directory / "signal-harness.py"
    script.write_text(
        "import json, os, signal, subprocess\n"
        "from pathlib import Path\n"
        "from src.review_backend import run_fresh_agent_json\n"
        f"root = Path({str(tree.directory)!r})\n"
        "result = {'callbacks': []}\n"
        "previous_handler = signal.getsignal(signal.SIGTERM)\n"
        "def on_exit(process):\n"
        "    alive = []\n"
        "    for path in root.glob('*.pid'):\n"
        "        pid = int(path.read_text())\n"
        "        observed = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],\n"
        "            capture_output=True, text=True, timeout=2)\n"
        "        state = observed.stdout.strip()\n"
        "        if state and not state.startswith('Z'): alive.append(pid)\n"
        "    result['callbacks'].append(alive)\n"
        "try:\n"
        "    run_fresh_agent_json('Return JSON.', work_dir=root / 'artifacts',\n"
        f"        environment={tree.environment!r}, timeout_seconds=20, on_process_exit=on_exit)\n"
        "except BaseException as error:\n"
        "    result['exception'] = type(error).__name__\n"
        "    result['exit_code'] = getattr(error, 'code', None)\n"
        "result['handler_restored'] = signal.getsignal(signal.SIGTERM) == previous_handler\n"
        f"Path({str(result_path)!r}).write_text(json.dumps(result))\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    process = subprocess.Popen(
        [sys.executable, str(script)], cwd=REPO_ROOT, env=environment,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        tree.wait_ready()
        os.kill(process.pid, signum)
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, (stdout.decode(), stderr.decode())
        result = json.loads(result_path.read_text())
        assert result["exception"] == exception_name
        assert result["exit_code"] == exit_code
        assert result["handler_restored"] is True
        assert result["callbacks"] == [[]]
        tree.assert_stopped()
        assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=3)
        tree.cleanup()


@pytest.mark.parametrize("failing_callback", ["start", "exit"])
def test_callback_error_does_not_leave_live_tree(tree_factory, failing_callback: str) -> None:
    tree = tree_factory()
    exits: list[list[int]] = []

    def on_start(process) -> None:
        tree.wait_ready()
        if failing_callback == "start":
            raise RuntimeError("callback-sentinel")
        tree.release()

    def on_exit(process) -> None:
        exits.append(tree.live_pids())
        if failing_callback == "exit":
            raise RuntimeError("callback-sentinel")

    with pytest.raises(backend.AgentReviewError, match="callback-sentinel"):
        _invoke(tree, timeout_seconds=10, on_process_start=on_start, on_process_exit=on_exit)
    assert exits == [[]]
    tree.assert_stopped()
    assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"


def test_preexisting_cancel_never_starts_process_and_records_failed_receipt(tree_factory) -> None:
    tree = tree_factory()
    cancel = threading.Event()
    cancel.set()
    callbacks: list[str] = []
    with pytest.raises(backend.AgentReviewError, match="cancel"):
        _invoke(tree, timeout_seconds=10, cancel_event=cancel,
                on_process_start=lambda process: callbacks.append("start"),
                on_process_exit=lambda process: callbacks.append("exit"))
    assert callbacks == []
    assert tree.pids == []
    assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"


def test_output_limit_counts_both_streams_and_cleans_detached_tree(tree_factory, monkeypatch) -> None:
    tree = tree_factory(output_flood=True)
    monkeypatch.setattr(backend, "_MAX_RESPONSE_BYTES", 1024)
    exits: list[list[int]] = []
    started = time.monotonic()
    # Neither stream exceeds 1024 bytes alone; together they exceed the cap.
    with pytest.raises(backend.AgentReviewError, match="output.*limit|limit.*output"):
        _invoke(tree, timeout_seconds=10,
                on_process_exit=lambda process: exits.append(tree.live_pids()))
    assert time.monotonic() - started < 5
    tree.wait_ready()
    assert exits == [[]]
    tree.assert_stopped()
    assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"


def _assert_deferred_signal(
    tree_factory, injection_point: str, callback_raises: bool,
    signum: int, exception_name: str, exit_code: int | None,
) -> None:
    tree = tree_factory()
    result_path = tree.directory / "race-result.json"
    script = tree.directory / "race-harness.py"
    script.write_text(
        "import json, os, signal, subprocess, threading, time\n"
        "from pathlib import Path\n"
        "from src import review_backend as backend\n"
        f"root = Path({str(tree.directory)!r})\n"
        f"executable = {str(tree.executable)!r}\n"
        f"injection_point = {injection_point!r}\n"
        f"callback_raises = {callback_raises!r}\n"
        "result = {'callbacks': [], 'signals_sent': 0, 'start_called': False}\n"
        "cancel = threading.Event()\n"
        "previous_handlers = {number: signal.getsignal(number)\n"
        "    for number in (signal.SIGTERM, signal.SIGINT)}\n"
        "real_popen = subprocess.Popen\n"
        "real_register = backend.register_active_subprocess\n"
        "real_parse = backend.strict_json_object\n"
        "def inject():\n"
        "    result['signals_sent'] += 1\n"
        f"    os.kill(os.getpid(), {int(signum)})\n"
        "def popen(*args, **kwargs):\n"
        "    child = real_popen(*args, **kwargs)\n"
        "    argv = args[0] if args else kwargs['args']\n"
        "    if injection_point == 'spawn-return' and argv[0] == executable:\n"
        "        inject()\n"
        "    return child\n"
        "def register(child):\n"
        "    if injection_point == 'registration': inject()\n"
        "    real_register(child)\n"
        "def parse(text):\n"
        "    parsed = real_parse(text)\n"
        "    if injection_point == 'parse-success' and not result['signals_sent']: inject()\n"
        "    return parsed\n"
        "subprocess.Popen = popen\n"
        "backend.register_active_subprocess = register\n"
        "backend.strict_json_object = parse\n"
        "def on_start(child):\n"
        "    result['start_called'] = True\n"
        "    deadline = time.monotonic() + 8\n"
        "    while not (root / 'ready').exists():\n"
        "        if time.monotonic() >= deadline: raise TimeoutError('tree readiness')\n"
        "        time.sleep(0.01)\n"
        "    if callback_raises: raise RuntimeError('callback-after-signal')\n"
        "    if injection_point == 'parse-success': (root / 'release').touch()\n"
        "    if injection_point == 'event-lock':\n"
        "        with cancel._cond:\n"
        "            inject()\n"
        "            result['signals_sent'] += 1\n"
        f"            os.kill(os.getpid(), {int(signal.SIGINT if signum == signal.SIGTERM else signal.SIGTERM)})\n"
        "        result['event_lock_released'] = True\n"
        "        result['external_cancel_set'] = cancel.is_set()\n"
        "def on_exit(child):\n"
        "    alive = []\n"
        "    for path in root.glob('*.pid'):\n"
        "        pid = int(path.read_text())\n"
        "        observed = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],\n"
        "            capture_output=True, text=True, timeout=2)\n"
        "        state = observed.stdout.strip()\n"
        "        if state and not state.startswith('Z'): alive.append(pid)\n"
        "    result['callbacks'].append(alive)\n"
        "    result['reaped_at_exit'] = child.returncode is not None\n"
        "try:\n"
        "    backend.run_fresh_agent_json('Return JSON.', work_dir=root / 'artifacts',\n"
        f"        environment={tree.environment!r}, timeout_seconds=20,\n"
        "        cancel_event=cancel, on_process_start=on_start, on_process_exit=on_exit)\n"
        "except BaseException as error:\n"
        "    result['exception'] = type(error).__name__\n"
        "    result['exit_code'] = getattr(error, 'code', None)\n"
        "result['handlers_restored'] = all(signal.getsignal(number) == handler\n"
        "    for number, handler in previous_handlers.items())\n"
        f"Path({str(result_path)!r}).write_text(json.dumps(result))\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(script)], cwd=REPO_ROOT,
        env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=12)
        assert process.returncode == 0, (stdout.decode(), stderr.decode())
        result = json.loads(result_path.read_text())
        assert result["signals_sent"] == (2 if injection_point == "event-lock" else 1)
        if injection_point == "event-lock":
            assert result["event_lock_released"] is True
            assert result["external_cancel_set"] is False
        assert result["start_called"] is True
        assert result["exception"] == exception_name
        assert result["exit_code"] == exit_code
        assert result["handlers_restored"] is True
        assert result["reaped_at_exit"] is True
        assert result["callbacks"] == [[]]
        tree.wait_ready()
        tree.assert_stopped()
        assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=3)
        tree.cleanup()


@pytest.mark.parametrize("injection_point", ["spawn-return", "registration"])
@pytest.mark.parametrize("callback_raises", [False, True], ids=["callback-ok", "callback-error"])
@pytest.mark.parametrize("signum,exception_name,exit_code", [
    (signal.SIGTERM, "SystemExit", 143),
    (signal.SIGINT, "KeyboardInterrupt", None),
], ids=["sigterm", "keyboard-interrupt"])
def test_signal_during_spawn_or_registration_is_deferred_until_tree_cleanup(
    tree_factory, injection_point: str, callback_raises: bool,
    signum: int, exception_name: str, exit_code: int | None,
) -> None:
    _assert_deferred_signal(
        tree_factory, injection_point, callback_raises, signum, exception_name, exit_code,
    )


def test_sigterm_during_successful_json_parse_keeps_failed_receipt(tree_factory) -> None:
    _assert_deferred_signal(tree_factory, "parse-success", False, signal.SIGTERM, "SystemExit", 143)


@pytest.mark.parametrize("signum,exception_name,exit_code", [
    (signal.SIGTERM, "SystemExit", 143),
    (signal.SIGINT, "KeyboardInterrupt", None),
], ids=["sigterm-first", "keyboard-interrupt-first"])
def test_signal_while_cancel_event_condition_is_locked_never_reenters_event_set(
    tree_factory, signum: int, exception_name: str, exit_code: int | None,
) -> None:
    _assert_deferred_signal(tree_factory, "event-lock", False, signum, exception_name, exit_code)


@pytest.mark.parametrize("failure_count", [1, 2])
def test_cleanup_retries_transient_process_table_errors(tree_factory, monkeypatch, failure_count: int) -> None:
    tree = tree_factory()
    cancel = threading.Event()
    real_close = review_process.ReviewProcessTree.close
    real_process_table = review_process._process_table
    in_cleanup = False
    failures = 0
    exits: list[list[int]] = []

    def close(tracked_tree) -> None:
        nonlocal in_cleanup
        in_cleanup = True
        try:
            real_close(tracked_tree)
        finally:
            in_cleanup = False

    def process_table():
        nonlocal failures
        if (in_cleanup and threading.current_thread() is threading.main_thread()
                and failures < failure_count):
            failures += 1
            raise OSError("temporary process inspection failure")
        return real_process_table()

    def on_start(process) -> None:
        tree.wait_ready()
        cancel.set()

    monkeypatch.setattr(review_process.ReviewProcessTree, "close", close)
    monkeypatch.setattr(review_process, "_process_table", process_table)
    with pytest.raises(backend.AgentReviewError, match="cancel"):
        _invoke(tree, timeout_seconds=10, cancel_event=cancel, on_process_start=on_start,
                on_process_exit=lambda process: exits.append(tree.live_pids()))
    assert failures == failure_count
    assert exits == [[]]
    tree.assert_stopped()


def test_fallback_does_not_signal_reaped_group_after_exit_callback_error(tree_factory, monkeypatch) -> None:
    tree = tree_factory()
    group_signals: list[tuple[int, int]] = []
    exits: list[list[int]] = []

    def signal_group(process, signum: int) -> None:
        group_signals.append((process.pid, signum))
        raise AssertionError("must not signal an already reaped process group")

    def on_start(process) -> None:
        tree.wait_ready()
        tree.release()

    def on_exit(process) -> None:
        assert process.returncode is not None
        exits.append(tree.live_pids())
        raise RuntimeError("post-cleanup-callback-sentinel")

    monkeypatch.setattr(bounded_process, "_signal_process_group", signal_group)
    with pytest.raises(backend.AgentReviewError, match="post-cleanup-callback-sentinel"):
        _invoke(tree, timeout_seconds=10, on_process_start=on_start, on_process_exit=on_exit)
    assert exits == [[]]
    assert group_signals == []
    tree.assert_stopped()


def test_owner_token_is_redacted_even_if_reviewer_echoes_it(tree_factory, monkeypatch) -> None:
    tree = tree_factory()
    script = tree.executable.read_text()
    tree.executable.write_text(script.replace(
        "print(json.dumps(",
        f"os.write(2, os.environ[{review_process.OWNER_TOKEN_ENV!r}].encode())\nprint(json.dumps(",
        1,
    ))
    captured_tokens: list[str] = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        argv = args[0] if args else kwargs["args"]
        if argv[0] == str(tree.executable):
            captured_tokens.append(kwargs["env"].get(review_process.OWNER_TOKEN_ENV, ""))
        return real_popen(*args, **kwargs)

    def on_start(process) -> None:
        tree.wait_ready()
        tree.release()

    monkeypatch.setattr(subprocess, "Popen", popen)
    payload, receipt = _invoke(tree, timeout_seconds=10, on_process_start=on_start)
    assert payload == {"ok": True}
    assert receipt["status"] == "completed"
    capture_count = len(captured_tokens)
    assert capture_count == 1
    token_present = bool(captured_tokens[0])
    assert token_present, "review process environment omitted its ownership token"
    # Store only the boolean in assertions, so a failed test cannot print a token.
    leaked = any(captured_tokens[0].encode() in path.read_bytes()
                 for path in tree.artifacts.iterdir() if path.is_file())
    assert not leaked, "ownership token appeared in review artifacts"
    assert "[REDACTED]" in (tree.artifacts / "stderr.log").read_text()
    tree.assert_stopped()


def test_new_unmarked_unrelated_process_survives_review_cleanup(tree_factory, monkeypatch) -> None:
    tree = tree_factory()
    unrelated: list[subprocess.Popen[bytes]] = []
    inspected_pids: list[int] = []
    real_has_owner_token = review_process._has_owner_token
    exits: list[list[int]] = []

    def has_owner_token(item, token, deadline):
        inspected_pids.append(item.pid)
        return real_has_owner_token(item, token, deadline)

    def on_start(process) -> None:
        tree.wait_ready()
        environment = dict(os.environ)
        environment.pop(review_process.OWNER_TOKEN_ENV, None)
        # This starts after the baseline snapshot, at the same time as review
        # descendants. Age and user id alone must not make it an owned process.
        unrelated.append(subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        ))
        tree.release()

    monkeypatch.setattr(review_process, "_has_owner_token", has_owner_token)
    try:
        payload, receipt = _invoke(
            tree, timeout_seconds=10, on_process_start=on_start,
            on_process_exit=lambda process: exits.append(tree.live_pids()),
        )
        assert payload == {"ok": True}
        assert receipt["status"] == "completed"
        assert exits == [[]]
        tree.assert_stopped()
        assert len(unrelated) == 1
        assert unrelated[0].pid in inspected_pids
        assert unrelated[0].poll() is None
        assert _alive(unrelated[0].pid)
    finally:
        for process in unrelated:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)


def test_unreadable_unrelated_candidate_cannot_block_owned_orphan_cleanup(tree_factory, monkeypatch) -> None:
    tree = tree_factory()
    unrelated: list[subprocess.Popen[bytes]] = []
    tracked_trees: list[Any] = []
    inspected_pids: list[int] = []
    exits: list[int] = []
    real_process_table = review_process._process_table
    real_has_owner_token = review_process._has_owner_token

    def track_root_only(tracked_tree) -> None:
        # Model the real sampling race deterministically: the CLI starts and
        # exits between descendant samples. Only marker recovery can find them.
        root = real_process_table()[tracked_tree.process.pid]
        tracked_tree._known[root.pid] = root
        tracked_tree._root_observed = True
        tracked_trees.append(tracked_tree)

    def process_table():
        table = real_process_table()
        if unrelated and unrelated[0].pid in table:
            first = unrelated[0].pid
            return {first: table[first], **{pid: item for pid, item in table.items() if pid != first}}
        return table

    def has_owner_token(item, token, deadline):
        inspected_pids.append(item.pid)
        if unrelated and item.pid == unrelated[0].pid:
            raise OSError("injected ownership inspection failure for unrelated process")
        return real_has_owner_token(item, token, deadline)

    def on_start(process) -> None:
        tree.wait_ready()
        assert set(tracked_trees[0]._known) == {process.pid}
        environment = dict(os.environ)
        environment.pop(review_process.OWNER_TOKEN_ENV, None)
        unrelated.append(subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        ))
        tree.release()

    monkeypatch.setattr(review_process.ReviewProcessTree, "start", track_root_only)
    monkeypatch.setattr(review_process, "_process_table", process_table)
    monkeypatch.setattr(review_process, "_has_owner_token", has_owner_token)
    try:
        with pytest.raises(backend.AgentReviewError, match="ownership inspection"):
            _invoke(tree, timeout_seconds=10, on_process_start=on_start,
                    on_process_exit=lambda process: exits.append(process.pid))
        assert inspected_pids[0] == unrelated[0].pid
        assert exits == [], "incomplete ownership checks must not report completed cleanup"
        tree.assert_stopped()
        orphan_pids = {int((tree.directory / name).read_text())
                       for name in ("child.pid", "grandchild.pid")}
        assert orphan_pids.issubset(inspected_pids)
        assert unrelated[0].poll() is None
        assert _alive(unrelated[0].pid)
        assert json.loads((tree.artifacts / "receipt.json").read_text())["status"] == "failed"
    finally:
        for process in unrelated:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
