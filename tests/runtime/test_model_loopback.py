"""Private model routing: real stream transport and shell/container boundaries."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from src.scripts.lib import model_loopback as route


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "src/scripts/lib/container_agent.sh"


def configuration(root: Path, port: int, *, selected: str = "local") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(
        'model_provider = "remote"\n'
        '[model_providers.remote]\nbase_url = "https://example.invalid/v1"\n'
        f'[model_providers.local]\nbase_url = "http://example:secret@127.0.0.1:{port}/v1?key=private"\n'
    )
    (root / "chosen.config.toml").write_text(f'model_provider = "{selected}"\n')


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_only_selected_provider_is_routed_and_guest_config_is_not_an_input(tmp_path: Path) -> None:
    configuration(tmp_path, 3425)
    assert route.selected_endpoint(tmp_path, "") is None
    assert route.selected_endpoint(tmp_path, "chosen") == ("127.0.0.1", 3425)
    (tmp_path / "chosen.config.toml").write_text(
        'model_provider = "local"\n'
        '[model_providers.local]\nbase_url = "https://[::1]:9443/v1"\n'
    )
    assert route.selected_endpoint(tmp_path, "chosen") == ("::1", 9443)


@pytest.mark.parametrize("url", ["http://localhost:12345/v1", "http://127.2.3.4:8888"])
def test_loopback_spellings(tmp_path: Path, url: str) -> None:
    (tmp_path / "config.toml").write_text(
        f'model_provider="selected"\n[model_providers.selected]\nbase_url="{url}"\n'
    )
    assert route.selected_endpoint(tmp_path, "") is not None


def test_symlink_and_credential_bearing_invalid_configuration_fail_without_disclosure(tmp_path: Path) -> None:
    secret = 'model_provider="selected"\n[model_providers.selected]\nbase_url="http://secret:token@[broken"\n'
    target = tmp_path / "outside.toml"
    target.write_text(secret)
    source = tmp_path / "auth"
    source.mkdir()
    (source / "config.toml").symlink_to(target)
    for symlink in (True, False):
        if not symlink:
            (source / "config.toml").unlink()
            (source / "config.toml").write_text(secret)
        result = subprocess.run(
            [sys.executable, str(Path(route.__file__)), "endpoint", "--source-home", str(source)],
            text=True, capture_output=True, timeout=5,
        )
        assert result.returncode == 125
        assert "secret" not in result.stdout + result.stderr
        assert "token" not in result.stdout + result.stderr


class Backend(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class RespondAfterHalfClose(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data = bytearray()
        while chunk := self.request.recv(4096):
            data.extend(chunk)
        self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
        self.request.sendall(bytes(data))


def start(server: socketserver.BaseServer) -> None:
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()


def exchange(address: tuple, payload: bytes) -> bytes:
    with socket.create_connection(address, timeout=3) as client:
        client.sendall(payload)
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := client.recv(4096):
            chunks.append(chunk)
        return b"".join(chunks)


def test_two_hop_stream_preserves_bytes_half_close_and_recovers_after_backend_restart() -> None:
    with tempfile.TemporaryDirectory(prefix="model-route-", dir="/tmp") as directory:
        backend = Backend(("127.0.0.1", 0), RespondAfterHalfClose)
        backend_address = backend.server_address
        start(backend)
        path = Path(directory) / "provider.sock"
        unix = route.UnixForwarder(path, backend_address)
        start(unix)
        frontend = route.TcpForwarder("127.0.0.1", 0, str(path))
        start(frontend)
        payload = b"Authorization: Bearer test-only\r\n\r\n" + b"data: raw-token\n\n" * 2000
        try:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert exchange(frontend.server_address, payload).endswith(payload)
            backend.shutdown()
            backend.server_close()
            with socket.create_connection(frontend.server_address, timeout=3) as client:
                client.settimeout(3)
                assert client.recv(1) == b""
            backend = Backend(backend_address, RespondAfterHalfClose)
            start(backend)
            assert exchange(frontend.server_address, b"recovered").endswith(b"recovered")
        finally:
            for server in (frontend, unix, backend):
                server.shutdown()
                server.server_close()


def test_unix_socket_can_bind_in_deep_workspace_path(tmp_path: Path) -> None:
    directory = tmp_path / ("a" * 80) / ("b" * 80)
    directory.mkdir(parents=True)
    path = directory / "provider.sock"
    before = Path.cwd()
    server = route.UnixForwarder(path, ("127.0.0.1", 3425))
    try:
        assert path.is_socket()
        assert Path.cwd() == before
    finally:
        server.server_close()


def test_client_forwards_termination_and_releases_listener(tmp_path: Path) -> None:
    port = free_port()
    ready = tmp_path / "ready"
    child_code = "from pathlib import Path;import os,time;Path(" + repr(str(ready)) + ").write_text(str(os.getpid()));time.sleep(60)"
    process = subprocess.Popen([
        sys.executable, str(Path(route.__file__)), "client", "--host", "127.0.0.1",
        "--port", str(port), "--socket", str(tmp_path / "unused.sock"),
        "--", sys.executable, "-c", child_code,
    ])
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
        with socket.socket() as check:
            check.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("replace", ["directory", "file"])
def test_guest_symlink_replacement_cannot_overwrite_host_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace: str,
) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    victim = outside / "relay.py"
    victim.write_bytes(b"preserve host contents")
    original = route.tempfile.mkdtemp

    def replaced_directory(*args, **kwargs):
        leaf = Path(original(*args, **kwargs))
        if replace == "directory":
            leaf.rmdir()
            leaf.symlink_to(outside, target_is_directory=True)
        else:
            (leaf / "relay.py").symlink_to(victim)
        return str(leaf)

    monkeypatch.setattr(route.tempfile, "mkdtemp", replaced_directory)
    with pytest.raises(OSError):
        route.run_container(("127.0.0.1", 3425), home, 1, ["unused", "command"])
    assert victim.read_bytes() == b"preserve host contents"
    assert sorted(p.name for p in outside.iterdir()) == ["relay.py"]


def test_socket_bind_remains_anchored_when_guest_renames_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    moved = home / "moved"
    home.mkdir()
    outside.mkdir()
    called = []
    original = route.UnixForwarder

    class RenameBeforeBind(original):
        def __init__(self, path, target, **kwargs):
            path.parent.rename(moved)
            path.parent.symlink_to(outside, target_is_directory=True)
            super().__init__(path, target, **kwargs)

    def child(_command):
        assert (moved / "provider.sock").is_socket()
        assert list(outside.iterdir()) == []
        called.append(True)
        return 17

    monkeypatch.setattr(route, "UnixForwarder", RenameBeforeBind)
    monkeypatch.setattr(route, "run_child", child)
    with pytest.raises(OSError):  # Cleanup refuses the guest's replacement symlink.
        route.run_container(("127.0.0.1", 3425), home, 1, ["unused", "command"])
    assert called == [True]
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("mode", ["one_shot", "session", "figure_command"])
def test_dispatch_starts_private_route_and_cleans_it_after_nonzero_child(
    tmp_path: Path, mode: str,
) -> None:
    trusted = tmp_path / "trusted"
    home = tmp_path / "guest-home"
    workspace = tmp_path / "workspace"
    binaries = tmp_path / "bin"
    for directory in (home, workspace, binaries):
        directory.mkdir()
    port = free_port()
    configuration(trusted / ".codex", port)
    # A guest cannot redirect the host-side bridge by changing its own profile.
    configuration(home / ".codex", free_port())
    fake_headless = binaries / "headless"
    fake_headless.write_text(
        f"#!{sys.executable}\n"
        "import os,socket,json,sys\n"
        "s=socket.socket()\n"
        "try:s.bind(('127.0.0.1',int(os.environ['EXPECTED_PORT'])))\n"
        "except OSError:bound=True\n"
        "else:bound=False\n"
        "s.close()\n"
        "assert bound, 'selected model listener did not start'\n"
        "print(json.dumps({'args':sys.argv[1:],'listener_ready':bound}))\n"
        "raise SystemExit(17)\n"
    )
    fake_headless.chmod(0o755)
    docker = binaries / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys,subprocess\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\n"
        "if a[0]=='inspect':print('true');raise SystemExit(0)\n"
        "if a[0]=='info':raise SystemExit(0)\n"
        "i=next(i for i,v in enumerate(a) if v.endswith('/relay.py'))\n"
        "p=Path(os.environ['GUEST_HOME'])/a[i].split('/home/agent-home/',1)[1]\n"
        "assert (p.parent/'provider.sock').is_socket()\n"
        "assert (p.parent.stat().st_mode & 0o777)==0o700\n"
        "assert all((x.stat().st_mode & 0o777)==0o600 for x in p.parent.iterdir())\n"
        "cmd=[sys.executable]+a[i:]\n"
        "cmd=[v.replace('/home/agent-home/',os.environ['GUEST_HOME']+'/').replace('/usr/local/bin/headless',os.environ['FAKE_HEADLESS']) for v in cmd]\n"
        "raise SystemExit(subprocess.call(cmd))\n"
    )
    docker.chmod(0o755)
    env = os.environ | {
        "PATH": f"{binaries}:{os.environ['PATH']}", "CONTAINER_AUTH_ROOT": str(trusted),
        "GUEST_HOME": str(home), "EXPECTED_PORT": str(port), "FAKE_HEADLESS": str(fake_headless),
        "AGENT_DOCKER_RUN_RETRIES": "1", "AGENT_DOCKER_DAEMON_RETRIES": "1",
        "AGENT_CONTAINER_MOUNT_DOCKER_SOCK": "0", "AGENT_CONTAINER_GPUS": "",
        "AGENT_PROFILE": "chosen",
    }
    for key in ("AGENTIC_CONTAINER_TEMP_ROOT", "AGENT_CONTAINER_MODEL_LOOPBACK", "AGENT_CONTAINER_SESSION_NAME"):
        env.pop(key, None)
    q = shlex.quote
    if mode == "figure_command":
        invocation = f"run_command_prompt codex {q(str(fake_headless))} {q(str(workspace))} 1 test-image bridge {q(str(home))}"
    else:
        suffix = " '' '' reuse-session" if mode == "session" else ""
        invocation = f"run_agent_prompt codex prompt test-model chosen {q(str(workspace))} 1 test-image bridge {q(str(home))}{suffix}"
    result = subprocess.run(
        ["bash", "-c", f"source {q(str(HELPER))}\n{invocation}"],
        capture_output=True, text=True, env=env, timeout=15,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    assert '"listener_ready": true' in result.stdout
    assert "secret" not in result.stdout + result.stderr
    assert not list(home.glob(".model-loopback-*"))


@pytest.mark.parametrize("network", ["host", "none", "container:other"])
def test_other_network_modes_do_not_gain_a_model_route(tmp_path: Path, network: str) -> None:
    trusted = tmp_path / "trusted"
    home = tmp_path / "home"
    home.mkdir()
    configuration(trusted / ".codex", free_port())
    q = shlex.quote
    result = subprocess.run(
        ["bash", "-c", f"source {q(str(HELPER))}\nrun_container_agent_command codex chosen {q(network)} {q(str(home))} 1 bash -c 'exit 19'"],
        env=os.environ | {"CONTAINER_AUTH_ROOT": str(trusted)},
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 19
    assert list(home.iterdir()) == []
