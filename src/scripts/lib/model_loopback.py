"""Route a selected host-local Codex provider through a private shared socket.

The provider identity comes from the host auth root, never the writable guest
home. Requests remain byte streams: this helper does not parse or log prompts,
authorization headers, responses, or provider credentials.
"""

from __future__ import annotations

import argparse
import errno
import ipaddress
import json
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import tomllib
from urllib.parse import urlsplit

try:
    from . import stage_codex_profile as config_io
except ImportError:  # Direct execution from the shell wrapper.
    import stage_codex_profile
    config_io = stage_codex_profile


def read_config(path: Path) -> dict:
    try:
        content = config_io.read_regular_file(
            path, config_io.MAX_PROFILE_BYTES, "Codex endpoint configuration",
        )
    except FileNotFoundError:
        return {}
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        raise ValueError("invalid Codex endpoint configuration") from None


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def selected_endpoint(source_home: Path, profile: str) -> tuple[str, int] | None:
    """Resolve only the selected provider in the trusted host configuration."""
    if profile and not config_io.PROFILE_NAME.fullmatch(profile):
        raise ValueError("invalid Codex profile name")
    base = read_config(source_home / "config.toml")
    override = read_config(source_home / f"{profile}.config.toml") if profile else {}
    provider = override.get("model_provider", base.get("model_provider"))
    if not isinstance(provider, str):
        return None
    details: dict = {}
    for config in (base, override):
        providers = config.get("model_providers", {})
        item = providers.get(provider, {}) if isinstance(providers, dict) else {}
        if isinstance(item, dict):
            details.update(item)
    raw_url = details.get("base_url")
    if not isinstance(raw_url, str):
        return None
    try:
        url = urlsplit(raw_url)
        host = url.hostname
        port = url.port if url.port is not None else {"http": 80, "https": 443}.get(url.scheme)
    except ValueError:
        raise ValueError("invalid selected model endpoint") from None
    if url.scheme not in {"http", "https"} or not host or not port or not is_loopback(host):
        return None
    return host, port


def copy_stream(left: socket.socket, right: socket.socket) -> None:
    peers = {left: right, right: left}
    readers = [left, right]
    while readers:
        ready, _, _ = select.select(readers, [], [], 1)
        for source in ready:
            data = source.recv(65536)
            if data:
                peers[source].sendall(data)
            else:
                readers.remove(source)
                try:
                    peers[source].shutdown(socket.SHUT_WR)
                except OSError:
                    pass


class ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, (UnixForwarder, TcpForwarder))
        try:
            if isinstance(server.target, str):
                upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                try:
                    upstream.settimeout(5)
                    upstream.connect(server.target)
                except BaseException:
                    upstream.close()
                    raise
            else:
                upstream = socket.create_connection(server.target, timeout=5)
            with upstream:
                upstream.settimeout(None)
                copy_stream(self.request, upstream)
        except OSError:
            # A failed connection is visible to the original model client.
            # Its existing retry policy can reconnect to a recovered backend.
            pass


class UnixForwarder(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, path: Path, target: tuple[str, int], *, directory_fd: int | None = None):
        self.target = target
        # Linux's Unix-socket path limit is shorter than valid workspace paths.
        # Anchor the directory before changing cwd: a running guest can rename
        # entries in its shared home. No serving threads exist yet.
        previous = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
        parent = None
        try:
            parent = (
                os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                if directory_fd is None else os.dup(directory_fd)
            )
            os.fchdir(parent)
            super().__init__(path.name, ForwardHandler)
            try:
                os.chmod(path.name, 0o600, follow_symlinks=False)
            except BaseException:
                self.server_close()
                raise
        finally:
            os.fchdir(previous)
            os.close(previous)
            if parent is not None:
                os.close(parent)


class TcpForwarder(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str, port: int, target: str):
        self.target = target
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        super().__init__((host, port), ForwardHandler)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        super().server_bind()


def serve(server: socketserver.BaseServer) -> None:
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()


def run_child(command: list[str]) -> int:
    child = subprocess.Popen(command)
    old_handlers = {}

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, forward)
        result = child.wait()
        return result if result >= 0 else 128 - result
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def client(host: str, port: int, unix_path: str, command: list[str]) -> int:
    if not is_loopback(host):
        raise ValueError("model relay may bind only a loopback address")
    servers: list[TcpForwarder] = []
    addresses = ["127.0.0.1", "::1"] if host.lower() == "localhost" else [host]
    try:
        for address in addresses:
            try:
                server = TcpForwarder(address, port, unix_path)
            except OSError as exc:
                if address == "::1" and host.lower() == "localhost" and exc.errno in {
                    errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL,
                }:
                    continue
                raise
            servers.append(server)
            serve(server)
        return run_child(command)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def run_container(
    endpoint: tuple[str, int], home: Path, inner_count: int, command: list[str],
) -> int:
    if not 0 < inner_count < len(command):
        raise ValueError("invalid container command boundary")
    home = home.resolve(strict=True)
    leaf = Path(tempfile.mkdtemp(prefix=".model-loopback-", dir=home))
    server = None
    directory_fd = None
    try:
        directory_fd = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        # The guest may already be running. Do not follow replaced directories
        # or helper files when writing into its shared home.
        for name, source in (
            ("relay.py", Path(__file__)),
            ("stage_codex_profile.py", Path(__file__).with_name("stage_codex_profile.py")),
        ):
            descriptor = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(source.read_bytes())
        sock_path = leaf / "provider.sock"
        server = UnixForwarder(sock_path, endpoint, directory_fd=directory_fd)
        serve(server)
        guest_root = f"/home/agent-home/{leaf.name}"
        wrapped = command[:-inner_count] + [
            os.environ.get("AGENT_CONTAINER_WRAPPER_PYTHON", "python"),
            guest_root + "/relay.py", "client", "--host", endpoint[0],
            "--port", str(endpoint[1]), "--socket", guest_root + "/provider.sock",
            "--", *command[-inner_count:],
        ]
        return run_child(wrapped)
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if directory_fd is not None:
            os.close(directory_fd)
        if shutil.rmtree.avoids_symlink_attacks:
            shutil.rmtree(leaf)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    endpoint_parser = commands.add_parser("endpoint")
    endpoint_parser.add_argument("--source-home", type=Path, required=True)
    endpoint_parser.add_argument("--profile", default="")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--endpoint", required=True)
    run_parser.add_argument("--home", type=Path, required=True)
    run_parser.add_argument("--inner-count", type=int, required=True)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    client_parser = commands.add_parser("client")
    client_parser.add_argument("--host", required=True)
    client_parser.add_argument("--port", type=int, required=True)
    client_parser.add_argument("--socket", required=True)
    client_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "endpoint":
            endpoint = selected_endpoint(args.source_home, args.profile)
            if endpoint:
                print(json.dumps(endpoint))
            return 0
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            raise ValueError("missing model command")
        if args.action == "client":
            return client(args.host, args.port, args.socket, command)
        endpoint = json.loads(args.endpoint)
        if not (
            isinstance(endpoint, list) and len(endpoint) == 2
            and isinstance(endpoint[0], str) and is_loopback(endpoint[0])
            and type(endpoint[1]) is int and 0 < endpoint[1] < 65536
        ):
            raise ValueError("invalid model loopback route")
        return run_container((endpoint[0], endpoint[1]), args.home, args.inner_count, command)
    except (OSError, ValueError):
        # Exception text can include credentials from malformed URL/config input.
        print("[container_agent] could not establish the selected model loopback route", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
