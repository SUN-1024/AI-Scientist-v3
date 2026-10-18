"""Manage a dedicated Docker volume without weakening filesystem checks."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path


def directory(path: Path) -> Path:
    path.mkdir(mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError(f"native state directory has an unsafe owner or type: {path}")
    return path


def initialize(root: Path, seed: Path, profile: str) -> Path:
    for name in ("sessions", "tmp", "workspaces"):
        directory(root / name)
    session = Path(tempfile.mkdtemp(prefix="session-", dir=root / "sessions"))
    home = directory(session / "home")
    codex = directory(home / ".codex")
    client = directory(session / "docker-client")
    (client / "config.json").write_text('{"auths":{}}\n', encoding="utf-8")
    for name in ("config.toml", f"{profile}.config.toml"):
        source = seed / ".codex" / name
        if source.is_file():
            shutil.copyfile(source, codex / name)
            (codex / name).chmod(0o600)
    return session


def probe(root: Path, image: str) -> dict[str, object]:
    """Prove native rename, permissions, and daemon-visible sibling bind paths."""
    with tempfile.TemporaryDirectory(prefix="filesystem-probe-", dir=root / "tmp") as temporary:
        path = Path(temporary)
        first, second = path / "first", path / "second"
        first.mkdir()
        second.mkdir()
        (first / "sentinel").write_text("first", encoding="utf-8")
        (second / "sentinel").write_text("second", encoding="utf-8")
        descriptor = os.open(path, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            exchange = libc.renameat2
            exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            exchange.restype = ctypes.c_int
            if exchange(descriptor, b"first", descriptor, b"second", 2):
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
        finally:
            os.close(descriptor)
        if (first / "sentinel").read_text() != "second" or (second / "sentinel").read_text() != "first":
            raise ValueError("atomic directory exchange did not preserve both trees")
        sentinel = path / "sibling-sentinel"
        sentinel.write_text("same daemon path", encoding="utf-8")
        expected = sentinel.stat()
        code = "import json,os,sys; p=sys.argv[1]; s=os.stat(p); print(json.dumps({'inode':s.st_ino,'content':open(p).read()}))"
        result = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
             "--mount", f"type=bind,source={root},target={root},readonly", "--entrypoint", "python",
             image, "-c", code, str(sentinel)],
            capture_output=True, text=True, timeout=45, check=True,
        )
        observed = json.loads(result.stdout)
        if observed != {"inode": expected.st_ino, "content": "same daemon path"}:
            raise ValueError("sibling container did not see the same sentinel inode and content")
        sentinel.chmod(0)
        try:
            try:
                sentinel.read_bytes()
            except PermissionError:
                pass
            else:
                raise ValueError("native state does not enforce unreadable file permissions")
        finally:
            sentinel.chmod(0o600)
        return {"atomic_exchange": True, "permissions": True, "sibling_bind": True,
                "sibling_inode": expected.st_ino, "root": str(root)}


def export_workspace(root: Path, relative: str, destination: Path) -> None:
    selection = Path(relative)
    if selection.is_absolute() or not selection.parts or any(part in {".", ".."} for part in selection.parts):
        raise ValueError("workspace must be a relative path below workspaces/")
    source = root / "workspaces"
    for part in selection.parts:
        source /= part
        if source.is_symlink() or not source.is_dir():
            raise ValueError("workspace source components must be real directories")
    if any(destination.iterdir()):
        raise ValueError("export destination must be empty")
    omitted = [".venv", ".agent_home", ".codex", ".claude", ".gemini", ".opencode",
               ".env", ".env.*", "auth.json", "__pycache__", ".pytest_cache"]
    shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns(*omitted))
    (destination / "runner-export.json").write_text(json.dumps({
        "source": str(source), "exported_at_unix": time.time(),
        "omitted_runtime_directories": omitted, "symlinks": "preserved without following",
        "purpose": "read-only inspection snapshot; resume only from the native workspace",
        "sealed_receipts_and_native_path_references": "copied without rewriting",
    }, indent=2) + "\n", encoding="utf-8")
    for directory_path, _, files in os.walk(destination, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory_path) / name
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                path.chmod(stat.S_IMODE(metadata.st_mode) & ~0o222)
        path = Path(directory_path)
        path.chmod(stat.S_IMODE(path.lstat().st_mode) & ~0o222)


def validate_workspace(root: Path, arguments: list[str]) -> None:
    if arguments[:2] != ["research", "harness"] or "--help" in arguments or "-h" in arguments:
        return
    targets = []
    for index, argument in enumerate(arguments):
        if argument in {"--workspace", "--resume"} and index + 1 < len(arguments):
            targets.append(arguments[index + 1])
        elif argument.startswith(("--workspace=", "--resume=")):
            targets.append(argument.split("=", 1)[1])
    allowed = root / "workspaces"
    if not targets:
        raise ValueError(f"native Harness requires --workspace or --resume below {allowed}")
    for target in targets:
        path = Path(target).expanduser()
        if not path.is_absolute() or path.resolve() == allowed or not path.resolve().is_relative_to(allowed):
            raise ValueError(f"workspace must be below selected native volume: {allowed}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("root", type=Path)
    init.add_argument("seed", type=Path)
    init.add_argument("profile")
    info = subparsers.add_parser("info")
    info.add_argument("root", type=Path)
    info.add_argument("volume")
    info.add_argument("source")
    check = subparsers.add_parser("probe")
    check.add_argument("root", type=Path)
    check.add_argument("image")
    export = subparsers.add_parser("export")
    export.add_argument("root", type=Path)
    export.add_argument("workspace")
    export.add_argument("destination", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("root", type=Path)
    validate.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "init":
        print(initialize(args.root, args.seed, args.profile))
    elif args.command == "info":
        print(json.dumps({"volume": args.volume, "native_root": str(args.root),
                          "workspaces": str(args.root / "workspaces"),
                          "temp": str(args.root / "tmp"), "source_checkout": args.source}, indent=2))
    elif args.command == "probe":
        print(json.dumps(probe(args.root, args.image), indent=2))
    elif args.command == "export":
        export_workspace(args.root, args.workspace, args.destination)
    else:
        validate_workspace(args.root, args.arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
