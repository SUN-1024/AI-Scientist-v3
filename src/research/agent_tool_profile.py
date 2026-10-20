"""Private per-invocation Codex profiles shared by bounded research tools."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tomllib
from typing import Any, Mapping

from src.scripts.lib.stage_codex_profile import PROFILE_NAME, stage_profile


def _regular(path: Path, maximum: int) -> bytes:
    path = path.expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Agent-tool input path contains a symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Agent-tool input must be a regular file")
        body = stream.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("Agent-tool input exceeds its size bound")
    return body


def _private(path: Path, body: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return repr(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return "{" + ", ".join(json.dumps(key) + " = " + _toml_value(item)
                                for key, item in value.items()) + "}"
    raise ValueError("Unsupported or non-finite Codex profile value")


@dataclass
class PrivateToolProfile:
    environment: dict[str, str]
    path: Path
    source_sha256: str
    expected_sha256: str

    def verify(self) -> str:
        if hashlib.sha256(_regular(self.path, 1024 * 1024)).hexdigest() != self.expected_sha256:
            raise ValueError("Private agent-tool profile changed during the call")
        return self.expected_sha256

    def add_server(self, name: str, configuration: dict[str, Any]) -> None:
        self.verify()
        config = tomllib.loads(_regular(self.path, 1024 * 1024).decode())
        servers = config.setdefault("mcp_servers", {})
        if name in servers:
            raise ValueError(f"The selected profile already defines {name}")
        servers[name] = configuration
        text = "\n".join(json.dumps(key) + " = " + _toml_value(value)
                           for key, value in config.items()) + "\n"
        if tomllib.loads(text) != config:
            raise ValueError("Private Codex profile values did not survive serialization")
        fd = os.open(self.path, os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW)
        with os.fdopen(fd, "wb") as stream:
            stream.write(text.encode())
        self.expected_sha256 = hashlib.sha256(text.encode()).hexdigest()


def prepare_tool_profile(*, temporary: Path, environment: Mapping[str, str], agent: str,
                         model: str | None, profile: str | None) -> PrivateToolProfile:
    if agent != "codex" or not model or not profile or not PROFILE_NAME.fullmatch(profile):
        raise ValueError("Research tools require an explicit Codex model and profile")
    original_home = Path(environment.get("CODEX_HOME") or str(Path(
        environment.get("HOME", str(Path.home()))) / ".codex")).expanduser().absolute()
    original_profile = original_home / (profile + ".config.toml")
    original_bytes = _regular(original_profile, 1024 * 1024)
    tomllib.loads(original_bytes.decode())
    private_home = temporary / "agent-tool-home"
    codex_home = private_home / ".codex"
    private_home.mkdir(mode=0o700)
    codex_home.mkdir(mode=0o700)
    stage_profile(original_home, codex_home, profile, catalog_destination_home=codex_home)
    if _regular(original_profile, 1024 * 1024) != original_bytes:
        raise ValueError("Selected profile changed while staging research tools")
    auth = original_home / "auth.json"
    if auth.exists() or auth.is_symlink():
        _private(codex_home / "auth.json", _regular(auth, 1024 * 1024))
    staged = codex_home / (profile + ".config.toml")
    return PrivateToolProfile(
        {**environment, "HOME": str(private_home), "CODEX_HOME": str(codex_home)}, staged,
        hashlib.sha256(original_bytes).hexdigest(), hashlib.sha256(_regular(staged, 1024 * 1024)).hexdigest(),
    )
