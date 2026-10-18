"""Probe the Linux orchestrator without invoking a model or training a task."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path


def main() -> int:
    # A script's directory replaces cwd on sys.path. Load the mounted checkout,
    # just as `python -m src.agsci.main` does for normal runner invocations.
    sys.path.insert(0, str(Path.cwd()))
    checks: list[dict[str, object]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    def command(name: str, argv: list[str], *, cwd: Path | None = None) -> bool:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=45, cwd=cwd)
            detail = (result.stdout or result.stderr).strip().splitlines()
            record(name, result.returncode == 0, detail[0][:300] if detail else f"exit {result.returncode}")
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            record(name, False, str(exc)[:300])
            return False

    record("Linux", platform.system() == "Linux", platform.platform())
    record("Python", sys.version_info[:2] == (3, 12), sys.version.split()[0])
    record("isolated Python", sys.prefix == "/opt/research-runner/venv", sys.executable)
    for module in ("hydra", "omegaconf", "numpy", "scipy", "pandas", "httpx", "pypdf", "pytest"):
        try:
            importlib.import_module(module)
            record(f"import {module}", True, "loaded")
        except ImportError as exc:
            record(f"import {module}", False, str(exc))
    command("locked dependencies", [sys.executable, "-m", "pip", "check"])
    record("lock export", True, hashlib.sha256(Path("/opt/research-runner/requirements.lock.txt").read_bytes()).hexdigest())
    command("Git checkout", ["git", "rev-parse", "HEAD"])
    command("Docker client/server", ["docker", "version", "--format", "{{.Client.Version}} / {{.Server.Version}}"])
    images = sys.argv[1:] or [os.environ.get("RESEARCH_CPU_IMAGE", "unified-ai-scientist/research-cpu:2026.10")]
    for image in images:
        command(f"image {image}", ["docker", "image", "inspect", image, "--format", "{{.Id}} {{.Os}}/{{.Architecture}}"])
    command("Node", ["node", "--version"])
    command("Codex", ["codex", "--version"])
    command("Headless", ["headless", "--version"])
    command("uv", ["uv", "--version"])
    native_root = os.environ.get("RESEARCH_RUNNER_NATIVE_ROOT")
    if not native_root:
        record("native state filesystem", False, "use --native-state for durable Harness state")
    else:
        try:
            from native_state import probe

            result = probe(Path(native_root), images[0])
            record("native state filesystem", True, json.dumps(result, sort_keys=True))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            record("native state filesystem", False, str(exc)[:300])
    try:
        from src.research.preflight import _probe_descriptor_pinning

        _probe_descriptor_pinning(Path.cwd())
        record("Harness descriptor preflight", True, "directory alias retains inode identity")
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        record("Harness descriptor preflight", False, str(exc))
    with tempfile.TemporaryDirectory(prefix="linux-runner-preflight-") as temporary:
        path = Path(temporary)
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            pinned = Path(f"/proc/self/fd/{descriptor}")
            (pinned / "probe.txt").write_text("pinned directory\n", encoding="utf-8")
            record("Linux directory pinning", (path / "probe.txt").read_text() == "pinned directory\n", "/proc/self/fd directory operations work")
        except OSError as exc:
            record("Linux directory pinning", False, str(exc))
        finally:
            os.close(descriptor)
        (path / "probe.tex").write_text("\\documentclass{article}\n\\begin{document}Harness runner.\\end{document}\n", encoding="utf-8")
        compiled = command("TeX compile", ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "probe.tex"], cwd=path)
        record("TeX PDF", compiled and (path / "probe.pdf").is_file(), "probe.pdf")
    try:
        codex_dir = Path(os.environ["CODEX_HOME"])
        profile = os.environ.get("RESEARCH_RUNNER_CODEX_PROFILE", "agsci-magpie")
        config = tomllib.loads((codex_dir / f"{profile}.config.toml").read_text())
        provider = config["model_providers"][config["model_provider"]]
        url = provider["base_url"].rstrip("/") + "/models"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=15) as response:
            payload = json.loads(response.read(2 * 1024 * 1024))
            record("provider model catalogue", response.status == 200 and isinstance(payload, (dict, list)), f"HTTP {response.status}; no model invocation")
    except (OSError, ValueError, KeyError) as exc:
        # Do not print provider configuration or response content.
        record("provider model catalogue", False, type(exc).__name__)
    ok = all(check["ok"] for check in checks)
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
