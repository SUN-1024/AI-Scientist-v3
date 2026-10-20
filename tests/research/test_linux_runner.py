"""The outer Linux runner preserves bind paths and limits host exposure."""
from __future__ import annotations

import json
import importlib.util
import os
import shutil
import socket
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[2] / "scripts/run-research-harness-container.sh"
REPO_ROOT = RUNNER.parent.parent


def native_helpers():
    location = RUNNER.parent.parent / "src/docker/research-runner/native_state.py"
    spec = importlib.util.spec_from_file_location("research_runner_native_state", location)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deepseek_profile_is_direct_and_keeps_the_key_out_of_files() -> None:
    provider_root = REPO_ROOT / "configs/research-providers"
    profile = tomllib.loads((provider_root / "agsci-deepseek.config.toml.example").read_text())
    provider = profile["model_providers"]["deepseek_direct"]
    assert profile["model"] == "deepseek-flash"
    assert profile["model_provider"] == "deepseek_direct"
    assert profile["model_catalog_json"] == "deepseek-models.json"
    assert provider == {
        "name": "DeepSeek direct",
        "base_url": "https://api.deepseek.com/",
        "wire_api": "responses",
        "env_key": "DEEPSEEK_API_KEY",
        "requires_openai_auth": False,
        "request_max_retries": 3,
        "stream_max_retries": 3,
    }
    assert "experimental_bearer_token" not in provider
    catalog = json.loads((provider_root / "deepseek-models.json").read_text())
    assert [row["slug"] for row in catalog["models"]] == ["deepseek-flash", "deepseek-v4-pro"]


def test_runner_keeps_paths_and_private_host_config(tmp_path: Path) -> None:
    root = tmp_path / "checkout with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(RUNNER, root / "scripts/runner")
    common = tmp_path / "parent/.git"
    common.mkdir(parents=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "docker-args.json"
    env_file = tmp_path / "outer.env"
    env_file.write_text("TEST_PROVIDER_KEY=placeholder\n")
    docker = commands / "docker"
    docker.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["TEST_DOCKER_LOG"]).write_text(json.dumps(sys.argv[1:]))\n')
    docker.chmod(0o755)
    git = commands / "git"
    git.write_text('#!/usr/bin/env python3\nimport os\nprint(os.environ["TEST_GIT_COMMON"])\n')
    git.chmod(0o755)
    # Keep the socket path short enough for macOS's Unix-socket path limit.
    with tempfile.TemporaryDirectory(prefix="runner-socket-", dir="/tmp") as socket_dir, socket.socket(socket.AF_UNIX) as daemon_socket:
        socket_path = Path(socket_dir) / "d.sock"
        daemon_socket.bind(str(socket_path))
        result = subprocess.run(
            ["bash", str(root / "scripts/runner"), "research", "harness", "case", "--dry-run"],
            capture_output=True, text=True,
            env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
                 "DOCKER_BIN": str(docker), "DOCKER_HOST": f"unix://{socket_path}",
                 "RESEARCH_RUNNER_STATE_DIR": str(root / "scratch/research-runner"),
                 "AGENTIC_TEMP_ROOT": str(root / "scratch/tmp"),
                 "RESEARCH_RUNNER_ENV_FILE": str(env_file),
                 "DEEPSEEK_API_KEY": "secret-not-logged",
                 "TEST_DOCKER_LOG": str(log), "TEST_GIT_COMMON": str(common)},
        )
    assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text())
    mounts = [args[index + 1] for index, value in enumerate(args) if value == "--mount"]
    assert f"type=bind,source={root},target={root}" in mounts
    assert f"type=bind,source={common},target={common},readonly" in mounts
    assert len(mounts) == 4  # checkout, Docker socket, broker storage, Git metadata
    broker_parent = root / "scratch/research-runner/broker-storage"
    assert f"type=bind,source={broker_parent},target=/agsci-state" in mounts
    assert f"AGENTIC_BROKER_ROOT=/agsci-state/broker-{os.getuid()}" in args
    assert f"AGENTIC_BROKER_HOST_ROOT={broker_parent}/broker-{os.getuid()}" in args
    config_path = Path(args[args.index("--config") + 1])
    assert json.loads((config_path / "config.json").read_text()) == {"auths": {}}
    assert config_path.is_relative_to(root / "scratch/research-runner")
    assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert args[-4:] == ["research", "harness", "case", "--dry-run"]
    assert "HTTP_PROXY=" in args
    assert "DOCKER_HOST=unix:///var/run/docker.sock" in args
    assert "USER=agent" in args and "LOGNAME=agent" in args
    assert args[args.index("DEEPSEEK_API_KEY") - 1] == "--env"
    assert "secret-not-logged" not in args
    assert args[args.index("--env-file") + 1] == str(env_file)
    assert "TEST_PROVIDER_KEY=placeholder" not in args


def test_runner_rejects_remote_daemon_before_making_state(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(RUNNER, root / "scripts/runner")
    result = subprocess.run(
        ["bash", str(root / "scripts/runner"), "--help"], capture_output=True, text=True,
        env={**os.environ, "DOCKER_BIN": "/usr/bin/true", "DOCKER_HOST": "tcp://remote:2376"},
    )
    assert result.returncode == 2
    assert "requires a local Unix Docker socket" in result.stderr
    assert not (root / "scratch").exists()


def test_builder_exports_frozen_dependencies_without_host_environment_changes(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    build_script = RUNNER.with_name("build-research-runner-image.sh")
    shutil.copy2(build_script, root / "scripts/builder")
    shutil.copytree(RUNNER.parent.parent / "src/docker/research-runner", root / "src/docker/research-runner")
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "calls.jsonl"
    stub = '#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nwith Path(os.environ["TEST_COMMAND_LOG"]).open("a") as out: out.write(json.dumps([Path(sys.argv[0]).name,*sys.argv[1:]])+"\\n")\nif "--output-file" in sys.argv: Path(sys.argv[sys.argv.index("--output-file")+1]).write_text("example==1.0\\n")\n'
    for tool in ("uv", "docker"):
        path = commands / tool
        path.write_text(stub)
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(root / "scripts/builder")], capture_output=True, text=True,
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
             "DOCKER_BIN": str(commands / "docker"), "DOCKER_HOST": "unix:///test/docker.sock",
             "TEST_COMMAND_LOG": str(log)},
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    export = calls[0]
    assert export[:2] == ["uv", "export"]
    assert all(flag in export for flag in ("--frozen", "--all-groups", "--no-emit-project", "--no-header"))
    assert "build" in calls[1] and "image" in calls[2] and "inspect" in calls[2]
    assert "HTTP_PROXY=" in calls[1]
    build_context = Path(calls[1][-1])
    assert not build_context.exists()


def test_native_workspace_rejects_another_volume_or_symlink_escape(tmp_path: Path) -> None:
    native = native_helpers()
    root = tmp_path / "native"
    (root / "workspaces").mkdir(parents=True)
    native.validate_workspace(root, ["research", "harness", "task", "--workspace", str(root / "workspaces/study")])
    with pytest.raises(ValueError, match="requires --workspace"):
        native.validate_workspace(root, ["research", "harness", "task"])
    with pytest.raises(ValueError, match="selected native volume"):
        native.validate_workspace(root, ["research", "harness", "--resume", str(tmp_path / "other/study")])
    (root / "workspaces/escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="selected native volume"):
        native.validate_workspace(root, ["research", "harness", f"--workspace={root / 'workspaces/escape/study'}"])


def test_native_profile_copy_includes_its_model_catalog(tmp_path: Path) -> None:
    native = native_helpers()
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "config.toml").write_text('model_provider = "deepseek_direct"\n')
    (seed / "agsci-deepseek.config.toml").write_text(
        'model = "deepseek-flash"\n'
        'model_provider = "deepseek_direct"\n'
        'model_catalog_json = "deepseek-models.json"\n'
        '[model_providers.deepseek_direct]\n'
        'name = "DeepSeek direct"\n'
        'base_url = "https://api.deepseek.com/"\n'
        'env_key = "DEEPSEEK_API_KEY"\n'
    )
    catalog = '{"models":[{"slug":"deepseek-flash"}]}\n'
    (seed / "deepseek-models.json").write_text(catalog)
    destination = tmp_path / "destination"
    native.copy_selected_profile(seed, destination, "agsci-deepseek")
    assert (destination / "config.toml").read_text() == 'model_provider = "deepseek_direct"\n'
    assert (destination / "agsci-deepseek.config.toml").is_file()
    assert (destination / "deepseek-models.json").read_text() == catalog
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in destination.iterdir())


def test_native_profile_copy_rejects_catalog_outside_private_home(tmp_path: Path) -> None:
    native = native_helpers()
    seed = tmp_path / "seed"
    seed.mkdir()
    outside = tmp_path / "outside-models.json"
    outside.write_text('{"models":[]}\n')
    (seed / "agsci-deepseek.config.toml").write_text(
        f'model_catalog_json = "{outside}"\n'
    )
    with pytest.raises(ValueError, match="inside the profile home"):
        native.copy_selected_profile(seed, tmp_path / "destination", "agsci-deepseek")


def test_export_is_readonly_snapshot_without_credentials_or_receipt_rewrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native = native_helpers()
    root = tmp_path / "native"
    source = root / "workspaces/study"
    source.mkdir(parents=True)
    receipt = '{"source":"/var/lib/docker/volumes/native/_data/workspaces/study"}\n'
    (source / "receipt.json").write_text(receipt)
    (source / ".env").write_text("PRIVATE_KEY=not-for-export\n")
    (source / ".agent_home").mkdir()
    (source / ".agent_home/auth.json").write_text("private")
    (source / "external-link").symlink_to(tmp_path / "not-copied")
    os.mkfifo(source / ".agent_stream.fifo")
    (source / "fifo-link").symlink_to(".agent_stream.fifo")
    monkeypatch.chdir(source)
    with socket.socket(socket.AF_UNIX) as runtime_socket:
        runtime_socket.bind(".runtime.sock")
    destination = tmp_path / "snapshot"
    destination.mkdir()
    native.export_workspace(root, "study", destination)
    try:
        assert (destination / "receipt.json").read_text() == receipt
        assert not (destination / ".env").exists()
        assert not (destination / ".agent_home").exists()
        assert (destination / "external-link").is_symlink()
        assert (destination / "fifo-link").is_symlink()
        assert not (destination / ".agent_stream.fifo").exists()
        assert not (destination / ".runtime.sock").exists()
        assert json.loads((destination / "runner-export.json").read_text())["omitted_runtime_special_files"] == [
            {"path": ".agent_stream.fifo", "type": "fifo"},
            {"path": ".runtime.sock", "type": "socket"},
        ]
        assert (destination / "receipt.json").stat().st_mode & 0o222 == 0
        assert destination.stat().st_mode & 0o222 == 0
        with pytest.raises(ValueError, match="must be empty"):
            native.export_workspace(root, "study", destination)
        with pytest.raises(ValueError, match="relative path"):
            native.export_workspace(root, "../sessions", destination)
    finally:
        destination.chmod(0o700)
