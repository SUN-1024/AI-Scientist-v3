"""The variants wrapper must seal its separate task-source mount itself."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.scripts.lib.variant_task_shadow import prepare_variant_task_shadow
from tests.task_drafting.test_task_contract import _write_scaffold

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "src/scripts/variant_agent.sh"


def _case(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, private, work = (tmp_path / name for name in ("task", "private", "work"))
    _write_scaffold(source)
    private.mkdir(mode=0o700)
    work.mkdir()
    return source, private, work


def test_shadow_omits_final_bytes_and_keeps_public_search_inputs(tmp_path: Path) -> None:
    source, private, work = _case(tmp_path)
    (source / "manual_validation.json").write_text('{"sealed_result":123}')
    values = prepare_variant_task_shadow(source, private, work)
    view = Path(values[0])
    for name in ("evaluate_final.py", "task_contract.json", "manual_validation.json"):
        assert (view / name).read_bytes() == b""
    assert list((view / "resources/final").iterdir()) == []
    assert (view / "resources/search/dev.json").read_bytes() == (
        source / "resources/search/dev.json"
    ).read_bytes()
    assert (source / "manual_validation.json").read_text() == '{"sealed_result":123}'


def test_shadow_rejects_public_hardlink_to_sealed_data(tmp_path: Path) -> None:
    source, private, work = _case(tmp_path)
    os.link(source / "resources/final/test.json", source / "public-copy.json")
    with pytest.raises(ValueError, match="aliases sealed content"):
        prepare_variant_task_shadow(source, private, work)


def test_shadow_rejects_private_storage_inside_agent_workspace(tmp_path: Path) -> None:
    source, _, work = _case(tmp_path)
    private = work / "private"
    private.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="outside agent mounts"):
        prepare_variant_task_shadow(source, private, work)


@pytest.mark.parametrize("step", ["variant_edit", "variant_verify"])
@pytest.mark.parametrize("posthoc", [False, True])
def test_wrapper_seals_without_inherited_controls(
    tmp_path: Path, step: str, posthoc: bool,
) -> None:
    source, _, work = _case(tmp_path)
    runtime = source / ".venv/bin"
    runtime.mkdir(parents=True)
    (runtime / "python").write_text("managed runtime binary placeholder")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "docker.json"
    docker = fake_bin / "docker"
    docker.write_text(
        f"#!{sys.executable}\nimport pathlib,json,sys\nargs=sys.argv[1:]\n"
        "if args and args[0]=='run':\n"
        " mounts=[args[i+1] for i,x in enumerate(args[:-1]) if x=='-v']\n"
        " task=pathlib.Path(next(x.rsplit(':',2)[0] for x in mounts if x.endswith(':/task-source:ro')))\n"
        " data={'args':args,'final_files':list((task/'resources/final').iterdir()),"
        "'contract_bytes':(task/'task_contract.json').stat().st_size,"
        "'final_evaluator_bytes':(task/'evaluate_final.py').stat().st_size,"
        "'search_exists':(task/'resources/search/dev.json').is_file()}\n"
        f" pathlib.Path({str(captured)!r}).write_text(json.dumps(data,default=str))\n"
    )
    docker.chmod(0o755)
    headless = fake_bin / "headless"
    headless.write_text("#!/bin/sh\nexit 0\n")
    headless.chmod(0o755)
    auth = tmp_path / "auth"
    auth.mkdir()
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    command = [
        "bash", str(WRAPPER), "--agent", "opencode", "--work-dir", str(work),
        "--prompt", "test only", "--containerize", "--container-auth-root", str(auth),
        "--task-source-dir", str(source), "--skill-step", step,
    ]
    if posthoc:
        command.append("--posthoc-env-policy")
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}", "WRAPPER_PYTHON": sys.executable,
        "HEADLESS_BIN": str(headless), "TMPDIR": str(tmp_path),
        "AGENT_CHATGPT_SUBSCRIPTION": "0", "AGENT_TASK_SEALED_FILES": "",
        "AGENT_TASK_SEALED_DIRS": "", "AGENT_CONTAINER_MOUNT_DOCKER_SOCK": "1",
        "AGENT_CONTAINER_EXTRA_MOUNTS": f"{source}:/leaked-task:ro",
        "AGENT_CONTAINER_SESSION_NAME": "unsafe-reused-session",
        "AGENT_VARIANT_AUDIT_ROOT": str(audit),
    }
    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
    snapshot = json.loads(captured.read_text())
    assert snapshot["final_files"] == []
    assert snapshot["contract_bytes"] == snapshot["final_evaluator_bytes"] == 0
    assert snapshot["search_exists"] is True
    args = snapshot["args"]
    assert f"{source}:/task-source:ro" not in args
    assert f"{source}:/leaked-task:ro" not in args
    assert "/var/run/docker.sock:/var/run/docker.sock" not in args
    assert f"{source.resolve()}/.venv:/task-source/.venv:ro" in args
    assert "TASK_PYTHON=/task-source/.venv/bin/python" in args
    assert len(list(audit.glob(f"*_{step}.jsonl"))) == 1


def test_wrapper_refuses_host_execution_for_contracted_source(tmp_path: Path) -> None:
    source, _, work = _case(tmp_path)
    run = subprocess.run(
        ["bash", str(WRAPPER), "--work-dir", str(work), "--prompt", "test",
         "--task-source-dir", str(source)],
        cwd=ROOT, env=os.environ | {"WRAPPER_PYTHON": sys.executable, "TMPDIR": str(tmp_path)},
        capture_output=True, text=True, timeout=10,
    )
    assert run.returncode != 0
    assert "require container isolation" in run.stderr
