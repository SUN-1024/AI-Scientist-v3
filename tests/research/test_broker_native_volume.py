"""Opt-in integration check using a disposable volume and two sibling containers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("AGSCI_TEST_NATIVE_BROKER") != "1",
    reason="set AGSCI_TEST_NATIVE_BROKER=1 to use a disposable local Docker volume",
)


def test_native_broker_socket_and_mask_are_visible_to_daemon():
    root = Path(__file__).resolve().parents[2]
    docker = os.environ.get("DOCKER_BIN") or shutil.which("docker")
    if not docker and Path("/Applications/Docker.app/Contents/Resources/bin/docker").exists():
        docker = "/Applications/Docker.app/Contents/Resources/bin/docker"
    assert docker, "local Docker is required for the requested integration test"
    suffix = uuid.uuid4().hex
    volume = f"agsci-broker-test-{suffix}"
    state = root / "scratch" / f"broker-test-{suffix}"
    code = r'''
import json, os, socket, stat, subprocess
from pathlib import Path
from src.final_evaluation.broker_storage import broker_root, ensure_broker_root, broker_host_path
from src.final_evaluation.runner import _EvaluatorLaunchWitness

root = ensure_broker_root(broker_root())
directory = root / ("draft-search-" + "a" * 32)
directory.mkdir(mode=0o700)
mask = directory / "scoring-mask"
mask.mkdir(mode=0o700)
(mask / "sentinel").write_text("private-mask")
address = directory / "candidate.sock"
with socket.socket(socket.AF_UNIX) as server, _EvaluatorLaunchWitness(directory) as witness:
    server.bind(str(address))
    address.chmod(0o600)
    server.listen(1)
    server.settimeout(5)
    host = broker_host_path(directory)
    host_mask = broker_host_path(mask)
    assert str(host).startswith(os.environ["RESEARCH_RUNNER_NATIVE_ROOT"] + "/")
    assert host.stat().st_ino == directory.stat().st_ino
    assert len(os.fsencode(address)) <= 103
    assert len(os.fsencode(host / "candidate.sock")) > 107
    child = """import json,os,socket
from pathlib import Path
assert Path('/mask/sentinel').read_text() == 'private-mask'
assert (Path('/broker').stat().st_mode & 0o777) == 0o700
assert (Path('/broker/candidate.sock').stat().st_mode & 0o777) == 0o600
with socket.socket(socket.AF_UNIX) as client:
    client.connect('/broker/candidate.sock')
    client.sendall(b'native-broker')
with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as witness:
    witness.connect('/broker/evaluator-start.sock')
    witness.sendall(b'evaluator-started-v1')
print(json.dumps({'uid': os.getuid(), 'mask': 'private-mask'}))
"""
    command = ["docker", "run", "--rm", "--network", "none", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--cidfile", str(directory / "evaluator.cid"),
               "--mount", f"type=bind,source={host},target=/broker",
               "--mount", f"type=bind,source={host_mask},target=/mask,readonly",
               "--entrypoint", "python", os.environ["RESEARCH_CPU_IMAGE"], "-c", child]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=45)
    assert json.loads(result.stdout) == {"uid": os.getuid(), "mask": "private-mask"}
    connection, _ = server.accept()
    with connection:
        assert connection.recv(32) == b"native-broker"
    assert witness.observed()
    assert len((directory / "evaluator.cid").read_text().strip()) == 64
    print(json.dumps({"socket": True, "launch_witness": True, "mask": True,
                      "cidfile": True, "local_root": str(root), "host_root": str(host.parent)}))
'''
    environment = dict(os.environ)
    environment.pop("RESEARCH_RUNNER_ENV_FILE", None)
    environment.update({
        "DOCKER_BIN": docker,
        "RESEARCH_RUNNER_VOLUME": volume,
        "RESEARCH_RUNNER_STATE_DIR": str(state),
        "RESEARCH_RUNNER_AUTH_ROOT": str(state / "no-auth"),
    })
    try:
        result = subprocess.run(
            ["bash", str(root / "scripts/run-research-harness-container.sh"),
             "--native-state", "--exec", "python", "-c", code],
            cwd=root, env=environment, capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stderr
        observed = json.loads(result.stdout)
        assert all(observed[name] for name in ("socket", "launch_witness", "mask", "cidfile"))
        assert observed["local_root"] == f"/agsci-state/broker-{os.getuid()}"
        assert volume in observed["host_root"]
    finally:
        # This random test volume never contains user study state.
        subprocess.run([docker, "volume", "rm", volume], capture_output=True, timeout=30)
        shutil.rmtree(state, ignore_errors=True)
