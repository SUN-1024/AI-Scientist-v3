"""Broker aliases retain private storage while exposing only daemon bind paths."""
import os
from pathlib import Path
import socket
import tempfile

import pytest

from src.final_evaluation import broker_storage as storage
from src.final_evaluation import runner


@pytest.fixture
def private_root(monkeypatch):
    # Use a short real directory, including on macOS where /tmp is a symlink.
    with tempfile.TemporaryDirectory(prefix="ab-", dir="/tmp") as temporary:
        root = Path(temporary).resolve() / "b"
        monkeypatch.setenv("AGENTIC_BROKER_ROOT", str(root))
        monkeypatch.setenv("AGENTIC_BROKER_HOST_ROOT", str(root))
        yield root


def test_private_override_preserves_short_sockets_and_recovery(private_root):
    assert storage.broker_root() == private_root
    assert storage.ensure_broker_root(private_root) == private_root
    assert private_root.stat().st_mode & 0o777 == 0o700
    directory = private_root / ("draft-search-" + "a" * 32)
    directory.mkdir(mode=0o700)
    address = directory / "candidate.sock"
    assert len(os.fsencode(address)) <= 103
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(address))
        assert storage.broker_host_path(directory) == directory
    address.unlink()
    with runner._EvaluatorLaunchWitness(directory) as witness:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.connect(str(witness.path))
            client.sendall(witness._TOKEN)
        assert witness.observed()


@pytest.mark.parametrize("local,host", [
    ("/short/b", None), (None, "/host/b"), ("", "/host/b"),
    ("relative/b", "/host/b"), ("/short/../b", "/host/b"),
    ("/short/./b", "/host/b"), ("/short/b", "/host/b,readonly"),
    ("/short/b", "/host/other"), ("/" + "x" * 39, "/host/" + "x" * 39),
    ("/short//b", "/host/b"), ("/short/b", "/host/b\n"),
])
def test_override_rejects_partial_unsafe_or_long_roots(monkeypatch, local, host):
    for name, value in (("AGENTIC_BROKER_ROOT", local), ("AGENTIC_BROKER_HOST_ROOT", host)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="broker"):
        storage.broker_root()


def test_override_rejects_unrelated_aliases_and_unsafe_parents(private_root, monkeypatch):
    with tempfile.TemporaryDirectory(prefix="ah-", dir="/tmp") as temporary:
        host = Path(temporary).resolve() / "b"
        monkeypatch.setenv("AGENTIC_BROKER_HOST_ROOT", str(host))
        with pytest.raises(ValueError, match="same directory"):
            storage.ensure_broker_root(private_root)
        assert not private_root.exists()
    monkeypatch.setenv("AGENTIC_BROKER_HOST_ROOT", str(private_root))
    private_root.parent.chmod(0o755)
    try:
        with pytest.raises(ValueError, match="unsafe broker"):
            storage.ensure_broker_root(private_root)
    finally:
        private_root.parent.chmod(0o700)


def test_bind_translation_rejects_escape_and_symlink(private_root):
    storage.ensure_broker_root(private_root)
    for path in (private_root.parent / "outside", private_root / ".." / "outside"):
        with pytest.raises(ValueError, match="escapes"):
            storage.broker_host_path(path)
    (private_root / "escape").symlink_to(private_root.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe broker bind"):
        storage.broker_host_path(private_root / "escape" / "mask")


def test_alias_migration_refuses_existing_tmp_recovery(private_root, monkeypatch):
    old = private_root.parent / "old"
    old.mkdir(mode=0o700)
    monkeypatch.setattr(storage, "_default_broker_root", lambda: old)
    monkeypatch.setattr(storage, "legacy_broker_root", lambda: private_root.parent / "missing")
    directory = old / "final-example"
    directory.mkdir(mode=0o700)
    (directory / "candidate.cid").write_text("a" * 64)
    with pytest.raises(ValueError, match="legacy broker recovery"):
        storage.reject_legacy_broker_directory(private_root / directory.name)
    assert (directory / "candidate.cid").read_text() == "a" * 64


def test_default_paths_remain_untranslated(monkeypatch):
    monkeypatch.delenv("AGENTIC_BROKER_ROOT", raising=False)
    monkeypatch.delenv("AGENTIC_BROKER_HOST_ROOT", raising=False)
    assert storage.broker_root() == Path(f"/tmp/agentic-scientist-broker-{os.getuid()}")
    path = Path("/test/broker/scoring-mask")
    assert storage.broker_host_path(path) == path


def test_both_container_commands_translate_mounts_but_keep_cidfiles_local(tmp_path, monkeypatch):
    from types import SimpleNamespace

    local = Path("/agsci-state/broker-501")
    host = Path("/var/lib/docker/volumes/test-state/_data/broker-501")
    broker = local / "final-example"
    calls = []

    def translate(path):
        calls.append(path)
        return host / path.relative_to(local)

    monkeypatch.setattr(runner, "broker_host_path", translate)
    task, program, results = tmp_path / "task", tmp_path / "program", tmp_path / "results"
    task.mkdir()
    program.mkdir()
    scoring = task / ".venv/lib/python3.12/site-packages/src/investigation"
    scoring.mkdir(parents=True)
    evaluator = runner._evaluation_command(
        task_dir=task, program_dir=program, entrypoint=program / "program.py",
        evaluator=task / "evaluate.py", results_dir=results, python_bin=Path("/python"),
        container_image="test-image", gpu_count=0, environment={}, broker_dir=broker,
    )
    contract = SimpleNamespace(compute=SimpleNamespace(final=SimpleNamespace(
        cpu_cores=1, memory_gb=1, candidate_gpu_count=0,
    )))
    candidate = runner._candidate_command(
        task_dir=task, program_dir=program, entrypoint=program / "program.py",
        container_image="test-image", contract=contract, environment={}, broker_dir=broker,
    )
    assert calls == [broker, broker / "scoring-mask"]
    assert f"type=bind,src={host}/final-example,dst=/broker" in evaluator
    assert evaluator[evaluator.index("--cidfile") + 1] == str(broker / "evaluator.cid")
    assert candidate[candidate.index("--cidfile") + 1] == str(broker / "candidate.cid")
    assert any(f"src={host}/final-example/scoring-mask," in arg for arg in candidate)
    assert not any("AGENTIC_BROKER_HOST_ROOT=" in arg for arg in evaluator + candidate)
