from __future__ import annotations

import json
import errno
import shutil
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agsci import doctor
from src.investigation.program_spec import render_default_program_spec_toml
from src.pipeline.cli import load_pipeline_config
from src.pipeline.descriptor_paths import DescriptorPathError
from src.research import cli, preflight
from src.research.preflight import _probe_transaction_filesystem
from tests.research.test_unified_research import make_task


@pytest.fixture(autouse=True)
def descriptor_pinning_available(monkeypatch):
    """Fake Host cases must not inherit the test runner's filesystem limits."""
    probe = Mock()
    monkeypatch.setattr(preflight, "_probe_descriptor_pinning", probe)
    monkeypatch.setattr(preflight, "_probe_transaction_filesystem", Mock())
    return probe


def runtime_host(tmp_path: Path, **changes) -> doctor.Host:
    return replace(doctor.Host(
        which=lambda name: f"/fixture/bin/{name}",
        run=Mock(side_effect=AssertionError("unexpected subprocess")),
        probe_docker=Mock(return_value=(True, "")),
        image_exists=Mock(return_value=(True, "")),
        environ={}, repo_root=tmp_path,
    ), **changes)


def complete_task(tmp_path: Path) -> Path:
    task = make_task(tmp_path / "task")
    (task / "capture.py").write_text("# Fixture capture\n")
    (task / "evaluate.py").write_text("# Fixture evaluator\n")
    (task / "program_0" / "src").mkdir(parents=True)
    (task / "program_0" / "src" / "program.py").write_text("# Fixture baseline\n")
    (task / "program_0" / "program_spec.toml").write_text(render_default_program_spec_toml())
    return task


def test_native_failure_stops_before_prepare_or_online_retrieval(tmp_path, monkeypatch, capsys):
    task = complete_task(tmp_path)
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path, which=lambda _name: None)
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    prepare = Mock(side_effect=AssertionError("must not prepare or spend on retrieval"))
    monkeypatch.setattr(cli, "prepare", prepare)
    assert cli.main([
        "run", str(task), "--workspace", str(workspace), "container.enabled=false",
        "execution.phases=investigation", "agent.name=codex",
    ]) == 2
    prepare.assert_not_called()
    host.probe_docker.assert_not_called()
    assert not workspace.exists()
    assert not (task / "research_context").exists()
    assert "codex" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["docker_cli", "daemon", "image", "auth_root", "env_file"])
def test_container_failures_are_actionable_before_prepare(tmp_path, monkeypatch, capsys, failure):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path)
    overrides = [f"container.auth_root={tmp_path}", "container.env_file=''"]
    expected = "docker"
    if failure == "docker_cli":
        host.which = lambda name: None if name == "docker" else f"/fixture/bin/{name}"
    elif failure == "daemon":
        host.probe_docker.return_value = (False, "daemon unavailable")
    elif failure == "image":
        host.image_exists.return_value = (False, "not installed")
        expected = "agent image"
    elif failure == "auth_root":
        overrides[0] = f"container.auth_root={tmp_path / 'missing_auth'}"
        expected = "auth root"
    else:
        overrides[1] = "container.env_file=missing.env"
        expected = "env file"
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    prepare = Mock(side_effect=AssertionError("must not prepare or retrieve"))
    monkeypatch.setattr(cli, "prepare", prepare)
    assert cli.main(["run", str(task), "--workspace", str(workspace), *overrides]) == 2
    prepare.assert_not_called()
    assert not workspace.exists()
    assert expected in capsys.readouterr().err
    host.run.assert_not_called()


def test_native_profile_uses_source_and_resume_keeps_execution_path(tmp_path, monkeypatch, capsys):
    task = complete_task(tmp_path)
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path)
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    pipeline = Mock(return_value=0)
    monkeypatch.setattr("src.pipeline.cli.main", pipeline)
    overrides = [
        "container.enabled=false", "execution.phases=investigation", "agent.name=codex",
        "agent.profile=local_magpie", "agent.model=custom/model",
        f"container.auth_root={tmp_path / 'no_standard_login'}",
    ]
    assert cli.main(["run", str(task), "--workspace", str(workspace), "--offline", *overrides]) == 0
    actual_args = pipeline.call_args.args[0]
    assert f"task.dir={json.dumps(str(workspace / 'input_task'))}" in actual_args
    assert "agent.profile=local_magpie" in actual_args
    assert "preflight warning: credentials:" in capsys.readouterr().err
    shutil.rmtree(task)
    monkeypatch.setattr(cli, "ensure_execution_ready", Mock(side_effect=AssertionError("no retrieval on resume")))
    assert cli.main(["run", "--resume", str(workspace)]) == 0
    assert pipeline.call_count == 2
    host.probe_docker.assert_not_called()
    host.image_exists.assert_not_called()
    host.run.assert_not_called()


@pytest.mark.parametrize("command,flags", [("run", ["--dry-run"]), ("prepare", ["--offline"])])
def test_prepare_and_dry_run_do_not_probe_execution_runtime(tmp_path, monkeypatch, command, flags):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    probe = Mock(side_effect=AssertionError("must not probe execution runtime"))
    monkeypatch.setattr(cli, "ensure_execution_ready", probe)
    assert cli.main([command, str(task), "--workspace", str(workspace), *flags]) == 0
    probe.assert_not_called()
    assert workspace.exists() is (command == "prepare")


def test_scaffold_failure_and_native_drafting_are_detected(tmp_path):
    task = make_task(tmp_path / "task")
    host = runtime_host(tmp_path)
    cfg = load_pipeline_config([
        "container.enabled=false", "execution.phases=investigation", "agent.name=codex",
    ])
    results = preflight.execution_checks(cfg, task, host=host)
    assert any(result.name == "task" and result.status == doctor.FAIL for result in results)
    cfg.execution.phases = "task_drafting,investigation"
    results = preflight.execution_checks(cfg, task, host=host)
    assert any(result.name == "task drafting" and result.status == doctor.FAIL for result in results)
    # A Docker drafting run can start from a brief; it need not have an evaluator yet.
    cfg.container.enabled = True
    cfg.container.auth_root = str(tmp_path)
    cfg.container.env_file = ""
    results = preflight.execution_checks(cfg, task, host=host)
    assert not any(result.status == doctor.FAIL for result in results)


def test_source_task_check_does_not_mutate_resolved_destination(tmp_path):
    task = complete_task(tmp_path)
    destination = tmp_path / "not_created" / "input_task"
    cfg = load_pipeline_config([
        f"task.dir={destination}", "container.enabled=false",
        "execution.phases=investigation", "agent.name=codex",
    ])
    results = preflight.execution_checks(cfg, task, host=runtime_host(tmp_path))
    assert not any(result.status == doctor.FAIL for result in results)
    assert cfg.task.dir == str(destination)
    assert not destination.exists()


@pytest.mark.parametrize("phases", ["ablations", "final_evaluation", "ablations,final_evaluation"])
def test_harness_downstream_resume_preflight_accepts_exact_saved_worker_pool(
    tmp_path, monkeypatch, phases,
):
    from tests.final_evaluation.test_final_evaluation import _write_contract_task, _write_investigation

    task = _write_contract_task(tmp_path)
    _write_investigation(tmp_path / "run")
    contract_path = task / "task_contract.json"
    contract = json.loads(contract_path.read_text())
    for compute in contract["compute"].values():
        compute.update(
            accelerator="gpu", accelerator_model="fixture", gpu_count=1,
            evaluator_gpu_count=1, candidate_gpu_count=0, precision="float32",
        )
    contract_path.write_text(json.dumps(contract))
    pool = '"device=GPU-aaaa,GPU-bbbb"'
    environment = {"AGENT_CONTAINER_GPUS": pool, "AGENTIC_TEMP_ROOT": str(tmp_path)}
    host = runtime_host(tmp_path, environ=environment)
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    cfg = load_pipeline_config([
        "preset=research_harness", f"task.dir={task}", f"execution.phases={phases}",
        "container.gpu_count=1", "investigation.execution.workers=2",
        f"container.auth_root={tmp_path}", "container.env_file=''", f"paths.run_dir={tmp_path / 'run'}",
    ])

    checks = preflight.execution_checks(cfg, task, host=host)
    assert not [check for check in checks if check.status == doctor.FAIL]
    assert next(check for check in checks if check.name == "required gpus").status == doctor.OK
    preflight.ensure_execution_ready(cfg, task)
    assert cfg.investigation.workers == 2
    assert host.environ == environment
    host.run.assert_not_called()


@pytest.mark.parametrize("phases,needs_gpu", [
    ("posthoc", False), ("paper", False), ("review", False), ("revision", False),
    ("posthoc,paper,review,revision", False),
    ("task_drafting", True), ("investigation", True), ("ablations", True), ("final_evaluation", True),
])
def test_saved_gpu_study_can_resume_writing_without_a_gpu(tmp_path, phases, needs_gpu):
    from tests.final_evaluation.test_final_evaluation import _write_contract_task, _write_investigation

    task = _write_contract_task(tmp_path)
    _write_investigation(tmp_path / "run")
    contract_path = task / "task_contract.json"
    contract = json.loads(contract_path.read_text())
    for compute in contract["compute"].values():
        compute.update(accelerator="gpu", accelerator_model="fixture", gpu_count=1,
                       evaluator_gpu_count=1, candidate_gpu_count=0, precision="float32")
    contract_path.write_text(json.dumps(contract))
    host = runtime_host(tmp_path, environ={"AGENTIC_TEMP_ROOT": str(tmp_path)},
                        which=lambda name: None if name == "nvidia-smi" else f"/fixture/bin/{name}")
    cfg = load_pipeline_config([
        "preset=research_harness", f"task.dir={task}", f"execution.phases={phases}",
        "container.gpu_count=1", f"container.auth_root={tmp_path}", "container.env_file=''",
        f"paths.run_dir={tmp_path / 'run'}",
    ])
    if phases in {"review", "revision"}:
        paper = tmp_path / "run/paper"
        paper.mkdir(parents=True)
        (paper / "paper.pdf").write_bytes(b"%PDF-1.4\n")
        review = paper / "ai_review.before_revision.json"
        review.write_text('{"Summary":"fixture review"}\n')
        (paper / "latest_manifest.json").write_text(json.dumps({
            "paper_pdf": str(paper / "paper.pdf"),
            "before_revision": {"status": "completed", "review_json": str(review)},
        }))
        cfg.paths.paper_dir = str(paper)
    assert doctor.check_required_gpus(host, cfg).status == doctor.FAIL
    results = preflight.execution_checks(cfg, task, host=host)
    gpu_check = next(check for check in results if check.name == "required gpus")
    assert gpu_check.status == (doctor.FAIL if needs_gpu else doctor.OK)
    if not needs_gpu:
        assert not [check for check in results if check.status == doctor.FAIL]
    host.run.assert_not_called()


def test_missing_task_brief_is_a_preflight_failure(tmp_path):
    task = complete_task(tmp_path)
    (task / "task.md").unlink()
    cfg = load_pipeline_config(["container.enabled=false", "execution.phases=investigation"])
    results = preflight.execution_checks(cfg, task, host=runtime_host(tmp_path))
    assert any(result.status == doctor.FAIL and "task.md" in result.detail for result in results)


@pytest.mark.parametrize("phases,versioning,required", [
    ("investigation", True, True),
    ("review", False, True),
    ("revision", False, True),
    ("investigation", False, False),
    ("task_drafting", False, False),
])
def test_descriptor_probe_is_scoped_to_checkpoint_or_recovery_operations(tmp_path, phases, versioning, required):
    task = complete_task(tmp_path)
    cfg = load_pipeline_config([f"execution.phases={phases}", f"versioning.enabled={str(versioning).lower()}"])
    probe = Mock(side_effect=DescriptorPathError("no safe directory descriptor path is available"))
    results = preflight.execution_checks(cfg, task, host=runtime_host(tmp_path), descriptor_probe=probe)
    checks = [result for result in results if result.name == "host descriptor pinning"]
    if required:
        probe.assert_called_once_with(task)
        assert len(checks) == 1 and checks[0].status == doctor.FAIL
        assert "scripts/run-research-harness-container.sh" in checks[0].hint
    else:
        probe.assert_not_called()
        assert checks == []


def test_unsupported_descriptor_pinning_stops_harness_before_literature(tmp_path, monkeypatch, capsys):
    from src.research import harness
    task = complete_task(tmp_path)
    workspace = tmp_path / "workspace"
    before = {path.relative_to(task): path.read_bytes() for path in task.rglob("*") if path.is_file()}
    host = runtime_host(tmp_path)
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    monkeypatch.setattr(preflight, "_probe_descriptor_pinning",
                        Mock(side_effect=DescriptorPathError("no safe directory descriptor path is available")))
    prepare = Mock(side_effect=AssertionError("must not prepare or spend literature/model credits"))
    monkeypatch.setattr(harness, "prepare", prepare)
    assert cli.main([
        "harness", str(task), "--workspace", str(workspace), "--quality-target", "development",
        "execution.phases=investigation", "container.enabled=false", "versioning.enabled=true",
    ]) == 2
    prepare.assert_not_called()
    assert not workspace.exists()
    assert before == {path.relative_to(task): path.read_bytes() for path in task.rglob("*") if path.is_file()}
    error = capsys.readouterr().err
    assert "host descriptor pinning" in error
    assert "scripts/run-research-harness-container.sh" in error


def test_transaction_probe_checks_output_and_explicit_temporary_filesystems(tmp_path):
    task = complete_task(tmp_path)
    output_root, temporary_root = tmp_path / "outputs", tmp_path / "temporary"
    output_root.mkdir()
    temporary_root.mkdir()
    workspace = output_root / "new-study"
    cfg = load_pipeline_config([f"paths.run_dir={workspace / 'run'}", "execution.phases=investigation"])
    probe = Mock()
    host = runtime_host(tmp_path, environ={"AGENTIC_TEMP_ROOT": str(temporary_root), "TMPDIR": "/tmp"})
    results = preflight.execution_checks(cfg, task, host=host, transaction_probe=probe)
    assert [call.args[0] for call in probe.call_args_list] == [output_root, temporary_root]
    assert not workspace.exists()
    assert any(result.name == "runtime filesystem transactions" and result.status == doctor.OK for result in results)


def test_transaction_failure_stops_harness_before_paid_calls(tmp_path, monkeypatch, capsys):
    from src.research import harness
    task = complete_task(tmp_path)
    workspace = tmp_path / "workspace"
    host = runtime_host(tmp_path, environ={"AGENTIC_TEMP_ROOT": str(tmp_path)})
    monkeypatch.setattr(preflight.doctor, "Host", lambda: host)
    monkeypatch.setattr(preflight, "_probe_transaction_filesystem",
                        Mock(side_effect=OSError(errno.ENOTEMPTY, "Directory not empty")))
    prepare = Mock(side_effect=AssertionError("must not spend literature/model credits"))
    monkeypatch.setattr(harness, "prepare", prepare)
    assert cli.main([
        "harness", str(task), "--workspace", str(workspace), "--quality-target", "development",
        "execution.phases=investigation", "container.enabled=false", "versioning.enabled=true",
    ]) == 2
    prepare.assert_not_called()
    assert not workspace.exists()
    assert not (task / "research_context").exists()
    assert "runtime filesystem transactions" in capsys.readouterr().err


def test_failed_transaction_probe_cleans_only_its_private_directory(tmp_path, monkeypatch):
    from src.pipeline import portable_publication
    retained = tmp_path / "retained.txt"
    retained.write_text("keep")
    monkeypatch.setattr(portable_publication, "atomic_exchange_prepared",
                        Mock(side_effect=OSError(errno.ENOTEMPTY, "Directory not empty")))
    with pytest.raises(OSError, match="Directory not empty"):
        _probe_transaction_filesystem(tmp_path)
    assert list(tmp_path.iterdir()) == [retained]
    assert retained.read_text() == "keep"


def test_successful_transaction_probe_removes_its_private_directory(tmp_path):
    retained = tmp_path / "retained.txt"
    retained.write_text("keep")
    _probe_transaction_filesystem(tmp_path)
    assert list(tmp_path.iterdir()) == [retained]
    assert retained.read_text() == "keep"
