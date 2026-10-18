"""Coding-agent retry settings remain independent and survive orchestration."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.investigation.config import ConfigValidationError, ExecutionConfig, config_to_flat_dict
from src.investigation.config_loader import load_config, write_hydra_snapshot
from src.investigation.parallel_helpers import build_parallel_worker_command
from src.pipeline.cli import load_pipeline_config
from src.pipeline.config import describe_pipeline_config, validate_pipeline_config
from src.pipeline.paths import derive_paths
from src.pipeline.phases import build_investigation_cmd


def test_retry_defaults_and_drafting_settings_are_independent(tmp_path: Path) -> None:
    cfg = load_config([])
    defaults = ExecutionConfig()
    assert cfg.execution.agent_transient_max_attempts == defaults.agent_transient_max_attempts == 3
    assert cfg.execution.agent_transient_backoff_seconds == defaults.agent_transient_backoff_seconds == 2
    assert cfg.task_drafting.agent_transient_max_attempts == 4

    cfg = load_config([
        "execution.agent_transient_max_attempts=7",
        "execution.agent_transient_backoff_seconds=0.25",
        "task_drafting.agent_transient_max_attempts=2",
    ])
    assert cfg.execution.agent_transient_max_attempts == 7
    assert cfg.execution.agent_transient_backoff_seconds == 0.25
    assert cfg.task_drafting.agent_transient_max_attempts == 2
    write_hydra_snapshot(cfg, tmp_path)
    saved = yaml.safe_load((tmp_path / ".hydra" / "config.yaml").read_text())
    assert saved["execution"]["agent_transient_max_attempts"] == 7
    assert saved["execution"]["agent_transient_backoff_seconds"] == 0.25
    assert saved["task_drafting"]["agent_transient_max_attempts"] == 2
    flat = config_to_flat_dict(cfg)
    assert flat["execution.agent_transient_max_attempts"] == "7"
    assert flat["execution.agent_transient_backoff_seconds"] == "0.25"


@pytest.mark.parametrize("attempts,backoff", [(1, 0), (10, 60), (3, 0.5)])
def test_retry_bounds_accept_integers_and_fractional_backoff(attempts: int, backoff: float) -> None:
    cfg = load_config([
        f"execution.agent_transient_max_attempts={attempts}",
        f"execution.agent_transient_backoff_seconds={backoff}",
    ])
    assert cfg.execution.agent_transient_max_attempts == attempts
    assert cfg.execution.agent_transient_backoff_seconds == backoff


INVALID_RETRY_VALUES = [
    ("agent_transient_max_attempts", value)
    for value in ("0", "11", "1.5", "true", "null", "abc")
] + [
    ("agent_transient_backoff_seconds", value)
    for value in ("-0.1", "60.1", "true", "null", "abc", ".inf", ".nan")
]


@pytest.mark.parametrize("field,value", INVALID_RETRY_VALUES)
def test_retry_invalid_values_rejected_by_stage_loader(field: str, value: str) -> None:
    with pytest.raises(ConfigValidationError) as error:
        load_config([f"execution.{field}={value}"])
    assert f"execution.{field}" in error.value.message


@pytest.mark.parametrize("field,value", INVALID_RETRY_VALUES)
def test_retry_invalid_values_rejected_by_pipeline(field: str, value: str) -> None:
    cfg = load_pipeline_config([
        "task.dir=examples/circpack",
        "container.enabled=false",
        "execution.phases=investigation",
        f"investigation.execution.{field}={value}",
    ])
    errors = validate_pipeline_config(cfg)
    assert any(f"investigation.execution.{field}" in error for error in errors)


def test_pipeline_and_parallel_workers_keep_coding_retry_overrides(tmp_path: Path) -> None:
    pipeline = load_pipeline_config([
        "task.dir=examples/circpack",
        "container.enabled=false",
        "execution.phases=investigation",
        "investigation.execution.agent_transient_max_attempts=8",
        "investigation.execution.agent_transient_backoff_seconds=0.125",
        "investigation.task_drafting.agent_transient_max_attempts=2",
    ])
    assert validate_pipeline_config(pipeline) == []
    paths = replace(derive_paths(pipeline), task_dir=Path("examples/circpack").resolve())
    command = build_investigation_cmd(pipeline, paths)
    assert not any(arg.startswith("task_drafting.") for arg in command)
    stage = load_config(command[3:])
    assert stage.execution.agent_transient_max_attempts == 8
    assert stage.execution.agent_transient_backoff_seconds == 0.125
    assert stage.task_drafting.agent_transient_max_attempts == 4

    worker_command = build_parallel_worker_command(
        cfg=stage,
        task_dir=paths.task_dir,
        instruct_path=tmp_path / "instructions.md",
        results_dir=tmp_path / "worker",
        global_memory_dir=tmp_path / "memory",
        assigned_cycles=1,
        worker_index=1,
    )
    worker = load_config(worker_command[3:])
    assert worker.execution.agent_transient_max_attempts == 8
    assert worker.execution.agent_transient_backoff_seconds == 0.125
    rows = {row.key: row for row in describe_pipeline_config()}
    assert rows["investigation.execution.agent_transient_max_attempts"].default == "3"
    assert rows["investigation.execution.agent_transient_backoff_seconds"].default == "2.0"


@pytest.mark.parametrize("campaign_workers,expected", [(0, 4), (7, 7)])
def test_worker_retains_campaign_worker_count(
    tmp_path: Path, campaign_workers: int, expected: int
) -> None:
    cfg = load_config([
        "execution.workers=4",
        f"execution.campaign_workers={campaign_workers}",
    ])
    command = build_parallel_worker_command(
        cfg=cfg,
        task_dir=Path("examples/circpack").resolve(),
        instruct_path=tmp_path / "instructions.md",
        results_dir=tmp_path / "worker",
        global_memory_dir=tmp_path / "memory",
        assigned_cycles=2,
        worker_index=1,
        global_cycle=6,
    )
    worker = load_config(command[3:])
    assert worker.execution.workers == 1
    assert worker.execution.campaign_workers == expected
    assert worker.execution.cycle_offset == 4
    assert worker.execution.total_cycles == cfg.execution.max_idea_cycles


@pytest.mark.parametrize("value", ["-1", "1.5", "true", "null", "abc"])
def test_campaign_workers_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ConfigValidationError) as error:
        load_config([f"execution.campaign_workers={value}"])
    assert "execution.campaign_workers" in error.value.message
