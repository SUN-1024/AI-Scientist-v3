"""Portfolio config, root bootstrap, and downstream handoff integration."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from src.investigation import cli as investigation_cli
from src.investigation.config import ConfigValidationError
from src.investigation.config_loader import load_config
from src.investigation.portfolio_controller import PortfolioConfig, PortfolioController
from src.investigation.search_tree import _fingerprint
from src.orchestrator import state as state_store
from src.pipeline.cli import load_pipeline_config
from src.pipeline.config import (
    investigation_completion_payload,
    validate_completed_investigation,
    validate_pipeline_config,
)
from src.pipeline.paths import derive_paths
from src.pipeline.phases import build_investigation_cmd, build_task_drafting_cmd
from tests.investigation.test_investigation_cli import CLI, _docker_available, _main_setup
from tests.program_testutils import make_program_dir


PORTFOLIO_OVERRIDES = [
    "portfolio.enabled=true", "container.enabled=true", "execution.workers=2",
    "execution.harness_evaluated_attempts=true", "execution.max_program_evals_per_attempt=1",
    "execution.max_idea_cycles=20",
]


def test_defaults_preserve_legacy_and_harness_preset_forwards_portfolio() -> None:
    assert not load_config([]).portfolio.enabled
    cfg = load_pipeline_config([
        "preset=research_harness", "task.dir=examples/circpack",
        "execution.phases=investigation",
    ])
    assert validate_pipeline_config(cfg) == []
    assert cfg.investigation.workers == 2
    assert cfg.investigation.portfolio_enabled
    assert not cfg.investigation.adaptive_enabled
    paths = derive_paths(cfg)
    stage = load_config(build_investigation_cmd(cfg, paths)[3:])
    assert stage.portfolio.enabled and stage.execution.workers == 2
    drafting = load_config(build_task_drafting_cmd(cfg, paths)[3:])
    assert not drafting.portfolio.enabled and not drafting.adaptive_investigation.enabled
    assert drafting.execution.workers == 1


@pytest.mark.parametrize("override, expected", [
    ("portfolio.policy_version=2", "policy_version"),
    ("portfolio.initial_ideas=true", "initial_ideas"),
    ("portfolio.promote_ideas=5", "promote_ideas"),
    ("portfolio.max_valid_evals=13", "planned evaluations"),
    ("portfolio.min_improvement=.nan", "min_improvement"),
    ("execution.max_idea_cycles=19", "max_idea_cycles"),
    ("execution.workers=5", "workers"),
    ("execution.max_program_evals_per_attempt=2", "max_program_evals_per_attempt"),
    ("execution.harness_evaluated_attempts=false", "host evaluation"),
    ("execution.prune_every_n_cycles=4", "pruning disabled"),
    ("adaptive_investigation.enabled=true", "both be enabled"),
    ("container.enabled=false", "container.enabled"),
])
def test_portfolio_invalid_config_rejected_by_stage_and_pipeline(override: str, expected: str) -> None:
    overrides = [item for item in PORTFOLIO_OVERRIDES if item.split("=")[0] != override.split("=")[0]] + [override]
    with pytest.raises(ConfigValidationError) as error:
        load_config(overrides)
    assert expected in error.value.message
    cfg = load_pipeline_config([
        "task.dir=examples/circpack", "execution.phases=investigation",
        *[item if item.startswith("container.") else "investigation." + item for item in overrides],
    ])
    assert any(expected in item for item in validate_pipeline_config(cfg))


def test_portfolio_child_lease_is_single_worker_and_not_a_pipeline_override(tmp_path: Path) -> None:
    lease = tmp_path / "lease.json"
    args = [
        "container.enabled=true", "execution.harness_evaluated_attempts=true",
        "execution.max_program_evals_per_attempt=1", f"execution.portfolio_lease_file={lease}",
    ]
    assert load_config(args).execution.portfolio_lease_file == str(lease)
    with pytest.raises(ConfigValidationError) as error:
        load_config([*args, "execution.workers=2"])
    assert "child lease requires workers=1" in error.value.message
    with pytest.raises(ConfigValidationError) as error:
        load_config([*args[:-1], "execution.portfolio_lease_file=relative.json"])
    assert "absolute host lease path" in error.value.message
    cfg = load_pipeline_config([
        "task.dir=examples/circpack", "execution.phases=investigation",
        f"investigation.execution.portfolio_lease_file={lease}",
    ])
    assert any("cannot be overridden" in item for item in validate_pipeline_config(cfg))


def test_parallel_resume_stays_rejected_without_portfolio_state(tmp_path: Path) -> None:
    with pytest.raises(ConfigValidationError) as error:
        load_config(["execution.workers=2", "execution.resume=true"])
    assert "parallel resume requires" in error.value.message
    with pytest.raises(ConfigValidationError) as error:
        load_config([*PORTFOLIO_OVERRIDES, "execution.resume=true", f"paths.results_dir={tmp_path}"])
    assert "portfolio resume state" in error.value.message


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("qualified", [False, True])
def test_root_incumbent_is_qualified_before_portfolio_dispatch(tmp_path: Path, workers: int, qualified: bool) -> None:
    _, results_dir, cfg = _main_setup(tmp_path, max_idea_cycles=20)
    cfg.container.enabled = True
    cfg.portfolio.enabled = True
    cfg.execution.workers = workers
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.max_program_evals_per_attempt = 1
    calls = []

    def qualify(**kwargs):
        calls.append("qualify")
        if qualified:
            kwargs["best_manifest"].update(combined_score=1.0, correct=True)
            kwargs["outer_state"].update(best_combined_score=1.0, best_correct=True)
        return qualified

    def dispatch(**kwargs):
        calls.append("dispatch")
        assert kwargs["results_dir"] == results_dir.resolve()
        assert kwargs["best_program_dir"].is_relative_to(results_dir.resolve())
        assert kwargs["best_manifest"]["combined_score"] == 1.0
        assert kwargs["outer_state"]["portfolio_enabled"] is True
        return 0

    with (
        patch(f"{CLI}.load_config", return_value=cfg), _docker_available(),
        patch(f"{CLI}._docker_image_exists", return_value=(True, "")),
        patch(f"{CLI}.prepare_task_runtime", return_value=True),
        patch("src.investigation.main_entry.runtime_baseline_score", return_value=1.0),
        patch("src.investigation.main_entry.reconcile_harness_incumbent", side_effect=qualify),
        patch(f"{CLI}.run_portfolio_campaign", side_effect=dispatch),
        patch(f"{CLI}.run_parallel_campaign", side_effect=AssertionError("legacy parallel dispatched")),
        patch(f"{CLI}.run_idea_generation", side_effect=AssertionError("root generated an idea")),
    ):
        assert investigation_cli.main() == (0 if qualified else 2)
    assert calls == (["qualify", "dispatch"] if qualified else ["qualify"])


def test_child_lease_lock_precedes_task_setup_and_releases_on_failure(tmp_path: Path) -> None:
    task_dir, results_dir, cfg = _main_setup(tmp_path)
    cfg.execution.portfolio_lease_file = str(tmp_path / "lease.json")
    calls = []

    @contextmanager
    def lease_lock(**kwargs):
        assert kwargs["task_dir"] == task_dir.resolve()
        assert kwargs["results_dir"] == results_dir.resolve()
        calls.append("lock")
        try:
            yield
        finally:
            calls.append("unlock")

    def setup(*args):
        calls.append("setup")
        return False

    with (
        patch(f"{CLI}.load_config", return_value=cfg),
        patch("src.investigation.portfolio_campaign.external_portfolio_worker_lock", side_effect=lease_lock),
        patch(f"{CLI}.run_task_setup", side_effect=setup),
    ):
        assert investigation_cli.main() == 2
    assert calls == ["lock", "setup", "unlock"]


def _completed_portfolio(tmp_path: Path, *, baseline_score: float = 1.0) -> tuple[Path, PortfolioController]:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.md").write_text("Test whole idea portfolio\n")
    (task / "evaluate.py").write_text("print('development')\n")
    workspace = tmp_path / "results"
    seed = workspace / "seed"
    make_program_dir(seed, content="print('seed')\n")
    policy = PortfolioConfig(enabled=True, initial_ideas=1, promote_ideas=1,
        tuning_trials_per_idea=1, replication_evals=1, max_valid_evals=3, max_transport_failures=1)
    controller = PortfolioController(workspace, task, policy)
    controller.qualify_baseline({"combined_score": baseline_score, "correct": True, "receipt_id": "baseline"}, seed)
    for index in range(3):
        lease = controller.reserve_next(1)
        assert lease is not None
        if lease["stage"] == "replication":
            program = Path(lease["parent_program_dir"])
        else:
            program = Path(lease["result_dir"]) / "program_1"
            make_program_dir(program, content=f"print({index})\n")
        controller.record_receipt(lease["lease_id"], {
            "score": 2.0 + index, "correct": True, "program_dir": str(program),
            "source_hash": _fingerprint(program), "receipt_id": f"receipt-{index}",
        })
    assert controller.handoff() is not None
    candidate = controller.data["frozen_candidate"]
    state_store.save_best_manifest(workspace, {
        "program_dir": candidate["program_dir"], "entrypoint_path": "src/program.py",
        "combined_score": candidate["score"], "correct": True,
        "source_hash": candidate["source_hash"], "root_idea_id": candidate["root_idea_id"],
        "portfolio_lease_id": candidate["lease_id"], "source_attempt_id": "portfolio_attempt",
    })
    state_store.save_outer_state(workspace, {
        "current_cycle": 3, "portfolio_enabled": True, "portfolio_status": controller.data["status"],
        "best_program_dir": candidate["program_dir"], "best_combined_score": candidate["score"], "best_correct": True,
    })
    return workspace, controller


def test_baseline_portfolio_handoff_is_completed_without_a_new_winner(tmp_path: Path) -> None:
    workspace, controller = _completed_portfolio(tmp_path, baseline_score=10.0)
    assert controller.data["frozen_candidate"]["root_idea_id"] == "baseline"
    manifest = json.loads((workspace / "best_program_manifest.json").read_text())
    manifest["source_attempt_id"] = None
    state_store.save_best_manifest(workspace, manifest)
    assert any("retained its baseline" in error for error in validate_completed_investigation(workspace))
    payload = investigation_completion_payload(workspace)
    assert payload and payload["outcome"] == "completed_no_winner"


def test_completed_portfolio_requires_matching_handoff_state_and_candidate(tmp_path: Path) -> None:
    workspace, controller = _completed_portfolio(tmp_path)
    assert validate_completed_investigation(workspace) == []
    payload = investigation_completion_payload(workspace)
    assert payload and payload["portfolio_state_hash"] == controller.data["state_hash"]
    handoff_path = workspace / "validation_handoff.json"
    saved = handoff_path.read_text()
    handoff = json.loads(saved)
    handoff["candidate"]["source_hash"] = "0" * 64
    handoff_path.write_text(json.dumps(handoff))
    assert any("handoff" in error for error in validate_completed_investigation(workspace))
    assert investigation_completion_payload(workspace) is None
    handoff_path.write_text(saved)
    candidate = Path(controller.data["frozen_candidate"]["program_dir"])
    (candidate / "src/program.py").write_text("print('changed after selection')\n")
    assert any("source has changed" in error for error in validate_completed_investigation(workspace))
    assert investigation_completion_payload(workspace) is None


def test_portfolio_completion_cannot_fall_back_to_legacy_when_state_is_missing(tmp_path: Path) -> None:
    workspace, _ = _completed_portfolio(tmp_path)
    (workspace / "portfolio_investigation.json").unlink()
    assert validate_completed_investigation(workspace)
    assert investigation_completion_payload(workspace) is None


def test_external_worker_reuses_exact_parent_receipt_without_baseline_requalification(tmp_path: Path) -> None:
    task_dir, results_dir, cfg = _main_setup(tmp_path)
    parent = tmp_path / "qualified_parent" / "program_1"
    make_program_dir(parent, content="print('jointly tuned parent')\n")
    cfg.paths.seed_program = str(parent)
    cfg.container.enabled = True
    cfg.execution.portfolio_lease_file = str(results_dir / "portfolio_lease.json")
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.max_program_evals_per_attempt = 1
    context = {"root_idea_id": "idea_0001", "lease_id": "idea_0001_tuning_002", "parent": {
        "program_dir": str(parent), "score": 7.5, "correct": True,
        "source_hash": _fingerprint(parent), "receipt_id": "verified-parent-receipt",
    }}

    @contextmanager
    def lease_lock(**kwargs):
        yield

    def load_context(**kwargs):
        assert kwargs["seed_program_dir"] == parent.resolve()
        return context

    def cycles(**kwargs):
        manifest = kwargs["best_manifest"]
        assert manifest["combined_score"] == 7.5 and manifest["correct"] is True
        assert manifest["parent_receipt_id"] == "verified-parent-receipt"
        assert Path(manifest["program_dir"]).is_relative_to(results_dir)
        assert _fingerprint(Path(manifest["program_dir"])) == context["parent"]["source_hash"]
        return 0

    with (
        patch(f"{CLI}.load_config", return_value=cfg), _docker_available(),
        patch(f"{CLI}._docker_image_exists", return_value=(True, "")),
        patch("src.investigation.portfolio_campaign.external_portfolio_worker_lock", side_effect=lease_lock),
        patch("src.investigation.portfolio_campaign.load_external_portfolio_context", side_effect=load_context),
        patch(f"{CLI}.prepare_task_runtime", side_effect=AssertionError("worker requalified task")),
        patch("src.investigation.main_entry.reconcile_harness_incumbent", side_effect=AssertionError("worker requalified parent")),
        patch("src.investigation.campaign_loop.run_cycles_and_finalize", side_effect=cycles),
    ):
        assert investigation_cli.main() == 0


def test_parallel_portfolio_resume_accepts_only_the_frozen_policy(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.md").write_text("Portfolio resume task\n")
    (task / "evaluate.py").write_text("print('development')\n")
    run = tmp_path / "run"
    workspace = run / "investigation"
    seed = workspace / "seed"
    make_program_dir(seed, content="print('baseline')\n")
    policy = PortfolioConfig(enabled=True, initial_ideas=2, promote_ideas=1,
        tuning_trials_per_idea=1, replication_evals=1, max_valid_evals=4, max_transport_failures=1)
    controller = PortfolioController(workspace, task, policy)
    controller.qualify_baseline({"combined_score": 1.0, "correct": True}, seed)
    state_store.save_outer_state(workspace, {"current_cycle": 0, "best_program_dir": str(seed)})
    state_store.save_best_manifest(workspace, {"program_dir": str(seed), "entrypoint_path": "src/program.py"})
    overrides = [
        "portfolio.enabled=true", "portfolio.initial_ideas=2", "portfolio.promote_ideas=1",
        "portfolio.tuning_trials_per_idea=1", "portfolio.replication_evals=1",
        "portfolio.max_valid_evals=4", "portfolio.max_transport_failures=1",
        "execution.max_idea_cycles=5", "execution.workers=2", "execution.resume=true",
        "execution.harness_evaluated_attempts=true", "execution.max_program_evals_per_attempt=1",
    ]
    assert load_config([*overrides, "container.enabled=true", f"paths.task_dir={task}", f"paths.results_dir={workspace}"]).execution.resume
    pipeline = load_pipeline_config([
        f"task.dir={task}", f"paths.run_dir={run}", "execution.phases=investigation",
        *["investigation." + item for item in overrides],
    ])
    assert validate_pipeline_config(pipeline) == []
    with pytest.raises(ConfigValidationError) as error:
        load_config([*overrides, "portfolio.min_improvement=0.1", "container.enabled=true", f"paths.task_dir={task}", f"paths.results_dir={workspace}"])
    assert "policy/task/protocol changed" in error.value.message
