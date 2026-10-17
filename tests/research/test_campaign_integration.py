"""Exercise real campaign selection and scoring with deterministic agent fixtures.

These tests verify execution/data flow. They do not measure LLM research quality.
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path

from src.investigation.campaign_loop import run_cycles_and_finalize
from src.investigation.config import InvestigationConfig
from src.investigation.files import find_best_program_in_experiment
from src.investigation.models import IdeaGenerationResult, RunAttemptResult
from src.orchestrator.paper_data import build_paper_data
from src.orchestrator import state as state_store
from src.research.config import load_config
from src.research.literature import prepare_literature
from tests.program_testutils import make_program_dir
from tests.research.test_unified_research import make_task, basis
from tests.investigation.test_campaign_ideas import _IDEA_BUNDLE


def execute_fixture(root: Path, *, collision: bool = False):
    task = make_task(root / "task")
    catalog = prepare_literature(task, load_config(task).literature, offline=True)
    results = root / "results"
    results.mkdir()
    seed = task / "program_0"
    make_program_dir(seed, content="X = 0\n")
    cfg = InvestigationConfig()
    cfg.execution.search_strategy = "tree"
    cfg.execution.max_idea_cycles = 3
    cfg.execution.max_idea_generation_attempts = 1
    cfg.paper.enabled = False
    used_seeds = []

    def generate(**kwargs):
        workspace = Path(kwargs["output_dir"])
        workspace.mkdir(parents=True)
        bundle = {
            **_IDEA_BUNDLE, "idea_id": f"idea_{kwargs['cycle_index']}",
            "research_basis": basis(catalog, "collision" if collision else "differentiated"),
        }
        (workspace / "idea_bundle.json").write_text(json.dumps(bundle))
        (workspace / "target_rubrics.json").write_text(json.dumps(bundle["target_rubrics"]))
        return IdeaGenerationResult("Fixture agent generated a test plan", 0, workspace)

    def attempt(**kwargs):
        parent = Path(kwargs["seed_program_dir"])
        used_seeds.append(runpy.run_path(str(parent / "src" / "program.py"))["X"])
        idea = json.loads(Path(kwargs["idea_file"]).read_text())
        experiment = results / "ideas" / kwargs["idea_dir_name"]
        program = experiment / "program_1"
        x = [2, 1, 3][len(used_seeds) - 1]
        make_program_dir(program, content=f"X = {x}\n")
        # Execute generated Python, then compute a real deterministic objective.
        measured_x = runpy.run_path(str(program / "src" / "program.py"))["X"]
        score = -(measured_x - 3) ** 2
        (program / "metrics.json").write_text(json.dumps({"combined_score": score}))
        (program / "correct.json").write_text(json.dumps({"correct": True}))
        (experiment / "rubric_status.json").write_text(json.dumps({
            "idea_id": idea["idea_id"], "attempt_id": kwargs["attempt_id"], "overall_pass": True,
            "rubrics": [{"id": "does_the_thing", "pass": True}], "metrics": {"combined_score": score},
        }))
        return RunAttemptResult("Fixture program executed", 0, experiment)

    code = run_cycles_and_finalize(
        cfg=cfg, task_dir=task, results_dir=results, global_memory_dir=results,
        instruct_path=task / "task.md", idea_instruct_path=task / "task.md",
        paper_instruct_path=task / "task.md", paper_contributions_path=None,
        containerize=False, container_image="", container_network="", container_auth_root=root,
        container_env_file=None, container_session_name=None, container_mount_root=None,
        outer_state={"current_cycle": 0, "failure_streak": 0},
        best_manifest={"program_dir": str(seed), "combined_score": -9, "correct": True},
        best_program_dir=seed, start_cycle=1, max_program_evals_per_attempt=1,
        main_started_at=0, log_event_fn=lambda *a, **k: None,
        log_step_start_fn=lambda *a, **k: 0, log_step_end_fn=lambda *a, **k: None,
        format_elapsed_fn=lambda _: "fixture", run_idea_generation_fn=generate,
        run_single_attempt_fn=attempt, run_paper_generation_fn=lambda **k: None,
        reconcile_legacy_idea_local_memory_fn=lambda **k: None,
        find_best_program_in_experiment_fn=find_best_program_in_experiment,
        read_json_fn=state_store.load_json, utc_now_fn=lambda: "2026-10-17T00:00:00Z",
    )
    return code, used_seeds, results


def test_full_campaign_branches_without_losing_incumbent_and_reaches_paper_data(tmp_path):
    code, used_seeds, results = execute_fixture(tmp_path)
    assert code == 0
    # Cycle 3 returns to the lower-scoring X=1 branch, while X=2 was the incumbent.
    assert used_seeds == [0, 2, 1]
    best = json.loads((results / "best_program_manifest.json").read_text())
    assert best["combined_score"] == 0
    tree = json.loads((results / "search_tree.json").read_text())
    assert tree["nodes"]["cycle_0003"]["parent_id"] == "cycle_0002"
    data = build_paper_data(results)
    assert len(data["all_programs"]) == 3
    assert all(row["research_basis"]["closest_work"][0]["source_id"] == "prior" for row in data["all_programs"])
    assert len(data["research_assessments"]) == 3
    from src.research.evidence import merge_assessments
    ablation_data = {"study_type": "ablation"}
    merge_assessments(ablation_data, results)
    assert len(ablation_data["research_assessments"]) == 3


def test_supported_collision_stops_before_any_experiment(tmp_path):
    code, used_seeds, results = execute_fixture(tmp_path, collision=True)
    assert code == 3
    assert used_seeds == []
    decision = json.loads((results / "research_decision.json").read_text())
    assert decision["research_basis"]["status"] == "collision"
