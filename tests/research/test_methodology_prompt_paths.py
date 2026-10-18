"""Frozen methodology rules reach the prompts consumed by real wrapper paths."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.investigation.campaign_budget import render_campaign_budget, write_campaign_budget
from src.investigation.config import InvestigationConfig
from src.investigation.paper_prompt_context import paper_workspace_context_replacements
from src.research.config import ResearchConfig
from src.research.evidence import render_context
from src.research.methodology import write_methodology
from src.research.paper_readiness import assess_readiness, write_readiness
from tests.e2e.test_agent_execution_pipeline import (
    EXPERIMENT_DIR_RE, _script_path, _write_fake_agent, _write_idea_and_rubrics,
    _write_task_dir,
)


def _agenda(task: Path) -> None:
    (task / "research_agenda.json").write_text(json.dumps({
        "schema_version": 1, "research_question": "Does the bounded treatment change the outcome?",
        "study_design": {"question": "Frozen study-design fixture"},
    }), encoding="utf-8")


def _assert_rules(prompt: str, marker: dict, stage: str) -> None:
    assert marker["rules_sha256"] in prompt
    rules = marker["stage_rules"][stage]
    for rule in ([rules] if isinstance(rules, str) else rules):
        assert rule in prompt


def test_no_marker_keeps_legacy_no_catalog_and_no_budget_text(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    # Historically an absent catalog returns before reading an agenda.
    (task / "research_agenda.json").write_text("not json")
    expected = "No structured literature dossier is attached. Do not infer that novelty has been verified."
    assert render_context(task) == expected
    assert render_context(task, for_paper=True) == expected
    assert render_campaign_budget(tmp_path) == (
        "No resolved campaign budget was supplied. Follow the explicit wrapper evaluation limit "
        "and the task's scientific constraints; do not infer total cycles."
    )


@pytest.mark.parametrize("catalog", [False, True])
def test_idea_and_paper_context_include_rules_even_without_literature(tmp_path, catalog):
    task = tmp_path / "task"
    marker = write_methodology(task)
    _agenda(task)
    if catalog:
        (task / "research_context/catalog.json").write_text(json.dumps({
            "schema_version": 1, "status": "unavailable", "sources": [],
        }))
    _assert_rules(render_context(task), marker, "idea")
    paper = paper_workspace_context_replacements(tmp_path)["RESEARCH_CONTEXT"]
    _assert_rules(paper, marker, "paper")
    assert "Frozen study-design fixture" in paper


@pytest.mark.parametrize("with_budget", [False, True])
def test_coding_context_includes_frozen_design_but_idea_budget_does_not_duplicate_it(tmp_path, with_budget):
    task = tmp_path / "task"
    marker = write_methodology(task)
    _agenda(task)
    if with_budget:
        write_campaign_budget(tmp_path, InvestigationConfig(), cycle=1,
                              max_program_evals_per_attempt=1, outer_state={})
    _assert_rules(render_campaign_budget(tmp_path), marker, "coding")
    assert "Frozen study-design fixture" in render_campaign_budget(tmp_path)
    assert marker["rules_sha256"] not in render_campaign_budget(tmp_path, cycle_index=1)


@pytest.mark.parametrize("stage", ["idea", "coding"])
def test_actual_shell_wrapper_prompt_carries_stage_rules_once(tmp_path, stage):
    task = _write_task_dir(tmp_path)
    marker = write_methodology(task)
    _agenda(task)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_agent(bin_dir, "headless")
    env = {**os.environ, "HEADLESS_BIN": str(bin_dir / "headless"),
           "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "WRAPPER_PYTHON": sys.executable, "RUBRIC_ID": "score_threshold"}
    command = ["bash", str(_script_path("idea.sh" if stage == "idea" else "run.sh")),
               "--agent", "codex", "--task-dir", str(task),
               "--results-dir", str(tmp_path / "results")]
    if stage == "idea":
        output = tmp_path / "idea"
        command += ["--output-dir", str(output), "--cycle-index", "1"]
    else:
        idea, rubrics = _write_idea_and_rubrics(tmp_path)
        command += ["--idea-file", str(idea), "--rubrics-file", str(rubrics),
                    "--attempt-id", "idea_0001_attempt_01"]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if stage == "coding":
        match = EXPERIMENT_DIR_RE.search(completed.stdout + completed.stderr)
        assert match is not None
        output = Path(match.group(1).strip())
    prompt = (output / ("idea_prompt.md" if stage == "idea" else "prompt.md")).read_text()
    _assert_rules(prompt, marker, stage)
    assert prompt.count(marker["rules_sha256"]) == 1
    assert "Frozen study-design fixture" in prompt


def test_readiness_requires_enabled_methodology_declarations_and_records_hashes(tmp_path):
    workspace = tmp_path / "workspace"
    marker = write_methodology(workspace / "input_task")
    report = write_readiness(workspace, ResearchConfig(), {})
    method = report["checks"]["methodology"]
    assert method["status"] == "needs_evidence"
    assert method["rules_sha256"] == marker["rules_sha256"]
    assert method["policy_version"] == marker["policy_version"]
    assert method["assessment_scope"] == "frozen_initial_agenda"
    assert method["selected_candidate_design_assessed"] is False
    assert method["scientific_validity_certified"] is False
    assert any(gap["check"] == "methodology" for gap in report["gaps"])
    assert any(action["kind"] == "methodology_followup" for action in report["next_actions"])
    assert any(action["kind"] == "compare_selected_design" for action in report["next_actions"])
    assert report["ready_for_human_review"] is False
    assert "does not certify scientific, causal, or clinical validity" in report["note"]
    saved_actions = json.loads((workspace / "research_next_actions.json").read_text())
    assert saved_actions["methodology"]["rules_sha256"] == marker["rules_sha256"]
    assert marker["rules_sha256"] in (workspace / "paper_readiness.md").read_text()


def test_legacy_readiness_does_not_add_methodology_requirement(tmp_path):
    cfg = ResearchConfig()
    cfg.methodology.enabled = False
    report = assess_readiness(tmp_path, cfg, {})
    assert "methodology" not in report["checks"]
    assert not any(action["check"] == "methodology" for action in report["next_actions"])


def test_enabled_readiness_does_not_treat_a_missing_marker_as_legacy(tmp_path):
    report = assess_readiness(tmp_path, ResearchConfig(), {})
    assert report["checks"]["methodology"]["status"] == "needs_evidence"
    assert report["checks"]["methodology"]["rules_sha256"] is None
    assert any(gap["check"] == "methodology" for gap in report["gaps"])
