"""Validate chart types in the real strategy gate before starting a paper writer."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from src.orchestrator.figure_manifest_schema import SUPPORTED_FIGURE_TYPES


REPO_ROOT = Path(__file__).resolve().parents[2]


def _validate_strategy(tmp_path: Path, figure_type: str) -> tuple[int, dict[str, object]]:
    payload = {
        "status": "ready", "selected_prompt_family": "paper_standalone",
        "recommended_title": "Fixture", "central_question": "A bounded question",
        "central_thesis": "A supported statement", "narrative_rationale": "A clear rationale",
        "method_groups": [{"name": "control", "role": "baseline", "include_reason": "comparison"}],
        "results_focus": [], "figure_plan": [], "forbidden_headings": [],
        "forbidden_phrases": [], "notes_for_writer": [],
        "supplementary_plan": {"purpose": "Supporting detail", "planned_subsections": [
            {"heading": "Details", "grounded_content": "Fixture evidence", "source_ids": ["fixture"]},
        ]},
        "figure_specs": [{
            "figure_id": "fixture_plot", "figure_type": figure_type, "required": True,
            "preferred_generator": "generate-concept-figure" if figure_type in {"diagram", "flow_diagram"} else "matplotlib",
            "max_panels": 1, "max_categories_per_panel": 1, "max_series_per_panel": 1,
            "allow_progression": False,
        }],
    }
    (tmp_path / "paper_strategy.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "paper_strategy.md").write_text("A strategy memo.\n", encoding="utf-8")
    (tmp_path / "paper_data.json").write_text(
        json.dumps({"grounding_sources": {"fixture": {}}}), encoding="utf-8",
    )
    script = (REPO_ROOT / "src/scripts/paper.sh").read_text(encoding="utf-8")
    start = script.index("\nvalidate_strategy_with_feedback() {") + 1
    end = script.index('\npaper_gate_start "references" "phase=initial"', start)
    feedback = tmp_path / "feedback.json"
    assignments = {
        "REPO_ROOT": str(REPO_ROOT), "PYTHON_BIN": sys.executable, "OUTPUT_DIR": str(tmp_path),
        "STRATEGY_VALIDATION_FEEDBACK_JSON_PATH": str(feedback),
        "STRATEGY_VALIDATION_FEEDBACK_MD_PATH": str(tmp_path / "feedback.md"),
    }
    command = "\n".join(f"{name}={shlex.quote(value)}" for name, value in assignments.items())
    command += "\n" + script[start:end] + "\nvalidate_strategy_with_feedback\n"
    env = {"PATH": os.defpath}
    if "HOME" in os.environ:
        env["HOME"] = os.environ["HOME"]
    result = subprocess.run(
        ["bash", "-c", command], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=20,
    )
    assert feedback.is_file(), result.stderr
    # The gate reports invalid types; it must not silently relabel the strategy.
    assert json.loads((tmp_path / "paper_strategy.json").read_text(encoding="utf-8")) == payload
    return result.returncode, json.loads(feedback.read_text(encoding="utf-8"))


@pytest.mark.parametrize("figure_type", [*sorted(SUPPORTED_FIGURE_TYPES), "flow_diagram", "horizontal_bar"])
def test_strategy_gate_accepts_manifest_types_and_existing_aliases(tmp_path: Path, figure_type: str) -> None:
    returncode, feedback = _validate_strategy(tmp_path, figure_type)
    assert returncode == 0, feedback
    assert feedback["ok"] is True


@pytest.mark.parametrize("figure_type", [
    "horizontal_dot_plot", "paired_points_and_interval", "faceted_condition_dot_plot",
])
def test_strategy_gate_rejects_unrepresentable_chart_types(tmp_path: Path, figure_type: str) -> None:
    returncode, feedback = _validate_strategy(tmp_path, figure_type)
    assert returncode == 1
    assert feedback["ok"] is False
    errors = str(feedback["hard_errors"])
    assert "figure_specs[0].figure_type" in errors
    assert figure_type in errors
    assert "scatter" in errors and "composite" in errors
