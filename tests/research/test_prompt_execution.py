"""Run the real idea shell wrapper with a deterministic CLI fixture."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from src.investigation.search_tree import SearchTree
from src.research.config import LiteratureConfig
from src.research.literature import prepare_literature
from tests.e2e.test_agent_execution_pipeline import _write_fake_agent, _write_task_dir


def test_real_idea_wrapper_receives_literature_and_the_selected_branch(tmp_path):
    task = _write_task_dir(tmp_path)
    (task / "prior.md").write_text("Method: fixture baseline. Limitation: a fixed search step.\n")
    prepare_literature(task, LiteratureConfig(sources=[{
        "id": "prior_method", "title": "Fixture comparison source", "path": "prior.md",
    }]), offline=True)
    results = tmp_path / "results"
    results.mkdir()
    SearchTree(results, task).choose(1, {"combined_score": 1, "correct": True}, task / "program_0")
    agent_bin = tmp_path / "bin"
    agent_bin.mkdir()
    _write_fake_agent(agent_bin, "headless")
    output = tmp_path / "idea"
    environment = {
        **os.environ,
        "PATH": str(agent_bin) + os.pathsep + os.environ["PATH"],
        "HEADLESS_BIN": str(agent_bin / "headless"), "WRAPPER_PYTHON": sys.executable,
    }
    script = Path(__file__).resolve().parents[2] / "src" / "scripts" / "idea.sh"
    completed = subprocess.run([
        "bash", str(script), "--agent", "codex", "--task-dir", str(task),
        "--results-dir", str(results), "--cycle-index", "1", "--output-dir", str(output),
    ], env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prompt = (output / "idea_prompt.md").read_text()
    assert "Fixture comparison source" in prompt
    assert "research_basis" in prompt
    assert "prior_method:chars_" in prompt
    assert '"node_id": "root"' in prompt
    assert "global best remains" in prompt
    assert "{{RESEARCH_CONTEXT}}" not in prompt
    assert "{{SEARCH_PARENT_CONTEXT}}" not in prompt
    assert str(task.resolve()) not in prompt
    parent = json.loads((results / "search_parent.json").read_text())
    assert parent["relative_program_dir"] == "program_0"
