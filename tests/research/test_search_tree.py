from __future__ import annotations

import json
import pytest

from src.investigation.search_tree import SearchTree, render_parent_context
from tests.program_testutils import make_program_dir


def root_tree(tmp_path):
    task = tmp_path / "task"
    results = tmp_path / "results"
    task.mkdir()
    results.mkdir()
    make_program_dir(task / "program_0", content="VALUE = 0\n")
    program = task / "program_0"
    return task, results, program, SearchTree(results, task, width=4, explore_every=3)


def test_tree_can_expand_a_correct_nonwinning_branch_and_resume(tmp_path):
    task, results, root, tree = root_tree(tmp_path)
    best = {"combined_score": 0, "correct": True}
    assert tree.choose(1, best, root)["node_id"] == "root"
    make_program_dir(results / "first", content="VALUE = 5\n")
    first = results / "first"
    tree.record(cycle=1, idea={"idea_id": "i1", "hypothesis": "A"}, candidate=(first, "src/program.py", 5, True), exit_code=0)
    assert tree.choose(2, {"combined_score": 5, "correct": True}, first)["node_id"] == "cycle_0001"
    make_program_dir(results / "second", content="VALUE = 4\n")
    second = results / "second"
    tree.record(cycle=2, idea={"idea_id": "i2", "hypothesis": "B"}, candidate=(second, "src/program.py", 4, True), exit_code=0)
    chosen = tree.choose(3, {"combined_score": 5, "correct": True}, first)
    assert chosen["node_id"] == "cycle_0002"
    assert chosen["score"] == 4
    restored = SearchTree(results, task, width=4, explore_every=3)
    assert restored.choose(3, best, root) == chosen
    assert "global best remains" in render_parent_context(results, 3)
    assert "cycle_0002" not in render_parent_context(results, 4)


def test_changed_sources_bad_configuration_and_external_parents_are_rejected(tmp_path):
    task, results, root, tree = root_tree(tmp_path)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    (root / "src" / "program.py").write_text("VALUE = 99\n")
    with pytest.raises(ValueError, match="changed after evaluation"):
        tree.choose(1, {}, root)
    with pytest.raises(ValueError, match="config mismatch"):
        SearchTree(results, task, width=5)
    state = json.loads((results / "search_tree.json").read_text())
    state["nodes"]["root"]["program_dir"] = str(tmp_path / "outside")
    (results / "search_tree.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="outside"):
        SearchTree(results, task).choose(1, {}, root)


def test_failed_attempts_are_recorded_but_not_selected(tmp_path):
    task, results, root, tree = root_tree(tmp_path)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    tree.record(cycle=1, idea={"idea_id": "fail", "hypothesis": "Failed execution"}, candidate=None, exit_code=1)
    assert tree.choose(2, {"combined_score": 0, "correct": True}, root)["node_id"] == "root"
    state = json.loads((results / "search_tree.json").read_text())
    assert state["nodes"]["cycle_0001"]["exit_code"] == 1
    assert state["nodes"]["cycle_0001"]["correct"] is False


def test_real_investigation_loader_preserves_branching_overrides():
    from src.investigation.config_loader import load_config

    cfg = load_config(['execution.search_strategy=tree', 'execution.search_width=5',
                       'execution.search_explore_every=4'])
    assert cfg.execution.search_strategy == 'tree'
    assert cfg.execution.search_width == 5
    assert cfg.execution.search_explore_every == 4
