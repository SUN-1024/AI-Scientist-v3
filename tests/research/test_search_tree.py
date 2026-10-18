from __future__ import annotations

import json
import pytest

from src.investigation.search_tree import SearchTree, render_parent_context
from tests.program_testutils import make_program_dir


def root_tree(tmp_path, *, width=4):
    task = tmp_path / "task"
    results = tmp_path / "results"
    task.mkdir()
    results.mkdir()
    make_program_dir(task / "program_0", content="VALUE = 0\n")
    program = task / "program_0"
    return task, results, program, SearchTree(results, task, width=width, explore_every=3)


def record_program(tree, results, cycle, name, content, score):
    program = results / name
    make_program_dir(program, content=content)
    tree.record(cycle=cycle, idea={"idea_id": name, "hypothesis": f"Hypothesis {name}"},
                candidate=(program, "src/program.py", score, True), exit_code=0)
    return program


def repeated_source_tree(tmp_path):
    task, results, root, tree = root_tree(tmp_path, width=2)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    first = record_program(tree, results, 1, "a_first", "VALUE = 10\n", 10)
    tree.choose(2, {"combined_score": 10, "correct": True}, first, selection_policy="incumbent")
    record_program(tree, results, 2, "a_copy", "VALUE = 10\n", 10)
    tree.choose(3, {"combined_score": 10, "correct": True}, first, selection_policy="incumbent")
    distinct = record_program(tree, results, 3, "b", "VALUE = 9\n", 9)
    return task, results, root, tree, first, distinct


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


def test_new_tree_counts_distinct_sources_before_beam_width(tmp_path):
    task, results, root, tree, first, distinct = repeated_source_tree(tmp_path)
    chosen = tree.choose(4, {"combined_score": 10, "correct": True}, first, selection_policy="explore")
    assert chosen["program_dir"] == str(distinct.resolve())
    assert chosen["node_id"] == "cycle_0003"
    state = json.loads((results / "search_tree.json").read_text())
    assert state["selection_policy_version"] == 2
    assert set(state["nodes"]) == {"root", "cycle_0001", "cycle_0002", "cycle_0003"}
    evidence = {node["node_id"]: node for node in chosen["journal"]["candidates"]}
    assert evidence["cycle_0001"]["analysis"] == "Hypothesis a_first"
    assert evidence["cycle_0002"]["analysis"] == "Hypothesis a_copy"
    # The persisted policy and the selected node survive replay.
    assert SearchTree(results, task, width=2).choose(4, {}, root) == chosen


def test_new_representative_inherits_visits_to_all_same_source_nodes(tmp_path):
    task, results, root, tree = root_tree(tmp_path, width=2)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    first = record_program(tree, results, 1, "a_first", "VALUE = 10\n", 10)
    tree.choose(2, {}, first, selection_policy="incumbent")
    distinct = record_program(tree, results, 2, "b", "VALUE = 9\n", 9)
    tree.choose(3, {}, distinct, selection_policy="incumbent")
    # The higher score makes this fresh node the representative of the same code.
    record_program(tree, results, 3, "a_copy", "VALUE = 10\n", 11)
    tree.choose(4, {}, first, selection_policy="incumbent")
    tree.choose(5, {}, root, selection_policy="incumbent")
    tree.choose(6, {}, root, selection_policy="incumbent")
    # Count earlier leases even if that individual node leaves the eligible set.
    tree.data["nodes"]["cycle_0001"]["is_replication"] = True
    chosen = tree.choose(7, {}, first, selection_policy="explore")
    assert chosen["node_id"] == "cycle_0002"
    assert chosen["program_dir"] == str(distinct.resolve())
    # A per-representative counter would incorrectly see cycle_0003 as unvisited.
    assert list(tree.data["leases"].values()).count("cycle_0003") == 0


def test_equal_scores_do_not_merge_different_sources(tmp_path):
    _, results, root, tree = root_tree(tmp_path, width=2)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    first = record_program(tree, results, 1, "a", "VALUE = 10\n", 10)
    tree.choose(2, {}, first, selection_policy="incumbent")
    second = record_program(tree, results, 2, "b", "VALUE = 20\n", 10)
    chosen = tree.choose(3, {}, first, selection_policy="explore")
    assert chosen["program_dir"] == str(second.resolve())
    assert tree.data["nodes"]["cycle_0001"]["source_sha256"] != chosen["source_sha256"]


@pytest.mark.parametrize("version", [None, 1])
def test_legacy_policy_and_serialization_remain_unchanged_on_resume(tmp_path, version):
    task, results, root, tree, first, _ = repeated_source_tree(tmp_path)
    legacy = tree.data
    if version is None:
        del legacy["selection_policy_version"]
    else:
        legacy["selection_policy_version"] = version
    state_path = results / "search_tree.json"
    state_path.write_text(json.dumps(legacy, sort_keys=True, indent=2))
    resumed = SearchTree(results, task, width=2)
    chosen = resumed.choose(4, {}, first, selection_policy="explore")
    # The historical beam contains two A nodes, and the new A node has no visits.
    assert chosen["node_id"] == "cycle_0002"
    expected_state = {**legacy, "leases": {**legacy["leases"], "4": "cycle_0002"}}
    expected_text = json.dumps(expected_state, sort_keys=True, indent=2)
    assert state_path.read_text() == expected_text
    parent_text = (results / "search_parent.json").read_text()
    # Replaying a lease must reproduce both files even with a different preference.
    replayed = SearchTree(results, task, width=2).choose(4, {}, root, selection_policy="incumbent")
    assert replayed == chosen
    assert state_path.read_text() == expected_text
    assert (results / "search_parent.json").read_text() == parent_text


@pytest.mark.parametrize("version", [0, 3, -1, "2", True, 2.0, None])
def test_unknown_or_malformed_selection_policy_version_is_rejected(tmp_path, version):
    task, results, root, tree = root_tree(tmp_path)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    state_path = results / "search_tree.json"
    state = json.loads(state_path.read_text())
    state["selection_policy_version"] = version
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="selection_policy_version"):
        SearchTree(results, task)


@pytest.mark.parametrize("digest", [None, "", "invalid", "g" * 64])
def test_missing_or_invalid_hash_never_merges_unrelated_nodes(tmp_path, digest):
    _, results, root, tree = root_tree(tmp_path)
    tree.choose(1, {"combined_score": 0, "correct": True}, root)
    first = record_program(tree, results, 1, "a", "VALUE = 10\n", 10)
    tree.choose(2, {}, first, selection_policy="incumbent")
    record_program(tree, results, 2, "b", "VALUE = 20\n", 9)
    a, b = (tree.data["nodes"][key] for key in ("cycle_0001", "cycle_0002"))
    if digest is None:
        del a["source_sha256"]
        del b["source_sha256"]
    else:
        a["source_sha256"] = b["source_sha256"] = digest
    assert tree._exploration_pool([a, b]) == [a, b, tree.data["nodes"]["root"]]
    # A fallback group key never bypasses source validation at selection time.
    with pytest.raises(ValueError, match="changed after evaluation"):
        tree.choose(3, {}, first, selection_policy="explore")


def test_real_investigation_loader_preserves_branching_overrides():
    from src.investigation.config_loader import load_config

    cfg = load_config(['execution.search_strategy=tree', 'execution.search_width=5',
                       'execution.search_explore_every=4'])
    assert cfg.execution.search_strategy == 'tree'
    assert cfg.execution.search_width == 5
    assert cfg.execution.search_explore_every == 4
