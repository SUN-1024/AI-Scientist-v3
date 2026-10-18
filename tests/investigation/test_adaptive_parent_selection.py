"""Adaptive stage actions select their actual parent, including score ties."""
from src.investigation.search_tree import SearchTree
from tests.program_testutils import make_program_dir


def test_incumbent_action_inherits_exact_incumbent_while_exploration_branches(tmp_path):
    task, results = tmp_path / "task", tmp_path / "results"
    seed, candidate = task / "program_0", results / "candidate/program_1"
    make_program_dir(seed, content="print('baseline')\n")
    make_program_dir(candidate, content="print('distinct mechanism')\n")
    tree = SearchTree(results, task)
    incumbent = {"combined_score": 0.5, "correct": True}
    first = tree.choose(1, incumbent, seed, selection_policy="incumbent")
    assert first["program_dir"] == str(seed.resolve())
    tree.record(cycle=1, idea={"idea_id": "first", "hypothesis": "Distinct mechanism"},
                candidate=(candidate, "src/program.py", 0.5, True), exit_code=0)
    # A score tie does not promote the new code to incumbent.
    exact = tree.choose(2, incumbent, seed, selection_policy="incumbent")
    assert exact["node_id"] == "root"
    exploratory = tree.choose(3, incumbent, seed, selection_policy="explore")
    assert exploratory["node_id"] == "cycle_0001"
    # Lease replay remains pinned even when the caller now prefers another parent.
    resumed = SearchTree(results, task).choose(3, incumbent, seed, selection_policy="incumbent")
    assert resumed["node_id"] == exploratory["node_id"]


def test_source_grouping_preserves_exact_incumbent_and_leased_parent(tmp_path):
    task, results = tmp_path / "task", tmp_path / "results"
    seed, copy = task / "program_0", results / "candidate/program_1"
    make_program_dir(seed, content="print('same source')\n")
    make_program_dir(copy, content="print('same source')\n")
    tree = SearchTree(results, task)
    incumbent = {"combined_score": 0.5, "correct": True}
    tree.choose(1, incumbent, seed, selection_policy="incumbent")
    tree.record(cycle=1, idea={"idea_id": "copy", "hypothesis": "Repeated source"},
                candidate=(copy, "src/program.py", 0.6, True), exit_code=0)
    # The highest-ranked representative must not replace an exact incumbent path.
    exact = tree.choose(2, incumbent, seed, selection_policy="incumbent")
    assert exact["node_id"] == "root"
    # Root shares the representative's hash, so it must not enter the pool twice.
    exploratory = tree.choose(3, incumbent, seed, selection_policy="explore")
    assert exploratory["node_id"] == "cycle_0001"
    assert exploratory["program_dir"] == str(copy.resolve())
    # A pending lease remains pinned to its exact node and path on resume.
    resumed = SearchTree(results, task).choose(3, incumbent, seed, selection_policy="incumbent")
    assert resumed == exploratory
