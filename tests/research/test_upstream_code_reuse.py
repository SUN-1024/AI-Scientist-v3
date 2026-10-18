"""Behavioral checks for the actual V1/V2 adaptations and their runtime hook."""
import copy
import math

import pytest

from src.investigation.search_tree import SearchTree
from src.research.research_agenda import normalize_agenda
from src.research.upstream.v1_reflection import refine_agenda
from src.research.upstream.v2_journal import Journal, MetricValue
from tests.program_testutils import make_program_dir


def test_v1_reflects_then_stops_only_on_unchanged_json_with_full_frozen_context():
    initial = normalize_agenda({"schema_version": 1, "research_question": "Original question"})
    calls = []

    def request(prompt, round_index):
        calls.append((prompt, round_index))
        revised = {**initial, "research_question": "Narrow, testable question"}
        return {**revised, "_reflection": {"done": True, "changes": "Narrowed scope"}}

    result = refine_agenda(initial, frozen_context="FULL SOURCE and ACTUAL CODE receipt 123",
                           request=request, normalize=normalize_agenda)
    assert [index for _, index in calls] == [2, 3]
    assert all("FULL SOURCE and ACTUAL CODE receipt 123" in prompt for prompt, _ in calls)
    assert result.history[1]["changed"] is True  # A premature stop request was ignored.
    assert result.converged is True
    assert result.agenda["research_question"] == "Narrow, testable question"
    assert "_reflection" not in result.agenda
    assert initial["research_question"] == "Original question"
    assert result.to_dict()["independent_novelty_evidence"] is False


def test_v1_does_not_treat_done_marker_in_evidence_as_convergence():
    initial = normalize_agenda({"proposed_contribution": 'Quoted source says "I am done"'})
    calls = []

    def request(prompt, round_index):
        calls.append(round_index)
        return copy.deepcopy(initial)

    result = refine_agenda(initial, frozen_context='Source: "I am done"', request=request,
                           normalize=normalize_agenda, num_reflections=4)
    assert calls == [2, 3, 4]
    assert result.converged is False
    assert result.to_dict()["status"] == "reflection_budget_exhausted"


def test_v1_invalid_reflection_or_provider_failure_cannot_become_valid_proposal():
    initial = normalize_agenda({})
    with pytest.raises(ValueError, match="boolean"):
        refine_agenda(initial, frozen_context="frozen", normalize=normalize_agenda,
                      request=lambda *_: {**initial, "_reflection": {"done": "true"}})

    def failure(*_):
        raise OSError("Provider transport failed")

    with pytest.raises(OSError, match="Provider transport failed"):
        refine_agenda(initial, frozen_context="frozen", normalize=normalize_agenda, request=failure)


def test_v2_metric_ordering_handles_loss_and_rejects_noncomparable_results():
    a = MetricValue({"dataset_a": 0.2, "dataset_b": 0.4}, maximize=False, name="loss")
    b = MetricValue({"dataset_a": 0.5, "dataset_b": 0.6}, maximize=False, name="loss")
    assert max([a, b]) == a
    assert b > MetricValue(None, maximize=False, name="loss")
    with pytest.raises(ValueError, match="different names"):
        _ = a > MetricValue(0.9, maximize=True, name="accuracy")
    with pytest.raises(ValueError, match="different dataset sets"):
        _ = a > MetricValue({"dataset_c": 0.1}, maximize=False, name="loss")
    for value in (math.nan, math.inf, True, {"a": None}):
        with pytest.raises(ValueError, match="finite"):
            MetricValue(value, maximize=True, name="score")


def test_v2_journal_uses_real_quality_flags_replication_roles_and_full_evidence():
    rows = {
        "root": {"node_id": "root", "parent_id": None, "score": 0.6, "correct": True,
                 "analysis": "Working baseline", "visual_feedback": ["Stable curve"]},
        "repeat": {"node_id": "repeat", "parent_id": "root", "score": 0.99,
                   "correct": True, "is_replication": True},
        "bad_plot": {"node_id": "bad_plot", "parent_id": "root", "score": 0.9,
                     "correct": True, "plot_check": "failed"},
        "failure": {"node_id": "failure", "parent_id": "root", "score": 0.1, "correct": False},
        "repair": {"node_id": "repair", "parent_id": "failure", "score": None, "correct": False},
    }
    journal = Journal.from_search_records(rows)
    assert journal.get_best_node().id == "root"
    assert journal.get_node_by_id("repair").stage_name == "debug"
    assert journal.get_node_by_id("repair").debug_depth == 1
    assert [node.id for node in journal.debug_candidates(max_depth=2)] == ["repair"]
    assert journal.debug_candidates(max_depth=1) == []
    evidence = next(row for row in journal.selection_evidence() if row["node_id"] == "root")
    assert evidence["metric_value"] == 0.6
    assert evidence["analysis"] == "Working baseline"
    assert evidence["visual_feedback"] == ["Stable curve"]


def test_v2_journal_rejects_parent_cycles():
    rows = {name: {"node_id": name, "parent_id": parent, "score": None, "correct": False}
            for name, parent in (("a", "b"), ("b", "a"))}
    with pytest.raises(ValueError, match="parent cycle"):
        Journal.from_search_records(rows)


def test_search_tree_consumes_v2_debug_selection_and_persists_lease(tmp_path):
    task, results = tmp_path / "task", tmp_path / "results"
    baseline, failed = task / "program_0", results / "failed/program_1"
    make_program_dir(baseline, content="print('baseline')\n")
    make_program_dir(failed, content="print('failed implementation')\n")
    tree = SearchTree(results, task)
    manifest = {"combined_score": 0.1, "correct": False}
    tree.choose(1, manifest, baseline)
    tree.record(cycle=1, idea={"idea_id": "x", "hypothesis": "Test mechanism"},
                candidate=(failed, "src/program.py", 0.2, False), exit_code=0)
    selection = tree.choose(2, manifest, baseline)
    assert selection["node_id"] == "cycle_0001"
    assert selection["journal"]["upstream"] == "AI-Scientist-v2"
    assert selection["journal"]["stage"] == "debug"
    assert selection["journal"]["debug_depth"] == 1
    resumed = SearchTree(results, task).choose(2, manifest, baseline)
    assert resumed["node_id"] == selection["node_id"]
