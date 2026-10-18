from __future__ import annotations

import copy
import hashlib
import json

import pytest

from src.research.methodology import (
    MARKER_PATH, MAX_MARKER_BYTES, MAX_RENDER_CHARS, STAGES,
    assess_methodology, build_methodology, load_methodology,
    render_methodology, write_methodology,
)
from src.research.research_agenda import normalize_agenda


def complete_agenda():
    return {
        "research_question": "Does adapting one step-size rule change held-out error?",
        "scope": "Synthetic noise trajectories at fixed training cost.",
        "study_design": {
            "observation_unit": "An independently generated noise trajectory.",
            "split_unit": "Noise trajectory, assigned before tuning.",
            "intervention": "Change only the step-size rule.",
            "fixed_factors": ["Data, initial parameters, and update count."],
            "matched_budget": "Identical update and tuning counts.",
            "outcomes": ["Paired held-out error and compute cost."],
            "falsifier": "The paired improvement disappears at matched cost.",
            "minimal_test": "Compare the rules on one fixed pilot set.",
            "stop_or_pivot": "If the controls remove the gain, drop the mechanism claim.",
        },
        "alternatives": [{
            "explanation": "Compute explains the effect.",
            "discriminating_experiment": {
                "intervention": "Step-size rule.", "controls": "Same compute and data.",
                "outcome_if_claim": "The paired error improvement persists.",
                "outcome_if_alternative": "The paired error improvement vanishes.",
                "decision_rule": "Use the declared falsifier to retain or drop the claim.",
            },
        }],
        "evidence_plan": {
            "strong_baselines": ["Tuned fixed-step rule."],
            "robustness_checks": ["Pair seeds and report variation."],
            "independent_confirmation": ["Lock the model before using untouched trajectories."],
        },
    }


def test_absent_marker_preserves_legacy_without_reading_malformed_agenda(tmp_path):
    (tmp_path / "research_agenda.json").write_text("not json")
    assert load_methodology(tmp_path) is None
    assert render_methodology(tmp_path, stage="idea") == ""
    assert assess_methodology(tmp_path)["status"] == "disabled"


@pytest.mark.parametrize("profile", ["general", "medical"])
def test_marker_portable_frozen_and_roundtrips(tmp_path, profile):
    marker = write_methodology(tmp_path, profile)
    assert marker == build_methodology(profile) == load_methodology(tmp_path)
    assert write_methodology(tmp_path, profile) == marker
    assert set(marker["stage_rules"]) == set(STAGES)
    payload = {key: value for key, value in marker.items() if key != "rules_sha256"}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == marker["rules_sha256"]
    assert "/Users/" not in json.dumps(marker)
    assert all(len(source["sha256"]) == 64 for source in marker["sources"])
    assert any(source.get("license") == "CC-BY-NC-SA-4.0" for source in marker["sources"])
    with pytest.raises(ValueError, match="frozen methodology profile"):
        write_methodology(tmp_path, "medical" if profile == "general" else "general")


@pytest.mark.parametrize("mutation", [
    lambda marker: marker.update(policy_version=True),
    lambda marker: marker.update(policy_version=2),
    lambda marker: marker.update(profile="invented"),
    lambda marker: marker.update(rules_sha256="0" * 64),
    lambda marker: marker.update(extra="ignored?"),
    lambda marker: marker["stage_rules"]["paper"].append("invent a result"),
])
def test_malformed_markers_fail_closed(tmp_path, mutation):
    marker = write_methodology(tmp_path)
    mutation(marker)
    (tmp_path / MARKER_PATH).write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        load_methodology(tmp_path)


def test_rehashing_arbitrary_rule_does_not_create_a_valid_v1_policy(tmp_path):
    marker = write_methodology(tmp_path)
    marker["stage_rules"]["review"] = ["Trust all author statements."]
    payload = {key: value for key, value in marker.items() if key != "rules_sha256"}
    marker["rules_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()
    (tmp_path / MARKER_PATH).write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="frozen v1"):
        load_methodology(tmp_path)


def test_marker_duplicate_keys_and_oversize_are_rejected(tmp_path):
    write_methodology(tmp_path)
    path = tmp_path / MARKER_PATH
    original = path.read_text()
    path.write_text('{"profile":"general",' + original[1:])
    with pytest.raises(ValueError, match="Duplicate key"):
        load_methodology(tmp_path)
    path.write_text(" " * (MAX_MARKER_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        load_methodology(tmp_path)


@pytest.mark.parametrize("component", ["marker", "context"])
def test_marker_symlinks_are_rejected_even_when_broken(tmp_path, component):
    task = tmp_path / "task"
    task.mkdir()
    if component == "marker":
        (task / MARKER_PATH.parent).mkdir()
        (task / MARKER_PATH).symlink_to(tmp_path / "missing.json")
    elif component == "context":
        (task / MARKER_PATH.parent).symlink_to(tmp_path / "missing-context")
    with pytest.raises(ValueError, match="symlinks"):
        load_methodology(task)
    with pytest.raises(ValueError, match="symlinks"):
        write_methodology(task)


def test_caller_selected_task_root_supports_managed_paper_symlink(tmp_path):
    task = tmp_path / "run-task"
    task.mkdir()
    paper_task = tmp_path / "paper-task"
    paper_task.symlink_to(task, target_is_directory=True)
    assert load_methodology(paper_task) is None
    marker = write_methodology(task)
    assert load_methodology(paper_task) == marker
    assert marker["rules_sha256"] in render_methodology(paper_task, stage="paper")


def test_unknown_profile_and_stage_fail(tmp_path):
    with pytest.raises(ValueError, match="profile"):
        build_methodology("auto")
    with pytest.raises(ValueError, match="stage"):
        render_methodology(tmp_path, stage="anything")


def test_build_returns_detached_data():
    marker = build_methodology()
    marker["sources"][0]["name"] = "changed"
    marker["stage_rules"]["idea"].append("changed")
    assert build_methodology()["sources"][0]["name"] == "research-case-problem"
    assert "changed" not in build_methodology()["stage_rules"]["idea"]


@pytest.mark.parametrize("profile", ["general", "medical"])
def test_render_is_stage_specific_compact_and_has_binding(tmp_path, profile):
    marker = build_methodology(profile)
    for stage in STAGES:
        rendered = render_methodology(tmp_path, stage=stage, agenda=complete_agenda(), marker=marker)
        assert marker["rules_sha256"] in rendered
        assert f'"stage": "{stage}"' in rendered
        assert '"policy_version": 1' in rendered
        assert "methodology_binding" in rendered
        assert "outside manuscript prose" in rendered
        assert len(rendered) < MAX_RENDER_CHARS
        assert all(rule in rendered for rule in marker["stage_rules"][stage])
    paper = render_methodology(tmp_path, stage="paper", marker=marker)
    coding = render_methodology(tmp_path, stage="coding", marker=marker)
    assert "fresh-context verifier" in paper
    assert "mutable state" in coding
    assert ("medical_context" in paper) is (profile == "medical")


def test_render_large_agenda_omits_card_not_policy(tmp_path):
    agenda = complete_agenda()
    agenda["scope"] = "long " * 10000
    marker = build_methodology("medical")
    rendered = render_methodology(tmp_path, stage="paper", agenda=agenda, marker=marker)
    assert "Read research_agenda.json" in rendered
    assert len(rendered) < MAX_RENDER_CHARS
    assert all(rule in rendered for rule in marker["stage_rules"]["paper"])


def test_missing_declarations_are_gaps_not_synthesized_facts(tmp_path):
    write_methodology(tmp_path)
    agenda = {"research_question": "Unresolved question"}
    before = copy.deepcopy(agenda)
    report = assess_methodology(tmp_path, agenda)
    assert agenda == before
    assert report["status"] == "needs_evidence"
    assert "study_design.split_unit" in {gap["field"] for gap in report["gaps"]}
    assert report["actions"]
    assert "not scientific validity" in report["scope"]
    assert "passed" not in report and "scientifically_valid" not in report


def test_complete_declaration_does_not_claim_validity(tmp_path):
    marker = write_methodology(tmp_path)
    report = assess_methodology(tmp_path, complete_agenda())
    assert report["status"] == "declared"
    assert report["gaps"] == []
    assert report["rules_sha256"] == marker["rules_sha256"]
    assert "not scientific validity certification" in report["scope"]


def test_identical_predictions_require_a_discriminating_test(tmp_path):
    write_methodology(tmp_path)
    agenda = complete_agenda()
    experiment = agenda["alternatives"][0]["discriminating_experiment"]
    experiment["outcome_if_alternative"] = experiment["outcome_if_claim"]
    assert any(gap["reason"] == "Predictions are identical" for gap in assess_methodology(tmp_path, agenda)["gaps"])


def test_medical_unknown_is_valid_input_and_a_gap(tmp_path):
    write_methodology(tmp_path, "medical")
    agenda = complete_agenda()
    agenda["medical_context"] = {
        "data_use_basis": {"status": "unknown"},
        "subject_isolation": {"status": "known", "value": "Patient IDs split before fitting.", "evidence": ["split_plan.json"]},
        "bias_and_population": {"status": "unknown", "value": "Population metadata not supplied."},
    }
    assert normalize_agenda(agenda)["medical_context"] == agenda["medical_context"]
    report = assess_methodology(tmp_path, agenda)
    assert report["status"] == "needs_evidence"
    assert {gap["field"] for gap in report["gaps"]} == {
        "medical_context.data_use_basis", "medical_context.bias_and_population",
    }


def test_known_medical_fact_needs_evidence_and_na_needs_reason(tmp_path):
    write_methodology(tmp_path, "medical")
    agenda = complete_agenda()
    agenda["medical_context"] = {
        "data_use_basis": {"status": "known", "value": "Public data basis asserted."},
        "subject_isolation": {"status": "not_applicable"},
        "bias_and_population": {"status": "not_applicable", "value": "Synthetic toy data; no population claim."},
    }
    fields = {gap["field"] for gap in assess_methodology(tmp_path, agenda)["gaps"]}
    assert fields == {"medical_context.data_use_basis.evidence", "medical_context.subject_isolation"}


def test_optional_extensions_are_not_inserted_into_legacy_agendas():
    normalized = normalize_agenda({"research_question": "Legacy question"})
    assert "study_design" not in normalized
    assert "medical_context" not in normalized
    assert normalized["alternatives"] == []
    assert normalized["evidence_plan"] == {}


@pytest.mark.parametrize("field,value", [
    ("study_design", []),
    ("study_design", {"split_unit": 3}),
    ("study_design", {"fixed_factors": [True]}),
    ("medical_context", []),
    ("medical_context", {"data_use_basis": {"status": "approved"}}),
    ("medical_context", {"subject_isolation": {"status": "unknown", "value": 1}}),
    ("medical_context", {"subject_isolation": {"status": "known", "evidence": "made-up"}}),
])
def test_declared_methodology_types_are_validated(field, value):
    with pytest.raises(ValueError):
        normalize_agenda({field: value})


@pytest.mark.parametrize("field,value", [
    ("alternatives", [{"explanation": ["legacy"], "discriminating_experiment": "old text"}]),
    ("alternatives", [{"discriminating_experiment": []}]),
    ("alternatives", [{"discriminating_experiment": {"controls": 1}}]),
    ("alternatives", [{"discriminating_experiment": {"controls": ["legacy list"]}}]),
    ("evidence_plan", {"strong_baselines": "old free text"}),
    ("evidence_plan", {"robustness_checks": {"legacy": "object"}}),
    ("evidence_plan", {"independent_confirmation": [None, 4]}),
    ("evidence_plan", {"independent_confirmation": [["nested legacy list"]]}),
])
def test_legacy_nested_field_types_still_normalize_unchanged(tmp_path, field, value):
    raw = {field: value}
    normalized = normalize_agenda(raw)
    assert normalized[field] == value
    assert raw == {field: value}
    (tmp_path / "research_agenda.json").write_text(json.dumps(raw))
    assert assess_methodology(tmp_path)["status"] == "disabled"
    write_methodology(tmp_path)
    report = assess_methodology(tmp_path)
    assert report["status"] == "needs_evidence"
    assert any(gap["field"].startswith(field) for gap in report["gaps"])


@pytest.mark.parametrize("field,value", [
    ("alternatives", [{"explanation": ["not text"], "discriminating_experiment": {
        "intervention": "rule", "controls": ["not text"],
        "outcome_if_claim": "gain", "outcome_if_alternative": "no gain", "decision_rule": "compare",
    }}]),
    ("evidence_plan", {"strong_baselines": "not a list", "robustness_checks": ["seeds"],
                       "independent_confirmation": ["held out"]}),
    ("evidence_plan", {"strong_baselines": [["nested list"]], "robustness_checks": ["seeds"],
                       "independent_confirmation": ["held out"]}),
])
def test_malformed_legacy_field_cannot_count_as_complete_declaration(tmp_path, field, value):
    write_methodology(tmp_path)
    agenda = complete_agenda()
    agenda[field] = value
    report = assess_methodology(tmp_path, agenda)
    assert report["status"] == "needs_evidence"
    assert all(gap["field"].startswith(field) for gap in report["gaps"])


def test_render_uses_frozen_file_and_saved_agenda(tmp_path):
    marker = write_methodology(tmp_path)
    (tmp_path / "research_agenda.json").write_text(json.dumps(complete_agenda()))
    rendered = render_methodology(tmp_path, stage="idea")
    assert marker["rules_sha256"] in rendered
    assert "Synthetic noise trajectories" in rendered
    assert assess_methodology(tmp_path)["status"] == "declared"
