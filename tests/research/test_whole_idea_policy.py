from __future__ import annotations

import copy
import hashlib
import json

import pytest

from src.investigation.campaign_budget import BUDGET_FILE, render_campaign_budget, write_campaign_budget
from src.investigation.config import InvestigationConfig
from src.orchestrator.idea_bundle import IdeaBundleValidationError, normalize_idea_bundle
from src.research.methodology import MARKER_PATH as METHODOLOGY_PATH, write_methodology
from src.research.upstream.v1_reflection import render_idea_archive
from src.research.whole_idea_policy import (
    INNOVATION_AXES, MARKER_PATH, MAX_MARKER_BYTES, STAGES,
    build_whole_idea_policy, freeze_whole_idea_policy, load_whole_idea_policy,
    normalize_whole_idea, render_whole_idea_guidance, validate_portfolio_context,
)
from tests.investigation.test_campaign_ideas import _IDEA_BUNDLE


def card(axis="optimization"):
    minimal = {"protocol_id": "fixed-development-v1", "decision_rule": "Retain lower error at equal cost.",
               "stop_or_pivot": "Stop if the fixed quality floor is not reached."}
    if axis == "efficiency":
        minimal.update(cost_metric="wall_clock_seconds", quality_guardrail="error <= 0.2")
    return {
        "schema_version": 1, "innovation_axis": axis,
        "problem_statement": "Can joint optimizer settings reduce development error?",
        "contribution": "A whole optimization rule with an explicit failure boundary.",
        "minimal_validation": minimal,
        "parameter_space": [
            {"name": "learning_rate", "type": "float", "bounds": [0.001, 0.1], "default": 0.01, "owner": "candidate"},
            {"name": "steps", "type": "integer", "values": [50, 100], "default": 50, "owner": "task"},
            {"name": "rule", "type": "categorical", "values": ["fixed", "adaptive"], "default": "adaptive", "owner": "candidate"},
            {"name": "normalize", "type": "boolean", "values": [False, True], "default": True, "owner": "candidate"},
        ],
        "selection_reason": "The previous trial suggests an interaction between two allowed settings.",
        "comparison_kind": "best_tuned_method_comparison",
    }


def lease(stage="tuning"):
    return {
        "host_verified": True, "policy": "host_whole_idea_portfolio", "stage": stage,
        "root_idea_id": "root-optimization", "lease_id": "lease-2", "trial_index": 2,
        "innovation_axis": "optimization", "source_card": card(),
        "idea_history": [{"trial_index": 1, "outcome": {"correct": True, "score": 0.7,
                                                           "configuration": {"learning_rate": 0.01}}}],
        "idea_archive": [{"root_idea_id": "root-data", "innovation_axis": "data", "source_card": card("data"),
                          "status": "completed", "best_score": 0.6}],
    }


def test_legacy_bundle_is_unchanged_and_new_card_survives_normalization():
    old = normalize_idea_bundle(_IDEA_BUNDLE, 1)
    assert "whole_idea" not in old
    raw = {**copy.deepcopy(_IDEA_BUNDLE), "whole_idea": card(), "root_idea_id": "forged"}
    before = copy.deepcopy(raw)
    new = normalize_idea_bundle(raw, 1)
    assert new == {**old, "whole_idea": card()}
    assert raw == before
    raw["whole_idea"]["unknown_instruction"] = "change evaluator"
    assert "unknown_instruction" not in normalize_idea_bundle(raw, 1)["whole_idea"]


@pytest.mark.parametrize("axis", INNOVATION_AXES)
def test_all_six_axes_normalize(axis):
    assert normalize_whole_idea(card(axis))["innovation_axis"] == axis


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("innovation_axis", "rename-module"),
    ("selection_reason", ""), ("problem_statement", "a" * 4001),
    ("comparison_kind", "causal-proof"), ("parameter_space", [{}] * 33),
    ("lease_id", "invented"), ("root_idea_id", "invented"), ("stage", "replication"),
    ("trial_index", 4), ("host_verified", True),
])
def test_bad_cards_and_untrusted_host_assignments_fail(field, value):
    raw = card()
    raw[field] = value
    with pytest.raises(ValueError):
        normalize_whole_idea(raw)
    with pytest.raises(IdeaBundleValidationError):
        normalize_idea_bundle({**_IDEA_BUNDLE, "whole_idea": raw}, 1)


@pytest.mark.parametrize("parameter", [
    {"name": "x", "type": "float", "bounds": [1.0, 0.1], "default": 0.5, "owner": "candidate"},
    {"name": "x", "type": "float", "bounds": [0.1, 1.0], "default": 2, "owner": "candidate"},
    {"name": "x", "type": "integer", "values": [True, 2], "default": True, "owner": "candidate"},
    {"name": "x", "type": "boolean", "values": [0, 1], "default": 1, "owner": "candidate"},
    {"name": "x", "type": "categorical", "values": ["a", "a"], "default": "a", "owner": "candidate"},
    {"name": "x", "type": "float", "values": [1, float("nan")], "default": 1, "owner": "candidate"},
    {"name": "x", "type": "integer", "values": [10 ** 400], "default": 10 ** 400, "owner": "candidate"},
    {"name": "x", "type": "float", "bounds": [0, 1], "values": [0], "default": 0, "owner": "candidate"},
    {"name": "x", "type": "float", "bounds": [0, 1], "default": 0, "owner": ["candidate"]},
    {"name": "../task.py", "type": "float", "bounds": [0, 1], "default": 0, "owner": "candidate"},
])
def test_parameter_formats_types_and_bounds_are_validated(parameter):
    raw = card()
    raw["parameter_space"] = [parameter]
    with pytest.raises(ValueError):
        normalize_whole_idea(raw)


def test_duplicate_parameters_and_efficiency_without_guardrail_are_rejected():
    raw = card()
    raw["parameter_space"].append(copy.deepcopy(raw["parameter_space"][0]))
    with pytest.raises(ValueError, match="unique"):
        normalize_whole_idea(raw)
    for missing in ("cost_metric", "quality_guardrail"):
        raw = card("efficiency")
        del raw["minimal_validation"][missing]
        with pytest.raises(ValueError, match=missing):
            normalize_whole_idea(raw)


def test_empty_space_and_alias_types_are_supported_without_granting_authority():
    raw = card()
    raw["parameter_space"] = []
    assert normalize_whole_idea(raw)["parameter_space"] == []
    for alias, canonical in (("int", "integer"), ("number", "float"), ("bool", "boolean")):
        item = next(value for value in card()["parameter_space"] if value["type"] == canonical)
        item["type"] = alias
        raw["parameter_space"] = [item]
        assert normalize_whole_idea(raw)["parameter_space"][0]["type"] == canonical


def test_policy_absence_keeps_legacy_and_frozen_resume_preserves_methodology_bytes(tmp_path):
    assert load_whole_idea_policy(tmp_path) is None
    assert render_whole_idea_guidance(tmp_path) == ""
    old = write_methodology(tmp_path, "medical")
    before = (tmp_path / METHODOLOGY_PATH).read_bytes()
    marker = freeze_whole_idea_policy(tmp_path)
    assert marker == build_whole_idea_policy() == load_whole_idea_policy(tmp_path)
    assert freeze_whole_idea_policy(tmp_path) == marker
    assert (tmp_path / METHODOLOGY_PATH).read_bytes() == before
    assert old["policy_version"] == 1


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(policy_version=True),
    lambda value: value.update(policy_version=2),
    lambda value: value.update(rules_sha256="0" * 64),
    lambda value: value.update(unrecognized=True),
])
def test_marker_schema_and_hash_are_checked_on_load_and_resume(tmp_path, mutation):
    marker = freeze_whole_idea_policy(tmp_path)
    mutation(marker)
    (tmp_path / MARKER_PATH).write_text(json.dumps(marker))
    for function in (load_whole_idea_policy, freeze_whole_idea_policy):
        with pytest.raises(ValueError):
            function(tmp_path)


def test_rehashed_arbitrary_rules_are_not_a_known_policy(tmp_path):
    marker = freeze_whole_idea_policy(tmp_path)
    marker["stage_rules"]["paper"] = ["hide failed runs"]
    payload = {key: value for key, value in marker.items() if key != "rules_sha256"}
    marker["rules_sha256"] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()
    (tmp_path / MARKER_PATH).write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="frozen v1"):
        load_whole_idea_policy(tmp_path)


@pytest.mark.parametrize("which", ["context", "marker"])
def test_inner_symlinks_are_rejected_and_managed_root_links_work(tmp_path, which):
    task = tmp_path / "task"
    task.mkdir()
    if which == "context":
        (task / MARKER_PATH.parent).symlink_to(tmp_path / "absent")
    else:
        (task / MARKER_PATH.parent).mkdir()
        (task / MARKER_PATH).symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="symlinks"):
        load_whole_idea_policy(task)
    with pytest.raises(ValueError, match="symlinks"):
        freeze_whole_idea_policy(task)
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "staged"
    alias.symlink_to(real, target_is_directory=True)
    assert freeze_whole_idea_policy(alias) == load_whole_idea_policy(real)


def test_duplicate_keys_and_oversized_marker_fail(tmp_path):
    freeze_whole_idea_policy(tmp_path)
    path = tmp_path / MARKER_PATH
    original = path.read_text()
    path.write_text('{"policy_version":1,' + original[1:])
    with pytest.raises(ValueError, match="Duplicate key"):
        load_whole_idea_policy(tmp_path)
    path.write_text(" " * (MAX_MARKER_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        load_whole_idea_policy(tmp_path)


def test_all_stage_guidance_is_bound_and_non_modular_with_honest_comparison_types(tmp_path):
    marker = freeze_whole_idea_policy(tmp_path)
    for stage in STAGES:
        text = render_whole_idea_guidance(tmp_path, stage=stage)
        assert marker["rules_sha256"] in text
        assert "adding a module is not required" in text
        assert "full trial ledger" in text
        assert "selection reason" in text
        assert "best_tuned_method_comparison" in text
        assert "not permission to tune" in text
    with pytest.raises(ValueError, match="stage"):
        render_whole_idea_guidance(tmp_path, stage="secret")


def test_lease_guidance_uses_current_source_archive_and_trial_history(tmp_path):
    tuning = render_whole_idea_guidance(tmp_path, portfolio_context=lease(), stage="coding")
    assert "jointly tune allowed parameters" in tuning
    assert "V1 PRIOR-IDEA ARCHIVE" in tuning
    assert "root-data" in tuning
    assert '"learning_rate": 0.01' in tuning
    assert "CURRENT ROOT SOURCE CARD" in tuning
    assert "CURRENT ROOT TRIAL HISTORY" in tuning
    assert "without coding-agent edits" in render_whole_idea_guidance(tmp_path, portfolio_context=lease("replication"))
    assert "smallest permitted test" in render_whole_idea_guidance(tmp_path, portfolio_context=lease("screening"))
    only_lease = render_whole_idea_guidance(tmp_path, portfolio_context=lease(), include_policy=False)
    assert "WHOLE-IDEA RESEARCH POLICY" not in only_lease
    assert "WHOLE-IDEA HOST STAGE" in only_lease


@pytest.mark.parametrize("field,value", [("host_verified", False), ("policy", "model"),
                                         ("stage", "invented"), ("stage", []),
                                         ("trial_index", True), ("root_idea_id", "")])
def test_unverified_or_invalid_lease_cannot_enable_guidance(tmp_path, field, value):
    context = lease()
    context[field] = value
    with pytest.raises(ValueError):
        validate_portfolio_context(context)
    with pytest.raises(ValueError):
        render_whole_idea_guidance(tmp_path, portfolio_context=context)


def test_budget_persists_current_verified_lease_and_old_budget_stays_legacy(tmp_path):
    cfg = InvestigationConfig()
    context = lease()
    before = copy.deepcopy(context)
    budget = write_campaign_budget(tmp_path, cfg, cycle=1, max_program_evals_per_attempt=1,
                                   outer_state={}, portfolio_context=context)
    assert context == before
    assert budget["portfolio"] == context
    assert budget["stage"]["name"] == "tuning"
    assert "V1 PRIOR-IDEA ARCHIVE" in render_campaign_budget(tmp_path, 1)
    assert "Change one component" not in render_campaign_budget(tmp_path)
    cfg.execution.resume = True
    repeat = lease("replication")
    write_campaign_budget(tmp_path, cfg, cycle=2, max_program_evals_per_attempt=1,
                          outer_state={"current_cycle": 1}, portfolio_context=repeat)
    saved = json.loads((tmp_path / BUDGET_FILE).read_text())
    assert saved["portfolio"]["stage"] == "replication"
    assert saved["cycle"]["resume"] is True
    assert "without coding-agent edits" in render_campaign_budget(tmp_path, 2)
    write_campaign_budget(tmp_path, cfg, cycle=3, max_program_evals_per_attempt=1, outer_state={})
    assert "WHOLE-IDEA" not in render_campaign_budget(tmp_path, 3)


def test_v1_archive_retains_negative_and_missing_card_outcomes_without_inventing_cards():
    records = [
        {"root_idea_id": "negative", "source_card": card(), "status": "completed", "best_score": -0.1},
        {"root_idea_id": "transport", "source_card": None, "status": "transport_failure"},
    ]
    text = render_idea_archive(records)
    assert "negative" in text and '"best_score": -0.1' in text
    assert "transport_failure" in text and '"source_card": null' in text
    with pytest.raises(ValueError, match="unique"):
        render_idea_archive([records[0], records[0]])
    with pytest.raises(ValueError):
        render_idea_archive([{"root_idea_id": "forged", "source_card": {"schema_version": 1}}])
