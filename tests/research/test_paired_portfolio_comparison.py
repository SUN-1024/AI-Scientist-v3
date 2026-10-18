"""Final method contrasts use frozen paired observations, never an anchor score."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from src.research.paired_comparison import build_portfolio_comparisons, _score


def method(tmp_path, name, *, selected=False, control=False, order="correct"):
    labels = [0, 1, 0, 1]
    predictions = {"correct": [0, 1, 0, 1], "wrong": [1, 0, 1, 0], "ties": [0, 0, 0, 0]}[order]
    values = np.tile(predictions, (2, 3, 1))
    spec = {"schema_version": 1, "metric": "mean_condition_worst_view_auc",
            "source_metric": "combined_score", "prediction_key": "candidate",
            "dataset_sha256": "a" * 64, "sample_ids": ["test:0", "test:1", "test:2", "test:3"],
            "condition_ids": [0, 0.15], "view_ids": [16, 32, 64],
            "bootstrap_draws": 100, "bootstrap_seed": 7907}
    episodes = [{"seed": seed, "solution": {
        "evaluation_plane": "final", "smoke_only": False,
        "paired_comparison_spec": spec,
        "paired_predictions": {"labels": labels, "values": {"candidate": values.tolist()}},
    }} for seed in (101, 211)]
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps({"schema_version": 1, "episodes": episodes}))
    return {"method_id": name, "is_selected": selected, "is_control": control,
            "source_sha256": "b" * 64,
            "evaluation_evidence_path": str(path),
            "evaluation_evidence_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "final_manifest": {"correct": True, "metrics": {"combined_score": {"correct": 1.0, "wrong": 0.0, "ties": 0.5}[order]}},
            "actual_evaluations": {"screening": 1, "tuning": 4, "replication": 0}}


def collection(tmp_path, *, selected="correct", controls=("wrong", "ties")):
    return {"status": "completed", "set_sha256": "c" * 64,
            "selected_method_id": "proposal", "selected_proposal_root_id": "proposal",
            "methods": [method(tmp_path, "proposal", selected=True, order=selected),
                        *(method(tmp_path, f"control_{i}", control=True, order=value)
                          for i, value in enumerate(controls))]}


def rewrite(row, mutate):
    from pathlib import Path
    path = Path(row["evaluation_evidence_path"])
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))
    row["evaluation_evidence_sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_controls_are_compared_and_minimum_effect_is_grounded(tmp_path):
    data = collection(tmp_path)
    report = build_portfolio_comparisons(data)
    assert report["status"] == "verified"
    assert [row["estimate"] for row in report["comparisons"]] == [1, 0.5]
    assert report["aggregate_metrics"] == {"candidate_minus_controls_min": 0.5, "paired_controls_ci95_low_min": 0.5}
    assert report["matched_hpo_verified"] is False
    assert [row["control_method_id"] for row in report["comparisons"]] == ["control_0", "control_1"]
    assert data["selected_method_id"] == "proposal"


def test_negative_control_comparison_is_retained_without_reselection(tmp_path):
    data = collection(tmp_path, selected="ties", controls=("wrong", "correct"))
    report = build_portfolio_comparisons(data)
    assert report["status"] == "verified"
    assert report["aggregate_metrics"]["candidate_minus_controls_min"] == -0.5
    assert report["selected_method_id"] == "proposal"


def test_score_uses_worst_view_and_counts_ties():
    values = np.array([[[[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 0, 0]],
                        [[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]]])
    assert _score(values, np.array([1, 3]), np.array([0, 2])) == 0.25


def test_identity_contrast_has_zero_uncertainty_and_seed_order_does_not_matter(tmp_path):
    data = collection(tmp_path, controls=("correct",))
    rewrite(data["methods"][1], lambda payload: payload["episodes"].reverse())
    report = build_portfolio_comparisons(data)
    assert report["status"] == "verified"
    assert report["comparisons"][0]["ci95"] == [0, 0]


@pytest.mark.parametrize("field,value", [
    ("dataset_sha256", "d" * 64), ("sample_ids", ["test:1", "test:0", "test:2", "test:3"]),
    ("condition_ids", [0.15, 0]), ("view_ids", [64, 32, 16]), ("bootstrap_seed", 9000),
])
def test_mismatched_pairing_cannot_produce_positive_metrics(tmp_path, field, value):
    data = collection(tmp_path, controls=("wrong",))
    def mutate(payload):
        for episode in payload["episodes"]:
            episode["solution"]["paired_comparison_spec"][field] = value
    rewrite(data["methods"][1], mutate)
    report = build_portfolio_comparisons(data)
    assert report["status"] == "incomplete" and report["aggregate_metrics"] == {}
    assert "not identically paired" in report["gaps"][0]


def test_missing_spec_tamper_and_nonreproduced_final_score_are_gaps(tmp_path):
    data = collection(tmp_path, controls=("wrong",))
    data["methods"][1]["evaluation_evidence_sha256"] = "sha256:" + "0" * 64
    assert "hash changed" in build_portfolio_comparisons(data)["gaps"][0]
    data = collection(tmp_path, controls=("wrong",))
    data["methods"][1]["final_manifest"]["metrics"]["combined_score"] = 0.8
    assert "do not reproduce" in build_portfolio_comparisons(data)["gaps"][0]
    data = collection(tmp_path, controls=("wrong",))
    rewrite(data["methods"][1], lambda payload: payload["episodes"][0]["solution"].pop("paired_comparison_spec"))
    assert "unsupported" in build_portfolio_comparisons(data)["gaps"][0]


@pytest.mark.parametrize("state", ["partial", "no_proposal", "incorrect", "no_control"])
def test_incomplete_or_unqualified_collection_cannot_claim_superiority(tmp_path, state):
    data = collection(tmp_path)
    if state == "partial":
        data["status"] = "incomplete"
    elif state == "no_proposal":
        data["selected_proposal_root_id"] = None
    elif state == "incorrect":
        data["methods"][1]["final_manifest"]["correct"] = False
    else:
        data["methods"] = data["methods"][:1]
    original = deepcopy(data)
    report = build_portfolio_comparisons(data)
    assert report["status"] == "incomplete" and report["aggregate_metrics"] == {}
    assert data == original
