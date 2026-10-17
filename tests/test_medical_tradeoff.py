"""A high scalar score must not hide damage to segmentation utility."""

from pathlib import Path
import runpy

import pytest

assess = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           "scripts/research/assess_medical_tradeoff.py"))["assess"]


def metrics():
    return {"combined_score":0.90,"heldout_macro_dice":0.80,
            "forget_retrain_probability_mae":0.03,
            "original_heldout_macro_dice":0.70,"retrain_heldout_macro_dice":0.71,
            "original_forget_retrain_probability_mae":0.08,
            "finetune_heldout_macro_dice":0.79,"finetune_forget_retrain_probability_mae":0.05,
            "neggrad_heldout_macro_dice":0.78,"neggrad_forget_retrain_probability_mae":0.04}


def test_development_promise_is_separate_from_an_independent_scientific_claim():
    result = assess(metrics(), {"correct":True})
    assert result["promising_development"] is True
    assert result["scientific_claim_qualified"] is False
    assert result["equal_time_assessed"] is False


def test_high_scalar_score_does_not_hide_lower_utility():
    data = metrics();data["heldout_macro_dice"] = 0.77
    result = assess(data, {"correct":True})
    assert result["promising_development"] is False
    assert "finetune:heldout_utility_at_least_equal" in result["failed_comparisons"]


def test_same_forget_distance_does_not_count_as_improvement():
    data = metrics();data["forget_retrain_probability_mae"] = 0.04
    assert assess(data, {"correct":True})["promising_development"] is False


@pytest.mark.parametrize("value", [float("nan"),float("inf"),-1,1.1,True,None])
def test_missing_or_invalid_metric_blocks_promise(value):
    data = metrics();data["heldout_macro_dice"] = value
    assert assess(data, {"correct":True})["status"] == "invalid_or_missing_metrics"


def test_runtime_failure_or_undertrained_reference_blocks_promise():
    assert assess(metrics(), {"correct":False})["promising_development"] is False
    data = metrics();data["retrain_heldout_macro_dice"] = 0.49
    assert assess(data, {"correct":True})["promising_development"] is False
