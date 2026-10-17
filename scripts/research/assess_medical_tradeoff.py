#!/usr/bin/env python3
"""Classify a native medical-development result without altering its score."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def assess(metrics: dict, correct: dict) -> dict:
    required = ("combined_score", "heldout_macro_dice", "forget_retrain_probability_mae",
                "original_heldout_macro_dice", "retrain_heldout_macro_dice",
                "original_forget_retrain_probability_mae",
                "finetune_heldout_macro_dice", "finetune_forget_retrain_probability_mae",
                "neggrad_heldout_macro_dice", "neggrad_forget_retrain_probability_mae")
    invalid = [key for key in required if isinstance(metrics.get(key), bool)
               or not isinstance(metrics.get(key), (int, float))
               or not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1]
    if invalid:
        return {"status":"invalid_or_missing_metrics", "invalid":invalid,
                "promising_development":False,"scientific_claim_qualified":False}
    reference_quality = all(metrics[key] >= 0.50 for key in
                            ("original_heldout_macro_dice", "retrain_heldout_macro_dice"))
    comparisons = {}
    for name in ("original", "finetune", "neggrad"):
        dice, mae = name + "_heldout_macro_dice", name + "_forget_retrain_probability_mae"
        comparisons[name] = {
            "candidate_heldout_dice":metrics["heldout_macro_dice"],
            "baseline_heldout_dice":metrics[dice],
            "heldout_utility_at_least_equal":metrics["heldout_macro_dice"] >= metrics[dice],
            "candidate_forget_mae":metrics["forget_retrain_probability_mae"],
            "baseline_forget_mae":metrics[mae],
            "forget_distance_lower":metrics["forget_retrain_probability_mae"] < metrics[mae],
        }
    failed = [name + ":" + condition for name, row in comparisons.items()
              for condition in ("heldout_utility_at_least_equal", "forget_distance_lower")
              if not row[condition]]
    execution_valid = correct.get("correct") is True
    promising = execution_valid and reference_quality and not failed
    return {"status":"promising_development" if promising else "tradeoff_not_demonstrated",
            "execution_valid":execution_valid,"reference_quality_pass":reference_quality,
            "combined_score":metrics["combined_score"],"comparisons":comparisons,
            "failed_comparisons":failed,"promising_development":promising,
            "equal_time_assessed":False,"independent_final_assessed":False,
            "scientific_claim_qualified":False,
            "scope":"Previously observed SEARCH benchmark; scalar improvement is not sufficient"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(json.loads((args.program / "metrics.json").read_text()),
                    json.loads((args.program / "correct.json").read_text()))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
