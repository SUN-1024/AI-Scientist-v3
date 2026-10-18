"""Artifact-only paired comparisons for a declared multi-view binary metric.

The caller must first verify the frozen final method collection. This module
reads its hash-bound predictions. It never selects, trains or calls a candidate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _observations(method: dict[str, Any]) -> dict[str, Any]:
    manifest = method.get("final_manifest")
    if not isinstance(manifest, dict) or manifest.get("correct") is not True:
        raise ValueError("A comparison method has no qualified final execution")
    path = Path(method["evaluation_evidence_path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("Final comparison evidence is missing or is a symlink")
    raw = path.read_bytes()
    bound_hash = method.get("evaluation_evidence_sha256")
    if not isinstance(bound_hash, str) or hashlib.sha256(raw).hexdigest() != bound_hash.removeprefix("sha256:"):
        raise ValueError("Final comparison evidence hash changed")
    evidence = json.loads(raw)
    episodes = evidence.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Final comparison needs recorded seed observations")
    records: dict[int, np.ndarray] = {}
    common = None
    labels = None
    for episode in episodes:
        seed = episode.get("seed")
        solution = episode.get("solution", {})
        spec = solution.get("paired_comparison_spec")
        if (type(seed) is not int or seed in records or not isinstance(spec, dict)
                or spec.get("schema_version") != 1
                or spec.get("metric") != "mean_condition_worst_view_auc"
                or spec.get("source_metric") != "combined_score"
                or spec.get("prediction_key") != "candidate"
                or solution.get("evaluation_plane") != "final"
                or solution.get("smoke_only") is not False):
            raise ValueError("Missing or unsupported prespecified paired-comparison evidence")
        draws = spec.get("bootstrap_draws")
        random_seed = spec.get("bootstrap_seed")
        if (type(draws) is not int or not 100 <= draws <= 10000
                or type(random_seed) is not int or not 0 <= random_seed < 2**32):
            raise ValueError("Invalid prespecified paired bootstrap settings")
        dataset = spec.get("dataset_sha256")
        samples = spec.get("sample_ids")
        conditions, views = spec.get("condition_ids"), spec.get("view_ids")
        if (not isinstance(dataset, str) or len(dataset) != 64
                or any(char not in "0123456789abcdef" for char in dataset)
                or not isinstance(samples, list) or not samples
                or any(not isinstance(item, (str, int)) or isinstance(item, bool) for item in samples)
                or len({str(item) for item in samples}) != len(samples)
                or not isinstance(conditions, list) or not conditions
                or len(set(map(str, conditions))) != len(conditions)
                or not isinstance(views, list) or not views
                or len(set(map(str, views))) != len(views)):
            raise ValueError("Invalid paired sample, condition, view or dataset identity")
        predictions = solution.get("paired_predictions", {})
        observed_labels = np.asarray(predictions.get("labels"))
        values = np.asarray(predictions.get("values", {}).get("candidate"), dtype=float)
        if (observed_labels.shape != (len(samples),)
                or set(observed_labels.tolist()) != {0, 1}
                or values.shape != (len(conditions), len(views), len(samples))
                or not np.isfinite(values).all()):
            raise ValueError("Invalid paired labels or prediction shape")
        if common is None:
            common, labels = spec, observed_labels
        elif common != spec or labels is None or not np.array_equal(labels, observed_labels):
            raise ValueError("Pairing identity changed between seed observations")
        records[seed] = values
    if labels is None:
        raise ValueError("Final comparison has no binary observations")
    seeds = sorted(records)
    values = np.stack([records[seed] for seed in seeds])
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    point = _score(values, positive, negative)
    declared = manifest.get("metrics", {}).get("combined_score")
    if (isinstance(declared, bool) or not isinstance(declared, (float, int))
            or not math.isfinite(declared) or not math.isclose(point, declared, abs_tol=1e-10)):
        raise ValueError("Predictions do not reproduce the sealed final score")
    return {"spec": common, "labels": labels, "seeds": seeds, "values": values, "score": point}


def _score(values: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    delta = values[..., positive, None] - values[..., None, negative]
    auc = ((delta > 0) + 0.5 * (delta == 0)).mean(axis=(-2, -1))
    return float(auc.min(axis=-1).mean())


def _paired_effect(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if (candidate["spec"] != control["spec"] or candidate["seeds"] != control["seeds"]
            or not np.array_equal(candidate["labels"], control["labels"])):
        raise ValueError("Candidate and control final observations are not identically paired")
    spec, labels = candidate["spec"], candidate["labels"]
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(spec["bootstrap_seed"])
    n_seeds = len(candidate["seeds"])
    differences = []
    for _ in range(spec["bootstrap_draws"]):
        seeds = rng.integers(0, n_seeds, n_seeds)
        positive = rng.choice(pos, len(pos), replace=True)
        negative = rng.choice(neg, len(neg), replace=True)
        differences.append(_score(candidate["values"][seeds], positive, negative)
                           - _score(control["values"][seeds], positive, negative))
    return {
        "metric": spec["metric"], "estimate": candidate["score"] - control["score"],
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "draws": spec["bootstrap_draws"], "bootstrap_seed": spec["bootstrap_seed"],
        "training_seeds": candidate["seeds"], "sample_count": len(labels),
        "method": "paired stratified sample resampling and training-seed cluster resampling; conditions and views remain paired",
        "interval_scope": "marginal percentile interval for this prespecified pair; not a simultaneous confidence band",
    }


def build_portfolio_comparisons(verified_collection: dict[str, Any]) -> dict[str, Any]:
    """Derive all declared control contrasts from an already verified collection."""
    result: dict[str, Any] = {
        "schema_version": 1, "status": "incomplete", "comparisons": [], "gaps": [],
        "aggregate_metrics": {}, "set_sha256": verified_collection.get("set_sha256"),
        "selected_method_id": verified_collection.get("selected_method_id"),
        "selected_proposal_root_id": verified_collection.get("selected_proposal_root_id"),
        "selection_basis": "development-frozen method set; final observations never select a method",
        "matched_hpo_verified": False,
        "scope": "Declared within-dataset paired observations; not evidence of patient-level or external generalization",
    }
    if verified_collection.get("status") != "completed":
        result["gaps"].append("The frozen final method collection is incomplete")
        return result
    methods = verified_collection.get("methods", [])
    selected = [method for method in methods if method.get("is_selected") is True]
    controls = [method for method in methods if method.get("is_control") is True]
    if len(selected) != 1 or not controls:
        result["gaps"].append("Exactly one frozen selected method and at least one declared control are required")
        return result
    if not result["selected_proposal_root_id"]:
        result["gaps"].append("Development selected a control or bootstrap; there is no selected proposal superiority claim")
    try:
        candidate = _observations(selected[0])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["gaps"].append(str(exc))
        return result
    for method in controls:
        try:
            comparison = _paired_effect(candidate, _observations(method))
            result["comparisons"].append({
                "candidate_method_id": selected[0]["method_id"],
                "control_method_id": method["method_id"],
                "candidate_source_sha256": selected[0]["source_sha256"],
                "control_source_sha256": method["source_sha256"],
                "candidate_evidence_sha256": selected[0]["evaluation_evidence_sha256"],
                "control_evidence_sha256": method["evaluation_evidence_sha256"],
                "control_actual_evaluations": method.get("actual_evaluations"),
                **comparison,
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["gaps"].append(f"{method.get('method_id')}: {exc}")
    if not result["gaps"]:
        result["status"] = "verified"
        result["aggregate_metrics"] = {
            "candidate_minus_controls_min": min(row["estimate"] for row in result["comparisons"]),
            "paired_controls_ci95_low_min": min(row["ci95"][0] for row in result["comparisons"]),
        }
    return result
