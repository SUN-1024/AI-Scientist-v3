"""Trusted evaluator. Owns labels, RNGs, batch schedules, checkpoints and metrics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import secrets
from pathlib import Path
import sys
import time

import numpy as np

try:
    from . import protocol as p
except ImportError:
    import protocol as p


def load_handle(path, pooling=None):
    spec = importlib.util.spec_from_file_location(f"mil_{time.monotonic_ns()}", path)
    if spec is None or spec.loader is None:
        raise ValueError("candidate module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if pooling is not None:
        module.POOLING = pooling
    if not callable(getattr(module, "handle", None)):
        raise ValueError("candidate must implement handle(request)")
    return module.handle


def auc(labels, scores):
    """Tie-aware AUROC from raw logits; suitable for this small balanced task."""
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("AUROC requires both classes")
    differences = positive[:, None] - negative[None, :]
    return float(((differences > 0) + 0.5 * (differences == 0)).mean())


def predict(handle, x):
    response = handle({"op": "predict", "x": x.tolist()})
    values = np.asarray(response["logits"], dtype=np.float64)
    if values.shape != (len(x),) or not np.isfinite(values).all():
        raise ValueError("candidate must return one finite raw logit per bag")
    return values


def permute_instances(x, seed):
    rng = np.random.default_rng(seed)
    order = np.stack([rng.permutation(x.shape[1]) for _ in range(len(x))])
    return x[np.arange(len(x))[:, None], order]


def _assert_same_predictions(expected, actual, check):
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise ValueError(f"MIL interface violation: {check}")


def _checked_pointwise_predictions(handle, view, rng):
    """Bind all finite invariance checks to the actual scored full-batch values."""
    values = predict(handle, view)
    size = len(view)
    row_order = rng.permutation(size)
    reordered = predict(handle, permute_instances(view[row_order], int(rng.integers(2**63))))
    _assert_same_predictions(values[row_order], reordered, "full-batch row/instance permutation")
    probe_ids = rng.choice(size, min(7, size), replace=False)
    probe = predict(handle, permute_instances(view[probe_ids], int(rng.integers(2**63))))
    _assert_same_predictions(values[probe_ids], probe, "scored full batch versus independent probe")
    chunk_order = rng.permutation(size)
    cursor = 0
    while cursor < size:
        count = min(int(rng.integers(1, 8)), size - cursor)
        ids = chunk_order[cursor:cursor + count]
        chunk = predict(handle, permute_instances(view[ids], int(rng.integers(2**63))))
        _assert_same_predictions(values[ids], chunk, "scored full batch versus independent chunks")
        cursor += count
    # Each scored value must also be obtained from its bag in isolation. A model
    # cannot pass by returning one function for full batches and another for probes.
    pointwise = np.empty_like(values)
    for index in rng.permutation(size):
        bag = permute_instances(view[index:index + 1], int(rng.integers(2**63)))
        pointwise[index] = predict(handle, bag)[0]
    _assert_same_predictions(values, pointwise, "scored full batch versus independent single bags")
    return pointwise


def score_views(handle, x, y, *, prediction_seed=None):
    # Scoring order is a fresh host-only seed. It is not sent in RPC requests.
    # Statistical training seeds remain frozen. Legitimate pointwise MIL predictions
    # do not change with this ordering; the seed is retained in host evidence.
    prediction_seed = secrets.randbits(63) if prediction_seed is None else int(prediction_seed)
    rng = np.random.default_rng(prediction_seed)
    order = rng.permutation(len(x))
    inverse = np.argsort(order)
    predictions = []
    for length in p.BAG_SIZES:
        view = permute_instances(x[order, :length], int(rng.integers(2**63)))
        predictions.append(_checked_pointwise_predictions(handle, view, rng)[inverse])
    predictions = np.stack(predictions)
    scores = [auc(y, values) for values in predictions]
    return {"auc_16": scores[0], "auc_32": scores[1], "auc_64": scores[2],
            "worst_auc": min(scores), "prediction_seed": prediction_seed}, predictions


def train_condition(handle, data, *, seed, noise_rate, smoke=False):
    """Only noisy training labels cross the candidate boundary. Host selects checkpoint."""
    started = time.monotonic()
    handle({"op": "reset", "seed": seed})
    train_y = data["train_y"].copy()
    n_flips = round(len(train_y) * noise_rate)
    flips = np.random.default_rng(seed + 1000).permutation(len(train_y))[:n_flips]
    train_y[flips] = 1 - train_y[flips]
    initialized = handle({"op": "init", "train_x": data["train_x"].tolist(),
                          "train_y": train_y.tolist(), "learning_rate": p.LEARNING_RATE,
                          "weight_decay": p.WEIGHT_DECAY,
                          "gradient_clip_norm": p.GRADIENT_CLIP_NORM})
    # This field is a diagnostic, not proof of effective model capacity.
    parameters = initialized.get("parameters")
    if type(parameters) is not int or not 1 <= parameters <= p.MAX_PARAMETERS:
        raise ValueError("candidate reported parameter count outside the task budget")
    rng = np.random.default_rng(seed)
    steps = p.SMOKE_STEPS if smoke else p.MAX_STEPS
    check_every = p.SMOKE_STEPS if smoke else p.CHECK_EVERY
    best_score, selected_step, trace = -1.0, 0, []
    for completed in range(0, steps, check_every):
        count = min(check_every, steps - completed)
        indices = [rng.choice(len(train_y), min(p.BATCH_SIZE, len(train_y)), replace=False).tolist()
                   for _ in range(count)]
        handle({"op": "train", "batch_indices": indices})
        scores, _ = score_views(handle, data["val_x"], data["val_y"])
        step = completed + count
        trace.append({"step": step, **scores})
        if scores["worst_auc"] > best_score:
            best_score, selected_step = scores["worst_auc"], step
            handle({"op": "snapshot", "name": "selected"})
    handle({"op": "restore", "name": "selected"})
    return {"seed": seed, "noise_rate": noise_rate, "steps_run": steps,
            "selected_step": selected_step, "flipped_training_bags": n_flips,
            "reported_parameters": parameters, "seconds": time.monotonic() - started,
            "selection_trace": trace}


def paired_bootstrap(candidate, control, labels, n_seeds, *, draws=2000):
    """Paired seed/bag bootstrap. Noise conditions and bag sizes remain paired."""
    candidate, control = np.asarray(candidate), np.asarray(control)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    rng = np.random.default_rng(7907)
    differences = []
    def score(values, pos, neg):
        delta = values[:, :, pos, None] - values[:, :, None, neg]
        by_condition_view = ((delta > 0) + 0.5 * (delta == 0)).mean(axis=(-2, -1))
        return np.min(by_condition_view, axis=1).mean()
    for _ in range(draws):
        selected_seeds = rng.integers(0, n_seeds, n_seeds)
        conditions = (selected_seeds[:, None] * len(p.NOISE_RATES)
                      + np.arange(len(p.NOISE_RATES))[None, :]).reshape(-1)
        pos = rng.choice(positive, len(positive), replace=True)
        neg = rng.choice(negative, len(negative), replace=True)
        differences.append(score(candidate[conditions], pos, neg) - score(control[conditions], pos, neg))
    point = score(candidate, positive, negative) - score(control, positive, negative)
    return {"comparison": "candidate_minus_lme", "metric": "mean_condition_worst_view_auc",
            "estimate": float(point), "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "draws": draws, "seed": 7907,
            "method": "paired stratified bag resampling and training-seed cluster resampling; noise/views paired",
            "scope": "within-dataset synthetic bags; few seeds and bags; no patient-level inference"}


def read_data(root, plane):
    path = root / "resources" / plane / "data.npz"
    manifest = json.loads((root / "data_manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[f"{plane}_sha256"]:
        raise ValueError("frozen resource hash changed")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    expected = {"train_x", "train_y", "val_x", "val_y"}
    if plane == "final":
        expected |= {"test_x", "test_y"}
    if set(data) != expected:
        raise ValueError("resource contains unexpected arrays")
    for split in ("train", "val", "test"):
        if split + "_x" not in data:
            continue
        x, y = data[split + "_x"], data[split + "_y"]
        length = 16 if split == "train" else 64
        if x.ndim != 3 or x.shape[1] != length or y.shape != (len(x),):
            raise ValueError("invalid feature or label shape")
        if not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError("finite features and both binary bag classes are required")
    return data


def evaluate(program_path, results_dir, *, plane, smoke=False, controls=("lme",)):
    root = Path(os.environ.get("TASK_SOURCE_DIR", Path(__file__).resolve().parent))
    data = read_data(root, plane)
    if smoke:
        # A development smoke checks wiring, not feasibility or scientific quality.
        for split in ("train", "val", "test"):
            if split + "_y" not in data:
                continue
            labels = data[split + "_y"]
            subset = np.concatenate([np.flatnonzero(labels == value)[:8] for value in (0, 1)])
            data[split + "_x"] = data[split + "_x"][subset]
            data[split + "_y"] = labels[subset]
    seeds = p.SEARCH_SEEDS if plane == "search" else p.FINAL_SEEDS
    noise_rates = p.NOISE_RATES
    if smoke:
        seeds, noise_rates = seeds[:1], p.NOISE_RATES[:1]
    target = "val" if plane == "search" else "test"
    rows, predictions, traces = {}, {}, {}
    for method in ("candidate", *controls):
        handle = load_handle(program_path) if method == "candidate" else load_handle(root / "reference.py", method)
        rows[method], predictions[method], traces[method] = [], [], []
        for seed in seeds:
            for rate in noise_rates:
                record = train_condition(handle, data, seed=seed, noise_rate=rate, smoke=smoke)
                scores, prediction = score_views(handle, data[target + "_x"], data[target + "_y"])
                traces[method].append(record.pop("selection_trace"))
                rows[method].append({**record, **scores})
                predictions[method].append(prediction)
    scores = {name: float(np.mean([row["worst_auc"] for row in records]))
              for name, records in rows.items()}
    if "lme" not in scores:
        raise ValueError("the frozen ordinary-LME comparator is required")
    comparison = {"comparison": "candidate_minus_lme", "estimate": scores["candidate"] - scores["lme"],
                  "ci95": [-1.0, 1.0], "method": "not computed in wiring smoke", "draws": 0}
    if not smoke:
        comparison = paired_bootstrap(predictions["candidate"], predictions["lme"], data[target + "_y"], len(seeds))
    # Both search and final require an exact declared mapping of finite scalar metrics.
    metrics = {"combined_score": scores["candidate"], "lme_score": scores["lme"],
               "candidate_minus_lme": comparison["estimate"],
               "paired_ci95_low": comparison["ci95"][0], "paired_ci95_high": comparison["ci95"][1],
               **{f"auc_{length}": float(np.mean([row[f"auc_{length}"] for row in rows["candidate"]]))
                  for length in p.BAG_SIZES}}
    episodes = []
    for seed in seeds:
        condition_ids = [index for index, row in enumerate(rows["candidate"]) if row["seed"] == seed]
        conditions = {name: [records[index] for index in condition_ids] for name, records in rows.items()}
        # solution is the supported, bounded JSON payload for host-authored observations.
        # Evaluation labels are saved only here; they never cross the candidate boundary.
        solution = {"host_authored": True, "evaluation_plane": plane, "evaluation_split": target,
                    "smoke_only": smoke, "noise_rates": list(noise_rates), "bag_sizes": list(p.BAG_SIZES),
                    "conditions": conditions, "paired_comparison": comparison,
                    "selection_traces": {name: [records[index] for index in condition_ids]
                                         for name, records in traces.items()},
                    "paired_predictions": {"labels": data[target + "_y"].tolist(),
                                           "values": {name: np.asarray(values)[condition_ids].tolist()
                                                      for name, values in predictions.items()}},
                    "selection_metric": "validation worst AUROC over paired views, per noise condition",
                    "ranking_score_space": "raw_logits",
                    "parameter_count_status": "candidate-reported diagnostic; source review required",
                    "scope": "new patch-disjoint synthetic PathMNIST bags, not patient/external generalization"}
        episodes.append({"seed": seed, "score": float(np.mean([row["worst_auc"] for row in conditions["candidate"]])),
                         "correct": True,
                         "duration_seconds": float(sum(row["seconds"] for records in conditions.values() for row in records)),
                         "solution": solution})
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "correct.json").write_text(json.dumps({"correct": True}) + "\n")
    (output / "evaluation_evidence.json").write_text(json.dumps({"schema_version": 1, "episodes": episodes}, indent=2) + "\n")
    if plane == "search" and os.environ.get("AGENTIC_TASK_QUALIFICATION") == "1":
        (output / "qualification.json").write_text(json.dumps({"schema_version": 1, "feasible": True}) + "\n")
    return metrics


def main(plane):
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    evaluate(args.program_path, args.results_dir, plane=plane,
             smoke=os.environ.get("AGENTIC_TASK_DRAFT_SMOKE") == "1")
