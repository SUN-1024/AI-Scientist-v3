"""Fixed-budget CPU evaluator. Only the small MIL head is trained each round."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

MAX_STEPS = 160
CHECK_EVERY = 20
PATIENCE = 3
MAX_FIT_SECONDS = 20
MAX_PARAMETERS = 20000
BATCH_SIZE = 32
NOISE_RATES = (0.0, 0.15)


def load_candidate(program_path):
    spec = importlib.util.spec_from_file_location("mil_candidate", program_path)
    if spec is None or spec.loader is None:
        raise ValueError("Invalid candidate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_model", None)):
        raise ValueError("Candidate must expose build_model(input_dim)")
    return module


def checked_logits(model, x):
    logits = model(x)
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(x),):
        raise ValueError("forward must return one unbounded logit per bag, shape (B,)")
    if not torch.isfinite(logits).all():
        raise ValueError("Candidate returned non-finite logits")
    return logits


def score_views(model, x, y):
    model.eval()
    with torch.no_grad():
        small = torch.sigmoid(checked_logits(model, x[:, :16])).numpy()
        large = torch.sigmoid(checked_logits(model, x)).numpy()
    base_auc, diluted_auc = float(roc_auc_score(y, small)), float(roc_auc_score(y, large))
    positive = y > 0
    return {
        "base_auc": base_auc, "diluted_auc": diluted_auc,
        "worst_auc": min(base_auc, diluted_auc),
        "positive_probability_change": float((large[positive] - small[positive]).mean()),
    }, small, large


def fit(module, train_x, train_y, val_x, val_y, *, seed, noise_rate):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = module.build_model(train_x.shape[-1]).cpu()
    if not isinstance(model, nn.Module):
        raise ValueError("build_model must return torch.nn.Module")
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if not 1 <= n_params <= MAX_PARAMETERS:
        raise ValueError(f"Parameter budget exceeded: {n_params} > {MAX_PARAMETERS}")
    n_flips = round(len(train_y) * noise_rate)
    # Fixed corruption indices are shared by all candidates for a given seed.
    labels = train_y.copy()
    flip_ids = np.random.default_rng(seed + 1000).permutation(len(labels))[:n_flips]
    labels[flip_ids] = 1 - labels[flip_ids]
    y = torch.from_numpy(labels)
    learning_rate = float(getattr(model, "learning_rate", 0.003))
    smoothing = float(getattr(model, "label_smoothing", 0.0))
    if not 0.0001 <= learning_rate <= 0.01 or not 0 <= smoothing <= 0.2:
        raise ValueError("learning_rate or label_smoothing outside the declared bounds")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_score, stale, selected_step = -1.0, 0, 0
    started = time.monotonic()
    stop_reason = "step_limit"
    trace = []
    for step in range(1, MAX_STEPS + 1):
        if step > 1 and time.monotonic() - started >= MAX_FIT_SECONDS:
            stop_reason = "time_limit"
            break
        model.train()
        indices = rng.choice(len(train_x), BATCH_SIZE, replace=False)
        optimizer.zero_grad(set_to_none=True)
        logits = checked_logits(model, train_x[indices])
        target = y[indices] * (1 - smoothing) + 0.5 * smoothing
        loss = loss_fn(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % CHECK_EVERY == 0:
            scores, _, _ = score_views(model, val_x, val_y)
            score = scores["worst_auc"]
            trace.append({"step": step, "loss": float(loss.detach()), **scores})
            if score > best_score + 0.001:
                best_score, stale, selected_step = score, 0, step
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= PATIENCE:
                stop_reason = "early_stopping"
                break
    if selected_step == 0:
        raise ValueError("Time budget ended before the first validation checkpoint")
    model.load_state_dict(best_state)
    return model, {"noise_rate": noise_rate, "seed": seed, "train_seconds": time.monotonic() - started,
                   "steps_run": step, "selected_step": selected_step, "stop_reason": stop_reason,
                   "parameters": n_params, "flipped_training_bags": n_flips, "trace": trace}


def locate_data():
    candidates = [Path(__file__).resolve().parent / "resources/search.npz"]
    if os.environ.get("TASK_SOURCE_DIR"):
        candidates.append(Path(os.environ["TASK_SOURCE_DIR"]) / "resources/search.npz")
    for path in candidates:
        if path.is_file():
            manifest = json.loads((path.parent / "data_manifest.json").read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["search_sha256"]:
                raise ValueError("Search feature hash changed")
            return path
    raise FileNotFoundError("Prepare resources/search.npz before running the experiment")


def evaluate(program_path, results_dir, *, holdout=None, seeds=(7,)):
    started = time.monotonic()
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    data = np.load(locate_data(), allow_pickle=False)
    train_x, val_x = torch.from_numpy(data["train_x"]), torch.from_numpy(data["val_x"])
    train_y, val_y = data["train_y"], data["val_y"]
    test = np.load(holdout, allow_pickle=False) if holdout else None
    module = load_candidate(program_path)
    rows = []
    all_predictions = {}
    for seed in seeds:
        for rate in NOISE_RATES:
            model, record = fit(module, train_x, train_y, val_x, val_y, seed=seed, noise_rate=rate)
            x, y = (torch.from_numpy(test["test_x"]), test["test_y"]) if test is not None else (val_x, val_y)
            scores, small, large = score_views(model, x, y)
            record.update(scores)
            rows.append(record)
            prefix = f"seed{seed}_noise{rate}"
            all_predictions[prefix + "_base"] = small
            all_predictions[prefix + "_diluted"] = large
            all_predictions[prefix + "_labels"] = y
            print(f"seed={seed} noise={rate:.2f} base_auc={scores['base_auc']:.4f} "
                  f"diluted_auc={scores['diluted_auc']:.4f} steps={record['steps_run']} "
                  f"seconds={record['train_seconds']:.2f}", flush=True)
    metrics = {
        "combined_score": float(np.mean([row["worst_auc"] for row in rows])),
        "base_auc": float(np.mean([row["base_auc"] for row in rows])),
        "diluted_auc": float(np.mean([row["diluted_auc"] for row in rows])),
        "seconds": time.monotonic() - started,
        "evaluation_split": "holdout" if test is not None else "validation",
        "setting": "synthetic PathMNIST bags; bag labels only; fixed handcrafted PCA features",
        "conditions": [{k: v for k, v in row.items() if k != "trace"} for row in rows],
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "correct.json").write_text(json.dumps({"correct": True, "message": "Fixed-budget evaluation completed"}) + "\n")
    (output / "training_trace.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", **all_predictions)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--seeds", default="7")
    args = parser.parse_args()
    try:
        evaluate(args.program_path, args.results_dir, holdout=args.holdout,
                 seeds=tuple(int(x) for x in args.seeds.split(",")))
    except Exception as exc:
        out = Path(args.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "correct.json").write_text(json.dumps({"correct": False, "message": str(exc)}) + "\n")
        (out / "metrics.json").write_text(json.dumps({"combined_score": 0, "error": str(exc)}) + "\n")
        raise
