"""CPU development evaluator; train only the small MIL head using bag labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

try:
    from .protocol import EvaluationProtocol, load_protocol
except ImportError:  # Direct execution from a copied task directory.
    from protocol import EvaluationProtocol, load_protocol


def load_candidate(program_path):
    spec = importlib.util.spec_from_file_location("mil_candidate", program_path)
    if spec is None or spec.loader is None:
        raise ValueError("Invalid candidate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build_model", None)):
        raise ValueError("Candidate must expose build_model(input_dim)")
    return module


def seed_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checked_logits(model, x):
    logits = model(x)
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(x),):
        raise ValueError("forward must return one unbounded logit per bag, shape (B,)")
    if not torch.isfinite(logits).all():
        raise ValueError("Candidate returned non-finite logits")
    return logits


def permute_instances(x, seed):
    rng = np.random.default_rng(seed)
    orders = np.stack([rng.permutation(x.shape[1]) for _ in range(len(x))])
    return x[torch.arange(len(x))[:, None], torch.from_numpy(orders)]


def check_mil_interface(model, x):
    """Reject observed use of row order or instance positions, on both views.

    This finite probe enforces the interface on sampled inputs. It is not a proof
    for all possible inputs, nor an isolation boundary against hostile programs.
    """
    sample = x[:min(17, len(x))]
    row_order = np.random.default_rng(913).permutation(len(sample))
    for length in (16, x.shape[1]):
        view = sample[:, :length]
        reference = checked_logits(model, view)
        reordered = checked_logits(model, view[row_order])
        if not torch.allclose(reordered, reference[row_order], rtol=1e-4, atol=1e-5):
            raise ValueError("MIL interface violation: predictions depend on batch row order; "
                             "predict each bag independently")
        shuffled = checked_logits(model, permute_instances(view, 919 + length))
        if not torch.allclose(shuffled, reference, rtol=1e-4, atol=1e-5):
            raise ValueError("MIL interface violation: predictions depend on instance positions; "
                             "use permutation-invariant pooling instead of slicing the first instances")


def score_views(model, x, y):
    model.eval()
    with torch.no_grad():
        check_mil_interface(model, x)
        # Labels remain outside model.forward. Fixed row and instance permutations
        # prevent the constructed order from exposing labels or witness positions.
        order = np.random.default_rng(929).permutation(len(x))
        inverse = np.argsort(order)
        small_x = permute_instances(x[order, :16], 937)
        large_x = permute_instances(x[order], 941)
        small = checked_logits(model, small_x).numpy()[inverse]
        large = checked_logits(model, large_x).numpy()[inverse]
        small_p = torch.sigmoid(torch.from_numpy(small).double()).numpy()
        large_p = torch.sigmoid(torch.from_numpy(large).double()).numpy()
    # AUROC uses raw logits. Float32 sigmoid can create saturation-induced ties.
    base_auc, diluted_auc = float(roc_auc_score(y, small)), float(roc_auc_score(y, large))
    positive, negative = y > 0, y == 0
    base_margin = float(small[positive].mean() - small[negative].mean())
    diluted_margin = float(large[positive].mean() - large[negative].mean())
    return {
        "base_auc": base_auc, "diluted_auc": diluted_auc,
        "worst_auc": min(base_auc, diluted_auc), "auc_change": diluted_auc - base_auc,
        "positive_probability_change": float((large_p[positive] - small_p[positive]).mean()),
        "negative_probability_change": float((large_p[negative] - small_p[negative]).mean()),
        "positive_logit_change": float((large[positive] - small[positive]).mean()),
        "negative_logit_change": float((large[negative] - small[negative]).mean()),
        "base_logit_margin": base_margin, "diluted_logit_margin": diluted_margin,
        "logit_margin_change": diluted_margin - base_margin,
    }, small, large


def fit(module, train_x, train_y, val_x, val_y, *, seed, noise_rate, protocol=None):
    protocol = protocol or EvaluationProtocol()
    seed_rngs(seed)
    rng = np.random.default_rng(seed)
    model = module.build_model(train_x.shape[-1]).cpu()
    if not isinstance(model, nn.Module):
        raise ValueError("build_model must return torch.nn.Module")
    named_parameters = dict(model.named_parameters())
    n_params = sum(p.numel() for p in named_parameters.values())
    trainable = {name: p for name, p in named_parameters.items() if p.requires_grad}
    if not 1 <= n_params <= protocol.max_parameters:
        raise ValueError(f"Parameter budget exceeded: {n_params} > {protocol.max_parameters}")
    if not trainable:
        raise ValueError("Model must have at least one trainable parameter")
    if len(train_x) < protocol.batch_size:
        raise ValueError("batch_size exceeds the number of training bags")
    n_flips = round(len(train_y) * noise_rate)
    labels = train_y.copy()
    flip_ids = np.random.default_rng(seed + 1000).permutation(len(labels))[:n_flips]
    labels[flip_ids] = 1 - labels[flip_ids]
    y = torch.from_numpy(labels)
    learning_rate = protocol.learning_rate
    if learning_rate is None:
        learning_rate = float(getattr(model, "learning_rate", 0.003))
    smoothing = protocol.label_smoothing
    if smoothing is None:
        smoothing = float(getattr(model, "label_smoothing", 0.0))
    if not 0.0001 <= learning_rate <= 0.01 or not 0 <= smoothing <= 0.2:
        raise ValueError("learning_rate or label_smoothing outside the declared bounds")
    optimizer = torch.optim.Adam(trainable.values(), lr=learning_rate, weight_decay=protocol.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_score, stale, selected_step = -1.0, 0, 0
    started = time.monotonic()
    stop_reason, steps_run = "step_limit", 0
    trace, connected_names = [], set()
    for step in range(1, protocol.max_steps + 1):
        if step > 1 and time.monotonic() - started >= protocol.max_fit_seconds:
            stop_reason = "time_limit"
            break
        model.train()
        indices = rng.choice(len(train_x), protocol.batch_size, replace=False)
        optimizer.zero_grad(set_to_none=True)
        logits = checked_logits(model, train_x[indices])
        target = y[indices] * (1 - smoothing) + 0.5 * smoothing
        loss = loss_fn(logits, target)
        loss.backward()
        connected_names.update(name for name, p in trainable.items() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(trainable.values(), protocol.gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        steps_run = step
        if step % protocol.check_every == 0 or step == protocol.max_steps:
            scores, _, _ = score_views(model, val_x, val_y)
            score = scores["worst_auc"]
            trace.append({"step": step, "loss": float(loss.detach()), **scores})
            if selected_step == 0 or score > best_score + protocol.min_delta:
                best_score, stale, selected_step = score, 0, step
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if protocol.patience is not None and stale >= protocol.patience:
                stop_reason = "early_stopping"
                break
    if selected_step == 0:
        raise ValueError("Time budget ended before the first validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "noise_rate": noise_rate, "seed": seed, "train_seconds": time.monotonic() - started,
        "steps_run": steps_run, "selected_step": selected_step, "stop_reason": stop_reason,
        "training_hit_step_cap": steps_run == protocol.max_steps,
        "selected_checkpoint_at_step_cap": selected_step == protocol.max_steps,
        "last_validation_step": trace[-1]["step"],
        "parameters": n_params, "total_parameters": n_params,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "observed_gradient_connected_parameters": sum(named_parameters[n].numel() for n in connected_names),
        "inactive_parameter_names": sorted(set(named_parameters) - connected_names),
        "frozen_parameter_names": sorted(set(named_parameters) - set(trainable)),
        "gradient_diagnostic_scope": "union of grad is not None across completed training steps; not effective capacity",
        "learning_rate": learning_rate, "label_smoothing": smoothing,
        "flipped_training_bags": n_flips, "trace": trace,
    }


def locate_data(search_data=None):
    candidates = [Path(search_data)] if search_data is not None else [Path(__file__).resolve().parent / "resources/search.npz"]
    if search_data is None and os.environ.get("TASK_SOURCE_DIR"):
        candidates.append(Path(os.environ["TASK_SOURCE_DIR"]) / "resources/search.npz")
    for path in candidates:
        if path.is_file():
            if path.name != "search.npz":
                raise ValueError("Development data must be named search.npz")
            manifest = json.loads((path.parent / "data_manifest.json").read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["search_sha256"]:
                raise ValueError("Search feature hash changed")
            return path
    raise FileNotFoundError("Prepare resources/search.npz before running the experiment")


def locate_protocol(explicit=None):
    if explicit is not None:
        return Path(explicit)
    candidates = [Path(__file__).resolve().parent / "evaluation_protocol.json"]
    if os.environ.get("TASK_SOURCE_DIR"):
        candidates.append(Path(os.environ["TASK_SOURCE_DIR"]) / "evaluation_protocol.json")
    return next((path for path in candidates if path.is_file()), None)


def evaluate(program_path, results_dir, *, holdout=None, seeds=None, protocol_path=None, search_data=None):
    started = time.monotonic()
    protocol_source = locate_protocol(protocol_path)
    protocol = load_protocol(protocol_source, seeds=seeds)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(protocol.num_threads)
    data_path = locate_data(search_data)
    with np.load(data_path, allow_pickle=False) as data:
        if set(data.files) != {"train_x", "train_y", "val_x", "val_y"}:
            raise ValueError("Search archive must contain only train/validation features and bag labels")
        train_x, val_x = torch.from_numpy(data["train_x"]), torch.from_numpy(data["val_x"])
        train_y, val_y = data["train_y"], data["val_y"]
    if train_x.ndim != 3 or val_x.ndim != 3 or train_x.shape[1] != 16 or val_x.shape[1] != 64:
        raise ValueError("Expected 16-instance training bags and paired 64-instance validation bags")
    for x, y in ((train_x, train_y), (val_x, val_y)):
        if len(x) != len(y) or y.ndim != 1 or set(np.unique(y)) != {0.0, 1.0} or not torch.isfinite(x).all():
            raise ValueError("Search data require finite features and both binary bag-label classes")
    test = np.load(holdout, allow_pickle=False) if holdout else None
    (output / "evaluation_protocol.json").write_text(json.dumps(protocol.to_dict(), indent=2) + "\n")
    rows, all_predictions = [], {}
    for seed in protocol.seeds:
        seed_rngs(seed)
        module = load_candidate(program_path)
        for rate in protocol.noise_rates:
            model, record = fit(module, train_x, train_y, val_x, val_y, seed=seed, noise_rate=rate, protocol=protocol)
            x, y = (torch.from_numpy(test["test_x"]), test["test_y"]) if test is not None else (val_x, val_y)
            scores, small, large = score_views(model, x, y)
            record.update(scores)
            rows.append(record)
            prefix = f"seed{seed}_noise{rate}"
            all_predictions[prefix + "_base_logits"] = small
            all_predictions[prefix + "_diluted_logits"] = large
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
        "protocol": protocol.to_dict(), "protocol_sha256": protocol.sha256,
        "protocol_source": str(protocol_source) if protocol_source else "legacy training defaults",
        "search_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "program_sha256": hashlib.sha256(Path(program_path).read_bytes()).hexdigest(),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_code_sha256": hashlib.sha256(Path(__file__).with_name("protocol.py").read_bytes()).hexdigest(),
        "ranking_score_space": "raw_logits",
        "probabilities_used_for": "positive/negative probability-shift diagnostics only",
        "interface_check": "sampled row equivariance and instance invariance at each checkpoint and final scoring",
        "prediction_order": "original bag order after fixed row and instance permutations",
        "conditions": [{k: v for k, v in row.items() if k != "trace"} for row in rows],
    }
    if test is not None:
        test.close()
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "correct.json").write_text(json.dumps({"correct": True, "message": "Development evaluation completed"}) + "\n")
    (output / "training_trace.json").write_text(json.dumps(rows, indent=2) + "\n")
    np.savez_compressed(output / "predictions.npz", **all_predictions)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--seeds", help="Legacy override; cannot be combined with a protocol file")
    parser.add_argument("--protocol", type=Path, help="Frozen development protocol JSON")
    parser.add_argument("--search-data", type=Path, help="Explicit search.npz with adjacent data_manifest.json")
    args = parser.parse_args()
    try:
        evaluate(args.program_path, args.results_dir, holdout=args.holdout,
                 seeds=tuple(int(x) for x in args.seeds.split(",")) if args.seeds else None,
                 protocol_path=args.protocol, search_data=args.search_data)
    except Exception as exc:
        out = Path(args.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "correct.json").write_text(json.dumps({"correct": False, "message": str(exc)}) + "\n")
        (out / "metrics.json").write_text(json.dumps({"combined_score": 0, "error": str(exc)}) + "\n")
        raise
