"""Host-owned whole-idea screening through the existing sealed candidate RPC."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

try:
    from . import protocol as p
    from .configuration import configuration_from_response, validate_device_report
except ImportError:
    import protocol as p
    from configuration import configuration_from_response, validate_device_report

try:
    # prepare.py freezes these existing helpers beside the generated task.
    from shared_helpers import evaluator as shared
except ModuleNotFoundError:
    from examples.pathmnist_mil_harness import evaluator as shared

METRIC_NAMES = (
    "combined_score", "lme_score", "candidate_minus_lme", "paired_ci95_low", "paired_ci95_high",
    "auc_16", "auc_32", "auc_64", "candidate_train_rpc_seconds", "candidate_condition_seconds",
    "candidate_parameters", "candidate_training_examples",
)


def configuration_digest(configuration):
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def train_condition(handle, data, *, seed, noise_rate, configuration, smoke=False,
                    is_candidate=True, expected_model=None, expected_torch=None):
    """The host chooses noisy labels, minibatches, all checkpoints and step count."""
    started = time.monotonic()
    handle({"op": "reset", "seed": seed})
    if is_candidate:
        observed = configuration_from_response(handle({"op": "get_configuration"}))
        if configuration is not None and observed != configuration:
            raise ValueError("candidate configuration changed across conditions or resets")
        configuration = observed
    if configuration is None:
        raise ValueError("a validated training configuration is required")
    train_y = data["train_y"].copy()
    n_flips = round(len(train_y) * noise_rate)
    flips = np.random.default_rng(seed + 1000).permutation(len(train_y))[:n_flips]
    train_y[flips] = 1 - train_y[flips]
    request = {"op": "init", "train_x": data["train_x"].tolist(), "train_y": train_y.tolist(),
               **{key: configuration[key] for key in ("learning_rate", "weight_decay", "gradient_clip_norm")}}
    if is_candidate:
        request["configuration"] = configuration
    initialized = handle(request)
    parameters = initialized.get("parameters")
    if type(parameters) is not int or not 1 <= parameters <= p.MAX_PARAMETERS:
        raise ValueError("candidate measured parameter count is outside the task budget")
    device, trainable = None, parameters
    if is_candidate:
        trainable = initialized.get("trainable_parameters")
        if type(trainable) is not int or not 1 <= trainable <= parameters:
            raise ValueError("candidate trainable parameter count is invalid")
        device = validate_device_report(initialized.get("device"), expected_model=expected_model,
                                        expected_torch=expected_torch)
    rng = np.random.default_rng(seed)
    steps = p.SMOKE_STEPS if smoke else p.MAX_STEPS
    check_every = p.SMOKE_STEPS if smoke else p.CHECK_EVERY
    best_score, selected_step, trace, train_seconds, examples = -1.0, 0, [], 0.0, 0
    for completed in range(0, steps, check_every):
        count = min(check_every, steps - completed)
        indices = [rng.choice(len(train_y), min(configuration["batch_size"], len(train_y)),
                              replace=False).tolist() for _ in range(count)]
        before = time.monotonic()
        trained = handle({"op": "train", "batch_indices": indices})
        train_seconds += time.monotonic() - before
        if type(trained.get("steps")) is not int or trained["steps"] != count:
            raise ValueError("candidate did not report the host-requested number of optimizer steps")
        examples += sum(map(len, indices))
        scores, _ = shared.score_views(handle, data["val_x"], data["val_y"])
        step = completed + count
        trace.append({"step": step, **scores})
        if scores["worst_auc"] > best_score:
            best_score, selected_step = scores["worst_auc"], step
            handle({"op": "snapshot", "name": "selected"})
    handle({"op": "restore", "name": "selected"})
    return {"seed": seed, "noise_rate": noise_rate, "steps_run": steps, "selected_step": selected_step,
            "flipped_training_bags": n_flips, "reported_parameters": parameters,
            "reported_trainable_parameters": trainable, "configuration": configuration,
            "configuration_sha256": configuration_digest(configuration), "device": device,
            "host_train_rpc_seconds": train_seconds, "host_training_examples": examples,
            "seconds": time.monotonic() - started, "selection_trace": trace}


def evaluate(program_path, results_dir, *, plane, smoke=False):
    if plane not in {"search", "final"}:
        raise ValueError("plane must be search or final")
    # The native runtime supplies both. Never execute candidate source in the label-owning host.
    if not os.environ.get("AGENTIC_CANDIDATE_SOCKET") or str(program_path) != "/broker-code/candidate_proxy.py":
        raise ValueError("portfolio evaluation requires the existing sealed Harness candidate proxy")
    root = Path(os.environ.get("TASK_SOURCE_DIR", Path(__file__).resolve().parent))
    contract = json.loads((root / "task_contract.json").read_text())
    expected_model = contract["compute"][plane]["accelerator_model"]
    expected_torch = next(item["version"] for item in contract["provenance"]["dependencies"]
                          if item["name"] == "torch")
    handle = shared.load_handle(program_path)
    configuration = None
    data = shared.read_data(root, plane)
    data_manifest = json.loads((root / "data_manifest.json").read_text())
    if smoke:
        for split in ("train", "val", "test"):
            if split + "_y" in data:
                labels = data[split + "_y"]
                subset = np.concatenate([np.flatnonzero(labels == value)[:8] for value in (0, 1)])
                data[split + "_x"], data[split + "_y"] = data[split + "_x"][subset], labels[subset]
    seeds = p.SEARCH_SEEDS if plane == "search" else p.FINAL_SEEDS
    noise_rates = p.NOISE_RATES
    if smoke:
        seeds, noise_rates = seeds[:1], noise_rates[:1]
    target = "val" if plane == "search" else "test"
    rows, predictions, traces = {}, {}, {}
    for method in ("candidate", "lme"):
        active = handle if method == "candidate" else shared.load_handle(root / "reference.py", "lme")
        config = configuration if method == "candidate" else dict(p.DEFAULT_CONFIGURATION)
        rows[method], predictions[method], traces[method] = [], [], []
        for seed in seeds:
            for rate in noise_rates:
                record = train_condition(active, data, seed=seed, noise_rate=rate, configuration=config,
                                         smoke=smoke, is_candidate=method == "candidate",
                                         expected_model=expected_model, expected_torch=expected_torch)
                if method == "candidate":
                    configuration = config = record["configuration"]
                scores, prediction = shared.score_views(active, data[target + "_x"], data[target + "_y"])
                traces[method].append(record.pop("selection_trace"))
                rows[method].append({**record, **scores})
                predictions[method].append(prediction)
    scores = {name: float(np.mean([row["worst_auc"] for row in records])) for name, records in rows.items()}
    comparison = {"comparison": "candidate_minus_lme", "estimate": scores["candidate"] - scores["lme"],
                  "ci95": [-1.0, 1.0], "method": "not computed in wiring smoke", "draws": 0}
    if not smoke:
        comparison = shared.paired_bootstrap(predictions["candidate"], predictions["lme"],
                                              data[target + "_y"], len(seeds))
    candidate_rows = rows["candidate"]
    metrics = {"combined_score": scores["candidate"], "lme_score": scores["lme"],
               "candidate_minus_lme": comparison["estimate"], "paired_ci95_low": comparison["ci95"][0],
               "paired_ci95_high": comparison["ci95"][1],
               **{f"auc_{length}": float(np.mean([row[f"auc_{length}"] for row in candidate_rows]))
                  for length in p.BAG_SIZES},
               "candidate_train_rpc_seconds": sum(row["host_train_rpc_seconds"] for row in candidate_rows),
               "candidate_condition_seconds": sum(row["seconds"] for row in candidate_rows),
               "candidate_parameters": max(row["reported_parameters"] for row in candidate_rows),
               "candidate_training_examples": sum(row["host_training_examples"] for row in candidate_rows)}
    episodes = []
    for seed in seeds:
        ids = [index for index, row in enumerate(candidate_rows) if row["seed"] == seed]
        conditions = {name: [records[index] for index in ids] for name, records in rows.items()}
        solution = {"host_authored": True, "evaluation_plane": plane, "evaluation_split": target,
                    "smoke_only": smoke, "noise_rates": list(noise_rates), "bag_sizes": list(p.BAG_SIZES),
                    "conditions": conditions, "configuration": configuration,
                    "configuration_sha256": configuration_digest(configuration), "paired_comparison": comparison,
                    "selection_traces": {name: [records[index] for index in ids] for name, records in traces.items()},
                    "paired_predictions": {"labels": data[target + "_y"].tolist(),
                                           "values": {name: np.asarray(values)[ids].tolist()
                                                      for name, values in predictions.items()}},
                    "paired_comparison_spec": {
                        "schema_version": 1, "metric": "mean_condition_worst_view_auc",
                        "source_metric": "combined_score", "prediction_key": "candidate",
                        "dataset_sha256": data_manifest[f"{plane}_sha256"],
                        "sample_ids": [f"{target}:{index}" for index in range(len(data[target + "_y"]))],
                        "condition_ids": list(noise_rates), "view_ids": list(p.BAG_SIZES),
                        "bootstrap_draws": 2000, "bootstrap_seed": 7907,
                    },
                    "selection_metric": "validation worst AUROC across paired views per condition; mean across conditions",
                    "ranking_score_space": "raw_logits",
                    "timing_scope": {"host_train_rpc_seconds": "candidate train RPC wall time, including transport and synchronization; excludes initialization and scoring",
                                     "seconds": "one condition wall time including reset, init, training, validation checks and checkpoint operations; excludes final scoring",
                                     "comparability": "cross-device throughput is not controlled; do not infer latency superiority from these screening timings"},
                    "capacity_and_device_status": "measured by candidate code and validated for shape/range; source review required; allocation enforced independently by native Docker runtime",
                    "reference_status": "frozen CPU LME with default configuration; diagnostic anchor, not an equally tuned method comparison",
                    "scope": "synthetic within-dataset patch bags; no patient or external clinical validation"}
        episodes.append({"seed": seed, "score": float(np.mean([row["worst_auc"] for row in conditions["candidate"]])),
                         "correct": True, "duration_seconds": float(sum(row["seconds"] for records in conditions.values() for row in records)),
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
