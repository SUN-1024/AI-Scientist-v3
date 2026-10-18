"""Freeze and run a search-only development calibration of three archived controls.

This reuses validation bags for checkpoint selection and budget diagnosis. It is
not independent evidence of method superiority. There is no holdout argument.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import sklearn
import torch

try:
    from .protocol import EvaluationProtocol
except ImportError:
    from protocol import EvaluationProtocol


PROGRAM_NAMES = ("baseline", "log_mean_exp", "sparse_gated_attention")
BUDGETS = (160, 320, 640)
SEEDS = (7, 11, 19)
NOISE_RATES = (0.0, 0.15)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_programs():
    root = Path(__file__).resolve().parents[2]
    archived = root / "docs/research/pathmnist-mil-pilot-2026-10-18/programs"
    return {name: archived / name / "src/program.py" for name in PROGRAM_NAMES}


def freeze_calibration(search_data, output, programs=None):
    """Copy every execution input and write the full plan before the first fit."""
    search_data, output = Path(search_data).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Calibration output already exists; use a new directory")
    if search_data.name != "search.npz":
        raise ValueError("Calibration only accepts search.npz; no holdout data")
    manifest = json.loads(search_data.with_name("data_manifest.json").read_text())
    search_hash = sha256(search_data)
    if search_hash != manifest["search_sha256"]:
        raise ValueError("Search feature hash changed")
    # Inspect only allowed array names, without loading any other archive.
    with np.load(search_data, allow_pickle=False) as data:
        if set(data.files) != {"train_x", "train_y", "val_x", "val_y"}:
            raise ValueError("Calibration accepts only train/validation features and bag labels")
    programs = programs or default_programs()
    if set(programs) != set(PROGRAM_NAMES):
        raise ValueError(f"Calibration requires exactly these controls: {PROGRAM_NAMES}")
    output.mkdir(parents=True)
    frozen = output / "frozen"
    (frozen / "resources").mkdir(parents=True)
    shutil.copyfile(search_data, frozen / "resources/search.npz")
    (frozen / "resources/data_manifest.json").write_text(json.dumps({
        "search_sha256": search_hash,
        "scope": "search-only development calibration; original test data not copied or opened",
    }, indent=2) + "\n")
    for name in ("evaluate.py", "protocol.py", "calibrate.py"):
        shutil.copyfile(Path(__file__).with_name(name), frozen / name)
    for label, source in programs.items():
        target = frozen / "programs" / label / "program.py"
        target.parent.mkdir(parents=True)
        shutil.copyfile(source, target)
    protocols = {}
    for budget in BUDGETS:
        protocol = EvaluationProtocol(
            seeds=SEEDS, noise_rates=NOISE_RATES, max_steps=budget,
            patience=None, learning_rate=0.003, label_smoothing=0.0,
        )
        path = frozen / f"protocol_{budget}.json"
        path.write_text(json.dumps(protocol.to_dict(), indent=2) + "\n")
        protocols[str(budget)] = {"path": str(path.relative_to(output)), "sha256": protocol.sha256}
    files = {str(path.relative_to(output)): sha256(path) for path in sorted(frozen.rglob("*")) if path.is_file()}
    plan = {
        "schema_version": 1, "state": "frozen_before_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "development calibration of training budget and diagnostics; not a new method comparison",
        "data_split": "search.npz train and validation only",
        "held_out_data_used": False,
        "programs": list(PROGRAM_NAMES), "budgets": list(BUDGETS),
        "seeds": list(SEEDS), "noise_rates": list(NOISE_RATES),
        "protocols": protocols, "input_file_sha256": files,
        "optimizer_control": "same Adam learning_rate=0.003, smoothing=0, batch=32, weight_decay=1e-4",
        "checkpoint_rule": "max validation min(AUROC16, AUROC64), min_delta=.001, checked every20 and last step",
        "early_stopping": "disabled to expose step-cap sensitivity; no independent convergence guarantee",
        "analysis_rule": "report all methods/budgets/conditions, rank changes, seed variation and selected-checkpoint caps; no holdout selection",
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__},
    }
    (output / "calibration_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "calibration_plan.sha256").write_text(sha256(output / "calibration_plan.json") + "\n")
    return plan


def verify_frozen(output, plan):
    output = Path(output)
    expected_plan = (output / "calibration_plan.sha256").read_text().strip()
    if sha256(output / "calibration_plan.json") != expected_plan:
        raise ValueError("Frozen calibration plan changed")
    if json.loads((output / "calibration_plan.json").read_text()) != plan:
        raise ValueError("In-memory calibration plan differs from frozen plan")
    for relative, expected in plan["input_file_sha256"].items():
        if sha256(output / relative) != expected:
            raise ValueError(f"Frozen calibration input changed: {relative}")


def summarize(output, plan):
    verify_frozen(output, plan)
    rows = []
    for label in plan["programs"]:
        for budget in plan["budgets"]:
            result = Path(output) / "evaluations" / label / str(budget)
            metrics = json.loads((result / "metrics.json").read_text())
            if metrics["evaluation_split"] != "validation" or metrics["protocol_sha256"] != plan["protocols"][str(budget)]["sha256"]:
                raise ValueError("Calibration metrics do not match the frozen search-only protocol")
            expected_hashes = {
                "search_sha256": "frozen/resources/search.npz",
                "program_sha256": f"frozen/programs/{label}/program.py",
                "evaluator_sha256": "frozen/evaluate.py",
                "protocol_code_sha256": "frozen/protocol.py",
            }
            if any(metrics.get(key) != plan["input_file_sha256"][path]
                   for key, path in expected_hashes.items()):
                raise ValueError("Calibration result provenance does not match frozen input hashes")
            if json.loads((result / "correct.json").read_text()).get("correct") is not True:
                raise ValueError("Calibration evaluation did not pass correctness checks")
            conditions = metrics["conditions"]
            expected_pairs = {(seed, noise) for seed in plan["seeds"] for noise in plan["noise_rates"]}
            if len(conditions) != len(expected_pairs) or {(c["seed"], c["noise_rate"]) for c in conditions} != expected_pairs:
                raise ValueError("Calibration results must cover each seed and noise condition exactly once")
            seed_scores = {str(seed): float(np.mean([c["worst_auc"] for c in conditions if c["seed"] == seed]))
                           for seed in plan["seeds"]}
            rows.append({
                "program": label, "max_steps": budget, "combined_score": metrics["combined_score"],
                "base_auc": metrics["base_auc"], "diluted_auc": metrics["diluted_auc"],
                "seed_combined_scores": seed_scores,
                "seed_score_sample_std": float(np.std(list(seed_scores.values()), ddof=1)),
                "training_hit_step_cap_count": sum(c["training_hit_step_cap"] for c in conditions),
                "selected_checkpoint_at_step_cap_count": sum(c["selected_checkpoint_at_step_cap"] for c in conditions),
                "selected_steps": [c["selected_step"] for c in conditions],
                "stop_reasons": sorted({c["stop_reason"] for c in conditions}),
                "total_parameters": sorted({c["total_parameters"] for c in conditions}),
                "trainable_parameters": sorted({c["trainable_parameters"] for c in conditions}),
                "observed_gradient_connected_parameters": sorted({c["observed_gradient_connected_parameters"] for c in conditions}),
                "inactive_parameter_names": sorted({n for c in conditions for n in c["inactive_parameter_names"]}),
                **{"mean_" + name: float(np.mean([c[name] for c in conditions])) for name in (
                    "auc_change", "positive_probability_change", "negative_probability_change",
                    "positive_logit_change", "negative_logit_change", "logit_margin_change")},
                "evaluation_seconds": metrics["seconds"],
                "conditions": conditions,
            })
    summary = {
        "state": "completed", "plan_sha256": sha256(Path(output) / "calibration_plan.json"),
        "held_out_data_used": False, "fit_count": len(PROGRAM_NAMES) * len(BUDGETS) * len(SEEDS) * len(NOISE_RATES),
        "note": "Descriptive development calibration on previously used validation bags. No independent performance or research-superiority claim.",
        "rows": rows,
    }
    (Path(output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_calibration(search_data, output, *, programs=None):
    plan = freeze_calibration(search_data, output, programs)
    output = Path(output).resolve()
    for label in plan["programs"]:
        for budget in plan["budgets"]:
            verify_frozen(output, plan)
            result_dir = output / "evaluations" / label / str(budget)
            result_dir.mkdir(parents=True)
            command = [sys.executable, str(output / "frozen/evaluate.py"),
                       "--program_path", str(output / "frozen/programs" / label / "program.py"),
                       "--results_dir", str(result_dir),
                       "--protocol", str(output / plan["protocols"][str(budget)]["path"]),
                       "--search-data", str(output / "frozen/resources/search.npz")]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
            (result_dir / "stdout.log").write_text(completed.stdout)
            (result_dir / "stderr.log").write_text(completed.stderr)
            verify_frozen(output, plan)
            if completed.returncode:
                raise ValueError(f"Calibration evaluation failed: {label}/{budget}; inspect stderr.log")
            print(f"Completed {label}: {budget} steps, 3 seeds x 2 noise conditions", flush=True)
    return summarize(output, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_calibration(args.search_data, args.output)
    print(json.dumps([{k: row[k] for k in ("program", "max_steps", "combined_score", "selected_checkpoint_at_step_cap_count")}
                      for row in result["rows"]], indent=2))
