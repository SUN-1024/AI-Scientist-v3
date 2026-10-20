"""Recompute the certificate-only baseline from the evaluation-set bag labels.

The first benchmark built the certificate from training bags and evaluated on bag-disjoint test bags.
Because the splits are bag-disjoint, no test instance can lie in the training negative union, so that
predictor degenerates to "all positive" on the test set. The correct reference uses the bag labels of
the evaluation set itself: a test instance is certified negative when it lies in a negative test bag.
Model predictions are unchanged, so only the baseline is recomputed here.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(os.environ.get("CASE_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))
from src.research.audit.mil_assumption_audit import certificate  # noqa: E402

import os
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("CASE_DATA", HERE / "mil2002"))
SEEDS = (11, 23, 37, 41, 53)
TRAIN_FRACTION = 0.7
BUCKETS = 4096
DATASETS = {
    "elephant": "image", "fox": "image", "tiger": "image",
    "trec9_1": "text", "trec9_2": "text", "trec9_3": "text", "trec9_4": "text",
    "trec9_7": "text", "trec9_9": "text", "trec9_10": "text",
}


def hash_index(index: int) -> int:
    return int(hashlib.sha1(str(index).encode()).hexdigest()[:8], 16) % BUCKETS


def load(name: str, kind: str):
    rows = []
    for line in (DATA / f"{name}.svm").read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split()
        _iid, bag, label = tokens[0].split(":")
        rows.append((int(bag), 1 if int(label) == 1 else 0))
    bags = collections.OrderedDict()
    labels = []
    for bag, label in rows:
        bags.setdefault(bag, []).append(len(labels))
        labels.append(label)
    return bags, np.array(labels)


def f1(predicted, truth):
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(predicted & truth)); fp = int(np.sum(predicted & ~truth)); fn = int(np.sum(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def split(names, labels, seed):
    rng = np.random.default_rng(seed)
    positives = [name for name, label in zip(names, labels) if label == 1]
    negatives = [name for name, label in zip(names, labels) if label == 0]
    rng.shuffle(positives); rng.shuffle(negatives)
    cut_pos = int(round(len(positives) * TRAIN_FRACTION))
    cut_neg = int(round(len(negatives) * TRAIN_FRACTION))
    return positives[:cut_pos] + negatives[:cut_neg], positives[cut_pos:] + negatives[cut_neg:]


def main() -> int:
    bench = json.loads((EVIDENCE / "benchmark-models.json").read_text())
    by_key = {(row["dataset"], row["architecture"]): row for row in bench["results"]}
    updated = []
    for name, kind in DATASETS.items():
        bags, truth = load(name, kind)
        names = list(bags)
        bag_labels = np.array([1 if truth[bags[bag]].max() == 1 else 0 for bag in names])
        per_seed = []
        for seed in SEEDS:
            _train_names, test_names = split(names, bag_labels, seed)
            test_labels = {bag: int(bag_labels[names.index(bag)]) for bag in test_names}
            test_instances = [i for bag in test_names for i in bags[bag]]
            test_truth = np.array([truth[i] for i in test_instances])
            cert = certificate(len(truth), [(bags[bag], test_labels[bag]) for bag in test_names])
            negative_union = set(cert["negative_union"])
            predicted = np.array([i not in negative_union for i in test_instances])
            per_seed.append({
                "seed": seed,
                "certificate_test_f1": round(f1(predicted, test_truth), 4),
                "certificate_test_precision": round(float(np.mean(predicted[test_truth == 1])), 4),
                "certificate_test_positive_recall": round(float(np.mean(predicted[test_truth == 1])), 4),
                "test_certified_negative": len(negative_union),
                "test_positives": int(test_truth.sum()),
                "test_instances": len(test_instances),
                "forced_positive_in_test": len(cert["forced_positive"]),
            })
        summary = {key: {"mean": round(float(np.mean([row[key] for row in per_seed])), 4),
                         "std": round(float(np.std([row[key] for row in per_seed])), 4)}
                   for key in ("certificate_test_f1", "certificate_test_positive_recall")}
        for architecture in ("linear", "max", "attention"):
            row = by_key[(name, architecture)]
            row["certificate_test_only_f1"] = summary["certificate_test_f1"]
            row["certificate_test_only_positive_recall"] = summary["certificate_test_positive_recall"]
            row["certificate_test_per_seed"] = per_seed
        updated.append({"dataset": name, "summary": summary})
    bench["certificate_baseline_note"] = (
        "certificate_test_only_f1 uses the evaluation-set bag labels; the training-bag certificate "
        "is reported separately because bag-disjoint splits make it vacuous on the test set")
    (HERE / "benchmark-models.json").write_text(json.dumps(bench, indent=1, sort_keys=True) + "\n")
    for row in updated:
        print(row["dataset"], row["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
