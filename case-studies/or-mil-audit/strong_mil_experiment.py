"""Stronger neural MIL baselines: gated attention (Ilse et al.) and a two-layer encoder.

Reviewers of the benchmark audit will ask whether the conclusion survives a stronger architecture
than the 64-unit single-layer models. This script trains a gated-attention MIL with dropout and a
two-layer encoder on the ten datasets, three seeds each, and reports held-out instance F1 against
the prevalence baseline.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

_CASE_REPO = os.environ.get("CASE_REPO")
REPO = Path(_CASE_REPO) if _CASE_REPO else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.research.audit.mil_assumption_audit import certificate  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CASE_OUT", HERE))
DATA = Path(os.environ.get("CASE_DATA", HERE / "mil2002"))
SEEDS = (11, 23, 37)
TRAIN_FRACTION = 0.7
EPOCHS = 60
HIDDEN = 128
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
        features = {}
        for token in tokens[1:]:
            if ":" in token:
                index, value = token.split(":", 1)
                key = hash_index(int(index)) if kind == "text" else int(index)
                features[key] = features.get(key, 0.0) + float(value)
        rows.append((int(bag), 1 if int(label) == 1 else 0, features))
    width = BUCKETS if kind == "text" else max((max(f) for _, _, f in rows), default=0) + 1
    matrix = np.zeros((len(rows), width), dtype=np.float32)
    bags = collections.OrderedDict()
    labels = []
    for position, (bag, label, features) in enumerate(rows):
        for key, value in features.items():
            matrix[position, key] = value
        if kind == "text":
            norm = np.linalg.norm(matrix[position])
            if norm > 0:
                matrix[position] /= norm
        bags.setdefault(bag, []).append(position)
        labels.append(label)
    return torch.from_numpy(matrix), bags, np.array(labels)


class GatedAttentionMIL(nn.Module):
    """Ilse-style gated attention with a two-layer encoder and dropout."""

    def __init__(self, width: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(width, HIDDEN), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        )
        self.attention_v = nn.Linear(HIDDEN, 64)
        self.attention_u = nn.Linear(HIDDEN, 64)
        self.attention_w = nn.Linear(64, 1)
        self.score = nn.Linear(HIDDEN, 1)

    def instance_scores(self, features):
        return self.score(self.encoder(features)).squeeze(-1)

    def forward_bag(self, features):
        hidden = self.encoder(features)
        v = torch.tanh(self.attention_v(hidden))
        u = torch.sigmoid(self.attention_u(hidden))
        weights = torch.softmax(self.attention_w(v * u).squeeze(-1), dim=0)
        scores = torch.sigmoid(self.score(hidden).squeeze(-1))
        return torch.sum(weights * scores), scores


def train(model, matrix, bags, bag_labels, order, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    names = list(order)
    for _ in range(EPOCHS):
        np.random.default_rng(seed).shuffle(names)
        for name in names:
            features = matrix[bags[name]]
            bag_score, _ = model.forward_bag(features)
            loss = nn.functional.binary_cross_entropy(torch.clamp(bag_score, 1e-6, 1 - 1e-6),
                                                     torch.tensor(float(bag_labels[name])))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def f1(predicted, truth):
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.sum(predicted & truth)); fp = int(np.sum(predicted & ~truth)); fn = int(np.sum(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def run(name: str, kind: str):
    matrix, bags, truth = load(name, kind)
    names = list(bags)
    labels = np.array([1 if truth[bags[bag]].max() == 1 else 0 for bag in names])
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        positives = [bag for bag, label in zip(names, labels) if label == 1]
        negatives = [bag for bag, label in zip(names, labels) if label == 0]
        rng.shuffle(positives); rng.shuffle(negatives)
        cut_pos = int(round(len(positives) * TRAIN_FRACTION))
        cut_neg = int(round(len(negatives) * TRAIN_FRACTION))
        train_names = positives[:cut_pos] + negatives[:cut_neg]
        test_names = positives[cut_pos:] + negatives[cut_neg:]
        train_labels = {bag: int(labels[names.index(bag)]) for bag in train_names}
        test_instances = [i for bag in test_names for i in bags[bag]]
        test_truth = np.array([truth[i] for i in test_instances])
        model = GatedAttentionMIL(matrix.shape[1])
        train(model, matrix, bags, train_labels, train_names, seed)
        with torch.no_grad():
            predicted = torch.sigmoid(model.instance_scores(matrix[test_instances])).numpy() >= 0.5
        prevalence = f1(np.ones(len(test_truth), dtype=bool), test_truth)
        cert = certificate(len(truth), [(bags[bag], train_labels[bag]) for bag in train_names])
        propagation = np.array([i not in set(cert["negative_union"]) for i in test_instances])
        rows.append({"seed": seed, "model_f1": f1(predicted, test_truth),
                     "model_positive_recall": round(float(np.mean(predicted[test_truth == 1])), 4),
                     "prevalence_f1": prevalence,
                     "propagation_f1": f1(propagation, test_truth)})
    return {"dataset": name, "architecture": "gated_attention_2layer",
            "summary": {key: {"mean": round(float(np.mean([row[key] for row in rows])), 4),
                              "std": round(float(np.std([row[key] for row in rows])), 4)}
                        for key in ("model_f1", "model_positive_recall", "prevalence_f1", "propagation_f1")},
            "per_seed": rows}


def main() -> int:
    results = [run(name, kind) for name, kind in DATASETS.items()]
    payload = {"schema_version": 1,
               "environment": {"torch": torch.__version__, "device": "cpu", "hidden": HIDDEN,
                               "epochs": EPOCHS, "seeds": list(SEEDS),
                               "optimizer": "Adam lr=1e-3 weight_decay=1e-4",
                               "architecture": "gated attention, two-layer encoder, dropout 0.25"},
               "results": results}
    (OUT / "strong-mil-experiment.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for row in results:
        summary = row["summary"]
        print(f"{row['dataset']:10s} gated F1 {summary['model_f1']['mean']:.3f} "
              f"recall {summary['model_positive_recall']['mean']:.3f} "
              f"prevalence {summary['prevalence_f1']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
