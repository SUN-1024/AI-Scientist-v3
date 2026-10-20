"""Certificate baseline versus trained MIL models on ten OR-MIL datasets.

For each dataset: stratified 70/30 bag split (5 seeds), certificate from train bag labels,
three model families (linear max-pooling, max-pooling MLP, attention MLP), evaluation on held-out
bag instances. Image features are 230-dimensional; TREC9 features are hashed into 4096 buckets.
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

REPO = Path(os.environ.get("CASE_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))
from src.research.audit.mil_assumption_audit import certificate  # noqa: E402

HERE = Path(__file__).resolve().parent
EVIDENCE = Path(os.environ.get("CASE_OUT", HERE))
if os.environ.get("CASE_DATA"):
    DATA = Path(os.environ["CASE_DATA"]) / "mil2002"
else:
    DATA = HERE / "mil2002"
SEEDS = tuple(int(value) for value in os.environ.get("CASE_SEEDS", "11,23,37,41,53").split(",") if value)
IMAGE_EPOCHS = int(os.environ.get("CASE_EPOCHS", "60"))
TEXT_EPOCHS = int(os.environ.get("CASE_EPOCHS", "40"))
TRAIN_FRACTION = 0.7
BUCKETS = 4096
HIDDEN = 64
DEFAULT_DATASETS = {
    "elephant": "image", "fox": "image", "tiger": "image",
    "trec9_1": "text", "trec9_2": "text", "trec9_3": "text", "trec9_4": "text",
    "trec9_7": "text", "trec9_9": "text", "trec9_10": "text",
}
if os.environ.get("CASE_DATASETS"):
    DATASETS = {name: DEFAULT_DATASETS[name] for name in os.environ["CASE_DATASETS"].split(",") if name}
else:
    DATASETS = DEFAULT_DATASETS


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


class LinearMil(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, 1)

    def instance_scores(self, features):
        return self.linear(features).squeeze(-1)


class MaxPoolMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def instance_scores(self, features):
        return self.encoder(features).squeeze(-1)


class AttentionMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU())
        self.score = nn.Linear(HIDDEN, 1)
        self.attention = nn.Linear(HIDDEN, 1)

    def instance_scores(self, features):
        return self.score(self.encoder(features)).squeeze(-1)

    def forward_bag(self, features):
        hidden = self.encoder(features)
        weights = torch.softmax(self.attention(hidden).squeeze(-1), dim=0)
        scores = torch.sigmoid(self.score(hidden).squeeze(-1))
        return torch.sum(weights * scores), scores


def train(model, matrix, bags, bag_labels, order, *, seed, epochs):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    names = list(order)
    for _ in range(epochs):
        np.random.default_rng(seed).shuffle(names)
        for name in names:
            features = matrix[bags[name]]
            if isinstance(model, AttentionMLP):
                bag_score, _ = model.forward_bag(features)
            else:
                bag_score = torch.sigmoid(model.instance_scores(features)).max()
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


def run(name: str, kind: str, architecture: str):
    matrix, bags, truth = load(name, kind)
    names = list(bags)
    labels = np.array([1 if truth[bags[bag]].max() == 1 else 0 for bag in names])
    epochs = IMAGE_EPOCHS if kind == "image" else TEXT_EPOCHS
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
        model = {"linear": LinearMil, "max": MaxPoolMLP, "attention": AttentionMLP}[architecture](matrix.shape[1])
        train(model, matrix, bags, train_labels, train_names, seed=seed, epochs=epochs)
        with torch.no_grad():
            predicted = torch.sigmoid(model.instance_scores(matrix[test_instances])).numpy() >= 0.5
        cert = certificate(len(truth), [(bags[bag], train_labels[bag]) for bag in train_names])
        certificate_pred = np.array([i not in set(cert["negative_union"]) for i in test_instances])
        rows.append({"seed": seed,
                     "model_f1": f1(predicted, test_truth),
                     "model_positive_recall": round(float(np.mean(predicted[test_truth == 1])), 4),
                     "certificate_only_f1": f1(certificate_pred, test_truth),
                     "certificate_positive_recall": 1.0,
                     "test_positives": int(test_truth.sum())})
    return {"dataset": name, "modality": kind, "architecture": architecture,
            "summary": {key: {"mean": round(float(np.mean([row[key] for row in rows])), 4),
                              "std": round(float(np.std([row[key] for row in rows])), 4)}
                        for key in ("model_f1", "model_positive_recall", "certificate_only_f1")},
            "per_seed": rows}


def main() -> int:
    results = [run(name, kind, architecture)
               for name, kind in DATASETS.items()
               for architecture in ("linear", "max", "attention")]
    payload = {"schema_version": 1,
               "environment": {"torch": torch.__version__, "device": "cpu", "hidden": HIDDEN,
                               "buckets": BUCKETS, "epochs": {"image": IMAGE_EPOCHS, "text": TEXT_EPOCHS},
                               "train_fraction": TRAIN_FRACTION, "seeds": list(SEEDS)},
               "datasets": DATASETS, "results": results}
    (EVIDENCE / "benchmark-models.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    for row in results:
        print(row["dataset"], row["architecture"], row["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
