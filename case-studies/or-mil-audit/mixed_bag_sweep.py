"""Controlled mixed-bag experiment on the Corel archives.

The archived Corel datasets are label-pure, so bag-label propagation solves instance evaluation
trivially. This script builds mixed versions: a fraction of each positive bag's instances is
replaced by instances drawn from negative bags (at least one positive instance remains, so the OR
label is unchanged and negative bags stay pure). It then repeats the model-versus-baseline
comparison on the mixed data, where propagation is no longer perfect.
"""
from __future__ import annotations

import collections
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

EVIDENCE = Path(os.environ.get("CASE_OUT", HERE))
DATA = Path(os.environ.get("CASE_DATA", EVIDENCE / "mil2002"))
SEEDS = (11, 23, 37, 41, 53)
TRAIN_FRACTION = 0.7
EPOCHS = 60
HIDDEN = 64
MIX_FRACTIONS = (0.0, 0.2, 0.4, 0.6)
DATASETS = ("elephant", "fox", "tiger")


def load_svm(path: Path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split()
        _iid, bag, label = tokens[0].split(":")
        features = {}
        for token in tokens[1:]:
            if ":" in token:
                index, value = token.split(":", 1)
                features[int(index)] = float(value)
        rows.append((int(bag), 1 if int(label) == 1 else 0, features))
    width = max((max(features) for _, _, features in rows), default=0) + 1
    return rows, width


def build_mixed(rows, seed, mix_fraction):
    rng = np.random.default_rng(seed)
    bags = collections.OrderedDict()
    for bag, label, features in rows:
        bags.setdefault(bag, []).append((label, features))
    bag_labels = {bag: (1 if any(label == 1 for label, _ in members) else 0)
                  for bag, members in bags.items()}
    negatives = [item for bag, members in bags.items() if bag_labels[bag] == 0
                 for item in members]
    rng.shuffle(negatives)
    pool = list(negatives)
    mixed = {}
    for bag, members in bags.items():
        positives = [item for item in members if item[0] == 1]
        negatives_here = [item for item in members if item[0] == 0]
        if bag_labels[bag] == 0:
            mixed[bag] = negatives_here
            continue
        keep = max(1, int(round(len(positives) * (1 - mix_fraction))))
        chosen = positives[:keep]
        needed = len(positives) - keep
        taken = [pool.pop() for _ in range(min(needed, len(pool)))]
        mixed[bag] = chosen + taken
    return mixed, bag_labels


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


def tensorise(mixed, width):
    order = []
    labels = []
    bag_indices = collections.OrderedDict()
    for bag, members in mixed.items():
        bag_indices[bag] = []
        for label, features in members:
            bag_indices[bag].append(len(order))
            order.append(features)
            labels.append(label)
    matrix = np.zeros((len(order), width), dtype=np.float32)
    for position, features in enumerate(order):
        for index, value in features.items():
            matrix[position, index] = value
    return torch.from_numpy(matrix), bag_indices, np.array(labels)


def train(model, matrix, bags, bag_labels, order, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    names = list(order)
    for _ in range(EPOCHS):
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


def run(name: str, architecture: str, mixed_data, bag_labels, width: int, mix_fraction: float):
    matrix, bags, truth = tensorise(mixed_data, width)
    names = list(bags)
    labels = np.array([bag_labels[bag] for bag in names])
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
        train_labels = {bag: int(bag_labels[bag]) for bag in train_names}
        test_instances = [i for bag in test_names for i in bags[bag]]
        test_truth = np.array([truth[i] for i in test_instances])
        model = {"linear": LinearMil, "max": MaxPoolMLP, "attention": AttentionMLP}[architecture](width)
        train(model, matrix, bags, train_labels, train_names, seed)
        with torch.no_grad():
            predicted = torch.sigmoid(model.instance_scores(matrix[test_instances])).numpy() >= 0.5
        propagation = np.array([1 if bag_labels[bag] == 1 else 0 for bag in test_names
                                for _ in bags[bag]])
        cert = certificate(len(truth), [(bags[bag], bag_labels[bag]) for bag in test_names])
        negative_union = set(cert["negative_union"])
        certificate_pred = np.array([i not in negative_union for i in test_instances])
        rows.append({"seed": seed,
                     "model_f1": f1(predicted, test_truth),
                     "propagation_f1": f1(propagation, test_truth),
                     "prevalence_f1": f1(np.ones(len(test_truth), dtype=bool), test_truth),
                     "certificate_f1": f1(certificate_pred, test_truth)})
    summary = {key: {"mean": round(float(np.mean([row[key] for row in rows])), 4),
                     "std": round(float(np.std([row[key] for row in rows])), 4)}
               for key in ("model_f1", "propagation_f1", "prevalence_f1", "certificate_f1")}
    return {"dataset": name, "architecture": architecture, "mixed_fraction": mix_fraction,
            "summary": summary}


def main() -> int:
    results = []
    for name in DATASETS:
        rows, width = load_svm(DATA / f"{name}.svm")
        for mix_fraction in MIX_FRACTIONS:
            mixed_data, bag_labels = build_mixed(rows, seed=20261020, mix_fraction=mix_fraction)
            for architecture in ("linear", "max", "attention"):
                results.append(run(name, architecture, mixed_data, bag_labels, width, mix_fraction))
        print(name, "done")
    payload = {"schema_version": 1,
               "protocol": {"mix_fractions": list(MIX_FRACTIONS), "seeds": list(SEEDS),
                            "construction": "replace a fraction m of each positive bag's instances with instances drawn from negative bags, keeping at least one positive instance",
                            "analytic_propagation_f1": "2(1-m)/(2-m)"},
               "environment": {"torch": torch.__version__, "epochs": EPOCHS, "hidden": HIDDEN},
               "results": results}
    (EVIDENCE / "mixed-bag-sweep.json").write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
