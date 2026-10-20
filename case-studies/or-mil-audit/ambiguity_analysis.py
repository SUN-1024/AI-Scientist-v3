"""Version-space size and per-instance label marginals under the OR certificate.

For a positive bag with residual of size r, the bag label allows any nonempty subset of the
residual to be positive: 2^r - 1 assignments. An unresolved instance is positive in 2^(r-1) of
them. This script computes the exact version-space log-size and the mean marginal probability for
the twelve archived datasets, and for the finite construction in the note.
"""
from __future__ import annotations

import collections
import collections
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(os.environ.get("CASE_REPO", str(Path(__file__).resolve().parents[2])))
sys.path.insert(0, str(REPO))
from src.research.audit.mil_assumption_audit import certificate  # noqa: E402

import os
HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CASE_OUT", HERE))
DATA = Path(os.environ.get("CASE_DATA", HERE / "mil2002"))
DATASETS = ["elephant", "fox", "tiger"] + [f"trec9_{name}" for name in ("1", "2", "3", "4", "7", "9", "10")]


def load_svm(path: Path):
    bags = collections.OrderedDict()
    labels = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _iid, bag, label = line.split()[0].split(":")
        bags.setdefault(int(bag), []).append(len(labels))
        labels.append(1 if int(label) == 1 else 0)
    return bags, labels


def version_space(bags, labels):
    bag_labels = {bag: (1 if max(labels[i] for i in indices) == 1 else 0)
                  for bag, indices in bags.items()}
    cert = certificate(len(labels), [(indices, bag_labels[bag]) for bag, indices in bags.items()])
    negative = set(cert["negative_union"])
    residuals = [len([i for i in indices if i not in negative])
                 for bag, indices in bags.items() if bag_labels[bag] == 1]
    log_size = sum(math.log10(2 ** r - 1) if r > 0 else 0 for r in residuals)
    marginals = []
    for bag, indices in bags.items():
        if bag_labels[bag] != 1:
            continue
        residual = [i for i in indices if i not in negative]
        r = len(residual)
        if r == 0:
            continue
        probability = 2 ** (r - 1) / (2 ** r - 1)
        marginals.extend([probability] * r)
    return {"positive_bags": len(residuals), "residual_sizes": sorted(residuals),
            "log10_version_space": round(log_size, 3),
            "mean_unresolved_positive_probability": round(sum(marginals) / len(marginals), 4) if marginals else None,
            "min_unresolved_positive_probability": round(min(marginals), 4) if marginals else None,
            "max_unresolved_positive_probability": round(max(marginals), 4) if marginals else None}


def main() -> int:
    report = {"schema_version": 1, "datasets": {}}
    for name in DATASETS:
        bags, labels = load_svm(DATA / f"{name}.svm")
        report["datasets"][name] = version_space(bags, labels)
    # the finite construction: four bags over eight points, repaired system from the note
    construction = {"bags": {"B0": ([0, 1, 2], 1), "B1": ([2, 3, 4], 1), "B2": ([4, 5, 6], 1), "B3": ([6, 7, 0], 1)}}
    labels = [0] * 8
    for indices, label in construction["bags"].values():
        for index in indices:
            labels[index] = max(labels[index], label)
    report["construction_original_four_bags"] = version_space(
        {bag: list(indices) for bag, (indices, _label) in construction["bags"].items()}, labels)
    (OUT / "ambiguity-analysis.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for name, row in report["datasets"].items():
        print(name, row)
    print("construction:", report["construction_original_four_bags"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
