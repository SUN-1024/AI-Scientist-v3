"""Query-complexity bounds for exact instance identification under OR bag queries.

Certification of forced labels is free once bag labels are known; identifying every instance label
needs queries. The version space has prod_b (2^{r_b} - 1) labellings, so any adaptive binary-query
strategy needs at least sum_b log2(2^{r_b} - 1) >= sum_b (r_b - 1) queries. Querying every
unresolved instance as a singleton identifies everything in sum_b r_b queries. This script
computes both bounds for the twelve archive datasets.
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


def bounds(bags, labels):
    bag_labels = {bag: (1 if max(labels[i] for i in indices) == 1 else 0)
                  for bag, indices in bags.items()}
    cert = certificate(len(labels), [(indices, bag_labels[bag]) for bag, indices in bags.items()])
    negative = set(cert["negative_union"])
    residuals = [len([i for i in indices if i not in negative])
                 for bag, indices in bags.items() if bag_labels[bag] == 1 and len(indices) > 0]
    residuals = [r for r in residuals if r > 0]
    upper = sum(residuals)
    lower_bits = sum(math.log2(2 ** r - 1) for r in residuals)
    lower_linear = sum(max(0, r - 1) for r in residuals)
    return {"positive_bags": len(residuals), "unresolved": upper,
            "singleton_upper": upper,
            "information_lower_log2": round(lower_bits, 2),
            "information_lower_ceil": math.ceil(lower_bits),
            "linear_lower": lower_linear,
            "gap_to_upper": upper - math.ceil(lower_bits)}


def main() -> int:
    report = {"schema_version": 1,
              "claim": "exact identification costs between sum log2(2^r-1) and sum r queries",
              "datasets": {}}
    for name in DATASETS:
        bags, labels = load_svm(DATA / f"{name}.svm")
        report["datasets"][name] = bounds(bags, labels)
    (OUT / "identification-bounds.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for name, row in report["datasets"].items():
        print(name, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
