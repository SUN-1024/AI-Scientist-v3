"""Regenerate every certificate table in the audit note from the raw datasets.

The note reports certificate statistics for MUSK (UCI CSV), Corel image archives and TREC9 text
categories (svm-light). Those numbers were originally produced by ad-hoc scripts; this module
reproduces all of them from the raw files in one command and can compare the result with the saved
JSON records, so the tables are checkable end to end.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from .mil_assumption_audit import certificate_stats

MUSK_FILES = {"musk1": "musk/musk1.csv", "musk2": "musk/musk2.csv"}
COREL_FILES = {"elephant": "mil2002/elephant.svm", "fox": "mil2002/fox.svm",
               "tiger": "mil2002/tiger.svm"}
TREC9_FILES = {f"trec9_{name}": f"mil2002/trec9_{name}.svm" for name in ("1", "2", "3", "4", "7", "9", "10")}


def load_musk(path: Path) -> tuple[dict[int, list[int]], list[int], dict[int, int]]:
    """Rows are `molecule, conformation, features..., instance_label`; the name gives the bag label."""
    bags: dict[int, list[int]] = collections.OrderedDict()
    labels: list[int] = []
    bag_labels: dict[int, int] = {}
    with path.open(newline="") as stream:
        import csv
        for row in csv.reader(stream):
            if not row:
                continue
            molecule = row[0].strip()
            if molecule not in bag_labels:
                bag_labels[molecule] = 1 if molecule.startswith("MUSK-") else 0
            bags.setdefault(molecule, []).append(len(labels))
            labels.append(int(float(row[-1])))
    return bags, labels, bag_labels


def load_svm(path: Path) -> tuple[dict[int, list[int]], list[int]]:
    """svm-light lines `instance_id:bag_id:instance_label feature:value ...`."""
    bags: dict[int, list[int]] = collections.OrderedDict()
    labels: list[int] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _instance, bag, label = line.split()[0].split(":")
        bags.setdefault(int(bag), []).append(len(labels))
        labels.append(1 if int(label) == 1 else 0)
    return bags, labels


def _row(path: Path, bags: dict[Any, list[int]], labels: list[int],
         bag_labels: dict[Any, int] | None) -> dict[str, Any]:
    if bag_labels is None:
        bag_labels = {bag: (1 if max(labels[i] for i in indices) == 1 else 0)
                      for bag, indices in bags.items()}
    labelled = [(indices, bag_labels[bag]) for bag, indices in bags.items()]
    stats = certificate_stats(len(labels), labelled, labels)
    positive_bag_instances = sum(len(indices) for bag, indices in bags.items() if bag_labels[bag] == 1)
    return {
        "file": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bag_labels_are_or_of_instance_labels": all(
            max(labels[i] for i in indices) == bag_labels[bag]
            for bag, indices in bags.items()),
        "positive_bag_instances": positive_bag_instances,
        **stats,
    }


def compute(root: Path) -> dict[str, dict[str, Any]]:
    root = root.expanduser().resolve()
    report: dict[str, dict[str, Any]] = {"musk": {}, "corel": {}, "trec9": {}}
    for name, relative in MUSK_FILES.items():
        path = root / relative
        if not path.is_file():
            continue
        bags, labels, bag_labels = load_musk(path)
        row = _row(path, bags, labels, bag_labels)
        row["instance_ground_truth_available"] = not all(
            labels[i] == bag_labels[bag] for bag, indices in bags.items() for i in indices)
        report["musk"][name] = row
    for group, files in (("corel", COREL_FILES), ("trec9", TREC9_FILES)):
        for name, relative in files.items():
            path = root / relative
            if not path.is_file():
                continue
            bags, labels = load_svm(path)
            report[group][name] = _row(path, bags, labels, None)
    return report


def compare(computed: dict[str, dict[str, Any]], records_root: Path) -> dict[str, Any]:
    """Compare with the saved per-dataset statistics records; missing records are skipped."""
    checks: dict[str, Any] = {}
    for record_name, group in (("musk-certificate-stats.json", "musk"),
                                ("mil2002-certificate-stats.json", "corel"),
                                ("trec9-certificate-stats.json", "trec9")):
        path = records_root / record_name
        if not path.is_file():
            continue
        saved = json.loads(path.read_text()).get("datasets", {})
        for name, row in computed[group].items():
            expected = saved.get(name)
            if expected is None:
                checks[f"{group}/{name}"] = "missing_record"
                continue
            fields = ("bags", "instances", "positive_bags", "forced_negative", "forced_positive",
                      "unresolved", "identifiable_fraction", "certified_positive_recall",
                      "forced_negative_precision", "positive_bag_instances", "true_positive_instances")
            diffs = {field: (row.get(field), expected.get(field))
                     for field in fields
                     if field in expected and row.get(field) != expected.get(field)}
            checks[f"{group}/{name}"] = "match" if not diffs else {"mismatch": diffs}
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing musk/ and mil2002/ raw data")
    parser.add_argument("--records", type=Path, help="Directory with the saved statistics JSON")
    parser.add_argument("--output", type=Path, help="Write the computed tables here")
    args = parser.parse_args(argv)
    computed = compute(args.root)
    checks = compare(computed, args.records) if args.records else {}
    report = {"tables": computed, "checks": checks}
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True))
    failures = [name for name, value in checks.items() if value != "match" and value != "missing_record"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
