"""Regenerate the note's certificate tables from raw data."""
from __future__ import annotations

import json
from pathlib import Path

from src.research.audit.certificate_tables import compare, compute, load_musk, load_svm, main


def write_dataset_tree(root: Path) -> None:
    (root / "musk").mkdir(parents=True)
    # Rows repeat the bag label; one positive molecule with two conformers, one negative molecule.
    (root / "musk/musk1.csv").write_text(
        "MUSK-a,c1,1.0,0.0,1\n"
        "MUSK-a,c2,0.5,-0.5,1\n"
        "NON-MUSK-b,c3,0.0,1.0,0\n"
    )
    (root / "mil2002").mkdir()
    # svm-light: instance:bag:label ; bag 0 positive (two positives), bag 1 negative (two negatives).
    (root / "mil2002/elephant.svm").write_text(
        "0:0:1 1:0.1 2:0.2\n"
        "1:0:1 1:0.3 2:0.4\n"
        "2:1:-1 1:0.5 2:0.6\n"
        "3:1:-1 1:0.7 2:0.8\n"
    )


def test_load_musk_groups_rows_by_molecule(tmp_path: Path) -> None:
    write_dataset_tree(tmp_path)
    bags, labels, bag_labels = load_musk(tmp_path / "musk/musk1.csv")
    assert bags == {"MUSK-a": [0, 1], "NON-MUSK-b": [2]}
    assert labels == [1, 1, 0]
    assert bag_labels == {"MUSK-a": 1, "NON-MUSK-b": 0}


def test_load_svm_reads_instance_bag_label(tmp_path: Path) -> None:
    write_dataset_tree(tmp_path)
    bags, labels = load_svm(tmp_path / "mil2002/elephant.svm")
    assert bags == {0: [0, 1], 1: [2, 3]}
    assert labels == [1, 1, 0, 0]


def test_compute_reports_certificate_counts(tmp_path: Path) -> None:
    write_dataset_tree(tmp_path)
    tables = compute(tmp_path)
    musk = tables["musk"]["musk1"]
    assert musk["bags"] == 2 and musk["instances"] == 3
    assert musk["forced_negative"] == 1 and musk["unresolved"] == 2
    assert musk["identifiable_fraction"] == "1/3"
    corel = tables["corel"]["elephant"]
    assert corel["bags"] == 2 and corel["instances"] == 4
    assert corel["forced_negative"] == 2 and corel["forced_positive"] == 0
    assert corel["certified_positive_recall"] == "0/2"
    assert corel["bag_labels_are_or_of_instance_labels"] is True


def test_compare_matches_records_and_detects_stale_values(tmp_path: Path) -> None:
    write_dataset_tree(tmp_path)
    tables = compute(tmp_path)
    record = {"datasets": {"elephant": {key: value for key, value in tables["corel"]["elephant"].items()
                                        if key in {"bags", "instances", "forced_positive"}}}}
    (tmp_path / "mil2002-certificate-stats.json").write_text(json.dumps(record))
    checks = compare(tables, tmp_path)
    assert checks["corel/elephant"] == "match"
    stale = json.loads(json.dumps(record))
    stale["datasets"]["elephant"]["forced_positive"] = 3
    (tmp_path / "mil2002-certificate-stats.json").write_text(json.dumps(stale))
    checks = compare(tables, tmp_path)
    assert checks["corel/elephant"]["mismatch"]["forced_positive"] == (0, 3)


def test_cli_exit_code_reflects_comparison(tmp_path: Path, capsys) -> None:
    write_dataset_tree(tmp_path)
    tables = compute(tmp_path)
    record = {"datasets": {"elephant": {key: value for key, value in tables["corel"]["elephant"].items()
                                        if key in {"bags", "instances"}}}}
    records = tmp_path / "records"
    records.mkdir()
    (records / "mil2002-certificate-stats.json").write_text(json.dumps(record))
    assert main(["--root", str(tmp_path), "--records", str(records)]) == 0
    stale = json.loads(json.dumps(record))
    stale["datasets"]["elephant"]["bags"] = 99
    (records / "mil2002-certificate-stats.json").write_text(json.dumps(stale))
    assert main(["--root", str(tmp_path), "--records", str(records)]) == 1
