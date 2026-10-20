"""Run the case-study analysis scripts on a synthetic one-bag dataset."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASE = REPO / "case-studies" / "or-mil-audit"

# One positive bag over two instances plus one negative bag; residual size 2.
SVM = "0:0:1 1:0.1\n1:0:1 1:0.2\n2:1:-1 1:0.3\n"


def run_script(tmp_path: Path, name: str) -> dict:
    data = tmp_path / "data"
    out = tmp_path / "out"
    (data / "mil2002").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    (data / "mil2002" / "tiny.svm").write_text(SVM)
    env = {**os.environ, "CASE_DATA": str(data), "CASE_OUT": str(out),
           "CASE_DATASETS": "tiny", "CASE_REPO": str(REPO)}
    subprocess.run([sys.executable, str(CASE / name)], env=env, check=True,
                   capture_output=True, timeout=120)
    output = out / ("identification-bounds.json" if "identification" in name else "ambiguity-analysis.json")
    return json.loads(output.read_text())


def test_identification_bounds(tmp_path: Path) -> None:
    report = run_script(tmp_path, "identification_bounds.py")
    row = report["datasets"]["tiny"]
    assert row["positive_bags"] == 1
    assert row["unresolved"] == 2
    assert row["singleton_upper"] == 2
    assert row["information_lower_ceil"] == 2
    assert row["gap_to_upper"] == 0


def test_version_space(tmp_path: Path) -> None:
    report = run_script(tmp_path, "ambiguity_analysis.py")
    row = report["datasets"]["tiny"]
    assert row["log10_version_space"] == 0.477
    assert row["mean_unresolved_positive_probability"] == 0.6667
    construction = report["construction_original_four_bags"]
    assert construction["positive_bags"] == 4
    assert construction["log10_version_space"] == 3.38
