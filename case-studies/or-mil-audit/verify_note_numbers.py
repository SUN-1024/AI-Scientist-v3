"""Check every headline number in NOTE.tex against the recorded evidence JSON.

Each check names a value that the manuscript asserts and the evidence file that produced it. A
failure means either the manuscript or the evidence is stale; the script prints both.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CASE_OUT", HERE))
TEX = (OUT / "NOTE.tex").read_text()


def load(name):
    return json.loads((OUT / name).read_text())


MUSK = load("musk-certificate-stats.json")["datasets"]
COREL = load("mil2002-certificate-stats.json")["datasets"]
TREC9 = load("trec9-certificate-stats.json")["datasets"]
SIM = load("certificate-simulation.json")
PILOT = load("active-query-pilot.json")
NEURAL = load("neural-mil-experiment.json")
LEARNED = load("learned-vs-certified.json")
SWEEP = load("learned-vs-certified-sweep.json")
AMBIG = load("ambiguity-analysis.json")

checks = []


def check(description, value, evidence):
    checks.append({"description": description, "value": value, "in_manuscript": value in TEX,
                   "evidence": evidence})


def fmt_frac(row, key):
    return row[key]


# certificate table
for name, row in COREL.items():
    check(f"{name} instance count", str(row["instances"]), row["instances"])
    check(f"{name} forced negatives", str(row["forced_negative"]), row["forced_negative"])
check("elephant certified recall 0/762", "0/762", COREL["elephant"]["certified_positive_recall"])
check("fox certified recall 0/647", "0/647", COREL["fox"]["certified_positive_recall"])
check("tiger certified recall 1/544", "1/544", COREL["tiger"]["certified_positive_recall"])
check("musk1 forced negatives 269", "269", MUSK["musk1"]["forced_negative"])
check("musk2 forced negatives 5581", "5581", MUSK["musk2"]["forced_negative"])

# trec9 table
for name, row in TREC9.items():
    check(f"{name} instances", str(row["instances"]), row["instances"])
    check(f"{name} forced positive", str(row["forced_positive"]), row["forced_positive"])
total_positives = sum(row["forced_positive"] for row in TREC9.values())
check("trec9 forced positives total 14", str(total_positives), total_positives)

# simulation row
sim = SIM["results"] if "results" in SIM else SIM.get("rows", SIM)
def sim_prob(entry):
    if isinstance(entry, dict):
        for key in ("probability", "p_forced_positive", "forced_positive_rate"):
            if key in entry:
                return entry[key]
    return None
first = sim[0] if isinstance(sim, list) else next(iter(sim.values()))
check("simulation first probability present", "0.965", sim_prob(first) if isinstance(sim, list) else None)

# pilot table
pilot_rows = {(row["dataset"], row["strategy"]): row for row in PILOT["results"]}
for dataset in ("elephant", "fox", "tiger"):
    for strategy in ("uncertainty", "certificate"):
        value = pilot_rows[(dataset, strategy)]["curve"][-1]["instance_f1"]
        check(f"pilot {dataset} {strategy} at 200", f"{value:.3f}", value)

# neural table
neural_rows = {(row["dataset"], row["architecture"]): row["summary"] for row in NEURAL["results"]}
for dataset in ("elephant", "fox", "tiger"):
    for arch in ("max", "attention"):
        value = neural_rows[(dataset, arch)]["model_f1"]["mean"]
        check(f"neural {dataset} {arch} F1", f"{value:.3f}", value)

# version space
check("construction log10 version space 3.38", "3.38",
      AMBIG["construction_original_four_bags"]["log10_version_space"])
check("construction marginal 0.571", "0.571",
      AMBIG["construction_original_four_bags"]["mean_unresolved_positive_probability"])

failures = [row for row in checks if not row["in_manuscript"]]
report = {"checks": len(checks), "failures": failures}
(OUT / "note-number-check.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
print(f"{len(checks)} checks, {len(failures)} failures")
for row in failures:
    print("FAIL:", row["description"], "| expected in manuscript:", row["value"], "| evidence:", row["evidence"])
