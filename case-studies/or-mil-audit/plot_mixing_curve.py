"""Plot propagation and model F1 against the mixing fraction (Figure 1 of the second paper)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CASE_OUT", HERE))
EVIDENCE = OUT
SWEEP = json.loads((OUT / "mixed-bag-sweep.json").read_text())
rows: dict[str, dict[float, dict]] = {}
for result in SWEEP["results"]:
    rows.setdefault(result["dataset"], {}).setdefault(result["mixed_fraction"], {})[result["architecture"]] = result["summary"]

fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), sharey=True)
for axis, (dataset, mixes) in zip(axes, rows.items()):
    fractions = sorted(mixes)
    propagation = [mixes[m]["linear"]["propagation_f1"]["mean"] for m in fractions]
    formula = [2 * (1 - m) / (2 - m) for m in fractions]
    attention = [mixes[m]["attention"]["model_f1"]["mean"] for m in fractions]
    maxpool = [mixes[m]["max"]["model_f1"]["mean"] for m in fractions]
    linear = [mixes[m]["linear"]["model_f1"]["mean"] for m in fractions]
    axis.plot(fractions, propagation, "o-", color="#1f4e79", label="propagation (measured)")
    axis.plot(fractions, formula, "--", color="#1f4e79", alpha=0.6, label="$2(1-m)/(2-m)$")
    axis.plot(fractions, attention, "s-", color="#c00000", label="attention")
    axis.plot(fractions, maxpool, "^-", color="#7f7f7f", label="max-pool MLP")
    axis.plot(fractions, linear, "v-", color="#bfbfbf", label="linear")
    axis.set_title(dataset)
    axis.set_xlabel("mixing fraction $m$")
    axis.set_ylim(0.2, 1.05)
    axis.grid(alpha=0.3)
axes[0].set_ylabel("instance $F_1$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(OUT / "mixing-curve.pdf")
fig.savefig(OUT / "mixing-curve.png", dpi=160)
print("wrote", OUT / "mixing-curve.pdf")
