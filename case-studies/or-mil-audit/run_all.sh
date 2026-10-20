#!/usr/bin/env bash
# Run the OR-MIL audit case-study pipeline.
#
# Usage: CASE_DATA=/data CASE_OUT=/out [CASE_DATASETS=elephant,trec9_1] bash run_all.sh [--skip-torch]
# Requires: a data directory prepared by fetch_archive.sh (or an equivalent extraction) and,
# unless --skip-torch is passed, PyTorch on CPU for the model comparisons.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

"$PY" ambiguity_analysis.py
"$PY" identification_bounds.py

if [ "${1:-}" != "--skip-torch" ]; then
  "$PY" recompute_baseline.py            # needs benchmark-models.json from a previous run
  "$PY" benchmark_models.py
  "$PY" strong_mil_experiment.py
  "$PY" mixed_bag_experiment.py
  "$PY" mixed_bag_sweep.py
  "$PY" plot_mixing_curve.py
fi

if [ -f "${CASE_OUT:-.}/NOTE.tex" ]; then
  "$PY" verify_note_numbers.py
fi
echo "case-study pipeline finished; outputs in ${CASE_OUT:-.}"
