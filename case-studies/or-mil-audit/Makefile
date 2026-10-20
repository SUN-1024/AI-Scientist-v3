# Convenience targets for the OR-MIL audit case study.
PYTHON ?= python
CASE_DATA ?= data
CASE_OUT ?= out

data:
	CASE_DATA=$(CASE_DATA) bash fetch_archive.sh

analysis:
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) ambiguity_analysis.py
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) identification_bounds.py

models:
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) benchmark_models.py
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) strong_mil_experiment.py
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) mixed_bag_experiment.py
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) mixed_bag_sweep.py
	CASE_DATA=$(CASE_DATA) CASE_OUT=$(CASE_OUT) $(PYTHON) plot_mixing_curve.py

all: data analysis models

.PHONY: data analysis models all
