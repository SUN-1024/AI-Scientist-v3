"""All planned controls reach paper and readiness through real final receipts."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.final_evaluation.comparison_set import SET_MANIFEST, freeze_portfolio_final_set
from src.final_evaluation.runner import FinalEvaluationError
from src.investigation.paper_prompt_context import paper_workspace_context_replacements
from src.orchestrator.paper_data import _merge_final_evaluation
from src.pipeline.review import _selected_novelty_evidence
from src.research.config import ResearchConfig
from src.research.paper_readiness import assess_readiness
from src.research.portfolio_final_evidence import FIELD, verify_portfolio_final_for_paper
from tests.final_evaluation import test_portfolio_final_set as final_fixture
from tests.final_evaluation.test_final_evaluation import _write_contract_task, _write_investigation, run_final_evaluation


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def final_study(tmp_path, monkeypatch, request):
    settings = getattr(request, "param", {})
    workspace = tmp_path / "workspace"
    run = workspace / "run"
    run.mkdir(parents=True)

    def task_writer(directory):
        task = _write_contract_task(directory)
        contract = read(task / "task_contract.json")
        contract["metrics"]["final"] = [{"name": name, "data_role": "sealed_test"} for name in (
            "combined_score", "candidate_minus_lme", "paired_diagnostic_ci95_low")]
        contract["protocol"].update(final_seeds=[2, 3, 4], final_reset_sequence=[2, 3, 4])
        write(task / "task_contract.json", contract)
        source = """import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--program_path', required=True)
p.add_argument('--results_dir', required=True)
a = p.parse_args()
source = Path(a.program_path).read_text()
proposal = 'idea_0001' in source
negative = NEGATIVE_EFFECT
values = [0., .1, .9, 1.] if proposal != negative else [.5, .5, .5, .5]
score = 1. if proposal != negative else .5
seeds = [2, 3] if MISSING_CONTROL_SEED and 'idea_0003' in source else [2, 3, 4]
spec = {'schema_version': 1, 'metric': 'mean_condition_worst_view_auc', 'source_metric': 'combined_score',
        'prediction_key': 'candidate', 'bootstrap_draws': 100, 'bootstrap_seed': 17,
        'dataset_sha256': 'a' * 64, 'sample_ids': ['s0', 's1', 's2', 's3'],
        'condition_ids': ['fixed'], 'view_ids': ['fixed']}
episodes = [{'seed': seed, 'score': score, 'correct': True, 'duration_seconds': 0.1,
             'solution': {'evaluation_plane': 'final', 'smoke_only': False,
                          'configuration': {'learning_rate': .1},
                          'paired_comparison_spec': spec,
                          'paired_predictions': {'labels': [0, 0, 1, 1], 'values': {'candidate': [[values]]}}}}
            for seed in seeds]
if MISSING_CONFIGURATION and 'idea_0003' in source:
    for episode in episodes:
        episode['solution'].pop('configuration')
out = Path(a.results_dir)
out.mkdir(parents=True, exist_ok=True)
(out / 'metrics.json').write_text(json.dumps({'combined_score': score, 'candidate_minus_lme': .9,
                                            'paired_diagnostic_ci95_low': .8}))
(out / 'correct.json').write_text(json.dumps({'correct': True}))
(out / 'evaluation_evidence.json').write_text(json.dumps({'schema_version': 1, 'episodes': episodes}))
"""
        source = source.replace("NEGATIVE_EFFECT", repr(settings.get("negative", False)))
        source = source.replace("MISSING_CONTROL_SEED", repr(settings.get("missing_seed", False)))
        source = source.replace("MISSING_CONFIGURATION", repr(settings.get("missing_configuration", False)))
        (task / "evaluate_final.py").write_text(source)
        return task

    monkeypatch.setattr(final_fixture, "_write_contract_task", task_writer)
    task, root, _unused_output, controller = final_fixture.study(
        run, winner=settings.get("winner", "proposal"), promote=settings.get("promote", 3),
        final_configuration=None,
    )
    final = run / "final_evaluation"
    if settings.get("partial"):
        calls = []

        def fail_control(command, **kwargs):
            calls.append(command)
            if len(calls) == 2:
                return subprocess.CompletedProcess(command, 1, "", "A retained final fixture failure")
            return subprocess.run(command, **kwargs)

        with pytest.raises(FinalEvaluationError):
            final_fixture.evaluate(task, root, final, run=fail_control)
    elif settings.get("freeze_only"):
        freeze_portfolio_final_set(task_dir=task, investigation_results_dir=root, output_dir=final)
    else:
        final_fixture.evaluate(task, root, final)
    cfg = ResearchConfig()
    cfg.methodology.enabled = False
    cfg.harness.quality.positive_effect_metric = "candidate_minus_controls_min"
    cfg.harness.quality.positive_effect_interval_low_metric = "paired_controls_ci95_low_min"
    data = {"results_dir": str(root), "best_correct": {"program_dir": controller.data["frozen_candidate"]["program_dir"]}}
    return SimpleNamespace(workspace=workspace, run=run, task=task, root=root, final=final, controller=controller,
                           cfg=cfg, data=data)


def project(study):
    return verify_portfolio_final_for_paper(study.data, task_dir=study.task,
        investigation_results_dir=study.root, final_evaluation_dir=study.final)


def test_complete_final_collection_exposes_all_methods_budgets_and_paired_metrics(final_study):
    study = final_study
    projection = project(study)
    assert projection["status"] == "verified", projection
    assert projection["collection_complete"] and projection["all_methods_qualified"]
    assert projection["method_count"] == projection["completed_method_count"] == 3
    assert [row["role"] for row in projection["methods"]] == ["proposal", "control", "control"]
    assert all(row["actual_evaluations"]["tuning"] == 1 for row in projection["methods"])
    assert all(row["planned_allocation"]["guaranteed_tuning"] == 1 for row in projection["methods"])
    assert all(row["observations"]["seeds"] == [2, 3, 4] for row in projection["methods"])
    assert projection["paired_comparison"]["aggregate_metrics"] == {
        "candidate_minus_controls_min": .5, "paired_controls_ci95_low_min": .5}
    assert projection["matched_hpo_verified"] is False
    study.data[FIELD] = projection
    assert verify_portfolio_final_for_paper(study.data) == projection
    readiness = assess_readiness(study.workspace, study.cfg, {})["checks"]
    assert readiness["portfolio_final_comparison"]["status"] == "verified"
    assert readiness["scientific_effect"]["status"] == "positive"
    assert readiness["scientific_effect"]["estimate"] == .5
    assert readiness["scientific_effect"]["evidence_source"] == "research_portfolio_final.paired_comparison"


@pytest.mark.parametrize("final_study", [{"negative": True}], indirect=True)
def test_positive_default_diagnostic_cannot_replace_negative_tuned_control_effect(final_study):
    study = final_study
    readiness = assess_readiness(study.workspace, study.cfg, {})["checks"]
    assert readiness["confirmation"]["metrics"]["candidate_minus_lme"] == .9
    assert readiness["scientific_effect"]["status"] == "not_positive"
    assert readiness["scientific_effect"]["estimate"] == -.5
    study.cfg.harness.quality.positive_effect_metric = "candidate_minus_lme"
    study.cfg.harness.quality.positive_effect_interval_low_metric = "paired_diagnostic_ci95_low"
    changed = assess_readiness(study.workspace, study.cfg, {})["checks"]
    assert changed["scientific_effect"]["status"] == "missing"


@pytest.mark.parametrize("final_study", [{"partial": True}, {"freeze_only": True}, {"missing_seed": True}], indirect=True)
def test_incomplete_final_method_set_or_control_seed_coverage_cannot_pass(final_study):
    study = final_study
    projection = project(study)
    assert projection["status"] == "incomplete", projection
    assert len(projection["methods"]) == 3
    assert not projection["all_methods_qualified"]
    readiness = assess_readiness(study.workspace, study.cfg, {})["checks"]
    assert readiness["portfolio_final_comparison"]["status"] == "incomplete"
    assert readiness["scientific_effect"]["status"] == "not_qualified"


@pytest.mark.parametrize("final_study", [{"winner": "control"}, {"promote": 1}, {"missing_configuration": True}], indirect=True)
def test_selected_control_or_untuned_controls_are_reportable_but_do_not_pass_superiority(final_study):
    study = final_study
    projection = project(study)
    assert projection["status"] == "verified", projection
    assert projection["collection_complete"]
    checks = assess_readiness(study.workspace, study.cfg, {})["checks"]
    assert checks["scientific_effect"]["status"] == "not_qualified"
    study.cfg.harness.quality.require_positive_final = False
    checks = assess_readiness(study.workspace, study.cfg, {})["checks"]
    assert checks["portfolio_final_comparison"]["status"] == "verified"
    assert checks["scientific_effect"]["status"] == "not_required"


@pytest.mark.parametrize("damage", ["missing_control_manifest", "control_evidence", "set", "saved_projection"])
def test_missing_or_tampered_control_cannot_hide_behind_valid_winner(final_study, damage):
    study = final_study
    projection = project(study)
    study.data[FIELD] = projection
    control = projection["methods"][-1]
    if damage == "missing_control_manifest":
        Path(control["final_manifest_path"]).unlink()
    elif damage == "control_evidence":
        Path(control["evaluation_evidence_path"]).chmod(0o600)
        Path(control["evaluation_evidence_path"]).write_text("{}")
    elif damage == "set":
        (study.final / SET_MANIFEST).chmod(0o600)
        write(study.final / SET_MANIFEST, {})
    else:
        study.data[FIELD]["methods"][-1]["metrics"]["combined_score"] = .01
    assert (study.final / "final_manifest.json").is_file()
    current = verify_portfolio_final_for_paper(study.data)
    assert current["verification_status"] == "invalid_or_missing", current
    checks = assess_readiness(study.workspace, study.cfg, {})["checks"]
    if damage != "saved_projection":
        assert checks["scientific_effect"]["status"] != "positive"


def test_paper_assembly_prompt_and_review_revalidate_every_control(final_study):
    study = final_study
    frozen = deepcopy(study.controller.data["frozen_candidate"])
    _merge_final_evaluation(study.data, study.final, task_dir=study.task, investigation_results_dir=study.root)
    projection = study.data[FIELD]
    assert projection["status"] == "verified", projection
    grounding = study.data["grounding_sources"]
    assert grounding["final.portfolio.idea_0003.metric.combined_score"]["value"] == .5
    assert grounding["final.portfolio.comparison.candidate_minus_controls_min"]["value"] == .5
    paper = study.run / "paper"
    write(paper / "paper_data.json", study.data)
    prompt = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    assert "FROZEN PORTFOLIO FINAL COMPARISON EVIDENCE" in prompt
    assert '"method_id": "idea_0003"' in prompt
    paths = SimpleNamespace(paper_dir=paper, investigation_dir=study.root, run_dir=study.run)
    packet = _selected_novelty_evidence(paths)
    assert read(packet)[FIELD] == projection
    assert study.controller.data["frozen_candidate"] == frozen
    Path(projection["methods"][-1]["evaluation_evidence_path"]).chmod(0o600)
    Path(projection["methods"][-1]["evaluation_evidence_path"]).write_text("{}")
    stale = _selected_novelty_evidence(paths)
    assert stale != packet
    assert read(stale)[FIELD]["verification_status"] == "invalid_or_missing"
    prompt = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    assert '"verification_status": "invalid_or_missing"' in prompt


@pytest.mark.parametrize("final_study", [{"partial": True}], indirect=True)
def test_partial_assembly_keeps_all_expected_methods_without_complete_comparison_grounding(final_study):
    study = final_study
    _merge_final_evaluation(study.data, study.final, task_dir=study.task, investigation_results_dir=study.root)
    assert study.data[FIELD]["status"] == "incomplete"
    assert study.data[FIELD]["method_count"] == 3
    assert study.data[FIELD]["completed_method_count"] == 1
    assert not any(key.startswith("final.portfolio.comparison.") for key in study.data["grounding_sources"])


def test_legacy_single_final_adds_no_portfolio_field_or_readiness_check(tmp_path):
    task = _write_contract_task(tmp_path)
    root, _ = _write_investigation(tmp_path)
    final = tmp_path / "final"
    run_final_evaluation(task_dir=task, investigation_results_dir=root, output_dir=final, python_bin=Path(sys.executable))
    data = {"results_dir": str(root)}
    _merge_final_evaluation(data, final, task_dir=task, investigation_results_dir=root)
    assert FIELD not in data
    assert not any(key.startswith("final.portfolio.") for key in data["grounding_sources"])
    assert verify_portfolio_final_for_paper(data) is None
    assert "portfolio_final_comparison" not in assess_readiness(tmp_path, ResearchConfig(), {})["checks"]
