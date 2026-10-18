"""Sealed host receipts also preserve negative observations and source identity."""
from __future__ import annotations

import json
from unittest.mock import patch

from src.investigation.adaptive_runtime import recovered_attempt, validated_development_outcome
from src.investigation.harness_eval import SealedHarnessConfig
from src.investigation.harness_incumbent import validated_harness_candidate_score
from src.variants_pipeline.host_eval import HostEvaluationResult
from tests.investigation.test_harness_eval import _evaluate, _sealed_experiment


def test_brokered_search_is_bound_by_host_wrapper_and_false_result_is_not_promoted(tmp_path):
    experiment, task, rubrics, python = _sealed_experiment(tmp_path)
    program = experiment / "program_1"

    def evaluate(**kwargs):
        (program / "metrics.json").write_text('{"combined_score": 0.9}')
        (program / "correct.json").write_text('{"correct": false}')
        # The inner broker receipt is retained in the existing wrapper envelope.
        (program / "evaluation_manifest.json").write_text(json.dumps({
            "execution_boundary": "brokered_search", "returncode": 0,
            "command": ["docker", "run"],
        }))
        return HostEvaluationResult(returncode=0, correct=False, metrics={"combined_score": 0.9}, command=("docker", "run"))

    with patch("src.variants_pipeline.host_eval.evaluate_variant_program", side_effect=evaluate) as evaluator:
        sealed = SealedHarnessConfig(task_source_dir=task, sandbox_image="sha256:test", sandbox_gpus=None, environment={})
        code = _evaluate(experiment, rubrics, python_bin=str(python), sealed_config=sealed)
        assert _evaluate(experiment, rubrics, python_bin=str(python), sealed_config=sealed) == 0
        assert evaluator.call_count == 1
    assert code == 0
    manifest = json.loads((program / "evaluation_manifest.json").read_text())
    assert manifest["execution_boundary"] == "sandbox"
    args = {"task_dir": task, "attempt_id": manifest["attempt_id"], "idea_id": manifest["idea_id"], "native": False}
    outcome = validated_development_outcome(program, **args)
    assert outcome and outcome["correct"] is False and outcome["score"] == 0.9
    assert validated_harness_candidate_score(program, attempt_id=manifest["attempt_id"], idea_id=manifest["idea_id"]) is None
    assert recovered_attempt(experiment, **args) is not None
    (program / "metrics.json").write_text('{"combined_score": 100}')
    assert validated_development_outcome(program, **args) is None
    assert recovered_attempt(experiment, **args) is None


def test_sealed_host_rejects_source_changes_during_execution(tmp_path):
    experiment, task, rubrics, python = _sealed_experiment(tmp_path)
    program = experiment / "program_1"

    def evaluate(**kwargs):
        (program / "metrics.json").write_text('{"combined_score": 0.9}')
        (program / "correct.json").write_text('{"correct": true}')
        (program / "src/program.py").write_text("print('changed during eval')\n")
        return HostEvaluationResult(returncode=0, correct=True, metrics={"combined_score": 0.9}, command=("docker", "run"))

    with patch("src.variants_pipeline.host_eval.evaluate_variant_program", side_effect=evaluate):
        code = _evaluate(experiment, rubrics, python_bin=str(python), sealed_config=SealedHarnessConfig(
            task_source_dir=task, sandbox_image="sha256:test", sandbox_gpus=None, environment={}))
    assert code == 1
    manifest = json.loads((program / "evaluation_manifest.json").read_text())
    assert manifest["error"] == "candidate changed during sealed evaluation"
    assert validated_development_outcome(program, task_dir=task, attempt_id=manifest["attempt_id"], idea_id=manifest["idea_id"], native=False) is None
