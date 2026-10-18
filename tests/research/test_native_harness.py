"""Native development receipts must never masquerade as sealed evaluations."""
import json
import sys

import pytest

from src.investigation.harness_incumbent import reconcile_harness_incumbent
from src.investigation.native_harness import ensure_native_baseline, run_native_evaluation, validated_native_score
from tests.program_testutils import make_program_dir


def fixture_task(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    program = task / 'program_0'
    make_program_dir(program, content='SCORE = 0.6\n')
    (task / 'evaluate.py').write_text('''import argparse, json, runpy
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--program_path');p.add_argument('--results_dir')
a=p.parse_args();out=Path(a.results_dir)
score=runpy.run_path(a.program_path)['SCORE']
(out/'metrics.json').write_text(json.dumps({'combined_score':score}))
(out/'correct.json').write_text(json.dumps({'correct':True}))
''')
    return task, program


def test_native_baseline_evaluates_once_and_does_not_modify_task(tmp_path):
    task, program = fixture_task(tmp_path)
    results = tmp_path / 'results'
    results.mkdir()
    args = dict(task_dir=task, seed_program_dir=program, results_dir=results, timeout_seconds=10)
    assert ensure_native_baseline(**args) == 0.6
    receipt = results / '.native-baseline/program_0/evaluation_manifest.json'
    before = receipt.read_bytes()
    assert ensure_native_baseline(**args) == 0.6
    assert receipt.read_bytes() == before
    assert not (program / 'metrics.json').exists()
    record = json.loads(before)
    assert record['execution_boundary'] == 'host'
    assert record['producer'] == 'harness_eval'
    (program / 'src/program.py').write_text('SCORE = 0.7\n')
    with pytest.raises(ValueError, match='changed'):
        ensure_native_baseline(**args)


def test_native_receipt_requires_matching_code_evaluator_and_metrics(tmp_path):
    task, program = fixture_task(tmp_path)
    evaluator = task / 'evaluate.py'
    assert run_native_evaluation(program_dir=program, evaluator=evaluator, python_bin=sys.executable,
        timeout_seconds=10, environment={}, attempt_id='test', idea_id='test') == 0
    assert validated_native_score(program, evaluator) == 0.6
    for path, replacement in ((program/'src/program.py', 'SCORE = 1\n'),
                              (program/'metrics.json', '{"combined_score":1}'),
                              (evaluator, '# changed')):
        original = path.read_text()
        path.write_text(replacement)
        assert validated_native_score(program, evaluator) is None
        path.write_text(original)
        assert validated_native_score(program, evaluator) == 0.6


def test_native_candidate_is_reusable_only_in_native_mode(tmp_path):
    task, seed = fixture_task(tmp_path)
    results = tmp_path/'results'
    candidate = results/'idea/program_1'
    make_program_dir(candidate, content='SCORE = 0.9\n')
    evaluator = task/'evaluate.py'
    assert run_native_evaluation(program_dir=candidate, evaluator=evaluator, python_bin=sys.executable,
        timeout_seconds=10, environment={}, attempt_id='one', idea_id='idea') == 0
    arguments = dict(task_dir=task, results_dir=results, seed_program_dir=seed,
        baseline_score=0.6, baseline_correct=True)
    manifest = {'program_dir': str(candidate), 'combined_score': 0.9}
    assert reconcile_harness_incumbent(**arguments, best_manifest=manifest, outer_state={})
    assert manifest['combined_score'] == 0.6
    manifest = {'program_dir': str(candidate), 'combined_score': 0.9}
    assert reconcile_harness_incumbent(**arguments, best_manifest=manifest, outer_state={},
        native_evaluator=evaluator)
    assert manifest['combined_score'] == 0.9


def test_native_baseline_rejects_contracted_task(tmp_path):
    task, program = fixture_task(tmp_path)
    (task/'task_contract.json').write_text('{}')
    with pytest.raises(ValueError, match='sealed runtime'):
        ensure_native_baseline(task_dir=task, seed_program_dir=program,
            results_dir=tmp_path/'results', timeout_seconds=10)
