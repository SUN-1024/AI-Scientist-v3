"""Native development receipts must never masquerade as sealed evaluations."""
import json
import subprocess
import sys
from unittest.mock import Mock

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


def declare_inputs(task):
    (task / 'protocol.py').write_text('STEPS = 2\n')
    (task / 'resources').mkdir()
    (task / 'resources' / 'search.npz').write_bytes(b'fixture data')
    (task / 'native_evaluation_inputs.json').write_text(json.dumps({
        'schema_version': 1,
        'required': ['protocol.py', 'resources/search.npz'],
        'optional': ['evaluation_protocol.json'],
    }))


def evaluate_fixture(task, program):
    return run_native_evaluation(
        program_dir=program, evaluator=task / 'evaluate.py', python_bin=sys.executable,
        timeout_seconds=10, environment={}, attempt_id='test', idea_id='test',
    )


def test_receipt_binds_declared_helpers_data_and_optional_protocol(tmp_path):
    task, program = fixture_task(tmp_path)
    declare_inputs(task)
    evaluator = task / 'evaluate.py'
    assert evaluate_fixture(task, program) == 0
    receipt = json.loads((program / 'evaluation_manifest.json').read_text())
    fingerprints = receipt['evaluation_inputs_sha256']
    assert set(fingerprints) == {
        'native_evaluation_inputs.json', 'protocol.py', 'resources/search.npz', 'evaluation_protocol.json',
    }
    assert fingerprints['evaluation_protocol.json'] is None
    for relative in ('protocol.py', 'resources/search.npz', 'native_evaluation_inputs.json'):
        path = task / relative
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        assert validated_native_score(program, evaluator) is None
        path.write_bytes(original)
        assert validated_native_score(program, evaluator) == 0.6
        path.unlink()
        assert validated_native_score(program, evaluator) is None
        path.write_bytes(original)
    optional = task / 'evaluation_protocol.json'
    optional.write_text('{"max_steps": 10}')
    assert validated_native_score(program, evaluator) is None
    assert evaluate_fixture(task, program) == 0
    assert validated_native_score(program, evaluator) == 0.6
    optional.write_text('{"max_steps": 20}')
    assert validated_native_score(program, evaluator) is None
    optional.unlink()
    assert validated_native_score(program, evaluator) is None


def test_legacy_receipt_is_valid_only_without_a_declaration(tmp_path):
    task, program = fixture_task(tmp_path)
    assert evaluate_fixture(task, program) == 0
    receipt_path = program / 'evaluation_manifest.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.pop('evaluation_inputs_sha256')
    receipt_path.write_text(json.dumps(receipt))
    evaluator = task / 'evaluate.py'
    assert validated_native_score(program, evaluator) == 0.6
    declare_inputs(task)
    assert validated_native_score(program, evaluator) is None
    (task / 'native_evaluation_inputs.json').unlink()
    assert validated_native_score(program, evaluator) == 0.6


def test_unrelated_task_files_do_not_invalidate_declared_inputs(tmp_path):
    task, program = fixture_task(tmp_path)
    declare_inputs(task)
    assert evaluate_fixture(task, program) == 0
    (task / 'notes.md').write_text('This is not an evaluation input.\n')
    assert validated_native_score(program, task / 'evaluate.py') == 0.6


@pytest.mark.parametrize('change', ['helper', 'optional_added', 'required_deleted', 'declaration_deleted'])
def test_input_changes_during_evaluation_reject_the_receipt(tmp_path, monkeypatch, change):
    task, program = fixture_task(tmp_path)
    declare_inputs(task)

    def fake_run(command, **kwargs):
        (program / 'metrics.json').write_text('{"combined_score": 0.6}')
        (program / 'correct.json').write_text('{"correct": true}')
        if change == 'helper':
            (task / 'protocol.py').write_text('STEPS = 999\n')
        elif change == 'optional_added':
            (task / 'evaluation_protocol.json').write_text('{}')
        elif change == 'required_deleted':
            (task / 'resources' / 'search.npz').unlink()
        else:
            (task / 'native_evaluation_inputs.json').unlink()
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    monkeypatch.setattr('src.investigation.native_harness.subprocess.run', fake_run)
    assert evaluate_fixture(task, program) == 1
    receipt = json.loads((program / 'evaluation_manifest.json').read_text())
    assert receipt['exit_code'] == receipt['returncode'] == 1
    assert 'inputs changed during execution' in (program / 'harness_evaluate_stderr.log').read_text()
    assert validated_native_score(program, task / 'evaluate.py') is None


@pytest.mark.parametrize('relative', ['/outside.py', '../outside.py', 'x/../../outside.py',
                                      'C:/outside.py', 'x\\outside.py', './protocol.py',
                                      'resources//search.npz', ''])
def test_invalid_dependency_paths_block_evaluation(tmp_path, monkeypatch, relative):
    task, program = fixture_task(tmp_path)
    (task / 'native_evaluation_inputs.json').write_text(json.dumps({
        'schema_version': 1, 'required': [relative],
    }))
    runner = Mock(side_effect=AssertionError('invalid declarations must not execute'))
    monkeypatch.setattr('src.investigation.native_harness.subprocess.run', runner)
    with pytest.raises(ValueError, match='path'):
        evaluate_fixture(task, program)
    runner.assert_not_called()


@pytest.mark.parametrize('link_kind', ['file', 'parent_directory', 'declaration'])
def test_dependency_symlinks_are_rejected(tmp_path, monkeypatch, link_kind):
    task, program = fixture_task(tmp_path)
    declare_inputs(task)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'search.npz').write_bytes(b'other data')
    if link_kind == 'file':
        link = task / 'resources' / 'search.npz'
        link.unlink()
        link.symlink_to(outside / 'search.npz')
    elif link_kind == 'parent_directory':
        (task / 'resources' / 'search.npz').unlink()
        (task / 'resources').rmdir()
        (task / 'resources').symlink_to(outside, target_is_directory=True)
    else:
        declaration = task / 'native_evaluation_inputs.json'
        external = outside / 'declaration.json'
        declaration.rename(external)
        declaration.symlink_to(external)
    runner = Mock(side_effect=AssertionError('symlink inputs must not execute'))
    monkeypatch.setattr('src.investigation.native_harness.subprocess.run', runner)
    with pytest.raises(ValueError):
        evaluate_fixture(task, program)
    runner.assert_not_called()


def test_duplicate_declaration_keys_cannot_hide_required_inputs(tmp_path, monkeypatch):
    task, program = fixture_task(tmp_path)
    (task / 'native_evaluation_inputs.json').write_text(
        '{"schema_version": 1, "required": ["missing.py"], "required": []}'
    )
    runner = Mock(side_effect=AssertionError('ambiguous declarations must not execute'))
    monkeypatch.setattr('src.investigation.native_harness.subprocess.run', runner)
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate_fixture(task, program)
    runner.assert_not_called()
