"""Downstream work cannot bypass the preceding task's verified completion."""
from pathlib import Path
import json
import runpy
import subprocess
import sys
import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/research/continue_medical_experiment.py'))


@pytest.mark.parametrize('state,verified', [
    ({'running':True,'exit_code':0}, {'ready':True}),
    ({'running':False,'exit_code':None}, {'ready':True}),
    ({'running':False,'exit_code':4}, {'ready':True}),
    ({'running':False,'exit_code':0}, {'ready':False}),
])
def test_downstream_refuses_unfinished_failed_or_unverified_draft(state,verified):
    with pytest.raises(ValueError):
        M['require_ready'](state,verified)


def test_completed_verified_draft_can_continue_with_original_protocol():
    M['require_ready']({'running':False,'exit_code':0},{'ready':True})
    values = M['overrides']('/latest/published/task','sha256:verified')
    assert 'preset=medical_unlearning' in values
    assert 'task.dir=/latest/published/task' in values
    assert 'container.image=sha256:verified' in values
    assert not any(v.startswith(('investigation.execution.','ablations.execution.','final.')) for v in values)


@pytest.mark.parametrize('changed', [False, True])
def test_release_checks_corrected_metric_before_remaining_gates(tmp_path, changed):
    root = Path(__file__).resolve().parents[1]
    metric = runpy.run_path(str(root / 'scripts/research/fix_medical_boundary_f1.py'))
    source = metric['COUNTS'] + '\n\n' + metric['MACRO'] + '\n'
    (tmp_path / 'evaluate.py').write_text(source)
    (tmp_path / 'evaluate_final.py').write_text(
        source.replace('precision + recall', 'precision + 2 * recall') if changed else source
    )
    run = subprocess.run([sys.executable, '-c', M['VERIFY'], str(tmp_path), metric['EXPECTED_AST']],
                         cwd=root, text=True, capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert not result['ready']
    if changed:
        assert 'Verified boundary metric changed: evaluate_final.py' in result['reason']
    else:
        # The correct metric alone cannot release a task with no contract/receipts.
        assert 'task_contract.json' in result['reason']
