"""Downstream work cannot bypass the preceding task's verified completion."""
from pathlib import Path
import runpy
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
