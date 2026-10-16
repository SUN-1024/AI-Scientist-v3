import hashlib
import json
import pytest
from src.task_drafting.design import TaskDesignError, load_task_design
from src.task_drafting.recovery import restore_design
from tests.task_drafting.test_task_design import _design, _review


def artifacts(tmp_path):
    task = tmp_path / 'task.md'; task.write_text('Fixed research protocol')
    source = tmp_path / 'accepted'; source.mkdir()
    design = source / 'task_design.json'; design.write_text(json.dumps(_design(task)))
    review = source / 'accepted_review.json'
    review.write_text(json.dumps(_review(load_task_design(design, canonical_task=task).fingerprint)))
    receipt = {'schema_version': 1, 'verdict': 'pass',
               'design_sha256': hashlib.sha256(design.read_bytes()).hexdigest(),
               'review_sha256': hashlib.sha256(review.read_bytes()).hexdigest()}
    (source / 'design_receipt.json').write_text(json.dumps(receipt))
    return source, task


def test_recovery_preserves_accepted_design(tmp_path):
    source, task = artifacts(tmp_path)
    output = tmp_path / 'restored'
    restore_design(source, task, output)
    assert (output / 'task_design.json').read_bytes() == (source / 'task_design.json').read_bytes()
    assert json.loads((output / 'recovery.json').read_text())['scaffold_validation_reused'] is False


@pytest.mark.parametrize('change', ['task', 'design', 'review'])
def test_recovery_rejects_changed_protocol_or_artifacts(tmp_path, change):
    source, task = artifacts(tmp_path)
    target = task if change == 'task' else source / ('task_design.json' if change == 'design' else 'accepted_review.json')
    target.write_text(target.read_text() + ' ')
    with pytest.raises(TaskDesignError):
        restore_design(source, task, tmp_path / 'restored')
