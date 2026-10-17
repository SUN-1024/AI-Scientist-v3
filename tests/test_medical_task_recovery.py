"""Recovery seeds must satisfy the real publication resource ownership consumer."""
from pathlib import Path
import runpy
import pytest

ROOT = Path(__file__).resolve().parents[1]
M = runpy.run_path(str(ROOT / 'scripts/research/medical_task_recovery.py'))


@pytest.mark.skipif(not Path('/proc/self/fd').exists(), reason='Real publication filesystem is Linux')
def test_recovery_seed_has_owned_identical_resources_and_no_stale_runtime(tmp_path):
    from src.task_drafting.publication import _require_matching_managed_resources
    source=tmp_path/'saved';source.mkdir()
    (source/'task.md').write_text('frozen canonical task\n')
    (source/'evaluate.py').write_text('# exact evaluator\n')
    (source/'resources').mkdir();(source/'resources/packet.bin').write_bytes(b'fixed packet')
    (source/'.venv/bin').mkdir(parents=True);(source/'.venv/bin/python').write_text('old environment')
    (source/'.agentic_runtime.json').write_text('{}')
    result=M['stage_recovered_task'](source,tmp_path/'seed')
    assert result['scientific_validation_reused'] is False
    _require_matching_managed_resources(source/'resources',tmp_path/'seed/resources')
    assert (tmp_path/'seed/resources/packet.bin').read_bytes()==b'fixed packet'
    assert (tmp_path/'seed/evaluate.py').read_bytes()==(source/'evaluate.py').read_bytes()
    assert not (tmp_path/'seed/.venv').exists()
    assert not (tmp_path/'seed/.agentic_runtime.json').exists()
    assert (source/'.venv/bin/python').read_text()=='old environment'


def test_recovery_never_overwrites_an_existing_destination(tmp_path):
    source=tmp_path/'saved';source.mkdir();(source/'task.md').write_text('frozen')
    target=tmp_path/'existing';target.mkdir();(target/'keep').write_text('keep')
    with pytest.raises(FileExistsError):M['stage_recovered_task'](source,target)
    assert (target/'keep').read_text()=='keep'
