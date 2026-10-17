"""Recovery keeps the newest work, prevents duplicate jobs, and honors authorization."""
import os
import configparser
from pathlib import Path
import runpy
import shlex
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = runpy.run_path(str(ROOT / 'scripts/research/resume_medical_drafting.py'))
STATE = runpy.run_path(str(ROOT / 'scripts/research/medical_job_state.py'))
JOB = '20261017_070528_ba5c6c51'


def create_draft(path):
    (path / 'task').mkdir(parents=True)
    (path / 'task/task.md').write_text('unchanged task')
    (path / '_task_design').mkdir()
    (path / '_task_design/design_receipt.json').write_text('{}')
    return path


def test_discover_latest_nested_draft(tmp_path):
    early = create_draft(tmp_path / 'task_drafting/_task_drafting/draft_old')
    latest = create_draft(tmp_path / 'task_drafting/_task_drafting/draft_new')
    os.utime(early / 'task/task.md', ns=(10, 10))
    os.utime(latest / 'task/task.md', ns=(20, 20))
    assert STATE['latest_draft'](tmp_path) == latest


def test_inspect_direct_recovery_and_orphan_container(tmp_path):
    directory = tmp_path / 'jobs' / JOB
    directory.mkdir(parents=True)
    (directory / 'exit-code').write_text('124\n')
    draft = create_draft(tmp_path / 'results' / JOB / 'task_drafting')
    state = STATE['inspect_job'](tmp_path, JOB, process_ids=[], container_ids=['orphan'])
    assert state['running'] is True
    assert state['source_draft'] == str(draft)
    assert state['exit_code'] == 124


def authorized():
    return {'budget_authorization': {'automatic_recovery_authorized': True, 'overall_time_limit_seconds': None}}


@pytest.mark.parametrize('state', [
    {'running': True, 'exit_code': None, 'source_draft': '/new'},
    {'running': False, 'exit_code': 0, 'source_draft': '/new'},
    {'running': False, 'exit_code': 4, 'source_draft': None},
])
def test_refuse_active_successful_or_missing_draft(state):
    with pytest.raises(ValueError):
        LAUNCH['require_recoverable'](authorized(), state)


def test_stale_record_status_does_not_block_verified_stopped_job():
    record = authorized() | {'status': 'running_drafting_recovery'}
    state = {'running': False, 'exit_code': 124, 'source_draft': '/latest'}
    LAUNCH['require_recoverable'](record, state)
    with pytest.raises(ValueError):
        LAUNCH['require_recoverable']({}, state)


def test_uncapped_runner_and_early_failure_receipt(tmp_path):
    launcher = {'CFG': {'gpu_uuids': ['gpu0', 'gpu3']}, 'REMOTE': str(tmp_path),
                'preamble': lambda _: 'false\n'}
    cmd = LAUNCH['drafting_command']('/job', '/result', '/latest', '/base')
    assert cmd[cmd.index('--accepted-design-dir') + 1] == '/latest/_task_design'
    assert cmd[cmd.index('--qualification-timeout-seconds') + 1] == '0'
    script = LAUNCH['runner_text'](launcher, str(tmp_path), cmd)
    assert 'timeout --' not in script
    result = subprocess.run(['bash', '-c', script])
    assert result.returncode == 1
    assert (tmp_path / 'exit-code').read_text() == '1\n'


@pytest.mark.parametrize('budget,elapsed,expected', [(0, 10000000, 604800), (7200, 100, 7070), (7200, 8000, -830)])
def test_shared_deadline_can_be_disabled_without_changing_finite_mode(budget, elapsed, expected):
    library = ROOT / 'src/scripts/lib/task_qualification.sh'
    script = f'source {shlex.quote(str(library))}\nQUALIFICATION_TIMEOUT_SECONDS={budget}\nSECONDS={elapsed}\nqualification_remaining_seconds 30'
    value = subprocess.check_output(['bash', '-c', script], text=True).strip()
    assert int(value) == expected


@pytest.mark.parametrize('index', ['', 'https://pypi.tuna.tsinghua.edu.cn/simple'])
def test_review_package_index_is_scoped_to_disposable_judge_home(tmp_path, index):
    import sys
    home = tmp_path / 'judge'; home.mkdir()
    library = ROOT / 'src/scripts/lib/task_qualification.sh'
    env = os.environ | {'PYTHON_BIN': sys.executable, 'TASK_REVIEW_PACKAGE_INDEX': index}
    subprocess.run(['bash', '-c', f'source {shlex.quote(str(library))}\nprepare_alignment_package_index {shlex.quote(str(home))}'],env=env,check=True)
    config_path = home / '.config/pip/pip.conf'
    if not index:
        assert not config_path.exists()
    else:
        config = configparser.ConfigParser(); config.read(config_path)
        assert config['global']['index-url'] == index
        assert config_path.stat().st_mode & 0o777 == 0o600
