#!/usr/bin/env python3
"""Inspect or resume a stopped investigation from its existing framework state."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
from pathlib import Path
import runpy
import shlex
import uuid

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = '''
import hashlib,json,sys
from pathlib import Path
from src.pipeline.config import validate_investigation_resume_state
from src.pipeline.resume_config import read_run_idea_cycle_budget
p=Path(sys.argv[1]);expected=json.loads(sys.argv[2])
errors=validate_investigation_resume_state(p/'investigation')
budget=read_run_idea_cycle_budget(p)
state=json.loads((p/'investigation/outer_state.json').read_text())
for name,digest in expected.items():
    if hashlib.sha256((p/'task'/name).read_bytes()).hexdigest()!=digest:
        errors.append('Private evaluator changed: '+name)
phase_events=[]
for line in (p/'pipeline_events.jsonl').read_text().splitlines():
    event=json.loads(line)
    if event.get('event') in ('phase_start','phase_end'):
        phase_events.append({'event':event['event'],**event.get('fields',{})})
last=phase_events[-1] if phase_events else {}
later=[n for n in ('ablations','final_evaluation','posthoc','paper')
       if (p/n).exists() and any((p/n).iterdir())]
print(json.dumps({'errors':errors,'cycle_budget':budget,'current_cycle':state.get('current_cycle'),
                  'last_phase_event':last,'later_outputs':later}))
'''

ARCHIVE = '''import json,shutil,sys
from pathlib import Path
p=Path(sys.argv[1]);a=Path(sys.argv[2]);a.mkdir(parents=True,exist_ok=False)
for name in ('exit-code','pid','run.log','run.sh'):
    shutil.copy2(p/name,a/name)
print(json.dumps({'archive':str(a)}))
'''


def require_resumable(record: dict, state: dict, witness: dict, validation: dict) -> None:
    """A retry must preserve existing research and cannot overlap another run."""
    authorization = record.get('budget_authorization', {})
    if (not authorization.get('automatic_recovery_authorized')
            or authorization.get('overall_time_limit_seconds', 'missing') is not None
            or authorization.get('overall_model_call_limit', 'missing') is not None):
        raise ValueError('Uncapped automatic recovery was not authorized')
    if record.get('phase') != 'investigation' or state['running']:
        raise ValueError('Only a stopped investigation can resume; refusing overlap')
    if state['exit_code'] in (None, 0):
        raise ValueError('Investigation has no recorded failed exit')
    blockers = set(record.get('blocking_findings', []))
    development = record.get('development_campaign')
    if development:
        if (development.get('version') != 2
                or development.get('max_idea_cycles') != 4
                or development.get('final_evaluation_enabled') is not False
                or not record.get('research_restart_authorization', {}).get('human_authorized')):
            raise ValueError('Development recovery must retain its authorized final-data boundary')
        # This finding blocks confirmatory claims and final evaluation, not
        # the separately authorized SEARCH-only development campaign.
        blockers.discard('independent_confirmatory_evidence_not_available')
    if not validation.get('ready') or blockers:
        raise ValueError('Scientific validation is not valid for this source')
    event = witness['last_phase_event']
    if (event.get('event') != 'phase_end' or event.get('phase_name') != 'investigation'
            or event.get('status') != 'failed' or witness['later_outputs']):
        raise ValueError('Latest failed phase is not an unfinished investigation')
    budget, current = witness['cycle_budget'], witness['current_cycle']
    expected = record['current_investigation_progress']['max_cycles']
    if development and expected != development['max_idea_cycles']:
        raise ValueError('Development recovery cannot reset its frozen cycle budget')
    if (witness['errors'] or type(budget) is not int or budget != expected
            or type(current) is not int or not 0 <= current < budget):
        raise ValueError('No valid unfinished framework checkpoint with the original cycle budget')


def recovery_overrides(record: dict, source: str, image: str) -> list[str]:
    if record.get('development_campaign'):
        restart = runpy.run_path(str(ROOT / 'scripts/research/restart_medical_research.py'))
        return restart['overrides'](source, image)
    continuation = runpy.run_path(str(ROOT / 'scripts/research/continue_medical_experiment.py'))
    return continuation['overrides'](source, image) + continuation['agent_wait_overrides'](record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Resume only after the current run stops')
    args = parser.parse_args()
    launcher = runpy.run_path(str(ROOT / 'scripts/remote-agsci'))
    draft = runpy.run_path(str(ROOT / 'scripts/research/resume_medical_drafting.py'))
    continuation = runpy.run_path(str(ROOT / 'scripts/research/continue_medical_experiment.py'))
    remote, base = launcher['remote'], launcher['REMOTE']
    record_path = ROOT / 'docs/research/medical-unlearning/execution.json'
    (ROOT / 'scratch').mkdir(exist_ok=True)
    with (ROOT / 'scratch/medical-recovery-submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(record_path.read_text())
        job = record['job_id']
        if record['remote_results'] != base + '/results/' + job:
            raise ValueError('The recorded run must match its original results directory')
        directory, results = base + '/jobs/' + job, record['remote_results']
        source = record['qualified_source_task']['path']
        source_job = record['qualified_source_task']['source_job']
        if source != base + '/results/' + source_job + '/task':
            raise ValueError('Qualified source is outside its recorded result directory')
        probe = (ROOT / 'scripts/research/medical_job_state.py').read_text()

        def inspect(identity: str = job) -> dict:
            command = 'python3 - ' + shlex.join([base, identity]) + " <<'PY'\n" + probe + '\nPY\n'
            return json.loads(remote(command, capture=True).stdout)

        def run_python(checkout: str, script: str, arguments: list[str]) -> dict:
            command = ('cd ' + shlex.quote(checkout) + '\n'
                       + shlex.join([base + '/.venv/bin/python', '-', *arguments])
                       + " <<'PY'\n" + script + '\nPY\n')
            return json.loads(remote(command, capture=True).stdout)

        state = inspect()
        source_state = inspect(source_job)
        validation = {'ready': False, 'reason': 'Qualified source is active or failed'}
        if not source_state['running'] and source_state['exit_code'] == 0:
            validation = run_python(base + '/jobs/' + source_job, continuation['VERIFY'],
                                    [source, record['boundary_f1_repair']['expected_metric_ast_sha256']])
        digests = {name: value['after'] for name, value in record['boundary_f1_repair']['files'].items()}
        witness = run_python(directory, CHECKPOINT, [results, json.dumps(digests)])
        try:
            require_resumable(record, state, witness, validation)
        except ValueError as error:
            if args.resume:
                raise
            print(json.dumps({'job_id': job, 'ready': False, 'reason': str(error),
                              'state': state, 'checkpoint': witness, 'validation': validation}, indent=2))
            return
        if not args.resume:
            print(json.dumps({'job_id': job, 'ready': True, 'state': state,
                              'checkpoint': witness, 'validation': validation}, indent=2))
            return

        # Reuse the frozen job code and the original results directory. No research
        # files or environments are copied, reset, or removed by this launcher.
        extra = (recovery_overrides(record, source, validation['image_id'])
                 + ['investigation.execution.resume=true',
                    'investigation.execution.max_idea_cycles=' + str(witness['cycle_budget']),
                    'run_id=' + job, 'paths.run_dir=' + results])
        command = launcher['args_for']('run', extra, directory)
        runner = draft['runner_text'](launcher, directory, command)
        attempt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        archive = directory + '/attempts/' + attempt
        require_resumable(record, inspect(), run_python(directory, CHECKPOINT,
                          [results, json.dumps(digests)]), validation)
        run_python(directory, ARCHIVE, [directory, archive])
        path = archive + '/resume.sh'
        remote('printf %s ' + shlex.quote(runner) + ' > ' + shlex.quote(path) + '\n')
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        recovery = {'attempt_id': attempt, 'job_id': job, 'prior_exit_code': state['exit_code'],
                    'archive': archive, 'resumed_at': now, 'checkpoint': witness,
                    'model_watchdog_seconds': 1800 if record.get('development_campaign') else 7200,
                    'scientific_protocol_changed': False,
                    'framework_code_replaced': False}
        record.setdefault('investigation_recoveries', []).append(recovery)
        record.update(updated_at=now, status='submitting_investigation_checkpoint_resume')
        draft['atomic_record'](record_path, record)
        remote('cd ' + shlex.quote(directory) + '\ntest "$(cat exit-code)" = '
               + shlex.quote(str(state['exit_code'])) + '\nmv exit-code '
               + shlex.quote(archive + '/stopped-exit-code') + '\nnohup bash ' + shlex.quote(path)
               + ' > run.log 2>&1 < /dev/null &\necho $! > pid\n')
        if record.get('development_campaign'):
            sampler = run_python(directory, '''import json,subprocess,sys
from pathlib import Path
d=Path(sys.argv[1]);a=Path(sys.argv[2]);pid=int((d/'pid').read_text())
csv=a/'gpu_samples.csv';state=a/'gpu_watch.json'
with (a/'gpu_watch.log').open('w')as out:
 p=subprocess.Popen([sys.executable,str(d/'scripts/research/watch_research_gpus.py'),
    '--pid',str(pid),'--uuids',sys.argv[3],'--csv',str(csv),'--state',str(state)],
    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'csv':str(csv),'state':str(state),'target_pid':pid}))
''', [directory,archive,','.join(launcher['CFG']['gpu_uuids'])])
            record.setdefault('gpu_sampling_history', []).append(record['gpu_sampling'])
            record['gpu_sampling'] = sampler
        record['status'] = 'running_investigation'
        record['substage'] = 'idea_generation_cycle_' + str(witness['current_cycle'] + 1)
        record['next_action'] = {'requires_user_decision': False,
                                'action': 'Monitor the resumed investigation and original downstream phases. '
                                          'If investigation fails again, inspect its retained framework checkpoint '
                                          'with resume_medical_investigation.py before --resume; do not redraft.'}
        if record.get('development_campaign'):
            record['next_action']['action'] = (
                'Directly monitor the same four development cycles and inspect native results. '
                'The old final stays quarantined, all schedules stay paused, and no later phase is enabled.')
        record['last_monitor_check'] = {'checked_at': now, 'status': 'resumed_existing_investigation_checkpoint',
                                        'restart_performed': True, 'repair_applied': True,
                                        'prior_exit_code': state['exit_code'], 'attempt_id': attempt}
        draft['atomic_record'](record_path, record)
        print(json.dumps(recovery, indent=2))


if __name__ == '__main__':
    main()
