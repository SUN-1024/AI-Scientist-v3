#!/usr/bin/env python3
"""Continue the medical study only after its published draft is fully qualified."""
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
# Run inside the selected job checkout so verification uses its harness version.
VERIFY = '''
import json,sys
from pathlib import Path
from src.investigation.task_alignment import load_task_alignment_report
from src.investigation.task_contract import load_task_contract
from src.task_drafting.publication import validate_published_task_plan
from src.task_drafting.qualification import load_task_qualification_receipt
p=Path(sys.argv[1])
try:
    load_task_contract(p/'task_contract.json')
    alignment=load_task_alignment_report(p/'.agentic_task_validation.json',scaffold_dir=p)
    if alignment.verdict != 'pass':
        raise ValueError('Independent alignment did not pass')
    plan=validate_published_task_plan(p)
    receipt=load_task_qualification_receipt(p/'.agentic_task_qualification.json',expected_publication_plan_sha256=plan)
    if not receipt.baseline_correct or not receipt.baseline_feasible:
        raise ValueError('Full production baseline was not correct and feasible')
    print(json.dumps({'ready':True,'publication_plan_sha256':plan,'image_id':receipt.image_id,
                      'runtime_cache_key':receipt.runtime_cache_key}))
except (OSError,ValueError,RuntimeError) as e:
    print(json.dumps({'ready':False,'reason':type(e).__name__+': '+str(e)[:500]}))
'''


def require_ready(state: dict, verification: dict) -> None:
    if state['running']:
        raise ValueError('Previous task is still active; refusing overlap')
    if state['exit_code'] != 0:
        raise ValueError('Draft task did not complete successfully')
    if not verification.get('ready'):
        raise ValueError('Independent validation/qualification receipts are not valid')


def overrides(source: str, image: str) -> list[str]:
    return ['preset=medical_unlearning', 'task.dir=' + source,
            "execution.phases='investigation,ablations,final_evaluation,posthoc,paper'",
            'container.image=' + image]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Submit if all verification gates pass')
    args = parser.parse_args()
    launcher = runpy.run_path(str(ROOT / 'scripts/remote-agsci'))
    recovery = runpy.run_path(str(ROOT / 'scripts/research/resume_medical_drafting.py'))
    remote = launcher['remote']; base = launcher['REMOTE']
    record_path = ROOT / 'docs/research/medical-unlearning/execution.json'
    (ROOT / 'scratch').mkdir(exist_ok=True)
    with (ROOT / 'scratch/medical-recovery-submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(record_path.read_text()); old = record['job_id']
        if record['phase'] != 'task_drafting':
            raise ValueError('Current job is already a downstream phase; inspect it instead')
        probe = (ROOT / 'scripts/research/medical_job_state.py').read_text()
        def inspect():
            result = remote('python3 - ' + shlex.join([base, old]) + " <<'PY'\n" + probe + '\nPY\n', capture=True)
            return json.loads(result.stdout)
        state = inspect()
        source = state['published_task']
        directory = base + '/jobs/' + old
        verification = {'ready': False, 'reason': 'Draft task is still active'}
        if not state['running'] and state['exit_code'] == 0:
            result = remote('cd ' + shlex.quote(directory) + '\n' + shlex.quote(base + '/.venv/bin/python')
                            + ' - ' + shlex.quote(source) + " <<'PY'\n" + VERIFY + '\nPY\n', capture=True)
            verification = json.loads(result.stdout)
        if not args.run:
            print(json.dumps({'job_id':old,'state':state,'verification':verification},indent=2))
            return
        require_ready(state, verification)
        job = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        directory, results = base + '/jobs/' + job, base + '/results/' + job
        launcher['sync'](directory); launcher['stage_auth'](directory)
        remote('mkdir -p ' + shlex.quote(results) + '\n')
        # Pin the exact image that passed qualification; no floating-tag switch.
        cmd = launcher['args_for']('run', overrides(source,verification['image_id']),directory)
        cmd += ['run_id='+job,'paths.run_dir='+results]
        require_ready(inspect(), verification)
        runner = recovery['runner_text'](launcher,directory,cmd)
        remote('printf %s ' + shlex.quote(runner) + ' > ' + shlex.quote(directory+'/run.sh') + '\n')
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        record.setdefault('execution_history',[]).append({
            key:record.get(key) for key in ('job_id','status','phase','remote_log','remote_results','last_monitor_check')})
        record.update(updated_at=now,job_id=job,status='submitting_investigation',phase='investigation',
                      remote_log=directory+'/run.log',remote_results=results)
        record['qualified_source_task']={'path':source,'verified_at':now,**verification}
        record['verification_gate'].update(current_passed=record['verification_gate']['required_before_downstream'],current_pending=[])
        record['next_action']={'requires_user_decision':False,'action':'Monitor each downstream phase; diagnose and repair actual failures; preserve scientific protocol and the two assigned A40s'}
        recovery['atomic_record'](record_path,record)
        remote('cd '+shlex.quote(directory)+'\nnohup bash run.sh > run.log 2>&1 < /dev/null &\necho $! > pid\n')
        record['status']='running_investigation'
        record['last_monitor_check']={'checked_at':now,'status':'submitted_qualified_investigation',
                                     'restart_performed':False,'repair_applied':False,'scientific_results_available':False}
        recovery['atomic_record'](record_path,record)
        print(job)


if __name__ == '__main__':
    main()
