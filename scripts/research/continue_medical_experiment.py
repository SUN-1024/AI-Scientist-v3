#!/usr/bin/env python3
"""Continue the medical study only after its published draft is fully qualified."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import runpy
import re
import shlex
import uuid

ROOT = Path(__file__).resolve().parents[2]
# Run inside the selected job checkout so verification uses its harness version.
VERIFY = '''
import ast,hashlib,json,sys
from pathlib import Path
from src.investigation.task_alignment import load_task_alignment_report
from src.investigation.task_contract import load_task_contract
from src.task_drafting.publication import validate_published_task_plan
from src.task_drafting.qualification import load_task_qualification_receipt
p=Path(sys.argv[1])
try:
    expected_ast = sys.argv[2] if len(sys.argv)>2 else ''
    if expected_ast:
        for name in ('evaluate.py','evaluate_final.py'):
            nodes={n.name:n for n in ast.parse((p/name).read_text()).body if isinstance(n,ast.FunctionDef)}
            tree=ast.Module(body=[nodes[n] for n in ('boundary_f1_counts','macro_boundary_f1')],type_ignores=[])
            actual=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
            if actual != expected_ast:
                raise ValueError('Verified boundary metric changed: '+name)
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


def require_initialization_retry(record: dict, state: dict, inventory: dict) -> None:
    """Only a stopped startup failure with no usable research work can restart."""
    entries=set(inventory['entries'])
    empty_startup=inventory.get('startup_failure') and not entries-{'pipeline_events.jsonl'}
    runtime_bookkeeping={'pipeline_events.jsonl','task','investigation','.task.runtime-build.lock',
                         '.agentic_resume_policy.json','.investigation_completion.json'}
    runtime_startup=(inventory.get('no_usable_checkpoint')
                     and not set(inventory.get('investigation_entries',[]))-{'events.jsonl'}
                     and all(entry in runtime_bookkeeping or re.fullmatch(r'\.task\.agentic_task_ready\.[0-9a-f]{64}',entry)
                             for entry in entries))
    if (record.get('phase') != 'investigation' or state['running']
            or state['exit_code'] in (None, 0)
            or not record.get('qualified_source_task', {}).get('path')
            or not (empty_startup or runtime_startup)):
        raise ValueError('Initialization retry would overlap or discard existing research work')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Submit if all verification gates pass')
    parser.add_argument('--retry-initialization', action='store_true',
                        help='Retry only a failed initialization with no research checkpoint')
    args = parser.parse_args()
    launcher = runpy.run_path(str(ROOT / 'scripts/remote-agsci'))
    recovery = runpy.run_path(str(ROOT / 'scripts/research/resume_medical_drafting.py'))
    remote = launcher['remote']; base = launcher['REMOTE']
    record_path = ROOT / 'docs/research/medical-unlearning/execution.json'
    (ROOT / 'scratch').mkdir(exist_ok=True)
    with (ROOT / 'scratch/medical-recovery-submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(record_path.read_text()); old = record['job_id']
        if record['phase'] != 'task_drafting' and not args.retry_initialization:
            raise ValueError('Current job is already a downstream phase; inspect it instead')
        probe = (ROOT / 'scripts/research/medical_job_state.py').read_text()
        def inspect(identity=old):
            result = remote('python3 - ' + shlex.join([base, identity]) + " <<'PY'\n" + probe + '\nPY\n', capture=True)
            return json.loads(result.stdout)
        def initialization_inventory():
            code = '''from pathlib import Path
import json,sys
base,job=Path(sys.argv[1]),sys.argv[2]
root=base/'results'/job;log=(base/'jobs'/job/'run.log').read_text()
print(json.dumps({'entries':[p.name for p in root.iterdir()],
'investigation_entries':[p.name for p in (root/'investigation').iterdir()] if (root/'investigation').is_dir() else [],
'no_usable_checkpoint':'No usable investigation checkpoint was written; fix the original error and restart with a new run_id and a new paths.run_dir.' in log,
'startup_failure':'Could not prepare run-local task or investigation workspace:' in log}))
'''
            result=remote('python3 - '+shlex.join([base,old])+" <<'PY'\n"+code+'\nPY\n',capture=True)
            return json.loads(result.stdout)
        state = inspect()
        source_job = old
        if args.retry_initialization:
            require_initialization_retry(record,state,initialization_inventory())
            source=record['qualified_source_task']['path']
            source_job=Path(source).parent.name
            if source != base+'/results/'+source_job+'/task':
                raise ValueError('Qualified source is outside its recorded result directory')
        else:
            source = state['published_task']
        source_state = inspect(source_job) if source_job != old else state
        directory = base + '/jobs/' + source_job
        verification = {'ready': False, 'reason': 'Qualified source is still active' if source_state['running']
                        else 'Qualified source stopped with exit '+str(source_state['exit_code'])}
        if not source_state['running'] and source_state['exit_code'] == 0:
            result = remote('cd ' + shlex.quote(directory) + '\n' + shlex.quote(base + '/.venv/bin/python')
                            + ' - ' + shlex.join([source,record.get('boundary_f1_repair',{}).get('expected_metric_ast_sha256','')])
                            + " <<'PY'\n" + VERIFY + '\nPY\n', capture=True)
            verification = json.loads(result.stdout)
        if record.get('blocking_findings'):
            verification = {'ready':False,'reason':'Unresolved scientific findings: '+str(record['blocking_findings'])}
        if not args.run:
            print(json.dumps({'job_id':old,'state':state,'verification':verification},indent=2))
            return
        require_ready(source_state, verification)
        job = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        directory, results = base + '/jobs/' + job, base + '/results/' + job
        launcher['sync'](directory); launcher['stage_auth'](directory)
        remote('mkdir -p ' + shlex.quote(results) + '\n')
        # Pin the exact image that passed qualification; no floating-tag switch.
        cmd = launcher['args_for']('run', overrides(source,verification['image_id']),directory)
        cmd += ['run_id='+job,'paths.run_dir='+results]
        require_ready(inspect(source_job), verification)
        if args.retry_initialization:
            require_initialization_retry(record,inspect(),initialization_inventory())
        runner = recovery['runner_text'](launcher,directory,cmd)
        remote('printf %s ' + shlex.quote(runner) + ' > ' + shlex.quote(directory+'/run.sh') + '\n')
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        record.setdefault('execution_history',[]).append({
            key:record.get(key) for key in ('job_id','status','phase','remote_log','remote_results','last_monitor_check')})
        record.update(updated_at=now,job_id=job,status='submitting_investigation',phase='investigation',
                      substage='initializing_investigation',
                      remote_log=directory+'/run.log',remote_results=results)
        record['qualified_source_task']={'path':source,'source_job':source_job,'verified_at':now,**verification}
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
