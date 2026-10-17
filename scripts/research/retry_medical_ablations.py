#!/usr/bin/env python3
"""Repair the stopped ablation handoff without repeating investigation or final."""
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

WITNESS = r'''
import hashlib,json,sys
from pathlib import Path
from src.investigation.task_runtime import _tooling_digest
from src.pipeline.resume_config import read_run_idea_cycle_budget
r=json.loads(sys.argv[1]);expected=json.loads(sys.argv[2]);p=Path(r['remote_results'])
errors=[]
for name,item in r['boundary_f1_repair']['files'].items():
    if hashlib.sha256((p/'task'/name).read_bytes()).hexdigest()!=item['after']:
        errors.append('Private evaluator changed: '+name)
for relative,digest in expected.items():
    if hashlib.sha256((p/relative).read_bytes()).hexdigest()!=digest:
        errors.append('Sealed final changed: '+relative)
events=[json.loads(line) for line in (p/'ablations/events.jsonl').read_text().splitlines()]
last=next(e for e in reversed(events) if e['event']=='pipeline_end')
outer=json.loads((p/'investigation/outer_state.json').read_text())
attempts=[json.loads(line) for line in (p/'investigation/attempts_history.jsonl').read_text().splitlines()]
best=json.loads((p/'investigation/best_program_manifest.json').read_text())
print(json.dumps({'errors':errors,'tooling_sha256':_tooling_digest(),
    'cycle_budget':read_run_idea_cycle_budget(p),'current_cycle':outer['current_cycle'],
    'completed_attempts':len(attempts),'ablation_exit_code':last['fields']['exit_code'],
    'best_correct':best['correct'],'best_score':best['combined_score'],
    'best_manifest_sha256':hashlib.sha256((p/'investigation/best_program_manifest.json').read_bytes()).hexdigest()}))
'''

BUILD_COMMAND = r'''
import json,sys
from src.pipeline.cli import load_pipeline_config
from src.pipeline.paths import derive_paths
from src.pipeline.phases import build_ablations_cmd
cfg=load_pipeline_config(json.loads(sys.argv[1]));command=build_ablations_cmd(cfg,derive_paths(cfg))
assert command[1:3]==['-m','src.variants_pipeline.cli']
print(json.dumps(command))
'''

ARCHIVE_AND_PATCH = r'''
import hashlib,json,shutil,sys
from pathlib import Path
from src.investigation.task_runtime import _tooling_digest
directory,archive=Path(sys.argv[1]),Path(sys.argv[2]);payload=json.loads(sys.argv[3])
before=_tooling_digest();assert before==payload['tooling_sha256']
archive.mkdir(parents=True,exist_ok=False)
for name in ('exit-code','pid','run.log','run.sh'):
    shutil.copy2(directory/name,archive/name)
wrapper=directory/'src/scripts/variant_agent.sh'
shutil.copy2(wrapper,archive/'variant_agent.before.sh')
original=hashlib.sha256(wrapper.read_bytes()).hexdigest()
wrapper.write_text(payload['wrapper'])
assert _tooling_digest()==before
print(json.dumps({'archive':str(archive),'wrapper_before_sha256':original,
    'wrapper_after_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    'tooling_sha256_unchanged':before}))
'''


def require_retry(record: dict, state: dict, witness: dict, validation: dict) -> None:
    auth = record.get('budget_authorization', {})
    if (not auth.get('automatic_recovery_authorized')
            or auth.get('overall_time_limit_seconds', 'missing') is not None):
        raise ValueError('Automatic recovery was not authorized')
    if state['running'] or state['exit_code'] != 0:
        raise ValueError('The original completed pipeline must be fully stopped')
    if not validation.get('ready') or witness['errors']:
        raise ValueError('Source gates or sealed final integrity checks failed')
    if record.get('ablation_recoveries'):
        raise ValueError('An ablation retry exists; preserve and inspect its stage checkpoint')
    if not record.get('sealed_final_result', {}).get('result_directory_immutable'):
        raise ValueError('The sealed final result must be retained')
    if (witness['cycle_budget'] != 2 or witness['current_cycle'] != 2
            or witness['completed_attempts'] != 2 or not witness['best_correct']
            or witness['ablation_exit_code'] != 4):
        raise ValueError('This is not the recorded pre-evaluation ablation failure')
    if witness['tooling_sha256'] != record['runtime_snapshot_repair']['new_tooling_sha256']:
        raise ValueError('Frozen scientific harness changed')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    launcher = runpy.run_path(str(ROOT/'scripts/remote-agsci'))
    recovery = runpy.run_path(str(ROOT/'scripts/research/resume_medical_drafting.py'))
    continuation = runpy.run_path(str(ROOT/'scripts/research/continue_medical_experiment.py'))
    remote, base = launcher['remote'], launcher['REMOTE']
    record_path = ROOT/'docs/research/medical-unlearning/execution.json'
    with (ROOT/'scratch/medical-recovery-submit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(record_path.read_text());job = record['job_id']
        directory, results = base+'/jobs/'+job, record['remote_results']
        if results != base+'/results/'+job:
            raise ValueError('The original research results directory must be preserved')
        proof = json.loads((ROOT/'docs/research/medical-unlearning/validation'/job/'final_evaluation/verification.json').read_text())
        final_hashes = {k:v for k,v in proof['artifact_sha256'].items() if k.startswith('final_evaluation/')}
        if len(final_hashes) != 5:
            raise ValueError('All original final outputs must have recorded hashes')
        probe = (ROOT/'scripts/research/medical_job_state.py').read_text()
        def run_python(script: str, arguments: list[str], checkout: str = directory) -> dict | list:
            command = ('cd '+shlex.quote(checkout)+'\n'
                       +shlex.join([base+'/.venv/bin/python','-',*arguments])
                       +" <<'PY'\n"+script+'\nPY\n')
            return json.loads(remote(command,capture=True).stdout)
        def inspect() -> dict:
            # Keep the inspection shell outside the frozen checkout. Its cwd
            # must not look like a still-running research process to the guard.
            return run_python(probe,[base,job],checkout=base+'/workspace')
        source = record['qualified_source_task']['path']
        validation = run_python(continuation['VERIFY'],
            [source,record['boundary_f1_repair']['expected_metric_ast_sha256']])
        witness = run_python(WITNESS,[json.dumps(record),json.dumps(final_hashes)])
        state = inspect()
        try:
            require_retry(record,state,witness,validation)
        except ValueError as error:
            if args.resume:raise
            print(json.dumps({'ready':False,'reason':str(error),'state':state,'witness':witness},indent=2));return
        if not args.resume:
            print(json.dumps({'ready':True,'state':state,'witness':witness,'validation':validation},indent=2));return
        attempt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        archive = directory+'/attempts/ablations_'+attempt
        output = results+'/ablations_recovery_'+attempt
        extra = continuation['overrides'](source,validation['image_id'])+[
            'run_id='+job,'paths.run_dir='+results,'paths.ablations_dir='+output]
        command = run_python(BUILD_COMMAND,[json.dumps(launcher['args_for']('run',extra,directory)[4:])])
        selector = '"device='+','.join(launcher['CFG']['gpu_uuids'])+'"'
        command = ['env','VARIANT_EVAL_SANDBOX_GPUS='+selector,*command]
        runner = recovery['runner_text'](launcher,directory,command)
        require_retry(record,inspect(),run_python(WITNESS,[json.dumps(record),json.dumps(final_hashes)]),validation)
        patch = run_python(ARCHIVE_AND_PATCH,[directory,archive,json.dumps({
            'tooling_sha256':witness['tooling_sha256'],
            'wrapper':(ROOT/'src/scripts/variant_agent.sh').read_text()})])
        remote('printf %s '+shlex.quote(runner)+' > '+shlex.quote(archive+'/resume.sh')+'\n')
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        detail={'attempt_id':attempt,'job_id':job,'archive':archive,'output_dir':output,
                'started_at':now,'command':command,'wrapper_patch':patch,'final_hashes':final_hashes,
                'scientific_protocol_changed':False,'sealed_final_repeated':False}
        record.setdefault('ablation_recoveries',[]).append(detail)
        record.update(updated_at=now,status='submitting_ablations_recovery_final_not_qualified',
                      phase='ablations',substage='initializing_ablations_recovery')
        record['phase_output_overrides']={'ablations':output}
        recovery['atomic_record'](record_path,record)
        remote('cd '+shlex.quote(directory)+'\ntest "$(cat exit-code)" = 0\nmv exit-code '
               +shlex.quote(archive+'/stopped-exit-code')+'\nnohup bash '+shlex.quote(archive+'/resume.sh')
               +' > run.log 2>&1 < /dev/null &\necho $! > pid\n')
        record['status']='running_ablations_recovery_final_not_qualified'
        record['substage']='regenerating_ablation_candidates'
        record['next_action']={'requires_user_decision':False,'action':
            'Monitor only the repaired ablations run. Keep original investigation, sealed final outputs and negative final claim result unchanged. Do not tune candidates or repeat final; report a separate scientific decision after engineering recovery.'}
        recovery['atomic_record'](record_path,record)
        print(json.dumps(detail,indent=2))


if __name__=='__main__':
    main()
