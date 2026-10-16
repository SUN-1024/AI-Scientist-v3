#!/usr/bin/env python3
"""Submit one explicitly authorized, 7200-second recovery of medical task drafting."""
import datetime
import json
from pathlib import Path
import runpy
import shlex
import uuid

ROOT = Path(__file__).resolve().parents[2]
launcher = runpy.run_path(str(ROOT / 'scripts/remote-agsci'))
remote, sync, stage_auth = (launcher[k] for k in ('remote', 'sync', 'stage_auth'))
base = launcher['REMOTE']
record_path = ROOT / 'docs/research/medical-unlearning/execution.json'
record = json.loads(record_path.read_text())
if record['status'] != 'failed_drafting_timeout_awaiting_budget_decision':
    raise SystemExit('Recovery submission requires the recorded stopped drafting task')
old = record['job_id']
source = base + '/results/' + old + '/task_drafting/_task_drafting/draft_20261016_141410_1792160050430039797'
job = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
directory = base + '/jobs/' + job
results = base + '/results/' + job
# Existing process check before any submission; no other services are stopped.
remote(f'test -f {base}/jobs/{old}/exit-code\n'
       'if pgrep -f "[s]rc.agsci.main run|[s]rc/scripts/task.sh" >/dev/null; then echo "Existing task process; refusing duplicate" >&2; exit 1; fi\n')
sync(directory)
stage_auth(directory)
remote(f'mkdir -p {results}\ncp -a {source}/task {results}/task\n'
       f'cp {directory}/src/prompts/draft_task.md {directory}/recovery_prompt.md\n')
addition = '''\n\n# Authorized interruption recovery\nThe task directory contains interrupted, unvalidated implementation from a previous run. The accepted design was independently reviewed and is restored at TASK_DESIGN_PATH. Continue existing candidate and evaluator code. Do not redesign the study or alter the canonical task, partitioning, budgets, scientific requirements, or gates. First inspect existing files and finish missing evaluate_final.py and task_contract.json; validate existing code and fix actual defects. Preserve the self-contained sealed final evaluator requirement. Avoid spending the drafting budget on repeated dependency installation; the wrapper owns sanitized runtime bootstrap. If draft-time dependencies are essential, use a temporary venv and the Tsinghua PyPI mirror. Finish implementation and return promptly so the wrapper can run all validation gates.\n'''
remote(f'printf %s {shlex.quote(addition)} >> {directory}/recovery_prompt.md\n')
cmd = ['bash', 'src/scripts/task.sh', '--agent', 'opencode', '--agent-model', 'qwen-plan/qwen3.8-max',
       '--task-dir', results + '/task', '--output-dir', results + '/task_drafting',
       '--accepted-design-dir', source + '/_task_design', '--instruct', directory + '/recovery_prompt.md',
       '--workflow', 'robust', '--design-max-repairs', '2', '--contract-max-repairs', '2',
       '--smoke-max-repairs', '2', '--alignment-max-repairs', '2', '--qualification-max-repairs', '2',
       '--draft-max-attempts', '1', '--max-idea-cycles', '2', '--max-program-evals-per-attempt', '1',
       '--qualification-timeout-seconds', '7200', '--containerize', '--container-network', 'host',
       '--container-auth-root', directory + '/auth', '--container-env-file', base + '/.env']
selector = '"device=' + ','.join(launcher['CFG']['gpu_uuids']) + '"'
runner = ('#!/bin/bash\nset -euo pipefail\n' + launcher['preamble'](directory)
          + 'export AGENT_CONTAINER_GPUS=' + shlex.quote(selector) + '\n'
          + 'set +e\ntimeout --signal=TERM --kill-after=30s 7200s ' + shlex.join(cmd)
          + '\nrc=$?\nprintf "%s\\n" "$rc" > exit-code\nexit "$rc"\n')
remote(f'printf %s {shlex.quote(runner)} > {directory}/run.sh\n'
       f'cd {directory}\nnohup bash run.sh > run.log 2>&1 < /dev/null &\necho $! > pid\n')
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
record['previous_execution'] = {k: record[k] for k in ('job_id', 'status', 'remote_log', 'remote_results', 'last_monitor_check')}
record.update(updated_at=now, job_id=job, status='running_drafting_recovery', phase='task_drafting',
              remote_log=directory + '/run.log', remote_results=results)
record['recovery'] = {'authorized_at': now, 'additional_seconds': 7200, 'source_draft': source,
                      'scope': 'Complete interrupted scaffold and run unchanged robust validation gates',
                      'repair': 'Validated accepted-design recovery; retained code copied to separate task; external 7200s timeout',
                      'validation': '4 recovery tests passed; shell syntax passed', 'training_started': False}
record['last_monitor_check'] = {'checked_at': now, 'status': 'submitted_recovery', 'restart_performed': True,
                               'repair_applied': True, 'scientific_results_available': False}
record['next_action'] = {'requires_user_decision': False, 'action': 'Monitor recovery gates within authorized 7200 seconds; preserve failures; no automatic budget extension'}
record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
print(job)
