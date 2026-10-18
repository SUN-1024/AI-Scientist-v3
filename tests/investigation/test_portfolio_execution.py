"""Portfolio execution uses concurrent isolated workers and validated host receipts.

The subprocess fixture simulates the host evaluator. It never invokes a model,
GPU, Docker, or final evaluation data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from src.investigation import portfolio_campaign as campaign
from src.investigation.config import InvestigationConfig
from src.investigation.container_gpus import GPU_POOL_ENV, GPU_SELECTOR_ENV
from src.investigation.models import ParallelWorkerResult
from src.investigation.parallel_helpers import build_parallel_worker_command, run_parallel_worker
from src.investigation.portfolio_controller import PortfolioConfig, PortfolioController, read_portfolio_state
from src.orchestrator import state as state_store
from tests.program_testutils import make_program_dir


# This script runs in a real subprocess. The first two workers rendezvous so
# serial implementations cannot pass by producing the same output eventually.
WORKER_SOURCE = r'''
import json, os, shutil, sys, time
from pathlib import Path
from src.investigation.harness_incumbent import _program_source_records
from src.investigation.native_harness import _hash
from src.investigation.program_clone import clone_program_dir
from tests.program_testutils import make_program_dir
args = dict(arg.split('=', 1) for arg in sys.argv[1:] if '=' in arg)
file = Path(args['execution.portfolio_lease_file'])
envelope = json.loads(file.read_text())
lease = envelope['lease']
root = Path(envelope['root_results_dir'])
worker = Path(lease['result_dir'])
task = Path(args['paths.task_dir'])
events = root / 'fixture_events'
events.mkdir(exist_ok=True)
start = time.monotonic()
(events / (lease['lease_id'] + '.start')).write_text(json.dumps({'time':start,'gpu':os.environ.get('AGENT_CONTAINER_GPUS'),'worker':lease['worker_id']}))
if lease['stage'] == 'screening' and lease['root_idea_id'] in {'idea_0001','idea_0002'}:
    deadline = time.monotonic() + 10
    while not all((events / ('idea_%04d_screening_000.start' % i)).exists() for i in (1,2)):
        if time.monotonic() > deadline:
            raise RuntimeError('workers did not overlap')
        time.sleep(.01)
    time.sleep(.6 if lease['root_idea_id'] == os.environ.get('FIXTURE_SLOW_ID','idea_0002') else .01)
idea = worker / 'ideas' / lease['root_idea_id']
program = idea / '01_attempt' / 'program_1'
if lease['stage'] == 'replication':
    clone_program_dir(src_dir=Path(lease['parent_program_dir']), dst_dir=program)
else:
    make_program_dir(program, content="print(%r)\n" % lease['lease_id'])
index = int(lease['root_idea_id'].split('_')[-1]) if lease['root_idea_id'] != 'baseline' else 0
score = .5 + .05 * index + (.15 if lease['stage'] == 'tuning' else 0)
correct = index != 3
if lease['stage'] == 'replication':
    score, correct = lease['parent']['score'], True
bundle = {'idea_id':lease['root_idea_id'],'title':'Whole idea '+str(index),'hypothesis':'An entire idea improves development performance.','target_rubrics':[],'whole_idea':{'schema_version':1,'innovation_axis':lease['innovation_axis'],'problem_statement':'One bounded research question.','contribution':'A coherent whole-idea contribution.','minimal_validation':{'protocol_id':'fixed-v1','decision_rule':'Retain stronger correct candidate.','stop_or_pivot':'Stop at budget.','cost_metric':'runtime','quality_guardrail':'correct is true'},'parameter_space':[],'selection_reason':'Smallest permitted screen.'}}
(idea / 'idea_bundle.json').write_text(json.dumps(bundle))
(program / 'metrics.json').write_text(json.dumps({'combined_score':score}))
(program / 'correct.json').write_text(json.dumps({'correct':correct}))
manifest = {'producer':'sandbox_host_eval','execution_boundary':'sandbox','program_id':'program_1','exit_code':0,'returncode':0,'attempt_id':lease['root_idea_id']+'_attempt_01','idea_id':lease['root_idea_id'],'lease_id':lease['lease_id'],'adaptive_source_sha256':_program_source_records(program),'adaptive_evaluator_sha256':_hash(task/'evaluate.py'),'adaptive_result_sha256':{name:_hash(program/name) for name in ('metrics.json','correct.json')}}
(program / 'evaluation_manifest.json').write_text(json.dumps(manifest))
(events / (lease['lease_id'] + '.end')).write_text(str(time.monotonic()))
'''


class CampaignFixture:
    def __init__(self, root: Path, monkeypatch):
        self.root = root
        self.task = root / 'task'
        self.seed = self.task / 'program_0'
        make_program_dir(self.seed, content="print('baseline')\n")
        (self.task / 'task.md').write_text('Compare whole ideas under a fixed development protocol.\n')
        (self.task / 'evaluate.py').write_text('# deterministic fixture evaluator\n')
        self.results = root / 'run'
        self.results.mkdir()
        self.worker_script = root / 'fixture_worker.py'
        self.worker_script.write_text(WORKER_SOURCE)
        self.cfg = InvestigationConfig()
        self.cfg.agent.name = 'codex'
        self.cfg.agent.model = 'fixture-model'
        self.cfg.execution.workers = 2
        self.cfg.execution.max_program_evals_per_attempt = 1
        self.cfg.execution.harness_evaluated_attempts = True
        self.cfg.execution.max_idea_cycles = 20
        self.cfg.container.gpu_count = 1
        # Fixture subprocesses are native; production configuration separately
        # requires the existing sealed container boundary.
        self.cfg.container.enabled = False
        self.cfg.portfolio = PortfolioConfig(enabled=True, initial_ideas=4, promote_ideas=2,
            tuning_trials_per_idea=1, replication_evals=1, max_valid_evals=7)
        self.best = {'program_dir':str(self.seed), 'combined_score':.4, 'correct':True}
        self.outer = {}
        self.events = []
        self.processes = []
        self.cleanups = 0
        self.lock = threading.Lock()
        monkeypatch.setenv(GPU_POOL_ENV, 'device=0,1')
        monkeypatch.setenv('PYTHONPATH', str(Path(__file__).resolve().parents[2]))

    def build(self, **kwargs):
        command = build_parallel_worker_command(**kwargs)
        return [sys.executable, str(self.worker_script), *command[3:]]

    def register(self, proc):
        with self.lock:
            self.processes.append(proc)

    def unregister(self, proc):
        with self.lock:
            self.processes.remove(proc)

    def terminate(self, **kwargs):
        self.cleanups += 1
        with self.lock:
            processes = list(self.processes)
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        return [proc.pid for proc in processes]

    def worker(self, plan):
        return run_parallel_worker(plan, popen_factory=subprocess.Popen,
            register_subprocess=self.register, unregister_subprocess=self.unregister,
            terminate_process_group=lambda proc, **kwargs: proc.terminate(), ensure_text=str)

    def run(self, worker=None):
        return campaign.run_portfolio_campaign(cfg=self.cfg, task_dir=self.task, results_dir=self.results,
            global_memory_dir=self.results, instruct_path=self.task/'task.md', outer_state=self.outer,
            best_manifest=self.best, best_program_dir=Path(self.best['program_dir']),
            build_parallel_worker_command_fn=self.build, run_parallel_worker_fn=worker or self.worker,
            log_event_fn=lambda name, **kwargs: self.events.append((name, kwargs)),
            terminate_all_active_processes_fn=self.terminate,
            stop_container_session_fn=lambda *args, **kwargs: True,
            build_container_session_name_fn=lambda **kwargs: 'fixture')


def test_real_subprocess_workers_overlap_reuse_slot_and_keep_disjoint_gpu_assignments(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    assert fixture.run() == 0
    events = fixture.results/'fixture_events'
    read_start = lambda name: json.loads((events/(name+'.start')).read_text())
    first = read_start('idea_0001_screening_000')
    second = read_start('idea_0002_screening_000')
    third = read_start('idea_0003_screening_000')
    assert first['gpu'] != second['gpu']
    assert third['gpu'] == first['gpu']
    assert third['time'] < float((events/'idea_0002_screening_000.end').read_text())
    state = read_portfolio_state(fixture.results, fixture.task)
    assert state['valid_evaluations'] == 7
    assert state['outcomes']['idea_0003_screening_000']['correct'] is False
    assert 'idea_0003' not in state['promoted_ideas']
    assert fixture.best['root_idea_id'] == 'idea_0004'
    assert fixture.outer['best_program_dir'] == state['frozen_candidate']['program_dir']
    assert len(state_store.tail_jsonl(state_store.attempts_history_path(fixture.results), max_lines=100)) == 7
    assert not fixture.processes


@pytest.mark.parametrize('slow_id', ['idea_0001', 'idea_0002'])
def test_completion_order_does_not_change_promotion_or_final_winner(tmp_path, monkeypatch, slow_id):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    monkeypatch.setenv('FIXTURE_SLOW_ID', slow_id)
    assert fixture.run() == 0
    state = read_portfolio_state(fixture.results, fixture.task)
    assert state['promoted_ideas'] == ['idea_0004', 'idea_0002']
    assert fixture.best['root_idea_id'] == 'idea_0004'
    assert state['frozen_candidate']['source_hash'] == fixture.best['source_hash']


def test_receipt_crash_recovery_does_not_launch_or_evaluate_completed_lease_again(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    publish = campaign._publish_outcome
    crashed = False
    def crash_after_root_history(**kwargs):
        nonlocal crashed
        publish(**kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError('coordinator crash before controller commit')
    monkeypatch.setattr(campaign, '_publish_outcome', crash_after_root_history)
    with pytest.raises(RuntimeError, match='before controller commit'):
        fixture.run()
    # The host receipt is durable; recovery must not pay for another child.
    completed = {path.name.removesuffix('.end') for path in (fixture.results/'fixture_events').glob('*.end')}
    monkeypatch.setattr(campaign, '_publish_outcome', publish)
    started = []
    def resumed_worker(plan):
        lease = json.loads((plan.results_dir/campaign.LEASE_FILE).read_text())['lease']
        started.append(lease['lease_id'])
        return fixture.worker(plan)
    assert fixture.run(resumed_worker) == 0
    assert not (completed & set(started))
    rows = state_store.tail_jsonl(state_store.attempts_history_path(fixture.results), max_lines=100)
    assert len(rows) == len({row['portfolio_lease_id'] for row in rows}) == 7


def test_interrupt_cleans_children_before_thread_pool_waits(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    released = threading.Event()
    entered = threading.Event()
    def hanging_worker(plan):
        entered.set()
        assert released.wait(5), 'shutdown waited before process cleanup'
        return ParallelWorkerResult(plan, 130, plan.results_dir/'log', 0, '')
    def interrupted_wait(*args, **kwargs):
        assert entered.wait(3)
        raise KeyboardInterrupt
    def terminate(**kwargs):
        fixture.cleanups += 1
        released.set()
        return []
    monkeypatch.setattr(campaign, 'wait', interrupted_wait)
    fixture.terminate = terminate
    with pytest.raises(KeyboardInterrupt):
        fixture.run(hanging_worker)
    assert fixture.cleanups == 1
    state = read_portfolio_state(fixture.results, fixture.task)
    assert state['pending'] and not state['outcomes']


def test_worker_command_forwards_exact_parent_one_trial_and_container_gpu_count(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    fixture.cfg.container.enabled = True
    fixture.cfg.container.image = 'sha256:fixture'
    fixture.cfg.container.gpu_count = 2
    fixture.cfg.execution.harness_evaluation_timeout_seconds = 321
    controller = PortfolioController(fixture.results, fixture.task, fixture.cfg.portfolio)
    controller.qualify_baseline(fixture.best, fixture.seed)
    monkeypatch.setenv(GPU_POOL_ENV, 'device=0,1,2,3')
    lease = controller.reserve_next(2, gpu_selector='device=2,3')
    plan = campaign.build_portfolio_worker_plan(cfg=fixture.cfg, task_dir=fixture.task,
        results_dir=fixture.results, instruct_path=fixture.task/'task.md', lease=lease,
        context=controller.context(lease), build_parallel_worker_command_fn=build_parallel_worker_command)
    flags = dict(item.split('=',1) for item in plan.command[3:])
    assert flags['container.gpu_count'] == '2'
    assert flags['execution.harness_evaluation_timeout_seconds'] == '321'
    assert flags['paths.seed_program'] == str(fixture.seed)
    assert flags['execution.workers'] == flags['execution.max_idea_cycles'] == flags['execution.max_program_evals_per_attempt'] == '1'
    assert flags['portfolio.enabled'] == flags['adaptive_investigation.enabled'] == flags['paper.enabled'] == 'false'
    assert plan.environment[GPU_SELECTOR_ENV] == 'device=2,3'
    assert json.loads((plan.results_dir/campaign.LEASE_FILE).read_text())['lease'] == lease


def test_live_worker_pid_blocks_resume_and_second_coordinator_lock_is_exclusive(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    worker_dir = fixture.results/'pending'
    worker_dir.mkdir()
    state_store.write_json(worker_dir/campaign.PROCESS_FILE, {'pid':os.getpid(),'status':'running'})
    with pytest.raises(ValueError, match='live'):
        campaign._confirm_old_work_stopped(cfg=fixture.cfg, worker_dir=worker_dir,
            stop_container_session_fn=lambda *args, **kwargs: True,
            build_container_session_name_fn=lambda **kwargs: 'fixture', log_event_fn=lambda *args, **kwargs: None)
    with campaign._exclusive_lock(fixture.results/campaign.COORDINATOR_LOCK):
        with pytest.raises(ValueError, match='already active'):
            fixture.run()


def test_process_marker_write_failure_stops_spawned_child_and_unregisters_handle(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    from src.investigation.models import ParallelWorkerPlan
    plan = ParallelWorkerPlan(worker_index=1,assigned_cycles=1,cycle_start=1,cycle_end=1,
        results_dir=fixture.results,command=[sys.executable,'-c','import time;time.sleep(20)',
            'execution.portfolio_lease_file=fixture'])
    writer = state_store.write_private_text
    failed = False
    def fail_running_marker(path, text):
        nonlocal failed
        if path.name == campaign.PROCESS_FILE and json.loads(text).get('status') == 'running':
            failed = True
            raise OSError('fixture lifecycle persistence failure')
        return writer(path,text)
    monkeypatch.setattr(state_store,'write_private_text',fail_running_marker)
    result = fixture.worker(plan)
    assert failed and result.exit_code == 127
    assert not fixture.processes
    marker = json.loads((fixture.results/campaign.PROCESS_FILE).read_text())
    assert marker['status'] == 'exited'


def test_invalid_completed_receipt_stops_instead_of_repeating_scientific_evaluation(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path,monkeypatch)
    launches = []
    def tampered_worker(plan):
        lease = json.loads((plan.results_dir/campaign.LEASE_FILE).read_text())['lease']
        launches.append(lease['lease_id'])
        result = fixture.worker(plan)
        if lease['root_idea_id'] == 'idea_0001':
            program = next(plan.results_dir.glob('ideas/*/*_attempt/program_1'))
            (program/'metrics.json').write_text('{"combined_score":99}')
        return result
    assert fixture.run(tampered_worker) == 2
    assert len(launches) == len(set(launches))
    state = read_portfolio_state(fixture.results,fixture.task)
    assert state['status'] == 'incomplete'
    assert 'idea_0001_screening_000' not in state['outcomes']
    assert not (fixture.results/'validation_handoff.json').exists()


def test_truncated_process_marker_recovers_valid_receipt_without_worker_rerun(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    publish = campaign._publish_outcome
    crashed = False
    def crash_after_receipt(**kwargs):
        nonlocal crashed
        publish(**kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError('fixture interrupted final marker publication')
    monkeypatch.setattr(campaign, '_publish_outcome', crash_after_receipt)
    with pytest.raises(RuntimeError, match='marker publication'):
        fixture.run()
    completed = {path.name.removesuffix('.end') for path in (fixture.results/'fixture_events').glob('*.end')}
    for lease_id in completed:
        (fixture.results/'portfolio_trials'/lease_id/campaign.PROCESS_FILE).write_text('{"pid":')
    monkeypatch.setattr(campaign, '_publish_outcome', publish)
    launches = []
    def resumed(plan):
        launches.append(plan.results_dir.name)
        return fixture.worker(plan)
    assert fixture.run(resumed) == 0
    assert not completed.intersection(launches)
    assert any(name == 'portfolio_process_marker_recovered' for name, _ in fixture.events)
    assert read_portfolio_state(fixture.results, fixture.task)['valid_evaluations'] == 7


def test_truncated_lease_envelope_is_restored_only_from_durable_reservation(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    controller = PortfolioController(fixture.results, fixture.task, fixture.cfg.portfolio)
    controller.qualify_baseline(fixture.best, fixture.seed)
    lease = controller.reserve_next(1, gpu_selector=campaign._slot_gpu_selector(fixture.cfg, 1))
    child = Path(lease['result_dir'])
    child.mkdir(parents=True)
    (child/campaign.LEASE_FILE).write_text('{"root_results_dir":')
    assert fixture.run() == 0
    assert json.loads((child/campaign.LEASE_FILE).read_text())['lease'] == lease
    assert read_portfolio_state(fixture.results, fixture.task)['valid_evaluations'] == 7


def test_corrupt_marker_still_blocks_live_lease_process_and_held_worker_lock(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    child = fixture.results/'pending'
    child.mkdir()
    (child/campaign.PROCESS_FILE).write_text('{"pid":')
    arguments = dict(cfg=fixture.cfg, worker_dir=child, stop_container_session_fn=lambda *a, **k:True,
        build_container_session_name_fn=lambda **k:'fixture', log_event_fn=lambda *a, **k:None)
    with campaign._exclusive_lock(child/campaign.WORKER_LOCK):
        with pytest.raises(ValueError, match='already active'):
            campaign._confirm_old_work_stopped(**arguments)
    proc = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(20)',
        'execution.portfolio_lease_file='+str(child/campaign.LEASE_FILE)])
    try:
        with pytest.raises(ValueError, match='process for this portfolio lease is live'):
            campaign._confirm_old_work_stopped(**arguments)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_missing_process_inventory_never_authorizes_corrupt_marker_recovery(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    child = fixture.results/'pending'
    child.mkdir()
    (child/campaign.PROCESS_FILE).write_text('{"pid":')
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **k:subprocess.CompletedProcess(a,1,'','unavailable'))
    with pytest.raises(ValueError, match='inventory is unavailable'):
        campaign._confirm_old_work_stopped(cfg=fixture.cfg,worker_dir=child,
            stop_container_session_fn=lambda *a,**k:True,
            build_container_session_name_fn=lambda **k:'fixture',log_event_fn=lambda *a,**k:None)


def test_process_start_identity_distinguishes_pid_reuse_but_unknown_identity_blocks(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    child = fixture.results/'pending'
    child.mkdir()
    original = {'platform':'posix-ps','pid':os.getpid(),'uid':os.getuid(),
        'start_time':'Sat Oct 17 10:00:00 2026','command_sha256':'a'*64}
    current = {**original,'start_time':'Sun Oct 18 10:00:00 2026','command_sha256':'b'*64}
    state_store.write_private_text(child/campaign.PROCESS_FILE,
        json.dumps({'pid':os.getpid(),'status':'running','process_identity':original}))
    kwargs = dict(cfg=fixture.cfg,worker_dir=child,stop_container_session_fn=lambda *a,**k:True,
        build_container_session_name_fn=lambda **k:'fixture',log_event_fn=lambda *a,**k:None)
    monkeypatch.setattr(campaign,'portfolio_process_identity',lambda pid:current)
    campaign._confirm_old_work_stopped(**kwargs)
    monkeypatch.setattr(campaign,'portfolio_process_identity',lambda pid:None)
    with pytest.raises(ValueError,match='Cannot establish identity'):
        campaign._confirm_old_work_stopped(**kwargs)


def test_baseline_retains_bootstrap_identity_after_portfolio_handoff(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    fixture.best.update(combined_score=.99, source_attempt_id='bootstrap')
    assert fixture.run() == 0
    assert fixture.best['root_idea_id'] == fixture.best['portfolio_lease_id'] == 'baseline'
    assert fixture.best['source_attempt_id'] == 'bootstrap'


def test_root_lock_reenters_only_in_owning_thread(tmp_path, monkeypatch):
    fixture = CampaignFixture(tmp_path, monkeypatch)
    errors = []
    def another_thread():
        try:
            with campaign.portfolio_coordinator_lock(fixture.results):
                errors.append('incorrectly acquired')
        except ValueError as exc:
            errors.append(str(exc))
    with campaign.portfolio_coordinator_lock(fixture.results):
        thread = threading.Thread(target=another_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert errors and 'already active' in errors[0]
        assert fixture.run() == 0


def test_duplicate_cli_root_lock_prevents_task_setup_and_baseline_state_mutation(tmp_path):
    from unittest.mock import patch
    from src.investigation import cli
    from tests.investigation.test_investigation_cli import CLI, _main_setup
    task, results, cfg = _main_setup(tmp_path, max_idea_cycles=20)
    cfg.portfolio.enabled = True
    cfg.container.enabled = True
    cfg.execution.workers = 2
    cfg.execution.harness_evaluated_attempts = True
    cfg.execution.max_program_evals_per_attempt = 1
    cfg.execution.resume = False
    results.mkdir(parents=True, exist_ok=True)
    manifest = {'program_dir':str(task/'program_0'),'combined_score':99.,'correct':True,'source_attempt_id':'winning-attempt'}
    outer = {'current_cycle':7,'best_program_dir':str(task/'program_0'),'best_combined_score':99.}
    state_store.save_best_manifest(results,manifest)
    state_store.save_outer_state(results,outer)
    task_before = (task/'task.md').read_bytes()
    with campaign._exclusive_lock(results/campaign.COORDINATOR_LOCK), \
         patch(f'{CLI}.load_config',return_value=cfg), \
         patch(f'{CLI}.run_task_setup',side_effect=AssertionError('duplicate task setup')), \
         patch(f'{CLI}.prepare_task_runtime',side_effect=AssertionError('duplicate runtime bootstrap')), \
         patch('src.investigation.main_entry.reconcile_harness_incumbent',side_effect=AssertionError('duplicate baseline write')):
        assert cli.main() == 2
    assert state_store.load_json(results/state_store.BEST_MANIFEST_FILE) == manifest
    assert state_store.load_json(results/state_store.OUTER_STATE_FILE) == outer
    assert (task/'task.md').read_bytes() == task_before
