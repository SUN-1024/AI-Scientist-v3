"""The existing idea/coding loop obeys externally leased whole-idea work."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from src.investigation import campaign_loop
from src.investigation import portfolio_campaign as portfolio
from src.investigation.config import InvestigationConfig
from src.investigation.files import find_best_program_in_experiment
from src.investigation.harness_eval import run_harness_evaluation, SealedHarnessConfig
from src.investigation.models import IdeaGenerationResult, RunAttemptResult
from src.investigation.parallel_helpers import build_parallel_worker_command
from src.investigation.portfolio_controller import PortfolioController, PortfolioConfig
from src.orchestrator import state as state_store
from src.variants_pipeline.host_eval import HostEvaluationResult
from tests.program_testutils import make_program_dir


def whole_card(axis='problem'):
    return {'schema_version':1, 'innovation_axis':axis,
        'problem_statement':'Improve a bounded task with a coherent whole idea.',
        'contribution':'Test joint candidate choices under one protocol.',
        'minimal_validation':{'protocol_id':'fixed-development-v1','decision_rule':'Retain a better correct candidate.',
                              'stop_or_pivot':'Stop at the declared trial limit.',
                              **({'cost_metric':'runtime_seconds','quality_guardrail':'correct is true'} if axis=='efficiency' else {})},
        'parameter_space':[
            {'name':'alpha','type':'float','bounds':[0.,1.],'default':.2,'owner':'candidate'},
            {'name':'width','type':'integer','values':[8,16],'default':8,'owner':'candidate'},
            {'name':'epochs','type':'integer','values':[5,10],'default':5,'owner':'task'},
        ], 'selection_reason':'Smallest task-permitted development test.'}


class ExternalFixture:
    def __init__(self, tmp_path):
        self.root = tmp_path
        self.task = tmp_path/'task'
        self.seed = self.task/'program_0'
        make_program_dir(self.seed, content="print('baseline')\n")
        (self.task/'task.md').write_text('Fixed task, fixed evaluator, whole-idea screen and joint tuning.\n')
        (self.task/'evaluate.py').write_text('# trusted evaluator fixture\n')
        self.python = self.task/'.venv/bin/python'
        self.python.parent.mkdir(parents=True)
        self.python.write_text('managed container runtime placeholder\n')
        self.results = tmp_path/'campaign'
        self.results.mkdir()
        self.policy = PortfolioConfig(enabled=True,initial_ideas=1,promote_ideas=1,
            tuning_trials_per_idea=1,replication_evals=1,max_valid_evals=3)
        self.controller = PortfolioController(self.results,self.task,self.policy)
        self.controller.qualify_baseline({'combined_score':.5,'correct':True},self.seed)
        self.generated = []
        self.coded = []
        self.host_calls = []
        self.cfg = InvestigationConfig()
        self.cfg.agent.name = 'codex'
        self.cfg.execution.max_idea_cycles = 1
        self.cfg.execution.max_program_evals_per_attempt = 1
        self.cfg.execution.harness_evaluated_attempts = True
        self.cfg.execution.search_strategy = 'tree'  # Must be ignored for external leases.
        self.cfg.execution.agent_transient_backoff_seconds = 0
        self.cfg.container.enabled = True
        self.cfg.container.image = 'sha256:fixture'
        self.cfg.paths.seed_program = str(self.seed)
        self.transport_once = False

    def generate(self, **kwargs):
        self.generated.append(kwargs)
        bundle = {'idea_id':'model_local_name','title':'Original whole idea','hypothesis':'Joint choices improve the bounded task.',
            'implementation_plan':['Tune both alpha and width together.'], 'target_rubrics':[
                {'id':'source','description':'Candidate source exists.','type':'implementation',
                 'verification':{'type':'source_contains','patterns':['print(']}}],
            'notes_for_agent':'Use the frozen task constraints.', 'whole_idea':whole_card()}
        if self.lease['stage']=='tuning':
            bundle['whole_idea']['parameter_space'][0]['default'] = .7
            bundle['whole_idea']['parameter_space'][1]['default'] = 16
            bundle['title'] = 'An accidental local trial title'
        workspace=kwargs['output_dir'];workspace.mkdir(parents=True)
        state_store.write_json(workspace/'idea_bundle.json',bundle)
        state_store.write_json(workspace/'target_rubrics.json',bundle['target_rubrics'])
        return IdeaGenerationResult('Fixture idea.',0,workspace)

    def attempt(self, **kwargs):
        self.coded.append(kwargs)
        workspace=Path(self.lease['result_dir'])/'ideas'/kwargs['idea_dir_name']/kwargs['attempt_dir_name']
        if self.transport_once:
            self.transport_once=False
            workspace.mkdir(parents=True)
            return RunAttemptResult('Provider transport failure.',1,workspace,agent_failure_kind='transient_provider')
        make_program_dir(workspace/'program_1',content="print(%r)\n" % self.lease['lease_id'])
        shutil.copyfile(self.task/'evaluate.py',workspace/'evaluate.py')
        code=run_harness_evaluation(experiment_dir=workspace,attempt_id=kwargs['attempt_id'],
            idea_id=self.lease['root_idea_id'],rubrics_path=kwargs['rubrics_file'],python_bin=str(self.python),
            global_memory_md=Path(self.lease['result_dir'])/'memory.md',
            global_memory_jsonl=Path(self.lease['result_dir'])/'memory.jsonl',timeout_seconds=10,
            sealed_config=SealedHarnessConfig(task_source_dir=self.task,sandbox_image='sha256:fixture',sandbox_gpus=None,environment={}))
        return RunAttemptResult('Fixture coded and host evaluated.',code,workspace)

    def evaluate(self, **kwargs):
        self.host_calls.append(kwargs)
        program=kwargs['task_bundle_dir']/'program_1'
        score=.8 if self.lease['stage']=='screening' else .85
        (program/'metrics.json').write_text(json.dumps({'combined_score':score}))
        (program/'correct.json').write_text('{"correct":true}')
        return HostEvaluationResult(returncode=0,correct=True,metrics={'combined_score':score},command=('docker','run'))

    def reserve(self):
        self.lease=self.controller.reserve_next(1)
        self.cfg.paths.seed_program=self.lease['parent_program_dir']
        plan=portfolio.build_portfolio_worker_plan(cfg=self.cfg,task_dir=self.task,results_dir=self.results,
            instruct_path=self.task/'task.md',lease=self.lease,context=self.controller.context(self.lease),
            build_parallel_worker_command_fn=build_parallel_worker_command)
        self.cfg.execution.portfolio_lease_file=str(plan.results_dir/portfolio.LEASE_FILE)
        return self.lease

    def run_child(self, resume=False):
        child=Path(self.lease['result_dir'])
        state_store.ensure_histories(child)
        self.cfg.execution.resume=resume
        outer=state_store.load_or_init_outer_state(child,self.seed,resume)
        best={'program_dir':str(self.seed),'combined_score':.5,'correct':True}
        with patch('src.variants_pipeline.host_eval.evaluate_variant_program',side_effect=self.evaluate), \
             patch.object(campaign_loop,'SearchTree',side_effect=AssertionError('external child must not select a local tree parent')):
            return campaign_loop.run_cycles_and_finalize(cfg=self.cfg,task_dir=self.task,results_dir=child,
                global_memory_dir=child,instruct_path=self.task/'task.md',idea_instruct_path=self.task/'task.md',
                paper_instruct_path=self.task/'task.md',paper_contributions_path=None,
                containerize=True,container_image='sha256:fixture',container_network='none',
                container_auth_root=self.root,container_env_file=None,container_session_name=None,container_mount_root=None,
                outer_state=outer,best_manifest=best,best_program_dir=self.seed,start_cycle=1,max_program_evals_per_attempt=1,
                main_started_at=0,log_event_fn=lambda *a,**k:None,log_step_start_fn=lambda *a,**k:0,
                log_step_end_fn=lambda *a,**k:None,format_elapsed_fn=lambda _: 'fixture',
                run_idea_generation_fn=self.generate,run_single_attempt_fn=self.attempt,
                run_paper_generation_fn=lambda **k:pytest.fail('child must not write paper'),
                reconcile_legacy_idea_local_memory_fn=lambda **k:None,
                find_best_program_in_experiment_fn=find_best_program_in_experiment,
                read_json_fn=state_store.load_json,utc_now_fn=lambda:'2026-10-18T00:00:00Z')

    def commit_child(self):
        outcome=portfolio._receipt_for_lease(self.lease,self.task)
        assert outcome
        self.controller.record_receipt(self.lease['lease_id'],outcome)
        return outcome


def test_external_whole_idea_tunes_exact_own_parent_and_repeats_without_model(tmp_path):
    fixture=ExternalFixture(tmp_path)
    fixture.reserve()
    assert fixture.run_child()==0
    screened=fixture.commit_child()
    tune=fixture.reserve()
    assert tune['parent_program_dir']==screened['program_dir']
    assert fixture.run_child()==0
    tuned=fixture.commit_child()
    assert fixture.coded[-1]['seed_program_dir']==Path(screened['program_dir'])
    assert tuned['idea_bundle']['title']=='Original whole idea'
    assert tuned['idea_bundle']['whole_idea']['parameter_space'][0]['default']==.7
    assert tuned['idea_bundle']['whole_idea']['parameter_space'][1]['default']==16
    repeat=fixture.reserve()
    assert repeat['stage']=='replication'
    assert fixture.run_child()==0
    repeated=fixture.commit_child()
    assert len(fixture.generated)==len(fixture.coded)==2
    assert len(fixture.host_calls)==3
    assert repeated['source_hash']==tuned['source_hash']
    assert repeated['idea_bundle']['whole_idea']['parameter_space'][0]['default']==.7
    assert fixture.controller.handoff()['candidate']['program_dir']==tuned['program_dir']
    assert not (Path(repeat['result_dir'])/'search_tree.json').exists()


def test_external_parent_transport_retry_resumes_same_idea_in_unused_workspace(tmp_path):
    fixture=ExternalFixture(tmp_path)
    fixture.reserve()
    fixture.cfg.execution.agent_transient_max_attempts=1
    fixture.transport_once=True
    assert fixture.run_child()==0  # The parent judges the missing host receipt.
    assert portfolio._receipt_for_lease(fixture.lease,fixture.task) is None
    assert fixture.run_child(resume=True)==0
    assert len(fixture.generated)==1
    assert [entry['attempt_id'] for entry in fixture.coded]==['idea_0001_attempt_01','idea_0001_attempt_02']
    assert len(fixture.host_calls)==1


def test_external_lease_rejects_parent_path_or_source_changes_before_agent_calls(tmp_path):
    fixture=ExternalFixture(tmp_path)
    fixture.reserve()
    other=fixture.task/'another_parent'
    shutil.copytree(fixture.seed,other)
    fixture.cfg.paths.seed_program=str(other)
    assert fixture.run_child()==2
    fixture.cfg.paths.seed_program=str(fixture.seed)
    (fixture.seed/'src/program.py').write_text("print('changed')\n")
    assert fixture.run_child()==2
    assert not fixture.generated and not fixture.coded


@pytest.mark.parametrize('mutation', ['claim','protocol','space','task_parameter'])
def test_joint_tuning_preserves_whole_claim_protocol_and_task_controls(mutation):
    source=whole_card()
    context={'root_idea_id':'idea_0001','lease_id':'trial2','stage':'tuning','innovation_axis':'problem','source_card':source}
    proposed=deepcopy(source)
    proposed['parameter_space'][0]['default']=.7
    proposed['parameter_space'][1]['default']=16
    bundle={'idea_id':'model','whole_idea':proposed}
    assert portfolio.bind_portfolio_idea(bundle,context)['whole_idea']['parameter_space'][1]['default']==16
    if mutation=='claim': proposed['contribution']='A different claim.'
    elif mutation=='protocol': proposed['minimal_validation']['protocol_id']='another-protocol'
    elif mutation=='space': proposed['parameter_space'][0]['bounds']=[0.,2.]
    else: proposed['parameter_space'][2]['default']=10
    with pytest.raises(ValueError):
        portfolio.bind_portfolio_idea(bundle,context)
