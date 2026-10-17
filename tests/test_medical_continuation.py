"""Downstream work cannot bypass the preceding task's verified completion."""
from pathlib import Path
import json
import runpy
import subprocess
import sys
import pytest

M = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/research/continue_medical_experiment.py'))


@pytest.mark.parametrize('state,verified', [
    ({'running':True,'exit_code':0}, {'ready':True}),
    ({'running':False,'exit_code':None}, {'ready':True}),
    ({'running':False,'exit_code':4}, {'ready':True}),
    ({'running':False,'exit_code':0}, {'ready':False}),
])
def test_downstream_refuses_unfinished_failed_or_unverified_draft(state,verified):
    with pytest.raises(ValueError):
        M['require_ready'](state,verified)


def test_completed_verified_draft_can_continue_with_original_protocol():
    M['require_ready']({'running':False,'exit_code':0},{'ready':True})
    values = M['overrides']('/latest/published/task','sha256:verified')
    assert 'preset=medical_unlearning' in values
    assert 'task.dir=/latest/published/task' in values
    assert 'container.image=sha256:verified' in values
    assert not any(v.startswith(('investigation.execution.','ablations.execution.','final.')) for v in values)


def test_uncapped_model_wait_changes_only_agent_watchdogs():
    record={'budget_authorization':{'automatic_recovery_authorized':True,
                                   'overall_time_limit_seconds':None,
                                   'overall_model_call_limit':None}}
    values=M['agent_wait_overrides'](record)
    assert set(values)=={'investigation.execution.idea_timeout_seconds=7200',
                        'investigation.execution.run_timeout_seconds=7200'}
    # The medical preset keeps all fixed scientific work and evaluator deadlines.
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs/preset/medical_unlearning.yaml')
    assert cfg.investigation.execution.harness_evaluation_timeout_seconds==2100
    assert cfg.investigation.execution.max_idea_cycles==2
    assert cfg.investigation.execution.max_program_evals_per_attempt==1
    assert cfg.ablations.execution.max_evals_per_variant==1
    assert cfg.final.timeout_seconds==3900


@pytest.mark.parametrize('change',['authorization','time_limit','call_limit','missing_time','missing_calls'])
def test_model_watchdog_override_requires_explicit_uncapped_authorization(change):
    auth={'automatic_recovery_authorized':True,'overall_time_limit_seconds':None,
          'overall_model_call_limit':None}
    if change=='authorization':auth['automatic_recovery_authorized']=False
    if change=='time_limit':auth['overall_time_limit_seconds']=3600
    if change=='call_limit':auth['overall_model_call_limit']=10
    if change=='missing_time':auth.pop('overall_time_limit_seconds')
    if change=='missing_calls':auth.pop('overall_model_call_limit')
    assert M['agent_wait_overrides']({'budget_authorization':auth})==[]


@pytest.mark.parametrize('change', ['', 'active', 'success', 'unknown_exit', 'research_exists',
                                   'other_phase', 'different_failure', 'missing_source'])
def test_initialization_retry_preserves_research_and_requires_a_qualified_source(change):
    record={'phase':'investigation','qualified_source_task':{'path':'/qualified/task'}}
    state={'running':False,'exit_code':1}
    inventory={'entries':['pipeline_events.jsonl'],'startup_failure':True}
    if change=='active':state['running']=True
    if change=='success':state['exit_code']=0
    if change=='unknown_exit':state['exit_code']=None
    if change=='research_exists':inventory['entries'].append('investigation')
    if change=='other_phase':record['phase']='final_evaluation'
    if change=='different_failure':inventory['startup_failure']=False
    if change=='missing_source':record['qualified_source_task']={}
    if change:
        with pytest.raises(ValueError):M['require_initialization_retry'](record,state,inventory)
    else:M['require_initialization_retry'](record,state,inventory)


@pytest.mark.parametrize('extra', ['', 'idea_1', 'run_summary.csv'])
def test_runtime_preflight_retry_requires_host_no_checkpoint_witness_and_no_research(extra):
    record={'phase':'investigation','qualified_source_task':{'path':'/qualified/task'}}
    state={'running':False,'exit_code':2}
    inventory={'entries':['task','investigation','pipeline_events.jsonl','.task.runtime-build.lock',
                          '.investigation_completion.json','.agentic_resume_policy.json',
                          '.task.agentic_task_ready.'+'a'*64],
               'startup_failure':False,'no_usable_checkpoint':True,
               'investigation_entries':['events.jsonl']+([extra] if extra else [])}
    if extra:
        with pytest.raises(ValueError):M['require_initialization_retry'](record,state,inventory)
    else:
        M['require_initialization_retry'](record,state,inventory)
        inventory['no_usable_checkpoint']=False
        with pytest.raises(ValueError):M['require_initialization_retry'](record,state,inventory)


@pytest.mark.parametrize('changed', [False, True])
def test_release_checks_corrected_metric_before_remaining_gates(tmp_path, changed):
    root = Path(__file__).resolve().parents[1]
    metric = runpy.run_path(str(root / 'scripts/research/fix_medical_boundary_f1.py'))
    source = metric['COUNTS'] + '\n\n' + metric['MACRO'] + '\n'
    (tmp_path / 'evaluate.py').write_text(source)
    (tmp_path / 'evaluate_final.py').write_text(
        source.replace('precision + recall', 'precision + 2 * recall') if changed else source
    )
    run = subprocess.run([sys.executable, '-c', M['VERIFY'], str(tmp_path), metric['EXPECTED_AST']],
                         cwd=root, text=True, capture_output=True, timeout=20)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert not result['ready']
    if changed:
        assert 'Verified boundary metric changed: evaluate_final.py' in result['reason']
    else:
        # The correct metric alone cannot release a task with no contract/receipts.
        assert 'task_contract.json' in result['reason']
