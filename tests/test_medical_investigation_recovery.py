"""Checkpoint recovery must not discard research or bypass scientific gates."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest

M=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/research/resume_medical_investigation.py'))


def valid_case():
    return ({'phase':'investigation','budget_authorization':{
                'automatic_recovery_authorized':True,'overall_time_limit_seconds':None,
                'overall_model_call_limit':None},'blocking_findings':[],
             'current_investigation_progress':{'max_cycles':2}},
            {'running':False,'exit_code':2},
            {'errors':[],'cycle_budget':2,'current_cycle':0,'later_outputs':[],
             'last_phase_event':{'event':'phase_end','phase_name':'investigation','status':'failed'}},
            {'ready':True})


def test_stopped_failure_keeps_original_unfinished_checkpoint():
    record,state,witness,validation=valid_case()
    original=deepcopy((record,state,witness,validation))
    M['require_resumable'](record,state,witness,validation)
    assert (record,state,witness,validation)==original


def test_development_resume_preserves_four_cycles_and_keeps_the_old_final_disabled():
    record,state,witness,validation=valid_case()
    record.update(development_campaign={'version':2,'max_idea_cycles':4,'final_evaluation_enabled':False},
                  research_restart_authorization={'human_authorized':True},
                  blocking_findings=['independent_confirmatory_evidence_not_available'])
    record['current_investigation_progress']['max_cycles']=4
    witness['cycle_budget']=4
    M['require_resumable'](record,state,witness,validation)
    overrides=M['recovery_overrides'](record,'/source','image')
    assert 'execution.phases=investigation' in overrides
    assert 'investigation.execution.max_idea_cycles=4' in overrides
    assert not any('final_evaluation' in x for x in overrides)
    record['blocking_findings'].append('metric_changed')
    with pytest.raises(ValueError):M['require_resumable'](record,state,witness,validation)


@pytest.mark.parametrize('change',['final_enabled','no_authorization'])
def test_development_resume_does_not_relax_the_final_data_boundary(change):
    record,state,witness,validation=valid_case()
    record.update(development_campaign={'version':2,'max_idea_cycles':4,'final_evaluation_enabled':False},
                  research_restart_authorization={'human_authorized':True})
    if change=='final_enabled':record['development_campaign']['final_evaluation_enabled']=True
    if change=='no_authorization':record['research_restart_authorization']={}
    with pytest.raises(ValueError):M['require_resumable'](record,state,witness,validation)


@pytest.mark.parametrize('change',[
    'active','success','unknown_exit','wrong_phase','no_authorization','time_cap','call_cap',
    'invalid_source','scientific_blocker','incomplete_phase','other_failed_phase','completed_phase',
    'later_research','broken_checkpoint','changed_cycle_budget','completed_cycles',
    'invalid_cycle','boolean_cycle','missing_budget',
])
def test_unsafe_or_unrelated_recovery_is_refused(change):
    record,state,witness,validation=valid_case()
    if change=='active':state['running']=True
    if change=='success':state['exit_code']=0
    if change=='unknown_exit':state['exit_code']=None
    if change=='wrong_phase':record['phase']='final_evaluation'
    if change=='no_authorization':record['budget_authorization']['automatic_recovery_authorized']=False
    if change=='time_cap':record['budget_authorization']['overall_time_limit_seconds']=7200
    if change=='call_cap':record['budget_authorization']['overall_model_call_limit']=4
    if change=='invalid_source':validation['ready']=False
    if change=='scientific_blocker':record['blocking_findings']=['metric_changed']
    if change=='incomplete_phase':witness['last_phase_event']['event']='phase_start'
    if change=='other_failed_phase':witness['last_phase_event']['phase_name']='final_evaluation'
    if change=='completed_phase':witness['last_phase_event']['status']='completed'
    if change=='later_research':witness['later_outputs']=['final_evaluation']
    if change=='broken_checkpoint':witness['errors']=['missing program source']
    if change=='changed_cycle_budget':witness['cycle_budget']=3
    if change=='completed_cycles':witness['current_cycle']=2
    if change=='invalid_cycle':witness['current_cycle']=-1
    if change=='boolean_cycle':witness['current_cycle']=False
    if change=='missing_budget':witness['cycle_budget']=None
    with pytest.raises(ValueError):M['require_resumable'](record,state,witness,validation)


def test_checkpoint_probe_rejects_real_changed_private_evaluator(tmp_path):
    """Exercise the remote probe with actual framework checkpoint validation."""
    import hashlib,json,subprocess,sys
    investigation=tmp_path/'investigation';investigation.mkdir()
    (investigation/'outer_state.json').write_text('{"current_cycle":0}')
    (investigation/'best_program_manifest.json').write_text('{}')
    (tmp_path/'task').mkdir();(tmp_path/'task/evaluate.py').write_text('changed')
    (tmp_path/'pipeline_events.jsonl').write_text('')
    expected={'evaluate.py':hashlib.sha256(b'original').hexdigest()}
    result=subprocess.run([sys.executable,'-c',M['CHECKPOINT'],str(tmp_path),json.dumps(expected)],
                          text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr
    witness=json.loads(result.stdout)
    assert 'Private evaluator changed: evaluate.py' in witness['errors']
    assert any('usable program' in error for error in witness['errors'])


def test_recovery_archive_preserves_failed_exit_logs_and_research(tmp_path):
    import subprocess,sys
    job=tmp_path/'jobs/run';job.mkdir(parents=True)
    files={'exit-code':'2\n','pid':'123\n','run.log':'failed: timeout\n','run.sh':'#!/bin/bash\nexit 2\n'}
    for name,body in files.items():(job/name).write_text(body)
    research=tmp_path/'results/run/investigation';research.mkdir(parents=True)
    (research/'partial_candidate.py').write_text('important saved work')
    archive=job/'attempts/retry1'
    command=[sys.executable,'-c',M['ARCHIVE'],str(job),str(archive)]
    result=subprocess.run(command,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    for name,body in files.items():
        assert (job/name).read_text()==body
        assert (archive/name).read_text()==body
    assert (research/'partial_candidate.py').read_text()=='important saved work'
    # Repeated submission cannot overwrite retained evidence.
    result=subprocess.run(command,text=True,capture_output=True,timeout=10)
    assert result.returncode!=0
    assert (job/'exit-code').read_text()=='2\n'
