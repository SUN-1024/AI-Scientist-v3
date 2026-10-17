"""Ablation recovery preserves the completed study and its sealed final result."""
import copy
from pathlib import Path
import runpy

import pytest

MODULE=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/research/retry_medical_ablations.py'))


def fixture():
    record={'budget_authorization':{'automatic_recovery_authorized':True,'overall_time_limit_seconds':None},
            'sealed_final_result':{'result_directory_immutable':True},
            'runtime_snapshot_repair':{'new_tooling_sha256':'frozen'}}
    state={'running':False,'exit_code':0}
    witness={'errors':[],'cycle_budget':2,'current_cycle':2,'completed_attempts':2,
             'best_correct':True,'ablation_exit_code':4,'tooling_sha256':'frozen'}
    return record,state,witness,{'ready':True}


def test_accepts_only_completed_campaign_with_retained_final_and_failed_ablations():
    MODULE['require_retry'](*fixture())


@pytest.mark.parametrize('change',[
    ('state','running',True),('state','exit_code',None),('state','exit_code',2),
    ('witness','cycle_budget',3),('witness','completed_attempts',1),
    ('witness','errors',['sealed final changed']),('witness','best_correct',False),
    ('witness','ablation_exit_code',0),('witness','tooling_sha256','new'),
    ('validation','ready',False),('record','ablation_recoveries',[{'output_dir':'existing'}]),
])
def test_rejects_overlap_protocol_drift_final_changes_and_discarded_retry(change):
    values=list(copy.deepcopy(fixture()));index={'record':0,'state':1,'witness':2,'validation':3}[change[0]]
    values[index][change[1]]=change[2]
    with pytest.raises(ValueError):MODULE['require_retry'](*values)
