"""A new adaptive campaign must retain the old final and refuse overlap."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           "scripts/research/restart_medical_research.py"))


def _case():
    record = {"research_restart_authorization":{"human_authorized":True},
              "qualified_source_task":{"publication_plan_sha256":"plan","image_id":"image"},
              "variant_agent_isolation_finding":{"repair":{"real_container_canary_pass":True}},
              "automation_control":{"all_paused":True}}
    state = {"running":False,"exit_code":143}
    source = {"ready":True,"publication_plan_sha256":"plan","image_id":"image"}
    retained = {str(i):{"unchanged":True} for i in range(5)}
    transport = {"passed":True,"image_id":"image",
                 "checks":[{"mode":s}for s in
                           ("success","stream_control","headers","chunks","total")]}
    return deepcopy((record,state,source,retained,transport))


def test_accepts_new_development_campaign_without_old_final_reuse():
    MODULE["require_restart"](*_case())
    args = MODULE["overrides"]("/frozen/source", "image")
    assert "execution.phases=investigation" in args
    assert "investigation.execution.max_program_evals_per_attempt=1" in args
    assert not any("final_evaluation" in arg for arg in args)


@pytest.mark.parametrize("change", ["authorization","running","no_exit","duplicate",
    "source","plan","image","final_changed","final_missing","isolation",
    "transport","transport_image","transport_missing","schedules"])
def test_refuses_drift_overlap_unverified_transport_and_lost_old_results(change):
    r,s,v,f,t = _case()
    if change == "authorization":r["research_restart_authorization"] = {}
    if change == "running":s["running"] = True
    if change == "no_exit":s["exit_code"] = None
    if change == "duplicate":r["development_campaign"] = {"version":2}
    if change == "source":v["ready"] = False
    if change == "plan":v["publication_plan_sha256"] = "other"
    if change == "image":v["image_id"] = "other"
    if change == "final_changed":f["0"]["unchanged"] = False
    if change == "final_missing":f.pop("0")
    if change == "isolation":r["variant_agent_isolation_finding"]["repair"] = {}
    if change == "transport":t["passed"] = False
    if change == "transport_image":t["image_id"] = "other"
    if change == "transport_missing":t["checks"].pop()
    if change == "schedules":r["automation_control"]["all_paused"] = False
    with pytest.raises(ValueError):MODULE["require_restart"](r,s,v,f,t)


def test_new_record_does_not_carry_old_progress_or_claim_confirmatory_evidence():
    r,s,v,f,t = _case()
    r.update(job_id="old",scientific_results_available=True,paper_available=True,
             current_investigation_progress={"cycle":2},
             sealed_final_result={"correct":False},phase_output_overrides={"ablations":"old"})
    new = MODULE["campaign_record"](r,"new","/jobs/new","/results/new",v,{},"/archive")
    assert new["scientific_results_available"] is False
    assert new["paper_available"] is False
    assert "current_investigation_progress" not in new
    assert "sealed_final_result" not in new
    assert "phase_output_overrides" not in new
    assert new["development_campaign"]["final_evaluation_enabled"] is False
    assert new["final_evaluation_policy"]["enabled"] is False
    assert new["previous_study"]["results_retained"] is True
