"""Recovery must preserve used evaluation budgets and the one-time final run."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           "scripts/research/resume_medical_variant_checkpoint.py"))


def _case():
    record = {
        "budget_authorization": {"automatic_recovery_authorized":True,
                                 "overall_time_limit_seconds":None},
        "runtime_snapshot_repair":{"new_tooling_sha256":"frozen"},
        "variant_agent_isolation_finding":{"repair":{"real_container_canary_pass":True}},
    }
    state = {"running":False,"exit_code":143}
    witness = {"errors":[],"completed":sorted(MODULE["COMPLETED"]),
               "unused":MODULE["REMAINING"],
               "variant_ids":sorted(MODULE["COMPLETED"] | set(MODULE["REMAINING"])),
               "tooling_sha256":"frozen"}
    return deepcopy((record,state,witness,{"ready":True}))


def test_accepts_only_original_unused_variant_budgets():
    MODULE["require_checkpoint"](*_case())


@pytest.mark.parametrize("change", ["active","wrong_exit","already_used","missing_variant",
                                   "bad_source","changed_final","changed_tooling",
                                   "no_canary","duplicate","no_authorization"])
def test_refuses_unsafe_or_repeated_recovery(change):
    r,s,w,v = _case()
    if change == "active":s["running"] = True
    if change == "wrong_exit":s["exit_code"] = 0
    if change == "already_used":w["unused"] = w["unused"][1:]
    if change == "missing_variant":w["variant_ids"] = w["variant_ids"][1:]
    if change == "bad_source":v["ready"] = False
    if change == "changed_final":w["errors"] = ["sealed final changed"]
    if change == "changed_tooling":w["tooling_sha256"] = "different"
    if change == "no_canary":r["variant_agent_isolation_finding"]["repair"] = {}
    if change == "duplicate":r["variant_checkpoint_recoveries"] = [{"attempt":"previous"}]
    if change == "no_authorization":r["budget_authorization"] = {}
    with pytest.raises(ValueError):MODULE["require_checkpoint"](r,s,w,v)
