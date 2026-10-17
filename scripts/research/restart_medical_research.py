#!/usr/bin/env python3
"""Start a new, development-only campaign after retaining the previous study."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import uuid

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / "docs/research/medical-unlearning/restart_research_context.md"
EXPECTED_TIMEOUTS = {"timeout":300000,"headerTimeout":90000,"chunkTimeout":90000}


def require_restart(record, state, source, retained, transport):
    authorization = record.get("research_restart_authorization", {})
    if not authorization.get("human_authorized"):
        raise ValueError("The human user must authorize a new research campaign")
    if state["running"] or state["exit_code"] is None:
        raise ValueError("The previous campaign must be fully stopped")
    if record.get("development_campaign"):
        raise ValueError("A new development campaign already exists; inspect its checkpoint")
    expected = record["qualified_source_task"]
    if (not source.get("ready")
            or source["publication_plan_sha256"] != expected["publication_plan_sha256"]
            or source["image_id"] != expected["image_id"]):
        raise ValueError("The frozen benchmark no longer matches its qualified source")
    if len(retained) != 5 or not all(x["unchanged"] for x in retained.values()):
        raise ValueError("All five previous sealed final artifacts must remain unchanged")
    repair = record.get("variant_agent_isolation_finding", {}).get("repair", {})
    if not repair.get("real_container_canary_pass"):
        raise ValueError("An actual container source-isolation check is required")
    if (not transport.get("passed") or transport.get("image_id") != source["image_id"]
            or {x["mode"] for x in transport.get("checks", [])}
               != {"success", "stream_control", "headers", "chunks", "total"}
            or not all(x.get("low_effort_verified") for x in transport.get("checks", []))):
        raise ValueError("Actual model transport and timeout checks must pass")
    if not record.get("automation_control", {}).get("all_paused"):
        raise ValueError("Keep scheduled tasks paused as the human user requested")


def overrides(source: str, image: str) -> list[str]:
    return ["preset=medical_unlearning", "task.dir=" + source,
            "execution.phases=investigation", "container.image=" + image,
            "investigation.execution.max_idea_cycles=4",
            "investigation.execution.max_program_evals_per_attempt=1",
            "investigation.execution.idea_timeout_seconds=1800",
            "investigation.execution.run_timeout_seconds=1800",
            "investigation.execution.harness_evaluation_timeout_seconds=2100"]


def campaign_record(previous: dict, job: str, directory: str, results: str,
                    source: dict, proof: dict, archive: str) -> dict:
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    keep = ("tool", "agent", "model", "host", "budget_authorization",
            "automation_control", "qualified_source_task", "boundary_f1_repair",
            "runtime_snapshot_repair", "runtime_repairs", "verification_gate")
    record = {k:previous[k] for k in keep if k in previous}
    record.update(updated_at=now, job_id=job, status="submitting_development_campaign",
                  phase="investigation", substage="revalidating_frozen_baseline",
                  remote_log=directory + "/run.log", remote_results=results,
                  phases=["investigation"], scientific_results_available=False,
                  formal_experiment_results_available=False, paper_available=False,
                  submission_authorized=False, full_study_complete=False,
                  blocking_findings=["independent_confirmatory_evidence_not_available"],
                  previous_study={"job_id":previous["job_id"], "record":archive,
                      "sealed_final_correct":False,
                      "source_isolation_of_earlier_ablations_qualified":False,
                      "results_retained":True, "sealed_final_repeat_authorized":False},
                  research_restart_authorization=previous["research_restart_authorization"],
                  development_campaign={"version":2,"started_at":now,
                      "scope":"Adaptive development on the previously observed SEARCH benchmark",
                      "research_question":"Improve forget/retrain agreement while preserving fine-tuning utility",
                      "max_idea_cycles":4,"evaluation_calls_per_attempt":1,
                      "old_final_quarantined":True,"final_evaluation_enabled":False,
                      "confirmatory_claims_authorized":False,"startup_proof":proof},
                  final_evaluation_policy={"enabled":False,
                      "reason":"The old sealed packet has already been observed; do not reuse it for adaptive selection",
                      "previous_final_unchanged":True},
                  last_monitor_check={"checked_at":now,"status":"submitted_new_development_campaign",
                      "check_mode":"manual","gpu_training_observed":False,
                      "source_gates_valid":source["ready"]},
                  next_action={"requires_user_decision":False,
                      "action":"Directly verify the new baseline, model transport and isolated agents; inspect each native development evaluation. Keep old final and all scheduled tasks closed. Independently compare utility and forgetting before expanded experiments."})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    launcher = runpy.run_path(str(ROOT / "scripts/remote-agsci"))
    recovery = runpy.run_path(str(ROOT / "scripts/research/resume_medical_drafting.py"))
    continuation = runpy.run_path(str(ROOT / "scripts/research/continue_medical_experiment.py"))
    record_path = ROOT / "docs/research/medical-unlearning/execution.json"
    with (ROOT / "scratch/medical-recovery-submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(record_path.read_text())
        base = launcher["REMOTE"]
        old = record["job_id"]
        source_path = record["qualified_source_task"]["path"]
        source_job = record["qualified_source_task"]["source_job"]

        def remote_python(code, argv=(), *, cwd=None):
            command = ("cd " + shlex.quote(cwd or base + "/workspace") + "\n"
                       + shlex.join([base + "/.venv/bin/python", "-", *argv])
                       + " <<'PY'\n" + code + "\nPY\n")
            return json.loads(launcher["remote"](command, capture=True).stdout)

        probe = (ROOT / "scripts/research/medical_job_state.py").read_text()
        state = remote_python(probe, [base, old])
        if record.get("development_campaign"):
            if args.run:
                raise ValueError("The development campaign already exists; do not submit a duplicate")
            print(json.dumps({"ready":False,"reason":"Development campaign already exists",
                              "job_id":old,"state":state}, indent=2))
            return
        source = remote_python(continuation["VERIFY"],
            [source_path, record["boundary_f1_repair"]["expected_metric_ast_sha256"]],
            cwd=base + "/jobs/" + source_job)
        evidence_path = Path(record["sealed_final_result"]["local_evidence"])
        expected = {k:v for k,v in json.loads(evidence_path.read_text())["artifact_sha256"].items()
                    if k.startswith("final_evaluation/")}
        retained = remote_python("""from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]);expected=json.loads(sys.argv[2])
print(json.dumps({k:{'expected':v,'actual':hashlib.sha256((base/k).read_bytes()).hexdigest(),
    'unchanged':hashlib.sha256((base/k).read_bytes()).hexdigest()==v} for k,v in expected.items()}))
""", [record["remote_results"], json.dumps(expected)])
        transport = remote_python("""import json
from pathlib import Path
print((Path('/home/sunguanqun/AI-Scientist-v3/scratch/opencode-low-effort-transport-verification.json')).read_text())
""")
        require_restart(record, state, source, retained, transport)
        for relative, key in (("src/scripts/variant_agent.sh", "wrapper_sha256"),
                              ("src/scripts/lib/variant_task_shadow.py", "helper_sha256")):
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != record["variant_agent_isolation_finding"]["repair"][key]:
                raise ValueError("Isolation wrapper changed since its actual-container test")
        config = json.loads((ROOT / "configs/opencode-qwen.json").read_text())
        options = config["provider"]["qwen-plan"]["options"]
        if {k:options[k] for k in EXPECTED_TIMEOUTS} != EXPECTED_TIMEOUTS:
            raise ValueError("The deployment must use the verified model wait settings")
        if config["provider"]["qwen-plan"]["models"]["qwen3.8-max"].get("options") != {"reasoningEffort":"low"}:
            raise ValueError("The model reasoning profile must match the actual request-body check")
        if not args.run:
            print(json.dumps({"ready":True,"previous_job":old,"state":state,
                              "source":source,"phases":["investigation"],
                              "old_final_unchanged":True,"transport_passed":True}, indent=2))
            return
        job = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        directory, results = base + "/jobs/" + job, base + "/results/" + job
        launcher["sync"](directory)
        launcher["stage_auth"](directory)
        context = CONTEXT.read_text()
        staged = remote_python("""import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);context=sys.argv[2];hashes={}
for name in ('src/prompts/idea.md','src/prompts/harness_edit.md'):
 p=root/name;original=p.read_bytes();(p.with_suffix(p.suffix+'.before-study2')).write_bytes(original)
 p.write_text(original.decode()+'\\n\\n'+context)
 hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
(root/'research_context.md').write_text(context)
print(json.dumps({'prompt_sha256':hashes,'context_sha256':hashlib.sha256(context.encode()).hexdigest()}))
""", [directory, context])
        command = launcher["args_for"]("run", overrides(source_path, source["image_id"]), directory)
        command += ["run_id=" + job, "paths.run_dir=" + results]
        # Recheck after staging; never launch beside an old process or container.
        require_restart(record, remote_python(probe, [base, old]), source, retained, transport)
        runner = recovery["runner_text"](launcher, directory, command)
        launcher["remote"]("printf %s " + shlex.quote(runner) + " > "
                           + shlex.quote(directory + "/run.sh") + "\n")
        archive = ROOT / "docs/research/medical-unlearning/studies" / old
        archive.mkdir(parents=True, exist_ok=False)
        previous = json.loads(json.dumps(record))
        previous.update(status="stopped_for_user_authorized_new_research",
                        last_exit_code=state["exit_code"], research_process_running=False)
        (archive / "execution.json").write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
        (archive / "final_integrity.json").write_text(json.dumps(retained, indent=2) + "\n")
        phase_audit = ROOT / "docs/research/medical-unlearning/phase_audit.json"
        (archive / "phase_audit.json").write_bytes(phase_audit.read_bytes())
        proof = {"old_state":state,"source_gates":source,"sealed_final_hashes":retained,
                 "transport":transport,"prompt_overrides":staged,
                 "configuration_sha256":hashlib.sha256((ROOT / "configs/opencode-qwen.json").read_bytes()).hexdigest()}
        record = campaign_record(record, job, directory, results, source, proof,
                                 str(archive / "execution.json"))
        recovery["atomic_record"](record_path, record)
        launcher["remote"]("cd " + shlex.quote(directory)
                           + "\nnohup bash run.sh > run.log 2>&1 < /dev/null &\necho $! > pid\n")
        sampler = remote_python("""import json,subprocess,sys
from pathlib import Path
d=Path(sys.argv[1]);pid=int((d/'pid').read_text());csv=d/'gpu_samples.csv';state=d/'gpu_watch.json'
with (d/'gpu_watch.log').open('w') as out:
 p=subprocess.Popen([sys.executable,str(d/'scripts/research/watch_research_gpus.py'),
    '--pid',str(pid),'--uuids',sys.argv[2],'--csv',str(csv),'--state',str(state)],
    stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'csv':str(csv),'state':str(state),
    'script':str(d/'scripts/research/watch_research_gpus.py'),'target_pid':pid}))
""", [directory, ",".join(launcher["CFG"]["gpu_uuids"])])
        record["gpu_sampling"] = sampler
        record["status"] = "running_new_development_campaign"
        recovery["atomic_record"](record_path, record)
        print(json.dumps({"job_id":job,"results":results,"phases":["investigation"],
                          "previous_study":str(archive),"old_final_repeated":False}))


if __name__ == "__main__":
    main()
