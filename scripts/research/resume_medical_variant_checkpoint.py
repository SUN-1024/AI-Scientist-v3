#!/usr/bin/env python3
"""Inspect or resume the three unused variant budgets after the source-view fix."""

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
COMPLETED = {
    "control_best", "baseline_reference", "ablation_no_boundary_anchor",
    "ablation_no_orthogonalization",
}
REMAINING = [
    "baseline_gradient_ascent", "baseline_joint_neggrad", "baseline_boundary_retain_only",
]

CHECKPOINT = r'''
from pathlib import Path
import hashlib,json,sys
from src.investigation.task_runtime import _tooling_digest
from src.variants_pipeline.pipeline import _load_variant_specs_from_disk
from src.variants_pipeline.execute import _canonical_rubric_passed, _check_analysis_files
from src.variants_pipeline.rubric_contract import build_rubric_contract
p=json.loads(sys.argv[1]);output=Path(p['output']);base=Path(p['results'])
errors=[];completed=[];unused=[];hashes={}
for relative,expected in p['final_hashes'].items():
    if hashlib.sha256((base/relative).read_bytes()).hexdigest()!=expected:
        errors.append('sealed final changed: '+relative)
specs=_load_variant_specs_from_disk(output)
for spec in specs:
    manifest=json.loads((spec.variant_dir/'run_manifest.json').read_text())
    bundle=spec.variant_dir/'task_bundle'
    evaluated=list(bundle.glob('program_*/evaluation_manifest.json'))
    if manifest['status']=='completed':
        required=[r['id'] for r in build_rubric_contract(spec.variant_id,spec.kind,spec.base_program)['rubrics']]
        if manifest['eval_count']!=1 or len(evaluated)!=1:
            errors.append('completed budget mismatch: '+spec.variant_id);continue
        program=evaluated[0].parent;index=int(program.name.split('_')[1])
        rubric=json.loads((program/'rubric_status.json').read_text())
        evaluation=json.loads(evaluated[0].read_text())
        if (not _canonical_rubric_passed(program,index,spec.variant_id,required,rubric)
                or evaluation['returncode']!=0
                or not all(_check_analysis_files(bundle).values())):
            errors.append('invalid completed native receipt: '+spec.variant_id);continue
        completed.append(spec.variant_id)
        paths=[spec.variant_dir/'run_manifest.json',spec.variant_dir/'variant_spec.json',
               spec.variant_dir/'target_rubrics.json',bundle/'rubric_status.json']
        for candidate in sorted(bundle.glob('program_*')):
            for relative in ('program_spec.toml','src/program.py','metrics.json','correct.json',
                             'evaluation_manifest.json','evaluation_evidence.json','rubric_status.json',
                             'analysis/run_summary.csv','analysis/metric_values.csv','analysis/timeseries.csv'):
                path=candidate/relative
                if path.is_file():paths.append(path)
        for path in paths:
            if path.is_symlink() or not path.is_file():raise ValueError('unsafe receipt: '+str(path))
            hashes[str(path.relative_to(output))]=hashlib.sha256(path.read_bytes()).hexdigest()
    elif not evaluated and manifest.get('eval_count',0)==0:
        unused.append(spec.variant_id)
    else:
        errors.append('an incomplete variant already used an evaluation budget: '+spec.variant_id)
selection=output/'generation/variant_selection.json'
hashes[str(selection.relative_to(output))]=hashlib.sha256(selection.read_bytes()).hexdigest()
print(json.dumps({'errors':errors,'completed':completed,'unused':unused,
                  'variant_ids':[s.variant_id for s in specs],'hashes':hashes,
                  'tooling_sha256':_tooling_digest()}))
'''

WORKER = r'''
from pathlib import Path
import hashlib,json,logging,os,shutil,sys
from datetime import datetime,timezone
from src.investigation.evaluator_tree import managed_evaluator_tree_digest
from src.investigation.task_runtime import _tooling_digest
from src.variants_pipeline.cli import load_variants_config
from src.variants_pipeline.execute import execute_variant_agents, _check_analysis_files
from src.variants_pipeline.materialize import materialize_variant_workspaces
from src.variants_pipeline.models import VariantRunResult,VariantSelection
from src.variants_pipeline.pipeline import (_load_variant_specs_from_disk,
    _load_candidates_from_disk,_write_pipeline_summary)
from src.variants_pipeline.aggregate import aggregate_variant_csvs
from src.variants_pipeline.readme import generate_variants_readme
from src.variants_pipeline.retrofit import retrofit_bundle_complete

p=json.loads(Path(sys.argv[1]).read_text());output=Path(p['output']);base=Path(p['results']);task=base/'task'
archive=Path(p['archive']);w=p['witness'];logging.basicConfig(level=logging.INFO)
def verify_retained():
    for relative,expected in w['hashes'].items():
        path=output/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('retained checkpoint changed: '+relative)
    for relative,expected in p['final_hashes'].items():
        if hashlib.sha256((base/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError('sealed final changed: '+relative)
    if _tooling_digest()!=w['tooling_sha256']:raise ValueError('scientific harness changed')
def event(name,**fields):
    with (output/'events.jsonl').open('a') as f:
        f.write(json.dumps({'time':datetime.now(timezone.utc).isoformat(),
                            'event':name,'fields':fields})+'\n')
verify_retained()
retained_backup=archive/'retained_checkpoint';retained_backup.mkdir(mode=0o700)
for relative in w['hashes']:
    target=retained_backup/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(output/relative,target)
command=p['original_command'];args=command[command.index('src.variants_pipeline.cli')+1:]
cfg=load_variants_config(args)
if (cfg.execution.max_ablations!=2 or cfg.execution.max_extra_baselines!=3
        or cfg.execution.max_evals_per_variant!=1 or cfg.execution.variant_timeout_seconds!=2400):
    raise ValueError('original comparison budgets changed')
if Path(cfg.paths.task_dir)!=task or Path(cfg.paths.output_dir)!=output:
    raise ValueError('original task/output paths changed')
specs=_load_variant_specs_from_disk(output);by_id={s.variant_id:s for s in specs}
bundle=output/'_extracted_bundle';retrofit=output/'_retrofitted_bundle'
if not retrofit_bundle_complete(retrofit,bundle,task_source_dir=task):
    raise ValueError('retrofitted seed is not intact')
for name in p['remaining']:
    directory=by_id[name].variant_dir
    if list((directory/'task_bundle').glob('program_*/evaluation_manifest.json')):
        raise ValueError('unused variant already evaluated: '+name)
backup=archive/'exposed_unused_workspaces';backup.mkdir(mode=0o700)
for name in p['remaining']:
    shutil.move(str(by_id[name].variant_dir),str(backup/name))
candidates={c.variant_id:c for c in _load_candidates_from_disk(output/'generation')}
fresh=materialize_variant_workspaces(bundle_dir=bundle,retrofitted_bundle_dir=retrofit,
    variants_run_dir=output,selection=VariantSelection(
        accepted=[candidates[n] for n in p['remaining']],rejected=[],controls=[]),
    agent=cfg.agent.name,agent_model=cfg.agent.model,agent_profile=cfg.agent.profile,
    reasoning_effort=cfg.agent.reasoning_effort,max_evals_per_variant=1,custom_only=True)
retained=[VariantRunResult(variant_id=n,status='completed',eval_count=1,rubric_passed=True,
    exit_code=0,output='Retained native execution; prior authoring source view was exposed.',
    analysis_files=_check_analysis_files(by_id[n].variant_dir/'task_bundle')) for n in p['completed']]
event('checkpoint_resume',retained_variants=p['completed'],unused_variant_budgets=p['remaining'],
      prior_source_isolation_qualified=False,final_repeated=False)
new=execute_variant_agents(variant_specs=fresh,variants_run_dir=output,
    agent=cfg.agent.name,agent_model=cfg.agent.model,agent_profile=cfg.agent.profile,
    reasoning_effort=cfg.agent.reasoning_effort,
    timeout_seconds=cfg.execution.variant_timeout_seconds,
    evaluation_timeout_seconds=cfg.execution.evaluation_timeout_seconds,
    timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
    max_evals_per_variant=1,num_replications=cfg.execution.num_replications,
    sandbox_image=cfg.execution.sandbox_image or cfg.container.image,
    task_source_dir=task,evaluator_source_dir=task,
    evaluator_source_digest=managed_evaluator_tree_digest(task),containerize=True,
    container_image=cfg.container.image,container_network=cfg.container.network,
    container_auth_root=Path(cfg.container.auth_root),
    container_env_file=Path(cfg.container.env_file) if cfg.container.env_file else None,
    log_event_fn=event)
verify_retained()
specs=_load_variant_specs_from_disk(output)
_write_pipeline_summary(output,specs,retained+new)
summary_path=output/'pipeline_summary.json';summary=json.loads(summary_path.read_text())
summary['source_isolation']={'qualified_for_entire_campaign':False,
    'prior_exposed_authoring_views':p['completed'],
    'repaired_execution_views':[x.variant_id for x in new],
    'generation_and_selection_precede_repair':True,
    'interpretation':'Native execution validity is separate from complete source-isolation qualification. Retained earlier results remain exploratory.'}
summary_path.write_text(json.dumps(summary,indent=2)+'\n')
(output/'ISOLATION_STATUS.md').write_text(
    '# Source isolation status\n\nThe earlier authoring and generation views exposed sealed task source. '
    'Their complete access history is unavailable. The earlier four native executions are retained '
    'as exploratory observations, not fully isolated scientific evidence. The last three execution '
    'views were repaired and their unused one-evaluation budgets were preserved. '
    'The one-time sealed final result predates this exposure and was not repeated.\n')
event('checkpoint_execution_end',new_completed=sum(x.status=='completed' for x in new),
      new_total=len(new),entire_campaign_source_isolation_qualified=False)
if not all(x.rubric_passed for x in new):raise SystemExit(1)
# Call the original native report functions with the already validated receipts.
aggregate_variant_csvs(output,original_task_name=cfg.paths.task_name or 'task')
readme=generate_variants_readme(variant_specs=specs,output_dir=output,task_dir=task,
    primary_task_name=cfg.paths.task_name,
    timeout_seconds=cfg.execution.readme_timeout_seconds,
    timeout_grace_seconds=cfg.execution.variant_timeout_grace_seconds,
    agent=cfg.agent.name,agent_model=cfg.agent.model,agent_profile=cfg.agent.profile,
    reasoning_effort=cfg.agent.reasoning_effort,task_source_dir=task,containerize=True,
    container_image=cfg.container.image,container_network=cfg.container.network,
    container_auth_root=Path(cfg.container.auth_root),
    container_env_file=Path(cfg.container.env_file) if cfg.container.env_file else None)
if not readme or not Path(readme).is_file():raise ValueError('native README was not produced')
verify_retained()
event('checkpoint_recovery_end',return_code=0,entire_campaign_source_isolation_qualified=False)
raise SystemExit(0)
'''


def require_checkpoint(record: dict, state: dict, witness: dict, validation: dict) -> None:
    auth = record.get("budget_authorization", {})
    if (not auth.get("automatic_recovery_authorized")
            or auth.get("overall_time_limit_seconds", "missing") is not None):
        raise ValueError("Automatic uncapped recovery is not authorized")
    if state["running"] or state["exit_code"] != 143:
        raise ValueError("The intentionally stopped run must be fully stopped")
    if not validation.get("ready") or witness["errors"]:
        raise ValueError("Source gates, native receipts or sealed final integrity failed")
    if set(witness["completed"]) != COMPLETED or set(witness["unused"]) != set(REMAINING):
        raise ValueError("Only the recorded four-complete, three-unused checkpoint is eligible")
    if set(witness["variant_ids"]) != COMPLETED | set(REMAINING):
        raise ValueError("The original seven variant identities changed")
    if witness["tooling_sha256"] != record["runtime_snapshot_repair"]["new_tooling_sha256"]:
        raise ValueError("Frozen scientific harness changed")
    if record.get("variant_checkpoint_recoveries"):
        raise ValueError("Checkpoint recovery already exists; inspect it without resetting budgets")
    repair = record.get("variant_agent_isolation_finding", {}).get("repair", {})
    if not repair.get("real_container_canary_pass"):
        raise ValueError("An actual wrapper/container isolation check is required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    launcher = runpy.run_path(str(ROOT / "scripts/remote-agsci"))
    recovery = runpy.run_path(str(ROOT / "scripts/research/resume_medical_drafting.py"))
    continuation = runpy.run_path(str(ROOT / "scripts/research/continue_medical_experiment.py"))
    path = ROOT / "docs/research/medical-unlearning/execution.json"
    with (ROOT / "scratch/medical-recovery-submit.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(path.read_text());job = record["job_id"]
        base = launcher["REMOTE"];directory = base + "/jobs/" + job
        if (record["remote_results"] != base + "/results/" + job
                or not record["phase_output_overrides"]["ablations"].startswith(
                    record["remote_results"] + "/ablations_recovery_")):
            raise ValueError("The original job and ablation output scope must be preserved")
        def remote_python(code: str, argv: list[str], *, cwd: str = directory) -> dict:
            command = "cd " + shlex.quote(cwd) + "\n" + shlex.join(
                [base + "/.venv/bin/python", "-", *argv]) + " <<'PY'\n" + code + "\nPY\n"
            return json.loads(launcher["remote"](command, capture=True).stdout)
        def inspect() -> dict:
            return remote_python((ROOT / "scripts/research/medical_job_state.py").read_text(),
                                 [base, job], cwd=base + "/workspace")
        proof = json.loads(Path(record["sealed_final_result"]["local_evidence"]).read_text())
        final_hashes = {k:v for k,v in proof["artifact_sha256"].items() if k.startswith("final_evaluation/")}
        if len(final_hashes) != 5:
            raise ValueError("All five sealed final hashes are required")
        payload = {"results":record["remote_results"],
                   "output":record["phase_output_overrides"]["ablations"],
                   "final_hashes":final_hashes}
        repair = record.get("variant_agent_isolation_finding", {}).get("repair", {})
        for relative, key in (("src/scripts/variant_agent.sh", "wrapper_sha256"),
                              ("src/scripts/lib/variant_task_shadow.py", "helper_sha256")):
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != repair.get(key):
                raise ValueError("The code to deploy must match the actual container canary")
        validation = remote_python(continuation["VERIFY"], [
            record["qualified_source_task"]["path"], record["boundary_f1_repair"]["expected_metric_ast_sha256"]])
        witness = remote_python(CHECKPOINT, [json.dumps(payload)])
        state = inspect()
        try:
            require_checkpoint(record,state,witness,validation)
        except ValueError as error:
            if args.resume:
                raise
            print(json.dumps({"ready":False,"reason":str(error),"state":state},indent=2));return
        if not args.resume:
            print(json.dumps({"ready":True,"state":state,"completed":witness["completed"],
                              "unused":witness["unused"]},indent=2));return
        require_checkpoint(record,inspect(),remote_python(CHECKPOINT,[json.dumps(payload)]),validation)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        archive = directory + "/attempts/variant_checkpoint_" + stamp
        payload.update(archive=archive,witness=witness,completed=sorted(COMPLETED),remaining=REMAINING,
                       original_command=record["ablation_recoveries"][-1]["command"])
        patch = {"src/scripts/variant_agent.sh":(ROOT / "src/scripts/variant_agent.sh").read_text(),
                 "src/scripts/lib/variant_task_shadow.py":(ROOT / "src/scripts/lib/variant_task_shadow.py").read_text()}
        deploy = r'''
from pathlib import Path
import json,shutil,sys,hashlib
from src.investigation.task_runtime import _tooling_digest
root=Path.cwd();p=json.loads(sys.argv[1]);patch=json.loads(sys.argv[2]);worker=sys.argv[3]
archive=Path(p['archive']);archive.mkdir(mode=0o700)
for name in ('run.log','pid','exit-code'):shutil.copy2(root/name,archive/name)
for relative,text in patch.items():
    target=root/relative
    if target.exists():
        backup=archive/'code_before'/relative;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
assert _tooling_digest()==p['witness']['tooling_sha256']
(archive/'worker.py').write_text(worker);(archive/'payload.json').write_text(json.dumps(p,indent=2)+'\n')
(archive/'agent_audit').mkdir(mode=0o700)
print(json.dumps({'archive':str(archive),'tooling_sha256':_tooling_digest(),
                  'patch_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in patch.items()}}))
'''
        deployed = remote_python(deploy,[json.dumps(payload),json.dumps(patch),WORKER])
        command = [base + "/.venv/bin/python",archive + "/worker.py",archive + "/payload.json"]
        runner = recovery["runner_text"](launcher,directory,command)
        runner = runner.replace("umask 077\n", "umask 077\nexport AGENT_VARIANT_AUDIT_ROOT=" + shlex.quote(archive + "/agent_audit") + "\nexport VARIANT_EVAL_SANDBOX_GPUS=" + shlex.quote('"device=' + ','.join(launcher['CFG']['gpu_uuids']) + '"') + "\n")
        launcher["remote"]("printf %s " + shlex.quote(runner) + " > " + shlex.quote(archive + "/resume.sh") + "\n")
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        detail = {"attempt_id":stamp,"archive":archive,"started_at":now,"deployed":deployed,
                  "completed_retained":sorted(COMPLETED),"remaining":REMAINING,
                  "original_variant_budgets_preserved":True,"sealed_final_repeated":False}
        record.setdefault("variant_checkpoint_recoveries",[]).append(detail)
        record.update(updated_at=now,status="submitting_ablations_checkpoint_recovery",
                      phase="ablations",substage="recreating_only_unused_variant_views")
        record["variant_agent_audit"] = {"root":archive + "/agent_audit","private":True}
        recovery["atomic_record"](path,record)
        launcher["remote"]("cd " + shlex.quote(directory) + "\ntest \"$(cat exit-code)\" = 143\nmv exit-code " + shlex.quote(archive + "/stopped-exit-code") + "\nnohup bash " + shlex.quote(archive + "/resume.sh") + " > run.log 2>&1 < /dev/null &\necho $! > pid\n")
        record["status"] = "running_ablations_checkpoint_recovery_prior_isolation_unqualified"
        record["substage"] = "executing_three_unused_baseline_budgets"
        record["variant_agent_isolation_finding"]["status"] = "fixed_for_new_views_prior_artifacts_exposed"
        record["next_action"] = {"requires_user_decision":False,"action":
            "Verify real live seals and complete only three unused baseline budgets; preserve four earlier exposed-view observations as exploratory and the negative sealed final unchanged. Do not certify entire-campaign isolation or release paper."}
        recovery["atomic_record"](path,record)
        print(json.dumps(detail,indent=2))


if __name__ == "__main__":
    main()
