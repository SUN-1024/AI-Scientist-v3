"""Journal a new postdraft plan without replacing frozen initial research inputs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from src.orchestrator.state import write_private_text
from src.pipeline.safe_fs import UnsafeGluePathError, open_directory
from src.pipeline.task_snapshot import (derive_task_snapshot, reseal_run_task, task_snapshot_is_complete,
                                        read_task_snapshot_name)
from src.pipeline.task_tree import task_tree_manifest_at
from .campaign_plan import compile_campaign_plan, load_campaign_plan
from .harness_state import read_json, sha256
from .novelty import content_hash, freeze_evidence
from .novelty_review import (draft_research_agenda, review_novelty, load_frozen_novelty_report,
                             _replay_call, campaign_binding)

MARKER = "postdraft_campaign_plan_version"
ROOT = Path("campaign/postdraft")
JOURNAL = ROOT / "transaction.json"
PLAN_TASK = ROOT / "planned_task"
DRAFT_TASK = ROOT / "drafted_task"
CYCLE = Path("novelty/postdraft")
UPDATES = ("research_agenda.json", "research_context/host_context.json",
           "research_context/campaign_plan.json", "research_context/harness_novelty.json")
STATUSES = {"draft_ready", "draft_frozen", "authoring", "plan_validated", "reviewing", "publishing", "frozen"}


def _path(workspace: Path, relative: Path | str) -> Path:
    path = workspace / relative
    for ancestor in (path, *path.parents):
        if ancestor == workspace.parent:
            break
        if ancestor.is_symlink():
            raise ValueError("Postdraft paths cannot contain symlinks")
    if not path.absolute().is_relative_to(workspace.absolute()):
        raise ValueError("Postdraft path escapes workspace")
    return path


def _dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def postdraft_enabled(workspace: Path) -> bool:
    """The creation journal, not a newly inferred configuration, opts a study in."""
    preparation = _path(workspace, "research_preparation.json")
    if not preparation.exists():
        return False
    saved = read_json(preparation).get("initial_harness_state", {})
    version = saved.get(MARKER) if isinstance(saved, dict) else None
    if version is not None and (type(version) is not int or version != 1):
        raise ValueError("Unsupported postdraft plan marker")
    state_path = _path(workspace, "harness_state.json")
    if state_path.exists() and read_json(state_path).get(MARKER) != version:
        raise ValueError("Postdraft marker differs from the creation journal")
    return version == 1


def _digest(path: Path) -> str:
    from .workspace import task_digest
    return task_digest(path)


def _complete(path: Path) -> bool:
    try:
        return task_snapshot_is_complete(path)
    except UnsafeGluePathError as exc:
        raise ValueError(f"Postdraft snapshot is invalid: {exc}") from exc


def _initial_binding(workspace: Path) -> dict[str, Any]:
    manifest_path = _path(workspace, "research_manifest.json")
    manifest = read_json(manifest_path)
    task = _path(workspace, "input_task")
    if _digest(task) != manifest.get("input_task_sha256"):
        raise ValueError("Original postdraft research input changed")
    files = ("input_task/task.md", "input_task/research_agenda.json",
             "input_task/research_context/catalog.json", "novelty/initial/novelty_report.json")
    return {"input_task_sha256": manifest["input_task_sha256"],
            "research_manifest_sha256": sha256(manifest_path),
            "initial_artifacts": {name: sha256(_path(workspace, name)) for name in files}}


def _records(path: Path) -> dict[str, Any]:
    with open_directory(path, role="postdraft task") as fd:
        return task_tree_manifest_at(fd, relative_dir=Path(),
            ignore_patterns=("/.agentic_task_snapshot.json", ".venv/", "/.agentic_runtime.json"))


def _verify_overlay(draft: Path, runtime: Path, updates: dict[str, str]) -> None:
    original, current = _records(draft), _records(runtime)
    for name, text in updates.items():
        generated = ("file", hashlib.sha256(text.encode()).hexdigest(), "0o600")
        found = current.pop(name, None)
        before = original.pop(name, None)
        if found is not None:
            found = tuple(found)
        if before is not None:
            before = tuple(before)
        if found not in (before, generated):
            raise ValueError(f"Postdraft publication output changed: {name}")
    if current != original:
        raise ValueError("Drafted task changed outside the postdraft declaration files")


def _read_journal(workspace: Path) -> dict[str, Any]:
    journal = read_json(_path(workspace, JOURNAL))
    if journal.get("schema_version") != 1 or journal.get("status") not in STATUSES:
        raise ValueError("Invalid postdraft transaction")
    if journal.get("initial_binding") != _initial_binding(workspace):
        raise ValueError("Postdraft original input binding changed")
    if journal["status"] == "draft_ready":
        if content_hash(_records(_path(workspace, "run/task"))) != journal.get("draft_basis_sha256"):
            raise ValueError("Successful draft changed before freezing")
        return journal
    draft = _path(workspace, DRAFT_TASK)
    if not _complete(draft) or _digest(draft) != journal.get("drafted_task_sha256"):
        raise ValueError("Frozen drafted task changed")
    if freeze_evidence(draft) != freeze_evidence(_path(workspace, "input_task")):
        raise ValueError("Drafting changed the frozen literature or methodology")
    if journal["status"] in {"publishing", "frozen"}:
        outputs = journal.get("outputs")
        if (not isinstance(outputs, dict) or set(outputs) != set(UPDATES)
                or any(not isinstance(value, str) or len(value.encode()) > 2 * 1024 * 1024
                       for value in outputs.values())):
            raise ValueError("Invalid postdraft publication outputs")
        rows = journal.get("review_artifacts")
        if not isinstance(rows, dict) or not rows:
            raise ValueError("Missing postdraft call artifacts")
        for name in rows:
            if (not isinstance(name, str) or ".." in Path(name).parts
                    or not Path(name).is_relative_to(CYCLE)):
                raise ValueError("Invalid postdraft review artifact path")
    return journal


def _verify_author(workspace: Path, journal: dict[str, Any], context: dict[str, Any],
                   agenda: dict[str, Any]) -> None:
    cycle = _path(workspace, CYCLE)
    seed = read_json(cycle / "research_seed_agenda.json")
    expected_seed = {"schema_version": 1,
        "agenda": read_json(_path(workspace, "input_task/research_agenda.json")),
        "derived_from": {**journal["initial_binding"], "drafted_task_sha256": journal["drafted_task_sha256"]}}
    from .research_agenda import normalize_seed_agenda, normalize_agenda
    if seed != normalize_seed_agenda(expected_seed):
        raise ValueError("Postdraft author seed differs from the initial agenda")
    frozen = {**freeze_evidence(_path(workspace, DRAFT_TASK)), **campaign_binding(context),
              "seed_agenda_sha256": content_hash(seed)}
    directories = sorted(path for path in cycle.glob("agenda_*") if path.is_dir())
    ordered = ["agenda_draft", "agenda_reflection_2", "agenda_reflection_3"]
    if not directories or [path.name for path in directories] != ordered[:len(directories)]:
        raise ValueError("Postdraft author rounds are incomplete or out of order")
    final = None
    for directory in directories:
        final = _replay_call(directory, read_json(directory / "receipt.json"), "research_agenda", frozen)
    assert final is not None
    final.pop("_reflection", None)
    if final != normalize_agenda(agenda):
        raise ValueError("Postdraft agenda differs from the final author response")


def verify_postdraft_campaign(workspace: Path, *, require_frozen: bool = False,
                             runtime_task: Path | None = None) -> dict[str, Any] | None:
    """Validate the complete derivation chain; absence is legacy only without opt-in."""
    workspace = Path(workspace)
    if not postdraft_enabled(workspace):
        if _path(workspace, JOURNAL).exists():
            raise ValueError("Postdraft journal has no creation marker")
        return None
    if not _path(workspace, JOURNAL).exists():
        if require_frozen:
            raise ValueError("Postdraft campaign awaits a qualified drafted task")
        return {"status": "awaiting_draft", "plan_task": PLAN_TASK.as_posix(), "provenance": {}}
    journal = _read_journal(workspace)
    provenance = {**journal["initial_binding"],
        "drafted_task": {"path": DRAFT_TASK.as_posix(), "sha256": journal.get("drafted_task_sha256")},
        "transaction": {"path": JOURNAL.as_posix(), "sha256": sha256(_path(workspace, JOURNAL))}}
    result = {"status": journal["status"], "plan_task": PLAN_TASK.as_posix(), "provenance": provenance}
    if journal["status"] != "frozen":
        if require_frozen:
            raise ValueError("Postdraft campaign has not completed publication")
        return result
    planned = _path(workspace, PLAN_TASK)
    if not _complete(planned) or _digest(planned) != journal.get("planned_task_sha256"):
        raise ValueError("Frozen postdraft planned task changed")
    if journal.get("runtime_task_sha256") != journal["planned_task_sha256"]:
        raise ValueError("Postdraft publication snapshot hash differs from the frozen planned task")
    plan = load_campaign_plan(planned, required=True)
    assert plan is not None
    if plan["plan_sha256"] != journal.get("plan_sha256"):
        raise ValueError("Postdraft plan differs from its transaction")
    report = load_frozen_novelty_report(planned, _path(workspace, CYCLE))
    if len(report.get("model_judgments", [])) < journal["reviewer_count"]:
        raise ValueError("Postdraft independent reviews are incomplete")
    for name, digest in journal["review_artifacts"].items():
        if sha256(_path(workspace, name)) != digest:
            raise ValueError("Postdraft author or reviewer receipt changed")
    updates = journal["outputs"]
    _verify_author(workspace, journal, read_json(planned / "research_context/host_context.json"),
                   read_json(planned / "research_agenda.json"))
    _verify_overlay(_path(workspace, DRAFT_TASK), planned, updates)
    for name, text in updates.items():
        if (_path(workspace, PLAN_TASK) / name).read_text() != text:
            raise ValueError("Frozen postdraft declaration differs from its journal")
    if runtime_task is not None:
        if load_campaign_plan(runtime_task, required=True) != plan:
            raise ValueError("Runtime and frozen postdraft plans differ")
        if freeze_evidence(runtime_task) != freeze_evidence(planned):
            raise ValueError("Runtime postdraft literature or methodology changed")
        for relative, text in updates.items():
            path = runtime_task / relative
            if path.is_symlink() or not path.is_file() or path.read_text() != text:
                raise ValueError("Runtime postdraft published declaration changed: " + relative)
    provenance.update(planned_task={"path": PLAN_TASK.as_posix(), "sha256": journal["planned_task_sha256"]},
                      review_artifacts=journal["review_artifacts"],
                      publication={"task": "run/task", "snapshot_sha256": journal["runtime_task_sha256"]})
    result.update({key: plan[key] for key in ("plan_sha256", "host_context_sha256", "agenda_sha256")})
    return result


def resolve_campaign_plan_task(workspace: Path, *, require_frozen: bool = True) -> Path:
    result = verify_postdraft_campaign(workspace, require_frozen=require_frozen)
    return _path(workspace, result["plan_task"] if result is not None else "input_task")


def _review_artifacts(workspace: Path) -> dict[str, str]:
    return {path.relative_to(workspace).as_posix(): sha256(path)
            for path in sorted(_path(workspace, CYCLE).rglob("*")) if path.is_file()}


def _call_settings(cfg: Any, pipeline_cfg: Any) -> dict[str, Any]:
    from .harness import _author_settings, _reviewer_settings
    return {"author": _author_settings(pipeline_cfg), "reviewer": _reviewer_settings(pipeline_cfg),
            "literature": asdict(cfg.literature), "reviewer_count": cfg.harness.novelty_reviewers}


def record_postdraft_success(workspace: Path, paths: Any, cfg: Any, pipeline_cfg: Any) -> bool:
    """Persist the successful child boundary before cleanup or snapshot copying."""
    if not postdraft_enabled(workspace):
        return False
    journal_path = _path(workspace, JOURNAL)
    if journal_path.exists():
        _read_journal(workspace)
        return True
    journal = {"schema_version": 1, "status": "draft_ready", "initial_binding": _initial_binding(workspace),
        "draft_basis_sha256": content_hash(_records(paths.task_dir)),
        "configuration_sha256": content_hash(_call_settings(cfg, pipeline_cfg)),
        "reviewer_count": cfg.harness.novelty_reviewers}
    _dump(journal_path, journal)
    return True


def _qualified_published_draft(task: Path) -> bool:
    """Recover the child-to-host crash gap from fully validated child evidence."""
    receipt = task / ".agentic_task_qualification.json"
    publication = task / ".agentic_task_publication.json"
    if not any(path.exists() or path.is_symlink() for path in (receipt, publication)):
        return False
    from src.pipeline.cli import _validated_drafted_scaffold
    from src.task_drafting.publication import validate_published_task_plan
    from src.task_drafting.qualification import load_task_qualification_receipt
    from src.investigation.evaluator_tree import managed_evaluator_tree_digest
    from src.investigation.task_runtime import _requirements_digest

    if not _validated_drafted_scaffold(task):
        raise ValueError("Published draft has no current passing alignment receipt")
    digest = validate_published_task_plan(task)
    qualified = load_task_qualification_receipt(receipt, expected_publication_plan_sha256=digest)
    if (qualified.evaluator_tree_sha256 != managed_evaluator_tree_digest(task)
            or qualified.requirements_sha256 != _requirements_digest(task / "requirements.txt")):
        raise ValueError("Published draft qualification no longer matches its runtime inputs")
    return True


def ensure_postdraft_campaign_plan(workspace: Path, paths: Any, cfg: Any, pipeline_cfg: Any,
                                  *, recovery: bool = False) -> bool:
    """Finish a draft once; recovery returns False only before drafting succeeded."""
    if not postdraft_enabled(workspace):
        return False
    journal_path = _path(workspace, JOURNAL)
    if recovery and not journal_path.exists():
        if not _qualified_published_draft(paths.task_dir):
            return False
        record_postdraft_success(workspace, paths, cfg, pipeline_cfg)
    from .harness import _resolved_campaign_context

    settings = _call_settings(cfg, pipeline_cfg)
    configuration_sha256 = content_hash(settings)
    if journal_path.exists():
        journal = _read_journal(workspace)
        if journal["configuration_sha256"] != configuration_sha256:
            raise ValueError("Postdraft author or reviewer configuration changed")
    else:
        record_postdraft_success(workspace, paths, cfg, pipeline_cfg)
        journal = _read_journal(workspace)
    if journal["status"] == "draft_ready":
        from src.investigation.task_runtime import discard_managed_task_runtime
        discard_managed_task_runtime(paths.task_dir)
        draft = _path(workspace, DRAFT_TASK)
        # A published orphan may follow a crash before the first journal write.
        # Reuse only an identical source, never silently replace the snapshot.
        if not draft.exists():
            derive_task_snapshot(paths.task_dir, draft,
                                 task_name=read_task_snapshot_name(paths.task_dir) or paths.task_name)
        if not _complete(draft) or _records(draft) != _records(paths.task_dir):
            raise ValueError("Postdraft drafted snapshot does not match the successful draft")
        if freeze_evidence(draft) != freeze_evidence(_path(workspace, "input_task")):
            raise ValueError("Drafting changed frozen source evidence")
        journal.update(status="draft_frozen", drafted_task_sha256=_digest(draft))
        _dump(journal_path, journal)
    draft = _path(workspace, DRAFT_TASK)
    context = _resolved_campaign_context(draft, pipeline_cfg)
    if journal.get("host_context_sha256", context["host_context_sha256"]) != context["host_context_sha256"]:
        raise ValueError("Postdraft resolved execution budget changed")
    if journal["status"] == "frozen":
        verify_postdraft_campaign(workspace, require_frozen=True, runtime_task=paths.task_dir)
        return True
    _verify_overlay(draft, paths.task_dir, journal.get("outputs", {}))
    cycle = _path(workspace, CYCLE)
    if journal["status"] != "publishing":
        seed = {"schema_version": 1, "agenda": read_json(_path(workspace, "input_task/research_agenda.json")),
                "derived_from": {**journal["initial_binding"], "drafted_task_sha256": journal["drafted_task_sha256"]}}
        journal.update(status="authoring", host_context_sha256=context["host_context_sha256"])
        _dump(journal_path, journal)
        agenda = draft_research_agenda(draft, cfg.literature, cycle, reviewer_settings=settings["author"],
            host_context=context, seed_agenda=seed, resume=cycle.exists())
        plan = compile_campaign_plan(agenda, context)
        journal.update(status="plan_validated", plan_sha256=plan["plan_sha256"])
        _dump(journal_path, journal)
        journal["status"] = "reviewing"
        _dump(journal_path, journal)
        report = review_novelty(draft, agenda, cfg.literature, cycle,
            reviewer_settings=settings["reviewer"], reviewer_count=cfg.harness.novelty_reviewers,
            host_context=context, compiled_plan=plan, resume=True)
        if len(report.get("model_judgments", [])) < cfg.harness.novelty_reviewers:
            raise ValueError("Postdraft independent novelty review is incomplete")
        values = dict(zip(UPDATES, (agenda, context, plan, report), strict=True))
        journal.update(status="publishing", outputs={name: json.dumps(value, ensure_ascii=False, indent=2) + "\n"
                                                     for name, value in values.items()},
                       review_artifacts=_review_artifacts(workspace))
        _dump(journal_path, journal)
    planned = _path(workspace, PLAN_TASK)
    if not planned.exists():
        derive_task_snapshot(draft, planned, task_name=read_task_snapshot_name(draft) or paths.task_name,
                             updates=journal["outputs"])
    if not _complete(planned):
        raise ValueError("Postdraft planned snapshot publication is incomplete")
    published_plan = load_campaign_plan(planned, required=True)
    if published_plan is None or published_plan["plan_sha256"] != journal["plan_sha256"]:
        raise ValueError("Postdraft publication plan changed")
    report = load_frozen_novelty_report(planned, cycle)
    if len(report.get("model_judgments", [])) < journal["reviewer_count"]:
        raise ValueError("Postdraft publication reviews are incomplete")
    _verify_author(workspace, journal, context, read_json(planned / "research_agenda.json"))
    for name, digest in journal["review_artifacts"].items():
        if sha256(_path(workspace, name)) != digest:
            raise ValueError("Postdraft paid-call artifact changed before publication")
    _verify_overlay(draft, planned, journal["outputs"])
    _verify_overlay(draft, paths.task_dir, journal["outputs"])
    for relative, text in journal["outputs"].items():
        destination = paths.task_dir / relative
        if destination.exists() and destination.read_text() == text:
            continue
        write_private_text(destination, text)
    reseal_run_task(paths)
    if not _complete(paths.task_dir):
        raise ValueError("Postdraft runtime snapshot did not become ready")
    journal.update(status="frozen", planned_task_sha256=_digest(planned), runtime_task_sha256=_digest(paths.task_dir))
    _dump(journal_path, journal)
    verify_postdraft_campaign(workspace, require_frozen=True, runtime_task=paths.task_dir)
    return True
