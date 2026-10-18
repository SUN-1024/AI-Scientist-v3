"""Postdraft completion is resumable and never rewrites the initial study."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pipeline.task_snapshot import derive_task_snapshot, task_snapshot_is_complete
from src.research import harness, postdraft_plan as post
from src.research.campaign_plan import build_host_context, load_campaign_plan
from src.research.config import ResearchConfig, write_config
from src.research.harness_state import sha256
from src.research.novelty_review import review_novelty
from src.research.workspace import prepare, resume
from tests.research import test_novelty_gate as fixtures
from tests.research.test_campaign_plan import agenda as planned_agenda, policy

study = fixtures.study


def make_postdraft(study, tmp_path, monkeypatch):
    source, literature, initial, review = study
    (source / "program_0/src").mkdir(parents=True)
    (source / "program_0/src/program.py").write_text("def run():\n    return 1\n")
    cfg = ResearchConfig(literature=literature)
    write_config(source / "research.yaml", cfg)
    workspace = tmp_path / "managed"
    fixtures.install_provider(monkeypatch, [review])

    def initial_review(task, _cfg):
        report = review_novelty(task, initial, literature, workspace / "novelty/initial")
        return {"research_agenda.json": json.dumps(initial),
                "research_context/harness_novelty.json": json.dumps(report)}

    marker = {post.MARKER: 1}
    manifest = prepare(source, workspace, offline=True, overrides=[], after_literature=initial_review,
                       initial_state=marker)
    post._dump(workspace / "harness_state.json", {**marker, "schema_version": 1,
                "input_task_sha256": manifest["input_task_sha256"]})
    runtime = workspace / "run/task"
    derive_task_snapshot(workspace / "input_task", runtime, task_name="test")
    paths = SimpleNamespace(task_dir=runtime, task_name="test")
    context_fn = lambda task, _cfg: build_host_context(task, policy(), 2, {"test": 1})
    monkeypatch.setattr(harness, "_resolved_campaign_context", context_fn)
    monkeypatch.setattr(harness, "_author_settings", lambda _: None)
    monkeypatch.setattr(harness, "_reviewer_settings", lambda _: None)
    context = context_fn(runtime, None)
    agenda = {**initial, **planned_agenda(context)}
    calls = fixtures.install_provider(monkeypatch, [agenda, {**agenda, "_reflection": {"done": True}}, review, review])
    return workspace, paths, cfg, SimpleNamespace(), calls


def test_full_postdraft_chain_and_completed_recovery_replay(study, tmp_path, monkeypatch):
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    before = {str(path): sha256(path) for root in (workspace / "input_task", workspace / "novelty/initial")
              for path in root.rglob("*") if path.is_file()}
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True) is False
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    assert len(calls) == 4
    evidence = post.verify_postdraft_campaign(workspace, require_frozen=True, runtime_task=paths.task_dir)
    assert evidence["status"] == "frozen"
    assert post.resolve_campaign_plan_task(workspace) == workspace / post.PLAN_TASK
    assert load_campaign_plan(paths.task_dir) == load_campaign_plan(workspace / post.PLAN_TASK)
    assert task_snapshot_is_complete(paths.task_dir)
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True)
    assert len(calls) == 4
    assert before == {name: sha256(Path(name)) for name in before}
    resume(workspace)


def test_publish_interruption_reuses_all_completed_calls(study, tmp_path, monkeypatch):
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    original = post.write_private_text
    def interrupt(path, text):
        if path == paths.task_dir / "research_context/campaign_plan.json":
            raise OSError("interrupted publication")
        return original(path, text)
    monkeypatch.setattr(post, "write_private_text", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    assert post.verify_postdraft_campaign(workspace)["status"] == "publishing"
    assert len(calls) == 4
    monkeypatch.setattr(post, "write_private_text", original)
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True)
    assert len(calls) == 4


def test_draft_ready_interruption_recovery_does_not_lose_success(study, tmp_path, monkeypatch):
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    derive = post.derive_task_snapshot
    monkeypatch.setattr(post, "derive_task_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("copy interrupted")))
    with pytest.raises(OSError, match="copy interrupted"):
        post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    assert post.verify_postdraft_campaign(workspace)["status"] == "draft_ready"
    assert not calls
    monkeypatch.setattr(post, "derive_task_snapshot", derive)
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True)
    assert len(calls) == 4


def test_partial_review_reuses_completed_author_and_review_calls(study, tmp_path, monkeypatch):
    workspace, paths, cfg, pipeline, _ = make_postdraft(study, tmp_path, monkeypatch)
    _source, _literature, initial, review = study
    context = harness._resolved_campaign_context(paths.task_dir, pipeline)
    agenda = {**initial, **planned_agenda(context)}
    calls = fixtures.install_provider(monkeypatch, [agenda, {**agenda, "_reflection": {"done": True}}, review,
                                                   {"error": "interrupted reviewer"}])
    with pytest.raises(ValueError, match="incomplete"):
        post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    assert len(calls) == 4
    first_receipt = (workspace / post.CYCLE / "reviewer_1/receipt.json").read_bytes()
    retries = fixtures.install_provider(monkeypatch, [review])
    assert post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True)
    assert len(retries) == 1
    assert (workspace / post.CYCLE / "reviewer_1/receipt.json").read_bytes() == first_receipt


def test_pending_runtime_tamper_fails_before_new_paid_calls(study, tmp_path, monkeypatch):
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    draft = post.draft_research_agenda
    monkeypatch.setattr(post, "draft_research_agenda", lambda *a, **k: (_ for _ in ()).throw(ValueError("pause author")))
    with pytest.raises(ValueError, match="pause author"):
        post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    (paths.task_dir / "program_0/src/program.py").write_text("changed source")
    monkeypatch.setattr(post, "draft_research_agenda", draft)
    with pytest.raises(ValueError, match="outside"):
        post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True)
    assert not calls


@pytest.mark.parametrize("target", ["input_task/task.md", "campaign/postdraft/drafted_task/program_0/src/program.py",
                                   "campaign/postdraft/planned_task/research_agenda.json",
                                   "novelty/postdraft/reviewer_1/assessment.json"])
def test_frozen_chain_tamper_fails_without_new_calls(study, tmp_path, monkeypatch, target):
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    path = workspace / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline, recovery=True)
    assert len(calls) == 4


def test_legacy_resolver_has_no_side_effects(tmp_path):
    assert post.verify_postdraft_campaign(tmp_path) is None
    assert post.resolve_campaign_plan_task(tmp_path) == tmp_path / "input_task"
    assert list(tmp_path.iterdir()) == []


def test_marker_cannot_be_added_only_to_live_state(study, tmp_path, monkeypatch):
    workspace, _paths, _cfg, _pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    journal = json.loads((workspace / "research_preparation.json").read_text())
    journal["initial_harness_state"] = None
    post._dump(workspace / "research_preparation.json", journal)
    with pytest.raises(ValueError, match="marker"):
        post.verify_postdraft_campaign(workspace)
    assert not calls


@pytest.mark.parametrize("relative", ["research_context/harness_novelty.json", "research_context/catalog.json",
                                     "research_context/methodology.json"])
def test_runtime_research_evidence_cannot_change_after_publication(study, tmp_path, monkeypatch, relative):
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    post.ensure_postdraft_campaign_plan(workspace, paths, cfg, pipeline)
    target = paths.task_dir / relative
    payload = json.loads(target.read_text())
    if relative.endswith("harness_novelty.json"):
        payload["status"] = "supported" if payload["status"] != "supported" else "needs_evidence"
    elif relative.endswith("catalog.json"):
        payload["sources"][0]["title"] += " changed"
    else:
        payload["profile"] = "medical"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        post.verify_postdraft_campaign(workspace, require_frozen=True, runtime_task=paths.task_dir)
    assert len(calls) == 4


def test_success_journal_survives_cleanup_crash_and_pipeline_resume_skips_draft(study, tmp_path, monkeypatch):
    from src.pipeline import cli as pipeline_cli
    from tests.pipeline.test_pipeline_cli_dryrun import _mock_successful_investigation
    workspace, paths, cfg, pipeline, calls = make_postdraft(study, tmp_path, monkeypatch)
    monkeypatch.setenv("AGENTIC_TEMP_ROOT", str(tmp_path.resolve()))
    monkeypatch.setattr(pipeline_cli, "_pin_evaluation_images", lambda *a: None)
    monkeypatch.setattr(pipeline_cli, "_verify_image_leases", lambda *a: None)
    _mock_successful_investigation(monkeypatch)
    commands = []
    monkeypatch.setattr(pipeline_cli, "run_cmd", lambda command, **kwargs: commands.append(command) or 0)
    cleanup = pipeline_cli.discard_managed_task_runtime
    monkeypatch.setattr(pipeline_cli, "discard_managed_task_runtime", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    def finalizer(action, resolved, actual_paths):
        if action == "draft_complete":
            return post.record_postdraft_success(workspace, actual_paths, cfg, pipeline)
        if action == "verify":
            post.verify_postdraft_campaign(workspace, require_frozen=True, runtime_task=actual_paths.task_dir)
            return True
        return post.ensure_postdraft_campaign_plan(workspace, actual_paths, cfg, pipeline, recovery=action == "recover")
    argv = [f"task.dir={workspace / 'input_task'}", f"paths.run_dir={workspace / 'run'}", "run_id=test",
            "execution.phases=task_drafting,investigation", "execution.fail_fast=false"]
    assert pipeline_cli.main(argv, task_finalizer=finalizer) == 130
    assert len(commands) == 1 and not calls
    assert post.verify_postdraft_campaign(workspace)["status"] == "draft_ready"
    monkeypatch.setattr(pipeline_cli, "discard_managed_task_runtime", cleanup)
    assert pipeline_cli.main(argv, task_finalizer=finalizer) == 0
    assert len(commands) == 2
    assert sum("execution.task_only=true" in command for command in commands) == 1
    assert len(calls) == 4


@pytest.mark.skipif(not Path('/proc/self/fd').exists(), reason="Published qualification fixture requires Linux")
def test_child_receipts_can_recover_before_first_host_journal(tmp_path):
    from tests.task_drafting.test_task_drafting_publication import _make_fx
    from src.investigation.evaluator_tree import managed_evaluator_tree_digest
    from src.investigation.task_runtime import _requirements_digest
    fx = _make_fx(tmp_path / "published")
    fx.publish(qualification_receipt_path=fx.write_receipt(fx.plan_sha256()))
    receipt = fx.task_dir / ".agentic_task_qualification.json"
    payload = json.loads(receipt.read_text())
    payload.update(evaluator_tree_sha256=managed_evaluator_tree_digest(fx.task_dir),
                   requirements_sha256=_requirements_digest(fx.task_dir / "requirements.txt"))
    post._dump(receipt, payload)
    assert post._qualified_published_draft(fx.task_dir)
    payload["publication_plan_sha256"] = "0" * 64
    post._dump(receipt, payload)
    with pytest.raises(ValueError):
        post._qualified_published_draft(fx.task_dir)
