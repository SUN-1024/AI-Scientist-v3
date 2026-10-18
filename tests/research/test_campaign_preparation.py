"""Executable campaigns are repaired and reviewed before the workspace is sealed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.research import harness, novelty_review, workspace
from src.research.campaign_plan import (
    CAMPAIGN_PLAN_PATH, HOST_CONTEXT_PATH, PlanValidationError,
    compile_campaign_plan, load_campaign_plan,
)
from src.research.config import ResearchConfig, write_config
from src.research.research_agenda import load_seed_agenda, normalize_agenda
from tests.research.test_campaign_plan import card, context, policy
from tests.research import test_novelty_gate as fixtures

study = fixtures.study
install_provider = fixtures.install_provider


def proposed_agenda(agenda, ctx):
    return normalize_agenda({**agenda, "execution_plan": {
        "schema_version": 1,
        "roots": [card(index) for index in range(ctx["allocation"]["initial_ideas"])],
        "allocation": deepcopy(ctx["allocation"]),
    }})


def payload(prompt):
    return json.JSONDecoder().raw_decode(prompt.split("\n\nINPUT DATA:\n", 1)[1])[0]


def read(path):
    return json.loads(path.read_text())


def test_first_author_call_binds_host_context_and_old_proposal_then_repairs_budget(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    ctx = context(task)
    valid = proposed_agenda(agenda, ctx)
    invalid = deepcopy(valid)
    invalid["execution_plan"]["allocation"]["total_evaluations"] = 30
    seed = {"schema_version": 1, "agenda": invalid,
            "derived_from": {"workspace": "/previous/study", "proposal_sha256": "a" * 64},
            "provenance": {"prior_calls_are_reused": False}}
    (task / "research_seed_agenda.json").write_text(json.dumps(seed))
    calls = install_provider(monkeypatch, [invalid, valid, {**valid, "_reflection": {"done": True}}])
    output = tmp_path / "author"
    result = novelty_review.draft_research_agenda(task, cfg, output, host_context=ctx,
                                                  seed_agenda=load_seed_agenda(task))
    first = payload(calls[0]["messages"][0]["content"])
    assert first["host_context"] == ctx
    assert first["host_context"]["allocation"]["total_evaluations"] == 16
    assert first["host_context"]["seed"]["status"] == "not_evaluated"
    assert first["selected_candidate_packet"] is None
    assert first["seed_agenda"]["derived_from"] == seed["derived_from"]
    correction = calls[1]["messages"][0]["content"]
    assert "HOST PLAN VALIDATION ERRORS" in correction
    assert "Host allocation is 16; received 30" in correction
    assert result == valid
    assert read(output / "agenda_draft/assessment.json") == invalid
    trace = read(output / "agenda_reflection_trace.json")
    assert trace["history"][0]["plan_validation_errors"][0]["code"] == "allocation_mismatch"
    assert trace["history"][1]["plan_validation_errors"] == []
    assert read(output / "plan_validation.json")["status"] == "valid"
    assert read(output / "campaign_plan.json") == compile_campaign_plan(valid, ctx)
    receipts = {path: path.read_bytes() for path in output.glob("*/receipt.json")}
    replay_calls = install_provider(monkeypatch, [{"error": "must not call"}])
    assert novelty_review.draft_research_agenda(task, cfg, output, host_context=ctx,
        seed_agenda=load_seed_agenda(task), resume=True) == valid
    assert not replay_calls
    assert all(path.read_bytes() == data for path, data in receipts.items())


def test_invalid_final_plan_stops_before_independent_review(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    ctx = context(task)
    invalid = proposed_agenda(agenda, ctx)
    invalid["execution_plan"]["allocation"]["total_evaluations"] = 30
    calls = install_provider(monkeypatch, [{**invalid, "_reflection": {"done": True}}])
    output = tmp_path / "cycle"
    with pytest.raises(PlanValidationError, match="Host allocation is 16; received 30"):
        harness._novelty_cycle(task, ResearchConfig(literature=cfg), output, offline=False,
                               reviewer_settings=None, host_context=ctx)
    assert len(calls) == 3
    assert not list(output.glob("reviewer_*"))
    assert read(output / "plan_validation.json")["status"] == "invalid"
    assert read(output / "plan_validation.json")["execution_started"] is False
    assert read(output / "research_agenda.json")["execution_plan"]["allocation"]["total_evaluations"] == 30


def test_compiled_plan_and_context_review_then_freeze_as_one_transaction(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    # No search is requested. Local papers are prepared through the normal
    # workspace stage, while only the author/reviewer fixture can make calls.
    disk_cfg = ResearchConfig(literature=deepcopy(cfg))
    disk_cfg.literature.model_search.model = ""
    write_config(task / "research.yaml", disk_cfg)
    ctx = context(task)
    valid = proposed_agenda(agenda, ctx)
    calls = install_provider(monkeypatch, [valid, {**valid, "_reflection": {"done": True}}, review])
    location = tmp_path / "workspace"
    output = location / "novelty/initial"

    def after_literature(staged, frozen_cfg):
        actual = context(staged)
        # Use a test transport config only for author/reviewer fixtures.
        frozen_cfg.literature = cfg
        report = harness._novelty_cycle(staged, frozen_cfg, output, offline=False,
            reviewer_settings=None, host_context=actual, resume_calls=output.exists())
        approved = read(output / "research_agenda.json")
        return {name: json.dumps(value, ensure_ascii=False, indent=2) + "\n" for name, value in {
            "research_agenda.json": approved, str(HOST_CONTEXT_PATH): actual,
            str(CAMPAIGN_PLAN_PATH): compile_campaign_plan(approved, actual),
            "research_context/harness_novelty.json": report,
        }.items()}

    manifest = workspace.prepare(task, location, offline=False, overrides=[], after_literature=after_literature)
    assert len(calls) == 4
    plan = load_campaign_plan(location / "input_task", required=True)
    assert plan["allocation"]["total_evaluations"] == 16
    journal = read(location / "research_preparation.json")
    assert journal["state"] == "prepared"
    assert {str(HOST_CONTEXT_PATH), str(CAMPAIGN_PLAN_PATH), "research_agenda.json"} <= set(journal["generated_outputs"])
    for name in ("reviewer_1", "reviewer_2"):
        prompt = (output / name / "prompt.txt").read_text()
        data = json.loads(prompt.split("\n\nFROZEN INPUT DATA:\n")[1])
        assert data["host_context"] == ctx and data["compiled_plan"] == plan
        assert data["frozen_evidence"]["campaign_plan_sha256"] == plan["plan_sha256"]
    receipts = {path: path.read_bytes() for path in output.glob("*/receipt.json")}
    replay_calls = install_provider(monkeypatch, [{"error": "must not call"}])
    assert workspace.resume(location)[1] == manifest
    assert novelty_review.load_frozen_novelty_report(location / "input_task", output)["status"] == "supported"
    assert not replay_calls
    assert all(path.read_bytes() == data for path, data in receipts.items())


@pytest.mark.parametrize("changed", ["host_context.json", "campaign_plan.json"])
def test_changed_plan_artifacts_fail_before_any_paid_resume(study, tmp_path, monkeypatch, changed):
    task, cfg, agenda, review = study
    ctx = context(task)
    valid = proposed_agenda(agenda, ctx)
    plan = compile_campaign_plan(valid, ctx)
    output = tmp_path / "review"
    install_provider(monkeypatch, [review])
    novelty_review.review_novelty(task, valid, cfg, output, host_context=ctx, compiled_plan=plan)
    path = output / changed
    data = read(path)
    data["allocation"]["total_evaluations"] = 30
    path.write_text(json.dumps(data))
    calls = install_provider(monkeypatch, [{"error": "must not call"}])
    with pytest.raises(novelty_review.InvalidReviewResume):
        novelty_review.review_novelty(task, valid, cfg, output, host_context=ctx, compiled_plan=plan, resume=True)
    assert not calls


@pytest.mark.parametrize("review_started", [True, False])
def test_author_resume_rebuilds_missing_plan_only_before_independent_review(study, tmp_path, monkeypatch, review_started):
    task, cfg, agenda, review = study
    ctx = context(task)
    valid = proposed_agenda(agenda, ctx)
    output = tmp_path / "initial"
    install_provider(monkeypatch, [valid, {**valid, "_reflection": {"done": True}}, review])
    if review_started:
        harness._novelty_cycle(task, ResearchConfig(literature=cfg), output, offline=False,
                               reviewer_settings=None, host_context=ctx)
    else:
        novelty_review.draft_research_agenda(task, cfg, output, host_context=ctx)
    path = output / "campaign_plan.json"
    original = path.read_bytes()
    path.unlink()
    calls = install_provider(monkeypatch, [{"error": "must not call"}])
    if review_started:
        with pytest.raises(novelty_review.InvalidReviewResume, match="frozen campaign plan is missing"):
            harness._novelty_cycle(task, ResearchConfig(literature=cfg), output, offline=False,
                                   reviewer_settings=None, host_context=ctx, resume_calls=True)
        assert not path.exists()
    else:
        assert novelty_review.draft_research_agenda(task, cfg, output, host_context=ctx, resume=True) == valid
        assert path.read_bytes() == original
    assert not calls


def test_replay_checks_actual_review_prompt_context_not_only_rehashed_receipt(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    ctx = context(task)
    valid = proposed_agenda(agenda, ctx)
    plan = compile_campaign_plan(valid, ctx)
    output = tmp_path / "review"
    install_provider(monkeypatch, [review])
    novelty_review.review_novelty(task, valid, cfg, output, host_context=ctx, compiled_plan=plan)
    directory = output / "reviewer_1"
    path = directory / "prompt.txt"
    before, body = path.read_text().split("\n\nFROZEN INPUT DATA:\n", 1)
    data = json.loads(body)
    data.pop("compiled_plan")
    path.write_text(before + "\n\nFROZEN INPUT DATA:\n" + json.dumps(data))
    receipt_path = directory / "receipt.json"
    receipt = read(receipt_path)
    receipt["prompt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    records, errors = novelty_review.load_review_records(task, valid, output)
    assert len(records) == 1 and any("frozen campaign inputs" in error for error in errors)


@pytest.mark.parametrize("ready_scaffold", [True, False])
def test_harness_uses_final_resolved_budget_and_defers_briefs(study, tmp_path, monkeypatch, ready_scaffold):
    from src.research import cli
    from tests.program_testutils import make_program_dir

    task, cfg, agenda, review = study
    if ready_scaffold:
        make_program_dir(task / "program_0", content="def handle(x):\n    return x\n")
        for name in ("capture.py", "evaluate.py", "evaluate_final.py"):
            (task / name).write_text("# Test fixture. No evaluator is executed.\n")
        contract = read(Path(__file__).resolve().parents[2] / "examples/pathmnist_mil_portfolio/task_contract.json")
        contract["canonical_task"]["sha256"] = hashlib.sha256((task / "task.md").read_bytes()).hexdigest()
        (task / "task_contract.json").write_text(json.dumps(contract))
    write_config(task / "research.yaml", ResearchConfig(literature=cfg))
    actual_policy = policy()
    actual_policy.update(promote_ideas=6, max_valid_evals=32)
    valid = proposed_agenda(agenda, context(task, actual_policy)) if ready_scaffold else agenda
    calls = install_provider(monkeypatch, [valid, {**valid, "_reflection": {"done": True}}, review])
    monkeypatch.setattr(harness, "ensure_execution_ready", lambda *_: None)
    monkeypatch.setattr(harness, "_author_settings", lambda *_: None)
    monkeypatch.setattr(harness, "_reviewer_settings", lambda *_: None)
    monkeypatch.setattr(harness, "write_readiness", lambda *_: {"status": "fixture"})
    executed = []
    monkeypatch.setattr("src.pipeline.cli.main", lambda argv, **_: executed.append(argv) or 0)
    location = tmp_path / "harness"
    assert cli.main(["harness", str(task), "--workspace", str(location), "--quality-target", "development",
                     "execution.phases=investigation", "container.enabled=false",
                     "investigation.execution.workers=2", "investigation.portfolio.initial_ideas=6",
                     "investigation.portfolio.promote_ideas=6", "investigation.portfolio.tuning_trials_per_idea=4",
                     "investigation.portfolio.replication_evals=2", "investigation.portfolio.max_valid_evals=32"]) == 0
    assert len(calls) == 4 and len(executed) == 1
    first = payload(calls[0]["messages"][0]["content"])
    state = read(location / "harness_state.json")
    if ready_scaffold:
        assert first["host_context"]["allocation"]["total_evaluations"] == 32
        assert first["host_context"]["allocation"]["guaranteed_tuning_trials_per_root"] == 4
        assert first["host_context"]["workers"] == 2
        assert state["campaign_plan_version"] == 1
        assert load_campaign_plan(location / "input_task", required=True)
    else:
        assert "host_context" not in first
        assert "campaign_plan_version" not in state
        assert state["campaign_plan"]["status"] == "awaiting_draft"
        assert state["postdraft_campaign_plan_version"] == 1
        assert state["campaign_plan"]["execution_mode"] == "postdraft_plan"
        assert load_campaign_plan(location / "input_task") is None


def test_legacy_author_prompt_and_replay_keep_original_bytes(study, tmp_path, monkeypatch):
    from src.research.novelty import freeze_evidence
    task, cfg, agenda, _review = study
    output = tmp_path / "legacy"
    calls = install_provider(monkeypatch, [agenda, {**agenda, "_reflection": {"done": True}}])
    novelty_review.draft_research_agenda(task, cfg, output)
    expected = ((novelty_review.PROMPTS / "research_agenda.md").read_text()
                + "\n\nINPUT DATA:\n" + json.dumps({
                    "task": (task / "task.md").read_text(), "selected_candidate_research_basis": None,
                    "selected_candidate_packet": None, "frozen_evidence": freeze_evidence(task),
                    "literature": novelty_review.source_packet(task),
                }, ensure_ascii=False))
    assert calls[0]["messages"][0]["content"].encode() == expected.encode()
    before = {path: path.read_bytes() for path in output.glob("*/*") if path.is_file()}
    assert "HOST PLAN VALIDATION ERRORS" not in calls[1]["messages"][0]["content"]
    replay = install_provider(monkeypatch, [{"error": "must not call"}])
    novelty_review.draft_research_agenda(task, cfg, output, resume=True)
    assert not replay
    assert all(path.read_bytes() == content for path, content in before.items())
