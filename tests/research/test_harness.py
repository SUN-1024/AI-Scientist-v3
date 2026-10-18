from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.research import cli, harness
from src.research.harness_state import HarnessState, verify_stage_records
from src.research.paper_readiness import assess_readiness
from src.research.config import ResearchConfig, load_config
from tests.research.test_unified_research import make_task


@pytest.mark.parametrize("author_model,review_model", [
    ("deepseek/deepseek-flash", "codex/gpt-6.1-sol"),
    ("codex/gpt-6.1-sol", "deepseek/deepseek-flash"),
])
def test_agenda_author_and_independent_review_use_separate_config(tmp_path, monkeypatch, author_model, review_model):
    from src.research import novelty_review, research_agenda
    cfg = SimpleNamespace(
        agent=SimpleNamespace(name="codex", model=author_model, profile="author-profile", reasoning_effort="low"),
        review=SimpleNamespace(backend="agent", agent_name=None, agent_model=review_model,
                               agent_profile="review-profile", agent_reasoning_effort="high", timeout_seconds=60),
    )
    author = harness._author_settings(cfg)
    reviewer = harness._reviewer_settings(cfg)
    assert author["model"] == author_model and author["profile"] == "author-profile"
    assert reviewer["model"] == review_model and reviewer["profile"] == "review-profile"
    assert author["reasoning_effort"] == "low" and reviewer["reasoning_effort"] == "high"
    monkeypatch.setattr(research_agenda, "load_agenda", lambda _: None)
    draft = Mock(return_value={"intent": "exploratory"})
    review = Mock(return_value={"status": "needs_evidence"})
    monkeypatch.setattr(novelty_review, "draft_research_agenda", draft)
    monkeypatch.setattr(novelty_review, "review_novelty", review)
    harness._novelty_cycle(tmp_path, ResearchConfig(), tmp_path / "cycle", offline=False,
                           author_settings=author, reviewer_settings=reviewer)
    assert draft.call_args.kwargs["reviewer_settings"] == author
    assert review.call_args.kwargs["reviewer_settings"] == reviewer
    cfg.review.backend = "openai"
    assert harness._reviewer_settings(cfg) is None
    assert harness._author_settings(cfg) == author


@pytest.mark.parametrize("legacy_adaptive", [False, True])
def test_harness_dry_run_selects_complete_controlled_pipeline_without_effects(tmp_path, monkeypatch, capsys, legacy_adaptive):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "prepare", Mock(side_effect=AssertionError("must not prepare")))
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock(side_effect=AssertionError("must not probe runtime")))
    overrides = (["investigation.portfolio.enabled=false", "investigation.adaptive_investigation.enabled=true",
                  "investigation.execution.workers=1"] if legacy_adaptive else [])
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--dry-run", *overrides]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["requested_phases"] == ["task_drafting", "investigation", "ablations", "final_evaluation", "posthoc", "paper", "review", "revision"]
    from src.pipeline.cli import load_pipeline_config
    from src.pipeline.paths import derive_paths
    from src.pipeline.phases import build_investigation_cmd
    cfg = load_pipeline_config(preview["pipeline_argv"])
    command = build_investigation_cmd(cfg, derive_paths(cfg))
    assert ("adaptive_investigation.enabled=true" in command) is legacy_adaptive
    assert ("portfolio.enabled=true" in command) is not legacy_adaptive
    assert cfg.review.backend == "agent"
    assert not workspace.exists()
    assert not (task / "research_context").exists()


def test_high_impact_still_requires_a_controlled_investigation(tmp_path, capsys):
    task = make_task(tmp_path / "task")
    assert cli.main(["harness", str(task), "--dry-run", "investigation.portfolio.enabled=false",
                     "investigation.adaptive_investigation.enabled=false"]) == 2
    assert "requires portfolio.enabled=true or adaptive_investigation.enabled=true" in capsys.readouterr().err


def test_high_impact_cannot_silently_skip_required_phases(tmp_path, capsys):
    task = make_task(tmp_path / "task")
    assert cli.main(["harness", str(task), "--dry-run", "execution.phases=investigation,paper"]) == 2
    assert "requires all research phases" in capsys.readouterr().err


def test_failure_keeps_lifecycle_and_separate_readiness(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())
    def failed_pipeline(argv, *, event_callback):
        event_callback("phase_start", phase_name="investigation")
        raise RuntimeError("provider transport failed")
    monkeypatch.setattr("src.pipeline.cli.main", failed_pipeline)
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--offline",
                     "--quality-target", "development", "execution.phases=investigation,paper",
                     "container.enabled=false"]) == 1
    state = json.loads((workspace / "harness_state.json").read_text())
    readiness = json.loads((workspace / "paper_readiness.json").read_text())
    assert state["stages"]["investigation"]["status"] == "failed"
    assert state["stages"]["paper"]["status"] == "pending"
    assert "transport failed" in state["error"]
    assert readiness["submission_ready"] is False
    assert readiness["checks"]["artifact"]["status"] == "missing_or_invalid"


def _run_completed_fixture(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())
    calls = []
    def completed(argv, *, event_callback):
        calls.append(argv)
        root = workspace / "run/investigation"
        root.mkdir(parents=True)
        (root / "fixture.json").write_text('{"measurement": 1}')
        event_callback("phase_start", phase_name="investigation")
        event_callback("phase_end", phase_name="investigation", status="completed", return_code=0)
        return 0
    monkeypatch.setattr("src.pipeline.cli.main", completed)
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--offline",
                     "--quality-target", "development", "execution.phases=investigation",
                     "container.enabled=false"]) == 0
    return task, workspace, calls


def test_resume_skips_unchanged_completed_stage_but_does_not_call_it_scientifically_ready(tmp_path, monkeypatch):
    task, workspace, calls = _run_completed_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(harness, "prepare", Mock(side_effect=AssertionError("no repeat literature")))
    assert cli.main(["harness", "--resume", str(workspace)]) == 0
    assert len(calls) == 1
    report = json.loads((workspace / "paper_readiness.json").read_text())
    assert report["status"] == "evidence_gaps"
    assert report["checks"]["experiment"]["status"] == "missing"
    assert report["submission_ready"] is False


def test_resume_rejects_modified_stage_and_frozen_configuration(tmp_path, monkeypatch, capsys):
    task, workspace, calls = _run_completed_fixture(tmp_path, monkeypatch)
    assert cli.main(["harness", "--resume", str(workspace), "agent.model=changed"]) == 2
    assert "frozen harness setting" in capsys.readouterr().err
    (workspace / "run/investigation/fixture.json").write_text('{"measurement": 1000}')
    assert cli.main(["harness", "--resume", str(workspace)]) == 2
    assert "evidence changed" in capsys.readouterr().err
    assert len(calls) == 1


def test_receipts_bind_bytes_and_paper_filename_alone_is_not_readiness(tmp_path):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    paper = workspace / "run/paper"
    paper.mkdir(parents=True)
    (paper / "paper.pdf").write_bytes(b"%PDF fixture")
    state = HarnessState(workspace, phases=["paper"], quality_target="high_impact", input_sha256="abc")
    state.observe("phase_end", phase_name="paper", status="completed", return_code=0)
    assert verify_stage_records(workspace, state.payload["stages"]["paper"])
    report = assess_readiness(workspace, load_config(task), state.payload)
    assert report["checks"]["artifact"]["status"] == "missing_or_invalid"
    assert not report["ready_for_human_review"]
    (paper / "paper.pdf").write_bytes(b"changed")
    assert not verify_stage_records(workspace, state.payload["stages"]["paper"])


@pytest.mark.parametrize("initial", ["", '{"event":"investigation"}\n'])
def test_completed_investigation_accepts_only_append_only_workflow_memory(tmp_path, initial):
    from src.research.harness_state import stage_records

    root = tmp_path / "run/investigation"
    root.mkdir(parents=True)
    memory = root / "memory.jsonl"
    memory.write_text(initial)
    code = root / "program.py"
    code.write_text("frozen program\n")
    stage = {"status": "completed", "artifacts": stage_records(tmp_path, "investigation")}
    memory.write_text(initial + '{"event":"paper"}\n')
    assert verify_stage_records(tmp_path, stage)
    code.write_text("changed program\n")
    assert not verify_stage_records(tmp_path, stage)
    code.write_text("frozen program\n")
    memory.write_text('{"event":"replacement"}\n')
    assert verify_stage_records(tmp_path, stage) is (initial == "")
    memory.write_text(initial + 'invalid appended record\n')
    assert not verify_stage_records(tmp_path, stage)
    memory.write_text(initial + '[]\n')
    assert not verify_stage_records(tmp_path, stage)


def test_other_jsonl_artifacts_remain_byte_frozen(tmp_path):
    from src.research.harness_state import stage_records

    root = tmp_path / "run/investigation"
    root.mkdir(parents=True)
    log = root / "evaluation_events.jsonl"
    log.write_text('{"score":1}\n')
    stage = {"status": "completed", "artifacts": stage_records(tmp_path, "investigation")}
    log.write_text(log.read_text() + '{"score":2}\n')
    assert not verify_stage_records(tmp_path, stage)


@pytest.mark.parametrize("estimate,lower,expected", [(0.1, -0.02, "not_positive"),
                                                       (0.1, 0.0, "not_positive"),
                                                       (0.1, 0.02, "positive")])
def test_final_execution_success_is_separate_from_comparative_effect(tmp_path, monkeypatch, estimate, lower, expected):
    from src.research import paper_readiness
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.harness.quality.positive_effect_metric = "candidate_minus_reference"
    cfg.harness.quality.positive_effect_interval_low_metric = "paired_ci95_low"
    monkeypatch.setattr(paper_readiness, "_confirmation", lambda *_: {
        "status": "verified", "correct": True,
        "metrics": {"candidate_minus_reference": estimate, "paired_ci95_low": lower},
    })
    report = assess_readiness(tmp_path / "workspace", cfg, {})
    assert report["checks"]["confirmation"]["status"] == "verified"
    assert report["checks"]["scientific_effect"]["status"] == expected
    assert report["submission_ready"] is False


def test_pipeline_observer_has_no_global_leak(monkeypatch):
    from src.pipeline import cli as pipeline
    monkeypatch.setattr(pipeline, "_runtime_log_event", Mock())
    def execute(_argv):
        pipeline.log_event("phase_start", phase_name="paper")
        return 7
    monkeypatch.setattr(pipeline, "_main", execute)
    callback = Mock()
    assert pipeline.main([], event_callback=callback) == 7
    callback.assert_called_once_with("phase_start", phase_name="paper")
    pipeline.log_event("phase_start", phase_name="paper")
    assert callback.call_count == 1


def test_posthoc_omits_nonexistent_ablation_input_in_native_pilot(tmp_path):
    from src.pipeline.cli import load_pipeline_config
    from src.pipeline.phases import build_posthoc_cmd
    from src.pipeline.paths import derive_paths
    cfg = load_pipeline_config(["execution.phases=investigation,posthoc,paper", f"paths.run_dir={tmp_path}"])
    assert not any(arg.startswith("paths.evidence_dir=") for arg in build_posthoc_cmd(cfg, derive_paths(cfg)))


def test_full_native_preflight_fails_before_literature(tmp_path, monkeypatch, capsys):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "prepare", Mock(side_effect=AssertionError("must not spend literature credits")))
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "container.enabled=false"]) == 2
    assert "ablations requires container.enabled=true" in capsys.readouterr().err
    assert not workspace.exists()


def test_failed_review_can_resume_without_blessing_manuscript_edits(tmp_path):
    from src.research.harness import _resume_plan
    workspace = tmp_path / "workspace"
    paper = workspace / "run/paper"
    paper.mkdir(parents=True)
    (paper / "paper.pdf").write_bytes(b"%PDF original")
    (paper / "paper.tex").write_text("Original manuscript")
    (paper / "latest_manifest.json").write_text('{}')
    state = HarnessState(workspace, phases=["paper", "review", "revision"], quality_target="high_impact", input_sha256="abc")
    state.observe("phase_end", phase_name="paper", status="completed")
    (paper / "latest_manifest.json").write_text('{"ai_review":{"status":"failed"}}')
    state.observe("phase_end", phase_name="review", status="degraded")
    assert _resume_plan(workspace, state.payload, {"paper", "review", "revision"}) == {"review", "revision"}
    (paper / "paper.tex").write_text("Changed during failed review")
    with pytest.raises(ValueError, match="paper evidence changed"):
        _resume_plan(workspace, state.payload, {"paper", "review", "revision"})


def test_novelty_failure_retries_on_resume_without_retraining(tmp_path, monkeypatch):
    selected = Mock(side_effect=ValueError("provider temporarily unavailable"))
    monkeypatch.setattr(harness, "_selected_novelty", selected)
    _, workspace, calls = _run_completed_fixture(tmp_path, monkeypatch)
    assert selected.call_count == 1
    assert cli.main(["harness", "--resume", str(workspace)]) == 0
    assert selected.call_count == 2
    assert len(calls) == 1


def test_actual_candidate_packet_uses_host_receipt_idea_id_not_missing_best_manifest_field(tmp_path):
    import sys
    from tests.research.test_native_harness import fixture_task
    from tests.program_testutils import make_program_dir
    from src.investigation.native_harness import run_native_evaluation
    from src.research.harness_evidence import build_candidate_packet
    from src.research.novelty import validate_candidate_packet
    workspace = tmp_path / "workspace"
    (workspace / "run").mkdir(parents=True)
    task, _ = fixture_task(workspace / "run")
    (task / "task.md").write_text("Test a bounded score improvement")
    root = workspace / "run/investigation"
    attempt = root / "ideas/idea_1/attempt_1"
    program = attempt / "program_1"
    make_program_dir(program, content="SCORE = 0.7\n")
    assert run_native_evaluation(program_dir=program, evaluator=task / "evaluate.py", python_bin=sys.executable,
                                 timeout_seconds=10, environment={}, attempt_id="idea_1_attempt_1", idea_id="idea_1") == 0
    (root / "best_program_manifest.json").write_text(json.dumps({
        "program_dir": str(program), "entrypoint_path": "src/program.py", "source_attempt_id": "idea_1_attempt_1",
        "combined_score": 0.7, "correct": True,
    }))
    (attempt / "idea_bundle.json").write_text(json.dumps({"idea_id": "idea_1", "research_basis": {
        "status": "needs_verification", "contribution": "Candidate mechanism", "closest_work": [],
    }}))
    packet = build_candidate_packet(workspace)
    validate_candidate_packet(packet)
    assert packet["selection"]["idea_id"] == "idea_1"
    assert packet["evaluation"]["metrics"]["combined_score"] == 0.7
    assert any("SCORE = 0.7" in item["text"] for item in packet["source_files"])
    (program / "metrics.json").write_text('{"combined_score":99}')
    with pytest.raises(ValueError, match="current host receipt"):
        build_candidate_packet(workspace)
