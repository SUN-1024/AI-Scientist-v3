from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.research import cli, harness, workspace as preparation
from src.research.config import ResearchConfig, load_config, write_config
from tests.research.test_unified_research import make_task
from tests.research import test_novelty_gate as novelty_fixtures

study = novelty_fixtures.study


def test_interrupted_initial_review_reuses_sealed_literature_without_source(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    retrieve = Mock(wraps=preparation.prepare_literature)
    monkeypatch.setattr(preparation, "prepare_literature", retrieve)
    with pytest.raises(KeyboardInterrupt):
        preparation.prepare(task, workspace, offline=True, overrides=[],
                            after_literature=Mock(side_effect=KeyboardInterrupt))
    frozen = preparation.task_digest(workspace / "input_task")
    task.rename(tmp_path / "source_moved")
    manifest = preparation.prepare(task, workspace, offline=True, overrides=[], resume_preparation=True)
    assert retrieve.call_count == 1
    assert manifest["input_task_sha256"] == frozen
    assert preparation.resume(workspace)[1] == manifest


@pytest.mark.parametrize("changed", ["source.md", "research_context/text/prior.txt"])
def test_interrupted_preparation_rejects_changed_inputs_before_provider(tmp_path, monkeypatch, changed):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    with pytest.raises(KeyboardInterrupt):
        preparation.prepare(task, workspace, offline=True, overrides=[],
                            after_literature=Mock(side_effect=KeyboardInterrupt))
    (workspace / "input_task" / changed).write_text("changed evidence")
    retrieve = Mock(side_effect=AssertionError("must not fetch after input changes"))
    monkeypatch.setattr(preparation, "prepare_literature", retrieve)
    with pytest.raises(ValueError, match="changed"):
        preparation.prepare(task, workspace, offline=True, overrides=[], resume_preparation=True)
    assert not retrieve.called


def test_copy_interruption_keeps_original_source_binding(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    copy = preparation.copy_task
    def interrupted_copy(source, destination):
        destination.mkdir()
        (destination / "partial.txt").write_text("unfinished copy")
        raise KeyboardInterrupt
    monkeypatch.setattr(preparation, "copy_task", interrupted_copy)
    with pytest.raises(KeyboardInterrupt):
        preparation.prepare(task, workspace, offline=True, overrides=[])
    (task / "task.md").write_text("changed source")
    with pytest.raises(ValueError, match="Source task changed"):
        preparation.load_preparation(workspace)
    (task / "task.md").write_text("# Goal\nTest a bounded optimization hypothesis.\n")
    monkeypatch.setattr(preparation, "copy_task", copy)
    preparation.prepare(task, workspace, offline=True, overrides=[], resume_preparation=True)
    assert list((workspace / "preparation_attempts").glob("input_task_*/partial.txt"))


def test_generated_input_transaction_recovers_without_repeating_review(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    outputs = {"research_context/harness_novelty.json": '{"status":"needs_evidence"}\n',
               "research_agenda.json": '{"intent":"exploratory"}\n'}
    review = Mock(return_value=outputs)
    write = preparation.write_private_text
    def interrupted_write(path, text):
        write(path, text)
        if path.name == "harness_novelty.json":
            raise KeyboardInterrupt
    monkeypatch.setattr(preparation, "write_private_text", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        preparation.prepare(task, workspace, offline=True, overrides=[], after_literature=review)
    assert json.loads((workspace / preparation.PREPARATION_FILE).read_text())["state"] == "finalizing"
    monkeypatch.setattr(preparation, "write_private_text", write)
    preparation.prepare(task, workspace, offline=True, overrides=[], after_literature=review, resume_preparation=True)
    assert review.call_count == 1
    assert (workspace / "input_task/research_agenda.json").read_text() == outputs["research_agenda.json"]
    preparation.resume(workspace)


def test_empty_failed_retrieval_stays_retryable_but_local_evidence_can_continue(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.literature.queries = ["fixture literature query"]
    cfg.literature.sources = []
    write_config(task / "research.yaml", cfg)
    search = Mock(return_value=([], [{"stage": "model_search", "error": "temporary API failure"}]))
    monkeypatch.setattr("src.research.literature.search_literature", search)
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match="resume to retry"):
        preparation.prepare(task, workspace, offline=False, overrides=[])
    assert not (workspace / "research_manifest.json").exists()
    assert preparation.load_preparation(workspace)[1]["state"] == "literature"
    search.return_value = ([{"id": "prior", "title": "Recovered prior", "path": "source.md"}], [])
    preparation.prepare(task, workspace, offline=False, overrides=[], resume_preparation=True)
    assert list((workspace / "preparation_attempts").glob("literature_*/catalog.json"))
    assert search.call_count == 2
    local_task = make_task(tmp_path / "local_task")
    local_cfg = load_config(local_task)
    local_cfg.literature.queries = ["fixture literature query"]
    write_config(local_task / "research.yaml", local_cfg)
    search.return_value = ([], [{"stage": "model_search", "error": "temporary API failure"}])
    manifest = preparation.prepare(local_task, tmp_path / "local_workspace", offline=False, overrides=[])
    assert manifest["literature_status"] == "partial"


def test_harness_resume_before_manifest_preserves_configuration_and_avoids_retrieval(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())
    pipeline = Mock(return_value=0)
    monkeypatch.setattr("src.pipeline.cli.main", pipeline)
    novelty = harness._novelty_cycle
    monkeypatch.setattr(harness, "_novelty_cycle", Mock(side_effect=KeyboardInterrupt))
    retrieve = Mock(wraps=preparation.prepare_literature)
    monkeypatch.setattr(preparation, "prepare_literature", retrieve)
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--offline",
                     "--quality-target", "development", "execution.phases=investigation", "container.enabled=false"]) == 130
    assert not pipeline.called
    assert not (workspace / "research_manifest.json").exists()
    assert cli.main(["harness", "--resume", str(workspace), "agent.model=changed"]) == 2
    monkeypatch.setattr(harness, "_novelty_cycle", novelty)
    assert cli.main(["harness", "--resume", str(workspace)]) == 0
    assert retrieve.call_count == 1
    assert pipeline.call_count == 1
    state = json.loads((workspace / "harness_state.json").read_text())
    assert state["preparation"]["status"] == "prepared"
    assert "error" not in state


def test_harness_can_recover_missing_state_before_literature(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())
    monkeypatch.setattr("src.pipeline.cli.main", Mock(return_value=0))
    retrieve = preparation.prepare_literature
    monkeypatch.setattr(preparation, "prepare_literature", Mock(side_effect=KeyboardInterrupt))
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--offline",
                     "--quality-target", "development", "execution.phases=investigation", "container.enabled=false"]) == 130
    (workspace / "harness_state.json").unlink()
    monkeypatch.setattr(preparation, "prepare_literature", retrieve)
    assert cli.main(["harness", "--resume", str(workspace)]) == 0


def test_harness_recovers_manifest_published_before_lifecycle_save(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())
    monkeypatch.setattr("src.pipeline.cli.main", Mock(return_value=0))
    prepare = harness.prepare
    def interrupted_prepare(*args, **kwargs):
        prepare(*args, **kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(harness, "prepare", interrupted_prepare)
    assert cli.main(["harness", str(task), "--workspace", str(workspace), "--offline",
                     "--quality-target", "development", "execution.phases=investigation", "container.enabled=false"]) == 130
    assert (workspace / "research_manifest.json").exists()
    assert json.loads((workspace / "harness_state.json").read_text())["input_task_sha256"] is None
    monkeypatch.setattr(harness, "prepare", prepare)
    assert cli.main(["harness", "--resume", str(workspace)]) == 0


@pytest.mark.parametrize("name", ["research_agenda.json", "research_context/harness_novelty.json"])
@pytest.mark.parametrize("tamper", [False, True])
def test_hard_kill_output_temporary_is_verified_and_archived(tmp_path, monkeypatch, name, tamper):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    outputs = {"research_context/harness_novelty.json": '{"status":"needs_evidence"}\n',
               "research_agenda.json": '{"intent":"exploratory"}\n'}
    review = Mock(return_value=outputs)
    write = preparation.write_private_text
    temporary = workspace / "input_task" / name
    temporary = temporary.with_name("." + temporary.name + ".abcdef0123456789.tmp")
    def hard_kill(path, text):
        if path == workspace / "input_task" / name:
            temporary.write_bytes(b"changed" if tamper else text.encode()[:5])
            temporary.chmod(0o600)
            raise KeyboardInterrupt
        write(path, text)
    monkeypatch.setattr(preparation, "write_private_text", hard_kill)
    with pytest.raises(KeyboardInterrupt):
        preparation.prepare(task, workspace, offline=True, overrides=[], after_literature=review)
    monkeypatch.setattr(preparation, "write_private_text", write)
    if tamper:
        with pytest.raises(ValueError, match="Interrupted preparation output changed"):
            preparation.prepare(task, workspace, offline=True, overrides=[], after_literature=review, resume_preparation=True)
    else:
        preparation.prepare(task, workspace, offline=True, overrides=[], after_literature=review, resume_preparation=True)
        assert not temporary.exists()
        assert review.call_count == 1
        assert list((workspace / "preparation_attempts").glob("generated_output_*"))
        preparation.resume(workspace)


@pytest.mark.parametrize("tamper", [False, True])
def test_online_harness_initial_review_resume_reuses_first_paid_review(study, tmp_path, monkeypatch, tamper):
    from tests.research.test_novelty_resume import install_agent
    task, cfg, agenda, review = study
    write_config(task / "research.yaml", ResearchConfig(literature=cfg))
    (task / "research_agenda.json").write_text(json.dumps(agenda))
    workspace = tmp_path / "workspace"
    calls = install_agent(monkeypatch, review, fail_second=True)
    retrieve = Mock(wraps=preparation.prepare_literature)
    monkeypatch.setattr(preparation, "prepare_literature", retrieve)
    monkeypatch.setattr(harness, "ensure_execution_ready", Mock())
    pipeline = Mock(return_value=0)
    monkeypatch.setattr("src.pipeline.cli.main", pipeline)
    assert cli.main(["harness", str(task), "--workspace", str(workspace),
                     "--quality-target", "development", "execution.phases=investigation", "container.enabled=false"]) == 1
    assert len(calls) == 2 and not pipeline.called
    assert not (workspace / "research_manifest.json").exists()
    first_path = workspace / "novelty/initial/reviewer_1/receipt.json"
    first = first_path.read_bytes()
    if tamper:
        assessment = first_path.with_name("assessment.json")
        assessment.write_bytes(assessment.read_bytes() + b" ")
    assert cli.main(["harness", "--resume", str(workspace)]) == (1 if tamper else 0)
    assert len(calls) == (2 if tamper else 3)
    assert retrieve.call_count == 1
    assert pipeline.call_count == (0 if tamper else 1)
    assert first_path.read_bytes() == first
