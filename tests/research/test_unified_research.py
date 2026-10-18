from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.agsci.main import main as agsci
from src.research import cli, literature
from src.research.config import LiteratureConfig, ResearchConfig, load_config, write_config
from src.research.evidence import load_catalog, normalize_basis, render_context, validate_basis_for_task
from src.research.workspace import prepare, resume, task_digest, write_report
from src.orchestrator.idea_bundle import normalize_idea_bundle
from tests.investigation.test_campaign_ideas import _IDEA_BUNDLE


def make_task(path: Path) -> Path:
    path.mkdir()
    (path / "task.md").write_text("# Goal\nTest a bounded optimization hypothesis.\n")
    (path / "source.md").write_text(
        "# Prior method\nThe baseline uses a fixed step size.\n\n"
        "# Experiments\nTest accuracy is measured on disjoint data.\n\n"
        "# Limitation\nAn adaptive step size was not tested.\n"
    )
    cfg = ResearchConfig(literature=LiteratureConfig(sources=[{
        "id": "prior", "title": "A fixture prior method", "year": 2024,
        "venue": "Fixture", "url": "https://example.org/paper", "path": "source.md",
    }]))
    write_config(path / "research.yaml", cfg)
    return path


def basis(catalog, status="differentiated"):
    source = catalog["sources"][0]
    return {
        "status": status, "contribution": "Test an adaptive step size under the same budget.",
        "alternative_explanation": "More update steps explain the difference.",
        "discriminating_test": "Match total update steps and change only the step-size rule.",
        "closest_work": [{"source_id": source["id"], "overlap": "Same optimization task.",
                          "difference": "Adaptive rather than fixed step size.",
                          "evidence_ids": [source["passages"][0]["id"]]}],
    }


def test_prepare_keeps_source_immutable_and_resume_verifies_evidence(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    original = task_digest(task)
    network = Mock(side_effect=AssertionError("offline retrieval must not use network"))
    monkeypatch.setattr(literature, "download", network)
    workspace = tmp_path / "workspace"
    manifest = prepare(task, workspace, offline=True, overrides=["agent.name=codex"])
    assert task_digest(task) == original
    assert not (task / "research_context").exists()
    cfg, restored = resume(workspace)
    assert restored == manifest
    assert cfg.search.strategy == "tree"
    report = write_report(workspace)
    assert report["full_text_count"] == 1
    assert report["research_validity"] == "requires_evidence_review"
    assert report["pipeline_exit_code"] is None
    assert not network.called
    (workspace / "input_task" / "source.md").write_text("changed")
    with pytest.raises(ValueError, match="inputs changed"):
        resume(workspace)


def test_basis_is_preserved_and_forged_passage_is_rejected(tmp_path):
    task = make_task(tmp_path / "task")
    catalog = literature.prepare_literature(task, load_config(task).literature, offline=True)
    bundle = normalize_idea_bundle({**_IDEA_BUNDLE, "research_basis": basis(catalog)}, 1)
    validate_basis_for_task(task, bundle)
    assert bundle["research_basis"]["closest_work"][0]["source_id"] == "prior"
    bundle["research_basis"]["closest_work"][0]["evidence_ids"] = ["prior:invented"]
    with pytest.raises(ValueError, match="Invented passage"):
        validate_basis_for_task(task, bundle)
    bundle["research_basis"]["closest_work"][0]["source_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown prior-work"):
        validate_basis_for_task(task, bundle)


def test_abstract_only_does_not_establish_novelty_or_collision(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    cfg = LiteratureConfig(queries=["optimization"], fetch_full_text=False, proxy_fallback="")
    monkeypatch.setattr(literature, "search_literature", lambda *_args, **_kwargs: ([{
        "paperId": "1", "title": "A method", "abstract": "We use a fixed step size.",
    }], []))
    catalog = literature.prepare_literature(task, cfg)
    for status in ("differentiated", "collision"):
        with pytest.raises(ValueError, match="full-text passage"):
            validate_basis_for_task(task, {"research_basis": basis(catalog, status)})
    validate_basis_for_task(task, {"research_basis": basis(catalog, "needs_verification")})


def test_text_tampering_is_detected_and_empty_retrieval_stays_unknown(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    catalog = literature.prepare_literature(task, load_config(task).literature, offline=True)
    text_path = task / "research_context" / catalog["sources"][0]["text_path"]
    text_path.write_text("tampered")
    with pytest.raises(ValueError, match="text changed"):
        load_catalog(task)
    empty_task = tmp_path / "empty"
    empty_task.mkdir()
    monkeypatch.setattr(literature, "search_literature", lambda *_args: ([], [{"stage": "model_search", "error": "HTTP 429"}]))
    catalog = literature.prepare_literature(empty_task, LiteratureConfig(queries=["unknown topic"]))
    assert catalog["status"] == "unavailable"
    assert catalog["novelty_status"] == "not_assessed"
    assert catalog["retrieval_errors"]
    unresolved = {"status": "needs_verification", "closest_work": [],
                  "contribution": "An exploratory candidate.", "alternative_explanation": "Unknown.",
                  "discriminating_test": "Measure the effect against the baseline."}
    validate_basis_for_task(empty_task, {"research_basis": unresolved})
    assert "needs_verification" in render_context(empty_task)


def test_schema_and_paths_reject_invalid_inputs(tmp_path):
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.literature.sources[0]["path"] = "../outside.md"
    write_config(task / "research.yaml", cfg)
    with pytest.raises(ValueError, match="stay inside"):
        load_config(task)
    for bad in (None, {}, {"status": "novel"}):
        with pytest.raises(ValueError):
            normalize_basis(bad)
    with pytest.raises(ValueError, match="cited comparisons"):
        normalize_basis({"status": "differentiated", "closest_work": [],
                         "contribution": "X", "alternative_explanation": "Y", "discriminating_test": "Z"})


def test_cli_dry_run_is_read_only_and_dispatches_real_pipeline(tmp_path, monkeypatch, capsys):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    assert agsci(["research", "run", str(task), "--workspace", str(workspace), "--dry-run", "agent.name=codex"]) == 0
    assert not workspace.exists()
    preview = json.loads(capsys.readouterr().out)
    assert "investigation.execution.search_strategy=tree" in preview["pipeline_argv"]
    from src.pipeline.cli import load_pipeline_config
    from src.pipeline.paths import derive_paths
    from src.pipeline.phases import build_investigation_cmd
    resolved = load_pipeline_config(preview["pipeline_argv"])
    invocation = build_investigation_cmd(resolved, derive_paths(resolved))
    forwarded = next(value.split("=", 1)[1] for value in invocation if value.startswith("execution.search_strategy="))
    assert forwarded == "'tree'"
    received = []

    def pipeline(argv):
        received.append(argv)
        return 7

    monkeypatch.setattr("src.pipeline.cli.main", pipeline)
    assert cli.main(["run", str(task), "--workspace", str(workspace), "--offline", "agent.name=codex"]) == 7
    assert received
    report = json.loads((workspace / "research_report.json").read_text())
    assert report["state"] == "pipeline_failed"
    assert report["pipeline_exit_code"] == 7
    assert cli.main(["run", "--resume", str(workspace)]) == 7
    assert len(received) == 2
    assert cli.main(["run", "--resume", str(workspace), "investigation.execution.max_idea_cycles=2"]) == 7
    assert cli.main(["run", "--resume", str(workspace)]) == 7
    assert "investigation.execution.max_idea_cycles=2" in received[-1]
    assert cli.main(["run", str(task), "paths.run_dir=elsewhere", "--dry-run"]) == 2


def test_init_template_is_reusable_and_never_overwrites(tmp_path, capsys):
    project = tmp_path / "project"
    assert cli.main(["init", str(project), "--template", "circpack", "--query", "circle packing"]) == 0
    assert (project / "program_0").is_dir()
    assert (project / "evaluate.py").is_file()
    assert load_config(project).literature.queries == ["circle packing"]
    before = task_digest(project)
    assert cli.main(["init", str(project), "--template", "circpack"]) == 2
    assert task_digest(project) == before


def test_paper_context_retains_novelty_uncertainty(tmp_path):
    from src.investigation.paper_prompt_context import paper_workspace_context_replacements

    task = make_task(tmp_path / "task")
    literature.prepare_literature(task, load_config(task).literature, offline=True)
    context = paper_workspace_context_replacements(tmp_path)["RESEARCH_CONTEXT"]
    assert "agent assessments" in context
    assert "prior" in context
    assert "do not establish novelty" in context


def test_full_text_citations_can_address_sections_outside_suggested_excerpts(tmp_path):
    task = make_task(tmp_path / "task")
    (task / "source.md").write_text("method results baseline " * 300 + "A distinct late section.")
    catalog = literature.prepare_literature(task, load_config(task).literature, offline=True)
    text = (task / "research_context" / catalog["sources"][0]["text_path"]).read_text()
    comparison = basis(catalog)
    start = text.index("A distinct late section.")
    comparison["closest_work"][0]["evidence_ids"] = [f"prior:chars_{start}_{len(text)}"]
    assert comparison["closest_work"][0]["evidence_ids"][0] not in {
        passage["id"] for passage in catalog["sources"][0]["passages"]
    }
    validate_basis_for_task(task, {"research_basis": comparison})
    comparison["closest_work"][0]["evidence_ids"] = [f"prior:chars_{start}_{len(text) + 1}"]
    with pytest.raises(ValueError, match="outside the source"):
        validate_basis_for_task(task, {"research_basis": comparison})


def test_blank_pdf_is_not_full_text():
    import io
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="No extractable text"):
        literature._extract_text(buffer.getvalue(), pdf=True)


def test_duplicate_local_sources_and_unknown_pipeline_options_fail_early(tmp_path):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    assert cli.main(["run", str(task), "--workspace", str(workspace), "unknown_key=1"]) == 2
    assert not workspace.exists()
    cfg = load_config(task)
    cfg.literature.sources.append(dict(cfg.literature.sources[0]))
    write_config(task / "research.yaml", cfg)
    with pytest.raises(ValueError, match="Duplicate local source ID"):
        load_config(task)


def test_pipeline_exception_is_recorded_as_failure(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    workspace = tmp_path / "workspace"
    monkeypatch.setattr("src.pipeline.cli.main", Mock(side_effect=RuntimeError("fixture failure")))
    with pytest.raises(RuntimeError, match="fixture failure"):
        cli.main(["run", str(task), "--workspace", str(workspace), "--offline"])
    report = json.loads((workspace / "research_report.json").read_text())
    assert report["state"] == "pipeline_failed"
    assert report["pipeline_exit_code"] == 1


@pytest.mark.parametrize('prompt', ['idea.md', 'harness_edit.md'])
def test_idea_prompt_keeps_interface_when_task_uses_custom_headings(tmp_path, prompt):
    from src.investigation.files import load_instructions

    task = tmp_path / 'task'
    task.mkdir()
    brief = '# Clinical pilot\n## Candidate interface\nReturn one logit per bag. Do not use patch labels.\n'
    (task / 'task.md').write_text(brief)
    template = Path(__file__).resolve().parents[2] / 'src/prompts' / prompt
    rendered = load_instructions(template, task, tmp_path / 'results')
    assert brief in rendered
    assert 'TASK_BRIEF' not in rendered
    if prompt == 'idea.md':
        assert 'Shell tools may be used for bounded file I/O' in rendered
        assert 'Do not execute the candidate, evaluator' in rendered
    else:
        assert 'Do not execute, source, or import any candidate' in rendered
