"""Required AI Scientist disclosure reaches generation, compile, and delivery."""
import json
from pathlib import Path
import textwrap

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from src.orchestrator import paper_step
from src.research.evidence import render_context
from src.research.config import ResearchConfig
from src.research.paper_readiness import assess_readiness
from src.research.upstream.disclosure import (
    DISCLOSURE_TEXT, MARKER_PATH, inspect_pdf_disclosure, load_integration_marker,
    validate_rendered_disclosure, write_integration_marker,
)
from tests.paper_step_support import fake_compile, write_valid_artifacts, REFS_PAGE, SUPPLEMENT_PAGE


def _pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): writer._add_object(font)})})
    lines = [part for line in text.splitlines() for part in textwrap.wrap(line, 80)]
    content = "BT /F1 10 Tf 14 TL 40 740 Td\n"
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")
        content += f"({escaped}) Tj T*\n"
    stream = DecodedStreamObject()
    stream.set_data((content + "ET\n").encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as handle:
        writer.write(handle)


def test_marker_is_frozen_and_prompt_includes_disclosure_without_literature(tmp_path):
    task = tmp_path / "task"
    write_integration_marker(task)
    prompt = render_context(task, for_paper=True)
    assert DISCLOSURE_TEXT in prompt
    assert "Do not invent human review" in prompt
    assert DISCLOSURE_TEXT not in render_context(task, for_paper=False)
    assert load_integration_marker(task)["disclosure_required"] is True
    marker = task / MARKER_PATH
    payload = json.loads(marker.read_text())
    payload["disclosure_required"] = False
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="marker changed"):
        render_context(task, for_paper=True)


def test_harness_freezes_marker_and_refuses_its_removal_on_resume(tmp_path, monkeypatch):
    from src.research.workspace import resume
    from tests.research.test_harness import _run_completed_fixture
    _, workspace, _ = _run_completed_fixture(tmp_path, monkeypatch)
    marker = workspace / "input_task" / MARKER_PATH
    assert marker.is_file()
    resume(workspace)
    marker.unlink()
    with pytest.raises(ValueError, match="Prepared research inputs changed"):
        resume(workspace)


def test_snapshot_ignore_rule_cannot_drop_required_marker(tmp_path):
    (tmp_path / ".agentic_task_ignore").write_text("research_context/upstream_integration.json\n")
    with pytest.raises(ValueError, match="must retain"):
        write_integration_marker(tmp_path)


@pytest.mark.parametrize("heading", ["Disclosure", "AI Assistance Disclosure", "3 Methods", "Abstract"])
def test_rendered_disclosure_is_accepted_in_prominent_sections(heading):
    report = validate_rendered_disclosure([heading + "\n" + DISCLOSURE_TEXT], required=True)
    assert report["status"] == "verified"
    assert report["scope"] == "required_rendered_statement_only_not_full_legal_compliance"


def test_reference_or_later_section_cannot_satisfy_prominent_disclosure():
    assert validate_rendered_disclosure(["References\n" + DISCLOSURE_TEXT], required=True)["hard_errors"]
    assert validate_rendered_disclosure(["Methods\nNo disclosure.\nReferences\n" + DISCLOSURE_TEXT], required=True)["hard_errors"]
    assert validate_rendered_disclosure(["Methods\nThe AI Scientist is a related work."], required=True)["hard_errors"]


def test_compile_rejects_disclosure_only_in_tex_comment_and_passes_visible_text(tmp_path, monkeypatch):
    write_valid_artifacts(tmp_path)
    write_integration_marker(tmp_path / "task")
    tex_path = tmp_path / "paper.tex"
    tex = tex_path.read_text()
    tex_path.write_text(tex + "\n% Disclosure: " + DISCLOSURE_TEXT + "\n")
    monkeypatch.setattr(paper_step, "compile_latex", fake_compile)
    monkeypatch.setattr(paper_step, "read_pdf_page_texts", lambda *_: ["Main text", REFS_PAGE, SUPPLEMENT_PAGE])
    with pytest.raises(RuntimeError, match="AI Scientist disclosure validation failed"):
        paper_step.validate_and_compile_paper(tmp_path)
    failed = json.loads((tmp_path / "validation_report.json").read_text())
    assert failed["ai_scientist_disclosure_validation"]["status"] == "missing_or_invalid"
    # The normal author/repair path changes the TeX; validation never inserts it.
    tex_path.write_text(tex.replace("\\section{Conclusion}", "\\section*{Disclosure}\n" + DISCLOSURE_TEXT + "\n\\section{Conclusion}"))
    monkeypatch.setattr(paper_step, "read_pdf_page_texts", lambda *_: ["Disclosure\n" + DISCLOSURE_TEXT, REFS_PAGE, SUPPLEMENT_PAGE])
    report = paper_step.validate_and_compile_paper(tmp_path)
    assert report["ok"] is True
    assert report["ai_scientist_disclosure_validation"]["status"] == "verified"


def test_real_pdf_and_readiness_recheck_current_artifact(tmp_path):
    write_integration_marker(tmp_path / "input_task")
    pdf = tmp_path / "run/paper/paper.pdf"
    _pdf(pdf, "Disclosure\n" + DISCLOSURE_TEXT)
    assert inspect_pdf_disclosure(pdf, required=True)["status"] == "verified"
    assert assess_readiness(tmp_path, ResearchConfig(), {})["checks"]["disclosure"]["status"] == "verified"
    _pdf(pdf, "Methods\nNo attribution statement.")
    report = assess_readiness(tmp_path, ResearchConfig(), {})
    assert report["checks"]["disclosure"]["status"] == "missing_or_invalid"
    assert any(gap["check"] == "disclosure" for gap in report["gaps"])
    assert any(action["check"] == "disclosure" and action["kind"] == "repair_manuscript" for action in report["next_actions"])


def test_legacy_v3_without_marker_keeps_existing_compile_behavior(tmp_path, monkeypatch):
    write_valid_artifacts(tmp_path)
    monkeypatch.setattr(paper_step, "compile_latex", fake_compile)
    monkeypatch.setattr(paper_step, "read_pdf_page_texts", lambda *_: ["Main text", REFS_PAGE, SUPPLEMENT_PAGE])
    report = paper_step.validate_and_compile_paper(tmp_path)
    assert report["ok"] is True
    assert report["ai_scientist_disclosure_validation"]["required"] is False
    assert DISCLOSURE_TEXT not in render_context(tmp_path / "task", for_paper=True)
    assert assess_readiness(tmp_path, ResearchConfig(), {})["checks"]["disclosure"]["status"] == "not_required"
