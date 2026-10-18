"""Host-required attribution for manuscripts using the adapted V1/V2 workflow.

This verifies a specific rendered disclosure statement. It is not a general
legal-compliance determination or a substitute for human manuscript review.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata
from typing import Any

from src.orchestrator.state import write_private_text

MARKER_PATH = Path("research_context/upstream_integration.json")
DISCLOSURE_TEXT = (
    "This manuscript was produced using The AI Scientist, with machine-generated "
    "contributions to the research workflow and manuscript drafting."
)
SOURCE_COMMITS = {
    "AI-Scientist-v1": "1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb",
    "AI-Scientist-v2": "96bd51617cfdbb494a9fc283af00fe090edfae48",
}
_HEADING = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?[ \t]+)?"
    r"(?:AI(?:[ \t]+Assistance)?[ \t]+Disclosure|Disclosure|Abstract|"
    r"(?:Materials[ \t]+and[ \t]+)?Methods?)[ \t]*(?:\n|:)")
_NEXT_SECTION = re.compile(
    r"(?im)^[ \t]*(?:\d+(?:\.\d+)*[.)]?[ \t]+[A-Z][^\n]{0,80}|"
    r"References|Bibliography|Introduction|Results|Discussion|Conclusions?|"
    r"Acknowledg(?:e)?ments|Supplementary Information|Appendix)[ \t]*\n")
_STATEMENT = re.compile(
    r"\bThis (?:manuscript|paper|report) (?:was|has been) "
    r"(?:produced|generated|prepared) (?:using|with) (?:The )?AI Scientist\b",
    re.IGNORECASE,
)


def integration_marker() -> dict[str, Any]:
    return {"schema_version": 1, "integration": "v1_v2_v3_harness",
            "source_commits": dict(SOURCE_COMMITS), "disclosure_required": True,
            "license": "AI Scientist Source Code License v1.0, December 2025",
            "required_statement": DISCLOSURE_TEXT}


def load_integration_marker(task_dir: Path) -> dict[str, Any] | None:
    # Paper workspaces use a host-created alias for their immutable task copy.
    path = task_dir.resolve() / MARKER_PATH
    if not path.exists() and not path.is_symlink():
        return None
    if path.parent.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > 16_384:
        raise ValueError("Invalid upstream integration disclosure marker")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker != integration_marker():
        raise ValueError("Upstream integration disclosure marker changed or is unsupported")
    return marker


def write_integration_marker(task_dir: Path) -> dict[str, Any]:
    """Write host provenance before the Harness freezes the prepared task."""
    from src.investigation.files import load_task_copy_ignore_patterns, task_copy_path_is_ignored
    if task_copy_path_is_ignored(str(MARKER_PATH), is_dir=False, patterns=load_task_copy_ignore_patterns(task_dir)):
        raise ValueError(".agentic_task_ignore must retain the upstream integration disclosure marker")
    path = task_dir / MARKER_PATH
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("Upstream integration marker cannot follow a symlink")
    existing = load_integration_marker(task_dir)
    if existing is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_private_text(path, json.dumps(integration_marker(), indent=2) + "\n")
    return integration_marker()


def disclosure_prompt(task_dir: Path) -> str:
    if load_integration_marker(task_dir) is None:
        return ""
    return (
        "\n\nREQUIRED AI SCIENTIST DISCLOSURE (host workflow requirement):\n"
        "This study uses adapted AI Scientist V1/V2 code. Put a prominent, visible "
        "Disclosure section in the manuscript, or put the statement in the Abstract "
        "or Methods. Use this sentence: \n" + DISCLOSURE_TEXT + "\n"
        "Keep it in the compiled PDF and all revisions. A TeX comment, metadata, "
        "footnote, reference entry, or watermark alone does not satisfy this check. "
        "Do not invent human review, author approval, or manual verification.\n"
    )


def paper_requires_disclosure(paper_dir: Path) -> bool:
    return load_integration_marker(paper_dir / "task") is not None


def validate_rendered_disclosure(page_texts: list[str] | None, *, required: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "required": required, "checked": required, "status": "not_required",
        "detail": "No adapted V1/V2 workflow marker is attached to this legacy task",
        "statement": DISCLOSURE_TEXT, "matched_page": None, "hard_errors": [],
        "scope": "required_rendered_statement_only_not_full_legal_compliance",
    }
    if not required:
        return report
    if not page_texts or not all(isinstance(text, str) for text in page_texts):
        report.update(status="missing_or_invalid", detail="Compiled PDF text is unavailable",
                      hard_errors=["Cannot inspect the compiled PDF for the required AI Scientist disclosure"])
        return report
    for index, raw in enumerate(page_texts):
        text = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
        headings = list(_HEADING.finditer(text))
        # A section heading can be at the bottom of one page. Include the next
        # page while keeping the search window short and before the next section.
        if index + 1 < len(page_texts):
            text += "\n" + unicodedata.normalize("NFKC", page_texts[index + 1])
        for heading in headings:
            section = text[heading.end():heading.end() + 1800]
            boundary = _NEXT_SECTION.search(section)
            if boundary is not None:
                section = section[:boundary.start()]
            if _STATEMENT.search(" ".join(section.split())):
                report.update(status="verified", matched_page=index + 1,
                              detail="Required attribution found in an extracted rendered manuscript section",
                              location=heading.group().strip())
                return report
    report.update(status="missing_or_invalid", detail="Required AI Scientist attribution is absent from a prominent rendered manuscript section", hard_errors=[
        "Required AI Scientist disclosure is missing from the rendered Abstract, Methods, or dedicated Disclosure section. Add: " + DISCLOSURE_TEXT
    ])
    return report


def inspect_pdf_disclosure(pdf_path: Path, *, required: bool) -> dict[str, Any]:
    if not required:
        return validate_rendered_disclosure(None, required=False)
    try:
        if pdf_path.is_symlink() or not pdf_path.is_file() or pdf_path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("PDF is missing or not a bounded regular file")
        from pypdf import PdfReader
        pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
        return validate_rendered_disclosure(pages, required=True)
    except Exception as exc:
        report = validate_rendered_disclosure(None, required=True)
        report["hard_errors"].append(f"PDF disclosure inspection failed: {exc}")
        return report
