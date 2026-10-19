"""Bounded PDF extraction with a recorded check for encoded font-name debris.

Page coverage and this narrow diagnostic do not establish semantic fidelity.
Alternative extraction creates new text; it never changes a saved source view.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from src.investigation.bounded_process import BoundedProcessError, run_bounded_process


def _diagnostic(text: str) -> dict[str, Any]:
    count = len(re.findall(r"/#[0-9A-Fa-f]{2}", text))
    fraction = 4 * count / max(1, len(text))
    return {"encoded_font_escape_tokens": count,
            "encoded_font_escape_character_fraction": fraction,
            "suspect": count >= 32 and fraction >= .005}


def _poppler(body: bytes, expected_pages: int) -> tuple[list[str], dict[str, Any]]:
    command = shutil.which("pdftotext")
    if command is None:
        raise ValueError("pdftotext is unavailable")
    with tempfile.TemporaryDirectory(prefix="agsci-pdf-text-") as temporary:
        root = Path(temporary)
        path = root / "input.pdf"
        path.write_bytes(body)
        path.chmod(0o600)
        result = run_bounded_process(
            [command, "-layout", "-enc", "UTF-8", str(path), "-"], cwd=root,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": str(root), "LC_ALL": "C"},
            timeout_seconds=20, max_output_bytes=2 * 1024 * 1024,
        )
    if result.returncode:
        raise ValueError("pdftotext did not complete successfully")
    text = result.stdout.decode("utf-8")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    if len(pages) != expected_pages or any(not page.strip() for page in pages):
        raise ValueError("pdftotext did not preserve all nonempty PDF pages")
    return pages, {"method": "pdftotext_layout", "executable": command,
                   "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
                   "stderr": result.stderr.decode("utf-8", errors="replace")[:2000],
                   "output_limit_bytes": 2 * 1024 * 1024, "timeout_seconds": 20}


def extract_pdf(body: bytes, *, max_characters: int) -> tuple[str, str, dict[str, Any]]:
    import io
    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(body))
    total_pages = len(reader.pages)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
        if sum(map(len, pages)) > max_characters:
            break
    if not any(page.strip() for page in pages):
        raise ValueError("No extractable text; scanned PDFs need OCR before use")

    def marked(values: list[str]) -> str:
        return "\n\n".join(f"[Page {i + 1}]\n{text}" for i, text in enumerate(values))

    text = marked(pages)
    initial = _diagnostic(text)
    provenance: dict[str, Any] = {
        "policy_version": 1, "input_sha256": hashlib.sha256(body).hexdigest(),
        "method": "pypdf", "pypdf_version": pypdf.__version__, "total_pages": total_pages,
        "initial_diagnostic": initial, "semantic_fidelity_verified": False,
    }
    if initial["suspect"]:
        try:
            alternative, receipt = _poppler(body, total_pages)
            proposed = marked(alternative)
            diagnostic = _diagnostic(proposed)
            # Reject a page-complete result that drops most of the observed text.
            if diagnostic["suspect"] or len(proposed) < .6 * len(text):
                raise ValueError("Alternative text failed the extraction-quality checks")
            pages, text = alternative, proposed
            provenance.update(method="pdftotext_layout", alternative=receipt)
        except (ValueError, OSError, UnicodeError, BoundedProcessError) as exc:
            provenance["alternative"] = {"status": "failed", "reason": str(exc)}
    diagnostic = _diagnostic(text)
    covered = len(pages) == total_pages and all(page.strip() for page in pages)
    partial = not covered or len(text) > max_characters
    availability = ("full_text_unreliable" if diagnostic["suspect"] else
                    "full_text_partial" if partial else "full_text")
    selected = text[:max_characters]
    provenance.update(pages_extracted=len(pages), complete_nonempty_page_coverage=covered,
                      selected_diagnostic=diagnostic,
                      selected_text_sha256=hashlib.sha256(selected.encode()).hexdigest(),
                      text_truncated=len(text) > max_characters)
    return selected, availability, provenance
