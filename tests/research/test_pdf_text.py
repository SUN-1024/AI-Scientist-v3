"""Extraction failures remain visible and cannot qualify detailed citations."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from src.research import pdf_text
from src.investigation.bounded_process import ProcessOutputLimitError, ProcessTimeoutError
from src.research.config import LiteratureConfig
from src.research.evidence import load_catalog
from src.research.literature import _extract_text, prepare_literature
from src.research.novelty import citation_text
from src.research.novelty_review import source_packet

BROKEN = "teac hing /#0C exam ple /#5C /#28 /#29 " * 100
CLEAN = "Teaching sequences identify the unique consistent target. " * 100


def reader(monkeypatch, pages):
    import pypdf

    monkeypatch.setattr(pypdf, "PdfReader", lambda _: SimpleNamespace(
        pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in pages]))


def poppler_command(tmp_path, monkeypatch, output):
    executable = tmp_path / "pdftotext"
    executable.write_text(f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
                          "assert Path(sys.argv[-2]).read_bytes()==b'%PDF-fixture'\n"
                          f"sys.stdout.buffer.write({output!r})\n")
    executable.chmod(0o700)
    monkeypatch.setattr(pdf_text.shutil, "which", lambda _: str(executable))


def test_ordinary_text_is_byte_stable_and_does_not_trigger_fallback(monkeypatch):
    reader(monkeypatch, ["first\nΩ", "second"])
    monkeypatch.setattr(pdf_text, "_poppler", lambda *_: pytest.fail("unnecessary fallback"))
    metadata = {}
    text, status = _extract_text(b"%PDF-fixture", pdf=True, provenance=metadata)
    assert (text, status) == ("[Page 1]\nfirst\nΩ\n\n[Page 2]\nsecond", "full_text")
    assert metadata["method"] == "pypdf" and metadata["complete_nonempty_page_coverage"]
    assert metadata["semantic_fidelity_verified"] is False


def test_broken_fonts_use_a_real_bounded_alternative_process(tmp_path, monkeypatch):
    reader(monkeypatch, [BROKEN, BROKEN])
    poppler_command(tmp_path, monkeypatch, (CLEAN + "\f" + CLEAN + "\f").encode())
    metadata = {}
    text, status = _extract_text(b"%PDF-fixture", pdf=True, provenance=metadata)
    assert status == "full_text" and "Teaching sequences" in text
    assert metadata["method"] == "pdftotext_layout"
    assert metadata["initial_diagnostic"]["suspect"]
    assert not metadata["selected_diagnostic"]["suspect"]
    assert metadata["selected_text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert metadata["pages_extracted"] == metadata["total_pages"] == 2


@pytest.mark.parametrize("alternative", [None, "short\fshort\f", CLEAN + "\f", CLEAN + "\f\f"])
def test_failed_or_incomplete_alternative_does_not_become_full_text(tmp_path, monkeypatch, alternative):
    reader(monkeypatch, [BROKEN, BROKEN])
    if alternative is None:
        monkeypatch.setattr(pdf_text.shutil, "which", lambda _: None)
    else:
        poppler_command(tmp_path, monkeypatch, alternative.encode())
    metadata = {}
    _text, status = _extract_text(b"%PDF-fixture", pdf=True, provenance=metadata)
    assert status == "full_text_unreliable"
    assert metadata["alternative"]["status"] == "failed"


def test_page_coverage_does_not_hide_character_truncation(monkeypatch):
    reader(monkeypatch, [CLEAN])
    text, status, metadata = pdf_text.extract_pdf(b"%PDF-fixture", max_characters=100)
    assert len(text) == 100 and status == "full_text_partial"
    assert metadata["complete_nonempty_page_coverage"] and metadata["text_truncated"]


@pytest.mark.parametrize("failure", [ProcessTimeoutError, ProcessOutputLimitError])
def test_bounded_alternative_failure_remains_unreliable(monkeypatch, failure):
    reader(monkeypatch, [BROKEN])

    def fail(*_):
        raise failure()

    monkeypatch.setattr(pdf_text, "_poppler", fail)
    _text, status, metadata = pdf_text.extract_pdf(b"%PDF-fixture", max_characters=200_000)
    assert status == "full_text_unreliable"
    assert metadata["alternative"]["status"] == "failed"


def test_suspect_source_and_provenance_survive_catalog_and_packet(tmp_path, monkeypatch):
    reader(monkeypatch, [BROKEN])
    monkeypatch.setattr(pdf_text.shutil, "which", lambda _: None)
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-fixture")
    cfg = LiteratureConfig(sources=[{"id": "paper", "title": "Teaching",
                                    "path": "paper.pdf"}])
    source = prepare_literature(tmp_path, cfg, offline=True)["sources"][0]
    assert source["availability"] == "full_text_unreliable"
    assert load_catalog(tmp_path)["sources"][0] == source
    packet_source = source_packet(tmp_path)["sources"][0]
    assert packet_source["extraction"] == source["extraction"]
    with pytest.raises(ValueError, match="complete saved full text"):
        citation_text(tmp_path, {"paper": source}, "paper", {})
    catalog_path = tmp_path / "research_context/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["sources"][0]["extraction"]["input_sha256"] = "0" * 64
    catalog_path.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="extraction provenance"):
        load_catalog(tmp_path)
