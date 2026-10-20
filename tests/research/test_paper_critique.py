"""Independent multi-reviewer manuscript critique."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from src.research.paper_critique import critique, load_source, sha256

CRITIQUE = {
    "verdict": "accept_as_note",
    "blocking": [],
    "non_blocking": [{"quote": "a sentence", "problem": "wording", "fix": "tighten it"}],
    "prior_art_gaps": ["cite the closest theorem explicitly"],
    "venue_fit": {"comment": "fits", "replication": "no", "diagnostic_note": "best fit"},
    "required_extra_evidence": ["human check of the formalisation"],
    "verdict_reasoning": "The note is narrow and its claims match the evidence.",
}

TEX = r"""\documentclass{article}
\begin{document}
\section{Claim}
A finite construction with nine compatible maps.
\end{document}
"""


FAKE = r"""
import json, sys, uuid
from pathlib import Path
message = __MESSAGE__
prompt = Path(sys.argv[sys.argv.index("--prompt-file") + 1]).read_text()
assert "MANUSCRIPT:" in prompt
print(json.dumps({"protocolVersion": 1, "type": "result", "exitCode": 0,
                  "data": {"finalMessage": json.dumps(message),
                           "nativeSessionId": str(uuid.uuid4()),
                           "model": sys.argv[sys.argv.index("--model") + 1]}}))
"""


def settings_for(tmp_path: Path, message: dict | None = None) -> dict:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    codex_home = tmp_path / "codex"
    codex_home.mkdir(exist_ok=True)
    executable = tmp_path / "headless"
    body = FAKE.replace("__MESSAGE__", repr(message if message is not None else CRITIQUE))
    executable.write_text(f"#!{sys.executable}\n" + body)
    executable.chmod(0o700)
    return {"agent": "codex", "model": "codex/fixture", "profile": "fixture",
            "environment": {"HEADLESS_BIN": str(executable), "HOME": str(home),
                            "CODEX_HOME": str(codex_home), "PATH": os.environ["PATH"]},
            "timeout_seconds": 15}


def test_load_source_accepts_tex_and_markdown(tmp_path: Path) -> None:
    tex = tmp_path / "note.tex"
    tex.write_text(TEX)
    assert load_source(tex) == TEX
    markdown = tmp_path / "note.md"
    markdown.write_text("# Claim\n")
    assert load_source(markdown) == "# Claim\n"
    with pytest.raises(ValueError):
        load_source(tmp_path / "missing.tex")
    with pytest.raises(ValueError):
        other = tmp_path / "note.docx"
        other.write_text("x")
        load_source(other)


def test_two_reviewers_get_isolated_directories_and_a_report(tmp_path: Path) -> None:
    manuscript = tmp_path / "note.tex"
    manuscript.write_text(TEX)
    output = tmp_path / "critique"
    report = critique(manuscript, output, settings=settings_for(tmp_path), reviewers=2)
    assert report["verdicts"] == ["accept_as_note", "accept_as_note"]
    assert report["consensus"] == "accept_as_note"
    assert report["blocking_total"] == 0
    assert report["manuscript_sha256"] == sha256(TEX)
    assert report["independence_limit"].startswith("separate stateless model calls")
    for index in (1, 2):
        directory = output / f"reviewer_{index}"
        assert (directory / "assessment.json").is_file()
        assert (directory / "source.txt").read_text() == TEX
        assert (directory / "call" / "receipt.json").is_file()
    saved = json.loads((output / "critique_report.json").read_text())
    assert saved["reviewers"][0]["native_session_id"]
    assert len({row["directory"] for row in saved["reviewers"]}) == 2


def test_mixed_verdicts_are_reported_as_revise(tmp_path: Path) -> None:
    manuscript = tmp_path / "note.tex"
    manuscript.write_text(TEX)
    accept = critique(manuscript, tmp_path / "a", settings=settings_for(tmp_path), reviewers=2)
    assert accept["consensus"] == "accept_as_note"

    blocking = dict(CRITIQUE, verdict="revise",
                    blocking=[{"quote": "a sentence", "problem": "overclaim", "fix": "soften"}])
    revise = critique(manuscript, tmp_path / "b", settings=settings_for(tmp_path, blocking), reviewers=2)
    assert revise["consensus"] == "revise"
    assert revise["blocking_total"] == 2


def test_invalid_critique_is_rejected(tmp_path: Path) -> None:
    manuscript = tmp_path / "note.tex"
    manuscript.write_text(TEX)
    bad_verdict = dict(CRITIQUE, verdict="looks_fine")
    with pytest.raises(ValueError, match="verdict"):
        critique(manuscript, tmp_path / "out", settings=settings_for(tmp_path, bad_verdict), reviewers=2)
    missing = {key: value for key, value in CRITIQUE.items() if key != "prior_art_gaps"}
    with pytest.raises(ValueError, match="missing keys"):
        critique(manuscript, tmp_path / "out2", settings=settings_for(tmp_path, missing), reviewers=2)
    malformed = dict(CRITIQUE, blocking=[{"quote": "only a quote"}])
    with pytest.raises(ValueError, match="quote, problem and fix"):
        critique(manuscript, tmp_path / "out3", settings=settings_for(tmp_path, malformed), reviewers=2)


def test_existing_reviewer_output_is_not_overwritten(tmp_path: Path) -> None:
    manuscript = tmp_path / "note.tex"
    manuscript.write_text(TEX)
    output = tmp_path / "critique"
    (output / "reviewer_1").mkdir(parents=True)
    with pytest.raises(ValueError, match="already contains"):
        critique(manuscript, output, settings=settings_for(tmp_path), reviewers=2)


def test_cli_critique_command(tmp_path: Path, monkeypatch) -> None:
    from src.research.cli import main as cli_main

    manuscript = tmp_path / "note.tex"
    manuscript.write_text(TEX)
    environment = settings_for(tmp_path)["environment"]
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    output = tmp_path / "cli-critique"
    assert cli_main(["critique", str(manuscript), "--output", str(output),
                     "--model", "codex/fixture", "--profile", "fixture", "--reviewers", "2"]) == 0
    report = json.loads((output / "critique_report.json").read_text())
    assert report["verdicts"] == ["accept_as_note", "accept_as_note"]
    assert report["manuscript_sha256"] == sha256(TEX)


def test_settings_requirements(tmp_path: Path) -> None:
    manuscript = tmp_path / "note.tex"
    manuscript.write_text(TEX)
    with pytest.raises(ValueError, match="codex model and profile"):
        critique(manuscript, tmp_path / "out", settings={"agent": "claude", "model": "m", "profile": "p"})
    with pytest.raises(ValueError, match="two to four"):
        critique(manuscript, tmp_path / "out", settings=settings_for(tmp_path), reviewers=1)
