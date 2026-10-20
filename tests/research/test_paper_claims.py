"""Claim-rule checks for manuscripts."""
from __future__ import annotations

import json
from pathlib import Path

from src.research.paper_claims import check_claims, main

RULES = {
    "required": [{"pattern": r"manifest SHA-256", "reason": "bundle citation"}],
    "forbidden": [
        {"pattern": r"submission-ready", "reason": "no readiness claim"},
        {"pattern": r"peer[- ]reviewed", "reason": "no peer-review claim"},
    ],
}


def test_passes_when_required_present_and_forbidden_absent() -> None:
    report = check_claims("The manifest SHA-256 is abc. No human review was performed.", RULES)
    assert report["status"] == "passed"
    assert report["required"][0]["found"] is True
    assert all(row["matches"] == 0 for row in report["forbidden"])


def test_fails_when_required_missing() -> None:
    report = check_claims("The bundle is included.", RULES)
    assert report["status"] == "failed"
    assert report["required"][0]["found"] is False


def test_fails_on_forbidden_phrase() -> None:
    report = check_claims("The note is submission-ready; manifest SHA-256 is abc.", RULES)
    assert report["status"] == "failed"
    matches = {row["pattern"]: row["matches"] for row in report["forbidden"]}
    assert matches["submission-ready"] == 1


def test_unless_context_allows_a_disclaimer() -> None:
    rules = {"forbidden": [{"pattern": r"submission-ready",
                            "unless": r"(?:do|does) not claim"}]}
    report = check_claims("We do not claim that this note is submission-ready.", rules)
    assert report["status"] == "passed"
    report = check_claims("This note is submission-ready.", rules)
    assert report["status"] == "failed"


def test_forbidden_phrase_is_case_insensitive() -> None:
    report = check_claims("manifest sha-256 abc. This was Peer Reviewed.", RULES)
    assert report["status"] == "failed"


def test_cli_exit_code(tmp_path: Path, capsys) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(RULES))
    good = tmp_path / "good.tex"
    good.write_text("manifest SHA-256 abc.")
    assert main([str(good), "--rules", str(rules)]) == 0
    bad = tmp_path / "bad.tex"
    bad.write_text("submission-ready")
    assert main([str(bad), "--rules", str(rules)]) == 1
    output = tmp_path / "out.json"
    main([str(good), "--rules", str(rules), "--output", str(output)])
    assert json.loads(output.read_text())["status"] == "passed"
