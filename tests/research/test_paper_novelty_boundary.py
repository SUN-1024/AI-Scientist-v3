"""Execute the shell host boundary without launching paper agents or containers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["paper.sh", "paper_edit.sh"])
def boundary(request, tmp_path):
    script = (ROOT / "src/scripts" / request.param).read_text()
    begin = script.index("refresh_selected_novelty_context() {")
    end = script.index("\n}\n", begin) + 3
    function = script[begin:end]
    host = tmp_path / "host"
    for package in (host / "src", host / "src/research", host / "src/orchestrator"):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
    (host / "src/research/evidence.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def verify_selected_novelty_for_paper(payload, *, investigation_results_dir):\n"
        "    assert str(investigation_results_dir) == os.environ['EXPECTED_INVESTIGATION']\n"
        "    return json.loads(Path(os.environ['HOST_PROJECTION']).read_text())\n"
        "def verify_methodology_for_paper(payload, *, investigation_results_dir):\n"
        "    assert str(investigation_results_dir) == os.environ['EXPECTED_INVESTIGATION']\n"
        "    return json.loads(Path(os.environ['HOST_METHODOLOGY_PROJECTION']).read_text())\n"
    )
    (host / "src/research/harness_state.py").write_text(
        "import json\ndef read_json(path):\n"
        "    if path.is_symlink(): raise ValueError('unsafe paper data')\n"
        "    return json.loads(path.read_text())\n"
    )
    (host / "src/orchestrator/state.py").write_text(
        "def write_private_text(path, text):\n    path.write_text(text)\n"
    )
    paper = tmp_path / "paper"
    paper.mkdir()
    projection = tmp_path / "host_projection.json"
    methodology = tmp_path / "host_methodology_projection.json"
    methodology.write_text("null")
    investigation = tmp_path / "managed/run"
    environment = {
        **os.environ, "PYTHON_BIN": sys.executable, "REPO_ROOT": str(host), "OUTPUT_DIR": str(paper),
        "SOURCE_RESULTS_DIR": str(investigation), "INVESTIGATION_RESULTS_DIR": "",
        "EXPECTED_INVESTIGATION": str(investigation), "HOST_PROJECTION": str(projection),
        "HOST_METHODOLOGY_PROJECTION": str(methodology),
    }
    def run(mode="refresh"):
        return subprocess.run(["bash", "-c", function + '\nrefresh_selected_novelty_context "$CHECK_MODE"\n'],
                              env={**environment, "CHECK_MODE": mode}, capture_output=True, text=True)
    return paper / "paper_data.json", projection, run, script


def test_paper_boundary_refreshes_only_selected_projection_then_checks_it(boundary):
    paper, projection, run, _script = boundary
    original = {"totals": {"evaluated_programs": 4}, "final_evaluation": {"qualified": False}}
    selected = {"status": "needs_evidence", "verification_status": "host_verified", "gaps": [{"code": "unresolved"}]}
    paper.write_text(json.dumps(original))
    projection.write_text(json.dumps(selected))
    assert run().returncode == 0
    assert json.loads(paper.read_text()) == {**original, "selected_novelty_review": selected}
    saved = paper.read_bytes()
    assert run("verify_current").returncode == 0
    assert paper.read_bytes() == saved


def test_paper_boundary_rejects_changed_proof_after_audit_without_rewriting(boundary):
    paper, projection, run, _script = boundary
    selected = {"status": "supported", "candidate_packet_sha256": "a" * 64}
    paper.write_text(json.dumps({"selected_novelty_review": selected}))
    saved = paper.read_bytes()
    projection.write_text(json.dumps({"status": "needs_evidence", "gaps": [{"code": "candidate_changed"}]}))
    result = run("verify_current")
    assert result.returncode != 0
    assert "changed after paper review" in result.stderr
    assert paper.read_bytes() == saved


def test_paper_boundary_leaves_nonharness_data_unchanged(boundary):
    paper, projection, run, _script = boundary
    paper.write_text('{"totals": {"evaluated_programs": 1}}')
    saved = paper.read_bytes()
    projection.write_text("null")
    assert run().returncode == 0
    assert run("verify_current").returncode == 0
    assert paper.read_bytes() == saved


def test_paper_boundary_refreshes_and_rechecks_methodology_without_novelty(boundary):
    paper, projection, run, _script = boundary
    original = {"totals": {"evaluated_programs": 1}}
    paper.write_text(json.dumps(original))
    projection.write_text("null")
    methodology_path = projection.with_name("host_methodology_projection.json")
    methodology = {"status": "needs_evidence", "assessment_scope": "frozen_initial_agenda",
                   "rules_sha256": "a" * 64}
    methodology_path.write_text(json.dumps(methodology))
    assert run().returncode == 0
    assert json.loads(paper.read_text()) == {**original, "research_methodology": methodology}
    saved = paper.read_bytes()
    assert run("verify_current").returncode == 0
    methodology_path.write_text(json.dumps({**methodology, "rules_sha256": "b" * 64}))
    result = run("verify_current")
    assert result.returncode != 0
    assert "Frozen methodology evidence changed after paper review" in result.stderr
    assert paper.read_bytes() == saved


def test_paper_boundary_runs_before_review_audit_and_finalization(boundary):
    _paper, _projection, _run, script = boundary
    assert "build_paper_review_packet() {\n  refresh_selected_novelty_context || return 1" in script
    assert "prepare_readonly_paper_audit_workspace() {\n  refresh_selected_novelty_context || return 1" in script
    assert "finalize_paper_evidence_audit() {\n  refresh_selected_novelty_context verify_current || return 1" in script
    assert 'refresh_selected_novelty_context verify_current\n"$PYTHON_BIN" - "$REPO_ROOT" "$OUTPUT_DIR"' in script
    if "RESUME_EXISTING=" in script:
        refresh = script.index('if [ "$RESUME_EXISTING" -eq 1 ]; then\n  refresh_selected_novelty_context')
        assert refresh < script.index('verify_resumed_final_outcome(')
