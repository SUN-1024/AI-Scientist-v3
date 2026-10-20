"""Real subprocess transport, strict review judgments, evidence, and provenance."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import review_backend as backend


def review_payload() -> dict:
    return {
        "Summary": "A manuscript with limited independent evaluation.",
        "Strengths": ["The protocol is clear."], "Weaknesses": ["Only one seed."],
        "Questions": [], "Limitations": ["Small validation set."], "Ethical Concerns": False,
        "Originality": 2, "Quality": 2, "Clarity": 3, "Significance": 2,
        "Soundness": 2, "Presentation": 3, "Contribution": 2, "Overall": 3,
        "Confidence": 3, "Decision": "Reject",
        "Evidence References": [{"id": "r1", "source": "paper", "locator": "Section 4",
                                  "observation": "The reported evaluation uses one seed."}],
        "Concerns": [{"id": "c1", "severity": "major", "claim": "Stability is unknown.",
                      "requested_change": "Report multiple seeds.", "evidence_refs": ["r1"]}],
        "Requested Experiments": [{"id": "x1", "rationale": "Establish stability.",
                                   "design": "Compare three seeds.", "success_criterion": "Report the paired uncertainty.",
                                   "evidence_refs": ["r1"]}],
    }


def _fake_cli(tmp_path: Path, *, message: str | None = None, exit_code: int = 0, stderr: str = "") -> Path:
    executable = tmp_path / "headless"
    envelope = {"protocolVersion": 1, "type": "result", "command": "invoke", "exitCode": 0,
                "data": {"finalMessage": message or json.dumps(review_payload()),
                         "model": "codex/gpt-6.1-sol", "provider": "magpie", "nativeSessionId": "fresh-123"}}
    executable.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                          "assert '--session' not in sys.argv and '--run' not in sys.argv\n"
                          "assert sys.argv[sys.argv.index('--allow')+1] == 'read-only'\n"
                          "assert Path.cwd().name.startswith('agsci-independent-review-')\n"
                          "assert 'SECRET_UNRELATED' not in os.environ\n"
                          f"print({json.dumps(json.dumps(envelope))})\n"
                          f"print({stderr!r}, file=sys.stderr)\nsys.exit({exit_code})\n")
    executable.chmod(0o700)
    return executable


def _env(executable: Path) -> dict[str, str]:
    return {"HEADLESS_BIN": str(executable), "PATH": os.environ["PATH"], "HOME": str(executable.parent),
            "SECRET_UNRELATED": "must-not-leak", "OPENAI_API_KEY": "sk-test-secret-never-log"}


def test_real_subprocess_uses_fresh_context_and_records_provider(tmp_path: Path) -> None:
    executable = _fake_cli(tmp_path)
    result, receipt = backend.run_fresh_agent_json(
        "Inspect supplied evidence only", work_dir=tmp_path / "artifacts", model="codex/gpt-6.1-sol",
        profile="agsci-magpie", reasoning_effort="high", environment=_env(executable), timeout_seconds=10,
    )
    assert result == review_payload()
    assert receipt["status"] == "completed"
    assert receipt["observed_provider"] == "magpie"
    assert receipt["native_session_id"] == "fresh-123"
    assert receipt["prompt_sha256"] == backend.sha256("Inspect supplied evidence only")
    assert json.loads((tmp_path / "artifacts/receipt.json").read_text())["independent_context"] is True


def test_agent_review_environment_passes_deepseek_key_only_by_allowlist() -> None:
    environment = backend.agent_review_environment({
        "PATH": "/usr/bin",
        "DEEPSEEK_API_KEY": "deepseek-review-secret",
        "OPENAI_API_KEY": "openai-unrelated",
        "UNRELATED_SECRET": "must-not-leak",
    })
    assert environment["DEEPSEEK_API_KEY"] == "deepseek-review-secret"
    assert environment["OPENAI_API_KEY"] == "openai-unrelated"
    assert "UNRELATED_SECRET" not in environment


@pytest.mark.parametrize("message,exit_code,expected", [
    ("not JSON", 0, "exactly one JSON"),
    ('{"x":1,"x":2}', 0, "duplicate"),
    ('{"x":NaN}', 0, "non-finite"),
    ('{"x":1}', 7, "exited with code 7"),
])
def test_failure_preserves_sanitized_receipt(tmp_path: Path, message: str, exit_code: int, expected: str) -> None:
    executable = _fake_cli(tmp_path, message=message, exit_code=exit_code, stderr="token=sk-test-secret-never-log")
    with pytest.raises(backend.AgentReviewError, match=expected):
        backend.run_fresh_agent_json("Review", work_dir=tmp_path / "artifacts", environment=_env(executable))
    text = (tmp_path / "artifacts/stderr.log").read_text()
    assert "sk-test-secret-never-log" not in text and "[REDACTED]" in text
    assert json.loads((tmp_path / "artifacts/receipt.json").read_text())["status"] == "failed"


def test_timeout_is_explicit_failure(tmp_path: Path) -> None:
    executable = tmp_path / "headless"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    executable.chmod(0o700)
    with pytest.raises(backend.AgentReviewError, match="timed out"):
        backend.run_fresh_agent_json("Review", work_dir=tmp_path / "artifacts", environment=_env(executable), timeout_seconds=1)
    assert json.loads((tmp_path / "artifacts/receipt.json").read_text())["status"] == "failed"


def test_keyboard_interrupt_reaps_real_agent_process(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "headless"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    executable.chmod(0o700)
    wait = subprocess.Popen.wait
    interrupted = []

    def cancel_once(process, timeout=None):
        if timeout is not None and not interrupted:
            interrupted.append(process)
            raise KeyboardInterrupt
        return wait(process, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", cancel_once)
    with pytest.raises(KeyboardInterrupt):
        backend.run_fresh_agent_json("Review", work_dir=tmp_path / "artifacts",
                                     environment=_env(executable), timeout_seconds=10)
    assert len(interrupted) == 1
    assert interrupted[0].poll() is not None


@pytest.mark.parametrize("field,value", [
    ("Overall", True), ("Originality", 5), ("Decision", "Maybe"), ("Decision", []),
    ("Concerns", []), ("Evidence References", []),
    ("Requested Experiments", [{"id": "x", "rationale": "r", "design": "d", "success_criterion": "s", "evidence_refs": ["unknown"]}]),
])
def test_review_contract_rejects_unsupported_judgments(field: str, value: object) -> None:
    payload = review_payload()
    payload[field] = value
    with pytest.raises(backend.AgentReviewError):
        backend.validate_agent_review(payload, source_ids={"paper"})


def test_accept_cannot_hide_major_concerns_and_references_must_exist() -> None:
    payload = review_payload()
    payload["Decision"] = "Accept"
    with pytest.raises(backend.AgentReviewError, match="Accept cannot"):
        backend.validate_agent_review(payload, source_ids={"paper"})
    payload = review_payload()
    payload["Evidence References"][0]["source"] = "fabricated"
    with pytest.raises(backend.AgentReviewError, match="unknown evidence"):
        backend.validate_agent_review(payload, source_ids={"paper"})


def test_review_embeds_exact_inputs_and_all_independent_judgments(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\nactual-paper")
    evidence = tmp_path / "metrics.json"
    evidence.write_text('{"accuracy":0.7,"seed":42}')
    calls = []

    def call(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return review_payload(), {"status": "completed", "prompt_sha256": backend.sha256(prompt)}

    monkeypatch.setattr(backend, "run_fresh_agent_json", call)
    result = backend.review_with_agent(paper, "Full manuscript evidence. " * 30, evidence_paths=[evidence], reviewer_count=2)
    provenance = result["Provenance"]
    assert len(calls) == 3  # two independent reviews, one fresh meta-review
    assert calls[0][0] == calls[1][0]  # no previous reviewer judgment contaminates blind reviews
    assert "independent area chair" in calls[2][0]
    assert provenance["paper_sha256"] == backend.sha256(paper.read_bytes())
    assert provenance["inputs"][1]["sha256"] == backend.sha256(evidence.read_bytes())
    assert len(provenance["reviews"]) == 2
    assert '"accuracy":0.7' in provenance["prompt_snapshot"].replace('\\"', '"')
    assert result["Decision"] == "Reject"  # completed review is not a pass


def test_evidence_symlink_and_missing_file_fail_before_any_agent(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\nactual-paper")
    target = tmp_path / "outside.json"
    target.write_text("{}")
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(target)
    monkeypatch.setattr(backend, "run_fresh_agent_json", lambda *_a, **_kw: pytest.fail("must validate input first"))
    with pytest.raises(backend.AgentReviewError, match="symlink"):
        backend.review_with_agent(paper, "Manuscript evidence. " * 30, evidence_paths=[symlink])
    with pytest.raises(FileNotFoundError):
        backend.review_with_agent(paper, "Manuscript evidence. " * 30, evidence_paths=[tmp_path / "missing.json"])


def test_failed_review_binds_failure_to_pdf(tmp_path: Path, monkeypatch) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\nactual-paper")
    def fail(*_args, **_kwargs):
        raise backend.AgentReviewError("provider failed", provenance={"status": "failed"})
    monkeypatch.setattr(backend, "run_fresh_agent_json", fail)
    with pytest.raises(backend.AgentReviewError) as error:
        backend.review_with_agent(paper, "Manuscript evidence. " * 30)
    assert error.value.provenance["paper_sha256"] == backend.sha256(paper.read_bytes())
    assert error.value.provenance["status"] == "failed"


def test_managed_review_plumbs_agent_and_checks_pdf_hash(tmp_path: Path, monkeypatch) -> None:
    from src.investigation.final_ai_review import run_final_ai_review
    from tests.pipeline.test_final_ai_review import _Env
    env = _Env(tmp_path)
    monkeypatch.setenv("HEADLESS_BIN", "/trusted/headless")
    monkeypatch.setenv("CODEX_HOME", "/trusted/codex")
    monkeypatch.setenv("SECRET_UNRELATED", "excluded-secret")
    monkeypatch.setattr("src.investigation.final_ai_review.shutil.which", lambda _name: None)
    evidence = tmp_path / "metrics.json"
    evidence.write_text('{"seeds": 1}\n')
    captured = []
    def execute(command, _log, environment):
        captured.append((command, environment))
        payload = review_payload()
        pdf = Path(command[2])
        supplied_evidence = Path(command[command.index("--evidence") + 1])
        assert supplied_evidence != evidence
        assert supplied_evidence.name == evidence.name
        assert supplied_evidence.read_bytes() == evidence.read_bytes()
        payload["Provenance"] = {"backend": "agent", "independent_context": True,
                                 "paper_sha256": backend.sha256(pdf.read_bytes()), "inputs": [
                                     {"id": "paper", "name": "paper.pdf",
                                      "sha256": backend.sha256(pdf.read_bytes()), "bytes": pdf.stat().st_size},
                                     {"id": "evidence_1", "name": supplied_evidence.name,
                                      "sha256": backend.sha256(supplied_evidence.read_bytes()),
                                      "bytes": supplied_evidence.stat().st_size},
                                 ]}
        Path(command[command.index("--output") + 1]).write_text(json.dumps(payload))
        return "completed independent judgment", 0
    result = run_final_ai_review(
        repo_root=env.repo_root, paper_dir=env.paper_dir, results_dir=env.results_dir,
        execute_command=execute, reviewer_options={"backend": "agent", "agent": "codex",
        "model": "codex/gpt-6.1-sol", "profile": "agsci-magpie", "reasoning_effort": "high",
        "reviewer_count": 2, "evidence_paths": [tmp_path / "metrics.json"]},
    )
    assert result.status == "completed" and result.decision == "Reject"
    command, environment = captured[0]
    assert command[:2] == [sys.executable, str(env.repo_root / "src" / "aireview.py")]
    assert "--script" not in command
    assert command[command.index("--agent-profile") + 1] == "agsci-magpie"
    assert Path(command[command.index("--evidence") + 1]).name == "metrics.json"
    assert environment["HEADLESS_BIN"] == "/trusted/headless"
    assert "SECRET_UNRELATED" not in environment

    def stale(command, _log, _env):
        payload = review_payload()
        payload["Provenance"] = {"backend": "agent", "independent_context": True,
                                 "paper_sha256": "0" * 64, "inputs": [{"id": "paper"}]}
        Path(command[command.index("--output") + 1]).write_text(json.dumps(payload))
        return "stale response", 0
    result = run_final_ai_review(
        repo_root=env.repo_root, paper_dir=env.paper_dir, results_dir=env.results_dir,
        execute_command=stale, uv_executable="uv", reviewer_options={"backend": "agent"},
    )
    assert result.status == "failed" and "not bound" in result.error
    assert not (env.paper_dir / "ai_review.json").exists()


def test_agsci_agent_backend_does_not_require_openai_key_or_uv(tmp_path: Path) -> None:
    from src.agsci.review import main
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\n")
    calls = []
    rc = main(
        [str(paper), "--backend", "agent", "--agent-model", "codex/gpt-6.1-sol",
         "--agent-profile", "agsci-magpie", "--reasoning-effort", "high"],
        environ={"HEADLESS_BIN": "/trusted/headless"}, which=lambda _name: None,
        runner=lambda command, environment: calls.append((command, environment)) or 0, repo_root=tmp_path,
    )
    assert rc == 0 and len(calls) == 1
    assert calls[0][0][:2] == [sys.executable, str(tmp_path / "src" / "aireview.py")]
    assert calls[0][1]["HEADLESS_BIN"] == "/trusted/headless"
    assert "OPENAI_API_KEY" not in calls[0][1]


def test_agent_cli_runs_with_only_core_pdf_dependency(tmp_path: Path) -> None:
    """A fresh process extracts a real PDF and reviews it with legacy packages blocked."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    paper = tmp_path / "paper.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (" + b"The manuscript evaluates one seed. " * 6 + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(paper)
    executable = _fake_cli(tmp_path)
    output = tmp_path / "review.json"
    script = Path(__file__).resolve().parents[2] / "src" / "aireview.py"
    # Simulate the installed project runtime without uv's legacy dependency set.
    launcher = """
import builtins, runpy, sys
original_import = builtins.__import__
blocked = {'openai', 'backoff', 'numpy', 'pymupdf', 'pymupdf4llm'}
def core_only_import(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ModuleNotFoundError(f'blocked optional package: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = core_only_import
runpy.run_path(sys.argv.pop(1), run_name='__main__')
"""
    environment = _env(executable)
    environment.pop("OPENAI_API_KEY")
    completed = subprocess.run(
        [sys.executable, "-c", launcher, str(script), str(paper), "--backend", "agent",
         "--output", str(output), "--timeout-seconds", "10"],
        env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    review = json.loads(output.read_text())
    assert review["Decision"] == "Reject"
    assert review["Provenance"]["paper_sha256"] == backend.sha256(paper.read_bytes())
    assert review["Provenance"]["independent_context"] is True
