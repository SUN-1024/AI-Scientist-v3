"""Recorded render metadata must remain verifiable without discarding its prior evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.orchestrator import paper_artifact_provenance, paper_step
from src.orchestrator.paper_evidence_audit import (
    audit_input_sha256, prepare_readonly_workspace, remove_readonly_workspace,
)
from src.research.harness_state import HarnessState, verify_stage_records


def _workspace(root: Path, *, failed: bool = False) -> tuple[bytes, bytes]:
    (root / "figures").mkdir(parents=True)
    (root / "paper.tex").write_text("% @evidence[[figure_manifest.json]]\n")
    (root / "paper_strategy.json").write_text('{"figure_specs": []}\n')
    script = root / "build.py"
    script.write_text("# Original figure manifest producer.\n")
    (root / "figures/overview.prompts.json").write_text('{"panels": [{"id": "overview"}]}\n')
    row = {"figure_id": "overview", "generator": None if failed else "generate-concept-figure",
           "prompt_log_path": "figures/overview.prompts.json"}
    if failed:
        row.update(attempted_generator="generate-concept-figure", fallback_reason="earlier failure")
    manifest = root / "figure_manifest.json"
    manifest.write_text(json.dumps({"figures": [row]}, indent=1) + "\n")
    sidecar = root / "figure_manifest.json.prov.json"
    sidecar.write_text(json.dumps({
        "schema_name": "scientific-evidence.artifact-manifest",
        "artifact": {"path": "figure_manifest.json", "kind": "derived",
                     "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "derived_from": ["@evidence[[paper_strategy.json]]"],
        "generated_by": {"script_path": "build.py", "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()},
    }, indent=1) + "\n")
    subprocess.check_call([sys.executable, str(paper_step.SCIENTIFIC_EVIDENCE_SCRIPT),
                           "emit-bundle", "--root-dir", str(root)])
    return manifest.read_bytes(), sidecar.read_bytes()


def _verify(root: Path) -> dict:
    report = paper_step.validate_evidence_bundle(root)
    assert report["checked"]
    assert report["hard_errors"] == []
    assert report["summary"]["dangling_references"] == 0
    return report


@pytest.mark.parametrize("success", [False, True])
def test_render_outcome_preserves_recorded_input_and_verifies_new_provenance(tmp_path: Path, success: bool) -> None:
    original, original_provenance = _workspace(tmp_path, failed=success)
    _verify(tmp_path)
    if success:
        paper_step.record_concept_figure_success(tmp_path, "overview")
    else:
        paper_step.record_concept_figure_failure(tmp_path, "overview", "renderer unavailable")
    manifest = tmp_path / "figure_manifest.json"
    record = json.loads((tmp_path / "figure_manifest.json.prov.json").read_text())
    assert record["artifact"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    prior = tmp_path / record["derived_from"][0].removeprefix("@evidence[[").removesuffix("]]")
    assert prior.read_bytes() == original
    archives = list(tmp_path.glob(".figure_manifest.provenance.*.json"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == original_provenance
    assert _verify(tmp_path)["summary"]["manifests_checked"] >= 2
    row = json.loads(manifest.read_text())["figures"][0]
    assert row["generator"] == ("generate-concept-figure" if success else None)


def test_reasserting_the_same_outcome_does_not_rewrite_recorded_manifest(tmp_path: Path) -> None:
    _workspace(tmp_path)
    paper_step.record_concept_figure_failure(tmp_path, "overview", "renderer unavailable")
    paths = [tmp_path / "figure_manifest.json", tmp_path / "figure_manifest.json.prov.json"]
    previous = [(p.read_bytes(), p.stat().st_ino) for p in paths]
    archive_names = {p.name for p in tmp_path.glob(".figure_manifest.*")}
    paper_step.record_concept_figure_failure(tmp_path, "overview", "renderer unavailable")
    assert [(p.read_bytes(), p.stat().st_ino) for p in paths] == previous
    assert {p.name for p in tmp_path.glob(".figure_manifest.*")} == archive_names
    _verify(tmp_path)


@pytest.mark.parametrize("unsafe", ["stale_hash", "symlink", "fifo"])
def test_render_outcome_does_not_launder_stale_or_unsafe_input_provenance(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "paper"
    original, original_provenance = _workspace(root)
    sidecar = root / "figure_manifest.json.prov.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(original_provenance)
    if unsafe == "stale_hash":
        payload = json.loads(original_provenance)
        payload["artifact"]["sha256"] = "f" * 64
        sidecar.write_text(json.dumps(payload))
        stale = sidecar.read_bytes()
    else:
        sidecar.unlink()
        if unsafe == "symlink":
            sidecar.symlink_to(outside)
        else:
            os.mkfifo(sidecar, 0o600)
    with pytest.raises(RuntimeError, match="Recorded paper artifact hash|Unsafe paper provenance|not a regular file"):
        paper_step.record_concept_figure_failure(root, "overview", "renderer unavailable")
    assert (root / "figure_manifest.json").read_bytes() == original
    assert outside.read_bytes() == original_provenance
    assert not list(root.glob(".figure_manifest.before.*"))
    if unsafe == "stale_hash":
        assert sidecar.read_bytes() == stale
    elif unsafe == "symlink":
        assert sidecar.is_symlink()


def test_provenance_write_failure_restores_both_registered_files(tmp_path: Path, monkeypatch) -> None:
    original, original_provenance = _workspace(tmp_path)
    publish = paper_artifact_provenance._publish

    def fail_sidecar(directory_fd, name, data, **kwargs):
        if name == "figure_manifest.json.prov.json" and data != original_provenance:
            raise OSError("injected sidecar publication failure")
        return publish(directory_fd, name, data, **kwargs)

    monkeypatch.setattr(paper_artifact_provenance, "_publish", fail_sidecar)
    with pytest.raises(OSError, match="injected"):
        paper_step.record_concept_figure_failure(tmp_path, "overview", "renderer unavailable")
    assert (tmp_path / "figure_manifest.json").read_bytes() == original
    assert (tmp_path / "figure_manifest.json.prov.json").read_bytes() == original_provenance
    assert list(tmp_path.glob(".figure_manifest.before.*.json"))
    _verify(tmp_path)


def _registered_pdf(root: Path) -> tuple[bytes, bytes]:
    pdf = root / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nprevious development fixture\n")
    sidecar = root / "paper.pdf.prov.json"
    sidecar.write_text(json.dumps({
        "schema_name": "scientific-evidence.artifact-manifest",
        "artifact": {"path": "paper.pdf", "kind": "manuscript",
                     "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()},
        "derived_from": ["@evidence[[paper.tex]]"],
    }) + "\n")
    return pdf.read_bytes(), sidecar.read_bytes()


def test_latex_compilation_retains_old_pdf_before_writing_and_records_replacement(tmp_path: Path, monkeypatch) -> None:
    _workspace(tmp_path)
    original, original_provenance = _registered_pdf(tmp_path)
    logs = [{"returncode": 0, "command": ["fixture-compiler"]}]

    def compile_fixture(**kwargs):
        assert kwargs["paper_dir"] == tmp_path
        archives = list(tmp_path.glob(".paper.pdf.before.*.pdf"))
        assert len(archives) == 1 and archives[0].read_bytes() == original
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.7\nupdated development fixture\n")
        (tmp_path / "paper.fls").write_text("INPUT paper.tex\nOUTPUT paper.pdf\n")
        return logs

    monkeypatch.setattr(paper_step, "_compile_latex", compile_fixture)
    assert paper_step.compile_latex(paper_dir=tmp_path) == logs
    record = json.loads((tmp_path / "paper.pdf.prov.json").read_text())
    assert record["artifact"]["sha256"] == hashlib.sha256((tmp_path / "paper.pdf").read_bytes()).hexdigest()
    assert record["named_outputs"]["operation"] == "latex_compile"
    assert "@evidence[[paper.tex]]" in record["derived_from"]
    assert "@evidence[[paper.fls]]" in record["derived_from"]
    assert list(tmp_path.glob(".paper.pdf.provenance.*.json"))[0].read_bytes() == original_provenance
    _verify(tmp_path)


@pytest.mark.parametrize("failure", ["compile", "sidecar"])
def test_failed_compile_or_record_publication_restores_previous_pdf(tmp_path: Path, monkeypatch, failure: str) -> None:
    _workspace(tmp_path)
    original, original_provenance = _registered_pdf(tmp_path)
    publish = paper_artifact_provenance._publish

    def compile_fixture(**kwargs):
        (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.7\n" + b"partial fixture" * 400000)
        if failure == "compile":
            raise RuntimeError("injected compiler failure")
        return []

    def fail_sidecar(directory_fd, name, data, **kwargs):
        if failure == "sidecar" and name == "paper.pdf.prov.json" and data != original_provenance:
            raise RuntimeError("injected compiler provenance failure")
        return publish(directory_fd, name, data, **kwargs)

    monkeypatch.setattr(paper_step, "_compile_latex", compile_fixture)
    monkeypatch.setattr(paper_artifact_provenance, "_publish", fail_sidecar)
    with pytest.raises(RuntimeError, match="injected compiler"):
        paper_step.compile_latex(paper_dir=tmp_path)
    assert (tmp_path / "paper.pdf").read_bytes() == original
    assert (tmp_path / "paper.pdf.prov.json").read_bytes() == original_provenance
    _verify(tmp_path)


def test_compile_does_not_replace_a_pdf_that_already_violates_its_record(tmp_path: Path, monkeypatch) -> None:
    _workspace(tmp_path)
    _registered_pdf(tmp_path)
    changed = b"%PDF-1.7\nunrecorded change\n"
    (tmp_path / "paper.pdf").write_bytes(changed)

    def unexpected_compile(**kwargs):
        raise AssertionError("must reject stale provenance before invoking the compiler")

    monkeypatch.setattr(paper_step, "_compile_latex", unexpected_compile)
    with pytest.raises(RuntimeError, match="Recorded paper artifact hash"):
        paper_step.compile_latex(paper_dir=tmp_path)
    assert (tmp_path / "paper.pdf").read_bytes() == changed
    assert not list(tmp_path.glob(".paper.pdf.before.*"))


def test_readonly_audit_snapshot_retains_new_records_and_exact_input_hashes(tmp_path: Path) -> None:
    root = tmp_path / "paper"
    _workspace(root)
    _registered_pdf(root)
    paper_step.record_concept_figure_failure(root, "overview", "renderer unavailable")
    with paper_artifact_provenance.recorded_pdf_compilation(root, "paper.tex"):
        (root / "paper.pdf").write_bytes(b"%PDF-1.7\nupdated snapshot fixture\n")
    _verify(root)
    expected = audit_input_sha256(root)
    readonly = tmp_path / "readonly"
    try:
        prepare_readonly_workspace(root, readonly, evidence_script_path=paper_step.SCIENTIFIC_EVIDENCE_SCRIPT)
        report = json.loads((readonly / "scientific-evidence/verification-report.json").read_text())
        assert report["summary"]["dangling_references"] == 0
        assert audit_input_sha256(readonly) == expected
        for path in root.glob(".figure_manifest.*"):
            assert (readonly / path.name).read_bytes() == path.read_bytes()
        assert not ((readonly / "paper.pdf").stat().st_mode & 0o222)
    finally:
        if readonly.exists():
            remove_readonly_workspace(readonly)


def test_host_report_replaces_writer_report_only_after_preserving_its_observed_record(tmp_path: Path) -> None:
    _workspace(tmp_path)
    report = tmp_path / "validation_report.json"
    sidecar = tmp_path / "validation_report.json.prov.json"
    original = b'{"status":"local_checks_only"}\n'
    original_record = b'{"artifact":{"sha256":"already_stale"}}\n'
    report.write_bytes(original)
    sidecar.write_bytes(original_record)
    with pytest.raises(RuntimeError):
        paper_step.validate_and_compile_paper(tmp_path)
    archived = list((tmp_path / "paper_diagnostics/writer_validation_record/round_0000").glob("attempt_*"))
    assert len(archived) == 1
    assert (archived[0] / "validation_report.json").read_bytes() == original
    assert (archived[0] / "validation_report.json.prov.json").read_bytes() == original_record
    assert json.loads(report.read_text())["ok"] is False
    assert not sidecar.exists()
    _verify(tmp_path)


def test_host_operational_report_remains_bound_by_completed_harness_receipt(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "run/paper"
    root.mkdir(parents=True)
    report = root / "validation_report.json"
    report.write_text('{"ok": false}\n')
    state = HarnessState(workspace, phases=["paper"], quality_target="development", input_sha256="fixture")
    state.observe("phase_end", phase_name="paper", status="completed", return_code=0)
    assert verify_stage_records(workspace, state.payload["stages"]["paper"])
    report.write_text('{"ok": true}\n')
    assert not verify_stage_records(workspace, state.payload["stages"]["paper"])
