"""A final review must refer to the exact ordered evidence packet supplied by the host."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.investigation.final_ai_review import run_final_ai_review
from tests.paper.test_agent_review import review_payload
from tests.pipeline.test_final_ai_review import (
    _Env, _interrupt_publication, _repo_root, _write_manifest,
)
from tests.revision_test_support import refresh_evidence_audit, write_workspace


def _supplied_evidence(command: list[str]) -> list[Path]:
    return [Path(command[index + 1]) for index, value in enumerate(command) if value == "--evidence"]


def _review_for_packet(command: list[str]) -> dict:
    pdf_index = next(index for index, value in enumerate(command) if value.endswith("aireview.py")) + 1
    paths = [Path(command[pdf_index]), *_supplied_evidence(command)]
    backend = command[command.index("--backend") + 1] if "--backend" in command else "openai"
    inputs = [
        {"id": f"evidence_{index}" if index else "paper", "name": path.name,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": len(path.read_bytes())}
        for index, path in enumerate(paths)
    ]
    inputs[0]["text_sha256"] = hashlib.sha256(b"Reviewer-extracted paper text").hexdigest()
    payload = review_payload()
    payload["Provenance"] = {
        "backend": backend, "independent_context": True,
        "paper_sha256": inputs[0]["sha256"], "inputs": inputs,
    }
    return payload


def _return_review(command: list[str], payload: dict) -> tuple[str, int]:
    Path(command[command.index("--output") + 1]).write_text(json.dumps(payload), encoding="utf-8")
    return "independent review completed", 0


@pytest.mark.parametrize("backend", ["agent", "openai"])
def test_complete_receipt_binds_ordered_snapshots_with_duplicate_names(tmp_path: Path, backend: str) -> None:
    env = _Env(tmp_path)
    evidence = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "selected_novelty.json"
        path.write_text(json.dumps({"selected_candidate": index}), encoding="utf-8")
        evidence.append(path)
    options = {"backend": backend, "evidence_paths": evidence}

    def execute(command, *_args):
        supplied = _supplied_evidence(command)
        assert len(supplied) == 2 and supplied[0] != supplied[1]
        for actual, original in zip(supplied, evidence):
            assert actual != original and actual.name == original.name
            assert actual.read_bytes() == original.read_bytes()
        payload = _review_for_packet(command)
        payload["Evidence References"][0]["source"] = "evidence_2"
        return _return_review(command, payload)

    result = env.run(execute, reviewer_options=options)
    assert result.status == "completed"
    assert result.review_path is not None and result.review_path.is_file()
    assert options["evidence_paths"] == evidence
    assert env.manifest()["ai_review"]["status"] == "completed"


@pytest.mark.parametrize("defect", ["missing", "hash", "reordered", "renamed", "size", "invented"])
@pytest.mark.parametrize("backend", ["agent", "openai"])
def test_same_pdf_receipt_cannot_omit_or_change_configured_evidence(tmp_path: Path, defect: str, backend: str) -> None:
    env = _Env(tmp_path)
    evidence = []
    for index in range(2):
        path = tmp_path / f"evidence_{index}.json"
        path.write_text(json.dumps({"candidate": index}), encoding="utf-8")
        evidence.append(path)
    (env.paper_dir / "ai_review.json").write_text('{"Decision": "Accept"}')

    def execute(command, *_args):
        payload = _review_for_packet(command)
        inputs = payload["Provenance"]["inputs"]
        if defect == "missing":
            del inputs[1:]
        elif defect == "hash":
            inputs[1]["sha256"] = "0" * 64
        elif defect == "reordered":
            inputs[1:] = inputs[:0:-1]
        elif defect == "renamed":
            inputs[1]["name"] = "old_candidate.json"
        elif defect == "size":
            inputs[1]["bytes"] += 1
        elif defect == "invented":
            inputs[1]["id"] = "unconfigured_source"
            payload["Evidence References"][0]["source"] = "unconfigured_source"
        return _return_review(command, payload)

    result = env.run(execute, reviewer_options={"backend": backend, "evidence_paths": evidence})
    assert result.status == "failed" and "configured evidence" in result.error
    assert result.review_path is None
    assert not (env.paper_dir / "ai_review.json").exists()
    assert env.manifest()["ai_review"]["status"] == "failed"


@pytest.mark.parametrize("target", ["source", "snapshot", "paper"])
@pytest.mark.parametrize("backend", ["agent", "openai"])
def test_input_changes_during_review_fail_before_publication(tmp_path: Path, target: str, backend: str) -> None:
    env = _Env(tmp_path)
    evidence = tmp_path / "selected_novelty.json"
    evidence.write_text('{"selected_candidate": "current"}')

    def execute(command, *_args):
        payload = _review_for_packet(command)
        changed = evidence if target == "source" else (
            Path(command[command.index("--output") - 1]) if target == "paper" else _supplied_evidence(command)[0]
        )
        changed.write_bytes(b"changed after the packet was read")
        return _return_review(command, payload)

    result = env.run(execute, reviewer_options={"backend": backend, "evidence_paths": [evidence]})
    assert result.status == "failed" and f"changed during {backend} review" in result.error
    assert not (env.paper_dir / "ai_review.json").exists()


@pytest.mark.parametrize("unsafe", ["missing", "file_symlink", "directory_symlink"])
@pytest.mark.parametrize("backend", ["agent", "openai"])
def test_unsafe_evidence_fails_before_invocation(tmp_path: Path, unsafe: str, backend: str) -> None:
    env = _Env(tmp_path)
    source_dir = tmp_path / "actual"
    source_dir.mkdir()
    source = source_dir / "selected_novelty.json"
    source.write_text("{}")
    evidence = tmp_path / "missing.json"
    if unsafe == "file_symlink":
        evidence.symlink_to(source)
    elif unsafe == "directory_symlink":
        link = tmp_path / "alias"
        link.symlink_to(source_dir, target_is_directory=True)
        evidence = link / source.name

    result = env.run(
        lambda *_args: pytest.fail("unsafe evidence must fail before provider invocation"),
        reviewer_options={"backend": backend, "evidence_paths": [evidence]},
    )
    assert result.status == "failed"
    assert not (env.paper_dir / "ai_review.json").exists()


@pytest.mark.parametrize("backend", ["agent", "openai"])
def test_evidence_builder_reads_recovered_paper_data_without_mutating_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    run_dir = tmp_path / "run"
    paper_dir = run_dir / "paper"
    replacement = run_dir / ".paper.revision-review"
    for directory, candidate in ((paper_dir, "before"), (replacement, "after")):
        write_workspace(directory, title=candidate)
        data_path = directory / "paper_data.json"
        data = json.loads(data_path.read_text())
        data["generated_at"] = candidate
        data_path.write_text(json.dumps(data))
        _write_manifest(directory / "latest_manifest.json", paper_dir / "paper.pdf")
        refresh_evidence_audit(directory)
    _interrupt_publication(monkeypatch, paper_dir, replacement)
    original_evidence = tmp_path / "metrics.json"
    original_evidence.write_text('{"seeds": 1}')
    options = {"backend": backend, "evidence_paths": [original_evidence]}
    observed = []

    def build_evidence():
        data = json.loads((paper_dir / "paper_data.json").read_text())
        observed.append(data["generated_at"])
        evidence = paper_dir / "selected_novelty.json"
        evidence.write_text(json.dumps({"paper_generation": data["generated_at"]}))
        return evidence

    def execute(command, *_args):
        supplied = _supplied_evidence(command)
        assert len(supplied) == 2
        assert json.loads(supplied[1].read_text())["paper_generation"] == "after"
        return _return_review(command, _review_for_packet(command))

    result = run_final_ai_review(
        repo_root=_repo_root(tmp_path), paper_dir=paper_dir, paper_root=paper_dir,
        results_dir=run_dir, execute_command=execute, reviewer_options=options,
        reviewer_evidence_builder=build_evidence,
    )
    assert result.status == "completed"
    assert observed == ["after"]
    assert options == {"backend": backend, "evidence_paths": [original_evidence]}
    assert not (run_dir / ".paper.revision-publication.json").exists()
