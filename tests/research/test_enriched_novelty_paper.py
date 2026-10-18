"""Real selected-review chains reach paper context and bibliography verification."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from unittest.mock import Mock

import pytest

from src.investigation.native_harness import run_native_evaluation
from src.investigation.paper_prompt_context import paper_workspace_context_replacements
from src.orchestrator.paper_data import FinalEvaluationMode, build_paper_data
from src.research import paper_references as refs
from src.research.config import ResearchConfig, write_config
from src.research.evidence import verify_selected_novelty_for_paper
from src.research.harness_evidence import build_candidate_packet
from src.research.novelty import evaluate_novelty
from src.research.review_enrichment import ensure_selected_novelty
from src.research.workspace import prepare
from tests.program_testutils import make_program_dir
from tests.research import test_novelty_gate as fixtures
from tests.research.test_review_enrichment import bytes_below, install_models, install_retrieval, read, write
from tests.research.test_selected_novelty_paper import stage_paper

study = fixtures.study


@pytest.fixture
def enriched_study(study, tmp_path, monkeypatch):
    task, literature, agenda, review = study
    cfg = ResearchConfig(literature=literature)
    write_config(task / "research.yaml", cfg)
    (task / "evaluate.py").write_text('''import argparse, json, pathlib, runpy
p=argparse.ArgumentParser()
p.add_argument('--program_path');p.add_argument('--results_dir')
a=p.parse_args();out=pathlib.Path(a.results_dir)
score=runpy.run_path(a.program_path)['SCORE']
(out/'metrics.json').write_text(json.dumps({'combined_score':score}))
(out/'correct.json').write_text(json.dumps({'correct':True}))
''')
    # Preparation uses the explicit local source. Every online transport used
    # later is replaced below; unexpected preparation retrieval fails the test.
    monkeypatch.setattr("src.research.literature.search_literature", Mock(side_effect=AssertionError("unexpected preparation search")))
    monkeypatch.setattr("src.research.literature.download", Mock(side_effect=AssertionError("unexpected preparation download")))
    workspace = tmp_path / "managed"
    initial = {"schema_version": 1, "selected_review_enrichment_version": 1, "quality_target": "development"}
    manifest = prepare(task, workspace, offline=False, overrides=[], initial_state=initial,
        after_literature=lambda staged, _cfg: {
            "research_agenda.json": json.dumps(agenda),
            "research_context/harness_novelty.json": json.dumps(evaluate_novelty(staged, agenda)),
        })
    write(workspace / "harness_state.json", {**initial, "input_task_sha256": manifest["input_task_sha256"]})
    shutil.copytree(workspace / "input_task", workspace / "run/task")
    investigation = workspace / "run/investigation"
    idea = investigation / "ideas/idea_1"
    program = idea / "program_1"
    make_program_dir(program, content="SCORE = 0.7\n")
    write(idea / "idea_bundle.json", {"idea_id": "idea_1", "research_basis": {
        "status": "needs_verification", "contribution": "A bounded development study",
        "alternative_explanation": "A changed budget could explain the observation",
        "discriminating_test": "Match the budget and compare paired seeds", "closest_work": [],
    }})
    assert run_native_evaluation(program_dir=program, evaluator=workspace / "run/task/evaluate.py",
        python_bin=sys.executable, timeout_seconds=10, environment={},
        attempt_id="idea_1_attempt_1", idea_id="idea_1") == 0
    write(investigation / "best_program_manifest.json", {
        "program_dir": str(program), "entrypoint_path": "src/program.py",
        "source_attempt_id": "idea_1_attempt_1", "combined_score": 0.7, "correct": True,
    })
    packet = build_candidate_packet(workspace)
    actual = fixtures.candidate_review(review, packet)
    actual["candidate_assessment"]["claim_code_evidence_map"][0]["finding"] = (
        "The source declares SCORE=0.7 and its development receipt records 0.7; confirmation remains unresolved."
    )
    calls = install_models(monkeypatch, agenda, actual)
    searches = install_retrieval(monkeypatch)
    pointer = ensure_selected_novelty(workspace, cfg, packet=packet,
                                      author_settings=None, reviewer_settings=None)
    assert pointer["status"] == "supported" and pointer["cycle"].endswith("cycle_1")
    write(workspace / "novelty/latest.json", pointer)
    data = build_paper_data(investigation, final_evaluation_mode=FinalEvaluationMode.OMIT)
    assert data["selected_novelty_review"]["verification_status"] == "host_verified", data["selected_novelty_review"]
    return workspace, cfg, packet, pointer, data, calls, searches


def test_enriched_primary_text_reaches_actual_paper_projection_and_prompt(enriched_study, tmp_path):
    workspace, cfg, packet, pointer, data, calls, searches = enriched_study
    selected = data["selected_novelty_review"]
    assert selected["status"] == "supported"
    assert selected["provenance"]["evidence_reference"] == pointer["evidence_reference"]
    evidence = selected["literature_evidence"]
    assert len(evidence["sources"]) == 1
    source = evidence["sources"][0]
    assert source["title"] == "Matched Adaptive Steps Study 1"
    assert source["availability"] == "full_text" and source["passages"]
    text = (workspace / source["saved_text"]).read_text()
    assert "matched optimization budgets" in text
    assert hashlib.sha256(text.encode()).hexdigest() == source["text_sha256"]
    assert verify_selected_novelty_for_paper(data) == selected
    paper = stage_paper(tmp_path, workspace, data)
    context = paper_workspace_context_replacements(paper)["RESEARCH_CONTEXT"]
    selected_context = context.split("SELECTED-METHOD NOVELTY REVIEW FEEDBACK:", 1)[1]
    assert '"literature_evidence"' in selected_context
    assert source["title"] in selected_context and "matched optimization budgets" in selected_context
    assert '"evidence_reference"' in selected_context and source["id"] in selected_context
    before = bytes_below(workspace / "novelty")
    write(workspace / "run/final_evaluation/invocation.json", {})
    assert ensure_selected_novelty(workspace, cfg, packet=packet,
        author_settings=None, reviewer_settings=None) == pointer
    assert verify_selected_novelty_for_paper(data) == selected
    refreshed = build_paper_data(workspace / "run/investigation", final_evaluation_mode=FinalEvaluationMode.OMIT)
    assert refreshed["selected_novelty_review"] == selected
    assert bytes_below(workspace / "novelty") == before
    assert len(calls) == 10 and len(searches) == 1


@pytest.mark.parametrize("change", ["source", "ancestor_receipt", "pointer"])
def test_enriched_paper_tamper_downgrades_without_paid_calls(enriched_study, tmp_path, change):
    workspace, _cfg, _packet, pointer, data, calls, searches = enriched_study
    source = data["selected_novelty_review"]["literature_evidence"]["sources"][0]
    if change == "source":
        (workspace / source["saved_text"]).write_text("changed full text")
    elif change == "ancestor_receipt":
        path = workspace / pointer["evidence_reference"]["chain"] / "cycle_0/reviewer_1/receipt.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        changed = read(workspace / "novelty/latest.json")
        changed["evidence_reference"]["evidence_task"] = "../../outside"
        write(workspace / "novelty/latest.json", changed)
    failed = verify_selected_novelty_for_paper(data)
    assert failed["status"] == "needs_evidence" and failed["verification_status"] == "invalid_or_missing"
    assert failed["gaps"] and "literature_evidence" not in failed
    context = paper_workspace_context_replacements(stage_paper(tmp_path, workspace, data))["RESEARCH_CONTEXT"]
    selected_context = context.split("SELECTED-METHOD NOVELTY REVIEW FEEDBACK:", 1)[1]
    assert '"status": "needs_evidence"' in selected_context
    assert source["title"] not in selected_context
    assert len(calls) == 10 and len(searches) == 1


def test_actual_enriched_projection_supplies_bibliography_and_validates_before_cache(enriched_study, tmp_path, monkeypatch):
    workspace, _cfg, _packet, pointer, data, calls, searches = enriched_study
    metadata_calls = []
    def source_metadata(record, _cfg, output):
        metadata_calls.append(record["url"])
        raw = json.dumps({"title": record["title"], "authors": ["Verified Primary Author"], "year": 2026}).encode()
        name = hashlib.sha256(raw).hexdigest() + ".json"
        (output / name).write_bytes(raw)
        return {"title": record["title"], "authors": ["Verified Primary Author"], "year": 2026,
                "url": record["url"], "metadata_status": "source_metadata", "entry_type": "article",
                "metadata_file": name, "metadata_sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(refs, "source_metadata", source_metadata)
    monkeypatch.setattr(refs, "search_literature", Mock(side_effect=AssertionError("adopted full text already supplies the minimum")))
    output = tmp_path / "references"
    output.mkdir()
    kwargs = {"queries": ["matched adaptive steps"], "min_refs": 1, "max_refs": 1,
              "plan_metadata": {"queries": ["matched adaptive steps"]}, "context": {}, "paper_data": data}
    assert refs.write_references_for_plan(workspace / "run/task", output, **kwargs) == 1
    source = data["selected_novelty_review"]["literature_evidence"]["sources"][0]
    provenance = read(output / "references_provenance.json")
    assert provenance["selected_literature_evidence"]["adopted_source_ids"] == [source["id"]]
    assert provenance["selected_literature_evidence"]["evidence_reference"] == pointer["evidence_reference"]
    assert "Matched Adaptive Steps Study 1" in (output / "references.bib").read_text()
    assert "Verified Primary Author" in (output / "references.bib").read_text()
    assert refs.write_references_for_plan(workspace / "run/task", output, **kwargs) == 1
    assert metadata_calls == [source["verified_origin_url"]]
    (workspace / source["saved_text"]).write_text("changed after bibliography cache publication")
    with pytest.raises(ValueError, match="stale|verified"):
        refs.write_references_for_plan(workspace / "run/task", output, **kwargs)
    assert metadata_calls == [source["verified_origin_url"]] and len(calls) == 10 and len(searches) == 1
