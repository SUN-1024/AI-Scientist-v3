"""Adopted review sources reach bibliography metadata verification before cache use."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import shutil
from unittest.mock import Mock

import pytest

from src.research import paper_references as refs
from src.research.config import LiteratureConfig, ResearchConfig, load_config, write_config
from src.research.literature import passages, prepare_literature
from src.research.novelty import content_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def study(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    task = managed / "run/task"
    task.mkdir(parents=True)
    (task / "task.md").write_text("Compare complete methods under matched budgets.")
    sources = []
    for index in range(8):
        name = f"old_{index}"
        (task / (name + ".txt")).write_text(f"Old primary paper {index} full method.")
        sources.append({"id": name, "title": name, "path": name + ".txt",
                        "url": f"https://arxiv.org/abs/2401.1000{index}"})
    cfg = ResearchConfig(literature=LiteratureConfig(sources=sources))
    write_config(task / "research.yaml", cfg)
    base_catalog = prepare_literature(task, cfg.literature, offline=False)
    enriched = managed / "novelty/selected_enrichment/test/retrieval/round_01/evidence_task"
    shutil.copytree(task, enriched)
    body = "Newly adopted primary work: complete method and matched evaluation."
    text_path = enriched / "research_context/text/new_primary.txt"
    text_path.write_text(body)
    new = {"id": "new_primary", "title": "Publisher Metadata: 2601.12345!", "url": "https://arxiv.org/abs/2601.12345",
           "origin": "https://arxiv.org/pdf/2601.12345", "verified_origin_url": "https://arxiv.org/pdf/2601.12345",
           "year": 2099, "authors": ["Unverified model author"], "metadata_status": "agent_suggested_unverified",
           "availability": "full_text", "text_path": "text/new_primary.txt",
           "text_sha256": hashlib.sha256(body.encode()).hexdigest(), "passages": passages(body, "new_primary", [])}
    catalog = deepcopy(base_catalog)
    catalog["sources"].append(new)
    write(enriched / "research_context/catalog.json", catalog)
    packet = {"candidate_source_sha256": "candidate-fixture", "selection": {"program_relative_path": "run/program"}}
    reference = {"schema_version": 1, "kind": "selected_review_enrichment",
                 "evidence_task": enriched.relative_to(managed).as_posix(), "seal_sha256": "a" * 64}
    pointer = {"cycle": "novelty/selected_enrichment/test/cycle_1", "evidence_reference": reference}
    pointer_path = managed / "novelty/latest.json"
    write(pointer_path, pointer)
    selected = {"verification_status": "host_verified", "candidate_packet_sha256": content_hash(packet),
                "provenance": {"managed_workspace": str(managed), "evidence_reference": reference,
                               "pointer_sha256": hashlib.sha256(pointer_path.read_bytes()).hexdigest()}}
    data = {"selected_novelty_review": deepcopy(selected)}
    monkeypatch.setattr("src.research.evidence.verify_selected_novelty_for_paper", lambda _: deepcopy(selected))
    monkeypatch.setattr("src.research.harness_evidence.build_candidate_packet", lambda _: deepcopy(packet))

    def resolve(actual_workspace, actual_pointer, *, candidate_packet):
        assert actual_workspace == managed and actual_pointer == pointer and candidate_packet == packet
        return enriched

    monkeypatch.setattr("src.research.review_enrichment.resolve_selected_evidence_task", resolve)
    calls = []

    def metadata(record, _cfg, destination):
        calls.append(record["url"])
        title = "Publisher metadata " + record["url"].split("/")[-1]
        raw = json.dumps({"title": title, "author": "Primary Author"}).encode()
        name = hashlib.sha256(raw).hexdigest() + ".json"
        (destination / name).write_bytes(raw)
        return {"title": title, "authors": ["Primary Author"], "year": 2025,
                "url": record["url"], "metadata_status": "source_metadata", "entry_type": "article",
                "metadata_file": name, "metadata_sha256": hashlib.sha256(raw).hexdigest()}

    monkeypatch.setattr(refs, "source_metadata", metadata)
    monkeypatch.setattr(refs, "search_literature", Mock(side_effect=AssertionError("old sources already meet minimum")))
    output = tmp_path / "paper"
    output.mkdir()
    kwargs = {"queries": ["closest methods"], "min_refs": 2, "max_refs": 3,
              "plan_metadata": {"queries": ["closest methods"]}, "context": {"fixture": True}}
    return task, enriched, managed, data, output, kwargs, calls, base_catalog


def test_adopted_paper_is_considered_before_old_eight_even_when_minimum_is_met(study):
    task, _enriched, _managed, data, output, kwargs, calls, _catalog = study
    assert refs.write_references_for_plan(task, output, paper_data=data, **kwargs) == 3
    assert calls[0] == "https://arxiv.org/pdf/2601.12345"
    provenance = read(output / "references_provenance.json")
    binding = provenance["selected_literature_evidence"]
    assert binding["adopted_source_ids"] == ["new_primary"]
    assert binding["evidence_reference"] == data["selected_novelty_review"]["provenance"]["evidence_reference"]
    assert provenance["references"][0]["authors"] == ["Primary Author"]
    bib = (output / "references.bib").read_text()
    assert "Publisher metadata 2601.12345" in bib
    assert "Unverified model author" not in bib and "2099" not in bib
    assert refs.write_references_for_plan(task, output, paper_data=data, **kwargs) == 3
    assert len(calls) == 3


def test_conflicting_landing_and_pdf_are_rejected_and_landing_cannot_reenter(study):
    task, enriched, _managed, data, output, kwargs, calls, _catalog = study
    path = enriched / "research_context/catalog.json"
    catalog = read(path)
    landing = "https://arxiv.org/abs/2401.10000"
    catalog["sources"][-1]["url"] = landing
    write(path, catalog)
    refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    assert landing not in calls
    assert "https://arxiv.org/pdf/2601.12345" not in calls
    provenance = read(output / "references_provenance.json")
    assert "landing page" in provenance["retrieval_errors"][0]["error"]
    assert all(row["paper_id"] != landing for row in provenance["references"])


def test_different_publisher_title_is_rejected_without_landing_retry(study):
    task, enriched, _managed, data, output, kwargs, calls, _catalog = study
    path = enriched / "research_context/catalog.json"
    catalog = read(path)
    source = catalog["sources"][-1]
    source.update(url="https://arxiv.org/abs/2401.10000", origin="https://arxiv.org/pdf/2401.10000",
                  verified_origin_url="https://arxiv.org/pdf/2401.10000")
    write(path, catalog)
    refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    assert calls.count("https://arxiv.org/pdf/2401.10000") == 1
    assert "https://arxiv.org/abs/2401.10000" not in calls
    provenance = read(output / "references_provenance.json")
    assert "citation title" in provenance["retrieval_errors"][0]["error"]
    assert "Publisher metadata 2401.10000" not in (output / "references.bib").read_text()


@pytest.mark.parametrize("scheme", ["arxiv", "doi"])
def test_publisher_metadata_identifier_must_match_reviewed_origin(study, monkeypatch, scheme):
    task, enriched, _managed, data, output, kwargs, calls, _catalog = study
    if scheme == "doi":
        path = enriched / "research_context/catalog.json"
        catalog = read(path)
        catalog["sources"][-1].update(url="https://doi.org/10.1000/2601.12345",
            origin="https://doi.org/10.1000/2601.12345", verified_origin_url="https://doi.org/10.1000/2601.12345")
        write(path, catalog)
    original = refs.source_metadata

    def wrong_identifier(record, cfg, destination):
        result = original(record, cfg, destination)
        if "2601.12345" in record["url"]:
            result["url"] = ("https://arxiv.org/abs/2601.99999" if scheme == "arxiv"
                             else "https://doi.org/10.1000/other-paper")
        return result

    monkeypatch.setattr(refs, "source_metadata", wrong_identifier)
    refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    assert len(calls) == 4
    provenance = read(output / "references_provenance.json")
    assert "citation identifier" in provenance["retrieval_errors"][0]["error"]
    assert "Publisher metadata 2601.12345" not in (output / "references.bib").read_text()


@pytest.mark.parametrize("change", ["source", "pointer", "projection", "unverified", "outside"])
def test_invalid_declared_evidence_fails_before_network_or_cached_reuse(study, monkeypatch, change):
    task, enriched, managed, data, output, kwargs, calls, _catalog = study
    refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    before = len(calls)
    if change == "source":
        (enriched / "research_context/text/new_primary.txt").write_text("tampered source")
    elif change == "pointer":
        write(managed / "novelty/latest.json", {"evidence_reference": {"evidence_task": "../../outside"}})
    elif change == "projection":
        data["selected_novelty_review"]["candidate_packet_sha256"] = "changed"
    elif change == "unverified":
        monkeypatch.setattr("src.research.evidence.verify_selected_novelty_for_paper", lambda _: {"verification_status": "unverified"})
    else:
        monkeypatch.setattr("src.research.review_enrichment.resolve_selected_evidence_task", lambda *_a, **_kw: managed.parent)
    with pytest.raises(ValueError):
        refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    assert len(calls) == before


def test_rehashed_cache_cannot_replace_selected_catalog_binding(study):
    task, _enriched, _managed, data, output, kwargs, calls, _catalog = study
    refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    root = next((output / "reference_search").iterdir())
    path = root / "references_provenance.json"
    provenance = read(path)
    provenance["selected_literature_evidence"]["catalog_sha256"] = "b" * 64
    write(path, provenance)
    receipt = read(root / "receipt.json")
    receipt["references_provenance.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(root / "receipt.json", receipt)
    with pytest.raises(ValueError, match="binding changed"):
        refs.write_references_for_plan(task, output, paper_data=data, **kwargs)
    assert len(calls) == 3


def test_legacy_no_enrichment_keeps_fingerprint_and_zero_call_cache(study):
    task, _enriched, _managed, _data, output, kwargs, calls, catalog = study
    assert refs.write_references_for_plan(task, output, **kwargs) == 3
    expected = hashlib.sha256(json.dumps({"config": asdict(load_config(task).literature),
        "catalog": catalog, "plan": kwargs["plan_metadata"], "min_refs": kwargs["min_refs"],
        "max_refs": kwargs["max_refs"], "context": kwargs["context"], "queries": kwargs["queries"],
        "retrieval_policy": "bounded_supplement_v1"}, sort_keys=True).encode()).hexdigest()[:24]
    assert (output / "reference_search" / expected / "receipt.json").is_file()
    assert refs.write_references_for_plan(task, output, paper_data={"selected_novelty_review": {"provenance": {}}}, **kwargs) == 3
    assert len(calls) == 3
    assert "selected_literature_evidence" not in read(output / "references_provenance.json")


def test_adopted_sources_can_share_a_bibliography_with_local_catalog_origins(study, monkeypatch):
    task, _enriched, _managed, data, output, kwargs, calls, _catalog = study
    config = load_config(task)
    config.literature.sources = []
    monkeypatch.setattr(refs, "load_config", lambda _: config)
    assert refs.write_references_for_plan(task, output, paper_data=data, **kwargs) == 3
    assert len(calls) == 3
    provenance = read(output / "references_provenance.json")
    assert provenance["references"][0]["reviewed_source_id"] == "new_primary"
    assert len(provenance["references"]) == 3
