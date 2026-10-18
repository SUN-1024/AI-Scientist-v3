from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from src.research import paper_references as refs
from src.research.config import load_config, write_config
from src.research.literature import prepare_literature
from tests.research.test_unified_research import make_task


SOURCE_HTML = b'''<html><head>
<meta name="citation_title" content="A source title, not the model title">
<meta name="citation_author" content="Doe, Jane">
<meta name="citation_author" content="Smith, Alex">
<meta name="citation_date" content="2024/08/01">
<meta name="citation_journal_title" content="A source venue">
</head><body>Example primary source.</body></html>'''


def online_task(path, monkeypatch):
    task = make_task(path)
    cfg = load_config(task)
    cfg.literature.sources = []
    write_config(task / "research.yaml", cfg)
    monkeypatch.setattr(refs, "search_literature", lambda *_args: ([{
        "url": "https://arxiv.org/abs/2408.06292", "title": "An unverified model title", "year": 2099,
    }], []))
    monkeypatch.setattr(refs, "download", lambda *_args, **_kwargs: SOURCE_HTML)
    return task


def write(task, output, **kwargs):
    output.mkdir(exist_ok=True)
    return refs.write_references_for_plan(task, output, queries=["closest methods"], min_refs=1,
                                          max_refs=3, plan_metadata={"queries": ["closest methods"]}, context={}, **kwargs)


def test_writer_uses_primary_metadata_instead_of_model_claims_and_caches_it(tmp_path, monkeypatch):
    task = online_task(tmp_path / "task", monkeypatch)
    output = tmp_path / "paper"
    assert write(task, output) == 1
    bib = (output / "references.bib").read_text()
    assert "A source title" in bib and "Doe, Jane and Smith, Alex" in bib and "2024" in bib
    assert "2099" not in bib and "An unverified model title" not in bib
    provenance = json.loads((output / "references_provenance.json").read_text())
    row = provenance["references"][0]
    assert row["metadata_status"] == "source_metadata"
    assert (output / row["metadata_file"]).read_bytes() == SOURCE_HTML
    monkeypatch.setattr(refs, "search_literature", Mock(side_effect=AssertionError("cached references must not spend tokens")))
    monkeypatch.setattr(refs, "download", Mock(side_effect=AssertionError("cached references must not redownload")))
    assert write(task, output) == 1
    (output / row["metadata_file"]).write_text("changed")
    with pytest.raises(ValueError, match="metadata changed"):
        write(task, output)


def test_missing_authors_does_not_turn_model_metadata_into_bibtex(tmp_path, monkeypatch):
    task = online_task(tmp_path / "task", monkeypatch)
    monkeypatch.setattr(refs, "download", lambda *_args, **_kwargs: b'<meta name="citation_title" content="No author or year">')
    output = tmp_path / "paper"
    output.mkdir()
    (output / "references.bib").write_text("previous bibliography")
    with pytest.raises(ValueError, match="Only 0/1"):
        write(task, output)
    assert (output / "references.bib").read_text() == "previous bibliography"
    result = json.loads(next(output.glob("reference_search/*/attempt_*/result.json")).read_text())
    assert result["count"] == 0 and result["retrieval_errors"]


def test_doi_metadata_accept_header_and_safe_bibtex(tmp_path, monkeypatch):
    calls = []
    def download(url, _cfg, *, accept):
        calls.append((url, accept))
        return json.dumps({"title": "Title & 50% results", "author": [{"family": "Doe", "given": "Jane"}],
                           "issued": {"date-parts": [[2025, 2, 1]]}, "container-title": ["Journal"],
                           "type": "article-journal", "DOI": "10.1234/test"}).encode()
    monkeypatch.setattr(refs, "download", download)
    cfg = load_config(make_task(tmp_path / "task")).literature
    metadata = refs.source_metadata({"url": "https://doi.org/10.1234/test"}, cfg, tmp_path)
    _, bib = refs.bibliography_entry(metadata)
    assert calls == [("https://doi.org/10.1234/test", "application/vnd.citationstyles.csl+json")]
    assert r"Title \& 50\% results" in bib and "@article{" in bib


def test_offline_writer_uses_explicit_local_authors_without_any_network(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.literature.sources[0]["authors"] = ["Doe, Jane"]
    write_config(task / "research.yaml", cfg)
    prepare_literature(task, cfg.literature, offline=True)
    monkeypatch.setattr(refs, "search_literature", Mock(side_effect=AssertionError("offline")))
    monkeypatch.setattr(refs, "download", Mock(side_effect=AssertionError("offline")))
    output = tmp_path / "paper"
    assert write(task, output) == 1
    provenance = json.loads((output / "references_provenance.json").read_text())
    assert provenance["offline"] is True
    assert provenance["references"][0]["metadata_status"] == "user_supplied"


def test_real_paper_shell_retrieval_branch_never_calls_legacy_search(tmp_path, monkeypatch):
    task = online_task(tmp_path / "task", monkeypatch)
    output = tmp_path / "paper"
    output.mkdir()
    plan = output / "reference_query_plan.json"
    plan.write_text(json.dumps({"status": "ready", "central_topic": "Methods", "queries": [
        {"query": "closest methods", "category": "core_method", "rationale": "Find direct controls"},
    ]}))
    root = Path(__file__).resolve().parents[2]
    # Execute the production embedded Python block with only experiment-data loading stubbed.
    script = (root / "src/scripts/paper.sh").read_text().split("retrieve_references_from_plan() {", 1)[1]
    embedded = script.split("<<'PY'\n", 1)[1].split("\nPY\n}", 1)[0]
    monkeypatch.setattr("src.orchestrator.paper_data.build_paper_data", lambda *_a, **_k: {"generated_at": "fixture"})
    for name in ("semantic_scholar_key_forbidden_error", "retrieve_bibtex_entries"):
        monkeypatch.setattr("src.orchestrator.semantic_scholar." + name, Mock(side_effect=AssertionError("legacy search must not run")))
    monkeypatch.setattr(sys, "argv", ["-", str(root), str(task), str(tmp_path), str(tmp_path), str(tmp_path),
                                     str(output), "1", "3", "default", str(plan), "", "discover", "", "", "", "qualified_only"])
    with pytest.raises(SystemExit) as exited:
        exec(compile(embedded, "paper.sh:retrieve_references_from_plan", "exec"), {"__name__": "__main__"})
    assert exited.value.code == 0
    assert "A source title" in (output / "references.bib").read_text()


def test_paper_reference_count_failure_does_not_fall_back(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.literature.sources = []
    write_config(task / "research.yaml", cfg)
    search = Mock(return_value=([], [{"stage": "model_search", "error": "HTTP 429"}]))
    monkeypatch.setattr(refs, "search_literature", search)
    with pytest.raises(ValueError, match="No legacy API fallback"):
        write(task, tmp_path / "paper")
    assert search.call_count == 1  # Do not replay unavailable or ambiguous API requests.
    result = json.loads(next((tmp_path / "paper").glob("reference_search/*/attempt_*/result.json")).read_text())
    assert result["stop_reason"] == "search_unavailable"


def _mock_search_batches(monkeypatch, batches):
    calls = []
    def search(cfg, destination):
        rows = batches[len(calls)]
        calls.append(cfg)
        trace_dir = destination / "model_search"
        trace_dir.mkdir()
        (trace_dir / "trace.json").write_text(json.dumps({"papers": rows, "status": "ready"}))
        (trace_dir / "01_native_search.json").write_text(json.dumps({"sources": rows}))
        return rows, []
    monkeypatch.setattr(refs, "search_literature", search)
    return calls


def _paper(index):
    return {"url": f"https://publisher.example/paper/{index}", "title": f"Unverified title {index}"}


@pytest.mark.parametrize("overlap", [0, 5])
def test_bounded_supplement_reaches_25_verified_references_and_freezes_rounds(tmp_path, monkeypatch, overlap):
    task = online_task(tmp_path / "task", monkeypatch)
    calls = _mock_search_batches(monkeypatch, [
        [_paper(index) for index in range(20)],
        [_paper(index) for index in range(overlap)] + [_paper(index) for index in range(20, 30 - overlap)],
    ])
    monkeypatch.setattr(refs, "download", lambda url, *_args, **_kwargs:
                        SOURCE_HTML.replace(b"A source title, not the model title",
                                            f"Verified paper {url.rsplit('/', 1)[1]}".encode()))
    output = tmp_path / "paper"
    kwargs = dict(queries=["closest methods"], min_refs=25, max_refs=25,
                  plan_metadata={"queries": ["closest methods"]}, context={})
    assert refs.write_references_for_plan(task, output, **kwargs) == 25
    assert [call.max_papers for call in calls] == [20, 10]
    assert all(call.fetch_full_text is False for call in calls)
    assert "additional distinct" in calls[1].queries[-1]
    assert all(len(query) <= 500 for query in calls[1].queries)
    assert len(calls[1].queries) <= 8
    provenance = json.loads((output / "references_provenance.json").read_text())
    assert provenance["count"] == len(provenance["references"]) == 25
    assert len({row["title"] for row in provenance["references"]}) == 25
    assert provenance["status"] == "ready" and provenance["stop_reason"] == "minimum_met"
    assert [row["new_verified_references"] for row in provenance["search_rounds"]] == [20, 5]
    assert len({row["search_trace"] for row in provenance["search_rounds"]}) == 2
    receipt = json.loads(next(output.glob("reference_search/*/receipt.json")).read_text())
    assert len(receipt["search_artifacts"]) == 4
    monkeypatch.setattr(refs, "search_literature", Mock(side_effect=AssertionError("must use frozen receipt")))
    assert refs.write_references_for_plan(task, output, **kwargs) == 25
    trace = output / provenance["search_rounds"][1]["search_trace"]
    trace.write_text("changed search receipt")
    with pytest.raises(ValueError, match="search artifact changed"):
        refs.write_references_for_plan(task, output, **kwargs)


@pytest.mark.parametrize("duplicate_identity", ["title", "doi"])
def test_duplicate_source_metadata_exhausts_three_rounds_without_inflating_count(tmp_path, monkeypatch, duplicate_identity):
    task = online_task(tmp_path / "task", monkeypatch)
    calls = _mock_search_batches(monkeypatch, [[_paper(1)], [_paper(2)], [_paper(3)]])
    # Distinct URLs all expose the same primary-source title and author metadata.
    if duplicate_identity == "doi":
        monkeypatch.setattr(refs, "download", lambda url, *_args, **_kwargs:
                            SOURCE_HTML.replace(b"A source title, not the model title", url.encode()).replace(
                                b"</head>", b'<meta name="citation_doi" content="10.1234/shared"></head>'))
    output = tmp_path / "paper"
    output.mkdir()
    (output / "references.bib").write_text("previous bibliography")
    with pytest.raises(ValueError, match="Only 1/2.*3 search rounds.*request_budget_exhausted"):
        refs.write_references_for_plan(task, output, queries=["closest methods"], min_refs=2,
                                       max_refs=3, plan_metadata={}, context={})
    assert len(calls) == 3
    assert all(1 <= call.max_papers <= 3 for call in calls)
    result = json.loads(next(output.glob("reference_search/*/attempt_*/result.json")).read_text())
    assert result["count"] == 1 and result["status"] == "exhausted"
    assert [row["new_verified_references"] for row in result["search_rounds"]] == [1, 0, 0]
    assert not list(output.glob("reference_search/*/receipt.json"))
    assert (output / "references.bib").read_text() == "previous bibliography"


def test_third_round_repairs_missing_metadata_and_overlapping_results(tmp_path, monkeypatch):
    task = online_task(tmp_path / "task", monkeypatch)
    calls = _mock_search_batches(monkeypatch, [
        [_paper(0), _paper(1)], [_paper(1), _paper(2)], [_paper(3)],
    ])
    def source(url, *_args, **_kwargs):
        if url.endswith("/0"):
            return b'<meta name="citation_title" content="Authors and date are unavailable">'
        return SOURCE_HTML.replace(b"A source title, not the model title", url.encode())
    monkeypatch.setattr(refs, "download", source)
    output = tmp_path / "paper"
    assert refs.write_references_for_plan(task, output, queries=["closest methods"], min_refs=3,
                                          max_refs=4, plan_metadata={}, context={}) == 3
    result = json.loads((output / "references_provenance.json").read_text())
    assert len(calls) == 3 and result["stop_reason"] == "minimum_met"
    assert [row["verified_total"] for row in result["search_rounds"]] == [1, 2, 3]
    assert len(result["retrieval_errors"]) == 1


def test_online_explicit_sources_satisfy_minimum_without_search(tmp_path, monkeypatch):
    task = make_task(tmp_path / "task")
    cfg = load_config(task)
    cfg.literature.sources[0]["authors"] = ["Doe, Jane"]
    write_config(task / "research.yaml", cfg)
    monkeypatch.setattr(refs, "search_literature", Mock(side_effect=AssertionError("minimum already met")))
    monkeypatch.setattr(refs, "download", Mock(side_effect=AssertionError("explicit local metadata")))
    output = tmp_path / "paper"
    assert write(task, output) == 1
    result = json.loads((output / "references_provenance.json").read_text())
    assert result["search_rounds"] == [] and result["stop_reason"] == "minimum_met"
