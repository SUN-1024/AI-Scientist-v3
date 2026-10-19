"""Reviewed-gap retrieval never turns model leads or failed calls into evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil

import pytest

from src.research import review_retrieval as rr
from src.research.novelty import content_hash
from src.research.novelty_review import review_novelty
from src.research.workspace import task_digest
from tests.research import test_novelty_gate as fixtures

study = fixtures.study


def write(path, value):
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def reviewed(study, tmp_path, monkeypatch):
    source, cfg, agenda, review = study
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task = workspace / "input_task"
    shutil.copytree(source, task)
    review = copy.deepcopy(review)
    review["verdict"] = "needs_evidence"
    review["checks"]["closest_work_adequate"] = "uncertain"
    review["next_actions"] = [{"kind": "retrieve_literature", "reason": "Retrieve the primary paper Accurate Adaptive Optimization and its full text."},
                              {"kind": "inspect_full_text", "reason": "Inspect task.md and host allocation."},
                              {"kind": "run_discriminating_experiment", "reason": "Run a matched comparison."}]
    fixtures.install_provider(monkeypatch, [review])
    cycle = workspace / "novelty/initial"
    report = review_novelty(task, agenda, cfg, cycle)
    assert report["status"] == "needs_evidence"
    assert report["machine_checks"]["passed"]
    return workspace, task, cfg, agenda, review, cycle


def proposal(number=1, **overrides):
    return {"title": f"Accurate Adaptive Optimization Method {number}",
            "url": f"https://arxiv.org/abs/2601.0000{number}",
            "openAccessPdf": {"url": f"https://arxiv.org/pdf/2601.0000{number}"},
            "metadata_status": "agent_suggested_unverified", **overrides}


def mocks(monkeypatch, records=None, *, fail_download=False):
    records = records or [proposal()]
    searches, downloads = [], []
    def search(cfg, output):
        searches.append(copy.deepcopy(cfg))
        write(output / "agent_response.json", records)
        return copy.deepcopy(records), []
    def download(url, cfg):
        downloads.append(url)
        if fail_download:
            raise ValueError("Network unavailable")
        return b"%PDF-" + url.encode()
    def extract(body, *, pdf, provenance=None):
        number = body.decode()[-1]
        return f"Accurate Adaptive Optimization Method {number}\nFull methods and evaluation.", "full_text"
    monkeypatch.setattr(rr, "download", download)
    monkeypatch.setattr(rr, "_extract_text", extract)
    return search, searches, downloads


def invoke(reviewed, search, **kwargs):
    workspace, task, cfg, _agenda, _review, cycle = reviewed
    return rr.enrich_review_evidence(workspace, task, cycle, cfg,
                                    journal_dir=workspace / "novelty/enrichment", base_input_sha256=task_digest(workspace / "input_task"),
                                    search_backend=search, search_backend_id="fixture-proposals-v1", **kwargs)


def test_verified_requests_ignore_non_retrieval_actions_and_deduplicate(reviewed):
    _workspace, task, _cfg, _agenda, _review, cycle = reviewed
    packet = rr.extract_review_requests(task, cycle)
    assert len(packet["requests"]) == 1
    assert len(packet["receipts"]) == 2
    assert packet["skipped"][0]["reason"] == "not_a_paper_retrieval_request"
    assert packet["binding"]["parent_report_sha256"] == hashlib.sha256((cycle / "novelty_report.json").read_bytes()).hexdigest()


def test_unverified_or_changed_review_does_not_issue_queries(reviewed, monkeypatch):
    search, calls, downloads = mocks(monkeypatch)
    cycle = reviewed[-1]
    (cycle / "reviewer_1/assessment.json").write_text("{}")
    result = invoke(reviewed, search)
    assert result["status"] == "stopped"
    assert result["reason"] == "review_integrity_or_structure_gap"
    assert not calls and not downloads


def test_snapshot_retains_all_old_sources_and_seals_raw_bytes(reviewed, monkeypatch):
    workspace, task, cfg, agenda, review, cycle = reviewed
    catalog = read(task / "research_context/catalog.json")
    for index in range(10):
        row = {**catalog["sources"][0], "id": f"old_{index}", "passages": []}
        catalog["sources"].append(row)
        agenda["source_selection"].append({"source_id": f"old_{index}", "relation": "relevant", "reason": "Retain this additional baseline source."})
    write(task / "research_context/catalog.json", catalog)
    # Generate receipts for the expanded parent, rather than changing a reviewed catalog.
    cycle = workspace / "novelty/expanded"
    fixtures.install_provider(monkeypatch, [review])
    review_novelty(task, agenda, cfg, cycle)
    reviewed = workspace, task, cfg, agenda, review, cycle
    (task / "dataset.bin").write_bytes(b"private dataset bytes must not be projected")
    before = task_digest(task)
    old_bytes = (task / "research_context/catalog.json").read_bytes()
    search, calls, downloads = mocks(monkeypatch)
    result = invoke(reviewed, search)
    evidence = rr.validate_enrichment_result(workspace, result)
    assert result["status"] == "enriched"
    assert len(calls) == len(downloads) == 1
    assert task_digest(task) == before
    assert not (evidence / "dataset.bin").exists()
    after = read(evidence / "research_context/catalog.json")
    assert after["sources"][:-1] == catalog["sources"]
    assert len(after["sources"]) == 12  # cfg.max_papers=8 must never truncate inherited sources.
    assert (evidence / "research_context/catalog_history" / (hashlib.sha256(old_bytes).hexdigest() + ".json")).read_bytes() == old_bytes
    new = after["sources"][-1]
    assert new["availability"] == "full_text"
    assert new["discovery_status"] == "agent_suggested_unverified"
    assert hashlib.sha256((evidence / "research_context" / new["raw_path"]).read_bytes()).hexdigest() == new["download_sha256"]
    assert rr.load_enrichment_result(workspace, workspace / "novelty/enrichment", base_input_sha256=before,
                                     expected_result_sha256=content_hash(result)) == result


def test_completed_round_replays_without_network_and_detects_tampering(reviewed, monkeypatch):
    workspace, _task, *_ = reviewed
    search, calls, downloads = mocks(monkeypatch)
    result = invoke(reviewed, search)
    assert invoke(reviewed, search) == result
    assert len(calls) == len(downloads) == 1
    path = workspace / result["evidence_task"] / "research_context/catalog.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        invoke(reviewed, search)
    assert len(calls) == len(downloads) == 1


def test_resume_downloaded_bytes_after_parse_interruption(reviewed, monkeypatch):
    search, calls, downloads = mocks(monkeypatch)
    original = rr._extract_text
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(rr, "_extract_text", interrupted)
    with pytest.raises(KeyboardInterrupt):
        invoke(reviewed, search)
    monkeypatch.setattr(rr, "_extract_text", original)
    assert invoke(reviewed, search)["status"] == "enriched"
    assert len(calls) == len(downloads) == 1


def test_ambiguous_search_is_not_automatically_repeated(reviewed, monkeypatch):
    _unused, calls, downloads = mocks(monkeypatch)
    def interrupted(cfg, output):
        calls.append(cfg)
        (output / "provider_response.json").write_text('{"possibly_successful":true}')
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        invoke(reviewed, interrupted)
    result = invoke(reviewed, interrupted)
    assert result["status"] == "stopped"
    assert len(calls) == 1 and not downloads
    assert (reviewed[0] / "novelty/enrichment/round_01/search_01/provider_response.json").exists()


@pytest.mark.parametrize("failure", ["partial", "wrong_title", "html", "untrusted_host"])
def test_unusable_primary_text_never_counts_as_enrichment(reviewed, monkeypatch, failure):
    rows = [proposal(openAccessPdf={"url": "https://untrusted.example/paper.pdf"})] if failure == "untrusted_host" else None
    search, calls, downloads = mocks(monkeypatch, rows)
    if failure == "partial":
        monkeypatch.setattr(rr, "_extract_text", lambda *a, **k: ("Accurate Adaptive Optimization Method 1", "full_text_partial"))
    if failure == "wrong_title":
        monkeypatch.setattr(rr, "_extract_text", lambda *a, **k: ("Some other completely different paper", "full_text"))
    if failure == "html":
        monkeypatch.setattr(rr, "download", lambda *a, **k: b"<html>Access denied</html>")
    result = invoke(reviewed, search)
    assert result["status"] == "stopped"
    assert result["reason"] == "no_new_verified_full_text"
    assert not (reviewed[0] / "novelty/enrichment/round_01/evidence_task").exists()
    if failure == "untrusted_host":
        assert not downloads


def test_changed_budget_or_backend_cannot_reset_cumulative_allowance(reviewed, monkeypatch):
    search, _calls, _downloads = mocks(monkeypatch)
    invoke(reviewed, search)
    with pytest.raises(ValueError, match="policy/configuration"):
        invoke(reviewed, search, budget=rr.RetrievalBudget(max_total_search_calls=100))


def test_download_byte_budget_is_reserved_before_network(reviewed, monkeypatch):
    search, calls, downloads = mocks(monkeypatch)
    result = invoke(reviewed, search, budget=rr.RetrievalBudget(max_total_download_bytes=10))
    assert result["status"] == "stopped"
    assert len(calls) == 1 and not downloads


def test_round_budget_is_cumulative_across_new_review_cycles(reviewed, monkeypatch):
    workspace, task, cfg, agenda, review, _cycle = reviewed
    search, calls, downloads = mocks(monkeypatch)
    budget = rr.RetrievalBudget(max_rounds=1)
    first = invoke(reviewed, search, budget=budget)
    new_task = workspace / first["evidence_task"]
    new_cycle = workspace / "novelty/second_review"
    next_review = copy.deepcopy(review)
    next_review["next_actions"] = [{"kind": "retrieve_literature", "reason": "Retrieve a second exact primary paper to resolve remaining coverage."}]
    fixtures.install_provider(monkeypatch, [next_review])
    review_novelty(new_task, agenda, cfg, new_cycle)
    result = invoke((workspace, new_task, cfg, agenda, next_review, new_cycle), search, budget=budget)
    assert result["reason"] == "round_budget_exhausted"
    assert len(calls) == len(downloads) == 1


def test_nonconfigured_search_and_offline_stop_honestly(reviewed, monkeypatch):
    workspace, task, cfg, _agenda, _review, cycle = reviewed
    cfg.model_search.model = ""
    result = rr.enrich_review_evidence(workspace, task, cycle, cfg, journal_dir=workspace / "novelty/no_backend",
                                      base_input_sha256=task_digest(task))
    assert result["reason"] == "search_not_configured"
    search, calls, downloads = mocks(monkeypatch)
    assert invoke(reviewed, search, offline=True)["reason"] == "offline_retrieval_disabled"
    assert not calls and not downloads


def test_snapshot_rejects_symlink_sources(reviewed, monkeypatch, tmp_path):
    search, calls, downloads = mocks(monkeypatch)
    outside = tmp_path / "outside"
    outside.write_text("outside")
    (reviewed[1] / "research_context/escape").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        invoke(reviewed, search)
    assert not calls and not downloads


def test_readonly_delivery_rechecks_parent_receipts(reviewed, monkeypatch):
    workspace, task, _cfg, _agenda, _review, cycle = reviewed
    search, _calls, _downloads = mocks(monkeypatch)
    result = invoke(reviewed, search)
    path = cycle / "reviewer_2/receipt.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="binding"):
        rr.load_enrichment_result(workspace, workspace / "novelty/enrichment", base_input_sha256=task_digest(task),
                                  expected_result_sha256=content_hash(result))


def test_context_limit_retains_fitting_additions_and_preserves_rejected_bytes(reviewed, monkeypatch):
    workspace = reviewed[0]
    search, calls, downloads = mocks(monkeypatch, [proposal(1), proposal(2), proposal(3)])
    original = rr._extract_text
    def extract(body, **kwargs):
        text, status = original(body, **kwargs)
        return (text + ("Methods. " * 5000 if body.endswith(b"2") else "")), status
    monkeypatch.setattr(rr, "_extract_text", extract)
    monkeypatch.setattr(rr, "MAX_PACKET_CHARS", 5000)
    result = invoke(reviewed, search)
    assert result["status"] == "enriched"
    assert len(result["new_source_ids"]) == 2
    journal = read(workspace / "novelty/enrichment/journal.json")
    rejected = journal["rounds"][0]["downloads"][1]
    assert rejected["status"] == "not_adopted_context_limit"
    assert (workspace / rejected["raw"]).is_file()
    assert (workspace / rejected["text"]).is_file()
    assert len(calls) == 1 and len(downloads) == 3
    assert invoke(reviewed, search) == result


def test_no_fitting_addition_stops_with_context_reason(reviewed, monkeypatch):
    search, _calls, _downloads = mocks(monkeypatch)
    monkeypatch.setattr(rr, "MAX_PACKET_CHARS", 10)
    result = invoke(reviewed, search)
    assert result["status"] == "stopped"
    assert result["reason"] == "source_packet_budget"


def test_native_search_requires_saved_source_url_receipts(reviewed, monkeypatch):
    workspace, task, cfg, _agenda, _review, cycle = reviewed
    _unused, _calls, downloads = mocks(monkeypatch)
    def search(search_cfg, output):
        (output / "model_search").mkdir()
        write(output / "model_search/trace.json", {"sources": [{"url": "https://arxiv.org/abs/2601.00001"}]})
        return [proposal(metadata_status="model_extracted_unverified")], []
    monkeypatch.setattr(rr, "search_literature", search)
    result = rr.enrich_review_evidence(workspace, task, cycle, cfg, journal_dir=workspace / "novelty/native",
                                      base_input_sha256=task_digest(task))
    assert result["status"] == "enriched"
    assert len(downloads) == 1


def test_search_result_tamper_never_reissues_network(reviewed, monkeypatch):
    workspace = reviewed[0]
    search, calls, downloads = mocks(monkeypatch)
    invoke(reviewed, search)
    response = workspace / "novelty/enrichment/round_01/search_01/agent_response.json"
    response.write_text("[]")
    with pytest.raises(ValueError, match="search result artifacts"):
        invoke(reviewed, search)
    assert len(calls) == len(downloads) == 1


def test_download_limit_is_cumulative_and_new_sources_are_not_retrieved_again(reviewed, monkeypatch):
    workspace, _task, cfg, agenda, review, _cycle = reviewed
    budget = rr.RetrievalBudget(max_total_download_calls=1)
    search, calls, downloads = mocks(monkeypatch)
    result = invoke(reviewed, search, budget=budget)
    new_task = workspace / result["evidence_task"]
    new_agenda = copy.deepcopy(agenda)
    new_agenda["source_selection"].append({"source_id": result["new_source_ids"][0], "relation": "relevant", "reason": "A new inspected source supports broader context."})
    next_review = copy.deepcopy(review)
    next_review["next_actions"] = [{"kind": "retrieve_literature", "reason": "Retrieve another primary paper about a different learning mechanism."}]
    next_cycle = workspace / "novelty/next_review"
    fixtures.install_provider(monkeypatch, [next_review])
    review_novelty(new_task, new_agenda, cfg, next_cycle)
    second_search, second_calls, second_downloads = mocks(monkeypatch, [proposal(2)])
    second = invoke((workspace, new_task, cfg, new_agenda, next_review, next_cycle), second_search, budget=budget)
    assert second["status"] == "stopped"
    assert len(calls) == len(downloads) == len(second_calls) == 1
    assert not second_downloads


def test_changed_base_input_is_rejected_before_calls(reviewed, monkeypatch):
    workspace, task, cfg, _agenda, _review, cycle = reviewed
    digest = task_digest(task)
    (task / "task.md").write_text("Changed experiment")
    search, calls, downloads = mocks(monkeypatch)
    with pytest.raises(ValueError, match="Base input changed"):
        rr.enrich_review_evidence(workspace, task, cycle, cfg, journal_dir=workspace / "novelty/base_changed",
                                  base_input_sha256=digest, search_backend=search, search_backend_id="fixture")
    assert not calls and not downloads


def test_journal_symlink_is_rejected_before_lock_or_external_calls(reviewed, monkeypatch, tmp_path):
    workspace, task, cfg, _agenda, _review, cycle = reviewed
    outside = tmp_path / "outside_directory"
    outside.mkdir()
    (workspace / "novelty/link").symlink_to(outside, target_is_directory=True)
    search, calls, downloads = mocks(monkeypatch)
    with pytest.raises(ValueError, match="symlink"):
        rr.enrich_review_evidence(workspace, task, cycle, cfg, journal_dir=workspace / "novelty/link/chain",
                                  base_input_sha256=task_digest(task), search_backend=search, search_backend_id="fixture")
    assert not calls and not downloads and not list(outside.iterdir())


def test_verified_review_with_missing_full_text_can_retrieve(reviewed, monkeypatch):
    workspace, task, cfg, agenda, review, _cycle = reviewed
    catalog = read(task / "research_context/catalog.json")
    catalog["sources"][0]["availability"] = "abstract"
    write(task / "research_context/catalog.json", catalog)
    agenda = copy.deepcopy(agenda)
    agenda["closest_work"] = []
    agenda["source_selection"][0].update(relation="relevant", reason="Only the abstract is available; retrieve complete primary evidence before comparison.")
    review = copy.deepcopy(review)
    review["source_findings"] = []
    cycle = workspace / "novelty/abstract_only"
    fixtures.install_provider(monkeypatch, [review])
    report = review_novelty(task, agenda, cfg, cycle)
    assert report["machine_checks"]["passed"] is False
    assert report["machine_checks"]["review_artifacts_valid"] is True
    assert {gap["code"] for gap in report["machine_checks"]["gaps"]} == {"missing_closest_work"}
    before = (cycle / "novelty_report.json").read_bytes()
    packet = rr.extract_review_requests(task, cycle)
    assert packet["reason"] is None
    assert packet["machine_gaps"] == report["machine_checks"]["gaps"]
    search, calls, downloads = mocks(monkeypatch)
    result = invoke((workspace, task, cfg, agenda, review, cycle), search)
    assert result["status"] == "enriched"
    assert len(calls) == len(downloads) == 1
    assert (cycle / "novelty_report.json").read_bytes() == before


def test_missing_passages_and_comparison_evidence_are_repairable_but_not_passed(reviewed, monkeypatch):
    workspace, task, cfg, agenda, review, _cycle = reviewed
    agenda = copy.deepcopy(agenda)
    agenda["closest_work"][0]["evidence"] = []
    cycle = workspace / "novelty/missing_passages"
    fixtures.install_provider(monkeypatch, [review])
    report = review_novelty(task, agenda, cfg, cycle)
    assert report["machine_checks"]["passed"] is False
    assert report["machine_checks"]["review_artifacts_valid"] is True
    assert {gap["code"] for gap in report["machine_checks"]["gaps"]} == {"missing_passages", "missing_comparison_evidence"}
    search, calls, downloads = mocks(monkeypatch)
    assert invoke((workspace, task, cfg, agenda, review, cycle), search)["status"] == "enriched"
    assert len(calls) == len(downloads) == 1
    assert read(cycle / "novelty_report.json")["machine_checks"]["passed"] is False


def test_altered_source_still_blocks_retrieval_before_calls(reviewed, monkeypatch):
    (reviewed[1] / "research_context/text/prior.txt").write_text("Altered source bytes")
    search, calls, downloads = mocks(monkeypatch)
    result = invoke(reviewed, search)
    assert result["reason"] == "review_integrity_or_structure_gap"
    assert not calls and not downloads


def test_non_literature_machine_gap_still_blocks_retrieval(reviewed, monkeypatch):
    workspace, task, cfg, agenda, review, _cycle = reviewed
    agenda = copy.deepcopy(agenda)
    agenda["proposed_contribution"] = ""
    cycle = workspace / "novelty/structural_gap"
    fixtures.install_provider(monkeypatch, [review])
    report = review_novelty(task, agenda, cfg, cycle)
    assert report["machine_checks"]["review_artifacts_valid"] is True
    search, calls, downloads = mocks(monkeypatch)
    result = invoke((workspace, task, cfg, agenda, review, cycle), search)
    assert result["reason"] == "review_integrity_or_structure_gap"
    assert not calls and not downloads


@pytest.mark.parametrize("deadline", [0, -1, True, "12345", 10 ** 400, float("nan"), float("inf"), -float("inf")])
def test_deadline_requires_finite_positive_absolute_timestamp(reviewed, monkeypatch, deadline):
    search, calls, downloads = mocks(monkeypatch)
    with pytest.raises(ValueError, match="deadline_unix"):
        invoke(reviewed, search, deadline_unix=deadline)
    assert not calls and not downloads


def test_expired_outer_deadline_does_not_start_new_search(reviewed, monkeypatch):
    search, calls, downloads = mocks(monkeypatch)
    monkeypatch.setattr(rr.time, "time", lambda: 1000.0)
    result = invoke(reviewed, search, deadline_unix=999.0)
    assert result["reason"] == "elapsed_budget_exhausted"
    assert not calls and not downloads


def test_outer_deadline_clamps_search_and_proxy_download_timeouts(reviewed, monkeypatch):
    workspace = reviewed[0]
    search, calls, downloads = mocks(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(rr.time, "time", lambda: now[0])
    def elapsed_search(cfg, output):
        result = search(cfg, output)
        now[0] += 2.5
        return result
    original_download = rr.download
    timeouts = []
    def bounded_download(url, cfg):
        timeouts.append(cfg.timeout_seconds)
        return original_download(url, cfg)
    monkeypatch.setattr(rr, "download", bounded_download)
    result = invoke(reviewed, elapsed_search, deadline_unix=1009.0)
    assert result["status"] == "enriched"
    assert calls[0].model_search.timeout_seconds == 9
    assert timeouts == [3]  # Two possible proxy/direct attempts fit in 6.5 seconds.
    journal = read(workspace / "novelty/enrichment/journal.json")
    assert journal["fixed"]["deadline_unix"] == 1009.0
    with pytest.raises(ValueError, match="changed on resume"):
        invoke(reviewed, elapsed_search, deadline_unix=2000.0)
    assert len(calls) == len(downloads) == 1
    now[0] = 2000.0
    assert invoke(reviewed, elapsed_search, deadline_unix=1009.0) == result
    assert len(calls) == len(downloads) == 1


def test_deadline_rechecked_after_search_before_any_download(reviewed, monkeypatch):
    search, calls, downloads = mocks(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(rr.time, "time", lambda: now[0])
    def elapsed_search(cfg, output):
        result = search(cfg, output)
        now[0] = 1005.5
        return result
    result = invoke(reviewed, elapsed_search, deadline_unix=1006.0)
    assert result["reason"] == "elapsed_budget_exhausted"
    assert len(calls) == 1 and not downloads


def test_native_campaign_timeout_reserves_possible_proxy_attempts(reviewed, monkeypatch):
    workspace, task, cfg, _agenda, _review, cycle = reviewed
    _unused, _calls, downloads = mocks(monkeypatch)
    timeouts = []
    monkeypatch.setattr(rr.time, "time", lambda: 1000.0)
    def search(search_cfg, output):
        timeouts.append(search_cfg.model_search.timeout_seconds)
        (output / "model_search").mkdir()
        write(output / "model_search/trace.json", {"sources": [{"url": "https://arxiv.org/abs/2601.00001"}]})
        return [proposal(metadata_status="model_extracted_unverified")], []
    monkeypatch.setattr(rr, "search_literature", search)
    result = rr.enrich_review_evidence(workspace, task, cycle, cfg, journal_dir=workspace / "novelty/native_deadline",
                                      base_input_sha256=task_digest(task), deadline_unix=1009.0)
    assert result["status"] == "enriched"
    assert timeouts == [1]  # At most four logical requests, each with proxy fallback.
    assert len(downloads) == 1


def test_conflicting_arxiv_landing_and_pdf_are_rejected_before_download(reviewed, monkeypatch):
    # Title and returned PDF would match paper B. That must not attach B to A's URL.
    search, calls, downloads = mocks(monkeypatch, [proposal(2, url="https://arxiv.org/abs/2601.00001")])
    result = invoke(reviewed, search)
    assert result["status"] == "stopped"
    assert len(calls) == 1 and not downloads
    journal = read(reviewed[0] / "novelty/enrichment/journal.json")
    reason = journal["rounds"][0]["searches"][0]["rejected_records"][0]["reason"]
    assert "Conflicting or unproven" in reason
    assert "2601.00001" in reason and "2601.00002" in reason


def test_same_arxiv_work_versions_preserve_downloaded_origin_identity(reviewed, monkeypatch):
    page = "https://arxiv.org/abs/2601.00001v1"
    pdf = "https://export.arxiv.org/pdf/2601.00001v2.pdf"
    search, calls, downloads = mocks(monkeypatch, [proposal(1, url=page, openAccessPdf={"url": pdf})])
    monkeypatch.setattr(rr, "_extract_text", lambda *args, **kwargs: ("Accurate Adaptive Optimization Method 1\nFull methods.", "full_text"))
    result = invoke(reviewed, search)
    assert result["status"] == "enriched"
    assert len(calls) == len(downloads) == 1
    catalog = read(reviewed[0] / result["evidence_task"] / "research_context/catalog.json")
    source = catalog["sources"][-1]
    assert source["url"] == page
    assert source["verified_origin_url"] == source["origin"] == pdf
    assert source["primary_identity"] == {"scheme": "arxiv", "value": "2601.00001", "version": "v2"}
    assert "cross-version content equivalence" in source["identity_check"]


@pytest.mark.parametrize("page,pdf,title,identity", [
    ("https://ojs.aaai.org/index.php/AAAI/article/view/9016",
     "https://ojs.aaai.org/index.php/AAAI/article/download/9016/8875",
     "Learning Instance Concepts from Multiple-Instance Data with Bags as Distributions",
     {"scheme": "aaai_ojs", "value": "AAAI/9016"}),
    ("https://jmlr.org/papers/v17/15-171.html",
     "https://jmlr.org/papers/volume17/15-171/15-171.pdf",
     "Multiple-Instance Learning from Distributions",
     {"scheme": "jmlr", "value": "17/15-171"}),
])
def test_primary_publisher_full_text_is_sealed_with_its_article_identity(reviewed, monkeypatch, page, pdf, title, identity):
    search, calls, downloads = mocks(monkeypatch, [proposal(title=title, url=page, openAccessPdf={"url": pdf})])
    monkeypatch.setattr(rr, "_extract_text", lambda *args, **kwargs: (title + "\nFull methods and evaluation.", "full_text"))
    result = invoke(reviewed, search)
    assert result["status"] == "enriched"
    assert len(calls) == 1 and downloads == [pdf]
    catalog = read(reviewed[0] / result["evidence_task"] / "research_context/catalog.json")
    source = catalog["sources"][-1]
    assert source["url"] == page
    assert source["verified_origin_url"] == source["origin"] == pdf
    assert source["primary_identity"] == identity
    assert source["discovery_status"] == "agent_suggested_unverified"
    assert invoke(reviewed, search) == result
    assert len(calls) == 1 and downloads == [pdf]


@pytest.mark.parametrize("page,pdf", [
    ("https://ojs.aaai.org/index.php/AAAI/article/view/9016",
     "https://ojs.aaai.org.example.com/index.php/AAAI/article/download/9016/8875"),
    ("https://jmlr.org/papers/v17/15-171.html",
     "https://jmlr.org.example.com/papers/volume17/15-171/15-171.pdf"),
])
def test_primary_publisher_lookalike_download_host_is_rejected(page, pdf):
    assert not rr.compatible_primary_urls(page, pdf)
    with pytest.raises(ValueError, match="not a recognized primary"):
        rr._primary_pdf(proposal(url=page, openAccessPdf={"url": pdf}), suggested=True, trace_sources=[])


@pytest.mark.parametrize("landing,pdf,scheme,value", [
    ("https://doi.org/10.1145/12345", "https://dl.acm.org/doi/pdf/10.1145/12345", "doi", "10.1145/12345"),
    ("https://link.springer.com/article/10.1007/ABC-123", "https://link.springer.com/content/pdf/10.1007/abc-123.pdf", "doi", "10.1007/abc-123"),
    ("https://proceedings.mlr.press/v80/ilse18a.html", "https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf", "mlr", "v80/ilse18a"),
    ("https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Example_CVPR_2023_paper.html", "https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Example_CVPR_2023_paper.pdf", "cvf", "content/cvpr2023/lin_example_cvpr_2023_paper"),
    ("https://openreview.net/forum?id=AbC123", "https://openreview.net/pdf?id=AbC123", "openreview", "AbC123"),
    ("https://arxiv.org/abs/hep-th/9901001v1", "https://arxiv.org/pdf/hep-th/9901001v3.pdf", "arxiv", "hep-th/9901001"),
    ("https://ojs.aaai.org/index.php/AAAI/article/view/9016", "https://ojs.aaai.org/index.php/AAAI/article/download/9016/8875", "aaai_ojs", "AAAI/9016"),
    ("https://ojs.aaai.org/index.php/AAAI/article/view/9016", "https://ojs.aaai.org/index.php/AAAI/article/view/9016/8875", "aaai_ojs", "AAAI/9016"),
    ("https://jmlr.org/papers/v17/15-171.html", "https://jmlr.org/papers/volume17/15-171/15-171.pdf", "jmlr", "17/15-171"),
    ("https://www.jmlr.org/papers/v17/15-171.html", "https://jmlr.org/papers/v17/15-171/15-171.pdf", "jmlr", "17/15-171"),
])
def test_known_landing_pdf_identifiers_agree(landing, pdf, scheme, value):
    assert rr.compatible_primary_urls(landing, pdf)
    identity = rr.primary_source_identity(pdf)
    assert (identity["scheme"], identity["value"]) == (scheme, value)


@pytest.mark.parametrize("landing,pdf", [
    ("https://doi.org/10.1145/11111", "https://dl.acm.org/doi/pdf/10.1145/22222"),
    ("https://proceedings.mlr.press/v80/ilse18a.html", "https://proceedings.mlr.press/v81/ilse18a/ilse18a.pdf"),
    ("https://openreview.net/forum?id=PaperA", "https://openreview.net/pdf?id=PaperB"),
    ("https://doi.org/10.1145/11111", "https://arxiv.org/pdf/2601.00001"),
    ("https://ojs.aaai.org/index.php/AAAI/article/view/9016", "https://ojs.aaai.org/index.php/AAAI/article/download/9017/8875"),
    ("https://ojs.aaai.org/index.php/AAAI/article/view/9016", "https://ojs.aaai.org/index.php/AIIDE/article/download/9016/8875"),
    ("https://jmlr.org/papers/v17/15-171.html", "https://jmlr.org/papers/volume18/15-171/15-171.pdf"),
    ("https://jmlr.org/papers/v17/15-171.html", "https://jmlr.org/papers/volume17/15-171/other-paper.pdf"),
])
def test_conflicting_or_unproven_cross_namespace_identifiers_fail_before_download(landing, pdf):
    assert not rr.compatible_primary_urls(landing, pdf)
    with pytest.raises(ValueError, match="Conflicting or unproven"):
        rr._primary_pdf(proposal(url=landing, openAccessPdf={"url": pdf}), suggested=True, trace_sources=[])
