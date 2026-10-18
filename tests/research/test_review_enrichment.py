"""Selected interpretation cycles retain all evidence and never reroll a verdict."""
from __future__ import annotations

import copy
import json
import threading

import pytest

from src.research import review_enrichment as selected
from src.research.config import ResearchConfig, write_config
from src.research.novelty import content_hash
from src.research.workspace import prepare
from tests.research import test_novelty_gate as fixtures

study = fixtures.study


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def bytes_below(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def managed(study, tmp_path):
    task, literature, agenda, review = study
    cfg = ResearchConfig(literature=literature)
    write_config(task / "research.yaml", cfg)
    workspace = tmp_path / "managed"
    initial = {"schema_version": 1, "selected_review_enrichment_version": 1}
    manifest = prepare(task, workspace, offline=True, overrides=[], initial_state=initial,
                       after_literature=lambda *_: {"research_agenda.json": json.dumps(agenda)})
    write(workspace / "harness_state.json", {**initial, "input_task_sha256": manifest["input_task_sha256"]})
    packet = fixtures.candidate_packet()
    packet["research_basis"] = {"contribution": "Adaptive steps with matched compute", "status": "needs_verification"}
    return workspace, cfg, agenda, fixtures.candidate_review(review, packet), packet


def install_models(monkeypatch, agenda, review, *, terminal_at=1, failed_author_at=None,
                   failed_reviewer_at=None, action="retrieve_literature", verdict="needs_evidence", done=False):
    calls = []
    lock = threading.Lock()
    monkeypatch.setenv("AGSCI_LITERATURE_API_KEY", "fixture-key")

    def provider(_url, body, _headers, **_kwargs):
        prompt = body["messages"][0]["content"]
        reviewer = "\n\nFROZEN INPUT DATA:\n" in prompt
        marker = "\n\nFROZEN INPUT DATA:\n" if reviewer else "\n\nINPUT DATA:\n"
        payload, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
        cycle = len(payload["literature"]["sources"]) - 1
        with lock:
            calls.append({"phase": "reviewer" if reviewer else "author", "cycle": cycle, "payload": payload})
        if not reviewer:
            if failed_author_at == cycle:
                raise ValueError("author transport failed")
            proposed = copy.deepcopy(agenda)
            for source in payload["literature"]["sources"][1:]:
                proposed["source_selection"].append({"source_id": source["id"], "relation": "relevant",
                    "reason": "Additional paper retains the unresolved matched-control concern."})
            if done and "AUTHOR REFLECTION:" in prompt:
                proposed["_reflection"] = {"done": True}
            value = proposed
        else:
            if failed_reviewer_at == cycle and "strongest prior-art collision" in prompt:
                raise ValueError("review transport failed")
            value = copy.deepcopy(review)
            if cycle < terminal_at:
                value["verdict"] = verdict
                value["next_actions"] = [{"kind": action,
                    "reason": f"Find the primary paper on matched adaptive steps round {cycle + 1}; control budget remains unresolved."}]
            else:
                value["next_actions"] = []
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr("src.research.model_search._http_json", provider)
    return calls


def install_retrieval(monkeypatch, *, available=True):
    calls = []

    def search(cfg, output):
        calls.append(cfg.queries)
        index = len(calls)
        url = f"https://arxiv.org/abs/2601.0000{index}"
        write(output / "model_search/trace.json", {"sources": [{"url": url}]})
        records = [{"title": f"Matched Adaptive Steps Study {index}", "url": url}]
        return records if available else [], []

    monkeypatch.setattr("src.research.review_retrieval.search_literature", search)
    monkeypatch.setattr("src.research.review_retrieval.download", lambda url, cfg: b"%PDF-1.7 " + url.encode())
    monkeypatch.setattr("src.research.review_retrieval._extract_text", lambda body, **_: (
        f"Matched Adaptive Steps Study {body.decode().split('0000')[-1]}\n" +
        "The paper studies matched optimization budgets and fixed data splits. " * 40, "full_text"))
    return calls


def run(managed, **kwargs):
    workspace, cfg, _agenda, _review, packet = managed
    return selected.ensure_selected_novelty(workspace, cfg, packet=packet,
                                            author_settings=None, reviewer_settings=None, **kwargs)


def chain_dir(managed):
    return managed[0] / "novelty/selected_enrichment" / content_hash(managed[-1])


def test_two_cycle_success_replay_and_paper_boundary_are_readonly(managed, monkeypatch):
    workspace, _cfg, agenda, review, packet = managed
    calls = install_models(monkeypatch, agenda, review)
    searches = install_retrieval(monkeypatch)
    original = bytes_below(workspace / "input_task")
    pointer = run(managed)
    assert pointer["status"] == "supported" and pointer["cycle"].endswith("cycle_1")
    assert len(calls) == 10 and len(searches) == 1
    assert original == bytes_below(workspace / "input_task")
    directory = chain_dir(managed)
    before = bytes_below(directory)
    assert run(managed) == pointer
    evidence = selected.resolve_selected_evidence_task(workspace, pointer, candidate_packet=packet)
    assert evidence == directory / "retrieval/round_01/evidence_task"
    assert before == bytes_below(directory)
    assert len(calls) == 10 and len(searches) == 1
    feedback = read(directory / "cycle_1/review_feedback.json")
    assert feedback["unresolved_requests"] == feedback["review_requests"]
    assert feedback["resolved_requests"] == [] and feedback["parent_report"]["status"] == "needs_evidence"
    assert len(feedback["new_source_ids"]) == 1
    assert all(call["payload"]["selected_candidate_packet"] == packet for call in calls)
    assert all("host_context" not in call["payload"] for call in calls)


@pytest.mark.parametrize("final_marker", ["invocation.json", "comparison_set.json"])
def test_final_blocks_new_calls_but_reuses_completed_cycle(managed, monkeypatch, final_marker):
    workspace, _cfg, agenda, review, packet = managed
    calls = install_models(monkeypatch, agenda, review, terminal_at=0)
    write(workspace / "run/final_evaluation" / final_marker, {})
    with pytest.raises(ValueError, match="Final evaluation"):
        run(managed)
    assert not calls and not chain_dir(managed).exists()
    (workspace / "run/final_evaluation" / final_marker).unlink()
    pointer = run(managed)
    before = bytes_below(chain_dir(managed))
    write(workspace / "run/final_evaluation" / final_marker, {})
    assert run(managed) == pointer
    assert selected.resolve_selected_evidence_task(workspace, pointer, candidate_packet=packet) == workspace / "input_task"
    assert bytes_below(chain_dir(managed)) == before and len(calls) == 5


@pytest.mark.parametrize("change", ["candidate", "source", "ancestor", "feedback", "pointer", "new_source", "symlink"])
def test_tamper_fails_without_paid_calls(managed, monkeypatch, change):
    workspace, _cfg, agenda, review, packet = managed
    calls = install_models(monkeypatch, agenda, review)
    searches = install_retrieval(monkeypatch)
    pointer = run(managed)
    directory = chain_dir(managed)
    if change == "candidate":
        packet["evaluation"]["metrics"]["combined_score"] = 99
    elif change == "source":
        path = workspace / "input_task/research_context/text/prior.txt"
        path.write_text(path.read_text() + "changed")
    elif change == "ancestor":
        path = directory / "cycle_0/reviewer_1/01_novelty_review.json"
        path.write_text(path.read_text() + " ")
    elif change == "feedback":
        path = directory / "cycle_1/review_feedback.json"
        value = read(path)
        value["unresolved_requests"] = []
        write(path, value)
    elif change == "pointer":
        pointer["evidence_reference"]["evidence_task"] = "../../other"
    elif change == "new_source":
        path = next((directory / "retrieval/round_01/evidence_task/research_context/text").glob("retrieved_*.txt"))
        path.write_text("changed")
    else:
        path = directory / "cycle_0/candidate_packet.json"
        body = path.read_bytes()
        path.unlink()
        external = workspace / "external.json"
        external.write_bytes(body)
        path.symlink_to(external)
    with pytest.raises((ValueError, FileNotFoundError)):
        selected.resolve_selected_evidence_task(workspace, pointer, candidate_packet=packet)
    assert len(calls) == 10 and len(searches) == 1


@pytest.mark.parametrize("action,available,reason", [
    ("run_discriminating_experiment", True, "no_actionable_literature_request"),
    ("retrieve_literature", False, "no_new_verified_full_text"),
])
def test_unsupported_stop_does_not_reroll(managed, monkeypatch, action, available, reason):
    _workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, terminal_at=9, action=action)
    searches = install_retrieval(monkeypatch, available=available)
    pointer = run(managed)
    assert pointer["status"] == "needs_evidence" and pointer["cycle"].endswith("cycle_0")
    assert read(chain_dir(managed) / "chain.json")["stop_reason"] == reason
    before = bytes_below(chain_dir(managed))
    assert run(managed) == pointer
    assert bytes_below(chain_dir(managed)) == before and len(calls) == 5
    assert len(searches) == int(action == "retrieve_literature")


def test_at_most_two_enrichments_even_when_reviews_keep_requesting(managed, monkeypatch):
    _workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, terminal_at=9)
    searches = install_retrieval(monkeypatch)
    pointer = run(managed)
    assert pointer["cycle"].endswith("cycle_2") and pointer["status"] == "needs_evidence"
    assert len(calls) == 15 and len(searches) == 2
    before = bytes_below(chain_dir(managed))
    assert run(managed) == pointer and before == bytes_below(chain_dir(managed))


def test_incomplete_later_cycle_preserves_prior_latest_and_attempt_budget(managed, monkeypatch):
    workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, failed_reviewer_at=1)
    searches = install_retrieval(monkeypatch)
    pointer = run(managed)
    assert pointer["cycle"].endswith("cycle_0")
    write(workspace / "novelty/latest.json", pointer)
    previous = bytes_below(chain_dir(managed) / "cycle_0")
    latest = (workspace / "novelty/latest.json").read_bytes()
    ledger = read(chain_dir(managed) / "chain.json")
    assert sum(row["charged"] for row in ledger["attempts"]) == 10
    for _ in range(15):
        assert run(managed) == pointer
    assert previous == bytes_below(chain_dir(managed) / "cycle_0")
    assert latest == (workspace / "novelty/latest.json").read_bytes()
    assert len(calls) <= 18 and len(searches) == 1
    assert not (chain_dir(managed) / "cycle_1/seal.json").exists()


def test_configuration_or_provenance_change_is_not_a_new_budget(managed, monkeypatch):
    workspace, cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, terminal_at=0)
    run(managed)
    cfg.harness.novelty_reviewers = 3
    with pytest.raises(ValueError, match="configuration"):
        run(managed)
    cfg.harness.novelty_reviewers = 2
    state = read(workspace / "research_preparation.json")
    state["initial_harness_state"].pop("selected_review_enrichment_version")
    write(workspace / "research_preparation.json", state)
    with pytest.raises(ValueError, match="trusted fresh preparation"):
        run(managed)
    assert len(calls) == 5


def test_elapsed_budget_stops_resume_without_new_calls(managed, monkeypatch):
    _workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, failed_reviewer_at=1)
    searches = install_retrieval(monkeypatch)
    pointer = run(managed)
    before_count = len(calls)
    created = read(chain_dir(managed) / "chain.json")["created_at_unix"]
    monkeypatch.setattr(selected.time, "time", lambda: created + selected.MAX_ELAPSED_SECONDS + 1)
    assert run(managed) == pointer
    assert len(calls) == before_count and len(searches) == 1


@pytest.mark.parametrize("interruption", ["failed", "started"])
def test_author_failure_resume_reuses_frozen_budget(managed, monkeypatch, interruption):
    _workspace, _cfg, agenda, review, _packet = managed
    failed = install_models(monkeypatch, agenda, review, terminal_at=0, failed_author_at=0)
    with pytest.raises(ValueError, match="author transport"):
        run(managed)
    directory = chain_dir(managed)
    receipt = directory / "cycle_0/agenda_draft/receipt.json"
    if interruption == "started":
        value = read(receipt)
        value["status"] = "started"
        write(receipt, value)
        ledger = read(directory / "chain.json")
        ledger["attempts"][-1].update(state="started", charged=3)
        write(directory / "chain.json", ledger)
    failed_bytes = bytes_below(receipt.parent)
    calls = install_models(monkeypatch, agenda, review, terminal_at=0)
    pointer = run(managed)
    assert pointer["status"] == "supported"
    assert len(failed) == 1 and len(calls) == 5
    archived = next((directory / "cycle_0/retry_attempts/agenda_draft").iterdir())
    assert bytes_below(archived) == failed_bytes
    ledger = read(directory / "chain.json")
    assert sum(row["charged"] for row in ledger["attempts"]) == (8 if interruption == "started" else 6)
    assert run(managed) == pointer and len(calls) == 5


def test_crash_after_seal_recovers_without_rewriting_completed_cycle(managed, monkeypatch):
    _workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, terminal_at=0)
    original_dump = selected._dump
    def interrupt(path, value):
        if path.name == "chain.json" and value["cycles"] and value["cycles"][-1]["state"] == "completed":
            raise OSError("crash after seal")
        original_dump(path, value)
    monkeypatch.setattr(selected, "_dump", interrupt)
    with pytest.raises(OSError, match="crash after seal"):
        run(managed)
    cycle = chain_dir(managed) / "cycle_0"
    assert (cycle / "seal.json").is_file()
    before = bytes_below(cycle)
    monkeypatch.setattr(selected, "_dump", original_dump)
    pointer = run(managed)
    assert pointer["status"] == "supported" and len(calls) == 5
    assert before == bytes_below(cycle)


@pytest.mark.parametrize("change", ["missing_journal", "reset_ledger"])
def test_old_attempts_cannot_receive_a_reset_budget(managed, monkeypatch, change):
    _workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review, terminal_at=0, failed_author_at=0)
    with pytest.raises(ValueError):
        run(managed)
    journal = chain_dir(managed) / "chain.json"
    if change == "missing_journal":
        journal.unlink()
    else:
        data = read(journal)
        data["attempts"] = []
        write(journal, data)
    with pytest.raises(ValueError, match="budget|ledger"):
        run(managed)
    assert len(calls) == 1


def test_final_created_during_retrieval_does_not_open_new_cycle(managed, monkeypatch):
    workspace, _cfg, agenda, review, _packet = managed
    calls = install_models(monkeypatch, agenda, review)
    install_retrieval(monkeypatch)
    original = selected.enrich_review_evidence
    def retrieve(*args, **kwargs):
        result = original(*args, **kwargs)
        write(workspace / "run/final_evaluation/invocation.json", {})
        return result
    monkeypatch.setattr(selected, "enrich_review_evidence", retrieve)
    pointer = run(managed)
    assert pointer["cycle"].endswith("cycle_0")
    assert not (chain_dir(managed) / "cycle_1").exists() and len(calls) == 5


@pytest.mark.parametrize("status", ["overlap", "no_novelty_claim"])
def test_terminal_nonpassing_results_stop_without_enrichment(managed, monkeypatch, status):
    _workspace, _cfg, agenda, review, _packet = managed
    if status == "no_novelty_claim":
        agenda["intent"] = "replication"
    else:
        review["overlap_assessment"] = {"core_problem_same": True, "key_method_same": True,
            "main_contribution_same": True, "remaining_difference_meaningful": False}
    calls = install_models(monkeypatch, agenda, review, terminal_at=9, verdict=status)
    searches = install_retrieval(monkeypatch)
    pointer = run(managed)
    assert pointer["status"] == status and pointer["cycle"].endswith("cycle_0")
    assert len(calls) == 5 and not searches
    assert run(managed) == pointer and len(calls) == 5


def test_early_author_convergence_and_incomplete_review_resume(managed, monkeypatch):
    _workspace, _cfg, agenda, review, _packet = managed
    first = install_models(monkeypatch, agenda, review, done=True, failed_reviewer_at=0, terminal_at=0)
    with pytest.raises(ValueError, match="No complete selected review"):
        run(managed)
    assert len(first) == 4
    before = bytes_below(chain_dir(managed) / "cycle_0/agenda_reflection_2")
    second = install_models(monkeypatch, agenda, review, done=True, terminal_at=0)
    pointer = run(managed)
    assert pointer["status"] == "supported" and len(second) == 1
    assert before == bytes_below(chain_dir(managed) / "cycle_0/agenda_reflection_2")
    assert sum(row["charged"] for row in read(chain_dir(managed) / "chain.json")["attempts"]) == 5


def test_author_review_and_retrieval_share_one_frozen_deadline(managed, monkeypatch):
    _workspace, _cfg, agenda, review, _packet = managed
    install_models(monkeypatch, agenda, review)
    install_retrieval(monkeypatch)
    observed = []
    for name in ("draft_research_agenda", "review_novelty", "enrich_review_evidence"):
        original = getattr(selected, name)
        def observe(*args, _original=original, _name=name, **kwargs):
            observed.append((_name, kwargs["deadline_unix"]))
            return _original(*args, **kwargs)
        monkeypatch.setattr(selected, name, observe)
    pointer = run(managed)
    chain = read(chain_dir(managed) / "chain.json")
    expected = chain["created_at_unix"] + chain["fixed"]["max_elapsed_seconds"]
    assert len(observed) == 5 and {row[1] for row in observed} == {expected}
    assert read(chain_dir(managed) / "retrieval/journal.json")["fixed"]["deadline_unix"] == expected
    assert run(managed) == pointer and len(observed) == 5
