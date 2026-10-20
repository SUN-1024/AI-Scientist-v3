from __future__ import annotations

import copy
import hashlib
import json
import threading

import pytest

from src.research.config import LiteratureConfig, ModelSearchConfig
from src.research.literature import prepare_literature
from src.research.evidence import render_context
from src.research.novelty import REVIEW_CHECKS, CANDIDATE_CHECKS, content_hash, evaluate_novelty
from src.research.novelty_review import (
    draft_research_agenda, load_frozen_novelty_report, normalize_review, review_novelty, source_packet,
)
from src.research.research_agenda import agenda_from_basis, load_agenda


@pytest.fixture
def study(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.md").write_text("Test whether adaptive step sizes improve noisy-label optimization.")
    text = ("Setting: noisy labels, with fixed training and validation partitions. "
            "Method: use a fixed step size and identical update counts. "
            "Evaluation: accuracy averaged over five seeds, with matched compute. "
            "Contribution: establishes a fixed-step baseline; adaptive schedules were not tested.")
    (task / "paper.txt").write_text(text)
    cfg = LiteratureConfig(sources=[{"id": "prior", "title": "Fixed-Step Baseline", "path": "paper.txt"}],
                           model_search=ModelSearchConfig(provider="custom", model="fixture-model",
                                                           base_url="http://127.0.0.1:9000/v1",
                                                           api_format="chat_completions"))
    prepare_literature(task, cfg, offline=True)
    citation = {"passage_id": f"prior:chars_0_{len(text)}", "quote": text}
    agenda = {
        "schema_version": 1, "intent": "novel_contribution", "claim_type": "method",
        "research_question": "Does adapting step size help when label noise changes during training?",
        "proposed_contribution": "Extend the fixed-step baseline with a noise-dependent step-size rule.",
        "why_meaningful": "Test whether a specific adaptation handles changing label noise under the same budget.",
        "scope": "Synthetic time-varying corruption; no claim of absolute originality.",
        "closest_work": [{
            "source_id": "prior", "problem_setting": "Noisy-label training with fixed partitions.",
            "core_method": "A fixed learning step.", "contribution_boundary": "Adaptive schedules were not tested.",
            "overlap": "Same optimizer and task.", "difference": "Only the step-size rule changes.",
            "why_meaningful": "Tests a previously untested noise-adaptation mechanism in these conditions.",
            "evidence": [{"aspect": aspect, **citation} for aspect in
                         ("problem_setting", "method", "evaluation", "contribution_boundary")],
        }],
        "source_selection": [{"source_id": "prior", "relation": "closest", "reason": "Uses the same optimizer and noisy-label task."}],
        "source_scan": {"queries": ["adaptive noisy-label step sizes"], "coverage_limits": "One provided primary source; reviewer must assess if more are needed."},
        "alternatives": [{"explanation": "More updates rather than adaptation explain the gain.",
                          "discriminating_experiment": {
                              "intervention": "Change only the step-size rule.", "controls": "Same data, seeds, update count, and parameter count.",
                              "outcome_if_claim": "Adaptive rule gains accuracy at equal update counts.",
                              "outcome_if_alternative": "The gain disappears at equal update counts.",
                              "decision_rule": "Reject mechanism claim if paired-seed improvement is not robust across noise conditions.",
                          }}],
        "evidence_plan": {"strong_baselines": ["Tuned fixed step and standard schedule."],
                          "robustness_checks": ["Five fixed paired seeds, multiple noise conditions."],
                          "independent_confirmation": ["Untouched noise trajectories, held out before tuning."]},
    }
    review = {
        "verdict": "differentiated", "rationale": "A meaningful incremental test is possible under the stated conditions; experiments remain to be done.",
        "checks": {key: "pass" for key in REVIEW_CHECKS},
        "source_findings": [{"source_id": "prior", "assessment": "The source does not test adaptation; the shared optimizer is not itself novelty.", "evidence": [citation]}],
        "overlap_assessment": {"core_problem_same": True, "key_method_same": False,
                               "main_contribution_same": False, "remaining_difference_meaningful": True},
        "next_actions": [],
    }
    return task, cfg, agenda, review


def install_provider(monkeypatch, values):
    calls = []
    lock = threading.Lock()
    novelty_start = None
    monkeypatch.setenv("AGSCI_LITERATURE_API_KEY", "fixture-key")

    def provider(_url, body, _headers, **_kwargs):
        nonlocal novelty_start
        with lock:
            index = len(calls)
            prompt = body["messages"][0]["content"]
            if "\n\nFROZEN INPUT DATA:\n" in prompt:
                if novelty_start is None:
                    novelty_start = index
                # Bind fixtures to reviewer identity, not thread scheduling.
                index = novelty_start + int("Examine the strongest prior-art collision" in prompt)
            calls.append(copy.deepcopy(body))
            value = values[min(index, len(values) - 1)]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr("src.research.model_search._http_json", provider)
    return calls


def test_incremental_contribution_can_pass_without_absolute_originality(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [review])
    report = review_novelty(task, agenda, cfg, tmp_path / "review")
    assert report["status"] == "supported"
    assert report["acceptance_guaranteed"] is False
    assert report["machine_checks"]["meaning_or_originality_proven"] is False
    assert len(calls) == 2
    assert all(len(call["messages"]) == 1 for call in calls)
    assert calls[0]["messages"][0]["content"] != calls[1]["messages"][0]["content"]
    assert report["next_actions"][0]["kind"] == "run_discriminating_experiment"
    assert load_frozen_novelty_report(task, tmp_path / "review")["status"] == "supported"


def test_self_asserted_novelty_and_reviews_do_not_pass(study):
    task, _cfg, agenda, review = study
    result = evaluate_novelty(task, agenda, reviews=[review, review])
    assert result["status"] == "needs_evidence"
    assert result["model_judgments"] == []
    assert {gap["code"] for gap in result["gaps"]} >= {"unverified_review", "independent_reviews_missing"}


def test_missing_closest_paper_and_abstract_only_method_claims_fail(study):
    task, _cfg, agenda, _review = study
    missing = {**agenda, "closest_work": []}
    assert "missing_closest_work" in {gap["code"] for gap in evaluate_novelty(task, missing)["gaps"]}
    path = task / "research_context/catalog.json"
    catalog = json.loads(path.read_text())
    catalog["sources"][0]["availability"] = "abstract"
    path.write_text(json.dumps(catalog))
    report = evaluate_novelty(task, agenda)
    assert report["status"] == "needs_evidence"
    assert any("complete saved full text" in gap["detail"] for gap in report["gaps"])


@pytest.mark.parametrize("change", ["source", "response", "assessment", "prompt", "catalog", "agenda"])
def test_frozen_review_detects_changed_evidence(study, tmp_path, monkeypatch, change):
    task, cfg, agenda, review = study
    install_provider(monkeypatch, [review])
    output = tmp_path / "review"
    review_novelty(task, agenda, cfg, output)
    if change == "source":
        path = task / "research_context/text/prior.txt"
        path.write_text(path.read_text() + "tampered")
    elif change == "catalog":
        path = task / "research_context/catalog.json"
        data = json.loads(path.read_text())
        data["sources"][0]["title"] = "Changed title"
        path.write_text(json.dumps(data))
    elif change == "agenda":
        path = output / "research_agenda.json"
        data = json.loads(path.read_text())
        data["proposed_contribution"] = "A different, unreviewed claim."
        path.write_text(json.dumps(data))
    else:
        name = {"response": "01_novelty_review.json", "assessment": "assessment.json", "prompt": "prompt.txt"}[change]
        path = output / "reviewer_1" / name
        path.write_text(path.read_text() + " ")
    report = load_frozen_novelty_report(task, output)
    assert report["status"] == "needs_evidence"
    assert report["gaps"]


def test_exact_span_and_quote_are_both_checked(study):
    task, _cfg, agenda, _review = study
    agenda["closest_work"][0]["evidence"][0]["quote"] = "The paper proves my new method is best."
    result = evaluate_novelty(task, agenda)
    assert result["status"] == "needs_evidence"
    assert any(gap["code"] == "invalid_passage" for gap in result["gaps"])


def test_replication_is_valuable_without_a_novel_method_claim(study):
    task, _cfg, agenda, _review = study
    agenda["intent"] = "replication"
    result = evaluate_novelty(task, agenda)
    assert result["status"] == "no_novelty_claim"
    assert result["honest_paper_writing_allowed"] is True


def test_supported_serious_overlap_requires_two_grounded_assessments(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    overlap = copy.deepcopy(review)
    overlap.update(verdict="overlap", rationale="The actual candidate uses the same fixed rule and contributes the same experiment.")
    overlap["checks"]["difference_meaningful"] = "fail"
    overlap["overlap_assessment"] = {"core_problem_same": True, "key_method_same": True,
                                     "main_contribution_same": True, "remaining_difference_meaningful": False}
    install_provider(monkeypatch, [overlap])
    result = review_novelty(task, agenda, cfg, tmp_path / "overlap")
    assert result["status"] == "overlap"
    assert result["honest_paper_writing_allowed"] is True
    assert any(action["kind"] == "revise_contribution" for action in result["next_actions"])
    install_provider(monkeypatch, [review, overlap])
    result = review_novelty(task, agenda, cfg, tmp_path / "disagree")
    assert result["status"] == "needs_evidence"


def test_overlap_without_core_collision_is_not_a_rejection(study):
    _task, _cfg, _agenda, review = study
    review["verdict"] = "overlap"
    with pytest.raises(ValueError, match="collision assessment"):
        normalize_review(review)


def test_review_missing_source_evidence_cannot_pass(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    review["source_findings"] = []
    install_provider(monkeypatch, [review])
    report = review_novelty(task, agenda, cfg, tmp_path / "review")
    assert report["status"] == "needs_evidence"
    assert any(gap["code"] == "review_coverage_missing" for gap in report["gaps"])


def test_no_discriminating_prediction_or_untouched_confirmation_is_a_gap(study):
    task, _cfg, agenda, _review = study
    experiment = agenda["alternatives"][0]["discriminating_experiment"]
    experiment["outcome_if_alternative"] = experiment["outcome_if_claim"]
    agenda["evidence_plan"]["independent_confirmation"] = []
    report = evaluate_novelty(task, agenda)
    codes = {gap["code"] for gap in report["gaps"]}
    assert {"identical_predictions", "evidence_plan_incomplete"} <= codes


def test_draft_is_saved_and_legacy_basis_is_not_upgraded(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    install_provider(monkeypatch, [agenda])
    output = tmp_path / "agenda"
    draft = draft_research_agenda(task, cfg, output, research_basis={"status": "replication", "contribution": "replicate"})
    assert draft == agenda
    assert load_agenda(output) == agenda
    assert (output / "agenda_draft/receipt.json").is_file()
    old = agenda_from_basis({"status": "differentiated", "contribution": "old claim", "closest_work": []})
    assert evaluate_novelty(task, old)["status"] == "needs_evidence"


def test_failed_api_call_never_becomes_positive_review(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    monkeypatch.setenv("AGSCI_LITERATURE_API_KEY", "fixture-key")
    monkeypatch.setattr("src.research.model_search._http_json", lambda *_args, **_kwargs: {"error": "budget exhausted"})
    result = review_novelty(task, agenda, cfg, tmp_path / "review")
    assert result["status"] == "needs_evidence"
    assert result["model_judgments"] == []
    assert len(list((tmp_path / "review").glob("reviewer_*/receipt.json"))) == 2


def test_shared_fresh_agent_backend_and_transport_tamper(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = []

    def run(prompt, *, work_dir, **settings):
        calls.append(settings)
        work_dir.mkdir()
        data = {"protocolVersion": 1, "type": "result", "exitCode": 0,
                "data": {"finalMessage": json.dumps(review)}}
        raw = json.dumps(data)
        (work_dir / "stdout.json").write_text(raw)
        transport = {"status": "completed", "independent_context": True,
                     "native_session_id": f"session-{work_dir.parent.name}", "observed_provider": "fixture",
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "stdout_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        (work_dir / "receipt.json").write_text(json.dumps(transport))
        return review, transport

    monkeypatch.setattr("src.review_backend.run_fresh_agent_json", run)
    output = tmp_path / "agent_review"
    result = review_novelty(task, agenda, cfg, output,
                            reviewer_settings={"agent": "codex", "model": "fixture", "reasoning_effort": "high"})
    assert result["status"] == "supported"
    assert len(calls) == 2 and calls[0]["reasoning_effort"] == "high"
    assert result["model_judgments"][0]["receipt"]["backend"] == "agent_stateless"
    path = output / "reviewer_1/transport/receipt.json"
    path.write_text(path.read_text() + " ")
    assert load_frozen_novelty_report(task, output)["status"] == "needs_evidence"


def test_pre_experiment_feedback_reaches_creative_and_paper_context(study):
    task, _cfg, agenda, _review = study
    report = evaluate_novelty(task, agenda)
    (task / "research_context/harness_novelty.json").write_text(json.dumps(report))
    for paper in (False, True):
        context = render_context(task, for_paper=paper)
        assert "HARNESS AGENDA REVIEW FEEDBACK" in context
        assert "independent_reviews_missing" in context
        assert "not proof about a later selected candidate" in context


def candidate_packet(code="def forward(bags):\n    return adaptive_noise_step(bags)\n"):
    source_hash = hashlib.sha256(code.encode()).hexdigest()
    return {
        "schema_version": 1, "selection": {"idea_id": 3, "program_relative_path": "programs/program_3"},
        "candidate_source_sha256": source_hash,
        "source_files": [{"path": "src/program.py", "sha256": source_hash, "text": code}],
        "evaluation": {"source_receipt_validated": True, "metrics": {"combined_score": 0.59, "matched_baseline": 0.58},
                       "correct": {"correct": True}, "evidence": {"status": "host_verified"}},
        "protocol": {"seeds": [7, 11, 19]},
    }


def candidate_review(review, packet):
    result = copy.deepcopy(review)
    result["candidate_assessment"] = {
        "candidate_packet_sha256": content_hash(packet),
        "checks": {key: "pass" for key in CANDIDATE_CHECKS},
        "claim_code_evidence_map": [{"claim": "Adaptive step implemented; observed development gain remains provisional.",
                                     "code_paths": ["src/program.py"],
                                     "evaluation_refs": ["evaluation.metrics", "evaluation.correct"],
                                     "finding": "Forward calls adaptive_noise_step; 0.59 vs 0.58 is a development comparison, not independent confirmation."}],
    }
    return result


def test_selected_review_binds_actual_code_and_verified_outcomes(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    packet = candidate_packet()
    calls = install_provider(monkeypatch, [candidate_review(review, packet)])
    output = tmp_path / "selected"
    result = review_novelty(task, agenda, cfg, output, candidate_packet=packet)
    assert result["status"] == "supported"
    assert result["frozen_evidence"]["candidate_packet_sha256"] == content_hash(packet)
    prompt = calls[0]["messages"][0]["content"]
    assert "adaptive_noise_step" in prompt and '"combined_score": 0.59' in prompt
    assert load_frozen_novelty_report(task, output, candidate_packet=packet)["status"] == "supported"
    assert load_frozen_novelty_report(task, output)["status"] == "needs_evidence"
    # The old idea/agenda is identical, but B implements only the baseline.
    other = candidate_packet("def forward(bags):\n    return baseline(bags)\n")
    assert load_frozen_novelty_report(task, output, candidate_packet=other)["status"] == "needs_evidence"
    changed_result = copy.deepcopy(packet)
    changed_result["evaluation"]["metrics"]["combined_score"] = 0.55
    assert load_frozen_novelty_report(task, output, candidate_packet=changed_result)["status"] == "needs_evidence"


def test_selected_draft_sees_code_even_when_idea_claim_is_unchanged(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    packet = candidate_packet("def forward(bags):\n    return ordinary_log_mean_exp(bags)\n")
    calls = install_provider(monkeypatch, [agenda])
    draft_research_agenda(task, cfg, tmp_path / "draft", research_basis={"contribution": "purported adaptive mechanism"},
                          candidate_packet=packet)
    prompt = calls[0]["messages"][0]["content"]
    assert "ordinary_log_mean_exp" in prompt and "purported adaptive mechanism" in prompt
    receipt = json.loads((tmp_path / "draft/agenda_draft/receipt.json").read_text())
    assert receipt["frozen_evidence"]["candidate_packet_sha256"] == content_hash(packet)


def test_idea_only_review_cannot_certify_implementation(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    packet = candidate_packet()
    install_provider(monkeypatch, [review])
    result = review_novelty(task, agenda, cfg, tmp_path / "idea_only", candidate_packet=packet)
    assert result["status"] == "needs_evidence"
    assert any(gap["code"] == "candidate_review_unbound" for gap in result["gaps"])
    actual = candidate_review(review, packet)
    actual["candidate_assessment"]["checks"]["implementation_matches_claim"] = "fail"
    install_provider(monkeypatch, [actual])
    result = review_novelty(task, agenda, cfg, tmp_path / "absent_mechanism", candidate_packet=packet)
    assert result["status"] == "needs_evidence"
    assert any(gap["code"] == "candidate_claim_unimplemented_or_unverified" for gap in result["gaps"])


def test_selected_code_hash_without_validated_outcome_is_insufficient(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    calls = install_provider(monkeypatch, [review])
    with pytest.raises(ValueError, match="selected program identity"):
        review_novelty(task, agenda, cfg, tmp_path / "hash_only", candidate_packet={"schema_version": 1, "candidate_source_sha256": "a" * 64})
    packet = candidate_packet()
    packet["evaluation"]["source_receipt_validated"] = False
    with pytest.raises(ValueError, match="host-validated evaluation"):
        review_novelty(task, agenda, cfg, tmp_path / "no_outcome", candidate_packet=packet)
    assert calls == []


def test_v1_reflection_uses_real_transport_and_frozen_source_candidate_context(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    packet = candidate_packet()
    revised = copy.deepcopy(agenda)
    revised['scope'] += ' Candidate evidence supports development testing only.'
    reflected = {**revised, '_reflection': {'done': True, 'changes': 'Narrow the claim.', 'remaining_gaps': ['Independent confirmation.']}}
    converged = {**revised, '_reflection': {'done': True, 'changes': '', 'remaining_gaps': ['Independent confirmation.']}}
    calls = install_provider(monkeypatch, [agenda, reflected, converged])
    output = tmp_path / 'reflection'
    result = draft_research_agenda(task, cfg, output, candidate_packet=packet)
    assert result == revised
    assert len(calls) == 3
    source = (task / 'research_context/text/prior.txt').read_text()
    sessions = set()
    bindings = []
    for round_index, directory in enumerate(['agenda_draft', 'agenda_reflection_2', 'agenda_reflection_3']):
        prompt = calls[round_index]['messages'][0]['content']
        # JSON input preserves complete source text on every fresh request.
        assert source in prompt
        assert 'adaptive_noise_step' in prompt and '"combined_score": 0.59' in prompt
        receipt = json.loads((output / directory / 'receipt.json').read_text())
        assert receipt['status'] == 'complete'
        assert receipt['frozen_evidence']['candidate_packet_sha256'] == content_hash(packet)
        bindings.append(receipt['frozen_evidence'])
        sessions.add(receipt['session_id'])
        response_path = output / directory / receipt['response']
        assert hashlib.sha256(response_path.read_bytes()).hexdigest() == receipt['response_sha256']
    assert len(sessions) == 3 and bindings[0] == bindings[1] == bindings[2]
    trace = json.loads((output / 'agenda_reflection_trace.json').read_text())
    assert trace['status'] == 'converged'
    assert trace['author_proposal_only'] is True and trace['independent_novelty_evidence'] is False
    # An explicit stop while changing the agenda does not stop reflection.
    assert trace['history'][1]['changed'] is True
    assert trace['history'][2]['changed'] is False
    assert evaluate_novelty(task, result)['status'] == 'needs_evidence'


def test_v1_invalid_reflection_does_not_publish_a_completed_agenda(study, tmp_path, monkeypatch):
    task, cfg, agenda, _review = study
    malformed = {**agenda, '_reflection': {'done': 'yes'}}
    calls = install_provider(monkeypatch, [agenda, malformed])
    output = tmp_path / 'invalid_reflection'
    with pytest.raises(ValueError, match='Reflection.done'):
        draft_research_agenda(task, cfg, output)
    assert len(calls) == 2
    assert not (output / 'research_agenda.json').exists()
    trace = json.loads((output / 'agenda_reflection_trace.json').read_text())
    assert trace['status'] == 'failed'
    assert trace['recovery']['source_directory'] == 'agenda_draft'
    recovery = json.loads((output / 'agenda_reflection_recovery.json').read_text())
    assert recovery['status'] == 'verified_last_valid_author_agenda'
    assert recovery['source_directory'] == 'agenda_draft'
    assert recovery['agenda_sha256'] == content_hash(recovery['agenda'])
    assert not list(output.glob('reviewer_*'))


def test_v1_third_round_schema_failure_recovers_second_reflection_in_harness(study, tmp_path, monkeypatch):
    from src.research.config import ResearchConfig
    from src.research.harness import _novelty_cycle

    task, cfg, agenda, _review = study
    revised = copy.deepcopy(agenda)
    revised["scope"] += " This is the last schema-valid reflection."
    second = {
        **revised,
        "_reflection": {
            "done": False,
            "changes": "Narrow the claim and retain unresolved evidence.",
            "remaining_gaps": ["Independent confirmation."],
        },
    }
    malformed_third = {
        **revised,
        "_reflection": {
            "done": False,
            "changes": "Try another revision.",
            "remaining_gaps": "This must be a list.",
        },
    }
    calls = install_provider(monkeypatch, [agenda, second, malformed_third])
    output = tmp_path / "recover-second-reflection"
    report = _novelty_cycle(
        task, ResearchConfig(literature=cfg), output, offline=False,
        reviewer_settings=None, research_basis={"contribution": "Original idea"},
    )
    assert len(calls) == 3
    assert report["status"] == "needs_evidence"
    assert any(gap["code"] == "review_execution_failed" for gap in report["gaps"])
    assert report["agenda_recovery"]["source_directory"] == "agenda_reflection_2"
    assert load_agenda(output) == revised
    trace = json.loads((output / "agenda_reflection_trace.json").read_text())
    assert trace["status"] == "failed"
    assert trace["recovery"]["source_directory"] == "agenda_reflection_2"
    assert not list(output.glob("reviewer_*"))


def test_v1_reflection_transport_error_keeps_harness_fallback_unverified(study, tmp_path, monkeypatch):
    from src.research.config import ResearchConfig
    from src.research.harness import _novelty_cycle

    task, cfg, agenda, _review = study
    packet = candidate_packet()
    monkeypatch.setenv('AGSCI_LITERATURE_API_KEY', 'fixture-key')
    calls = []

    def provider(_url, body, _headers, **_kwargs):
        calls.append(copy.deepcopy(body))
        if len(calls) == 1:
            return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(agenda)}}]}
        return {'error': 'reflection transport unavailable'}

    monkeypatch.setattr('src.research.model_search._http_json', provider)
    output = tmp_path / 'fallback'
    result = _novelty_cycle(task, ResearchConfig(literature=cfg), output, offline=False,
                            reviewer_settings=None, research_basis={'contribution': 'Original idea'},
                            candidate_packet=packet)
    assert len(calls) == 2
    assert result['status'] == 'needs_evidence' and result['model_judgments'] == []
    assert any(gap['code'] == 'review_execution_failed' for gap in result['gaps'])
    assert not list(output.glob('reviewer_*'))
    assert result['agenda_recovery']['source_directory'] == 'agenda_draft'
    assert load_agenda(output) == agenda
    receipt = json.loads((output / 'agenda_reflection_2/receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['frozen_evidence']['candidate_packet_sha256'] == content_hash(packet)
    trace = json.loads((output / 'agenda_reflection_trace.json').read_text())
    assert trace['status'] == 'failed' and trace['independent_novelty_evidence'] is False


@pytest.fixture
def source_selection_study(study, tmp_path):
    original_task, cfg, agenda, review = study
    task = tmp_path / "source_selection_task"
    task.mkdir()
    (task / "task.md").write_bytes((original_task / "task.md").read_bytes())
    (task / "paper.txt").write_bytes((original_task / "paper.txt").read_bytes())
    other_text = ("Related method: optimize a different model family. "
                  "Evaluation: larger data and a different compute budget. "
                  "Contribution boundary: no matched comparison with the fixed-step baseline.")
    (task / "related.txt").write_text(other_text)
    cfg = copy.deepcopy(cfg)
    cfg.sources.extend([
        {"id": "related", "title": "Related study outside closest_work", "path": "related.txt"},
        {"id": "metadata_only", "title": "Unavailable source", "url": "https://example.org/unavailable"},
    ])
    prepare_literature(task, cfg, offline=True)
    agenda = copy.deepcopy(agenda)
    agenda["source_selection"].extend([
        {"source_id": "related", "relation": "excluded", "reason": "Different model family and compute budget."},
        {"source_id": "metadata_only", "relation": "excluded", "reason": "Full text is not available; no method claim relies on this source."},
    ])
    agenda["source_scan"]["coverage_limits"] += " Related study uses a different compute budget."
    return task, cfg, agenda, review, other_text


def test_review_source_packet_includes_full_text_outside_closest_work(source_selection_study):
    task, _cfg, agenda, _review, other_text = source_selection_study
    assert {row["source_id"] for row in agenda["closest_work"]} == {"prior"}
    author_packet = source_packet(task)
    reviewer_packet = source_packet(task, agenda)
    assert reviewer_packet == author_packet
    rows = {row["id"]: row for row in reviewer_packet["sources"]}
    assert rows["prior"]["full_text"] == (task / "paper.txt").read_text()
    assert rows["related"]["full_text"] == other_text
    assert rows["metadata_only"]["availability"] == "metadata"
    assert "full_text" not in rows["metadata_only"]


def test_each_independent_reviewer_receives_all_prepared_full_text(source_selection_study, tmp_path, monkeypatch):
    task, cfg, agenda, review, other_text = source_selection_study
    calls = install_provider(monkeypatch, [review])
    review_novelty(task, agenda, cfg, tmp_path / "complete_source_reviews")
    assert len(calls) == 2
    for call in calls:
        prompt = call["messages"][0]["content"]
        payload = json.loads(prompt.split("\n\nFROZEN INPUT DATA:\n", 1)[1])
        rows = {row["id"]: row for row in payload["literature"]["sources"]}
        assert rows["related"]["full_text"] == other_text
        assert rows["prior"]["full_text"] == (task / "paper.txt").read_text()
        assert payload["agenda"]["source_selection"] == agenda["source_selection"]
    for receipt_path in (tmp_path / "complete_source_reviews").glob("reviewer_*/receipt.json"):
        receipt = json.loads(receipt_path.read_text())
        assert receipt["source_packet_sha256"] == content_hash(source_packet(task, agenda))
        assert receipt["source_coverage"] == {
            "policy": "all_prepared_sources_v1", "source_ids": ["prior", "related", "metadata_only"],
            "saved_text_source_ids": ["prior", "related"], "full_text_source_ids": ["prior", "related"],
        }


def test_non_closest_full_text_keeps_packet_limit_and_fails_before_review(source_selection_study, tmp_path, monkeypatch):
    from src.research.novelty_review import MAX_PACKET_CHARS
    task, cfg, agenda, review, _other_text = source_selection_study
    text_path = task / "research_context/text/related.txt"
    body = ("complete source text " * 31_000).encode()
    text_path.write_bytes(body)
    catalog_path = task / "research_context/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    source = next(row for row in catalog["sources"] if row["id"] == "related")
    source.update(text_sha256=hashlib.sha256(body).hexdigest(), passages=[])
    catalog_path.write_text(json.dumps(catalog))
    calls = install_provider(monkeypatch, [review])
    assert MAX_PACKET_CHARS == 600_000
    with pytest.raises(ValueError, match="exceeds 600,000"):
        source_packet(task)
    with pytest.raises(ValueError, match="exceeds 600,000"):
        review_novelty(task, agenda, cfg, tmp_path / "oversized_review")
    assert not calls


@pytest.mark.parametrize("legacy", [False, True])
def test_old_narrow_review_packet_cannot_be_replayed_as_complete_source_review(source_selection_study, tmp_path, monkeypatch, legacy):
    from src.research import novelty_review
    task, cfg, agenda, review, _other_text = source_selection_study
    complete_packet = novelty_review.source_packet
    def old_packet(task, agenda=None):
        packet = complete_packet(task, agenda)
        if agenda is not None:
            selected = {row["source_id"] for row in agenda["closest_work"]}
            for row in packet["sources"]:
                if row["id"] not in selected:
                    row.pop("full_text", None)
        return packet
    calls = install_provider(monkeypatch, [review])
    monkeypatch.setattr(novelty_review, "source_packet", old_packet)
    output = tmp_path / "old_narrow_reviews"
    report = review_novelty(task, agenda, cfg, output)
    assert len(report["model_judgments"]) == 2
    assert report["status"] == "supported"
    if legacy:
        for path in output.glob("reviewer_*/receipt.json"):
            receipt = json.loads(path.read_text())
            receipt["schema_version"] = 1
            receipt.pop("source_packet_sha256")
            receipt.pop("source_coverage")
            path.write_text(json.dumps(receipt))
    monkeypatch.setattr(novelty_review, "source_packet", complete_packet)
    current = load_frozen_novelty_report(task, output)
    assert current["status"] == "needs_evidence"
    assert current["model_judgments"] == []
    assert any("all prepared source texts" in str(gap) for gap in current["gaps"])
    assert len(calls) == 2


def test_legacy_full_source_prompt_remains_verifiable_without_new_coverage_fields(study, tmp_path, monkeypatch):
    task, cfg, agenda, review = study
    install_provider(monkeypatch, [review])
    output = tmp_path / "legacy_complete_reviews"
    review_novelty(task, agenda, cfg, output)
    for path in output.glob("reviewer_*/receipt.json"):
        receipt = json.loads(path.read_text())
        receipt["schema_version"] = 1
        receipt.pop("source_packet_sha256")
        receipt.pop("source_coverage")
        path.write_text(json.dumps(receipt))
    report = load_frozen_novelty_report(task, output)
    assert report["status"] == "supported"
    assert all(row["receipt"]["source_packet_sha256"] == content_hash(source_packet(task)) for row in report["model_judgments"])


def test_source_coverage_label_alone_does_not_validate_saved_packet(source_selection_study, tmp_path, monkeypatch):
    task, cfg, agenda, review, _other_text = source_selection_study
    install_provider(monkeypatch, [review])
    output = tmp_path / "changed_coverage"
    review_novelty(task, agenda, cfg, output)
    for path in output.glob("reviewer_*/receipt.json"):
        receipt = json.loads(path.read_text())
        receipt["source_packet_sha256"] = "0" * 64
        path.write_text(json.dumps(receipt))
    report = load_frozen_novelty_report(task, output)
    assert report["status"] == "needs_evidence"
    assert not report["model_judgments"]
    assert any("source packet receipt changed" in str(gap) for gap in report["gaps"])
