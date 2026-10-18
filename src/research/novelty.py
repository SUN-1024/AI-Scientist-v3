"""Evidence-bound novelty decisions, without pretending that novelty is computable.

The machine checks provenance and completeness. Separate model calls assess meaning.
Neither layer proves originality, scientific validity, or conference acceptance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .evidence import _passage_span, load_catalog
from .literature import CONTEXT_DIRECTORY
from .research_agenda import EVIDENCE_ASPECTS, normalize_agenda

REVIEW_CHECKS = (
    "closest_work_adequate", "attribution_faithful", "difference_meaningful",
    "experiment_discriminates", "claim_scope_calibrated",
)
CANDIDATE_CHECKS = (
    "implementation_matches_claim", "reported_outcomes_faithful", "comparison_scope_calibrated",
)
ACTION_TYPES = {
    "retrieve_literature", "inspect_full_text", "revise_contribution",
    "design_discriminating_experiment", "run_discriminating_experiment", "independent_review",
}


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_candidate_packet(packet: object) -> dict[str, Any]:
    """Require inspectable code and host-validated outcomes, not a code hash alone.

    The harness constructs this packet from the frozen selected candidate and
    verifies its evaluation receipts. A caller's dict is not a new trust root.
    """
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise ValueError("Candidate packet must be a schema_version=1 object")
    fingerprint = packet.get("candidate_source_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Candidate packet requires the selected source SHA-256")
    selected = packet.get("selection")
    if not isinstance(selected, dict) or not isinstance(selected.get("program_relative_path"), str) or not selected["program_relative_path"]:
        raise ValueError("Candidate packet requires the selected program identity")
    files = packet.get("source_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Candidate packet must include actual selected source files")
    seen = set()
    total = 0
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str) or not isinstance(row.get("text"), str):
            raise ValueError("Candidate source entries require relative path, SHA-256, and exact text")
        path = PurePosixPath(row["path"])
        if not row["path"] or path.is_absolute() or ".." in path.parts or row["path"] in seen:
            raise ValueError("Invalid or duplicate candidate source path")
        seen.add(row["path"])
        body = row["text"].encode("utf-8")
        total += len(body)
        if hashlib.sha256(body).hexdigest() != row.get("sha256"):
            raise ValueError("Candidate source text/hash mismatch")
    if total > 2 * 1024 * 1024:
        raise ValueError("Candidate source packet exceeds 2 MiB; do not silently truncate implementation evidence")
    evaluation = packet.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("source_receipt_validated") is not True:
        raise ValueError("Candidate packet needs a host-validated evaluation receipt bound to its source")
    if not isinstance(evaluation.get("metrics"), dict) or not evaluation["metrics"]:
        raise ValueError("Candidate packet needs the actual selected evaluation metrics")
    if not isinstance(evaluation.get("correct"), dict) or not evaluation["correct"]:
        raise ValueError("Candidate packet needs the selected correctness evidence")
    return packet


def freeze_evidence(task_dir: Path) -> dict[str, Any]:
    from .methodology import load_methodology

    catalog = load_catalog(task_dir)
    if catalog is None:
        raise ValueError("No research_context/catalog.json is available")
    catalog_path = task_dir / CONTEXT_DIRECTORY / "catalog.json"
    frozen = {
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "sources": {row["id"]: {key: row.get(key) for key in (
            "text_sha256", "download_sha256", "availability", "origin", "url",
        )} for row in catalog["sources"]},
        "integrity_scope": "Saved source bytes and catalog; publisher identity and scientific truth are not authenticated.",
    }
    marker = load_methodology(task_dir)
    if marker is not None:
        frozen["methodology"] = {
            "policy_version": marker["policy_version"], "profile": marker["profile"],
            "rules_sha256": marker["rules_sha256"],
        }
    return frozen


def citation_text(task_dir: Path, sources: dict[str, dict], source_id: str,
                  citation: dict[str, Any], *, require_full_text: bool = True) -> str:
    source = sources.get(source_id)
    if source is None:
        raise ValueError(f"Unknown source: {source_id}")
    if require_full_text and source.get("availability") != "full_text":
        raise ValueError(f"{source_id}: detailed comparison requires complete saved full text")
    if not source.get("text_path"):
        raise ValueError(f"{source_id}: source has no saved text")
    body = (task_dir / CONTEXT_DIRECTORY / source["text_path"]).read_text(encoding="utf-8")
    text = _passage_span(citation.get("passage_id", ""), source_id, body)
    if citation.get("quote") != text:
        raise ValueError(f"{source_id}: quote does not match its exact source span")
    return text


def evaluate_novelty(task_dir: Path, agenda: dict[str, Any], *,
                     reviews: list[dict[str, Any]] | None = None,
                     review_dir: Path | None = None,
                     expected_catalog_sha256: str | None = None,
                     candidate_packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an actionable report; never score novelty by words or citations counted.

    Submitted ``reviews`` remain unverified. Only bound provider response artifacts
    in ``review_dir`` can satisfy the independent-model-review requirement.
    ``supported`` means a bounded proposed contribution passed these checks. It
    does not mean an experiment succeeded, a result is new, or a paper is ready.
    """
    agenda = normalize_agenda(agenda)
    gaps: list[dict[str, str]] = []
    actions: list[dict[str, str]] = []

    def gap(code: str, detail: str, action: str) -> None:
        gaps.append({"code": code, "detail": detail})
        actions.append({"kind": action, "reason": detail})

    frozen: dict[str, Any] = {"agenda_sha256": content_hash(agenda)}
    if candidate_packet is not None:
        frozen["candidate_packet_sha256"] = content_hash(candidate_packet)
        try:
            validate_candidate_packet(candidate_packet)
        except (ValueError, TypeError, KeyError) as exc:
            gap("candidate_evidence_invalid", str(exc), "run_discriminating_experiment")
    catalog: dict[str, Any] = {"sources": []}
    try:
        frozen.update(freeze_evidence(task_dir))
        catalog = load_catalog(task_dir) or catalog
        if expected_catalog_sha256 and frozen["catalog_sha256"] != expected_catalog_sha256:
            gap("catalog_changed", "The literature catalog differs from the frozen review input.", "inspect_full_text")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        gap("source_integrity", str(exc), "inspect_full_text")
    sources = {row["id"]: row for row in catalog["sources"]}
    closest = [row["source_id"] for row in agenda["closest_work"]]
    no_claim = agenda["intent"] in {"replication", "negative_result"}
    required_fields = ("research_question", "proposed_contribution", "why_meaningful", "scope")
    for field in required_fields:
        if not agenda[field].strip():
            gap("missing_" + field, f"State agenda.{field} explicitly.", "revise_contribution")
    if not closest:
        gap("missing_closest_work", "Identify and compare the closest prior work before claiming a difference.", "retrieve_literature")
    if len(closest) != len(set(closest)):
        gap("duplicate_closest_work", "Closest-work comparisons contain duplicate source IDs.", "revise_contribution")
    checked_citations = 0
    for row in agenda["closest_work"]:
        source_id = row["source_id"]
        for field in ("problem_setting", "core_method", "contribution_boundary", "overlap", "difference", "why_meaningful"):
            if not row[field].strip():
                gap("incomplete_comparison", f"{source_id}: state {field}.", "inspect_full_text")
        aspects = set()
        if not row["evidence"]:
            gap("missing_passages", f"{source_id}: comparison has no exact full-text evidence.", "inspect_full_text")
        for evidence in row["evidence"]:
            aspect = evidence.get("aspect")
            if aspect not in EVIDENCE_ASPECTS:
                gap("invalid_evidence_aspect", f"{source_id}: identify the claim supported by each passage.", "inspect_full_text")
                continue
            try:
                citation_text(task_dir, sources, source_id, evidence)
                checked_citations += 1
                aspects.add(aspect)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                gap("invalid_passage", str(exc), "inspect_full_text")
        for aspect in sorted(EVIDENCE_ASPECTS - aspects):
            gap("missing_comparison_evidence", f"{source_id}: supply full-text evidence for {aspect}.", "inspect_full_text")
    selection = {}
    for row in agenda["source_selection"]:
        sid = row.get("source_id")
        if not isinstance(sid, str) or sid not in sources or sid in selection:
            gap("invalid_source_selection", "Source selection contains unknown or duplicate IDs.", "retrieve_literature")
            continue
        selection[sid] = row
        if row.get("relation") not in {"closest", "relevant", "excluded"} or not isinstance(row.get("reason"), str) or not row.get("reason", "").strip():
            gap("source_selection_rationale", f"{sid}: explain its relevance or exclusion.", "retrieve_literature")
        if (row.get("relation") == "closest") != (sid in closest):
            gap("closest_selection_mismatch", f"{sid}: closest-source selection and comparisons disagree.", "inspect_full_text")
    for sid in sorted(set(sources) - set(selection)):
        gap("untriaged_source", f"{sid}: explain whether this retrieved source is closest, relevant, or excluded.", "retrieve_literature")
    scan = agenda["source_scan"]
    if not isinstance(scan.get("queries"), list) or not scan.get("queries") or not all(
        isinstance(query, str) and query.strip() for query in scan.get("queries", [])
    ) or not isinstance(scan.get("coverage_limits"), str) or not scan.get("coverage_limits", "").strip():
        gap("search_scope_missing", "Record search queries and coverage limits; do not imply exhaustive originality.", "retrieve_literature")
    if not agenda["alternatives"]:
        gap("alternative_missing", "State a competing explanation and a test that can distinguish it.", "design_discriminating_experiment")
    for row in agenda["alternatives"]:
        experiment = row.get("discriminating_experiment")
        fields = ("intervention", "controls", "outcome_if_claim", "outcome_if_alternative", "decision_rule")
        if not isinstance(row.get("explanation"), str) or not row.get("explanation", "").strip() or not isinstance(experiment, dict):
            gap("alternative_incomplete", "Give each alternative a concrete discriminating experiment.", "design_discriminating_experiment")
            continue
        for field in fields:
            if not isinstance(experiment.get(field), str) or not experiment.get(field, "").strip():
                gap("experiment_incomplete", f"Discriminating experiment is missing {field}.", "design_discriminating_experiment")
        if experiment.get("outcome_if_claim") == experiment.get("outcome_if_alternative"):
            gap("identical_predictions", "The stated predictions are identical; this test cannot distinguish the explanations as written.", "design_discriminating_experiment")
    for field in ("strong_baselines", "robustness_checks", "independent_confirmation"):
        value = agenda["evidence_plan"].get(field)
        if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
            gap("evidence_plan_incomplete", f"Plan {field}; match comparisons to the claimed contribution.", "design_discriminating_experiment")

    machine_gaps = list(gaps)
    verified_reviews: list[dict[str, Any]] = []
    if review_dir is not None:
        from .novelty_review import load_review_records
        verified_reviews, errors = load_review_records(task_dir, agenda, review_dir, candidate_packet=candidate_packet)
        for error in errors:
            gap("review_receipt_invalid", error, "independent_review")
    if reviews:
        gap("unverified_review", "Submitted reviewer dictionaries are not bound to host-saved provider response receipts.", "independent_review")
    for review in verified_reviews:
        for finding in review["assessment"].get("source_findings", []):
            sid = finding.get("source_id")
            for evidence in finding.get("evidence", []):
                try:
                    citation_text(task_dir, sources, sid, evidence)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    gap("review_citation_invalid", str(exc), "independent_review")
        assessment = review["assessment"]
        if candidate_packet is not None:
            candidate_assessment = assessment.get("candidate_assessment")
            if not isinstance(candidate_assessment, dict) or candidate_assessment.get("candidate_packet_sha256") != frozen["candidate_packet_sha256"]:
                gap("candidate_review_unbound", "Reviewer did not assess the exact selected implementation/outcome packet.", "independent_review")
            else:
                checks = candidate_assessment.get("checks", {})
                if not isinstance(checks, dict) or any(checks.get(key) not in {"pass", "fail", "uncertain"} for key in CANDIDATE_CHECKS):
                    gap("candidate_checks_missing", "Reviewer must check implementation, observed outcomes, and comparison scope.", "independent_review")
                elif assessment["verdict"] == "differentiated" and any(checks[key] != "pass" for key in CANDIDATE_CHECKS):
                    gap("candidate_claim_unimplemented_or_unverified", "Selected code or observed results do not establish the contribution as stated.", "revise_contribution")
                mapping = candidate_assessment.get("claim_code_evidence_map")
                code_paths = {row.get("path") for row in candidate_packet.get("source_files", []) if isinstance(row, dict)}
                if not isinstance(mapping, list) or not mapping:
                    gap("claim_code_evidence_missing", "Reviewer must map claims to actual source files and observed evaluation evidence.", "independent_review")
                else:
                    for row in mapping:
                        if not isinstance(row, dict) or not isinstance(row.get("claim"), str) or not row.get("claim", "").strip() or not isinstance(row.get("finding"), str) or not row.get("finding", "").strip():
                            gap("claim_code_evidence_invalid", "Candidate claim map requires a claim and source-grounded finding.", "independent_review")
                            continue
                        paths = row.get("code_paths")
                        references = row.get("evaluation_refs")
                        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or path not in code_paths for path in paths):
                            gap("claim_code_evidence_invalid", "Candidate claim map cites unknown or missing source files.", "independent_review")
                        if not isinstance(references, list) or not references or any(
                            not isinstance(ref, str) or ref not in {"evaluation.metrics", "evaluation.correct", "evaluation.evidence", "evaluation.manifest", "protocol"} for ref in references
                        ):
                            gap("claim_code_evidence_invalid", "Candidate claim map must cite observed evaluation fields or protocol.", "independent_review")
                        elif any((ref == "protocol" and "protocol" not in candidate_packet) or
                                 (ref.startswith("evaluation.") and ref.split(".", 1)[1] not in candidate_packet.get("evaluation", {}))
                                 for ref in references):
                            gap("claim_code_evidence_invalid", "Candidate claim map cites an absent evaluation/protocol field.", "independent_review")
        findings = {item.get("source_id") for item in assessment.get("source_findings", []) if item.get("evidence")}
        if not set(closest) <= findings:
            gap("review_coverage_missing", "Each independent review must cite evidence for every closest source.", "independent_review")
        if assessment["verdict"] == "differentiated" and any(
            assessment["checks"].get(check) != "pass" for check in REVIEW_CHECKS
        ):
            gap("review_check_unresolved", "A differentiation judgment has failed or uncertain scientific checks.", "independent_review")
        for action in assessment.get("next_actions", []):
            if action.get("kind") in ACTION_TYPES:
                actions.append(action)
    if len(verified_reviews) < 2 and not no_claim:
        gap("independent_reviews_missing", "Obtain two separate, source-grounded model reviews of the same frozen agenda.", "independent_review")
    verdicts = [review["assessment"]["verdict"] for review in verified_reviews]
    status = "needs_evidence"
    if no_claim:
        status = "no_novelty_claim"
    elif not gaps and len(verdicts) >= 2 and all(value == "differentiated" for value in verdicts):
        status = "supported"
    elif not gaps and len(verdicts) >= 2 and all(value == "overlap" for value in verdicts):
        status = "overlap"
        actions.append({"kind": "revise_contribution", "reason": "Independent reviews find the core problem, method, and main contribution coincide. Retain the baseline and seek a substantive new claim."})
    elif len(verdicts) >= 2:
        gap("review_disagreement_or_uncertainty", "Reviews disagree or retain uncertainty. Resolve the cited evidence or narrow the claim.", "independent_review")
    if status == "supported":
        actions.append({"kind": "run_discriminating_experiment", "reason": "Test the proposed difference, alternatives, strong baselines, and independent confirmation. Literature support does not establish experimental success."})
    unique_actions = list({(row["kind"], row["reason"]): row for row in actions}.values())
    return {
        "schema_version": 1, "status": status, "claim_type": agenda["claim_type"],
        "intent": agenda["intent"], "closest_source_ids": closest,
        "machine_checks": {"passed": not machine_gaps, "gaps": machine_gaps,
                           "scope": "Source integrity, agenda structure, and exact source citations; scientific meaning is assessed by reviewers.",
                           "review_artifacts_valid": not any(row["code"] in {
                               "review_receipt_invalid", "review_citation_invalid", "review_coverage_missing",
                           } for row in gaps),
                           "validated_citation_count": checked_citations,
                           "meaning_or_originality_proven": False},
        "model_judgments": verified_reviews, "unverified_judgments": reviews or [],
        "human_approval": {"status": "not_recorded", "required_for_automated_experiments": False},
        "gaps": gaps, "next_actions": unique_actions, "frozen_evidence": frozen,
        "supported_scope": ("The implemented contribution and reported development outcomes were assessed against the exact selected source/evaluation packet; independent confirmation and manuscript validation remain separate."
                            if candidate_packet is not None else
                            "A bounded proposed contribution is supported by the saved literature and independent model judgments; results require separate verification."),
        "independence_limit": "Separate stateless calls, possibly the same model; not independent human peer review.",
        "honest_paper_writing_allowed": True, "acceptance_guaranteed": False,
    }
