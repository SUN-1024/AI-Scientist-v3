"""Evidence-based readiness. Process success and paper existence are separate facts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.orchestrator.state import write_private_text
from src.final_evaluation.runner import FinalEvaluationError
from .config import ResearchConfig
from .harness_state import read_json, sha256, verify_stage_records, now


def _result(status: str, detail: str, **fields: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **fields}


def _investigation(workspace: Path) -> dict[str, Any]:
    from src.investigation.adaptive_runtime import validated_development_outcome
    task = workspace / "run/task"
    root = workspace / "run/investigation"
    verified = []
    for path in sorted(root.rglob("evaluation_manifest.json")):
        try:
            manifest = read_json(path)
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            attempt_id, idea_id = manifest.get("attempt_id"), manifest.get("idea_id")
            if not isinstance(attempt_id, str) or not attempt_id or not isinstance(idea_id, str) or not idea_id:
                continue
            outcome = validated_development_outcome(
                path.parent, task_dir=task, attempt_id=attempt_id,
                idea_id=idea_id, native=manifest.get("execution_boundary") == "host",
            )
            if outcome is not None:
                verified.append({"path": str(path.relative_to(workspace)), **outcome})
        except (OSError, ValueError, TypeError):
            continue
    return _result("verified" if verified else "missing", "Hash-bound host evaluation receipts",
                   verified_evaluation_count=len(verified), evaluations=verified)


def _comparisons(workspace: Path, state: dict[str, Any], cfg: ResearchConfig) -> dict[str, Any]:
    from src.variants_pipeline.pipeline import _load_variant_specs_from_disk, _validate_pipeline_summary
    root = workspace / "run/ablations"
    if not verify_stage_records(workspace, state.get("stages", {}).get("ablations", {})):
        return _result("missing", "No completed, unchanged host-recorded ablation stage")
    try:
        specs = _load_variant_specs_from_disk(root)
        error = _validate_pipeline_summary(root, specs)
        if error:
            return _result("invalid", error)
        summary = read_json(root / "pipeline_summary.json")
        completed = [item for item in summary["variants"] if item.get("status") == "completed" and item.get("rubric_passed") is True]
        ids = {item["variant_id"] for item in completed}
        kinds = {item["kind"] for item in completed}
        missing = sorted(set(cfg.harness.quality.required_comparisons) - ids)
        if "control" not in kinds:
            missing.append("a verified control variant")
        if "baseline" not in kinds:
            missing.append("a verified reference baseline")
        if cfg.harness.quality.require_ablation and "ablation" not in kinds:
            missing.append("a verified mechanism ablation")
        return _result("verified" if not missing else "incomplete", "Canonical comparison artifacts checked against host stage hashes",
                       completed_variants=sorted(ids), missing=missing,
                       independent_confirmation=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result("invalid", str(exc))


def _confirmation(workspace: Path, cfg: ResearchConfig) -> dict[str, Any]:
    from src.final_evaluation.runner import FinalEvaluationError, resolve_frozen_incumbent, verify_final_manifest
    from src.investigation.task_contract import load_task_contract, TASK_CONTRACT_FILENAME
    from src.investigation.evaluation_evidence import load_evaluation_evidence
    task = workspace / "run/task"
    try:
        contract = load_task_contract(task / TASK_CONTRACT_FILENAME)
        program, _ = resolve_frozen_incumbent(workspace / "run/investigation")
        manifest = verify_final_manifest(workspace / "run/final_evaluation", task_dir=task, program_dir=program)
        evidence = load_evaluation_evidence(workspace / "run/final_evaluation/results")
        episodes = evidence["episodes"] if evidence else []
        seeds = {episode["seed"] for episode in episodes}
        expected = set(contract.protocol.final_seeds)
        gaps = []
        if seeds != expected:
            gaps.append("Final observation seeds do not match the frozen contract")
        if seeds.intersection(contract.protocol.search_seeds):
            gaps.append("Final seeds overlap search seeds")
        if len(seeds) < cfg.harness.quality.min_confirmation_seeds:
            gaps.append(f"Only {len(seeds)} confirmation seeds; declared minimum is {cfg.harness.quality.min_confirmation_seeds}")
        if manifest.get("correct") is not True:
            gaps.append("The frozen final execution/correctness criterion was not met")
        return _result("verified" if not gaps else "not_qualified", "Contract-bound final results and candidate hashes checked",
                       manifest_sha256=sha256(workspace / "run/final_evaluation/final_manifest.json"),
                       outcome=manifest.get("outcome", "qualified"), seeds=sorted(seeds), gaps=gaps,
                       metrics=manifest["metrics"], program_tree_sha256=manifest["program_tree_sha256"])
    except (OSError, ValueError, KeyError, TypeError, FinalEvaluationError) as exc:
        return _result("missing_or_invalid", str(exc))


def _portfolio_confirmation(workspace: Path, cfg: ResearchConfig) -> dict[str, Any] | None:
    """Check the entire precommitted method set; preserve legacy single finals."""
    from .portfolio_final_evidence import verify_portfolio_final_for_paper

    workspace = workspace.absolute()
    projection = verify_portfolio_final_for_paper(
        {"results_dir": str(workspace / "run/investigation")}, task_dir=workspace / "run/task",
        investigation_results_dir=workspace / "run/investigation",
        final_evaluation_dir=workspace / "run/final_evaluation",
    )
    if projection is None:
        return None
    if projection["verification_status"] != "host_verified":
        return _result("missing_or_invalid", projection["detail"],
                       collection_complete=False, all_methods_qualified=False,
                       gaps=projection["gaps"], provenance=projection["provenance"])
    gaps = []
    if not projection["collection_complete"] or not projection["all_methods_qualified"]:
        gaps.append("The complete frozen final method set has not met its confirmation protocol")
    for method in projection["methods"]:
        count = len(method["observations"]["seeds"])
        if count < cfg.harness.quality.min_confirmation_seeds:
            gaps.append(f"{method['method_id']} has {count} confirmation seeds; declared minimum is "
                        f"{cfg.harness.quality.min_confirmation_seeds}")
        gaps.extend(f"{method['method_id']}: {detail}" for detail in method["observations"]["gaps"])
    controls = [method for method in projection["methods"] if method["is_control"]]
    return _result("verified" if not gaps else "incomplete",
                   "All frozen final methods, host manifests and seed observations revalidated",
                   collection_complete=projection["collection_complete"], all_methods_qualified=projection["all_methods_qualified"],
                   selected_proposal_root_id=projection["selected_proposal_root_id"],
                   observed_configurations_complete=projection["observed_configurations_complete"],
                   tuned_controls_verified=bool(controls) and all(method["tuned"] is True
                       and method["actual_evaluations"]["tuning"] > 0 for method in controls),
                   methods=projection["methods"], paired_comparison=projection["paired_comparison"],
                   gaps=gaps, evidence_gaps=projection["gaps"], provenance=projection["provenance"],
                   matched_hpo_verified=False)


def _evidence_bundle(workspace: Path) -> dict[str, Any] | None:
    """Verify a shipped reproduction bundle; absent bundles are not a gap."""
    from .evidence_bundle import verify_bundle

    paper = workspace / "run/paper"
    bundles = sorted(paper.glob("reproduction-bundle-*.tar.gz"))
    if not bundles:
        return None
    if len(bundles) > 1:
        return _result("missing_or_invalid", "More than one reproduction bundle is present; keep exactly one")
    manuscript = paper / "paper.tex"
    if not manuscript.is_file() or manuscript.is_symlink():
        return _result("missing_or_invalid", "A reproduction bundle ships without a regular paper.tex")
    try:
        report = verify_bundle(bundles[0], manuscript=manuscript)
    except (OSError, ValueError, TypeError) as exc:
        return _result("missing_or_invalid", str(exc))
    return _result("verified" if report["status"] == "verified" else "failed",
                   "Bundle files match MANIFEST.json and the manuscript cites the manifest hash",
                   archive=str(bundles[0].relative_to(workspace)),
                   checked_files=report["checked_files"],
                   manifest_sha256=report["manifest_sha256"],
                   manuscript_cites_manifest=report.get("manuscript_cites_manifest"))


def _paper(workspace: Path, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.pipeline.config import validate_review_input
    from src.orchestrator.paper_finalize import _paper_audit_is_current_and_passing
    paper = workspace / "run/paper"
    try:
        errors = validate_review_input(workspace / "run", paper)
        if errors:
            return _result("missing_or_invalid", "; ".join(errors)), _result("pending", "No validated manuscript")
        validation = read_json(paper / "validation_report.json")
        stages = state.get("stages", {})
        paper_sealed = any(verify_stage_records(workspace, stages.get(phase, {})) for phase in ("paper", "revision"))
        artifact_ok = validation.get("ok") is True and paper_sealed
        audit_ok = artifact_ok and _paper_audit_is_current_and_passing(paper)
        artifact = _result("verified" if artifact_ok else "draft", "Manuscript validation and host artifact hashes",
                           pdf=str(paper / "paper.pdf"), tex=str(paper / "paper.tex"),
                           pdf_sha256=sha256(paper / "paper.pdf"))
        claims = _result("verified" if audit_ok else "pending_or_failed", "Current claim/evidence audit; model judgment is not a scientific proof")
        return artifact, claims
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _result("missing_or_invalid", str(exc)), _result("pending", "No current claim/evidence audit")


def _review(workspace: Path, cfg: ResearchConfig, state: dict[str, Any]) -> dict[str, Any]:
    from src.investigation.review_summary import parse_review_summary
    paper = workspace / "run/paper"
    try:
        stages = state.get("stages", {})
        if not any("run/paper/ai_review.json" in stages.get(phase, {}).get("artifacts", {}) and
                   "run/paper/paper.pdf" in stages.get(phase, {}).get("artifacts", {}) and
                   verify_stage_records(workspace, stages.get(phase, {})) for phase in ("review", "revision")):
            return _result("pending_or_invalid", "No unchanged host-recorded independent review stage")
        payload = read_json(paper / "ai_review.json")
        provenance = payload.get("Provenance", {})
        current = provenance.get("paper_sha256") == sha256(paper / "paper.pdf")
        score, decision = parse_review_summary(payload)
        if not current or provenance.get("independent_context") is not True or score is None or decision is None:
            return _result("pending_or_invalid", "Review needs valid scores, fresh independent context, and the current PDF hash")
        invocations = provenance.get("invocations", [])
        reviews = provenance.get("reviews", [])
        if (provenance.get("backend") != "agent" or not invocations or not reviews
                or any(item.get("status") != "completed" for item in invocations)):
            return _result("pending_or_invalid", "Independent reviewer invocation receipts are missing or incomplete")
        concerns = [item for review in [payload, *reviews] for item in review.get("Concerns", [])]
        unresolved = [item for item in concerns if isinstance(item, dict) and item.get("severity", "").lower() in {"critical", "major", "blocking"}]
        requested = [item for review in [payload, *reviews] for item in review.get("Requested Experiments", [])]
        passed = score >= cfg.harness.quality.min_review_score and decision == "Accept" and not unresolved and not requested
        return _result("passed" if passed else "changes_required", "Independent model review of the current PDF; not a venue decision",
                       score=score, decision=decision, unresolved_concerns=unresolved,
                       requested_experiments=requested, provenance=provenance)
    except (OSError, ValueError, TypeError) as exc:
        return _result("pending_or_invalid", str(exc))


def assess_readiness(workspace: Path, cfg: ResearchConfig, state: dict[str, Any]) -> dict[str, Any]:
    artifact, claims = _paper(workspace, state)
    checks = {"artifact": artifact, "experiment": _investigation(workspace),
              "comparison": _comparisons(workspace, state, cfg), "confirmation": _confirmation(workspace, cfg),
              "claim": claims, "review": _review(workspace, cfg, state)}
    portfolio_confirmation = _portfolio_confirmation(workspace, cfg)
    if portfolio_confirmation is not None:
        checks["portfolio_final_comparison"] = portfolio_confirmation
    from .upstream.disclosure import load_integration_marker, inspect_pdf_disclosure
    try:
        disclosure_required = load_integration_marker(workspace / "input_task") is not None
        checks["disclosure"] = inspect_pdf_disclosure(workspace / "run/paper/paper.pdf", required=disclosure_required)
    except (OSError, ValueError, TypeError) as exc:
        disclosure_required = True
        checks["disclosure"] = _result("missing_or_invalid", str(exc))
    from .methodology import assess_methodology

    try:
        methodology = assess_methodology(workspace / "input_task")
        if ((cfg.methodology.enabled or methodology.get("enabled") is True)
                and ((workspace / "harness_state.json").exists() or (workspace / "harness_state.json").is_symlink())):
            from .evidence import verify_methodology_for_paper

            verified = verify_methodology_for_paper(
                {"results_dir": str(workspace.absolute() / "run")},
                investigation_results_dir=workspace / "run",
            )
            if verified is not None:
                methodology = {key: verified[key] for key in (
                    "enabled", "status", "policy_version", "rules_sha256", "profile",
                    "gaps", "actions", "scope", "detail", "verification_status", "provenance",
                ) if key in verified}
        if cfg.methodology.enabled and methodology.get("enabled") is not True:
            methodology = _result(
                "needs_evidence", "Enabled research methodology has no frozen marker", enabled=True,
                policy_version=None, rules_sha256=None,
                scope="Declared methodology coverage only; not scientific validity certification.",
                actions=["Recover the unchanged frozen methodology marker for this enabled study before review."],
            )
    except (OSError, ValueError, TypeError) as exc:
        methodology = _result(
            "needs_evidence", str(exc), enabled=True,
            policy_version=None, rules_sha256=None,
            scope="Declared methodology coverage only; not scientific validity certification.",
            actions=["Recover the frozen methodology rules and declared study design before review."],
        )
    if methodology.get("enabled") is True:
        methodology.update(assessment_scope="frozen_initial_agenda",
                           selected_candidate_design_assessed=False,
                           scientific_validity_certified=False)
        checks["methodology"] = methodology
    from .evidence import verify_postdraft_execution_agenda

    postdraft = verify_postdraft_execution_agenda(workspace)
    if postdraft is not None:
        checks["postdraft_execution_agenda"] = postdraft
    evidence_bundle = _evidence_bundle(workspace)
    if evidence_bundle is not None:
        checks["evidence_bundle"] = evidence_bundle
    quality = cfg.harness.quality
    effect = (portfolio_confirmation.get("paired_comparison", {}).get("aggregate_metrics", {})
              if portfolio_confirmation is not None else checks["confirmation"].get("metrics", {}))
    if not quality.require_positive_final:
        checks["scientific_effect"] = _result("not_required", "The declared study does not require a positive superiority result")
    elif portfolio_confirmation is not None and (
        portfolio_confirmation.get("status") != "verified"
        or portfolio_confirmation.get("selected_proposal_root_id") is None
        or portfolio_confirmation.get("tuned_controls_verified") is not True
        or portfolio_confirmation.get("observed_configurations_complete") is not True
        or portfolio_confirmation.get("paired_comparison", {}).get("status") != "verified"
    ):
        checks["scientific_effect"] = _result("not_qualified",
            "Proposal superiority requires a complete qualified final set, a development-selected proposal, "
            "every declared control's tuning trials, matching development/final configurations, and verified paired control comparisons. "
            "Default diagnostic baseline metrics cannot satisfy this requirement.",
            evidence_source="research_portfolio_final.paired_comparison")
    elif not quality.positive_effect_metric or not quality.positive_effect_interval_low_metric:
        checks["scientific_effect"] = _result("undeclared", "Declare a signed candidate-minus-reference metric and its uncertainty lower-bound metric before the study; correct=true is not evidence of improvement")
    elif quality.positive_effect_metric not in effect or quality.positive_effect_interval_low_metric not in effect:
        checks["scientific_effect"] = _result("missing",
            "The frozen portfolio paired-control summary lacks the declared effect and interval metrics; diagnostic metrics are not substitutes"
            if portfolio_confirmation is not None else
            "The sealed final manifest lacks the declared comparative effect and interval metrics")
    else:
        estimate, lower = effect[quality.positive_effect_metric], effect[quality.positive_effect_interval_low_metric]
        positive = estimate > quality.positive_effect_min and lower > quality.positive_effect_min
        checks["scientific_effect"] = _result("positive" if positive else "not_positive",
            "Signed comparative effect and prespecified interval bound, separate from executable correctness",
            estimate=estimate, interval_lower=lower, threshold=quality.positive_effect_min)
        if portfolio_confirmation is not None:
            checks["scientific_effect"]["evidence_source"] = "research_portfolio_final.paired_comparison"
    try:
        from .novelty_review import load_frozen_novelty_report
        from .harness_evidence import build_candidate_packet
        from .novelty import content_hash
        pointer = read_json(workspace / "novelty/latest.json")
        packet = build_candidate_packet(workspace)
        if pointer.get("candidate_packet_sha256") != content_hash(packet):
            raise ValueError("Novelty reviews do not cover the selected code, protocol, and host-scored results")
        relative = Path(pointer.get("cycle", ""))
        cycle = workspace / relative
        if not pointer.get("cycle") or relative.is_absolute() or not cycle.resolve().is_relative_to((workspace / "novelty").resolve()):
            raise ValueError("Selected candidate has no bound novelty review cycle")
        novelty = load_frozen_novelty_report(workspace / "input_task", cycle, candidate_packet=packet)
        checks["novelty"] = novelty
    except (ImportError, OSError, ValueError, TypeError, FinalEvaluationError) as exc:
        checks["novelty"] = _result("needs_evidence", str(exc))
    expected = {"artifact": "verified", "experiment": "verified", "comparison": "verified", "confirmation": "verified",
                "claim": "verified", "review": "passed", "novelty": "supported",
                "disclosure": "verified" if disclosure_required else "not_required",
                "scientific_effect": "positive" if quality.require_positive_final else "not_required"}
    if "methodology" in checks:
        expected["methodology"] = "declared"
    if postdraft is not None:
        expected["postdraft_execution_agenda"] = "declared"
    if "evidence_bundle" in checks:
        expected["evidence_bundle"] = "verified"
    if portfolio_confirmation is not None:
        expected["portfolio_final_comparison"] = "verified"
    gaps = [{"check": key, "status": checks[key].get("status"), "detail": checks[key].get("detail", checks[key].get("gaps", []))}
            for key, status in expected.items() if checks[key].get("status") != status]
    target = state.get("quality_target", cfg.harness.quality_target)
    if target != "high_impact":
        gaps.append({"check": "quality_target", "status": "development", "detail": "This run was configured as a development study"})
    complete = not gaps
    final_opened = (workspace / "run/final_evaluation/invocation.json").exists()
    actions = []
    action_text = {
        "artifact": ("repair_manuscript", "Inspect the paper validation report and repair the editable manuscript and PDF."),
        "experiment": ("verify_development", "Recover host-scored development evaluations; do not infer success from model logs."),
        "comparison": ("controlled_comparisons", "Complete the declared controls, reference baselines, and mechanism ablations."),
        "confirmation": ("independent_confirmation", "Use the frozen one-time final evaluator and report all held-out seed observations."),
        "portfolio_final_comparison": ("independent_confirmation", "Recover and report every method in the frozen final comparison set. Preserve all completed and failed final invocations; do not rerun completed methods or choose a new winner from final outcomes."),
        "scientific_effect": ("new_study", "Report the comparative effect and uncertainty honestly. A failed superiority criterion does not justify tuning on confirmation observations."),
        "claim": ("audit_claims", "Reconcile every numerical and causal claim with current evidence and rerun the manuscript audit."),
        "disclosure": ("repair_manuscript", "Add the required AI Scientist statement to a visible Disclosure, Abstract, or Methods section, recompile, and rerun validation."),
        "review": ("resolve_review", "Resolve each independent reviewer concern; text edits must not invent missing experiments."),
        "novelty": ("verify_novelty", "Read the closest full methods, resolve source gaps, and test the stated discriminating hypothesis."),
        "methodology": ("methodology_coverage", "Resolve the declared study-design gaps using existing evidence. Methodology coverage does not certify causal or clinical validity. Preserve the frozen confirmation protocol; design changes after confirmation require a new study."),
        "postdraft_execution_agenda": ("methodology_coverage", "Resolve the reviewed execution agenda's declared design gaps and recover its frozen draft, review, publication, and plan bindings. This coverage check does not assess the selected implementation's actual design."),
        "evidence_bundle": ("repair_manuscript", "Rebuild the reproduction bundle from the current artifacts, cite its MANIFEST.json hash in the manuscript, and re-run verification."),
        "quality_target": ("new_study", "Declare a full research study before making a high-impact readiness claim."),
    }
    for gap in gaps:
        kind, instruction = action_text[gap["check"]]
        new_study = final_opened and gap["check"] in {"experiment", "confirmation", "portfolio_final_comparison", "novelty", "scientific_effect"}
        if gap["check"] == "confirmation" and final_opened:
            instruction = "Keep the completed confirmation result. Any method change requires a new study and a new frozen confirmation protocol."
        actions.append({"check": gap["check"], "kind": kind, "instruction": instruction,
                        "requires_new_study": new_study})
    for requested in checks["review"].get("requested_experiments", []):
        actions.append({"check": "review", "kind": "requested_experiment", "request": requested,
                        "requires_new_study": final_opened})
    actions.extend({"check": "novelty", "kind": "novelty_followup", "request": item,
                    "requires_new_study": final_opened and item.get("kind") in {"discriminating_experiment", "reformulate_contribution"}}
                   for item in checks["novelty"].get("next_actions", []) if isinstance(item, dict))
    actions.extend({"check": "methodology", "kind": "methodology_followup", "instruction": item,
                    "requires_new_study": final_opened}
                   for item in checks.get("methodology", {}).get("actions", []) if isinstance(item, str))
    actions.extend({"check": "postdraft_execution_agenda", "kind": "methodology_followup", "instruction": item,
                    "requires_new_study": final_opened}
                   for item in (postdraft or {}).get("actions", []) if isinstance(item, str))
    note = "All machine checks passing permits human scientific review. It does not promise novelty, journal acceptance, or authorize submission."
    if "methodology" in checks:
        actions.append({
            "check": "methodology", "kind": "compare_selected_design",
            "instruction": "Compare the selected implementation and actual controls with the frozen initial study design. This coverage check assesses the initial agenda only. Preserve the final protocol; changes after confirmation require a new study.",
            "requires_new_study": False,
        })
        note += " Methodology coverage checks the frozen initial agenda only, not the selected candidate's actual design. It does not certify scientific, causal, or clinical validity."
    if postdraft is not None:
        actions.append({
            "check": "postdraft_execution_agenda", "kind": "compare_selected_design",
            "instruction": "Compare the selected implementation and actual controls with the separately frozen postdraft execution agenda. Declared agenda coverage does not verify that the implementation follows it.",
            "requires_new_study": False,
        })
        note += " Postdraft execution-agenda coverage is reported separately and does not certify the selected candidate's design."
    return {
        "schema_version": 1, "assessed_at": now(), "quality_target": target,
        "status": "ready_for_human_review" if complete else "evidence_gaps",
        "checks": checks, "gaps": gaps, "next_actions": actions,
        "confirmation_protocol_frozen": final_opened, "ready_for_human_review": complete,
        "submission_ready": False, "human_approval": "required", "submitted": False,
        "note": note,
    }


def write_readiness(workspace: Path, cfg: ResearchConfig, state: dict[str, Any]) -> dict[str, Any]:
    report = assess_readiness(workspace, cfg, state)
    write_private_text(workspace / "paper_readiness.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    next_actions: dict[str, Any] = {
        "schema_version": 1, "status": "proposed_not_executed", "confirmation_protocol_frozen": report["confirmation_protocol_frozen"],
        "actions": report["next_actions"], "automatic_confirmation_set_tuning": False,
    }
    methodology = report["checks"].get("methodology")
    if methodology is not None:
        next_actions["methodology"] = {key: methodology.get(key) for key in (
            "status", "policy_version", "rules_sha256", "scope", "assessment_scope",
            "selected_candidate_design_assessed", "scientific_validity_certified",
        )}
    postdraft = report["checks"].get("postdraft_execution_agenda")
    if postdraft is not None:
        next_actions["postdraft_execution_agenda"] = {key: postdraft.get(key) for key in (
            "status", "verification_status", "policy_version", "rules_sha256", "scope", "assessment_scope",
            "provenance", "selected_candidate_design_assessed", "scientific_validity_certified",
        )}
    write_private_text(workspace / "research_next_actions.json", json.dumps(next_actions, ensure_ascii=False, indent=2) + "\n")
    rows = ["# Research harness evidence readiness", "", f"State: **{report['status']}**", "", "| Check | Status |", "|---|---|"]
    rows += [f"| {key} | {value.get('status', 'unknown')} |" for key, value in report["checks"].items()]
    if methodology is not None:
        rows += ["", f"Methodology policy: {methodology.get('policy_version', 'unverified')}; "
                 f"rules SHA-256: `{methodology.get('rules_sha256', 'unverified')}`.",
                 str(methodology.get("scope", "Declared methodology coverage only."))]
    rows += ["", report["note"], "", "## Evidence gaps", ""]
    rows += [f"- {item['check']}: {item['detail']}" for item in report["gaps"]] or ["- No machine-detected gaps. Human review is required."]
    rows += ["", "## Next actions", ""]
    rows += [f"- {item.get('instruction', json.dumps(item.get('request', {}), ensure_ascii=False))}" +
             (" A new study is required." if item.get("requires_new_study") else "") for item in report["next_actions"]]
    write_private_text(workspace / "paper_readiness.md", "\n".join(rows) + "\n")
    return report
