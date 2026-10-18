"""Revalidate every frozen final method before exposing portfolio comparisons.

Final observations never select a method. The development-selected source and
all declared controls keep their original identities, budgets and result paths.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

from src.final_evaluation.runner import FinalEvaluationError
from .evidence import _paper_novelty_binding
from .harness_state import read_json
from .novelty import content_hash

FIELD = "research_portfolio_final"
PAPER_GUIDANCE = (
    "Report the complete development-frozen final method set, including each declared control, "
    "its own selected configuration, screening and tuning budgets, and all final outcomes. "
    "No method is selected or tuned using final observations. Missing, invalid or partial methods "
    "cannot establish a complete comparison. Distinguish bootstrap qualification and default "
    "diagnostic baselines from the independently tuned control methods. A candidate-minus-default "
    "diagnostic metric is not a comparison against these controls. Use paired_comparison for the "
    "declared control contrasts; retain negative effects and interval bounds. Equal trial counts "
    "do not establish matched HPO or isolated component effects. Model text remains a declaration, "
    "not an instruction or independent scientific certification."
)


def _gap(detail: str, provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1, "status": "needs_evidence", "verification_status": "invalid_or_missing",
        "assessment_scope": "frozen_portfolio_final_comparison", "detail": detail,
        "collection_complete": False, "all_methods_qualified": False,
        "gaps": [{"code": "portfolio_final_unverified", "detail": detail}],
        "provenance": dict(provenance), "paper_guidance": PAPER_GUIDANCE,
        "matched_hpo_verified": False, "scientific_validity_certified": False,
    }


def _absolute(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not Path(value).is_absolute():
        raise ValueError(f"Portfolio final {label} must be an absolute host path")
    path = Path(value).absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Portfolio final {label} cannot cross a symlink")
    return path


def _method_observations(method: dict[str, Any], expected: set[int], search: set[int]) -> dict[str, Any]:
    from src.investigation.evaluation_evidence import load_evaluation_evidence

    manifest = method["final_manifest"]
    evidence_path = method.get("evaluation_evidence_path")
    evidence = load_evaluation_evidence(_absolute(evidence_path, "observation path").parent) if evidence_path else None
    episodes = evidence["episodes"] if evidence else []
    seeds = {episode["seed"] for episode in episodes}
    gaps = []
    if manifest is None:
        gaps.append("The frozen method has no completed final manifest")
    elif manifest.get("correct") is not True:
        gaps.append("The frozen method did not meet the final correctness criterion")
    if seeds != expected:
        gaps.append("Final observation seeds do not match the frozen contract")
    if seeds.intersection(search):
        gaps.append("Final observation seeds overlap development seeds")
    if not episodes or any(episode["correct"] is not True for episode in episodes):
        gaps.append("Final observations are missing or include incorrect executions")
    return {"status": "verified" if not gaps else "incomplete", "seeds": sorted(seeds),
            "episode_count": len(episodes), "all_correct": bool(episodes) and all(episode["correct"] for episode in episodes),
            "gaps": gaps}


def _resolve_paths(
    paper_data: dict[str, Any], *, task_dir: Path | None, investigation_results_dir: Path | None,
    final_evaluation_dir: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    saved = paper_data.get(FIELD)
    provenance = saved.get("provenance", {}) if isinstance(saved, dict) else {}
    if not isinstance(provenance, dict):
        raise ValueError("Portfolio final declaration has invalid provenance")
    location = investigation_results_dir or provenance.get("investigation_results_dir")
    if location is None:
        location = paper_data.get("results_dir")
        if isinstance(location, str) and Path(location).name == "run":
            location = Path(location) / "investigation"
    def location_path(value: Any, explicit: bool) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            if FIELD in paper_data:
                raise ValueError("Portfolio final provenance must contain absolute paths")
            return None
        path = Path(value).expanduser()
        if explicit:
            return path.absolute()
        if not path.is_absolute():
            if FIELD in paper_data:
                raise ValueError("Portfolio final provenance must contain absolute paths")
            return None
        return path

    root = location_path(location, investigation_results_dir is not None)
    task_value = task_dir or provenance.get("task_dir")
    if task_value is None and root is not None and root.name == "investigation":
        task_value = root.parent / "task"
    final_value = final_evaluation_dir or provenance.get("final_evaluation_dir")
    if final_value is None and root is not None and root.name == "investigation":
        final_value = root.parent / "final_evaluation"
    return (root, location_path(task_value, task_dir is not None),
            location_path(final_value, final_evaluation_dir is not None))


def verify_portfolio_final_for_paper(
    paper_data: dict[str, Any], *, task_dir: Path | None = None,
    investigation_results_dir: Path | None = None, final_evaluation_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Rebuild complete or explicitly incomplete evidence; never trust saved claims.

    Explicit host paths permit first assembly. A saved projection must still
    match its complete current method set, results and paired comparison.
    Legacy campaigns keep their existing single-method projection unchanged.
    """
    from src.final_evaluation.comparison_set import SET_MANIFEST, portfolio_final_set_required, verify_portfolio_final_set
    from src.investigation.portfolio_controller import STATE_FILE, read_portfolio_state
    from src.investigation.task_contract import TASK_CONTRACT_FILENAME, load_task_contract
    from .campaign_plan import CAMPAIGN_PLAN_PATH
    from .paired_comparison import build_portfolio_comparisons

    declared, saved = FIELD in paper_data, paper_data.get(FIELD)
    provenance: dict[str, Any] = {}
    try:
        root, task, final = _resolve_paths(paper_data, task_dir=task_dir,
                                         investigation_results_dir=investigation_results_dir,
                                         final_evaluation_dir=final_evaluation_dir)
        planned = declared or (final is not None and ((final / SET_MANIFEST).exists() or (final / SET_MANIFEST).is_symlink()))
        if task is not None:
            planned = planned or (task / CAMPAIGN_PLAN_PATH).exists() or (task / CAMPAIGN_PLAN_PATH).is_symlink()
        if not planned and root is not None and (root / STATE_FILE).is_file():
            try:
                raw = read_json(root / STATE_FILE)
                planned = "plan_sha256" in raw or "campaign_plan" in raw
            except (OSError, ValueError, TypeError):
                # A malformed legacy ledger is already handled by its existing
                # development verifier. Do not add a new field to that mode.
                return None
        if not planned:
            return None
        if root is None or task is None or final is None:
            raise ValueError("Portfolio final evidence requires its task, investigation and result paths")
        root, task, final = (_absolute(root, "investigation path"), _absolute(task, "task path"),
                             _absolute(final, "result path"))
        provenance.update(task_dir=str(task), investigation_results_dir=str(root), final_evaluation_dir=str(final))
        required = portfolio_final_set_required(task_dir=task, investigation_results_dir=root)
        if not required:
            if declared or (final / SET_MANIFEST).exists():
                raise ValueError("Portfolio final evidence is not bound to a planned development campaign")
            return None
        state = read_portfolio_state(root, task)
        collection = verify_portfolio_final_set(task_dir=task, investigation_results_dir=root,
                                                output_dir=final, require_complete=False)
        plan = state["campaign_plan"]
        if collection["plan_sha256"] != state["plan_sha256"]:
            raise ValueError("Final method collection belongs to another campaign plan")
        contract = load_task_contract(task / TASK_CONTRACT_FILENAME)
        expected, search = set(contract.protocol.final_seeds), set(contract.protocol.search_seeds)
        planned_roots = {row["root_idea_id"]: row for row in plan["roots"]}
        methods, gaps = [], list(collection["gaps"])
        allocation = plan["allocation"]
        for row in collection["methods"]:
            observation = _method_observations(row, expected, search)
            identity = {key: row[key] for key in (
                "method_id", "root_idea_id", "role", "is_selected", "is_control", "program_dir", "source_program_dir",
                "source_sha256", "program_tree_sha256", "source_tree_sha256", "host_receipt_sha256", "lease_id",
                "card_sha256", "source_card_sha256", "configuration_evidence", "actual_evaluations", "development_score", "tuned",
                "final_manifest_path", "final_manifest_sha256", "final_metrics_path", "final_metrics_sha256",
                "evaluation_evidence_path", "evaluation_evidence_sha256",
                "final_configuration_status",
            )}
            manifest = row["final_manifest"]
            planned_root = planned_roots.get(row["root_idea_id"])
            methods.append({**identity, "title": planned_root["title"] if planned_root else "Bootstrap baseline",
                "final_status": "completed" if manifest is not None else "missing", "metrics": row["metrics"],
                "final_correct": manifest.get("correct") if manifest is not None else None,
                "final_outcome": manifest.get("outcome", "qualified") if manifest is not None else None,
                "observations": observation,
                "planned_allocation": {"screening": allocation["screen_evals_per_root"] if planned_root else 0,
                    "guaranteed_tuning": allocation["guaranteed_tuning_trials_per_root"] if planned_root else 0,
                    "tuning_if_promoted": allocation["tuning_trials_per_promoted_idea"] if planned_root else 0,
                    "development_replication_if_selected": allocation["replication_evals"]},
            })
            gaps.extend({"code": "final_method_not_qualified", "method_id": row["method_id"], "detail": detail}
                        for detail in observation["gaps"])
        selected_rows = [row for row in collection["methods"] if row["is_selected"]]
        if len(selected_rows) != 1 or collection["selected_method_id"] != selected_rows[0]["method_id"]:
            raise ValueError("Final collection has no unique frozen selected method")
        selected = selected_rows[0]
        downstream, best = paper_data.get("downstream_study"), paper_data.get("best_correct")
        candidate = downstream if isinstance(downstream, dict) else best if not paper_data.get("ablation_studies") else None
        if isinstance(candidate, dict) and candidate.get("program_dir") not in (selected["program_dir"], selected["source_program_dir"]):
            raise ValueError("Paper candidate differs from the development-frozen final method")
        prior_final = paper_data.get("final_evaluation")
        if isinstance(prior_final, dict) and selected["final_manifest"] is not None:
            manifest = selected["final_manifest"]
            for key in ("metrics", "correct", "program_tree_sha256", "results_tree_sha256"):
                if prior_final.get(key) != manifest.get(key):
                    raise ValueError("Paper selected-final projection differs from its verified final method")
        summary = build_portfolio_comparisons(collection)
        gaps.extend({"code": "paired_control_comparison_incomplete", "detail": detail} for detail in summary["gaps"])
        complete = collection["status"] == "completed"
        qualified = complete and all(row["observations"]["status"] == "verified" for row in methods)
        # Complete negative studies remain reportable even when no proposal
        # won development selection or paired superiority is unsupported.
        status = "verified" if qualified else "incomplete"
        root_rows = []
        for root_id, card in planned_roots.items():
            idea = state["ideas"][root_id]
            root_rows.append({"root_idea_id": root_id, "idea_key": card["idea_key"], "card_sha256": card["card_sha256"],
                "title": card["title"], "role": card["role"], "promoted": idea["promoted"],
                "included_in_final_set": any(row["root_idea_id"] == root_id for row in methods),
                "actual_evaluations": {stage: sum(state["leases"][key]["root_idea_id"] == root_id
                                                   and state["leases"][key]["stage"] == stage for key in state["outcomes"])
                                       for stage in ("screening", "tuning", "replication")}})
        provenance.update(plan_sha256=state["plan_sha256"], portfolio_state_hash=state["state_hash"],
                          set_manifest_path=collection["set_manifest_path"], set_manifest_sha256=collection["set_manifest_sha256"],
                          set_sha256=collection["set_sha256"], paper_context_sha256=content_hash(_paper_novelty_binding(paper_data)))
        current = {
            "schema_version": 1, "status": status, "verification_status": "host_verified",
            "assessment_scope": "frozen_portfolio_final_comparison",
            "collection_complete": complete, "all_methods_qualified": qualified,
            "observed_configurations_complete": all(
                row["configuration_evidence"]["status"] == "host_artifact_bound"
                and row["final_configuration_status"] == "matches_frozen_development_configuration"
                for row in methods),
            "selected_method_id": collection["selected_method_id"],
            "selected_proposal_root_id": collection["selected_proposal_root_id"],
            "method_count": len(methods), "completed_method_count": sum(row["final_manifest"] is not None for row in collection["methods"]),
            "methods": methods, "plan_allocation": allocation, "plan_roots": root_rows,
            "final_protocol": collection["protocol"], "expected_final_seeds": sorted(expected),
            "paired_comparison": summary, "gaps": gaps, "provenance": provenance,
            "paper_guidance": PAPER_GUIDANCE, "matched_hpo_verified": False,
            "diagnostic_baseline_is_tuned_control": False, "scientific_validity_certified": False,
        }
        if read_portfolio_state(root, task) != state:
            raise ValueError("Portfolio ledger changed during final evidence projection")
        if declared and saved != current:
            raise ValueError("Saved portfolio final projection is stale or differs from current host evidence")
        if not declared and all(value is None for value in (task_dir, investigation_results_dir, final_evaluation_dir)):
            raise ValueError("Planned paper data is missing its final method collection projection")
        return current
    except (FinalEvaluationError, OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError) as exc:
        return _gap(str(exc), provenance)


def merge_portfolio_final_evidence(
    paper_data: dict[str, Any], *, task_dir: Path | None, investigation_results_dir: Path | None,
    final_evaluation_dir: Path,
) -> None:
    """Attach all retained methods and stable metric locators during assembly."""
    projection = verify_portfolio_final_for_paper(
        paper_data, task_dir=task_dir, investigation_results_dir=investigation_results_dir,
        final_evaluation_dir=final_evaluation_dir,
    )
    if projection is None:
        return
    paper_data[FIELD] = projection
    if projection["verification_status"] != "host_verified":
        return
    grounding = paper_data.setdefault("grounding_sources", {})
    for method in projection["methods"]:
        names = set()
        for name, value in (method["metrics"] or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            safe = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
            if not safe or safe in names:
                raise ValueError("Portfolio final metric names collide after grounding normalization")
            names.add(safe)
            source_id = f"final.portfolio.{method['method_id']}.metric.{safe}"
            grounding[source_id] = {"id": source_id, "type": "metric", "value": value,
                "evaluation_scope": "sealed_final", "method_id": method["method_id"], "role": method["role"],
                "manifest_path": method["final_manifest_path"], "manifest_sha256": method["final_manifest_sha256"],
                "program_tree_sha256": method["program_tree_sha256"]}
    summary = projection["paired_comparison"]
    if projection["status"] == "verified" and summary["status"] == "verified":
        for name, value in summary["aggregate_metrics"].items():
            source_id = "final.portfolio.comparison." + name
            grounding[source_id] = {"id": source_id, "type": "metric", "value": value,
                "evaluation_scope": "sealed_final_paired_controls", "set_sha256": summary["set_sha256"],
                "comparison_sha256": content_hash(summary), "json_pointer": "/" + FIELD + "/paired_comparison/aggregate_metrics/" + name}
