"""Resolved, cycle-specific campaign context shared by idea and coding prompts."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.orchestrator import state as state_store
from .config import InvestigationConfig

BUDGET_FILE = "effective_campaign_budget.json"


def write_campaign_budget(
    results_dir: Path, cfg: InvestigationConfig, *, cycle: int,
    max_program_evals_per_attempt: int, outer_state: dict[str, Any],
    selected_agent: str | None = None, selected_model: str | None = None,
    adaptive_context: dict[str, Any] | None = None,
    portfolio_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh from resolved config on every cycle, including resumed campaigns."""
    execution = cfg.execution
    total = execution.total_cycles or execution.max_idea_cycles
    position = execution.cycle_offset + cycle
    workers = execution.campaign_workers or execution.workers
    # Advisory scheduling only: no hidden changes to scientific evaluation settings.
    fraction = position / max(total, 1)
    stage = "exploration" if fraction <= 0.4 else "refinement" if fraction <= 0.8 else "confirmation"
    records = state_store.tail_jsonl(state_store.attempts_history_path(results_dir), max_lines=100000)
    attempts = [row for row in records if isinstance(row, dict)]
    evaluated = [row for row in attempts if row.get("evaluation_performed") is True or ("evaluation_performed" not in row and isinstance(row.get("metrics"), dict))]
    budget = {
        "schema_version": 1,
        "source": "resolved_execution_config_and_persisted_history",
        "cycle": {
            "local_index": cycle,
            "local_total": execution.max_idea_cycles,
            "local_remaining_including_current": max(0, execution.max_idea_cycles - cycle + 1),
            "global_position": position,
            "global_total": total,
            "global_remaining_including_current": max(0, total - position + 1) if workers == 1 else None,
            "global_remaining_note": "Unknown across concurrent worker leases" if workers > 1 else "Scheduled cycles, not completed experiments",
            "resume": execution.resume,
            "persisted_current_cycle": outer_state.get("current_cycle", 0),
        },
        "workers": {"campaign_workers": workers, "worker_index": execution.worker_index, "scope": "this worker; global cycles are shared, not multiplied by workers"},
        "search": {"strategy": execution.search_strategy, "width": execution.search_width, "explore_every": execution.search_explore_every, "width_meaning": "retained parent frontier, not candidates per cycle"},
        "evaluation": {
            "mode": "host_harness" if execution.harness_evaluated_attempts else "agent_managed",
            "max_evaluator_calls_per_attempt": 1 if execution.harness_evaluated_attempts else max_program_evals_per_attempt,
            "scientific_protocol": "The frozen task/evaluator still controls data, seeds, training steps, and acceptance. Scheduler settings do not relax these constraints.",
        },
        "agent": {
            "name": selected_agent or cfg.agent.name,
            "model": selected_model or cfg.agent.model, "reasoning_effort": cfg.agent.reasoning_effort,
            "max_idea_generation_attempts": execution.max_idea_generation_attempts,
            "coding_transient_retry_enabled": execution.harness_evaluated_attempts,
            "coding_transient_retry_scope": "host-harness coding failures before scoring; agent-managed evaluations are not automatically retried",
            "coding_transient_max_attempts": execution.agent_transient_max_attempts,
            "coding_transient_backoff_seconds": execution.agent_transient_backoff_seconds,
        },
        "timeouts_seconds": {"idea": execution.idea_timeout_seconds, "coding": execution.run_timeout_seconds, "host_evaluation": execution.harness_evaluation_timeout_seconds},
        "observed_local_history": {"attempt_records": len(attempts), "evaluated_attempts": len(evaluated), "provider_retry_records": sum(row.get("retry_scheduled") is True for row in attempts)},
        "stage": {"name": stage, "policy": "advisory fixed cycle fractions: exploration <=40%, refinement <=80%, confirmation >80%; not an adaptive experiment manager"},
    }
    if adaptive_context is not None:
        budget["stage"] = adaptive_context
    if portfolio_context is not None:
        from src.research.whole_idea_policy import validate_portfolio_context

        portfolio = validate_portfolio_context(portfolio_context)
        budget["portfolio"] = portfolio
        if "campaign_plan" in portfolio:
            budget["campaign_plan"] = portfolio["campaign_plan"]
        budget["stage"] = {
            "name": portfolio["stage"], "action": portfolio.get("action"),
            "policy": "host_whole_idea_portfolio",
            "root_idea_id": portfolio["root_idea_id"], "lease_id": portfolio["lease_id"],
            "trial_index": portfolio["trial_index"],
        }
    state_store.write_json(results_dir / BUDGET_FILE, budget)
    snapshot = results_dir / "campaign_budgets" / f"cycle_{cycle:04d}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    state_store.write_json(snapshot, budget)
    return budget


def render_campaign_budget(results_dir: Path, cycle_index: int | None = None) -> str:
    """One renderer for every prompt; direct standalone wrappers may lack a campaign."""
    methodology = ""
    from src.research.whole_idea_policy import render_whole_idea_guidance

    task_dir = results_dir / "task"
    # idea.sh supplies a cycle index and gets idea rules from render_context.
    # run.sh omits it, so the coding prompt gets exactly one rules block here.
    if cycle_index is None:
        from src.research.methodology import load_methodology, render_methodology
        from src.research.research_agenda import load_agenda

        marker = load_methodology(task_dir)
        if marker is not None:
            methodology = render_methodology(
                task_dir, stage="coding", agenda=load_agenda(task_dir), marker=marker,
            )
    path = results_dir / BUDGET_FILE
    if not path.is_file():
        whole_idea = render_whole_idea_guidance(task_dir, stage="coding" if cycle_index is None else "idea",
                                               include_policy=cycle_index is None)
        return "No resolved campaign budget was supplied. Follow the explicit wrapper evaluation limit and the task's scientific constraints; do not infer total cycles." + methodology + whole_idea
    budget = json.loads(path.read_text(encoding="utf-8"))
    if cycle_index is not None and budget["cycle"]["local_index"] != cycle_index:
        raise ValueError("effective campaign budget does not match this idea cycle")
    if budget.get("portfolio", {}).get("planned_card") is not None:
        # Saved evidence retains every complete bundle. The prompt displays the
        # complete root plan once, then references it beside historical settings.
        budget = deepcopy(budget)
        for row in budget["portfolio"].get("idea_history", []):
            bundle = row.get("outcome", {}).get("idea_bundle")
            if isinstance(bundle, dict):
                for key in ("planned_card", "complete_implementation", "implementation_plan"):
                    bundle.pop(key, None)
                bundle["complete_plan_ref"] = "portfolio.planned_card"
    guidance = {
        "exploration": "Compare a distinct plausible mechanism with the selected parent. State a falsifiable prediction and preserve controls.",
        "refinement": "Use recorded development evidence to refine a promising mechanism or resolve a concrete failure. Do not infer scientific failure from provider errors.",
        "confirmation": "Prefer a matched control or narrow robustness check for the strongest supported idea. Use only permitted development data; request no hidden test data and claim no independent confirmation unless the fixed protocol supplies it.",
    }
    adaptive_guidance = {
        "tune_baseline_training": "Keep the model architecture fixed. Test a single allowed training setting or convergence diagnosis; do not change frozen protocol, data, seeds, or the evaluator. If training settings are protocol-fixed, diagnose baseline behavior instead of silently tuning them.",
        "test_distinct_mechanism": "Develop a distinct, falsifiable mechanism for the selected branch. State its difference from the nearest baseline, the matched control, predicted observations and the condition that would refute it. Use development evidence only.",
        "refine_supported_mechanism": "Refine a supported mechanism or resolve the specific stagnation diagnosis. Change one component, retain matched controls and report negative findings. A comparison need not beat the incumbent to provide valid evidence.",
        "replicate_frozen_candidate": "The host repeats the exact frozen candidate without coding-agent edits. This measures repeatability under the same development protocol, not independent confirmation.",
    }
    stage = budget["stage"]
    stage_guidance = adaptive_guidance.get(stage.get("action"), guidance.get(stage.get("name"), "Follow the host stage lease."))
    portfolio = budget.get("portfolio")
    whole_idea = render_whole_idea_guidance(
        task_dir, stage="coding" if cycle_index is None else "idea", portfolio_context=portfolio,
        include_policy=cycle_index is None,
    )
    if portfolio is not None:
        stage_guidance = "Follow the verified whole-idea host lease and its stage guidance below."
        if portfolio.get("planned_card") is not None:
            stage_guidance += (
                " The portfolio.planned_card above is the complete approved root plan. "
                "Implement its complete_implementation, hypothesis, falsifier and decision_rule. "
                "Do not invent a replacement idea. Screening uses its approved whole_idea parameters. "
                "Tuning starts from this root's own best source and changes only permitted candidate defaults. "
                "The prequalified baseline is bootstrap evidence, not an independently tuned control. "
                "A planned control root follows the same screening and promotion rules as every other root.")
    return (
        "The host resolved this scheduling budget from the current configuration and state. "
        "It overrides stale campaign round counts or scheduler examples in the task brief. "
        "It does not override hard data, supervision, training, compute, interface, or acceptance constraints. "
        "One scheduled cycle is not proof of one completed evaluation.\n\n```json\n"
        + json.dumps(budget, ensure_ascii=False, indent=2)
        + "\n```\n\nStage action: " + stage_guidance + methodology + whole_idea
    )
