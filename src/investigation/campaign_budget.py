"""Resolved, cycle-specific campaign context shared by idea and coding prompts."""
from __future__ import annotations

import json
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
    state_store.write_json(results_dir / BUDGET_FILE, budget)
    snapshot = results_dir / "campaign_budgets" / f"cycle_{cycle:04d}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    state_store.write_json(snapshot, budget)
    return budget


def render_campaign_budget(results_dir: Path, cycle_index: int | None = None) -> str:
    """One renderer for every prompt; direct standalone wrappers may lack a campaign."""
    path = results_dir / BUDGET_FILE
    if not path.is_file():
        return "No resolved campaign budget was supplied. Follow the explicit wrapper evaluation limit and the task's scientific constraints; do not infer total cycles."
    budget = json.loads(path.read_text(encoding="utf-8"))
    if cycle_index is not None and budget["cycle"]["local_index"] != cycle_index:
        raise ValueError("effective campaign budget does not match this idea cycle")
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
    return (
        "The host resolved this scheduling budget from the current configuration and state. "
        "It overrides stale campaign round counts or scheduler examples in the task brief. "
        "It does not override hard data, supervision, training, compute, interface, or acceptance constraints. "
        "One scheduled cycle is not proof of one completed evaluation.\n\n```json\n"
        + json.dumps(budget, ensure_ascii=False, indent=2)
        + "\n```\n\nStage action: " + stage_guidance
    )
