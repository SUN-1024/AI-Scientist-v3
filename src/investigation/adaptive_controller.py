"""Deterministic, evidence-driven investigation stages and resumable work leases.

Only development evaluation receipts drive this controller. Final confirmation and
ablations belong to the outer pipeline, which consumes validation_handoff.json.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.orchestrator import state as state_store
from .native_harness import evaluation_inputs_fingerprint
from .search_tree import _fingerprint

STATE_FILE = "adaptive_investigation.json"
HANDOFF_FILE = "validation_handoff.json"


@dataclass
class AdaptiveInvestigationConfig:
    enabled: bool = False
    policy_version: int = 1
    min_tuning_evals: int = 2
    min_creative_evals: int = 3
    min_refinement_evals: int = 2
    stagnation_evals: int = 2
    replication_evals: int = 2
    max_valid_evals: int = 20
    max_transport_failures: int = 6
    min_improvement: float = 0.0


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def task_protocol_identity(task_dir: Path) -> dict[str, str]:
    """Bind policy to canonical task text, contract and declared evaluator inputs."""
    files = {}
    for name in ("task.md", "task_contract.json", "evaluate.py", "study_plan.json"):
        path = task_dir / name
        if path.is_symlink():
            raise ValueError(f"Adaptive task identity cannot follow a symlink: {name}")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    inputs = evaluation_inputs_fingerprint(task_dir / "evaluate.py")
    return {"task_hash": _digest(files), "protocol_hash": _digest({"files": files, "inputs": inputs})}


class AdaptiveController:
    def __init__(self, results_dir: Path, task_dir: Path, policy: AdaptiveInvestigationConfig):
        self.results_dir, self.task_dir, self.policy = results_dir, task_dir, policy
        self.path = results_dir / STATE_FILE
        identity = {"policy_hash": _digest(asdict(policy)), **task_protocol_identity(task_dir)}
        self.data: dict[str, Any] = {
            "schema_version": 1, **identity, "policy": asdict(policy),
            "status": "running", "main_stage": "baseline_qualification", "substage": "qualification",
            "transition_reason": "awaiting_validated_baseline", "pending": None,
            "completed_outcome_ids": [], "outcomes": {}, "completed_cycles": [],
            "valid_evaluations": 0, "transport_failures": 0,
            "stage_counts": {}, "stagnation": 0, "best": None,
            "replications": {}, "aggregates": {}, "transition_journal": [],
        }
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink() or self.path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Invalid adaptive investigation state")
            loaded = json.loads(self.path.read_text())
            if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
                raise ValueError("Unsupported adaptive investigation state")
            if any(loaded.get(key) != value for key, value in identity.items()):
                raise ValueError("Adaptive policy/task/protocol changed; start a new campaign")
            self.data = loaded
        self._save()

    def _save(self) -> None:
        state_store.write_private_text(self.path, json.dumps(self.data, sort_keys=True, indent=2) + "\n")

    def _transition(self, main: str, sub: str, reason: str, *, status: str = "running") -> None:
        self.data["transition_journal"].append({
            "sequence": len(self.data["transition_journal"]),
            "from": {"main_stage": self.data["main_stage"], "substage": self.data["substage"]},
            "to": {"main_stage": main, "substage": sub}, "reason": reason,
            "valid_evaluations": self.data["valid_evaluations"],
        })
        self.data.update(main_stage=main, substage=sub, transition_reason=reason, status=status, stagnation=0)

    def qualify_baseline(self, manifest: dict[str, Any], program_dir: Path) -> None:
        """Caller has already validated the bootstrap through the existing host path."""
        if self.data["main_stage"] != "baseline_qualification":
            return
        score = manifest.get("combined_score")
        if manifest.get("correct") is not True or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Adaptive investigation requires a host-qualified baseline")
        self.data["baseline"] = {"score": float(score), "program_dir": str(program_dir.resolve()), "source_hash": _fingerprint(program_dir)}
        self.data["best"] = dict(self.data["baseline"])
        self._transition("baseline_tuning", "optimization", "host_baseline_qualified")
        self._save()

    def reconcile(self, outer_state: dict[str, Any]) -> int:
        """Pending work wins over legacy current_cycle, which could mean idea drafted."""
        pending = self.data.get("pending")
        if pending:
            return int(pending["cycle"])
        return max([0, *self.data["completed_cycles"], int(outer_state.get("current_cycle", 0))]) + 1

    def plan_cycle(self, cycle: int) -> dict[str, Any] | None:
        pending = self.data.get("pending")
        if pending:
            if pending["cycle"] != cycle:
                raise ValueError("Adaptive pending cycle must complete before another is scheduled")
            return dict(pending)
        if self.data["status"] != "running":
            return None
        # Successful transitions are applied in record_outcome BEFORE budget checks.
        if self.data["valid_evaluations"] >= self.policy.max_valid_evals:
            self.stop("scientific_evaluation_budget_exhausted")
            return None
        if self.data["transport_failures"] >= self.policy.max_transport_failures:
            self.stop("transport_failure_budget_exhausted")
            return None
        main, sub = self.data["main_stage"], self.data["substage"]
        if main == "baseline_qualification":
            raise ValueError("Qualify the host baseline before scheduling adaptive work")
        if sub == "replication":
            action, parent = "replicate_frozen_candidate", "frozen"
        elif main == "baseline_tuning":
            action, parent = "tune_baseline_training", "incumbent"
        elif sub == "exploration":
            action, parent = "test_distinct_mechanism", "explore"
        else:
            action, parent = "refine_supported_mechanism", "incumbent"
        lease = {
            "cycle": cycle, "lease_id": f"cycle_{cycle:04d}", "main_stage": main,
            "substage": sub, "action": action, "parent_policy": parent,
            "objective": "replication" if sub == "replication" else "comparison",
            "phase": "planned", "reason": self.data["transition_reason"],
        }
        if parent == "frozen":
            lease["frozen_candidate"] = dict(self.data["stage_winner"])
        self.data["pending"] = lease
        self._save()
        return dict(lease)

    def update_pending(self, **fields: Any) -> None:
        if not self.data.get("pending"):
            raise ValueError("No pending adaptive work")
        self.data["pending"].update(fields)
        self._save()

    def record_transport_failure(self, failure_id: str) -> None:
        ids = self.data.setdefault("transport_failure_ids", [])
        if failure_id not in ids:
            ids.append(failure_id)
            self.data["transport_failures"] += 1
            self._save()

    def _begin_replication(self, reason: str) -> None:
        self.data["stage_winner"] = dict(self.data["best"])
        key = self.data["main_stage"]
        self.data["replications"][key] = []
        self._transition(key, "replication", reason)

    def record_outcome(self, outcome_id: str, outcome: dict[str, Any] | None, *, failure_kind: str | None = None) -> bool:
        """Record one trusted observation; failed transport never counts as science."""
        if outcome_id in self.data["completed_outcome_ids"]:
            previous = self.data["outcomes"][outcome_id]
            if previous["outcome"] != outcome or previous["failure_kind"] != failure_kind:
                raise ValueError("Completed adaptive outcome cannot change")
            return False
        lease = self.data.get("pending")
        if not lease:
            raise ValueError("Adaptive outcome has no pending lease")
        if lease.get("phase") == "outcome_recorded":
            raise ValueError("A pending adaptive lease already has an outcome")
        if outcome is not None:
            score = outcome.get("score")
            if not outcome.get("receipt_id") or type(outcome.get("correct")) is not bool or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Adaptive scientific outcome needs a validated finite receipt")
            receipt_ids = [row["outcome"]["receipt_id"] for row in self.data["outcomes"].values() if row["outcome"]]
            if outcome["receipt_id"] in receipt_ids:
                raise ValueError("A scientific receipt cannot be counted twice")
            if lease["substage"] == "replication" and outcome.get("source_hash") != lease["frozen_candidate"]["source_hash"]:
                raise ValueError("Replication changed the frozen candidate")
            self.data["valid_evaluations"] += 1
            key = f"{lease['main_stage']}/{lease['substage']}"
            self.data["stage_counts"][key] = self.data["stage_counts"].get(key, 0) + 1
            improved = outcome["correct"] and (self.data["best"] is None or score > self.data["best"]["score"] + self.policy.min_improvement)
            better = outcome["correct"] and (self.data["best"] is None or score > self.data["best"]["score"])
            if better and lease["substage"] != "replication":
                self.data["best"] = {"score": score, "program_dir": outcome["program_dir"], "source_hash": outcome["source_hash"], "receipt_id": outcome["receipt_id"]}
            self.data["stagnation"] = 0 if improved else self.data["stagnation"] + 1
            count = self.data["stage_counts"][key]
            if lease["substage"] == "replication":
                repeats = self.data["replications"][lease["main_stage"]]
                repeats.append({"score": score, "correct": outcome["correct"], "receipt_id": outcome["receipt_id"]})
                if len(repeats) >= self.policy.replication_evals:
                    scores = [row["score"] for row in repeats]
                    self.data["aggregates"][lease["main_stage"]] = {"n": len(scores), "mean": mean(scores), "std": pstdev(scores), "all_correct": all(row["correct"] for row in repeats), "evidence_scope": "same_frozen_development_protocol_repeatability", "independent_confirmation": False, "source_hash": lease["frozen_candidate"]["source_hash"]}
                    if not all(row["correct"] for row in repeats):
                        self.stop("frozen_candidate_replication_failed")
                    elif lease["main_stage"] == "baseline_tuning":
                        self._transition("creative_research", "exploration", "baseline_tuning_replicated")
                    else:
                        self._transition("ready_for_validation", "handoff", "development_research_replicated", status="ready_for_validation")
            elif lease["main_stage"] == "baseline_tuning" and count >= self.policy.min_tuning_evals:
                self._begin_replication("minimum_tuning_evidence_collected")
            elif lease["substage"] == "exploration" and count >= self.policy.min_creative_evals and (improved or self.data["stagnation"] >= self.policy.stagnation_evals):
                self._transition("creative_research", "refinement", "supported_improvement" if improved else "exploration_stagnation_requires_narrow_diagnosis")
            elif lease["substage"] == "refinement" and count >= self.policy.min_refinement_evals:
                remaining = self.policy.max_valid_evals - self.data["valid_evaluations"]
                if self.data["stagnation"] >= self.policy.stagnation_evals:
                    self._begin_replication("refinement_stagnation_after_minimum_evidence")
                elif remaining <= self.policy.replication_evals:
                    self._begin_replication("reserve_remaining_evaluations_for_frozen_replication")
        self.data["completed_outcome_ids"].append(outcome_id)
        self.data["outcomes"][outcome_id] = {"cycle": lease["cycle"], "main_stage": lease["main_stage"], "substage": lease["substage"], "outcome": outcome, "failure_kind": failure_kind}
        self.data["pending"]["phase"] = "outcome_recorded"
        self.data["pending"]["outcome_id"] = outcome_id
        self._save()
        return True

    def finish_cycle(self, cycle: int) -> None:
        pending = self.data.get("pending")
        if not pending or pending["cycle"] != cycle or pending.get("phase") != "outcome_recorded":
            raise ValueError("Cannot finish adaptive work before recording its outcome")
        if cycle not in self.data["completed_cycles"]:
            self.data["completed_cycles"].append(cycle)
        self.data["pending"] = None
        self._save()
        if self.data["status"] == "ready_for_validation":
            self._write_handoff()

    def resume_schedule(self, max_cycles: int) -> bool:
        """Reopen only a deliberately enlarged outer schedule, never a hard stop."""
        previous_limit = self.data.get("exhausted_cycle_limit", max([0, *self.data["completed_cycles"]]))
        if not (self.data["status"] == "incomplete"
                and self.data["transition_reason"] == "scheduled_cycle_budget_exhausted"
                and max_cycles > previous_limit
                and self.data["valid_evaluations"] < self.policy.max_valid_evals
                and self.data["transport_failures"] < self.policy.max_transport_failures):
            return False
        stagnation = self.data["stagnation"]
        self._transition(self.data["main_stage"], self.data["substage"], "scheduled_cycle_budget_extended")
        self.data["stagnation"] = stagnation
        self.data["resumed_cycle_limit"] = max_cycles
        self._save()
        return True

    def stop(self, reason: str, *, cycle_limit: int | None = None) -> None:
        stagnation = self.data["stagnation"]
        self._transition(self.data["main_stage"], self.data["substage"], reason, status="incomplete")
        self.data["stagnation"] = stagnation
        if cycle_limit is not None:
            self.data["exhausted_cycle_limit"] = cycle_limit
        self._save()

    def _write_handoff(self) -> None:
        state_store.write_private_text(self.results_dir / HANDOFF_FILE, json.dumps({
            "schema_version": 1, "status": "ready_for_validation", "candidate": self.data["stage_winner"],
            "policy_hash": self.data["policy_hash"], "task_hash": self.data["task_hash"], "protocol_hash": self.data["protocol_hash"],
            "development_aggregates": self.data["aggregates"],
            "required_phases": ["ablations", "final_evaluation", "posthoc", "paper"],
            "independent_confirmation_complete": False, "submission_ready": False,
        }, sort_keys=True, indent=2) + "\n")

    def context(self, lease: dict[str, Any]) -> dict[str, Any]:
        return {**lease, "policy": "host_evidence_driven", "status": self.data["status"],
                "valid_evaluations": self.data["valid_evaluations"],
                "remaining_valid_evaluations": max(0, self.policy.max_valid_evals - self.data["valid_evaluations"]),
                "transport_failures": self.data["transport_failures"],
                "independent_confirmation": False}
