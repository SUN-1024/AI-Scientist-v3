"""Host-owned, durable scheduling of whole ideas and their joint tuning trials.

This controller does not execute programs or trust worker-written metrics. Its
caller supplies already host-validated receipts. Screening uses a cohort barrier;
tuning follows each idea's own best result; repeats use a frozen source snapshot.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
import stat
from typing import Any, Iterator

from src.orchestrator import state as state_store
from src.research.upstream.v2_journal import MetricValue as V2MetricValue
from .adaptive_controller import _digest, task_protocol_identity
from .search_tree import _fingerprint

STATE_FILE = "portfolio_investigation.json"
HANDOFF_FILE = "validation_handoff.json"
LOCK_FILE = ".portfolio_investigation.lock"
INNOVATION_AXES = ("problem", "efficiency", "data", "objective", "optimization", "structure")
_MAX_STATE_BYTES = 32 * 1024 * 1024
_PLAN_HOST_FIELDS = {"root_idea_id", "idea_key", "card_sha256"}


def _planned_root_fields(root: dict[str, Any]) -> dict[str, Any]:
    return {"card_sha256": root["card_sha256"], "idea_key": root["idea_key"],
            "role": root["role"],
            "planned_card": {key: deepcopy(value) for key, value in root.items() if key not in _PLAN_HOST_FIELDS}}


def _validate_live_plan(data: dict[str, Any], task_dir: Path) -> None:
    # A legacy campaign keeps its original scheduling contract on recovery.
    if "plan_sha256" not in data:
        return
    from src.research.campaign_plan import load_campaign_plan

    plan = load_campaign_plan(task_dir, portfolio_policy=data["policy"], required=True)
    if plan is None or plan != data.get("campaign_plan") or plan["plan_sha256"] != data["plan_sha256"]:
        raise ValueError("Portfolio campaign plan changed; start a new campaign")


@dataclass
class PortfolioConfig:
    enabled: bool = False
    policy_version: int = 1
    initial_ideas: int = 4
    promote_ideas: int = 2
    tuning_trials_per_idea: int = 4
    replication_evals: int = 2
    max_valid_evals: int = 14
    max_transport_failures: int = 6
    min_improvement: float = 0.0


def _validate_policy(policy: PortfolioConfig) -> None:
    if type(policy.enabled) is not bool or type(policy.policy_version) is not int or policy.policy_version != 1:
        raise ValueError("Unsupported portfolio policy")
    for key in ("initial_ideas", "promote_ideas", "tuning_trials_per_idea", "replication_evals",
                "max_valid_evals", "max_transport_failures"):
        if type(getattr(policy, key)) is not int or getattr(policy, key) < 1:
            raise ValueError(f"Portfolio {key} must be a positive integer")
    if policy.promote_ideas > policy.initial_ideas:
        raise ValueError("Cannot promote more ideas than the screening cohort")
    improvement = policy.min_improvement
    if isinstance(improvement, bool) or not isinstance(improvement, (int, float)) or not math.isfinite(improvement) or improvement < 0:
        raise ValueError("Portfolio min_improvement must be a finite nonnegative number")
    required = policy.initial_ideas + policy.promote_ideas * policy.tuning_trials_per_idea + policy.replication_evals
    if policy.max_valid_evals < required:
        raise ValueError(f"Portfolio scientific budget must reserve screening, tuning and repeats ({required})")


def _check_file(fd: int) -> None:
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
            or info.st_mode & 0o022):
        raise ValueError("Unsafe portfolio state or lock file")


@contextmanager
def _locked_root(results_dir: Path, *, create: bool) -> Iterator[int]:
    """Pin the path without following directory symlinks; lock one stable inode."""
    try:
        root_fd = state_store._create_memory_directory_no_follow(results_dir) if create else _open_root(results_dir)
    except OSError as exc:
        raise ValueError("Unsafe or unavailable portfolio directory") from exc
    lock_fd = None
    try:
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | (os.O_CREAT if create else 0)
        lock_fd = os.open(LOCK_FILE, flags, 0o600, dir_fd=root_fd)
        _check_file(lock_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX if create else fcntl.LOCK_SH)
        yield root_fd
    except OSError as exc:
        raise ValueError("Unsafe or unavailable portfolio state") from exc
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        os.close(root_fd)


def _open_root(path: Path) -> int:
    absolute = path.absolute()
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_state(root_fd: int) -> dict[str, Any] | None:
    try:
        fd = os.open(STATE_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    except FileNotFoundError:
        return None
    try:
        _check_file(fd)
        if os.fstat(fd).st_size > _MAX_STATE_BYTES:
            raise ValueError("Portfolio state is too large")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
        _validate_state(value)
        return value
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError("Invalid portfolio state") from exc
    finally:
        os.close(fd)


def _validate_state(data: dict[str, Any]) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported portfolio state")
    payload = {key: value for key, value in data.items() if key != "state_hash"}
    if data.get("state_hash") != _digest(payload):
        raise ValueError("Portfolio state integrity mismatch")
    policy = PortfolioConfig(**data["policy"])
    _validate_policy(policy)
    if data["policy_hash"] != _digest(asdict(policy)):
        raise ValueError("Portfolio policy identity mismatch")
    planned_roots = {}
    if "plan_sha256" in data:
        from src.research.campaign_plan import validate_campaign_plan

        plan = validate_campaign_plan(data.get("campaign_plan"), portfolio_policy=data["policy"])
        if plan["plan_sha256"] != data["plan_sha256"]:
            raise ValueError("Portfolio campaign plan identity mismatch")
        planned_roots = {root["root_idea_id"]: root for root in plan["roots"]}
        if data["baseline"] is not None and set(data["ideas"]) != set(planned_roots):
            raise ValueError("Portfolio roots differ from the approved campaign plan")
        for root_id, idea in data["ideas"].items():
            planned = planned_roots.get(root_id)
            if (planned is None or idea["innovation_axis"] != planned["whole_idea"]["innovation_axis"]
                    or any(idea.get(key) != value for key, value in _planned_root_fields(planned).items())):
                raise ValueError("Portfolio root card differs from the approved campaign plan")
    elif "campaign_plan" in data:
        raise ValueError("Portfolio campaign plan lacks its bound identity")
    if data["status"] not in {"running", "incomplete", "ready_for_validation"}:
        raise ValueError("Invalid portfolio status")
    if data["transport_failures"] != len(data["failures"]):
        raise ValueError("Portfolio failure ledger is inconsistent")
    leases, outcomes, pending = data["leases"], data["outcomes"], data["pending"]
    if (not isinstance(leases, dict) or not isinstance(outcomes, dict) or not isinstance(pending, list)
            or len(pending) != len(set(pending)) or not set(outcomes).issubset(leases)
            or not set(pending).issubset(leases) or set(pending) & set(outcomes)
            or data["valid_evaluations"] != len(outcomes)
            or len(outcomes) + len(pending) > policy.max_valid_evals):
        raise ValueError("Portfolio receipt or reservation ledger is inconsistent")
    if data["receipt_ids"] != {row["receipt_id"]: key for key, row in outcomes.items()} or len(data["receipt_ids"]) != len(outcomes):
        raise ValueError("Portfolio receipt ledger contains duplicates")
    recoveries = data.get("transport_recoveries", [])
    if not isinstance(recoveries, list):
        raise ValueError("Invalid portfolio transport recovery journal")
    for sequence, recovery in enumerate(recoveries):
        count = recovery.get("transport_failures")
        if (recovery.get("sequence") != sequence or type(count) is not int
                or not 0 < count <= data["transport_failures"]
                or recovery.get("failure_sequences") != list(range(count))
                or recovery.get("receipt_ids") != {
                    row["lease_id"]: outcomes.get(row["lease_id"], {}).get("receipt_id")
                    for row in data["failures"][:count]}
                or any(row["lease_id"] not in outcomes for row in data["failures"][:count])
                or type(recovery.get("valid_evaluations")) is not int
                or not 0 < recovery["valid_evaluations"] <= data["valid_evaluations"]):
            raise ValueError("Portfolio recovery journal differs from retained failures or receipts")
    if len({leases[key]["root_idea_id"] for key in pending}) != len(pending):
        raise ValueError("Portfolio has multiple active leases for one idea")
    if len({leases[key]["worker_id"] for key in pending}) != len(pending):
        raise ValueError("Portfolio has multiple active leases for one worker")
    for key, lease in leases.items():
        if lease.get("lease_id") != key or any(lease.get(field) != data[field] for field in ("policy_hash", "task_hash", "protocol_hash")):
            raise ValueError("Portfolio lease identity mismatch")
        if planned_roots:
            planned = planned_roots.get(lease.get("root_idea_id"))
            if (lease.get("plan_sha256") != data["plan_sha256"]
                    or lease.get("card_sha256") != (planned["card_sha256"] if planned else None)
                    or lease.get("role") != (planned["role"] if planned else "bootstrap")
                    or (planned is not None and lease.get("innovation_axis") != planned["whole_idea"]["innovation_axis"])):
                raise ValueError("Portfolio lease campaign plan or card identity mismatch")
        if lease.get("root_idea_id") != "baseline" and lease.get("root_idea_id") not in data["ideas"]:
            raise ValueError("Portfolio lease points to an unknown idea")
        if lease.get("stage") not in {"screening", "tuning", "replication"}:
            raise ValueError("Invalid portfolio lease stage")
        if (lease.get("parent_source_hash") != lease["parent"].get("source_hash")
                or lease.get("parent_program_dir") != lease["parent"].get("program_dir")
                or lease.get("parent_receipt_id") != lease["parent"].get("receipt_id")):
            raise ValueError("Portfolio parent identity mismatch")
    for key, outcome in outcomes.items():
        PortfolioController._validate_score(outcome.get("score"), outcome.get("correct"))
        if planned_roots:
            from .portfolio_campaign import bind_portfolio_idea

            lease = leases[key]
            if any(outcome.get(field) != lease.get(field) for field in ("plan_sha256", "card_sha256", "role")):
                raise ValueError("Portfolio outcome campaign plan identity mismatch")
            if bind_portfolio_idea(outcome.get("idea_bundle"), _context(data, lease)) != outcome.get("idea_bundle"):
                raise ValueError("Portfolio outcome differs from its bound complete plan card")
        if leases[key]["stage"] == "replication" and outcome.get("source_hash") != leases[key]["parent_source_hash"]:
            raise ValueError("Portfolio repeat ledger changed its frozen source")
    for root_id, idea in data["ideas"].items():
        expected_history = sorted(key for key in outcomes if leases[key]["root_idea_id"] == root_id and leases[key]["stage"] != "replication")
        if idea["history"] != expected_history:
            raise ValueError("Portfolio root history differs from its receipt ledger")
    if data["status"] == "ready_for_validation":
        aggregate = data["aggregates"].get("frozen_candidate", {})
        repeats = [row for key, row in outcomes.items() if leases[key]["stage"] == "replication"]
        if (pending or not data.get("frozen_candidate") or aggregate.get("n") != policy.replication_evals
                or aggregate.get("all_correct") is not True or len(repeats) != policy.replication_evals
                or any(row["correct"] is not True or row["source_hash"] != data["frozen_candidate"]["source_hash"] for row in repeats)):
            raise ValueError("Portfolio validation handoff lacks completed correct frozen repeats")


def read_portfolio_state(results_dir: Path, task_dir: Path | None = None,
                         policy: PortfolioConfig | None = None) -> dict[str, Any]:
    """Read and validate host state without creating or changing any file."""
    with _locked_root(Path(results_dir), create=False) as root_fd:
        data = _read_state(root_fd)
        if data is None:
            raise ValueError("Portfolio state is missing")
        expected = task_protocol_identity(Path(task_dir)) if task_dir is not None else {}
        if policy is not None:
            expected["policy_hash"] = _digest(asdict(policy))
        if any(data.get(key) != value for key, value in expected.items()):
            raise ValueError("Portfolio policy/task/protocol changed; start a new campaign")
        if task_dir is not None:
            _validate_live_plan(data, Path(task_dir))
        return deepcopy(data)


def _context(data: dict[str, Any], lease: dict[str, Any]) -> dict[str, Any]:
    root = data["ideas"].get(lease["root_idea_id"], {})
    history = [{"lease_id": key, "stage": data["leases"][key]["stage"],
                "trial_index": data["leases"][key]["trial_index"], "outcome": data["outcomes"][key]}
               for key in sorted(root.get("history", []))]
    bundle = next((row["outcome"].get("idea_bundle") for row in history if row["stage"] == "screening"), None)
    source_card = (root["planned_card"]["whole_idea"] if "planned_card" in root else
                   bundle.get("whole_idea", bundle) if isinstance(bundle, dict) else None)
    archive = []
    for root_id, idea in sorted(data["ideas"].items()):
        if root_id == lease["root_idea_id"] or (not idea["history"] and "planned_card" not in idea):
            continue
        screen = next((data["outcomes"][key] for key in idea["history"] if data["leases"][key]["stage"] == "screening"), None)
        card = idea.get("planned_card") or (screen.get("idea_bundle") if screen else None)
        archive.append({"root_idea_id": root_id, "innovation_axis": idea["innovation_axis"],
                        "source_card": card.get("whole_idea", card) if isinstance(card, dict) else None,
                        "status": "promoted" if idea["promoted"] else "screened" if idea["history"] else "planned",
                        "best_score": idea["best"]["score"] if idea["best"] else None})
    planned_context = ({"planned_card": root["planned_card"]} if "planned_card" in root else {})
    if "campaign_plan" in data:
        planned_context["campaign_plan"] = {key: data["campaign_plan"][key] for key in (
            "plan_sha256", "host_context_sha256", "agenda_sha256", "allocation")}
    return deepcopy({**lease, **planned_context, "policy": "host_whole_idea_portfolio", "whole_idea": True,
                     "single_candidate": True, "idea_history": history, "source_card": source_card, "idea_archive": archive,
                     "own_best": root.get("best"), "policy_config": data["policy"],
                     "status": data["status"], "valid_evaluations": data["valid_evaluations"],
                     "remaining_valid_evaluations": data["policy"]["max_valid_evals"] - data["valid_evaluations"],
                     "transport_failures": data["transport_failures"], "independent_confirmation": False})


def validate_external_lease(payload: dict[str, Any], task_dir: Path, results_dir: Path,
                            seed_program_dir: Path) -> dict[str, Any]:
    """Return trusted live context for a child; workers cannot grant themselves leases."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid external portfolio lease")
    lease = payload.get("lease", payload)
    if not isinstance(lease, dict):
        raise ValueError("Invalid external portfolio lease")
    root_results = payload.get("root_results_dir")
    if not isinstance(root_results, str) or not Path(root_results).is_absolute():
        raise ValueError("External portfolio lease needs an absolute root results directory")
    root_path, child_path = Path(root_results).absolute(), Path(results_dir).absolute()
    if child_path != Path(lease.get("result_dir", "")).absolute() or not child_path.is_relative_to(root_path / "portfolio_trials"):
        raise ValueError("External portfolio lease uses a different result directory")
    data = read_portfolio_state(root_path, task_dir)
    lease_id = lease.get("lease_id")
    if lease_id not in data["pending"] or data["leases"][lease_id] != lease:
        raise ValueError("External portfolio lease differs from its pending host record")
    if Path(seed_program_dir).absolute() != Path(lease["parent_program_dir"]).absolute():
        raise ValueError("External portfolio seed differs from its leased parent directory")
    if _fingerprint(Path(seed_program_dir)) != lease["parent_source_hash"]:
        raise ValueError("External portfolio seed changed from its leased parent")
    return _context(data, lease)


class PortfolioController:
    """The single logical writer; OS locks serialize competing parent invocations."""

    def __init__(self, results_dir: Path, task_dir: Path, policy: PortfolioConfig):
        _validate_policy(policy)
        self.results_dir, self.task_dir = Path(results_dir).absolute(), Path(task_dir).absolute()
        self.policy = deepcopy(policy)
        self.path = self.results_dir / STATE_FILE
        self.identity = {"policy_hash": _digest(asdict(policy)), **task_protocol_identity(self.task_dir)}
        self._initialized = False
        self.data: dict[str, Any] = {
            "schema_version": 1, **self.identity, "policy": asdict(policy),
            "status": "running", "stage": "baseline_qualification", "main_stage": "baseline_qualification",
            "transition_reason": "awaiting_validated_baseline", "baseline": None, "best": None,
            "frozen_candidate": None, "ideas": {}, "leases": {}, "pending": [], "outcomes": {},
            "receipt_ids": {}, "valid_evaluations": 0, "transport_failures": 0, "failures": [],
            "promoted_ideas": [], "aggregates": {}, "transition_journal": [], "selection_reason": None,
        }
        with self._transaction():
            pass

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with _locked_root(self.results_dir, create=True) as root_fd:
            actual = {"policy_hash": _digest(asdict(self.policy)), **task_protocol_identity(self.task_dir)}
            if any(self.identity.get(key) != value for key, value in actual.items()):
                raise ValueError("Portfolio policy/task/protocol changed; start a new campaign")
            loaded = _read_state(root_fd)
            if self._initialized and loaded is None and "plan_sha256" in self.identity:
                raise ValueError("Bound portfolio state is missing; cannot recover planned leases")
            if not self._initialized:
                if loaded is None:
                    from src.research.campaign_plan import load_campaign_plan

                    plan = load_campaign_plan(self.task_dir, portfolio_policy=asdict(self.policy))
                    if plan is not None:
                        if any((self.results_dir / name).exists() or (self.results_dir / name).is_symlink()
                               for name in ("portfolio_trials", HANDOFF_FILE)):
                            raise ValueError("Bound portfolio state is missing for existing planned work")
                        self.identity["plan_sha256"] = plan["plan_sha256"]
                        self.data.update(campaign_plan=plan, plan_sha256=plan["plan_sha256"])
                elif "plan_sha256" in loaded:
                    self.identity["plan_sha256"] = loaded["plan_sha256"]
                self._initialized = True
            if loaded is not None:
                if any(loaded.get(key) != value for key, value in self.identity.items()):
                    raise ValueError("Portfolio policy/task/protocol changed; start a new campaign")
                self.data = loaded
            _validate_live_plan(self.data, self.task_dir)
            previous = deepcopy(self.data)
            try:
                yield
                self.data["state_hash"] = _digest({key: value for key, value in self.data.items() if key != "state_hash"})
                _validate_state(self.data)
                state_store._atomic_replace_regular_file_at(root_fd, STATE_FILE,
                    (json.dumps(self.data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(), 0o600)
                os.fsync(root_fd)
                if self.data["status"] == "ready_for_validation":
                    self._write_handoff()
            except BaseException:
                self.data = previous
                raise

    @property
    def status(self) -> str:
        self.data = read_portfolio_state(self.results_dir, self.task_dir, self.policy)
        return self.data["status"]

    def pending_leases(self, worker_id: str | int | None = None) -> list[dict[str, Any]]:
        self.data = read_portfolio_state(self.results_dir, self.task_dir, self.policy)
        return [deepcopy(self.data["leases"][key]) for key in self.data["pending"]
                if worker_id is None or self.data["leases"][key]["worker_id"] == str(worker_id)]

    def pending_for_worker(self, worker_id: str | int) -> dict[str, Any] | None:
        return next(iter(self.pending_leases(worker_id)), None)

    def context(self, lease: dict[str, Any]) -> dict[str, Any]:
        self.data = read_portfolio_state(self.results_dir, self.task_dir, self.policy)
        if self.data["leases"].get(lease.get("lease_id")) != lease:
            raise ValueError("Portfolio context requires an unchanged host lease")
        return _context(self.data, lease)

    def _transition(self, stage: str, reason: str, *, status: str = "running") -> None:
        self.data["transition_journal"].append({"sequence": len(self.data["transition_journal"]),
            "from": self.data["stage"], "to": stage, "reason": reason,
            "valid_evaluations": self.data["valid_evaluations"]})
        self.data.update(stage=stage, main_stage=stage, transition_reason=reason, status=status)

    def qualify_baseline(self, manifest: dict[str, Any], program_dir: Path) -> None:
        with self._transaction():
            score = manifest.get("combined_score", manifest.get("score"))
            self._validate_score(score, manifest.get("correct"))
            assert isinstance(score, (int, float))
            if manifest.get("correct") is not True:
                raise ValueError("Portfolio requires a host-qualified correct baseline")
            baseline = {"score": float(score), "correct": True, "program_dir": str(Path(program_dir).absolute()),
                        "source_hash": _fingerprint(Path(program_dir)), "receipt_id": manifest.get("receipt_id"),
                        "root_idea_id": "baseline", "lease_id": "baseline"}
            baseline.update({key: manifest[key] for key in ("attempt_id", "idea_id") if key in manifest})
            if "campaign_plan" in self.data:
                baseline.update(evidence_role="bootstrap", tuned_comparator=False)
            if self.data["baseline"] is not None:
                if self.data["baseline"] != baseline:
                    raise ValueError("Qualified portfolio baseline cannot change")
                return
            self.data.update(baseline=baseline, best=deepcopy(baseline))
            if "campaign_plan" in self.data:
                self.data["ideas"] = {root["root_idea_id"]: {
                    "root_idea_id": root["root_idea_id"], "innovation_axis": root["whole_idea"]["innovation_axis"],
                    **_planned_root_fields(root), "history": [], "best": None, "promoted": False,
                } for root in self.data["campaign_plan"]["roots"]}
            else:
                self.data["ideas"] = {f"idea_{index + 1:04d}": {
                    "root_idea_id": f"idea_{index + 1:04d}", "innovation_axis": INNOVATION_AXES[index % len(INNOVATION_AXES)],
                    "history": [], "best": None, "promoted": False,
                } for index in range(self.policy.initial_ideas)}
            self._transition("screening", "host_baseline_qualified_whole_idea_cohort_open")

    @staticmethod
    def _validate_score(score: Any, correct: Any) -> None:
        if type(correct) is not bool or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Portfolio scientific outcome needs boolean correctness and a finite score")

    @staticmethod
    def _rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stable = sorted(candidates, key=lambda item: item["lease_id"])
        return sorted(stable, key=lambda item: V2MetricValue(item["score"], maximize=True, name="combined_score"), reverse=True)

    def _check_parent(self, parent: dict[str, Any]) -> None:
        fd = _open_root(Path(parent["program_dir"]))
        os.close(fd)
        if _fingerprint(Path(parent["program_dir"])) != parent["source_hash"]:
            raise ValueError("Portfolio leased parent sources changed")

    def reserve_next(self, worker_id: str | int, gpu_selector: str | None = None) -> dict[str, Any] | None:
        worker_id = str(worker_id)
        if not worker_id or (gpu_selector is not None and not isinstance(gpu_selector, str)):
            raise ValueError("Portfolio worker and GPU identifiers must be strings")
        with self._transaction():
            if self.data["status"] != "running":
                return None
            for key in self.data["pending"]:
                lease = self.data["leases"][key]
                if lease["worker_id"] == worker_id:
                    if gpu_selector != lease["gpu_selector"]:
                        raise ValueError("Pending portfolio lease cannot change its GPU assignment")
                    self._check_parent(lease["parent"])
                    return deepcopy(lease)
            if self.data["stage"] == "baseline_qualification":
                raise ValueError("Qualify the host baseline before reserving portfolio work")
            stage = self.data["stage"]
            busy = {self.data["leases"][key]["root_idea_id"] for key in self.data["pending"]}
            if stage == "screening":
                choices = [key for key, row in sorted(self.data["ideas"].items()) if not row["history"] and key not in busy]
                if not choices:
                    return None
                root_id, trial, parent = choices[0], 0, self.data["baseline"]
            elif stage == "tuning":
                counts = {key: sum(self.data["leases"][item]["stage"] == "tuning" for item in self.data["ideas"][key]["history"])
                          for key in self.data["promoted_ideas"]}
                choices = sorted((count, key) for key, count in counts.items()
                                 if count < self.policy.tuning_trials_per_idea and key not in busy)
                if not choices:
                    return None
                count, root_id = choices[0]
                trial, parent = count + 1, self.data["ideas"][root_id]["best"]
            elif stage == "replication":
                parent = self.data["frozen_candidate"]
                root_id = parent["root_idea_id"]
                if root_id in busy:
                    return None
                trial = 1 + sum(row["stage"] == "replication" and key in self.data["outcomes"] for key, row in self.data["leases"].items())
                if trial > self.policy.replication_evals:
                    return None
            else:
                return None
            used = self.data["valid_evaluations"] + len(self.data["pending"])
            # Do not let exploration or tuning consume the fixed-source repeat quota.
            limit = self.policy.max_valid_evals - (self.policy.replication_evals if stage != "replication" else 0)
            if used >= limit:
                self._transition(stage, "scientific_evaluation_budget_exhausted", status="incomplete")
                return None
            self._check_parent(parent)
            lease_id = f"{root_id}_{stage}_{trial:03d}"
            if lease_id in self.data["leases"]:
                raise ValueError("Portfolio scheduler attempted to reuse a completed lease")
            axis = self.data["ideas"].get(root_id, {}).get("innovation_axis", "optimization")
            action = {"screening": "propose_whole_idea", "tuning": "tune_whole_idea", "replication": "replicate_frozen_candidate"}[stage]
            lease = {"lease_id": lease_id, "cycle": len(self.data["leases"]) + 1,
                     "root_idea_id": root_id, "stage": stage, "main_stage": stage,
                     "substage": "replication" if stage == "replication" else "optimization" if stage == "tuning" else "exploration",
                     "action": action, "innovation_axis": axis, "trial_index": trial,
                     "parent": deepcopy(parent), "parent_program_dir": parent["program_dir"],
                     "parent_source_hash": parent["source_hash"], "parent_receipt_id": parent["receipt_id"],
                     "parent_policy": "frozen" if stage == "replication" else "own_best" if stage == "tuning" else "baseline",
                     "objective": "replication" if stage == "replication" else "comparison",
                     **self.identity, "result_dir": str(self.results_dir / "portfolio_trials" / lease_id),
                     "worker_id": worker_id, "gpu_selector": gpu_selector, "reason": self.data["transition_reason"]}
            if "plan_sha256" in self.data:
                root = self.data["ideas"].get(root_id, {})
                lease.update(card_sha256=root.get("card_sha256"), role=root.get("role", "bootstrap"))
            if stage == "replication":
                lease["frozen_candidate"] = deepcopy(parent)
            self.data["leases"][lease_id] = lease
            self.data["pending"].append(lease_id)
            return deepcopy(lease)

    def record_receipt(self, lease_id: str, trusted_outcome: dict[str, Any]) -> bool:
        with self._transaction():
            outcome = deepcopy(trusted_outcome)
            if not isinstance(outcome, dict):
                raise ValueError("Portfolio outcome must be a host-validated receipt")
            if "plan_sha256" in self.data:
                from .portfolio_campaign import bind_portfolio_idea

                lease = self.data["leases"].get(lease_id)
                if lease is None:
                    raise ValueError("Portfolio receipt has no pending lease")
                outcome["idea_bundle"] = bind_portfolio_idea(outcome.get("idea_bundle"), _context(self.data, lease))
                if any(field in outcome and outcome[field] != lease.get(field)
                       for field in ("plan_sha256", "card_sha256", "role")):
                    raise ValueError("Portfolio receipt campaign plan identity mismatch")
                outcome.update({field: lease.get(field) for field in ("plan_sha256", "card_sha256", "role")})
            if lease_id in self.data["outcomes"]:
                if self.data["outcomes"][lease_id] != outcome:
                    raise ValueError("Completed portfolio receipt cannot change")
                return False
            if lease_id not in self.data["pending"]:
                raise ValueError("Portfolio receipt has no pending lease")
            self._validate_score(outcome.get("score"), outcome.get("correct"))
            receipt_id, source_hash, program_dir = (outcome.get(key) for key in ("receipt_id", "source_hash", "program_dir"))
            if (not isinstance(receipt_id, str) or not receipt_id or not isinstance(source_hash, str)
                    or not isinstance(program_dir, str) or not Path(program_dir).is_absolute()):
                raise ValueError("Portfolio receipt needs an ID, source hash and absolute program directory")
            if receipt_id in self.data["receipt_ids"]:
                raise ValueError("A scientific receipt cannot be counted twice")
            lease = self.data["leases"][lease_id]
            if lease["stage"] == "replication" and source_hash != lease["parent_source_hash"]:
                raise ValueError("Replication changed the frozen candidate")
            if _fingerprint(Path(program_dir)) != source_hash:
                raise ValueError("Portfolio receipt source hash does not match its program")
            # Preserve full cards, settings, negative outcomes and host evidence.
            json.dumps(outcome, allow_nan=False)
            self.data["outcomes"][lease_id] = outcome
            self.data["receipt_ids"][receipt_id] = lease_id
            self.data["pending"].remove(lease_id)
            self.data["valid_evaluations"] += 1
            if lease["stage"] != "replication":
                root = self.data["ideas"][lease["root_idea_id"]]
                root["history"].append(lease_id)
                root["history"].sort()
                if outcome["correct"]:
                    candidate = {key: outcome[key] for key in ("score", "correct", "program_dir", "source_hash", "receipt_id")}
                    candidate.update({key: outcome[key] for key in ("attempt_id", "idea_id") if key in outcome})
                    candidate.update(root_idea_id=lease["root_idea_id"], lease_id=lease_id)
                    if "plan_sha256" in self.data:
                        candidate.update({field: lease.get(field) for field in ("plan_sha256", "card_sha256", "role")})
                    root["best"] = self._rank([candidate, *([root["best"]] if root["best"] else [])])[0]
                    self.data["best"] = self._rank([self.data["best"], candidate])[0]
            if self.data["status"] == "running":
                self._advance()
            return True

    def _advance(self) -> None:
        stage = self.data["stage"]
        if self.data["pending"]:
            return
        if stage == "screening" and all(row["history"] for row in self.data["ideas"].values()):
            ranked = self._rank([row["best"] for row in self.data["ideas"].values() if row["best"]])
            promoted = [row["root_idea_id"] for row in ranked[:self.policy.promote_ideas]]
            self.data["promoted_ideas"] = promoted
            self.data["screening_ranking"] = [{"root_idea_id": row["root_idea_id"], "score": row["score"],
                                                "lease_id": row["lease_id"]} for row in ranked]
            for root_id in promoted:
                self.data["ideas"][root_id]["promoted"] = True
            if promoted:
                self._transition("tuning", "screening_cohort_complete_promote_correct_roots_by_score_then_stable_id")
            else:
                self._freeze("no_qualified_new_idea")
        elif stage == "tuning":
            if all(sum(self.data["leases"][key]["stage"] == "tuning" for key in self.data["ideas"][root_id]["history"])
                   >= self.policy.tuning_trials_per_idea for root_id in self.data["promoted_ideas"]):
                reason = "highest_correct_development_score_after_joint_tuning" if self.data["best"]["root_idea_id"] != "baseline" else "no_improvement"
                self._freeze(reason)
        elif stage == "replication":
            repeats = [self.data["outcomes"][key] for key in sorted(self.data["outcomes"])
                       if self.data["leases"][key]["stage"] == "replication"]
            if len(repeats) == self.policy.replication_evals:
                scores = [row["score"] for row in repeats]
                correct = all(row["correct"] for row in repeats)
                self.data["aggregates"]["frozen_candidate"] = {
                    "n": len(scores), "mean": mean(scores), "std": pstdev(scores), "all_correct": correct,
                    "receipt_ids": [row["receipt_id"] for row in repeats],
                    "source_hash": self.data["frozen_candidate"]["source_hash"],
                    "evidence_scope": "same_frozen_development_protocol_repeatability", "independent_confirmation": False}
                self._transition("ready_for_validation" if correct else "replication",
                                 "frozen_candidate_repeats_complete" if correct else "frozen_candidate_replication_failed",
                                 status="ready_for_validation" if correct else "incomplete")

    def _freeze(self, reason: str) -> None:
        self.data["frozen_candidate"] = deepcopy(self.data["best"])
        self.data["selection_reason"] = reason
        self._transition("replication", reason)

    def record_failure(self, lease_id: str, category: str, retryable: bool = True) -> bool:
        if not isinstance(category, str) or not category or type(retryable) is not bool:
            raise ValueError("Portfolio failure needs a category and boolean retryable flag")
        with self._transaction():
            if lease_id not in self.data["pending"]:
                raise ValueError("Portfolio failure has no pending lease")
            self.data["failures"].append({"sequence": len(self.data["failures"]), "lease_id": lease_id,
                                         "category": category, "retryable": retryable})
            self.data["transport_failures"] += 1
            if self.data["transport_failures"] >= self.policy.max_transport_failures:
                self._transition(self.data["stage"], "transport_failure_budget_exhausted", status="incomplete")
            elif not retryable:
                self._transition(self.data["stage"], f"nonretryable_failure:{category}", status="incomplete")
            return True

    def stop(self, reason: str) -> None:
        with self._transaction():
            if self.data["status"] != "ready_for_validation":
                self._transition(self.data["stage"], reason, status="incomplete")

    def resume_after_transport_recovery(self) -> bool:
        """Resume only after every failed lease has a fresh, valid host receipt.

        This explicit transaction never refunds failures or scientific work.
        Ordinary resume still respects an incomplete campaign. In particular,
        the next transport failure still trips an exhausted transport budget.
        """
        from .portfolio_campaign import _receipt_for_lease

        with self._transaction():
            previous = self.data.get("transport_recoveries", [])
            if self.data["status"] != "incomplete":
                if previous and previous[-1]["transport_failures"] == self.data["transport_failures"]:
                    return False
                raise ValueError("Transport recovery requires an incomplete campaign")
            if self.data["transition_reason"] not in {
                "transport_failure_budget_exhausted",
                "nonretryable_failure:evaluation_only_recovery_required",
            }:
                raise ValueError("Campaign did not stop for a recoverable transport failure")
            failed = sorted({row["lease_id"] for row in self.data["failures"]})
            if not failed:
                raise ValueError("Transport recovery requires a retained failure ledger")
            if any(key not in self.data["outcomes"] for key in failed):
                return False
            receipts = {}
            for key in failed:
                lease = self.data["leases"][key]
                self._check_parent(lease["parent"])
                verified = _receipt_for_lease(lease, self.task_dir, _context(self.data, lease))
                if verified is not None and "plan_sha256" in self.data:
                    verified.update({field: lease.get(field) for field in ("plan_sha256", "card_sha256", "role")})
                if verified is None or verified != self.data["outcomes"][key]:
                    raise ValueError("Transport recovery requires revalidated receipts for every failed lease")
                receipts[key] = verified["receipt_id"]
            recovery = {"sequence": len(previous), "transport_failures": self.data["transport_failures"],
                        "failure_sequences": [row["sequence"] for row in self.data["failures"]],
                        "receipt_ids": receipts, "valid_evaluations": self.data["valid_evaluations"]}
            self.data.setdefault("transport_recoveries", []).append(recovery)
            self._transition(self.data["stage"], "transport_failures_recovered_from_validated_receipts")
            self._advance()
            return True

    def _handoff(self) -> dict[str, Any] | None:
        if self.data["status"] != "ready_for_validation":
            return None
        delta = self.data["frozen_candidate"]["score"] - self.data["baseline"]["score"]
        repeats = self.data["aggregates"]["frozen_candidate"]
        repeat_delta = repeats["mean"] - self.data["baseline"]["score"]
        supports_improvement = (
            self.data["frozen_candidate"]["root_idea_id"] != "baseline"
            and repeats["all_correct"]
            and delta > self.policy.min_improvement
            and repeat_delta > self.policy.min_improvement
        )
        handoff = {"schema_version": 1, "status": "ready_for_validation", "mode": "whole_idea_portfolio",
                "candidate": deepcopy(self.data["frozen_candidate"]), **self.identity,
                "development_aggregates": deepcopy(self.data["aggregates"]),
                "selection_reason": self.data["selection_reason"], "baseline_score": self.data["baseline"]["score"],
                "development_score_delta": delta, "replication_score_delta": repeat_delta,
                "claim_supported": supports_improvement,
                "claim_scope": "development_repeat_mean_improvement_over_qualified_baseline",
                "required_phases": ["ablations", "final_evaluation", "posthoc", "paper"],
                "independent_confirmation_complete": False, "submission_ready": False}
        if "plan_sha256" in self.data:
            handoff.update(
                campaign_plan=deepcopy(self.data["campaign_plan"]),
                baseline_evidence_role="bootstrap", baseline_tuned_comparator=False,
                planned_root_execution=[{
                    "root_idea_id": root_id, "card_sha256": root["card_sha256"], "role": root["role"],
                    "promoted": root["promoted"], "own_best": deepcopy(root["best"]),
                    "actual_evaluations": {stage: sum(
                        self.data["leases"][key]["root_idea_id"] == root_id
                        and self.data["leases"][key]["stage"] == stage for key in self.data["outcomes"])
                        for stage in ("screening", "tuning", "replication")},
                } for root_id, root in sorted(self.data["ideas"].items())],
            )
        return handoff

    def _write_handoff(self) -> None:
        handoff = self._handoff()
        if handoff is not None:
            state_store.write_private_text(self.results_dir / HANDOFF_FILE, json.dumps(handoff, sort_keys=True, indent=2) + "\n")

    def handoff(self) -> dict[str, Any] | None:
        self.data = read_portfolio_state(self.results_dir, self.task_dir, self.policy)
        return self._handoff()
