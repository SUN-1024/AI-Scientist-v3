"""Freeze a planned portfolio's method comparison before any final invocation.

The collection chooses no new winner and runs no search. Its inputs are the
terminal controller's selected method and every planned control's own best.
Each method uses the existing sealed, once-only final evaluator unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator

from src.investigation.adaptive_runtime import validated_development_outcome
from src.investigation.portfolio_controller import STATE_FILE, read_portfolio_state
from src.investigation.program_spec import entrypoint_relpath_from_spec
from src.investigation.search_tree import _fingerprint
from src.investigation.task_contract import TASK_CONTRACT_FILENAME, load_task_contract, task_scaffold_fingerprint

from .runner import (
    FinalEvaluationError, _file_digest, _load_json, _safe_declared_file, _tree_digest,
    _utc_now, _write_exclusive_json, resolve_frozen_incumbent, verify_final_manifest,
)

SET_MANIFEST = "comparison_set.json"
_EXECUTION_MANIFEST = "comparison_execution.json"
_METHOD_MANIFEST = "comparison_method.json"
_LOCK = ".comparison_set.lock"
_METHOD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _path(path: Path) -> Path:
    """Reject symlink ancestors before normalizing a managed path."""
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise FinalEvaluationError(f"comparison set path contains a symlink: {part}")
    return absolute


def _inside(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise FinalEvaluationError("comparison method needs an absolute host path")
    path = _path(Path(raw))
    if not path.is_relative_to(root) or path == root:
        raise FinalEvaluationError("comparison method escaped its investigation")
    return path


@contextmanager
def _locked(output_dir: Path) -> Iterator[None]:
    output_dir = _path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = output_dir.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise FinalEvaluationError("comparison output must be owner-controlled")
    descriptor = os.open(output_dir / _LOCK, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise FinalEvaluationError("unsafe comparison set lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def portfolio_final_set_required(*, task_dir: Path, investigation_results_dir: Path) -> bool:
    """Recognize planned state; never upgrade an existing legacy campaign."""
    from src.research.campaign_plan import CAMPAIGN_PLAN_PATH, HOST_CONTEXT_PATH

    # Do not add new path restrictions to an unplanned legacy invocation.
    root, task = Path(investigation_results_dir).expanduser(), Path(task_dir).expanduser()
    state = root / STATE_FILE
    if state.exists() or state.is_symlink():
        # Read the small discriminator without changing legacy state or files.
        # Full state validation is mandatory in the planned path below.
        raw = _load_json(state, "portfolio state")
        planned = ("campaign_plan" in raw or "plan_sha256" in raw
                   or any(isinstance(row, dict) and "planned_card" in row
                          for row in raw.get("ideas", {}).values()))
        if planned:
            _state(_path(root), _path(task))
        return planned
    if any((task / path).exists() or (task / path).is_symlink()
           for path in (CAMPAIGN_PLAN_PATH, HOST_CONTEXT_PATH)):
        raise FinalEvaluationError("planned final comparison has no portfolio state")
    agenda = task / "research_agenda.json"
    if agenda.is_file() and "execution_plan" in _load_json(agenda, "research agenda"):
        raise FinalEvaluationError("declared final comparison has no portfolio state")
    return False


def _state(root: Path, task: Path) -> dict[str, Any]:
    try:
        state = read_portfolio_state(root, task)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FinalEvaluationError(f"invalid planned portfolio: {exc}") from exc
    if "campaign_plan" not in state:
        raise FinalEvaluationError("final comparison requires a bound campaign plan")
    if state["status"] != "ready_for_validation" or state["pending"]:
        raise FinalEvaluationError("final comparison requires a terminal qualified portfolio")
    if state.get("frozen_candidate") != state.get("best"):
        raise FinalEvaluationError("portfolio final candidate differs from its global best")
    candidates = [state["baseline"], *(row["best"] for row in state["ideas"].values() if row["best"])]
    if min(candidates, key=lambda row: (-row["score"], row["lease_id"])) != state["best"]:
        raise FinalEvaluationError("portfolio final candidate is not the original global best")
    return state


def _configuration(program: Path, receipt: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    """Keep source declarations distinct from receipt-bound observed settings."""
    declared = {row["name"]: row["default"] for row in card.get("parameter_space", [])}
    evidence = program / "evaluation_evidence.json"
    expected = receipt.get("adaptive_result_sha256", {}).get(evidence.name)
    result = {"status": "source_bound_only", "configuration": None,
              "configuration_sha256": None, "declared_configuration": declared,
              "artifact_path": None, "artifact_sha256": None}
    if expected is None:
        return result
    if _file_digest(evidence) != "sha256:" + expected:
        raise FinalEvaluationError("executed configuration artifact changed after host evaluation")
    payload = _load_json(evidence, "executed configuration evidence")
    result.update(artifact_path=str(evidence), artifact_sha256="sha256:" + expected)
    configs = [payload["configuration"]] if isinstance(payload.get("configuration"), dict) else []
    for episode in payload.get("episodes", []):
        solution = episode.get("solution") if isinstance(episode, dict) else None
        if isinstance(solution, dict) and isinstance(solution.get("configuration"), dict):
            configs.append(solution["configuration"])
    if configs:
        if any(config != configs[0] for config in configs):
            raise FinalEvaluationError("executed configuration changed across development observations")
        result.update(status="host_artifact_bound", configuration=configs[0],
                      configuration_sha256=_digest(configs[0]))
    return result


def _method(root: Path, task: Path, state: dict[str, Any], best: dict[str, Any],
            *, selected: bool, execution_program: Path | None = None) -> dict[str, Any]:
    root_id = best.get("root_idea_id")
    if not isinstance(root_id, str) or not _METHOD_ID.fullmatch(root_id):
        raise FinalEvaluationError("invalid comparison root ID")
    planned = state["ideas"].get(root_id)
    role = planned["role"] if planned else "bootstrap"
    source = _inside(root, best.get("program_dir"))
    source_tree = _tree_digest(source)
    if _fingerprint(source) != best.get("source_hash") or best.get("correct") is not True:
        raise FinalEvaluationError(f"comparison own-best source is invalid: {root_id}")
    receipt = _load_json(source / "evaluation_manifest.json", "host evaluation receipt")
    attempt_id, idea_id = receipt.get("attempt_id"), receipt.get("idea_id")
    if not isinstance(attempt_id, str) or not isinstance(idea_id, str):
        raise FinalEvaluationError(f"comparison own-best has no valid host receipt identity: {root_id}")
    observed = validated_development_outcome(
        source, task_dir=task, attempt_id=attempt_id, idea_id=idea_id, native=False,
    )
    if observed is None or observed["correct"] is not True:
        raise FinalEvaluationError(f"comparison own-best has no valid host receipt: {root_id}")
    for key in ("score", "correct", "program_dir", "source_hash"):
        if best.get(key) != observed.get(key):
            raise FinalEvaluationError(f"comparison own-best differs from its host receipt: {root_id}")
    source_card_hash, card = None, {}
    if planned:
        lease_id = best.get("lease_id")
        lease = state["leases"].get(lease_id)
        outcome = state["outcomes"].get(lease_id)
        if (not lease or not outcome or lease["stage"] == "replication"
                or lease["root_idea_id"] != root_id or planned["best"] != best):
            raise FinalEvaluationError(f"comparison own-best is not its root's original outcome: {root_id}")
        correct_history = [key for key in planned["history"] if state["outcomes"][key]["correct"]]
        if min(correct_history, key=lambda key: (-state["outcomes"][key]["score"], key)) != lease_id:
            raise FinalEvaluationError(f"comparison method is not the original root own-best: {root_id}")
        if any(outcome.get(key) != value for key, value in observed.items()):
            raise FinalEvaluationError(f"comparison outcome differs from its host receipt: {root_id}")
        if (best.get("receipt_id") != observed["receipt_id"] or best.get("idea_id") != root_id
                or best.get("attempt_id") != observed["attempt_id"]):
            raise FinalEvaluationError(f"comparison receipt belongs to another root: {root_id}")
        result = _inside(root, lease["result_dir"])
        if (result != root / "portfolio_trials" / lease_id
                or source.name != "program_1" or not source.parent.name.endswith("_attempt")
                or source.parent.parent.parent != result / "ideas"):
            raise FinalEvaluationError("comparison receipt is outside its original trial layout")
        envelope = _load_json(result / "portfolio_lease.json", "portfolio trial lease")
        if envelope != {"root_results_dir": str(root), "lease": lease}:
            raise FinalEvaluationError("comparison trial lease changed")
        idea_file = _inside(root, outcome.get("idea_file"))
        if idea_file != source.parent.parent / "idea_bundle.json":
            raise FinalEvaluationError("comparison source card belongs to another attempt")
        bundle = _load_json(idea_file, "comparison source card")
        if bundle != outcome.get("idea_bundle") or bundle.get("idea_id") != root_id:
            raise FinalEvaluationError("comparison source card changed")
        source_card_hash, card = _file_digest(idea_file), bundle["whole_idea"]
    elif root_id != "baseline" or best != state["baseline"]:
        raise FinalEvaluationError("comparison selected an unknown root")
    elif best.get("receipt_id") not in (None, observed["receipt_id"]):
        raise FinalEvaluationError("bootstrap receipt changed")
    program = _path(execution_program) if execution_program else source
    if _fingerprint(program) != observed["source_hash"]:
        raise FinalEvaluationError("selected execution source differs from the host-verified method")
    entrypoint = entrypoint_relpath_from_spec(program, required=False) or "src/program.py"
    _safe_declared_file(program, entrypoint, "comparison entrypoint")
    actual = {stage: sum(state["leases"][key]["root_idea_id"] == root_id
                         and state["leases"][key]["stage"] == stage for key in state["outcomes"])
              for stage in ("screening", "tuning", "replication")}
    return {
        "method_id": root_id, "root_idea_id": root_id, "role": role,
        "is_selected": selected, "is_control": role == "control",
        "output_relative_path": "." if selected else "comparators/" + root_id,
        "program_dir": str(program), "entrypoint_path": entrypoint,
        "source_program_dir": str(source), "source_sha256": observed["source_hash"],
        "program_tree_sha256": _tree_digest(program), "source_tree_sha256": source_tree,
        "host_receipt_sha256": "sha256:" + observed["receipt_id"],
        "lease_id": best["lease_id"], "card_sha256": planned["card_sha256"] if planned else None,
        "source_card_sha256": source_card_hash,
        "configuration_evidence": _configuration(source, receipt, card),
        "actual_evaluations": actual, "development_score": observed["score"],
        "tuned": actual["tuning"] > 0,
    }


def _expected(*, task_dir: Path, investigation_results_dir: Path) -> dict[str, Any]:
    root, task = _path(investigation_results_dir), _path(task_dir)
    try:
        state = _state(root, task)
        contract = load_task_contract(task / TASK_CONTRACT_FILENAME)
        program, selected_entrypoint = resolve_frozen_incumbent(root)
        selected = _method(root, task, state, state["frozen_candidate"],
                           selected=True, execution_program=program)
        if program / selected["entrypoint_path"] != selected_entrypoint:
            raise FinalEvaluationError("selected final entrypoint differs from its host-evaluated program specification")
        methods = [selected]
        for root_id, planned in sorted(state["ideas"].items()):
            if planned["role"] != "control" or root_id == selected["root_idea_id"]:
                continue
            if planned["best"] is None:
                raise FinalEvaluationError(f"required control has no valid own-best: {root_id}")
            methods.append(_method(root, task, state, planned["best"], selected=False))
        raw_contract = _load_json(task / TASK_CONTRACT_FILENAME, "task contract")
        gaps = [{"code": "executed_configuration_unverified", "method_id": method["method_id"],
                 "detail": "Frozen source is verified; the host receipt has no observed configuration object."}
                for method in methods if method["configuration_evidence"]["status"] != "host_artifact_bound"]
        proposal = selected["root_idea_id"] if selected["role"] == "proposal" else None
        if proposal is None:
            gaps.append({"code": "no_selected_proposal", "method_id": selected["method_id"],
                         "detail": "Development selected a control or bootstrap; proposal superiority is not established."})
        return {
            "schema_version": 1, "task_dir": str(task), "investigation_results_dir": str(root),
            "task_tree_sha256": task_scaffold_fingerprint(task),
            "evaluator_sha256": _file_digest(_safe_declared_file(task, contract.evaluators.final, "final evaluator")),
            "protocol": {key: raw_contract[key] for key in ("protocol", "metrics", "compute", "data")},
            "plan_sha256": state["plan_sha256"], "portfolio_state_sha256": _file_digest(root / STATE_FILE),
            "selected_method_id": selected["method_id"], "selected_proposal_root_id": proposal,
            "methods": methods, "gaps": gaps,
        }
    except FinalEvaluationError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FinalEvaluationError(f"invalid final comparison inputs: {exc}") from exc


def _load_frozen(*, task_dir: Path, investigation_results_dir: Path, output_dir: Path) -> dict[str, Any]:
    frozen = _load_json(_path(output_dir) / SET_MANIFEST, "final comparison set")
    payload = {key: value for key, value in frozen.items() if key != "set_sha256"}
    if frozen.get("set_sha256") != _digest(payload):
        raise FinalEvaluationError("final comparison set integrity mismatch")
    if not isinstance(payload.get("frozen_at"), str) or not payload["frozen_at"]:
        raise FinalEvaluationError("final comparison set has no freeze time")
    expected = _expected(task_dir=task_dir, investigation_results_dir=investigation_results_dir)
    if {key: value for key, value in payload.items() if key != "frozen_at"} != expected:
        raise FinalEvaluationError("final comparison method set, plan, protocol, or host evidence changed")
    return frozen


def _freeze(*, task_dir: Path, investigation_results_dir: Path, output_dir: Path) -> dict[str, Any]:
    path = output_dir / SET_MANIFEST
    if path.exists() or path.is_symlink():
        return _load_frozen(task_dir=task_dir, investigation_results_dir=investigation_results_dir, output_dir=output_dir)
    # Only our empty lock may precede the freeze. This catches old selected-only
    # finals, partial methods, deleted set manifests, and unexpected subfolders.
    if any(child.name != _LOCK for child in output_dir.iterdir()):
        raise FinalEvaluationError("final artifacts exist without a prior frozen comparison set")
    value = {**_expected(task_dir=task_dir, investigation_results_dir=investigation_results_dir),
             "frozen_at": _utc_now()}
    value["set_sha256"] = _digest(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".comparison-set-", dir=output_dir)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return _load_frozen(task_dir=task_dir, investigation_results_dir=investigation_results_dir, output_dir=output_dir)


def freeze_portfolio_final_set(*, task_dir: Path, investigation_results_dir: Path,
                               output_dir: Path) -> dict[str, Any]:
    """Publish all method identities atomically, before any final data access."""
    output_dir = _path(output_dir)
    with _locked(output_dir):
        return _freeze(task_dir=task_dir, investigation_results_dir=investigation_results_dir, output_dir=output_dir)


def resolve_portfolio_final_method(*, task_dir: Path, investigation_results_dir: Path,
                                   output_dir: Path, set_dir: Path, method_id: str,
                                   execution: dict[str, Any]) -> tuple[Path, Path]:
    """Private runner boundary: an ID must still have its original host receipt."""
    frozen = _load_frozen(task_dir=task_dir, investigation_results_dir=investigation_results_dir, output_dir=set_dir)
    method = next((row for row in frozen["methods"] if row["method_id"] == method_id), None)
    if method is None or _path(output_dir) != _path(set_dir) / method["output_relative_path"]:
        raise FinalEvaluationError("final method is outside its frozen comparison set")
    recorded = _verify_execution(_path(set_dir), frozen, required=True)
    if recorded is None:
        raise FinalEvaluationError("final methods lack their frozen execution protocol")
    if recorded["settings"] != execution:
        raise FinalEvaluationError("final method execution differs from its frozen comparison protocol")
    _method_binding(_path(set_dir), frozen, recorded, method, required=True)
    program = Path(method["program_dir"])
    return program, program / method["entrypoint_path"]


def _execution_settings(*, task_dir: Path, python_bin: Path, timeout_seconds: int = 21600,
                        container_image: str | None = None,
                        _allow_native_for_tests: bool = False, **_ignored: Any) -> dict[str, Any]:
    contract = load_task_contract(task_dir / TASK_CONTRACT_FILENAME)
    if timeout_seconds < 1:
        raise FinalEvaluationError("final evaluation timeout must be positive")
    return {"container_image": container_image, "python_bin": str(python_bin.expanduser().resolve()),
            "timeout_seconds": min(timeout_seconds, contract.compute.final.wall_clock_seconds),
            "native_for_tests": bool(_allow_native_for_tests)}


def _verify_execution(output_dir: Path, frozen: dict[str, Any], *, required: bool = False) -> dict[str, Any] | None:
    path = output_dir / _EXECUTION_MANIFEST
    started = any(path.exists() or path.is_symlink() for method in frozen["methods"]
                  for name in ("invocation.json", "final_manifest.json", "results")
                  for path in [output_dir / method["output_relative_path"] / name])
    if not path.exists() and not path.is_symlink():
        if required or started:
            raise FinalEvaluationError("final methods lack their frozen execution protocol")
        return None
    value = _load_json(path, "comparison execution protocol")
    if (set(value) != {"schema_version", "set_sha256", "settings", "requested_container_image", "execution_sha256"}
            or value["schema_version"] != 1 or value["set_sha256"] != frozen["set_sha256"]
            or value["execution_sha256"] != _digest({key: item for key, item in value.items() if key != "execution_sha256"})
            or not isinstance(value["settings"], dict)
            or set(value["settings"]) != {"container_image", "python_bin", "timeout_seconds", "native_for_tests"}):
        raise FinalEvaluationError("invalid frozen comparison execution protocol")
    for method in frozen["methods"]:
        _method_binding(output_dir, frozen, value, method, required=started)
    return value


def _method_binding(output_dir: Path, frozen: dict[str, Any], execution: dict[str, Any],
                    method: dict[str, Any], *, required: bool, create: bool = False) -> None:
    directory = _path(output_dir / method["output_relative_path"])
    path = directory / _METHOD_MANIFEST
    expected = {"schema_version": 1, "set_sha256": frozen["set_sha256"],
                "execution_sha256": execution["execution_sha256"], "method_id": method["method_id"],
                "program_tree_sha256": method["program_tree_sha256"]}
    if path.exists() or path.is_symlink():
        if _load_json(path, "comparison method binding") != expected:
            raise FinalEvaluationError("comparison method differs from its frozen set or execution protocol")
    elif create:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = directory.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise FinalEvaluationError("comparison method output must be owner-controlled")
        _write_exclusive_json(path, expected)
    elif required:
        raise FinalEvaluationError("final comparison method has no prior set and execution binding")


def verify_portfolio_final_set(*, task_dir: Path, investigation_results_dir: Path,
                               output_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    """Verify every retained result against the unchanged frozen method set."""
    output_dir = _path(output_dir)
    frozen = _load_frozen(task_dir=task_dir, investigation_results_dir=investigation_results_dir, output_dir=output_dir)
    execution = _verify_execution(output_dir, frozen)
    rows, gaps = [], deepcopy(frozen["gaps"])
    for method in frozen["methods"]:
        directory = _path(output_dir / method["output_relative_path"])
        path = directory / "final_manifest.json"
        completed = path.exists() or path.is_symlink()
        if require_complete and not completed:
            raise FinalEvaluationError(f"final comparison method is incomplete: {method['method_id']}")
        manifest = verify_final_manifest(directory, task_dir=task_dir,
                                         program_dir=Path(method["program_dir"])) if completed else None
        if manifest is not None:
            if execution is None:
                raise FinalEvaluationError("final methods lack their frozen execution protocol")
            if (manifest.get("compute_usage", {}).get("wall_clock_limit_seconds")
                    != execution["settings"]["timeout_seconds"]):
                raise FinalEvaluationError("final method changed its frozen compute protocol")
        evidence = directory / "results/evaluation_evidence.json"
        metrics_path = directory / "results/metrics.json"
        configuration_status = "not_evaluated"
        if completed:
            from src.investigation.evaluation_evidence import load_evaluation_evidence

            expected_configuration = method["configuration_evidence"]["configuration"]
            final_evidence = load_evaluation_evidence(directory / "results")
            episodes = (final_evidence or {}).get("episodes", [])
            observed_configurations = [episode.get("solution", {}).get("configuration")
                                       if isinstance(episode.get("solution"), dict) else None
                                       for episode in episodes]
            if expected_configuration is None:
                configuration_status = "development_configuration_unavailable"
            elif not observed_configurations or any(config is None for config in observed_configurations):
                configuration_status = "final_configuration_unavailable"
                gaps.append({"code": "final_configuration_unverified", "method_id": method["method_id"],
                             "detail": "The final evaluator did not report the frozen configuration for every episode."})
            elif any(config != expected_configuration for config in observed_configurations):
                raise FinalEvaluationError(f"final method changed its frozen configuration: {method['method_id']}")
            else:
                configuration_status = "matches_frozen_development_configuration"
        rows.append({**deepcopy(method), "final_manifest": manifest,
                     "final_manifest_path": str(path), "final_manifest_sha256": _file_digest(path) if completed else None,
                     "metrics": manifest["metrics"] if manifest is not None else None,
                     "final_metrics_path": str(metrics_path) if completed else None,
                     "final_metrics_sha256": _file_digest(metrics_path) if completed else None,
                     "final_configuration_status": configuration_status,
                     "evaluation_evidence_path": str(evidence) if completed and evidence.is_file() else None,
                     "evaluation_evidence_sha256": _file_digest(evidence) if completed and evidence.is_file() else None})
    return {"schema_version": 1, "status": "completed" if all(row["final_manifest"] for row in rows) else "incomplete",
            "set_manifest_path": str(output_dir / SET_MANIFEST), "set_manifest_sha256": _file_digest(output_dir / SET_MANIFEST),
            "set_sha256": frozen["set_sha256"], "selected_method_id": frozen["selected_method_id"],
            "selected_proposal_root_id": frozen["selected_proposal_root_id"], "methods": rows, "gaps": gaps,
            "plan_sha256": frozen["plan_sha256"], "task_tree_sha256": frozen["task_tree_sha256"],
            "protocol": deepcopy(frozen["protocol"]),
            "execution_manifest_path": str(output_dir / _EXECUTION_MANIFEST) if execution else None,
            "execution_manifest_sha256": _file_digest(output_dir / _EXECUTION_MANIFEST) if execution else None}


def run_portfolio_final_evaluation(*, task_dir: Path, investigation_results_dir: Path,
                                   output_dir: Path, **runner_options: Any) -> dict[str, Any]:
    """Run the precommitted methods under one final protocol, with no agent calls."""
    from .runner import run_final_evaluation

    if "_portfolio_method" in runner_options:
        raise FinalEvaluationError("a comparison caller cannot override its selected method")
    output_dir = _path(output_dir)
    with _locked(output_dir):
        frozen = _freeze(task_dir=task_dir, investigation_results_dir=investigation_results_dir, output_dir=output_dir)
        recorded = _verify_execution(output_dir, frozen)
        options = dict(runner_options)
        image = options.get("container_image")
        if image:
            from src.investigation.container_images import resolve_immutable_image

            if recorded and image in (recorded["requested_container_image"], recorded["settings"]["container_image"]):
                options["container_image"] = recorded["settings"]["container_image"]
            else:
                try:
                    options["container_image"] = options.get("_resolve_container_image", resolve_immutable_image)(image)
                except ValueError as exc:
                    raise FinalEvaluationError(f"final comparison image preflight failed: {exc}") from exc
            options.setdefault("container_image_recovery_ref", image)
        settings = _execution_settings(task_dir=Path(task_dir), **options)
        if recorded is None:
            recorded = {"schema_version": 1, "set_sha256": frozen["set_sha256"], "settings": settings,
                        "requested_container_image": image}
            recorded["execution_sha256"] = _digest(recorded)
            _write_exclusive_json(output_dir / _EXECUTION_MANIFEST, recorded)
        elif recorded["settings"] != settings:
            raise FinalEvaluationError("comparison execution protocol changed after freezing")
        # Bind every method before the first one can observe final data. A
        # restart cannot change execution metadata independently of the methods.
        for method in frozen["methods"]:
            _method_binding(output_dir, frozen, recorded, method, required=True, create=True)
        # Preflight all completed results before starting any missing method.
        verified = verify_portfolio_final_set(task_dir=task_dir, investigation_results_dir=investigation_results_dir,
                                              output_dir=output_dir, require_complete=False)
        for method in verified["methods"]:
            invocation = output_dir / method["output_relative_path"] / "invocation.json"
            if method["final_manifest"] is None and (invocation.exists() or invocation.is_symlink()):
                raise FinalEvaluationError("final evaluation already started; refusing rerun of an unfinished comparison method")
        for method in verified["methods"]:
            if method["final_manifest"] is not None:
                continue
            run_final_evaluation(task_dir=task_dir, investigation_results_dir=investigation_results_dir,
                                 output_dir=output_dir / method["output_relative_path"],
                                 _portfolio_method=(output_dir, method["method_id"]), **options)
            verify_portfolio_final_set(task_dir=task_dir, investigation_results_dir=investigation_results_dir,
                                       output_dir=output_dir, require_complete=False)
        return verify_portfolio_final_set(task_dir=task_dir, investigation_results_dir=investigation_results_dir,
                                          output_dir=output_dir)
