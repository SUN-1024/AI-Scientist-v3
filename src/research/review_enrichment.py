"""Bounded literature feedback for the interpretation of a selected candidate.

The candidate, execution plan and observed results never change in this loop.
Each complete review is sealed before retrieval can start. Failed paid attempts
remain charged across process restarts. Delivery verification is read-only.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.orchestrator.state import write_private_text
from .agent_literature import build_agent_literature_backend
from .config import ResearchConfig
from .novelty import content_hash, freeze_evidence, validate_candidate_packet
from .novelty_review import (
    InvalidReviewResume, _configuration_hash, _replay_call, draft_research_agenda,
    load_frozen_novelty_report, load_review_records, review_novelty, source_packet,
)
from .research_agenda import load_agenda
from .review_retrieval import (
    RetrievalBudget, enrich_review_evidence, extract_review_requests,
    load_enrichment_result, validate_enrichment_result,
)
from .workspace import task_digest

VERSION = 1
REFLECTIONS = 3
MAX_ENRICHMENTS = 2
RETRY_ALLOWANCE = 3
MAX_ELAPSED_SECONDS = 7200
_TERMINAL = {"supported", "overlap", "no_novelty_claim"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value: Any) -> None:
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _path(workspace: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        if not str(value) or str(path) != str(value) or ".." in path.parts:
            raise ValueError("Selected evidence requires canonical workspace-relative paths")
        path = workspace / path
    if not path.is_relative_to(workspace) or ".." in path.parts or path != Path(os.path.abspath(path)):
        raise ValueError("Selected evidence path escaped its workspace")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Selected evidence paths cannot cross symlinks")
    return path


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"Invalid selected evidence artifact: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Selected evidence artifact must be an object")
    return value


def _tree(path: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Selected evidence must be a regular directory")
    files = {}
    for parent, directories, names in os.walk(path, followlinks=False):
        for name in directories + names:
            child = Path(parent) / name
            mode = child.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Selected evidence cannot contain symlinks or special files")
            relative = child.relative_to(path).as_posix()
            if stat.S_ISREG(mode) and relative not in exclude:
                files[relative] = _sha(child)
    return dict(sorted(files.items()))


@contextmanager
def _lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Invalid selected review lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Selected review chain is already running") from exc
        yield
    finally:
        os.close(fd)


def _trusted_base(workspace: Path) -> dict[str, Any]:
    state = _read(_path(workspace, "harness_state.json"))
    preparation = _read(_path(workspace, "research_preparation.json"))
    manifest = _read(_path(workspace, "research_manifest.json"))
    initial = preparation.get("initial_harness_state")
    if (state.get("selected_review_enrichment_version") != VERSION
            or not isinstance(initial, dict) or initial.get("selected_review_enrichment_version") != VERSION
            or preparation.get("schema_version") != 1 or preparation.get("state") != "prepared"
            or manifest.get("schema_version") != 1 or manifest.get("state") != "prepared"):
        raise ValueError("Selected enrichment was not enabled by trusted fresh preparation")
    base = _path(workspace, "input_task")
    _tree(base)
    digest = task_digest(base)
    if any(row.get("input_task_sha256") != digest for row in (state, preparation, manifest)):
        raise ValueError("Selected enrichment base input changed")
    return {"base_input_sha256": digest,
            "preparation_initial_state_sha256": content_hash(initial),
            "preparation_sha256": _sha(workspace / "research_preparation.json"),
            "manifest_sha256": _sha(workspace / "research_manifest.json")}


def _final_opened(workspace: Path) -> bool:
    return any(_path(workspace, f"run/final_evaluation/{name}").exists()
               for name in ("invocation.json", "comparison_set.json"))


def _chain_path(workspace: Path, packet: dict[str, Any]) -> Path:
    validate_candidate_packet(packet)
    if not isinstance(packet.get("research_basis"), dict):
        raise ValueError("Selected candidate requires its research basis")
    return _path(workspace, "novelty/selected_enrichment/" + content_hash(packet))


def _fixed(cfg, packet, author, reviewer, offline, base):
    count = cfg.harness.novelty_reviewers
    if type(count) is not int or not 2 <= count <= 4:
        raise ValueError("Selected review requires two to four reviewers")
    return {**base, "version": VERSION, "candidate_packet_sha256": content_hash(packet),
            "configuration_sha256": content_hash(asdict(cfg)),
            "author_configuration_sha256": _configuration_hash(cfg.literature, author),
            "reviewer_configuration_sha256": _configuration_hash(cfg.literature, reviewer),
            "author_settings_sha256": content_hash(author), "reviewer_settings_sha256": content_hash(reviewer),
            "reviewer_count": count, "reflection_count": REFLECTIONS, "max_enrichments": MAX_ENRICHMENTS,
            "max_author_reviewer_attempts": (REFLECTIONS + count) * (MAX_ENRICHMENTS + 1) + RETRY_ALLOWANCE,
            "max_elapsed_seconds": MAX_ELAPSED_SECONDS, "retrieval_budget": asdict(RetrievalBudget()),
            "offline": bool(offline), "novelty_review_enabled": bool(cfg.harness.novelty_review)}


def _read_chain(directory: Path) -> dict[str, Any]:
    chain = _read(directory / "chain.json")
    if (chain.get("schema_version") != 1 or not isinstance(chain.get("fixed"), dict)
            or not isinstance(chain.get("cycles"), list) or not isinstance(chain.get("attempts"), list)
            or type(chain.get("created_at_unix")) not in (int, float)):
        raise ValueError("Invalid selected enrichment chain")
    fixed = chain["fixed"]
    if len(chain["cycles"]) > fixed["max_enrichments"] + 1:
        raise ValueError("Selected enrichment cycle budget exceeded")
    for index, row in enumerate(chain["cycles"]):
        if row.get("index") != index or row.get("state") not in {"started", "completed"}:
            raise ValueError("Selected review sequence changed")
        if row["state"] != "completed" and index != len(chain["cycles"]) - 1:
            raise ValueError("Unfinished selected review has a successor")
    for attempt in chain["attempts"]:
        if (attempt.get("stage") not in {"author", "reviewer"}
                or type(attempt.get("charged")) is not int or attempt["charged"] < 0
                or type(attempt.get("reserved")) is not int or not 0 <= attempt["charged"] <= attempt["reserved"]
                or attempt.get("state") not in {"started", "finished"}):
            raise ValueError("Selected review attempt ledger changed")
    charged = sum(row["charged"] for row in chain["attempts"])
    observed = sum(_receipt_count(directory / f"cycle_{row['index']}", phase)
                   for row in chain["cycles"] for phase in ("research_agenda", "novelty_review"))
    if charged > fixed["max_author_reviewer_attempts"] or observed > charged:
        raise ValueError("Selected review attempt budget or ledger changed")
    return chain


def _pointer(workspace, directory, fixed, seal):
    return {"schema_version": 1, "status": seal["status"], "cycle": seal["cycle"],
            "candidate_source_sha256": seal["candidate_source_sha256"],
            "candidate_packet_sha256": fixed["candidate_packet_sha256"],
            "candidate_program": seal["candidate_program"], "assessed_at": seal["assessed_at"],
            "evidence_reference": {"schema_version": 1, "kind": "selected_review_enrichment",
                "chain": directory.relative_to(workspace).as_posix(),
                "chain_binding_sha256": content_hash(fixed), "base_input_sha256": fixed["base_input_sha256"],
                "seal": seal["cycle"] + "/seal.json", "seal_sha256": _sha(workspace / seal["cycle"] / "seal.json"),
                "evidence_task": seal["evidence_task"], "evidence_task_sha256": seal["evidence_task_sha256"]}}


def _verify_cycle(workspace, directory, chain, row, packet, parent):
    cycle = _path(workspace, directory / f"cycle_{row['index']}")
    seal = _read(cycle / "seal.json")
    if row.get("seal_sha256") != _sha(cycle / "seal.json") or seal.get("cycle") != cycle.relative_to(workspace).as_posix():
        raise ValueError("Selected cycle seal changed")
    if seal.get("files") != _tree(cycle, exclude=("seal.json",)):
        raise ValueError("Selected cycle artifacts changed")
    if (seal.get("chain_binding_sha256") != content_hash(chain["fixed"])
            or seal.get("candidate_packet_sha256") != content_hash(packet)
            or seal.get("candidate_source_sha256") != packet["candidate_source_sha256"]
            or seal.get("candidate_program") != packet["selection"]["program_relative_path"]):
        raise ValueError("Selected cycle candidate or configuration changed")
    task = _path(workspace, seal["evidence_task"])
    _tree(task)
    if task_digest(task) != seal.get("evidence_task_sha256"):
        raise ValueError("Selected cycle evidence task changed")
    if parent is None:
        if task != workspace / "input_task" or seal.get("parent_seal_sha256") is not None or seal.get("enrichment") is not None:
            raise ValueError("Selected initial cycle has invalid ancestry")
    else:
        parent_path = workspace / parent["cycle"]
        if (seal.get("parent_seal_sha256") != _sha(parent_path / "seal.json")
                or seal.get("parent_review_sha256") != _sha(parent_path / "novelty_report.json")):
            raise ValueError("Selected cycle parent review changed")
        result = load_enrichment_result(workspace, directory / "retrieval",
                  base_input_sha256=chain["fixed"]["base_input_sha256"], candidate_packet=packet,
                  evidence_task=seal["evidence_task"], expected_result_sha256=content_hash(seal["enrichment"]))
        if result != seal["enrichment"] or result["binding"]["review_cycle"] != parent["cycle"]:
            raise ValueError("Selected cycle retrieval parent changed")
        if _read(cycle / "review_feedback.json") != _feedback(workspace, parent, result, packet):
            raise ValueError("Selected cycle lost unresolved parent feedback")
    agenda = _read(cycle / "research_agenda.json")
    records, errors = load_review_records(task, agenda, cycle, candidate_packet=packet)
    if errors or len(records) != chain["fixed"]["reviewer_count"]:
        raise ValueError("Selected cycle lacks the complete independent review set")
    report = load_frozen_novelty_report(task, cycle, candidate_packet=packet)
    if content_hash(report) != seal.get("replayed_report_sha256") or report["status"] != seal["status"]:
        raise ValueError("Selected cycle report replay changed")
    # Replay author responses as well as reviewer responses at the paper boundary.
    authors = [cycle / "agenda_draft", *sorted(cycle.glob("agenda_reflection_[2-5]"))]
    for author in authors:
        receipt = _read(author / "receipt.json")
        if receipt.get("configuration_sha256") != chain["fixed"]["author_configuration_sha256"]:
            raise ValueError("Selected author configuration changed")
        frozen = {**freeze_evidence(task), "candidate_packet_sha256": content_hash(packet)}
        for filename, key in (("research_seed_agenda.json", "seed_agenda_sha256"), ("review_feedback.json", "review_feedback_sha256")):
            if (cycle / filename).exists():
                frozen[key] = content_hash(_read(cycle / filename))
        final_agenda = _replay_call(author, receipt, "research_agenda", frozen)
    final_agenda.pop("_reflection", None)
    if final_agenda != agenda:
        raise ValueError("Selected agenda differs from the final author response")
    for record in records:
        receipt = _read(Path(record["receipt"]["path"]))
        if receipt.get("configuration_sha256") != chain["fixed"]["reviewer_configuration_sha256"]:
            raise ValueError("Selected reviewer configuration changed")
    return seal


def _verified_completed(workspace, directory, chain, packet):
    parent = None
    for row in chain["cycles"]:
        if row["state"] == "completed":
            parent = _verify_cycle(workspace, directory, chain, row, packet, parent)
    return parent


def resolve_selected_evidence_task(workspace: Path, pointer: dict[str, Any], *, candidate_packet: dict[str, Any]) -> Path:
    """Verify all ancestors, sources and receipts without invoking any backend."""
    workspace = Path(os.path.abspath(workspace))
    directory = _chain_path(workspace, candidate_packet)
    base = _trusted_base(workspace)
    chain = _read_chain(directory)
    if any(chain["fixed"].get(key) != value for key, value in base.items()):
        raise ValueError("Selected chain preparation binding changed")
    parent = None
    for row in chain["cycles"]:
        if row["state"] != "completed":
            break
        parent = _verify_cycle(workspace, directory, chain, row, candidate_packet, parent)
        expected = _pointer(workspace, directory, chain["fixed"], parent)
        if pointer.get("cycle") == expected["cycle"]:
            if pointer != expected:
                raise ValueError("Selected evidence pointer changed")
            return _path(workspace, parent["evidence_task"])
    raise ValueError("Pointer has no completed selected review cycle")


def _feedback(workspace, parent, result, packet):
    cycle = _path(workspace, parent["cycle"])
    task = _path(workspace, parent["evidence_task"])
    requests = extract_review_requests(task, cycle, candidate_packet=packet)
    new_sources = [source for source in source_packet(_path(workspace, result["evidence_task"]))["sources"]
                   if source["id"] in result["new_source_ids"]]
    # Retrieved is not resolved: only the next independent review can settle it.
    return {"schema_version": 1, "scope": "selected_interpretation",
            "parent_review_sha256": _sha(cycle / "novelty_report.json"),
            "parent_report": load_frozen_novelty_report(task, cycle, candidate_packet=packet),
            "review_requests": requests["requests"], "resolved_requests": [],
            "unresolved_requests": requests["requests"], "new_source_ids": result["new_source_ids"],
            "new_source_information": [{key: source.get(key) for key in
                ("id", "title", "url", "availability", "metadata_status", "text_sha256")} for source in new_sources],
            "retrieval_result_sha256": content_hash(result)}


def _receipt_count(directory, phase):
    count = 0
    if directory.exists():
        for path in directory.rglob("receipt.json"):
            # Agent transport receipts are not additional author/reviewer calls.
            if _read(path).get("phase") == phase:
                count += 1
    return count


def _pending_author(cycle):
    trace = cycle / "agenda_reflection_trace.json"
    if trace.is_file() and _read(trace).get("status") in {"converged", "reflection_budget_exhausted"}:
        history = _read(trace).get("history", [])
        rounds = ["agenda_draft", *(f"agenda_reflection_{row['round']}" for row in history if row['round'] > 1)]
        if (not history or len(rounds) != len(set(rounds))
                or len(rounds) > REFLECTIONS
                or any(not (cycle / name / "receipt.json").is_file()
                       or _read(cycle / name / "receipt.json").get("status") != "complete" for name in rounds)):
            raise InvalidReviewResume("Completed author trace has missing or unfinished receipts")
        if _read(trace)["status"] == "converged":
            final = _read(cycle / rounds[-1] / "assessment.json")
            previous = _read(cycle / rounds[-2] / "assessment.json") if len(rounds) > 1 else {}
            stop = final.pop("_reflection", {}).get("done")
            previous.pop("_reflection", None)
            if stop is not True or final != previous:
                raise InvalidReviewResume("Completed author convergence disagrees with its responses")
        elif len(rounds) != REFLECTIONS:
            raise InvalidReviewResume("Completed author trace has missing reflection rounds")
        return 0
    return REFLECTIONS - sum(1 for name in ["agenda_draft", "agenda_reflection_2", "agenda_reflection_3"]
        if (cycle / name / "receipt.json").is_file() and _read(cycle / name / "receipt.json").get("status") == "complete")


def _run_stage(workspace, directory, chain, cycle, stage, reserve, function):
    if reserve == 0:
        return function()
    fixed = chain["fixed"]
    used = sum(row["charged"] for row in chain["attempts"])
    if (used + reserve > fixed["max_author_reviewer_attempts"]
            or time.time() - chain["created_at_unix"] >= fixed["max_elapsed_seconds"]):
        raise ValueError("Selected review attempt or elapsed budget exhausted")
    if _final_opened(workspace):
        raise ValueError("Final evaluation has started; selected review is read-only")
    phase = "research_agenda" if stage == "author" else "novelty_review"
    before = _receipt_count(cycle, phase)
    attempt = {"cycle": cycle.name, "stage": stage, "reserved": reserve, "charged": reserve,
               "state": "started", "started_at_unix": time.time()}
    chain["attempts"].append(attempt)
    _dump(directory / "chain.json", chain)
    completed = False
    try:
        result = function()
        completed = True
        return result
    finally:
        observed = _receipt_count(cycle, phase) - before
        # Completed replay has no new receipt and costs zero. On an exception
        # without a receipt, retain the reservation because the outcome is unknown.
        charged = min(reserve, observed) if observed > 0 or completed else reserve
        attempt.update(state="finished", charged=charged,
                       observed_receipts=observed, finished_at_unix=time.time())
        _dump(directory / "chain.json", chain)


def ensure_selected_novelty(workspace: Path, cfg: ResearchConfig, *, packet: dict[str, Any],
                            author_settings: dict[str, Any] | None,
                            reviewer_settings: dict[str, Any] | None, offline: bool = False) -> dict[str, Any]:
    """Return a fully reviewed pointer, preserving earlier cycles on failure.

    The caller publishes novelty/latest.json only after this function returns.
    A first incomplete cycle raises; a later incomplete cycle returns the prior
    completed pointer. Resume retries only unfinished calls within the same budget.
    """
    workspace = Path(os.path.abspath(workspace))
    packet = copy.deepcopy(packet)
    cfg = copy.deepcopy(cfg)
    author_settings = copy.deepcopy(author_settings)
    reviewer_settings = copy.deepcopy(reviewer_settings)
    directory = _chain_path(workspace, packet)
    base = _trusted_base(workspace)
    fixed = _fixed(cfg, packet, author_settings, reviewer_settings, offline, base)
    if _final_opened(workspace):
        if not (directory / "chain.json").exists():
            raise ValueError("Final evaluation has started; no selected review may begin")
        chain = _read_chain(directory)
        if chain["fixed"] != fixed:
            raise ValueError("Selected review configuration changed")
        parent = _verified_completed(workspace, directory, chain, packet)
        if parent is None:
            raise ValueError("Final evaluation has started with no complete selected review")
        return _pointer(workspace, directory, fixed, parent)
    if offline or not cfg.harness.novelty_review:
        raise ValueError("Selected enrichment requires enabled online independent review")
    with _lock(directory):
        if (directory / "chain.json").exists():
            chain = _read_chain(directory)
            if chain["fixed"] != fixed:
                raise ValueError("Selected review configuration, budget or preparation changed")
        else:
            if any(path.name != ".lock" for path in directory.iterdir()):
                raise ValueError("Selected chain journal is missing; its prior budget cannot be reset")
            if _final_opened(workspace):
                raise ValueError("Final evaluation has started; no selected review may begin")
            chain = {"schema_version": 1, "fixed": fixed, "created_at_unix": time.time(),
                     "cycles": [], "attempts": [], "stop_reason": None}
            _dump(directory / "chain.json", chain)
        parent = _verified_completed(workspace, directory, chain, packet)
        def current():
            if parent is None:
                raise ValueError("No complete selected review; previous latest pointer remains unchanged")
            return _pointer(workspace, directory, fixed, parent)
        while True:
            if chain.get("stop_reason"):
                return current()
            if parent is not None and parent["status"] in _TERMINAL:
                chain["stop_reason"] = parent["status"]
                _dump(directory / "chain.json", chain)
                return current()
            if _final_opened(workspace):
                return current()
            if time.time() - chain["created_at_unix"] >= fixed["max_elapsed_seconds"]:
                chain["stop_reason"] = "elapsed_budget_exhausted"
                _dump(directory / "chain.json", chain)
                return current()
            pending = chain["cycles"][-1] if chain["cycles"] and chain["cycles"][-1]["state"] == "started" else None
            if pending is None:
                index = len(chain["cycles"])
                if index > fixed["max_enrichments"]:
                    chain["stop_reason"] = "enrichment_round_budget_exhausted"
                    _dump(directory / "chain.json", chain)
                    return current()
                result = feedback = None
                task = workspace / "input_task"
                if parent is not None:
                    backend, backend_id = (build_agent_literature_backend(author_settings)
                        if not cfg.literature.model_search.model and author_settings is not None else (None, None))
                    result = enrich_review_evidence(workspace, workspace / parent["evidence_task"], workspace / parent["cycle"], cfg.literature,
                        journal_dir=directory / "retrieval", base_input_sha256=fixed["base_input_sha256"], candidate_packet=packet,
                        budget=RetrievalBudget(**fixed["retrieval_budget"]), offline=False,
                        search_backend=backend, search_backend_id=backend_id,
                        deadline_unix=chain["created_at_unix"] + fixed["max_elapsed_seconds"])
                    if result["status"] != "enriched":
                        chain["stop_reason"] = result["reason"]
                        _dump(directory / "chain.json", chain)
                        return current()
                    enriched_task = validate_enrichment_result(workspace, result)
                    if enriched_task is None:
                        raise ValueError("Enriched review result has no verified evidence task")
                    task = enriched_task
                    feedback = _feedback(workspace, parent, result, packet)
                if _final_opened(workspace):
                    return current()
                pending = {"index": index, "state": "started", "evidence_task": task.relative_to(workspace).as_posix(),
                           "evidence_task_sha256": task_digest(task), "enrichment": result, "review_feedback": feedback}
                chain["cycles"].append(pending)
                _dump(directory / "chain.json", chain)
            cycle = _path(workspace, directory / f"cycle_{pending['index']}")
            task = _path(workspace, pending["evidence_task"])
            _tree(task)
            if task_digest(task) != pending["evidence_task_sha256"]:
                raise ValueError("Unfinished selected review evidence changed")
            if parent is not None:
                load_enrichment_result(workspace, directory / "retrieval", base_input_sha256=fixed["base_input_sha256"],
                    candidate_packet=packet, evidence_task=pending["evidence_task"], expected_result_sha256=content_hash(pending["enrichment"]))
                if pending["review_feedback"] != _feedback(workspace, parent, pending["enrichment"], packet):
                    raise ValueError("Unfinished selected review feedback changed")
            elif pending["index"] != 0 or task != workspace / "input_task" or pending["enrichment"] is not None:
                raise ValueError("Unfinished selected initial cycle changed")
            cycle.mkdir(exist_ok=True)
            if (cycle / "seal.json").exists():
                # The seal is written before the chain update. Recover that
                # atomic-publication window without rewriting any cycle bytes.
                recovered = {**pending, "state": "completed", "seal_sha256": _sha(cycle / "seal.json")}
                parent = _verify_cycle(workspace, directory, chain, recovered, packet, parent)
                pending.update(state="completed", seal_sha256=recovered["seal_sha256"])
                _dump(directory / "chain.json", chain)
                continue
            original_agenda = load_agenda(workspace / "input_task")
            seed = ({"schema_version": 1, "agenda": original_agenda,
                     "derived_from": {"path": "input_task/research_agenda.json",
                                      "sha256": _sha(workspace / "input_task/research_agenda.json"),
                                      "status": "not_evaluated_for_selected_candidate"}}
                    if original_agenda is not None else None)
            try:
                agenda = _run_stage(workspace, directory, chain, cycle, "author", _pending_author(cycle), lambda:
                    draft_research_agenda(task, cfg.literature, cycle, research_basis=packet["research_basis"],
                        reviewer_settings=author_settings, candidate_packet=packet, num_reflections=REFLECTIONS,
                        resume=True, seed_agenda=seed, review_feedback=pending["review_feedback"],
                        deadline_unix=chain["created_at_unix"] + fixed["max_elapsed_seconds"]))
                remaining = fixed["reviewer_count"] - sum(1 for index in range(1, fixed["reviewer_count"] + 1)
                    if (cycle / f"reviewer_{index}/receipt.json").is_file()
                    and _read(cycle / f"reviewer_{index}/receipt.json").get("status") == "complete")
                _run_stage(workspace, directory, chain, cycle, "reviewer", remaining, lambda:
                    review_novelty(task, agenda, cfg.literature, cycle, reviewer_count=fixed["reviewer_count"],
                        reviewer_settings=reviewer_settings, candidate_packet=packet, resume=True,
                        deadline_unix=chain["created_at_unix"] + fixed["max_elapsed_seconds"]))
                records, errors = load_review_records(task, agenda, cycle, candidate_packet=packet)
                if errors or len(records) != fixed["reviewer_count"]:
                    return current()
            except InvalidReviewResume:
                raise
            except (ValueError, OSError, TypeError, KeyError):
                if parent is None:
                    raise
                return current()
            report = load_frozen_novelty_report(task, cycle, candidate_packet=packet)
            seal = {"schema_version": 1, "cycle": cycle.relative_to(workspace).as_posix(),
                    "chain_binding_sha256": content_hash(fixed), "candidate_packet_sha256": content_hash(packet),
                    "candidate_source_sha256": packet["candidate_source_sha256"],
                    "candidate_program": packet["selection"]["program_relative_path"],
                    "evidence_task": pending["evidence_task"], "evidence_task_sha256": task_digest(task),
                    "enrichment": pending["enrichment"], "status": report["status"],
                    "parent_seal_sha256": _sha(workspace / parent["cycle"] / "seal.json") if parent else None,
                    "parent_review_sha256": _sha(workspace / parent["cycle"] / "novelty_report.json") if parent else None,
                    "replayed_report_sha256": content_hash(report), "assessed_at": datetime.now(timezone.utc).isoformat(),
                    "files": _tree(cycle, exclude=("seal.json",))}
            _dump(cycle / "seal.json", seal)
            pending.update(state="completed", seal_sha256=_sha(cycle / "seal.json"))
            _dump(directory / "chain.json", chain)
            parent = _verify_cycle(workspace, directory, chain, pending, packet, parent)
