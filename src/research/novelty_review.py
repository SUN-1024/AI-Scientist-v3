"""Stateless provider reviews and replayable provenance for the novelty gate."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import signal
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.orchestrator.state import write_private_text
from .config import LiteratureConfig
from .evidence import load_catalog
from .literature import CONTEXT_DIRECTORY
from .model_search import SearchSession, parse_object, response_text
from .novelty import ACTION_TYPES, REVIEW_CHECKS, content_hash, evaluate_novelty, freeze_evidence, validate_candidate_packet
from .research_agenda import normalize_agenda
from .methodology import render_methodology
from .whole_idea_policy import render_whole_idea_guidance

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "research_harness"
MAX_PACKET_CHARS = 600_000
logger = logging.getLogger(__name__)


class InvalidReviewResume(ValueError):
    """Saved completed work cannot be trusted or does not match this invocation."""


def _dump(path: Path, value: Any) -> None:
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path, *, limit: int = 8 * 1024 * 1024) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Invalid or missing review artifact: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pin_input(path: Path, value: dict[str, Any], *, resume: bool,
               required_on_resume: bool = True) -> None:
    """Preserve the first input bytes and reject a changed resumed proposal."""
    if path.exists() or path.is_symlink():
        if _read(path) != value:
            raise InvalidReviewResume(f"Saved campaign input changed: {path.name}")
    else:
        if resume and required_on_resume and any(path.parent.glob("*/receipt.json")):
            raise InvalidReviewResume(f"Saved calls are missing campaign input: {path.name}")
        _dump(path, value)


def campaign_binding(host_context: dict[str, Any] | None,
                     compiled_plan: dict[str, Any] | None = None,
                     agenda: dict[str, Any] | None = None) -> dict[str, str]:
    """Bind only explicit campaign inputs; absence preserves legacy receipts."""
    if host_context is None:
        if compiled_plan is not None:
            raise ValueError("A compiled campaign plan requires its host context")
        return {}
    from .campaign_plan import validate_host_context, validate_campaign_plan
    context = validate_host_context(host_context)
    binding = {"host_context_sha256": context["host_context_sha256"]}
    if compiled_plan is not None:
        plan = validate_campaign_plan(compiled_plan, host_context=context, agenda=agenda)
        binding["campaign_plan_sha256"] = plan["plan_sha256"]
    return binding


def saved_campaign_inputs(output: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    context_path, plan_path = output / "host_context.json", output / "campaign_plan.json"
    context = _read(context_path) if context_path.exists() or context_path.is_symlink() else None
    plan = _read(plan_path) if plan_path.exists() or plan_path.is_symlink() else None
    if (context is None) != (plan is None):
        raise ValueError("Frozen campaign review requires both host context and compiled plan")
    return context, plan


def _verify_prompt_campaign(payload: dict[str, Any], frozen: dict[str, Any], *, author: bool) -> None:
    context, plan = payload.get("host_context"), payload.get("compiled_plan")
    binding = campaign_binding(context, plan, None if author else payload.get("agenda"))
    for key in ("host_context_sha256", "campaign_plan_sha256"):
        if binding.get(key) != frozen.get(key):
            raise ValueError("Saved prompt does not contain its frozen campaign inputs")
    if author:
        seed = payload.get("seed_agenda")
        if (content_hash(seed) if seed is not None else None) != frozen.get("seed_agenda_sha256"):
            raise ValueError("Saved author prompt does not contain its frozen seed agenda")
        feedback = payload.get("review_feedback")
        if (content_hash(feedback) if feedback is not None else None) != frozen.get("review_feedback_sha256"):
            raise ValueError("Saved author prompt does not contain its frozen review feedback")


def _campaign_author_guidance() -> str:
    return """HOST-BOUND CAMPAIGN PLAN:
The host_context contains the resolved execution budget, worker count, fixed task
contracts, and safe seed source. The seed is not_evaluated. It is not a selected
candidate or evidence of scientific success. If seed_agenda is supplied, it is a
previous proposal with derived_from provenance. Revise it against this context;
do not treat its old budget or prior paid calls as current execution or review.

Add execution_plan with exactly schema_version=1, roots, and allocation. Copy
host_context.allocation exactly. Return exactly allocation.initial_ideas complete
root cards. Each root requires exactly title, role (proposal|control), hypothesis,
complete_implementation, falsifier, decision_rule, and whole_idea. All six text
fields must be nonempty; complete_implementation must explain the full executable
idea under the fixed candidate interface. Do not put host IDs or hashes in cards.
Different cards may use the same innovation axis. Preserve strong controls.

whole_idea uses schema_version=1, innovation_axis
(problem|efficiency|data|objective|optimization|structure), problem_statement,
contribution, minimal_validation, parameter_space, and selection_reason.
minimal_validation contains protocol_id, decision_rule, and stop_or_pivot;
efficiency cards also require cost_metric and quality_guardrail. Use only a
protocol that the task contract permits. parameter_space is a list of objects:
name, type (integer|float|categorical|boolean), default, owner
(candidate|task|evaluator), and exactly one of values or numeric bounds. Declare
only candidate-owned tunables within the fixed task contract. Empty parameter
lists are allowed for fixed controls. Never add undeclared keys.

Screening, tuning, and repeats must fit allocation.total_evaluations. Tuning is
guaranteed for every root only when promote_ideas equals initial_ideas. Keep
ablation and confirmation limits separate from the portfolio evaluation count.
The host compiles this plan before independent reviewers or training can run.
"""


def source_packet(task_dir: Path, agenda: dict[str, Any] | None = None) -> dict[str, Any]:
    """Give authors and reviewers all saved texts, including non-closest sources.

    The agenda can attribute methods or exclusions through source_selection and
    source_scan as well as closest_work. Every such claim needs the same source
    evidence that the author received. Oversized packets fail without truncation.
    """
    catalog = load_catalog(task_dir) or {"sources": [], "status": "unavailable", "queries": []}
    rows = []
    for source in catalog["sources"]:
        row = {key: source.get(key) for key in ("id", "title", "year", "venue", "url", "origin",
                                               "availability", "metadata_status", "passages", "text_sha256")}
        if "extraction" in source:
            row["extraction"] = source["extraction"]
        if source.get("text_path"):
            row["full_text"] = (task_dir / CONTEXT_DIRECTORY / source["text_path"]).read_text(encoding="utf-8")
        rows.append(row)
    packet = {"retrieval_status": catalog["status"], "queries": catalog.get("queries", []),
              "retrieval_errors": catalog.get("retrieval_errors", []), "sources": rows}
    if len(json.dumps(packet, ensure_ascii=False)) > MAX_PACKET_CHARS:
        raise ValueError("Complete source packet exceeds 600,000 characters; split the research question/closest-work review rather than silently truncating methods")
    return packet


def _review_prompt_payload(prompt: str) -> dict[str, Any]:
    marker = "\n\nFROZEN INPUT DATA:\n"
    if marker not in prompt:
        raise ValueError("Review prompt has no bound source packet")
    payload = json.loads(prompt.split(marker, 1)[1])
    if not isinstance(payload, dict) or not isinstance(payload.get("literature"), dict):
        raise ValueError("Review prompt has an invalid source packet")
    return payload


def _source_packet_binding(packet: dict[str, Any]) -> dict[str, Any]:
    rows = packet.get("sources")
    if not isinstance(rows, list) or not all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows):
        raise ValueError("Review source packet requires identified sources")
    return {
        "source_packet_sha256": content_hash(packet),
        "source_coverage": {
            "policy": "all_prepared_sources_v1", "source_ids": [row["id"] for row in rows],
            "saved_text_source_ids": [row["id"] for row in rows if "full_text" in row],
            "full_text_source_ids": [row["id"] for row in rows if row.get("availability") == "full_text" and "full_text" in row],
        },
    }


def _call_source_binding(prompt: str, phase: str) -> dict[str, Any]:
    return _source_packet_binding(_review_prompt_payload(prompt)["literature"]) if phase == "novelty_review" else {}


def _source_tools_packet(prompt: str, phase: str) -> dict[str, Any]:
    if phase == "novelty_review":
        return _review_prompt_payload(prompt)["literature"]
    marker = "\n\nINPUT DATA:\n"
    if phase != "research_agenda" or marker not in prompt:
        raise ValueError("Source tools require a bound research prompt")
    payload, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
    if not isinstance(payload, dict) or not isinstance(payload.get("literature"), dict):
        raise ValueError("Author prompt has no complete source packet")
    return payload["literature"]


def normalize_review(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("verdict") not in {
        "differentiated", "needs_evidence", "overlap", "no_novelty_claim",
    }:
        raise ValueError("Invalid novelty reviewer verdict")
    if not isinstance(raw.get("rationale"), str) or not raw["rationale"].strip():
        raise ValueError("Novelty review requires a reasoned rationale")
    checks = raw.get("checks")
    if not isinstance(checks, dict) or any(checks.get(key) not in {"pass", "fail", "uncertain"} for key in REVIEW_CHECKS):
        raise ValueError("Novelty review requires explicit scientific checks")
    findings = raw.get("source_findings")
    if not isinstance(findings, list):
        raise ValueError("Novelty review requires source_findings")
    for row in findings:
        if not isinstance(row, dict) or not isinstance(row.get("source_id"), str) or not isinstance(row.get("assessment"), str) or not row["assessment"].strip():
            raise ValueError("Invalid source finding")
        if not isinstance(row.get("evidence"), list) or not all(isinstance(item, dict) for item in row["evidence"]):
            raise ValueError("Source finding evidence must contain citation objects")
    overlap = raw.get("overlap_assessment", {})
    if not isinstance(overlap, dict) or any(value is not None and type(value) is not bool for value in overlap.values()):
        raise ValueError("Overlap assessment must use booleans or null")
    if raw["verdict"] == "overlap" and not (
        overlap.get("core_problem_same") is True and overlap.get("key_method_same") is True
        and overlap.get("main_contribution_same") is True and overlap.get("remaining_difference_meaningful") is False
    ):
        raise ValueError("Overlap requires a complete core-problem/method/contribution collision assessment")
    actions = raw.get("next_actions", [])
    if not isinstance(actions, list) or not all(
        isinstance(row, dict) and row.get("kind") in ACTION_TYPES and isinstance(row.get("reason"), str)
        and row["reason"].strip() for row in actions
    ):
        raise ValueError("Invalid reviewer next_actions")
    return {**raw, "next_actions": actions, "overlap_assessment": overlap}


def _configuration_hash(cfg: LiteratureConfig, settings: dict[str, Any] | None) -> str:
    # Save only a digest: explicit agent environments can contain credentials.
    configuration = ({"backend": "model_search_stateless", "model_search": asdict(cfg.model_search),
                      "proxy_fallback": cfg.proxy_fallback} if settings is None else
                     {"backend": "agent_stateless", "settings": settings})
    return content_hash(configuration)


def _freeze_for_call(task_dir: Path, *, resume: bool) -> dict[str, Any]:
    try:
        return freeze_evidence(task_dir)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        if resume:
            raise InvalidReviewResume(f"Frozen evidence is invalid: {exc}") from exc
        raise


def _check_binding(receipt: dict[str, Any], cfg: LiteratureConfig,
                   settings: dict[str, Any] | None) -> None:
    backend = "model_search_stateless" if settings is None else "agent_stateless"
    if receipt.get("schema_version") != 2 or receipt.get("configuration_sha256") != _configuration_hash(cfg, settings):
        raise ValueError("Saved call configuration is missing or changed; use a new output directory")
    if receipt.get("backend") != backend or receipt.get("model") != (
        cfg.model_search.model if settings is None else settings.get("model")
    ) or (settings is None and receipt.get("provider") != cfg.model_search.provider):
        raise ValueError("Saved call provider, model, or backend changed")
    enabled = settings is not None and settings.get("source_lookup", False)
    if type(enabled) is not bool:
        raise ValueError("source_lookup must be boolean")
    if bool(enabled) != (receipt.get("source_lookup_requested") is True):
        raise ValueError("Saved source-tool configuration changed")
    if enabled and receipt.get("status") == "complete" and not isinstance(receipt.get("source_lookup"), dict):
        raise ValueError("Completed call is missing its source-tool evidence")


def _replay_call(directory: Path, receipt: dict[str, Any], phase: str,
                 frozen: dict[str, Any], *, prompt: str | None = None,
                 expected_source_packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify saved bytes, then derive the result again from the raw response."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Invalid review directory")
    if not isinstance(receipt, dict):
        raise ValueError("Invalid call receipt")
    if receipt.get("status") != "complete" or receipt.get("backend") not in {
        "model_search_stateless", "agent_stateless",
    } or receipt.get("phase") != phase:
        raise ValueError("Review did not complete through the configured stateless provider")
    if not isinstance(receipt.get("session_id"), str) or not receipt["session_id"]:
        raise ValueError("Missing independent review session")
    if receipt.get("frozen_evidence") != frozen:
        raise ValueError("Review is stale: source catalog or agenda changed")
    if prompt is not None and hashlib.sha256(prompt.encode("utf-8")).hexdigest() != receipt.get("prompt_sha256"):
        raise ValueError("Saved call prompt changed")
    response_path = directory / str(receipt.get("response", ""))
    if response_path.is_symlink() or not response_path.resolve().is_relative_to(directory.resolve()) or not response_path.is_file():
        raise ValueError("Invalid provider response path")
    if any(part.is_symlink() for part in response_path.parents if part != directory.parent):
        raise ValueError("Provider response path crosses a symlink")
    for path, expected_hash in ((directory / "prompt.txt", receipt.get("prompt_sha256")),
                                (directory / "assessment.json", receipt.get("assessment_sha256")),
                                (response_path, receipt.get("response_sha256"))):
        if path.is_symlink() or not path.is_file() or _sha(path) != expected_hash:
            raise ValueError(f"Review artifact changed: {path.name}")
    if phase == "novelty_review":
        prompt_path = directory / "prompt.txt"
        if prompt_path.stat().st_size > 1_800_000:
            raise ValueError("Review prompt exceeds its bound")
        saved_payload = _review_prompt_payload(prompt_path.read_text(encoding="utf-8"))
        _verify_prompt_campaign(saved_payload, frozen, author=False)
        binding = _source_packet_binding(saved_payload["literature"])
        if saved_payload.get("frozen_evidence") != frozen or content_hash(saved_payload.get("agenda")) != frozen.get("agenda_sha256"):
            raise ValueError("Review prompt does not cover its frozen evidence and agenda")
        if expected_source_packet is None and prompt is not None:
            expected_source_packet = _review_prompt_payload(prompt)["literature"]
        if expected_source_packet is not None and binding != _source_packet_binding(expected_source_packet):
            raise ValueError("Review source packet does not contain all prepared source texts")
        # Legacy receipts remain readable only when the actual saved prompt
        # proves full coverage. A claimed policy label alone is insufficient.
        if any(key in receipt for key in binding) and any(receipt.get(key) != value for key, value in binding.items()):
            raise ValueError("Review source packet receipt changed")
    elif (frozen.get("host_context_sha256") or frozen.get("seed_agenda_sha256")
          or frozen.get("review_feedback_sha256")):
        saved_prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
        marker = "\n\nINPUT DATA:\n"
        if marker not in saved_prompt:
            raise ValueError("Author prompt has no bound campaign input")
        payload, _ = json.JSONDecoder().raw_decode(saved_prompt.split(marker, 1)[1])
        if not isinstance(payload, dict) or payload.get("frozen_evidence") != frozen:
            raise ValueError("Author prompt differs from its frozen evidence")
        _verify_prompt_campaign(payload, frozen, author=True)
    provider_response = _read(response_path)
    if not isinstance(provider_response, dict):
        raise ValueError("Invalid provider response")
    normalize = normalize_review if phase == "novelty_review" else normalize_agenda
    if receipt["backend"] == "agent_stateless":
        from src.review_backend import strict_json_object
        transport = _read(directory / "transport/receipt.json")
        if not isinstance(transport, dict):
            raise ValueError("Invalid agent transport receipt")
        if receipt.get("source_lookup_requested") is True:
            from .agent_source_tools import verify_source_tools
            if receipt.get("source_lookup") != transport.get("source_lookup"):
                raise ValueError("Agent source-tool receipt differs from its transport")
            saved_prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
            complete_packet = _source_tools_packet(saved_prompt, phase)
            verify_source_tools(directory / "transport", receipt["source_lookup"],
                                expected_packet=complete_packet)
            expected_binding = _source_packet_binding(complete_packet)
            if any(receipt.get(key) != value for key, value in expected_binding.items()):
                raise ValueError("Source-tool packet differs from its call source binding")
        native_session = transport.get("native_session_id")
        if (_sha(directory / "transport/receipt.json") != receipt.get("transport_receipt_sha256")
                or transport.get("status") != "completed" or transport.get("independent_context") is not True
                or transport.get("prompt_sha256") != receipt["prompt_sha256"]
                or transport.get("stdout_sha256") != receipt["response_sha256"]
                or (receipt.get("schema_version") == 2 and (not isinstance(native_session, str) or not native_session))
                or native_session != receipt.get("native_session_id")
                or transport.get("observed_provider") != receipt.get("provider")
                or ("model" in transport and transport["model"] != receipt.get("model"))):
            raise ValueError("Agent transport receipt is stale or incomplete")
        if provider_response.get("protocolVersion") != 1 or provider_response.get("type") != "result" or provider_response.get("exitCode") != 0:
            raise ValueError("Agent response did not complete")
        data = provider_response["data"]
        if not isinstance(data, dict):
            raise ValueError("Agent response has no result data")
        if "nativeSessionId" in data and data["nativeSessionId"] != native_session:
            raise ValueError("Agent response native session differs from its transport receipt")
        for name, observed in (("model", "observed_model"), ("provider", "observed_provider")):
            if name in data and data[name] != transport.get(observed):
                raise ValueError("Agent response provider/model differs from its transport receipt")
        replay = normalize(strict_json_object(data["finalMessage"]))
    else:
        replay = normalize(parse_object(response_text(provider_response), "verdict" if phase == "novelty_review" else "schema_version"))
    assessment = normalize(_read(directory / "assessment.json"))
    if replay != assessment:
        raise ValueError("Assessment differs from the actual saved provider response")
    return replay


def _existing_call(directory: Path, cfg: LiteratureConfig, phase: str, frozen: dict[str, Any],
                   settings: dict[str, Any] | None, *, prompt: str | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Invalid review directory")
        receipt_path = directory / "receipt.json"
        # A crash can occur between mkdir and the first atomic receipt write.
        if not receipt_path.exists() and not receipt_path.is_symlink():
            if any(path.name != "prompt.txt" for path in directory.iterdir()):
                raise ValueError("Saved call artifacts have no receipt")
            return {}, None
        receipt = _read(receipt_path)
        if not isinstance(receipt, dict):
            raise ValueError("Invalid call receipt")
        _check_binding(receipt, cfg, settings)
        if receipt.get("status") == "complete":
            return receipt, _replay_call(directory, receipt, phase, frozen, prompt=prompt)
        if receipt.get("status") not in {"started", "failed"} or any(
            name in receipt for name in ("response_sha256", "assessment_sha256", "transport_receipt_sha256")
        ):
            raise ValueError("Invalid incomplete call receipt")
        if receipt.get("phase") != phase or receipt.get("frozen_evidence") != frozen:
            raise ValueError("Incomplete call inputs changed")
        prompt_path = directory / "prompt.txt"
        if prompt_path.is_symlink() or not prompt_path.is_file() or _sha(prompt_path) != receipt.get("prompt_sha256"):
            raise ValueError("Incomplete call prompt artifact changed")
        if prompt is not None and hashlib.sha256(prompt.encode("utf-8")).hexdigest() != receipt.get("prompt_sha256"):
            raise ValueError("Incomplete call prompt changed")
        return receipt, None
    except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        raise InvalidReviewResume(f"{directory.name}: {exc}") from exc


def _preflight_resume(directories: list[Path], cfg: LiteratureConfig, phase: str,
                      frozen: dict[str, Any], settings: dict[str, Any] | None, *,
                      prompts: dict[str, str] | None = None) -> None:
    sessions: set[str] = set()
    native_sessions: set[str] = set()
    incomplete_agenda = False
    for directory in directories:
        receipt, result = _existing_call(directory, cfg, phase, frozen, settings,
                                          prompt=(prompts or {}).get(directory.name))
        if result is None:
            incomplete_agenda = phase == "research_agenda"
            continue
        if incomplete_agenda:
            raise InvalidReviewResume("A completed agenda round follows a missing prior result")
        session, native = receipt["session_id"], receipt.get("native_session_id")
        if session in sessions or (native and native in native_sessions):
            raise InvalidReviewResume("Duplicate independent review session in saved calls")
        sessions.add(session)
        if native:
            native_sessions.add(native)


def _model_call(prompt: str, cfg: LiteratureConfig, directory: Path, phase: str,
                frozen: dict[str, Any], reviewer_settings: dict[str, Any] | None = None, *,
                resume: bool = False, cancel_event: threading.Event | None = None,
                deadline_unix: float | None = None) -> dict[str, Any]:
    if deadline_unix is not None and (type(deadline_unix) not in (int, float)
            or not math.isfinite(deadline_unix) or deadline_unix <= 0):
        raise ValueError("Research model deadline must be finite positive Unix seconds")
    existing = resume and (directory.exists() or directory.is_symlink())
    if existing:
        _receipt, replay = _existing_call(directory, cfg, phase, frozen, reviewer_settings, prompt=prompt)
        if replay is not None:
            return replay
    remaining = int(deadline_unix - time.time()) if deadline_unix is not None else None
    if remaining is not None and remaining < 1:
        raise ValueError("Research model call deadline exhausted")
    if existing:
        # Preserve every byte of an interrupted attempt before creating a fresh session.
        archive = directory.parent / "retry_attempts" / directory.name
        if any(path.is_symlink() or (path.exists() and not path.is_dir())
               for path in (archive.parent, archive)):
            raise InvalidReviewResume("Invalid retry archive directory")
        archive.mkdir(parents=True, exist_ok=True)
        directory.rename(archive / str(uuid.uuid4()))
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Independent novelty review cancelled")
    if reviewer_settings is not None:
        execution_settings = dict(reviewer_settings)
        if remaining is not None:
            execution_settings["timeout_seconds"] = min(execution_settings.get("timeout_seconds", 1800), remaining)
        return _agent_call(prompt, directory, phase, frozen, execution_settings,
                           configuration_sha256=_configuration_hash(cfg, reviewer_settings),
                           cancel_event=cancel_event)
    session = SearchSession(cfg, directory, **({"deadline_unix": deadline_unix} if deadline_unix is not None else {}))
    write_private_text(directory / "prompt.txt", prompt)
    receipt: dict[str, Any] = {
        "schema_version": 2, "session_id": str(uuid.uuid4()), "backend": "model_search_stateless",
        "configuration_sha256": _configuration_hash(cfg, None),
        "provider": cfg.model_search.provider, "model": cfg.model_search.model,
        "frozen_evidence": frozen, "prompt_sha256": _sha(directory / "prompt.txt"),
        "status": "started", "phase": phase, **_call_source_binding(prompt, phase),
    }
    _dump(directory / "receipt.json", receipt)
    try:
        text, _ = session.ask(prompt, phase, native=False)
        parsed = parse_object(text, "verdict" if phase == "novelty_review" else "schema_version")
        result = normalize_review(parsed) if phase == "novelty_review" else normalize_agenda(parsed)
        responses = [row["response"] for row in session.calls if row.get("response")]
        if not responses:
            raise ValueError("Provider call has no saved response receipt")
        response = responses[-1]
        _dump(directory / "assessment.json", result)
        completed = {**receipt, "status": "complete", "response": response,
                     "response_sha256": _sha(directory / response),
                     "assessment_sha256": _sha(directory / "assessment.json"), "calls": session.calls}
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc) or type(exc).__name__, calls=session.calls)
        _dump(directory / "receipt.json", receipt)
        raise
    # Atomic publication is the seal. An interruption before it leaves the
    # started receipt; an interruption after it must not downgrade completion.
    _dump(directory / "receipt.json", completed)
    return result


def _agent_call(prompt: str, directory: Path, phase: str, frozen: dict[str, Any],
                settings: dict[str, Any], *, configuration_sha256: str,
                cancel_event: threading.Event | None = None) -> dict[str, Any]:
    from src.review_backend import run_fresh_agent_json

    allowed = {"agent", "model", "profile", "reasoning_effort", "timeout_seconds", "environment", "source_lookup"}
    if set(settings) - allowed:
        raise ValueError("Unknown independent-review agent setting")
    enabled = settings.get("source_lookup", False)
    if type(enabled) is not bool:
        raise ValueError("source_lookup must be boolean")
    execution_settings = {key: value for key, value in settings.items() if key != "source_lookup"}
    lookup_packet = _source_tools_packet(prompt, phase) if enabled else None
    directory.mkdir()
    write_private_text(directory / "prompt.txt", prompt)
    receipt: dict[str, Any] = {
        "schema_version": 2, "session_id": str(uuid.uuid4()), "backend": "agent_stateless",
        "configuration_sha256": configuration_sha256,
        "model": settings.get("model"), "frozen_evidence": frozen,
        "prompt_sha256": _sha(directory / "prompt.txt"), "status": "started", "phase": phase,
        **_call_source_binding(prompt, phase),
    }
    if enabled:
        receipt.update(source_lookup_requested=True, **_source_packet_binding(lookup_packet))
    _dump(directory / "receipt.json", receipt)
    try:
        lifecycle: dict[str, Any] = {"cancel_event": cancel_event} if cancel_event is not None else {}
        if enabled:
            lifecycle["source_lookup_packet"] = lookup_packet
        parsed, transport = run_fresh_agent_json(prompt, work_dir=directory / "transport", **execution_settings, **lifecycle)
        if transport.get("status") != "completed" or transport.get("independent_context") is not True:
            raise ValueError("Reviewer did not complete in an independent agent context")
        result = normalize_review(parsed) if phase == "novelty_review" else normalize_agenda(parsed)
        _dump(directory / "assessment.json", result)
        completed = {**receipt, "status": "complete", "response": "transport/stdout.json",
                     "response_sha256": _sha(directory / "transport/stdout.json"),
                     "assessment_sha256": _sha(directory / "assessment.json"),
                     "transport_receipt_sha256": _sha(directory / "transport/receipt.json"),
                     "provider": transport.get("observed_provider"),
                     "native_session_id": transport.get("native_session_id")}
        if enabled:
            from .agent_source_tools import verify_source_tools
            source_receipt = transport.get("source_lookup")
            verify_source_tools(directory / "transport", source_receipt, expected_packet=lookup_packet)
            completed["source_lookup"] = source_receipt
    except BaseException as exc:
        receipt.update(status="failed", error=str(exc) or type(exc).__name__)
        _dump(directory / "receipt.json", receipt)
        raise
    _dump(directory / "receipt.json", completed)
    return result


def draft_research_agenda(task_dir: Path, cfg: LiteratureConfig, output: Path, *,
                          research_basis: dict[str, Any] | None = None,
                          reviewer_settings: dict[str, Any] | None = None,
                          candidate_packet: dict[str, Any] | None = None,
                          num_reflections: int = 3, resume: bool = False,
                          host_context: dict[str, Any] | None = None,
                          seed_agenda: dict[str, Any] | None = None,
                          review_feedback: dict[str, Any] | None = None,
                          deadline_unix: float | None = None) -> dict[str, Any]:
    """V1's bounded author-reflection loop; never independent novelty evidence."""
    from .upstream.v1_reflection import refine_agenda

    if type(num_reflections) is not int or not 1 <= num_reflections <= 5:
        raise ValueError("Use one to five total agenda reflection rounds")
    output.mkdir(parents=True, exist_ok=True)
    task_path = task_dir / "task.md"
    task = task_path.read_text(encoding="utf-8") if task_path.is_file() else ""
    frozen = _freeze_for_call(task_dir, resume=resume)
    if candidate_packet is not None:
        validate_candidate_packet(candidate_packet)
        frozen["candidate_packet_sha256"] = content_hash(candidate_packet)
    frozen.update(campaign_binding(host_context))
    if seed_agenda is not None:
        from .research_agenda import normalize_seed_agenda
        seed_agenda = normalize_seed_agenda(seed_agenda)
        frozen["seed_agenda_sha256"] = content_hash(seed_agenda)
    if review_feedback is not None:
        if (not isinstance(review_feedback, dict) or review_feedback.get("schema_version") != 1
                or review_feedback.get("scope") not in {"initial_agenda", "selected_interpretation"}
                or not isinstance(review_feedback.get("parent_review_sha256"), str)
                or len(review_feedback["parent_review_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in review_feedback["parent_review_sha256"])
                or len(json.dumps(review_feedback, ensure_ascii=False, allow_nan=False)) > 500_000):
            raise ValueError("Review feedback requires a bounded, attributed agenda or interpretation scope")
        if review_feedback["scope"] == "selected_interpretation" and candidate_packet is None:
            raise ValueError("Selected interpretation feedback requires its unchanged candidate packet")
        frozen["review_feedback_sha256"] = content_hash(review_feedback)
    review_started = any(output.glob("reviewer_*/receipt.json"))
    if resume and host_context is not None and review_started and not (output / "campaign_plan.json").is_file():
        raise InvalidReviewResume("Independent review started but its frozen campaign plan is missing")
    if resume:
        directories = sorted(path for path in output.glob("agenda_*") if path.name != "agenda_reflection_trace.json")
        ordered = ["agenda_draft", *(f"agenda_reflection_{index}" for index in range(2, num_reflections + 1))]
        if [path.name for path in directories] != ordered[:len(directories)]:
            raise InvalidReviewResume("Saved agenda rounds differ from this invocation")
        _preflight_resume(directories, cfg, "research_agenda", frozen, reviewer_settings)
    if host_context is not None:
        _pin_input(output / "host_context.json", host_context, resume=resume)
        if resume and ((output / "campaign_plan.json").exists() or (output / "campaign_plan.json").is_symlink()):
            try:
                campaign_binding(host_context, _read(output / "campaign_plan.json"),
                                 normalize_agenda(_read(output / "research_agenda.json")))
            except (ValueError, OSError, TypeError, KeyError) as exc:
                raise InvalidReviewResume(f"Saved campaign plan changed: {exc}") from exc
    if seed_agenda is not None:
        _pin_input(output / "research_seed_agenda.json", seed_agenda, resume=resume)
    if review_feedback is not None:
        _pin_input(output / "review_feedback.json", review_feedback, resume=resume)
    if candidate_packet is not None:
        _dump(output / "candidate_packet.json", candidate_packet)
    guidance = render_methodology(task_dir, stage="agenda")
    whole_idea_guidance = render_whole_idea_guidance(task_dir, stage="agenda")
    if whole_idea_guidance:
        guidance += "\n\n" + whole_idea_guidance
    if host_context is not None:
        guidance += "\n\n" + _campaign_author_guidance()
    if review_feedback is not None:
        guidance += ("\n\nATTRIBUTED REVIEW FEEDBACK:\n"
                     "The review_feedback is evidence and criticism from the previous frozen cycle. "
                     "Use the new full texts to address its requests. Preserve unresolved concerns. "
                     "Do not assume a citation request, model description, or exhausted budget proves novelty. "
                     "Do not seek a passing verdict. Calibrate or withdraw claims when the evidence requires it.")
        if review_feedback["scope"] == "selected_interpretation":
            guidance += ("\nThis is interpretation of an already selected implementation. "
                         "Its code, data, metrics, configuration, comparisons, and execution plan are fixed. "
                         "Revise only attribution, explanation, and contribution scope. "
                         "List needed experiments as unresolved work, never as completed results.")
    inputs = {
        "task": task, "selected_candidate_research_basis": research_basis,
        "selected_candidate_packet": candidate_packet, "frozen_evidence": frozen,
        "literature": source_packet(task_dir),
    }
    if host_context is not None:
        inputs["host_context"] = host_context
    if seed_agenda is not None:
        inputs["seed_agenda"] = seed_agenda
    if review_feedback is not None:
        inputs["review_feedback"] = review_feedback
    prompt = (PROMPTS / "research_agenda.md").read_text(encoding="utf-8") + ("\n\n" + guidance if guidance else "") + "\n\nINPUT DATA:\n" + json.dumps(inputs, ensure_ascii=False)
    if len(prompt.encode("utf-8")) > 1800_000:
        raise ValueError("Agenda prompt exceeds 1.8 MB; split the study rather than truncating source or implementation evidence")
    deadline: dict[str, Any] = {"deadline_unix": deadline_unix} if deadline_unix is not None else {}
    agenda = _model_call(prompt, cfg, output / "agenda_draft", "research_agenda", frozen, reviewer_settings, resume=resume, **deadline)
    def plan_errors(proposed: dict[str, Any]) -> list[dict[str, Any]]:
        from .campaign_plan import compile_campaign_plan, PlanValidationError
        if host_context is None:
            return []
        try:
            compile_campaign_plan(proposed, host_context)
        except PlanValidationError as exc:
            return exc.errors
        return []
    try:
        validation: dict[str, Any] = {"validation_errors": plan_errors} if host_context is not None else {}
        reflection = refine_agenda(
            agenda, frozen_context=prompt, normalize=normalize_agenda,
            num_reflections=num_reflections,
            request=lambda reflection_prompt, round_index: _model_call(
                reflection_prompt, cfg, output / f"agenda_reflection_{round_index}",
                "research_agenda", frozen, reviewer_settings, resume=resume, **deadline,
            ),
            **validation,
        )
    except InvalidReviewResume:
        raise
    except (ValueError, OSError, TypeError, KeyError) as exc:
        _dump(output / "agenda_reflection_trace.json", {
            "schema_version": 1, "upstream": "AI-Scientist-v1", "status": "failed",
            "author_proposal_only": True, "independent_novelty_evidence": False,
            "frozen_evidence": frozen, "initial_agenda_sha256": content_hash(agenda),
            "num_reflections": num_reflections, "error": str(exc),
        })
        raise
    _dump(output / "agenda_reflection_trace.json", {
        **reflection.to_dict(), "frozen_evidence": frozen,
    })
    agenda = reflection.agenda
    _dump(output / "research_agenda.json", agenda)
    if host_context is not None:
        from .campaign_plan import compile_campaign_plan, PlanValidationError
        try:
            plan = compile_campaign_plan(agenda, host_context)
        except PlanValidationError as exc:
            _dump(output / "plan_validation.json", {
                "schema_version": 1, "status": "invalid", "errors": exc.errors,
                "host_context_sha256": frozen["host_context_sha256"],
                "agenda_sha256": content_hash(agenda),
                "independent_review_started": False, "execution_started": False,
            })
            raise
        _pin_input(output / "campaign_plan.json", plan, resume=resume, required_on_resume=review_started)
        _dump(output / "plan_validation.json", {
            "schema_version": 1, "status": "valid", "errors": [],
            "host_context_sha256": frozen["host_context_sha256"], "plan_sha256": plan["plan_sha256"],
            "agenda_sha256": content_hash(agenda),
        })
    return agenda


def review_novelty(task_dir: Path, agenda: dict[str, Any], cfg: LiteratureConfig,
                   output: Path, *, reviewer_count: int = 2,
                   reviewer_settings: dict[str, Any] | None = None,
                   candidate_packet: dict[str, Any] | None = None,
                   resume: bool = False,
                   cancel_event: threading.Event | None = None,
                   host_context: dict[str, Any] | None = None,
                   compiled_plan: dict[str, Any] | None = None,
                   deadline_unix: float | None = None) -> dict[str, Any]:
    """Run independent reviewers together, after validating all shared inputs.

    Agent calls stop their owned process trees before cancellation returns.
    Non-main-thread callers can cancel with ``cancel_event``; this function does
    not install signal handlers there. HTTP calls retain their transport timeout
    and are joined before returning. Failed reviews remain evidence gaps.
    """
    if type(reviewer_count) is not int or not 2 <= reviewer_count <= 4:
        raise ValueError("Use two to four independent novelty review contexts")
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.glob("reviewer_*")):
        raise ValueError("Novelty review output already contains review receipts; use a new cycle directory")
    agenda = normalize_agenda(agenda)
    frozen = {**_freeze_for_call(task_dir, resume=resume), "agenda_sha256": content_hash(agenda)}
    try:
        frozen.update(campaign_binding(host_context, compiled_plan, agenda))
        if host_context is not None and compiled_plan is None:
            raise ValueError("Independent campaign review requires a compiled plan")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        if resume:
            raise InvalidReviewResume(f"Frozen campaign inputs are invalid: {exc}") from exc
        raise
    if candidate_packet is not None:
        validate_candidate_packet(candidate_packet)
        frozen["candidate_packet_sha256"] = content_hash(candidate_packet)
    packet = source_packet(task_dir, agenda)
    template = (PROMPTS / "novelty_review.md").read_text(encoding="utf-8")
    guidance = render_methodology(task_dir, stage="novelty", agenda=agenda)
    whole_idea_guidance = render_whole_idea_guidance(task_dir, stage="novelty")
    if whole_idea_guidance:
        guidance += "\n\n" + whole_idea_guidance
    if guidance:
        template += "\n\n" + guidance
    if host_context is not None:
        template += ("\n\nReview the supplied compiled_plan against host_context and agenda. "
                     "The seed is not_evaluated. Check the complete root ideas, matched controls, "
                     "executable interfaces, fixed allocation, promotion limits, and separate "
                     "confirmation scope. The host compilation verifies schema and budget; "
                     "it does not establish scientific quality or future success.")
    prompts = {}
    for index in range(reviewer_count):
        focus = ("Examine the strongest constructive case for a meaningful incremental contribution, then try to falsify it."
                 if index % 2 == 0 else
                 "Examine the strongest prior-art collision and alternative explanation, then identify any defensible incremental path.")
        inputs = {
            "agenda": agenda, "literature": packet, "frozen_evidence": frozen,
            "selected_candidate_packet": candidate_packet,
        }
        if host_context is not None:
            inputs.update(host_context=host_context, compiled_plan=compiled_plan)
        prompt = template + "\n\nREVIEW FOCUS:\n" + focus + "\n\nFROZEN INPUT DATA:\n" + json.dumps(inputs, ensure_ascii=False)
        if len(prompt.encode("utf-8")) > 1800_000:
            raise ValueError("Novelty prompt exceeds 1.8 MB; split the study rather than truncating source or implementation evidence")
        prompts[f"reviewer_{index + 1}"] = prompt
    pending = dict(prompts)
    retries: list[Path] = []
    if resume:
        directories = sorted(output.glob("reviewer_*"))
        allowed = {f"reviewer_{index}" for index in range(1, reviewer_count + 1)}
        if any(path.name not in allowed for path in directories):
            raise InvalidReviewResume("Saved reviewer count differs from this invocation")
        _preflight_resume(directories, cfg, "novelty_review", frozen, reviewer_settings, prompts=prompts)
        for directory in directories:
            _receipt, result = _existing_call(directory, cfg, "novelty_review", frozen,
                                               reviewer_settings, prompt=prompts[directory.name])
            if result is not None:
                pending.pop(directory.name)
                logger.info("Novelty %s cache_reused", directory.name)
                continue
            archive = output / "retry_attempts" / directory.name
            if any(path.is_symlink() or (path.exists() and not path.is_dir())
                   for path in (archive.parent, archive)):
                raise InvalidReviewResume("Invalid retry archive directory")
            retries.append(directory)
    if host_context is not None:
        assert compiled_plan is not None  # Validated before building any reviewer prompt.
        _pin_input(output / "host_context.json", host_context, resume=resume)
        _pin_input(output / "campaign_plan.json", compiled_plan, resume=resume)
    cancellation = cancel_event if cancel_event is not None else threading.Event()
    if cancellation.is_set():
        raise CancelledError("Independent novelty review cancelled")
    if candidate_packet is not None:
        _dump(output / "candidate_packet.json", candidate_packet)
    _dump(output / "research_agenda.json", agenda)
    # Archive only after the entire resume preflight succeeds. Each worker then
    # owns one new directory; no worker mutates a shared retry parent or input.
    for directory in retries:
        archive = output / "retry_attempts" / directory.name
        archive.mkdir(parents=True, exist_ok=True)
        directory.rename(archive / str(uuid.uuid4()))
    if pending:
        executor = None
        previous_handlers = {}
        interrupted_signal = None
        failure = None

        def interrupt(signum: int, _frame: object) -> None:
            nonlocal interrupted_signal
            if interrupted_signal is None:
                interrupted_signal = signum

        try:
            if threading.current_thread() is threading.main_thread():
                for signum in (signal.SIGTERM, signal.SIGINT):
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, interrupt)
            executor = ThreadPoolExecutor(max_workers=reviewer_count, thread_name_prefix="novelty-review")
            futures = {}
            for name, prompt in pending.items():
                if interrupted_signal is not None:
                    cancellation.set()
                if cancellation.is_set():
                    break
                logger.info("Novelty %s start", name)
                deadline: dict[str, Any] = {"deadline_unix": deadline_unix} if deadline_unix is not None else {}
                future = executor.submit(_model_call, prompt, cfg, output / name, "novelty_review",
                                         frozen, reviewer_settings, cancel_event=cancellation, **deadline)
                futures[future] = name
            remaining = set(futures)
            while remaining:
                if interrupted_signal is not None:
                    cancellation.set()
                done, remaining = wait(remaining, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except InvalidReviewResume:
                        raise
                    except (ValueError, OSError, TypeError, KeyError):
                        logger.info("Novelty %s failed", futures[future])
                        continue
                    logger.info("Novelty %s completed", futures[future])
        except BaseException as exc:
            failure = exc
            cancellation.set()
        finally:
            # Signal handlers only record a flag, so they cannot jump out of
            # thread registration, reenter Event.set's lock, or abort cleanup.
            try:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        if interrupted_signal is not None:
            cancellation.set()
            if interrupted_signal == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + interrupted_signal)
        if failure is not None:
            raise failure
        if cancellation.is_set():
            raise CancelledError("Independent novelty review cancelled")
    if resume:
        _preflight_resume(sorted(output.glob("reviewer_*")), cfg, "novelty_review", frozen, reviewer_settings)
    campaign_kwargs: dict[str, Any] = ({"host_context": host_context, "compiled_plan": compiled_plan}
                                      if host_context is not None else {})
    report = evaluate_novelty(task_dir, agenda, review_dir=output,
                              expected_catalog_sha256=frozen["catalog_sha256"], candidate_packet=candidate_packet,
                              **campaign_kwargs)
    _dump(output / "novelty_report.json", report)
    return report


def load_review_records(task_dir: Path, agenda: dict[str, Any], output: Path, *,
                        candidate_packet: dict[str, Any] | None = None,
                        host_context: dict[str, Any] | None = None,
                        compiled_plan: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Revalidate all bound bytes and replay parsing of the actual provider response."""
    verified: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        expected = {**freeze_evidence(task_dir), "agenda_sha256": content_hash(agenda)}
        saved_context, saved_plan = saved_campaign_inputs(output)
        if host_context is not None and (saved_context != host_context or saved_plan != compiled_plan):
            raise ValueError("Saved campaign context or plan changed after review")
        host_context = host_context if host_context is not None else saved_context
        compiled_plan = compiled_plan if compiled_plan is not None else saved_plan
        expected.update(campaign_binding(host_context, compiled_plan, agenda))
        if host_context is not None:
            from .campaign_plan import HOST_CONTEXT_PATH, CAMPAIGN_PLAN_PATH, load_campaign_plan
            if any((task_dir / path).exists() or (task_dir / path).is_symlink()
                   for path in (HOST_CONTEXT_PATH, CAMPAIGN_PLAN_PATH)):
                published = load_campaign_plan(task_dir, required=True)
                if published != compiled_plan:
                    raise ValueError("Published campaign plan differs from the reviewed plan")
        complete_sources = source_packet(task_dir, agenda)
        if candidate_packet is not None:
            validate_candidate_packet(candidate_packet)
            expected["candidate_packet_sha256"] = content_hash(candidate_packet)
            saved_packet = _read(output / "candidate_packet.json")
            if content_hash(saved_packet) != expected["candidate_packet_sha256"]:
                raise ValueError("Candidate evidence changed: saved source/result packet differs from the current selected candidate")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return [], [str(exc)]
    seen = set()
    native_sessions = set()
    for directory in sorted(output.glob("reviewer_*")):
        try:
            receipt = _read(directory / "receipt.json")
            assessment = _replay_call(directory, receipt, "novelty_review", expected, expected_source_packet=complete_sources)
            session_id = receipt["session_id"]
            if session_id in seen:
                raise ValueError("Duplicate independent review session")
            seen.add(session_id)
            native_session = receipt.get("native_session_id")
            if native_session and native_session in native_sessions:
                raise ValueError("Duplicate native reviewer session")
            if native_session:
                native_sessions.add(native_session)
            verified.append({"assessment": assessment, "receipt": {
                "path": str(directory / "receipt.json"), "sha256": _sha(directory / "receipt.json"),
                "session_id": session_id, "provider": receipt.get("provider"), "model": receipt.get("model"),
                "backend": receipt["backend"], "response_sha256": receipt["response_sha256"],
                **_source_packet_binding(complete_sources),
            }})
        except (ValueError, OSError, TypeError, KeyError) as exc:
            errors.append(f"{directory.name}: {exc}")
    return verified, errors


def load_frozen_novelty_report(task_dir: Path, output: Path, *,
                              candidate_packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recheck saved inputs, not the saved report's status, at a delivery boundary."""
    saved = _read(output / "novelty_report.json")
    agenda = normalize_agenda(_read(output / "research_agenda.json"))
    frozen = saved.get("frozen_evidence", {})
    report = evaluate_novelty(task_dir, agenda, review_dir=output,
                              expected_catalog_sha256=frozen.get("catalog_sha256"), candidate_packet=candidate_packet)
    if frozen.get("candidate_packet_sha256") and candidate_packet is None:
        report["status"] = "needs_evidence"
        report["gaps"].append({"code": "candidate_packet_required", "detail": "Rebuild the current selected source/evaluation packet before accepting this candidate-bound review."})
    elif candidate_packet is not None and frozen.get("candidate_packet_sha256") != content_hash(candidate_packet):
        report["status"] = "needs_evidence"
        report["gaps"].append({"code": "candidate_packet_changed", "detail": "Selected implementation or observed evaluation evidence changed after novelty review."})
        report["next_actions"].append({"kind": "independent_review", "reason": "Review the actual selected implementation and current verified outcomes."})
    if frozen.get("agenda_sha256") != content_hash(agenda):
        report["status"] = "needs_evidence"
        report["gaps"].append({"code": "agenda_changed", "detail": "Saved agenda differs from the frozen report."})
        report["next_actions"].append({"kind": "independent_review", "reason": "Review the changed agenda against the current frozen sources."})
    return report


def load_completed_novelty_report(task_dir: Path, output: Path, cfg: LiteratureConfig, *,
                                  reviewer_count: int,
                                  reviewer_settings: dict[str, Any] | None = None,
                                  candidate_packet: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Reuse a complete bound review regardless of its scientific verdict.

    This is a resume decision, not approval of the claim. All evidence and
    provider responses are rechecked. Incomplete or stale calls need a new cycle;
    an unchanged negative assessment must not be rerolled into a positive one.
    """
    if type(reviewer_count) is not int or not 2 <= reviewer_count <= 4:
        raise ValueError("Selected review requires two to four reviewers")
    saved = _read(output / "novelty_report.json")
    report = load_frozen_novelty_report(task_dir, output, candidate_packet=candidate_packet)
    if (report.get("frozen_evidence") != saved.get("frozen_evidence")
            or report.get("machine_checks", {}).get("review_artifacts_valid") is not True
            or len(report.get("model_judgments", [])) != reviewer_count):
        return None
    # Missing scientific evidence is a valid review outcome. Missing bindings
    # or malformed implementation assessments are not completed candidate reviews.
    if any(row.get("code") in {
        "candidate_review_unbound", "candidate_checks_missing",
        "claim_code_evidence_missing", "claim_code_evidence_invalid",
    } for row in report.get("gaps", [])):
        return None
    directories = sorted(output.glob("reviewer_*"))
    if {path.name for path in directories} != {f"reviewer_{i}" for i in range(1, reviewer_count + 1)}:
        return None
    for directory in directories:
        _check_binding(_read(directory / "receipt.json"), cfg, reviewer_settings)
    return report
