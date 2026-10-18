"""Stateless provider reviews and replayable provenance for the novelty gate."""

from __future__ import annotations

import hashlib
import json
import uuid
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

PROMPTS = Path(__file__).resolve().parents[1] / "prompts" / "research_harness"
MAX_PACKET_CHARS = 600_000


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
    provider_response = _read(response_path)
    if not isinstance(provider_response, dict):
        raise ValueError("Invalid provider response")
    normalize = normalize_review if phase == "novelty_review" else normalize_agenda
    if receipt["backend"] == "agent_stateless":
        from src.review_backend import strict_json_object
        transport = _read(directory / "transport/receipt.json")
        if not isinstance(transport, dict):
            raise ValueError("Invalid agent transport receipt")
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
                resume: bool = False) -> dict[str, Any]:
    if resume and (directory.exists() or directory.is_symlink()):
        _receipt, replay = _existing_call(directory, cfg, phase, frozen, reviewer_settings, prompt=prompt)
        if replay is not None:
            return replay
        # Preserve every byte of an interrupted attempt before creating a fresh session.
        archive = directory.parent / "retry_attempts" / directory.name
        if any(path.is_symlink() or (path.exists() and not path.is_dir())
               for path in (archive.parent, archive)):
            raise InvalidReviewResume("Invalid retry archive directory")
        archive.mkdir(parents=True, exist_ok=True)
        directory.rename(archive / str(uuid.uuid4()))
    if reviewer_settings is not None:
        return _agent_call(prompt, directory, phase, frozen, reviewer_settings,
                           configuration_sha256=_configuration_hash(cfg, reviewer_settings))
    session = SearchSession(cfg, directory)
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
        receipt.update(status="complete", response=response, response_sha256=_sha(directory / response),
                       assessment_sha256=_sha(directory / "assessment.json"), calls=session.calls)
        _dump(directory / "receipt.json", receipt)
        return result
    except (ValueError, OSError, TypeError, KeyError) as exc:
        receipt.update(status="failed", error=str(exc), calls=session.calls)
        _dump(directory / "receipt.json", receipt)
        raise


def _agent_call(prompt: str, directory: Path, phase: str, frozen: dict[str, Any],
                settings: dict[str, Any], *, configuration_sha256: str) -> dict[str, Any]:
    from src.review_backend import run_fresh_agent_json

    allowed = {"agent", "model", "profile", "reasoning_effort", "timeout_seconds", "environment"}
    if set(settings) - allowed:
        raise ValueError("Unknown independent-review agent setting")
    directory.mkdir()
    write_private_text(directory / "prompt.txt", prompt)
    receipt: dict[str, Any] = {
        "schema_version": 2, "session_id": str(uuid.uuid4()), "backend": "agent_stateless",
        "configuration_sha256": configuration_sha256,
        "model": settings.get("model"), "frozen_evidence": frozen,
        "prompt_sha256": _sha(directory / "prompt.txt"), "status": "started", "phase": phase,
        **_call_source_binding(prompt, phase),
    }
    _dump(directory / "receipt.json", receipt)
    try:
        parsed, transport = run_fresh_agent_json(prompt, work_dir=directory / "transport", **settings)
        if transport.get("status") != "completed" or transport.get("independent_context") is not True:
            raise ValueError("Reviewer did not complete in an independent agent context")
        result = normalize_review(parsed) if phase == "novelty_review" else normalize_agenda(parsed)
        _dump(directory / "assessment.json", result)
        receipt.update(status="complete", response="transport/stdout.json",
                       response_sha256=_sha(directory / "transport/stdout.json"),
                       assessment_sha256=_sha(directory / "assessment.json"),
                       transport_receipt_sha256=_sha(directory / "transport/receipt.json"),
                       provider=transport.get("observed_provider"),
                       native_session_id=transport.get("native_session_id"))
        _dump(directory / "receipt.json", receipt)
        return result
    except (ValueError, OSError, TypeError, KeyError) as exc:
        receipt.update(status="failed", error=str(exc))
        _dump(directory / "receipt.json", receipt)
        raise


def draft_research_agenda(task_dir: Path, cfg: LiteratureConfig, output: Path, *,
                          research_basis: dict[str, Any] | None = None,
                          reviewer_settings: dict[str, Any] | None = None,
                          candidate_packet: dict[str, Any] | None = None,
                          num_reflections: int = 3, resume: bool = False) -> dict[str, Any]:
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
    if resume:
        directories = sorted(path for path in output.glob("agenda_*") if path.name != "agenda_reflection_trace.json")
        ordered = ["agenda_draft", *(f"agenda_reflection_{index}" for index in range(2, num_reflections + 1))]
        if [path.name for path in directories] != ordered[:len(directories)]:
            raise InvalidReviewResume("Saved agenda rounds differ from this invocation")
        _preflight_resume(directories, cfg, "research_agenda", frozen, reviewer_settings)
    if candidate_packet is not None:
        _dump(output / "candidate_packet.json", candidate_packet)
    prompt = (PROMPTS / "research_agenda.md").read_text(encoding="utf-8") + "\n\nINPUT DATA:\n" + json.dumps({
        "task": task, "selected_candidate_research_basis": research_basis,
        "selected_candidate_packet": candidate_packet, "frozen_evidence": frozen,
        "literature": source_packet(task_dir),
    }, ensure_ascii=False)
    if len(prompt.encode("utf-8")) > 1800_000:
        raise ValueError("Agenda prompt exceeds 1.8 MB; split the study rather than truncating source or implementation evidence")
    agenda = _model_call(prompt, cfg, output / "agenda_draft", "research_agenda", frozen, reviewer_settings, resume=resume)
    try:
        reflection = refine_agenda(
            agenda, frozen_context=prompt, normalize=normalize_agenda,
            num_reflections=num_reflections,
            request=lambda reflection_prompt, round_index: _model_call(
                reflection_prompt, cfg, output / f"agenda_reflection_{round_index}",
                "research_agenda", frozen, reviewer_settings, resume=resume,
            ),
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
    return agenda


def review_novelty(task_dir: Path, agenda: dict[str, Any], cfg: LiteratureConfig,
                   output: Path, *, reviewer_count: int = 2,
                   reviewer_settings: dict[str, Any] | None = None,
                   candidate_packet: dict[str, Any] | None = None,
                   resume: bool = False) -> dict[str, Any]:
    """Review in separate contexts; errors remain evidence gaps, never positive reviews."""
    if type(reviewer_count) is not int or not 2 <= reviewer_count <= 4:
        raise ValueError("Use two to four independent novelty review contexts")
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.glob("reviewer_*")):
        raise ValueError("Novelty review output already contains review receipts; use a new cycle directory")
    agenda = normalize_agenda(agenda)
    frozen = {**_freeze_for_call(task_dir, resume=resume), "agenda_sha256": content_hash(agenda)}
    if candidate_packet is not None:
        validate_candidate_packet(candidate_packet)
        frozen["candidate_packet_sha256"] = content_hash(candidate_packet)
    packet = source_packet(task_dir, agenda)
    template = (PROMPTS / "novelty_review.md").read_text(encoding="utf-8")
    prompts = {}
    for index in range(reviewer_count):
        focus = ("Examine the strongest constructive case for a meaningful incremental contribution, then try to falsify it."
                 if index % 2 == 0 else
                 "Examine the strongest prior-art collision and alternative explanation, then identify any defensible incremental path.")
        prompt = template + "\n\nREVIEW FOCUS:\n" + focus + "\n\nFROZEN INPUT DATA:\n" + json.dumps({
            "agenda": agenda, "literature": packet, "frozen_evidence": frozen,
            "selected_candidate_packet": candidate_packet,
        }, ensure_ascii=False)
        if len(prompt.encode("utf-8")) > 1800_000:
            raise ValueError("Novelty prompt exceeds 1.8 MB; split the study rather than truncating source or implementation evidence")
        prompts[f"reviewer_{index + 1}"] = prompt
    if resume:
        directories = sorted(output.glob("reviewer_*"))
        allowed = {f"reviewer_{index}" for index in range(1, reviewer_count + 1)}
        if any(path.name not in allowed for path in directories):
            raise InvalidReviewResume("Saved reviewer count differs from this invocation")
        _preflight_resume(directories, cfg, "novelty_review", frozen, reviewer_settings, prompts=prompts)
    if candidate_packet is not None:
        _dump(output / "candidate_packet.json", candidate_packet)
    _dump(output / "research_agenda.json", agenda)
    for name, prompt in prompts.items():
        try:
            _model_call(prompt, cfg, output / name, "novelty_review", frozen, reviewer_settings, resume=resume)
        except InvalidReviewResume:
            raise
        except (ValueError, OSError, TypeError, KeyError):
            continue
    if resume:
        _preflight_resume(sorted(output.glob("reviewer_*")), cfg, "novelty_review", frozen, reviewer_settings)
    report = evaluate_novelty(task_dir, agenda, review_dir=output,
                              expected_catalog_sha256=frozen["catalog_sha256"], candidate_packet=candidate_packet)
    _dump(output / "novelty_report.json", report)
    return report


def load_review_records(task_dir: Path, agenda: dict[str, Any], output: Path, *,
                        candidate_packet: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Revalidate all bound bytes and replay parsing of the actual provider response."""
    verified: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        expected = {**freeze_evidence(task_dir), "agenda_sha256": content_hash(agenda)}
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
