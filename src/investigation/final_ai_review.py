"""Run the standalone AI reviewer for a wrapper-produced final paper."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

from src.orchestrator.paper_manifest import paper_publication_lock
from src.pipeline.portable_publication import (
    portable_replacement_pending,
)
from src.pipeline.revision_publication import recover_incomplete_promotion_locked
from src.investigation.pinned_files import (
    absolute_path,
    atomic_write_at,
    copy_regular_at,
    manifest_lock_at,
    open_directory_tree_no_follow,
    open_relative_directory_no_follow,
    read_regular_at,
)
from src.investigation.review_summary import ReviewDecision, parse_review_summary as _review_summary
from src.review_inputs import (
    PAPER_PDF_MAX_BYTES as _PAPER_PDF_MAX_BYTES,
    REVIEW_EVIDENCE_MAX_BYTES as _REVIEW_EVIDENCE_MAX_BYTES,
    read_review_input as _read_review_input,
    review_input_record as _review_input_record,
)

ReviewStatus = Literal["completed", "failed"]
ExecuteCommand = Callable[[list[str], Path, dict[str, str]], tuple[str, int]]

_ENV_FILE_MAX_BYTES = 1024 * 1024
_PAPER_MANIFEST_MAX_BYTES = 1024 * 1024
_REVIEW_OUTPUT_MAX_BYTES = 4 * 1024 * 1024
_REVIEW_LOG_MAX_BYTES = 16 * 1024 * 1024
_OPENAI_ENVIRONMENT_KEYS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
}
_UV_INDEX_CREDENTIAL_PATTERN = re.compile(r"UV_INDEX_[A-Z0-9_]+_(?:USERNAME|PASSWORD)")
_ARTIFACT_STEM_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)*"
)
_MANIFEST_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}")

_ENVIRONMENT_ALLOWLIST = {
    "AGENTIC_COST_ROOT",
    "ALL_PROXY",
    "CURL_CA_BUNDLE",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "NETRC",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "UV_CACHE_DIR",
    "UV_CONFIG_FILE",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX",
    "UV_INDEX_STRATEGY",
    "UV_INDEX_URL",
    "UV_KEYRING_PROVIDER",
    "UV_OFFLINE",
    "UV_SYSTEM_CERTS",
    "XDG_CACHE_HOME",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


@dataclass(frozen=True)
class FinalAiReviewResult:
    """Observable outcome of the non-gating final review stage."""

    status: ReviewStatus
    review_path: Path | None
    log_path: Path
    error: str | None = None
    overall_score: int | float | None = None
    decision: ReviewDecision | None = None


def _read_regular_file_no_follow(path: Path, *, max_bytes: int, description: str) -> str:
    expanded_path = path.expanduser().absolute()
    directory_fd = os.open(
        expanded_path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        descriptor = os.open(
            expanded_path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"{description} is not regular: {expanded_path}")
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"{description} is too large: {expanded_path}")
    return content.decode("utf-8")


def _read_env_file_no_follow(path: Path) -> str:
    return _read_regular_file_no_follow(
        path,
        max_bytes=_ENV_FILE_MAX_BYTES,
        description="configured environment file",
    )


def _validate_review_input_receipt(
    provenance: dict[str, object], expected_inputs: list[dict[str, object]]
) -> None:
    """Bind every returned source identifier to the host's ordered input packet."""
    if provenance.get("paper_sha256") != expected_inputs[0]["sha256"]:
        raise ValueError("agent review is not bound to the reviewed PDF")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != len(expected_inputs):
        raise ValueError("agent review inputs do not match the configured evidence")
    for index, (actual, expected) in enumerate(zip(inputs, expected_inputs)):
        if not isinstance(actual, dict):
            raise ValueError("agent review has invalid evidence provenance")
        # The reviewer extracts the paper text. The host binds its PDF bytes and
        # checks a supplied text hash's format without claiming to re-extract it.
        allowed = set(expected) | ({"text_sha256"} if index == 0 else set())
        if (
            set(actual) - allowed
            or type(actual.get("bytes")) is not int
            or {key: actual.get(key) for key in expected} != expected
        ):
            raise ValueError("agent review inputs do not match the configured evidence")
        if "text_sha256" in actual and (
            not isinstance(actual["text_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", actual["text_sha256"])
        ):
            raise ValueError("agent review has an invalid paper text hash")


def read_paper_manifest(path: Path) -> dict[str, object]:
    """Read a bounded regular manifest without following the final path component."""
    try:
        manifest = json.loads(
            _read_regular_file_no_follow(
                path,
                max_bytes=_PAPER_MANIFEST_MAX_BYTES,
                description="paper manifest",
            )
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid paper manifest JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"paper manifest must contain a JSON object: {path}")
    return manifest


def _openai_environment_from_file(path: Path) -> dict[str, str]:
    selected: dict[str, str] = {}
    for raw_line in _read_env_file_no_follow(path).splitlines():
        assignment = raw_line.strip()
        if not assignment or assignment.startswith("#"):
            continue
        parts = assignment.split(None, 1)
        if parts[0] == "export":
            if len(parts) == 1:
                continue
            assignment = parts[1]
        key, separator, raw_value = assignment.partition("=")
        key = key.strip()
        if not separator or key not in _OPENAI_ENVIRONMENT_KEYS:
            continue
        selected[key] = _parse_env_value(raw_value, key=key)
    return selected


def _parse_env_value(raw_value: str, *, key: str) -> str:
    value = raw_value.strip()
    quote: str | None = None
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
        elif character == "#" and (index == 0 or value[index - 1].isspace()):
            value = value[:index].rstrip()
            break
    try:
        return " ".join(shlex.split(value, comments=False, posix=True))
    except ValueError as exc:
        raise ValueError(f"invalid quoted value for {key} in configured environment file") from exc


def final_ai_review_environment(
    environment: Mapping[str, str],
    *,
    env_file: Path | None = None,
) -> dict[str, str]:
    """Keep approved runtime values and load only OpenAI settings from an env file."""
    selected = {
        key: value
        for key, value in environment.items()
        if key in _ENVIRONMENT_ALLOWLIST or _UV_INDEX_CREDENTIAL_PATTERN.fullmatch(key)
    }
    if env_file is not None and not selected.get("OPENAI_API_KEY"):
        for key, value in _openai_environment_from_file(env_file).items():
            if not selected.get(key):
                selected[key] = value
    return selected


def _read_paper_manifest_at(paper_root_fd: int) -> dict[str, object]:
    try:
        manifest = json.loads(
            read_regular_at(
                paper_root_fd,
                "latest_manifest.json",
                max_bytes=_PAPER_MANIFEST_MAX_BYTES,
                description="paper manifest",
            ).decode("utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid paper manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("paper manifest must contain a JSON object")
    return manifest


def _publish_review_status(
    *,
    paper_root_fd: int,
    status: ReviewStatus,
    review_path: Path | None,
    log_path: Path,
    paper_pdf: Path,
    manifest_key: str,
    latest_review_path: Path,
    latest_log_path: Path,
) -> None:
    with manifest_lock_at(paper_root_fd):
        manifest = _read_paper_manifest_at(paper_root_fd)
        manifest_pdf = manifest.get("paper_pdf")
        if not isinstance(manifest_pdf, str) or absolute_path(Path(manifest_pdf)) != paper_pdf:
            raise ValueError("latest paper manifest no longer points to the reviewed PDF")
        manifest[manifest_key] = {
            "status": status,
            "review_json": str(absolute_path(review_path)) if review_path is not None else None,
            "log": str(absolute_path(log_path)),
        }
        manifest["ai_review"] = {
            "status": status,
            "review_json": (
                str(absolute_path(latest_review_path)) if review_path is not None else None
            ),
            "log": str(absolute_path(latest_log_path)),
        }
        atomic_write_at(
            paper_root_fd,
            "latest_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            mode=0o644,
        )


def _write_with_latest_alias(
    paper_dir_fd: int,
    *,
    artifact_name: str,
    latest_name: str,
    content: bytes,
    mode: int,
) -> None:
    atomic_write_at(paper_dir_fd, artifact_name, content, mode=mode)
    if artifact_name != latest_name:
        atomic_write_at(paper_dir_fd, latest_name, content, mode=mode)


def _clear_latest_review_alias(paper_dir_fd: int) -> None:
    try:
        os.unlink("ai_review.json", dir_fd=paper_dir_fd)
    except FileNotFoundError:
        pass


def _write_failure_log(
    paper_dir_fd: int,
    *,
    log_name: str,
    error: str,
    log_text: str | None = None,
) -> None:
    try:
        existing_log = (
            read_regular_at(
                paper_dir_fd,
                log_name,
                max_bytes=_REVIEW_LOG_MAX_BYTES,
            )
            .decode("utf-8", errors="replace")
            .rstrip()
        )
    except (FileNotFoundError, OSError, ValueError):
        existing_log = ""
    current_output = existing_log or (log_text or "").rstrip()
    failure_record = f"[ai review failure]\n{error}"
    content = f"{current_output}\n\n{failure_record}\n" if current_output else f"{failure_record}\n"
    _write_with_latest_alias(
        paper_dir_fd,
        artifact_name=log_name,
        latest_name="ai_review.log",
        content=content.encode("utf-8"),
        mode=0o600,
    )


def _failed_result(
    *,
    paper_root_fd: int,
    paper_dir_fd: int,
    log_path: Path,
    log_name: str,
    error: str,
    log_text: str | None = None,
    paper_pdf: Path,
    manifest_key: str,
    latest_review_path: Path,
    latest_log_path: Path,
) -> FinalAiReviewResult:
    _clear_latest_review_alias(paper_dir_fd)
    _write_failure_log(paper_dir_fd, log_name=log_name, error=error, log_text=log_text)
    _publish_review_status(
        paper_root_fd=paper_root_fd,
        status="failed",
        review_path=None,
        log_path=log_path,
        paper_pdf=paper_pdf,
        manifest_key=manifest_key,
        latest_review_path=latest_review_path,
        latest_log_path=latest_log_path,
    )
    return FinalAiReviewResult("failed", None, log_path, error)


def run_final_ai_review(
    *,
    repo_root: Path,
    paper_dir: Path,
    results_dir: Path,
    execute_command: ExecuteCommand,
    env_file: Path | None = None,
    uv_executable: str | None = None,
    paper_root: Path | None = None,
    artifact_stem: str = "ai_review",
    manifest_key: str = "ai_review",
    reviewer_options: Mapping[str, object] | None = None,
    reviewer_evidence_builder: Callable[[], Path | None] | None = None,
) -> FinalAiReviewResult:
    """Create a named review artifact and refresh the latest-review aliases."""
    if not _ARTIFACT_STEM_PATTERN.fullmatch(artifact_stem) or len(artifact_stem) > 128:
        raise ValueError(f"invalid artifact_stem: {artifact_stem!r}")
    if not _MANIFEST_KEY_PATTERN.fullmatch(manifest_key):
        raise ValueError(f"invalid manifest_key: {manifest_key!r}")
    repo_root = repo_root.expanduser().resolve()
    paper_root = (
        absolute_path(paper_root)
        if paper_root is not None
        else absolute_path(results_dir) / "paper"
    )
    paper_dir = absolute_path(paper_dir)
    try:
        paper_relative = paper_dir.relative_to(paper_root)
    except ValueError as exc:
        raise ValueError(f"paper directory is outside the results paper root: {paper_dir}") from exc
    paper_pdf = paper_dir / "paper.pdf"
    review_name = f"{artifact_stem}.json"
    log_name = f"{artifact_stem}.log"
    review_path = paper_dir / review_name
    log_path = paper_dir / log_name
    latest_review_path = paper_dir / "ai_review.json"
    latest_log_path = paper_dir / "ai_review.log"
    with paper_publication_lock(paper_root):
        if portable_replacement_pending(paper_root):
            recover_incomplete_promotion_locked(paper_root)
            if portable_replacement_pending(paper_root):
                raise ValueError("portable revision publication cleanup remains incomplete")
        if paper_dir != paper_root and portable_replacement_pending(paper_dir):
            raise ValueError(
                "pending nested revision publication cannot be recovered under the paper-root lock"
            )
        if reviewer_evidence_builder is not None:
            evidence = reviewer_evidence_builder()
            if evidence is not None:
                options = dict(reviewer_options or {})
                configured_evidence = options.get("evidence_paths", [])
                if not isinstance(configured_evidence, (list, tuple)):
                    raise ValueError("review evidence_paths must be a list of paths")
                options["evidence_paths"] = [*configured_evidence, evidence]
                reviewer_options = options
        try:
            paper_root_fd = open_directory_tree_no_follow(paper_root)
        except OSError as exc:
            raise ValueError(f"paper root must not be a symlink: {paper_root}") from exc
        try:
            paper_dir_fd = open_relative_directory_no_follow(paper_root_fd, paper_relative)
            try:
                return _run_pinned_final_ai_review(
                    repo_root=repo_root,
                    results_dir=absolute_path(results_dir),
                    paper_root_fd=paper_root_fd,
                    paper_dir_fd=paper_dir_fd,
                    paper_pdf=paper_pdf,
                    review_path=review_path,
                    log_path=log_path,
                    review_name=review_name,
                    log_name=log_name,
                    manifest_key=manifest_key,
                    latest_review_path=latest_review_path,
                    latest_log_path=latest_log_path,
                    execute_command=execute_command,
                    env_file=env_file,
                    uv_executable=uv_executable,
                    reviewer_options=reviewer_options,
                )
            finally:
                os.close(paper_dir_fd)
        finally:
            os.close(paper_root_fd)


def _run_pinned_final_ai_review(
    *,
    repo_root: Path,
    results_dir: Path,
    paper_root_fd: int,
    paper_dir_fd: int,
    paper_pdf: Path,
    review_path: Path,
    log_path: Path,
    review_name: str,
    log_name: str,
    manifest_key: str,
    latest_review_path: Path,
    latest_log_path: Path,
    execute_command: ExecuteCommand,
    env_file: Path | None,
    uv_executable: str | None,
    reviewer_options: Mapping[str, object] | None,
) -> FinalAiReviewResult:
    try:
        _read_paper_manifest_at(paper_root_fd)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError("missing regular latest paper manifest") from exc
    _clear_latest_review_alias(paper_dir_fd)
    _write_with_latest_alias(
        paper_dir_fd,
        artifact_name=log_name,
        latest_name="ai_review.log",
        content=b"",
        mode=0o600,
    )
    _publish_review_status(
        paper_root_fd=paper_root_fd,
        status="failed",
        review_path=None,
        log_path=log_path,
        paper_pdf=paper_pdf,
        manifest_key=manifest_key,
        latest_review_path=latest_review_path,
        latest_log_path=latest_log_path,
    )
    reviewer_script = repo_root / "src" / "aireview.py"
    if reviewer_script.is_symlink() or not reviewer_script.is_file():
        return _failed_result(
            paper_root_fd=paper_root_fd,
            paper_dir_fd=paper_dir_fd,
            log_path=log_path,
            log_name=log_name,
            error=f"missing regular AI reviewer script: {reviewer_script}",
            paper_pdf=paper_pdf,
            manifest_key=manifest_key,
            latest_review_path=latest_review_path,
            latest_log_path=latest_log_path,
        )
    agent_backend = bool(reviewer_options and reviewer_options.get("backend") == "agent")
    bound_inputs = agent_backend or bool((reviewer_options or {}).get("evidence_paths"))
    executable = (
        sys.executable if agent_backend
        else shutil.which("uv") if uv_executable is None else uv_executable
    )
    if not executable:
        return _failed_result(
            paper_root_fd=paper_root_fd,
            paper_dir_fd=paper_dir_fd,
            log_path=log_path,
            log_name=log_name,
            error="uv executable is not available",
            paper_pdf=paper_pdf,
            manifest_key=manifest_key,
            latest_review_path=latest_review_path,
            latest_log_path=latest_log_path,
        )
    try:
        if agent_backend:
            from src.review_backend import agent_review_environment
            review_environment = agent_review_environment(os.environ)
        else:
            review_environment = final_ai_review_environment(os.environ, env_file=env_file)
        review_environment["AGENTIC_COST_ROOT"] = str(results_dir)
    except (OSError, ValueError) as exc:
        return _failed_result(
            paper_root_fd=paper_root_fd,
            paper_dir_fd=paper_dir_fd,
            log_path=log_path,
            log_name=log_name,
            error=str(exc),
            paper_pdf=paper_pdf,
            manifest_key=manifest_key,
            latest_review_path=latest_review_path,
            latest_log_path=latest_log_path,
        )

    with tempfile.TemporaryDirectory(prefix=".ai-review-") as temporary_dir:
        temporary_root = Path(temporary_dir).resolve()
        temporary_pdf = temporary_root / "paper.pdf"
        temporary_review = temporary_root / "review.json"
        temporary_log = temporary_root / "review.log"
        try:
            with temporary_pdf.open("wb") as destination:
                copy_regular_at(
                    paper_dir_fd,
                    "paper.pdf",
                    destination,
                    max_bytes=_PAPER_PDF_MAX_BYTES,
                    description="final paper PDF",
                )
        except ValueError as exc:
            return _failed_result(
                paper_root_fd=paper_root_fd,
                paper_dir_fd=paper_dir_fd,
                log_path=log_path,
                log_name=log_name,
                error=str(exc),
                paper_pdf=paper_pdf,
                manifest_key=manifest_key,
                latest_review_path=latest_review_path,
                latest_log_path=latest_log_path,
            )
        except (FileNotFoundError, OSError):
            return _failed_result(
                paper_root_fd=paper_root_fd,
                paper_dir_fd=paper_dir_fd,
                log_path=log_path,
                log_name=log_name,
                error=f"missing regular final paper PDF: {paper_pdf}",
                paper_pdf=paper_pdf,
                manifest_key=manifest_key,
                latest_review_path=latest_review_path,
                latest_log_path=latest_log_path,
            )
        evidence_snapshots: list[tuple[Path, Path, dict[str, object]]] = []
        expected_inputs: list[dict[str, object]] = []
        if bound_inputs:
            try:
                expected_inputs.append(_review_input_record(
                    _read_review_input(temporary_pdf, max_bytes=_PAPER_PDF_MAX_BYTES),
                    source_id="paper", name="paper.pdf",
                ))
                evidence_paths = (reviewer_options or {}).get("evidence_paths", [])
                if not isinstance(evidence_paths, (list, tuple)):
                    raise ValueError("review evidence_paths must be a list of paths")
                for index, configured_path in enumerate(tuple(evidence_paths), 1):
                    if not isinstance(configured_path, (str, os.PathLike)):
                        raise ValueError("review evidence_paths must contain paths")
                    source_path = absolute_path(Path(configured_path))
                    content = _read_review_input(
                        source_path, max_bytes=_REVIEW_EVIDENCE_MAX_BYTES
                    )
                    # Separate directories preserve names even for duplicate basenames.
                    snapshot_dir = temporary_root / f"evidence_{index}"
                    snapshot_dir.mkdir()
                    snapshot_path = snapshot_dir / source_path.name
                    snapshot_path.write_bytes(content)
                    record = _review_input_record(
                        content, source_id=f"evidence_{index}", name=source_path.name
                    )
                    expected_inputs.append(record)
                    evidence_snapshots.append((source_path, snapshot_path, record))
            except (OSError, ValueError) as exc:
                return _failed_result(
                    paper_root_fd=paper_root_fd, paper_dir_fd=paper_dir_fd,
                    log_path=log_path, log_name=log_name, error=str(exc),
                    paper_pdf=paper_pdf, manifest_key=manifest_key,
                    latest_review_path=latest_review_path, latest_log_path=latest_log_path,
                )
        command = [executable] if agent_backend else [executable, "run", "--script"]
        command += [
            str(reviewer_script),
            str(temporary_pdf),
            "--output",
            str(temporary_review),
        ]
        if reviewer_options:
            for key, flag in {
                "backend": "--backend", "agent": "--agent", "model": "--agent-model",
                "profile": "--agent-profile", "reasoning_effort": "--reasoning-effort",
                "reviewer_count": "--reviewer-count", "timeout_seconds": "--timeout-seconds",
            }.items():
                value = reviewer_options.get(key)
                if value is not None:
                    command.extend([flag, str(value)])
            evidence_paths = (
                [snapshot for _, snapshot, _ in evidence_snapshots]
                if bound_inputs else reviewer_options.get("evidence_paths", [])
            )
            if not isinstance(evidence_paths, (list, tuple)):
                raise ValueError("review evidence_paths must be a list of paths")
            for evidence in evidence_paths:
                command.extend(["--evidence", str(evidence)])
        output, exit_code = execute_command(command, temporary_log, review_environment)
        try:
            log_text = _read_regular_file_no_follow(
                temporary_log,
                max_bytes=_REVIEW_LOG_MAX_BYTES,
                description="ai_review.log",
            )
        except (FileNotFoundError, OSError, ValueError):
            log_text = output
        _write_with_latest_alias(
            paper_dir_fd,
            artifact_name=log_name,
            latest_name="ai_review.log",
            content=log_text.encode("utf-8"),
            mode=0o600,
        )
        error: str | None
        payload: object = None
        if exit_code != 0:
            error = f"review command failed with exit code {exit_code}"
        elif temporary_review.is_symlink() or not temporary_review.is_file():
            error = f"review command did not create {review_name}"
        else:
            try:
                payload = json.loads(
                    _read_regular_file_no_follow(
                        temporary_review,
                        max_bytes=_REVIEW_OUTPUT_MAX_BYTES,
                        description=review_name,
                    )
                )
            except json.JSONDecodeError:
                error = f"{review_name} does not contain valid JSON"
            else:
                error = (
                    None if isinstance(payload, dict) else f"{review_name} must contain an object"
                )
        if error is None and bound_inputs:
            try:
                backend = "agent" if agent_backend else "openai"
                if not isinstance(payload, dict):
                    raise ValueError(f"{backend} review must be an object")
                provenance = payload.get("Provenance")
                if not isinstance(provenance, dict) or provenance.get("backend") != backend or provenance.get("independent_context") is not True:
                    raise ValueError(f"{backend} review has no independent provenance")
                _validate_review_input_receipt(provenance, expected_inputs)
                paper_record = expected_inputs[0]
                for content in (
                    _read_review_input(temporary_pdf, max_bytes=_PAPER_PDF_MAX_BYTES),
                    read_regular_at(
                        paper_dir_fd, "paper.pdf", max_bytes=_PAPER_PDF_MAX_BYTES,
                        description="final paper PDF",
                    ),
                ):
                    if _review_input_record(content, source_id="paper", name="paper.pdf") != paper_record:
                        raise ValueError(f"reviewed PDF changed during {backend} review")
                for source_path, snapshot_path, record in evidence_snapshots:
                    for input_path in (source_path, snapshot_path):
                        content = _read_review_input(
                            input_path, max_bytes=_REVIEW_EVIDENCE_MAX_BYTES
                        )
                        if _review_input_record(
                            content, source_id=str(record["id"]), name=source_path.name
                        ) != record:
                            raise ValueError(f"configured evidence changed during {backend} review")
                if agent_backend:
                    from src.review_backend import validate_agent_review
                    validate_agent_review(
                        {key: value for key, value in payload.items() if key != "Provenance"},
                        source_ids={str(item["id"]) for item in expected_inputs},
                    )
                else:
                    from src.aireview import _validate_review
                    _validate_review(payload, context="OpenAI evidence review")
            except (OSError, ValueError) as exc:
                error = str(exc)
        if error is not None:
            return _failed_result(
                paper_root_fd=paper_root_fd,
                paper_dir_fd=paper_dir_fd,
                log_path=log_path,
                log_name=log_name,
                error=error,
                log_text=log_text,
                paper_pdf=paper_pdf,
                manifest_key=manifest_key,
                latest_review_path=latest_review_path,
                latest_log_path=latest_log_path,
            )
        assert isinstance(payload, dict)
        _write_with_latest_alias(
            paper_dir_fd,
            artifact_name=review_name,
            latest_name="ai_review.json",
            content=(json.dumps(payload, indent=2) + "\n").encode("utf-8"),
            mode=0o600,
        )

    try:
        _publish_review_status(
            paper_root_fd=paper_root_fd,
            status="completed",
            review_path=review_path,
            log_path=log_path,
            paper_pdf=paper_pdf,
            manifest_key=manifest_key,
            latest_review_path=latest_review_path,
            latest_log_path=latest_log_path,
        )
    except ValueError as exc:
        _clear_latest_review_alias(paper_dir_fd)
        _write_failure_log(paper_dir_fd, log_name=log_name, error=str(exc))
        return FinalAiReviewResult("failed", None, log_path, str(exc))
    overall_score, decision = _review_summary(payload)
    return FinalAiReviewResult(
        "completed",
        review_path,
        log_path,
        overall_score=overall_score,
        decision=decision,
    )
