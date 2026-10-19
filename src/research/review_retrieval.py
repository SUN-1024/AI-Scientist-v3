"""Bounded retrieval from verified independent-review requests.

Design attribution: adopts the bounded query/result feedback pattern of
AI-Scientist V1 check_idea_novelty (1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb,
ai_scientist/generate_ideas.py) and V2 generate_temp_free_idea
(96bd51617cfdbb494a9fc283af00fe090edfae48, ai_scientist/perform_ideation_temp_free.py).
This is a new host implementation, not copied upstream search or verdict code.
Unlike those loops, a reviewer request is a search lead, never a paper fact;
new full text is sealed in a new task view, and independent review remains the
coordinator's responsibility. This module never invokes experiments or reviews.

Successful calls are checkpointed. An interrupted, unsealed call is retained
and charged, never automatically repeated. Byte reservations conservatively
include both download attempts when the existing downloader uses a proxy fallback.
The injected search backend must obey cfg.model_search.timeout_seconds, make at
most one bounded search campaign, and save its provider evidence below output.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from src.orchestrator.state import write_private_text
from .config import LiteratureConfig
from .evidence import load_catalog
from .literature import MAX_DOWNLOAD_BYTES, _extract_text, download, passages
from .model_search import _identity, search_literature
from .novelty import content_hash, freeze_evidence
from .novelty_review import MAX_PACKET_CHARS, load_frozen_novelty_report, source_packet
from .workspace import task_digest

LITERATURE_REPAIRABLE_MACHINE_GAPS = frozenset({
    "missing_closest_work", "missing_passages", "missing_comparison_evidence",
    "incomplete_comparison", "untriaged_source", "source_selection_rationale",
    "search_scope_missing",
})
REVIEW_INTEGRITY_GAPS = frozenset({
    "source_integrity", "catalog_changed", "campaign_plan_integrity", "agenda_changed",
    "candidate_evidence_invalid", "candidate_review_unbound", "candidate_packet_required",
    "candidate_packet_changed", "claim_code_evidence_invalid", "review_receipt_invalid",
    "review_citation_invalid", "review_coverage_missing", "independent_reviews_missing",
})

EVIDENCE_TASK_FILES = ("task.md", "research.yaml", "research_agenda.json", "research_seed_agenda.json")

SearchBackend = Callable[[LiteratureConfig, Path], tuple[list[dict[str, Any]], list[dict[str, str]]]]
PRIMARY_HOSTS = (
    "arxiv.org", "openreview.net", "proceedings.mlr.press", "jmlr.org", "aaai.org",
    "openaccess.thecvf.com", "cv-foundation.org", "aclanthology.org",
    "proceedings.neurips.cc", "papers.nips.cc", "proceedings.iclr.cc",
    "dl.acm.org", "ieeexplore.ieee.org", "ieee.org", "springer.com",
    "nature.com", "sciencedirect.com", "cell.com", "science.org",
    "pnas.org", "plos.org", "bmj.com", "wiley.com", "onlinelibrary.wiley.com",
    "academic.oup.com", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "biorxiv.org", "medrxiv.org", "europepmc.org", "zenodo.org",
)


@dataclass(frozen=True)
class RetrievalBudget:
    max_rounds: int = 2
    max_queries_per_round: int = 4
    max_new_fulltexts_per_round: int = 4
    max_total_search_calls: int = 8
    max_total_download_calls: int = 8
    max_total_download_bytes: int = 2 * 8 * (MAX_DOWNLOAD_BYTES + 1)
    max_elapsed_seconds: int = 1800

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"Retrieval budget {name} must be a positive integer")
        if self.max_rounds > 8 or self.max_queries_per_round > 16 or self.max_new_fulltexts_per_round > 16:
            raise ValueError("Retrieval rounds/queries/fulltexts exceed the bounded host limits")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(path: Path, value: Any) -> None:
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f"Invalid retrieval artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(workspace: Path, path: Path | str) -> Path:
    raw_value = str(path)
    path = Path(path)
    if not path.is_absolute():
        raw = PurePosixPath(str(path))
        if ".." in raw.parts or str(raw) != raw_value or not raw.parts:
            raise ValueError("Retrieval paths must be canonical workspace-relative paths")
        path = workspace / path
    try:
        relative = path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("Retrieval path escaped its workspace") from exc
    if ".." in relative.parts or path != Path(os.path.abspath(path)):
        raise ValueError("Retrieval path is not canonical")
    current = workspace
    for part in relative.parts:
        if current.is_symlink():
            raise ValueError("Retrieval paths cannot cross symlinks")
        current /= part
    if current.is_symlink() or workspace.is_symlink():
        raise ValueError("Retrieval paths cannot cross symlinks")
    return path


def _tree(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Evidence must be a regular directory")
    files = {}
    for parent, directories, names in os.walk(path, followlinks=False):
        for name in directories + names:
            child = Path(parent) / name
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError("Evidence trees cannot contain symlinks or special files")
            if stat.S_ISREG(mode):
                files[child.relative_to(path).as_posix()] = _sha(child)
    return dict(sorted(files.items()))


def _binary(path: Path, body: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as stream:
        os.chmod(stream.name, 0o600)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


@contextmanager
def _lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Invalid retrieval lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This retrieval chain is already running") from exc
        yield
    finally:
        os.close(fd)


def extract_review_requests(task_dir: Path, review_dir: Path, *,
                            candidate_packet: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rebuild the report from provider receipts; ignore unverified judgments.

    Returns a bound request packet, not a claim that cited papers exist. Only
    retrieval actions and paper-specific full-text actions are actionable.
    """
    _tree(task_dir)
    _tree(review_dir)
    report = load_frozen_novelty_report(task_dir, review_dir, candidate_packet=candidate_packet)
    checks = report.get("machine_checks", {})
    binding = {
        "parent_report_sha256": _sha(review_dir / "novelty_report.json"),
        "parent_evaluated_report_sha256": content_hash(report),
        "parent_task_sha256": task_digest(task_dir),
        "frozen_evidence": report.get("frozen_evidence", {}),
        "candidate_packet_sha256": content_hash(candidate_packet) if candidate_packet is not None else None,
    }
    requests, receipts, skipped = [], [], []
    # Missing literature is precisely what this stage can repair. These gaps
    # remain in the request packet; they are never converted into passed checks.
    machine_gaps = checks.get("gaps", [])
    report_gaps = report.get("gaps", [])
    gap_fields = {"machine_gaps": copy.deepcopy(machine_gaps), "review_gaps": copy.deepcopy(report_gaps)}
    if report.get("status") != "needs_evidence":
        return {"binding": binding, "requests": [], "receipts": [], **gap_fields,
                "reason": "review_does_not_request_evidence"}
    repairable = (isinstance(machine_gaps, list) and all(isinstance(row, dict)
                  and row.get("code") in LITERATURE_REPAIRABLE_MACHINE_GAPS for row in machine_gaps))
    integrity_gap = (not isinstance(report_gaps, list) or any(not isinstance(row, dict)
                     or row.get("code") in REVIEW_INTEGRITY_GAPS for row in report_gaps))
    if (checks.get("review_artifacts_valid") is not True or integrity_gap or not repairable
            or (checks.get("passed") is not True and not machine_gaps)
            or len(report.get("model_judgments", [])) < 2):
        return {"binding": binding, "requests": [], "receipts": [], **gap_fields,
                "reason": "review_integrity_or_structure_gap"}
    seen = set()
    for judgment in report.get("model_judgments", []):
        receipt = judgment["receipt"]
        receipt_path = Path(receipt["path"])
        if not receipt_path.is_relative_to(review_dir) or receipt_path.is_symlink() or _sha(receipt_path) != receipt["sha256"]:
            raise ValueError("Parent review receipt differs from the verified report")
        receipts.append({"path": receipt_path.relative_to(review_dir).as_posix(), "sha256": receipt["sha256"]})
        for index, action in enumerate(judgment["assessment"].get("next_actions", [])):
            kind, reason = action.get("kind"), action.get("reason", "")
            if kind not in {"retrieve_literature", "inspect_full_text"}:
                continue
            if not isinstance(reason, str) or not reason.strip() or len(reason) > 8000:
                skipped.append({"kind": kind, "reason": "unbounded_or_empty_request"})
                continue
            if kind == "inspect_full_text" and not re.search(r"paper|article|supplement|doi\b|arxiv|文献|论文|全文", reason, re.I):
                skipped.append({"kind": kind, "reason": "not_a_paper_retrieval_request"})
                continue
            key = content_hash({"kind": kind, "reason": " ".join(reason.split()).casefold()})
            if key not in seen:
                requests.append({"id": key, "kind": kind, "reason": reason,
                                 "review_receipt_sha256": receipt["sha256"], "action_index": index})
                seen.add(key)
    if len(requests) > 64:
        raise ValueError("Too many reviewer retrieval requests")
    return {"binding": binding, "requests": requests, "receipts": receipts, "skipped": skipped, **gap_fields,
            "reason": None if requests else "no_actionable_literature_request"}



def primary_source_identity(url: str) -> dict[str, str]:
    """Extract a stable work identity from a public source URL, without I/O.

    The identifier describes what the URL names. It does not authenticate a
    publisher or prove that different identifier namespaces describe one work.
    arXiv revisions share a work identity; an explicit revision is retained.
    Unknown URL forms use their exact normalized URL rather than guessed IDs.
    """
    if not isinstance(url, str):
        raise ValueError("Primary source identity requires a URL")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Primary source identity requires a public HTTP(S) URL")
    path = unquote(parsed.path).rstrip("/")
    if host in {"arxiv.org", "export.arxiv.org"}:
        match = re.fullmatch(r"/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(v\d+)?(?:\.pdf)?", path)
        if match:
            identity = {"scheme": "arxiv", "value": match[1].casefold()}
            if match[2]:
                identity["version"] = match[2]
            return identity
    doi = None
    doi_match = re.search(r"/(?:doi(?:/(?:abs|abstract|full|pdf|pdfplus|epdf))?|article|content/pdf)/(10\.\d{4,9}/.+)", path, re.I)
    if host in {"doi.org", "dx.doi.org"}:
        doi = path.lstrip("/")
    elif doi_match:
        doi = doi_match[1]
        if "/content/pdf/" in path:
            doi = doi.removesuffix(".pdf")
    if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        return {"scheme": "doi", "value": doi.casefold()}
    if host in {"openreview.net"} and path in {"/forum", "/pdf"}:
        values = parse_qs(parsed.query).get("id", [])
        if len(values) == 1 and values[0]:
            return {"scheme": "openreview", "value": values[0]}
    if host == "ojs.aaai.org":
        match = re.fullmatch(r"/index\.php/([A-Za-z0-9_-]+)/article/(?:view|download)/(\d+)(?:/\d+)?", path)
        if match:
            # OJS's last numeric component is a galley, not a different article.
            # Keep the journal path so unrelated proceedings cannot collide.
            return {"scheme": "aaai_ojs", "value": match[1] + "/" + match[2]}
    if host == "jmlr.org":
        match = re.fullmatch(r"/papers/v(\d+)/([A-Za-z0-9_-]+)\.html", path)
        if not match:
            match = re.fullmatch(r"/papers/(?:volume|v)(\d+)/([A-Za-z0-9_-]+)/\2\.pdf", path)
        if match:
            return {"scheme": "jmlr", "value": match[1] + "/" + match[2]}
    if host in {"pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov"}:
        match = re.search(r"/(?:pmc/)?articles/(PMC\d+)(?:/|$)", path, re.I)
        if match:
            return {"scheme": "pmc", "value": match[1].upper()}
    if host == "proceedings.mlr.press":
        match = re.fullmatch(r"/(v\d+)/([^/]+)\.html", path)
        if match:
            return {"scheme": "mlr", "value": match[1] + "/" + match[2]}
        match = re.fullmatch(r"/(v\d+)/([^/]+)/([^/]+)\.pdf", path)
        if match and match[3] in {match[2], match[2] + "-supp", match[2] + "-supplementary"}:
            return {"scheme": "mlr", "value": match[1] + "/" + match[2]}
    if host in {"openaccess.thecvf.com", "cv-foundation.org"}:
        match = re.fullmatch(r"/(?:openaccess/)?(content/[^/]+|content_[^/]+)/(?:html|papers|supplemental)/([^/]+)\.(?:html|pdf)", path)
        if match:
            stem = re.sub(r"_supplemental$", "_paper", match[2])
            return {"scheme": "cvf", "value": (match[1] + "/" + stem).casefold()}
    if host in {"proceedings.neurips.cc", "papers.nips.cc"}:
        match = re.fullmatch(r"/(?:paper_files/)?paper/(\d{4})/(?:hash|file)/([a-fA-F0-9]{32})(?:-[^/]+)?\.(?:html|pdf)", path)
        if match:
            return {"scheme": "neurips", "value": match[1] + "/" + match[2].lower()}
    if host == "aclanthology.org":
        key = path.lstrip("/").removesuffix(".pdf")
        if re.fullmatch(r"(?:\d{4}\.[A-Za-z0-9_-]+\.\d+|[A-Za-z]\d{2}-\d{4})", key):
            return {"scheme": "acl", "value": key}
    if host == "ieeexplore.ieee.org":
        match = re.fullmatch(r"/document/(\d+)", path)
        values = parse_qs(parsed.query).get("arnumber", [])
        key = match[1] if match else values[0] if path == "/stamp/stamp.jsp" and len(values) == 1 else ""
        if key.isdigit():
            return {"scheme": "ieee", "value": key}
    return {"scheme": "url", "value": _identity(url)}


def compatible_primary_urls(landing_url: str, pdf_url: str) -> bool:
    """Require an equal known work ID, or exactly equal unknown document URLs.

    A publisher DOI and an arXiv ID need independent cross-identity evidence.
    Model proposals or matching titles alone are not that evidence.
    """
    landing, fulltext = primary_source_identity(landing_url), primary_source_identity(pdf_url)
    return (landing["scheme"], landing["value"]) == (fulltext["scheme"], fulltext["value"])


def _primary_pdf(record: dict[str, Any], *, suggested: bool, trace_sources: list[dict]) -> tuple[str, str]:
    title, url = record.get("title"), record.get("url")
    if not isinstance(title, str) or not 8 <= len(title.strip()) <= 1000 or not isinstance(url, str):
        raise ValueError("Paper proposal requires a bounded title and URL")
    known = {_identity(row["url"]) for row in trace_sources if isinstance(row, dict) and isinstance(row.get("url"), str)}
    if not suggested and _identity(url) not in known:
        raise ValueError("Paper URL has no saved search receipt")
    access = record.get("openAccessPdf")
    pdf = access.get("url") if isinstance(access, dict) else None
    match = re.fullmatch(r"https?://(?:www\.)?(?:export\.)?arxiv\.org/(?:abs|html|pdf)/([\w./-]+)", url)
    if not pdf and match:
        pdf = "https://arxiv.org/pdf/" + match[1].removesuffix(".pdf")
    if not pdf and urlsplit(url).path.lower().endswith(".pdf"):
        pdf = url
    if not isinstance(pdf, str):
        raise ValueError("No primary full-text PDF URL was resolved")
    parsed = urlsplit(pdf)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"https", "http"} or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("Invalid public paper URL")
    if not any(host == item or host.endswith("." + item) for item in PRIMARY_HOSTS) and not host.endswith((".edu", ".ac.uk", ".edu.cn")):
        raise ValueError("PDF origin is not a recognized primary repository/publisher/university")
    if not suggested and _identity(pdf) not in known and not match:
        raise ValueError("PDF URL has no saved search receipt")
    if not compatible_primary_urls(url, pdf):
        landing_identity, pdf_identity = primary_source_identity(url), primary_source_identity(pdf)
        raise ValueError("Conflicting or unproven landing/PDF work identities: "
                         + json.dumps({"landing": landing_identity, "pdf": pdf_identity}, sort_keys=True))
    return title.strip(), pdf


def _title_matches(title: str, text: str) -> bool:
    # Punctuation, line wraps and PDF ligatures may differ; do not accept merely
    # a few shared topic words. Require the complete normalized title up front.
    import unicodedata
    def normalize(value: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKC", value).casefold() if ch.isalnum())
    needle = normalize(title)
    return len(needle) >= 8 and needle in normalize(text[:16000])


def validate_enrichment_result(workspace: Path, result: dict[str, Any]) -> Path | None:
    """Resolve and verify a completed snapshot at the coordinator boundary."""
    if result.get("status") != "enriched":
        return None
    path = _inside(workspace, result["evidence_task"])
    _tree(path)
    if task_digest(path) != result["evidence_task_sha256"]:
        raise ValueError("Enrichment evidence task changed after sealing")
    if freeze_evidence(path) != result["frozen_evidence"]:
        raise ValueError("Enrichment catalog or policy changed after sealing")
    return path


def enrich_review_evidence(workspace: Path, task_dir: Path, review_dir: Path, cfg: LiteratureConfig, *,
                           journal_dir: Path, base_input_sha256: str,
                           candidate_packet: dict[str, Any] | None = None,
                           budget: RetrievalBudget | None = None, offline: bool = False,
                           search_backend: SearchBackend | None = None,
                           search_backend_id: str | None = None,
                           deadline_unix: float | None = None) -> dict[str, Any]:
    """Run or resume one evidence-enrichment round in an append-only chain.

    All paths must be inside workspace. journal_dir must be below novelty/ and
    outside task_dir/review_dir. Call again with the NEW verified review to use
    the next round. Same parent input replays the existing result. Search budget
    counts bounded campaigns (external campaigns have at most four HTTP calls).
    Download-byte budget counts conservative transfer reservations, not only
    successful response sizes. No interrupted call is automatically retried.
    deadline_unix is the coordinator's fixed absolute Unix deadline. It is frozen
    across resumes; the earlier of it and this chain's own elapsed limit applies.
    """
    workspace = Path(os.path.abspath(workspace))
    task_dir, review_dir, journal_dir = [_inside(workspace, path) for path in (task_dir, review_dir, journal_dir)]
    if not journal_dir.is_relative_to(workspace / "novelty") or journal_dir == workspace / "novelty":
        raise ValueError("Retrieval journal must be a dedicated directory below workspace/novelty")
    if any(journal_dir.is_relative_to(path) or path.is_relative_to(journal_dir) for path in (task_dir, review_dir)):
        # A previous round's evidence_task legitimately lies inside this journal.
        if not (task_dir.is_relative_to(journal_dir) and not review_dir.is_relative_to(journal_dir)
                and not journal_dir.is_relative_to(task_dir) and not journal_dir.is_relative_to(review_dir)):
            raise ValueError("Retrieval journal overlaps its input task or review")
    budget = budget or RetrievalBudget()
    budget.validate()
    if deadline_unix is not None:
        if type(deadline_unix) not in {int, float}:
            raise ValueError("Retrieval deadline_unix must be a finite positive Unix timestamp")
        try:
            deadline_unix = float(deadline_unix)
        except OverflowError as exc:
            raise ValueError("Retrieval deadline_unix must be a finite positive Unix timestamp") from exc
        if not math.isfinite(deadline_unix) or deadline_unix <= 0:
            raise ValueError("Retrieval deadline_unix must be a finite positive Unix timestamp")
    if not re.fullmatch(r"[a-f0-9]{64}", base_input_sha256):
        raise ValueError("Retrieval requires the frozen base input SHA-256")
    base = _inside(workspace, "input_task")
    _tree(base)
    if task_digest(base) != base_input_sha256:
        raise ValueError("Base input changed before literature enrichment")
    backend_id = search_backend_id if search_backend is not None else "model_search_v1"
    if not isinstance(backend_id, str) or not 1 <= len(backend_id) <= 256:
        raise ValueError("Injected search backend needs a frozen bounded identity")
    packet = extract_review_requests(task_dir, review_dir, candidate_packet=candidate_packet)
    config_binding = content_hash(asdict(cfg))
    fixed = {"budget": asdict(budget), "base_input_sha256": base_input_sha256,
             "candidate_packet_sha256": packet["binding"]["candidate_packet_sha256"],
             "search_backend_id": backend_id, "configuration_sha256": config_binding, "offline": offline,
             "deadline_unix": deadline_unix}
    with _lock(journal_dir):
        return _enrich_locked(workspace, task_dir, review_dir, cfg, journal_dir, packet,
                              fixed, budget, search_backend or search_literature)



def _validate_saved_calls(workspace: Path, row: dict[str, Any]) -> None:
    for search in row["searches"]:
        output = _inside(workspace, search["directory"])
        if search.get("files") is not None and _tree(output) != search["files"]:
            raise ValueError("Saved search result artifacts changed")
        if search["status"] == "completed":
            saved = _read(output / "retrieval_result.json")
            if saved != {"records": search["records"], "errors": search["errors"]}:
                raise ValueError("Search journal differs from its sealed result")
    for item in row["downloads"]:
        if not any(search["status"] == "completed" and search["request"]["id"] == item["request_id"]
                   and item["record"] in search["records"] for search in row["searches"]):
            raise ValueError("Download is not bound to a sealed search proposal")
        for key in ("raw", "text"):
            if item.get(key + "_sha256") and _sha(_inside(workspace, item[key])) != item[key + "_sha256"]:
                raise ValueError("Saved downloaded evidence changed")


def _enrich_locked(workspace, task_dir, review_dir, cfg, directory, packet, fixed, budget, backend):
    journal_path = directory / "journal.json"
    if journal_path.exists():
        journal = _read(journal_path)
        if journal.get("schema_version") != 1 or journal.get("fixed") != fixed:
            raise ValueError("Retrieval policy/configuration/base/candidate changed on resume")
    else:
        journal = {"schema_version": 1, "fixed": fixed, "created_at_unix": time.time(), "rounds": []}
        _dump(journal_path, journal)
    def save():
        _dump(journal_path, journal)
    def remaining():
        own_deadline = journal["created_at_unix"] + budget.max_elapsed_seconds
        outer_deadline = fixed["deadline_unix"]
        return min(own_deadline, outer_deadline if outer_deadline is not None else own_deadline) - time.time()
    def call_timeout(configured, attempts):
        # Existing transports require integer seconds. If even one second per
        # possible attempt cannot fit, do not start the external call.
        return min(configured, max(0, math.floor(remaining() / attempts)))
    parent = {**packet["binding"], "parent_task": task_dir.relative_to(workspace).as_posix(),
              "review_cycle": review_dir.relative_to(workspace).as_posix(),
              "request_packet_sha256": content_hash(packet)}
    rounds = journal["rounds"]
    current = next((row for row in rounds if row["parent"] == parent), None)
    if current is None:
        if any(row["status"] not in {"completed", "stopped"} for row in rounds):
            raise ValueError("Finish the existing retrieval parent before opening another round")
        if len(rounds) >= budget.max_rounds:
            return {"status": "stopped", "reason": "round_budget_exhausted", "binding": parent}
        if rounds:
            previous = rounds[-1].get("result", {})
            if previous.get("status") != "enriched" or previous.get("evidence_task_sha256") != parent["parent_task_sha256"]:
                # Coordinators may copy the sealed snapshot and publish their
                # reviewed agenda/plan there. Verify source inheritance instead.
                prior_path = validate_enrichment_result(workspace, previous)
                if prior_path is None or freeze_evidence(prior_path) != freeze_evidence(task_dir):
                    raise ValueError("New retrieval parent does not inherit the preceding frozen evidence")
        current = {"index": len(rounds) + 1, "parent": parent, "status": "started",
                   "requests": packet, "searches": [], "downloads": [], "created_at_unix": time.time()}
        rounds.append(current)
        save()
    round_dir = directory / f"round_{current['index']:02d}"
    round_dir.mkdir(exist_ok=True)
    _inside(workspace, round_dir)
    def stop(reason):
        result = {"status": "stopped", "reason": reason, "binding": parent,
                  "round": current["index"], "journal": journal_path.relative_to(workspace).as_posix()}
        current.update(status="stopped", result=result)
        save()
        return result
    # Completed calls and snapshots are never silently repaired or reissued.
    _validate_saved_calls(workspace, current)
    if current["status"] in {"completed", "stopped"}:
        validate_enrichment_result(workspace, current["result"])
        return current["result"]
    if current["status"] == "publishing":
        return _publish(workspace, current, save)
    if packet.get("reason"):
        return stop(packet["reason"])
    if fixed["offline"]:
        return stop("offline_retrieval_disabled")
    if fixed["search_backend_id"] == "model_search_v1" and not cfg.model_search.model:
        return stop("search_not_configured")
    if remaining() <= 0:
        return stop("elapsed_budget_exhausted")
    used_requests = {search["request"]["id"] for row in rounds[:-1] for search in row["searches"]}
    chosen = [request for request in packet["requests"] if request["id"] not in used_requests][:budget.max_queries_per_round]
    if not chosen:
        return stop("repeated_requests_without_new_evidence")
    stop_cause = "no_new_verified_full_text"
    search_attempts = (4 * (2 if cfg.proxy_fallback and not cfg.model_search.proxy_url else 1)
                       if fixed["search_backend_id"] == "model_search_v1" else 1)
    for request in chosen:
        entry = next((row for row in current["searches"] if row["request"]["id"] == request["id"]), None)
        if entry is not None:
            if entry["status"] == "started":
                entry.update(status="interrupted", error="Ambiguous prior call retained; no automatic repeated API call")
                save()
            continue
        if sum(len(row["searches"]) for row in rounds) >= budget.max_total_search_calls or remaining() <= 0:
            stop_cause = "search_budget_exhausted" if remaining() > 0 else "elapsed_budget_exhausted"
            break
        if call_timeout(cfg.model_search.timeout_seconds, search_attempts) < 1:
            stop_cause = "elapsed_budget_exhausted"
            break
        output = round_dir / f"search_{len(current['searches']) + 1:02d}"
        output.mkdir()
        entry = {"request": request, "status": "started", "directory": output.relative_to(workspace).as_posix(),
                 "started_at_unix": time.time()}
        current["searches"].append(entry)
        save()  # Charge before any external call, including ambiguous failures.
        search_cfg = copy.deepcopy(cfg)
        search_cfg.sources = []
        search_cfg.queries = [request["reason"]]
        search_cfg.max_papers = budget.max_new_fulltexts_per_round
        timeout = call_timeout(cfg.model_search.timeout_seconds, search_attempts)
        if timeout < 1:
            entry.update(status="not_started_deadline", error="Deadline leaves no transport attempt", files=_tree(output))
            stop_cause = "elapsed_budget_exhausted"
            save()
            break
        search_cfg.model_search = replace(search_cfg.model_search, max_search_calls=1, max_continuations=0,
                                         timeout_seconds=timeout)
        entry.update(request_timeout_seconds=timeout, transport_attempt_bound=search_attempts)
        save()
        # Recheck after the durable reservation, directly before the backend.
        timeout = call_timeout(cfg.model_search.timeout_seconds, search_attempts)
        if timeout < 1:
            entry.update(status="not_started_deadline", error="Deadline elapsed before backend call", files=_tree(output))
            stop_cause = "elapsed_budget_exhausted"
            save()
            break
        search_cfg.model_search.timeout_seconds = timeout
        try:
            records, errors = backend(search_cfg, output)
            if not isinstance(records, list) or len(records) > search_cfg.max_papers or not all(isinstance(row, dict) for row in records) or not isinstance(errors, list):
                raise ValueError("Search backend returned malformed records/errors")
            if len(json.dumps([records, errors], ensure_ascii=False)) > 512_000:
                raise ValueError("Search backend records exceed the bounded metadata packet")
            # Persist returned data even when an injected backend uses a distinct
            # receipt schema. Its proposals alone never establish source identity.
            _dump(output / "retrieval_result.json", {"records": records, "errors": errors})
            entry.update(status="completed", records=records, errors=errors)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            entry.update(status="failed", error=str(exc))
        entry.update(files=_tree(output), finished_at_unix=time.time())
        save()
    catalog = load_catalog(task_dir)
    existing_hashes = {row.get("download_sha256") for row in catalog["sources"]}
    existing_hashes.update(row["raw_sha256"] for row in current["downloads"] if row["status"] == "accepted")
    existing_urls = {_identity(row["url"]) for row in catalog["sources"] if row.get("url")}
    records = []
    for search in current["searches"]:
        if search["status"] != "completed":
            continue
        trace = _inside(workspace, search["directory"]) / "model_search/trace.json"
        trace_sources = _read(trace).get("sources", []) if trace.is_file() else []
        for record in search["records"]:
            try:
                suggested = fixed["search_backend_id"] != "model_search_v1" and record.get("metadata_status") == "agent_suggested_unverified"
                title, pdf = _primary_pdf(record, suggested=suggested, trace_sources=trace_sources)
                if _identity(record["url"]) in existing_urls:
                    # Existing incomplete sources may be upgraded by an appended
                    # version; existing complete sources are not downloaded again.
                    if any(row.get("availability") == "full_text" and row.get("url") and _identity(row["url"]) == _identity(record["url"]) for row in catalog["sources"]):
                        continue
                if not any(row[1] == pdf for row in records):
                    records.append((title, pdf, record, search["request"]["id"]))
            except (ValueError, TypeError, KeyError) as exc:
                search.setdefault("rejected_records", []).append({"title": record.get("title"), "reason": str(exc)})
    save()
    for title, pdf, record, request_id in records:
        if sum(row.get("status") == "accepted" for row in current["downloads"]) >= budget.max_new_fulltexts_per_round:
            break
        item = next((row for row in current["downloads"] if row["url"] == pdf), None)
        if item is None:
            attempts = [item for row in rounds for item in row["downloads"]]
            reservation = (MAX_DOWNLOAD_BYTES + 1) * (2 if cfg.proxy_fallback else 1)
            if (len(attempts) >= budget.max_total_download_calls or
                    sum(item["reserved_bytes"] for item in attempts) + reservation > budget.max_total_download_bytes or remaining() <= 0):
                stop_cause = "download_budget_exhausted" if remaining() > 0 else "elapsed_budget_exhausted"
                break
            download_attempts = 2 if cfg.proxy_fallback else 1
            if call_timeout(cfg.timeout_seconds, download_attempts) < 1:
                stop_cause = "elapsed_budget_exhausted"
                break
            item = {"url": pdf, "title": title, "record": record, "request_id": request_id,
                    "status": "started", "reserved_bytes": reservation, "started_at_unix": time.time()}
            current["downloads"].append(item)
            save()
            timeout = call_timeout(cfg.timeout_seconds, download_attempts)
            if timeout < 1:
                item.update(status="not_started_deadline", error="Deadline elapsed before download call")
                stop_cause = "elapsed_budget_exhausted"
                save()
                break
            item.update(request_timeout_seconds=timeout, transport_attempt_bound=download_attempts)
            try:
                body = download(pdf, replace(cfg, timeout_seconds=timeout))
                if len(body) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Downloaded body exceeds the per-source limit")
                raw = round_dir / f"paper_{len(current['downloads']):02d}.pdf"
                _binary(raw, body)
                item.update(status="downloaded", raw=raw.relative_to(workspace).as_posix(), raw_sha256=_sha(raw), downloaded_bytes=len(body))
                save()  # Parsing can resume without another download.
            except (ValueError, OSError, TypeError) as exc:
                item.update(status="failed", error=str(exc))
                save()
                continue
        elif item["status"] == "started":
            item.update(status="interrupted", error="Ambiguous download retained; no automatic repeated request")
            save()
        if item["status"] != "downloaded":
            continue
        try:
            body = _inside(workspace, item["raw"]).read_bytes()
            if not body.startswith(b"%PDF-"):
                raise ValueError("Primary URL did not return a PDF")
            text, availability = _extract_text(body, pdf=True)
            if availability != "full_text":
                raise ValueError("Incomplete/scanned full text cannot satisfy enrichment")
            if not _title_matches(title, text):
                raise ValueError("Downloaded first-page text does not match the proposed complete title")
            if item["raw_sha256"] in existing_hashes:
                raise ValueError("Downloaded bytes duplicate an already frozen source")
            text_path = round_dir / (Path(item["raw"]).stem + ".txt")
            write_private_text(text_path, text)
            item.update(status="accepted", text=text_path.relative_to(workspace).as_posix(), text_sha256=_sha(text_path), finished_at_unix=time.time())
            existing_hashes.add(item["raw_sha256"])
        except (ValueError, OSError, TypeError, KeyError) as exc:
            item.update(status="rejected", error=str(exc))
        save()
    accepted = [row for row in current["downloads"] if row["status"] == "accepted"]
    if not accepted:
        return stop(stop_cause if remaining() > 0 else "elapsed_budget_exhausted")
    # A copy is private to this cycle; base task and old receipt inputs never move.
    before = {"research_context/" + name: digest for name, digest in _tree(task_dir / "research_context").items()}
    before.update({name: _sha(task_dir / name) for name in EVIDENCE_TASK_FILES if (task_dir / name).is_file()})
    staging = Path(tempfile.mkdtemp(prefix="evidence_task.staging_", dir=round_dir))
    shutil.copytree(task_dir / "research_context", staging / "research_context")
    for name in EVIDENCE_TASK_FILES:
        if (task_dir / name).is_file():
            shutil.copyfile(task_dir / name, staging / name)
    if task_digest(task_dir) != parent["parent_task_sha256"] or _tree(staging) != before:
        raise ValueError("Parent evidence changed during snapshot copy")
    context = staging / "research_context"
    history = context / "catalog_history"
    history.mkdir(exist_ok=True)
    original = context / "catalog.json"
    archived = history / (_sha(original) + ".json")
    if archived.exists() and archived.read_bytes() != original.read_bytes():
        raise ValueError("Catalog history collision")
    _binary(archived, original.read_bytes())
    (context / "raw").mkdir(exist_ok=True)
    (context / "text").mkdir(exist_ok=True)
    new_ids = []
    projected_packet = source_packet(task_dir)
    for item in accepted:
        source_id = "retrieved_" + item["raw_sha256"][:24]
        if any(row["id"] == source_id for row in catalog["sources"]):
            raise ValueError("New source identity collides with frozen catalog")
        raw_path, text_path = f"raw/{source_id}.pdf", f"text/{source_id}.txt"
        record = item["record"]
        text = _inside(workspace, item["text"]).read_text(encoding="utf-8")
        new_source = {
            "id": source_id, "title": item["title"], "year": record.get("year"), "venue": record.get("venue"),
            "url": record["url"], "origin": item["url"], "availability": "full_text",
            "verified_origin_url": item["url"], "primary_identity": primary_source_identity(item["url"]),
            "metadata_status": "primary_content_title_matched_other_metadata_unverified",
            "discovery_status": record.get("metadata_status", "unverified"),
            "search_sources": record.get("search_sources", []), "text_path": text_path,
            "text_sha256": item["text_sha256"], "download_sha256": item["raw_sha256"], "raw_path": raw_path,
            "passages": passages(text, source_id, [row["reason"] for row in packet["requests"]]),
            "retrieval_request_id": item["request_id"],
            "identity_check": "Landing and PDF URLs name the same work under a recognized identifier (or the same exact unknown URL); the downloaded PDF contains the complete normalized title near its start. Explicit arXiv revisions are retained and may differ. Publisher identity, authors, dates, cross-version content equivalence and scientific truth are not authenticated.",
        }
        projected_source = {key: new_source.get(key) for key in (
            "id", "title", "year", "venue", "url", "origin", "availability", "metadata_status", "passages", "text_sha256")}
        projected_source["full_text"] = text
        projected_packet["sources"].append(projected_source)
        if len(json.dumps(projected_packet, ensure_ascii=False)) > MAX_PACKET_CHARS:
            projected_packet["sources"].pop()
            item.update(status="not_adopted_context_limit", error="Complete source packet would exceed its frozen review context limit; raw/text evidence retained outside the snapshot")
            save()
            continue
        for relative, key in ((raw_path, "raw"), (text_path, "text")):
            if (context / relative).exists():
                raise ValueError("New source would overwrite frozen source bytes")
            shutil.copyfile(_inside(workspace, item[key]), context / relative)
        catalog["sources"].append(new_source)
        new_ids.append(source_id)
    if not new_ids:
        return stop("source_packet_budget")
    catalog["review_enrichment"] = {"parent_catalog_sha256": packet["binding"]["frozen_evidence"]["catalog_sha256"],
                                    "parent_report_sha256": parent["parent_report_sha256"], "new_source_ids": new_ids}
    _dump(context / "catalog.json", catalog)
    load_catalog(staging)
    source_packet(staging)  # No truncation or dropped old sources to fit a packet.
    result = {"status": "enriched", "reason": "new_primary_full_text_requires_fresh_author_and_independent_review",
              "round": current["index"], "binding": parent, "new_source_ids": new_ids,
              "snapshot_scope": "public_task_metadata_and_complete_research_context_v1",
              "evidence_task": (round_dir / "evidence_task").relative_to(workspace).as_posix(),
              "evidence_task_sha256": task_digest(staging), "frozen_evidence": freeze_evidence(staging),
              "journal": journal_path.relative_to(workspace).as_posix()}
    current.update(status="publishing", staging=staging.relative_to(workspace).as_posix(), result=result)
    save()
    return _publish(workspace, current, save)


def _publish(workspace, current, save):
    destination = _inside(workspace, current["result"]["evidence_task"])
    if not destination.exists():
        staging = _inside(workspace, current["staging"])
        _tree(staging)
        if task_digest(staging) != current["result"]["evidence_task_sha256"]:
            raise ValueError("Staged evidence changed before publication")
        os.rename(staging, destination)
    validate_enrichment_result(workspace, current["result"])
    current.update(status="completed", completed_at_unix=time.time())
    save()
    return current["result"]


def load_enrichment_result(workspace: Path, journal_dir: Path, *, base_input_sha256: str,
                           candidate_packet: dict[str, Any] | None = None,
                           evidence_task: str | None = None,
                           expected_result_sha256: str | None = None) -> dict[str, Any]:
    """Read-only delivery verification of a sealed chain; never call a backend.

    The coordinator should bind content_hash(result) in its own frozen pointer
    and pass it as expected_result_sha256. Original inputs and review receipts
    remain required; a self-described result alone is not a trust root.
    """
    workspace = Path(os.path.abspath(workspace))
    directory = _inside(workspace, journal_dir)
    if not directory.is_relative_to(workspace / "novelty"):
        raise ValueError("Retrieval journal is outside novelty")
    base = _inside(workspace, "input_task")
    _tree(base)
    if task_digest(base) != base_input_sha256:
        raise ValueError("Frozen base input changed")
    journal = _read(directory / "journal.json")
    if (journal.get("schema_version") != 1 or journal["fixed"]["base_input_sha256"] != base_input_sha256
            or journal["fixed"]["candidate_packet_sha256"] != (content_hash(candidate_packet) if candidate_packet is not None else None)):
        raise ValueError("Retrieval chain belongs to a different base input or selected candidate")
    matches = [row for row in journal["rounds"] if row.get("result", {}).get("status") == "enriched"
               and (evidence_task is None or row["result"]["evidence_task"] == evidence_task)]
    if not matches:
        raise ValueError("No sealed enrichment result matches this request")
    chosen = matches[-1]
    for row in journal["rounds"][:chosen["index"]]:
        if row["status"] not in {"completed", "stopped"}:
            raise ValueError("Enrichment chain contains an unfinished parent")
        parent = row["parent"]
        task = _inside(workspace, parent["parent_task"])
        review = _inside(workspace, parent["review_cycle"])
        packet = extract_review_requests(task, review, candidate_packet=candidate_packet)
        expected = {**packet["binding"], "parent_task": task.relative_to(workspace).as_posix(),
                    "review_cycle": review.relative_to(workspace).as_posix(), "request_packet_sha256": content_hash(packet)}
        if parent != expected or row["requests"] != packet or row["result"]["binding"] != expected:
            raise ValueError("Parent request/report/receipt binding changed")
        _validate_saved_calls(workspace, row)
        validate_enrichment_result(workspace, row["result"])
    result = chosen["result"]
    if expected_result_sha256 is not None and content_hash(result) != expected_result_sha256:
        raise ValueError("Enrichment result differs from its coordinator binding")
    return copy.deepcopy(result)
