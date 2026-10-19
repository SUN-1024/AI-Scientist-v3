"""Retrieve primary paper records and keep inspectable text evidence.

Availability is not a claim that a model read or understood an entire paper.
Failed requests stay visible and never become a successful novelty check.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LiteratureConfig
from .model_search import search_literature

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
CONTEXT_DIRECTORY = "research_context"


def _public_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username:
        raise ValueError("Paper URLs must be public HTTP(S) URLs without embedded credentials")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    if any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise ValueError("Paper URL resolves to a non-public address")


class _PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_url(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("X-api-key")
        return redirected


def download(url: str, cfg: LiteratureConfig, *, accept: str | None = None) -> bytes:
    _public_url(url)
    headers = {"User-Agent": "AgenticScientist-Research/0.1"}
    if accept:
        headers["Accept"] = accept
    proxies: list[dict[str, str] | None] = [None]
    if cfg.proxy_fallback:
        proxies.append({"https": cfg.proxy_fallback, "http": cfg.proxy_fallback})
    last_error: Exception | None = None
    for proxy in proxies:
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler(proxy), _PublicRedirect(),
            )
            with opener.open(urllib.request.Request(url, headers=headers), timeout=cfg.timeout_seconds) as response:
                body = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(body) > MAX_DOWNLOAD_BYTES:
                raise ValueError("Paper response exceeds 20 MiB")
            return body
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
    raise ValueError(f"Retrieval failed for {url}: {last_error}")


def _extract_text(body: bytes, *, pdf: bool,
                  provenance: dict[str, Any] | None = None) -> tuple[str, str]:
    if pdf:
        from .pdf_text import extract_pdf

        text, availability, metadata = extract_pdf(body, max_characters=MAX_TEXT_CHARS)
        if provenance is not None:
            provenance.update(metadata)
        return text, availability
    else:
        text = body.decode("utf-8")
        partial = len(text) > MAX_TEXT_CHARS
    if not text.strip():
        raise ValueError("No extractable text; scanned PDFs need OCR before use")
    return text[:MAX_TEXT_CHARS], "full_text_partial" if partial else "full_text"


def _source_id(source: dict[str, Any]) -> str:
    provided = source.get("id")
    if provided is not None:
        if not isinstance(provided, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", provided):
            raise ValueError("Source IDs must use 1-80 letters, digits, underscores, or hyphens")
        return provided
    identity = str(source.get("paperId") or source.get("url") or source["title"])
    return "paper_" + hashlib.sha256(identity.encode()).hexdigest()[:12]


def passages(text: str, source_id: str, queries: list[str]) -> list[dict[str, str]]:
    terms = set(re.findall(r"[a-z]{4,}", " ".join(queries).lower()))
    terms |= {"method", "algorithm", "experiment", "limitation", "baseline", "results"}
    chunks = []
    # Offsets identify the exact substring in the saved text, even after selection.
    for start in range(0, len(text), 900):
        chunk = text[start:start + 900]
        rank = sum(chunk.lower().count(term) for term in terms)
        chunks.append((rank, start, chunk))
    chosen = sorted(sorted(chunks, key=lambda row: (-row[0], row[1]))[:3], key=lambda row: row[1])
    return [
        {"id": f"{source_id}:chars_{start}_{start + len(chunk)}",
         "locator": f"characters {start}:{start + len(chunk)}", "text": chunk}
        for _, start, chunk in chosen
    ]


def prepare_literature(task_dir: Path, cfg: LiteratureConfig, *, offline: bool = False) -> dict[str, Any]:
    output = task_dir / CONTEXT_DIRECTORY
    if output.exists():
        raise ValueError("Prepared literature already exists; create a new research workspace to refresh it")
    output.mkdir()
    text_dir = output / "text"
    text_dir.mkdir()
    errors: list[dict[str, str]] = []
    records: list[dict[str, Any]] = [dict(source) for source in cfg.sources]
    if not offline and cfg.queries:
        retrieved, search_errors = search_literature(cfg, output)
        records.extend(retrieved)
        errors.extend(search_errors)
    elif cfg.queries:
        errors.append({"stage": "search", "error": "Online searches skipped in offline mode"})

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_titles: set[str] = set()
    for record in records:
        source_id = _source_id(record)
        title_key = re.sub(r"\W+", "", record["title"].casefold())
        if source_id in seen or title_key in seen_titles:
            continue
        seen.add(source_id)
        seen_titles.add(title_key)
        text = ""
        availability = "metadata"
        raw_hash = None
        extraction: dict[str, Any] = {}
        origin = record.get("url") or "local source"
        if record.get("path"):
            path = task_dir / record["path"]
            if not path.resolve().is_relative_to(task_dir.resolve()) or path.is_symlink():
                raise ValueError("Local source escaped the prepared task")
            if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
                raise ValueError("Local sources must be UTF-8 .md/.txt files or PDFs")
            if path.stat().st_size > MAX_DOWNLOAD_BYTES:
                raise ValueError("Local source exceeds 20 MiB")
            body = path.read_bytes()
            text, availability = _extract_text(body, pdf=path.suffix.lower() == ".pdf",
                                               provenance=extraction)
            raw_hash = hashlib.sha256(body).hexdigest()
            origin = record["path"]
        elif not offline and cfg.fetch_full_text:
            access = record.get("openAccessPdf")
            pdf_url = access.get("url") if isinstance(access, dict) else None
            if pdf_url:
                try:
                    body = download(pdf_url, cfg)
                    text, availability = _extract_text(body, pdf=True, provenance=extraction)
                    raw_hash = hashlib.sha256(body).hexdigest()
                    origin = pdf_url
                except Exception as exc:
                    errors.append({"stage": "full_text", "source_id": source_id, "error": str(exc)})
        if not text and isinstance(record.get("abstract"), str) and record["abstract"].strip():
            text = record["abstract"]
            availability = "abstract"
        text_path = f"text/{source_id}.txt" if text else None
        if text_path:
            (output / text_path).write_text(text, encoding="utf-8")
        sources.append({
            "id": source_id, "title": record["title"], "year": record.get("year"),
            "venue": record.get("venue"), "url": record.get("url"),
            "authors": record.get("authors"),
            "origin": origin, "availability": availability,
            "metadata_status": record.get("metadata_status", "user_supplied"),
            "model_summary": record.get("model_summary"), "search_sources": record.get("search_sources", []),
            "text_path": text_path, "text_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
            "download_sha256": raw_hash, "passages": passages(text, source_id, cfg.queries),
            **({"extraction": extraction} if extraction else {}),
        })
        if len(sources) >= cfg.max_papers:
            break
    catalog = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "queries": cfg.queries, "offline": offline, "sources": sources, "retrieval_errors": errors,
        "status": "ready" if sources and not errors else "partial" if sources else "unavailable",
        "novelty_status": "not_assessed",
        "model_search_trace": "model_search/trace.json" if not offline and cfg.queries else None,
    }
    (output / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Literature sources", "", "Retrieval does not establish novelty. See the cited text for each comparison.", ""]
    for source in sources:
        lines += [f"- {source['id']}: {source['title']} ({source['availability']}); {source['url'] or source['origin']}"]
    if errors:
        lines += ["", "## Incomplete retrieval", "", *[f"- {error['stage']}: {error['error']}" for error in errors]]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return catalog
