"""Route unified paper citations through model search and source metadata.

The model finds candidates. Authors, title and date come from publisher/arXiv
citation tags, DOI content negotiation, or explicit local user metadata.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import unquote, urlsplit

from src.orchestrator.state import write_private_text
from .config import LiteratureConfig, load_config
from .evidence import load_catalog
from .literature import download
from .model_search import _identity, search_literature


class CitationTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = dict(attrs)
        name = (values.get("name") or "").lower()
        content = values.get("content")
        if name.startswith("citation_") and content:
            self.tags.setdefault(name, []).append(content.strip())


def _first(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).strip() if value is not None else ""


def _year(value: Any) -> int | None:
    match = re.search(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b", str(value))
    return int(match[0]) if match else None


def _doi_metadata(url: str, cfg: LiteratureConfig) -> tuple[dict[str, Any], bytes]:
    raw = download(url, cfg, accept="application/vnd.citationstyles.csl+json")
    payload = json.loads(raw)
    authors = []
    for author in payload.get("author", []):
        name = author.get("literal") or ", ".join(filter(None, [author.get("family"), author.get("given")]))
        if name:
            authors.append(name)
    date_parts = payload.get("issued", {}).get("date-parts", [])
    metadata = {
        "title": _first(payload.get("title")), "authors": authors,
        "year": _year(date_parts[0][0]) if date_parts and date_parts[0] else None,
        "venue": _first(payload.get("container-title")),
        "doi": payload.get("DOI"), "url": url,
        "entry_type": "article" if payload.get("type") == "article-journal" else "inproceedings" if payload.get("type") == "paper-conference" else "misc",
        "volume": payload.get("volume"), "number": payload.get("issue"), "pages": payload.get("page"),
    }
    return metadata, raw


def source_metadata(record: dict[str, Any], cfg: LiteratureConfig, destination: Path) -> dict[str, Any]:
    url = str(record.get("url") or "")
    if record.get("path") and record.get("authors"):
        metadata = {key: record.get(key) for key in ("title", "authors", "year", "venue", "url")}
        metadata.update(entry_type="misc", metadata_status="user_supplied", metadata_source=record["path"])
        raw = json.dumps(metadata, ensure_ascii=False).encode()
        suffix = ".json"
    else:
        parsed = urlsplit(url)
        if parsed.hostname in {"doi.org", "dx.doi.org"}:
            metadata, raw = _doi_metadata(url, cfg)
            suffix = ".json"
        else:
            if parsed.hostname in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
                identifier = re.sub(r"^/(abs|pdf|html)/", "", parsed.path).removesuffix(".pdf")
                if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[\w.-]+/\d{7})(?:v\d+)?", identifier):
                    raise ValueError("Unrecognized arXiv paper URL")
                url = "https://arxiv.org/abs/" + identifier
            raw = download(url, cfg, accept="text/html")
            parser = CitationTags()
            parser.feed(raw.decode("utf-8", errors="replace"))
            tags = parser.tags
            metadata = {
                "title": _first(tags.get("citation_title")), "authors": tags.get("citation_author", []),
                "year": _year(_first(tags.get("citation_publication_date") or tags.get("citation_date") or tags.get("citation_year"))),
                "venue": _first(tags.get("citation_journal_title") or tags.get("citation_conference_title")),
                "doi": _first(tags.get("citation_doi")), "url": url,
                "entry_type": "article" if tags.get("citation_journal_title") else "inproceedings" if tags.get("citation_conference_title") else "misc",
            }
            suffix = ".html"
            if not (metadata["title"] and metadata["authors"] and metadata["year"]) and metadata["doi"]:
                doi = unquote(str(metadata["doi"])).removeprefix("https://doi.org/").strip()
                if not re.fullmatch(r"10\.\d{4,9}/[^\s?#]+", doi):
                    raise ValueError("Invalid DOI in source citation metadata")
                url = "https://doi.org/" + doi
                metadata, raw = _doi_metadata(url, cfg)
                suffix = ".json"
        metadata.update(metadata_status="source_metadata", metadata_source=url)
    if not metadata.get("title") or not metadata.get("authors") or not metadata.get("year"):
        raise ValueError("Source does not provide title, authors and publication year; model fields cannot fill these gaps")
    body_hash = hashlib.sha256(raw).hexdigest()
    filename = body_hash + suffix
    write_private_text(destination / filename, raw.decode("utf-8", errors="replace"))
    # Hash the actual UTF-8 text saved to disk, including any decoding replacement.
    metadata["metadata_file"] = filename
    metadata["metadata_sha256"] = hashlib.sha256((destination / filename).read_bytes()).hexdigest()
    return metadata


def _tex(value: Any) -> str:
    value = " ".join(str(value).split())
    escapes = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "#": r"\#", "$": r"\$",
               "%": r"\%", "&": r"\&", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in value)


def bibliography_entry(metadata: dict[str, Any]) -> tuple[str, str]:
    identity = metadata.get("doi") or metadata.get("url") or metadata["title"]
    citation_key = "Ref" + hashlib.sha256(str(identity).encode()).hexdigest()[:12]
    fields = {"title": metadata["title"], "author": " and ".join(metadata["authors"]), "year": metadata["year"]}
    kind = metadata.get("entry_type", "misc")
    if kind not in {"article", "inproceedings", "misc"}:
        kind = "misc"
    venue_key = "journal" if kind == "article" else "booktitle" if kind == "inproceedings" else "howpublished"
    fields[venue_key] = metadata.get("venue")
    fields.update({key: metadata.get(key) for key in ("url", "doi", "volume", "number", "pages")})
    body = ",\n".join(f"  {key} = {{{_tex(value)}}}" for key, value in fields.items() if value)
    return citation_key, f"@{kind}{{{citation_key},\n{body}\n}}"


def write_references_for_plan(task_dir: Path, output_dir: Path, *, queries: list[str], min_refs: int,
                              max_refs: int, plan_metadata: dict[str, Any], context: dict[str, Any]) -> int:
    cfg = load_config(task_dir).literature
    catalog = load_catalog(task_dir) or {"sources": [], "offline": False}
    if not 1 <= min_refs <= max_refs:
        raise ValueError("Invalid paper reference count limits")
    fingerprint = hashlib.sha256(json.dumps({
        "config": asdict(cfg), "catalog": catalog, "plan": plan_metadata,
        "min_refs": min_refs, "max_refs": max_refs, "context": context,
    }, sort_keys=True).encode()).hexdigest()[:24]
    root = output_dir / "reference_search" / fingerprint
    if root.is_symlink() or (root.parent.exists() and root.parent.is_symlink()):
        raise ValueError("Reference cache cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipt.json"
    if receipt_path.is_symlink():
        raise ValueError("Reference cache receipt cannot be a symlink")
    if receipt_path.is_file():
        cached_receipt = json.loads(receipt_path.read_text())
        cached = {}
        for name in ("references.bib", "references_provenance.json"):
            path = root / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != cached_receipt.get(name):
                raise ValueError("Cached reference artifact changed")
            cached[name] = path.read_text()
        for row in json.loads(cached["references_provenance.json"])["references"]:
            path = output_dir / row["metadata_file"]
            if (path.is_symlink() or not path.resolve().is_relative_to(output_dir.resolve())
                    or hashlib.sha256(path.read_bytes()).hexdigest() != row["metadata_sha256"]):
                raise ValueError("Cached source citation metadata changed")
        for name, content in cached.items():
            write_private_text(output_dir / name, content)
        return int(cached_receipt["count"])
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=root))
    metadata_dir = attempt / "source_metadata"
    metadata_dir.mkdir()
    errors: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = list(cfg.sources)
    offline = bool(catalog.get("offline"))
    if not offline:
        search_cfg = replace(cfg, queries=queries, max_papers=min(max_refs, 20), fetch_full_text=False)
        retrieved, errors = search_literature(search_cfg, attempt)
        candidates.extend(retrieved)
        candidates.extend(catalog["sources"])
    references = []
    bibtex = []
    seen: set[str] = set()
    seen_keys: set[str] = set()
    for candidate in candidates:
        identity = _identity(str(candidate.get("url") or "")) or str(candidate.get("path") or candidate["title"])
        if identity in seen:
            continue
        seen.add(identity)
        try:
            if offline and not (candidate.get("path") and candidate.get("authors")):
                raise ValueError("Offline references require local sources with user-supplied authors, title and year")
            metadata = source_metadata(candidate, cfg, metadata_dir)
            key, entry = bibliography_entry(metadata)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            metadata["metadata_file"] = str((metadata_dir / metadata["metadata_file"]).relative_to(output_dir))
            references.append({
                "id": "ref.model_search." + key, "paper_id": str(candidate.get("url") or candidate.get("path")),
                "citation_key": key, "query": " | ".join(queries), "source_url": metadata.get("url") or str(candidate.get("path")),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), **metadata,
            })
            bibtex.append(entry)
        except (ValueError, OSError, TypeError, AttributeError, KeyError) as exc:
            errors.append({"stage": "citation_metadata", "source": str(candidate.get("url") or candidate.get("path")), "error": str(exc)})
        if len(references) >= max_refs:
            break
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(), **context,
        "provider": "configured_model_search", "model_provider": cfg.model_search.provider,
        "queries": queries, "reference_query_plan": plan_metadata, "count": len(references),
        "references": references, "retrieval_errors": errors, "offline": offline,
        "search_trace": str((attempt / "model_search/trace.json").relative_to(output_dir)) if not offline else None,
    }
    write_private_text(attempt / "result.json", json.dumps(summary, ensure_ascii=False, indent=2))
    if len(references) < min_refs:
        raise ValueError(f"Only {len(references)}/{min_refs} references have source metadata; inspect {attempt}. No legacy API fallback was used.")
    contents = {"references.bib": "\n\n".join(bibtex) + "\n",
                "references_provenance.json": json.dumps(summary, ensure_ascii=False, indent=2)}
    receipt: dict[str, Any] = {"count": len(references)}
    for name, content in contents.items():
        write_private_text(root / name, content)
        receipt[name] = hashlib.sha256(content.encode()).hexdigest()
    write_private_text(receipt_path, json.dumps(receipt))
    for name, content in contents.items():
        write_private_text(output_dir / name, content)
    return len(references)
