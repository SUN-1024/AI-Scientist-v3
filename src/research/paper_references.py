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
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import unicodedata
from urllib.parse import unquote, urlsplit

import yaml

from src.orchestrator.state import write_private_text
from .config import ExternalSearchConfig, LiteratureConfig, ModelSearchConfig, load_config, validate_model_config
from .evidence import load_catalog
from .literature import download
from .model_search import _identity, search_literature

_MAX_REFERENCE_SEARCH_ROUNDS = 3
_MAX_SEARCH_CANDIDATES = 20


def _runtime_reference_provider(cfg: LiteratureConfig) -> tuple[LiteratureConfig, dict[str, str] | None]:
    """Allow citation transport recovery without changing the frozen study."""
    configured = os.environ.get("AGSCI_PAPER_REFERENCE_PROVIDER_CONFIG", "").strip()
    if not configured:
        return cfg, None
    path = Path(configured).expanduser().resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError("Reference provider config exceeds 64 KiB")
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise ValueError("Invalid reference provider YAML") from None
    if not isinstance(payload, dict) or set(payload) != {"literature"}:
        raise ValueError("Reference provider config must contain only literature.model_search")
    literature = payload["literature"]
    if not isinstance(literature, dict) or set(literature) != {"model_search"}:
        raise ValueError("Reference provider config cannot change sources or study settings")
    values = literature["model_search"]
    if not isinstance(values, dict) or not values:
        raise ValueError("Reference provider config needs a non-empty model_search mapping")
    values = dict(values)
    external = values.pop("external_search", {})
    if not isinstance(external, dict):
        raise ValueError("Reference provider external_search must be a mapping")
    try:
        model = ModelSearchConfig(**values, external_search=ExternalSearchConfig(**external))
    except TypeError:
        raise ValueError("Reference provider config has unsupported model_search fields") from None
    validate_model_config(model)
    return replace(cfg, model_search=model), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


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


def _metadata_identities(metadata: dict[str, Any]) -> set[str]:
    """Count alternate URLs for the same source-verified paper only once."""
    identities = {"title:" + re.sub(r"\W+", "", str(metadata["title"]).casefold())}
    doi = str(metadata.get("doi") or "").strip().casefold()
    if doi:
        identities.add("doi:" + doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/"))
    return identities


def _adopted_url_identities(record: dict[str, Any]) -> set[str]:
    """Reserve both reviewed origins and landing aliases before any rejection."""
    from .review_retrieval import primary_source_identity

    identities = set()
    for field in ("url", "verified_origin_url", "origin"):
        url = str(record.get(field) or "")
        if url:
            identities.add(_identity(url))
            try:
                primary = primary_source_identity(url)
            except ValueError:
                # Existing local sources use filesystem paths as origins. Keep
                # the exact alias without interpreting it as a public URL.
                continue
            identities.add(f"primary:{primary['scheme']}:{primary['value']}")
    return identities


def _adopted_metadata(record: dict[str, Any], cfg: LiteratureConfig, destination: Path) -> dict[str, Any]:
    """Bind publisher metadata to the full text used by the selected review."""
    from .review_retrieval import compatible_primary_urls, primary_source_identity

    origin = str(record.get("verified_origin_url") or record.get("origin") or "")
    landing = str(record.get("url") or "")
    if not origin or not compatible_primary_urls(landing, origin):
        raise ValueError("Adopted source landing page does not match its verified full-text origin")
    expected = primary_source_identity(origin)
    # Never let suggested authors/path take the user-supplied metadata shortcut.
    query = {"title": record["title"], "url": origin if expected["scheme"] in {"arxiv", "doi"} else landing}
    metadata = source_metadata(query, cfg, destination)
    def normalize(value: Any) -> str:
        return "".join(char for char in unicodedata.normalize("NFKC", str(value)).casefold() if char.isalnum())

    if not normalize(record["title"]) or normalize(record["title"]) != normalize(metadata.get("title", "")):
        raise ValueError("Adopted source citation title does not match the reviewed full text")
    if expected["scheme"] != "url":
        actual = [primary_source_identity(str(metadata.get("url") or ""))]
        if metadata.get("doi"):
            doi = str(metadata["doi"])
            actual.append(primary_source_identity(doi if doi.startswith("http") else "https://doi.org/" + doi))
        matching = [identity for identity in actual if identity["scheme"] == expected["scheme"]]
        if not matching or any(identity["value"] != expected["value"] for identity in matching):
            raise ValueError("Adopted source citation identifier does not match its verified full-text origin")
    metadata["reviewed_source_id"] = record["id"]
    metadata["reviewed_origin_url"] = origin
    return metadata


def _supplementary_queries(queries: list[str], checked: list[str], *, remaining: int) -> list[str]:
    # Keep repair context bounded even when supplied sources have very long URLs.
    guidance = (f"Find at least {remaining} additional distinct papers on the same topics. "
                "Prefer primary pages with title, author and date metadata. "
                "Do not repeat checked papers or their preprint/publisher versions. Checked sources: ")
    for identity in reversed(checked):
        addition = json.dumps(identity, ensure_ascii=False) + "; "
        if len(guidance) + len(addition) <= 500:
            guidance += addition
    return [*queries[:7], guidance]


def _selected_reference_sources(paper_data: dict[str, Any] | None,
                                catalog: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Resolve adopted sources only through the current host-verified selection."""
    if paper_data is None:
        return [], None
    saved = paper_data.get("selected_novelty_review")
    provenance = saved.get("provenance") if isinstance(saved, dict) else None
    if not isinstance(provenance, dict) or "evidence_reference" not in provenance:
        return [], None
    from .evidence import verify_selected_novelty_for_paper
    from .harness_evidence import build_candidate_packet
    from .harness_state import read_json, sha256
    from .novelty import content_hash
    from .review_enrichment import resolve_selected_evidence_task

    verified = verify_selected_novelty_for_paper(paper_data)
    if not isinstance(verified, dict) or verified.get("verification_status") != "host_verified" or verified != saved:
        raise ValueError("Selected bibliography evidence is stale or not host verified")
    provenance = verified["provenance"]
    location = provenance.get("managed_workspace")
    if not isinstance(location, str) or not Path(location).is_absolute():
        raise ValueError("Selected bibliography evidence has no managed workspace")
    managed = Path(location)
    packet = build_candidate_packet(managed)
    if content_hash(packet) != verified.get("candidate_packet_sha256"):
        raise ValueError("Selected bibliography candidate binding changed")
    pointer_path = managed / "novelty/latest.json"
    pointer = read_json(pointer_path)
    if (sha256(pointer_path) != provenance.get("pointer_sha256")
            or pointer.get("evidence_reference") != provenance.get("evidence_reference")):
        raise ValueError("Selected bibliography evidence pointer changed")
    evidence_task = resolve_selected_evidence_task(managed, pointer, candidate_packet=packet)
    if not evidence_task.is_relative_to(managed):
        raise ValueError("Selected bibliography evidence escaped the managed workspace")
    enriched = load_catalog(evidence_task)
    if enriched is None:
        raise ValueError("Selected bibliography evidence has no validated catalog")
    original = {row["id"]: row for row in catalog["sources"]}
    adopted = [row for row in enriched["sources"] if row != original.get(row["id"])]
    # These remain discovery candidates. source_metadata must still verify all
    # bibliography fields; saved model metadata cannot fill missing authors.
    binding = {
        "citation_identity_policy": "reviewed_fulltext_v1",
        "evidence_reference": provenance["evidence_reference"],
        "candidate_packet_sha256": verified["candidate_packet_sha256"],
        "selected_projection_sha256": content_hash(verified),
        "pointer_sha256": provenance["pointer_sha256"],
        "catalog_sha256": sha256(evidence_task / "research_context/catalog.json"),
        "adopted_source_ids": [row["id"] for row in adopted],
    }
    return adopted, binding


def write_references_for_plan(task_dir: Path, output_dir: Path, *, queries: list[str], min_refs: int,
                              max_refs: int, plan_metadata: dict[str, Any], context: dict[str, Any],
                              paper_data: dict[str, Any] | None = None) -> int:
    cfg = load_config(task_dir).literature
    catalog = load_catalog(task_dir) or {"sources": [], "offline": False}
    runtime_provider = None
    if not catalog.get("offline"):
        cfg, runtime_provider = _runtime_reference_provider(cfg)
    if not 1 <= min_refs <= max_refs:
        raise ValueError("Invalid paper reference count limits")
    adopted, selected_binding = _selected_reference_sources(paper_data, catalog)
    fingerprint_inputs = {
        "config": asdict(cfg), "catalog": catalog, "plan": plan_metadata,
        "min_refs": min_refs, "max_refs": max_refs, "context": context, "queries": queries,
        "retrieval_policy": "bounded_supplement_v1",
    }
    if selected_binding is not None:
        fingerprint_inputs["selected_literature_evidence"] = selected_binding
    if runtime_provider is not None:
        fingerprint_inputs["runtime_reference_provider"] = runtime_provider
    fingerprint = hashlib.sha256(json.dumps(fingerprint_inputs, sort_keys=True).encode()).hexdigest()[:24]
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
        cached_provenance = json.loads(cached["references_provenance.json"])
        if selected_binding is not None and cached_provenance.get("selected_literature_evidence") != selected_binding:
            raise ValueError("Cached selected bibliography evidence binding changed")
        for row in cached_provenance["references"]:
            path = output_dir / row["metadata_file"]
            if (path.is_symlink() or not path.resolve().is_relative_to(output_dir.resolve())
                    or hashlib.sha256(path.read_bytes()).hexdigest() != row["metadata_sha256"]):
                raise ValueError("Cached source citation metadata changed")
        for relative, expected_hash in cached_receipt.get("search_artifacts", {}).items():
            path = output_dir / relative
            if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash):
                raise ValueError("Cached reference search artifact changed")
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
        candidates.extend(catalog["sources"])
    references: list[dict[str, Any]] = []
    bibtex: list[str] = []
    seen: set[str] = set()
    checked: list[str] = []
    seen_keys: set[str] = set()
    seen_metadata: set[str] = set()
    adopted_identities: set[str] = set()

    def add_candidates(batch: list[dict[str, Any]], batch_queries: list[str], *, reviewed: bool = False) -> int:
        considered = 0
        for candidate in batch:
            if len(references) >= max_refs:
                break
            considered += 1
            identity = _identity(str(candidate.get("url") or "")) or str(candidate.get("path") or candidate["title"])
            if reviewed:
                adopted_identities.update(_adopted_url_identities(candidate))
            elif adopted_identities and _adopted_url_identities(candidate) & adopted_identities:
                continue
            if identity in seen:
                continue
            seen.add(identity)
            checked.append(identity)
            try:
                if offline and not (candidate.get("path") and candidate.get("authors")):
                    raise ValueError("Offline references require local sources with user-supplied authors, title and year")
                metadata = (_adopted_metadata(candidate, cfg, metadata_dir) if reviewed
                            else source_metadata(candidate, cfg, metadata_dir))
                key, entry = bibliography_entry(metadata)
                identities = _metadata_identities(metadata)
                if key in seen_keys or identities & seen_metadata:
                    seen_metadata.update(identities)
                    continue
                seen_keys.add(key)
                seen_metadata.update(identities)
                metadata["metadata_file"] = str((metadata_dir / metadata["metadata_file"]).relative_to(output_dir))
                references.append({
                    "id": "ref.model_search." + key, "paper_id": str(candidate.get("url") or candidate.get("path")),
                    "citation_key": key, "query": " | ".join(batch_queries), "source_url": metadata.get("url") or str(candidate.get("path")),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(), **metadata,
                })
                bibtex.append(entry)
            except (ValueError, OSError, TypeError, AttributeError, KeyError) as exc:
                errors.append({"stage": "citation_metadata", "source": str(candidate.get("url") or candidate.get("path")), "error": str(exc)})
        return considered

    add_candidates(adopted, queries, reviewed=True)
    add_candidates(candidates, queries)
    rounds: list[dict[str, Any]] = []
    search_artifacts: dict[str, str] = {}
    stop_reason = "offline_sources_exhausted" if offline else "request_budget_exhausted"
    for index in range(0 if offline else _MAX_REFERENCE_SEARCH_ROUNDS):
        if len(references) >= min_refs:
            break
        remaining = min_refs - len(references)
        round_queries = queries if index == 0 else _supplementary_queries(queries, checked, remaining=remaining)
        # Some candidates overlap or lack usable metadata. Permit bounded excess
        # retrieval while capping the published bibliography at max_refs.
        limit = min(_MAX_SEARCH_CANDIDATES, max_refs, max(2 * remaining, cfg.results_per_query))
        search_cfg = replace(cfg, queries=round_queries, max_papers=limit, fetch_full_text=False)
        round_dir = attempt / f"search_round_{index + 1:02d}"
        round_dir.mkdir()
        retrieved, search_errors = search_literature(search_cfg, round_dir)
        errors.extend(search_errors)
        before = len(references)
        considered = add_candidates(retrieved[:limit], round_queries)
        trace_path = round_dir / "model_search" / "trace.json"
        for path in sorted((round_dir / "model_search").glob("*")):
            if path.is_symlink():
                raise ValueError("Reference search artifact cannot be a symlink")
            if path.is_file():
                search_artifacts[str(path.relative_to(output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rounds.append({
            "round": index + 1, "queries": round_queries, "requested_limit": limit,
            "candidates_returned": len(retrieved), "candidates_checked": considered,
            "new_verified_references": len(references) - before, "verified_total": len(references),
            "search_trace": str(trace_path.relative_to(output_dir)) if trace_path.is_file() else None,
        })
        if not retrieved:
            stop_reason = "search_unavailable" if search_errors else "no_candidates"
            break
    if len(references) >= min_refs:
        stop_reason = "minimum_met"
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(), **context,
        "provider": "configured_model_search", "model_provider": cfg.model_search.provider,
        "queries": queries, "reference_query_plan": plan_metadata, "count": len(references),
        "references": references, "retrieval_errors": errors, "offline": offline,
        "status": "ready" if len(references) >= min_refs else "exhausted", "stop_reason": stop_reason,
        "search_round_limit": _MAX_REFERENCE_SEARCH_ROUNDS, "search_rounds": rounds,
        "search_trace": rounds[0]["search_trace"] if rounds else None,
    }
    if selected_binding is not None:
        summary["selected_literature_evidence"] = selected_binding
    if runtime_provider is not None:
        summary["runtime_reference_provider"] = runtime_provider
    write_private_text(attempt / "result.json", json.dumps(summary, ensure_ascii=False, indent=2))
    if len(references) < min_refs:
        raise ValueError(f"Only {len(references)}/{min_refs} references have source metadata; stopped after {len(rounds)} search rounds ({stop_reason}); inspect {attempt}. No legacy API fallback was used.")
    contents = {"references.bib": "\n\n".join(bibtex) + "\n",
                "references_provenance.json": json.dumps(summary, ensure_ascii=False, indent=2)}
    receipt: dict[str, Any] = {"count": len(references), "search_artifacts": search_artifacts}
    for name, content in contents.items():
        write_private_text(root / name, content)
        receipt[name] = hashlib.sha256(content.encode()).hexdigest()
    write_private_text(receipt_path, json.dumps(receipt))
    for name, content in contents.items():
        write_private_text(output_dir / name, content)
    return len(references)
