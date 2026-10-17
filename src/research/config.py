"""Small, explicit project configuration for the unified research entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass
class LiteratureConfig:
    queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    max_papers: int = 8
    results_per_query: int = 5
    fetch_full_text: bool = True
    timeout_seconds: int = 15
    proxy_fallback: str = "http://127.0.0.1:7890"


@dataclass
class SearchConfig:
    strategy: str = "tree"
    width: int = 4
    explore_every: int = 3


@dataclass
class ResearchConfig:
    schema_version: int = 1
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def _mapping(value: object, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise ValueError(f"{label} must be a mapping with keys {sorted(allowed)}")
    return value


def load_config(task_dir: Path) -> ResearchConfig:
    path = task_dir / "research.yaml"
    if not path.is_file():
        raise ValueError(f"Missing {path}; use 'agsci research init' or copy the documented config")
    raw = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        {"schema_version", "literature", "search"}, "research.yaml",
    )
    if type(raw.get("schema_version", 1)) is not int or raw.get("schema_version", 1) != 1:
        raise ValueError("research.yaml schema_version must be 1")
    literature = _mapping(
        raw.get("literature", {}), set(LiteratureConfig.__dataclass_fields__), "literature",
    )
    search = _mapping(raw.get("search", {}), set(SearchConfig.__dataclass_fields__), "search")
    cfg = ResearchConfig(literature=LiteratureConfig(**literature), search=SearchConfig(**search))
    lit = cfg.literature
    if not isinstance(lit.queries, list) or not all(
        isinstance(query, str) and 1 <= len(query.strip()) <= 500 for query in lit.queries
    ) or len(lit.queries) > 8:
        raise ValueError("literature.queries must contain at most 8 non-empty queries (<=500 chars)")
    if not isinstance(lit.sources, list):
        raise ValueError("literature.sources must be a list")
    source_ids: set[str] = set()
    source_titles: set[str] = set()
    for source in lit.sources:
        _mapping(source, {"id", "title", "year", "venue", "url", "path"}, "literature source")
        if not isinstance(source.get("title"), str) or not source["title"].strip():
            raise ValueError("Each local literature source needs a title")
        if not isinstance(source.get("path"), str) or not source["path"].strip():
            raise ValueError("Each local literature source needs a project-relative path")
        relative = Path(source["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Literature source paths must stay inside the task directory")
        resolved = (task_dir / relative).resolve()
        if not resolved.is_relative_to(task_dir.resolve()) or not resolved.is_file():
            raise ValueError(f"Literature source is unavailable: {relative}")
        for key in ("id", "venue", "url"):
            if key in source and not isinstance(source[key], str):
                raise ValueError(f"Literature source {key} must be a string")
        if "year" in source and (type(source["year"]) is not int or source["year"] < 0):
            raise ValueError("Literature source year must be a non-negative integer")
        identifier = source.get("id")
        if identifier is not None:
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", identifier):
                raise ValueError("Source IDs must use 1-80 letters, digits, underscores, or hyphens")
            if identifier in source_ids:
                raise ValueError(f"Duplicate local source ID: {identifier}")
            source_ids.add(identifier)
        title = re.sub(r"\W+", "", source["title"].casefold())
        if title in source_titles:
            raise ValueError(f"Duplicate local source title: {source['title']}")
        source_titles.add(title)
    for name, value, upper in (
        ("max_papers", lit.max_papers, 20),
        ("results_per_query", lit.results_per_query, 20),
        ("timeout_seconds", lit.timeout_seconds, 60),
    ):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"literature.{name} must be an integer from 1 to {upper}")
    if len(lit.sources) > lit.max_papers:
        raise ValueError("Increase max_papers to include every explicitly supplied source")
    if type(lit.fetch_full_text) is not bool or not isinstance(lit.proxy_fallback, str):
        raise ValueError("Invalid literature fetch_full_text or proxy_fallback")
    if not isinstance(cfg.search.strategy, str) or cfg.search.strategy not in {"best", "tree"}:
        raise ValueError("search.strategy must be best or tree")
    for name in ("width", "explore_every"):
        value = getattr(cfg.search, name)
        if type(value) is not int or not 2 <= value <= 100:
            raise ValueError(f"search.{name} must be an integer from 2 to 100")
    return cfg


def write_config(path: Path, cfg: ResearchConfig) -> None:
    path.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True), encoding="utf-8")
