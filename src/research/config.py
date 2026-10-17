"""Small, explicit project configuration for the unified research entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

import yaml

from .providers import API_FORMATS, PROVIDERS


@dataclass
class ExternalSearchConfig:
    provider: str = "searxng"
    base_url: str = ""
    api_key_env: str = ""
    max_results: int = 5


@dataclass
class ModelSearchConfig:
    provider: str = "custom"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    api_format: str = ""
    mode: str = ""
    api_path: str = ""
    api_key_header: str = ""
    api_key_prefix: str = ""
    proxy_url: str = ""
    timeout_seconds: int = 180
    max_output_tokens: int = 6000
    max_search_calls: int = 4
    max_continuations: int = 2
    request_overrides: dict[str, Any] = field(default_factory=dict)
    external_search: ExternalSearchConfig = field(default_factory=ExternalSearchConfig)


@dataclass
class LiteratureConfig:
    queries: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    max_papers: int = 8
    results_per_query: int = 5
    fetch_full_text: bool = True
    timeout_seconds: int = 15
    proxy_fallback: str = "http://127.0.0.1:7890"
    model_search: ModelSearchConfig = field(default_factory=ModelSearchConfig)


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


def validate_model_config(cfg: ModelSearchConfig) -> None:
    if not isinstance(cfg.provider, str) or cfg.provider not in PROVIDERS:
        raise ValueError(f"Unknown model search provider; choose from {', '.join(PROVIDERS)}")
    for name in ("model", "base_url", "api_key_env", "api_format", "mode", "api_path",
                 "api_key_header", "api_key_prefix", "proxy_url"):
        if not isinstance(getattr(cfg, name), str) or "\n" in getattr(cfg, name) or "\r" in getattr(cfg, name):
            raise ValueError(f"model_search.{name} must be a single-line string")
    if cfg.api_format and cfg.api_format not in API_FORMATS:
        raise ValueError(f"api_format must be one of {sorted(API_FORMATS)}")
    if cfg.mode and cfg.mode not in {"native", "external"}:
        raise ValueError("model_search.mode must be native or external")
    for name, low, high in (("timeout_seconds", 1, 600), ("max_output_tokens", 256, 32768),
                            ("max_search_calls", 1, 8), ("max_continuations", 0, 3)):
        value = getattr(cfg, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"model_search.{name} must be an integer from {low} to {high}")
    ext = cfg.external_search
    if not isinstance(ext.provider, str) or ext.provider not in {"searxng", "tavily", "brave", "zhipu"}:
        raise ValueError("external_search.provider must be searxng, tavily, brave, or zhipu")
    for name in ("base_url", "api_key_env"):
        if not isinstance(getattr(ext, name), str):
            raise ValueError(f"external_search.{name} must be a string")
    if type(ext.max_results) is not int or not 1 <= ext.max_results <= 20:
        raise ValueError("external_search.max_results must be an integer from 1 to 20")
    for name in (cfg.api_key_env, ext.api_key_env):
        if name and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("api_key_env must be an environment variable name, not an API key")
    for url in (cfg.base_url, ext.base_url, cfg.proxy_url):
        if url:
            parsed = urlsplit(url)
            if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.fragment:
                raise ValueError("API/proxy URLs must be HTTP(S), without embedded credentials or fragments")
            if parsed.query:
                raise ValueError("Put API query parameters in api_path, not base_url")
    if cfg.api_path and (not cfg.api_path.startswith("/") or cfg.api_path.startswith("//")):
        raise ValueError("api_path must start with one slash and stay on the configured host")
    if any(key.casefold() in {"api_key", "api-key", "key", "token", "access_token"}
           for key in parse_qs(urlsplit(cfg.api_path).query)):
        raise ValueError("API keys must be configured through api_key_env, not URL query parameters")
    if not isinstance(cfg.request_overrides, dict):
        raise ValueError("request_overrides must be a mapping")
    blocked = {"model", "messages", "input", "contents", "stream", "headers", "api_key", "authorization", "password", "token"}
    def check_fields(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or key.casefold() in blocked:
                    raise ValueError("request_overrides cannot replace input/model/authentication fields")
                check_fields(child)
        elif isinstance(value, list):
            for child in value:
                check_fields(child)
        elif value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("request_overrides must contain JSON-compatible values")
    check_fields(cfg.request_overrides)


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
    model_search = dict(_mapping(literature.get("model_search", {}), set(ModelSearchConfig.__dataclass_fields__), "model_search"))
    external = _mapping(model_search.pop("external_search", {}), set(ExternalSearchConfig.__dataclass_fields__), "external_search")
    model = ModelSearchConfig(**model_search, external_search=ExternalSearchConfig(**external))
    validate_model_config(model)
    cfg = ResearchConfig(literature=LiteratureConfig(**{**literature, "model_search": model}), search=SearchConfig(**search))
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
        _mapping(source, {"id", "title", "year", "venue", "url", "path", "authors"}, "literature source")
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
        if "authors" in source and (not isinstance(source["authors"], list) or not source["authors"]
                                    or not all(isinstance(name, str) and 1 <= len(name.strip()) <= 300
                                               for name in source["authors"])):
            raise ValueError("Literature source authors must be a list of non-empty names")
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
