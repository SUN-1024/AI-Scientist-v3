"""Unified research entry point, preserving the existing V3 execution engine."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import ResearchConfig, load_config, write_config
from .preflight import ensure_execution_ready
from .providers import PROVIDERS
from .workspace import copy_task, execution_overrides, prepare, record_execution, resume, write_report

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = {
    "circpack": "CPU circle packing; a shipped evaluator and baseline",
    "grpo": "GPU language-model optimization; stage model resources before execution",
}
RESERVED_OVERRIDES = {"task.dir", "paths.run_dir", "investigation.paths.task_dir", "investigation.paths.results_dir"}


def _validate_overrides(overrides: list[str]) -> None:
    for value in overrides:
        if "=" not in value or value.startswith("-"):
            raise ValueError(f"Expected a pipeline key=value override, got {value!r}")
        key = value.split("=", 1)[0].lstrip("+~")
        if key in RESERVED_OVERRIDES or key.startswith("investigation.execution.search_"):
            raise ValueError(f"{key} is managed by the research workspace/research.yaml")


def pipeline_args(workspace: Path, cfg: ResearchConfig, overrides: list[str]) -> list[str]:
    return [
        f"task.dir={json.dumps(str(workspace / 'input_task'))}",
        f"paths.run_dir={json.dumps(str(workspace / 'run'))}",
        f"investigation.execution.search_strategy={cfg.search.strategy}",
        f"investigation.execution.search_width={cfg.search.width}",
        f"investigation.execution.search_explore_every={cfg.search.explore_every}",
        *overrides,
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agsci research", description="V1 literature/templates + V2 branching + V3 verified execution")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("templates", help="List reusable baseline/evaluator templates")
    providers = sub.add_parser("providers", help="List model providers or print a model_search YAML template")
    providers.add_argument("--config", choices=tuple(PROVIDERS))
    init = sub.add_parser("init", help="Create a research project without overwriting existing files")
    init.add_argument("project", type=Path)
    source = init.add_mutually_exclusive_group()
    source.add_argument("--template", choices=tuple(TEMPLATES))
    source.add_argument("--brief", type=Path, help="Create from a research brief; V3 task drafting is experimental")
    init.add_argument("--query", action="append", default=[], help="Repeat for focused literature searches")
    init.add_argument("--provider", choices=tuple(PROVIDERS), default="custom")
    init.add_argument("--model", default="", help="Model ID from your chosen provider")
    for name in ("prepare", "run", "harness"):
        command = sub.add_parser(name, help="Prepare literature" if name == "prepare" else "Prepare literature and execute the V1–V3 fusion workflow")
        command.add_argument("task", type=Path, nargs="?")
        command.add_argument("--workspace", type=Path)
        command.add_argument("--resume", type=Path, help="Reuse immutable inputs and continue the saved run")
        command.add_argument("--offline", action="store_true")
        command.add_argument("--dry-run", action="store_true", help="Validate/preview without files, network, agents, or training")
        if name == "harness":
            command.add_argument("--quality-target", choices=("high_impact", "development"), default=None)

    report = sub.add_parser("report", help="Inspect the saved run summary")
    report.add_argument("workspace", type=Path)
    critique = sub.add_parser("critique", help="Run independent model reviews of a manuscript draft")
    critique.add_argument("manuscript", type=Path, help="Path to .tex, .pdf or .md")
    critique.add_argument("--output", type=Path, required=True, help="Directory for receipts and the report")
    critique.add_argument("--model", required=True)
    critique.add_argument("--profile", required=True)
    critique.add_argument("--reviewers", type=int, default=2)
    critique.add_argument("--reasoning-effort", default="high")
    critique.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args, overrides = parser.parse_known_args(argv)
    if args.command in {"prepare", "run", "harness"} and args.task and "=" in str(args.task):
        overrides.insert(0, str(args.task))
        args.task = None
    try:
        _validate_overrides(overrides)
        if args.command not in {"prepare", "run", "harness"} and overrides:
            raise ValueError("Pipeline overrides are only accepted by prepare/run/harness")
        if args.command == "harness":
            from .harness import run_harness
            from .preflight import runtime_tool_path
            with runtime_tool_path():
                return run_harness(args, overrides)
        if args.command == "templates":
            for name, description in TEMPLATES.items():
                print(f"{name}: {description}")
            return 0
        if args.command == "providers":
            if args.config:
                spec = PROVIDERS[args.config]
                model_cfg = ResearchConfig().literature.model_search
                model_cfg.provider, model_cfg.base_url, model_cfg.api_key_env = args.config, spec.base_url, spec.api_key_env
                model_cfg.api_format, model_cfg.mode = spec.api_format, spec.mode
                if args.config == "deepseek":
                    model_cfg.external_search.provider = "deepseek"
                print(yaml.safe_dump({"literature": {"model_search": asdict(model_cfg)}}, sort_keys=False, allow_unicode=True), end="")
            else:
                for name, spec in PROVIDERS.items():
                    print(f"{name:25} {spec.mode:8} {spec.api_format:24} {spec.label}")
                print("Model IDs and credentials are user-configured. Native search requires a supported model and account.")
            return 0
        if args.command == "init":
            destination = args.project.expanduser().absolute()
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"Project already exists: {destination}")
            cfg = ResearchConfig()
            cfg.literature.queries = args.query
            cfg.literature.model_search.provider = args.provider
            cfg.literature.model_search.model = args.model
            if args.provider == "deepseek":
                cfg.literature.model_search.external_search.provider = "deepseek"
            if args.brief:
                brief = args.brief.expanduser().read_text(encoding="utf-8")
                if not brief.strip():
                    raise ValueError("Research brief is empty")
                destination.mkdir(parents=True)
                (destination / "task.md").write_text(brief, encoding="utf-8")
            else:
                template = args.template or "circpack"
                copy_task(ROOT / "examples" / template, destination)
                (destination / "baseline_provenance.json").write_text(json.dumps({
                    "template": template, "repository": "https://github.com/SakanaAI/AI-Scientist-v3",
                    "source": f"examples/{template}", "license": "Apache-2.0",
                    "qualification": "The V3 runtime must validate this baseline before experiments.",
                }, indent=2) + "\n", encoding="utf-8")
            write_config(destination / "research.yaml", cfg)
            load_config(destination)
            print(f"Created research project: {destination}")
            print("Edit research.yaml for the model provider, model ID, key environment variable and sources; then run agsci research prepare.")
            return 0
        if args.command == "critique":
            from .paper_critique import critique
            report = critique(args.manuscript, args.output, reviewers=args.reviewers, settings={
                "agent": "codex", "model": args.model, "profile": args.profile,
                "reasoning_effort": args.reasoning_effort, "timeout_seconds": args.timeout_seconds,
            })
            print(json.dumps({"consensus": report["consensus"], "verdicts": report["verdicts"],
                              "blocking_total": report["blocking_total"],
                              "report": str(args.output / "critique_report.json")}, ensure_ascii=False))
            return 0
        if args.command == "report":
            path = args.workspace.expanduser() / "research_report.json"
            resume(args.workspace.expanduser())
            print(path.read_text(encoding="utf-8"))
            return 0
        if args.resume:
            if args.task or args.workspace or args.offline:
                raise ValueError("--resume uses the original task, workspace, and literature; omit task/--workspace/--offline")
            workspace = args.resume.expanduser().resolve()
            cfg, manifest = resume(workspace)
            overrides = [*execution_overrides(workspace, manifest), *overrides]
            _validate_overrides(overrides)
        else:
            if args.task is None:
                raise ValueError("Provide a task directory or --resume WORKSPACE")
            task_dir = args.task.expanduser().resolve()
            cfg = load_config(task_dir)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
            workspace = (args.workspace or ROOT / "results" / "unified" / task_dir.name / run_id).expanduser().absolute()
            if workspace.exists() or workspace.is_symlink():
                raise ValueError("Workspace exists; use --resume or choose a new path")
            if workspace.resolve().is_relative_to(task_dir):
                raise ValueError("The research workspace must be outside its source task directory")
        command = pipeline_args(workspace, cfg, overrides)
        # Compose before any retrieval or staging so typos cannot create a run.
        from src.pipeline.cli import config_error_message, load_pipeline_config
        try:
            pipeline_cfg = load_pipeline_config(command)
        except Exception as exc:
            raise ValueError(config_error_message(exc)) from exc
        if args.dry_run:
            print(json.dumps({
                "dry_run": True, "workspace": str(workspace),
                "prepare_literature": not bool(args.resume), "queries": cfg.literature.queries,
                "local_sources": len(cfg.literature.sources), "search_strategy": cfg.search.strategy,
                "literature_model": {"provider": cfg.literature.model_search.provider, "model": cfg.literature.model_search.model,
                                     "mode": cfg.literature.model_search.mode or PROVIDERS[cfg.literature.model_search.provider].mode},
                "pipeline_argv": command, "note": "No agent, network request, training, or workspace creation occurred.",
            }, ensure_ascii=False, indent=2))
            return 0
        # Resume performs no retrieval and can use a scaffold drafted in run/task.
        # Its existing runtime checks remain owned by the execution pipeline.
        if args.command == "run" and not args.resume:
            ensure_execution_ready(pipeline_cfg, task_dir)
        if not args.resume:
            manifest = prepare(task_dir, workspace, offline=args.offline, overrides=overrides)
        write_report(workspace)
        print(f"Research workspace: {workspace}", flush=True)
        if args.command == "prepare":
            return 0
        from src.pipeline.cli import main as pipeline_main

        record_execution(workspace, manifest, overrides)
        try:
            exit_code = pipeline_main(command)
        except KeyboardInterrupt:
            write_report(workspace, exit_code=130)
            raise
        except Exception:
            write_report(workspace, exit_code=1)
            raise
        write_report(workspace, exit_code=exit_code)
        return exit_code
    except (ValueError, OSError) as exc:
        print(f"agsci research: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
