"""``agsci review``: independently review a manuscript PDF.

OpenAI remains the default. ``--backend agent`` uses the locally configured
Headless agent/profile with a fresh read-only context and strict JSON evidence.
Only approved runtime settings reach the subprocess; secrets are not logged.
The pipeline additionally binds and publishes the review through its manifest.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from src.investigation.final_ai_review import final_ai_review_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEWER_SCRIPT = REPO_ROOT / "src" / "aireview.py"

Runner = Callable[[list[str], dict[str, str]], int]


def _run_inherit_streams(command: list[str], environment: dict[str, str]) -> int:
    """Run the reviewer with the caller's stdin/stdout/stderr and return its exit code."""
    return subprocess.run(command, env=environment, check=False).returncode


def reviewer_command(
    pdf_path: Path, *, uv: str | None, output: Path | None, reviewer_script: Path = REVIEWER_SCRIPT
) -> list[str]:
    """Use current Python for agent review, or uv for legacy OpenAI dependencies."""
    command = [sys.executable] if uv is None else [uv, "run", "--script"]
    command.extend([str(reviewer_script), str(pdf_path)])
    if output is not None:
        command.extend(["--output", str(output)])
    return command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agsci review",
        description=(
            "Independently review a paper PDF. The default openai backend needs "
            "OPENAI_API_KEY; --backend agent uses HEADLESS_BIN and your local agent "
            "profile with the current Python environment. The OpenAI backend runs through uv."
        ),
    )
    parser.add_argument("pdf", type=Path, help="paper PDF to review")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the review JSON to this file (it is always printed to stdout)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="env file to load OpenAI settings from (default: <repo>/.env when it exists)",
    )
    parser.add_argument("--backend", choices=("openai", "agent"), default="openai")
    parser.add_argument("--agent", default="codex")
    parser.add_argument("--agent-model")
    parser.add_argument("--agent-profile")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"))
    parser.add_argument("--reviewer-count", type=int, default=1)
    parser.add_argument("--evidence", action="append", type=Path, default=[])
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    runner: Runner = _run_inherit_streams,
    repo_root: Path = REPO_ROOT,
) -> int:
    """Validate the inputs, build the restricted environment, and run the reviewer."""
    args = _parser().parse_args(argv)
    pdf_path: Path = args.pdf
    if not pdf_path.is_file():
        print(f"agsci review: not a regular file: {pdf_path}", file=sys.stderr)
        return 2

    env_file: Path | None = args.env_file
    if env_file is None:
        default_env_file = repo_root / ".env"
        if default_env_file.is_file():
            env_file = default_env_file
    elif not env_file.is_file():
        print(f"agsci review: env file not found: {env_file}", file=sys.stderr)
        return 2

    source = os.environ if environ is None else environ
    if args.backend == "agent":
        from src.review_backend import agent_review_environment
        environment = agent_review_environment(source)
    else:
        environment = final_ai_review_environment(source, env_file=env_file)
    if args.backend == "openai" and not environment.get("OPENAI_API_KEY"):
        where = f" or in {env_file}" if env_file is not None else ""
        print(
            "agsci review: OPENAI_API_KEY is not set in the environment"
            f"{where}; export it or pass --env-file",
            file=sys.stderr,
        )
        return 1

    uv = which("uv") if args.backend == "openai" else None
    if args.backend == "openai" and uv is None:
        print(
            "agsci review: 'uv' not found on PATH; the reviewer runs as a uv script "
            "(see docs/install.md)",
            file=sys.stderr,
        )
        return 1

    reviewer_script = repo_root / "src" / "aireview.py"
    command = reviewer_command(pdf_path, uv=uv, output=args.output, reviewer_script=reviewer_script)
    if args.backend == "agent":
        command.extend(["--backend", "agent", "--agent", args.agent,
                        "--reviewer-count", str(args.reviewer_count),
                        "--timeout-seconds", str(args.timeout_seconds)])
        for flag, value in [("--agent-model", args.agent_model), ("--agent-profile", args.agent_profile),
                            ("--reasoning-effort", args.reasoning_effort)]:
            if value:
                command.extend([flag, value])
        for evidence in args.evidence:
            command.extend(["--evidence", str(evidence.expanduser().absolute())])
    return runner(command, environment)


if __name__ == "__main__":
    raise SystemExit(main())
