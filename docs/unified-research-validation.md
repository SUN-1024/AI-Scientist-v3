# Unified research development validation

Date: 2026-10-17. Base revision: `b72f0b014e80b59e1c8d881055b6353b1777927d`.

## Execution and evidence integration

- 178 focused tests passed on macOS with Python 3.12.13 and GNU Bash 5.2.37.
- The Linux-dependent upstream test `test_pending_revision_publication_recovery` passed separately in an ephemeral `python:3.12-slim` container. It was excluded from the macOS command because that path needs Linux descriptor filesystem support.
- Ruff passed for the new modules and modified Python integration files. Mypy passed for all seven new source files.
- The actual idea shell wrapper was exercised with a deterministic agent fixture. It received the literature dossier and selected branch. No unresolved new prompt placeholders remained.
- The CPU campaign fixture executed generated Python and evaluated its objective. Cycle 3 expanded a valid lower-scoring candidate while the global incumbent remained separate. Literature comparisons reached both program rows and ablation-style paper data.
- Invalid citations, changed source text, invalid full-text spans, duplicate local source IDs, blank PDFs, changed frozen inputs, existing workspace overwrites, and unknown pipeline options were rejected.
- Resume retained the most recent effective model/budget overrides. Pipeline exceptions were recorded as failures.

The agent fixtures validate integration and execution flow. They do not measure LLM research quality.

## Reproduction

Use GNU Bash on PATH. On macOS, point both temporary-directory variables at an existing canonical directory rather than a path below the `/var` symlink.

```bash
mkdir -p scratch/test-tmp
export TMPDIR="$PWD/scratch/test-tmp"
export AGENTIC_TEMP_ROOT="$TMPDIR"
.venv/bin/python -m pytest -q \
  tests/research \
  tests/investigation/test_campaign_ideas.py \
  tests/pipeline/test_pipeline_config.py \
  tests/pipeline/test_agsci_cli.py \
  tests/paper/test_paper_prompting.py \
  tests/paper/test_paper_data.py \
  tests/e2e/test_agent_execution_pipeline.py::IdeaShExecutionTests::test_idea_sh_executes_agent_and_validates_artifacts \
  -k 'not test_pending_revision_publication_recovery'
```

On Linux, run the descriptor test without that exclusion. Locked dependencies came from `uv.lock`; the Aliyun PyPI mirror was used for installation. The lock file was unchanged.

## Runtime boundary

The native `init`, `run --dry-run`, `prepare`, and `report` commands were exercised. The host doctor found the selected Codex login and required host tools, but the full V3 agent image was not installed. A complete LLM-driven experiment-to-paper run has not been completed in this worktree. No matched-budget comparison against the original versions has been run.

The initial Semantic Scholar live request returned HTTP 429, and the dossier retained the failure as `unavailable`. It did not claim that novelty had been checked. The retrieval adapter is being replaced with configurable model-based search at the user's request; its validation will be recorded separately.
