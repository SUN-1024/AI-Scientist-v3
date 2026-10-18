# Unified research development validation

Updated: 2026-10-18. Base revision: `b72f0b014e80b59e1c8d881055b6353b1777927d`.

## Live integration check, 2026-10-18

The [PathMNIST pilot report](pathmnist-mil-pilot-2026-10-18.md) records a real Magpie/GPT native-search → idea → coding → host-evaluation run through `research run`. It evaluated the baseline and two candidates. Neither candidate improved validation, and the search retained the baseline. A post-search, two-seed holdout check also favored the baseline over the highest validation-score challenger. The run did not include paper writing or sealed container evaluation.

The latest regression commands passed 292 main tests plus 5 baseline-checkpoint tests (297 total). Native baseline receipt validation, effective tree configuration and complete task-brief prompt tests are included. One Linux-specific descriptor test was excluded on macOS; its earlier Linux check passed. Ruff and mypy passed. Frozen code, metrics, data hashes, source availability and interface checks accompany the report.

A transient upstream 502 prevented one coding call from producing a candidate. It is recorded as an execution failure, not scientific underperformance. The final dedicated Magpie profile increases bounded request/stream retries to three; recovery under injected faults has not been tested.

## Execution and evidence integration

- 228 focused tests passed on macOS with Python 3.12.13 and GNU Bash 5.2.37 after the model-search and paper-reference integration.
- The Linux-dependent upstream test `test_pending_revision_publication_recovery` passed separately in an ephemeral `python:3.12-slim` container. It was excluded from the macOS command because that path needs Linux descriptor filesystem support.
- Ruff passed for the new modules and modified Python integration files. Mypy passed for all ten new source files.
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
  tests/investigation/test_harness_eval.py \
  tests/investigation/test_investigation_cli.py \
  tests/checkpoints/test_best_manifest_bootstrap.py \
  tests/pipeline/test_pipeline_config.py \
  tests/pipeline/test_agsci_cli.py \
  tests/paper/test_paper_prompting.py \
  tests/paper/test_paper_data.py \
  tests/paper/test_paper_references.py \
  tests/e2e/test_agent_execution_pipeline.py::IdeaShExecutionTests::test_idea_sh_executes_agent_and_validates_artifacts \
  -k 'not test_pending_revision_publication_recovery'
```

On Linux, run the descriptor test without that exclusion. Locked dependencies came from `uv.lock`; the Aliyun PyPI mirror was used for installation. The lock file was unchanged.

## Runtime boundary

The native `init`, `run --dry-run`, `prepare`, `report`, and investigation commands were exercised. A live LLM investigation and native post-search holdout check are now complete. The full V3 agent image was not installed for this pilot. A complete LLM-driven experiment-to-paper run has not been completed in this worktree. No matched-budget comparison against the original versions has been run.

## Configurable model-based retrieval

The previous front-of-study Semantic Scholar adapter has been removed. There are 28 provider/runtime configuration presets, six HTTP protocol formats, and four external search backends. The configured user model performs native web search or plans queries and parses results obtained from the configured search backend. The actual paper shell routes `research.yaml` tasks, including post-draft enrichment, through the same model configuration. Legacy tasks without `research.yaml` retain upstream behavior.

- Native HTTP fixtures cover OpenAI, OpenRouter, Anthropic, Gemini Interactions, Gemini generateContent, DashScope, Doubao, Zhipu, xAI, and Perplexity response shapes.
- External HTTP fixtures execute the full model-plan → search → model-extraction sequence with SearXNG, Tavily, Brave, and Zhipu backend responses.
- Tests cover plain-text conversion to JSON, Anthropic pause continuation, citation receipts, hallucinated source/PDF URLs, API errors, 401/429, redirect refusal, missing credentials, missing backend configuration, and removal of echoed credentials from saved artifacts.
- Model summaries remain metadata, never original abstracts or full text. Model-only answers without structured search sources are rejected.
- All 28 generated configuration templates validate. Custom paths, authentication headers, parameter overrides, and removal of default request fields are exercised.
- Provider protocol coverage uses actual local HTTP integration tests with simulated responses. Magpie was additionally tested live on 2026-10-18 through the user's locally configured upstream authentication. Other presets have not been verified with live user accounts. Fixtures do not establish service availability, model access, search quality, or billing.
- Six citation bridge tests verify source-derived authors/title/year, DOI content negotiation, safe BibTeX fields, offline local citations, cache/hash verification, refusal to invent missing metadata, and preservation of existing outputs on failure. One executes the production paper-shell Python block and rejects any call to the old retrieval functions.
- Two live primary-metadata checks succeeded: arXiv `2408.06292` returned the AI Scientist title, six authors and 2024; DOI `10.1126/science.220.4598.671` returned Optimization by Simulated Annealing, three authors and 1983. These checks did not call a model or a discovery API.

The native CLI was also exercised with an offline project, its configured provider/model preserved in preview and report. The research launcher selects an available modern Bash and uses a canonical task temporary directory on macOS. No Semantic Scholar API endpoint remains in the research retrieval code. Setup and account-specific configuration are in [the model search guide](model-search.zh-CN.md).

The regression command emitted one upstream `scholarly` SyntaxWarning from its legacy compatibility test; no test failed.
