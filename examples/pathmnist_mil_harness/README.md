# Contracted PathMNIST MIL CPU task

This is a small, real-data task for the full research Harness. It supports investigation,
canonical variants/ablations, sealed final evaluation, and paper evidence collection.
It uses `schema_version: 1` and `python_rpc_v1:handle`. It does not train a CNN.

The checked-in directory is a source template. Generate a frozen task under an ignored
scratch directory before a Harness run. It refuses to overwrite an existing task.

`research.yaml` is a configurable example. Its literature default is
`deepseek-flash` through the official DeepSeek API, with DeepSeek's standalone
Search API as the external search backend. Agent calls use the project's
`agsci-deepseek` Codex profile and also connect directly to DeepSeek. Export
`DEEPSEEK_API_KEY` for both paths; the files contain no key. See the
[model guide](../../docs/model-search.zh-CN.md).
Author and code models use `agent.*`; independent review overrides use `review.agent_*`.
These choices affect this run, not the user's global Codex configuration.
New tasks enable the `medical` methodology profile. The Harness freezes its rules
before agenda generation. Patient-level isolation and clinical validity remain
unknown unless supported by evidence; patch-disjoint bags do not establish either.
See the [methodology guide](../../docs/research-methodology.zh-CN.md).

```bash
.venv/bin/python examples/pathmnist_mil_harness/prepare_data.py \
  --archive scratch/weak-supervision/data/pathmnist.npz \
  --task scratch/weak-supervision-harness/task
```

The source archive must have official MD5 `a8b06965200029087d5bd730944a56c1`.
Its source is the [MedMNIST Zenodo archive](https://zenodo.org/records/10519652),
licensed CC BY 4.0. Install the exact packages in `requirements.txt`. Use a Mainland
China package mirror first, then the command-scoped local proxy if necessary.
The CPU Docker image supplies Python and the Harness tools. The Harness installs
these exact packages into its managed task virtual environment. Set
`AGENTIC_TASK_PACKAGE_INDEX` to the approved package mirror for that installation.

`requirements.txt` is the Linux Harness lock. It pins `torch==2.11.0+cpu`, so pip
cannot silently select the large CUDA-enabled PyPI distribution. Configure the CPU
wheel index in the Linux image's pip settings, for example:

```dockerfile
ENV PIP_EXTRA_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
```

Keep the normal package mirror as the primary index for NumPy, scikit-learn and other
dependencies. If the mainland CPU wheel mirror is unavailable, use the official
`https://download.pytorch.org/whl/cpu/` index through the command-scoped proxy.
The Harness accepts only exact `package==version` lines. Do not put pip options,
platform markers or direct wheel URLs into this task lock.

For macOS data preparation and local smoke checks, retain the macOS PyTorch wheel:

```bash
python -m pip install numpy==2.4.2 scikit-learn==1.8.0 'torch==2.11.0' threadpoolctl==3.6.0
```

The Linux `+cpu` lock is not a macOS install file. This does not change an existing
macOS environment. The actual contracted evaluation still runs in Linux containers.


Data preparation reconstructs old patch selections from the official raw archive
using the exact old sampler: train 600 x 16 / seed 17, validation 100 x 64 / seed 29,
test 80 x 64 / seed 41. It never opens any old holdout file. Fresh validation/test bags
use only the remaining deduplicated patches in each official split. Four balanced
64-patch bags consume 3 tumor patches and 253 non-tumor patches. Actual bag counts
are written into `data_manifest.json`; requested counts are only upper bounds.
All patch IDs and construction audits stay under final resources. Neither search nor
candidate inputs contain patch labels. The new task retains the old training bags.

The scaler and PCA fit on the deduplicated official train images only. Paired views
retain each positive witness at sizes 16, 32, and 64. Patch-disjoint here means exact
image and patch separation. It does not establish patient or hospital separation.

For a local wiring check (not a sealed research result):

```bash
AGENTIC_TASK_DRAFT_SMOKE=1 \
  .venv/bin/python scratch/weak-supervision-harness/task/evaluate.py \
  --program_path scratch/weak-supervision-harness/task/program_0/src/program.py \
  --results_dir scratch/weak-supervision-harness/smoke
```

Run actual investigation and final confirmation through the contracted Harness. Direct
execution loads the candidate in the evaluator process and is for development checks
only. Isolation is supplied by the Harness CPU containers, not this Python file.

The initial candidate is a shared-classifier equal attention/LME mixture. It is not a
new contribution claim. `reference.py` freezes attention, LME, mean, max, and top-2.
The evaluator always measures candidate and frozen LME with the same training protocol.
Readiness requires these exact canonical variant IDs: `baseline_attention`,
`baseline_lme`, `baseline_mean`, `baseline_max`, `baseline_top2`, `control_best`, and
`ablation_no_primary_mechanism`. The five pooling variants have kind `baseline`.
The unchanged best candidate is the automatic `control`; the primary-mechanism removal
has kind `ablation`. The detailed implementation and generation rules are in `task.md`.
A missing mechanism or comparison leaves readiness incomplete.

Set `ablations.execution.max_extra_baselines=5` on the full Harness command. Its default
cap of 3 is too small for the five required baselines. The default ablation cap of 3
already allows the required removal experiment. Keep automatic controls enabled.
A separate variants CLI uses `execution.max_extra_baselines=5`. Expect at least seven
canonical variants, plus the optional automatic `baseline_reference`.

The host sends only training features, noisy bag labels, batch indices, and unlabeled
prediction features to `handle`. The host selects checkpoints using validation labels. Every scored prediction is
checked against full-batch permutations, independent probes, random chunks, and the
same bag predicted alone. Fresh host-only seeds vary scoring order. These checks use
a 10,000-call RPC budget; the small bags keep each request bounded.
Final labels remain in the evaluator. Final `metrics.json` contains scalar AUROCs, the LME score, and the paired effect/interval.
`evaluation_evidence.json` contains the host-authored condition records, paired per-bag
predictions, labels, validation traces, and bootstrap method. The interval
is exploratory because the number of independent bags and training seeds is small.
The source does not assert that the candidate improves LME or meets a venue standard.
