# PathMNIST whole-idea portfolio case

This is a new GPU/HPO template. It leaves the earlier CPU study unchanged. Use it to
exercise the Harness's parallel idea search and full-method tuning on a cheap real-data
case. It does not claim a novel method, clinical validity, or top-venue readiness.

The integration reuses the existing PathMNIST data constructor, contract artifact
builder, reference implementation, permutation checks, AUROC, and paired bootstrap.
It uses the native `python_rpc_v1:handle` broker and sealed Docker runtime. There is no
separate job executor. The Harness scheduler assigns one GPU to each candidate worker.

The host validates the candidate's eight-parameter configuration before training and
holds data, seeds, 160-step budget, validation checkpoint selection, and scoring fixed.
Ideas can change the problem hypothesis, efficiency target, objective, or structure.
An efficiency hypothesis still needs the fixed quality guardrail and a matched-device
timing study. The initial candidate is only an attention/LME mixture.

`research.yaml` enables the medical methodology and whole-idea policy. Build whole ideas
first, then use development results for joint HPO. Explain a selected method with the
comparison type that matches the claim: independently tuned methods, matched component
removal, or configuration sensitivity. Report all attempted configurations and failures.
Do not reinterpret unequal configurations as proof that one component caused a gain.

## Prepare a new frozen task

Do this only after checking the target GPU and CUDA runtime. The original archive must
match the official MD5 checked by the existing preparation helper. No download occurs.

```bash
.venv/bin/python examples/pathmnist_mil_portfolio/prepare.py \
  --archive scratch/weak-supervision/data/pathmnist.npz \
  --task scratch/pathmnist-portfolio/task \
  --cuda-torch-version '2.11.0+cu126' \
  --cuda-wheel-index https://download.pytorch.org/whl/cu126 \
  --accelerator-model 'NVIDIA GeForce RTX 4080 SUPER'
```

The CUDA wheel/model above is the target configuration checked during development,
not evidence that CUDA execution has passed. Preparation refuses an existing output
directory. It reconstructs historical used patch IDs from the original archive and
excludes them from fresh validation/test bags. It does not read prior holdout files.
It preserves the same fresh heldout construction as the CPU template; do not call it
a new independent holdout if that fresh set has since been evaluated or inspected.

Preparation freezes unmodified shared scoring code under `shared_helpers`, writes hashes for
all host helpers, and binds `requirements.txt` and the selected GPU model in the task
contract. The source template is not a prepared study. Its `UNCONFIGURED` device model
must be replaced by preparation. Candidate code never gets final paths or patch IDs.

## CUDA runtime setup

Keep NumPy 2.4.2, scikit-learn 1.8.0, threadpoolctl 3.6.0, and the exact selected Torch
CUDA build. The current source lock is `torch==2.11.0+cu126`, with an existing Python
3.12 Linux wheel verified at the official CUDA 12.6 index. CPU Torch does not satisfy
this lock. macOS unit fixtures do not establish CUDA compatibility.

The native task runtime installs exact package pins only. It does not accept pip index
flags in `requirements.txt`. Set `AGENTIC_TASK_PACKAGE_INDEX` to a mainland package
mirror for ordinary dependencies. Configure the runner image's `PIP_EXTRA_INDEX_URL`
to a verified CUDA wheel mirror. Prefer a working mainland mirror; use the official
index through the command-scoped proxy if a suitable mirror is unavailable.

`prepare.py --cuda-wheel-index` records this required image setting in the hashed
`runtime_setup.json`. That file is setup metadata, not an installer. The native runtime
must apply the image setting before task installation. A successful metadata write
does not prove installation or a working GPU. Test CUDA allocation and computation
through the native runner before interpreting research results.

The candidate requires one visible CUDA device and checks its local index, name,
UUID, Torch version, and CUDA version. The host checks this report against the frozen
model/version. Native Docker binding supplies the actual allocation boundary.
These reports and parameter counts are candidate-side measurements, so arbitrary
generated implementations still need source review. The host independently records
step schedules, training examples, and wall time.

## Evidence and cost

`combined_score` remains mean condition-level worst-view AUROC. All HPO trials use the
same data, seeds, 160 steps, and four checkpoint opportunities. Search has two seeds
and two label-noise conditions. Final has five separate seeds and the same protocol.
Final evaluation occurs only after candidate/configuration freeze.

The output adds candidate train-RPC time, condition wall time, reported parameters,
and host-counted training examples. Train-RPC time includes transport/synchronization;
condition time also includes initialization and validation. Different GPU models/load
are not controlled. The CPU LME anchor is frozen and untuned; it is not an equal-search
baseline or a GPU latency benchmark. Use matched canonical comparisons for claims.

Normal evaluation refuses direct candidate execution. Use the existing Harness task
qualification, search, and final runners. Unit tests use synthetic CPU arrays and mock
RPC replies. They validate boundaries, configuration rejection, frozen budget behavior,
and evidence shape; they do not read real final resources or claim GPU validation.
