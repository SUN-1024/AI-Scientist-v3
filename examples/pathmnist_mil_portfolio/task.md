# Whole-idea screening and joint HPO on synthetic PathMNIST bags

Develop and optimize a complete research idea. A useful contribution can concern the
problem definition, efficiency, data use, objective, optimization, or model structure.
It need not add a module. State a falsifiable prediction and the closest prior methods.
Preserve a plausible direction when it can make a meaningful advance over those methods.

This is a cheap Harness research case. It uses fixed image features and synthetic bags.
It does not establish WSI, patient-level, clinical, or external validity. The seed is a
known attention/LME mixture. It carries no novelty or paper-quality claim.

## First optimize the complete idea

Use a minimum complete implementation, then jointly tune its allowed parameters.
Do not begin by optimizing modules one at a time. Explore distinct hypotheses and
use the same development score and budget for each. Keep every attempt and its actual
configuration, including unsuccessful trials. Development results can guide selection.
Do not select on final results or hide failed trials.

Only `program_0/src` is editable. Implement the idea there. Architecture and training
objective may change within the capacity budget and interface. The host validates one
fixed configuration before training. A candidate must return the same configuration
after every reset. `configuration_contract.json` is the authoritative search space:

| Parameter | Allowed range or values | Seed |
| --- | --- | --- |
| learning_rate | 0.00001 to 0.03 | 0.003 |
| weight_decay | 0 to 0.01 | 0.0001 |
| batch_size | 16, 32, 64 | 32 |
| hidden_dim | 16, 32, 64, 128 | 32 |
| dropout | 0 to 0.5 | 0 |
| pooling_temperature | 0.1 to 10 | 1 |
| gradient_clip_norm | 0.1 to 10 | 5 |
| hybrid_mix | 0 to 1 | 0.5 |

Unknown parameters, booleans as numbers, nonfinite values, and out-of-range values
are rejected. In the whole-idea parameter-space declaration, optimizer settings and
batch size have owner `task`; architecture settings have owner `candidate`. Neither
owner can change the bounds. Seeds, splits, metrics, step budget, and selection rules
are evaluator-owned and cannot be tuned. If an architecture does not use a listed
architecture parameter, state that fact rather than pretending it was effective.

## Host-owned protocol

- The host sends training features and noisy bag labels only. Training bags contain
  16 patches. Validation and final views contain 16, 32, or 64 patches.
- Data construction reuses the existing exact-duplicate filter, train-fitted scaler
  and PCA, historical-patch exclusion, and fresh-bag sampler. It never reads prior
  holdout files. Train bags are reused. Actual fresh bag counts are in the manifest.
- Search seeds are 17 and 29. Final seeds are 101, 211, 307, 401, and 503.
- Each seed has noise rates 0 and 0.15. The host flips the exact same training labels
  across trials and provides all minibatch indices using the approved batch size.
- Each condition has 160 optimizer steps. The host selects checkpoints at steps
  40, 80, 120, and 160. Ties keep the first checkpoint. Smoke uses 8 steps and one
  condition; it checks wiring only and cannot support quality or efficiency claims.
- Use Adam with the host-approved learning rate and weight decay, and the supplied
  gradient norm bound. Execute exactly one optimizer step per host minibatch.
  A different robust objective or model is allowed. Extra hidden training is not.
- Model capacity is at most 20,000 total parameters. The seed reports measured total
  and trainable parameter counts. These are candidate-side telemetry, not independent
  attestation; source review remains necessary for arbitrary candidate implementations.
- The only ranking score is mean condition-level worst AUROC over the three views.
  Scoring uses raw logits. All trials use the same data, seeds, and step count.
  Batch size changes training examples seen; the host records that cost explicitly.
- The existing evaluator checks independent per-bag predictions, chunk/batch agreement,
  and row/instance permutation invariance. Validation labels select checkpoints only
  in the host. Final labels never select checkpoints or configurations.

## GPU and RPC contract

Use the existing `python_rpc_v1:handle` runtime and sealed Docker broker. Each candidate
gets exactly one assigned CUDA device, visible as `cuda:0`. CPU fallback is forbidden.
The evaluator has no GPU. Do not launch independent jobs, access a second device,
read files for feedback, use a network, or inspect task/final paths in candidate code.
Only the fixed broker proxy is loaded by the host. The candidate receives no task or
results mount. Native Docker allocation is authoritative; candidate device reports
are checked for consistency but are still candidate-reported telemetry.

| Operation | Request and response |
| --- | --- |
| reset | Host seed; clear state and initialize RNGs. This is always the first RPC. |
| get_configuration | No data; return only `{configuration: {...eight parameters...}}`. |
| init | Training data, approved configuration and optimizer values; return measured total/trainable parameters and the CUDA device report. |
| train | Host minibatch index lists; execute one step per list, synchronize CUDA, and return the step count. |
| predict | Features only; return one finite raw logit per bag. |
| snapshot / restore | Save or restore model parameters using the host's opaque name. |

The device report gives `device_type`, `cuda_available`, `device_count`,
`logical_device_index`, `device_name`, `device_uuid`, `torch_version`, and `cuda_version`.
The host checks one CUDA device, local index zero, nonempty identity, and the exact
frozen device model and Torch version. The native runner separately binds GPU access.

Timing fields are costs, not quality scores. The host measures candidate train-RPC
wall time, including transport and synchronization. Condition time also includes
initialization, validation checks, and checkpoint operations. Final scoring is outside
that condition time. Different GPU models and host load can change these values.
Efficiency claims need a separate matched-hardware measurement with a quality guardrail.

## Comparisons after selecting and freezing a complete idea

Every trial includes an unchanged CPU LME anchor using the default configuration.
It reuses the established reference implementation and receives the same 160-step
budget. It is not an equally tuned competitor, and its CPU time is not a GPU benchmark.

Required canonical IDs are `baseline_lme` (kind `baseline`, base `best`) and the
automatic `control_best` (kind `control`, base `best`). Build the LME method in candidate
code and use a declared development tuning budget comparable to the main method.
The seed's internal fixed LME score does not replace this comparison. Additional
classical pooling or nearest-method controls should answer the primary claim.

Label each explanatory comparison accurately:

- `best_tuned_method_comparison`: methods may have different development-selected
  parameters; disclose both search spaces, trial counts, and selection criteria.
- `matched_component_ablation`: remove the named mechanism and hold other settings
  fixed, or use a declared matched retuning rule. This can support a component claim.
- `configuration_sensitivity`: vary parameters. This describes sensitivity and does
  not isolate a component when multiple factors change.

The required explanation depends on the contribution. Do not invent a removable
module for an efficiency or problem-setting idea. Keep unfavorable outcomes visible.
Freeze candidate code, configuration, and comparisons before sealed final evaluation.
Bootstrap intervals describe this small synthetic case only. Paper readiness also
requires source-grounded novelty, appropriate controls, uncertainty, and a stronger
independent research setting. This case alone cannot prove a top-venue contribution.
