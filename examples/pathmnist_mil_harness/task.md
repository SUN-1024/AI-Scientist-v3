# CPU weakly supervised MIL under background dilution and bag-label noise

Develop a permutation-invariant bag classifier. Each bag contains fixed patch features.
Training uses bag labels only. The positive class means at least one tumor patch.
The host owns all labels, data splits, random seeds, optimizer settings, batch indices,
checkpoint selection, and scoring. Modify only `program_0/src` (or the copied candidate's `src`).
Keep the JSON `handle(request)` interface described below.

## Research question and evidence limits

Can a small pooling method preserve rare positive evidence when extra background
patches enter a bag, without losing robustness to noisy bag labels? Explain a concrete
mechanism and test it against competing explanations. A useful direction can extend
existing work. Establish the meaningful difference from close methods using full text.
The initial shared-classifier 50% gated attention / 50% ordinary log-mean-exp mixture
is a development starting point. It is not a verified novelty contribution. An earlier
pilot tied ordinary log-mean-exp; that result is not evidence of superiority here.

This task uses synthetic bags from PathMNIST 28 and fixed color/texture features.
It is not WSI model reproduction or a patient-level clinical study. There is no learned
CNN. New validation and test patches exclude the exact patches used in the earlier
pilot. The train split is reused. Exact image duplicates are removed across official
splits. Patient separation is not verified. This is within-dataset patch confirmation.
Never describe it as external or patient-level generalization.

## Frozen study

- Train on 600 bags with 16 patches. Features are fitted on official train images only.
- Positive bags contain one or two witnesses. Negative bags contain no witnesses.
- Paired validation and test views have 16, 32, and 64 patches. Extra patches are negative.
- Data preparation caps fresh validation at 56 bags and fresh test at 32 bags. It records
  actual counts from unused patch capacity; do not assume more data exist.
- Search seeds: 17 and 29. Final seeds: 101, 211, 307, 401, and 503.
- Independent conditions flip exactly round(600 * rate) bag labels for rates 0 and 0.15.
- Each condition uses 640 Adam steps, batch size 32, learning rate 0.003,
  weight decay 0.0001, and gradient clip norm 5. The host supplies all minibatch IDs.
- Host validation selects among checkpoints at 160, 320, 480, and 640 steps. Ties keep
  the first checkpoint. The selection score is worst AUROC over the three bag sizes.
- Search score: mean of condition-level worst-view AUROCs. Higher is better.
- Required classical controls: gated attention, ordinary LME, mean, max, and top-2.
  They share the small encoder and instance classifier where applicable. Attention
  adds gate parameters. Report parameter counts and this capacity difference.
- Every normal evaluation includes a frozen ordinary-LME control. Ablations must test
  the proposed mechanism and alternate pooling. Do not infer causality from score alone.
- Final evaluation uses the frozen candidate, fixed protocol, and ordinary LME, with
  paired per-bag predictions and a paired bag/seed bootstrap. Final data never choose
  a model or checkpoint. Report effect, uncertainty, and failures as observed.

## Required canonical variants

The ablation generation stage must use these exact IDs and kinds. Use snake_case IDs
without renaming them. Readiness checks each completed, rubric-passed canonical result;
the evaluator's internal LME score does not replace a canonical baseline variant.

| variant_id | kind | base_program | Required implementation or purpose |
| --- | --- | --- | --- |
| `baseline_attention` | `baseline` | `best` | Gated attention pooling with the frozen small encoder and classifier. |
| `baseline_lme` | `baseline` | `best` | Ordinary log-mean-exp of instance logits, with temperature 1 and division by bag size. |
| `baseline_mean` | `baseline` | `best` | Arithmetic mean of instance logits. |
| `baseline_max` | `baseline` | `best` | Maximum instance logit. |
| `baseline_top2` | `baseline` | `best` | Mean of the two largest instance logits. |
| `control_best` | `control` | `best` | Automatic pipeline control; evaluate the unchanged frozen best candidate. |
| `ablation_no_primary_mechanism` | `ablation` | `best` | Remove only the best candidate's primary new mechanism relative to the seed. |

Propose the five `baseline` rows and the `ablation` row in `variant_bundle.json`.
Do not propose `control_best`: it is a reserved ID that the pipeline creates.
The five classical baselines must match the corresponding `MIL` implementation in
frozen `reference.py`. Keep the encoder, classifier, host training protocol and budget
fixed. Copy the required implementation into the editable candidate source; do not
import a host evaluator helper at candidate runtime. Report the attention gate's extra
parameters. The optional automatic `baseline_reference` is the original hybrid seed;
it does not substitute for any of these five classical baselines.

For `ablation_no_primary_mechanism`, name the actual added component from the best-versus-
seed diff and state a falsifiable mechanism hypothesis. Disable only that component.
Do not label removal of a component already in the seed as a new-mechanism ablation.
If no new mechanism exists, report the gap and leave paper readiness incomplete.
Do not invent an ablation or a completed comparison to satisfy this requirement.
Additional mechanism ablations are allowed after these required comparisons.

Budget: pass `ablations.execution.max_extra_baselines=5` to the full Harness.
The default extra-baseline cap is 3 and would drop two required controls. The default
`ablations.execution.max_ablations=3` accommodates the required mechanism ablation.
This yields at least seven canonical variants, including `control_best`, and may add
`baseline_reference`. Keep automatic controls enabled (`custom_only=false`).

## Candidate JSON interface

`reset`: `{op: "reset", seed: integer}`. Clear all state and seed RNGs.
`init`: receives `train_x`, noisy `train_y`, `learning_rate`, `weight_decay`,
`gradient_clip_norm`. Return `{parameters: integer}`. This is a diagnostic subject to
source inspection, not trusted evidence of effective capacity. Budget: 20,000 parameters.
`train`: receives a list of host-selected `batch_indices`. Execute one Adam step per
list. Return a small JSON object. Do not change the number of steps or hyperparameters.
`predict`: receives only `x`; return `{logits: [one finite raw logit per bag]}`.
`snapshot`: save model parameters under host `name`.
`restore`: restore model parameters under host `name`.

Evaluation labels, patch labels, patch IDs, final results, and evaluator files are not
candidate inputs. Predict each bag independently. Use permutation-invariant pooling.
At every checkpoint and final scoring call, the evaluator compares the actual full-batch
outputs with row/instance permutations, independent probes, random chunks and every
bag scored alone. A fresh host-only seed varies the prediction order. The evaluated
score uses the checked single-bag outputs. These remain finite empirical checks.
Training may learn a robust objective or architecture, but it must preserve the host
optimizer, batch schedule, labels and budget. No network, external data, filesystem
feedback, result writing, or evaluator edits. Use CPU float32.

## Publication boundary

A successful program run proves execution only. It does not prove novelty, a positive
scientific result, or paper readiness. A high-impact claim also needs close-work reading,
mechanism controls, uncertainty analysis, and a stronger independent setting beyond
this small synthetic task. Negative or indistinguishable results must remain visible.
