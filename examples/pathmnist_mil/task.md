# Fast weak-supervision pathology pilot

## Scientific question

When the positive tumor evidence is held fixed, does adding non-tumor tissue
cause a small attention MIL classifier to lose discrimination? Can a small
aggregation change reduce that loss under clean and mildly corrupted bag labels?

This is a controlled development study using synthetic bags of real PathMNIST
colon histology patches. It is not a clinical WSI study, patient-level validation,
or evidence of a new algorithm. Attention2Minority, rank induction, IBMIL and
PseMix are close prior work. Read the supplied full texts before novelty claims.
If a candidate is already known, label it as a replication or control and measure
it honestly. Do not reject the whole research question merely because sparse
attention, negative mixing, or small-lesion selection already exists.

## Fixed experiment

- Cached label-free color/texture features, standardized and reduced to 48 PCA
  dimensions using the official training partition only. No backbone training.
- 600 training bags, 100 validation bags, and an external 80-bag holdout.
- Training bags contain 16 patches. Positive bags contain 1 or 2 tumor patches.
- Validation compares paired bags of length 16 and 64. The first 16 patches are
  identical. All 48 added patches are non-tumor. Labels and positive evidence stay
  fixed. Ground-truth patch labels were used only during bag construction.
- The model receives only image features and bag-level supervision. Do not access
  original patch labels, the raw dataset, construction audit, or holdout files.
- Each candidate is trained with clean bag labels and with 15% fixed symmetric
  bag-label flips, using matched seeds and corruption across methods.
- CPU only. The frozen `evaluation_protocol.json`, when present, sets seeds,
  optimizer steps, validation cadence, patience, and training time cap. Without
  that file, use the quick defaults: 2 threads, batch size 32, at most 160 steps,
  validation every 20 steps, patience 3, and a 20-second training time cap.
- At most 20,000 total parameters, including frozen or inactive parameters. No downloads, pretrained models, subprocess
  training jobs, separate feature training, or exhaustive hyperparameter grids.

## Candidate interface and permitted edits

Edit only `program_<n>/src/program.py`. Export `build_model(input_dim)`, returning
a `torch.nn.Module`. Its `forward(bags)` receives shape `(batch, instances, 48)`
and must return finite logits of shape `(batch,)`. It must support bag sizes 16
and 64, with permutation-invariant aggregation and independent predictions for
each bag regardless of batch row order. Use only torch and standard math.
Do not read any files or inspect the evaluator from candidate code.

The fixed evaluator owns optimization, labels, early stopping, and scoring.
Optional model attributes are `learning_rate` in [0.0001, 0.01] (default 0.003)
and `label_smoothing` in [0, 0.2] (default 0). The frozen protocol takes precedence
when it fixes these optimizer settings for a matched comparison. Keep the baseline encoder width 32
unless an explicit matched-capacity control justifies changing it.
Do not modify evaluator code, cached features, or budgets. Never write metrics
yourself. In the fast preset the host evaluates exactly once after the coding
agent returns. Do not run training or evaluation inside that agent turn.
Do not spawn other agents or use external services. Work only on this attempt's
candidate and required idea/rubric artifacts.

## Selection and evidence

Maximize `combined_score`: the mean, over the two label-noise conditions, of
min(AUROC on 16-patch bags, AUROC on paired 64-patch bags).
Record both AUROCs, positive and negative probability changes after adding
background, class-margin change, runtime, actual training steps, and parameter
counts. Use the mean score across all frozen seeds. Improvement on validation is a search
observation, not a final result. An independent post-search script evaluates the
frozen baseline and selected candidate on the external holdout using two new
seeds. Never use holdout performance to revise a candidate.

Use the host-provided effective campaign budget for the number of idea cycles
and remaining attempts. Do not infer the campaign budget from this scientific
brief. Prefer small, interpretable changes with an explicit competing explanation.
Keep more than one plausible branch; compare a known control, then refine using
per-condition results. A known method can be a useful control. Preserve negative
results and do not claim that a larger search guarantees an improvement.

## Competing explanations and limits

1. Background dilution: fixed lesion evidence loses influence as non-tumor count
   increases; a length-stable aggregator should reduce the paired decline.
2. Feature insufficiency: low-resolution handcrafted features do not separate
   tumor and non-tumor patches; changing pooling will not reliably solve it.
3. Label-noise overfitting: gains on clean labels disappear under identical flips.

Original patient IDs are unavailable here. Exact image duplicates across splits
are removed, but patient independence is not established. The test images come
from the official external PathMNIST partition. Synthetic bag AUROC cannot be
reported as clinical slide AUROC. A fixed-feature pilot can justify a next study,
not a publication-level novelty or effectiveness claim.
