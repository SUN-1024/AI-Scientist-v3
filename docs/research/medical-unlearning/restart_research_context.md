# Second development campaign: preserve utility during patient unlearning

This is a new AI-Scientist v3 development campaign authorized by the human user
on 2026-10-17. The previous campaign is archived. This campaign reuses the
already qualified ACDC SEARCH benchmark; it is not an independent confirmatory
test. The previous FINAL packet and final results are quarantined. Do not read,
evaluate, tune on, or summarize them in this campaign.

## Research question

Can a patient-level deletion update improve agreement with clean Retrain while
retaining at least the segmentation utility of retain-only fine-tuning under
the same supplied interface and measured-time controls?

Preserve the anatomy-preserving patient-unlearning direction. Develop it from
the closest useful mechanism; do not reject it merely because related methods
exist. A novelty claim requires full-text comparisons later and is not an
acceptance criterion for this development batch.

## Permitted previous DEVELOPMENT observations

- The starting search baseline scored 0.8177455611. The first boundary-anchored
  candidate scored 0.8194708063, but the scalar score alone did not establish a
  useful utility/forgetting trade-off.
- Removing its boundary term scored 0.8198481998 on the same development seed.
  This does not support the original boundary-benefit hypothesis.
- Removing orthogonalization changed its development score by only -0.00001986.
  One seed does not establish a stable component contribution.
- These observations can motivate hypotheses, but the earlier ablation
  authoring views were not fully isolated. Treat them as exploratory leads.

## Priority experiments

1. Audit the allocation of the 80 offered update RPCs. A method that spends
   half its calls on forget-only work may lose much of the retain optimization
   available to the fine-tuning comparison. State this possible confound.
2. Start with a small, implementable retain-first method. Consider brief or
   bounded forget pressure, followed by retain learning, instead of immediately
   combining many gates, projection rules, boundary weights and damping terms.
3. If using a retain anchor, select trust or step strength using only allowed
   training observations. Never use heldout labels or Retrain outputs inside
   the candidate. Do not invent unavailable curvature, loss or gradient RPCs.
4. Change one mechanism per cycle. Explain which control could distinguish
   targeted forgetting from ordinary retain learning or global model damage.

## Evidence required before calling a candidate promising

Keep the evaluator's metrics, score and correctness rules unchanged. The
framework owns the scalar incumbent-improvement rubric; do not try to replace
it or claim that it proves a method trade-off. Plan for development heldout
Dice versus the supplied fine-tuning and NegGrad+ comparisons, and forget
probability MAE versus those comparisons and Original. This SEARCH evaluator
does not emit an equal-time result. Do not invent one or claim superiority
without an expanded, separately frozen equal-time development comparison.
The campaign operator
will independently compute these comparisons from native results. Use actual
metrics; do not assume improvement. A candidate that improves only the scalar
score is not evidence of a successful method. Report all failed comparisons
before allowing expanded experiments or a paper claim.

## Execution discipline

- Keep proposals concise: one hypothesis, at most five implementation steps,
  and only the concrete rubrics needed for this cycle.
- Read the supplied starter once. Use small targeted edits. Do not scan data
  directories or large CSV/JSON artifacts to discover the interface.
- Do not write long introductory commentary or repeatedly restate the plan.
- Follow all original clone, memory, source-isolation and harness rules. An
  implementation agent writes exactly one candidate and exits; the host owns
  its only evaluation. Do not run extra tests or private evaluator code.
- Fixed benchmark: search seed 1701, 400 reference updates, 80 offered deletion
  update RPCs, 1800 seconds per search evaluation and the two assigned A40s.
  No architecture, evaluator, data partition or candidate-interface changes.
- This initial batch has four idea cycles. It has no overall wall-clock limit.
  The model transport has finite per-request waits so a silent request cannot
  consume the entire research session without an explicit failure.
- Only investigation is enabled for this batch. No old final re-evaluation,
  posthoc or paper generation is authorized by this campaign configuration.
