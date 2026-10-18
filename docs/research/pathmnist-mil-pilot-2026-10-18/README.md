# PathMNIST MIL pilot evidence

See the [Chinese experiment and integration report](../../pathmnist-mil-pilot-2026-10-18.md).

- `programs/`: frozen baseline and two evaluated candidates; source, program spec, validation metrics, training traces and available rubric/idea records.
- `data_manifest.json`: dataset, construction constraints and data hashes.
- `source_index.json`: primary literature links, availability, text hashes and retrieval errors. Full texts are not redistributed here.
- `holdout_comparison_plan.json`: comparison programs fixed before holdout evaluation.
- `holdout_summary.json`: baseline, selected incumbent and best validation-score challenger, seeds 11 and 19. The selected incumbent is the baseline and reuses its evaluation.
- `interface_checks.json`: permutation checks on synthetic inputs, including equal attention scores.
- `posthoc_score_change.json`: descriptive analysis of already-saved validation predictions, with no additional training or method changes.
- `SHA256.json`: hashes of these exported evidence files, excluding itself and this index.

These are native development results from synthetic bags of real pathology patches. They are not a clinical WSI study, a patient-independent validation, or evidence of a new algorithm.
