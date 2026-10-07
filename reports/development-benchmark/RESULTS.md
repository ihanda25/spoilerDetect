# Development challenge run

All 64 examples are assistant-authored synthetic challenges and pending human review. These numbers measure agreement with provisional assistant labels, not representative production accuracy. No test rows, training, paid calls, downloads, or threshold tuning.

CPU epoch-2 inference completed in 1.2s. Frozen validation threshold: 0.21579217910766602. Text-only input, right truncation at 512 tokens including special tokens; batch size 4, two CPU threads.

The scorer intentionally uses assistant proposals only, ignoring annotation.human_label even if a human response is later recorded. Human-label evaluation requires a separately named evaluator.

52 binary proposals scored; 12 ambiguous proposals excluded from all binary metrics. All 64 predictions retained. Scores are uncalibrated softmax scores.

| Surface | N | TP | FP | FN | TN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 52 | 10 | 3 | 10 | 29 | 0.769 | 0.500 | 0.606 |
| headline | 12 | 3 | 0 | 1 | 8 | 1.000 | 0.750 | 0.857 |
| comment | 12 | 3 | 3 | 1 | 5 | 0.500 | 0.750 | 0.600 |
| review | 16 | 1 | 0 | 7 | 8 | 1.000 | 0.125 | 0.222 |
| youtube_transcript | 12 | 3 | 0 | 1 | 8 | 1.000 | 0.750 | 0.857 |

## Full-review truncation diagnostic

Twelve reviews exceed 512 tokens. Each work has an early reveal, the same reveal at the end, and a long craft-only control. Late-reveal and control retained token IDs are identical and their scores agree within 1e-5. The model therefore cannot observe the decisive late reveal. Repetitive padding is deliberately artificial.

| Work | Early reveal score | Late reveal score | Negative control score |
| --- | ---: | ---: | ---: |
| glass_harbor | 0.133220 | 0.110122 | 0.110122 |
| last_orchard | 0.165892 | 0.130895 | 0.130895 |
| paper_moon | 0.165844 | 0.128096 | 0.128096 |
| winter_score | 0.257345 | 0.148484 | 0.148484 |

## Interpretation and artifacts

Four fictional works and templated contrasts are correlated, hand-constructed probes. Three extra copies of the missing-antecedent transcript probe are intentional; all are ambiguous and excluded from metrics. Do not interpret sample prevalence, per-surface differences, or these pooled metrics as traffic estimates. No confidence intervals or generalization claims are justified here.

Transcript timestamps are invented passage ranges, not observed video or annotated event spans. No video-level false warnings/hour, temporal localization, real ASR robustness, or viewer-progress accuracy is measured. Fictional story context helps annotators; this review-only model does not receive it.

`predictions.jsonl` retains lengths, truncation, visible token IDs/decoded text and every score. `results.json` records checkpoint/data hashes, environment and counts. `ANNOTATION.md` defines the pending human-review workflow. `annotation.schema.json` describes each row.

Checks passed: finite bounded scores, threshold equality and just-below boundary, perfect/inverted/empty/no-positive cases, invalid-score rejection, row counts/provenance, timestamp ordering, exact right-truncated token input, and late/control input-score equivalence.

Reproduce from repository root (offline):

```sh
.venv/bin/python scripts/build_development_benchmark.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/evaluate_development_benchmark.py
```

Use human review to revise the policy and collect independent real development data before drawing product conclusions. Keep a future test set sealed and do not tune the frozen threshold on this pack.
