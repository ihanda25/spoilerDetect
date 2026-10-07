# RoBERTa on the original IMDb test split

Completed October 6, 2026. All 63,836 validation and 64,405 test reviews scored on Tesla T4; approximately 63 minutes including startup. Raw review texts retokenized with pinned RoBERTa, first 512 tokens, FP32, batch 16. 10,252 test reviews exceeded the token budget.

Each model threshold selected on validation for maximum spoiler F1, then frozen before test. Results use review-level dataset tags.

| Model | Threshold | Precision | Recall | Spoiler F1 | False positives | Missed spoilers |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM review epoch2 | 0.215792 | 53.3% | 68.5% | 59.9% | 9,945 | 5,229 |
| RoBERTa pinned external checkpoint | 0.652240 | 52.5% | 65.7% | 58.4% | 9,848 | 5,692 |

RoBERTa is slightly worse at this selected operating point: 1.6 percentage points lower F1, 463 more missed spoilers, 97 fewer false positives. ROC-AUC 0.811 vs MiniLM 0.821; average precision 0.625 vs 0.659. These small observed differences have not been tested for statistical significance. This external checkpoint is not a demonstrated improvement on this large dataset; it remains useful as a separate baseline for product excerpts, where performance can differ.

At default threshold 0.5, RoBERTa precision 42.8%, recall 81.8%, F1 56.2%, accuracy 67.2%; higher recall is accompanied by 18,116 false positives. Default thresholds are not equivalent operating points across models.

External checkpoint training overlap with our IMDb test is unknown; these are descriptive same-data scores, not proven independent generalization. Movie-disjoint splits apply to our MiniLM, but external training provenance cannot establish RoBERTa independence. IMDb whole-review tags differ from the product excerpt policy and do not replace the product evaluation set.

Downloaded full results/checkpoints. Local verification confirmed exact original label/source-line order for both splits, recomputed validation threshold and all test metrics. See `test-results.json`, `threshold-selection.json`, `verification.json`, and `roberta-imdb-results.zip`.
