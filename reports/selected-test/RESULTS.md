# Held-out test: review-only MiniLM epoch 2

Threshold 0.21579218 selected on validation and frozen before test evaluation. No test threshold tuning.

Test reviews: 64,405.

| Operating point | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Frozen validation threshold | 0.5331 | 0.6847 | 0.5994 |
| Default 0.5 | 0.7230 | 0.4095 | 0.5229 |

False positives: 9,945; false negatives: 5,229.
ROC-AUC: 0.8211; average precision: 0.6588.
