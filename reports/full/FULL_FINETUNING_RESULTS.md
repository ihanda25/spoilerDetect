# Full-data MiniLM experiment

**Status: running**

This is a full-data experiment, not an established performance ceiling. The pilot report remains in `../FINETUNING_RESULTS.md`.

## Method

- Start from pretrained MiniLM with a fresh spoiler classification head; seed 42.
- Use all eligible training reviews; preserve the original movie-disjoint split.
- Exclude all duplicated normalized texts, including conflicting duplicates.
- Use 512-token windows (510 text tokens plus special tokens), overlapping by 64 text tokens.
- Cover every token; no head-only truncation or maximum number of windows per review.
- Pool the largest spoiler logit margin across a review’s windows, then apply one review-level binary loss. Individual windows do not inherit positive labels.
- AdamW, learning rate 2e-5, 5% warmup, linear decay, up to 3 complete epochs; no class oversampling.
- Select checkpoint by validation average precision, then choose an F1 threshold on validation only.
- Compare initial and trained heads on the same complete test split; report fixed 0.5 and validation-selected thresholds.

| Split | Reviews | Spoilers | Exceed 192 tokens | Exceed 512 tokens |
| --- | ---: | ---: | ---: | ---: |
| train | 444,644 | 116,785 | 283,403 | 77,349 |
| validation | 63,836 | 17,358 | 42,021 | 11,860 |
| test | 64,405 | 16,583 | 40,658 | 10,671 |

## Progress

Last update: 2026-09-23 23:21:53 -0400

Phase: training; epoch: 1; batch: 100.

Final test results are pending. No completed performance claim is made.

## Limitations

- Scores remain uncalibrated. Maximum pooling can increase false positives on long reviews.
- Whole-review supervision does not provide sentence-level labels. Short-snippet performance needs separate evaluation.
- All text is covered, but each window sees at most 512 tokens; cross-window relationships are not modeled.
- One seed and one learning-rate setting do not establish a ceiling.
- Same movie partitions as the pilot, but a larger test set and different text coverage: headline scores are not directly comparable.
- Source: Rishabh Misra, IMDb Spoiler Dataset v1, DOI 10.13140/RG.2.2.11584.15362.
