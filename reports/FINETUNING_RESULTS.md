# MiniLM: before and after fine-tuning

## Result

The fine-tuned model achieves **77.4% accuracy** versus 72.5% for always predicting no spoiler. It detects **42.5% of labeled spoilers**, with 63.2% precision. This is a first baseline, not reliable spoiler protection.

On the fixed invented short-text checks, it misses **3 of 3 explicit spoilers** at the 0.5 threshold. Review-level improvement has not translated into successful short-snippet detection here.

## What was compared

**Before:** pretrained MiniLM with a randomly initialized two-class classification head. MiniLM has no built-in spoiler labels; this is an untrained-head baseline, not a meaningful zero-shot spoiler detector.

**After:** the same model fine-tuned end-to-end on IMDb review labels. The best of two epochs was selected using validation spoiler F1, then evaluated on the same untouched test examples as the before run.

## Data and setup

- Model: `microsoft/MiniLM-L12-H384-uncased`; revision `44acabbec0ef496f6dbc93adadea57f376b7c0ec`.
- Source: [IMDb Spoiler Dataset v1, Rishabh Misra](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset).
- Local pilot on a sampled subset, not training on all 573,913 reviews.
- Movie-disjoint splits; all repeated normalized review texts excluded before sampling.
- Related movies/franchises and common reviewers may still cross splits.
- Seed: 42; device: mps; batch size: 16; epochs: 2; AdamW learning rate: 0.00002.
- Fixed decision threshold: 0.5. No threshold search on test data.
- Training/evaluation runtime including initial inference: 3.7 minutes.

| Split | Reviews | Movies | Spoilers | Truncated at 192 tokens |
| --- | ---: | ---: | ---: | ---: |
| train | 8000 | 1159 | 2109 | 5089 |
| validation | 1000 | 153 | 272 | 656 |
| test | 1000 | 154 | 275 | 640 |

## Held-out test results

| Metric | Before: untrained head | After: fine-tuned | Always no spoiler |
| --- | ---: | ---: | ---: |
| accuracy | 0.2750 | 0.7740 | 0.7250 |
| precision | 0.2750 | 0.6324 | 0.0000 |
| recall | 1.0000 | 0.4255 | 0.0000 |
| f1 | 0.4314 | 0.5087 | 0.0000 |
| roc_auc | 0.5028 | 0.7782 | 0.5000 |
| average_precision | 0.2685 | 0.6013 | 0.2750 |
| brier_score | 0.2523 | 0.1594 | 0.2750 |

Precision, recall, and F1 refer to the spoiler class. Higher is better except Brier score (mean squared probability error), where lower is better. Accuracy alone can hide missed spoilers.

### Confusion matrices

| Run | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| before | 0 | 725 | 0 | 275 |
| after | 657 | 68 | 158 | 117 |

## Predictions on invented examples

These examples were fixed before training and contain invented plot events. They are qualitative checks, not a representative benchmark. Percentages are **uncalibrated model scores**, not established spoiler probabilities.

| Text | Expected | Before prediction (spoiler score) | After prediction (spoiler score) |
| --- | --- | --- | --- |
| The acting was excellent and the soundtrack was beautiful. | no spoiler | spoiler (50.5%) | no spoiler (11.4%) |
| In the final scene, the detective discovers that her brother is the murderer. | spoiler | spoiler (50.5%) | no spoiler (30.5%) |
| I cannot wait to watch the next episode this weekend. | no spoiler | spoiler (50.4%) | no spoiler (12.1%) |
| The captain dies saving her brother, and the ship sinks in the ending. | spoiler | spoiler (50.5%) | no spoiler (46.4%) |
| That finale killed me. What an incredible performance! | no spoiler | spoiler (50.4%) | no spoiler (11.9%) |
| Will the captain die next season? Here is my theory. | no spoiler | spoiler (50.5%) | no spoiler (13.9%) |
| The missing daughter was alive all along, hiding in the basement. | spoiler | spoiler (50.5%) | no spoiler (18.8%) |
| An interview with the actor about costumes and filming locations. | no spoiler | spoiler (50.4%) | no spoiler (12.1%) |

## Validation history

| Epoch | Mean training loss | Validation spoiler F1 |
| --- | ---: | ---: |
| 1 | 0.5732 | 0.0000 |
| 2 | 0.5091 | 0.3974 |

## Limitations

- Whole-review labels can be noisy and subjective; they are not sentence-level ground truth.
- Inputs use only the first 192 tokens. Spoilers later in a review can be missed while the review remains labeled positive.
- No claims about YouTube titles, webpages, video, unfamiliar genres, or calibrated percentages follow from this experiment.
- No repeated-seed experiments or confidence intervals; small differences may be sampling variation.
- Untrained-head performance depends on random initialization and is not evidence of the base model’s general language ability.

## Reproduce and use

```sh
.venv/bin/python scripts/prepare_training.py
.venv/bin/python scripts/finetune.py
.venv/bin/python scripts/predict.py "The acting was excellent."
```

Requires the pinned base checkpoint in `models/base`. Saved fine-tuned model: `models/minilm-spoiler`.
Machine-readable results: `reports/results.json`; per-review test scores: `reports/test-predictions.jsonl`.
