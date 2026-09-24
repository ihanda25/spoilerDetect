"""Render measured results; never supply placeholder performance numbers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(result):
    data = json.loads((ROOT / 'reports/training-data.json').read_text())
    after = result['after']
    demo_spoilers = [r for r in result['examples'] if r['expected_label'] == 1]
    demo_misses = sum(r['after'] < .5 for r in demo_spoilers)
    lines = ['# MiniLM: before and after fine-tuning', '',
        '## Result', '',
        f'The fine-tuned model achieves **{after["accuracy"]:.1%} accuracy** versus '
        f'{result["always_no_spoiler"]["accuracy"]:.1%} for always predicting no spoiler. '
        f'It detects **{after["recall"]:.1%} of labeled spoilers**, with {after["precision"]:.1%} precision. '
        'This is a first baseline, not reliable spoiler protection.', '',
        f'On the fixed invented short-text checks, it misses **{demo_misses} of {len(demo_spoilers)} explicit spoilers** '
        'at the 0.5 threshold. Review-level improvement has not translated into successful short-snippet detection here.', '',
        '## What was compared', '',
        '**Before:** pretrained MiniLM with a randomly initialized two-class classification head. '
        'MiniLM has no built-in spoiler labels; this is an untrained-head baseline, not a meaningful zero-shot spoiler detector.', '',
        '**After:** the same model fine-tuned end-to-end on IMDb review labels. The best of two epochs '
        'was selected using validation spoiler F1, then evaluated on the same untouched test examples as the before run.', '',
        '## Data and setup', '',
        f'- Model: `{result["model"]}`; revision `{result["revision"]}`.',
        '- Source: [IMDb Spoiler Dataset v1, Rishabh Misra](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset).',
        '- Local pilot on a sampled subset, not training on all 573,913 reviews.',
        '- Movie-disjoint splits; all repeated normalized review texts excluded before sampling.',
        '- Related movies/franchises and common reviewers may still cross splits.',
        f'- Seed: {result["seed"]}; device: {result["device"]}; batch size: 16; epochs: 2; AdamW learning rate: 0.00002.',
        '- Fixed decision threshold: 0.5. No threshold search on test data.',
        f'- Training/evaluation runtime including initial inference: {result["elapsed_seconds"]/60:.1f} minutes.', '',
        '| Split | Reviews | Movies | Spoilers | Truncated at 192 tokens |',
        '| --- | ---: | ---: | ---: | ---: |']
    for split, values in data['splits'].items():
        lines.append(f'| {split} | {values["rows"]} | {values["movies"]} | {values["spoilers"]} | {result["truncated_review_counts"][split]} |')
    lines += ['', '## Held-out test results', '',
        '| Metric | Before: untrained head | After: fine-tuned | Always no spoiler |',
        '| --- | ---: | ---: | ---: |']
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision', 'brier_score']:
        values = [result[s][metric] for s in ['before', 'after', 'always_no_spoiler']]
        lines.append(f'| {metric} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |')
    lines += ['', 'Precision, recall, and F1 refer to the spoiler class. Higher is better except Brier score '
              '(mean squared probability error), where lower is better. Accuracy alone can hide missed spoilers.', '',
              '### Confusion matrices', '', '| Run | True negatives | False positives | False negatives | True positives |',
              '| --- | ---: | ---: | ---: | ---: |']
    for run in ['before', 'after']:
        (tn, fp), (fn, tp) = result[run]['confusion_matrix']
        lines.append(f'| {run} | {tn} | {fp} | {fn} | {tp} |')
    lines += ['', '## Predictions on invented examples', '',
        'These examples were fixed before training and contain invented plot events. They are qualitative checks, '
        'not a representative benchmark. Percentages are **uncalibrated model scores**, not established spoiler probabilities.', '',
        '| Text | Expected | Before prediction (spoiler score) | After prediction (spoiler score) |',
        '| --- | --- | --- | --- |']
    for row in result['examples']:
        def prediction(score):
            return f'{"spoiler" if score >= .5 else "no spoiler"} ({score:.1%})'
        lines.append(f'| {row["text"]} | {"spoiler" if row["expected_label"] else "no spoiler"} | {prediction(row["before"])} | {prediction(row["after"])} |')
    lines += ['', '## Validation history', '', '| Epoch | Mean training loss | Validation spoiler F1 |',
              '| --- | ---: | ---: |']
    for epoch in result['history']:
        lines.append(f'| {epoch["epoch"]} | {epoch["mean_training_loss"]:.4f} | {epoch["validation"]["f1"]:.4f} |')
    lines += ['', '## Limitations', '',
        '- Whole-review labels can be noisy and subjective; they are not sentence-level ground truth.',
        '- Inputs use only the first 192 tokens. Spoilers later in a review can be missed while the review remains labeled positive.',
        '- No claims about YouTube titles, webpages, video, unfamiliar genres, or calibrated percentages follow from this experiment.',
        '- No repeated-seed experiments or confidence intervals; small differences may be sampling variation.',
        '- Untrained-head performance depends on random initialization and is not evidence of the base model’s general language ability.', '',
        '## Reproduce and use', '', '```sh',
        '.venv/bin/python scripts/prepare_training.py', '.venv/bin/python scripts/finetune.py',
        '.venv/bin/python scripts/predict.py "The acting was excellent."', '```', '',
        'Requires the pinned base checkpoint in `models/base`. Saved fine-tuned model: `models/minilm-spoiler`.',
        'Machine-readable results: `reports/results.json`; per-review test scores: `reports/test-predictions.jsonl`.', '']
    (ROOT / 'reports/FINETUNING_RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    render(json.loads((ROOT / 'reports/results.json').read_text()))
