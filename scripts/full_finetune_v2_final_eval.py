"""Final test-set before/after report for the v2 full-data MiniLM run.

Run only after all 3 epochs of scripts/full_finetune_v2.py have completed
(reports/full-v2/status.json will say epoch 3/3, epoch_complete). Compares the
pretrained base (fresh, untrained head) against the best-validation-F1 checkpoint
on the full 64,405-review test split.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import EXAMPLES, metrics
from full_finetune_v2 import BATCH, MAX_LENGTH, ROOT, Reviews, batch_tensors, evaluate

REPORT = ROOT / 'reports/full-v2'
OUT = ROOT / 'models/minilm-full'


def score_examples(model, tokenizer, device):
    model.eval()
    inputs = tokenizer([t for t, _ in EXAMPLES], padding='max_length', truncation=True,
                       max_length=MAX_LENGTH, return_tensors='pt').to(device)
    with torch.inference_mode():
        return model(**inputs).logits.softmax(-1)[:, 1].cpu().tolist()


def main():
    status = json.loads((REPORT / 'status.json').read_text())
    if not (status.get('epoch') == status.get('epochs_planned') and status.get('status') == 'epoch_complete'):
        raise SystemExit(f'Not all epochs are complete yet: {status}. Refusing to run final eval.')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    test = Reviews('test')
    labels = test.labels.tolist()

    print('Evaluating pretrained (untrained head) baseline on test set...', flush=True)
    before_model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / 'models/base', num_labels=2, local_files_only=True).to(device)
    before_scores = evaluate(before_model, test, tokenizer, device, 'BEFORE test')
    before_examples = score_examples(before_model, tokenizer, device)
    before_metrics = metrics(labels, before_scores)
    print('BEFORE', before_metrics, flush=True)
    del before_model

    print('Evaluating best fine-tuned checkpoint on test set...', flush=True)
    after_model = AutoModelForSequenceClassification.from_pretrained(
        OUT / 'best', local_files_only=True).to(device)
    after_scores = evaluate(after_model, test, tokenizer, device, 'AFTER test')
    after_examples = score_examples(after_model, tokenizer, device)
    after_metrics = metrics(labels, after_scores)
    print('AFTER', after_metrics, flush=True)

    result = {
        'model': 'microsoft/MiniLM-L12-H384-uncased', 'max_length': MAX_LENGTH, 'batch_size': BATCH,
        'epochs': status['epochs_planned'], 'device': str(device),
        'no_windowing': True, 'best_validation_f1': status['best_f1'], 'history': status['history'],
        'before': before_metrics, 'after': after_metrics,
        'always_no_spoiler': metrics(labels, [0.] * len(labels)),
        'examples': [{'text': t, 'expected_label': l, 'before': b, 'after': a}
                     for (t, l), b, a in zip(EXAMPLES, before_examples, after_examples)],
    }
    (REPORT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')

    lines = ['# Full-data MiniLM experiment v2 (no windowing) — final results', '',
        '**Status: complete**', '',
        'Plain 512-token truncation, no overlapping windows, no max-pooling. See '
        '`PLAN.md` in this directory for the full rationale. The pilot report remains '
        'in `../FINETUNING_RESULTS.md`; the abandoned windowed attempt is in `../full/`.',
        '', '## Test-set results (64,405 reviews)', '',
        '| | accuracy | precision | recall | f1 | roc_auc | average_precision |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        f'| Always no-spoiler | {result["always_no_spoiler"]["accuracy"]:.3f} | - | 0.000 | 0.000 | - | - |',
        f'| Before (untrained head) | {before_metrics["accuracy"]:.3f} | {before_metrics["precision"]:.3f} | '
        f'{before_metrics["recall"]:.3f} | {before_metrics["f1"]:.3f} | {before_metrics["roc_auc"]:.3f} | '
        f'{before_metrics["average_precision"]:.3f} |',
        f'| After (fine-tuned, {status["epochs_planned"]} epochs) | {after_metrics["accuracy"]:.3f} | '
        f'{after_metrics["precision"]:.3f} | {after_metrics["recall"]:.3f} | {after_metrics["f1"]:.3f} | '
        f'{after_metrics["roc_auc"]:.3f} | {after_metrics["average_precision"]:.3f} |',
        '', 'Best validation F1 during training: ' + f'{status["best_f1"]:.4f}', '',
        '## Limitations', '',
        '- Reviews over 512 tokens (17.4% of train) are truncated; content past that '
        'point is invisible to the model.',
        '- Scores are not calibrated probabilities.',
        '- Same movie-disjoint split as the pilot and the abandoned windowed attempt.',
        '- Source: Rishabh Misra, IMDb Spoiler Dataset v1, DOI 10.13140/RG.2.2.11584.15362.',
        '']
    (REPORT / 'FULL_V2_RESULTS.md').write_text('\n'.join(lines))
    print('Report: reports/full-v2/FULL_V2_RESULTS.md', flush=True)


if __name__ == '__main__':
    main()
