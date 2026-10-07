"""One held-out evaluation: review epoch 2, validation-selected threshold frozen."""
import json
import time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import full_finetune_v2 as review
from tune_validation_thresholds import operating_point

ROOT = review.ROOT
OUT = ROOT / 'reports/selected-test'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'results.json').exists():
        print('Test results already exist; refusing an unnecessary repeat.', flush=True)
        return
    start = time.time()
    selection = json.loads((ROOT / 'reports/threshold-tuning/results.json').read_text())
    threshold = selection['review_epoch2']['max_f1']['threshold']
    # Freeze the choice before loading test data. Do not search thresholds on test.
    review.atomic_json(OUT / 'selection.json', dict(checkpoint='models/minilm-full/checkpoint.pt',
        epoch=2, threshold=threshold, selection_source='validation maximum F1',
        selected_at=time.strftime('%Y-%m-%d %H:%M:%S %z')))
    review.atomic_json(OUT / 'status.json', dict(status='running', threshold=threshold))
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    torch.set_num_threads(6)
    device = torch.device('mps')
    checkpoint = torch.load(ROOT / 'models/minilm-full/checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['completed_epochs'] == 2
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'models/base', local_files_only=True, num_labels=2)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    test = review.Reviews('test')
    scores = np.asarray(review.evaluate(model, test, tokenizer, device, 'Selected test'))
    assert len(scores) == len(test) and np.isfinite(scores).all()
    np.savez_compressed(OUT / 'scores.npz', scores=scores, labels=test.labels)
    tuned = operating_point(test.labels, scores, threshold)
    default = review.metrics(test.labels, scores)
    result = dict(checkpoint='review_epoch2', split='test', reviews=len(test),
        frozen_validation_threshold=threshold, selected_threshold_metrics=tuned,
        default_threshold_metrics=default, elapsed_seconds=time.time()-start)
    review.atomic_json(OUT / 'results.json', result)
    lines = ['# Held-out test: review-only MiniLM epoch 2', '',
        f'Threshold {threshold:.8f} selected on validation and frozen before test evaluation. No test threshold tuning.', '',
        f'Test reviews: {len(test):,}.', '',
        '| Operating point | Precision | Recall | F1 |', '| --- | ---: | ---: | ---: |',
        f"| Frozen validation threshold | {tuned['precision']:.4f} | {tuned['recall']:.4f} | {tuned['f1']:.4f} |",
        f"| Default 0.5 | {default['precision']:.4f} | {default['recall']:.4f} | {default['f1']:.4f} |", '',
        f"False positives: {tuned['false_positives']:,}; false negatives: {tuned['false_negatives']:,}.",
        f"ROC-AUC: {default['roc_auc']:.4f}; average precision: {default['average_precision']:.4f}."]
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    review.atomic_json(OUT / 'status.json', dict(status='complete', elapsed_seconds=time.time()-start))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        review.atomic_json(OUT / 'status.json', dict(status='failed', error=str(exc)))
        raise
