"""Compare saved checkpoints on validation only; never train or access test data."""
import gc
import json
import time
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import full_finetune_v2 as review
import full_finetune_v2_plot as plot

ROOT = review.ROOT
OUT = ROOT / 'reports/threshold-tuning'


def operating_point(labels, scores, threshold):
    tn, fp, fn, tp = confusion_matrix(labels, scores >= threshold, labels=[0, 1]).ravel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return dict(threshold=float(threshold), precision=float(precision), recall=float(recall),
                f1=float(2 * tp / max(1, 2 * tp + fp + fn)),
                false_positives=int(fp), false_negatives=int(fn))


def summarize(labels, scores):
    p, r, thresholds = precision_recall_curve(labels, scores)
    p, r = p[:-1], r[:-1]
    f = 2 * p * r / np.maximum(p + r, 1e-15)
    result = {'default': operating_point(labels, scores, 0.5),
              'max_f1': operating_point(labels, scores, thresholds[np.argmax(f)])}
    for target in (0.6, 0.7, 0.8, 0.9):
        candidates = np.flatnonzero(r >= target)
        best = candidates[np.argmax(p[candidates])]
        result[f'recall_{target:.0%}'] = operating_point(labels, scores, thresholds[best])
    result['average_precision'] = float(review.average_precision_score(labels, scores))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = torch.device('mps')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; refusing an unexpectedly slow CPU run')
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    results = {}
    specs = [('review_epoch1', review, ROOT / 'models/minilm-full/best'),
             ('review_epoch2', review, ROOT / 'models/minilm-full/checkpoint.pt'),
             ('plot_epoch1', plot, ROOT / 'models/minilm-full-plot/best')]
    for name, module, path in specs:
        review.atomic_json(OUT / 'status.json', dict(status='running', checkpoint=name))
        dataset = (review.Reviews('validation') if module is review else
                   plot.PairReviews('validation', plot.load_plot_summaries(tokenizer)))
        model = AutoModelForSequenceClassification.from_pretrained(
            ROOT / 'models/base' if path.is_file() else path, local_files_only=True, num_labels=2)
        if path.is_file():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            assert checkpoint['completed_epochs'] == 2
            model.load_state_dict(checkpoint['model'])
            del checkpoint
        model.to(device)
        scores = np.asarray(module.evaluate(model, dataset, tokenizer, device, name))
        assert len(scores) == len(dataset) and np.isfinite(scores).all()
        np.savez_compressed(OUT / f'{name}.npz', scores=scores, labels=dataset.labels)
        results[name] = summarize(dataset.labels, scores)
        review.atomic_json(OUT / 'results.json', results)
        print(name, json.dumps(results[name]), flush=True)
        del model, dataset
        gc.collect()
        torch.mps.empty_cache()
    lines = ['# Validation threshold comparison', '',
             'Thresholds selected on validation, not unbiased test results. No training performed.', '',
             '| Checkpoint | Operating point | Threshold | Precision | Recall | F1 | False positives |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, values in results.items():
        for point, v in values.items():
            if isinstance(v, dict):
                lines.append(f"| {name} | {point} | {v['threshold']:.4f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['false_positives']} |")
    (OUT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    review.atomic_json(OUT / 'status.json', dict(status='complete', elapsed_seconds=time.time()-start))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        review.atomic_json(OUT / 'status.json', dict(status='failed', error=str(exc)))
        raise
