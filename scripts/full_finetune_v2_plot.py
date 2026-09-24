"""Full-data MiniLM fine-tuning, v3: review + movie plot_summary pairs.

Same infrastructure as scripts/full_finetune_v2.py (plain truncation, no
windowing, fixed 512-token padding, one epoch per invocation then exit,
resumable checkpoint), but each input is now a BERT-style pair:
[CLS] plot_summary [SEP] review [SEP]
so the model can compare what the review says against what's actually known
to happen in that movie, instead of judging review text in isolation.

Separate output/report paths from v2 so the review-only baseline (best
validation F1 0.5602, epoch 1) is preserved untouched for comparison. See
reports/full-v2-plot/PLAN.md for rationale and status.
"""
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/full'
OUT = ROOT / 'models/minilm-full-plot'
REPORT = ROOT / 'reports/full-v2-plot'
MAX_LENGTH = 512
PLOT_MAX = 224     # covers p90 (216 tokens) of all 1,572 movies' plot_summary; max observed is 267
REVIEW_MAX = MAX_LENGTH - 3 - PLOT_MAX   # 285; -3 for [CLS] and two [SEP]
BATCH = 16
EPOCHS = 3
LR = 2e-5
SEED = 42


def load_plot_summaries(tokenizer):
    with zipfile.ZipFile(ROOT / 'data/raw/imdb/dataset.zip') as z:
        with z.open('IMDB_movie_details.json') as f:
            rows = [json.loads(line) for line in f]
    ids = tokenizer([r['plot_summary'] for r in rows], truncation=True,
                    max_length=PLOT_MAX, add_special_tokens=False)['input_ids']
    return {r['movie_id']: seq for r, seq in zip(rows, ids)}


class PairReviews:
    def __init__(self, split, plot_lookup):
        self.tokens = np.memmap(DATA / f'{split}.tokens.bin', dtype=np.int32, mode='r')
        self.offsets = np.load(DATA / f'{split}.offsets.npy')
        self.labels = np.load(DATA / f'{split}.labels.npy')
        self.movie_ids = np.load(DATA / f'{split}.movie_ids.npy')
        self.plot_lookup = plot_lookup

    def __len__(self):
        return len(self.labels)

    def encode(self, index, tokenizer):
        review_ids = self.tokens[self.offsets[index]:self.offsets[index + 1]][:REVIEW_MAX].tolist()
        plot_ids = self.plot_lookup.get(str(self.movie_ids[index]), [])
        ids = [tokenizer.cls_token_id] + plot_ids + [tokenizer.sep_token_id] + review_ids + [tokenizer.sep_token_id]
        types = [0] * (len(plot_ids) + 2) + [1] * (len(review_ids) + 1)
        pad = MAX_LENGTH - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [tokenizer.pad_token_id] * pad
        types = types + [0] * pad
        return ids, mask, types


def batch_tensors(dataset, indices, tokenizer, device):
    ids, masks, types = [], [], []
    for i in indices:
        a, m, t = dataset.encode(i, tokenizer)
        ids.append(a)
        masks.append(m)
        types.append(t)
    return ({'input_ids': torch.tensor(ids, device=device),
             'attention_mask': torch.tensor(masks, device=device),
             'token_type_ids': torch.tensor(types, device=device)},
            torch.tensor(dataset.labels[indices].astype(np.int64), device=device))


def metrics(labels, scores):
    predictions = np.array(scores) >= 0.5
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predictions)), 'precision': float(p),
            'recall': float(r), 'f1': float(f), 'roc_auc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)),
            'brier_score': float(brier_score_loss(labels, scores)),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def evaluate(model, dataset, tokenizer, device, log_prefix):
    model.eval()
    scores = []
    n = len(dataset)
    with torch.inference_mode():
        for start in range(0, n, BATCH):
            indices = np.arange(start, min(start + BATCH, n))
            inputs, _ = batch_tensors(dataset, indices, tokenizer, device)
            scores.extend(model(**inputs).logits.softmax(-1)[:, 1].cpu().tolist())
            if start % (BATCH * 200) == 0:
                print(f'{log_prefix}: {start}/{n}', flush=True)
    return scores


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def log(msg):
    print(msg, flush=True)
    with (REPORT / 'training.log').open('a') as f:
        f.write(msg + '\n')


def main():
    start_time = time.time()
    torch.manual_seed(SEED)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    plot_lookup = load_plot_summaries(tokenizer)
    log(f'Loaded plot summaries for {len(plot_lookup)} movies '
        f'(PLOT_MAX={PLOT_MAX}, REVIEW_MAX={REVIEW_MAX})')
    train = PairReviews('train', plot_lookup)
    validation = PairReviews('validation', plot_lookup)

    checkpoint_path = OUT / 'checkpoint.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            ROOT / 'models/base', num_labels=2, local_files_only=True)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        optimizer.load_state_dict(checkpoint['optimizer'])
        completed_epochs = checkpoint['completed_epochs']
        best_f1 = checkpoint['best_f1']
        history = checkpoint['history']
        log(f'Resumed from checkpoint: {completed_epochs} epoch(s) already completed, best_f1={best_f1:.4f}')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            ROOT / 'models/base', num_labels=2, local_files_only=True,
            id2label={0: 'no_spoiler', 1: 'spoiler'}, label2id={'no_spoiler': 0, 'spoiler': 1})
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        completed_epochs = 0
        best_f1 = -1.
        history = []
        log('Starting fresh from pretrained base checkpoint.')

    if completed_epochs >= EPOCHS:
        log(f'All {EPOCHS} epochs already completed. Nothing to do.')
        return

    epoch = completed_epochs + 1
    n_train = len(train)
    steps_per_epoch = (n_train + BATCH - 1) // BATCH
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = max(1, int(total_steps * 0.1))
    step0 = completed_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0., (total_steps - step) / (total_steps - warmup_steps))

    rng = np.random.default_rng(SEED + epoch)
    order = rng.permutation(n_train)

    atomic_json(REPORT / 'status.json', {
        'status': 'running', 'device': str(device), 'phase': 'training',
        'epoch': epoch, 'epochs_planned': EPOCHS, 'step': 0, 'total_steps_this_epoch': steps_per_epoch,
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z')})

    log(f'=== Epoch {epoch}/{EPOCHS} starting, {steps_per_epoch} steps ===')
    model.train()
    total_loss = 0.
    for step, start in enumerate(range(0, n_train, BATCH), 1):
        indices = order[start:start + BATCH]
        inputs, labels = batch_tensors(train, indices, tokenizer, device)
        for g in optimizer.param_groups:
            g['lr'] = LR * lr_at(step0 + step)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**inputs, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if step % 200 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / step * (steps_per_epoch - step)
            log(f'Epoch {epoch}/{EPOCHS} step {step}/{steps_per_epoch} '
                f'loss={total_loss/step:.4f} elapsed={elapsed:.0f}s eta_this_epoch={eta/60:.1f}m')
            atomic_json(REPORT / 'status.json', {
                'status': 'running', 'device': str(device), 'phase': 'training',
                'epoch': epoch, 'epochs_planned': EPOCHS, 'step': step,
                'total_steps_this_epoch': steps_per_epoch, 'mean_loss': total_loss / step,
                'eta_this_epoch_minutes': eta / 60,
                'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z')})

    log(f'Epoch {epoch}/{EPOCHS} training done in {time.time()-start_time:.0f}s, running validation...')
    atomic_json(REPORT / 'status.json', {
        'status': 'running', 'device': str(device), 'phase': 'validation',
        'epoch': epoch, 'epochs_planned': EPOCHS,
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z')})
    val_scores = evaluate(model, validation, tokenizer, device, f'Epoch {epoch} validation')
    val_metrics = metrics(validation.labels.tolist(), val_scores)
    history.append({'epoch': epoch, 'mean_training_loss': total_loss / steps_per_epoch, 'validation': val_metrics})
    log(f'Epoch {epoch}/{EPOCHS} VALIDATION {val_metrics}')

    improved = val_metrics['f1'] > best_f1
    if improved:
        best_f1 = val_metrics['f1']
        model.save_pretrained(OUT / 'best')
        tokenizer.save_pretrained(OUT / 'best')
        log(f'New best model saved to {OUT / "best"} (f1={best_f1:.4f})')

    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'completed_epochs': epoch, 'best_f1': best_f1, 'history': history}, checkpoint_path)

    atomic_json(REPORT / 'status.json', {
        'status': 'epoch_complete', 'device': str(device), 'epoch': epoch, 'epochs_planned': EPOCHS,
        'best_f1': best_f1, 'validation': val_metrics, 'history': history,
        'elapsed_this_run_seconds': time.time() - start_time,
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'next_step': 'Awaiting explicit user decision on the next epoch (see PLAN.md policy).'})
    log(f'=== Epoch {epoch}/{EPOCHS} complete and saved. Stopping (one epoch per invocation). ===')


if __name__ == '__main__':
    main()
