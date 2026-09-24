"""Full-data MiniLM training with review-level supervision over all text windows.

Run offline after prepare_full.py. Resume from the last checkpoint with --resume.
The pilot checkpoint and report are never overwritten.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from finetune import ROOT, metrics, EXAMPLES

OUT = ROOT / 'models/minilm-full'
REPORT = ROOT / 'reports/full'
EPOCHS = 3
CHUNK_BUDGET = 24
LR = 2e-5


class Reviews:
    def __init__(self, split):
        directory = ROOT / 'data/full'
        self.tokens = np.memmap(directory / f'{split}.tokens.bin', dtype=np.int32, mode='r')
        self.offsets = np.load(directory / f'{split}.offsets.npy', mmap_mode='r')
        self.labels = np.load(directory / f'{split}.labels.npy', mmap_mode='r')
        self.lines = np.load(directory / f'{split}.source_lines.npy', mmap_mode='r')
        self.counts = np.maximum(1, (np.maximum(0, np.diff(self.offsets) - 510) + 445) // 446 + 1)

    def __len__(self):
        return len(self.labels)

    def chunks(self, index):
        ids = self.tokens[self.offsets[index]:self.offsets[index + 1]]
        result = []
        for start in range(0, max(1, len(ids)), 446):
            result.append([101] + ids[start:start + 510].tolist() + [102])
            if start + 510 >= len(ids):
                break
        return result

    def batches(self, epoch=None):
        order = np.arange(len(self))
        if epoch is not None:
            np.random.default_rng(42 + epoch).shuffle(order)
        current, count = [], 0
        for index in order:
            if current and count + self.counts[index] > CHUNK_BUDGET:
                yield current
                current, count = [], 0
            current.append(int(index))
            count += int(self.counts[index])
        if current:
            yield current


def collate(dataset, indices):
    chunks, sizes = [], []
    for index in indices:
        windows = dataset.chunks(index)
        chunks.extend(windows)
        sizes.append(len(windows))
    width = min(512, int(math.ceil(max(map(len, chunks)) / 8) * 8))
    ids = torch.zeros((len(chunks), width), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, chunk in enumerate(chunks):
        ids[i, :len(chunk)] = torch.tensor(chunk)
        mask[i, :len(chunk)] = 1
    return {'input_ids': ids, 'attention_mask': mask}, sizes


def review_logits(model, inputs, sizes, device):
    # Memory is bounded for unusually long reviews using gradient checkpointing
    # during training and separate forward microbatches of at most 24 windows.
    margins = []
    for start in range(0, len(inputs['input_ids']), CHUNK_BUDGET):
        logits = model(**{k: v[start:start + CHUNK_BUDGET].to(device) for k,v in inputs.items()}).logits
        margins.append(logits[:, 1] - logits[:, 0])
    margins = torch.cat(margins)
    return torch.stack([group.max() for group in margins.split(sizes)])


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def render_report(state):
    data = json.loads((ROOT / 'reports/full-data.json').read_text())
    lines = ['# Full-data MiniLM experiment', '', f'**Status: {state["status"]}**', '',
        'This is a full-data experiment, not an established performance ceiling. The pilot report remains in `../FINETUNING_RESULTS.md`.', '',
        '## Method', '',
        '- Start from pretrained MiniLM with a fresh spoiler classification head; seed 42.',
        '- Use all eligible training reviews; preserve the original movie-disjoint split.',
        '- Exclude all duplicated normalized texts, including conflicting duplicates.',
        '- Use 512-token windows (510 text tokens plus special tokens), overlapping by 64 text tokens.',
        '- Cover every token; no head-only truncation or maximum number of windows per review.',
        '- Pool the largest spoiler logit margin across a review’s windows, then apply one review-level binary loss. Individual windows do not inherit positive labels.',
        '- AdamW, learning rate 2e-5, 5% warmup, linear decay, up to 3 complete epochs; no class oversampling.',
        '- Select checkpoint by validation average precision, then choose an F1 threshold on validation only.',
        '- Compare initial and trained heads on the same complete test split; report fixed 0.5 and validation-selected thresholds.', '',
        '| Split | Reviews | Spoilers | Exceed 192 tokens | Exceed 512 tokens |',
        '| --- | ---: | ---: | ---: | ---: |']
    for split, info in data['splits'].items():
        lines.append(f'| {split} | {info["reviews"]:,} | {info["spoilers"]:,} | {info["over_192"]:,} | {info["over_512"]:,} |')
    lines += ['', '## Progress', '', f'Last update: {state["updated_at"]}', '',
              f'Phase: {state.get("phase", "initializing")}; epoch: {state.get("epoch", 0)}; '
              f'batch: {state.get("batch", 0)}.', '']
    if 'results' in state:
        r = state['results']
        lines += ['## Test results', '', '| Metric | Before: untrained head | After: threshold 0.5 |',
                  '| --- | ---: | ---: |']
        for metric in ['accuracy','precision','recall','f1','roc_auc','average_precision','brier_score']:
            lines.append(f'| {metric} | {r["before"][metric]:.4f} | {r["after"][metric]:.4f} |')
        lines += ['', f'Always-no-spoiler accuracy: {r["majority_accuracy"]:.4f}.', '',
                  f'Validation-selected threshold: {r["threshold"]:.4f}; test metrics at that threshold:',
                  '', '```json', json.dumps(r['threshold_metrics'], indent=2), '```', '',
                  '## Fixed invented examples', '', '| Text | Expected | Before spoiler score | After spoiler score |',
                  '| --- | --- | ---: | ---: |']
        for x in r['examples']:
            lines.append(f'| {x["text"]} | {x["label"]} | {x["before"]:.1%} | {x["after"]:.1%} |')
    else:
        lines += ['Final test results are pending. No completed performance claim is made.', '']
    lines += ['## Limitations', '',
        '- Scores remain uncalibrated. Maximum pooling can increase false positives on long reviews.',
        '- Whole-review supervision does not provide sentence-level labels. Short-snippet performance needs separate evaluation.',
        '- All text is covered, but each window sees at most 512 tokens; cross-window relationships are not modeled.',
        '- One seed and one learning-rate setting do not establish a ceiling.',
        '- Same movie partitions as the pilot, but a larger test set and different text coverage: headline scores are not directly comparable.',
        '- Source: Rishabh Misra, IMDb Spoiler Dataset v1, DOI 10.13140/RG.2.2.11584.15362.', '']
    (REPORT / 'FULL_FINETUNING_RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(6)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    datasets = {s: Reviews(s) for s in ['train','validation','test']}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    assert (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id) == (101,102,0)
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'models/base', num_labels=2,
        id2label={0:'no_spoiler',1:'spoiler'}, label2id={'no_spoiler':0,'spoiler':1},
        attn_implementation='eager', local_files_only=True).to(device)
    model.gradient_checkpointing_enable()
    plans = [list(datasets['train'].batches(epoch)) for epoch in range(EPOCHS)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = sum(map(len, plans))
    warmup = max(1, int(steps * .05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: step / warmup if step < warmup else max(0, (steps-step)/(steps-warmup)))
    state = {'status':'running','device':str(device),'history':[], 'phase':'initial evaluation'}
    started = time.time()

    def update(**fields):
        state.update(fields)
        state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        atomic_json(REPORT / 'status.json', state)
        render_report(state)

    def predict(split):
        model.eval()
        scores = []
        with torch.inference_mode():
            for step, indices in enumerate(datasets[split].batches(), 1):
                inputs, sizes = collate(datasets[split], indices)
                scores.extend(review_logits(model, inputs, sizes, device).sigmoid().cpu().tolist())
                if step % 200 == 0:
                    update(evaluation_reviews=len(scores))
                    print(f'Evaluation {split}: {len(scores)}/{len(datasets[split])}', flush=True)
        return np.asarray(scores)

    def examples():
        model.eval()
        inputs = tokenizer([x[0] for x in EXAMPLES], padding=True, return_tensors='pt').to(device)
        with torch.inference_mode():
            return model(**inputs).logits.softmax(-1)[:,1].cpu().tolist()

    best, start_epoch, start_batch = -1., 0, 0
    if args.resume:
        # This is an optimizer checkpoint generated locally by this script.
        checkpoint = torch.load(OUT / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch, start_batch, best = checkpoint['epoch'], checkpoint['next_batch'], checkpoint['best']
        state['history'] = checkpoint['history']
        torch.set_rng_state(checkpoint['rng'])
        if device.type == 'mps' and checkpoint.get('mps_rng') is not None:
            torch.mps.set_rng_state(checkpoint['mps_rng'])
        del checkpoint
    update()
    if not (REPORT / 'before.npy').exists():
        before = predict('test')
        np.save(REPORT / 'before.npy', before)
        atomic_json(REPORT / 'before-examples.json', examples())
    before = np.load(REPORT / 'before.npy')

    def save(epoch, next_batch):
        payload = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(), 'epoch':epoch, 'next_batch':next_batch,
            'best':best, 'history':state['history'], 'rng':torch.get_rng_state(),
            'mps_rng':torch.mps.get_rng_state() if device.type=='mps' else None}
        torch.save(payload, OUT / 'last.tmp')
        (OUT / 'last.tmp').replace(OUT / 'last.pt')

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        loss_total, review_count = 0., 0
        epoch_start = time.time()
        update(phase='training', epoch=epoch+1, batch=start_batch, total_batches=len(plans[epoch]))
        for step in range(start_batch, len(plans[epoch])):
            indices = plans[epoch][step]
            inputs, sizes = collate(datasets['train'], indices)
            target = torch.tensor(datasets['train'].labels[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            margins = review_logits(model, inputs, sizes, device)
            loss = F.binary_cross_entropy_with_logits(margins, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            loss_total += loss.item() * len(indices)
            review_count += len(indices)
            if (step+1) % 50 == 0:
                elapsed = time.time()-epoch_start
                eta = elapsed / (step+1-start_batch) * (len(plans[epoch])-step-1)
                update(batch=step+1, mean_loss=loss_total/review_count,
                       epoch_remaining_minutes=eta/60, session_train_reviews=review_count)
                print(f'Epoch {epoch+1}/{EPOCHS} batch {step+1}/{len(plans[epoch])} loss={loss_total/review_count:.4f} epoch ETA={eta/60:.1f}m', flush=True)
            if (step+1) % 500 == 0:
                save(epoch, step+1)
        update(phase='validation')
        validation = predict('validation')
        scores = metrics(datasets['validation'].labels, validation)
        state['history'].append({'epoch':epoch+1,'validation':scores})
        if scores['average_precision'] > best:
            best = scores['average_precision']
            model.save_pretrained(OUT / 'best')
            tokenizer.save_pretrained(OUT / 'best')
            np.save(REPORT / 'best-validation.npy', validation)
        save(epoch+1, 0)
        start_batch = 0
        print('VALIDATION', scores, flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(OUT / 'best', attn_implementation='eager', local_files_only=True).to(device)
    update(phase='final test evaluation')
    after = predict('test')
    np.save(REPORT / 'after.npy', after)
    validation = np.load(REPORT / 'best-validation.npy')
    p, r, thresholds = precision_recall_curve(datasets['validation'].labels, validation)
    f1 = 2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f1)])
    test_labels = datasets['test'].labels
    threshold_results = metrics(test_labels, (after >= threshold).astype(float))
    threshold_results = {k:threshold_results[k] for k in ['accuracy','precision','recall','f1','confusion_matrix']}
    result = {'before':metrics(test_labels,before), 'after':metrics(test_labels,after),
        'threshold':threshold,'threshold_metrics':threshold_results,
        'majority_accuracy':float((test_labels==0).mean()), 'elapsed_session_seconds':time.time()-started,
        'examples':[{'text':t,'label':y,'before':b,'after':a} for (t,y),b,a in
            zip(EXAMPLES,json.loads((REPORT/'before-examples.json').read_text()),examples())]}
    atomic_json(REPORT/'results.json',result)
    atomic_json(OUT/'best/inference_settings.json',{'max_length':512,'overlap':64,'pooling':'max_logit_margin',
        'threshold':threshold,'calibrated':False})
    update(status='complete',phase='complete',results=result)
    print('COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        REPORT.mkdir(parents=True, exist_ok=True)
        atomic_json(REPORT/'failure.json',{'error':repr(error),'time':time.strftime('%Y-%m-%d %H:%M:%S %z')})
        if (REPORT/'status.json').exists():
            state = json.loads((REPORT/'status.json').read_text())
            state.update(status='failed', phase=repr(error), updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
            atomic_json(REPORT/'status.json', state)
            render_report(state)
        raise
