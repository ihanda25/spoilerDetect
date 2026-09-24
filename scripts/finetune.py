"""Run an honest untrained-head vs fine-tuned MiniLM comparison locally."""
import json
import os
import random
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 192
SEED = 42
EXAMPLES = [
    ('The acting was excellent and the soundtrack was beautiful.', 0),
    ('In the final scene, the detective discovers that her brother is the murderer.', 1),
    ('I cannot wait to watch the next episode this weekend.', 0),
    ('The captain dies saving her brother, and the ship sinks in the ending.', 1),
    ('That finale killed me. What an incredible performance!', 0),
    ('Will the captain die next season? Here is my theory.', 0),
    ('The missing daughter was alive all along, hiding in the basement.', 1),
    ('An interview with the actor about costumes and filming locations.', 0),
]


def metrics(labels, scores):
    predictions = np.array(scores) >= 0.5
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predictions)), 'precision': float(p),
            'recall': float(r), 'f1': float(f), 'roc_auc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)),
            'brier_score': float(brier_score_loss(labels, scores)),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def main():
    start = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(6)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:', device, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ROOT / 'models/base', num_labels=2, local_files_only=True,
        id2label={0: 'no_spoiler', 1: 'spoiler'}, label2id={'no_spoiler': 0, 'spoiler': 1})
    model.to(device)
    rows = {split: [json.loads(line) for line in (ROOT / f'data/processed/{split}.jsonl').read_text().splitlines()]
            for split in ['train', 'validation', 'test']}
    truncation = {}
    encoded = {}
    for split, records in rows.items():
        full = tokenizer([r['text'] for r in records], truncation=False, verbose=False)
        truncation[split] = sum(len(ids) > MAX_LENGTH for ids in full['input_ids'])
        encoded[split] = tokenizer([r['text'] for r in records], truncation=True,
                                   max_length=MAX_LENGTH, padding='max_length', return_tensors='pt')

    def batches(split, shuffle=False):
        return DataLoader(list(range(len(rows[split]))), batch_size=16, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(SEED) if shuffle else None)

    def predict(split):
        model.eval()
        scores = []
        with torch.inference_mode():
            for indices in batches(split):
                inputs = {k: v[indices].to(device) for k, v in encoded[split].items()}
                scores.extend(model(**inputs).logits.softmax(-1)[:, 1].cpu().tolist())
        return scores

    def examples():
        model.eval()
        inputs = tokenizer([t for t, _ in EXAMPLES], padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors='pt').to(device)
        with torch.inference_mode():
            return model(**inputs).logits.softmax(-1)[:, 1].cpu().tolist()

    labels = [r['label'] for r in rows['test']]
    before = predict('test')
    before_examples = examples()
    (ROOT / 'reports/before.json').write_text(json.dumps({'metrics': metrics(labels, before),
        'scores': before, 'example_scores': before_examples}, indent=2) + '\n')
    print('BEFORE', metrics(labels, before), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = 2 * len(batches('train'))
    warmup = max(1, int(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        step / warmup if step < warmup else max(0, (total_steps - step) / (total_steps - warmup)))
    best_f1 = -1
    history = []
    for epoch in range(1, 3):
        model.train()
        total_loss = 0
        for step, indices in enumerate(batches('train', shuffle=True), 1):
            inputs = {k: v[indices].to(device) for k, v in encoded['train'].items()}
            target = torch.tensor([rows['train'][i]['label'] for i in indices.tolist()], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=target).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % 25 == 0:
                print(f'Epoch {epoch}/2 step {step}/{len(batches("train"))} loss={total_loss/step:.4f} elapsed={time.time()-start:.0f}s', flush=True)
        validation = metrics([r['label'] for r in rows['validation']], predict('validation'))
        history.append({'epoch': epoch, 'mean_training_loss': total_loss / step, 'validation': validation})
        print('VALIDATION', history[-1], flush=True)
        if validation['f1'] > best_f1:
            best_f1 = validation['f1']
            model.save_pretrained(ROOT / 'models/minilm-spoiler')
            tokenizer.save_pretrained(ROOT / 'models/minilm-spoiler')
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'models/minilm-spoiler', local_files_only=True).to(device)
    after = predict('test')
    after_examples = examples()
    result = {'model': 'microsoft/MiniLM-L12-H384-uncased',
              'revision': json.loads((ROOT / 'models/base/source.json').read_text())['sha'],
              'seed': SEED, 'device': str(device), 'max_length': MAX_LENGTH,
              'batch_size': 16, 'epochs': 2, 'learning_rate': 2e-5,
              'truncated_review_counts': truncation, 'elapsed_seconds': time.time() - start,
              'before': metrics(labels, before), 'after': metrics(labels, after),
              'always_no_spoiler': metrics(labels, [0.] * len(labels)), 'history': history,
              'examples': [{'text': text, 'expected_label': label, 'before': b, 'after': a}
                           for (text, label), b, a in zip(EXAMPLES, before_examples, after_examples)]}
    (ROOT / 'reports/results.json').write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'reports/test-predictions.jsonl').open('w') as stream:
        for row, b, a in zip(rows['test'], before, after):
            stream.write(json.dumps({'source_line': row['source_line'], 'movie_id': row['movie_id'],
                                     'label': row['label'], 'before': b, 'after': a}) + '\n')
    from render_results import render
    render(result)
    print('AFTER', result['after'], flush=True)
    print('Report: reports/FINETUNING_RESULTS.md', flush=True)


if __name__ == '__main__':
    main()
