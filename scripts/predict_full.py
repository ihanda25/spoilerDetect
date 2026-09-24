"""Offline inference using the full-data model and all overlapping windows."""
import argparse
import json
import numpy as np
import torch
from full_training import OUT, Reviews, collate, review_logits
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    args = parser.parse_args()
    if not args.text.strip():
        parser.error('Text must not be empty.')
    path = OUT/'best'
    settings_path = path/'inference_settings.json'
    if not settings_path.exists():
        parser.error('Full training and final evaluation have not completed yet.')
    settings = json.loads(settings_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
    ids = tokenizer(args.text, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    dataset = Reviews.__new__(Reviews)
    dataset.tokens = np.asarray(ids, dtype=np.int32)
    dataset.offsets = np.array([0,len(ids)])
    inputs, sizes = collate(dataset, [0])
    with torch.inference_mode():
        score = review_logits(model, inputs, sizes, 'cpu').sigmoid().item()
    print(json.dumps({'label':'spoiler' if score >= settings['threshold'] else 'no_spoiler',
        'spoiler_score':score,'threshold':settings['threshold'], 'calibrated':False,
        'windows':sizes[0], 'truncated':False}, indent=2))
