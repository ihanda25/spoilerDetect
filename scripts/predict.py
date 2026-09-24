"""Offline local inference using the fine-tuned checkpoint."""
import argparse
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    args = parser.parse_args()
    if not args.text.strip():
        parser.error('Text must not be empty.')
    path = ROOT / 'models/minilm-spoiler'
    if not path.exists():
        parser.error('Fine-tuned model does not exist yet. Run scripts/finetune.py first.')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
    encoded = tokenizer(args.text, truncation=True, max_length=192, return_tensors='pt')
    with torch.inference_mode():
        score = model(**encoded).logits.softmax(-1)[0, 1].item()
    print(json.dumps({'label': 'spoiler' if score >= .5 else 'no_spoiler',
                      'spoiler_score': score, 'calibrated': False,
                      'truncated': len(tokenizer(args.text)['input_ids']) > 192}, indent=2))
