"""Tokenize every eligible review without truncation; preserve pilot movie splits."""
import json
import os
from collections import Counter
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import numpy as np
from transformers import AutoTokenizer
from prepare_training import records, digest, split_for, ROOT

DEST = ROOT / 'data/full'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    counts = Counter(digest(r['review_text']) for r in records())
    streams = {s: (DEST / f'{s}.tokens.bin').open('wb') for s in ['train', 'validation', 'test']}
    metadata = {s: {'offsets': [0], 'labels': [], 'source_lines': [], 'movie_ids': []} for s in streams}
    pending = []
    excluded = 0

    def flush():
        texts = tokenizer([r['review_text'] for _, r in pending], add_special_tokens=False,
                          truncation=False, verbose=False)['input_ids']
        for (line, row), ids in zip(pending, texts):
            split = split_for(row['movie_id'])
            values = metadata[split]
            np.asarray(ids, dtype=np.int32).tofile(streams[split])
            values['offsets'].append(values['offsets'][-1] + len(ids))
            values['labels'].append(int(row['is_spoiler']))
            values['source_lines'].append(line)
            values['movie_ids'].append(row['movie_id'])
        pending.clear()

    for line, row in enumerate(records(), 1):
        if not row['review_text'].strip() or counts[digest(row['review_text'])] != 1:
            excluded += 1
            continue
        pending.append((line, row))
        if len(pending) == 256:
            flush()
        if line % 50000 == 0:
            print(f'Tokenized {line} source reviews', flush=True)
    if pending:
        flush()
    movie_sets = {s: set(v['movie_ids']) for s, v in metadata.items()}
    assert all(movie_sets[a].isdisjoint(movie_sets[b]) for a,b in
               [('train','validation'),('train','test'),('validation','test')])
    report = {'excluded_rows': excluded, 'max_window_tokens': 512, 'overlap_tokens': 64,
              'split_method': 'Same SHA256 movie split as pilot; no sampling; no truncation', 'splits': {}}
    for split, values in metadata.items():
        streams[split].close()
        for key, array in values.items():
            np.save(DEST / f'{split}.{key}.npy', np.asarray(array))
        lengths = np.diff(values['offsets']) + 2
        report['splits'][split] = {
            'reviews': len(lengths), 'movies': len(movie_sets[split]),
            'spoilers': sum(values['labels']), 'over_192': int((lengths > 192).sum()),
            'over_512': int((lengths > 512).sum()),
            'token_length_percentiles': {str(p): float(np.percentile(lengths, p)) for p in [50,90,95,99,100]},
            'tokens': values['offsets'][-1]}
    (ROOT / 'reports/full-data.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
