"""Deterministic movie-disjoint reservoir samples, excluding all repeated texts."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SIZES = {'train': 8000, 'validation': 1000, 'test': 1000}


def digest(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()


def records():
    with ZipFile(ROOT / 'data/raw/imdb/dataset.zip') as archive:
        with archive.open('IMDB_reviews.json') as stream:
            for line in stream:
                yield json.loads(line)


def split_for(movie):
    bucket = int(hashlib.sha256(('42:' + movie).encode()).hexdigest(), 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


if __name__ == '__main__':
    counts = Counter(digest(r['review_text']) for r in records())
    samples = {s: [] for s in SIZES}
    rngs = {s: random.Random(42 + i) for i, s in enumerate(SIZES)}
    eligible = Counter()
    excluded = 0
    for line_number, row in enumerate(records(), 1):
        text = row['review_text'].strip()
        if not text or counts[digest(text)] != 1:
            excluded += 1
            continue
        split = split_for(row['movie_id'])
        eligible[split] += 1
        record = {'source_line': line_number, 'movie_id': row['movie_id'],
                  'text': text, 'label': int(row['is_spoiler'])}
        if len(samples[split]) < SIZES[split]:
            samples[split].append(record)
        else:
            index = rngs[split].randrange(eligible[split])
            if index < SIZES[split]:
                samples[split][index] = record
    output = ROOT / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    movie_sets = {s: {r['movie_id'] for r in rows} for s, rows in samples.items()}
    text_sets = {s: {digest(r['text']) for r in rows} for s, rows in samples.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert movie_sets[a].isdisjoint(movie_sets[b])
        assert text_sets[a].isdisjoint(text_sets[b])
    manifest = {'seed': 42, 'excluded_duplicate_or_empty_rows': excluded,
                'split_method': 'SHA256(42:movie_id) modulo 100, 80/10/10; reservoir sample per split',
                'splits': {}}
    for split, rows in samples.items():
        assert len(rows) == SIZES[split]
        assert {r['label'] for r in rows} == {0, 1}
        path = output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest['splits'][split] = {'rows': len(rows), 'movies': len(movie_sets[split]),
                                    'spoilers': sum(r['label'] for r in rows),
                                    'eligible_rows': eligible[split],
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT / 'reports/training-data.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
