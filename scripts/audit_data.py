"""Validate and summarize the collected data without printing plot spoilers."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def audit():
    archive = ROOT / 'data/raw/imdb/dataset.zip'
    labels = Counter()
    seen = {}
    review_movies = set()
    duplicates = conflicts = empty = missing = rows = total_chars = 0
    with ZipFile(archive) as z:
        movies = {}
        with z.open('IMDB_movie_details.json') as stream:
            for line in stream:
                row = json.loads(line)
                movies[row['movie_id']] = row
        with z.open('IMDB_reviews.json') as stream:
            for line in stream:
                row = json.loads(line)
                rows += 1
                missing += int(not all(k in row for k in ('review_text', 'is_spoiler', 'movie_id')))
                text, label = row.get('review_text'), row.get('is_spoiler')
                if not isinstance(text, str) or type(label) is not bool:
                    raise ValueError(f'Invalid text/label at review row {rows}')
                labels['spoiler' if label else 'no_spoiler'] += 1
                empty += int(not text.strip())
                total_chars += len(text)
                review_movies.add(row['movie_id'])
                # Whitespace/case normalization for a conservative duplicate audit.
                digest = hashlib.sha256(' '.join(text.lower().split()).encode()).digest()
                if digest in seen:
                    duplicates += 1
                    conflicts += int(seen[digest] != label)
                else:
                    seen[digest] = label
        files = [{'name': i.filename, 'bytes': i.file_size} for i in z.infolist()]
    source = json.loads((archive.parent / 'source-metadata.json').read_text())
    with archive.open('rb') as f:
        checksum = hashlib.file_digest(f, 'sha256').hexdigest()
    report = {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset',
        'author': 'Rishabh Misra', 'dataset_version': 1,
        'license_as_listed_by_source': source['licenseName'],
        'archive_bytes': archive.stat().st_size, 'archive_sha256': checksum,
        'files': files, 'reviews': rows, 'labels': dict(labels),
        'spoiler_fraction': labels['spoiler'] / rows,
        'movie_metadata_records': len(movies), 'unique_review_movie_ids': len(review_movies),
        'movie_ids_without_metadata': len(review_movies - movies.keys()),
        'empty_reviews': empty, 'rows_missing_required_fields': missing,
        'duplicate_normalized_text_rows': duplicates,
        'duplicate_rows_disagreeing_with_first_label': conflicts,
        'mean_review_characters': total_chars / rows,
        'notes': [
            'Labels describe whole reviews, not individual sentences or review titles.',
            'Raw data is preserved; no cleaning, sampling, splitting, or training performed.',
            'Group by movie_id and address duplicates before creating evaluation splits.',
            'Plot summaries and synopses are metadata, not labeled negative examples.'
        ]
    }
    destination = ROOT / 'reports/data-audit.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit()
