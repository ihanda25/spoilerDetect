"""Build fresh blinded dev review inputs; never overwrite decisions or expose test."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/eval-refresh'
POLICY = 'unstarted-viewer-public-premise-v1'


def main():
    source = ROOT / 'reports/real-review-pack/candidates.jsonl'
    annotations = ROOT / 'reports/real-review-pack/annotated-human-and-ai.jsonl'
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    dev = [r for r in rows if r['split'] == 'dev']
    # Used only to select a re-review batch; no previous judgments enter review inputs.
    prior = {r['id']: r for r in map(json.loads, annotations.read_text().splitlines())
             if r['split'] == 'dev'}
    priority_ids = {r['id'] for r in dev if prior[r['id']]['label_provenance'] == 'human'
                    or prior[r['id']]['label'] in (None, 1)}
    for r in dev:
        r.update(label=None, label_provenance='human', source_label=None,
                 review_status='pending_human_review', reviewer_id=None,
                 reviewed_at=None, review_rationale=None,
                 group_review_status='pending_human_verification',
                 eligible_for_evaluation=False, annotation_policy=POLICY)
    random.Random(20261005).shuffle(dev)
    batch = [r for r in dev if r['id'] in priority_ids]
    outputs = {'dev-review.jsonl': dev, 'first-batch.jsonl': batch}
    manifest = {
        'policy': POLICY, 'purpose': 'dev label refresh, not a new independent test',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'dev_count': len(dev), 'first_batch_count': len(batch),
        'test_count_preserved': sum(r['split'] == 'test' for r in rows),
        'first_batch_selection': 'previous human decisions plus AI positive/uncertain dev rows; biased review queue, not a scored subset',
        'rows': [{'id': r['id'], 'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest(),
                  'group_id': r['group_id']} for r in dev],
    }
    serialized = {name: ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset)
                  for name, subset in outputs.items()}
    serialized['manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    # Refuse changed existing artifacts, so reruns cannot erase reviewed work.
    for name, body in serialized.items():
        path = OUT / name
        if path.exists() and path.read_text() != body:
            raise ValueError(f'Refusing to overwrite changed artifact: {path}')
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in serialized.items():
        (OUT / name).write_text(body)
    print(f'Prepared {len(dev)} blinded dev rows; first batch {len(batch)}. Test untouched.')


if __name__ == '__main__':
    main()
