"""Validate/version AI development diagnostics. Never changes human-gold gates."""
import collections
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/eval-refresh'


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text):
    return ' '.join(text.casefold().split())


def validate(rows, original):
    ids, texts = set(), set()
    for r in rows:
        if r['id'] in ids or normalized(r['text']) in texts:
            raise ValueError('Duplicate ID/text')
        ids.add(r['id']); texts.add(normalized(r['text']))
        if r['split'] != 'dev' or r['label_provenance'] != 'ai' or r['review_status'] != 'ai_reviewed':
            raise ValueError('Only explicitly AI-reviewed dev allowed')
        if r.get('eligible_for_evaluation') is not False:
            raise ValueError('Do not bypass human-gold eligibility')
        if r['label'] is not None and (type(r['label']) is not int or r['label'] not in (0, 1)):
            raise ValueError('Invalid label')
        if not r.get('reviewer_id') or not r.get('review_rationale') or not r.get('reviewed_at'):
            raise ValueError('Missing annotation evidence')
        if datetime.fromisoformat(r['reviewed_at'].replace('Z', '+00:00')).utcoffset() is None:
            raise ValueError('Review timestamp requires time zone')
        if r['label'] == 1 and (not r.get('revealing_quote') or r['revealing_quote'] not in r['text']):
            raise ValueError('Positive requires exact revealing quote')
        if r['label'] is None and not r.get('missing_context'):
            raise ValueError('Uncertain requires context gap')
        if r['id'] in original:
            old = original[r['id']]
            if old['split'] != 'dev' or r['text'] != old['text'] or r['group_id'] != old['group_id']:
                raise ValueError('Original split/text/group changed')
    return ids


def counts(rows):
    return dict(total=len(rows), labels=dict(collections.Counter(
        'uncertain' if r['label'] is None else 'spoiler' if r['label'] else 'safe' for r in rows)),
        surfaces=dict(collections.Counter(r['surface'] for r in rows)),
        groups=len({r['group_id'] for r in rows}))


def write_new(path, text):
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Preserve existing export; version changes instead: {path}')
    path.write_text(text)


def main():
    candidates_path = ROOT / 'reports/real-review-pack/candidates.jsonl'
    original_rows = read_rows(candidates_path)
    original = {r['id']: r for r in original_rows}
    rows = read_rows(OUT / 'first-batch-ai-decisions.jsonl') + read_rows(OUT / 'remaining-dev-ai-decisions.jsonl')
    adjudications = json.loads((OUT / 'context-adjudications.json').read_text())
    for r in rows:
        r['slice'] = 'real_content_dev'
        if r['id'] in adjudications:
            r['prior_ai_decision'] = {k: r.get(k) for k in ('label', 'review_rationale', 'revealing_quote', 'missing_context', 'reviewer_id', 'reviewed_at')}
            r.update(adjudications[r['id']])
            r['reviewer_id'] = 'codex-assistant'
            r['annotation_method'] = 'AI adjudication with separately sourced premise/scene evidence'
    dev_ids = {r['id'] for r in original_rows if r['split'] == 'dev'}
    if validate(rows, original) != dev_ids:
        raise ValueError('Require all original dev rows exactly once')
    challenge = read_rows(OUT / 'source-challenge.jsonl')
    validate(rows + challenge, original)
    verification = json.loads((OUT / 'source-verification.json').read_text())
    verified = {r['id']: r for r in verification['rows']}
    # Read only held-out grouping/source metadata to guard additions; no held-out labels used.
    test_groups = {r['group_id'] for r in original_rows if r['split'] == 'test'}
    test_keys = {k for r in original_rows if r['split'] == 'test' for k in r.get('work_keys', [])}
    test_sources = {r['source_group'] for r in original_rows if r['split'] == 'test'}
    word_budgets = collections.Counter()
    for r in challenge:
        proof = verified.get(r['id'], {})
        if not proof.get('verified') or proof.get('text_sha256') != sha(r['text']) or proof.get('source_url') != r['source_url']:
            raise ValueError('Challenge text requires matching source verification')
        if r['id'] in original or r['group_id'] in test_groups or r.get('source_url') in test_sources:
            raise ValueError('Challenge collides with original/held-out source')
        if set(r.get('work_keys', [])) & test_keys:
            raise ValueError('Held-out work overlap')
        if not r.get('work_keys') or not r.get('source_url') or not r.get('source_retrieved_at'):
            raise ValueError('Missing challenge work/source provenance')
        if r.get('surface') != 'plot_excerpt' or r.get('slice') != 'source_plot_challenge':
            raise ValueError('Do not disguise plot excerpts as product surfaces')
        r['quote_verification'] = proof['method']
        word_budgets[r['source_url']] += len(r['text'].split())
    if any(n > 25 for n in word_budgets.values()):
        raise ValueError('Copyrighted challenge quote budget exceeded')
    combined = rows + challenge
    baseline = json.loads((ROOT / 'reports/three-model-comparison/results.json').read_text())['models']['roberta']['config']
    summary = dict(status='AI development diagnostic prepared; not scored',
        annotation_policy='unstarted-viewer-public-premise-v1',
        slices={'real_content_dev': counts(rows), 'source_plot_challenge': counts(challenge)},
        total=counts(combined), source_file_sha256=sha(candidates_path.read_text()),
        baseline={**baseline, 'input':'text_only', 'max_length':512, 'strategy':'truncate'},
        held_out_rows_preserved=sum(r['split']=='test' for r in original_rows),
        source_verification_file='source-verification.json',
        quote_words_by_source=dict(word_budgets),
        limitations=['AI judgments, not human gold', 'Challenge is curator-selected source plot fragments, not natural product traffic',
                     'Short popular-work excerpts; no full-review/ASR generalization', 'Work alias verification remains incomplete',
                     'Minor plot events and major twists must be reported separately'])
    write_new(OUT / 'reviewed-dev-v2.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in combined))
    write_new(OUT / 'summary-v2.json', json.dumps(summary, indent=2) + '\n')
    # Freeze ID order/text for model inputs independently of all annotations/evidence.
    write_new(OUT / 'blind-inputs-v2.jsonl', ''.join(json.dumps({k:r[k] for k in ('id','text','surface','split')}, ensure_ascii=False) + '\n' for r in combined))
    write_new(OUT / 'input-metadata-v2.json', json.dumps({r['id']:{k:r[k] for k in ('work_title','group_id','slice')} for r in combined}, indent=2) + '\n')
    print(json.dumps(summary['total']))


if __name__ == '__main__':
    main()
