"""Prepare blinded review materials without exposing assistant labels or model scores."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/development-benchmark'
OUT = ROOT / 'reports/development-benchmark'


def main():
    rows = [json.loads(line) for line in (DATA/'examples.jsonl').read_text().splitlines()]
    random.Random(20261002).shuffle(rows)
    sheet = ['# Human review sheet — synthetic development pack', '',
        f'{len(rows)} assistant-authored fictional examples. All await independent human review. There are no real users, videos, or source-corpus rows here. Decisions and timestamps are not supplied by humans yet.', '',
        'Read ANNOTATION.md first. Assume the viewer has not started the work and knows only its public premise. Use spoiler / safe / uncertain. Record the exact triggering quote (if any), rationale, reviewer ID, and UTC review time. Do not open assistant proposals or model outputs before completing your independent pass. All text below is complete; long reviews intentionally contain repetitive padding.', '',
        'Neutral review IDs and shuffled order reduce label anchoring. Work facts below define the fictional story for annotation; they were not sent to the model. Do not infer a decision from the fact that a case is included. Human responses can be entered in review-response-template.json; leave status pending_human_review until actually reviewed.', '']
    responses, key, proposals = [], [], []
    for i, r in enumerate(rows, 1):
        rid = f'R{i:03d}'
        sheet += [f'## {rid} — {r["surface"].replace("_", " ")}', '',
                  f'**Work:** {r["work_title"]}', '',
                  f'**Public premise:** {r["context"]["public_premise"]}', '',
                  f'**Private story fact for annotators:** {r["context"]["private_story_fact"]}', '']
        if 'timestamp' in r:
            t = r['timestamp']
            sheet += [f'**Synthetic passage time:** {t["start_seconds"]//60:02d}:{t["start_seconds"]%60:02d}–{t["end_seconds"]//60:02d}:{t["end_seconds"]%60:02d}. Adjacent transcript unavailable.', '']
        sheet += ['**Text to label (complete):**', '', r['text'], '',
                  '**Human decision:** ______  **Triggering quote:** ______', '',
                  '**Rationale / missing context:** ______', '',
                  '**Reviewer / UTC timestamp:** ______', '']
        responses.append(dict(review_id=rid, status='pending_human_review', human_label=None,
            triggering_quote=None, rationale=None, reviewer_id=None, reviewed_at=None, adjudicator_id=None))
        key.append(dict(review_id=rid, example_id=r['id']))
        fact = r['context']['private_story_fact']
        pos = r['text'].find(fact)
        evidence = dict(start=pos, end=pos+len(fact), quote=fact) if pos >= 0 else None
        proposals.append(dict(assistant_evidence_span=evidence, review_id=rid, example_id=r['id'], proposed_label=r['proposed_label'],
                              annotation_rationale=r['annotation_rationale']))
    (OUT/'REVIEW_SHEET.md').write_text('\n'.join(sheet)+'\n')
    (OUT/'review-response-template.json').write_text(json.dumps(responses, indent=2)+'\n')
    (OUT/'review-key.json').write_text(json.dumps(key, indent=2)+'\n')
    (OUT/'assistant-proposals.json').write_text(json.dumps(proposals, indent=2)+'\n')
    assert len(responses) == len(rows) and all(r['human_label'] is None for r in responses)
    assert all(word not in '\n'.join(sheet) for word in ('early_reveal', 'proposed_label', '0.215792'))
    print(f'Prepared {len(rows)} blinded review entries, blank response template, and separate proposal/key files.')


if __name__ == '__main__':
    main()
