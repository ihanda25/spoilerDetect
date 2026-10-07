"""Deterministic assistant-authored development challenges; never reads corpus splits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/development-benchmark'
WORKS = [
    dict(id='glass_harbor', title='Glass Harbor', premise='A radio engineer investigates a silent coastal beacon.', reveal='Mara discovers that her father deliberately sank the rescue ship.', theory='The mayor might have sabotaged the beacon.', opinion='The foggy photography and spare music create a patient, uneasy mood.', question='Did her father know more than he admitted?', asr='mara finds out her dad sank the rescue boat on purpose'),
    dict(id='last_orchard', title='The Last Orchard', premise='A botanist travels to a remote seed vault after a crop failure.', reveal='The vault contains no seeds; Ivo burned them before the expedition arrived.', theory='Perhaps the vault keeper is hiding a second orchard.', opinion='The practical sets are lovely, although the middle section feels repetitive.', question='Was Ivo lying about what was inside?', asr='there arent any seeds ivo burned the lot before they got there'),
    dict(id='paper_moon', title='Paper Moon Station', premise='A courier joins the crew of an isolated lunar station.', reveal='At the end, the courier is revealed to be the station commander from the future.', theory='I think the courier could be a spy from Earth.', opinion='The cast sells the awkward humor, but the dialogue occasionally sounds stiff.', question='Why do those two have the same scar?', asr='turns out the courier is the commander from the future at the end'),
    dict(id='winter_score', title='The Winter Score', premise='A violinist enters a competition in a snowbound city.', reveal='Nell wins the final by playing the score stolen by her mentor, who then confesses.', theory='Maybe the rival has a secret deal with the judges.', opinion='The performances sound rich and the rehearsal scenes have convincing energy.', question='So the mentor wrote it, right?', asr='nell wins playing the stolen score and then her mentor admits it'),
]


def build():
    rows = []
    for w in WORKS:
        def add(surface, text, label, tag, rationale, **extra):
            rows.append(dict(id=f"{w['id']}-{surface}-{tag}", surface=surface, text=text,
                work_id=w['id'], work_title=w['title'], context=dict(public_premise=w['premise'],
                private_story_fact=w['reveal'], viewer_progress='Before starting; public premise known'),
                provenance=dict(kind='assistant_authored_synthetic', source_split='synthetic_development',
                                author='Codex assistant', created_date='2026-10-02'),
                proposed_label=label, annotation_rationale=rationale, challenge=tag,
                annotation=dict(status='pending_human_review', human_label=None, reviewer_id=None,
                                reviewed_at=None, rationale=None, adjudicator_id=None), **extra))
        fact_reason='States a consequential later event absent from the public premise.'
        premise_reason='Only restates the stipulated public premise; no later event is disclosed.'
        opinion_reason='Judges craft or experience without disclosing a story outcome.'
        ambiguous_reason='An indirect question may imply a reveal; needs a policy judgment about insinuation.'
        add('headline', f"{w['title']} ending explained: {w['reveal']}", 1, 'explicit_reveal', fact_reason)
        add('headline', f"First look at {w['title']}: {w['premise']}", 0, 'public_premise', premise_reason)
        add('headline', f"{w['title']} review: gorgeous sound, uneven pacing", 0, 'craft_only', opinion_reason)
        add('headline', f"{w['title']}: {w['question']}", None, 'insinuation', ambiguous_reason)
        add('comment', f"okay but {w['reveal']} i did NOT see that coming lol", 1, 'slang_reveal', fact_reason)
        add('comment', f"Haven't watched yet. My guess: {w['theory']}", 0, 'explicit_speculation', 'Explicit pre-viewing speculation, not an asserted event; theory is not the stipulated reveal.')
        add('comment', f"Spoiler-free thoughts: {w['opinion']}", 0, 'spoiler_keyword_negative', opinion_reason)
        add('comment', f"Yeah, sure... {w['question']}", None, 'sarcastic_reply', 'Sarcastic reply lacks its parent comment; polarity and implied disclosure are unresolved.')
        filler = ('The lighting gives the rooms a soft texture. The soundtrack leaves plenty of quiet space. '
                  'Some shots linger longer than I prefer. The costumes look carefully chosen. ')*32
        intro = f"My review of {w['title']}. {w['opinion']} "
        pair = w['id'] + '-review-placement'
        add('review', intro + w['reveal'] + ' ' + filler, 1, 'early_reveal', fact_reason + ' Reveal precedes repetitive padding.', diagnostic_pair_id=pair)
        add('review', intro + filler + w['reveal'], 1, 'late_reveal', fact_reason + ' Identical reveal follows deliberately long synthetic padding.', diagnostic_pair_id=pair)
        add('review', intro + filler, 0, 'long_craft_only', opinion_reason + ' Deliberately repetitive length control; not a realistic sampled review.')
        add('review', f"{w['title']} follows this setup: {w['premise']} {w['opinion']} I would watch it again.", 0, 'short_premise_review', premise_reason + ' Remaining statements are opinions.')
        add('youtube_transcript', f"all right ending discussion {w['asr']}", 1, 'asr_reveal', fact_reason + ' Lowercase, missing punctuation simulates ASR, not real ASR output.', timestamp=dict(start_seconds=600, end_seconds=618), video_id=w['id']+'-discussion')
        add('youtube_transcript', f"welcome back today the setup is simple {w['premise'].lower()}", 0, 'spoken_premise', premise_reason, timestamp=dict(start_seconds=12, end_seconds=26), video_id=w['id']+'-discussion')
        add('youtube_transcript', f"no spoilers in this section just the music {w['opinion'].lower()}", 0, 'spoken_craft', opinion_reason, timestamp=dict(start_seconds=120, end_seconds=137), video_id=w['id']+'-discussion')
        add('youtube_transcript', 'and then she does it and that changes everything you know what i mean', None, 'missing_antecedent', 'Passage omits referents and event; adjacent transcript context is unavailable.', timestamp=dict(start_seconds=420, end_seconds=430), video_id=w['id']+'-discussion')
    validate(rows)
    return rows


def validate(rows):
    assert rows and len({r['id'] for r in rows}) == len(rows)
    assert set(r['surface'] for r in rows) == {'headline', 'comment', 'review', 'youtube_transcript'}
    for r in rows:
        assert r['provenance']['source_split'] == 'synthetic_development'
        assert r['annotation']['status'] in ('pending_human_review', 'human_reviewed', 'adjudicated')
        assert r['proposed_label'] in (None, 0, 1) and r['text'].strip()
        if r['surface'] == 'youtube_transcript':
            assert 0 <= r['timestamp']['start_seconds'] < r['timestamp']['end_seconds']


if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    rows = build()
    (DATA / 'examples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    print(f'Wrote {len(rows)} synthetic development examples; all pending human review.')
