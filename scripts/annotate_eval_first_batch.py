"""Persist individually reviewed AI judgments, separately from human decisions."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/eval-refresh'
# Decisions written after reading every displayed excerpt, without model predictions.
# Each tuple: label, rationale, exact revealing quote, relevance, missing evidence.
DECISIONS = {
    '8d6f3e107020fae5': (1, 'Discloses Arthur remaining in the hotel dream level during a specific later water kick, beyond the general dream-heist premise.', "Arthur's level at that point is the hotel level; he never goes any deeper", 'story', None),
    'c451c3925da2d618': (1, 'Explains the characters’ coordinated chain of kicks to return through dream levels; this describes a concrete plot strategy beyond the general premise.', 'they need to set up a chain of kicks', 'story', None),
    'd983fd06cdb41420': (0, 'The humanity-ending language is a joke about sitcom quality, not a literal disclosed story outcome.', None, 'opinion', None),
    '0d0cce825a8dd445': (0, 'Criticizes the competition format and use of an actor’s generated voice; no contestant outcome or plot revelation is stated.', None, 'production', None),
    'e4bebccb4736c2dc': (0, 'Reports representation statistics in publishing, not an event within a story.', None, 'off_topic', None),
    '181eeee6a40b0df4': (None, 'The garden/reality interpretation could reveal a scene or be philosophical commentary; the truncated sentence does not establish which.', None, 'story', 'Identify the garden reference and whether it discloses a later scene or ending interpretation.'),
    'fc094a4baa457d2f': (0, 'Describes the subjects, period and comic style; it gives no later event or outcome.', None, 'premise_or_style', None),
    'e75078198946bfd3': (0, 'Describes competing climbers and documentary themes without saying who succeeds, fails or dies.', None, 'premise_or_style', None),
    '241424661d510ace': (1, 'Explicitly reveals an offer to pay in mid-season one, a specific event beyond the starting premise. Later explanation is only hinted at.', 'the rich couple offered to pay', 'story', None),
    '48974228f824d086': (0, 'Evaluates a publicly identified judge; no competition result or elimination is disclosed.', None, 'cast_or_opinion', None),
    '26505bdaa97b6ae0': (0, 'Discusses viewing schedules and entertainment availability, not a Game of Thrones plot event.', None, 'off_topic', None),
    'f1b5a4fafd3c61b9': (0, 'Generic requests to be still and a vague statement about finishing do not identify an event or outcome.', None, 'dialogue', None),
    '4b875b55e0ff53c1': (0, 'A books-roundup headline contains no story details.', None, 'off_topic', None),
    'f74fd1beed395944': (0, 'Evaluates game design and identifies a sequel; it does not reveal a narrative event.', None, 'opinion', None),
    '97f24e9ce5396025': (0, 'Discusses open-world map design rather than the film’s plot.', None, 'off_topic', None),
    '89aa80ddc9cfd17b': (0, 'Warns an unnamed traveler about danger but identifies no particular attack, twist or character fate.', None, 'dialogue', None),
    '9413f71394ae8856': (0, 'Introduces the protagonist, journey and motivation in a broad setup description; no later resolution is given.', None, 'premise_or_style', None),
    '6ec4ebe7b53e35dd': (0, 'Discusses HTML/CSS applications and cross-device software, not the show’s story.', None, 'off_topic', None),
    'd1ec33947010af34': (0, 'Reports an awards-attendance arrangement rather than a fictional or competition outcome.', None, 'off_topic', None),
    '9bdc0c0b503c5989': (0, 'A good-night greeting names Scales but discloses no event, identity twist or fate.', None, 'dialogue', None),
    '56931649e3580dc8': (0, 'A generic command names Scales but does not identify the target or disclose what happens.', None, 'dialogue', None),
    '15176b64fbe657e7': (0, 'Discusses a possible trilogy and franchise repetition without revealing events in it.', None, 'production', None),
    '9043ebbfe8b9e757': (1, 'Explicitly locates Walt meeting Gustavo Fring in season two; a later character encounter is revealed even though no major twist is stated.', 'In the 11th episode of season 2, Walt meets Gustavo Fring', 'story', None),
    'b2194d26d6bea8bf': (None, 'The couple’s breakup is an explicit event, but this short review does not establish whether it is the public setup or a later revelation.', None, 'story', 'Verify the work’s public premise and whether the breakup is a later event.'),
    'f686ff442c178524': (None, 'The blade’s history could reveal meaningful backstory or be generic ominous dialogue; the excerpt gives no owner or incident.', None, 'story', 'Identify the blade and whether its violent history is a story revelation.'),
    '2bbbc47562c917df': (0, 'Reports financial performance rather than an event in the movie.', None, 'off_topic', None),
    'b8e7b02dff904091': (0, 'A generic search statement does not identify the missing person or reveal an outcome.', None, 'dialogue', None),
    '897ba3563e6c4c23': (0, 'Describes setting, period and tone; no specific character event or resolution is supplied.', None, 'premise_or_style', None),
    '4b72935273d2df09': (0, 'Describes a concert performance and visual mood without a story or performance outcome.', None, 'opinion', None),
    '4672fba6a2294d90': (0, 'Describes a protagonist, prose format and literary quality; no later event is stated.', None, 'premise_or_style', None),
    'a118a81bb9e1c201': (0, 'A question about a location does not answer the traveler’s purpose or disclose an event.', None, 'dialogue', None),
    '501b0421ca5055b1': (None, 'The location and proximity revelation may advance the dragon search, but the excerpt does not establish what was already public or known.', None, 'story', 'Verify the scene context and public premise for the dragon lands/proximity revelation.'),
    '3b138ba78eefae17': (0, 'Lists adaptation credits, cast and comic quality; no story event is present in the displayed text.', None, 'cast_or_opinion', None),
    'bf4a5e2bad17538b': (None, 'Identifying the search target as a dragon may be public setup or a reveal; surrounding dialogue and public premise are needed.', None, 'story', 'Verify whether searching for a dragon is public premise or a later identity revelation.'),
    '306c34936af4afce': (0, 'Reports an author’s intention about viewing an adaptation, not the adaptation’s plot.', None, 'production', None),
}


def main():
    rows = list(map(json.loads, (OUT / 'first-batch.jsonl').read_text().splitlines()))
    assert {r['id'].removeprefix('real-') for r in rows} == set(DECISIONS)
    timestamp = datetime.now(timezone.utc).isoformat()
    for r in rows:
        label, rationale, quote, relevance, missing = DECISIONS[r['id'].removeprefix('real-')]
        assert quote is None or quote in r['text']
        r.update(label=label, label_provenance='ai', review_status='ai_reviewed',
                 reviewer_id='codex-assistant', reviewed_at=timestamp,
                 review_rationale=rationale, revealing_quote=quote,
                 relevance=relevance, missing_context=missing,
                 annotation_method='individual excerpt review under refreshed policy; no new plot-source verification',
                 group_review_status='pending_human_verification', eligible_for_evaluation=False)
    output = OUT / 'first-batch-ai-decisions.jsonl'
    if output.exists():
        raise ValueError('Existing decisions preserved; use a versioned output for changes.')
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps({'reviewed':len(rows), 'labels':dict(Counter('uncertain' if r['label'] is None else 'spoiler' if r['label'] else 'safe' for r in rows)), 'provenance':'ai'}))


if __name__ == '__main__':
    main()
