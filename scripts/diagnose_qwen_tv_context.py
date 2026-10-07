"""Targeted accurate-context diagnostic; cached alone responses, no training."""
import argparse
import copy
import json
import re
from pathlib import Path
import compare_qwen_rag as q

OUT = q.ROOT / 'reports/qwen-tv-verified-context'
STRICT = q.check_answer

def normalized_check(raw, row, arm):
    """Allow whitespace differences only; all content and schema checks stay strict."""
    try:
        answer = json.loads(raw)
        normalized = copy.deepcopy(row)
        normalized['excerpt'] = ' '.join(row['excerpt'].split())
        normalized['passages'] = [dict(p, text=' '.join(p['text'].split())) for p in row['passages']]
        for field in ('excerpt_quote', 'passage_quote'):
            if isinstance(answer.get(field), str):
                answer[field] = ' '.join(answer[field].split())
        checked, valid = STRICT(json.dumps(answer), normalized, arm)
        return (json.loads(raw), True) if valid else (checked, False)
    except (ValueError, TypeError, AttributeError):
        return dict(label='UNCERTAIN', explanation='Invalid format or ungrounded quote'), False

# Passages paraphrased from the cached episode summaries; no review labels sent.
SPECS = {
 '547': ('Nikita', 'ep33', 'In season two, Amanda has Ryan placed in the cell with Percy to help bring down Oversight. Percy remains confined while his Guardians prepare a breakout.', 1, 'Reveals the later confinement and power shift; passage supports confinement.'),
 '395': ('LifeOnMars2006', 'ep16', 'In the series finale, Sam returns to the future during a failed sting operation. He then chooses to jump off a building to return to the past and the people he knows there.', 1, 'Explicit series-ending return and jump.'),
 '652': ('Nikita', 'ep11', 'In All the Way, Birkhoff discovers the shell program, Nikita is captured, and Thom discovers Alex is the mole. Alex accidentally kills Thom, frames him as the mole, and is promoted to field agent.', 1, 'Reveals a death and consequential promotion.'),
 '627': ('Nikita', 'ep34', 'In Sanctuary, Sean tracks Nikita using Alex’s watch and is captured. Nikita and the others persuade him to join their cause against Percy.', 1, 'Later capture is supported; clothing and restraint details are not verified.'),
 '328': ('Homeland', 'ep3', 'In Clean Skin, Lynne is assassinated and her diamond necklace is stolen. The necklace is sold for $400,000. A young couple purchases a house near an airport; the CIA investigates potential recipients of the necklace money in the next episode.', None, 'Specific plot financing inference, but uncertainty versus consequential reveal is ambiguous under this policy.'),
 '565': ('Nikita', 'ep21', 'In Betrayals, Percy reveals that Nikita killed Alex’s father. Alex later confronts Nikita and shoots her.', 0, 'The excerpt names an episode and vaguely praises its ending without revealing an event.'),
 '307': ('Homeland', 'ep12', 'In Marine One, Brody brings a suicide vest into the bunker. Dana’s plea leads him to abandon the bombing. Carrie later undergoes electroconvulsive therapy.', 0, 'Production location fact; supplied plot must not contaminate the decision.'),
 '545': ('Nikita', 'ep33', 'In Pale Fire, Amanda has Ryan placed in Percy’s cell. Percy’s Guardians plan to secure his release.', 0, 'Generic leadership behavior; no named consequential outcome.'),
 '593': ('Nikita', 'ep11', 'In All the Way, Birkhoff discovers the shell program. Alex accidentally kills Thom and is promoted to field agent.', 0, 'A generic statement about writing an exploit does not reveal these events.'),
 '638': ('Nikita', 'ep34', 'In Sanctuary, Sean tracks Nikita, is captured, and is persuaded to join her cause.', 0, 'Being a decorated soldier alone reveals no consequential later outcome.'),
 '568': ('Nikita', 'ep21', 'In Betrayals, Alex learns Nikita killed her father and later shoots Nikita.', 0, 'Vague statement about violence; no event or outcome.'),
 '562': ('Nikita', 'ep16', 'In Echoes, Amanda recommends cancelling Alex. Michael tracks down Nikita’s residence and visits her with a shotgun.', 1, 'Original safe markup conflicts with the revealed episode-ending encounter.'),
 '324': ('Homeland', 'ep12', 'In Marine One, Carrie contacts Dana about Brody’s impending attack. Dana’s plea causes Brody to abandon the suicide bombing. Carrie later receives electroconvulsive therapy.', 1, 'Original safe markup conflicts with a season-finale outcome.'),
 '578': ('Nikita', 'ep31', 'In Fair Trade, Alex poses as a stripper to get herself arrested and deported to St. Petersburg. An immigration officer sells deported women to traffickers; Alex kills the traffickers and helps free the women.', 1, 'Original safe markup conflicts with a specific later covert travel tactic.')
}

def prepare():
    base = q.ROOT / 'reports/qwen-alone-expanded'
    allrows = json.loads((base/'inputs.json').read_text())
    original = json.loads((base/'annotations.json').read_text())
    prior = json.loads((base/'alone.json').read_text())
    episodes = [json.loads(l) for l in (q.ROOT/'data/raw/episode-context/episodes.jsonl').read_text().splitlines()]
    inputs, audit = [], {}
    for suffix, (work, anchor, passage, label, reason) in SPECS.items():
        row = copy.deepcopy(next(r for r in allrows if r['id'] == 'tvtropes:validation:'+suffix))
        ep = next(e for e in episodes if e['work_page'] == work and e['episode_anchor'] == anchor)
        row['passages'] = [dict(id='verified-'+suffix, text=passage, source_url=ep['plot_source_url'])]
        inputs.append(row)
        audit[row['id']] = dict(id=row['id'], label=label, original_label=original[row['id']]['label'], reason=reason, source_url=ep['plot_source_url'], cached_plot_sha256=q.sha(ep['plot']), label_provenance='source-backed AI policy audit, not blind or human gold', human_verified=False)
    manifest = dict(model=q.MODEL, system=q.SYSTEM, rows=len(inputs), inputs_sha256=q.sha(inputs), temperature=0, max_completion_tokens=512, validator='Strict schema and whitespace-normalized exact substring; no content edits', retrieval='Manually selected source-backed paraphrases, not automatic retrieval', annotations_sha256=q.sha(audit), warning='Targeted selection after viewing errors; reused cached alone answers; non-blind AI label audit; development diagnostic, not held-out performance.')
    baseline = dict(config=dict(manifest, arm='alone', reuse_source=str(base/'alone.json')), predictions={})
    tv = []
    for row in allrows:
        if row['slice'] != 'tv_markup_dev': continue
        pred = copy.deepcopy(prior['predictions'][row['id']])
        if not pred.get('api_error'):
            pred['answer'], pred['valid'] = normalized_check(pred['raw'], row, 'alone')
        tv.append(dict(id=row['id'], original_valid=prior['predictions'][row['id']]['valid'], normalized_valid=pred['valid'], original_label=original[row['id']]['label'], prediction=pred['answer']['label']))
        if row['id'] in audit: baseline['predictions'][row['id']] = pred
    for name, obj in [('inputs',inputs),('manifest',manifest),('label-audit',audit),('alone-normalized',baseline),('tv-whitespace-audit',tv)]: q.save(OUT/(name+'.json'), obj)
    return inputs, manifest, audit, baseline, {r['id']:original[r['id']] for r in inputs}

def report(inputs, audit, baseline, original):
    context = json.loads((OUT/'rag.json').read_text())
    q.summarize(inputs, {'alone':baseline, 'context':context}, audit)
    reviewed = json.loads((OUT/'comparison.json').read_text())
    q.summarize(inputs, {'alone':baseline, 'context':context}, original)
    markup = json.loads((OUT/'comparison.json').read_text())
    q.save(OUT/'comparison-reviewed.json', reviewed)
    q.save(OUT/'comparison-original-markup.json', markup)
    q.save(OUT/'comparison.json', reviewed)
    semantic = {}
    for arm, cache in [('alone', baseline), ('context', context)]:
        counts = dict(tp=0, tn=0, fp=0, fn=0, abstentions=0, malformed=0)
        for row in inputs:
            label = audit[row['id']]['label']
            if label is None: continue
            try: raw_label = json.loads(cache['predictions'][row['id']]['raw'])['label']
            except (ValueError, KeyError): counts['malformed'] += 1; continue
            if raw_label == 'UNCERTAIN': counts['abstentions'] += 1
            else: counts['tp' if label == 1 and raw_label == 'SPOILER' else 'fn' if label == 1 else 'fp' if raw_label == 'SPOILER' else 'tn'] += 1
        semantic[arm] = counts
    q.save(OUT/'semantic-decisions.json', dict(warning='Raw classification only; does not certify valid supporting evidence.', arms=semantic))
    rows=[]
    for r in inputs:
        a=baseline['predictions'][r['id']]['answer']['label']; b=context['predictions'].get(r['id'],{}).get('answer',{}).get('label','PENDING')
        rows.append(f"| {r['id'].split(':')[-1]} | {audit[r['id']]['label']} | {a} | {b} |")
    lines=['# Qwen accurate-context TV diagnostic','',manifest_warning(),'','No training or purchases. Same cached alone inputs and frozen classification prompt; context adds manually curated source-backed passages. Whitespace-only normalized quote validation is used for both arms.','', '| Source row | AI policy label (1 spoiler, 0 safe, None unresolved) | Alone | Context |', '|---|---|---|---|', *rows, '', '## Interpretation', '', 'Raw decisions: both arms identify all seven source-audited spoilers. Alone marks five of six safe references SAFE and abstains on the vague ending reference; context marks all six SAFE. The ambiguous financing example changes SAFE to SPOILER but remains unscored. There is no demonstrated spoiler recall gain here.', '', 'Three context responses fail exact evidence validation because U+2019 apostrophes in supplied passages become U+0019 control characters in returned passage quotes. Their raw labels are SPOILER; this is a quotation encoding failure, not a semantic miss. The validator remains strict on content. Accepted evidence-backed recall is consequently 4/7 versus 7/7 for alone. Raw-label agreement must not be presented as evidence-backed reliability.', '', 'Next: fix quote serialization (prefer returned character offsets or passage IDs with locally extracted quotes), then test automatically retrieved context on a larger, independently labeled set. This targeted set cannot establish overall model quality.', '', '## Reviewed-label metrics', '', '```json', json.dumps(reviewed['arms'], indent=2), '```', '', 'Original markup agreement is saved separately. Source-backed AI audit is not human gold; three original safe labels disclose later events, one original spoiler label is a vague ending reference, and one reference remains ambiguous.', '', 'The full 60-row baseline has five additional valid spoiler predictions after whitespace normalization: recall changes from 18/30 to 23/30. This is a scoring correction, not a new inference gain. Eight previously invalid quotes become valid; original caches remain intact.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def manifest_warning():
    return '**Targeted development experiment, selected after inspecting errors. This tests accurate supplied context, not retrieval quality or generalization.**'

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run', action='store_true'); args=parser.parse_args()
    q.OUT=OUT; inputs, manifest, audit, baseline, original=prepare()
    # Verify whitespace tolerance does not accept fabricated evidence.
    r=dict(excerpt='Alex  kills Thom.', passages=[])
    good=json.dumps(dict(label='SPOILER',excerpt_quote='Alex kills Thom.',passage_id='',passage_quote='',explanation='Death'))
    assert normalized_check(good,r,'alone')[1]
    assert not normalized_check(good.replace('kills','saves'),r,'alone')[1]
    q.check_answer=normalized_check
    print('Prepared',len(inputs),'targeted paired examples; validator checks passed.',flush=True)
    if args.run: q.run(inputs, manifest, 20, arms=('rag',), annotations=audit)
    if (OUT/'rag.json').exists(): report(inputs,audit,baseline,original)
