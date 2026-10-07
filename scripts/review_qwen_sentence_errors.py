import json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'reports/qwen-alone-sentences-100'
errors=json.loads((out/'errors.json').read_text())
# Apply the frozen consequential-reveal policy to the displayed excerpt, not its lost parent paragraph.
reviews={
'2913':(0,'Dialogue about client preferences, not a consequential outcome or hidden identity.'),
'7247':(1,'Explicitly discloses the failure of an attempted intervention against a serial rapist. Episodic outcomes count; a spoiler need not affect the entire series.'),
'3445':(1,'The supplied public premise does not disclose parallel universes; the sentence reveals a later structural development.'),
'8425':(0,'Truncated ordinary dialogue and a facial expression; no identifiable consequential reveal.'),
'8573':(0,'A character symptom contrasted with thinking about military service; no named later event, identity, or fate is revealed in this excerpt.'),
'8520':(0,'General personality/interest in mysteries, consistent with the premise.'),
'8551':(0,'An emotional reaction and generic question, with no disclosed event or fate.'),
'3279':(None,'Unnamed woman and missing lead-in; may be a later escape scene, but matching the surrounding plot would add information absent from the excerpt.'),
'3306':(None,'Being together can mean companionship or confirmation of romance. Need the surrounding text and a consistent relationship-spoiler boundary.'),
'2904':(None,'Serenity Valley is opening backstory, not a late-series resolution. Whether this specific battle detail exceeds the allowed public premise is a policy boundary; do not force a label. Qwen’s claim about a series-finale consequence is misleading: the intended pilot aired last.'),
'3473':(0,'A character name and episode title alone do not reveal a consequential plot event. Qwen treats mere existence of an episode character as sufficient.'),
'2955':(0,'Vague weapon-effectiveness comparison; missing referent and no disclosed outcome.'),
'3367':(0,'Generic question about calling another character; no revealing event.'),
'7235':(1,'Reveals prior arrests/crimes and a specific financial result. The clause about conviction is an expectation, not confirmation; Qwen overstates that part in its explanation.'),
'8585':(0,'Minor comic character behavior, not a consequential later outcome. The text does not reveal what was stolen or its importance.'),
'3254':(None,'Unnamed character and generic romance/career summary; cannot confidently resolve the referent or premise-versus-later-outcome boundary.'),
'7196':(None,'An antagonist roster could be introductory description or disclose later antagonist identities. The allowed premise does not establish this; Qwen’s marketing-material justification is not evidenced.'),
'7216':(1,'Root exploits the protagonists’ rescue behavior and defeats them: an explicit later tactical outcome.'),
'7302':(0,'A rhetorical dialogue fragment does not disclose the episode’s hidden identity. Do not add that twist from plot context.'),
'7240':(None,'Both woman and it are unresolved references; the excerpt cannot be confidently scored SAFE or SPOILER.'),
'2965':(None,'This fate is missing and the quote is truncated; uncertainty is justified without reconstructing the parent paragraph.'),
'3510':(1,'Explicitly discloses an in-story explanation said to arrive in the season-one finale. Classification follows the revealing excerpt; exact source timing remains unverified.'),
'7266':(None,'Technical backstory can be world-building or a meaningful later revelation. The timing and consequence of this specific claim are not established by the available plot summary.'),
'2949':(None,'The unspecified test and its referents are missing. Qwen invents a likely loyalty/Alliance interpretation without supporting evidence.'),
'2894':(0,'An incidental object is discarded; no consequential outcome, hidden identity, or fate is disclosed.'),
'8563':(0,'Describes a provocative scene, not by itself a consequential outcome or twist. Qwen equates a specific scene with a spoiler without identifying a consequential reveal.'),
'2948':(1,'Explicitly reveals a later episode’s climactic threat against Jayne. Ariel is not the pilot; Qwen’s chronology and SAFE justification are wrong.'),
'7227':(None,'Missing groups and earlier clauses, plus apparently/speculative wording; cannot establish the asserted later plot reveal.'),
'7229':(1,'Explicitly reveals an episode-ending acquisition of knowledge about the secret Machine.'),
'7215':(0,'Recurring recognition/clothing characterization, without an important identity twist. Qwen’s claim that Carter is introduced later is wrong; she appears in the pilot.'),
'7309':(None,'This has no referent; a vague explosive flashback description does not establish a specific fate or outcome.')
}
sources={
 '3445':['https://en.wikipedia.org/wiki/Fringe_season_3'],
 '3279':['https://en.wikipedia.org/wiki/Olivia_(Fringe_episode)'],
 '2904':['https://en.wikipedia.org/wiki/Serenity_(Firefly_episode)'],
 '7216':['https://en.wikipedia.org/wiki/Firewall_(Person_of_Interest)'],
 '7302':['https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep7'],
 '2965':['https://en.wikipedia.org/wiki/Ariel_(Firefly_episode)'],
 '2894':['https://en.wikipedia.org/wiki/Ariel_(Firefly_episode)'],
 '2948':['https://en.wikipedia.org/wiki/Ariel_(Firefly_episode)'],
 '7229':['https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep22'],
 '7215':['https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep1'],
 '7247':['https://en.wikipedia.org/wiki/Person_of_Interest_season_1#ep4']
}
records=[]
for e in errors:
 k=e['id'].split(':')[-1];label,reason=reviews[k];p=e['prediction'];raw=json.loads(p['raw'])['label']
 if label is None: category='unresolved_excerpt_or_policy_boundary'
 elif not p['valid']:category='quote_validation_failure'
 elif (p['answer']['label']=='SPOILER')==bool(label):category='dataset_policy_disagreement'
 else:category='model_classification_error'
 records.append(dict(id=e['id'],title=e['title'],excerpt=e['excerpt'],original_label=e['source_label'],review_label=label,raw_model_label=raw,accepted_model_label=p['answer']['label'],valid=p['valid'],category=category,rationale=reason,source_urls=sources.get(k,[]),source_check='cached episode summary and/or browsed page' if k in sources else 'excerpt-only policy judgment; no plot verification claimed',reviewer='assistant',human_verified=False,blind_review=False))
assert len(records)==len(errors)==31 and len({r['id'] for r in records})==31
counts=dict(Counter(r['category'] for r in records));changes=sum(r['review_label'] is not None and r['review_label']!=r['original_label'] for r in records)
summary=dict(reviewed=31,total_run=100,unreviewed=69,categories=counts,resolved_original_label_changes=changes,warning='Only disagreements reviewed, with model answers visible. These are non-blind AI policy judgments, not independent human gold. Do not derive corrected overall accuracy from selectively relabeled errors. Original labels, predictions and scores unchanged.')
(out/'error-review.json').write_text(json.dumps(dict(summary=summary,records=records),indent=2,ensure_ascii=False)+'\n')
lines=['# Review of 31 Qwen disagreements and uncertain answers','',summary['warning'],'','| Review category | Count |','|---|---:|']
lines += [f'| {k} | {v} |' for k,v in counts.items()]
lines += ['',f'Resolved labels that differ from original markup: {changes}. The other 69 sentences have not been independently reviewed, so no revised full-run accuracy is reported. No API calls, retraining, or changes to historical scores.','', '## Concrete model issues', '', '- Rejecting an episode-level outcome because it is not a whole-series twist (7247).', '- Calling Ariel the pilot and dismissing its climactic threat (2948).', '- Treating a character name, a provocative scene, or recurring recognition as consequential spoilers (3473, 8563, 7215).', '- Invalid evidence quotes remain failures (7216, 7302); underlying raw labels agree with this audit, but quotes were altered.', '', '## Recommended next step', '', 'Review all 100 examples independently with model predictions hidden, using one explicit policy for public premise, episode outcomes, romance, and minor scene details. Mark fragments with missing referents as unresolved instead of forcing a binary label. Preserve these labels separately from the source dataset. Then evaluate Qwen again against that locked review; avoid tuning the prompt to this error-only audit.','', '## Every reviewed example','']
for r in records:
 lines += [f'### {r["id"]} — {r["title"]}', '',r['excerpt'],'',f'Original: {r["original_label"]}; reviewed: {r["review_label"]}; raw Qwen: {r["raw_model_label"]}; accepted Qwen: {r["accepted_model_label"]}. Category: {r["category"]}.','',r['rationale'],'']
 if r['source_urls']:lines += ['Sources: '+', '.join(f'[plot reference {i+1}]({url})' for i,url in enumerate(r['source_urls']))+'.','']
(out/'ERROR_REVIEW.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
