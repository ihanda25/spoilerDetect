"""Lock a prediction-hidden AI review before opening predictions and scoring."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/qwen-alone-sentences-100'
OUT=BASE/'blind-review'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');temp.replace(path)

def lock():
    mapping=json.loads((OUT/'mapping.json').read_text());labels=[]
    for part in ['a','b']:
        inputs=json.loads((OUT/f'inputs-{part}.json').read_text());review=json.loads((OUT/f'labels-{part}.json').read_text())
        assert isinstance(review,list) and len(review)==50
        assert len({r['id'] for r in review})==50 and {r['id'] for r in review}=={r['id'] for r in inputs}
        for r in review:
            assert r['label'] is None or type(r['label']) is int and r['label'] in [0,1]
            assert r['rationale'] and r['confidence'] in ['high','medium','low']
            assert isinstance(r['source_urls'],list) and r['source_check']
            labels.append(dict(r,original_id=mapping[r['id']],reviewer='prediction-hidden independent AI reviewer '+part,human_verified=False))
    labels=sorted(labels,key=lambda r:r['id']);assert len({r['original_id'] for r in labels})==100
    sources={p.name:digest(p) for p in [OUT/'POLICY.txt',OUT/'inputs-a.json',OUT/'inputs-b.json',OUT/'labels-a.json',OUT/'labels-b.json',OUT/'mapping.json']}
    path=OUT/'LOCK.json'
    if path.exists():assert json.loads(path.read_text())['source_sha256']==sources,'Locked review changed; preserve it and use a new version.'
    else:
        save(OUT/'locked-labels.json',labels)
        save(path,dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,labels_sha256=digest(OUT/'locked-labels.json'),warning='Two prediction-hidden AI reviewers, disjoint 50-row assignments; not human gold, not double review. Parent had prior exposure; labels authored by reviewers without forked history. No revised prompts or model inference.'))
    assert json.loads(path.read_text())['labels_sha256']==digest(OUT/'locked-labels.json')
    return labels

def score(labels):
    # Predictions are read only after review has passed validation and been locked.
    predictions=json.loads((BASE/'alone.json').read_text())['predictions'];inputs=json.loads((BASE/'inputs.json').read_text());original=json.loads((BASE/'annotations.json').read_text())
    refs={r['original_id']:r for r in labels};groups={};changes=[];cases=[]
    for group in ['all']+sorted({r['slice'] for r in inputs}):
        counts=Counter()
        for r in inputs:
            if group!='all' and r['slice']!=group:continue
            ref=refs[r['id']];p=predictions[r['id']];label=ref['label'];answer=p['answer']['label']
            if label is None:counts['unresolved_reference']+=1;continue
            counts['positives' if label else 'negatives']+=1
            if not p['valid']:counts['invalid']+=1
            if answer=='UNCERTAIN':counts['abstentions']+=1
            else:counts['tp' if label and answer=='SPOILER' else 'fn' if label else 'fp' if answer=='SPOILER' else 'tn']+=1
        pos=counts['positives'];neg=counts['negatives'];total=pos+neg
        groups[group]=dict(counts,recall=counts['tp']/pos if pos else None,false_positive_rate=counts['fp']/neg if neg else None,precision=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,coverage=1-counts['abstentions']/total if total else None,accuracy_with_abstentions_in_denominator=(counts['tp']+counts['tn'])/total if total else None)
    for r in inputs:
        ref=refs[r['id']];p=predictions[r['id']];item=dict(id=r['id'],blind_id=ref['id'],title=r['title'],excerpt=r['excerpt'],source_label=original[r['id']]['label'],review_label=ref['label'],review_rationale=ref['rationale'],prediction=p['answer'],raw_prediction=json.loads(p['raw']).get('label'),valid=p['valid'])
        if ref['label']!=original[r['id']]['label']:changes.append(item)
        if ref['label'] is None or p['answer']['label']=='UNCERTAIN' or (p['answer']['label']=='SPOILER')!=bool(ref['label']):cases.append(item)
    result=dict(warning='Independent prediction-hidden AI labels, not human gold. Unresolved references excluded from binary scores; abstained positives count as missed spoilers. Full 100-row review locked before scorer read predictions.',reference_counts=dict(Counter('spoiler' if r['label']==1 else 'safe' if r['label']==0 else 'unresolved' for r in labels)),groups=groups,labels_sha256=digest(OUT/'locked-labels.json'),resolved_source_label_changes=sum(x['review_label'] is not None for x in changes))
    save(OUT/'scores.json',result);save(OUT/'changed-labels.json',changes);save(OUT/'remaining-cases.json',cases)
    lines=['# Prediction-hidden review of all 100 Qwen sentences','',result['warning'],'','Two independent AI reviewers each reviewed 50 anonymized examples. They received titles, excerpts, supplied public premises and a fixed policy, with no model answers or source labels. They were allowed to check cached episode plots, but neither used plot sources; these are excerpt-and-policy judgments, not source-verified facts. Labels were validated and locked before the scorer opened predictions. No second reviewer per sentence; no inter-rater agreement claim.','',f'References: {result["reference_counts"]}. Resolved source-label changes: {result["resolved_source_label_changes"]}.','', '| Show | Spoilers caught | Safe falsely flagged | Abstentions on resolved labels | Unresolved references |','|---|---:|---:|---:|---:|']
    for name,m in groups.items():lines.append(f'| {name} | {m.get("tp",0)}/{m.get("positives",0)} | {m.get("fp",0)}/{m.get("negatives",0)} | {m.get("abstentions",0)} | {m.get("unresolved_reference",0)} |')
    lines+=['','## Overall', '', '```json', json.dumps(groups['all'],indent=2), '```', '','Accuracy includes abstentions as non-correct decisions but excludes unresolved reference labels. Original scores are preserved. An AI label review is not a human evaluation; results apply to this deliberately sampled development set only.','', 'See POLICY.txt, locked-labels.json and LOCK.json for exact labeling decisions and integrity hashes. remaining-cases.json includes unresolved references, model mismatches and abstentions for follow-up.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':score(lock())
