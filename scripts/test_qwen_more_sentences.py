"""100 fresh Qwen sentence decisions, no retrieval or training; resumable."""
import argparse
from collections import Counter
import json
from pathlib import Path
import compare_qwen_rag as q
from diagnose_qwen_tv_context import normalized_check
OUT=q.ROOT/'reports/qwen-alone-sentences-100'

def prepare():
    prior=json.loads((q.ROOT/'reports/qwen-alone-expanded/inputs.json').read_text())
    used_ids={r['id'] for r in prior}; used_text={' '.join(r['excerpt'].split()) for r in prior}
    source=q.ROOT/'data/processed/tv-training-experiment/train.jsonl'
    candidates=[json.loads(l) for l in source.read_text().splitlines()]
    inputs=[];annotations={};selection=[]
    for index,work in enumerate(['Firefly','Fringe','Sherlock','PersonOfInterest']):
        for label,count in [(1,13 if index%2==0 else 12),(0,12 if index%2==0 else 13)]:
            pool=sorted((r for r in candidates if r['work_page']==work and r['label']==label and r['id'] not in used_ids and 35<=len(r['text'])<=1500),key=lambda r:q.sha(r['id']))
            chosen=[]
            for r in pool:
                norm=' '.join(r['text'].split())
                if norm in used_text: continue
                used_text.add(norm);chosen.append(r)
                if len(chosen)==count:break
            assert len(chosen)==count,(work,label,len(chosen))
            for r in chosen:
                inputs.append(dict(id=r['id'],title=r['canonical_title'],excerpt=r['text'],public_premise=r['public_premise'],slice=work,passages=[]))
                annotations[r['id']]=dict(id=r['id'],label=label,label_provenance='original_corpus_markup',human_verified=False)
                selection.append(dict(id=r['id'],work_page=work,source_split=r['source_split'],source_row=r['source_row'],label=label))
    inputs=sorted(inputs,key=lambda r:q.sha(r['id']))
    assert len(inputs)==100 and sum(a['label'] for a in annotations.values())==50
    manifest=dict(model=q.MODEL,system=q.SYSTEM,inputs_sha256=q.sha(inputs),annotations_sha256=q.sha(annotations),rows=100,temperature=0,max_completion_tokens=512,arms=['alone'],retrieval=False,validator='Strict schema, whitespace-only normalized quote substring',selection='Deterministic ID-hash sample: 25 per show, 50 positive/50 negative total, text length 35-1500 characters, exclude prior IDs and normalized duplicate text',warning='Corpus spoiler markup, not human gold. Source is the prior classifier training partition; these are new Qwen inference examples, not a held-out cross-model test. Titles and public premises supplied, no plot passages or retrieval.')
    for name,obj in [('inputs',inputs),('annotations',annotations),('selection',selection),('manifest',manifest)]:q.save(OUT/(name+'.json'),obj)
    (OUT/'PLAN.md').write_text('# Qwen-only sentence test\n\n100 new sentences across four shows, balanced by corpus spoiler markup. No RAG, training, or purchases. Existing Groq Free-plan key, adaptive pacing and bounded retries. Each answer saved immediately; rerun the same script with --run to resume. No labels or plot contexts sent to the model.\n\nLabels are noisy corpus markup, not human gold; source is the former classifier training partition. Selection is deterministic and excludes previously tested text.\n')
    return inputs,manifest,annotations

def report(inputs,annotations):
    cache=json.loads((OUT/'alone.json').read_text());q.summarize(inputs,{'alone':cache},annotations)
    metrics=json.loads((OUT/'comparison.json').read_text())['arms']['alone']
    status=json.loads((OUT/'status.json').read_text())
    lines=['# Qwen-only: 100 fresh TV sentences','',f'Complete: {status["complete"]}. No retrieved plot context or training. Title and public premise supplied.','', '| Show | Spoilers caught | Safe falsely flagged | Abstentions |', '|---|---:|---:|---:|']
    for name in ['all','Firefly','Fringe','Sherlock','PersonOfInterest']:
        m=metrics[name];lines.append(f'| {name} | {m.get("tp",0)}/{m.get("positives",0)} | {m.get("fp",0)}/{m.get("negatives",0)} | {m.get("abstentions",0)} |')
    lines+=['', 'Recall includes abstained positives as missed spoilers. Whitespace-only normalized quotation validation; other evidence edits remain invalid. Corpus markup is not human gold and may disagree with the spoiler policy. This balanced sample does not represent natural prevalence. Rows come from the previous classifier training partition, but none were included in the previous Qwen sentence evaluation. No held-out or production-accuracy claim.', '', 'Errors and abstentions are saved in errors.json with source labels, text, raw model decisions, and validation outcomes. Review them before interpreting disagreements as model failures.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    errors=[]
    for r in inputs:
        p=cache['predictions'].get(r['id'])
        if p and (not p['valid'] or p['answer']['label']=='UNCERTAIN' or (p['answer']['label']=='SPOILER')!=bool(annotations[r['id']]['label'])):
            errors.append(dict(id=r['id'],title=r['title'],excerpt=r['excerpt'],source_label=annotations[r['id']]['label'],prediction=p))
    q.save(OUT/'errors.json',errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    q.OUT=OUT;q.check_answer=normalized_check
    inputs,manifest,annotations=prepare()
    print('Prepared 100 new unique sentences; balanced 50/50; no retrieval.',flush=True)
    if args.run:q.run(inputs,manifest,110,arms=('alone',),annotations=annotations)
    if (OUT/'alone.json').exists():report(inputs,annotations)
