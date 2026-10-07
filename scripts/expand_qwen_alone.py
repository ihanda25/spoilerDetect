"""Qwen-only exploratory dev evaluation; no retrieval, training or held-out test."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import compare_qwen_rag as q

OUT = q.ROOT/'reports/qwen-alone-expanded'


def prepare():
    base=q.ROOT/'reports/eval-refresh'
    rows=[json.loads(l) for l in (base/'blind-inputs-v2.jsonl').read_text().splitlines()]
    meta=json.loads((base/'input-metadata-v2.json').read_text())
    contexts=json.loads((q.ROOT/'reports/context-comparison/contexts-v1.json').read_text())
    annotations={r['id']:r for r in map(json.loads,(base/'reviewed-dev-v2.jsonl').read_text().splitlines())}
    inputs=[]
    for row in rows:
        title=meta[row['id']].get('work_title')
        inputs.append(dict(id=row['id'],title=title,excerpt=row['text'],
                           public_premise=contexts.get(title,{}).get('premise',''),
                           slice=meta[row['id']]['slice'],passages=[]))
    tv=[json.loads(l) for l in (q.ROOT/'data/processed/tv-training-experiment/validation.jsonl').read_text().splitlines()]
    selected=[]
    for label in (0,1):
        candidates=sorted([r for r in tv if r['label']==label],key=lambda r:q.sha(r['id']))
        if len(candidates)<30:raise ValueError('Insufficient balanced TV development rows')
        selected.extend(candidates[:30])
    # Freeze hash-selected order; labels never enter API payloads.
    selected=sorted(selected,key=lambda r:q.sha(r['id']))
    for row in selected:
        inputs.append(dict(id=row['id'],title=row['canonical_title'],excerpt=row['text'],
                           public_premise=row['public_premise'],slice='tv_markup_dev',passages=[]))
        annotations[row['id']]=dict(id=row['id'],label=row['label'],label_provenance='original_corpus_markup',human_verified=False)
    assert len(inputs)==267 and len({r['id'] for r in inputs})==267
    manifest=dict(model=q.MODEL,system=q.SYSTEM,inputs_sha256=q.sha(inputs),rows=len(inputs),
                  temperature=0,max_completion_tokens=512,arms=['alone'],retrieval=False,
                  annotations_sha256=q.sha({r['id']:annotations[r['id']] for r in inputs}),
                  slices=dict(Counter(r['slice'] for r in inputs)),
                  warning='Exploratory development only: AI labels and TV corpus markup, not human gold. Plot challenges share sources. No held-out test.')
    q.save(OUT/'inputs.json',inputs);q.save(OUT/'manifest.json',manifest)
    q.save(OUT/'annotations.json',{r['id']:annotations[r['id']] for r in inputs})
    path=OUT/'alone.json'
    if not path.exists():
        old=q.ROOT/'reports/qwen-rag'
        cache=json.loads((old/'alone.json').read_text())
        prior={r['id']:r for r in json.loads((old/'inputs.json').read_text())}
        imported={}
        assert cache['config']['system']==q.SYSTEM and cache['config']['model']==q.MODEL
        assert cache['config']['temperature']==0 and cache['config']['max_completion_tokens']==512
        assert cache['config']['inputs_sha256']==q.sha(list(prior.values()))
        for row in inputs:
            p=cache['predictions'].get(row['id'])
            if p is not None:
                assert all(row[k]==prior[row['id']][k] for k in ('title','excerpt','public_premise'))
                imported[row['id']]=dict(p,reused_from='reports/qwen-rag/alone.json')
        q.save(path,dict(config=dict(manifest,arm='alone'),predictions=imported))
        q.save(OUT/'reuse.json',dict(count=len(imported),prior_cache_sha256=q.sha(cache),matched_fields=['title','excerpt','public_premise'],model_and_prompt_unchanged=True))
    return inputs,manifest,{r['id']:annotations[r['id']] for r in inputs}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--max-requests',type=int,default=240)
    args=parser.parse_args()
    rows,manifest,annotations=prepare()
    q.OUT=OUT
    q.summarize(rows,{'alone':json.loads((OUT/'alone.json').read_text())},annotations)
    print('Prepared',len(rows),'Qwen-alone examples; reused',json.loads((OUT/'reuse.json').read_text())['count'],'answers.',flush=True)
    if not args.prepare:
        q.save(OUT/'process.json',dict(pid=os.getpid(),model=q.MODEL,account_plan='Free, user confirmed',pacing='adaptive with 50% headroom',arms=['alone']))
        q.run(rows,manifest,args.max_requests,arms=('alone',),annotations=annotations)
        status=json.loads((OUT/'status.json').read_text())
        metrics=json.loads((OUT/'comparison.json').read_text())['arms']['alone']
        preds=json.loads((OUT/'alone.json').read_text())['predictions']
        lines=['# Expanded Qwen-alone results — October 6, 2026','',
               'Status: '+('completed all 267 examples.' if status['complete'] else 'partial; '+status['reason']+'.'),'',
               '| Slice | Spoilers caught | Safe falsely flagged | Abstentions on binary labels | Pending |',
               '| --- | ---: | ---: | ---: | ---: |']
        for name in ['real_content_dev','source_plot_challenge','tv_markup_dev']:
            m=metrics[name]
            lines.append(f'| {name} | {m.get("tp",0)}/{m.get("positives",0)} | {m.get("fp",0)}/{m.get("negatives",0)} | {m.get("abstentions",0)} | {m.get("pending",0)} |')
        lines+=['',f'Invalid answers: {sum(not p["valid"] for p in preds.values())}; API JSON-generation failures: {sum(bool(p.get("api_error")) for p in preds.values())}. These become UNCERTAIN, not safe. Recall includes abstained positives as misses.',
                '', 'Real-content and source-plot challenge labels are AI-reviewed; TV labels are corpus markup. None are human gold. Only five real-content positives are present. TV is a balanced 30/30 diagnostic from three shows, not natural prevalence or a new unseen benchmark. Source-plot challenges share sources and may be familiar to pretraining. Keep slices separate; do not interpret aggregate accuracy as production performance.',
                '', 'Same Qwen3.8-27B model, frozen system instructions, title/public premise inputs, temperature zero and 512 output-token ceiling. No plot retrieval or training. 79 identical earlier answers are reused with provenance. Updated transport: 4s minimum spacing, 50% token headroom, rate-limit headers and bounded short 429 retries. Specific HTTP 400 json_validate_failed responses are recorded as invalid abstentions so unrelated examples continue. Other API errors still stop. No paid fallback or purchases.']
        (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
