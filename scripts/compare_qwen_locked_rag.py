"""Frozen-label paired Qwen diagnostic with a persisted local hybrid vector index."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import numpy as np
import compare_qwen_rag as q
from diagnose_qwen_tv_context import normalized_check
from groq_usage_guard import Guard, BudgetStop

BASE=q.ROOT/'reports/qwen-alone-sentences-100'
OUT=q.ROOT/'reports/qwen-locked-hybrid-rag'
INDEX=q.ROOT/'data/processed/qwen-locked-hybrid-rag'
EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
TITLE_WORK={'Firefly (2002 TV series)':'Firefly','Fringe (2008 TV series)':'Fringe','Sherlock (2010 TV series)':'Sherlock','Person of Interest (2011 TV series)':'PersonOfInterest'}
STOP=set('a an the is are was were to of in on at with for from by and or but it its this that these those he she they his her their as be been do does did not when then'.split())

def tokens(text):return [t for t in q.terms(text) if t not in STOP]
def ascii_quotes(text):return ' '.join(text.translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"'})).split())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class ConservativePacer(q.Pacer):
    def delay(self,payload,now=None):
        wait=super().delay(payload,now)
        if self.last is None:return wait
        elapsed=(time.monotonic() if now is None else now)-self.last
        return max(wait,8-elapsed,1.75*60*self.spent/self.tpm-elapsed)

def chunks(plot,limit=460):
    # Keep sentence boundaries where possible; hard-split long sentences by words.
    sentences=re.split(r'(?<=[.!?])\s+',ascii_quotes(plot));pieces=[]
    for sentence in sentences:
        if len(sentence)<=limit:pieces.append(sentence);continue
        current=''
        for word in sentence.split():
            if len(current)+len(word)+1>limit and current:pieces.append(current);current=''
            current=(current+' '+word).strip()
        if current:pieces.append(current)
    result=[];current=''
    for piece in pieces:
        if current and len(current)+len(piece)+1>limit:result.append(current);current=''
        current=(current+' '+piece).strip()
    if current:result.append(current)
    return result

def embed(texts):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL,revision=EMBEDDING_REVISION,local_files_only=True)
    model=AutoModel.from_pretrained(EMBEDDING_MODEL,revision=EMBEDDING_REVISION,local_files_only=True).eval()
    arrays=[]
    with torch.inference_mode():
        for start in range(0,len(texts),16):
            x=tokenizer(texts[start:start+16],padding=True,truncation=True,max_length=256,return_tensors='pt')
            h=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1)
            v=(h*mask).sum(1)/mask.sum(1).clamp(min=1);v=torch.nn.functional.normalize(v,p=2,dim=1)
            arrays.append(v.numpy())
    return np.concatenate(arrays)

def retrieve(row,corpus,vectors,query):
    ids=[i for i,p in enumerate(corpus) if p['title']==row['title']]
    docs=[Counter(tokens(corpus[i]['episode_title']+' '+corpus[i]['text'])) for i in ids]
    df=Counter(t for d in docs for t in d);avg=sum(sum(d.values()) for d in docs)/len(docs)
    bm=[]
    for index,d in zip(ids,docs):
        score=0
        for t in set(tokens(row['excerpt'])):
            f=d[t]
            if f:score+=math.log(1+(len(docs)-df[t]+.5)/(df[t]+.5))*f*2.5/(f+1.5*(.25+.75*sum(d.values())/max(avg,1)))
        bm.append((score,index))
    lexical=sorted((x for x in bm if x[0]>0),key=lambda x:(-x[0],x[1]))[:20]
    similarities=vectors[ids]@query
    dense=sorted(zip(similarities.tolist(),ids),key=lambda x:(-x[0],x[1]))[:20]
    fused=Counter()
    for ranking in [lexical,dense]:
        for rank,(_,i) in enumerate(ranking,1):fused[i]+=1/(60+rank)
    chosen=sorted(fused,key=lambda i:(-fused[i],i))[:3]
    lex=dict((i,s) for s,i in bm);cos=dict((i,s) for s,i in dense)
    return [dict(id=corpus[i]['id'],text=corpus[i]['text'],source_url=corpus[i]['source_url'],episode_title=corpus[i]['episode_title'],rrf_score=fused[i],bm25_score=lex[i],cosine_similarity=float(vectors[i]@query)) for i in chosen]

def prepare():
    lock=json.loads((BASE/'blind-review/LOCK.json').read_text());labels_path=BASE/'blind-review/locked-labels.json'
    assert digest(labels_path)==lock['labels_sha256']
    refs={r['original_id']:r for r in json.loads(labels_path.read_text())}
    baseline=json.loads((BASE/'alone.json').read_text());original=json.loads((BASE/'inputs.json').read_text())
    assert baseline['config']['system']==q.SYSTEM and baseline['config']['model']==q.MODEL
    assert baseline['config']['inputs_sha256']==q.sha(original)
    source=q.ROOT/'data/raw/episode-context/episodes.jsonl'
    episodes=[json.loads(l) for l in source.read_text().splitlines()]
    corpus=[]
    for title,work in TITLE_WORK.items():
        selected=sorted((e for e in episodes if e['work_page']==work),key=lambda e:(e['plot_source_url'],e['episode_title']))
        for e in selected:
            for index,text in enumerate(chunks(e['plot'])):
                corpus.append(dict(id='plot-'+q.sha([e['plot_source_url'],index,text])[:16],title=title,episode_title=e['episode_title'],source_url=e['plot_source_url'],text=text))
    assert len({p['id'] for p in corpus})==len(corpus)
    index_config=dict(source_sha256=digest(source),corpus_sha256=q.sha(corpus),embedding_model=EMBEDDING_MODEL,revision=EMBEDDING_REVISION,encoding='attention-mask mean pooling, L2 normalization, max 256 tokens',chunking='nonoverlapping sentence/word chunks <=460 chars; whitespace collapse and ASCII quote conversion')
    INDEX.mkdir(parents=True,exist_ok=True)
    index_manifest=INDEX/'manifest.json'
    if index_manifest.exists():
        assert json.loads(index_manifest.read_text())==index_config
        vectors=np.load(INDEX/'vectors.npy');queries=np.load(INDEX/'queries.npy')
    else:
        print('Embedding',len(corpus),'plot chunks and 100 sentences locally; no training.',flush=True)
        vectors=embed([p['episode_title']+'. '+p['text'] for p in corpus]);queries=embed([r['excerpt'] for r in original])
        np.save(INDEX/'vectors.npy',vectors);np.save(INDEX/'queries.npy',queries);q.save(INDEX/'corpus.json',corpus);q.save(index_manifest,index_config)
    assert vectors.shape==(len(corpus),384) and queries.shape==(100,384)
    inputs=[]
    for row,query in zip(original,queries):
        row=copy.deepcopy(row);row['passages']=retrieve(row,corpus,vectors,query)
        assert all(next(p for p in corpus if p['id']==c['id'])['title']==row['title'] for c in row['passages'])
        inputs.append(row)
    manifest=dict(model=q.MODEL,system=q.SYSTEM,temperature=0,max_completion_tokens=512,rows=100,inputs_sha256=q.sha(inputs),locked_labels_sha256=lock['labels_sha256'],baseline_cache_sha256=digest(BASE/'alone.json'),retrieval='Within-title exact cosine search + BM25, top20 each, reciprocal rank fusion k60, top3; sentence-only query; no label-based filtering or tuning',index=index_config,validator='Strict schema and whitespace-only normalized substring, identical to alone arm',warning='Locked prediction-hidden AI development labels; 17 unresolved references excluded from binary metrics. Context corpus incomplete: episode summaries, no comics. Existing alone outputs reused; identical prompt/settings; cached-alone comparison is not a simultaneous rerun. No training or purchases.')
    for name,obj in [('inputs',inputs),('manifest',manifest),('annotations',{ident:dict(label=r['label']) for ident,r in refs.items()})]:q.save(OUT/(name+'.json'),obj)
    if not (OUT/'alone.json').exists():q.save(OUT/'alone.json',dict(baseline,reused_from=str(BASE/'alone.json')))
    q.save(OUT/'retrieval-diagnostics.json',dict(chunks=len(corpus),episodes=dict(Counter(e['work_page'] for e in episodes if e['work_page'] in TITLE_WORK.values())),rows_with_passages=sum(bool(r['passages']) for r in inputs),cross_title_passages=0,warning='Passage availability and source coverage are not relevance/answerability measures; retrieval relevance not independently labeled.'))
    (OUT/'PLAN.md').write_text('# Locked Qwen-alone versus hybrid RAG\n\n100 unchanged inputs, frozen prompt/model/settings and locked AI labels. Reuse alone outputs; run only RAG. Local persisted all-MiniLM-L6-v2 vector index plus BM25, title filter, RRF k60, top3 plot chunks. No fine-tuning or purchases. Groq Free-plan requests paced at >=8s with measured token headroom and bounded retries. Cumulative hard caps: 120 request attempts, 100000 input/output tokens; unknown usage reserved conservatively, no paid fallback. Every answer saved; rerun --run to resume. Binary metrics exclude 17 unresolved labels; report evidence validation separately from raw labels. Source corpus is incomplete, notably Firefly comic revelations.\n')
    return inputs,manifest,{ident:dict(label=r['label']) for ident,r in refs.items()}

def report(inputs,annotations):
    caches={arm:json.loads((OUT/(arm+'.json')).read_text()) for arm in ['alone','rag']}
    q.summarize(inputs,caches,annotations)
    accepted=json.loads((OUT/'comparison.json').read_text())
    raw=copy.deepcopy(caches)
    for cache in raw.values():
        for p in cache['predictions'].values():
            try:
                a=json.loads(p['raw'])
                if a.get('label') in ['SAFE','SPOILER','UNCERTAIN'] and not p.get('api_error'):p['answer']=a;p['valid']=True
            except (ValueError,TypeError):pass
    q.summarize(inputs,raw,annotations);q.save(OUT/'raw-label-comparison.json',json.loads((OUT/'comparison.json').read_text()));q.save(OUT/'comparison.json',accepted)
    changes=[]
    for row in inputs:
        a=caches['alone']['predictions'][row['id']];b=caches['rag']['predictions'].get(row['id'])
        if b and (a['answer']['label']!=b['answer']['label'] or not b['valid']):changes.append(dict(id=row['id'],title=row['title'],excerpt=row['excerpt'],reference=annotations[row['id']]['label'],alone=a,rag=b,passages=row['passages']))
    q.save(OUT/'changed-decisions.json',changes)
    status=json.loads((OUT/'status.json').read_text());lines=['# Locked Qwen-alone versus hybrid RAG','',f'Complete: {status["complete"]}. Same 100 inputs and locked AI labels; 17 unresolved references excluded. Alone outputs reused, RAG outputs newly generated. No training or purchases.','', '| Arm | Spoilers caught | Safe falsely flagged | Abstentions | Invalid answers (all 100) |','|---|---:|---:|---:|---:|']
    for arm in ['alone','rag']:
        m=accepted['arms'][arm]['all'];lines.append(f'| {arm} | {m.get("tp",0)}/{m.get("positives",0)} | {m.get("fp",0)}/{m.get("negatives",0)} | {m.get("abstentions",0)} | {sum(not p["valid"] for p in caches[arm]["predictions"].values())} |')
    lines+=['','Actual retrieval: title-filtered local MiniLM vector index plus BM25, sentence-only query, top3 chunks by fixed RRF. Corpus consists of cached episode summaries; incomplete coverage, particularly comic-only revelations. No label-based tuning. Passage existence does not establish passage relevance.','', 'Accepted results require valid evidence quotes. raw-label-comparison.json separates semantic decisions from quotation failures; raw results do not certify evidence grounding. changed-decisions.json preserves passages and both responses for inspection.','', 'These are AI development labels, not human gold. Locked labels and original baseline files remain unchanged.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({a:accepted['arms'][a]['all'] for a in ['alone','rag']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    q.OUT=OUT;q.check_answer=normalized_check;q.Pacer=ConservativePacer
    inputs,manifest,annotations=prepare()
    print('Prepared paired 100-row comparison; locked labels verified.',flush=True)
    if args.run:
        guard=Guard(q,OUT,max_requests=120,max_tokens=100000)
        q.urlopen=guard.open
        try:q.run(inputs,manifest,120,arms=('rag',),annotations=annotations)
        except BudgetStop as exc:
            count=len(json.loads((OUT/'rag.json').read_text())['predictions'])
            q.save(OUT/'status.json',dict(complete=False,reason=str(exc),predictions=dict(rag=count)))
            print('STOPPED:',str(exc),flush=True)
    if (OUT/'rag.json').exists():report(inputs,annotations)
