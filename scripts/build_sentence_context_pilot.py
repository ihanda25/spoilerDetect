"""Build a reproducible sentence/context preparation pilot; never launches training."""
import csv,hashlib,json,re,tarfile,io
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/context-training-data-audit'
OUT=ROOT/'data/processed/sentence-context-pilot'

def digest(text):return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def rank(row):return hashlib.sha256(str(row['source_row']).encode()).hexdigest()

def main():
 contexts=json.loads((REPORT/'pilot-contexts.json').read_text())
 judgments_path=REPORT/'pilot-quality-judgments.json'
 judgments=json.loads(judgments_path.read_text()) if judgments_path.exists() else {}
 raw={}
 for split in ('train','validation','test'):
  with (ROOT/f'data/raw/tvtropes/{split}.csv').open() as f:raw[split]=list(csv.DictReader(f))
 with tarfile.open(ROOT/'data/raw/tvtropes/original.tar.gz') as t:
  m=next(m for m in t.getmembers() if 'dev2' in m.name and m.isfile())
  raw['reserved_dev2']=list(csv.DictReader(io.TextIOWrapper(t.extractfile(m))))
 # Held-out texts used only for exact duplicate exclusion, not source mapping or label tuning.
 heldout={digest(r['sentence']) for split in ('validation','test','reserved_dev2') for r in raw[split]}
 traincounts=Counter(digest(r['sentence']) for r in raw['train'])
 trainlabels=defaultdict(set)
 for r in raw['train']:trainlabels[digest(r['sentence'])].add(r['spoiler'])
 contradictory={d for d,labels in trainlabels.items() if len(labels)>1}
 allpairs=[];eligible=[];seen=set();excluded=Counter();inventory=[]
 for split,rows in raw.items():
  c=Counter(r['page'] for r in rows)
  for page,count in sorted(c.items()):
   context=contexts.get(page)
   inventory.append({'split':split,'work_page':page,'rows':count,'mapping_status':'assistant_checked_series_identity' if context else 'unresolved','source_url':context['source_url'] if context else None,'canonical_title':context['canonical_title'] if context else None,'episode_mapping':'not_verified','context_scope':context['scope'] if context else None})
  if split not in ('train','validation'):continue
  for i,r in enumerate(rows):
   if r['page'] not in contexts:continue
   context=contexts[r['page']];assert context['split']==split
   item={'id':f'tvtropes:{split}:{i}','source_row':i,'source_split':split,'work_page':r['page'],'canonical_title':context['canonical_title'],'text':r['sentence'],'label':int(r['spoiler']=='True'),'public_premise':context['premise'],'plot_context':context['plot'],'context_source_url':context['source_url'],'context_scope':context['scope'],'human_verified':False,'training_allowed':False,'review_status':'pending_sentence_context_and_label_review'}
   allpairs.append(item);d=digest(item['text'])
   if split=='train':
    if d in heldout:excluded['heldout_text_overlap']+=1;continue
    if d in contradictory:excluded['contradictory_training_text']+=1;continue
    if d in seen:excluded['duplicate_training_text']+=1;continue
    seen.add(d)
   eligible.append(item)
 OUT.mkdir(parents=True,exist_ok=True)
 jsonl(OUT/'candidate-pairs.jsonl',allpairs)
 selected={};quality=[]
 for split,target in [('train',50),('validation',20)]:
  selected[split]=[]
  for page,context in contexts.items():
   if context['split']!=split:continue
   rs=sorted([r for r in eligible if r['source_split']==split and r['work_page']==page],key=rank)
   chosen=[]
   for label in (0,1):chosen.extend([r for r in rs if r['label']==label][:target//2])
   ids={r['id'] for r in chosen};chosen.extend([r for r in rs if r['id'] not in ids][:target-len(chosen)])
   selected[split].extend(sorted(chosen,key=rank))
   # Fixed label-independent sample for context adequacy inspection, not benchmark scoring.
   for r in sorted([r for r in allpairs if r['work_page']==page],key=rank)[:8]:quality.append(dict(r,context_adequacy='pending',concerns=[],reviewer='assistant_first_pass'))
  jsonl(OUT/f'{split}.jsonl',selected[split])
 for r in quality:
  if r['id'] in judgments:r.update(judgments[r['id']])
 jsonl(OUT/'quality-review.jsonl',quality)
 jsonl(REPORT/'work-inventory.jsonl',inventory)
 trainpages={r['work_page'] for r in selected['train']};devpages={r['work_page'] for r in selected['validation']}
 assert trainpages.isdisjoint(devpages)
 assert not ({digest(r['text']) for r in selected['train']} & heldout)
 assert all(r['source_split']!='test' and not r['training_allowed'] for rs in selected.values() for r in rs)
 manifest={'date':'2026-10-06','status':'preparation_pilot_only','training_started':False,'training_ready':False,'mapped_work_pages':len(contexts),'total_work_pages':len(inventory),'mapped_candidate_rows':len(allpairs),'mapped_candidate_rows_by_split':dict(Counter(r['source_split'] for r in allpairs)),'mapping_coverage_is_lower_bound':True,'remaining_unmapped_pages':sum(r['mapping_status']=='unresolved' for r in inventory),'cleaning_exclusions_train':dict(excluded),'pilot':{s:{'rows':len(rs),'positives':sum(r['label'] for r in rs),'works':len({r['work_page'] for r in rs})} for s,rs in selected.items()},'quality_sample_rows':len(quality),'quality_adequacy_counts':dict(Counter(r['context_adequacy'] for r in quality)),'context_sha256':hashlib.sha256((REPORT/'pilot-contexts.json').read_bytes()).hexdigest(),'raw_hashes':{s:hashlib.sha256((ROOT/f'data/raw/tvtropes/{s}.csv').read_bytes()).hexdigest() for s in ('train','validation','test')},'output_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')},'limitations':['Seven selected TV shows; not exhaustive mapping of the corpus','Identity checks are assistant judgments using source pages and sampled characters, not human verification','Short work-level paraphrases, often premise-only; not episode-complete context','No claim of canonical IMDb-ID matching or franchise-disjointness','Original spoiler-markup labels preserved, not relabeled into product policy','Validation is development and sample-reviewed; test and dev2 excluded from pilot','Balanced per-work pilot is not realistic browsing prevalence','Existing imported RoBERTa checkpoint may have unknown corpus exposure','All pilot rows require context/label review before training']}
 save(REPORT/'pilot-manifest.json',manifest)
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
