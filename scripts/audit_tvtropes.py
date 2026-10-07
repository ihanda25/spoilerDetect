"""Read-only structural audit of downloaded TV Tropes splits; no ML training."""
import csv,io,json,hashlib,tarfile,collections,itertools,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/tvtropes'
OUT=ROOT/'reports/tvtropes-audit'
def norm(s):return ' '.join(s.casefold().split())
def rows(b):return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))
def main():
 OUT.mkdir(exist_ok=True,parents=True)
 splits={s:rows((RAW/f'{s}.csv').read_bytes()) for s in ['train','validation','test']}
 report={'splits':{},'overlap':{},'files':{},'original_archive':{}}
 for name in RAW.iterdir():report['files'][name.name]={'bytes':name.stat().st_size,'sha256':hashlib.sha256(name.read_bytes()).hexdigest()}
 for s,rs in splits.items():
  texts=collections.Counter(norm(r['sentence']) for r in rs)
  lens=sorted(len(r['sentence'].split()) for r in rs)
  report['splits'][s]={'rows':len(rs),'columns':list(rs[0]),'labels':dict(collections.Counter(r['spoiler'] for r in rs)),'pages':len({r['page'] for r in rs}),'tropes':len({r['trope'] for r in rs}),'empty_texts':sum(not norm(r['sentence']) for r in rs),'missing_fields':sum(any(v is None or not v.strip() for v in r.values()) for r in rs),'duplicate_normalized_extra_rows':sum(v-1 for v in texts.values()),'word_lengths':{str(p):lens[min(len(lens)-1,int((len(lens)-1)*p/100))] for p in [0,50,90,95,99,100]},'explicit_spoiler_word_rows':sum('spoiler' in norm(r['sentence']) for r in rs)}
 for a,b in itertools.combinations(splits,2):
  report['overlap'][a+'__'+b]={key:len({norm(r[key]) for r in splits[a]} & {norm(r[key]) for r in splits[b]}) for key in ['sentence','page','trope']}
 labels=collections.defaultdict(set)
 for rs in splits.values():
  for r in rs:labels[norm(r['sentence'])].add(r['spoiler'])
 report['conflicting_normalized_texts']=sum(len(v)>1 for v in labels.values())
 report['conflict_locations']=[{'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'rows':[{'split':s,'row_index':i,'label':r['spoiler']} for s,rs in splits.items() for i,r in enumerate(rs) if norm(r['sentence'])==t]} for t,v in labels.items() if len(v)>1]
 with tarfile.open(RAW/'original.tar.gz') as archive:
  for member in archive.getmembers():
   if member.isfile():
    content=archive.extractfile(member).read()
    rs=rows(content)
    report['original_archive'][member.name]={'rows':len(rs),'pages':len({r['page'] for r in rs}),'page_overlap_with_downloaded':{s:len({norm(r['page']) for r in rs}&{norm(r['page']) for r in sr}) for s,sr in splits.items()},'sha256':hashlib.sha256(content).hexdigest(),'identical_to_downloaded':[s for s in splits if content==(RAW/f'{s}.csv').read_bytes()]}
 # Deterministic development-only sample; never sample the held-out test.
 sample=[]
 rng=random.Random(42)
 for s in ['train','validation']:
  for label in ['True','False']:
   candidates=[(i,r) for i,r in enumerate(splits[s]) if r['spoiler']==label]
   for i,r in rng.sample(candidates,15):sample.append({'split':s,'row_index':i,**r})
 (RAW/'sample-review.json').write_text(json.dumps(sample,indent=2))
 (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
