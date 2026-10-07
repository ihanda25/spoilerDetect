import json,zipfile,csv,tarfile
from pathlib import Path
from collections import Counter
import numpy as np
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'data/raw/imdb/dataset.zip') as z:
 with z.open('IMDB_movie_details.json') as f:movies={r['movie_id']:r for r in map(json.loads,f)}
 details={field:{'nonempty_movies':sum(bool(r.get(field,'').strip()) for r in movies.values()),'word_percentiles':{str(p):round(float(np.percentile([len(r[field].split()) for r in movies.values() if r.get(field,'').strip()],p)),1) for p in (50,90,100)}} for field in ('plot_summary','plot_synopsis')}
 raw=Counter();missing=Counter()
 with z.open('IMDB_reviews.json') as f:
  for r in map(json.loads,f):
   raw['reviews']+=1;raw['spoilers']+=int(r['is_spoiler'])
   m=movies.get(r['movie_id'],{})
   for field in ('plot_summary','plot_synopsis'):
    if m.get(field,'').strip():raw['reviews_with_'+field]+=1
   if not m:missing[r['movie_id']]+=1
splits={};sets={}
for split in ('train','validation','test'):
 ids=np.load(root/f'data/full/{split}.movie_ids.npy');labels=np.load(root/f'data/full/{split}.labels.npy');sets[split]=set(map(str,ids))
 c=Counter(map(str,ids));stats={'reviews':len(labels),'spoilers':int(labels.sum()),'movies':len(c)}
 for field in ('plot_summary','plot_synopsis'):
  coverage=[i for i in c if movies.get(i,{}).get(field,'').strip()]
  covered_ids=set(coverage)
  stats[field]={'movies':len(coverage),'reviews':sum(c[i] for i in coverage),'spoilers':int(sum(int(label) for i,label in zip(ids,labels) if str(i) in covered_ids))}
 splits[split]=stats
sentences={}
for split in ('train','validation','test'):
 with (root/f'data/raw/tvtropes/{split}.csv').open() as f:rs=list(csv.DictReader(f))
 sentences[split]={'rows':len(rs),'spoilers':sum(r['spoiler']=='True' for r in rs),'work_pages':len({r['page'] for r in rs}),'has_plot_field':False}
with tarfile.open(root/'data/raw/tvtropes/original.tar.gz') as t:
 dev2=[m for m in t.getmembers() if 'dev2' in m.name and m.isfile()]
 for m in dev2:
  import io
  rs=list(csv.DictReader(io.TextIOWrapper(t.extractfile(m))))
  sentences['reserved_dev2']={'rows':len(rs),'spoilers':sum(r['spoiler']=='True' for r in rs),'work_pages':len({r['page'] for r in rs})}
result={'date':'2026-10-06','imdb_metadata_movies':len(movies),'context_fields':details,'raw_imdb':dict(raw),'missing_metadata_ids':dict(missing),'prepared_movie_disjoint_splits':splits,'movie_overlap':{a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]},'sentence_supervision_tvtropes':sentences,'paired_sentence_context_ready':0,'paired_sentence_context_ready_note':'No existing joined sentence/plot training corpus. TV Tropes work-page mapping to context must be built and audited; IMDb labels are review-level.'}
path=root/'reports/context-training-data-audit';path.mkdir(exist_ok=True);(path/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
