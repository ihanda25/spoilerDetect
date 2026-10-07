#!/usr/bin/env python3
"""Bounded public-excerpt collection; no labels, model calls or external code execution.
Run --collect once; otherwise rebuild deterministically from the saved excerpts.
Only writes data/raw/product-review and reports/real-review-pack.
"""
import argparse, collections, concurrent.futures, csv, datetime, hashlib, html, json, re, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/product-review'; OUT=ROOT/'reports/real-review-pack'
TODAY=datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(t):return hashlib.sha256(t.encode()).hexdigest()
def plain(t):return re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',t or ''))).strip()
def norm(t):return re.sub(r'[^a-z0-9]+',' ',t.lower()).strip()
def excerpt(t,n=35):
 w=t.split();return ' '.join(w[:n]),len(w)>n
LOG=[]
def fetch(u):
 try:
  with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'PublicExcerptResearch/1.0'}),timeout=20) as r:
   b=r.read(3_000_000)
  LOG.append({'url':u,'status':'ok','response_sha256':hashlib.sha256(b).hexdigest(),'collected_at':TODAY})
  return b
 except Exception as e:
  LOG.append({'url':u,'status':'error','error':str(e),'collected_at':TODAY});return b''
def base(text,surface,url,source_group,work,**kw):
 return dict(text=text,surface=surface,source_url=url,source_group=source_group,work_title=work,collected_at=TODAY,**kw)
def collect():
 rows=[];used=set()
 feeds=[('reviews',f'https://www.theguardian.com/{sec}/{sec}+tone/reviews/rss') for sec in ['film','tv-and-radio','books','games']]
 feeds += [('headlines',f'https://www.theguardian.com/{sec}/rss') for sec in ['film','tv-and-radio','books','games']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex: blobs=list(ex.map(fetch,[u for _,u in feeds]))
 for (surface,u),blob in zip(feeds,blobs):
  if not blob:continue
  doc=ET.fromstring(blob);rights=plain(doc.findtext('channel/copyright'))
  count=0
  for item in doc.findall('channel/item'):
   url=item.findtext('link');title=plain(item.findtext('title'))
   if url in used:continue
   if surface=='reviews' and ' review' not in title.lower():continue
   text=plain(item.findtext('description')).replace('Continue reading...','').strip() if surface=='reviews' else title
   text,cut=excerpt(text,35 if surface=='reviews' else 30)
   if len(text.split())<6:continue
   work=re.split(r' review\b',title,flags=re.I)[0] if surface=='reviews' else None
   rows.append(base(text,surface,url,url,work,source_title=title if surface=='headlines' else None,publisher='The Guardian',source_type='rss_review_standfirst' if surface=='reviews' else 'rss_headline',feed_url=u,published_at=item.findtext('pubDate'),author=item.findtext('{http://purl.org/dc/elements/1.1/}creator'),license_metadata=rights,license_url=None,redistribution_status='copyrighted_brief_excerpt_not_open_license',excerpt_truncated=cut,transformations='HTML removed; entities decoded; whitespace normalized; leading word cap',independence='New collection; training-corpus membership unknown',grouping_context=title))
   used.add(url);count+=1
   if count>=20:break
 # Manually selected real story IDs from public search results; no spoiler labels used.
 stories=[('31887748','Blade Runner'),('43517301','Severance'),('29011467','Dune'),('29533685','The Matrix'),('9367483','Game of Thrones'),('4262192','Breaking Bad'),('16260628','The Good Place'),('3735039','Inception')]
 urls=[f'https://hn.algolia.com/api/v1/search?tags=comment,story_{sid}&hitsPerPage=60' for sid,_ in stories]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:blobs=list(ex.map(fetch,urls))
 authors=set()
 for (sid,work),u,blob in zip(stories,urls,blobs):
  if not blob:continue
  hits=json.loads(blob).get('hits',[]);count=0
  for h in sorted(hits,key=lambda x:digest('review-pack-v1:'+x['objectID'])):
   text=plain(h.get('comment_text'));author=h.get('author')
   if len(text.split())<10 or author in authors:continue
   text,cut=excerpt(text,35);url='https://news.ycombinator.com/item?id='+h['objectID']
   rows.append(base(text,'comments',url,'https://news.ycombinator.com/item?id='+sid,work,source_title=h.get('story_title'),publisher='Hacker News',source_type='public_user_comment',feed_url=u,published_at=h.get('created_at'),author=author,license_metadata='No open text license verified; public API access is not a copyright license',license_url=None,redistribution_status='copyrighted_brief_excerpt_not_open_license',excerpt_truncated=cut,transformations='HTML removed; entities decoded; whitespace normalized; leading 35-word cap',independence='Not sourced from IMDb/TVTropes; historical public text may occur in pretraining',grouping_context=text))
   authors.add(author);count+=1
   if count>=12:break
 # Separate curator-transcribed source excerpts; not fabricated or automatically labeled.
 rows += json.loads((RAW/'subtitle-excerpts.json').read_text())
 (RAW/'collected-excerpts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
 (RAW/'collection-log.json').write_text(json.dumps(LOG,indent=2)+'\n')
 return rows

def build(rows):
 surface_map={"headlines":"headline","comments":"comment","reviews":"review","transcripts":"youtube_transcript"}
 for r in rows:
  r["surface"]=surface_map.get(r["surface"],r["surface"])
  if r["surface"] not in surface_map.values():raise ValueError("Unsupported surface: "+r["surface"])
 # Union connected source/work components before creating any annotation fields.
 parent=list(range(len(rows)))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 def union(a,b):parent[find(b)]=find(a)
 works={norm(r['work_title']) for r in rows if r.get('work_title')}
 owners={};seen={};clean=[]
 for r in rows:
  key=norm(r['text'])
  if key in seen:continue
  seen[key]=1;clean.append(r)
 rows=clean;parent=list(range(len(rows)))
 for i,r in enumerate(rows):
  context=' '+norm(r.get('grouping_context','')+' '+r['text'])+' '
  r['work_keys']=sorted({norm(r['work_title'])} if r.get('work_title') else set())
  r['work_keys']=sorted(set(r['work_keys'])|{w for w in works if len(w)>4 and ' '+w+' ' in context})
  keys=['source:'+r['source_group']]+['work:'+w for w in r['work_keys']]
  for k in keys:
   if k in owners:union(i,owners[k])
   else:owners[k]=i
 groups=collections.defaultdict(list)
 for i in range(len(rows)):groups[find(i)].append(i)
 for indices in groups.values():
  keys=sorted({r for i in indices for r in (['source:'+rows[i]['source_group']]+['work:'+w for w in rows[i]['work_keys']])})
  gid='group-'+digest('|'.join(keys))[:16]
  split='test' if int(digest('real-review-v1:'+gid)[:8],16)%10<3 else 'dev'
  # Two independent films deliberately occupy separate splits, before labels.
  if any(rows[i].get('work_title')=='Sintel' for i in indices):split='dev'
  if any(rows[i].get('work_title')=='Tears of Steel' for i in indices):split='test'
  for i in indices:rows[i].update(group_id=gid,split=split)
 for r in rows:
  r.update(id='real-'+digest(r['source_url']+'|'+r['text'])[:16],label=None,label_provenance='human',source_label=None,review_status='pending_human_review',reviewer_id=None,reviewed_at=None,review_rationale=None,context=None,group_review_status='pending_human_verification',eligible_for_evaluation=False)
  r['context']={'source_type':r['source_type'],'source_group':r['source_group'],'work_title':r.get('work_title'),'excerpt_truncated':r['excerpt_truncated'],'note':r.get('transcript_caveat','Brief excerpt only; use source for context. Label the displayed text, not unseen source content.')}
  r.pop('grouping_context',None)
 rows.sort(key=lambda r:digest('blind-order-v1:'+r['id']))
 for split in ['dev','test']:
  subset=[r for r in rows if r['split']==split]
  (OUT/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset))
 (OUT/'candidates.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 # Split/predictions hidden from annotators; URLs available for verifying context.
 columns=['id','text','surface','work_title','source_url','timestamp_start','timestamp_end','label','review_status','reviewer_id','reviewed_at','review_rationale','group_review_status']
 with (OUT/'blind-review.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 manifest={r['id']:{'group_id':r['group_id'],'split':r['split'],'text_sha256':digest(r['text']),'source_group':r['source_group'],'work_keys':r['work_keys']} for r in rows}
 (OUT/'split-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 assert len({r['id'] for r in rows})==len(rows)
 for field in ['group_id','source_group']:
  a={r[field] for r in rows if r['split']=='dev'};b={r[field] for r in rows if r['split']=='test'};assert not a&b
 a={w for r in rows if r['split']=='dev' for w in r['work_keys']};b={w for r in rows if r['split']=='test' for w in r['work_keys']};assert not a&b
 assert all(r['label'] is None and r['label_provenance']=='human' and not r['eligible_for_evaluation'] for r in rows)
 summary={'created_at':TODAY,'candidate_count':len(rows),'surface_counts':dict(collections.Counter(r['surface'] for r in rows)),'split_counts':dict(collections.Counter(r['split'] for r in rows)),'surface_by_split':{s:dict(collections.Counter(r['surface'] for r in rows if r['split']==s)) for s in ['dev','test']},'groups':len(groups),'reviewed':0,'labeled':0,'evaluated':0,'checks':{'unique_ids':True,'exact_normalized_text_deduplicated':True,'source_groups_disjoint':True,'declared_work_keys_disjoint':True,'labels_all_null':True},'grouping_caveat':'Unknown works, aliases and secondary references need human verification before evaluation. No claim of exhaustive semantic work disjointness.','manifest_sha256':digest((OUT/'split-manifest.json').read_text())}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--collect',action='store_true');args=a.parse_args()
 RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'blind-review.csv').exists():raise SystemExit('Existing pack protected; use a new version before rebuilding to avoid overwriting human work.')
 build(collect() if args.collect else json.loads((RAW/'collected-excerpts.json').read_text()))
