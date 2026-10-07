"""Add label-blind episode retrieval to frozen sentence pilot; no training."""
import hashlib,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'data/processed/sentence-context-pilot'
OUT=ROOT/'data/processed/sentence-context-pilot-episodes'
REPORT=ROOT/'reports/context-training-data-audit'
STOP=set('a an and are as at be been but by for from had has have he her him his i if in is it its me my of on or our she so that the their them then there these they this to was we were what when which who will with you your'.split())
def words(s):return [w for w in re.findall(r"[a-z0-9]+",s.lower()) if w not in STOP]
def load(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def make_index(episodes):
 index=defaultdict(list)
 for episode in episodes:
  text=episode['plot'];sentences=re.split(r'(?<=[.!?])\s+',text);chunks=[];current=[];count=0
  for sentence in sentences:
   # Split unusually long sentences into manageable word blocks.
   tokens=sentence.split()
   for start in range(0,len(tokens),120):
    part=' '.join(tokens[start:start+120]);size=len(part.split())
    if count+size>140 and current:chunks.append(' '.join(current));current=[];count=0
    current.append(part);count+=size
  if current:chunks.append(' '.join(current))
  for i,text in enumerate(chunks):
   index[episode['work_page']].append({'episode':episode,'chunk_index':i,'text':text,'terms':Counter(words(text)),'word_count':len(words(text))})
 return index

def explicit_episodes(text,episodes):
 # Handles commas inside quotes, shortened titles, and a part number after a quote.
 quoted=re.findall(r'(?=["“]([^"”]+)(?:["”]|$))',text)
 matches=[]
 for ep in episodes:
  for q in quoted:
   qnorm=re.sub(r'[^a-z0-9]','',q.lower())
   if len(qnorm)<4:continue
   aliases=ep['title_aliases']+[ep['episode_title']]
   exact=any(re.sub(r'[^a-z0-9]','',a.lower())==qnorm for a in aliases)
   prefix=any(re.sub(r'[^a-z0-9]','',a.lower()).startswith(qnorm) for a in aliases)
   if not (exact or prefix):continue
   part=re.search(re.escape(q)+r'["”]\s*part\s*(\d+)',text,re.I)
   if part and not re.search(r'part\s*'+part.group(1)+r'\b',ep['episode_title'],re.I):continue
   if prefix and not exact and not part:
    candidates=[e for e in episodes if any(re.sub(r'[^a-z0-9]','',a.lower()).startswith(qnorm) for a in e['title_aliases']+[e['episode_title']])]
    if len(candidates)!=1:continue
   matches.append(ep['episode_title'])
 return set(matches)

def retrieve(text,page,index,episodes):
 docs=index[page];query=Counter(words(text));n=len(docs);avg=sum(d['word_count'] for d in docs)/max(1,n)
 df=Counter(t for d in docs for t in d['terms'])
 named=explicit_episodes(text,[e for e in episodes if e['work_page']==page])
 scored=[]
 for d in docs:
  if named and d['episode']['episode_title'] not in named:continue
  score=0
  for term in query:
   freq=d['terms'][term]
   if not freq:continue
   idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
   score+=idf*freq*2.5/(freq+1.5*(.25+.75*d['word_count']/max(avg,1)))
  scored.append((score,d))
 scored.sort(key=lambda x:(-x[0],x[1]['episode']['episode_title'],x[1]['chunk_index']))
 # Exact named episodes may have no lexical overlap; still attach as an explicit source.
 selected=[(score,d) for score,d in scored[:2] if score>0 or named]
 return selected,'explicit_episode_title' if named else 'within_work_bm25_candidate'

def add(row,index,episodes,tokenizer):
 row=dict(row);row['series_context_source_url']=row.get('context_source_url');selected,method=retrieve(row['text'],row['work_page'],index,episodes)
 sentence_ids=tokenizer.encode(row['text'],add_special_tokens=False)
 used_sentence=sentence_ids[:256];sentence_text=tokenizer.decode(used_sentence,skip_special_tokens=True) if len(sentence_ids)>256 else row['text']
 budget=512-tokenizer.num_special_tokens_to_add(pair=True)-len(used_sentence)
 overview_ids=tokenizer.encode('Series background: '+row['plot_context'],add_special_tokens=False)[:96]
 overview=tokenizer.decode(overview_ids,skip_special_tokens=True)
 parts=[overview];provenance=[]
 remaining=budget-len(overview_ids)-2
 passage_count=0
 for score,d in selected:
  ep=d['episode'];text=f"Episode: {ep['episode_title']}. {d['text']}";ids=tokenizer.encode(text,add_special_tokens=False)
  # Reserve room for a second retrieved passage when present.
  allocation=remaining if len(selected)==1 or passage_count else max(1,remaining//2)
  retained=ids[:allocation]
  if not retained:continue
  parts.append(tokenizer.decode(retained,skip_special_tokens=True));remaining-=len(retained)+2;passage_count+=1
  provenance.append({'episode_title':ep['episode_title'],'episode_url':ep['episode_url'],'source_url':ep['plot_source_url'],'chunk_index':d['chunk_index'],'chunk_sha256':sha(d['text']),'retrieval_score':round(score,6),'retained_tokens':len(retained),'chunk_truncated':len(retained)<len(ids)})
 context='\n'.join(parts)
 encoded=tokenizer(sentence_text,context,truncation='only_second',max_length=512,add_special_tokens=True)
 assert len(encoded['input_ids'])<=512
 row.update(series_plot_context=row['plot_context'],plot_context=context,episode_selection_method=method,retrieved_passages=provenance,encoded_length=len(encoded['input_ids']),sentence_tokens=len(sentence_ids),sentence_truncated=len(sentence_ids)>256,sentence_input=sentence_text,input_ids=encoded['input_ids'],attention_mask=encoded['attention_mask'],context_scope='Series background plus retrieved episode passages; relevance/event support unverified',context_source_url=None,context_source_urls=[r['source_url'] for r in provenance],episode_context_review='pending',training_allowed=False)
 if 'context_adequacy' in row:row['series_context_adequacy']=row.pop('context_adequacy')
 return row

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 episodes=load(ROOT/'data/raw/episode-context/episodes.jsonl');index=make_index(episodes)
 tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/context-pilot-tokenizer',local_files_only=True)
 outputs={}
 for name in ['candidate-pairs','train','validation','quality-review']:
  rows=load(INPUT/(name+'.jsonl'));outputs[name]=[add(r,index,episodes,tokenizer) for r in rows]
  assert [r['id'] for r in rows]==[r['id'] for r in outputs[name]]
  assert [r['label'] for r in rows]==[r['label'] for r in outputs[name]]
  (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs[name]))
 allrows=outputs['candidate-pairs'];pilot=outputs['train']+outputs['validation']
 assert {r['work_page'] for r in outputs['train']}.isdisjoint({r['work_page'] for r in outputs['validation']})
 source_sets={s:{url.split('#')[0] for r in outputs[s] for url in r['context_source_urls']} for s in ['train','validation']}
 assert source_sets['train'].isdisjoint(source_sets['validation'])
 manifest={'date':'2026-10-06','status':'episode_context_added_pending_quality_review','training_started':False,'training_ready':False,'retrieval':'Within-work BM25 over episode-plot chunks, explicit quoted episode titles take priority; top 2 passages plus up to 96 series-background tokens; labels excluded from query/scoring','max_length':512,'max_sentence_tokens':256,'max_series_background_tokens':96,'special_pair_tokens':tokenizer.num_special_tokens_to_add(pair=True),'episodes':len(episodes),'chunks':sum(len(v) for v in index.values()),'candidate_rows':len(allrows),'candidate_with_episode_context':sum(bool(r['retrieved_passages']) for r in allrows),'pilot_rows':len(pilot),'pilot_with_episode_context':sum(bool(r['retrieved_passages']) for r in pilot),'pilot_selection_methods':dict(Counter(r['episode_selection_method'] for r in pilot)),'pilot_max_encoded_length':max(r['encoded_length'] for r in pilot),'pilot_sentence_truncations':sum(r['sentence_truncated'] for r in pilot),'pilot_context_truncations':sum(any(p['chunk_truncated'] for p in r['retrieved_passages']) for r in pilot),'frozen_labels_and_ids_preserved':True,'train_validation_work_and_source_disjoint':True,'episode_corpus_sha256':sha((ROOT/'data/raw/episode-context/episodes.jsonl').read_text()),'output_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')},'limitations':['Lexical match is not proof of correct event or episode','Original markup labels and fragmented sentences require review','Some rows discuss films/remakes/comics not in episode corpus','All seasons indexed; not personalized to viewer progress','Pilot was selected after dataset exploration; development only','Source rights/caveats from prior audit still apply','No claim of production readiness']}
 (REPORT/'episode-pilot-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
