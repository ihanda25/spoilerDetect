"""Stream public CSV text and save aggregate evidence only; no model loading."""
import csv,io,json,hashlib,urllib.request,collections
from pathlib import Path
revision='9da61fd253654914524f7eba74b0fa6461c0f974'
result={'revision':revision,'accessed':'2026-10-02','splits':{}}
hashes={}; ids={}
for name,file in [('train','train.csv'),('validation','valid.csv')]:
 url=f'https://huggingface.co/datasets/bhavyagiri/imdb-spoiler/resolve/{revision}/data/{file}'
 labels=collections.Counter();texts=collections.Counter();indices=set();empty=0
 with urllib.request.urlopen(url,timeout=60) as r:
  reader=csv.DictReader(io.TextIOWrapper(r,encoding='utf-8'))
  for row in reader:
   labels[row['label']]+=1;indices.add(row.get('Unnamed: 0',row.get(''))) ;text=row['text'];empty+=not bool(text.strip())
   texts[hashlib.sha256(text.encode()).hexdigest()]+=1
 result['splits'][name]={'url':url,'rows':sum(labels.values()),'label_counts':dict(labels),'empty_texts':empty,'extra_exact_duplicate_rows':sum(v-1 for v in texts.values()),'unique_source_indices':len(indices)}
 hashes[name]=set(texts);ids[name]=indices
result['cross_split_exact_text_overlap']=len(hashes['train']&hashes['validation'])
result['cross_split_source_index_overlap']=len(ids['train']&ids['validation'])
Path(__file__).with_name('subset-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
