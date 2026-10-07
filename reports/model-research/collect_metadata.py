"""Fetch only public JSON metadata/configs and model cards; never model artifacts."""
import concurrent.futures, json, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IDS=['bhavyagiri/roberta-base-finetuned-imdb-spoilers','eesuan/imdb-spoiler-distilbert','Zritze/imdb-spoiler-bertOrigDataset','lhx1101/spoiler_detection_model','tcjordan3/bert-base-spoiler-detection','leoole/spoiler-detector','toosharm/spoiler-detector']
def fetch(url):
    try:
        with urllib.request.urlopen(url,timeout=30) as r:
            text=r.read(2_000_000).decode()
        try:return json.loads(text)
        except ValueError:return text
    except Exception as e:return {'error':str(e)}
extra=fetch('https://huggingface.co/api/models?author=Zritze&search=spoiler')
if isinstance(extra,list): IDS.extend(x['id'] for x in extra if 'roberta' in x['id'])
def one(repo):
    meta=fetch('https://huggingface.co/api/models/'+repo+'?blobs=true')
    sha=meta.get('sha','main'); base=f'https://huggingface.co/{repo}/resolve/{sha}/'
    out={'repo':repo,'metadata_url':'https://huggingface.co/api/models/'+repo,'metadata':meta}
    for name in ['config.json','tokenizer_config.json','README.md']:
        out[name]={'url':base+name,'content':fetch(base+name)}
    return out
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex: rows=list(ex.map(one,IDS))
(ROOT/'metadata-snapshot.json').write_text(json.dumps({'accessed':'2026-10-02','models':rows},indent=2))
for x in rows:
    m=x['metadata'];print('\nMODEL',x['repo'],'SHA',m.get('sha'),'LICENSE',m.get('cardData',{}).get('license'))
    print('FILES',[(f['rfilename'],f.get('size')) for f in m.get('siblings',[]) if '/' not in f['rfilename']])
    print('CONFIG',x['config.json']['content']);print('TOKENIZER',x['tokenizer_config.json']['content']);print('CARD',x['README.md']['content'])
