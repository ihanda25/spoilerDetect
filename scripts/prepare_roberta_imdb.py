"""Export original raw text for exact existing MiniLM validation/test row order."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from prepare_training import records, split_for, ROOT

def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    dest=ROOT/'data/roberta-imdb-evaluation';dest.mkdir(exist_ok=True)
    arrays={split:{key:np.load(ROOT/f'data/full/{split}.{key}.npy') for key in ('source_lines','labels','movie_ids')} for split in ('validation','test')}
    lookup={}
    for split,values in arrays.items():
        assert len(values['source_lines'])==len(values['labels'])==len(values['movie_ids'])
        assert np.all(np.diff(values['source_lines'])>0)
        for i,line in enumerate(values['source_lines']):
            assert int(line) not in lookup
            lookup[int(line)]=(split,i)
    assert set(arrays['validation']['movie_ids']).isdisjoint(set(arrays['test']['movie_ids']))
    train_movies=set(np.load(ROOT/'data/full/train.movie_ids.npy'))
    assert all(train_movies.isdisjoint(set(v['movie_ids'])) for v in arrays.values())
    counts={s:0 for s in arrays}
    streams={s:(dest/(s+'.jsonl.tmp')).open('w') for s in arrays}
    try:
        for line,row in enumerate(records(),1):
            if line not in lookup:continue
            split,i=lookup[line];values=arrays[split]
            assert i==counts[split]
            assert row['movie_id']==str(values['movie_ids'][i])
            assert int(row['is_spoiler'])==int(values['labels'][i])
            assert split_for(row['movie_id'])==split
            assert row['review_text'].strip()
            # Preserve raw source whitespace/casing, as MiniLM's full tokenizer did.
            record=dict(source_line=line,movie_id=row['movie_id'],text=row['review_text'],label=int(row['is_spoiler']))
            streams[split].write(json.dumps(record,ensure_ascii=False)+'\n')
            counts[split]+=1
            if sum(counts.values())%20000==0:print('Exported',sum(counts.values()),'reviews',flush=True)
    finally:
        for f in streams.values():f.close()
    for split in arrays:
        assert counts[split]==len(arrays[split]['labels'])
        tmp=dest/(split+'.jsonl.tmp');path=dest/(split+'.jsonl')
        if path.exists():
            assert file_sha(path)==file_sha(tmp),'Refuse changed existing export'
            tmp.unlink()
        else:tmp.replace(path)
    for src,name in [('reports/selected-test/results.json','minilm-test-results.json'),('reports/threshold-tuning/results.json','minilm-validation-results.json')]:shutil.copy2(ROOT/src,dest/name)
    manifest=dict(version=1,dataset='Rishabh Misra IMDb Spoiler Dataset v1',
        dataset_zip_sha256=file_sha(ROOT/'data/raw/imdb/dataset.zip'),
        method='Exact raw reviews selected and ordered by saved MiniLM source_lines/movie_ids/labels; no sampling, added rows, or reused token IDs.',
        split_method='SHA256(42:movie_id) mod100, movie-disjoint; original full deduplication exclusions retained.',
        license_source='https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset',
        license_metadata='Source-reported CC BY 4.0; attribution Rishabh Misra, IMDb Spoiler Dataset, 2019, DOI10.13140/RG.2.2.11584.15362.',
        overlap_caveat='Zritze training corpus/split undisclosed; held out for our MiniLM, not established held out for the external RoBERTa.',splits={})
    for split,v in arrays.items():
        path=dest/(split+'.jsonl')
        manifest['splits'][split]=dict(file=path.name,rows=counts[split],spoilers=int(v['labels'].sum()),movies=len(set(v['movie_ids'])),sha256=file_sha(path),source_lines_sha256=file_sha(ROOT/f'data/full/{split}.source_lines.npy'),labels_sha256=file_sha(ROOT/f'data/full/{split}.labels.npy'),movie_ids_sha256=file_sha(ROOT/f'data/full/{split}.movie_ids.npy'))
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out=ROOT/'reports/roberta-imdb-test';out.mkdir(exist_ok=True)
    shutil.copy2(dest/'manifest.json',out/'input-manifest.json')
    package=ROOT/'models/roberta-imdb-data.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(dest.glob('*.json*')):z.write(path,path.name)
    (out/'data-package.json').write_text(json.dumps(dict(sha256=file_sha(package),bytes=package.stat().st_size),indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    print('ZIP bytes',package.stat().st_size,'sha256',file_sha(package))

if __name__=='__main__':main()
