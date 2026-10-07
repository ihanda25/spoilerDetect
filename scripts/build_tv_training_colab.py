"""Package the frozen mapped TV data for an exploratory hosted-GPU comparison."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'data/processed/sentence-context-pilot-episodes/candidate-pairs.jsonl'
rows=[json.loads(l) for l in source.read_text().splitlines()]
out=ROOT/'data/processed/tv-training-experiment';out.mkdir(parents=True,exist_ok=True)
manifest=dict(date='2026-10-06',purpose='User-authorized exploratory one-epoch TV adaptation, sentence vs episode-context',training_quality='Unverified original corpus labels and retrieved contexts; not production gold',model='Zritze/imdb-spoiler-robertaOrigDatasetLR1',revision='56fee120f8495ccfc5001e3fbd1478656d17001a',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),training_rows=744,validation_rows=198,epochs_per_arm=1,arms=['sentence','context'],test_excluded=True,hashes={})
for split in ['train','validation']:
 selected=[r for r in rows if r['source_split']==split]
 assert len(selected)==manifest['training_rows' if split=='train' else 'validation_rows']
 path=out/(split+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
 manifest['hashes'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(out/'experiment.json').write_text(json.dumps(manifest,indent=2)+'\n')
from train_tv_roberta import load_pack
load_pack(out)
package=out/'tv-training-inputs.zip'
with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
 for path in [out/'train.jsonl',out/'validation.jsonl',out/'experiment.json',ROOT/'scripts/train_tv_roberta.py']:z.write(path,path.name)
sha=hashlib.sha256(package.read_bytes()).hexdigest()
launch=f'''import hashlib, zipfile, subprocess, sys, json, shutil, base64
from pathlib import Path
from IPython.display import display, HTML
package=Path('/content/tv-training-inputs.zip')
assert package.exists(), 'Upload tv-training-inputs.zip through Files sidebar'
assert hashlib.sha256(package.read_bytes()).hexdigest()=={sha!r}, 'Wrong input package'
ROOT=Path('/content/tv-roberta-20261006'); ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(package) as z:
    assert all((ROOT/i.filename).resolve().is_relative_to(ROOT.resolve()) for i in z.infolist())
    z.extractall(ROOT)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','accelerate>=1.1,<2','scikit-learn','safetensors'])
import torch
assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
print('GPU:',torch.cuda.get_device_name(0),flush=True)
for arm in ['sentence','context']:
    status=subprocess.call([sys.executable,str(ROOT/'train_tv_roberta.py'),'--root',str(ROOT),'--arm',arm,'--allow-exploratory-corpus-labels'])
    assert status==0, 'Training failed; checkpoint retained in runtime'
    report=json.loads((ROOT/'results'/arm/'report.json').read_text())
    print(arm, json.dumps(report['after'],indent=2),flush=True)
    archive=Path(shutil.make_archive('/content/tv-'+arm+'-epoch1','zip',ROOT/'results'/arm/'model'))
    display(HTML(f'<a download="{{archive.name}}" href="/files{{archive}}">Download {{arm}} epoch1 model</a>'))
summary={{a:json.loads((ROOT/'results'/a/'report.json').read_text()) for a in ['sentence','context']}}
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2))
from google.colab import files
small=ROOT/'download-results';small.mkdir(exist_ok=True)
shutil.copy(ROOT/'comparison.json',small/'comparison.json')
for arm in ['sentence','context']:
    for name in ['report.json','predictions.jsonl','run-config.json','before.json']:
        shutil.copy(ROOT/'results'/arm/name,small/(arm+'-'+name))
results=Path(shutil.make_archive('/content/tv-training-results','zip',small))
data=base64.b64encode(results.read_bytes()).decode()
display(HTML(f'<a download="tv-training-results.zip" href="data:application/zip;base64,{{data}}">Download TV comparison results</a>'))
print('TV COMPARISON COMPLETE; ONE EPOCH PER ARM; NO AUTOMATIC SECOND EPOCH',flush=True)
'''
# Model files use files.download separately: HTML /files links are not portable across Colab output frames.
launch=launch.replace("display(HTML(f'<a download=\"{archive.name}\" href=\"/files{archive}\">Download {arm} epoch1 model</a>'))", "print('Saved model archive:',str(archive),flush=True)")
report=ROOT/'reports/tv-training';report.mkdir(exist_ok=True)
(report/'launch.py').write_text(launch)
(report/'experiment.json').write_text(json.dumps(dict(manifest,package_sha256=sha),indent=2)+'\n')
nb=dict(nbformat=4,nbformat_minor=5,metadata={'colab':{'name':'Spoiler_TV_Roberta_Pilot.ipynb','gpuType':'T4'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},cells=[dict(cell_type='markdown',metadata={},source=['# TV RoBERTa: sentence vs episode context\n','744 training / 198 validation examples, seven work-disjoint TV shows. Unverified corpus labels and context; exploratory only. One epoch per arm from the same pinned spoiler checkpoint, class-weighted loss, fixed cutoff .5. No test tuning or purchases. Existing Colab GPU units only. Upload tv-training-inputs.zip. Save model archives before runtime deletion.\n']),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=launch.splitlines(keepends=True)),dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=["from google.colab import files\n","files.download('/content/tv-sentence-epoch1.zip')\n","files.download('/content/tv-context-epoch1.zip')\n"])])
(ROOT/'notebooks/Spoiler_TV_Roberta_Pilot.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
print('Package',package,'bytes',package.stat().st_size,'sha256',sha)
