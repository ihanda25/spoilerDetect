import hashlib, zipfile, subprocess, sys, json, shutil, base64
from pathlib import Path
from IPython.display import display, HTML
package=Path('/content/tv-training-inputs.zip')
assert package.exists(), 'Upload tv-training-inputs.zip through Files sidebar'
assert hashlib.sha256(package.read_bytes()).hexdigest()=='595515efe4a164d6a8ab28ae3838c37bb2db00f58cec13ee01a09f0a8459c745', 'Wrong input package'
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
    print('Saved model archive:',str(archive),flush=True)
summary={a:json.loads((ROOT/'results'/a/'report.json').read_text()) for a in ['sentence','context']}
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2))
from google.colab import files
small=ROOT/'download-results';small.mkdir(exist_ok=True)
shutil.copy(ROOT/'comparison.json',small/'comparison.json')
for arm in ['sentence','context']:
    for name in ['report.json','predictions.jsonl','run-config.json','before.json']:
        shutil.copy(ROOT/'results'/arm/name,small/(arm+'-'+name))
results=Path(shutil.make_archive('/content/tv-training-results','zip',small))
data=base64.b64encode(results.read_bytes()).decode()
display(HTML(f'<a download="tv-training-results.zip" href="data:application/zip;base64,{data}">Download TV comparison results</a>'))
print('TV COMPARISON COMPLETE; ONE EPOCH PER ARM; NO AUTOMATIC SECOND EPOCH',flush=True)
