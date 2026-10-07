"""GPU-only notebook and small code launcher; upload raw data ZIP separately."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    report=ROOT/'reports/roberta-imdb-test'
    data=json.loads((report/'data-package.json').read_text())
    code_bytes=(ROOT/'scripts/evaluate_roberta_imdb.py').read_bytes()
    ast.parse(code_bytes)
    code_hash=hashlib.sha256(code_bytes).hexdigest()
    launcher=f'''import base64,hashlib,zipfile,subprocess,sys,json,shutil
from pathlib import Path
from IPython.display import display,HTML
ROOT=Path('/content/roberta-imdb-v1');ROOT.mkdir(exist_ok=True)
package=Path('/content/roberta-imdb-data.zip')
assert package.exists(), 'Upload roberta-imdb-data.zip using Files sidebar first'
assert hashlib.sha256(package.read_bytes()).hexdigest()=={data['sha256']!r}, 'Data ZIP hash mismatch'
with zipfile.ZipFile(package) as z:
    assert all((ROOT/item.filename).resolve().is_relative_to(ROOT.resolve()) for item in z.infolist())
    z.extractall(ROOT)
code=base64.b64decode({base64.b64encode(code_bytes).decode()!r})
assert hashlib.sha256(code).hexdigest()=={code_hash!r}
(ROOT/'evaluate_roberta_imdb.py').write_bytes(code)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','safetensors>=0.4.3','scikit-learn'])
import torch
assert torch.cuda.is_available(), 'GPU required; no local/CPU fallback'
print('GPU:',torch.cuda.get_device_name(0),flush=True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
command=[sys.executable,'-u',str(ROOT/'evaluate_roberta_imdb.py'),'--data-dir',str(ROOT),'--out',str(OUT),'--batch-size','16']
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in process.stdout:print(line,end='',flush=True)
status=process.wait()
archive=Path(shutil.make_archive('/content/roberta-imdb-results','zip',OUT))
for path in sorted(OUT.glob('*.json'))+[archive]:
    mime='application/zip' if path.suffix=='.zip' else 'application/json'
    encoded=base64.b64encode(path.read_bytes()).decode()
    display(HTML(f'<a download="{{path.name}}" href="data:{{mime}};base64,{{encoded}}">Download IMDb {{path.name}}</a>'))
assert status==0, 'Evaluation failed; download checkpoint files above for resume'
print('ROBERTA IMDB EVALUATION COMPLETE',flush=True)
'''
    ast.parse(launcher)
    (ROOT/'models/roberta-imdb-launch.py').write_text(launcher)
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'colab':{'name':'Spoiler_Roberta_IMDb_Test.ipynb','gpuType':'T4'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},cells=[
        dict(cell_type='markdown',metadata={},source=['# RoBERTa IMDb validation/test comparison\n','Upload `roberta-imdb-data.zip` through the Files sidebar. 63,836 validation and 64,405 test reviews, exact original MiniLM split/order. GPU inference only, no purchases or fine-tuning. Threshold chosen on validation and saved before test inference. External model training overlap unknown; scores are not proof of independence. Download result ZIP before runtime deletion.\n']),
        dict(cell_type='code',metadata={},source=launcher.splitlines(True),execution_count=None,outputs=[])])
    (ROOT/'notebooks/Spoiler_Roberta_IMDb_Test.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    (report/'code-manifest.json').write_text(json.dumps(dict(code_sha256=code_hash,data_package=data,launcher_sha256=hashlib.sha256(launcher.encode()).hexdigest()),indent=2)+'\n')
    print('Code SHA256:',code_hash)

if __name__=='__main__':main()
