"""Build a small reproducible context diagnostic package, without model weights."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    files={
        'compare_three_models.py':ROOT/'scripts/compare_three_models.py',
        'compare_context_models.py':ROOT/'scripts/compare_context_models.py',
        'blind.jsonl':ROOT/'reports/eval-refresh/blind-inputs-v2.jsonl',
        'metadata.json':ROOT/'reports/eval-refresh/input-metadata-v2.json',
        'contexts.json':ROOT/'reports/context-comparison/contexts-v1.json',
        'annotations.jsonl':ROOT/'reports/eval-refresh/reviewed-dev-v2.jsonl',
    }
    package=ROOT/'models/context-comparison-inputs.zip'
    package.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in files.items():
            z.writestr(name,path.read_bytes())
    payload=package.read_bytes()
    digest=hashlib.sha256(payload).hexdigest()
    # Embed only authorized project code and public development excerpts, no credentials.
    bootstrap=f'''import base64, hashlib, io, zipfile, subprocess, sys, json, shutil
from pathlib import Path
from IPython.display import HTML, display
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
ROOT=Path('/content/spoiler-context-v1'); ROOT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    assert all((ROOT/item.filename).resolve().is_relative_to(ROOT.resolve()) for item in z.infolist())
    z.extractall(ROOT)
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','safetensors>=0.4.3'])
import torch
assert torch.cuda.is_available(), 'GPU required; no CPU fallback'
print('GPU:',torch.cuda.get_device_name(0),flush=True)
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
command=[sys.executable,'-u',str(ROOT/'compare_context_models.py'),'--data',str(ROOT/'blind.jsonl'),'--metadata',str(ROOT/'metadata.json'),'--contexts',str(ROOT/'contexts.json'),'--annotations',str(ROOT/'annotations.jsonl'),'--out',str(OUT)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
for line in process.stdout: print(line,end='',flush=True)
status=process.wait()
archive=shutil.make_archive('/content/spoiler-context-results','zip',OUT)
for path in sorted(OUT.glob('*.json'))+[Path(archive)]:
    mime='application/zip' if path.suffix=='.zip' else 'application/json'
    data=base64.b64encode(path.read_bytes()).decode()
    display(HTML(f'<a download="{{path.name}}" href="data:{{mime}};base64,{{data}}">Download context {{path.name}}</a>'))
assert status==0, 'Inference failed; caches above permit resuming'
print('CONTEXT COMPARISON COMPLETE',flush=True)
'''
    ast.parse(bootstrap)
    report=ROOT/'reports/context-comparison'
    manifest=dict(package_sha256=digest,bytes=len(payload),files={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()})
    (report/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'models/context-colab-launch.py').write_text(bootstrap)
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'colab':{'name':'Spoiler_Context_Comparison.ipynb','gpuType':'T4'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},cells=[
        dict(cell_type='markdown',metadata={},source=['# Supplied-context development diagnostic\n','207 AI-reviewed dev excerpts; fixed text-only RoBERTa vs title-only Qwen vs title+plot Qwen. 94 have summaries; others reuse title-only predictions. No training, API, purchases or test tuning. Existing Colab GPU units only. Download results before deleting runtime. Context challenge shares source plots; report separately from real content.\n']),
        dict(cell_type='code',metadata={},source=bootstrap.splitlines(True),outputs=[],execution_count=None)])
    (ROOT/'notebooks/Spoiler_Context_Comparison.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
