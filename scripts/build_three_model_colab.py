"""Package the owned epoch-2 model and blind development inputs for Colab."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    import torch
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    stage=ROOT/'models/three-model-colab';stage.mkdir(exist_ok=True)
    destination=stage/'minilm-epoch2'
    checkpoint=torch.load(ROOT/'models/minilm-full/checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['completed_epochs']==2,'Expected review-only epoch2'
    config=AutoConfig.from_pretrained(ROOT/'models/base',local_files_only=True,num_labels=2)
    config.id2label={0:'NON-SPOILER',1:'SPOILER'};config.label2id={'NON-SPOILER':0,'SPOILER':1}
    model=AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.save_pretrained(destination,safe_serialization=True)
    AutoTokenizer.from_pretrained(ROOT/'models/base',local_files_only=True).save_pretrained(destination)
    rows=[json.loads(s) for s in (ROOT/'reports/real-review-pack/annotated-human-and-ai.jsonl').read_text().splitlines()]
    dev=[r for r in rows if r['split']=='dev']
    blind=[{k:r[k] for k in ('id','text','surface','split')} for r in dev]
    annotations=[{k:r.get(k) for k in ('id','label','label_provenance','review_status','reviewer_id','reviewed_at','decision_captured_at')} for r in dev]
    (stage/'dev-blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in blind))
    (stage/'annotations.json').write_text(json.dumps(annotations,indent=2))
    shutil.copy2(ROOT/'scripts/compare_three_models.py',stage/'compare_three_models.py')
    package=ROOT/'models/three-model-comparison-inputs.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(stage))
    manifest=dict(dev_rows=len(dev),test_rows_excluded=len(rows)-len(dev),checkpoint_completed_epochs=2,
        package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),bytes=package.stat().st_size,
        annotation_counts={kind:sum(r['label_provenance']==kind for r in dev) for kind in ('human','ai')})
    out=ROOT/'reports/three-model-comparison';out.mkdir(exist_ok=True)
    (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cells=[]
    def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
    def code(s):
        ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None))
    md('# Three-model spoiler comparison\nGPU inference only: owned MiniLM review-only epoch2, pinned Zritze RoBERTa, Qwen2.5-3B-Instruct without RAG. Text-only inputs, 147 development excerpts. Human/AI labels reported separately; incomplete human metadata and unverified grouping mean exploratory agreement only. Held-out test is excluded. No threshold fitting. Upload the owned inputs ZIP when asked. Files under /content are ephemeral: download caches after each stage. No API keys, Drive mounts, training, or paid provisioning.\n')
    code("import subprocess,sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','safetensors>=0.4.3'])\nimport torch\nassert torch.cuda.is_available(), 'Select a T4 GPU runtime'\nprint(torch.cuda.get_device_name(0))\n")
    code("from google.colab import files\nfrom pathlib import Path\nimport hashlib,io,zipfile\nname='three-model-comparison-inputs.zip'\npackage=Path('/content')/name\nif not package.exists():\n    uploaded=files.upload()\n    assert name in uploaded, 'Upload '+name\npayload=package.read_bytes()\nassert hashlib.sha256(payload).hexdigest()=="+repr(manifest['package_sha256'])+"\nROOT=Path('/content/three-model-comparison');ROOT.mkdir(exist_ok=True)\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:\n    assert all((ROOT/r.filename).resolve().is_relative_to(ROOT.resolve()) for r in z.infolist())\n    z.extractall(ROOT)\nOUT=ROOT/'results';OUT.mkdir(exist_ok=True)\n")
    code("import shutil\ndef run_model(name):\n    process=subprocess.Popen([sys.executable,str(ROOT/'compare_three_models.py'),'predict','--data',str(ROOT/'dev-blind.jsonl'),'--out',str(OUT),'--model',name,'--minilm',str(ROOT/'minilm-epoch2')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)\n    for line in process.stdout: print(line,end='',flush=True)\n    assert process.wait()==0, 'Model inference failed; inspect output'\ndef archive():\n    shutil.make_archive('/content/three-model-results','zip',OUT)\n    files.download('/content/three-model-results.zip')\n")
    for name in ('minilm','roberta','llm'):
        md('Run '+name+' (no training). The LLM download is approximately 6 GB; it runs on the T4, with one model loaded at a time.')
        code('run_model('+repr(name)+')\narchive()\n')
    code("completed=subprocess.run([sys.executable,str(ROOT/'compare_three_models.py'),'report','--data',str(ROOT/'dev-blind.jsonl'),'--annotations',str(ROOT/'annotations.json'),'--out',str(OUT)],capture_output=True,text=True)\nprint(completed.stdout)\nif completed.returncode: print(completed.stderr)\ncompleted.check_returncode()\narchive()\n")
    nb=dict(nbformat=4,nbformat_minor=5,metadata={'colab':{'name':'Spoiler_Three_Model_Comparison.ipynb','gpuType':'T4'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},cells=cells)
    (ROOT/'notebooks/Spoiler_Three_Model_Comparison.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    print(json.dumps(manifest,indent=2))
    print(package)
if __name__=='__main__':main()
