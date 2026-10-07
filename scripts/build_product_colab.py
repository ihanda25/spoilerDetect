"""Build a self-contained Colab runner; never embeds copyrighted candidates."""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 cells=[]
 def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
 def code(s):
  ast.parse(s);cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
 md('''# Real-content spoiler evaluation — no training
Review the candidate pack with the local blind-review tool first. Upload the **complete reviewed JSONL** containing both dev and test, with verified work/source groups. This notebook refuses pending labels; uncertain reviewed records are excluded from binary metrics.

Select T4. Compare RoBERTa truncation vs overlapping windows on **development only**. Select the configuration and threshold with highest development spoiler F1, freeze the selection, then score test once. These brief excerpts are not a full-review or full-video benchmark. No production-quality claim follows from a small convenience sample.

Files under /content are temporary. Download the cache ZIP after a stage; upload it on a later session to resume with identical data/model/config. Changing library versions may invalidate caches. No Drive mount or paid resources are configured.
''')
 source=(ROOT/'scripts/product_eval.py').read_text()
 code('from pathlib import Path\nPath("/content/product_eval.py").write_text('+repr(source)+')\n')
 code('''import json, sys, subprocess, zipfile, shutil
from google.colab import files
subprocess.check_call([sys.executable,'-m','pip','install','-q','transformers==4.57.1','safetensors>=0.4.3'])
import torch
assert torch.cuda.is_available(), 'Connect a GPU runtime first'
print(torch.cuda.get_device_name(0))
sys.path.insert(0,'/content')
import product_eval as pe
''')
 md('Upload reviewed-spoiler-examples.jsonl, and optionally a prior product-eval-cache.zip. Original unreviewed candidates will be rejected. Keep all rows and split/group IDs unchanged during review.')
 code('''uploaded=files.upload()
jsonls=[name for name in uploaded if name.endswith('.jsonl')]
assert len(jsonls)==1, 'Upload exactly one complete reviewed JSONL'
DATA='/content/reviewed.jsonl'
Path(DATA).write_bytes(uploaded[jsonls[0]])
OUT=Path('/content/product-eval'); OUT.mkdir(exist_ok=True)
for name, payload in uploaded.items():
    if name.endswith('.zip'):
        import io
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for entry in z.infolist():
                target=(OUT/entry.filename).resolve()
                assert target.is_relative_to(OUT.resolve()), 'Unsafe archive path'
            z.extractall(OUT)
rows, data_hash=pe.load_data(DATA)
for split in ('dev','test'):
    pe.require_split_reviews(rows,split,'human')
assert all(r.get('group_review_status')=='verified' and r.get('eligible_for_evaluation') is True for r in rows), 'Verify source/work grouping before evaluation'
print('Human review gate passed:',len(rows),'rows. Dataset hash:',data_hash)
''')
 md('Run development predictions and freeze the winning configuration. These two candidates use the same pretrained weights; only text coverage differs. No model fitting occurs.')
 code('''MODEL='Zritze/imdb-spoiler-robertaOrigDatasetLR1'
REVISION='56fee120f8495ccfc5001e3fbd1478656d17001a'
def run(*args):
    subprocess.run([sys.executable,'/content/product_eval.py',*map(str,args)],check=True)
def predict(mode,split,selection=None):
    args=['predict','--data',DATA,'--model',MODEL,'--revision',REVISION,'--allow-download','--device','cuda','--require-cuda','--split',split,'--mode',mode,'--batch-size','8','--cache',str(OUT/f'{split}-{mode}.json')]
    if selection:args+=['--selection',str(selection)]
    run(*args)
for mode in ('truncate','window'):
    predict(mode,'dev')
SELECTION=OUT/'selection.json'
run('select','--data',DATA,'--cache',OUT/'dev-truncate.json','--cache',OUT/'dev-window.json','--selection',SELECTION,'--label-policy','human')
shutil.make_archive('/content/product-eval-cache','zip',OUT)
files.download('/content/product-eval-cache.zip')
''')
 md('Final held-out evaluation. Set RUN_LOCKED_TEST=True only after reviewing the frozen development selection. No threshold changes after viewing test results. Test caches and results cannot be silently overwritten with a different selection.')
 code('''RUN_LOCKED_TEST=False
if RUN_LOCKED_TEST:
    selection=pe.read_sealed(SELECTION)
    mode=selection['config']['mode']
    predict(mode,'test',SELECTION)
    run('evaluate','--data',DATA,'--cache',OUT/f'test-{mode}.json','--selection',SELECTION,'--output',OUT/'test-results.json')
    shutil.make_archive('/content/product-eval-cache','zip',OUT)
    files.download('/content/product-eval-cache.zip')
else:
    print('Test remains locked. Review the development selection before enabling this cell.')
''')
 nb=dict(nbformat=4,nbformat_minor=5,metadata={'colab':{'name':'Spoiler_Real_Content_Evaluation.ipynb','gpuType':'T4'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},cells=cells)
 path=ROOT/'notebooks/Spoiler_Real_Content_Evaluation.ipynb';path.write_text(json.dumps(nb,indent=1)+'\n');print(path)
if __name__=='__main__':main()
