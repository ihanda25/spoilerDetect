"""Refresh the saved Colab launcher from audited code and checkpoint backups."""
import ast,base64,hashlib,io,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
launch=root/'models/gemini-context-launch.py'
source=launch.read_text()
payload_node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
old=base64.b64decode(ast.literal_eval(payload_node.value.args[0]))
with zipfile.ZipFile(io.BytesIO(old)) as z:
 files={n:z.read(n) for n in z.namelist() if not n.startswith('checkpoint-backups/')}
for name in ('compare_gemini_context.py','compare_three_models.py','compare_context_models.py'):
 files[name]=(root/'scripts'/name).read_bytes()
for arm in ('gemini_title','gemini_context'):
 files['checkpoint-backups/'+arm+'.json']=(root/'reports/gemini-context-comparison'/f'{arm}.json').read_bytes()
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in files.items():z.writestr(name,data)
payload=buf.getvalue();digest=hashlib.sha256(payload).hexdigest()
lines=source.splitlines()
lines=["payload=base64.b64decode("+repr(base64.b64encode(payload).decode())+")" if l.startswith('payload=') else "assert hashlib.sha256(payload).hexdigest()=="+repr(digest) if l.startswith('assert hashlib.sha256(payload)') else l for l in lines]
source='\n'.join(lines)+'\n'
source=source.replace("'google-genai'", "'google-genai==2.12.1'")
source=source.replace("    complete=run(ROOT,client)","    complete=run(ROOT,client,max_requests=5)")
marker="sys.path.insert(0,str(ROOT))"
restore="""# Restore backups only when the runtime has no checkpoint; never overwrite newer answers.
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
for backup in (ROOT/'checkpoint-backups').glob('*.json'):
    if not (OUT/backup.name).exists(): shutil.copyfile(backup,OUT/backup.name)
"""
if restore not in source:source=source.replace(marker,restore+marker)
launch.write_text(source)
notebook=root/'notebooks/Spoiler_Gemini_Context.ipynb'
nb=json.loads(notebook.read_text());codecells=[c for c in nb['cells'] if c['cell_type']=='code'];assert len(codecells)==1
codecells[0]['source']=source.splitlines(keepends=True);codecells[0]['outputs']=[];codecells[0]['execution_count']=None
notebook.write_text(json.dumps(nb,indent=1)+'\n')
manifest={'package_sha256':digest,'bytes':len(payload),'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()},'max_requests':5,'request_gap_seconds':30,'sdk':'2.12.1'}
(root/'reports/gemini-context-comparison/package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built launcher with checkpoint backups:',len(payload),'bytes')
