"""Download only model data, pinned to the revision used for this experiment."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '44acabbec0ef496f6dbc93adadea57f376b7c0ec'
MODEL = 'microsoft/MiniLM-L12-H384-uncased'
FILES = ['README.md', 'config.json', 'pytorch_model.bin', 'special_tokens_map.json',
         'tokenizer_config.json', 'vocab.txt']

if __name__ == '__main__':
    destination = ROOT / 'models/base'
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        path = destination / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.part')
            subprocess.run(['curl', '-fSL', '--retry', '2', '--max-time', '300',
                f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', '-o', str(partial)], check=True)
            partial.replace(path)
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    source = destination / 'source.json'
    if not source.exists():
        source.write_text(json.dumps({'id': MODEL, 'sha': REVISION}) + '\n')
    (ROOT / 'reports/base-model-files.json').write_text(json.dumps(
        {'model': MODEL, 'revision': REVISION, 'sha256': hashes}, indent=2) + '\n')
    print('Base checkpoint ready:', destination)
