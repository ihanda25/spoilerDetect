"""Download the author's IMDb spoiler dataset, version 1 (standard library only)."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/raw/imdb'
URL = 'https://www.kaggle.com/api/v1/datasets/download/rmisra/imdb-spoiler-dataset?datasetVersionNumber=1'


def download(url, path):
    partial = path.with_suffix(path.suffix + '.part')
    subprocess.run(['curl', '-fSL', '--retry', '2', '--max-time', '600', url,
                    '-o', str(partial)], check=True)
    partial.replace(path)


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    if not (DEST / 'dataset.zip').exists():
        download(URL, DEST / 'dataset.zip')
    metadata = DEST / 'kaggle-search-metadata.json'
    download('https://www.kaggle.com/api/v1/datasets/list?search=imdb-spoiler-dataset', metadata)
    matches = json.loads(metadata.read_text())
    source = next(item for item in matches if item['ref'] == 'rmisra/imdb-spoiler-dataset')
    (DEST / 'source-metadata.json').write_text(json.dumps(source, indent=2) + '\n')
    print('Downloaded to', DEST)
