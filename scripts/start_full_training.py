"""Start a durable local training process; do not start a duplicate job."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/full'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    pidfile = REPORT / 'process.json'
    if pidfile.exists():
        pid = json.loads(pidfile.read_text())['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit(f'PID {pid} still exists. Inspect it before starting another training run.')
    if (ROOT/'models/minilm-full/last.pt').exists() and not args.resume:
        raise SystemExit('A checkpoint already exists; use --resume to continue it.')
    command = [sys.executable, '-u', str(ROOT/'scripts/full_training.py')]
    if args.resume:
        command.append('--resume')
    with (REPORT/'training.log').open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    pidfile.write_text(json.dumps({'pid':process.pid,'command':command}, indent=2)+'\n')
    print('Started local training PID', process.pid)
    print('Log:', REPORT/'training.log')
