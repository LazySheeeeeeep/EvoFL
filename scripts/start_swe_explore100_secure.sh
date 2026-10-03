#!/usr/bin/env bash
set -euo pipefail
printf 'Waiting for non-echoed credential input.\n'
IFS= read -r -s DEEPSEEK_API_KEY
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
python3 - <<'PY'
import os
import subprocess
from pathlib import Path

root = Path('/mnt/d/projects/EvoluteFL')
out = root / 'runs/swe_explore_v5_stratified100_deepseek_20260910'
pidfile = out / 'evaluation.pid'
if pidfile.exists():
    pid = int(pidfile.read_text())
    command = Path(f'/proc/{pid}/cmdline')
    if command.exists() and b'run_swe_explore_v5_stratified.py' in command.read_bytes():
        raise RuntimeError('Evaluation is already running')
env = dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1')
with (out / 'evaluation.log').open('a') as log:
    process = subprocess.Popen(
        ['python3', '-u', 'scripts/run_swe_explore_v5_stratified.py'],
        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pidfile.write_text(str(process.pid) + '\n')
print(f'Evaluation started, PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
