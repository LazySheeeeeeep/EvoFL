#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'DeepSeek credential (process only): '
    IFS= read -r -s DEEPSEEK_API_KEY
    printf '\n'
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
fi
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
/home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
output = root / 'runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922'
output.mkdir(parents=True, exist_ok=True)
log = (output / 'experiment.log').open('a')
process = subprocess.Popen(
    [sys.executable, '-u', 'scripts/run_rq1_expanded.py', '--phase', 'full', '--workers', '4'],
    cwd=root,
    env=dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1'),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(output / 'experiment.pid').write_text(str(process.pid) + '\n')
print(f'Expanded RQ1 worker started: PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
