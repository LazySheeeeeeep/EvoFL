#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
if [ -z "${QWEN_API_KEY:-}" ]; then
    printf 'Qwen API key (process only): '
    IFS= read -r -s QWEN_API_KEY
    printf '\n'
    QWEN_API_KEY=${QWEN_API_KEY%$'\r'}
fi
test -n "$QWEN_API_KEY"
export QWEN_API_KEY
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
/home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys

root = Path.cwd()
output = root / 'runs/rq1_main_qwen3_8_flash_eval500_20260929'
output.mkdir(parents=True, exist_ok=True)
log = (output / 'experiment.log').open('a')
process = subprocess.Popen(
    [sys.executable, '-u', 'scripts/run_rq1_qwen_main_eval.py',
     '--limit', '500', '--workers', '4', '--output', str(output),
     '--resume-index', '163'],
    cwd=root,
    env=dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1'),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(output / 'experiment.pid').write_text(str(process.pid) + '\n')
print(f'Qwen main evaluation started: PID={process.pid}')
PY
unset QWEN_API_KEY
