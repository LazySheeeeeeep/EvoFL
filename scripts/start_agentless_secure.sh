#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'Waiting for non-echoed credential input.\n'
    IFS= read -r -s DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
fi
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
~/.venvs/evolutefl-agentless/bin/python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
root = Path('/mnt/d/projects/EvoluteFL')
out = root / 'runs/agentless_fl_heldout30_deepseek_20260915'
cases = json.loads((out / 'selected_cases.json').read_text())
for case in cases[:3]:
    result = json.loads((out / 'cases' / case['instance_id'] / 'result.json').read_text())
    if result['status'] != 'completed':
        raise RuntimeError('Three-case smoke must complete before full launch')
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        if b'scripts/run_agentless_heldout30.py' in proc.read_bytes():
            raise RuntimeError('Agentless experiment is already running')
    except (FileNotFoundError, PermissionError):
        pass
with (out / 'experiment.log').open('a') as log:
    process = subprocess.Popen(
        [str(Path.home() / '.venvs/evolutefl-agentless/bin/python'), '-u',
         'scripts/run_agentless_heldout30.py', '--limit', '30'], cwd=root,
        env=dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1'),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(out / 'experiment.pid').write_text(str(process.pid) + '\n')
print(f'Agentless 30-case experiment started, PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
