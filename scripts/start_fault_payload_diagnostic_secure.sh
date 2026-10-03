#!/usr/bin/env bash
set -euo pipefail
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'Waiting for non-echoed credential input.\n'
    IFS= read -r -s DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
fi
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
python3 - <<'PY'
import os
import subprocess
from pathlib import Path

root = Path('/mnt/d/projects/EvoluteFL')
out = root / 'runs/fault_payload_diagnostic_20260912'
out.mkdir(parents=True, exist_ok=True)
pidfile = out / 'evaluation.pid'
if pidfile.exists():
    command = Path(f'/proc/{int(pidfile.read_text())}/cmdline')
    if command.exists() and b'run_fault_payload_diagnostic.py' in command.read_bytes():
        raise RuntimeError('Diagnostic is already running')
env = dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1')
with (out / 'evaluation.log').open('a') as log:
    process = subprocess.Popen(
        ['python3', '-u', 'scripts/run_fault_payload_diagnostic.py'],
        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
pidfile.write_text(str(process.pid) + '\n')
print(f'Diagnostic started, PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
