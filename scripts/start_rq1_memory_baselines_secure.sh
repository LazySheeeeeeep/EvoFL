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
output = root / 'runs/rq1_memory_baselines_20260924'
for arm in ('episodic', 'flat_reflection', 'oracle_function_reflection'):
    arm_dir = output / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    pid_file = arm_dir / 'experiment.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists() and f'--arm\0{arm}\0'.encode() in cmdline.read_bytes():
            print(f'{arm} already running: PID={pid}')
            continue
    with (arm_dir / 'experiment.log').open('a') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', 'scripts/run_rq1_memory_baseline.py',
             '--arm', arm, '--limit', '500'],
            cwd=root,
            env=dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_file.write_text(str(process.pid) + '\n')
    print(f'{arm} worker started: PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
