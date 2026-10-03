#!/usr/bin/env bash
set -euo pipefail
export EVOLUTION_PHASE="${1:-audit}"
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
out = root / 'runs/evidence_reflector_heldout30_deepseek_20260914'
out.mkdir(parents=True, exist_ok=True)
pidfile = out / 'experiment.pid'
if pidfile.exists():
    command = Path(f'/proc/{int(pidfile.read_text())}/cmdline')
    if command.exists() and b'run_evidence_reflector_comparison.py' in command.read_bytes():
        raise RuntimeError('Experiment is already running')
with (out / 'experiment.log').open('a') as log:
    process = subprocess.Popen(
        ['python3', '-u', 'scripts/run_evidence_reflector_comparison.py', '--phase', os.environ['EVOLUTION_PHASE']],
        cwd=root, env=dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1'),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
pidfile.write_text(str(process.pid) + '\n')
print(f'Experiment phase={os.environ["EVOLUTION_PHASE"]} started, PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
