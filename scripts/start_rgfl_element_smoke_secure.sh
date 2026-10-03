#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
pid_file=runs/external_fl_baseline_preflight_20260924/rgfl_pilot/live_element_smoke/experiment.pid
if [ -f "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        printf 'RGFL element-stage smoke already running: PID=%s\n' "$old_pid"
        exit 0
    fi
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'DeepSeek credential (process only): '
    IFS= read -r -s DEEPSEEK_API_KEY
    printf '\n'
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
fi
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
export PYTHONUNBUFFERED=1
/home/lql/.venvs/evolutefl-rgfl/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys

output = Path('runs/external_fl_baseline_preflight_20260924/rgfl_pilot/live_element_smoke').resolve()
output.mkdir(parents=True, exist_ok=True)
log = (output / 'experiment.log').open('a', encoding='utf-8')
process = subprocess.Popen(
    [sys.executable, '-u', 'scripts/run_rgfl_element_smoke.py'],
    cwd=Path.cwd(), env=os.environ.copy(), stdin=subprocess.DEVNULL,
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
(output / 'experiment.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
print(f'RGFL element-stage smoke started: PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
