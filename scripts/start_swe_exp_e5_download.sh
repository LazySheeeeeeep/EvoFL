#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
output=runs/external_fl_baseline_preflight_20260924/swe_exp_loc
mkdir -p "$output"
pid_file="$output/e5_download.pid"
if [ -f "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        printf 'E5 download already running: PID=%s\n' "$old_pid"
        exit 0
    fi
fi
PYTHONPATH=src:scripts /home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess

output = Path('runs/external_fl_baseline_preflight_20260924/swe_exp_loc').resolve()
log = (output / 'e5_download.log').open('a', encoding='utf-8')
process = subprocess.Popen(
    ['.venv-jina/bin/python', '-u', 'scripts/prepare_swe_exp_e5_model.py'],
    cwd=Path.cwd(), env={**os.environ, 'PYTHONPATH': 'src:scripts'},
    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
    start_new_session=True,
)
(output / 'e5_download.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
print(f'E5 download started: PID={process.pid}')
PY
