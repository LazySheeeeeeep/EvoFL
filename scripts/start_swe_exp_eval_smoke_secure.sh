#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
output=runs/external_fl_baseline_preflight_20260924/swe_exp_loc/evaluation_smoke_5memory
mkdir -p "$output"
pid_file="$output/experiment.pid"
if [ -f "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        printf 'SWE-Exp-Loc evaluation smoke already running: PID=%s\n' "$old_pid"
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
export PYTHONPATH=src:scripts
export TOKENIZERS_PARALLELISM=false
export EVOLUTEFL_DIRECT_HTTP_TIMEOUT=1
export PYTHONUNBUFFERED=1
.venv-jina/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess
import sys

output = Path('runs/external_fl_baseline_preflight_20260924/swe_exp_loc/evaluation_smoke_5memory').resolve()
with (output / 'experiment.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(
        [sys.executable, '-u', 'scripts/run_swe_exp_loc_baseline.py',
         '--limit', '1', '--allow-partial-smoke', '--output', str(output)],
        cwd=Path.cwd(), env=os.environ.copy(), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
(output / 'experiment.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
print(f'SWE-Exp-Loc evaluation smoke started: PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
