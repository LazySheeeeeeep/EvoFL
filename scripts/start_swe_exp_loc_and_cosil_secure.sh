#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL

root=runs/external_fl_baseline_preflight_20260924
cosil_output="$root/cosil_full500_results"
swe_exp_output="$root/swe_exp_loc"
mkdir -p "$cosil_output" "$swe_exp_output"

is_running() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid=$(cat "$pid_file")
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

if is_running "$cosil_output/experiment.pid" && is_running "$swe_exp_output/experience_smoke.pid"; then
    echo 'Both external runs are already active.'
    exit 0
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
export PYTHONUNBUFFERED=1
export SWE_EXP_LIMIT=${SWE_EXP_LIMIT:-5}

/home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
import os
from pathlib import Path
import subprocess
import sys

root = Path('runs/external_fl_baseline_preflight_20260924').resolve()
jobs = [
    ('cosil_full500_results', 'experiment.pid', 'experiment.log',
     [sys.executable, '-u', 'scripts/run_cosil_pilot.py',
      '--limit', '500', '--pilot', str(root / 'cosil_full500'),
      '--output', str(root / 'cosil_full500_results'),
      '--reuse-pilot', str(root / 'cosil_pilot_30'),
      '--reuse-results', str(root / 'cosil_pilot_30_results')]),
    ('swe_exp_loc', 'experience_smoke.pid', 'experience_smoke.log',
     [sys.executable, '-u', 'scripts/build_swe_exp_loc_memory.py',
      '--limit', os.environ['SWE_EXP_LIMIT']]),
]
for directory, pid_name, log_name, command in jobs:
    output = root / directory
    pid_file = output / pid_name
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
            print(f'{directory}: already running')
            continue
        except (ValueError, ProcessLookupError):
            pass
    with (output / log_name).open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=Path.cwd(), env=os.environ.copy(),
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(str(process.pid) + '\n', encoding='utf-8')
    print(f'{directory}: started PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
