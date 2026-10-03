#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
root=runs/external_fl_baseline_preflight_20260924/swe_exp_loc
output="$root/evaluation_full335"
mkdir -p "$output"

PYTHONPATH=src:scripts /home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('runs/external_fl_baseline_preflight_20260924/swe_exp_loc')
progress = json.loads((root / 'experience_build_progress.json').read_text())
if progress.get('status') != 'complete' or progress.get('completed') != 397:
    raise SystemExit('SWE-Exp-Loc historical extraction is not complete')
manifest = json.loads((root / 'e5_issue_types.manifest.json').read_text())
if manifest.get('count') != 335:
    raise SystemExit('Independent E5 memory index must contain exactly 335 histories')
smoke = json.loads((root / 'evaluation_smoke_5memory/comparison_summary.json').read_text())
if (smoke.get('processed') != 1 or smoke['cases'][0].get('status') != 'completed'
        or smoke.get('instructor_calls', 0) < 1):
    raise SystemExit('The one-case evaluation smoke did not complete with Instructor guidance')
print('SWE-Exp-Loc full evaluation gates passed: 397 extracted, 335 indexed, live smoke completed')
PY

pid_file="$output/experiment.pid"
if [ -f "$pid_file" ]; then
    old_pid=$(cat "$pid_file")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        printf 'SWE-Exp-Loc full evaluation already running: PID=%s\n' "$old_pid"
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
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export EVOLUTEFL_DIRECT_HTTP_TIMEOUT=1
.venv-jina/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess
import sys

output = Path('runs/external_fl_baseline_preflight_20260924/swe_exp_loc/evaluation_full335').resolve()
with (output / 'experiment.log').open('a', encoding='utf-8') as log:
    process = subprocess.Popen(
        [sys.executable, '-u', 'scripts/run_swe_exp_loc_baseline.py',
         '--limit', '500', '--output', str(output)],
        cwd=Path.cwd(), env=os.environ.copy(), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
(output / 'experiment.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
print(f'SWE-Exp-Loc full evaluation started: PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
