#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
/home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
from pathlib import Path
import os
import subprocess

output = Path('runs/external_fl_baseline_preflight_20260924/cosil_pilot_30_results').resolve()
output.mkdir(parents=True, exist_ok=True)
log = (output / 'finalizer.log').open('a', encoding='utf-8')
process = subprocess.Popen(
    ['bash', 'scripts/finalize_cosil_30.sh'], cwd=Path.cwd(),
    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
    env=os.environ.copy(), start_new_session=True,
)
(output / 'finalizer.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
print(f'CoSIL finalizer started: PID={process.pid}')
PY
