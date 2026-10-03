#!/usr/bin/env bash
set -euo pipefail
printf 'Waiting for non-echoed credential input.\n'
IFS= read -r -s DEEPSEEK_API_KEY
DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
python3 - <<'PY'
import subprocess
from pathlib import Path

root = Path('/mnt/d/projects/EvoluteFL')
out = root / 'runs/v5_train40_deepseek_v4_flash_20260910_continuation'
with (out / 'training.log').open('a') as log:
    process = subprocess.Popen(
        ['/bin/bash', str(root / 'scripts/run_v5_train40_continuation.sh')],
        cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
(out / 'training.pid').write_text(str(process.pid) + '\n')
print(f'Training started, PID={process.pid}')
PY
unset DEEPSEEK_API_KEY
