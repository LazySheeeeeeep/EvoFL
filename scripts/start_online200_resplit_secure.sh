#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'DeepSeek API key (process only): '
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
output = root / "runs/online200_resplit_adaptation_eval300_20261003"
output.mkdir(parents=True, exist_ok=True)
log = (output / "experiment.log").open("a", encoding="utf-8")
process = subprocess.Popen(
    [
        sys.executable,
        "-u",
        "scripts/run_online200_resplit_adaptation.py",
        "--workers",
        "12",
        "--output",
        str(output),
    ],
    cwd=root,
    env=dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1"),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(output / "experiment.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
print(f"Online-200 resplit adaptation started: PID={process.pid}")
PY

unset DEEPSEEK_API_KEY
