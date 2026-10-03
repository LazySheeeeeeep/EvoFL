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
output = root / "runs/external_fl_baseline_preflight_20260924/swe_exp_loc/retry_failed_20260930"
output.mkdir(parents=True, exist_ok=True)
log = (output / "experiment.log").open("a", encoding="utf-8")
process = subprocess.Popen(
    [
        str(root / ".venv-jina/bin/python"),
        "-u",
        "scripts/retry_swe_exp_loc_failures.py",
        "--source-summary",
        "runs/external_fl_baseline_preflight_20260924/swe_exp_loc/evaluation_full335/comparison_summary.json",
        "--output",
        str(output),
        "--attempts",
        "3",
    ],
    cwd=root,
    env=dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1"),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(output / "experiment.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
print(f"SWE-Exp-Loc failure retry started: PID={process.pid}")
PY

unset DEEPSEEK_API_KEY
