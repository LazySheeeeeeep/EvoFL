#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL

if ! curl -fsS --max-time 5 http://127.0.0.1:8008/health >/dev/null 2>&1; then
    bash scripts/start_jina_v3_embedding_server.sh >/dev/null
    for _ in $(seq 1 60); do
        if curl -fsS --max-time 5 http://127.0.0.1:8008/health >/dev/null 2>&1; then
            break
        fi
        sleep 5
    done
fi
curl -fsS --max-time 10 http://127.0.0.1:8008/health >/dev/null

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
output = root / "runs/external_fl_baseline_preflight_20260924/agentless_retrieval_pilot"
output.mkdir(parents=True, exist_ok=True)
log = (output / "experiment.log").open("a", encoding="utf-8")
process = subprocess.Popen(
    [sys.executable, "-u", "scripts/run_agentless_retrieval_pilot.py"],
    cwd=root,
    env=dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1"),
    stdin=subprocess.DEVNULL,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(output / "experiment.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
print(f"Agentless retrieval pilot started: PID={process.pid}")
PY

unset DEEPSEEK_API_KEY
