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
export DEEPSEEK_API_KEY PYTHONPATH=src PYTHONUNBUFFERED=1
/home/lql/.venvs/evolutefl-agentless/bin/python - <<'PY'
import os, subprocess, sys
from pathlib import Path
root=Path.cwd(); output=root/'runs/online_temporal100_adaptation_eval400_20261003'; output.mkdir(parents=True, exist_ok=True)
log=(output/'experiment.log').open('a')
p=subprocess.Popen([sys.executable,'-u','scripts/run_online_temporal_adaptation.py','--count','100','--workers','12','--output',str(output)],cwd=root,env=dict(os.environ,PYTHONPATH='src',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(output/'experiment.pid').write_text(str(p.pid)+'\n')
print(f'Online-100 temporal started: PID={p.pid}')
PY
unset DEEPSEEK_API_KEY
