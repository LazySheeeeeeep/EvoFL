#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'Waiting for non-echoed credential input.\n'
    IFS= read -r -s DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
fi
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
PYTHONPATH=src ~/.venvs/evolutefl-agentless/bin/python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys
root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from run_rq1_temporal import verify, OUT
verify()
from evolutefl.llm.client import OpenAICompatibleClient
cfg = json.loads((OUT / 'config.json').read_text())
client = OpenAICompatibleClient.from_config(cfg['llm'])
response = client.chat(
    messages=[{'role': 'user', 'content': 'Call ready with ok=true. This is a protocol connectivity check.'}],
    tools=[{'type':'function','function':{'name':'ready','description':'Confirm connection.',
           'parameters':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']}}}],
    tool_choice={'type':'function','function':{'name':'ready'}})
calls = response.get('tool_calls') or []
if (not calls or calls[0].get('function', {}).get('name') != 'ready'
        or json.loads(calls[0]['function']['arguments']).get('ok') is not True):
    raise RuntimeError('Native tool preflight failed; no experiment launched')
(OUT / 'provider_preflight.json').write_text(json.dumps({'model':cfg['llm']['model'],
    'native_tool_call':True, 'usage':response.get('raw',{}).get('usage',{})}, indent=2))
with (OUT / 'experiment.log').open('a') as log:
    process = subprocess.Popen([sys.executable, '-u', 'scripts/run_rq1_temporal.py', '--phase', 'full'],
        env=dict(os.environ, PYTHONPATH='src', PYTHONUNBUFFERED='1'), cwd=root,
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(OUT / 'experiment.pid').write_text(str(process.pid)+'\n')
print(f'RQ1 worker started: PID={process.pid}; metadata audit -> acquisition -> frozen evaluation')
PY
unset DEEPSEEK_API_KEY
