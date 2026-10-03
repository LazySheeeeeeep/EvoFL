#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    printf 'DeepSeek credential (process only): '
    IFS= read -r -s DEEPSEEK_API_KEY
    printf '\n'
    DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY%$'\r'}
fi
test -n "$DEEPSEEK_API_KEY"
export DEEPSEEK_API_KEY
export PYTHONUNBUFFERED=1
exec /home/lql/.venvs/evolutefl-locagent/bin/python -u scripts/run_locagent_live_smoke.py "$@"
