#!/usr/bin/env bash
set -euo pipefail

runner_pid="${1:?runner pid is required}"
summary="${2:?success-only summary path is required}"
python_bin="${3:-python3}"

while kill -0 "$runner_pid" 2>/dev/null; do
    completed="$("$python_bin" - "$summary" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    print(int(json.loads(path.read_text(encoding="utf-8")).get("processed") or 0))
PY
)"
    if [ "$completed" -ge 500 ]; then
        printf 'success-only reached %s/500; stopping runner %s before its duplicate failure phase\n' \
            "$completed" "$runner_pid"
        kill -TERM "$runner_pid"
        exit 0
    fi
    sleep 5
done
