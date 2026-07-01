#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/runs/logs/jina_v3_embedding_server.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  ps -p "$PID" -o pid,etime,pcpu,pmem,cmd || true
else
  echo "no pid file"
fi
du -sh "$HOME/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3" 2>/dev/null || true
find "$HOME/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3" -type f 2>/dev/null | wc -l
