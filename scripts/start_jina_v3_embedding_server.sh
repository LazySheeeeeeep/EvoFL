#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p runs/logs
source "$HOME/.venvs/evolutefl-jina/bin/activate"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
nohup python scripts/serve_jina_v3_embeddings.py --host 127.0.0.1 --port 8008 --device cpu \
  > runs/logs/jina_v3_embedding_server.log 2>&1 &
echo "$!" > runs/logs/jina_v3_embedding_server.pid
echo "started pid=$(cat runs/logs/jina_v3_embedding_server.pid)"
