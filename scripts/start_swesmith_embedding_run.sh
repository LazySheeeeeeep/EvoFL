#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${KEY77_API_KEY:-}" ]]; then
  echo "KEY77_API_KEY is required." >&2
  exit 1
fi

RUN_NAME="${1:-swe_smith_5_embedding_gpt5mini_20260616}"
MODEL="${2:-gpt-5-mini}"
SEED="${3:-20260616}"
mkdir -p runs/logs

PYTHONPATH=src KEY77_API_KEY="$KEY77_API_KEY" nohup python3 -u scripts/run_swesmith_case_by_case.py \
  --config runs/key77_gpt4omini_config_20260615.json \
  --sample-size 5 \
  --seed "$SEED" \
  --output-dir "runs/$RUN_NAME" \
  --provider key77 \
  --model "$MODEL" \
  --enable-embedding \
  --retrieval-mode embedding \
  --embedding-base-url http://127.0.0.1:8008 \
  > "runs/logs/$RUN_NAME.log" 2>&1 &

echo "$!" > "runs/logs/$RUN_NAME.pid"
echo "started pid=$(cat "runs/logs/$RUN_NAME.pid") log=runs/logs/$RUN_NAME.log"
