#!/usr/bin/env bash
set -uo pipefail

cd /mnt/d/projects/EvoluteFL
mkdir -p runs/logs

exec env PYTHONPATH=src python3 scripts/replay_existing_trajectories.py \
  --output-dir runs/replay_all_existing_trajectories_v2skills_20260717 \
  --provider key77 \
  --model gpt-5-mini \
  --enable-embedding \
  --retrieval-mode embedding \
  --embedding-base-url http://127.0.0.1:8008 \
  --embedding-min-score 0.48 \
  --rebuild-embeddings-after-case \
  --resume \
  >> runs/logs/replay_all_existing_trajectories_v2skills_20260717.log 2>&1
