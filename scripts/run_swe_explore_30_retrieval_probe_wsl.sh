#!/usr/bin/env bash
set -uo pipefail

cd /mnt/d/projects/EvoluteFL
mkdir -p runs/logs

exec env PYTHONPATH=src python3 scripts/run_swe_explore_abstractor_skill_probe.py \
  --output-dir runs/swe_explore_30_abstractor_retrieval_diagnostic_gpt5mini_jina_20260717 \
  --config runs/key77_gpt4omini_config_20260615.json \
  --provider key77 \
  --model gpt-5-mini \
  --sample-size 30 \
  --seed 20260717 \
  --embedding-base-url http://127.0.0.1:8008 \
  --embedding-min-score 0.48 \
  --exclude-all-runs-selected-cases \
  --resume \
  >> runs/logs/swe_explore_30_abstractor_retrieval_diagnostic_gpt5mini_jina_20260717.log 2>&1
