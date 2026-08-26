#!/usr/bin/env bash
# Replay one completed V3 trajectory into the configured V3 SkillBank.
set -euo pipefail

: "${KEY77_API_KEY:?KEY77_API_KEY must be set for the replay process}"
: "${V3_REPLAY_SOURCE_RUN:?V3_REPLAY_SOURCE_RUN must be set}"
: "${V3_REPLAY_OUTPUT_DIR:?V3_REPLAY_OUTPUT_DIR must be set}"

cd /mnt/d/projects/EvoluteFL
exec env PYTHONPATH=src python3 scripts/replay_staged_case_evolution.py \
  --source-run-dir "$V3_REPLAY_SOURCE_RUN" \
  --output-dir "$V3_REPLAY_OUTPUT_DIR" \
  --config config/evolutefl.global.json \
  --provider key77 \
  --model gpt-5-mini \
  --sample-size 1 \
  --skill-bank-path skill_pools/skill_bank_v3/skills.jsonl \
  --embedding-cache-path skill_pools/skill_bank_v3/embeddings/jina_v3/skill_embeddings.jsonl \
  --retrieval-mode embedding \
  --embedding-base-url http://127.0.0.1:8008 \
  --rebuild-embeddings-after-case \
  --reflect-success
