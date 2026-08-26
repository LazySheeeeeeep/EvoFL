#!/usr/bin/env bash
# Run one V3 smoke case inside WSL. The caller supplies KEY77_API_KEY only as
# a process environment variable; this helper never stores credentials.
set -euo pipefail

: "${KEY77_API_KEY:?KEY77_API_KEY must be set for the smoke process}"

cd /mnt/d/projects/EvoluteFL
extra_args=()
if [[ -n "${V3_SKILL_BANK_PATH:-}" ]]; then
  extra_args+=(--skill-bank-path "$V3_SKILL_BANK_PATH")
fi
if [[ -n "${V3_EMBEDDING_CACHE_PATH:-}" ]]; then
  extra_args+=(--embedding-cache-path "$V3_EMBEDDING_CACHE_PATH")
fi
exec env PYTHONPATH=src python3 scripts/run_swesmith_case_by_case.py \
  --selected-cases-file "${V3_SELECTED_CASES_FILE:?}" \
  --output-dir "${V3_OUTPUT_DIR:?}" \
  --provider key77 \
  --model gpt-5-mini \
  --max-steps "${V3_MAX_STEPS:-12}" \
  --case-timeout-seconds "${V3_CASE_TIMEOUT_SECONDS:-900}" \
  --enable-embedding \
  --retrieval-mode embedding \
  --embedding-base-url http://127.0.0.1:8008 \
  --rebuild-embeddings-after-case \
  "${extra_args[@]}"
