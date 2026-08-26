#!/usr/bin/env bash
# Run a sequential V3 training experiment in WSL. Credentials are supplied by
# the parent process and are never written to disk by this helper.
set -euo pipefail

: "${KEY77_API_KEY:?KEY77_API_KEY must be set for the training process}"
: "${V3_SELECTED_CASES_FILE:?V3_SELECTED_CASES_FILE must be set}"
: "${V3_OUTPUT_DIR:?V3_OUTPUT_DIR must be set}"
: "${V3_SKILL_BANK_PATH:?V3_SKILL_BANK_PATH must be set}"
: "${V3_EMBEDDING_CACHE_PATH:?V3_EMBEDDING_CACHE_PATH must be set}"

cd /mnt/d/projects/EvoluteFL
args=(
  --selected-cases-file "$V3_SELECTED_CASES_FILE"
  --output-dir "$V3_OUTPUT_DIR"
  --provider key77
  --model gpt-5-mini
  --max-steps "${V3_MAX_STEPS:-30}"
  --case-timeout-seconds "${V3_CASE_TIMEOUT_SECONDS:-900}"
  --skill-bank-path "$V3_SKILL_BANK_PATH"
  --embedding-cache-path "$V3_EMBEDDING_CACHE_PATH"
  --enable-embedding
  --retrieval-mode embedding
  --embedding-base-url http://127.0.0.1:8008
  --rebuild-embeddings-after-case
)
if [[ -n "${V3_RESUME:-}" ]]; then
  args+=(--resume)
fi
if [[ -n "${V3_REFLECT_SUCCESS:-}" ]]; then
  args+=(--reflect-success)
fi
exec env PYTHONPATH=src python3 scripts/run_swesmith_case_by_case.py "${args[@]}"
