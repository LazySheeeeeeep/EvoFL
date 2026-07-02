#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${KEY77_API_KEY:-}" ]]; then
  echo "KEY77_API_KEY is required." >&2
  exit 1
fi

RUN_NAME="${1:-rerun_existing_swesmith_reflector_general_20260702}"
MODEL="${2:-gpt-5-mini}"
CONFIG="${3:-runs/key77_gpt4omini_config_20260615.json}"

mkdir -p runs/logs

PYTHONPATH=src python3 -u scripts/rerun_case_evolution_from_existing_runs.py \
  --repo-contains "swesmith/" \
  --output-dir "runs/$RUN_NAME" \
  --config "$CONFIG" \
  --provider key77 \
  --model "$MODEL" \
  --reset-skill-bank \
  --reset-embedding-cache \
  --general-every 10 \
  --general-window-size 10 \
  --run-final-general-reflection \
  > "runs/logs/$RUN_NAME.log" 2>&1
