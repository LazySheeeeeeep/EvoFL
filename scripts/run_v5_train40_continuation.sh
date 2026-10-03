#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
set -a
source /home/lql/.config/evolutefl/env.sh
set +a
test -n "${DEEPSEEK_API_KEY:-}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
python3 scripts/prepare_v5_train40.py
out=runs/v5_train40_deepseek_v4_flash_20260910_continuation
python3 scripts/run_swesmith_case_by_case.py \
  --config config/evolutefl.global.json --provider deepseek --model deepseek-v4-flash \
  --sample-size 40 --seed 20260911 --selected-cases-file "$out/selected_cases.json" \
  --output-dir "$out" --max-steps 30 --case-timeout-seconds 900 --resume
python3 scripts/report_v5_run.py "$out"
python3 scripts/rescore_v5_run.py "$out"
