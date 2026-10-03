#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/projects/EvoluteFL
root=runs/external_fl_baseline_preflight_20260924
output="$root/cosil_pilot_30_results"
for _ in $(seq 1 1440); do
    if [ -f "$output/comparison_summary.json" ]; then
        processed=$(/home/lql/.venvs/evolutefl-agentless/bin/python -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["processed"])' \
            "$output/comparison_summary.json")
        if [ "$processed" = 30 ]; then
            /home/lql/.venvs/evolutefl-agentless/bin/python \
                scripts/rescore_cosil_pilot.py \
                --pilot "$root/cosil_pilot_30" \
                --results "$output" \
                --reuse-results "$root/cosil_pilot_results_v2" \
                --limit 30
            exit 0
        fi
    fi
    if [ -f "$output/experiment.pid" ]; then
        pid=$(cat "$output/experiment.pid")
        if [[ "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
            printf 'CoSIL runner exited before 30 cases; processed=%s\n' "${processed:-0}" >&2
            exit 1
        fi
    fi
    sleep 60
done
printf 'CoSIL 30-case finalizer timed out\n' >&2
exit 1
