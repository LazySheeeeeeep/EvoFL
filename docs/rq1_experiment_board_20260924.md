# RQ1 experiment board

Last checked: 2026-09-27 (Asia/Shanghai). Inspect each run's
`comparison_summary.json` for live progress. The common evaluation target is the
frozen 500-case temporal SWE-bench manifest with existing-function patch labels.
Published scores on Lite/Verified are not directly comparable to this target.

| Experiment | Role | Current state | Next gate |
| --- | --- | --- | --- |
| EvoluteFL / No-Skill / Agentless-FL | Main three-arm RQ1 comparison | Acquisition 400/400 and frozen paired evaluation 500/500 complete | Audit final function-level scores and costs |
| Episodic-Retrieval | Internal memory-format ablation | All 400 acquisition episodes frozen; full single-worker evaluation running | Finish evaluation and audit memory selection and function-level scores |
| Flat-Reflection Memory | Internal memory-format ablation | All 400 reflections built with zero failures; full single-worker evaluation running | Finish evaluation and audit memory selection and function-level scores |
| Oracle-Function Reflection | Privileged acquisition-side reflection control | 400/400 historical lessons generated with zero failures; full single-worker evaluation running | Finish paired evaluation and audit whether oracle-derived lessons change function-level outcomes |
| SWE-Exp-Loc (adapted) | Historical experience comparison | 397/397 historical experiences built; E5 index contains 335 independent memories; one-case frozen evaluation smoke completed with six Instructor calls; full 500-case evaluation active | Monitor full-run adapter failures and score after all 500 cases |
| RepoMem-Loc (adapted) | Repository-history memory comparison | Public requests history acquired; 2840 pre-issue ancestors audited for an acquisition case | Linked-issue overlap filtering, patches, summaries and LocAgent memory tools |
| LocAgent | External graph-guided FL baseline | Frozen Flask case live smoke and source-backed function normalization completed; Top-1 on this one case; acquisition smoke remains unscored | Audit cost and adaptation before any larger FL-only evaluation; one-case smoke is not a baseline result |
| RGFL | External reasoning-guided FL baseline | Paid file/element smokes; four author reasoning/ranking stages passed real-source offline mock (160 calls) | Resolve embedding backend, live full-stage cost, ambiguous method names |
| CoSIL | External function-level FL candidate | Corrected 30-case pilot complete, 500 source structures prepared, full evaluation active with first 30 reused | Monitor remaining author inference and score all cases |

Execution rules: keep the active three-arm runner and hashed inputs untouched.
External code, indexes, outputs, and environment setup use separate directories.
All added arms use the same case IDs and function-label mapper; report input,
model, tool, and budget differences explicitly. Start at most small acquisition
smokes while the main run is training; avoid overlapping full paid evaluations
with its four workers until provider throughput and disk use are measured.
The memory-format ablations built from sequential EvoluteFL acquisition traces
are not independent no-memory acquisitions. Full SWE-Exp and RepoMem methods
require additional artifacts; adapted variants must be named as such.
Oracle-Function Reflection sees known repair functions only for historical
acquisition cases. It never receives evaluation-case oracle functions or patches;
it is a privileged-input control for trajectory-based reflection, not a
deployable no-oracle method.

External-source audit and blockers: `docs/external_fl_baseline_preflight_20260924.md`.
Latest parallel preparation: `docs/external_baseline_readiness_20260927.md`.
Frozen patch-free input: `runs/external_fl_baseline_preflight_20260924/evaluation_public.jsonl`.
CoSIL pilot readiness: `runs/external_fl_baseline_preflight_20260924/cosil_pilot/pilot_readiness.json`.
CoSIL live pilot: `runs/external_fl_baseline_preflight_20260924/cosil_pilot_results_v2/comparison_summary.json`.
Focused WSL verification: 12 tests passed for external input isolation and memory preparation.
