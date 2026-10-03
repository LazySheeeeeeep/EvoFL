# Agentless-FL held-out comparison

The September 15 pilot reuses the exact 30-case manifest, source archives,
function ground truth, and completed no-skill / expanded-bank outputs from
`runs/evidence_reflector_heldout30_deepseek_20260914`. It does not evolve skills.

## Baseline identity

Official OpenAutoCoder/Agentless commit
`5ce5888b9f149beaace393957a55ea8ee46c9f71`, using `LLMFL.localize` followed by
`localize_function_from_compressed_files` with top-3 files and compressed assignments.
Prompts, skeleton generation and location extraction remain upstream code.
The source parser receives actual repository-relative archive paths.

Report this as **Agentless-FL (LLM file + compressed related-function)**, not
the complete latest Agentless system. Embedding retrieval/merge, fine-grained
line localization, repair and patch validation are not executed. This is a
function-level pilot; conclusions do not establish superiority over every
Agentless configuration.

## Model adaptation and ranking

- Same official DeepSeek `deepseek-v4-flash`, temperature 0, environment credential.
- 4096 output tokens instead of upstream's 300; a length-truncated response gets
  one identical-message retry at 8192 with thinking disabled, as in the frozen
  Explorer configuration. No score-based retries or prompt tuning.
- Single sample per stage, no semantic location retry or temperature escalation.
- Use model-output order across files and locations. Resolve exact qualified
  functions or unique leaf names in source; deduplicate and take the first five.
- A class is not a function and is not expanded into its methods. Ignored class,
  variable and ambiguous locations are logged. Unknown explicit function names
  retain rank slots. No ground truth is available to the conversion function.
- The official related-element prompt requests a set, not an explicit global
  rank. Output-order ranking is therefore an evaluation adapter, not a claim
  that upstream calibrated these ranks. Preserve raw output for sensitivity analysis.

## Reporting and limitations

All 30 cases remain in execution counts. Function metrics use the same 29
function-eligible cases; the remaining case is retained and excluded explicitly.
API failures do not disappear from the metric denominator. Main metrics are
AnyHit Top-1/3/5 and MRR truncated at five. Every response and provider usage is
retained. Dollar estimates require verified contemporary pricing.

Historical Explorer results were generated earlier, not contemporaneously.
This is model-configuration aligned, but provider-time drift remains possible.
No skill bank or historical result is rewritten. Output hashes and the pinned
baseline source hashes are recorded before inference.

## Run (WSL only)

Use `~/.venvs/evolutefl-agentless/bin/python`, with `PYTHONPATH=src`.
Run `scripts/run_agentless_heldout30.py --limit 3` for smoke, then
`bash scripts/start_agentless_secure.sh` for the remaining cases. The launcher
requires three completed smoke cases and rejects a duplicate running job.
Credentials are read without echo and inherited by the experiment process;
they are not saved in configuration or result files.

Results: `runs/agentless_fl_heldout30_deepseek_20260915/comparison_summary.json`.
