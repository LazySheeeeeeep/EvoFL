# RQ1: chronological experience acquisition and frozen FL comparison

## Question

How does EvoluteFL compare with existing fault-localization approaches under
the same issue/repository inputs, model and function-level scoring protocol?

## Data

- Acquisition: target 200 function-eligible real SWE-bench cases, selected in
  seeded random order from 561 candidates created before 2020, excluding the
  entire Lite/Verified union. GitHub PR merge time must also precede 2020.
- Evaluation: all 178 retained-record-unexposed Verified candidates created
  from 2021 onward. This is a temporal subset, not the full Verified benchmark.
- 68 of the initial 246 late Verified candidates were excluded by retained
  exposure records. Deleted historical records cannot certify global novelty.
- Source snapshots and patch-function eligibility still require materialization.
  Cases without old-side function targets are excluded before any arm runs.
- Exact patch duplicates and linked issue groups are checked across acquisition
  and evaluation. PR-body links are an operational grouping heuristic, not a
  guarantee of detecting all semantic duplicates or issue cross-references.
- `created_at` is PR creation, a task-time proxy, not original issue publication.
  This controls memory label availability relative to that proxy; it does not
  establish historical availability of every edit to the issue text.
- No SWE-smith, old SkillBank, evaluation patch, or evaluation trajectory is
  supplied to acquisition agents. Evaluation labels are used only offline.

## Execution

WSL only; DeepSeek official deepseek-v4-flash, temperature 0. Native Explorer
tools, max_steps=30, runtime=1200 seconds, current frozen v5 prompt package.
No embedding, repository tests or patch generation. Archives come from official
GitHub at exact base_commit; repair patches are never applied to the snapshot.

Build a separate empty bank, learn sequentially, then freeze its SHA-256.
All 200 eligible acquisition attempts count regardless of localization outcome;
only completed Explorer traces enter the existing reflection eligibility logic.
The first two are protocol smoke gates, never accuracy gates. Uncertain partial
transactions and infrastructure failures stop for inspection rather than skip
or replace difficult cases. GitHub rate limits wait until reset.

Evaluation arms: EvoluteFL with the frozen bank, Explorer without skill services
(RQ2 control), and pinned Agentless-FL file/related-function localization.
Agentless retains its own localization mechanism and output budget; this is a
same-model comparison, not a claim of equal token consumption. Line localization,
patching and testing are not run. This is the first external RQ1 baseline, not
a completed comparison against every planned baseline. Cycle arm order by case.

Primary metrics: strict patch-derived function Top-1/3/5 and MRR@5, any-target hit.
Empty predictions and execution failures on function-eligible cases remain
zero-score outcomes. Report completion, load rate, tool calls, time, token use
and paired changes alongside scores. Final paired confidence intervals and
per-project diagnostics should be produced after the run, without retuning.

## Artifacts

`runs/rq1_temporal_deepseek_20260915/` contains pinned inputs, exposure audit,
PR metadata, source archives/checksums, per-case bank checkpoints, trajectories,
reflection outputs, frozen final bank, and paired evaluation results.

No existing SkillBank or previous run is deleted or overwritten. Credentials
are inherited in process environment only. This protocol does not eliminate
closed-model pretraining contamination. Existing development cases are not
reported as independent evaluation cases.
