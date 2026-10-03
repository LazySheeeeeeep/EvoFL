# Expanded RQ1: temporal audit and memory baselines

Status: audit running; baseline adapters and inference are not launched.

## User decision update

SWE-Exp is excluded from this round at the user's request; the discussion below
is retained as a feasibility record, not an active execution plan. Active ready
arms are with-skill, no-skill, and Agentless-FL. RepoMem remains pending faithful
implementation and is not silently replaced with an adapted APR method.

Continue requesting `deepseek-v4-flash` on the official endpoint. The provider's
September 10 release notes state this legacy name is routed to V4.1 Flash; the
request name is not a pinned old model snapshot. Record returned model metadata
and dates. A new credential is supplied only to processes, never this document.

Source: https://api-docs.deepseek.com/updates/

The earlier four-arm cost estimate below is superseded. With three arms, a full
500-case rerun would be 1,500 localization runs before any justified result reuse.
Parallel evaluation (initial target: four isolated case workers) is planned but
not yet implemented or benchmarked. SkillBank evolution stays sequential.

## Dataset target

- Acquisition target: 400 total, retaining the completed original 200 and adding
  200 eligible historical cases. Keep an independent copy of the original bank.
- Evaluation target: approximately 500 total function-eligible cases, not 500
  extra cases. Preserve the original temporal Verified subset where compatible
  with the final label/overlap audit, and add roughly 327 late full-SWE-bench cases.
- Acquisition PR creation and merge must precede 2020-01-01. Evaluation PR creation
  must be on or after 2021-01-01. The 2020 buffer stays unused.
- This is a custom temporal SWE-bench evaluation, not SWE-bench Verified 500.
- Selection is fixed before inference and independent of localization success,
  memory retrieval, or reflection outcome. The old 173 cases are already observed;
  report them separately from newly held-out additions.

The independent audit reads changed Python source at immutable base commits,
checks exact patch reconstruction, records hashes, and flags targets introduced
only by the repair. It does not silently adopt a new function-label policy.
Provisional manifests must be reviewed and frozen before paid evaluation.

## Relevant methods and reproduction decisions

| Method | Evidence inspected | Decision |
| --- | --- | --- |
| Agentless-FL | Existing pinned adapter, upstream 5ce5888b9f149beaace393957a55ea8ee46c9f71 | Retain existing external baseline; reuse matching old predictions only |
| SWE-Exp | Local upstream 6b5c92ed0a6fc14de972c5d499673e2c4f03ce33, README, exp_agent.py, select_agent.py, instructor.py | Prioritize explicitly named FL adaptation; confirmation and adapter smoke pending |
| Repository Memory / RepoMem | Paper v2 and author homepage | Relevant direct localization memory baseline; official implementation not found in inspected sources, faithful implementation remains pending |
| MemFL | Paper v1, sections IV-C and IV-D | Not a direct same-input baseline: depends on failing tests, stack traces and coverage on Java Defects4J |

Sources:

- https://github.com/cslsolow/SWE-Exp
- https://arxiv.org/html/2510.01003v2
- https://boshi-wang.github.io/
- https://arxiv.org/html/2506.03585v1

Do not use unrelated GitHub repositories named RepoMem as the paper's artifact.
Do not import published headline scores into the custom function-level table.

## SWE-Exp adaptation contract to settle before launch

The original code extracts successful perspectives and failed reflections from
repair trajectories, embeds issue-type descriptions with multilingual-e5-large-
instruct, recalls ten candidates, and performs LLM experience selection. Its
selector excludes experiences from the same repository. Its Instructor controls
search/view/modify/finish and adapts experience to the next action.

A localization-only adaptation must explicitly document:

1. Replacing repair success with a fixed function-localization outcome definition.
2. Removing repair/test execution and introducing ordered function predictions.
3. The provenance of acquisition trajectories. Using EvoluteFL trajectories is
   a controlled memory-transfer experiment, NOT a reproduction of SWE-Exp's
   original no-experience search-tree trajectory collection.
4. Whether the original Instructor/action space is retained. Injecting its lessons
   into EvoluteFL's Explorer alone must instead be named "EvoluteFL + SWE-Exp-style
   memory", not an independent reproduction of SWE-Exp-FL.
5. The original E5 retrieval, cross-repository exclusion, top-10 recall, selection,
   and task-specific experience adaptation. Any changed component is an explicit
   variant, not a silent engineering substitution.

Neither public pretrained experience trees nor test-case reference patches may
enter the baseline's memory. Construct and freeze memory from audited acquisition
data only. Small interface smoke tests must use acquisition cases, not evaluation
cases selected after seeing results.

## RepoMem constraints

The paper uses historical commit/linked-issue memory and summaries of frequently
edited files with LocAgent. Its reported metric is file-level complete-target
coverage, not the current function-level any-target-hit metric.

A reproduction needs pinned LocAgent, commit ancestry/time restrictions before
each test snapshot, target-issue overlap removal, source-snapshot summary caching,
and explicit function-output adaptation. It also receives a different volume and
kind of history from the 400 trajectory acquisition cases; report these memory
sources and construction costs instead of calling the inputs identical. Do not
start a costly 500-case run before those components pass a source audit and smoke.

## Budget and sequencing

Initial intended arms: EvoluteFL with skills, EvoluteFL without skills,
Agentless-FL, and one accepted memory baseline. RepoMem remains pending rather
than being replaced with a mislabeled heuristic.

At 500 eligible cases and 173 reusable old no-skill/Agentless predictions, this
requires approximately 1,654 new evaluation runs for four arms:
500 with-skill + 327 no-skill + 327 Agentless-FL + 500 memory baseline.
This excludes 200 additional EvoluteFL acquisition cases, baseline memory
construction, potential dedicated baseline trajectory acquisition, and retries.
Every additional full baseline costs another 500 evaluation runs.

Sequence: metadata/source audit -> final manifest and baseline contract ->
acquisition-case smoke -> acquisition/memory construction -> frozen paired
evaluation -> common scorer/status/cost report. The audit worker does not
automatically cross the unresolved baseline/label-policy gate.
