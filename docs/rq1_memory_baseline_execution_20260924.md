# RQ1 memory baseline execution status

The active 400-acquisition/500-evaluation runner has only three arms:
EvoluteFL, No-Skill, and Agentless-FL. Do not modify its frozen protocol or
hashed source files while it is running.

## Additional arms

| Arm | Acquisition memory | Evaluation interface | Status |
| --- | --- | --- | --- |
| Episodic-Retrieval | Historical issue, observed investigation, final prediction | Retrieve one episode and expose its compact trajectory to the same Explorer | Adapter and Fake-LLM integration tested; paid inference not run |
| Flat-Reflection Memory | One direct reflection from a historical trajectory and repair; no v5 gap investigator | Retrieve one lesson and expose it to the same Explorer | One-call builder and adapter tested; reflection calls not run |
| SWE-Exp-Loc (adapted) | SWE-Exp-style perspectives/reflections from audited acquisition only | Localization-only adaptation with declared changes to original search/repair action space | Local upstream inspected; no adapter or inference |
| RepoMem-Loc (adapted) | Historical commits, linked issues, repository summaries available before each base commit | Function-level localization adaptation | No verified official implementation or historical Git cache; not ready |

The first two are *memory-format ablations*, not external published baselines.
Acquisition Explorer trajectories are sequential EvoluteFL trajectories and may
contain previously loaded skills. They cannot be described as independent
no-memory acquisitions. The offline preparation output records this per episode.
The old 200 training metrics may use an older target policy: recompute outcomes
against the frozen existing-function targets before making reflection labels.

## Current safe parallel work

`python3 scripts/prepare_rq1_memory_baselines.py` runs without network, model
calls, or writes to the active experiment. It checks temporal cutoffs and
training/evaluation overlap, then writes provisional `episodes.jsonl`,
`flat_reflection_inputs.jsonl`, and `preparation_summary.json` under
`runs/rq1_memory_baseline_preparation_20260924/`. Re-run after all 400 training
cases finish; freeze the memory inputs and their hashes before paid inference.

The 500-case evaluation must use the same frozen manifest and function-level
labels as the active RQ1. Run additional arms in a separate output directory,
with independent case locks, cost logs, and status counts. Never report adapted
SWE-Exp or RepoMem as a faithful official reproduction. Full evaluation should
wait until the memory builders and 1-case acquisition-only smoke tests pass;
running extra arms during the four-worker main evaluation would add API
contention and offers no speed guarantee.

## Prepared execution

After the main acquisition reaches 400, re-run the offline preparation command
and check `status=complete`. Then run:

```bash
PYTHONPATH=src python3 scripts/build_rq1_flat_reflections.py --dry-run --limit 5
PYTHONPATH=src python3 scripts/run_rq1_memory_baseline.py --arm episodic --dry-run
```

The first paid smoke should use `--arm episodic --limit 1` only after checking
the acquisition freeze. Flat Reflection requires one direct reflection per
acquisition case; `build_rq1_flat_reflections.py --limit 1` is its paid smoke.
Both builders checkpoint per source case. After review, build all 400 flat
reflections and run the `flat_reflection` evaluation arm. The extra runner
refuses to evaluate before all 400 acquisitions and 400 prepared episodes exist.
It uses an empty SkillBank, an independent source/work directory, the same
frozen Explorer prompt/config and function targets, and writes memory source
IDs and retrieval scores for audit. Its initial-payload artifact remains the
unmodified Explorer payload; `memory_selection.json` records the injected item.
