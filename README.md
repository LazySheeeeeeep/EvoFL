# EvoluteFL v5

Function-level fault localization with evidence-grounded investigation experience.
The active workflow uses only Fault Skills; Project and Strategy are inactive.

## Localization

```text
Issue + repository -> initial source exploration -> load_fault_skill once
-> continue investigation -> finish_localization (up to five functions)
```

V5 tools: `grep`, `find_symbol`, `read_symbol`, `read_file`, `read_observation`,
and `write`. Python symbol search returns definition IDs, signatures and ranges;
symbol reading includes decorators and supports continuation within the body.
Source reads return up to 200 complete lines and target 16,000 source characters
per page (a single oversized line is delivered whole). `next_start_line`
identifies unread content. The observation layer preserves the entire page;
it never clips serialized tool results. Replay returns exactly the originally
delivered observation, not hidden raw content. Write saves notes only under the
run's `artifacts/` directory. Fault loading selects one of nine families,
uses a title/trigger catalog selector, then validates the selected knowledge.
At most one Skill is loaded as a native tool response. No embedding service
is required; an empty catalog is a valid no-Skill result.

Calls may include `purpose`, `based_on` observation IDs and `candidate_updates`.
Source definitions appearing in tool results are not automatic hypotheses.
Logs retain exact tool responses and a complete action index. Default limits:
30 steps, three finalization turns and 900 seconds.

## Evolution

```text
Completed trace -> indexed investigation
-> read-only supplementary investigation (at most 16 calls)
-> supported process lesson -> independent Fault catalog selection
-> evidence-driven Reflector -> create / rewrite / preserve / no_update
```

The investigator uses `grep`, `find_symbol`, `read_symbol`, `read_file`, and `read_observation` against the
same buggy source. Ground truth is training-only. Original `obs-` evidence
and supplementary `supp-` discoveries are separate. Every resolved finding
anchors a reviewed original source observation. Unresolved findings never
update the bank. Repository fingerprints prevent replay on changed code;
old trajectory formats are rejected.

Skills contain family, subtype, title, trigger and unordered knowledge strings.
Evidence references remain in run artifacts, outside Skill cards. Existing
complete-card versioning is reused. The active bank is
`skill_pools/skill_bank_v5/skills.jsonl`.

## WSL Commands

Run from `/mnt/d/projects/EvoluteFL`. Configure credentials using environment
variables such as `DEEPSEEK_API_KEY`, rather than result files.

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m evolutefl.cli show-tools
PYTHONPATH=src python3 scripts/run_swesmith_case_by_case.py \
  --provider deepseek --model deepseek-v4-flash --sample-size 2 \
  --seed 20260907 --max-steps 30 --case-timeout-seconds 900 \
  --output-dir runs/v5_smoke_2_deepseek_20260907
python3 scripts/report_v5_run.py runs/v5_smoke_2_deepseek_20260907
```

Separate `run-case-evolution` requires `--case-run-dir`, `--repo-path`,
`--repo`, issue input and ground-truth functions. For SWE-smith mutation patches,
supply `--ground-truth-patch-file` and `--patch-direction clean_to_buggy`.
A second successful Explorer run and repository tests are not required.

Top-5 hit/miss remains a patch-function ranking outcome, not a judgment of
investigation quality. The Investigator attributes each supported finding to
evidence acquisition, interpretation, candidate ranking, or a useful observed
practice, retaining uncertainty where the record is insufficient. One shared
Reflector prompt uses these findings for both hit and miss cases. The summary
records `learning_foci`; these are diagnostic metadata, not new Skill types.
Older conclusions without this field remain unattributed (`uncertain`).

Inspect `investigation_index.json`, `observations.jsonl`, `trajectory.jsonl`,
then `case_evolution/supplementary/`, `investigation_conclusion.json`,
`fault_reflector_output.json`, `applied_updates.json`, and
`case_evolution_summary.json`. The two-case smoke tests protocol and
traceability, not an accuracy improvement; frozen-bank ablations remain necessary.
