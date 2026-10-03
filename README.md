# EvoFL: Self-Evolving Fault Localization via Retrospective-Analysis-based Skill Distillation

EvoFL is a function-level fault localization framework that learns reusable
investigation skills from completed localization trajectories. Instead of
memorizing patch functions or answer-like summaries, EvoFL reconstructs the
investigation process, identifies evidence gaps through a read-only
supplementary investigation, and distills executable investigation lessons
into a versioned Fault Skill bank.

The active workflow uses Fault Skills only. Project Skills and Strategy Skills
are not used by the main experiment.

## Method Overview

```text
Issue + repository
  -> initial source exploration
  -> load one Fault Skill (or continue without a Skill)
  -> evidence-guided localization
  -> finish_localization with up to five ranked functions

Completed trajectory
  -> reconstruct the actual investigation
  -> perform read-only supplementary investigation
  -> identify evidence gaps and supported process lessons
  -> independently select the Fault family and update target
  -> create / rewrite / preserve / no_update
```

During localization, the Explorer retains the exact tool observations and a
complete action index. During evolution, original `obs-` observations and
supplementary `supp-` observations remain separate. A resolved investigation
lesson must be anchored to reviewed evidence; unresolved investigations do not
modify the Skill bank.

Fault Skills contain a family, title, trigger, and an unordered list of
investigation knowledge strings. Skill cards are versioned and stored as JSONL.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/evolutefl/` | Core Explorer, trajectory logging, retrospective investigation, Reflector, Skill bank, and evaluation code. |
| `scripts/run_main.py` | Self-contained main experiment runner for chronological acquisition, Skill evolution, and frozen-bank evaluation. |
| `config/evolutefl.global.json` | Active EvoFL configuration, including model settings and the seven runtime prompt paths. |
| `prompt_records/` | The seven prompts required by the active v5 localization and evolution workflow. |
| `skill_pools/skill_bank_v5/` | Local Fault Skill bank location. The bank is generated during runs and is not tracked. |

## Requirements

- Linux or WSL. The main runner uses `fcntl` for process-safe run and case locks.
- Python `>=3.10`.
- Network access for the configured LLM endpoint and, when metadata is not
  cached, GitHub API/archive requests.
- `DEEPSEEK_API_KEY` in the process environment for model-backed phases.
- Optional `GITHUB_TOKEN` or `GH_TOKEN` to increase GitHub API rate limits.
- Two historical local run directories used as frozen inputs:

```text
runs/rq1_temporal_deepseek_20260915/
runs/rq1_expanded400_eval500_deepseek_20260922/
```

These directories contain the historical manifests, metadata, source caches,
frozen initial Skill bank, and audit candidates. They are local experiment
inputs and are intentionally excluded from Git. The main runner fails closed
when a required frozen input is missing or changed.

## Installation

```bash
cd /mnt/d/projects/EvoluteFL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install requests
```

Credentials must be supplied only through environment variables:

```bash
export DEEPSEEK_API_KEY='...'
export GITHUB_TOKEN='...'   # optional
export PYTHONPATH=src
```

Do not place credentials in configuration files or run artifacts.

## Main Experiment

The runner writes to:

```text
runs/main_experiment/
```

The phases are:

| Phase | Model calls | Description |
|---|---:|---|
| `prepare` | No | Validates historical inputs, copies frozen manifests and prompts, and writes a hashed protocol. |
| `rescore` | No | Recomputes the historical function ground truth without running the model. |
| `audit` | No model calls | Performs temporal, duplicate, and function-eligibility auditing, then freezes 200 additional acquisition cases and 500 evaluation cases. |
| `full` | Yes | Runs the preparation checks followed by 200 additional acquisition/evolution cases and 500 frozen-bank evaluation cases. |

Run only preparation:

```bash
PYTHONPATH=src python3 scripts/run_main.py --phase prepare --workers 4
```

Run the complete experiment:

```bash
DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" \
PYTHONPATH=src \
python3 scripts/run_main.py --phase full --workers 4
```

`--workers` accepts values from `1` to `8`. Lower concurrency is useful when
the LLM endpoint or Docker/WSL resources are constrained.

The runner is checkpointed:

- Completed Explorer cases are reused through `result.json`.
- Completed evolution cases are reused through
  `case_evolution/case_evolution_summary.json`.
- Completed evaluation cases are reused through `paired/<instance_id>.json`.
- The frozen Skill bank hash is checked before and during evaluation.

## Outputs

Important files under `runs/main_experiment/` include:

```text
protocol.json
config.json
progress.json
training_summary.json
frozen_skills.jsonl
training_complete.json
evaluation_cases.json
paired/<instance_id>.json
comparison_summary.json
```

Per-case localization artifacts include:

```text
run_context.json
initial_payload.json
observations.jsonl
trajectory.jsonl
investigation_index.json
fault_skill_search.json
result.json
```

Per-case evolution artifacts include:

```text
case_evolution/supplementary/
investigation_conclusion.json
evolution_queries.json
fault_reflector_output.json
applied_updates.json
case_evolution_summary.json
```

The final `comparison_summary.json` contains function-level
`Top-1`, `Top-3`, `Top-5`, and `MRR`, together with status and Skill-loading
statistics.

## Reproducibility Notes

- Prompts, configuration, source files, and frozen manifests are hashed in
  `protocol.json`.
- Evaluation uses a frozen Skill bank and a predeclared evaluation manifest.
- Ground-truth functions are used during training and scoring, not injected
  into the Explorer input.
- The repository contains only the main experiment entry point; external
  baselines, ablations, diagnostics, and historical run outputs remain local
  and are excluded from Git.
