# EvoFL: Self-Evolving Fault Localization via Retrospective-Analysis-based Skill Distillation

## 📌 Introduction

EvoFL converts both successful and failed localization trajectories into reusable localization skills through ground-truth-guided retrospective analysis.

- **Successful trajectories:** EvoFL removes unreferenced observations, dead-end exploration, and redundant re-inspection, preserving only the diagnostic chain relevant to the ground-truth faulty entities.
- **Failed trajectories:** EvoFL retrospectively reflects on intermediate judgments and salvages valid analyses that narrow the investigation toward the fault, even when the final localization is incorrect.

![EvoFL workflow](assets/workflow.png)

## 📥 Inputs and Outputs

### Input: SWE-bench

EvoFL takes an issue description and a buggy repository snapshot as input. To
connect SWE-bench, normalize each instance to the following fields:

| Field | Description |
|---|---|
| `instance_id` | SWE-bench instance identifier used as the stable case key. |
| `repo` | Repository name in `owner/repository` form. |
| `base_commit` | Buggy snapshot commit used to materialize the repository. |
| `problem_statement` | Issue description visible to the localization agent. |
| `created_at` | Issue creation time used by the chronological split. |
| `patch` | Developer repair patch. It is used for experience learning and scoring, not injected into Explorer. |
| `function_ground_truth` | Functions modified by the developer patch on the buggy snapshot. |

The paper uses three curated SWE-bench sources:

```text
princeton-nlp/SWE-bench
princeton-nlp/SWE-bench_Lite
princeton-nlp/SWE-bench_Verified
```

Instances are split chronologically rather than randomly:

```text
Experience acquisition: created and merged before 2020-01-01
Temporal buffer:        2020-01-01 to 2020-12-31
Evaluation:             created on or after 2021-01-01
```

After normalization, place the frozen EvoFL manifests under the run directory
expected by `run_main.py`:

```text
runs/rq1_temporal_deepseek_20260915/
runs/rq1_expanded400_eval500_deepseek_20260922/
```

Repository snapshots are materialized from `repo` and `base_commit`. The patch
and `function_ground_truth` remain hidden from the localization process and are
used only during retrospective learning and final scoring.

### Outputs

All outputs are written under:

```text
runs/main_experiment/
```

```text
runs/main_experiment/
├── protocol.json
├── config.json
├── bootstrap_skills.jsonl
├── frozen_skills.jsonl
├── training/
│   └── <instance_id>/
│       ├── explorer/
│       │   ├── trajectory.jsonl
│       │   ├── observations.jsonl
│       │   ├── investigation_index.json
│       │   ├── fault_skill_search.json
│       │   └── result.json
│       ├── evolution/
│       │   ├── supplementary/
│       │   ├── investigation_conclusion.json
│       │   ├── fault_reflector_output.json
│       │   ├── applied_updates.json
│       │   └── case_evolution_summary.json
│       └── skills.jsonl
├── with_skill/
│   └── cases/
│       └── <instance_id>/
│           ├── trajectory.jsonl
│           ├── observations.jsonl
│           ├── fault_skill_search.json
│           └── result.json
├── paired/
│   └── <instance_id>.json
└── comparison_summary.json
```

| Output | Content |
|---|---|
| `trajectory.jsonl` | Chronological reasoning, tool actions, observations, and Skill-loading events. |
| `observations.jsonl` | Exact tool responses delivered to the agent. |
| `investigation_index.json` | Compact investigation timeline with observation references. |
| `fault_skill_search.json` | Fault-family routing, selector output, and validator decision. |
| `result.json` | Final ranked suspicious functions and localization status. |
| `skills.jsonl` | Case-local Skill Bank transaction after create, rewrite, preserve, or no-update. |
| `frozen_skills.jsonl` | Frozen Skill Bank used for evaluation on the 500 held-out instances. |
| `paired/<instance_id>.json` | Predictions and metrics for each evaluated function. |
| `comparison_summary.json` | Aggregate Top-1, Top-3, Top-5, and MRR results. |

Ranked suspicious functions use the following identity format:

```text
relative/path/to/file.py::Class.method
```

## 🛠️ Environment Setup

EvoFL should be run in Linux or WSL with Python 3.10 or newer. The main runner
uses `fcntl`, so native Windows Python is not supported.

### 1. Clone the repository

```bash
git clone <repository-url> EvoFL
cd EvoFL
```

### 2. Create an environment

Using Conda:

```bash
conda create -n evofl python=3.10 -y
conda activate evofl
```

Or using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install requests
```

### 4. Check the installation

```bash
PYTHONPATH=src python3 scripts/run_main.py --help
```

## 🚀 Quick Start

### 1. Configure the API

EvoFL uses an OpenAI-compatible Chat Completions API.

- `OPENAI_API_KEY`: API credential. This must be set before running the model-backed phases.
- `OPENAI_BASE_URL`: API base URL. Set this when using a provider other than the default OpenAI endpoint.
- `OPENAI_MODEL`: model identifier used by the provider.
- `EVOLUTEFL_SUPPORTS_TOOL_CHOICE=false`: optional compatibility switch for providers that support tools but reject an explicit `tool_choice` parameter.
- `GITHUB_TOKEN` or `GH_TOKEN`: optional GitHub credential used when repository metadata or source files are not already cached.

The configured endpoint must accept OpenAI-compatible requests at:

```text
{OPENAI_BASE_URL}/chat/completions
```

Do not write API keys or tokens into configuration files.

### 2. Run the experiment

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://your-provider.example/v1"
export OPENAI_MODEL="your-model-name"
export PYTHONPATH=src

python3 scripts/run_main.py \
  --phase full \
  --workers 4
```

`--workers` accepts values from `1` to `8`.

### 3. Output

All experiment outputs are written to:

```text
runs/main_experiment/
```

Completed cases are reused automatically when the runner is restarted.
