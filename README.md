# EvoFL: Self-Evolving Fault Localization via Retrospective-Analysis-based Skill Distillation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL-orange)
![Runner](https://img.shields.io/badge/Runner-run__main.py-green)

This document describes how to set up EvoFL, connect the required datasets and
API services, and run the main experiment.

## 🛠️ Environment Setup

We recommend running EvoFL in Linux or WSL with Python 3.10 or newer. The main
runner uses `fcntl`, so native Windows Python is not supported.

### 1. Create the environment

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

### 2. Install EvoFL

```bash
cd /mnt/d/projects/EvoluteFL
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install requests
```

### 3. Verify the environment

```bash
python3 --version
python -c "import requests; print('requests', requests.__version__)"
PYTHONPATH=src python3 scripts/run_main.py --help
```

## 📥 Dataset Access

The repository does not redistribute the preprocessed historical run data.
Before running the experiment, place the following two directories under
`runs/`:

```text
runs/
├── rq1_temporal_deepseek_20260915/
└── rq1_expanded400_eval500_deepseek_20260922/
```

### Required inputs for `rq1_temporal_deepseek_20260915`

The historical base run must contain:

```text
protocol.json
config.json
frozen_skills.jsonl
training_complete.json
evaluation_candidates.json
comparison_summary.json
pr_metadata/
sources/
```

### Required inputs for `rq1_expanded400_eval500_deepseek_20260922`

The audit cache must contain:

```text
original_training.json
original_evaluation.json
training_candidates.json
evaluation_extension_order.json
pr_metadata/
function_audit/
source_files/
```

The original issues and repair patches are derived from the SWE-bench family:

```text
princeton-nlp/SWE-bench
princeton-nlp/SWE-bench_Lite
princeton-nlp/SWE-bench_Verified
```

EvoFL reads the frozen EvoFL manifests under `runs/`; it does not download or
rebuild those manifests automatically. The runner validates the input hashes
and stops if a required file is missing or changed.

## 🔑 API Access

### DeepSeek API

Export the official DeepSeek credential for the phases that call the model:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

The main runner uses:

```text
Provider: DeepSeek
Model:    deepseek-v4-flash
Base URL: https://api.deepseek.com
```

### GitHub API

EvoFL can reuse cached PR metadata and source files. If a required metadata
entry is not cached, export a GitHub token to avoid low anonymous rate limits:

```bash
export GITHUB_TOKEN="your_github_token"
```

`GH_TOKEN` is also supported.

### Credential Safety

Do not place API keys in `config/`, prompts, scripts, or run outputs. All
credentials are read from process environment variables.

## 🚀 How to Run

### Quick Start

Set the environment and run the complete main experiment:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export GITHUB_TOKEN="your_github_token"   # optional
export PYTHONPATH=src

python3 scripts/run_main.py \
  --phase full \
  --workers 4
```

The complete run performs:

```text
prepare
  -> rescore
  -> audit
  -> 200 additional acquisition and evolution cases
  -> 500 frozen-bank evaluation cases
```

### Phase-by-Phase Execution

#### 1. Prepare frozen inputs

This phase validates historical data, copies prompts and manifests, and writes
the hashed protocol. It does not call the model.

```bash
PYTHONPATH=src python3 scripts/run_main.py --phase prepare --workers 4
```

#### 2. Recompute historical ground truth

This phase recomputes the historical function-level labels without model calls.

```bash
PYTHONPATH=src python3 scripts/run_main.py --phase rescore --workers 4
```

#### 3. Audit and freeze the case split

This phase performs temporal, duplicate, and function-eligibility checks and
freezes the additional acquisition and evaluation manifests.

```bash
PYTHONPATH=src python3 scripts/run_main.py --phase audit --workers 4
```

#### 4. Run acquisition, evolution, and evaluation

```bash
PYTHONPATH=src python3 scripts/run_main.py --phase full --workers 4
```

`--workers` accepts values from `1` to `8`. Reduce concurrency when API,
Docker, or WSL resources are constrained.

## 📊 Outputs

All main experiment outputs are written to:

```text
runs/main_experiment/
```

The most important result files are:

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

Per-case localization and reflection artifacts are stored under:

```text
runs/main_experiment/training/<instance_id>/
runs/main_experiment/with_skill/cases/<instance_id>/
```

The final `comparison_summary.json` reports function-level
`Top-1`, `Top-3`, `Top-5`, and `MRR`.

## ✅ Resume and Reproducibility

- Explorer results are reused through `result.json`.
- Evolution results are reused through
  `case_evolution/case_evolution_summary.json`.
- Evaluation results are reused through `paired/<instance_id>.json`.
- The frozen Skill bank is hash-checked before and during evaluation.
- Prompt, configuration, source, and manifest hashes are stored in
  `protocol.json`.
- Ground-truth functions are used for training and scoring, not injected into
  the localization input.

