# EvoFL: Self-Evolving Fault Localization via Retrospective-Analysis-based Skill Distillation

## 📌 Introduction

EvoFL converts both successful and failed localization trajectories into reusable localization skills through ground-truth-guided retrospective analysis.

- **Successful trajectories:** EvoFL removes unreferenced observations, dead-end exploration, and redundant re-inspection, preserving only the diagnostic chain relevant to the ground-truth faulty entities.
- **Failed trajectories:** EvoFL retrospectively reflects on intermediate judgments and salvages valid analyses that narrow the investigation toward the fault, even when the final localization is incorrect.

![EvoFL workflow](assets/workflow.png)

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
