# EvoFL: Self-Evolving Fault Localization via Retrospective-Analysis-based Skill Distillation

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

## 🔑 API Access

EvoFL uses an OpenAI-compatible Chat Completions API. You can connect any
provider that implements the same interface.

Set the API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Set the API endpoint when it is not the default OpenAI endpoint:

```bash
export OPENAI_BASE_URL="https://your-provider.example/v1"
```

Set the model name used by the provider:

```bash
export OPENAI_MODEL="your-model-name"
```

Some providers support tool calling but reject an explicit `tool_choice`
parameter. In that case, disable it:

```bash
export EVOLUTEFL_SUPPORTS_TOOL_CHOICE=false
```

The endpoint must accept OpenAI-compatible requests at:

```text
{OPENAI_BASE_URL}/chat/completions
```

Do not commit API keys or write them into configuration files.

## 🚀 Quick Start

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

All experiment outputs are written to:

```text
runs/main_experiment/
```

Completed cases are reused automatically when the runner is restarted.
