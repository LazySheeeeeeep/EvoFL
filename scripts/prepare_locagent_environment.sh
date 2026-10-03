#!/usr/bin/env bash
set -euo pipefail
environment_path="/home/lql/.venvs/evolutefl-locagent"
python3 -m venv "$environment_path"
"$environment_path/bin/python" -m pip install --disable-pip-version-check 'pip==25.3'
"$environment_path/bin/python" -m pip install --disable-pip-version-check \
  'networkx==3.4.2' 'matplotlib==3.9.2'
"$environment_path/bin/python" -m pip install --disable-pip-version-check \
  'networkx==3.4.2' 'matplotlib==3.9.2' 'toml==0.10.2' \
  'numpy==1.26.4' 'libcst==1.5.0' 'datasets==3.1.0' \
  'litellm==1.63.14' 'httpx==0.27.2' 'openai==1.66.5' \
  'huggingface-hub==0.26.2' 'tokenizers==0.20.3' \
  'bm25s==0.2.3' 'faiss-cpu==1.8.0' 'PyStemmer==2.2.0.3' \
  'llama-index-core==0.11.22' 'llama-index-retrievers-bm25==0.4.0' \
  'llama-index-embeddings-openai==0.2.5' 'llama-index-llms-openai==0.2.16' \
  'llama-index-embeddings-azure-openai==0.2.5' \
  'ipython==8.29.0' 'rapidfuzz==3.10.1' 'unidiff==0.7.5' \
  'tree-sitter==0.21.3' 'tree-sitter-languages==1.10.2' \
  'llama-index-readers-file==0.2.2'
"$environment_path/bin/python" -m pip install --disable-pip-version-check \
  'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
"$environment_path/bin/python" -m pip check
