"""Cache the SWE-Exp author's E5 model in WSL without touching active runs."""
from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi, snapshot_download

from swe_exp_loc_retrieval import MODEL_ID


def prepare(*, offline: bool = False) -> dict:
    if offline:
        path = snapshot_download(repo_id=MODEL_ID, local_files_only=True)
        return {"status": "cached", "model_id": MODEL_ID, "path": path}
    info = HfApi().model_info(MODEL_ID)
    files = {item.rfilename for item in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    if weights not in files:
        raise ValueError("No compatible E5 model weight file in repository")
    path = snapshot_download(repo_id=MODEL_ID, revision=info.sha,
                             allow_patterns=[weights, "config.json", "tokenizer.json",
                                             "tokenizer_config.json", "special_tokens_map.json",
                                             "sentencepiece.bpe.model", "*.model"])
    return {"status": "cached", "model_id": MODEL_ID, "revision": info.sha,
            "weights": weights, "path": path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(offline=args.offline), indent=2))
