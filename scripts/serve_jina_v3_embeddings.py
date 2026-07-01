from __future__ import annotations

import argparse
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel


DEFAULT_MODEL = "jinaai/jina-embeddings-v3"


class EmbedRequest(BaseModel):
    texts: list[str]
    task: str | None = "retrieval.passage"
    model: str | None = DEFAULT_MODEL
    truncate_dim: int | None = 1024


def build_app(*, model_name: str, device: str) -> FastAPI:
    app = FastAPI(title="EvoluteFL local Jina-v3 embedding server")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": model_name, "device": device}

    @app.post("/embed")
    def embed(request: EmbedRequest) -> dict[str, Any]:
        task = request.task or "retrieval.passage"
        with torch.inference_mode():
            embeddings = model.encode(
                request.texts,
                task=task,
                truncate_dim=request.truncate_dim,
            )
        if hasattr(embeddings, "tolist"):
            vectors = embeddings.tolist()
        else:
            vectors = [list(vector) for vector in embeddings]
        return {"model": model_name, "task": task, "embeddings": vectors}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local Jina-v3 embeddings for EvoluteFL.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    app = build_app(model_name=args.model, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
