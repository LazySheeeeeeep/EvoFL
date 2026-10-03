"""SWE-Exp-style E5 recall over acquisition issue types for FL adaptation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_swe_exp_loc import DESTINATION
from prepare_rq1_memory_baselines import write_json


MODEL_ID = "intfloat/multilingual-e5-large-instruct"
TASK = ("Given the prior issues, your task is to analyze a current issue's "
        "problem statement and select the most relevant prior issue that could help resolve it.")


def issue_type_text(row: dict) -> str:
    return f"issue_type: {row['issue_type']} \ndescription: {row['description']}"


def query_text(row: dict) -> str:
    return f"Instruct: {TASK}\nQuery: {issue_type_text(row)}"


def read_experiences(directory: Path, *, include_skill_conditioned: bool = False) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not all(isinstance(row.get(key), str) and row[key].strip()
                   for key in ("instance_id", "repo", "issue_type", "description")):
            raise ValueError("Invalid SWE-Exp-Loc experience: " + str(path))
        if row.get("source_skill_conditioned") and not include_skill_conditioned:
            continue
        rows.append(row)
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate SWE-Exp-Loc memory")
    return rows


def fingerprint(rows: list[dict]) -> str:
    relevant = [{"instance_id": row["instance_id"], "repo": row["repo"],
                 "issue_type": row["issue_type"], "description": row["description"]}
                for row in rows]
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("E5 embedding dimension mismatch")
    return sum(x * y for x, y in zip(a, b))


def recall(*, query: dict, current_repo: str, experiences: list[dict],
           indexed: list[dict], query_vector: list[float], k: int = 10) -> list[dict]:
    if len(experiences) != len(indexed):
        raise ValueError("E5 index and experience bank have different sizes")
    if not query.get("issue_type") or not query.get("description"):
        raise ValueError("Query needs an issue type and description")
    scored = []
    for experience, embedding in zip(experiences, indexed):
        if experience["instance_id"] != embedding["instance_id"]:
            raise ValueError("E5 index identity mismatch")
        if experience["repo"] == current_repo:
            continue
        score = 100 * cosine(query_vector, embedding["vector"])
        scored.append((score, experience["instance_id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"instance_id": cid, "score": round(score, 6)} for score, cid in scored[:k]]


class E5Encoder:
    """Author-equivalent mean pooling, normalization, and query instruction."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        output = []
        torch = self.torch
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[start:start + batch_size], max_length=1024,
                                   padding=True, truncation=True, return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                tokens = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"][..., None]
                pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
            output.extend(vectors.cpu().tolist())
        return output


def build_index(experiences: list[dict], path: Path, *, encoder=None) -> dict:
    if not experiences:
        raise ValueError("Cannot index an empty experience bank")
    stamp = fingerprint(experiences)
    manifest_path = path.with_suffix(".manifest.json")
    if path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("model_id") == MODEL_ID and manifest.get("experience_fingerprint") == stamp
                and manifest.get("index_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()):
            return {"status": "cached", "count": len(experiences), "path": str(path)}
    encoder = encoder or E5Encoder()
    vectors = encoder.encode([issue_type_text(row) for row in experiences])
    if len(vectors) != len(experiences):
        raise ValueError("E5 encoder returned the wrong number of vectors")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"instance_id": row["instance_id"], "vector": vector}) + "\n"
                            for row, vector in zip(experiences, vectors)), encoding="utf-8")
    write_json(manifest_path, {"model_id": MODEL_ID, "experience_fingerprint": stamp,
                               "count": len(experiences), "index_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"status": "built", "count": len(experiences), "path": str(path)}


def load_index(experiences: list[dict], path: Path) -> list[dict]:
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID or manifest["experience_fingerprint"] != fingerprint(experiences):
        raise ValueError("Stale E5 index")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["index_sha256"]:
        raise ValueError("Changed E5 index")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, default=DESTINATION / "experiences")
    parser.add_argument("--index", type=Path, default=DESTINATION / "e5_issue_types.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-skill-conditioned", action="store_true")
    args = parser.parse_args()
    experiences = read_experiences(args.experiences,
        include_skill_conditioned=args.include_skill_conditioned)
    report = ({"status": "dry_run", "experience_count": len(experiences), "model_id": MODEL_ID}
              if args.dry_run else build_index(experiences, args.index))
    print(json.dumps(report, indent=2))
