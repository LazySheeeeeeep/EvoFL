"""Acquire and audit pre-issue Git history for a RepoMem-Loc acquisition smoke."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/external_fl_baseline_preflight_20260924/repomem_loc"


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, timeout=600)


def prepare(case: dict, clone: bool = False) -> dict:
    repo = case["repo"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid public repository name")
    target = ROOT / "runs/external_baseline_runtime/repomem_history" / (repo.replace("/", "__") + ".git")
    url = f"https://github.com/{repo}.git"
    if not target.exists():
        if not clone:
            raise ValueError("History absent; use --clone to acquire public Git history")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--bare", "--filter=blob:none", "--single-branch", "--no-tags", url, str(target)],
                       check=True, timeout=600)
    if git(target, "remote", "get-url", "origin").strip() != url:
        raise ValueError("Unexpected history remote")
    if git(target, "rev-parse", "--is-shallow-repository").strip() != "false":
        raise ValueError("Incomplete shallow history")
    base_commit = case["base_commit"]
    git(target, "cat-file", "-e", f"{base_commit}^{{commit}}")
    cutoff = int(datetime.fromisoformat(case["created_at"].replace("Z", "+00:00")).timestamp())
    commits = git(target, "log", base_commit, "--max-count=7000", "--format=%H%x1f%ct%x1f%s").splitlines()
    rows, excluded = [], []
    target_number = case["instance_id"].rsplit("-", 1)[-1]
    for line in commits:
        sha, timestamp, title = line.split("\x1f", 2)
        if int(timestamp) >= cutoff or re.search(rf"(?<!\d)#{re.escape(target_number)}(?!\d)", title):
            excluded.append(sha)
            continue
        rows.append({"sha": sha, "committed_at": int(timestamp), "title": title})
    # Paths are metadata, so the partial clone need not download source blobs yet.
    files = git(target, "log", base_commit, "--max-count=7000", "--format=COMMIT:%H", "--name-only").splitlines()
    allowed = {row["sha"] for row in rows}
    current, counts = None, Counter()
    for line in files:
        if line.startswith("COMMIT:"):
            current = line[7:]
        elif line and current in allowed:
            counts[line] += 1
    output = BASE / case["instance_id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "commit_catalog.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"instance_id": case["instance_id"], "repo": repo, "base_commit": base_commit,
              "issue_created_at": case["created_at"], "ancestor_window": len(commits),
              "eligible_commit_count": len(rows), "excluded_count": len(excluded),
              "top_200_modified_paths": counts.most_common(200), "model_calls": 0,
              "status": "history_metadata_audited_not_inference_ready",
              "remaining": ["linked issue retrieval with timestamp/text-overlap audit",
                            "commit patch retrieval", "base-snapshot functionality summaries",
                            "BM25 tools and LocAgent integration"],
              "note": "Only ancestor and timestamp checks completed; linked-issue overlap filtering is still required."}
    (output / "history_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "top_200_modified_paths"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=ROOT / "runs/rq1_temporal_deepseek_20260915/materialized/psf__requests-1376.json")
    parser.add_argument("--clone", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.case.read_text())
    public = {key: source[key] for key in ("instance_id", "repo", "base_commit", "created_at")}
    print(json.dumps(prepare(public, args.clone), indent=2))
