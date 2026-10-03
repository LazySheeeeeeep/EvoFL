"""Freeze a patch-free, cross-repository CoSIL pilot from the RQ1 evaluation set."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

if __package__:
    from .prepare_cosil_acquisition_smoke import PUBLIC_FIELDS, prepare
else:
    from prepare_cosil_acquisition_smoke import PUBLIC_FIELDS, prepare


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
PUBLIC = ROOT / "runs/external_fl_baseline_preflight_20260924/evaluation_public.jsonl"
OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/cosil_pilot"


def select_cases(rows: list[dict], limit: int = 10) -> list[dict]:
    if limit < 1:
        raise ValueError("Pilot limit must be positive")
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate public case ID")
    if any(set(row) != set(PUBLIC_FIELDS) or not all(row.values()) for row in rows):
        raise ValueError("Pilot input must contain only complete public fields")
    selected = []
    seen_repos = set()
    for row in rows:
        if row["repo"] not in seen_repos:
            selected.append(row)
            seen_repos.add(row["repo"])
            if len(selected) == limit:
                return selected
    selected_ids = {row["instance_id"] for row in selected}
    selected.extend(row for row in rows if row["instance_id"] not in selected_ids)
    return selected[:limit]


def audit(run: Path = RUN, public: Path = PUBLIC, output: Path = OUTPUT,
          limit: int = 10, materialize_ready: bool = False) -> dict:
    rows = [json.loads(line) for line in public.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    selected = select_cases(rows, limit)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected_cases.jsonl"
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                     for row in selected), encoding="utf-8")
    ready = [row for row in selected if (run / "sources" / f"{row['instance_id']}.tar.gz").is_file()
             and (run / "materialized" / f"{row['instance_id']}.json").is_file()]
    prepared = []
    if materialize_ready:
        structures = output / "repo_structures"
        structures.mkdir(exist_ok=True)
        for row in ready:
            case_id = row["instance_id"]
            case_output = output / "prepared" / case_id
            prepare(case_id, source=run, output=case_output, public_manifest=public)
            shutil.copyfile(case_output / "repo_structures" / f"{case_id}.json",
                            structures / f"{case_id}.json")
            prepared.append(row)
        (output / "ready_cases.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared),
            encoding="utf-8")
    report = {
        "selected_count": len(selected),
        "distinct_repos": len({row["repo"] for row in selected}),
        "selected_ids": [row["instance_id"] for row in selected],
        "ready_ids": [row["instance_id"] for row in ready],
        "not_yet_materialized_ids": [row["instance_id"] for row in selected if row not in ready],
        "prepared_count": len(prepared),
        "public_manifest_sha256": hashlib.sha256(public.read_bytes()).hexdigest(),
        "selected_manifest_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "note": "Offline public-input preparation only; no model calls or baseline scores.",
    }
    (output / "pilot_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--materialize-ready", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.public, args.output, args.limit,
                           args.materialize_ready), ensure_ascii=False, indent=2))
