"""Resume audited CoSIL structure preparation without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_cosil_acquisition_smoke import prepare
from prepare_cosil_pilot import PUBLIC, RUN, select_cases

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/external_fl_baseline_preflight_20260924"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def run(output: Path, limit: int = 500) -> dict:
    rows = select_cases([json.loads(line) for line in PUBLIC.read_text().splitlines() if line], limit)
    output.mkdir(parents=True, exist_ok=True)
    structures = output / "repo_structures"
    structures.mkdir(exist_ok=True)
    (output / "selected_cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"selected_count": len(rows), "prepared_count": 0, "model_calls": 0,
              "public_manifest_sha256": digest(PUBLIC), "cases": []}
    ready = []
    for row in rows:
        case_id = row["instance_id"]
        archive = RUN / "sources" / f"{case_id}.tar.gz"
        materialized = RUN / "materialized" / f"{case_id}.json"
        entry = {"instance_id": case_id, "status": "source_pending"}
        try:
            if archive.is_file() and materialized.is_file():
                archive_hash = digest(archive)
                if json.loads(materialized.read_text())["original_archive_sha256"] != archive_hash:
                    raise ValueError("Archive checksum mismatch")
                destination = structures / f"{case_id}.json"
                candidates = [output / "prepared" / case_id]
                candidates += [BASE / name / "prepared" / case_id
                               for name in ("cosil_pilot_30", "cosil_pilot")]
                prepared = None
                for candidate in candidates:
                    metadata = candidate / "preparation.json"
                    source = candidate / "repo_structures" / f"{case_id}.json"
                    if metadata.is_file() and source.is_file():
                        manifest = json.loads(metadata.read_text())
                        structure = json.loads(source.read_text())
                        public = json.loads((candidate / "case.jsonl").read_text())
                        if (manifest.get("source_archive_sha256") == archive_hash
                                and public == row and structure.get("base_commit") == row["base_commit"]):
                            prepared = candidate
                            break
                if prepared is None:
                    prepared = output / "prepared" / case_id
                    prepare(case_id, source=RUN, output=prepared, public_manifest=PUBLIC)
                source = prepared / "repo_structures" / f"{case_id}.json"
                local_prepared = output / "prepared" / case_id
                local_prepared.mkdir(parents=True, exist_ok=True)
                for name in ("case.jsonl", "preparation.json"):
                    target = local_prepared / name
                    if not target.exists():
                        os.link(prepared / name, target)
                if destination.exists():
                    if digest(destination) != digest(source):
                        raise ValueError("Existing structure differs from audited source")
                else:
                    os.link(source, destination)
                local_structures = local_prepared / "repo_structures"
                local_structures.mkdir(exist_ok=True)
                if not (local_structures / source.name).exists():
                    os.link(source, local_structures / source.name)
                ready.append(row)
                entry.update(status="ready", archive_sha256=archive_hash,
                             structure_sha256=digest(source), prepared=str(prepared))
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["cases"].append(entry)
        report["prepared_count"] = len(ready)
        (output / "ready_cases.jsonl").write_text("".join(json.dumps(item) + "\n" for item in ready))
        temporary = output / "pilot_readiness.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "pilot_readiness.json")
        print(f"{len(report['cases'])}/{len(rows)} {case_id}: {entry['status']}", flush=True)
    return report


def repair_links(output: Path) -> dict:
    report = json.loads((output / "pilot_readiness.json").read_text())
    selected = [json.loads(line) for line in (output / "selected_cases.jsonl").read_text().splitlines()]
    if len(report["cases"]) != len(selected) or report["prepared_count"] != len(selected):
        raise ValueError("Preparation is incomplete")
    repaired = 0
    for row, entry in zip(selected, report["cases"]):
        if entry["status"] != "ready" or entry["instance_id"] != row["instance_id"]:
            raise ValueError("Preparation case mismatch")
        candidate = Path(entry["prepared"]).resolve(strict=True)
        if not candidate.is_relative_to(ROOT.resolve()):
            raise ValueError("Prepared path outside workspace")
        case_id = row["instance_id"]
        if json.loads((candidate / "case.jsonl").read_text()) != row:
            raise ValueError("Prepared public input changed")
        if json.loads((candidate / "preparation.json").read_text())["source_archive_sha256"] != entry["archive_sha256"]:
            raise ValueError("Prepared source hash changed")
        source = candidate / "repo_structures" / f"{case_id}.json"
        if json.loads(source.read_text())["base_commit"] != row["base_commit"]:
            raise ValueError("Prepared base commit changed")
        destination = output / "prepared" / case_id
        (destination / "repo_structures").mkdir(parents=True, exist_ok=True)
        for original, target in ((candidate / "case.jsonl", destination / "case.jsonl"),
                                 (candidate / "preparation.json", destination / "preparation.json"),
                                 (source, destination / "repo_structures" / source.name)):
            if target.exists():
                if not os.path.samefile(original, target):
                    raise ValueError("Conflicting existing prepared file")
            else:
                os.link(original, target)
                repaired += 1
    return {"status": "complete", "selected_count": len(selected), "links_created": repaired}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "cosil_full500")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--repair-links-only", action="store_true")
    args = parser.parse_args()
    if args.repair_links_only:
        print(json.dumps(repair_links(args.output.resolve())))
    else:
        run(args.output.resolve(), args.limit)
