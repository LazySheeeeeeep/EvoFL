"""Build the author's LocAgent graph from a checksum-verified base archive."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import pickle
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "runs/external_baseline_sources/LocAgent"
BASE = ROOT / "runs/external_fl_baseline_preflight_20260924"


def prepare(case_id: str, source: Path, output: Path) -> dict:
    if Path(case_id).name != case_id or "\\" in case_id:
        raise ValueError("Invalid case ID")
    materialized = json.loads((source / "materialized" / f"{case_id}.json").read_text())
    base_commit = materialized.get("base_commit")
    if base_commit is None:
        evaluation_cases = source / "evaluation_cases.json"
        if not evaluation_cases.is_file():
            raise ValueError("Base commit missing from materialized metadata")
        matching = [case for case in json.loads(evaluation_cases.read_text(encoding="utf-8"))
                    if case.get("instance_id") == case_id]
        if len(matching) != 1 or not matching[0].get("base_commit"):
            raise ValueError("Base commit not uniquely available for case")
        base_commit = matching[0]["base_commit"]
    archive = source / "sources" / f"{case_id}.tar.gz"
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    if archive_hash != materialized["original_archive_sha256"]:
        raise ValueError("Archive checksum mismatch")
    sys.path.insert(0, str(AUTHOR))
    from dependency_graph.build_graph import build_graph, VERSION

    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="evolutefl-locagent-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive, "r:gz") as members:
            archive_roots = {PurePosixPath(member.name).parts[0] for member in members}
            if len(archive_roots) != 1:
                raise ValueError("Expected a single archive root")
            archive_root = next(iter(archive_roots))
            if not base_commit.startswith(archive_root.split("-")[-1]):
                raise ValueError("Archive revision mismatch")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive path")
                if not member.isfile():
                    continue
                destination = root.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with members.extractfile(member) as stream:
                    destination.write_bytes(stream.read())
        graph = build_graph(str(root / archive_root), global_import=True)
        with (output / f"{case_id}.pkl").open("wb") as stream:
            pickle.dump(graph, stream)
    report = {"status": "completed", "instance_id": case_id,
              "base_commit": base_commit, "archive_sha256": archive_hash,
              "graph_version": VERSION, "nodes": len(graph), "edges": graph.number_of_edges(),
              "node_types": dict(Counter(data.get("type") for _, data in graph.nodes(data=True))),
              "elapsed_seconds": round(time.monotonic() - started, 3), "model_calls": 0}
    (output / "graph_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="psf__requests-1376")
    parser.add_argument("--source", type=Path, default=ROOT / "runs/rq1_temporal_deepseek_20260915")
    parser.add_argument("--output", type=Path, default=BASE / "locagent_acquisition_graph")
    args = parser.parse_args()
    print(json.dumps(prepare(args.case_id, args.source, args.output), indent=2))
