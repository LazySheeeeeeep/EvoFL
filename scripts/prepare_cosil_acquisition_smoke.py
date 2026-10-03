"""Build one CoSIL acquisition smoke input from an audited base-source tarball.

This reads the archive without cloning a repository or exposing its patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "runs/rq1_temporal_deepseek_20260915"
EXTERNAL = ROOT / "runs/external_baseline_sources/CoSIL"
OUTPUT = ROOT / "runs/external_fl_baseline_preflight_20260924/cosil_acquisition_smoke"
PUBLIC_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")


def load_public_case(case_id: str, source: Path, public_manifest: Path | None) -> dict:
    if not case_id or Path(case_id).name != case_id or "/" in case_id or "\\" in case_id:
        raise ValueError("Invalid case ID")
    if public_manifest is None:
        case = json.loads((source / "materialized" / f"{case_id}.json").read_text(encoding="utf-8"))
    else:
        matches = []
        with public_manifest.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    row = json.loads(line)
                    if row.get("instance_id") == case_id:
                        matches.append(row)
        if len(matches) != 1:
            raise ValueError("Case ID must occur exactly once in public manifest")
        case = matches[0]
    if case.get("instance_id") != case_id:
        raise ValueError("Case ID mismatch")
    return case


def prepare(case_id: str, source: Path = HISTORICAL, output: Path = OUTPUT,
            cosil: Path = EXTERNAL, public_manifest: Path | None = None) -> dict:
    case = load_public_case(case_id, source, public_manifest)
    materialized = json.loads((source / "materialized" / f"{case_id}.json").read_text(encoding="utf-8"))
    archive = source / "sources" / f"{case_id}.tar.gz"
    if materialized.get("instance_id") != case_id:
        raise ValueError("Materialized archive case ID mismatch")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != materialized["original_archive_sha256"]:
        raise ValueError("Base-source archive checksum mismatch")
    if any(not isinstance(case.get(key), str) or not case[key].strip() for key in PUBLIC_FIELDS):
        raise ValueError("Incomplete public case fields")

    sys.path.insert(0, str(cosil))
    from get_repo_structure.get_repo_structure import parse_python_file

    top_name = case["repo"].split("/")[-1]
    structure = {top_name: {}}
    file_count = 0
    with tarfile.open(archive, "r:gz") as files:
        roots = {PurePosixPath(member.name).parts[0] for member in files
                 if PurePosixPath(member.name).parts}
        if len(roots) != 1:
            raise ValueError("Expected one base-source root")
        archive_root = next(iter(roots))
        if not case["base_commit"].startswith(archive_root.split("-")[-1]):
            raise ValueError("Archive root does not match base commit")
        for member in files:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != archive_root or ".." in parts:
                raise ValueError("Unsafe archive member")
            if not (member.isdir() or member.isfile()):
                continue
            if len(parts) == 1:
                continue
            parent = structure[top_name]
            for part in parts[1:-1]:
                parent = parent.setdefault(part, {})
            if member.isdir():
                parent.setdefault(parts[-1], {})
                continue
            if parts[-1].endswith(".py"):
                stream = files.extractfile(member)
                if stream is None:
                    raise ValueError(f"Cannot read {member.name}")
                content = stream.read().decode("utf-8", errors="replace")
                classes, functions, lines = parse_python_file(member.name, content)
                parent[parts[-1]] = {"classes": classes, "functions": functions, "text": lines}
            else:
                parent[parts[-1]] = {}
            file_count += 1

    output.mkdir(parents=True, exist_ok=True)
    public = {key: case[key] for key in PUBLIC_FIELDS}
    (output / "case.jsonl").write_text(json.dumps(public, ensure_ascii=False) + "\n", encoding="utf-8")
    structure_dir = output / "repo_structures"
    structure_dir.mkdir(exist_ok=True)
    (structure_dir / f"{case_id}.json").write_text(json.dumps({
        "repo": case["repo"], "base_commit": case["base_commit"],
        "instance_id": case_id, "structure": structure,
    }, ensure_ascii=False), encoding="utf-8")
    report = {"case_id": case_id, "source_archive_sha256": digest,
              "structure_file_count": file_count, "public_input": str(output / "case.jsonl"),
              "structure_dir": str(structure_dir)}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="psf__requests-1376")
    parser.add_argument("--source", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cosil", type=Path, default=EXTERNAL)
    parser.add_argument("--public-manifest", type=Path,
                        help="Patch-free JSONL manifest, required for expanded RQ1 cases")
    args = parser.parse_args()
    print(json.dumps(prepare(args.case_id, args.source, args.output, args.cosil,
                             args.public_manifest), indent=2))
