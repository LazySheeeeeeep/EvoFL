"""Prepare a reproducible continuation sample without changing the SkillBank."""
import json
from pathlib import Path

from run_swesmith_case_by_case import select_cases


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v5_train40_deepseek_v4_flash_20260910_continuation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "selected_cases.json").exists():
        print("Existing sample retained", flush=True)
        return
    excluded = set()
    manifests = []
    for path in (ROOT / "runs").glob("**/selected_cases.json"):
        if path.is_relative_to(OUT):
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            continue
        if not isinstance(rows, list):
            continue
        excluded.update(r["instance_id"] for r in rows if isinstance(r, dict) and r.get("instance_id"))
        manifests.append(str(path.relative_to(ROOT)))
    cases = select_cases(40, 20260911, "SWE-bench/SWE-smith-py", "train", True, excluded)
    (OUT / "selected_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    (OUT / "sampling_audit.json").write_text(json.dumps({
        "seed": 20260911, "sample_size": len(cases), "excluded_id_count": len(excluded),
        "excluded_manifest_paths": manifests, "overlap_count": len(excluded & {c["instance_id"] for c in cases}),
        "prefer_local_images": True}, indent=2))
    bank = ROOT / "skill_pools/skill_bank_v5/skills.jsonl"
    (OUT / "initial_skill_bank.jsonl").write_bytes(bank.read_bytes())
    print(f"Prepared {len(cases)} cases; excluded {len(excluded)} prior IDs", flush=True)


if __name__ == "__main__":
    main()
