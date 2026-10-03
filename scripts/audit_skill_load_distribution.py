"""Audit Fault Skill routing, validation, loading, and transfer in the main run."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922"
BANK = RUN / "frozen_skills.jsonl"
SUMMARY = RUN / "with_skill_failure_retries_20260929/comparison_summary_with_retries.json"
RETRY_ROOT = RUN / "with_skill_failure_retries_20260929/cases"
OUTPUT = ROOT / "runs/main_skill_load_distribution_20261001.json"

FAMILIES = [
    "missing_functionality",
    "state_assignment",
    "validation_control",
    "interface_contract",
    "transformation_representation",
    "algorithm_computation",
    "dispatch_resolution",
    "lifecycle_timing_resource",
    "environment_integration",
]


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def active_bank() -> tuple[list[dict], dict[str, dict]]:
    rows = [
        json.loads(line)
        for line in BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    active: dict[str, dict] = {}
    for row in rows:
        if row.get("status") != "active":
            continue
        skill_id = row["skill_id"]
        current = active.get(skill_id)
        if current is None or int(row.get("version") or 0) >= int(
            current.get("version") or 0
        ):
            active[skill_id] = row
    flattened = []
    for row in active.values():
        nested = row.get("skill") if isinstance(row.get("skill"), dict) else {}
        flattened.append(
            {
                **nested,
                **{key: value for key, value in row.items() if key != "skill"},
            }
        )
    return flattened, {row["skill_id"]: row for row in flattened}


def search_for(case_id: str) -> dict:
    retry = RETRY_ROOT / case_id / "attempt_1/fault_skill_search.json"
    path = retry if retry.is_file() else RUN / "with_skill/cases" / case_id / "fault_skill_search.json"
    return read(path, {})


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def family_report(rows: list[dict], bank_by_family: dict[str, list[dict]]) -> list[dict]:
    result = []
    for family in FAMILIES:
        routed = [row for row in rows if row["requested_family"] == family]
        selected = [row for row in routed if row["selector_selected_id"]]
        loaded = [row for row in routed if row["loaded_skill_id"]]
        alias = [row for row in loaded if row["loaded_primary_family"] != family]
        valid = [
            row
            for row in loaded
            if row["with_top1"] is not None and row["no_top1"] is not None
        ]
        wins = sum(row["with_top1"] and not row["no_top1"] for row in valid)
        losses = sum(row["no_top1"] and not row["with_top1"] for row in valid)
        bank = bank_by_family.get(family, [])
        result.append(
            {
                "family": family,
                "active_skill_count": len(bank),
                "routed_cases": len(routed),
                "selector_selected_cases": len(selected),
                "validator_loaded_cases": len(loaded),
                "selector_selection_rate": rate(len(selected), len(routed)),
                "validator_acceptance_rate": rate(len(loaded), len(selected)),
                "family_load_rate": rate(len(loaded), len(routed)),
                "loaded_via_alias_cases": len(alias),
                "unique_loaded_skills": len({row["loaded_skill_id"] for row in loaded}),
                "loaded_skill_coverage": rate(
                    len({row["loaded_skill_id"] for row in loaded}), len(bank)
                ),
                "loaded_top1": rate(
                    sum(bool(row["with_top1"]) for row in valid), len(valid)
                ),
                "no_skill_top1_same_cases": rate(
                    sum(bool(row["no_top1"]) for row in valid), len(valid)
                ),
                "top1_wins": wins,
                "top1_losses": losses,
                "top1_delta": rate(
                    sum(bool(row["with_top1"]) - bool(row["no_top1"]) for row in valid),
                    len(valid),
                ),
            }
        )
    return result


def skill_report(rows: list[dict], active: list[dict]) -> list[dict]:
    route_counts = Counter()
    for row in rows:
        for skill in active:
            if row["requested_family"] in (skill.get("retrieval_families") or []):
                route_counts[skill["skill_id"]] += 1
    by_skill = defaultdict(list)
    for row in rows:
        if row["loaded_skill_id"]:
            by_skill[row["loaded_skill_id"]].append(row)
    result = []
    for skill in active:
        skill_id = skill["skill_id"]
        loaded = by_skill.get(skill_id, [])
        valid = [
            row
            for row in loaded
            if row["with_top1"] is not None and row["no_top1"] is not None
        ]
        result.append(
            {
                "skill_id": skill_id,
                "fault_family": skill.get("fault_family"),
                "retrieval_families": skill.get("retrieval_families") or [],
                "title": skill.get("title"),
                "load_count": len(loaded),
                "load_rate_all_500": len(loaded) / len(rows) if rows else None,
                "load_rate_eligible_routes": rate(
                    len(loaded), route_counts[skill_id]
                ),
                "eligible_routes": route_counts[skill_id],
                "top1_loaded_cases": rate(
                    sum(bool(row["with_top1"]) for row in valid), len(valid)
                ),
                "top1_no_skill_same_cases": rate(
                    sum(bool(row["no_top1"]) for row in valid), len(valid)
                ),
                "top1_wins": sum(
                    row["with_top1"] and not row["no_top1"] for row in valid
                ),
                "top1_losses": sum(
                    row["no_top1"] and not row["with_top1"] for row in valid
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (-row["load_count"], row["fault_family"], row["skill_id"]),
    )


def main() -> None:
    active, _ = active_bank()
    summary = read(SUMMARY, {})
    cases = summary.get("cases") or []
    rows = []
    errors = []
    for case in cases:
        case_id = case["instance_id"]
        search = search_for(case_id)
        if not search:
            errors.append({"instance_id": case_id, "error": "missing fault_skill_search"})
            continue
        trace = search.get("search_trace") or {}
        selector = trace.get("selector") or {}
        matched = search.get("matched_skill") or {}
        loaded_id = matched.get("skill_id") or (case.get("arms", {}).get("with_skill", {}) or {}).get(
            "loaded_skill_id"
        )
        bank_card = next(
            (skill for skill in active if skill["skill_id"] == loaded_id), None
        )
        with_arm = (case.get("arms") or {}).get("with_skill") or {}
        no_arm = (case.get("arms") or {}).get("no_skill") or {}
        rows.append(
            {
                "instance_id": case_id,
                "requested_family": search.get("fault_family"),
                "selector_selected_id": selector.get("selected_skill_id"),
                "validator_applicable": (trace.get("validator") or {}).get("applicable"),
                "loaded_skill_id": loaded_id,
                "loaded_primary_family": bank_card.get("fault_family") if bank_card else None,
                "loaded_via_alias": bool(trace.get("loaded_via_alias")),
                "catalog_count": trace.get("catalog_count"),
                "catalog_char_count": trace.get("catalog_char_count"),
                "with_status": with_arm.get("status"),
                "no_status": no_arm.get("status"),
                "with_top1": (with_arm.get("metrics") or {}).get("top1"),
                "no_top1": (no_arm.get("metrics") or {}).get("top1"),
            }
        )
    bank_by_family = defaultdict(list)
    for skill in active:
        bank_by_family[skill.get("fault_family")].append(skill)
    families = family_report(rows, bank_by_family)
    skills = skill_report(rows, active)
    loaded_ids = {row["loaded_skill_id"] for row in rows if row["loaded_skill_id"]}
    report = {
        "evaluation_target": len(cases),
        "evaluation_audited": len(rows),
        "errors": errors,
        "bank": {
            "active_total": len(active),
            "families": dict(Counter(skill.get("fault_family") for skill in active)),
            "loaded_unique_total": len(loaded_ids),
            "zero_load_total": len(active) - len(loaded_ids),
        },
        "overall": {
            "routed_cases": sum(row["requested_family"] in FAMILIES for row in rows),
            "selector_selected": sum(bool(row["selector_selected_id"]) for row in rows),
            "validator_loaded": sum(bool(row["loaded_skill_id"]) for row in rows),
            "load_rate": rate(
                sum(bool(row["loaded_skill_id"]) for row in rows), len(rows)
            ),
            "loaded_unique_skills": len(loaded_ids),
            "cache_or_alias_loads": sum(row["loaded_via_alias"] for row in rows),
        },
        "families": families,
        "skills": skills,
        "loaded_skill_rank": [row for row in skills if row["load_count"] > 0],
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k != "skills"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
