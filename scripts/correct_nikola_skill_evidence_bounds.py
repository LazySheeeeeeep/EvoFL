"""Version one reviewed Skill correction, preserving all other bank entries."""
from copy import deepcopy
import json
from pathlib import Path

from evolutefl.config import load_config
from evolutefl.skills import make_skill_bank


TARGET = "fault_skill_action_function_rewritten_to_call_an_undefined_instance_helper_while"
KNOWLEDGE = [
    "When a function on a suspected issue path calls a helper whose definition is unclear, inspect the receiver's class, bases, and dynamic attribute providers. If the helper remains unresolved, trace whether the call can execute under the reported conditions and affect the reported output before increasing the caller's priority.",
    "Unused imports or caches can suggest that an implementation path changed. Use them to identify dependencies and call sites worth inspecting, then check how those components participate in the reported behavior; keep missing use as a hypothesis until that connection is supported.",
    "Compare a suspicious candidate's behavior with callers, existing test assertions, and the issue's expected result. A test or TODO documents an expectation or known concern to investigate; when they disagree with the issue, establish whether the reported execution follows that path before retaining or excluding the candidate.",
]


def main():
    bank = make_skill_bank(load_config("config/evolutefl.global.json"))
    before = {s.skill_id: s.to_dict() for s in bank.active_skills()}
    card = deepcopy(before[TARGET])
    if card["skill"]["knowledge"] == KNOWLEDGE:
        print("Reviewed correction already applied")
        return
    card["skill"]["trigger"] = (
        "An issue concerns incomplete or malformed output, and inspection of a potentially related rendering/action path reveals a helper call with unclear implementation or dependencies whose role is no longer evident."
    )
    card["skill"]["knowledge"] = KNOWLEDGE
    update = {"operation": "rewrite", "skill_type": "fault_skill", "target_skill_id": TARGET,
              "skill": {**card["skill"], "fault_family": card["fault_family"],
                        "fault_subtype": card["fault_subtype"], "retrieval_families": card["retrieval_families"]}}
    out = Path("runs/skill_evidence_bounds_correction")
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "skill_bank_before.jsonl"
    if backup.exists():
        raise RuntimeError("Review existing correction artifacts before applying again")
    backup.write_bytes(bank.path.read_bytes())
    result = bank.apply_update(update)
    after = {s.skill_id: s.to_dict() for s in bank.active_skills()}
    assert before.keys() == after.keys()
    assert all(before[k] == after[k] for k in before if k != TARGET)
    (out / "correction.json").write_text(json.dumps(
        {"origin": "manual evidence-bounds review; not an LLM experiment",
         "before": before[TARGET], "after": after[TARGET], "result": result}, indent=2))
    print(json.dumps({"active_skill_count": len(after), "result": result}))


if __name__ == "__main__":
    main()
