"""Report size-only statistics for frozen Flat-Reflection model inputs."""
from __future__ import annotations

import json

from build_rq1_flat_reflections import flat_reflection_input
from prepare_rq1_memory_baselines import EXPANDED, OUTPUT, read_json


def main() -> None:
    cases = {case["instance_id"]: case for name in
             ("original_training.json", "training_additions.json")
             for case in read_json(EXPANDED / name)}
    episodes = [json.loads(line) for line in (OUTPUT / "episodes.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    sizes = sorted((len(json.dumps(flat_reflection_input(episode, cases[episode["instance_id"]]),
                                   ensure_ascii=False)), episode["instance_id"])
                   for episode in episodes)
    if not sizes:
        raise ValueError("No acquisition episodes")
    print(json.dumps({"count": len(sizes), "min_chars": sizes[0][0],
                      "median_chars": sizes[len(sizes) // 2][0],
                      "p90_chars": sizes[int(len(sizes) * 0.9)][0],
                      "max_chars": sizes[-1][0],
                      "median_case_id": sizes[len(sizes) // 2][1],
                      "max_case_id": sizes[-1][1]}, indent=2))


if __name__ == "__main__":
    main()
