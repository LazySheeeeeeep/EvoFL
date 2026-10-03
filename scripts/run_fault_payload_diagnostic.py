"""Matched-prefix continuations: accepted Fault knowledge versus a null result.

No selector, evolution, or answer-bearing input is used by the continuation.
Original source observations must replay exactly before any live LLM call.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evolutefl.explorer.v5_agent import V5ExplorerAgent
from evolutefl.investigation import InvestigationLog, read_observations
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from run_swe_explore_v5_stratified import read, write, sha, safe_archive_filter, strict_metrics

SOURCE = ROOT / "runs/swe_explore_v5_stratified100_deepseek_20260910"
OUT = ROOT / "runs/fault_payload_diagnostic_20260912"
CASES = ["sphinx-doc__sphinx-8269", "sympy__sympy-17655", "matplotlib__matplotlib-23476",
         "pylint-dev__pylint-6386", "psf__requests-2931", "pylint-dev__pylint-6528",
         "astropy__astropy-13398"]


class PrefixObservationLog(InvestigationLog):
    def __init__(self, *args, originals, load_step, **kwargs):
        super().__init__(*args, **kwargs)
        self.originals, self.load_step = originals, load_step

    def observe(self, record, result, *, error=None):
        if record["step"] < self.load_step:
            oid = f"{self.prefix}-{len(self.observations) + 1:05d}"
            original = self.originals[oid]
            if original["call_id"] != record["call_id"] or original["tool"] != record["tool"]:
                raise ValueError("Prefix action identity differs")
            # File traversal order is not stable after archive extraction. Source
            # identity is checked before any live request; preserve the exact
            # historical observation, including its ordering and pagination.
            payload = original["model_payload"]
            result, error = copy.deepcopy(payload["result"]), payload.get("error")
            record["replayed_observation"] = True
        return super().observe(record, result, error=error)


class PrefixReplayAgent(V5ExplorerAgent):
    def __init__(self, *, prefix_dir, inject_skill, **kwargs):
        super().__init__(**kwargs)
        self.prefix_dir = Path(prefix_dir)
        self.inject_skill = inject_skill
        self.fault_request = read(self.prefix_dir / "fault_skill_request.json")
        self.load_step = self.fault_request["step"]
        self.prefix = {row["step"]: row for row in read(self.prefix_dir / "llm_trace.json")
                       if row["step"] <= self.load_step and row.get("finish_reason") != "length"}
        self.original_observations = read_observations(self.prefix_dir / "observations.jsonl")
        self.investigation_log_type = lambda *a, **k: PrefixObservationLog(
            *a, originals=self.original_observations, load_step=self.load_step, **k)
        self.verified = False

    def _chat_step(self, request, *, step, traces, directory, deadline):
        if step <= self.load_step:
            row = copy.deepcopy(self.prefix[step])
            names = [tool["function"]["name"] for tool in request["tools"]]
            if names != row["request_tools"]:
                raise ValueError("Prefix tool schemas differ from the original run")
            row["replayed_prefix"] = True
            traces.append(row)
            write(directory / "llm_trace.json", traces)
            self.thinking_disabled_after_recovery = row.get("recovery_thinking_disabled", False)
            return row["response"]
        if not self.verified:
            raise ValueError("Prefix verification failed: live continuation forbidden")
        if step == self.load_step + 1:
            write(directory / "continuation_input.json", request)
        return super()._chat_step(request, step=step, traces=traces,
                                  directory=directory, deadline=deadline)

    def load_fault(self, args, issue, log):
        expected = {key: row for key, row in self.original_observations.items()
                    if row["step"] < self.load_step}
        if set(expected) != set(log.observations):
            raise ValueError("Prefix observation IDs differ")
        for key, row in expected.items():
            if log.observations[key]["model_payload"] != row["model_payload"]:
                raise ValueError(f"Prefix source observation differs: {key}")
        if args != self.fault_request["request"]:
            raise ValueError("Prefix Fault request differs")
        original = read(self.prefix_dir / "run_context.json")["repository_identity"]
        current = read(log.directory / "run_context.json")["repository_identity"]
        if current != original:
            raise ValueError("Repository identity differs from original pre-run state")
        search = read(self.prefix_dir / "fault_skill_search.json")
        card = search.get("matched_skill") if self.inject_skill else None
        write(log.directory / "prefix_verification.json", {
            "verified": True, "original_case_dir": str(self.prefix_dir),
            "load_step": self.load_step, "observation_count": len(expected),
            "observation_mode": "exact historical payload replay after repository identity verification",
            "repository_identity": current,
            "intervention": "accepted_skill" if self.inject_skill else "null",
            "actual_skill_id": (card or {}).get("skill_id"),
        })
        self.verified = True
        return {"fault_family": args["fault_family"], "matched_skill": card,
                "search_trace": {"diagnostic_replay": True}}


def cleanup(work):
    root = (OUT / "work").resolve()
    if work.is_symlink() or work.resolve() == root or not work.resolve().is_relative_to(root):
        raise ValueError("Unsafe diagnostic workspace cleanup")
    if work.exists():
        shutil.rmtree(work)


def summary():
    rows = [read(p) for p in sorted((OUT / "paired").glob("*.json"))]
    groups = {}
    for loaded in (True, False):
        subset = [r for r in rows if r["originally_loaded"] == loaded]
        groups["accepted_skill_cases" if loaded else "null_controls"] = {
            "pairs": len(subset), "arms": {
                arm: {"completed": sum(r["arms"][arm]["status"] == "completed" for r in subset),
                      **{k: sum(r["arms"][arm]["metrics"][k] for r in subset) / len(subset)
                         if subset else None for k in ("top1", "top3", "top5", "mrr")}}
                for arm in ("null", "accepted_skill")}}
    write(OUT / "summary.json", {"completed_pairs": len(rows), "groups": groups, "cases": rows})


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=len(CASES))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--system-prompt", type=Path)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if not OUT.is_relative_to((ROOT / "runs").resolve()) or OUT == (ROOT / "runs").resolve():
        raise ValueError("Diagnostic output must be a subdirectory of workspace runs")
    if not 1 <= args.limit <= len(CASES) or args.repeats < 1:
        raise ValueError("Invalid diagnostic sample/repeat count")
    frozen = read(SOURCE / "protocol.json")
    for name, key in (("frozen_config.json", "config_sha256"), ("frozen_skills.jsonl", "bank_sha256"),
                      ("selected_cases.json", "selection_sha256")):
        if sha(SOURCE / name) != frozen[key]:
            raise ValueError(f"Frozen input changed: {name}")
    for filename, digest in frozen["prompt_hashes"].items():
        if sha(Path(filename)) != digest:
            raise ValueError("Frozen prompt changed")
    cfg = read(SOURCE / "frozen_config.json")
    prompt_path = args.system_prompt or Path(cfg["explorer"]["system_prompt_path"])
    system_prompt = prompt_path.read_text()
    keyenv = cfg["llm"].get("api_key_env", "DEEPSEEK_API_KEY")
    if not os.environ.get(keyenv):
        raise ValueError(f"Required credential environment variable absent: {keyenv}")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {"cases": CASES[:args.limit], "repeats": args.repeats,
                "source": str(SOURCE), "source_config_sha256": frozen["config_sha256"],
                "bank_sha256": frozen["bank_sha256"], "model": cfg["llm"]["model"],
                "design": "same original with_skill pre-load prefix; no selector; frozen accepted card vs null",
                "prefix_observations": "historical exact payloads; archive and full source identity verified",
                "null_controls": "original null selections receive identical null payloads in both arms",
                "exploratory_diagnostic": True, "evolution": False,
                "runner_sha256": sha(Path(__file__)),
                "system_prompt_sha256": sha(prompt_path),
                "prompt_intervention": bool(args.system_prompt),
                "tool_source_hashes": {name: sha(ROOT / name) for name in (
                    "src/evolutefl/tools/symbols.py", "src/evolutefl/explorer/agent.py")},
                "agent_sha256": sha(ROOT / "src/evolutefl/explorer/v5_agent.py")}
    existing = read(OUT / "protocol.json")
    if existing is not None and existing != protocol:
        raise ValueError("Diagnostic protocol changed; use a separate output directory")
    write(OUT / "protocol.json", protocol)
    (OUT / "explorer_system_snapshot.txt").write_text(system_prompt)
    for index, instance_id in enumerate(CASES[:args.limit]):
        prefix = SOURCE / "with_skill/cases" / instance_id
        old_pair = read(SOURCE / "paired" / f"{instance_id}.json")
        if sha(SOURCE / "sources" / f"{instance_id}.tar.gz") != old_pair["archive_sha256"]:
            raise ValueError("Source archive hash mismatch")
        for repeat in range(1, args.repeats + 1):
            pairfile = OUT / "paired" / f"{instance_id}__{repeat}.json"
            if pairfile.exists():
                continue
            row = {"instance_id": instance_id, "repeat": repeat,
                   "originally_loaded": bool(old_pair["arms"]["with_skill"]["loaded_skill_id"]), "arms": {}}
            arms = ["null", "accepted_skill"]
            if (index + repeat) % 2:
                arms.reverse()
            for arm in arms:
                directory = OUT / arm / f"{instance_id}__{repeat}"
                saved = read(directory / "diagnostic_result.json")
                if saved:
                    row["arms"][arm] = saved
                    continue
                if (directory / "trajectory.jsonl").exists():
                    raise ValueError(f"Unfinished diagnostic run needs review: {directory}")
                write(OUT / "current_case.json", {"case": instance_id, "repeat": repeat, "arm": arm})
                print(f"{index + 1}/{args.limit} repeat={repeat}/{args.repeats} {instance_id} {arm}", flush=True)
                work = OUT / "work" / instance_id
                cleanup(work)
                work.mkdir(parents=True)
                with tarfile.open(SOURCE / "sources" / f"{instance_id}.tar.gz") as archive:
                    archive.extractall(work, filter=safe_archive_filter)
                roots = list(work.iterdir())
                if len(roots) != 1 or not roots[0].is_dir():
                    raise ValueError("Unexpected archive layout")
                prompt = (prefix / "initial_payload.json")
                # Only the original issue/repo fields go into Explorer. Patch and
                # stored ground truth are used below solely by the scorer.
                initial = read(prompt)
                task = {"instance_id": instance_id, "repo": initial["repo"],
                        "base_commit": initial["base_commit"], "bug_report": initial["bug_report"],
                        "repo_path": str(roots[0]), "run_dir": str(directory)}
                agent = PrefixReplayAgent(prefix_dir=prefix, inject_skill=arm == "accepted_skill",
                    llm_client=OpenAICompatibleClient.from_config(cfg["llm"]), skill_bank=make_skill_bank(cfg),
                    config=copy.deepcopy(cfg), system_prompt=system_prompt)
                result = agent.run(task)
                if not agent.verified:
                    raise RuntimeError(f"Prefix failed before live continuation: {result.get('error')}")
                record = {"status": result["status"], "predictions": result["ranked_functions"],
                          "metrics": strict_metrics(result["ranked_functions"], old_pair["functions"]),
                          "steps": result["steps"], "continuation_steps": result["steps"] - agent.load_step,
                          "forced_finish": result["forced_finish"], "error": result.get("error"),
                          "prefix_verified": True}
                write(directory / "diagnostic_result.json", record)
                row["arms"][arm] = record
                cleanup(work)
            write(pairfile, row)
            summary()
    write(OUT / "current_case.json", {"phase": "finished"})


if __name__ == "__main__":
    main()
