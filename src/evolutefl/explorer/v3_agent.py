"""V3 Explorer: staged Project -> Fault -> Strategy Skill loading.

The legacy agent remains available for historical runs.  This module keeps the
new protocol explicit so V3 trajectories cannot accidentally look like V2
ones during later reflection.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from evolutefl.json_utils import append_jsonl, write_json
from evolutefl.json_utils import extract_json_object
from evolutefl.config import resolve_path
from evolutefl.skills.fault_taxonomy import (
    FAULT_FAMILIES,
    compact_fault_taxonomy,
    validate_fault_family,
)
from evolutefl.tools import ToolRegistry, register_builtin_tools
from evolutefl.tools.registry import openai_function_tool

from .agent import (
    FINISH_TOOL_NAME,
    PROJECT_SKILL_TOOL,
    STRATEGY_SKILL_TOOL,
    ExplorerAgent,
    _finish_tool_schema,
    _collect_evidence_from_trajectory,
    _is_test_path,
    _named_tool_choice,
    _nearest_python_symbol,
    _project_skill_tool_schema,
    _rank_evidence_candidates,
    _synthetic_tool_call,
    _tool_arguments,
    _tool_error,
    _tool_snapshot,
    _tool_success,
    _strategy_skill_tool_schema,
)

FAULT_SKILL_TOOL = "load_fault_skill"


def _fault_skill_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=FAULT_SKILL_TOOL,
        description=(
            "Classify the defect into exactly one fixed Fault family, then load at most one "
            "matching fine-grained Fault Skill. When categories overlap, choose the family "
            "whose responsibility first introduces the incorrect state."
        ),
        properties={
            "fault_family": {
                "type": "string",
                "enum": list(FAULT_FAMILIES),
                "description": "The single fixed top-level Fault family.",
            },
            "fault_signature": {"type": "string", "description": "A concise abstract behavior mismatch."},
            "project_context": {"type": "string", "description": "Observed architecture or responsibility boundary."},
            "suspected_path": {"type": "string", "description": "Likely code path or boundary to inspect next."},
        },
        required=["fault_family", "fault_signature", "project_context", "suspected_path"],
    )


def _fault_request_text(arguments: dict[str, Any]) -> str:
    return "\n".join(
        f"{key}: {str(arguments.get(key) or '').strip()}"
        for key in ("fault_family", "fault_signature", "project_context", "suspected_path")
        if str(arguments.get(key) or "").strip()
    )


def _compact_project_context(skill: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the Project model needed to interpret an Issue request."""
    if not isinstance(skill, dict):
        return None
    return {
        key: skill.get(key)
        for key in ("skill_id", "repo_id", "title")
        if skill.get(key)
    }


def _fault_selection_context(
    arguments: dict[str, Any],
    *,
    orientation: dict[str, Any],
    project_skill: dict[str, Any] | None,
) -> str:
    """Build the compact context used by the within-family LLM selector."""
    sections = ["fault request:\n" + _fault_request_text(arguments)]
    if orientation:
        sections.append("repository orientation:\n" + json.dumps(orientation, ensure_ascii=False))
    if project_skill:
        sections.append("project model:\n" + json.dumps(project_skill, ensure_ascii=False))
    return "\n\n".join(section for section in sections if section.strip())


def _repository_manifest(repo_path: Path) -> dict[str, Any]:
    """A bounded orientation aid, not a substitute for repository exploration."""
    try:
        entries = sorted(item.name for item in repo_path.iterdir())[:40]
        files = sorted(
            str(item.relative_to(repo_path))
            for item in repo_path.rglob("*")
            if item.is_file() and item.suffix in {".py", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
        )[:80]
    except OSError:
        entries, files = [], []
    return {"root_entries": entries, "representative_files": files}


def _observed_evidence_snapshot(run_dir: Path, repo_path: Path) -> tuple[set[str], set[str]]:
    """Return inspected definition candidates and observed source paths.

    Grep is useful for finding files, but its broad matches must not by
    themselves make the agent appear ready to compare function candidates.
    Only definitions actually returned by ``read_file`` count as candidates.
    """
    evidence = _collect_evidence_from_trajectory(run_dir / "trajectory.jsonl")
    candidates: set[str] = set()
    trajectory = run_dir / "trajectory.jsonl"
    if trajectory.exists():
        for line in trajectory.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
                content = json.loads(event.get("content") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if event.get("event") != "tool_result" or event.get("name") != "read_file":
                continue
            result = content.get("result") if content.get("ok") else None
            if not isinstance(result, dict):
                continue
            path_value = str(result.get("path") or "")
            if not path_value.endswith(".py") or _is_test_path(path_value):
                continue
            for item in result.get("content") or []:
                text = str(item.get("text") or "").lstrip()
                if not text.startswith(("def ", "async def ")):
                    continue
                symbol = _nearest_python_symbol(repo_path / path_value, int(item.get("line") or 1))
                if symbol:
                    candidates.add(f"{path_value}::{symbol}")
    source_paths = {
        path for path in evidence
        if path.endswith(".py") and not _is_test_path(path)
    }
    return candidates, source_paths


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated for finalization]"


def _compact_skill_for_finish(skill: dict[str, Any]) -> dict[str, Any]:
    card = skill.get("skill") if isinstance(skill.get("skill"), dict) else skill
    knowledge = card.get("knowledge") if isinstance(card, dict) else None
    if isinstance(knowledge, list):
        knowledge = [
            _truncate_text(item.get("text") if isinstance(item, dict) else item, 900)
            for item in knowledge[:3]
        ]
    else:
        knowledge = _truncate_text(knowledge, 1800)
    return {
        "skill_id": skill.get("skill_id"),
        "title": card.get("title") if isinstance(card, dict) else None,
        "trigger": _truncate_text(card.get("trigger") if isinstance(card, dict) else None, 700),
        "knowledge": knowledge,
    }


class V3ExplorerAgent(ExplorerAgent):
    """Explorer protocol where each Skill is loaded at its own decision point."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        cfg = self.config.get("explorer", {}) or {}
        self.fault_selector_prompt = resolve_path(cfg.get(
            "fault_skill_selector_prompt_path", "prompt_records/explorer/fault_skill_selector_v1.txt"
        )).read_text(encoding="utf-8")
        self.fault_validator_prompt = resolve_path(cfg.get(
            "fault_skill_validator_prompt_path", "prompt_records/explorer/fault_skill_validator_v1.txt"
        )).read_text(encoding="utf-8")
        self.fault_skill_selector_max_tokens = max(
            512,
            int(cfg.get("fault_skill_selector_max_tokens", 4096)),
        )
        self.fault_skill_selector_retry_max_tokens = max(
            self.fault_skill_selector_max_tokens,
            int(cfg.get("fault_skill_selector_retry_max_tokens", 8192)),
        )
        self.fault_skill_selector_attempts = max(
            1,
            int(cfg.get("fault_skill_selector_attempts", 3)),
        )
        self.fault_skill_evidence_budget = max(0, int(cfg.get("fault_skill_evidence_budget", 2)))
        self.fault_candidate_min_count = max(1, int(cfg.get("fault_candidate_min_count", 2)))
        self.fault_min_observations_before_stagnation = max(
            self.fault_skill_evidence_budget,
            int(cfg.get("fault_min_observations_before_stagnation", 4)),
        )
        self.stage_stagnation_turns = max(1, int(cfg.get("stage_stagnation_turns", 2)))
        self.strategy_skill_verification_budget = max(
            0,
            int(cfg.get("strategy_skill_verification_budget", 1)),
        )
        self.orientation_min_source_paths = max(1, int(cfg.get("orientation_min_source_paths", 2)))
        self.orientation_min_observations_before_stagnation = max(
            int(cfg.get("orientation_tool_budget", 3)),
            int(cfg.get("orientation_min_observations_before_stagnation", 3)),
        )

    def run(self, task: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        issue = str(task.get("bug_report") or "").strip()
        if not issue:
            raise ValueError(
                "V3 Explorer requires a non-empty bug_report/problem_statement; "
                "repository-only browsing cannot perform the Fault Skill stage."
            )
        run_dir = Path(task["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        repo_path = Path(task["repo_path"]).resolve()
        repo = str(task.get("repo") or "")
        registry = ToolRegistry()
        register_builtin_tools(registry, run_dir=run_dir, repo_path=str(repo_path))
        write_json(run_dir / "tools.json", [
            *registry.list_tools(), _tool_snapshot(_project_skill_tool_schema()),
            _tool_snapshot(_fault_skill_tool_schema()),
            _tool_snapshot(_strategy_skill_tool_schema(self.skill_bank.strategy_catalog())),
            _tool_snapshot(_finish_tool_schema()),
        ])

        manifest = _repository_manifest(repo_path)
        initial_payload = {
            "stage": "repository_orientation",
            "objective": (
                "Inspect repository structure and representative code to form a minimal system model, "
                "then call load_project_skill."
            ),
            "repo": repo,
            "base_commit": task.get("base_commit", ""),
            "repository_manifest": manifest,
        }
        write_json(run_dir / "initial_payload.json", initial_payload)
        write_json(run_dir / "repository_manifest.json", manifest)
        loaded: dict[str, Any] = {"project_skill": None, "fault_skill": None, "strategy_skill": None}
        write_json(run_dir / "loaded_skills.json", loaded)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(initial_payload, ensure_ascii=False, indent=2)}]
        project_attempted = issue_attempted = strategy_attempted = issue_revealed = False
        strategy_validation_observations = 0
        strategy_transition_reason: str | None = None
        fault_skill_evidence_observations = 0
        issue_repo_observations = 0
        issue_known_candidates: set[str] = set()
        issue_known_paths: set[str] = set()
        issue_stagnant_steps = 0
        issue_transition_reason: str | None = None
        orientation_repo_calls = 0
        orientation_known_paths: set[str] = set()
        orientation_stagnant_steps = 0
        orientation_transition_reason: str | None = None
        llm_trace: list[dict[str, Any]] = []
        final_result: dict[str, Any] | None = None
        step, invalid = 0, 0
        started = time.monotonic()

        while self.max_steps is None or step < self.max_steps:
            step += 1
            if self._runtime_expired(started):
                break
            stage = _stage(project_attempted, issue_attempted, strategy_attempted)
            tools = self._tools(
                registry,
                stage,
                fault_skill_loaded=loaded["fault_skill"] is not None,
                fault_skill_evidence_observations=fault_skill_evidence_observations,
            )
            forced_tool = self._required_v3_control_tool(
                step=step, project_attempted=project_attempted, issue_attempted=issue_attempted,
                strategy_attempted=strategy_attempted,
                strategy_skill_loaded=loaded["strategy_skill"] is not None,
                strategy_validation_observations=strategy_validation_observations,
                orientation_repo_calls=orientation_repo_calls,
                fault_skill_loaded=loaded["fault_skill"] is not None,
                fault_skill_evidence_observations=fault_skill_evidence_observations,
                issue_repo_observations=issue_repo_observations,
                issue_candidate_count=len(issue_known_candidates),
                issue_stagnant_steps=issue_stagnant_steps,
                orientation_source_path_count=len(orientation_known_paths),
                orientation_stagnant_steps=orientation_stagnant_steps,
            )
            supports_tool_choice = bool(getattr(self.llm_client, "supports_tool_choice", True))
            tool_choice: str | dict[str, Any] | None = (
                _named_tool_choice(forced_tool) if forced_tool and supports_tool_choice else
                "auto" if supports_tool_choice else None
            )
            if forced_tool:
                if not supports_tool_choice:
                    # DeepSeek reasoning mode rejects ``tool_choice``.  At a
                    # mandatory transition, a one-tool schema preserves the
                    # native-call protocol without that unsupported field.
                    tools = [
                        tool for tool in tools
                        if str((tool.get("function") or {}).get("name") or "") == forced_tool
                    ]
                messages.append({"role": "user", "content": f"Complete the current stage now by calling {forced_tool}."})
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "stage_finalization", "step": step, "stage": stage, "required_tool": forced_tool})
            response = self.llm_client.chat(messages=messages, tools=tools, tool_choice=tool_choice)
            llm_trace.append({"step": step, "stage": stage, "response": response})
            calls = response.get("tool_calls") or []
            if not calls:
                invalid += 1
                if invalid <= self.response_retries:
                    messages.append({"role": "user", "content": "Continue using the available native tool for the current localization stage."})
                    continue
                break
            invalid = 0
            messages.append({"role": "assistant", "content": response.get("content") or "", "tool_calls": calls})
            append_jsonl(run_dir / "trajectory.jsonl", {"event": "assistant_tool_calls", "step": step, "stage": stage, "tool_calls": calls})
            pending_stage_payloads: list[dict[str, Any]] = []
            allowed_tool_names = {
                str((tool.get("function") or {}).get("name") or "")
                for tool in tools
            }
            repo_calls_this_step = 0
            strategy_validation_calls_this_step = 0
            for call in calls:
                name = str((call.get("function") or {}).get("name") or "")
                if name not in allowed_tool_names:
                    message = _tool_error(
                        call,
                        f"Tool {name!r} is unavailable during {stage}. Use only the tools exposed for this stage.",
                    )
                    append_jsonl(
                        run_dir / "trajectory.jsonl",
                        {"event": "stage_tool_rejected", "step": step, "stage": stage, "tool_name": name},
                    )
                elif name == PROJECT_SKILL_TOOL:
                    if project_attempted:
                        message = _tool_error(call, "Project Skill retrieval was already attempted.")
                    else:
                        message, loaded_skill = self._load_stage_skill(
                            call, "project_skill", run_dir, step, repo_id=repo
                        )
                        loaded["project_skill"], project_attempted = loaded_skill, True
                        if not issue_revealed:
                            # Native tool calling requires the role=tool reply
                            # immediately after its assistant tool_call. Queue
                            # the stage-transition user message until all tool
                            # replies from this assistant turn have been added.
                            pending_stage_payloads.append({
                                "stage": "fault_skill_selection",
                                "objective": (
                                    "Classify the issue into one fixed Fault family using the repository "
                                    "orientation, then call load_fault_skill."
                                ),
                                "issue": issue,
                                "project_skill": loaded_skill,
                                "fault_taxonomy": compact_fault_taxonomy(),
                            })
                elif name == FAULT_SKILL_TOOL:
                    if not project_attempted:
                        message = _tool_error(call, "Project Skill retrieval must be attempted before the issue stage.")
                    elif issue_attempted:
                        message = _tool_error(call, "Fault Skill retrieval was already attempted.")
                    else:
                        try:
                            validate_fault_family(_tool_arguments(call).get("fault_family"))
                        except (TypeError, ValueError) as exc:
                            message = _tool_error(call, str(exc))
                        else:
                            message, loaded_skill = self._load_fault_skill(
                                call,
                                run_dir,
                                step,
                                issue=issue,
                                repository_manifest=manifest,
                                project_skill=loaded["project_skill"],
                            )
                            loaded["fault_skill"], issue_attempted = loaded_skill, True
                            issue_known_candidates, issue_known_paths = _observed_evidence_snapshot(
                                run_dir,
                                repo_path,
                            )
                            pending_stage_payloads.append({
                                "stage": "fault_guided_localization",
                                "objective": (
                                    "Use the issue as the primary anchor and the loaded Fault Skill as guidance. "
                                    "Collect repository evidence, form concrete function candidates, then call "
                                    "load_strategy_skill when a ranking uncertainty remains."
                                ),
                                "fault_skill": loaded_skill,
                            })
                elif name == STRATEGY_SKILL_TOOL:
                    if not issue_attempted:
                        message = _tool_error(call, "Fault Skill retrieval must be attempted before diagnostic ranking.")
                    elif strategy_attempted:
                        message = _tool_error(call, "Strategy Skill retrieval was already attempted.")
                    else:
                        message, loaded_skill = self._load_stage_skill(call, "strategy_skill", run_dir, step)
                        loaded["strategy_skill"], strategy_attempted = loaded_skill, True
                        strategy_validation_observations = 0
                        if loaded_skill is None:
                            strategy_transition_reason = "no_strategy_skill_matched"
                            progress = {
                                "event": "diagnostic_ranking_progress",
                                "step": step,
                                "strategy_skill_loaded": False,
                                "strategy_validation_observations": 0,
                                "verification_budget": self.strategy_skill_verification_budget,
                                "ready_to_finish": True,
                                "transition_reason": strategy_transition_reason,
                            }
                            append_jsonl(run_dir / "trajectory.jsonl", progress)
                            write_json(run_dir / "diagnostic_ranking_progress.json", progress)
                        pending_stage_payloads.append({
                            "stage": "diagnostic_ranking",
                            "objective": (
                                "Use repository evidence to resolve the candidate comparison. Treat the Strategy "
                                "Skill as a conditional hint, verify its distinguishing observation when one was "
                                "loaded, then call finish_localization."
                            ),
                            "strategy_skill": loaded_skill,
                        })
                elif name == FINISH_TOOL_NAME:
                    strategy_ready_to_finish = (
                        strategy_attempted
                        and (
                            loaded["strategy_skill"] is None
                            or strategy_validation_observations >= self.strategy_skill_verification_budget
                        )
                    )
                    if project_attempted and issue_attempted and strategy_ready_to_finish:
                        final_result = self._parse_finish_tool_call(call)
                        append_jsonl(run_dir / "trajectory.jsonl", {"event": "finish", "step": step, "result": final_result})
                        break
                    message = _tool_error(call, "Complete each Skill attempt and perform the required Strategy Skill verification before finish.")
                else:
                    strategy_validation_open = (
                        stage == "diagnostic_ranking"
                        and loaded["strategy_skill"] is not None
                        and strategy_validation_observations < self.strategy_skill_verification_budget
                    )
                    if (
                        stage == "diagnostic_ranking"
                        and loaded["strategy_skill"] is not None
                        and not strategy_validation_open
                    ):
                        message = _tool_error(
                            call,
                            "Strategy verification is complete. Call finish_localization now.",
                        )
                        append_jsonl(
                            run_dir / "trajectory.jsonl",
                            {"event": "strategy_verification_tool_rejected", "step": step, "tool_name": name},
                        )
                    elif strategy_validation_open and strategy_validation_calls_this_step >= 1:
                        message = _tool_error(
                            call,
                            "Use only one targeted repository observation to verify the loaded Strategy Skill, then finish.",
                        )
                        append_jsonl(
                            run_dir / "trajectory.jsonl",
                            {"event": "strategy_verification_tool_rejected", "step": step, "tool_name": name},
                        )
                    else:
                        message = self._execute_repo_tool(registry, call)
                        repo_calls_this_step += 1
                        if strategy_validation_open:
                            strategy_validation_calls_this_step += 1
                        if not project_attempted:
                            orientation_repo_calls += 1
                        if issue_attempted and not strategy_attempted:
                            issue_repo_observations += 1
                            if loaded["fault_skill"] is not None:
                                fault_skill_evidence_observations += 1
                            skill_id = (
                                loaded["fault_skill"].get("skill_id")
                                if loaded["fault_skill"] is not None
                                else None
                            )
                            append_jsonl(
                                run_dir / "trajectory.jsonl",
                                {
                                    "event": "fault_skill_evidence_observation",
                                    "step": step,
                                    "tool_name": name,
                                    "skill_id": skill_id,
                                    "count": fault_skill_evidence_observations,
                                    "repository_observation_count": issue_repo_observations,
                                },
                            )
                        if strategy_attempted and loaded["strategy_skill"] is not None:
                            strategy_validation_observations += 1
                            append_jsonl(
                                run_dir / "trajectory.jsonl",
                                {
                                    "event": "strategy_skill_evidence_observation",
                                    "step": step,
                                    "tool_name": name,
                                    "skill_id": loaded["strategy_skill"].get("skill_id"),
                                    "count": strategy_validation_observations,
                                },
                            )
                messages.append(message)
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "tool_result", "step": step, **message})
            if stage == "repository_orientation" and not project_attempted and repo_calls_this_step:
                _, observed_paths = _observed_evidence_snapshot(run_dir, repo_path)
                new_paths = sorted(observed_paths - orientation_known_paths)
                if new_paths:
                    orientation_stagnant_steps = 0
                else:
                    orientation_stagnant_steps += 1
                orientation_known_paths.update(observed_paths)
                orientation_sufficient = (
                    orientation_repo_calls >= int(
                        (self.config.get("explorer", {}) or {}).get("orientation_tool_budget", 3)
                    )
                    and len(orientation_known_paths) >= self.orientation_min_source_paths
                )
                orientation_stalled = (
                    orientation_repo_calls >= self.orientation_min_observations_before_stagnation
                    and orientation_stagnant_steps >= self.stage_stagnation_turns
                )
                if orientation_sufficient:
                    orientation_transition_reason = "orientation_sufficient"
                elif orientation_stalled:
                    orientation_transition_reason = "orientation_stagnated"
                progress = {
                    "event": "repository_orientation_progress",
                    "step": step,
                    "repository_observation_count": orientation_repo_calls,
                    "source_path_count": len(orientation_known_paths),
                    "new_source_paths": new_paths,
                    "consecutive_stagnant_steps": orientation_stagnant_steps,
                    "evidence_sufficient": orientation_sufficient,
                    "search_stagnated": orientation_stalled,
                    "transition_reason": orientation_transition_reason,
                }
                append_jsonl(run_dir / "trajectory.jsonl", progress)
                write_json(run_dir / "repository_orientation_progress.json", progress)
                if orientation_transition_reason:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Repository orientation is sufficient for a system model. "
                            "Call load_project_skill now with the repository summary, observed component roles, and responsibility boundaries."
                        ),
                    })
                    append_jsonl(
                        run_dir / "trajectory.jsonl",
                        {
                            "event": "stage_transition_recommended",
                            "step": step,
                            "from_stage": stage,
                            "to_stage": "fault_skill_selection",
                            "reason": orientation_transition_reason,
                        },
                    )
            if stage == "diagnostic_ranking" and strategy_attempted:
                if loaded["strategy_skill"] is None:
                    strategy_transition_reason = "no_strategy_skill_matched"
                elif strategy_validation_observations >= self.strategy_skill_verification_budget:
                    strategy_transition_reason = "strategy_skill_verified"
                progress = {
                    "event": "diagnostic_ranking_progress",
                    "step": step,
                    "strategy_skill_loaded": loaded["strategy_skill"] is not None,
                    "strategy_validation_observations": strategy_validation_observations,
                    "verification_budget": self.strategy_skill_verification_budget,
                    "ready_to_finish": strategy_transition_reason is not None,
                    "transition_reason": strategy_transition_reason,
                }
                append_jsonl(run_dir / "trajectory.jsonl", progress)
                write_json(run_dir / "diagnostic_ranking_progress.json", progress)
                if strategy_transition_reason:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Candidate comparison is complete. Call finish_localization now using the "
                            "concrete function candidates and repository evidence already collected."
                        ),
                    })
            if stage == "fault_guided_localization" and not strategy_attempted and repo_calls_this_step:
                observed_candidates, observed_paths = _observed_evidence_snapshot(run_dir, repo_path)
                new_candidates = sorted(observed_candidates - issue_known_candidates)
                new_paths = sorted(observed_paths - issue_known_paths)
                if new_candidates or new_paths:
                    issue_stagnant_steps = 0
                else:
                    issue_stagnant_steps += 1
                issue_known_candidates.update(observed_candidates)
                issue_known_paths.update(observed_paths)
                evidence_sufficient = (
                    issue_repo_observations >= self.fault_skill_evidence_budget
                    and len(issue_known_candidates) >= self.fault_candidate_min_count
                )
                stalled = (
                    issue_repo_observations >= self.fault_min_observations_before_stagnation
                    and issue_stagnant_steps >= self.stage_stagnation_turns
                )
                if evidence_sufficient:
                    issue_transition_reason = "evidence_sufficient"
                elif stalled:
                    issue_transition_reason = "search_stagnated"
                progress = {
                    "event": "fault_exploration_progress",
                    "step": step,
                    "repository_observation_count": issue_repo_observations,
                    "fault_skill_evidence_observations": fault_skill_evidence_observations,
                    "candidate_count": len(issue_known_candidates),
                    "source_path_count": len(issue_known_paths),
                    "new_candidates": new_candidates,
                    "new_source_paths": new_paths,
                    "consecutive_stagnant_steps": issue_stagnant_steps,
                    "evidence_sufficient": evidence_sufficient,
                    "search_stagnated": stalled,
                    "transition_reason": issue_transition_reason,
                }
                append_jsonl(run_dir / "trajectory.jsonl", progress)
                write_json(run_dir / "fault_exploration_progress.json", progress)
                if issue_transition_reason:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Repository evidence is ready for candidate comparison. "
                            "Call load_strategy_skill now with the concrete candidates, their relationship, "
                            "and the remaining ranking question."
                        ),
                    })
                    append_jsonl(
                        run_dir / "trajectory.jsonl",
                        {
                            "event": "stage_transition_recommended",
                            "step": step,
                            "from_stage": stage,
                            "to_stage": "diagnostic_ranking",
                            "reason": issue_transition_reason,
                        },
                    )
            for stage_payload in pending_stage_payloads:
                if stage_payload["stage"] == "fault_skill_selection":
                    write_json(run_dir / "issue_payload.json", stage_payload)
                    append_jsonl(run_dir / "trajectory.jsonl", {
                        "event": "issue_revealed", "step": step, "issue_payload": stage_payload,
                    })
                    issue_revealed = True
                else:
                    write_json(run_dir / f"{stage_payload['stage']}_instruction.json", stage_payload)
                messages.append({
                    "role": "user",
                    "content": json.dumps(stage_payload, ensure_ascii=False, indent=2),
                })
                append_jsonl(run_dir / "trajectory.jsonl", {
                    "event": "stage_instruction", "step": step,
                    "stage": stage_payload["stage"], "objective": stage_payload["objective"],
                })
            write_json(run_dir / "loaded_skills.json", loaded)
            if final_result:
                break

        # Protocol-safe convergence: mark unattempted stages explicitly, then force native finish.
        if not project_attempted:
            message, loaded["project_skill"] = self._load_stage_skill(_synthetic_tool_call(PROJECT_SKILL_TOOL, {
                "repository_summary": "insufficient orientation evidence",
                "component_roles": ["unknown component"],
                "responsibility_boundaries": ["unknown boundary"],
            }), "project_skill", run_dir, -1, repo_id=repo)
            project_attempted = True
            if not issue_revealed:
                issue_payload = {
                    "stage": "fault_skill_selection",
                    "objective": "Classify the issue into one Fault family, then call load_fault_skill.",
                    "issue": issue,
                    "project_skill": loaded["project_skill"],
                    "fault_taxonomy": compact_fault_taxonomy(),
                }
                write_json(run_dir / "issue_payload.json", issue_payload)
                messages.append({"role": "user", "content": json.dumps(issue_payload, ensure_ascii=False)})
                append_jsonl(run_dir / "trajectory.jsonl", {
                    "event": "issue_revealed", "step": "forced_project_stage", "issue_payload": issue_payload,
                })
                issue_revealed = True
        if not issue_attempted:
            message, loaded["fault_skill"] = self._load_fault_skill(
                _synthetic_tool_call(FAULT_SKILL_TOOL, {
                    "fault_family": "missing_functionality", "fault_signature": issue or "unknown issue",
                    "project_context": "insufficient orientation evidence", "suspected_path": "unknown"}),
                run_dir,
                -1,
                issue=issue,
                repository_manifest=manifest,
                project_skill=loaded["project_skill"],
            )
            issue_attempted = True
        if not strategy_attempted:
            message, loaded["strategy_skill"] = self._load_stage_skill(_synthetic_tool_call(STRATEGY_SKILL_TOOL, {
                "selected_skill_id": "none", "candidate_functions": [], "candidate_relationship": "unresolved",
                "unresolved_question": "finalization", "observed_evidence": "insufficient evidence", "selection_reason": "no selection"}), "strategy_skill", run_dir, -1)
            strategy_attempted = True
        if final_result is None:
            final_result = self._forced_finish(
                messages,
                run_dir,
                str(repo_path),
                llm_trace,
                reason="V3 stage budget reached.",
                compact_context=self._forced_finish_context(
                    issue=issue,
                    loaded_skills=loaded,
                    run_dir=run_dir,
                    repo_path=repo_path,
                    fault_skill_evidence_observations=fault_skill_evidence_observations,
                ),
            )
        result = {"instance_id": task.get("instance_id"), "status": "completed",
                  "ranked_functions": final_result["action"].get("ranked_functions", []),
                  "final_summary": final_result["action"].get("summary", ""), "finish": final_result,
                  "stage_completion": {"project_skill_attempted": project_attempted,
                                       "fault_skill_attempted": issue_attempted,
                                       "strategy_skill_attempted": strategy_attempted,
                                       "fault_skill_evidence_observations": fault_skill_evidence_observations,
                                       "issue_repository_observations": issue_repo_observations,
                                       "issue_candidate_count": len(issue_known_candidates),
                                       "issue_stagnant_steps": issue_stagnant_steps,
                                       "issue_transition_reason": issue_transition_reason,
                                       "orientation_source_path_count": len(orientation_known_paths),
                                       "orientation_stagnant_steps": orientation_stagnant_steps,
                                        "orientation_transition_reason": orientation_transition_reason,
                                        "strategy_validation_observations": strategy_validation_observations,
                                        "strategy_transition_reason": strategy_transition_reason}}
        write_json(run_dir / "loaded_skills.json", loaded)
        write_json(run_dir / "llm_trace.json", llm_trace)
        write_json(run_dir / "result.json", result)
        return result

    def _tools(
        self,
        registry: ToolRegistry,
        stage: str,
        *,
        fault_skill_loaded: bool,
        fault_skill_evidence_observations: int,
    ) -> list[dict[str, Any]]:
        if stage == "repository_orientation":
            return [*registry.list_openai_tools(), _project_skill_tool_schema()]
        if stage == "fault_skill_selection":
            return [_fault_skill_tool_schema()]
        if stage == "fault_guided_localization":
            tools = [*registry.list_openai_tools()]
            if (
                not fault_skill_loaded
                or fault_skill_evidence_observations >= self.fault_skill_evidence_budget
            ):
                tools.append(_strategy_skill_tool_schema(self.skill_bank.strategy_catalog()))
            return tools
        return [*registry.list_openai_tools(), _finish_tool_schema()]

    def _required_v3_control_tool(
        self,
        *,
        step: int,
        project_attempted: bool,
        issue_attempted: bool,
        strategy_attempted: bool,
        strategy_skill_loaded: bool,
        strategy_validation_observations: int,
        orientation_repo_calls: int,
        fault_skill_loaded: bool,
        fault_skill_evidence_observations: int,
        issue_repo_observations: int,
        issue_candidate_count: int,
        issue_stagnant_steps: int,
        orientation_source_path_count: int,
        orientation_stagnant_steps: int,
    ) -> str | None:
        """Advance on evidence sufficiency or stalled exploration, then converge."""
        orientation_sufficient = (
            orientation_repo_calls >= int(
                (self.config.get("explorer", {}) or {}).get("orientation_tool_budget", 3)
            )
            and orientation_source_path_count >= self.orientation_min_source_paths
        )
        orientation_stalled = (
            orientation_repo_calls >= self.orientation_min_observations_before_stagnation
            and orientation_stagnant_steps >= self.stage_stagnation_turns
        )
        if not project_attempted and (orientation_sufficient or orientation_stalled):
            return PROJECT_SKILL_TOOL
        # Fault classification is a required transition, even when no subtype is loaded.
        # Completing it immediately preserves a usable evidence window after a
        # matching card is injected.
        if project_attempted and not issue_attempted:
            return FAULT_SKILL_TOOL
        evidence_sufficient = (
            issue_repo_observations >= self.fault_skill_evidence_budget
            and issue_candidate_count >= self.fault_candidate_min_count
        )
        search_stagnated = (
            issue_repo_observations >= self.fault_min_observations_before_stagnation
            and issue_stagnant_steps >= self.stage_stagnation_turns
        )
        if issue_attempted and not strategy_attempted and (evidence_sufficient or search_stagnated):
            return STRATEGY_SKILL_TOOL
        if strategy_attempted and (
            not strategy_skill_loaded
            or strategy_validation_observations >= self.strategy_skill_verification_budget
        ):
            return FINISH_TOOL_NAME
        if self.max_steps is None:
            return None
        remaining = self.max_steps - step + 1
        needs_strategy_verification = (
            strategy_attempted
            and strategy_skill_loaded
            and strategy_validation_observations < self.strategy_skill_verification_budget
        )
        needed = int(not project_attempted) + int(not issue_attempted) + int(not strategy_attempted) + int(needs_strategy_verification) + 1
        if remaining > max(self.finalization_steps, needed):
            return None
        if not project_attempted:
            return PROJECT_SKILL_TOOL
        if not issue_attempted:
            return FAULT_SKILL_TOOL
        if (
            fault_skill_loaded
            and fault_skill_evidence_observations < self.fault_skill_evidence_budget
        ):
            return None
        if not strategy_attempted:
            return STRATEGY_SKILL_TOOL
        if not strategy_skill_loaded or strategy_validation_observations >= self.strategy_skill_verification_budget:
            return FINISH_TOOL_NAME
        return None

    def _forced_finish_context(
        self,
        *,
        issue: str,
        loaded_skills: dict[str, Any],
        run_dir: Path,
        repo_path: Path,
        fault_skill_evidence_observations: int,
    ) -> dict[str, Any]:
        """Build the bounded evidence packet used by V3 native finalization."""
        evidence = _collect_evidence_from_trajectory(run_dir / "trajectory.jsonl")
        observed = _rank_evidence_candidates(repo_path, evidence, limit=8)
        return {
            "issue": _truncate_text(issue, 6000),
            "loaded_skills": {
                name: _compact_skill_for_finish(skill)
                for name, skill in loaded_skills.items()
                if skill is not None
            },
            "fault_skill_evidence_observations": fault_skill_evidence_observations,
            "observed_function_candidates": observed,
        }

    def _load_fault_skill(
        self,
        call: dict[str, Any],
        run_dir: Path,
        step: int,
        *,
        issue: str,
        repository_manifest: dict[str, Any],
        project_skill: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        try:
            args = _tool_arguments(call)
            project_request = _read_json_if_present(run_dir / "project_skill_request.json")
            orientation = {
                "project_request": project_request.get("request") or {},
                "repository_root_entries": (repository_manifest.get("root_entries") or [])[:20],
            }
            compact_project_skill = _compact_project_context(project_skill)
            selection_context = _fault_selection_context(
                args,
                orientation=orientation,
                project_skill=compact_project_skill,
            )
            family = validate_fault_family(args.get("fault_family"))
            if not selection_context:
                raise ValueError("Fault Skill request was empty.")
            request_context = {
                "issue_report": issue,
                "orientation": orientation,
                "project_skill": compact_project_skill,
            }
            request = {
                "event": "fault_skill_request",
                "step": step,
                "request": args,
                "request_context": request_context,
                "selection_context": selection_context,
            }
            append_jsonl(run_dir / "trajectory.jsonl", request)
            write_json(run_dir / "fault_skill_request.json", request)
            if "fault_skill" not in self.skill_bank.enabled_skill_types:
                matched, trace = None, {
                    "candidate_selected_skill_ids": [],
                    "selector": {"selected_skill_id": None, "reason": "Fault Skill loading disabled."},
                    "validator": {"applicable": False, "reason": "Fault Skill loading disabled."},
                    "notes": ["Fault Skill loading disabled."],
                }
            else:
                search = self.skill_bank.fault_catalog(family)
                candidates = search.get("candidate_skills") or []
                catalog_json = json.dumps(candidates, ensure_ascii=False)
                selector = {"selected_skill_id": None, "reason": "No Fault Skill exists in this family."}
                selector_attempts: list[dict[str, Any]] = []
                selector_protocol_failed = False
                if candidates:
                    selector_payload = {
                        "issue_report": issue,
                        "orientation": orientation,
                        "project_skill": compact_project_skill,
                        "request": args,
                        "candidates": candidates,
                    }
                    selector, selector_attempts = self._select_fault_skill(
                        selector_payload,
                        candidate_ids={str(item.get("skill_id")) for item in candidates},
                    )
                    selector_protocol_failed = not bool(selector_attempts[-1].get("valid"))
                selected_id = str(selector.get("selected_skill_id") or "").strip()
                candidate_ids = {str(item.get("skill_id")) for item in candidates}
                matched = self.skill_bank.get_active_skill(selected_id, skill_type="fault_skill") if selected_id in candidate_ids else None
                validation = {"applicable": False, "reason": "No candidate selected."}
                if matched is not None:
                    validator_response = self.llm_client.chat(
                        messages=[{"role": "system", "content": self.fault_validator_prompt},
                                  {"role": "user", "content": json.dumps({
                                      "issue_report": issue,
                                      "orientation": orientation,
                                      "project_skill": compact_project_skill,
                                      "request": args,
                                      "candidate": matched,
                                  }, ensure_ascii=False)}],
                        tool_choice="none", response_format={"type": "json_object"}, temperature=0,
                        max_tokens=self.project_skill_validator_max_tokens,
                    )
                    validation = extract_json_object(validator_response.get("content") or "{}")
                    if not bool(validation.get("applicable")):
                        matched = None
                trace = search.get("skill_search_trace") or {}
                trace["catalog_char_count"] = len(catalog_json)
                trace["selector_input_char_count"] = len(json.dumps({
                    "issue_report": issue,
                    "orientation": orientation,
                    "project_skill": compact_project_skill,
                    "request": args,
                    "candidates": candidates,
                }, ensure_ascii=False))
                trace["selector"] = selector
                trace["selector_attempts"] = selector_attempts
                trace["selector_protocol_failed"] = selector_protocol_failed
                trace["validator"] = validation
            payload = {"skill_type": "fault_skill", "fault_family": family,
                       "selection_context": selection_context,
                       "matched_skill": matched, "search_trace": trace}
        except Exception as exc:  # retrieval failure is a completed empty attempt
            payload = {"skill_type": "fault_skill", "matched_skill": None, "search_trace": {"notes": [str(exc)]}}
        write_json(run_dir / "fault_skill_search.json", payload)
        append_jsonl(run_dir / "trajectory.jsonl", {"event": "fault_skill_loaded", "step": step,
            "loaded_skill_id": (payload.get("matched_skill") or {}).get("skill_id"), **payload})
        return _tool_success(call, payload, name=FAULT_SKILL_TOOL, wrapped=False), payload.get("matched_skill")

    def _select_fault_skill(
        self,
        payload: dict[str, Any],
        *,
        candidate_ids: set[str],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run the selector with fixed-schema retries and auditable diagnostics."""

        attempts: list[dict[str, Any]] = []
        attempt_max_tokens = self.fault_skill_selector_max_tokens
        for attempt_number in range(1, self.fault_skill_selector_attempts + 1):
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.fault_selector_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                tool_choice="none",
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=attempt_max_tokens,
            )
            content = str(response.get("content") or "")
            diagnostic = _selector_response_diagnostic(response, attempt_number=attempt_number)
            diagnostic["requested_max_tokens"] = attempt_max_tokens
            try:
                parsed = extract_json_object(content)
                selected = parsed.get("selected_skill_id")
                selected_id = str(selected or "").strip()
                reason = str(parsed.get("reason") or "").strip()
                has_selection_field = "selected_skill_id" in parsed
                known_selection = not selected_id or selected_id in candidate_ids
                valid = has_selection_field and known_selection and bool(reason)
                diagnostic.update({
                    "valid": valid,
                    "parsed_selected_skill_id": selected_id or None,
                    "validation_error": None if valid else _selector_validation_error(
                        has_selection_field=has_selection_field,
                        known_selection=known_selection,
                        has_reason=bool(reason),
                    ),
                })
                attempts.append(diagnostic)
                if valid:
                    return {
                        "selected_skill_id": selected_id or None,
                        "reason": reason,
                    }, attempts
            except (TypeError, ValueError) as exc:
                diagnostic.update({"valid": False, "validation_error": str(exc)})
                attempts.append(diagnostic)
            if diagnostic.get("finish_reason") == "length":
                attempt_max_tokens = self.fault_skill_selector_retry_max_tokens

        return {
            "selected_skill_id": None,
            "reason": (
                "Selector protocol failed after "
                f"{len(attempts)} fixed-schema attempts; no Fault Skill was loaded."
            ),
        }, attempts


def _read_json_if_present(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _selector_response_diagnostic(
    response: dict[str, Any],
    *,
    attempt_number: int,
) -> dict[str, Any]:
    raw = response.get("raw") if isinstance(response.get("raw"), dict) else {}
    choice = (raw.get("choices") or [{}])[0]
    choice = choice if isinstance(choice, dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = str(response.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    return {
        "attempt": attempt_number,
        "model": raw.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
        "content": content,
        "content_char_count": len(content),
        "reasoning_content": reasoning,
        "reasoning_char_count": len(reasoning),
    }


def _selector_validation_error(
    *,
    has_selection_field: bool,
    known_selection: bool,
    has_reason: bool,
) -> str:
    problems: list[str] = []
    if not has_selection_field:
        problems.append("missing selected_skill_id")
    if not known_selection:
        problems.append("selected_skill_id is not in the supplied catalog")
    if not has_reason:
        problems.append("missing reason")
    return "; ".join(problems) or "invalid selector response"


def _stage(project_attempted: bool, issue_attempted: bool, strategy_attempted: bool) -> str:
    if not project_attempted:
        return "repository_orientation"
    if not issue_attempted:
        return "fault_skill_selection"
    if not strategy_attempted:
        return "fault_guided_localization"
    return "diagnostic_ranking"
