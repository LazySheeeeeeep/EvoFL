"""V3 Explorer: staged Project -> Issue -> Strategy Skill loading.

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
from evolutefl.tools import ToolRegistry, register_builtin_tools
from evolutefl.tools.registry import openai_function_tool

from .agent import (
    FINISH_TOOL_NAME,
    PROJECT_SKILL_TOOL,
    STRATEGY_SKILL_TOOL,
    ExplorerAgent,
    _finish_tool_schema,
    _collect_evidence_from_trajectory,
    _named_tool_choice,
    _project_skill_tool_schema,
    _rank_evidence_candidates,
    _synthetic_tool_call,
    _tool_arguments,
    _tool_error,
    _tool_snapshot,
    _tool_success,
    _strategy_skill_tool_schema,
)

ISSUE_SKILL_TOOL = "load_issue_skill"


def _issue_skill_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=ISSUE_SKILL_TOOL,
        description=(
            "Load one Issue Skill after the issue is available. Describe the symptom and the "
            "observed system boundary in reusable terms, then use the returned card to guide "
            "source-code exploration."
        ),
        properties={
            "issue_type": {"type": "string", "description": "A reusable symptom or issue family."},
            "issue_signature": {"type": "string", "description": "A concise abstract behavior mismatch."},
            "project_context": {"type": "string", "description": "Observed architecture or responsibility boundary."},
            "suspected_path": {"type": "string", "description": "Likely code path or boundary to inspect next."},
        },
        required=["issue_type", "issue_signature", "project_context", "suspected_path"],
    )


def _issue_query(arguments: dict[str, Any]) -> str:
    return "\n".join(
        f"{key}: {str(arguments.get(key) or '').strip()}"
        for key in ("issue_type", "issue_signature", "project_context", "suspected_path")
        if str(arguments.get(key) or "").strip()
    )


def _compact_project_context(skill: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only the Project model needed to interpret an Issue request."""
    if not isinstance(skill, dict):
        return None
    return {
        key: skill.get(key)
        for key in ("skill_id", "value", "title", "trigger")
        if skill.get(key)
    }


def _issue_retrieval_query(
    arguments: dict[str, Any],
    *,
    orientation: dict[str, Any],
    project_skill: dict[str, Any] | None,
) -> str:
    """Build the compact semantic query used only for Issue Skill recall."""
    sections = ["issue request:\n" + _issue_query(arguments)]
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
        self.issue_selector_prompt = resolve_path(cfg.get(
            "issue_skill_selector_prompt_path", "prompt_records/explorer/issue_skill_selector_v3.txt"
        )).read_text(encoding="utf-8")
        self.issue_validator_prompt = resolve_path(cfg.get(
            "issue_skill_validator_prompt_path", "prompt_records/explorer/issue_skill_validator_v3.txt"
        )).read_text(encoding="utf-8")
        self.issue_skill_evidence_budget = max(0, int(cfg.get("issue_skill_evidence_budget", 2)))

    def run(self, task: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        issue = str(task.get("bug_report") or "").strip()
        if not issue:
            raise ValueError(
                "V3 Explorer requires a non-empty bug_report/problem_statement; "
                "repository-only browsing cannot perform the Issue Skill stage."
            )
        run_dir = Path(task["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        repo_path = Path(task["repo_path"]).resolve()
        repo = str(task.get("repo") or "")
        registry = ToolRegistry()
        register_builtin_tools(registry, run_dir=run_dir, repo_path=str(repo_path))
        write_json(run_dir / "tools.json", [
            *registry.list_tools(), _tool_snapshot(_project_skill_tool_schema()),
            _tool_snapshot(_issue_skill_tool_schema()),
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
            "instance_id": task.get("instance_id"),
            "repo": repo,
            "base_commit": task.get("base_commit", ""),
            "repository_manifest": manifest,
        }
        write_json(run_dir / "initial_payload.json", initial_payload)
        write_json(run_dir / "repository_manifest.json", manifest)
        loaded: dict[str, Any] = {"project_skill": None, "issue_skill": None, "strategy_skill": None}
        write_json(run_dir / "loaded_skills.json", loaded)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(initial_payload, ensure_ascii=False, indent=2)}]
        project_attempted = issue_attempted = strategy_attempted = issue_revealed = False
        strategy_evidence_observed = True
        issue_skill_evidence_observations = 0
        orientation_repo_calls = 0
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
                issue_skill_loaded=loaded["issue_skill"] is not None,
                issue_skill_evidence_observations=issue_skill_evidence_observations,
            )
            forced_tool = self._required_v3_control_tool(
                step=step, project_attempted=project_attempted, issue_attempted=issue_attempted,
                strategy_attempted=strategy_attempted, strategy_evidence_observed=strategy_evidence_observed,
                orientation_repo_calls=orientation_repo_calls,
                issue_skill_loaded=loaded["issue_skill"] is not None,
                issue_skill_evidence_observations=issue_skill_evidence_observations,
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
                        message, loaded_skill = self._load_stage_skill(call, "project_skill", run_dir, step)
                        loaded["project_skill"], project_attempted = loaded_skill, True
                        if not issue_revealed:
                            # Native tool calling requires the role=tool reply
                            # immediately after its assistant tool_call. Queue
                            # the stage-transition user message until all tool
                            # replies from this assistant turn have been added.
                            pending_stage_payloads.append({
                                "stage": "issue_skill_selection",
                                "objective": (
                                    "Use the issue and repository orientation to describe a reusable issue "
                                    "pattern, then call load_issue_skill."
                                ),
                                "issue": issue,
                                "project_skill": loaded_skill,
                            })
                elif name == ISSUE_SKILL_TOOL:
                    if not project_attempted:
                        message = _tool_error(call, "Project Skill retrieval must be attempted before the issue stage.")
                    elif issue_attempted:
                        message = _tool_error(call, "Issue Skill retrieval was already attempted.")
                    else:
                        message, loaded_skill = self._load_issue_skill(
                            call,
                            run_dir,
                            step,
                            issue=issue,
                            repository_manifest=manifest,
                            project_skill=loaded["project_skill"],
                        )
                        loaded["issue_skill"], issue_attempted = loaded_skill, True
                        pending_stage_payloads.append({
                            "stage": "issue_guided_localization",
                            "objective": (
                                "Use the issue as the primary anchor and the loaded Issue Skill as guidance. "
                                "Collect repository evidence, form concrete function candidates, then call "
                                "load_strategy_skill when a ranking uncertainty remains."
                            ),
                            "issue_skill": loaded_skill,
                        })
                elif name == STRATEGY_SKILL_TOOL:
                    if not issue_attempted:
                        message = _tool_error(call, "Issue Skill retrieval must be attempted before diagnostic ranking.")
                    elif strategy_attempted:
                        message = _tool_error(call, "Strategy Skill retrieval was already attempted.")
                    else:
                        message, loaded_skill = self._load_stage_skill(call, "strategy_skill", run_dir, step)
                        loaded["strategy_skill"], strategy_attempted = loaded_skill, True
                        strategy_evidence_observed = loaded_skill is None
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
                    if project_attempted and issue_attempted and strategy_attempted and strategy_evidence_observed:
                        final_result = self._parse_finish_tool_call(call)
                        append_jsonl(run_dir / "trajectory.jsonl", {"event": "finish", "step": step, "result": final_result})
                        break
                    message = _tool_error(call, "Complete each Skill attempt and test a loaded Strategy Skill with grep, read_file, or write before finish.")
                else:
                    message = self._execute_repo_tool(registry, call)
                    if not project_attempted:
                        orientation_repo_calls += 1
                    if issue_attempted and loaded["issue_skill"] is not None and not strategy_attempted:
                        issue_skill_evidence_observations += 1
                        append_jsonl(
                            run_dir / "trajectory.jsonl",
                            {
                                "event": "issue_skill_evidence_observation",
                                "step": step,
                                "tool_name": name,
                                "skill_id": loaded["issue_skill"].get("skill_id"),
                                "count": issue_skill_evidence_observations,
                            },
                        )
                    if strategy_attempted and loaded["strategy_skill"] is not None:
                        strategy_evidence_observed = True
                        append_jsonl(run_dir / "trajectory.jsonl", {"event": "strategy_skill_evidence_observation", "step": step, "tool_name": name, "skill_id": loaded["strategy_skill"].get("skill_id")})
                messages.append(message)
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "tool_result", "step": step, **message})
            for stage_payload in pending_stage_payloads:
                if stage_payload["stage"] == "issue_skill_selection":
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
                "project_type": "unknown", "architecture_signature": "insufficient orientation evidence",
                "component_roles": ["unknown component"], "responsibility_boundary": "unknown boundary"}), "project_skill", run_dir, -1)
            project_attempted = True
            if not issue_revealed:
                issue_payload = {
                    "stage": "issue_skill_selection",
                    "objective": "Use the issue and repository orientation, then call load_issue_skill.",
                    "issue": issue,
                    "project_skill": loaded["project_skill"],
                }
                write_json(run_dir / "issue_payload.json", issue_payload)
                messages.append({"role": "user", "content": json.dumps(issue_payload, ensure_ascii=False)})
                append_jsonl(run_dir / "trajectory.jsonl", {
                    "event": "issue_revealed", "step": "forced_project_stage", "issue_payload": issue_payload,
                })
                issue_revealed = True
        if not issue_attempted:
            message, loaded["issue_skill"] = self._load_issue_skill(
                _synthetic_tool_call(ISSUE_SKILL_TOOL, {
                    "issue_type": "unknown", "issue_signature": issue or "unknown issue",
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
                    issue_skill_evidence_observations=issue_skill_evidence_observations,
                ),
            )
        result = {"instance_id": task.get("instance_id"), "status": "completed",
                  "ranked_functions": final_result["action"].get("ranked_functions", []),
                  "final_summary": final_result["action"].get("summary", ""), "finish": final_result,
                  "stage_completion": {"project_skill_attempted": project_attempted,
                                       "issue_skill_attempted": issue_attempted,
                                       "strategy_skill_attempted": strategy_attempted,
                                       "issue_skill_evidence_observations": issue_skill_evidence_observations}}
        write_json(run_dir / "loaded_skills.json", loaded)
        write_json(run_dir / "llm_trace.json", llm_trace)
        write_json(run_dir / "result.json", result)
        return result

    def _tools(
        self,
        registry: ToolRegistry,
        stage: str,
        *,
        issue_skill_loaded: bool,
        issue_skill_evidence_observations: int,
    ) -> list[dict[str, Any]]:
        if stage == "repository_orientation":
            return [*registry.list_openai_tools(), _project_skill_tool_schema()]
        if stage == "issue_skill_selection":
            return [_issue_skill_tool_schema()]
        if stage == "issue_guided_localization":
            tools = [*registry.list_openai_tools()]
            if (
                not issue_skill_loaded
                or issue_skill_evidence_observations >= self.issue_skill_evidence_budget
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
        strategy_evidence_observed: bool,
        orientation_repo_calls: int,
        issue_skill_loaded: bool,
        issue_skill_evidence_observations: int,
    ) -> str | None:
        """Reserve enough steps for all control transitions near the budget."""
        if not project_attempted and orientation_repo_calls >= int(
            (self.config.get("explorer", {}) or {}).get("orientation_tool_budget", 3)
        ):
            return PROJECT_SKILL_TOOL
        # The Issue card is a required transition, not an optional side path.
        # Completing it immediately preserves a usable evidence window after a
        # matching card is injected.
        if project_attempted and not issue_attempted:
            return ISSUE_SKILL_TOOL
        if self.max_steps is None:
            return None
        remaining = self.max_steps - step + 1
        needed = int(not project_attempted) + int(not issue_attempted) + int(not strategy_attempted) + int(not strategy_evidence_observed) + 1
        if remaining > max(self.finalization_steps, needed):
            return None
        if not project_attempted:
            return PROJECT_SKILL_TOOL
        if not issue_attempted:
            return ISSUE_SKILL_TOOL
        if (
            issue_skill_loaded
            and issue_skill_evidence_observations < self.issue_skill_evidence_budget
        ):
            return None
        if not strategy_attempted:
            return STRATEGY_SKILL_TOOL
        if strategy_evidence_observed:
            return FINISH_TOOL_NAME
        return None

    def _forced_finish_context(
        self,
        *,
        issue: str,
        loaded_skills: dict[str, Any],
        run_dir: Path,
        repo_path: Path,
        issue_skill_evidence_observations: int,
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
            "issue_skill_evidence_observations": issue_skill_evidence_observations,
            "observed_function_candidates": observed,
        }

    def _load_issue_skill(
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
            query = _issue_retrieval_query(
                args,
                orientation=orientation,
                project_skill=compact_project_skill,
            )
            if not query:
                raise ValueError("Issue Skill request was empty.")
            request_context = {
                "issue_report": issue,
                "orientation": orientation,
                "project_skill": compact_project_skill,
            }
            request = {
                "event": "issue_skill_request",
                "step": step,
                "request": args,
                "request_context": request_context,
                "retrieval_query": query,
            }
            append_jsonl(run_dir / "trajectory.jsonl", request)
            write_json(run_dir / "issue_skill_request.json", request)
            if "issue_skill" not in self.skill_bank.enabled_skill_types:
                matched, trace = None, {
                    "candidate_selected_skill_ids": [],
                    "selector": {"selected_skill_id": None, "reason": "Issue Skill loading disabled."},
                    "validator": {"applicable": False, "reason": "Issue Skill loading disabled."},
                    "notes": ["Issue Skill loading disabled."],
                }
            else:
                search = self.skill_bank.search_issue_candidates(query, limit=self.project_skill_candidate_limit)
                candidates = search.get("candidate_skills") or []
                selector = {"selected_skill_id": None, "reason": "No threshold-qualified Issue Skill candidate."}
                if candidates:
                    selector_response = self.llm_client.chat(
                        messages=[{"role": "system", "content": self.issue_selector_prompt},
                                  {"role": "user", "content": json.dumps({
                                      "issue_report": issue,
                                      "orientation": orientation,
                                      "project_skill": compact_project_skill,
                                      "request": args,
                                      "candidates": candidates,
                                  }, ensure_ascii=False)}],
                        tool_choice="none", response_format={"type": "json_object"}, temperature=0,
                        max_tokens=self.project_skill_selector_max_tokens,
                    )
                    selector = extract_json_object(selector_response.get("content") or "{}")
                selected_id = str(selector.get("selected_skill_id") or "").strip()
                candidate_ids = {str(item.get("skill_id")) for item in candidates}
                matched = self.skill_bank.get_active_skill(selected_id, skill_type="issue_skill") if selected_id in candidate_ids else None
                validation = {"applicable": False, "reason": "No candidate selected."}
                if matched is not None:
                    validator_response = self.llm_client.chat(
                        messages=[{"role": "system", "content": self.issue_validator_prompt},
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
                trace["selector"] = selector
                trace["validator"] = validation
            payload = {"skill_type": "issue_skill", "query": args.get("issue_type"), "retrieval_query": query,
                       "matched_skill": matched, "search_trace": trace}
        except Exception as exc:  # retrieval failure is a completed empty attempt
            payload = {"skill_type": "issue_skill", "matched_skill": None, "search_trace": {"notes": [str(exc)]}}
        write_json(run_dir / "issue_skill_search.json", payload)
        append_jsonl(run_dir / "trajectory.jsonl", {"event": "issue_skill_loaded", "step": step,
            "loaded_skill_id": (payload.get("matched_skill") or {}).get("skill_id"), **payload})
        return _tool_success(call, payload, name=ISSUE_SKILL_TOOL, wrapped=False), payload.get("matched_skill")


def _read_json_if_present(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stage(project_attempted: bool, issue_attempted: bool, strategy_attempted: bool) -> str:
    if not project_attempted:
        return "repository_orientation"
    if not issue_attempted:
        return "issue_skill_selection"
    if not strategy_attempted:
        return "issue_guided_localization"
    return "diagnostic_ranking"
