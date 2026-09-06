from __future__ import annotations

import ast
import collections
import json
import time
from pathlib import Path
from typing import Any

from evolutefl.config import resolve_path
from evolutefl.json_utils import append_jsonl, extract_json_object, write_json
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import SkillBankV0, make_skill_bank
from evolutefl.tools import ToolRegistry, register_builtin_tools
from evolutefl.tools.registry import openai_function_tool


PROJECT_SKILL_TOOL = "load_project_skill"
STRATEGY_SKILL_TOOL = "load_strategy_skill"
PROJECT_SKILL_SELECTOR_TOOL = "select_project_skill"
PROJECT_SKILL_VALIDATOR_TOOL = "validate_project_skill"
STRATEGY_SKILL_VALIDATOR_TOOL = "validate_strategy_skill"
FINISH_TOOL_NAME = "finish_localization"
STRATEGY_EVIDENCE_TOOL = "repository_evidence"


class ValidatorProtocolError(ValueError):
    """A structured validator did not produce a usable decision after retries."""

    def __init__(
        self,
        message: str,
        *,
        attempt_count: int,
        response: dict[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.attempt_count = attempt_count
        self.response = response


class ExplorerAgent:
    """Three-stage fault-localization agent with mid-trajectory skill loading."""

    def __init__(
        self,
        *,
        llm_client: Any,
        skill_bank: SkillBankV0,
        config: dict[str, Any],
        system_prompt: str,
    ) -> None:
        self.llm_client = llm_client
        self.skill_bank = skill_bank
        self.config = config
        self.system_prompt = system_prompt
        explorer_cfg = config.get("explorer", {})
        self.max_steps = _parse_optional_positive_int(explorer_cfg.get("max_steps", 30))
        runtime_limit = explorer_cfg.get("max_runtime_seconds")
        self.max_runtime_seconds = (
            float(runtime_limit) if runtime_limit not in (None, "", "none", "None") else None
        )
        self.response_retries = int(explorer_cfg.get("response_retries", 2))
        self.finalization_steps = max(3, int(explorer_cfg.get("finalization_steps", 3)))
        self.forced_finish_max_tokens = int(explorer_cfg.get("forced_finish_max_tokens", 4096))
        # Candidate recall is wider than the one-card runtime injection. The
        # selector and structural validator still reject irrelevant cards.
        self.project_skill_candidate_limit = max(
            1, int(explorer_cfg.get("project_skill_candidate_limit", 8))
        )
        self.project_skill_selector_max_tokens = max(
            512, int(explorer_cfg.get("project_skill_selector_max_tokens", 2048))
        )
        self.project_skill_validator_max_tokens = max(
            256, int(explorer_cfg.get("project_skill_validator_max_tokens", 4096))
        )
        self.strategy_skill_validator_max_tokens = max(
            256, int(explorer_cfg.get("strategy_skill_validator_max_tokens", 4096))
        )
        selector_prompt_path = resolve_path(
            explorer_cfg.get(
                "project_skill_selector_prompt_path",
                "prompt_records/explorer/project_skill_selector_v0.txt",
            )
        )
        self.project_skill_selector_prompt = selector_prompt_path.read_text(encoding="utf-8")
        validator_prompt_path = Path(
            explorer_cfg.get(
                "project_skill_validator_prompt_path",
                "prompt_records/explorer/project_skill_validator_v0.txt",
            )
        )
        self.project_skill_validator_prompt = validator_prompt_path.read_text(encoding="utf-8")
        strategy_validator_prompt_path = resolve_path(
            explorer_cfg.get(
                "strategy_skill_validator_prompt_path",
                "prompt_records/explorer/strategy_skill_validator_v0.txt",
            )
        )
        self.strategy_skill_validator_prompt = strategy_validator_prompt_path.read_text(encoding="utf-8")

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(task["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        repo_path = str(Path(task["repo_path"]).resolve())
        repo = str(task.get("repo") or "")
        issue = str(task.get("bug_report") or "")

        registry = ToolRegistry()
        register_builtin_tools(registry, run_dir=run_dir, repo_path=repo_path)
        write_json(
            run_dir / "tools.json",
            [
                *registry.list_tools(),
                _tool_snapshot(_project_skill_tool_schema()),
                _tool_snapshot(_strategy_skill_tool_schema(self.skill_bank.strategy_catalog())),
                _tool_snapshot(_finish_tool_schema()),
            ],
        )

        user_payload = {
            "instance_id": task.get("instance_id"),
            "repo": repo,
            "base_commit": task.get("base_commit", ""),
            "bug_report": issue,
        }
        write_json(run_dir / "initial_payload.json", user_payload)
        write_json(run_dir / "loaded_skills.json", {"project_skill": None, "strategy_skill": None})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]
        llm_trace: list[dict[str, Any]] = []
        # Ablation runs may intentionally suppress one card type.  They still
        # traverse both control stages: otherwise Strategy-only would form its
        # diagnostic request before the observations normally made during the
        # Project stage.  A disabled stage therefore returns an explicit empty
        # result when invoked, while preserving the same conversation shape.
        project_enabled = "project_skill" in self.skill_bank.enabled_skill_types
        strategy_enabled = "strategy_skill" in self.skill_bank.enabled_skill_types
        project_attempted = False
        strategy_attempted = False
        strategy_evidence_observed = True
        loaded_skills: dict[str, Any] = {"project_skill": None, "strategy_skill": None}
        final_result: dict[str, Any] | None = None
        invalid_responses = 0
        started_at = time.monotonic()
        step = 0

        while self.max_steps is None or step < self.max_steps:
            step += 1
            if self._runtime_expired(started_at):
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {"event": "runtime_timeout", "step": step, "max_runtime_seconds": self.max_runtime_seconds},
                )
                project_attempted, strategy_attempted = self._complete_missing_stages(
                    messages=messages,
                    run_dir=run_dir,
                    repo=repo,
                    issue=issue,
                    project_attempted=project_attempted,
                    strategy_attempted=strategy_attempted,
                    loaded_skills=loaded_skills,
                    reason="Runtime timeout reached.",
                )
                final_result = self._forced_finish(
                    messages, run_dir, repo_path, llm_trace, reason="Runtime timeout reached."
                )
                break

            stage = _stage_name(project_attempted, strategy_attempted)
            forced_tool = self._required_tool_for_remaining_steps(
                step=step,
                project_attempted=project_attempted,
                strategy_attempted=strategy_attempted,
                strategy_evidence_observed=strategy_evidence_observed,
            )
            tools = self._tools_for_stage(
                registry,
                stage,
                strategy_evidence_observed=strategy_evidence_observed,
            )
            tool_choice: str | dict[str, Any] = "auto"
            if forced_tool:
                tool_choice = (
                    "required"
                    if forced_tool == STRATEGY_EVIDENCE_TOOL
                    else _named_tool_choice(forced_tool)
                )
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {"event": "stage_finalization", "step": step, "stage": stage, "required_tool": forced_tool},
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Use grep, read_file, or write now to test the loaded Strategy Skill's "
                            "distinguishing condition against repository evidence."
                            if forced_tool == STRATEGY_EVIDENCE_TOOL
                            else _forced_stage_prompt(forced_tool)
                        ),
                    }
                )

            response = self.llm_client.chat(messages=messages, tools=tools, tool_choice=tool_choice)
            llm_trace.append({"step": step, "stage": stage, "response": response})
            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {"event": "assistant_tool_calls", "step": step, "stage": stage, "tool_calls": tool_calls},
                )
                for tool_call in tool_calls:
                    name = str(((tool_call.get("function") or {}).get("name") or ""))
                    if name == PROJECT_SKILL_TOOL:
                        if project_attempted:
                            tool_message = _tool_error(tool_call, "Project Skill retrieval was already attempted.")
                        elif not _has_project_skill_request_shape(tool_call):
                            # Providers may emit an incomplete tool payload despite the JSON schema.
                            # Keep this stage open so the next turn can repair the control request.
                            tool_message = _tool_error(
                                tool_call,
                                "load_project_skill requires non-empty project_type, architecture_signature, "
                                "component_roles, and responsibility_boundary. Retry with all fields based on "
                                "repository observations.",
                            )
                        else:
                            tool_message, loaded = self._load_stage_skill(
                                tool_call, "project_skill", run_dir, step
                            )
                            project_attempted = True
                            loaded_skills["project_skill"] = loaded
                    elif name == STRATEGY_SKILL_TOOL:
                        if not project_attempted:
                            tool_message = _tool_error(tool_call, "Project Skill retrieval must be attempted first.")
                        elif strategy_attempted:
                            tool_message = _tool_error(tool_call, "Strategy Skill retrieval was already attempted.")
                        else:
                            tool_message, loaded = self._load_stage_skill(
                                tool_call, "strategy_skill", run_dir, step
                            )
                            strategy_attempted = True
                            loaded_skills["strategy_skill"] = loaded
                            strategy_evidence_observed = loaded is None
                    elif name == FINISH_TOOL_NAME:
                        if project_attempted and strategy_attempted and strategy_evidence_observed:
                            try:
                                final_result = self._parse_finish_tool_call(tool_call)
                            except Exception as exc:  # noqa: BLE001 - keep protocol failure inside the loop.
                                tool_message = _tool_error(tool_call, f"Invalid finish_localization arguments: {exc}")
                            else:
                                append_jsonl(
                                    run_dir / "trajectory.jsonl",
                                    {"event": "finish", "step": step, "result": final_result},
                                )
                                break
                        if not project_attempted or not strategy_attempted:
                            error = "Both Project Skill and Strategy Skill retrieval attempts are required before finish."
                        else:
                            error = (
                                "Use a repository tool after loading the Strategy Skill before finishing. "
                                "The observation must test its distinguishing condition."
                            )
                        tool_message = _tool_error(tool_call, error)
                    else:
                        tool_message = self._execute_repo_tool(registry, tool_call)
                        if strategy_attempted and loaded_skills["strategy_skill"] is not None:
                            strategy_evidence_observed = True
                            append_jsonl(
                                run_dir / "trajectory.jsonl",
                                {
                                    "event": "strategy_skill_evidence_observation",
                                    "step": step,
                                    "tool_name": name,
                                    "skill_id": loaded_skills["strategy_skill"].get("skill_id"),
                                },
                            )
                    messages.append(tool_message)
                    append_jsonl(
                        run_dir / "trajectory.jsonl",
                        {"event": "tool_result", "step": step, **tool_message},
                    )
                write_json(run_dir / "loaded_skills.json", loaded_skills)
                if final_result is not None:
                    break
                continue

            invalid_responses += 1
            if invalid_responses <= self.response_retries:
                required = PROJECT_SKILL_TOOL if not project_attempted else (
                    STRATEGY_SKILL_TOOL if not strategy_attempted else FINISH_TOOL_NAME
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Continue the localization through the available native tools. "
                            f"The next required control action is {required}."
                        ),
                    }
                )
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {"event": "response_repair", "step": step, "stage": stage, "required_tool": required},
                )
                continue

            project_attempted, strategy_attempted = self._complete_missing_stages(
                messages=messages,
                run_dir=run_dir,
                repo=repo,
                issue=issue,
                project_attempted=project_attempted,
                strategy_attempted=strategy_attempted,
                loaded_skills=loaded_skills,
                reason="Native tool response retries were exhausted.",
            )
            final_result = self._forced_finish(
                messages,
                run_dir,
                repo_path,
                llm_trace,
                reason="Native tool response retries were exhausted.",
            )
            break

        if final_result is None:
            project_attempted, strategy_attempted = self._complete_missing_stages(
                messages=messages,
                run_dir=run_dir,
                repo=repo,
                issue=issue,
                project_attempted=project_attempted,
                strategy_attempted=strategy_attempted,
                loaded_skills=loaded_skills,
                reason="Maximum steps reached.",
            )
            final_result = self._forced_finish(
                messages, run_dir, repo_path, llm_trace, reason="Maximum steps reached."
            )

        result = {
            "instance_id": task.get("instance_id"),
            "status": "completed",
            "ranked_functions": final_result["action"].get("ranked_functions", []),
            "final_summary": final_result["action"].get("summary", ""),
            "finish": final_result,
            "stage_completion": {
                "project_skill_attempted": project_attempted,
                "strategy_skill_attempted": strategy_attempted,
            },
        }
        write_json(run_dir / "loaded_skills.json", loaded_skills)
        write_json(run_dir / "llm_trace.json", llm_trace)
        write_json(run_dir / "result.json", result)
        return result

    def _load_stage_skill(
        self,
        tool_call: dict[str, Any],
        skill_type: str,
        run_dir: Path,
        step: int,
        *,
        repo_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        name = PROJECT_SKILL_TOOL if skill_type == "project_skill" else STRATEGY_SKILL_TOOL
        if skill_type not in self.skill_bank.enabled_skill_types:
            payload = self._record_disabled_skill_stage(
                skill_type=skill_type,
                run_dir=run_dir,
                step=step,
                reason=f"{skill_type} loading is disabled for this ablation run.",
            )
            return _tool_success(tool_call, payload, name=name, wrapped=False), None
        if skill_type == "strategy_skill":
            try:
                return self._load_strategy_skill(
                    tool_call,
                    _tool_arguments(tool_call),
                    run_dir,
                    step,
                )
            except Exception as exc:  # noqa: BLE001 - invalid selection becomes an empty attempt.
                payload = {
                    "skill_type": "strategy_skill",
                    "selected_skill_id": None,
                    "matched_skill": None,
                    "search_trace": {
                        "retrieval_mode": "llm_trigger_catalog_selection_v1",
                        "selected_skill_ids": [],
                        "notes": [f"Strategy Skill selection failed: {exc}"],
                    },
                }
                request_event = {
                    "event": "strategy_skill_request",
                    "step": step,
                    "selected_skill_id": None,
                    "request": {},
                    "strategy_catalog": self.skill_bank.strategy_catalog(),
                    "error": str(exc),
                }
                append_jsonl(run_dir / "trajectory.jsonl", request_event)
                write_json(run_dir / "strategy_skill_request.json", request_event)
                write_json(run_dir / "strategy_skill_search.json", payload)
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {
                        "event": "strategy_skill_loaded",
                        "step": step,
                        "loaded_skill_id": None,
                        **payload,
                    },
                )
                return _tool_success(
                    tool_call,
                    payload,
                    name=STRATEGY_SKILL_TOOL,
                    wrapped=False,
                ), None
        request_written = False
        try:
            arguments = _tool_arguments(tool_call)
            if skill_type == "project_skill" and repo_id:
                repository_summary = str(
                    arguments.get("repository_summary")
                    or arguments.get("architecture_signature")
                    or arguments.get("project_type")
                    or ""
                ).strip()
                request_event = {
                    "event": "project_skill_request",
                    "step": step,
                    "repo_id": repo_id,
                    "query": repo_id,
                    "request": arguments,
                    "repository_summary": repository_summary,
                }
                append_jsonl(run_dir / "trajectory.jsonl", request_event)
                write_json(run_dir / "project_skill_request.json", request_event)
                search = self.skill_bank.search_project_for_repo(repo_id)
                normalized_repo_id = str(
                    (search.get("skill_search_trace") or {}).get("repo_id") or repo_id
                )
                matched = (search.get("matched_skills") or [None])[0]
                payload = {
                    "skill_type": "project_skill",
                    "repo_id": normalized_repo_id,
                    "matched_skill": matched,
                    "search_trace": search.get("skill_search_trace", {}),
                }
                write_json(run_dir / "project_skill_search.json", payload)
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {
                        "event": "project_skill_loaded",
                        "step": step,
                        "repo_id": repo_id,
                        "loaded_skill_id": (matched or {}).get("skill_id"),
                        "matched_skill": matched,
                        "search_trace": payload["search_trace"],
                    },
                )
                return _tool_success(
                    tool_call, payload, name=PROJECT_SKILL_TOOL, wrapped=False
                ), matched
            project_type = str(arguments.get("project_type") or "").strip()
            if not project_type:
                raise ValueError("project_type must be non-empty")
            retrieval_query = _project_retrieval_query(arguments)
            request_event = {
                "event": f"{skill_type}_request",
                "step": step,
                "query": project_type,
                "retrieval_query": retrieval_query,
                "request": arguments,
            }
            append_jsonl(run_dir / "trajectory.jsonl", request_event)
            write_json(run_dir / f"{skill_type}_request.json", request_event)
            request_written = True
            if self.skill_bank.retrieval_mode in {"embedding", "hybrid"}:
                candidate_search = self.skill_bank.search_project_candidates(
                    retrieval_query,
                    limit=self.project_skill_candidate_limit,
                )
                selector = self._select_project_skill_candidate(
                    request=arguments,
                    candidates=candidate_search.get("candidate_skills") or [],
                    candidate_selected_skill_ids=(
                        candidate_search.get("skill_search_trace", {}).get(
                            "candidate_selected_skill_ids"
                        )
                        or []
                    ),
                    run_dir=run_dir,
                    step=step,
                )
                selected_skill_id = str(selector.get("selected_skill_id") or "").strip()
                matched = (
                    self.skill_bank.get_active_skill(
                        selected_skill_id,
                        skill_type="project_skill",
                    )
                    if selected_skill_id not in {"", "none", "null"}
                    else None
                )
                search_trace = candidate_search.get("skill_search_trace", {})
                search_trace["selector"] = selector
                search_trace["selected_skill_ids"] = (
                    [matched["skill_id"]] if matched else []
                )
            else:
                search = self.skill_bank.search_for_stage(skill_type, retrieval_query, limit=1)
                matched = (search.get("matched_skills") or [None])[0]
                search_trace = search.get("skill_search_trace", {})
            payload = {
                "skill_type": skill_type,
                "query": project_type,
                "retrieval_query": retrieval_query,
                "matched_skill": matched,
                "search_trace": search_trace,
            }
            write_json(run_dir / f"{skill_type}_search.json", payload)
            append_jsonl(
                run_dir / "trajectory.jsonl",
                {
                    "event": f"{skill_type}_loaded",
                    "step": step,
                    "query": project_type,
                    "retrieval_query": retrieval_query,
                    "loaded_skill_id": (matched or {}).get("skill_id"),
                    "matched_skill": matched,
                    "search_trace": payload["search_trace"],
                },
            )
        except Exception as exc:  # noqa: BLE001 - a failed search is a completed empty attempt.
            if not request_written:
                request_event = {
                    "event": f"{skill_type}_request",
                    "step": step,
                    "query": "",
                    "request": {},
                    "error": str(exc),
                }
                append_jsonl(run_dir / "trajectory.jsonl", request_event)
                write_json(run_dir / f"{skill_type}_request.json", request_event)
            payload = {
                "skill_type": skill_type,
                "query": "",
                "matched_skill": None,
                "search_trace": {"notes": [f"Skill retrieval failed: {exc}"]},
            }
            write_json(run_dir / f"{skill_type}_search.json", payload)
            append_jsonl(
                run_dir / "trajectory.jsonl",
                {"event": f"{skill_type}_loaded", "step": step, "loaded_skill_id": None, **payload},
            )
            return _tool_success(tool_call, payload, name=name, wrapped=False), None
        return _tool_success(tool_call, payload, name=name, wrapped=False), matched

    def _record_disabled_skill_stage(
        self,
        *,
        skill_type: str,
        run_dir: Path,
        step: int,
        reason: str,
    ) -> dict[str, Any]:
        """Persist a compatible empty stage for a one-type ablation run."""

        payload = {
            "skill_type": skill_type,
            "matched_skill": None,
            "search_trace": {
                "retrieval_mode": "disabled_skill_type_ablation_v1",
                "selected_skill_ids": [],
                "notes": [reason],
            },
            "disabled": True,
            "reason": reason,
        }
        request_event = {
            "event": f"{skill_type}_request",
            "step": step,
            "request": {},
            "disabled": True,
            "reason": reason,
        }
        if skill_type == "strategy_skill":
            request_event["selected_skill_id"] = None
            request_event["strategy_catalog"] = []
        else:
            request_event["query"] = ""
            request_event["retrieval_query"] = ""
        append_jsonl(run_dir / "trajectory.jsonl", request_event)
        write_json(run_dir / f"{skill_type}_request.json", request_event)
        write_json(run_dir / f"{skill_type}_search.json", payload)
        append_jsonl(
            run_dir / "trajectory.jsonl",
            {
                "event": f"{skill_type}_loaded",
                "step": step,
                "loaded_skill_id": None,
                **payload,
            },
        )
        return payload

    def _select_project_skill_candidate(
        self,
        *,
        request: dict[str, Any],
        candidates: list[dict[str, Any]],
        candidate_selected_skill_ids: list[str],
        run_dir: Path,
        step: int,
    ) -> dict[str, Any]:
        candidate_ids = [
            str(candidate.get("skill_id") or "")
            for candidate in candidates
            if str(candidate.get("skill_id") or "")
        ]
        if not candidate_ids:
            selection = {
                "selected_skill_id": "none",
                "reason": "Embedding recall returned no active Project Skill candidates.",
                "selection_mode": "empty_candidate_catalog",
            }
            write_json(run_dir / "project_skill_selector.json", selection)
            return selection

        compact_candidates = [
            {
                "skill_id": candidate["skill_id"],
                "value": candidate.get("value"),
                "title": candidate.get("title"),
                "trigger": candidate.get("trigger"),
            }
            for candidate in candidates
        ]
        selector_input = {
            "observed_system": {
                "project_type": request.get("project_type"),
                "architecture_signature": request.get("architecture_signature"),
                "component_roles": request.get("component_roles"),
                "responsibility_boundary": request.get("responsibility_boundary"),
            },
            "candidate_project_skills": compact_candidates,
        }
        response: dict[str, Any] | None = None
        try:
            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": self.project_skill_selector_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(selector_input, ensure_ascii=False, indent=2),
                    },
                ],
                tools=[_project_skill_selector_tool_schema(candidate_ids)],
                tool_choice=_named_tool_choice(PROJECT_SKILL_SELECTOR_TOOL),
                temperature=0,
                max_tokens=self.project_skill_selector_max_tokens,
            )
            selector_call = next(
                (
                    call
                    for call in (response.get("tool_calls") or [])
                    if ((call.get("function") or {}).get("name"))
                    == PROJECT_SKILL_SELECTOR_TOOL
                ),
                None,
            )
            parsed = (
                _tool_arguments(selector_call)
                if selector_call is not None
                else extract_json_object(response.get("content") or "")
            )
            selected_skill_id = str(parsed.get("selected_skill_id") or "").strip()
            if selected_skill_id not in {*candidate_ids, "none"}:
                raise ValueError(
                    "selected_skill_id must be one candidate ID or 'none'."
                )
            selection = {
                "selected_skill_id": selected_skill_id,
                "reason": str(parsed.get("reason") or "").strip(),
                "selection_mode": "llm_compact_candidate_selection_v1",
            }
            if selected_skill_id != "none":
                candidate = next(
                    item for item in compact_candidates if item["skill_id"] == selected_skill_id
                )
                validation = self._validate_project_skill_candidate(
                    observed_system=selector_input["observed_system"],
                    candidate=candidate,
                    run_dir=run_dir,
                    step=step,
                )
                selection["validation"] = validation
                if not validation["approved"]:
                    selection["selected_skill_id"] = "none"
                    selection["reason"] = validation["reason"]
                    selection["selection_mode"] = "validation_rejected_no_match_v1"
        except Exception as exc:  # noqa: BLE001 - an uncertain selector must not inject a recalled card.
            selection = {
                "selected_skill_id": "none",
                "reason": "Selector failed; no Project Skill was injected.",
                "selection_mode": "selector_failure_no_match_v1",
                "selector_error": str(exc),
            }
        debug = {
            "step": step,
            "input": selector_input,
            "response_content": (response or {}).get("content", ""),
            "response_tool_calls": (response or {}).get("tool_calls", []),
            **selection,
        }
        write_json(run_dir / "project_skill_selector.json", debug)
        append_jsonl(
            run_dir / "trajectory.jsonl",
            {
                "event": "project_skill_selected",
                "step": step,
                **selection,
                "candidate_skill_ids": candidate_ids,
            },
        )
        return selection

    def _validate_project_skill_candidate(
        self,
        *,
        observed_system: dict[str, Any],
        candidate: dict[str, Any],
        run_dir: Path,
        step: int,
    ) -> dict[str, Any]:
        validation_input = {
            "observed_system": observed_system,
            "selected_candidate": candidate,
        }
        response: dict[str, Any] | None = None
        validation_attempt_count = 0
        try:
            response, parsed, validation_attempt_count = self._call_validator_with_retry(
                prompt=self.project_skill_validator_prompt,
                validation_input=validation_input,
                tool_schema=_project_skill_validator_tool_schema(),
                tool_name=PROJECT_SKILL_VALIDATOR_TOOL,
                max_tokens=self.project_skill_validator_max_tokens,
            )
            approved = parsed.get("approved") is True
            reason = str(parsed.get("reason") or "").strip()
            role_evidence = parsed.get("role_evidence")
            if not isinstance(role_evidence, list) or not all(
                isinstance(item, str) and item.strip() for item in role_evidence
            ):
                approved = False
                reason = reason or "Validator returned no concrete role evidence."
            validation = {
                "approved": approved,
                "reason": reason or "Validator did not establish a structural match.",
                "role_evidence": role_evidence if isinstance(role_evidence, list) else [],
                "validation_mode": "llm_structural_validation_v1",
            }
        except Exception as exc:  # noqa: BLE001 - validation failures must not inject a card.
            response = getattr(exc, "response", response)
            validation_attempt_count = int(
                getattr(exc, "attempt_count", validation_attempt_count)
            )
            validation = {
                "approved": False,
                "reason": "Project Skill validation failed; no card was injected.",
                "role_evidence": [],
                "validation_mode": "validator_failure_no_match_v1",
                "validator_error": str(exc),
            }
        write_json(
            run_dir / "project_skill_validation.json",
            {
                "step": step,
                "input": validation_input,
                "response_content": (response or {}).get("content", ""),
                "response_tool_calls": (response or {}).get("tool_calls", []),
                "validation_attempt_count": validation_attempt_count,
                **validation,
            },
        )
        append_jsonl(
            run_dir / "trajectory.jsonl",
            {"event": "project_skill_validated", "step": step, **validation},
        )
        return validation

    def _load_strategy_skill(
        self,
        tool_call: dict[str, Any],
        arguments: dict[str, Any],
        run_dir: Path,
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        catalog = self.skill_bank.strategy_catalog()
        selected_skill_id = str(arguments.get("selected_skill_id") or "").strip()
        if not selected_skill_id and not catalog:
            selected_skill_id = "none"
            arguments = {**arguments, "selected_skill_id": "none"}
        if not selected_skill_id:
            raise ValueError("selected_skill_id must be a Strategy Skill ID or 'none'")
        strategy_query = "\n".join(
            str(arguments.get(key) or "").strip()
            for key in (
                "candidate_relationship",
                "unresolved_question",
                "observed_evidence",
            )
            if str(arguments.get(key) or "").strip()
        )
        search = self.skill_bank.select_strategy_skill(
            selected_skill_id,
            query=strategy_query,
        )
        matched = (search.get("matched_skills") or [None])[0]
        validation: dict[str, Any] | None = None
        if matched:
            validation = self._validate_strategy_skill_candidate(
                diagnostic_state={
                    "candidate_functions": arguments.get("candidate_functions") or [],
                    "candidate_relationship": arguments.get("candidate_relationship"),
                    "unresolved_question": arguments.get("unresolved_question"),
                    "observed_evidence": arguments.get("observed_evidence"),
                },
                candidate={
                    "skill_id": matched.get("skill_id"),
                    "value": matched.get("value"),
                    "title": matched.get("title"),
                    "trigger": matched.get("trigger"),
                },
                run_dir=run_dir,
                step=step,
            )
            if not validation["approved"]:
                matched = None
        request_event = {
            "event": "strategy_skill_request",
            "step": step,
            "selected_skill_id": selected_skill_id,
            "request": arguments,
            "strategy_catalog": catalog,
        }
        append_jsonl(run_dir / "trajectory.jsonl", request_event)
        write_json(run_dir / "strategy_skill_request.json", request_event)
        payload = {
            "skill_type": "strategy_skill",
            "selected_skill_id": selected_skill_id,
            "selection_reason": str(arguments.get("selection_reason") or "").strip(),
            "matched_skill": matched,
            "search_trace": search.get("skill_search_trace", {}),
            "validation": validation,
        }
        write_json(run_dir / "strategy_skill_search.json", payload)
        append_jsonl(
            run_dir / "trajectory.jsonl",
            {
                "event": "strategy_skill_loaded",
                "step": step,
                "selected_skill_id": selected_skill_id,
                "loaded_skill_id": (matched or {}).get("skill_id"),
                "matched_skill": matched,
                "search_trace": payload["search_trace"],
                "validation": validation,
            },
        )
        return _tool_success(
            tool_call,
            payload,
            name=STRATEGY_SKILL_TOOL,
            wrapped=False,
        ), matched

    def _validate_strategy_skill_candidate(
        self,
        *,
        diagnostic_state: dict[str, Any],
        candidate: dict[str, Any],
        run_dir: Path,
        step: int,
    ) -> dict[str, Any]:
        validation_input = {
            "diagnostic_state": diagnostic_state,
            "selected_candidate": candidate,
        }
        response: dict[str, Any] | None = None
        validation_attempt_count = 0
        try:
            response, parsed, validation_attempt_count = self._call_validator_with_retry(
                prompt=self.strategy_skill_validator_prompt,
                validation_input=validation_input,
                tool_schema=_strategy_skill_validator_tool_schema(),
                tool_name=STRATEGY_SKILL_VALIDATOR_TOOL,
                max_tokens=self.strategy_skill_validator_max_tokens,
            )
            approved = parsed.get("approved") is True
            reason = str(parsed.get("reason") or "").strip()
            condition_evidence = parsed.get("condition_evidence")
            if not isinstance(condition_evidence, list) or not all(
                isinstance(item, str) and item.strip() for item in condition_evidence
            ):
                approved = False
                reason = reason or "Validator returned no concrete trigger-condition evidence."
            validation = {
                "approved": approved,
                "reason": reason or "Validator did not establish a Strategy Skill match.",
                "condition_evidence": condition_evidence if isinstance(condition_evidence, list) else [],
                "validation_mode": "llm_diagnostic_condition_validation_v1",
            }
        except Exception as exc:  # noqa: BLE001 - validation failures must not inject a card.
            response = getattr(exc, "response", response)
            validation_attempt_count = int(
                getattr(exc, "attempt_count", validation_attempt_count)
            )
            validation = {
                "approved": False,
                "reason": "Strategy Skill validation failed; no card was injected.",
                "condition_evidence": [],
                "validation_mode": "validator_failure_no_match_v1",
                "validator_error": str(exc),
            }
        write_json(
            run_dir / "strategy_skill_validation.json",
            {
                "step": step,
                "input": validation_input,
                "response_content": (response or {}).get("content", ""),
                "response_tool_calls": (response or {}).get("tool_calls", []),
                "validation_attempt_count": validation_attempt_count,
                **validation,
            },
        )
        append_jsonl(
            run_dir / "trajectory.jsonl",
            {"event": "strategy_skill_validated", "step": step, **validation},
        )
        return validation

    def _call_validator_with_retry(
        self,
        *,
        prompt: str,
        validation_input: dict[str, Any],
        tool_schema: dict[str, Any],
        tool_name: str,
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        """Request a strict JSON applicability decision with one stable retry.

        Validators are internal protocol checks rather than Explorer-visible
        actions.  Some OpenAI-compatible providers intermittently return an
        empty assistant message for a forced *nested* tool call, even though
        the same model reliably honors JSON-object response formatting.  Keep
        native tools for Explorer actions and use the validators' documented
        JSON contract here, without weakening their evidence requirements.
        """

        del tool_schema, tool_name
        last_error: Exception | None = None
        response: dict[str, Any] | None = None
        for attempt in range(1, 3):
            try:
                response = self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(validation_input, ensure_ascii=False, indent=2)},
                    ],
                    tool_choice="none",
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max_tokens,
                )
                parsed = extract_json_object(response.get("content") or "")
                return response, parsed, attempt
            except Exception as exc:  # noqa: BLE001 - one stable retry for an empty provider payload.
                last_error = exc
        raise ValidatorProtocolError(
            str(last_error or "Validator did not return a structured decision."),
            attempt_count=attempt,
            response=response,
        )

    def _complete_missing_stages(
        self,
        *,
        messages: list[dict[str, Any]],
        run_dir: Path,
        repo: str,
        issue: str,
        project_attempted: bool,
        strategy_attempted: bool,
        loaded_skills: dict[str, Any],
        reason: str,
    ) -> tuple[bool, bool]:
        if not project_attempted:
            call = _synthetic_tool_call(
                PROJECT_SKILL_TOOL,
                {
                    "project_type": "unknown project or subsystem type",
                    "architecture_signature": "No evidence-based architecture signature was available before finalization.",
                    "component_roles": [],
                    "responsibility_boundary": "Unknown; infer only from observed repository evidence.",
                },
            )
            message, loaded = self._load_stage_skill(call, "project_skill", run_dir, -1)
            messages.extend([{"role": "assistant", "content": "", "tool_calls": [call]}, message])
            loaded_skills["project_skill"] = loaded
            project_attempted = True
        if not strategy_attempted:
            call = _synthetic_tool_call(
                STRATEGY_SKILL_TOOL,
                {
                    "selected_skill_id": "none",
                    "candidate_functions": [],
                    "candidate_relationship": "Candidate relationship was unresolved at finalization.",
                    "unresolved_question": reason,
                    "observed_evidence": "No stable diagnostic context was formed before finalization.",
                    "selection_reason": "No Strategy Skill was selected because finalization began before a reliable comparison was available.",
                },
            )
            message, loaded = self._load_stage_skill(call, "strategy_skill", run_dir, -1)
            messages.extend([{"role": "assistant", "content": "", "tool_calls": [call]}, message])
            loaded_skills["strategy_skill"] = loaded
            strategy_attempted = True
        write_json(run_dir / "loaded_skills.json", loaded_skills)
        return project_attempted, strategy_attempted

    def _required_tool_for_remaining_steps(
        self,
        *,
        step: int,
        project_attempted: bool,
        strategy_attempted: bool,
        strategy_evidence_observed: bool,
    ) -> str | None:
        if self.max_steps is None:
            return None
        remaining = self.max_steps - step + 1
        required_count = (
            int(not project_attempted)
            + int(not strategy_attempted)
            + int(not strategy_attempted or not strategy_evidence_observed)
            + 1
        )
        if remaining > max(self.finalization_steps, required_count):
            return None
        if not project_attempted:
            return PROJECT_SKILL_TOOL
        if not strategy_attempted:
            return STRATEGY_SKILL_TOOL
        if not strategy_evidence_observed:
            return STRATEGY_EVIDENCE_TOOL
        return FINISH_TOOL_NAME

    def _tools_for_stage(
        self,
        registry: ToolRegistry,
        stage: str,
        *,
        strategy_evidence_observed: bool,
    ) -> list[dict[str, Any]]:
        if stage == "initial_exploration":
            return [*registry.list_openai_tools(), _project_skill_tool_schema()]
        if stage == "project_guided_inspection":
            return [
                *registry.list_openai_tools(),
                _strategy_skill_tool_schema(self.skill_bank.strategy_catalog()),
            ]
        if not strategy_evidence_observed:
            return registry.list_openai_tools()
        return [*registry.list_openai_tools(), _finish_tool_schema()]

    @staticmethod
    def _execute_repo_tool(registry: ToolRegistry, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = str(((tool_call.get("function") or {}).get("name") or ""))
        try:
            result = registry.call_tool(name, _tool_arguments(tool_call))
            return _tool_success(tool_call, {"ok": True, "result": result}, name=name, wrapped=False)
        except Exception as exc:  # noqa: BLE001 - return failures to the model.
            return _tool_error(tool_call, f"Tool {name} failed: {exc}")

    def _forced_finish(
        self,
        messages: list[dict[str, Any]],
        run_dir: Path,
        repo_path: str,
        llm_trace: list[dict[str, Any]],
        *,
        reason: str,
        compact_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finish_messages = messages
        if compact_context is None:
            messages.append({"role": "user", "content": _forced_stage_prompt(FINISH_TOOL_NAME, reason)})
        else:
            # A fresh, bounded finalization context avoids replaying a long tool
            # transcript into reasoning providers that may spend their budget
            # thinking instead of issuing the native finish call.
            finish_context = {"reason": reason, **compact_context}
            write_json(run_dir / "forced_finish_context.json", finish_context)
            append_jsonl(
                run_dir / "trajectory.jsonl",
                {"event": "forced_finish_context", "context": finish_context},
            )
            finish_messages = [
                {
                    "role": "system",
                    "content": (
                        "Finalize fault localization now. Use only the provided "
                        "finish_localization tool. Do not explain your reasoning or "
                        "request more tools; rank only functions supported by the compact evidence."
                    ),
                },
                {"role": "user", "content": json.dumps(finish_context, ensure_ascii=False)},
            ]
        response = self.llm_client.chat(
            messages=finish_messages,
            tools=[_finish_tool_schema()],
            tool_choice=(
                _named_tool_choice(FINISH_TOOL_NAME)
                if getattr(self.llm_client, "supports_tool_choice", True)
                else None
            ),
            max_tokens=self.forced_finish_max_tokens,
            temperature=0,
        )
        llm_trace.append({"step": "forced_finish", "response": response})
        append_jsonl(run_dir / "trajectory.jsonl", {"event": "forced_finish", "response": response})
        for call in response.get("tool_calls") or []:
            if ((call.get("function") or {}).get("name")) == FINISH_TOOL_NAME:
                try:
                    return self._parse_finish_tool_call(call)
                except Exception:
                    break
        fallback = _deterministic_finish_from_trajectory(run_dir, Path(repo_path), reason=reason)
        append_jsonl(run_dir / "trajectory.jsonl", {"event": "deterministic_finish", "result": fallback})
        return fallback

    @staticmethod
    def _parse_finish_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
        arguments = _tool_arguments(tool_call)
        ranked = arguments.get("ranked_functions")
        if not isinstance(ranked, list):
            raise ValueError("finish_localization.ranked_functions must be a list")
        return {
            "thought": str(arguments.get("thought") or ""),
            "action": {
                "type": "finish",
                "ranked_functions": [str(item) for item in ranked],
                "summary": str(arguments.get("summary") or ""),
            },
        }

    def _runtime_expired(self, started_at: float) -> bool:
        return self.max_runtime_seconds is not None and (
            time.monotonic() - started_at
        ) >= self.max_runtime_seconds


def _stage_name(project_attempted: bool, strategy_attempted: bool) -> str:
    if not project_attempted:
        return "initial_exploration"
    if not strategy_attempted:
        return "project_guided_inspection"
    return "diagnostic_ranking"


def _tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = (tool_call.get("function") or {}).get("arguments") or "{}"
    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return arguments


def _has_project_skill_request_shape(tool_call: dict[str, Any]) -> bool:
    """Reject malformed native Project Skill calls without consuming the one attempt."""
    try:
        arguments = _tool_arguments(tool_call)
    except Exception:  # noqa: BLE001 - converted to a normal tool error in the loop.
        return False
    return bool(
        str(arguments.get("project_type") or "").strip()
        and str(arguments.get("architecture_signature") or "").strip()
        and str(arguments.get("responsibility_boundary") or "").strip()
        and isinstance(arguments.get("component_roles"), list)
        and any(str(role).strip() for role in arguments["component_roles"])
    )


def _tool_success(
    tool_call: dict[str, Any], payload: dict[str, Any], *, name: str, wrapped: bool = True
) -> dict[str, Any]:
    content = {"ok": True, "result": payload} if wrapped else payload
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id") or "call_unknown",
        "name": name,
        "content": json.dumps(content, ensure_ascii=False),
    }


def _tool_error(tool_call: dict[str, Any], error: str) -> dict[str, Any]:
    name = str(((tool_call.get("function") or {}).get("name") or ""))
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id") or "call_unknown",
        "name": name,
        "content": json.dumps({"ok": False, "error": error}, ensure_ascii=False),
    }


def _synthetic_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_forced_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _named_tool_choice(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


def _forced_stage_prompt(tool_name: str, reason: str = "") -> str:
    prefix = f"{reason} " if reason else ""
    return f"{prefix}Use the available evidence and call {tool_name} now."


def _project_skill_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=PROJECT_SKILL_TOOL,
        description=(
            "Complete repository orientation and load the Project Skill stored for this exact repository. "
            "The framework resolves the repository identity; provide only the static structure observed "
            "before the issue is revealed."
        ),
        properties={
            "repository_summary": {
                "type": "string",
                "description": "Concise evidence-based summary of the repository's observed structure.",
            },
            "component_roles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Observed components and their stable responsibilities.",
            },
            "responsibility_boundaries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Observed data, state, control, dispatch, or adaptation handoffs.",
            },
        },
        required=["repository_summary", "component_roles", "responsibility_boundaries"],
    )


def _project_retrieval_query(arguments: dict[str, Any]) -> str:
    project_type = str(arguments.get("project_type") or "").strip()
    architecture_signature = str(arguments.get("architecture_signature") or "").strip()
    responsibility_boundary = str(arguments.get("responsibility_boundary") or "").strip()
    raw_roles = arguments.get("component_roles")
    component_roles = [
        str(role).strip()
        for role in (raw_roles if isinstance(raw_roles, list) else [])
        if str(role).strip()
    ]
    parts = [
        f"project_type: {project_type}",
        f"architecture_signature: {architecture_signature}",
        f"component_roles: {', '.join(component_roles)}",
        f"responsibility_boundary: {responsibility_boundary}",
    ]
    return "\n".join(part for part in parts if part.split(":", 1)[1].strip())


def _project_skill_selector_tool_schema(candidate_ids: list[str]) -> dict[str, Any]:
    return openai_function_tool(
        name=PROJECT_SKILL_SELECTOR_TOOL,
        description=(
            "Select one Project Skill whose title and trigger match the observed architecture, "
            "or select 'none' when no candidate is structurally applicable."
        ),
        properties={
            "selected_skill_id": {
                "type": "string",
                "enum": [*candidate_ids, "none"],
                "description": "A candidate Project Skill ID, or 'none'.",
            },
            "reason": {
                "type": "string",
                "description": "Brief applicability decision based on component roles, flow, and boundary.",
            },
        },
        required=["selected_skill_id", "reason"],
    )


def _project_skill_validator_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=PROJECT_SKILL_VALIDATOR_TOOL,
        description=(
            "Approve a Project Skill only when every central trigger role and handoff "
            "is supported by the observed repository structure."
        ),
        properties={
            "approved": {
                "type": "boolean",
                "description": "True only when the selected trigger is structurally established.",
            },
            "role_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete observed evidence for the trigger's central roles and handoff.",
            },
            "reason": {
                "type": "string",
                "description": "Brief decision explaining approval or the missing structural condition.",
            },
        },
        required=["approved", "role_evidence", "reason"],
    )


def _strategy_skill_validator_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=STRATEGY_SKILL_VALIDATOR_TOOL,
        description=(
            "Approve a Strategy Skill only when the observed candidate relationship, unresolved "
            "question, and repository evidence establish every central condition in its trigger."
        ),
        properties={
            "approved": {
                "type": "boolean",
                "description": "True only when the selected strategy trigger is concretely established.",
            },
            "condition_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete evidence for the selected trigger's required diagnostic conditions.",
            },
            "reason": {
                "type": "string",
                "description": "Brief decision explaining approval or the missing diagnostic condition.",
            },
        },
        required=["approved", "condition_evidence", "reason"],
    )


def _strategy_skill_tool_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    catalog_text = "\n".join(
        f"- {item['skill_id']} | {item['title']} | {item['trigger']}"
        for item in catalog
    ) or "- none | No active Strategy Skills are available."
    allowed_ids = [str(item["skill_id"]) for item in catalog]
    return openai_function_tool(
        name=STRATEGY_SKILL_TOOL,
        description=(
            "Select and load one Strategy Skill after forming competing function candidates. "
            "Choose when its candidate-role relationship, disputed contract, and distinguishing repository "
            "observation all match. Concrete components may instantiate the same generic roles, while a shared "
            "role pair or action such as tracing is insufficient when the trigger defines a different contract. "
            "Choose 'none' otherwise. The selected Skill proposes a conditional repository check rather than "
            "a ranking conclusion. Its full knowledge "
            f"is returned only after this call.\nAvailable Strategy Skills:\n{catalog_text}"
        ),
        properties={
            "selected_skill_id": {
                "type": "string",
                "enum": [*allowed_ids, "none"],
                "description": "One applicable Strategy Skill ID from the catalog, or 'none'.",
            },
            "candidate_functions": {"type": "array", "items": {"type": "string"}, "description": "Current candidates as path::qualified_name."},
            "candidate_relationship": {"type": "string", "description": "Generic causal, data-flow, call, or responsibility relation instantiated by the concrete candidates."},
            "unresolved_question": {"type": "string", "description": "The specific diagnostic question whose answer would distinguish and rank the competing hypotheses."},
            "observed_evidence": {"type": "string", "description": "Concise repository evidence already observed; describe the state of the diagnosis rather than a solution plan."},
            "selection_reason": {"type": "string", "description": "Explain how the candidate relationship and unresolved question match the selected trigger, or why none applies."},
        },
        required=[
            "selected_skill_id",
            "candidate_functions",
            "candidate_relationship",
            "unresolved_question",
            "observed_evidence",
            "selection_reason",
        ],
    )


def _finish_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=FINISH_TOOL_NAME,
        description=(
            "Finish with up to five evidence-supported function-level candidates. Preserve plausible "
            "pre-Skill candidates when a loaded Skill was not confirmed by distinguishing repository evidence."
        ),
        properties={
            "ranked_functions": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 5,
                "description": "Up to five ranked path::qualified_name candidates supported by repository observations.",
            },
            "summary": {"type": "string", "description": "Brief code-evidence summary."},
            "thought": {"type": "string", "description": "Optional concise decision summary."},
        },
        required=["ranked_functions", "summary"],
    )


def _tool_snapshot(schema: dict[str, Any]) -> dict[str, Any]:
    function = schema["function"]
    return {"name": function["name"], "description": function["description"], "parameters": function["parameters"]}


def _parse_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "unbounded", "infinite"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _deterministic_finish_from_trajectory(run_dir: Path, repo_path: Path, *, reason: str) -> dict[str, Any]:
    evidence = _collect_evidence_from_trajectory(run_dir / "trajectory.jsonl")
    ranked = _rank_evidence_candidates(repo_path, evidence)
    return {
        "thought": "Deterministic fallback ranked functions from observed repository evidence.",
        "action": {
            "type": "finish",
            "ranked_functions": ranked,
            "summary": f"{reason} Ranked the best candidates from observed grep/read_file evidence.",
        },
    }


def _collect_evidence_from_trajectory(path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return evidence
    for order, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if event.get("event") != "tool_result" or event.get("name") not in {"grep", "read_file"}:
            continue
        try:
            content = json.loads(event.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        result = content.get("result") if content.get("ok") else None
        if not isinstance(result, dict):
            continue
        if event.get("name") == "grep":
            for match in result.get("matches") or []:
                _add_evidence(evidence, str(match.get("path") or ""), _safe_int(match.get("line"), 1), 2.0, order)
        else:
            path_value = str(result.get("path") or "")
            for item in result.get("content") or []:
                text = str(item.get("text") or "").lstrip()
                score = 2.5 if text.startswith(("def ", "class ", "async def ")) else 0.08
                _add_evidence(evidence, path_value, _safe_int(item.get("line"), 1), score, order)
    return evidence


def _add_evidence(evidence: dict[str, dict[str, Any]], path: str, line: int, score: float, order: int) -> None:
    if not path or path.startswith(".git/"):
        return
    record = evidence.setdefault(path, {"score": 0.0, "lines": collections.Counter(), "last_order": 0})
    record["score"] += score
    record["lines"][line] += score
    record["last_order"] = max(record["last_order"], order)


def _rank_evidence_candidates(repo_path: Path, evidence: dict[str, dict[str, Any]], limit: int = 5) -> list[str]:
    scored: list[tuple[float, str]] = []
    for path_value, record in evidence.items():
        if not path_value.endswith(".py"):
            continue
        score = float(record["score"]) * (0.35 if _is_test_path(path_value) else 1.0)
        line = int(record["lines"].most_common(1)[0][0]) if record["lines"] else 1
        symbol = _nearest_python_symbol(repo_path / path_value, line)
        if symbol:
            scored.append((score, f"{path_value}::{symbol}"))
    ranked: list[str] = []
    for _, candidate in sorted(scored, reverse=True):
        if candidate not in ranked:
            ranked.append(candidate)
        if len(ranked) >= limit:
            break
    return ranked


def _nearest_python_symbol(path: Path, line: int) -> str:
    if not path.exists():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return ""
    symbols: list[tuple[int, str]] = []
    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join([*parents, child.name])
                symbols.append((int(getattr(child, "lineno", 1)), name))
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)
    visit(tree, [])
    before = [item for item in symbols if item[0] <= line]
    return max(before, default=(0, path.stem), key=lambda item: item[0])[1]


def _is_test_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("test") or "/test" in lowered or lowered.startswith("tests/")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run_explorer(
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    llm_client: Any | None = None,
    skill_bank: SkillBankV0 | None = None,
) -> dict[str, Any]:
    explorer_cfg = config.get("explorer", {})
    prompt_path = resolve_path(explorer_cfg.get("system_prompt_path", "prompt_records/explorer/explorer_system_v0.txt"))
    bank = skill_bank or make_skill_bank(config)
    client = llm_client or OpenAICompatibleClient.from_config(config.get("llm", {}))
    if str(explorer_cfg.get("workflow_version") or "v2").lower() == "v3":
        from .v3_agent import V3ExplorerAgent

        return V3ExplorerAgent(
            llm_client=client,
            skill_bank=bank,
            config=config,
            system_prompt=prompt_path.read_text(encoding="utf-8"),
        ).run(task)
    return ExplorerAgent(
        llm_client=client,
        skill_bank=bank,
        config=config,
        system_prompt=prompt_path.read_text(encoding="utf-8"),
    ).run(task)
