"""Fault-only Explorer with observable investigation decisions."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evolutefl.config import resolve_path
from evolutefl.investigation import InvestigationLog, instrument_tools, repository_identity, delivered_observation
from evolutefl.tools.symbols import register_symbol_tools
from evolutefl.tools.registry import openai_function_tool
from evolutefl.json_utils import append_jsonl, extract_json_object, write_json
from evolutefl.skills.fault_taxonomy import compact_fault_taxonomy, validate_fault_family
from evolutefl.tools import ToolRegistry, register_builtin_tools
from .agent import _finish_tool_schema
from .v3_agent import V3ExplorerAgent, _fault_skill_tool_schema


class TruncatedResponseError(RuntimeError):
    """The bounded output-length recovery did not produce a complete response."""


class ExplorerDeadlineExceeded(RuntimeError):
    pass


def _response_choice(response: dict) -> dict:
    return ((response.get("raw") or {}).get("choices") or [{}])[0]


def fault_skill_model_view(search: dict) -> dict:
    """Keep selection diagnostics on disk, never in the investigation context."""
    card = search.get("matched_skill")
    matched = None if not card else {
        key: card[key] for key in ("skill_id", "skill_type", "fault_family",
                                  "fault_subtype", "title", "trigger", "knowledge")
        if key in card
    }
    return {"skill_type": "fault_skill", "fault_family": search.get("fault_family"),
            "status": "loaded" if matched else "not_loaded", "matched_skill": matched}


class V5ExplorerAgent:
    # Reuse the fixed-schema selector protocol, not the legacy stage machine.
    _select_fault_skill = V3ExplorerAgent._select_fault_skill
    investigation_log_type = InvestigationLog

    def __init__(self, *, llm_client: Any, skill_bank: Any, config: dict, system_prompt: str) -> None:
        self.llm_client, self.skill_bank, self.config, self.system_prompt = llm_client, skill_bank, config, system_prompt
        self.thinking_disabled_after_recovery = False
        cfg = config.get("explorer", {})
        self.fault_selector_prompt = resolve_path(cfg.get("fault_skill_selector_prompt_path", "prompt_records/explorer/fault_skill_selector_v1.txt")).read_text(encoding="utf-8")
        self.fault_validator_prompt = resolve_path(cfg.get("fault_skill_validator_prompt_path", "prompt_records/explorer/fault_skill_validator_v1.txt")).read_text(encoding="utf-8")
        self.fault_skill_selector_max_tokens = int(cfg.get("fault_skill_selector_max_tokens", 4096))
        self.fault_skill_selector_retry_max_tokens = int(cfg.get("fault_skill_selector_retry_max_tokens", 8192))
        self.fault_skill_selector_attempts = int(cfg.get("fault_skill_selector_attempts", 3))

    def _official_deepseek_v4(self) -> bool:
        return (
            urlsplit(str(getattr(self.llm_client, "base_url", ""))).hostname == "api.deepseek.com"
            and str(getattr(self.llm_client, "model", "")).startswith("deepseek-v4-")
        )

    def _assistant_message(self, response: dict) -> dict:
        message = {"role": "assistant", "content": response.get("content")}
        if response.get("tool_calls"):
            message["tool_calls"] = response["tool_calls"]
        if self._official_deepseek_v4():
            # DeepSeek thinking + tools requires successful assistant reasoning
            # to round-trip. It remains protocol history, not source evidence.
            original = _response_choice(response).get("message") or {}
            if "reasoning_content" in original:
                message["reasoning_content"] = original["reasoning_content"]
        return message

    def _chat_step(self, request: dict, *, step: int, traces: list, directory: Path,
                   deadline: float) -> dict:
        cfg = self.config.get("explorer", {})
        retries = max(0, int(cfg.get("truncation_retries", 1)))
        base_extra = dict(getattr(self.llm_client, "extra_body", {}) or {})
        base_extra.update(request.get("extra_body") or {})
        base_budget = int(base_extra.get("max_tokens", request.get("max_tokens",
            getattr(self.llm_client, "max_tokens", 4096))))
        for attempt in range(retries + 1):
            if time.monotonic() >= deadline:
                raise ExplorerDeadlineExceeded("runtime_timeout")
            kwargs = dict(request)
            budget = base_budget
            disabled = self.thinking_disabled_after_recovery
            if disabled:
                kwargs["extra_body"] = {**base_extra, "thinking": {"type": "disabled"}}
            if attempt:
                budget = max(base_budget, int(cfg.get("truncation_retry_max_tokens", 8192)))
                extra = dict(base_extra)
                extra["max_tokens"] = budget
                if self._official_deepseek_v4() and cfg.get("deepseek_disable_thinking_on_truncation", True):
                    extra["thinking"] = {"type": "disabled"}
                    disabled = True
                    # A recovered tool turn has no reasoning_content. Keep this
                    # conversation non-thinking so later requests accept it.
                    self.thinking_disabled_after_recovery = True
                kwargs.update(max_tokens=budget, extra_body=extra)
            response = self.llm_client.chat(**kwargs)
            reason = _response_choice(response).get("finish_reason")
            traces.append({"step": step, "attempt": attempt + 1,
                "request_tools": [s["function"]["name"] for s in request["tools"]],
                "request_max_tokens": budget, "recovery_thinking_disabled": disabled,
                "finish_reason": reason, "response": response})
            write_json(directory / "llm_trace.json", traces)
            if reason != "length":
                return response
            append_jsonl(directory / "trajectory.jsonl", {"event": "response_truncated",
                "step": step, "attempt": attempt + 1, "max_tokens": budget,
                "usage": (response.get("raw") or {}).get("usage", {}),
                "will_retry": attempt < retries})
            # Never execute even apparently valid calls from a truncated turn.
            # Replay the same messages and tool schemas; no source is compressed.
        raise TruncatedResponseError("response_truncated")

    def load_fault(self, args: dict, issue: str, log: InvestigationLog) -> dict:
        family = validate_fault_family(args.get("fault_family"))
        for field in ("fault_signature", "project_context", "suspected_path"):
            if not isinstance(args.get(field), str) or not args[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        request = {"issue_report": issue, "request": args,
                   "orientation": {"observed_context": args["project_context"],
                                   "prior_investigation": log.timeline[-4:]}}
        if "fault_skill" not in self.skill_bank.enabled_skill_types:
            candidates, trace = [], {"disabled": True}
        else:
            catalog = self.skill_bank.fault_catalog(family)
            candidates, trace = catalog.get("candidate_skills", []), catalog.get("skill_search_trace", {})
        selected, attempts = {"selected_skill_id": None, "reason": "Empty or disabled family catalog."}, []
        if candidates:
            selected, attempts = self._select_fault_skill({**request, "candidates": candidates},
                candidate_ids={c["skill_id"] for c in candidates})
        matched = self.skill_bank.get_active_skill(selected["selected_skill_id"], skill_type="fault_skill") if selected.get("selected_skill_id") else None
        validation = {"applicable": False, "reason": "No candidate selected."}
        if matched:
            response = self.llm_client.chat(messages=[{"role": "system", "content": self.fault_validator_prompt},
                {"role": "user", "content": json.dumps({**request, "candidate": matched}, ensure_ascii=False)}],
                response_format={"type": "json_object"}, tool_choice="none", temperature=0)
            validation = extract_json_object(response.get("content") or "{}")
            if validation.get("applicable") is not True:
                matched = None
        trace.update({"selector": selected, "selector_attempts": attempts, "validator": validation,
                      "loaded_skill_primary_family": (matched or {}).get("fault_family"),
                      "loaded_via_alias": bool(matched and matched.get("fault_family") != family),
                      "catalog_count": len(candidates), "catalog_char_count": len(json.dumps(candidates)),
                      "selector_input_char_count": len(json.dumps({**request, "candidates": candidates}))})
        return {"skill_type": "fault_skill", "fault_family": family, "matched_skill": matched, "search_trace": trace}

    def run(self, task: dict) -> dict:
        issue = str(task.get("bug_report") or "").strip()
        if not issue:
            raise ValueError("V5 requires non-empty bug_report/problem_statement")
        root, directory = Path(task["repo_path"]).resolve(strict=True), Path(task["run_dir"]).resolve()
        if directory.is_relative_to(root):
            raise ValueError("run_dir must be outside the inspected repository")
        if (directory / "trajectory.jsonl").exists():
            raise ValueError("Refusing to overwrite an existing Explorer trajectory")
        directory.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("explorer", {})
        maximum = cfg.get("max_steps", 30)
        maximum = 30 if maximum is None else max(2, int(maximum))
        grace = max(1, int(cfg.get("finalization_steps", 3)))
        timeout = float(cfg.get("max_runtime_seconds", 900))
        started = time.monotonic()
        registry = register_builtin_tools(ToolRegistry(), run_dir=directory / "artifacts", repo_path=str(root))
        register_symbol_tools(registry, root)
        registry.register("read_observation", lambda observation_id: delivered_observation(log.observations, observation_id),
            openai_function_tool(name="read_observation", description="Replay a previously delivered observation exactly by ID. For additional source lines use the source page's next_start_line.",
                properties={"observation_id": {"type": "string"}}, required=["observation_id"]))
        repo_schemas = instrument_tools(registry.list_openai_tools())
        fault_schema, finish_schema = instrument_tools([_fault_skill_tool_schema(), _finish_tool_schema()])
        log = self.investigation_log_type(directory, model_observation_char_limit=int(cfg.get("model_observation_char_limit", 2400)))
        identity = repository_identity(root)
        write_json(directory / "run_context.json", {"trace_version": "v5", "repo_path": str(root),
            "repo": task.get("repo"), "source_state": task.get("source_state"),
            "repository_identity": identity})
        payload = {"repo": task.get("repo"), "base_commit": task.get("base_commit", ""),
                   "bug_report": issue, "fault_families": compact_fault_taxonomy()}
        write_json(directory / "initial_payload.json", payload)
        write_json(directory / "tools.json", [*repo_schemas, fault_schema, finish_schema])
        write_json(directory / "loaded_skills.json", {"fault_skill": None})
        append_jsonl(directory / "trajectory.jsonl", {"event": "run_started", "trace_version": "v5"})
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": json.dumps(payload)}]
        traces, attempted, invalid, finished = [], False, 0, False
        forced_fault_attempt, forced_finish = False, False
        result = {"trace_version": "v5", "instance_id": task.get("instance_id"), "status": "agent_interrupted",
                  "ranked_functions": [], "final_summary": "", "forced_finish": False}
        step = 0
        fault_attempt_step = min(maximum - 1, max(1, int(cfg.get("fault_skill_attempt_step", 4))))
        try:
            while step < maximum + grace and not finished:
                if time.monotonic() - started >= timeout:
                    result["error"] = "runtime_timeout"
                    break
                step += 1
                if not attempted and step >= fault_attempt_step:
                    required = "load_fault_skill"
                elif attempted and step >= maximum - 1:
                    required = "finish_localization"
                else:
                    required = None
                schemas = [*repo_schemas, finish_schema if attempted else fault_schema]
                if required:
                    schemas = [fault_schema if required == "load_fault_skill" else finish_schema]
                    if required == "load_fault_skill":
                        forced_fault_attempt = True
                    else:
                        forced_finish = True
                    messages.append({"role": "user", "content": (
                        f"The exploration budget is closing. The next protocol action is {required}. "
                        "Call this provided native tool now using the evidence already collected."
                    )})
                choice = {"type": "function", "function": {"name": required}} if required else "auto"
                request_kwargs = {"messages": messages, "tools": schemas, "tool_choice": choice}
                if required:
                    # Reasoning providers may consume the ordinary response
                    # budget before emitting the required native control call.
                    request_kwargs["max_tokens"] = int(cfg.get("finalization_max_tokens", 8192))
                response = self._chat_step(request_kwargs, step=step, traces=traces,
                    directory=directory, deadline=started + timeout)
                calls = response.get("tool_calls") or []
                if not calls:
                    invalid += 1
                    if self._official_deepseek_v4():
                        messages.append(self._assistant_message(response))
                    messages.append({"role": "user", "content": "Continue using a provided native tool; finish through finish_localization when available."})
                    if invalid > int(cfg.get("response_retries", 2)):
                        result["error"] = "native_tool_protocol_failed"
                        break
                    continue
                invalid = 0
                assistant = self._assistant_message(response)
                messages.append(assistant)
                append_jsonl(directory / "trajectory.jsonl", {"event": "assistant", "step": step, **assistant})
                for position, call in enumerate(calls):
                    if position:
                        # Keep the live context bounded. The complete assistant
                        # request remains in trajectory.jsonl, while the next
                        # turn receives one concrete observation to reason from.
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({
                            "error": "Use one repository action per assistant response so its result can be inspected before the next action."})})
                        continue
                    name = call.get("function", {}).get("name", "")
                    record = None
                    try:
                        args = json.loads(call["function"].get("arguments") or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("Tool arguments must be an object")
                        args, record = log.begin(step, call, args)
                        allowed = {s["function"]["name"] for s in schemas}
                        if finished or name not in allowed or (name == "load_fault_skill" and attempted):
                            raise ValueError(f"Tool unavailable in the current state. Available tools: {', '.join(sorted(allowed))}")
                        if name == "load_fault_skill":
                            validate_fault_family(args.get("fault_family"))
                            for key in ("fault_signature", "project_context", "suspected_path"):
                                if not isinstance(args.get(key), str) or not args[key].strip():
                                    raise ValueError(f"Missing {key}")
                            attempted = True
                            request = {"event": "fault_skill_request", "step": step, "request": args}
                            append_jsonl(directory / "trajectory.jsonl", request)
                            write_json(directory / "fault_skill_request.json", request)
                            try:
                                value = self.load_fault(args, issue, log)
                            except Exception as exc:
                                value = {"fault_family": args["fault_family"], "matched_skill": None,
                                         "search_trace": {"error": str(exc), "retrieval_failed": True}}
                            write_json(directory / "fault_skill_search.json", value)
                            write_json(directory / "loaded_skills.json", {"fault_skill": value.get("matched_skill")})
                            append_jsonl(directory / "trajectory.jsonl", {"event": "fault_skill_loaded", "step": step,
                                "loaded_skill_id": (value.get("matched_skill") or {}).get("skill_id"), **value})
                            value = fault_skill_model_view(value)
                        elif name == "finish_localization":
                            ranked = args.get("ranked_functions")
                            if not isinstance(ranked, list) or len(ranked) > 5 or not all(isinstance(f, str) and "::" in f for f in ranked):
                                raise ValueError("ranked_functions must contain up to five path::qualified_name strings")
                            if not isinstance(args.get("summary"), str):
                                raise ValueError("summary must be a string")
                            result.update(status="completed", ranked_functions=ranked, final_summary=args["summary"],
                                          forced_finish=forced_finish)
                            finished, value = True, {"finished": True}
                            append_jsonl(directory / "trajectory.jsonl", {"event": "finish", "step": step, **args})
                        else:
                            value = registry.call_tool(name, args)
                        messages.append(log.observe(record, value))
                    except Exception as exc:
                        if record is None:
                            _, record = log.begin(step, call, {})
                        messages.append(log.observe(record, None, error=str(exc)))
                        if required:
                            messages.append({"role": "user", "content": (
                                f"The previous call is unavailable. Do not call historical tools such as grep, read_file, or write. "
                                f"Invoke {required} exactly once using the sole currently provided native tool."
                            )})
        except (TruncatedResponseError, ExplorerDeadlineExceeded) as exc:
            result.update(error=str(exc))
        except Exception as exc:
            result.update(status="llm_request_failed", error=str(exc))
        result.update(steps=step, runtime_seconds=round(time.monotonic() - started, 3),
                      llm_request_count=len(traces),
                      truncated_response_count=sum(t.get("finish_reason") == "length" for t in traces),
                      truncation_recovery_count=sum(t.get("attempt", 1) > 1 and t.get("finish_reason") != "length" for t in traces),
                      fault_skill_attempted=attempted, fault_skill_attempt_step=fault_attempt_step,
                      forced_fault_skill_attempt=forced_fault_attempt)
        write_json(directory / "result.json", result)
        write_json(directory / "metadata.json", result)
        write_json(directory / "llm_trace.json", traces)
        log.save()
        return result
