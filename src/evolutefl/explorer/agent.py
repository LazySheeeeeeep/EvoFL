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
from evolutefl.skills import SkillBankV0, assemble_skill_context, make_skill_bank, render_skill_context
from evolutefl.tools import ToolRegistry, register_builtin_tools
from evolutefl.tools.registry import openai_function_tool


OUTPUT_CONTRACT = {
    "format": {
        "finish_tool": "Call finish_localization with ranked_functions and summary.",
    }
}

FINISH_TOOL_NAME = "finish_localization"


class ExplorerAgent:
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
        self.max_runtime_seconds = float(runtime_limit) if runtime_limit not in (None, "", "none", "None") else None
        self.response_retries = int(config.get("explorer", {}).get("response_retries", 2))
        self.checkpoint_interval = int(config.get("explorer", {}).get("checkpoint_interval", 8))
        self.finalization_steps = int(config.get("explorer", {}).get("finalization_steps", 3))
        self.forced_finish_max_tokens = int(config.get("explorer", {}).get("forced_finish_max_tokens", 4096))
        assembler_cfg = config.get("assembler", {})
        self.max_per_dimension = assembler_cfg.get("max_per_dimension")
        self.max_knowledge_chars = int(assembler_cfg.get("max_knowledge_chars", 900))

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(task["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        repo_path = str(Path(task["repo_path"]).resolve())
        repo = str(task.get("repo") or "")
        issue = str(task.get("bug_report") or "")

        registry = ToolRegistry()
        register_builtin_tools(registry, run_dir=run_dir, repo_path=repo_path)
        write_json(run_dir / "tools.json", registry.list_tools())

        skill_search = self.skill_bank.search_for_explorer(repo, issue)
        matched_skills = skill_search["matched_skills"]
        assembled_context = assemble_skill_context(matched_skills, self.max_per_dimension, self.max_knowledge_chars)
        rendered_context = render_skill_context(assembled_context) if matched_skills else ""

        write_json(run_dir / "skill_search_trace.json", skill_search["skill_search_trace"])
        write_json(run_dir / "matched_skills.json", matched_skills)
        write_json(run_dir / "assembled_context.json", assembled_context)
        (run_dir / "rendered_context.txt").write_text(rendered_context, encoding="utf-8")

        user_payload = {
            "instance_id": task.get("instance_id"),
            "repo": repo,
            "base_commit": task.get("base_commit", ""),
            "repo_path": repo_path,
            "bug_report": issue,
            "matched_skills": matched_skills,
            "skill_search_trace": skill_search["skill_search_trace"],
            "assembled_context": assembled_context,
            "rendered_localization_context": rendered_context,
            "output_contract": OUTPUT_CONTRACT,
        }
        write_json(run_dir / "initial_payload.json", user_payload)
        append_jsonl(
            run_dir / "trajectory.jsonl",
            {
                "event": "skill_context",
                "skill_search_trace": skill_search["skill_search_trace"],
                "matched_skill_ids": [skill["skill_id"] for skill in matched_skills],
                "assembled_context": assembled_context,
                "rendered_context_path": str(run_dir / "rendered_context.txt"),
            },
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]
        llm_trace: list[dict[str, Any]] = []
        final_result: dict[str, Any] | None = None
        repair_attempts = 0
        finalization_entered = False
        started_at = time.monotonic()
        step = 0

        while self.max_steps is None or step < self.max_steps:
            step += 1
            if self._runtime_expired(started_at):
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {
                        "event": "runtime_timeout",
                        "step": step,
                        "max_runtime_seconds": self.max_runtime_seconds,
                    },
                )
                final_result = self._forced_finish(
                    messages,
                    run_dir,
                    repo_path,
                    llm_trace,
                    reason="Runtime timeout reached.",
                )
                break

            finalization_mode = (
                self.max_steps is not None
                and step > max(0, self.max_steps - self.finalization_steps)
            )
            if finalization_mode and not finalization_entered:
                finalization_entered = True
                messages.append(
                    {
                        "role": "user",
                        "content": self._finalization_prompt(),
                    }
                )
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "finalization_mode", "step": step})
            elif self.checkpoint_interval > 0 and step > 1 and (step - 1) % self.checkpoint_interval == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Decision checkpoint: summarize the strongest function-level candidates from the evidence so far. "
                            "If you have enough evidence, call finish_localization now. "
                            "If not, make at most one targeted tool call that can decide between candidates."
                        ),
                    }
                )
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "decision_checkpoint", "step": step})

            response = self.llm_client.chat(
                messages=messages,
                tools=[_finish_tool_schema()] if finalization_mode else self._openai_tools(registry),
                tool_choice="auto",
            )
            llm_trace.append({"step": step, "response": response})
            tool_calls = response.get("tool_calls") or []
            content = response.get("content") or ""
            if tool_calls:
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "assistant_tool_calls", "step": step, "tool_calls": tool_calls})
                finish_call = self._find_finish_tool_call(tool_calls)
                if finish_call:
                    final_result = self._parse_finish_tool_call(finish_call)
                    append_jsonl(run_dir / "trajectory.jsonl", {"event": "finish", "step": step, "result": final_result})
                    break
                if finalization_mode:
                    append_jsonl(
                        run_dir / "trajectory.jsonl",
                        {"event": "ignored_non_finish_tool_calls", "step": step, "tool_calls": tool_calls},
                    )
                    for tool_call in tool_calls:
                        tool_call_id = tool_call.get("id") or "call_unknown"
                        function = tool_call.get("function") or {}
                        name = function.get("name") or ""
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "content": json.dumps(
                                {
                                    "ok": False,
                                    "error": (
                                        "Finalization mode is active. Stop searching and call "
                                        "finish_localization with the best evidence-supported candidates."
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        }
                        messages.append(tool_message)
                        append_jsonl(run_dir / "trajectory.jsonl", {"event": "tool_result", "step": step, **tool_message})
                    final_result = self._forced_finish(
                        messages,
                        run_dir,
                        repo_path,
                        llm_trace,
                        reason="Finalization mode received non-finish tool calls.",
                    )
                    break
                for tool_call in tool_calls:
                    tool_message = self._execute_tool_call(registry, tool_call)
                    messages.append(tool_message)
                    append_jsonl(run_dir / "trajectory.jsonl", {"event": "tool_result", "step": step, **tool_message})
                continue

            try:
                final_result = self._parse_finish(content)
                append_jsonl(run_dir / "trajectory.jsonl", {"event": "finish", "step": step, "result": final_result})
                break
            except ValueError as exc:
                if repair_attempts < self.response_retries:
                    repair_attempts += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response did not contain a valid final JSON object. "
                                "Call finish_localization now with ranked_functions and summary."
                            ),
                        }
                    )
                    append_jsonl(
                        run_dir / "trajectory.jsonl",
                        {"event": "response_repair", "step": step, "error": str(exc), "attempt": repair_attempts},
                    )
                    continue
                append_jsonl(
                    run_dir / "trajectory.jsonl",
                    {
                        "event": "response_repair_exhausted",
                        "step": step,
                        "error": str(exc),
                        "attempts": repair_attempts,
                    },
                )
                final_result = self._forced_finish(
                    messages,
                    run_dir,
                    repo_path,
                    llm_trace,
                    reason=f"Final response parse failed after {repair_attempts} repair attempts.",
                )
                break

        if final_result is None:
            final_result = self._forced_finish(messages, run_dir, repo_path, llm_trace, reason="Maximum steps reached.")

        result = {
            "instance_id": task.get("instance_id"),
            "status": "completed",
            "ranked_functions": final_result["action"].get("ranked_functions", []),
            "final_summary": final_result["action"].get("summary", ""),
            "finish": final_result,
        }
        write_json(run_dir / "llm_trace.json", llm_trace)
        write_json(run_dir / "result.json", result)
        return result

    def _execute_tool_call(self, registry: ToolRegistry, tool_call: dict[str, Any]) -> dict[str, Any]:
        tool_call_id = tool_call.get("id") or "call_unknown"
        function = tool_call.get("function") or {}
        name = function.get("name") or ""
        raw_args = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            result = registry.call_tool(name, arguments)
            content = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 - tool failures are returned to the model.
            content = {
                "ok": False,
                "error": f"Tool {name} failed: {exc}. Please choose another action or correct the arguments.",
            }
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(content, ensure_ascii=False),
        }

    def _parse_finish(self, content: str) -> dict[str, Any]:
        payload = extract_json_object(content)
        action = payload.get("action") or {}
        if action.get("type") != "finish":
            raise ValueError("Final JSON action.type must be 'finish'.")
        ranked = action.get("ranked_functions")
        if not isinstance(ranked, list):
            raise ValueError("Final JSON action.ranked_functions must be a list.")
        action.setdefault("summary", "")
        payload.setdefault("thought", "")
        return payload

    def _forced_finish(
        self,
        messages: list[dict[str, Any]],
        run_dir: Path,
        repo_path: str,
        llm_trace: list[dict[str, Any]],
        *,
        reason: str = "Maximum steps reached.",
    ) -> dict[str, Any]:
        messages.append(
            {
                "role": "user",
                "content": self._finalization_prompt(stop_reason=reason),
            }
        )
        response = self.llm_client.chat(
            messages=messages,
            tools=[_finish_tool_schema()],
            tool_choice="auto",
            max_tokens=self.forced_finish_max_tokens,
            temperature=0,
        )
        llm_trace.append({"step": "forced_finish", "response": response})
        append_jsonl(run_dir / "trajectory.jsonl", {"event": "forced_finish", "response": response})
        tool_calls = response.get("tool_calls") or []
        finish_call = self._find_finish_tool_call(tool_calls)
        if finish_call:
            return self._parse_finish_tool_call(finish_call)
        try:
            return self._parse_finish(response.get("content") or "")
        except ValueError:
            fallback = _deterministic_finish_from_trajectory(run_dir, Path(repo_path), reason=reason)
            append_jsonl(run_dir / "trajectory.jsonl", {"event": "deterministic_finish", "result": fallback})
            if fallback["action"]["ranked_functions"]:
                return fallback
            return {
                "thought": "Forced finish failed to produce valid localization.",
                "action": {
                    "type": "finish",
                    "ranked_functions": [],
                    "summary": f"{reason} A valid final localization could not be produced.",
                },
            }

    @staticmethod
    def _finalization_prompt(*, stop_reason: str = "") -> str:
        prefix = f"{stop_reason} " if stop_reason else ""
        return (
            f"{prefix}Finalization mode: stop exploring and call finish_localization. "
            "Use the evidence already observed. The available action is finish_localization. "
            "If uncertain, still rank the best evidence-supported function candidates instead of continuing to search."
        )

    def _runtime_expired(self, started_at: float) -> bool:
        return self.max_runtime_seconds is not None and (time.monotonic() - started_at) >= self.max_runtime_seconds

    @staticmethod
    def _openai_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
        return [*registry.list_openai_tools(), _finish_tool_schema()]

    @staticmethod
    def _find_finish_tool_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") == FINISH_TOOL_NAME:
                return tool_call
        return None

    def _parse_finish_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        ranked = arguments.get("ranked_functions")
        if not isinstance(ranked, list):
            raise ValueError("finish_localization.ranked_functions must be a list.")
        return {
            "thought": arguments.get("thought", ""),
            "action": {
                "type": "finish",
                "ranked_functions": [str(item) for item in ranked],
                "summary": str(arguments.get("summary") or ""),
            },
        }


def _parse_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "unbounded", "infinite"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _deterministic_finish_from_trajectory(run_dir: Path, repo_path: Path, *, reason: str) -> dict[str, Any]:
    evidence = _collect_evidence_from_trajectory(run_dir / "trajectory.jsonl")
    ranked = _rank_evidence_candidates(repo_path, evidence)
    return {
        "thought": "Deterministic fallback ranked functions from observed grep/read_file evidence.",
        "action": {
            "type": "finish",
            "ranked_functions": ranked,
            "summary": (
                f"{reason} The model did not produce a valid finish response, so EvoluteFL "
                "ranked the best candidates from observed repository evidence."
            ),
        },
    }


def _collect_evidence_from_trajectory(path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return evidence
    for order, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "tool_result":
            continue
        name = event.get("name")
        if name not in {"grep", "read_file"}:
            continue
        try:
            content = json.loads(event.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        result = content.get("result") if content.get("ok") else None
        if not isinstance(result, dict):
            continue
        if name == "grep":
            for match in result.get("matches") or []:
                path_value = str(match.get("path") or "")
                line_no = _safe_int(match.get("line"), default=1)
                _add_evidence(evidence, path_value, line_no, score=2.0, order=order, source="grep")
        elif name == "read_file":
            path_value = str(result.get("path") or "")
            lines = result.get("content") or []
            if lines:
                for item in lines:
                    text = str(item.get("text") or "").lstrip()
                    score = 2.5 if text.startswith(("def ", "class ", "async def ")) else 0.08
                    _add_evidence(
                        evidence,
                        path_value,
                        _safe_int(item.get("line"), default=_safe_int(result.get("start_line"), default=1)),
                        score=score,
                        order=order,
                        source="read_file",
                    )
            else:
                _add_evidence(evidence, path_value, _safe_int(result.get("start_line"), default=1), score=1.0, order=order, source="read_file")
    return evidence


def _add_evidence(
    evidence: dict[str, dict[str, Any]],
    path_value: str,
    line_no: int,
    *,
    score: float,
    order: int,
    source: str,
) -> None:
    if not path_value or path_value.startswith(".git/"):
        return
    record = evidence.setdefault(
        path_value,
        {
            "score": 0.0,
            "lines": collections.Counter(),
            "line_scores": collections.Counter(),
            "last_order": 0,
            "sources": collections.Counter(),
        },
    )
    if source == "grep" and record["sources"]["grep"] >= 8:
        score = min(score, 0.25)
    record["score"] += score
    record["lines"][line_no] += 1
    record["line_scores"][line_no] += score
    record["last_order"] = max(record["last_order"], order)
    record["sources"][source] += 1


def _rank_evidence_candidates(repo_path: Path, evidence: dict[str, dict[str, Any]], *, limit: int = 5) -> list[str]:
    scored: list[tuple[float, str]] = []
    max_order = max((int(record.get("last_order", 0)) for record in evidence.values()), default=1)
    for path_value, record in evidence.items():
        if not path_value.endswith(".py"):
            continue
        score = float(record.get("score", 0.0))
        if _is_test_path(path_value):
            score *= 0.35
        score += 0.75 * (int(record.get("last_order", 0)) / max_order)
        line_no = _representative_line(
            record.get("line_scores") or record.get("lines") or collections.Counter()
        )
        qualname = _nearest_python_symbol(repo_path / path_value, line_no)
        if not qualname:
            continue
        scored.append((score, f"{path_value}::{qualname}"))
    ranked: list[str] = []
    for _, candidate in sorted(scored, key=lambda item: item[0], reverse=True):
        if candidate not in ranked:
            ranked.append(candidate)
        if len(ranked) >= limit:
            break
    return ranked


def _representative_line(lines: collections.Counter) -> int:
    if not lines:
        return 1
    return int(lines.most_common(1)[0][0])


def _nearest_python_symbol(path: Path, line_no: int) -> str:
    if not path.exists():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return ""
    symbols: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join([*parents, child.name])
                priority = 1 if isinstance(child, ast.ClassDef) else 2
                symbols.append((int(getattr(child, "lineno", 1)), priority, qualname))
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    before = [item for item in symbols if item[0] <= line_no]
    if before:
        before.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return before[0][2]
    if symbols:
        symbols.sort(key=lambda item: (abs(item[0] - line_no), -item[1]))
        return symbols[0][2]
    return path.stem


def _is_test_path(path_value: str) -> bool:
    lowered = path_value.lower()
    return lowered.startswith("test") or "/test" in lowered or lowered.startswith("tests/")


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finish_tool_schema() -> dict[str, Any]:
    return openai_function_tool(
        name=FINISH_TOOL_NAME,
        description=(
            "Finish fault localization. Call this when you have the best evidence-supported "
            "function-level ranked candidates, or when you must stop with the best available evidence."
        ),
        properties={
            "ranked_functions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ranked function candidates as path::qualified_name.",
            },
            "summary": {"type": "string", "description": "Brief evidence-based localization summary."},
            "thought": {"type": "string", "description": "Optional short reasoning summary."},
        },
        required=["ranked_functions", "summary"],
    )


def run_explorer(
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    llm_client: Any | None = None,
    skill_bank: SkillBankV0 | None = None,
) -> dict[str, Any]:
    explorer_cfg = config.get("explorer", {})
    prompt_path = resolve_path(explorer_cfg.get("system_prompt_path", "prompt_records/explorer/explorer_system_v0.txt"))
    system_prompt = prompt_path.read_text(encoding="utf-8")
    bank = skill_bank or make_skill_bank(config)
    client = llm_client or OpenAICompatibleClient.from_config(config.get("llm", {}))
    return ExplorerAgent(llm_client=client, skill_bank=bank, config=config, system_prompt=system_prompt).run(task)
