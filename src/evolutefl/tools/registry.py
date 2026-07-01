from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ToolCallable = Callable[..., Any]


@dataclass
class RegisteredTool:
    name: str
    func: ToolCallable
    openai_schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, name: str, func: ToolCallable, openai_schema: dict[str, Any]) -> None:
        if not name:
            raise ValueError("Tool name cannot be empty.")
        self._tools[name] = RegisteredTool(name=name, func=func, openai_schema=openai_schema)

    def list_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.openai_schema for tool in self._tools.values()]

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in self._tools.values():
            function = tool.openai_schema.get("function", {})
            tools.append(
                {
                    "name": function.get("name", tool.name),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        return tool.func(**(arguments or {}))


def openai_function_tool(
    *,
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }

