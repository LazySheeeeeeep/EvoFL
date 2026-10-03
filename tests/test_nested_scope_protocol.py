from evolutefl.explorer.agent import _finish_tool_schema
from evolutefl.tools import ToolRegistry
from evolutefl.tools.symbols import register_symbol_tools


def test_nested_function_has_own_identity_and_body(tmp_path):
    (tmp_path / "a.py").write_text(
        "class Worker:\n    def outer(self):\n        def inner():\n            return 1\n        return inner()\n")
    registry = register_symbol_tools(ToolRegistry(), tmp_path)
    result = registry.call_tool("find_symbol", {"name": "inner", "path": "a.py"})
    symbol = result["symbols"][0]
    assert symbol["symbol_id"] == "a.py::Worker.outer.inner"
    assert (symbol["start_line"], symbol["end_line"]) == (3, 4)
    body = registry.call_tool("read_symbol", {"symbol_id": symbol["symbol_id"]})
    assert body["start_line"] == 3 and body["end_line"] == 4
    schemas = {s["function"]["name"]: s["function"] for s in registry.list_openai_tools()}
    assert "nested" in schemas["find_symbol"]["description"]
    finish = _finish_tool_schema()["function"]["parameters"]["properties"]["ranked_functions"]
    assert "Class.outer.inner" in finish["description"]
