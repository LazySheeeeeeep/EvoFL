import json

import pytest

from evolutefl.investigation import InvestigationLog, delivered_observation
from evolutefl.tools import ToolRegistry
from evolutefl.tools.builtin import read_file
from evolutefl.tools.symbols import register_symbol_tools


def test_source_pagination_delivers_every_line_once(tmp_path):
    lines = [f"# line {i} " + "x" * 150 for i in range(350)]
    (tmp_path / "a.py").write_text("\n".join(lines))
    start, delivered = 1, []
    while start:
        page = read_file(str(tmp_path), "a.py", start_line=start)
        assert page["content"][0]["line"] == start
        assert page["end_line"] == page["content"][-1]["line"]
        delivered.extend(x["text"] for x in page["content"])
        start = page["next_start_line"]
    assert delivered == lines


def test_long_line_is_delivered_whole(tmp_path):
    line = "#" + "x" * 20000
    (tmp_path / "a.py").write_text(line + "\nx = 1\n")
    page = read_file(str(tmp_path), "a.py")
    assert page["content"] == [{"line": 1, "text": line}]
    assert page["next_start_line"] == 2


def test_explicit_range_reaches_target_and_pagination_continues(tmp_path):
    (tmp_path / "a.py").write_text("\n".join(f"# line {i}" for i in range(1, 501)))
    first = read_file(str(tmp_path), "a.py", end_line=430)
    assert (first["start_line"], first["end_line"], first["next_start_line"]) == (1, 200, 201)
    continuation = read_file(str(tmp_path), "a.py", start_line=first["next_start_line"], end_line=430)
    assert continuation["start_line"] == 201 and continuation["next_start_line"] == 401
    target = read_file(str(tmp_path), "a.py", start_line=380, end_line=430)
    assert target["content"][0] == {"line": 380, "text": "# line 380"}
    assert target["end_line"] == 430 and target["next_start_line"] is None


def test_symbol_navigation_and_paged_method(tmp_path):
    (tmp_path / "a.py").write_text("class A:\n    @staticmethod\n    async def run():\n        x = 1\n        return x\n\ndef run():\n    return 2\n")
    registry = register_symbol_tools(ToolRegistry(), tmp_path)
    found = registry.call_tool("find_symbol", {"name": "run", "limit": 1})
    assert found["total_matches"] == 2 and found["next_offset"] == 1
    symbol = found["symbols"][0]
    assert symbol["symbol_id"] == "a.py::A.run"
    page = registry.call_tool("read_symbol", {"symbol_id": symbol["symbol_id"], "max_lines": 2})
    assert page["start_line"] == 2 and page["next_start_line"] == 4
    assert page["content"][0]["text"].strip() == "@staticmethod"
    rest = registry.call_tool("read_symbol", {"symbol_id": symbol["symbol_id"], "start_line": 4})
    assert rest["end_line"] == 5 and rest["next_start_line"] is None
    assert registry.call_tool("find_symbol", {"name": "A.run"})["total_matches"] == 1


def test_symbols_reject_escape_and_report_parse_error(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "outside.py").write_text("def f(): pass")
    (repo / "bad.py").write_text("def f(:")
    registry = register_symbol_tools(ToolRegistry(), repo)
    with pytest.raises(ValueError):
        registry.call_tool("read_symbol", {"symbol_id": "../outside.py::f"})
    assert registry.call_tool("find_symbol", {"name": "f"})["parse_errors"]


def test_observation_replay_never_reveals_undelivered_content(tmp_path):
    old = {"obs-1": {"result": {"secret": "not delivered"},
                     "model_payload": {"observation_id": "obs-1", "result_preview": "visible", "truncated": True}}}
    assert "not delivered" not in json.dumps(delivered_observation(old, "obs-1"))
    log = InvestigationLog(tmp_path, model_observation_char_limit=500)
    _, record = log.begin(1, {"id": "c", "function": {"name": "read_symbol"}}, {})
    page = {"content": [{"line": 1, "text": "x" * 4000}]}
    message = log.observe(record, page)
    assert json.loads(message["content"])["result"] == page
    assert delivered_observation(log.observations, "obs-00001") == json.loads(message["content"])
