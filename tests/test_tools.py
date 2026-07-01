from __future__ import annotations

from evolutefl.tools.builtin import grep


def test_grep_supports_regex(tmpdir) -> None:
    repo = tmpdir.mkdir("repo")
    repo.join("sample.py").write("def invert_logic():\n    pass\n")
    literal = grep(str(repo), "invert|inverted", glob="*.py", max_results=10)
    assert literal["matches"] == []
    regex = grep(str(repo), "invert|inverted", glob="*.py", max_results=10, use_regex=True)
    assert len(regex["matches"]) == 1
    assert regex["use_regex"] is True
