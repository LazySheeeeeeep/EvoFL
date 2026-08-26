from __future__ import annotations

from evolutefl.tools.builtin import MAX_GREP_MATCH_TEXT_CHARS, grep


def test_grep_supports_regex(tmpdir) -> None:
    repo = tmpdir.mkdir("repo")
    repo.join("sample.py").write("def invert_logic():\n    pass\n")
    literal = grep(str(repo), "invert|inverted", glob="*.py", max_results=10)
    assert literal["matches"] == []
    regex = grep(str(repo), "invert|inverted", glob="*.py", max_results=10, use_regex=True)
    assert len(regex["matches"]) == 1
    assert regex["use_regex"] is True


def test_grep_bounds_a_single_oversized_match_line(tmpdir) -> None:
    repo = tmpdir.mkdir("repo")
    repo.join("generated.py").write("needle " + "x" * 10_000 + " tail\n")

    result = grep(str(repo), "needle", glob="*.py", max_results=1)

    text = result["matches"][0]["text"]
    assert len(text) <= MAX_GREP_MATCH_TEXT_CHARS + 100
    assert "[truncated" in text
    assert text.endswith("tail")
