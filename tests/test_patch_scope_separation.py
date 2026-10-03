import difflib

import pytest

from evolutefl.evaluation import symbols_from_patch


def map_change(tmp_path, old, new, side="new"):
    path = tmp_path / "module.py"
    source = new if side == "new" else old
    path.write_text(source)
    patch = "diff --git a/module.py b/module.py\n" + "".join(
        difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                             fromfile="a/module.py", tofile="b/module.py"))
    result = symbols_from_patch(patch, tmp_path, source_side=side)
    assert path.read_text() == source
    return result


@pytest.mark.parametrize("side", ["old", "new"])
def test_removed_method_and_property_are_functions(tmp_path, side):
    old = "class A:\n    def removed(self):\n        return 1\n    @property\n    def value(self):\n        return 2\n    keep = 3\n"
    new = "class A:\n    keep = 3\n"
    result = map_change(tmp_path, old, new, side)
    assert result["functions"] == ["module.py::A.removed", "module.py::A.value"]
    assert result["classes"] == []


def test_class_base_change_is_not_function(tmp_path):
    result = map_change(tmp_path, "class A(Base):\n    pass\n", "class A:\n    pass\n")
    assert result["functions"] == []
    assert result["classes"] == ["module.py::A"]


def test_blank_boundary_not_enclosing_function(tmp_path):
    old = "def outer():\n    def inner():\n        return 1\n\n    return inner\n"
    new = "def outer():\n    def inner():\n        return 2\n    return inner\n"
    assert map_change(tmp_path, old, new)["functions"] == ["module.py::outer.inner"]


def test_decorator_change_maps_method(tmp_path):
    result = map_change(tmp_path, "@old\ndef f():\n    pass\n", "@new\ndef f():\n    pass\n")
    assert result["functions"] == ["module.py::f"]


def test_mismatched_source_rejected(tmp_path):
    (tmp_path / "module.py").write_text("def unrelated():\n    pass\n")
    patch = "diff --git a/module.py b/module.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
    with pytest.raises(ValueError, match="does not match"):
        symbols_from_patch(patch, tmp_path, source_side="new")


def test_exact_relocated_hunk_uses_actual_coordinates(tmp_path):
    (tmp_path / "module.py").write_text("# extra\n# header\ndef f():\n    return 2\n")
    patch = "diff --git a/module.py b/module.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    result = symbols_from_patch(patch, tmp_path, source_side="new")
    assert result["functions"] == ["module.py::f"]
    assert all(e["line"] == 4 for e in result["evidence"])


@pytest.mark.parametrize("side", ["old", "new"])
@pytest.mark.parametrize("deleted", [False, True])
def test_added_and_deleted_files(tmp_path, side, deleted):
    present = (side == "old") == deleted
    if present:
        (tmp_path / "module.py").write_text("def f():\n    pass\n")
    prefix = "-" if deleted else "+"
    header = "@@ -1,2 +0,0 @@" if deleted else "@@ -0,0 +1,2 @@"
    patch = ("diff --git a/module.py b/module.py\n"
             + ("deleted" if deleted else "new") + " file mode 100644\n"
             + header + "\n" + prefix + "def f():\n" + prefix + "    pass\n")
    assert symbols_from_patch(patch, tmp_path, source_side=side)["functions"] == ["module.py::f"]
