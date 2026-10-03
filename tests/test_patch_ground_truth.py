from pathlib import Path
from tempfile import TemporaryDirectory

from evolutefl.evaluation import evaluate_ranked_functions, functions_from_patch


def test_functions_from_patch_maps_deleted_and_added_lines_to_function() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "pkg" / "module.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "class Worker:\n"
            "    def transform(self, value):\n"
            "        current = value\n"
            "        return current\n",
            encoding="utf-8",
        )
        patch = (
            "diff --git a/pkg/module.py b/pkg/module.py\n"
            "--- a/pkg/module.py\n"
            "+++ b/pkg/module.py\n"
            "@@ -2,3 +2,3 @@ class Worker:\n"
            "     def transform(self, value):\n"
            "-        current = value\n"
            "+        current = normalize(value)\n"
            "         return current\n"
        )

        assert functions_from_patch(patch, root) == ["pkg/module.py::Worker.transform"]


def test_evaluate_ranked_functions_accepts_qualified_suffix() -> None:
    metrics = evaluate_ranked_functions(
        ["pkg/module.py::transform"],
        ["pkg/module.py::Worker.transform"],
    )

    assert metrics["rank"] == 1
    assert metrics["top1"] is True


def test_evaluate_ranked_functions_normalizes_inner_class_method_separator() -> None:
    metrics = evaluate_ranked_functions(
        ["pkg/module.py::Worker::transform"],
        ["pkg/module.py::Worker.transform"],
    )

    assert metrics["rank"] == 1
    assert metrics["top1"] is True


def test_functions_from_patch_does_not_cross_into_next_function_for_deleted_tail() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = root / "pkg" / "module.py"
        path.parent.mkdir(parents=True)
        path.write_text(
            "def target():\n"
            "    return value\n"
            "def later():\n"
            "    return 1\n",
            encoding="utf-8",
        )
        patch = (
            "diff --git a/pkg/module.py b/pkg/module.py\n"
            "--- a/pkg/module.py\n"
            "+++ b/pkg/module.py\n"
            "@@ -1,5 +1,4 @@\n"
            " def target():\n"
            "-    setup()\n"
            "     return value\n"
            "-\n"
            " def later():\n"
        )

        assert functions_from_patch(patch, root, source_side="new") == ["pkg/module.py::target"]


def test_evaluate_ranked_functions_accepts_dotted_python_module_identity() -> None:
    metrics = evaluate_ranked_functions(
        ["sphinx.util.inspect.signature_from_str"],
        ["sphinx/util/inspect.py::signature_from_str"],
    )

    assert metrics["rank"] == 1
    assert metrics["top1"] is True


def test_evaluate_ranked_functions_accepts_dotted_module_with_separator() -> None:
    metrics = evaluate_ranked_functions(
        ["grafanalib.core::_deep_update"],
        ["grafanalib/core.py::_deep_update"],
    )

    assert metrics["rank"] == 1
    assert metrics["top1"] is True


def test_evaluate_ranked_functions_accepts_path_module_without_py_suffix() -> None:
    metrics = evaluate_ranked_functions(
        ["grafanalib/core._deep_update"],
        ["grafanalib/core.py::_deep_update"],
    )

    assert metrics["rank"] == 1
    assert metrics["top1"] is True
