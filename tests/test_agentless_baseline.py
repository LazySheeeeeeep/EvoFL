import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("agentless_baseline", Path(__file__).resolve().parents[1] / "scripts/run_agentless_heldout30.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_rank_does_not_expand_classes_or_use_truth():
    code = "class A:\n    def run(self): pass\nclass B:\n    def run(self): pass\ndef f(): pass\n"
    preds, ignored = baseline.rank_functions({"a.py": ["class: A\nfunction: A\nfunction: run\nfunction: f\nfunction: A.run\nfunction: f"]}, {"a.py": code})
    assert preds == ["a.py::f", "a.py::A.run"]
    assert len(ignored) == 3


def test_unknown_predictions_keep_their_rank():
    preds, _ = baseline.rank_functions({"a.py": ["function: missing\nfunction: f"]}, {"a.py": "def f(): pass"})
    assert preds == ["a.py::missing", "a.py::f"]


def test_nested_and_async_names():
    code = "class A:\n    async def f(self):\n        def inner(): pass\n"
    preds, _ = baseline.rank_functions({"a.py": ["function: inner\nfunction: A.f"]}, {"a.py": code})
    assert preds == ["a.py::A.f.inner", "a.py::A.f"]
