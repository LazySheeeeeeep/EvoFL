import importlib.util
import sys
from pathlib import Path

from evolutefl.evaluation.patch_ground_truth import function_identity_matches


def runner():
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('expansion99', scripts / 'run_experience_expansion99.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_identity_preserves_nested_scope():
    assert function_identity_matches('pkg/a.py::Outer.method.inner', 'pkg/a.py::Outer.method.inner', strict=True)
    assert not function_identity_matches('pkg/a.py::inner', 'pkg/a.py::Outer.method.inner', strict=True)
    assert not function_identity_matches('pkg/a.py::Outer.method', 'pkg/a.py::Outer.method.inner', strict=True)


def test_empty_resubstitution_report(tmp_path):
    module = runner()
    module.OUT = tmp_path
    module.evaluate_report([])
    report = module.read(tmp_path / 'comparison_summary.json')
    assert report['evaluation_kind'] == 'training-set resubstitution'
    assert report['completed_triplets'] == 0
    assert report['arms']['expanded_bank']['top1'] is None


def test_cleanup_rejects_outside_workspace(tmp_path):
    import pytest
    module = runner()
    module.OUT = tmp_path / 'experiment'
    protected = tmp_path / 'protected'
    protected.mkdir()
    with pytest.raises(ValueError):
        module.clean(protected)
    assert protected.exists()
