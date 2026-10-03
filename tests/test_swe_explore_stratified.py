import io
import tarfile
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path('scripts').resolve()))
from run_swe_explore_v5_stratified import balanced_sample, strict_metrics, strict_regions, clean_workspace
from run_swe_explore_v5_stratified import safe_archive_filter


def test_safe_relative_document_link(tmp_path):
    with tarfile.open(tmp_path / 'test.tar', 'w') as tar:
        regular = tarfile.TarInfo('repo/docs/theme/main/static/image.png')
        regular.size = 3
        tar.addfile(regular, io.BytesIO(b'png'))
        link = tarfile.TarInfo('repo/docs/theme/epub/static/image.png')
        link.type = tarfile.SYMTYPE
        link.linkname = '../../main/static/image.png'
        tar.addfile(link)
    dest = tmp_path / 'out'
    dest.mkdir()
    with tarfile.open(tmp_path / 'test.tar') as tar:
        tar.extractall(dest, filter=safe_archive_filter)
    path = dest / link.name
    assert path.is_symlink()
    assert path.read_bytes() == b'png'


@pytest.mark.parametrize('name,target', [('repo/link', '../../escape'), ('repo/link', '/tmp/escape')])
def test_unsafe_links_rejected(tmp_path, name, target):
    member = tarfile.TarInfo(name)
    member.type = tarfile.SYMTYPE
    member.linkname = target
    with pytest.raises(tarfile.FilterError):
        safe_archive_filter(member, tmp_path)


def test_existing_link_cannot_redirect_extraction(tmp_path):
    dest = tmp_path / 'out'
    dest.mkdir()
    (dest / 'redirect').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(tarfile.FilterError):
        safe_archive_filter(tarfile.TarInfo('redirect/escape'), dest)


@pytest.mark.parametrize('kind', [tarfile.LNKTYPE, tarfile.CHRTYPE])
def test_hardlink_escape_and_special_file_rejected(tmp_path, kind):
    member = tarfile.TarInfo('entry')
    member.type = kind
    member.linkname = '../outside'
    with pytest.raises(tarfile.FilterError):
        safe_archive_filter(member, tmp_path)


def test_materialization_failure_excluded(tmp_path, monkeypatch):
    import run_swe_explore_v5_stratified as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    row = runner.materialization_failure({'instance_id': 'bad', 'repo': 'r', 'base_commit': 'c'}, 'unsafe archive')
    runner.aggregate([row])
    report = runner.read(tmp_path / 'comparison_summary.json')
    assert report['materialization_failed_count'] == 1
    assert report['evaluated_pairs'] == 0
    assert report['arms']['with_skill']['function_evaluable'] == 0


def test_balanced_reproducible_selection():
    rows = [{'repo': r, 'instance_id': f'{r}-{i}'} for r, n in [('a', 50), ('b', 20), ('c', 1)] for i in range(n)]
    selected, quotas = balanced_sample(rows, 11, 10)
    assert quotas == {'a': 5, 'b': 5, 'c': 1}
    assert len({r['instance_id'] for r in selected}) == 11
    assert selected == balanced_sample(rows, 11, 10)[0]


def test_strict_function_and_region_policy(tmp_path):
    assert strict_metrics(['a.py::f'], ['a.py::C.f'])['rank'] is None
    (tmp_path / 'a.py').write_text('class C:\n    def f(self):\n        pass\n')
    assert strict_regions(['a.py::f', 'a.py::missing', 'a.py::C'], tmp_path) == []
    assert strict_regions(['a.py::C.f'], tmp_path) == [('a.py', 2, 3)]


def test_cleanup_rejects_outside(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        clean_workspace(tmp_path)


def test_skipped_case_separate_from_metrics(tmp_path, monkeypatch):
    import run_swe_explore_v5_stratified as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    skipped = {'instance_id': 'skipped', 'repo': 'a', 'functions': [], 'skip_reason': 'user decision',
               'arms': {a: {'status': 'skipped_materialization', 'metrics': strict_metrics([], [])} for a in runner.ARMS}}
    completed = {'instance_id': 'done', 'repo': 'a', 'functions': ['a.py::f'],
                 'arms': {a: {'status': 'completed', 'metrics': strict_metrics(['a.py::f'], ['a.py::f'])} for a in runner.ARMS}}
    runner.aggregate([skipped, completed])
    report = runner.read(tmp_path / 'comparison_summary.json')
    assert report['skipped_count'] == 1
    assert report['evaluated_pairs'] == 1
    assert report['arms']['with_skill']['function_evaluable'] == 1
    assert report['arms']['with_skill']['micro']['top1'] == 1
