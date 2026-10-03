import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rq1_temporal import split_pool, exposure


def case(cid, year, patch=None):
    return dict(instance_id=cid, repo='owner/repo', base_commit='abc',
                problem_statement='A real issue', patch=patch or cid, created_at=str(year)+'-06-01T00:00:00Z')


def test_split_excludes_reserved_and_exposed():
    a, b, c, d = case('r-1', 2019), case('r-2', 2018), case('r-3', 2022), case('r-4', 2023)
    train, test = split_pool([a,b,c,d], [b], [c,d], {'r-4'})
    assert [x['instance_id'] for x in train] == ['r-1']
    assert [x['instance_id'] for x in test] == ['r-3']


def test_split_removes_cross_split_duplicate_patch_and_empty_issue():
    a, b, c = case('r-1', 2019, 'same'), case('r-2', 2022, 'same'), case('r-3', 2019)
    c['problem_statement'] = ''
    train, test = split_pool([a,b,c], [], [b], set())
    assert train == []
    assert len(test) == 1


def test_exposure_reads_nested_and_skips_repo_sources(tmp_path):
    (tmp_path/'selected_cases.json').write_text('[{"instance_id":"r-1"}]')
    (tmp_path/'work').mkdir()
    (tmp_path/'work'/'task.json').write_text('{"instance_id":"r-2"}')
    ids, sources = exposure(tmp_path)
    assert ids == {'r-1'}
    assert len(sources) == 1
