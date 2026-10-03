import sys
from pathlib import Path
from unittest.mock import patch
import pytest
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_rq1_expanded as runner


def test_new_only_function_never_used_as_fallback():
    assert runner.existing_targets({'existing_function_targets': [], 'functions': ['a.py::new']}) == []
    with pytest.raises(KeyError):
        runner.existing_targets({'functions': ['a.py::new']})


def test_existing_targets_deduplicated():
    assert runner.existing_targets({'existing_function_targets': ['a.py::old', 'a.py::old'],
                                    'new_only_function_targets': ['a.py::new']}) == ['a.py::old']


def test_main_runner_evaluates_only_the_proposed_method():
    assert runner.ARMS == ('with_skill',)


def test_explorer_only_sees_issue_and_repo_not_labels(tmp_path):
    case = {'instance_id': 'a-1', 'repo': 'a/b', 'base_commit': 'abc', 'problem_statement': 'bug',
            'patch': 'SECRET PATCH', 'function_ground_truth': ['secret.py::answer']}
    cfg = {'llm': {}, 'skill_bank': {'enabled_skill_types': ['fault_skill']}}
    with patch.object(runner.OpenAICompatibleClient, 'from_config'), patch.object(runner, 'make_skill_bank'), \
         patch.object(runner, 'run_explorer', return_value={'status': 'completed'}) as call:
        runner.explorer(case, cfg, tmp_path, tmp_path / 'run', tmp_path / 'empty_skills.jsonl')
        task = call.call_args.kwargs['task']
        assert 'patch' not in task and 'function_ground_truth' not in task
        assert 'SECRET' not in repr(task)
        assert call.call_args.kwargs['config']['skill_bank']['enabled_skill_types'] == []
    assert cfg['skill_bank']['enabled_skill_types'] == ['fault_skill']


def test_reuse_completed_result_without_model(tmp_path):
    directory = tmp_path / 'run'
    runner.write(directory / 'result.json', {'status': 'completed'})
    with patch.object(runner, 'run_explorer') as call:
        assert runner.explorer({}, {}, tmp_path, directory, tmp_path / 'bank')['status'] == 'completed'
        call.assert_not_called()


def test_frozen_hash_changes_fail_closed(tmp_path):
    value = tmp_path / 'input.json'
    value.write_text('{}')
    runner.write(tmp_path / 'protocol.json', {'hashes': {str(value): runner.sha(value)}})
    with patch.object(runner, 'OUT', tmp_path):
        runner.verify_protocol()
        value.write_text('{"changed":true}')
        with pytest.raises(ValueError, match='Frozen extension input changed'):
            runner.verify_protocol()


def _fake_isolated_case(args):
    root, index = args
    runner.OUT = Path(root)
    case = {'instance_id': f'case-{index}', 'repo': 'a/b', 'function_ground_truth': ['a.py::f']}
    workspace = Path(root) / 'work' / case['instance_id']
    workspace.mkdir(parents=True)
    with patch.object(runner, 'materialize', return_value=(workspace, workspace)), \
         patch.object(runner, 'explorer', return_value={'status': 'completed', 'ranked_functions': ['a.py::f']}), \
         patch.object(runner.bench, 'clean_workspace'):
        time.sleep(0.1)
        result = runner.evaluate_case((index, case))
    return os.getpid(), result


def test_four_process_case_isolation_smoke(tmp_path):
    bank = tmp_path / 'frozen_skills.jsonl'
    bank.write_text('')
    runner.write(tmp_path / 'training_complete.json', {'bank_sha256': runner.sha(bank)})
    runner.write(tmp_path / 'config.json', {'llm': {}})
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_fake_isolated_case, [(str(tmp_path), i) for i in range(8)]))
    assert len({pid for pid, _ in results}) > 1
    assert len(list((tmp_path / 'paired').glob('*.json'))) == 8
    for _, row in results:
        assert set(row['arms']) == set(runner.ARMS)
        assert all(arm['metrics']['top1'] for arm in row['arms'].values())
    assert runner.sha(bank) == runner.read(tmp_path / 'training_complete.json')['bank_sha256']
