"""Chronological extension: existing-function audit, serial learning, parallel FL.

Independent of the completed RQ1 run. All credentials are environment-only.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from collections import Counter

from audit_rq1_temporal_expansion import Audit, DEFAULT_OUT as AUDIT_CACHE, temporal_train_ok
from audit_rq1_expansion_candidates import digest
from run_rq1_temporal import ROOT, OUT as ORIGINAL, read, write, sha, verify as verify_original
import run_swe_explore_v5_stratified as bench
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution
from evolutefl.skills import make_skill_bank

OUT = ROOT / 'runs/rq1_expanded400_eval500_v4flash_existingfunc_20260922'
ARMS = ('with_skill', 'no_skill', 'agentless_fl')
POLICY = 'changed_patch_functions_intersect_existing_base_functions_v1'


def existing_targets(mapping):
    # An empty list is meaningful: never fall back to new-only repair functions.
    return list(dict.fromkeys(mapping['existing_function_targets']))


def signature(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rescore_original():
    worker = ExtensionAudit(OUT)
    predictions = {c['instance_id']: c for c in read(ORIGINAL / 'comparison_summary.json')['cases']}
    rows = []
    for index, case in enumerate(read(OUT / 'original_evaluation.json'), 1):
        progress('rescore_original', index=index, target=178, instance_id=case['instance_id'])
        truth = existing_targets(worker.mapping(case))
        previous = predictions[case['instance_id']]
        row = {'instance_id': case['instance_id'], 'functions': truth, 'arms': {}}
        for arm in ARMS:
            old = previous.get('arms', {}).get(arm)
            if old is not None and truth:
                row['arms'][arm] = bench.strict_metrics(list(dict.fromkeys(old['predictions']))[:5], truth)
        rows.append(row)
        eligible = [r for r in rows if r['functions'] and len(r['arms']) == len(ARMS)]
        write(OUT / 'original_predictions_rescored.json', {'processed': len(rows), 'eligible': len(eligible),
            'label_policy': POLICY, 'new_model_calls': 0,
            'arms': {a: {k: sum(r['arms'][a][k] for r in eligible) / len(eligible) if eligible else None
                          for k in ('top1', 'top3', 'top5', 'mrr')} for a in ARMS}, 'cases': rows})


def prepare(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        verify_protocol()
        return
    verify_original()
    if sha(ORIGINAL / 'frozen_skills.jsonl') != read(ORIGINAL / 'training_complete.json')['bank_sha256']:
        raise ValueError('Historical bank changed')
    cfg = copy.deepcopy(read(ORIGINAL / 'config.json'))
    cfg['llm'].update(base_url='https://api.deepseek.com', api_key_env='DEEPSEEK_API_KEY',
                      model='deepseek-v4-flash', supports_tool_choice=False, temperature=0)
    if cfg['llm'].get('api_key'):
        raise ValueError('Credentials must be environment-only')
    for section in ('explorer', 'reflection'):
        for key, value in cfg[section].items():
            if key.endswith('prompt_path') and value:
                target = OUT / 'prompts' / Path(value).name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(value, target)
                cfg[section][key] = str(target)
    shutil.copyfile(ORIGINAL / 'frozen_skills.jsonl', OUT / 'bootstrap_skills.jsonl')
    (OUT / 'empty_skills.jsonl').touch(exist_ok=True)
    if (OUT / 'empty_skills.jsonl').stat().st_size:
        raise ValueError('Baseline bank is not empty')
    cfg['skill_bank']['path'] = str(OUT / 'bootstrap_skills.jsonl')
    write(OUT / 'config.json', cfg)
    for name in ('original_training.json', 'original_evaluation.json', 'training_candidates.json', 'evaluation_extension_order.json'):
        shutil.copyfile(AUDIT_CACHE / name, OUT / name)
    code = [*sorted((ROOT / 'src').rglob('*.py')), Path(__file__),
            ROOT / 'scripts/audit_rq1_temporal_expansion.py', ROOT / 'scripts/audit_rq1_expansion_candidates.py',
            ROOT / 'scripts/run_rq1_temporal.py', ROOT / 'scripts/run_swe_explore_v5_stratified.py',
            ROOT / 'scripts/run_agentless_heldout30.py']
    from run_agentless_heldout30 import UPSTREAM
    code.extend(sorted(UPSTREAM.rglob('*.py')))
    frozen = code + list((OUT / 'prompts').iterdir()) + [OUT / n for n in (
        'config.json', 'bootstrap_skills.jsonl', 'empty_skills.jsonl', 'original_training.json',
        'original_evaluation.json', 'training_candidates.json', 'evaluation_extension_order.json')]
    write(path, {'training_target': 400, 'bootstrap_count': 200, 'additional_target': 200,
        'evaluation_target': 500, 'workers': workers, 'arms': list(ARMS), 'label_policy': POLICY,
        'model_request': 'deepseek-v4-flash', 'provider_alias_note': 'Official legacy alias now routes to V4.1 Flash; not a pinned historical snapshot.',
        'history_note': '200-case historical bank retained; new acquisition and all evaluation use existing-function targets.',
        'train_before': '2020-01-01', 'test_created_from': '2021-01-01',
        'reuse_old_predictions': False, 'hashes': {str(p): sha(p) for p in frozen}})


def verify_protocol():
    for path, expected in read(OUT / 'protocol.json')['hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen extension input changed: ' + path)


def progress(phase, **fields):
    write(OUT / 'progress.json', {'phase': phase, 'time': time.time(), **fields})


class ExtensionAudit(Audit):
    def metadata(self, case):
        path = self.output / 'pr_metadata' / (case['instance_id'] + '.json')
        cached = AUDIT_CACHE / 'pr_metadata' / path.name
        if not path.exists() and cached.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, path)
        return super().metadata(case)

    def mapping(self, case):
        # Reuse immutable source-cache evidence, not an unverified mapping shortcut.
        cached = AUDIT_CACHE / 'function_audit' / (case['instance_id'] + '.json')
        if cached.exists():
            mapping = Audit(AUDIT_CACHE).mapping(case)
            write(self.output / 'function_audit' / cached.name, mapping)
            return mapping
        return super().mapping(case)


def audit():
    if (OUT / 'selection_complete.json').exists():
        for path, expected in read(OUT / 'selection_complete.json')['hashes'].items():
            if sha(Path(path)) != expected:
                raise ValueError('Selected manifest changed')
        return
    worker = ExtensionAudit(OUT)
    old_train = read(OUT / 'original_training.json')
    old_test = read(OUT / 'original_evaluation.json')
    test_groups = {g for c in old_test for g in worker.metadata(c)['issue_groups']}
    train_groups = {g for c in old_train for g in worker.metadata(c)['issue_groups']}
    train_hashes = {digest(c['patch']) for c in old_train}
    train_issues = {(c['repo'], digest(c['problem_statement'])) for c in old_train}
    additions, audit_rows = [], []
    for index, case in enumerate(read(OUT / 'training_candidates.json'), 1):
        if len(additions) == 200:
            break
        cid = case['instance_id']
        progress('audit_training', index=index, accepted=len(additions), target=200, instance_id=cid)
        meta = worker.metadata(case)
        reasons = []
        if not temporal_train_ok(meta):
            reasons.append('late_or_unmerged')
        if test_groups.intersection(meta['issue_groups']):
            reasons.append('shared_original_test_issue')
        ph, ih = digest(case['patch']), (case['repo'], digest(case['problem_statement']))
        if ph in train_hashes or ih in train_issues or train_groups.intersection(meta['issue_groups']):
            reasons.append('duplicate_acquisition_issue_or_patch')
        mapping = worker.mapping(case) if not reasons else None
        if mapping is not None and not existing_targets(mapping):
            reasons.append('no_existing_function_target')
        if not reasons:
            additions.append({**case, 'function_ground_truth': existing_targets(mapping), 'merged_at': meta['merged_at']})
            train_groups.update(meta['issue_groups'])
            train_hashes.add(ph)
            train_issues.add(ih)
        audit_rows.append({'instance_id': cid, 'reasons': reasons})
        write(OUT / 'training_audit.json', audit_rows)
    if len(additions) < 200:
        raise RuntimeError(f'Only {len(additions)} eligible additions; no automatic temporal relaxation')
    write(OUT / 'training_additions.json', additions)
    selected, audit_rows, seen_hashes, seen_issues, seen_groups = [], [], set(), set(), set()
    for index, case in enumerate(old_test + read(OUT / 'evaluation_extension_order.json'), 1):
        if len(selected) == 500:
            break
        cid = case['instance_id']
        progress('audit_evaluation', index=index, accepted=len(selected), target=500, instance_id=cid)
        meta = worker.metadata(case)
        ph, ih = digest(case['patch']), (case['repo'], digest(case['problem_statement']))
        reasons = []
        if not meta.get('created_at') or meta['created_at'] < '2021-01-01':
            reasons.append('early_evaluation')
        if train_groups.intersection(meta['issue_groups']) or ph in train_hashes or ih in train_issues:
            reasons.append('acquisition_overlap')
        if ph in seen_hashes or ih in seen_issues or seen_groups.intersection(meta['issue_groups']):
            reasons.append('duplicate_evaluation')
        mapping = worker.mapping(case) if not reasons else None
        if mapping is not None and not existing_targets(mapping):
            reasons.append('no_existing_function_target')
        if not reasons:
            selected.append({**case, 'function_ground_truth': existing_targets(mapping)})
            seen_hashes.add(ph)
            seen_issues.add(ih)
            seen_groups.update(meta['issue_groups'])
        audit_rows.append({'instance_id': cid, 'reasons': reasons})
        write(OUT / 'evaluation_audit.json', audit_rows)
        write(OUT / 'evaluation_cases.json', selected)
    if len(selected) < 500:
        raise RuntimeError(f'Only {len(selected)} eligible test cases; do not silently reduce or replace sample')
    paths = [OUT / 'training_additions.json', OUT / 'evaluation_cases.json']
    write(OUT / 'selection_complete.json', {'hashes': {str(p): sha(p) for p in paths},
        'additional_training_count': len(additions), 'evaluation_count': len(selected), 'label_policy': POLICY})


def materialize(case):
    bench.OUT = OUT
    if shutil.disk_usage(OUT).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free; no automatic artifact deletion')
    cid = case['instance_id']
    archive = OUT / 'sources' / (cid + '.tar.gz')
    previous = ORIGINAL / 'sources' / archive.name
    if not archive.exists() and previous.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous, archive)
    workspace = OUT / 'work' / cid
    root, archive_sha = bench.unpack(case, workspace)
    old = read(OUT / 'materialized' / (cid + '.json'))
    if old and old['original_archive_sha256'] != archive_sha:
        raise ValueError('Archive checksum changed')
    evidence = read(OUT / 'function_audit' / (cid + '.json'))
    for source in evidence['sources']:
        if sha(root / source['path']) != source['sha256']:
            raise ValueError('Archive and audited base source differ')
    write(OUT / 'materialized' / (cid + '.json'), {'instance_id': cid,
          'original_archive_sha256': archive_sha, 'function_ground_truth': case['function_ground_truth']})
    return workspace, root


def explorer(case, config, root, directory, bank):
    path = directory / 'result.json'
    if path.exists():
        return read(path)
    local = copy.deepcopy(config)
    local['skill_bank']['path'] = str(bank)
    if bank.name == 'empty_skills.jsonl':
        local['skill_bank']['enabled_skill_types'] = []
    directory.mkdir(parents=True, exist_ok=True)
    return run_explorer(task={'instance_id': case['instance_id'], 'repo': case['repo'],
        'base_commit': case['base_commit'], 'bug_report': case['problem_statement'],
        'repo_path': str(root), 'run_dir': str(directory)}, config=local,
        llm_client=OpenAICompatibleClient.from_config(local['llm']), skill_bank=make_skill_bank(local))


def train():
    config = read(OUT / 'config.json')
    bank, rows = OUT / 'bootstrap_skills.jsonl', []
    for index, case in enumerate(read(OUT / 'training_additions.json'), 1):
        directory = OUT / 'training' / case['instance_id']
        record = read(directory / 'record.json')
        if record:
            if record['input_sha256'] != sha(bank) or record['output_sha256'] != sha(Path(record['output_bank'])):
                raise ValueError('Broken training checkpoint chain')
            bank = Path(record['output_bank'])
            rows.append(record)
            continue
        progress('training_explorer', completed=200 + len(rows), target=400, instance_id=case['instance_id'])
        workspace, root = materialize(case)
        result = explorer(case, config, root, directory / 'explorer', bank)
        local = copy.deepcopy(config)
        transaction = directory / 'skills.jsonl'
        summary = read(directory / 'evolution/case_evolution_summary.json')
        if summary is None:
            if transaction.exists():
                raise RuntimeError('Interrupted evolution transaction requires inspection')
            shutil.copyfile(bank, transaction)
            local['skill_bank']['path'] = str(transaction)
            progress('training_reflection', completed=200 + len(rows), target=400, instance_id=case['instance_id'])
            summary = run_case_evolution(case_run_dir=directory / 'explorer', repo=case['repo'],
                issue=case['problem_statement'], repo_path=root, config=local,
                llm_client=OpenAICompatibleClient.from_config(local['llm']),
                ground_truth_functions=case['function_ground_truth'], ground_truth_patch=case['patch'],
                patch_metadata={'source': 'SWE-bench historical repair', 'direction': 'buggy_base_to_repaired'},
                output_dir=directory / 'evolution')
        next_bank = transaction if summary['status'] == 'completed' else bank
        record = {'instance_id': case['instance_id'], 'input_sha256': sha(bank), 'output_bank': str(next_bank),
            'output_sha256': sha(next_bank), 'explorer_status': result.get('status'),
            'evolution_status': summary['status'],
            'metrics': bench.strict_metrics(result.get('ranked_functions', [])[:5], case['function_ground_truth'])}
        write(directory / 'record.json', record)
        bank = next_bank
        rows.append(record)
        write(OUT / 'training_summary.json', {'historical_bootstrap': 200, 'new_completed': len(rows), 'cases': rows})
        if len(rows) == 100:
            shutil.copyfile(bank, OUT / 'skills_after300.jsonl')
        bench.clean_workspace(workspace)
        if index <= 2 and (result.get('status') != 'completed' or summary['status'] == 'failed'):
            raise RuntimeError('Training smoke protocol failed; no outcome-based replacement')
    frozen = OUT / 'frozen_skills.jsonl'
    if frozen.exists() and sha(frozen) != sha(bank):
        raise ValueError('Frozen bank differs from completed chain')
    shutil.copyfile(bank, frozen)
    write(OUT / 'training_complete.json', {'count': 200 + len(rows), 'bank_sha256': sha(frozen)})


def evaluate_case(job):
    """Each process runs all arms of one case; globals/workspaces are never shared."""
    index, case = job
    import fcntl
    import run_agentless_heldout30 as agentless
    cid = case['instance_id']
    (OUT / 'jobs').mkdir(exist_ok=True)
    with (OUT / 'jobs' / (cid + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank = OUT / 'frozen_skills.jsonl'
        bank_sha = read(OUT / 'training_complete.json')['bank_sha256']
        if sha(bank) != bank_sha:
            raise ValueError('Evaluation bank changed')
        finished = OUT / 'paired' / (cid + '.json')
        if finished.exists():
            row = read(finished)
            if row['bank_sha256'] != bank_sha or row['label_policy'] != POLICY:
                raise ValueError('Incompatible previous result')
            return row
        config = read(OUT / 'config.json')
        agentless.OUT, agentless.SOURCE = OUT / 'agentless_fl', OUT
        (agentless.OUT / 'work').mkdir(parents=True, exist_ok=True)
        row = {'instance_id': cid, 'repo': case['repo'], 'functions': case['function_ground_truth'],
               'bank_sha256': bank_sha, 'label_policy': POLICY, 'arms': {}}
        for arm in ARMS[index % 3:] + ARMS[:index % 3]:
            write(OUT / 'jobs' / (cid + '.json'), {'phase': arm, 'instance_id': cid, 'time': time.time()})
            workspace, root = materialize(case)
            if arm == 'agentless_fl':
                result = agentless.run_case(case, config['llm'])
                predictions = result.get('predictions', [])
            else:
                result = explorer(case, config, root, OUT / arm / 'cases' / cid,
                                  bank if arm == 'with_skill' else OUT / 'empty_skills.jsonl')
                predictions = result.get('ranked_functions', [])
            predictions = list(dict.fromkeys(predictions))[:5]
            search = read(OUT / arm / 'cases' / cid / 'fault_skill_search.json', {})
            row['arms'][arm] = {'status': result.get('status'), 'predictions': predictions,
                'metrics': bench.strict_metrics(predictions, case['function_ground_truth']),
                'loaded_skill_id': (search.get('matched_skill') or {}).get('skill_id'),
                'forced_finish': result.get('forced_finish'), 'steps': result.get('steps'),
                'runtime_seconds': result.get('runtime_seconds')}
        if sha(bank) != bank_sha:
            raise ValueError('Bank changed during evaluation')
        write(finished, row)
        bench.clean_workspace(workspace)
        write(OUT / 'jobs' / (cid + '.json'), {'phase': 'completed', 'time': time.time()})
        return row


def summarize():
    rows = [read(p) for p in sorted((OUT / 'paired').glob('*.json'))]
    summary = {'processed': len(rows), 'target': 500, 'label_policy': POLICY, 'arms': {}, 'cases': rows}
    for arm in ARMS:
        summary['arms'][arm] = {k: sum(r['arms'][arm]['metrics'][k] for r in rows) / len(rows) if rows else None
                                for k in ('top1', 'top3', 'top5', 'mrr')}
        summary['arms'][arm]['statuses'] = dict(Counter(r['arms'][arm]['status'] for r in rows))
        summary['arms'][arm]['loaded_skill_count'] = sum(bool(r['arms'][arm].get('loaded_skill_id')) for r in rows)
        summary['arms'][arm]['forced_finish_count'] = sum(bool(r['arms'][arm].get('forced_finish')) for r in rows)
    write(OUT / 'comparison_summary.json', summary)
    return summary


def evaluate():
    workers = read(OUT / 'protocol.json')['workers']
    jobs = list(enumerate(read(OUT / 'evaluation_cases.json')))
    # Training smoke gates the native Explorer/reflection protocol. Evaluation
    # uses the predeclared sample without selecting cases by localization outcome.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=verify_protocol) as pool:
        futures = {pool.submit(evaluate_case, job): job[1]['instance_id'] for job in jobs}
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            summary = summarize()
            progress('evaluation', completed=summary['processed'], target=500, workers=workers)
    progress('finished', evaluated=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'rescore', 'audit', 'full'], default='prepare')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (OUT / 'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare(args.workers)
            if args.phase == 'prepare':
                return
            rescore_original()
            if args.phase == 'rescore':
                return
            audit()
            if args.phase == 'full':
                if not os.getenv('DEEPSEEK_API_KEY'):
                    raise ValueError('Missing process credential')
                train()
                evaluate()
        except Exception as exc:
            write(OUT / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)[:500],
                  'time': time.time(), 'progress': read(OUT / 'progress.json')})
            raise


if __name__ == '__main__':
    main()
