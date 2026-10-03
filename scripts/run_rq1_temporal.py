"""Audited chronological acquisition and frozen function-FL evaluation in WSL."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from collections import Counter

import run_swe_explore_v5_stratified as bench
from evolutefl.config import load_config, llm_config
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution
from evolutefl.skills import make_skill_bank

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/rq1_temporal_deepseek_20260915'
read, write, sha = bench.read, bench.write, bench.sha
ARMS = ('with_skill',)


def exposure(root):
    ids, sources = set(), []
    names = {'selected_cases.json', 'training_cases.json', 'training_candidates.json',
             'evaluation_cases.json', 'task.json', 'initial_payload.json', 'training_record.json'}
    def collect(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get('instance_id'), str):
                ids.add(obj['instance_id'])
            for value in obj.values():
                collect(value)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'work', 'repos', 'repos_base', 'repos_work',
                   'sources', '.git', OUT.name} and not (Path(folder) / d).is_symlink()]
        for name in sorted(names.intersection(files)):
            path = Path(folder) / name
            if not path.is_symlink():
                collect(json.loads(path.read_text(encoding='utf-8-sig')))
                sources.append(str(path))
    return ids, sources


def cached_dataset(name, filename):
    from datasets import Dataset
    paths = sorted((Path.home() / '.cache/huggingface/datasets' / name).rglob(filename))
    if len(paths) != 1:
        raise ValueError(f'Expected one pinned cache for {name}/{filename}, got {len(paths)}')
    return list(Dataset.from_file(str(paths[0]))), {'path': str(paths[0]), 'sha256': sha(paths[0])}


def split_pool(full, lite, verified, exposed):
    reserved = {c['instance_id'] for c in lite + verified}
    valid = lambda c: all(str(c.get(k) or '').strip() for k in
                         ('repo', 'instance_id', 'base_commit', 'problem_statement', 'patch', 'created_at'))
    train = [c for c in full if valid(c) and c['instance_id'] not in reserved
             and c['created_at'] < '2020-01-01']
    test = [c for c in verified if valid(c) and c['created_at'] >= '2021-01-01'
            and c['instance_id'] not in exposed]
    # Labels are used only to remove exact duplicate patches across acquisition/evaluation.
    digest = lambda c: hashlib.sha256(c['patch'].replace('\r\n', '\n').encode()).hexdigest()
    test_hashes = {digest(c) for c in test}
    train = [c for c in train if digest(c) not in test_hashes]
    unique = []
    seen = set()
    for c in sorted(train, key=lambda c: c['instance_id']):
        key = (c['repo'], digest(c))
        if key not in seen:
            unique.append(c)
            seen.add(key)
    random.Random(20260915).shuffle(unique)
    return unique, sorted(test, key=lambda c: c['instance_id'])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return
    specs = [('SWE-bench___swe-bench', 'swe-bench-test.arrow'),
             ('SWE-bench___swe-bench_lite', 'swe-bench_lite-test.arrow'),
             ('SWE-bench___swe-bench_verified', 'swe-bench_verified-test.arrow')]
    loaded = [cached_dataset(*s) for s in specs]
    exposed, files = exposure(ROOT / 'runs')
    train, test = split_pool(*(x[0] for x in loaded), exposed)
    if not train or not test:
        raise ValueError('No acquisition or unexposed evaluation candidates')
    write(OUT / 'acquisition_candidates.json', train)
    write(OUT / 'evaluation_candidates.json', test)
    write(OUT / 'exposure_audit.json', {'instance_ids': sorted(exposed), 'files': files,
          'limitation': 'Conservative retained-record exclusion; deleted historical exposures cannot be certified.'})
    cfg = load_config()
    cfg['llm'] = llm_config(cfg, 'deepseek', 'deepseek-v4-flash')
    cfg['llm']['temperature'] = 0
    if cfg['llm'].get('api_key'):
        raise ValueError('Environment credentials only')
    cfg['explorer'].update(workflow_version='v5', max_steps=30, max_runtime_seconds=1200)
    cfg['reflection']['strict_function_matching'] = True
    cfg['embedding']['enabled'] = False
    cfg['skill_bank'].update(path=str(OUT / 'initial_skills.jsonl'),
                            enabled_skill_types=['fault_skill'], retrieval_mode='lexical')
    hashes = {}
    for section in ('explorer', 'reflection'):
        for key, value in list(cfg[section].items()):
            if key.endswith('prompt_path') and value:
                target = OUT / 'prompts' / Path(value).name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(value, target)
                cfg[section][key] = str(target)
                hashes[str(target)] = sha(target)
    for name in ('initial_skills.jsonl', 'empty_skills.jsonl'):
        path = OUT / name
        if path.exists() and path.stat().st_size:
            raise ValueError('Refusing to clear existing bank')
        path.touch(exist_ok=True)
    write(OUT / 'config.json', cfg)
    for path in [OUT / 'config.json', OUT / 'acquisition_candidates.json', OUT / 'evaluation_candidates.json',
                 OUT / 'initial_skills.jsonl', OUT / 'empty_skills.jsonl',
                 *sorted((ROOT / 'src').rglob('*.py')), Path(__file__), Path(bench.__file__)]:
        hashes[str(path)] = sha(path)
    write(OUT / 'protocol.json', {'target_acquisition_count': 200, 'seed': 20260915,
        'acquisition_candidate_count': len(train), 'evaluation_candidate_count': len(test),
        'acquisition_before': '2020-01-01T00:00:00Z', 'evaluation_created_from': '2021-01-01',
        'acquisition_selection': 'seeded random order; first 200 timestamp/defect-group/function-eligible cases',
        'evaluation_selection': 'all late Verified candidates not in retained exposure records; function filter before inference',
        'patch': 'repair patch; map old-side functions, never apply to Explorer snapshot',
        'arms': list(ARMS), 'model': cfg['llm']['model'], 'datasets': [x[1] for x in loaded],
        'pretraining_contamination': 'Not eliminated by chronological memory isolation',
        'hashes': hashes})
    write(OUT / 'progress_summary.json', {'phase': 'prepared', 'training_target': 200,
          'acquisition_candidates': len(train), 'evaluation_candidates': len(test),
          'training_repositories': dict(Counter(c['repo'] for c in train)),
          'evaluation_repositories': dict(Counter(c['repo'] for c in test))})


def verify():
    for path, digest in read(OUT / 'protocol.json')['hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen experiment input changed: ' + path)


def pr_metadata(case):
    import requests
    path = OUT / 'pr_metadata' / (case['instance_id'] + '.json')
    if path.exists():
        return read(path)
    number = case['instance_id'].rsplit('-', 1)[1]
    headers = {'Accept': 'application/vnd.github+json'}
    token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    for attempt in range(4):
        response = requests.get(f"https://api.github.com/repos/{case['repo']}/pulls/{number}",
                                headers=headers, timeout=45)
        if response.status_code in (403, 429) and response.headers.get('X-RateLimit-Remaining') == '0':
            delay = max(30, min(3700, int(response.headers.get('X-RateLimit-Reset', time.time()+3600))-int(time.time())+5))
            write(OUT / 'current_case.json', {'phase': 'github_rate_limit_wait', 'instance_id': case['instance_id'], 'wait_seconds': delay})
            time.sleep(delay)
            continue
        response.raise_for_status()
        raw = response.json()
        data = {k: raw.get(k) for k in ('number', 'html_url', 'created_at', 'merged_at', 'merge_commit_sha', 'body')}
        data['source'] = 'GitHub pull request API'
        data['fetched_at'] = time.time()
        # Linked issue identities catch multiple PRs addressing a shared issue.
        import re
        body = raw.get('body') or ''
        links = re.findall(r'https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)', body)
        local = re.findall(r'(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)', body)
        data['issue_groups'] = sorted({f'{repo}#{n}' for repo, n in links} | {case['repo']+'#'+n for n in local})
        write(path, data)
        return data
    raise RuntimeError('GitHub metadata unavailable after bounded attempts')


def materialize(case):
    bench.OUT = OUT
    if shutil.disk_usage(OUT).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free; no automatic artifact deletion')
    work = OUT / 'work' / case['instance_id']
    root, digest = bench.unpack(case, work)
    path = OUT / 'materialized' / (case['instance_id'] + '.json')
    saved = read(path)
    if saved and saved['original_archive_sha256'] != digest:
        raise ValueError('Repository archive changed')
    if not saved:
        symbols = bench.symbols_from_patch(case['patch'], root, source_side='old')
        saved = {**case, 'original_archive_sha256': digest, 'function_ground_truth': symbols['functions'],
                 'classes': symbols['classes']}
        write(path, saved)
    return work, root, saved


def explorer(case, cfg, root, directory, bank):
    result = read(directory / 'result.json')
    if result is not None:
        return result
    local = copy.deepcopy(cfg)
    local['skill_bank']['path'] = str(bank)
    if bank.name == 'empty_skills.jsonl':
        local['skill_bank']['enabled_skill_types'] = []
    directory.mkdir(parents=True, exist_ok=True)
    return run_explorer(task={'instance_id': case['instance_id'], 'repo': case['repo'],
          'base_commit': case['base_commit'], 'bug_report': case['problem_statement'],
          'repo_path': str(root), 'run_dir': str(directory)}, config=local,
          llm_client=OpenAICompatibleClient.from_config(local['llm']), skill_bank=make_skill_bank(local))


def train(cfg, limit):
    bank = OUT / 'initial_skills.jsonl'
    rows, excluded = [], []
    # Resolve evaluation PR groups before learning from any acquisition patch.
    groups = set()
    for c in read(OUT / 'evaluation_candidates.json'):
        groups.update(pr_metadata(c)['issue_groups'])
    for case in read(OUT / 'acquisition_candidates.json'):
        if len(rows) >= limit:
            break
        cid = case['instance_id']
        path = OUT / 'training' / cid / 'record.json'
        saved = read(path)
        if saved:
            if saved['input_sha256'] != sha(bank) or saved['output_sha256'] != sha(Path(saved['output_bank'])):
                raise ValueError('Broken acquisition checkpoint chain')
            bank = Path(saved['output_bank'])
            rows.append(saved)
            continue
        meta = pr_metadata(case)
        if not meta['merged_at'] or meta['merged_at'] >= '2020-01-01T00:00:00Z' or groups.intersection(meta['issue_groups']):
            excluded.append({'instance_id': cid, 'reason': 'late/unmerged or shared evaluation issue'})
            write(OUT / 'acquisition_excluded.json', excluded)
            continue
        write(OUT / 'current_case.json', {'phase': 'acquisition_explorer', 'index': len(rows)+1, 'instance_id': cid})
        work, root, mapped = materialize(case)
        if not mapped['function_ground_truth']:
            excluded.append({'instance_id': cid, 'reason': 'no old-side function target'})
            write(OUT / 'acquisition_excluded.json', excluded)
            bench.clean_workspace(work)
            continue
        directory = path.parent
        trace = directory / 'explorer'
        result = explorer(case, cfg, root, trace, bank)
        local = copy.deepcopy(cfg)
        transaction = directory / 'skills.jsonl'
        summary_path = directory / 'evolution/case_evolution_summary.json'
        summary = read(summary_path)
        if summary is None:
            if transaction.exists():
                raise RuntimeError('Interrupted evolution transaction needs inspection: '+cid)
            shutil.copyfile(bank, transaction)
            local['skill_bank']['path'] = str(transaction)
            write(OUT / 'current_case.json', {'phase': 'acquisition_reflector', 'index': len(rows)+1, 'instance_id': cid})
            summary = run_case_evolution(case_run_dir=trace, repo=case['repo'], issue=case['problem_statement'],
                repo_path=root, config=local, llm_client=OpenAICompatibleClient.from_config(cfg['llm']),
                ground_truth_functions=mapped['function_ground_truth'], ground_truth_patch=case['patch'],
                patch_metadata={'source': 'SWE-bench historical repair', 'direction': 'buggy_base_to_repaired'},
                output_dir=directory / 'evolution')
        next_bank = transaction if summary['status'] == 'completed' else bank
        record = {'instance_id': cid, 'merged_at': meta['merged_at'], 'input_sha256': sha(bank),
             'output_bank': str(next_bank), 'output_sha256': sha(next_bank), 'explorer_status': result.get('status'),
             'evolution_status': summary['status'], 'decision': summary.get('decision'),
             'metrics': bench.strict_metrics(result.get('ranked_functions', [])[:5], mapped['function_ground_truth'])}
        write(path, record)
        bank = next_bank
        rows.append(record)
        bench.clean_workspace(work)
        write(OUT / 'training_summary.json', {'processed': len(rows), 'target': 200, 'cases': rows,
              'bank': str(bank), 'active_skills': len(make_skill_bank({**cfg, 'skill_bank': {**cfg['skill_bank'], 'path':str(bank)}}).active_skills())})
        print(f'ACQUIRED {len(rows)}/200 {cid} {summary["status"]}', flush=True)
        if len(rows) <= 2 and (result.get('status') != 'completed' or summary['status'] == 'failed'):
            raise RuntimeError('Acquisition smoke protocol failed; inspect before full run')
    if len(rows) < limit:
        raise RuntimeError('Not enough eligible acquisition cases')
    if limit == 200:
        target = OUT / 'frozen_skills.jsonl'
        if target.exists() and sha(target) != sha(bank):
            raise ValueError('Frozen bank differs')
        shutil.copyfile(bank, target)
        write(OUT / 'training_complete.json', {'count': len(rows), 'bank_sha256': sha(target)})


def evaluate(cfg):
    bank = OUT / 'frozen_skills.jsonl'
    digest = read(OUT / 'training_complete.json')['bank_sha256']
    rows = []
    for index, case in enumerate(read(OUT / 'evaluation_candidates.json')):
        verify()
        if sha(bank) != digest:
            raise ValueError('Frozen bank modified')
        cid = case['instance_id']
        path = OUT / 'paired' / (cid+'.json')
        if path.exists():
            rows.append(read(path))
            continue
        work, root, mapped = materialize(case)
        row = {'instance_id': cid, 'repo': case['repo'], 'functions': mapped['function_ground_truth'], 'arms': {}}
        if mapped['function_ground_truth']:
            for arm in ARMS:
                write(OUT / 'current_case.json', {'phase': arm, 'index': index+1, 'instance_id': cid})
                work, root, _ = materialize(case)
                result = explorer(case, cfg, root, OUT / arm / 'cases' / cid, bank)
                predictions = result.get('ranked_functions', [])
                search = read(OUT / arm / 'cases' / cid / 'fault_skill_search.json', {})
                row['arms'][arm] = {'status': result.get('status'), 'predictions': predictions,
                    'metrics': bench.strict_metrics(predictions[:5], mapped['function_ground_truth']),
                    'loaded_skill_id': (search.get('matched_skill') or {}).get('skill_id'),
                    'forced_finish': result.get('forced_finish'), 'steps': result.get('steps'),
                    'runtime_seconds': result.get('runtime_seconds')}
            if sha(bank) != digest:
                raise ValueError('Bank modified during evaluation')
        else:
            row['exclusion'] = 'No old-side function target; excluded before inference for all arms'
        write(path, row)
        rows.append(row)
        eligible = [r for r in rows if r['functions']]
        write(OUT / 'comparison_summary.json', {'processed': len(rows), 'function_eligible': len(eligible),
          'arms': {a: {k: sum(r['arms'][a]['metrics'][k] for r in eligible)/len(eligible) if eligible else None
                      for k in ('top1','top3','top5','mrr')} for a in ARMS}, 'cases': rows})
        bench.clean_workspace(work)
    write(OUT / 'current_case.json', {'phase': 'finished', 'evaluated': len(rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare','smoke','full'], default='prepare')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (OUT / 'experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare()
            verify()
            if args.phase == 'prepare':
                print(json.dumps(read(OUT / 'progress_summary.json')), flush=True)
                return
            cfg = read(OUT / 'config.json')
            if not os.getenv(cfg['llm']['api_key_env']):
                raise ValueError('Missing credential environment')
            train(cfg, 2 if args.phase == 'smoke' else 200)
            if args.phase == 'full':
                evaluate(cfg)
        except Exception as exc:
            write(OUT / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)[:500],
                  'current_case': read(OUT / 'current_case.json'), 'time': time.time()})
            raise


if __name__ == '__main__':
    main()
