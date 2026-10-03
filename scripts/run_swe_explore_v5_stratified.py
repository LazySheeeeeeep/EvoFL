"""Frozen, project-balanced, paired v5 Explorer evaluation (no evolution)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evolutefl.config import load_config, llm_config
from evolutefl.evaluation import symbols_from_patch
from evolutefl.evaluation.patch_ground_truth import (
    normalize_python_module_identity, split_function_identity, python_symbol_spans,
)
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.skills import make_skill_bank
from run_swe_explore_explorer_compare import _merge_case, _download_github_archive, score_case

OUT = ROOT / 'runs/swe_explore_v5_stratified100_deepseek_20260910'
ARMS = ('no_skill', 'with_skill')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_sample(cases, n, seed):
    groups = defaultdict(list)
    for c in cases:
        groups[c['repo']].append(c)
    if n > len(cases):
        raise ValueError('Not enough eligible cases')
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    quotas = Counter()
    while sum(quotas.values()) < n:
        available = [k for k in sorted(groups) if quotas[k] < len(groups[k])]
        key = min(available, key=lambda k: (quotas[k], k))
        quotas[key] += 1
    result = [c for k in sorted(groups) for c in groups[k][:quotas[k]]]
    rng.shuffle(result)
    return result, dict(quotas)


def prepare():
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return
    if (OUT / 'selected_cases.json').exists():
        raise RuntimeError('Partial preparation: inspect before replacing frozen selection')
    explore = load_dataset('SWE-Explore-Bench/SWE-Explore-Bench', split='train')
    verified = {r['instance_id']: r for r in load_dataset('princeton-nlp/SWE-bench_Verified', split='test')}
    candidates = [_merge_case(r, verified[r['instance_id']]) for r in explore
                  if r.get('dataset') == 'verified' and r['instance_id'] in verified
                  and (r.get('ground_truth') or {}).get('read_core_regions')]
    candidates = [r for r in candidates if all(r.get(k) for k in ('problem_statement', 'repo', 'base_commit', 'patch'))]
    cases, quotas = balanced_sample(candidates, 100, 20260910)
    write(OUT / 'selected_cases.json', cases)
    cfg = load_config('config/evolutefl.global.json')
    source_bank = Path(cfg['skill_bank']['path'])
    frozen_bank = OUT / 'frozen_skills.jsonl'
    frozen_bank.write_bytes(source_bank.read_bytes())
    (OUT / 'empty_skills.jsonl').write_text('')
    for section in cfg.values():
        if isinstance(section, dict):
            for key, value in list(section.items()):
                if key.endswith('prompt_path') and isinstance(value, str) and Path(value).is_file():
                    target = OUT / 'prompts' / Path(value).name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(Path(value).read_bytes())
                    section[key] = str(target)
    cfg['explorer'].update(workflow_version='v5', max_steps=30, max_runtime_seconds=1200)
    cfg['embedding']['enabled'] = False
    cfg['skill_bank'].update(path=str(frozen_bank), enabled_skill_types=['fault_skill'], retrieval_mode='lexical')
    cfg['llm'] = llm_config(cfg, 'deepseek', 'deepseek-v4-flash')
    cfg['llm']['temperature'] = 0
    if cfg['llm'].get('api_key'):
        raise ValueError('Use environment credentials only')
    write(OUT / 'frozen_config.json', cfg)
    write(OUT / 'protocol.json', {
        'sample_size': 100, 'seed': 20260910, 'sampling': 'approximately equal project quotas, capped by availability',
        'project_quotas': quotas, 'candidate_count': len(candidates),
        'bank_sha256': sha(frozen_bank), 'active_skills': len(make_skill_bank(cfg).active_skills()),
        'selection_sha256': sha(OUT / 'selected_cases.json'),
        'config_sha256': sha(OUT / 'frozen_config.json'),
        'prompt_hashes': {str(p): sha(p) for p in (OUT / 'prompts').iterdir()},
        'arms': list(ARMS), 'model': 'deepseek-v4-flash', 'evolution': False,
        'source': 'GitHub official archive at Verified base_commit; repair patch NOT applied',
        'primary_metric': 'strict patch-based function Top1/3/5 and MRR, any target hit',
        'secondary_metric': 'SWE-Explore read-core region overlap proxy, not defect-function ground truth',
        'prior_exposure': 'Historical outputs were deleted; globally unseen status cannot be certified',
    })
    print(json.dumps(read(OUT / 'protocol.json')), flush=True)


def clean_workspace(path):
    root = (OUT / 'work').resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError('Unsafe workspace cleanup target')
    if path.exists():
        shutil.rmtree(path)


def safe_archive_filter(member, destination):
    """Check symlinks relative to their parent, including on older tarfile builds."""
    if not member.issym():
        return tarfile.data_filter(member, destination)
    if os.path.isabs(member.linkname):
        raise tarfile.AbsoluteLinkError(member)
    # Retain the standard member-path, ownership and permission checks. Only
    # the buggy symlink-target calculation is replaced, not the safety policy.
    probe = copy.copy(member)
    probe.type = tarfile.REGTYPE
    checked = tarfile.data_filter(probe, destination)
    root = Path(destination).resolve()
    target = (root / Path(member.name).parent / member.linkname).resolve()
    if not target.is_relative_to(root):
        raise tarfile.LinkOutsideDestinationError(member, str(target))
    checked = copy.copy(checked)
    checked.type = tarfile.SYMTYPE
    checked.linkname = member.linkname
    checked.mode = None
    return checked


def materialization_failure(case, error):
    return {'instance_id': case['instance_id'], 'repo': case['repo'], 'base_commit': case['base_commit'],
            'functions': [], 'materialization_error': str(error),
            'arms': {a: {'status': 'materialization_failed', 'predictions': [],
                         'metrics': strict_metrics([], [])} for a in ARMS}}


def unpack(case, workspace):
    clean_workspace(workspace)
    workspace.mkdir(parents=True)
    archive = OUT / 'sources' / (case['instance_id'] + '.tar.gz')
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temp = archive.with_suffix('.download')
        _download_github_archive(f"https://codeload.github.com/{case['repo']}/tar.gz/{case['base_commit']}", temp)
        with tarfile.open(temp) as tar:
            tar.getmembers()
        temp.replace(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(workspace, filter=safe_archive_filter)
    roots = list(workspace.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError('Expected one archive repository root')
    return roots[0], sha(archive)


def strict_metrics(predictions, truth):
    rank = None
    for i, pred in enumerate(predictions, 1):
        path, name = split_function_identity(pred)
        for target in truth:
            target_path, target_name = split_function_identity(target)
            p, q = normalize_python_module_identity(pred, path, name.replace('::', '.'), target_path)
            if p == target_path and q == target_name:
                rank = i
                break
        if rank:
            break
    return {'rank': rank, **{f'top{k}': bool(rank and rank <= k) for k in (1, 3, 5)},
            'mrr': 1 / rank if rank else 0}


def strict_regions(predictions, root):
    regions = []
    for pred in predictions[:5]:
        path, name = split_function_identity(pred)
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root.resolve()) or not path.endswith('.py'):
            continue
        spans = python_symbol_spans(resolved)
        matched = [s for s in spans if s['kind'] == 'function' and s['qualified_name'] == name.replace('::', '.')]
        if len(matched) == 1:
            s = matched[0]
            regions.append((path, s['start'], s['end']))
    return regions


def aggregate(rows):
    summary = {'processed_cases': len(rows), 'planned_cases': 100, 'arms': {}}
    summary['skipped_cases'] = [r['instance_id'] for r in rows if r.get('skip_reason')]
    summary['skipped_count'] = len(summary['skipped_cases'])
    summary['materialization_failed_cases'] = [r['instance_id'] for r in rows if r.get('materialization_error')]
    summary['materialization_failed_count'] = len(summary['materialization_failed_cases'])
    summary['evaluated_pairs'] = len(rows) - summary['skipped_count'] - summary['materialization_failed_count']
    for arm in ARMS:
        eligible = [r for r in rows if r.get('functions')]
        group = [r['arms'][arm] for r in eligible]
        means = lambda xs: {k: sum(x['metrics'][k] for x in xs) / len(xs) if xs else None for k in ('top1', 'top3', 'top5', 'mrr')}
        projects = {p: means([r['arms'][arm] for r in eligible if r['repo'] == p]) for p in sorted({r['repo'] for r in eligible})}
        summary['arms'][arm] = dict(
            completed=sum(r['arms'][arm]['status'] == 'completed' for r in rows),
            function_evaluable=len(group), micro=means(group), per_project=projects,
            macro={k: sum(v[k] for v in projects.values()) / len(projects) if projects else None for k in ('top1', 'top3', 'top5', 'mrr')},
            loaded=sum(bool(r['arms'][arm].get('loaded_skill_id')) for r in rows),
            selector_selected=sum(bool(r['arms'][arm].get('selector_selected_id')) for r in rows),
            validator_passed=sum(r['arms'][arm].get('validator_applicable') is True for r in rows),
            forced_finish=sum(bool(r['arms'][arm].get('forced_finish')) for r in rows))
    eligible = [r for r in rows if r.get('functions')]
    summary['positive_top1'] = [r['instance_id'] for r in eligible if r['arms']['with_skill']['metrics']['top1'] and not r['arms']['no_skill']['metrics']['top1']]
    summary['negative_top1'] = [r['instance_id'] for r in eligible if not r['arms']['with_skill']['metrics']['top1'] and r['arms']['no_skill']['metrics']['top1']]
    summary['cases'] = rows
    write(OUT / 'comparison_summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    cfg = read(OUT / 'frozen_config.json')
    protocol = read(OUT / 'protocol.json')
    for path, expected in {**protocol['prompt_hashes'], str(OUT / 'frozen_skills.jsonl'): protocol['bank_sha256'],
                           str(OUT / 'frozen_config.json'): protocol['config_sha256'], str(OUT / 'selected_cases.json'): protocol['selection_sha256']}.items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen experimental input changed: ' + path)
    rows = []
    skip_cases = read(OUT / 'skip_cases.json', {})
    selected_ids = {c['instance_id'] for c in read(OUT / 'selected_cases.json')}
    if not set(skip_cases).issubset(selected_ids):
        raise ValueError('Skip manifest contains an unselected case')
    for index, case in enumerate(read(OUT / 'selected_cases.json')[:args.limit]):
        cid = case['instance_id']
        paired = OUT / 'paired' / (cid + '.json')
        if paired.exists():
            rows.append(read(paired))
            continue
        if cid in skip_cases:
            row = {'instance_id': cid, 'repo': case['repo'], 'base_commit': case['base_commit'],
                   'functions': [], 'skip_reason': skip_cases[cid],
                   'arms': {arm: {'status': 'skipped_materialization', 'predictions': [],
                                   'metrics': strict_metrics([], [])} for arm in ARMS}}
            write(paired, row)
            rows.append(row)
            aggregate(rows)
            print(f'{index + 1}/100 {cid} skipped by explicit user decision', flush=True)
            continue
        write(OUT / 'current_case.json', {'index': index + 1, 'instance_id': cid, 'phase': 'materializing'})
        workspace = OUT / 'work' / cid
        if shutil.disk_usage(OUT).free < 8 * 1024**3:
            raise RuntimeError('Less than 8 GiB free; pausing without deleting experiment data')
        try:
            root, archive_hash = unpack(case, workspace)
        except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
            row = materialization_failure(case, exc)
            write(paired, row)
            rows.append(row)
            aggregate(rows)
            print(f'{index + 1}/100 {cid} materialization failed: {exc}', flush=True)
            continue
        row = {'instance_id': cid, 'repo': case['repo'], 'base_commit': case['base_commit'], 'archive_sha256': archive_hash, 'arms': {}}
        try:
            scopes = symbols_from_patch(case['patch'], root, source_side='old')
            row.update(functions=scopes['functions'], classes=scopes['classes'])
        except ValueError as exc:
            row.update(functions=[], mapping_error=str(exc))
        # Counterbalance order to reduce time/provider drift. Both arms start from the same immutable archive.
        for arm in ARMS[::1 if index % 2 == 0 else -1]:
            try:
                root, _ = unpack(case, workspace)
            except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
                row = materialization_failure(case, exc)
                break
            arm_cfg = copy.deepcopy(cfg)
            if arm == 'no_skill':
                arm_cfg['skill_bank']['path'] = str(OUT / 'empty_skills.jsonl')
            dest = OUT / arm / 'cases' / cid
            dest.mkdir(parents=True, exist_ok=True)
            write(dest / 'task.json', case)
            write(OUT / 'current_case.json', {'index': index + 1, 'instance_id': cid, 'phase': arm})
            print(f'{index + 1}/100 {cid} {arm}', flush=True)
            result = read(dest / 'result.json')
            if result is None:
                try:
                    result = run_explorer(task={'instance_id': cid, 'repo': case['repo'], 'base_commit': case['base_commit'],
                        'bug_report': case['problem_statement'], 'repo_path': str(root), 'run_dir': str(dest)},
                        config=arm_cfg, llm_client=OpenAICompatibleClient.from_config(arm_cfg['llm']), skill_bank=make_skill_bank(arm_cfg))
                except Exception as exc:
                    result = {'status': 'failed', 'error': str(exc), 'ranked_functions': []}
                    write(dest / 'result.json', result)
            search = read(dest / 'fault_skill_search.json', {})
            predictions = result.get('ranked_functions', [])
            # Evaluate regions on the original snapshot even if write changed the workspace.
            try:
                root, _ = unpack(case, workspace)
            except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
                row = materialization_failure(case, exc)
                break
            regions = strict_regions(predictions, root)
            trace = search.get('search_trace') or {}
            row['arms'][arm] = dict(status=result.get('status'), predictions=predictions,
                metrics=strict_metrics(predictions, row['functions']), loaded_skill_id=(search.get('matched_skill') or {}).get('skill_id'),
                steps=result.get('steps'), forced_finish=result.get('forced_finish', False),
                fault_load_step=result.get('fault_skill_attempt_step'), runtime_seconds=result.get('runtime_seconds'),
                fault_family=search.get('fault_family'), catalog_count=trace.get('catalog_count', 0),
                selector_selected_id=(trace.get('selector') or {}).get('selected_skill_id'),
                validator_applicable=(trace.get('validator') or {}).get('applicable'),
                region_proxy=score_case(case, regions, root, prediction_kind='strict_top5_function_regions'))
            if arm == 'no_skill' and row['arms'][arm]['loaded_skill_id']:
                raise RuntimeError('Baseline unexpectedly loaded a Skill')
        if sha(OUT / 'frozen_skills.jsonl') != protocol['bank_sha256']:
            raise RuntimeError('Frozen bank changed during evaluation')
        write(paired, row)
        rows.append(row)
        aggregate(rows)
        clean_workspace(workspace)
        if index < 2 and any(v['status'] != 'completed' for v in row['arms'].values()):
            raise RuntimeError('Smoke case failed; inspect before continuing remaining cases')
    aggregate(rows)
    write(OUT / 'current_case.json', {'phase': 'finished', 'processed_cases': len(rows)})


if __name__ == '__main__':
    main()
