"""Resumable, model-free temporal/source audit for 400 acquisition / 500 evaluation.

Writes only to a new experiment directory. Fetches changed Python files at the
base commit, not complete checkouts. No training or evaluation is launched here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import time
from urllib.parse import quote

import requests

from audit_rq1_expansion_candidates import digest, visible_issue_groups
from run_rq1_temporal import OUT as ORIGINAL, ROOT, cached_dataset, read, write, sha, verify
from evolutefl.evaluation.patch_ground_truth import parse_patch_files, symbols_from_patch, python_symbol_spans

DEFAULT_OUT = ROOT / 'runs/rq1_expanded400_eval500_deepseek_20260922'


def temporal_train_ok(metadata):
    cutoff = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    return all(metadata.get(k) and dt.datetime.fromisoformat(metadata[k].replace('Z', '+00:00')) < cutoff
               for k in ('created_at', 'merged_at'))


def safe_source_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Unsafe patch path')
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Patch path escaped audit cache')
    return target


def stratified_order(cases, seed):
    """Interleave shuffled repository queues proportionally, without outcome data."""
    rng, buckets = random.Random(seed), defaultdict(list)
    for case in sorted(cases, key=lambda c: c['instance_id']):
        buckets[case['repo']].append(case)
    for repo in sorted(buckets):
        rng.shuffle(buckets[repo])
    totals = {k: len(v) for k, v in buckets.items()}
    used, result = Counter(), []
    while len(result) < len(cases):
        repo = min((k for k in buckets if used[k] < totals[k]),
                   key=lambda k: ((used[k] + 0.5) / totals[k], k))
        result.append(buckets[repo][used[repo]])
        used[repo] += 1
    return result


class Audit:
    def __init__(self, output):
        self.output = output.resolve()
        if not self.output.is_relative_to((ROOT / 'runs').resolve()) or self.output.is_relative_to(ORIGINAL.resolve()):
            raise ValueError('Use a separate output directory under runs')
        self.output.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')

    def progress(self, phase, **fields):
        write(self.output / 'audit_progress.json', {'phase': phase, 'updated_at': time.time(), **fields})

    def get(self, url):
        headers = {'Accept': 'application/vnd.github+json'} if url.startswith('https://api.github.com/') else {}
        if self.token and url.startswith('https://api.github.com/'):
            headers['Authorization'] = 'Bearer ' + self.token
        for attempt in range(4):
            response = self.session.get(url, headers=headers, timeout=(15, 90))
            if response.status_code in (403, 429) and (
                response.headers.get('X-RateLimit-Remaining') == '0' or response.headers.get('Retry-After')
            ):
                wait = max(60, min(3700, int(response.headers.get('X-RateLimit-Reset', time.time() + 3600)) - int(time.time()) + 5))
                prior = read(self.output / 'audit_progress.json', {})
                self.progress('github_rate_limit_wait', resume_phase=prior.get('phase'),
                              request=url, wait_seconds=wait, retry_after_epoch=time.time() + wait)
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise RuntimeError('GitHub bounded retry exhausted; resume audit from checkpoints')

    def metadata(self, case):
        cid = case['instance_id']
        target = self.output / 'pr_metadata' / (cid + '.json')
        if target.exists():
            return read(target)
        old = ORIGINAL / 'pr_metadata' / (cid + '.json')
        if old.exists():
            result = read(old)
            source = {'path': str(old), 'sha256': sha(old)}
        else:
            number = cid.rsplit('-', 1)[1]
            if not number.isdigit():
                raise ValueError('Invalid PR identifier')
            raw = self.get(f"https://api.github.com/repos/{case['repo']}/pulls/{number}").json()
            result = {k: raw.get(k) for k in ('number', 'html_url', 'created_at', 'merged_at', 'merge_commit_sha', 'body')}
            source = {'url': result['html_url'], 'fetched_at': time.time()}
        result = {**result, 'issue_groups': sorted(visible_issue_groups(case, result)),
                  'audit_source': source, 'issue_parser': 'visible_PR_body_without_HTML_comments_v1'}
        write(target, result)
        return result

    def mapping(self, case):
        cid = case['instance_id']
        target = self.output / 'function_audit' / (cid + '.json')
        if target.exists():
            saved = read(target)
            if saved['patch_sha256'] != digest(case['patch']):
                raise ValueError('Changed patch input')
            for source in saved['sources']:
                if sha(Path(source['cache_path'])) != source['sha256']:
                    raise ValueError('Changed cached source')
            return saved
        root = self.output / 'source_files' / cid
        sources, existing = [], set()
        for info in parse_patch_files(case['patch']):
            path = info['old_path']
            if not path.endswith('.py') or info.get('new_file'):
                continue
            destination = safe_source_path(root, path)
            url = f"https://raw.githubusercontent.com/{case['repo']}/{case['base_commit']}/{quote(path, safe='/')}"
            if not destination.exists():
                data = self.get(url).content
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_suffix(destination.suffix + '.download')
                temp.write_bytes(data)
                temp.replace(destination)
            sources.append({'path': path, 'url': url, 'cache_path': str(destination), 'sha256': sha(destination)})
            for symbol in python_symbol_spans(destination):
                if symbol['kind'] == 'function':
                    existing.add(f"{path}::{symbol['qualified_name']}")
        mapped = symbols_from_patch(case['patch'], root, source_side='old')
        functions = mapped['functions']
        result = {'instance_id': cid, 'repo': case['repo'], 'base_commit': case['base_commit'],
                  'patch_sha256': digest(case['patch']), 'sources': sources,
                  'functions': functions, 'classes': mapped['classes'],
                  'existing_function_targets': [f for f in functions if f in existing],
                  'new_only_function_targets': [f for f in functions if f not in existing],
                  'mapping_evidence': mapped['evidence'],
                  'policy': 'unchanged existing mapper; changed lines on both sides; new-only targets flagged separately'}
        write(target, result)
        return result

    def prepare(self):
        target = self.output / 'audit_protocol.json'
        if target.exists():
            protocol = read(target)
            for path, expected in protocol['hashes'].items():
                if sha(Path(path)) != expected:
                    raise ValueError('Audit input/code changed: ' + path)
            return
        verify()
        full, source = cached_dataset('SWE-bench___swe-bench', 'swe-bench-test.arrow')
        by_id = {c['instance_id']: c for c in full}
        prior = ROOT / 'runs/rq1_expansion_audit_20260922'
        candidates = read(prior / 'remaining_training_candidates.json')
        reopened = {c['instance_id'] for c in read(prior / 'template_overlap_reaudit_candidates.json')}
        train = [by_id[c['instance_id']] for c in candidates if c['status'] != 'excluded' or c['instance_id'] in reopened]
        extension = [by_id[c['instance_id']] for c in read(prior / 'additional_evaluation_candidates.json') if not c['reasons']]
        manifests = {'training_candidates.json': train,
                     'evaluation_extension_order.json': stratified_order(extension, 20260922),
                     'original_evaluation.json': read(ORIGINAL / 'evaluation_candidates.json'),
                     'original_training.json': [by_id[c['instance_id']] for c in read(ORIGINAL / 'training_summary.json')['cases']]}
        for name, value in manifests.items():
            write(self.output / name, value)
        hashed = [self.output / n for n in manifests]
        hashed += [Path(__file__), ROOT / 'scripts/audit_rq1_expansion_candidates.py',
                   ROOT / 'src/evolutefl/evaluation/patch_ground_truth.py']
        write(target, {'training_target': 400, 'evaluation_target': 500, 'seed': 20260922,
                       'dataset': source, 'train_before': '2020-01-01', 'test_created_from': '2021-01-01',
                       'original_run': str(ORIGINAL), 'selection': 'fixed repository-proportional interleaving; no outcome filtering',
                       'models_invoked': False, 'hashes': {str(p): sha(p) for p in hashed}})

    def run(self):
        self.prepare()
        old_train = read(self.output / 'original_training.json')
        candidates = read(self.output / 'training_candidates.json')
        old_test = read(self.output / 'original_evaluation.json')
        original_test_groups = {g for c in old_test for g in self.metadata(c)['issue_groups']}
        training_groups, training_patches, training_issues = set(), set(), set()
        eligible_train, training_rows = [], []
        # Audit cached PRs first to make useful progress even when GitHub is rate-limited.
        ordered = sorted(old_train + candidates,
                         key=lambda c: (not (ORIGINAL / 'pr_metadata' / (c['instance_id'] + '.json')).exists(), c['instance_id']))
        original_ids = {c['instance_id'] for c in old_train}
        for index, case in enumerate(ordered, 1):
            cid = case['instance_id']
            self.progress('training_time_and_function_audit', index=index, total=len(ordered), instance_id=cid)
            meta = self.metadata(case)
            reasons = []
            if not temporal_train_ok(meta):
                reasons.append('late_or_unmerged')
            if original_test_groups.intersection(meta['issue_groups']):
                reasons.append('shared_original_test_issue')
            mapped = self.mapping(case) if not reasons else None
            if mapped is not None and not mapped['functions']:
                reasons.append('no_mapped_function')
            if cid in original_ids and reasons:
                raise ValueError('Original training audit discrepancy: ' + cid)
            if not reasons:
                training_groups.update(meta['issue_groups'])
                training_patches.add(digest(case['patch']))
                training_issues.add((case['repo'], digest(case['problem_statement'])))
                if cid not in original_ids:
                    eligible_train.append(cid)
            training_rows.append({'instance_id': cid, 'repo': case['repo'], 'original_training': cid in original_ids,
                                  'reasons': reasons, 'merged_at': meta.get('merged_at'),
                                  'new_only_function_targets': mapped['new_only_function_targets'] if mapped else []})
            write(self.output / 'training_audit.json', training_rows)
        selected, evaluation_rows = [], []
        seen_patches, seen_issues = set(), set()
        evaluation_order = old_test + read(self.output / 'evaluation_extension_order.json')
        old_ids = {c['instance_id'] for c in old_test}
        for index, case in enumerate(evaluation_order, 1):
            if len(selected) >= 500:
                break
            cid = case['instance_id']
            self.progress('evaluation_time_and_function_audit', index=index, selected=len(selected),
                          target=500, instance_id=cid)
            meta = self.metadata(case)
            ph, ih = digest(case['patch']), (case['repo'], digest(case['problem_statement']))
            reasons = []
            if not meta.get('created_at') or meta['created_at'] < '2021-01-01':
                reasons.append('early_evaluation_PR')
            if training_groups.intersection(meta['issue_groups']) or ph in training_patches or ih in training_issues:
                reasons.append('shared_training_issue_or_patch')
            if ph in seen_patches or ih in seen_issues:
                reasons.append('duplicate_evaluation_patch_or_issue_text')
            mapped = self.mapping(case) if not reasons else None
            if mapped is not None and not mapped['functions']:
                reasons.append('no_mapped_function')
            if not reasons:
                selected.append(case)
                seen_patches.add(ph)
                seen_issues.add(ih)
            evaluation_rows.append({'instance_id': cid, 'repo': case['repo'], 'original_evaluation': cid in old_ids,
                                    'reasons': reasons, 'new_only_function_targets': mapped['new_only_function_targets'] if mapped else []})
            write(self.output / 'evaluation_audit.json', evaluation_rows)
            write(self.output / 'provisional_evaluation_cases.json', selected)
        additions = [c for c in candidates if c['instance_id'] in set(eligible_train)][:200]
        write(self.output / 'provisional_training_cases.json', old_train + additions)
        write(self.output / 'audit_summary.json', {
            'status': 'audit_complete_requires_protocol_review',
            'training_count': len(old_train) + len(additions), 'evaluation_count': len(selected),
            'training_additions_eligible': len(eligible_train),
            'training_repos': dict(Counter(c['repo'] for c in old_train + additions)),
            'evaluation_repos': dict(Counter(c['repo'] for c in selected)),
            'new_function_policy_review_count': sum(bool(r['new_only_function_targets']) for r in training_rows + evaluation_rows),
            'before_launch': ['freeze label policy and all baseline adapters', 'audit baseline memory temporal boundaries',
                              'resolve infrastructure failures consistently', 'verify final counts and source hashes'],
            'warning': 'provisional manifests; no paid model runs started by this audit'})
        self.progress('audit_complete_requires_protocol_review', training=len(old_train) + len(additions), evaluation=len(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    audit = Audit(args.output_dir)
    import fcntl
    with (audit.output / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            audit.prepare() if args.prepare_only else audit.run()
        except Exception as exc:
            write(audit.output / 'audit_failure.json', {'error_type': type(exc).__name__,
                  'error': str(exc)[:600], 'time': time.time(), 'progress': read(audit.output / 'audit_progress.json')})
            raise


if __name__ == '__main__':
    main()
