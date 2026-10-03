"""Read-only candidate audit; no model calls, downloads, or frozen-run edits."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from run_rq1_temporal import OUT, ROOT, cached_dataset, exposure


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def digest(text):
    return hashlib.sha256(text.replace('\r\n', '\n').strip().encode()).hexdigest()


def counts(rows):
    return {'count': len(rows), 'repos': dict(sorted(Counter(c['repo'] for c in rows).items())),
            'years': dict(sorted(Counter(c['created_at'][:4] for c in rows).items()))}


def visible_issue_groups(case, meta):
    body = re.sub(r'<!--.*?-->', '', meta.get('body') or '', flags=re.S)
    links = re.findall(r'https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)', body)
    local = re.findall(r'(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)', body)
    return {f'{repo}#{n}' for repo, n in links} | {case['repo'] + '#' + n for n in local}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Audit output must be outside the frozen run')
    full, source = cached_dataset('SWE-bench___swe-bench', 'swe-bench-test.arrow')
    lite, _ = cached_dataset('SWE-bench___swe-bench_lite', 'swe-bench_lite-test.arrow')
    verified, _ = cached_dataset('SWE-bench___swe-bench_verified', 'swe-bench_verified-test.arrow')
    training = read(OUT / 'training_summary.json')['cases']
    trained_ids = {c['instance_id'] for c in training}
    original_train = read(OUT / 'acquisition_candidates.json')
    original_test = read(OUT / 'evaluation_candidates.json')
    original_test_ids = {c['instance_id'] for c in original_test}
    original_exposure = set(read(OUT / 'exposure_audit.json')['instance_ids'])
    current_exposure, exposure_files = exposure(ROOT / 'runs')
    exposed = original_exposure | current_exposure | trained_ids | original_test_ids
    metadata = {p.stem: read(p) for p in (OUT / 'pr_metadata').glob('*.json')}
    mapped = {p.stem: read(p) for p in (OUT / 'materialized').glob('*.json')}
    groups = {g for c in original_test for g in metadata.get(c['instance_id'], {}).get('issue_groups', [])}
    clean_groups = {g for c in original_test for g in visible_issue_groups(c, metadata.get(c['instance_id'], {}))}
    trained_cases = [c for c in full if c['instance_id'] in trained_ids]
    training_hashes = {digest(c['patch']) for c in trained_cases}
    training_issues = {(c['repo'], digest(c['problem_statement'])) for c in trained_cases}
    training_groups = {g for c in trained_cases for g in metadata.get(c['instance_id'], {}).get('issue_groups', [])}
    test_hashes = {digest(c['patch']) for c in original_test}
    excluded_records = read(OUT / 'acquisition_excluded.json', [])
    remaining = []
    for case in original_train:
        cid = case['instance_id']
        if cid in trained_ids:
            continue
        meta, mapping = metadata.get(cid), mapped.get(cid)
        reasons, pending = [], []
        if not meta:
            pending.append('merge_time_and_shared_issue_metadata')
        elif not meta.get('merged_at') or meta['merged_at'] >= '2020-01-01T00:00:00Z':
            reasons.append('late_or_unmerged')
        if meta and groups.intersection(meta.get('issue_groups', [])):
            reasons.append('shared_original_evaluation_issue')
        if digest(case['patch']) in test_hashes:
            reasons.append('duplicate_evaluation_patch')
        if mapping is None:
            pending.append('source_function_mapping')
        elif not mapping.get('function_ground_truth'):
            reasons.append('no_mapped_function')
        remaining.append({'instance_id': cid, 'repo': case['repo'], 'created_at': case['created_at'],
                          'merged_at': meta.get('merged_at') if meta else None,
                          'status': 'excluded' if reasons else 'pending' if pending else 'eligible_cached',
                          'reasons': reasons, 'pending': pending})
    recoverable = []
    for row in remaining:
        if row['reasons'] != ['shared_original_evaluation_issue']:
            continue
        case = next(c for c in original_train if c['instance_id'] == row['instance_id'])
        meta = metadata[row['instance_id']]
        if not clean_groups.intersection(visible_issue_groups(case, meta)):
            recoverable.append({**row,
                'template_only_overlapping_issue_ids': sorted(groups.intersection(meta.get('issue_groups', []))),
                'visible_issue_groups': sorted(visible_issue_groups(case, meta)),
                'next_check': 'source_function_mapping_and_issue_dedup_manual_review'})
    valid = lambda c: all(str(c.get(k) or '').strip() for k in
                         ('repo', 'instance_id', 'base_commit', 'problem_statement', 'patch', 'created_at'))
    late = [c for c in full if valid(c) and c['created_at'] >= '2021-01-01']
    previous_extra = [c for c in late if c['instance_id'] not in original_exposure | original_test_ids]
    extra, seen_patches, seen_issues = [], set(), set()
    for case in sorted(previous_extra, key=lambda c: c['instance_id']):
        cid, repo = case['instance_id'], case['repo']
        ph, ih = digest(case['patch']), (repo, digest(case['problem_statement']))
        meta = metadata.get(cid)
        reasons = []
        if cid in exposed:
            reasons.append('retained_historical_exposure')
        if ph in training_hashes or ih in training_issues:
            reasons.append('duplicate_training_patch_or_issue_text')
        if meta and training_groups.intersection(meta.get('issue_groups', [])):
            reasons.append('shared_training_issue')
        if (repo, ph) in seen_patches or ih in seen_issues:
            reasons.append('duplicate_within_extension')
        if ph in test_hashes:
            reasons.append('duplicate_original_evaluation_patch')
        if not reasons:
            seen_patches.add((repo, ph))
            seen_issues.add(ih)
        extra.append({'instance_id': cid, 'repo': repo, 'created_at': case['created_at'],
                      'status': 'excluded' if reasons else 'pending', 'reasons': reasons,
                      'shared_issue_metadata_cached': meta is not None,
                      'function_mapping_cached': cid in mapped,
                      'python_files_in_patch': len(re.findall(r'^\+\+\+ b/.+\.py$', case['patch'], re.M))})
    reserved = {c['instance_id'] for c in lite + verified}
    early_reserved = [c for c in full if valid(c) and c['created_at'] < '2020-01-01'
                      and c['instance_id'] in reserved]
    buffer = [c for c in full if valid(c) and '2020-01-01' <= c['created_at'] < '2021-01-01']
    trained_repo = Counter(c['repo'] for c in trained_cases)
    test_repo = Counter(c['repo'] for c in original_test)
    remaining_repo = Counter(c['repo'] for c in remaining if c['status'] != 'excluded')
    eligible_test = [c for c in read(OUT / 'comparison_summary.json')['cases'] if c.get('functions')]
    report = {
        'source': source, 'audit_type': 'local_metadata_only_no_model_calls',
        'original': {'training_completed': len(training), 'acquisition_pool': len(original_train),
                     'evaluation_candidates': len(original_test), 'evaluation_function_eligible': len(eligible_test),
                     'training': counts(trained_cases), 'test_candidates': counts(original_test)},
        'remaining_original_training': {
            **counts(remaining), 'status_counts': dict(Counter(c['status'] for c in remaining)),
            'exclusion_reasons': dict(Counter(r for c in remaining for r in c['reasons'])),
            'pending_checks': dict(Counter(r for c in remaining if c['status'] != 'excluded' for r in c['pending'])),
            'not_known_excluded': counts([c for c in remaining if c['status'] != 'excluded']),
            'historical_exclusion_log_count': len(excluded_records)},
        'template_comment_false_overlap_candidates': counts(recoverable),
        'late_test_extension': {
            'full_late_valid': len(late), 'previously_reported_candidates': len(previous_extra),
            'status_counts': dict(Counter(c['status'] for c in extra)),
            'exclusion_reasons': dict(Counter(r for c in extra for r in c['reasons'])),
            'remaining': counts([c for c in extra if not c['reasons']])},
        'alternative_training_pools_not_proposed_for_automatic_use': {
            'early_lite_or_verified_reserved': counts(early_reserved), 'year_2020_buffer': counts(buffer)},
        'repo_coverage': [{'repo': repo, 'trained': trained_repo[repo], 'remaining_training_not_excluded': remaining_repo[repo],
                          'original_test_candidates': test_repo[repo]}
                         for repo in sorted(set(trained_repo) | set(remaining_repo) | set(test_repo))],
        'fresh_exposure_audit': {'instance_count': len(current_exposure), 'source_file_count': len(exposure_files)},
        'limitations': [
            'Pending candidates are not function-eligible counts. No source download or new PR metadata requests.',
            'Shared-issue checks depend on cached PR bodies; missing metadata and unlinked duplicate issues remain unaudited.',
            'Retained-record exposure audit cannot certify deleted or unrecorded experiments.',
            'Source mapping must settle newly introduced function policy before a new evaluation manifest is frozen.',
            'Original test outcomes are already observed; expanded-bank reevaluation is follow-up, not fresh blind evaluation.']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [('audit_summary.json', report), ('remaining_training_candidates.json', remaining),
                        ('additional_evaluation_candidates.json', extra),
                        ('template_overlap_reaudit_candidates.json', recoverable)]:
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
