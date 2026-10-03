"""Read-only audit of a fixed completed snapshot; does not alter experiment inputs."""
import ast
import json
import re
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evolutefl.evaluation.patch_ground_truth import parse_patch_files, symbols_from_patch

OUT = ROOT / 'runs/experience_expansion99_deepseek_20260913'
SOURCE = ROOT / 'runs/swe_explore_v5_stratified100_deepseek_20260910'
read = lambda p: json.loads(p.read_text())
cases = read(OUT / 'selected_cases.json')
snapshot = read(OUT / 'comparison_summary.json')
rows = snapshot['cases']
report = {'snapshot_cases': len(rows), 'label_checks': [], 'input_checks': [], 'scores': {}}

for case in cases:
    cid = case['instance_id']
    with tarfile.open(SOURCE / 'sources' / f'{cid}.tar.gz') as archive, tempfile.TemporaryDirectory() as td:
        members = {m.name.split('/', 1)[1]: m for m in archive.getmembers() if '/' in m.name and m.isfile()}
        for f in parse_patch_files(case['patch']):
            path = f['old_path']
            if not path.endswith('.py') or f.get('new_file'):
                continue
            dest = Path(td) / path
            if not dest.resolve().is_relative_to(Path(td).resolve()):
                raise ValueError('Unsafe patch path')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.extractfile(members[path]).read())
        try:
            mapped = symbols_from_patch(case['patch'], td, source_side='old')
            mismatch = mapped['functions'] != case['function_ground_truth']
            old_symbols = {}
            for path in {x.split('::')[0] for x in mapped['functions']}:
                file = Path(td) / path
                if not file.exists():
                    continue
                def walk(node, parents=()):
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            name = '.'.join((*parents, child.name))
                            old_symbols[f'{path}::{name}'] = 'class' if isinstance(child, ast.ClassDef) else 'function'
                            walk(child, (*parents, child.name))
                        else:
                            walk(child, parents)
                walk(ast.parse(file.read_text()))
            report['label_checks'].append({'instance_id': cid, 'mismatch': mismatch,
                'target_count': len(mapped['functions']),
                'class_targets': [t for t in mapped['functions'] if old_symbols.get(t) == 'class'],
                'absent_old_targets': [t for t in mapped['functions'] if t not in old_symbols],
                'issue_mentions_target_leaf_heuristic': any(
                    re.search(r'(?<!\w)' + re.escape(t.split('::')[-1].split('.')[-1]) + r'(?!\w)', case['problem_statement'])
                    for t in mapped['functions'] if not t.split('.')[-1].startswith('__'))})
        except ValueError as exc:
            report['label_checks'].append({'instance_id': cid, 'error': str(exc),
                                           'stored_targets': case['function_ground_truth']})

for row in rows:
    cid = row['instance_id']
    original = next(c for c in cases if c['instance_id'] == cid)
    for arm in ('no_skill', 'old_bank', 'expanded_bank'):
        payload = read(OUT / arm / 'cases' / cid / 'initial_payload.json')
        report['input_checks'].append({'instance_id': cid, 'arm': arm,
            'unexpected_keys': sorted(set(payload) - {'repo', 'base_commit', 'bug_report', 'fault_families'}),
            'issue_unchanged': payload['bug_report'] == original['problem_statement'].strip()})

for arm in ('no_skill', 'old_bank', 'expanded_bank'):
    results = []
    for row in rows:
        truth = set(row['function_ground_truth'])
        if not truth:
            continue
        preds = row['arms'][arm]['predictions']
        rank = next((i for i, p in enumerate(preds, 1) if p in truth), None)
        metrics = {**{f'top{k}': bool(rank and rank <= k) for k in (1, 3, 5)}, 'mrr': 1 / rank if rank else 0}
        results.append({'instance_id': row['instance_id'], 'metrics': metrics,
            'differs': metrics != {k: row['arms'][arm]['metrics'][k] for k in metrics},
            'target_count': len(truth), 'recall5': len(set(preds[:5]) & truth) / len(truth),
            'all_targets5': truth.issubset(set(preds[:5]))})
    report['scores'][arm] = {'n': len(results),
        **{k: sum(r['metrics'][k] for r in results) / len(results) for k in ('top1', 'top3', 'top5', 'mrr')},
        'literal_vs_saved_differences': [r['instance_id'] for r in results if r['differs']],
        'macro_recall5': sum(r['recall5'] for r in results) / len(results),
        'all_targets5_rate': sum(r['all_targets5'] for r in results) / len(results)}
report['target_count_distribution'] = dict(Counter(len(c['function_ground_truth']) for c in cases))
dest = OUT / 'metric_audit_snapshot.json'
dest.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ('label_checks', 'input_checks')}, indent=2))
print('label errors:', [r for r in report['label_checks'] if r.get('error') or r.get('mismatch') or r.get('class_targets')])
print('input anomalies:', [r for r in report['input_checks'] if r['unexpected_keys'] or not r['issue_unchanged']])
print('issue target-leaf matches:', sum(bool(r.get('issue_mentions_target_leaf_heuristic')) for r in report['label_checks']))
