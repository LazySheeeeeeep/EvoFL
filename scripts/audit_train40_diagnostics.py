"""Read-only diagnostics of issue clues, exact scoring and failure reflection."""
import json
import re
from collections import Counter
from pathlib import Path

from audit_v5_accuracy import rank

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v5_train40_deepseek_v4_flash_20260910_continuation'


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    records = read(OUT / 'summary.json')['cases']
    selected = {c['instance_id']: c for c in read(OUT / 'selected_cases.json')}
    rows = []
    for c in records:
        cid = c['instance_id']
        d = OUT / 'cases' / cid
        issue = selected[cid]['problem_statement']
        gt = c.get('ground_truth_functions', [])
        named = [g for g in gt if re.search(r'(?<![\w])' + re.escape(g.split('::')[-1].split('.')[-1]) + r'(?![\w])', issue)]
        initial = read(d / 'initial_payload.json')
        search = read(d / 'fault_skill_search.json')
        conclusion = read(d / 'case_evolution/investigation_conclusion.json')
        update = read(d / 'case_evolution/fault_reflector_output.json').get('skill_updates', {}).get('fault_skill', {})
        obs_path = d / 'observations.jsonl'
        obs_text = obs_path.read_text() if obs_path.exists() else ''
        match = re.search(r'(func_[^.]+|lm_rewrite|combine_module|pr_\d+)', cid)
        rows.append(dict(instance_id=cid, issue=issue, ground_truth=gt,
            predictions=c.get('ranked_functions', []), reported_rank=c.get('function_metrics', {}).get('rank'),
            exact_rank=rank(c.get('ranked_functions', []), gt), named_gt_functions=named,
            initial_keys=list(initial), issue_unchanged=initial.get('bug_report', '').strip() == issue.strip(),
            mutation_kind=match.group(0).split('__')[0] if match else 'other',
            initial_has_gt_fields=bool({'patch', 'ground_truth_functions', 'instance_id'} & set(initial)),
            metadata_marker_observed='.evolutefl_swesmith_case.json' in obs_text,
            loaded_skill_id=(search.get('matched_skill') or {}).get('skill_id'),
            evolution_status=c.get('evolution_status'), conclusion=conclusion,
            update=update))
    eligible = [r for r in rows if r['ground_truth']]
    def stats(group):
        return dict(n=len(group), top1=sum(r['exact_rank'] == 1 for r in group),
                    top5=sum(r['exact_rank'] is not None and r['exact_rank'] <= 5 for r in group))
    report = dict(total=len(rows), eligible=len(eligible),
        exact_matching_changes=[r['instance_id'] for r in eligible if r['exact_rank'] != r['reported_rank']],
        named_gt_issue=stats([r for r in eligible if r['named_gt_functions']]),
        no_named_gt_issue=stats([r for r in eligible if not r['named_gt_functions']]),
        loaded=stats([r for r in eligible if r['loaded_skill_id']]),
        not_loaded=stats([r for r in eligible if not r['loaded_skill_id']]),
        mutation_kinds=dict(Counter(r['mutation_kind'] for r in rows)),
        leaked_initial_fields=[r['instance_id'] for r in rows if r['initial_has_gt_fields']],
        marker_in_observations=[r['instance_id'] for r in rows if r['metadata_marker_observed']],
        issue_changed=[r['instance_id'] for r in rows if r['initial_keys'] and not r['issue_unchanged']],
        cases=rows)
    (OUT / 'diagnostic_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2))
    print('NAMED SUCCESS EXAMPLES')
    for row in [r for r in eligible if r['named_gt_functions'] and r['exact_rank'] == 1][:4]:
        print(row['instance_id'], row['issue'][:1600])


if __name__ == '__main__':
    main()
