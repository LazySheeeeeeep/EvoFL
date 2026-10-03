"""Count cached benchmark eligibility without materializing repos or calling LLMs."""
import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def main():
    explore = list(load_dataset('SWE-Explore-Bench/SWE-Explore-Bench', split='train'))
    verified = {r['instance_id']: r for r in load_dataset('princeton-nlp/SWE-bench_Verified', split='test')}
    print('COLUMNS', list(explore[0]))
    print('FIRST_METADATA', json.dumps({k:v for k,v in explore[0].items() if k not in ('read_step_info', 'ground_truth')}, ensure_ascii=False)[:2500])
    print('GROUND_TRUTH_EXAMPLE', json.dumps(explore[0].get('ground_truth'), ensure_ascii=False)[:1000])
    candidates = [r for r in explore if r.get('dataset') == 'verified'
                  and r['instance_id'] in verified and (r.get('ground_truth') or {}).get('read_core_regions')]
    ready = [r for r in candidates if all(str(verified[r['instance_id']].get(k) or '').strip()
                                         for k in ('problem_statement', 'base_commit', 'repo', 'patch'))]
    seen = set()
    paths = []
    for directory, dirs, files in os.walk('runs'):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                   {'repos', 'repos_base', 'repos_work', 'cases', 'resume_attempts'}]
        if 'selected_cases.json' in files:
            paths.append(Path(directory) / 'selected_cases.json')
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError):
            continue
        if isinstance(data, list):
            seen.update(r['instance_id'] for r in data if isinstance(r, dict) and r.get('instance_id'))
    report = dict(total=len(explore), unique_ids=len({r['instance_id'] for r in explore}),
                  dataset_distribution=dict(Counter(r.get('dataset') for r in explore)),
                  current_runner_candidates=len(candidates), with_required_metadata=len(ready),
                  repo_distribution=dict(Counter(verified[r['instance_id']]['repo'] for r in ready)),
                  previously_selected=sum(r['instance_id'] in seen for r in ready),
                  not_in_retained_manifests=sum(r['instance_id'] not in seen for r in ready),
                  note='Metadata eligibility only; source AST/function ground truth and snapshot materialization not checked.')
    out = Path('runs/swe_explore_eligibility_audit')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
