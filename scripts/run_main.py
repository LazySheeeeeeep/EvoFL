"""Main experiment: serial Skill evolution and parallel fault localization.

The runner performs the chronological acquisition, evidence-driven Skill
evolution, and frozen-bank evaluation of the proposed method. All credentials
must be supplied through the process environment.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evolutefl.evaluation.patch_ground_truth import (  # noqa: E402
    normalize_python_module_identity,
    parse_patch_files,
    python_symbol_spans,
    split_function_identity,
    symbols_from_patch,
)
from evolutefl.explorer import run_explorer
from evolutefl.llm.client import OpenAICompatibleClient
from evolutefl.reflection import run_case_evolution
from evolutefl.skills import make_skill_bank

ORIGINAL = ROOT / "runs/rq1_temporal_deepseek_20260915"
AUDIT_CACHE = ROOT / "runs/rq1_expanded400_eval500_deepseek_20260922"
OUT = ROOT / "runs/main_experiment"
ARMS = ('with_skill',)
POLICY = 'changed_patch_functions_intersect_existing_base_functions_v1'


def main_llm_config(config):
    llm = dict(config.get('llm') or {})
    llm['base_url'] = (
        os.getenv('OPENAI_BASE_URL')
        or llm.get('base_url')
        or 'https://api.openai.com/v1'
    )
    llm['api_key_env'] = 'OPENAI_API_KEY'
    llm['model'] = (
        os.getenv('EVOLUTEFL_MODEL')
        or os.getenv('OPENAI_MODEL')
        or llm.get('model')
        or 'gpt-4o-mini'
    )
    supports_tool_choice = os.getenv('EVOLUTEFL_SUPPORTS_TOOL_CHOICE')
    if supports_tool_choice is None:
        llm['supports_tool_choice'] = bool(llm.get('supports_tool_choice', True))
    else:
        llm['supports_tool_choice'] = supports_tool_choice.strip().lower() not in {
            '0',
            'false',
            'no',
            'off',
        }
    llm['temperature'] = 0
    return llm


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.replace('\r\n', '\n').strip().encode()).hexdigest()


def visible_issue_groups(case, metadata):
    body = re.sub(r'<!--.*?-->', '', metadata.get('body') or '', flags=re.S)
    links = re.findall(r'https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)', body)
    local = re.findall(r'(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)', body)
    return {f'{repo}#{number}' for repo, number in links} | {
        case['repo'] + '#' + number for number in local
    }


def temporal_train_ok(metadata):
    cutoff = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    return all(
        metadata.get(key)
        and dt.datetime.fromisoformat(metadata[key].replace('Z', '+00:00')) < cutoff
        for key in ('created_at', 'merged_at')
    )


def safe_source_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or '\\' in relative:
        raise ValueError('Unsafe patch path')
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Patch path escaped audit cache')
    return target


def verify_original():
    root = ROOT.resolve()
    for path, expected in read(ORIGINAL / 'protocol.json')['hashes'].items():
        frozen = Path(path)
        if not frozen.exists():
            # The historical protocol hashed helper scripts that are now
            # consolidated into run_main.py.
            continue
        try:
            relative = frozen.resolve().relative_to(root)
        except ValueError:
            relative = None
        if relative is not None and relative.parts and relative.parts[0] in {'src', 'scripts'}:
            # Source hashes belong to the historical implementation. The new
            # main runner creates its own protocol after preparation.
            continue
        if sha(frozen) != expected:
            raise ValueError('Frozen historical input changed: ' + path)


def clean_workspace(path):
    root = (OUT / 'work').resolve()
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root) or resolved == root:
        raise ValueError('Unsafe workspace cleanup target')
    if path.exists():
        shutil.rmtree(path)


def safe_archive_filter(member, destination):
    if not member.issym():
        return tarfile.data_filter(member, destination)
    if os.path.isabs(member.linkname):
        raise tarfile.AbsoluteLinkError(member)
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


def download_github_archive(url, destination):
    curl = shutil.which('curl')
    if curl:
        subprocess.run(
            [
                curl,
                '--fail',
                '--location',
                '--http1.1',
                '--ipv4',
                '--connect-timeout',
                '15',
                '--max-time',
                '600',
                '--retry',
                '2',
                '--retry-all-errors',
                '--retry-delay',
                '5',
                '--output',
                str(destination),
                url,
            ],
            check=True,
            timeout=1830,
        )
        return
    with urllib.request.urlopen(url, timeout=180) as response:
        destination.write_bytes(response.read())


def unpack(case, workspace):
    clean_workspace(workspace)
    workspace.mkdir(parents=True)
    archive = OUT / 'sources' / (case['instance_id'] + '.tar.gz')
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temp = archive.with_suffix('.download')
        download_github_archive(
            f"https://codeload.github.com/{case['repo']}/tar.gz/{case['base_commit']}",
            temp,
        )
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
    for index, prediction in enumerate(predictions, 1):
        path, name = split_function_identity(prediction)
        for target in truth:
            target_path, target_name = split_function_identity(target)
            predicted_path, predicted_name = normalize_python_module_identity(
                prediction,
                path,
                name.replace('::', '.'),
                target_path,
            )
            if predicted_path == target_path and predicted_name == target_name:
                rank = index
                break
        if rank:
            break
    return {
        'rank': rank,
        **{f'top{k}': bool(rank and rank <= k) for k in (1, 3, 5)},
        'mrr': 1 / rank if rank else 0,
    }


class Audit:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.token = os.getenv('GITHUB_TOKEN') or os.getenv('GH_TOKEN')

    def get(self, url):
        headers = {'Accept': 'application/vnd.github+json'} if url.startswith('https://api.github.com/') else {}
        if self.token and url.startswith('https://api.github.com/'):
            headers['Authorization'] = 'Bearer ' + self.token
        for attempt in range(4):
            response = self.session.get(url, headers=headers, timeout=(15, 90))
            if response.status_code in (403, 429) and (
                response.headers.get('X-RateLimit-Remaining') == '0'
                or response.headers.get('Retry-After')
            ):
                wait = max(
                    60,
                    min(
                        3700,
                        int(response.headers.get('X-RateLimit-Reset', time.time() + 3600))
                        - int(time.time())
                        + 5,
                    ),
                )
                time.sleep(wait)
                continue
            if response.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        raise RuntimeError('GitHub bounded retry exhausted')

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
            raw = self.get(
                f"https://api.github.com/repos/{case['repo']}/pulls/{number}"
            ).json()
            result = {
                key: raw.get(key)
                for key in (
                    'number',
                    'html_url',
                    'created_at',
                    'merged_at',
                    'merge_commit_sha',
                    'body',
                )
            }
            source = {'url': result['html_url'], 'fetched_at': time.time()}
        result = {
            **result,
            'issue_groups': sorted(visible_issue_groups(case, result)),
            'audit_source': source,
            'issue_parser': 'visible_PR_body_without_HTML_comments_v1',
        }
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
            url = (
                f"https://raw.githubusercontent.com/{case['repo']}/"
                f"{case['base_commit']}/{quote(path, safe='/')}"
            )
            if not destination.exists():
                data = self.get(url).content
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_suffix(destination.suffix + '.download')
                temp.write_bytes(data)
                temp.replace(destination)
            sources.append(
                {
                    'path': path,
                    'url': url,
                    'cache_path': str(destination),
                    'sha256': sha(destination),
                }
            )
            for symbol in python_symbol_spans(destination):
                if symbol['kind'] == 'function':
                    existing.add(f"{path}::{symbol['qualified_name']}")
        mapped = symbols_from_patch(case['patch'], root, source_side='old')
        functions = mapped['functions']
        result = {
            'instance_id': cid,
            'repo': case['repo'],
            'base_commit': case['base_commit'],
            'patch_sha256': digest(case['patch']),
            'sources': sources,
            'functions': functions,
            'classes': mapped['classes'],
            'existing_function_targets': [f for f in functions if f in existing],
            'new_only_function_targets': [f for f in functions if f not in existing],
            'mapping_evidence': mapped['evidence'],
            'policy': 'unchanged existing mapper; changed lines on both sides; new-only targets flagged separately',
        }
        write(target, result)
        return result


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
                row['arms'][arm] = strict_metrics(list(dict.fromkeys(old['predictions']))[:5], truth)
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
    cfg['llm'] = main_llm_config(cfg)
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
    code = [*sorted((ROOT / 'src').rglob('*.py')), Path(__file__)]
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
    if shutil.disk_usage(OUT).free < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB free; no automatic artifact deletion')
    cid = case['instance_id']
    archive = OUT / 'sources' / (cid + '.tar.gz')
    previous = ORIGINAL / 'sources' / archive.name
    if not archive.exists() and previous.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(previous, archive)
    workspace = OUT / 'work' / cid
    root, archive_sha = unpack(case, workspace)
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
            'metrics': strict_metrics(result.get('ranked_functions', [])[:5], case['function_ground_truth'])}
        write(directory / 'record.json', record)
        bank = next_bank
        rows.append(record)
        write(OUT / 'training_summary.json', {'historical_bootstrap': 200, 'new_completed': len(rows), 'cases': rows})
        if len(rows) == 100:
            shutil.copyfile(bank, OUT / 'skills_after300.jsonl')
        clean_workspace(workspace)
        if index <= 2 and (result.get('status') != 'completed' or summary['status'] == 'failed'):
            raise RuntimeError('Training smoke protocol failed; no outcome-based replacement')
    frozen = OUT / 'frozen_skills.jsonl'
    if frozen.exists() and sha(frozen) != sha(bank):
        raise ValueError('Frozen bank differs from completed chain')
    shutil.copyfile(bank, frozen)
    write(OUT / 'training_complete.json', {'count': 200 + len(rows), 'bank_sha256': sha(frozen)})


def evaluate_case(job):
    """Each process evaluates one case; globals/workspaces are never shared."""
    index, case = job
    import fcntl
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
        row = {'instance_id': cid, 'repo': case['repo'], 'functions': case['function_ground_truth'],
               'bank_sha256': bank_sha, 'label_policy': POLICY, 'arms': {}}
        for arm in ARMS:
            write(OUT / 'jobs' / (cid + '.json'), {'phase': arm, 'instance_id': cid, 'time': time.time()})
            workspace, root = materialize(case)
            result = explorer(case, config, root, OUT / arm / 'cases' / cid, bank)
            predictions = result.get('ranked_functions', [])
            predictions = list(dict.fromkeys(predictions))[:5]
            search = read(OUT / arm / 'cases' / cid / 'fault_skill_search.json', {})
            row['arms'][arm] = {'status': result.get('status'), 'predictions': predictions,
                'metrics': strict_metrics(predictions, case['function_ground_truth']),
                'loaded_skill_id': (search.get('matched_skill') or {}).get('skill_id'),
                'forced_finish': result.get('forced_finish'), 'steps': result.get('steps'),
                'runtime_seconds': result.get('runtime_seconds')}
        if sha(bank) != bank_sha:
            raise ValueError('Bank changed during evaluation')
        write(finished, row)
        clean_workspace(workspace)
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
                configured_llm = (read(OUT / 'config.json') or {}).get('llm') or {}
                api_key_env = configured_llm.get('api_key_env') or 'OPENAI_API_KEY'
                if not os.getenv(api_key_env):
                    raise ValueError('Missing process credential')
                train()
                evaluate()
        except Exception as exc:
            write(OUT / 'failure.json', {'error_type': type(exc).__name__, 'error': str(exc)[:500],
                  'time': time.time(), 'progress': read(OUT / 'progress.json')})
            raise


if __name__ == '__main__':
    main()
