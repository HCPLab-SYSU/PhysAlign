#!/usr/bin/env python3
"""Portable DATA converter. No API keys, model clients, evaluation or scoring."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from adapter import SourceAdapter, verify_adapter_snapshot
from compiler import Compiler
from io_utils import (read_json, write_json, safe_path, empty_ledger, seal_directory,
                      verify_directory, review_record, sha_bytes)
from reference_checks import require, digest
from reference_integrity import ReviewLedger
from reference_registry import validate_registry, entry_root
from reference_pipeline import (config, dependency_hashes, validate_frozen,
                                validate_release_instances, prepare_probe)
from review_workflow import prospective, apply_decision, write_review_page, CONFIRMATIONS

ROOT = Path(__file__).resolve().parent
VERSION = 'physalign_converter_v1.2.0_native_draft'


def new_output(path):
    final = Path(path).resolve()
    require(not final.exists(), 'OUTPUT_EXISTS_USE_NEW_DIRECTORY', str(final))
    final.parent.mkdir(parents=True, exist_ok=True)
    # Keep interrupted staging for diagnosis. Never recursively delete user data.
    staging = Path(tempfile.mkdtemp(prefix='.' + final.name + '.staging-', dir=final.parent))
    return final, staging


def publish(staging, final):
    require(not final.exists(), 'OUTPUT_EXISTS_USE_NEW_DIRECTORY')
    seal_directory(staging)
    staging.rename(final)


def selected_ids(adapter, args):
    ids = args.problem_id or sorted(adapter.by_id)
    if args.ids_file:
        require(not args.problem_id, 'CHOOSE_IDS_FILE_OR_PROBLEM_IDS')
        p = Path(args.ids_file)
        if p.suffix.lower() == '.json':
            ids = read_json(p)
            if isinstance(ids, dict):
                ids = ids.get('problem_ids')
            require(isinstance(ids, list) and all(isinstance(i, str) for i in ids), 'IDS_JSON_EXPECTS_STRING_ARRAY_OR_PROBLEM_IDS')
        else:
            ids = [l.strip() for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    require(len(set(ids)) == len(ids), 'DUPLICATE_SELECTED_ID')
    require(all(pid in adapter.by_id for pid in ids), 'SELECTED_ID_NOT_IN_MANIFEST')
    return ids[:args.limit] if args.limit is not None else ids


def source_args(parser):
    parser.add_argument('--workspace', required=True, help='PhysGraph annotation workspace (contains passes/, blind/, reviews/)')
    parser.add_argument('--dataset-root', help='override relocated original SFT image root')
    parser.add_argument('--problem-id', action='append')
    parser.add_argument('--ids-file', help='newline IDs or JSON array / {problem_ids: [...]}')
    parser.add_argument('--limit', type=positive_int)


def positive_int(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def inventory(args):
    adapter = SourceAdapter(args.workspace, args.dataset_root)
    report = adapter.inventory(selected_ids(adapter, args))
    report['implementation_hashes'] = dependency_hashes()
    final, stage = new_output(args.output)
    write_json(stage / 'inventory.json', report)
    vocabulary = {'roles': {}, 'relations': {}}
    for item in report['problems']:
        if item['state'] != 'source_ready':
            continue
        loaded = adapter.load(item['problem_id'])
        doc = loaded['docs']['pass4']
        for collection, field, slot in (('physical_nodes', 'type', 'roles'), ('relations', 'predicate', 'relations')):
            for r in doc[collection]:
                token = loaded['problem']['source_dataset'] + '::' + r[field]
                entry = vocabulary[slot].setdefault(token, {'count': 0, 'examples': []})
                entry['count'] += 1
                if len(entry['examples']) < 3:
                    entry['examples'].append({'problem_id': item['problem_id'], 'record_id': r['id']})
    write_json(stage / 'semantic_vocabulary.PRIVATE.json', vocabulary)
    publish(stage, final)
    print(json.dumps({'output': str(final), 'counts': report['counts']}, ensure_ascii=False))


def draft(args):
    adapter = SourceAdapter(args.workspace, args.dataset_root)
    ids = selected_ids(adapter, args)
    registry = read_json(args.registry) if args.registry else config('semantic_registry.json')
    ledger = read_json(args.registry_ledger) if args.registry_ledger else empty_ledger()
    # Validate per source index below if source-backed semantic examples exist.
    final, stage = new_output(args.output)
    catalog = {'schema_version': 'physalign_draft_catalog_v1', 'converter_version': VERSION,
               'implementation_hashes': dependency_hashes(), 'tasks_requested': args.tasks.split(','),
               'status': 'pending_benchmark_QA_review', 'probes': [], 'problems': [],
               'registry_hash': digest(registry), 'partition_policy': 'unassigned_team_dedup_required'}
    write_json(stage / 'registry.json', registry)
    write_json(stage / 'registry_ledger.json', ledger)
    for n, pid in enumerate(ids, 1):
        blocked, count = [], 0
        asset_rel = 'problems/' + pid
        try:
            loaded = adapter.load(pid)
            adapted = adapter.materialize(loaded, stage / asset_rel)
            compiler = Compiler(adapted, registry, ledger)
            for task in catalog['tasks_requested']:
                try:
                    proposals = compiler.proposals(task)
                except (ValueError, KeyError, TypeError) as exc:
                    blocked.append(error_record(exc, task, []))
                    continue
                for proposal in proposals:
                    scope = sorted({r['ref']['record_id'] for r in proposal['rows']})
                    try:
                        d, env, preview = compiler.compile(task, proposal, args.max_views)
                        candidate, target = prospective(d, env)
                        qid = d['key']['logical_probe_id']
                        require(not any(i['logical_probe_id'] == qid for i in catalog['probes']), 'DUPLICATE_LOGICAL_ID')
                        rel = 'probes/' + qid
                        serial_env = copy.deepcopy(env)
                        serial_env['asset_root'] = '.'  # Actual root comes only from the confined catalog descriptor.
                        write_json(stage / rel / 'draft.json', d)
                        write_json(stage / rel / 'environment.PRIVATE.json', serial_env)
                        write_json(stage / rel / 'preview.PRIVATE.json', preview)
                        item = {'logical_probe_id': qid, 'problem_id': pid, 'task_id': task,
                                'language': d['key']['language'], 'group_id': adapted['audit']['group_id'],
                                'content_group_hint': adapted['audit']['content_group_hint'],
                                'asset_root': asset_rel, 'draft': rel + '/draft.json',
                                'environment': rel + '/environment.PRIVATE.json', 'preview': rel + '/preview.PRIVATE.json',
                                'review_target_root': target['content_root'], 'source_scope': scope}
                        catalog['probes'].append(item)
                        count += 1
                    except (ValueError, KeyError, TypeError) as exc:
                        blocked.append(error_record(exc, task, scope))
            blocked.extend(compiler.blocked)
            info = {'problem_id': pid, 'source_state': 'source_ready', 'draft_probes': count,
                    'blocked_queries': blocked, 'source_audit': asset_rel + '/source_audit.json'}
        except (ValueError, OSError, KeyError, TypeError) as exc:
            info = {'problem_id': pid, 'source_state': 'blocked', 'draft_probes': 0,
                    'blocked_queries': [error_record(exc, 'source', [])]}
        catalog['problems'].append(info)
        print(f'[{n}/{len(ids)}] {pid}: {count} draft QA; {len(info["blocked_queries"])} blocked', flush=True)
    catalog['counts'] = {'problems': len(ids), 'draft_probes': len(catalog['probes']),
                         'by_task': dict(Counter(i['task_id'] for i in catalog['probes'])),
                         'blocked_reasons': dict(Counter(b['code'] for p in catalog['problems'] for b in p['blocked_queries']))}
    write_json(stage / 'catalog.PRIVATE.json', catalog)
    template = {'schema_version': 'physalign_decisions_v1', 'catalog_root': digest(catalog),
                'decisions': [{'logical_probe_id': i['logical_probe_id'], 'review_target_root': i['review_target_root'],
                    'status': 'pending', 'reviewer': '', 'blind_answer': '', 'note': '',
                    'confirmations': {k: False for k in CONFIRMATIONS}} for i in catalog['probes']]}
    write_json(stage / 'decisions.template.json', template)
    write_review_page(stage, catalog)
    write_json(stage / 'report.json', {'counts': catalog['counts'], 'problems': catalog['problems']})
    publish(stage, final)
    print(json.dumps({'output': str(final), **catalog['counts'], 'next': 'Open review.html locally; export decisions outside this immutable draft directory.'}, ensure_ascii=False))


def error_record(exc, task, scope):
    return {'task_id': task, 'scope': scope, 'code': getattr(exc, 'code', type(exc).__name__), 'detail': str(exc)}


def load_probe(root, item):
    d = read_json(safe_path(root, item['draft']))
    env = read_json(safe_path(root, item['environment']))
    require(env['asset_root'] == '.', 'STORED_ENV_ROOT_MUST_BE_RELATIVE')
    env['asset_root'] = str(safe_path(root, item['asset_root']))
    return d, env


def verify_descriptor(item, key, rebuilt):
    audit = rebuilt['audit']
    for field in ('problem_id', 'group_id', 'content_group_hint'):
        require(item[field] == audit[field], 'DESCRIPTOR_SOURCE_MISMATCH', field)
    for field in ('logical_probe_id', 'problem_id', 'task_id', 'language'):
        require(item[field] == key[field], 'DESCRIPTOR_KEY_MISMATCH', field)


def freeze(args):
    source = Path(args.draft).resolve()
    verify_directory(source)
    catalog = read_json(source / 'catalog.PRIVATE.json')
    require(catalog['implementation_hashes'] == dependency_hashes(), 'CONVERTER_CHANGED_REGENERATE_DRAFT')
    decisions = read_json(args.decisions)
    require(set(decisions) == {'schema_version', 'catalog_root', 'decisions'} and
            decisions['schema_version'] == 'physalign_decisions_v1', 'DECISION_FILE_SCHEMA')
    require(decisions['catalog_root'] == digest(catalog), 'DECISION_CATALOG_STALE')
    mapping = {}
    known = {i['logical_probe_id'] for i in catalog['probes']}
    for d in decisions['decisions']:
        qid = d['logical_probe_id']
        require(qid in known and qid not in mapping, 'UNKNOWN_OR_DUPLICATE_DECISION')
        require(d['status'] in ('approved', 'rejected', 'pending'), 'DECISION_STATUS_INVALID')
        mapping[qid] = d
    selected = [i for i in catalog['probes'] if mapping.get(i['logical_probe_id'], {}).get('status') == 'approved']
    require(selected, 'NO_EXPLICITLY_APPROVED_QA')
    final, stage = new_output(args.output)
    all_frozen, items, copied = [], [], set()
    for n, item in enumerate(selected, 1):
        d, env = load_probe(source, item)
        rebuilt = verify_adapter_snapshot(env)
        verify_descriptor(item, d['key'], rebuilt)
        decision = mapping[item['logical_probe_id']]
        frozen, final_env = apply_decision(d, env, decision)
        require(frozen['private']['content_root'] == item['review_target_root'], 'CATALOG_REVIEW_ROOT_MISMATCH')
        # Copy only required original snapshots, audit and actual declared assets.
        rels = {r['relative_path'] for r in final_env['source_files'].values()}
        rels.update(a['relative_path'] for a in final_env['assets'].values())
        rels.add('source_audit.json')
        for rel in sorted(rels):
            dest_rel = item['asset_root'] + '/' + rel
            dest = safe_path(stage, dest_rel)
            src = safe_path(final_env['asset_root'], rel)
            if dest_rel not in copied:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                copied.add(dest_rel)
            else:
                require(sha_bytes(src.read_bytes()) == sha_bytes(dest.read_bytes()), 'COPY_DESTINATION_COLLISION')
        final_env['asset_root'] = str(stage / item['asset_root'])
        validate_frozen(frozen, final_env)
        qid = item['logical_probe_id']
        rel = 'private/' + qid
        stored_env = copy.deepcopy(final_env)
        stored_env['asset_root'] = '.'
        write_json(stage / rel / 'environment.json', stored_env)
        write_json(stage / rel / 'frozen.json', frozen)
        write_json(stage / rel / 'decision.json', decision)
        public = {}
        for condition in ('raw', 'gold'):
            inp = frozen['core']['public_' + condition]
            attachments = [{'asset_id': aid, 'path': item['asset_root'] + '/' + final_env['assets'][aid]['relative_path'],
                            'bytes_sha256': final_env['assets'][aid]['bytes_sha256']} for aid in inp['attachment_ids']]
            record = {'envelope': frozen['envelope'], 'task_id': item['task_id'], 'condition': condition,
                      'input': {'messages': inp['messages'], 'attachments': attachments}}
            path = 'public/' + condition + '/' + qid + '.json'
            write_json(stage / path, record)
            public[condition] = path
        items.append({k: item[k] for k in ('logical_probe_id', 'problem_id', 'task_id', 'language', 'group_id', 'content_group_hint', 'asset_root')}
                     | {'instance_id': frozen['envelope']['instance_id'], 'frozen': rel + '/frozen.json',
                        'environment': rel + '/environment.json', 'decision': rel + '/decision.json', 'public': public})
        all_frozen.append(frozen)
        print(f'[{n}/{len(selected)}] frozen {qid}', flush=True)
    validate_release_instances(all_frozen)
    manifest = {'schema_version': 'physalign_data_release_v1', 'converter_version': VERSION,
                'implementation_hashes': dependency_hashes(), 'source_catalog_root': digest(catalog),
                'partition_policy': 'unassigned_team_dedup_required', 'instances': items,
                'counts': {'probes': len(items), 'problems': len({i['group_id'] for i in items}),
                           'by_task': dict(Counter(i['task_id'] for i in items)),
                           'unapproved_not_exported': len(known) - len(items)}}
    manifest['release_id'] = 'PAR-' + digest(manifest)
    write_json(stage / 'release.json', manifest)
    write_json(stage / 'private/review_decisions.json', decisions)
    write_json(stage / 'private/source_catalog.json', catalog)
    # Convenient paired QA manifests; attachments stay relative to release root.
    for condition in ('raw', 'gold'):
        write_json(stage / ('qa_' + condition + '.json'), [read_json(stage / i['public'][condition]) for i in items])
    write_json(stage / 'private/mappings.json', [f['private'] for f in all_frozen])
    publish(stage, final)
    print(json.dumps({'output': str(final), 'release_id': manifest['release_id'], 'counts': manifest['counts']}, ensure_ascii=False))


def validate(args):
    root = Path(args.path).resolve()
    verify_directory(root)
    if (root / 'catalog.PRIVATE.json').exists():
        catalog = read_json(root / 'catalog.PRIVATE.json')
        if 'native_profile' in catalog:
            from native_run import validate_native_output
            return validate_native_output(args)
        require(catalog['implementation_hashes'] == dependency_hashes(), 'CONVERTER_VERSION_MISMATCH')
        for item in catalog['probes']:
            d, env = load_probe(root, item)
            rebuilt = verify_adapter_snapshot(env)
            verify_descriptor(item, d['key'], rebuilt)
            p = prepare_probe(d, env, verify_reviews=False)
            require(p == read_json(safe_path(root, item['preview'])), 'DRAFT_PREVIEW_MISMATCH')
            require(prospective(d, env)[1]['content_root'] == item['review_target_root'], 'DRAFT_REVIEW_ROOT_MISMATCH')
        print(json.dumps({'valid': True, 'kind': 'draft_not_approved', 'probes': len(catalog['probes'])}))
        return
    release = read_json(root / 'release.json')
    require(release['implementation_hashes'] == dependency_hashes(), 'CONVERTER_VERSION_MISMATCH')
    core = {k: v for k, v in release.items() if k != 'release_id'}
    require(release['release_id'] == 'PAR-' + digest(core), 'RELEASE_ID_MISMATCH')
    frozen_items = []
    for item in release['instances']:
        env = read_json(safe_path(root, item['environment']))
        require(env['asset_root'] == '.', 'STORED_ENV_ROOT_MUST_BE_RELATIVE')
        env['asset_root'] = str(safe_path(root, item['asset_root']))
        rebuilt = verify_adapter_snapshot(env)
        f = read_json(safe_path(root, item['frozen']))
        verify_descriptor(item, f['private'], rebuilt)
        require(item['instance_id'] == f['envelope']['instance_id'], 'DESCRIPTOR_INSTANCE_MISMATCH')
        validate_frozen(f, env)
        # The downloadable decision, private ledger and final frozen core must
        # agree, not merely have independent well-formed hashes.
        decision = read_json(safe_path(root, item['decision']))
        candidate = {'view': copy.deepcopy(f['core']['raw_view']), 'key': copy.deepcopy(f['core']['private_core'])}
        candidate['key']['packet_plan'].update(state='pending_review', review_ref=None, empty_reason=None)
        undecided_env = copy.deepcopy(env)
        for rid in ('qa-packet', 'qa-closure', 'qa-instance'):
            undecided_env['ledger']['records'].pop(rid, None)
        regenerated, checked_env = apply_decision(candidate, undecided_env, decision)
        require(regenerated == f and checked_env == env, 'EXPLICIT_DECISION_LEDGER_MISMATCH')
        frozen_items.append(f)
        for condition in ('raw', 'gold'):
            public = read_json(safe_path(root, item['public'][condition]))
            require(set(public) == {'envelope', 'task_id', 'condition', 'input'}, 'PUBLIC_OUTER_FIELDS')
            inp = f['core']['public_' + condition]
            expected = {'messages': inp['messages'], 'attachments': [{'asset_id': aid,
                        'path': item['asset_root'] + '/' + env['assets'][aid]['relative_path'],
                        'bytes_sha256': env['assets'][aid]['bytes_sha256']} for aid in inp['attachment_ids']]}
            require(public == {'envelope': f['envelope'], 'task_id': f['private']['task_id'], 'condition': condition,
                               'input': expected}, 'PUBLIC_EXPORT_MISMATCH')
    validate_release_instances(frozen_items)
    for condition in ('raw', 'gold'):
        require(read_json(root / ('qa_' + condition + '.json')) ==
                [read_json(safe_path(root, i['public'][condition])) for i in release['instances']], 'QA_AGGREGATE_MISMATCH')
    require(read_json(root / 'private/mappings.json') == [f['private'] for f in frozen_items], 'MAPPING_AGGREGATE_MISMATCH')
    print(json.dumps({'valid': True, 'kind': 'reviewed_data_release', 'probes': len(frozen_items)}))


def package(args):
    dest = Path(args.output).resolve()
    require(not dest.exists(), 'OUTPUT_EXISTS_USE_NEW_FILE')
    # Explicit code allowlist; never include datasets, credentials or external workspaces.
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.html')) + [ROOT/'README.md', ROOT/'requirements.txt']
    files += list((ROOT/'config').glob('*.json')) + list((ROOT/'schemas').glob('*.json'))
    files += list((ROOT/'tests').glob('*.py'))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, 'physalign_converter/' + path.relative_to(ROOT).as_posix())
    print(json.dumps({'output': str(dest), 'files': len(files), 'bytes_sha256': sha_bytes(dest.read_bytes())}))


def registry_review(args):
    registry = read_json(args.registry)
    require(registry['example_only'] is False, 'EXAMPLE_REGISTRY_NOT_FOR_RELEASE')
    ledger = read_json(args.ledger) if args.ledger else empty_ledger()
    wanted = set(args.approve)
    require(len(wanted) == len(args.approve), 'DUPLICATE_REGISTRY_APPROVAL')
    found = set()
    for kind, collection in (('role', 'roles'), ('relation', 'relations')):
        for entry in registry[collection]:
            token = kind + ':' + entry['canonical_id']
            if token not in wanted:
                continue
            entry['status'] = 'approved'
            root = entry_root(entry)
            rid = 'semantic-' + root
            entry['review_ref'] = rid
            ledger['records'][rid] = review_record(rid, 'semantic_entry', root, args.reviewer, [args.evidence])
            found.add(token)
    require(found == wanted, 'REGISTRY_APPROVAL_ENTRY_NOT_FOUND')
    validate_registry(registry, ReviewLedger(ledger, production=True), production=True)
    final, stage = new_output(args.output)
    write_json(stage / 'registry.json', registry)
    write_json(stage / 'ledger.json', ledger)
    publish(stage, final)
    print(json.dumps({'output': str(final), 'explicitly_approved': sorted(found)}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('inventory', help='read-only source readiness and exact vocabulary inventory')
    source_args(i); i.add_argument('--output', required=True); i.set_defaults(run=inventory)
    from native_run import native_draft, validate_native_output
    n = sub.add_parser('native-draft', help='native-only read-only inventory, closed groups, metadata and development review; NO release')
    source_args(n); n.add_argument('--output', required=True)
    n.add_argument('--span-policy-approval', help='explicit human approval JSON for the exact versioned span rule; default proposed only')
    n.add_argument('--development-count', type=positive_int, default=25)
    n.add_argument('--seed', type=int, default=2027)
    n.add_argument('--max-views', type=positive_int, default=128)
    n.set_defaults(run=native_draft)
    nv = sub.add_parser('native-validate', help='rebuild source, shadow closure, metadata and all draft previews offline')
    nv.add_argument('--path', required=True); nv.set_defaults(run=validate_native_output)
    d = sub.add_parser('draft', help='compile pending QA + local private review UI')
    source_args(d); d.add_argument('--output', required=True)
    d.add_argument('--tasks', default='T01,T02,T03,T04,T05')
    d.add_argument('--registry'); d.add_argument('--registry-ledger'); d.add_argument('--max-views', type=positive_int, default=128)
    d.set_defaults(run=draft)
    f = sub.add_parser('freeze', help='export ONLY explicitly approved QA to a new portable release')
    f.add_argument('--draft', required=True); f.add_argument('--decisions', required=True); f.add_argument('--output', required=True); f.set_defaults(run=freeze)
    v = sub.add_parser('validate', help='offline full data/source/review/hash validation, no model evaluation')
    v.add_argument('--path', required=True); v.set_defaults(run=validate)
    z = sub.add_parser('package', help='zip only converter code/config/tests for colleagues')
    z.add_argument('--output', required=True); z.set_defaults(run=package)
    r = sub.add_parser('registry-review', help='record an EXPLICIT human approval of named registry entries')
    r.add_argument('--registry', required=True); r.add_argument('--ledger'); r.add_argument('--approve', action='append', required=True)
    r.add_argument('--reviewer', required=True); r.add_argument('--evidence', required=True); r.add_argument('--output', required=True); r.set_defaults(run=registry_review)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'draft':
        tasks = args.tasks.split(',')
        require(len(tasks) == len(set(tasks)) and set(tasks) <= {'T01', 'T02', 'T03', 'T04', 'T05'}, 'UNSUPPORTED_OR_DUPLICATE_TASK')
    args.run(args)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
