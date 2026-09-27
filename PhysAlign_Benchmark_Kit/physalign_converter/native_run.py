"""Stage A only: immutable native draft, real yield, development candidates.

No model client, QA approval inheritance, runner, scorer or aggregate release.
"""
from __future__ import annotations
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from adapter import SourceAdapter, STAGES, COLLECTIONS, verify_adapter_snapshot
from io_utils import read_json, write_json, sha_bytes, empty_ledger, safe_path, verify_directory
from reference_checks import require, digest
from reference_pipeline import dependency_hashes, config, prepare_probe
from review_workflow import prospective, write_review_page, CONFIRMATIONS
from native_projection import PROFILE, SPAN_RULE, check_approval
from native_compiler import NativeCompiler, DISABLED
from native_diagnostics import describe_probe, chance_summary


def workspace_snapshot(root):
    """Integrity hashing only; Pass5 is never parsed or used as observed truth."""
    root = Path(root)
    files = [root / 'workspace_config.json', root / 'reviews/state.json', root / 'blind/manifest.jsonl']
    for stage in (*STAGES, 'pass5'):
        files.extend(sorted((root / 'passes' / stage).glob('*.json')))
    return {p.relative_to(root).as_posix(): sha_bytes(p.read_bytes()) for p in files}


def dedup_members(records):
    """Exact reviewed identity and exact effective text+pixels only.

    Different versions of one source identity block the entire identity group.
    Near duplicates are NOT claimed to be solved by an exact hash.
    """
    identity, content = defaultdict(list), defaultdict(list)
    for r in records:
        identity[r['source_identity']].append(r)
    conflicts = {r['problem_id'] for group in identity.values()
                 if len({r['content_group_hint'] for r in group}) > 1 for r in group}
    for r in records:
        if r['problem_id'] not in conflicts:
            content[r['content_group_hint']].append(r)
    for group in content.values():
        ids = sorted(r['problem_id'] for r in group)
        gid = 'ND-' + digest(ids)
        for r in group:
            r.update(dedup_group=gid, representative_id=ids[0],
                     dedup_state='representative' if r['problem_id'] == ids[0] else 'exact_duplicate',
                     split='candidate_unassigned_not_release')
    for r in records:
        if r['problem_id'] in conflicts:
            r.update(dedup_group='ND-conflict-' + r['source_identity'], representative_id=None,
                     dedup_state='identity_version_conflict', split='blocked')


def development_ids(probes, count, seed):
    """Deterministic coverage-oriented development selection, NOT random audit."""
    tags = defaultdict(set)
    for p in probes:
        m = p['metadata']
        tags[p['problem_id']].update((('interface', m['interface']), ('language', p['language']),
             ('source', m['source_namespace']), ('representation', m['representation_kind']),
             ('availability', m['public_text_availability']), ('images', 'multi' if m['image_count'] > 1 else 'single'),
             ('repair', str(m['repair_applied_to_query']))))
    selected, covered = [], set()
    while tags and len(selected) < count:
        pid = min(tags, key=lambda p: (-len(tags[p] - covered), digest([seed, 'native_development_v1', p])))
        covered.update(tags.pop(pid))
        selected.append(pid)
    return selected


def summarize(catalog, inventory, membership):
    probes = catalog['probes']
    interfaces = {}
    for name in ('T02-single', 'T03-image', 'T03-text'):
        items = [p for p in probes if p['metadata']['interface'] == name]
        groups = [g for p in catalog['problems'] for g in p.get('groups', []) if g.get('interface') == name]
        interfaces[name] = {'mothers': len({p['problem_id'] for p in items}),
                            'enumerated_groups': len(groups),
                            'saved_query_groups': len({p['metadata']['group_id'] for p in items}),
                            'probes': len(items), 'chance_mother_macro': chance_summary(items)}
    L = [p for p in probes if p['metadata']['L_candidate']]
    P = [p for p in probes if p['metadata']['packet_nonempty_candidate']]
    pids = {p['logical_probe_id'] for p in P}
    LP = [p for p in L if p['logical_probe_id'] in pids]
    def size(items):
        return {'probes': len(items), 'mothers': len({p['problem_id'] for p in items})}
    strata = {}
    for field in ('public_text_availability', 'representation_kind', 'candidate_count',
                  'same_native_type_candidate_count', 'native_types_homogeneous', 'image_count',
                  'source_language', 'source_namespace', 'source_domain', 'selector', 'packet_policy_id'):
        strata[field] = dict(Counter(str(p['metadata'][field]) for p in probes))
    strata['nearest_outcome'] = dict(Counter(p['metadata']['nearest_region']['outcome'] for p in probes))
    strata['L_text_availability'] = dict(Counter(p['metadata']['public_text_availability'] for p in L))
    blockers = [b for p in catalog['problems'] for b in p['blocked_queries']]
    span_entries = [e for p in catalog['problems'] for e in p.get('span_sidecar_entries', [])]
    return {'schema_version': 'native_dry_run_report_v1', 'status': 'DRAFT_NOT_BENCHMARK_RELEASE',
        'profile': PROFILE, 'inventory_counts': inventory['counts'],
        'membership_states': dict(Counter(r['dedup_state'] for r in membership['members'])),
        'interfaces': interfaces, 'B_candidate': size(probes), 'L_candidate': size(L),
        'P_nonempty_candidate_NOT_approved': size(P), 'L_intersect_P_candidate': size(LP),
        'L_mothers_intersect_P_candidate_mothers': len({p['problem_id'] for p in L} & {p['problem_id'] for p in P}),
        'P_approved': {'probes': 0, 'mothers': 0}, 'approved_release': {'probes': 0, 'mothers': 0},
        'evidence_groups': len({p['metadata']['evidence_group'] for p in probes}), 'strata': strata,
        'blocked_primary_reasons': dict(Counter(b['code'] for b in blockers)),
        'blocked_all_reasons': dict(Counter(r for b in blockers for r in b.get('reasons', [b['code']]))),
        'policy_disabled': {task: {'mothers': len(catalog['problems']), 'code': 'policy_disabled',
                                  'not_annotation_error': True} for task in DISABLED},
        'span_sidecar_states': dict(Counter(e['state'] for e in span_entries)),
        'span_proposed_mothers': len({p['problem_id'] for p in catalog['problems'] if p.get('span_sidecar_entries')}),
        'span_actual_recovered_probe_count': sum(p['metadata']['repair_applied_to_query'] for p in probes),
        'span_counterfactual_yield': 'not_measured_without_project_rule_approval',
        'development_mothers': len(membership['development_problem_ids']),
        'limitations': ['all_QA_and_packets_pending', 'L_means_image_read_origin_not_vision_required',
                       'exact_dedup_only_near_duplicate_team_review_pending',
                       'candidate_split_is_not_final_test_release', 'provisional_diagram_asset_role_needs_development_check',
                       'no_model_calls_or_empirical_model_scores', 'no_rule_inheritance_or_batch_audit']}


def native_draft(args):
    from convert import new_output, publish, selected_ids, error_record, VERSION
    require(20 <= args.development_count <= 30, 'DEVELOPMENT_COUNT_MUST_BE_20_TO_30')
    adapter = SourceAdapter(args.workspace, args.dataset_root)
    ids = selected_ids(adapter, args)
    before = workspace_snapshot(adapter.root)
    approval = read_json(args.span_policy_approval) if args.span_policy_approval else None
    check_approval(approval)
    final, stage = new_output(args.output)
    write_json(stage / 'source_integrity.before.json', before)
    write_json(stage / 'span_rule.json', SPAN_RULE)
    write_json(stage / 'span_rule_approval.TEMPLATE.json', {'policy_id': SPAN_RULE['policy_id'],
               'rule_hash': digest(SPAN_RULE), 'status': 'proposed', 'reviewer': '', 'evidence_ref': ''})
    write_json(stage / 'registry.json', config('semantic_registry.json'))
    write_json(stage / 'registry_ledger.json', empty_ledger())
    catalog = {'schema_version': 'physalign_draft_catalog_v1', 'converter_version': VERSION,
               'native_profile': PROFILE, 'implementation_hashes': dependency_hashes(),
               'tasks_requested': ['T02', 'T03'], 'status': 'pending_benchmark_QA_review',
               'registry_hash': digest(config('semantic_registry.json')),
               'partition_policy': 'exact_dedup_development_reserved_rest_unassigned',
               'probes': [], 'problems': []}
    inventory = {'schema_version': 'native_read_only_inventory_v1', 'problems': [],
                 'source_pass5_policy': 'integrity_hash_only_never_parsed_for_truth_or_selection'}
    membership = {'schema_version': 'native_membership_v1', 'selected_problem_ids': ids,
                  'workspace_snapshot_root': digest(before), 'members': [], 'development_problem_ids': [],
                  'seed': args.seed, 'dedup_policy': 'exact_effective_context_pixels_and_source_identity_v1',
                  'near_duplicate_policy': 'not_resolved_requires_team_cross_source_check',
                  'formal_test_split': 'not_assigned'}
    ready = {}
    for n, pid in enumerate(ids, 1):
        try:
            loaded = adapter.load(pid)
            a = adapter.materialize(loaded, stage / 'problems' / pid, native=True, span_approval=approval)
            ready[pid] = a
            source = loaded['problem']
            record = {'problem_id': pid, 'state': 'source_ready',
                      'counts': {c: len(a['source'][c]) for c in COLLECTIONS},
                      'adapter_issues': a['audit']['issues'], 'source_warnings': loaded['warnings']}
            membership['members'].append({'problem_id': pid,
                'source_identity': digest([source['source_dataset'], source['source_split'], source['source_sample_id']]),
                'source_dataset': source['source_dataset'], 'source_split': source['source_split'],
                'source_sample_id': source['source_sample_id'], 'language': source['language'],
                'source_files': a['env']['source_files'], 'source_group_id': a['audit']['group_id'],
                'content_group_hint': a['audit']['content_group_hint'],
                'image_versions': {i: v['bytes_sha256'] for i, v in a['env']['assets'].items()},
                'native_projection': a['env']['native_profile']})
        except (ValueError, KeyError, TypeError) as exc:
            record = {'problem_id': pid, 'state': 'blocked', **error_record(exc, 'source', [])}
        inventory['problems'].append(record)
        if n % 25 == 0 or n == len(ids):
            print(f'[inventory {n}/{len(ids)}] verified source mothers: {len(ready)}', flush=True)
    inventory['counts'] = dict(Counter(r['state'] for r in inventory['problems']))
    dedup_members(membership['members'])
    active = [r for r in membership['members'] if r['dedup_state'] == 'representative']
    write_json(stage / 'inventory.json', inventory)
    write_json(stage / 'membership.pending.json', membership)
    seen = set()
    for n, member in enumerate(active, 1):
        pid = member['problem_id']
        a = ready[pid]
        compiler = NativeCompiler(a)
        info = {'problem_id': pid, 'source_state': 'source_ready', 'draft_probes': 0,
                'blocked_queries': [], 'source_audit': 'problems/' + pid + '/source_audit.json',
                'span_sidecar_entries': read_json(stage / 'problems' / pid / 'native_span_sidecar.json')['entries']}
        for task in ('T02', 'T03'):
            for proposal in compiler.proposals(task):
                try:
                    d, env, preview = compiler.compile(task, proposal, args.max_views)
                    _, target = prospective(d, env)
                    qid = d['key']['logical_probe_id']
                    require(qid not in seen, 'DUPLICATE_LOGICAL_ID')
                    seen.add(qid)
                    rel = 'probes/' + qid
                    stored = copy.deepcopy(env)
                    stored['asset_root'] = '.'
                    metadata = describe_probe(compiler, d, env, preview, proposal)
                    write_json(stage / rel / 'draft.json', d)
                    write_json(stage / rel / 'environment.PRIVATE.json', stored)
                    write_json(stage / rel / 'preview.PRIVATE.json', preview)
                    write_json(stage / rel / 'metadata.PRIVATE.json', metadata)
                    item = {'logical_probe_id': qid, 'problem_id': pid, 'task_id': task, 'language': d['key']['language'],
                        'group_id': a['audit']['group_id'], 'content_group_hint': a['audit']['content_group_hint'],
                        'asset_root': 'problems/' + pid, 'draft': rel + '/draft.json',
                        'environment': rel + '/environment.PRIVATE.json', 'preview': rel + '/preview.PRIVATE.json',
                        'review_target_root': target['content_root'],
                        'source_scope': sorted({r['ref']['record_id'] for r in proposal['rows']}),
                        'metadata_path': rel + '/metadata.PRIVATE.json', 'metadata': metadata}
                    catalog['probes'].append(item)
                    info['draft_probes'] += 1
                except (ValueError, KeyError, TypeError) as exc:
                    info['blocked_queries'].append({**error_record(exc, task, proposal['native']['source_scope']),
                                                   'group_id': proposal['native']['group_id']})
        info['blocked_queries'].extend(compiler.blocked)
        info['groups'] = compiler.groups
        info['dependency_report'] = 'problems/' + pid + '/native_dependencies.PRIVATE.json'
        write_json(stage / info['dependency_report'], compiler.shadow)
        catalog['problems'].append(info)
        print(f'[native {n}/{len(active)}] {pid}: {info["draft_probes"]} draft; {len(info["blocked_queries"])} blocked', flush=True)
    dev = development_ids(catalog['probes'], args.development_count, args.seed)
    membership['development_problem_ids'] = dev
    development_groups = {r['dedup_group'] for r in membership['members'] if r['problem_id'] in dev}
    for r in membership['members']:
        if r['dedup_group'] in development_groups:
            r['split'] = 'development_reserved'
    for item in catalog['probes']:
        item['split'] = 'development_reserved' if item['problem_id'] in dev else 'candidate_unassigned_not_release'
    membership['membership_root'] = digest(membership)
    catalog['membership_root'] = membership['membership_root']
    report = summarize(catalog, inventory, membership)
    catalog['counts'] = {'problems': len(active), 'draft_probes': len(catalog['probes']),
                          'by_task': dict(Counter(p['task_id'] for p in catalog['probes']))}
    write_json(stage / 'membership.json', membership)
    write_json(stage / 'development_candidates.json', {'problem_ids': dev, 'seed': args.seed,
               'method': 'deterministic_coverage_first_NOT_a_random_release_audit',
               'membership_root': membership['membership_root'], 'all_probes_of_selected_mothers_in_review': True})
    write_json(stage / 'catalog.PRIVATE.json', catalog)
    write_json(stage / 'report.json', report)
    write_json(stage / 'decisions.template.json', {'schema_version': 'physalign_decisions_v1', 'catalog_root': digest(catalog),
        'decisions': [{'logical_probe_id': p['logical_probe_id'], 'review_target_root': p['review_target_root'],
                       'status': 'pending', 'reviewer': '', 'blind_answer': '', 'note': '',
                       'confirmations': {k: False for k in CONFIRMATIONS}} for p in catalog['probes'] if p['problem_id'] in dev]})
    write_review_page(stage, catalog, set(dev))
    after = workspace_snapshot(adapter.root)
    require(before == after, 'SOURCE_WORKSPACE_CHANGED_DURING_NATIVE_DRY_RUN')
    # Recheck original image bytes, not only their copied snapshots.
    for pid in ready:
        source = adapter.by_id[pid]
        for image in source['images']:
            require(sha_bytes(safe_path(adapter.dataset, image['path']).read_bytes()) == image['sha256'],
                    'ORIGINAL_IMAGE_CHANGED_DURING_NATIVE_DRY_RUN', pid)
    write_json(stage / 'source_integrity.after.json', after)
    write_json(stage / 'module_status.json', {'executed': ['source_review_hash_validation', 'source_image_byte_pixel_dimension_validation',
        'native_shadow_closure_before_filtering', 'single_query_direct_source_mapping', 'raw_gold_preview_rendering',
        'public_text_availability', 'public_only_geometry_diagnostic', 'exact_dedup_and_development_selection',
        'before_after_original_integrity_check'],
        'not_implemented_this_stage': ['model_runner', 'prediction_scorer', 'planned_denominator_aggregator',
        'control_images', 'batch_approval_inheritance', 'random_audit_CI_and_two_round_stop', 'formal_release'],
        'UI_status': 'existing_static_private_review; development_mothers_only; no_new_human_review_performed'})
    publish(stage, final)
    print(json.dumps({'output': str(final), 'B_candidate': report['B_candidate'],
                      'interfaces': report['interfaces'], 'P_approved': report['P_approved']}, ensure_ascii=False), flush=True)


def validate_native_output(args):
    from convert import load_probe, verify_descriptor
    root = Path(args.path).resolve()
    verify_directory(root)
    catalog = read_json(root / 'catalog.PRIVATE.json')
    require(catalog.get('native_profile') == PROFILE, 'NOT_NATIVE_DRAFT')
    require(catalog['implementation_hashes'] == dependency_hashes(), 'CONVERTER_VERSION_MISMATCH')
    membership = read_json(root / 'membership.json')
    require(digest({k: v for k, v in membership.items() if k != 'membership_root'}) == membership['membership_root'] ==
            catalog['membership_root'], 'MEMBERSHIP_CHANGED')
    require(read_json(root / 'source_integrity.before.json') == read_json(root / 'source_integrity.after.json'),
            'ORIGINAL_INTEGRITY_MISMATCH')
    # Verify source/sidecar snapshots even for mothers with ZERO exported probes.
    # They still influence membership, deduplication and the blocked-yield report.
    for member in membership['members']:
        base = safe_path(root, 'problems/' + member['problem_id'])
        for spec in member['source_files'].values():
            require(sha_bytes(safe_path(base, spec['relative_path']).read_bytes()) == spec['bytes_sha256'],
                    'MEMBER_SOURCE_VERSION_CHANGED')
        spec = member['native_projection']['sidecar']
        require(sha_bytes(safe_path(base, spec['relative_path']).read_bytes()) == spec['bytes_sha256'],
                'MEMBER_SIDECAR_VERSION_CHANGED')
    for n, item in enumerate(catalog['probes'], 1):
        d, env = load_probe(root, item)
        rebuilt = verify_adapter_snapshot(env)
        verify_descriptor(item, d['key'], rebuilt)
        preview = prepare_probe(d, env, verify_reviews=False)
        require(preview == read_json(safe_path(root, item['preview'])), 'DRAFT_PREVIEW_MISMATCH')
        require(prospective(d, env)[1]['content_root'] == item['review_target_root'], 'DRAFT_REVIEW_ROOT_MISMATCH')
        compiler = NativeCompiler(rebuilt)
        proposal = next(p for p in compiler.proposals(item['task_id']) if p['native'] == env['native_query'])
        meta = describe_probe(compiler, d, env, preview, proposal)
        require(meta == item['metadata'] == read_json(safe_path(root, item['metadata_path'])), 'NATIVE_METADATA_CHANGED')
        require(d['key']['packet_plan']['state'] == 'pending_review', 'NATIVE_DRAFT_APPROVAL_FORBIDDEN')
        if n % 50 == 0:
            print(f'[validate {n}/{len(catalog["probes"])}]', flush=True)
    expected = summarize(catalog, read_json(root / 'inventory.json'), membership)
    require(expected == read_json(root / 'report.json'), 'NATIVE_REPORT_CHANGED')
    print(json.dumps({'valid': True, 'kind': 'native_draft_NOT_release', 'probes': len(catalog['probes'])}), flush=True)
