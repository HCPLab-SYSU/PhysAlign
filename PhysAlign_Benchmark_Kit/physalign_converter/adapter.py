"""Read-only adapter for the actual PhysGraph annotation workspace layout.

Raw rows remain byte-bound snapshots. Resolved spans and identity projections
are separate; names, source solutions and Pass5 never create observed truth.
"""
from __future__ import annotations
import copy
from collections import Counter
from pathlib import Path
from reference_checks import require, digest, text_digest, resolve_span, validate_locator
from reference_integrity import active_origin, source_ref_key, observation_review_root
from io_utils import (read_json, parse_json, write_json, sha_bytes, safe_path, safe_id,
                      empty_ledger, review_record)
from renderer import asset_record
from source_validation import effective_problem, validate_document

def normalized_context(problem):
    options = []
    for option in problem['segments']['options']:
        require(set(option) in ({'label', 'text'}, {'id', 'text'}), 'SOURCE_OPTION_SCHEMA_UNKNOWN')
        options.append({'id': option.get('id', option.get('label')), 'text': option['text']})
    return {'stem': problem['segments']['stem'], 'query': problem['segments']['query'],
            'options': options, 'image_ids': [i['image_id'] for i in problem['images']]}


def normalize_loaded(loaded, root, source_files, assets):
    problem = loaded['problem']
    context = normalized_context(problem)
    source = loaded['docs']['pass4']
    sha = source_files['pass4']['bytes_sha256']
    def ref(collection, rid, field):
        return {'collection': collection, 'record_id': rid, 'target_field': field, 'source_file_sha256': sha}
    index = {'example_only': False, 'namespace': problem['source_dataset'], 'language': problem['language'],
             'context': context, 'records': {}, 'identity_map': {}, 'entity_locators': {},
             'entity_types': {}, 'anchor_origins': {}}
    for collection in COLLECTIONS:
        for record in source[collection]:
            index['records'][source_ref_key(ref(collection, record['id'], 'id'))] = copy.deepcopy(record)
    issues, locators, spans = [], {}, {}
    for v in source['visual_nodes']:
        loc = {'kind': 'visual', 'image_id': v['image_id'],
               'geometry': {'type': 'bbox', 'bbox_1000': copy.deepcopy(v['bbox_1000'])}}
        try:
            validate_locator(loc, context, assets)
            locators[v['id']] = loc
            index['anchor_origins'][source_ref_key(ref('visual_nodes', v['id'], 'text'))] = active_origin(loc, context, assets)
        except ValueError as exc:
            issues.append({'scope': v['id'], 'code': 'GEOMETRY_UNSUPPORTED', 'detail': str(exc)})
    for m in source['text_mentions']:
        span = resolve_span(context[m['section']], m['quote'], m['occurrence'])
        spans[m['id']] = span
        if span['status'] != 'resolved_exact':
            issues.append({'scope': m['id'], 'code': 'SPAN_REVIEW_REQUIRED', 'detail': span})
            continue
        start, end = span['span']
        loc = {'kind': 'text', 'section': m['section'], 'start': start, 'end': end,
               'quote': m['quote'], 'section_sha256': text_digest(context[m['section']])}
        locators[m['id']] = loc
        index['anchor_origins'][source_ref_key(ref('text_mentions', m['id'], 'quote'))] = active_origin(loc, context, assets)
    physical = {p['id']: p for p in source['physical_nodes']}
    parent = {p: p for p in physical}
    def find(p):
        require(p in parent, 'IDENTITY_ENDPOINT_NOT_PHYSICAL', p)
        while parent[p] != p:
            p = parent[p]
        return p
    identity_edges = []
    for b in source['bindings']:
        if b['type'] == 'same_entity_as':
            left, right = find(b['from_id']), find(b['to_id'])
            parent[max(left, right)] = min(left, right)
            identity_edges.append(b['id'])
    for pid in physical:
        canonical = 'PE-' + digest({'dataset': problem['source_dataset'], 'split': problem['source_split'],
                                   'sample': problem['source_sample_id'], 'source_id': find(pid)})
        index['identity_map'][pid] = canonical
    for canonical in sorted(set(index['identity_map'].values())):
        members = [p for p, c in index['identity_map'].items() if c == canonical]
        types = {physical[p]['type'] for p in members}
        require(len(types) == 1, 'IDENTITY_TYPE_CONFLICT', str(members))
        index['entity_types'][canonical] = next(iter(types))
        vids = sorted({v for p in members for v in physical[p]['visual_anchor_ids']})
        # Never keep only a convenient subset of a multi-part entity's geometry.
        if any(v not in locators for v in vids):
            index['entity_locators'][canonical] = []
            issues.append({'scope': members, 'code': 'ENTITY_DISPLAY_INCOMPLETE'})
        else:
            unique = {digest(locators[v]): locators[v] for v in vids}
            index['entity_locators'][canonical] = [unique[k] for k in sorted(unique)]
    ledger = empty_ledger()
    obs = {}
    for v in source['visual_nodes']:
        if v['type'] != 'text_glyph' or not v['text'].strip() or v['id'] not in locators:
            continue
        oid = 'o_' + v['id']
        record = {'observation_id': oid, 'state': 'approved',
                  'origin': active_origin(locators[v['id']], context, assets), 'text': v['text'],
                  'source_records': [ref('visual_nodes', v['id'], 'text')], 'source_kind': 'text_glyph',
                  'asset_role': 'diagram', 'review_ref': 'source-observation-' + oid}
        obs[oid] = record
        rid = record['review_ref']
        ledger['records'][rid] = review_record(rid, 'observation', observation_review_root(record),
            loaded['review']['stages']['pass4']['reviewer'],
            ['inherited-source-pass4-review:' + source_files['review']['bytes_sha256'], 'pass4:' + sha])
    env = {'production': True, 'assets': assets, 'asset_root': str(root), 'source_index': index,
           'observations': {'schema_version': 'physalign_observations_v2_1', 'observations': obs},
           'ledger': ledger, 'source_files': source_files,
           'enabled_variants': ['T01-one', 'T02-one', 'T03-one', 'T04-set', 'T05-one']}
    audit = {'adapter_version': 'physgraph_workspace_v1', 'problem_id': problem['problem_id'],
             'source_dataset': problem['source_dataset'], 'source_sample_id': problem['source_sample_id'],
             'source_split': problem.get('source_split'), 'span_decisions': spans, 'issues': issues,
             'identity_binding_ids': identity_edges, 'source_warnings': loaded['warnings'],
             'image_role_policy': 'provisional_diagram_requires_QA_reviewer_confirmation',
             'geometry_policy': 'explicit_source_bbox_no_inferred_keypoint_order',
             'pass5_policy': 'not_read_not_used_for_selection_or_truth',
             'group_id': 'PG-' + digest({'dataset': problem['source_dataset'], 'split': problem['source_split'], 'sample': problem['source_sample_id']}),
             'content_group_hint': digest({'context': context, 'pixels': [a['pixels_sha256'] for a in assets.values()]})}
    return {'env': env, 'source': source, 'locators': locators, 'audit': audit, 'sha': sha}

COLLECTIONS = ('visual_nodes', 'physical_nodes', 'text_mentions', 'quantities',
               'bindings', 'relations', 'constraints', 'ambiguities')
STAGES = ('pass1', 'pass2', 'pass3', 'pass4')


def verify_adapter_snapshot(env):
    """Rebuild the adapter boundary from portable original snapshots, offline."""
    from reference_integrity import verify_source_snapshots
    from reference_checks import verify_image_assets
    root = Path(env['asset_root']).resolve()
    files = env['source_files']
    require(set(files) == set(STAGES) | {'manifest', 'review'}, 'ADAPTER_SNAPSHOT_FILES')
    verify_source_snapshots(files, env['source_index'], {k: v['bytes_sha256'] for k, v in files.items()}, root)
    originals = {i: env['assets'][i] for i in env['source_index']['context']['image_ids']}
    verify_image_assets(originals, root)
    parsed = {k: read_json(safe_path(root, v['relative_path'])) for k, v in files.items()}
    review = parsed['review']
    require(review.get('exclusion', {}).get('status') != 'excluded', 'SOURCE_EXCLUDED')
    problem = effective_problem(parsed['manifest'], review)
    docs, warnings = {}, []
    for stage in STAGES:
        status = review.get('stages', {}).get(stage, {})
        require(status.get('status') == 'approved' and status.get('document_sha256') == digest(parsed[stage]),
                'SOURCE_REVIEW_HASH_STALE', stage)
        issues = validate_document(stage, parsed[stage], problem, docs)
        require(not any(i['level'] == 'error' for i in issues), 'SOURCE_SCHEMA_INVALID', stage)
        warnings.extend({'stage': stage, **i} for i in issues)
        docs[stage] = parsed[stage]
    require({i['image_id'] for i in problem['images']} == set(originals), 'SOURCE_IMAGE_SET_MISMATCH')
    for img in problem['images']:
        asset = originals[img['image_id']]
        require((img['sha256'], img['width'], img['height']) ==
                (asset['bytes_sha256'], asset['width'], asset['height']), 'SOURCE_IMAGE_HASH_STALE')
    loaded = {'manifest': parsed['manifest'], 'review': review, 'problem': problem,
              'docs': docs, 'warnings': warnings}
    rebuilt = normalize_loaded(loaded, root, files, originals)
    if 'native_profile' in env:
        from native_projection import project_native, verify_sidecar_bytes
        verify_sidecar_bytes(env)
        rebuilt = project_native(rebuilt, root, env['native_profile']['approval'])
        require(rebuilt['env']['native_profile'] == env['native_profile'], 'NATIVE_PROFILE_CHANGED')
    for key in ('source_index', 'observations'):
        require(rebuilt['env'][key] == env[key], 'ADAPTER_PROJECTION_CHANGED', key)
    for rid, record in rebuilt['env']['ledger']['records'].items():
        require(env['ledger']['records'].get(rid) == record, 'SOURCE_OBSERVATION_REVIEW_CHANGED')
    require(read_json(root / 'source_audit.json') == rebuilt['audit'], 'SOURCE_AUDIT_CHANGED')
    return rebuilt


class SourceAdapter:
    def __init__(self, workspace, dataset_root=None):
        self.root = Path(workspace).resolve()
        config = read_json(self.root / 'workspace_config.json')
        dataset_path = Path(dataset_root or config['dataset_dir'])
        if dataset_root is None and config.get('path_base') == 'workspace' and not dataset_path.is_absolute():
            dataset_path = self.root / dataset_path
        self.dataset = dataset_path.resolve()
        require(self.dataset.is_dir(), 'DATASET_ROOT_NOT_FOUND_USE_OVERRIDE', str(self.dataset))
        self.review_state = read_json(self.root / 'reviews/state.json')
        manifest = self.root / 'blind/manifest.jsonl'
        self.rows = [parse_json(l) for l in manifest.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        ids = [safe_id(r['problem_id']) for r in self.rows]
        require(len(set(ids)) == len(ids), 'DUPLICATE_MANIFEST_ID')
        self.by_id = dict(zip(ids, self.rows))

    def load(self, pid):
        row = copy.deepcopy(self.by_id[pid])
        review = copy.deepcopy(self.review_state.get('problems', {}).get(pid, {}))
        require(review.get('exclusion', {}).get('status') != 'excluded', 'SOURCE_EXCLUDED')
        problem = effective_problem(row, review)
        require(problem.get('language') in ('en', 'zh'), 'LANGUAGE_UNRESOLVED')
        require(all(isinstance(problem.get(k), str) and problem[k].strip()
                    for k in ('source_dataset', 'source_split', 'source_sample_id')), 'SOURCE_IDENTITY_MISSING')
        docs, blobs, warnings = {}, {}, []
        for stage in STAGES:
            path = safe_path(self.root, f'passes/{stage}/{safe_id(pid)}.json')
            require(path.is_file(), 'SOURCE_STAGE_MISSING', stage)
            blob = path.read_bytes()
            doc = parse_json(blob.decode('utf-8-sig'))
            status = review.get('stages', {}).get(stage, {})
            require(status.get('status') == 'approved', 'SOURCE_REVIEW_NOT_APPROVED', stage)
            require(status.get('document_sha256') == digest(doc), 'SOURCE_REVIEW_HASH_STALE', stage)
            require(status.get('reviewer', '').strip(), 'SOURCE_REVIEWER_MISSING', stage)
            issues = validate_document(stage, doc, problem, docs)
            errors = [i for i in issues if i['level'] == 'error']
            require(not errors, 'SOURCE_SCHEMA_INVALID', stage + ': ' + str(errors[:3]))
            warnings.extend({'stage': stage, **i} for i in issues)
            docs[stage], blobs[stage] = doc, blob
        # A Pass5-only exclusion is intentionally not a G_obs exclusion.
        return {'manifest': row, 'problem': problem, 'review': review,
                'docs': docs, 'blobs': blobs, 'warnings': warnings}

    def materialize(self, loaded, root, *, native=False, span_approval=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        sources = root / 'sources'
        sources.mkdir()
        source_files = {}
        for stage, blob in loaded['blobs'].items():
            path = sources / (stage + '.json')
            path.write_bytes(blob)
            source_files[stage] = {'relative_path': path.relative_to(root).as_posix(), 'bytes_sha256': sha_bytes(blob)}
        for name in ('manifest', 'review'):
            path = sources / (name + '.json')
            write_json(path, loaded[name])
            source_files[name] = {'relative_path': path.relative_to(root).as_posix(), 'bytes_sha256': sha_bytes(path.read_bytes())}
        problem = loaded['problem']
        context = normalized_context(problem)
        require(bool(context['image_ids']), 'NO_SOURCE_IMAGES')
        require(len(set(context['image_ids'])) == len(context['image_ids']), 'DUPLICATE_SOURCE_IMAGE_ID')
        assets = {}
        for item in problem['images']:
            iid = safe_id(item['image_id'])
            require(not iid.startswith('V'), 'SOURCE_IMAGE_RENDER_ID_COLLISION')
            path = safe_path(self.dataset, item['path'])
            blob = path.read_bytes()
            require(sha_bytes(blob) == item['sha256'], 'SOURCE_IMAGE_HASH_STALE', iid)
            target = root / 'images' / (iid + path.suffix.lower())
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(blob)
            assets[iid] = asset_record(target, root)
            require((assets[iid]['width'], assets[iid]['height']) == (item['width'], item['height']),
                    'SOURCE_IMAGE_DIMENSIONS_STALE', iid)
        adapted = normalize_loaded(loaded, root, source_files, assets)
        write_json(root / 'source_audit.json', adapted['audit'])
        if native:
            from native_projection import project_native
            adapted = project_native(adapted, root, span_approval, write=True)
        return adapted
    def inventory(self, ids=None):
        reports = []
        for pid in ids if ids is not None else sorted(self.by_id):
            try:
                loaded = self.load(pid)
                reports.append({'problem_id': pid, 'state': 'source_ready',
                                'has_pass5': (self.root / 'passes/pass5' / (pid + '.json')).is_file(),
                                'counts': {c: len(loaded['docs']['pass4'][c]) for c in COLLECTIONS},
                                'warnings': loaded['warnings']})
            except (ValueError, OSError, KeyError, TypeError) as exc:
                reports.append({'problem_id': pid, 'state': 'blocked',
                                'code': getattr(exc, 'code', type(exc).__name__), 'detail': str(exc)})
        return {'schema_version': 'physalign_inventory_v1', 'counts': dict(Counter(r['state'] for r in reports)),
                'pass5_policy': 'optional_not_a_source_truth_gate', 'problems': reports}
