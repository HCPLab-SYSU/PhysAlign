"""Versioned native-only projections; originals and source reviews never change."""
from __future__ import annotations
import copy
from pathlib import Path
from reference_checks import require, digest, resolve_span, text_digest
from reference_integrity import source_ref_key, active_origin
from io_utils import read_json, write_json, sha_bytes, safe_path

PROFILE = 'native_binding_draft_v1'
SPAN_RULE = {'policy_id': 'unique_exact_sidecar_v1',
             'preconditions': ['positive_integer_occurrence_out_of_range',
                               'nonempty_quote_exactly_once_in_current_exact_section'],
             'operation': 'use_the_only_exact_character_span_without_modifying_source',
             'matching': 'literal_nonoverlap_1based_v1'}


def check_approval(approval):
    if approval is None:
        return
    require(set(approval) == {'policy_id', 'rule_hash', 'status', 'reviewer', 'evidence_ref'},
            'SPAN_POLICY_APPROVAL_FIELDS')
    require(approval['policy_id'] == SPAN_RULE['policy_id'] and
            approval['rule_hash'] == digest(SPAN_RULE) and approval['status'] == 'approved',
            'SPAN_POLICY_NOT_APPROVED_OR_STALE')
    require(all(isinstance(approval[k], str) and approval[k].strip()
                for k in ('reviewer', 'evidence_ref')), 'SPAN_POLICY_EVIDENCE_REQUIRED')


def project_native(adapted, root, approval=None, *, write=False):
    check_approval(approval)
    a = copy.deepcopy(adapted)
    env, source, index = a['env'], a['source'], a['env']['source_index']
    context, assets = index['context'], env['assets']
    entries = []
    for m in source['text_mentions']:
        decision = resolve_span(context[m['section']], m['quote'], m['occurrence'],
                                unique_policy_approved=approval is not None)
        if decision['status'] not in ('unique_exact_repair_proposed', 'repaired_unique_exact'):
            continue
        loc = {'kind': 'text', 'section': m['section'], 'start': decision['proposed_span'][0],
               'end': decision['proposed_span'][1], 'quote': m['quote'],
               'section_sha256': text_digest(context[m['section']])}
        entry = {'record_id': m['id'], 'source_file_sha256': a['sha'],
                 'section_sha256': loc['section_sha256'], 'old_occurrence': m['occurrence'],
                 'old_locator': None, 'proposed_locator': loc, 'new_occurrence': 1,
                 'state': 'applied_under_approved_rule' if approval else 'proposed',
                 'rule_hash': digest(SPAN_RULE), 'policy_ref': digest(approval) if approval else None}
        entry['repair_id'] = 'SR-' + digest(entry)
        entries.append(entry)
        if approval is not None:
            a['locators'][m['id']] = loc
            ref = {'collection': 'text_mentions', 'record_id': m['id'],
                   'source_file_sha256': a['sha']}
            index['anchor_origins'][source_ref_key(ref)] = active_origin(loc, context, assets)
            a['audit']['span_decisions'][m['id']] = decision
    applied = {e['record_id'] for e in entries if e['state'] == 'applied_under_approved_rule'}
    a['audit']['issues'] = [i for i in a['audit']['issues']
                           if not (i['code'] == 'SPAN_REVIEW_REQUIRED' and i['scope'] in applied)]
    sidecar = {'schema_version': 'native_span_sidecar_v1', 'profile': PROFILE,
               'source_hashes': {k: v['bytes_sha256'] for k, v in env['source_files'].items()},
               'rule': SPAN_RULE, 'approval': approval, 'entries': entries}
    path = Path(root) / 'native_span_sidecar.json'
    if write:
        write_json(path, sidecar)
    else:
        require(read_json(path) == sidecar, 'NATIVE_SIDECAR_SOURCE_OR_POLICY_CHANGED')
    env['native_profile'] = {'profile_id': PROFILE, 'approval': approval,
                             'sidecar': {'relative_path': path.name,
                                         'bytes_sha256': sha_bytes(path.read_bytes())}}
    env['enabled_variants'] = ['T02-one', 'T03-one']
    a['repair_refs'] = {e['record_id']: [e['repair_id']] for e in entries if e['record_id'] in applied}
    # Use the UNION of existing explicit entity geometry and represents edges.
    # No name matching, labels->entity inference or gold-type filtering.
    visuals = {v['id']: v for v in source['visual_nodes']}
    entity_vids = {c: set() for c in index['entity_locators']}
    for p in source['physical_nodes']:
        entity_vids[index['identity_map'][p['id']]].update(p['visual_anchor_ids'])
    for b in source['bindings']:
        if b['type'] == 'represents':
            require(b['from_id'] in visuals and b['to_id'] in index['identity_map'],
                    'NATIVE_REPRESENTS_ENDPOINT_INVALID')
            entity_vids[index['identity_map'][b['to_id']]].add(b['from_id'])
    for cid, vids in entity_vids.items():
        if any(v not in a['locators'] for v in vids):
            index['entity_locators'][cid] = []
            a['audit']['issues'].append({'code': 'NATIVE_CANDIDATE_DISPLAY_INCOMPLETE',
                                         'scope': sorted(vids), 'canonical_entity': cid})
        else:
            unique = {digest(a['locators'][v]): a['locators'][v] for v in vids}
            index['entity_locators'][cid] = [unique[k] for k in sorted(unique)]
    a['audit'].update(adapter_version='physgraph_workspace_native_v1',
                       native_profile=copy.deepcopy(env['native_profile']),
                       entity_visual_sources={k: sorted(v) for k, v in entity_vids.items()},
                       entity_projection_policy='union_explicit_visual_anchors_and_represents_v1',
                       span_repair_refs=a['repair_refs'])
    if write:
        write_json(Path(root) / 'source_audit.json', a['audit'])
    return a


def verify_sidecar_bytes(env):
    p = env['native_profile']
    require(p['profile_id'] == PROFILE, 'NATIVE_PROFILE_UNKNOWN')
    path = safe_path(env['asset_root'], p['sidecar']['relative_path'])
    require(sha_bytes(path.read_bytes()) == p['sidecar']['bytes_sha256'], 'NATIVE_SIDECAR_BYTES_CHANGED')
