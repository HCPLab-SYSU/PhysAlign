"""Source-position, R/E/H and review-reference checks for the 2.1 boundary.

SourceIndex is produced by the actual source adapter from verified snapshots.
The in-memory ledger is an offline authorization boundary, not a cryptographic
signature service; deployments must load it from their trusted review store.
"""
from __future__ import annotations
import copy
from typing import Any
from reference_checks import require, exact_keys, digest, text_digest, validate_locator, canonical_bytes
from reference_normalizers import meaningful_text
from schema_tools import validate_schema


def origin_fingerprint(origin: dict[str, Any]) -> str:
    return digest(origin)


def active_origin(locator: dict[str, Any], context: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    validate_locator(locator, context, assets)
    if locator['kind'] == 'visual':
        record = assets[locator['image_id']]
        version = record.get('bytes_sha256')
        require(isinstance(version, str) and len(version) == 64 and all(c in '0123456789abcdef' for c in version), 'ASSET_VERSION_MISSING')
    else:
        version = text_digest(context[locator['section']])
    return {'locator': copy.deepcopy(locator), 'source_version': version}


def validate_origin(origin, context, assets):
    exact_keys(origin, ('locator', 'source_version'))
    require(origin == active_origin(origin['locator'], context, assets), 'ORIGIN_VERSION_MISMATCH')


def validate_observation_location(oid, obs, alias, bundle, assets, anchor_map):
    require(obs.get('observation_id') == oid and obs.get('state') == 'approved', 'OBSERVATION_UNAPPROVED')
    require(meaningful_text(obs.get('text')), 'EMPTY_OBSERVATION_TEXT')
    require(alias in bundle['anchors'] and alias in anchor_map, 'OBSERVATION_ANCHOR_MISSING')
    validate_origin(obs['origin'], bundle['context'], assets)
    source = obs['origin']
    public = active_origin(bundle['anchors'][alias], bundle['context'], assets)
    private = anchor_map[alias]['origin']
    # All locators are in ORIGINAL source coordinates. Display transforms are
    # validated separately; they never change which source region an R denotes.
    require(origin_fingerprint(source) == origin_fingerprint(public) == origin_fingerprint(private), 'OBSERVATION_LOCATION_MISMATCH', oid + ' -> ' + alias)
    require(oid in anchor_map[alias]['observation_ids'], 'OBSERVATION_NOT_IN_ANCHOR_MAP')


def validate_observation_bindings(store, bundle, assets, anchor_map, read_targets):
    validate_schema('observation_store.schema.json', store)
    observations = store['observations']
    require(set(anchor_map) == set(bundle['anchors']), 'R_MAP_COVERAGE')
    obs_alias = {}
    for alias, meta in anchor_map.items():
        require(meta['origin'] == active_origin(bundle['anchors'][alias], bundle['context'], assets), 'ANCHOR_ORIGIN_MISMATCH')
        for oid in meta['observation_ids']:
            require(oid in observations and oid not in obs_alias, 'DUPLICATE_OR_UNKNOWN_OBSERVATION_ALIAS')
            validate_observation_location(oid, observations[oid], alias, bundle, assets, anchor_map)
            obs_alias[oid] = alias
    for t in read_targets:
        oid, alias = t['observation_id'], t['anchor_alias']
        require(oid in observations and obs_alias.get(oid) == alias, 'READ_OBSERVATION_ANCHOR_MISMATCH')
        obs = observations[oid]
        require(t['expected'] == obs['text'], 'READ_TEXT_MISMATCH')
        require(t['origin_fingerprint'] == origin_fingerprint(obs['origin']), 'READ_ORIGIN_MISMATCH')
    return obs_alias


class ReviewLedger:
    def __init__(self, data: dict[str, Any], *, production: bool):
        validate_schema('review_ledger.schema.json', data)
        require(not production or data['example_only'] is False, 'SYNTHETIC_LEDGER_NOT_FOR_RELEASE')
        self.data = copy.deepcopy(data)
        self.production = production
        for rid, rec in data['records'].items():
            require(rid == rec['review_id'], 'REVIEW_ID_MISMATCH')

    def require_review(self, ref: str, kind: str, root: str) -> dict[str, Any]:
        record = self.data['records'].get(ref)
        require(isinstance(record, dict), 'REVIEW_NOT_FOUND', str(ref))
        require(record['status'] == 'approved' and record['kind'] == kind and record['content_root'] == root, 'AUDIT_STALE_OR_WRONG_SCOPE', str(ref))
        require(meaningful_text(record['reviewer']) and bool(record['evidence_refs']), 'REVIEW_EVIDENCE_MISSING')
        return copy.deepcopy(record)


def observation_review_root(obs):
    return digest({k:v for k,v in obs.items() if k != 'review_ref'})


def packet_review_root(plan, obs_map, policy_id):
    return digest({'policy_id':policy_id, 'state':plan['state'], 'empty_reason':plan['empty_reason'],
                   'observations':[{'observation_id':oid,'content_hash':observation_review_root(obs_map[oid])} for oid in plan['observation_ids']]})


def source_ref_key(ref):
    return ref['source_file_sha256'] + ':' + ref['collection'] + ':' + ref['record_id']


def resolve_pointer(value, pointer):
    """Exact JSON Pointer or one literal top-level field, never eval/fuzzy lookup."""
    if not pointer.startswith('/'):
        require(isinstance(value,dict) and pointer in value, 'SOURCE_FIELD_MISSING', pointer)
        return value[pointer]
    for token in pointer[1:].split('/'):
        token = token.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            require(token.isdigit() and (token == '0' or not token.startswith('0')), 'SOURCE_POINTER_INVALID')
            i=int(token);require(i<len(value),'SOURCE_POINTER_INVALID');value=value[i]
        else:
            require(isinstance(value,dict) and token in value,'SOURCE_FIELD_MISSING',token);value=value[token]
    return value


def source_record(ref, index):
    key = source_ref_key(ref)
    require(key in index['records'], 'SOURCE_RECORD_MISSING', key)
    return index['records'][key]


def source_value(ref, index):
    return resolve_pointer(source_record(ref, index), ref['target_field'])


def canonical_entity(value, index):
    require(isinstance(value,str) and value in index['identity_map'], 'SOURCE_IDENTITY_UNRESOLVED')
    return {'kind':'entity','id':index['identity_map'][value]}


def validate_private_public_maps(key, bundle, assets, source_index):
    """Executed inside the integrated validation entry, not optional downstream."""
    require(set(key['candidate_map']) == {c['alias'] for c in bundle['candidates']}, 'E_MAP_COVERAGE')
    require(set(key['reference_map']) == set(bundle['reference_views']) == set(bundle['used_reference_ids']), 'H_MAP_COVERAGE')
    if key['task_id'] == 'T05':
        rc = key['relation_context']
        require(isinstance(rc,dict) and rc['fixed_reference_alias'] in key['reference_map'], 'T05_REFERENCE_MISSING')
        require(key['reference_map'][rc['fixed_reference_alias']]['canonical_target'] == rc['fixed_target'], 'T05_FIXED_TARGET_MISMATCH')
    else:
        require(not key['reference_map'] and key['relation_context'] is None, 'UNEXPECTED_REFERENCE_MAP')
    for ref in key['source_records']:
        source_value(ref, source_index)
    for alias, meta in key['reference_map'].items():
        locs = bundle['reference_views'][alias]
        require(meta['origins'] == [active_origin(x,bundle['context'],assets) for x in locs], 'H_DISPLAY_ORIGIN_MISMATCH')
        for ref in meta['source_records']:
            require(canonical_entity(source_value(ref,source_index),source_index) == meta['canonical_target'], 'H_SOURCE_TARGET_MISMATCH')
        entity = meta['canonical_target']['id']
        require(entity in source_index['entity_locators'], 'H_ENTITY_UNKNOWN')
        allowed = {digest(x) for x in source_index['entity_locators'][entity]}
        require(all(digest(x) in allowed for x in locs), 'H_ENTITY_DISPLAY_MISMATCH')
    for c in bundle['candidates']:
        target = key['candidate_map'][c['alias']]
        require(target['kind'] == c['kind'], 'E_DOMAIN_MISMATCH')
        if target['kind'] == 'entity':
            require(target['id'] in source_index['entity_locators'], 'E_ENTITY_UNKNOWN')
            allowed={digest(x) for x in source_index['entity_locators'][target['id']]}
            require(all(digest(x) in allowed for x in c['locators']), 'E_ENTITY_DISPLAY_MISMATCH')
    # R source provenance is also independently looked up, not copied from obs.
    for alias, meta in key['anchor_map'].items():
        require(meta['source_ids'] and len(set(meta['source_ids'])) == len(meta['source_ids']), 'R_SOURCE_IDS_INVALID')
        for sid in meta['source_ids']:
            require(sid in source_index['anchor_origins'] and source_index['anchor_origins'][sid] == meta['origin'], 'R_SOURCE_ORIGIN_MISMATCH')


def validate_observation_sources(store, source_index, ledger):
    for oid, obs in store['observations'].items():
        require(oid == obs['observation_id'], 'OBSERVATION_ID_MISMATCH')
        if obs['state'] != 'approved':
            continue  # It cannot be used by a read/packet/anchor mapping.
        require(meaningful_text(obs['text']), 'EMPTY_OBSERVATION_TEXT')
        ledger.require_review(obs['review_ref'], 'observation', observation_review_root(obs))
        for ref in obs['source_records']:
            require(source_value(ref,source_index) == obs['text'], 'OBSERVATION_SOURCE_TEXT_MISMATCH')
            origin = source_index['anchor_origins'].get(source_ref_key(ref))
            require(origin == obs['origin'], 'OBSERVATION_SOURCE_ORIGIN_MISMATCH')


def verify_source_snapshots(source_files, index, declared_hashes, root):
    """Verify current original snapshot bytes and raw record lookup bindings.

    This is not the source-specific approval-state machine. The adapter must
    validate that separately. Records in SourceIndex are exact original rows;
    repaired spans/identity views live in sidecars, not rewritten record copies.
    """
    import hashlib,json
    from pathlib import Path
    root=Path(root).resolve()
    require({name:r['bytes_sha256'] for name,r in source_files.items()}==declared_hashes,'SOURCE_SNAPSHOT_HASH_MAP_MISMATCH')
    parsed={}
    for name,r in source_files.items():
        p=(root/r['relative_path']).resolve()
        require(p.is_relative_to(root) and p.is_file(),'SOURCE_SNAPSHOT_FILE_MISSING',name)
        blob=p.read_bytes();sha=hashlib.sha256(blob).hexdigest()
        require(sha==r['bytes_sha256'],'SOURCE_SNAPSHOT_BYTES_CHANGED',name)
        # A source may have non-JSON auxiliary files; only hashes are needed for
        # those. All indexed records, however, must resolve to JSON snapshots.
        try: parsed[sha]=json.loads(blob.decode('utf-8'))
        except (ValueError,UnicodeError): pass
    for token,row in index['records'].items():
        parts=token.split(':',2)
        require(len(parts)==3,'SOURCE_RECORD_KEY_INVALID')
        sha,collection,rid=parts
        require(sha in parsed and isinstance(parsed[sha],dict) and isinstance(parsed[sha].get(collection),list),'SOURCE_RECORD_SNAPSHOT_UNAVAILABLE')
        matches=[r for r in parsed[sha][collection] if isinstance(r,dict) and r.get('id')==rid]
        require(len(matches)==1 and matches[0]==row,'SOURCE_INDEX_RECORD_MISMATCH',token)

