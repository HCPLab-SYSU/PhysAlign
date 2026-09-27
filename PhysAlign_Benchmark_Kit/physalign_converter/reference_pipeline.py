"""Normalized draft -> paired data artifacts -> reviewed immutable release.

Data construction only: no provider calls, prediction parsing or model scoring.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from reference_checks import (require, digest, canonical_bytes, build_packet, validate_view_bundle,
    validate_pair, validate_generated_question_references, verify_image_assets, validate_batch_identity)
from reference_integrity import (ReviewLedger, active_origin, origin_fingerprint, validate_observation_bindings,
    validate_observation_sources, validate_private_public_maps, packet_review_root, source_record,
    source_value, canonical_entity, source_ref_key, verify_source_snapshots)
from reference_registry import (validate_registry, resolve_relation, entry_root, render_qualifiers,
    render_relation_question, validate_direction_preconditions)
from reference_geometry import validate_render_manifest
from response_contract import validate_contract
from schema_tools import validate_schema, schema_bundle

ROOT=Path(__file__).resolve().parent

def config(name): return json.loads((ROOT/'config'/name).read_text(encoding='utf-8'))

def dependency_hashes():
    # Byte hashes bind actual code/normalizer definitions, not names alone.
    files=list((ROOT/'config').glob('*.json'))+list((ROOT/'schemas').glob('*.json'))+list(ROOT.glob('*.py'))+list(ROOT.glob('*.html'))
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def key_core(key):
    return {k:copy.deepcopy(v) for k,v in key.items() if k not in ('instance_id','content_root','instance_review_refs')}


def neutral_aliases(targets, logical_probe_id, *, variant_id='main', seed=2027):
    """Two deterministic SHA-256 orderings; no gold target argument exists."""
    require(type(seed) is int and isinstance(logical_probe_id,str) and bool(logical_probe_id),'ALIAS_SEED_OR_ID')
    unique={digest(t):copy.deepcopy(t) for t in targets}
    require(len(unique)==len(targets) and len(targets)>=2,'CANDIDATE_UNIVERSE_INVALID')
    base={'seed':seed,'logical_probe_id':logical_probe_id,'variant_id':variant_id}
    ordered=sorted(unique.values(),key=lambda t:digest({**base,'phase':'alias','target':t}))
    mapping={'E'+str(i+1):t for i,t in enumerate(ordered)}
    display=sorted(mapping,key=lambda a:digest({**base,'phase':'display','alias':a}))
    return mapping,display


def validate_candidate_generation(key,view,env):
    require(key['selection_policy_id']=='neutral_hash_per_probe_v2_1','CANDIDATE_POLICY_UNREGISTERED')
    if key['task_id']=='T01':
        targets=[{'kind':'role','id':r['canonical_id']} for r in env['registry']['roles'] if r['status']=='approved']
    else:
        index=env['source_index'];allowed=None
        if key['task_id']=='T05':
            rc=key['relation_context'];entries=[r for r in env['registry']['relations'] if r['canonical_id']==rc['canonical_predicate']]
            require(len(entries)==1,'RELATION_ENTRY_MISSING')
            allowed=entries[0]['object_types' if rc['missing_side']=='object' else 'subject_types']
        targets=[{'kind':'entity','id':pid} for pid,locs in index['entity_locators'].items()
                 if locs and any(l['kind']=='visual' for l in locs) and (allowed is None or index['entity_types'].get(pid) in allowed)]
    mapping,order=neutral_aliases(targets,key['logical_probe_id'],variant_id=key['variant_id'],seed=config('protocol_v2.json')['candidates']['selection_seed'])
    require(mapping==key['candidate_map'],'CANDIDATE_POOL_OR_ALIAS_MAPPING_MISMATCH')
    require([c['alias'] for c in view['candidates']]==order,'CANDIDATE_DISPLAY_ORDER_MISMATCH')

def source_truth(key, source_index):
    """Truth remains a direct projection of the grouped observational fields."""
    targets=[]
    for ref in key['source_records']:
        value=source_value(ref,source_index)
        if key['task_id']=='T01':
            require(isinstance(value,str),'ROLE_SOURCE_TYPE_INVALID')
            targets.append({'kind':'role','id':value})
        else:
            values=value if isinstance(value,list) else [value]
            targets.extend(canonical_entity(v,source_index) for v in values)
    dedup={digest(t):t for t in targets}
    require(set(dedup)=={digest(t) for t in key['gold_targets']},'GOLD_SOURCE_MISMATCH')


def closure_root(key, view, source_index):
    return digest({'task_id':key['task_id'],'source_records':key['source_records'],
        'source_index_hash':digest(source_index), 'query_anchor_ids':view['query_anchor_ids'],
        'query_anchors':{r:view['anchors'][r] for r in view['query_anchor_ids']},
        'reference_map':key['reference_map'], 'candidate_map':key['candidate_map'],
        'gold_targets':key['gold_targets'], 'relation_context':None if key['relation_context'] is None else
        {k:v for k,v in key['relation_context'].items() if k!='closure_review_ref'}})


def _relation(key, view, env, ledger):
    rc=key['relation_context']; registry=env['registry'];index=env['source_index']
    entries=[e for e in registry['relations'] if e['canonical_id']==rc['canonical_predicate']]
    require(len(entries)==1,'RELATION_ENTRY_MISSING');e=entries[0]
    require(rc['registry_entry_hash']==entry_root(e),'REGISTRY_ENTRY_HASH_MISMATCH')
    require(rc['missing_side'] in ('object','subject'),'MISSING_SIDE_INVALID')
    direction=e['queries'][rc['missing_side']]
    require(key['response_contract']['cardinality'] in direction['allowed_output'],'DIRECTION_CARDINALITY_INVALID')
    validate_direction_preconditions(direction,key['source_records'],index)
    qr=render_qualifiers(e,direction,key['source_records'],index,view['context'])
    require(qr==rc['public_qualifiers'],'QUALIFIER_SOURCE_OR_RENDER_MISMATCH')
    h=key['reference_map'][rc['fixed_reference_alias']]
    expected_h_refs=[]
    fixed_types=e['subject_types'] if rc['missing_side']=='object' else e['object_types']
    candidate_types=e['object_types'] if rc['missing_side']=='object' else e['subject_types']
    require(index['entity_types'].get(rc['fixed_target']['id']) in fixed_types,'FIXED_ENDPOINT_TYPE_MISMATCH')
    for target in key['candidate_map'].values():
        require(index['entity_types'].get(target['id']) in candidate_types,'CANDIDATE_ENDPOINT_DOMAIN_MISMATCH')
    for ref in key['source_records']:
        record=source_record(ref,index)
        resolved=resolve_relation(registry,index['namespace'],record['predicate'],ledger=ledger,production=env['production'],evidence_source_index=env.get('registry_evidence_index',index))
        require(resolved['entry']['canonical_id']==e['canonical_id'],'RELATION_SOURCE_ALIAS_MISMATCH')
        missing=rc['missing_side']; fixed='subject' if missing=='object' else 'object'
        if resolved['transform']=='swap': missing,fixed=fixed,missing
        if e['symmetric']:
            field = ref['target_field'].lstrip('/')
            require(field in ('subject_id', 'object_id'), 'RELATION_MISSING_SIDE_SOURCE_MISMATCH')
            missing = field.removesuffix('_id')
            fixed = 'subject' if missing == 'object' else 'object'
        require(ref['target_field'] in (missing+'_id','/'+missing+'_id'),'RELATION_MISSING_SIDE_SOURCE_MISMATCH')
        require(canonical_entity(record[fixed+'_id'],index)==rc['fixed_target'],'RELATION_FIXED_ENDPOINT_MISMATCH')
        expected_h_refs.append({**ref,'target_field':fixed+'_id'})
    # All contributing relations must be retained for the fixed side as well.
    canon=lambda refs:{digest({**r,'target_field':r['target_field'].lstrip('/')}) for r in refs}
    require(canon(h['source_records'])==canon(expected_h_refs),'H_SOURCE_COVERAGE')
    return render_relation_question(e,rc['missing_side'],key['response_contract']['cardinality'],rc['fixed_reference_alias'],qr,view['language'])


def binding_question(key, view, env, ledger=None):
    task=config('task_registry_v2.json')[key['task_id']]
    card=key['response_contract']['cardinality']
    require(key['response_contract']['binding_key']==task['binding_key'] and card in task['allowed_cardinalities'],'TASK_OUTPUT_CONTRACT')
    if key['task_id']=='T05': return _relation(key,view,env,ledger)
    require(key['task_id']!='T06','T06_DISABLED_IN_REFERENCE_CORE')
    require(len(view['query_anchor_ids'])==1,'CORE_QUERY_ANCHOR_ARITY')
    question=task['question'][view['language']]
    if isinstance(question,dict): question=question[card]
    return question.format(anchor=view['query_anchor_ids'][0])


def _public_locator_text(loc,context,alias):
    # Never expose source versions, private entity names, or source file paths.
    if loc['kind']=='visual':
        return f"{alias}: {loc['image_id']} {json.dumps(loc['geometry'],ensure_ascii=False)}"
    s,e=loc['start'],loc['end'];text=context[loc['section']]
    # Full prefix and suffix make identical repeated quotes unambiguous without
    # asking the model to count Unicode offsets. Escaping is textual, not HTML.
    return f"{alias} ({loc['section']}): " + text[:s] + f"[{alias}]" + text[s:e] + f"[/{alias}]" + text[e:]


def render_messages(view, question, contract, attachments, display_manifest):
    prompts=config('prompts_v2.json')[view['language']];context=view['context']
    anchors='\n'.join(_public_locator_text(l,context,a) for a,l in view['anchors'].items())
    refs='\n'.join(_public_locator_text(l,context,a) for a,ls in view['reference_views'].items() for l in ls)
    candidates='\n'.join(c['alias']+': '+c['label'] if c['kind']=='role' else
        '\n'.join(_public_locator_text(l,context,c['alias']) for l in c['locators']) for c in view['candidates'])
    textual=[]
    for a,l in view['anchors'].items():
        if l['kind']=='text': textual.append(_public_locator_text(l,context,a))
    legend=[]
    displays={d['output_asset_id']:d for d in display_manifest}
    for a in attachments:
        if a in displays:
            d=displays[a]; marks=', '.join(m['alias'] for m in d['marks'])
            legend.append(f"{a}: {d['source_image_id']}; {marks}")
        else: legend.append(a+': '+('original source image' if view['language']=='en' else '原图'))
    candidate_label='<candidate ID>' if view['language']=='en' else '<候选编号>'
    transcription_label='<transcription>' if view['language']=='en' else '<转写>'
    output={contract['binding_key']:candidate_label if contract['cardinality']=='one' else [candidate_label, '...']}
    if contract['read_shape']=='string': output={'read':transcription_label,**output}
    elif contract['read_shape']=='object_by_anchor': output={'read':{r:transcription_label for r in contract['read_anchor_ids']},**output}
    read=(prompts['read'].format(read_anchors=', '.join(contract['read_anchor_ids']))+'\n') if contract['read_shape']!='none' else ''
    packet=json.dumps(view['recognition_packet'],ensure_ascii=False) if view['recognition_packet'] else prompts['packet_empty']
    values={'stem':context['stem'],'query':context['query'],'options':json.dumps(context['options'],ensure_ascii=False),
        'attachment_legend':'\n'.join(legend),'text_locator_views':'\n'.join(textual),
        'anchor_block':anchors,'reference_block':refs,'candidate_block':candidates,
        'read_instruction':read,'binding_question':question,'selection_instruction':prompts[contract['cardinality']],
        'recognition_packet':packet,'output_contract':json.dumps(output,ensure_ascii=False)}
    return {'system':prompts['system'],'user':prompts['user'].format(**values)}


def validate_packet_selection(key, env):
    """Verify deterministic selected IDs; permission review is separate."""
    plan=key['packet_plan'];policy=key['packet_policy_id']
    if policy=='target_literal_only_v1':
        require(key['task_id']=='T03','PACKET_POLICY_FAMILY_MISMATCH')
        expected=[t['observation_id'] for t in key['read_targets']]
    elif policy=='all_atomic_glyphs_in_presented_diagrams_v1':
        from reference_checks import select_atomic_observations
        store={oid:{**o,'image_id':o['origin']['locator'].get('image_id')} for oid,o in env['observations']['observations'].items()}
        expected=select_atomic_observations(store,set(env['source_index']['context']['image_ids']),config('packet_selection_v2.json')[policy])['observation_ids']
    else:
        require(False,'PACKET_POLICY_UNREGISTERED')
    if plan['state'] in ('approved_nonempty', 'pending_review'):
        require(plan['observation_ids']==expected,'PACKET_SELECTION_POLICY_MISMATCH')
    elif plan['state']=='approved_empty':
        require(not plan['observation_ids'],'PACKET_STATE_CONTENT_MISMATCH')
        # Unsafe text can be rejected by a task-specific permission review even
        # if it passed the lexical selection rule. Missing generation is not it.
        if expected:
            require(plan['empty_reason']=='only_unsafe_binding_statement','EMPTY_PACKET_WITH_PERMITTED_SELECTION')
        else:
            require(plan['empty_reason'] in ('no_independent_readable_primitive','only_unsafe_binding_statement'),'EMPTY_REASON_POLICY_MISMATCH')

def prepare_probe(draft, env, *, verify_reviews=True):
    """Generate final prompts/key core from normalized input, or raise.

    verify_reviews=False is an AUDIT preview only: output is never frozen and
    cannot be scored/run by the public entry points. It still runs all structure,
    source-position, schema and registry-shape checks.
    """
    require(set(draft)=={'view','key'},'DRAFT_FIELDS')
    require(type(env['production']) is bool,'PRODUCTION_MODE_REQUIRED')
    require(not env['production'] or env['source_index']['example_only'] is False,'SYNTHETIC_SOURCE_NOT_FOR_RELEASE')
    # Object insertion order is not a JSON contract. Canonicalize before text
    # rendering so a save/load or group member's JSON formatter cannot change QA.
    key=json.loads(canonical_bytes(draft['key']));raw=json.loads(canonical_bytes(draft['view']))
    validate_schema('draft_mapping.schema.json',key)
    validate_schema('view_bundle.schema.json',raw)
    require(not any(k in key for k in ('instance_id','content_root','instance_review_refs')),'DRAFT_HAS_FROZEN_FIELDS')
    require(raw['recognition_packet']==[],'RAW_HAS_PACKET')
    require(raw['context']==env['source_index']['context'],'SOURCE_CONTEXT_MISMATCH')
    require(raw['language']==key['language']==env['source_index']['language'],'SOURCE_LANGUAGE_MISMATCH')
    require(key['task_id']==raw['task_id'],'TASK_MISMATCH')
    require(key['schema_version']=='physalign_private_v2_1','PRIVATE_VERSION_MISMATCH')
    verify_source_snapshots(env['source_files'],env['source_index'],key['source_hashes'],env['asset_root'])
    require(bool(key.get('variant_id')),'VARIANT_ID_REQUIRED')
    variant=key['task_id']+'-'+key['response_contract']['cardinality']
    require(variant in env['enabled_variants'],'TASK_VARIANT_NOT_ENABLED')
    require(not env['production'] or set(env['enabled_variants'])<=set(config('protocol_v2.json')['core_variants']),'EXTENSION_REQUIRES_SEPARATE_RELEASE')
    ledger=ReviewLedger(env['ledger'],production=env['production']) if verify_reviews else None
    validate_registry(env['registry'],ledger,production=env['production'],evidence_source_index=env.get('registry_evidence_index',env['source_index']))
    # Derive both read truth and packet from the SAME source observation+origin.
    for t in key['read_targets']:
        require(t['observation_id'] in env['observations']['observations'],'OBSERVATION_MISSING')
        obs=env['observations']['observations'][t['observation_id']]
        derived={'expected':obs['text'],'origin_fingerprint':origin_fingerprint(obs['origin'])}
        for name,value in derived.items():
            require(name not in t or t[name]==value,'READ_TRUTH_OVERRIDE')
            t[name]=value
    if verify_reviews:
        validate_observation_sources(env['observations'],env['source_index'],ledger)
    observations=env['observations']['observations']
    validate_packet_selection(key, env)
    obs_alias=validate_observation_bindings(env['observations'],raw,env['assets'],key['anchor_map'],key['read_targets'])
    packet=build_packet(key['packet_plan'],observations,obs_alias,public_bundle=raw,assets=env['assets'],anchor_map=key['anchor_map'], preview=not verify_reviews)
    if verify_reviews:
        ledger.require_review(key['packet_plan']['review_ref'],'packet',packet_review_root(key['packet_plan'],observations,key['packet_policy_id']))
    gold=copy.deepcopy(raw);gold['recognition_packet']=packet
    role_labels=None
    if key['task_id']=='T01':
        roles={r['canonical_id']:r for r in env['registry']['roles'] if r['status']=='approved'}
        require(all(t['id'] in roles for t in key['candidate_map'].values()),'ROLE_UNREGISTERED')
        role_labels={a:roles[t['id']]['label'][key['language']] for a,t in key['candidate_map'].items()}
    for view in (raw,gold):
        validate_schema('view_bundle.schema.json',view)
        validate_view_bundle(view,env['assets'],role_alias_labels=role_labels)
    validate_pair(raw,gold,packet)
    validate_contract(key)
    for t in key['read_targets']:
        obs = observations[t['observation_id']]
        require(raw['anchors'][t['anchor_alias']]['kind']=='visual' and obs['asset_role']=='diagram' and obs['source_kind']=='text_glyph', 'MAIN_JOINT_REQUIRES_IMAGE_READING')
    require(key['response_contract']['read_anchor_ids']==raw['read_anchor_ids'],'PUBLIC_READ_CONTRACT_MISMATCH')
    validate_private_public_maps(key,raw,env['assets'],env['source_index'])
    validate_candidate_generation(key,raw,env)
    source_truth(key,env['source_index'])
    if 'native_profile' in env:
        from native_compiler import validate_native_query
        validate_native_query({'view': raw, 'key': key}, env)
    question=binding_question(key,raw,env,ledger)
    validate_generated_question_references(question,raw['reference_views'])
    cr=closure_root(key,raw,env['source_index'])
    require(bool(key['closure_review_refs']),'CLOSURE_REVIEW_REQUIRED')
    if key['task_id']=='T05':
        require(key['relation_context']['closure_review_ref'] in key['closure_review_refs'],'DIRECTION_CLOSURE_REF_MISMATCH')
    if verify_reviews:
        for rid in key['closure_review_refs']: ledger.require_review(rid,'query_closure',cr)
    required=validate_render_manifest(raw,env['assets'],env['displays'])
    require(set(env['assets'])==required,'UNDECLARED_OR_UNUSED_ASSET')
    # Always inspect the REAL local files; a dimensions-only asset map is not a
    # complete instance. This also binds actual displayed output bytes.
    verify_image_assets(env['assets'],Path(env['asset_root']))
    attachments=sorted(required)
    inputs={}
    for condition,view in (('raw',raw),('gold',gold)):
        inputs[condition]={'messages':render_messages(view,question,key['response_contract'],attachments,env['displays']),
                          'attachment_ids':attachments}
    placeholder={**key,'instance_id':'UNFROZEN_PREVIEW','content_root':'0'*64,'instance_review_refs':['UNFROZEN_PREVIEW']}
    validate_schema('private_mapping.schema.json',placeholder)
    core={'private_core':key_core(key),'raw_view':raw,'gold_view':gold,'public_raw':inputs['raw'],'public_gold':inputs['gold'],
          'assets':env['assets'],'displays':env['displays'],'source_index_hash':digest(env['source_index']),
          'observation_store_hash':digest(env['observations']),'semantic_registry_hash':digest(env['registry']),
          'implementation_hashes':dependency_hashes(),'source_index_is_synthetic':env['source_index']['example_only'],
          'renderer_profile':env['renderer_profile']}
    if 'native_profile' in env:
        core['native_projection'] = copy.deepcopy(env['native_profile'])
        core['native_query'] = copy.deepcopy(env['native_query'])
    return {'core':core,'content_root':digest(core),'closure_root':cr,'packet_review_root':packet_review_root(key['packet_plan'],observations,key['packet_policy_id']),
            'review_checks_completed':verify_reviews}


def freeze_probe(draft, env, instance_review_refs):
    require('native_profile' not in env, 'NATIVE_DRAFT_ONLY_BATCH_AUDIT_NOT_IMPLEMENTED')
    prepared=prepare_probe(draft,env,verify_reviews=True)
    require(isinstance(instance_review_refs,list) and bool(instance_review_refs) and len(set(instance_review_refs))==len(instance_review_refs),'INSTANCE_REVIEW_REFS_REQUIRED')
    ledger=ReviewLedger(env['ledger'],production=env['production'])
    reviews=[ledger.require_review(r,'instance',prepared['content_root']) for r in instance_review_refs]
    iid='PAI-'+digest({'content_root':prepared['content_root'],'reviews':reviews})
    key={**prepared['core']['private_core'],'content_root':prepared['content_root'],'instance_id':iid,'instance_review_refs':instance_review_refs}
    validate_schema('private_mapping.schema.json',key)
    envlp={'schema_version':'physalign_envelope_v2_1','logical_probe_id':key['logical_probe_id'],'variant_id':key['variant_id'],'instance_id':iid,
        'language':key['language'],'content_root':prepared['content_root'],
        'public_raw_hash':digest(prepared['core']['public_raw']),'public_gold_hash':digest(prepared['core']['public_gold']),'private_hash':digest(key)}
    validate_schema('instance_envelope.schema.json',envlp)
    return {'envelope':envlp,'private':key,'core':prepared['core'],'reviews':reviews,
            'artifact_mode':'production' if env['production'] else 'synthetic_example_not_benchmark_data'}


def validate_frozen(frozen, env):
    validate_schema('instance_envelope.schema.json',frozen['envelope'])
    validate_schema('private_mapping.schema.json',frozen['private'])
    key=frozen['private'];envelope=frozen['envelope'];core=frozen['core']
    for field in ('logical_probe_id','variant_id','instance_id','language','content_root'):
        require(envelope[field]==key[field],'ENVELOPE_KEY_ID_MISMATCH',field)
    require(envelope['private_hash']==digest(key),'PRIVATE_HASH_MISMATCH')
    require(core['private_core']==key_core(key),'PRIVATE_CORE_MISMATCH')
    require(digest(core)==key['content_root'],'CONTENT_ROOT_MISMATCH')
    require(envelope['public_raw_hash']==digest(core['public_raw']) and envelope['public_gold_hash']==digest(core['public_gold']),'PUBLIC_HASH_MISMATCH')
    draft={'view':core['raw_view'],'key':key_core(key)}
    regenerated=freeze_probe(draft,env,key['instance_review_refs'])
    require(regenerated==frozen,'FROZEN_REGENERATION_MISMATCH')


def validate_release_instances(frozen_items):
    validate_batch_identity([x['private'] for x in frozen_items])
    ids=[x['envelope']['instance_id'] for x in frozen_items]
    require(len(ids)==len(set(ids)),'DUPLICATE_INSTANCE_ID')
    for f in frozen_items:
        for field in ('logical_probe_id','variant_id','instance_id','language','content_root'):
            require(f['private'][field]==f['envelope'][field],'ENVELOPE_KEY_ID_MISMATCH')
