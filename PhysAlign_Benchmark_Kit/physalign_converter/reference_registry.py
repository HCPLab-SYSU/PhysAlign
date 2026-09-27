"""Typed, exact, direction-specific semantic registration for small registries."""
from __future__ import annotations
import copy
from string import Formatter
from reference_checks import require, digest
from reference_integrity import resolve_pointer, source_record, source_value
from schema_tools import validate_schema


def entry_root(entry):
    return digest({k:v for k,v in entry.items() if k != 'review_ref'})


def placeholders(template):
    result=set()
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None:
            require(name.isidentifier() and not spec and conversion is None,'UNSAFE_TEMPLATE_PLACEHOLDER')
            result.add(name)
    return result


def validate_registry(registry, ledger=None, *, production=True, evidence_source_index=None):
    validate_schema('semantic_registry.schema.json',registry)
    require(not production or registry['example_only'] is False, 'EXAMPLE_REGISTRY_NOT_FOR_RELEASE')
    seen_ids=set(); aliases=set()
    for role in registry['roles']:
        require(('role',role['canonical_id']) not in seen_ids,'DUPLICATE_REGISTRY_ID');seen_ids.add(('role',role['canonical_id']))
        if role['status']=='approved' and ledger is not None:
            ledger.require_review(role['review_ref'],'semantic_entry',entry_root(role))
    for e in registry['relations']:
        require(('relation',e['canonical_id']) not in seen_ids,'DUPLICATE_REGISTRY_ID');seen_ids.add(('relation',e['canonical_id']))
        if e['status']=='approved' and ledger is not None:
            ledger.require_review(e['review_ref'],'semantic_entry',entry_root(e))
        qs={q['qualifier_id']:q for q in e['public_qualifiers']}
        require(len(qs)==len(e['public_qualifiers']),'DUPLICATE_QUALIFIER_ID')
        for q in qs.values():
            require(q['qualifier_id'].isidentifier(),'QUALIFIER_ID_INVALID')
            require(bool(q['enum_labels']) == (q['value_type']=='enum'),'QUALIFIER_ENUM_TABLE_INVALID')
            for t in q['display_template'].values():
                require(placeholders(t)=={'value'},'QUALIFIER_TEMPLATE_INVALID')
        for side,d in e['queries'].items():
            require(set(d['required_qualifier_ids'])<=set(qs),'UNKNOWN_REQUIRED_QUALIFIER')
            allowed={'reference','qualifiers'}
            for localized in d['question'].values():
                for template in localized.values():
                    ph=placeholders(template)
                    require('reference' in ph and ph<=allowed,'RELATION_TEMPLATE_INVALID')
                    require(not qs or 'qualifiers' in ph,'QUALIFIERS_NOT_DISPLAYED')
            if d['closure_requirement']=='registered_unique_semantics_with_source_evidence':
                require(d['cardinality']=='contextually_single' and bool(d['uniqueness_preconditions']), 'UNIQUE_RULE_WITHOUT_PRECONDITIONS')
            if d['cardinality']=='contextually_single':
                require(d['allowed_output']==['one'],'UNIQUE_DIRECTION_OUTPUT_CONTRACT')
        require({x['polarity'] for x in e['examples']}=={'positive','negative'},'SEMANTIC_EXAMPLES_INCOMPLETE')
        for x in e['examples']:
            require(x['basis']!='source' or bool(x['source_records']),'SOURCE_EXAMPLE_MISSING_POINTERS')
            if x['basis']=='synthetic':
                require(not x['source_records'],'SYNTHETIC_EXAMPLE_HAS_SOURCE_CLAIM')
                payload={'basis':'synthetic','description':x['description']}
            else:
                require(evidence_source_index is not None,'SEMANTIC_EXAMPLE_SOURCE_INDEX_REQUIRED')
                payload={'basis':'source','source_records':x['source_records'],
                         'resolved_values':[source_value(r,evidence_source_index) for r in x['source_records']]}
            require(x['evidence_snapshot_hash']==digest(payload),'SEMANTIC_EXAMPLE_EVIDENCE_HASH_MISMATCH')
        if e['symmetric']:
            require(set(e['subject_types'])==set(e['object_types']),'SYMMETRIC_ENDPOINT_DOMAINS')
            for f in ('cardinality','allowed_output','required_qualifier_ids','uniqueness_preconditions','closure_requirement'):
                require(e['queries']['object'][f]==e['queries']['subject'][f],'SYMMETRIC_DIRECTION_CONTRADICTION')
        for a in e['aliases']:
            key=(a['source_namespace'],a['raw_predicate'])
            require(key not in aliases,'UNREGISTERED_OR_AMBIGUOUS_PREDICATE');aliases.add(key)
    return registry


def resolve_relation(registry, namespace, raw_predicate, *, ledger=None, production=True, evidence_source_index=None):
    validate_registry(registry,ledger,production=production,evidence_source_index=evidence_source_index)
    found=[]
    for entry in registry['relations']:
        if entry['status']!='approved': continue
        for a in entry['aliases']:
            if (a['source_namespace'],a['raw_predicate'])==(namespace,raw_predicate):
                found.append({'entry':entry,'transform':a['endpoint_transform']})
    require(len(found)==1,'UNREGISTERED_OR_AMBIGUOUS_PREDICATE')
    return found[0]


def render_qualifiers(entry, direction, source_refs, source_index, context):
    """No caller-supplied arbitrary qualifier values or free expression string."""
    result=[]
    for q in entry['public_qualifiers']:
        values=[]; used=[]
        for ref in source_refs:
            if ref['collection']!=q['source']['collection']: continue
            record=source_record(ref,source_index)
            # Missing optional qualifier is allowed, but a partial grouped value
            # is NOT filled in from one of several records.
            try:
                value=resolve_pointer(record,q['source']['field_pointer'])
            except ValueError:
                value=None
            values.append(value)
            used.append({**ref,'target_field':q['source']['field_pointer']})
        required=q['required'] or q['qualifier_id'] in direction['required_qualifier_ids']
        if not values or all(v is None for v in values):
            require(not required,'REQUIRED_QUALIFIER_MISSING',q['qualifier_id']);continue
        require(all(digest(v)==digest(values[0]) for v in values),'GROUP_QUALIFIER_INCONSISTENT')
        v=values[0]; vt=q['value_type']
        if vt=='enum':
            require(isinstance(v,str) and v in q['enum_labels'],'QUALIFIER_ENUM_INVALID')
            label=q['enum_labels'][v]
        elif vt in ('integer','number'):
            import math
            require((type(v) is int if vt=='integer' else type(v) in (int,float) and math.isfinite(v)),'QUALIFIER_TYPE_INVALID')
            label={'en':str(v),'zh':str(v)}
        else:
            from reference_checks import validate_locator
            require(isinstance(v,dict) and v.get('kind')=='text','QUALIFIER_TEXT_SPAN_REQUIRED')
            validate_locator(v,context,{})
            label={'en':v['quote'],'zh':v['quote']}
        rendered={lang:q['display_template'][lang].format(value=label[lang]) for lang in ('en','zh')}
        result.append({'qualifier_id':q['qualifier_id'],'value':copy.deepcopy(v),'source_records':used,'rendered':rendered})
    return result


def validate_direction_preconditions(direction, source_refs, source_index):
    if direction['closure_requirement']!='registered_unique_semantics_with_source_evidence': return
    for p in direction['uniqueness_preconditions']:
        values=[]
        for ref in source_refs:
            if ref['collection']==p['source']['collection']:
                try: v=resolve_pointer(source_record(ref,source_index),p['source']['field_pointer'])
                except ValueError: v=None
                values.append(v)
        require(bool(values),'UNIQUENESS_PRECONDITION_MISSING')
        if p['operator']=='equals':
            require(all(digest(v)==digest(p['value']) for v in values),'UNIQUENESS_PRECONDITION_FALSE')
        else:
            require(all(v is not None for v in values),'UNIQUENESS_PRECONDITION_FALSE')


def render_relation_question(entry, missing_side, cardinality, reference_alias, qualifiers, language):
    require(missing_side in ('subject','object'),'MISSING_SIDE_INVALID')
    d=entry['queries'][missing_side]
    require(cardinality in d['allowed_output'],'DIRECTION_CARDINALITY_INVALID')
    text='; '.join(q['rendered'][language] for q in qualifiers)
    return d['question'][language][cardinality].format(reference=reference_alias,qualifiers=text).strip()

