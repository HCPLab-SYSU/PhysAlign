"""Explicit local reviewer decisions. A preview is never an approval record."""
import copy
from pathlib import Path
from reference_checks import require, digest
from reference_pipeline import prepare_probe, freeze_probe
from io_utils import review_record

CONFIRMATIONS = ('source_semantics', 'complete_targets', 'distinct_candidates',
                 'rendered_locations', 'packet_no_binding_leakage', 'diagram_asset_scope')


def prospective(draft, env):
    """Compute the *prospective* frozen core without creating any approval ledger."""
    candidate = copy.deepcopy(draft)
    plan = candidate['key']['packet_plan']
    require(plan['state'] == 'pending_review', 'NOT_PENDING_DRAFT')
    plan['state'] = 'approved_nonempty' if plan['observation_ids'] else 'approved_empty'
    plan['empty_reason'] = None if plan['observation_ids'] else 'no_independent_readable_primitive'
    plan['review_ref'] = 'qa-packet'
    return candidate, prepare_probe(candidate, env, verify_reviews=False)


def apply_decision(draft, env, decision):
    candidate, prepared = prospective(draft, env)
    require(set(decision) == {'logical_probe_id', 'review_target_root', 'status', 'reviewer',
                             'blind_answer', 'confirmations', 'note'}, 'DECISION_FIELDS')
    require(decision['logical_probe_id'] == draft['key']['logical_probe_id'], 'DECISION_PROBE_MISMATCH')
    require(decision['review_target_root'] == prepared['content_root'], 'DECISION_STALE')
    require(decision['status'] == 'approved', 'DECISION_NOT_APPROVED')
    require(isinstance(decision['note'], str), 'REVIEW_NOTE_INVALID')
    require(isinstance(decision['blind_answer'], str) and decision['blind_answer'].strip(), 'BLIND_ANSWER_REQUIRED')
    require(set(decision['confirmations']) == set(CONFIRMATIONS) and
            all(v is True for v in decision['confirmations'].values()), 'REVIEW_CONFIRMATIONS_REQUIRED')
    final_env = copy.deepcopy(env)
    evidence = ['local-human-decision:' + digest(decision), 'review-target:' + prepared['content_root']]
    for rid, kind, root in (('qa-packet', 'packet', prepared['packet_review_root']),
                            ('qa-closure', 'query_closure', prepared['closure_root']),
                            ('qa-instance', 'instance', prepared['content_root'])):
        require(rid not in final_env['ledger']['records'], 'QA_REVIEW_ID_COLLISION')
        final_env['ledger']['records'][rid] = review_record(rid, kind, root, decision['reviewer'], evidence)
    return freeze_probe(candidate, final_env, ['qa-instance']), final_env


def write_review_page(root, catalog, problem_ids=None):
    """Self-contained local file; no HTTP server, no network/CDN, no source edits."""
    from io_utils import read_json
    payload = []
    for item in catalog['probes']:
        if problem_ids is not None and item['problem_id'] not in problem_ids:
            continue
        preview = read_json(Path(root) / item['preview'])
        payload.append({**item, 'core': preview['core'],
                        'audit_url': item['asset_root'] + '/source_audit.json',
                        'source_url': item['asset_root'] + '/sources/pass4.json'})
    import json
    data = json.dumps({'catalog_root': digest(catalog), 'items': payload,
                       'native_profile': catalog.get('native_profile'),
                       'confirmations': list(CONFIRMATIONS)}, ensure_ascii=False)
    data = data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (Path(__file__).parent / 'review.html').read_text(encoding='utf-8')
    (Path(root) / 'review.html').write_text(template.replace('__REVIEW_DATA__', data), encoding='utf-8')
