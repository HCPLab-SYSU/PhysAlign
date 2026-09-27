"""Review ingestion, finite-population audit, and portable native DATA release.

The original converter stays byte-identical. This is a separate, explicitly
versioned authorization layer; it never calls the legacy native-forbidden freeze.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import random
import secrets
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

from native_release_support import (KIT, VERSION, POLICY, SIGNOFF_CHECKS, SERIOUS, exact, nonempty,
    implementation_hashes, approval_template, check_signoff, normalize_reviews, merge_reviews,
    mother_groups, review_summary, risk_tags, audit_statistics)
from io_utils import read_json, write_json, safe_path, sha_bytes, verify_directory, review_record, file_manifest, seal_directory
from reference_checks import require, digest
from convert import new_output, load_probe, verify_descriptor
from adapter import verify_adapter_snapshot
from native_run import validate_native_output
from reference_pipeline import dependency_hashes, prepare_probe, validate_release_instances, schema_bundle
from review_workflow import prospective, CONFIRMATIONS
from schema_tools import validate_schema


def publish(staging, final):
    """Publish a new immutable output; bounded retry for Windows reader locks.

    Never delete/replace an existing destination and never downgrade to a
    non-atomic overwrite after a rename failure.
    """
    require(not final.exists(), 'OUTPUT_EXISTS_USE_NEW_DIRECTORY')
    seal_directory(staging)
    for attempt in range(8):
        require(not final.exists(), 'OUTPUT_EXISTS_USE_NEW_DIRECTORY')
        try:
            staging.rename(final)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 1.0))


def check_output(path, *protected):
    out = Path(path).resolve()
    for root in (KIT, *protected):
        root = Path(root).resolve()
        require(out != root and not out.is_relative_to(root), 'OUTPUT_MUST_BE_OUTSIDE_INPUT_AND_KIT')
    require(not out.exists(), 'OUTPUT_EXISTS_USE_NEW_DIRECTORY')


def copy_tree(source, target):
    """Only declared, hash-checked files. Never follow symlinks/junctions."""
    source, target = Path(source).resolve(), Path(target).resolve()
    require(source != target and not target.is_relative_to(source), 'COPY_INSIDE_SOURCE')
    for p in source.rglob('*'):
        require(not p.is_symlink() and not (getattr(p.stat(follow_symlinks=False), 'st_file_attributes', 0) & 0x400),
                'REPARSE_POINT_NOT_ALLOWED')
    verify_directory(source)
    expected = read_json(source / 'FILE_MANIFEST.json')
    require(not target.exists(), 'COPY_TARGET_EXISTS')
    target.mkdir(parents=True)
    for rel in [*sorted(expected), 'FILE_MANIFEST.json']:
        src = safe_path(source, rel)
        dst = safe_path(target, rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    verify_directory(source)
    verify_directory(target)
    require(expected == read_json(target / 'FILE_MANIFEST.json'), 'SOURCE_CHANGED_WHILE_COPYING')


def load_project(path, *, deep=False):
    root = Path(path).resolve()
    verify_directory(root)
    p = read_json(root / 'project.json')
    require(p['schema_version'] == 'native_review_project_v1', 'PROJECT_SCHEMA')
    require(p['release_implementation'] == implementation_hashes(), 'RELEASE_TOOLS_VERSION_CHANGED')
    require(p['project_id'] == 'NRP-' + digest({k: v for k, v in p.items() if k != 'project_id'}), 'PROJECT_ROOT_CHANGED')
    source = root / 'source'
    verify_directory(source)
    catalog = read_json(source / 'catalog.PRIVATE.json')
    require(p['source_catalog_root'] == digest(catalog), 'PROJECT_SOURCE_CATALOG_CHANGED')
    require(p['source_manifest_root'] == digest(read_json(source / 'FILE_MANIFEST.json')), 'PROJECT_SOURCE_BYTES_CHANGED')
    require(catalog['implementation_hashes'] == dependency_hashes(), 'SOURCE_CONVERTER_VERSION_CHANGED')
    require(catalog.get('native_profile') == 'native_binding_draft_v1', 'NATIVE_SOURCE_REQUIRED')
    if deep:
        validate_native_output(SimpleNamespace(path=source))
    return root, p, catalog


def review_page(dest, catalog, source, *, context, role='individual', mother_ids=None, blind_first=None, required=None):
    groups = mother_groups(catalog)
    ids = list(groups) if mother_ids is None else list(mother_ids)
    byid = {p['logical_probe_id']: p for p in catalog['probes']}
    payload = []
    base = os.path.relpath(source, Path(dest).parent).replace('\\', '/')
    for pid in ids:
        qids = groups[pid]
        if blind_first:
            qids = [blind_first[pid], *[q for q in qids if q != blind_first[pid]]]
        for qid in qids:
            item = byid[qid]
            payload.append({**item, 'core': read_json(safe_path(source, item['preview']))['core'],
                            'asset_base': base + '/' + item['asset_root']})
    data = {'catalog_root': digest(catalog), 'context_root': context, 'role': role,
            'items': payload, 'confirmations': list(CONFIRMATIONS), 'blind_first': blind_first or {},
            'required_mothers': list(required or ids)}
    encoded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (KIT / 'tools/native_release_review.html').read_text(encoding='utf-8')
    Path(dest).write_text(template.replace('__DATA__', encoded), encoding='utf-8')


def project_init(draft, output):
    source = Path(draft).resolve()
    check_output(output, source)
    validate_native_output(SimpleNamespace(path=source))
    final, stage = new_output(output)
    copy_tree(source, stage / 'source')
    catalog = read_json(stage / 'source/catalog.PRIVATE.json')
    p = {'schema_version': 'native_review_project_v1', 'release_tools_version': VERSION,
         'release_implementation': implementation_hashes(), 'source_catalog_root': digest(catalog),
         'source_manifest_root': digest(read_json(stage / 'source/FILE_MANIFEST.json'))}
    p['project_id'] = 'NRP-' + digest(p)
    write_json(stage / 'project.json', p)
    review_page(stage / 'review.html', catalog, stage / 'source', context=p['project_id'])
    write_json(stage / 'decisions.TEMPLATE.json', {'schema_version': 'native_release_reviews_v1',
        'catalog_root': digest(catalog), 'context_root': p['project_id'], 'role': 'individual', 'decisions': []})
    publish(stage, final)
    print(json.dumps({'project': str(final), 'project_id': p['project_id'], 'source_unchanged': True,
                      'next': 'Open review.html; import old development decisions; export outside this directory.'}, ensure_ascii=False))


def read_receipt(path, project, catalog):
    root = Path(path).resolve()
    verify_directory(root)
    r = read_json(root / 'receipt.json')
    require(r['project_id'] == project['project_id'], 'RECEIPT_PROJECT_STALE')
    merged = merge_reviews(r['original_documents'], catalog, project['project_id'])
    require(merged == r['decisions'] and r['summary'] == review_summary(catalog, merged), 'RECEIPT_RECOMPUTATION_FAILED')
    require(r['receipt_id'] == 'NRI-' + digest({k: v for k, v in r.items() if k != 'receipt_id'}), 'RECEIPT_CHANGED')
    return r


def import_review(project_path, documents, output):
    root, p, catalog = load_project(project_path)
    check_output(output, root)
    docs = [read_json(d) for d in documents]
    merged = merge_reviews(docs, catalog, p['project_id'])
    r = {'schema_version': 'native_review_receipt_v1', 'project_id': p['project_id'],
         'original_documents': docs, 'decisions': merged, 'summary': review_summary(catalog, merged)}
    r['receipt_id'] = 'NRI-' + digest(r)
    final, stage = new_output(output)
    write_json(stage / 'receipt.json', r)
    publish(stage, final)
    print(json.dumps({'receipt': str(final), **r['summary'], 'source_unchanged': True}, ensure_ascii=False))


def campaign_init(output):
    check_output(output)
    path = Path(output).resolve()
    path.mkdir(parents=True)
    header = {'schema_version': 'native_audit_campaign_v1', 'campaign_nonce': secrets.token_hex(32),
              'policy': POLICY, 'release_implementation': implementation_hashes()}
    header['campaign_id'] = 'NAC-' + digest(header)
    write_json(path / 'campaign.json', header)
    write_json(path / 'policy.TEMPLATE.json', approval_template(digest(header)))
    print('Copy policy.TEMPLATE.json OUTSIDE the campaign; obtain a real team signoff. Never restart campaigns to reroll samples.')


def campaign_header(path):
    c = read_json(Path(path) / 'campaign.json')
    require(c['campaign_id'] == 'NAC-' + digest({k: v for k, v in c.items() if k != 'campaign_id'})
            and c['policy'] == POLICY and c['release_implementation'] == implementation_hashes(), 'CAMPAIGN_CHANGED')
    return c


def make_audit_plan(project, catalog, membership, campaign, signoff, development, number, previous):
    require(campaign['policy'] == POLICY and campaign['release_implementation'] == implementation_hashes()
            and campaign['campaign_id'] == 'NAC-' + digest({k: v for k, v in campaign.items() if k != 'campaign_id'}), 'CAMPAIGN_CHANGED')
    require(project['project_id'] == 'NRP-' + digest({k: v for k, v in project.items() if k != 'project_id'}), 'PROJECT_CHANGED')
    require(project['source_catalog_root'] == digest(catalog) and catalog['membership_root'] == membership['membership_root']
            and membership['membership_root'] == digest({k: v for k, v in membership.items() if k != 'membership_root'}),
            'AUDIT_SOURCE_MEMBERSHIP_CHANGED')
    require(development['project_id'] == project['project_id'], 'DEVELOPMENT_PROJECT_STALE')
    dev_decisions = merge_reviews(development['original_documents'], catalog, project['project_id'])
    require(development['decisions'] == dev_decisions and development['summary'] == review_summary(catalog, dev_decisions)
            and development['receipt_id'] == 'NRI-' + digest({k: v for k, v in development.items() if k != 'receipt_id'}), 'DEVELOPMENT_RECEIPT_CHANGED')
    check_signoff(signoff, digest(campaign))
    require(number in (1, 2), 'AT_MOST_TWO_AUDIT_ROUNDS')
    dev = set(membership['development_problem_ids'])
    groups = mother_groups(catalog)
    require(dev <= set(development['summary']['approved_complete_mothers']), 'DEVELOPMENT_REVIEW_INCOMPLETE')
    require(not any(s in ('systemic', 'information', 'unresolved') for ss in
                    development['summary']['defective_or_unresolved_mothers'].values() for s in ss),
            'DEVELOPMENT_HAS_UNRESOLVED_RULE_DEFECT')
    candidates = sorted(set(groups) - dev)
    require(candidates, 'NO_TEST_CANDIDATE_MOTHERS_DEVELOPMENT_STAYS_RESERVED')
    # No user-selectable seed and no resampling within one round.
    seed = digest([campaign['campaign_nonce'], number, 'fixed_simple_random_without_replacement'])
    rng = random.Random(int(seed, 16))
    sample = sorted(rng.sample(candidates, min(POLICY['sample_mothers'], len(candidates))))
    secondary = sorted(rng.sample(sample, min(POLICY['secondary_mothers'], len(sample))))
    tags = {pid: set() for pid in candidates}
    for item in catalog['probes']:
        if item['problem_id'] in tags:
            tags[item['problem_id']].update(risk_tags(item))
    all_tags = set().union(*tags.values())
    covered = set().union(*(tags[pid] for pid in sample))
    risk = []
    remaining = set(candidates) - set(sample)
    while all_tags - covered and remaining and len(risk) < POLICY['risk_extra_budget']:
        pid = min(remaining, key=lambda pid: (-len(tags[pid] - covered), digest([seed, 'risk', pid])))
        remaining.remove(pid)
        if not tags[pid] - covered:
            break
        risk.append(pid)
        covered.update(tags[pid])
    first = {pid: min(groups[pid], key=lambda q: digest([seed, 'blind-first', q])) for pid in sample + risk}
    plan = {'schema_version': 'native_audit_plan_v1', 'project_id': project['project_id'],
            'source_project': project, 'source_membership': membership,
            'catalog_root': digest(catalog), 'campaign': campaign, 'policy_signoff': signoff,
            'round_number': number, 'previous_assessment_root': previous,
            'development_receipt': development, 'candidate_mothers': candidates,
            'sample_mothers': sample, 'secondary_mothers': secondary, 'risk_mothers': risk,
            'uncovered_risk_tags': sorted(all_tags - covered), 'blind_first': first,
            'release_implementation': implementation_hashes()}
    plan['audit_id'] = 'NAP-' + digest(plan)
    return plan


def check_retry_revision(previous, project, catalog):
    """Known failures cannot disappear merely because round 2 misses them.

    Require a NEW candidate version. Previously flagged mothers must be absent
    or have changed reviewed content. Systemic/info findings expand to the
    whole affected selector/packet rule, not just sampled bad probes.
    A change is not proof of correctness: the new audit/signoff is still needed.
    """
    require(project['project_id'] != previous['plan']['project_id'], 'RETRY_REQUIRES_NEW_CANDIDATE_VERSION_NOT_SEED_REROLL')
    old_catalog = previous['catalog']
    old_items = {p['logical_probe_id']: p for p in old_catalog['probes']}
    known = set(previous['result']['confirmed_excluded_mothers'])
    affected_rules = set()
    def rule(p):
        m = p['metadata']
        return (m['interface'], m['selector'], m['packet_policy_id'])
    for name in ('primary_document', 'secondary_document', 'resolution_document'):
        for d in (previous[name] or {}).get('decisions', []):
            if d['status'] == 'rejected':
                item = old_items[d['logical_probe_id']]
                known.add(item['problem_id'])
                if d['severity'] in ('systemic', 'information'):
                    affected_rules.add(rule(item))
    old_roots, new_roots = {}, {}
    for item in old_catalog['probes']:
        old_roots.setdefault(item['problem_id'], set()).add(item['review_target_root'])
        if rule(item) in affected_rules:
            known.add(item['problem_id'])
    for item in catalog['probes']:
        new_roots.setdefault(item['problem_id'], set()).add(item['review_target_root'])
    for pid in known & set(new_roots):
        require(old_roots[pid] != new_roots[pid], 'KNOWN_DEFECT_OR_AFFECTED_RULE_UNCHANGED_IN_RETRY', pid)


def load_audit(path, project, catalog, membership):
    root = Path(path).resolve()
    verify_directory(root)
    plan = read_json(root / 'audit_plan.json')
    require(plan['project_id'] == project['project_id'], 'AUDIT_PROJECT_STALE')
    expected = make_audit_plan(project, catalog, membership, plan['campaign'], plan['policy_signoff'],
                               plan['development_receipt'], plan['round_number'], plan['previous_assessment_root'])
    require(expected == plan, 'AUDIT_PLAN_RECOMPUTATION_FAILED')
    return plan


def audit_plan(project_path, campaign_path, policy_path, development_path, previous_path=None):
    root, p, catalog = load_project(project_path)
    campaign = Path(campaign_path).resolve()
    header = campaign_header(campaign)
    signoff = read_json(policy_path)
    check_signoff(signoff, digest(header))
    membership = read_json(root / 'source/membership.json')
    development = read_receipt(development_path, p, catalog)
    # One writer per campaign; do not remove another process's lock.
    lock = campaign / '.planning.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.close(fd)
        number = 2 if (campaign / 'round1').exists() else 1
        require(not (campaign / 'round2').exists(), 'AT_MOST_TWO_AUDIT_ROUNDS')
        prior = None
        if number == 2:
            require(previous_path is not None, 'SECOND_ROUND_REQUIRES_PREVIOUS_ASSESSMENT')
            prior = read_assessment(previous_path)
            previous_plan = read_json(campaign / 'round1/audit_plan.json')
            verify_directory(campaign / 'round1')
            require(prior['plan'] == previous_plan and not prior['result']['accepted'], 'PREVIOUS_ROUND_NOT_FAILED')
            require(prior['result']['review_complete'] or prior['result']['systemic_probe_ids'], 'FINISH_PREVIOUS_AUDIT_FIRST')
            require(previous_plan['policy_signoff'] == signoff, 'CAMPAIGN_POLICY_CHANGED')
            check_retry_revision(prior, p, catalog)
        else:
            require(previous_path is None, 'UNEXPECTED_PREVIOUS_ASSESSMENT')
        validate_native_output(SimpleNamespace(path=root / 'source'))
        plan = make_audit_plan(p, catalog, membership, header, signoff, development, number, digest(prior) if prior else None)
        final, stage = new_output(campaign / ('round' + str(number)))
        # Bind and preserve prior failure; the next round cannot reset its budget.
        write_json(stage / 'audit_plan.json', plan)
        if prior:
            write_json(stage / 'previous_assessment.json', prior)
        # Page references the existing read-only review project. Keep the two
        # directory locations together; no private data is put in the code kit.
        mids = plan['sample_mothers'] + plan['risk_mothers']
        for role in ('primary', 'secondary', 'adjudication'):
            review_page(stage / (role + '.html'), catalog, root / 'source', context=plan['audit_id'], role=role,
                        mother_ids=mids, blind_first=plan['blind_first'],
                        required=plan['secondary_mothers'] if role == 'secondary' else mids)
        # staging and final are siblings, so relative evidence links are stable.
        publish(stage, final)
        print(json.dumps({'audit': str(final), 'audit_id': plan['audit_id'], 'N': len(plan['candidate_mothers']),
                          'random_mothers': len(plan['sample_mothers']), 'risk_extra': len(plan['risk_mothers']),
                          'second_review_base': len(plan['secondary_mothers']), 'round': number}, ensure_ascii=False))
    finally:
        lock.unlink()


def assessment_record(plan, catalog, primary_doc, secondary_doc, resolution_doc):
    allowed = {p['logical_probe_id'] for p in catalog['probes']
               if p['problem_id'] in set(plan['sample_mothers'] + plan['risk_mothers'])}
    a = normalize_reviews(primary_doc, catalog, context_root=plan['audit_id'], role='primary', allowed_ids=allowed)
    b = normalize_reviews(secondary_doc, catalog, context_root=plan['audit_id'], role='secondary', allowed_ids=allowed)
    c = normalize_reviews(resolution_doc, catalog, context_root=plan['audit_id'], role='adjudication', allowed_ids=allowed) if resolution_doc else {}
    result = audit_statistics(plan, catalog, a, b, c)
    record = {'schema_version': 'native_audit_assessment_v1', 'plan': plan, 'catalog': catalog,
              'primary_document': primary_doc, 'secondary_document': secondary_doc,
              'resolution_document': resolution_doc, 'result': result}
    record['assessment_id'] = 'NAA-' + digest(record)
    return record


def read_assessment(path):
    root = Path(path).resolve()
    verify_directory(root)
    r = read_json(root / 'assessment.json')
    check_assessment(r)
    return r


def check_assessment(record):
    plan = record['plan']
    require(make_audit_plan(plan['source_project'], record['catalog'], plan['source_membership'],
                            plan['campaign'], plan['policy_signoff'], plan['development_receipt'],
                            plan['round_number'], plan['previous_assessment_root']) == plan, 'ASSESSMENT_PLAN_CHANGED')
    require(assessment_record(record['plan'], record['catalog'], record['primary_document'],
                             record['secondary_document'], record['resolution_document']) == record, 'AUDIT_ASSESSMENT_CHANGED')


def audit_assess(project_path, audit_path, primary_path, secondary_path, resolution_path, output):
    root, p, catalog = load_project(project_path)
    plan = load_audit(audit_path, p, catalog, read_json(root / 'source/membership.json'))
    check_output(output, root, audit_path)
    record = assessment_record(plan, catalog, read_json(primary_path), read_json(secondary_path),
                               read_json(resolution_path) if resolution_path else None)
    if plan['round_number'] == 2:
        prior = read_json(Path(audit_path) / 'previous_assessment.json')
        check_assessment(prior)
        require(digest(prior) == plan['previous_assessment_root'], 'PREVIOUS_ASSESSMENT_STALE')
        record['previous_assessment'] = prior
        # Keep the standard record stable; chain is a separate artifact.
        prior = record.pop('previous_assessment')
    else:
        prior = None
    final, stage = new_output(output)
    write_json(stage / 'assessment.json', record)
    if prior:
        write_json(stage / 'previous_assessment.json', prior)
    publish(stage, final)
    print(json.dumps(record['result'], ensure_ascii=False, indent=2))


def selection_plan(project, catalog, membership, *, receipt=None, assessment=None, split='test', previous=None, audited_subset=False):
    require(split in ('test', 'development'), 'SPLIT_MUST_BE_EXPLICIT')
    dev = set(membership['development_problem_ids'])
    groups = mother_groups(catalog)
    allowed = set(groups) & dev if split == 'development' else set(groups) - dev
    if receipt is not None:
        require(assessment is None and not audited_subset, 'CHOOSE_INDIVIDUAL_OR_BATCH')
        require(receipt['project_id'] == project['project_id'], 'RECEIPT_PROJECT_STALE')
        decisions = merge_reviews(receipt['original_documents'], catalog, project['project_id'])
        summary = review_summary(catalog, decisions)
        require(summary == receipt['summary'] and decisions == receipt['decisions'], 'RECEIPT_CHANGED')
        require(not any(s in ('systemic', 'information', 'unresolved') for vals in
                        summary['defective_or_unresolved_mothers'].values() for s in vals), 'UNRESOLVED_OR_SYSTEMIC_REVIEW_BLOCKS_EXPORT')
        selected = allowed & set(summary['approved_complete_mothers'])
        mode, proof = 'individual', receipt
    else:
        require(split == 'test' and assessment is not None, 'BATCH_IS_TEST_ONLY')
        check_assessment(assessment)
        audit = assessment['plan']
        require(audit['project_id'] == project['project_id'] and assessment['catalog'] == catalog, 'AUDIT_SOURCE_MISMATCH')
        expected = make_audit_plan(project, catalog, membership, audit['campaign'], audit['policy_signoff'],
                                  audit['development_receipt'], audit['round_number'], audit['previous_assessment_root'])
        require(expected == audit, 'AUDIT_PLAN_CHANGED')
        if audit['round_number'] == 2:
            require(previous is not None and digest(previous) == audit['previous_assessment_root'], 'PREVIOUS_AUDIT_CHAIN_REQUIRED')
            check_assessment(previous)
            require(previous['plan']['round_number'] == 1 and previous['plan']['campaign'] == audit['campaign']
                    and not previous['result']['accepted'] and previous['plan']['policy_signoff'] == audit['policy_signoff'], 'INVALID_TWO_ROUND_CHAIN')
            require(previous['result']['review_complete'] or previous['result']['systemic_probe_ids'], 'PREVIOUS_ROUND_WAS_NOT_TERMINAL')
            check_retry_revision(previous, project, catalog)
        else:
            require(previous is None and audit['previous_assessment_root'] is None, 'UNEXPECTED_AUDIT_CHAIN')
        if audited_subset:
            result = assessment['result']
            require(not result['systemic_probe_ids'] and not result['unresolved_probe_ids'], 'UNRESOLVED_OR_SYSTEMIC_REVIEW_BLOCKS_EXPORT')
            primary = {d['logical_probe_id']: d for d in assessment['primary_document']['decisions']}
            secondary = {d['logical_probe_id']: d for d in assessment['secondary_document']['decisions']}
            selected = {pid for pid in allowed if all(primary.get(q, {}).get('status') == 'approved'
                        and (q not in secondary or secondary[q]['status'] == 'approved') for q in groups[pid])}
            mode = 'audited_individual'
        else:
            require(assessment['result']['accepted'], 'BATCH_AUDIT_NOT_ACCEPTED')
            selected = set(assessment['result']['retained_mothers'])
            mode = 'batch'
        require(selected <= allowed, 'DEVELOPMENT_TEST_LEAKAGE')
        proof = assessment
    require(selected, 'NO_COMPLETE_APPROVED_MOTHERS_IN_REQUESTED_SPLIT')
    selected_items = [item for item in catalog['probes'] if item['problem_id'] in selected]
    member_map = {m['problem_id']: m for m in membership['members']}
    require(all(member_map[pid]['dedup_state'] == 'representative' for pid in selected), 'NONREPRESENTATIVE_MEMBER')
    require(len({member_map[pid]['content_group_hint'] for pid in selected}) == len(selected), 'DUPLICATE_RELEASE_MOTHER')
    return {'schema_version': 'native_release_request_v1', 'project_id': project['project_id'],
            'mode': mode, 'split': split, 'release_implementation': implementation_hashes(),
            'proof_root': digest(proof), 'previous_assessment_root': digest(previous) if previous else None,
            'selected_mothers': sorted(selected), 'probe_ids': sorted(i['logical_probe_id'] for i in selected_items),
            'excluded_mothers': sorted(set(groups) - selected),
            'development_reserved_mothers': sorted(dev),
            'packet_policy': 'gold_only_for_approved_nonempty_packet; raw_for_every_B',
            'coverage_limit': 'finite_population_batch_audit' if mode == 'batch' else 'selected_fully_reviewed_mothers_not_population_audit',
            'near_duplicate_policy': 'explicit_team_signoff_required_not_automatically_solved'}


def release_plan(project_path, reviews_path, assessment_path, split, output, audited_subset=False):
    root, p, catalog = load_project(project_path)
    require(bool(reviews_path) != bool(assessment_path), 'CHOOSE_REVIEWS_OR_ASSESSMENT')
    receipt = read_receipt(reviews_path, p, catalog) if reviews_path else None
    assessment = read_assessment(assessment_path) if assessment_path else None
    prior_path = Path(assessment_path) / 'previous_assessment.json' if assessment_path else None
    previous = read_json(prior_path) if prior_path and prior_path.exists() else None
    request = selection_plan(p, catalog, read_json(root / 'source/membership.json'), receipt=receipt,
                             assessment=assessment, split=split, previous=previous, audited_subset=audited_subset)
    check_output(output, root, reviews_path or assessment_path)
    final, stage = new_output(output)
    write_json(stage / 'request.json', request)
    write_json(stage / 'review_proof.PRIVATE.json', receipt or assessment)
    if previous:
        write_json(stage / 'previous_assessment.json', previous)
    write_json(stage / 'approval.TEMPLATE.json', approval_template(digest(request)))
    publish(stage, final)
    print(json.dumps({'request': str(final), 'mothers': len(request['selected_mothers']), 'probes': len(request['probe_ids']),
                      'split': split, 'next': 'Copy approval.TEMPLATE.json outside; obtain real signoff; then export.'}, ensure_ascii=False))


def verified_request(project_path, request_path, approval_path, *, deep=False):
    root, project, catalog = load_project(project_path, deep=deep)
    reqroot = Path(request_path).resolve()
    verify_directory(reqroot)
    request = read_json(reqroot / 'request.json')
    proof = read_json(reqroot / 'review_proof.PRIVATE.json')
    previous = read_json(reqroot / 'previous_assessment.json') if (reqroot / 'previous_assessment.json').exists() else None
    expected = selection_plan(project, catalog, read_json(root / 'source/membership.json'),
                              receipt=proof if request['mode'] == 'individual' else None,
                              assessment=proof if request['mode'] != 'individual' else None,
                              split=request['split'], previous=previous, audited_subset=request['mode'] == 'audited_individual')
    require(request == expected, 'RELEASE_REQUEST_RECOMPUTATION_FAILED')
    approval = read_json(approval_path)
    check_signoff(approval, digest(request))
    return root, project, catalog, request, proof, approval


def authorization_for(item, request, proof, approval):
    qid, pid = item['logical_probe_id'], item['problem_id']
    if request['mode'] == 'individual':
        evidence = [proof['decisions'][qid]]
        human = 'approved_individual'
    else:
        evidence = [d for name in ('primary_document', 'secondary_document', 'resolution_document')
                    for d in (proof[name] or {}).get('decisions', []) if d['logical_probe_id'] == qid and d['status'] != 'pending']
        human = 'approved_sampled' if evidence else 'not_sampled'
    return {'schema_version': 'native_authorization_v1', 'mode': request['mode'],
            'request_root': digest(request), 'proof_root': digest(proof), 'signoff_root': digest(approval),
            'review_target_root': item['review_target_root'], 'human_item_review': human,
            'review_evidence': evidence, 'release_approver': approval['reviewer']}


def freeze_native(source, item, authorization):
    """Separate native authorization path, using all original strict validators.

    Batch-derived ledger references identify an explicit audit, not fabricated
    per-item human decisions. Original draft documents remain pending/unchanged.
    """
    d, env = load_probe(source, item)
    rebuilt = verify_adapter_snapshot(env)
    verify_descriptor(item, d['key'], rebuilt)
    candidate, target = prospective(d, env)
    require(target['content_root'] == item['review_target_root'] == authorization['review_target_root'], 'AUTHORIZATION_TARGET_STALE')
    reviewer = authorization['release_approver']
    evidence = ['native-authorization:' + digest(authorization), 'review-proof:' + authorization['proof_root'],
                'approval-mode:' + authorization['mode']]
    for rid, kind, content in (('qa-packet', 'packet', target['packet_review_root']),
                               ('qa-closure', 'query_closure', target['closure_root']),
                               ('qa-instance', 'instance', target['content_root'])):
        require(rid not in env['ledger']['records'], 'QA_LEDGER_COLLISION')
        env['ledger']['records'][rid] = review_record(rid, kind, content, reviewer, evidence)
    prepared = prepare_probe(candidate, env, verify_reviews=True)
    require(prepared['content_root'] == target['content_root'], 'APPROVAL_CHANGED_PUBLIC_CONTENT')
    iid = 'PAI-' + digest({'content_root': target['content_root'], 'authorization': authorization})
    key = {**prepared['core']['private_core'], 'content_root': target['content_root'],
           'instance_id': iid, 'instance_review_refs': ['qa-instance']}
    envelope = {'schema_version': 'physalign_envelope_v2_1', 'logical_probe_id': key['logical_probe_id'],
                'variant_id': key['variant_id'], 'instance_id': iid, 'language': key['language'],
                'content_root': key['content_root'], 'public_raw_hash': digest(prepared['core']['public_raw']),
                'public_gold_hash': digest(prepared['core']['public_gold']), 'private_hash': digest(key)}
    validate_schema('private_mapping.schema.json', key)
    validate_schema('instance_envelope.schema.json', envelope)
    frozen = {'envelope': envelope, 'private': key, 'core': prepared['core'], 'authorization': authorization}
    return frozen, env


def public_record(item, frozen, env, condition, split):
    public = frozen['core']['public_' + condition]
    attachments = []
    for aid in public['attachment_ids']:
        a = env['assets'][aid]
        suffix = Path(a['relative_path']).suffix.lower()
        require(suffix in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'), 'UNSUPPORTED_PUBLIC_IMAGE_EXTENSION')
        attachments.append({'asset_id': aid, 'path': 'images/' + a['bytes_sha256'] + suffix,
                            'sha256': a['bytes_sha256'], 'width': a['width'], 'height': a['height']})
    # Fixed allowlist: no envelope private hash, key, source names, gold aliases,
    # audit metadata or filesystem paths are part of the model input record.
    return {'schema_version': 'physalign_public_qa_v1', 'instance_id': frozen['envelope']['instance_id'],
            'logical_probe_id': item['logical_probe_id'], 'task_id': item['task_id'],
            'interface': item['metadata']['interface'], 'language': item['language'], 'split': split,
            'condition': condition, 'input': {'messages': copy.deepcopy(public['messages']), 'attachments': attachments}}


def release_payload(source, catalog, request, proof, approval):
    selected = [i for i in catalog['probes'] if i['logical_probe_id'] in set(request['probe_ids'])]
    frozen, rows, members = [], {'raw': [], 'gold': []}, []
    for position, item in enumerate(selected, 1):
        auth = authorization_for(item, request, proof, approval)
        f, env = freeze_native(source, item, auth)
        frozen.append(f)
        P = f['private']['packet_plan']['state'] == 'approved_nonempty'
        L = bool(f['private']['read_targets'])
        require(P == item['metadata']['packet_nonempty_candidate'] and L == item['metadata']['L_candidate'], 'B_L_P_MEMBERSHIP_CHANGED')
        require(item['metadata']['interface'] != 'T03-text' or (not P and not L), 'TEXT_BRANCH_HAS_IMAGE_READ_OR_PACKET')
        for condition in (('raw', 'gold') if P else ('raw',)):
            rows[condition].append(public_record(item, f, env, condition, request['split']))
        members.append({'logical_probe_id': item['logical_probe_id'], 'instance_id': f['envelope']['instance_id'],
                        'problem_id': item['problem_id'], 'group_id': item['group_id'],
                        'content_group_hint': item['content_group_hint'], 'interface': item['metadata']['interface'],
                        'split': request['split'], 'B': True, 'L': L, 'P': P,
                        'human_item_review': auth['human_item_review'],
                        'metadata': {**item['metadata'], 'P_approved': P, 'benchmark_QA_approval': 'approved_' + request['mode'],
                                     'packet_state': f['private']['packet_plan']['state'], 'human_item_review': auth['human_item_review']}})
        if position % 25 == 0 or position == len(selected):
            print(f'[release-check {position}/{len(selected)}] source, packet and public/private mappings verified', flush=True)
    validate_release_instances(frozen)
    # Fixed mother-level sets for an EXTERNAL scorer; no prediction aggregation.
    pools = {s: {'probe_ids': [m['logical_probe_id'] for m in members if m[s]],
                 'mother_ids': sorted({m['problem_id'] for m in members if m[s]})} for s in ('B', 'L', 'P')}
    return frozen, rows, members, pools


def synthetic_guard(catalog, request, synthetic):
    selected = [p for p in catalog['probes'] if p['problem_id'] in request['selected_mothers']]
    flags = [p['metadata']['source_namespace'].upper().startswith('SYNTHETIC') or p['problem_id'].upper().startswith('SYNTHETIC') for p in selected]
    if synthetic:
        require(all(flags), 'SYNTHETIC_MODE_ONLY_FOR_SYNTHETIC_INPUT')
    else:
        require(not any(flags), 'SYNTHETIC_DATA_CANNOT_BE_BENCHMARK_RELEASE')


def build_release_manifest(project, request, approval, members, pools, rows, synthetic):
    manifest = {'schema_version': VERSION, 'artifact_mode': 'synthetic_example_NOT_benchmark_data' if synthetic else 'reviewed_data_release',
                'project_id': project['project_id'], 'request_root': digest(request), 'signoff_root': digest(approval),
                'release_implementation': implementation_hashes(), 'source_implementation': dependency_hashes(),
                'split': request['split'], 'approval_mode': request['mode'],
                'counts': {s: {'probes': len(v['probe_ids']), 'mothers': len(v['mother_ids'])} for s, v in pools.items()},
                'public_root': 'public', 'attachment_paths_relative_to': 'public',
                'private_files_not_for_model': True, 'model_runner_or_scorer_included': False,
                'research_controls_and_measurement_validation': 'external_not_certified_by_data_export',
                'coverage_limit': request['coverage_limit']}
    manifest['release_id'] = 'NREL-' + digest({'manifest': manifest, 'members': members, 'public': rows})
    return manifest


def export_release(project_path, request_path, approval_path, output, synthetic=False):
    check_output(output, project_path, request_path)
    root, project, catalog, request, proof, approval = verified_request(project_path, request_path, approval_path)
    synthetic_guard(catalog, request, synthetic)
    final, stage = new_output(output)
    # Entire frozen source candidate snapshot supports audit replay, including
    # exclusions and original defect denominators. Everything stays PRIVATE.
    copy_tree(root, stage / 'private/project')
    copy_tree(request_path, stage / 'private/request')
    write_json(stage / 'private/signoff.json', approval)
    fs, rows, members, pools = release_payload(stage / 'private/project/source', catalog, request, proof, approval)
    for f in fs:
        write_json(stage / 'private/frozen' / (f['envelope']['instance_id'] + '.json'), f)
    byid = {i['logical_probe_id']: i for i in catalog['probes']}
    for condition, records in rows.items():
        for r in records:
            item = byid[r['logical_probe_id']]
            _, env = load_probe(stage / 'private/project/source', item)
            for a in r['input']['attachments']:
                src = safe_path(env['asset_root'], env['assets'][a['asset_id']]['relative_path'])
                dst = safe_path(stage / 'public', a['path'])
                require(sha_bytes(src.read_bytes()) == a['sha256'], 'PUBLIC_IMAGE_SOURCE_CHANGED')
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                require(sha_bytes(dst.read_bytes()) == a['sha256'], 'PUBLIC_IMAGE_COLLISION')
        write_json(stage / 'public' / ('qa_' + condition + '.json'), records)
    write_json(stage / 'private/membership.json', members)
    write_json(stage / 'private/pools.json', pools)
    write_json(stage / 'private/mappings.json', [f['private'] for f in fs])
    for name, schema in schema_bundle().items():
        write_json(stage / 'private/contract_schemas' / name, schema)
    manifest = build_release_manifest(project, request, approval, members, pools, rows, synthetic)
    write_json(stage / 'release.json', manifest)
    (stage / 'README.md').write_text(
        '# PhysAlign native data release\n\n'
        + ('SYNTHETIC EXAMPLE ONLY — NOT BENCHMARK DATA.\n\n' if synthetic else '')
        + 'Model inputs: public/qa_raw.json (B), public/qa_gold.json (P only).\n'
        + 'Resolve every attachment path relative to public/. Send input.messages and the actual image bytes.\n'
        + 'NEVER send private/, review HTML, release manifests, provenance, or answer keys to a model.\n'
        + 'Each probe/condition requires an independent context in your external runner.\n'
        + 'Private mappings, B/L/P membership, source snapshots and review proofs support external scoring/audit.\n'
        + 'No model runner/scorer/metric implementation or research-control certification is included.\n', encoding='utf-8')
    # Validate new output before publishing: no partial directory looks released.
    from io_utils import seal_directory
    seal_directory(stage)
    validate_release(stage)
    # Validation is read-only; publish preserves hashes and refuses overwrite.
    publish(stage, final)
    print(json.dumps({'release': str(final), **manifest}, ensure_ascii=False))


def validate_release(path):
    root = Path(path).resolve()
    verify_directory(root)
    manifest = read_json(root / 'release.json')
    pr, project, catalog, request, proof, approval = verified_request(root / 'private/project', root / 'private/request',
                                                                   root / 'private/signoff.json', deep=True)
    synthetic = manifest['artifact_mode'] == 'synthetic_example_NOT_benchmark_data'
    require(manifest['artifact_mode'] in ('synthetic_example_NOT_benchmark_data', 'reviewed_data_release'), 'RELEASE_MODE_INVALID')
    synthetic_guard(catalog, request, synthetic)
    fs, rows, members, pools = release_payload(pr / 'source', catalog, request, proof, approval)
    require(manifest == build_release_manifest(project, request, approval, members, pools, rows, synthetic), 'RELEASE_MANIFEST_REBUILD_MISMATCH')
    require(read_json(root / 'private/membership.json') == members and read_json(root / 'private/pools.json') == pools,
            'RELEASE_MEMBERSHIP_OR_POOLS_CHANGED')
    require(read_json(root / 'private/mappings.json') == [f['private'] for f in fs], 'PRIVATE_MAPPINGS_CHANGED')
    for f in fs:
        require(read_json(root / 'private/frozen' / (f['envelope']['instance_id'] + '.json')) == f, 'FROZEN_REBUILD_MISMATCH')
    allowed_public = {'qa_raw.json', 'qa_gold.json'}
    for condition in rows:
        require(read_json(root / 'public' / ('qa_' + condition + '.json')) == rows[condition], 'PUBLIC_QA_REBUILD_MISMATCH')
        for r in rows[condition]:
            for a in r['input']['attachments']:
                allowed_public.add(a['path'])
                require(sha_bytes(safe_path(root / 'public', a['path']).read_bytes()) == a['sha256'], 'PUBLIC_IMAGE_BYTES_CHANGED')
    require({p.relative_to(root / 'public').as_posix() for p in (root / 'public').rglob('*') if p.is_file()} == allowed_public,
            'UNDECLARED_PUBLIC_FILE_POSSIBLE_PRIVATE_LEAK')
    for name, schema in schema_bundle().items():
        require(read_json(root / 'private/contract_schemas' / name) == schema, 'SCHEMA_BUNDLE_CHANGED')
    require(manifest['release_implementation'] == implementation_hashes() and manifest['source_implementation'] == dependency_hashes(), 'RELEASE_CODE_VERSION_CHANGED')
    require(manifest['request_root'] == digest(request) and manifest['signoff_root'] == digest(approval)
            and manifest['project_id'] == project['project_id'] and manifest['split'] == request['split']
            and manifest['approval_mode'] == request['mode'], 'RELEASE_AUTHORIZATION_CHANGED')
    require(manifest['counts'] == {s: {'probes': len(v['probe_ids']), 'mothers': len(v['mother_ids'])} for s, v in pools.items()}, 'RELEASE_COUNTS_CHANGED')
    require(manifest['release_id'] == 'NREL-' + digest({'manifest': {k: v for k, v in manifest.items() if k != 'release_id'},
                                                     'members': members, 'public': rows}), 'RELEASE_ID_CHANGED')
    print(json.dumps({'valid': True, 'artifact_mode': manifest['artifact_mode'], 'release_id': manifest['release_id'],
                      'counts': manifest['counts'], 'prediction_scoring_performed': False}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('review-init'); p.add_argument('--draft', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('review-import'); p.add_argument('--project', required=True); p.add_argument('--decisions', action='append', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('campaign-init'); p.add_argument('--output', required=True)
    p = sub.add_parser('audit-plan'); p.add_argument('--project', required=True); p.add_argument('--campaign', required=True); p.add_argument('--policy-approval', required=True); p.add_argument('--development-review', required=True); p.add_argument('--previous-assessment')
    p = sub.add_parser('audit-assess'); p.add_argument('--project', required=True); p.add_argument('--audit', required=True); p.add_argument('--primary', required=True); p.add_argument('--secondary', required=True); p.add_argument('--adjudication'); p.add_argument('--output', required=True)
    p = sub.add_parser('release-plan'); p.add_argument('--project', required=True); g = p.add_mutually_exclusive_group(required=True); g.add_argument('--reviews'); g.add_argument('--assessment'); p.add_argument('--split', choices=('development', 'test'), required=True); p.add_argument('--output', required=True); p.add_argument('--audited-subset', action='store_true')
    p = sub.add_parser('export'); p.add_argument('--project', required=True); p.add_argument('--request', required=True); p.add_argument('--approval', required=True); p.add_argument('--output', required=True); p.add_argument('--synthetic-example', action='store_true')
    p = sub.add_parser('validate'); p.add_argument('--path', required=True)
    args = parser.parse_args(argv)
    if args.command == 'review-init': project_init(args.draft, args.output)
    elif args.command == 'review-import': import_review(args.project, args.decisions, args.output)
    elif args.command == 'campaign-init': campaign_init(args.output)
    elif args.command == 'audit-plan': audit_plan(args.project, args.campaign, args.policy_approval, args.development_review, args.previous_assessment)
    elif args.command == 'audit-assess': audit_assess(args.project, args.audit, args.primary, args.secondary, args.adjudication, args.output)
    elif args.command == 'release-plan': release_plan(args.project, args.reviews, args.assessment, args.split, args.output, args.audited_subset)
    elif args.command == 'export': export_release(args.project, args.request, args.approval, args.output, args.synthetic_example)
    elif args.command == 'validate': validate_release(args.path)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
