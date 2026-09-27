"""Native data release contracts. No model calls and no inferred approvals."""
from __future__ import annotations
import copy
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import sys

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'physalign_converter'))
from io_utils import read_json, sha_bytes
from reference_checks import require, digest
from review_workflow import CONFIRMATIONS

VERSION = 'native_data_release_v1'
POLICY = {'policy_id': 'PLAN_6_finite_population_v1', 'sample_mothers': 100,
          'secondary_mothers': 25, 'risk_extra_budget': 20, 'max_rounds': 2,
          'alpha_numerator': 1, 'alpha_denominator': 40,
          'max_defect_numerator': 1, 'max_defect_denominator': 20,
          'unit': 'mother_with_at_least_one_serious_defect',
          'sampling': 'simple_random_without_replacement',
          'removal_policy': 'whole_confirmed_defective_mother_only'}
SIGNOFF_CHECKS = ('source_and_rules_checked', 'known_systemic_and_unresolved_issues_zero',
                 'near_duplicates_and_external_splits_checked', 'not_selected_by_model_performance',
                 'review_evidence_authentic', 'data_use_authorized')
SEVERITIES = ('none', 'minor', 'serious', 'systemic', 'information', 'unresolved')
SERIOUS = {'serious', 'systemic', 'information'}


def implementation_hashes():
    names = ('native_release.py', 'native_release_support.py', 'native_release_review.html')
    return {n: sha_bytes((KIT / 'tools' / n).read_bytes()) for n in names}


def exact(obj, fields, code):
    require(isinstance(obj, dict) and set(obj) == set(fields), code)


def nonempty(value, code):
    require(isinstance(value, str) and bool(value.strip()), code)


def approval_template(target):
    return {'schema_version': 'native_release_signoff_v1', 'target_root': target,
            'status': 'pending', 'reviewer': '', 'evidence_ref': '',
            'confirmations': {k: False for k in SIGNOFF_CHECKS}}


def check_signoff(record, target):
    exact(record, ('schema_version', 'target_root', 'status', 'reviewer', 'evidence_ref', 'confirmations'), 'SIGNOFF_FIELDS')
    require(record['schema_version'] == 'native_release_signoff_v1' and record['target_root'] == target,
            'SIGNOFF_WRONG_VERSION_OR_TARGET')
    require(record['status'] == 'approved', 'EXPLICIT_SIGNOFF_REQUIRED')
    nonempty(record['reviewer'], 'SIGNOFF_REVIEWER_REQUIRED')
    nonempty(record['evidence_ref'], 'SIGNOFF_EVIDENCE_REQUIRED')
    exact(record['confirmations'], SIGNOFF_CHECKS, 'SIGNOFF_CONFIRMATIONS')
    require(all(x is True for x in record['confirmations'].values()), 'SIGNOFF_CONFIRMATIONS')


def normalize_reviews(doc, catalog, *, context_root, role='individual', allowed_ids=None):
    """Legacy development exports remain valid for individual review ONLY.

    A new context cannot be inferred from a legacy JSON. Rejection without a
    defect classification remains unresolved, never silently isolated.
    """
    by_id = {p['logical_probe_id']: p for p in catalog['probes']}
    allowed = set(by_id) if allowed_ids is None else set(allowed_ids)
    legacy = doc.get('schema_version') == 'physalign_decisions_v1'
    if legacy:
        exact(doc, ('schema_version', 'catalog_root', 'decisions'), 'REVIEW_FILE_FIELDS')
        require(role == 'individual', 'LEGACY_REVIEW_NOT_RANDOM_AUDIT')
    else:
        exact(doc, ('schema_version', 'catalog_root', 'context_root', 'role', 'decisions'), 'REVIEW_FILE_FIELDS')
        require(doc['schema_version'] == 'native_release_reviews_v1' and doc['context_root'] == context_root
                and doc['role'] == role, 'REVIEW_CONTEXT_OR_ROLE_MISMATCH')
    require(doc['catalog_root'] == digest(catalog), 'REVIEW_CATALOG_STALE')
    require(isinstance(doc['decisions'], list), 'REVIEW_LIST_REQUIRED')
    found = {}
    base = ('logical_probe_id', 'review_target_root', 'status', 'reviewer', 'blind_answer', 'confirmations', 'note')
    for original in doc['decisions']:
        exact(original, base if legacy else (*base, 'severity', 'sequence'), 'REVIEW_DECISION_FIELDS')
        d = copy.deepcopy(original)
        qid = d['logical_probe_id']
        require(isinstance(qid, str) and qid in allowed and qid in by_id and qid not in found,
                'UNKNOWN_OR_DUPLICATE_REVIEW')
        require(d['review_target_root'] == by_id[qid]['review_target_root'], 'REVIEW_TARGET_STALE')
        require(d['status'] in ('pending', 'approved', 'rejected'), 'REVIEW_STATUS_INVALID')
        require(all(isinstance(d[k], str) for k in ('reviewer', 'blind_answer', 'note')), 'REVIEW_TEXT_REQUIRED')
        exact(d['confirmations'], CONFIRMATIONS, 'REVIEW_CONFIRMATIONS_FIELDS')
        require(all(type(x) is bool for x in d['confirmations'].values()), 'REVIEW_BOOLEAN_REQUIRED')
        if legacy:
            d.update(severity='unresolved' if d['status'] == 'rejected' else 'none', sequence=None)
        require(d['severity'] in SEVERITIES, 'REVIEW_SEVERITY_INVALID')
        require(d['sequence'] is None or (type(d['sequence']) is int and d['sequence'] > 0), 'REVIEW_SEQUENCE_INVALID')
        if d['status'] != 'pending':
            nonempty(d['reviewer'], 'REVIEWER_REQUIRED')
            nonempty(d['blind_answer'], 'BLIND_ANSWER_REQUIRED')
            if not legacy and role != 'individual':
                require(d['sequence'] is not None, 'REVIEW_SEQUENCE_REQUIRED')
        if d['status'] == 'approved':
            require(d['severity'] in ('none', 'minor') and all(d['confirmations'].values()), 'APPROVAL_CHECKS_REQUIRED')
        elif d['status'] == 'rejected':
            require(d['severity'] not in ('none', 'minor'), 'REJECTION_NEEDS_DEFECT_CLASS')
            nonempty(d['note'], 'REJECTION_NOTE_REQUIRED')
        found[qid] = d
    return found


def merge_reviews(documents, catalog, context_root):
    merged = {}
    for doc in documents:
        part = normalize_reviews(doc, catalog, context_root=context_root)
        for qid, d in part.items():
            old = merged.get(qid)
            if old is None or old['status'] == 'pending':
                merged[qid] = d
            elif d['status'] != 'pending':
                require(old == d, 'CONFLICTING_REVIEWS_USE_ONE_ADJUDICATED_EXPORT', qid)
    return merged


def mother_groups(catalog):
    groups = defaultdict(list)
    for p in catalog['probes']:
        groups[p['problem_id']].append(p['logical_probe_id'])
    return {k: sorted(v) for k, v in sorted(groups.items())}


def review_summary(catalog, decisions):
    groups = mother_groups(catalog)
    completed = [pid for pid, ids in groups.items()
                 if all(decisions.get(q, {}).get('status') == 'approved' for q in ids)]
    defects = {pid: sorted({decisions[q]['severity'] for q in ids
                           if decisions.get(q, {}).get('status') == 'rejected'}) for pid, ids in groups.items()}
    return {'approved_complete_mothers': completed,
            'defective_or_unresolved_mothers': {k: v for k, v in defects.items() if v},
            'status_counts': dict(Counter(decisions.get(p['logical_probe_id'], {}).get('status', 'pending')
                                         for p in catalog['probes']))}


def hypergeom_upper(N, n, k):
    """Exact one-sided 97.5% finite-population upper confidence bound on D.

    Invert P_D[X <= k] >= 1/40 using INTEGER comparisons (no floating-point
    threshold, binomial substitution, or changed order of operations).
    Equality is retained conservatively. n=0 conveys no information.
    """
    require(all(type(v) is int for v in (N, n, k)) and 0 <= k <= n <= N, 'INVALID_AUDIT_COUNTS')
    if n == 0:
        return N
    denominator = math.comb(N, n)
    def retained(D):
        low, high = max(0, n - (N-D)), min(k, n, D)
        numerator = sum(math.comb(D, x) * math.comb(N-D, n-x) for x in range(low, high+1))
        return POLICY['alpha_denominator'] * numerator >= POLICY['alpha_numerator'] * denominator
    lo, hi = k, N
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if retained(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def finite_risk(N, n, k, removed_confirmed):
    require(type(removed_confirmed) is int and 0 <= removed_confirmed <= N, 'REMOVAL_COUNT_INVALID')
    require(removed_confirmed <= k + N - n, 'REMOVED_DEFECTS_INCONSISTENT_WITH_ORIGINAL_SAMPLE')
    upper = hypergeom_upper(N, n, k)
    # Independent risk checks may establish more known defects than a rare
    # primary-sample interval covers. Never let a negative numerator pass.
    upper = max(upper, removed_confirmed)
    remain = N - removed_confirmed
    num = max(0, upper - removed_confirmed)
    return {'population_mothers': N, 'sample_mothers': n, 'detected_sample_mothers': k,
            'upper_defective_mothers_original': upper, 'removed_confirmed_mothers': removed_confirmed,
            'remaining_population_mothers': remain, 'upper_remaining_numerator': num,
            'upper_remaining_denominator': remain,
            'upper_remaining_decimal': float(Fraction(num, remain)) if remain else None,
            'passes_5_percent': remain > 0 and num * 20 <= remain,
            'confidence': 'one_sided_97.5_percent_hypergeometric',
            'limitation': 'defects_detectable_by_this_audit_not_a_zero_missed_error_guarantee'}


def risk_tags(item):
    m = item['metadata']
    return {digest(['rule', m['interface'], m['selector'], m['packet_policy_id']]),
            digest(['source', m['source_namespace']]),
            digest(['language', item['language']]), digest(['representation', m['representation_kind']]),
            digest(['repair', m['repair_applied_to_query']]), digest(['images', 'multi' if m['image_count'] > 1 else 'single'])}


def audit_statistics(plan, catalog, primary, secondary, resolutions):
    """Compute eligibility without writing an approval. Original findings retained."""
    groups = mother_groups(catalog)
    sample = set(plan['sample_mothers'])
    risk = set(plan['risk_mothers'])
    reviewed = sample | risk
    qids = {q for pid in reviewed for q in groups[pid]}
    require(set(primary) <= qids and set(secondary) <= qids, 'AUDIT_OUT_OF_SAMPLE_REVIEW')
    require(set(resolutions) <= qids, 'AUDIT_UNKNOWN_RESOLUTION')
    pending_primary = sorted(q for q in qids if primary.get(q, {}).get('status', 'pending') == 'pending')
    serious_mothers, flagged, systemic, unresolved = set(), set(), [], []
    for pid in reviewed:
        for q in groups[pid]:
            for slot in (primary, secondary):
                d = slot.get(q, {})
                if d.get('status') == 'rejected':
                    flagged.add(pid)
                    if d.get('severity') in SERIOUS:
                        serious_mothers.add(pid)
                    if d.get('severity') in ('systemic', 'information'):
                        systemic.append(q)
    secondary_missed = {pid for pid in reviewed for q in groups[pid]
                        if secondary.get(q, {}).get('severity') in SERIOUS
                        and primary.get(q, {}).get('severity') not in SERIOUS}
    required_secondary = (reviewed if secondary_missed else set(plan['secondary_mothers']) | flagged)
    pending_secondary = sorted(q for pid in required_secondary for q in groups[pid]
                               if secondary.get(q, {}).get('status', 'pending') == 'pending')
    confirmed_bad = set()
    for pid in reviewed:
        for q in groups[pid]:
            a, b = primary.get(q), secondary.get(q)
            if a is None or a['status'] == 'pending':
                continue
            if b and b['status'] != 'pending':
                require(a['reviewer'] != b['reviewer'], 'SECOND_REVIEWER_MUST_BE_INDEPENDENT', q)
            disagreement = b and b['status'] != 'pending' and (a['status'], a['severity']) != (b['status'], b['severity'])
            if q in resolutions:
                require(disagreement or a['severity'] == 'unresolved' or (b and b['severity'] == 'unresolved'),
                        'UNNECESSARY_RESOLUTION_NOT_ALLOWED', q)
                final = resolutions[q]
                require(final['reviewer'] not in {a['reviewer'], b['reviewer'] if b else ''}, 'INDEPENDENT_ADJUDICATOR_REQUIRED')
            elif disagreement or a['severity'] == 'unresolved' or (b and b['severity'] == 'unresolved'):
                unresolved.append(q)
                continue
            else:
                final = a
            if final['severity'] in ('systemic', 'information'):
                systemic.append(q)
            if final['status'] == 'rejected':
                if final['severity'] == 'serious':
                    confirmed_bad.add(pid)
                    serious_mothers.add(pid)
                else:
                    unresolved.append(q)
    # Never erase an observed defect count after isolation/adjudication.
    k = len(sample & serious_mothers)
    stats = finite_risk(len(plan['candidate_mothers']), len(sample), k, len(confirmed_bad))
    reasons = []
    if pending_primary: reasons.append('PRIMARY_INCOMPLETE')
    if pending_secondary: reasons.append('SECONDARY_INCOMPLETE_OR_ESCALATION_REQUIRED')
    if unresolved: reasons.append('UNRESOLVED_DISAGREEMENTS')
    if systemic: reasons.append('SYSTEMIC_OR_INFORMATION_DEFECT_WITHDRAW_RULE_VERSION')
    if plan['uncovered_risk_tags']: reasons.append('RISK_COVERAGE_INCOMPLETE_NO_INHERITANCE')
    if not stats['passes_5_percent']: reasons.append('FINITE_POPULATION_BOUND_ABOVE_LIMIT')
    for slot in (primary, secondary):
        for pid in reviewed:
            ds = [slot[q] for q in groups[pid] if slot.get(q, {}).get('status') not in (None, 'pending')]
            if ds:
                require(len({d['reviewer'] for d in ds}) == 1, 'ONE_REVIEWER_PER_MOTHER_PER_ROLE')
                require(all(d['sequence'] is not None for d in ds), 'AUDIT_SEQUENCE_REQUIRED')
                seqs = [d['sequence'] for d in ds]
                require(len(set(seqs)) == len(seqs), 'DUPLICATE_REVIEW_SEQUENCE')
                require(min(ds, key=lambda d: d['sequence'])['logical_probe_id'] == plan['blind_first'][pid],
                        'AUDIT_BLIND_FIRST_ORDER_NOT_RESPECTED', pid)
    return {'accepted': not reasons, 'blockers': reasons, 'risk': stats,
            'review_complete': not (pending_primary or pending_secondary or unresolved),
            'pending_primary_probe_ids': pending_primary, 'pending_secondary_probe_ids': pending_secondary,
            'unresolved_probe_ids': sorted(set(unresolved)), 'systemic_probe_ids': sorted(set(systemic)),
            'secondary_escalated_to_all_audit_mothers': bool(secondary_missed),
            'required_secondary_mothers': sorted(required_secondary),
            'confirmed_excluded_mothers': sorted(confirmed_bad),
            'retained_mothers': sorted(set(plan['candidate_mothers']) - confirmed_bad),
            'original_observed_serious_sample_mothers': sorted(sample & serious_mothers),
            'risk_additional_mothers_not_in_random_denominator': sorted(risk)}
