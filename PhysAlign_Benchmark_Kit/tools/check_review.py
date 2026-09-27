"""Read-only developer review-file check. NEVER approves, freezes or edits data."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'physalign_converter'))
from io_utils import read_json, safe_path, sha_bytes
from reference_checks import require, digest
from review_workflow import CONFIRMATIONS


def summarize_review(catalog, development, decisions):
    require(catalog.get('native_profile') == 'native_binding_draft_v1', 'EXPECTED_NATIVE_DRAFT')
    require(set(decisions) == {'schema_version', 'catalog_root', 'decisions'} and
            decisions['schema_version'] == 'physalign_decisions_v1', 'REVIEW_FILE_SCHEMA')
    require(decisions['catalog_root'] == digest(catalog), 'REVIEW_FROM_DIFFERENT_CATALOG')
    require(development['membership_root'] == catalog['membership_root'], 'DEVELOPMENT_MEMBERSHIP_STALE')
    dev = set(development['problem_ids'])
    expected = {p['logical_probe_id']: p for p in catalog['probes'] if p['problem_id'] in dev}
    require(isinstance(decisions['decisions'], list), 'REVIEW_DECISIONS_MUST_BE_LIST')
    found = {}
    for d in decisions['decisions']:
        require(isinstance(d, dict) and set(d) == {'logical_probe_id', 'review_target_root', 'status',
            'reviewer', 'blind_answer', 'confirmations', 'note'}, 'REVIEW_DECISION_FIELDS')
        qid = d['logical_probe_id']
        require(isinstance(qid, str) and qid in expected and qid not in found, 'UNKNOWN_NONDEV_OR_DUPLICATE_PROBE')
        require(d['review_target_root'] == expected[qid]['review_target_root'], 'REVIEW_TARGET_STALE')
        require(d['status'] in ('pending', 'approved', 'rejected'), 'REVIEW_STATUS_INVALID')
        require(all(isinstance(d[k], str) for k in ('reviewer', 'blind_answer', 'note')), 'REVIEW_TEXT_FIELDS')
        require(isinstance(d['confirmations'], dict) and set(d['confirmations']) == set(CONFIRMATIONS) and
                all(type(x) is bool for x in d['confirmations'].values()), 'REVIEW_CONFIRMATIONS_SCHEMA')
        if d['status'] != 'pending':
            require(d['reviewer'].strip() and d['blind_answer'].strip(), 'REVIEWER_AND_BLIND_ANSWER_REQUIRED')
        if d['status'] == 'approved':
            require(all(d['confirmations'].values()), 'APPROVAL_REQUIRES_ALL_CONFIRMATIONS')
        found[qid] = d
    mothers = defaultdict(Counter)
    for qid, p in expected.items():
        mothers[p['problem_id']][found.get(qid, {}).get('status', 'pending')] += 1
    return {'schema_version': 'native_development_review_check_v1', 'valid_review_file': True,
        'catalog_root': digest(catalog), 'expected_development_mothers': len(dev),
        'expected_development_probes': len(expected),
        'status_counts': dict(sum(mothers.values(), Counter())),
        'complete_mothers': sorted(p for p, counts in mothers.items() if not counts['pending']),
        'remaining_probe_ids': sorted(q for q in expected if found.get(q, {}).get('status', 'pending') == 'pending'),
        'rejected': [{'problem_id': expected[q]['problem_id'], **d} for q, d in found.items() if d['status'] == 'rejected'],
        'missing_rejection_notes': sorted(q for q, d in found.items() if d['status'] == 'rejected' and not d['note'].strip()),
        'not_a_release_approval': True, 'not_a_semantic_reaudit': True,
        'source_and_draft_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--decisions', required=True)
    args = parser.parse_args()
    root = Path(args.draft).resolve()
    manifest = read_json(root / 'FILE_MANIFEST.json')
    # Verify the two descriptors used here. Full images/source validation remains
    # native-validate, deliberately not misrepresented as completed by this tool.
    for rel in ('catalog.PRIVATE.json', 'development_candidates.json'):
        require(sha_bytes(safe_path(root, rel).read_bytes()) == manifest[rel], 'DRAFT_DESCRIPTOR_BYTES_CHANGED')
    report = summarize_review(read_json(root / 'catalog.PRIVATE.json'),
                              read_json(root / 'development_candidates.json'), read_json(args.decisions))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
