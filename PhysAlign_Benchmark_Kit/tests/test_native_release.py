"""Regression evidence uses synthetic drawings/reviewers ONLY."""
import contextlib
import copy
import io
import itertools
import json
import math
import re
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'tools'))
import native_release as release
import native_release_support as support
import run_demo
from io_utils import read_json, write_json, seal_directory, file_manifest
from reference_checks import digest


def approve_template(template):
    d = copy.deepcopy(template)
    d.update(status='approved', reviewer='SYNTHETIC_TEAM_APPROVER', evidence_ref='SYNTHETIC_TEST_NOT_HUMAN')
    d['confirmations'] = {k: True for k in d['confirmations']}
    return d


def decision(item, *, reviewer='SYNTHETIC_A', sequence=1, severity='none'):
    return {'logical_probe_id': item['logical_probe_id'], 'review_target_root': item['review_target_root'],
            'status': 'approved' if severity in ('none', 'minor') else 'rejected', 'reviewer': reviewer,
            'blind_answer': 'SYNTHETIC independent answer', 'confirmations': {k: True for k in support.CONFIRMATIONS},
            'note': 'SYNTHETIC audit evidence', 'severity': severity, 'sequence': sequence}


def document(catalog, context, role, decisions):
    return {'schema_version': 'native_release_reviews_v1', 'catalog_root': digest(catalog),
            'context_root': context, 'role': role, 'decisions': decisions}


class MathTests(unittest.TestCase):
    def test_exact_hypergeom_against_enumerated_subsets(self):
        for N in range(1, 9):
            for n in range(N+1):
                samples = list(itertools.combinations(range(N), n))
                for k in range(n+1):
                    admissible = [D for D in range(N+1)
                                  if 40 * sum(sum(x < D for x in s) <= k for s in samples) >= len(samples)]
                    self.assertEqual(support.hypergeom_upper(N, n, k), max(admissible))

    def test_census_and_no_information(self):
        self.assertEqual(support.hypergeom_upper(521, 0, 0), 521)
        self.assertEqual(support.hypergeom_upper(521, 521, 9), 9)
        self.assertEqual(support.hypergeom_upper(0, 0, 0), 0)

    def test_removal_order_and_fixed_original_denominator(self):
        r = support.finite_risk(531, 100, 1, 1)
        self.assertEqual(r['upper_remaining_numerator'], r['upper_defective_mothers_original'] - 1)
        self.assertEqual(r['upper_remaining_denominator'], 530)
        self.assertEqual(r['detected_sample_mothers'], 1)
        r = support.finite_risk(10, 10, 2, 2)
        self.assertEqual((r['upper_remaining_numerator'], r['upper_remaining_denominator']), (0, 8))
        self.assertTrue(r['passes_5_percent'])

    def test_invalid_counts_and_empty_release(self):
        for args in ((10, 11, 0), (5, 2, 3), (True, 0, 0), (4, -1, 0)):
            with self.assertRaises(ValueError): support.hypergeom_upper(*args)
        self.assertFalse(support.finite_risk(2, 2, 2, 2)['passes_5_percent'])

    def test_monotone_upper_count(self):
        vals = [support.hypergeom_upper(1099, 100, k) for k in range(6)]
        self.assertEqual(vals, sorted(vals))


class AuditLogicTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {'probes': [{'problem_id': str(i), 'logical_probe_id': 'Q'+str(i), 'review_target_root': digest(i)} for i in range(120)]}
        self.plan = {'sample_mothers': [str(i) for i in range(100)], 'risk_mothers': ['100'],
                     'secondary_mothers': [str(i) for i in range(25)], 'candidate_mothers': [str(i) for i in range(120)],
                     'uncovered_risk_tags': [], 'blind_first': {str(i): 'Q'+str(i) for i in range(101)}}
        self.a = {p['logical_probe_id']: decision(p, sequence=i+1) for i,p in enumerate(self.catalog['probes'][:101])}
        self.b = {p['logical_probe_id']: decision(p, sequence=i+1, reviewer='SYNTHETIC_B') for i,p in enumerate(self.catalog['probes'][:25])}

    def stats(self, resolutions=None):
        return support.audit_statistics(self.plan, self.catalog, self.a, self.b, resolutions or {})

    def test_zero_defect_accepts_without_fake_per_item_review(self):
        r = self.stats()
        self.assertTrue(r['accepted'])
        self.assertEqual(r['risk']['sample_mothers'], 100)
        self.assertEqual(len(r['retained_mothers']), 120)

    def test_missing_primary_and_secondary_block(self):
        self.a.pop('Q30'); self.b.pop('Q5')
        r = self.stats()
        self.assertFalse(r['accepted'])
        self.assertIn('Q30', r['pending_primary_probe_ids'])
        self.assertIn('Q5', r['pending_secondary_probe_ids'])

    def test_second_review_missed_serious_escalates_and_keeps_detection(self):
        self.b['Q5'].update(status='rejected', severity='serious')
        r = self.stats()
        self.assertTrue(r['secondary_escalated_to_all_audit_mothers'])
        self.assertFalse(r['accepted'])
        self.assertIn('Q100', r['pending_secondary_probe_ids'])
        self.assertIn('Q5', r['unresolved_probe_ids'])
        self.assertEqual(r['risk']['detected_sample_mothers'], 1)

    def test_same_reviewer_is_not_independent(self):
        self.b['Q0']['reviewer'] = 'SYNTHETIC_A'
        with self.assertRaisesRegex(ValueError, 'INDEPENDENT'): self.stats()

    def test_known_bad_mother_removed_but_count_not_erased(self):
        for slot in (self.a, self.b): slot['Q1'].update(status='rejected', severity='serious')
        r = self.stats()
        self.assertEqual(r['confirmed_excluded_mothers'], ['1'])
        self.assertEqual(r['risk']['detected_sample_mothers'], 1)
        self.assertEqual(r['risk']['remaining_population_mothers'], 119)

    def test_risk_extra_not_in_random_denominator(self):
        self.a['Q100'].update(status='rejected', severity='serious')
        self.b['Q100'] = decision(self.catalog['probes'][100], reviewer='SYNTHETIC_B', sequence=101, severity='serious')
        r = self.stats()
        self.assertEqual(r['risk']['detected_sample_mothers'], 0)
        self.assertEqual(r['risk']['sample_mothers'], 100)
        self.assertEqual(r['confirmed_excluded_mothers'], ['100'])

    def test_one_systemic_issue_blocks_entire_rule_version(self):
        self.a['Q1'].update(status='rejected', severity='systemic')
        self.assertIn('SYSTEMIC_OR_INFORMATION_DEFECT_WITHDRAW_RULE_VERSION', self.stats()['blockers'])

    def test_uncovered_risk_disallows_inheritance(self):
        self.plan['uncovered_risk_tags'] = ['new-rule']
        self.assertIn('RISK_COVERAGE_INCOMPLETE_NO_INHERITANCE', self.stats()['blockers'])

    def test_blind_first_order_preserved(self):
        extra = {**self.catalog['probes'][0], 'logical_probe_id': 'Qextra'}
        self.catalog['probes'].append(extra)
        self.a['Qextra'] = decision(extra, sequence=1)
        self.a['Q0']['sequence'] = 2
        with self.assertRaisesRegex(ValueError, 'BLIND_FIRST'): self.stats()

    def test_adjudicator_cannot_be_first_reviewer(self):
        self.b['Q0'].update(status='rejected', severity='serious')
        with self.assertRaisesRegex(ValueError, 'ADJUDICATOR'):
            self.stats({'Q0': decision(self.catalog['probes'][0], reviewer='SYNTHETIC_A')})


class ReleaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nr_')
        cls.base = Path(cls.tmp.name)
        shutil.copytree(KIT / 'examples/synthetic_demo', cls.base / 's')
        with contextlib.redirect_stdout(io.StringIO()):
            release.project_init(cls.base / 's/draft', cls.base / 'p')
        cls.catalog = read_json(cls.base / 'p/source/catalog.PRIVATE.json')
        cls.project = read_json(cls.base / 'p/project.json')
        cls.legacy = read_json(cls.base / 'p/source/decisions.template.json')
        for d in cls.legacy['decisions']:
            d.update(status='approved', reviewer='SYNTHETIC_HUMAN_FIXTURE', blind_answer='SYNTHETIC FIRST ANSWER')
            d['confirmations'] = {k: True for k in d['confirmations']}
        write_json(cls.base / 'legacy.json', cls.legacy)
        with contextlib.redirect_stdout(io.StringIO()):
            release.import_review(cls.base / 'p', [cls.base / 'legacy.json'], cls.base / 'r')
            release.release_plan(cls.base / 'p', cls.base / 'r', None, 'development', cls.base / 'q')
        write_json(cls.base / 'approval.json', approve_template(read_json(cls.base / 'q/approval.TEMPLATE.json')))
        with contextlib.redirect_stdout(io.StringIO()):
            release.export_release(cls.base / 'p', cls.base / 'q', cls.base / 'approval.json', cls.base / 'out', synthetic=True)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_old_reviews_import_and_complete_mother(self):
        r = read_json(self.base / 'r/receipt.json')
        self.assertEqual(r['summary']['status_counts'], {'approved': 3})
        self.assertEqual(r['original_documents'], [self.legacy])

    def test_public_private_split_and_P_subset(self):
        raw = read_json(self.base / 'out/public/qa_raw.json')
        gold = read_json(self.base / 'out/public/qa_gold.json')
        self.assertEqual((len(raw), len(gold)), (3, 2))
        self.assertNotIn('T03-text', [r['interface'] for r in gold])
        for r in raw + gold:
            self.assertEqual(set(r['input']), {'messages', 'attachments'})
            self.assertFalse({'key','gold_targets','candidate_map','expected_binding_aliases'} & set(r))
            for a in r['input']['attachments']:
                self.assertTrue((self.base / 'out/public' / a['path']).is_file())
                self.assertNotIn('private', a['path'])
        manifest = read_json(self.base / 'out/release.json')
        self.assertEqual(manifest['counts']['L']['probes'], 1)
        self.assertEqual(manifest['artifact_mode'], 'synthetic_example_NOT_benchmark_data')

    def test_source_and_pending_packet_unchanged(self):
        self.assertEqual(file_manifest(self.base / 's/draft'), file_manifest(self.base / 'p/source'))
        for p in self.catalog['probes']:
            self.assertEqual(read_json(self.base / 'p/source' / p['draft'])['key']['packet_plan']['state'], 'pending_review')

    def test_relocated_release_revalidates(self):
        target = self.base / 'moved'
        shutil.copytree(self.base / 'out', target)
        with contextlib.redirect_stdout(io.StringIO()): release.validate_release(target)

    def test_synthetic_data_not_exported_as_real(self):
        with self.assertRaisesRegex(ValueError, 'SYNTHETIC_DATA_CANNOT'):
            with contextlib.redirect_stdout(io.StringIO()):
                release.export_release(self.base / 'p', self.base / 'q', self.base / 'approval.json', self.base / 'badreal')

    def test_development_never_becomes_test(self):
        with self.assertRaisesRegex(ValueError, 'NO_COMPLETE_APPROVED'):
            release.release_plan(self.base / 'p', self.base / 'r', None, 'test', self.base / 'badtest')

    def test_pending_signoff_cannot_export(self):
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_SIGNOFF'):
            release.verified_request(self.base / 'p', self.base / 'q', self.base / 'q/approval.TEMPLATE.json')

    def test_stale_signoff_fails(self):
        bad = read_json(self.base / 'approval.json'); bad['target_root'] = '0'*64
        write_json(self.base / 'bad_sign.json', bad)
        with self.assertRaisesRegex(ValueError, 'WRONG_VERSION_OR_TARGET'):
            release.verified_request(self.base / 'p', self.base / 'q', self.base / 'bad_sign.json')

    def test_missing_approval_does_not_export_partial_mother(self):
        d = copy.deepcopy(self.legacy); d['decisions'].pop()
        write_json(self.base / 'partial.json', d)
        with contextlib.redirect_stdout(io.StringIO()): release.import_review(self.base / 'p', [self.base / 'partial.json'], self.base / 'partial')
        with self.assertRaisesRegex(ValueError, 'NO_COMPLETE_APPROVED'):
            release.release_plan(self.base / 'p', self.base / 'partial', None, 'development', self.base / 'partialplan')

    def test_conflicting_human_files_not_last_writer_wins(self):
        d = copy.deepcopy(self.legacy); d['decisions'][0]['blind_answer'] = 'DIFFERENT_HUMAN_REVIEW'
        with self.assertRaisesRegex(ValueError, 'CONFLICTING_REVIEWS'):
            support.merge_reviews([self.legacy, d], self.catalog, self.project['project_id'])

    def test_old_rejections_require_classification(self):
        d = copy.deepcopy(self.legacy); d['decisions'][0].update(status='rejected', note='source issue')
        normalized = support.merge_reviews([d], self.catalog, self.project['project_id'])
        self.assertEqual(next(iter(normalized.values()))['severity'], 'unresolved')

    def test_legacy_cannot_be_relabelled_as_random_audit(self):
        with self.assertRaisesRegex(ValueError, 'LEGACY_REVIEW_NOT_RANDOM_AUDIT'):
            support.normalize_reviews(self.legacy, self.catalog, context_root='audit-root', role='primary')

    def test_public_answer_injection_rejected_even_if_resealed(self):
        target = self.base / 'tamper'
        shutil.copytree(self.base / 'out', target)
        rows = read_json(target / 'public/qa_raw.json'); rows[0]['answer'] = 'E2'
        write_json(target / 'public/qa_raw.json', rows); seal_directory(target)
        with self.assertRaisesRegex(ValueError, 'PUBLIC_QA_REBUILD'):
            with contextlib.redirect_stdout(io.StringIO()): release.validate_release(target)

    def test_extra_private_file_in_public_rejected_even_if_resealed(self):
        target = self.base / 'leak'
        shutil.copytree(self.base / 'out', target)
        write_json(target / 'public/FILE_MANIFEST.json', {'private_answer': 'E2'}); seal_directory(target)
        with self.assertRaisesRegex(ValueError, 'UNDECLARED_PUBLIC'):
            with contextlib.redirect_stdout(io.StringIO()): release.validate_release(target)

    def test_changed_image_rejected(self):
        target = self.base / 'badimage'
        shutil.copytree(self.base / 'out', target)
        image = next((target / 'public/images').glob('*'))
        image.write_bytes(b'SYNTHETIC CORRUPT IMAGE'); seal_directory(target)
        with self.assertRaisesRegex(ValueError, 'PUBLIC_IMAGE_BYTES'):
            with contextlib.redirect_stdout(io.StringIO()): release.validate_release(target)

    def test_output_cannot_modify_kit_or_existing_project(self):
        for target in (KIT / 'real_output', self.base / 'p/real_output'):
            with self.assertRaisesRegex(ValueError, 'OUTSIDE_INPUT_AND_KIT'):
                release.check_output(target, self.base / 'p')

    def test_version_fingerprints_separate_from_original_converter(self):
        self.assertEqual(self.catalog['implementation_hashes'], release.dependency_hashes())
        self.assertEqual(set(support.implementation_hashes()), {'native_release.py','native_release_support.py','native_release_review.html'})

    def test_new_review_page_javascript_syntax(self):
        node = shutil.which('node')
        if not node: self.skipTest('node unavailable; not a browser test')
        text = (self.base / 'p/review.html').read_text(encoding='utf-8')
        self.assertNotIn('__DATA__', text)
        scripts = re.findall(r'<script>(.*?)</script>', text, re.DOTALL)
        self.assertEqual(len(scripts), 1)
        proc = subprocess.run([node, '--check'], input=scripts[0], capture_output=True, encoding='utf-8')
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_new_review_page_dom_workflow(self):
        node = shutil.which('node')
        if not node: self.skipTest('node unavailable')
        proc = subprocess.run([node, str(KIT / 'tests/review_dom_smoke.js'), str(self.base / 'p/review.html')],
                              capture_output=True, encoding='utf-8')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_windows_transient_publish_lock_retries_without_overwrite(self):
        source = self.base / 'rename_stage'; source.mkdir()
        write_json(source / 'x.json', {'synthetic': True})
        destination = self.base / 'rename_final'
        original = Path.rename
        attempts = []
        def renamed(path, target):
            attempts.append(str(path))
            if len(attempts) < 3:
                error = PermissionError('SYNTHETIC_WINDOWS_LOCK'); error.winerror = 5
                raise error
            return original(path, target)
        with patch.object(Path, 'rename', renamed), patch.object(release.time, 'sleep'):
            release.publish(source, destination)
        self.assertEqual(len(attempts), 3)
        self.assertTrue((destination / 'x.json').is_file())
        with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
            release.publish(self.base / 'nonexistent', destination)

    def test_manifest_public_root_cannot_be_changed_to_private(self):
        target = self.base / 'badroot'
        shutil.copytree(self.base / 'out', target)
        m = read_json(target / 'release.json'); m['public_root'] = 'private'
        write_json(target / 'release.json', m); seal_directory(target)
        with self.assertRaisesRegex(ValueError, 'RELEASE_MANIFEST_REBUILD'):
            with contextlib.redirect_stdout(io.StringIO()): release.validate_release(target)


class BatchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='nb_')
        cls.base = Path(cls.tmp.name)
        shutil.copytree(KIT / 'examples/synthetic_demo', cls.base / 'seed')
        annotation = cls.base / 'seed/source/annotation'
        row = read_json(annotation / 'blind/manifest.jsonl')
        oldpid = row['problem_id']
        docs = {s: read_json(annotation / 'passes' / s / (oldpid+'.json')) for s in ('pass1','pass2','pass3','pass4')}
        rows, state = [], {'problems': {}}
        for i in range(26):
            pid = 'SYNTHETIC_MOTHER_' + str(i).zfill(2)
            r = copy.deepcopy(row)
            r.update(problem_id=pid, source_sample_id=pid)
            r['segments']['stem'] += ' Synthetic case ' + str(i) + '.'
            r['raw_question'] = r['segments']['stem'] + ' ' + r['segments']['query']
            rows.append(r)
            state['problems'][pid] = {'stages': {}}
            for stage, doc in docs.items():
                d = copy.deepcopy(doc); d['problem_id'] = pid
                write_json(annotation / 'passes' / stage / (pid+'.json'), d)
                state['problems'][pid]['stages'][stage] = {'status': 'approved', 'reviewer': 'SYNTHETIC_FIXTURE', 'document_sha256': digest(d)}
        (annotation / 'blind/manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
        write_json(annotation / 'reviews/state.json', state)
        # Only the synthetic fixture publisher uses the bounded Windows rename
        # retry; the production source converter bytes/semantics stay unchanged.
        with contextlib.redirect_stdout(io.StringIO()), patch('convert.publish', release.publish):
            run_demo.convert_main(['native-draft','--workspace',str(annotation),'--dataset-root',str(cls.base / 'seed/source/dataset'),'--output',str(cls.base / 'd')])
            release.project_init(cls.base / 'd', cls.base / 'p')
        cls.catalog = read_json(cls.base / 'p/source/catalog.PRIVATE.json')
        cls.project = read_json(cls.base / 'p/project.json')
        cls.membership = read_json(cls.base / 'p/source/membership.json')
        dev = set(cls.membership['development_problem_ids'])
        assert len(dev) == 25, 'fixture must follow real default development policy'
        cls.dev_doc = document(cls.catalog, cls.project['project_id'], 'individual',
                              [decision(p, sequence=i+1) for i,p in enumerate(cls.catalog['probes']) if p['problem_id'] in dev])
        write_json(cls.base / 'dev.json', cls.dev_doc)
        with contextlib.redirect_stdout(io.StringIO()):
            release.import_review(cls.base / 'p', [cls.base / 'dev.json'], cls.base / 'devr')
            release.campaign_init(cls.base / 'campaign')
        write_json(cls.base / 'policy.json', approve_template(read_json(cls.base / 'campaign/policy.TEMPLATE.json')))
        with contextlib.redirect_stdout(io.StringIO()):
            release.audit_plan(cls.base / 'p', cls.base / 'campaign', cls.base / 'policy.json', cls.base / 'devr')
        cls.plan = read_json(cls.base / 'campaign/round1/audit_plan.json')
        byid = {p['logical_probe_id']: p for p in cls.catalog['probes']}
        groups = support.mother_groups(cls.catalog)
        ids = []
        for pid in cls.plan['sample_mothers'] + cls.plan['risk_mothers']:
            first = cls.plan['blind_first'][pid]
            ids += [first] + [q for q in groups[pid] if q != first]
        cls.primary = document(cls.catalog, cls.plan['audit_id'], 'primary', [decision(byid[q], sequence=i+1) for i,q in enumerate(ids)])
        cls.secondary = document(cls.catalog, cls.plan['audit_id'], 'secondary', [decision(byid[q], sequence=i+1, reviewer='SYNTHETIC_B') for i,q in enumerate(ids)])
        write_json(cls.base / 'a.json', cls.primary); write_json(cls.base / 'b.json', cls.secondary)
        with contextlib.redirect_stdout(io.StringIO()):
            release.audit_assess(cls.base / 'p', cls.base / 'campaign/round1', cls.base / 'a.json', cls.base / 'b.json', None, cls.base / 'assessment')
            release.release_plan(cls.base / 'p', None, cls.base / 'assessment', 'test', cls.base / 'request')
        write_json(cls.base / 'approval.json', approve_template(read_json(cls.base / 'request/approval.TEMPLATE.json')))

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_real_pipeline_batch_plan_and_census(self):
        self.assertEqual(len(self.plan['candidate_mothers']), 1)
        self.assertEqual(len(self.plan['sample_mothers']), 1)
        record = read_json(self.base / 'assessment/assessment.json')
        self.assertTrue(record['result']['accepted'])
        self.assertEqual(record['result']['risk']['upper_defective_mothers_original'], 0)

    def test_audited_subset_is_explicit_and_not_population_inheritance(self):
        with contextlib.redirect_stdout(io.StringIO()):
            release.release_plan(self.base / 'p', None, self.base / 'assessment', 'test', self.base / 'subset', audited_subset=True)
        req = read_json(self.base / 'subset/request.json')
        self.assertEqual(req['mode'], 'audited_individual')
        self.assertEqual(req['coverage_limit'], 'selected_fully_reviewed_mothers_not_population_audit')
        write_json(self.base / 'subset_sign.json', approve_template(read_json(self.base / 'subset/approval.TEMPLATE.json')))
        release.verified_request(self.base / 'p', self.base / 'subset', self.base / 'subset_sign.json')

    def test_batch_release_end_to_end(self):
        with contextlib.redirect_stdout(io.StringIO()):
            release.export_release(self.base / 'p', self.base / 'request', self.base / 'approval.json', self.base / 'out', synthetic=True)
        manifest = read_json(self.base / 'out/release.json')
        self.assertEqual(manifest['split'], 'test')
        self.assertEqual(manifest['approval_mode'], 'batch')
        self.assertEqual(manifest['counts']['B']['mothers'], 1)
        self.assertEqual({m['human_item_review'] for m in read_json(self.base / 'out/private/membership.json')}, {'approved_sampled'})

    def test_unsampled_authorization_is_not_faked_human(self):
        proof = read_json(self.base / 'assessment/assessment.json')
        item = next(p for p in self.catalog['probes'] if p['problem_id'] not in self.plan['sample_mothers'])
        auth = release.authorization_for(item, read_json(self.base / 'request/request.json'), proof, read_json(self.base / 'approval.json'))
        self.assertEqual(auth['human_item_review'], 'not_sampled')
        self.assertEqual(auth['review_evidence'], [])

    def test_sample_cannot_be_reselected_in_existing_round(self):
        with self.assertRaisesRegex(ValueError, 'SECOND_ROUND_REQUIRES'):
            with contextlib.redirect_stdout(io.StringIO()):
                release.audit_plan(self.base / 'p', self.base / 'campaign', self.base / 'policy.json', self.base / 'devr')

    def test_accepted_round_cannot_be_rerolled(self):
        with self.assertRaisesRegex(ValueError, 'PREVIOUS_ROUND_NOT_FAILED'):
            with contextlib.redirect_stdout(io.StringIO()):
                release.audit_plan(self.base / 'p', self.base / 'campaign', self.base / 'policy.json', self.base / 'devr', self.base / 'assessment')

    def test_sample_tamper_detected_after_resealing(self):
        target = self.base / 'badplan'
        shutil.copytree(self.base / 'campaign/round1', target)
        plan = read_json(target / 'audit_plan.json'); plan['sample_mothers'] = []
        write_json(target / 'audit_plan.json', plan); seal_directory(target)
        with self.assertRaisesRegex(ValueError, 'PLAN_RECOMPUTATION'):
            release.load_audit(target, self.project, self.catalog, self.membership)

    def test_maximum_two_failed_rounds_and_chain(self):
        campaign = self.base / 'two_rounds'
        shutil.copytree(self.base / 'campaign', campaign)
        a = copy.deepcopy(self.primary); b = copy.deepcopy(self.secondary)
        for doc in (a,b):
            for d in doc['decisions']: d.update(status='rejected', severity='systemic')
        write_json(self.base / 'bad_a.json', a); write_json(self.base / 'bad_b.json', b)
        with contextlib.redirect_stdout(io.StringIO()):
            release.audit_assess(self.base / 'p', campaign / 'round1', self.base / 'bad_a.json', self.base / 'bad_b.json', None, self.base / 'failed1')
        with self.assertRaisesRegex(ValueError, 'NEW_CANDIDATE_VERSION'):
            with contextlib.redirect_stdout(io.StringIO()):
                release.audit_plan(self.base / 'p', campaign, self.base / 'policy.json', self.base / 'devr', self.base / 'failed1')
        # A second reserved attempt must stop a third even if interrupted;
        # this synthetic directory is not a fabricated accepted audit.
        (campaign / 'round2').mkdir()
        with self.assertRaisesRegex(ValueError, 'AT_MOST_TWO'):
            with contextlib.redirect_stdout(io.StringIO()):
                release.audit_plan(self.base / 'p', campaign, self.base / 'policy.json', self.base / 'devr', self.base / 'failed1')

    def test_known_error_and_unsampled_affected_rule_carry_to_retry(self):
        a = copy.deepcopy(self.primary); b = copy.deepcopy(self.secondary)
        for d in a['decisions']: d.update(status='rejected', severity='systemic')
        for d in b['decisions']: d.update(status='rejected', severity='systemic')
        prior = release.assessment_record(self.plan, self.catalog, a, b, None)
        project = {**self.project, 'project_id': 'SYNTHETIC_NEW_VERSION'}
        with self.assertRaisesRegex(ValueError, 'UNCHANGED_IN_RETRY'):
            release.check_retry_revision(prior, project, self.catalog)
        revised = copy.deepcopy(self.catalog)
        for p in revised['probes']:
            p['review_target_root'] = digest(['SYNTHETIC_REPAIRED', p['review_target_root']])
        release.check_retry_revision(prior, project, revised)
        unaffected_by_sample = next(p for p in revised['probes'] if p['problem_id'] not in self.plan['sample_mothers'])['problem_id']
        original = {p['logical_probe_id']: p for p in self.catalog['probes']}
        for p in revised['probes']:
            if p['problem_id'] == unaffected_by_sample:
                p['review_target_root'] = original[p['logical_probe_id']]['review_target_root']
        with self.assertRaisesRegex(ValueError, 'UNCHANGED_IN_RETRY'):
            release.check_retry_revision(prior, project, revised)


if __name__ == '__main__': unittest.main()
