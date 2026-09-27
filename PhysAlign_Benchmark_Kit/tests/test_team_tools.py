"""Synthetic-only tests; no original workspace or human review is modified."""
import contextlib
import copy
import io
import json
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'tools'))
import run_demo
import check_review
import share_kit


class TeamToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.demo = Path(cls.temp.name) / 'synthetic'
        with contextlib.redirect_stdout(io.StringIO()):
            run_demo.make_demo(cls.demo)
        def read(rel):
            return json.loads((cls.demo / rel).read_text(encoding='utf-8'))
        cls.catalog = read('draft/catalog.PRIVATE.json')
        cls.development = read('draft/development_candidates.json')
        cls.decisions = read('draft/decisions.template.json')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_demo_covers_three_interfaces_without_real_data(self):
        kinds = [p['metadata']['interface'] for p in self.catalog['probes']]
        self.assertCountEqual(kinds, ['T02-single', 'T03-image', 'T03-text'])
        self.assertEqual({p['problem_id'] for p in self.catalog['probes']}, {'SYNTHETIC_TEST_ONLY'})
        self.assertTrue(all(p['metadata']['P_approved'] is False for p in self.catalog['probes']))

    def test_pending_review_is_not_complete_or_approved(self):
        result = check_review.summarize_review(self.catalog, self.development, self.decisions)
        self.assertEqual(result['status_counts'], {'pending': 3})
        self.assertEqual(len(result['remaining_probe_ids']), 3)
        self.assertEqual(result['complete_mothers'], [])
        self.assertTrue(result['not_a_release_approval'])

    def test_partial_review_keeps_missing_pending(self):
        d = copy.deepcopy(self.decisions)
        d['decisions'] = d['decisions'][:1]
        d['decisions'][0].update(status='rejected', reviewer='SYNTHETIC_TEST_REVIEWER',
                                 blind_answer='SYNTHETIC: ambiguous', note='SYNTHETIC test note')
        result = check_review.summarize_review(self.catalog, self.development, d)
        self.assertEqual(result['status_counts'], {'rejected': 1, 'pending': 2})
        self.assertEqual(len(result['rejected']), 1)

    def test_stale_catalog_and_target_rejected(self):
        d = copy.deepcopy(self.decisions)
        d['catalog_root'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'DIFFERENT_CATALOG'):
            check_review.summarize_review(self.catalog, self.development, d)
        d = copy.deepcopy(self.decisions)
        d['decisions'][0]['review_target_root'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'TARGET_STALE'):
            check_review.summarize_review(self.catalog, self.development, d)

    def test_duplicate_and_non_development_records_rejected(self):
        d = copy.deepcopy(self.decisions)
        d['decisions'].append(copy.deepcopy(d['decisions'][0]))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_PROBE'):
            check_review.summarize_review(self.catalog, self.development, d)
        dev = {**self.development, 'problem_ids': []}
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_NONDEV'):
            check_review.summarize_review(self.catalog, dev, self.decisions)

    def test_approved_requires_real_fields_and_all_confirmations(self):
        d = copy.deepcopy(self.decisions)
        item = d['decisions'][0]
        item.update(status='approved', reviewer='SYNTHETIC_TEST_REVIEWER', blind_answer='SYNTHETIC_TEST_E1')
        with self.assertRaisesRegex(ValueError, 'ALL_CONFIRMATIONS'):
            check_review.summarize_review(self.catalog, self.development, d)
        item['confirmations'] = {k: True for k in item['confirmations']}
        result = check_review.summarize_review(self.catalog, self.development, d)
        self.assertEqual(result['status_counts']['approved'], 1)
        self.assertTrue(result['not_a_release_approval'])

    def test_wrong_boolean_and_unsubmitted_text_are_not_approved(self):
        d = copy.deepcopy(self.decisions)
        item = d['decisions'][0]
        item['confirmations'][next(iter(item['confirmations']))] = 'true'
        with self.assertRaisesRegex(ValueError, 'CONFIRMATIONS_SCHEMA'):
            check_review.summarize_review(self.catalog, self.development, d)

    def test_demo_refuses_existing_output(self):
        with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
            run_demo.make_demo(self.demo)

    def test_sharing_rejects_extra_review_or_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(self.demo, root / 'examples/synthetic_demo')
            with patch.object(share_kit, 'ROOT', root):
                baseline = share_kit.files()
                self.assertTrue(baseline)
                # Temporary files generated solely by a test, not user data.
                extra = root / 'examples/synthetic_demo/accidental_real_review.json'
                extra.write_text('{}', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'UNDECLARED_DEMO_SOURCE_FILE'):
                    share_kit.files()
                extra.unlink()
                (root / '.env').write_text('SYNTHETIC_DUMMY_NOT_A_CREDENTIAL', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'UNDECLARED_FILE'):
                    share_kit.files()

    def test_manifest_refresh_requires_exact_previous_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(self.demo, root / 'examples/synthetic_demo')
            with patch.object(share_kit, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', ['share_kit.py', 'seal']): share_kit.main()
                old = (root / 'KIT_MANIFEST.json').read_bytes()
                (root / 'README.md').write_text('SYNTHETIC updated documentation', encoding='utf-8')
                with patch.object(sys, 'argv', ['share_kit.py', 'refresh', '--previous-manifest-sha256', '0'*64]):
                    with self.assertRaisesRegex(ValueError, 'PREVIOUS_MANIFEST_HASH_MISMATCH'): share_kit.main()
                self.assertEqual((root / 'KIT_MANIFEST.json').read_bytes(), old)
                with patch.object(sys, 'argv', ['share_kit.py', 'refresh', '--previous-manifest-sha256', hashlib.sha256(old).hexdigest()]):
                    share_kit.main()
                with patch.object(sys, 'argv', ['share_kit.py', 'verify']): share_kit.main()

    def test_unknown_html_is_not_added_to_sharing_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(self.demo, root / 'examples/synthetic_demo')
            (root / 'tools').mkdir()
            (root / 'tools/accidental_real_review.html').write_text('SYNTHETIC rejection test', encoding='utf-8')
            with patch.object(share_kit, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'UNDECLARED_FILE'): share_kit.files()


if __name__ == '__main__':
    unittest.main()
