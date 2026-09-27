"""Protocol tests with synthetic models; never call real inference APIs."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from test_two_experiments import fixture, rehash, pipeline
from physalign.adapters import AdapterInfo, ModelResponse
from physalign.controls import original_ids
from physalign.dataset import PublicDataset, sections
from physalign.planning import prepare_plan
from physalign.runner import run_evaluation
from physalign.storage import atomic_json, canonical, file_hash, fingerprint, loads, read_json
from server_eval.supplement_eval import Supplement, draft, prepare, reorder_candidates, run
from server_eval.supplement_source import Release, freeze_keys, original_text
from server_eval.supplement_reporting import score, grading_queue, compare


class SyntheticAdapter:
    info = AdapterInfo('supplement-fixture', 'not-a-real-model', 'fixture', '1', '{}', '{}', False)

    def __init__(self, source):
        self.calls = []
        self.answers = {x['instance_id']: x['answer'] for x in read_json(source / 'private/answers.json')}
        self.by_question = {sections(x['input']['messages']['user'])['Local question']: self.answers[x['instance_id']]
                           for x in read_json(source / 'public/qa_raw.json')}

    def generate(self, req):
        self.calls.append(req)
        if req.user.startswith('SYNTHETIC ORIGINAL'):
            return ModelResponse('{"answer":"A","explanation":"fixture only"}')
        return ModelResponse(canonical(self.by_question[sections(req.user)['Local question']]))


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = fixture(self.root / 'release')
        raw = read_json(self.source / 'public/qa_raw.json')
        members = {m['instance_id']: m for m in read_json(self.source / 'private/membership.json')}
        maps = read_json(self.source / 'private/mappings.json')
        by_id = {m['instance_id']: m for m in maps}
        mothers = set()
        for row in raw:
            iid = row['instance_id']; mother = members[iid]['problem_id']
            evidence = loads(sections(row['input']['messages']['user'])['Evidence locations and candidates'])
            anchor = next(iter(evidence['anchors'].values()))
            by_id[iid]['key']['source_query'] = {'kind': 'quantity', 'locator': anchor}
            if mother in mothers:
                continue
            mothers.add(mother)
            ids = original_ids(row['input']['messages']['user'])
            assets = {a['asset_id']: a for a in row['input']['attachments']}
            manifest = {'problem_id': mother, 'raw_question': 'SYNTHETIC ORIGINAL ' + mother,
                        'segments': {'options': []}, 'source_dataset': 'synthetic-fixture', 'source_sample_id': mother,
                        'images': [{'image_id': i, **{k: assets[i][k] for k in ('sha256', 'width', 'height')}} for i in ids]}
            atomic_json(self.source / ('private/s0/private/project/source/problems/' + mother + '/sources/manifest.json'), manifest)
        atomic_json(self.source / 'private/mappings.json', maps)
        atomic_json(self.source / 'private/merge_request.json', {'components': [{'name': 'fixture', 'relative_path': 'private/s0'}]})
        rehash(self.source)
        pipeline.convert_release(self.source, self.root / 'bundle')
        self.base = prepare_plan(self.root / 'bundle', split='all_provisional', bootstrap_resamples=4)
        draft(self.source, self.base, self.root / 'draft')
        self.keyfile = self.root / 'keys.json'
        self.keys = read_json(self.root / 'draft/answer-keys.template.json')
        for entry in self.keys['problems'].values():
            entry['answer_key'] = {'kind': 'choice', 'accepted': ['A'], 'source_reference': 'synthetic answer', 'reliable': True}
        atomic_json(self.keyfile, self.keys)

    def build(self, **kwargs):
        prepare(self.source, self.base, self.root / 'study', answer_keys=self.keyfile, **kwargs)
        return Supplement(self.root / 'study')

    def test_end_to_end_reads_main_without_writing_and_preserves_resume(self):
        study = self.build()
        adapter = SyntheticAdapter(self.source)
        main = self.root / 'main'
        run_evaluation(self.base, adapter, main, dataset_root=self.root / 'bundle')
        before = {p.relative_to(main).as_posix(): file_hash(p) for p in main.rglob('*') if p.is_file()}
        output = self.root / 'supplement-run'
        run(study, adapter, output)
        calls = len(adapter.calls)
        run(study, adapter, output, resume=True)
        self.assertEqual(len(adapter.calls), calls)
        report = score(study, output, self.root / 'report', main_run=main)
        self.assertEqual(report['original_solving']['SolveAcc'], 1)
        self.assertEqual(report['original_solving']['association']['mean_binding_correct'], 1)
        self.assertEqual(report['control_comparisons']['candidate_order_000']['raw']['metrics']['canonical_consistency'], 1)
        self.assertIsNotNone(report['original_solving']['accuracy_intervals'])
        after = {p.relative_to(main).as_posix(): file_hash(p) for p in main.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_inference_never_reads_answers_and_originals_have_no_probe_content(self):
        study = self.build()
        adapter = SyntheticAdapter(self.source)
        original_open = Path.open
        def checked(path, *args, **kwargs):
            if path.name == 'private.json' or '/private/' in path.as_posix():
                raise AssertionError('Private material read during inference')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', checked):
            run(study, adapter, self.root / 'run')
        for request in adapter.calls:
            if request.user.startswith('SYNTHETIC ORIGINAL'):
                self.assertEqual([i.asset_id for i in request.images], ['img_0'])
                self.assertNotIn('Local reading information', request.user)
                self.assertNotIn('accepted', request.system)

    def test_candidate_reorder_changes_only_array_not_aliases_or_images(self):
        row = read_json(self.source / 'public/qa_raw.json')[0]
        original = row['input']; changed, order = reorder_candidates(original, 2027, row['instance_id'], 0)
        self.assertEqual(original['attachments'], changed['attachments'])
        self.assertEqual(original['messages']['system'], changed['messages']['system'])
        a, b = sections(original['messages']['user']), sections(changed['messages']['user'])
        evidence_a, evidence_b = loads(a.pop('Evidence locations and candidates')), loads(b.pop('Evidence locations and candidates'))
        self.assertEqual(a, b)
        self.assertNotEqual([c['alias'] for c in evidence_a['candidates']], order)
        self.assertEqual(sorted(evidence_a['candidates'], key=lambda c: c['alias']), sorted(evidence_b['candidates'], key=lambda c: c['alias']))
        self.assertEqual(evidence_a['anchors'], evidence_b['anchors'])

    def test_keys_bound_to_input_and_no_key_is_not_wrong_answer(self):
        mother = next(iter(self.keys['problems']))
        self.keys['problems'][mother]['input_hash'] = 'wrong'
        atomic_json(self.keyfile, self.keys)
        with self.assertRaisesRegex(ValueError, 'different original prompt'):
            self.build()
        with self.assertRaisesRegex(ValueError, 'No reliable original answers'):
            prepare(self.source, self.base, self.root / 'nokey')
        plan = prepare(self.source, self.base, self.root / 'unscored', allow_unscored_solving=True)
        study = Supplement(self.root / 'unscored')
        run(study, SyntheticAdapter(self.source), self.root / 'run')
        report = score(study, self.root / 'run', self.root / 'report')
        self.assertEqual(report['original_solving']['answer_eligible_mothers'], 0)
        self.assertIsNone(report['original_solving']['SolveAcc'])
        self.assertEqual(plan['readiness']['answer_eligible_mothers'], 0)

    def test_missing_response_preserves_denominator(self):
        study = self.build()
        run(study, SyntheticAdapter(self.source), self.root / 'run')
        records = list((self.root / 'run/results').glob('*.json'))
        result = next(p for p in records if read_json(p)['condition'] == 'solve.solve')
        result.unlink()  # Simulate interrupted finalization in synthetic fixture only.
        report = score(study, self.root / 'run', self.root / 'report', allow_incomplete=True)
        self.assertEqual(report['original_solving']['answer_eligible_mothers'], 3)
        self.assertIsNone(report['original_solving']['SolveAcc'])
        self.assertEqual(report['original_solving']['SolveAcc_bounds'], {'lower': 2/3, 'upper': 1})

    def test_manual_grade_requires_exact_response_reference_and_identity(self):
        mother = next(iter(self.keys['problems']))
        self.keys['problems'][mother]['answer_key'] = {'kind': 'human', 'reference_answer': 'synthetic A', 'reliable': True, 'source_reference': 'fixture'}
        atomic_json(self.keyfile, self.keys)
        study = self.build()
        run(study, SyntheticAdapter(self.source), self.root / 'run')
        grading_queue(study, self.root / 'run', self.root / 'queue.json')
        queue = read_json(self.root / 'queue.json')
        self.assertEqual(len(queue['grades']), 1)
        queue['grades'][0].update(correct=1, human_confirmed=True, grader_id='test double', rationale='synthetic fixture')
        atomic_json(self.root / 'queue.json', queue)
        self.assertEqual(score(study, self.root / 'run', self.root / 'report', grades_path=self.root / 'queue.json')['original_solving']['SolveAcc'], 1)
        queue['grades'][0]['response_hash'] = 'tampered'
        atomic_json(self.root / 'queue.json', queue)
        with self.assertRaisesRegex(ValueError, 'response/reference changed'):
            score(study, self.root / 'run', self.root / 'bad-report', grades_path=self.root / 'queue.json')

    def test_original_options_preserved_not_overwritten_by_local_query(self):
        manifest = {'raw_question': 'Full original question', 'segments': {'options': [{'label': 'A', 'text': 'original option'}]}}
        self.assertEqual(original_text(manifest), 'Full original question\n\nOriginal answer options:\nA. original option')
        manifest['raw_question'] += '\nA. original option'
        self.assertEqual(original_text(manifest), manifest['raw_question'])

    def test_catalog_restored_images_are_copied_into_portable_study(self):
        source = Release(self.source, self.base)
        originals = source.originals()
        catalog_dir = self.root / 'matched'; catalog_dir.mkdir()
        catalog = {'schema_version': 'physalign_matched_originals_v1', 'source_identity': source.identity, 'problems': {}}
        for mother, original in originals.items():
            inp = deepcopy(original['input'])
            a = deepcopy(inp['attachments'][0])
            data = (self.source / 'public' / a['path']).read_bytes()
            a['path'] = '_supplement_originals/' + a['sha256'] + '.png'
            a['asset_id'] = 'restored-image'
            target = catalog_dir / a['path']; target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
            inp['attachments'].append(a)
            catalog['problems'][mother] = {'match_status': 'matched', 'provenance': {'fixture': True},
                                           'source_input_hash': original['input_hash'], 'input': inp}
            self.keys['problems'][mother]['input_hash'] = fingerprint(inp)
        catalog['catalog_hash'] = fingerprint(catalog)
        atomic_json(catalog_dir / 'originals.json', catalog); atomic_json(self.keyfile, self.keys)
        study = self.build(original_catalog=catalog_dir / 'originals.json')
        self.assertTrue(study.plan['supplement_images'])
        first = next(r for r in study.plan['requests'] if r['kind'] == 'solve')
        request = study.request(first, 'fixture', '{}')
        self.assertEqual(len(request.images), 2)
        self.assertEqual(request.images[0].data, request.images[1].data)
        # A changed catalog cannot be used with previously matched keys.
        catalog['problems'][next(iter(originals))]['input']['messages']['user'] += ' changed'
        atomic_json(catalog_dir / 'originals.json', catalog)
        with self.assertRaisesRegex(ValueError, 'catalog changed'):
            prepare(self.source, self.base, self.root / 'bad-study', original_catalog=catalog_dir / 'originals.json', answer_keys=self.keyfile)

    def test_no_image_keeps_gold_packet_and_reorder_pairing(self):
        study = self.build()
        rows = {(r['instance_id'], r['condition']): r for r in study.plan['requests']}
        for ref in study.plan['reference_inputs']:
            iid, cond = ref['instance_id'], ref['condition']
            row = rows[iid, 'no_image.' + cond]
            self.assertEqual(row['input']['attachments'], [])
            self.assertEqual(row['input']['messages'], ref['input']['messages'])
            if cond == 'gold':
                a = rows[iid, 'candidate_order_000.raw']['input']
                b = rows[iid, 'candidate_order_000.gold']['input']
                self.assertEqual(a['attachments'], b['attachments'])
                self.assertEqual(sections(a['messages']['user'])['Evidence locations and candidates'],
                                 sections(b['messages']['user'])['Evidence locations and candidates'])

    def test_compare_exports_figures_and_refuses_mixed_plan_denominators(self):
        study = self.build()
        paths = []
        for index in range(2):
            adapter = SyntheticAdapter(self.source)
            adapter.info = AdapterInfo('fixture', 'synthetic-' + str(index), 'fixture', '1', '{}', '{}', False)
            run_path, report_path = self.root / f'run-{index}', self.root / f'report-{index}'
            run(study, adapter, run_path)
            score(study, run_path, report_path)
            paths.append(report_path / 'report.json')
        result = compare(paths, self.root / 'comparison')
        self.assertEqual(result['table_rows'], 2)
        self.assertIn('SolveAcc', (self.root / 'comparison/comparison.csv').read_text('utf-8-sig'))
        import importlib.util
        if importlib.util.find_spec('matplotlib') is not None:
            self.assertGreater((self.root / 'comparison/comparison.png').stat().st_size, 1000)
            self.assertGreater((self.root / 'comparison/comparison.pdf').stat().st_size, 1000)
        wrong = read_json(paths[1]); wrong['plan_hash'] = 'another-denominator'
        atomic_json(paths[1], wrong)
        with self.assertRaisesRegex(ValueError, 'one frozen supplement plan'):
            compare(paths, self.root / 'bad-comparison')


if __name__ == '__main__':
    unittest.main()
