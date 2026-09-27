"""Scientific integration checks; all people/models here are explicit test doubles."""

from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from test_evaluation import SAMPLES, RecordingAdapter, native_bundle, rehash
from physalign.adapters import ModelResponse, SmokeAdapter
from physalign.controls import boundary_distance_squared, nearest_region, permutation, sections
from physalign.dataset import PublicDataset
from physalign.hf_adapter import freeze_snapshot, transport_messages, verify_snapshot
from physalign.human import AUDIT_CHECKS, create_session, human_report, public_task, seal_study, submit
from physalign.panel import panel_report
from physalign.solving import grade_original, validate_answer_key
from physalign.storage import atomic_json, canonical, fingerprint, read_json
from physalign.study import SOLVE_SYSTEM, Study, draft_spec, prepare_study, run_study, verify_final_seal
from physalign.study_reporting import export_grading_queue, score_study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='physalign-study-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = native_bundle(self.root / 'bundle')
        self.study_path, self.run = self.root / 'study', self.root / 'run'
        # Make physical candidate identity distinguishable by public geometry.
        for filename in ('public/qa_raw.json', 'public/qa_gold.json'):
            rows = read_json(self.bundle / filename)
            for r in rows:
                parts = sections(r['input']['messages']['user'])
                e = json.loads(parts['Evidence locations and candidates'])
                e['candidates'][1]['locators'][0]['geometry']['bbox_1000'] = [500, 500, 600, 600]
                parts['Evidence locations and candidates'] = canonical(e)
                from physalign.controls import compose
                r['input']['messages']['user'] = compose(parts)
            atomic_json(self.bundle / filename, rows)
        rehash(self.bundle)
        self.spec = draft_spec(self.bundle)
        for mother in ('A', 'B', 'C'):
            self.spec['original_problems'][mother] = {
                'original': {'system': SOLVE_SYSTEM, 'user': 'ORIGINAL ' + mother,
                             'asset_ids': ['img_0'], 'source_reference': 'synthetic-fixture',
                             'unchanged_original_verified': True},
                'answer_key': {'kind': 'choice', 'accepted': ['A'], 'reliable': True, 'source_reference': 'synthetic-fixture-key'}}
        for iid in ('p1', 'p2', 'p3'):
            self.spec['controls'][iid]['nearest'] = {'field': 'owner', 'anchor_id': 'R1', 'eligibility_basis': 'synthetic ownership fixture'}

    def prepare(self, **kwargs):
        path = self.root / 'spec.json'
        atomic_json(path, self.spec)
        return prepare_study(self.bundle, self.study_path, split='development', spec_path=path,
                             bootstrap_resamples=kwargs.pop('bootstrap_resamples', 0), **kwargs)

    @staticmethod
    def physical_answer(request):
        if request.user.startswith('ORIGINAL '):
            return ModelResponse('{"answer":"A"}')
        parts = sections(request.user)
        e = json.loads(parts['Evidence locations and candidates'])
        candidate = next(c for c in e['candidates'] if c.get('locators', [{}])[0].get('geometry', {}).get('bbox_1000', [0])[0] == 10)
        return ModelResponse(canonical({'read': '2 kg', 'owner': candidate['alias']}))

    def test_full_study_is_correct_under_alias_and_order_changes(self):
        plan = self.prepare(bootstrap_resamples=12)
        self.assertEqual(len(plan['requests']), 48)
        adapter = RecordingAdapter(self.physical_answer)
        run_study(self.study_path, adapter, self.run)
        report = score_study(self.study_path, self.run)
        self.assertEqual(len(adapter.calls), 48)
        self.assertEqual(report['experiments']['main']['raw_all']['metrics']['BAcc'], 1)
        for variant in ('perm_alias_000', 'perm_order_000', 'perm_both_000'):
            for condition in ('raw', 'gold'):
                result = report['control_comparisons'][variant][condition]['metrics']
                self.assertEqual((result['delta_BAcc'], result['canonical_consistency']), (0, 1))
        solve = report['original_solving']
        self.assertEqual(solve['SolveAcc'], 1)
        self.assertIsNone(solve['association']['delta_assoc'])  # Empty wrong-answer group.
        self.assertEqual(report['nearest_region']['metrics']['BAcc_control'], 1)

    def test_original_solving_and_association_match_independent_mother_math(self):
        self.prepare()
        def answer(req):
            if req.user.startswith('ORIGINAL '):
                return ModelResponse('{"answer":"A"}' if req.user == 'ORIGINAL A' else '{"answer":"B"}')
            parts = sections(req.user)
            good = parts['Local question'] in {'Probe p1', 'Probe p4'}
            return ModelResponse(canonical({'read': '2 kg', 'owner': 'E1' if good else 'E2'}))
        run_study(self.study_path, RecordingAdapter(answer), self.run)
        result = score_study(self.study_path, self.run)['original_solving']
        self.assertAlmostEqual(result['SolveAcc'], 1 / 3)
        self.assertAlmostEqual(result['association']['binding_by_problem']['A'], 2 / 3)
        self.assertEqual(result['association']['mean_binding_wrong'], 0)
        self.assertAlmostEqual(result['association']['delta_assoc'], 2 / 3)
        self.assertEqual(result['correctly_solved_with_binding_errors'], ['A'])

    def test_no_images_really_means_zero_pixel_payload_and_geometry_is_disclosed(self):
        self.spec['controls']['p1'].update(no_geometry_compatible=True, no_geometry_basis='Explicit synthetic interface check')
        plan = self.prepare()
        rows = {r['condition']: r for r in plan['requests'] if r['instance_id'] == 'p1'}
        raw = rows['main.raw']['input']
        ng = rows['no_image_geometry.raw']['input']
        self.assertEqual(raw['messages'], ng['messages'])
        self.assertFalse(ng['attachments'])
        strict = rows['no_image_no_geometry.raw']['input']
        self.assertFalse(strict['attachments'])
        self.assertNotIn('bbox_1000', strict['messages']['user'])
        self.assertNotIn('object-a', strict['messages']['user'])
        adapter = RecordingAdapter()
        run_study(self.study_path, adapter, self.run)
        self.assertEqual(sum(not r.images for r in adapter.calls), 6)

    def test_runtime_and_blind_human_path_do_not_open_private_keys(self):
        self.prepare()
        study = Study(self.study_path)
        session_path = create_session(self.study_path, self.root / 'humans', participant='reader', cohort=0)
        session = read_json(session_path / 'session.json')
        original_open = Path.open
        def forbid_private(path, *args, **kwargs):
            if path.name == 'private.json' or path.resolve().is_relative_to((self.bundle / 'private').resolve()):
                raise AssertionError('Hidden answer accessed on public path')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', forbid_private):
            run_study(self.study_path, SmokeAdapter(), self.run)
            task = public_task(study, session, session['tasks'][0])
        self.assertNotIn('private_reference', task)
        self.assertNotIn('object-a', canonical(task))

    def test_original_context_has_no_probe_views_or_gold(self):
        plan = self.prepare()
        rows = [r for r in plan['requests'] if r['kind'] == 'solve']
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row['input']['messages']['user'], 'ORIGINAL ' + row['problem_id'])
            self.assertEqual([a['asset_id'] for a in row['input']['attachments']], ['img_0'])
            self.assertNotIn('R1', canonical(row['input']))

    def test_permutation_reference_and_gold_use_identical_rendered_images(self):
        plan = self.prepare()
        rows = {(r['instance_id'], r['condition']): r for r in plan['requests']}
        for variant in ('permutation_reference', 'perm_alias_000', 'perm_order_000', 'perm_both_000'):
            a, b = rows['p1', variant + '.raw'], rows['p1', variant + '.gold']
            self.assertEqual(a['input']['attachments'], b['input']['attachments'])
        main = rows['p1', 'main.raw']['input']['attachments'][0]
        ref = rows['p1', 'permutation_reference.raw']['input']['attachments'][0]
        self.assertEqual(ref['height'], main['height'] + 32)
        from PIL import Image
        with Image.open(self.study_path / main['path']) as a, Image.open(self.study_path / ref['path']) as b:
            # A region far from the one bounding-box outline retains source pixels.
            self.assertEqual(a.convert('RGB').crop((300, 250, 500, 300)).tobytes(), b.crop((300, 282, 500, 332)).tobytes())

    def test_invalid_outputs_are_not_counted_as_stable_physical_predictions(self):
        self.prepare()
        run_study(self.study_path, SmokeAdapter(), self.run)
        report = score_study(self.study_path, self.run)
        self.assertEqual(report['control_comparisons']['perm_alias_000']['raw']['metrics']['canonical_consistency'], 0)

    def test_manual_original_grading_is_tied_to_exact_response_and_reference(self):
        self.spec['original_problems']['A']['answer_key'] = {'kind': 'human', 'reliable': True,
            'source_reference': 'synthetic rubric', 'reference_answer': 'Synthetic correct answer A'}
        self.prepare()
        run_study(self.study_path, RecordingAdapter(self.physical_answer), self.run)
        self.assertIsNone(score_study(self.study_path, self.run)['original_solving']['SolveAcc'])
        path = self.root / 'grades.json'
        queue = export_grading_queue(self.study_path, self.run, path)
        self.assertEqual(len(queue['grades']), 1)
        queue['grades'][0].update(correct=1, human_confirmed=True, grader_id='test-double', rationale='Synthetic fixture only')
        atomic_json(path, queue)
        self.assertEqual(score_study(self.study_path, self.run, grades_path=path)['original_solving']['SolveAcc'], 1)
        queue['grades'][0]['response_hash'] = 'wrong-run'
        atomic_json(path, queue)
        with self.assertRaisesRegex(ValueError, 'response/reference changed'):
            score_study(self.study_path, self.run, grades_path=path)

    def test_absent_answer_keys_are_not_wrong_answers_or_inferred_from_graph(self):
        self.spec['original_problems']['A']['answer_key'] = None
        self.prepare(bootstrap_resamples=8)
        run_study(self.study_path, RecordingAdapter(self.physical_answer), self.run)
        result = score_study(self.study_path, self.run)['original_solving']
        self.assertEqual(result['answer_eligible_mothers'], 2)
        self.assertEqual(result['ineligible_no_reliable_answer'], ['A'])
        self.assertEqual(result['SolveAcc'], 1)
        self.assertEqual(result['intervals']['n_clusters'], 2)

    def test_human_counterbalance_first_answers_audit_and_seal(self):
        self.prepare()
        study = Study(self.study_path)
        # A later seal must not retroactively label earlier model runs as audited.
        run_study(self.study_path, SmokeAdapter(), self.run)
        sessions = self.root / 'humans'
        assignments = []
        for cohort in (0, 1):
            path = create_session(self.study_path, sessions, participant=f'fixture-reader-{cohort}', cohort=cohort)
            s = read_json(path / 'session.json')
            assignments.append({t['instance_id']: t['condition'] for t in s['tasks']})
            for t in s['tasks']:
                response = {'text': '{"read":"2 kg","owner":"E1"}', 'ambiguous': False,
                            'notes': 'Synthetic test fixture, not real participant data', 'human_confirmed': True}
                submit(study, path, t['task_id'], response)
                with self.assertRaises(FileExistsError):
                    submit(study, path, t['task_id'], response)
        for iid in ('p1', 'p2', 'p3'):
            self.assertNotEqual(assignments[0][iid], assignments[1][iid])
        with self.assertRaisesRegex(ValueError, 'cannot cross'):
            create_session(self.study_path, sessions, participant='fixture-reader-0', cohort=1)
        result = human_report(self.study_path, sessions)
        self.assertTrue(result['complete_answer_coverage'])
        self.assertFalse(result['audit_complete_and_accepted'])
        with self.assertRaisesRegex(ValueError, 'audit'):
            seal_study(self.study_path, sessions)
        path = create_session(self.study_path, sessions, participant='fixture-auditor', mode='audit')
        for task in read_json(path / 'session.json')['tasks']:
            submit(study, path, task['task_id'], {'decision': 'accept', 'checks': dict.fromkeys(AUDIT_CHECKS, True),
                                                'notes': 'Synthetic fixture only', 'human_confirmed': True})
        seal = seal_study(self.study_path, sessions)
        self.assertEqual(seal['plan_hash'], study.plan['plan_hash'])
        self.assertEqual(seal['human_report']['summaries']['raw']['JAcc']['value'], 1)
        self.assertEqual(seal['human_report']['paired']['delta_BAcc_gold_raw'], 0)
        self.assertIsNone(score_study(self.study_path, self.run)['human_same_interface'])
        sealed_run = self.root / 'sealed-run'
        run_study(self.study_path, RecordingAdapter(self.physical_answer), sealed_run)
        report = score_study(self.study_path, sealed_run)
        self.assertEqual(report['study_seal_hash'], seal['seal_hash'])
        self.assertEqual(report['human_model_same_subset']['gold']['BAcc']['human_minus_model'], 0)
        self.assertEqual(report['readiness']['human_audit'], 'sealed_before_inference')

    def test_human_display_dimensions_and_adjudication_history_are_auditable(self):
        self.prepare()
        study = Study(self.study_path)
        root = self.root / 'humans'
        path = create_session(self.study_path, root, participant='fixture-reader')
        session = read_json(path / 'session.json')
        task = session['tasks'][0]
        row = study.by_key[task['instance_id'], task['condition']]
        dimensions = [{'asset_id': a['asset_id'], 'natural_width': a['width'], 'natural_height': a['height'],
                       'displayed_width': a['width'], 'displayed_height': a['height']} for a in row['input']['attachments']]
        response = {'text': '{}', 'ambiguous': True, 'notes': 'Synthetic ambiguity for adjudication test',
                    'human_confirmed': True, 'display': dimensions}
        bad = deepcopy(response)
        bad['display'][0]['natural_width'] += 1
        with self.assertRaisesRegex(ValueError, 'different image dimensions'):
            submit(study, path, task['task_id'], bad)
        submit(study, path, task['task_id'], response)
        judge = create_session(self.study_path, root, participant='fixture-judge', mode='adjudicate')
        judge_session = read_json(judge / 'session.json')
        judge_task = next(t for t in judge_session['tasks'] if t['task_id'] == task['task_id'])
        visible = public_task(study, judge_session, judge_task, root)
        self.assertEqual(visible['prior_reviews'][0]['response']['notes'], response['notes'])
        self.assertEqual(human_report(self.study_path, root)['display_audit_coverage']['with_browser_dimensions'], 1)

    def test_failed_audit_cannot_be_approved(self):
        self.prepare()
        study = Study(self.study_path)
        path = create_session(self.study_path, self.root / 'humans', participant='fixture-auditor', mode='audit')
        task = read_json(path / 'session.json')['tasks'][0]
        checks = dict.fromkeys(AUDIT_CHECKS, True)
        checks['packet_no_binding_leakage'] = False
        with self.assertRaisesRegex(ValueError, 'failed audit'):
            submit(study, path, task['task_id'], {'decision': 'accept', 'checks': checks, 'notes': '', 'human_confirmed': True})

    def test_panel_differences_keep_shared_mothers_and_all_planned_requests(self):
        self.prepare(bootstrap_resamples=8)
        first, second = RecordingAdapter(self.physical_answer), RecordingAdapter()
        first.info = replace(first.info, model_id='Qwen/Qwen3.5-9B')
        second.info = replace(second.info, model_id='Qwen/Qwen3.5-27B')
        run_study(self.study_path, first, self.run)
        other = self.root / 'other'
        run_study(self.study_path, second, other)
        report = panel_report(self.study_path, [self.run, other], self.root / 'panel.json')
        result = next(iter(report['comparisons'].values()))
        self.assertEqual(result['differences']['BAcc'], -1)
        self.assertEqual(result['intervals']['n_clusters'], 3)
        self.assertEqual(result['paired_intervals']['n_clusters'], 2)


class ControlMathTests(unittest.TestCase):
    def test_boundary_distance_not_center_distance_and_pixel_aspect_ratio(self):
        self.assertEqual(boundary_distance_squared([0, 0, 100, 100], [50, 50, 200, 200], 1000, 1000), 0)
        self.assertEqual(boundary_distance_squared([0, 0, 100, 100], [100, 0, 200, 100], 1000, 1000), 0)
        self.assertEqual(boundary_distance_squared([0, 0, 100, 100], [200, 300, 400, 500], 2000, 1000), 80000)

    def test_permutations_are_bijective_seeded_and_separate_factors(self):
        aliases = ['E1', 'E2', 'E3']
        for mode in ('alias', 'order', 'both'):
            first = permutation(aliases, seed=7, probe_id='p', index=0, mode=mode)
            self.assertEqual(first, permutation(aliases, seed=7, probe_id='p', index=0, mode=mode))
            self.assertEqual(set(first[0].values()), set(aliases))
            self.assertEqual(sorted(first[1]), [0, 1, 2])
            if mode == 'alias':
                self.assertEqual(first[1], [0, 1, 2])
            if mode == 'order':
                self.assertEqual(first[0], dict(zip(aliases, aliases)))

    def test_grading_contracts_do_not_guess_free_text_or_quantity_tolerance(self):
        k = {'kind': 'choice', 'accepted': ['B'], 'reliable': True, 'source_reference': 'fixture'}
        self.assertEqual(grade_original('{"answer":"B"}', k), 1)
        self.assertEqual(grade_original('I think B', k), 0)
        self.assertEqual(grade_original('{"answer":"b"}', k), 0)
        q = {'kind': 'quantity', 'unit_scales': {'kg': '1', 'g': '.001'}, 'absolute_tolerance': '0',
             'expected': '2 kg', 'reliable': True, 'source_reference': 'fixture'}
        self.assertEqual(grade_original('{"answer":"2000 g"}', q), 1)
        self.assertEqual(grade_original('{"answer":"2000.001 g"}', q), 0)

    def test_hf_message_transport_has_exact_public_text_and_ordered_image_objects(self):
        ds = PublicDataset(SAMPLES)
        from physalign.adapters import ModelRequest
        r = ds.items['sample_04', 'raw']
        m = r.record['input']['messages']
        request = ModelRequest('id', m['system'], m['user'], ds.images(r), '{}')
        messages = transport_messages(request, lambda data: data)
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        self.assertEqual(messages[1]['content'][0]['text'], request.user)
        self.assertEqual([m['image'] for m in messages[1]['content'] if m['type'] == 'image'], [a.data for a in request.images])

    def test_local_snapshot_freeze_detects_changed_weights_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root / 'model'
            model.mkdir()
            atomic_json(model / 'config.json', {'model_type': 'qwen3_5'})
            (model / 'model.safetensors').write_bytes(b'synthetic weights fixture')
            snapshot = freeze_snapshot(model, 'Qwen/Qwen3.5-9B', root / 'snapshot.json')
            self.assertEqual(verify_snapshot(snapshot), model.resolve())
            (model / 'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Checkpoint file changed'):
                verify_snapshot(snapshot)

    def test_strict_no_image_removes_visual_geometry_but_keeps_original_text_locators(self):
        from physalign.controls import compose, no_image_input
        item = deepcopy(PublicDataset(SAMPLES).items['sample_04', 'raw'].record['input'])
        parts = sections(item['messages']['user'])
        e = json.loads(parts['Evidence locations and candidates'])
        text = {'kind': 'text', 'section': 'statement', 'start': 0, 'end': 1, 'text': 'x'}
        e['candidates'][0]['locators'].append(text)
        parts['Evidence locations and candidates'] = canonical(e)
        item['messages']['user'] = compose(parts)
        out = no_image_input(item, remove_geometry=True)
        after = json.loads(sections(out['messages']['user'])['Evidence locations and candidates'])
        self.assertEqual(after['candidates'][0]['locators'], [text])
        self.assertNotIn('bbox_1000', out['messages']['user'])


class StarterStudyTests(unittest.TestCase):
    def test_actual_starter_runs_all_model_side_experiments_and_missing_answers_stay_missing(self):
        with tempfile.TemporaryDirectory(prefix='physalign-starter-study-') as d:
            root = Path(d)
            plan = prepare_study(SAMPLES, root / 'study', split='framework_development', bootstrap_resamples=0)
            self.assertEqual(len(plan['requests']), 54)
            self.assertEqual(plan['nearest_ids'], ['sample_04', 'sample_05', 'sample_06'])
            self.assertEqual(len(plan['readiness']['missing_original_answer_keys']), 3)
            run_study(root / 'study', SmokeAdapter(), root / 'run')
            report = score_study(root / 'study', root / 'run')
            self.assertEqual(report['service']['completed'], 54)
            self.assertIsNone(report['original_solving']['SolveAcc'])
            self.assertEqual(report['original_solving']['answer_eligible_mothers'], 0)
