"""Protocol integration tests: actual image transport, estimands and first outcomes.

Synthetic answers below are test fixtures, never benchmark model measurements.
All bundle mutations are confined to TemporaryDirectory copies.
"""

import base64
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from physalign.adapters import AdapterInfo, InfrastructureError, ModelResponse, SmokeAdapter
from physalign.cli import main
from physalign.dataset import MARKERS, PublicDataset, sections
from physalign.planning import prepare_plan, save_plan, load_plan
from physalign.reporting import score_run
from physalign.runner import run_evaluation
from physalign.storage import atomic_json, canonical, confined, digest, file_hash, read_json, run_lock


SAMPLES = Path(__file__).resolve().parents[1] / 'examples/synthetic'


def rehash(root):
    atomic_json(root / 'FILE_MANIFEST.json', {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'FILE_MANIFEST.json'
    })


def native_bundle(root):
    """Three T03 probes (2 + 1 per mother), two T02 probes, three Gold views."""
    source = PublicDataset(SAMPLES)
    asset = source.images(source.items['sample_04', 'raw'])[-1]
    (root / 'public/images').mkdir(parents=True)
    (root / 'public/images/original.png').write_bytes(asset.data)
    attachment = {'asset_id': 'img_0', 'path': 'images/original.png',
                  'sha256': asset.sha256, 'width': asset.width, 'height': asset.height}
    locator = {'kind': 'visual', 'image_id': 'img_0',
               'geometry': {'type': 'bbox', 'bbox_1000': [10, 20, 30, 40]}}
    evidence = {'anchors': {'R1': locator}, 'candidates': [
        {'alias': 'E1', 'locators': [locator]}, {'alias': 'E2', 'locators': [locator]}]}
    raw, gold, keys = [], [], []
    for iid, mother, task in [('p1', 'A', 'T03'), ('p2', 'A', 'T03'), ('p3', 'B', 'T03'),
                              ('p4', 'A', 'T02'), ('p5', 'C', 'T02')]:
        reading = task == 'T03'
        packet = [{'anchor_id': 'R1', 'text': '2 kg'}] if reading else None
        output = {'owner': '<one candidate ID>'}
        if reading:
            output['read'] = '<literal transcription>'
        bodies = ['A 2 kg body is shown.', 'What happens?', '[]', canonical(evidence),
                  'Original image IDs: img_0', f'Probe {iid}', canonical(output), 'None.']
        user = '\n'.join(f'[{name}]\n{body}' for name, body in zip(MARKERS, bodies))
        record = {'schema_version': 'physalign_public_qa_v1', 'instance_id': iid,
                  'logical_probe_id': 'logical-' + iid, 'task_id': task,
                  'interface': task + '-image', 'language': 'en', 'split': 'development',
                  'condition': 'raw', 'input': {'messages': {'system': 'Return JSON.', 'user': user},
                                               'attachments': [attachment]}}
        raw.append(record)
        if packet:
            g = deepcopy(record)
            g['condition'] = 'gold'
            g['input']['messages']['user'] = user.removesuffix('None.') + canonical(packet)
            gold.append(g)
        keys.append({'instance_id': iid, 'logical_probe_id': 'logical-' + iid, 'problem_id': mother,
                     'probe_type': task, 'split': 'development', 'review_status': 'model_approved',
                     'human_reviewed': False, 'cluster_id': mother,
                     'binding': {'kind': 'single', 'domains': [
                         {'field': 'owner', 'candidates': {'E1': 'object-a', 'E2': 'object-b'}}],
                         'gold': ['object-a']},
                     'readings': [{'field': 'read', 'anchor_id': 'R1', 'expected': '2 kg',
                                   'rule': {'kind': 'ocr', 'normalizer_id': 'ocr_label_v2_1'}}] if reading else [],
                     'permitted_gold_packet': packet})
    atomic_json(root / 'manifest.json', {'schema_version': 'physalign_eval_bundle_v1',
                                       'image_paths_relative_to': 'public',
                                       'formal_release_eligibility_checked': False})
    atomic_json(root / 'public/qa_raw.json', raw)
    atomic_json(root / 'public/qa_gold.json', gold)
    atomic_json(root / 'private/probes.json', keys)
    rehash(root)
    return root


class RecordingAdapter:
    info = AdapterInfo('fixture', 'synthetic-not-a-model', 'v1', '1',
                       '{"temperature":0,"max_output_tokens":100}', '{"resize":false}', False)

    def __init__(self, behavior=None):
        self.calls = []
        self.behavior = behavior

    def generate(self, request):
        self.calls.append(request)
        return self.behavior(request) if self.behavior else ModelResponse('{}', 'stop')


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='physalign-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = native_bundle(self.root / 'bundle')
        self.run = self.root / 'run'

    def plan(self, **kwargs):
        return prepare_plan(self.bundle, split='development', bootstrap_resamples=kwargs.pop('bootstrap_resamples', 0), **kwargs)

    def edit(self, name, change):
        value = read_json(self.bundle / name)
        change(value)
        atomic_json(self.bundle / name, value)
        rehash(self.bundle)

    def test_end_to_end_matches_hand_calculation_and_freezes_pairing(self):
        def answer(req):
            part = sections(req.user)
            iid = part['Local question'].split()[-1]
            is_gold = part['Local reading information'] != 'None.'
            c = iid != 'p2'
            b = is_gold or iid in {'p1', 'p4'}
            return ModelResponse(canonical({'read': '2 kg' if c else '3 kg', 'owner': 'E1' if b else 'E2'}))
        plan = self.plan(bootstrap_resamples=32)
        adapter = RecordingAdapter(answer)
        run_evaluation(plan, adapter, self.run)
        report = score_run(self.run)
        raw = report['raw_all']['metrics']
        self.assertEqual((raw['CAcc'], raw['JAcc'], raw['BAcc_L'], raw['BAcc']), (.75, .25, .25, .375))
        self.assertAlmostEqual(raw['BAcc_given_C'], 1 / 3)
        self.assertIsNone(raw['BAcc_gold'])  # No made-up Gold for the two T02 probes.
        pair = report['paired']['metrics']
        self.assertEqual((pair['BAcc'], pair['BAcc_gold'], pair['delta_BAcc_gold_raw']), (.25, 1., .75))
        self.assertEqual(report['pairing']['planned_paired_probes'], 3)
        self.assertEqual(report['service']['infrastructure_missing'], 0)
        self.assertEqual(report['intervals']['raw_all']['n_clusters'], 3)
        self.assertEqual(report['intervals']['paired']['n_clusters'], 2)
        self.assertEqual(report['uniform_candidate_baseline']['raw_all']['binding']['value'], .5)
        rows = [json.loads(line) for line in (self.run / 'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(r['C'] is None and r['J'] is None for r in rows if r['condition'] == 'gold'))
        # Byte-for-byte same images/settings, different fresh request objects.
        for iid in plan['paired_ids']:
            a, b = [r for r in adapter.calls if sections(r.user)['Local question'] == f'Probe {iid}']
            self.assertIsNot(a, b)
            self.assertEqual((a.system, a.images, a.settings_json), (b.system, b.images, b.settings_json))
            self.assertNotEqual(a.request_id, b.request_id)

    def test_inference_cannot_read_private_files_and_passes_real_images(self):
        plan = self.plan()
        original_open = Path.open
        def public_only(path, *args, **kwargs):
            if path.resolve().is_relative_to((self.bundle / 'private').resolve()):
                raise AssertionError('Private file read on the model execution path')
            return original_open(path, *args, **kwargs)
        adapter = RecordingAdapter()
        with patch.object(Path, 'open', public_only):
            run_evaluation(plan, adapter, self.run)
        self.assertEqual(len(adapter.calls), 8)
        for req in adapter.calls:
            self.assertEqual(set(vars(req)), {'request_id', 'system', 'user', 'images', 'settings_json'})
            self.assertTrue(all(digest(i.data) == i.sha256 for i in req.images))
            for url, asset in zip(req.image_data_urls(), req.images):
                self.assertEqual(base64.b64decode(url.split(',', 1)[1]), asset.data)
            self.assertNotIn('object-a', req.user)

    def test_custom_weights_are_frozen_before_predictions(self):
        plan = self.plan(type_weights={'full': {'binding': {'T02': .25, 'T03': .75}}})
        self.assertEqual(plan['weights']['full']['binding'], {'T02': .25, 'T03': .75})
        self.assertEqual(plan['weights']['paired']['binding'], {'T03': 1.0})
        save_plan(plan, self.root / 'plan.json')
        self.assertEqual(load_plan(self.root / 'plan.json'), plan)
        with self.assertRaises(FileExistsError):
            save_plan(plan, self.root / 'plan.json')
        with self.assertRaises(ValueError):
            self.plan(type_weights={'full': {'binding': {'T02': 1}}})

    def test_raw_only_schedule_does_not_infer_gold_failures(self):
        plan = self.plan(conditions=('raw',))
        self.assertEqual(len(plan['requests']), 5)
        run_evaluation(plan, SmokeAdapter(), self.run)
        report = score_run(self.run)
        self.assertIsNone(report['paired'])
        self.assertEqual(report['service']['infrastructure_missing'], 0)

    def test_malformed_refusal_empty_and_truncated_outputs_are_never_retried(self):
        for index, text in enumerate(('', 'I cannot answer.', '{"owner":', 'not json', '{"owner":"E2"}')):
            with self.subTest(text=text):
                adapter = RecordingAdapter(lambda req: ModelResponse(text, 'length'))
                folder = self.root / f'case-{index}'
                run_evaluation(self.plan(), adapter, folder)
                self.assertEqual(len(adapter.calls), 8)
                report = score_run(folder)
                self.assertEqual(report['service']['infrastructure_missing'], 0)
                self.assertEqual(report['raw_all']['metrics']['BAcc'], 0)
                records = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(all(r['attempt_count'] == 1 and r['response']['text'] == text for r in records))

    def test_only_certified_infrastructure_is_retried_up_to_twice(self):
        counts = {}
        def behavior(req):
            counts[req.request_id] = counts.get(req.request_id, 0) + 1
            if counts[req.request_id] <= 2:
                raise InfrastructureError('timeout_without_response')
            return ModelResponse('{}')
        adapter = RecordingAdapter(behavior)
        run_evaluation(self.plan(), adapter, self.run)
        self.assertEqual(len(adapter.calls), 24)
        self.assertTrue(all(n == 3 for n in counts.values()))
        self.assertEqual(score_run(self.run)['service']['completed_outputs'], 8)
        for i in range(0, 24, 3):
            self.assertIs(adapter.calls[i], adapter.calls[i + 1])

    def test_missing_gold_retains_paired_denominator_and_bounds(self):
        def behavior(req):
            parts = sections(req.user)
            if parts['Local question'] == 'Probe p2' and parts['Local reading information'] != 'None.':
                raise InfrastructureError('service_unavailable')
            return ModelResponse('{"read":"2 kg","owner":"E1"}')
        adapter = RecordingAdapter(behavior)
        run_evaluation(self.plan(), adapter, self.run)
        report = score_run(self.run)
        self.assertEqual(len(adapter.calls), 10)
        self.assertEqual(report['pairing']['planned_paired_probes'], 3)
        self.assertEqual(report['service']['infrastructure_missing'], 1)
        summary = report['paired']['summaries']['BAcc_gold']
        self.assertIsNone(summary['value'])
        self.assertEqual((summary['lower'], summary['upper'], summary['weighted_service_coverage']), (.75, 1., .75))
        self.assertIsNone(report['paired']['metrics']['delta_BAcc_gold_raw'])
        self.assertEqual(report['paired']['delta_BAcc_bounds'], {'lower': -.25, 'upper': 0.})

    def test_nonretryable_infrastructure_stops_after_first_attempt(self):
        def behavior(req):
            raise InfrastructureError('unsupported_input_size', retryable=False)
        adapter = RecordingAdapter(behavior)
        run_evaluation(self.plan(), adapter, self.run)
        self.assertEqual(len(adapter.calls), 8)
        report = score_run(self.run)
        self.assertEqual(report['service']['infrastructure_missing'], 8)
        self.assertIsNone(report['raw_all']['metrics']['BAcc'])
        self.assertEqual(report['raw_all']['summaries']['BAcc']['lower'], 0)
        self.assertEqual(report['raw_all']['summaries']['BAcc']['upper'], 1)

    def test_resume_keeps_first_outputs_without_new_calls(self):
        adapter, plan = RecordingAdapter(), self.plan()
        run_evaluation(plan, adapter, self.run)
        original = (self.run / 'predictions.jsonl').read_bytes()
        run_evaluation(plan, adapter, self.run, resume=True)
        self.assertEqual(len(adapter.calls), 8)
        self.assertEqual((self.run / 'predictions.jsonl').read_bytes(), original)
        adapter.info = replace(adapter.info, settings_json='{"temperature":1}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_evaluation(plan, adapter, self.run, resume=True)
        self.assertEqual(len(adapter.calls), 8)

    def test_interrupted_unknown_response_is_missing_not_rerun(self):
        def crash(req):
            raise RuntimeError('simulated crash after submission')
        adapter, plan = RecordingAdapter(crash), self.plan()
        with self.assertRaisesRegex(RuntimeError, 'simulated crash'):
            run_evaluation(plan, adapter, self.run)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            score_run(self.run)
        diagnostic = score_run(self.run, allow_incomplete=True)
        self.assertEqual(len(diagnostic['service']['pending_requests']), 8)
        adapter.behavior = None
        run_evaluation(plan, adapter, self.run, resume=True)
        self.assertEqual(len(adapter.calls), 8)  # One interrupted call + seven fresh calls.
        report = score_run(self.run)
        self.assertEqual(report['service']['infrastructure_missing'], 1)
        records = [json.loads(s) for s in (self.run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(records[0]['reason'], 'interrupted_request_unknown_response')

    def test_finished_response_survives_crash_before_terminal_record(self):
        adapter, plan = RecordingAdapter(), self.plan()
        def crash_at_terminal(path, value):
            if path.parent.name == 'results':
                raise RuntimeError('simulated terminal write crash')
            return atomic_json(path, value)
        with patch('physalign.runner.atomic_json', crash_at_terminal):
            with self.assertRaisesRegex(RuntimeError, 'terminal write crash'):
                run_evaluation(plan, adapter, self.run)
        run_evaluation(plan, adapter, self.run, resume=True)
        self.assertEqual(len(adapter.calls), 8)
        self.assertEqual(score_run(self.run)['service']['infrastructure_missing'], 0)

    def test_attempt_and_export_tampering_are_rejected(self):
        adapter, plan = RecordingAdapter(), self.plan()
        run_evaluation(plan, adapter, self.run)
        path = next((self.run / 'attempts').glob('*.finished.json'))
        original = path.read_bytes()
        value = read_json(path)
        value['response']['text'] = '{"owner":"E1"}'
        atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, 'Corrupt run record'):
            score_run(self.run)
        path.write_bytes(original)
        with (self.run / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'Prediction export differs'):
            score_run(self.run)

    def test_code_change_forbids_mixed_implementation_resume(self):
        adapter, plan = RecordingAdapter(), self.plan()
        run_evaluation(plan, adapter, self.run)
        with patch('physalign.runner.code_hashes', return_value={'runner.py': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'changed'):
                run_evaluation(plan, adapter, self.run, resume=True)
        self.assertEqual(len(adapter.calls), 8)

    def test_lock_and_source_write_guards(self):
        with run_lock(self.run):
            with self.assertRaisesRegex(RuntimeError, 'locked'):
                run_evaluation(self.plan(), RecordingAdapter(), self.run)
        with self.assertRaisesRegex(ValueError, 'outside'):
            save_plan(self.plan(), self.bundle / 'plan.json')
        with self.assertRaisesRegex(ValueError, 'outside'):
            run_evaluation(self.plan(), RecordingAdapter(), self.bundle / 'run')

    def test_modified_private_source_cannot_be_scored(self):
        plan = self.plan()
        run_evaluation(plan, SmokeAdapter(), self.run)
        with (self.bundle / 'private/probes.json').open('a', encoding='utf-8') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            score_run(self.run)

    def test_gold_prompt_or_image_order_change_is_rejected(self):
        self.edit('public/qa_gold.json', lambda rows: rows[0]['input']['messages'].update(system='Different instructions'))
        with self.assertRaisesRegex(ValueError, 'instructions differ'):
            self.plan()

    def test_image_hash_and_actual_dimensions_are_checked(self):
        image_path = self.bundle / 'public/images/original.png'
        original = image_path.read_bytes()
        image_path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.plan()
        image_path.write_bytes(original)
        for name in ('public/qa_raw.json', 'public/qa_gold.json'):
            self.edit(name, lambda rows: [r['input']['attachments'][0].update(width=1) for r in rows])
        with self.assertRaisesRegex(ValueError, 'Actual image dimensions'):
            self.plan()

    def test_mother_and_cluster_split_integrity(self):
        for name in ('public/qa_raw.json', 'public/qa_gold.json', 'private/probes.json'):
            self.edit(name, lambda rows: rows[1].update(split='test'))
        with self.assertRaisesRegex(ValueError, 'mother crosses'):
            self.plan()

    def test_one_mother_cannot_have_multiple_cluster_ids(self):
        self.edit('private/probes.json', lambda rows: rows[1].update(cluster_id='another'))
        with self.assertRaisesRegex(ValueError, 'multiple bootstrap clusters'):
            self.plan()

    def test_formal_test_requires_explicit_eligibility_and_excludes_reservations(self):
        for name in ('public/qa_raw.json', 'public/qa_gold.json', 'private/probes.json'):
            self.edit(name, lambda rows: [r.update(split='test') for r in rows])
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            prepare_plan(self.bundle, split='test', bootstrap_resamples=0)
        self.edit('manifest.json', lambda m: m.update(formal_release_eligibility_checked=True))
        reserved = self.root / 'reserved.json'
        atomic_json(reserved, {'problem_ids': ['A']})
        with self.assertRaisesRegex(ValueError, 'Development-reserved'):
            prepare_plan(self.bundle, split='test', reservation_files=[reserved], bootstrap_resamples=0)
        self.edit('private/probes.json', lambda rows: [r.update(cluster_id='duplicate') for r in rows])
        with self.assertRaisesRegex(ValueError, 'cluster contains'):
            prepare_plan(self.bundle, split='test', problem_ids=['B'], reservation_files=[reserved], bootstrap_resamples=0)

    def test_frozen_duplicate_clusters_determine_bootstrap_unit(self):
        self.edit('private/probes.json', lambda rows: [r.update(cluster_id='shared') for r in rows])
        plan = self.plan(bootstrap_resamples=8)
        run_evaluation(plan, SmokeAdapter(), self.run)
        report = score_run(self.run)
        self.assertEqual(report['intervals']['raw_all']['n_clusters'], 1)
        self.assertEqual(report['intervals']['paired']['n_clusters'], 1)

    def test_invalid_private_map_or_output_contract_is_rejected(self):
        self.edit('private/probes.json', lambda rows: rows[0]['binding']['domains'][0]['candidates'].update(E3='third'))
        with self.assertRaisesRegex(ValueError, 'public candidates'):
            self.plan()

    def test_native_text_anchor_does_not_become_visual_reading(self):
        def edit_text(rows):
            for row in rows:
                if row['instance_id'] != 'p1':
                    continue
                parts = sections(row['input']['messages']['user'])
                evidence = json.loads(parts['Evidence locations and candidates'])
                evidence['anchors']['R1'] = {'kind': 'text', 'section': 'statement', 'start': 2, 'end': 6, 'text': '2 kg'}
                parts['Evidence locations and candidates'] = canonical(evidence)
                row['input']['messages']['user'] = '\n'.join(f'[{m}]\n{parts[m]}' for m in MARKERS)
                row['interface'] = 'T03-text'
        for name in ('public/qa_raw.json', 'public/qa_gold.json'):
            self.edit(name, edit_text)
        plan = self.plan()
        self.assertEqual(plan['probes'][0]['probe_type'], 'T03')
        self.assertEqual(plan['weights']['full']['joint'], {'T03': 1.0})

    def test_native_relation_preserves_declared_endpoint_order(self):
        def change_key(rows):
            row = rows[3]
            row['binding'] = {'kind': 'relation', 'domains': [
                {'field': 'z_start', 'candidates': {'E1': 'a', 'E2': 'b'}},
                {'field': 'm_relation', 'candidates': {'K1': 'connects', 'K2': 'other'}},
                {'field': 'a_end', 'candidates': {'E1': 'a', 'E2': 'b'}}], 'gold': ['a', 'connects', 'b']}
        self.edit('private/probes.json', change_key)
        def change_public(rows):
            row = rows[3]
            parts = sections(row['input']['messages']['user'])
            parts['Output format'] = canonical({'z_start': '<one candidate ID>', 'm_relation': '<one candidate ID>', 'a_end': '<one candidate ID>'})
            evidence = json.loads(parts['Evidence locations and candidates'])
            evidence['candidates'].extend({'alias': alias, 'locators': [evidence['anchors']['R1']]} for alias in ('K1', 'K2'))
            parts['Evidence locations and candidates'] = canonical(evidence)
            row['input']['messages']['user'] = '\n'.join(f'[{m}]\n{parts[m]}' for m in MARKERS)
        self.edit('public/qa_raw.json', change_public)
        plan = self.plan(problem_ids=['A'])
        adapter = RecordingAdapter(lambda req: ModelResponse('{"z_start":"E1","m_relation":"K1","a_end":"E2"}'))
        run_evaluation(plan, adapter, self.run)
        report = score_run(self.run)
        self.assertEqual(report['raw_all']['by_type']['T02']['BAcc'], 1)
        self.assertEqual(report['uniform_candidate_baseline']['raw_all']['status'], 'not_defined_for_mixed_set_or_relation_pool')

    def test_native_set_and_exact_quantity_contract_reach_existing_scorer(self):
        def change_key(rows):
            rows[0]['readings'][0]['rule'] = {'kind': 'quantity', 'unit_scales': {'kg': '1', 'g': '0.001'}, 'absolute_tolerance': '0'}
            rows[3]['binding']['kind'] = 'set'
            rows[3]['binding']['gold'] = ['object-a', 'object-b']
        self.edit('private/probes.json', change_key)
        def change_public(rows):
            row = rows[3]
            parts = sections(row['input']['messages']['user'])
            parts['Output format'] = '{"owner":["<candidate ID>"]}'
            row['input']['messages']['user'] = '\n'.join(f'[{m}]\n{parts[m]}' for m in MARKERS)
        self.edit('public/qa_raw.json', change_public)
        def behavior(req):
            iid = sections(req.user)['Local question']
            return ModelResponse('{"owner":["E2","E1"]}' if iid == 'Probe p4' else '{"read":"2000 g","owner":"E1"}')
        run_evaluation(self.plan(), RecordingAdapter(behavior), self.run)
        score_run(self.run)
        rows = [json.loads(line) for line in (self.run / 'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        scored = {(r['instance_id'], r['condition']): r for r in rows}
        self.assertEqual(scored['p1', 'raw']['C'], 1)  # Explicit quantity conversion.
        self.assertEqual(scored['p2', 'raw']['C'], 0)  # Literal OCR rule remains literal.
        self.assertEqual((scored['p4', 'raw']['B'], scored['p4', 'raw']['set_f1']), (1, 1))

    def test_empty_gold_packet_is_a_real_pair_but_not_nonempty_packet_support(self):
        raw = read_json(self.bundle / 'public/qa_raw.json')
        extra = deepcopy(raw[3])
        extra['condition'] = 'gold'
        self.edit('public/qa_gold.json', lambda rows: rows.append(extra))
        self.edit('private/probes.json', lambda rows: rows[3].update(permitted_gold_packet=[]))
        plan = self.plan()
        self.assertIn('p4', plan['paired_ids'])
        self.assertFalse(next(p for p in plan['probes'] if p['instance_id'] == 'p4')['packet_nonempty'])
        run_evaluation(plan, SmokeAdapter(), self.run)
        report = score_run(self.run)
        self.assertEqual(report['pairing']['planned_paired_probes'], 4)
        self.assertEqual(report['paired']['nonempty_packets']['weighted_packet_coverage'], .5)

    def test_gold_cannot_change_question_or_add_binding_to_packet(self):
        def add_binding(rows):
            row = rows[0]
            parts = sections(row['input']['messages']['user'])
            parts['Local reading information'] = '[{"anchor_id":"R1","text":"2 kg","owner":"E1"}]'
            row['input']['messages']['user'] = '\n'.join(f'[{m}]\n{parts[m]}' for m in MARKERS)
        self.edit('public/qa_gold.json', add_binding)
        with self.assertRaisesRegex(ValueError, 'only anchored transcriptions'):
            self.plan()

    def test_changed_public_input_cannot_reuse_frozen_plan(self):
        plan = self.plan()
        self.edit('public/qa_raw.json', lambda rows: rows[3]['input']['messages'].update(system='Edited after preparation'))
        adapter = RecordingAdapter()
        with self.assertRaisesRegex(ValueError, 'changed after preparation'):
            run_evaluation(plan, adapter, self.run)
        self.assertFalse(adapter.calls)

    def test_returned_model_versions_are_preserved_and_mixed_versions_flagged(self):
        def behavior(req):
            version = 'v2' if sections(req.user)['Local reading information'] != 'None.' else 'v1'
            return ModelResponse('{}', 'stop', '{"input_tokens":10,"output_tokens":2}', version, 'provider-123')
        run_evaluation(self.plan(), RecordingAdapter(behavior), self.run)
        report = score_run(self.run)
        self.assertTrue(report['provenance']['multiple_returned_versions'])
        self.assertEqual(report['provenance']['returned_model_versions'], ['v1', 'v2'])
        self.assertEqual(report['provenance']['response_version_unavailable'], 0)

    def test_overflow_in_irrelevant_output_field_cannot_crash_reporting(self):
        text = '{"read":"2 kg","owner":"E1","irrelevant":1e999}'
        adapter = RecordingAdapter(lambda req: ModelResponse(text))
        run_evaluation(self.plan(), adapter, self.run)
        report = score_run(self.run)
        self.assertEqual(report['raw_all']['metrics']['BAcc'], 1)
        self.assertEqual(report['service']['completed_outputs'], 8)
        self.assertEqual(len(adapter.calls), 8)
        rows = [json.loads(line) for line in (self.run / 'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(r['parsed_output'] is None and r['object_valid'] and
                            r['parsed_output_status'] == 'not_serializable_original_response_preserved' for r in rows))
        predictions = [json.loads(line) for line in (self.run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(r['response']['text'] == text for r in predictions))

    def test_reserved_cluster_can_be_excluded_without_reserved_mother_in_bundle(self):
        for name in ('public/qa_raw.json', 'public/qa_gold.json', 'private/probes.json'):
            self.edit(name, lambda rows: [r.update(split='test') for r in rows])
        self.edit('manifest.json', lambda row: row.update(formal_release_eligibility_checked=True))
        reservation = self.root / 'reservation.json'
        atomic_json(reservation, {'problem_ids': ['absent-mother'], 'cluster_ids': ['A']})
        with self.assertRaisesRegex(ValueError, 'cluster contains'):
            prepare_plan(self.bundle, split='test', bootstrap_resamples=0, reservation_files=[reservation])

    def test_cli_prepare_run_score_and_resume(self):
        import contextlib
        import io
        plan_path = self.root / 'cli-plan.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['prepare', '--dataset', str(self.bundle), '--plan', str(plan_path),
                                   '--split', 'development', '--bootstrap-resamples', '0']), 0)
            args = ['run', '--plan', str(plan_path), '--output', str(self.run), '--adapter', 'smoke']
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args + ['--resume']), 0)
            self.assertEqual(main(['score', '--run', str(self.run)]), 0)
        self.assertFalse(read_json(self.run / 'report.json')['scientific_run'])

    def test_path_escape_and_ambiguous_json_are_rejected(self):
        for relative in ('../private/key.json', 'C:/secret.json', '/etc/passwd', 'public/../private/a', 'a\\b'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                confined(self.bundle, relative)
        from physalign.storage import loads
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loads(text)


class SyntheticStarterTests(unittest.TestCase):
    def test_synthetic_export_nine_requests_roundtrip_and_all_hashes_unchanged(self):
        source = PublicDataset(SAMPLES)
        before = {name: file_hash(SAMPLES / name) for name in source.inventory}
        plan = prepare_plan(SAMPLES, split='framework_development', bootstrap_resamples=16)
        self.assertEqual(len(plan['probes']), 6)
        self.assertEqual(plan['paired_ids'], ['sample_04', 'sample_05', 'sample_06'])
        self.assertEqual({p['problem_id'] for p in plan['probes']}, set(plan['reserved_problem_ids']))
        with tempfile.TemporaryDirectory(prefix='physalign-starter-test-') as temp:
            folder = Path(temp) / 'run'
            adapter = RecordingAdapter()
            run_evaluation(plan, adapter, folder)
            self.assertEqual(len(adapter.calls), 9)
            for row, request in zip(plan['requests'], adapter.calls):
                item = source.items[row['instance_id'], row['condition']]
                public = item.record['input']
                self.assertEqual(request.system, public['messages']['system'])
                self.assertEqual(request.user, public['messages']['user'])
                self.assertEqual([i.asset_id for i in request.images], [a['asset_id'] for a in public['attachments']])
                self.assertEqual([i.sha256 for i in request.images], [a['sha256'] for a in public['attachments']])
            report = score_run(folder)
            self.assertEqual(report['service']['planned_requests'], 9)
            self.assertEqual(report['raw_all']['metrics']['BAcc'], 0)
            self.assertIsNone(report['raw_all']['metrics']['BAcc_gold'])
            self.assertEqual(report['paired']['metrics']['BAcc_gold'], 0)
            self.assertEqual(report['provenance']['human_reviewed_probes'], 0)
        self.assertEqual(before, {name: file_hash(SAMPLES / name) for name in source.inventory})
