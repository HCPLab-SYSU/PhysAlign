import copy
import contextlib
import io
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import workspace
from adapter import SourceAdapter, verify_adapter_snapshot
from io_utils import read_json, write_json
from reference_checks import digest
from reference_pipeline import prepare_probe, freeze_probe
from native_compiler import NativeCompiler, singleton_basis
from native_projection import SPAN_RULE
from native_diagnostics import read_availability, nearest_region, chance_summary
from native_run import dedup_members, development_ids
from convert import main


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation, self.dataset, self.pid = workspace(self.root)

    def mutate(self, edit):
        path = self.annotation / 'passes/pass4' / (self.pid + '.json')
        d = read_json(path)
        edit(d)
        write_json(path, d)
        review = self.annotation / 'reviews/state.json'
        state = read_json(review)
        state['problems'][self.pid]['stages']['pass4']['document_sha256'] = digest(d)
        write_json(review, state)

    def compiler(self, approval=None):
        adapter = SourceAdapter(self.annotation, self.dataset)
        adapted = adapter.materialize(adapter.load(self.pid), self.root / 'materialized',
                                      native=True, span_approval=approval)
        return NativeCompiler(adapted)

    def test_native_real_source_projection_fixture(self):
        c = self.compiler()
        verify_adapter_snapshot(c.env)
        for task in ('T02', 'T03'):
            proposals = c.proposals(task)
            self.assertEqual(len(proposals), 1)
            d, env, p = c.compile(task, proposals[0])
            self.assertEqual(prepare_probe(d, env, verify_reviews=False), p)
            self.assertEqual(d['key']['packet_plan']['state'], 'pending_review')
            with self.assertRaisesRegex(ValueError, 'NATIVE_DRAFT_ONLY'):
                freeze_probe(d, env, ['not_approved'])

    def test_plural_one_edge_never_single(self):
        # Source mention need not resolve for the finite grammar regression.
        self.assertEqual(singleton_basis('two negative charges', [], {})['code'], 'EXPLICIT_PLURAL_REFERENCE')
        def change(d):
            # Existing exact quote refers to more than one object but ONE edge.
            d['text_mentions'][0]['quote'] = 'Block A and block B'
        self.mutate(change)
        c = self.compiler()
        self.assertEqual(c.proposals('T02'), [])
        self.assertIn('EXPLICIT_PLURAL_REFERENCE', c.groups[0]['blocking_reasons'])
        self.assertEqual(len(c.proposals('T03')), 1)

    def test_unknown_singular_is_not_assumed(self):
        for quote in ('it', 'the object', 'negative charge', 'mass', 'A and B'):
            self.assertEqual(singleton_basis(quote, [], {})['state'], 'blocked')
        for quote in ('Block A', 'point P_1', '物块甲', '电阻R'):
            self.assertEqual(singleton_basis(quote, [], {})['state'], 'supported')

    def test_exact_two_negative_charges_one_source_edge_end_to_end(self):
        manifest = self.annotation / 'blind/manifest.jsonl'
        row = read_json(manifest)
        row['segments']['stem'] = row['segments']['stem'].replace('Block A', 'two negative charges')
        row['raw_question'] = row['raw_question'].replace('Block A', 'two negative charges')
        write_json(manifest, row)
        p3path = self.annotation / 'passes/pass3' / (self.pid + '.json')
        p3 = read_json(p3path)
        p3['text_mentions'][0]['quote'] = 'two negative charges'
        write_json(p3path, p3)
        reviewpath = self.annotation / 'reviews/state.json'
        state = read_json(reviewpath)
        state['problems'][self.pid]['stages']['pass3']['document_sha256'] = digest(p3)
        write_json(reviewpath, state)
        self.mutate(lambda d: d['text_mentions'][0].update(quote='two negative charges'))
        c = self.compiler()
        self.assertEqual(sum(b['type'] == 'refers_to' for b in c.source['bindings']), 1)
        self.assertEqual(c.proposals('T02'), [])
        self.assertIn('EXPLICIT_PLURAL_REFERENCE', c.groups[0]['blocking_reasons'])

    def test_image_read_comes_from_glyph_not_quantity_formula(self):
        self.mutate(lambda d: d['quantities'][0].update(value_latex='(8-2)/3', unit_latex='kg'))
        c = self.compiler()
        draft, _, _ = c.compile('T03', c.proposals('T03')[0])
        self.assertEqual(draft['key']['read_targets'][0]['expected'], 'm')

    def test_atomic_symbol_requires_source_proof(self):
        self.mutate(lambda d: d['physical_nodes'][0].update(symbol='A'))
        c = self.compiler()
        rows = c.proposals('T02')[0]['rows']
        self.assertEqual(singleton_basis('A', rows, c.rows)['state'], 'supported')
        self.assertEqual(singleton_basis('B', rows, c.rows)['state'], 'blocked')

    def test_t04_single_subject_policy_disabled_and_unknown_predicate_untouched(self):
        self.mutate(lambda d: (d['constraints'][0].update(subject_ids=['p001']),
                               d['relations'][0].update(predicate='completely_unknown_directional_word')))
        c = self.compiler()
        self.assertEqual(c.proposals('T04'), [])
        self.assertEqual(c.proposals('T05'), [])
        self.assertEqual(c.blocked[0]['code'], 'policy_disabled')
        self.assertEqual(len(c.proposals('T02')), 1)
        self.assertEqual(len(c.proposals('T03')), 1)
        self.assertEqual(c.source['relations'][0]['predicate'], 'completely_unknown_directional_word')

    def test_same_public_anchor_two_owners_blocks_and_no_fallback(self):
        self.mutate(lambda d: d['quantities'].append({**copy.deepcopy(d['quantities'][0]), 'id': 'q002', 'owner_id': 'p002'}))
        c = self.compiler()
        self.assertEqual(c.proposals('T03'), [])
        self.assertIn('QUANTITY_IDENTITY_OR_OWNER_CONFLICT', c.groups[0]['blocking_reasons'])
        self.assertEqual(c.groups[0]['source_scope'], ['q001', 'q002'])

    def test_bad_quantity_shadows_second_owner_but_independent_group_survives(self):
        def change(d):
            d['visual_nodes'].append({**copy.deepcopy(d['visual_nodes'][2]), 'id': 'v004', 'text': 'M',
                                      'bbox_1000': [600, 200, 650, 280]})
            d['quantities'].extend([
                {**copy.deepcopy(d['quantities'][0]), 'id': 'q002', 'owner_id': ''},
                {**copy.deepcopy(d['quantities'][0]), 'id': 'q003', 'owner_id': 'p002',
                 'visual_anchor_ids': ['v004'], 'text_mention_ids': [], 'provenance': ['IMAGE']}])
        self.mutate(change)
        p1path = self.annotation / 'passes/pass1' / (self.pid + '.json')
        p1 = read_json(p1path)
        p1['visual_nodes'].append(read_json(self.annotation / 'passes/pass4' / (self.pid + '.json'))['visual_nodes'][-1])
        write_json(p1path, p1)
        state = read_json(self.annotation / 'reviews/state.json')
        state['problems'][self.pid]['stages']['pass1']['document_sha256'] = digest(p1)
        write_json(self.annotation / 'reviews/state.json', state)
        c = self.compiler()
        proposals = c.proposals('T03')
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]['rows'][0]['ref']['record_id'], 'q003')
        self.assertTrue(any(g['state'] == 'blocked' and 'q002' in g['source_scope'] and 'q001' in g['source_scope']
                            for g in c.groups))
        self.assertTrue(any(r['record_id'] == 'q002' for r in c.shadow['records']))

    def test_unbounded_bad_quantity_is_still_wide_block(self):
        self.mutate(lambda d: d['quantities'].append({**copy.deepcopy(d['quantities'][0]), 'id': 'q002',
            'visual_anchor_ids': ['v001'], 'text_mention_ids': [], 'owner_id': '', 'provenance': ['IMAGE']}))
        c = self.compiler()
        self.assertEqual(c.proposals('T03'), [])
        self.assertEqual(len(c.proposals('T02')), 1)

    def test_bad_occurrence_can_add_second_reference_target(self):
        def change(d):
            d['text_mentions'][-1]['id'] = 'm005'
            d['query_target']['mention_id'] = 'm005'
            d['text_mentions'].insert(-1, {**d['text_mentions'][0], 'id': 'm004', 'occurrence': 99})
            d['bindings'].append({**d['bindings'][-1], 'id': 'b004', 'from_id': 'm004', 'to_id': 'p002'})
        self.mutate(change)
        c = self.compiler()
        self.assertEqual(c.proposals('T02'), [])
        self.assertIn('REFERENCE_POSSIBLE_SECOND_TARGET_UNRESOLVED', c.groups[0]['blocking_reasons'])
        self.assertIn('REFERENCE_TARGET_NOT_SINGLE', c.groups[0]['blocking_reasons'])

    def test_text_fallback_retains_owner_and_has_no_C_J_or_packet(self):
        self.mutate(lambda d: d['quantities'][0].update(visual_anchor_ids=[], provenance=['TEXT']))
        c = self.compiler()
        p = c.proposals('T03')[0]
        self.assertEqual(p['native']['interface'], 'T03-text')
        self.assertEqual(p['rows'][0]['target'], 'p001')
        d, env, preview = c.compile('T03', p)
        self.assertFalse(d['key']['joint_eligible'])
        self.assertEqual(d['key']['read_targets'], [])
        self.assertEqual(d['key']['packet_plan']['observation_ids'], [])
        self.assertEqual(d['key']['response_contract']['read_shape'], 'none')

    def test_sidecar_default_is_proposed_and_source_unchanged(self):
        self.mutate(lambda d: (d['text_mentions'][2].update(quote='block B', occurrence=99, role='entity'),
                               d['bindings'][-1].update(from_id='m003', to_id='p002')))
        path = self.annotation / 'passes/pass4' / (self.pid + '.json')
        original = path.read_bytes()
        c = self.compiler()
        self.assertEqual(c.proposals('T02'), [])
        sidecar = read_json(self.root / 'materialized/native_span_sidecar.json')
        self.assertEqual(sidecar['entries'][0]['state'], 'proposed')
        self.assertEqual(path.read_bytes(), original)
        verify_adapter_snapshot(c.env)

    def test_approved_unique_sidecar_applies_offsets_and_hash_invalidation(self):
        self.mutate(lambda d: (d['text_mentions'][2].update(quote='block B', occurrence=99, role='entity'),
                               d['bindings'][-1].update(from_id='m003', to_id='p002')))
        approval = {'policy_id': SPAN_RULE['policy_id'], 'rule_hash': digest(SPAN_RULE),
                    'status': 'approved', 'reviewer': 'SYNTHETIC_TEST_ONLY', 'evidence_ref': 'SYNTHETIC_EXPLICIT_RULE_APPROVAL'}
        c = self.compiler(approval)
        d, env, _ = c.compile('T02', c.proposals('T02')[0])
        loc = d['view']['anchors']['R1']
        self.assertEqual((loc['start'], loc['end']), (12, 19))
        self.assertTrue(d['key']['anchor_map']['R1']['repair_refs'])
        verify_adapter_snapshot(env)
        sidecar = self.root / 'materialized/native_span_sidecar.json'
        obj = read_json(sidecar)
        obj['entries'][0]['new_occurrence'] = 2
        write_json(sidecar, obj)
        with self.assertRaisesRegex(ValueError, 'SIDECAR_BYTES_CHANGED'):
            verify_adapter_snapshot(env)

    def test_source_change_invalidates_native_projection(self):
        c = self.compiler()
        path = self.root / 'materialized/sources/pass4.json'
        obj = read_json(path)
        obj['text_mentions'][0]['occurrence'] = 2
        write_json(path, obj)
        with self.assertRaisesRegex(ValueError, 'SNAPSHOT_BYTES_CHANGED'):
            verify_adapter_snapshot(c.env)

    def test_proposal_cannot_drop_one_of_multiple_same_target_rows(self):
        self.mutate(lambda d: d['bindings'].append({**d['bindings'][-1], 'id': 'b004'}))
        c = self.compiler()
        p = c.proposals('T02')[0]
        self.assertEqual(len(p['rows']), 2)
        p['rows'] = p['rows'][:1]
        with self.assertRaisesRegex(ValueError, 'NOT_COMPLETE'):
            c.compile('T02', p)

    def test_cli_native_round_trip_is_draft_and_dev_only(self):
        output = self.root / 'native'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['native-draft', '--workspace', str(self.annotation), '--output', str(output)])
            main(['native-validate', '--path', str(output)])
        report = read_json(output / 'report.json')
        self.assertEqual(report['B_candidate']['probes'], 2)
        self.assertEqual(report['P_approved']['probes'], 0)
        self.assertEqual(report['L_candidate']['probes'], 1)
        self.assertTrue((output / 'review.html').is_file())
        self.assertFalse((output / 'release.json').exists())


class DiagnosticTests(unittest.TestCase):
    def test_options_and_alt_are_real_copy_channels(self):
        for raw in ({'messages': {'user': 'Stem: which value? Options: A. 2 kg B. 3 kg'}},
                    {'messages': {'user': 'Read R1'}, 'attachments': [{'path': 'x.png', 'alt': 'Mass = 2kg'}]}):
            m = read_availability(raw, '2 kg')
            self.assertEqual(m['public_text_availability'], 'verbatim_available')
            fields = dict(__import__('native_diagnostics').public_text_fields(raw))
            for hit in m['full_hits']:
                self.assertEqual(fields[hit['field']][hit['start']:hit['end']], hit['text'])

    def test_unicode_offsets_are_characters_not_bytes(self):
        raw = {'messages': {'user': '中文😀之后 2kg'}}
        hit = read_availability(raw, '2 kg')['full_hits'][0]
        self.assertEqual(hit['start'], 6)
        self.assertEqual(hit['end'], 9)
        raw = {'messages': {'user': '阻值 2Ω'}}
        hit = read_availability(raw, '2 Ω')['full_hits'][0]
        self.assertEqual((hit['start'], hit['end'], hit['text']), (3, 5, '2Ω'))

    def test_not_found_is_not_vision_certificate(self):
        m = read_availability({'messages': {'user': 'Choose an object.'}}, 'z_9')
        self.assertEqual(m['public_text_availability'], 'not_found')
        self.assertIsNone(m['vision_required'])

    def test_normalizer_does_not_compute_or_reverse_operations(self):
        for expected, text in [('(8-2)/3', '8-(2/3)'), ('2 cm', '0.02 m'), ('m', 'M'), ('2', '22')]:
            result = read_availability({'messages': {'user': text}}, expected)
            self.assertNotEqual(result['public_text_availability'], 'verbatim_available')
        self.assertEqual(read_availability({'messages': {'user': 'anything'}}, r'\frac{1}{2}')['public_text_availability'], 'unknown')
        self.assertEqual(read_availability({'messages': {'user': r'$2\,\mathrm{kg}$'}}, '2kg')['public_text_availability'], 'verbatim_available')

    def test_private_packet_or_image_paths_not_scanned(self):
        result = read_availability({'messages': {'user': 'question'}, 'private': {'packet': '2 kg'},
                                     'attachments': [{'path': '2 kg.png'}]}, '2 kg')
        self.assertEqual(result['public_text_availability'], 'not_found')

    @staticmethod
    def view():
        def loc(box):
            return {'kind': 'visual', 'image_id': 'img', 'geometry': {'type': 'bbox', 'bbox_1000': box}}
        return {'query_anchor_ids': ['R1'], 'anchors': {'R1': loc([490, 490, 510, 510])},
                'candidates': [{'alias': 'E1', 'kind': 'entity', 'locators': [loc([600, 490, 700, 510])]},
                               {'alias': 'E2', 'kind': 'entity', 'locators': [loc([490, 700, 510, 800])]}]}

    def test_nearest_uses_pixel_aspect_ratio_and_order_of_operations(self):
        # Horizontal gap=100px, vertical gap=20px despite normalized 100 vs200.
        n = nearest_region(self.view(), {'img': (1000, 100)})
        self.assertEqual(n['predicted_alias'], 'E2')
        self.assertEqual(n['squared_distance_numerators']['E1'] / n['distance_denominator'], 10000)
        self.assertEqual(n['squared_distance_numerators']['E2'] / n['distance_denominator'], 400)
        self.assertEqual(tuple(inspect.signature(nearest_region).parameters), ('view', 'image_sizes'))

    def test_nearest_text_inapplicable_ties_do_not_choose_gold(self):
        v = self.view()
        v['anchors']['R1'] = {'kind': 'text'}
        self.assertEqual(nearest_region(v, {})['reason'], 'text_R_has_no_visual_origin')
        v = self.view()
        v['candidates'][1]['locators'] = copy.deepcopy(v['candidates'][0]['locators'])
        n = nearest_region(v, {'img': (100, 100)})
        self.assertEqual(n['status'], 'tie')
        self.assertIsNone(n['predicted_alias'])

    def test_mother_macro_chance_parentheses(self):
        p = [{'problem_id': 'a', 'metadata': {'candidate_count': 2}},
             {'problem_id': 'a', 'metadata': {'candidate_count': 4}},
             {'problem_id': 'b', 'metadata': {'candidate_count': 10}}]
        r = chance_summary(p)
        self.assertEqual((r['numerator'], r['denominator']), (19, 80))  # ((1/2+1/4)/2+1/10)/2

    def test_exact_dedup_conflicts_and_development_no_overlap(self):
        r = [{'problem_id': 'a', 'source_identity': 'one', 'content_group_hint': 'x'},
             {'problem_id': 'b', 'source_identity': 'two', 'content_group_hint': 'x'},
             {'problem_id': 'c', 'source_identity': 'three', 'content_group_hint': 'y'},
             {'problem_id': 'd', 'source_identity': 'three', 'content_group_hint': 'z'}]
        dedup_members(r)
        self.assertEqual([x['dedup_state'] for x in r], ['representative', 'exact_duplicate', 'identity_version_conflict', 'identity_version_conflict'])


if __name__ == '__main__':
    unittest.main()
