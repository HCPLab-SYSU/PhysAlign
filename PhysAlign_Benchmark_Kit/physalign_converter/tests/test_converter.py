import copy
import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import SourceAdapter, verify_adapter_snapshot
from compiler import Compiler
from convert import main
from io_utils import read_json, write_json, empty_ledger, safe_path, parse_json
from reference_checks import digest, resolve_span, ContractError
from reference_pipeline import config, prepare_probe, freeze_probe, validate_frozen
from reference_integrity import active_origin
from review_workflow import prospective, apply_decision, CONFIRMATIONS
from fixtures import workspace, registry


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotation, self.dataset, self.pid = workspace(self.root)

    def adapter(self):
        return SourceAdapter(self.annotation, self.dataset)

    def adapted(self):
        a = self.adapter()
        return a.materialize(a.load(self.pid), self.root / 'materialized')

    def compiler(self, **registry_options):
        return Compiler(self.adapted(), *registry(**registry_options))

    def case(self, task='T03', **kwargs):
        c = self.compiler(**kwargs)
        return c.compile(task, c.proposals(task)[0])

    def decision(self, d, env):
        return {'logical_probe_id': d['key']['logical_probe_id'], 'review_target_root': prospective(d, env)[1]['content_root'],
                'status': 'approved', 'reviewer': 'SYNTHETIC_TEST_NOT_HUMAN', 'blind_answer': 'SYNTHETIC_TEST_E1',
                'confirmations': {c: True for c in CONFIRMATIONS}, 'note': 'Test ONLY; not real human review.'}

    def mutate_source(self, stage, edit, reapprove=True):
        path = self.annotation / 'passes' / stage / (self.pid + '.json')
        doc = read_json(path); edit(doc); write_json(path, doc)
        if reapprove:
            path = self.annotation / 'reviews/state.json'; state = read_json(path)
            state['problems'][self.pid]['stages'][stage]['document_sha256'] = digest(doc)
            write_json(path, state)

    def test_source_ready_and_reconstructs(self):
        self.assertEqual(self.adapter().inventory()['counts'], {'source_ready': 1})
        a = self.adapted(); verify_adapter_snapshot(a['env'])

    def test_pipeline_relative_dataset_path_is_workspace_relative(self):
        write_json(self.annotation / 'workspace_config.json',
                   {'path_base': 'workspace', 'dataset_dir': '../dataset'})
        adapter = SourceAdapter(self.annotation)
        self.assertEqual(adapter.dataset, self.dataset.resolve())
        self.assertEqual(adapter.inventory()['counts'], {'source_ready': 1})
        materialized = adapter.materialize(adapter.load(self.pid), self.root / 'portable')
        verify_adapter_snapshot(materialized['env'])

    def test_explicit_dataset_override_takes_precedence(self):
        write_json(self.annotation / 'workspace_config.json',
                   {'path_base': 'workspace', 'dataset_dir': '../missing'})
        self.assertEqual(SourceAdapter(self.annotation, self.dataset).dataset, self.dataset.resolve())

    def test_stale_approval_blocks(self):
        self.mutate_source('pass4', lambda d: d.update(domain='mixed'), reapprove=False)
        with self.assertRaisesRegex(ValueError, 'HASH_STALE'): self.adapter().load(self.pid)

    def test_schema_invalid_even_with_fresh_approval(self):
        self.mutate_source('pass4', lambda d: d['bindings'][0].update(to_id='p999'))
        with self.assertRaisesRegex(ValueError, 'SCHEMA_INVALID'): self.adapter().load(self.pid)

    def test_pass5_not_read(self):
        path = self.annotation / 'passes/pass5' / (self.pid + '.json'); path.parent.mkdir()
        path.write_text('INVALID SECRET SOLUTION', encoding='utf-8')
        self.assertTrue(self.adapter().load(self.pid))

    def test_exclusion_gate(self):
        path = self.annotation / 'reviews/state.json'; state = read_json(path)
        state['problems'][self.pid]['exclusion'] = {'status': 'excluded'}; write_json(path, state)
        with self.assertRaisesRegex(ValueError, 'EXCLUDED'): self.adapter().load(self.pid)

    def test_pass5_only_exclusion_does_not_block(self):
        path = self.annotation / 'reviews/state.json'; state = read_json(path)
        state['problems'][self.pid]['pass5_exclusion'] = {'status': 'excluded'}; write_json(path, state)
        self.assertTrue(self.adapter().load(self.pid))

    def test_pending_preview_without_forged_approval(self):
        d, env, p = self.case()
        self.assertEqual(d['key']['packet_plan']['state'], 'pending_review')
        self.assertFalse(p['review_checks_completed'])
        self.assertNotIn('qa-instance', env['ledger']['records'])
        with self.assertRaises(ValueError): freeze_probe(d, env, ['qa-instance'])

    def test_prospective_does_not_mutate(self):
        d, env, _ = self.case(); before = digest([d, env]); prospective(d, env)
        self.assertEqual(before, digest([d, env]))

    def test_explicit_decision_roundtrip(self):
        d, env, _ = self.case(); f, e = apply_decision(d, env, self.decision(d, env)); validate_frozen(f, e)
        self.assertEqual(f['core']['raw_view']['recognition_packet'], [])
        self.assertEqual(f['core']['gold_view']['recognition_packet'][0]['text'], 'm')

    def test_stale_review_rejected(self):
        d, env, _ = self.case(); decision = self.decision(d, env); decision['review_target_root'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'DECISION_STALE'): apply_decision(d, env, decision)

    def test_confirmation_cannot_be_omitted(self):
        d, env, _ = self.case(); decision = self.decision(d, env); decision['confirmations']['complete_targets'] = False
        with self.assertRaisesRegex(ValueError, 'CONFIRMATIONS'): apply_decision(d, env, decision)

    def test_textual_read_cannot_enter_main_joint_pool(self):
        d, env, _ = self.case('T02'); loc = d['view']['anchors']['R1']; oid = 'o_forbidden_text'
        ref = {'collection': 'text_mentions', 'record_id': 'm001', 'target_field': 'quote', 'source_file_sha256': d['key']['source_hashes']['pass4']}
        origin = active_origin(loc, d['view']['context'], env['assets'])
        env['observations']['observations'][oid] = {'observation_id': oid, 'state': 'approved', 'origin': origin,
            'text': loc['quote'], 'source_records': [ref], 'source_kind': 'text_mention', 'asset_role': 'text', 'review_ref': 'test'}
        d['key']['anchor_map']['R1']['observation_ids'] = [oid]
        d['view']['read_anchor_ids'] = ['R1']; d['key']['response_contract'].update(read_shape='string', read_anchor_ids=['R1'])
        d['key']['read_targets'] = [{'anchor_alias': 'R1', 'observation_id': oid, 'normalizer_id': 'ocr_literal_v2_1'}]
        d['key']['joint_eligible'] = True
        with self.assertRaisesRegex(ValueError, 'MAIN_JOINT_REQUIRES_IMAGE_READING'): prepare_probe(d, env, verify_reviews=False)

    def test_plural_reference_keeps_all_and_blocks_core(self):
        self.mutate_source('pass4', lambda d: d['bindings'].append({**d['bindings'][-1], 'id': 'b004', 'to_id': 'p002'}))
        c = self.compiler(); groups = c.proposals('T02')
        self.assertEqual(len(groups), 1); self.assertEqual(len(groups[0]['rows']), 2)
        with self.assertRaisesRegex(ValueError, 'SET_EXTENSION_NOT_CORE'): c.compile('T02', groups[0])

    def test_same_entity_merge_from_explicit_edges_only(self):
        self.mutate_source('pass4', lambda d: d['bindings'].append({'id': 'b004', 'type': 'same_entity_as',
            'from_id': 'p001', 'to_id': 'p002', 'provenance': ['IMAGE'], 'evidence_visual_ids': ['v001', 'v002'], 'evidence_mention_ids': []}))
        a = self.adapted(); self.assertEqual(len(set(a['env']['source_index']['identity_map'].values())), 1)

    def test_no_name_based_identity_merge(self):
        self.mutate_source('pass4', lambda d: d['physical_nodes'][1].update(name='Block 1'))
        self.assertEqual(len(set(self.adapted()['env']['source_index']['identity_map'].values())), 2)

    def test_t01_registered_roles(self):
        d, env, _ = self.case('T01'); self.assertEqual(d['key']['gold_targets'], [{'kind': 'role', 'id': 'body'}])

    def test_empty_registry_blocks_roles_not_other_tasks(self):
        c = Compiler(self.adapted(), config('semantic_registry.json'), empty_ledger())
        with self.assertRaisesRegex(ValueError, 'ROLE_REGISTRY_REQUIRED'): c.proposals('T01')
        self.assertEqual(len(c.proposals('T03')), 1)

    def test_t04_set_including_all_subjects(self):
        d, env, _ = self.case('T04'); self.assertEqual(len(d['key']['gold_targets']), 2)
        self.assertEqual(d['key']['response_contract']['cardinality'], 'set')

    def test_t05_directed(self):
        d, env, _ = self.case('T05'); f, e = apply_decision(d, env, self.decision(d, env)); validate_frozen(f, e)

    def test_t05_inverse(self):
        d, env, _ = self.case('T05', inverse=True)
        self.assertEqual(d['key']['source_records'][0]['target_field'], 'subject_id')

    def test_t05_symmetric_reverse_endpoint(self):
        c = self.compiler(symmetric=True); gs = c.proposals('T05'); self.assertEqual(len(gs), 2)
        for g in gs:
            d, env, _ = c.compile('T05', g); f, e = apply_decision(d, env, self.decision(d, env)); validate_frozen(f, e)

    def test_duplicate_candidate_geometry_blocks(self):
        self.mutate_source('pass4', lambda d: d['physical_nodes'][1].update(visual_anchor_ids=['v001']))
        c = self.compiler()
        with self.assertRaisesRegex(ValueError, 'INDISTINGUISHABLE_CANDIDATES'): c.compile('T03', c.proposals('T03')[0])

    def test_snapshot_reprojection_detects_tampering(self):
        a = self.adapted(); a['env']['source_index']['entity_types'][next(iter(a['env']['source_index']['entity_types']))] = 'force'
        with self.assertRaisesRegex(ValueError, 'PROJECTION_CHANGED'): verify_adapter_snapshot(a['env'])

    def test_safe_paths_and_strict_json(self):
        for name in ('../secret', 'C:/secret', '/secret', 'images\\secret'):
            with self.subTest(name=name), self.assertRaises(ValueError): safe_path(self.root, name)
        for blob in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.subTest(blob=blob), self.assertRaises(ValueError): parse_json(blob)

    def test_unique_span_proposed_not_auto_repaired(self):
        self.assertEqual(resolve_span('block A', 'block A', 9)['status'], 'unique_exact_repair_proposed')
        self.assertIsNone(resolve_span('block A', 'block A', 9)['span'])

    def test_source_options_label_to_id_exact_mapping(self):
        path = self.annotation / 'blind/manifest.jsonl'; row = read_json(path)
        row['segments']['options'] = [{'label': 'A', 'text': 'unchanged $x$'}]
        write_json(path, row)
        a = self.adapted()
        self.assertEqual(a['env']['source_index']['context']['options'], [{'id': 'A', 'text': 'unchanged $x$'}])
        verify_adapter_snapshot(a['env'])

    def test_source_image_change_is_not_silently_accepted(self):
        path = self.dataset / 'images/diagram.png'; path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'SOURCE_IMAGE_HASH_STALE'): self.adapted()

    def test_unicode_nonoverlap_offsets_not_utf16_or_bytes(self):
        text = '😀e\u0301\r\n甲甲甲'
        self.assertEqual(resolve_span(text, '甲甲', 1)['span'], [5, 7])
        self.assertIsNone(resolve_span(text, '甲甲', 2)['span'])

    def test_serialization_order_does_not_change_prompt(self):
        d, env, p = self.case()
        d2 = parse_json(__import__('json').dumps(d, sort_keys=True))
        self.assertEqual(prepare_probe(d2, env, verify_reviews=False), p)

    def test_many_aliases_roundtrip_numeric_and_lexical_order(self):
        def extra(d):
            for i in range(4, 15):
                d['visual_nodes'].append({**d['visual_nodes'][2], 'id': f'v{i:03}', 'text': f'F{i}',
                                           'bbox_1000': [10*i, 100, 10*i+5, 130]})
        self.mutate_source('pass1', extra); self.mutate_source('pass4', extra)
        d, env, p = self.case('T02')
        self.assertIn('R10', d['view']['anchors'])
        d2 = parse_json(__import__('json').dumps(d, sort_keys=True))
        self.assertEqual(prepare_probe(d2, env, verify_reviews=False), p)

    def test_unknown_quantity_does_not_become_blank_answer(self):
        self.mutate_source('pass4', lambda d: d['quantities'][0].update(owner_id=''))
        c = self.compiler()
        with self.assertRaisesRegex(ValueError, 'OWNER_UNRESOLVED'): c.proposals('T03')

    def test_approved_hash_invalid_occurrence_is_still_unresolved(self):
        self.mutate_source('pass3', lambda d: d['text_mentions'][2].update(occurrence=99))
        self.mutate_source('pass4', lambda d: d['text_mentions'][2].update(occurrence=99))
        a = self.adapted()
        self.assertNotIn('m003', a['locators'])
        self.assertEqual(a['audit']['span_decisions']['m003']['status'], 'ambiguous_occurrence')

    def test_schema_error_is_a_local_contract_error(self):
        from schema_tools import validate_schema
        with self.assertRaisesRegex(ValueError, 'BENCHMARK_SCHEMA_INVALID'):
            validate_schema('view_bundle.schema.json', {})

    def test_ambiguous_span_blocked(self):
        self.assertEqual(resolve_span('A A', 'A', 9)['status'], 'ambiguous_occurrence')

    def test_chinese_codepoint_offsets(self):
        annotation, dataset, pid = workspace(self.root / '中文 空格', 'zh')
        a = SourceAdapter(annotation, dataset); adapted = a.materialize(a.load(pid), self.root / '中文输出')
        c = Compiler(adapted, *registry()); d, env, _ = c.compile('T02', c.proposals('T02')[0])
        self.assertEqual(d['view']['anchors']['R1']['quote'], '物块甲')
        self.assertEqual(d['view']['anchors']['R1']['end'], 3)

    def test_cli_partial_freeze_relocation_and_private_allowlist(self):
        draft = self.root / 'draft'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['draft', '--workspace', str(self.annotation), '--output', str(draft), '--tasks', 'T02,T03,T04'])
            main(['validate', '--path', str(draft)])
        cat = read_json(draft / 'catalog.PRIVATE.json'); self.assertEqual(len(cat['probes']), 3)
        decision = read_json(draft / 'decisions.template.json'); decision['decisions'] = decision['decisions'][:1]
        row = decision['decisions'][0]; row.update(status='approved', reviewer='SYNTHETIC_TEST_NOT_HUMAN', blind_answer='test', confirmations={k: True for k in CONFIRMATIONS})
        path = self.root / 'decisions.json'; write_json(path, decision)
        release = self.root / 'release'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['freeze', '--draft', str(draft), '--decisions', str(path), '--output', str(release)])
        moved = self.root / '搬迁 空格 release'; shutil.copytree(release, moved)
        with contextlib.redirect_stdout(io.StringIO()): main(['validate', '--path', str(moved)])
        qa = read_json(moved / 'qa_raw.json'); self.assertEqual(len(qa), 1)
        text = str(qa[0]['input']); self.assertNotIn('gold_targets', text); self.assertNotIn('Block 1', text)
        self.assertNotIn(str(self.dataset), text); self.assertNotIn('source_file_sha256', text)
        with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
            with contextlib.redirect_stdout(io.StringIO()): main(['freeze', '--draft', str(draft), '--decisions', str(path), '--output', str(release)])


if __name__ == '__main__':
    unittest.main()
