"""Real companion generation/preflight paths with controlled HF SDK doubles."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
import test_hf_adapter as hf_fixture
from physalign import hf_adapter as original
from physalign.adapters import InfrastructureError
from physalign.dataset import PublicDataset
from physalign.planning import prepare_plan
from physalign.reporting import score_run
from physalign.runner import run_evaluation
from physalign.storage import atomic_json, canonical, fingerprint, loads, read_json
from server_eval import qwen35_4b as driver
from server_eval.qwen35_4b_adapter import MODEL_ID, Qwen35_4BAdapter, backend, freeze_snapshot
from server_eval import compare_models
from test_two_experiments import fixture, FixtureAdapter, pipeline


class Qwen4Tests(unittest.TestCase):
    def setUp(self):
        self.fx = hf_fixture.HFAdapterTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.root = self.fx.root
        self.fx.torch.cuda.device_count = lambda: 1
        self.fx.model.hf_device_map = {'vision': 0, 'language': 0}
        self.fx.device.index = 0
        self.weights = [('vision.weight', SimpleNamespace(device=SimpleNamespace(type='cuda', index=0), numel=lambda: 16)),
                        ('language.weight', SimpleNamespace(device=SimpleNamespace(type='cuda', index=0), numel=lambda: 32))]
        self.buffers = [('rotary.inv_freq', SimpleNamespace(device=SimpleNamespace(type='cuda', index=0), numel=lambda: 4))]
        self.fx.model.named_parameters = lambda: iter(self.weights)
        self.fx.model.named_buffers = lambda: iter(self.buffers)
        self.fx.processor.tokenizer = SimpleNamespace(encode=lambda text, **kw: [81], unk_token_id=0)
        self.pieces = {80: 'reasoning with {"owner":"WRONG"}', 81: '</think>',
                       82: '\n\n{"owner":"E1"}', 83: 'EXTRA PROSE', 99: '<eos>'}
        def decode(value, **kw):
            ids = value.tolist() if hasattr(value, 'tolist') else value
            return ''.join(self.pieces[i] for i in ids if not (kw['skip_special_tokens'] and i in (81, 99)))
        self.fx.processor.decode = Mock(side_effect=decode)
        model = self.root / 'model'
        model.mkdir()
        atomic_json(model / 'config.json', {'model_type': 'qwen3_5', 'text_config': {'max_position_embeddings': 262144}})
        (model / 'model.safetensors').write_bytes(b'not real weights: SDK fixture')
        self.snapshot = self.root / 'snapshot.json'
        freeze_snapshot(model, self.snapshot)
        self.config = {'snapshot_manifest': str(self.snapshot), 'gpu_count': 1, 'thinking': True,
                       'max_new_tokens': 8, 'max_input_tokens': 32}
        self.set_output([80, 81, 82, 99])

    def set_output(self, ids):
        self.fx.model.generate.return_value = hf_fixture.Tensor([11, 12, 13] + ids, (1, 3 + len(ids)))

    def adapter(self, **overrides):
        return Qwen35_4BAdapter({**self.config, **overrides})

    def setup_plan(self):
        with redirect_stdout(StringIO()):
            self.source = fixture(self.root / 'release')
            pipeline.convert_release(self.source, self.root / 'bundle')
        self.ds = PublicDataset(self.root / 'bundle')
        self.plan = prepare_plan(self.ds.root, split=pipeline.SPLIT, bootstrap_resamples=10)
        self.preflight = self.root / 'preflight.json'
        with redirect_stdout(StringIO()):
            driver.preflight(self.plan, self.ds, 'raw_gold', self.config, self.preflight)

    def test_private_registration_retains_original_registry_and_exact_identity(self):
        before = dict(original.MODELS)
        adapter = self.adapter()
        self.assertEqual(original.MODELS, before)
        self.assertNotIn(MODEL_ID, original.MODELS)
        self.assertEqual(backend.MODELS[MODEL_ID], 'qwen3_5')
        self.assertEqual(adapter.info.model_id, MODEL_ID)
        kwargs = self.fx.sdk.AutoModelForImageTextToText.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs['max_memory'], {0: '18GiB'})
        self.assertTrue(kwargs['local_files_only'])
        wrong = read_json(self.snapshot)
        wrong['model_id'] = 'Qwen/Qwen3.5-9B'
        atomic_json(self.snapshot, wrong)
        with self.assertRaisesRegex(ValueError, 'exact Qwen'):
            self.adapter()

    def test_default_is_nonthinking_with_2048_budget(self):
        config = dict(self.config)
        del config['thinking'], config['max_new_tokens']
        adapter = Qwen35_4BAdapter(config)
        settings = adapter.info.manifest()['settings']
        self.assertFalse(settings['thinking'])
        self.assertEqual(adapter.max_new, 2048)
        self.assertEqual(adapter.config['thinking'], False)
        self.assertEqual(settings['output_budget_scope'], 'reasoning_and_final_combined')
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.adapter(thinking='true')

    def test_single_cuda_load_without_device_map_is_accepted_after_tensor_audit(self):
        del self.fx.model.hf_device_map
        adapter = self.adapter(thinking=False)
        self.assertEqual(self.fx.model.hf_device_map, {'': 0})
        audit = adapter.info.manifest()['preprocessing']['gpu_residency_audit']
        self.assertEqual(audit['device_map_source'], 'inferred_from_actual_single_cuda_device')
        self.assertEqual(audit['tensor_devices'], {'parameters': {'cuda:0': 2}, 'buffers': {'cuda:0': 1}})
        self.assertEqual(audit['parameter_count'], 48)
        self.assertEqual(self.fx.sdk.AutoModelForImageTextToText.from_pretrained.call_count, 1)
        self.set_output([82, 99])
        self.assertEqual(adapter.generate(self.fx.request(adapter)).text, '\n\n{"owner":"E1"}')

    def test_empty_device_map_is_accepted_only_with_verified_cuda_tensors(self):
        self.fx.model.hf_device_map = {}
        self.adapter()
        self.assertEqual(self.fx.model.hf_device_map, {'': 0})

    def test_real_cpu_meta_and_wrong_gpu_parameters_still_stop_before_inference(self):
        for device_type, index in [('cpu', None), ('meta', None), ('cuda', 1)]:
            with self.subTest(device=device_type, index=index):
                self.fx.model.hf_device_map = {}
                self.weights[1][1].device = SimpleNamespace(type=device_type, index=index)
                with self.assertRaisesRegex(ValueError, 'outside the assigned CUDA'):
                    self.adapter()
                self.assertEqual(self.fx.model.hf_device_map, {})
        self.fx.model.generate.assert_not_called()

    def test_gpu_embedding_alone_does_not_hide_cpu_parameters_or_buffers(self):
        self.weights[1][1].device = SimpleNamespace(type='cpu', index=None)
        with self.assertRaisesRegex(ValueError, 'parameters:language.weight=cpu'):
            self.adapter()
        self.weights[1][1].device = SimpleNamespace(type='cuda', index=0)
        self.buffers[0][1].device = SimpleNamespace(type='cpu', index=None)
        with self.assertRaisesRegex(ValueError, 'buffers:rotary.inv_freq=cpu'):
            self.adapter()

    def test_explicit_disk_offload_metadata_is_not_overwritten(self):
        self.fx.model.hf_device_map = {'vision': 0, 'language': 'disk'}
        with self.assertRaisesRegex(ValueError, 'device map contains'):
            self.adapter()
        self.assertEqual(self.fx.model.hf_device_map['language'], 'disk')

    def test_existing_multi_gpu_mapping_is_retained(self):
        self.fx.torch.cuda.device_count = lambda: 2
        self.weights[1][1].device = SimpleNamespace(type='cuda', index=1)
        self.fx.model.hf_device_map = {'vision': 0, 'language': 1}
        adapter = self.adapter(gpu_count=2)
        self.assertEqual(adapter.model.hf_device_map, {'vision': 0, 'language': 1})
        self.assertEqual(adapter.info.manifest()['preprocessing']['gpu_residency_audit']['device_map_source'], 'reported')

    def test_thinking_splits_only_marker_and_preserves_full_generation(self):
        adapter = self.adapter()
        response = adapter.generate(self.fx.request(adapter))
        self.assertEqual(response.text, '\n\n{"owner":"E1"}')
        trace, usage = loads(response.generation_json), loads(response.usage_json)
        self.assertEqual(trace['token_ids'], [80, 81, 82, 99])
        self.assertIn('WRONG', trace['reasoning_content'])
        self.assertIn('</think>', trace['decoded_with_special_tokens'])
        self.assertTrue(trace['reasoning_closed'])
        self.assertEqual(usage['reasoning_tokens'] + usage['final_tokens'], usage['output_tokens'])
        self.assertEqual(self.fx.processor.apply_chat_template.call_args.kwargs['enable_thinking'], True)
        self.assertFalse(self.fx.model.generate.call_args.kwargs['generation_config'].do_sample)
        self.assertEqual(response.finish_reason, 'stop')

    def test_unfinished_thinking_never_promotes_reasoning_json_or_retries(self):
        self.set_output([80] * 8)
        adapter = self.adapter()
        response = adapter.generate(self.fx.request(adapter))
        self.assertEqual(response.text, '')
        self.assertEqual(response.finish_reason, 'length')
        self.assertFalse(loads(response.generation_json)['reasoning_closed'])
        self.assertEqual(self.fx.model.generate.call_count, 1)

    def test_final_prose_and_extra_marker_are_not_repaired(self):
        self.set_output([80, 81, 83, 82, 81, 99])
        adapter = self.adapter()
        response = adapter.generate(self.fx.request(adapter))
        self.assertEqual(response.text, 'EXTRA PROSE\n\n{"owner":"E1"}')
        self.assertEqual(loads(response.generation_json)['final_token_offset'], 2)

    def test_explicit_nonthinking_preserves_entire_suffix(self):
        self.set_output([83, 82, 99])
        adapter = self.adapter(thinking=False)
        response = adapter.generate(self.fx.request(adapter))
        self.assertEqual(response.text, 'EXTRA PROSE\n\n{"owner":"E1"}')
        self.assertFalse(self.fx.processor.apply_chat_template.call_args.kwargs['enable_thinking'])
        self.assertNotIn('reasoning_content', loads(response.generation_json))

    def test_input_budget_and_oom_remain_terminal_infrastructure_errors(self):
        adapter = self.adapter(max_input_tokens=2)
        with self.assertRaisesRegex(InfrastructureError, 'frozen_input'):
            adapter.generate(self.fx.request(adapter))
        self.fx.model.generate.assert_not_called()
        adapter = self.adapter()
        self.fx.model.generate.side_effect = self.fx.torch.cuda.OutOfMemoryError()
        with self.assertRaises(InfrastructureError) as caught:
            adapter.generate(self.fx.request(adapter))
        self.assertFalse(caught.exception.retryable)

    def test_preflight_checks_every_public_request_with_actual_thinking_template(self):
        self.setup_plan()
        report = driver.checked_preflight(self.plan, 'raw_gold', self.config, self.preflight)
        self.assertTrue(report['thinking'])
        self.assertEqual(len(report['requests']), len(self.plan['requests']))
        self.assertTrue(all(c.kwargs['enable_thinking'] for c in self.fx.processor.apply_chat_template.call_args_list))
        self.fx.sdk.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        changed = {**self.config, 'thinking': False}
        with self.assertRaisesRegex(ValueError, 'changed'):
            driver.checked_preflight(self.plan, 'raw_gold', changed, self.preflight)
        report['requests'].pop()
        report['report_hash'] = fingerprint({k: v for k, v in report.items() if k != 'report_hash'})
        atomic_json(self.preflight, report)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            driver.checked_preflight(self.plan, 'raw_gold', self.config, self.preflight)

    def test_preflight_over_budget_fails_before_gpu_or_inference(self):
        self.setup_plan()
        config = {**self.config, 'max_input_tokens': 2}
        with redirect_stdout(StringIO()), self.assertRaisesRegex(ValueError, 'budget exceeded'):
            driver.preflight(self.plan, self.ds, 'raw_gold', config, self.preflight)
        with patch.object(driver, 'wait_for_gpus') as wait, self.assertRaises(ValueError):
            driver.run(self.plan, self.ds, 'raw_gold', config, self.root / 'run', self.preflight, gpus=['0'])
        wait.assert_not_called()
        self.fx.model.generate.assert_not_called()

    def test_scoring_and_completed_resume_reuse_existing_protocol(self):
        self.setup_plan()
        adapter = FixtureAdapter(self.source, self.ds)
        adapter.info = replace(adapter.info, model_id=MODEL_ID, settings_json=canonical({'thinking': True, 'max_new_tokens': 32768}),
                               preprocessing_json=canonical({'cuda_visible_devices': '0'}))
        output = self.root / 'run'
        run_evaluation(self.plan, adapter, output, adapter_configuration_hash=fingerprint(self.config))
        with patch.object(driver, 'wait_for_gpus') as wait, patch.object(driver, 'Qwen35_4BAdapter') as load, redirect_stdout(StringIO()):
            driver.run(self.plan, self.ds, 'raw_gold', self.config, output, self.preflight, gpus=['0'])
            report = driver.score(self.plan, self.ds, 'raw_gold', self.config, output)
        wait.assert_not_called()
        load.assert_not_called()
        self.assertEqual(report['raw_all'], score_run(output)['raw_all'])
        models, _ = compare_models.audit_models(self.plan, [output], allow_non_scientific=True)
        self.assertEqual(models[0]['label'], 'Qwen3.5-4B (thinking)')
        self.assertEqual(compare_models.protocol_rows(models)[0]['reasoning'], 'thinking')
        with self.assertRaisesRegex(ValueError, 'config changed'):
            driver.audit_existing(self.plan, {**self.config, 'thinking': False}, output)
        with self.assertRaisesRegex(ValueError, 'GPU assignment'):
            driver.run(self.plan, self.ds, 'raw_gold', self.config, output, self.preflight, gpus=['1'])

    def test_full_study_keeps_actual_nonthinking_with_custom_display_label(self):
        self.setup_plan()
        self.config['thinking'] = False
        self.set_output([82, 99])
        from physalign.study import prepare_study, run_study
        from physalign.study_reporting import score_study
        study_root = self.root / 'study'
        prepare_study(self.ds.root, study_root, split=pipeline.SPLIT, bootstrap_resamples=0)
        plan, study, kind = driver.source(study_root=study_root)
        with redirect_stdout(StringIO()):
            report = driver.preflight(plan, study, kind, self.config, self.root / 'study-preflight.json')
        self.assertEqual(len(report['requests']), len(plan['requests']))
        adapter = self.adapter()
        run_study(study_root, adapter, self.root / 'study-run', configuration_hash=fingerprint(self.config))
        scored = score_study(study_root, self.root / 'study-run')
        self.assertEqual(scored['service']['completed'], len(plan['requests']))
        self.assertFalse(scored['adapter']['settings']['thinking'])
        from physalign_viz.data import load_bundle
        labels = {MODEL_ID: 'Example display label'}
        bundle = load_bundle(study_root, [self.root / 'study-run'], labels=labels, diagnostic_ci=False)
        self.assertEqual(bundle['models'][0]['label'], 'Example display label')
        self.assertFalse(bundle['model_protocols'][MODEL_ID]['settings']['thinking'])



if __name__ == '__main__':
    unittest.main()
