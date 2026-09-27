"""Exercise the actual HF adapter against controlled SDK doubles, without weights/CUDA."""

from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from physalign.adapters import InfrastructureError, ModelRequest
from physalign.dataset import ImageData
from physalign.hf_adapter import HFAdapter, MODELS, freeze_snapshot, inspect_study_inputs, verify_snapshot, image_budget_kwargs
from physalign.storage import atomic_json, canonical, digest, loads


class Tensor:
    def __init__(self, values, shape):
        self.values, self.shape = values, shape

    def __getitem__(self, key):
        row, selection = key
        return Tensor(self.values[selection], (len(self.values[selection]),))

    def tolist(self):
        return self.values

    def numel(self):
        from math import prod
        return prod(self.shape)


class Batch(dict):
    def to(self, device):
        self.destination = device
        return self


class HFAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='physalign-hf-double-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.device = SimpleNamespace(type='cuda')
        self.batch = Batch(input_ids=Tensor([11, 12, 13], (1, 3)),
                           pixel_values=Tensor([], (1, 3, 8, 8)))
        self.processor = SimpleNamespace(
            chat_template='synthetic template', image_processor=SimpleNamespace(to_dict=lambda: {'synthetic': True}),
            apply_chat_template=Mock(side_effect=lambda *a, **k: self.batch),
            decode=Mock(side_effect=lambda t, **kw: '{"owner":"E1"}' if kw['skip_special_tokens'] else '{"owner":"E1"}<eos>'))
        self.model = SimpleNamespace(hf_device_map={'vision': 0, 'language': 1},
            get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(device=self.device)),
            generation_config=SimpleNamespace(eos_token_id=[99], pad_token_id=0, bos_token_id=1),
            generate=Mock(return_value=Tensor([11, 12, 13, 80, 99], (1, 5))))
        self.model.eval = lambda: self.model
        class OOM(RuntimeError):
            pass
        self.torch = SimpleNamespace(__version__='test-double', bfloat16='bf16', float16='fp16',
            version=SimpleNamespace(cuda='synthetic'), manual_seed=Mock(), inference_mode=nullcontext,
            cuda=SimpleNamespace(device_count=lambda: 2, is_bf16_supported=lambda: True,
                get_device_name=lambda i: 'synthetic GPU', get_device_properties=lambda i: SimpleNamespace(total_memory=24 * 2**30),
                manual_seed_all=Mock(), empty_cache=Mock(), OutOfMemoryError=OOM))
        self.sdk = SimpleNamespace(__version__='test-double', GenerationConfig=lambda **kw: SimpleNamespace(**kw),
            AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=self.processor)),
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=Mock(return_value=self.model)))
        modules = patch.dict('sys.modules', {'torch': self.torch, 'transformers': self.sdk,
                                            'accelerate': SimpleNamespace(__version__='test-double')})
        modules.start()
        self.addCleanup(modules.stop)
        stream = BytesIO()
        Image.new('RGB', (8, 8), 'red').save(stream, format='PNG')
        data = stream.getvalue()
        self.image = ImageData('img_0', 'image/png', 8, 8, digest(data), data)

    def adapter(self, model_id='Qwen/Qwen3.5-9B', **overrides):
        model = self.root / model_id.split('/')[-1]
        model.mkdir()
        atomic_json(model / 'config.json', {'model_type': MODELS[model_id]})
        (model / 'model.safetensors').write_bytes(b'synthetic SDK fixture, not weights')
        manifest = self.root / (model.name + '.json')
        freeze_snapshot(model, model_id, manifest)
        return HFAdapter({'snapshot_manifest': str(manifest), 'gpu_count': 2, 'thinking': False,
                          'max_new_tokens': 8, 'max_input_tokens': 32, **overrides})

    def request(self, adapter, *, images=True):
        return ModelRequest('test', 'EXACT public system', 'EXACT public user',
                            (self.image,) if images else (), canonical(adapter.info.manifest()['settings']))

    def test_all_three_checkpoint_families_use_offline_multimodal_loader(self):
        for name in MODELS:
            with self.subTest(model=name):
                adapter = self.adapter(name)
                kwargs = self.sdk.AutoModelForImageTextToText.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs['device_map'], 'balanced')
                self.assertEqual(kwargs['max_memory'], {0: '18GiB', 1: '18GiB'})
                self.assertTrue(kwargs['local_files_only'])
                self.assertFalse(kwargs['trust_remote_code'])
                self.assertEqual(kwargs['dtype'], 'bf16')
                self.assertEqual(kwargs['attn_implementation'], 'sdpa')
                self.assertEqual(adapter.template_kwargs, {} if name.startswith('OpenGVLab') else {'enable_thinking': False})

    def test_actual_generate_path_uses_fresh_messages_only_suffix_and_explicit_greedy_config(self):
        adapter = self.adapter()
        first = adapter.generate(self.request(adapter))
        second = adapter.generate(self.request(adapter, images=False))
        calls = self.processor.apply_chat_template.call_args_list
        self.assertEqual([m['role'] for m in calls[0].args[0]], ['system', 'user'])
        self.assertIsNot(calls[0].args[0], calls[1].args[0])
        self.assertEqual(calls[0].args[0][0]['content'][0]['text'], 'EXACT public system')
        self.assertEqual(len(calls[1].args[0][1]['content']), 1)
        self.assertNotIn('truncation', calls[0].kwargs)
        self.assertFalse(calls[0].kwargs['processor_kwargs']['text_kwargs']['truncation'])
        self.assertFalse(calls[0].kwargs['enable_thinking'])
        image = calls[0].args[0][1]['content'][2]['image']
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(loads(first.generation_json)['token_ids'], [80, 99])
        self.assertEqual((first.text, first.finish_reason), ('{"owner":"E1"}', 'stop'))
        self.assertEqual(loads(second.usage_json)['input_images'], 0)
        cfg = self.model.generate.call_args.kwargs['generation_config']
        self.assertEqual((cfg.do_sample, cfg.num_beams, cfg.max_new_tokens), (False, 1, 8))
        self.assertIsNot(self.model.generate.call_args_list[0].kwargs['generation_config'], cfg)
        self.assertEqual(self.torch.manual_seed.call_count, 2)

    def test_context_overflow_is_missing_with_no_generation_or_truncation(self):
        adapter = self.adapter(max_input_tokens=2)
        with self.assertRaises(InfrastructureError) as cm:
            adapter.generate(self.request(adapter))
        self.assertFalse(cm.exception.retryable)
        self.model.generate.assert_not_called()

    def test_qwen_image_cap_is_forwarded_at_call_and_recorded(self):
        options = image_budget_kwargs('Qwen/Qwen3.5-9B', 'multi-image-256-v1')
        self.processor.image_processor.patch_size = 16
        self.processor.image_processor.merge_size = 2
        self.batch['image_grid_thw'] = Tensor([[1, 32, 32]], (1, 3))
        adapter = self.adapter(processor_call_kwargs=options)
        adapter.generate(self.request(adapter))
        call = self.processor.apply_chat_template.call_args.kwargs
        self.assertEqual(call['processor_kwargs']['images_kwargs'], {'min_pixels': 65536, 'max_pixels': 262144})
        self.assertFalse(call['processor_kwargs']['text_kwargs']['truncation'])
        self.assertEqual(adapter.info.manifest()['preprocessing']['processor_call_kwargs'], options)
        self.assertNotIn('processor_call_kwargs', self.sdk.AutoProcessor.from_pretrained.call_args.kwargs)
        self.assertEqual(options, image_budget_kwargs('Qwen/Qwen3.5-9B', 'multi-image-256-v1'))

    def test_qwen_ignored_cap_fails_before_generation(self):
        self.processor.image_processor.patch_size = 16
        self.processor.image_processor.merge_size = 2
        self.batch['image_grid_thw'] = Tensor([[1, 64, 64]], (1, 3))
        adapter = self.adapter(processor_call_kwargs=image_budget_kwargs('Qwen/Qwen3.5-9B', 'multi-image-256-v1'))
        with self.assertRaisesRegex(ValueError, 'ignored the per-image'):
            adapter.generate(self.request(adapter))
        self.model.generate.assert_not_called()

    def test_internvl_one_tile_cap_and_ignored_cap_detection(self):
        self.processor.image_seq_length = 256
        self.batch['pixel_values'] = Tensor([], (1, 3, 448, 448))
        adapter = self.adapter('OpenGVLab/InternVL3_5-8B-HF',
            processor_call_kwargs=image_budget_kwargs('OpenGVLab/InternVL3_5-8B-HF', 'multi-image-256-v1'))
        adapter.generate(self.request(adapter))
        call = self.processor.apply_chat_template.call_args.kwargs
        self.assertEqual(call['processor_kwargs']['images_kwargs']['max_patches'], 1)
        self.model.generate.reset_mock()
        self.batch['pixel_values'] = Tensor([], (13, 3, 448, 448))
        with self.assertRaisesRegex(ValueError, 'ignored the one-tile'):
            adapter.generate(self.request(adapter))
        self.model.generate.assert_not_called()

    def test_processor_call_overrides_cannot_enable_text_truncation(self):
        with self.assertRaisesRegex(ValueError, 'Only image preprocessing'):
            self.adapter(processor_call_kwargs={'text_kwargs': {'truncation': True}})

    def test_preflight_and_inference_use_identical_bounded_processor_options(self):
        self.processor.image_processor.patch_size = 16
        self.processor.image_processor.merge_size = 2
        self.batch['image_grid_thw'] = Tensor([[1, 32, 32]], (1, 3))
        adapter = self.adapter(processor_call_kwargs=image_budget_kwargs('Qwen/Qwen3.5-9B', 'multi-image-256-v1'))
        study = SimpleNamespace(plan={'plan_hash': 'fixture', 'requests': [
            {'instance_id': 'p1', 'condition': 'main.raw', 'input_hash': 'fixture'}]},
            request=lambda *args: self.request(adapter))
        with patch('physalign.study.Study', return_value=study):
            inspect_study_inputs('fixture', adapter.config, self.root / 'bounded.json')
        before = self.processor.apply_chat_template.call_args.kwargs
        adapter.generate(self.request(adapter))
        self.assertEqual(before, self.processor.apply_chat_template.call_args.kwargs)

    def test_oom_is_missing_without_adaptive_downsampling_or_second_generation(self):
        adapter = self.adapter()
        self.model.generate.side_effect = self.torch.cuda.OutOfMemoryError('synthetic OOM')
        with self.assertRaises(InfrastructureError) as cm:
            adapter.generate(self.request(adapter))
        self.assertFalse(cm.exception.retryable)
        self.assertEqual(self.model.generate.call_count, 1)
        self.torch.cuda.empty_cache.assert_called_once()

    def test_missing_pixels_is_a_processor_error_and_not_an_answer(self):
        adapter = self.adapter()
        del self.batch['pixel_values']
        with self.assertRaisesRegex(ValueError, 'omitted image tensors'):
            adapter.generate(self.request(adapter))
        self.model.generate.assert_not_called()

    def test_cpu_offload_and_implicit_thinking_are_rejected(self):
        self.model.hf_device_map['language'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'offloaded'):
            self.adapter()
        with self.assertRaisesRegex(ValueError, 'thinking=false'):
            self.adapter('Qwen/Qwen3.5-27B', thinking=True)

    def test_decode_keeps_malformed_or_truncated_response_without_repair(self):
        adapter = self.adapter(max_new_tokens=2)
        self.model.generate.return_value = Tensor([11, 12, 13, 80, 81], (1, 5))
        self.processor.decode.side_effect = lambda *a, **k: 'thinking\n{"owner":'
        response = adapter.generate(self.request(adapter))
        self.assertEqual(response.text, 'thinking\n{"owner":')
        self.assertEqual(response.finish_reason, 'length')

    def test_preflight_reuses_exact_encoder_without_loading_model(self):
        adapter = self.adapter()
        self.sdk.AutoModelForImageTextToText.from_pretrained.reset_mock()
        study = SimpleNamespace(plan={'plan_hash': 'fixture', 'requests': [
            {'instance_id': 'p1', 'condition': 'main.raw', 'input_hash': 'fixture'}]},
            request=lambda *args: self.request(adapter))
        with patch('physalign.study.Study', return_value=study):
            report = inspect_study_inputs('unused fixture', adapter.config, self.root / 'preflight.json')
            overflow = inspect_study_inputs('unused fixture', {**adapter.config, 'max_input_tokens': 2}, self.root / 'overflow.json')
        self.assertTrue(report['all_within_budget'])
        self.assertFalse(overflow['all_within_budget'])
        self.assertEqual(report['requests'][0]['input_tokens'], 3)
        self.sdk.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        self.model.generate.assert_not_called()

    def test_native_context_includes_reserved_output_budget(self):
        adapter = self.adapter()
        adapter.native_context = 10  # 3 input + 8 reserved output > 10.
        with self.assertRaises(InfrastructureError) as cm:
            adapter.generate(self.request(adapter))
        self.assertFalse(cm.exception.retryable)
        self.model.generate.assert_not_called()

    def test_added_processor_configuration_is_not_silently_loaded_after_freeze(self):
        adapter = self.adapter()
        atomic_json(adapter.root / 'new_processor_config.json', {'different': True})
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            verify_snapshot(adapter.snapshot)
