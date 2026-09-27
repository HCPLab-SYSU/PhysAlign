"""Local HF inference for the three requested checkpoints; no automatic downloads.

Reference APIs: Qwen3.5 and InternVL official Transformers model documentation.
Imports of torch/transformers/Pillow are lazy so scoring stays CPU/stdlib-only.
"""

from io import BytesIO
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import platform

from .adapters import AdapterInfo, InfrastructureError, ModelResponse
from .dataset import require
from .storage import canonical, file_hash, fingerprint, read_json, write_new

MODELS = {"Qwen/Qwen3.5-9B": "qwen3_5", "Qwen/Qwen3.5-27B": "qwen3_5",
          "OpenGVLab/InternVL3_5-8B-HF": "internvl"}


def image_budget_kwargs(model_id, profile):
    """An explicit pre-run image policy; never selected using model outcomes."""
    require(model_id in MODELS, 'Unsupported model for image budget')
    require(profile in {'checkpoint-default', 'multi-image-256-v1'}, 'Unknown visual profile')
    if profile == 'checkpoint-default':
        return {}
    if MODELS[model_id] == 'qwen3_5':
        # Qwen3.5 has patch_size=16, merge_size=2: 1024 pixels per visual token.
        return {'images_kwargs': {'min_pixels': 65536, 'max_pixels': 262144}}
    # InternVL has 256 visual tokens per 448x448 tile. One tile => no extra thumbnail.
    return {'images_kwargs': {'crop_to_patches': True, 'min_patches': 1, 'max_patches': 1}}


def processing_kwargs(overrides=None):
    overrides = deepcopy(overrides or {})
    require(isinstance(overrides, dict) and set(overrides) <= {'images_kwargs'}, 'Only image preprocessing call overrides are allowed')
    require(isinstance(overrides.get('images_kwargs', {}), dict), 'images_kwargs must be an object')
    # Processing options belong here in Transformers >=5.4, not in template kwargs.
    overrides['text_kwargs'] = {'truncation': False}
    return overrides


def library_versions():
    result = {}
    for package in ('tokenizers', 'safetensors', 'huggingface-hub', 'torchvision'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def freeze_snapshot(model_path, model_id, output):
    """Run ON THE SERVER after obtaining weights; hashes weights and processors."""
    root = Path(model_path).resolve()
    require(root.is_dir() and model_id in MODELS, "Expected a local directory for one of the three requested models")
    require(not Path(output).resolve().is_relative_to(root), "Store the snapshot manifest outside the model directory")
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*'))
             if p.is_file() and not any(s.startswith('.') for s in p.relative_to(root).parts)}
    require("config.json" in files and any(p.endswith('.safetensors') for p in files), "Incomplete local HF snapshot")
    config = read_json(root / 'config.json')
    require(config['model_type'] == MODELS[model_id], "Checkpoint architecture does not match requested model family")
    require(not config.get('quantization_config'), 'Primary checkpoints must be unquantized')
    payload = {"schema_version": "physalign_model_snapshot_v1", "model_id": model_id,
               "model_path_hint": str(root), "files": files, "architecture": config['model_type']}
    result = {**payload, "snapshot_hash": fingerprint(payload)}
    write_new(Path(output), result)
    return result


def verify_snapshot(manifest, path=None):
    require(manifest.get('schema_version') == 'physalign_model_snapshot_v1', "Unknown checkpoint manifest")
    require(manifest['snapshot_hash'] == fingerprint({k: v for k, v in manifest.items() if k != 'snapshot_hash'}), "Checkpoint manifest changed")
    root = Path(path or manifest['model_path_hint']).resolve()
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and not any(s.startswith('.') for s in p.relative_to(root).parts)}
    require(actual == set(manifest['files']), 'Checkpoint file inventory changed after freezing')
    for relative, expected in manifest['files'].items():
        # A normal HF cache snapshot uses file symlinks into sibling blobs/.
        # Permit those read-only references, verifying the target BYTES. This
        # exception applies only to model snapshots, never to dataset paths.
        require(isinstance(relative, str) and relative and '\\' not in relative and
                not PurePosixPath(relative).is_absolute() and not PureWindowsPath(relative).drive and
                not any(s in {'.', '..', ''} for s in relative.split('/')), 'Unsafe model snapshot file path')
        require(file_hash(root / relative) == expected, f"Checkpoint file changed: {relative}")
    return root


def transport_messages(request, decode_image):
    """Exactly two turns, image identity followed by its actual image, in order."""
    content = [{"type": "text", "text": request.user}]
    for a in request.images:
        content.extend(({"type": "text", "text": f"\nImage asset ID: {a.asset_id}\n"},
                        {"type": "image", "image": decode_image(a.data)}))
    return [{"role": "system", "content": [{"type": "text", "text": request.system}]},
            {"role": "user", "content": content}]


def encode_request(processor, image_module, request, template_kwargs, processor_call_kwargs=None):
    def decode(data):
        with image_module.open(BytesIO(data)) as image:
            image.load()
            return image.convert('RGB')
    messages = transport_messages(request, decode)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors='pt',
        processor_kwargs=processing_kwargs(processor_call_kwargs), **template_kwargs)
    require(not request.images or ('pixel_values' in inputs and inputs['pixel_values'].numel() > 0),
            'Processor omitted image tensors from a multimodal request')
    image_kwargs = (processor_call_kwargs or {}).get('images_kwargs', {})
    if request.images and image_kwargs.get('max_pixels') == 262144:
        grid = inputs.get('image_grid_thw')
        require(grid is not None and grid.shape[0] == len(request.images), 'Qwen image count/order contract was not preserved')
        require(processor.image_processor.patch_size == 16 and processor.image_processor.merge_size == 2,
                '256-token image profile requires Qwen patch_size=16 and merge_size=2')
        require(all(t == 1 and h * w // 4 <= 256 for t, h, w in grid.tolist()), 'Qwen processor ignored the per-image token cap')
    if request.images and image_kwargs.get('max_patches') == 1:
        require(inputs['pixel_values'].shape[0] == len(request.images), 'InternVL processor ignored the one-tile-per-image cap')
        require(tuple(inputs['pixel_values'].shape[-2:]) == (448, 448), 'InternVL profile requires 448x448 tiles')
        require(processor.image_seq_length == 256, 'InternVL profile requires 256 visual tokens per tile')
    return inputs


def inspect_study_inputs(study_root, config, output, *, progress=None):
    """CPU processor preflight: exact images and tokenization, no model loaded."""
    from PIL import Image
    from transformers import AutoProcessor
    from .study import Study
    from .storage import atomic_json
    study = Study(study_root)
    snapshot = read_json(Path(config['snapshot_manifest']))
    root = verify_snapshot(snapshot, config.get('model_path'))
    require(config.get('thinking') is False and snapshot['model_id'] in MODELS, 'Unsupported preflight model/thinking mode')
    overrides = config.get('processor_kwargs', {})
    require(isinstance(overrides, dict) and not set(overrides) & {'revision', 'trust_remote_code', 'local_files_only'}, 'Invalid processor overrides')
    processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False, **overrides)
    template = {'enable_thinking': False} if MODELS[snapshot['model_id']] == 'qwen3_5' else {}
    max_input = config.get('max_input_tokens', 32768)
    max_new = config.get('max_new_tokens', 2048)
    model_config = read_json(root / 'config.json')
    context = model_config.get('text_config', model_config).get('max_position_embeddings')
    rows = []
    for row in study.plan['requests']:
        req = study.request(row, 'processor-preflight', '{}')
        inputs = encode_request(processor, Image, req, template, config.get('processor_call_kwargs'))
        length = inputs['input_ids'].shape[-1]
        entry = {'instance_id': row['instance_id'], 'condition': row['condition'], 'input_hash': row['input_hash'],
                 'images': len(req.images), 'input_tokens': length,
                 'within_frozen_budget': length <= max_input,
                 'within_native_context': length + max_new <= context if context else None,
                 'tensor_shapes': {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}}
        rows.append(entry)
        if progress:
            progress({'checked': len(rows), 'total': len(study.plan['requests']), 'input_tokens': length})
    report = {'schema_version': 'physalign_processor_preflight_v1', 'plan_hash': study.plan['plan_hash'],
              'model_id': snapshot['model_id'], 'snapshot_hash': snapshot['snapshot_hash'],
              'configuration_hash': fingerprint(config), 'max_input_tokens': max_input, 'max_new_tokens': max_new,
              'native_context': context, 'requests': rows, 'model_loaded': False,
              'all_within_budget': all(r['within_frozen_budget'] and r['within_native_context'] is not False for r in rows),
              'gpu_memory_verified': False}
    atomic_json(Path(output), report)
    return report


class HFAdapter:
    def __init__(self, config):
        allowed = {"snapshot_manifest", "model_path", "gpu_count", "max_memory_gib", "dtype",
                   "max_new_tokens", "max_input_tokens", "thinking", "seed", "processor_kwargs", "processor_call_kwargs"}
        require(set(config) <= allowed and 'snapshot_manifest' in config, "Unknown/missing HF adapter configuration")
        require(config.get('thinking') is False, "This primary protocol requires explicit thinking=false; use a separately frozen ablation for thinking")
        self.snapshot = read_json(Path(config['snapshot_manifest']))
        self.root = verify_snapshot(self.snapshot, config.get('model_path'))
        require(self.snapshot['model_id'] in MODELS, "Unsupported model")
        self.config = dict(config)
        self.processor_call_kwargs = deepcopy(config.get('processor_call_kwargs', {}))
        processing_kwargs(self.processor_call_kwargs)
        import torch
        import transformers
        import accelerate
        from PIL import Image, __version__ as pillow_version
        from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
        self.torch, self.Image, self.GenerationConfig = torch, Image, GenerationConfig
        gpu_count = config.get('gpu_count')
        require(type(gpu_count) is int and gpu_count > 0 and torch.cuda.device_count() == gpu_count,
                "CUDA_VISIBLE_DEVICES must expose exactly gpu_count GPUs; do not use torchrun")
        dtype_name = config.get('dtype', 'bfloat16')
        require(dtype_name in {'bfloat16', 'float16'}, "Use an explicitly recorded 16-bit precision; no implicit quantization")
        if dtype_name == 'bfloat16':
            require(torch.cuda.is_bf16_supported(), "This CUDA runtime does not support BF16")
        memory = config.get('max_memory_gib', 18)
        require(type(memory) is int and 1 <= memory <= 22, "GPU weight memory budget must leave activation headroom")
        self.max_input = config.get('max_input_tokens', 32768)
        self.max_new = config.get('max_new_tokens', 2048)
        require(type(self.max_input) is int and self.max_input > 0 and type(self.max_new) is int and self.max_new > 0, "Invalid token budget")
        model_config = read_json(self.root / 'config.json')
        self.native_context = model_config.get('text_config', model_config).get('max_position_embeddings')
        self.seed = config.get('seed', 2027)
        require(type(self.seed) is int, "Seed must be an integer")
        kwargs = config.get('processor_kwargs', {})
        require(isinstance(kwargs, dict) and not set(kwargs) & {'revision', 'trust_remote_code', 'local_files_only'}, "Invalid processor overrides")
        self.processor = AutoProcessor.from_pretrained(str(self.root), local_files_only=True,
                                                      trust_remote_code=False, **kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(self.root), local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, dtype_name), device_map='balanced',
            max_memory={i: f'{memory}GiB' for i in range(gpu_count)}, attn_implementation='sdpa').eval()
        placement = getattr(self.model, 'hf_device_map', {})
        require(placement and not any(str(v) in {'cpu', 'disk', 'meta'} for v in placement.values()),
                "Model was offloaded outside GPUs; increase the assigned GPU group before freezing a run")
        self.input_device = self.model.get_input_embeddings().weight.device
        require(self.input_device.type == 'cuda', "Model embeddings are not on a GPU")
        self.template_kwargs = {'enable_thinking': False} if MODELS[self.snapshot['model_id']] == 'qwen3_5' else {}
        settings = {"do_sample": False, "num_beams": 1, "max_new_tokens": self.max_new,
                    "max_input_tokens": self.max_input, "seed": self.seed, "thinking": False,
                    "thinking_control": "enable_thinking=false" if self.template_kwargs else "unchanged_system_no_thinking_prompt",
                    "use_cache": True, "repetition_penalty": 1.0, "output_policy": "entire_generated_suffix_no_json_repair"}
        image_processor = getattr(self.processor, 'image_processor', None)
        details = {"backend": "transformers", "dtype": dtype_name, "attention": "sdpa",
                   "device_map": {k: str(v) for k, v in placement.items()}, "max_memory_gib": memory,
                   "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
                   "gpu_total_bytes": [torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)],
                   "cuda_visible_devices": os.environ.get('CUDA_VISIBLE_DEVICES'), "cuda_version": torch.version.cuda,
                   "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                   "accelerate": accelerate.__version__, "pillow": pillow_version,
                   "processor_class": type(self.processor).__name__, "processor_revision": self.snapshot['snapshot_hash'],
                   "tokenizer_class": type(self.processor.tokenizer).__name__ if hasattr(self.processor, 'tokenizer') else None,
                   "additional_library_versions": library_versions(),
                   "processor_overrides": kwargs, "image_processor": image_processor.to_dict() if image_processor else None,
                   "processor_call_kwargs": self.processor_call_kwargs,
                   "native_context_tokens": self.native_context,
                   "chat_template_hash": fingerprint(self.processor.chat_template),
                   "transport_version": "asset_id_before_image_v1", "weight_quantization": None}
        self.info = AdapterInfo('hf-local', self.snapshot['model_id'], 'local-sha256:' + self.snapshot['snapshot_hash'],
                                '1', canonical(settings), canonical(details), True)

    def generate(self, request):
        from .storage import loads
        require(loads(request.settings_json) == self.info.manifest()['settings'], "Request decoding settings differ from loaded adapter")
        torch = self.torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        inputs = None
        try:
            inputs = encode_request(self.processor, self.Image, request, self.template_kwargs, self.processor_call_kwargs)
            n_input = inputs['input_ids'].shape[-1]
            if n_input > self.max_input:
                raise InfrastructureError('frozen_input_token_budget_exceeded', retryable=False)
            if self.native_context and n_input + self.max_new > self.native_context:
                raise InfrastructureError('native_context_budget_exceeded', retryable=False)
            inputs = inputs.to(self.input_device)
            # A fresh GenerationConfig avoids inherited sampling/beam defaults.
            defaults = self.model.generation_config
            generation = self.GenerationConfig(max_new_tokens=self.max_new, do_sample=False, num_beams=1,
                use_cache=True, repetition_penalty=1.0, eos_token_id=defaults.eos_token_id,
                pad_token_id=defaults.pad_token_id, bos_token_id=defaults.bos_token_id)
            with torch.inference_mode():
                output = self.model.generate(**inputs, generation_config=generation)
            suffix = output[0, n_input:]
            token_ids = suffix.tolist()
            text = self.processor.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            raw = self.processor.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            eos = defaults.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            ended = bool(token_ids) and token_ids[-1] in eos
            usage = {"input_tokens": n_input, "output_tokens": len(token_ids), "input_images": len(request.images),
                     "processor_tensor_shapes": {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}}
            return ModelResponse(text, 'stop' if ended else 'length' if len(token_ids) >= self.max_new else 'other',
                                 canonical(usage), self.info.revision,
                                 generation_json=canonical({"token_ids": token_ids, "decoded_with_special_tokens": raw}))
        except torch.cuda.OutOfMemoryError:
            # The nonstreaming call produced no returned output. Never reduce
            # images, context or precision in response to a difficult probe.
            if inputs is not None:
                del inputs
            torch.cuda.empty_cache()
            raise InfrastructureError('cuda_out_of_memory', retryable=False) from None


def create_adapter(config):
    return HFAdapter(config)
