"""Register Qwen3.5-4B in a private instance of the existing HF backend.

The backend source and its original model registry remain unchanged. Thus old
run hashes and resumes keep working; the companion's module hash identifies
the new model registration in each 4B run manifest.
"""
from dataclasses import replace
from copy import deepcopy
from collections import Counter
import importlib.util
from pathlib import Path
import sys

from physalign import hf_adapter as original
from physalign.dataset import require
from physalign.storage import canonical, file_hash, loads, read_json

MODEL_ID = 'Qwen/Qwen3.5-4B'
ARCHITECTURE = 'qwen3_5'


def _load_backend():
    # A separate module namespace keeps even the original in-memory registry intact.
    name = 'physalign._qwen35_4b_companion_backend'
    spec = importlib.util.spec_from_file_location(name, original.__file__)
    require(spec is not None and spec.loader is not None, 'Cannot load the existing HF backend')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.MODELS = {**module.MODELS, MODEL_ID: ARCHITECTURE}
    return module


backend = _load_backend()


def validate_snapshot(config):
    snapshot = read_json(Path(config['snapshot_manifest']))
    require(snapshot.get('model_id') == MODEL_ID and snapshot.get('architecture') == ARCHITECTURE,
            'The 4B companion requires an exact Qwen/Qwen3.5-4B qwen3_5 snapshot')
    return snapshot


def thinking_enabled(config):
    value = config.get('thinking', False)
    require(type(value) is bool, 'thinking must be a boolean')
    return value


def verify_gpu_residency(model, gpu_count):
    """Audit real tensors, including single-device loads without hf_device_map.

    Transformers 5.17 skips Accelerate dispatch for a one-device map. In that
    path hf_device_map may be absent even though all weights are on CUDA.
    Never infer offload (or residency) from that metadata's absence alone.
    """
    devices, bad, parameter_count = {}, [], 0
    cuda_indices = set()
    for kind, tensors in (('parameters', model.named_parameters()), ('buffers', model.named_buffers())):
        counts = Counter()
        for name, tensor in tensors:
            if kind == 'parameters':
                parameter_count += tensor.numel()
            device = tensor.device
            label = f'cuda:{device.index}' if device.type == 'cuda' else device.type
            counts[label] += 1
            if device.type != 'cuda' or device.index not in range(gpu_count):
                if len(bad) < 5:
                    bad.append(f'{kind}:{name}={label}')
            else:
                cuda_indices.add(device.index)
        devices[kind] = dict(sorted(counts.items()))
    placement = getattr(model, 'hf_device_map', None)
    details = {'tensor_devices': devices, 'parameter_count': parameter_count,
               'device_map_source': 'reported' if placement else 'inferred_from_actual_single_cuda_device'}
    print(canonical({'stage': 'gpu_residency_check', 'model': MODEL_ID, **details}), flush=True)
    require(parameter_count > 0, 'Cannot verify GPU residency: model has no parameters')
    require(not bad, 'Model tensors are outside the assigned CUDA devices: ' + '; '.join(bad))
    if placement:
        require(isinstance(placement, dict), 'Invalid model hf_device_map metadata')
        allowed = {str(index) for index in range(gpu_count)} | {f'cuda:{index}' for index in range(gpu_count)}
        require(all(str(value) in allowed for value in placement.values()),
                'Model device map contains CPU/disk/meta or an unassigned device: ' +
                canonical({key: str(value) for key, value in placement.items()}))
    else:
        require(len(cuda_indices) == 1, 'Missing device map for a multi-device model; cannot infer one root device')
        # Supply compatibility metadata only AFTER checking every parameter and
        # buffer. The original frozen backend can then perform its own checks.
        model.hf_device_map = {'': next(iter(cuda_indices))}
    return details


class Qwen35_4BAdapter(backend.HFAdapter):
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, loaded):
        # The parent's assignment runs before its metadata-only residency guard.
        # This local hook avoids editing the core used by completed 9B/27B runs.
        self._gpu_residency = verify_gpu_residency(loaded, self.config['gpu_count'])
        self._model = loaded

    def __init__(self, config):
        validate_snapshot(config)
        self.thinking = thinking_enabled(config)
        effective = {**config, 'thinking': self.thinking,
                     'max_new_tokens': config.get('max_new_tokens', 32768 if self.thinking else 2048)}
        # Reuse the frozen loader/generator without changing old run hashes.
        # Its initial false setting is replaced before any request is processed.
        super().__init__({**effective, 'thinking': False})
        self.config = deepcopy(effective)
        self.template_kwargs = {'enable_thinking': self.thinking}
        settings = loads(self.info.settings_json)
        settings.update(thinking=self.thinking, thinking_control=f'enable_thinking={str(self.thinking).lower()}',
                        output_policy=('final_suffix_after_first_think_end_token_no_json_repair' if self.thinking
                                       else 'entire_generated_suffix_no_json_repair'),
                        output_budget_scope='reasoning_and_final_combined')
        if self.thinking:
            marker = self.processor.tokenizer.encode('</think>', add_special_tokens=False)
            require(len(marker) == 1 and marker[0] != self.processor.tokenizer.unk_token_id,
                    'The frozen Qwen tokenizer must define an atomic </think> token')
            self.think_end_id = marker[0]
            settings['think_end_token_id'] = self.think_end_id
        details = loads(self.info.preprocessing_json)
        details.update(model_registration='qwen35_4b_companion_v3_gpu_residency',
                       gpu_residency_audit=self._gpu_residency,
                       backend_source_sha256=file_hash(Path(original.__file__)))
        self.info = replace(self.info, implementation_version='3', settings_json=canonical(settings),
                            preprocessing_json=canonical(details))

    def generate(self, request):
        response = super().generate(request)
        if not self.thinking:
            return response
        trace = loads(response.generation_json)
        ids = trace['token_ids']
        # The official thinking prompt ends in <think>\n. Only its first atomic
        # closing token starts the final channel. Do not search for or repair JSON.
        closed = self.think_end_id in ids
        end = ids.index(self.think_end_id) if closed else len(ids)
        final_ids = ids[end + 1:] if closed else []
        final = self.processor.decode(final_ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False) if closed else ''
        reasoning = self.processor.decode(ids[:end], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
        trace.update(decoded_entire_suffix=response.text, reasoning_content=reasoning,
                     reasoning_closed=closed, final_token_offset=end + 1 if closed else None,
                     final_text=final)
        usage = loads(response.usage_json)
        usage.update(reasoning_tokens=end + int(closed), final_tokens=len(final_ids))
        # An unfinished thought is an empty completed output, scored as invalid;
        # it is not an infrastructure failure and must not trigger another try.
        return replace(response, text=final, usage_json=canonical(usage), generation_json=canonical(trace))


def create_adapter(config):
    return Qwen35_4BAdapter(config)


def freeze_snapshot(model_path, output):
    return backend.freeze_snapshot(model_path, MODEL_ID, output)
