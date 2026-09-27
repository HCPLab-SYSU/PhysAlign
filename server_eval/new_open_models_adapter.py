"""Frozen local-HF adapters for the four additional open-weight models.

The public request plan, prompt strings, image bytes, generation policy and
token budgets match the existing PhysAlign local-model protocol.  Only
checkpoint-required loading/chat/image-processing differences live here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
import os
import platform

from physalign.adapters import AdapterInfo, InfrastructureError, ModelResponse
from physalign.dataset import require
from physalign.hf_adapter import verify_snapshot as verify_snapshot_files
from physalign.storage import canonical, file_hash, fingerprint, loads, read_json, write_new


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    architecture: str
    architecture_class: str
    loader: str
    trust_remote_code: bool
    gpu_count: int
    attention: str
    thinking_control: str
    system_transport: str
    visual_profile: str


SPECS = {
    "gemma4-26b-a4b": ModelSpec(
        "gemma4-26b-a4b", "google/gemma-4-26B-A4B-it", "gemma4",
        "Gemma4ForConditionalGeneration", "AutoModelForMultimodalLM", False, 4, "sdpa",
        "enable_thinking=false", "native_system_user",
        "gemma4-max-280-soft-tokens-per-image",
    ),
    "glm46v-flash": ModelSpec(
        "glm46v-flash", "zai-org/GLM-4.6V-Flash", "glm4v",
        "Glm4vForConditionalGeneration", "AutoModelForMultimodalLM", False, 2, "sdpa",
        "enable_thinking=false", "native_system_user",
        "glm46v-64-to-256-merged-tokens-per-image",
    ),
    "molmo2-8b": ModelSpec(
        "molmo2-8b", "allenai/Molmo2-8B", "molmo2",
        "Molmo2ForConditionalGeneration", "AutoModelForImageTextToText", True, 2, "sdpa",
        "checkpoint-has-no-thinking-switch", "system_and_user_joined_by_two_newlines",
        "molmo2-one-local-crop-plus-global-crop-per-image",
    ),
    "kimi-vl-a3b": ModelSpec(
        "kimi-vl-a3b", "moonshotai/Kimi-VL-A3B-Instruct", "kimi_vl",
        "KimiVLForConditionalGeneration", "AutoModel", True, 2, "flash_attention_2",
        "checkpoint-has-no-thinking-switch", "native_system_user",
        "kimi-1024-premerge-patch-cap-per-image",
    ),
}
BY_MODEL_ID = {spec.model_id: spec for spec in SPECS.values()}


def spec_for(value: str | ModelSpec) -> ModelSpec:
    if isinstance(value, ModelSpec):
        require(value in SPECS.values(), "Unregistered model specification")
        return value
    spec = SPECS.get(value) or BY_MODEL_ID.get(value)
    require(spec is not None, "Unknown new open model: " + str(value))
    return spec


def expected_processor_options(spec: ModelSpec) -> tuple[dict, dict]:
    """Freeze comparable bounded-resolution profiles before any outcomes exist."""
    if spec.key == "gemma4-26b-a4b":
        return {}, {"images_kwargs": {"max_soft_tokens": 280}}
    if spec.key == "glm46v-flash":
        # GLM counts two temporal copies in its pixel-area bounds.  These
        # values therefore correspond to 64..256 merged visual tokens.
        return {}, {"images_kwargs": {"size": {
            "shortest_edge": 100352, "longest_edge": 401408,
        }}}
    if spec.key == "molmo2-8b":
        return {}, {"images_kwargs": {"max_crops": 1}}
    # Kimi caps raw 14x14 patches before its 2x2 merge.  Padding to even grids
    # is checkpoint-native, so the exact encoded length is audited preflight.
    return {"in_token_limit": 1024}, {}


def validate_config(config: dict, expected: str | ModelSpec | None = None) -> tuple[ModelSpec, dict]:
    allowed = {"snapshot_manifest", "model_path", "gpu_count", "max_memory_gib", "dtype",
               "max_new_tokens", "max_input_tokens", "thinking", "seed", "processor_kwargs",
               "processor_call_kwargs", "attn_implementation"}
    require(isinstance(config, dict) and set(config) <= allowed and "snapshot_manifest" in config,
            "Unknown/missing new-model adapter configuration")
    snapshot = read_json(Path(config["snapshot_manifest"]))
    spec = spec_for(snapshot.get("model_id"))
    if expected is not None:
        wanted = expected if isinstance(expected, ModelSpec) else spec_for(expected)
        require(spec == wanted, f"Configuration snapshot is not {wanted.model_id}")
    require(snapshot.get("architecture") == spec.architecture and
            snapshot.get("architecture_class") == spec.architecture_class and
            snapshot.get("trust_remote_code") is spec.trust_remote_code,
            "Snapshot identity does not match the frozen model registry")
    require(config.get("gpu_count") == spec.gpu_count,
            f"{spec.model_id} requires the frozen {spec.gpu_count}-GPU assignment")
    require(config.get("max_memory_gib", 18) == 18, "Frozen RTX3090 weight placement is 18 GiB per GPU")
    require(config.get("dtype", "bfloat16") == "bfloat16", "Primary protocol requires BF16")
    require(config.get("attn_implementation") == spec.attention,
            f"{spec.model_id} requires attn_implementation={spec.attention}")
    require(config.get("thinking") is False, "Primary protocol requires explicit thinking=false")
    require(config.get("seed", 2027) == 2027, "Primary protocol seed must remain 2027")
    require(config.get("max_input_tokens", 32768) == 32768, "Primary input budget must remain 32768")
    require(config.get("max_new_tokens", 2048) == 2048, "Primary output budget must remain 2048")
    processor_kwargs, call_kwargs = expected_processor_options(spec)
    require(config.get("processor_kwargs", {}) == processor_kwargs,
            "Processor construction options differ from the frozen visual profile")
    require(config.get("processor_call_kwargs", {}) == call_kwargs,
            "Processor call options differ from the frozen visual profile")
    return spec, snapshot


def freeze_snapshot(model_path: str | Path, model: str, output: str | Path) -> dict:
    """Hash the complete local checkpoint, including any trusted custom code."""
    spec = spec_for(model)
    root = Path(model_path).resolve()
    output = Path(output).resolve()
    require(root.is_dir(), "Expected a complete local Hugging Face checkpoint directory")
    require(not output.is_relative_to(root), "Store the snapshot manifest outside the model directory")
    files = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(root.rglob("*"))
             if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)}
    require("config.json" in files and any(name.endswith(".safetensors") for name in files),
            "Incomplete local HF snapshot")
    require("tokenizer_config.json" in files and
            ({"processor_config.json", "preprocessor_config.json"} & set(files)),
            "Checkpoint is missing tokenizer/processor configuration")
    tokenizer_config = read_json(root / "tokenizer_config.json")
    require("chat_template.jinja" in files or isinstance(tokenizer_config.get("chat_template"), str),
            "Checkpoint is missing its chat template")
    checkpoint = read_json(root / "config.json")
    require(checkpoint.get("model_type") == spec.architecture,
            f"Checkpoint model_type must be {spec.architecture}")
    require(spec.architecture_class in (checkpoint.get("architectures") or []),
            f"Checkpoint architectures must include {spec.architecture_class}")
    require(not checkpoint.get("quantization_config"), "Primary checkpoints must be unquantized")
    if spec.trust_remote_code:
        processor_metadata = {}
        for name in ("processor_config.json", "preprocessor_config.json"):
            if name in files:
                processor_metadata.update(read_json(root / name))
        model_map = checkpoint.get("auto_map") if isinstance(checkpoint.get("auto_map"), dict) else {}
        processor_map = (processor_metadata.get("auto_map")
                         if isinstance(processor_metadata.get("auto_map"), dict) else {})

        def mapped_code_exists(target) -> bool:
            if not isinstance(target, str) or "." not in target:
                return False
            module = target.rsplit("--", 1)[-1].rsplit(".", 1)[0].replace(".", "/") + ".py"
            return module in files

        require(mapped_code_exists(model_map.get(spec.loader)) and
                mapped_code_exists(processor_map.get("AutoProcessor")),
                "Custom-code checkpoint is missing its frozen model/processor Python implementation")
    payload = {
        "schema_version": "physalign_model_snapshot_v1",
        "model_id": spec.model_id,
        "model_path_hint": str(root),
        "files": files,
        "architecture": spec.architecture,
        "architecture_class": spec.architecture_class,
        "trust_remote_code": spec.trust_remote_code,
    }
    result = {**payload, "snapshot_hash": fingerprint(payload)}
    write_new(output, result)
    return result


def verified_model_root(config: dict, expected: str | ModelSpec | None = None) -> tuple[ModelSpec, dict, Path]:
    spec, snapshot = validate_config(config, expected)
    root = verify_snapshot_files(snapshot, config.get("model_path"))
    return spec, snapshot, root


def processor_call_kwargs(value: dict | None) -> dict:
    value = deepcopy(value or {})
    require(isinstance(value, dict) and set(value) <= {"images_kwargs"},
            "Only image preprocessing call overrides are allowed")
    require(isinstance(value.get("images_kwargs", {}), dict), "images_kwargs must be an object")
    value["text_kwargs"] = {"truncation": False}
    return value


def load_processor(root: Path, spec: ModelSpec, config: dict):
    from transformers import AutoProcessor
    kwargs = deepcopy(config.get("processor_kwargs", {}))
    require(isinstance(kwargs, dict) and not set(kwargs) & {"revision", "trust_remote_code", "local_files_only"},
            "Invalid processor overrides")
    processor = AutoProcessor.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=spec.trust_remote_code, **kwargs
    )
    if spec.key == "kimi-vl-a3b":
        require(getattr(processor.image_processor, "in_token_limit", None) == 1024,
                "Kimi processor did not apply the frozen 1024-patch input cap")
    return processor


def template_kwargs(spec: ModelSpec) -> dict:
    return {"enable_thinking": False} if spec.key in {"gemma4-26b-a4b", "glm46v-flash"} else {}


def transport_messages(spec: ModelSpec, request, decode_image) -> list[dict]:
    if spec.system_transport == "native_system_user":
        messages = [{"role": "system", "content": [{"type": "text", "text": request.system}]}]
        first_text = request.user
    else:
        # Molmo2's official template rejects a system turn and requires strict
        # user/assistant alternation.  Preserve both strings without inventing
        # new wording; only the role boundary becomes a fixed blank line.
        messages = []
        first_text = request.system + "\n\n" + request.user
    content = [{"type": "text", "text": first_text}]
    for asset in request.images:
        content.extend((
            {"type": "text", "text": f"\nImage asset ID: {asset.asset_id}\n"},
            {"type": "image", "image": decode_image(asset.data)},
        ))
    messages.append({"role": "user", "content": content})
    return messages


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def validate_visual_encoding(spec: ModelSpec, processor, inputs, image_count: int) -> None:
    if not image_count:
        return
    require("pixel_values" in inputs and inputs["pixel_values"].numel() > 0,
            "Processor omitted image tensors from a multimodal request")
    image_processor = processor.image_processor
    if spec.key == "gemma4-26b-a4b":
        require(inputs["pixel_values"].shape[0] == image_count,
                "Gemma processor changed image count/order")
        token_types = inputs.get("mm_token_type_ids")
        if token_types is not None:
            require(int(_scalar(token_types.sum())) <= 280 * image_count,
                    "Gemma encoded more than 280 visual tokens per image")
    elif spec.key == "glm46v-flash":
        grid = inputs.get("image_grid_thw")
        require(grid is not None and grid.shape[0] == image_count,
                "GLM processor changed image count/order")
        require(getattr(image_processor, "patch_size", None) == 14 and
                getattr(image_processor, "merge_size", None) == 2,
                "GLM visual profile requires patch_size=14 and merge_size=2")
        counts = [int(t * h * w // 4) for t, h, w in grid.tolist()]
        require(all(64 <= count <= 256 for count in counts),
                "GLM processor ignored the frozen 64..256 visual-token bounds")
    elif spec.key == "molmo2-8b":
        crops, grids = inputs.get("image_num_crops"), inputs.get("image_grids")
        require(crops is not None and grids is not None and len(crops) == image_count and grids.shape[0] == image_count,
                "Molmo2 processor changed image count/order")
        # max_crops=1 produces one local crop plus the checkpoint's mandatory
        # global resized crop.
        crop_counts = [int(value) for value in crops.tolist()]
        require(all(value == 2 for value in crop_counts) and inputs["pixel_values"].shape[0] == sum(crop_counts),
                "Molmo2 processor ignored max_crops=1")
    else:
        grids = inputs.get("image_grid_hws")
        require(grids is not None and grids.shape[0] == image_count,
                "Kimi processor changed image count/order")
        require(getattr(image_processor, "in_token_limit", None) == 1024 and
                list(getattr(image_processor, "merge_kernel_size", [])) == [2, 2],
                "Kimi visual profile requires 1024 raw patches and 2x2 merging")
        require(inputs["pixel_values"].shape[0] == sum(int(h * w) for h, w in grids.tolist()),
                "Kimi patch tensor does not match its per-image grids")


def encode_request(processor, image_module, spec: ModelSpec, request, call_kwargs: dict | None = None):
    def decode(data: bytes):
        with image_module.open(BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")

    messages = transport_messages(spec, request, decode)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_call_kwargs(call_kwargs),
        **template_kwargs(spec),
    )
    require("input_ids" in inputs and len(inputs["input_ids"].shape) == 2 and inputs["input_ids"].shape[0] == 1,
            "Processor must return one untruncated token sequence")
    validate_visual_encoding(spec, processor, inputs, len(request.images))
    if spec.key == "glm46v-flash":
        # Required by the official GLM Transformers path; use the exact same
        # tensor set during CPU preflight and GPU generation.
        inputs.pop("token_type_ids", None)
    return inputs


def native_context(root: Path) -> int | None:
    config = read_json(root / "config.json")
    return config.get("text_config", config).get("max_position_embeddings")


def library_versions() -> dict:
    result = {}
    for package in ("tokenizers", "safetensors", "huggingface-hub", "torchvision", "tiktoken", "einops",
                    "flash-attn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def verify_gpu_residency(model, gpu_count: int, model_id: str) -> dict:
    placement = getattr(model, "hf_device_map", None)
    require(isinstance(placement, dict) and placement, "Accelerate did not report the model device map")
    allowed = {str(index) for index in range(gpu_count)} | {f"cuda:{index}" for index in range(gpu_count)}
    require(all(str(device) in allowed for device in placement.values()),
            "Model device map contains CPU/disk/meta or an unassigned GPU: " +
            canonical({name: str(device) for name, device in placement.items()}))
    counts, bad, used, parameter_count = Counter(), [], set(), 0
    for name, parameter in model.named_parameters():
        parameter_count += parameter.numel()
        device = parameter.device
        label = f"cuda:{device.index}" if device.type == "cuda" else device.type
        counts[label] += 1
        if device.type == "cuda" and device.index in range(gpu_count):
            used.add(device.index)
        elif len(bad) < 5:
            bad.append(f"{name}={label}")
    require(parameter_count > 0 and not bad, "Model parameters are outside assigned GPUs: " + "; ".join(bad))
    require(used == set(range(gpu_count)), "Balanced placement did not use every assigned GPU")
    details = {"parameter_tensor_devices": dict(sorted(counts.items())), "parameter_count": parameter_count,
               "reported_device_map": {name: str(device) for name, device in placement.items()}}
    print(canonical({"stage": "gpu_residency_check", "model": model_id, **details}), flush=True)
    return details


def _move_inputs(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}


class NewOpenModelAdapter:
    def __init__(self, config: dict, expected: str | ModelSpec | None = None):
        self.spec, self.snapshot, self.root = verified_model_root(config, expected)
        self.config = deepcopy(config)
        self.processor_call_kwargs = deepcopy(config.get("processor_call_kwargs", {}))
        processor_call_kwargs(self.processor_call_kwargs)

        import accelerate
        import torch
        import transformers
        from PIL import Image, __version__ as pillow_version
        from transformers import GenerationConfig

        self.torch, self.Image, self.GenerationConfig = torch, Image, GenerationConfig
        gpu_count = config["gpu_count"]
        require(torch.cuda.device_count() == gpu_count,
                "CUDA_VISIBLE_DEVICES must expose exactly gpu_count GPUs; do not use torchrun")
        require(torch.cuda.is_bf16_supported(), "This CUDA runtime does not support BF16")
        self.max_input, self.max_new, self.seed = 32768, 2048, 2027
        self.native_context = native_context(self.root)
        self.processor = load_processor(self.root, self.spec, config)

        model_class = getattr(transformers, self.spec.loader)
        self.model = model_class.from_pretrained(
            str(self.root),
            local_files_only=True,
            trust_remote_code=self.spec.trust_remote_code,
            dtype=torch.bfloat16,
            device_map="balanced",
            max_memory={index: "18GiB" for index in range(gpu_count)},
            attn_implementation=self.spec.attention,
        ).eval()
        self.gpu_residency = verify_gpu_residency(self.model, gpu_count, self.spec.model_id)
        embeddings = self.model.get_input_embeddings()
        self.input_device = embeddings.weight.device
        require(self.input_device.type == "cuda", "Model input embeddings are not on a GPU")

        settings = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": self.max_new,
            "max_input_tokens": self.max_input,
            "seed": self.seed,
            "thinking": False,
            "thinking_control": self.spec.thinking_control,
            "use_cache": True,
            "repetition_penalty": 1.0,
            "output_policy": "entire_generated_suffix_no_json_repair",
        }
        image_processor = getattr(self.processor, "image_processor", None)
        custom_code = {name: digest for name, digest in self.snapshot["files"].items() if name.endswith(".py")}
        details = {
            "backend": "transformers",
            "dtype": "bfloat16",
            "attention": self.spec.attention,
            "device_map": self.gpu_residency["reported_device_map"],
            "gpu_residency_audit": self.gpu_residency,
            "max_memory_gib": 18,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
            "gpu_total_bytes": [torch.cuda.get_device_properties(index).total_memory for index in range(gpu_count)],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_version": torch.version.cuda,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "pillow": pillow_version,
            "additional_library_versions": library_versions(),
            "model_loader": self.spec.loader,
            "processor_class": type(self.processor).__name__,
            "tokenizer_class": type(self.processor.tokenizer).__name__ if hasattr(self.processor, "tokenizer") else None,
            "processor_revision": self.snapshot["snapshot_hash"],
            "processor_overrides": deepcopy(config.get("processor_kwargs", {})),
            "processor_call_kwargs": self.processor_call_kwargs,
            "image_processor": image_processor.to_dict() if image_processor and hasattr(image_processor, "to_dict") else None,
            "visual_profile": self.spec.visual_profile,
            "system_transport": self.spec.system_transport,
            "transport_version": "asset_id_before_image_v1",
            "native_context_tokens": self.native_context,
            "chat_template_hash": fingerprint(self.processor.chat_template),
            "trust_remote_code": self.spec.trust_remote_code,
            "frozen_custom_code_sha256": custom_code,
            "weight_quantization": None,
        }
        self.info = AdapterInfo(
            "hf-local-open-panel", self.spec.model_id,
            "local-sha256:" + self.snapshot["snapshot_hash"], "1",
            canonical(settings), canonical(details), True,
        )

    def generate(self, request):
        require(loads(request.settings_json) == self.info.manifest()["settings"],
                "Request decoding settings differ from loaded adapter")
        torch = self.torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        inputs = None
        try:
            inputs = encode_request(
                self.processor, self.Image, self.spec, request, self.processor_call_kwargs
            )
            n_input = inputs["input_ids"].shape[-1]
            if n_input > self.max_input:
                raise InfrastructureError("frozen_input_token_budget_exceeded", retryable=False)
            if self.native_context and n_input + self.max_new > self.native_context:
                raise InfrastructureError("native_context_budget_exceeded", retryable=False)
            inputs = _move_inputs(inputs, self.input_device)
            defaults = self.model.generation_config
            generation = self.GenerationConfig(
                max_new_tokens=self.max_new,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                repetition_penalty=1.0,
                eos_token_id=defaults.eos_token_id,
                pad_token_id=defaults.pad_token_id,
                bos_token_id=defaults.bos_token_id,
            )
            with torch.inference_mode():
                output = self.model.generate(**inputs, generation_config=generation)
            if hasattr(output, "sequences"):
                output = output.sequences
            suffix = output[0, n_input:]
            token_ids = suffix.tolist()
            text = self.processor.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            raw = self.processor.decode(suffix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            eos = defaults.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            ended = bool(token_ids) and token_ids[-1] in eos
            usage = {
                "input_tokens": n_input,
                "output_tokens": len(token_ids),
                "input_images": len(request.images),
                "processor_tensor_shapes": {
                    name: list(value.shape) for name, value in inputs.items() if hasattr(value, "shape")
                },
            }
            finish = "stop" if ended else "length" if len(token_ids) >= self.max_new else "other"
            return ModelResponse(
                text, finish, canonical(usage), self.info.revision,
                generation_json=canonical({
                    "token_ids": token_ids,
                    "decoded_with_special_tokens": raw,
                }),
            )
        except torch.cuda.OutOfMemoryError:
            if inputs is not None:
                del inputs
            torch.cuda.empty_cache()
            raise InfrastructureError("cuda_out_of_memory", retryable=False) from None


def create_adapter(config: dict):
    return NewOpenModelAdapter(config)
