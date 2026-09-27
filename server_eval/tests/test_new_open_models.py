"""CPU-only contract tests for the four-model extension."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from physalign.storage import atomic_json, read_json
from server_eval.new_open_models_adapter import (
    SPECS,
    expected_processor_options,
    freeze_snapshot,
    processor_call_kwargs,
    spec_for,
    template_kwargs,
    transport_messages,
    validate_config,
    validate_visual_encoding,
)


class Tensor:
    def __init__(self, values=None, shape=(1,), scalar=None):
        self.values, self.shape, self.scalar = values, shape, scalar

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def tolist(self):
        return self.values

    def __len__(self):
        return self.shape[0]

    def sum(self):
        return SimpleNamespace(item=lambda: self.scalar)


class NewOpenModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def checkpoint(self, key):
        spec = SPECS[key]
        root = self.root / key
        root.mkdir()
        config = {"model_type": spec.architecture, "architectures": [spec.architecture_class]}
        if spec.trust_remote_code:
            config["auto_map"] = {spec.loader: "modeling_fixture.FixtureModel"}
            (root / "modeling_fixture.py").write_text("class FixtureModel: pass\n", encoding="utf-8")
        atomic_json(root / "config.json", config)
        atomic_json(root / "tokenizer_config.json", {"chat_template": "{{ messages }}"})
        preprocessor = {"processor_class": "FixtureProcessor"}
        if spec.trust_remote_code:
            preprocessor["auto_map"] = {"AutoProcessor": "processing_fixture.FixtureProcessor"}
            (root / "processing_fixture.py").write_text("class FixtureProcessor: pass\n", encoding="utf-8")
        atomic_json(root / "preprocessor_config.json", preprocessor)
        (root / "model.safetensors").write_bytes(b"fixture")
        return root

    def config(self, key, manifest):
        spec = SPECS[key]
        processor, call = expected_processor_options(spec)
        return {
            "snapshot_manifest": str(manifest), "gpu_count": spec.gpu_count,
            "max_memory_gib": 18, "dtype": "bfloat16", "thinking": False,
            "attn_implementation": spec.attention,
            "seed": 2027, "max_input_tokens": 32768, "max_new_tokens": 2048,
            "processor_kwargs": processor, "processor_call_kwargs": call,
        }

    def test_registry_has_exact_requested_model_ids_and_gpu_groups(self):
        self.assertEqual(
            {spec.model_id for spec in SPECS.values()},
            {"google/gemma-4-26B-A4B-it", "zai-org/GLM-4.6V-Flash",
             "allenai/Molmo2-8B", "moonshotai/Kimi-VL-A3B-Instruct"},
        )
        self.assertEqual([SPECS[key].gpu_count for key in SPECS], [4, 2, 2, 2])
        self.assertFalse(SPECS["gemma4-26b-a4b"].trust_remote_code)
        self.assertFalse(SPECS["glm46v-flash"].trust_remote_code)
        self.assertTrue(SPECS["molmo2-8b"].trust_remote_code)
        self.assertTrue(SPECS["kimi-vl-a3b"].trust_remote_code)
        self.assertEqual([SPECS[key].attention for key in SPECS],
                         ["sdpa", "sdpa", "sdpa", "flash_attention_2"])

    def test_freeze_and_config_validation_are_exact_for_all_four_models(self):
        for key, spec in SPECS.items():
            with self.subTest(model=key):
                output = self.root / (key + ".snapshot.json")
                result = freeze_snapshot(self.checkpoint(key), key, output)
                self.assertEqual(result["model_id"], spec.model_id)
                self.assertEqual(result["architecture"], spec.architecture)
                self.assertIs(result["trust_remote_code"], spec.trust_remote_code)
                observed, snapshot = validate_config(self.config(key, output), key)
                self.assertEqual(observed, spec)
                self.assertEqual(snapshot, read_json(output))

    def test_wrong_architecture_quantization_and_missing_custom_code_are_rejected(self):
        root = self.checkpoint("molmo2-8b")
        (root / "modeling_fixture.py").unlink()
        with self.assertRaisesRegex(ValueError, "Python"):
            freeze_snapshot(root, "molmo2-8b", self.root / "missing-code.json")
        root = self.checkpoint("glm46v-flash")
        config = read_json(root / "config.json")
        config["quantization_config"] = {"bits": 4}
        atomic_json(root / "config.json", config)
        with self.assertRaisesRegex(ValueError, "unquantized"):
            freeze_snapshot(root, "glm46v-flash", self.root / "quantized.json")
        with self.assertRaisesRegex(ValueError, "model_type"):
            freeze_snapshot(root, "gemma4-26b-a4b", self.root / "wrong.json")

    def test_primary_decoding_and_visual_options_cannot_drift(self):
        root = self.checkpoint("gemma4-26b-a4b")
        manifest = self.root / "gemma.json"
        freeze_snapshot(root, "gemma4-26b-a4b", manifest)
        config = self.config("gemma4-26b-a4b", manifest)
        for key, value, message in (
            ("thinking", True, "thinking=false"),
            ("dtype", "float16", "BF16"),
            ("attn_implementation", "eager", "attn_implementation=sdpa"),
            ("seed", 1, "2027"),
            ("max_new_tokens", 512, "2048"),
            ("gpu_count", 2, "4-GPU"),
        ):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, message):
                validate_config({**config, key: value})
        changed = {**config, "processor_call_kwargs": {}}
        with self.assertRaisesRegex(ValueError, "visual profile"):
            validate_config(changed)

    def test_molmo_folds_unsupported_system_role_without_rewriting_text(self):
        request = SimpleNamespace(
            system="SYSTEM BYTES", user="USER BYTES",
            images=(SimpleNamespace(asset_id="asset-1", data=b"one"),
                    SimpleNamespace(asset_id="asset-2", data=b"two")),
        )
        messages = transport_messages(spec_for("molmo2-8b"), request, lambda data: data.decode())
        self.assertEqual([message["role"] for message in messages], ["user"])
        content = messages[0]["content"]
        self.assertEqual(content[0]["text"], "SYSTEM BYTES\n\nUSER BYTES")
        self.assertEqual([item["image"] for item in content if item["type"] == "image"], ["one", "two"])
        gemma = transport_messages(spec_for("gemma4-26b-a4b"), request, lambda data: data.decode())
        self.assertEqual([message["role"] for message in gemma], ["system", "user"])
        self.assertEqual(gemma[0]["content"][0]["text"], "SYSTEM BYTES")

    def test_template_and_processor_call_controls_are_explicit(self):
        self.assertEqual(template_kwargs(spec_for("gemma4-26b-a4b")), {"enable_thinking": False})
        self.assertEqual(template_kwargs(spec_for("glm46v-flash")), {"enable_thinking": False})
        self.assertEqual(template_kwargs(spec_for("molmo2-8b")), {})
        value = processor_call_kwargs({"images_kwargs": {"max_crops": 1}})
        self.assertEqual(value["text_kwargs"], {"truncation": False})
        with self.assertRaisesRegex(ValueError, "Only image"):
            processor_call_kwargs({"text_kwargs": {"truncation": True}})

    def test_each_visual_profile_checks_real_processor_outputs(self):
        pixel = Tensor(shape=(2, 3, 4))
        gemma_processor = SimpleNamespace(image_processor=SimpleNamespace(max_soft_tokens=280))
        validate_visual_encoding(SPECS["gemma4-26b-a4b"], gemma_processor,
                                 {"pixel_values": pixel, "mm_token_type_ids": Tensor(scalar=540)}, 2)

        glm_processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=14, merge_size=2))
        validate_visual_encoding(SPECS["glm46v-flash"], glm_processor,
                                 {"pixel_values": pixel,
                                  "image_grid_thw": Tensor([[1, 16, 16], [1, 32, 32]], (2, 3))}, 2)

        molmo_processor = SimpleNamespace(image_processor=SimpleNamespace())
        validate_visual_encoding(SPECS["molmo2-8b"], molmo_processor,
                                 {"pixel_values": Tensor(shape=(4, 3)),
                                  "image_num_crops": Tensor([2, 2], (2,)),
                                  "image_grids": Tensor(shape=(2, 4))}, 2)

        kimi_processor = SimpleNamespace(image_processor=SimpleNamespace(
            in_token_limit=1024, merge_kernel_size=[2, 2]))
        validate_visual_encoding(SPECS["kimi-vl-a3b"], kimi_processor,
                                 {"pixel_values": Tensor(shape=(2048, 3)),
                                  "image_grid_hws": Tensor([[32, 32], [16, 64]], (2, 2))}, 2)


if __name__ == "__main__":
    unittest.main()
