from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_visual_feedback import render_pass1_overlays  # noqa: E402
from physgraph_api_client import APIResult, load_pass1_system_instructions  # noqa: E402
from physgraph_annotation_lib import json_sha256, read_json, read_jsonl, write_json_atomic  # noqa: E402
from physgraph_pixel_geometry import (  # noqa: E402
    PanelSpec,
    build_problem_panels,
    detect_horizontal_panels,
    merge_panel_documents,
    normalize_panel_document,
    normalize_pixel_correction_patch,
    panel_contract,
    panel_view_from_standard,
    pixel_node_to_standard,
    render_panel_assets,
    validate_panel_document,
)
from run_high_confidence_physgraph_closed_loop import (  # noqa: E402
    allocate_append_ids,
    apply_correction_patch,
    correct_pass1_by_panels,
    correct_pass1,
    discover_latest_pixel_checkpoint,
    discover_implicit_accept_pixel_patch_artifact,
    discover_valid_panel_generation_artifact,
    discover_valid_downstream_artifact,
    discover_valid_downstream_prefix,
    evaluate_pixel_correction_patch,
    force_verified_pass1_into_pass4,
    generate_pass4_from_prefix,
    generate_pass1_by_panels,
    load_pass1_cache,
    load_pixel_correction_checkpoint,
    save_pass1_cache,
    validate_correction_patch,
    validate_downstream,
    validate_downstream_prefix,
    validate_pass4_consolidation,
    validate_pass1,
    validate_pass1_geometry,
)


class ClosedLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = {
            "problem_id": "p_test",
            "images": [{"image_id": "img_0", "path": "images/test.png", "width": 200, "height": 100}],
            "segments": {"stem": "A diagram.", "query": "What is shown?", "options": []},
        }
        self.node = {
            "id": "v001",
            "type": "line_segment",
            "subtype": "wire",
            "text": "",
            "image_id": "img_0",
            "bbox_1000": [100, 200, 500, 220],
            "keypoints_1000": [[100, 210], [500, 210]],
            "center_1000": [-1, -1],
            "radius_1000": -1,
            "confidence": "high",
        }
        self.pass1 = {"problem_id": "p_test", "visual_nodes": [deepcopy(self.node)], "ambiguities": []}

    def test_valid_pass1_passes_closed_loop_geometry(self) -> None:
        self.assertEqual(validate_pass1(self.pass1, self.problem), [])

    def test_pass1_system_prompt_excludes_downstream_schema_definitions(self) -> None:
        prompt = load_pass1_system_instructions(ROOT, ROOT / "physgraph_annotation")
        self.assertNotIn("PASS2 JSON Schema", prompt)
        self.assertNotIn('"physical_node":', prompt)
        self.assertNotIn('"quantity":', prompt)
        self.assertIn('"pixel_node":', prompt)
        self.assertIn('"ambiguity":', prompt)
        self.assertIn("禁止你自行归一化", prompt)

    def test_pass1_cache_rejects_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            save_pass1_cache(workspace, self.problem, self.pass1, {"source": "unit"})
            self.assertEqual(load_pass1_cache(workspace, self.problem), self.pass1)
            cache_path = workspace / "pipeline_cache" / "verified_pass1" / "p_test.json"
            envelope = read_json(cache_path)
            envelope["pass1"]["visual_nodes"][0]["bbox_1000"] = [110, 200, 500, 220]
            write_json_atomic(cache_path, envelope)
            self.assertIsNone(load_pass1_cache(workspace, self.problem))

    def test_pixel_panel_coordinates_are_deterministically_normalized(self) -> None:
        panel = PanelSpec(
            image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
            source_width_px=200, source_height_px=100,
            x0_px=0, y0_px=20, x1_px=200, y1_px=100,
        )
        pixel_node = {
            "id": "local1", "type": "line_segment", "subtype": "wire", "text": "",
            "bbox_px": [20, 10, 100, 20], "keypoints_px": [[20, 15], [100, 15]],
            "center_px": [-1, -1], "radius_px": -1, "confidence": "high",
        }
        standard = pixel_node_to_standard(pixel_node, panel)
        self.assertEqual(standard["bbox_1000"], [100, 300, 500, 400])
        self.assertEqual(standard["keypoints_1000"], [[100, 350], [500, 350]])
        self.assertEqual(standard["image_id"], "img_0")

    def test_panel_document_rejects_coordinates_outside_crop(self) -> None:
        panel = PanelSpec(
            image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
            source_width_px=200, source_height_px=100,
            x0_px=0, y0_px=0, x1_px=200, y1_px=100,
        )
        node = {
            "id": "local1", "type": "line_segment", "subtype": "wire", "text": "",
            "bbox_px": [20, 10, 250, 20], "keypoints_px": [[20, 15], [250, 15]],
            "center_px": [-1, -1], "radius_px": -1, "confidence": "high",
        }
        document = {**panel_contract(panel, "p_test"), "visual_nodes": [node], "ambiguities": []}
        codes = {issue["code"] for issue in validate_panel_document(document, panel=panel, problem_id="p_test")}
        self.assertIn("pixel_bbox", codes)

    def test_panel_normalization_uses_keypoint_evidence_but_rejects_global_xywh(self) -> None:
        panel = PanelSpec(
            image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
            source_width_px=200, source_height_px=100,
            x0_px=0, y0_px=0, x1_px=200, y1_px=100,
        )
        base = {
            "subtype": "wire", "text": "", "confidence": "high",
            "center_px": [50, 50], "radius_px": 5,
        }
        document = {
            **panel_contract(panel, "p_test"),
            "visual_nodes": [
                {
                    **base, "id": "line", "type": "line_segment",
                    "bbox_px": [100, 40, 20, 20],
                    "keypoints_px": [[20, 20], [60, 30], [100, 40]],
                },
                {
                    **base, "id": "shape", "type": "object_shape",
                    "bbox_px": [120, 50, 160, 70], "keypoints_px": [[110, 60]],
                },
            ],
            "ambiguities": [],
        }
        normalized, audit = normalize_panel_document(document, panel=panel)
        self.assertEqual(validate_panel_document(normalized, panel=panel, problem_id="p_test"), [])
        self.assertEqual(normalized["visual_nodes"][0]["bbox_px"], [20, 20, 100, 40])
        self.assertEqual(normalized["visual_nodes"][1]["bbox_px"], [110, 50, 160, 70])
        self.assertTrue(audit["changed"])

        xywh = deepcopy(document)
        xywh["visual_nodes"] = [
            {
                **base, "id": f"n{index}", "type": "text_glyph", "text": "x",
                "bbox_px": [150, 70, 20, 10], "keypoints_px": [],
            }
            for index in range(3)
        ]
        rejected, rejected_audit = normalize_panel_document(xywh, panel=panel)
        self.assertTrue(rejected_audit["likely_xywh_document"])
        codes = {
            issue["code"]
            for issue in validate_panel_document(rejected, panel=panel, problem_id="p_test")
        }
        self.assertIn("pixel_bbox", codes)

    def test_pixel_patch_normalization_resets_non_circle_geometry_only(self) -> None:
        patch = {
            "problem_id": "p_test", "image_id": "img_0", "panel_id": "img_0_p01",
            "coordinate_space": "crop_pixels", "verdict": "correct", "summary": "point",
            "operations": [{
                "action": "replace", "target_id": "v001", "reason": "visible point",
                "node": {
                    "id": "v001", "type": "point_mark", "subtype": "dot", "text": "",
                    "bbox_px": [45, 45, 55, 55], "keypoints_px": [],
                    "center_px": [50, 50], "radius_px": 5, "confidence": "high",
                },
            }],
        }
        normalized, audit = normalize_pixel_correction_patch(patch)
        self.assertEqual(normalized["operations"][0]["node"]["center_px"], [-1, -1])
        self.assertEqual(normalized["operations"][0]["node"]["radius_px"], -1)
        self.assertTrue(audit["changed"])

    def test_panel_merge_renumbers_local_ids_and_ambiguities(self) -> None:
        panel = PanelSpec(
            image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
            source_width_px=200, source_height_px=100,
            x0_px=0, y0_px=0, x1_px=200, y1_px=100,
        )
        node = {
            "id": "local7", "type": "line_segment", "subtype": "wire", "text": "",
            "bbox_px": [20, 10, 100, 20], "keypoints_px": [[20, 15], [100, 15]],
            "center_px": [-1, -1], "radius_px": -1, "confidence": "high",
        }
        document = {
            **panel_contract(panel, "p_test"), "visual_nodes": [node],
            "ambiguities": [{
                "id": "maybe", "scope": "visual", "description": "unclear endpoint",
                "candidate_ids": ["local7"], "evidence_visual_ids": ["local7"],
                "evidence_mention_ids": [],
            }],
        }
        merged = merge_panel_documents(problem_id="p_test", panels=[panel], documents=[document])
        self.assertEqual(merged["visual_nodes"][0]["id"], "v001")
        self.assertEqual(merged["ambiguities"][0]["id"], "u001")
        self.assertEqual(merged["ambiguities"][0]["candidate_ids"], ["v001"])

    def test_horizontal_panel_split_uses_only_large_blank_bands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compound.png"
            image = Image.new("L", (200, 400), 255)
            for y1, y2 in ((20, 130), (210, 360)):
                for x in range(20, 180):
                    image.putpixel((x, y1), 0)
                    image.putpixel((x, y2), 0)
                for y in range(y1, y2 + 1):
                    image.putpixel((20, y), 0)
                    image.putpixel((179, y), 0)
            image.save(path)
            panels = detect_horizontal_panels(
                image_path=path, image_id="img_0", source_path="images/compound.png",
            )
            self.assertEqual(len(panels), 2)
            self.assertEqual(panels[0].y1_px, panels[1].y0_px)
            raw, grid = render_panel_assets(dataset_dir=Path(temporary), panel=PanelSpec(
                image_id="img_0", panel_id="img_0_p01", source_path="compound.png",
                source_width_px=200, source_height_px=400,
                x0_px=0, y0_px=0, x1_px=200, y1_px=panels[0].y1_px,
            ), output_dir=Path(temporary) / "assets")
            self.assertTrue(raw.is_file())
            self.assertTrue(grid.is_file())

    def test_panel_pixel_generation_and_acceptance_integration(self) -> None:
        class GenerateClient:
            endpoint_label = "https://unit.test/v1"

            def request_json(self, **kwargs):
                panel = PanelSpec(
                    image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
                    source_width_px=200, source_height_px=100,
                    x0_px=0, y0_px=0, x1_px=200, y1_px=100,
                )
                document = {
                    **panel_contract(panel, "p_test"),
                    "visual_nodes": [{
                        "id": "local1", "type": "line_segment", "subtype": "wire", "text": "",
                        "bbox_px": [20, 20, 100, 24], "keypoints_px": [[20, 22], [100, 22]],
                        "center_px": [-1, -1], "radius_px": -1, "confidence": "high",
                    }],
                    "ambiguities": [],
                }
                result = APIResult(
                    text="{}", api_mode="chat", request_id="generate", model="test-multimodal-model",
                    usage={}, duration_seconds=0.01, status="stop",
                )
                return document, result

        class AcceptClient:
            endpoint_label = "https://unit.test/v1"

            def request_json(self, **kwargs):
                patch = {
                    "problem_id": "p_test", "image_id": "img_0", "panel_id": "img_0_p01",
                    "coordinate_space": "crop_pixels", "verdict": "accept",
                    "operations": [], "summary": "verified",
                }
                result = APIResult(
                    text="{}", api_mode="chat", request_id="accept", model="test-multimodal-model",
                    usage={}, duration_seconds=0.01, status="stop",
                )
                return patch, result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            args = SimpleNamespace(
                pass1_attempts=1, pass1_max_output_tokens=1024,
                correction_rounds=2, correction_attempts=1, correction_max_output_tokens=1024,
                model="test-multimodal-model", reasoning="high", image_detail="original",
            )
            progress = {"api_usage": {}, "api_usage_by_phase": {"pass1_generate": {}, "pass1_correct": {}}}
            generated, issues, panels, assets = generate_pass1_by_panels(
                api_client=GenerateClient(), args=args, system_prompt="pixel", problem=self.problem,
                dataset_dir=root, run_dir=root / "run", progress=progress,
                progress_path=root / "progress.json", review_notes=None, resume_inventory=None,
            )
            self.assertEqual(issues, [])
            self.assertEqual(generated["visual_nodes"][0]["bbox_1000"], [100, 200, 500, 240])
            verified, issues, audit = correct_pass1_by_panels(
                api_client=AcceptClient(), args=args, system_prompt="correct", problem=self.problem,
                dataset_dir=root, initial_pass1=generated, panels=panels, assets=assets,
                run_dir=root / "run", progress=progress, progress_path=root / "progress.json",
            )
            self.assertEqual(issues, [])
            self.assertIsNotNone(verified)
            self.assertEqual(audit["status"], "accepted")
            self.assertEqual(audit["panels"][0]["accepted_round"], 1)

    def test_append_ids_are_allocated_locally_without_global_collision(self) -> None:
        second = deepcopy(self.node)
        second["id"] = "v002"
        current = {
            "problem_id": "p_test",
            "visual_nodes": [deepcopy(self.node), second],
            "ambiguities": [],
        }
        appended = deepcopy(self.node)
        appended["id"] = "v002"
        patch = {
            "problem_id": "p_test", "verdict": "correct", "summary": "missing line",
            "operations": [{
                "action": "append", "target_id": "", "node": appended,
                "reason": "add a distinct segment",
            }],
        }
        normalized, allocation = allocate_append_ids(patch, current)
        self.assertEqual(normalized["operations"][0]["node"]["id"], "__panel_append_001")
        self.assertEqual(allocation[0]["model_temporary_id"], "v002")
        self.assertEqual(validate_correction_patch(normalized, "p_test", current), [])
        candidate, _ = apply_correction_patch(current, normalized)
        self.assertEqual([node["id"] for node in candidate["visual_nodes"]], ["v001", "v002", "v003"])

    def test_pixel_checkpoint_reuses_valid_geometry_and_accepted_panels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            panels = build_problem_panels(self.problem, root)
            run = root / "run"
            run.mkdir()
            checkpoint = run / "p_test.pass1.unverified_after_correction.json"
            write_json_atomic(checkpoint, self.pass1)
            write_json_atomic(
                run / "p_test.pass1.panel_manifest.json",
                [panel.as_record() for panel in panels],
            )
            write_json_atomic(
                run / "p_test.pass1.correction_audit.json",
                {
                    "mode": "per_panel_crop_pixels",
                    "status": "unverified",
                    "final_pass1_sha256": json_sha256(self.pass1),
                    "panels": [{
                        "panel": panels[0].as_record(),
                        "status": "accepted",
                        "accepted_round": 1,
                    }],
                },
            )
            candidate, accepted, reason = load_pixel_correction_checkpoint(
                resume_path=checkpoint,
                problem=self.problem,
                dataset_dir=root,
            )
            self.assertEqual(candidate, self.pass1)
            self.assertEqual(accepted, {"img_0_p01"})
            self.assertEqual(reason, "validated_pixel_checkpoint")

    def test_discovery_finds_latest_valid_pixel_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "dataset" / "images" / "test.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (200, 100), "white").save(image_path)
            panels = build_problem_panels(self.problem, root / "dataset")
            run = root / "batch_runs" / "20260101_000000"
            run.mkdir(parents=True)
            checkpoint = run / "p_test.pass1.unverified_after_correction.json"
            write_json_atomic(checkpoint, self.pass1)
            write_json_atomic(run / "p_test.pass1.panel_manifest.json", [panel.as_record() for panel in panels])
            write_json_atomic(run / "p_test.pass1.correction_audit.json", {
                "mode": "per_panel_crop_pixels", "status": "unverified",
                "final_pass1_sha256": json_sha256(self.pass1),
                "panels": [{"panel": panels[0].as_record(), "status": "unverified_after_max_rounds", "rounds": []}],
            })
            candidate, accepted, path, history, reason = discover_latest_pixel_checkpoint(
                workspace=root,
                problem=self.problem,
                dataset_dir=root / "dataset",
            )
            self.assertEqual(candidate, self.pass1)
            self.assertEqual(accepted, set())
            self.assertEqual(path, checkpoint)
            self.assertEqual(history, {"img_0_p01": set()})
            self.assertEqual(reason, "validated_pixel_checkpoint")



    def test_pass4_consolidation_rejects_erased_core_state(self) -> None:
        payload = {
            "pass2": {"physical_nodes": [{"id": "p001"}]},
            "pass3": {
                "new_or_updated_physical_nodes": [],
                "text_mentions": [{"id": "m001"}],
                "quantities": [{"id": "q001"}],
                "constraints": [],
                "query_target": {"mention_id": "m001"},
            },
            "pass4": {
                "physical_nodes": [],
                "text_mentions": [],
                "quantities": [],
                "constraints": [],
                "query_target": {"mention_id": "m001"},
            },
        }
        issues = validate_pass4_consolidation(payload)
        paths = {issue["path"] for issue in issues}
        self.assertIn("$.physical_nodes", paths)
        self.assertIn("$.text_mentions", paths)
        self.assertIn("$.quantities", paths)

    def test_pass4_only_recovery_keeps_validated_prefix_immutable(self) -> None:
        physical = {
            "id": "p001", "type": "wire", "subtype": "wire", "name": "wire", "symbol": "",
            "visual_anchor_ids": ["v001"], "text_mention_ids": ["m001"],
            "provenance": ["IMAGE", "TEXT"], "confidence": "high",
        }
        mention = {
            "id": "m001", "section": "query", "quote": "What is shown?",
            "occurrence": 1, "role": "query_target",
        }
        query_target = {
            "mention_id": "m001", "target_kind": "other", "target_symbol_latex": "",
            "target_node_ids": ["p001"], "location_node_ids": [], "time_or_event_node_ids": [],
        }
        prefix = {
            "pass2": {
                "problem_id": "p_test",
                "physical_nodes": [{**deepcopy(physical), "text_mention_ids": [], "provenance": ["IMAGE"]}],
                "bindings": [{
                    "id": "b001", "type": "represents", "from_id": "v001", "to_id": "p001",
                    "provenance": ["IMAGE"], "evidence_visual_ids": ["v001"], "evidence_mention_ids": [],
                }],
                "ambiguities": [],
            },
            "pass3": {
                "problem_id": "p_test", "new_or_updated_physical_nodes": [deepcopy(physical)],
                "text_mentions": [deepcopy(mention)], "quantities": [],
                "bindings": [{
                    "id": "b001", "type": "refers_to", "from_id": "m001", "to_id": "p001",
                    "provenance": ["TEXT"], "evidence_visual_ids": [], "evidence_mention_ids": ["m001"],
                }],
                "constraints": [], "query_target": deepcopy(query_target), "ambiguities": [],
            },
        }
        valid_pass4 = {
            "schema_version": "physgraph_obs_v0.1", "problem_id": "p_test", "domain": "mechanics",
            "visual_nodes": deepcopy(self.pass1["visual_nodes"]), "physical_nodes": [deepcopy(physical)],
            "text_mentions": [deepcopy(mention)], "quantities": [],
            "bindings": [
                {
                    "id": "b001", "type": "represents", "from_id": "v001", "to_id": "p001",
                    "provenance": ["IMAGE"], "evidence_visual_ids": ["v001"], "evidence_mention_ids": [],
                },
                {
                    "id": "b002", "type": "refers_to", "from_id": "m001", "to_id": "p001",
                    "provenance": ["TEXT"], "evidence_visual_ids": [], "evidence_mention_ids": ["m001"],
                },
            ],
            "relations": [], "constraints": [], "query_target": deepcopy(query_target), "ambiguities": [],
        }
        erased = deepcopy(valid_pass4)
        erased["physical_nodes"] = []
        erased["text_mentions"] = []

        class FakeClient:
            endpoint_label = "https://unit.test/v1"

            def __init__(self):
                self.responses = [{"pass4": erased}, {"pass4": valid_pass4}]
                self.calls = []

            def request_json(self, **kwargs):
                self.calls.append(kwargs)
                return deepcopy(self.responses.pop(0)), APIResult(
                    text="{}", api_mode="chat", request_id=str(len(self.calls)), model="test-multimodal-model",
                    usage={"total_tokens": 1}, duration_seconds=0.01, status="stop",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                downstream_attempts=2, model="test-multimodal-model", reasoning="high",
                image_detail="original", downstream_max_output_tokens=4096,
            )
            progress = {"api_usage": {}, "api_usage_by_phase": {}}
            before = deepcopy(prefix)
            client = FakeClient()
            recovered, issues = generate_pass4_from_prefix(
                api_client=client, args=args, system_prompt="unit", problem=self.problem,
                pass1=self.pass1, prefix=prefix, prefix_source_path=root / "api_response.json",
                run_dir=root / "run", progress=progress, progress_path=root / "progress.json",
                review_notes=None,
            )
            self.assertEqual(issues, [])
            self.assertEqual(prefix, before)
            self.assertEqual(validate_downstream(recovered, self.pass1, self.problem), [])
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0]["image_paths"], [])
            self.assertTrue((root / "run" / "p_test.downstream.pass4_recovery.attempt2.assembled.json").is_file())







    def test_no_effect_pixel_patch_is_implicitly_accepted(self) -> None:
        class NoEffectClient:
            endpoint_label = "https://unit.test/v1"

            def request_json(self, **kwargs):
                patch = {
                    "problem_id": "p_test", "image_id": "img_0", "panel_id": "img_0_p01",
                    "coordinate_space": "crop_pixels", "verdict": "correct",
                    "operations": [{
                        "action": "replace", "target_id": "v001",
                        "node": {
                            "id": "v001", "type": "line_segment", "subtype": "wire", "text": "",
                            "bbox_px": [20, 20, 100, 24], "keypoints_px": [[20, 22], [100, 22]],
                            "center_px": [-1, -1], "radius_px": -1, "confidence": "high",
                        },
                        "reason": "values already match after inspection",
                    }],
                    "summary": "no effective change",
                }
                return patch, APIResult(
                    text="{}", api_mode="chat", request_id="noop", model="test-multimodal-model",
                    usage={}, duration_seconds=0.01, status="stop",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            panel = PanelSpec(
                image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
                source_width_px=200, source_height_px=100,
                x0_px=0, y0_px=0, x1_px=200, y1_px=100,
            )
            raw, grid = render_panel_assets(dataset_dir=root, panel=panel, output_dir=root / "assets")
            initial = deepcopy(self.pass1)
            initial["visual_nodes"][0]["bbox_1000"] = [100, 200, 500, 240]
            initial["visual_nodes"][0]["keypoints_1000"] = [[100, 220], [500, 220]]
            args = SimpleNamespace(
                correction_rounds=3, correction_attempts=1, model="test-multimodal-model",
                reasoning="high", image_detail="original", correction_max_output_tokens=1024,
            )
            progress = {"api_usage": {}, "api_usage_by_phase": {"pass1_correct": {}}}
            verified, issues, audit = correct_pass1_by_panels(
                api_client=NoEffectClient(), args=args, system_prompt="unit", problem=self.problem,
                dataset_dir=root, initial_pass1=initial, panels=[panel], assets={panel.panel_id: (raw, grid)},
                run_dir=root / "run", progress=progress, progress_path=root / "progress.json",
            )
            self.assertEqual(issues, [])
            self.assertEqual(verified, initial)
            self.assertEqual(audit["panels"][0]["acceptance_source"], "implicit_accept_no_effect")
            self.assertEqual(audit["panels"][0]["rounds"][0]["model_verdict"], "correct")

    def test_correction_state_oscillation_is_rejected(self) -> None:
        class OscillatingClient:
            endpoint_label = "https://unit.test/v1"

            def __init__(self):
                self.calls = 0

            def request_json(self, **kwargs):
                self.calls += 1
                bbox = [30, 20, 100, 24] if self.calls == 1 else [20, 20, 100, 24]
                keypoints = [[bbox[0], 22], [100, 22]]
                patch = {
                    "problem_id": "p_test", "image_id": "img_0", "panel_id": "img_0_p01",
                    "coordinate_space": "crop_pixels", "verdict": "correct",
                    "operations": [{
                        "action": "replace", "target_id": "v001",
                        "node": {
                            "id": "v001", "type": "line_segment", "subtype": "wire", "text": "",
                            "bbox_px": bbox, "keypoints_px": keypoints,
                            "center_px": [-1, -1], "radius_px": -1, "confidence": "high",
                        }, "reason": "toggle endpoints",
                    }], "summary": "toggle",
                }
                return patch, APIResult(
                    text="{}", api_mode="chat", request_id=str(self.calls), model="test-multimodal-model",
                    usage={}, duration_seconds=0.01, status="stop",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            panel = PanelSpec(
                image_id="img_0", panel_id="img_0_p01", source_path="images/test.png",
                source_width_px=200, source_height_px=100,
                x0_px=0, y0_px=0, x1_px=200, y1_px=100,
            )
            raw, grid = render_panel_assets(dataset_dir=root, panel=panel, output_dir=root / "assets")
            initial = deepcopy(self.pass1)
            initial["visual_nodes"][0]["bbox_1000"] = [100, 200, 500, 240]
            initial["visual_nodes"][0]["keypoints_1000"] = [[100, 220], [500, 220]]
            args = SimpleNamespace(
                correction_rounds=2, correction_attempts=1, model="test-multimodal-model",
                reasoning="high", image_detail="original", correction_max_output_tokens=1024,
            )
            progress = {"api_usage": {}, "api_usage_by_phase": {"pass1_correct": {}}}
            client = OscillatingClient()
            verified, issues, _ = correct_pass1_by_panels(
                api_client=client, args=args, system_prompt="unit", problem=self.problem,
                dataset_dir=root, initial_pass1=initial, panels=[panel], assets={panel.panel_id: (raw, grid)},
                run_dir=root / "run", progress=progress, progress_path=root / "progress.json",
            )
            self.assertIsNone(verified)
            self.assertEqual(issues[0]["code"], "correction_oscillation")
            self.assertEqual(client.calls, 2)

    def test_keypoint_outside_bbox_is_rejected(self) -> None:
        document = deepcopy(self.pass1)
        document["visual_nodes"][0]["keypoints_1000"][1] = [700, 210]
        codes = {issue["code"] for issue in validate_pass1_geometry(document)}
        self.assertIn("keypoint_outside_bbox", codes)

    def test_non_circle_cannot_claim_center_or_radius(self) -> None:
        document = deepcopy(self.pass1)
        document["visual_nodes"][0]["center_1000"] = [300, 210]
        document["visual_nodes"][0]["radius_1000"] = 10
        codes = {issue["code"] for issue in validate_pass1_geometry(document)}
        self.assertIn("unexpected_circle_geometry", codes)

    def test_arc_center_may_lie_outside_tight_arc_bbox(self) -> None:
        document = deepcopy(self.pass1)
        node = document["visual_nodes"][0]
        node.update({
            "type": "arc",
            "subtype": "shallow_arc",
            "bbox_1000": [100, 100, 300, 180],
            "keypoints_1000": [[100, 170], [200, 100], [300, 170]],
            "center_1000": [200, 700],
            "radius_1000": 600,
        })
        codes = {issue["code"] for issue in validate_pass1_geometry(document)}
        self.assertNotIn("center_outside_bbox", codes)

    def test_replace_patch_changes_only_target_node(self) -> None:
        replacement = deepcopy(self.node)
        replacement["bbox_1000"] = [110, 200, 490, 220]
        replacement["keypoints_1000"] = [[110, 210], [490, 210]]
        patch = {
            "problem_id": "p_test",
            "verdict": "correct",
            "operations": [{
                "action": "replace", "target_id": "v001",
                "node": replacement, "reason": "tighten endpoints",
            }],
            "summary": "one correction",
        }
        self.assertEqual(validate_correction_patch(patch, "p_test", self.pass1), [])
        candidate, audit = apply_correction_patch(self.pass1, patch)
        self.assertEqual(candidate["visual_nodes"][0]["bbox_1000"], [110, 200, 490, 220])
        self.assertEqual(audit[0]["action"], "replace")

    def test_delete_append_renumbers_and_remaps_ambiguity(self) -> None:
        second = deepcopy(self.node)
        second["id"] = "v002"
        second["bbox_1000"] = [600, 200, 800, 220]
        second["keypoints_1000"] = [[600, 210], [800, 210]]
        current = {
            "problem_id": "p_test",
            "visual_nodes": [deepcopy(self.node), second],
            "ambiguities": [{
                "id": "u001", "scope": "visual", "description": "test",
                "candidate_ids": ["v001", "v002"],
                "evidence_visual_ids": ["v001", "v002"],
                "evidence_mention_ids": [],
            }],
        }
        appended = deepcopy(self.node)
        appended["id"] = "new001"
        appended["bbox_1000"] = [50, 500, 300, 520]
        appended["keypoints_1000"] = [[50, 510], [300, 510]]
        patch = {
            "problem_id": "p_test", "verdict": "correct", "summary": "replace set",
            "operations": [
                {"action": "delete", "target_id": "v001", "node": None, "reason": "duplicate"},
                {"action": "append", "target_id": "", "node": appended, "reason": "missing line"},
            ],
        }
        self.assertEqual(validate_correction_patch(patch, "p_test", current), [])
        candidate, _ = apply_correction_patch(current, patch)
        self.assertEqual([node["id"] for node in candidate["visual_nodes"]], ["v001", "v002"])
        self.assertEqual(candidate["ambiguities"][0]["candidate_ids"], ["v001"])
        self.assertEqual(candidate["ambiguities"][0]["evidence_visual_ids"], ["v001"])

    def test_accept_must_not_contain_operations(self) -> None:
        patch = {
            "problem_id": "p_test", "verdict": "accept", "summary": "bad",
            "operations": [{"action": "delete", "target_id": "v001", "node": None, "reason": "x"}],
        }
        codes = {issue["code"] for issue in validate_correction_patch(patch, "p_test", self.pass1)}
        self.assertIn("accept_has_operations", codes)

    def test_downstream_cannot_modify_verified_visual_nodes(self) -> None:
        payload = {
            "pass4": {"visual_nodes": [{"id": "tampered"}]},
        }
        normalized, changes = force_verified_pass1_into_pass4(payload, self.pass1)
        self.assertIsNotNone(changes)
        self.assertEqual(normalized["pass4"]["visual_nodes"], self.pass1["visual_nodes"])
        self.assertIsNot(normalized["pass4"]["visual_nodes"], self.pass1["visual_nodes"])
        self.assertEqual(changes["before_count"], 1)
        self.assertEqual(changes["after_count"], 1)

    def test_downstream_validator_rejects_tampered_visual_nodes(self) -> None:
        # The wrapper intentionally lacks valid Pass 2/3 content; this assertion
        # isolates the additional immutable-boundary error.
        payload = {
            "pass2": {},
            "pass3": {},
            "pass4": {"visual_nodes": [{"id": "tampered"}]},
        }
        codes = {issue["code"] for issue in validate_downstream(payload, self.pass1, self.problem)}
        self.assertIn("verified_pass1_not_copied", codes)

    def test_corrected_overlay_must_be_seen_again_and_accepted(self) -> None:
        replacement = deepcopy(self.node)
        replacement["bbox_1000"] = [110, 200, 490, 220]
        replacement["keypoints_1000"] = [[110, 210], [490, 210]]
        patches = [
            {
                "problem_id": "p_test", "verdict": "correct", "summary": "tighten",
                "operations": [{
                    "action": "replace", "target_id": "v001", "node": replacement,
                    "reason": "tighten the line box",
                }],
            },
            {"problem_id": "p_test", "verdict": "accept", "operations": [], "summary": "verified"},
        ]

        class FakeClient:
            endpoint_label = "https://unit.test/v1"

            def __init__(self, responses):
                self.responses = responses
                self.calls = []

            def request_json(self, **kwargs):
                self.calls.append(kwargs)
                response = deepcopy(self.responses.pop(0))
                result = APIResult(
                    text="{}", api_mode="chat", request_id=f"r{len(self.calls)}",
                    model="test-multimodal-model", usage={"total_tokens": 1},
                    duration_seconds=0.01, status="stop",
                )
                return response, result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            client = FakeClient(patches)
            args = SimpleNamespace(
                correction_rounds=2, correction_attempts=1, model="test-multimodal-model",
                reasoning="high", image_detail="original", correction_max_output_tokens=1024,
            )
            progress = {
                "api_usage": {},
                "api_usage_by_phase": {"pass1_correct": {}},
            }
            verified, issues, audit = correct_pass1(
                api_client=client,
                args=args,
                system_prompt="unit",
                problem=self.problem,
                dataset_dir=root,
                image_paths=[image_path],
                initial_pass1=self.pass1,
                run_dir=root / "run",
                progress=progress,
                progress_path=root / "progress.json",
            )
            self.assertEqual(issues, [])
            self.assertIsNotNone(verified)
            self.assertEqual(verified["visual_nodes"][0]["bbox_1000"], [110, 200, 490, 220])
            self.assertEqual(audit["accepted_round"], 2)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(len(client.calls[0]["image_paths"]), 2)
            self.assertEqual(len(client.calls[1]["image_paths"]), 2)
            self.assertIn("correction_round2_input", client.calls[1]["image_paths"][1].name)
            self.assertTrue((root / "run" / "p_test.pass1.verified.json").is_file())

    def test_last_round_correction_fails_closed_without_accept(self) -> None:
        replacement = deepcopy(self.node)
        replacement["bbox_1000"] = [110, 200, 490, 220]
        replacement["keypoints_1000"] = [[110, 210], [490, 210]]

        class FakeClient:
            endpoint_label = "https://unit.test/v1"

            def request_json(self, **kwargs):
                patch = {
                    "problem_id": "p_test", "verdict": "correct", "summary": "tighten",
                    "operations": [{
                        "action": "replace", "target_id": "v001", "node": deepcopy(replacement),
                        "reason": "tighten the line box",
                    }],
                }
                result = APIResult(
                    text="{}", api_mode="chat", request_id="r1", model="test-multimodal-model",
                    usage={}, duration_seconds=0.01, status="stop",
                )
                return patch, result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            args = SimpleNamespace(
                correction_rounds=1, correction_attempts=1, model="test-multimodal-model",
                reasoning="high", image_detail="original", correction_max_output_tokens=1024,
            )
            progress = {"api_usage": {}, "api_usage_by_phase": {"pass1_correct": {}}}
            verified, issues, audit = correct_pass1(
                api_client=FakeClient(), args=args, system_prompt="unit", problem=self.problem,
                dataset_dir=root, image_paths=[image_path], initial_pass1=self.pass1,
                run_dir=root / "run", progress=progress, progress_path=root / "progress.json",
            )
            self.assertIsNone(verified)
            self.assertEqual(issues[0]["code"], "correction_unverified_after_max_rounds")
            self.assertEqual(audit["status"], "unverified_after_max_rounds")
            self.assertTrue((root / "run" / "p_test.pass1.unverified_after_correction.json").is_file())

    def test_overlay_renderer_creates_auditable_png_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "test.png"
            image_path.parent.mkdir()
            Image.new("RGB", (200, 100), "white").save(image_path)
            paths, manifest = render_pass1_overlays(
                problem=self.problem,
                dataset_dir=root,
                pass1=self.pass1,
                output_dir=root / "overlays",
                label="unit",
            )
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertGreater(paths[0].stat().st_size, 0)
            self.assertEqual(manifest[0]["source_width"], 200)
            self.assertEqual(manifest[0]["node_count"], 1)
            self.assertTrue((root / "overlays" / "p_test.unit.overlay_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
