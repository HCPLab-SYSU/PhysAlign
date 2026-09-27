from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_annotation_lib import (  # noqa: E402
    blank_document,
    canonicalize_pass4_visual_nodes,
    json_sha256,
    validate_document,
)
from physgraph_annotation_lib import write_json_atomic, write_jsonl_atomic  # noqa: E402
from prepare_physgraph_annotation import split_stem_query_options  # noqa: E402
from run_physgraph_annotation_review import AnnotationState  # noqa: E402
from workspace_fixtures import example_workspace


class PromptSegmentationTests(unittest.TestCase):
    def test_inline_options_are_split(self) -> None:
        segments, metadata = split_stem_query_options(
            "A body moves uniformly. The speed is () A．1 m/s B．2 m/s C．3 m/s D．4 m/s"
        )
        self.assertEqual([item["label"] for item in segments["options"]], ["A", "B", "C", "D"])
        self.assertIn("The speed", segments["query"])
        self.assertIn(metadata["confidence"], {"medium", "high"})

    def test_entity_labels_are_not_mistaken_for_options(self) -> None:
        segments, metadata = split_stem_query_options(
            "The switch starts at position A. At t=0 it is moved to position B. Immediately after contact with B: What is the current through R?"
        )
        self.assertEqual(segments["options"], [])
        self.assertIn("position B", segments["stem"])
        self.assertTrue(segments["query"].startswith("What is"))
        self.assertEqual(metadata["confidence"], "high")

    def test_question_phrase_is_separated(self) -> None:
        segments, metadata = split_stem_query_options(
            "A block is shown in the figure. Which of the following statements is correct?\nA. One\nB. Two"
        )
        self.assertEqual(segments["stem"], "A block is shown in the figure.")
        self.assertTrue(segments["query"].startswith("Which of the following"))
        self.assertEqual(metadata["confidence"], "high")


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = {
            "problem_id": "physics_000001",
            "images": [{"image_id": "img_0", "path": "images/a.png"}],
            "segments": {"stem": "A block A is shown.", "query": "Find the acceleration.", "options": []},
        }
        self.visual_node = {
            "id": "v001",
            "type": "object_shape",
            "subtype": "block",
            "text": "",
            "image_id": "img_0",
            "bbox_1000": [100, 100, 300, 300],
            "keypoints_1000": [],
            "center_1000": [-1, -1],
            "radius_1000": -1,
            "confidence": "high",
        }

    def test_valid_minimal_pass1(self) -> None:
        document = {"problem_id": "physics_000001", "visual_nodes": [self.visual_node], "ambiguities": []}
        self.assertEqual(validate_document("pass1", document, self.problem), [])

    def test_empty_pass1_is_rejected_for_image_problem(self) -> None:
        document = {"problem_id": "physics_000001", "visual_nodes": [], "ambiguities": []}
        codes = {item["code"] for item in validate_document("pass1", document, self.problem)}
        self.assertIn("empty_visual_extraction", codes)

    def test_pass4_visual_node_is_restored_exactly_from_pass1(self) -> None:
        changed = dict(self.visual_node)
        changed["center_1000"] = [200, 200]
        payload = {
            "pass1": {"problem_id": "physics_000001", "visual_nodes": [self.visual_node], "ambiguities": []},
            "pass4": {"visual_nodes": [changed]},
        }
        changes = canonicalize_pass4_visual_nodes(payload)
        self.assertEqual(payload["pass4"]["visual_nodes"], [self.visual_node])
        self.assertEqual(changes[0]["visual_node_id"], "v001")
        self.assertEqual(changes[0]["changed_fields"], ["center_1000"])

    def test_pass4_cannot_add_a_visual_node_absent_from_pass1(self) -> None:
        pass1 = {"problem_id": "physics_000001", "visual_nodes": [self.visual_node], "ambiguities": []}
        unknown = dict(self.visual_node)
        unknown["id"] = "v002"
        pass4 = blank_document("pass4", "physics_000001")
        pass4["visual_nodes"] = [unknown]
        issues = validate_document("pass4", pass4, self.problem, {"pass1": pass1})
        self.assertIn("pass1_unknown_visual", {item["code"] for item in issues})

    def test_answer_field_is_rejected(self) -> None:
        document = {
            "problem_id": "physics_000001",
            "visual_nodes": [self.visual_node],
            "ambiguities": [],
            "answer": "A",
        }
        codes = {item["code"] for item in validate_document("pass1", document, self.problem)}
        self.assertIn("answer_leakage_key", codes)
        self.assertIn("extra_key", codes)

    def test_option_text_cannot_be_a_stem_mention(self) -> None:
        pass1 = {"problem_id": "physics_000001", "visual_nodes": [self.visual_node], "ambiguities": []}
        pass2 = {
            "problem_id": "physics_000001",
            "physical_nodes": [{
                "id": "p001", "type": "body", "subtype": "block", "name": "block A", "symbol": "A",
                "visual_anchor_ids": ["v001"], "text_mention_ids": [], "provenance": ["IMAGE"], "confidence": "high",
            }],
            "bindings": [{
                "id": "b001", "type": "represents", "from_id": "v001", "to_id": "p001", "provenance": ["IMAGE"],
                "evidence_visual_ids": ["v001"], "evidence_mention_ids": [],
            }],
            "ambiguities": [],
        }
        document = {
            "problem_id": "physics_000001",
            "new_or_updated_physical_nodes": [],
            "text_mentions": [{"id": "m001", "section": "stem", "quote": "friction", "occurrence": 1, "role": "query_target"}],
            "quantities": [], "bindings": [], "constraints": [],
            "query_target": {"mention_id": "m001", "target_kind": "acceleration", "target_symbol_latex": "a", "target_node_ids": ["p001"], "location_node_ids": [], "time_or_event_node_ids": []},
            "ambiguities": [],
        }
        issues = validate_document("pass3", document, self.problem, {"pass1": pass1, "pass2": pass2})
        self.assertIn("quote_not_found", {item["code"] for item in issues})


class PreparedWorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.workspace = example_workspace(Path(temporary.name))

    def test_manifest_is_answer_blind_and_complete(self) -> None:
        records = [json.loads(line) for line in (self.workspace / "blind" / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
        summary = json.loads((self.workspace / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), summary["problems"])
        serialized = json.dumps(records, ensure_ascii=False).lower()
        for key in ('"answer":', '"reasoning":', '"solution":', '"conversations":'):
            self.assertNotIn(key, serialized)
        self.assertEqual(sum(len(record["images"]) for record in records), summary["images"])

    def test_server_payload_is_blind_and_prompts_resolve(self) -> None:
        app = AnnotationState(self.workspace, ROOT / "physgraph_review_app", ROOT / "physgraph_annotation" / "prompts")
        first_id = app.problems[0]["problem_id"]
        payload = app.problem_payload(first_id)
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        for key in ('"answer":', '"reasoning":', '"solution":', '"conversations":'):
            self.assertNotIn(key, serialized)
        for stage in ("pass1", "pass2", "pass3", "pass4"):
            self.assertNotIn("{problem_id}", payload["prompts"][stage]["task"])
            self.assertTrue((self.workspace / payload["prompts"][stage]["schema_path"]).is_file())

    def test_approval_state_is_persisted_in_isolated_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            problem = {
                "workspace_schema_version": 1,
                "problem_id": "physics_000001",
                "source_record_index": 0,
                "source_dataset": "test",
                "source_sample_id": "test-1",
                "source_split": "train",
                "language": "en",
                "images": [{"image_id": "img_0", "path": "images/a.png", "width": 100, "height": 100, "sha256": ""}],
                "raw_question": "A block is shown. Find its acceleration.",
                "segments": {"stem": "A block is shown.", "query": "Find its acceleration.", "options": []},
                "segmentation": {"method": "query_phrase", "confidence": "high", "review_status": "machine_split", "edited": False},
            }
            write_jsonl_atomic(workspace / "blind" / "manifest.jsonl", [problem])
            write_json_atomic(workspace / "workspace_config.json", {"workspace_schema_version": 1, "dataset_dir": str(workspace), "manifest": "blind/manifest.jsonl"})
            write_json_atomic(workspace / "reviews" / "state.json", {"workspace_schema_version": 1, "problems": {}})
            (workspace / "reviews" / "history.jsonl").write_text("", encoding="utf-8")
            for stage in ("pass1", "pass2", "pass3", "pass4"):
                (workspace / "passes" / stage).mkdir(parents=True)
            app = AnnotationState(workspace, ROOT / "physgraph_review_app", ROOT / "physgraph_annotation" / "prompts")
            document = {
                "problem_id": "physics_000001",
                "visual_nodes": [{
                    "id": "v001", "type": "object_shape", "subtype": "block", "text": "", "image_id": "img_0",
                    "bbox_1000": [100, 100, 300, 300], "keypoints_1000": [], "center_1000": [-1, -1], "radius_1000": -1,
                    "confidence": "high",
                }],
                "ambiguities": [],
            }
            payload = app.save_stage("physics_000001", "pass1", {"action": "approve", "reviewer": "tester", "note": "", "document": document})
            self.assertEqual(payload["stages"]["pass1"]["status"], "approved")
            persisted = json.loads((workspace / "reviews" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["problems"]["physics_000001"]["stages"]["pass1"]["status"], "approved")

    def test_geometry_only_pass1_save_syncs_pass4_and_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            problem = {
                "workspace_schema_version": 1, "problem_id": "physics_000001",
                "source_record_index": 0, "source_dataset": "test", "source_sample_id": "test-1",
                "source_split": "train", "language": "en",
                "images": [{"image_id": "img_0", "path": "images/a.png", "width": 100, "height": 100, "sha256": ""}],
                "raw_question": "A block is shown. Find its acceleration.",
                "segments": {"stem": "A block is shown.", "query": "Find its acceleration.", "options": []},
                "segmentation": {"method": "query_phrase", "confidence": "high", "review_status": "approved", "edited": False},
            }
            old_node = {
                "id": "v001", "type": "object_shape", "subtype": "block", "text": "", "image_id": "img_0",
                "bbox_1000": [100, 100, 300, 300], "keypoints_1000": [],
                "center_1000": [-1, -1], "radius_1000": -1, "confidence": "high",
            }
            pass1 = {"problem_id": "physics_000001", "visual_nodes": [old_node], "ambiguities": []}
            pass4 = blank_document("pass4", "physics_000001")
            pass4["visual_nodes"] = [old_node]
            write_jsonl_atomic(workspace / "blind" / "manifest.jsonl", [problem])
            write_json_atomic(workspace / "workspace_config.json", {"workspace_schema_version": 1, "dataset_dir": str(workspace), "manifest": "blind/manifest.jsonl"})
            write_json_atomic(workspace / "reviews" / "state.json", {
                "workspace_schema_version": 1,
                "problems": {"physics_000001": {"stages": {
                    "pass1": {"status": "approved", "document_sha256": json_sha256(pass1), "validation_errors": 0},
                    "pass4": {"status": "approved", "document_sha256": json_sha256(pass4), "validation_errors": 0},
                }}},
            })
            (workspace / "reviews" / "history.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (workspace / "reviews" / "history.jsonl").write_text("", encoding="utf-8")
            for stage in ("pass1", "pass2", "pass3", "pass4"):
                (workspace / "passes" / stage).mkdir(parents=True)
            write_json_atomic(workspace / "passes" / "pass1" / "physics_000001.json", pass1)
            write_json_atomic(workspace / "passes" / "pass4" / "physics_000001.json", pass4)
            cache = workspace / "pipeline_cache" / "verified_pass1" / "physics_000001.json"
            write_json_atomic(cache, {"dummy": True})
            edited = json.loads(json.dumps(pass1))
            edited["visual_nodes"][0]["bbox_1000"] = [120, 130, 320, 330]

            app = AnnotationState(workspace, ROOT / "physgraph_review_app", ROOT / "physgraph_annotation" / "prompts")
            payload = app.save_stage("physics_000001", "pass1", {
                "action": "save", "reviewer": "tester", "note": "geometry adjustment",
                "expected_sha256": json_sha256(pass1), "document": edited,
            })

            saved_pass4 = json.loads((workspace / "passes" / "pass4" / "physics_000001.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_pass4["visual_nodes"], edited["visual_nodes"])
            self.assertTrue(payload["save_result"]["pass4_visual_nodes_synced"])
            self.assertEqual(payload["stages"]["pass1"]["status"], "draft")
            self.assertEqual(payload["stages"]["pass4"]["status"], "draft")
            self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
