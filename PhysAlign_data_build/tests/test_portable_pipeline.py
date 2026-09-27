from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_annotation_lib import read_json, read_jsonl, resolve_workspace_path  # noqa: E402
from physgraph_pipeline.adapters import PhysicsDatasetAdapter, json_pointer_get  # noqa: E402
from physgraph_pipeline.config import ConfigError, load_pipeline_config  # noqa: E402
from physgraph_pipeline.doctor import diagnose_config, diagnose_workspace  # noqa: E402
from physgraph_pipeline.workspace import migrate_workspace, prepare_workspace  # noqa: E402
from run_physgraph_annotation_review import AnnotationState  # noqa: E402
from workspace_fixtures import example_workspace


class JsonPointerTests(unittest.TestCase):
    def test_pointer_unescapes_tokens_and_indexes_lists(self) -> None:
        value = {"a/b": {"~items": ["zero", "one"]}}
        self.assertEqual(json_pointer_get(value, "/a~1b/~0items/1"), "one")

    def test_missing_pointer_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ConfigError, "JSON Pointer"):
            json_pointer_get({"id": "x"}, "/missing")


class PortablePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        image = self.dataset / "images" / "题图 #1.svg"
        image.parent.mkdir()
        image.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180">'
            '<rect x="80" y="90" width="80" height="50" fill="#8aa4ff"/>'
            '<line x1="20" y1="140" x2="300" y2="140" stroke="#222"/>'
            "</svg>",
            encoding="utf-8",
        )
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        record = {
            "id": "portable_physics_0001",
            "question": "<image>\nA block rests on a horizontal surface. Find the normal force.",
            "images": ["images/题图 #1.svg"],
            "answer": "This source-only answer must never enter the blind manifest.",
            "metadata": {
                "split": "test",
                "language": "en",
                "question_images": [
                    {
                        "path": "images/题图 #1.svg",
                        "width": 320,
                        "height": 180,
                        "format": "SVG",
                        "sha256": digest,
                    }
                ],
            },
        }
        self.records = self.dataset / "questions.jsonl"
        self.records.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        self.workspace = self.root / "workspace"
        self.config_path = self.root / "dataset_config.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset": {
                        "name": "portable-test",
                        "records": "dataset/questions.jsonl",
                        "format": "jsonl",
                        "media_root": "dataset",
                        "fields": {
                            "id": "/id",
                            "question": "/question",
                            "images": "/images",
                            "metadata_images": "/metadata/question_images",
                            "source_split": "/metadata/split",
                            "language": "/metadata/language",
                        },
                    },
                    "annotation": {
                        "workspace": "workspace",
                        "prompt_dir": str(ROOT / "physgraph_annotation" / "prompts"),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_adapter_validates_and_normalizes_media(self) -> None:
        config = load_pipeline_config(self.config_path)
        problems = PhysicsDatasetAdapter(config).load()
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].images[0]["path"], "images/题图 #1.svg")
        self.assertEqual(problems[0].images[0]["width"], 320)
        self.assertTrue(diagnose_config(config)["ok"])

    def test_prepare_is_blind_portable_and_servable(self) -> None:
        config = load_pipeline_config(self.config_path)
        result = prepare_workspace(config)
        self.assertEqual(result["problems"], 1)

        manifest = read_jsonl(self.workspace / "blind" / "manifest.jsonl")
        serialized = json.dumps(manifest, ensure_ascii=False).lower()
        self.assertNotIn("source-only answer", serialized)
        self.assertNotIn('"answer"', serialized)

        workspace_config = read_json(self.workspace / "workspace_config.json")
        self.assertEqual(workspace_config["path_base"], "workspace")
        self.assertFalse(Path(workspace_config["dataset_dir"]).is_absolute())
        dataset_dir = resolve_workspace_path(self.workspace, workspace_config, "dataset_dir")
        self.assertEqual(dataset_dir, self.dataset.resolve())

        state = AnnotationState(
            self.workspace,
            ROOT / "physgraph_review_app",
            self.workspace / "prompts",
        )
        self.assertIn("images/题图 #1.svg", state.media_paths)
        self.assertTrue((state.dataset_dir / "images" / "题图 #1.svg").is_file())
        report = diagnose_workspace(self.workspace)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["missing_media"], 0)

    def test_copied_project_tree_keeps_relative_paths(self) -> None:
        prepare_workspace(load_pipeline_config(self.config_path))
        copied = self.root / "copied_project"
        copied_dataset = copied / "dataset"
        copied_workspace = copied / "workspace"
        shutil.copytree(self.dataset, copied_dataset)
        shutil.copytree(self.workspace, copied_workspace)
        copied_config = read_json(copied_workspace / "workspace_config.json")
        self.assertEqual(
            resolve_workspace_path(copied_workspace, copied_config, "dataset_dir"),
            copied_dataset.resolve(),
        )
        self.assertTrue(diagnose_workspace(copied_workspace)["ok"])

    def test_migration_preserves_pass_and_review_files(self) -> None:
        prepare_workspace(load_pipeline_config(self.config_path))
        pass_path = self.workspace / "passes" / "pass1" / "portable_physics_0001.json"
        pass_path.write_text('{"sentinel":"keep"}\n', encoding="utf-8")
        state_path = self.workspace / "reviews" / "state.json"
        state_before = state_path.read_bytes()

        workspace_config = read_json(self.workspace / "workspace_config.json")
        workspace_config["dataset_dir"] = r"C:\old-computer\missing-dataset"
        (self.workspace / "workspace_config.json").write_text(
            json.dumps(workspace_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = migrate_workspace(self.workspace, self.dataset, self.records)
        self.assertFalse(Path(result["dataset_dir"]).is_absolute())
        self.assertEqual(pass_path.read_text(encoding="utf-8"), '{"sentinel":"keep"}\n')
        self.assertEqual(state_path.read_bytes(), state_before)
        self.assertTrue(list((self.workspace / "migrations").glob("path_migration_*.json")))


class CurrentWorkspaceRegressionTests(unittest.TestCase):
    def test_current_workspace_resolves_every_manifest_image(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = example_workspace(Path(temporary.name))
        config = read_json(workspace / "workspace_config.json")
        dataset_dir = resolve_workspace_path(workspace, config, "dataset_dir")
        manifest = read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl"))
        missing = [
            image["path"]
            for problem in manifest
            for image in problem.get("images", [])
            if not (dataset_dir / image["path"]).is_file()
        ]
        self.assertEqual(missing, [])
        state = AnnotationState(
            workspace,
            ROOT / "physgraph_review_app",
            ROOT / "physgraph_annotation" / "prompts",
        )
        self.assertEqual(state.dataset_dir, dataset_dir)
        self.assertEqual(
            state.media_paths,
            {
                image["path"].replace("\\", "/").lstrip("/")
                for problem in manifest
                for image in problem.get("images", [])
            },
        )


if __name__ == "__main__":
    unittest.main()
