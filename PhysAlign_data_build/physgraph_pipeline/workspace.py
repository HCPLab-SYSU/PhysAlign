"""Prepare, migrate, and export relocatable PhysGraph annotation workspaces."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import PhysicsDatasetAdapter, _safe_problem_id
from .config import ConfigError, PipelineConfig


from .paths import ROOT, SCRIPTS
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_annotation_lib import (  # noqa: E402
    STAGES,
    WORKSPACE_SCHEMA_VERSION,
    contract_json_schemas,
    effective_problem,
    json_sha256,
    portable_path,
    read_json,
    read_jsonl,
    resolve_workspace_path,
    validate_document,
    write_json_atomic,
    write_jsonl_atomic,
)
from prepare_physgraph_annotation import (  # noqa: E402
    assert_blind,
    clean_prompt,
    split_stem_query_options,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_media(dataset_dir: Path, manifest: list[dict[str, Any]]) -> None:
    """Verify declared image bytes before reusing an annotation workspace."""
    dataset_dir = dataset_dir.resolve()
    for problem in manifest:
        for image in problem.get("images", []):
            relative = str(image.get("path", "")).replace("\\", "/")
            path = (dataset_dir / relative).resolve()
            if not path.is_relative_to(dataset_dir) or not path.is_file():
                raise ConfigError(f"媒体文件缺失或路径逃逸 dataset_dir：{relative}")
            expected = image.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
                raise ConfigError(f"媒体 SHA-256 与 blind manifest 不一致：{relative}")


def _prompt_dir(config: PipelineConfig) -> Path:
    raw = config.annotation.get("prompt_dir")
    if isinstance(raw, str) and raw:
        path = config.resolve(raw, must_exist=True)
    else:
        path = ROOT / "physgraph_annotation" / "prompts"
    if not path.is_dir():
        raise ConfigError(f"prompt_dir 不是目录：{path}")
    return path


def _blind_record(problem: Any) -> dict[str, Any]:
    raw_question = clean_prompt(problem.raw_question)
    if problem.supplied_segments is None:
        segments, segmentation = split_stem_query_options(raw_question)
    else:
        segments = problem.supplied_segments
        segmentation = {
            "method": "configured_source_fields",
            "confidence": "high",
            "review_status": "machine_split",
            "edited": False,
        }
    return {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "problem_id": problem.problem_id,
        "source_record_index": problem.source_record_index,
        "source_dataset": problem.source_dataset,
        "source_sample_id": problem.source_sample_id,
        "source_split": problem.source_split,
        "language": problem.language,
        "images": problem.images,
        "raw_question": raw_question,
        "segments": segments,
        "segmentation": segmentation,
    }


def prepare_workspace(config: PipelineConfig, *, force: bool = False) -> dict[str, Any]:
    adapter = PhysicsDatasetAdapter(config)
    workspace = config.annotation_path("workspace")
    source_roots = {adapter.media_root, adapter.records_path.parent}
    if any(
        workspace == source_root
        or workspace in source_root.parents
        or source_root in workspace.parents
        for source_root in source_roots
    ):
        raise ConfigError("annotation workspace 不能与源数据目录重叠或互为父目录")
    manifest_path = workspace / "blind" / "manifest.jsonl"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"workspace 已存在：{manifest_path}；如只刷新 manifest，请显式加 --force"
        )

    records = [_blind_record(problem) for problem in adapter.load()]
    if len({item["problem_id"] for item in records}) != len(records):
        raise ConfigError("适配后出现重复 problem_id")
    assert_blind(records)
    if manifest_path.exists():
        # A refresh must never attach old approvals to different source data.
        previous = read_jsonl(manifest_path)
        source_hash = read_json(workspace / "summary.json").get("source_annotations_sha256")
        if records != previous or source_hash != sha256_file(adapter.records_path):
            raise ConfigError("源数据已变化；请建立新 workspace，不能通过 --force 复用旧审核记录")
    write_jsonl_atomic(manifest_path, records)

    for stage in STAGES:
        (workspace / "passes" / stage).mkdir(parents=True, exist_ok=True)
    (workspace / "reviews").mkdir(parents=True, exist_ok=True)
    (workspace / "exports").mkdir(parents=True, exist_ok=True)
    history_path = workspace / "reviews" / "history.jsonl"
    if not history_path.exists():
        history_path.write_text("", encoding="utf-8")
    state_path = workspace / "reviews" / "state.json"
    if not state_path.exists():
        write_json_atomic(
            state_path,
            {"workspace_schema_version": WORKSPACE_SCHEMA_VERSION, "problems": {}},
        )

    schema_dir = workspace / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for stage, schema in contract_json_schemas().items():
        write_json_atomic(schema_dir / f"{stage}.schema.json", schema)
    bundled_schema_dir = ROOT / "physgraph_annotation" / "schemas"
    if bundled_schema_dir.is_dir():
        for source in bundled_schema_dir.glob("*.schema.json"):
            shutil.copy2(source, schema_dir / source.name)

    prompts = workspace / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for source in _prompt_dir(config).glob("*.txt"):
        shutil.copy2(source, prompts / source.name)

    source_digest = sha256_file(adapter.records_path)
    model_policy = config.annotation.get(
        "model_policy",
        {
            "passes_1_to_4": "OpenAI-compatible multimodal model; configured at run time",
            "answers_visible": False,
        },
    )
    workspace_config = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "path_base": "workspace",
        "dataset_dir": portable_path(adapter.media_root, workspace),
        "source_annotations": portable_path(adapter.records_path, workspace),
        "pipeline_config": portable_path(config.path, workspace),
        "prompt_dir": "prompts",
        "manifest": "blind/manifest.jsonl",
        "model_policy": model_policy,
        "approval_gate": "Pass 4 must be valid and manually approved before export",
        "pipeline_config_sha256": config.sha256,
    }
    write_json_atomic(workspace / "workspace_config.json", workspace_config)
    summary = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "purpose": "answer-blind Pass 1-4 Observed-PhysGraph annotation",
        "source_annotations": portable_path(adapter.records_path, workspace),
        "source_annotations_sha256": source_digest,
        "source_dataset_dir": portable_path(adapter.media_root, workspace),
        "pipeline_config_sha256": config.sha256,
        "problems": len(records),
        "images": sum(len(item["images"]) for item in records),
        "sources": dict(Counter(item["source_dataset"] for item in records)),
        "segmentation_confidence": dict(
            Counter(item["segmentation"]["confidence"] for item in records)
        ),
        "answer_solution_present": False,
        "pass5_enabled": False,
    }
    write_json_atomic(workspace / "summary.json", summary)
    return {"workspace": str(workspace), **summary}


def migrate_workspace(
    workspace: Path,
    dataset_dir: Path,
    source_annotations: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    dataset_dir = dataset_dir.resolve()
    if not (workspace / "workspace_config.json").is_file():
        raise ConfigError(f"不是 PhysGraph workspace：{workspace}")
    if not dataset_dir.is_dir():
        raise ConfigError(f"dataset_dir 不存在：{dataset_dir}")
    if (
        workspace == dataset_dir
        or workspace in dataset_dir.parents
        or dataset_dir in workspace.parents
    ):
        raise ConfigError("workspace 与 dataset_dir 不能重叠或互为父目录")
    config = read_json(workspace / "workspace_config.json")
    manifest = read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl"))
    missing: list[str] = []
    for problem in manifest:
        for image in problem.get("images", []):
            relative = str(image.get("path", "")).replace("\\", "/")
            candidate = (dataset_dir / relative).resolve()
            if candidate != dataset_dir and dataset_dir not in candidate.parents:
                raise ConfigError(f"manifest 图片路径逃逸 dataset_dir：{relative}")
            if not candidate.is_file():
                missing.append(relative)
    if missing:
        raise ConfigError(f"新 dataset_dir 缺少 {len(missing)} 张图片：{missing[:5]}")
    verify_media(dataset_dir, manifest)

    if source_annotations is None:
        old_source = resolve_workspace_path(workspace, config, "source_annotations", must_exist=False)
        candidate = dataset_dir / old_source.name
        source_annotations = candidate if candidate.is_file() else old_source
        if not source_annotations.is_file():
            raise ConfigError("找不到源记录；请显式提供 --source-annotations")
    if source_annotations is not None:
        source_annotations = source_annotations.resolve()
        if not source_annotations.is_file():
            raise ConfigError(f"source annotations 不存在：{source_annotations}")
        expected = read_json(workspace / "summary.json").get("source_annotations_sha256")
        if expected and sha256_file(source_annotations) != expected:
            raise ConfigError("source annotations SHA-256 与 workspace summary 不一致")

    # Absolute legacy paths can contain personal account names. Migration
    # needs the new paths and content hashes, not copies of old user paths.
    config.pop("legacy_paths", None)
    config.update(
        {
            "path_base": "workspace",
            "dataset_dir": portable_path(dataset_dir, workspace),
        }
    )
    if source_annotations is not None:
        config["source_annotations"] = portable_path(source_annotations, workspace)
    write_json_atomic(workspace / "workspace_config.json", config)
    migration = {
        "schema_version": 1,
        "migrated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": ".",
        "dataset_dir": config["dataset_dir"],
        "source_annotations": config.get("source_annotations", ""),
        "manifest_problems": len(manifest),
        "manifest_images": sum(len(item.get("images", [])) for item in manifest),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_json_atomic(workspace / "migrations" / f"path_migration_{stamp}.json", migration)
    return migration


def export_approved_graphs(workspace: Path, output: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"输出已存在：{output}")
    config = read_json(workspace / "workspace_config.json")
    manifest = read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl"))
    by_id = {item["problem_id"]: item for item in manifest}
    if len(by_id) != len(manifest):
        raise ConfigError("blind manifest 包含重复 problem_id")
    state = read_json(workspace / "reviews" / "state.json")
    rows: list[dict[str, Any]] = []
    for problem_id, entry in state.get("problems", {}).items():
        if entry.get("exclusion", {}).get("status") == "excluded":
            continue
        pass4_state = entry.get("stages", {}).get("pass4", {})
        if pass4_state.get("status") != "approved" or problem_id not in by_id:
            continue
        _safe_problem_id(problem_id, 0)
        problem = effective_problem(by_id[problem_id], entry)
        repair = entry.get("segmentation_repair")
        if (problem.get("segmentation", {}).get("review_status") == "needs_review" or
                isinstance(repair, dict) and not repair.get("segmentation_confirmed", False)):
            raise ConfigError(f"题目切分尚未确认：{problem_id}")
        previous: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            approved = entry.get("stages", {}).get(stage, {})
            path = workspace / "passes" / stage / f"{problem_id}.json"
            if approved.get("status") != "approved" or not path.is_file():
                raise ConfigError(f"导出需要 Pass 1–4 全部批准：{problem_id}/{stage}")
            document = read_json(path)
            if approved.get("document_sha256") != json_sha256(document):
                raise ConfigError(f"批准后的内容已改变或缺少审核哈希：{problem_id}/{stage}")
            errors = [issue for issue in validate_document(stage, document, problem, previous)
                      if issue["level"] == "error"]
            if errors:
                raise ConfigError(f"已批准结果未通过当前校验：{problem_id}/{stage}")
            previous[stage] = document
        verify_media(resolve_workspace_path(workspace, config, "dataset_dir"), [problem])
        rows.append(
            {
                "problem_id": problem_id,
                "source_dataset": problem.get("source_dataset", ""),
                "source_sample_id": problem.get("source_sample_id", ""),
                "source_split": problem.get("source_split", ""),
                "language": problem.get("language", ""),
                "images": problem.get("images", []),
                "observed_physgraph": previous["pass4"],
                "review": {
                    "status": "approved",
                    "reviewer": pass4_state.get("reviewer", ""),
                    "approved_at_utc": pass4_state.get("updated_at_utc", ""),
                    "document_sha256": pass4_state.get("document_sha256", ""),
                },
            }
        )
    if not rows:
        raise ConfigError("没有可导出的已批准 Pass 4")
    write_jsonl_atomic(output, rows)
    return {
        "output": str(output),
        "records": len(rows),
        "sha256": sha256_file(output),
    }
