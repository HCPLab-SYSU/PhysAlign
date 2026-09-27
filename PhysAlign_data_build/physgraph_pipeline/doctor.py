"""Read-only diagnostics for a dataset configuration or annotation workspace."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .adapters import PhysicsDatasetAdapter
from .config import ConfigError, PipelineConfig
from .workspace import verify_media


from .paths import ROOT, SCRIPTS
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_annotation_lib import (  # noqa: E402
    read_json,
    read_jsonl,
    resolve_source_annotations,
    resolve_workspace_path,
)


def _dependency(name: str, import_name: str, required_for: str) -> dict[str, Any]:
    return {
        "name": name,
        "available": importlib.util.find_spec(import_name) is not None,
        "required_for": required_for,
    }


def diagnose_workspace(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"workspace": str(workspace)}
    config_path = workspace / "workspace_config.json"
    if not config_path.is_file():
        return {**details, "ok": False, "errors": [f"缺少 {config_path}"], "warnings": []}
    try:
        config = read_json(config_path)
        dataset_dir = resolve_workspace_path(workspace, config, "dataset_dir")
        details["dataset_dir"] = str(dataset_dir)
        try:
            details["source_annotations"] = str(resolve_source_annotations(workspace, config))
        except (ValueError, FileNotFoundError) as error:
            warnings.append(str(error))
        manifest_path = workspace / config.get("manifest", "blind/manifest.jsonl")
        manifest = read_jsonl(manifest_path)
        if not manifest:
            errors.append(f"blind manifest 为空：{manifest_path}")
        missing_media: list[str] = []
        escaping_media: list[str] = []
        for problem in manifest:
            for image in problem.get("images", []):
                relative = str(image.get("path", "")).replace("\\", "/")
                candidate = (dataset_dir / relative).resolve()
                if candidate != dataset_dir and dataset_dir not in candidate.parents:
                    escaping_media.append(relative)
                elif not candidate.is_file():
                    missing_media.append(relative)
        if escaping_media:
            errors.append(f"{len(escaping_media)} 个媒体路径逃逸 dataset root")
        if missing_media:
            errors.append(f"缺少 {len(missing_media)} 个媒体文件，例如 {missing_media[:3]}")
        if not missing_media and not escaping_media:
            try:
                verify_media(dataset_dir, manifest)
            except ConfigError as error:
                errors.append(str(error))
        details.update(
            {
                "problems": len(manifest),
                "images": sum(len(item.get("images", [])) for item in manifest),
                "missing_media": len(missing_media),
                "pass_files": {
                    stage: len(list((workspace / "passes" / stage).glob("*.json")))
                    for stage in ("pass1", "pass2", "pass3", "pass4", "pass5")
                },
            }
        )
        if config.get("path_base") != "workspace":
            warnings.append("workspace 尚未声明 path_base=workspace，建议执行 migrate")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    return {**details, "ok": not errors, "errors": errors, "warnings": warnings}


def diagnose_config(config: PipelineConfig) -> dict[str, Any]:
    errors: list[str] = []
    details: dict[str, Any] = {"config": str(config.path)}
    try:
        adapter = PhysicsDatasetAdapter(config)
        records = adapter.load()
        details.update(
            {
                "dataset": config.dataset["name"],
                "records": len(records),
                "images": sum(len(item.images) for item in records),
                "workspace": str(config.annotation_path("workspace")),
                "media_root": str(adapter.media_root),
            }
        )
    except (ConfigError, OSError, ValueError) as error:
        errors.append(str(error))
    return {**details, "ok": not errors, "errors": errors, "warnings": []}


def doctor_report(
    *, config: PipelineConfig | None = None, workspace: Path | None = None
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if config is not None:
        checks.append(diagnose_config(config))
        configured_workspace = config.annotation_path("workspace")
        if (configured_workspace / "workspace_config.json").is_file():
            checks.append(diagnose_workspace(configured_workspace))
    if workspace is not None:
        checks.append(diagnose_workspace(workspace))
    dependencies = [
        _dependency("Pillow", "PIL", "缺少图片尺寸 metadata 时的数据适配、几何与叠加图"),
        _dependency("NumPy", "numpy", "像素级几何闭环"),
        _dependency("httpx", "httpx", "API 标注"),
        _dependency("OpenAI SDK", "openai", "API 标注"),
    ]
    return {
        "ok": bool(checks) and all(check["ok"] for check in checks),
        "python": {
            "version": platform.python_version(),
            "supported": sys.version_info >= (3, 11),
            "executable": sys.executable,
        },
        "dependencies": dependencies,
        "checks": checks,
    }
