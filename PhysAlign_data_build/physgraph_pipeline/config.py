"""Versioned JSON configuration for portable physics-dataset annotation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """A configuration error with a user-actionable message."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} 必须是 JSON object")
    return value


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    data: dict[str, Any]

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def dataset(self) -> dict[str, Any]:
        return _mapping(self.data.get("dataset"), "$.dataset")

    @property
    def annotation(self) -> dict[str, Any]:
        return _mapping(self.data.get("annotation"), "$.annotation")

    def resolve(self, value: str, *, must_exist: bool = False) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("路径必须是非空字符串")
        raw = Path(value)
        resolved = raw.resolve() if raw.is_absolute() else (self.directory / raw).resolve()
        if must_exist and not resolved.exists():
            raise ConfigError(f"配置路径不存在：{resolved}")
        return resolved

    def dataset_path(self, key: str, *, must_exist: bool = False) -> Path:
        value = self.dataset.get(key)
        if not isinstance(value, str):
            raise ConfigError(f"$.dataset.{key} 必须是路径字符串")
        return self.resolve(value, must_exist=must_exist)

    def annotation_path(self, key: str, *, must_exist: bool = False) -> Path:
        value = self.annotation.get(key)
        if not isinstance(value, str):
            raise ConfigError(f"$.annotation.{key} 必须是路径字符串")
        return self.resolve(value, must_exist=must_exist)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_pipeline_config(path: Path) -> PipelineConfig:
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"配置文件不存在：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"配置 JSON 语法错误：{path}:{error.lineno}:{error.colno} {error.msg}"
        ) from error
    data = _mapping(value, "$")
    if data.get("schema_version") != 1:
        raise ConfigError("$.schema_version 当前必须为 1")
    dataset = _mapping(data.get("dataset"), "$.dataset")
    annotation = _mapping(data.get("annotation"), "$.annotation")
    for location, mapping, keys in (
        ("$.dataset", dataset, ("name", "records", "media_root")),
        ("$.annotation", annotation, ("workspace",)),
    ):
        for key in keys:
            if not isinstance(mapping.get(key), str) or not mapping[key].strip():
                raise ConfigError(f"{location}.{key} 必须是非空字符串")
    fields = _mapping(dataset.get("fields"), "$.dataset.fields")
    if not isinstance(fields.get("id"), str):
        raise ConfigError("$.dataset.fields.id 必须是 JSON Pointer")
    if not isinstance(fields.get("question"), str) and not isinstance(
        fields.get("messages"), str
    ):
        raise ConfigError("fields.question 和 fields.messages 至少配置一个")
    source_format = dataset.get("format", "auto")
    if source_format not in {"auto", "json", "jsonl"}:
        raise ConfigError("$.dataset.format 仅允许 auto/json/jsonl")
    return PipelineConfig(path=path, data=data)
