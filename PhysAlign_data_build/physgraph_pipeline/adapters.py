"""Configuration-driven adapter for JSON/JSONL multimodal physics problems."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import ConfigError, PipelineConfig


MISSING = object()
WINDOWS_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def json_pointer_get(value: Any, pointer: str, default: Any = MISSING) -> Any:
    if pointer == "":
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ConfigError(f"无效 JSON Pointer：{pointer!r}")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                current = current[int(token)]
            elif isinstance(current, dict):
                current = current[token]
            else:
                raise KeyError(token)
        except (KeyError, IndexError, ValueError):
            if default is not MISSING:
                return default
            raise ConfigError(f"数据中找不到 JSON Pointer：{pointer}") from None
    return current


def read_records(path: Path, source_format: str, records_pointer: str = "") -> list[dict[str, Any]]:
    if source_format == "auto":
        source_format = "jsonl" if path.suffix.lower() == ".jsonl" else "json"
    try:
        if source_format == "jsonl":
            records: Any = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            records = json.loads(path.read_text(encoding="utf-8"))
            if records_pointer:
                records = json_pointer_get(records, records_pointer)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"数据 JSON 语法错误：{path}:{error.lineno}:{error.colno} {error.msg}"
        ) from error
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ConfigError("数据源必须解析为 JSON object 数组")
    if not records:
        raise ConfigError("数据源为空")
    return records


def _optional(record: dict[str, Any], pointer: Any, default: Any = "") -> Any:
    if not isinstance(pointer, str) or not pointer:
        return default
    return json_pointer_get(record, pointer, default)


def _safe_problem_id(value: Any, index: int) -> str:
    if value is None or isinstance(value, (dict, list, bool)):
        raise ConfigError(f"第 {index} 条记录的 id 必须是字符串或数字")
    problem_id = str(value).strip()
    if not problem_id:
        raise ConfigError(f"第 {index} 条记录的 id 为空")
    if len(problem_id) > 180 or problem_id in {".", ".."} or WINDOWS_UNSAFE.search(problem_id):
        raise ConfigError(
            f"第 {index} 条记录的 id 不能安全用作文件名：{problem_id!r}"
        )
    if problem_id.endswith((" ", ".")):
        raise ConfigError(f"第 {index} 条记录的 id 不能以空格或点结尾")
    if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", problem_id.split(".")[0]):
        raise ConfigError(f"第 {index} 条记录的 id 是 Windows 保留文件名")
    return problem_id


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _question(record: dict[str, Any], fields: dict[str, Any], index: int) -> str:
    direct = _optional(record, fields.get("question"), MISSING)
    if direct is not MISSING:
        if not isinstance(direct, str) or not direct.strip():
            raise ConfigError(f"第 {index} 条记录的 question 不是非空字符串")
        return direct
    messages = _optional(record, fields.get("messages"), None)
    if not isinstance(messages, list):
        raise ConfigError(f"第 {index} 条记录的 messages 不是数组")
    role_key = fields.get("message_role_key", "from")
    text_key = fields.get("message_text_key", "value")
    user_roles = set(fields.get("user_roles", ["human", "user"]))
    for message in messages:
        if isinstance(message, dict) and message.get(role_key) in user_roles:
            text = _message_text(message.get(text_key))
            if text.strip():
                return text
    raise ConfigError(f"第 {index} 条记录找不到 user/human 问题文本")


def _normalize_images(value: Any, index: int) -> list[str]:
    if value in (None, ""):
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
        raise ConfigError(f"第 {index} 条记录的 images 必须是字符串或字符串数组")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int, int, str]:
    try:
        from PIL import Image
    except ImportError as error:
        raise ConfigError(
            f"图片 metadata 缺少尺寸，读取 {path.name} 需要 Pillow；请先安装项目依赖"
        ) from error
    with Image.open(path) as image:
        return int(image.width), int(image.height), str(image.format or path.suffix[1:]).upper()


def _media_path(media_root: Path, raw: str, problem_id: str) -> tuple[Path, str]:
    path = Path(raw)
    candidate = path.resolve() if path.is_absolute() else (media_root / path).resolve()
    if candidate != media_root and media_root not in candidate.parents:
        raise ConfigError(f"{problem_id} 的图片路径逃逸 media_root：{raw}")
    if not candidate.is_file():
        raise ConfigError(f"{problem_id} 的图片不存在：{candidate}")
    return candidate, candidate.relative_to(media_root).as_posix()


@dataclass(frozen=True)
class AdaptedProblem:
    problem_id: str
    source_record_index: int
    source_dataset: str
    source_sample_id: str
    source_split: str
    language: str
    raw_question: str
    images: list[dict[str, Any]]
    supplied_segments: dict[str, Any] | None


class PhysicsDatasetAdapter:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dataset = config.dataset
        self.fields = self.dataset["fields"]
        self.records_path = config.dataset_path("records", must_exist=True)
        self.media_root = config.dataset_path("media_root", must_exist=True)
        if not self.media_root.is_dir():
            raise ConfigError(f"media_root 不是目录：{self.media_root}")

    def load(self) -> list[AdaptedProblem]:
        records = read_records(
            self.records_path,
            self.dataset.get("format", "auto"),
            self.dataset.get("records_pointer", ""),
        )
        output = [self._adapt(record, index) for index, record in enumerate(records)]
        ids = [item.problem_id.casefold() for item in output]
        if len(ids) != len(set(ids)):
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            raise ConfigError(f"数据集包含重复 problem id：{duplicates[:10]}")
        return output

    def _adapt(self, record: dict[str, Any], index: int) -> AdaptedProblem:
        problem_id = _safe_problem_id(json_pointer_get(record, self.fields["id"]), index)
        raw_question = _question(record, self.fields, index)
        metadata_images = _optional(record, self.fields.get("metadata_images"), [])
        if metadata_images in (None, ""):
            metadata_images = []
        if not isinstance(metadata_images, list):
            raise ConfigError(f"{problem_id} 的 metadata_images 必须是数组")
        raw_images = _optional(record, self.fields.get("images"), MISSING)
        if raw_images is MISSING:
            raw_images = [
                item.get("path")
                for item in metadata_images
                if isinstance(item, dict) and item.get("path")
            ]
        image_values = _normalize_images(raw_images, index)
        images: list[dict[str, Any]] = []
        for image_index, raw in enumerate(image_values):
            path, relative = _media_path(self.media_root, raw, problem_id)
            metadata = metadata_images[image_index] if image_index < len(metadata_images) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            actual_sha = _sha256(path)
            declared_sha = str(metadata.get("sha256", ""))
            if declared_sha and declared_sha != actual_sha:
                raise ConfigError(f"{problem_id} 图片 {image_index + 1} 的 SHA-256 不匹配")
            width = int(metadata.get("width", 0) or 0)
            height = int(metadata.get("height", 0) or 0)
            image_format = str(metadata.get("format", "") or "").upper()
            if width <= 0 or height <= 0 or not image_format:
                detected_width, detected_height, detected_format = _image_size(path)
                width = width or detected_width
                height = height or detected_height
                image_format = image_format or detected_format
            images.append(
                {
                    "image_id": f"img_{image_index}",
                    "path": relative,
                    "width": width,
                    "height": height,
                    "sha256": actual_sha,
                    "format": image_format,
                    "bytes": path.stat().st_size,
                }
            )

        supplied_segments = None
        stem_pointer = self.fields.get("stem")
        query_pointer = self.fields.get("query")
        if isinstance(stem_pointer, str) or isinstance(query_pointer, str):
            stem = str(_optional(record, stem_pointer, "") or "").strip()
            query = str(_optional(record, query_pointer, "") or "").strip()
            raw_options = _optional(record, self.fields.get("options"), [])
            options: list[dict[str, str]] = []
            if raw_options not in (None, ""):
                if not isinstance(raw_options, list):
                    raise ConfigError(f"{problem_id} 的 options 必须是数组")
                for option_index, option in enumerate(raw_options):
                    if isinstance(option, dict):
                        label = str(option.get("label", chr(65 + option_index)))
                        text = str(option.get("text", option.get("value", "")))
                    else:
                        label, text = chr(65 + option_index), str(option)
                    options.append({"label": label, "text": text})
            if not stem and not query:
                raise ConfigError(f"{problem_id} 的 stem/query 不能同时为空")
            supplied_segments = {"stem": stem, "query": query, "options": options}

        return AdaptedProblem(
            problem_id=problem_id,
            source_record_index=index,
            source_dataset=str(self.dataset.get("name", "")),
            source_sample_id=str(
                _optional(record, self.fields.get("source_sample_id"), problem_id)
                or problem_id
            ),
            source_split=str(_optional(record, self.fields.get("source_split"), "") or ""),
            language=str(_optional(record, self.fields.get("language"), "") or ""),
            raw_question=raw_question,
            images=images,
            supplied_segments=supplied_segments,
        )


def walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)
