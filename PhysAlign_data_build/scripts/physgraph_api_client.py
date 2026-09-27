#!/usr/bin/env python
"""Shared, secret-safe OpenAI-compatible API client for PhysGraph annotation."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from openai import APIStatusError, OpenAI


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
SUPPORTED_API_MODES = {"chat", "responses", "auto"}


class PhysGraphAPIError(RuntimeError):
    """A sanitized API or response error safe to persist in logs."""


class PhysGraphAPIResponseError(PhysGraphAPIError):
    """A returned response that could not be parsed as the required JSON."""

    def __init__(self, message: str, result: "APIResult") -> None:
        super().__init__(message)
        self.result = result


class ExclusiveRunLock:
    """Prevent two annotation workers from spending API tokens concurrently."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.token = uuid.uuid4().hex
        self.fd: int | None = None

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        if os.name == "nt":
            # On Windows os.kill(pid, 0) terminates the process. Query its
            # status through a read-only handle instead; uncertainty keeps
            # the lock in place.
            import _winapi

            try:
                handle = _winapi.OpenProcess(0x1000, False, pid)
            except OSError as error:
                return error.winerror != 87  # ERROR_INVALID_PARAMETER: no PID
            try:
                return _winapi.GetExitCodeProcess(handle) == _winapi.STILL_ACTIVE
            except OSError:
                return True
            finally:
                _winapi.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner = {"pid": os.getpid(), "token": self.token, "created_at_epoch": time.time()}
        for _ in range(2):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing_text = self.path.read_text(encoding="utf-8")
                    existing = json.loads(existing_text)
                    existing_pid = int(existing.get("pid", 0))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    existing_text = ""
                    existing_pid = 0
                if self._pid_is_running(existing_pid):
                    raise PhysGraphAPIError(
                        f"已有标注进程持有运行锁（PID {existing_pid}）：{self.path}"
                    )
                # Delete a stale lock only if it has not changed since inspection.
                try:
                    if self.path.read_text(encoding="utf-8") == existing_text:
                        self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            os.write(self.fd, (json.dumps(owner) + "\n").encode("utf-8"))
            os.fsync(self.fd)
            return
        raise PhysGraphAPIError(f"无法安全取得标注运行锁：{self.path}")

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if existing.get("token") == self.token:
                self.path.unlink()
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass


@dataclass(frozen=True)
class APIResult:
    text: str
    api_mode: str
    request_id: str
    model: str
    usage: dict[str, int]
    duration_seconds: float
    status: str

    def log_record(self) -> dict[str, Any]:
        return {
            "ok": True,
            "api_mode": self.api_mode,
            "request_id": self.request_id,
            "model": self.model,
            "usage": self.usage,
            "duration_seconds": round(self.duration_seconds, 3),
            "status": self.status,
        }


def normalize_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"https", "http"} or not parsed.netloc:
        raise ValueError("API base URL必须是合法的http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("API base URL不得包含用户名、密码、query或fragment")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("非本地API base URL必须使用HTTPS")
    return base_url


def safe_endpoint_label(base_url: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def load_api_key(env_name: str = DEFAULT_API_KEY_ENV) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise PhysGraphAPIError(
            f"环境变量{env_name}未设置。请通过进程环境提供密钥；不要写入脚本、配置、日志或提交记录。"
        )
    return key


def image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise PhysGraphAPIError("API响应中没有可解析的JSON object")
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PhysGraphAPIError(f"API响应不是合法JSON：{exc.msg}") from exc
    if not isinstance(value, dict):
        raise PhysGraphAPIError("API响应顶层不是JSON object")
    return value


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _prompt(root: Path, workspace: Path, name: str) -> str:
    """Load a public prompt copied into the workspace, with a package fallback."""
    candidates = (
        workspace / "prompts" / f"{name}.txt",
        root / "physgraph_annotation" / "prompts" / f"{name}.txt",
    )
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    raise PhysGraphAPIError(
        f"缺少公开提示词 {name}.txt；请重新运行 prepare 生成完整工作区"
    )


def _annotation_policy(root: Path, workspace: Path, stages: Iterable[str]) -> str:
    sections = [_prompt(root, workspace, "system")]
    sections.extend(_prompt(root, workspace, stage) for stage in stages)
    return "\n\n".join(sections)


def load_gobs_system_instructions(root: Path, workspace: Path) -> str:
    policy = _annotation_policy(root, workspace, ("pass1", "pass2", "pass3", "pass4"))
    schemas = []
    for stage in ("pass1", "pass2", "pass3", "pass4"):
        schema = json.loads((workspace / "schemas" / f"{stage}.schema.json").read_text(encoding="utf-8"))
        schemas.append(f"{stage.upper()} JSON Schema:\n{compact_json(schema)}")
    return (
        "你是API模式下的PhysGraph标注器。以下标注规范和JSON Schema是本次请求中唯一权威的本地规则。"
        "你不能访问本地文件或工具，所以必须直接依据这里提供的规则和用户消息中的题目、题图完成任务。\n\n"
        f"公开标注规范：\n{policy}\n\n"
        + "\n\n".join(schemas)
    )


def load_pass1_system_instructions(root: Path, workspace: Path) -> str:
    policy = _annotation_policy(root, workspace, ("pass1",))
    schema = json.loads((workspace / "schemas" / "pass1_pixel.schema.json").read_text(encoding="utf-8"))
    return (
        "你是API模式下的PhysGraph Pass 1面板视觉提取器。你只生成当前裁剪面板的像素空间视觉节点，"
        "不创建任何物理实体、物理量、关系或约束。最终数据使用0–1000归一化坐标；"
        "但本请求是防止坐标换算错误的内部捕获阶段，必须严格按裁剪面板的原始像素坐标输出bbox_px、"
        "keypoints_px、center_px和radius_px。之后由本地程序确定性映射到原图并归一化，禁止你自行归一化。"
        "以下规则和内部Schema是唯一权威要求。\n\n"
        f"Pass 1规则：\n{policy.strip()}\n\n"
        f"PASS1 PIXEL JSON Schema：\n{compact_json(schema)}"
    )


def load_pass1_correction_system_instructions(workspace: Path) -> str:
    schema = json.loads(
        (workspace / "schemas" / "pass1_pixel_correction.schema.json").read_text(encoding="utf-8")
    )
    pixel_schema = json.loads(
        (workspace / "schemas" / "pass1_pixel.schema.json").read_text(encoding="utf-8")
    )
    schema["$defs"] = {"pixel_node": pixel_schema["$defs"]["pixel_node"]}
    schema["properties"]["operations"]["items"]["properties"]["node"]["oneOf"][1] = {
        "$ref": "#/$defs/pixel_node"
    }
    return (
        "你是PhysGraph Pass 1面板视觉定位复核器。你会同时看到原始面板、带像素网格的同尺寸面板和候选节点叠加图。"
        "你的任务不是重新解题，也不是重新输出完整Pass 1，而是逐节点检查候选框、关键点、圆心、半径、OCR文本和视觉类型，"
        "然后只输出最小纠错补丁。所有坐标必须是当前裁剪面板左上角为原点的原始像素整数；"
        "必须依据面板给出的crop_size_px检查边界，禁止输出0–1000归一化坐标、网页截图坐标或原图全局坐标。"
        "accept必须对应空operations；correct必须至少包含一个operation。replace必须提供完整替换节点且target_id为现有ID；"
        "delete的node必须为null；append的target_id必须为空字符串且node必须为完整新节点。"
        "append节点的id只是面板内非空临时标签；本地程序会忽略其全局编号并确定性分配无冲突ID，禁止为了猜全局ID而改变几何或语义。"
        "不得因为边界框包含曲线内部空白就误删曲线；应结合keypoints判断。看不清时降低confidence，禁止伪精确。\n\n"
        f"像素纠错补丁JSON Schema：\n{compact_json(schema)}"
    )


def load_downstream_system_instructions(root: Path, workspace: Path) -> str:
    policy = _annotation_policy(root, workspace, ("pass2", "pass3", "pass4"))
    schemas: list[str] = []
    for stage in ("pass2", "pass3", "pass4"):
        schema = json.loads((workspace / "schemas" / f"{stage}.schema.json").read_text(encoding="utf-8"))
        schemas.append(f"{stage.upper()} JSON Schema：\n{compact_json(schema)}")
    return (
        "你是API模式下的PhysGraph Pass 2–4标注器。Pass 1已经由独立视觉流程生成并复核，绝对不得修改。"
        "以下规则和Schema是唯一权威要求。\n\n"
        f"公开 Pass 2–4 规则：\n{policy}\n\n"
        + "\n\n".join(schemas)
    )


def load_pass5_system_instructions(root: Path, workspace: Path) -> str:
    policy = _prompt(root, workspace, "pass5")
    schema = json.loads((workspace / "schemas" / "pass5.schema.json").read_text(encoding="utf-8"))
    return (
        "你是API模式下的PhysGraph Pass 5解答步骤对齐器。不得重新解题。"
        "以下规则和Schema是唯一权威要求。\n\n"
        f"Pass 5规则：\n{policy.strip()}\n\n"
        f"PASS5 JSON Schema：\n{compact_json(schema)}"
    )


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump(exclude_none=True)
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {}
    output: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, int):
            output[key] = value
    return output


def merge_usage(total: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        total[key] = total.get(key, 0) + value


def _chat_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "".join(parts)
    return ""


class PhysGraphAPIClient:
    def __init__(
        self,
        *,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url: str | None = None,
        api_mode: str = "chat",
        timeout_seconds: float = 900.0,
        transport_retries: int = 2,
    ) -> None:
        if api_mode not in SUPPORTED_API_MODES:
            raise ValueError(f"Unsupported API mode: {api_mode}")
        self.api_key_env = api_key_env
        self.api_key = load_api_key(api_key_env)
        configured_url = base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = normalize_base_url(configured_url)
        self.api_mode = api_mode
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 30.0)),
            max_retries=transport_retries,
        )

    @property
    def endpoint_label(self) -> str:
        return safe_endpoint_label(self.base_url)

    def _sanitize_error(self, exc: Exception) -> PhysGraphAPIError:
        message = str(exc).replace(self.api_key, "[REDACTED]")
        message = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", message, flags=re.IGNORECASE)
        return PhysGraphAPIError(f"{type(exc).__name__}: {message[:1500]}")

    def _responses_request(
        self,
        *,
        model: str,
        reasoning_effort: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[Path],
        image_detail: str,
        max_output_tokens: int,
    ) -> APIResult:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_data_url(path),
                    "detail": image_detail,
                }
            )
        started = time.monotonic()
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
            text={"format": {"type": "json_object"}},
        )
        text = response.output_text or ""
        if not text.strip():
            raise PhysGraphAPIError("Responses API返回了空文本")
        return APIResult(
            text=text,
            api_mode="responses",
            request_id=str(getattr(response, "_request_id", "") or ""),
            model=str(getattr(response, "model", model) or model),
            usage=_usage_dict(getattr(response, "usage", None)),
            duration_seconds=time.monotonic() - started,
            status=str(getattr(response, "status", "completed") or "completed"),
        )

    def _chat_request(
        self,
        *,
        model: str,
        reasoning_effort: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[Path],
        image_detail: str,
        max_output_tokens: int,
    ) -> APIResult:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(path), "detail": image_detail},
                }
            )
        started = time.monotonic()
        user_content: str | list[dict[str, Any]] = content if len(content) > 1 else user_prompt
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_output_tokens,
            response_format={"type": "json_object"},
        )
        if not response.choices:
            raise PhysGraphAPIError("Chat Completions API没有返回choice")
        text = _chat_text(response.choices[0].message.content)
        if not text.strip():
            raise PhysGraphAPIError("Chat Completions API返回了空文本")
        return APIResult(
            text=text,
            api_mode="chat",
            request_id=str(getattr(response, "_request_id", "") or ""),
            model=str(getattr(response, "model", model) or model),
            usage=_usage_dict(getattr(response, "usage", None)),
            duration_seconds=time.monotonic() - started,
            status=str(getattr(response.choices[0], "finish_reason", "completed") or "completed"),
        )

    def request_json(
        self,
        *,
        model: str,
        reasoning_effort: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[Path] = (),
        image_detail: str = "original",
        max_output_tokens: int = 65536,
    ) -> tuple[dict[str, Any], APIResult]:
        paths = [Path(path) for path in image_paths]
        for path in paths:
            if not path.is_file():
                raise PhysGraphAPIError(f"题图不存在：{path}")
        try:
            if self.api_mode == "responses":
                result = self._responses_request(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=paths,
                    image_detail=image_detail,
                    max_output_tokens=max_output_tokens,
                )
            elif self.api_mode == "chat":
                result = self._chat_request(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=paths,
                    image_detail=image_detail,
                    max_output_tokens=max_output_tokens,
                )
            else:
                try:
                    result = self._responses_request(
                        model=model,
                        reasoning_effort=reasoning_effort,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_paths=paths,
                        image_detail=image_detail,
                        max_output_tokens=max_output_tokens,
                    )
                except APIStatusError as exc:
                    if exc.status_code not in {404, 405, 501}:
                        raise
                    result = self._chat_request(
                        model=model,
                        reasoning_effort=reasoning_effort,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_paths=paths,
                        image_detail=image_detail,
                        max_output_tokens=max_output_tokens,
                    )
            try:
                document = extract_json_object(result.text)
            except PhysGraphAPIError as exc:
                # The provider may have billed this response even though it is
                # malformed, so retain request/usage metadata for accounting.
                raise PhysGraphAPIResponseError(str(exc), result) from exc
            return document, result
        except PhysGraphAPIError:
            raise
        except Exception as exc:
            raise self._sanitize_error(exc) from exc


def write_attempt_log(
    path: Path,
    *,
    endpoint: str,
    result: APIResult | None = None,
    error: Exception | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {"endpoint": endpoint}
    if result is not None:
        record.update(result.log_record())
    if error is not None:
        record.update(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error)[:1500],
            }
        )
    elif result is None:
        record.update(
            {
                "ok": False,
                "error_type": "UnknownError",
                "error": "unknown error",
            }
        )
    if extra:
        record.update(extra)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
