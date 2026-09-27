#!/usr/bin/env python
"""Shared contracts and validation for the local PhysGraph annotation workflow."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "physgraph_obs_v0.1"
WORKSPACE_SCHEMA_VERSION = 1
STAGES = ("pass1", "pass2", "pass3", "pass4")
STAGE_LABELS = {
    "pass1": "OCR 与视觉图元",
    "pass2": "图元—物理实体绑定",
    "pass3": "题干 grounding",
    "pass4": "Observed-PhysGraph 合并",
}

VISUAL_TYPES = (
    "text_glyph",
    "point_mark",
    "line_segment",
    "polyline",
    "curve",
    "circle",
    "arc",
    "arrow",
    "angle_mark",
    "region",
    "axis",
    "graph_curve",
    "object_shape",
    "component_symbol",
    "field_symbol",
    "unknown_visual",
)
PHYSICAL_TYPES = (
    "point",
    "body",
    "particle",
    "surface",
    "inclined_plane",
    "rod",
    "rope",
    "spring",
    "pulley",
    "pivot",
    "trajectory",
    "force",
    "velocity",
    "acceleration",
    "charge",
    "electric_field",
    "magnetic_field",
    "wire",
    "junction",
    "battery",
    "resistor",
    "switch",
    "ammeter",
    "voltmeter",
    "capacitor",
    "coil",
    "conductor_rod",
    "event",
    "system",
    "unknown_physical",
)
DOMAINS = ("mechanics", "electromagnetism", "mixed")
CONFIDENCE = ("high", "medium", "low")
PROVENANCE = ("IMAGE", "TEXT")
BINDING_TYPES = ("labels", "represents", "same_entity_as", "refers_to")
MENTION_SECTIONS = ("stem", "query")
MENTION_ROLES = ("entity", "quantity", "constraint", "relation", "query_target", "other")

ID_PATTERNS = {
    "visual": re.compile(r"^v\d{3,}$"),
    "physical": re.compile(r"^p\d{3,}$"),
    "mention": re.compile(r"^m\d{3,}$"),
    "quantity": re.compile(r"^q\d{3,}$"),
    "binding": re.compile(r"^b\d{3,}$"),
    "relation": re.compile(r"^r\d{3,}$"),
    "constraint": re.compile(r"^c\d{3,}$"),
    "ambiguity": re.compile(r"^u\d{3,}$"),
}

PROHIBITED_ANNOTATION_KEYS = {
    "answer",
    "answers",
    "correct_answer",
    "final_answer",
    "reasoning",
    "solution",
    "standard_solution",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_ATOMIC_REPLACE_ATTEMPTS = 20
_ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.05
_ATOMIC_REPLACE_MAX_DELAY_SECONDS = 1.0
_RETRYABLE_REPLACE_ERRNOS = {errno.EACCES, errno.EBUSY, errno.EPERM}
_RETRYABLE_WINDOWS_REPLACE_ERRORS = {5, 32, 33}


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Atomically replace ``path``, tolerating transient Windows file handles.

    Editors, antivirus scanners, and file indexers can briefly open a JSON file
    without ``FILE_SHARE_DELETE``.  During that window ``os.replace`` raises
    WinError 5/32 even though the directory ACL is valid.  Retrying the atomic
    operation preserves the old valid document until the new one can be swapped
    in; it never falls back to an in-place, partially visible write.
    """

    delay = _ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            if attempt:
                print(
                    f"[atomic-write] recovered after {attempt} retries: {path}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        except OSError as exc:
            retryable = (
                isinstance(exc, PermissionError)
                or exc.errno in _RETRYABLE_REPLACE_ERRNOS
                or getattr(exc, "winerror", None) in _RETRYABLE_WINDOWS_REPLACE_ERRORS
            )
            if not retryable or attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                raise
            if attempt == 0:
                print(
                    f"[atomic-write] target temporarily busy; retrying: {path}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(delay)
            delay = min(delay * 2, _ATOMIC_REPLACE_MAX_DELAY_SECONDS)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # A failed replace remains visible through the raised original
            # exception; cleanup failure must not hide that root cause.
            pass


def write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def write_jsonl_atomic(path: Path, values: Iterable[dict[str, Any]]) -> None:
    # Preserve streaming behaviour for manifests that may be much larger than
    # ordinary JSON documents; only the final replace needs the retry policy.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        _replace_with_retry(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SEGMENTATION_REPAIR_STATUSES = (
    "downstream_reannotation_required",
    "generated_pending_review",
    "offline_recovered_pending_review",
    "resolved",
)


def effective_problem(
    problem: dict[str, Any],
    review_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a blind problem with its audited segmentation overlay applied.

    The blind manifest remains immutable.  Every consumer that can generate or
    validate Pass 2--4 must use this view so the review UI, batch runner and
    training export cannot silently disagree about the question text.
    """

    output = deepcopy(problem)
    override = (review_entry or {}).get("segmentation_override")
    if isinstance(override, dict):
        segments = override.get("segments")
        segmentation = override.get("segmentation")
        if isinstance(segments, dict) and isinstance(segmentation, dict):
            output["segments"] = deepcopy(segments)
            output["segmentation"] = deepcopy(segmentation)
    return output


def effective_manifest(
    manifest: Iterable[dict[str, Any]],
    review_state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Apply review-state segmentation overlays to a manifest."""

    problems = (review_state or {}).get("problems", {})
    if not isinstance(problems, dict):
        problems = {}
    return [
        effective_problem(problem, problems.get(problem.get("problem_id"), {}))
        for problem in manifest
    ]


def segmentation_repair_status(review_entry: dict[str, Any] | None) -> str:
    repair = (review_entry or {}).get("segmentation_repair")
    if not isinstance(repair, dict):
        return ""
    status = repair.get("status")
    return status if isinstance(status, str) else ""


def problem_exclusion_status(review_entry: dict[str, Any] | None) -> str:
    """Return the durable annotation/training exclusion state for a problem."""

    exclusion = (review_entry or {}).get("exclusion")
    if not isinstance(exclusion, dict):
        return ""
    status = exclusion.get("status")
    return status if isinstance(status, str) else ""


def problem_is_excluded(review_entry: dict[str, Any] | None) -> bool:
    """True only for an explicit, audited exclusion tombstone."""

    return problem_exclusion_status(review_entry) == "excluded"


def pass5_exclusion_status(review_entry: dict[str, Any] | None) -> str:
    """Return a stage-scoped Pass 5 exclusion without hiding valid G_obs data."""

    exclusion = (review_entry or {}).get("pass5_exclusion")
    if not isinstance(exclusion, dict):
        return ""
    status = exclusion.get("status")
    return status if isinstance(status, str) else ""


def pass5_is_excluded(review_entry: dict[str, Any] | None) -> bool:
    """True when solution alignment is excluded but Pass 1--4 remain usable."""

    return pass5_exclusion_status(review_entry) == "excluded"


def problem_semantics_sha256(problem: dict[str, Any]) -> str:
    """Hash only the text boundary that Pass 2--4 semantically depend on."""

    return json_sha256({
        "problem_id": problem.get("problem_id"),
        "segments": problem.get("segments"),
        "segmentation": problem.get("segmentation"),
    })


def canonicalize_pass4_visual_nodes(payload: Any) -> list[dict[str, Any]]:
    """Restore retained Pass 4 visual nodes from their exact Pass 1 records.

    Pass 4 may omit a duplicate visual node, but it must never edit a retained
    node. Unknown IDs, duplicate Pass 1 IDs, and malformed records are left
    untouched so that the validator rejects them instead of concealing a model
    error. The returned change list is suitable for a metadata-only audit log.
    """

    if not isinstance(payload, dict):
        return []
    pass1 = payload.get("pass1")
    pass4 = payload.get("pass4")
    if not isinstance(pass1, dict) or not isinstance(pass4, dict):
        return []
    pass1_records = pass1.get("visual_nodes")
    pass4_records = pass4.get("visual_nodes")
    if not isinstance(pass1_records, list) or not isinstance(pass4_records, list):
        return []

    id_counts: dict[str, int] = {}
    for record in pass1_records:
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            node_id = record["id"]
            id_counts[node_id] = id_counts.get(node_id, 0) + 1
    canonical = {
        record["id"]: record
        for record in pass1_records
        if isinstance(record, dict)
        and isinstance(record.get("id"), str)
        and id_counts.get(record["id"]) == 1
    }

    changes: list[dict[str, Any]] = []
    normalized: list[Any] = []
    missing = object()
    for index, record in enumerate(pass4_records):
        node_id = record.get("id") if isinstance(record, dict) else None
        source = canonical.get(node_id) if isinstance(node_id, str) else None
        if source is None or record == source:
            normalized.append(record)
            continue
        fields = sorted(
            key
            for key in set(record) | set(source)
            if record.get(key, missing) != source.get(key, missing)
        )
        changes.append(
            {
                "pass4_index": index,
                "visual_node_id": node_id,
                "changed_fields": fields,
                "before_sha256": json_sha256(record),
                "after_sha256": json_sha256(source),
            }
        )
        normalized.append(deepcopy(source))
    pass4["visual_nodes"] = normalized
    return changes


def blank_document(stage: str, problem_id: str) -> dict[str, Any]:
    if stage == "pass1":
        return {"problem_id": problem_id, "visual_nodes": [], "ambiguities": []}
    if stage == "pass2":
        return {
            "problem_id": problem_id,
            "physical_nodes": [],
            "bindings": [],
            "ambiguities": [],
        }
    if stage == "pass3":
        return {
            "problem_id": problem_id,
            "new_or_updated_physical_nodes": [],
            "text_mentions": [],
            "quantities": [],
            "bindings": [],
            "constraints": [],
            "query_target": {
                "mention_id": "",
                "target_kind": "other",
                "target_symbol_latex": "",
                "target_node_ids": [],
                "location_node_ids": [],
                "time_or_event_node_ids": [],
            },
            "ambiguities": [],
        }
    if stage == "pass4":
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_id": problem_id,
            "domain": "mechanics",
            "visual_nodes": [],
            "physical_nodes": [],
            "text_mentions": [],
            "quantities": [],
            "bindings": [],
            "relations": [],
            "constraints": [],
            "query_target": {
                "mention_id": "",
                "target_kind": "other",
                "target_symbol_latex": "",
                "target_node_ids": [],
                "location_node_ids": [],
                "time_or_event_node_ids": [],
            },
            "ambiguities": [],
        }
    raise ValueError(f"Unknown stage: {stage}")


def _issue(
    issues: list[dict[str, str]], level: str, code: str, path: str, message: str
) -> None:
    issues.append({"level": level, "code": code, "path": path, "message": message})


def _check_exact_keys(
    issues: list[dict[str, str]], value: Any, expected: set[str], path: str
) -> bool:
    if not isinstance(value, dict):
        _issue(issues, "error", "type", path, "必须是 JSON object")
        return False
    actual = set(value)
    for missing in sorted(expected - actual):
        _issue(issues, "error", "missing_key", path, f"缺少字段 {missing}")
    for extra in sorted(actual - expected):
        _issue(issues, "error", "extra_key", path, f"不允许字段 {extra}")
    return not (expected - actual or actual - expected)


def _check_id(
    issues: list[dict[str, str]], value: Any, kind: str, path: str
) -> None:
    if not isinstance(value, str) or not ID_PATTERNS[kind].fullmatch(value):
        _issue(issues, "error", "invalid_id", path, f"必须符合 {kind} ID 规则")


def _check_string(issues: list[dict[str, str]], value: Any, path: str) -> None:
    if not isinstance(value, str):
        _issue(issues, "error", "type", path, "必须是字符串")


def _check_string_list(
    issues: list[dict[str, str]], value: Any, path: str, allow_empty: bool = True
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _issue(issues, "error", "type", path, "必须是字符串数组")
        return []
    if not allow_empty and not value:
        _issue(issues, "error", "empty", path, "不能为空")
    if len(value) != len(set(value)):
        _issue(issues, "error", "duplicate", path, "数组中存在重复 ID")
    return value


def _check_provenance_and_evidence(
    issues: list[dict[str, str]], item: dict[str, Any], path: str
) -> None:
    provenance = _check_string_list(
        issues, item.get("provenance"), f"{path}.provenance", allow_empty=False
    )
    invalid = [value for value in provenance if value not in PROVENANCE]
    if invalid:
        _issue(
            issues,
            "error",
            "enum",
            f"{path}.provenance",
            f"仅允许 {', '.join(PROVENANCE)}",
        )
    visual = _check_string_list(
        issues, item.get("evidence_visual_ids"), f"{path}.evidence_visual_ids"
    )
    mentions = _check_string_list(
        issues, item.get("evidence_mention_ids"), f"{path}.evidence_mention_ids"
    )
    if "IMAGE" in provenance and not visual:
        _issue(issues, "error", "missing_evidence", path, "IMAGE 来源必须有视觉证据")
    if "TEXT" in provenance and not mentions:
        _issue(issues, "error", "missing_evidence", path, "TEXT 来源必须有 mention 证据")
    if visual and "IMAGE" not in provenance:
        _issue(issues, "error", "provenance_mismatch", path, "有视觉证据但缺少 IMAGE 来源")
    if mentions and "TEXT" not in provenance:
        _issue(issues, "error", "provenance_mismatch", path, "有文本证据但缺少 TEXT 来源")


def _check_no_prohibited_keys(
    issues: list[dict[str, str]], value: Any, path: str = "$"
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in PROHIBITED_ANNOTATION_KEYS:
                _issue(
                    issues,
                    "error",
                    "answer_leakage_key",
                    f"{path}.{key}",
                    "Pass 1–4 禁止包含答案或标准解析字段",
                )
            _check_no_prohibited_keys(issues, child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_no_prohibited_keys(issues, child, f"{path}[{index}]")


def _check_sequential_ids(
    issues: list[dict[str, str]], records: Any, kind: str, path: str, require_contiguous: bool = True
) -> set[str]:
    if not isinstance(records, list):
        _issue(issues, "error", "type", path, "必须是数组")
        return set()
    ids: list[str] = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            _issue(issues, "error", "type", f"{path}[{index}]", "必须是 object")
            continue
        identifier = item.get("id")
        _check_id(issues, identifier, kind, f"{path}[{index}].id")
        if isinstance(identifier, str):
            ids.append(identifier)
    if len(ids) != len(set(ids)):
        _issue(issues, "error", "duplicate_id", path, "存在重复 ID")
    prefixes = {
        "visual": "v",
        "physical": "p",
        "mention": "m",
        "quantity": "q",
        "binding": "b",
        "relation": "r",
        "constraint": "c",
        "ambiguity": "u",
    }
    expected = [f"{prefixes[kind]}{index:03d}" for index in range(1, len(ids) + 1)]
    if require_contiguous and ids and ids != expected:
        _issue(
            issues,
            "error",
            "unstable_id_sequence",
            path,
            f"ID 应按顺序为 {', '.join(expected)}",
        )
    return set(ids)


def _validate_visual_nodes(
    issues: list[dict[str, str]], records: Any, image_ids: set[str], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "visual", path)
    if not isinstance(records, list):
        return ids
    expected = {
        "id",
        "type",
        "subtype",
        "text",
        "image_id",
        "bbox_1000",
        "keypoints_1000",
        "center_1000",
        "radius_1000",
        "confidence",
    }
    for index, node in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, node, expected, item_path):
            if not isinstance(node, dict):
                continue
        if node.get("type") not in VISUAL_TYPES:
            _issue(issues, "error", "enum", f"{item_path}.type", "视觉类型不在允许枚举中")
        for field in ("subtype", "text", "image_id"):
            _check_string(issues, node.get(field), f"{item_path}.{field}")
        if node.get("image_id") not in image_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.image_id", "未知 image_id")
        bbox = node.get("bbox_1000")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
            or not all(0 <= value <= 1000 for value in bbox)
        ):
            _issue(issues, "error", "bbox", f"{item_path}.bbox_1000", "必须是 0–1000 的四个整数")
        elif bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            _issue(issues, "error", "bbox_order", f"{item_path}.bbox_1000", "坐标顺序必须为 x1,y1,x2,y2")
        for field in ("keypoints_1000",):
            points = node.get(field)
            if not isinstance(points, list) or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 1000 for value in point)
                for point in points
            ):
                _issue(issues, "error", "coordinates", f"{item_path}.{field}", "关键点必须是 0–1000 的整数坐标对")
        center = node.get("center_1000")
        if (
            not isinstance(center, list)
            or len(center) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value < -1 or value > 1000 for value in center)
        ):
            _issue(issues, "error", "coordinates", f"{item_path}.center_1000", "圆心必须为两个 -1–1000 的整数")
        radius = node.get("radius_1000")
        if not isinstance(radius, int) or isinstance(radius, bool) or radius < -1 or radius > 1000:
            _issue(issues, "error", "radius", f"{item_path}.radius_1000", "半径必须为 -1–1000 的整数")
        if node.get("confidence") not in CONFIDENCE:
            _issue(issues, "error", "enum", f"{item_path}.confidence", "confidence 仅允许 high/medium/low")
    return ids


def _validate_physical_nodes(
    issues: list[dict[str, str]], records: Any, visual_ids: set[str], mention_ids: set[str], path: str,
    require_contiguous: bool = True,
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "physical", path, require_contiguous)
    if not isinstance(records, list):
        return ids
    expected = {
        "id",
        "type",
        "subtype",
        "name",
        "symbol",
        "visual_anchor_ids",
        "text_mention_ids",
        "provenance",
        "confidence",
    }
    for index, node in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, node, expected, item_path):
            if not isinstance(node, dict):
                continue
        if node.get("type") not in PHYSICAL_TYPES:
            _issue(issues, "error", "enum", f"{item_path}.type", "物理实体类型不在允许枚举中")
        for field in ("subtype", "name", "symbol"):
            _check_string(issues, node.get(field), f"{item_path}.{field}")
        anchors = _check_string_list(issues, node.get("visual_anchor_ids"), f"{item_path}.visual_anchor_ids")
        mentions = _check_string_list(issues, node.get("text_mention_ids"), f"{item_path}.text_mention_ids")
        provenance = _check_string_list(issues, node.get("provenance"), f"{item_path}.provenance", allow_empty=False)
        if any(value not in visual_ids for value in anchors):
            _issue(issues, "error", "unknown_reference", f"{item_path}.visual_anchor_ids", "包含未知 visual ID")
        if any(value not in mention_ids for value in mentions):
            _issue(issues, "error", "unknown_reference", f"{item_path}.text_mention_ids", "包含未知 mention ID")
        if any(value not in PROVENANCE for value in provenance):
            _issue(issues, "error", "enum", f"{item_path}.provenance", "来源仅允许 IMAGE/TEXT")
        if anchors and "IMAGE" not in provenance:
            _issue(issues, "error", "provenance_mismatch", item_path, "有视觉锚点但缺少 IMAGE 来源")
        if mentions and "TEXT" not in provenance:
            _issue(issues, "error", "provenance_mismatch", item_path, "有文本 mention 但缺少 TEXT 来源")
        if "IMAGE" in provenance and not anchors:
            _issue(issues, "error", "missing_evidence", item_path, "IMAGE 物理实体必须有视觉锚点")
        if "TEXT" in provenance and not mentions:
            _issue(issues, "error", "missing_evidence", item_path, "TEXT 物理实体必须有 mention")
        if node.get("confidence") not in CONFIDENCE:
            _issue(issues, "error", "enum", f"{item_path}.confidence", "confidence 仅允许 high/medium/low")
    return ids


def _validate_mentions(
    issues: list[dict[str, str]], records: Any, problem: dict[str, Any], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "mention", path)
    if not isinstance(records, list):
        return ids
    expected = {"id", "section", "quote", "occurrence", "role"}
    segments = problem.get("segments", {})
    mention_positions: list[tuple[int, int]] = []
    for index, mention in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, mention, expected, item_path):
            if not isinstance(mention, dict):
                continue
        section = mention.get("section")
        if section not in MENTION_SECTIONS:
            _issue(issues, "error", "enum", f"{item_path}.section", "section 仅允许 stem/query")
        quote = mention.get("quote")
        if not isinstance(quote, str) or not quote:
            _issue(issues, "error", "empty_quote", f"{item_path}.quote", "quote 必须是非空原文")
        elif section in MENTION_SECTIONS and quote not in str(segments.get(section, "")):
            _issue(issues, "error", "quote_not_found", f"{item_path}.quote", f"quote 不在 {section} 原文中")
        occurrence = mention.get("occurrence")
        if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 1:
            _issue(issues, "error", "occurrence", f"{item_path}.occurrence", "occurrence 必须是正整数")
        if mention.get("role") not in MENTION_ROLES:
            _issue(issues, "error", "enum", f"{item_path}.role", "mention role 不在允许枚举中")
        if isinstance(quote, str) and quote and section in MENTION_SECTIONS:
            text = str(segments.get(section, ""))
            occurrence_value = occurrence if isinstance(occurrence, int) and occurrence > 0 else 1
            start = -1
            cursor = 0
            for _ in range(occurrence_value):
                start = text.find(quote, cursor)
                if start < 0:
                    break
                cursor = start + len(quote)
            mention_positions.append((MENTION_SECTIONS.index(section), start if start >= 0 else 10**9))
    if mention_positions and mention_positions != sorted(mention_positions):
        _issue(issues, "error", "mention_order", path, "text mention 必须按 stem、query 中的原文出现顺序编号")
    return ids


def _validate_bindings(
    issues: list[dict[str, str]], records: Any, all_ids: set[str], visual_ids: set[str], physical_ids: set[str], mention_ids: set[str], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "binding", path)
    if not isinstance(records, list):
        return ids
    expected = {
        "id", "type", "from_id", "to_id", "provenance",
        "evidence_visual_ids", "evidence_mention_ids",
    }
    for index, binding in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, binding, expected, item_path):
            if not isinstance(binding, dict):
                continue
        kind = binding.get("type")
        if kind not in BINDING_TYPES:
            _issue(issues, "error", "enum", f"{item_path}.type", "binding 类型不在允许枚举中")
        from_id, to_id = binding.get("from_id"), binding.get("to_id")
        if from_id not in all_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.from_id", "未知 from_id")
        if to_id not in all_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.to_id", "未知 to_id")
        if kind in {"labels", "represents"} and not (from_id in visual_ids and to_id in physical_ids):
            _issue(issues, "error", "binding_direction", item_path, f"{kind} 必须由 visual 指向 physical")
        if kind == "refers_to" and not (from_id in mention_ids and to_id in physical_ids):
            _issue(issues, "error", "binding_direction", item_path, "refers_to 必须由 mention 指向 physical")
        _check_provenance_and_evidence(issues, binding, item_path)
        if any(value not in visual_ids for value in binding.get("evidence_visual_ids", []) if isinstance(value, str)):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_visual_ids", "包含未知 visual ID")
        if any(value not in mention_ids for value in binding.get("evidence_mention_ids", []) if isinstance(value, str)):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_mention_ids", "包含未知 mention ID")
    return ids


def _validate_quantities(
    issues: list[dict[str, str]], records: Any, physical_ids: set[str], visual_ids: set[str], mention_ids: set[str], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "quantity", path)
    if not isinstance(records, list):
        return ids
    expected = {
        "id", "kind", "symbol_latex", "value_latex", "unit_latex", "owner_id",
        "visual_anchor_ids", "text_mention_ids", "provenance",
    }
    for index, quantity in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, quantity, expected, item_path):
            if not isinstance(quantity, dict):
                continue
        for field in ("kind", "symbol_latex", "value_latex", "unit_latex", "owner_id"):
            _check_string(issues, quantity.get(field), f"{item_path}.{field}")
        if quantity.get("kind") == "":
            _issue(issues, "error", "empty", f"{item_path}.kind", "kind 不能为空；无法归类时使用 other")
        owner = quantity.get("owner_id")
        if owner and owner not in physical_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.owner_id", "未知 owner_id")
        anchors = _check_string_list(issues, quantity.get("visual_anchor_ids"), f"{item_path}.visual_anchor_ids")
        mentions = _check_string_list(issues, quantity.get("text_mention_ids"), f"{item_path}.text_mention_ids")
        if any(value not in visual_ids for value in anchors):
            _issue(issues, "error", "unknown_reference", f"{item_path}.visual_anchor_ids", "包含未知 visual ID")
        if any(value not in mention_ids for value in mentions):
            _issue(issues, "error", "unknown_reference", f"{item_path}.text_mention_ids", "包含未知 mention ID")
        proxy = {
            "provenance": quantity.get("provenance"),
            "evidence_visual_ids": anchors,
            "evidence_mention_ids": mentions,
        }
        _check_provenance_and_evidence(issues, proxy, item_path)
    return ids


def _validate_constraints(
    issues: list[dict[str, str]], records: Any, physical_ids: set[str], visual_ids: set[str], mention_ids: set[str], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "constraint", path)
    if not isinstance(records, list):
        return ids
    expected = {
        "id", "kind", "subject_ids", "value_text", "provenance",
        "evidence_visual_ids", "evidence_mention_ids",
    }
    for index, constraint in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, constraint, expected, item_path):
            if not isinstance(constraint, dict):
                continue
        _check_string(issues, constraint.get("kind"), f"{item_path}.kind")
        _check_string(issues, constraint.get("value_text"), f"{item_path}.value_text")
        if constraint.get("kind") == "":
            _issue(issues, "error", "empty", f"{item_path}.kind", "kind 不能为空；无法归类时使用 other")
        subjects = _check_string_list(issues, constraint.get("subject_ids"), f"{item_path}.subject_ids", allow_empty=False)
        if any(value not in physical_ids for value in subjects):
            _issue(issues, "error", "unknown_reference", f"{item_path}.subject_ids", "包含未知 physical ID")
        _check_provenance_and_evidence(issues, constraint, item_path)
        if any(value not in visual_ids for value in constraint.get("evidence_visual_ids", []) if isinstance(value, str)):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_visual_ids", "包含未知 visual ID")
        if any(value not in mention_ids for value in constraint.get("evidence_mention_ids", []) if isinstance(value, str)):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_mention_ids", "包含未知 mention ID")
    return ids


def _validate_relations(
    issues: list[dict[str, str]], records: Any, physical_ids: set[str], quantity_ids: set[str], visual_ids: set[str], mention_ids: set[str], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "relation", path)
    if not isinstance(records, list):
        return ids
    expected = {
        "id", "predicate", "subject_id", "object_id", "quantity_id", "provenance",
        "evidence_visual_ids", "evidence_mention_ids", "confidence",
    }
    signatures: set[tuple[Any, ...]] = set()
    for index, relation in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, relation, expected, item_path):
            if not isinstance(relation, dict):
                continue
        for field in ("predicate", "subject_id", "object_id", "quantity_id"):
            _check_string(issues, relation.get(field), f"{item_path}.{field}")
        if not relation.get("predicate"):
            _issue(issues, "error", "empty", f"{item_path}.predicate", "predicate 不能为空；无法归类时使用 other")
        if relation.get("subject_id") not in physical_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.subject_id", "未知 physical subject_id")
        if relation.get("object_id") not in physical_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.object_id", "未知 physical object_id")
        quantity_id = relation.get("quantity_id")
        if quantity_id and quantity_id not in quantity_ids:
            _issue(issues, "error", "unknown_reference", f"{item_path}.quantity_id", "未知 quantity_id")
        _check_provenance_and_evidence(issues, relation, item_path)
        if any(value not in visual_ids for value in relation.get("evidence_visual_ids", []) if isinstance(value, str)):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_visual_ids", "包含未知 visual ID")
        if any(value not in mention_ids for value in relation.get("evidence_mention_ids", []) if isinstance(value, str)):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_mention_ids", "包含未知 mention ID")
        if relation.get("confidence") not in CONFIDENCE:
            _issue(issues, "error", "enum", f"{item_path}.confidence", "confidence 仅允许 high/medium/low")
        signature = (relation.get("predicate"), relation.get("subject_id"), relation.get("object_id"), quantity_id)
        if signature in signatures:
            _issue(issues, "error", "duplicate_relation", item_path, "存在重复关系")
        signatures.add(signature)
    return ids


def _validate_ambiguities(
    issues: list[dict[str, str]], records: Any, all_ids: set[str], visual_ids: set[str], mention_ids: set[str], path: str
) -> set[str]:
    ids = _check_sequential_ids(issues, records, "ambiguity", path)
    if not isinstance(records, list):
        return ids
    expected = {
        "id", "scope", "description", "candidate_ids",
        "evidence_visual_ids", "evidence_mention_ids",
    }
    for index, ambiguity in enumerate(records):
        item_path = f"{path}[{index}]"
        if not _check_exact_keys(issues, ambiguity, expected, item_path):
            if not isinstance(ambiguity, dict):
                continue
        for field in ("scope", "description"):
            _check_string(issues, ambiguity.get(field), f"{item_path}.{field}")
        if not ambiguity.get("description"):
            _issue(issues, "error", "empty", f"{item_path}.description", "description 不能为空")
        candidates = _check_string_list(issues, ambiguity.get("candidate_ids"), f"{item_path}.candidate_ids")
        visual = _check_string_list(issues, ambiguity.get("evidence_visual_ids"), f"{item_path}.evidence_visual_ids")
        mentions = _check_string_list(issues, ambiguity.get("evidence_mention_ids"), f"{item_path}.evidence_mention_ids")
        if any(value not in all_ids for value in candidates):
            _issue(issues, "error", "unknown_reference", f"{item_path}.candidate_ids", "包含未知 ID")
        if any(value not in visual_ids for value in visual):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_visual_ids", "包含未知 visual ID")
        if any(value not in mention_ids for value in mentions):
            _issue(issues, "error", "unknown_reference", f"{item_path}.evidence_mention_ids", "包含未知 mention ID")
    return ids


def _validate_query_target(
    issues: list[dict[str, str]], target: Any, physical_ids: set[str], mention_ids: set[str], path: str
) -> None:
    expected = {
        "mention_id", "target_kind", "target_symbol_latex", "target_node_ids",
        "location_node_ids", "time_or_event_node_ids",
    }
    if not _check_exact_keys(issues, target, expected, path):
        if not isinstance(target, dict):
            return
    for field in ("mention_id", "target_kind", "target_symbol_latex"):
        _check_string(issues, target.get(field), f"{path}.{field}")
    mention_id = target.get("mention_id")
    if mention_id and mention_id not in mention_ids:
        _issue(issues, "error", "unknown_reference", f"{path}.mention_id", "未知 query mention ID")
    if not target.get("target_kind"):
        _issue(issues, "error", "empty", f"{path}.target_kind", "target_kind 不能为空；无法归类时使用 other")
    for field in ("target_node_ids", "location_node_ids", "time_or_event_node_ids"):
        values = _check_string_list(issues, target.get(field), f"{path}.{field}")
        if any(value not in physical_ids for value in values):
            _issue(issues, "error", "unknown_reference", f"{path}.{field}", "包含未知 physical ID")


def validate_document(
    stage: str,
    document: Any,
    problem: dict[str, Any],
    previous: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Validate shape, IDs, references, evidence, provenance, and blind-pass safety."""

    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    previous = previous or {}
    issues: list[dict[str, str]] = []
    _check_no_prohibited_keys(issues, document)

    top_keys = {
        "pass1": {"problem_id", "visual_nodes", "ambiguities"},
        "pass2": {"problem_id", "physical_nodes", "bindings", "ambiguities"},
        "pass3": {
            "problem_id", "new_or_updated_physical_nodes", "text_mentions", "quantities",
            "bindings", "constraints", "query_target", "ambiguities",
        },
        "pass4": {
            "schema_version", "problem_id", "domain", "visual_nodes", "physical_nodes",
            "text_mentions", "quantities", "bindings", "relations", "constraints",
            "query_target", "ambiguities",
        },
    }[stage]
    if not _check_exact_keys(issues, document, top_keys, "$"):
        if not isinstance(document, dict):
            return issues
    if document.get("problem_id") != problem.get("problem_id"):
        _issue(issues, "error", "problem_id_mismatch", "$.problem_id", "problem_id 与当前题目不一致")
    image_ids = {image["image_id"] for image in problem.get("images", [])}

    if stage == "pass1":
        visual_ids = _validate_visual_nodes(issues, document.get("visual_nodes"), image_ids, "$.visual_nodes")
        if image_ids and not visual_ids:
            _issue(issues, "error", "empty_visual_extraction", "$.visual_nodes", "有题图时 Pass 1 至少应标注一个相关视觉图元")
        _validate_ambiguities(issues, document.get("ambiguities"), visual_ids, visual_ids, set(), "$.ambiguities")

    elif stage == "pass2":
        pass1 = previous.get("pass1", {})
        visual_ids = {item.get("id") for item in pass1.get("visual_nodes", []) if isinstance(item, dict)}
        physical_ids = _validate_physical_nodes(issues, document.get("physical_nodes"), visual_ids, set(), "$.physical_nodes")
        all_ids = visual_ids | physical_ids
        _validate_bindings(issues, document.get("bindings"), all_ids, visual_ids, physical_ids, set(), "$.bindings")
        _validate_ambiguities(issues, document.get("ambiguities"), all_ids, visual_ids, set(), "$.ambiguities")
        for index, node in enumerate(document.get("physical_nodes", []) if isinstance(document.get("physical_nodes"), list) else []):
            if isinstance(node, dict) and (node.get("provenance") != ["IMAGE"] or not node.get("visual_anchor_ids")):
                _issue(issues, "error", "pass2_image_only", f"$.physical_nodes[{index}]", "Pass 2 只允许创建有 IMAGE 锚点的实体")

    elif stage == "pass3":
        pass1 = previous.get("pass1", {})
        pass2 = previous.get("pass2", {})
        visual_ids = {item.get("id") for item in pass1.get("visual_nodes", []) if isinstance(item, dict)}
        base_physical = {item.get("id") for item in pass2.get("physical_nodes", []) if isinstance(item, dict)}
        mention_ids = _validate_mentions(issues, document.get("text_mentions"), problem, "$.text_mentions")
        if problem.get("segments", {}).get("query") and not mention_ids:
            _issue(issues, "error", "missing_text_grounding", "$.text_mentions", "存在问题文本时至少应有一个 query_target mention")
        updated_physical = _validate_physical_nodes(
            issues, document.get("new_or_updated_physical_nodes"), visual_ids, mention_ids,
            "$.new_or_updated_physical_nodes", require_contiguous=False,
        )
        physical_ids = base_physical | updated_physical
        if physical_ids:
            expected_physical = {f"p{index:03d}" for index in range(1, max(int(value[1:]) for value in physical_ids) + 1)}
            if physical_ids != expected_physical:
                _issue(issues, "error", "unstable_id_sequence", "$.new_or_updated_physical_nodes", "与 Pass 2 合并后的 physical ID 必须连续且稳定")
        all_ids = visual_ids | physical_ids | mention_ids
        _validate_bindings(issues, document.get("bindings"), all_ids, visual_ids, physical_ids, mention_ids, "$.bindings")
        _validate_quantities(issues, document.get("quantities"), physical_ids, visual_ids, mention_ids, "$.quantities")
        _validate_constraints(issues, document.get("constraints"), physical_ids, visual_ids, mention_ids, "$.constraints")
        _validate_query_target(issues, document.get("query_target"), physical_ids, mention_ids, "$.query_target")
        if problem.get("segments", {}).get("query") and not document.get("query_target", {}).get("mention_id"):
            _issue(issues, "error", "missing_query_target", "$.query_target.mention_id", "存在问题文本时必须指向 query mention")
        target_mention_id = document.get("query_target", {}).get("mention_id")
        target_mention = next((item for item in document.get("text_mentions", []) if isinstance(item, dict) and item.get("id") == target_mention_id), None)
        if target_mention_id and target_mention and (target_mention.get("section") != "query" or target_mention.get("role") != "query_target"):
            _issue(issues, "error", "query_target_role", "$.query_target.mention_id", "query_target 必须指向 section=query 且 role=query_target 的 mention")
        _validate_ambiguities(issues, document.get("ambiguities"), all_ids, visual_ids, mention_ids, "$.ambiguities")

    else:
        if document.get("schema_version") != SCHEMA_VERSION:
            _issue(issues, "error", "schema_version", "$.schema_version", f"必须为 {SCHEMA_VERSION}")
        if document.get("domain") not in DOMAINS:
            _issue(issues, "error", "enum", "$.domain", "domain 仅允许 mechanics/electromagnetism/mixed")
        visual_ids = _validate_visual_nodes(issues, document.get("visual_nodes"), image_ids, "$.visual_nodes")
        mention_ids = _validate_mentions(issues, document.get("text_mentions"), problem, "$.text_mentions")
        if problem.get("segments", {}).get("query") and not mention_ids:
            _issue(issues, "error", "missing_text_grounding", "$.text_mentions", "存在问题文本时至少应有一个 query_target mention")
        physical_ids = _validate_physical_nodes(issues, document.get("physical_nodes"), visual_ids, mention_ids, "$.physical_nodes")
        all_ids = visual_ids | physical_ids | mention_ids
        _validate_bindings(issues, document.get("bindings"), all_ids, visual_ids, physical_ids, mention_ids, "$.bindings")
        quantity_ids = _validate_quantities(issues, document.get("quantities"), physical_ids, visual_ids, mention_ids, "$.quantities")
        _validate_relations(issues, document.get("relations"), physical_ids, quantity_ids, visual_ids, mention_ids, "$.relations")
        _validate_constraints(issues, document.get("constraints"), physical_ids, visual_ids, mention_ids, "$.constraints")
        _validate_query_target(issues, document.get("query_target"), physical_ids, mention_ids, "$.query_target")
        if problem.get("segments", {}).get("query") and not document.get("query_target", {}).get("mention_id"):
            _issue(issues, "error", "missing_query_target", "$.query_target.mention_id", "存在问题文本时必须指向 query mention")
        target_mention_id = document.get("query_target", {}).get("mention_id")
        target_mention = next((item for item in document.get("text_mentions", []) if isinstance(item, dict) and item.get("id") == target_mention_id), None)
        if target_mention_id and target_mention and (target_mention.get("section") != "query" or target_mention.get("role") != "query_target"):
            _issue(issues, "error", "query_target_role", "$.query_target.mention_id", "query_target 必须指向 section=query 且 role=query_target 的 mention")
        _validate_ambiguities(issues, document.get("ambiguities"), all_ids | quantity_ids, visual_ids, mention_ids, "$.ambiguities")
        pass1_nodes = {
            item.get("id"): item for item in previous.get("pass1", {}).get("visual_nodes", []) if isinstance(item, dict)
        }
        for index, node in enumerate(document.get("visual_nodes", []) if isinstance(document.get("visual_nodes"), list) else []):
            if not isinstance(node, dict):
                continue
            if node.get("id") not in pass1_nodes:
                _issue(issues, "error", "pass1_unknown_visual", f"$.visual_nodes[{index}]", "Pass 4 视觉节点必须来自 Pass 1，不能新增节点")
            elif node != pass1_nodes[node["id"]]:
                _issue(issues, "error", "pass1_modified", f"$.visual_nodes[{index}]", "Pass 4 不得修改 Pass 1 的视觉节点；仅可去重删除")

    return issues


def validation_summary(issues: list[dict[str, str]]) -> dict[str, Any]:
    errors = sum(issue["level"] == "error" for issue in issues)
    warnings = sum(issue["level"] == "warning" for issue in issues)
    return {"valid": errors == 0, "errors": errors, "warnings": warnings, "issues": issues}


def contract_json_schemas() -> dict[str, dict[str, Any]]:
    """Return machine-readable schemas mirroring the strict UI contract."""

    string = {"type": "string"}
    id_array = {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
    provenance = {"type": "array", "items": {"enum": list(PROVENANCE)}, "minItems": 1, "uniqueItems": True}
    point = {
        "type": "array", "prefixItems": [{"type": "integer", "minimum": -1, "maximum": 1000}, {"type": "integer", "minimum": -1, "maximum": 1000}],
        "minItems": 2, "maxItems": 2,
    }
    defs: dict[str, Any] = {
        "visual_node": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "type", "subtype", "text", "image_id", "bbox_1000", "keypoints_1000", "center_1000", "radius_1000", "confidence"],
            "properties": {
                "id": {"type": "string", "pattern": r"^v\d{3,}$"}, "type": {"enum": list(VISUAL_TYPES)}, "subtype": string, "text": string, "image_id": string,
                "bbox_1000": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4},
                "keypoints_1000": {"type": "array", "items": point}, "center_1000": point,
                "radius_1000": {"type": "integer", "minimum": -1, "maximum": 1000}, "confidence": {"enum": list(CONFIDENCE)},
            },
        },
        "physical_node": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "type", "subtype", "name", "symbol", "visual_anchor_ids", "text_mention_ids", "provenance", "confidence"],
            "properties": {
                "id": {"type": "string", "pattern": r"^p\d{3,}$"}, "type": {"enum": list(PHYSICAL_TYPES)}, "subtype": string, "name": string, "symbol": string,
                "visual_anchor_ids": id_array, "text_mention_ids": id_array, "provenance": provenance, "confidence": {"enum": list(CONFIDENCE)},
            },
        },
        "text_mention": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "section", "quote", "occurrence", "role"],
            "properties": {"id": {"type": "string", "pattern": r"^m\d{3,}$"}, "section": {"enum": list(MENTION_SECTIONS)}, "quote": {"type": "string", "minLength": 1}, "occurrence": {"type": "integer", "minimum": 1}, "role": {"enum": list(MENTION_ROLES)}},
        },
        "binding": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "type", "from_id", "to_id", "provenance", "evidence_visual_ids", "evidence_mention_ids"],
            "properties": {"id": {"type": "string", "pattern": r"^b\d{3,}$"}, "type": {"enum": list(BINDING_TYPES)}, "from_id": string, "to_id": string, "provenance": provenance, "evidence_visual_ids": id_array, "evidence_mention_ids": id_array},
        },
        "quantity": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "kind", "symbol_latex", "value_latex", "unit_latex", "owner_id", "visual_anchor_ids", "text_mention_ids", "provenance"],
            "properties": {"id": {"type": "string", "pattern": r"^q\d{3,}$"}, "kind": {"type": "string", "minLength": 1}, "symbol_latex": string, "value_latex": string, "unit_latex": string, "owner_id": string, "visual_anchor_ids": id_array, "text_mention_ids": id_array, "provenance": provenance},
        },
        "constraint": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "kind", "subject_ids", "value_text", "provenance", "evidence_visual_ids", "evidence_mention_ids"],
            "properties": {"id": {"type": "string", "pattern": r"^c\d{3,}$"}, "kind": {"type": "string", "minLength": 1}, "subject_ids": {**id_array, "minItems": 1}, "value_text": string, "provenance": provenance, "evidence_visual_ids": id_array, "evidence_mention_ids": id_array},
        },
        "relation": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "predicate", "subject_id", "object_id", "quantity_id", "provenance", "evidence_visual_ids", "evidence_mention_ids", "confidence"],
            "properties": {"id": {"type": "string", "pattern": r"^r\d{3,}$"}, "predicate": {"type": "string", "minLength": 1}, "subject_id": string, "object_id": string, "quantity_id": string, "provenance": provenance, "evidence_visual_ids": id_array, "evidence_mention_ids": id_array, "confidence": {"enum": list(CONFIDENCE)}},
        },
        "query_target": {
            "type": "object", "additionalProperties": False,
            "required": ["mention_id", "target_kind", "target_symbol_latex", "target_node_ids", "location_node_ids", "time_or_event_node_ids"],
            "properties": {"mention_id": string, "target_kind": {"type": "string", "minLength": 1}, "target_symbol_latex": string, "target_node_ids": id_array, "location_node_ids": id_array, "time_or_event_node_ids": id_array},
        },
        "ambiguity": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "scope", "description", "candidate_ids", "evidence_visual_ids", "evidence_mention_ids"],
            "properties": {"id": {"type": "string", "pattern": r"^u\d{3,}$"}, "scope": string, "description": {"type": "string", "minLength": 1}, "candidate_ids": id_array, "evidence_visual_ids": id_array, "evidence_mention_ids": id_array},
        },
    }

    def schema(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
        return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False, "required": required, "properties": properties, "$defs": deepcopy(defs)}

    ref_array = lambda name: {"type": "array", "items": {"$ref": f"#/$defs/{name}"}}
    return {
        "pass1": schema(["problem_id", "visual_nodes", "ambiguities"], {"problem_id": string, "visual_nodes": ref_array("visual_node"), "ambiguities": ref_array("ambiguity")}),
        "pass2": schema(["problem_id", "physical_nodes", "bindings", "ambiguities"], {"problem_id": string, "physical_nodes": ref_array("physical_node"), "bindings": ref_array("binding"), "ambiguities": ref_array("ambiguity")}),
        "pass3": schema(["problem_id", "new_or_updated_physical_nodes", "text_mentions", "quantities", "bindings", "constraints", "query_target", "ambiguities"], {"problem_id": string, "new_or_updated_physical_nodes": ref_array("physical_node"), "text_mentions": ref_array("text_mention"), "quantities": ref_array("quantity"), "bindings": ref_array("binding"), "constraints": ref_array("constraint"), "query_target": {"$ref": "#/$defs/query_target"}, "ambiguities": ref_array("ambiguity")}),
        "pass4": schema(["schema_version", "problem_id", "domain", "visual_nodes", "physical_nodes", "text_mentions", "quantities", "bindings", "relations", "constraints", "query_target", "ambiguities"], {"schema_version": {"const": SCHEMA_VERSION}, "problem_id": string, "domain": {"enum": list(DOMAINS)}, "visual_nodes": ref_array("visual_node"), "physical_nodes": ref_array("physical_node"), "text_mentions": ref_array("text_mention"), "quantities": ref_array("quantity"), "bindings": ref_array("binding"), "relations": ref_array("relation"), "constraints": ref_array("constraint"), "query_target": {"$ref": "#/$defs/query_target"}, "ambiguities": ref_array("ambiguity")}),
    }

