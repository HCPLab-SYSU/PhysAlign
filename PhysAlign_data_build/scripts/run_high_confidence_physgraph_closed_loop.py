#!/usr/bin/env python
"""Resumable PhysGraph annotation with isolated Pass 1 and visual correction."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from physgraph_annotation_lib import (  # noqa: E402
    STAGES,
    effective_manifest,
    json_sha256,
    read_json,
    read_jsonl,
    resolve_workspace_path,
    segmentation_repair_status,
    validate_document,
    write_json_atomic,
)
from physgraph_api_client import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    ExclusiveRunLock,
    PhysGraphAPIClient,
    PhysGraphAPIError,
    PhysGraphAPIResponseError,
    compact_json,
    load_downstream_system_instructions,
    load_pass1_correction_system_instructions,
    load_pass1_system_instructions,
    merge_usage,
    normalize_base_url,
    safe_endpoint_label,
    write_attempt_log,
)
from physgraph_visual_feedback import render_pass1_overlays  # noqa: E402
from physgraph_id_canonicalization import canonicalize_downstream_ids  # noqa: E402
from physgraph_pixel_geometry import (  # noqa: E402
    PIXEL_NODE_KEYS,
    PanelSpec,
    build_problem_panels,
    merge_panel_documents,
    normalize_panel_document,
    normalize_pixel_correction_patch,
    panel_contract,
    panel_view_from_standard,
    pixel_patch_to_standard,
    render_panel_assets,
    render_panel_pixel_overlay,
    validate_panel_document,
)
from run_high_confidence_physgraph_annotation import (  # noqa: E402
    complete_and_valid,
    save_payload,
)


VISUAL_NODE_KEYS = {
    "id", "type", "subtype", "text", "image_id", "bbox_1000",
    "keypoints_1000", "center_1000", "radius_1000", "confidence",
}
GEOMETRY_KEYPOINT_TYPES = {
    "line_segment", "polyline", "curve", "arc", "arrow", "graph_curve",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-mode", choices=("chat", "responses", "auto"), default="chat")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--image-detail", choices=("auto", "low", "high", "original"), default="original")
    parser.add_argument("--pass1-max-output-tokens", type=int, default=32768)
    parser.add_argument("--correction-max-output-tokens", type=int, default=32768)
    parser.add_argument("--downstream-max-output-tokens", type=int, default=65536)
    parser.add_argument("--pass1-attempts", type=int, default=2)
    parser.add_argument("--correction-attempts", type=int, default=2)
    parser.add_argument("--correction-rounds", type=int, default=2)
    parser.add_argument(
        "--resume-correction-rounds", type=int, default=2,
        help="Maximum new visual correction rounds for a validated unfinished checkpoint",
    )
    parser.add_argument("--downstream-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-rejected", action="store_true")
    parser.add_argument(
        "--only-segmentation-repairs",
        action="store_true",
        help="Process only repaired segmentations whose Pass 2--4 are explicitly stale",
    )
    parser.add_argument(
        "--recover-progress", type=Path, default=None,
        help="Process only problem IDs listed in the failures array of a saved progress JSON",
    )
    parser.add_argument(
        "--no-auto-resume-checkpoints", action="store_true",
        help="Disable automatic reuse of the newest locally validated pixel checkpoint",
    )
    parser.add_argument("--ignore-pass1-cache", action="store_true")
    parser.add_argument(
        "--resume-unverified",
        type=Path,
        default=None,
        help=(
            "Resume one unverified Pass 1. A validated pixel-pipeline checkpoint reuses its geometry "
            "and accepted panels; older artifacts are used only as semantic inventory"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.model:
        if args.dry_run:
            args.model = "dry-run"
        else:
            parser.error("--model or OPENAI_MODEL is required")
    for name in ("pass1_attempts", "correction_attempts", "downstream_attempts"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.correction_rounds < 1 or args.correction_rounds > 3:
        parser.error("--correction-rounds must be between 1 and 3")
    if args.resume_correction_rounds < 1 or args.resume_correction_rounds > 2:
        parser.error("--resume-correction-rounds must be between 1 and 2")
    if args.resume_unverified is not None and args.recover_progress is not None:
        parser.error("--resume-unverified and --recover-progress cannot be used together")
    if args.only_rejected and args.recover_progress is not None:
        parser.error("--only-rejected and --recover-progress cannot be used together")
    if args.only_rejected and args.only_segmentation_repairs:
        parser.error("--only-rejected and --only-segmentation-repairs cannot be used together")
    if args.only_segmentation_repairs and (args.recover_progress is not None or args.resume_unverified is not None):
        parser.error("--only-segmentation-repairs cannot be combined with an explicit recovery target")
    return args


def validation_errors(
    stage: str,
    document: Any,
    problem: dict[str, Any],
    previous: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    return [
        {"stage": stage, **issue}
        for issue in validate_document(stage, document, problem, previous)
        if issue["level"] == "error"
    ]


def validate_pass1_geometry(document: Any) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not isinstance(document, dict) or not isinstance(document.get("visual_nodes"), list):
        return issues
    for index, node in enumerate(document["visual_nodes"]):
        if not isinstance(node, dict):
            continue
        path = f"$.visual_nodes[{index}]"
        bbox = node.get("bbox_1000")
        if not (
            isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
            and bbox[0] <= bbox[2] and bbox[1] <= bbox[3]
        ):
            continue
        points = node.get("keypoints_1000")
        if node.get("type") in GEOMETRY_KEYPOINT_TYPES and (
            not isinstance(points, list) or len(points) < 2
        ):
            issues.append({
                "stage": "pass1", "level": "error", "code": "missing_geometry_keypoints",
                "path": f"{path}.keypoints_1000", "message": "线、箭头和曲线至少需要两个关键点",
            })
        if isinstance(points, list):
            for point_index, point in enumerate(points):
                if (
                    isinstance(point, list) and len(point) == 2
                    and all(isinstance(value, int) and not isinstance(value, bool) for value in point)
                    and not (bbox[0] - 2 <= point[0] <= bbox[2] + 2 and bbox[1] - 2 <= point[1] <= bbox[3] + 2)
                ):
                    issues.append({
                        "stage": "pass1", "level": "error", "code": "keypoint_outside_bbox",
                        "path": f"{path}.keypoints_1000[{point_index}]",
                        "message": "关键点必须位于对应bbox内（允许2/1000容差）",
                    })
        center = node.get("center_1000")
        radius = node.get("radius_1000")
        if isinstance(center, list) and len(center) == 2:
            if (center[0] == -1) != (center[1] == -1):
                issues.append({
                    "stage": "pass1", "level": "error", "code": "partial_unknown_center",
                    "path": f"{path}.center_1000", "message": "未知圆心必须统一写为[-1,-1]",
                })
            elif (
                node.get("type") == "circle"
                and all(isinstance(value, int) and value >= 0 for value in center)
                and not (
                bbox[0] - 2 <= center[0] <= bbox[2] + 2
                and bbox[1] - 2 <= center[1] <= bbox[3] + 2
                )
            ):
                issues.append({
                    "stage": "pass1", "level": "error", "code": "center_outside_bbox",
                    "path": f"{path}.center_1000", "message": "circle的中心必须位于对应bbox内",
                })
        if node.get("type") in {"circle", "arc"}:
            if center == [-1, -1] or not isinstance(radius, int) or radius <= 0:
                issues.append({
                    "stage": "pass1", "level": "error", "code": "missing_circle_geometry",
                    "path": path, "message": "circle/arc必须提供可见圆心和正的radius_1000；无法确认时应改用curve并记录ambiguity",
                })
        elif center != [-1, -1] or radius != -1:
            issues.append({
                "stage": "pass1", "level": "error", "code": "unexpected_circle_geometry",
                "path": path, "message": "非circle/arc节点的center_1000必须为[-1,-1]且radius_1000必须为-1",
            })
        if node.get("type") == "text_glyph" and not str(node.get("text", "")).strip():
            issues.append({
                "stage": "pass1", "level": "error", "code": "empty_ocr_text",
                "path": f"{path}.text", "message": "text_glyph必须填写题图中实际可见文本",
            })
    return issues


def validate_pass1(document: Any, problem: dict[str, Any]) -> list[dict[str, str]]:
    return validation_errors("pass1", document, problem) + validate_pass1_geometry(document)


def validate_pass4_consolidation(payload: Any) -> list[dict[str, str]]:
    """Enforce the non-generative merge boundary between Pass 3 and Pass 4."""

    if not isinstance(payload, dict):
        return []
    pass2, pass3, pass4 = payload.get("pass2"), payload.get("pass3"), payload.get("pass4")
    if not all(isinstance(stage, dict) for stage in (pass2, pass3, pass4)):
        return []
    issues: list[dict[str, str]] = []

    pass2_nodes = pass2.get("physical_nodes")
    pass3_nodes = pass3.get("new_or_updated_physical_nodes")
    pass4_nodes = pass4.get("physical_nodes")
    if all(isinstance(records, list) for records in (pass2_nodes, pass3_nodes, pass4_nodes)):
        expected_ids = {
            node.get("id")
            for records in (pass2_nodes, pass3_nodes)
            for node in records
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        actual_ids = {
            node.get("id")
            for node in pass4_nodes
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        if expected_ids != actual_ids:
            issues.append({
                "stage": "pass4", "level": "error", "code": "pass4_merge_mismatch",
                "path": "$.physical_nodes",
                "message": "Pass 4 physical_nodes必须完整覆盖Pass 2与Pass 3合并后的实体ID",
            })

    for field in ("text_mentions", "quantities", "constraints", "query_target"):
        before, after = pass3.get(field), pass4.get(field)
        if before is not None and after is not None and before != after:
            issues.append({
                "stage": "pass4", "level": "error", "code": "pass4_merge_mismatch",
                "path": f"$.{field}",
                "message": f"Pass 4 {field}必须逐字段继承已验证的Pass 3",
            })
    return issues


def validate_no_empty_structural_regression(candidate: Any, prior: Any) -> list[dict[str, str]]:
    """Reject repair candidates that erase a previously populated core collection."""

    if not isinstance(candidate, dict) or not isinstance(prior, dict):
        return []
    issues: list[dict[str, str]] = []
    stage_fields = {
        "pass2": ("physical_nodes",),
        "pass3": ("new_or_updated_physical_nodes", "text_mentions", "quantities"),
        "pass4": ("physical_nodes", "text_mentions", "quantities"),
    }
    for stage, fields in stage_fields.items():
        candidate_stage, prior_stage = candidate.get(stage), prior.get(stage)
        if not isinstance(candidate_stage, dict) or not isinstance(prior_stage, dict):
            continue
        for field in fields:
            old, new = prior_stage.get(field), candidate_stage.get(field)
            if isinstance(old, list) and old and isinstance(new, list) and not new:
                issues.append({
                    "stage": stage, "level": "error", "code": "structural_regression",
                    "path": f"$.{field}",
                    "message": f"修复候选不得把上一轮非空的{stage}.{field}清空",
                })
    return issues


def validate_downstream(
    payload: Any,
    pass1: dict[str, Any],
    problem: dict[str, Any],
) -> list[dict[str, str]]:
    expected = {"pass2", "pass3", "pass4"}
    if not isinstance(payload, dict) or set(payload) != expected:
        return [{
            "level": "error", "code": "downstream_wrapper_shape", "path": "$",
            "message": "顶层必须恰好包含pass2、pass3、pass4",
        }]
    previous: dict[str, dict[str, Any]] = {"pass1": pass1}
    issues: list[dict[str, str]] = []
    for stage in ("pass2", "pass3", "pass4"):
        issues.extend(validation_errors(stage, payload[stage], problem, previous))
        if isinstance(payload[stage], dict):
            previous[stage] = payload[stage]
    if (
        isinstance(payload.get("pass4"), dict)
        and payload["pass4"].get("visual_nodes") != pass1.get("visual_nodes")
    ):
        issues.append({
            "stage": "pass4", "level": "error", "code": "verified_pass1_not_copied",
            "path": "$.visual_nodes",
            "message": "Pass 4 visual_nodes必须逐字段完整复制复核后的Pass 1",
        })
    issues.extend(validate_pass4_consolidation(payload))
    return issues


def force_verified_pass1_into_pass4(
    payload: Any,
    pass1: dict[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    """Make the verified Pass 1 boundary deterministic before validation/save.

    The downstream model is allowed to generate Pass 2--4, but it is not trusted
    to reproduce visual_nodes byte-for-byte.  Returning a deep copy also prevents
    an accidental mutation of the cached, verified Pass 1 object.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("pass4"), dict):
        return payload, None
    before = payload["pass4"].get("visual_nodes")
    verified = pass1.get("visual_nodes")
    if before == verified:
        return payload, None
    changes = {
        "before_sha256": json_sha256(before),
        "after_sha256": json_sha256(verified),
        "before_count": len(before) if isinstance(before, list) else -1,
        "after_count": len(verified) if isinstance(verified, list) else -1,
    }
    payload["pass4"]["visual_nodes"] = deepcopy(verified)
    return payload, changes


def normalize_downstream_payload(
    payload: Any,
    *,
    pass1: dict[str, Any],
    problem: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    """Apply every deterministic, auditable downstream normalization in order."""

    normalized, visual_changes = force_verified_pass1_into_pass4(payload, pass1)
    rules: list[dict[str, Any]] = []
    if visual_changes is not None:
        rules.append({
            "rule": "pass4_visual_nodes_exact_verified_pass1",
            "changes": visual_changes,
        })
    if isinstance(normalized, dict) and isinstance(normalized.get("pass3"), dict) and isinstance(normalized.get("pass4"), dict):
        inherited_fields = ("text_mentions", "quantities", "constraints", "query_target")
        changes: dict[str, dict[str, str]] = {}
        for field in inherited_fields:
            expected = normalized["pass3"].get(field)
            before = normalized["pass4"].get(field)
            if before != expected:
                changes[field] = {
                    "before_sha256": json_sha256(before),
                    "after_sha256": json_sha256(expected),
                }
                normalized["pass4"][field] = deepcopy(expected)
        if changes:
            rules.append({
                "rule": "pass4_exact_pass3_semantic_inheritance",
                "changes": changes,
            })
    normalized, bookkeeping_changes = canonicalize_downstream_ids(normalized, problem)
    if bookkeeping_changes is not None:
        rules.append(bookkeeping_changes)
    return normalized, rules


def pass1_prompt(
    problem: dict[str, Any],
    attempt: int,
    prior: dict[str, Any] | None,
    issues: list[dict[str, str]],
    review_notes: list[dict[str, str]] | None,
) -> str:
    repair = ""
    if prior is not None:
        repair = (
            "\n上一轮Pass 1如下：\n" + compact_json(prior)
            + "\n必须修复这些本地校验错误：\n" + compact_json(issues)
        )
    notes = ""
    if review_notes:
        notes = "\n人工退回意见（必须逐条处理）：\n" + compact_json(review_notes)
    return f"""任务阶段：独立PASS_1_VISUAL_EXTRACTION

问题编号：{problem['problem_id']}
这是第{attempt}次生成或结构修复。只允许使用下面的盲化题目记录和本次附加的原始题图：
{compact_json(problem)}

本轮只输出Pass 1完整JSON，禁止输出pass2/pass3/pass4，禁止创建物理实体、物理量、关系或约束。
定位要求：
1. bbox_1000必须紧密包围实际图元，不得把邻近文字、导线、元件或大片无关空白包含进来；曲线bbox是全部关键点的最小包围框。
2. 线段和箭头关键点必须是真实端点；曲线关键点沿曲线中心线按顺序采样；不能把bbox中心当作端点。
3. 每张图的width/height已在盲化记录中给出。坐标必须相对于该张原始题图计算后归一化到0–1000，不能相对于网页截图、缩放图或多图拼接画布计算。
4. text_glyph只框住对应文字墨迹并填写原文；物体轮廓与其文字标签必须分成不同节点。
5. 只有circle/arc填写真实可见的center_1000和正radius_1000；其他所有类型必须写center_1000=[-1,-1]、radius_1000=-1。
6. 输出前逐节点复核。视觉不确定时降低confidence并记录ambiguity，禁止伪精确。
7. ID按图像顺序从v001连续编号，多图时先按images数组顺序，再按每图从上到下、从左到右。

只输出符合Pass 1 Schema的JSON object，不要Markdown或解释。{repair}{notes}"""


def correction_prompt(
    problem: dict[str, Any],
    current_pass1: dict[str, Any],
    round_index: int,
    attempt: int,
    prior_patch: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> str:
    image_ids = [image["image_id"] for image in problem.get("images", [])]
    repair = ""
    if prior_patch is not None:
        repair = (
            "\n上一轮补丁如下：\n" + compact_json(prior_patch)
            + "\n补丁或应用后Pass 1存在以下错误，必须修复：\n" + compact_json(issues)
        )
    return f"""任务阶段：PASS_1_RENDERED_OVERLAY_VERIFICATION

问题编号：{problem['problem_id']}
纠错轮次：{round_index}，本轮第{attempt}次补丁尝试。
附件顺序：前{len(image_ids)}张是原始题图，image_id依次为{compact_json(image_ids)}；后{len(image_ids)}张是与其一一对应的候选节点叠加图。

当前Pass 1候选：
{compact_json(current_pass1)}

必须将原图与叠加图逐节点对照：
- bbox是否紧密覆盖正确对象；
- keypoints是否落在真实线、箭头或曲线上并按顺序排列；
- center/radius是否属于正确圆或圆弧；
- OCR文本、type、subtype、image_id是否正确；
- 是否漏掉与解题相关的图元，或加入了无关/重复节点。

只输出最小补丁JSON。完全正确时verdict=accept且operations=[]；否则verdict=correct并只列必要操作。
replace提供完整替换节点；delete删除错误或重复节点；append补充遗漏节点。不要重写未变节点。{repair}"""


def panel_pass1_prompt(
    *,
    problem: dict[str, Any],
    panel: PanelSpec,
    attempt: int,
    prior: dict[str, Any] | None,
    issues: list[dict[str, str]],
    review_notes: list[dict[str, str]] | None,
    semantic_inventory: list[dict[str, Any]] | None,
) -> str:
    repair = ""
    if prior is not None:
        repair = (
            "\n上一轮当前面板的像素输出如下：\n" + compact_json(prior)
            + "\n必须修复这些本地校验错误：\n" + compact_json(issues)
        )
    notes = ""
    if review_notes:
        notes = "\n人工退回意见（仅处理与当前面板有关的内容）：\n" + compact_json(review_notes)
    inventory = ""
    if semantic_inventory:
        inventory = (
            "\n下面是一次失败闭环留下的同图语义清单，仅可用于核对可能存在的type/text；"
            "旧几何坐标完全不可信，不得复制。当前面板不存在的节点不要输出：\n"
            + compact_json(semantic_inventory)
        )
    contract = panel_contract(panel, problem["problem_id"])
    return f"""任务阶段：PASS_1_PANEL_PIXEL_EXTRACTION

这是原图 {panel.image_id} 的不重叠面板 {panel.panel_id}，第{attempt}次生成或结构修复。
第一张附件是未经标记的原始面板裁剪；第二张附件是尺寸完全相同的像素网格副本。
面板契约：
{compact_json(contract)}

盲化题目记录仅用于判断哪些可见图元与解题可能相关：
{compact_json(problem)}

坐标硬规则：
1. 只使用当前裁剪面板左上角为(0,0)的原始像素整数；x范围0..{panel.width_px}，y范围0..{panel.height_px}。
2. 禁止输出0–1000归一化坐标，禁止使用原图全局坐标、网页截图坐标或模型内部缩放坐标。
3. bbox_px严格是[x_min,y_min,x_max,y_max]，后两项是最大坐标，绝不是width/height；必须满足x_min<=x_max、y_min<=y_max，并紧密包围墨迹。
4. keypoints_px必须落在真实线/曲线中心线上并位于bbox_px内；line_segment/polyline/curve/arc/arrow/graph_curve至少给两个关键点。
5. 只有type=circle或arc才填写真实center_px和正radius_px；其他所有类型（包括point_mark/object_shape/component_symbol）必须写center_px=[-1,-1]、radius_px=-1。
6. 只提取当前面板实际可见的节点，不跨空白带猜测相邻面板内容，不创建物理实体或关系。
7. 两张附件是同一裁剪；像素网格仅用于读坐标，不是题图图元。
8. 节点ID只需在当前面板内唯一；本地程序会跨面板稳定重编号。

只输出符合PASS1 PIXEL Schema的JSON object。元数据字段必须逐值复制面板契约。{inventory}{repair}{notes}"""


def panel_correction_prompt(
    *,
    problem: dict[str, Any],
    panel: PanelSpec,
    panel_view: dict[str, Any],
    round_index: int,
    attempt: int,
    prior_patch: dict[str, Any] | None,
    issues: list[dict[str, str]],
) -> str:
    repair = ""
    if prior_patch is not None:
        repair = (
            "\n上一补丁如下：\n" + compact_json(prior_patch)
            + "\n必须修复这些本地校验错误：\n" + compact_json(issues)
        )
    return f"""任务阶段：PASS_1_PANEL_PIXEL_VERIFICATION

问题编号：{problem['problem_id']}；面板：{panel.panel_id}；纠错轮次：{round_index}；本轮尝试：{attempt}。
附件顺序严格为：①原始面板裁剪；②同尺寸像素网格；③同尺寸候选节点叠加图。
三张附件的宽高都对应crop_size_px={compact_json([panel.width_px, panel.height_px])}，坐标原点均为面板左上角。

当前面板像素候选：
{compact_json(panel_view)}

逐节点检查bbox_px、keypoints_px、center_px/radius_px、OCR、type/subtype，并检查遗漏或重复。
必须输出crop_pixels坐标；禁止使用0–1000坐标或原图全局坐标。完全正确时verdict=accept且operations=[]；
否则只输出最小replace/delete/append补丁。append节点只需使用面板内非空临时ID，本地会重新分配全局唯一ID；
不得猜测其他面板的编号。若你打算给出的replace节点与当前节点逐字段相同，说明该项无需修改，必须直接返回accept，
不得提交无实际变化的correct补丁。bbox_px必须使用[x_min,y_min,x_max,y_max]而不是[x,y,width,height]；
非circle/arc类型即使外观像圆点，也必须写center_px=[-1,-1]、radius_px=-1。
补丁元数据problem_id/image_id/panel_id/coordinate_space必须与当前面板一致。{repair}"""


def downstream_prompt(
    problem: dict[str, Any],
    pass1: dict[str, Any],
    attempt: int,
    prior: dict[str, Any] | None,
    issues: list[dict[str, str]],
    review_notes: list[dict[str, str]] | None,
) -> str:
    repair = ""
    if prior is not None:
        repair = (
            "\n上一轮Pass 2–4如下：\n" + compact_json(prior)
            + "\n必须修复这些本地校验错误：\n" + compact_json(issues)
        )
    notes = ""
    if review_notes:
        notes = "\n人工退回意见（涉及Pass 2–4的部分必须处理）：\n" + compact_json(review_notes)
    return f"""任务阶段：PASS_2_TO_PASS_4_FROM_VERIFIED_PASS_1

问题编号：{problem['problem_id']}，这是第{attempt}次生成或修复。
唯一允许使用的盲化题目记录：
{compact_json(problem)}

已经由独立视觉流程生成并经叠加图复核的Pass 1：
{compact_json(pass1)}

不得修改、删除或新增Pass 1视觉节点。完成Pass 2、Pass 3和Pass 4；不得求解，不得读取或猜测答案与标准解析。
Pass 4的visual_nodes必须逐字段复制上述Pass 1；所有引用、provenance和evidence必须闭合。
最终只输出一个JSON object，顶层恰好为：
{{"pass2":<完整Pass2>,"pass3":<完整Pass3>,"pass4":<完整Pass4>}}
    不要输出Pass 1、Markdown或解释。{repair}{notes}"""


def pass4_recovery_prompt(
    *,
    problem: dict[str, Any],
    pass1: dict[str, Any],
    prefix: dict[str, Any],
    attempt: int,
    prior_pass4: dict[str, Any] | None,
    issues: list[dict[str, str]],
    review_notes: list[dict[str, str]] | None,
) -> str:
    repair = ""
    if prior_pass4 is not None:
        repair = (
            "\n上一轮Pass 4候选如下：\n" + compact_json(prior_pass4)
            + "\n必须修复这些本地校验错误：\n" + compact_json(issues)
        )
    notes = ""
    if review_notes:
        notes = "\n人工退回意见（仅处理涉及Pass 4的部分）：\n" + compact_json(review_notes)
    return f"""任务阶段：PASS_4_ONLY_RECOVERY_FROM_VALIDATED_PREFIX

问题编号：{problem['problem_id']}，这是第{attempt}次Pass 4定向恢复。
唯一允许使用的盲化题目记录：
{compact_json(problem)}

已复核且不可修改的Pass 1：
{compact_json(pass1)}

已通过本地严格校验且不可修改的Pass 2与Pass 3：
{compact_json(prefix)}

只重新生成Pass 4。必须遵守以下不可退化约束：
1. visual_nodes逐字段复制Pass 1；
2. physical_nodes完整覆盖Pass 2与Pass 3合并后的全部实体ID；
3. text_mentions、quantities、constraints和query_target逐字段继承Pass 3；
4. bindings合并已有直接证据，relations只保留IMAGE或TEXT直接支持的observed关系；
5. provenance与evidence严格一致，所有引用闭合；不得求解或加入DERIVED/CONSTRUCTED内容；
6. 禁止用空数组替代上述已有非空集合。

最终只输出一个符合Pass 4 Schema的完整JSON object。允许外层仅包一层{{"pass4": ...}}，
但即使输出完整Pass 2–4包装，本地也只会读取pass4并丢弃对Pass 2/3的任何改写。
不要输出Markdown或解释。{repair}{notes}"""


def validate_correction_patch(
    patch: Any,
    problem_id: str,
    current_pass1: dict[str, Any],
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not isinstance(patch, dict) or set(patch) != {"problem_id", "verdict", "operations", "summary"}:
        return [{"level": "error", "code": "patch_shape", "path": "$", "message": "纠错补丁顶层字段不正确"}]
    if patch.get("problem_id") != problem_id:
        issues.append({"level": "error", "code": "problem_id_mismatch", "path": "$.problem_id", "message": "problem_id不匹配"})
    verdict = patch.get("verdict")
    operations = patch.get("operations")
    if verdict not in {"accept", "correct"}:
        issues.append({"level": "error", "code": "patch_verdict", "path": "$.verdict", "message": "verdict只允许accept/correct"})
    if not isinstance(patch.get("summary"), str):
        issues.append({"level": "error", "code": "patch_summary", "path": "$.summary", "message": "summary必须是字符串"})
    if not isinstance(operations, list):
        return issues + [{"level": "error", "code": "patch_operations", "path": "$.operations", "message": "operations必须是数组"}]
    if verdict == "accept" and operations:
        issues.append({"level": "error", "code": "accept_has_operations", "path": "$.operations", "message": "accept时operations必须为空"})
    if verdict == "correct" and not operations:
        issues.append({"level": "error", "code": "correct_without_operations", "path": "$.operations", "message": "correct时至少需要一个操作"})
    known_ids = {node.get("id") for node in current_pass1.get("visual_nodes", []) if isinstance(node, dict)}
    targeted: set[str] = set()
    appended_ids: set[str] = set()
    for index, operation in enumerate(operations):
        path = f"$.operations[{index}]"
        if not isinstance(operation, dict) or set(operation) != {"action", "target_id", "node", "reason"}:
            issues.append({"level": "error", "code": "operation_shape", "path": path, "message": "操作字段不正确"})
            continue
        action, target_id, node = operation.get("action"), operation.get("target_id"), operation.get("node")
        if not isinstance(operation.get("reason"), str) or not operation["reason"].strip():
            issues.append({"level": "error", "code": "operation_reason", "path": f"{path}.reason", "message": "reason不能为空"})
        if action not in {"replace", "delete", "append"}:
            issues.append({"level": "error", "code": "operation_action", "path": f"{path}.action", "message": "未知操作类型"})
            continue
        if action in {"replace", "delete"}:
            if target_id not in known_ids:
                issues.append({"level": "error", "code": "unknown_target", "path": f"{path}.target_id", "message": "target_id不是现有视觉节点"})
            if target_id in targeted:
                issues.append({"level": "error", "code": "duplicate_target", "path": f"{path}.target_id", "message": "同一节点不能被多次修改"})
            targeted.add(target_id)
        if action == "replace":
            if not isinstance(node, dict) or set(node) != VISUAL_NODE_KEYS:
                issues.append({"level": "error", "code": "replacement_shape", "path": f"{path}.node", "message": "replace必须提供完整视觉节点"})
            elif node.get("id") != target_id:
                issues.append({"level": "error", "code": "replacement_id", "path": f"{path}.node.id", "message": "replace节点ID必须等于target_id"})
        elif action == "delete":
            if node is not None:
                issues.append({"level": "error", "code": "delete_node", "path": f"{path}.node", "message": "delete的node必须为null"})
        else:
            if target_id != "":
                issues.append({"level": "error", "code": "append_target", "path": f"{path}.target_id", "message": "append的target_id必须为空字符串"})
            if not isinstance(node, dict) or set(node) != VISUAL_NODE_KEYS:
                issues.append({"level": "error", "code": "append_shape", "path": f"{path}.node", "message": "append必须提供完整视觉节点"})
            elif not isinstance(node.get("id"), str) or node["id"] in known_ids or node["id"] in appended_ids:
                issues.append({"level": "error", "code": "append_id", "path": f"{path}.node.id", "message": "append临时ID必须唯一且不能与现有ID冲突"})
            elif isinstance(node, dict):
                appended_ids.add(node["id"])
    return issues


def allocate_append_ids(
    patch: dict[str, Any],
    current_pass1: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace model-provided append IDs with deterministic collision-free temporary IDs.

    Append IDs carry no semantics and are normalized again by apply_correction_patch.  The
    model only sees one panel, so it cannot reliably know IDs reserved by other panels.
    """
    normalized = deepcopy(patch)
    reserved = {
        node.get("id")
        for node in current_pass1.get("visual_nodes", [])
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    allocation_audit: list[dict[str, Any]] = []
    next_index = 1
    for operation_index, operation in enumerate(normalized.get("operations", [])):
        if not isinstance(operation, dict) or operation.get("action") != "append":
            continue
        node = operation.get("node")
        if not isinstance(node, dict):
            continue
        while True:
            allocated = f"__panel_append_{next_index:03d}"
            next_index += 1
            if allocated not in reserved:
                break
        original = node.get("id")
        node["id"] = allocated
        reserved.add(allocated)
        allocation_audit.append({
            "operation_index": operation_index,
            "model_temporary_id": original,
            "allocated_temporary_id": allocated,
        })
    return normalized, allocation_audit


def apply_correction_patch(
    current_pass1: dict[str, Any],
    patch: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = deepcopy(current_pass1)
    if patch["verdict"] == "accept":
        return candidate, []
    replacements: dict[str, dict[str, Any]] = {}
    deletions: set[str] = set()
    additions: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for operation in patch["operations"]:
        action, target_id, node = operation["action"], operation["target_id"], operation["node"]
        if action == "replace":
            replacements[target_id] = deepcopy(node)
        elif action == "delete":
            deletions.add(target_id)
        else:
            additions.append(deepcopy(node))
        audit.append({"action": action, "target_id": target_id, "reason": operation["reason"]})

    ordered: list[tuple[str, dict[str, Any]]] = []
    for node in candidate["visual_nodes"]:
        old_id = node["id"]
        if old_id in deletions:
            continue
        ordered.append((old_id, replacements.get(old_id, node)))
    ordered.extend((node["id"], node) for node in additions)
    id_map: dict[str, str] = {}
    normalized_nodes: list[dict[str, Any]] = []
    for index, (old_id, node) in enumerate(ordered, start=1):
        new_id = f"v{index:03d}"
        id_map[old_id] = new_id
        normalized = deepcopy(node)
        normalized["id"] = new_id
        normalized_nodes.append(normalized)
    candidate["visual_nodes"] = normalized_nodes
    for ambiguity in candidate.get("ambiguities", []):
        if not isinstance(ambiguity, dict):
            continue
        for field in ("candidate_ids", "evidence_visual_ids"):
            values = ambiguity.get(field)
            if isinstance(values, list):
                ambiguity[field] = list(dict.fromkeys(id_map[value] for value in values if value in id_map))
    return candidate, audit


def validate_pixel_correction_patch(
    patch: Any,
    *,
    problem_id: str,
    panel: PanelSpec,
    current_panel_view: dict[str, Any],
) -> list[dict[str, str]]:
    expected_top = {
        "problem_id", "image_id", "panel_id", "coordinate_space",
        "verdict", "operations", "summary",
    }
    if not isinstance(patch, dict) or set(patch) != expected_top:
        return [{"level": "error", "code": "pixel_patch_shape", "path": "$", "message": "像素纠错补丁顶层字段不正确"}]
    issues: list[dict[str, str]] = []
    expected_metadata = {
        "problem_id": problem_id,
        "image_id": panel.image_id,
        "panel_id": panel.panel_id,
        "coordinate_space": "crop_pixels",
    }
    for key, expected in expected_metadata.items():
        if patch.get(key) != expected:
            issues.append({"level": "error", "code": "pixel_patch_metadata", "path": f"$.{key}", "message": f"{key}必须等于{expected!r}"})
    verdict, operations = patch.get("verdict"), patch.get("operations")
    if verdict not in {"accept", "correct"}:
        issues.append({"level": "error", "code": "pixel_patch_verdict", "path": "$.verdict", "message": "verdict只允许accept/correct"})
    if not isinstance(patch.get("summary"), str):
        issues.append({"level": "error", "code": "pixel_patch_summary", "path": "$.summary", "message": "summary必须为字符串"})
    if not isinstance(operations, list):
        return issues + [{"level": "error", "code": "pixel_patch_operations", "path": "$.operations", "message": "operations必须为数组"}]
    if verdict == "accept" and operations:
        issues.append({"level": "error", "code": "accept_has_operations", "path": "$.operations", "message": "accept时operations必须为空"})
    if verdict == "correct" and not operations:
        issues.append({"level": "error", "code": "correct_without_operations", "path": "$.operations", "message": "correct时必须至少有一个操作"})
    known_ids = {node["id"] for node in current_panel_view.get("visual_nodes", []) if isinstance(node, dict) and "id" in node}
    targeted: set[str] = set()
    for index, operation in enumerate(operations):
        path = f"$.operations[{index}]"
        if not isinstance(operation, dict) or set(operation) != {"action", "target_id", "node", "reason"}:
            issues.append({"level": "error", "code": "pixel_operation_shape", "path": path, "message": "操作字段不正确"})
            continue
        action, target_id, node = operation.get("action"), operation.get("target_id"), operation.get("node")
        if not isinstance(operation.get("reason"), str) or not operation["reason"].strip():
            issues.append({"level": "error", "code": "pixel_operation_reason", "path": f"{path}.reason", "message": "reason不能为空"})
        if action not in {"replace", "delete", "append"}:
            issues.append({"level": "error", "code": "pixel_operation_action", "path": f"{path}.action", "message": "未知操作"})
            continue
        if action in {"replace", "delete"}:
            if target_id not in known_ids:
                issues.append({"level": "error", "code": "pixel_unknown_target", "path": f"{path}.target_id", "message": "target_id不属于当前面板"})
            if target_id in targeted:
                issues.append({"level": "error", "code": "pixel_duplicate_target", "path": f"{path}.target_id", "message": "同一节点不能重复修改"})
            targeted.add(target_id)
        if action == "delete":
            if node is not None:
                issues.append({"level": "error", "code": "pixel_delete_node", "path": f"{path}.node", "message": "delete的node必须为null"})
            continue
        if action == "append" and target_id != "":
            issues.append({"level": "error", "code": "pixel_append_target", "path": f"{path}.target_id", "message": "append的target_id必须为空"})
        if not isinstance(node, dict) or set(node) != PIXEL_NODE_KEYS:
            issues.append({"level": "error", "code": "pixel_patch_node_shape", "path": f"{path}.node", "message": "replace/append必须提供完整像素节点"})
            continue
        if action == "replace" and node.get("id") != target_id:
            issues.append({"level": "error", "code": "pixel_replacement_id", "path": f"{path}.node.id", "message": "replace节点ID必须等于target_id"})
        if action == "append":
            if not isinstance(node.get("id"), str) or not node["id"].strip():
                issues.append({"level": "error", "code": "pixel_append_id", "path": f"{path}.node.id", "message": "append节点必须提供非空临时ID；本地程序会重新分配全局唯一ID"})
        node_document = {**panel_contract(panel, problem_id), "visual_nodes": [node], "ambiguities": []}
        issues.extend(validate_panel_document(node_document, panel=panel, problem_id=problem_id))
    return issues


def evaluate_pixel_correction_patch(
    *,
    patch: dict[str, Any],
    problem: dict[str, Any],
    panel: PanelSpec,
    current_panel_view: dict[str, Any],
    current_pass1: dict[str, Any],
    seen_hashes: set[str],
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
]:
    """Validate, convert, apply, and classify one pixel correction without I/O."""
    problem_id = problem["problem_id"]
    issues = validate_pixel_correction_patch(
        patch,
        problem_id=problem_id,
        panel=panel,
        current_panel_view=current_panel_view,
    )
    if issues:
        return None, issues, [], [], patch.get("verdict")
    standard_patch = pixel_patch_to_standard(patch, panel)
    standard_patch, append_id_audit = allocate_append_ids(standard_patch, current_pass1)
    issues = validate_correction_patch(standard_patch, problem_id, current_pass1)
    if issues:
        return None, issues, [], append_id_audit, patch.get("verdict")
    candidate, operations_audit = apply_correction_patch(current_pass1, standard_patch)
    issues = validate_pass1(candidate, problem)
    effective_verdict = patch.get("verdict")
    if not issues and patch.get("verdict") == "correct" and candidate == current_pass1:
        effective_verdict = "implicit_accept_no_effect"
    elif not issues and patch.get("verdict") == "correct" and json_sha256(candidate) in seen_hashes:
        issues.append({
            "level": "error", "code": "correction_oscillation", "path": "$.operations",
            "message": "补丁使Pass 1回到本面板此前已经出现过的状态，拒绝振荡修改",
        })
    return candidate, issues, operations_audit, append_id_audit, effective_verdict


def _semantic_inventory(pass1: dict[str, Any] | None, image_id: str) -> list[dict[str, Any]] | None:
    if pass1 is None:
        return None
    inventory = [
        {
            "id": node.get("id", ""), "type": node.get("type", ""),
            "subtype": node.get("subtype", ""), "text": node.get("text", ""),
            "confidence": node.get("confidence", ""),
        }
        for node in pass1.get("visual_nodes", [])
        if isinstance(node, dict) and node.get("image_id") == image_id
    ]
    return inventory or None


def prepare_panel_assets(
    *,
    problem: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
) -> tuple[list[PanelSpec], dict[str, tuple[Path, Path]]]:
    problem_id = problem["problem_id"]
    panels = build_problem_panels(problem, dataset_dir)
    panel_dir = run_dir / "pixel_panels" / problem_id
    assets: dict[str, tuple[Path, Path]] = {}
    write_json_atomic(run_dir / f"{problem_id}.pass1.panel_manifest.json", [panel.as_record() for panel in panels])
    for panel in panels:
        assets[panel.panel_id] = render_panel_assets(
            dataset_dir=dataset_dir,
            panel=panel,
            output_dir=panel_dir,
        )
    return panels, assets


def load_pixel_correction_checkpoint(
    *,
    resume_path: Path,
    problem: dict[str, Any],
    dataset_dir: Path,
) -> tuple[dict[str, Any] | None, set[str], str]:
    """Load only a locally verifiable checkpoint produced by the pixel-panel pipeline."""
    problem_id = problem["problem_id"]
    audit_path = resume_path.parent / f"{problem_id}.pass1.correction_audit.json"
    manifest_path = resume_path.parent / f"{problem_id}.pass1.panel_manifest.json"
    if not audit_path.is_file() or not manifest_path.is_file():
        return None, set(), "missing_pixel_checkpoint_audit_or_manifest"
    try:
        candidate = read_json(resume_path)
        audit = read_json(audit_path)
        stored_manifest = read_json(manifest_path)
        panels = build_problem_panels(problem, dataset_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, set(), f"pixel_checkpoint_read_error:{exc}"
    if audit.get("mode") != "per_panel_crop_pixels":
        return None, set(), "not_a_pixel_panel_checkpoint"
    if audit.get("final_pass1_sha256") != json_sha256(candidate):
        return None, set(), "pixel_checkpoint_hash_mismatch"
    issues = validate_pass1(candidate, problem)
    if issues:
        codes = ",".join(sorted({str(issue.get("code", "invalid")) for issue in issues}))
        return None, set(), f"pixel_checkpoint_pass1_invalid:{codes}"
    current_manifest = [panel.as_record() for panel in panels]
    if stored_manifest != current_manifest:
        return None, set(), "pixel_checkpoint_panel_manifest_mismatch"
    panel_records = {panel.panel_id: panel.as_record() for panel in panels}
    accepted: set[str] = set()
    for entry in audit.get("panels", []):
        if not isinstance(entry, dict) or entry.get("status") != "accepted":
            continue
        record = entry.get("panel")
        panel_id = record.get("panel_id") if isinstance(record, dict) else None
        if panel_id in panel_records and record == panel_records[panel_id] and isinstance(entry.get("accepted_round"), int):
            accepted.add(panel_id)
    return candidate, accepted, "validated_pixel_checkpoint"


def load_checkpoint_panel_hashes(resume_path: Path, problem_id: str) -> dict[str, set[str]]:
    audit_path = resume_path.parent / f"{problem_id}.pass1.correction_audit.json"
    try:
        audit = read_json(audit_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    history: dict[str, set[str]] = {}
    for entry in audit.get("panels", []) if isinstance(audit, dict) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("panel"), dict):
            continue
        panel_id = entry["panel"].get("panel_id")
        if not isinstance(panel_id, str):
            continue
        hashes = history.setdefault(panel_id, set())
        for round_record in entry.get("rounds", []):
            if not isinstance(round_record, dict):
                continue
            for field in ("input_sha256", "output_sha256", "input_pass1_sha256", "output_pass1_sha256"):
                value = round_record.get(field)
                if isinstance(value, str) and value:
                    hashes.add(value)
    return history


def discover_latest_pixel_checkpoint(
    *,
    workspace: Path,
    problem: dict[str, Any],
    dataset_dir: Path,
) -> tuple[dict[str, Any] | None, set[str], Path | None, dict[str, set[str]], str]:
    """Find the newest fully authenticated local Pass 1 checkpoint for one problem."""
    problem_id = problem["problem_id"]
    candidates: list[Path] = []
    for parent_name in ("batch_runs", "reannotation_runs", "recovery_runs"):
        parent = workspace / parent_name
        if not parent.is_dir():
            continue
        for artifact_name in (
            f"{problem_id}.pass1.verified.json",
            f"{problem_id}.pass1.unverified_after_correction.json",
        ):
            candidates.extend(parent.glob(f"*/{artifact_name}"))
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    rejection_reasons: list[str] = []
    for path in candidates:
        candidate, accepted, reason = load_pixel_correction_checkpoint(
            resume_path=path,
            problem=problem,
            dataset_dir=dataset_dir,
        )
        if candidate is not None:
            return (
                candidate,
                accepted,
                path,
                load_checkpoint_panel_hashes(path, problem_id),
                reason,
            )
        rejection_reasons.append(f"{path.parent.name}:{reason}")
    reason = "no_checkpoint_artifacts" if not candidates else "no_valid_checkpoint:" + "|".join(rejection_reasons[:5])
    return None, set(), None, {}, reason


def discover_valid_panel_generation_artifact(
    *,
    workspace: Path,
    problem: dict[str, Any],
    panel: PanelSpec,
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any]]:
    """Recover a previous API panel generation after conservative normalization."""

    problem_id = problem["problem_id"]
    candidates: list[Path] = []
    for parent_name in ("batch_runs", "reannotation_runs", "recovery_runs"):
        parent = workspace / parent_name
        if not parent.is_dir():
            continue
        candidates.extend(parent.glob(
            f"*/{problem_id}.pass1.{panel.panel_id}.generate.attempt*.json"
        ))
    candidates = [
        path for path in candidates
        if not path.name.endswith(".api.json")
        and not path.name.endswith(".normalized.json")
    ]
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    rejected: list[dict[str, Any]] = []
    for path in candidates:
        try:
            manifest = read_json(path.parent / f"{problem_id}.pass1.panel_manifest.json")
            raw = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"path": str(path), "reason": f"read_error:{exc}"})
            continue
        if not isinstance(manifest, list) or panel.as_record() not in manifest:
            rejected.append({"path": str(path), "reason": "panel_manifest_mismatch"})
            continue
        normalized, normalization_audit = normalize_panel_document(raw, panel=panel)
        issues = validate_panel_document(
            normalized,
            panel=panel,
            problem_id=problem_id,
        )
        if not issues:
            return normalized, path, {
                "status": "recovered_valid_panel_generation",
                "source_path": str(path),
                "source_sha256": json_sha256(raw),
                "normalized_sha256": json_sha256(normalized),
                "normalization": normalization_audit,
                "rejected_newer_candidates": rejected,
            }
        rejected.append({
            "path": str(path),
            "reason": "strict_validation_failed",
            "issue_codes": sorted({str(issue.get("code", "invalid")) for issue in issues}),
            "normalization": normalization_audit,
        })
    return None, None, {
        "status": "no_valid_panel_generation_artifact",
        "candidate_count": len(candidates),
        "rejected_candidates": rejected[:8],
    }


def discover_implicit_accept_pixel_patch_artifact(
    *,
    workspace: Path,
    problem: dict[str, Any],
    panel: PanelSpec,
    current_panel_view: dict[str, Any],
    current_pass1: dict[str, Any],
    seen_hashes: set[str],
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any]]:
    """Recover a previous API correction transition under strict local authentication.

    A normalized no-effect correction is intrinsically safe. A real transition
    is reusable only when applying the patch to the current Pass 1 reproduces
    the sibling corrected artifact byte-for-byte and its successful API log
    agrees with the panel and verdict.
    """

    problem_id = problem["problem_id"]
    candidates: list[Path] = []
    for parent_name in ("batch_runs", "reannotation_runs", "recovery_runs"):
        parent = workspace / parent_name
        if parent.is_dir():
            candidates.extend(parent.glob(
                f"*/{problem_id}.pass1.{panel.panel_id}.correct.r*.attempt*.patch.json"
            ))
    candidates = [path for path in candidates if ".normalized." not in path.name]
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    rejected: list[dict[str, Any]] = []
    for path in candidates:
        try:
            manifest = read_json(path.parent / f"{problem_id}.pass1.panel_manifest.json")
            raw_patch = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"path": str(path), "reason": f"read_error:{exc}"})
            continue
        if not isinstance(manifest, list) or panel.as_record() not in manifest:
            rejected.append({"path": str(path), "reason": "panel_manifest_mismatch"})
            continue
        normalized_patch, normalization_audit = normalize_pixel_correction_patch(raw_patch)
        candidate, issues, operations_audit, append_audit, verdict = evaluate_pixel_correction_patch(
            patch=normalized_patch,
            problem=problem,
            panel=panel,
            current_panel_view=current_panel_view,
            current_pass1=current_pass1,
            seen_hashes=seen_hashes,
        )
        transition_source = ""
        if not issues and candidate == current_pass1 and verdict == "implicit_accept_no_effect":
            transition_source = "normalized_implicit_no_effect"
        else:
            corrected_path = Path(str(path)[:-len(".patch.json")] + ".corrected.json")
            api_log_path = Path(str(path)[:-len(".patch.json")] + ".api.json")
            try:
                stored_candidate = read_json(corrected_path)
                api_log = read_json(api_log_path)
            except (OSError, ValueError, json.JSONDecodeError):
                stored_candidate, api_log = None, None
            authenticated_log = (
                isinstance(api_log, dict)
                and api_log.get("ok") is True
                and api_log.get("pixel_panel") == panel.as_record()
                and api_log.get("model_verdict") == raw_patch.get("verdict")
                and api_log.get("effective_verdict") == verdict
                and api_log.get("operation_count") == len(normalized_patch.get("operations", []))
            )
            if not issues and candidate is not None and stored_candidate == candidate and authenticated_log:
                transition_source = "authenticated_corrected_artifact_and_api_log"
        if transition_source:
            return normalized_patch, path, {
                "status": "recovered_pixel_correction_transition",
                "source_path": str(path),
                "source_sha256": json_sha256(raw_patch),
                "normalized_sha256": json_sha256(normalized_patch),
                "input_pass1_sha256": json_sha256(current_pass1),
                "output_pass1_sha256": json_sha256(candidate),
                "effective_verdict": verdict,
                "transition_source": transition_source,
                "normalization": normalization_audit,
                "operations": operations_audit,
                "append_id_allocation": append_audit,
                "rejected_newer_candidates": rejected,
            }
        rejected.append({
            "path": str(path),
            "reason": "not_a_valid_authenticated_correction_transition",
            "issue_codes": sorted({str(issue.get("code", "invalid")) for issue in issues}),
            "effective_verdict": verdict,
        })
    return None, None, {
        "status": "no_reusable_pixel_correction_transition",
        "candidate_count": len(candidates),
        "rejected_candidates": rejected[:8],
    }


def generate_pass1_by_panels(
    *,
    api_client: PhysGraphAPIClient,
    args: argparse.Namespace,
    system_prompt: str,
    problem: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
    workspace: Path | None = None,
    progress: dict[str, Any],
    progress_path: Path,
    review_notes: list[dict[str, str]] | None,
    resume_inventory: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]], list[PanelSpec], dict[str, tuple[Path, Path]]]:
    problem_id = problem["problem_id"]
    panels, assets = prepare_panel_assets(
        problem=problem,
        dataset_dir=dataset_dir,
        run_dir=run_dir,
    )
    documents: list[dict[str, Any]] = []
    for panel_index, panel in enumerate(panels, start=1):
        raw_path, grid_path = assets[panel.panel_id]
        if (
            workspace is not None
            and not getattr(args, "no_auto_resume_checkpoints", False)
            and not getattr(args, "only_rejected", False)
            and not getattr(args, "ignore_pass1_cache", False)
        ):
            recovered, recovered_path, recovery_audit = discover_valid_panel_generation_artifact(
                workspace=workspace,
                problem=problem,
                panel=panel,
            )
            if recovered is not None and recovered_path is not None:
                write_json_atomic(
                    run_dir / f"{problem_id}.pass1.{panel.panel_id}.generate.recovered.normalized.json",
                    recovered,
                )
                write_json_atomic(
                    run_dir / f"{problem_id}.pass1.{panel.panel_id}.generate.recovery_audit.json",
                    recovery_audit,
                )
                progress["pass1_panel_artifact_hits"] = progress.get("pass1_panel_artifact_hits", 0) + 1
                progress.setdefault("pass1_panel_reuse", []).append({
                    "problem_id": problem_id,
                    "panel_id": panel.panel_id,
                    "path": str(recovered_path),
                })
                persist_progress(
                    progress_path,
                    progress,
                    phase=f"pass1_generate_panel_{panel_index}_of_{len(panels)}_artifact_reused",
                )
                documents.append(recovered)
                continue
        prior: dict[str, Any] | None = None
        issues: list[dict[str, str]] = []
        accepted: dict[str, Any] | None = None
        for attempt in range(1, args.pass1_attempts + 1):
            stem = f"{problem_id}.pass1.{panel.panel_id}.generate.attempt{attempt}"
            output, log = run_dir / f"{stem}.json", run_dir / f"{stem}.api.json"
            try:
                document, result = api_client.request_json(
                    model=args.model,
                    reasoning_effort=args.reasoning,
                    system_prompt=system_prompt,
                    user_prompt=panel_pass1_prompt(
                        problem=problem, panel=panel, attempt=attempt, prior=prior,
                        issues=issues, review_notes=review_notes,
                        semantic_inventory=_semantic_inventory(resume_inventory, panel.image_id),
                    ),
                    image_paths=[raw_path, grid_path],
                    image_detail=args.image_detail,
                    max_output_tokens=args.pass1_max_output_tokens,
                )
                merge_phase_usage(progress, "pass1_generate", result.usage)
                write_json_atomic(output, document)
                persist_progress(
                    progress_path, progress,
                    phase=f"pass1_generate_panel_{panel_index}_of_{len(panels)}_validate",
                )
            except PhysGraphAPIResponseError as exc:
                merge_phase_usage(progress, "pass1_generate", exc.result.usage)
                write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
                issues = [{"level": "error", "code": "api_response_json", "path": "$", "message": str(exc)}]
                persist_progress(progress_path, progress)
                continue
            except (PhysGraphAPIError, OSError, ValueError) as exc:
                write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
                issues = [{"level": "error", "code": "api_request", "path": "$", "message": str(exc)}]
                continue
            normalized_document, normalization_audit = normalize_panel_document(document, panel=panel)
            if normalization_audit["changed"]:
                write_json_atomic(
                    run_dir / f"{stem}.normalized.json",
                    normalized_document,
                )
                write_json_atomic(
                    run_dir / f"{stem}.normalization_audit.json",
                    normalization_audit,
                )
            issues = validate_panel_document(normalized_document, panel=panel, problem_id=problem_id)
            if not issues:
                write_attempt_log(
                    log, endpoint=api_client.endpoint_label, result=result,
                    extra={"pixel_panel": panel.as_record(), "normalization": normalization_audit},
                )
                accepted = normalized_document
                break
            write_attempt_log(
                log, endpoint=api_client.endpoint_label, result=result,
                error=ValueError("像素Pass1本地校验失败：" + "；".join(issue["message"] for issue in issues)),
                extra={"pixel_panel": panel.as_record(), "normalization": normalization_audit},
            )
            prior = normalized_document
        if accepted is None:
            return None, issues, panels, assets
        documents.append(accepted)
    merged = merge_panel_documents(problem_id=problem_id, panels=panels, documents=documents)
    issues = validate_pass1(merged, problem)
    write_json_atomic(run_dir / f"{problem_id}.pass1.pixel_merged.normalized.json", merged)
    if issues:
        return None, issues, panels, assets
    return merged, [], panels, assets


def _finish_panel_correction(
    *,
    current: dict[str, Any],
    accepted: bool,
    issues: list[dict[str, str]],
    audit: dict[str, Any],
    problem: dict[str, Any],
    dataset_dir: Path,
    run_dir: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, str]], dict[str, Any]]:
    problem_id = problem["problem_id"]
    final_overlays, final_manifest = render_pass1_overlays(
        problem=problem, dataset_dir=dataset_dir, pass1=current,
        output_dir=run_dir / "overlays", label="verified_final" if accepted else "unverified_final",
    )
    audit["final_overlay_paths"] = [str(path) for path in final_overlays]
    audit["final_overlay_manifest"] = final_manifest
    audit["final_pass1_sha256"] = json_sha256(current)
    audit["status"] = "accepted" if accepted else "unverified"
    artifact = run_dir / f"{problem_id}.pass1.{'verified' if accepted else 'unverified_after_correction'}.json"
    write_json_atomic(artifact, current)
    write_json_atomic(run_dir / f"{problem_id}.pass1.correction_audit.json", audit)
    return (current if accepted else None), issues, audit


def correct_pass1_by_panels(
    *,
    api_client: PhysGraphAPIClient,
    args: argparse.Namespace,
    system_prompt: str,
    problem: dict[str, Any],
    dataset_dir: Path,
    initial_pass1: dict[str, Any],
    panels: list[PanelSpec],
    assets: dict[str, tuple[Path, Path]],
    run_dir: Path,
    progress: dict[str, Any],
    progress_path: Path,
    preaccepted_panel_ids: set[str] | None = None,
    resumed_from: Path | None = None,
    max_rounds_per_panel: int | None = None,
    prior_panel_hashes: dict[str, set[str]] | None = None,
    workspace: Path | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]], dict[str, Any]]:
    current = deepcopy(initial_pass1)
    problem_id = problem["problem_id"]
    effective_max_rounds = max_rounds_per_panel or args.correction_rounds
    audit: dict[str, Any] = {
        "mode": "per_panel_crop_pixels",
        "acceptance_required_per_panel": True,
        "max_rounds_per_panel": effective_max_rounds,
        "oscillation_detection": True,
        "no_effect_policy": "implicit_accept_no_effect",
        "panels": [],
    }
    if resumed_from is not None:
        audit["resumed_from"] = str(resumed_from)
        audit["preaccepted_panel_ids"] = sorted(preaccepted_panel_ids or set())
    for panel_index, panel in enumerate(panels, start=1):
        panel_audit: dict[str, Any] = {"panel": panel.as_record(), "rounds": []}
        if panel.panel_id in (preaccepted_panel_ids or set()):
            panel_audit.update({
                "status": "accepted",
                "accepted_round": 0,
                "acceptance_source": "validated_resume_checkpoint",
            })
            audit["panels"].append(panel_audit)
            continue
        panel_accepted = False
        issues: list[dict[str, str]] = []
        raw_path, grid_path = assets[panel.panel_id]
        seen_hashes = set((prior_panel_hashes or {}).get(panel.panel_id, set()))
        seen_hashes.add(json_sha256(current))
        starting_round = 1
        if (
            workspace is not None
            and not getattr(args, "no_auto_resume_checkpoints", False)
            and not getattr(args, "only_rejected", False)
            and not getattr(args, "ignore_pass1_cache", False)
        ):
            recovery_view = panel_view_from_standard(
                problem_id=problem_id,
                pass1=current,
                panel=panel,
            )
            recovered_patch, recovered_patch_path, patch_recovery_audit = (
                discover_implicit_accept_pixel_patch_artifact(
                    workspace=workspace,
                    problem=problem,
                    panel=panel,
                    current_panel_view=recovery_view,
                    current_pass1=current,
                    seen_hashes=seen_hashes,
                )
            )
            if recovered_patch is not None and recovered_patch_path is not None:
                recovery_input_hash = json_sha256(current)
                (
                    recovered_candidate,
                    recovery_issues,
                    recovered_operations,
                    recovered_append_ids,
                    recovered_verdict,
                ) = evaluate_pixel_correction_patch(
                    patch=recovered_patch,
                    problem=problem,
                    panel=panel,
                    current_panel_view=recovery_view,
                    current_pass1=current,
                    seen_hashes=seen_hashes,
                )
                if recovery_issues or recovered_candidate is None:
                    raise ValueError("已认证像素纠错转移在复用时未能确定性重放")
                write_json_atomic(
                    run_dir / f"{problem_id}.pass1.{panel.panel_id}.correct.recovered.normalized.patch.json",
                    recovered_patch,
                )
                write_json_atomic(
                    run_dir / f"{problem_id}.pass1.{panel.panel_id}.correct.recovery_audit.json",
                    patch_recovery_audit,
                )
                progress["pass1_patch_artifact_hits"] = progress.get("pass1_patch_artifact_hits", 0) + 1
                progress.setdefault("pass1_patch_reuse", []).append({
                    "problem_id": problem_id,
                    "panel_id": panel.panel_id,
                    "path": str(recovered_patch_path),
                    "effective_verdict": recovered_verdict,
                })
                persist_progress(
                    progress_path,
                    progress,
                    phase=f"pass1_correct_panel_{panel_index}_of_{len(panels)}_transition_artifact_reused",
                )
                panel_audit["rounds"].append({
                    "round": 1,
                    "attempt": 0,
                    "model_verdict": recovered_patch.get("verdict"),
                    "verdict": recovered_verdict,
                    "operations": recovered_operations,
                    "input_sha256": recovery_input_hash,
                    "output_sha256": json_sha256(recovered_candidate),
                    "append_id_allocation": recovered_append_ids,
                    "artifact_recovery": patch_recovery_audit,
                })
                current = recovered_candidate
                seen_hashes.add(json_sha256(current))
                if recovered_verdict in {"accept", "implicit_accept_no_effect"}:
                    panel_audit.update({
                        "status": "accepted",
                        "accepted_round": 1,
                        "acceptance_source": f"recovered_{recovered_verdict}",
                    })
                    audit["panels"].append(panel_audit)
                    continue
                starting_round = 2
        for round_index in range(starting_round, effective_max_rounds + 1):
            panel_view = panel_view_from_standard(problem_id=problem_id, pass1=current, panel=panel)
            overlay_path = run_dir / "pixel_panels" / problem_id / f"{panel.panel_id}.correction_round{round_index}.png"
            render_panel_pixel_overlay(
                raw_crop_path=raw_path,
                pixel_nodes=panel_view["visual_nodes"],
                output_path=overlay_path,
            )
            prior_patch: dict[str, Any] | None = None
            applied = False
            for attempt in range(1, args.correction_attempts + 1):
                stem = f"{problem_id}.pass1.{panel.panel_id}.correct.r{round_index}.attempt{attempt}"
                patch_path = run_dir / f"{stem}.patch.json"
                corrected_path = run_dir / f"{stem}.corrected.json"
                log = run_dir / f"{stem}.api.json"
                try:
                    patch, result = api_client.request_json(
                        model=args.model,
                        reasoning_effort=args.reasoning,
                        system_prompt=system_prompt,
                        user_prompt=panel_correction_prompt(
                            problem=problem, panel=panel, panel_view=panel_view,
                            round_index=round_index, attempt=attempt,
                            prior_patch=prior_patch, issues=issues,
                        ),
                        image_paths=[raw_path, grid_path, overlay_path],
                        image_detail=args.image_detail,
                        max_output_tokens=args.correction_max_output_tokens,
                    )
                    merge_phase_usage(progress, "pass1_correct", result.usage)
                    write_json_atomic(patch_path, patch)
                    persist_progress(
                        progress_path, progress,
                        phase=f"pass1_correct_panel_{panel_index}_of_{len(panels)}_round_{round_index}_validate",
                    )
                except PhysGraphAPIResponseError as exc:
                    merge_phase_usage(progress, "pass1_correct", exc.result.usage)
                    write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
                    issues = [{"level": "error", "code": "api_response_json", "path": "$", "message": str(exc)}]
                    persist_progress(progress_path, progress)
                    continue
                except (PhysGraphAPIError, OSError, ValueError) as exc:
                    write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
                    issues = [{"level": "error", "code": "api_request", "path": "$", "message": str(exc)}]
                    continue
                normalized_patch, patch_normalization_audit = normalize_pixel_correction_patch(patch)
                if patch_normalization_audit["changed"]:
                    write_json_atomic(
                        run_dir / f"{stem}.normalized.patch.json",
                        normalized_patch,
                    )
                    write_json_atomic(
                        run_dir / f"{stem}.patch_normalization_audit.json",
                        patch_normalization_audit,
                    )
                (
                    candidate,
                    issues,
                    operations_audit,
                    append_id_audit,
                    effective_verdict,
                ) = evaluate_pixel_correction_patch(
                    patch=normalized_patch,
                    problem=problem,
                    panel=panel,
                    current_panel_view=panel_view,
                    current_pass1=current,
                    seen_hashes=seen_hashes,
                )
                if issues:
                    write_attempt_log(
                        log, endpoint=api_client.endpoint_label, result=result,
                        error=ValueError("面板像素纠错校验失败：" + "；".join(issue["message"] for issue in issues)),
                        extra={
                            "pixel_panel": panel.as_record(),
                            "patch_normalization": patch_normalization_audit,
                        },
                    )
                    prior_patch = normalized_patch
                    continue
                assert candidate is not None
                write_json_atomic(corrected_path, candidate)
                write_attempt_log(
                    log, endpoint=api_client.endpoint_label, result=result,
                    extra={
                        "pixel_panel": panel.as_record(),
                        "model_verdict": patch["verdict"],
                        "effective_verdict": effective_verdict,
                        "operation_count": len(normalized_patch["operations"]),
                        "patch_normalization": patch_normalization_audit,
                    },
                )
                panel_audit["rounds"].append({
                    "round": round_index, "attempt": attempt,
                    "model_verdict": patch["verdict"], "verdict": effective_verdict,
                    "operations": operations_audit, "input_sha256": json_sha256(current),
                    "output_sha256": json_sha256(candidate), "overlay_path": str(overlay_path),
                    "append_id_allocation": append_id_audit,
                    "patch_normalization": patch_normalization_audit,
                })
                current = candidate
                seen_hashes.add(json_sha256(current))
                applied = True
                if effective_verdict in {"accept", "implicit_accept_no_effect"}:
                    panel_accepted = True
                    panel_audit["accepted_round"] = round_index
                    panel_audit["acceptance_source"] = effective_verdict
                break
            if not applied:
                panel_audit["status"] = "invalid_or_api_failure"
                audit["panels"].append(panel_audit)
                return _finish_panel_correction(
                    current=current, accepted=False, issues=issues, audit=audit,
                    problem=problem, dataset_dir=dataset_dir, run_dir=run_dir,
                )
            if panel_accepted:
                break
        panel_audit["status"] = "accepted" if panel_accepted else "unverified_after_max_rounds"
        audit["panels"].append(panel_audit)
        if not panel_accepted:
            issues = [{
                "stage": "pass1", "level": "error",
                "code": "panel_unverified_after_max_rounds", "path": "$",
                "message": f"面板{panel.panel_id}连续{effective_max_rounds}轮仍产生真实修改，未获得accept",
            }]
            return _finish_panel_correction(
                current=current, accepted=False, issues=issues, audit=audit,
                problem=problem, dataset_dir=dataset_dir, run_dir=run_dir,
            )
    return _finish_panel_correction(
        current=current, accepted=True, issues=[], audit=audit,
        problem=problem, dataset_dir=dataset_dir, run_dir=run_dir,
    )


def pass1_cache_path(workspace: Path, problem_id: str) -> Path:
    return workspace / "pipeline_cache" / "verified_pass1" / f"{problem_id}.json"


def load_pass1_cache(
    workspace: Path,
    problem: dict[str, Any],
) -> dict[str, Any] | None:
    path = pass1_cache_path(workspace, problem["problem_id"])
    if not path.is_file():
        return None
    try:
        envelope = read_json(path)
        pass1 = envelope["pass1"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    validation_issues = validate_pass1(pass1, problem)
    metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
    if validation_issues and metadata.get("allow_reviewed_legacy_geometry") is True:
        validation_issues = validation_errors("pass1", pass1, problem)
    if (
        envelope.get("schema_version") != 2
        or envelope.get("problem_sha256") != json_sha256(problem)
        or envelope.get("pass1_sha256") != json_sha256(pass1)
        or validation_issues
    ):
        return None
    return pass1


def save_pass1_cache(
    workspace: Path,
    problem: dict[str, Any],
    pass1: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    write_json_atomic(
        pass1_cache_path(workspace, problem["problem_id"]),
        {
            "schema_version": 2,
            "problem_id": problem["problem_id"],
            "problem_sha256": json_sha256(problem),
            "pass1_sha256": json_sha256(pass1),
            "verified_at_utc": utc_now(),
            "metadata": metadata,
            "pass1": pass1,
        },
    )


def merge_phase_usage(progress: dict[str, Any], phase: str, usage: dict[str, int]) -> None:
    merge_usage(progress["api_usage"], usage)
    merge_usage(progress["api_usage_by_phase"].setdefault(phase, {}), usage)


def persist_progress(progress_path: Path, progress: dict[str, Any], *, phase: str | None = None) -> None:
    if phase is not None:
        progress["current_phase"] = phase
    progress["updated_at_utc"] = utc_now()
    write_json_atomic(progress_path, progress)


def generate_pass1(
    *,
    api_client: PhysGraphAPIClient,
    args: argparse.Namespace,
    system_prompt: str,
    problem: dict[str, Any],
    image_paths: list[Path],
    run_dir: Path,
    progress: dict[str, Any],
    progress_path: Path,
    review_notes: list[dict[str, str]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    prior: dict[str, Any] | None = None
    issues: list[dict[str, str]] = []
    problem_id = problem["problem_id"]
    for attempt in range(1, args.pass1_attempts + 1):
        output = run_dir / f"{problem_id}.pass1.generate.attempt{attempt}.json"
        log = run_dir / f"{problem_id}.pass1.generate.attempt{attempt}.api.json"
        try:
            document, result = api_client.request_json(
                model=args.model,
                reasoning_effort=args.reasoning,
                system_prompt=system_prompt,
                user_prompt=pass1_prompt(problem, attempt, prior, issues, review_notes),
                image_paths=image_paths,
                image_detail=args.image_detail,
                max_output_tokens=args.pass1_max_output_tokens,
            )
            merge_phase_usage(progress, "pass1_generate", result.usage)
            write_json_atomic(output, document)
            persist_progress(progress_path, progress, phase="pass1_generate_validate")
        except PhysGraphAPIResponseError as exc:
            merge_phase_usage(progress, "pass1_generate", exc.result.usage)
            write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
            issues = [{"level": "error", "code": "api_response_json", "path": "$", "message": str(exc)}]
            persist_progress(progress_path, progress)
            continue
        except (PhysGraphAPIError, OSError, ValueError) as exc:
            write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
            issues = [{"level": "error", "code": "api_request", "path": "$", "message": str(exc)}]
            continue
        issues = validate_pass1(document, problem)
        if not issues:
            write_attempt_log(log, endpoint=api_client.endpoint_label, result=result)
            return document, []
        write_attempt_log(
            log, endpoint=api_client.endpoint_label, result=result,
            error=ValueError("Pass 1本地校验失败：" + "；".join(issue["message"] for issue in issues)),
        )
        prior = document
    return None, issues


def correct_pass1(
    *,
    api_client: PhysGraphAPIClient,
    args: argparse.Namespace,
    system_prompt: str,
    problem: dict[str, Any],
    dataset_dir: Path,
    image_paths: list[Path],
    initial_pass1: dict[str, Any],
    run_dir: Path,
    progress: dict[str, Any],
    progress_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, str]], dict[str, Any]]:
    current = deepcopy(initial_pass1)
    problem_id = problem["problem_id"]
    correction_audit: dict[str, Any] = {
        "rounds": [],
        "acceptance_required": True,
        "max_rounds": args.correction_rounds,
    }
    overlay_dir = run_dir / "overlays"
    accepted = False
    for round_index in range(1, args.correction_rounds + 1):
        overlay_paths, overlay_manifest = render_pass1_overlays(
            problem=problem,
            dataset_dir=dataset_dir,
            pass1=current,
            output_dir=overlay_dir,
            label=f"correction_round{round_index}_input",
        )
        prior_patch: dict[str, Any] | None = None
        issues: list[dict[str, str]] = []
        applied = False
        for attempt in range(1, args.correction_attempts + 1):
            patch_path = run_dir / f"{problem_id}.pass1.correct.r{round_index}.attempt{attempt}.patch.json"
            corrected_path = run_dir / f"{problem_id}.pass1.correct.r{round_index}.attempt{attempt}.corrected.json"
            log = run_dir / f"{problem_id}.pass1.correct.r{round_index}.attempt{attempt}.api.json"
            try:
                patch, result = api_client.request_json(
                    model=args.model,
                    reasoning_effort=args.reasoning,
                    system_prompt=system_prompt,
                    user_prompt=correction_prompt(problem, current, round_index, attempt, prior_patch, issues),
                    image_paths=image_paths + overlay_paths,
                    image_detail=args.image_detail,
                    max_output_tokens=args.correction_max_output_tokens,
                )
                merge_phase_usage(progress, "pass1_correct", result.usage)
                write_json_atomic(patch_path, patch)
                persist_progress(progress_path, progress, phase=f"pass1_correct_round_{round_index}_validate")
            except PhysGraphAPIResponseError as exc:
                merge_phase_usage(progress, "pass1_correct", exc.result.usage)
                write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
                issues = [{"level": "error", "code": "api_response_json", "path": "$", "message": str(exc)}]
                persist_progress(progress_path, progress)
                continue
            except (PhysGraphAPIError, OSError, ValueError) as exc:
                write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
                issues = [{"level": "error", "code": "api_request", "path": "$", "message": str(exc)}]
                continue
            issues = validate_correction_patch(patch, problem_id, current)
            audit_operations: list[dict[str, Any]] = []
            candidate: dict[str, Any] | None = None
            if not issues:
                candidate, audit_operations = apply_correction_patch(current, patch)
                issues = validate_pass1(candidate, problem)
                if patch["verdict"] == "correct" and candidate == current:
                    issues.append({
                        "level": "error", "code": "correction_no_effect", "path": "$.operations",
                        "message": "correct补丁应用后没有产生任何变化",
                    })
            if issues:
                write_attempt_log(
                    log, endpoint=api_client.endpoint_label, result=result,
                    error=ValueError("纠错补丁本地校验失败：" + "；".join(issue["message"] for issue in issues)),
                )
                prior_patch = patch
                continue
            assert candidate is not None
            write_json_atomic(corrected_path, candidate)
            write_attempt_log(
                log,
                endpoint=api_client.endpoint_label,
                result=result,
                extra={
                    "correction": {
                        "round": round_index,
                        "verdict": patch["verdict"],
                        "operation_count": len(patch["operations"]),
                        "corrected_output": corrected_path.name,
                    }
                },
            )
            correction_audit["rounds"].append({
                "round": round_index,
                "attempt": attempt,
                "verdict": patch["verdict"],
                "operations": audit_operations,
                "input_pass1_sha256": json_sha256(current),
                "output_pass1_sha256": json_sha256(candidate),
                "overlay_manifest": overlay_manifest,
            })
            current = candidate
            applied = True
            if patch["verdict"] == "accept":
                accepted = True
                correction_audit["accepted_round"] = round_index
            break
        if not applied:
            return None, issues, correction_audit
        if accepted:
            break

    final_overlays, final_manifest = render_pass1_overlays(
        problem=problem,
        dataset_dir=dataset_dir,
        pass1=current,
        output_dir=overlay_dir,
        label="verified_final",
    )
    correction_audit["final_overlay_paths"] = [str(path) for path in final_overlays]
    correction_audit["final_overlay_manifest"] = final_manifest
    correction_audit["final_pass1_sha256"] = json_sha256(current)
    correction_audit["status"] = "accepted" if accepted else "unverified_after_max_rounds"
    pass1_artifact = (
        run_dir / f"{problem_id}.pass1.verified.json"
        if accepted
        else run_dir / f"{problem_id}.pass1.unverified_after_correction.json"
    )
    write_json_atomic(pass1_artifact, current)
    write_json_atomic(run_dir / f"{problem_id}.pass1.correction_audit.json", correction_audit)
    if not accepted:
        return None, [{
            "stage": "pass1",
            "level": "error",
            "code": "correction_unverified_after_max_rounds",
            "path": "$",
            "message": (
                f"连续{args.correction_rounds}轮均产生修改，最后一次修改尚未经过新叠加图accept确认；"
                "为避免未闭环结果进入下游，本题未保存"
            ),
        }], correction_audit
    return current, [], correction_audit


def generate_downstream(
    *,
    api_client: PhysGraphAPIClient,
    args: argparse.Namespace,
    system_prompt: str,
    problem: dict[str, Any],
    pass1: dict[str, Any],
    image_paths: list[Path],
    run_dir: Path,
    progress: dict[str, Any],
    progress_path: Path,
    review_notes: list[dict[str, str]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    prior: dict[str, Any] | None = None
    issues: list[dict[str, str]] = []
    problem_id = problem["problem_id"]
    for attempt in range(1, args.downstream_attempts + 1):
        output = run_dir / f"{problem_id}.downstream.attempt{attempt}.json"
        log = run_dir / f"{problem_id}.downstream.attempt{attempt}.api.json"
        postprocessing: dict[str, Any] = {}
        try:
            payload, result = api_client.request_json(
                model=args.model,
                reasoning_effort=args.reasoning,
                system_prompt=system_prompt,
                user_prompt=downstream_prompt(problem, pass1, attempt, prior, issues, review_notes),
                image_paths=image_paths,
                image_detail=args.image_detail,
                max_output_tokens=args.downstream_max_output_tokens,
            )
            merge_phase_usage(progress, "downstream", result.usage)
            write_json_atomic(output, payload)
            payload, normalization_rules = normalize_downstream_payload(
                payload, pass1=pass1, problem=problem,
            )
            if normalization_rules:
                normalized = run_dir / f"{problem_id}.downstream.attempt{attempt}.normalized.json"
                write_json_atomic(normalized, payload)
                postprocessing = {"postprocessing": {
                    "rules": normalization_rules,
                    "normalized_output": normalized.name,
                }}
            persist_progress(progress_path, progress, phase="downstream_validate")
        except PhysGraphAPIResponseError as exc:
            merge_phase_usage(progress, "downstream", exc.result.usage)
            write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
            issues = [{"level": "error", "code": "api_response_json", "path": "$", "message": str(exc)}]
            persist_progress(progress_path, progress)
            continue
        except (PhysGraphAPIError, OSError, ValueError) as exc:
            write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
            issues = [{"level": "error", "code": "api_request", "path": "$", "message": str(exc)}]
            continue
        issues = validate_downstream(payload, pass1, problem)
        if prior is not None:
            issues.extend(validate_no_empty_structural_regression(payload, prior))
        if not issues:
            write_attempt_log(log, endpoint=api_client.endpoint_label, result=result, extra=postprocessing)
            return payload, []
        write_attempt_log(
            log, endpoint=api_client.endpoint_label, result=result,
            error=ValueError("Pass 2–4本地校验失败：" + "；".join(issue["message"] for issue in issues)),
            extra=postprocessing,
        )
        prior = payload
    return None, issues


def _extract_pass4_document(payload: Any) -> tuple[Any, str]:
    if isinstance(payload, dict) and isinstance(payload.get("pass4"), dict):
        return deepcopy(payload["pass4"]), "pass4_key_extracted"
    return payload, "direct_pass4_object"


def generate_pass4_from_prefix(
    *,
    api_client: PhysGraphAPIClient,
    args: argparse.Namespace,
    system_prompt: str,
    problem: dict[str, Any],
    pass1: dict[str, Any],
    prefix: dict[str, Any],
    prefix_source_path: Path,
    run_dir: Path,
    progress: dict[str, Any],
    progress_path: Path,
    review_notes: list[dict[str, str]] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Regenerate only Pass 4 while keeping a validated Pass 2/3 prefix immutable."""

    prior_pass4: dict[str, Any] | None = None
    prior_wrapper: dict[str, Any] | None = None
    issues: list[dict[str, str]] = []
    problem_id = problem["problem_id"]
    for attempt in range(1, args.downstream_attempts + 1):
        raw_output = run_dir / f"{problem_id}.pass4_recovery.attempt{attempt}.json"
        candidate_output = run_dir / f"{problem_id}.pass4_recovery.attempt{attempt}.candidate.json"
        assembled_output = run_dir / f"{problem_id}.downstream.pass4_recovery.attempt{attempt}.assembled.json"
        log = run_dir / f"{problem_id}.pass4_recovery.attempt{attempt}.api.json"
        try:
            raw, result = api_client.request_json(
                model=args.model,
                reasoning_effort=args.reasoning,
                system_prompt=system_prompt,
                user_prompt=pass4_recovery_prompt(
                    problem=problem,
                    pass1=pass1,
                    prefix=prefix,
                    attempt=attempt,
                    prior_pass4=prior_pass4,
                    issues=issues,
                    review_notes=review_notes,
                ),
                image_paths=[],
                image_detail=args.image_detail,
                max_output_tokens=args.downstream_max_output_tokens,
            )
            merge_phase_usage(progress, "pass4_recovery", result.usage)
            write_json_atomic(raw_output, raw)
            pass4, extraction_rule = _extract_pass4_document(raw)
            candidate = {
                "pass2": deepcopy(prefix["pass2"]),
                "pass3": deepcopy(prefix["pass3"]),
                "pass4": pass4,
            }
            candidate, normalization_rules = normalize_downstream_payload(
                candidate, pass1=pass1, problem=problem,
            )
            write_json_atomic(candidate_output, candidate)
            persist_progress(progress_path, progress, phase="pass4_recovery_validate")
        except PhysGraphAPIResponseError as exc:
            merge_phase_usage(progress, "pass4_recovery", exc.result.usage)
            write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
            issues = [{"level": "error", "code": "api_response_json", "path": "$", "message": str(exc)}]
            persist_progress(progress_path, progress)
            continue
        except (PhysGraphAPIError, OSError, ValueError, KeyError) as exc:
            write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
            issues = [{"level": "error", "code": "api_request", "path": "$", "message": str(exc)}]
            persist_progress(progress_path, progress)
            continue

        issues = validate_downstream(candidate, pass1, problem)
        if prior_wrapper is not None:
            issues.extend(validate_no_empty_structural_regression(candidate, prior_wrapper))
        extra = {
            "pass4_recovery": {
                "prefix_source_path": str(prefix_source_path),
                "prefix_sha256": json_sha256(prefix),
                "extraction_rule": extraction_rule,
                "normalization_rules": normalization_rules,
            }
        }
        if not issues:
            write_json_atomic(assembled_output, candidate)
            write_attempt_log(log, endpoint=api_client.endpoint_label, result=result, extra=extra)
            return candidate, []
        write_attempt_log(
            log,
            endpoint=api_client.endpoint_label,
            result=result,
            error=ValueError("Pass 4定向恢复本地校验失败：" + "；".join(issue["message"] for issue in issues)),
            extra=extra,
        )
        prior_wrapper = candidate
        prior_pass4 = candidate.get("pass4") if isinstance(candidate.get("pass4"), dict) else None
    return None, issues


def _downstream_candidate_paths(workspace: Path, problem_id: str) -> list[Path]:
    candidates: set[Path] = set()
    patterns = (
        f"*/{problem_id}.downstream.attempt*.json",
        f"*/{problem_id}.downstream.pass4_recovery.*.assembled.json",
    )
    for parent_name in ("batch_runs", "reannotation_runs", "recovery_runs"):
        parent = workspace / parent_name
        if not parent.is_dir():
            continue
        for pattern in patterns:
            candidates.update(parent.glob(pattern))
    filtered = [
        path for path in candidates
        if ".normalized." not in path.name and ".api." not in path.name
    ]
    return sorted(filtered, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)


def _artifact_pass1_is_trusted(path: Path, problem_id: str, expected_hash: str) -> tuple[bool, str]:
    source_pass1_path = path.parent / f"{problem_id}.pass1.verified.json"
    audit_paths = (
        path.parent / f"{problem_id}.pass1.correction_audit.json",
        path.parent / f"{problem_id}.pass1.source_audit.json",
    )
    try:
        source_pass1 = read_json(source_pass1_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return False, f"pass1_read_error:{exc}"
    if json_sha256(source_pass1) != expected_hash:
        return False, "verified_pass1_hash_mismatch"
    for audit_path in audit_paths:
        if not audit_path.is_file():
            continue
        try:
            audit = read_json(audit_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if audit.get("status") == "accepted" and audit.get("final_pass1_sha256") == expected_hash:
            return True, audit_path.name
    return False, "accepted_pass1_audit_missing_or_mismatched"


def write_pass1_run_snapshot(run_dir: Path, problem_id: str, pass1: dict[str, Any]) -> None:
    """Persist the trusted Pass 1 boundary beside any new API downstream artifact."""

    snapshot = run_dir / f"{problem_id}.pass1.verified.json"
    expected_hash = json_sha256(pass1)
    if snapshot.is_file() and json_sha256(read_json(snapshot)) != expected_hash:
        raise ValueError("当前run_dir内Pass 1快照与已验证缓存不一致")
    if not snapshot.is_file():
        write_json_atomic(snapshot, pass1)
    correction_audit = run_dir / f"{problem_id}.pass1.correction_audit.json"
    if correction_audit.is_file():
        try:
            existing = read_json(correction_audit)
        except (OSError, json.JSONDecodeError, ValueError):
            existing = {}
        if existing.get("status") == "accepted" and existing.get("final_pass1_sha256") == expected_hash:
            return
    write_json_atomic(run_dir / f"{problem_id}.pass1.source_audit.json", {
        "status": "accepted",
        "source": "validated_runtime_pass1",
        "final_pass1_sha256": expected_hash,
        "recorded_at_utc": utc_now(),
    })


def validate_downstream_prefix(
    prefix: Any,
    *,
    pass1: dict[str, Any],
    problem: dict[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(prefix, dict) or set(prefix) != {"pass2", "pass3"}:
        return [{
            "level": "error", "code": "downstream_prefix_shape", "path": "$",
            "message": "可恢复前缀必须恰好包含pass2和pass3",
        }]
    previous: dict[str, dict[str, Any]] = {"pass1": pass1}
    issues: list[dict[str, str]] = []
    for stage in ("pass2", "pass3"):
        issues.extend(validation_errors(stage, prefix[stage], problem, previous))
        if isinstance(prefix[stage], dict):
            previous[stage] = prefix[stage]
    return issues


def discover_valid_downstream_artifact(
    *,
    workspace: Path,
    problem: dict[str, Any],
    pass1: dict[str, Any],
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any]]:
    """Recover an API response only when its Pass 1 and full validation agree."""

    problem_id = problem["problem_id"]
    expected_pass1_hash = json_sha256(pass1)
    rejected: list[dict[str, str]] = []
    for path in _downstream_candidate_paths(workspace, problem_id):
        trusted, trust_reason = _artifact_pass1_is_trusted(path, problem_id, expected_pass1_hash)
        if not trusted:
            rejected.append({"path": str(path), "reason": trust_reason})
            continue
        try:
            raw = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"read_error:{exc}"})
            continue
        normalized, normalization_rules = normalize_downstream_payload(
            raw, pass1=pass1, problem=problem,
        )
        issues = validate_downstream(normalized, pass1, problem)
        if issues:
            rejected.append({
                "path": str(path),
                "reason": "validation:" + ",".join(sorted({str(issue.get("code", "invalid")) for issue in issues})),
            })
            continue
        return normalized, path, {
            "source_path": str(path),
            "source_pass1_sha256": expected_pass1_hash,
            "pass1_trust_audit": trust_reason,
            "normalization_rules": normalization_rules,
            "validation_errors": 0,
        }
    return None, None, {"rejected_candidates": rejected[:10]}


def discover_valid_downstream_prefix(
    *,
    workspace: Path,
    problem: dict[str, Any],
    pass1: dict[str, Any],
) -> tuple[dict[str, Any] | None, Path | None, dict[str, Any]]:
    """Recover a fully valid API Pass 2/3 prefix when only Pass 4 is unusable."""

    problem_id = problem["problem_id"]
    expected_pass1_hash = json_sha256(pass1)
    rejected: list[dict[str, str]] = []
    for path in _downstream_candidate_paths(workspace, problem_id):
        trusted, trust_reason = _artifact_pass1_is_trusted(path, problem_id, expected_pass1_hash)
        if not trusted:
            rejected.append({"path": str(path), "reason": trust_reason})
            continue
        try:
            raw = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"read_error:{exc}"})
            continue
        normalized, normalization_rules = normalize_downstream_payload(
            raw, pass1=pass1, problem=problem,
        )
        if not isinstance(normalized, dict) or not all(stage in normalized for stage in ("pass2", "pass3")):
            rejected.append({"path": str(path), "reason": "missing_pass2_or_pass3"})
            continue
        prefix = {"pass2": deepcopy(normalized["pass2"]), "pass3": deepcopy(normalized["pass3"])}
        issues = validate_downstream_prefix(prefix, pass1=pass1, problem=problem)
        if issues:
            rejected.append({
                "path": str(path),
                "reason": "prefix_validation:" + ",".join(sorted({str(issue.get("code", "invalid")) for issue in issues})),
            })
            continue
        pass4_issues = validate_downstream(normalized, pass1, problem)
        return prefix, path, {
            "source_path": str(path),
            "source_pass1_sha256": expected_pass1_hash,
            "pass1_trust_audit": trust_reason,
            "normalization_rules": normalization_rules,
            "prefix_validation_errors": 0,
            "discarded_pass4_issue_codes": sorted({str(issue.get("code", "invalid")) for issue in pass4_issues}),
        }
    return None, None, {"rejected_candidates": rejected[:10]}


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    config = read_json(workspace / "workspace_config.json")
    dataset_dir = resolve_workspace_path(workspace, config, "dataset_dir")
    state = read_json(workspace / "reviews" / "state.json")
    manifest = effective_manifest(
        read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl")),
        state,
    )
    high = [item for item in manifest if item.get("segmentation", {}).get("confidence") == "high"]
    rejected_notes: dict[str, list[dict[str, str]]] = {}
    for problem in high:
        stages = state.get("problems", {}).get(problem["problem_id"], {}).get("stages", {})
        notes = [
            {"stage": stage, "note": str(stages.get(stage, {}).get("note", ""))}
            for stage in STAGES if stages.get(stage, {}).get("status") == "rejected"
        ]
        if notes:
            rejected_notes[problem["problem_id"]] = notes
    if args.only_rejected:
        pending = [problem for problem in high if problem["problem_id"] in rejected_notes]
    elif args.only_segmentation_repairs:
        pending = [
            problem for problem in high
            if segmentation_repair_status(
                state.get("problems", {}).get(problem["problem_id"], {})
            ) == "downstream_reannotation_required"
        ]
    else:
        pending = [problem for problem in high if not complete_and_valid(workspace, problem, state)]
    resume_inventory: dict[str, Any] | None = None
    resume_path: Path | None = None
    resume_checkpoint: dict[str, Any] | None = None
    resume_preaccepted_panels: set[str] = set()
    resume_checkpoint_history: dict[str, set[str]] = {}
    resume_checkpoint_reason = "not_requested"
    recovery_source: Path | None = None
    if args.recover_progress is not None:
        recovery_source = args.recover_progress.resolve()
        recovery_document = read_json(recovery_source)
        recovery_ids: list[str] = []
        for failure in recovery_document.get("failures", []) if isinstance(recovery_document, dict) else []:
            problem_id = failure.get("problem_id") if isinstance(failure, dict) else None
            if isinstance(problem_id, str) and problem_id not in recovery_ids:
                recovery_ids.append(problem_id)
        by_id = {problem["problem_id"]: problem for problem in high}
        unknown = [problem_id for problem_id in recovery_ids if problem_id not in by_id]
        if unknown:
            raise ValueError("--recover-progress包含不属于高切分置信度清单的problem_id：" + ", ".join(unknown))
        pending = [
            by_id[problem_id] for problem_id in recovery_ids
            if not complete_and_valid(workspace, by_id[problem_id], state)
        ]
    elif args.resume_unverified is not None:
        resume_path = args.resume_unverified.resolve()
        resume_inventory = read_json(resume_path)
        resume_problem_id = resume_inventory.get("problem_id") if isinstance(resume_inventory, dict) else None
        resume_problem = next((problem for problem in high if problem["problem_id"] == resume_problem_id), None)
        if resume_problem is None:
            raise ValueError("--resume-unverified中的problem_id不属于高切分置信度清单")
        if complete_and_valid(workspace, resume_problem, state):
            raise ValueError("--resume-unverified目标已经存在完整有效的Pass1–4，拒绝覆盖")
        pending = [resume_problem]
        resume_checkpoint, resume_preaccepted_panels, resume_checkpoint_reason = load_pixel_correction_checkpoint(
            resume_path=resume_path,
            problem=resume_problem,
            dataset_dir=dataset_dir,
        )
        if resume_checkpoint is not None:
            resume_checkpoint_history = load_checkpoint_panel_hashes(resume_path, resume_problem["problem_id"])
    if args.limit > 0:
        pending = pending[: args.limit]

    run_kind = (
        "recovery" if recovery_source is not None
        else "reannotation" if args.only_rejected
        else "segmentation_repair" if args.only_segmentation_repairs
        else "batch"
    )
    run_lock: ExclusiveRunLock | None = None
    if not args.dry_run:
        run_lock = ExclusiveRunLock(workspace / ".physgraph_api_annotation.lock")
        try:
            run_lock.acquire()
        except PhysGraphAPIError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        atexit.register(run_lock.release)

    run_parent = "dry_runs" if args.dry_run else f"{run_kind}_runs"
    run_dir = workspace / run_parent / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = workspace / (
        f"{run_kind}_dry_run.json" if args.dry_run else f"{run_kind}_progress.json"
    )
    recovery_preflight: list[dict[str, Any]] = []
    if args.dry_run:
        for problem in pending:
            problem_id = problem["problem_id"]
            repair_stale = segmentation_repair_status(
                state.get("problems", {}).get(problem_id, {})
            ) == "downstream_reannotation_required"
            cached = None if (args.ignore_pass1_cache or args.only_rejected) else load_pass1_cache(workspace, problem)
            if cached is not None:
                entry: dict[str, Any] = {"problem_id": problem_id, "pass1_source": "verified_cache"}
                if repair_stale:
                    entry["downstream_source"] = "blocked_stale_segmentation_artifacts"
                    entry["next_phase"] = "fresh_pass2_to_pass4"
                elif not args.no_auto_resume_checkpoints:
                    recovered, recovered_path, _ = discover_valid_downstream_artifact(
                        workspace=workspace, problem=problem, pass1=cached,
                    )
                    if recovered is not None:
                        entry["downstream_source"] = "validated_api_artifact"
                        entry["downstream_path"] = str(recovered_path)
                    else:
                        prefix, prefix_path, _ = discover_valid_downstream_prefix(
                            workspace=workspace, problem=problem, pass1=cached,
                        )
                        if prefix is not None:
                            entry["downstream_source"] = "validated_pass2_pass3_prefix"
                            entry["downstream_path"] = str(prefix_path)
                            entry["next_phase"] = "pass4_only_recovery"
                recovery_preflight.append(entry)
                continue
            if resume_checkpoint is not None and resume_checkpoint.get("problem_id") == problem_id:
                recovery_preflight.append({
                    "problem_id": problem_id,
                    "pass1_source": "explicit_pixel_checkpoint",
                    "path": str(resume_path),
                    "preaccepted_panel_ids": sorted(resume_preaccepted_panels),
                })
                continue
            if not args.no_auto_resume_checkpoints and not args.only_rejected and not args.ignore_pass1_cache:
                candidate, accepted, path, history, reason = discover_latest_pixel_checkpoint(
                    workspace=workspace, problem=problem, dataset_dir=dataset_dir,
                )
                if candidate is not None:
                    recovery_preflight.append({
                        "problem_id": problem_id,
                        "pass1_source": "auto_pixel_checkpoint",
                        "path": str(path),
                        "reason": reason,
                        "preaccepted_panel_ids": sorted(accepted),
                        "prior_state_hash_count": sum(len(values) for values in history.values()),
                    })
                    continue
            panel_reuse: list[dict[str, str]] = []
            if not args.no_auto_resume_checkpoints and not args.only_rejected and not args.ignore_pass1_cache:
                for panel in build_problem_panels(problem, dataset_dir):
                    recovered_panel, recovered_path, _ = discover_valid_panel_generation_artifact(
                        workspace=workspace,
                        problem=problem,
                        panel=panel,
                    )
                    if recovered_panel is not None and recovered_path is not None:
                        panel_reuse.append({"panel_id": panel.panel_id, "path": str(recovered_path)})
            recovery_preflight.append({
                "problem_id": problem_id,
                "pass1_source": "panel_generation_artifacts" if panel_reuse else "new_generation",
                "recovered_panels": panel_reuse,
                "next_phase": "merge_then_render_correction" if panel_reuse else "pass1_generation",
            })

    progress: dict[str, Any] = {
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "status": "dry_run" if args.dry_run else "running",
        "pipeline": "panel_crop_pixels_normalize_render_correct_then_pass2_4",
        "target_confidence": "high", "high_total": len(high),
        "valid_at_start": len(high) - sum(not complete_and_valid(workspace, problem, state) for problem in high),
        "queued_this_run": len(pending), "completed_this_run": 0, "failed_this_run": 0,
        "pass1_cache_hits": 0, "pass1_checkpoint_hits": 0, "downstream_artifact_hits": 0,
        "downstream_prefix_hits": 0, "pass1_panel_artifact_hits": 0,
        "pass1_patch_artifact_hits": 0,
        "current_problem_id": "", "current_phase": "",
        "failures": [], "checkpoint_reuse": [], "pass1_panel_reuse": [], "pass1_patch_reuse": [],
        "downstream_reuse": [], "downstream_prefix_reuse": [],
        "recovery_preflight": recovery_preflight,
        "run_dir": str(run_dir),
        "provider": {
            "endpoint": safe_endpoint_label(normalize_base_url(args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)),
            "api_mode": args.api_mode, "model": args.model,
            "reasoning_effort": args.reasoning, "image_detail": args.image_detail,
            "api_key_env": args.api_key_env,
        },
        "pipeline_settings": {
            "pass1_attempts": args.pass1_attempts,
            "correction_attempts": args.correction_attempts,
            "correction_rounds": args.correction_rounds,
            "resume_correction_rounds": args.resume_correction_rounds,
            "downstream_attempts": args.downstream_attempts,
            "recover_progress": str(recovery_source) if recovery_source else "",
            "auto_resume_checkpoints": not args.no_auto_resume_checkpoints,
            "only_segmentation_repairs": args.only_segmentation_repairs,
            "resume_unverified": str(args.resume_unverified.resolve()) if args.resume_unverified else "",
            "resume_mode": "pixel_correction_checkpoint" if resume_checkpoint is not None else "semantic_inventory_only",
            "resume_checkpoint_reason": resume_checkpoint_reason,
            "resume_preaccepted_panel_ids": sorted(resume_preaccepted_panels),
        },
        "api_usage": {},
        "api_usage_by_phase": {
            "pass1_generate": {}, "pass1_correct": {}, "downstream": {}, "pass4_recovery": {},
        },
    }
    write_json_atomic(progress_path, progress)
    if args.dry_run:
        print(json.dumps(progress, ensure_ascii=False, indent=2))
        return 0

    def mark_interrupted_at_exit() -> None:
        if progress.get("status") == "running":
            progress.update(
                status="interrupted",
                interrupted_at_utc=utc_now(),
            )
            persist_progress(progress_path, progress)

    atexit.register(mark_interrupted_at_exit)

    try:
        api_client = PhysGraphAPIClient(
            api_key_env=args.api_key_env, base_url=args.base_url, api_mode=args.api_mode,
            timeout_seconds=args.timeout_seconds, transport_retries=args.transport_retries,
        )
    except (PhysGraphAPIError, ValueError) as exc:
        progress.update(status="configuration_error", configuration_error=str(exc))
        persist_progress(progress_path, progress)
        print(str(exc), file=sys.stderr)
        assert run_lock is not None
        run_lock.release()
        atexit.unregister(run_lock.release)
        return 2

    pass1_system = load_pass1_system_instructions(ROOT, workspace)
    correction_system = load_pass1_correction_system_instructions(workspace)
    downstream_system = load_downstream_system_instructions(ROOT, workspace)
    for queue_index, problem in enumerate(pending, start=1):
        problem_id = problem["problem_id"]
        repair_stale = segmentation_repair_status(
            state.get("problems", {}).get(problem_id, {})
        ) == "downstream_reannotation_required"
        progress.update(current_problem_id=problem_id, queue_index=queue_index)
        persist_progress(progress_path, progress, phase="pass1_cache_check")
        image_paths = [dataset_dir / image["path"] for image in problem.get("images", [])]
        notes = rejected_notes.get(problem_id)
        issues: list[dict[str, str]] = []
        pass1 = None if (args.ignore_pass1_cache or args.only_rejected) else load_pass1_cache(workspace, problem)
        if pass1 is not None:
            progress["pass1_cache_hits"] += 1
            persist_progress(progress_path, progress, phase="pass1_cache_hit")
        else:
            checkpoint_candidate = None
            checkpoint_preaccepted: set[str] = set()
            checkpoint_path: Path | None = None
            checkpoint_history: dict[str, set[str]] = {}
            checkpoint_reason = "not_checked"
            if resume_checkpoint is not None and resume_checkpoint.get("problem_id") == problem_id:
                checkpoint_candidate = resume_checkpoint
                checkpoint_preaccepted = set(resume_preaccepted_panels)
                checkpoint_path = resume_path
                checkpoint_history = resume_checkpoint_history
                checkpoint_reason = resume_checkpoint_reason
            elif not args.no_auto_resume_checkpoints and not args.only_rejected and not args.ignore_pass1_cache:
                (
                    checkpoint_candidate,
                    checkpoint_preaccepted,
                    checkpoint_path,
                    checkpoint_history,
                    checkpoint_reason,
                ) = discover_latest_pixel_checkpoint(
                    workspace=workspace,
                    problem=problem,
                    dataset_dir=dataset_dir,
                )
            use_checkpoint = checkpoint_candidate is not None and checkpoint_path is not None
            if use_checkpoint:
                progress["pass1_checkpoint_hits"] += 1
                progress["checkpoint_reuse"].append({
                    "problem_id": problem_id,
                    "path": str(checkpoint_path),
                    "reason": checkpoint_reason,
                    "preaccepted_panel_ids": sorted(checkpoint_preaccepted),
                    "max_new_rounds_per_unaccepted_panel": args.resume_correction_rounds,
                })
                persist_progress(progress_path, progress, phase="pass1_resume_validated_pixel_checkpoint")
                panels, panel_assets = prepare_panel_assets(
                    problem=problem,
                    dataset_dir=dataset_dir,
                    run_dir=run_dir,
                )
                generated = deepcopy(checkpoint_candidate)
                issues = validate_pass1(generated, problem)
                write_json_atomic(run_dir / f"{problem_id}.pass1.pixel_checkpoint_reused.json", generated)
            else:
                persist_progress(progress_path, progress, phase="pass1_generate")
                generated, issues, panels, panel_assets = generate_pass1_by_panels(
                    api_client=api_client, args=args, system_prompt=pass1_system, problem=problem,
                    dataset_dir=dataset_dir, run_dir=run_dir, workspace=workspace, progress=progress,
                    progress_path=progress_path, review_notes=notes,
                    resume_inventory=resume_inventory if resume_inventory and resume_inventory.get("problem_id") == problem_id else None,
                )
            if generated is not None:
                persist_progress(progress_path, progress, phase="pass1_render_and_correct")
                pass1, issues, correction_audit = correct_pass1_by_panels(
                    api_client=api_client, args=args, system_prompt=correction_system,
                    problem=problem, dataset_dir=dataset_dir, initial_pass1=generated,
                    panels=panels, assets=panel_assets, run_dir=run_dir, progress=progress,
                    progress_path=progress_path,
                    preaccepted_panel_ids=checkpoint_preaccepted if use_checkpoint else None,
                    resumed_from=checkpoint_path if use_checkpoint else None,
                    max_rounds_per_panel=args.resume_correction_rounds if use_checkpoint else args.correction_rounds,
                    prior_panel_hashes=checkpoint_history if use_checkpoint else None,
                    workspace=workspace,
                )
                if pass1 is not None:
                    save_pass1_cache(
                        workspace, problem, pass1,
                        {
                            "model": args.model, "reasoning_effort": args.reasoning,
                            "image_detail": args.image_detail, "run_dir": str(run_dir),
                            "correction_audit_sha256": json_sha256(correction_audit),
                        },
                    )

        downstream = None
        if pass1 is not None:
            write_pass1_run_snapshot(run_dir, problem_id, pass1)
            recovered_downstream = None
            recovered_downstream_path: Path | None = None
            downstream_recovery_audit: dict[str, Any] = {}
            if not repair_stale and not args.no_auto_resume_checkpoints and not args.only_rejected and not args.ignore_pass1_cache:
                recovered_downstream, recovered_downstream_path, downstream_recovery_audit = discover_valid_downstream_artifact(
                    workspace=workspace,
                    problem=problem,
                    pass1=pass1,
                )
            if recovered_downstream is not None and recovered_downstream_path is not None:
                downstream = recovered_downstream
                progress["downstream_artifact_hits"] += 1
                progress["downstream_reuse"].append({
                    "problem_id": problem_id,
                    "path": str(recovered_downstream_path),
                })
                write_json_atomic(run_dir / f"{problem_id}.downstream.recovered.normalized.json", downstream)
                write_json_atomic(run_dir / f"{problem_id}.downstream.recovery_audit.json", downstream_recovery_audit)
                persist_progress(progress_path, progress, phase="downstream_validated_artifact_reused")
                issues = []
            else:
                recovered_prefix = None
                recovered_prefix_path: Path | None = None
                prefix_recovery_audit: dict[str, Any] = {}
                if not repair_stale and not args.no_auto_resume_checkpoints and not args.only_rejected and not args.ignore_pass1_cache:
                    recovered_prefix, recovered_prefix_path, prefix_recovery_audit = discover_valid_downstream_prefix(
                        workspace=workspace,
                        problem=problem,
                        pass1=pass1,
                    )
                if recovered_prefix is not None and recovered_prefix_path is not None:
                    progress["downstream_prefix_hits"] += 1
                    progress["downstream_prefix_reuse"].append({
                        "problem_id": problem_id,
                        "path": str(recovered_prefix_path),
                    })
                    write_json_atomic(run_dir / f"{problem_id}.downstream.prefix.recovered.json", recovered_prefix)
                    write_json_atomic(run_dir / f"{problem_id}.downstream.prefix.recovery_audit.json", prefix_recovery_audit)
                    persist_progress(progress_path, progress, phase="pass4_recovery_generate")
                    downstream, issues = generate_pass4_from_prefix(
                        api_client=api_client,
                        args=args,
                        system_prompt=downstream_system,
                        problem=problem,
                        pass1=pass1,
                        prefix=recovered_prefix,
                        prefix_source_path=recovered_prefix_path,
                        run_dir=run_dir,
                        progress=progress,
                        progress_path=progress_path,
                        review_notes=notes,
                    )
                else:
                    persist_progress(progress_path, progress, phase="downstream_generate")
                    downstream, issues = generate_downstream(
                        api_client=api_client, args=args, system_prompt=downstream_system,
                        problem=problem, pass1=pass1, image_paths=image_paths,
                        run_dir=run_dir, progress=progress, progress_path=progress_path,
                        review_notes=notes,
                    )
        succeeded = pass1 is not None and downstream is not None
        if succeeded:
            payload = {"pass1": pass1, **downstream}
            save_payload(
                workspace,
                payload,
                problem_id,
                reset_review=args.only_rejected,
                problem=problem,
                output_source="api_segmentation_repair" if repair_stale else "api_generation",
            )
            progress["completed_this_run"] += 1
        else:
            progress["failed_this_run"] += 1
            progress["failures"].append({
                "problem_id": problem_id,
                "phase": progress.get("current_phase", "unknown"),
                "issues": issues,
            })
        progress.update(current_problem_id="", current_phase="")
        persist_progress(progress_path, progress)
        print(f"[{queue_index}/{len(pending)}] {problem_id}: {'saved' if succeeded else 'failed'}", flush=True)

    progress.update(status="complete", finished_at_utc=utc_now(), current_problem_id="", current_phase="")
    persist_progress(progress_path, progress)
    assert run_lock is not None
    run_lock.release()
    atexit.unregister(run_lock.release)
    return 0 if progress["failed_this_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
