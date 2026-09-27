#!/usr/bin/env python
"""Pixel-space panel extraction and deterministic PhysGraph normalization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


VISUAL_TYPES = {
    "text_glyph", "point_mark", "line_segment", "polyline", "curve",
    "circle", "arc", "arrow", "angle_mark", "region", "axis",
    "graph_curve", "object_shape", "component_symbol", "field_symbol",
    "unknown_visual",
}
PIXEL_NODE_KEYS = {
    "id", "type", "subtype", "text", "bbox_px", "keypoints_px",
    "center_px", "radius_px", "confidence",
}
STANDARD_NODE_KEYS = {
    "id", "type", "subtype", "text", "image_id", "bbox_1000",
    "keypoints_1000", "center_1000", "radius_1000", "confidence",
}
AMBIGUITY_KEYS = {
    "id", "scope", "description", "candidate_ids",
    "evidence_visual_ids", "evidence_mention_ids",
}
GEOMETRY_KEYPOINT_TYPES = {
    "line_segment", "polyline", "curve", "arc", "arrow", "graph_curve",
}


@dataclass(frozen=True)
class PanelSpec:
    image_id: str
    panel_id: str
    source_path: str
    source_width_px: int
    source_height_px: int
    x0_px: int
    y0_px: int
    x1_px: int
    y1_px: int

    @property
    def width_px(self) -> int:
        return self.x1_px - self.x0_px

    @property
    def height_px(self) -> int:
        return self.y1_px - self.y0_px

    def as_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["width_px"] = self.width_px
        value["height_px"] = self.height_px
        return value


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _blank_row_runs(low_ink: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, blank in enumerate(low_ink.tolist()):
        if blank and start is None:
            start = index
        elif not blank and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(low_ink)))
    return runs


def detect_horizontal_panels(
    *,
    image_path: Path,
    image_id: str,
    source_path: str,
    max_panels: int = 4,
) -> list[PanelSpec]:
    """Split a tall compound diagram only at substantial ink-free row bands.

    Panels never overlap.  A split is accepted only when both neighbouring
    panels remain large enough, which keeps ordinary single diagrams intact.
    """

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("L")
    width, height = image.size
    if height < 240 or max_panels <= 1:
        cuts: list[int] = []
    else:
        pixels = np.asarray(image)
        dark_counts = (pixels < 180).sum(axis=1)
        # Require a truly ink-free band.  Allowing even two dark pixels can
        # split the interior of a sparse diagram containing only vertical lines.
        blank_threshold = 0
        minimum_gap = max(8, round(height * 0.018))
        minimum_panel = max(100, round(height * 0.15))
        candidates = [
            (start + end) // 2
            for start, end in _blank_row_runs(dark_counts <= blank_threshold)
            if end - start >= minimum_gap and start > 0 and end < height
        ]
        cuts = []
        previous = 0
        for candidate in candidates:
            if len(cuts) >= max_panels - 1:
                break
            if candidate - previous < minimum_panel or height - candidate < minimum_panel:
                continue
            cuts.append(candidate)
            previous = candidate

    boundaries = [0, *cuts, height]
    return [
        PanelSpec(
            image_id=image_id,
            panel_id=f"{image_id}_p{index:02d}",
            source_path=source_path,
            source_width_px=width,
            source_height_px=height,
            x0_px=0,
            y0_px=boundaries[index - 1],
            x1_px=width,
            y1_px=boundaries[index],
        )
        for index in range(1, len(boundaries))
    ]


def build_problem_panels(problem: dict[str, Any], dataset_dir: Path) -> list[PanelSpec]:
    panels: list[PanelSpec] = []
    for image in problem.get("images", []):
        image_path = dataset_dir / image["path"]
        detected = detect_horizontal_panels(
            image_path=image_path,
            image_id=image["image_id"],
            source_path=image["path"],
        )
        expected_width, expected_height = image.get("width"), image.get("height")
        for panel in detected:
            if expected_width and int(expected_width) != panel.source_width_px:
                raise ValueError(
                    f"{panel.image_id} manifest width={expected_width}, actual={panel.source_width_px}"
                )
            if expected_height and int(expected_height) != panel.source_height_px:
                raise ValueError(
                    f"{panel.image_id} manifest height={expected_height}, actual={panel.source_height_px}"
                )
        panels.extend(detected)
    return panels


def _nice_grid_step(length: int) -> int:
    target = max(1, length // 5)
    for step in (25, 50, 100, 200, 250, 500, 1000):
        if step >= target:
            return step
    return 1000


def render_panel_assets(
    *,
    dataset_dir: Path,
    panel: PanelSpec,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write a lossless raw crop and a same-size, crop-pixel calibration copy."""

    with Image.open(dataset_dir / panel.source_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
    crop = source.crop((panel.x0_px, panel.y0_px, panel.x1_px, panel.y1_px))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{panel.panel_id}.raw.png"
    grid_path = output_dir / f"{panel.panel_id}.pixel_grid.png"
    crop.save(raw_path, format="PNG", optimize=True)

    grid = crop.convert("RGBA")
    layer = Image.new("RGBA", grid.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    small = _font(max(10, min(18, round(min(grid.size) / 28))), bold=True)
    font_box = draw.textbbox((0, 0), "Mg", font=small)
    font_height = font_box[3] - font_box[1]
    x_step, y_step = _nice_grid_step(panel.width_px), _nice_grid_step(panel.height_px)
    for x in sorted(set([0, panel.width_px - 1, *range(0, panel.width_px, x_step)])):
        draw.line((x, 0, x, panel.height_px), fill=(0, 115, 230, 90), width=1)
        draw.rectangle((max(0, x - 2), 0, min(panel.width_px, x + 46), font_height + 6), fill=(255, 255, 255, 205))
        draw.text((max(1, x + 2), 1), f"x={x}", fill=(0, 72, 160, 255), font=small)
    for y in sorted(set([0, panel.height_px - 1, *range(0, panel.height_px, y_step)])):
        draw.line((0, y, panel.width_px, y), fill=(214, 48, 49, 90), width=1)
        label_y = max(0, min(panel.height_px - font_height - 5, y + 2))
        draw.rectangle((0, label_y, 48, label_y + font_height + 4), fill=(255, 255, 255, 205))
        draw.text((2, label_y), f"y={y}", fill=(170, 30, 35, 255), font=small)
    calibrated = Image.alpha_composite(grid, layer).convert("RGB")
    calibrated.save(grid_path, format="PNG", optimize=True)
    return raw_path, grid_path


def render_panel_pixel_overlay(
    *,
    raw_crop_path: Path,
    pixel_nodes: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Draw candidate pixel nodes without changing the crop coordinate frame."""

    with Image.open(raw_crop_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font(max(10, min(18, round(min(image.size) / 28))), bold=True)
    palette = ("#1769e0", "#d6473d", "#07885f", "#8950c8", "#c47610")
    for index, node in enumerate(pixel_nodes):
        color = palette[index % len(palette)]
        bbox = node.get("bbox_px")
        if isinstance(bbox, list) and len(bbox) == 4:
            draw.rectangle(tuple(bbox), outline=color, width=max(2, round(min(image.size) / 220)))
            label = str(node.get("id", "?"))
            label_box = draw.textbbox((0, 0), label, font=font)
            label_width = label_box[2] - label_box[0] + 6
            label_height = label_box[3] - label_box[1] + 4
            label_y = max(0, bbox[1] - label_height)
            draw.rectangle((bbox[0], label_y, bbox[0] + label_width, label_y + label_height), fill=color)
            draw.text((bbox[0] + 3, label_y + 1), label, fill="white", font=font)
        for point in node.get("keypoints_px", []):
            if isinstance(point, list) and len(point) == 2:
                radius = max(2, round(min(image.size) / 180))
                draw.ellipse(
                    (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                    fill=color,
                    outline="white",
                )
        center = node.get("center_px")
        if node.get("type") in {"circle", "arc"} and isinstance(center, list) and center != [-1, -1]:
            arm = max(3, round(min(image.size) / 140))
            draw.line((center[0] - arm, center[1], center[0] + arm, center[1]), fill=color, width=2)
            draw.line((center[0], center[1] - arm, center[0], center[1] + arm), fill=color, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def panel_contract(panel: PanelSpec, problem_id: str) -> dict[str, Any]:
    return {
        "problem_id": problem_id,
        "image_id": panel.image_id,
        "panel_id": panel.panel_id,
        "coordinate_space": "crop_pixels",
        "source_width_px": panel.source_width_px,
        "source_height_px": panel.source_height_px,
        "crop_origin_px": [panel.x0_px, panel.y0_px],
        "crop_size_px": [panel.width_px, panel.height_px],
    }


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"stage": "pass1_pixel", "level": "error", "code": code, "path": path, "message": message}


def _integer_point(point: Any, width: int, height: int) -> bool:
    return (
        isinstance(point, list) and len(point) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in point)
        and 0 <= point[0] <= width and 0 <= point[1] <= height
    )


def normalize_panel_document(
    document: Any,
    *,
    panel: PanelSpec,
) -> tuple[Any, dict[str, Any]]:
    """Apply narrow, geometry-proven pixel repairs before strict validation.

    The normalizer fixes schema sentinels, bbox axis order only when keypoints
    prove the intended rectangle, and bbox/keypoint containment. It deliberately
    refuses document-wide likely ``[x,y,width,height]`` output so that an invalid
    coordinate convention is retried instead of being silently reinterpreted.
    """

    normalized = deepcopy(document)
    audit: dict[str, Any] = {
        "rule": "conservative_panel_pixel_normalization_v1",
        "changed": False,
        "likely_xywh_document": False,
        "repairs": [],
        "dropped_node_ids": [],
    }
    if not isinstance(normalized, dict) or not isinstance(normalized.get("visual_nodes"), list):
        audit["status"] = "not_a_panel_document"
        return normalized, audit

    nodes = normalized["visual_nodes"]
    shaped_bboxes = [
        node.get("bbox_px")
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("bbox_px"), list)
        and len(node["bbox_px"]) == 4
        and all(isinstance(value, int) and not isinstance(value, bool) for value in node["bbox_px"])
    ]
    descending_count = sum(
        bbox[0] > bbox[2] or bbox[1] > bbox[3]
        for bbox in shaped_bboxes
    )
    likely_xywh = (
        len(shaped_bboxes) >= 3
        and descending_count / len(shaped_bboxes) >= 0.60
    )
    audit["likely_xywh_document"] = likely_xywh
    audit["bbox_count"] = len(shaped_bboxes)
    audit["descending_bbox_count"] = descending_count

    width, height = panel.width_px, panel.height_px
    kept_nodes: list[Any] = []
    dropped_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            kept_nodes.append(node)
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        path = f"$.visual_nodes[{index}]"

        if node_type not in {"circle", "arc"} and (
            node.get("center_px") != [-1, -1] or node.get("radius_px") != -1
        ):
            before = {"center_px": deepcopy(node.get("center_px")), "radius_px": node.get("radius_px")}
            node["center_px"] = [-1, -1]
            node["radius_px"] = -1
            audit["repairs"].append({
                "path": path,
                "rule": "non_circle_unknown_center_radius_sentinel",
                "before": before,
                "after": {"center_px": [-1, -1], "radius_px": -1},
            })

        bbox = node.get("bbox_px")
        points = node.get("keypoints_px")
        valid_bbox = (
            isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
        )
        valid_points = (
            [point for point in points if _integer_point(point, width, height)]
            if isinstance(points, list) else []
        )
        distinct_points = {tuple(point) for point in valid_points}

        if (
            valid_bbox and not likely_xywh
            and (bbox[0] > bbox[2] or bbox[1] > bbox[3])
            and node_type in GEOMETRY_KEYPOINT_TYPES
            and len(distinct_points) >= 2
        ):
            ordered = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
            if (
                0 <= ordered[0] <= ordered[2] <= width
                and 0 <= ordered[1] <= ordered[3] <= height
                and all(
                    ordered[0] - 2 <= point[0] <= ordered[2] + 2
                    and ordered[1] - 2 <= point[1] <= ordered[3] + 2
                    for point in valid_points
                )
            ):
                before = list(bbox)
                node["bbox_px"] = ordered
                bbox = node["bbox_px"]
                audit["repairs"].append({
                    "path": f"{path}.bbox_px", "rule": "keypoint_proven_axis_order",
                    "before": before, "after": ordered,
                })

        bbox = node.get("bbox_px")
        ascending_in_bounds = (
            isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
            and 0 <= bbox[0] <= bbox[2] <= width
            and 0 <= bbox[1] <= bbox[3] <= height
        )
        if ascending_in_bounds and valid_points:
            expanded = [
                min(bbox[0], *(point[0] for point in valid_points)),
                min(bbox[1], *(point[1] for point in valid_points)),
                max(bbox[2], *(point[0] for point in valid_points)),
                max(bbox[3], *(point[1] for point in valid_points)),
            ]
            if expanded != bbox:
                before = list(bbox)
                node["bbox_px"] = expanded
                bbox = expanded
                audit["repairs"].append({
                    "path": f"{path}.bbox_px", "rule": "include_valid_keypoints",
                    "before": before, "after": expanded,
                })

        if (
            not likely_xywh
            and node_type in GEOMETRY_KEYPOINT_TYPES
            and isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(value, int) and not isinstance(value, bool) for value in bbox)
            and (bbox[0] == bbox[2] or bbox[1] == bbox[3])
            and len(distinct_points) < 2
        ):
            if isinstance(node_id, str) and node_id:
                dropped_ids.add(node_id)
                audit["dropped_node_ids"].append(node_id)
            audit["repairs"].append({
                "path": path,
                "rule": "drop_degenerate_geometry_without_two_distinct_keypoints",
            })
            continue
        kept_nodes.append(node)

    if dropped_ids:
        normalized["visual_nodes"] = kept_nodes
        for ambiguity in normalized.get("ambiguities", []):
            if not isinstance(ambiguity, dict):
                continue
            for field in ("candidate_ids", "evidence_visual_ids"):
                values = ambiguity.get(field)
                if isinstance(values, list):
                    ambiguity[field] = [value for value in values if value not in dropped_ids]

    audit["changed"] = normalized != document
    audit["status"] = "normalized" if audit["changed"] else "unchanged"
    return normalized, audit


def normalize_pixel_correction_patch(
    patch: Any,
) -> tuple[Any, dict[str, Any]]:
    """Normalize only schema-defined non-circle sentinels inside patch nodes."""

    normalized = deepcopy(patch)
    audit: dict[str, Any] = {
        "rule": "pixel_patch_non_circle_sentinel_v1",
        "changed": False,
        "repairs": [],
    }
    if not isinstance(normalized, dict) or not isinstance(normalized.get("operations"), list):
        audit["status"] = "not_a_pixel_patch"
        return normalized, audit
    for index, operation in enumerate(normalized["operations"]):
        if not isinstance(operation, dict) or operation.get("action") not in {"replace", "append"}:
            continue
        node = operation.get("node")
        if not isinstance(node, dict) or node.get("type") in {"circle", "arc"}:
            continue
        if node.get("center_px") == [-1, -1] and node.get("radius_px") == -1:
            continue
        before = {"center_px": deepcopy(node.get("center_px")), "radius_px": node.get("radius_px")}
        node["center_px"] = [-1, -1]
        node["radius_px"] = -1
        audit["repairs"].append({
            "path": f"$.operations[{index}].node",
            "rule": "non_circle_unknown_center_radius_sentinel",
            "before": before,
            "after": {"center_px": [-1, -1], "radius_px": -1},
        })
    audit["changed"] = normalized != patch
    audit["status"] = "normalized" if audit["changed"] else "unchanged"
    return normalized, audit


def validate_panel_document(
    document: Any,
    *,
    panel: PanelSpec,
    problem_id: str,
) -> list[dict[str, str]]:
    expected_top = {
        *panel_contract(panel, problem_id).keys(), "visual_nodes", "ambiguities",
    }
    if not isinstance(document, dict) or set(document) != expected_top:
        return [_issue("panel_shape", "$", "像素Pass1顶层字段不正确")]
    issues: list[dict[str, str]] = []
    for key, expected in panel_contract(panel, problem_id).items():
        if document.get(key) != expected:
            issues.append(_issue("panel_metadata_mismatch", f"$.{key}", f"{key}必须等于{expected!r}"))
    nodes = document.get("visual_nodes")
    if not isinstance(nodes, list) or not nodes:
        issues.append(_issue("empty_panel_nodes", "$.visual_nodes", "含图像内容的面板至少需要一个视觉节点"))
        return issues
    known_ids: set[str] = set()
    width, height = panel.width_px, panel.height_px
    for index, node in enumerate(nodes):
        path = f"$.visual_nodes[{index}]"
        if not isinstance(node, dict) or set(node) != PIXEL_NODE_KEYS:
            issues.append(_issue("pixel_node_shape", path, "像素视觉节点字段不正确"))
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id or node_id in known_ids:
            issues.append(_issue("pixel_node_id", f"{path}.id", "节点ID必须为面板内唯一非空字符串"))
        else:
            known_ids.add(node_id)
        if node.get("type") not in VISUAL_TYPES:
            issues.append(_issue("pixel_node_type", f"{path}.type", "视觉类型不在允许枚举中"))
        if not isinstance(node.get("subtype"), str) or not isinstance(node.get("text"), str):
            issues.append(_issue("pixel_node_text", path, "subtype和text必须为字符串"))
        if node.get("confidence") not in {"high", "medium", "low"}:
            issues.append(_issue("pixel_node_confidence", f"{path}.confidence", "confidence不合法"))
        bbox = node.get("bbox_px")
        if not (
            isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(value, int) for value in bbox)
            and 0 <= bbox[0] <= bbox[2] <= width
            and 0 <= bbox[1] <= bbox[3] <= height
        ):
            issues.append(_issue("pixel_bbox", f"{path}.bbox_px", f"bbox_px必须位于0..{width}×0..{height}"))
            continue
        points = node.get("keypoints_px")
        if not isinstance(points, list):
            issues.append(_issue("pixel_keypoints", f"{path}.keypoints_px", "keypoints_px必须是数组"))
            points = []
        if node.get("type") in GEOMETRY_KEYPOINT_TYPES and len(points) < 2:
            issues.append(_issue("pixel_missing_keypoints", f"{path}.keypoints_px", "线、箭头或曲线至少需要两个关键点"))
        for point_index, point in enumerate(points):
            if not (
                isinstance(point, list) and len(point) == 2
                and all(isinstance(value, int) for value in point)
                and 0 <= point[0] <= width and 0 <= point[1] <= height
            ):
                issues.append(_issue("pixel_keypoint", f"{path}.keypoints_px[{point_index}]", "关键点超出面板像素边界"))
            elif not (bbox[0] - 2 <= point[0] <= bbox[2] + 2 and bbox[1] - 2 <= point[1] <= bbox[3] + 2):
                issues.append(_issue("pixel_keypoint_outside_bbox", f"{path}.keypoints_px[{point_index}]", "关键点必须位于bbox内"))
        center, radius = node.get("center_px"), node.get("radius_px")
        valid_center = (
            isinstance(center, list) and len(center) == 2
            and all(isinstance(value, int) for value in center)
            and (center == [-1, -1] or (0 <= center[0] <= width and 0 <= center[1] <= height))
        )
        if not valid_center or not isinstance(radius, int):
            issues.append(_issue("pixel_circle_geometry", path, "center_px/radius_px格式不正确"))
        elif node.get("type") in {"circle", "arc"}:
            if center == [-1, -1] or radius <= 0:
                issues.append(_issue("pixel_missing_circle", path, "circle/arc必须提供像素圆心和正半径"))
        elif center != [-1, -1] or radius != -1:
            issues.append(_issue("pixel_unexpected_circle", path, "非circle/arc必须写center_px=[-1,-1]、radius_px=-1"))
        if node.get("type") == "text_glyph" and not node.get("text", "").strip():
            issues.append(_issue("pixel_empty_ocr", f"{path}.text", "text_glyph必须填写可见文本"))

    ambiguities = document.get("ambiguities")
    if not isinstance(ambiguities, list):
        issues.append(_issue("pixel_ambiguities", "$.ambiguities", "ambiguities必须是数组"))
        return issues
    ambiguity_ids: set[str] = set()
    for index, ambiguity in enumerate(ambiguities):
        path = f"$.ambiguities[{index}]"
        if not isinstance(ambiguity, dict) or set(ambiguity) != AMBIGUITY_KEYS:
            issues.append(_issue("pixel_ambiguity_shape", path, "ambiguity字段不正确"))
            continue
        ambiguity_id = ambiguity.get("id")
        if not isinstance(ambiguity_id, str) or not ambiguity_id or ambiguity_id in ambiguity_ids:
            issues.append(_issue("pixel_ambiguity_id", f"{path}.id", "ambiguity ID必须唯一"))
        else:
            ambiguity_ids.add(ambiguity_id)
        if not isinstance(ambiguity.get("description"), str) or not ambiguity["description"].strip():
            issues.append(_issue("pixel_ambiguity_description", f"{path}.description", "description不能为空"))
        for field in ("candidate_ids", "evidence_visual_ids"):
            values = ambiguity.get(field)
            if not isinstance(values, list) or any(value not in known_ids for value in values):
                issues.append(_issue("pixel_ambiguity_reference", f"{path}.{field}", "视觉引用必须指向本面板节点"))
        if not isinstance(ambiguity.get("evidence_mention_ids"), list):
            issues.append(_issue("pixel_ambiguity_mentions", f"{path}.evidence_mention_ids", "evidence_mention_ids必须是数组"))
    return issues


def _to_normalized(value: int, size: int) -> int:
    return max(0, min(1000, round(value / size * 1000)))


def _to_source_pixel(value: int, size: int) -> int:
    return max(0, min(size, round(value / 1000 * size)))


def pixel_node_to_standard(node: dict[str, Any], panel: PanelSpec) -> dict[str, Any]:
    result = {
        "id": node["id"],
        "type": node["type"],
        "subtype": node["subtype"],
        "text": node["text"],
        "image_id": panel.image_id,
        "confidence": node["confidence"],
    }
    bbox = node["bbox_px"]
    result["bbox_1000"] = [
        _to_normalized(panel.x0_px + bbox[0], panel.source_width_px),
        _to_normalized(panel.y0_px + bbox[1], panel.source_height_px),
        _to_normalized(panel.x0_px + bbox[2], panel.source_width_px),
        _to_normalized(panel.y0_px + bbox[3], panel.source_height_px),
    ]
    result["keypoints_1000"] = [
        [
            _to_normalized(panel.x0_px + point[0], panel.source_width_px),
            _to_normalized(panel.y0_px + point[1], panel.source_height_px),
        ]
        for point in node["keypoints_px"]
    ]
    if node["center_px"] == [-1, -1]:
        result["center_1000"] = [-1, -1]
    else:
        result["center_1000"] = [
            _to_normalized(panel.x0_px + node["center_px"][0], panel.source_width_px),
            _to_normalized(panel.y0_px + node["center_px"][1], panel.source_height_px),
        ]
    result["radius_1000"] = (
        -1
        if node["radius_px"] == -1
        else round(node["radius_px"] / max(panel.source_width_px, panel.source_height_px) * 1000)
    )
    return result


def standard_node_to_pixel(node: dict[str, Any], panel: PanelSpec) -> dict[str, Any]:
    bbox = node["bbox_1000"]
    source_bbox = [
        _to_source_pixel(bbox[0], panel.source_width_px),
        _to_source_pixel(bbox[1], panel.source_height_px),
        _to_source_pixel(bbox[2], panel.source_width_px),
        _to_source_pixel(bbox[3], panel.source_height_px),
    ]
    result = {
        "id": node["id"], "type": node["type"], "subtype": node["subtype"],
        "text": node["text"], "confidence": node["confidence"],
        "bbox_px": [
            source_bbox[0] - panel.x0_px, source_bbox[1] - panel.y0_px,
            source_bbox[2] - panel.x0_px, source_bbox[3] - panel.y0_px,
        ],
        "keypoints_px": [
            [
                _to_source_pixel(point[0], panel.source_width_px) - panel.x0_px,
                _to_source_pixel(point[1], panel.source_height_px) - panel.y0_px,
            ]
            for point in node.get("keypoints_1000", [])
        ],
        "center_px": [-1, -1],
        "radius_px": -1,
    }
    if node.get("center_1000") != [-1, -1]:
        result["center_px"] = [
            _to_source_pixel(node["center_1000"][0], panel.source_width_px) - panel.x0_px,
            _to_source_pixel(node["center_1000"][1], panel.source_height_px) - panel.y0_px,
        ]
    if node.get("radius_1000") != -1:
        result["radius_px"] = round(node["radius_1000"] / 1000 * max(panel.source_width_px, panel.source_height_px))
    return result


def merge_panel_documents(
    *,
    problem_id: str,
    panels: list[PanelSpec],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(panels) != len(documents):
        raise ValueError("panel/document count mismatch")
    nodes: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for panel, document in zip(panels, documents, strict=True):
        local_map: dict[str, str] = {}
        for raw_node in document["visual_nodes"]:
            new_id = f"v{len(nodes) + 1:03d}"
            local_map[raw_node["id"]] = new_id
            node = pixel_node_to_standard(raw_node, panel)
            node["id"] = new_id
            nodes.append(node)
        for raw_ambiguity in document["ambiguities"]:
            ambiguity = deepcopy(raw_ambiguity)
            ambiguity["id"] = f"u{len(ambiguities) + 1:03d}"
            for field in ("candidate_ids", "evidence_visual_ids"):
                ambiguity[field] = [local_map[value] for value in ambiguity[field] if value in local_map]
            ambiguities.append(ambiguity)
    return {"problem_id": problem_id, "visual_nodes": nodes, "ambiguities": ambiguities}


def node_belongs_to_panel(node: dict[str, Any], panel: PanelSpec) -> bool:
    if node.get("image_id") != panel.image_id:
        return False
    bbox = node.get("bbox_1000", [0, 0, 0, 0])
    center_x = _to_source_pixel(round((bbox[0] + bbox[2]) / 2), panel.source_width_px)
    center_y = _to_source_pixel(round((bbox[1] + bbox[3]) / 2), panel.source_height_px)
    x_inside = panel.x0_px <= center_x < panel.x1_px or (
        panel.x1_px == panel.source_width_px and center_x == panel.x1_px
    )
    y_inside = panel.y0_px <= center_y < panel.y1_px or (
        panel.y1_px == panel.source_height_px and center_y == panel.y1_px
    )
    return x_inside and y_inside


def panel_view_from_standard(
    *,
    problem_id: str,
    pass1: dict[str, Any],
    panel: PanelSpec,
) -> dict[str, Any]:
    nodes = [standard_node_to_pixel(node, panel) for node in pass1["visual_nodes"] if node_belongs_to_panel(node, panel)]
    node_ids = {node["id"] for node in nodes}
    ambiguities: list[dict[str, Any]] = []
    for raw in pass1.get("ambiguities", []):
        evidence = [value for value in raw.get("evidence_visual_ids", []) if value in node_ids]
        candidates = [value for value in raw.get("candidate_ids", []) if value in node_ids]
        if evidence or candidates:
            ambiguity = deepcopy(raw)
            ambiguity["evidence_visual_ids"] = evidence
            ambiguity["candidate_ids"] = candidates
            ambiguities.append(ambiguity)
    return {**panel_contract(panel, problem_id), "visual_nodes": nodes, "ambiguities": ambiguities}


def pixel_patch_to_standard(
    patch: dict[str, Any],
    panel: PanelSpec,
) -> dict[str, Any]:
    converted = {
        "problem_id": patch["problem_id"],
        "verdict": patch["verdict"],
        "operations": [],
        "summary": patch["summary"],
    }
    for operation in patch["operations"]:
        node = operation["node"]
        converted["operations"].append({
            "action": operation["action"],
            "target_id": operation["target_id"],
            "node": pixel_node_to_standard(node, panel) if node is not None else None,
            "reason": operation["reason"],
        })
    return converted
