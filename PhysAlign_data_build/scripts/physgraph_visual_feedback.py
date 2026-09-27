#!/usr/bin/env python
"""Render auditable Pass 1 overlays for model and human visual verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from physgraph_annotation_lib import write_json_atomic


PALETTE = (
    "#1769e0", "#d6473d", "#07885f", "#8950c8", "#c47610",
    "#008aa6", "#bb3b84", "#557019", "#6f5a45", "#374b9b",
)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _coord(value: int, size: int, offset: int) -> int:
    return offset + round(value / 1000 * size)


def render_pass1_overlay(
    *,
    image_path: Path,
    image_id: str,
    visual_nodes: list[dict[str, Any]],
    output_path: Path,
    title: str,
    target_long_edge: int = 1200,
) -> dict[str, Any]:
    """Render one original image with boxes, keypoints, IDs, ticks, and legend."""

    with Image.open(image_path) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGB")
    source_width, source_height = source.size
    longest = max(source_width, source_height)
    scale = min(4.0, max(1.0, target_long_edge / longest))
    drawn_width = max(1, round(source_width * scale))
    drawn_height = max(1, round(source_height * scale))
    rendered = source.resize((drawn_width, drawn_height), Image.Resampling.LANCZOS)

    nodes = [node for node in visual_nodes if node.get("image_id") == image_id]
    margin_left, margin_top, margin_bottom, legend_width = 58, 72, 44, 370
    legend_row = 24
    canvas_height = max(
        margin_top + drawn_height + margin_bottom,
        margin_top + 42 + max(1, len(nodes)) * legend_row,
    )
    canvas_width = margin_left + drawn_width + legend_width
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(rendered, (margin_left, margin_top))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20, bold=True)
    body_font = _font(15)
    small_font = _font(12)
    draw.text((16, 14), title, fill="#172038", font=title_font)
    draw.text(
        (16, 42),
        f"{image_id} | original={source_width}x{source_height}px | overlay={drawn_width}x{drawn_height}px | coordinates=0..1000",
        fill="#4f5a70",
        font=small_font,
    )
    draw.rectangle(
        (margin_left, margin_top, margin_left + drawn_width, margin_top + drawn_height),
        outline="#20293a",
        width=2,
    )

    for tick in (0, 250, 500, 750, 1000):
        x = _coord(tick, drawn_width, margin_left)
        y = _coord(tick, drawn_height, margin_top)
        draw.line((x, margin_top - 7, x, margin_top), fill="#566076", width=2)
        draw.text((x - 12, margin_top - 24), str(tick), fill="#566076", font=small_font)
        draw.line((margin_left - 7, y, margin_left, y), fill="#566076", width=2)
        draw.text((4, y - 7), str(tick), fill="#566076", font=small_font)

    legend_x = margin_left + drawn_width + 18
    draw.text((legend_x, margin_top), "NODE LEGEND", fill="#172038", font=title_font)
    for index, node in enumerate(nodes):
        color = PALETTE[index % len(PALETTE)]
        bbox = node.get("bbox_1000")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1 = _coord(bbox[0], drawn_width, margin_left)
            y1 = _coord(bbox[1], drawn_height, margin_top)
            x2 = _coord(bbox[2], drawn_width, margin_left)
            y2 = _coord(bbox[3], drawn_height, margin_top)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, round(scale)))
            label = str(node.get("id", "?"))
            label_box = draw.textbbox((0, 0), label, font=body_font)
            label_width = label_box[2] - label_box[0] + 8
            label_height = label_box[3] - label_box[1] + 6
            label_y = max(margin_top, y1 - label_height)
            draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
            draw.text((x1 + 4, label_y + 2), label, fill="white", font=body_font)
        for point in node.get("keypoints_1000", []) if isinstance(node.get("keypoints_1000"), list) else []:
            if isinstance(point, list) and len(point) == 2 and all(isinstance(value, int) for value in point):
                px = _coord(point[0], drawn_width, margin_left)
                py = _coord(point[1], drawn_height, margin_top)
                radius = max(4, round(scale * 1.5))
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="white", width=1)
        center = node.get("center_1000")
        if (
            node.get("type") in {"circle", "arc"}
            and isinstance(center, list) and len(center) == 2
            and all(isinstance(value, int) and value >= 0 for value in center)
        ):
            cx = _coord(center[0], drawn_width, margin_left)
            cy = _coord(center[1], drawn_height, margin_top)
            arm = max(5, round(scale * 2))
            draw.line((cx - arm, cy, cx + arm, cy), fill=color, width=2)
            draw.line((cx, cy - arm, cx, cy + arm), fill=color, width=2)

        legend_y = margin_top + 36 + index * legend_row
        draw.rectangle((legend_x, legend_y + 3, legend_x + 14, legend_y + 17), fill=color)
        node_text = f"{node.get('id', '?')}  {node.get('type', '?')}  {node.get('subtype', '')}"
        draw.text((legend_x + 22, legend_y), node_text[:43], fill="#273149", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    return {
        "image_id": image_id,
        "source_path": str(image_path),
        "source_width": source_width,
        "source_height": source_height,
        "overlay_path": str(output_path),
        "overlay_width": canvas_width,
        "overlay_height": canvas_height,
        "node_count": len(nodes),
        "scale": round(scale, 6),
    }


def render_pass1_overlays(
    *,
    problem: dict[str, Any],
    dataset_dir: Path,
    pass1: dict[str, Any],
    output_dir: Path,
    label: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    output_paths: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for image in problem.get("images", []):
        image_id = image["image_id"]
        output_path = output_dir / f"{problem['problem_id']}.{label}.{image_id}.png"
        info = render_pass1_overlay(
            image_path=dataset_dir / image["path"],
            image_id=image_id,
            visual_nodes=pass1.get("visual_nodes", []),
            output_path=output_path,
            title=f"PASS 1 VISUAL CHECK | {problem['problem_id']} | {label}",
        )
        output_paths.append(output_path)
        manifest.append(info)
    manifest_path = output_dir / f"{problem['problem_id']}.{label}.overlay_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return output_paths, manifest
