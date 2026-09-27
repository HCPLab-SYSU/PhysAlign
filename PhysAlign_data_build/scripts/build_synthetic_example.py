"""Rebuild the original synthetic PNG and its source metadata (no external data)."""
from pathlib import Path
import hashlib
import json

from PIL import Image, ImageDraw


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "examples/minimal_physics"
    target = root / "images/block_on_surface.png"
    canvas = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(canvas)
    draw.line((60, 260, 580, 260), fill="#263238", width=4)
    draw.rectangle((210, 150, 350, 258), fill="#b8c9ff", outline="#263238", width=3)
    draw.line((350, 195, 500, 195), fill="#c62828", width=5)
    draw.polygon([(500, 195), (478, 184), (478, 206)], fill="#c62828")
    # Vector letter strokes keep the fixture independent of system fonts.
    draw.line([(263, 223), (278, 180), (293, 223)], fill="#263238", width=3)
    draw.line((269, 207, 287, 207), fill="#263238", width=3)
    draw.line([(418, 174), (418, 142), (440, 142)], fill="#c62828", width=3)
    draw.line((418, 156, 435, 156), fill="#c62828", width=3)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG")
    source = root / "annotations.jsonl"
    record = json.loads(source.read_text(encoding="utf-8"))
    relative = target.relative_to(root).as_posix()
    record["images"] = [relative]
    record["metadata"]["question_images"] = [{
        "path": relative, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "format": "PNG", "width": 640, "height": 360, "bytes": target.stat().st_size,
    }]
    source.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
