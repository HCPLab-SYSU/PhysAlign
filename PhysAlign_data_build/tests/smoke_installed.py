"""Run after installing the built wheel, from outside the source directory."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import physgraph_pipeline
from physgraph_pipeline.paths import ROOT as INSTALLED, SCRIPTS

SOURCE = Path(__file__).resolve().parents[1]
assert Path(physgraph_pipeline.__file__).resolve().parent != SOURCE / "physgraph_pipeline", "Install a wheel, not an editable checkout"
assert SCRIPTS.name == "physgraph_legacy"
for resource in ("physgraph_annotation/prompts/pass1.txt", "physgraph_annotation/schemas/pass5.schema.json",
                 "physgraph_review_app/index.html", "physgraph_review_app/app.js"):
    assert (INSTALLED / resource).is_file(), resource

with tempfile.TemporaryDirectory() as directory:
    temporary = Path(directory)
    shutil.copytree(SOURCE / "examples/minimal_physics", temporary / "dataset")
    config = json.loads((SOURCE / "configs/datasets/minimal_physics.json").read_text(encoding="utf-8"))
    config["dataset"].update(records="dataset/annotations.jsonl", media_root="dataset")
    config["annotation"] = {"workspace": "workspace"}
    (temporary / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for command in (["doctor", "--config", "config.json"], ["prepare", "--config", "config.json"],
                    ["doctor", "--workspace", "workspace"], ["validate", "--workspace", "workspace"],
                    ["validate-pass5", "--workspace", "workspace"]):
        subprocess.run([sys.executable, "-B", "-m", "physgraph_pipeline", *command],
                       cwd=temporary, check=True, timeout=30)
    if importlib.util.find_spec("openai") is not None:
        subprocess.run([sys.executable, "-B", "-m", "physgraph_pipeline", "run", "--workspace", "workspace", "--dry-run"],
                       cwd=temporary, check=True, timeout=30)
    from PIL import Image
    with Image.open(temporary / "dataset/images/block_on_surface.png") as diagram:
        diagram.load()
        assert diagram.size == (640, 360)
print("Installed wheel: resources, portable workspace, validation and offline CLI passed.")
