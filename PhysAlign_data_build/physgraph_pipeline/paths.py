"""Locate legacy entry points in a source checkout or installed wheel."""
from importlib.util import find_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if not (SCRIPTS / "physgraph_annotation_lib.py").is_file():
    spec = find_spec("physgraph_legacy")
    if spec is None or spec.origin is None:
        raise ImportError("Missing packaged PhysGraph entry points")
    SCRIPTS = Path(spec.origin).resolve().parent
