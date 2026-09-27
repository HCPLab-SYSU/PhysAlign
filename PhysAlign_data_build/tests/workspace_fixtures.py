"""Isolated workspaces built only from the repository's synthetic example."""
import json
from pathlib import Path
import shutil

from physgraph_pipeline.config import load_pipeline_config
from physgraph_pipeline.workspace import prepare_workspace

ROOT = Path(__file__).resolve().parents[1]


def example_workspace(root: Path) -> Path:
    dataset = root / 'dataset'
    shutil.copytree(ROOT / 'examples/minimal_physics', dataset)
    config = json.loads((ROOT / 'configs/datasets/minimal_physics.json').read_text(encoding='utf-8'))
    config['dataset']['records'] = 'dataset/annotations.jsonl'
    config['dataset']['media_root'] = 'dataset'
    config['annotation']['workspace'] = 'workspace'
    config['annotation'].pop('prompt_dir', None)
    path = root / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    prepare_workspace(load_pipeline_config(path))
    return root / 'workspace'
