"""Offline plotting entry point, including for the portable test interpreter."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physalign_viz.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
