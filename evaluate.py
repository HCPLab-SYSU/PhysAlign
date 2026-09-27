"""CLI entry point, including Python embedded runtimes with isolated sys.path."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physalign.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
