#!/usr/bin/env python
"""Import one blind four-pass model response into the annotation workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STAGES = ("pass1", "pass2", "pass3", "pass4")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--workspace", type=Path, required=True
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if set(payload) != set(STAGES):
        raise SystemExit(f"Expected exactly {STAGES}; got {tuple(payload)}")

    problem_ids = {payload[stage].get("problem_id") for stage in STAGES}
    if len(problem_ids) != 1 or not next(iter(problem_ids), ""):
        raise SystemExit(f"Four stages do not share one problem_id: {problem_ids}")
    problem_id = next(iter(problem_ids))

    for stage in STAGES:
        destination = args.workspace / "passes" / stage / f"{problem_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload[stage], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    print(problem_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
