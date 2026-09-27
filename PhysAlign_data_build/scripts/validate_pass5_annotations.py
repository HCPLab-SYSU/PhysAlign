#!/usr/bin/env python
"""Validate Pass 5 coverage and every saved solution-step alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from physgraph_annotation_lib import read_json, read_jsonl, resolve_source_annotations
from run_approved_physgraph_pass5 import approved_problem_ids, standard_solution, validate_pass5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    state = read_json(workspace / "reviews" / "state.json")
    approved = set(approved_problem_ids(state))
    pass5_dir = workspace / "passes" / "pass5"
    files = {path.stem for path in pass5_dir.glob("*.json")}
    source_path = resolve_source_annotations(workspace)
    source_records = read_jsonl(source_path) if source_path.suffix.lower() == ".jsonl" else read_json(source_path)
    if not isinstance(source_records, list):
        raise ValueError("Pass 5 当前要求源标注 JSON 顶层为 object 数组")
    manifest = {
        item["problem_id"]: item
        for item in read_jsonl(workspace / "blind" / "manifest.jsonl")
    }
    validation_errors: list[dict[str, object]] = []
    for problem_id in sorted(files):
        problem = manifest.get(problem_id)
        if problem is None:
            validation_errors.append(
                {"problem_id": problem_id, "errors": ["blind manifest 中不存在该题目"]}
            )
            continue
        source_index = problem.get("source_record_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(source_records):
            validation_errors.append(
                {"problem_id": problem_id, "errors": ["source_record_index 无效"]}
            )
            continue
        document = read_json(pass5_dir / f"{problem_id}.json")
        g_obs = read_json(workspace / "passes" / "pass4" / f"{problem_id}.json")
        errors = validate_pass5(
            document,
            problem_id,
            g_obs,
            standard_solution(
                source_records[source_index]
            ),
        )
        if errors:
            validation_errors.append({"problem_id": problem_id, "errors": errors})
    report = {
        "approved_all4": len(approved),
        "pass5_files": len(files),
        "missing": sorted(approved - files),
        "extra": sorted(files - approved),
        "validation_error_files": validation_errors,
        "valid": not (approved - files or files - approved or validation_errors),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
