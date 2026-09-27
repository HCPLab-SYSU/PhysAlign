#!/usr/bin/env python
"""Validate every saved PhysGraph Pass 1–4 document in an annotation workspace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from physgraph_annotation_lib import (
    STAGES,
    json_sha256,
    read_json,
    read_jsonl,
    validate_document,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-all-approved", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    config = read_json(workspace / "workspace_config.json")
    manifest = read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl"))
    by_id = {item["problem_id"]: item for item in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("Duplicate problem ids in blind manifest")
    state = read_json(workspace / "reviews" / "state.json")
    results: list[dict[str, Any]] = []
    totals = Counter()

    for problem_id, problem in by_id.items():
        override = state.get("problems", {}).get(problem_id, {}).get("segmentation_override")
        if override:
            problem = dict(problem)
            problem["segments"] = override["segments"]
            problem["segmentation"] = override["segmentation"]
        previous: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            path = workspace / "passes" / stage / f"{problem_id}.json"
            saved_state = state.get("problems", {}).get(problem_id, {}).get("stages", {}).get(stage, {})
            status = saved_state.get("status", "empty")
            if not path.is_file():
                totals[f"{stage}_empty"] += 1
                if status not in {"empty", None}:
                    results.append({"problem_id": problem_id, "stage": stage, "code": "state_without_file", "message": f"状态为 {status} 但结果文件不存在"})
                continue
            document = read_json(path)
            issues = validate_document(stage, document, problem, previous)
            errors = [issue for issue in issues if issue["level"] == "error"]
            totals[f"{stage}_saved"] += 1
            totals[f"{stage}_errors"] += len(errors)
            if status == "approved":
                totals[f"{stage}_approved"] += 1
            if errors:
                results.extend({"problem_id": problem_id, "stage": stage, **issue} for issue in errors)
            if status == "approved" and errors:
                results.append({"problem_id": problem_id, "stage": stage, "level": "error", "code": "invalid_approval", "path": "$", "message": "已批准结果未通过当前校验器"})
            if status == "approved" and stage in {"pass3", "pass4"} and problem.get("segmentation", {}).get("review_status") == "needs_review":
                results.append({"problem_id": problem_id, "stage": stage, "level": "error", "code": "unreviewed_segmentation", "path": "$.segments", "message": "低置信度题目切分尚未人工确认"})
            expected_hash = saved_state.get("document_sha256")
            if expected_hash and expected_hash != json_sha256(document):
                results.append({"problem_id": problem_id, "stage": stage, "level": "error", "code": "state_hash_mismatch", "path": "$", "message": "结果文件与审核状态记录的 SHA-256 不一致"})
            previous[stage] = document

    unknown_files: list[str] = []
    for stage in STAGES:
        for path in (workspace / "passes" / stage).glob("*.json"):
            if path.stem not in by_id:
                unknown_files.append(str(path))
    for path in unknown_files:
        results.append({"problem_id": "", "stage": "", "level": "error", "code": "unknown_pass_file", "path": path, "message": "文件名不对应任何 blind manifest problem_id"})

    approved_g_obs = totals["pass4_approved"]
    report = {
        "workspace": str(workspace),
        "problems": len(manifest),
        "totals": dict(sorted(totals.items())),
        "approved_g_obs": approved_g_obs,
        "all_g_obs_approved": approved_g_obs == len(manifest),
        "error_count": len(results),
        "errors": results,
    }
    if args.report:
        write_json_atomic(args.report.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if results or (args.require_all_approved and approved_g_obs != len(manifest)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
