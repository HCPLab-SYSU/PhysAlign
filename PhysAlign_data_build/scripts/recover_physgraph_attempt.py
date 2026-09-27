#!/usr/bin/env python
"""Locally recover a raw Pass 1-4 API attempt using invariant-only fixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from physgraph_annotation_lib import (  # noqa: E402
    STAGES,
    canonicalize_pass4_visual_nodes,
    json_sha256,
    read_json,
    read_jsonl,
    write_json_atomic,
)
from run_high_confidence_physgraph_annotation import (  # noqa: E402
    complete_and_valid,
    save_payload,
    validate_payload,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--attempt-json", type=Path, required=True)
    parser.add_argument("--expected-problem-id", required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--update-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    attempt_path = args.attempt_json.resolve()
    if not attempt_path.is_file():
        raise SystemExit(f"Attempt文件不存在：{attempt_path}")
    file_hash = hashlib.sha256(attempt_path.read_bytes()).hexdigest()
    if file_hash.lower() != args.expected_file_sha256.lower():
        raise SystemExit(
            "Attempt文件SHA-256不匹配；为避免修复错误文件，已停止。"
            f"\nexpected={args.expected_file_sha256.lower()}\nactual={file_hash.lower()}"
        )

    raw_payload = read_json(attempt_path)
    if not isinstance(raw_payload, dict) or set(raw_payload) != set(STAGES):
        raise SystemExit("Attempt顶层必须恰好包含pass1、pass2、pass3、pass4")
    problem_ids = {
        document.get("problem_id")
        for document in raw_payload.values()
        if isinstance(document, dict)
    }
    if problem_ids != {args.expected_problem_id}:
        raise SystemExit(f"Attempt内problem_id不一致：{sorted(str(value) for value in problem_ids)}")

    config = read_json(workspace / "workspace_config.json")
    manifest = read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl"))
    problem = next((item for item in manifest if item.get("problem_id") == args.expected_problem_id), None)
    if problem is None:
        raise SystemExit(f"盲化manifest中找不到题目：{args.expected_problem_id}")

    payload = deepcopy(raw_payload)
    changes = canonicalize_pass4_visual_nodes(payload)
    if not changes:
        raise SystemExit("没有发现可由Pass 1确定性恢复的Pass 4视觉节点差异")
    issues = validate_payload(payload, problem)
    if issues:
        print(json.dumps({"valid": False, "issues": issues}, ensure_ascii=False, indent=2))
        return 1

    existing = [
        workspace / "passes" / stage / f"{args.expected_problem_id}.json"
        for stage in STAGES
        if (workspace / "passes" / stage / f"{args.expected_problem_id}.json").exists()
    ]
    if existing:
        raise SystemExit("目标Pass文件已经存在，拒绝覆盖：" + "，".join(str(path) for path in existing))

    normalized_path = attempt_path.with_name(attempt_path.stem + ".normalized.json")
    recovery_path = attempt_path.with_name(attempt_path.stem + ".recovery.json")
    write_json_atomic(normalized_path, payload)
    save_payload(workspace, payload, args.expected_problem_id)
    if not complete_and_valid(workspace, problem):
        raise SystemExit("保存后重新读取校验失败；请检查磁盘中的Pass文件")

    recovery: dict[str, Any] = {
        "schema_version": 1,
        "recovered_at_utc": utc_now(),
        "problem_id": args.expected_problem_id,
        "source_attempt": str(attempt_path),
        "source_file_sha256": file_hash,
        "normalized_attempt": str(normalized_path),
        "rule": "pass4_visual_nodes_exact_pass1_subset",
        "changes": changes,
        "validation": {"valid": True, "errors": 0},
        "saved_pass_sha256": {
            stage: json_sha256(payload[stage]) for stage in STAGES
        },
    }
    write_json_atomic(recovery_path, recovery)

    if args.update_progress:
        progress_path = workspace / "batch_progress.json"
        progress = read_json(progress_path)
        before = list(progress.get("failures", []))
        remaining = [item for item in before if item.get("problem_id") != args.expected_problem_id]
        recovered_failure = len(remaining) < len(before)
        progress["failures"] = remaining
        progress["failed_this_run"] = len(remaining)
        if recovered_failure:
            progress["completed_this_run"] = int(progress.get("completed_this_run", 0)) + 1
        progress.setdefault("local_recoveries", []).append(
            {
                "problem_id": args.expected_problem_id,
                "recovery_audit": str(recovery_path),
                "recovered_at_utc": recovery["recovered_at_utc"],
            }
        )
        progress["status"] = "complete"
        progress["updated_at_utc"] = utc_now()
        write_json_atomic(progress_path, progress)

    print(json.dumps(recovery, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
