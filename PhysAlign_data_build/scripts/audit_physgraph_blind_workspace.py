#!/usr/bin/env python
"""Audit that the Pass 1–4 workspace contains no answer or solution payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physgraph_annotation_lib import PROHIBITED_ANNOTATION_KEYS, STAGES, read_json, read_jsonl


BLIND_MANIFEST_KEYS = {
    "workspace_schema_version",
    "problem_id",
    "source_record_index",
    "source_dataset",
    "source_sample_id",
    "source_split",
    "language",
    "images",
    "raw_question",
    "segments",
    "segmentation",
}
EXTRA_PROHIBITED = {"conversations", "assistant", "gpt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    return parser.parse_args()


def scan_keys(value: Any, path: str, findings: list[dict[str, str]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in PROHIBITED_ANNOTATION_KEYS | EXTRA_PROHIBITED:
                findings.append({"path": f"{path}.{key}", "key": key, "message": "禁止字段出现在 Pass 1–4 盲化工作区"})
            scan_keys(child, f"{path}.{key}", findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            scan_keys(child, f"{path}[{index}]", findings)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    config = read_json(workspace / "workspace_config.json")
    records = read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl"))
    findings: list[dict[str, str]] = []
    for index, record in enumerate(records):
        extra = set(record) - BLIND_MANIFEST_KEYS
        missing = BLIND_MANIFEST_KEYS - set(record)
        if extra:
            findings.append({"path": f"manifest[{index}]", "key": ",".join(sorted(extra)), "message": "blind manifest 存在未授权字段"})
        if missing:
            findings.append({"path": f"manifest[{index}]", "key": ",".join(sorted(missing)), "message": "blind manifest 缺少固定字段"})
        scan_keys(record, f"manifest[{index}]", findings)
    scanned_pass_files = 0
    for stage in STAGES:
        for path in (workspace / "passes" / stage).glob("*.json"):
            scanned_pass_files += 1
            scan_keys(read_json(path), str(path), findings)
    report = {
        "workspace": str(workspace),
        "blind_records": len(records),
        "scanned_pass_files": scanned_pass_files,
        "answer_solution_visible": bool(findings),
        "finding_count": len(findings),
        "findings": findings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
