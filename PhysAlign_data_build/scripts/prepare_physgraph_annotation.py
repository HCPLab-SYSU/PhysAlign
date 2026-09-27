#!/usr/bin/env python
"""Create an answer-blind, resumable PhysGraph annotation workspace."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from physgraph_annotation_lib import (
    STAGES,
    WORKSPACE_SCHEMA_VERSION,
    blank_document,
    contract_json_schemas,
    json_sha256,
    portable_path,
    read_json,
    write_json_atomic,
    write_jsonl_atomic,
)


BOILERPLATE = "Solve the following physics problem."
OPTION_PATTERN = re.compile(r"(?m)^\s*([A-H])(?:[.)．])\s*(.+?)(?=\n\s*[A-H](?:[.)．])\s*|\Z)", re.S)
QUERY_CUE = re.compile(
    r"(?i)^(which|what|find|calculate|determine|how|choose|select|state|draw|prove|show|is |are |does |do |can |at what|when )"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Rebuild manifest while preserving pass files and review state")
    return parser.parse_args()


def clean_prompt(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    lines = text.splitlines()
    while lines and lines[0].strip() == "<image>":
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip() == BOILERPLATE:
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def split_stem_query_options(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
    inline_option_start = re.search(r"(?:\(\s*\)|\?)\s*(?=A[.．]\s*)", question)
    if inline_option_start:
        prefix = question[: inline_option_start.end()]
        suffix = question[inline_option_start.end() :]
        suffix = re.sub(r"(?:^|[ \t]+)([A-H])[.．]\s*", r"\n\1. ", suffix).lstrip()
        question = f"{prefix}\n{suffix}"
    matches = list(OPTION_PATTERN.finditer(question))
    options: list[dict[str, str]] = []
    body = question
    if matches:
        first = matches[0]
        body = question[: first.start()].strip()
        options = [
            {"label": match.group(1), "text": " ".join(match.group(2).strip().splitlines())}
            for match in matches
        ]

    split_method = "last_query_sentence"
    confidence = "medium"
    stem, query = "", body
    numbered = re.search(r"(?m)^\s*\(1\)\s+", body)
    find_marker = re.search(r"(?i)(?:^|\n)\s*(find|calculate|determine)\s*:\s*", body)
    if numbered and numbered.start() > 0:
        stem, query = body[: numbered.start()].strip(), body[numbered.start() :].strip()
        split_method, confidence = "numbered_subquestions", "high"
    elif find_marker and find_marker.start() > 0:
        stem, query = body[: find_marker.start()].strip(), body[find_marker.start() :].strip()
        split_method, confidence = "find_marker", "high"
    else:
        query_phrase_matches = list(
            re.finditer(
                r"(?i)\b(which of the following\b|what (?:is|are)\b|how (?:far|long|much|many|fast)\b|find\b|calculate\b|determine\b)",
                body,
            )
        )
        if query_phrase_matches:
            marker = query_phrase_matches[-1]
            stem, query = body[: marker.start()].rstrip(" ,"), body[marker.start() :].strip()
            split_method, confidence = "query_phrase", "high"
        else:
            sentence_starts = [match.end() for match in re.finditer(r"(?<=[.!?])\s+", body)]
            candidates = [body[index:].strip() for index in sentence_starts]
            chosen: tuple[int, str] | None = None
            for index, candidate in zip(sentence_starts, candidates):
                if QUERY_CUE.match(candidate):
                    chosen = (index, candidate)
            if chosen:
                stem, query = body[: chosen[0]].strip(), chosen[1]
            elif QUERY_CUE.match(body):
                stem, query = "", body
                split_method, confidence = "query_only", "high"
            elif re.search(r"(?:\(\s*\)|\?)\s*$", body) and sentence_starts:
                last_start = sentence_starts[-1]
                stem, query = body[:last_start].strip(), body[last_start:].strip()
                split_method, confidence = "terminal_question_sentence", "medium"
            else:
                split_method, confidence = "unsplit", "low"
                stem, query = body, ""

    if options and query == "" and body:
        last_break = max(body.rfind("?"), body.rfind("."))
        if last_break > 0:
            previous_break = max(body.rfind("?", 0, last_break), body.rfind(".", 0, last_break))
            candidate = body[previous_break + 1 :].strip()
            if QUERY_CUE.match(candidate):
                stem, query = body[: previous_break + 1].strip(), candidate
                split_method, confidence = "option_question", "medium"

    segments = {"stem": stem, "query": query, "options": options}
    segmentation = {
        "method": split_method,
        "confidence": confidence,
        "review_status": "needs_review" if confidence == "low" else "machine_split",
        "edited": False,
    }
    return segments, segmentation


def image_records(sample: dict[str, Any], dataset_dir: Path) -> list[dict[str, Any]]:
    metadata_images = sample.get("metadata", {}).get("question_images", [])
    records: list[dict[str, Any]] = []
    image_value = sample.get("image", [])
    relative_images = [image_value] if isinstance(image_value, str) else list(image_value or [])
    for index, relative in enumerate(relative_images):
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"Invalid image path for {sample.get('id')}: {relative!r}")
        path = dataset_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for {sample['id']}: {relative}")
        metadata = metadata_images[index] if index < len(metadata_images) else {}
        records.append(
            {
                "image_id": f"img_{index}",
                "path": relative.replace("\\", "/"),
                "width": int(metadata.get("width", 0)),
                "height": int(metadata.get("height", 0)),
                "sha256": str(metadata.get("sha256", "")),
            }
        )
    return records


def build_blind_record(sample: dict[str, Any], source_index: int, dataset_dir: Path) -> dict[str, Any]:
    conversations = sample.get("conversations", [])
    human = next((item for item in conversations if item.get("from") == "human"), None)
    if human is None or not isinstance(human.get("value"), str):
        raise ValueError(f"Missing human prompt: {sample.get('id')}")
    raw_question = clean_prompt(human["value"])
    segments, segmentation = split_stem_query_options(raw_question)
    metadata = sample.get("metadata", {})
    return {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "problem_id": sample["id"],
        "source_record_index": source_index,
        "source_dataset": metadata.get("source_dataset", ""),
        "source_sample_id": metadata.get("source_sample_id", ""),
        "source_split": metadata.get("source_split", ""),
        "language": metadata.get("language", ""),
        "images": image_records(sample, dataset_dir),
        "raw_question": raw_question,
        "segments": segments,
        "segmentation": segmentation,
    }


def assert_blind(records: list[dict[str, Any]]) -> None:
    prohibited = {"answer", "reasoning", "solution", "conversations", "assistant", "gpt"}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in prohibited:
                    raise ValueError(f"Blind manifest leakage key at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(records, "$")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    workspace = args.workspace.resolve()
    dataset_dir = source.parent
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest_path = workspace / "blind" / "manifest.jsonl"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Workspace already prepared: {manifest_path}; use --force to refresh only the manifest")

    samples: list[dict[str, Any]] = read_json(source)
    records = [build_blind_record(sample, index, dataset_dir) for index, sample in enumerate(samples)]
    if len({record["problem_id"] for record in records}) != len(records):
        raise ValueError("Duplicate problem_id in source annotations")
    assert_blind(records)

    write_jsonl_atomic(manifest_path, records)
    for stage in STAGES:
        (workspace / "passes" / stage).mkdir(parents=True, exist_ok=True)
    (workspace / "reviews").mkdir(parents=True, exist_ok=True)
    (workspace / "exports").mkdir(parents=True, exist_ok=True)
    history = workspace / "reviews" / "history.jsonl"
    if not history.exists():
        history.write_text("", encoding="utf-8")
    state = workspace / "reviews" / "state.json"
    if not state.exists():
        write_json_atomic(state, {"workspace_schema_version": WORKSPACE_SCHEMA_VERSION, "problems": {}})

    schema_dir = workspace / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for stage, schema in contract_json_schemas().items():
        write_json_atomic(schema_dir / f"{stage}.schema.json", schema)

    source_digest = hashlib_sha256_file(source)
    summary = {
        "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
        "purpose": "answer-blind Pass 1–4 Observed-PhysGraph annotation",
        "source_annotations": portable_path(source, workspace),
        "source_annotations_sha256": source_digest,
        "source_dataset_dir": portable_path(dataset_dir, workspace),
        "problems": len(records),
        "images": sum(len(record["images"]) for record in records),
        "sources": dict(Counter(record["source_dataset"] for record in records)),
        "segmentation_confidence": dict(Counter(record["segmentation"]["confidence"] for record in records)),
        "answer_solution_present": False,
        "pass5_enabled": False,
    }
    write_json_atomic(workspace / "summary.json", summary)
    write_json_atomic(
        workspace / "workspace_config.json",
        {
            "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
            "path_base": "workspace",
            "dataset_dir": portable_path(dataset_dir, workspace),
            "source_annotations": portable_path(source, workspace),
            "manifest": "blind/manifest.jsonl",
            "model_policy": {
                "passes_1_to_4": "configured OpenAI-compatible multimodal model",
                "answers_visible": False,
            },
            "approval_gate": "Pass 4 must be valid and manually approved before any Pass 5 export",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def hashlib_sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
