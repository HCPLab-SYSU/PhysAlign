#!/usr/bin/env python
"""Build Pass 5 solution-step alignments through an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from physgraph_annotation_lib import (  # noqa: E402
    effective_manifest,
    json_sha256,
    problem_semantics_sha256,
    read_json,
    read_jsonl,
    resolve_source_annotations,
    write_json_atomic,
)
from physgraph_api_client import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    ExclusiveRunLock,
    PhysGraphAPIClient,
    PhysGraphAPIError,
    PhysGraphAPIResponseError,
    compact_json,
    load_pass5_system_instructions,
    merge_usage,
    normalize_base_url,
    safe_endpoint_label,
    write_attempt_log,
)

STEP_TYPES = {
    "read_given", "select_law", "define_variable", "form_equation", "derive",
    "calculate", "case_analysis", "check", "conclude",
}
STEP_KEYS = {
    "step_id", "text", "step_type", "uses_obs_node_ids",
    "uses_obs_quantity_ids", "uses_obs_relation_ids", "uses_obs_constraint_ids",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL; otherwise OPENAI_BASE_URL is used")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-mode", choices=("chat", "responses", "auto"), default="chat")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.model:
        if args.dry_run:
            args.model = "dry-run"
        else:
            parser.error("--model or OPENAI_MODEL is required")
    return args


def approved_problem_ids(state: dict[str, Any]) -> list[str]:
    selected: list[str] = []
    for problem_id, problem_state in state.get("problems", {}).items():
        stages = problem_state.get("stages", {})
        if all(stages.get(stage, {}).get("status") == "approved" for stage in ("pass1", "pass2", "pass3", "pass4")):
            selected.append(problem_id)
    return sorted(selected)


def pass5_is_current(
    workspace: Path,
    problem_id: str,
    problem: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    output = workspace / "passes" / "pass5" / f"{problem_id}.json"
    if not output.is_file():
        return False
    repair = state.get("problems", {}).get(problem_id, {}).get("segmentation_repair")
    if not isinstance(repair, dict):
        return True
    pass4_path = workspace / "passes" / "pass4" / f"{problem_id}.json"
    return (
        repair.get("status") == "resolved"
        and repair.get("pass5_status") == "complete"
        and pass4_path.is_file()
        and repair.get("pass5_problem_semantics_sha256") == problem_semantics_sha256(problem)
        and repair.get("pass5_g_obs_sha256") == json_sha256(read_json(pass4_path))
        and repair.get("pass5_document_sha256") == json_sha256(read_json(output))
    )


def standard_solution(record: dict[str, Any]) -> str:
    for container in (record, record.get("metadata", {})):
        if not isinstance(container, dict):
            continue
        for key in ("solution", "reasoning", "analysis", "explanation"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ("conversations", "messages"):
        messages = record.get(key, [])
        if not isinstance(messages, list):
            continue
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = message.get("from", message.get("role"))
            value = message.get("value", message.get("content"))
            if role in {"gpt", "assistant"} and isinstance(value, str) and value.strip():
                return value
    return ""


def standard_answer(record: dict[str, Any]) -> Any:
    if "answer" in record:
        return record["answer"]
    metadata = record.get("metadata", {})
    return metadata.get("answer", "") if isinstance(metadata, dict) else ""


def validate_pass5(
    document: Any,
    problem_id: str,
    g_obs: dict[str, Any],
    solution: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(solution, str) or not solution.strip():
        return ["源记录中找不到非空标准解答，无法执行Pass 5原文对齐"]
    if not isinstance(document, dict) or set(document) != {"problem_id", "solution_steps", "final_answer"}:
        return ["顶层字段必须恰好为 problem_id、solution_steps、final_answer"]
    if document.get("problem_id") != problem_id:
        errors.append("problem_id不匹配")
    steps = document.get("solution_steps")
    if not isinstance(steps, list):
        return errors + ["solution_steps必须是数组"]
    physical_ids = {item["id"] for item in g_obs.get("physical_nodes", [])}
    quantity_ids = {item["id"] for item in g_obs.get("quantities", [])}
    relation_ids = {item["id"] for item in g_obs.get("relations", [])}
    constraint_ids = {item["id"] for item in g_obs.get("constraints", [])}
    last_offset = -1
    for index, step in enumerate(steps, start=1):
        label = f"solution_steps[{index - 1}]"
        if not isinstance(step, dict) or set(step) != STEP_KEYS:
            errors.append(f"{label}字段不完整或包含额外字段")
            continue
        if step.get("step_id") != f"s{index:03d}":
            errors.append(f"{label}.step_id必须为s{index:03d}")
        text = step.get("text")
        if not isinstance(text, str) or not text:
            errors.append(f"{label}.text不能为空")
        else:
            offset = solution.find(text)
            if offset < 0:
                errors.append(f"{label}.text不是标准解答的原文片段")
            elif offset < last_offset:
                errors.append(f"{label}.text没有按标准解答原文顺序排列")
            else:
                last_offset = offset
        if step.get("step_type") not in STEP_TYPES:
            errors.append(f"{label}.step_type不在允许枚举中")
        checks = (
            ("uses_obs_node_ids", physical_ids),
            ("uses_obs_quantity_ids", quantity_ids),
            ("uses_obs_relation_ids", relation_ids),
            ("uses_obs_constraint_ids", constraint_ids),
        )
        for field, allowed in checks:
            values = step.get(field)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                errors.append(f"{label}.{field}必须是字符串数组")
            elif len(values) != len(set(values)):
                errors.append(f"{label}.{field}不能包含重复ID")
            elif any(value not in allowed for value in values):
                errors.append(f"{label}.{field}包含G_obs中不存在的ID")
    if not isinstance(document.get("final_answer"), str):
        errors.append("final_answer必须是字符串")
    return errors


def prompt_for(
    problem_id: str,
    question: str,
    g_obs: dict[str, Any],
    solution: str,
    answer: Any,
    attempt: int,
    prior: dict[str, Any] | None,
    errors: list[str],
) -> str:
    repair = ""
    if prior is not None:
        repair = (
            "\n这是修复轮。上一轮输出如下：\n"
            f"{compact_json(prior)}\n"
            "仅修复以下错误：\n"
            f"{compact_json(errors)}\n"
        )
    return f"""你是物理解答步骤对齐器，不是解题器。严格执行system消息中的Pass 5规则。

问题编号：{problem_id}
这是第{attempt}次生成或修复。

题目：
{question}

已经人工确认的Observed-PhysGraph：
{compact_json(g_obs)}

数据集原始标准解答：
{solution}

数据集原始标准答案字段：
{compact_json(answer)}

任务要求：保持标准解答数学含义不变；不要重新求解；不要增加标准解答中不存在的步骤；按原文顺序切分为原子步骤。每个step的text必须是标准解答中连续、逐字一致的原文片段，不得改写。只对某一步直接使用的G_obs ID进行对齐；仅依赖上一推导步骤时对应uses数组可以为空。不得修改G_obs，不得创建Derived-PhysGraph。

最终只输出合法JSON，不要Markdown或解释：
{{"problem_id":"{problem_id}","solution_steps":[{{"step_id":"s001","text":"标准解答原文片段","step_type":"read_given","uses_obs_node_ids":[],"uses_obs_quantity_ids":[],"uses_obs_relation_ids":[],"uses_obs_constraint_ids":[]}}],"final_answer":"..."}}
{repair}"""


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    state = read_json(workspace / "reviews" / "state.json")
    selected = approved_problem_ids(state)
    manifest = {
        item["problem_id"]: item
        for item in effective_manifest(read_jsonl(workspace / "blind" / "manifest.jsonl"), state)
    }
    source_path = resolve_source_annotations(workspace)
    source_records = read_jsonl(source_path) if source_path.suffix.lower() == ".jsonl" else read_json(source_path)
    if not isinstance(source_records, list) or not all(
        isinstance(record, dict) for record in source_records
    ):
        raise ValueError("Pass 5 当前要求源标注 JSON 顶层为 object 数组")
    originals: dict[str, dict[str, Any]] = {}
    for problem_id, problem in manifest.items():
        source_index = problem.get("source_record_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(source_records):
            raise ValueError(f"{problem_id} 的 source_record_index 无效")
        originals[problem_id] = source_records[source_index]
    output_dir = workspace / "passes" / "pass5"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        problem_id for problem_id in selected
        if not pass5_is_current(workspace, problem_id, manifest[problem_id], state)
    ]
    if args.limit > 0:
        pending = pending[: args.limit]
    missing_solutions = [
        problem_id
        for problem_id in pending
        if not standard_solution(originals[problem_id]).strip()
    ]
    if missing_solutions:
        raise ValueError(
            "以下源记录缺少标准解答，无法执行Pass 5："
            + ", ".join(missing_solutions[:10])
        )
    run_dir = workspace / "pass5_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = workspace / "pass5_progress.json"
    progress: dict[str, Any] = {
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "status": "dry_run" if args.dry_run else "running",
        "approved_total": len(selected), "queued_this_run": len(pending),
        "completed_this_run": 0, "failed_this_run": 0,
        "current_problem_id": "", "failures": [], "run_dir": str(run_dir),
        "provider": {
            "endpoint": safe_endpoint_label(normalize_base_url(args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)),
            "api_mode": args.api_mode,
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "api_key_env": args.api_key_env,
        },
        "api_usage": {},
    }
    write_json_atomic(progress_path, progress)
    if args.dry_run:
        print(json.dumps(progress, ensure_ascii=False, indent=2))
        return 0

    try:
        api_client = PhysGraphAPIClient(
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            api_mode=args.api_mode,
            timeout_seconds=args.timeout_seconds,
            transport_retries=args.transport_retries,
        )
    except (PhysGraphAPIError, ValueError) as exc:
        progress.update(status="configuration_error", configuration_error=str(exc), updated_at_utc=utc_now())
        write_json_atomic(progress_path, progress)
        print(str(exc), file=sys.stderr)
        return 2
    system_instructions = load_pass5_system_instructions(ROOT, workspace)
    run_lock = ExclusiveRunLock(workspace / ".physgraph_api_annotation.lock")
    try:
        run_lock.acquire()
    except PhysGraphAPIError as exc:
        progress.update(status="lock_error", lock_error=str(exc), updated_at_utc=utc_now())
        write_json_atomic(progress_path, progress)
        print(str(exc), file=sys.stderr)
        return 3
    atexit.register(run_lock.release)

    for queue_index, problem_id in enumerate(pending, start=1):
        progress.update(current_problem_id=problem_id, queue_index=queue_index, updated_at_utc=utc_now())
        write_json_atomic(progress_path, progress)
        record = originals[problem_id]
        problem = manifest[problem_id]
        g_obs = read_json(workspace / "passes" / "pass4" / f"{problem_id}.json")
        solution = standard_solution(record)
        question = problem.get("raw_question", "")
        answer = standard_answer(record)
        prior: dict[str, Any] | None = None
        errors: list[str] = []
        succeeded = False
        for attempt in range(1, args.max_attempts + 1):
            output = run_dir / f"{problem_id}.attempt{attempt}.json"
            log = run_dir / f"{problem_id}.attempt{attempt}.api.json"
            try:
                document, api_result = api_client.request_json(
                    model=args.model,
                    reasoning_effort=args.reasoning,
                    system_prompt=system_instructions,
                    user_prompt=prompt_for(problem_id, question, g_obs, solution, answer, attempt, prior, errors),
                    max_output_tokens=args.max_output_tokens,
                )
                merge_usage(progress["api_usage"], api_result.usage)
                write_json_atomic(output, document)
            except PhysGraphAPIResponseError as exc:
                merge_usage(progress["api_usage"], exc.result.usage)
                write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
                errors = [str(exc)]
                progress.update(updated_at_utc=utc_now())
                write_json_atomic(progress_path, progress)
                continue
            except (PhysGraphAPIError, OSError, ValueError) as exc:
                write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
                errors = [str(exc)]
                continue
            errors = validate_pass5(document, problem_id, g_obs, solution)
            if not errors:
                write_attempt_log(log, endpoint=api_client.endpoint_label, result=api_result)
                write_json_atomic(output_dir / f"{problem_id}.json", document)
                repair = state.get("problems", {}).get(problem_id, {}).get("segmentation_repair")
                if isinstance(repair, dict):
                    repair.update({
                        "pass5_status": "complete",
                        "pass5_completed_at_utc": utc_now(),
                        "pass5_problem_semantics_sha256": problem_semantics_sha256(problem),
                        "pass5_g_obs_sha256": json_sha256(g_obs),
                        "pass5_document_sha256": json_sha256(document),
                    })
                    write_json_atomic(workspace / "reviews" / "state.json", state)
                succeeded = True
                break
            write_attempt_log(
                log,
                endpoint=api_client.endpoint_label,
                result=api_result,
                error=ValueError("本地校验失败：" + "；".join(errors)),
            )
            prior = document
        if succeeded:
            progress["completed_this_run"] += 1
        else:
            progress["failed_this_run"] += 1
            progress["failures"].append({"problem_id": problem_id, "errors": errors})
        progress.update(current_problem_id="", updated_at_utc=utc_now())
        write_json_atomic(progress_path, progress)
        print(f"[{queue_index}/{len(pending)}] {problem_id}: {'saved' if succeeded else 'failed'}", flush=True)

    progress.update(status="complete", finished_at_utc=utc_now(), updated_at_utc=utc_now())
    write_json_atomic(progress_path, progress)
    run_lock.release()
    atexit.unregister(run_lock.release)
    return 0 if progress["failed_this_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
