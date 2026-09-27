#!/usr/bin/env python
"""Sequential, resumable API annotation for pending PhysGraph items.

The worker sends only the blind manifest entry, its images, the annotation policy,
and public schemas. It never opens source SFT annotations containing answers.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from physgraph_annotation_lib import (  # noqa: E402
    STAGES,
    canonicalize_pass4_visual_nodes,
    effective_manifest,
    json_sha256,
    problem_semantics_sha256,
    read_json,
    read_jsonl,
    resolve_workspace_path,
    segmentation_repair_status,
    validate_document,
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
    load_gobs_system_instructions,
    merge_usage,
    normalize_base_url,
    safe_endpoint_label,
    write_attempt_log,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--reasoning", default="high")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL; otherwise OPENAI_BASE_URL is used")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-mode", choices=("chat", "responses", "auto"), default="chat")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=65536)
    parser.add_argument("--image-detail", choices=("auto", "low", "high", "original"), default="original")
    parser.add_argument("--limit", type=int, default=0, help="0 means all remaining items")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--only-rejected",
        action="store_true",
        help="Reannotate only problems with at least one rejected stage",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.model:
        if args.dry_run:
            args.model = "dry-run"
        else:
            parser.error("--model or OPENAI_MODEL is required")
    return args


def complete_and_valid(
    workspace: Path,
    problem: dict[str, Any],
    review_state: dict[str, Any] | None = None,
) -> bool:
    if review_state is None:
        state_path = workspace / "reviews" / "state.json"
        review_state = read_json(state_path) if state_path.is_file() else {}
    review_entry = review_state.get("problems", {}).get(problem["problem_id"], {})
    repair_status = segmentation_repair_status(review_entry)
    if repair_status == "downstream_reannotation_required":
        return False
    if repair_status:
        metadata_path = workspace / "passes" / "metadata" / f"{problem['problem_id']}.json"
        if not metadata_path.is_file():
            return False
        try:
            metadata = read_json(metadata_path)
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        if metadata.get("problem_semantics_sha256") != problem_semantics_sha256(problem):
            return False
    previous: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        path = workspace / "passes" / stage / f"{problem['problem_id']}.json"
        if not path.is_file():
            return False
        try:
            document = read_json(path)
        except (OSError, json.JSONDecodeError):
            return False
        errors = [
            issue
            for issue in validate_document(stage, document, problem, previous)
            if issue["level"] == "error"
        ]
        if errors:
            return False
        previous[stage] = document
    return True


def validate_payload(
    payload: Any, problem: dict[str, Any]
) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != set(STAGES):
        return [{
            "level": "error",
            "code": "wrapper_shape",
            "path": "$",
            "message": "顶层必须恰好包含 pass1、pass2、pass3、pass4",
        }]
    issues: list[dict[str, str]] = []
    previous: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        document = payload[stage]
        stage_issues = validate_document(stage, document, problem, previous)
        issues.extend({"stage": stage, **issue} for issue in stage_issues if issue["level"] == "error")
        if isinstance(document, dict):
            previous[stage] = document
    return issues


def save_payload(
    workspace: Path,
    payload: dict[str, Any],
    problem_id: str,
    *,
    reset_review: bool = False,
    problem: dict[str, Any] | None = None,
    output_source: str = "api_generation",
) -> None:
    for stage in STAGES:
        destination = workspace / "passes" / stage / f"{problem_id}.json"
        write_json_atomic(destination, payload[stage])
    if problem is not None:
        write_json_atomic(
            workspace / "passes" / "metadata" / f"{problem_id}.json",
            {
                "schema_version": 1,
                "problem_id": problem_id,
                "problem_sha256": json_sha256(problem),
                "problem_semantics_sha256": problem_semantics_sha256(problem),
                "output_source": output_source,
                "saved_at_utc": utc_now(),
                "stage_sha256": {stage: json_sha256(payload[stage]) for stage in STAGES},
            },
        )
    state_path = workspace / "reviews" / "state.json"
    state = read_json(state_path) if state_path.is_file() else {"problems": {}}
    problem_entry = state.setdefault("problems", {}).setdefault(problem_id, {})
    repair = problem_entry.get("segmentation_repair")
    if reset_review or isinstance(repair, dict):
        now = utc_now()
        stages = problem_entry.setdefault("stages", {})
        stages_to_reset = STAGES if reset_review and not isinstance(repair, dict) else ("pass2", "pass3", "pass4")
        for stage in stages_to_reset:
            stages[stage] = {
                "status": "draft",
                "reviewer": "model-reannotation",
                "note": (
                    "已按修复后的完整题目切分重新生成，等待人工复审"
                    if isinstance(repair, dict)
                    else "已依据人工退回意见重新标注，等待复审"
                ),
                "updated_at_utc": now,
                "document_sha256": json_sha256(payload[stage]),
                "validation_errors": 0,
            }
        if isinstance(repair, dict):
            repair.update({
                "status": (
                    "offline_recovered_pending_review"
                    if output_source.startswith("offline_recovery")
                    else "generated_pending_review"
                ),
                "downstream_saved_at_utc": now,
                "downstream_source": output_source,
                "problem_semantics_sha256": problem_semantics_sha256(problem or {}),
            })
            pass5 = stages.get("pass5")
            if isinstance(pass5, dict):
                pass5.update({
                    "status": "draft",
                    "note": "题目切分及 Pass 2–4 已更新，需要重新执行 Pass 5",
                    "updated_at_utc": now,
                })
        write_json_atomic(state_path, state)


def prompt_for(
    problem: dict[str, Any],
    attempt: int,
    prior: dict[str, Any] | None,
    issues: list[dict[str, str]],
    review_notes: list[dict[str, str]] | None = None,
) -> str:
    blind_record = compact_json(problem)
    repair = ""
    if prior is not None:
        repair = (
            "\n这是校验修复轮。上一轮只含盲化信息的输出如下：\n"
            f"{compact_json(prior)}\n"
            "修复下列schema/引用错误，同时重新检查语义；不得为了通过校验而删除有视觉或文本证据的必要事实：\n"
            f"{compact_json(issues)}\n"
        )
    review_instruction = ""
    if review_notes:
        review_instruction = (
            "\n这是人工退回后的完整重标。必须针对以下退回意见重新检查题图；即使意见为空，也要从Pass 1开始重新标注全部四阶段：\n"
            f"{compact_json(review_notes)}\n"
        )
    return f"""你是 Observed-PhysGraph v0.1 高中物理数据标注器，不是物理解题器。

只处理问题 {problem['problem_id']}，这是第 {attempt} 次生成/修复尝试。
必须严格执行system消息中给出的公开标注规范以及Pass 1、Pass 2、Pass 3、Pass 4 Schema。必须查看本次请求附加的全部题图。
Pass 1视觉定位是本任务的最高优先级：所有bbox_1000、keypoints_1000、center_1000必须依据原始题图尺寸归一化计算。输出前逐个复核边界框确实紧密覆盖对应图元，不能覆盖相邻文字、导线或元件，不能把图元中心误作线段端点；不确定时降低confidence并记录ambiguity，禁止伪精确。

唯一允许使用的题目文本是下面这条盲化记录：
{blind_record}

严禁读取任何原始SFT答案、标准解析或solution文件。选项仅用于识别图片上下文，绝不是已知事实。不得求解，不得从正确选项或物理常识倒推图事实。

连续完成四阶段。最终回复只能是一个合法JSON object，不要Markdown、解释或代码围栏，顶层格式严格为：
{{"pass1":<Pass1完整JSON>,"pass2":<Pass2完整JSON>,"pass3":<Pass3完整JSON>,"pass4":<Pass4完整JSON>}}

每个子对象必须符合system消息中的对应schema。Pass 3自己的bindings必须从b001连续编号；Pass 4合并后的bindings也必须从b001连续编号。relation的subject_id和object_id必须引用physical node，quantity只能放入quantity_id。图像坐标必须按每张实际题图归一化到0-1000。Pass 2不得修改Pass 1；Pass 4必须忠实合并前三阶段、保留直接观察事实并删除所有DERIVED/CONSTRUCTED内容。
{repair}{review_instruction}"""


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    config = read_json(workspace / "workspace_config.json")
    dataset_dir = resolve_workspace_path(workspace, config, "dataset_dir")
    state = read_json(workspace / "reviews" / "state.json")
    manifest = effective_manifest(
        read_jsonl(workspace / config.get("manifest", "blind/manifest.jsonl")),
        state,
    )
    high = [item for item in manifest if item.get("segmentation", {}).get("confidence") == "high"]
    rejected_notes: dict[str, list[dict[str, str]]] = {}
    for item in high:
        problem_id = item["problem_id"]
        stages = state.get("problems", {}).get(problem_id, {}).get("stages", {})
        notes = [
            {"stage": stage, "note": str(stages.get(stage, {}).get("note", ""))}
            for stage in STAGES
            if stages.get(stage, {}).get("status") == "rejected"
        ]
        if notes:
            rejected_notes[problem_id] = notes
    if args.only_rejected:
        pending = [item for item in high if item["problem_id"] in rejected_notes]
    else:
        pending = [item for item in high if not complete_and_valid(workspace, item, state)]
    if args.limit > 0:
        pending = pending[: args.limit]

    run_kind = "reannotation" if args.only_rejected else "batch"
    run_dir = workspace / f"{run_kind}_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = workspace / f"{run_kind}_progress.json"
    progress: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "dry_run" if args.dry_run else "running",
        "target_confidence": "high",
        "high_total": len(high),
        "valid_at_start": len(high) - len([item for item in high if not complete_and_valid(workspace, item, state)]),
        "queued_this_run": len(pending),
        "completed_this_run": 0,
        "failed_this_run": 0,
        "current_problem_id": "",
        "failures": [],
        "run_dir": str(run_dir),
        "provider": {
            "endpoint": safe_endpoint_label(normalize_base_url(args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)),
            "api_mode": args.api_mode,
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "image_detail": args.image_detail,
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
    system_instructions = load_gobs_system_instructions(ROOT, workspace)
    run_lock = ExclusiveRunLock(workspace / ".physgraph_api_annotation.lock")
    try:
        run_lock.acquire()
    except PhysGraphAPIError as exc:
        progress.update(status="lock_error", lock_error=str(exc), updated_at_utc=utc_now())
        write_json_atomic(progress_path, progress)
        print(str(exc), file=sys.stderr)
        return 3
    atexit.register(run_lock.release)

    for index, problem in enumerate(pending, start=1):
        problem_id = problem["problem_id"]
        progress["current_problem_id"] = problem_id
        progress["queue_index"] = index
        progress["updated_at_utc"] = utc_now()
        write_json_atomic(progress_path, progress)
        issues: list[dict[str, str]] = []
        prior: dict[str, Any] | None = None
        succeeded = False
        for attempt in range(1, args.max_attempts + 1):
            output = run_dir / f"{problem_id}.attempt{attempt}.json"
            log = run_dir / f"{problem_id}.attempt{attempt}.api.json"
            postprocessing: dict[str, Any] = {}
            try:
                payload, api_result = api_client.request_json(
                    model=args.model,
                    reasoning_effort=args.reasoning,
                    system_prompt=system_instructions,
                    user_prompt=prompt_for(problem, attempt, prior, issues, rejected_notes.get(problem_id)),
                    image_paths=[dataset_dir / image["path"] for image in problem["images"]],
                    image_detail=args.image_detail,
                    max_output_tokens=args.max_output_tokens,
                )
                merge_usage(progress["api_usage"], api_result.usage)
                # Keep the provider's raw payload as the immutable attempt
                # artifact, then validate/save a separately normalized copy.
                write_json_atomic(output, payload)
                changes = canonicalize_pass4_visual_nodes(payload)
                if changes:
                    normalized_output = run_dir / f"{problem_id}.attempt{attempt}.normalized.json"
                    write_json_atomic(normalized_output, payload)
                    postprocessing = {
                        "postprocessing": {
                            "rule": "pass4_visual_nodes_exact_pass1_subset",
                            "changes": changes,
                            "normalized_output": normalized_output.name,
                        }
                    }
            except PhysGraphAPIResponseError as exc:
                merge_usage(progress["api_usage"], exc.result.usage)
                write_attempt_log(log, endpoint=api_client.endpoint_label, result=exc.result, error=exc)
                issues = [{
                    "level": "error",
                    "code": "api_response_json",
                    "path": "$",
                    "message": str(exc),
                }]
                progress["updated_at_utc"] = utc_now()
                write_json_atomic(progress_path, progress)
                continue
            except (PhysGraphAPIError, OSError, ValueError) as exc:
                write_attempt_log(log, endpoint=api_client.endpoint_label, error=exc)
                issues = [{
                    "level": "error",
                    "code": "api_request",
                    "path": "$",
                    "message": str(exc),
                }]
                continue
            issues = validate_payload(payload, problem)
            if not issues:
                write_attempt_log(
                    log,
                    endpoint=api_client.endpoint_label,
                    result=api_result,
                    extra=postprocessing,
                )
                save_payload(
                    workspace,
                    payload,
                    problem_id,
                    reset_review=args.only_rejected,
                    problem=problem,
                )
                succeeded = True
                break
            write_attempt_log(
                log,
                endpoint=api_client.endpoint_label,
                result=api_result,
                error=ValueError("本地校验失败：" + "；".join(item["message"] for item in issues)),
                extra=postprocessing,
            )
            prior = payload

        if succeeded:
            progress["completed_this_run"] += 1
        else:
            progress["failed_this_run"] += 1
            progress["failures"].append({"problem_id": problem_id, "issues": issues})
        progress["updated_at_utc"] = utc_now()
        progress["current_problem_id"] = ""
        write_json_atomic(progress_path, progress)
        print(
            f"[{index}/{len(pending)}] {problem_id}: "
            f"{'saved' if succeeded else 'failed'}",
            flush=True,
        )

    progress["status"] = "complete"
    progress["finished_at_utc"] = utc_now()
    progress["updated_at_utc"] = utc_now()
    write_json_atomic(progress_path, progress)
    run_lock.release()
    atexit.unregister(run_lock.release)
    return 0 if progress["failed_this_run"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
