#!/usr/bin/env python
"""Run the local answer-blind PhysGraph annotation and review workbench."""

from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import threading
import webbrowser
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from physgraph_annotation_lib import (
    STAGE_LABELS,
    STAGES,
    WORKSPACE_SCHEMA_VERSION,
    blank_document,
    effective_problem,
    json_sha256,
    problem_semantics_sha256,
    read_json,
    read_jsonl,
    resolve_workspace_path,
    validate_document,
    validation_summary,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_DIR = ROOT / "physgraph_review_app"
DEFAULT_PROMPT_DIR = ROOT / "physgraph_annotation" / "prompts"
ALLOWED_ACTIONS = {"save", "approve", "reject"}
ALLOWED_STATUSES = {"empty", "draft", "approved", "rejected"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument(
        "--reconcile-approved-repair-metadata",
        action="store_true",
        help=(
            "Validate every resolved segmentation repair whose Pass 1--4 are approved, "
            "then refresh only missing/stale pass metadata and exit"
        ),
    )
    return parser.parse_args()


class AnnotationState:
    def __init__(self, workspace: Path, static_dir: Path, prompt_dir: Path) -> None:
        self.workspace = workspace.resolve()
        self.static_dir = static_dir.resolve()
        self.prompt_dir = prompt_dir.resolve()
        config_path = self.workspace / "workspace_config.json"
        if not config_path.is_file():
            raise RuntimeError("Annotation workspace is missing; run prepare_physgraph_annotation.py first")
        self.config = read_json(config_path)
        self.dataset_dir = resolve_workspace_path(
            self.workspace, self.config, "dataset_dir"
        )
        self.manifest_path = self.workspace / self.config.get("manifest", "blind/manifest.jsonl")
        self.problems = read_jsonl(self.manifest_path)
        if not self.problems:
            raise RuntimeError("Blind manifest is empty")
        self.by_id = {item["problem_id"]: item for item in self.problems}
        if len(self.by_id) != len(self.problems):
            raise RuntimeError("Duplicate problem ids in blind manifest")
        self.media_paths = {
            str(image.get("path", "")).replace("\\", "/").lstrip("/")
            for problem in self.problems
            for image in problem.get("images", [])
            if isinstance(image, dict) and image.get("path")
        }
        self.state_path = self.workspace / "reviews" / "state.json"
        self.history_path = self.workspace / "reviews" / "history.jsonl"
        self.lock = threading.RLock()
        self.review_state = read_json(self.state_path)
        if self.review_state.get("workspace_schema_version") != WORKSPACE_SCHEMA_VERSION:
            raise RuntimeError("Unsupported review-state schema")
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict[str, str]:
        required = ["system", *STAGES]
        prompts: dict[str, str] = {}
        for name in required:
            path = self.prompt_dir / f"{name}.txt"
            if not path.is_file():
                raise RuntimeError(f"Missing prompt file: {path}")
            prompts[name] = path.read_text(encoding="utf-8").strip()
        return prompts

    def _problem_state(self, problem_id: str) -> dict[str, Any]:
        problems = self.review_state.setdefault("problems", {})
        entry = problems.setdefault(problem_id, {"stages": {}})
        entry.setdefault("stages", {})
        return entry

    def _stage_state(self, problem_id: str, stage: str) -> dict[str, Any]:
        entry = self._problem_state(problem_id)
        return entry["stages"].setdefault(stage, {"status": "empty"})

    def _pass_path(self, problem_id: str, stage: str) -> Path:
        return self.workspace / "passes" / stage / f"{problem_id}.json"

    def _load_pass(self, problem_id: str, stage: str) -> tuple[dict[str, Any], bool]:
        path = self._pass_path(problem_id, stage)
        if path.is_file():
            return read_json(path), True
        return blank_document(stage, problem_id), False

    def _problem(self, problem_id: str) -> dict[str, Any]:
        base = self.by_id.get(problem_id)
        if base is None:
            raise ValueError(f"Unknown problem_id: {problem_id}")
        return effective_problem(base, self.review_state.get("problems", {}).get(problem_id, {}))

    def _previous(self, problem_id: str) -> dict[str, dict[str, Any]]:
        previous: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            document, exists = self._load_pass(problem_id, stage)
            if exists:
                previous[stage] = document
        return previous

    def _validation(self, problem_id: str, stage: str, document: dict[str, Any]) -> dict[str, Any]:
        return validation_summary(
            validate_document(stage, document, self._problem(problem_id), self._previous(problem_id))
        )

    @staticmethod
    def _visual_identity(document: dict[str, Any] | None) -> list[tuple[Any, ...]] | None:
        if not isinstance(document, dict) or not isinstance(document.get("visual_nodes"), list):
            return None
        immutable = ("id", "type", "subtype", "text", "image_id")
        return [tuple(node.get(field) for field in immutable) for node in document["visual_nodes"] if isinstance(node, dict)]

    def _append_history(self, record: dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    def _write_state(self) -> None:
        write_json_atomic(self.state_path, self.review_state)

    def _validated_pass_metadata(
        self,
        problem_id: str,
        *,
        output_source: str,
        saved_at_utc: str,
    ) -> dict[str, Any]:
        """Build currentness metadata only after a strict Pass 1--4 chain check."""

        problem = self._problem(problem_id)
        previous: dict[str, dict[str, Any]] = {}
        documents: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            document, exists = self._load_pass(problem_id, stage)
            if not exists:
                raise ValueError(f"{problem_id}: missing {stage} document")
            errors = [
                issue
                for issue in validate_document(stage, document, problem, previous)
                if issue["level"] == "error"
            ]
            if errors:
                codes = ", ".join(
                    f"{issue.get('code', 'invalid')}@{issue.get('path', '$')}"
                    for issue in errors[:8]
                )
                raise ValueError(f"{problem_id}: {stage} is not strictly valid ({codes})")
            documents[stage] = document
            previous[stage] = document

        repair = self._problem_state(problem_id).get("segmentation_repair")
        after_segments_sha256 = repair.get("after_segments_sha256") if isinstance(repair, dict) else None
        if after_segments_sha256 and after_segments_sha256 != json_sha256(problem.get("segments")):
            raise ValueError(f"{problem_id}: repaired segments no longer match the audited repair boundary")

        return {
            "schema_version": 1,
            "problem_id": problem_id,
            "problem_sha256": json_sha256(problem),
            "problem_semantics_sha256": problem_semantics_sha256(problem),
            "output_source": output_source,
            "saved_at_utc": saved_at_utc,
            "stage_sha256": {
                stage: json_sha256(documents[stage])
                for stage in STAGES
            },
        }

    def _refresh_pass_metadata(
        self,
        problem_id: str,
        *,
        output_source: str,
        saved_at_utc: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Refresh a repair's metadata without touching any annotation document."""

        metadata = self._validated_pass_metadata(
            problem_id,
            output_source=output_source,
            saved_at_utc=saved_at_utc,
        )
        path = self.workspace / "passes" / "metadata" / f"{problem_id}.json"
        before = read_json(path) if path.is_file() else None
        current = (
            isinstance(before, dict)
            and before.get("problem_semantics_sha256") == metadata["problem_semantics_sha256"]
            and before.get("stage_sha256") == metadata["stage_sha256"]
        )
        if not current:
            if isinstance(before, dict):
                backup_stamp = saved_at_utc.replace(":", "").replace("+", "_")
                backup_path = (
                    self.workspace
                    / "passes"
                    / "metadata_backups"
                    / backup_stamp
                    / f"{problem_id}.json"
                )
                if not backup_path.exists():
                    write_json_atomic(backup_path, before)
            write_json_atomic(path, metadata)
        return not current, metadata

    def reconcile_approved_repair_metadata(self) -> dict[str, Any]:
        """Repair metadata drift caused by confirming segmentation after generation."""

        now = datetime.now(timezone.utc).isoformat()
        refreshed: list[str] = []
        already_current: list[str] = []
        skipped: list[dict[str, str]] = []
        for problem_id in self.by_id:
            entry = self.review_state.get("problems", {}).get(problem_id, {})
            repair = entry.get("segmentation_repair")
            if not isinstance(repair, dict) or repair.get("status") != "resolved":
                continue
            if not repair.get("segmentation_confirmed", False):
                skipped.append({"problem_id": problem_id, "reason": "segmentation_not_confirmed"})
                continue
            if not all(
                entry.get("stages", {}).get(stage, {}).get("status") == "approved"
                for stage in STAGES
            ):
                skipped.append({"problem_id": problem_id, "reason": "pass1_to_pass4_not_all_approved"})
                continue
            try:
                changed, metadata = self._refresh_pass_metadata(
                    problem_id,
                    output_source="review_reconciled_after_segmentation_confirmation",
                    saved_at_utc=now,
                )
            except ValueError as error:
                skipped.append({"problem_id": problem_id, "reason": str(error)})
                continue
            target = refreshed if changed else already_current
            target.append(problem_id)
            if changed:
                self._append_history({
                    "timestamp_utc": now,
                    "problem_id": problem_id,
                    "event": "pass_metadata_reconciled",
                    "reason": "resolved_segmentation_repair_with_strictly_valid_approved_passes",
                    "problem_semantics_sha256": metadata["problem_semantics_sha256"],
                    "stage_sha256": metadata["stage_sha256"],
                })
        return {
            "refreshed": refreshed,
            "already_current": already_current,
            "skipped": skipped,
        }

    @staticmethod
    def _format_options(options: Any) -> str:
        if not isinstance(options, list) or not options:
            return "（无选项）"
        return "\n".join(f"{item.get('label', '')}. {item.get('text', '')}" for item in options)

    def _render_stage_prompt(self, problem: dict[str, Any], stage: str, documents: dict[str, Any]) -> str:
        segments = problem["segments"]
        replacements = {
            "{problem_id}": problem["problem_id"],
            "{stem}": segments.get("stem", "") or "（空）",
            "{query}": segments.get("query", "") or "（空）",
            "{options}": self._format_options(segments.get("options", [])),
            "{pass1_json}": json.dumps(documents["pass1"], ensure_ascii=False, indent=2),
            "{pass2_json}": json.dumps(documents["pass2"], ensure_ascii=False, indent=2),
            "{pass3_json}": json.dumps(documents["pass3"], ensure_ascii=False, indent=2),
            "{pass2_physical_nodes}": json.dumps(documents["pass2"].get("physical_nodes", []), ensure_ascii=False, indent=2),
        }
        prompt = self.prompts[stage]
        for token, value in replacements.items():
            prompt = prompt.replace(token, value)
        return prompt

    def catalog(self) -> dict[str, Any]:
        with self.lock:
            items: list[dict[str, Any]] = []
            stage_counts = {stage: {status: 0 for status in ALLOWED_STATUSES} for stage in STAGES}
            validation_counts = {"valid": 0, "invalid": 0, "not_started": 0}
            segmentation_counts: dict[str, int] = {}
            repair_counts: dict[str, int] = {}
            approved_g_obs = 0
            for base in self.problems:
                problem_id = base["problem_id"]
                problem = self._problem(problem_id)
                statuses: dict[str, str] = {}
                validation_errors = 0
                active_stage = "pass1"
                for stage in STAGES:
                    document, exists = self._load_pass(problem_id, stage)
                    saved = self.review_state.get("problems", {}).get(problem_id, {}).get("stages", {}).get(stage, {})
                    status = saved.get("status", "draft" if exists else "empty")
                    if status not in ALLOWED_STATUSES:
                        status = "draft" if exists else "empty"
                    statuses[stage] = status
                    stage_counts[stage][status] += 1
                    if exists:
                        if "validation_errors" in saved and saved.get("document_sha256") == json_sha256(document):
                            validation_errors += int(saved["validation_errors"])
                        else:
                            validation_errors += self._validation(problem_id, stage, document)["errors"]
                for stage in STAGES:
                    if statuses[stage] != "approved":
                        active_stage = stage
                        break
                else:
                    active_stage = "pass4"
                if statuses["pass4"] == "approved":
                    approved_g_obs += 1
                if all(status == "empty" for status in statuses.values()):
                    validation_counts["not_started"] += 1
                elif validation_errors:
                    validation_counts["invalid"] += 1
                else:
                    validation_counts["valid"] += 1
                seg_status = problem["segmentation"].get("review_status", "unknown")
                segmentation_counts[seg_status] = segmentation_counts.get(seg_status, 0) + 1
                repair = self.review_state.get("problems", {}).get(problem_id, {}).get("segmentation_repair")
                repair_status = repair.get("status", "none") if isinstance(repair, dict) else "none"
                repair_counts[repair_status] = repair_counts.get(repair_status, 0) + 1
                items.append(
                    {
                        "problem_id": problem_id,
                        "source_dataset": problem["source_dataset"],
                        "source_sample_id": problem["source_sample_id"],
                        "language": problem["language"],
                        "image_count": len(problem["images"]),
                        "preview": (problem["segments"].get("stem") or problem["segments"].get("query") or problem["raw_question"])[:220],
                        "segmentation": problem["segmentation"],
                        "segmentation_repair": deepcopy(repair) if isinstance(repair, dict) else None,
                        "statuses": statuses,
                        "active_stage": active_stage,
                        "validation_errors": validation_errors,
                        "g_obs_approved": statuses["pass4"] == "approved",
                    }
                )
            return {
                "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
                "items": items,
                "stats": {
                    "problems": len(items),
                    "images": sum(item["image_count"] for item in items),
                    "stage_counts": stage_counts,
                    "validation": validation_counts,
                    "g_obs_approved": approved_g_obs,
                    "segmentation": segmentation_counts,
                    "segmentation_repairs": repair_counts,
                },
                "policy": {
                    "answer_solution_visible": False,
                    "pass5_enabled": False,
                    "model_policy": self.config.get(
                        "model_policy",
                        {
                            "passes_1_to_4": "configured multimodal model",
                            "answers_visible": False,
                        },
                    ),
                },
            }

    def problem_payload(self, problem_id: str) -> dict[str, Any]:
        with self.lock:
            problem = self._problem(problem_id)
            documents: dict[str, dict[str, Any]] = {}
            stages: dict[str, Any] = {}
            for stage in STAGES:
                document, exists = self._load_pass(problem_id, stage)
                documents[stage] = document
                state = self.review_state.get("problems", {}).get(problem_id, {}).get("stages", {}).get(stage, {})
                validation = self._validation(problem_id, stage, document) if exists else {"valid": False, "errors": 0, "warnings": 0, "issues": []}
                stages[stage] = {
                    "label": STAGE_LABELS[stage],
                    "exists": exists,
                    "status": state.get("status", "draft" if exists else "empty"),
                    "reviewer": state.get("reviewer", ""),
                    "note": state.get("note", ""),
                    "updated_at_utc": state.get("updated_at_utc", ""),
                    "document": document,
                    "document_sha256": json_sha256(document),
                    "validation": validation,
                }
            prompts = {
                stage: {
                    "system": self.prompts["system"],
                    "task": self._render_stage_prompt(problem, stage, documents),
                    "schema_path": f"schemas/{stage}.schema.json",
                }
                for stage in STAGES
            }
            return {
                "problem": problem,
                "segmentation_repair": deepcopy(
                    self.review_state.get("problems", {}).get(problem_id, {}).get("segmentation_repair")
                ),
                "stages": stages,
                "prompts": prompts,
                "gate": {
                    "g_obs_approved": stages["pass4"]["status"] == "approved",
                    "pass5_enabled": False,
                    "message": "Pass 5 在 G_obs 人工确认后单独启用；当前服务永不读取答案或标准解析。",
                },
            }

    def save_segmentation(self, problem_id: str, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._problem(problem_id)
            segments = body.get("segments")
            reviewer = str(body.get("reviewer", "")).strip() or "local-reviewer"
            if not isinstance(segments, dict) or set(segments) != {"stem", "query", "options"}:
                raise ValueError("segments 必须且只能包含 stem/query/options")
            if not isinstance(segments["stem"], str) or not isinstance(segments["query"], str):
                raise ValueError("stem/query 必须是字符串")
            if not segments["stem"].strip() and not segments["query"].strip():
                raise ValueError("stem/query 不能同时为空")
            if not isinstance(segments["options"], list):
                raise ValueError("options 必须是数组")
            normalized_options: list[dict[str, str]] = []
            for option in segments["options"]:
                if not isinstance(option, dict) or set(option) != {"label", "text"}:
                    raise ValueError("每个 option 必须且只能包含 label/text")
                if not isinstance(option["label"], str) or not isinstance(option["text"], str):
                    raise ValueError("option label/text 必须是字符串")
                normalized_options.append({"label": option["label"].strip(), "text": option["text"].strip()})
            normalized = {
                "stem": segments["stem"].strip(),
                "query": segments["query"].strip(),
                "options": normalized_options,
            }
            now = datetime.now(timezone.utc).isoformat()
            entry = self._problem_state(problem_id)
            before = self._problem(problem_id)["segments"]
            repair = entry.get("segmentation_repair")
            entry["segmentation_override"] = {
                "segments": normalized,
                "segmentation": {
                    "method": "manual_after_segmentation_repair" if isinstance(repair, dict) else "manual",
                    "confidence": "high",
                    "review_status": "approved",
                    "edited": True,
                    "reviewer": reviewer,
                    "updated_at_utc": now,
                    **({"repair_id": repair.get("repair_id", "")} if isinstance(repair, dict) else {}),
                },
            }
            if before != normalized:
                for stage in ("pass2", "pass3", "pass4"):
                    stage_state = entry["stages"].get(stage)
                    if stage_state:
                        stage_state["status"] = "draft"
                        stage_state["note"] = "题目切分已修改；Pass 1 保留，Pass 2–4 必须重做或重新确认"
                        stage_state["updated_at_utc"] = now
                        stage_state["stale_reason"] = "manual_segmentation_change"
                pass5_state = entry["stages"].get("pass5")
                if isinstance(pass5_state, dict):
                    pass5_state.update({
                        "status": "draft",
                        "note": "题目切分已修改，需要在 Pass 2–4 复审后重做 Pass 5",
                        "updated_at_utc": now,
                        "stale_reason": "manual_segmentation_change",
                    })
                if isinstance(repair, dict):
                    repair.update({
                        "status": "downstream_reannotation_required",
                        "segmentation_confirmed": True,
                        "segmentation_confirmed_at_utc": now,
                        "segmentation_confirmed_by": reviewer,
                        "after_segments_sha256": json_sha256(normalized),
                    })
            elif isinstance(repair, dict):
                repair.update({
                    "segmentation_confirmed": True,
                    "segmentation_confirmed_at_utc": now,
                    "segmentation_confirmed_by": reviewer,
                })
            self._write_state()
            self._append_history(
                {
                    "timestamp_utc": now,
                    "problem_id": problem_id,
                    "event": "segmentation_saved",
                    "reviewer": reviewer,
                    "before_sha256": json_sha256(before),
                    "after_sha256": json_sha256(normalized),
                }
            )
            return self.problem_payload(problem_id)

    def save_stage(self, problem_id: str, stage: str, body: dict[str, Any]) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        action = str(body.get("action", "save"))
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        reviewer = str(body.get("reviewer", "")).strip() or "local-reviewer"
        note = str(body.get("note", "")).strip()
        document = body.get("document")
        if not isinstance(document, dict):
            raise ValueError("document 必须是 JSON object")

        with self.lock:
            self._problem(problem_id)
            validation = self._validation(problem_id, stage, document)
            path = self._pass_path(problem_id, stage)
            before = read_json(path) if path.is_file() else None
            expected_sha256 = str(body.get("expected_sha256", "")).strip()
            if expected_sha256 and before is not None and expected_sha256 != json_sha256(before):
                raise ValueError("该阶段已被其他操作更新，请重新载入后再保存，避免覆盖较新的标注")
            if action == "approve":
                repair = self._problem_state(problem_id).get("segmentation_repair")
                if (
                    stage in {"pass2", "pass3", "pass4"}
                    and isinstance(repair, dict)
                    and not repair.get("segmentation_confirmed", False)
                ):
                    raise ValueError("该题属于切分修复清单；必须先核对并点击“确认题目切分”")
                if stage in {"pass3", "pass4"} and self._problem(problem_id)["segmentation"].get("review_status") == "needs_review":
                    raise ValueError("该题题目切分为低置信度，必须先人工确认题干、问题和选项")
                if not validation["valid"]:
                    raise ValueError(f"存在 {validation['errors']} 个错误，不能批准")
                stage_index = STAGES.index(stage)
                for prerequisite in STAGES[:stage_index]:
                    prior, exists = self._load_pass(problem_id, prerequisite)
                    if not exists:
                        raise ValueError(f"缺少前置结果 {prerequisite}")
                    prior_validation = self._validation(problem_id, prerequisite, prior)
                    if not prior_validation["valid"]:
                        raise ValueError(f"前置结果 {prerequisite} 未通过校验")

            write_json_atomic(path, document)
            now = datetime.now(timezone.utc).isoformat()
            stage_state = self._stage_state(problem_id, stage)
            stage_state.update(
                {
                    "status": {"save": "draft", "approve": "approved", "reject": "rejected"}[action],
                    "reviewer": reviewer,
                    "note": note,
                    "updated_at_utc": now,
                    "document_sha256": json_sha256(document),
                    "validation_errors": validation["errors"],
                }
            )
            current_index = STAGES.index(stage)
            if before != document:
                for downstream in STAGES[current_index + 1 :]:
                    downstream_state = self._problem_state(problem_id)["stages"].get(downstream)
                    if downstream_state and downstream_state.get("status") == "approved":
                        downstream_state["status"] = "draft"
                        downstream_state["note"] = f"上游 {stage} 已修改，需要重新确认"
                        downstream_state["updated_at_utc"] = now
            pass4_synced = False
            pass_metadata_refreshed = False
            if (
                stage == "pass1"
                and before != document
                and validation["valid"]
                and self._visual_identity(before) == self._visual_identity(document)
            ):
                pass4_path = self._pass_path(problem_id, "pass4")
                if pass4_path.is_file():
                    pass4_before = read_json(pass4_path)
                    if pass4_before.get("visual_nodes") != document.get("visual_nodes"):
                        pass4_after = deepcopy(pass4_before)
                        pass4_after["visual_nodes"] = deepcopy(document["visual_nodes"])
                        write_json_atomic(pass4_path, pass4_after)
                        pass4_validation = self._validation(problem_id, "pass4", pass4_after)
                        pass4_state = self._stage_state(problem_id, "pass4")
                        pass4_state.update({
                            "status": "draft",
                            "reviewer": reviewer,
                            "note": "Pass 1几何已人工调整；visual_nodes已自动同步，需要重新确认叠加图与Pass 4",
                            "updated_at_utc": now,
                            "document_sha256": json_sha256(pass4_after),
                            "validation_errors": pass4_validation["errors"],
                        })
                        pass5_state = self._problem_state(problem_id)["stages"].get("pass5")
                        if isinstance(pass5_state, dict) and pass5_state.get("status") == "approved":
                            pass5_state.update({
                                "status": "draft",
                                "note": "Pass 1/Pass 4视觉几何已修改，需要重新执行或复核Pass 5",
                                "updated_at_utc": now,
                            })
                        self._append_history({
                            "timestamp_utc": now,
                            "problem_id": problem_id,
                            "event": "pass1_geometry_synced_to_pass4",
                            "reviewer": reviewer,
                            "before_sha256": json_sha256(pass4_before),
                            "after_sha256": json_sha256(pass4_after),
                            "visual_nodes_sha256": json_sha256(document["visual_nodes"]),
                            "validation_errors": pass4_validation["errors"],
                        })
                        pass4_synced = True
            if action == "approve" and stage == "pass4":
                problem_entry = self._problem_state(problem_id)
                repair = problem_entry.get("segmentation_repair")
                if isinstance(repair, dict) and repair.get("segmentation_confirmed", False):
                    all_approved = all(
                        problem_entry.get("stages", {}).get(candidate, {}).get("status") == "approved"
                        for candidate in STAGES
                    )
                    if all_approved:
                        repair.update({
                            "status": "resolved",
                            "resolved_at_utc": now,
                            "resolved_by": reviewer,
                            "resolution": "segmentation_confirmed_and_pass1_to_pass4_approved",
                        })
                        pass_metadata_refreshed, metadata = self._refresh_pass_metadata(
                            problem_id,
                            output_source="review_approved_after_segmentation_confirmation",
                            saved_at_utc=now,
                        )
                        repair["pass_metadata_problem_semantics_sha256"] = metadata[
                            "problem_semantics_sha256"
                        ]
                        repair["pass_metadata_refreshed_at_utc"] = now
                        if pass_metadata_refreshed:
                            self._append_history({
                                "timestamp_utc": now,
                                "problem_id": problem_id,
                                "event": "pass_metadata_refreshed",
                                "reason": "segmentation_repair_resolved",
                                "problem_semantics_sha256": metadata["problem_semantics_sha256"],
                                "stage_sha256": metadata["stage_sha256"],
                            })
            if stage == "pass1" and before != document:
                cache_path = self.workspace / "pipeline_cache" / "verified_pass1" / f"{problem_id}.json"
                cache_path.unlink(missing_ok=True)
            self._write_state()
            self._append_history(
                {
                    "timestamp_utc": now,
                    "problem_id": problem_id,
                    "event": f"stage_{action}",
                    "stage": stage,
                    "reviewer": reviewer,
                    "note": note,
                    "before_sha256": json_sha256(before) if before is not None else "",
                    "after_sha256": json_sha256(document),
                    "validation_errors": validation["errors"],
                }
            )
            response = self.problem_payload(problem_id)
            response["save_result"] = {
                "pass4_visual_nodes_synced": pass4_synced,
                "pass1_cache_invalidated": stage == "pass1" and before != document,
                "pass_metadata_refreshed": pass_metadata_refreshed,
            }
            return response


class AnnotationServer(ThreadingHTTPServer):
    state: AnnotationState


class Handler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, path: Path, cache: bool = False) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def safe_child(root: Path, relative: str) -> Path:
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("Path escapes configured root")
        return candidate

    def _read_body(self) -> dict[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 8 * 1024 * 1024:
            raise ValueError("Invalid request size")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Request body must be an object")
        return value

    def _check_origin(self) -> None:
        host = self.headers.get("Host", "")
        parsed = urlparse("http://" + host)
        bound_host, bound_port = self.server.server_address[:2]
        allowed_hosts = {bound_host.lower()}
        if bound_host in {"127.0.0.1", "::1"}:
            allowed_hosts.update({"localhost", "127.0.0.1", "::1"})
        # Explicit remote listeners can use their externally configured host.
        if bound_host not in {"0.0.0.0", "::"} and parsed.hostname not in allowed_hosts:
            raise ValueError("Untrusted Host header")
        if parsed.username or parsed.password or parsed.path or (parsed.port or 80) != bound_port:
            raise ValueError("Invalid Host header")
        origin = self.headers.get("Origin")
        if origin and origin.rstrip("/") != "http://" + host:
            raise ValueError("Cross-origin requests are not allowed")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise ValueError("Cross-site requests are not allowed")

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            self._check_origin()
            if path == "/api/health":
                self.send_json({"ok": True, "service": "physgraph-annotation-review", "answer_solution_visible": False})
                return
            if path == "/api/catalog":
                self.send_json(self.server.state.catalog())
                return
            if path.startswith("/api/problem/"):
                problem_id = path.removeprefix("/api/problem/")
                if "/" in problem_id or not problem_id:
                    raise ValueError("Invalid problem id")
                self.send_json(self.server.state.problem_payload(problem_id))
                return
            if path.startswith("/api/schema/"):
                stage = path.removeprefix("/api/schema/")
                if stage not in STAGES:
                    raise ValueError("Unknown stage")
                self.send_file(self.server.state.workspace / "schemas" / f"{stage}.schema.json")
                return
            if path.startswith("/media/"):
                relative = path.removeprefix("/media/").replace("\\", "/").lstrip("/")
                if relative not in self.server.state.media_paths:
                    raise ValueError("Only media paths declared in the blind manifest are served")
                self.send_file(self.safe_child(self.server.state.dataset_dir, relative), cache=True)
                return
            relative = "index.html" if path == "/" else path.lstrip("/")
            self.send_file(self.safe_child(self.server.state.static_dir, relative))
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        try:
            self._check_origin()
            parts = path.strip("/").split("/")
            body = self._read_body()
            if len(parts) == 4 and parts[:2] == ["api", "problem"] and parts[3] == "segmentation":
                self.send_json(self.server.state.save_segmentation(parts[2], body))
                return
            if len(parts) == 5 and parts[:2] == ["api", "problem"] and parts[3] == "stage":
                self.send_json(self.server.state.save_stage(parts[2], parts[4], body))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def run_server(workspace: Path, static_dir: Path, prompt_dir: Path, host: str, port: int,
               open_browser: bool, *, allow_remote: bool = False) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_remote:
        raise ValueError("非本机 host 需要显式添加 --allow-remote")
    state = AnnotationState(workspace, static_dir, prompt_dir)
    server_type = AnnotationServer
    if ":" in host:
        class IPv6AnnotationServer(AnnotationServer):
            address_family = socket.AF_INET6
        server_type = IPv6AnnotationServer
    server = server_type((host, port), Handler)
    server.state = state
    bound_host, bound_port = server.server_address[:2]
    url_host = f"[{bound_host}]" if ":" in bound_host else bound_host
    url = f"http://{url_host}:{bound_port}/"
    print(f"PhysGraph annotation review: {url}", flush=True)
    print("Answer/solution isolation: ON", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping annotation review app.", flush=True)
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    if args.reconcile_approved_repair_metadata:
        state = AnnotationState(args.workspace, args.static_dir, args.prompt_dir)
        print(json.dumps(state.reconcile_approved_repair_metadata(), ensure_ascii=False, indent=2))
        return
    run_server(args.workspace, args.static_dir, args.prompt_dir, args.host, args.port,
               args.open_browser, allow_remote=args.allow_remote)


if __name__ == "__main__":
    main()
