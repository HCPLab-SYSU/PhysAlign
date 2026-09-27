"""Unified command line interface for portable PhysGraph annotation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import ConfigError, PipelineConfig, load_pipeline_config
from .doctor import doctor_report
from .workspace import export_approved_graphs, migrate_workspace, prepare_workspace


from .paths import ROOT, SCRIPTS
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_annotation_lib import read_json, resolve_workspace_path  # noqa: E402
from run_physgraph_annotation_review import run_server  # noqa: E402


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _config(value: str | None) -> PipelineConfig | None:
    return load_pipeline_config(Path(value)) if value else None


def _workspace(args: argparse.Namespace, config: PipelineConfig | None) -> Path:
    if getattr(args, "workspace", None):
        return Path(args.workspace).resolve()
    if config is not None:
        return config.annotation_path("workspace")
    raise ConfigError("必须通过 --config 或 --workspace 显式指定目标工作区")


def _serve(args: argparse.Namespace) -> int:
    config = _config(args.config)
    workspace = _workspace(args, config)
    workspace_config = read_json(workspace / "workspace_config.json")
    raw_prompt = workspace_config.get("prompt_dir")
    if isinstance(raw_prompt, str) and raw_prompt:
        prompt_dir = resolve_workspace_path(workspace, workspace_config, "prompt_dir")
    elif config is not None and isinstance(config.annotation.get("prompt_dir"), str):
        prompt_dir = config.resolve(config.annotation["prompt_dir"], must_exist=True)
    else:
        prompt_dir = ROOT / "physgraph_annotation" / "prompts"
    static_dir = Path(args.static_dir).resolve() if args.static_dir else ROOT / "physgraph_review_app"
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        raise ConfigError("非本机 host 需要显式添加 --allow-remote")
    run_server(
        workspace,
        static_dir,
        prompt_dir,
        args.host,
        args.port,
        args.open_browser,
        allow_remote=args.allow_remote,
    )
    return 0


def _run_annotation(args: argparse.Namespace) -> int:
    config = _config(args.config)
    workspace = _workspace(args, config)
    annotation = config.annotation if config is not None else {}
    model = args.model or annotation.get("model") or os.environ.get("OPENAI_MODEL")
    if not isinstance(model, str) or not model.strip():
        if args.dry_run:
            model = "dry-run"
        else:
            raise ConfigError(
                "未配置模型；请使用 --model、annotation.model 或 OPENAI_MODEL"
            )
    reasoning = (
        args.reasoning
        or annotation.get("reasoning")
        or os.environ.get("OPENAI_REASONING_EFFORT")
        or "medium"
    )
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(SCRIPTS / "run_high_confidence_physgraph_closed_loop.py"),
        "--workspace",
        str(workspace),
        "--model",
        model,
        "--reasoning",
        str(reasoning),
        "--api-mode",
        args.api_mode,
        "--image-detail",
        args.image_detail,
    ]
    if args.base_url:
        command += ["--base-url", args.base_url]
    if args.limit:
        command += ["--limit", str(args.limit)]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _run_pass5(args: argparse.Namespace) -> int:
    config = _config(args.config)
    workspace = _workspace(args, config)
    annotation = config.annotation if config is not None else {}
    model = args.model or annotation.get("model") or os.environ.get("OPENAI_MODEL")
    if not isinstance(model, str) or not model.strip():
        if args.dry_run:
            model = "dry-run"
        else:
            raise ConfigError(
                "未配置模型；请使用 --model、annotation.model 或 OPENAI_MODEL"
            )
    reasoning = (
        args.reasoning
        or annotation.get("reasoning")
        or os.environ.get("OPENAI_REASONING_EFFORT")
        or "medium"
    )
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(SCRIPTS / "run_approved_physgraph_pass5.py"),
        "--workspace",
        str(workspace),
        "--model",
        model,
        "--reasoning",
        str(reasoning),
        "--api-mode",
        args.api_mode,
    ]
    if args.base_url:
        command += ["--base-url", args.base_url]
    if args.limit:
        command += ["--limit", str(args.limit)]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _validate(args: argparse.Namespace) -> int:
    config = _config(args.config)
    workspace = _workspace(args, config)
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(SCRIPTS / "validate_physgraph_annotations.py"),
        "--workspace",
        str(workspace),
    ]
    if args.require_all_approved:
        command.append("--require-all-approved")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _validate_pass5(args: argparse.Namespace) -> int:
    config = _config(args.config)
    workspace = _workspace(args, config)
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(SCRIPTS / "validate_pass5_annotations.py"),
        "--workspace",
        str(workspace),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physgraph-pipeline",
        description="Prepare and review PhysGraph annotations on portable physics datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Read-only environment, path, and media diagnostics")
    doctor.add_argument("--config")
    doctor.add_argument("--workspace", type=Path)

    prepare = subparsers.add_parser("prepare", help="Prepare an answer-blind workspace from a dataset config")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--force", action="store_true", help="Refresh an unchanged source only, preserving Pass and review files")

    serve = subparsers.add_parser("serve", help="Start the local Pass 1-4 annotation/review UI")
    serve.add_argument("--config")
    serve.add_argument("--workspace", type=Path)
    serve.add_argument("--static-dir", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8770)
    serve.add_argument("--open-browser", action="store_true")
    serve.add_argument("--allow-remote", action="store_true")

    run = subparsers.add_parser("run", help="Run the closed-loop Pass 1-4 API annotator")
    run.add_argument("--config")
    run.add_argument("--workspace", type=Path)
    run.add_argument("--model")
    run.add_argument("--reasoning")
    run.add_argument("--base-url")
    run.add_argument("--api-mode", choices=("chat", "responses", "auto"), default="chat")
    run.add_argument("--image-detail", choices=("auto", "low", "high", "original"), default="original")
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--dry-run", action="store_true")

    run_pass5 = subparsers.add_parser(
        "run-pass5", help="Align source solutions after Pass 1-4 approval"
    )
    run_pass5.add_argument("--config")
    run_pass5.add_argument("--workspace", type=Path)
    run_pass5.add_argument("--model")
    run_pass5.add_argument("--reasoning")
    run_pass5.add_argument("--base-url")
    run_pass5.add_argument(
        "--api-mode", choices=("chat", "responses", "auto"), default="chat"
    )
    run_pass5.add_argument("--limit", type=int, default=0)
    run_pass5.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate", help="Run strict Pass 1-4 validation")
    validate.add_argument("--config")
    validate.add_argument("--workspace", type=Path)
    validate.add_argument("--require-all-approved", action="store_true")

    validate_pass5 = subparsers.add_parser(
        "validate-pass5", help="Validate Pass 5 coverage and alignments"
    )
    validate_pass5.add_argument("--config")
    validate_pass5.add_argument("--workspace", type=Path)

    export = subparsers.add_parser("export", help="Export approved G_obs records as JSONL")
    export.add_argument("--config")
    export.add_argument("--workspace", type=Path)
    export.add_argument("--output", type=Path, required=True)

    migrate = subparsers.add_parser("migrate", help="Convert a copied legacy workspace to relative paths")
    migrate.add_argument("--workspace", type=Path, required=True)
    migrate.add_argument("--dataset-dir", type=Path, required=True)
    migrate.add_argument("--source-annotations", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            config = _config(args.config)
            workspace = args.workspace
            if config is None and workspace is None:
                raise ConfigError("doctor 需要 --config 或 --workspace")
            report = doctor_report(config=config, workspace=workspace)
            _json(report)
            return 0 if report["ok"] and report["python"]["supported"] else 1
        if args.command == "prepare":
            _json(prepare_workspace(load_pipeline_config(Path(args.config)), force=args.force))
            return 0
        if args.command == "serve":
            return _serve(args)
        if args.command == "run":
            return _run_annotation(args)
        if args.command == "run-pass5":
            return _run_pass5(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "validate-pass5":
            return _validate_pass5(args)
        if args.command == "export":
            config = _config(args.config)
            _json(export_approved_graphs(_workspace(args, config), args.output))
            return 0
        if args.command == "migrate":
            _json(migrate_workspace(args.workspace, args.dataset_dir, args.source_annotations))
            return 0
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2
