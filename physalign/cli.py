"""CLI for prepare -> stateless inference -> offline deterministic scoring."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

from .adapters import AdapterInfo, ReplayAdapter, SmokeAdapter, _no_credentials
from .dataset import require
from .planning import load_plan, prepare_plan, save_plan
from .reporting import score_run
from .runner import run_evaluation
from .storage import canonical, fingerprint, read_json


def _adapter(name: str, config: dict):
    require(isinstance(config, dict), "Adapter configuration must be a JSON object")
    _no_credentials(config)
    if name == "smoke":
        require(not config, "Smoke adapter takes no configuration")
        return SmokeAdapter()
    if name == "replay":
        require(set(config) == {"file"}, "Replay config needs exactly one file path")
        return ReplayAdapter(config["file"])
    if name == "hf":
        from .hf_adapter import create_adapter
        return create_adapter(config)
    require(name.count(":") == 1, "Adapter must be smoke, replay, or importable.module:factory")
    module, factory = name.split(":")
    instance = getattr(importlib.import_module(module), factory)(config)
    require(isinstance(instance.info, AdapterInfo) and callable(instance.generate), "Adapter does not implement the model boundary")
    return instance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m physalign")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate exports and freeze a model-independent evaluation plan")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--split", required=True)
    prepare.add_argument("--conditions", nargs="+", choices=("raw", "gold"), default=["raw", "gold"])
    prepare.add_argument("--problem-id", action="append")
    prepare.add_argument("--weights", help="JSON with optional full/paired type-weight dictionaries")
    prepare.add_argument("--bootstrap-resamples", type=int, default=2000)
    prepare.add_argument("--seed", type=int, default=2027)
    prepare.add_argument("--confidence", type=float, default=.95)
    prepare.add_argument("--max-retries", type=int, default=2)
    prepare.add_argument("--reservation-file", action="append", default=[])
    run = commands.add_parser("run", help="Run one public probe per fresh context and record first outcomes")
    run.add_argument("--plan", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--dataset", help="Relocated copy of the unchanged bundle")
    run.add_argument("--adapter", default="smoke")
    run.add_argument("--adapter-config")
    run.add_argument("--resume", action="store_true")
    score = commands.add_parser("score", help="Audit response logs and score offline with private keys")
    score.add_argument("--run", required=True)
    score.add_argument("--dataset")
    score.add_argument("--output")
    score.add_argument("--allow-incomplete", action="store_true")
    from .study_cli import COMMANDS, execute, register
    register(commands)
    args = parser.parse_args(argv)
    try:
        if args.command in COMMANDS:
            return execute(args)
        if args.command == "prepare":
            plan = prepare_plan(args.dataset, split=args.split, conditions=tuple(args.conditions),
                                problem_ids=args.problem_id, type_weights=read_json(Path(args.weights)) if args.weights else None,
                                bootstrap_resamples=args.bootstrap_resamples, bootstrap_seed=args.seed,
                                confidence=args.confidence, max_retries=args.max_retries, reservation_files=args.reservation_file)
            save_plan(plan, args.plan)
            print(canonical({"plan": str(Path(args.plan).resolve()), "plan_hash": plan["plan_hash"],
                             "raw_probes": len(plan["probes"]), "paired_probes": len(plan["paired_ids"]),
                             "scheduled_requests": len(plan["requests"])}))
        elif args.command == "run":
            plan = load_plan(args.plan)
            config = read_json(Path(args.adapter_config)) if args.adapter_config else {}
            adapter = _adapter(args.adapter, config)
            result = run_evaluation(plan, adapter, args.output, dataset_root=args.dataset, resume=args.resume,
                                    adapter_configuration_hash=fingerprint(config),
                                    progress=lambda update: print(canonical(update), flush=True))
            print(canonical(result))
        else:
            result = score_run(args.run, dataset_root=args.dataset, output=args.output, allow_incomplete=args.allow_incomplete)
            print(canonical({"scientific_run": result["scientific_run"], "data_split": result["data_split"],
                             "raw_all": result["raw_all"]["metrics"],
                             "paired": None if result["paired"] is None else result["paired"]["metrics"],
                             "service": result["service"]}))
    except (ValueError, OSError, RuntimeError, ImportError, AttributeError) as exc:
        print(f"PhysAlign: {exc}", file=sys.stderr)
        return 2
    return 0
