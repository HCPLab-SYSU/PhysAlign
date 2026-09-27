"""Freeze, preflight, run/resume and score the four new open models."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.adapters import ModelRequest
from physalign.dataset import require
from physalign.planning import code_hashes, load_plan, verify_public_plan
from physalign.reporting import score_run
from physalign.runner import collect_records, run_evaluation, validate_manifest
from physalign.storage import atomic_json, canonical, file_hash, fingerprint, read_json, run_lock
from physalign.study import Study, run_study
from physalign.study_reporting import score_study
from server_eval.new_open_models_adapter import (
    SPECS,
    NewOpenModelAdapter,
    encode_request,
    freeze_snapshot,
    load_processor,
    native_context,
    spec_for,
    validate_config,
    verified_model_root,
)
from server_eval.gpu_resources import wait_for_gpus


DEFAULTS = {
    "gemma4-26b-a4b": {
        "config": "configs/server/gemma4-26b-a4b.json",
        "output": "runs/local-models/gemma4-26b-a4b-nonthinking",
        "preflight": "plans/gemma4-26b-a4b-nonthinking-v1.preflight.json",
        "snapshot": "models/gemma4-26b-a4b.snapshot.json",
    },
    "glm46v-flash": {
        "config": "configs/server/glm46v-flash.json",
        "output": "runs/local-models/glm46v-flash-nonthinking",
        "preflight": "plans/glm46v-flash-nonthinking-v1.preflight.json",
        "snapshot": "models/glm46v-flash.snapshot.json",
    },
    "molmo2-8b": {
        "config": "configs/server/molmo2-8b.json",
        "output": "runs/local-models/molmo2-8b-nonthinking",
        "preflight": "plans/molmo2-8b-nonthinking-v1.preflight.json",
        "snapshot": "models/molmo2-8b.snapshot.json",
    },
    "kimi-vl-a3b": {
        "config": "configs/server/kimi-vl-a3b.json",
        "output": "runs/local-models/kimi-vl-a3b-nonthinking",
        "preflight": "plans/kimi-vl-a3b-nonthinking-v1.preflight.json",
        "snapshot": "models/kimi-vl-a3b.snapshot.json",
    },
}


def log(**row):
    print(canonical(row), flush=True)


def extension_hashes() -> dict:
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ("new_open_models.py", "new_open_models_adapter.py")}


def source(plan_path=None, study_root=None, dataset=None):
    if study_root:
        study = Study(study_root)
        return study.plan, study, "study"
    plan = load_plan(plan_path)
    return plan, verify_public_plan(plan, dataset), "raw_gold"


def request_for(data, row, kind, settings="{}"):
    if kind == "study":
        return data.request(row, "processor-preflight", settings)
    item = data.items[row["instance_id"], row["condition"]]
    messages = item.record["input"]["messages"]
    return ModelRequest(
        "processor-preflight", messages["system"], messages["user"], data.images(item), settings
    )


def identity(plan, config, kind, model):
    spec, snapshot = validate_config(config, model)
    return {
        "schema_version": "physalign_new_open_model_preflight_v1",
        "plan_hash": plan["plan_hash"],
        "kind": kind,
        "model_id": spec.model_id,
        "configuration_hash": fingerprint(config),
        "snapshot_hash": snapshot["snapshot_hash"],
        "snapshot_file_sha256": file_hash(Path(config["snapshot_manifest"])),
        "core_hashes": code_hashes(all_modules=True),
        "extension_hashes": extension_hashes(),
        "visual_profile": spec.visual_profile,
        "system_transport": spec.system_transport,
    }


def preflight(plan, data, kind, config, output, model):
    from PIL import Image

    spec, _, model_root = verified_model_root(config, model)
    stamp = identity(plan, config, kind, spec)
    if kind == "raw_gold":
        data.verify_inventory()
    processor = load_processor(model_root, spec, config)
    context = native_context(model_root)
    rows = []
    for index, row in enumerate(plan["requests"], 1):
        request = request_for(data, row, kind)
        tensors = encode_request(
            processor, Image, spec, request, config.get("processor_call_kwargs")
        )
        length = tensors["input_ids"].shape[-1]
        within = length <= config["max_input_tokens"] and (
            context is None or length + config["max_new_tokens"] <= context
        )
        rows.append({
            "instance_id": row["instance_id"],
            "condition": row["condition"],
            "input_hash": row["input_hash"],
            "images": len(request.images),
            "input_tokens": length,
            "within_budget": within,
            "tensor_shapes": {
                name: list(value.shape) for name, value in tensors.items() if hasattr(value, "shape")
            },
        })
        del tensors, request
        if index == 1 or index % 50 == 0 or index == len(plan["requests"]):
            log(stage="processor_preflight", model=spec.model_id, checked=index,
                total=len(plan["requests"]), input_tokens=length)
    result = {
        **stamp,
        "requests": rows,
        "native_context": context,
        "model_loaded": False,
        "gpu_memory_verified": False,
        "thinking": False,
        "max_input_tokens": config["max_input_tokens"],
        "max_new_tokens": config["max_new_tokens"],
        "all_within_budget": all(row["within_budget"] for row in rows),
    }
    result["report_hash"] = fingerprint(result)
    atomic_json(Path(output), result)
    worst = sorted(rows, key=lambda row: row["input_tokens"], reverse=True)[:10]
    log(stage="preflight_summary", model=spec.model_id, requests=len(rows),
        over_budget=sum(not row["within_budget"] for row in rows),
        maximum_input=max(row["input_tokens"] for row in rows),
        worst_requests=[{
            "instance_id": row["instance_id"], "condition": row["condition"],
            "images": row["images"], "input_tokens": row["input_tokens"],
        } for row in worst], output=str(output))
    require(result["all_within_budget"],
            f"{spec.model_id} input budget exceeded; inspect preflight before inference")
    return result


def checked_preflight(plan, kind, config, path, model):
    result = read_json(Path(path))
    require(result.get("report_hash") == fingerprint({
        key: value for key, value in result.items() if key != "report_hash"
    }), "Preflight report changed")
    require(all(result.get(key) == value for key, value in identity(plan, config, kind, model).items()),
            "Preflight plan/config/snapshot/code changed; rerun preflight")
    expected = [(row["instance_id"], row["condition"], row["input_hash"])
                for row in plan["requests"]]
    observed = [(row["instance_id"], row["condition"], row["input_hash"])
                for row in result["requests"]]
    require(expected == observed and result["all_within_budget"] and
            all(row["within_budget"] for row in result["requests"]),
            "Preflight is incomplete or exceeds the input budget")
    return result


def audit_existing(plan, config, output, model):
    spec = spec_for(model)
    directory = Path(output)
    if not (directory / "manifest.json").exists():
        require(not directory.exists() or not any(path.name != ".lock" for path in directory.iterdir()),
                "Nonempty output without a manifest; choose a new run directory")
        return None, []
    manifest = read_json(directory / "manifest.json")
    validate_manifest(manifest)
    require(manifest["plan"] == plan and manifest["adapter"]["model_id"] == spec.model_id,
            "Existing run plan/model changed")
    require(manifest["adapter_provenance"]["configuration_hash"] == fingerprint(config),
            "Adapter config changed; use a new run directory")
    require(manifest["adapter_provenance"].get("module_hash") ==
            file_hash(Path(__file__).with_name("new_open_models_adapter.py")),
            "New-model adapter source changed; do not mix results across implementations")
    require(manifest["implementation_hashes"] == code_hashes(all_modules=True),
            "Frozen core implementation changed")
    return manifest, collect_records(directory, manifest, allow_incomplete=True)


def run(plan, data, kind, config, output, preflight_path, model, *, gpus, poll_seconds=30):
    spec = spec_for(model)
    checked_preflight(plan, kind, config, preflight_path, spec)
    require(gpus and all(re.fullmatch(r"0|[1-9][0-9]*", value) for value in gpus) and
            len(set(gpus)) == len(gpus), "Pass distinct physical GPU indices")
    require(len(gpus) == config["gpu_count"],
            "--gpus must expose exactly gpu_count devices from the frozen config")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    directory = Path(output).resolve()
    with run_lock(directory.parent / ("." + directory.name + "-controller")):
        manifest, records = audit_existing(plan, config, directory, spec)
        if manifest is not None:
            require(manifest["adapter"]["preprocessing"]["cuda_visible_devices"] == ",".join(gpus),
                    "Restore the original GPU assignment when resuming")
            if records and all(record is not None for record in records):
                log(stage="already_terminal", model=spec.model_id,
                    completed=sum(record["status"] == "completed" for record in records),
                    total=len(records))
                return
        wait_for_gpus(gpus, lambda stage, **row: log(stage=stage, **row), poll_seconds)
        adapter = NewOpenModelAdapter(config, spec)
        log(stage="model_loaded", model=spec.model_id, gpus=gpus, requests=len(plan["requests"]))
        if kind == "study":
            run_study(
                data.root, adapter, directory, resume=manifest is not None,
                configuration_hash=fingerprint(config), progress=lambda row: log(**row),
            )
        else:
            run_evaluation(
                plan, adapter, directory, dataset_root=data.root, resume=manifest is not None,
                adapter_configuration_hash=fingerprint(config), progress=lambda row: log(**row),
            )


def score(plan, data, kind, config, output, model):
    spec = spec_for(model)
    manifest, _ = audit_existing(plan, config, output, spec)
    require(manifest is not None, f"No existing {spec.model_id} run")
    result = (score_study(data.root, output, allow_incomplete=True) if kind == "study" else
              score_run(output, dataset_root=data.root, allow_incomplete=True))
    service = result["service"]
    metrics = (result["experiments"]["main"]["raw_all"]["metrics"] if kind == "study"
               else result["raw_all"]["metrics"])
    log(stage="scored", model=spec.model_id, service=service, metrics=metrics)
    missing = service.get("infrastructure_missing", 0)
    pending = service.get("pending", service.get("pending_requests", []))
    require(not missing and not pending,
            "Report saved with incomplete service; comparison figures require a complete run")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(SPECS))
    parser.add_argument("stage", choices=("freeze", "preflight", "run", "score", "all", "status"))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--plan", help="Existing Raw/Gold plan shared with the previous six models")
    inputs.add_argument("--study", help="Existing complete five-experiment study")
    parser.add_argument("--dataset", help="Relocated unchanged Raw/Gold dataset bundle")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--gpus", help="Comma-separated physical GPU indices")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--model-path", type=Path,
                        help="Local complete unquantized HF model directory for freeze")
    parser.add_argument("--snapshot-output", type=Path)
    args = parser.parse_args(argv)
    defaults = DEFAULTS[args.model]
    args.config = args.config or ROOT / defaults["config"]
    args.output = args.output or ROOT / defaults["output"]
    args.preflight_output = args.preflight_output or ROOT / defaults["preflight"]
    args.snapshot_output = args.snapshot_output or ROOT / defaults["snapshot"]
    try:
        spec = spec_for(args.model)
        if args.stage == "freeze":
            require(args.model_path, "freeze requires --model-path")
            snapshot = freeze_snapshot(args.model_path, spec.key, args.snapshot_output)
            log(stage="snapshot_frozen", model=spec.model_id,
                snapshot_hash=snapshot["snapshot_hash"], path=str(args.snapshot_output))
            return 0
        plan_path = args.plan or (None if args.study else ROOT / "plans/evaluation.json")
        require(plan_path or args.study, "Pass --plan or --study")
        require(1 <= args.poll_seconds <= 60, "Polling interval must be 1..60 seconds")
        config = read_json(args.config)
        validate_config(config, spec)
        if args.stage in ("all", "run"):
            raw_gpus = args.gpus if args.gpus is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
            gpus = raw_gpus.split(",")
            require(gpus and all(re.fullmatch(r"0|[1-9][0-9]*", gpu) for gpu in gpus) and
                    len(set(gpus)) == len(gpus) and len(gpus) == config["gpu_count"],
                    "Pass --gpus with exactly gpu_count distinct physical GPU indices")
        plan, data, kind = source(plan_path, args.study, args.dataset)
        from physalign.storage import outside_bundle
        outside_bundle(args.output, data.root)
        outside_bundle(args.preflight_output, data.root)
        if args.stage == "preflight" or (args.stage == "all" and not args.preflight_output.exists()):
            preflight(plan, data, kind, config, args.preflight_output, spec)
        if args.stage in ("all", "run"):
            run(plan, data, kind, config, args.output, args.preflight_output, spec,
                gpus=gpus, poll_seconds=args.poll_seconds)
        if args.stage in ("all", "score"):
            score(plan, data, kind, config, args.output, spec)
        if args.stage == "status":
            manifest, records = audit_existing(plan, config, args.output, spec)
            log(model=spec.model_id, planned=len(plan["requests"]),
                completed=sum(record is not None and record["status"] == "completed" for record in records),
                missing=sum(record is not None and record["status"] != "completed" for record in records),
                pending=sum(record is None for record in records) if manifest else len(plan["requests"]))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError, KeyError, AttributeError) as error:
        print(f"New open-model evaluation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
