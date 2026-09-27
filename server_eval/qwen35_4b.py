"""Freeze, preflight, run/resume and score Qwen3.5-4B on an existing plan."""
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
from server_eval.qwen35_4b_adapter import MODEL_ID, Qwen35_4BAdapter, backend, freeze_snapshot, validate_snapshot, thinking_enabled
from server_eval.gpu_resources import wait_for_gpus


def log(**row):
    print(canonical(row), flush=True)


def extension_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in ('qwen35_4b.py', 'qwen35_4b_adapter.py')}


def source(plan_path=None, study_root=None, dataset=None):
    if study_root:
        study = Study(study_root)
        return study.plan, study, 'study'
    plan = load_plan(plan_path)
    return plan, verify_public_plan(plan, dataset), 'raw_gold'


def request_for(data, row, kind, settings='{}'):
    if kind == 'study':
        return data.request(row, 'processor-preflight', settings)
    item = data.items[row['instance_id'], row['condition']]
    messages = item.record['input']['messages']
    return ModelRequest('processor-preflight', messages['system'], messages['user'], data.images(item), settings)


def identity(plan, config, kind):
    snapshot = validate_snapshot(config)
    return {'schema_version': 'physalign_qwen35_4b_preflight_v1', 'plan_hash': plan['plan_hash'], 'kind': kind,
            'model_id': MODEL_ID, 'configuration_hash': fingerprint(config),
            'snapshot_hash': snapshot['snapshot_hash'], 'snapshot_file_sha256': file_hash(Path(config['snapshot_manifest'])),
            'core_hashes': code_hashes(all_modules=True), 'extension_hashes': extension_hashes()}


def preflight(plan, data, kind, config, output):
    from PIL import Image
    from transformers import AutoProcessor
    thinking = thinking_enabled(config)
    stamp = identity(plan, config, kind)
    snapshot = validate_snapshot(config)
    model_root = backend.verify_snapshot(snapshot, config.get('model_path'))
    if kind == 'raw_gold':
        data.verify_inventory()
    overrides = config.get('processor_kwargs', {})
    require(isinstance(overrides, dict) and not set(overrides) & {'revision', 'local_files_only', 'trust_remote_code'}, 'Invalid processor overrides')
    processor = AutoProcessor.from_pretrained(str(model_root), local_files_only=True, trust_remote_code=False, **overrides)
    model_config = read_json(model_root / 'config.json')
    native_context = model_config.get('text_config', model_config).get('max_position_embeddings')
    rows = []
    for index, row in enumerate(plan['requests'], 1):
        req = request_for(data, row, kind)
        tensors = backend.encode_request(processor, Image, req, {'enable_thinking': thinking}, config.get('processor_call_kwargs'))
        length = tensors['input_ids'].shape[-1]
        rows.append({'instance_id': row['instance_id'], 'condition': row['condition'], 'input_hash': row['input_hash'],
                     'images': len(req.images), 'input_tokens': length,
                     'within_budget': length <= config['max_input_tokens'] and (native_context is None or length + config['max_new_tokens'] <= native_context),
                     'tensor_shapes': {key: list(value.shape) for key, value in tensors.items() if hasattr(value, 'shape')}})
        del tensors, req
        if index == 1 or index % 50 == 0 or index == len(plan['requests']):
            log(stage='processor_preflight', model=MODEL_ID, checked=index, total=len(plan['requests']), input_tokens=length)
    result = {**stamp, 'requests': rows, 'native_context': native_context, 'model_loaded': False, 'gpu_memory_verified': False,
              'thinking': thinking,
              'max_input_tokens': config['max_input_tokens'], 'max_new_tokens': config['max_new_tokens'],
              'all_within_budget': all(r['within_budget'] for r in rows)}
    result['report_hash'] = fingerprint(result)
    atomic_json(Path(output), result)
    log(stage='preflight_summary', model=MODEL_ID, requests=len(rows), over_budget=sum(not r['within_budget'] for r in rows),
        maximum_input=max(r['input_tokens'] for r in rows), output=str(output))
    require(result['all_within_budget'], 'Qwen4B input budget exceeded; inspect preflight before inference')
    return result


def checked_preflight(plan, kind, config, path):
    result = read_json(Path(path))
    require(result.get('report_hash') == fingerprint({k: v for k, v in result.items() if k != 'report_hash'}), 'Preflight report changed')
    require(all(result.get(k) == v for k, v in identity(plan, config, kind).items()), 'Preflight plan/config/snapshot/code changed; rerun preflight')
    expected = [(r['instance_id'], r['condition'], r['input_hash']) for r in plan['requests']]
    observed = [(r['instance_id'], r['condition'], r['input_hash']) for r in result['requests']]
    require(expected == observed and result['all_within_budget'] and all(r['within_budget'] for r in result['requests']),
            'Preflight is incomplete or exceeds the input budget')
    return result


def audit_existing(plan, config, output):
    directory = Path(output)
    if not (directory / 'manifest.json').exists():
        require(not directory.exists() or not any(p.name != '.lock' for p in directory.iterdir()),
                'Nonempty output without a manifest; choose a new run directory')
        return None, []
    manifest = read_json(directory / 'manifest.json')
    validate_manifest(manifest)
    require(manifest['plan'] == plan and manifest['adapter']['model_id'] == MODEL_ID, 'Existing run plan/model changed')
    require(manifest['adapter_provenance']['configuration_hash'] == fingerprint(config), 'Qwen4B config changed; use a new run directory')
    require(manifest['implementation_hashes'] == code_hashes(all_modules=True), 'Frozen core implementation changed')
    return manifest, collect_records(directory, manifest, allow_incomplete=True)


def run(plan, data, kind, config, output, preflight_path, *, gpus, poll_seconds=30):
    checked_preflight(plan, kind, config, preflight_path)
    require(gpus and all(re.fullmatch(r'0|[1-9][0-9]*', value) for value in gpus) and len(set(gpus)) == len(gpus), 'Pass distinct physical GPU indices')
    require(len(gpus) == config['gpu_count'], '--gpus must expose exactly gpu_count devices from the frozen config')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)
    directory = Path(output).resolve()
    # Separate supervisor lock covers waiting/model loading as well as inference.
    with run_lock(directory.parent / ('.' + directory.name + '-controller')):
        manifest, records = audit_existing(plan, config, directory)
        if manifest is not None:
            require(manifest['adapter']['preprocessing']['cuda_visible_devices'] == ','.join(gpus), 'Restore the original GPU assignment when resuming')
            if records and all(r is not None for r in records):
                log(stage='already_terminal', model=MODEL_ID, completed=sum(r['status'] == 'completed' for r in records), total=len(records))
                return
        wait_for_gpus(gpus, lambda stage, **row: log(stage=stage, **row), poll_seconds)
        adapter = Qwen35_4BAdapter(config)
        log(stage='model_loaded', model=MODEL_ID, gpus=gpus, requests=len(plan['requests']))
        if kind == 'study':
            run_study(data.root, adapter, directory, resume=manifest is not None, configuration_hash=fingerprint(config), progress=lambda row: log(**row))
        else:
            run_evaluation(plan, adapter, directory, dataset_root=data.root, resume=manifest is not None,
                           adapter_configuration_hash=fingerprint(config), progress=lambda row: log(**row))


def score(plan, data, kind, config, output):
    manifest, _ = audit_existing(plan, config, output)
    require(manifest is not None, 'No existing Qwen4B run')
    result = (score_study(data.root, output, allow_incomplete=True) if kind == 'study' else
              score_run(output, dataset_root=data.root, allow_incomplete=True))
    service = result['service']
    log(stage='scored', model=MODEL_ID, service=service,
        metrics=result['experiments']['main']['raw_all']['metrics'] if kind == 'study' else result['raw_all']['metrics'])
    missing = service.get('infrastructure_missing', 0)
    pending = service.get('pending', service.get('pending_requests', []))
    require(not missing and not pending, 'Report saved with incomplete service; comparison figures require a complete run')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'preflight', 'run', 'score', 'all', 'status'))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--plan', help='Existing Raw/Gold plan, shared with the other five models')
    inputs.add_argument('--study', help='Existing complete five-experiment study')
    parser.add_argument('--dataset', help='Relocated unchanged Raw/Gold dataset bundle')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/server/qwen35-4b.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/local-models/qwen35-4b-nonthinking')
    parser.add_argument('--preflight-output', type=Path, default=ROOT / 'plans/qwen35-4b-nonthinking-v1.preflight.json')
    parser.add_argument('--gpus', help='Comma-separated physical GPU indices; default CUDA_VISIBLE_DEVICES')
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--model-path', type=Path, help='Local complete unquantized HF model directory for freeze')
    parser.add_argument('--snapshot-output', type=Path, default=ROOT / 'models/qwen35-4b.snapshot.json')
    args = parser.parse_args(argv)
    try:
        if args.stage == 'freeze':
            require(args.model_path, 'freeze requires --model-path')
            snapshot = freeze_snapshot(args.model_path, args.snapshot_output)
            log(stage='snapshot_frozen', model=MODEL_ID, snapshot_hash=snapshot['snapshot_hash'], path=str(args.snapshot_output))
            return 0
        require(args.plan or args.study, 'Pass --plan or --study')
        require(1 <= args.poll_seconds <= 60, 'Polling interval must be 1..60 seconds')
        config = read_json(args.config)
        if args.stage in ('all', 'run'):
            gpus = (args.gpus if args.gpus is not None else os.environ.get('CUDA_VISIBLE_DEVICES', '')).split(',')
            require(gpus and all(re.fullmatch(r'0|[1-9][0-9]*', g) for g in gpus) and len(set(gpus)) == len(gpus)
                    and len(gpus) == config['gpu_count'], 'Pass --gpus with exactly gpu_count distinct physical GPU indices')
        plan, data, kind = source(args.plan, args.study, args.dataset)
        # All artifacts stay outside the immutable source bundle/study.
        from physalign.storage import outside_bundle
        outside_bundle(args.output, data.root)
        outside_bundle(args.preflight_output, data.root)
        if args.stage == 'preflight' or (args.stage == 'all' and not args.preflight_output.exists()):
            preflight(plan, data, kind, config, args.preflight_output)
        if args.stage in ('all', 'run'):
            run(plan, data, kind, config, args.output, args.preflight_output, gpus=gpus, poll_seconds=args.poll_seconds)
        if args.stage in ('all', 'score'):
            score(plan, data, kind, config, args.output)
        if args.stage == 'status':
            manifest, records = audit_existing(plan, config, args.output)
            log(model=MODEL_ID, planned=len(plan['requests']), completed=sum(r is not None and r['status'] == 'completed' for r in records),
                missing=sum(r is not None and r['status'] != 'completed' for r in records),
                pending=sum(r is None for r in records) if manifest else len(plan['requests']))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as error:
        print(f'Qwen3.5-4B evaluation: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
