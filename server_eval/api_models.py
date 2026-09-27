"""Run the two API models on the SAME frozen Raw/Gold plan as local models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.adapters import InfrastructureError, ModelRequest
from physalign.dataset import ImageData, require
from physalign.planning import load_plan, verify_public_plan
from physalign.reporting import score_run
from physalign.runner import collect_records, run_evaluation, validate_manifest
from physalign.storage import canonical, digest, fingerprint, loads, read_json, write_new
from server_eval.api_adapter import APIAdapter

MODEL_NAMES = ('gpt-6-astra-high', 'gemini-3.8-flash')


def log(**value):
    print(canonical(value), flush=True)


def connection_check(configs, output):
    """One synthetic two-image request per provider; never a benchmark score."""
    from PIL import Image
    require(not Path(output).exists(), 'Connection-check output exists; use a new filename')
    images = []
    for i, color in enumerate(('red', 'blue'), 1):
        buffer = BytesIO()
        Image.new('RGB', (64, 64), color).save(buffer, format='PNG')
        data = buffer.getvalue()
        images.append(ImageData(f'check_{i}', 'image/png', 64, 64, digest(data), data))
    report = {'schema_version': 'physalign_api_connection_check_v1', 'scientific_run': False,
              'created_at': datetime.now(timezone.utc).isoformat(), 'models': []}
    for name, config in configs.items():
        entry = {'name': name, 'model_id': config['model_id'], 'configuration_hash': fingerprint(config)}
        try:
            log(stage='api_check_started', model=name)
            adapter = APIAdapter(config)
            entry['listed'] = config['model_id'] in adapter.models()
            require(entry['listed'], 'Requested exact model ID absent from provider model list')
            request = ModelRequest('synthetic-connectivity-only', 'Return only a JSON object.',
                                   'Name the solid color of each image in order. Return {"first":"color","second":"color"}.',
                                   tuple(images), canonical(adapter.info.manifest()['settings']))
            response = adapter.generate(request)
            entry.update(status='connected', response=response.record(),
                         requested_settings=adapter.info.manifest()['settings'])
            log(stage='api_connected', model=name, returned_model=response.returned_model_version,
                finish_reason=response.finish_reason, usage=loads(response.usage_json))
        except InfrastructureError as error:
            entry.update(status='failed', error=error.code)
            log(stage='api_check_failed', model=name, error=error.code)
        except (ValueError, RuntimeError, OSError) as error:
            # Messages from configuration/transport validation contain no provider bodies.
            entry.update(status='failed', error=str(error))
            log(stage='api_check_failed', model=name, error=str(error))
        report['models'].append(entry)
    write_new(Path(output), report)
    require(all(r['status'] == 'connected' for r in report['models']), 'API connection check failed; see the saved check report')
    return report


def validate_existing(directory, plan, config):
    manifest = read_json(Path(directory) / 'manifest.json')
    validate_manifest(manifest)
    require(manifest['plan'] == plan, 'Existing API run uses a different plan')
    require(manifest['adapter']['model_id'] == config['model_id'], 'Existing API run uses a different model')
    require(manifest['adapter_provenance']['configuration_hash'] == fingerprint(config), 'API config changed; use a new output directory')
    return manifest


def run_models(plan, configs, output, *, dataset=None):
    verify_public_plan(plan, dataset).verify_inventory()
    # Resolve all credentials/configuration before making the first paid request.
    adapters = {name: APIAdapter(config) for name, config in configs.items()}
    for name, adapter in adapters.items():
        directory = Path(output) / name
        resume = (directory / 'manifest.json').exists()
        if resume:
            manifest = validate_existing(directory, plan, configs[name])
            records = collect_records(directory, manifest, allow_incomplete=True)
            if records and all(r is not None for r in records):
                log(stage='already_terminal', model=name, completed=sum(r['status'] == 'completed' for r in records), total=len(records))
                continue
        log(stage='api_run', model=name, requests=len(plan['requests']), resume=resume)
        run_evaluation(plan, adapter, directory, dataset_root=dataset, resume=resume,
                       adapter_configuration_hash=fingerprint(configs[name]),
                       progress=lambda row: log(model=name, **row))


def score_models(plan, configs, output, *, dataset=None):
    reports = []
    for name, config in configs.items():
        directory = Path(output) / name
        validate_existing(directory, plan, config)
        report = score_run(directory, dataset_root=dataset, allow_incomplete=True)
        reports.append(report)
        log(stage='scored', model=name, service=report['service'], metrics=report['raw_all']['metrics'],
            paired=None if report['paired'] is None else report['paired']['metrics'])
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'run', 'score', 'all', 'status'))
    parser.add_argument('--plan', help='Existing local-model frozen plan, e.g. plans/evaluation.json')
    parser.add_argument('--dataset', help='Relocated unchanged evaluator bundle')
    parser.add_argument('--output', help='API run root, with one directory per model')
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'configs/api')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--check-output', type=Path, help='New JSON file for a synthetic connectivity check')
    args = parser.parse_args(argv)
    try:
        configs = {name: read_json(args.config_dir / (name + '.json')) for name in dict.fromkeys(args.models)}
        if args.stage == 'check':
            output = args.check_output or ROOT / 'tmp' / ('api-connection-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json')
            connection_check(configs, output)
            log(stage='check_saved', path=str(output))
            return 0
        require(args.plan and args.output, '--plan and --output are required')
        plan = load_plan(args.plan)
        if args.stage in ('run', 'all'):
            run_models(plan, configs, args.output, dataset=args.dataset)
        if args.stage in ('score', 'all'):
            reports = score_models(plan, configs, args.output, dataset=args.dataset)
            require(all(not r['service']['pending_requests'] and not r['service']['infrastructure_missing'] for r in reports),
                    'Reports saved with incomplete service; comparison figures require complete runs')
        if args.stage == 'status':
            for name in configs:
                directory = Path(args.output) / name
                if not (directory / 'manifest.json').exists():
                    log(model=name, status='not_started')
                    continue
                manifest = validate_existing(directory, plan, configs[name])
                records = collect_records(directory, manifest, allow_incomplete=True)
                log(model=name, planned=len(records), completed=sum(r is not None and r['status'] == 'completed' for r in records),
                    missing=sum(r is not None and r['status'] != 'completed' for r in records), pending=sum(r is None for r in records))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f'API evaluation: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
