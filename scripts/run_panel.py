"""Launch model processes using a JSON panel with disjoint GPU groups."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.study import Study, verify_final_seal


def load_panel(path):
    panel = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(panel, dict) or set(panel) != {'jobs'} or not isinstance(panel['jobs'], list) or not panel['jobs']:
        raise ValueError('Panel must contain a nonempty jobs list')
    names, allocated = set(), set()
    for job in panel['jobs']:
        if not isinstance(job, dict) or set(job) != {'name', 'gpus', 'config'}:
            raise ValueError('Each job requires name, gpus and config')
        name, devices = job['name'], job['gpus']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in names:
            raise ValueError('Job names must be unique directory-safe identifiers')
        if (not isinstance(devices, list) or not devices or
                any(type(device) is not int or device < 0 for device in devices) or
                len(set(devices)) != len(devices) or allocated.intersection(devices)):
            raise ValueError('Jobs require nonempty, disjoint lists of GPU indices')
        if not isinstance(job['config'], str) or not (ROOT / job['config']).is_file():
            raise ValueError('Model config does not exist: ' + str(job['config']))
        names.add(name)
        allocated.update(devices)
    return panel['jobs']


def jobs(study, output, panel, resume=False):
    for job in panel:
        name, devices = job['name'], ','.join(str(device) for device in job['gpus'])
        run = Path(output) / name
        args = [sys.executable, str(ROOT / 'evaluate.py'), 'run-study', '--study', str(Path(study).resolve()),
                '--output', str(run.resolve()), '--adapter', 'hf', '--adapter-config', str((ROOT / job['config']).resolve())]
        if resume and (run / 'manifest.json').exists():
            args.append('--resume')
        yield name, devices, args


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--panel-config', default=str(ROOT / 'configs/panel.example.json'),
                   help='JSON jobs; relative model config paths are resolved from the project root')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    study = Study(args.study)
    commands = list(jobs(args.study, args.output, load_panel(args.panel_config), args.resume))
    if args.dry_run:
        print(json.dumps([{'model': n, 'CUDA_VISIBLE_DEVICES': d, 'argv': c} for n, d, c in commands], indent=2))
        return 0
    verify_final_seal(study)
    processes = []
    logs = Path(args.output).resolve() / 'launcher_logs'
    logs.mkdir(parents=True, exist_ok=True)
    try:
        for name, devices, command in commands:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            stream = (logs / (name + '.log')).open('ab' if args.resume else 'xb')
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            processes.append((name, process, stream))
            print(f'{name}: PID={process.pid}, GPUs={devices}, log={stream.name}', flush=True)
        failed = [name for name, process, _ in processes if process.wait() != 0]
        if failed:
            print('Failed model processes: ' + ', '.join(failed), file=sys.stderr)
            return 1
        return 0
    finally:
        for _, process, stream in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
