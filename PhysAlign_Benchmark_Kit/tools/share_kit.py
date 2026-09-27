"""Seal/verify/package only the declared code, docs and synthetic example.

Uses the standard library. Never includes sibling real data, .env or archives.
Unknown files and symlinks are rejected rather than silently included.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import tempfile
import os
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'KIT_MANIFEST.json'


def files():
    result = {}
    for p in sorted(ROOT.rglob('*')):
        if p.is_symlink():
            raise ValueError('SYMLINK_NOT_ALLOWED: ' + str(p))
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel == MANIFEST or '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        if not (rel in ('README.md', 'VERSION.json') or
                (rel.startswith('docs/') and p.suffix == '.md') or
                (rel.startswith(('tools/', 'tests/')) and p.suffix == '.py') or
                rel == 'tools/native_release_review.html' or
                rel == 'tests/review_dom_smoke.js' or
                (rel.startswith('physalign_converter/') and p.suffix in ('.py', '.json', '.html', '.md', '.txt')) or
                rel.startswith('examples/synthetic_demo/')):
            raise ValueError('UNDECLARED_FILE_NOT_FOR_SHARING: ' + rel)
        if p.name.startswith('.env') or p.suffix.lower() in ('.key', '.pem', '.log'):
            raise ValueError('SECRET_OR_LOG_NOT_FOR_SHARING: ' + rel)
        result[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    marker = ROOT / 'examples/synthetic_demo/SYNTHETIC_ONLY.json'
    if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8'))['synthetic'] is not True:
        raise ValueError('SYNTHETIC_DEMO_MARKER_REQUIRED')
    # The built-in source demo is exactly one artificial test mother. It cannot
    # be replaced with a real workspace while retaining this sharing contract.
    rows = (ROOT / 'examples/synthetic_demo/source/annotation/blind/manifest.jsonl').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(row) for row in rows if row.strip()]
    if len(rows) != 1 or rows[0]['source_dataset'] != 'SYNTHETIC_UNIT_TEST' or rows[0]['problem_id'] != 'SYNTHETIC_TEST_ONLY':
        raise ValueError('REAL_OR_UNKNOWN_DATA_NOT_ALLOWED_IN_BUILTIN_EXAMPLE')
    example = ROOT / 'examples/synthetic_demo/draft'
    declared = json.loads((example / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(example).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in example.rglob('*') if p.is_file() and p.name != 'FILE_MANIFEST.json'}
    if actual != declared:
        raise ValueError('SYNTHETIC_DRAFT_CHANGED_OR_EXTRA_FILE')
    # Reject unrecognized files outside the sealed draft too (e.g. accidentally
    # downloaded real review JSON placed beside the demo).
    outside = {r for r in result if r.startswith('examples/synthetic_demo/') and
               not r.startswith('examples/synthetic_demo/draft/')}
    expected = {'examples/synthetic_demo/SYNTHETIC_ONLY.json',
                'examples/synthetic_demo/source/dataset/images/diagram.png',
                'examples/synthetic_demo/source/annotation/workspace_config.json',
                'examples/synthetic_demo/source/annotation/blind/manifest.jsonl',
                'examples/synthetic_demo/source/annotation/reviews/state.json'}
    expected.update('examples/synthetic_demo/source/annotation/passes/' + s + '/SYNTHETIC_TEST_ONLY.json'
                    for s in ('pass1', 'pass2', 'pass3', 'pass4'))
    if outside != expected:
        raise ValueError('UNDECLARED_DEMO_SOURCE_FILE')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'verify', 'zip', 'refresh'])
    parser.add_argument('--output', help='new ZIP path OUTSIDE the kit; only for zip')
    parser.add_argument('--previous-manifest-sha256', help='maintainer refresh only: exact hash of the previous manifest')
    args = parser.parse_args()
    if (args.action == 'zip') != bool(args.output):
        parser.error('--output is required only for zip')
    if (args.action == 'refresh') != bool(args.previous_manifest_sha256):
        parser.error('--previous-manifest-sha256 is required only for refresh')
    manifest = ROOT / MANIFEST
    current = files()
    if args.action in ('seal', 'refresh'):
        record = {'schema_version': 'physalign_team_kit_v1', 'content_policy': 'code_docs_synthetic_only', 'files': current}
        if args.action == 'seal':
            with manifest.open('x', encoding='utf-8', newline='\n') as f:
                json.dump(record, f, ensure_ascii=False, sort_keys=True, indent=2)
                f.write('\n')
        else:
            expected = args.previous_manifest_sha256
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
                raise ValueError('PREVIOUS_MANIFEST_HASH_MISMATCH')
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                    prefix='.KIT_MANIFEST-', suffix='.tmp', dir=ROOT, delete=False) as f:
                json.dump(record, f, ensure_ascii=False, sort_keys=True, indent=2)
                f.write('\n'); f.flush(); os.fsync(f.fileno())
                temporary = Path(f.name)
            for attempt in range(8):
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
                    raise ValueError('MANIFEST_CHANGED_DURING_REFRESH')
                try:
                    temporary.replace(manifest)
                    break
                except PermissionError as exc:
                    if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                        raise
                    time.sleep(min(.05 * 2**attempt, 1.0))
    else:
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record['files'] != current:
            raise ValueError('KIT_CHANGED: obtain the matching version; do not silently reseal')
    if args.action == 'zip':
        dest = Path(args.output).resolve()
        if dest.is_relative_to(ROOT):
            raise ValueError('ZIP_MUST_BE_OUTSIDE_KIT')
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for rel in sorted([*current, MANIFEST]):
                z.write(ROOT / rel, 'PhysAlign_Benchmark_Kit/' + rel)
        print(json.dumps({'zip': str(dest), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}))
    print(json.dumps({'valid': True, 'action': args.action, 'files': len(current), 'real_data_included': False}))


if __name__ == '__main__':
    main()
