"""Local, strict JSON and confined-path I/O. Never used to rewrite source data."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from reference_checks import require, canonical_bytes


def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def _pairs(items):
    obj = {}
    for k, v in items:
        require(k not in obj, 'DUPLICATE_JSON_KEY', k)
        obj[k] = v
    return obj


def parse_json(blob):
    def bad(x):
        raise ValueError('NONFINITE_JSON: ' + x)
    return json.loads(blob, object_pairs_hook=_pairs, parse_constant=bad)


def read_json(path):
    return parse_json(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # All converter outputs are written inside a new staging directory.
    path.write_bytes(canonical_bytes(value) + b'\n')


def safe_path(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative and '\\' not in relative,
            'RELATIVE_POSIX_PATH_REQUIRED', str(relative))
    rel = Path(relative)
    require(not rel.is_absolute() and ':' not in relative and '..' not in rel.parts,
            'UNSAFE_RELATIVE_PATH', relative)
    path = (root / rel).resolve()
    require(path.is_relative_to(root) and path != root, 'PATH_ESCAPES_ROOT', relative)
    return path


def safe_id(value):
    require(isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value)),
            'UNSAFE_SOURCE_ID', str(value))
    return value


def empty_ledger():
    return {'schema_version': 'physalign_review_ledger_v2_1', 'example_only': False, 'records': {}}


def review_record(rid, kind, root, reviewer, evidence):
    require(isinstance(reviewer, str) and reviewer.strip(), 'REVIEWER_REQUIRED')
    return {'review_id': rid, 'kind': kind, 'content_root': root, 'status': 'approved',
            'reviewer': reviewer, 'evidence_refs': evidence}


def file_manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha_bytes(p.read_bytes())
            for p in sorted(root.rglob('*')) if p.is_file() and p != root / 'FILE_MANIFEST.json'}


def seal_directory(root):
    write_json(Path(root) / 'FILE_MANIFEST.json', file_manifest(root))


def verify_directory(root):
    require(read_json(Path(root) / 'FILE_MANIFEST.json') == file_manifest(root),
            'DIRECTORY_CONTENT_CHANGED')
