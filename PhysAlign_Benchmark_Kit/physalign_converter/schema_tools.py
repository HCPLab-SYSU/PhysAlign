"""Offline-only schema validation. No remote reference fetching."""
from pathlib import Path
import json
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parent
BASE = 'https://physalign.invalid/v2.1/schemas/'
_SCHEMAS = {p.name: json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'schemas').glob('*.json')}
_REGISTRY = Registry().with_resources((BASE+n, Resource.from_contents(s)) for n,s in _SCHEMAS.items())

def validate_schema(name, value):
    try:
        Draft202012Validator(_SCHEMAS[name], registry=_REGISTRY).validate(value)
    except ValidationError as exc:
        from reference_checks import ContractError
        raise ContractError('BENCHMARK_SCHEMA_INVALID', name + ':' + exc.json_path + ': ' + exc.message) from exc

def schema_bundle():
    return _SCHEMAS
