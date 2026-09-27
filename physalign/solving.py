"""Original-answer grading contracts and Eq. (8), without inferred answer keys."""

from .dataset import require
from .scoring import OCRRule, QuantityRule, _parse_response


def validate_answer_key(key):
    require(isinstance(key, dict) and key.get('reliable') is True and bool(key.get('source_reference')), 'Original answer key needs reliable=true and a reference source')
    kind = key.get('kind')
    require(kind in {'choice', 'choice_set', 'exact', 'quantity', 'human'}, 'Unknown original-answer grading contract')
    if kind in {'choice', 'exact'}:
        require(isinstance(key.get('accepted'), list) and key['accepted'] and all(isinstance(a, str) and a for a in key['accepted']), 'Declare accepted final-answer strings')
    elif kind == 'choice_set':
        require(isinstance(key.get('accepted'), list) and key['accepted'] and all(isinstance(a, str) and a for a in key['accepted']), 'Declare exact choice set')
    elif kind == 'quantity':
        rule = QuantityRule(key['unit_scales'], key['absolute_tolerance'])
        require(rule.normalize(key['expected']) is not None, 'Invalid original quantity key')
    else:
        require(isinstance(key.get('reference_answer'), str) and bool(key['reference_answer'].strip()), 'Human grading requires a frozen reference answer/rubric')


def grade_original(response, key):
    """None is ungraded human work; malformed served automatic answers score 0."""
    validate_answer_key(key)
    if key['kind'] == 'human':
        return None
    parsed = _parse_response(response)
    value = parsed.get('answer') if parsed is not None else None
    kind = key['kind']
    if kind in {'choice', 'exact'}:
        return int(isinstance(value, str) and value.strip() in key['accepted'])
    if kind == 'choice_set':
        return int(isinstance(value, list) and all(isinstance(v, str) for v in value)
                   and {v.strip() for v in value} == set(key['accepted']))
    rule = QuantityRule(key['unit_scales'], key['absolute_tolerance'])
    observed = rule.normalize(value)
    return int(observed is not None and rule.equivalent(observed, rule.normalize(key['expected'])))
