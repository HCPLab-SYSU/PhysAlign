"""Private, reproducible diagnostics. These never change source truth or gold.

No arithmetic evaluation, unit conversion or mathematical rewriting is used
by the public-text scanner. Geometry prediction has no private-key argument.
"""
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher
from collections import Counter
from fractions import Fraction
from reference_checks import digest, require
from reference_normalizers import normalize, NUMBER, NUMERIC_UNIT

LABEL = re.compile(r'(?:[A-Za-zΑ-Ωα-ω甲乙丙丁戊己庚辛壬癸](?:\d{1,2}|_[A-Za-z0-9]|_\{[A-Za-z0-9]{1,2}\})?|[+\-])')


def public_text_fields(public_raw):
    """Scan only actually sent messages and textual attachment descriptors.

    Paths, hashes, private keys and OCR stores are NOT public text channels.
    All stem/options/prompts/excerpts/geometry strings are in rendered messages.
    Future structured messages are supported without scanning image data URLs.
    """
    out = []
    def walk(value, path):
        if isinstance(value, str):
            out.append((path, value))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f'{path}/{i}')
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ('role', 'type', 'url', 'image_url', 'data', 'path', 'bytes_sha256', 'asset_id'):
                    continue
                walk(item, path + '/' + key)
    walk(public_raw.get('messages', {}), '/messages')
    for i, attachment in enumerate(public_raw.get('attachments', [])):
        if isinstance(attachment, dict):
            for key in ('alt', 'title', 'caption', 'text'):
                if key in attachment:
                    walk(attachment[key], f'/attachments/{i}/{key}')
    return out


def _hits(fields, expected, normalizer_id):
    """Finite lexical matching with exact original Unicode character offsets."""
    target = normalize(expected, normalizer_id)
    if target is None:
        return []
    # A permissive finite candidate grammar, followed by the EXISTING normalizer
    # for the equality decision. It never computes numbers or reorders operators.
    unit = NUMERIC_UNIT.fullmatch(target)
    if unit:
        number, name = unit.groups()
        body = re.escape(number).replace(r'\-', r'[\-−]')
        spacing = r'(?:\s|\\[,;:! ])*'
        body += spacing + r'(?:' + re.escape(name) + r'|\\(?:mathrm|text)\{' + re.escape(name) + r'\})'
    else:
        body = re.escape(target).replace(r'\-', r'[\-−]')
    pattern = re.compile(r'(?<![A-Za-z0-9_Α-Ωα-ω])(?:' + body + r')(?![A-Za-z0-9_Α-Ωα-ω])')
    hits = []
    for field, text in fields:
        comparison = unicodedata.normalize('NFC', text)
        offsets = [(i, i + 1) for i in range(len(text))]
        if comparison != text:
            offsets = []
            for tag, i, j, a, b in SequenceMatcher(None, text, comparison, autojunk=False).get_opcodes():
                if tag == 'equal':
                    offsets.extend((k, k + 1) for k in range(i, j))
                else:
                    offsets.extend((i, j) for _ in range(a, b))
        for match in pattern.finditer(comparison):
            s, e = offsets[match.start()][0], offsets[match.end() - 1][1]
            if normalize(text[s:e], normalizer_id) == target:
                hits.append({'field': field, 'start': s, 'end': e, 'text': text[s:e],
                             'normalized': target})
    return hits


def read_availability(public_raw, expected, normalizer_id='ocr_label_v2_1'):
    fields = public_text_fields(public_raw)
    normalized = normalize(expected, normalizer_id)
    finite = normalized is not None and bool(LABEL.fullmatch(normalized) or
                re.fullmatch(NUMBER, normalized) or NUMERIC_UNIT.fullmatch(normalized))
    # Exact supported full expressions remain discoverable even if component
    # parsing is unknown. Unsupported normalizer output is always unknown.
    hits = _hits(fields, expected, normalizer_id) if normalized is not None else []
    components = []
    if finite and normalized:
        match = NUMERIC_UNIT.fullmatch(normalized)
        if match:
            for token in match.groups():
                components.extend({**h, 'component': token} for h in _hits(fields, token, normalizer_id))
    if hits:
        status = 'verbatim_available'  # Full match has priority over component hits.
    elif not finite:
        status = 'unknown'
    elif components:
        status = 'components_available'
    else:
        status = 'not_found'
    # This counts literal positions in the ACTUAL serialized input, not unique
    # source mentions: repeated context excerpts can repeat the same source text.
    values = set()
    for _, text in fields:
        for match in re.finditer(r'(?<![A-Za-z0-9_])' + NUMBER + r'(?![A-Za-z0-9_])', text):
            values.add(match.group())
    return {'public_text_availability': status, 'normalizer_id': normalizer_id,
            'full_hits': hits, 'component_hits': components, 'scanned_fields': [p for p, _ in fields],
            'raw_text_root': digest(fields), 'hit_multiplicity': 'none' if not hits else 'one' if len(hits) == 1 else 'multiple',
            'distinct_public_numeric_literals': len(values),
            'unique_value_bound_to_R': 'not_established', 'vision_required': None,
            'scope_note': 'lexical_diagnostic_only_includes_public_geometry_and_repeated_excerpts',
            'priority': 'full_match_then_supported_components_then_lexical_absence; unsupported_is_unknown'}


def nearest_region(view, image_sizes):
    """Public-only R-center to E-rectangle distance in ORIGINAL source pixels.

    Integer numerator coordinates use denominator 2000 (bbox centers). Squared
    distances share denominator 4,000,000. This avoids rounding-induced ties and
    preserves unequal image aspect ratios. Multiple E parts take the minimum.
    Ties are reported, never broken using correctness or private source labels.
    """
    if len(view['query_anchor_ids']) != 1:
        return {'status': 'not_applicable', 'reason': 'query_arity'}
    anchor = view['anchors'][view['query_anchor_ids'][0]]
    if anchor['kind'] != 'visual':
        return {'status': 'not_applicable', 'reason': 'text_R_has_no_visual_origin'}
    if anchor['geometry']['type'] != 'bbox':
        return {'status': 'not_applicable', 'reason': 'unsupported_geometry'}
    iid = anchor['image_id']
    width, height = image_sizes[iid]
    x0, y0, x1, y1 = anchor['geometry']['bbox_1000']
    cx, cy = (x0 + x1) * width, (y0 + y1) * height
    distances = {}
    for candidate in view['candidates']:
        ds = []
        for loc in candidate.get('locators', []):
            if loc['kind'] != 'visual' or loc['image_id'] != iid:
                continue
            if loc['geometry']['type'] != 'bbox':
                return {'status': 'not_applicable', 'reason': 'unsupported_candidate_geometry'}
            left, top, right, bottom = loc['geometry']['bbox_1000']
            dx = max(2 * left * width - cx, 0, cx - 2 * right * width)
            dy = max(2 * top * height - cy, 0, cy - 2 * bottom * height)
            ds.append(dx * dx + dy * dy)
        if not ds:
            return {'status': 'not_applicable', 'reason': 'cross_image_candidate_distance_undefined'}
        distances[candidate['alias']] = min(ds)
    require(bool(distances), 'NEAREST_EMPTY_CANDIDATES')
    best = min(distances.values())
    winners = sorted(a for a, d in distances.items() if d == best)
    return {'status': 'tie' if len(winners) > 1 else 'predicted', 'predicted_alias': winners[0] if len(winners) == 1 else None,
            'tied_aliases': winners if len(winners) > 1 else [], 'squared_distance_numerators': distances,
            'distance_denominator': 4000000, 'policy_id': 'public_R_center_to_rectangle_pixel_squared_v1'}


def describe_probe(compiler, draft, env, prepared, proposal):
    view, key = draft['view'], draft['key']
    native = proposal['native']
    candidates = key['candidate_map']
    types = {a: compiler.index['entity_types'][t['id']] for a, t in candidates.items()}
    target_type = compiler.index['entity_types'][key['gold_targets'][0]['id']]
    visuals = compiler.rows['visual_nodes']
    by_candidate = {}
    for alias, target in candidates.items():
        vids = compiler.a['audit']['entity_visual_sources'][target['id']]
        kinds = {visuals[v]['type'] == 'text_glyph' for v in vids}
        by_candidate[alias] = 'label_only' if kinds == {True} else 'entity_geometry' if kinds == {False} else 'mixed_or_unresolved'
    kinds = set(by_candidate.values())
    representation = next(iter(kinds)) if len(kinds) == 1 else 'mixed_or_unresolved'
    nearest = nearest_region(view, {i: (env['assets'][i]['width'], env['assets'][i]['height'])
                                     for i in view['context']['image_ids']})
    # Correctness is attached AFTER the public-only predictor has finished.
    nearest['outcome'] = nearest['status']
    if nearest['status'] == 'predicted':
        nearest['outcome'] = 'correct' if candidates[nearest['predicted_alias']] in key['gold_targets'] else 'wrong'
    read_origin = 'image' if key['read_targets'] else 'source_text' if native['interface'] == 'T03-text' else 'none'
    availability = read_availability(prepared['core']['public_raw'], key['read_targets'][0]['expected'],
                                    key['read_targets'][0]['normalizer_id']) if key['read_targets'] else {
        'public_text_availability': 'not_applicable', 'vision_required': None}
    anchor = view['anchors'][view['query_anchor_ids'][0]]
    # Cross-family duplicates share evidence when exact anchor and owner agree.
    evidence = 'EG-' + digest([compiler.a['audit']['group_id'], anchor, key['gold_targets']])
    return {'schema_version': 'native_probe_metadata_v1', 'interface': native['interface'],
            'group_id': native['group_id'], 'evidence_group': evidence,
            'read_origin': read_origin, **availability, 'representation_kind': representation,
            'candidate_representation': by_candidate, 'candidate_native_types_PRIVATE': types,
            'candidate_count': len(candidates), 'same_native_type_candidate_count': sum(t == target_type for t in types.values()),
            'native_types_homogeneous': len(set(types.values())) == 1,
            'image_count': len(view['context']['image_ids']), 'source_language': key['language'],
            'source_namespace': compiler.index['namespace'], 'source_domain': compiler.source.get('domain', 'unknown'),
            'selector': native['selector'], 'fallback_reason': native.get('fallback_reason'),
            'singleton_basis': native.get('singleton_basis'), 'nearest_region': nearest,
            'chance_probability': {'numerator': 1, 'denominator': len(candidates)},
            'packet_state': key['packet_plan']['state'], 'packet_policy_id': key['packet_policy_id'],
            'packet_nonempty_candidate': bool(key['packet_plan']['observation_ids']),
            'P_approved': False, 'L_candidate': bool(key['read_targets']),
            'repair_policy_id': 'unique_exact_sidecar_v1' if compiler.a.get('repair_refs') else None,
            'repair_applied_to_query': any(key['anchor_map'][a]['repair_refs'] for a in view['query_anchor_ids']),
            'source_approved': True, 'transformation_passed': True,
            'benchmark_QA_approval': 'pending', 'human_item_review': 'not_sampled'}


def chance_summary(items):
    """Exact mother-macro diagnostic, NOT the not-yet-implemented scorer."""
    groups = {}
    for item in items:
        groups.setdefault(item['problem_id'], []).append(Fraction(1, item['metadata']['candidate_count']))
    if not groups:
        return None
    value = sum((sum(v, Fraction(0)) / len(v) for v in groups.values()), Fraction(0)) / len(groups)
    return {'numerator': value.numerator, 'denominator': value.denominator, 'decimal': float(value),
            'weighting': 'mean_over_probes_within_mother_then_equal_mean_over_mothers; diagnostic_not_final_metric'}
