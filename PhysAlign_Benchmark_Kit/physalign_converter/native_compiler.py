"""Native binding profile: enumerate first, close dependencies, then select.

The shadow graph includes unusable records. A shared *possible* exact origin
connects records even when their occurrence/owner/selector is invalid. Unknown
influence remains task-wide; it is never silently treated as an empty edge.
"""
from __future__ import annotations
import copy
import re
from collections import defaultdict
from compiler import Compiler
from reference_checks import require, digest, exact_matches, text_digest
from reference_integrity import source_ref_key
from reference_pipeline import config
from io_utils import empty_ledger, read_json
from native_projection import PROFILE

DISABLED = ('T01', 'T02-set', 'T04', 'T05', 'T05-set', 'T06')
SYMBOL = r'(?:[A-Za-zΑ-Ωα-ω](?:\d{1,2}|_[A-Za-z0-9]|_\{[A-Za-z0-9]{1,2}\})?|[甲乙丙丁戊己庚辛壬癸])'
NOUN = r'(?:block|body|particle|point|ball|sphere|charge|object|spring|resistor|capacitor|switch|rod|wire|coil|pulley|bead|ring|magnet|surface|plane|force|field|axis|line|segment|vector|ray)'
ATOMIC_EN = re.compile(r'(?:[Tt]he )?' + NOUN + r'\s+\$?(' + SYMBOL + r')\$?', re.I)
ATOMIC_ZH = re.compile(r'(?:物块|物体|质点|小球|球|点|电荷|弹簧|电阻|电容器|开关|杆|导线|线圈|滑轮|圆环|磁铁|表面|平面|力|场|坐标轴|直线|线段|矢量|射线)\s*\$?(' + SYMBOL + r')\$?')
PLURAL = re.compile(r'\b(?:two|three|four|five|both|several|all|each|every|pair|charges|blocks|particles|bodies|objects|points)\b|\b[2-9]\s+\w+|[两三四五六七八九]|(?:两个|两点|两物|各个|所有|它们|分别|以及)|\band\b|[、，,]', re.I)


def singleton_basis(quote, rows, source_rows):
    """A finite positive grammar, NOT the complement of a plural blacklist."""
    if PLURAL.search(quote):
        return {'state': 'blocked', 'code': 'EXPLICIT_PLURAL_REFERENCE'}
    m = ATOMIC_EN.fullmatch(quote.strip()) or ATOMIC_ZH.fullmatch(quote.strip())
    if not m:
        literal = quote.strip()
        if literal.startswith('$') and literal.endswith('$'):
            literal = literal[1:-1]
        proofs = []
        if re.fullmatch(SYMBOL, literal) and rows:
            for row in rows:
                target = source_rows['physical_nodes'][row['target']]
                if target['symbol'] == literal:
                    proofs.append({**row['ref'], 'collection': 'physical_nodes',
                                   'record_id': target['id'], 'target_field': 'symbol'})
                    continue
                links = [b for b in source_rows['bindings'].values()
                         if b['type'] in ('labels', 'represents') and b['to_id'] == target['id'] and
                         b['from_id'] in source_rows['visual_nodes'] and
                         source_rows['visual_nodes'][b['from_id']]['type'] == 'text_glyph' and
                         source_rows['visual_nodes'][b['from_id']]['text'].strip() == literal]
                if not links:
                    return {'state': 'blocked', 'code': 'SINGLETON_BASIS_UNKNOWN'}
                proofs.extend({**row['ref'], 'collection': 'bindings', 'record_id': b['id'],
                               'target_field': 'to_id'} for b in links)
            return {'state': 'supported', 'rule': 'atomic_symbol_with_explicit_source_symbol_or_label_v1',
                    'quote': quote, 'symbol': literal, 'source_pointers': [r['ref'] for r in rows] + proofs}
        return {'state': 'blocked', 'code': 'SINGLETON_BASIS_UNKNOWN'}
    # Exact reference edges + explicit singular object noun and atomic symbol.
    # Never use the private entity's name to fabricate a public description.
    return {'state': 'supported', 'rule': 'finite_explicit_object_symbol_v1',
            'quote': quote, 'symbol': m.group(1),
            'source_pointers': [r['ref'] for r in rows]}


class NativeCompiler(Compiler):
    def __init__(self, adapted):
        require(adapted['env'].get('native_profile', {}).get('profile_id') == PROFILE,
                'NATIVE_PROJECTION_REQUIRED')
        super().__init__(adapted, config('semantic_registry.json'), empty_ledger())
        self.groups = []
        self.shadow = {'schema_version': 'native_shadow_dependency_v1', 'records': [],
                       'edges': [], 'issues': [], 'groups': self.groups,
                       'policy_disabled': [{'task_id': t, 'code': 'policy_disabled'} for t in DISABLED]}
        self._enumerated = {}
        self._origins = {}
        self._possible = {}
        for col, records in self.rows.items():
            for rid, record in records.items():
                self.shadow['records'].append({'collection': col, 'record_id': rid,
                                               'source_record_hash': digest(record)})
        for rid, loc in self.locators.items():
            self._origins[rid] = digest(loc)
            self._possible[rid] = {digest(loc)}
        # Unresolved references still shadow EVERY exact possible occurrence.
        for m in self.source['text_mentions']:
            if m['id'] in self._possible:
                continue
            possibilities = set()
            for start, end in exact_matches(self.index['context'][m['section']], m['quote']):
                possibilities.add(digest({'kind': 'text', 'section': m['section'],
                    'start': start, 'end': end, 'quote': m['quote'],
                    'section_sha256': text_digest(self.index['context'][m['section']])}))
            self._possible[m['id']] = possibilities
        self.global_issues = []
        if self.source['ambiguities']:
            self.global_issues.append({'code': 'SOURCE_AMBIGUITY_SCOPE_UNBOUNDED',
                'records': [a['id'] for a in self.source['ambiguities']],
                'affects': ['query', 'identity', 'candidate', 'packet'], 'tasks': ['T02', 'T03']})
        for issue in self.a['audit']['issues']:
            if issue['code'] in ('ENTITY_DISPLAY_INCOMPLETE', 'NATIVE_CANDIDATE_DISPLAY_INCOMPLETE'):
                self.global_issues.append({'code': 'CANDIDATE_UNIVERSE_INCOMPLETE',
                    'records': issue['scope'], 'affects': ['candidate'], 'tasks': ['T02', 'T03']})
        bad_glyphs = [v['id'] for v in self.source['visual_nodes']
                      if v['type'] == 'text_glyph' and v['text'].strip() and v['id'] not in self.locators]
        if bad_glyphs:
            self.global_issues.append({'code': 'PACKET_SOURCE_UNRESOLVED', 'records': bad_glyphs,
                                       'affects': ['packet'], 'tasks': ['T02']})
        self.shadow['issues'].extend(copy.deepcopy(self.global_issues))
        self.shadow['anchor_possible_origins'] = {rid: {'resolved': self._origins.get(rid),
             'possible': sorted(origins), 'scope_unknown': not bool(origins)} for rid, origins in self._possible.items()}
        self.shadow['candidate_visual_sources'] = copy.deepcopy(self.a['audit']['entity_visual_sources'])
        self.shadow['identity_binding_ids'] = list(self.a['audit']['identity_binding_ids'])

    def _group(self, task, token, members, rows, reasons, **extra):
        gid = 'NG-' + digest([self.a['audit']['group_id'], task, token])
        reasons = list(reasons) + [x['code'] for x in self.global_issues if task in x['tasks']]
        info = {'group_id': gid, 'task_id': task, 'source_scope': sorted(members),
                'state': 'blocked' if reasons else 'eligible_for_compile',
                'blocking_reasons': sorted(set(reasons)), **extra}
        self.groups.append(info)
        for rid in sorted(members):
            self.shadow['edges'].append({'from_record': rid, 'to_group': gid, 'dependency': 'query_closure'})
        for p in self.source['physical_nodes']:
            if self.a['audit']['entity_visual_sources'][self.index['identity_map'][p['id']]]:
                self.shadow['edges'].append({'from_record': p['id'], 'to_group': gid, 'dependency': 'candidate_universe'})
        for rid in self.a['audit']['identity_binding_ids']:
            self.shadow['edges'].append({'from_record': rid, 'to_group': gid, 'dependency': 'approved_identity'})
        if task == 'T02':
            for v in self.source['visual_nodes']:
                if v['type'] == 'text_glyph':
                    self.shadow['edges'].append({'from_record': v['id'], 'to_group': gid, 'dependency': 'packet_enumeration'})
        if reasons:
            self.blocked.append({'task_id': task, 'scope': sorted(members), 'group_id': gid,
                                  'code': info['blocking_reasons'][0], 'reasons': info['blocking_reasons']})
            return None
        return {'rows': rows, 'native': info}

    def proposals(self, task):
        if task not in ('T02', 'T03'):
            self.blocked.append({'task_id': task, 'scope': [], 'code': 'policy_disabled'})
            return []
        if task not in self._enumerated:
            self._enumerated[task] = self._t02() if task == 'T02' else self._t03()
        return copy.deepcopy(self._enumerated[task])

    def _t02(self):
        origins, unknown = defaultdict(list), []
        for b in self.source['bindings']:
            if b['type'] != 'refers_to':
                continue
            row = {'anchor': b['from_id'], 'target': b['to_id'],
                   'ref': self.ref('bindings', b['id'], 'to_id')}
            self.shadow['edges'].append({'from_record': b['from_id'], 'to_record': b['id'], 'dependency': 'possible_reference_origin'})
            possible = self._possible.get(b['from_id'], set())
            if not possible:
                unknown.append(b['id'])
            for token in possible:
                origins[token].append(row)
        result = []
        if unknown:
            self.shadow['issues'].append({'code': 'REFERENCE_SCOPE_UNBOUNDED', 'records': unknown,
                                         'tasks': ['T02'], 'affects': ['query', 'identity']})
            self._group('T02', 'unlocated', unknown, [], ['REFERENCE_SCOPE_UNBOUNDED'])
        for token, rows in sorted(origins.items()):
            members = [r['ref']['record_id'] for r in rows]
            reasons = ['REFERENCE_SCOPE_UNBOUNDED'] if unknown else []
            if any(self._origins.get(r['anchor']) != token for r in rows):
                reasons.append('REFERENCE_POSSIBLE_SECOND_TARGET_UNRESOLVED')
            targets = {self.index['identity_map'].get(r['target']) for r in rows}
            if None in targets or len(targets) != 1:
                reasons.append('REFERENCE_TARGET_NOT_SINGLE')
            quote = self.rows['text_mentions'][rows[0]['anchor']]['quote']
            basis = singleton_basis(quote, rows, self.rows)
            if basis['state'] != 'supported':
                reasons.append(basis['code'])
            if any(self.rows['text_mentions'][r['anchor']]['role'] != 'entity' for r in rows):
                reasons.append('REFERENCE_NOT_EXPLICIT_ENTITY_MENTION')
            p = self._group('T02', token, members + unknown, rows, reasons,
                            interface='T02-single', singleton_basis=basis,
                            selector='exact_entity_mention_all_refers_to_v1',
                            possible_origin=token, canonical_targets=sorted(t for t in targets if t))
            if p is not None:
                result.append(p)
        return result

    def _t03(self):
        quantities = self.rows['quantities']
        adjacent, qanchors, selected, failures = defaultdict(set), {}, {}, {}
        unbounded = []
        for qid, q in quantities.items():
            glyphs = [v for v in q['visual_anchor_ids'] if self.rows['visual_nodes'][v]['type'] == 'text_glyph']
            mentions = [m for m in q['text_mention_ids'] if self.rows['text_mentions'][m]['role'] == 'quantity']
            possible = set()
            for rid in glyphs + mentions:
                possible.update(self._possible.get(rid, set()))
                self.shadow['edges'].append({'from_record': rid, 'to_record': qid,
                                             'dependency': 'possible_quantity_identity'})
            qanchors[qid] = possible
            for origin in possible:
                adjacent[origin].add(qid)
            errors = []
            if not q['owner_id'] or q['owner_id'] not in self.index['identity_map']:
                errors.append('QUANTITY_OWNER_UNRESOLVED')
            # An unusable or ambiguous image glyph is NOT an absent image glyph.
            chosen = glyphs if glyphs else mentions
            unique = {self._origins[r] for r in chosen if r in self._origins}
            if not chosen or len(unique) != 1 or any(r not in self._origins for r in chosen):
                errors.append('IMAGE_SELECTOR_UNRESOLVED' if glyphs else 'TEXT_SELECTOR_UNRESOLVED')
            else:
                selected[qid] = {'anchor': sorted(chosen)[0], 'origin': next(iter(unique)),
                                 'selector': 'existing_text_glyph_v1' if glyphs else 'no_image_glyph_exact_quantity_mention_v1',
                                 'interface': 'T03-image' if glyphs else 'T03-text',
                                 'fallback_reason': None if glyphs else 'no_source_text_glyph_anchor'}
                if glyphs and any(not self.rows['visual_nodes'][r]['text'].strip() for r in glyphs):
                    errors.append('EMPTY_IMAGE_GLYPH_NOT_TEXT_FALLBACK')
            if not possible or any(not self._possible.get(rid) for rid in glyphs + mentions):
                unbounded.append(qid)
            failures[qid] = errors
        if unbounded:
            self.shadow['issues'].append({'code': 'QUANTITY_SCOPE_UNBOUNDED', 'records': unbounded,
                                         'tasks': ['T03'], 'affects': ['query', 'identity']})
        # Close components BEFORE discarding any bad row. Shared possible text
        # anchors also prevent image/text fallback from hiding owner conflicts.
        components, remaining = [], set(quantities)
        while remaining:
            queue, component = [min(remaining)], set()
            while queue:
                qid = queue.pop()
                if qid in component:
                    continue
                component.add(qid)
                for origin in qanchors[qid]:
                    queue.extend(adjacent[origin] - component)
            remaining -= component
            components.append(component)
        result = []
        for component in components:
            reasons = [e for q in component for e in failures[q]]
            if unbounded:
                reasons.append('QUANTITY_SCOPE_UNBOUNDED')
            owners = {self.index['identity_map'].get(quantities[q]['owner_id']) for q in component}
            if None in owners or len(owners) != 1:
                reasons.append('QUANTITY_IDENTITY_OR_OWNER_CONFLICT')
            signatures = {digest({k: quantities[q][k] for k in ('kind', 'symbol_latex', 'value_latex', 'unit_latex')})
                          for q in component}
            if len(signatures) != 1:
                reasons.append('QUANTITIES_UNQUALIFIED_IDENTITY_CONFLICT')
            by_origin = defaultdict(list)
            for qid in sorted(component):
                if qid in selected:
                    by_origin[selected[qid]['origin']].append(qid)
            if not by_origin:
                by_origin['unresolved:' + digest(sorted(component))] = []
            for origin, qids in sorted(by_origin.items()):
                rows = [{'anchor': selected[q]['anchor'], 'target': quantities[q]['owner_id'],
                         'ref': self.ref('quantities', q, 'owner_id')} for q in qids]
                choice = selected[qids[0]] if qids else {'interface': 'T03-unresolved', 'selector': None,
                                                        'fallback_reason': 'no_legal_selector'}
                p = self._group('T03', origin, component | set(unbounded), rows, reasons,
                                interface=choice['interface'], selector=choice['selector'],
                                fallback_reason=choice['fallback_reason'],
                                dependency_component=sorted(component), possible_origin=origin,
                                canonical_targets=sorted(t for t in owners if t))
                if p is not None:
                    result.append(p)
        return result

    def compile(self, task, proposal, max_views=128):
        require(task in ('T02', 'T03'), 'NATIVE_TASK_POLICY_DISABLED')
        valid = self.proposals(task)
        require(any(p == proposal for p in valid), 'NATIVE_PROPOSAL_NOT_COMPLETE_OR_ELIGIBLE')
        original = self.env
        try:
            self.env = {**self.env, 'native_query': copy.deepcopy(proposal['native'])}
            return super().compile(task, proposal, max_views)
        finally:
            self.env = original


def validate_native_query(draft, env):
    """Recompute admissible groups from the FULL shadow source, not saved rows."""
    from adapter import verify_adapter_snapshot
    rebuilt = verify_adapter_snapshot(env)
    c = NativeCompiler(rebuilt)
    proposals = c.proposals(draft['key']['task_id'])
    matches = [p for p in proposals if p['native'] == env.get('native_query')]
    require(len(matches) == 1, 'NATIVE_QUERY_CLOSURE_CHANGED')
    proposal = matches[0]
    refs = {digest(r['ref']) for r in proposal['rows']}
    require(refs == {digest(r) for r in draft['key']['source_records']}, 'NATIVE_ALL_SOURCE_ROWS_REQUIRED')
    require(len(draft['view']['query_anchor_ids']) == 1, 'NATIVE_QUERY_ARITY')
    alias = draft['view']['query_anchor_ids'][0]
    require(draft['view']['anchors'][alias] == c.loc(proposal['rows'][0]['anchor']), 'NATIVE_SELECTOR_CHANGED')
    for a, meta in draft['key']['anchor_map'].items():
        expected = sorted({ref for rid, loc in c.locators.items() if loc == draft['view']['anchors'][a]
                           for ref in rebuilt.get('repair_refs', {}).get(rid, [])})
        require(meta['repair_refs'] == expected, 'NATIVE_REPAIR_REFERENCES_CHANGED')
    require(draft['key']['response_contract']['cardinality'] == 'one', 'NATIVE_SET_DISABLED')
