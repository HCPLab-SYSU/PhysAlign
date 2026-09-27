"""Conservative direct projections from reviewed G_obs, never a reasoning model."""
from __future__ import annotations
import copy
from collections import defaultdict
from reference_checks import require, digest, logical_id, select_atomic_observations
from reference_integrity import active_origin, source_ref_key, canonical_entity, ReviewLedger
from reference_registry import resolve_relation, entry_root, render_qualifiers, validate_registry
from reference_pipeline import config, neutral_aliases, prepare_probe
from renderer import render


class Compiler:
    def __init__(self, adapted, registry, registry_ledger):
        self.a = adapted
        self.env = copy.deepcopy(adapted['env'])
        self.env['registry'] = copy.deepcopy(registry)
        overlap = set(registry_ledger['records']) & set(self.env['ledger']['records'])
        require(not overlap, 'REGISTRY_LEDGER_ID_COLLISION')
        self.env['ledger']['records'].update(copy.deepcopy(registry_ledger['records']))
        validate_registry(registry, ReviewLedger(self.env['ledger'], production=True),
                          production=True, evidence_source_index=self.env['source_index'])
        self.source = adapted['source']
        self.index = self.env['source_index']
        self.locators = adapted['locators']
        self.rows = {c: {r['id']: r for r in self.source[c]} for c in self.source if isinstance(self.source[c], list)}
        self.blocked = []

    def ref(self, col, rid, field):
        return {'collection': col, 'record_id': rid, 'target_field': field, 'source_file_sha256': self.a['sha']}

    def loc(self, rid):
        require(rid in self.locators, 'ANCHOR_UNRESOLVED', rid)
        return self.locators[rid]

    def entity(self, pid):
        return canonical_entity(pid, self.index)

    def _by_visible_anchor(self, rows):
        groups = defaultdict(list)
        for r in rows:
            groups[digest(self.loc(r['anchor']))].append(r)
        return list(groups.values())

    def proposals(self, task):
        """Enumerate every relevant row before grouping/cardinality/display filtering."""
        source = self.source
        require(not source['ambiguities'], 'SOURCE_AMBIGUITIES_REQUIRE_ADJUDICATION')
        if task == 'T02':
            rows = []
            for b in source['bindings']:
                if b['type'] != 'refers_to':
                    continue
                require(b['from_id'] in self.rows['text_mentions'], 'REFERENCE_ENDPOINT_NOT_MENTION', b['id'])
                self.entity(b['to_id'])
                rows.append({'anchor': b['from_id'], 'ref': self.ref('bindings', b['id'], 'to_id'), 'target': b['to_id']})
            return [{'rows': g} for g in self._by_visible_anchor(rows)]
        if task == 'T03':
            rows = []
            for q in source['quantities']:
                require(q['owner_id'], 'QUANTITY_OWNER_UNRESOLVED', q['id'])
                # Multiple evidence snippets are not automatically the same quantity.
                glyphs = [v for v in q['visual_anchor_ids'] if self.rows['visual_nodes'][v]['type'] == 'text_glyph']
                mentions = [m for m in q['text_mention_ids'] if self.rows['text_mentions'][m]['role'] == 'quantity']
                anchors = glyphs if glyphs else mentions
                require(len(anchors) == 1, 'QUANTITY_EXACT_ANCHOR_REQUIRED', q['id'])
                rows.append({'anchor': anchors[0], 'ref': self.ref('quantities', q['id'], 'owner_id'), 'target': q['owner_id']})
            return [{'rows': g} for g in self._by_visible_anchor(rows)]
        if task == 'T04':
            rows = []
            for c in source['constraints']:
                anchors = [m for m in c['evidence_mention_ids'] if self.rows['text_mentions'][m]['role'] == 'constraint']
                require(len(anchors) == 1, 'CONDITION_EXACT_ANCHOR_REQUIRED', c['id'])
                rows.append({'anchor': anchors[0], 'ref': self.ref('constraints', c['id'], 'subject_ids'), 'targets': c['subject_ids']})
            return [{'rows': g} for g in self._by_visible_anchor(rows)]
        if task == 'T01':
            roles = {r['canonical_id'] for r in self.env['registry']['roles'] if r['status'] == 'approved'}
            require(len(roles) >= 2, 'ROLE_REGISTRY_REQUIRED')
            rows = []
            for b in source['bindings']:
                if b['type'] != 'represents':
                    continue
                require(b['from_id'] in self.rows['visual_nodes'] and b['to_id'] in self.rows['physical_nodes'],
                        'REPRESENTS_ENDPOINT_INVALID', b['id'])
                role = self.rows['physical_nodes'][b['to_id']]['type']
                rows.append({'anchor': b['from_id'], 'ref': self.ref('physical_nodes', b['to_id'], 'type'),
                             'role': role, 'binding_id': b['id']})
            return [{'rows': g} for g in self._by_visible_anchor(rows)]
        if task == 'T05':
            groups = defaultdict(list)
            # Unknown predicates are reported separately. No guessed aliases.
            # QA review must also inspect all original relations for missing synonyms.
            for r in source['relations']:
                try:
                    resolved = resolve_relation(self.env['registry'], self.index['namespace'], r['predicate'],
                        ledger=ReviewLedger(self.env['ledger'], production=True), evidence_source_index=self.index)
                except ValueError as exc:
                    self.blocked.append({'task_id': task, 'scope': [r['id']], 'code': getattr(exc, 'code', 'REGISTRY_INVALID'), 'detail': str(exc)})
                    continue
                e = resolved['entry']
                for side in ('object', 'subject'):
                    missing = side
                    if resolved['transform'] == 'swap':
                        missing = 'subject' if side == 'object' else 'object'
                    field = missing + '_id'
                    fixedfield = 'subject_id' if field == 'object_id' else 'object_id'
                    direction = 'object' if e['symmetric'] else side
                    ref = self.ref('relations', r['id'], field)
                    qualifiers = render_qualifiers(e, e['queries'][direction], [ref], self.index, self.index['context'])
                    visible = [{k: q[k] for k in ('qualifier_id', 'value', 'rendered')} for q in qualifiers]
                    token = digest([e['canonical_id'], direction, self.entity(r[fixedfield]), visible])
                    groups[token].append({'ref': ref, 'target': r[field], 'fixed': r[fixedfield],
                                          'fixedfield': fixedfield, 'side': direction, 'entry': e})
            return [{'rows': rows, 'relation': True} for rows in groups.values()]
        raise ValueError('Unsupported data task: ' + task)

    def compile(self, task, proposal, max_views=128):
        env = copy.deepcopy(self.env)
        rows = proposal['rows']
        if task == 'T03':
            signatures = {digest({k: self.rows['quantities'][r['ref']['record_id']][k]
                                  for k in ('kind', 'symbol_latex', 'value_latex', 'unit_latex')}) for r in rows}
            require(len(signatures) == 1, 'MULTIPLE_QUANTITIES_AT_UNQUALIFIED_ANCHOR')
        refs = {digest(r['ref']): r['ref'] for r in rows}
        refs = [refs[t] for t in sorted(refs)]
        if task == 'T01':
            gold = [{'kind': 'role', 'id': r['role']} for r in rows]
        else:
            gold = [self.entity(p) for r in rows for p in r.get('targets', [r.get('target')])]
        gold = list({digest(g): g for g in gold}.values())
        gold.sort(key=digest)
        require(gold, 'EMPTY_TARGET_SET')
        card = 'set' if task == 'T04' else ('one' if len(gold) == 1 else 'set')
        require(task + '-' + card in env['enabled_variants'], 'SET_EXTENSION_NOT_CORE', str([r['record_id'] for r in refs]))
        identity = {'namespace': self.index['namespace'], 'split': self.a['audit']['source_split'],
                    'sample': self.a['audit']['source_sample_id'], 'task': task}
        if task == 'T05':
            # Stable source IDs, never content hashes/answers/alias seed.
            identity.update(predicate=rows[0]['entry']['canonical_id'], side=rows[0]['side'],
                            fixed=sorted({r['fixed'] for r in rows}), relations=sorted({r['ref']['record_id'] for r in rows}))
        else:
            identity['anchor_ids'] = sorted({r['anchor'] for r in rows})
        qid = logical_id(identity)
        anchors, amap, reads = {}, {}, []
        observations = env['observations']['observations']
        def add_anchor(rid):
            loc = self.loc(rid)
            for alias, existing in anchors.items():
                if existing == loc:
                    return alias
            alias = 'R' + str(len(anchors) + 1)
            col = 'visual_nodes' if rid in self.rows['visual_nodes'] else 'text_mentions'
            # Retain every source ID at this exact origin, not just a representative.
            sids = sorted(source_ref_key(self.ref(col, other, 'id')) for other in self.rows[col]
                          if self.locators.get(other) == loc)
            oids = sorted(o for o, obs in observations.items() if obs['origin']['locator'] == loc)
            require(len(oids) <= 1, 'DUPLICATE_OBSERVATION_AT_SAME_ORIGIN', rid)
            anchors[alias] = copy.deepcopy(loc)
            repairs = sorted({ref for other in self.rows[col] if self.locators.get(other) == loc
                              for ref in self.a.get('repair_refs', {}).get(other, [])})
            amap[alias] = {'source_ids': sids, 'observation_ids': oids, 'repair_refs': repairs,
                           'origin': active_origin(loc, self.index['context'], env['assets'])}
            return alias
        query_anchors = []
        if task != 'T05':
            alias = add_anchor(rows[0]['anchor'])
            query_anchors = [alias]
            if task in ('T01', 'T03') and anchors[alias]['kind'] == 'visual':
                oids = amap[alias]['observation_ids']
                if oids:
                    reads = [{'anchor_alias': alias, 'observation_id': oids[0], 'normalizer_id': 'ocr_label_v2_1'}]
        if task == 'T03':
            policy = 'target_literal_only_v1'
            selected = [r['observation_id'] for r in reads]
        else:
            policy = 'all_atomic_glyphs_in_presented_diagrams_v1'
            selected = select_atomic_observations({o: {**v, 'image_id': v['origin']['locator']['image_id']}
                for o, v in observations.items()}, set(self.index['context']['image_ids']), config('packet_selection_v2.json')[policy])['observation_ids']
        for oid in selected:
            add_anchor(observations[oid]['source_records'][0]['record_id'])
        rmap, rviews, rc = {}, {}, None
        allowed = None
        if task == 'T05':
            first = rows[0]
            entry, side = first['entry'], first['side']
            fixed = self.entity(first['fixed'])
            locs = self.index['entity_locators'][fixed['id']]
            require(locs, 'FIXED_ENDPOINT_UNDISPLAYABLE')
            rviews = {'H1': copy.deepcopy(locs)}
            rmap = {'H1': {'canonical_target': fixed,
                          'source_records': [{**r['ref'], 'target_field': r['fixedfield']} for r in rows],
                          'origins': [active_origin(l, self.index['context'], env['assets']) for l in locs],
                          'identity_review_refs': []}}
            qualifiers = render_qualifiers(entry, entry['queries'][side], refs, self.index, self.index['context'])
            rc = {'canonical_predicate': entry['canonical_id'], 'missing_side': side, 'fixed_reference_alias': 'H1',
                  'fixed_target': fixed, 'registry_entry_hash': entry_root(entry), 'public_qualifiers': qualifiers,
                  'closure_review_ref': 'qa-closure'}
            allowed = entry['object_types' if side == 'object' else 'subject_types']
        if task == 'T01':
            targets = [{'kind': 'role', 'id': r['canonical_id']} for r in env['registry']['roles'] if r['status'] == 'approved']
        else:
            for p in self.source['physical_nodes']:
                require(not p['visual_anchor_ids'] or self.index['entity_locators'][self.index['identity_map'][p['id']]],
                        'CANDIDATE_UNIVERSE_INCOMPLETE', p['id'])
            targets = [{'kind': 'entity', 'id': p} for p, locs in self.index['entity_locators'].items()
                       if locs and (allowed is None or self.index['entity_types'][p] in allowed)]
        cmap, display_order = neutral_aliases(targets, qid, variant_id='main', seed=2027)
        require({digest(g) for g in gold} <= {digest(t) for t in targets}, 'GOLD_OUTSIDE_LEGAL_CANDIDATES')
        candidates = []
        for alias in display_order:
            target = cmap[alias]
            if task == 'T01':
                role = next(r for r in env['registry']['roles'] if r['canonical_id'] == target['id'])
                candidates.append({'alias': alias, 'kind': 'role', 'label': role['label'][self.index['language']]})
            else:
                candidates.append({'alias': alias, 'kind': 'entity', 'locators': copy.deepcopy(self.index['entity_locators'][target['id']])})
        view = {'task_id': task, 'language': self.index['language'], 'context': copy.deepcopy(self.index['context']),
                'anchors': anchors, 'query_anchor_ids': query_anchors, 'read_anchor_ids': [r['anchor_alias'] for r in reads],
                'reference_views': rviews, 'used_reference_ids': list(rviews), 'candidates': candidates, 'recognition_packet': []}
        key = {'schema_version': 'physalign_private_v2_1', 'problem_id': self.a['audit']['problem_id'],
               'logical_probe_id': qid, 'variant_id': 'main', 'task_id': task, 'language': self.index['language'],
               'source_records': refs, 'source_hashes': {k: v['bytes_sha256'] for k, v in env['source_files'].items()},
               'candidate_map': cmap, 'anchor_map': amap, 'reference_map': rmap, 'gold_targets': gold, 'read_targets': reads,
               'response_contract': {'binding_key': config('task_registry_v2.json')[task]['binding_key'], 'cardinality': card,
                   'read_shape': 'string' if reads else 'none', 'read_anchor_ids': view['read_anchor_ids']},
               'joint_eligible': bool(reads), 'packet_plan': {'state': 'pending_review', 'observation_ids': selected,
                    'empty_reason': None, 'review_ref': None}, 'packet_policy_id': policy,
               'selection_policy_id': 'neutral_hash_per_probe_v2_1', 'identity_review_refs': [],
               'closure_review_refs': ['qa-closure'], 'relation_context': rc,
               'pass5_analysis': {'status': 'unavailable', 'source_fact_step_map': {}}}
        render(env, view, 'views/' + qid, max_views)
        draft = {'view': view, 'key': key}
        prepared = prepare_probe(draft, env, verify_reviews=False)
        # Keep derived transcription/version data deterministic in the saved draft.
        draft['key'] = prepared['core']['private_core']
        return draft, env, prepared
