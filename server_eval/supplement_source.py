"""Read-only release import for original solving and shortcut controls.

Kept outside physalign/: adding to that package would change the implementation
hash of the API evaluations that are already in progress.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from physalign.contracts import decode_probe
from physalign.controls import original_ids
from physalign.dataset import require, sections
from physalign.scoring import score_response
from physalign.solving import validate_answer_key
from physalign.storage import canonical, confined, digest, file_hash, fingerprint, loads, read_json
from server_eval.two_experiments import RELEASE_SCHEMA, indexed, inventory_map, target_id

SOLVE_SYSTEM = '''Solve the original physics problem using its statement and attached original images.
Answer the requested question, including every required subpart. If the original statement contains earlier context questions and answers, treat them as supplied background and answer the currently requested task.
Return exactly one JSON object, without Markdown fences or text outside it:
{"answer": "final answer", "explanation": "concise worked solution"}
For a question with one correct option, answer must be that original choice identifier, not the option text. For a select-all-that-apply question with more than one correct option, answer must be an array of the original choice identifiers. For other questions, answer must be a string containing the final result, units when applicable, and labelled results for all requested subparts. For a proof, include the argument in explanation and a concise conclusion in answer. Do not invent new choice identifiers.'''


def original_text(manifest):
    """Preserve raw_question verbatim; append options only when not already there."""
    question = manifest['raw_question']
    require(isinstance(question, str) and question.strip(), 'Empty original question')
    options = manifest['segments']['options']
    require(isinstance(options, list), 'Original options must be an array')
    for option in options:
        require(isinstance(option, dict) and isinstance(option.get('label'), str)
                and isinstance(option.get('text'), str) and option['text'].strip(), 'Invalid original option')
    # Append the complete, originally labelled list if any option is absent.
    # Never infer a missing option from a local grounding query.
    if options and not all(o['text'] in question for o in options):
        question += '\n\nOriginal answer options:\n' + '\n'.join(o['label'] + '. ' + o['text'] for o in options)
    return question


class Release:
    def __init__(self, root, base_plan):
        self.root = Path(root).resolve()
        self.inventory = inventory_map(read_json(self.root / 'FILE_MANIFEST.json'))
        self.used = {}
        release = self.json('release.json')
        require(release.get('schema_version') == RELEASE_SCHEMA, 'Unexpected source release schema')
        require(release.get('attachment_paths_relative_to') == 'public', 'Unexpected source image path base')
        self.identity = {'release_sha256': file_hash(self.root / 'release.json'),
                         'inventory_sha256': file_hash(self.root / 'FILE_MANIFEST.json')}
        expected_inventory = base_plan['source_hashes'].get('private/source_inventory.json')
        require(expected_inventory == digest((canonical(self.inventory) + '\n').encode('utf-8')),
                'Main plan was converted from another release inventory')
        self.base = base_plan
        self.raw = indexed(self.json('public/qa_raw.json'), 'raw')
        self.gold = indexed(self.json('public/qa_gold.json'), 'gold')
        self.members = indexed(self.json('private/membership.json'), 'membership')
        self.maps = indexed(self.json('private/mappings.json'), 'mappings')
        self.answers = indexed(self.json('private/answers.json'), 'binding answers')
        ids = {p['instance_id'] for p in base_plan['probes']}
        require(ids and ids <= self.raw.keys() & self.members.keys() & self.maps.keys() & self.answers.keys(),
                'Base plan IDs are absent from this release')
        self.metadata = {p['instance_id']: p for p in base_plan['probes']}
        self.probes = {}
        for iid in sorted(ids):
            key, meta = self.maps[iid]['key'], self.metadata[iid]
            require(key['problem_id'] == self.members[iid]['problem_id'] == meta['problem_id'], 'Mother ID mismatch')
            require(key['logical_probe_id'] == meta['logical_probe_id'] == self.raw[iid]['logical_probe_id'], 'Logical ID mismatch')
            require(meta['problem_id'] not in release.get('quarantined_mother_ids', []), 'Quarantined mother')
            require(key['response_contract']['cardinality'] == 'one', 'Unsupported non-singleton binding')
            row = {'probe_id': iid, 'problem_id': meta['problem_id'], 'probe_type': meta['probe_type'],
                   'binding': {'kind': 'single', 'symmetric': False,
                       'domains': [{'field': key['response_contract']['binding_key'],
                                    'candidates': {a: target_id(v) for a, v in key['candidate_map'].items()}}],
                       'gold': [target_id(v) for v in key['gold_targets']]},
                   'readings': [{'field': r.get('field', 'read'), 'anchor_id': r['anchor_alias'],
                                 'expected': r['expected'], 'rule': {'kind': 'ocr', 'normalizer_id': r['normalizer_id']}}
                                for r in key['read_targets']], 'packet_nonempty': meta['packet_nonempty']}
            probe = decode_probe(row)
            require(probe.joint_eligible == meta['joint_eligible'], 'Recognition eligibility mismatch')
            require(score_response(probe, canonical(self.answers[iid]['answer'])).B == 1, 'Conflicting binding keys')
            self.probes[iid] = row
        for row in base_plan['requests']:
            public = (self.raw if row['condition'] == 'raw' else self.gold)[row['instance_id']]
            require(fingerprint(public['input']) == row['input_hash'], 'Release input differs from frozen main plan')
        # These are the only original sources used; passes/review/keys never enter a prompt.
        components = self.json('private/merge_request.json')['components']
        self.components = {c['name']: c['relative_path'] for c in components}

    def json(self, relative):
        path = confined(self.root, relative)
        require(relative in self.inventory and file_hash(path) == self.inventory[relative],
                'Source missing or changed: ' + relative)
        self.used[relative] = self.inventory[relative]
        return read_json(path)

    def originals(self):
        result = {}
        for iid in sorted(self.probes):
            member = self.members[iid]
            mother = member['problem_id']
            inp = self.raw[iid]['input']
            ids = original_ids(inp['messages']['user'])
            by_id = {a['asset_id']: a for a in inp['attachments']}
            attachments = [deepcopy(by_id[a]) for a in ids]
            if mother in result:
                require(attachments == result[mother]['input']['attachments'], 'Original images differ across mother probes')
                continue
            relative = (self.components[member['component']] + '/private/project/source/problems/'
                        + mother + '/sources/manifest.json')
            manifest = self.json(relative)
            require(manifest['problem_id'] == mother, 'Original source identity mismatch')
            expected = [(a['image_id'], a['sha256'], a['width'], a['height']) for a in manifest['images']]
            actual = [(a['asset_id'], a['sha256'], a['width'], a['height']) for a in attachments]
            require(expected == actual, 'Original image identity/order differs from source manifest: ' + mother)
            text = original_text(manifest)
            original = {'messages': {'system': SOLVE_SYSTEM, 'user': text}, 'attachments': attachments}
            result[mother] = {'input': original, 'input_hash': fingerprint(original), 'source_reference': relative,
                'source_manifest_sha256': self.inventory[relative],
                'source_dataset': manifest['source_dataset'], 'source_sample_id': manifest['source_sample_id'],
                'contains_context_answers': 'Context answer:' in manifest['raw_question'],
                'original_option_count': len(manifest['segments']['options'])}
        return result

    def nearest_declaration(self, iid):
        """Eligibility uses query kind + public geometry, never a correct target."""
        query = self.maps[iid]['key']['source_query']
        if query['kind'] not in {'label', 'quantity'}:
            return None, 'not_label_or_quantity_ownership'
        evidence = loads(sections(self.raw[iid]['input']['messages']['user'])['Evidence locations and candidates'])
        loc = query['locator']
        if loc['kind'] != 'visual':
            return None, 'text_anchor'
        anchors = [a for a, value in evidence['anchors'].items() if value == loc]
        if len(anchors) != 1:
            return None, 'source_locator_has_no_unique_public_anchor'
        if not all(any(v['kind'] == 'visual' and v['image_id'] == loc['image_id']
                       for v in c['locators']) for c in evidence['candidates']):
            return None, 'candidate_missing_comparable_region_on_anchor_image'
        return {'anchor_id': anchors[0], 'field': self.maps[iid]['key']['response_contract']['binding_key'],
                'eligibility_basis': 'source query is label/quantity; unique visual anchor; every candidate has a region on that image'}, None


def answer_template(source, originals):
    return {'schema_version': 'physalign_original_answer_keys_v1', 'source_identity': source.identity,
            'note': 'Fill only from reliable original-problem answers before inference. Binding answers are not solve keys.',
            'problems': {m: {'input_hash': x['input_hash'], 'source_reference': x['source_reference'],
                'source_dataset': x['source_dataset'], 'source_sample_id': x['source_sample_id'],
                'contains_context_answers': x['contains_context_answers'], 'context_answers_reviewed': False,
                'context_review_note': '', 'answer_key': None} for m, x in originals.items()}}


def freeze_keys(path, source, originals):
    keys = dict.fromkeys(originals)
    if path is None:
        return keys
    artifact = read_json(Path(path))
    require(artifact.get('schema_version') == 'physalign_original_answer_keys_v1', 'Unknown original key schema')
    require(artifact.get('source_identity') == source.identity, 'Keys belong to another source release')
    require(set(artifact['problems']) == set(originals), 'Answer template must cover exactly the selected mothers')
    for mother, original in originals.items():
        entry = artifact['problems'][mother]
        require(entry['input_hash'] == original['input_hash'], 'Answer key is bound to a different original prompt')
        key = entry['answer_key']
        if key is not None:
            validate_answer_key(key)
            if original['contains_context_answers']:
                require(entry.get('context_answers_reviewed') is True and entry.get('context_review_note', '').strip(),
                        'Review supplied context answers against the current target before grading: ' + mother)
            if key['kind'] == 'choice_set':
                require(len(set(key['accepted'])) == len(key['accepted']), 'Duplicate choices in reference answer')
        keys[mother] = key
    return keys


def apply_original_catalog(path, source, originals):
    """Explicit, hashed upstream originals; never rewrite the source release."""
    if path is None:
        return originals
    catalog = read_json(Path(path))
    require(catalog.get('schema_version') == 'physalign_matched_originals_v1'
            and catalog.get('source_identity') == source.identity, 'Original catalog source mismatch')
    require(catalog.get('catalog_hash') == fingerprint({k: v for k, v in catalog.items() if k != 'catalog_hash'}),
            'Original catalog changed')
    require(set(catalog['problems']) == set(originals), 'Original catalog mother selection changed')
    result = {}
    for mother, original in originals.items():
        entry = catalog['problems'][mother]
        require(entry['source_input_hash'] == original['input_hash'], 'Catalog is bound to a different source original')
        require(entry['match_status'] == 'matched' and entry.get('provenance'), 'Unresolved upstream match: ' + mother)
        inp = entry['input']
        require(set(inp) == {'messages', 'attachments'} and set(inp['messages']) == {'system', 'user'}
                and inp['messages']['system'] == SOLVE_SYSTEM and isinstance(inp['messages']['user'], str)
                and inp['messages']['user'].strip(), 'Invalid original public prompt')
        ids = [a['asset_id'] for a in inp['attachments']]
        require(ids and len(ids) == len(set(ids)), 'Invalid original image list')
        for a in inp['attachments']:
            require(set(a) == {'asset_id', 'path', 'sha256', 'width', 'height'}, 'Unexpected original attachment metadata')
            confined(Path(path).parent, a['path'])
        result[mother] = {**original, 'input': inp, 'input_hash': fingerprint(inp),
                         'upstream_provenance': entry['provenance'], 'upstream_catalog_hash': catalog['catalog_hash'],
                         'source_input_hash': original['input_hash']}
    return result


def source_summary(source, originals):
    exclusions, eligible = Counter(), 0
    for iid in source.probes:
        declaration, reason = source.nearest_declaration(iid)
        eligible += declaration is not None
        if reason:
            exclusions[reason] += 1
    return {'mothers': len(originals), 'probes': len(source.probes),
            'base_requests': len(source.base['requests']), 'nearest_eligible': eligible,
            'nearest_exclusions': dict(exclusions),
            'originals_with_options': sum(x['original_option_count'] > 0 for x in originals.values()),
            'originals_with_context_answers': sum(x['contains_context_answers'] for x in originals.values()),
            'original_answer_keys_in_release': 0}
