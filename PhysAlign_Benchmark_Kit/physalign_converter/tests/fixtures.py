"""Artificial physics drawings / approvals solely for regression tests."""
import copy
from pathlib import Path
from PIL import Image, ImageDraw
from io_utils import write_json, sha_bytes, empty_ledger, review_record
from reference_checks import digest, canonical_bytes
from source_validation import blank_document
from reference_registry import entry_root
from reference_pipeline import config


def workspace(root, language='en'):
    root = Path(root)
    annotation, dataset = root / 'annotation', root / 'dataset'
    image = dataset / 'images' / 'diagram.png'
    image.parent.mkdir(parents=True)
    im = Image.new('RGB', (320, 200), 'white')
    draw = ImageDraw.Draw(im)
    draw.rectangle((20, 80, 90, 140), outline='black', width=2)
    draw.rectangle((190, 80, 270, 140), outline='black', width=2)
    draw.text((25, 42), 'm', fill='black')
    im.save(image)
    pid = 'SYNTHETIC_TEST_ONLY'
    stem = 'Block A and block B are at rest. The mass is m.' if language == 'en' else '物块甲和物块乙保持静止。质量为 m。'
    query = 'Find the acceleration.' if language == 'en' else '求加速度。'
    phrases = ['Block A', 'are at rest', 'm', 'acceleration'] if language == 'en' else ['物块甲', '保持静止', 'm', '加速度']
    row = {'workspace_schema_version': 1, 'problem_id': pid, 'source_dataset': 'SYNTHETIC_UNIT_TEST',
           'source_sample_id': pid, 'source_split': 'test_fixture', 'language': language,
           'images': [{'image_id': 'img_0', 'path': 'images/diagram.png', 'width': 320, 'height': 200,
                       'sha256': sha_bytes(image.read_bytes())}],
           'raw_question': stem + ' ' + query, 'segments': {'stem': stem, 'query': query, 'options': []},
           'segmentation': {'confidence': 'high', 'method': 'synthetic_fixture', 'review_status': 'approved'}}
    docs = {s: blank_document(s, pid) for s in ('pass1', 'pass2', 'pass3', 'pass4')}
    visuals = []
    for i, typ, box, text in ((1, 'object_shape', [60, 400, 285, 705], ''),
                               (2, 'object_shape', [590, 400, 850, 705], ''),
                               (3, 'text_glyph', [70, 200, 115, 280], 'm')):
        visuals.append({'id': f'v{i:03}', 'type': typ, 'subtype': '', 'text': text, 'image_id': 'img_0',
                        'confidence': 'high', 'bbox_1000': box, 'keypoints_1000': [],
                        'center_1000': [-1, -1], 'radius_1000': -1})
    physical = [{'id': f'p{i:03}', 'type': 'body', 'subtype': '', 'name': f'Block {i}', 'symbol': '',
                 'visual_anchor_ids': [f'v{i:03}'], 'text_mention_ids': [], 'provenance': ['IMAGE'], 'confidence': 'high'} for i in (1, 2)]
    b2 = [{'id': f'b{i:03}', 'type': 'represents', 'from_id': f'v{i:03}', 'to_id': f'p{i:03}',
           'provenance': ['IMAGE'], 'evidence_visual_ids': [f'v{i:03}'], 'evidence_mention_ids': []} for i in (1, 2)]
    mentions = [{'id': f'm{i:03}', 'section': 'query' if i == 4 else 'stem', 'quote': phrase,
                 'occurrence': (2 if i == 3 and language == 'en' else 1),
                 'role': ('entity', 'constraint', 'quantity', 'query_target')[i-1]}
                for i, phrase in enumerate(phrases, 1)]
    # In English literal 'm' occurs in 'mass' before the separate symbol.
    b3 = [{'id': 'b001', 'type': 'refers_to', 'from_id': 'm001', 'to_id': 'p001',
           'provenance': ['TEXT'], 'evidence_visual_ids': [], 'evidence_mention_ids': ['m001']}]
    quantities = [{'id': 'q001', 'kind': 'mass', 'symbol_latex': 'm', 'value_latex': '', 'unit_latex': '',
                   'owner_id': 'p001', 'visual_anchor_ids': ['v003'], 'text_mention_ids': ['m003'], 'provenance': ['IMAGE', 'TEXT']}]
    constraints = [{'id': 'c001', 'kind': 'at_rest', 'subject_ids': ['p001', 'p002'], 'value_text': phrases[1],
                    'provenance': ['TEXT'], 'evidence_visual_ids': [], 'evidence_mention_ids': ['m002']}]
    target = {**docs['pass4']['query_target'], 'mention_id': 'm004', 'target_kind': 'acceleration', 'target_node_ids': ['p001']}
    docs['pass1']['visual_nodes'] = copy.deepcopy(visuals)
    docs['pass2'].update(physical_nodes=copy.deepcopy(physical), bindings=copy.deepcopy(b2))
    docs['pass3'].update(text_mentions=copy.deepcopy(mentions), quantities=copy.deepcopy(quantities),
                         constraints=copy.deepcopy(constraints), bindings=copy.deepcopy(b3), query_target=copy.deepcopy(target))
    docs['pass4'].update(visual_nodes=copy.deepcopy(visuals), physical_nodes=copy.deepcopy(physical),
                         text_mentions=copy.deepcopy(mentions), quantities=copy.deepcopy(quantities),
                         constraints=copy.deepcopy(constraints), bindings=copy.deepcopy(b2)+[{**b3[0], 'id': 'b003'}],
                         query_target=copy.deepcopy(target), relations=[{'id': 'r001', 'predicate': 'touches',
                         'subject_id': 'p001', 'object_id': 'p002', 'quantity_id': '', 'provenance': ['IMAGE'],
                         'evidence_visual_ids': ['v001', 'v002'], 'evidence_mention_ids': [], 'confidence': 'high'}])
    review = {'stages': {}}
    write_json(annotation / 'workspace_config.json', {'dataset_dir': str(dataset)})
    manifest = annotation / 'blind/manifest.jsonl'
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(canonical_bytes(row) + b'\n')
    for stage, doc in docs.items():
        write_json(annotation / 'passes' / stage / (pid + '.json'), doc)
        review['stages'][stage] = {'status': 'approved', 'reviewer': 'SYNTHETIC_TEST_NOT_HUMAN', 'document_sha256': digest(doc)}
    write_json(annotation / 'reviews/state.json', {'problems': {pid: review}})
    return annotation, dataset, pid


def registry(symmetric=False, inverse=False):
    data = copy.deepcopy(config('semantic_registry.example.json'))
    data['example_only'] = False
    data['registry_version'] = 'SYNTHETIC_REGRESSION_NOT_A_REAL_REGISTRY'
    data['roles'] = [{'canonical_id': code, 'label': {'en': code, 'zh': zh}, 'status': 'approved', 'review_ref': None}
                     for code, zh in [('body', '物体'), ('point', '点')]]
    entry = data['relations'][0]
    entry['canonical_id'] = 'touches'
    entry['symmetric'] = symmetric
    entry['subject_types'] = entry['object_types'] = ['body']
    entry['aliases'] = [{'source_namespace': 'SYNTHETIC_UNIT_TEST', 'raw_predicate': 'touches',
                         'endpoint_transform': 'swap' if inverse else 'identity'}]
    entry['public_qualifiers'] = []
    for side in ('object', 'subject'):
        entry['queries'][side] = {'cardinality': 'potentially_many', 'allowed_output': ['one', 'set'],
                                 'required_qualifier_ids': [], 'uniqueness_preconditions': [],
                                 'closure_requirement': 'candidate_scope_complete_review',
                                 'question': {lang: {'one': '{reference}: ' + side + '?',
                                                     'set': '{reference}: ' + side + ' set?'} for lang in ('en', 'zh')}}
    entry['status'] = 'approved'
    ledger = empty_ledger()
    for e in data['roles'] + data['relations']:
        rid = 'semantic-' + entry_root(e)
        e['review_ref'] = rid
        ledger['records'][rid] = review_record(rid, 'semantic_entry', entry_root(e),
                                              'SYNTHETIC_TEST_NOT_HUMAN', ['SYNTHETIC_REGRESSION'])
    return data, ledger
