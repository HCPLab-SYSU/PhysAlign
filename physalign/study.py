"""Freeze the five-experiment study into portable public requests + private keys."""

from copy import deepcopy
from pathlib import Path
import random

from .adapters import ModelRequest
from .contracts import encode_probe
from .controls import (nearest_region, no_image_input, original_ids, permutation,
                       permute_input, render_control_input)
from .dataset import ImageData, PublicDataset, image_metadata, load_truth, require
from .planning import code_hashes, prepare_plan, weights_for
from .runner import run_requests
from .storage import (canonical, confined, digest, file_hash, fingerprint, loads,
                      outside_bundle, read_json, write_new)

STUDY_SCHEMA = 'physalign_five_experiments_v1'
SOLVE_SYSTEM = ('Solve the original physics problem. Return a JSON object with an "answer" field '
                'containing your final answer. You may include an "explanation" field. '
                'For multiple-choice problems use the original choice identifiers in "answer".')


def draft_spec(dataset_root):
    """A reviewable draft; NEVER invents original answers or human approvals."""
    ds = PublicDataset(dataset_root)
    ds.verify_inventory()
    probes, meta = load_truth(ds)
    originals, controls = {}, {}
    native_originals = {}
    if 'public/problems.json' in ds.inventory:
        for r in ds.read_verified('public/problems.json'):
            require(r['problem_id'] not in native_originals, 'Duplicate original problem')
            native_originals[r['problem_id']] = r
    for iid, p in probes.items():
        item = ds.items[iid, 'raw']
        if p.problem_id not in originals:
            relative = f'private/source/{p.problem_id}/manifest.json'
            original = None
            if relative in ds.inventory:
                source = ds.read_verified(relative)
                question = source['raw_question']
                options = source['segments']['options']
                if options and not all(o['text'] in question for o in options):
                    question += '\n[Original answer options]\n' + canonical(options)
                original = {'system': SOLVE_SYSTEM, 'user': question,
                            'asset_ids': [a['image_id'] for a in source['images']],
                            'source_reference': relative, 'unchanged_original_verified': True}
            elif p.problem_id in native_originals:
                r = native_originals[p.problem_id]
                original = {'system': SOLVE_SYSTEM, 'user': r['user'], 'asset_ids': r['asset_ids'],
                            'source_reference': 'public/problems.json', 'unchanged_original_verified': True}
            originals[p.problem_id] = {'original': original, 'answer_key': None}
        nearest = None
        if ds.manifest['schema_version'] == 'physalign_eval_starter_samples_v1':
            m = ds.read_verified(f'private/evidence/{iid}/probe_metadata.json')
            if m['rule_family'] == 'quantity' and p.binding.kind == 'single' and len(p.readings) == 1:
                nearest = {'field': next(iter(p.binding.domains)), 'anchor_id': p.readings[0].anchor_id,
                           'eligibility_basis': 'approved_quantity_ownership_probe'}
        controls[iid] = {'nearest': nearest, 'no_geometry_compatible': False,
                         'no_geometry_basis': None, 'panel_count': None}
    return {'schema_version': 'physalign_study_spec_v1', 'original_problems': originals,
            'controls': controls, 'permutations_per_mode': 1, 'permutation_seed': 2027,
            'human_sample_mothers': None, 'human_seed': 2027,
            'note': 'Fill reliable original answer keys; inspect control eligibility. No human audit has been performed.'}


def prepare_study(dataset_root, output, *, split, spec_path=None, bootstrap_resamples=2000,
                  seed=2027, max_retries=2, problem_ids=None, reservation_files=()):
    base = prepare_plan(dataset_root, split=split, bootstrap_resamples=bootstrap_resamples,
                        bootstrap_seed=seed, max_retries=max_retries, problem_ids=problem_ids,
                        reservation_files=reservation_files)
    ds = PublicDataset(dataset_root)
    truth, metadata = load_truth(ds)
    selected = [truth[r['instance_id']] for r in base['probes']]
    selected_ids = {p.probe_id for p in selected}
    mothers = sorted({p.problem_id for p in selected})
    spec = read_json(Path(spec_path)) if spec_path else draft_spec(dataset_root)
    require(spec.get('schema_version') == 'physalign_study_spec_v1', 'Unknown study specification')
    require(selected_ids <= spec['controls'].keys() and set(mothers) <= spec['original_problems'].keys(), 'Study specification does not cover selected probes/mothers')
    n = spec['permutations_per_mode']
    require(type(n) is int and 1 <= n <= 100, 'Predeclare 1..100 permutations per mode')
    require(type(spec['permutation_seed']) is int and type(spec['human_seed']) is int, 'Seeds must be integers')
    if split == 'test':
        require(len(base['paired_ids']) == len(selected), 'Formal full-B Gold experiment requires an exported Gold view for EVERY binding probe, including audited empty packets')
    directory = Path(output).resolve()
    outside_bundle(directory, ds.root)
    require(not directory.exists(), 'Study output already exists; prepare into a new directory')
    # All semantic preparation happens before inference; model code never reads this.
    private = {'probes': {p.probe_id: encode_probe(p) for p in selected}, 'spec': spec,
               'nearest': {}, 'original_keys': {m: spec['original_problems'][m]['answer_key'] for m in mothers}}
    from .solving import validate_answer_key
    for key in private['original_keys'].values():
        if key is not None:
            validate_answer_key(key)
    directory.mkdir(parents=True)
    images_dir = directory / 'images'
    images_dir.mkdir()
    image_inventory = {}
    def store_image(a):
        name = a.sha256 + ('.png' if a.mime_type == 'image/png' else '.jpg')
        path = images_dir / name
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(a.data)
        require(file_hash(path) == a.sha256, 'Image content address collision')
        image_inventory['images/' + name] = a.sha256
        return {'asset_id': a.asset_id, 'path': 'images/' + name, 'sha256': a.sha256,
                'mime_type': a.mime_type, 'width': a.width, 'height': a.height}
    rows, groups, permutations, renderer = [], {}, {}, None
    originals_by_mother = {}
    def add(iid, mother, logical, variant, condition, inp, kind='probe'):
        row = {'instance_id': iid, 'problem_id': mother, 'logical_probe_id': logical,
               'condition': variant + '.' + condition, 'variant': variant, 'score_condition': condition,
               'kind': kind, 'input': inp, 'input_hash': fingerprint(inp)}
        rows.append(row)
        if kind == 'probe':
            groups.setdefault(variant, {}).setdefault(condition, []).append(iid)
    for p in selected:
        iid, mother = p.probe_id, p.problem_id
        raw_item = ds.items[iid, 'raw']
        raw_images = ds.images(raw_item)
        ids = original_ids(raw_item.record['input']['messages']['user'])
        source_images = tuple(next(a for a in raw_images if a.asset_id == name) for name in ids)
        identity = [(a.asset_id, a.sha256) for a in source_images]
        if mother in originals_by_mother:
            require(identity == [(a.asset_id, a.sha256) for a in originals_by_mother[mother]], 'Probes of one mother disagree on original image identities/order')
        originals_by_mother[mother] = source_images
        declaration = spec['controls'][iid]
        require(type(declaration['no_geometry_compatible']) is bool, 'No-geometry eligibility must be explicit')
        if declaration['no_geometry_compatible']:
            require(isinstance(declaration.get('no_geometry_basis'), str) and declaration['no_geometry_basis'].strip(), 'No-geometry subset requires an a-priori interface justification')
        if declaration['nearest'] is not None:
            d = declaration['nearest']
            require(p.binding.kind == 'single' and d['field'] in p.binding.domains and bool(d['eligibility_basis']), 'Nearest baseline only supports declared single-target ownership tasks')
            private['nearest'][iid] = nearest_region(raw_item.record['input'], raw_images, d)
        for condition in ('raw', 'gold'):
            if (iid, condition) not in ds.items:
                continue
            inp = deepcopy(ds.items[iid, condition].record['input'])
            inp['attachments'] = [store_image(a) for a in ds.images(ds.items[iid, condition])]
            logical = metadata[iid]['logical_probe_id']
            add(iid, mother, logical, 'main', condition, inp)
            if condition == 'raw':
                add(iid, mother, logical, 'no_image_geometry', 'raw', no_image_input(inp))
                if declaration['no_geometry_compatible']:
                    add(iid, mother, logical, 'no_image_no_geometry', 'raw', no_image_input(inp, remove_geometry=True))
            reference, renderer = render_control_input(inp, source_images, store_image)
            add(iid, mother, logical, 'permutation_reference', condition, reference)
            from .dataset import sections
            candidates = loads(sections(inp['messages']['user'])['Evidence locations and candidates'])['candidates']
            aliases = [c['alias'] for c in candidates]
            for mode in ('alias', 'order', 'both'):
                for index in range(n):
                    variant = f'perm_{mode}_{index:03d}'
                    mapping, order = permutation(aliases, seed=spec['permutation_seed'], probe_id=iid, index=index, mode=mode)
                    permutations.setdefault(variant, {})[iid] = {'old_to_new': mapping, 'order': order}
                    modified, _ = render_control_input(permute_input(inp, mapping, order), source_images, store_image)
                    add(iid, mother, logical, variant, condition, modified)
    missing_originals = []
    for mother in mothers:
        original = spec['original_problems'][mother]['original']
        if original is None:
            missing_originals.append(mother)
            continue
        require(original['unchanged_original_verified'] is True and bool(original['source_reference']), 'Original solving needs verified unchanged source input')
        require(all(isinstance(original[k], str) and original[k] for k in ('system', 'user')), 'Empty original input')
        assets = originals_by_mother[mother]
        require(original['asset_ids'] == [a.asset_id for a in assets], 'Original solve must use all original images, in source order, without probe overlays')
        add('solve:' + mother, mother, 'original:' + mother, 'solve', 'solve',
            {'messages': {'system': original['system'], 'user': original['user']},
             'attachments': [store_image(a) for a in assets]}, 'solve')
    group_weights = {}
    for variant, conditions in groups.items():
        ids = set(conditions['raw'])
        pairs = ids & set(conditions.get('gold', []))
        group_weights[variant] = {'full': weights_for([p for p in selected if p.probe_id in ids]),
                                  'paired': weights_for([p for p in selected if p.probe_id in pairs])}
    nearest_ids = sorted(private['nearest'])
    group_weights['nearest'] = weights_for([p for p in selected if p.probe_id in nearest_ids])
    # Strata and all subset membership are defined before any model outputs.
    strata = {}
    for p in selected:
        iid = p.probe_id
        from .dataset import sections
        count = len(loads(sections(ds.items[iid, 'raw'].record['input']['messages']['user'])['Evidence locations and candidates'])['candidates'])
        panel_count = spec['controls'][iid].get('panel_count')
        require(panel_count is None or type(panel_count) is int and panel_count >= 1, 'Panel count must be annotated, never inferred from image count')
        distance = private['nearest'].get(iid, {}).get('distance_pixels')
        band = ('not_applicable' if distance is None else '0' if distance == 0 else
                '(0,10]' if distance <= 10 else '(10,50]' if distance <= 50 else '(50,inf)')
        strata[iid] = {'candidate_count': count, 'panel_count': panel_count,
                       'nearest_distance_pixels': distance, 'nearest_distance_band': band}
    human_mothers = spec.get('human_sample_mothers')
    if human_mothers is None:
        human_mothers = random.Random(spec['human_seed']).sample(mothers, min(20, len(mothers)))
    require(isinstance(human_mothers, list) and human_mothers and len(human_mothers) == len(set(human_mothers)) and set(human_mothers) <= set(mothers), 'Human sample must be nonempty predeclared mother IDs')
    write_new(directory / 'private.json', private)
    payload = {'schema_version': STUDY_SCHEMA, 'base_plan': base, 'probes': base['probes'], 'requests': rows,
               'groups': groups, 'weights': group_weights, 'permutations': permutations,
               'nearest_ids': nearest_ids, 'strata': strata, 'renderer': renderer,
               'distance_stratification': {'unit': 'original_image_pixels', 'boundaries': [0, 10, 50],
                    'definition': 'minimum anchor-to-candidate boundary separation; no gold identity used'},
               'human_mothers': sorted(human_mothers), 'human_seed': spec['human_seed'],
               'retry_policy': base['retry_policy'], 'bootstrap': base['bootstrap'],
               'private_hash': file_hash(directory / 'private.json'), 'images': image_inventory,
               'scorer_hashes': code_hashes(), 'implementation_hashes': code_hashes(all_modules=True),
               'readiness': {'missing_original_inputs': missing_originals,
                             'missing_original_answer_keys': [m for m in mothers if private['original_keys'][m] is None],
                             'missing_gold_views': base['raw_only_ids'], 'human_audit': 'not_measured',
                             'requires_final_interface_audit': True},
               'model_panel': ['Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-27B', 'OpenGVLab/InternVL3_5-8B-HF']}
    plan = {**payload, 'plan_hash': fingerprint(payload)}
    write_new(directory / 'plan.json', plan)
    return plan


class Study:
    """Public loading path: does not open private.json."""
    def __init__(self, root, *, check_code=True):
        self.root = Path(root).resolve()
        self.plan = read_json(self.root / 'plan.json')
        p = self.plan
        require(p.get('schema_version') == STUDY_SCHEMA, 'Unknown study schema')
        require(p['plan_hash'] == fingerprint({k: v for k, v in p.items() if k != 'plan_hash'}), 'Study plan changed')
        if check_code:
            require(p['scorer_hashes'] == code_hashes(), 'Scoring code changed after study freeze')
            require(p['implementation_hashes'] == code_hashes(all_modules=True), 'Framework changed after study freeze; prepare a new study')
        keys = [(r['instance_id'], r['condition']) for r in p['requests']]
        require(len(keys) == len(set(keys)), 'Duplicate study request')
        for row in p['requests']:
            require(fingerprint(row['input']) == row['input_hash'], 'Study public input changed')
        self.by_key = dict(zip(keys, p['requests']))

    def images(self, row):
        images = []
        for a in row['input']['attachments']:
            require(self.plan['images'].get(a['path']) == a['sha256'], 'Image absent from frozen study')
            data = confined(self.root, a['path']).read_bytes()
            require(digest(data) == a['sha256'], 'Study image changed')
            mime, w, h = image_metadata(data)
            require((mime, w, h) == (a['mime_type'], a['width'], a['height']), 'Study image metadata mismatch')
            images.append(ImageData(a['asset_id'], mime, w, h, a['sha256'], data))
        return tuple(images)

    def request(self, row, request_id, settings):
        m = row['input']['messages']
        return ModelRequest(request_id, m['system'], m['user'], self.images(row), settings)

    def private(self):
        require(file_hash(self.root / 'private.json') == self.plan['private_hash'], 'Study private truth changed')
        return read_json(self.root / 'private.json')


def run_study(study_root, adapter, output, *, resume=False, progress=None, configuration_hash=None):
    study = Study(study_root)
    seal = verify_final_seal(study)
    return run_requests(study.plan, adapter, output, study.request, source_root=study.root,
                        resume=resume, progress=progress, adapter_configuration_hash=configuration_hash,
                        protocol_metadata={'study_seal_hash': seal['seal_hash'] if seal else None})


def verify_final_seal(study):
    final = study.plan['base_plan']['split'] == 'test'
    if final:
        require(not study.plan['readiness']['missing_original_inputs'], 'Final study has missing original inputs')
        require(len(study.plan['readiness']['missing_original_answer_keys']) < len({p['problem_id'] for p in study.plan['probes']}), 'No reliable original answers are available for the required solving experiment')
    path = study.root / 'seal.json'
    if not path.is_file():
        require(not final, 'Final test requires completed same-interface human evidence and seal-study before model runs')
        return None
    seal = read_json(path)
    require(seal['plan_hash'] == study.plan['plan_hash'] and seal['seal_hash'] == fingerprint({k: v for k, v in seal.items() if k != 'seal_hash'}), 'Final human evidence seal changed')
    report = seal['human_report']
    require(report['audit_complete_and_accepted'] and report['complete_answer_coverage'], 'Incomplete human evidence cannot seal a final study')
    return seal
