"""Isolated original-solving and shortcut-control CLI. No paid calls on prepare."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.adapters import ModelRequest
from physalign.cli import _adapter
from physalign.controls import compose, nearest_region, no_image_input, permutation
from physalign.dataset import ImageData, image_metadata, require, sections
from physalign.planning import code_hashes, load_plan
from physalign.runner import run_requests
from physalign.storage import (canonical, confined, digest, file_hash, fingerprint, loads,
                               outside_bundle, read_json, write_new)
from server_eval.supplement_source import (Release, SOLVE_SYSTEM, answer_template, apply_original_catalog,
                                          freeze_keys, source_summary)

SCHEMA = 'physalign_solve_controls_v1'


def companion_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('supplement_eval.py', 'supplement_source.py', 'supplement_reporting.py', 'two_experiments.py')}


def reorder_candidates(inp, seed, iid, index):
    """Intervene ONLY on the textual candidate-list order; aliases/pixels fixed."""
    result = deepcopy(inp)
    parts = sections(inp['messages']['user'])
    evidence = loads(parts['Evidence locations and candidates'])
    aliases = [c['alias'] for c in evidence['candidates']]
    _, order = permutation(aliases, seed=seed, probe_id=iid, index=index, mode='order')
    evidence['candidates'] = [evidence['candidates'][i] for i in order]
    parts['Evidence locations and candidates'] = canonical(evidence)
    result['messages']['user'] = compose(parts)
    return result, [aliases[i] for i in order]


def _new_directory(output, source):
    output = Path(output).resolve()
    outside_bundle(output, source.root)
    require(not source.root.is_relative_to(output), 'Output cannot contain the source release')
    require(not output.exists(), 'Output exists; use a new directory')
    output.mkdir(parents=True)
    return output


def draft(source_root, base_plan, output):
    source = Release(source_root, base_plan)
    originals = source.originals()
    output = _new_directory(output, source)
    write_new(output / 'answer-keys.template.json', answer_template(source, originals))
    write_new(output / 'original-prompts.json', originals)
    summary = source_summary(source, originals)
    write_new(output / 'readiness.json', summary)
    (output / 'solve-system-prompt.txt').write_text(SOLVE_SYSTEM + '\n', encoding='utf-8')
    return summary


def prepare(source_root, base_plan, output, *, experiments=('solve', 'controls'),
            answer_keys=None, original_catalog=None, allow_unscored_solving=False, order_repeats=1, seed=2027, progress=None):
    require(set(experiments) <= {'solve', 'controls'} and experiments, 'Unknown/empty experiment selection')
    require(type(order_repeats) is int and 1 <= order_repeats <= 100, 'Order repeats must be in 1..100')
    source = Release(source_root, base_plan)
    originals = apply_original_catalog(original_catalog, source, source.originals())
    keys = freeze_keys(answer_keys, source, originals)
    if 'solve' in experiments:
        require(any(k is not None for k in keys.values()) or allow_unscored_solving,
                'No reliable original answers: fill draft keys first, or explicitly use --allow-unscored-solving (no SolveAcc)')
    output = _new_directory(output, source)
    images, rows, nearest, excluded, orders, supplement_images = {}, [], {}, {}, {}, {}
    inputs = {(r['instance_id'], r['condition']): deepcopy((source.raw if r['condition'] == 'raw' else source.gold)[r['instance_id']]['input'])
              for r in base_plan['requests']}

    def images_for(inp):
        result = []
        for a in inp['attachments']:
            relative = 'public/' + a['path']
            extra = a['path'].startswith('_supplement_originals/')
            if extra:
                require(original_catalog is not None, 'Extra original image needs a verified catalog')
                path = confined(Path(original_catalog).resolve().parent, a['path'])
            else:
                require(source.inventory.get(relative) == a['sha256'], 'Image absent from source inventory')
                path = confined(source.root, relative)
            if relative not in images:
                data = path.read_bytes()
                require(digest(data) == a['sha256'], 'Source image changed: ' + relative)
                mime, width, height = image_metadata(data)
                require((width, height) == (a['width'], a['height']), 'Source image dimensions changed')
                images[relative] = {'sha256': a['sha256'], 'mime_type': mime, 'width': width,
                                    'height': height, 'byte_count': len(data)}
                if extra:
                    target = confined(output, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
                    require(file_hash(target) == a['sha256'], 'Copied supplemental image changed')
                    supplement_images[relative] = a['sha256']
                if progress and len(images) % 500 == 0:
                    progress({'stage': 'verify_images', 'images': len(images)})
            value = images[relative]
            # Geometry-only heuristic needs dimensions, not another in-memory image copy.
            result.append(ImageData(a['asset_id'], value['mime_type'], value['width'], value['height'], a['sha256'], b''))
        return result

    def add(iid, mother, logical, variant, condition, kind, inp):
        images_for(inp)
        rows.append({'instance_id': iid, 'problem_id': mother, 'logical_probe_id': logical,
                     'condition': variant + '.' + condition, 'variant': variant, 'score_condition': condition,
                     'kind': kind, 'input': inp, 'input_hash': fingerprint(inp)})

    # Verify every frozen reference image even when only original solving is planned.
    for row in base_plan['requests']:
        images_for(inputs[row['instance_id'], row['condition']])
    if 'controls' in experiments:
        for row in base_plan['requests']:
            iid, condition = row['instance_id'], row['condition']
            inp = inputs[iid, condition]
            add(iid, row['problem_id'], row['logical_probe_id'], 'no_image', condition, 'probe', no_image_input(inp))
            for index in range(order_repeats):
                modified, aliases = reorder_candidates(inp, seed, iid, index)
                variant = f'candidate_order_{index:03d}'
                orders.setdefault(variant, {})[iid] = aliases
                add(iid, row['problem_id'], row['logical_probe_id'], variant, condition, 'probe', modified)
        for iid in source.probes:
            declaration, reason = source.nearest_declaration(iid)
            if declaration is None:
                excluded[iid] = reason
            else:
                inp = source.raw[iid]['input']
                nearest[iid] = {**nearest_region(inp, images_for(inp), declaration), 'declaration': declaration}
    if 'solve' in experiments:
        for mother, original in originals.items():
            add('solve:' + mother, mother, 'solve:' + mother, 'solve', 'solve', 'solve', original['input'])

    private = {'probes': source.probes, 'original_keys': keys, 'nearest': nearest,
               'original_sources': {m: {k: v for k, v in x.items() if k != 'input'} for m, x in originals.items()}}
    write_new(output / 'private.json', private)
    main_inputs = [{'instance_id': iid, 'condition': condition, 'input': inp} for (iid, condition), inp in inputs.items()]
    summary = source_summary(source, originals)
    summary.update(answer_eligible_mothers=sum(k is not None for k in keys.values()),
                   human_grade_mothers=sum(k is not None and k['kind'] == 'human' for k in keys.values()),
                   request_counts=dict(Counter(r['condition'] for r in rows)),
                   allow_unscored_solving=allow_unscored_solving)
    payload = {'schema_version': SCHEMA, 'base_plan': base_plan, 'probes': base_plan['probes'],
               'requests': rows, 'reference_inputs': main_inputs, 'experiments': sorted(set(experiments)),
               'images': images, 'supplement_images': supplement_images,
               'source_hint': str(source.root), 'source_identity': source.identity,
               'source_files_used': source.used, 'private_hash': file_hash(output / 'private.json'),
               'scorer_hashes': code_hashes(), 'implementation_hashes': code_hashes(all_modules=True),
               'companion_hashes': companion_hashes(), 'retry_policy': base_plan['retry_policy'],
               'bootstrap': base_plan['bootstrap'], 'weights': {'main': base_plan['weights']},
               'order_seed': seed, 'candidate_orders': orders,
               'control_definitions': {'random': 'analytic uniform alias probability; item then mother then type averaging',
                   'nearest': 'label/quantity visual anchors with every candidate comparable; original-pixel bbox boundary distance; frozen candidate order breaks ties',
                   'no_image': 'all attachments removed; original text, geometry, aliases and any Gold packet retained',
                   'candidate_order': 'textual candidate array shuffled only; aliases, image bytes/order, and all other sections unchanged'},
               'nearest_exclusions': excluded, 'readiness': summary,
               'interpretation': 'Supplement to the frozen all_provisional main plan; not a newly certified unseen test.'}
    plan = {**payload, 'plan_hash': fingerprint(payload)}
    write_new(output / 'plan.json', plan)
    write_new(output / 'readiness.json', summary)
    return plan


class Supplement:
    """Inference opens the public plan and image files only, never private keys."""
    def __init__(self, root, source=None):
        self.root = Path(root).resolve()
        self.plan = read_json(self.root / 'plan.json')
        p = self.plan
        require(p.get('schema_version') == SCHEMA, 'Not a supplemental study')
        require(p['plan_hash'] == fingerprint({k: v for k, v in p.items() if k != 'plan_hash'}), 'Supplement plan changed')
        require(p['implementation_hashes'] == code_hashes(all_modules=True) and p['scorer_hashes'] == code_hashes(), 'Frozen framework changed')
        require(p['companion_hashes'] == companion_hashes(), 'Supplement implementation changed; prepare a new study')
        keys = [(r['instance_id'], r['condition']) for r in p['requests']]
        require(len(keys) == len(set(keys)), 'Duplicate supplement request')
        require(all(r['input_hash'] == fingerprint(r['input']) for r in p['requests']), 'Supplement input hash mismatch')
        self.source = Path(source or p['source_hint']).resolve()

    def images(self, row):
        result = []
        for a in row['input']['attachments']:
            relative = 'public/' + a['path']
            meta = self.plan['images'][relative]
            data = confined(self.root if relative in self.plan['supplement_images'] else self.source, relative).read_bytes()
            require(digest(data) == meta['sha256'] == a['sha256'], 'Frozen image changed')
            require(image_metadata(data) == (meta['mime_type'], a['width'], a['height']) and len(data) == meta['byte_count'], 'Image metadata changed')
            result.append(ImageData(a['asset_id'], meta['mime_type'], a['width'], a['height'], a['sha256'], data))
        return tuple(result)

    def request(self, row, rid, settings):
        messages = row['input']['messages']
        return ModelRequest(rid, messages['system'], messages['user'], self.images(row), settings)

    def snapshot(self, inp, rid, settings):
        images = []
        for a in inp['attachments']:
            meta = self.plan['images']['public/' + a['path']]
            images.append({'asset_id': a['asset_id'], **meta})
        return {'request_id': rid, 'messages': inp['messages'], 'images': images,
                'settings': settings, 'context': 'fresh_stateless_completion'}

    def private(self):
        require(file_hash(self.root / 'private.json') == self.plan['private_hash'], 'Supplement scoring keys changed')
        return read_json(self.root / 'private.json')

    def verify_images(self):
        for relative, meta in self.plan['images'].items():
            root = self.root if relative in self.plan['supplement_images'] else self.source
            require(file_hash(confined(root, relative)) == meta['sha256'], 'Frozen image changed: ' + relative)
        return len(self.plan['images'])


def run(study, adapter, output, *, resume=False, configuration_hash=None, progress=None):
    outside_bundle(Path(output), study.source)
    require(not study.source.is_relative_to(Path(output).resolve()), 'Run output cannot contain source')
    return run_requests(study.plan, adapter, output, study.request, source_root=study.root,
                        resume=resume, adapter_configuration_hash=configuration_hash, progress=progress,
                        protocol_metadata={'supplement_plan_hash': study.plan['plan_hash']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('draft', 'prepare'):
        p = commands.add_parser(command)
        p.add_argument('--source', required=True)
        p.add_argument('--base-plan', required=True)
        p.add_argument('--output', required=True)
        if command == 'prepare':
            p.add_argument('--experiments', nargs='+', choices=('solve', 'controls'), default=['solve', 'controls'])
            p.add_argument('--answer-keys')
            p.add_argument('--originals', help='Verified upstream original catalog from match_original_answers.py')
            p.add_argument('--allow-unscored-solving', action='store_true')
            p.add_argument('--order-repeats', type=int, default=1)
            p.add_argument('--seed', type=int, default=2027)
    for command in ('run', 'score', 'grading-queue'):
        p = commands.add_parser(command)
        p.add_argument('--study', required=True)
        p.add_argument('--source', help='Relocated unchanged source release; image paths stay relative')
        if command == 'run':
            p.add_argument('--output')
            p.add_argument('--adapter', default='smoke')
            p.add_argument('--adapter-config')
            p.add_argument('--resume', action='store_true')
            p.add_argument('--dry-run', action='store_true')
        else:
            p.add_argument('--run', required=True)
            p.add_argument('--output', required=True)
            p.add_argument('--main-run')
            p.add_argument('--grades')
            p.add_argument('--allow-incomplete', action='store_true')
    p = commands.add_parser('compare')
    p.add_argument('--report', action='append', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    log = lambda x: print(canonical(x), flush=True)
    try:
        if args.command in ('draft', 'prepare'):
            base = load_plan(args.base_plan)
            if args.command == 'draft':
                log(draft(args.source, base, args.output))
            else:
                plan = prepare(args.source, base, args.output, experiments=args.experiments,
                               answer_keys=args.answer_keys, original_catalog=args.originals,
                               allow_unscored_solving=args.allow_unscored_solving,
                               order_repeats=args.order_repeats, seed=args.seed, progress=log)
                log({'plan_hash': plan['plan_hash'], **plan['readiness']})
        elif args.command == 'compare':
            from server_eval.supplement_reporting import compare
            log(compare(args.report, args.output))
        else:
            study = Supplement(args.study, args.source)
            if args.command == 'run':
                if args.dry_run:
                    log({'requests': len(study.plan['requests']), 'images_verified': study.verify_images(),
                         'model_loaded': False, 'paid_requests': 0, 'readiness': study.plan['readiness']})
                    return 0
                require(args.output, '--output is required for inference')
                config = read_json(Path(args.adapter_config)) if args.adapter_config else {}
                adapter = _adapter(args.adapter, config)
                log(run(study, adapter, args.output, resume=args.resume,
                        configuration_hash=fingerprint(config), progress=log))
            elif args.command == 'score':
                from server_eval.supplement_reporting import score
                report = score(study, args.run, args.output, main_run=args.main_run, grades_path=args.grades,
                               allow_incomplete=args.allow_incomplete)
                log({'output': args.output, 'service': report['service'], 'original_solving': report['original_solving']})
            else:
                from server_eval.supplement_reporting import grading_queue
                log(grading_queue(study, args.run, args.output))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print('Supplement evaluation: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
