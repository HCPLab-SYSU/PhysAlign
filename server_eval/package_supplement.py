"""Copy a frozen supplement and its answer references into a portable data folder.

This does not change plans, load adapters, run models, or write existing datasets.
The package is for supplement_eval.py, not a replacement release for evaluate.py.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.dataset import require
from physalign.storage import canonical, confined, file_hash, fingerprint, loads, read_json, write_new
from server_eval.supplement_eval import Supplement


def package(study, source, originals, output):
    study, source, originals, output = [Path(p).resolve() for p in (study, source, originals, output)]
    for origin in (study, source, originals):
        require(not output.is_relative_to(origin) and not origin.is_relative_to(output),
                'Package and input directories must be separate')
    require(not output.exists(), 'Package already exists; use a new directory')
    supplement = Supplement(study, source)
    plan, private = supplement.plan, supplement.private()
    catalog = read_json(originals / 'originals.json')
    keys = read_json(originals / 'answer-keys.json')
    require(catalog['catalog_hash'] == fingerprint({k: v for k, v in catalog.items() if k != 'catalog_hash'}),
            'Original catalog changed')
    require(catalog['source_identity'] == keys['source_identity'] == plan['source_identity'],
            'Original catalog, keys and frozen study have different source identities')
    require(file_hash(source / 'release.json') == plan['source_identity']['release_sha256']
            and file_hash(source / 'FILE_MANIFEST.json') == plan['source_identity']['inventory_sha256'],
            'Source release identity changed')
    solve = {r['problem_id']: r for r in plan['requests'] if r['variant'] == 'solve'}
    require(solve and solve.keys() == catalog['problems'].keys() == keys['problems'].keys()
            == private['original_keys'].keys(), 'Incomplete mother coverage')
    public_rows, private_rows = [], []
    for mother, row in sorted(solve.items()):
        original, key_row = catalog['problems'][mother], keys['problems'][mother]
        key = private['original_keys'][mother]
        require(original['match_status'] == 'matched', 'Unmatched mother: ' + mother)
        require(original['input'] == row['input'] and key_row['input_hash'] == row['input_hash'],
                'Original question differs from frozen request: ' + mother)
        require(key is not None and key['reliable'] and key == key_row['answer_key'],
                'Missing or changed frozen answer: ' + mother)
        for attachment in row['input']['attachments']:
            relative = 'public/' + attachment['path']
            require(plan['images'][relative]['sha256'] == attachment['sha256'], 'Unfrozen question image')
        public_row = {'problem_id': mother, 'source_dataset': key_row['source_dataset'],
                      'source_sample_id': key_row['source_sample_id'], 'input_hash': row['input_hash'],
                      'input': row['input']}
        public_rows.append(public_row)
        reference = loads(key['reference_answer']) if key['kind'] == 'human' else {'accepted': key['accepted']}
        private_rows.append({**public_row, 'answer_key': key, 'reference_answer': reference,
                             'provenance': original['provenance']})

    # Preflight available space before creating the destination. Copy actual bytes;
    # no links back into files used by ongoing evaluations.
    copy_jobs = []
    def schedule(origin, relative, expected=None):
        require(origin.is_file(), 'Missing source file: ' + str(origin))
        confined(output, relative)
        copy_jobs.append((origin, relative, expected or file_hash(origin)))

    for name in ('plan.json', 'private.json', 'readiness.json'):
        schedule(study / name, name)
    for relative, meta in plan['images'].items():
        origin = study if relative in plan['supplement_images'] else source
        schedule(confined(origin, relative), relative, meta['sha256'])
    for relative, expected in plan['source_files_used'].items():
        schedule(confined(source, relative), 'provenance/release/' + relative, expected)
    schedule(source / 'FILE_MANIFEST.json', 'provenance/release/FILE_MANIFEST.json',
             plan['source_identity']['inventory_sha256'])
    for name in ('originals.json', 'answer-keys.json', 'matches.json', 'matches.csv', 'summary.json', 'upstream-snapshot.json'):
        schedule(originals / name, 'originals/' + name)
    for relative, expected in plan['supplement_images'].items():
        catalog_relative = relative.removeprefix('public/')
        schedule(confined(originals, catalog_relative), 'originals/' + catalog_relative, expected)
    schedule(ROOT / 'docs' / 'SUPPLEMENT_UNIFIED_DATA.md', 'README.md')
    schedule(ROOT / 'docs' / 'SUPPLEMENT_SOLVING_CONTROLS.md', 'protocol.md')
    require(len({relative for _, relative, _ in copy_jobs}) == len(copy_jobs), 'Duplicate package file')
    required = sum(path.stat().st_size for path, _, _ in copy_jobs)
    free = shutil.disk_usage(next(p for p in (output.parent, *output.parents) if p.exists())).free
    require(free > required + 100 * 1024**2, 'Not enough free space for an independent copy')

    output.mkdir(parents=True)
    inventory = {}
    for i, (origin, relative, expected) in enumerate(copy_jobs, 1):
        target = confined(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental overwriting even after preflight.
        with origin.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        require(file_hash(target) == expected, 'Source changed or copy failed: ' + relative)
        inventory[relative] = expected
        if i % 1000 == 0:
            print(canonical({'stage': 'package_copy', 'copied': i, 'total': len(copy_jobs)}), flush=True)
    for relative, rows in (('public/mothers.jsonl', public_rows), ('private/mothers-with-answers.jsonl', private_rows)):
        target = confined(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(canonical(row) + '\n')
        inventory[relative] = file_hash(target)
    prompts = {row['input']['messages']['system'] for row in public_rows}
    require(len(prompts) == 1, 'Expected one original-solving system prompt')
    prompt_path = output / 'solve-system-prompt.txt'
    with prompt_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(next(iter(prompts)) + '\n')
    inventory[prompt_path.name] = file_hash(prompt_path)

    # Explicit source override proves all runtime images resolve inside the new
    # package; retain the original frozen plan bytes/hash for run comparability.
    relocated = Supplement(output, output)
    relocated.verify_images()
    require(relocated.private() == private, 'Packaged private keys changed')
    summary = {'schema_version': 'physalign_supplement_package_v1', 'plan_hash': plan['plan_hash'],
               'source_identity': plan['source_identity'], 'mothers': len(solve),
               'grading_kinds': dict(Counter(row['answer_key']['kind'] for row in private_rows)),
               'images': len(plan['images']), 'requests': len(plan['requests']),
               'request_counts': dict(Counter(r['condition'] for r in plan['requests'])),
               'runtime_source': '.', 'all_images_verified': True,
               'release_provenance_is_partial_archive': True,
               'packager_sha256': file_hash(Path(__file__)),
               'files': inventory}
    # Written last: its absence identifies an incomplete packaging attempt.
    write_new(output / 'PACKAGE_MANIFEST.json', summary)
    return {k: v for k, v in summary.items() if k != 'files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'originals', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(canonical(package(args.study, args.source, args.originals, args.output)), flush=True)


if __name__ == '__main__':
    main()
