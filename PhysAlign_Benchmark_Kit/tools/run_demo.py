"""Create a clearly synthetic three-interface demo. Never reads real datasets."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'physalign_converter'))
sys.path.insert(0, str(KIT / 'physalign_converter/tests'))
from fixtures import workspace
from io_utils import read_json, write_json
from reference_checks import digest, require
from convert import main as convert_main


def make_demo(output):
    output = Path(output).resolve()
    require(not output.exists(), 'DEMO_OUTPUT_EXISTS_CHOOSE_NEW_DIRECTORY', str(output))
    output.mkdir(parents=True)
    annotation, dataset, pid = workspace(output / 'source', language='en')
    # Extend only our NEW artificial fixture with one text-only quantity, so the
    # demo exercises T02-single, T03-image and T03-text in one review page.
    manifest = annotation / 'blind/manifest.jsonl'
    row = read_json(manifest)
    row['segments']['stem'] += ' The mass of block B is 2 kg.'
    row['raw_question'] = row['segments']['stem'] + ' ' + row['segments']['query']
    write_json(manifest, row)
    state_path = annotation / 'reviews/state.json'
    state = read_json(state_path)
    for stage in ('pass3', 'pass4'):
        path = annotation / 'passes' / stage / (pid + '.json')
        doc = read_json(path)
        doc['text_mentions'][-1]['id'] = 'm005'
        doc['query_target']['mention_id'] = 'm005'
        doc['text_mentions'].insert(-1, {'id': 'm004', 'section': 'stem', 'quote': '2 kg',
                                        'occurrence': 1, 'role': 'quantity'})
        doc['quantities'].append({'id': 'q002', 'kind': 'mass', 'symbol_latex': 'm_B',
            'value_latex': '2', 'unit_latex': 'kg', 'owner_id': 'p002', 'visual_anchor_ids': [],
            'text_mention_ids': ['m004'], 'provenance': ['TEXT']})
        write_json(path, doc)
        state['problems'][pid]['stages'][stage]['document_sha256'] = digest(doc)
    write_json(state_path, state)
    # Do not leak the builder's machine path. Real-workflow CLI receives the
    # explicit --dataset-root override; source config is intentionally relative.
    write_json(annotation / 'workspace_config.json', {'dataset_dir': '../dataset',
               'demo_notice': 'SYNTHETIC_ONLY: use --dataset-root; not human-reviewed real data'})
    write_json(output / 'SYNTHETIC_ONLY.json', {'synthetic': True, 'problem_ids': [pid],
        'source_namespace': 'SYNTHETIC_UNIT_TEST',
        'warning': 'Artificial drawing, text and test approval markers. NOT benchmark data or human audit evidence.'})
    convert_main(['native-draft', '--workspace', str(annotation), '--dataset-root', str(dataset),
                  '--output', str(output / 'draft'), '--development-count', '25'])
    convert_main(['native-validate', '--path', str(output / 'draft')])
    report = read_json(output / 'draft/report.json')
    require({k: v['probes'] for k, v in report['interfaces'].items()} ==
            {'T02-single': 1, 'T03-image': 1, 'T03-text': 1}, 'DEMO_INTERFACE_COUNTS_CHANGED')
    print('SYNTHETIC DEMO ONLY. Open: ' + str(output / 'draft/review.html'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='new output directory; never overwrite')
    make_demo(parser.parse_args().output)


if __name__ == '__main__':
    main()
