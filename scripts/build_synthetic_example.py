"""Create six artificial contract fixtures and three programmatically drawn images.

No benchmark content, real review evidence, model output or credentials are read.
The approval-shaped records are explicit test stubs for the legacy bundle loader.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
MARKERS = ('Original statement', 'Original question', 'Original answer options',
           'Evidence locations and candidates', 'Images and locator views',
           'Local question', 'Output format', 'Local reading information')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def diagram(number):
    """A small metadata-free PNG: two boxes and a bitmap mass label."""
    width, height = 200, 140
    pixels = bytearray([255] * (width * height * 3))

    def rect(x0, y0, x1, y1, color):
        for y in range(y0, y1):
            for x in range(x0, x1):
                offset = (y * width + x) * 3
                pixels[offset:offset + 3] = bytes(color)

    rect(20, 42, 76, 112, (40, 100 + number * 15, 210))
    rect(120, 42, 176, 112, (235, 145, 40 + number * 15))
    glyphs = {'2': ('111', '001', '111', '100', '111'),
              '3': ('111', '001', '111', '001', '111'),
              '4': ('101', '101', '111', '001', '001'),
              'k': ('100', '101', '110', '101', '101'),
              'g': ('111', '101', '111', '001', '111'),
              ' ': ('000',) * 5}
    for i, character in enumerate(f'{number + 1} kg'):
        for y, row in enumerate(glyphs[character]):
            for x, filled in enumerate(row):
                if filled == '1':
                    rect(20 + i * 12 + x * 3, 17 + y * 3,
                         23 + i * 12 + x * 3, 20 + y * 3, (0, 0, 0))

    def chunk(kind, value):
        return struct.pack('>I', len(value)) + kind + value + struct.pack('>I', zlib.crc32(kind + value) & 0xffffffff)

    rows = b''.join(b'\x00' + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    data = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
    return data, width, height


def build(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('Choose a new output directory; existing files are never overwritten')
    destination.mkdir(parents=True)

    def write(relative, value):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical(value) + '\n', encoding='utf-8')

    note = 'SYNTHETIC TEST FIXTURE. Approval records are programmatic stubs, not human or model reviews.'
    write('SYNTHETIC_ONLY.json', {'synthetic': True, 'real_data_included': False, 'review_evidence': 'test_stubs_only', 'note': note})
    assets = {}
    mothers = [f'synthetic_mother_{i:02d}' for i in range(1, 4)]
    for index, mother in enumerate(mothers, 1):
        data, width, height = diagram(index)
        relative = f'images/{mother}.png'
        path = destination / 'public' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        assets[mother] = {'asset_id': 'img_0', 'path': relative, 'sha256': digest(data), 'width': width, 'height': height}
        write(f'private/source/{mother}/manifest.json', {
            'problem_id': mother, 'raw_question': f'SYNTHETIC ORIGINAL: Which block is blue in diagram {index}?',
            'segments': {'options': []}, 'source_dataset': 'SYNTHETIC_TEST_ONLY', 'source_sample_id': mother,
            'images': [{'image_id': 'img_0', 'sha256': digest(data), 'width': width, 'height': height}],
            'synthetic': True})

    raw, gold, answers, members = [], [], [], []
    for index in range(1, 7):
        reading = index > 3
        mother_index = (index - 1) % 3
        mother, iid = mothers[mother_index], f'sample_{index:02d}'
        logical = f'synthetic_logical_{index:02d}'
        task, interface, field = ('T03', 'T03-image', 'owner') if reading else ('T02', 'T02-single', 'referent')
        label = f'{mother_index + 2} kg'
        left = {'kind': 'visual', 'image_id': 'img_0', 'geometry': {'type': 'bbox', 'bbox_1000': [100, 300, 380, 800]}}
        right = {'kind': 'visual', 'image_id': 'img_0', 'geometry': {'type': 'bbox', 'bbox_1000': [600, 300, 880, 800]}}
        anchor = {'kind': 'visual', 'image_id': 'img_0', 'geometry': {'type': 'bbox', 'bbox_1000': [100, 120, 340, 240]}}
        evidence = {'anchors': {'R1': anchor if reading else left},
                    'candidates': [{'alias': 'E2', 'locators': [left]}, {'alias': 'E1', 'locators': [right]}]}
        output = {field: '<one candidate ID>'}
        answer = {field: 'E2'}
        if reading:
            output['read'], answer['read'] = '<literal transcription>', label
        bodies = [f'Synthetic diagram {mother_index + 1}: a blue block and an orange block.',
                  'Which block is blue?', '[]', canonical(evidence), 'Original image IDs: img_0',
                  (f'Read the mass at R1 and identify its block. {iid}' if reading else f'Identify the blue block indicated by R1. {iid}'),
                  canonical(output), 'None.']
        user = '\n'.join(f'[{name}]\n{body}' for name, body in zip(MARKERS, bodies))
        row = {'schema_version': 'physalign_public_qa_v1', 'instance_id': iid,
               'logical_probe_id': logical, 'task_id': task, 'interface': interface, 'language': 'en',
               'split': 'framework_development', 'condition': 'raw',
               'input': {'messages': {'system': 'Synthetic contract example. Return only the requested JSON.', 'user': user},
                         'attachments': [assets[mother]]}}
        raw.append(row)
        if reading:
            paired = deepcopy(row)
            paired['condition'] = 'gold'
            paired['input']['messages']['user'] = user.removesuffix('None.') + canonical([{'anchor_id': 'R1', 'text': label}])
            gold.append(paired)
        target_root = digest(('synthetic-target-' + iid).encode())
        blind_root, audit_root = digest(('synthetic-blind-' + iid).encode()), digest(('synthetic-audit-' + iid).encode())
        member = {'instance_id': iid, 'logical_probe_id': logical, 'problem_id': mother, 'interface': interface,
                  'split': 'framework_development', 'review_status': 'model_approved', 'human_reviewed': False,
                  'review_target_root': target_root, 'synthetic': True}
        members.append(member)
        answers.append({'instance_id': iid, 'logical_probe_id': logical, 'interface': interface,
                        'candidate_ids': ['E2', 'E1'], 'answer': answer, 'review_status': 'model_approved',
                        'human_reviewed': False, 'read_normalizer_id': 'ocr_label_v2_1' if reading else None,
                        'synthetic': True})
        key = {'schema_version': 'physalign_coverage_expansion_v1', 'logical_probe_id': logical,
               'problem_id': mother, 'human_reviewed': False,
               'response_contract': {'cardinality': 'one', 'binding_key': field},
               'candidate_map': {alias: {'kind': 'object', 'id': mother + suffix}
                                 for alias, suffix in [('E2', '_blue'), ('E1', '_orange')]},
               'gold_targets': [{'kind': 'object', 'id': mother + '_blue'}],
               'read_targets': [{'field': 'read', 'anchor_alias': 'R1', 'expected': label, 'normalizer_id': 'ocr_label_v2_1'}] if reading else [],
               'packet_plan': {'state': 'nonempty' if reading else 'empty'},
               'source_query': {'kind': 'quantity' if reading else 'object', 'locator': anchor if reading else left},
               'synthetic': True, 'note': note}
        base = f'private/evidence/{iid}/'
        write(base + 'private_key.json', key)
        write(base + 'result.json', {'logical_probe_id': logical, 'problem_id': mother, 'status': 'model_approved',
              'human_reviewed': False, 'review_target_root': target_root, 'flags': [], 'audit_root': audit_root,
              'blind_root': blind_root, 'synthetic': True, 'note': note})
        write(base + 'audit.json', {'record_root': audit_root, 'response': {'verdict': 'pass',
              'checks': {'packet_no_binding_leakage': {'status': 'pass'}}}, 'synthetic': True, 'note': note})
        write(base + 'blind.json', {'record_root': blind_root, 'response': {'ambiguous': False}, 'synthetic': True, 'note': note})
        write(base + 'probe_metadata.json', {'rule_family': 'quantity' if reading else 'reference', 'synthetic': True})
    write('public/qa_raw.json', raw)
    write('public/qa_gold.json', gold)
    write('private/answers.json', answers)
    write('private/membership.json', members)
    write('DEVELOPMENT_RESERVATION.json', {'problem_ids': mothers, 'cluster_ids': mothers, 'synthetic': True})
    write('manifest.json', {'schema_version': 'physalign_eval_starter_samples_v1', 'image_paths_relative_to': 'public',
          'formal_release_eligibility_checked': False, 'synthetic_example_only': True, 'human_reviewed': False,
          'counts': {'raw': 6, 'gold': 3, 'mothers': 3, 'probes': 6}, 'note': note})
    reference = destination / 'reference/ocr_normalizers.py'
    reference.parent.mkdir(parents=True)
    reference.write_bytes((ROOT / 'physalign/ocr_normalizers.py').read_bytes())
    write('FILE_MANIFEST.json', {p.relative_to(destination).as_posix(): digest(p.read_bytes())
                               for p in sorted(destination.rglob('*')) if p.is_file()})
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'examples/synthetic')
    args = parser.parse_args()
    print(build(args.output))
