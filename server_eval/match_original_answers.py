"""Recover original-task references from pinned public benchmark snapshots.

Fetches public data only. It never uses GPT/Gemini keys, evaluates a model, or
modifies the main dataset. Unknown joins fail closed instead of using fuzzy text.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physalign.dataset import image_metadata, require
from physalign.planning import load_plan
from physalign.storage import canonical, confined, digest, file_hash, fingerprint, read_json, write_new
from server_eval.supplement_source import Release, SOLVE_SYSTEM, answer_template

DATASETS = {
    'live2603': ('Shawn-wxh/livek12bench', 'default', 'en_2603', 1487),
    'live2605': ('Shawn-wxh/livek12bench', 'default', 'en_2605', 627),
    'seephys': ('SeePhys/SeePhys', 'default', 'train', 2000),
    'olympiad': ('Hothan/OlympiadBench', 'OE_MM_physics_en_COMP', 'train', 456),
    # Selected PhysElite IDs are <=923; the viewer stores IDs 1..1000 in these pages.
    'physelite': ('physelite/PhysElite', 'default', 'default', 1000),
}


def public_get(url):
    host = urlsplit(url).hostname
    require(url.startswith('https://') and host in {'huggingface.co', 'datasets-server.huggingface.co'}, 'Unexpected upstream host')
    last = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'PhysAlign-answer-audit/1'}), timeout=30) as response:
                return response.read()
        except OSError as error:
            last = error
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    raise last


def fetch(cache):
    """Sequential modest-rate reads; no dependency changes in active environments."""
    import json
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    for repo in sorted({v[0] for v in DATASETS.values()}):
        path = cache / (repo.replace('/', '--') + '-info.json')
        if not path.exists():
            data = public_get('https://huggingface.co/api/datasets/' + repo)
            require(json.loads(data).get('sha'), 'Upstream revision is missing')
            path.write_bytes(data)
    for name, (repo, config, split, extent) in DATASETS.items():
        for offset in range(0, extent, 100):
            path = cache / f'{name}-rows-{offset}.json'
            if path.exists():
                continue
            params = {'dataset': repo, 'config': config, 'split': split, 'offset': offset, 'length': 100}
            data = public_get('https://datasets-server.huggingface.co/rows?' + urlencode(params))
            require(isinstance(json.loads(data).get('rows'), list), 'Upstream rows are missing')
            path.write_bytes(data)
            print(canonical({'stage': 'upstream_page', 'dataset': name, 'offset': offset}), flush=True)
            time.sleep(.3)


def strip_image_markup(text):
    # Remove only attachment placeholders, never mathematical commands/expressions.
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', text)
    return re.sub(r'<(?:image|img)_?\d*>', '', text)


def text_identity(text):
    text = strip_image_markup(text)
    text = text.removeprefix('请解答下面的物理题。').strip()
    return re.sub(r'\s+', ' ', text).strip()


def display_text(text):
    text = strip_image_markup(text)
    # LiveK12 exports escaped line separators. Match whole separator pairs first;
    # never use replace('\\n', '\n'), which corrupts \\neq, \\nu and \\nabla.
    text = re.sub(r'(?:\\n){2,}', lambda m: '\n' * (len(m[0]) // 2), text)
    text = re.sub(r'\\n(?=$|\s|[A-Z]\.\s|\d|[（(])', '\n', text)
    return text.strip()


def statement_match(local, upstream):
    if text_identity(local) == text_identity(upstream):
        return 'same_text_after_whitespace_and_attachment_markup'
    # Recognition of the old import transformation is for matching ONLY. The
    # new solving prompt uses the upstream text, preserving mathematical commands.
    if text_identity(local) == text_identity(upstream.replace('\\n', '\n')):
        return 'identified_legacy_backslash_n_decode'
    return None


def damaged_latex_commands(local, upstream):
    pattern = r'\\(?:ne|neq|nu|nabla|neg|not|notin|nparallel|nsubseteq|nrightarrow)(?![A-Za-z])'
    before, after = Counter(re.findall(pattern, upstream)), Counter(re.findall(pattern, local))
    return dict(before - after)


def _rows(cache, name):
    rows = []
    repo, config, split, extent = DATASETS[name]
    info_path = cache / (repo.replace('/', '--') + '-info.json')
    revision = read_json(info_path)['sha']
    evidence = {info_path.name: file_hash(info_path)}
    for offset in range(0, extent, 100):
        path = cache / f'{name}-rows-{offset}.json'
        require(path.is_file(), 'Missing upstream cache page: ' + path.name + '; use --fetch')
        page = read_json(path)
        require(page.get('partial') is not True, 'Partial upstream export')
        expected_count = min(100, page['num_rows_total'] - offset)
        require(len(page['rows']) == expected_count, 'Incomplete upstream page')
        require([r['row_idx'] for r in page['rows']] == list(range(offset, offset + expected_count)), 'Unexpected upstream row ordering')
        require(all(not r.get('truncated_cells') for r in page['rows']), 'Truncated upstream answer/question')
        for item in page['rows']:
            rows.append({**item, 'cache_file': path.name})
        evidence[path.name] = file_hash(path)
    return rows, revision, evidence


def upstream_images(row, name):
    images = ([row.get(f'image_{i}') for i in range(1, 10)] if name == 'olympiad' else row['images'])
    return [a for a in images if a is not None]


def _reference(row, name, url):
    if name.startswith('live'):
        answer, solution = row['answer'], row.get('solution', '')
        # Multiple Choice is upstream metadata, not a guess from a lone "A" unit.
        if row['question_type'] == 'Multiple Choice' and len(answer) == 1 and re.fullmatch(r'[A-H]+', answer[0]):
            labels = answer[0]
            require(len(labels) == len(set(labels)), 'Repeated upstream choice label')
            return {'kind': 'choice' if len(labels) == 1 else 'choice_set',
                    'accepted': [labels] if len(labels) == 1 else list(labels), 'reliable': True, 'source_reference': url}
    elif name == 'seephys':
        answer, solution = row['answer'], row.get('reasoning', '')
    elif name == 'physelite':
        answer, solution = row['problem_answer'], row.get('problem_solution', '')
    else:
        answer, solution = row['final_answer'], row.get('solution', [])
    require(bool(answer) or bool(solution), 'Upstream has neither answer nor solution')
    # Open-ended physics needs equivalence, units and all-subpart coverage. Do
    # not turn diverse equations/proofs into a misleading exact-string accuracy.
    return {'kind': 'human', 'reliable': True, 'source_reference': url,
            'reference_answer': canonical({'final_answer': answer, 'solution': solution,
                'unit': row.get('unit'), 'answer_type': row.get('answer_type'),
                'official_tolerance': row.get('error'),
                'rubric': 'Grade all requested subparts using the official answer/solution; accept physically and mathematically equivalent forms with compatible units. Give 1 only if the whole requested answer is correct; otherwise 0. Do not grade by wording equality.'})}


def match(source_root, base, cache, output, *, local_archive=None):
    source = Release(source_root, base)
    originals = source.originals()
    cache, output = Path(cache).resolve(), Path(output).resolve()
    require(not output.exists(), 'Match output exists; use a new directory')
    require(not output.is_relative_to(source.root) and not source.root.is_relative_to(output), 'Output overlaps release')
    output.mkdir(parents=True)
    datasets = {name: _rows(cache, name) for name in DATASETS}
    archive = {}
    if local_archive:
        import json
        for line in Path(local_archive).open(encoding='utf-8'):
            item = json.loads(line)
            require(item['id'] not in archive, 'Duplicate original archive ID')
            archive[item['id']] = item
    catalog, matches, keys = {}, {}, answer_template(source, originals)
    counts, restored, repaired, damage_details = Counter(), [], [], {}
    for mother, original in originals.items():
        meta = original
        name = ('olympiad' if mother.startswith('olympiadbench__') else 'physelite' if mother.startswith('physelite__')
                else 'seephys' if mother.startswith('seephys__') else 'live2603' if '__en_2603_' in mother else 'live2605')
        rows, revision, _ = datasets[name]
        sid = meta['source_sample_id']
        if name == 'seephys':
            candidates = [r for r in rows if r['row_idx'] == int(sid.rsplit('_', 1)[-1])]
        elif name == 'physelite':
            candidates = [r for r in rows if str(r['row']['problem_id']) == sid and r['row']['language'] == 'en']
        else:
            candidates = [r for r in rows if str(r['row']['id']) == sid]
        require(len(candidates) == 1, 'No unique upstream identity: ' + mother)
        item = candidates[0]; row = item['row']
        question = (row.get('context') or '') + '\n\n' + row['question'] if name == 'olympiad' else row['problem_text'] if name == 'physelite' else row['question']
        manifest = source.json(original['source_reference'])
        method = statement_match(manifest['raw_question'], question)
        require(method is not None, 'Unexplained original statement difference: ' + mother)
        images = upstream_images(row, name)
        require(images, 'Original has no upstream question image')
        repo, config, split, _ = DATASETS[name]
        require(all('/--/' + revision + '/--/' in a['src'] and repo + '/' in a['src'] for a in images), 'Viewer image revision disagrees with pinned repository')
        local = original['input']['attachments']
        positions = list(range(len(local)))
        if len(local) != len(images):
            require(mother in archive, 'Partial image selection has no retained source-index provenance')
            am = archive[mother]['metadata']
            require(am['source_revision'] == revision and am['source_sample_id'] == sid, 'Archive source revision/ID mismatch')
            by_hash = {x['sha256']: x['source_image_index'] - 1 for x in am['question_images']}
            positions = [by_hash[a['sha256']] for a in local]
            require(len(set(positions)) == len(positions) and all(0 <= i < len(images) for i in positions), 'Invalid archived image indices')
            restored.append(mother)
        keep = dict(zip(positions, local))
        attachments, extra = [], []
        for index, image in enumerate(images):
            if index in keep:
                a = deepcopy(keep[index])
                require((a['width'], a['height']) == (image['width'], image['height']), 'Upstream/local original image geometry differs')
            else:
                # Only the omitted question images are fetched. Original images
                # already retained in the release keep their exact source bytes.
                url = image['src']
                cache_name = digest(urlsplit(url).path.encode('utf-8')) + '.bin'
                cached = cache / 'extra-images' / cache_name
                if not cached.exists():
                    data = public_get(url)
                    image_metadata(data)
                    cached.parent.mkdir(parents=True, exist_ok=True); cached.write_bytes(data)
                data = cached.read_bytes(); mime, w, h = image_metadata(data)
                require((w, h) == (image['width'], image['height']), 'Downloaded question image dimensions differ')
                sha = digest(data); relative = '_supplement_originals/' + sha + ('.png' if mime == 'image/png' else '.jpg')
                target = confined(output, relative); target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(data)
                a = {'path': relative, 'sha256': sha, 'width': w, 'height': h}
                extra.append({'source_image_index': index + 1, 'sha256': sha,
                              'source_url_without_signature': url.split('?', 1)[0],
                              'encoding': 'public Hugging Face viewer image; may be re-encoded'})
            a['asset_id'] = 'img_' + str(index)
            attachments.append(a)
        user = display_text(question)
        inp = {'messages': {'system': SOLVE_SYSTEM, 'user': user}, 'attachments': attachments}
        url = f'https://huggingface.co/datasets/{repo}/tree/{revision}'
        provenance = {'dataset': repo, 'revision': revision, 'config': config, 'split': split,
                      'source_sample_id': sid, 'row_idx': item['row_idx'], 'upstream_id': row.get('id', row.get('index', row.get('problem_id'))),
                      'cache_file': item['cache_file'], 'cache_sha256': file_hash(cache / item['cache_file']),
                      'upstream_row_hash': fingerprint(row), 'statement_match': method,
                      'original_image_positions': positions, 'source_image_count': len(images),
                      'added_question_images': extra, 'reference_source': url,
                      'image_check': 'retained source bytes + upstream ordered dimensions; omitted images restored using archived source indices'}
        damage = damaged_latex_commands(manifest['raw_question'], question)
        if damage and method == 'identified_legacy_backslash_n_decode':
            repaired.append(mother)
            damage_details[mother] = damage
        catalog[mother] = {'match_status': 'matched', 'source_input_hash': original['input_hash'], 'input': inp, 'provenance': provenance}
        key = _reference(row, name, url + ' | config=' + config + ', split=' + split + ', row_idx=' + str(item['row_idx']) + ', source_sample_id=' + sid)
        keys['problems'][mother].update(input_hash=fingerprint(inp), answer_key=key,
            context_answers_reviewed=True,
            context_review_note='Official context is supplied background; the current target is the separate official question field. The target final_answer/solution is kept in private scoring data only.')
        matches[mother] = {'source_dataset': repo, 'source_sample_id': sid, 'source_revision': revision,
                           'match_status': 'matched', 'grading_kind': key['kind'], 'statement_match': method,
                           'source_image_count': len(images), 'restored_images': len(extra)}
        counts[repo] += 1
    snapshot = {'datasets': {n: {'revision': d[1], 'files': d[2]} for n, d in datasets.items()},
                'local_archive_sha256': file_hash(Path(local_archive)) if local_archive else None,
                'matching_script_sha256': file_hash(Path(__file__)), 'created_at': datetime.now(timezone.utc).isoformat()}
    payload = {'schema_version': 'physalign_matched_originals_v1', 'source_identity': source.identity,
               'problems': catalog, 'upstream_snapshot_hash': fingerprint(snapshot)}
    payload['catalog_hash'] = fingerprint(payload)
    write_new(output / 'originals.json', payload)
    write_new(output / 'answer-keys.json', keys)
    write_new(output / 'matches.json', matches)
    write_new(output / 'upstream-snapshot.json', snapshot)
    summary = {'matched_mothers': len(catalog), 'by_source': dict(counts),
               'grading_kinds': dict(Counter(x['grading_kind'] for x in matches.values())),
               'mothers_needing_restored_question_images': restored,
               'mothers_with_legacy_latex_newline_damage': repaired,
               'damaged_latex_commands': damage_details,
               'restored_question_images': sum(x['restored_images'] for x in matches.values()),
               'source_and_active_runs_modified': False}
    write_new(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--base-plan', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-archive')
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args(argv)
    if args.fetch:
        fetch(args.cache)
    print(canonical(match(args.source, load_plan(args.base_plan), args.cache, args.output,
                          local_archive=args.local_archive)), flush=True)


if __name__ == '__main__':
    main()
