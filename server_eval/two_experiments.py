"""Release import, all-record Raw/Gold evaluation, and audited paper figures.

Standalone companion: does not patch the project's existing evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT = Path(os.environ.get('PHYSALIGN_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(PROJECT))

from physalign.adapters import ModelRequest
from physalign.controls import original_ids
from physalign.dataset import PublicDataset, load_truth, require, sections
from physalign.planning import code_hashes, load_plan, prepare_plan, save_plan, verify_public_plan
from physalign.reporting import score_run
from physalign.runner import collect_records, validate_manifest
from physalign.scoring import score_response
from physalign.storage import (atomic_json, atomic_text, canonical, confined, file_hash,
                               fingerprint, loads, read_json, run_lock, write_new)

JOBS = (
    ('qwen35-9b', 'Qwen/Qwen3.5-9B', 'Qwen3.5-9B', 2),
    ('qwen35-27b', 'Qwen/Qwen3.5-27B', 'Qwen3.5-27B', 4),
    ('internvl35-8b', 'OpenGVLab/InternVL3_5-8B-HF', 'InternVL3.5-8B', 2),
)
SPLIT = 'all_provisional'
RELEASE_SCHEMA = 'physalign_merged_model_evaluation_dataset_v1'
PIPELINE_SCHEMA = 'physalign_all_raw_gold_pipeline_v1'


def log(**values):
    print(canonical(values), flush=True)


def available_gpus():
    value = os.environ.get('PHYSALIGN_AVAILABLE_GPUS', '0,1,2,3,4,5,6,7')
    ids = tuple(v.strip() for v in value.split(','))
    require(all(re.fullmatch(r'0|[1-9][0-9]*', v) for v in ids) and len(set(ids)) == len(ids), 'GPU IDs must be unique nonnegative integers')
    require(len(ids) >= 4, 'At least four available GPUs are needed for the unchanged 27B configuration')
    return ids


def allocate_jobs(pending, occupied, gpu_ids, assignments=None):
    """Largest model first; disjoint groups; refill as soon as a group is free."""
    free = [g for g in gpu_ids if g not in occupied]
    selected = []
    for job in sorted(pending, key=lambda row: -row[3]):
        prior = (assignments or {}).get(job[0])
        group = tuple(prior) if prior is not None else tuple(free[:job[3]])
        if len(group) == job[3] and set(group) <= set(free):
            selected.append((job, group))
            free = [g for g in free if g not in group]
    return selected


def validate_public_locators(rows):
    count = 0
    for row in rows:
        try:
            parts = sections(row['input']['messages']['user'])
            evidence = loads(parts['Evidence locations and candidates'])
            ids = [a['asset_id'] for a in row['input']['attachments']]
            locators = list(evidence['anchors'].values()) + [loc for c in evidence['candidates'] for loc in c['locators']]
            for locator in locators:
                PublicDataset._validate_locator(locator, ids, parts)
                count += 1
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Public locator check failed: {row.get('instance_id')} / {row.get('condition')} / {row.get('interface')}: {error}") from error
    log(stage='public_locators_verified', records=len(rows), locators=count)


def indexed(rows, name):
    require(isinstance(rows, list), f'{name} must be an array')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('instance_id'), str), f'Invalid {name} record')
        iid = row['instance_id']
        require(iid not in result, f'Duplicate {name} instance: {iid}')
        result[iid] = row
    return result


def inventory_map(value):
    # Accept a flat SHA-256 map, or an explicitly named files container.
    if isinstance(value, dict) and 'files' in value and isinstance(value['files'], (dict, list)):
        value = value['files']
    if isinstance(value, list):
        paths = [r['path'] for r in value]
        require(len(paths) == len(set(paths)), 'Duplicate file inventory path')
        value = {r['path']: r['sha256'] for r in value}
    require(isinstance(value, dict) and value, 'Unknown or empty FILE_MANIFEST format')
    result = {}
    for path, item in value.items():
        sha = item.get('sha256') if isinstance(item, dict) else item
        require(isinstance(path, str) and isinstance(sha, str)
                and re.fullmatch('[0-9a-f]{64}', sha), f'Unsupported inventory entry: {path}')
        result[path] = sha
    return result


def target_id(value):
    require(isinstance(value, dict) and set(value) == {'kind', 'id'}, 'Unknown canonical target contract')
    require(all(isinstance(v, str) and v for v in value.values()), 'Empty canonical target')
    return canonical([value['kind'], value['id']])


def convert_release(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(source != destination and not destination.is_relative_to(source), 'Derived bundle must be outside source')
    release_path, inventory_path = source / 'release.json', source / 'FILE_MANIFEST.json'
    release = read_json(release_path)
    require(release.get('schema_version') == RELEASE_SCHEMA, 'Unexpected release schema')
    require(release.get('release_status') == 'model_reviewed_provisional', 'Unexpected release status; inspect new protocol')
    require(release.get('synthetic_example_only') is False, 'Synthetic release is not an evaluation dataset')
    require(release.get('attachment_paths_relative_to') == 'public', 'Unknown image path base')
    source_identity = {'release_sha256': file_hash(release_path), 'inventory_sha256': file_hash(inventory_path)}
    inventory = inventory_map(read_json(inventory_path))
    def source_json(relative):
        require(relative in inventory, f'Source file absent from inventory: {relative}')
        path = confined(source, relative)
        require(file_hash(path) == inventory[relative], f'Source hash mismatch: {relative}')
        return read_json(path)

    # Fail incompatible public contracts before hashing/copying the large archive.
    raw_rows = source_json('public/qa_raw.json')
    gold_rows = source_json('public/qa_gold.json')
    validate_public_locators(raw_rows + gold_rows)
    for i, (relative, sha) in enumerate(inventory.items(), 1):
        require(file_hash(confined(source, relative)) == sha, f'Source hash mismatch: {relative}')
        if i % 250 == 0:
            log(stage='verify_source', files=i, total=len(inventory))

    if destination.exists():
        ds = PublicDataset(destination)
        require(ds.manifest.get('source_identity') == source_identity, 'Existing conversion has a different source; choose a new tag')
        require(ds.manifest.get('converter_sha256') == file_hash(Path(__file__)), 'Converter changed; choose a new tag')
        ds.verify_inventory()
        load_truth(ds)
        log(stage='conversion_reused', directory=str(destination), counts=ds.manifest['counts'])
        return ds.manifest

    raw, gold = indexed(raw_rows, 'raw'), indexed(gold_rows, 'gold')
    answers = indexed(source_json('private/answers.json'), 'answers')
    maps = indexed(source_json('private/mappings.json'), 'mappings')
    membership = indexed(source_json('private/membership.json'), 'membership')
    require(set(raw) == set(answers) == set(maps) == set(membership), 'Public/private ID sets differ')
    require(set(gold) <= set(raw), 'Gold has no matching Raw')
    for iid, row in gold.items():
        require(row['split'] == raw[iid]['split'], f'Source Raw/Gold split mismatch: {iid}')
    observed = {'raw': len(raw), 'gold': len(gold), 'mothers': len({m['problem_id'] for m in membership.values()})}
    require(all(release['counts'][k] == v for k, v in observed.items()), 'Release counts disagree with records')
    quarantined = set(release.get('quarantined_mother_ids', []))
    selected_sources = {m['problem_id'] for m in membership.values()} | {m['key']['problem_id'] for m in maps.values()}
    require(not selected_sources & quarantined, f'Quarantined mother is still present: {sorted(selected_sources & quarantined)}')

    # Observed shared-original-image components; explicitly not semantic deduplication.
    mothers = sorted({m['problem_id'] for m in membership.values()})
    parent = {m: m for m in mothers}
    def find(m):
        while parent[m] != m:
            parent[m] = parent[parent[m]]
            m = parent[m]
        return m
    image_owner, mother_images = {}, {}
    for iid, row in raw.items():
        mother = membership[iid]['problem_id']
        attachments = row['input']['attachments']
        by_asset = {a['asset_id']: a for a in attachments}
        require(len(by_asset) == len(attachments), f'Duplicate image asset ID: {iid}')
        ids = original_ids(row['input']['messages']['user'])
        require(set(ids) <= set(by_asset), f'Missing original image: {iid}')
        identity = tuple(by_asset[a]['sha256'] for a in ids)
        require(mother not in mother_images or mother_images[mother] == identity, f'Mother original images disagree: {mother}')
        mother_images[mother] = identity
        for sha in identity:
            if sha in image_owner:
                a, b = find(mother), find(image_owner[sha])
                parent[max(a, b)] = min(a, b)
            else:
                image_owner[sha] = mother
    components = defaultdict(list)
    for mother in mothers:
        components[find(mother)].append(mother)
    cluster_map = {m: 'shared-original-' + fingerprint(sorted(group)) for group in components.values() for m in group}

    native, provenance = [], {}
    for iid, row in raw.items():
        member, answer, mapping = membership[iid], answers[iid], maps[iid]
        key = mapping['key']
        require(key['schema_version'] == 'physalign_coverage_expansion_v1', f'Unknown scoring schema: {iid}')
        require(all(x['logical_probe_id'] == row['logical_probe_id'] for x in (member, answer, key)), f'Logical ID mismatch: {iid}')
        require(member['interface'] == answer['interface'] == row['interface'], f'Interface mismatch: {iid}')
        require(member['split'] == row['split'] and row['split'] in {'development', 'test'}, f'Unexpected source split: {iid}')
        require(member['review_status'] == answer['review_status'] == 'model_approved', f'Unapproved released probe: {iid}')
        require(all(x['human_reviewed'] is False for x in (member, answer, mapping, key)), f'Unexpected human-review state: {iid}')
        contract = key['response_contract']
        require(contract['cardinality'] == 'one', f'Unsupported binding cardinality: {iid}')
        expected = {'T02-single': ('T02', 'referent'), 'T03-image': ('T03', 'owner'), 'T03-text': ('T03', 'owner')}
        require(row['interface'] in expected, f'Unknown interface: {iid}')
        require((row['task_id'], contract['binding_key']) == expected[row['interface']], f'Binding contract mismatch: {iid}')
        require(bool(key['read_targets']) == (row['interface'] == 'T03-image'), f'Unexpected reading eligibility: {iid}')
        domain = {alias: target_id(value) for alias, value in key['candidate_map'].items()}
        require(set(answer['candidate_ids']) == set(domain), f'Candidate-domain mismatch: {iid}')
        readings = [{'field': r.get('field', 'read'), 'anchor_id': r['anchor_alias'], 'expected': r['expected'],
                     'rule': {'kind': 'ocr', 'normalizer_id': r['normalizer_id']}} for r in key['read_targets']]
        packet = None
        if iid in gold:
            text = sections(gold[iid]['input']['messages']['user'])['Local reading information']
            packet = [] if text == 'None.' else loads(text)
            expected_packet = [{'anchor_id': r['anchor_id'], 'text': r['expected']} for r in readings]
            require(packet == expected_packet, f'Exported Gold differs from scored reading targets: {iid}')
        require((iid in gold) == (row['interface'] == 'T03-image'), f'Unexpected Gold coverage: {iid}')
        native.append({'instance_id': iid, 'logical_probe_id': row['logical_probe_id'], 'problem_id': member['problem_id'],
                       'probe_type': row['task_id'], 'split': SPLIT, 'review_status': 'model_approved', 'human_reviewed': False,
                       'cluster_id': cluster_map[member['problem_id']],
                       'binding': {'kind': 'single', 'domains': [{'field': contract['binding_key'], 'candidates': domain}],
                                   'gold': [target_id(v) for v in key['gold_targets']]},
                       'readings': readings, 'permitted_gold_packet': packet})
        provenance[iid] = {'original_public_split': row['split'], 'membership': member,
                           'mapping_review_method': mapping['review_method'], 'review_target_root': mapping['review_target_root'],
                           'source_problem_id': key['problem_id'], 'source_packet_plan': key['packet_plan']}

    manifest = {'schema_version': 'physalign_eval_bundle_v1', 'image_paths_relative_to': 'public',
                'formal_release_eligibility_checked': False, 'human_reviewed': False,
                'unseen_test_split_certified': False, 'release_status': 'model_reviewed_provisional',
                'usage': 'development_and_test_combined_all_provisional', 'source_identity': source_identity,
                'converter_sha256': file_hash(Path(__file__)), 'counts': {'probes': len(raw), **observed},
                'original_split_counts': dict(Counter(r['split'] for r in raw_rows)),
                'by_interface_raw': dict(Counter(r['interface'] for r in raw_rows)),
                'joint_probes': sum(bool(r['readings']) for r in native),
                'source_clusters': len(components),
                'clustering': 'Connected components of shared original-image SHA-256; semantic deduplication remains uncertified',
                'weighting': 'Existing equal task_id weights; mother means within task. T03-image and T03-text remain T03.'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=destination.name + '.building-', dir=destination.parent))
    log(stage='building_derived_bundle', directory=str(stage))
    for name, rows in (('qa_raw.json', raw_rows), ('qa_gold.json', gold_rows)):
        converted = deepcopy(rows)
        for row in converted:
            row['split'] = SPLIT
        write_new(stage / 'public' / name, converted)
    copied = {}
    for row in raw_rows + gold_rows:
        for attachment in row['input']['attachments']:
            relative = 'public/' + attachment['path']
            require(inventory.get(relative) == attachment['sha256'], f'Image inventory mismatch: {relative}')
            if relative in copied:
                require(copied[relative] == attachment['sha256'], 'Conflicting image hashes')
                continue
            src, dst = confined(source, relative), confined(stage, relative)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            require(file_hash(dst) == attachment['sha256'], f'Copied image mismatch: {relative}')
            copied[relative] = attachment['sha256']
            if len(copied) % 250 == 0:
                log(stage='copying_images', images=len(copied))
    log(stage='images_copied', images=len(copied))
    write_new(stage / 'private/probes.json', native)
    write_new(stage / 'private/original_provenance.json', provenance)
    write_new(stage / 'private/source_release.json', release)
    write_new(stage / 'private/source_inventory.json', inventory)
    write_new(stage / 'private/observed_source_clusters.json', cluster_map)
    write_new(stage / 'manifest.json', manifest)
    hashes = {p.relative_to(stage).as_posix(): file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    write_new(stage / 'FILE_MANIFEST.json', hashes)
    ds = PublicDataset(stage)
    ds.verify_inventory()
    truth, _ = load_truth(ds)
    for iid, probe in truth.items():
        scored = score_response(probe, canonical(answers[iid]['answer']))
        require(scored.B == 1 and (not probe.joint_eligible or scored.C == 1), f'Simplified answer contradicts canonical key: {iid}')
    require(source_identity == {'release_sha256': file_hash(release_path), 'inventory_sha256': file_hash(inventory_path)}, 'Source changed during conversion')
    stage.rename(destination)
    log(stage='converted', directory=str(destination), counts=manifest['counts'], joint=manifest['joint_probes'], clusters=len(components))
    return manifest


@dataclass
class Context:
    project: Path
    source: Path
    tag: str
    bootstrap: int = 2000
    bundle_root: Path | None = None
    visual_profile: str = 'checkpoint-default'

    @property
    def bundle(self): return self.bundle_root or self.project / 'datasets' / (self.tag + '-bundle')
    @property
    def plan_path(self): return self.project / 'plans' / (self.tag + '.json')
    @property
    def inputs(self): return self.project / 'plans' / (self.tag + '-inputs')
    @property
    def runs(self): return self.project / 'runs' / self.tag
    @property
    def figures(self): return self.project / 'figures' / self.tag

    def config_path(self, name): return self.inputs / (name + '.json')


def prepare(ctx):
    from physalign.hf_adapter import image_budget_kwargs
    # A new processor profile needs new configs/plan, not another copy of all images.
    if ctx.bundle_root is not None:
        ds = PublicDataset(ctx.bundle)
        require(ds.manifest.get('usage') == 'development_and_test_combined_all_provisional', 'Not a combined evaluation bundle')
        require(ds.manifest.get('source_identity') == {
            'release_sha256': file_hash(ctx.source / 'release.json'),
            'inventory_sha256': file_hash(ctx.source / 'FILE_MANIFEST.json')}, 'Reused bundle comes from another source release')
        # prepare_plan (or the existing-plan verification below) audits this inventory.
        log(stage='derived_bundle_reused', directory=str(ctx.bundle), counts=ds.manifest['counts'])
    else:
        convert_release(ctx.source, ctx.bundle)
    if ctx.plan_path.exists():
        plan = load_plan(ctx.plan_path)
        verify_public_plan(plan).verify_inventory()
        require(plan['bootstrap']['n_resamples'] == ctx.bootstrap, 'Bootstrap setting changed; use a new tag')
        require(Path(plan['dataset_hint']).resolve() == ctx.bundle.resolve(), 'Existing plan uses another bundle')
    else:
        plan = prepare_plan(ctx.bundle, split=SPLIT, conditions=('raw', 'gold'), bootstrap_resamples=ctx.bootstrap,
                            bootstrap_seed=2027, confidence=.95)
        save_plan(plan, ctx.plan_path)
    ctx.inputs.mkdir(parents=True, exist_ok=True)
    for name, model_id, _, devices in JOBS:
        output = ctx.config_path(name)
        if output.exists():
            require(read_json(output).get('processor_call_kwargs', {}) == image_budget_kwargs(model_id, ctx.visual_profile),
                    'Frozen visual profile changed; use a new tag')
            continue
        config = read_json(ctx.project / 'configs/server' / (name + '.json'))
        require(config['gpu_count'] == devices, 'GPU count does not match 2/4/2 allocation')
        snapshot = read_json(ctx.project / config['snapshot_manifest'])
        require(snapshot['model_id'] == model_id, f'Wrong model snapshot for {name}')
        path = ctx.inputs / (name + '.snapshot.json')
        if path.exists():
            require(read_json(path) == snapshot, f'Existing frozen snapshot differs: {name}')
        else:
            write_new(path, snapshot)
        config['snapshot_manifest'] = str(path.resolve())
        config['processor_call_kwargs'] = image_budget_kwargs(model_id, ctx.visual_profile)
        write_new(output, config)
    config_hashes = {name: fingerprint(read_json(ctx.config_path(name))) for name, *_ in JOBS}
    snapshot_hashes = {name: file_hash(Path(read_json(ctx.config_path(name))['snapshot_manifest'])) for name, *_ in JOBS}
    settings = [{k: read_json(ctx.config_path(name)).get(k) for k in ('dtype', 'thinking', 'seed', 'max_input_tokens', 'max_new_tokens')}
                for name, *_ in JOBS]
    require(all(s == settings[0] for s in settings) and settings[0]['thinking'] is False, 'Use comparable frozen model settings')
    frozen = {'schema_version': PIPELINE_SCHEMA, 'plan_hash': plan['plan_hash'], 'config_hashes': config_hashes,
              'available_gpus': list(available_gpus()),
              'visual_profile': ctx.visual_profile,
              'snapshot_file_hashes': snapshot_hashes, 'helper_sha256': file_hash(Path(__file__)), 'core_hashes': code_hashes(all_modules=True)}
    marker = ctx.inputs / 'pipeline.json'
    if marker.exists():
        require(read_json(marker) == frozen, 'Frozen execution inputs changed; use a new tag')
    else:
        write_new(marker, frozen)
    log(stage='prepared', plan=str(ctx.plan_path), raw=len(plan['probes']), gold=len(plan['paired_ids']),
        requests_per_model=len(plan['requests']), three_model_requests=3 * len(plan['requests']))
    return plan


def checked_plan(ctx):
    plan = load_plan(ctx.plan_path)
    marker = read_json(ctx.inputs / 'pipeline.json')
    require(marker['plan_hash'] == plan['plan_hash'] and marker['helper_sha256'] == file_hash(Path(__file__)), 'Plan/helper changed after freeze')
    require(marker['core_hashes'] == code_hashes(all_modules=True), 'Evaluator code changed after freeze')
    require(marker['available_gpus'] == list(available_gpus()), 'Available GPU pool changed after freeze; restore it or use a new tag')
    require(marker['visual_profile'] == ctx.visual_profile, 'Visual profile differs from frozen run')
    for name, *_ in JOBS:
        require(marker['config_hashes'][name] == fingerprint(read_json(ctx.config_path(name))), f'Frozen config changed: {name}')
        snapshot_path = Path(read_json(ctx.config_path(name))['snapshot_manifest'])
        require(marker['snapshot_file_hashes'][name] == file_hash(snapshot_path), f'Frozen snapshot changed: {name}')
    return plan


def preflight(ctx):
    from PIL import Image
    from transformers import AutoProcessor
    from physalign.hf_adapter import encode_request, verify_snapshot
    plan = checked_plan(ctx)
    ds = verify_public_plan(plan)
    ds.verify_inventory()
    failed = []
    for name, model_id, _, _ in JOBS:
        config = read_json(ctx.config_path(name))
        snapshot = read_json(Path(config['snapshot_manifest']))
        root = verify_snapshot(snapshot, config.get('model_path'))
        require(snapshot['model_id'] == model_id and config['thinking'] is False, 'Wrong preflight model/thinking mode')
        cfg = read_json(root / 'config.json')
        native_context = cfg.get('text_config', cfg).get('max_position_embeddings')
        kwargs = config.get('processor_kwargs', {})
        require(not set(kwargs) & {'trust_remote_code', 'local_files_only', 'revision'}, 'Invalid processor overrides')
        processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False, **kwargs)
        template = {'enable_thinking': False} if model_id.startswith('Qwen/') else {}
        rows = []
        for i, row in enumerate(plan['requests'], 1):
            item = ds.items[row['instance_id'], row['condition']]
            messages = item.record['input']['messages']
            request = ModelRequest('preflight', messages['system'], messages['user'], ds.images(item), '{}')
            tensors = encode_request(processor, Image, request, template, config.get('processor_call_kwargs'))
            length = tensors['input_ids'].shape[-1]
            within = length <= config['max_input_tokens'] and (not native_context or length + config['max_new_tokens'] <= native_context)
            rows.append({'instance_id': row['instance_id'], 'condition': row['condition'], 'input_tokens': length,
                         'images': len(request.images),
                         'within_budget': bool(within), 'tensor_shapes': {k: list(v.shape) for k, v in tensors.items() if hasattr(v, 'shape')}})
            del tensors, request
            if i == 1 or i % 50 == 0 or i == len(plan['requests']):
                log(stage='processor_preflight', model=name, checked=i, total=len(plan['requests']), input_tokens=length)
        result = {'plan_hash': plan['plan_hash'], 'configuration_hash': fingerprint(config), 'snapshot_hash': snapshot['snapshot_hash'],
                  'max_input_tokens': config['max_input_tokens'], 'max_new_tokens': config['max_new_tokens'],
                  'native_context': native_context, 'processor_call_kwargs': config.get('processor_call_kwargs', {}),
                  'requests': rows, 'all_within_budget': all(r['within_budget'] for r in rows), 'gpu_memory_verified': False}
        atomic_json(ctx.inputs / (name + '.preflight.json'), result)
        over = [r for r in rows if not r['within_budget']]
        log(stage='preflight_summary', model=name, requests=len(rows), over_budget=len(over),
            max_input_tokens=max(r['input_tokens'] for r in rows), configured_limit=config['max_input_tokens'],
            native_context=native_context, visual_profile=ctx.visual_profile,
            worst_requests=sorted(over, key=lambda r: -r['input_tokens'])[:3])
        if not result['all_within_budget']:
            failed.append(name)
        del processor
    require(not failed, f'Input budget exceeded: {failed}; inspect preflight reports before model runs')


def run_models(ctx):
    plan = checked_plan(ctx)
    verify_public_plan(plan).verify_inventory()
    ctx.runs.mkdir(parents=True, exist_ok=True)
    with run_lock(ctx.runs):
        # Validate all inputs before starting any GPU process.
        pending, resumed = [], {}
        for job in JOBS:
            name, model_id, _, _ = job
            config = read_json(ctx.config_path(name))
            pf = read_json(ctx.inputs / (name + '.preflight.json'))
            require(pf['plan_hash'] == plan['plan_hash'] and pf['configuration_hash'] == fingerprint(config)
                    and pf['all_within_budget'], f'Missing or stale successful preflight: {name}')
            output = ctx.runs / name
            manifest_path = output / 'manifest.json'
            resumed[name] = manifest_path.exists()
            if resumed[name]:
                manifest = read_json(manifest_path)
                validate_manifest(manifest)
                require(manifest['plan']['plan_hash'] == plan['plan_hash'] and manifest['adapter']['model_id'] == model_id,
                        'Existing run does not match frozen plan/model')
                require(manifest['adapter_provenance']['configuration_hash'] == fingerprint(config), 'Run config changed')
                records = collect_records(output, manifest, allow_incomplete=True)
                if all(r is not None and r['status'] == 'completed' for r in records):
                    log(stage='already_complete', model=name)
                    continue
            pending.append(job)
        gpu_ids = available_gpus()
        allocation_path = ctx.runs / 'gpu_assignments.json'
        assignments = read_json(allocation_path) if allocation_path.exists() else {}
        sizes = {job[0]: job[3] for job in JOBS}
        require(set(assignments) <= set(sizes), 'Unknown model in GPU assignment journal')
        for name, group in assignments.items():
            require(isinstance(group, list) and len(group) == len(set(group)) == sizes[name]
                    and set(group) <= set(gpu_ids), 'Invalid persisted GPU assignment')
        require(all(not resumed[job[0]] or job[0] in assignments for job in pending), 'Missing original GPU assignment for resumed run')
        active, failures = [], []
        log(stage='gpu_schedule', available_gpus=gpu_ids, priority='largest_model_first', pending=[j[0] for j in pending])
        try:
            while pending or active:
                for child in active[:]:
                    name, process, stream, devices = child
                    result = process.poll()
                    if result is not None:
                        stream.close()
                        active.remove(child)
                        if result != 0:
                            failures.append(name)
                        log(stage='model_exited', model=name, exit_code=result, released_gpus=devices)
                occupied = {g for _, _, _, group in active for g in group}
                for job, devices in allocate_jobs(pending, occupied, gpu_ids, assignments):
                    name = job[0]
                    assignments[name] = list(devices)
                    atomic_json(allocation_path, assignments)
                    logdir = ctx.runs / 'launcher_logs'
                    logdir.mkdir(exist_ok=True)
                    stream = (logdir / (name + '.log')).open('ab')
                    command = [sys.executable, '-u', str(ctx.project / 'evaluate.py'), 'run', '--plan', str(ctx.plan_path),
                               '--output', str(ctx.runs / name), '--adapter', 'hf', '--adapter-config', str(ctx.config_path(name))]
                    if resumed[name]:
                        command.append('--resume')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
                    try:
                        process = subprocess.Popen(command, cwd=ctx.project, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    except BaseException:
                        stream.close()
                        raise
                    active.append((name, process, stream, devices))
                    pending.remove(job)
                    log(stage='model_started', model=name, gpus=devices, pid=process.pid, resume=resumed[name])
                if active:
                    time.sleep(.5)
            require(not failures, f'Model processes failed: {failures}; see launcher_logs')
        finally:
            for _, process, stream, _ in active:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                stream.close()


def score_models(ctx):
    plan = checked_plan(ctx)
    reports, failures, settings = [], [], None
    for name, model_id, label, _ in JOBS:
        directory = ctx.runs / name
        log(stage='scoring', model=name)
        manifest = read_json(directory / 'manifest.json')
        validate_manifest(manifest)
        require(manifest['adapter_provenance']['configuration_hash'] == fingerprint(read_json(ctx.config_path(name))), 'Scored run uses a different frozen configuration')
        require(manifest['implementation_hashes'] == code_hashes(all_modules=True), 'Scored run uses different evaluator code')
        report = score_run(directory, allow_incomplete=True)
        require(report['plan_hash'] == plan['plan_hash'] and report['adapter']['model_id'] == model_id, 'Run identity mismatch')
        require(report['scientific_run'], 'Smoke/replay outputs cannot be used as real model results')
        current = {k: report['adapter']['settings'].get(k) for k in ('max_input_tokens', 'max_new_tokens', 'thinking', 'do_sample', 'num_beams', 'seed')}
        require(settings is None or settings == current, 'Models use different decoding/token budgets')
        settings = current
        service = report['service']
        log(stage='service', model=name, **{k: len(v) if isinstance(v, list) else v for k, v in service.items()})
        if service['pending_requests'] or service['infrastructure_missing'] or service['completed_outputs'] != service['planned_requests']:
            failures.append(name)
        reports.append({'name': name, 'model_id': model_id, 'label': label, 'report': report,
                        'report_sha256': file_hash(directory / 'report.json'), 'manifest_sha256': file_hash(directory / 'manifest.json')})
    require(not failures, f'Incomplete service for {failures}. Reports are saved; paper figures withheld. Resume preserves first outputs and does not retry terminal OOM/interrupted records.')
    return reports


def status(ctx):
    for name, *_ in JOBS:
        path = ctx.runs / name / 'status.json'
        log(model=name, status=read_json(path) if path.exists() else 'not_finished_or_not_started',
            log=str(ctx.runs / 'launcher_logs' / (name + '.log')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--tag', default='all-raw-gold-v1')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--bundle', type=Path, default=os.environ.get('PHYSALIGN_BUNDLE_ROOT'))
    parser.add_argument('--visual-profile', choices=('checkpoint-default', 'multi-image-256-v1'),
                        default=os.environ.get('PHYSALIGN_VISUAL_PROFILE', 'checkpoint-default'))
    parser.add_argument('stage', choices=('prepare', 'preflight', 'run', 'score', 'plot', 'all', 'status'))
    args = parser.parse_args(argv)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.tag) is not None, 'Use a simple tag without path separators')
    project = args.project.resolve()
    require(project == PROJECT, 'Set PHYSALIGN_ROOT to the same project as --project')
    os.chdir(project)
    ctx = Context(project, (args.dataset or project / 'datasets/physalign-final-v1').resolve(), args.tag, args.bootstrap,
                  args.bundle.resolve() if args.bundle else None, args.visual_profile)
    if args.stage == 'prepare': prepare(ctx)
    elif args.stage == 'preflight': preflight(ctx)
    elif args.stage == 'run': run_models(ctx)
    elif args.stage == 'score': score_models(ctx)
    elif args.stage == 'plot': plot_models(ctx, score_models(ctx))
    elif args.stage == 'status': status(ctx)
    else:
        prepare(ctx)
        preflight(ctx)
        run_models(ctx)
        plot_models(ctx, score_models(ctx))
    log(stage='finished', command=args.stage, tag=ctx.tag)


# Plotting is defined below so conversion and preflight remain independent of Matplotlib.
def plot_models(ctx, reports, *, preview=False):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from physalign.metrics import Pool
    from physalign_viz.plots import FigureBook, _point, theme

    plan = load_plan(ctx.plan_path)
    ds = verify_public_plan(plan)
    truth, _ = load_truth(ds)
    require(reports and all(m['report']['plan_hash'] == plan['plan_hash'] for m in reports), 'Different model plans')
    require(preview or all(m['report']['scientific_run'] for m in reports), 'Non-scientific fixtures must be stamped as preview')
    output = ctx.figures
    if output.exists():
        output = output.with_name(output.name + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True)

    def interval(model, scope, key):
        block = model['report']['intervals'][scope] or {}
        row = block.get('intervals', {}).get(key, {})
        return [row['lower'], row['upper']] if row.get('status') == 'ok' and block.get('n_clusters', 0) >= 2 else None

    visual_models = []
    interfaces = sorted({r['interface'] for r in plan['probes']})
    for model in reports:
        report = model['report']
        # Formatting adapter only: no invented study, solver, or human results.
        visual_models.append({'label': model['label'], 'report': {'experiments': {'main': {
            'raw_all': report['raw_all'], 'paired': report['paired'], 'intervals': report['intervals']}}}})
        scores_path = ctx.runs / model['name'] / 'scores.jsonl'
        with scores_path.open(encoding='utf-8') as stream:
            scores = [loads(line) for line in stream if line.strip()]
        raw_scores = {r['instance_id']: r for r in scores if r['condition'] == 'raw'}
        require(set(raw_scores) == {r['instance_id'] for r in plan['probes']}, 'Incomplete per-probe scores')
        model['interface_binding'] = {}
        for interface in interfaces:
            ids = [r['instance_id'] for r in plan['probes'] if r['interface'] == interface]
            probes = tuple(truth[iid] for iid in ids)
            value = Pool(probes).estimate({iid: raw_scores[iid]['B'] for iid in ids})['value']
            model['interface_binding'][interface] = {'BAcc': value, 'probes': len(ids), 'mothers': len({p.problem_id for p in probes})}
        model['scores_sha256'] = file_hash(scores_path)

    bundle = {'models': visual_models, 'status': 'non_scientific' if preview else 'all_provisional'}
    stamp = ('FIXTURE DATA - NOT MODEL RESULTS' if preview else
             'All-split provisional evaluation; model-reviewed labels; unseen-test status uncertified.')

    class PaperBook(FigureBook):
        def finish(self, fig, name, title, caption, **kwargs):
            if name == '02_reading_binding_quadrants':
                title = 'Reading and binding outcomes'
            if name == '03_gold_reading_control':
                title = 'Binding with exported correct-reading packets'
                kwargs['footer'] = 'Gold supplies the exported local readings. Gold CAcc and JAcc are not evaluated.'
            kwargs['footer'] = (kwargs.get('footer', '') + ' ' + stamp).strip()
            super().finish(fig, name, title, caption + ' ' + stamp, **kwargs)

    style = theme()
    style['svg.fonttype'] = 'none'
    with plt.rc_context(style):
        book = PaperBook(bundle, output / 'figures', width=7.2, dpi=600, formats=('pdf', 'svg', 'png'))
        fig, axes = book.figure(3.8)
        ax = axes[0, 0]
        metrics = [('CAcc', 'CAcc\n(reading L)'), ('BAcc', 'BAcc\n(all binding B)'),
                   ('JAcc', 'JAcc\n(joint L)'), ('BAcc_given_C', 'BAcc | C\n(joint L)')]
        offsets = np.linspace(-.20, .20, len(reports)) if len(reports) > 1 else [0]
        for i, model in enumerate(reports):
            for x, (key, _) in enumerate(metrics):
                value = model['report']['raw_all']['metrics'][key]
                _point(ax, x + offsets[i], value, interval(model, 'raw_all', key), book.colors[i], book.markers[i])
                if value is None:
                    ax.text(x + offsets[i], 4 + 7 * i, 'N/A', ha='center', color=book.colors[i], fontsize=7)
        ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
        book.percent(ax)
        book.legend(ax)
        book.finish(fig, '01_main_results', 'Reading and role binding',
                    'CAcc, JAcc and conditional binding use L; BAcc uses all B. Existing mother-within-task aggregation is preserved. '
                    'Intervals are 95% paired source-cluster bootstrap intervals when defined; missing intervals are not zero-width certainty.',
                    section='main', footer=f"B = {len(plan['probes'])}; L = {sum(r['joint_eligible'] for r in plan['probes'])}. Different metric populations are labeled.",
                    top=.83, bottom=.20)
        book.quadrants()
        book.gold_control()

        fig, axes = book.figure(3.8)
        ax = axes[0, 0]
        for i, model in enumerate(reports):
            for x, interface in enumerate(interfaces):
                _point(ax, x + offsets[i], model['interface_binding'][interface]['BAcc'], None, book.colors[i], book.markers[i])
        labels = [f"{iface}\n(n={reports[0]['interface_binding'][iface]['probes']})" for iface in interfaces]
        ax.set_xticks(range(len(interfaces)), labels)
        book.percent(ax)
        book.legend(ax)
        book.finish(fig, '04_interface_binding', 'Binding by interface',
                    'Descriptive interface-specific binding estimates, aggregated across contributing mothers within each interface. '
                    'No subgroup confidence intervals are computed in this diagnostic. Interfaces are not equally weighted components of the main BAcc.',
                    footer='Descriptive interface results. Main aggregation keeps the original T02 / T03 task definitions.', top=.83, bottom=.20)

    tables = output / 'tables'
    tables.mkdir()
    def number(value): return '--' if value is None else f'{100 * value:.2f}'
    def cell(model, scope, key):
        value = model['report'][scope]['metrics'][key]
        ci = interval(model, scope, key)
        return number(value) + (f' [{100*ci[0]:.2f}, {100*ci[1]:.2f}]' if ci else '')
    def tex_escape(value):
        return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')
    def table(name, headers, rows, latex_headers=None):
        with (tables / (name + '.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        tex = ['% ' + stamp, r'\begingroup\small\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{l' + 'r' * (len(headers)-1) + '}', r'\toprule',
               ' & '.join(latex_headers or [tex_escape(h) for h in headers]) + r' \\', r'\midrule']
        tex += [' & '.join(tex_escape(v) for v in row) + r' \\' for row in rows]
        tex += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '% Requires booktabs. Values are percentages; differences are percentage points. Brackets contain available 95% CIs.']
        atomic_text(tables / (name + '.tex'), '\n'.join(tex) + '\n')
        atomic_text(tables / (name + '.md'), stamp + '\n\n|' + '|'.join(headers) + '|\n|' + '|'.join(['---'] * len(headers)) + '|\n'
                    + '\n'.join('|' + '|'.join(str(v) for v in row) + '|' for row in rows) + '\n')
    table('01_main', ['Model', 'CAcc (%)', 'BAcc (%)', 'JAcc (%)', 'BAcc given C (%)'],
          [[m['label']] + [cell(m, 'raw_all', k) for k in ('CAcc', 'BAcc', 'JAcc', 'BAcc_given_C')] for m in reports])
    table('02_gold', ['Model', 'Paired Raw (%)', 'Gold (%)', 'Gold-Raw (pp)'],
          [[m['label']] + [cell(m, 'paired', k) for k in ('BAcc', 'BAcc_gold', 'delta_BAcc_gold_raw')] for m in reports])
    table('03_quadrants', ['Model', 'C1 B1 (%)', 'C1 B0 (%)', 'C0 B1 (%)', 'C0 B0 (%)'],
          [[m['label']] + [number(m['report']['raw_all']['quadrants'][k]) for k in ('q11', 'q10', 'q01', 'q00')] for m in reports])
    table('04_interfaces', ['Model', 'Interface', 'BAcc (%)', 'Probes', 'Mothers'],
          [[m['label'], iface, number(row['BAcc']), row['probes'], row['mothers']]
           for m in reports for iface, row in m['interface_binding'].items()])
    public_data = {'schema_version': 'physalign_two_experiment_figures_v1', 'status': stamp,
                   'visual_profile': ctx.visual_profile,
                   'plan_hash': plan['plan_hash'], 'bootstrap': plan['bootstrap'], 'dataset': ds.manifest,
                   'models': [{'model_id': m['model_id'], 'label': m['label'], 'report_sha256': m['report_sha256'],
                               'run_manifest_sha256': m['manifest_sha256'], 'scores_sha256': m['scores_sha256'],
                               'raw_all': m['report']['raw_all'], 'paired': m['report']['paired'],
                               'intervals': m['report']['intervals'], 'interface_binding': m['interface_binding']} for m in reports]}
    atomic_json(output / 'figure_data.json', public_data)
    captions = '# Figure captions\n\n' + stamp + '\n\n' + '\n\n'.join('## ' + e['id'] + '\n\n' + e['caption'] for e in book.entries)
    captions += '\n\nIntervals use connected components of shared original-image hashes. Semantic/near-duplicate certification and independent human certification remain incomplete. '
    captions += 'The per-probe sample sizes are descriptive counts, not the averaging weights. Global BAcc uses equal T02/T03 weights; T03-image and T03-text remain in T03.\n'
    if ctx.visual_profile == 'multi-image-256-v1':
        captions += '\nPreprocessing profile multi-image-256-v1: all public attachments and prompts are retained; Qwen images use 65536-262144 pixels (at most 256 visual tokens per image); InternVL uses one 448x448 tile (256 visual tokens) per image. The input ceiling remains 32768 and output reservation 2048 tokens. This is a bounded-resolution evaluation, not checkpoint-default high-resolution processing. Lower resolution may affect reading accuracy.\n'
    atomic_text(output / 'captions.md', captions)
    cards = []
    for entry in book.entries:
        links = ' '.join(f'<a href="figures/{html.escape(entry["files"][fmt])}">{fmt.upper()}</a>' for fmt in ('pdf', 'svg', 'png'))
        cards.append(f'<section><h2>{html.escape(entry["title"])}</h2><img src="figures/{html.escape(entry["files"]["svg"])}">'
                     f'<p>{links}</p><p>{html.escape(entry["caption"])}</p></section>')
    atomic_text(output / 'index.html', '<!doctype html><meta charset="utf-8"><title>PhysAlign: Raw and Gold</title>'
                '<style>body{font:16px system-ui;max-width:1000px;margin:35px auto;padding:0 20px;color:#203047}section{margin:40px 0}img{width:100%}a{margin-right:18px}</style>'
                '<h1>PhysAlign: Raw and Gold</h1><p>' + html.escape(stamp) + '</p><p><a href="figure_data.json">Full-precision data</a>'
                '<a href="captions.md">Captions</a></p>' + ''.join(cards))
    artifact_hashes = {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output / 'figure_manifest.json', {'plan_hash': plan['plan_hash'], 'helper_sha256': file_hash(Path(__file__)),
                'status': stamp, 'width_inches': 7.2, 'png_dpi': 600, 'figures': book.entries, 'files': artifact_hashes})
    log(stage='figures_written', directory=str(output), gallery=str(output / 'index.html'))
    return output


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError, ImportError, AttributeError) as error:
        print(f'PhysAlign two-experiment pipeline: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise SystemExit(2)
