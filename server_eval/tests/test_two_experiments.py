"""Contract fixtures only; no downloaded models and no fabricated benchmark claims."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('two_experiments', ROOT / 'server_eval/two_experiments.py')
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)

from physalign.adapters import AdapterInfo, ModelResponse
from physalign.controls import compose
from physalign.dataset import PublicDataset, load_truth, sections
from physalign.planning import prepare_plan, save_plan
from physalign.reporting import score_run
from physalign.runner import run_evaluation
from physalign.storage import atomic_json, canonical, file_hash, read_json


def rehash(root):
    atomic_json(root / 'FILE_MANIFEST.json', {p.relative_to(root).as_posix(): file_hash(p)
                for p in root.rglob('*') if p.is_file() and p.name != 'FILE_MANIFEST.json'})


def fixture(destination):
    samples = ROOT / 'examples/synthetic'
    shutil.copytree(samples / 'public', destination / 'public')
    raw, gold = read_json(samples / 'public/qa_raw.json'), read_json(samples / 'public/qa_gold.json')
    answers = read_json(samples / 'private/answers.json')
    source_members = {r['instance_id']: r for r in read_json(samples / 'private/membership.json')}
    members, maps = [], []
    for i, row in enumerate(raw):
        iid = row['instance_id']
        row['split'] = 'development' if i % 2 else 'test'
        for g in gold:
            if g['instance_id'] == iid:
                g['split'] = row['split']
        original = source_members[iid]
        members.append({**original, 'split': row['split'], 'review_status': 'model_approved', 'human_reviewed': False,
                        'component': 'fixture', 'original_split': 'unassigned'})
        maps.append({'instance_id': iid, 'human_reviewed': False, 'review_method': 'fixture', 'review_target_root': 'fixture',
                     'key': read_json(samples / f'private/evidence/{iid}/private_key.json')})
    # Add a binding-only T03-text contract, sharing a synthetic original image.
    text = deepcopy(raw[3])
    text.update(instance_id='fixture_text', logical_probe_id='fixture_text_logical', interface='T03-text')
    parts = sections(text['input']['messages']['user'])
    parts['Output format'] = canonical({'owner': '<candidate ID>'})
    text['input']['messages']['user'] = compose(parts)
    raw.append(text)
    member = deepcopy(members[3])
    member.update(instance_id=text['instance_id'], logical_probe_id=text['logical_probe_id'], interface='T03-text')
    members.append(member)
    mapping = deepcopy(maps[3])
    mapping['instance_id'] = text['instance_id']
    mapping['key'].update(logical_probe_id=text['logical_probe_id'], read_targets=[], packet_plan={'state': 'empty'})
    maps.append(mapping)
    answer = deepcopy(answers[3])
    answer.update(instance_id=text['instance_id'], logical_probe_id=text['logical_probe_id'], interface='T03-text', read_normalizer_id=None)
    del answer['answer']['read']
    answers.append(answer)
    release = {'schema_version': pipeline.RELEASE_SCHEMA, 'attachment_paths_relative_to': 'public',
               'release_status': 'model_reviewed_provisional', 'synthetic_example_only': False,
               'human_reviewed': False, 'unseen_test_split_certified': False, 'quarantined_mother_ids': [],
               'counts': {'raw': len(raw), 'gold': len(gold), 'mothers': len({m['problem_id'] for m in members})}}
    for name, value in [('public/qa_raw.json', raw), ('public/qa_gold.json', gold), ('private/membership.json', members),
                        ('private/mappings.json', maps), ('private/answers.json', answers), ('release.json', release)]:
        atomic_json(destination / name, value)
    rehash(destination)
    return destination


class FixtureAdapter:
    """Deterministic contract exercise, explicitly NOT a real model result."""
    def __init__(self, source, dataset, variant=0):
        self.info = AdapterInfo('fixture', f'fixture-{variant}', 'fixture', '1', '{}', '{}', False)
        self.calls = 0
        answers = {r['instance_id']: r for r in read_json(source / 'private/answers.json')}
        self.outputs = {}
        for i, ((iid, condition), item) in enumerate(sorted(dataset.items.items())):
            row = answers[iid]
            answer = deepcopy(row['answer'])
            binding = 'referent' if item.record['task_id'] == 'T02' else 'owner'
            if (i + variant) % (4 if condition == 'gold' else 3) == 0:
                answer[binding] = next(x for x in row['candidate_ids'] if x != answer[binding])
            if 'read' in answer and condition == 'raw' and (i + variant) % 3 == 1:
                answer['read'] = 'FIXTURE_WRONG_READING'
            self.outputs[item.record['input']['messages']['user']] = canonical(answer)

    def generate(self, request):
        self.calls += 1
        return ModelResponse(self.outputs[request.user], 'stop')


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='physalign-two-')
        self.root = Path(self.tmp.name)
        self.source = fixture(self.root / 'source')
        self.dest = self.root / 'converted'

    def tearDown(self):
        self.tmp.cleanup()

    def test_conversion_keeps_inputs_pairing_provenance_and_flags(self):
        before = {p.relative_to(self.source).as_posix(): file_hash(p) for p in self.source.rglob('*') if p.is_file()}
        manifest = pipeline.convert_release(self.source, self.dest)
        ds = PublicDataset(self.dest)
        truth, _ = load_truth(ds)
        self.assertEqual((manifest['counts']['raw'], manifest['joint_probes'], manifest['counts']['gold']), (7, 3, 3))
        self.assertEqual(manifest['source_clusters'], 3)
        self.assertFalse(manifest['formal_release_eligibility_checked'])
        self.assertFalse(truth['fixture_text'].joint_eligible)
        for row in read_json(self.source / 'public/qa_raw.json'):
            target = ds.items[row['instance_id'], 'raw'].record
            self.assertEqual(row['input'], target['input'])
            self.assertEqual(target['split'], pipeline.SPLIT)
        after = {p.relative_to(self.source).as_posix(): file_hash(p) for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(pipeline.convert_release(self.source, self.dest), manifest)

    def test_hashed_text_export_preserves_public_prompt(self):
        from physalign.storage import digest
        rows = read_json(self.source / 'public/qa_raw.json')
        row = rows[0]
        parts = sections(row['input']['messages']['user'])
        evidence = pipeline.loads(parts['Evidence locations and candidates'])
        text = parts['Original statement']
        evidence['anchors'][next(iter(evidence['anchors']))] = {
            'kind': 'text', 'section': 'stem', 'start': 0, 'end': 1,
            'quote': text[:1], 'section_sha256': digest(text.encode('utf-8'))}
        parts['Evidence locations and candidates'] = canonical(evidence)
        row['input']['messages']['user'] = compose(parts)
        atomic_json(self.source / 'public/qa_raw.json', rows)
        rehash(self.source)
        pipeline.convert_release(self.source, self.dest)
        actual = PublicDataset(self.dest).items[row['instance_id'], 'raw'].record
        self.assertEqual(actual['input'], row['input'])

    def test_new_visual_profile_reuses_bundle_and_freezes_new_configs(self):
        pipeline.convert_release(self.source, self.dest)
        before = file_hash(self.dest / 'FILE_MANIFEST.json')
        for name, model_id, _, size in pipeline.JOBS:
            atomic_json(self.root / 'models' / (name + '.snapshot.json'), {'model_id': model_id})
            atomic_json(self.root / 'configs/server' / (name + '.json'), {
                'gpu_count': size, 'thinking': False, 'dtype': 'bfloat16', 'seed': 2027,
                'max_input_tokens': 32768, 'max_new_tokens': 2048,
                'snapshot_manifest': 'models/' + name + '.snapshot.json'})
        ctx = pipeline.Context(self.root, self.source, 'new-profile', 40, self.dest, 'multi-image-256-v1')
        with patch.dict('os.environ', PHYSALIGN_AVAILABLE_GPUS='1,3,4,5,6,7'), \
             patch.object(pipeline, 'convert_release', side_effect=AssertionError('Must reuse existing bundle')):
            plan = pipeline.prepare(ctx)
            self.assertEqual(pipeline.checked_plan(ctx)['plan_hash'], plan['plan_hash'])
        self.assertEqual(file_hash(self.dest / 'FILE_MANIFEST.json'), before)
        cfg = read_json(ctx.config_path('internvl35-8b'))
        self.assertEqual(cfg['processor_call_kwargs']['images_kwargs']['max_patches'], 1)
        self.assertEqual((cfg['gpu_count'], cfg['max_input_tokens']), (2, 32768))
        self.assertEqual(len(plan['requests']), 10)

    def test_hash_corruption_is_rejected(self):
        with (self.source / 'public/qa_raw.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            pipeline.convert_release(self.source, self.dest)

    def test_quarantined_mother_is_rejected_without_silent_filter(self):
        value = read_json(self.source / 'release.json')
        value['quarantined_mother_ids'] = [read_json(self.source / 'private/membership.json')[0]['problem_id']]
        atomic_json(self.source / 'release.json', value)
        rehash(self.source)
        with self.assertRaisesRegex(ValueError, 'Quarantined'):
            pipeline.convert_release(self.source, self.dest)

    def test_wrong_simplified_answer_is_rejected(self):
        rows = read_json(self.source / 'private/answers.json')
        rows[0]['answer']['referent'] = 'E1'
        atomic_json(self.source / 'private/answers.json', rows)
        rehash(self.source)
        with self.assertRaisesRegex(ValueError, 'contradicts canonical'):
            pipeline.convert_release(self.source, self.dest)

    def test_gold_cannot_reveal_a_different_reading(self):
        rows = read_json(self.source / 'public/qa_gold.json')
        parts = sections(rows[0]['input']['messages']['user'])
        parts['Local reading information'] = canonical([{'anchor_id': 'R1', 'text': 'WRONG'}])
        rows[0]['input']['messages']['user'] = compose(parts)
        atomic_json(self.source / 'public/qa_gold.json', rows)
        rehash(self.source)
        with self.assertRaisesRegex(ValueError, 'Gold differs'):
            pipeline.convert_release(self.source, self.dest)

    def test_original_pair_splits_are_not_silently_repaired(self):
        rows = read_json(self.source / 'public/qa_gold.json')
        rows[0]['split'] = 'other'
        atomic_json(self.source / 'public/qa_gold.json', rows)
        rehash(self.source)
        with self.assertRaisesRegex(ValueError, 'Source Raw/Gold split mismatch'):
            pipeline.convert_release(self.source, self.dest)

    def test_shared_original_image_mothers_form_one_cluster(self):
        rows = read_json(self.source / 'private/membership.json')
        rows[0]['problem_id'] = 'fixture-additional-mother'
        atomic_json(self.source / 'private/membership.json', rows)
        release = read_json(self.source / 'release.json')
        release['counts']['mothers'] = 4
        atomic_json(self.source / 'release.json', release)
        rehash(self.source)
        manifest = pipeline.convert_release(self.source, self.dest)
        self.assertEqual(manifest['counts']['mothers'], 4)
        self.assertEqual(manifest['source_clusters'], 3)

    def test_partial_gold_run_score_and_resume_preserve_first_responses(self):
        pipeline.convert_release(self.source, self.dest)
        plan = prepare_plan(self.dest, split=pipeline.SPLIT, conditions=('raw', 'gold'), bootstrap_resamples=40)
        adapter = FixtureAdapter(self.source, PublicDataset(self.dest))
        run = self.root / 'run'
        run_evaluation(plan, adapter, run)
        self.assertEqual(adapter.calls, 10)
        before = file_hash(run / 'predictions.jsonl')
        report = score_run(run)
        self.assertFalse(report['scientific_run'])
        self.assertEqual(report['service']['completed_outputs'], 10)
        self.assertEqual(len(plan['paired_ids']), 3)
        run_evaluation(plan, adapter, run, resume=True)
        self.assertEqual(adapter.calls, 10)
        self.assertEqual(before, file_hash(run / 'predictions.jsonl'))

    def test_record_tampering_is_rejected(self):
        pipeline.convert_release(self.source, self.dest)
        plan = prepare_plan(self.dest, split=pipeline.SPLIT, conditions=('raw', 'gold'), bootstrap_resamples=40)
        run = self.root / 'run'
        run_evaluation(plan, FixtureAdapter(self.source, PublicDataset(self.dest)), run)
        path = next((run / 'results').glob('*.json'))
        record = read_json(path)
        record['response']['text'] = '{}'
        atomic_json(path, record)
        with self.assertRaises(ValueError):
            score_run(run)


class SchedulingTests(unittest.TestCase):
    def test_launcher_releases_pair_before_starting_third_worker(self):
        with tempfile.TemporaryDirectory(prefix='physalign-scheduler-') as directory:
            ctx = pipeline.Context(Path(directory), Path(directory) / 'source', 'schedule')
            plan = {'plan_hash': 'fixture-plan'}
            for name, *_ in pipeline.JOBS:
                atomic_json(ctx.config_path(name), {})
                atomic_json(ctx.inputs / (name + '.preflight.json'), {
                    'plan_hash': plan['plan_hash'], 'configuration_hash': pipeline.fingerprint({}), 'all_within_budget': True})
            launched = []
            class Child:
                def __init__(self, index): self.index, self.pid = index, 100 + index
                def poll(self): return None if self.index == 0 and len(launched) < 3 else 0
            def launch(command, **kwargs):
                child = Child(len(launched))
                launched.append((command, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                return child
            with patch.dict('os.environ', PHYSALIGN_AVAILABLE_GPUS='1,3,4,5,6,7'), \
                 patch.object(pipeline, 'checked_plan', return_value=plan), \
                 patch.object(pipeline, 'verify_public_plan', return_value=Mock()), \
                 patch.object(pipeline.subprocess, 'Popen', side_effect=launch), \
                 patch.object(pipeline.time, 'sleep'):
                pipeline.run_models(ctx)
            self.assertEqual([group for _, group in launched], ['1,3,4,5', '6,7', '6,7'])
            assigned = read_json(ctx.runs / 'gpu_assignments.json')
            self.assertEqual(assigned['internvl35-8b'], ['6', '7'])
            self.assertTrue(all('--adapter' in command and 'hf' in command for command, _ in launched))

    def test_six_card_launch_and_refill_avoid_unavailable_cards(self):
        ids = ('1', '3', '4', '5', '6', '7')
        first = pipeline.allocate_jobs(list(pipeline.JOBS), set(), ids)
        self.assertEqual([(j[0], group) for j, group in first], [
            ('qwen35-27b', ('1', '3', '4', '5')), ('qwen35-9b', ('6', '7'))])
        remaining = [j for j in pipeline.JOBS if j not in [entry[0] for entry in first]]
        self.assertEqual(pipeline.allocate_jobs(remaining, set(ids), ids), [])
        after_small = pipeline.allocate_jobs(remaining, {'1', '3', '4', '5'}, ids)
        self.assertEqual(after_small[0][1], ('6', '7'))
        after_large = pipeline.allocate_jobs(remaining, {'6', '7'}, ids)
        self.assertEqual(after_large[0][1], ('1', '3'))

    def test_resume_retains_original_gpu_group(self):
        ids = ('1', '3', '4', '5', '6', '7')
        pending = [pipeline.JOBS[0]]
        prior = {'qwen35-9b': ['6', '7']}
        self.assertEqual(pipeline.allocate_jobs(pending, set(), ids, prior)[0][1], ('6', '7'))
        self.assertEqual(pipeline.allocate_jobs(pending, {'6'}, ids, prior), [])

    def test_gpu_pool_rejects_duplicates_or_too_few_cards(self):
        for value in ['1,1,3,4', '1,3', '1,3,4,-1']:
            with self.subTest(value=value), patch.dict('os.environ', PHYSALIGN_AVAILABLE_GPUS=value), self.assertRaises(ValueError):
                pipeline.available_gpus()


if __name__ == '__main__':
    unittest.main()
