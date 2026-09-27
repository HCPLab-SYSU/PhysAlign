"""Synthetic integration tests; no API calls or changes to real runs."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from test_supplement_eval import SupplementTests, SyntheticAdapter
from physalign.adapters import AdapterInfo, InfrastructureError, ModelResponse
from physalign.runner import run_evaluation
from physalign.storage import atomic_json, canonical, file_hash, fingerprint, loads, read_json
from server_eval.api_adapter import APIAdapter
from server_eval.supplement_eval import run as run_supplement
from server_eval.llm_judge import make_plan, run, collect_grades, parse_judgment, validate_plan
from server_eval.llm_judge_reporting import score, compare


class FakeJudge:
    info = AdapterInfo('judge-fixture', 'synthetic-judge', 'fixture', '1', '{}', '{}', False)

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or [ModelResponse('{"verdict":"correct","rationale":"Synthetic reference agreement"}', 'stop'),
                                       ModelResponse('{"verdict":"incorrect","rationale":"Synthetic missing subpart"}', 'stop')]

    def generate(self, request):
        index = len(self.calls)
        self.calls.append(request)
        item = self.responses[index % len(self.responses)]
        if isinstance(item, Exception):
            raise item
        return item


class JudgeTests(unittest.TestCase):
    def setUp(self):
        SupplementTests.setUp(self)
        for mother in sorted(self.keys['problems'])[:2]:
            self.keys['problems'][mother]['answer_key'] = {
                'kind': 'human', 'reference_answer': canonical({'final_answer': 'A', 'solution': 'synthetic reference'}),
                'reliable': True, 'source_reference': 'synthetic'}
        atomic_json(self.keyfile, self.keys)
        self.study = SupplementTests.build(self)
        self.target = self.root / 'target'
        self.solver = SyntheticAdapter(self.source)
        run_supplement(self.study, self.solver, self.target)
        self.config = read_json(ROOT / 'configs/judges/gpt56-sol-medium.json')
        self.plan = make_plan(self.study, self.target, self.config, interval_seconds=0)
        self.judge_run = self.root / 'judge'

    def judge(self, adapter=None):
        adapter = adapter or FakeJudge()
        run(self.study, self.target, self.plan, self.judge_run, adapter=adapter)
        return adapter

    def test_score_resume_association_and_plots_without_mutating_target(self):
        before = {p.relative_to(self.target).as_posix(): file_hash(p) for p in self.target.rglob('*') if p.is_file()}
        adapter = self.judge()
        run(self.study, self.target, self.plan, self.judge_run, adapter=adapter, resume=True)
        self.assertEqual(len(adapter.calls), 2)
        main = self.root / 'main'
        run_evaluation(self.base, self.solver, main, dataset_root=self.root / 'bundle')
        report = score(self.study, self.target, self.judge_run, self.root / 'report', main_run=main)
        self.assertAlmostEqual(report['original_solving']['SolveAcc'], 2/3)
        self.assertEqual(report['original_solving']['automatic_subset']['SolveAcc'], 1)
        self.assertIsNotNone(report['original_solving']['association'])
        self.assertIsNone(report['manual_grades_hash'])
        self.assertFalse(report['grading']['human_confirmed'])
        self.assertEqual(report['grading']['graded'], 2)
        compare([self.root / 'report/report.json'], self.root / 'plots')
        self.assertIn('SolveAcc_LLM_judged', (self.root / 'plots/comparison.csv').read_text('utf-8-sig'))
        self.assertTrue((self.root / 'plots/comparison.png').is_file())
        after = {p.relative_to(self.target).as_posix(): file_hash(p) for p in self.target.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_anonymous_judge_sees_reference_candidate_and_all_original_images(self):
        adapter = self.judge()
        self.assertEqual(len(self.plan['eligible_problem_ids']), 2)
        for row, req in zip(self.plan['requests'], adapter.calls):
            payload = loads(req.user)
            self.assertEqual(set(payload), {'original_question', 'reference', 'candidate_response'})
            self.assertNotIn(self.solver.info.model_id, req.user)
            self.assertNotIn('target_request_id', req.user)
            self.assertEqual(payload['reference']['final_answer'], 'A')
            self.assertEqual(len(req.images), len(row['input']['attachments']))
            self.assertGreater(len(req.images), 0)

    def test_failed_or_ungradable_judge_does_not_score_zero_or_shrink_denominator(self):
        self.judge(FakeJudge([InfrastructureError('synthetic_timeout', retryable=False),
                             ModelResponse('{"verdict":"ungradable","rationale":"Reference conflict"}', 'stop')]))
        report = score(self.study, self.target, self.judge_run, self.root / 'report')
        solving = report['original_solving']
        self.assertIsNone(solving['SolveAcc'])
        self.assertEqual(solving['answer_eligible_mothers'], 3)
        self.assertEqual(len(solving['missing_or_ungraded']), 2)
        self.assertEqual(solving['SolveAcc_bounds'], {'lower': 1/3, 'upper': 1})
        self.assertEqual(report['grading']['graded'], 0)

    def test_strict_judge_response_parsing(self):
        invalid = ['not json', '{"verdict":"correct","rationale":"ok","extra":1}',
                   '{"verdict":"correct","verdict":"incorrect","rationale":"ok"}',
                   '{"verdict":"correct","rationale":""}',
                   '{"verdict":[],"rationale":"ok"}', 'null']
        for value in invalid:
            self.assertIsNone(parse_judgment({'text': value, 'finish_reason': 'stop'})[0])
        self.assertIsNone(parse_judgment({'text':'{"verdict":"correct","rationale":"ok"}', 'finish_reason':'length'})[0])
        self.assertEqual(parse_judgment({'text':'{"verdict":"incorrect","rationale":"Wrong unit"}', 'finish_reason':'stop'})[0], 0)

    def test_rejects_cross_run_and_changed_response_binding(self):
        other = self.root / 'other'
        run_supplement(self.study, SyntheticAdapter(self.source), other)
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_plan(self.study, other, self.plan)
        tampered = loads(canonical(self.plan))
        tampered['requests'][0]['target_response_hash'] = 'wrong'
        tampered['plan_hash'] = fingerprint({k: v for k, v in tampered.items() if k != 'plan_hash'})
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_plan(self.study, self.target, tampered)

    def test_rejects_changed_logged_judge_prompt(self):
        self.judge()
        path = next((self.judge_run / 'requests').glob('*.json'))
        data = read_json(path); data['messages']['user'] = 'new candidate'
        atomic_json(path, data)
        with self.assertRaises(ValueError):
            collect_grades(self.study, self.target, self.judge_run)

    def test_rejects_mixed_judge_protocol_and_human_report(self):
        self.judge()
        report = score(self.study, self.target, self.judge_run, self.root / 'report')
        other = loads(canonical(report)); other['run_id'] = 'another'
        other['grading']['protocol_hash'] = 'another protocol'
        atomic_json(self.root / 'other.json', other)
        with self.assertRaisesRegex(ValueError, 'protocols differ'):
            compare([self.root / 'report/report.json', self.root / 'other.json'], self.root / 'bad')
        with self.assertRaisesRegex(ValueError, 'only LLM-judge'):
            compare([self.root / 'report/deterministic/report.json'], self.root / 'bad')

    def test_real_transport_payload_preserves_requested_model_medium_and_images(self):
        import os
        config = {**self.config, 'api_key_env': 'PHYSALIGN_TEST_JUDGE_API_KEY'}
        with patch.dict(os.environ, {'PHYSALIGN_TEST_JUDGE_API_KEY':'synthetic-placeholder'}), patch('server_eval.api_adapter.load_environment'):
            adapter = APIAdapter(config)
        row = self.plan['requests'][0]
        request = self.study.request(row, 'fixture', adapter.info.settings_json)
        payload = adapter.payload(request)
        self.assertEqual(payload['model'], 'gpt-5.6-sol')
        self.assertEqual(payload['reasoning_effort'], 'medium')
        self.assertNotIn('temperature', payload)
        self.assertEqual(sum(x['type']=='image_url' for x in payload['messages'][1]['content']), len(row['input']['attachments']))


if __name__ == '__main__':
    unittest.main()
