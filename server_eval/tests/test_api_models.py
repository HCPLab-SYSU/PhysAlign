"""Cross-model scoring, recovery and comparison with synthetic HTTP responses."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from physalign.adapters import InfrastructureError
from physalign.dataset import PublicDataset
from physalign.planning import prepare_plan
from physalign.runner import run_evaluation
from physalign.storage import atomic_json, file_hash, read_json
from server_eval import api_models, compare_models
from test_api_adapter import config, envelope, environment
from test_two_experiments import fixture, FixtureAdapter, pipeline


class APIPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='physalign-api-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = fixture(self.root / 'release')
        self.bundle = self.root / 'bundle'
        with redirect_stdout(StringIO()):
            pipeline.convert_release(self.source, self.bundle)
        self.plan = prepare_plan(self.bundle, split=pipeline.SPLIT, bootstrap_resamples=10)
        self.outputs = FixtureAdapter(self.source, PublicDataset(self.bundle)).outputs
        self.configs = {name: config(name) for name in api_models.MODEL_NAMES}
        self.run_root = self.root / 'api-runs'
        self.env = environment()
        self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def http(self, adapter, suffix, payload=None):
        if suffix == '/models':
            return {'data': [{'id': adapter.config['model_id']}]}
        user = payload['messages'][1]['content'][0]['text']
        return envelope(self.outputs[user], model=adapter.config['model_id'])

    def run_apis(self):
        with patch.object(api_models.APIAdapter, '_json_request', autospec=True, side_effect=self.http) as http, redirect_stdout(StringIO()):
            api_models.run_models(self.plan, self.configs, self.run_root)
        return http.call_count

    def test_same_plan_produces_same_metrics_as_existing_local_fixture(self):
        self.assertEqual(self.run_apis(), 2 * len(self.plan['requests']))
        local = self.root / 'local'
        run_evaluation(self.plan, FixtureAdapter(self.source, PublicDataset(self.bundle)), local)
        with redirect_stdout(StringIO()):
            api_reports = api_models.score_models(self.plan, self.configs, self.run_root)
        from physalign.reporting import score_run
        expected = score_run(local)
        for report in api_reports:
            for key in ('raw_all', 'paired', 'intervals', 'pairing', 'uniform_candidate_baseline'):
                self.assertEqual(report[key], expected[key])
        manifest = read_json(self.run_root / 'gpt-6-astra-high/manifest.json')
        self.assertEqual(manifest['plan'], self.plan)
        self.assertEqual(manifest['adapter']['settings']['reasoning_effort'], 'high')

    def test_resume_completed_apis_makes_no_paid_calls(self):
        self.run_apis()
        hashes = {str(p): file_hash(p) for p in self.run_root.rglob('*.json')}
        self.assertEqual(self.run_apis(), 0)
        self.assertEqual(hashes, {str(p): file_hash(p) for p in self.run_root.rglob('*.json')})

    def test_resume_rejects_changed_api_model_settings(self):
        self.run_apis()
        self.configs['gpt-6-astra-high']['max_output_tokens'] += 1
        with self.assertRaises(ValueError):
            self.run_apis()

    def test_missing_api_response_preserves_denominators_and_blocks_figures(self):
        def missing(adapter, suffix, payload=None):
            raise InfrastructureError('fixture_timeout', retryable=False)
        with patch.object(api_models.APIAdapter, '_json_request', autospec=True, side_effect=missing), redirect_stdout(StringIO()):
            api_models.run_models(self.plan, self.configs, self.run_root)
            reports = api_models.score_models(self.plan, self.configs, self.run_root)
        self.assertIsNone(reports[0]['raw_all']['metrics']['BAcc'])
        self.assertEqual(reports[0]['raw_all']['support']['binding_raw']['planned_probes'], 7)
        with self.assertRaisesRegex(ValueError, 'Incomplete run'):
            compare_models.audit_models(self.plan, [self.run_root / 'gpt-6-astra-high'])

    def test_comparison_accepts_explicit_model_specific_budgets_and_records_them(self):
        self.run_apis()
        runs = [self.run_root / name for name in api_models.MODEL_NAMES]
        models, _ = compare_models.audit_models(self.plan, runs)
        rows = compare_models.protocol_rows(models)
        self.assertEqual([r['output_limit'] for r in rows], [16384, 8192])
        self.assertEqual([r['reasoning'] for r in rows], ['high', 'provider_default'])
        self.assertEqual(rows[0]['endpoint'], 'https://api.example.invalid/v1')

    def test_comparison_rejects_duplicate_models_and_different_plans(self):
        self.run_apis()
        run = self.run_root / 'gpt-6-astra-high'
        with self.assertRaisesRegex(ValueError, 'Duplicate model'):
            compare_models.audit_models(self.plan, [run, run])
        other = prepare_plan(self.bundle, split=pipeline.SPLIT, bootstrap_resamples=12)
        with self.assertRaisesRegex(ValueError, 'exact plan'):
            compare_models.audit_models(other, [run])

    def test_comparison_reaudits_responses_instead_of_trusting_edited_summary(self):
        self.run_apis()
        with redirect_stdout(StringIO()):
            api_models.score_models(self.plan, self.configs, self.run_root)
        run = self.run_root / 'gpt-6-astra-high'
        path = run / 'report.json'
        summary = read_json(path)
        expected = summary['raw_all']['metrics']['BAcc']
        summary['raw_all']['metrics']['BAcc'] = 123
        atomic_json(path, summary)
        models, _ = compare_models.audit_models(self.plan, [run])
        self.assertEqual(models[0]['report']['raw_all']['metrics']['BAcc'], expected)

    def test_custom_adapter_factory_runs_existing_five_experiment_study(self):
        from physalign.cli import _adapter
        from physalign.study import prepare_study, run_study
        from physalign.study_reporting import score_study
        study = self.root / 'study'
        plan = prepare_study(self.bundle, study, split=pipeline.SPLIT, bootstrap_resamples=0)
        adapter = _adapter('server_eval.api_adapter:create_adapter', self.configs['gpt-6-astra-high'])
        with patch.object(adapter, '_json_request', return_value=envelope('{}')) as http:
            run_study(study, adapter, self.root / 'study-run')
        self.assertEqual(http.call_count, len(plan['requests']))
        report = score_study(study, self.root / 'study-run')
        self.assertIn('main', report['experiments'])
        self.assertIn('no_image_geometry', report['experiments'])
        self.assertEqual(report['service']['completed'], len(plan['requests']))

    def test_five_model_preview_exports_all_populations_and_protocols(self):
        runs = []
        for i in range(5):
            directory = self.root / f'fixture-{i}'
            run_evaluation(self.plan, FixtureAdapter(self.source, PublicDataset(self.bundle), i), directory)
            runs.append(directory)
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            compare_models.audit_models(self.plan, runs)
        models, manifest = compare_models.audit_models(self.plan, runs, allow_non_scientific=True)
        output = self.root / 'figures'
        compare_models.render_comparison(self.plan, models, manifest, output, dpi=72, formats=('svg',))
        data = read_json(output / 'figure_data.json')
        self.assertEqual(len(data['models']), 5)
        self.assertIn('SYNTHETIC', data['status'])
        self.assertEqual(len(read_json(output / 'figure_manifest.json')['figures']), 4)
        self.assertEqual(data['models'][0]['report']['pairing']['planned_paired_probes'], 3)
        self.assertTrue((output / 'tables/05_binding_diagnostics.csv').exists())
        self.assertTrue((output / 'protocol.csv').exists())
        with self.assertRaises(ValueError):
            compare_models.render_comparison(self.plan, models, manifest, output, dpi=72, formats=('svg',))


if __name__ == '__main__':
    unittest.main()
