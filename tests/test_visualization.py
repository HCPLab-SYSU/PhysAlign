"""Hand-calculated estimands, audited input failures and real offline exports.

Every answer in this file is a synthetic test fixture, not a model result.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import csv
import importlib.util
import shutil
import tempfile
import unittest
import warnings

from test_evaluation import RecordingAdapter, native_bundle
from physalign.adapters import InfrastructureError, ModelResponse
from physalign.contracts import decode_probe
from physalign.dataset import sections
from physalign.human import AUDIT_CHECKS, create_session, seal_study, submit
from physalign.storage import atomic_json, canonical, file_hash, read_json
from physalign.study import SOLVE_SYSTEM, Study, draft_spec, prepare_study, run_study
from physalign.study_reporting import score_study
from physalign_viz.cli import main
from physalign_viz.data import BINDING_BINS, binned_solving, boot, load_bundle, metric_estimate
from physalign_viz.export import export_bundle, gallery
from physalign_viz.tables import latex_cell, latex_escape, table_specs


def fixture_answer(request, *, perfect=False, missing=False):
    if request.user.startswith('ORIGINAL '):
        return ModelResponse('{"answer":"A"}' if request.user == 'ORIGINAL A' else '{"answer":"B"}')
    parts = sections(request.user)
    iid = parts['Local question'].split()[-1]
    if missing and iid == 'p2':
        raise InfrastructureError('deliberate_missing_fixture', retryable=False)
    gold = parts['Local reading information'] != 'None.'
    return ModelResponse(canonical({'read': '2 kg' if perfect or iid != 'p2' else '3 kg',
                                    'owner': 'E1' if perfect or gold or iid in {'p1', 'p4'} else 'E2'}))


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='physalign-viz-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        bundle = native_bundle(cls.root / 'bundle')
        spec = draft_spec(bundle)
        for mother in ('A', 'B', 'C'):
            spec['original_problems'][mother] = {
                'original': {'system': SOLVE_SYSTEM, 'user': 'ORIGINAL ' + mother,
                             'asset_ids': ['img_0'], 'source_reference': 'synthetic-fixture',
                             'unchanged_original_verified': True},
                'answer_key': {'kind': 'choice', 'accepted': ['A'], 'reliable': True,
                               'source_reference': 'synthetic-fixture-key'}}
        atomic_json(cls.root / 'spec.json', spec)
        cls.study = cls.root / 'study'
        cls.plan = prepare_study(bundle, cls.study, split='development',
                                spec_path=cls.root / 'spec.json', bootstrap_resamples=20)
        cls.runs = []
        for i in range(2):
            adapter = RecordingAdapter(lambda req, i=i: fixture_answer(req, perfect=bool(i)))
            adapter.info = replace(adapter.info, model_id=f'Test model {i}')
            run = cls.root / f'run{i}'
            run_study(cls.study, adapter, run)
            score_study(cls.study, run)
            cls.runs.append(run)
        cls.bundle = load_bundle(cls.study, cls.runs, allow_non_scientific=True)

    def new_output(self):
        temp = tempfile.TemporaryDirectory(prefix='physalign-viz-export-')
        self.addCleanup(temp.cleanup)
        return Path(temp.name) / 'output'

    def test_display_preserves_calculation_order_and_gold_population(self):
        model = self.bundle['models'][0]
        for key, value in {'CAcc': .75, 'JAcc': .25, 'BAcc': .375, 'BAcc_given_C': 1/3}.items():
            self.assertAlmostEqual(metric_estimate(model, 'raw_all', key)['value'], value)
        self.assertEqual(metric_estimate(model, 'paired', 'BAcc')['value'], .25)
        self.assertEqual(metric_estimate(model, 'paired', 'BAcc_gold')['value'], 1)
        self.assertEqual(metric_estimate(model, 'paired', 'delta_BAcc_gold_raw')['value'], .75)
        tables = {t['id']: t for t in table_specs(self.bundle)}
        self.assertTrue(tables['02_gold']['rows'][0][1].startswith('25.0'))
        self.assertTrue(tables['02_gold']['rows'][0][3].startswith('75.0'))
        self.assertTrue(tables['01_main']['rows'][0][4].startswith('66.7'))

    def test_quadrants_are_joint_population_events_not_full_b(self):
        q = self.bundle['models'][0]['report']['experiments']['main']['raw_all']['quadrants']
        self.assertEqual([q[k] for k in ('q11', 'q10', 'q01', 'q00')], [.25, .5, 0, .25])
        self.assertEqual(sum(q.values()), 1)

    def test_eq8_ordinary_mother_mean_and_original_denominator(self):
        model = self.bundle['models'][0]
        rows = {r['problem_id']: r for r in model['diagnostics']['mother_records']}
        self.assertAlmostEqual(rows['A']['binding_mean'], 2/3)
        self.assertNotEqual(rows['A']['binding_mean'], .75)  # Incorrect type balancing.
        self.assertEqual(rows['A']['binding_probes'], 3)
        self.assertAlmostEqual(metric_estimate(model, 'solve', 'SolveAcc')['value'], 1/3)
        self.assertEqual(metric_estimate(model, 'solve', 'SolveAcc')['support']['planned_problems'], 3)

    def test_gold_transition_events_precede_weighted_aggregation(self):
        t = self.bundle['models'][0]['diagnostics']['gold_transitions']['summaries']
        values = {k: v['value'] for k, v in t.items()}
        self.assertEqual(values, {'wrong_to_correct': .75, 'correct_to_wrong': 0,
                                  'both_correct': .25, 'both_wrong': 0})
        self.assertEqual(values['wrong_to_correct'] - values['correct_to_wrong'],
                         metric_estimate(self.bundle['models'][0], 'paired', 'delta_BAcc_gold_raw')['value'])

    def test_model_contrasts_recompute_two_models_on_shared_draws(self):
        contrast = self.bundle['model_contrasts'][0]
        self.assertEqual((contrast['target'], contrast['reference']), ('Test model 1', 'Test model 0'))
        self.assertEqual(contrast['raw_all']['metrics']['BAcc'], .625)
        self.assertEqual(contrast['paired']['metrics']['BAcc_gold'], 0)
        self.assertEqual(contrast['paired']['metrics']['delta_BAcc_gold_raw'], -.75)
        self.assertEqual(contrast['raw_all']['intervals']['n_clusters'], 3)
        self.assertEqual(contrast['paired']['intervals']['n_clusters'], 2)
        interval = contrast['paired']['intervals']['intervals']['BAcc_gold']
        self.assertEqual((interval['lower'], interval['upper']), (0, 0))

    def test_source_cluster_bootstrap_keeps_repeated_mother_counts(self):
        probes = [decode_probe(p) for p in Study(self.study).private()['probes'].values()]
        plan = deepcopy(self.plan)
        for p in plan['probes']:
            if p['problem_id'] in {'A', 'B'}:
                p['cluster_id'] = 'shared-source'
        seen = []
        def stat(counts):
            seen.append(dict(counts))
            self.assertEqual(counts['A'], counts['B'])
            return {'x': counts['A'] / 2}
        result = boot(plan, probes, stat)
        self.assertEqual(result['n_clusters'], 2)
        self.assertTrue(any(c['A'] == 2 for c in seen))

    def test_fixed_bins_handle_edges_empty_bins_and_incomplete_outcomes(self):
        probes = [decode_probe(p) for p in Study(self.study).private()['probes'].values()]
        rows = [{'problem_id': m, 'cluster_id': m, 'binding_mean': b, 'answer_correct': a}
                for m, b, a in [('A', 0, 0), ('B', .2, 1), ('C', 1, 1)]]
        result = binned_solving(self.plan, probes, rows)
        self.assertEqual(result['edges'], list(BINDING_BINS))
        self.assertEqual([r['n_mothers'] for r in result['rows']], [1, 1, 0, 0, 1])
        self.assertEqual([r['value'] for r in result['rows']], [0, 1, None, None, 1])
        self.assertIsNone(result['rows'][2]['ci'])
        self.assertTrue(result['rows'][-1]['closed_right'])
        rows[1]['binding_mean'] = None
        self.assertEqual(binned_solving(self.plan, probes, rows)['rows'], [])

    def test_missing_service_is_not_wrong_answer_or_complete_case_deletion(self):
        run = self.root / 'missing-run'
        adapter = RecordingAdapter(lambda req: fixture_answer(req, missing=True))
        run_study(self.study, adapter, run)
        score_study(self.study, run)
        model = load_bundle(self.study, [run], allow_non_scientific=True)['models'][0]
        value = metric_estimate(model, 'raw_all', 'BAcc')
        self.assertIsNone(value['value'])
        self.assertEqual(value['support']['planned_probes'], 5)
        self.assertLess(value['bounds'][0], value['bounds'][1])
        self.assertEqual(model['diagnostics']['binned_solving']['rows'], [])
        self.assertEqual(model['report']['original_solving']['answer_eligible_mothers'], 3)

    def test_conditional_error_ci_reverses_endpoints_after_subtraction(self):
        model = deepcopy(self.bundle['models'][0])
        block = model['report']['experiments']['main']['intervals']['raw_all']['intervals']['BAcc_given_C']
        block.update(status='ok', lower=.2, upper=.8)
        interval = metric_estimate(model, 'raw_all', 'BindErr_given_C')['ci']
        self.assertAlmostEqual(interval[0], .2)
        self.assertAlmostEqual(interval[1], .8)
        block['status'] = 'withheld_undefined_statistic'
        self.assertIsNone(metric_estimate(model, 'raw_all', 'BindErr_given_C')['ci'])

    def test_read_only_audit_preserves_source_files_and_frozen_code(self):
        paths = [p for r in [self.study] + self.runs for p in r.rglob('*') if p.is_file()]
        before = {p: file_hash(p) for p in paths}
        load_bundle(self.study, self.runs, allow_non_scientific=True, diagnostic_ci=False)
        self.assertEqual(before, {p: file_hash(p) for p in paths})
        self.assertEqual(Study(self.study).plan['plan_hash'], self.plan['plan_hash'])

    def test_sealed_human_fixture_is_matched_and_unsealed_runs_stay_unmeasured(self):
        # Isolated test-double participants; no user study is sealed or modified.
        study_path = self.new_output()
        shutil.copytree(self.study, study_path)
        study = Study(study_path)
        sessions = study_path.parent / 'synthetic-human-sessions'
        for cohort in (0, 1):
            path = create_session(study_path, sessions, participant=f'synthetic-reader-{cohort}', cohort=cohort)
            for task in read_json(path / 'session.json')['tasks']:
                submit(study, path, task['task_id'], {'text': '{"read":"2 kg","owner":"E1"}',
                    'ambiguous': False, 'notes': 'Synthetic test fixture only', 'human_confirmed': True})
        path = create_session(study_path, sessions, participant='synthetic-auditor', mode='audit')
        for task in read_json(path / 'session.json')['tasks']:
            submit(study, path, task['task_id'], {'decision': 'accept', 'checks': dict.fromkeys(AUDIT_CHECKS, True),
                'notes': 'Synthetic test fixture only', 'human_confirmed': True})
        seal_study(study_path, sessions)
        run = study_path.parent / 'synthetic-sealed-run'
        run_study(study_path, RecordingAdapter(fixture_answer), run)
        score_study(study_path, run)
        model = load_bundle(study_path, [run], allow_non_scientific=True)['models'][0]
        comparison = model['report']['human_model_same_subset']['raw']['BAcc']
        self.assertEqual(comparison['human']['value'], 1)
        self.assertEqual(comparison['human_minus_model'], 1 - comparison['model']['value'])
        self.assertIsNone(self.bundle['models'][0]['report']['human_model_same_subset'])

    def test_stale_report_and_inconsistent_ci_are_rejected(self):
        path = self.runs[0] / 'study_report.json'
        saved = path.read_bytes()
        try:
            for mutation in ('metric', 'ci', 'run_id'):
                report = read_json(path)
                if mutation == 'metric':
                    report['experiments']['main']['raw_all']['metrics']['BAcc'] = .99
                elif mutation == 'ci':
                    report['experiments']['main']['intervals']['raw_all']['intervals']['CAcc']['estimate'] = .99
                else:
                    report['run_id'] = 'wrong-run'
                atomic_json(path, report)
                with self.assertRaisesRegex(ValueError, 'Stale|inconsistent'):
                    load_bundle(self.study, [self.runs[0]], allow_non_scientific=True)
                path.write_bytes(saved)
        finally:
            path.write_bytes(saved)

    def test_scientific_gate_duplicate_ids_and_mixed_budgets(self):
        with self.assertRaisesRegex(ValueError, 'non-scientific'):
            load_bundle(self.study, self.runs)
        with self.assertRaisesRegex(ValueError, 'Duplicate model'):
            load_bundle(self.study, [self.runs[0], self.runs[0]], allow_non_scientific=True)
        adapter = RecordingAdapter(fixture_answer)
        adapter.info = replace(adapter.info, model_id='Other budget', settings_json='{"max_new_tokens":55}')
        run = self.root / 'different-budget'
        run_study(self.study, adapter, run)
        score_study(self.study, run)
        with self.assertRaisesRegex(ValueError, 'budgets'):
            load_bundle(self.study, [self.runs[0], run], allow_non_scientific=True)
        compared = load_bundle(self.study, [self.runs[0], run], allow_non_scientific=True,
                               allow_model_specific_settings=True)
        self.assertIsNone(compared['shared_settings'])
        self.assertEqual(compared['model_protocols']['Other budget']['settings']['max_new_tokens'], 55)
        self.assertIn('Model-specific', compared['comparison_note'])
        out = self.new_output()
        export_bundle(compared, out, tables_only=True)
        self.assertEqual(read_json(out / 'model_protocols.json'), compared['model_protocols'])
        self.assertIn('Model-specific', (out / 'captions.md').read_text(encoding='utf-8'))

    def test_tables_only_exports_preserve_status_missingness_and_full_precision(self):
        out = self.new_output()
        manifest = export_bundle(self.bundle, out, tables_only=True)
        self.assertEqual(len(manifest['tables']), 16)
        self.assertEqual(manifest['figures'], [])
        self.assertEqual(read_json(out / 'figure_data.json'), self.bundle)
        rows = list(csv.reader((out / 'tables/01_main.csv').read_text(encoding='utf-8').splitlines()))
        self.assertEqual(rows[0][0], 'Data status')
        self.assertTrue(all(r[0] == 'non_scientific' for r in rows[1:]))
        for extension in ('md', 'tex'):
            self.assertIn('NON-SCIENTIFIC', (out / f'tables/01_main.{extension}').read_text(encoding='utf-8'))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            export_bundle(self.bundle, out, tables_only=True)

    def test_latex_caption_width_escape_and_repeated_header(self):
        out = self.new_output()
        export_bundle(self.bundle, out, tables_only=True)
        main_tex = (out / 'tables/01_main.tex').read_text(encoding='utf-8')
        self.assertLess(main_tex.index('\\caption'), main_tex.index('\\begin{tabular}'))
        self.assertIn('\\arraybackslash', main_tex)
        self.assertIn('\\linewidth-2\\tabcolsep', main_tex)
        self.assertNotIn('\\hline', main_tex)
        self.assertIn('\\newline', latex_cell('75.0 [25.0, 99.0]'))
        self.assertIn('\\allowbreak{}', latex_cell('75.0 [25.0, 99.0]'))
        self.assertEqual(latex_escape('A_1 & 25%'), r'A\_1 \& 25\%')
        self.assertIn('\\endfirsthead', (out / 'tables/05_controls.tex').read_text(encoding='utf-8'))

    def test_pdf_only_gallery_has_no_empty_or_pdf_image_source(self):
        page = gallery(self.bundle, [{'id': 'test', 'title': '<unsafe>', 'section': 'main',
            'files': {'pdf': 'test.pdf'}, 'caption': 'Caption'}], [])
        self.assertNotIn('<img', page)
        self.assertIn('href="figures/test.pdf"', page)
        self.assertIn('&lt;unsafe&gt;', page)

    def test_cli_protects_study_and_does_not_mix_demo_with_measurements(self):
        self.assertEqual(main(['--demo', '--study', str(self.study), '--output', str(self.new_output())]), 2)
        self.assertEqual(main(['--study', str(self.study), '--run', str(self.runs[0]),
                              '--output', str(self.study / 'figures')]), 2)


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional requirements-viz.txt not installed')
class RenderingTests(unittest.TestCase):
    def test_all_design_figures_and_core_table_previews_render_with_physical_width(self):
        from physalign_viz.demo import demo_bundle
        from physalign_viz.plots import _point
        import matplotlib.pyplot as plt
        from PIL import Image
        bundle = demo_bundle(n_resamples=0)
        self.assertTrue(all('Synthetic' in m['model_id'] for m in bundle['models']))
        self.assertEqual(len({m['report']['nearest_region']['metrics']['BAcc_control'] for m in bundle['models']}), 1)
        self.assertEqual(len({m['report']['uniform_random_full']['value'] for m in bundle['models']}), 1)
        self.assertEqual(len({m['report']['human_model_same_subset']['raw']['BAcc']['human']['value'] for m in bundle['models']}), 1)
        with tempfile.TemporaryDirectory(prefix='physalign-render-test-') as temp:
            out = Path(temp) / 'figures'
            with warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                manifest = export_bundle(bundle, out, dpi=90)
                fig, ax = plt.subplots()
                _point(ax, 0, .2, [.4, .8], '#0072B2')  # Valid percentile interval outside point.
                fig.canvas.draw()
                plt.close(fig)
            self.assertEqual(sum(f['status'] == 'generated' for f in manifest['figures']), 18)
            self.assertEqual(len(manifest['tables']), 16)
            for table in manifest['tables'][:4]:
                self.assertEqual(set(table['files']), {'csv', 'md', 'tex', 'pdf', 'svg', 'png'})
            for entry, folder in [(f, 'figures') for f in manifest['figures']] + [(t, 'tables') for t in manifest['tables'][:4]]:
                svg = (out / folder / entry['files']['svg']).read_text(encoding='utf-8')
                self.assertIn('width="396pt"', svg)
                self.assertIn('SYNTHETIC DESIGN PREVIEW', svg)
                pdf = (out / folder / entry['files']['pdf']).read_bytes()
                self.assertRegex(pdf, rb'/MediaBox\s*\[\s*0\s+0\s+396\s+')
                with Image.open(out / folder / entry['files']['png']) as im:
                    self.assertEqual(im.width, 495)
            self.assertEqual(file_hash(out / 'figure_data.json'), manifest['files']['figure_data.json'])


if __name__ == '__main__':
    unittest.main()
