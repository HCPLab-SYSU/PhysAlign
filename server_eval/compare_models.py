"""Audit and plot local/API Raw-Gold runs on one frozen benchmark plan.

Model-specific reasoning and image processing are disclosed, never equated.
This companion leaves the existing evaluator and server launchers unchanged.
"""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.dataset import load_truth, require
from physalign.metrics import Pool
from physalign.planning import code_hashes, load_plan, verify_public_plan
from physalign.reporting import score_run
from physalign.runner import validate_manifest
from physalign.storage import atomic_json, atomic_text, file_hash, loads, outside_bundle, read_json

LABELS = {'Qwen/Qwen3.5-4B': 'Qwen3.5-4B', 'Qwen/Qwen3.5-9B': 'Qwen3.5-9B', 'Qwen/Qwen3.5-27B': 'Qwen3.5-27B',
          'OpenGVLab/InternVL3_5-8B-HF': 'InternVL3.5-8B', 'gpt-6-astra': 'GPT-6 Astra (high)',
          'gemini-3.8-flash': 'Gemini-3.8-flash', 'google/gemma-4-26B-A4B-it': 'Gemma 4 26B-A4B',
          'zai-org/GLM-4.6V-Flash': 'GLM-4.6V-Flash', 'allenai/Molmo2-8B': 'Molmo2-8B',
          'moonshotai/Kimi-VL-A3B-Instruct': 'Kimi-VL-A3B-Instruct'}


def audit_models(plan, run_directories, *, dataset=None, labels=None, allow_non_scientific=False):
    ds = verify_public_plan(plan, dataset)
    truth, _ = load_truth(ds)
    models, seen = [], set()
    for source in run_directories:
        directory = Path(source).resolve()
        manifest = read_json(directory / 'manifest.json')
        validate_manifest(manifest)
        require(manifest['plan'] == plan, 'Model runs must share the exact plan, populations, weights and bootstrap settings')
        require(manifest['implementation_hashes'] == code_hashes(all_modules=True), 'Model run uses a different core evaluator implementation')
        model_id = manifest['adapter']['model_id']
        require(model_id not in seen, 'Duplicate model ID: select one complete run per model, including the chosen 27B recovery trial')
        require(manifest['adapter']['scientific_run'] or allow_non_scientific, 'Synthetic runs require explicit preview mode')
        seen.add(model_id)
        # Rebuild metrics from authoritative response journals, including image/input audit.
        report = score_run(directory, dataset_root=dataset, allow_incomplete=True)
        service = report['service']
        require(not service['pending_requests'] and not service['infrastructure_missing']
                and service['completed_outputs'] == service['planned_requests'],
                f'Incomplete run for {model_id}; report saved, comparison figures withheld')
        settings = report['adapter']['settings']
        default_label = LABELS.get(model_id, model_id)
        if model_id == 'gpt-6-astra':
            default_label = f'GPT-6 Astra ({settings.get("reasoning_effort", "unspecified")})'
        if model_id.startswith('Qwen/Qwen3.5-') and type(settings.get('thinking')) is bool:
            default_label += ' (thinking)' if settings['thinking'] else ' (non-thinking)'
        # Explicit display labels override the whole label, including its suffix.
        # The report and protocol rows continue to use actual adapter settings.
        label = (labels or {}).get(model_id, default_label)
        require(isinstance(label, str) and bool(label.strip()), 'Model labels must be nonempty strings')
        scores_path = directory / 'scores.jsonl'
        rows = [loads(line) for line in scores_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        raw = {r['instance_id']: r for r in rows if r['condition'] == 'raw'}
        require(set(raw) == {r['instance_id'] for r in plan['probes']}, 'Per-probe scores do not match plan')
        interfaces = {}
        for interface in sorted({r['interface'] for r in plan['probes']}):
            probes = tuple(truth[r['instance_id']] for r in plan['probes'] if r['interface'] == interface)
            value = Pool(probes).estimate({p.probe_id: raw[p.probe_id]['B'] for p in probes})['value']
            interfaces[interface] = {'BAcc': value, 'probes': len(probes), 'mothers': len({p.problem_id for p in probes})}
        models.append({'model_id': model_id, 'label': label, 'report': report, 'interface_binding': interfaces,
                       'run_directory': str(directory), 'run_id': manifest['run_id'],
                       'manifest_sha256': file_hash(directory / 'manifest.json'), 'report_sha256': file_hash(directory / 'report.json'),
                       'scores_sha256': file_hash(scores_path), 'adapter_provenance': manifest['adapter_provenance']})
    require(models, 'Provide at least one model run')
    return models, ds.manifest


def interval(model, scope, key):
    block = model['report']['intervals'].get(scope) or {}
    value = block.get('intervals', {}).get(key, {})
    return ([value['lower'], value['upper']] if value.get('status') == 'ok' and block.get('n_clusters', 0) >= 2 else None)


def protocol_rows(models):
    rows = []
    for model in models:
        adapter = model['report']['adapter']
        settings, prep = adapter['settings'], adapter['preprocessing']
        api = adapter['name'] == 'openai-compatible-api'
        rows.append({'model': model['label'], 'requested_model_id': model['model_id'], 'adapter': adapter['name'],
                     'reasoning': settings.get('reasoning_effort') or ('provider_default' if api else
                         'thinking' if settings.get('thinking') is True else
                         'non-thinking' if settings.get('thinking') is False else 'unspecified'),
                     'output_policy': settings.get('output_policy'),
                     'output_budget_scope': settings.get('output_budget_scope'),
                     'output_limit': settings.get('max_output_tokens', settings.get('max_new_tokens')),
                     'output_limit_field': settings.get('token_limit_field', 'max_new_tokens'),
                     'input_limit': 'provider_managed' if api else settings.get('max_input_tokens'),
                     'temperature': settings.get('temperature') if api else ('greedy' if settings.get('do_sample') is False else 'unspecified'),
                     'image_processing': prep.get('provider_image_processing', prep.get('processor_call_kwargs', {})),
                     'endpoint': settings.get('base_url'),
                     'returned_versions': model['report']['provenance']['returned_model_versions'],
                     'multiple_returned_versions': model['report']['provenance']['multiple_returned_versions'],
                     'run_id': model['run_id']})
    return rows


def render_comparison(plan, models, dataset_manifest, output, *, dpi=600, formats=('pdf', 'svg', 'png')):
    # Shared plotting module selects Agg and puts its cache in the temp directory.
    from physalign_viz.plots import FigureBook, _point, theme
    import matplotlib.pyplot as plt
    import numpy as np
    from physalign_viz.tables import latex_escape
    from physalign.storage import canonical

    output = Path(output).resolve()
    require(not output.exists(), 'Figure output already exists; choose a new directory')
    outside_bundle(output, Path(plan['dataset_hint']))
    for model in models:
        outside_bundle(output, Path(model['run_directory']))
    preview = any(not m['report']['scientific_run'] for m in models)
    stamp = ('SYNTHETIC FIXTURES - NOT MODEL RESULTS.' if preview else
             f'Dataset split: {plan["split"]}. ' + ('Unseen-test status uncertified.' if plan['split'] != 'test' else 'Frozen test plan.'))
    protocol_note = 'Model-specific reasoning, output budgets and image processing differ; see protocol.csv.'
    output.mkdir(parents=True)
    bundle = {'status': 'non_scientific' if preview else 'scientific', 'models': [
        {'label': m['label'], 'report': {'experiments': {'main': {
            'raw_all': m['report']['raw_all'], 'paired': m['report']['paired'], 'intervals': m['report']['intervals']}}}}
        for m in models]}

    class ComparisonBook(FigureBook):
        def finish(self, fig, name, title, caption, **kwargs):
            kwargs['footer'] = kwargs.get('footer', '') + ' ' + stamp
            kwargs['top'] = min(kwargs.get('top', .82), .79)
            kwargs['bottom'] = max(kwargs.get('bottom', .16), .21)
            super().finish(fig, name, title, caption + ' ' + stamp + ' ' + protocol_note, **kwargs)

    with plt.rc_context(theme()):
        book = ComparisonBook(bundle, output / 'figures', width=8.4, dpi=dpi, formats=formats)
        offsets = np.linspace(-.25, .25, len(models)) if len(models) > 1 else [0]
        fig, axes = book.figure(4.5)
        ax = axes[0, 0]
        metrics = [('CAcc', 'CAcc\n(reading L)'), ('BAcc', 'BAcc\n(all binding B)'),
                   ('JAcc', 'JAcc\n(joint L)'), ('BAcc_given_C', 'BAcc | C\n(joint L)')]
        for i, model in enumerate(models):
            for x, (key, _) in enumerate(metrics):
                value = model['report']['raw_all']['metrics'][key]
                _point(ax, x + offsets[i], value, interval(model, 'raw_all', key), book.colors[i], book.markers[i])
                if value is None:
                    ax.text(x + offsets[i], 5 + 7 * i, 'N/A', color=book.colors[i], ha='center', fontsize=7)
        ax.set_xticks(range(len(metrics)), [label for _, label in metrics])
        book.percent(ax)
        book.legend(ax)
        book.finish(fig, '01_main_results', 'Reading and role binding: local and API models',
                    'Original mother-within-task aggregation and frozen type weights. CAcc/JAcc/conditional binding use L; BAcc uses B. '
                    'Intervals are paired source-cluster bootstrap intervals when defined.',
                    footer=f'B = {len(plan["probes"])}; L = {sum(r["joint_eligible"] for r in plan["probes"])}. ' + protocol_note)
        book.quadrants()
        book.gold_control()
        fig, axes = book.figure(4.5)
        ax = axes[0, 0]
        interfaces = list(models[0]['interface_binding'])
        for i, model in enumerate(models):
            for x, interface in enumerate(interfaces):
                _point(ax, x + offsets[i], model['interface_binding'][interface]['BAcc'], None, book.colors[i], book.markers[i])
        ax.set_xticks(range(len(interfaces)), [f'{key}\n(n={models[0]["interface_binding"][key]["probes"]})' for key in interfaces])
        book.percent(ax)
        book.legend(ax)
        book.finish(fig, '04_interface_binding', 'Binding accuracy by interface',
                    'Descriptive mother-averaged estimates within each interface, without subgroup confidence intervals. '
                    'These do not replace the task-weighted main BAcc.', footer=protocol_note)

    tables = output / 'tables'
    tables.mkdir()
    def table(name, headers, rows):
        with (tables / (name + '.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        tex = ['% ' + stamp, r'\begingroup\scriptsize', r'\setlength{\tabcolsep}{3pt}',
               r'\begin{tabular}{l' + 'r' * (len(headers)-1) + '}', r'\toprule',
               ' & '.join(latex_escape(str(x)) for x in headers) + r' \\', r'\midrule']
        tex += [' & '.join(latex_escape(str(x)) for x in row) + r' \\' for row in rows]
        tex += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
        atomic_text(tables / (name + '.tex'), '\n'.join(tex) + '\n')
    def number(value):
        return '--' if value is None else f'{100 * value:.2f}'
    def cell(model, scope, key):
        report = model['report'][scope]
        if report is None:
            return '--'
        value = report['metrics'][key]
        ci = interval(model, scope, key)
        return number(value) + (f' [{number(ci[0])}, {number(ci[1])}]' if ci else '')
    keys = ('CAcc', 'BAcc', 'JAcc', 'BAcc_given_C')
    table('01_main', ['Model', *keys], [[m['label']] + [cell(m, 'raw_all', k) for k in keys] for m in models])
    table('02_gold', ['Model', 'Paired Raw', 'Gold', 'Gold-Raw (pp)'],
          [[m['label']] + [cell(m, 'paired', k) for k in ('BAcc', 'BAcc_gold', 'delta_BAcc_gold_raw')] for m in models])
    table('03_quadrants', ['Model', 'C1 B1', 'C1 B0', 'C0 B1', 'C0 B0'],
          [[m['label']] + [number(m['report']['raw_all']['quadrants'][k]) for k in ('q11', 'q10', 'q01', 'q00')] for m in models])
    table('04_interfaces', ['Model', 'Interface', 'BAcc', 'Probes', 'Mothers'],
          [[m['label'], key, number(row['BAcc']), row['probes'], row['mothers']] for m in models for key, row in m['interface_binding'].items()])
    table('05_binding_diagnostics', ['Model', 'BAcc_L', 'BindErr given C'],
          [[m['label'], cell(m, 'raw_all', 'BAcc_L'), cell(m, 'raw_all', 'BindErr_given_C')] for m in models])
    protocols = protocol_rows(models)
    with (output / 'protocol.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(protocols[0]))
        writer.writeheader()
        writer.writerows({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in protocols)
    public_data = {'schema_version': 'physalign_model_comparison_v1', 'status': stamp, 'plan_hash': plan['plan_hash'],
                   'bootstrap': plan['bootstrap'], 'dataset': dataset_manifest, 'protocol_note': protocol_note,
                   'protocols': protocols, 'models': models}
    atomic_json(output / 'figure_data.json', public_data)
    captions = '# Captions\n\n' + stamp + '\n\n' + protocol_note + '\n\n'
    captions += 'Tables display percentages; differences are percentage points. figure_data.json retains full precision. '
    captions += 'Undefined estimates/intervals are unavailable, not zeros. Gold is evaluated only on exported matched views; no Gold CAcc/JAcc.\n\n'
    captions += '\n\n'.join('## ' + e['id'] + '\n\n' + e.get('caption', e.get('reason', '')) for e in book.entries)
    atomic_text(output / 'captions.md', captions + '\n')
    cards = []
    for entry in book.entries:
        if entry['status'] != 'generated':
            cards.append('<section><h2>' + html.escape(entry['title']) + '</h2><p>' + html.escape(entry['reason']) + '</p></section>')
            continue
        links = ' '.join(f'<a href="figures/{html.escape(path)}">{fmt.upper()}</a>' for fmt, path in entry['files'].items())
        fmt = next((f for f in ('svg', 'png') if f in entry['files']), None)
        picture = f'<img src="figures/{html.escape(entry["files"][fmt])}">' if fmt else ''
        cards.append('<section><h2>' + html.escape(entry['title']) + '</h2>' + picture + '<p>' + links + '</p><p>' + html.escape(entry['caption']) + '</p></section>')
    atomic_text(output / 'index.html', '<!doctype html><meta charset="utf-8"><title>PhysAlign model comparison</title>'
                '<style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#203047}section{margin:40px 0}img{width:100%}a{margin-right:18px}</style>'
                '<h1>PhysAlign model comparison</h1><p>' + html.escape(stamp) + '</p><p>' + html.escape(protocol_note) + '</p>'
                '<p><a href="protocol.csv">Evaluation settings</a><a href="figure_data.json">Full data</a><a href="captions.md">Captions</a></p>' + ''.join(cards))
    atomic_json(output / 'figure_manifest.json', {'plan_hash': plan['plan_hash'], 'helper_sha256': file_hash(Path(__file__)),
                'status': stamp, 'width_inches': 8.4, 'png_dpi': dpi, 'figures': book.entries,
                'files': {p.relative_to(output).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--run', action='append', default=[])
    parser.add_argument('--dataset')
    parser.add_argument('--output')
    parser.add_argument('--panel', help='JSON with plan, runs, output and optional dataset/labels; paths relative to current directory')
    parser.add_argument('--allow-non-scientific', action='store_true')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--formats', nargs='+', choices=('pdf', 'svg', 'png'), default=['pdf', 'svg', 'png'])
    args = parser.parse_args(argv)
    try:
        spec = read_json(Path(args.panel)) if args.panel else {}
        require(set(spec) <= {'plan', 'runs', 'output', 'dataset', 'labels'}, 'Unknown panel configuration fields')
        plan = load_plan(args.plan or spec.get('plan'))
        runs = args.run or spec.get('runs', [])
        output, dataset = args.output or spec.get('output'), args.dataset or spec.get('dataset')
        require(output and runs and isinstance(runs, list), 'Provide output and model run directories')
        require(type(args.dpi) is int and 72 <= args.dpi <= 1200, 'DPI must be 72..1200')
        outside_bundle(Path(output), Path(dataset or plan['dataset_hint']))
        require(not Path(output).exists(), 'Figure output exists; choose a new directory')
        models, manifest = audit_models(plan, runs, dataset=dataset, labels=spec.get('labels'), allow_non_scientific=args.allow_non_scientific)
        result = render_comparison(plan, models, manifest, output, dpi=args.dpi, formats=tuple(dict.fromkeys(args.formats)))
        print(f'Comparison gallery: {result / "index.html"}')
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, TypeError, KeyError) as error:
        print(f'Model comparison: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
