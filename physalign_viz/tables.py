"""Numbers shared with plots, exported as tidy CSV, Markdown and booktabs LaTeX."""

import csv
import re
from io import StringIO
from pathlib import Path

from physalign.storage import atomic_text
from .data import ci_entry, metric_estimate


def number(value, interval=None, *, scale=100, signed=False):
    if value is None:
        return '--'
    text = f'{value*scale:+.1f}' if signed else f'{value*scale:.1f}'
    if interval is not None:
        text += f' [{interval[0]*scale:.1f}, {interval[1]*scale:.1f}]'
    return text


def latex_escape(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
               '↑': r'$\uparrow$', '↓': r'$\downarrow$'}
    return ''.join(mapping.get(c, c) for c in str(value))


def latex_cell(value):
    match = re.fullmatch(r'([+-]?[0-9]+\.[0-9]+) \[([^]]+)\]', str(value))
    if match:
        # Paragraph cells can wrap the interval at its comma in narrow appendix
        # columns. An unbreakable shortstack can overflow even a bounded p column.
        interval = match[2].replace(', ', ',' + r'\allowbreak{}')
        return match[1] + r'\newline{\scriptsize [' + interval + ']}'
    return latex_escape(value).replace(r'\_', r'\_\allowbreak{}')


def table_specs(bundle):
    tables = []
    def add(key, title, columns, rows, caption, *, main=False):
        if bundle.get('comparison_note'):
            caption += ' ' + bundle['comparison_note']
        tables.append({'id': key, 'title': title, 'columns': columns, 'rows': rows, 'caption': caption,
                       'section': 'main' if main else 'appendix'})
    def metric(m, scope, key):
        e = metric_estimate(m, scope, key)
        return number(e['value'], e['ci'])
    add('01_main', 'Primary evaluation', ['Model', 'CAcc L ↑', 'BAcc B ↑', 'JAcc L ↑', 'BindErr | C ↓', 'SolveAcc ↑'],
        [[m['label'], metric(m, 'raw_all', 'CAcc'), metric(m, 'raw_all', 'BAcc'), metric(m, 'raw_all', 'JAcc'),
          metric(m, 'raw_all', 'BindErr_given_C'), metric(m, 'solve', 'SolveAcc')] for m in bundle['models']],
        'Scores are percentages; bracketed intervals are clustered bootstrap CIs when defined. B/L and answer-eligible mother populations differ. -- means unavailable, never zero.', main=True)
    add('02_gold', 'Matched Raw / Gold binding', ['Model', 'Raw BAcc ↑', 'Gold BAcc ↑', 'Gold - Raw (pp)', 'Paired probes', 'Paired mothers'],
        [[m['label'], metric(m, 'paired', 'BAcc'), metric(m, 'paired', 'BAcc_gold'), metric(m, 'paired', 'delta_BAcc_gold_raw'),
          (m['report']['experiments']['main']['paired'] or {}).get('support', {}).get('binding_raw', {}).get('planned_probes', '--'),
          (m['report']['experiments']['main']['paired'] or {}).get('support', {}).get('binding_raw', {}).get('planned_problems', '--')]
         for m in bundle['models']], 'Both conditions use exactly the same frozen paired B pool. Differences are percentage points, with paired CIs. Gold has no CAcc or JAcc.', main=True)
    add('03_quadrants', 'Reading / binding outcome decomposition', ['Model', 'C1 B1 (%)', 'C1 B0 (%)', 'C0 B1 (%)', 'C0 B0 (%)'],
        [[m['label']] + [number(m['report']['experiments']['main']['raw_all']['quadrants'][k]) for k in ('q11', 'q10', 'q01', 'q00')]
         for m in bundle['models']], 'All four quadrants use L and the same weights. C1 B0 = CAcc - JAcc; it is not the conditional binding-error rate.', main=True)
    rows = []
    for m in bundle['models']:
        result = m['report']['original_solving']
        a = result['association'] or {}
        rows.append([m['label'], result['answer_eligible_mothers'], a.get('group_counts', {}).get('correct', '--'),
                     a.get('group_counts', {}).get('wrong', '--')] +
                    [number(a.get(k), ci_entry(result['intervals'], k)) for k in ('mean_binding_correct', 'mean_binding_wrong', 'delta_assoc')])
    add('04_solving', 'Binding association with independent original solutions',
        ['Model', 'Eligible mothers', 'Solved', 'Wrong', 'Mean b | solved (%)', 'Mean b | wrong (%)', 'Difference (pp)'], rows,
        'b is the ordinary mean of all B probes within a mother. Conditional means weight mothers equally. The contrast is descriptive, not causal; an empty group is undefined.', main=True)
    rows = []
    for m in bundle['models']:
        for variant, conditions in m['report']['control_comparisons'].items():
            for condition, c in conditions.items():
                v = c['metrics']
                rows.append([m['label'], variant, condition, 'same renderer' if variant.startswith('perm_') else 'matched main',
                             number(v['BAcc_reference']), number(v['BAcc_control']), number(v['delta_BAcc'], ci_entry(c['intervals'], 'delta_BAcc')),
                             number(v['canonical_consistency'])])
    add('05_controls', 'Fixed interface controls', ['Model', 'Control', 'Condition', 'Reference', 'BAcc ref (%)', 'BAcc control (%)', 'Delta (pp)', 'Consistency (%)'],
        rows, 'Each row uses its predeclared matched subset. Permutations are compared against the identity reference rendered by the same renderer; repeated permutations are not independent mother samples.')
    rows = []
    for m in bundle['models']:
        for kind, r in m['report']['experiments']['main']['raw_all']['by_type'].items():
            rows.append([m['label'], kind, r['support']['binding_raw']['planned_probes'], r['support']['binding_raw']['planned_problems'],
                         r['support']['joint_raw']['planned_probes']] + [number(r.get(k)) for k in ('CAcc', 'BAcc', 'JAcc', 'BindErr_given_C')])
    add('06_by_type', 'Per-type scores and support', ['Model', 'Type', 'B probes', 'B mothers', 'L probes', 'CAcc (%)', 'BAcc (%)', 'JAcc (%)', 'BindErr | C (%)'],
        rows, 'Each type uses mother aggregation. Undefined joint metrics for binding-only types stay unavailable; no fabricated type confidence intervals are supplied.')
    rows = []
    for m in bundle['models']:
        for name, levels in m['diagnostics']['strata'].items():
            for level, value in levels.items():
                rows.append([m['label'], name, level, value['n_probes'], value['n_mothers'],
                             number(value['summary']['value'], value['ci']), str(value['type_weights'])])
    add('07_strata', 'Exploratory fixed-metadata strata', ['Model', 'Stratum', 'Level', 'Probes', 'Mothers', 'BAcc (%)', 'Type weights'], rows,
        'Membership is fixed by study metadata. Aggregation and weights are recomputed within each fixed stratum. Cross-stratum differences may reflect task composition; they are not controlled interventions.')
    rows = []
    for m in bundle['models']:
        for r in m['diagnostics']['binned_solving']['rows']:
            label = f"[{r['low']:.1f},{r['high']:.1f}{']' if r['closed_right'] else ')'}"
            rows.append([m['label'], label, r['n_mothers'], r['n_clusters'], number(r['value'], r['ci'])])
    add('08_binding_bins', 'Exploratory solving by mother binding score', ['Model', 'Binding bin', 'Mothers', 'Source clusters', 'SolveAcc (%)'], rows,
        'Bins are fixed in visualization code. The same mother can fall in different bins for different models. Empty bins and undefined CIs are retained; a model with incomplete eligible outcomes has no curve.')
    rows = []
    for m in bundle['models']:
        entry = m['diagnostics']['gold_transitions']
        if entry:
            rows.append([m['label']] + [number(entry['summaries'][k]['value'], ci_entry(entry['intervals'], k))
                                       for k in ('wrong_to_correct', 'correct_to_wrong', 'both_correct', 'both_wrong')])
    add('09_gold_transitions', 'Probe-level binding transitions under Gold reading',
        ['Model', 'Wrong to correct (%)', 'Correct to wrong (%)', 'Both correct (%)', 'Both wrong (%)'], rows,
        'Form each paired event at probe level before mother/type aggregation. Wrong-to-correct minus correct-to-wrong equals the matched Gold-minus-Raw difference. This is not the option-intervention experiment.')
    rows = []
    for m in bundle['models']:
        for condition in ('raw', 'gold'):
            e = (m['report'].get('human_model_same_subset') or {}).get(condition, {}).get('BAcc')
            rows.append([m['label'], condition, number(e['model']['value']) if e else '--',
                         number(e['human']['value']) if e and e['human'] else '--', number(e['human_minus_model']) if e else '--'])
    add('10_human', 'Model versus same-interface reference', ['Model', 'Condition', 'Model BAcc (%)', 'Reference BAcc (%)', 'Reference - model (pp)'], rows,
        'All comparisons use the same human-evaluation subset and packet condition. Missing human measurements stay unavailable. In design-preview output only, reference values are synthetic illustrations, not participant evidence.')
    rows, fields = [], []
    for m in bundle['models']:
        r = m['report']
        s, invalid = r['service'], r['experiments']['main']['raw_all']['invalidity']['raw']
        rows.append([m['label'], s['planned'], s['completed'], s['infrastructure_missing'], s['pending'],
                     invalid['served_outputs'], invalid['invalid_objects'], number(invalid['object_invalid_rate_served'])])
        for field, v in invalid['fields'].items():
            fields.append([m['label'], field, v['required_served_fields'], v['invalid_fields'], number(v['invalid_rate_served'])])
    add('11_service', 'Response coverage and JSON-object validity', ['Model', 'Planned requests', 'Completed', 'Infrastructure missing', 'Pending', 'Raw served', 'Raw invalid JSON', 'Invalid / served (%)'], rows,
        'Unweighted operational counts. Completed invalid or truncated answers are outcomes; unserved infrastructure failures remain missing in the fixed benchmark denominator.')
    add('12_fields', 'Required output-field validity', ['Model', 'Field', 'Required served', 'Invalid fields', 'Invalid / required (%)'], fields,
        'A valid JSON object can still omit or invalidate required reading/binding fields. Field-specific denominators count only served outputs where the field is required.')
    rows = []
    for m in bundle['models']:
        r = m['report']
        n = r.get('nearest_region')
        rows.append([m['label'], number(r['uniform_random_full']['value']), number(n['metrics']['BAcc_reference']) if n else '--',
                     number(n['metrics']['BAcc_control']) if n else '--', number(n['weighted_full_pool_coverage']) if n else '--'])
    add('13_baselines', 'Shortcut baselines and eligible coverage', ['Model', 'Full-B chance (%)', 'Subset model (%)', 'Subset nearest (%)', 'Weighted full-B coverage (%)'], rows,
        'Uniform chance aggregates per-item correct-alias probabilities; it is undefined for an undeclared set/relation distribution. Nearest-region uses its exact eligible ownership subset, with weighted full-pool coverage disclosed.')
    rows = [[r['type'], r['probes_B'], r['probes_L'], r['mothers'], r['gold_probes']] for r in bundle['dataset']['by_type']]
    add('14_dataset', 'Frozen evaluated dataset composition', ['Type', 'B probes', 'L probes', 'Mothers', 'Paired Gold probes'], rows,
        'These are counts in the selected frozen study, not extrapolated benchmark totals. A mother can contribute to multiple types; do not sum type mother counts to obtain total mothers.')
    rows = []
    for m in bundle['models']:
        p = m['report']['experiments']['main']['paired']
        if p:
            r = p['nonempty_packets']
            rows.append([m['label'], r['support']['planned_probes'], r['support']['planned_problems'], number(r['weighted_packet_coverage']),
                         number(r['BAcc']['value']), number(r['BAcc_gold']['value'] if r['BAcc_gold'] else None), number(r['delta_BAcc_gold_raw'])])
    add('15_nonempty_packets', 'Nonempty local-reading packets', ['Model', 'Probes', 'Mothers', 'Paired-pool coverage (%)', 'Raw BAcc (%)', 'Gold BAcc (%)', 'Delta (pp)'], rows,
        'The predeclared nonempty-packet subset has its own frozen mother/type weights. Empty approved Gold packets remain in the paired main comparison; no inference is made for missing exported Gold views.')
    rows = []
    labels = {m['model_id']: m['label'] for m in bundle['models']}
    for r in bundle.get('model_contrasts', []):
        for scope in ('raw_all', 'paired'):
            if r[scope]:
                for metric, value in r[scope]['metrics'].items():
                    rows.append([labels[r['target']], labels[r['reference']], scope, metric,
                                 number(value, ci_entry(r[scope]['intervals'], metric))])
    add('16_model_contrasts', 'Shared-cluster model contrasts', ['Target', 'Reference', 'Population', 'Metric', 'Target - reference (pp)'], rows,
        'Both models share the same mother/source-cluster draws. Full Raw and paired Gold use their own sampling frames. CIs are pointwise, not adjusted for multiple comparisons; no significance stars are assigned.')
    return tables


def write_tables(bundle, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    entries = table_specs(bundle)
    notice = '' if bundle['status'] == 'scientific' else ('SYNTHETIC DESIGN PREVIEW; NOT EXPERIMENTAL RESULTS. '
              if bundle['status'] == 'synthetic_preview' else 'NON-SCIENTIFIC PIPELINE CHECK. ')
    for table in entries:
        key, columns, rows = table['id'], table['columns'], table['rows']
        stream = StringIO(newline='')
        writer = csv.writer(stream)
        # Machine-friendly unit metadata stays in table schema / figure_data.json;
        # CSV values match the displayed table, including unavailable markers.
        writer.writerow(['Data status'] + columns)
        writer.writerows([[bundle['status']] + row for row in rows])
        atomic_text(root / (key + '.csv'), stream.getvalue())
        clean = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')
        markdown = f"{notice}{table['title']}\n\n{table['caption']}\n\n"
        markdown += '| ' + ' | '.join(map(clean, columns)) + ' |\n'
        markdown += '| ' + ' | '.join('---' for _ in columns) + ' |\n'
        markdown += ''.join('| ' + ' | '.join(map(clean, row)) + ' |\n' for row in rows)
        atomic_text(root / (key + '.md'), markdown)
        # longtable handles large control/stratum tables without silently cutting rows.
        environment = 'longtable' if len(rows) > 12 else 'tabular'
        numeric = [all(isinstance(row[i], (int, float)) or str(row[i]) == '--' or
                       re.fullmatch(r'[+-]?[0-9]+(?:\.[0-9]+)?(?: \[[^]]+\])?', str(row[i])) for row in rows)
                   for i in range(len(columns))]
        weights = [1.5 if i == 0 else 1 if numeric[i] else 1.6 for i in range(len(columns))]
        total = sum(weights)
        # Bounded-width paragraph columns keep long identifiers inside the configured
        # line width. Main numeric cells put the CI beneath the point estimate.
        alignment = '@{}' + ''.join('>{' + (r'\raggedleft' if is_numeric else r'\raggedright') +
                    r'\arraybackslash}p{\dimexpr ' + f'{weight/total:.6f}' + r'\linewidth-2\tabcolsep\relax}'
                    for weight, is_numeric in zip(weights, numeric)) + '@{}'
        head = '\\toprule\n' + ' & '.join(latex_cell(c) for c in columns) + ' \\\\\n\\midrule\n'
        if environment == 'longtable':
            latex = '{\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{' + alignment + '}\n'
            latex += '\\caption{' + latex_escape(notice + table['title'] + '. ' + table['caption']) + '}\\label{tab:physalign-' + key + '}\\\\\n'
            latex += head + '\\endfirsthead\n' + head + '\\endhead\n\\bottomrule\n\\endlastfoot\n'
        else:
            latex = '\\begin{table}[t]\n\\centering\n\\caption{' + latex_escape(notice + table['title'] + '. ' + table['caption']) + '}\n'
            latex += '\\label{tab:physalign-' + key + '}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n'
            latex += '\\begin{tabular}{' + alignment + '}\n'
            latex += head
        latex += ''.join(' & '.join(latex_cell(c) for c in row) + ' \\\\\n' for row in rows)
        if not rows:
            latex += '\\multicolumn{' + str(len(columns)) + '}{l}{Unavailable: required observations are absent.} \\\\\n'
        latex += ('\\bottomrule\n' if environment == 'tabular' else '') + '\\end{' + environment + '}\n'
        if environment == 'tabular':
            latex += '\\end{table}\n'
        else:
            latex += '}\n'
        atomic_text(root / (key + '.tex'), latex)
        table['files'] = {fmt: key + '.' + fmt for fmt in ('csv', 'md', 'tex')}
    return entries
