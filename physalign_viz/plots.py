"""Vector-first figures with configurable physical width."""

from pathlib import Path
from textwrap import fill
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'physalign-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from .data import ci_entry, metric_estimate

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6B5CA5', '#8C6D31']
MARKERS = ['o', 's', '^', 'D', 'P', 'v']
INK, MUTED, GRID = '#203047', '#65758B', '#E5EAF0'
QUADRANTS = [('q11', 'Read correct / bind correct', '#238A83', ''),
             ('q10', 'Read correct / bind wrong', '#E59A45', '///'),
             ('q01', 'Read wrong / bind correct', '#7EAAD0', ''),
             ('q00', 'Read wrong / bind wrong', '#D7DEE7', '')]


def theme():
    return {'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
            'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
            'axes.edgecolor': '#BDC7D3', 'axes.labelcolor': INK, 'text.color': INK,
            'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .65,
            'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
            'grid.color': GRID, 'grid.linewidth': .6, 'legend.frameon': False,
            'legend.fontsize': 7.8, 'lines.linewidth': 1.6, 'pdf.fonttype': 42,
            'ps.fonttype': 42, 'svg.fonttype': 'path', 'svg.hashsalt': 'physalign-publication-v1',
            'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'mathtext.fontset': 'dejavusans'}


def _point(ax, x, y, interval, color, marker='o', *, horizontal=False, hollow=False):
    """Draw endpoints directly: asymmetric percentile CIs can exclude the estimate."""
    if y is None:
        return
    if interval is not None:
        lo, hi = [100*v for v in interval]
        if horizontal:
            ax.hlines(x, lo, hi, color=color, lw=1.25, alpha=.72)
            ax.plot([lo, hi], [x, x], '|', color=color, markersize=5)
        else:
            ax.vlines(x, lo, hi, color=color, lw=1.25, alpha=.72)
            ax.plot([x, x], [lo, hi], '_', color=color, markersize=5)
    ax.plot(100*y if horizontal else x, x if horizontal else 100*y, marker=marker,
            ms=5.5, color=color, mfc='white' if hollow else color, mec=color, mew=1.1, zorder=4)


class FigureBook:
    def __init__(self, bundle, output, *, width=5.5, dpi=450, formats=('pdf', 'svg', 'png')):
        self.bundle, self.models = bundle, bundle['models']
        self.root, self.width, self.dpi, self.formats = Path(output), width, dpi, tuple(formats)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = []
        self.labels = [m['label'] for m in self.models]
        self.colors = [COLORS[i % len(COLORS)] for i in range(len(self.models))]
        self.markers = [MARKERS[i % len(MARKERS)] for i in range(len(self.models))]

    def figure(self, height=3.2, *, ncols=1, nrows=1, **kwargs):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(self.width, height), squeeze=False, **kwargs)
        return fig, axes

    def legend(self, ax, *, loc='upper center', ncol=None):
        handles = [Line2D([], [], color=c, marker=s, linestyle='', label=l, markersize=5)
                   for c, s, l in zip(self.colors, self.markers, self.labels)]
        legend = ax.figure.legend(handles=handles, loc='upper center', ncol=ncol or min(3, len(handles)),
                                  bbox_to_anchor=(.52, .925), columnspacing=1.0, handletextpad=.4)
        legend.set_in_layout(False)

    def finish(self, fig, name, title, caption, *, section='appendix', footer='', top=.82, bottom=.16):
        if self.bundle.get('comparison_note'):
            caption += ' ' + self.bundle['comparison_note']
            footer += ' Model-specific protocols; see model_protocols.json.'
        title_artist = fig.suptitle(title, x=.075, y=.985, ha='left', fontsize=10.5, fontweight='semibold')
        title_artist.set_in_layout(False)
        stamp = {'synthetic_preview': 'SYNTHETIC DESIGN PREVIEW - NOT EXPERIMENTAL RESULTS',
                 'non_scientific': 'NON-SCIENTIFIC RUN - PIPELINE CHECK ONLY'}.get(self.bundle['status'])
        if stamp:
            fig.text(.075, .012, stamp, fontsize=6.5, color='#A84D30', weight='bold')
        if footer:
            fig.text(.075, .052 if stamp else .016, fill(footer, max(65, int(self.width*17))),
                     fontsize=6.7, color=MUTED, va='bottom')
        for ax in fig.axes:
            if ax.get_legend():
                ax.get_legend().set_in_layout(False)
        for legend in fig.legends:
            legend.set_in_layout(False)
        fig.tight_layout(rect=(0, bottom, 1, top), pad=.8, w_pad=1.1)
        # No tight bounding-box resize: vector physical width remains exactly fixed.
        files = {}
        for fmt in self.formats:
            path = self.root / (name + '.' + fmt)
            metadata = {'Creator': 'PhysAlign publication figures', 'CreationDate': None, 'ModDate': None} if fmt == 'pdf' else {'Date': None} if fmt == 'svg' else None
            fig.savefig(path, dpi=self.dpi, format=fmt, metadata=metadata)
            files[fmt] = path.name
        plt.close(fig)
        self.entries.append({'id': name, 'title': title, 'caption': caption, 'section': section,
                             'status': 'generated', 'files': files})

    def skip(self, name, title, reason):
        self.entries.append({'id': name, 'title': title, 'status': 'unavailable', 'reason': reason, 'section': 'appendix', 'files': {}})

    @staticmethod
    def percent(ax, *, horizontal=False, difference=False):
        ax.grid(axis='x' if horizontal else 'y')
        if horizontal:
            ax.set_xlabel('Difference (percentage points)' if difference else 'Accuracy (%)')
            if not difference:
                ax.set_xlim(-2, 102)
                ax.set_xticks([0, 25, 50, 75, 100])
        else:
            ax.set_ylabel('Difference (percentage points)' if difference else 'Accuracy (%)')
            if not difference:
                ax.set_ylim(-3, 103)
                ax.set_yticks([0, 25, 50, 75, 100])
        if difference:
            (ax.axvline if horizontal else ax.axhline)(0, color=MUTED, lw=.8, linestyle='--')

    def main_results(self):
        fig, axes = self.figure(3.6)
        ax = axes[0, 0]
        metrics = [('CAcc', 'CAcc\n(reading L)'), ('BAcc', 'BAcc\n(binding B)'),
                   ('JAcc', 'JAcc\n(joint L)'), ('BAcc_given_C', 'BAcc | C\n(joint L)'), ('SolveAcc', 'SolveAcc\n(eligible mothers)')]
        offsets = np.linspace(-.22, .22, len(self.models)) if len(self.models) > 1 else [0]
        for i, model in enumerate(self.models):
            for x, (key, _) in enumerate(metrics):
                value = metric_estimate(model, 'solve' if key == 'SolveAcc' else 'raw_all', key)
                _point(ax, x+offsets[i], value['value'], value['ci'], self.colors[i], self.markers[i])
                if value['value'] is None:
                    ax.text(x+offsets[i], 5+8*i, 'N/A', ha='center', fontsize=6.3, color=self.colors[i], rotation=90)
        ax.set_xticks(range(len(metrics)), [v for _, v in metrics])
        self.percent(ax)
        self.legend(ax)
        self.finish(fig, '01_main_results', 'Reading, grounding, and problem solving',
                    'Primary scores use their own frozen populations: CAcc/JAcc on L, BAcc on B, conditional binding on L, and SolveAcc on answer-eligible mothers. Error bars are available clustered bootstrap confidence intervals.',
                    section='main', footer='Separate populations are labeled; unavailable values and withheld intervals are not zeros.', top=.83, bottom=.18)

    def quadrants(self):
        fig, axes = self.figure(max(3.1, 1.8+.38*len(self.models)))
        ax = axes[0, 0]
        for y, model in enumerate(self.models):
            q = model['report']['experiments']['main']['raw_all']['quadrants']
            if any(q[k] is None for k, *_ in QUADRANTS):
                ax.text(50, y, 'Unavailable: incomplete joint support', ha='center', color=MUTED, fontsize=8)
                continue
            left = 0
            for key, _, color, hatch in QUADRANTS:
                width = 100*q[key]
                ax.barh(y, width, left=left, color=color, height=.55, edgecolor='white', linewidth=.8, hatch=hatch)
                if width >= 7:
                    ax.text(left+width/2, y, f'{width:.1f}', ha='center', va='center', fontsize=8,
                            color='white' if key == 'q11' else INK, weight='bold' if key == 'q10' else 'normal')
                left += width
        ax.set_yticks(range(len(self.models)), self.labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Weighted share of joint-eligible probes L (%)')
        handles = [Patch(facecolor=c, hatch=h, label=l) for _, l, c, h in QUADRANTS]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.51, .92), ncol=2, fontsize=7.2)
        self.finish(fig, '02_reading_binding_quadrants', 'When reading is correct but binding fails',
                    'Joint correctness quadrants on the same L pool. The hatched segment is q10 = CAcc - JAcc. It is a share of all L probes, not the conditional error rate 1 - JAcc/CAcc.',
                    section='main', footer='All four segments share the same L weights. Hatched: correct reading, wrong binding.', top=.77, bottom=.17)

    def gold_control(self):
        if not any(m['report']['experiments']['main']['paired'] for m in self.models):
            return self.skip('03_gold_reading_control', 'Gold reading control', 'No predeclared paired Gold views.')
        fig, axes = self.figure(max(3.3, 1.9+.36*len(self.models)), ncols=2, gridspec_kw={'width_ratios': [1.25, 1]})
        left, right = axes[0]
        for y, model in enumerate(self.models):
            a, b = metric_estimate(model, 'paired', 'BAcc'), metric_estimate(model, 'paired', 'BAcc_gold')
            if a['value'] is not None and b['value'] is not None:
                left.plot([100*a['value'], 100*b['value']], [y, y], color=self.colors[y], lw=2, alpha=.55)
            _point(left, y, a['value'], a['ci'], self.colors[y], horizontal=True, hollow=True)
            _point(left, y, b['value'], b['ci'], self.colors[y], horizontal=True)
            d = metric_estimate(model, 'paired', 'delta_BAcc_gold_raw')
            _point(right, y, d['value'], d['ci'], self.colors[y], horizontal=True)
            if d['value'] is None:
                right.text(0, y, 'N/A', ha='center', color=MUTED)
        for ax in (left, right):
            ax.set_yticks(range(len(self.models)), self.labels if ax is left else [])
            ax.invert_yaxis()
        left.set_title('Matched B: Raw to Gold', loc='left')
        right.set_title('Gold - Raw', loc='left')
        fig.legend(handles=[Line2D([], [], marker='o', mfc='white', color=INK, linestyle='', label='Raw'),
                             Line2D([], [], marker='o', color=INK, linestyle='', label='Gold')], ncol=2,
                    loc='upper center', bbox_to_anchor=(.5, .925), fontsize=7.5)
        self.percent(left, horizontal=True)
        self.percent(right, horizontal=True, difference=True)
        self.finish(fig, '03_gold_reading_control', 'Binding with verified local reading',
                    'Raw and Gold binding accuracies use exactly the same predeclared paired probes and weights. Right: paired difference and its shared-cluster bootstrap CI. Full Raw scores are not used as the reference.',
                    section='main', footer='Gold reveals only approved local readings. Its CAcc and JAcc are not evaluated.', top=.84, bottom=.17)

    def association(self):
        if not any(m['report']['original_solving']['association'] for m in self.models):
            return self.skip('04_solving_association', 'Binding and original solving', 'Independent original correctness or complete binding service is unavailable.')
        fig, axes = self.figure(3.45, ncols=2, gridspec_kw={'width_ratios': [1.2, 1]})
        left, right = axes[0]
        for y, model in enumerate(self.models):
            result = model['report']['original_solving']
            a = result['association']
            if not a:
                left.text(50, y, 'N/A', ha='center')
                continue
            values = [a['mean_binding_wrong'], a['mean_binding_correct']]
            if all(v is not None for v in values):
                left.plot(np.array(values)*100, [y, y], color=self.colors[y], lw=1.8, alpha=.5)
            for name, hollow in (('mean_binding_wrong', True), ('mean_binding_correct', False)):
                _point(left, y, a[name], ci_entry(result['intervals'], name), self.colors[y], horizontal=True, hollow=hollow)
            _point(right, y, a['delta_assoc'], ci_entry(result['intervals'], 'delta_assoc'), self.colors[y], horizontal=True)
        for ax in (left, right):
            ax.set_yticks(range(len(self.models)), self.labels if ax is left else [])
            ax.invert_yaxis()
        self.percent(left, horizontal=True)
        left.set_xlabel('Mean mother binding score (%)')
        left.set_title('Original wrong / correct', loc='left')
        self.percent(right, horizontal=True, difference=True)
        right.set_title('Correct - wrong', loc='left')
        fig.legend(handles=[Line2D([], [], marker='o', mfc='white', color=INK, linestyle='', label='Wrong'),
                             Line2D([], [], marker='o', color=INK, linestyle='', label='Correct')], ncol=2,
                    loc='upper center', bbox_to_anchor=(.5, .925), fontsize=7.5)
        self.finish(fig, '04_solving_association', 'Is binding associated with solving?',
                    'Each b_i is the ordinary mean of all B probes of mother i (Eq. 8). Conditional means group mothers by their independently graded original solution. Shared-cluster CIs resample eligible mother records; empty groups are undefined. This is an association, not a causal estimate.',
                    section='main', footer='Eq. (8): mean all B probes within each mother first. Association does not establish causation.', top=.84, bottom=.18)

    def type_heatmap(self):
        types = sorted({t for m in self.models for t in m['report']['experiments']['main']['raw_all']['by_type']})
        fig, axes = self.figure(max(3.65, 2+.35*len(types)), ncols=3)
        cmap = LinearSegmentedColormap.from_list('physalign_blue', ['#F1F5F9', '#B8D4E6', '#397BA3', '#183D5C'])
        cmap.set_bad('#F1F2F4')
        for ax, (metric, title) in zip(axes[0], [('CAcc', 'CAcc on L'), ('BAcc', 'BAcc on B'), ('BindErr_given_C', 'Binding error | C')]):
            matrix = np.array([[m['report']['experiments']['main']['raw_all']['by_type'].get(t, {}).get(metric)
                                if m['report']['experiments']['main']['raw_all']['by_type'].get(t, {}).get(metric) is not None else np.nan
                                for m in self.models] for t in types], dtype=float)
            artist = ax.imshow(matrix*100, cmap=cmap, vmin=0, vmax=100, aspect='auto')
            ax.set_title(title, loc='left', fontsize=8.5)
            ax.set_yticks(range(len(types)), types if ax is axes[0, 0] else [])
            ax.set_xticks(range(len(self.models)), self.labels, rotation=45, ha='right', rotation_mode='anchor')
            ax.tick_params(length=0)
            for y in range(len(types)):
                for x in range(len(self.models)):
                    value = matrix[y, x]
                    ax.text(x, y, '--' if np.isnan(value) else f'{value*100:.0f}', ha='center', va='center',
                            fontsize=7.3, color='white' if value > .58 else INK)
            for spine in ax.spines.values():
                spine.set_visible(False)
        self.finish(fig, '05_type_heatmap', 'Which probe types expose the gap?',
                    'Per-type CAcc, BAcc and conditional binding error. Each cell uses the documented mother aggregation; missing joint eligibility is shown as --. All panels have the same 0-100% color scale. Darker means a larger value, including more error in the third panel; it does not always mean better.',
                    section='main', footer='Cell values are percentages (0-100). Darker = larger; third panel is error (lower is better).', top=.88, bottom=.18)

    def stratified_curve(self, name, output_name, title, xlabel):
        levels = {k for m in self.models for k in m['diagnostics']['strata'][name] if k not in {'None', 'not_applicable'}}
        if not levels:
            return self.skip(output_name, title, 'This stratification was not annotated or has no applicable probes.')
        numeric = name != 'nearest_distance_band'
        order = ['0', '(0,10]', '(10,50]', '(50,inf)']
        levels = sorted(levels, key=lambda x: int(x) if numeric else order.index(x))
        xs = np.array([int(x) for x in levels] if numeric else list(range(len(levels))), dtype=float)
        fig, axes = self.figure(3.55)
        ax = axes[0, 0]
        for i, model in enumerate(self.models):
            entries = model['diagnostics']['strata'][name]
            ys = [entries.get(k, {}).get('summary', {}).get('value') for k in levels]
            if numeric:
                ax.plot(xs, [np.nan if y is None else 100*y for y in ys], color=self.colors[i], alpha=.8)
            for x, level, y in zip(xs, levels, ys):
                _point(ax, x, y, entries.get(level, {}).get('ci'), self.colors[i], self.markers[i])
        ax.set_xticks(xs, ['>50' if k == '(50,inf)' else k for k in levels])
        ax.set_xlabel(xlabel)
        ax.set_ylabel('BAcc within stratum (%)')
        ax.set_ylim(-3, 103)
        ax.grid(axis='y')
        self.legend(ax)
        self.finish(fig, output_name, title,
                    'Exploratory stratified BAcc using the existing within-stratum mother/type aggregation. Membership comes from the frozen study metadata. Confidence intervals, if defined, resample source clusters within each stratum. Support and weights are exported in the stratified table; differences between strata are not controlled causal effects.',
                    footer='Fixed metadata strata; type composition may differ. Lines connect observed levels only.', top=.83, bottom=.17)

    def controls(self):
        names = [('no_image_geometry', 'No image; geometry retained'),
                 ('no_image_no_geometry', 'No image or visual geometry'), ('permutation_reference', 'Re-rendered identity reference')]
        names = [(k, label) for k, label in names if any(k in m['report']['control_comparisons'] for m in self.models)]
        fig, axes = self.figure(3.6)
        ax = axes[0, 0]
        offsets = np.linspace(-.22, .22, len(self.models)) if len(self.models) > 1 else [0]
        for i, model in enumerate(self.models):
            for y, (variant, _) in enumerate(names):
                c = model['report']['control_comparisons'].get(variant, {}).get('raw')
                if c:
                    _point(ax, y+offsets[i], c['metrics']['delta_BAcc'], ci_entry(c['intervals'], 'delta_BAcc'), self.colors[i], self.markers[i], horizontal=True)
        ax.set_yticks(range(len(names)), [fill(label, 23) for _, label in names])
        ax.invert_yaxis()
        self.percent(ax, horizontal=True, difference=True)
        self.legend(ax)
        self.finish(fig, '09_interface_controls', 'How much does the interface matter?',
                    'Binding accuracy changes for image-removal and rendering controls. Each comparison uses the model main condition on the exact same applicable subset. The re-rendered identity reference isolates renderer changes before alias/order comparisons.',
                    footer='Control - matched main Raw. “No image” still exposes geometry unless explicitly removed.', top=.83, bottom=.17)

    def permutation(self):
        if not any(k.startswith('perm_') for m in self.models for k in m['report']['control_comparisons']):
            return self.skip('10_permutation_stability', 'Candidate identity and order', 'No frozen permutation controls.')
        fig, axes = self.figure(3.6, ncols=2)
        mode_markers = {'alias': 'o', 'order': 's', 'both': '^'}
        for ax, condition in zip(axes[0], ('raw', 'gold')):
            any_data = False
            for i, model in enumerate(self.models):
                for variant, conditions in model['report']['control_comparisons'].items():
                    if not variant.startswith('perm_') or condition not in conditions:
                        continue
                    value = conditions[condition]['metrics']
                    if value['canonical_consistency'] is None or value['delta_BAcc'] is None:
                        continue
                    any_data = True
                    ax.plot(100*value['canonical_consistency'], 100*value['delta_BAcc'], mode_markers[variant.split('_')[1]],
                            color=self.colors[i], ms=6, alpha=.8, mfc='white', mew=1.3)
            if not any_data:
                ax.text(.5, .5, 'Unavailable', transform=ax.transAxes, ha='center')
            ax.set_title(condition.capitalize(), loc='left')
            ax.set_xlim(-2, 102)
            ax.set_xticks([0, 50, 100])
            ax.set_xlabel('Canonical consistency (%)')
            self.percent(ax, difference=True)
        axes[0, 1].set_ylabel('')
        handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in zip(self.colors, self.labels)]
        handles += [Line2D([], [], color=MUTED, marker=s, mfc='white', linestyle='', label=m.capitalize()) for m, s in mode_markers.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .91), ncol=3, fontsize=7, columnspacing=1)
        self.finish(fig, '10_permutation_stability', 'Stable physical predictions under renaming?',
                    'Each point is one predeclared permutation. Colors identify models; shapes distinguish alias-only, order-only and joint perturbations. Both axes use the same-renderer identity reference. Consistently wrong predictions can have high consistency; invalid predictions do not count as consistent. No jitter or smoothing alters measured coordinates.',
                    footer='One point per permutation. Accuracy change is relative to the same-renderer reference.', top=.78, bottom=.18)

    def solving_curve(self):
        if not any(m['diagnostics']['binned_solving']['status'] == 'ok' for m in self.models):
            return self.skip('11_solving_by_binding', 'Solve accuracy across binding levels', 'Complete independently graded mother records are unavailable.')
        fig, axes = self.figure(3.55)
        ax = axes[0, 0]
        for i, model in enumerate(self.models):
            rows = model['diagnostics']['binned_solving']['rows']
            xs = [(r['low']+r['high'])/2 for r in rows]
            ys = [np.nan if r['value'] is None else 100*r['value'] for r in rows]
            ax.plot(np.array(xs)*100, ys, color=self.colors[i], alpha=.8)
            for x, row in zip(xs, rows):
                _point(ax, x*100, row['value'], row['ci'], self.colors[i], self.markers[i])
        self.percent(ax)
        ax.set_xlabel('Mother binding score bin (%)')
        ax.set_xticks([10, 30, 50, 70, 90], ['[0,20)', '[20,40)', '[40,60)', '[60,80)', '[80,100]'])
        ax.set_ylabel('Original SolveAcc in bin (%)')
        self.legend(ax)
        self.finish(fig, '11_solving_by_binding', 'Does stronger binding accompany better solving?',
                    'Exploratory conditional SolveAcc across fixed bins of b_i, the ordinary within-mother mean over all B probes. Bins are left-closed and right-open except the last. Entire model curves are withheld if eligible records have missing binding or original correctness; empty bins remain gaps. Bin counts and undefined-bootstrap events are exported. This cannot establish that binding errors cause solving failures.',
                    footer='Exploratory mother-level association; different models may place mothers in different bins.', top=.83, bottom=.17)

    def human(self):
        if not any(m['report'].get('human_model_same_subset') for m in self.models):
            return self.skip('12_human_reference', 'Human and model binding', 'No same-interface human evidence sealed before inference. No ceiling is imputed.')
        fig, axes = self.figure(3.45, ncols=2)
        synthetic = self.bundle['status'] == 'synthetic_preview'
        for ax, condition in zip(axes[0], ('raw', 'gold')):
            for y, model in enumerate(self.models):
                entry = (model['report'].get('human_model_same_subset') or {}).get(condition, {}).get('BAcc')
                if not entry or not entry['human']:
                    continue
                a, b = entry['model']['value'], entry['human']['value']
                if a is not None and b is not None:
                    ax.plot([100*a, 100*b], [y, y], color=self.colors[y], lw=1.7, alpha=.5)
                _point(ax, y, a, None, self.colors[y], horizontal=True)
                _point(ax, y, b, None, self.colors[y], marker='s', horizontal=True, hollow=True)
            ax.set_yticks(range(len(self.models)), self.labels if ax is axes[0, 0] else [])
            ax.invert_yaxis()
            ax.set_title(condition.capitalize(), loc='left')
            self.percent(ax, horizontal=True)
        self.finish(fig, '12_human_reference', 'Matched binding against an ' + ('illustrative reference' if synthetic else 'observed human reference'),
                    'Model and reference use exactly the same human-evaluation mother/probe subset and packet condition. Dots denote model estimates; open squares denote the reference. Human scores average participants within probes before mother/type aggregation. No human CI or human ceiling is invented.',
                    footer='Dot: model. Open square: ' + ('synthetic reference, not participant evidence.' if synthetic else 'actual same-interface human performance. No human CI is inferred.'), top=.87, bottom=.17)

    def gold_transitions(self):
        if not any(m['diagnostics']['gold_transitions'] for m in self.models):
            return self.skip('13_gold_transitions', 'Gold binding transitions', 'No paired Gold probes.')
        names = [('wrong_to_correct', 'Wrong to correct', '#238A83'), ('correct_to_wrong', 'Correct to wrong', '#D55E00'),
                 ('both_correct', 'Correct in both', '#9DC7C1'), ('both_wrong', 'Wrong in both', '#D7DEE7')]
        fig, axes = self.figure(3.35)
        ax = axes[0, 0]
        for y, model in enumerate(self.models):
            entry = model['diagnostics']['gold_transitions']
            if not entry or any(entry['summaries'][k]['value'] is None for k, *_ in names):
                ax.text(50, y, 'Unavailable', ha='center')
                continue
            left = 0
            for key, _, color in names:
                v = entry['summaries'][key]['value']*100
                ax.barh(y, v, left=left, height=.55, color=color, edgecolor='white', linewidth=.8)
                if v >= 7:
                    ax.text(left+v/2, y, f'{v:.1f}', ha='center', va='center', color='white' if key in {'wrong_to_correct', 'correct_to_wrong'} else INK, fontsize=8)
                left += v
        ax.set_yticks(range(len(self.models)), self.labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Weighted paired binding probes (%)')
        fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in names], ncol=2, loc='upper center', bbox_to_anchor=(.51, .92), fontsize=7.4)
        self.finish(fig, '13_gold_transitions', 'What changes after supplying local readings?',
                    'Paired binding transitions are computed per probe before the existing mother/type aggregation. Wrong-to-correct minus correct-to-wrong equals the paired Gold-minus-Raw BAcc difference. This is the Gold reading control, not the separate option-intervention experiment.',
                    footer='Gold gain = wrong-to-correct share - correct-to-wrong share, on the same paired B pool.', top=.79, bottom=.17)

    def profile(self):
        profile = self.bundle['dataset']
        rows = profile['by_type']
        fig, axes = self.figure(3.65, ncols=2)
        left, right = axes[0]
        x = np.arange(len(rows))
        left.bar(x-.16, [r['probes_B'] for r in rows], width=.3, color='#467FA0', label='B probes')
        left.bar(x+.16, [r['probes_L'] for r in rows], width=.3, color='#90BFB8', label='L probes')
        left.set_xticks(x, [r['type'] for r in rows], rotation=35, ha='right')
        left.set_ylabel('Probe count')
        left.set_title('Task composition', loc='left')
        left.legend(fontsize=7.5, loc='upper left')
        counts = sorted(profile['candidate_counts'], key=int)
        right.bar(range(len(counts)), [profile['candidate_counts'][c] for c in counts], color='#6C92AE', width=.7)
        right.set_xticks(range(len(counts)), counts)
        right.set_xlabel('Candidate count K')
        right.set_ylabel('Probe count')
        right.set_title('Candidate-set sizes', loc='left')
        for ax in (left, right):
            ax.grid(axis='y')
            ax.set_ylim(bottom=0)
        self.finish(fig, '14_dataset_profile', 'Benchmark coverage and interface complexity',
                    'Unweighted dataset composition from frozen study metadata. Counts describe the evaluated study, not an assumed larger benchmark. B and its joint-eligible subset L are shown separately; these bars are counts, not score weights.',
                    footer=f"{profile['mothers']} mothers; {profile['source_clusters']} source clusters; {profile['binding_probes']} B probes; {profile['paired_probes']} paired Gold probes.", top=.88, bottom=.17)

    def quality(self):
        fig, axes = self.figure(3.5, ncols=2)
        left, right = axes[0]
        for y, model in enumerate(self.models):
            r = model['report']
            service = r['service']
            coverage = service['completed']/service['planned'] if service['planned'] else None
            invalid = r['experiments']['main']['raw_all']['invalidity']['raw']['object_invalid_rate_served']
            _point(left, y, coverage, None, self.colors[y], horizontal=True)
            _point(right, y, invalid, None, self.colors[y], horizontal=True)
        for ax in (left, right):
            self.percent(ax, horizontal=True)
            ax.set_yticks(range(len(self.models)), self.labels if ax is left else [])
            ax.invert_yaxis()
        left.set_title('Completed / planned requests', loc='left', fontsize=8)
        right.set_title('Invalid Raw JSON / served Raw', loc='left', fontsize=8)
        left.set_xlabel('Service coverage (%)')
        right.set_xlabel('Object-invalid rate (%)')
        self.finish(fig, '15_service_and_invalidity', 'Separate model errors from missing service',
                    'Unweighted operational rates with explicitly different denominators: all planned requests for service coverage; completed main Raw outputs for JSON object invalidity. Missing infrastructure outcomes are not wrong answers, and invalid returned objects are not retried. Required-field errors are detailed in the table even when the JSON object is valid.',
                    footer='Operational rates are unweighted. Invalid JSON objects and invalid required fields are distinct.', top=.87, bottom=.17)

    def budget(self):
        fields = ('input_tokens', 'output_tokens')
        if not any(isinstance(r.get(k), (int, float)) for m in self.models for r in m['token_usage'] for k in fields):
            return self.skip('16_token_usage', 'Recorded inference sizes', 'Adapters did not record actual token usage. Budgets are not substituted for usage.')
        fig, axes = self.figure(3.7, ncols=2)
        for ax, field, title in zip(axes[0], fields, ('Input tokens', 'Generated tokens')):
            for i, model in enumerate(self.models):
                values = [r[field] for r in model['token_usage'] if type(r.get(field)) is int and r[field] >= 0]
                if values:
                    artist = ax.boxplot([values], positions=[i], widths=.46, patch_artist=True, showfliers=False,
                                        medianprops={'color': INK, 'linewidth': 1.5}, whiskerprops={'color': MUTED}, capprops={'color': MUTED})
                    artist['boxes'][0].set(facecolor=self.colors[i], alpha=.28, edgecolor=self.colors[i])
            ax.set_xticks(range(len(self.models)), self.labels, rotation=35, ha='right')
            ax.set_title(title, loc='left')
            ax.set_ylabel('Tokens per main Raw request')
            ax.set_ylim(bottom=0)
            ax.grid(axis='y')
        self.finish(fig, '16_token_usage', 'Actual inference sizes under the fixed budget',
                    'Recorded token distributions on completed main Raw requests only. Box: 25th-75th percentiles; line: median; whiskers: 1.5 IQR convention; outliers omitted from the visual only, retained in source data. These are descriptive distributions, not confidence intervals. Tokenizers and native image preprocessing differ between models.',
                    footer='Observed usage, not configured limits. Different native tokenizers are not a common compute unit.', top=.88, bottom=.19)

    def chance(self):
        if not any(m['report'].get('nearest_region') or m['report']['uniform_random_full']['value'] is not None for m in self.models):
            return self.skip('17_shortcut_baselines', 'Shortcut baselines', 'No defined full-pool chance or eligible nearest-region subset.')
        fig, axes = self.figure(3.5, ncols=2)
        left, right = axes[0]
        for y, model in enumerate(self.models):
            main = metric_estimate(model, 'raw_all', 'BAcc')
            random = model['report']['uniform_random_full']['value']
            _point(left, y, main['value'], main['ci'], self.colors[y], horizontal=True)
            _point(left, y, random, None, self.colors[y], 'x', horizontal=True)
            nearest = model['report'].get('nearest_region')
            if nearest:
                _point(right, y, nearest['metrics']['BAcc_reference'], ci_entry(nearest['intervals'], 'BAcc_reference'), self.colors[y], horizontal=True)
                _point(right, y, nearest['metrics']['BAcc_control'], None, self.colors[y], 'x', horizontal=True)
            else:
                right.text(50, y, 'N/A', ha='center', color=MUTED, fontsize=7)
        for ax in (left, right):
            ax.set_yticks(range(len(self.models)), self.labels if ax is left else [])
            ax.invert_yaxis()
            self.percent(ax, horizontal=True)
        left.set_title('Full B: uniform candidate chance', loc='left', fontsize=8)
        right.set_title('Ownership subset: nearest region', loc='left', fontsize=8)
        self.finish(fig, '17_shortcut_baselines', 'Can simple shortcuts explain the scores?',
                    'Dots denote model BAcc; crosses denote baselines. Uniform chance aggregates per-probe correct-alias probabilities with the model weights. Nearest-region compares only the exact eligible ownership subset. The panels use different declared populations and must not be interpreted as a shared leaderboard.',
                    footer='Dot: model. Cross: baseline. Nearest-subset coverage is reported in the baseline table.', top=.87, bottom=.17)

    def contrasts(self):
        if not self.bundle.get('model_contrasts'):
            return self.skip('18_model_contrasts', 'Paired model differences', 'At least two comparable models are required.')
        n = len(self.models)
        matrix = np.full((n, n), np.nan)
        np.fill_diagonal(matrix, 0)
        ids = [m['model_id'] for m in self.models]
        for r in self.bundle['model_contrasts']:
            i, j = ids.index(r['target']), ids.index(r['reference'])
            d = r['raw_all']['metrics']['BAcc']
            if d is not None:
                matrix[i, j], matrix[j, i] = 100*d, -100*d
        fig, axes = self.figure(max(3.8, 2+.4*n))
        ax = axes[0, 0]
        cmap = LinearSegmentedColormap.from_list('physalign_difference', ['#386D98', '#FAFAF8', '#BA623B'])
        cmap.set_bad('#ECEFF3')
        limit = max(5, float(np.nanmax(np.abs(matrix))))
        artist = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit)
        for i in range(n):
            for j in range(n):
                v = matrix[i, j]
                ax.text(j, i, '--' if np.isnan(v) else ('0' if i == j else f'{v:+.1f}'),
                        ha='center', va='center', color='white' if abs(v) > .65*limit else INK, fontsize=9)
        ax.set_xticks(range(n), self.labels, rotation=30, ha='right')
        ax.set_yticks(range(n), self.labels)
        ax.set_xlabel('Reference model (column)')
        ax.set_ylabel('Target model (row)')
        fig.colorbar(artist, ax=ax, shrink=.85, label='BAcc difference (pp)')
        self.finish(fig, '18_model_contrasts', 'Model differences on the same frozen probes',
                    'Each cell is row-model minus column-model BAcc on the same B pool. Paired CIs are computed by resampling the same source clusters for both models and are exported in the contrasts table. The matrix shows effect sizes, not significance stars; multiple contrasts are not multiplicity-adjusted.',
                    footer='Row - column, in percentage points. Shared-cluster confidence intervals are in Table 16.', top=.89, bottom=.18)

    def table_preview(self, table):
        with plt.rc_context(theme()):
            rows, columns = table['rows'], table['columns']
            fig, axes = self.figure(max(2.75, 1.45+.42*len(rows)))
            ax = axes[0, 0]
            ax.axis('off')
            widths = [1.5] + [1]*(len(columns)-1)
            widths = [w/sum(widths) for w in widths]
            cells = [[fill(str(c).replace(' [', '\n['), 16 if i == 0 else 18, break_long_words=False,
                           replace_whitespace=False) for i, c in enumerate(row)] for row in rows]
            headers = [fill(c, 12, break_long_words=False) for c in columns]
            artist = ax.table(cellText=cells, colLabels=headers, colWidths=widths,
                              cellLoc='right', colLoc='center', bbox=[0, .05, 1, .92])
            artist.auto_set_font_size(False)
            artist.set_fontsize(7.2)
            for (row, col), cell in artist.get_celld().items():
                cell.PAD = .08
                cell.visible_edges = 'TB' if row == 0 else 'B' if row == len(rows) else ''
                cell.set_edgecolor(INK if row in (0, len(rows)) else GRID)
                cell.set_linewidth(.7)
                if row == 0:
                    cell.get_text().set_weight('semibold')
                if col == 0:
                    cell.get_text().set_ha('left')
                if table['id'] == '03_quadrants' and col == 2:
                    cell.get_text().set_color('#B86D25')
                    cell.get_text().set_weight('semibold')
            self.finish(fig, table['id'], table['title'], table['caption'], section=table['section'],
                        footer='Percentages; bracketed clustered confidence intervals when available. -- = unavailable.', top=.9, bottom=.19)

    def render_all(self):
        with plt.rc_context(theme()):
            self.main_results()
            self.quadrants()
            self.gold_control()
            self.association()
            self.type_heatmap()
            self.stratified_curve('candidate_count', '06_candidate_count', 'Binding across candidate-set sizes', 'Number of candidates K')
            self.stratified_curve('panel_count', '07_panel_count', 'Binding across annotated panel counts', 'Annotated original panel count')
            self.stratified_curve('nearest_distance_band', '08_distance_bands', 'Binding across geometric separation', 'Nearest anchor-to-candidate distance (original pixels)')
            self.controls()
            self.permutation()
            self.solving_curve()
            self.human()
            self.gold_transitions()
            self.profile()
            self.quality()
            self.budget()
            self.chance()
            self.contrasts()
        return self.entries
