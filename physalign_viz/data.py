"""Audited inputs and explicitly exploratory diagnostics; never infer model answers."""

from bisect import bisect_right
from collections import Counter
from itertools import combinations
from math import fsum, isfinite
from pathlib import Path

from physalign.bootstrap import paired_cluster_bootstrap
from physalign.contracts import decode_probe, remap_probe
from physalign.dataset import require
from physalign.metrics import Pool, evaluate, uniform_candidate_baseline
from physalign.planning import metric_plan
from physalign.scoring import score_response
from physalign.storage import file_hash, read_json, run_lock
from physalign.study import Study, verify_final_seal
from physalign.study_reporting import audited_records, _comparison, _solving_report

SCHEMA = 'physalign_figure_data_v1'
MODEL_LABELS = {'Qwen/Qwen3.5-4B': 'Qwen3.5-4B', 'Qwen/Qwen3.5-9B': 'Qwen3.5-9B', 'Qwen/Qwen3.5-27B': 'Qwen3.5-27B',
                'OpenGVLab/InternVL3_5-8B-HF': 'InternVL3.5-8B',
                'gemini-3.8-flash': 'Gemini-3.8-flash'}
BINDING_BINS = (0.0, .2, .4, .6, .8, 1.0)
STRATA = ('candidate_count', 'panel_count', 'nearest_distance_band')


def ci_entry(block, metric):
    """A withheld CI stays withheld. A percentile CI need not contain the point."""
    if not block or 'intervals' not in block:
        return None
    value = block['intervals'].get(metric)
    if not value or value.get('status') != 'ok':
        return None
    lo, hi = value['lower'], value['upper']
    require(lo is not None and hi is not None and isfinite(lo) and isfinite(hi) and lo <= hi, 'Malformed confidence interval')
    return [lo, hi]


def estimate(value, *, interval=None, summary=None, support=None):
    return {'value': value, 'ci': interval, 'bounds':
            [summary['lower'], summary['upper']] if summary and summary.get('lower') is not None else None,
            'support': support}


def metric_estimate(model, scope, metric):
    if scope == 'solve':
        result = model['report']['original_solving']
        return estimate(result['SolveAcc'], interval=ci_entry(result['intervals'], 'SolveAcc'),
                        summary=result['SolveAcc_bounds'], support={'planned_problems': result['answer_eligible_mothers']})
    main = model['report']['experiments']['main']
    result = main[scope]
    if result is None:
        return estimate(None)
    support_key = 'joint_raw' if metric in {'CAcc', 'JAcc', 'BAcc_given_C', 'BindErr_given_C'} else 'binding_raw'
    interval = ci_entry(main['intervals'][scope], metric)
    if metric == 'BindErr_given_C':
        before = ci_entry(main['intervals'][scope], 'BAcc_given_C')
        interval = [1 - before[1], 1 - before[0]] if before else None
    return estimate(result['metrics'].get(metric), interval=interval, summary=result['summaries'].get(metric),
                    support=result['support'].get(support_key))


def boot(plan, probes, statistic):
    mothers = sorted({p.problem_id for p in probes})
    if not mothers or plan['bootstrap']['n_resamples'] == 0:
        return {'status': 'disabled_or_empty'}
    cluster_map = {p['problem_id']: p['cluster_id'] for p in plan['probes'] if p['problem_id'] in mothers}
    return paired_cluster_bootstrap(mothers, statistic, cluster_ids=cluster_map, **plan['bootstrap'])


def mother_records(probes, raw, correctness, clusters):
    """Eq. (8): plain mean of ALL B probes in each eligible mother, before grouping."""
    rows = []
    for mother, answer in sorted(correctness.items()):
        selected = [p for p in probes if p.problem_id == mother]
        values = [raw[p.probe_id].B if raw[p.probe_id] is not None else None for p in selected]
        b = fsum(values) / len(values) if values and all(v is not None for v in values) else None
        rows.append({'problem_id': mother, 'cluster_id': clusters[mother], 'answer_correct': answer,
                     'binding_mean': b, 'binding_probes': len(values)})
    return rows


def binned_solving(plan, probes, rows):
    """Descriptive conditional SolveAcc, with shared mother-cluster draws.

    Do not assign incomplete mothers to guessed bins or silently remove them.
    These outcome-defined bins are exploratory, never an intervention/causal test.
    """
    if not rows or any(r['binding_mean'] is None or r['answer_correct'] is None for r in rows):
        return {'status': 'unavailable_incomplete_or_no_eligible_mothers', 'rows': [], 'edges': list(BINDING_BINS)}
    groups = [[] for _ in range(len(BINDING_BINS) - 1)]
    for row in rows:
        index = min(bisect_right(BINDING_BINS, row['binding_mean']) - 1, len(groups) - 1)
        groups[index].append(row)
    def statistic(counts):
        result = {}
        for i, members in enumerate(groups):
            n = sum(counts[r['problem_id']] for r in members)
            result[str(i)] = fsum(counts[r['problem_id']] * r['answer_correct'] for r in members) / n if n else None
        return result
    ids = {r['problem_id'] for r in rows}
    intervals = boot(plan, [p for p in probes if p.problem_id in ids], statistic)
    result = []
    for i, members in enumerate(groups):
        result.append({'low': BINDING_BINS[i], 'high': BINDING_BINS[i + 1],
                       'closed_right': i == len(groups) - 1, 'n_mothers': len(members),
                       'n_clusters': len({r['cluster_id'] for r in members}),
                       'value': fsum(r['answer_correct'] for r in members) / len(members) if members else None,
                       'ci': ci_entry(intervals, str(i))})
    return {'status': 'ok', 'rows': result, 'edges': list(BINDING_BINS), 'intervals': intervals,
            'interpretation': 'Exploratory association, not causal; bins fixed in visualization version'}


def diagnostics(plan, probes, scores, report, *, diagnostic_ci=True):
    raw, gold = scores['main']['raw'], scores['main'].get('gold', {})
    clusters = {r['problem_id']: r['cluster_id'] for r in plan['probes']}
    mothers = mother_records(probes, raw, report['original_solving']['answer_correctness'], clusters)
    paired = tuple(p for p in probes if p.probe_id in gold)
    transitions = None
    if paired:
        pool = Pool(paired, plan['weights']['main']['paired']['binding'])
        events = {name: {} for name in ('wrong_to_correct', 'correct_to_wrong', 'both_correct', 'both_wrong')}
        for p in paired:
            a, b = raw[p.probe_id], gold[p.probe_id]
            values = (None,) * 4 if a is None or b is None else ((1-a.B)*b.B, a.B*(1-b.B), a.B*b.B, (1-a.B)*(1-b.B))
            for name, value in zip(events, values):
                events[name][p.probe_id] = value
        summaries = {name: pool.estimate(values) for name, values in events.items()}
        transitions = {'summaries': summaries,
            'intervals': boot(plan, paired, lambda counts: {k: pool.estimate(v, problem_multiplicities=counts)['value']
                               for k, v in events.items()}) if diagnostic_ci else None,
            'interpretation': 'Transitions in binding accuracy under the fixed Gold reading condition'}
    strata = {}
    for name in STRATA:
        strata[name] = {}
        levels = sorted({str(plan['strata'][p.probe_id][name]) for p in probes})
        for level in levels:
            subset = tuple(p for p in probes if str(plan['strata'][p.probe_id][name]) == level)
            pool = Pool(subset)
            values = {p.probe_id: raw[p.probe_id].B if raw[p.probe_id] is not None else None for p in subset}
            summary = pool.estimate(values)
            interval = boot(plan, subset, lambda counts: {'BAcc': pool.estimate(values, problem_multiplicities=counts)['value']}) if diagnostic_ci else None
            strata[name][level] = {'summary': summary, 'ci': ci_entry(interval, 'BAcc'),
                                   'intervals': interval, 'type_weights': dict(pool.type_weights),
                                   'n_probes': len(subset), 'n_mothers': len({p.problem_id for p in subset})}
    return {'mother_records': mothers, 'gold_transitions': transitions, 'strata': strata,
            'binned_solving': binned_solving(plan if diagnostic_ci else {**plan, 'bootstrap': {'n_resamples': 0}}, probes, mothers)}


def dataset_profile(plan, probes):
    rows = []
    paired = set(plan['groups']['main'].get('gold', []))
    for kind in sorted({p.probe_type for p in probes}):
        ps = [p for p in probes if p.probe_type == kind]
        rows.append({'type': kind, 'probes_B': len(ps), 'probes_L': sum(p.joint_eligible for p in ps),
                     'mothers': len({p.problem_id for p in ps}), 'gold_probes': sum(p.probe_id in paired for p in ps)})
    return {'by_type': rows, 'mothers': len({p.problem_id for p in probes}),
            'source_clusters': len({p['cluster_id'] for p in plan['probes']}),
            'candidate_counts': dict(sorted(Counter(str(s['candidate_count']) for s in plan['strata'].values()).items())),
            'binding_probes': len(probes), 'joint_probes': sum(p.joint_eligible for p in probes),
            'paired_probes': len(paired), 'human_mothers_planned': len(plan['human_mothers'])}


def model_contrasts(plan, probes, by_model):
    results = []
    pairs = set(plan['groups']['main'].get('gold', []))
    for first, second in combinations(by_model, 2):
        entry = {'reference': first, 'target': second}
        for scope, selected, keys in (
                ('raw_all', probes, ('CAcc', 'BAcc', 'JAcc', 'BAcc_given_C')),
                ('paired', [p for p in probes if p.probe_id in pairs], ('BAcc_gold', 'delta_BAcc_gold_raw'))):
            if not selected:
                entry[scope] = None
                continue
            spec = metric_plan(selected, plan['weights']['main']['full' if scope == 'raw_all' else 'paired'])
            def stat(counts=None):
                estimates = []
                for name in (first, second):
                    source = by_model[name]['main']
                    raw = {p.probe_id: source['raw'][p.probe_id] for p in selected}
                    gold = {p.probe_id: source['gold'][p.probe_id] for p in selected} if scope == 'paired' else None
                    estimates.append(evaluate(spec, raw, gold, problem_multiplicities=counts)['metrics'])
                a, b = estimates
                return {key: b[key]-a[key] if a[key] is not None and b[key] is not None else None for key in keys}
            entry[scope] = {'metrics': stat(), 'intervals': boot(plan, selected, stat)}
        results.append(entry)
    return results


def _equal(actual, expected, location):
    require(actual == expected, f'Stale or inconsistent scored report at {location}; score-study this run again')


def _audit_intervals(block, metrics):
    if not block or 'intervals' not in block:
        return
    for key, entry in block['intervals'].items():
        _equal(entry['estimate'], metrics[key], 'CI point/' + key)
        require(entry['valid_resamples'] + entry['undefined_resamples'] == block['n_resamples'], 'Inconsistent bootstrap support')
        if entry['status'] == 'ok':
            require(entry['undefined_resamples'] == 0 and entry['estimate'] is not None, 'CI reported despite undefined bootstrap support')
            interval = ci_entry(block, key)
            minimum = -1 if 'delta' in key else 0
            require(minimum-1e-12 <= interval[0] <= interval[1] <= 1+1e-12, 'CI outside score domain')
        else:
            require(entry['lower'] is None and entry['upper'] is None, 'Withheld CI must have absent bounds')


def audit_report(study, manifest, records, report, probes):
    """Re-score primary values and controls from FIRST-response logs; no run mutation."""
    plan = study.plan
    _equal((report['schema_version'], report['plan_hash'], report['run_id'], report['adapter']),
           ('physalign_study_report_v1', plan['plan_hash'], manifest['run_id'], manifest['adapter']), 'identity')
    _equal(report['scientific_run'], manifest['adapter']['scientific_run'], 'scientific status')
    service = {'planned': len(records), 'completed': sum(r is not None and r['status'] == 'completed' for r in records),
               'infrastructure_missing': sum(r is not None and r['status'] == 'infrastructure_missing' for r in records),
               'pending': sum(r is None for r in records)}
    _equal(report['service'], service, 'service coverage')
    scores = {}
    for row, record in zip(plan['requests'], records):
        if row['kind'] != 'probe':
            continue
        variant, condition, iid = row['variant'], row['score_condition'], row['instance_id']
        p = probes[iid]
        if variant in plan['permutations']:
            p = remap_probe(p, plan['permutations'][variant][iid]['old_to_new'])
        result = score_response(p, record['response']['text'], condition=condition) if record is not None and record['status'] == 'completed' else None
        scores.setdefault(variant, {}).setdefault(condition, {})[iid] = result
    for variant, condition_scores in scores.items():
        ps = [probes[iid] for iid in condition_scores['raw']]
        if variant in plan['permutations']:
            ps = [remap_probe(p, plan['permutations'][variant][p.probe_id]['old_to_new']) for p in ps]
        raw, gold = condition_scores['raw'], condition_scores.get('gold', {})
        result = report['experiments'][variant]
        _equal(result['raw_all'], evaluate(metric_plan(ps, plan['weights'][variant]['full']), raw,
                                          gold if set(gold) == set(raw) else None), variant + '/raw')
        pairs = [p for p in ps if p.probe_id in gold]
        paired = evaluate(metric_plan(pairs, plan['weights'][variant]['paired']), {p.probe_id: raw[p.probe_id] for p in pairs}, gold) if pairs else None
        _equal(result['paired'], paired, variant + '/paired')
        _audit_intervals(result['intervals']['raw_all'], result['raw_all']['metrics'])
        if paired:
            _audit_intervals(result['intervals']['paired'], paired['metrics'])
        if variant != 'main':
            reference = scores['permutation_reference'] if variant in plan['permutations'] else scores['main']
            for condition, after in condition_scores.items():
                weights = plan['weights'][variant]['paired' if condition == 'gold' else 'full']['binding']
                expected = _comparison({**plan, 'bootstrap': {'n_resamples': 0}}, [probes[i] for i in after], reference[condition], after, weights)
                _equal({k: v for k, v in report['control_comparisons'][variant][condition].items() if k != 'intervals'},
                       {k: v for k, v in expected.items() if k != 'intervals'}, variant + '/control')
                _audit_intervals(report['control_comparisons'][variant][condition]['intervals'], expected['metrics'])
    private = study.private()
    full_pool = metric_plan(list(probes.values()), plan['weights']['main']['full']).binding_pool
    chance = uniform_candidate_baseline(full_pool)['value'] if all(p.binding.kind == 'single' for p in probes.values()) else None
    _equal(report['uniform_random_full']['value'], chance, 'full-pool chance')
    if plan['nearest_ids']:
        ps = [probes[i] for i in plan['nearest_ids']]
        heuristic = {p.probe_id: score_response(p, private['nearest'][p.probe_id]['response']) for p in ps}
        expected = _comparison({**plan, 'bootstrap': {'n_resamples': 0}}, ps, scores['main']['raw'], heuristic, plan['weights']['nearest']['binding'])
        for key, value in expected.items():
            if key != 'intervals':
                _equal(report['nearest_region'][key], value, 'nearest/' + key)
        coverage = full_pool.estimate({p.probe_id: int(p.probe_id in private['nearest']) for p in probes.values()})['value']
        _equal(report['nearest_region']['weighted_full_pool_coverage'], coverage, 'nearest coverage')
        _audit_intervals(report['nearest_region']['intervals'], expected['metrics'])
    else:
        _equal(report['nearest_region'], None, 'nearest eligibility')
    # Human correctness comes from the user's scored artifact, pinned by its hash;
    # the original independent grading artifact hash is preserved in provenance.
    manual = {m: a for m, a in report['original_solving']['answer_correctness'].items()
              if private['original_keys'].get(m) and private['original_keys'][m]['kind'] == 'human' and a is not None}
    require(not manual or bool(report.get('manual_grades_hash')), 'Human original grades lack an evidence hash')
    require(all(type(v) is int and v in (0, 1) for v in manual.values()), 'Manual correctness must be binary')
    expected = _solving_report({**plan, 'bootstrap': {'n_resamples': 0}}, list(probes.values()), scores['main']['raw'], records, private, manual)
    _equal({k: v for k, v in report['original_solving'].items() if k != 'intervals'},
           {k: v for k, v in expected.items() if k != 'intervals'}, 'original solving')
    _audit_intervals(report['original_solving']['intervals'],
                     {'SolveAcc': expected['SolveAcc'], **(expected['association'] or {})})
    seal_hash = manifest.get('protocol_metadata', {}).get('study_seal_hash')
    _equal(report.get('study_seal_hash'), seal_hash, 'human seal')
    _equal(report.get('human_same_interface'), verify_final_seal(study)['human_report'] if seal_hash else None, 'human evidence')
    if seal_hash:
        expected = {}
        human = report['human_same_interface']
        for condition in ('raw', 'gold'):
            expected[condition] = {}
            ps = [p for p in probes.values() if p.problem_id in plan['human_mothers'] and p.probe_id in scores['main'].get(condition, {})]
            for metric, attribute in (('BAcc', 'B'), ('CAcc', 'C'), ('JAcc', 'J')):
                if condition == 'gold' and attribute != 'B':
                    continue
                subset = ps if attribute == 'B' else [p for p in ps if p.joint_eligible]
                model = Pool(tuple(subset)).estimate({p.probe_id: getattr(scores['main'][condition][p.probe_id], attribute)
                    if scores['main'][condition][p.probe_id] is not None else None for p in subset})
                h = human['summaries'][condition][metric] if human['summaries'][condition] else None
                expected[condition][metric] = {'model': model, 'human': h, 'human_minus_model':
                    h['value']-model['value'] if h is not None and h['value'] is not None and model['value'] is not None else None}
        _equal(report.get('human_model_same_subset'), expected, 'human/model matched subset')
    else:
        _equal(report.get('human_model_same_subset'), None, 'unmeasured human comparison')
    return scores


def load_bundle(study_root, run_directories, *, allow_non_scientific=False, allow_incomplete=False,
                labels=None, diagnostic_ci=True, progress=None, allow_model_specific_settings=False):
    study = Study(study_root)
    plan = study.plan
    probes = {iid: decode_probe(p) for iid, p in study.private()['probes'].items()}
    models, seen, shared, by_model = [], set(), None, {}
    protocols, all_settings_equal = {}, True
    for directory in run_directories:
        run = Path(directory).resolve()
        with run_lock(run):
            manifest, records = audited_records(study, run, allow_incomplete=allow_incomplete)
            report_path = run / 'study_report.json'
            require(report_path.is_file(), 'Run score-study before plotting')
            report = read_json(report_path)
            model_id = manifest['adapter']['model_id']
            require(model_id not in seen, 'Duplicate model ID; plot ablations in separate panels')
            seen.add(model_id)
            require(report['scientific_run'] or allow_non_scientific, 'Smoke/replay data require --allow-non-scientific and will be marked on every artifact')
            settings = manifest['adapter']['settings']
            comparable = {k: settings.get(k) for k in ('max_new_tokens', 'max_input_tokens', 'thinking', 'do_sample', 'num_beams', 'seed')}
            comparable.update({k: settings.get(k) for k in ('max_output_tokens', 'token_limit_field', 'reasoning_effort', 'temperature')})
            equal = shared is None or comparable == shared
            require(equal or allow_model_specific_settings, 'Model panel mixes decoding/thinking/token budgets')
            all_settings_equal = all_settings_equal and equal
            if shared is None:
                shared = comparable
            protocols[model_id] = {'settings': settings, 'preprocessing': manifest['adapter']['preprocessing'],
                                   'adapter': manifest['adapter']['name']}
            scores = audit_report(study, manifest, records, report, probes)
            by_model[model_id] = scores
            extra = diagnostics(plan, list(probes.values()), scores, report, diagnostic_ci=diagnostic_ci)
            tokens = []
            for row, record in zip(plan['requests'], records):
                if row['variant'] != 'main' or row['score_condition'] != 'raw' or record is None or record['status'] != 'completed':
                    continue
                usage = record['response'].get('usage', {})
                tokens.append({'problem_id': row['problem_id'], 'input_tokens': usage.get('input_tokens'),
                               'output_tokens': usage.get('output_tokens'), 'finish_reason': record['response'].get('finish_reason')})
            default_label = (f'GPT-6 Astra ({settings.get("reasoning_effort", "unspecified")})'
                             if model_id == 'gpt-6-astra' else MODEL_LABELS.get(model_id, model_id))
            if model_id.startswith('Qwen/Qwen3.5-') and type(settings.get('thinking')) is bool:
                default_label += ' (thinking)' if settings['thinking'] else ' (non-thinking)'
            label = (labels or {}).get(model_id, default_label)
            models.append({'model_id': model_id, 'label': label,
                           'report': report, 'diagnostics': extra, 'token_usage': tokens,
                           'source': {'run_id': manifest['run_id'], 'report_sha256': file_hash(report_path),
                                      'manifest_sha256': file_hash(run / 'manifest.json'), 'manual_grades_hash': report.get('manual_grades_hash')}})
            if progress:
                progress({'model': model_id, 'audited_requests': len(records)})
    require(models, 'At least one scored run is required')
    status = 'scientific' if all(m['report']['scientific_run'] for m in models) else 'non_scientific'
    return {'schema_version': SCHEMA, 'status': status, 'plan_hash': plan['plan_hash'], 'models': models,
            'dataset': dataset_profile(plan, list(probes.values())), 'bootstrap': plan['bootstrap'],
            'design': {'binding_bins': list(BINDING_BINS), 'diagnostic_ci': diagnostic_ci,
                       'stratified_analysis': 'exploratory; fixed within-stratum mother/type aggregation',
                       'causal_claim_supported': False}, 'shared_settings': shared if all_settings_equal else None,
            'model_protocols': protocols,
            'comparison_note': ('Model-specific reasoning, generation budgets and image processing; see model_protocols.json.'
                                if allow_model_specific_settings else None),
            'source_study_hash': file_hash(study.root / 'plan.json'),
            'model_contrasts': model_contrasts(plan, list(probes.values()), by_model)}
