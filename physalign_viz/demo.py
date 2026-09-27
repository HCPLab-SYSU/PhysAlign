"""Unmistakably synthetic DESIGN previews; never uses real model names or run logs."""

import random

from physalign.adapters import AdapterInfo
from physalign.metrics import Pool, uniform_candidate_baseline
from physalign.planning import weights_for
from physalign.scoring import BindingTarget, Probe, ReadTarget, score_response
from physalign.storage import canonical
from physalign.study_reporting import _condition_report, _comparison, _solving_report

from .data import SCHEMA, diagnostics, dataset_profile, model_contrasts


def demo_bundle(*, seed=2027, n_resamples=200):
    rng = random.Random(seed)
    probes, strata = [], {}
    mothers = [f'demo-{i:03d}' for i in range(36)]
    difficulty = {m: rng.random() for m in mothers}
    for i, mother in enumerate(mothers):
        for t in range(3):
            for j in range(1 + i % 3):
                iid = f'{mother}-{t}-{j}'
                k = (2, 4, 6, 8)[(i + j + t) % 4]
                domain = {f'E{n + 1}': f'entity-{n + 1}' for n in range(k)}
                p = Probe(iid, mother, f'Type {chr(65+t)}', BindingTarget('single', {'owner': domain}, ('entity-1',)),
                          (ReadTarget('read', 'R1', '2 kg'),) if t < 2 else (), packet_nonempty=t < 2)
                probes.append(p)
                strata[iid] = {'candidate_count': k, 'panel_count': 1 + i % 3,
                              'nearest_distance_band': ('0', '(0,10]', '(10,50]', '(50,inf)')[i % 4]}
    ids = [p.probe_id for p in probes]
    # Intentionally incomplete Gold eligibility demonstrates why full Raw cannot
    # be subtracted from paired Gold, even in a design preview.
    gold_ids = [p.probe_id for p in probes if int(p.problem_id[-3:]) % 5 != 0]
    pairs = [p for p in probes if p.probe_id in gold_ids]
    w = {'full': weights_for(probes), 'paired': weights_for(pairs)}
    plan = {'plan_hash': 'SYNTHETIC-DESIGN-PREVIEW',
            'probes': [{'instance_id': p.probe_id, 'problem_id': p.problem_id, 'cluster_id': p.problem_id} for p in probes],
            'bootstrap': {'n_resamples': n_resamples, 'seed': seed, 'confidence': .95},
            'weights': {'main': w}, 'strata': strata, 'groups': {'main': {'raw': ids, 'gold': gold_ids}},
            'human_mothers': mothers}
    models, by_model = [], {}
    human_values = {p.probe_id: int(rng.random() < .96) for p in probes}
    # Synthetic fixed shortcut predictions are shared by every model, just as
    # the real nearest-region baseline is frozen in the study's private plan.
    nearest_ps = [p for p in probes if p.probe_type == 'Type A']
    heuristic = {p.probe_id: score_response(p, canonical({'read': '2 kg',
                 'owner': 'E1' if i % 5 < 2 else 'E2'})) for i, p in enumerate(nearest_ps)}
    for index, (recognition, binding) in enumerate(((.88, .60), (.95, .77), (.91, .68))):
        model_id = f'Synthetic model {chr(65+index)}'
        def sample(p, condition='raw', shift=0):
            d = difficulty[p.problem_id]
            c = rng.random() < recognition - .14 * d
            b = rng.random() < max(.02, min(.99, binding - .23*d - .015*(strata[p.probe_id]['candidate_count']-2)
                                          + (.13 if condition == 'gold' else 0) + shift))
            return score_response(p, canonical({'read': '2 kg' if c else '3 kg', 'owner': 'E1' if b else 'E2'}), condition=condition)
        raw = {p.probe_id: sample(p) for p in probes}
        gold = {p.probe_id: sample(p, 'gold') for p in pairs}
        scores = {'main': {'raw': raw, 'gold': gold}, 'permutation_reference': {'raw': raw, 'gold': gold}}
        scores['no_image_geometry'] = {'raw': {p.probe_id: sample(p, shift=-.18) for p in probes}}
        strict = [p for p in probes if p.probe_type == 'Type B']
        scores['no_image_no_geometry'] = {'raw': {p.probe_id: sample(p, shift=-.27) for p in strict}}
        for mode in ('alias', 'order', 'both'):
            scores[f'perm_{mode}_000'] = {}
            for condition, source in (('raw', raw), ('gold', gold)):
                scores[f'perm_{mode}_000'][condition] = {iid: value if rng.random() < .82 else
                    sample(value.probe, condition, shift=-.05) for iid, value in source.items()}
        by_model[model_id] = scores
        reports, comparisons = {}, {}
        for variant, conditions in scores.items():
            ps = [p for p in probes if p.probe_id in conditions['raw']]
            pp = [p for p in ps if p.probe_id in conditions.get('gold', {})]
            weights = {'full': weights_for(ps), 'paired': weights_for(pp)}
            reports[variant] = _condition_report(plan, ps, conditions, weights)
            if variant != 'main':
                reference = scores['permutation_reference'] if variant.startswith('perm_') else scores['main']
                comparisons[variant] = {c: _comparison(plan, [p for p in probes if p.probe_id in values], reference[c], values,
                    weights['paired' if c == 'gold' else 'full']['binding']) for c, values in conditions.items()}
        keys = {m: {'kind': 'choice', 'accepted': ['A'], 'reliable': True, 'source_reference': 'SYNTHETIC FIXTURE ONLY'} for m in mothers}
        records = []
        for mother in mothers:
            probability = .68 + .10*index - .65*difficulty[mother]
            records.append({'problem_id': mother, 'condition': 'solve.solve', 'status': 'completed',
                            'response': {'text': canonical({'answer': 'A' if rng.random() < probability else 'B'})}})
        solving = _solving_report(plan, probes, raw, records, {'original_keys': keys}, {})
        nearest = _comparison(plan, nearest_ps, raw, heuristic, weights_for(nearest_ps)['binding'])
        nearest['weighted_full_pool_coverage'] = Pool(tuple(probes)).estimate({p.probe_id: int(p in nearest_ps) for p in probes})['value']
        human_comparison = {}
        for condition, source in (('raw', raw), ('gold', gold)):
            ps = tuple(probes if condition == 'raw' else pairs)
            pool = Pool(ps)
            model_score = pool.estimate({p.probe_id: source[p.probe_id].B for p in ps})
            # This is an illustrative reference, NOT simulated participant evidence.
            reference = pool.estimate({p.probe_id: human_values[p.probe_id] for p in ps})
            human_comparison[condition] = {'BAcc': {'model': model_score, 'human': reference,
                                                  'human_minus_model': reference['value'] - model_score['value']}}
        adapter = AdapterInfo('design-demo', model_id, 'synthetic', '1', '{}', '{}', False).manifest()
        n_requests = sum(len(v) for conditions in scores.values() for v in conditions.values()) + len(records)
        report = {'schema_version': 'physalign_study_report_v1', 'plan_hash': plan['plan_hash'],
                  'run_id': 'synthetic-' + str(index), 'scientific_run': False, 'adapter': adapter,
                  'experiments': reports, 'control_comparisons': comparisons, 'original_solving': solving,
                  'nearest_region': nearest, 'uniform_random_full': uniform_candidate_baseline(Pool(tuple(probes))),
                  'uniform_random_by_type': {t: uniform_candidate_baseline(Pool(tuple(p for p in probes if p.probe_type == t))) for t in ('Type A', 'Type B', 'Type C')},
                  'human_model_same_subset': human_comparison, 'human_same_interface': None,
                  'service': {'planned': n_requests, 'completed': n_requests, 'infrastructure_missing': 0, 'pending': 0}}
        extra = diagnostics(plan, probes, scores, report)
        usage = [{'problem_id': p.problem_id, 'input_tokens': 1200 + 420*strata[p.probe_id]['candidate_count'] + rng.randrange(500),
                  'output_tokens': 30+rng.randrange(150), 'finish_reason': 'stop'} for p in probes]
        models.append({'model_id': model_id, 'label': f'Demo {chr(65+index)}', 'report': report,
                       'diagnostics': extra, 'token_usage': usage, 'source': {'kind': 'synthetic_design_preview', 'seed': seed}})
    return {'schema_version': SCHEMA, 'status': 'synthetic_preview', 'plan_hash': plan['plan_hash'],
            'models': models, 'dataset': dataset_profile(plan, probes), 'bootstrap': plan['bootstrap'],
            'shared_settings': {}, 'design': {'binding_bins': [0, .2, .4, .6, .8, 1],
                'diagnostic_ci': True, 'causal_claim_supported': False, 'seed': seed},
            'notice': 'SYNTHETIC DESIGN PREVIEW. No real models, participants, or benchmark results.',
            'model_contrasts': model_contrasts(plan, probes, by_model)}
