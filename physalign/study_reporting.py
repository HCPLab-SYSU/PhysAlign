"""Five-experiment reports, using the original item/mother/type math throughout."""

from collections import defaultdict
from pathlib import Path

from .bootstrap import paired_cluster_bootstrap
from .contracts import decode_probe, remap_probe
from .dataset import require
from .metrics import Pool, evaluate, solving_association, uniform_candidate_baseline
from .planning import metric_plan
from .runner import collect_records, request_id, validate_manifest
from .scoring import score_response
from .solving import grade_original
from .storage import atomic_json, atomic_text, canonical, fingerprint, read_json, run_lock
from .study import Study, verify_final_seal


def audited_records(study, run, *, allow_incomplete=False):
    run = Path(run)
    manifest = read_json(run / 'manifest.json')
    validate_manifest(manifest)
    require(manifest['plan'] == study.plan, 'Run belongs to a different frozen study')
    expected_seal = manifest.get('protocol_metadata', {}).get('study_seal_hash')
    if expected_seal is not None or study.plan['base_plan']['split'] == 'test':
        seal = verify_final_seal(study)
        require(seal is not None and seal['seal_hash'] == expected_seal,
                'Run is not tied to this pre-inference human evidence seal')
    records = collect_records(run, manifest, allow_incomplete=allow_incomplete)
    settings = canonical(manifest['adapter']['settings'])
    for row, record in zip(study.plan['requests'], records):
        if record is None:
            continue
        rid = request_id(manifest, row)
        snapshot = study.request(row, rid, settings).public_snapshot()
        require(read_json(run / 'requests' / (rid + '.json')) == snapshot and
                record['request_hash'] == fingerprint(snapshot), 'Logged request differs from frozen public experiment')
    export = run / 'predictions.jsonl'
    if export.exists():
        require(export.read_text(encoding='utf-8') == ''.join(canonical(r) + '\n' for r in records if r is not None), 'Prediction export disagrees with authoritative logs')
    return manifest, records


def _bootstrap(plan, ids, statistic):
    if plan['bootstrap']['n_resamples'] == 0:
        return {'status': 'disabled_in_frozen_plan'}
    ids = set(ids)
    clusters = {p['problem_id']: p['cluster_id'] for p in plan['probes'] if p['problem_id'] in ids}
    return paired_cluster_bootstrap(sorted(ids), statistic, cluster_ids=clusters, **plan['bootstrap'])


def _condition_report(plan, probes, scores, weights):
    raw = scores['raw']
    spec = metric_plan(probes, weights['full'])
    full_gold = scores.get('gold') if set(scores.get('gold', {})) == set(raw) else None
    result = {'raw_all': evaluate(spec, raw, full_gold), 'paired': None}
    paired_probes = [p for p in probes if p.probe_id in scores.get('gold', {})]
    paired_spec = metric_plan(paired_probes, weights['paired']) if paired_probes else None
    paired_raw = {p.probe_id: raw[p.probe_id] for p in paired_probes}
    if paired_spec:
        result['paired'] = evaluate(paired_spec, paired_raw, scores['gold'])
    def intervals(sp, r, g):
        def stat(counts):
            m = evaluate(sp, r, g, problem_multiplicities=counts)['metrics']
            return {k: m[k] for k in ('CAcc', 'BAcc', 'JAcc', 'BAcc_given_C', 'BAcc_gold', 'delta_BAcc_gold_raw')}
        return _bootstrap(plan, {p.problem_id for p in sp.probes}, stat)
    result['intervals'] = {'raw_all': intervals(spec, raw, full_gold),
                           'paired': intervals(paired_spec, paired_raw, scores['gold']) if paired_spec else None}
    return result


def _comparison(plan, probes, before, after, weights):
    pool = Pool(tuple(probes), weights)
    def values(scores):
        return {p.probe_id: None if scores[p.probe_id] is None else scores[p.probe_id].B for p in probes}
    a, b = values(before), values(after)
    consistency = {}
    for p in probes:
        x, y = before[p.probe_id], after[p.probe_id]
        consistency[p.probe_id] = (None if x is None or y is None else
                                  int(x.canonical_binding is not None and y.canonical_binding is not None
                                      and x.canonical_binding == y.canonical_binding))
    def stat(counts=None):
        first = pool.estimate(a, problem_multiplicities=counts)['value']
        second = pool.estimate(b, problem_multiplicities=counts)['value']
        return {'BAcc_reference': first, 'BAcc_control': second,
                'delta_BAcc': second - first if first is not None and second is not None else None,
                'canonical_consistency': pool.estimate(consistency, problem_multiplicities=counts)['value']}
    x, y = pool.estimate(a), pool.estimate(b)
    return {'metrics': stat(), 'reference': x, 'control': y,
            'consistency': pool.estimate(consistency), 'invalid_predictions_count_as_consistent': False,
            'delta_bounds': {'lower': y['lower'] - x['upper'], 'upper': y['upper'] - x['lower']},
            'intervals': _bootstrap(plan, {p.problem_id for p in probes}, stat)}


def _manual_grades(path, records, private, manifest):
    if path is None:
        return {}, None
    artifact = read_json(Path(path))
    require(artifact.get('schema_version') == 'physalign_solve_grades_v1', 'Unknown manual grade artifact')
    require((artifact.get('plan_hash'), artifact.get('run_id')) ==
            (manifest['plan']['plan_hash'], manifest['run_id']), 'Manual grades belong to another plan/run')
    by_id = {r['request_id']: r for r in records if r is not None}
    result = {}
    for grade in artifact['grades']:
        mother = grade['problem_id']
        require(mother not in result, 'Duplicate final original grade; adjudicate disagreements first')
        key = private['original_keys'][mother]
        require(key is not None and key['kind'] == 'human', 'Manual grade supplied for an ineligible/automatic original')
        r = by_id[grade['request_id']]
        require(r['problem_id'] == mother and r['condition'] == 'solve.solve' and r['status'] == 'completed', 'Grade belongs to a different/unserved original request')
        require(grade['response_hash'] == fingerprint(r['response']) and grade['reference_hash'] == fingerprint(key), 'Grade source response/reference changed')
        require(type(grade['correct']) is int and grade['correct'] in (0, 1), 'Original correctness must be binary')
        require(grade.get('human_confirmed') is True and bool(grade.get('grader_id')) and bool(grade.get('rationale')), 'Manual grades need an actual human identity and rationale')
        result[mother] = grade['correct']
    return result, fingerprint(artifact)


def _solving_report(plan, probes, raw, records, private, manual):
    served = {r['problem_id']: r for r in records if r is not None and r['condition'] == 'solve.solve'}
    answers, missing = {}, []
    for mother, key in private['original_keys'].items():
        if key is None:
            continue  # Eligibility fixed BEFORE model inference, disclosed below.
        r = served.get(mother)
        value = (grade_original(r['response']['text'], key) if r is not None and r['status'] == 'completed' else None)
        if key['kind'] == 'human' and r is not None and r['status'] == 'completed':
            value = manual.get(mother)
        answers[mother] = value
        if value is None:
            missing.append(mother)
    n = len(answers)
    lower = sum(a for a in answers.values() if a is not None) / n if n else None
    upper = (sum(a for a in answers.values() if a is not None) + len(missing)) / n if n else None
    result = {'answer_correctness': answers, 'answer_eligible_mothers': n,
              'ineligible_no_reliable_answer': [m for m, k in private['original_keys'].items() if k is None],
              'missing_or_ungraded': missing, 'SolveAcc': lower if n and not missing else None,
              'SolveAcc_bounds': {'lower': lower, 'upper': upper}, 'association': None, 'intervals': None}
    relevant = [p for p in probes if p.problem_id in answers]
    if n and not missing and all(raw[p.probe_id] is not None for p in relevant):
        spec = metric_plan(probes, plan['weights']['main']['full'])
        result['association'] = solving_association(spec, raw, answers)
        def stat(counts):
            # The sampling frame is the predeclared (A_i, b_i) records only.
            # Keep other mothers at zero multiplicity to satisfy the full plan.
            counts = {**dict.fromkeys({p.problem_id for p in probes}, 0), **counts}
            m = solving_association(spec, raw, answers, problem_multiplicities=counts)
            return {k: m[k] for k in ('SolveAcc', 'mean_binding_correct', 'mean_binding_wrong', 'delta_assoc')}
        result['intervals'] = _bootstrap(plan, set(answers), stat)
        result['correctly_solved_with_binding_errors'] = [m for m, b in result['association']['binding_by_problem'].items() if answers[m] == 1 and b < 1]
    return result


def score_study(study_root, run_directory, *, grades_path=None, allow_incomplete=False):
    study = Study(study_root)
    plan, private = study.plan, study.private()
    run = Path(run_directory).resolve()
    with run_lock(run):
        manifest, records = audited_records(study, run, allow_incomplete=allow_incomplete)
        probes = {iid: decode_probe(r) for iid, r in private['probes'].items()}
        scores, scored_rows = {}, []
        for row, record in zip(plan['requests'], records):
            if row['kind'] != 'probe':
                continue
            variant, condition, iid = row['variant'], row['score_condition'], row['instance_id']
            p = probes[iid]
            if variant in plan['permutations']:
                p = remap_probe(p, plan['permutations'][variant][iid]['old_to_new'])
            score = score_response(p, record['response']['text'], condition=condition) if record is not None and record['status'] == 'completed' else None
            scores.setdefault(variant, {}).setdefault(condition, {})[iid] = score
            scored_rows.append({'instance_id': iid, 'problem_id': p.problem_id, 'variant': variant,
                                'condition': condition, 'C': None if score is None else score.C,
                                'B': None if score is None else score.B, 'J': None if score is None else score.J,
                                'canonical_binding': None if score is None else score.canonical_binding,
                                'field_valid': None if score is None else dict(score.field_valid)})
        reports, comparisons = {}, {}
        all_probes = list(probes.values())
        for variant, condition_scores in scores.items():
            ps = [probes[iid] for iid in condition_scores['raw']]
            if variant in plan['permutations']:
                ps = [remap_probe(p, plan['permutations'][variant][p.probe_id]['old_to_new']) for p in ps]
            reports[variant] = _condition_report(plan, ps, condition_scores, plan['weights'][variant])
            if variant != 'main':
                reference = scores['permutation_reference'] if variant in plan['permutations'] else scores['main']
                comparisons[variant] = {}
                for condition, after in condition_scores.items():
                    subset = [probes[iid] for iid in after]
                    weights = plan['weights'][variant]['paired' if condition == 'gold' else 'full']['binding']
                    comparisons[variant][condition] = _comparison(plan, subset, reference[condition], after, weights)
        nearest = None
        if plan['nearest_ids']:
            ps = [probes[i] for i in plan['nearest_ids']]
            heuristic = {p.probe_id: score_response(p, private['nearest'][p.probe_id]['response']) for p in ps}
            nearest = _comparison(plan, ps, scores['main']['raw'], heuristic, plan['weights']['nearest']['binding'])
            nearest['interpretation'] = 'Binding-only heuristic on the fixed ownership subset; no synthetic recognition score'
            nearest['per_probe'] = private['nearest']
            pool = metric_plan(all_probes, plan['weights']['main']['full']).binding_pool
            nearest['weighted_full_pool_coverage'] = pool.estimate({p.probe_id: int(p.probe_id in private['nearest']) for p in all_probes})['value']
        random_reference = {}
        for kind in sorted({p.probe_type for p in all_probes}):
            ps = tuple(p for p in all_probes if p.probe_type == kind)
            random_reference[kind] = (uniform_candidate_baseline(Pool(ps)) if all(p.binding.kind == 'single' for p in ps)
                                      else {'value': None, 'reason': 'No preregistered set/relation distribution'})
        random_full = (uniform_candidate_baseline(metric_plan(all_probes, plan['weights']['main']['full']).binding_pool)
                       if all(p.binding.kind == 'single' for p in all_probes)
                       else {'value': None, 'reason': 'No preregistered set/relation distribution; full pool preserved'})
        manual, grade_hash = _manual_grades(grades_path, records, private, manifest)
        solving = _solving_report(plan, all_probes, scores['main']['raw'], records, private, manual)
        strata = {}
        for name in ('candidate_count', 'panel_count', 'nearest_distance_band'):
            values = sorted({str(s[name]) for s in plan['strata'].values()})
            strata[name] = {}
            for value in values:
                ps = tuple(p for p in all_probes if str(plan['strata'][p.probe_id][name]) == value)
                strata[name][value] = Pool(ps).estimate({p.probe_id: None if scores['main']['raw'][p.probe_id] is None else scores['main']['raw'][p.probe_id].B for p in ps})
        seal_hash = manifest.get('protocol_metadata', {}).get('study_seal_hash')
        human = verify_final_seal(study)['human_report'] if seal_hash is not None else None
        human_comparison = None
        if human is not None:
            human_comparison = {}
            for condition in ('raw', 'gold'):
                ps = [p for p in all_probes if p.problem_id in plan['human_mothers'] and p.probe_id in scores['main'].get(condition, {})]
                estimates = {}
                for name, attribute in (('BAcc', 'B'), ('CAcc', 'C'), ('JAcc', 'J')):
                    if condition == 'gold' and attribute != 'B':
                        continue
                    eligible = ps if attribute == 'B' else [p for p in ps if p.joint_eligible]
                    model = Pool(tuple(eligible)).estimate({p.probe_id: None if scores['main'][condition][p.probe_id] is None else
                                                           getattr(scores['main'][condition][p.probe_id], attribute) for p in eligible})
                    h = human['summaries'][condition][name] if human['summaries'][condition] else None
                    estimates[name] = {'model': model, 'human': h, 'human_minus_model':
                        h['value'] - model['value'] if h is not None and h['value'] is not None and model['value'] is not None else None}
                human_comparison[condition] = estimates
        readiness = {**plan['readiness'], 'human_audit': 'sealed_before_inference' if human else 'not_measured_before_inference'}
        report = {'schema_version': 'physalign_study_report_v1', 'plan_hash': plan['plan_hash'], 'run_id': manifest['run_id'],
                  'adapter': manifest['adapter'], 'scientific_run': manifest['adapter']['scientific_run'],
                  'experiments': reports, 'control_comparisons': comparisons,
                  'uniform_random_by_type': random_reference, 'uniform_random_full': random_full, 'nearest_region': nearest,
                  'original_solving': solving, 'strata': strata, 'readiness': readiness,
                  'human_same_interface': human, 'study_seal_hash': seal_hash,
                  'human_model_same_subset': human_comparison,
                  'manual_grades_hash': grade_hash, 'service': {'planned': len(records),
                  'completed': sum(r is not None and r['status'] == 'completed' for r in records),
                  'infrastructure_missing': sum(r is not None and r['status'] == 'infrastructure_missing' for r in records),
                  'pending': sum(r is None for r in records)}}
        atomic_json(run / 'study_report.json', report)
        atomic_text(run / 'study_scores.jsonl', ''.join(canonical(r) + '\n' for r in scored_rows))
        return report


def export_grading_queue(study_root, run_directory, output):
    study = Study(study_root)
    private = study.private()
    with run_lock(Path(run_directory)):
        manifest, records = audited_records(study, run_directory)
        tasks = []
        for row, r in zip(study.plan['requests'], records):
            key = private['original_keys'].get(row['problem_id'])
            if row['kind'] != 'solve' or key is None or key['kind'] != 'human' or r['status'] != 'completed':
                continue
            tasks.append({'problem_id': row['problem_id'], 'request_id': r['request_id'], 'input': row['input'],
                          'response': r['response'], 'response_hash': fingerprint(r['response']),
                          'reference': key, 'reference_hash': fingerprint(key), 'correct': None,
                          'grader_id': '', 'human_confirmed': False, 'rationale': ''})
        from .storage import write_new
        result = {'schema_version': 'physalign_solve_grades_v1', 'plan_hash': study.plan['plan_hash'],
                  'run_id': manifest['run_id'], 'grades': tasks}
        write_new(Path(output), result)
        return result
