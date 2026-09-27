"""Read-only run audits and offline supplement reports; existing runs untouched."""
from collections import Counter
import csv
from pathlib import Path

from physalign.contracts import decode_probe
from physalign.dataset import require
from physalign.metrics import Pool, uniform_candidate_baseline
from physalign.planning import metric_plan, weights_for
from physalign.runner import collect_records, request_id, validate_manifest
from physalign.scoring import score_response
from physalign.storage import canonical, file_hash, fingerprint, read_json, write_new
from physalign.study_reporting import (_bootstrap, _comparison, _condition_report, _manual_grades,
                                      _solving_report)


def audited(study, run, *, main=False, allow_incomplete=False):
    run = Path(run)
    manifest = read_json(run / 'manifest.json')
    validate_manifest(manifest)
    plan = study.plan['base_plan'] if main else study.plan
    require(manifest['plan'] == plan, 'Run uses a different frozen plan')
    records = collect_records(run, manifest, allow_incomplete=allow_incomplete)
    inputs = {(r['instance_id'], r['condition']): r['input'] for r in
              (study.plan['reference_inputs'] if main else plan['requests'])}
    for row, record in zip(plan['requests'], records):
        if record is None:
            continue
        rid = request_id(manifest, row)
        snapshot = study.snapshot(inputs[row['instance_id'], row['condition']], rid, manifest['adapter']['settings'])
        require(read_json(run / 'requests' / (rid + '.json')) == snapshot
                and record['request_hash'] == fingerprint(snapshot), 'Logged public request differs from frozen input')
    # The terminal attempt journals, not a mutable report or a half-written
    # predictions export, are authoritative. This function never takes a run lock.
    return manifest, records


def _output_directory(output, *protected):
    path = Path(output).resolve()
    for parent in protected:
        parent = Path(parent).resolve()
        require(not path.is_relative_to(parent) and not parent.is_relative_to(path), 'Output overlaps a protected input/run')
    require(not path.exists(), 'Output exists; use a new report directory')
    path.mkdir(parents=True)
    return path


def grading_queue(study, run, output):
    require(not Path(output).exists(), 'Grade queue output exists')
    from physalign.storage import outside_bundle
    for parent in (study.root, study.source, Path(run)):
        outside_bundle(Path(output), parent)
    manifest, records = audited(study, run, allow_incomplete=True)
    private, tasks = study.private(), []
    for row, record in zip(study.plan['requests'], records):
        key = private['original_keys'].get(row['problem_id'])
        if row['kind'] != 'solve' or key is None or key['kind'] != 'human' or record is None or record['status'] != 'completed':
            continue
        tasks.append({'problem_id': row['problem_id'], 'request_id': record['request_id'],
                      'input': row['input'], 'response': record['response'],
                      'response_hash': fingerprint(record['response']), 'reference': key, 'reference_hash': fingerprint(key),
                      'correct': None, 'grader_id': '', 'human_confirmed': False, 'rationale': ''})
    artifact = {'schema_version': 'physalign_solve_grades_v1', 'plan_hash': study.plan['plan_hash'],
                'run_id': manifest['run_id'], 'grades': tasks}
    write_new(Path(output), artifact)
    return {'grading_tasks': len(tasks), 'output': str(output)}


def score(study, run, output, *, main_run=None, grades_path=None, allow_incomplete=False):
    manifest, records = audited(study, run, allow_incomplete=allow_incomplete)
    plan, private = study.plan, study.private()
    probes = {i: decode_probe(p) for i, p in private['probes'].items()}
    all_probes = list(probes.values())
    scores, detail = {}, []
    raw = dict.fromkeys(probes)
    main_manifest, main_records = None, []
    if main_run:
        main_manifest, main_records = audited(study, main_run, main=True, allow_incomplete=allow_incomplete)
        # Same model AND inference configuration is necessary for causal control deltas.
        require(manifest['adapter'] == main_manifest['adapter'], 'Main/supplement model or settings differ; use the same frozen adapter config')
        for row, record in zip(plan['base_plan']['requests'], main_records):
            p = probes[row['instance_id']]
            value = score_response(p, record['response']['text'], condition=row['condition']) if record is not None and record['status'] == 'completed' else None
            scores.setdefault('main', {}).setdefault(row['condition'], {})[p.probe_id] = value
        raw = scores['main']['raw']
    for row, record in zip(plan['requests'], records):
        if row['kind'] != 'probe':
            continue
        iid, condition, variant = row['instance_id'], row['score_condition'], row['variant']
        value = score_response(probes[iid], record['response']['text'], condition=condition) if record is not None and record['status'] == 'completed' else None
        scores.setdefault(variant, {}).setdefault(condition, {})[iid] = value
        detail.append({'instance_id': iid, 'problem_id': row['problem_id'], 'variant': variant, 'condition': condition,
                       'C': value.C if value else None, 'B': value.B if value else None, 'J': value.J if value else None,
                       'field_valid': dict(value.field_valid) if value else None,
                       'canonical_binding': value.canonical_binding if value else None})
    reports, comparisons = {}, {}
    for variant, conditions in scores.items():
        reports[variant] = _condition_report(plan, all_probes, conditions, plan['weights']['main'])
        if variant != 'main' and main_manifest:
            comparisons[variant] = {condition: _comparison(plan, [probes[i] for i in after],
                scores['main'][condition], after, plan['weights']['main']['paired' if condition == 'gold' else 'full']['binding'])
                for condition, after in conditions.items()}
    pool = metric_plan(all_probes, plan['weights']['main']['full']).binding_pool
    random = uniform_candidate_baseline(pool)
    nearest = None
    if private['nearest']:
        subset = [probes[i] for i in private['nearest']]
        heuristic = {p.probe_id: score_response(p, private['nearest'][p.probe_id]['response']) for p in subset}
        weights = weights_for(subset)['binding']
        nearest = {'BAcc': Pool(tuple(subset), weights).estimate({i: v.B for i, v in heuristic.items()}),
                   'eligible_probes': len(subset), 'total_probes': len(probes),
                   'weighted_full_pool_coverage': pool.estimate({i: int(i in heuristic) for i in probes})['value'],
                   'exclusion_counts': dict(Counter(plan['nearest_exclusions'].values())),
                   'per_probe': private['nearest'], 'comparison': None}
        if main_manifest:
            nearest['comparison'] = _comparison(plan, subset, raw, heuristic, weights)
    manual, grade_hash = _manual_grades(grades_path, records, private, manifest)
    solving = _solving_report(plan, all_probes, raw, records, private, manual)
    solving['scheduled'] = 'solve' in plan['experiments']
    solving['total_selected_mothers'] = len(private['original_keys'])
    solving['answer_key_coverage'] = solving['answer_eligible_mothers'] / len(private['original_keys'])
    answers = solving['answer_correctness']
    automatic = {m: value for m, value in answers.items() if private['original_keys'][m]['kind'] != 'human'}
    auto_missing = sum(v is None for v in automatic.values())
    auto_sum = sum(v for v in automatic.values() if v is not None)
    solving['automatic_subset'] = {'eligible_mothers': len(automatic), 'missing': auto_missing,
        'correct': auto_sum, 'SolveAcc': auto_sum / len(automatic) if automatic and not auto_missing else None,
        'interpretation': 'Only predeclared automatic grading contracts; not the full original-task accuracy'}
    if answers and all(v is not None for v in answers.values()):
        solving['accuracy_intervals'] = _bootstrap(plan, set(answers), lambda counts: {
            'SolveAcc': sum(counts[m] * a for m, a in answers.items()) / sum(counts.values())})
    else:
        solving['accuracy_intervals'] = None
    by_source = {}
    for mother, src in private['original_sources'].items():
        name = src['source_dataset']
        by_source.setdefault(name, {'mothers': 0, 'answer_eligible': 0, 'graded': 0, 'correct': 0})
        by_source[name]['mothers'] += 1
        if mother in answers:
            by_source[name]['answer_eligible'] += 1
            if answers[mother] is not None:
                by_source[name]['graded'] += 1
                by_source[name]['correct'] += answers[mother]
    for x in by_source.values():
        x['SolveAcc'] = x['correct'] / x['answer_eligible'] if x['answer_eligible'] and x['graded'] == x['answer_eligible'] else None
    solving['by_source'] = by_source
    report = {'schema_version': 'physalign_solve_controls_report_v1', 'plan_hash': plan['plan_hash'],
              'base_plan_hash': plan['base_plan']['plan_hash'], 'private_hash': plan['private_hash'],
              'run_id': manifest['run_id'], 'adapter': manifest['adapter'],
              'scientific_run': manifest['adapter']['scientific_run'],
              'main_run_id': main_manifest['run_id'] if main_manifest else None,
              'main_definition_hash': main_manifest['definition_hash'] if main_manifest else None,
              'manual_grades_hash': grade_hash, 'experiments': reports, 'control_comparisons': comparisons,
              'uniform_random': random, 'nearest_region': nearest, 'original_solving': solving,
              'control_definitions': plan['control_definitions'], 'readiness': plan['readiness'],
              'service': {'planned': len(records), 'completed': sum(r is not None and r['status'] == 'completed' for r in records),
                          'pending': sum(r is None for r in records),
                          'infrastructure_missing': sum(r is not None and r['status'] != 'completed' for r in records),
                          'finish_reasons': dict(Counter(r['response'].get('finish_reason') for r in records if r is not None and r['status'] == 'completed'))}}
    protected = [study.root, study.source, run] + ([main_run] if main_run else [])
    destination = _output_directory(output, *protected)
    write_new(destination / 'report.json', report)
    (destination / 'scores.jsonl').write_text(''.join(canonical(r) + '\n' for r in detail), encoding='utf-8')
    with (destination / 'solving_by_mother.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problem_id', 'source_dataset', 'answer_eligible', 'correct', 'mean_binding'])
        binding = (solving.get('association') or {}).get('binding_by_problem', {})
        for m, src in private['original_sources'].items():
            writer.writerow([m, src['source_dataset'], int(m in answers), answers.get(m), binding.get(m)])
    return report


def compare(paths, output):
    reports = [read_json(Path(p)) for p in paths]
    require(reports and all(r.get('schema_version') == 'physalign_solve_controls_report_v1' for r in reports), 'Not supplement reports')
    require(len({r['plan_hash'] for r in reports}) == 1, 'Compare requires one frozen supplement plan/denominator')
    require(len({r['run_id'] for r in reports}) == len(reports), 'Duplicate run')
    require(len({r['scientific_run'] for r in reports}) == 1, 'Cannot mix smoke and real model reports')
    destination = _output_directory(output, *(Path(p).parent for p in paths))
    variants = sorted(set().union(*(r['experiments'] for r in reports)))
    columns = ['model', 'SolveAcc', 'answer_eligible', 'answer_key_coverage', 'mean_binding_correct', 'mean_binding_wrong',
               'uniform_random_BAcc', 'nearest_region_BAcc'] + [v + '_BAcc' for v in variants]
    rows = []
    for r in reports:
        s = r['original_solving']; assoc = s.get('association') or {}
        rows.append([r['adapter']['model_id'], s['SolveAcc'], s['answer_eligible_mothers'], s['answer_key_coverage'],
                     assoc.get('mean_binding_correct'), assoc.get('mean_binding_wrong'), r['uniform_random']['value'],
                     (r.get('nearest_region') or {}).get('BAcc', {}).get('value')]
                    + [r['experiments'].get(v, {}).get('raw_all', {}).get('metrics', {}).get('BAcc') for v in variants])
    with (destination / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f); writer.writerow(columns); writer.writerows(rows)
    def cell(x):
        return 'N/A' if x is None else f'{x:.4f}' if isinstance(x, float) else str(x)
    (destination / 'comparison.md').write_text('| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---'] * len(columns)) + '|\n'
        + ''.join('| ' + ' | '.join(map(cell, row)) + ' |\n' for row in rows), encoding='utf-8')
    write_new(destination / 'sources.json', [{'report': str(Path(p).resolve()), 'sha256': file_hash(Path(p)),
                                           'run_id': r['run_id'], 'adapter': r['adapter']} for p, r in zip(paths, reports)])
    # Optional plotting dependency stays separate from all inference paths.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'output': str(destination), 'table_rows': len(rows), 'plot': 'matplotlib not installed; CSV/Markdown saved'}
    import math
    panels = [('SolveAcc', 1)] + [(v + ' BAcc', 8 + i) for i, v in enumerate(variants)]
    fig, axes = plt.subplots(math.ceil(len(panels) / 2), 2, figsize=(13, 4 * math.ceil(len(panels) / 2)), squeeze=False)
    for ax, (title, index) in zip(axes.flat, panels):
        values = [row[index] for row in rows]
        ax.bar(range(len(rows)), [v if v is not None else float('nan') for v in values])
        for i, value in enumerate(values):
            if value is None:
                ax.text(i, .02, 'N/A', ha='center')
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=25, ha='right')
        ax.set_ylim(0, 1); ax.set_title(title)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.savefig(destination / 'comparison.png', dpi=180); fig.savefig(destination / 'comparison.pdf'); plt.close(fig)
    return {'output': str(destination), 'table_rows': len(rows), 'plot': 'comparison.png/pdf'}
