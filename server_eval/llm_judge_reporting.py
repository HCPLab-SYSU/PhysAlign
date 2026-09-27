"""Reports that explicitly distinguish LLM judgments from human annotations."""
from collections import Counter
import csv
from pathlib import Path
import shutil

from physalign.contracts import decode_probe
from physalign.dataset import require
from physalign.scoring import score_response
from physalign.storage import canonical, file_hash, fingerprint, read_json, write_new
from physalign.study_reporting import _bootstrap, _solving_report
from server_eval.supplement_reporting import audited, _output_directory, score as deterministic_score
from server_eval.llm_judge import collect_grades

SCHEMA = 'physalign_solve_controls_llm_report_v1'


def score(study, target_run, judge_run, output, *, main_run=None, allow_incomplete=False):
    # All validation precedes report output. Inputs and existing reports stay read-only.
    grades, artifact = collect_grades(study, target_run, judge_run, allow_incomplete=allow_incomplete)
    manifest, records = audited(study, target_run, allow_incomplete=allow_incomplete)
    require(manifest['adapter']['scientific_run'] == artifact['judge_adapter']['scientific_run'],
            'Cannot use a smoke judge for real experiment results or vice versa')
    private = study.private()
    probes = {i: decode_probe(p) for i, p in private['probes'].items()}
    raw = dict.fromkeys(probes)
    if main_run:
        main_manifest, main_records = audited(study, main_run, main=True, allow_incomplete=allow_incomplete)
        require(manifest['adapter'] == main_manifest['adapter'], 'Main/supplement model or settings differ')
        for row, record in zip(study.plan['base_plan']['requests'], main_records):
            if row['condition'] == 'raw':
                raw[row['instance_id']] = (score_response(probes[row['instance_id']], record['response']['text'])
                    if record is not None and record['status'] == 'completed' else None)
    solving = _solving_report(study.plan, list(probes.values()), raw, records, private, grades)
    answers = solving['answer_correctness']
    if answers and all(v is not None for v in answers.values()):
        solving['accuracy_intervals'] = _bootstrap(study.plan, set(answers), lambda counts: {
            'SolveAcc': sum(counts[m] * a for m, a in answers.items()) / sum(counts.values())})
    else:
        solving['accuracy_intervals'] = None
    by_source = {}
    for mother, src in private['original_sources'].items():
        value = by_source.setdefault(src['source_dataset'], {'mothers': 0, 'answer_eligible': 0, 'graded': 0, 'correct': 0})
        value['mothers'] += 1
        if mother in answers:
            value['answer_eligible'] += 1
            if answers[mother] is not None:
                value['graded'] += 1
                value['correct'] += answers[mother]
    for value in by_source.values():
        value['SolveAcc'] = (value['correct'] / value['answer_eligible']
            if value['answer_eligible'] and value['graded'] == value['answer_eligible'] else None)
    solving['by_source'] = by_source
    known = [v for v in answers.values() if v is not None]
    solving['graded_subset'] = {'graded_mothers': len(known), 'correct': sum(known),
                               'accuracy': sum(known) / len(known) if known else None,
                               'interpretation': 'Descriptive only; does not replace full-cohort SolveAcc when grades are missing'}
    destination = _output_directory(output, study.root, study.source, target_run, judge_run,
                                     *([main_run] if main_run else []))
    # Reuse the frozen control implementation unchanged. Preserve this intermediate
    # deterministic report so the 160 automatic contracts can be independently audited.
    report = deterministic_score(study, target_run, destination / 'deterministic',
                                 main_run=main_run, allow_incomplete=allow_incomplete)
    report['original_solving'].update(solving)
    report['schema_version'] = SCHEMA
    report['grading'] = {'method': 'deterministic_contracts_plus_llm_judge',
                         'human_confirmed': False, 'protocol_hash': artifact['protocol_hash'],
                         'judge_adapter': artifact['judge_adapter'], 'judge_run_id': artifact['judge_run_id'],
                         'judge_definition_hash': artifact['judge_definition_hash'],
                         'judgments_hash': fingerprint(artifact), 'eligible': artifact['eligible'],
                         'graded': artifact['graded'], 'status_counts': artifact['status_counts'],
                         'returned_model_versions': artifact['returned_model_versions'],
                         'interpretation': 'Open-ended correctness is model-judged; not human-verified ground truth. All models must use this same protocol.'}
    report['llm_grades_hash'] = fingerprint(artifact)
    require(report['manual_grades_hash'] is None, 'LLM grades must not be labelled as human annotations')
    write_new(destination / 'judgments.json', artifact)
    shutil.copyfile(destination / 'deterministic/scores.jsonl', destination / 'scores.jsonl')
    with (destination / 'solving_by_mother.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['problem_id', 'source_dataset', 'answer_eligible', 'correct', 'grading_method', 'judge_status', 'mean_binding'])
        statuses = {r['problem_id']: r['status'] for r in artifact['judgments']}
        binding = (solving.get('association') or {}).get('binding_by_problem', {})
        for m, src in private['original_sources'].items():
            key = private['original_keys'][m]
            method = 'no_reference' if key is None else 'llm_judge' if key['kind'] == 'human' else 'deterministic'
            writer.writerow([m, src['source_dataset'], int(m in answers), answers.get(m), method,
                             statuses.get(m, 'target_unavailable' if method == 'llm_judge' else ''), binding.get(m)])
    # Write the final report last; an interrupted report job has no complete result.
    write_new(destination / 'report.json', report)
    return report


def compare(paths, output):
    reports = [read_json(Path(p)) for p in paths]
    require(reports and all(r.get('schema_version') == SCHEMA for r in reports), 'Use only LLM-judge supplement reports')
    for key in ('plan_hash', 'scientific_run'):
        require(len({r[key] for r in reports}) == 1, 'Incompatible reports: ' + key)
    require(len({r['run_id'] for r in reports}) == len(reports), 'Duplicate target run')
    require(len({r['grading']['protocol_hash'] for r in reports}) == 1, 'Judge models/settings/prompts/protocols differ')
    require(len({fingerprint(r['grading']['judge_adapter']) for r in reports}) == 1, 'Actual judge adapters differ')
    versions = set().union(*(r['grading']['returned_model_versions'] for r in reports))
    require(len(versions) <= 1, 'Returned judge model versions differ; do not silently pool them')
    destination = _output_directory(output, *(Path(p).parent for p in paths))
    variants = sorted(set().union(*(r['experiments'] for r in reports)))
    columns = ['model', 'SolveAcc_LLM_judged', 'SolveAcc_automatic_subset', 'answer_eligible', 'llm_graded',
               'ungraded', 'mean_binding_correct', 'mean_binding_wrong', 'uniform_random_BAcc', 'nearest_region_BAcc']
    columns += [v + '_BAcc' for v in variants]
    rows = []
    for r in reports:
        s = r['original_solving']; association = s.get('association') or {}
        rows.append([r['adapter']['model_id'], s['SolveAcc'], s['automatic_subset']['SolveAcc'],
                     s['answer_eligible_mothers'], r['grading']['graded'], len(s['missing_or_ungraded']),
                     association.get('mean_binding_correct'), association.get('mean_binding_wrong'),
                     r['uniform_random']['value'], (r.get('nearest_region') or {}).get('BAcc', {}).get('value')]
                    + [r['experiments'].get(v, {}).get('raw_all', {}).get('metrics', {}).get('BAcc') for v in variants])
    with (destination / 'comparison.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(columns); writer.writerows(rows)
    def cell(value):
        return 'N/A' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
    (destination / 'comparison.md').write_text(
        'Open-ended answers: LLM judge; choice answers: deterministic. Missing grades are not zeros.\n\n'
        + '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---'] * len(columns)) + '|\n'
        + ''.join('| ' + ' | '.join(map(cell, row)) + ' |\n' for row in rows), encoding='utf-8')
    write_new(destination / 'grading-protocol.json', {'protocol_hash': reports[0]['grading']['protocol_hash'],
              'judge_adapter': reports[0]['grading']['judge_adapter'], 'returned_versions': sorted(versions),
              'human_confirmed': False, 'sources': [{'path': str(Path(p).resolve()), 'sha256': file_hash(Path(p))} for p in paths]})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'output': str(destination), 'models': len(rows), 'plot': 'matplotlib missing; tables saved'}
    import math
    panels = [('SolveAcc (open-ended: LLM judge)', 1), ('SolveAcc (automatic subset)', 2)]
    panels += [(v + ' BAcc', 10 + i) for i, v in enumerate(variants)]
    fig, axes = plt.subplots(math.ceil(len(panels) / 2), 2, figsize=(13, 4 * math.ceil(len(panels) / 2)), squeeze=False)
    for ax, (title, index) in zip(axes.flat, panels):
        values = [r[index] for r in rows]
        ax.bar(range(len(rows)), [v if v is not None else float('nan') for v in values])
        for i, value in enumerate(values):
            if value is None:
                ax.text(i, .02, 'N/A', ha='center')
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=25, ha='right')
        ax.set_ylim(0, 1); ax.set_title(title)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout(); fig.savefig(destination / 'comparison.png', dpi=180); fig.savefig(destination / 'comparison.pdf'); plt.close(fig)
    return {'output': str(destination), 'models': len(rows), 'plot': 'comparison.png/pdf'}
