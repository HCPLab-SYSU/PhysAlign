"""Matched cross-model comparisons, resampling the SAME source clusters."""

from itertools import combinations
from pathlib import Path

from .contracts import decode_probe
from .dataset import require
from .metrics import evaluate
from .planning import metric_plan
from .scoring import score_response
from .storage import atomic_json, fingerprint, read_json, run_lock
from .study import Study
from .study_reporting import _bootstrap, audited_records


def panel_report(study_root, run_directories, output):
    study = Study(study_root)
    probes = {iid: decode_probe(row) for iid, row in study.private()['probes'].items()}
    all_probes = list(probes.values())
    raw_spec = metric_plan(all_probes, study.plan['weights']['main']['full'])
    pairs = set(study.plan['groups']['main'].get('gold', []))
    paired_probes = [p for p in all_probes if p.probe_id in pairs]
    paired_spec = metric_plan(paired_probes, study.plan['weights']['main']['paired']) if pairs else None
    models, scores, configs = {}, {}, []
    for directory in run_directories:
        directory = Path(directory)
        with run_lock(directory):
            manifest, records = audited_records(study, directory)
            name = manifest['adapter']['model_id']
            require(name not in models, 'Duplicate model ID in panel; report ablations separately')
            settings = manifest['adapter']['settings']
            configs.append({k: settings.get(k) for k in ('max_new_tokens', 'max_input_tokens', 'thinking', 'do_sample', 'num_beams', 'seed')})
            raw, gold = {}, {}
            for row, r in zip(study.plan['requests'], records):
                if row['variant'] != 'main':
                    continue
                iid, condition = row['instance_id'], row['score_condition']
                value = score_response(probes[iid], r['response']['text'], condition=condition) if r['status'] == 'completed' else None
                (raw if condition == 'raw' else gold)[iid] = value
            scores[name] = raw, gold
            models[name] = {'run_id': manifest['run_id'], 'adapter': manifest['adapter'],
                            'raw': evaluate(raw_spec, raw),
                            'paired': evaluate(paired_spec, {i: raw[i] for i in pairs}, gold) if pairs else None}
    require(len(models) >= 2, 'Panel comparison requires at least two models')
    require(all(c == configs[0] for c in configs), 'Panel mixes sampling/thinking/token budgets; split it into separate protocols')
    comparisons = {}
    for a, b in combinations(models, 2):
        def stat(counts=None):
            if counts is not None:
                counts = {**dict.fromkeys({p.problem_id for p in all_probes}, 0), **counts}
            first = evaluate(raw_spec, scores[a][0], problem_multiplicities=counts)['metrics']
            second = evaluate(raw_spec, scores[b][0], problem_multiplicities=counts)['metrics']
            result = {k: second[k] - first[k] if first[k] is not None and second[k] is not None else None
                      for k in ('CAcc', 'BAcc', 'JAcc', 'BAcc_given_C')}
            if pairs:
                ga = evaluate(paired_spec, {i: scores[a][0][i] for i in pairs}, scores[a][1], problem_multiplicities=counts)['metrics']
                gb = evaluate(paired_spec, {i: scores[b][0][i] for i in pairs}, scores[b][1], problem_multiplicities=counts)['metrics']
                for key in ('BAcc_gold', 'delta_BAcc_gold_raw'):
                    result['paired_' + key] = gb[key] - ga[key] if ga[key] is not None and gb[key] is not None else None
            return result
        comparisons[b + ' minus ' + a] = {'differences': stat(),
                    'intervals': _bootstrap(study.plan, {p.problem_id for p in all_probes},
                        lambda c: {k: v for k, v in stat(c).items() if not k.startswith('paired_')}),
                    'paired_intervals': _bootstrap(study.plan, {p.problem_id for p in paired_probes},
                        lambda c: {k: v for k, v in stat(c).items() if k.startswith('paired_')}) if pairs else None}
    result = {'schema_version': 'physalign_panel_report_v1', 'plan_hash': study.plan['plan_hash'],
              'models': models, 'comparisons': comparisons, 'shared_settings': configs[0],
              'requested_panel_complete': set(models) == set(study.plan['model_panel'])}
    atomic_json(Path(output), result)
    return result
