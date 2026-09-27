"""Isolated, reference-based LLM judging with durable first-response journals."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physalign.adapters import ModelRequest
from physalign.dataset import require
from physalign.runner import collect_records, request_id, run_requests, validate_manifest
from physalign.storage import canonical, file_hash, fingerprint, loads, read_json, write_new
from server_eval.api_adapter import APIAdapter
from server_eval.supplement_eval import Supplement
from server_eval.supplement_reporting import audited

SCHEMA = 'physalign_llm_judge_plan_v1'
PROMPT = Path(__file__).with_name('llm_judge_prompt.txt')


def implementation_hashes():
    return {name: file_hash(Path(__file__).with_name(name)) for name in
            ('llm_judge.py', 'llm_judge_reporting.py', 'llm_judge_prompt.txt', 'api_adapter.py')}


def separate_output(output, *roots):
    path = Path(output).resolve()
    for root in roots:
        root = Path(root).resolve()
        require(not path.is_relative_to(root) and not root.is_relative_to(path),
                'Judge output overlaps an input dataset or evaluation run')
    return path


def reference_value(key):
    # Synthetic/custom human keys may be plain text; upstream keys contain JSON.
    try:
        return loads(key['reference_answer'])
    except ValueError:
        return {'reference_answer': key['reference_answer']}


def make_plan(study, target_run, config, *, interval_seconds=2):
    require(type(interval_seconds) in (int, float) and 0 <= interval_seconds <= 60, 'Invalid call interval')
    # Keep credentials in the existing process-local environment loader only.
    from physalign.adapters import _no_credentials
    _no_credentials(config)
    manifest, records = audited(study, target_run)
    private, rows = study.private(), []
    eligible, unavailable = [], []
    prompt = PROMPT.read_text(encoding='utf-8').strip()
    for row, record in zip(study.plan['requests'], records):
        key = private['original_keys'].get(row['problem_id'])
        if row['kind'] != 'solve' or key is None or key['kind'] != 'human':
            continue
        mother = row['problem_id']
        eligible.append(mother)
        if record is None or record['status'] != 'completed':
            unavailable.append(mother)
            continue
        payload = {'original_question': row['input']['messages']['user'],
                   'reference': reference_value(key), 'candidate_response': record['response']['text']}
        inp = {'messages': {'system': prompt, 'user': canonical(payload)},
               'attachments': row['input']['attachments']}
        rows.append({'instance_id': 'judge:' + mother, 'logical_probe_id': 'judge:' + mother,
                     'problem_id': mother, 'condition': 'judge.solve', 'input': inp, 'input_hash': fingerprint(inp),
                     'target_request_id': record['request_id'], 'target_response_hash': fingerprint(record['response']),
                     'reference_hash': fingerprint(key)})
    policy = {'schema_version': 'physalign_llm_judge_protocol_v1', 'prompt': prompt, 'config': config,
              'code_hashes': implementation_hashes(), 'interval_seconds': interval_seconds,
              'retry_policy': {'max_retries': 2},
              'grading': 'reference_equivalence_all_parts_binary_v1', 'images': 'all_original_question_images',
              'blinding': 'target_model_identity_not_supplied',
              'selection': 'all_served_originals_with_human_contract; automatic_contracts_unchanged',
              'parse_policy': 'stop_and_exact_verdict_rationale_json; invalid_or_ungradable_is_missing',
              'resampling': 'preserve_first_response; retry_only_explicit_retryable_infrastructure_errors'}
    payload = {'schema_version': SCHEMA, 'supplement_plan_hash': study.plan['plan_hash'],
               'target_run_id': manifest['run_id'], 'target_definition_hash': manifest['definition_hash'],
               'protocol': policy, 'protocol_hash': fingerprint(policy),
               'eligible_problem_ids': eligible, 'unavailable_target_problem_ids': unavailable,
               'retry_policy': policy['retry_policy'], 'requests': rows}
    return {**payload, 'plan_hash': fingerprint(payload)}


def prepare(study, target_run, config, output, *, interval_seconds=2):
    output = separate_output(output, study.root, study.source, target_run)
    plan = make_plan(study, target_run, config, interval_seconds=interval_seconds)
    write_new(output, plan)
    return {'plan_hash': plan['plan_hash'], 'eligible': len(plan['eligible_problem_ids']),
            'judge_requests': len(plan['requests']), 'target_unavailable': len(plan['unavailable_target_problem_ids'])}


def validate_plan(study, target_run, plan):
    require(plan.get('schema_version') == SCHEMA, 'Not an LLM judge plan')
    require(plan['plan_hash'] == fingerprint({k: v for k, v in plan.items() if k != 'plan_hash'}), 'Judge plan changed')
    expected = make_plan(study, target_run, plan['protocol']['config'],
                         interval_seconds=plan['protocol']['interval_seconds'])
    require(plan == expected, 'Judge plan, source responses, reference answers or implementation changed')


class PacedJudge:
    """One in-flight request, no shared process/environment changes."""
    def __init__(self, adapter, interval):
        self.adapter, self.info, self.interval = adapter, adapter.info, interval
        self.last_end = None

    def generate(self, request):
        if self.last_end is not None:
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_end)))
        try:
            return self.adapter.generate(request)
        finally:
            self.last_end = time.monotonic()


def run(study, target_run, plan, output, *, resume=False, adapter=None, progress=None):
    output = separate_output(output, study.root, study.source, target_run)
    validate_plan(study, target_run, plan)
    config = plan['protocol']['config']
    adapter = adapter or APIAdapter(config)
    wrapped = PacedJudge(adapter, plan['protocol']['interval_seconds'])
    return run_requests(plan, wrapped, output,
        lambda row, rid, settings: study.request(row, rid, settings), source_root=study.root,
        resume=resume, progress=progress, adapter_configuration_hash=fingerprint(config),
        protocol_metadata={'grading_method': 'llm_judge', 'judge_protocol_hash': plan['protocol_hash'],
                           'target_run_id': plan['target_run_id']})


def parse_judgment(response):
    """Never convert a judge failure/refusal/ambiguous grade to a wrong answer."""
    if response.get('finish_reason') != 'stop':
        return None, 'judge_non_stop', ''
    try:
        value = loads(response['text'])
    except (ValueError, TypeError, KeyError):
        return None, 'judge_invalid_json', ''
    if (not isinstance(value, dict) or set(value) != {'verdict', 'rationale'}
            or value['verdict'] not in ('correct', 'incorrect', 'ungradable')
            or not isinstance(value['rationale'], str) or not value['rationale'].strip()):
        return None, 'judge_invalid_schema', ''
    return {'correct': 1, 'incorrect': 0, 'ungradable': None}[value['verdict']], value['verdict'], value['rationale']


def collect_grades(study, target_run, judge_run, *, allow_incomplete=False):
    judge_run = Path(judge_run)
    manifest = read_json(judge_run / 'manifest.json')
    validate_manifest(manifest)
    plan = manifest['plan']
    validate_plan(study, target_run, plan)
    # Prevent a run using a different real adapter from claiming this protocol.
    config, settings = plan['protocol']['config'], manifest['adapter']['settings']
    if manifest['adapter']['scientific_run']:
        require(manifest['adapter']['model_id'] == config['model_id'], 'Unexpected judge model')
        for name in ('reasoning_effort', 'max_output_tokens', 'token_limit_field', 'temperature', 'timeout_seconds', 'retry_delay_seconds'):
            require(settings.get(name) == config[name], 'Judge generation setting changed: ' + name)
        require(settings.get('base_url') == config['base_url'].rstrip('/'), 'Judge gateway changed')
        require(manifest['adapter']['preprocessing'].get('image_detail') == config['image_detail'], 'Judge image setting changed')
    records = collect_records(judge_run, manifest, allow_incomplete=allow_incomplete)
    grades, details, versions = {}, [], Counter()
    for row, record in zip(plan['requests'], records):
        mother = row['problem_id']
        correct, status, rationale = None, 'pending', ''
        if record is not None:
            rid = request_id(manifest, row)
            snapshot = study.snapshot(row['input'], rid, settings)
            require(read_json(judge_run / 'requests' / (rid + '.json')) == snapshot
                    and record['request_hash'] == fingerprint(snapshot), 'Judge request differs from frozen prompt/evidence')
            status = 'judge_infrastructure_missing'
            if record['status'] == 'completed':
                correct, status, rationale = parse_judgment(record['response'])
                versions[record['response'].get('returned_model_version') or 'unreported'] += 1
        if correct is not None:
            grades[mother] = correct
        details.append({'problem_id': mother, 'target_request_id': row['target_request_id'],
                        'target_response_hash': row['target_response_hash'], 'reference_hash': row['reference_hash'],
                        'judge_request_id': record['request_id'] if record else None,
                        'judge_response_hash': fingerprint(record['response']) if record and record['status']=='completed' else None,
                        'correct': correct, 'status': status, 'rationale': rationale})
    return grades, {'schema_version': 'physalign_llm_judgments_v1', 'grading_method': 'llm_judge',
                    'target_run_id': plan['target_run_id'], 'judge_run_id': manifest['run_id'],
                    'judge_definition_hash': manifest['definition_hash'], 'protocol_hash': plan['protocol_hash'],
                    'protocol': plan['protocol'], 'judge_adapter': manifest['adapter'],
                    'returned_model_versions': dict(versions), 'eligible': len(plan['eligible_problem_ids']),
                    'target_unavailable': plan['unavailable_target_problem_ids'], 'graded': len(grades),
                    'status_counts': dict(Counter(d['status'] for d in details)), 'judgments': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'run', 'score'):
        p = commands.add_parser(command)
        p.add_argument('--study', required=True)
        p.add_argument('--source')
        p.add_argument('--target-run', required=True)
        p.add_argument('--output', required=True)
        if command == 'prepare':
            p.add_argument('--config', default=str(ROOT / 'configs/judges/gpt56-sol-medium.json'))
            p.add_argument('--interval-seconds', type=float, default=2)
        elif command == 'run':
            p.add_argument('--plan', required=True)
            p.add_argument('--resume', action='store_true')
            p.add_argument('--dry-run', action='store_true')
        else:
            p.add_argument('--judge-run', required=True)
            p.add_argument('--main-run')
            p.add_argument('--allow-incomplete', action='store_true')
    p = commands.add_parser('compare')
    p.add_argument('--report', action='append', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('check')
    p.add_argument('--config', default=str(ROOT / 'configs/judges/gpt56-sol-medium.json'))
    p.add_argument('--generate', action='store_true', help='One paid synthetic connection/JSON test; no dataset used')
    p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    log = lambda value: print(canonical(value), flush=True)
    try:
        if args.command == 'compare':
            from server_eval.llm_judge_reporting import compare
            result = compare(args.report, args.output)
        elif args.command == 'check':
            require(not Path(args.output).exists(), 'Connection check output exists')
            config = read_json(Path(args.config))
            adapter = APIAdapter(config)
            require(config['model_id'] in adapter.models(), 'Requested judge model absent from gateway model list')
            result = {'model_id': config['model_id'], 'reasoning_effort': config['reasoning_effort'], 'model_list_confirmed': True,
                      'paid_requests': 0, 'config_hash': fingerprint(config)}
            if args.generate:
                request = ModelRequest('synthetic-judge-check', PROMPT.read_text(encoding='utf-8').strip(),
                    canonical({'original_question': 'A body travels at 2 m/s for 3 s. Find the distance in metres.',
                               'reference': {'final_answer': '6 m'},
                               'candidate_response': '{"answer":"6 m","explanation":"distance = speed times time = 6 m"}'}),
                    (), adapter.info.settings_json)
                response = adapter.generate(request)
                value = {'text': response.text, 'finish_reason': response.finish_reason}
                grade, status, rationale = parse_judgment(value)
                result.update(paid_requests=1, passed=grade == 1, status=status, rationale=rationale,
                              response={'text': response.text, 'finish_reason': response.finish_reason,
                                        'returned_model_version': response.returned_model_version,
                                        'usage': loads(response.usage_json), 'generation': loads(response.generation_json)})
            write_new(Path(args.output), result)
            log({k: v for k, v in result.items() if k != 'response'})
            return 0 if result.get('passed', True) else 2
        else:
            study = Supplement(args.study, args.source)
            if args.command == 'prepare':
                result = prepare(study, args.target_run, read_json(Path(args.config)), args.output,
                                 interval_seconds=args.interval_seconds)
            elif args.command == 'run':
                plan = read_json(Path(args.plan))
                if args.dry_run:
                    validate_plan(study, args.target_run, plan)
                    for row in plan['requests']:
                        study.images(row)
                    result = {'judge_requests': len(plan['requests']), 'paid_requests': 0, 'images_verified': True}
                else:
                    result = run(study, args.target_run, plan, args.output, resume=args.resume, progress=log)
            else:
                from server_eval.llm_judge_reporting import score
                report = score(study, args.target_run, args.judge_run, args.output,
                               main_run=args.main_run, allow_incomplete=args.allow_incomplete)
                result = {'output': args.output, 'SolveAcc': report['original_solving']['SolveAcc'],
                          'grading_method': report['grading']['method'], 'llm_graded': report['grading']['graded'],
                          'ungraded': len(report['original_solving']['missing_or_ungraded'])}
        log(result)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print('LLM judge: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
