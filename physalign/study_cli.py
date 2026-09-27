"""Commands for the complete study; CPU preparation does not import model SDKs."""

from pathlib import Path

from .storage import canonical, fingerprint, read_json, write_new


def register(commands):
    p = commands.add_parser('draft-study', help='Export a study specification with missing answers explicitly null')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('prepare-study', help='Freeze main, solving and shortcut/control experiment inputs')
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--spec')
    p.add_argument('--bootstrap-resamples', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--max-retries', type=int, default=2)
    p.add_argument('--problem-id', action='append')
    p.add_argument('--reservation-file', action='append', default=[])
    p = commands.add_parser('run-study', help='Run every frozen study request using one loaded model')
    p.add_argument('--study', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--adapter', default='hf')
    p.add_argument('--adapter-config')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p = commands.add_parser('score-study', help='Score all five-experiment model outputs offline')
    p.add_argument('--study', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--grades')
    p.add_argument('--allow-incomplete', action='store_true')
    p = commands.add_parser('grading-queue', help='Export original-solving responses with frozen references for human grading')
    p.add_argument('--study', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('freeze-model', help='Hash an existing local HF snapshot; never downloads a model')
    p.add_argument('--model-path', required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('inspect-model-inputs', help='Check exact processor image/token shapes on CPU without loading weights')
    p.add_argument('--study', required=True)
    p.add_argument('--adapter-config', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('human-create', help='Assign a real participant to an independent, counterbalanced session')
    p.add_argument('--study', required=True)
    p.add_argument('--sessions', required=True)
    p.add_argument('--participant', required=True)
    p.add_argument('--mode', choices=['answer', 'audit', 'adjudicate'], default='answer')
    p.add_argument('--cohort', type=int, choices=[0, 1], default=0)
    p = commands.add_parser('human-serve', help='Serve only the assigned human interface on localhost')
    p.add_argument('--study', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--port', type=int, default=8765)
    p = commands.add_parser('human-report', help='Summarize actual human submissions, ambiguity and adjudication')
    p.add_argument('--study', required=True)
    p.add_argument('--sessions', required=True)
    p.add_argument('--output', required=True)
    p = commands.add_parser('seal-study', help='Freeze completed human audit evidence before the final model panel')
    p.add_argument('--study', required=True)
    p.add_argument('--sessions', required=True)
    p = commands.add_parser('panel-report', help='Compare the three runs on one frozen study with paired uncertainty')
    p.add_argument('--study', required=True)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--output', required=True)


COMMANDS = {'draft-study', 'prepare-study', 'run-study', 'score-study', 'grading-queue', 'freeze-model', 'inspect-model-inputs',
            'human-create', 'human-serve', 'human-report', 'seal-study', 'panel-report'}


def execute(args):
    from .study import Study, draft_spec, prepare_study, run_study, verify_final_seal
    from .study_reporting import score_study, export_grading_queue
    from .human import create_session, serve_session, human_report, seal_study
    if args.command == 'draft-study':
        result = draft_spec(args.dataset)
        write_new(Path(args.output), result)
        print(canonical({'spec': str(Path(args.output).resolve()), 'note': result['note']}))
    elif args.command == 'prepare-study':
        plan = prepare_study(args.dataset, args.output, split=args.split, spec_path=args.spec,
                             bootstrap_resamples=args.bootstrap_resamples, seed=args.seed,
                             max_retries=args.max_retries, problem_ids=args.problem_id,
                             reservation_files=args.reservation_file)
        print(canonical({'study': str(Path(args.output).resolve()), 'plan_hash': plan['plan_hash'],
                         'requests': len(plan['requests']), 'readiness': plan['readiness']}))
    elif args.command == 'run-study':
        study = Study(args.study)
        if args.dry_run:
            for row in study.plan['requests']:
                study.images(row)
            print(canonical({'requests': len(study.plan['requests']), 'model_loaded': False,
                             'readiness': study.plan['readiness']}))
            return 0
        verify_final_seal(study)
        from .cli import _adapter
        config = read_json(Path(args.adapter_config)) if args.adapter_config else {}
        adapter = _adapter(args.adapter, config)
        result = run_study(args.study, adapter, args.output, resume=args.resume,
                          configuration_hash=fingerprint(config),
                          progress=lambda x: print(canonical(x), flush=True))
        print(canonical(result))
    elif args.command == 'score-study':
        report = score_study(args.study, args.run, grades_path=args.grades, allow_incomplete=args.allow_incomplete)
        print(canonical({'report': str(Path(args.run).resolve() / 'study_report.json'),
                         'main': report['experiments']['main']['raw_all']['metrics'],
                         'SolveAcc': report['original_solving']['SolveAcc'], 'service': report['service']}))
    elif args.command == 'grading-queue':
        result = export_grading_queue(args.study, args.run, args.output)
        print(canonical({'grading_tasks': len(result['grades']), 'output': args.output}))
    elif args.command == 'freeze-model':
        from .hf_adapter import freeze_snapshot
        result = freeze_snapshot(args.model_path, args.model_id, args.output)
        print(canonical({'snapshot_hash': result['snapshot_hash'], 'files': len(result['files'])}))
    elif args.command == 'inspect-model-inputs':
        from .hf_adapter import inspect_study_inputs
        result = inspect_study_inputs(args.study, read_json(Path(args.adapter_config)), args.output,
                                      progress=lambda x: print(canonical(x), flush=True))
        print(canonical({'output': args.output, 'all_within_budget': result['all_within_budget'], 'model_loaded': False}))
        return 0 if result['all_within_budget'] else 1
    elif args.command == 'human-create':
        print(create_session(args.study, args.sessions, participant=args.participant, mode=args.mode, cohort=args.cohort))
    elif args.command == 'human-serve':
        serve_session(args.study, args.session, port=args.port)
    elif args.command == 'human-report':
        result = human_report(args.study, args.sessions, output=args.output)
        print(canonical({'answer_coverage': result['complete_answer_coverage'], 'audits_accepted': result['audit_complete_and_accepted']}))
    elif args.command == 'seal-study':
        print(canonical({'seal_hash': seal_study(args.study, args.sessions)['seal_hash']}))
    elif args.command == 'panel-report':
        from .panel import panel_report
        result = panel_report(args.study, args.run, args.output)
        print(canonical({'models': list(result['models']), 'output': args.output}))
    return 0
