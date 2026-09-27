"""Offline publication artifacts. This entry point never loads or calls a model."""

import argparse
from pathlib import Path
import sys

from physalign.dataset import require
from physalign.storage import canonical, read_json
from .data import load_bundle
from .export import export_bundle


def main(argv=None):
    p = argparse.ArgumentParser(description='PhysAlign publication figures and tables from audited five-experiment runs')
    p.add_argument('--study')
    p.add_argument('--run', action='append', default=[], help='Repeat for each model on the SAME frozen study')
    p.add_argument('--output', required=True)
    p.add_argument('--labels', help='Optional JSON mapping exact model IDs to display labels')
    p.add_argument('--allow-non-scientific', action='store_true')
    p.add_argument('--allow-incomplete', action='store_true')
    p.add_argument('--allow-model-specific-settings', action='store_true',
                   help='Compare API/local reasoning and budgets with explicit per-model protocol disclosure')
    p.add_argument('--no-diagnostic-ci', action='store_true', help='Do not compute additional exploratory-stratum/transition CIs')
    p.add_argument('--tables-only', action='store_true')
    p.add_argument('--width', type=float, default=5.5, help='Physical figure width in inches')
    p.add_argument('--dpi', type=int, default=450)
    p.add_argument('--formats', nargs='+', choices=['pdf', 'svg', 'png'], default=['pdf', 'svg', 'png'])
    p.add_argument('--demo', action='store_true', help='Clearly labeled synthetic DESIGN PREVIEW, with no real model identities')
    p.add_argument('--demo-bootstrap-resamples', type=int, default=200)
    args = p.parse_args(argv)
    try:
        require(not Path(args.output).exists(), 'Output already exists; choose a new figure directory')
        if args.demo:
            require(not args.study and not args.run and not args.labels, '--demo cannot be mixed with real experiment inputs')
            require(args.demo_bootstrap_resamples == 0 or args.demo_bootstrap_resamples >= 2, 'Demo bootstrap count must be 0 or >=2')
            from .demo import demo_bundle
            bundle = demo_bundle(n_resamples=args.demo_bootstrap_resamples)
        else:
            require(args.study and args.run, 'Provide --study and at least one --run, or use --demo')
            out = Path(args.output).resolve()
            for source in [args.study] + args.run:
                require(not out.is_relative_to(Path(source).resolve()), 'Write figure artifacts outside frozen study/run directories')
            labels = read_json(Path(args.labels)) if args.labels else None
            require(labels is None or isinstance(labels, dict) and all(isinstance(k, str) and isinstance(v, str) and v for k, v in labels.items()), 'Labels must map model IDs to nonempty strings')
            bundle = load_bundle(args.study, args.run, allow_non_scientific=args.allow_non_scientific,
                allow_incomplete=args.allow_incomplete, labels=labels, diagnostic_ci=not args.no_diagnostic_ci,
                allow_model_specific_settings=args.allow_model_specific_settings,
                progress=lambda r: print(canonical(r), flush=True))
        print(canonical({'rendering': args.output, 'data_status': bundle['status']}), flush=True)
        manifest = export_bundle(bundle, args.output, width=args.width, dpi=args.dpi,
                                  formats=tuple(dict.fromkeys(args.formats)), tables_only=args.tables_only)
        print(canonical({'gallery': str(Path(args.output).resolve() / 'index.html'),
                         'figures': sum(f['status'] == 'generated' for f in manifest['figures']),
                         'unavailable': sum(f['status'] != 'generated' for f in manifest['figures']), 'tables': len(manifest['tables'])}))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f'Figure generation failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
