"""Portable gallery, figures, tables and exact numerical provenance."""

from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

from physalign.dataset import require
from physalign.storage import atomic_json, atomic_text, file_hash, fingerprint
from .tables import write_tables


def gallery(bundle, figures, tables):
    status = bundle['status']
    notice = {'synthetic_preview': 'SYNTHETIC DESIGN PREVIEW · All values are illustrative; no real model or human results.',
              'non_scientific': 'NON-SCIENTIFIC RUN · Pipeline validation only; these are not model benchmark results.',
              'scientific': 'Scored experiment artifacts · Source report hashes and analysis settings are recorded.'}[status]
    if bundle.get('comparison_note'):
        notice += ' ' + bundle['comparison_note']
    cards = []
    for f in figures:
        links = ' · '.join(f'<a href="figures/{escape(path)}">{fmt.upper()}</a>' for fmt, path in f['files'].items())
        preview = f['files'].get('svg', f['files'].get('png'))
        if preview:
            image = f'<a href="figures/{escape(preview)}"><img loading="lazy" src="figures/{escape(preview)}" alt="{escape(f["title"])}"></a>'
        elif f['files']:
            image = '<div class="missing">PDF export available below.<small>Add --formats pdf svg for inline previews.</small></div>'
        else:
            image = f'<div class="missing">Unavailable<br><small>{escape(f["reason"])}</small></div>'
        cards.append(f'<article class="card" data-section="{f["section"]}"><div class="eyebrow">{f["section"].upper()} · {escape(f["id"])}</div><h2>{escape(f["title"])}</h2>{image}<p>{escape(f.get("caption", ""))}</p><div class="links">{links}</div></article>')
    table_html = []
    for t in tables:
        links = ' · '.join(f'<a href="tables/{escape(path)}">{fmt.upper()}</a>' for fmt, path in t['files'].items())
        rows = ''.join('<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in row) + '</tr>' for row in t['rows'][:100])
        body = '<table><thead><tr>' + ''.join(f'<th>{escape(c)}</th>' for c in t['columns']) + '</tr></thead><tbody>' + rows + '</tbody></table>'
        note = f'<p class="muted">First 100 of {len(t["rows"])} rows shown. Downloads contain all rows.</p>' if len(t['rows']) > 100 else ''
        table_html.append(f'<details class="table-card"><summary>{escape(t["title"])} <span>{len(t["rows"])} rows</span></summary><p>{escape(t["caption"])}</p><div class="scroll">{body}</div>{note}<div class="links">{links}</div></details>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysAlign · Figure book</title>
<style>
:root{--ink:#203047;--muted:#65758b;--line:#dce4ec;--accent:#137f78}*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:var(--ink);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}header,main{max-width:1320px;margin:auto;padding:32px}header{padding-top:55px;padding-bottom:18px}.brand{font-size:12px;letter-spacing:.2em;color:var(--accent);font-weight:700}h1{font:500 clamp(32px,4vw,52px)/1.15 Georgia,serif;letter-spacing:-.025em;margin:15px 0}header p{max-width:900px;color:var(--muted)}.notice{padding:13px 18px;background:#fff3e8;border-left:3px solid #c7783c;font-size:13px}nav{display:flex;gap:8px;flex-wrap:wrap;margin:25px 0}button{padding:9px 18px;background:white;color:var(--ink);border:1px solid var(--line);border-radius:5px;cursor:pointer}button.active{background:var(--ink);color:white;border-color:var(--ink)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.card,.table-card{background:white;border:1px solid var(--line);border-radius:7px;padding:24px}.card .eyebrow{color:var(--accent);font-size:10px;font-weight:700;letter-spacing:.12em}.card h2{font-size:18px;font-weight:600;line-height:1.35;margin:9px 0 18px}.card img{width:100%;height:auto;display:block}.card p,.table-card p{color:var(--muted);font-size:12px;line-height:1.65}.links{font-size:12px;letter-spacing:.06em;font-weight:600}.links a{color:var(--accent);text-decoration:none}.links a:hover{text-decoration:underline}.missing{min-height:180px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#f5f7f9;border:1px dashed var(--line);padding:20px;color:var(--muted)}.missing small{font-size:12px;text-align:center}h2.section-title{font:400 28px Georgia,serif;margin:50px 0 20px}.table-card{margin:14px 0}.table-card summary{cursor:pointer;font-weight:600}.table-card summary span{font-size:12px;color:var(--muted);float:right}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;margin:18px 0}th{text-align:right;border-top:2px solid var(--ink);border-bottom:1px solid var(--line);padding:10px 12px;background:#f7f9fb;font-weight:600}td{text-align:right;padding:9px 12px;white-space:nowrap}td:first-child,th:first-child{text-align:left}tbody tr:last-child td{border-bottom:2px solid var(--ink)}tbody tr:nth-child(even){background:#fafbfd}.muted{color:var(--muted)}footer{font-size:12px;color:var(--muted);padding:35px 0}a{color:var(--accent)}[hidden]{display:none!important}@media(max-width:850px){header,main{padding:24px}.grid{grid-template-columns:1fr}.card{padding:18px}}@media print{nav{display:none}.card{break-inside:avoid}.grid{display:block}.card{margin-bottom:20px}}
</style></head><body><header><div class="brand">PHYSALIGN / EVIDENCE-GROUNDED EVALUATION</div><h1>Reading is only part of the story.</h1><p>A publication figure book for recognition, evidence binding, independent physics solving, and controls. Select figures according to the evidence; association alone does not establish causation.</p>''' + f'<div class="notice">{escape(notice)}</div>' + '''<nav><button class="active" data-filter="all">All figures</button><button data-filter="main">Main-paper candidates</button><button data-filter="appendix">Appendix diagnostics</button><a href="#tables">Tables ↓</a></nav></header><main><div class="grid">''' + ''.join(cards) + '''</div><h2 class="section-title" id="tables">Tables for the manuscript</h2><p class="muted">Captions appear above tables. Percentages, percentage-point differences, and sample counts have explicit units; -- is unavailable. CSV/LaTeX downloads retain every row.</p>''' + ''.join(table_html) + '''<footer>Numerical source: <a href="figure_data.json">figure_data.json</a> · Provenance and file hashes: <a href="figure_manifest.json">figure_manifest.json</a> · Figure captions: <a href="captions.md">captions.md</a>. PDF/SVG are vector exports at the declared physical width. No network resources are required to view this gallery.</footer></main>
<script>document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.card').forEach(x=>x.hidden=b.dataset.filter!=='all'&&x.dataset.section!==b.dataset.filter)}));</script></body></html>'''


def export_bundle(bundle, output, *, width=5.5, dpi=450, formats=('pdf', 'svg', 'png'), tables_only=False):
    root = Path(output).resolve()
    require(not root.exists(), 'Figure output already exists; choose a new directory to preserve the previous artifact/provenance')
    require(3 <= width <= 12 and type(dpi) is int and 72 <= dpi <= 1200, 'Invalid physical width or export DPI')
    require(formats and set(formats) <= {'pdf', 'svg', 'png'}, 'Formats must be pdf/svg/png')
    root.mkdir(parents=True)
    atomic_json(root / 'figure_data.json', bundle)
    tables = write_tables(bundle, root / 'tables')
    figures = []
    if not tables_only:
        from .plots import FigureBook
        figures = FigureBook(bundle, root / 'figures', width=width, dpi=dpi, formats=formats).render_all()
        preview_book = FigureBook(bundle, root / 'tables', width=width, dpi=dpi, formats=formats)
        for table in tables[:4]:
            preview_book.table_preview(table)
            table['files'].update(preview_book.entries[-1]['files'])
    atomic_text(root / 'index.html', gallery(bundle, figures, tables))
    if bundle.get('model_protocols'):
        atomic_json(root / 'model_protocols.json', bundle['model_protocols'])
    captions = ['# PhysAlign figure captions', '', 'Data status: ' + bundle['status'], '',
                'Intervals: clustered percentile bootstrap, using the recorded study configuration. Diagnostic strata and binding-score bins are exploratory.', '']
    if bundle.get('comparison_note'):
        captions.extend([bundle['comparison_note'], ''])
    for f in figures:
        captions.extend(['## ' + f['id'] + ': ' + f['title'], '', f.get('caption', f.get('reason', '')), ''])
    atomic_text(root / 'captions.md', '\n'.join(captions))
    versions = {'python': platform.python_version()}
    for package in ('matplotlib', 'numpy', 'Pillow'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    source = Path(__file__).parent
    manifest = {'schema_version': 'physalign_figure_manifest_v1', 'data_status': bundle['status'],
                'plan_hash': bundle['plan_hash'], 'figure_data_hash': fingerprint(bundle),
                'width_inches': width, 'dpi': dpi, 'formats': list(formats), 'versions': versions,
                'visualization_code_hashes': {p.name: file_hash(p) for p in sorted(source.glob('*.py'))},
                'figures': figures, 'tables': [{k: v for k, v in t.items() if k != 'rows'} for t in tables],
                'files': {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    atomic_json(root / 'figure_manifest.json', manifest)
    return manifest
