"""Actual human responses and interface audits; never synthesize human evidence."""

from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import secrets

from .contracts import decode_probe, remap_probe
from .dataset import require
from .metrics import Pool
from .scoring import score_response
from .storage import atomic_json, canonical, fingerprint, loads, read_json, run_lock, write_new
from .study import Study

AUDIT_CHECKS = ('source_version_approved', 'target_anchor_clear', 'question_matches_target',
                'candidate_coverage', 'ambiguity_resolved', 'packet_no_binding_leakage',
                'label_legibility', 'coordinate_alignment', 'overlay_no_occlusion')


def create_session(study_root, sessions_root, *, participant, mode='answer', cohort=0):
    study = Study(study_root)
    require(mode in {'answer', 'audit', 'adjudicate'}, 'Unknown human session role')
    require(isinstance(participant, str) and participant and all(c.isalnum() or c in '_-' for c in participant), 'Use a non-identifying participant code')
    require(type(cohort) is int and cohort in {0, 1}, 'Use counterbalance cohort 0 or 1')
    root = Path(sessions_root).resolve()
    tasks = []
    selected = set(study.plan['human_mothers'])
    for row in study.plan['requests']:
        if row['kind'] != 'probe' or row['problem_id'] not in selected:
            continue
        if mode == 'answer':
            if row['variant'] != 'main':
                continue
            has_gold = row['instance_id'] in study.plan['groups']['main'].get('gold', [])
            parity = int(fingerprint([study.plan['human_seed'], row['instance_id']]), 16) % 2
            condition = 'gold' if has_gold and (parity ^ cohort) else 'raw'
            if row['score_condition'] != condition:
                continue
        elif row['variant'] not in {'main', 'permutation_reference', 'no_image_no_geometry'}:
            continue
        tasks.append({'instance_id': row['instance_id'], 'condition': row['condition'],
                      'input_hash': row['input_hash'], 'task_id': fingerprint([row['instance_id'], row['condition'], row['input_hash']])})
    require(tasks, 'No predeclared human tasks')
    with run_lock(root):
        for path in root.glob('*/session.json'):
            previous = read_json(path)
            if previous['participant'] == participant and previous['plan_hash'] == study.plan['plan_hash']:
                require(previous['mode'] == mode and previous['cohort'] == cohort,
                        'A participant cannot cross Raw/Gold cohorts or blind/auditor roles in this study')
                raise ValueError('Participant session already exists; reopen it instead of obtaining another answer')
        session = {'schema_version': 'physalign_human_session_v1', 'plan_hash': study.plan['plan_hash'],
                   'participant': participant, 'mode': mode, 'cohort': cohort, 'tasks': tasks,
                   'created_at': datetime.now(timezone.utc).isoformat()}
        directory = root / f'{mode}-{participant}'
        require(not directory.exists(), 'Human session path already exists')
        session['session_hash'] = fingerprint(session)
        write_new(directory / 'session.json', session)
    return directory


def _session(study, directory):
    s = read_json(Path(directory) / 'session.json')
    require(s['session_hash'] == fingerprint({k: v for k, v in s.items() if k != 'session_hash'}), 'Human assignment changed')
    require(s['plan_hash'] == study.plan['plan_hash'], 'Human assignment belongs to another study')
    return s


def public_task(study, session, task, sessions_root=None):
    row = study.by_key[task['instance_id'], task['condition']]
    require(row['input_hash'] == task['input_hash'], 'Human input changed')
    req = study.request(row, task['task_id'], '{}')
    result = {'task_id': task['task_id'], 'input_hash': row['input_hash'], 'system': req.system, 'user': req.user,
              'images': [{'asset_id': a.asset_id, 'width': a.width, 'height': a.height, 'sha256': a.sha256, 'url': u}
                         for a, u in zip(req.images, req.image_data_urls())], 'mode': session['mode']}
    if session['mode'] != 'answer':
        p = decode_probe(study.private()['probes'][row['instance_id']])
        # Audit variants currently use identity maps. Never expose this branch in blind mode.
        result['private_reference'] = {'binding_fields': {f: dict(d) for f, d in p.binding.domains.items()},
                                       'gold': list(p.binding.gold), 'readings': [r.expected for r in p.readings]}
        result['audit_checks'] = list(AUDIT_CHECKS)
        if session['mode'] == 'adjudicate' and sessions_root is not None:
            history = []
            for path in Path(sessions_root).glob('*/session.json'):
                prior = _session(study, path.parent)
                file = path.parent / 'responses' / (task['task_id'] + '.json')
                if prior['mode'] != 'adjudicate' and file.exists():
                    record = read_json(file)
                    history.append({'participant': prior['participant'], 'role': prior['mode'],
                                    'record_hash': record['record_hash'], 'response': record['response']})
            result['prior_reviews'] = history
    return result


def submit(study, directory, task_id, response):
    directory = Path(directory)
    with run_lock(directory):
        session = _session(study, directory)
        task = next((t for t in session['tasks'] if t['task_id'] == task_id), None)
        require(task is not None, 'Task was not assigned to this participant')
        require(response.get('human_confirmed') is True, 'An actual human must confirm this submission')
        if session['mode'] == 'answer':
            require(isinstance(response.get('text'), str), 'Preserve the human response string')
            require(type(response.get('ambiguous')) is bool and isinstance(response.get('notes'), str), 'Report ambiguity and notes')
            require(set(response) - {'display'} == {'text', 'human_confirmed', 'ambiguous', 'notes'}, 'Unexpected blind response fields')
        else:
            require(response.get('decision') in {'accept', 'reject', 'revise'}, 'Audit needs an explicit decision')
            require(set(response.get('checks', {})) == set(AUDIT_CHECKS) and all(type(v) is bool for v in response['checks'].values()), 'Complete every interface audit check')
            require(isinstance(response.get('notes'), str), 'Audit notes must be text')
            require(response['decision'] != 'accept' or all(response['checks'].values()), 'Cannot approve failed audit checks')
            require(response['decision'] == 'accept' or bool(response['notes'].strip()), 'Rejected/revised tasks need reasons')
            if session['mode'] == 'adjudicate':
                require(bool(response['notes'].strip()), 'Adjudication requires a rationale')
        if 'display' in response:
            require(isinstance(response['display'], list), 'Display audit must be an image list')
            row = study.by_key[task['instance_id'], task['condition']]
            assets = row['input']['attachments']
            require([d['asset_id'] for d in response['display']] == [a['asset_id'] for a in assets], 'Displayed image order differs from task')
            for d, a in zip(response['display'], assets):
                require((d['natural_width'], d['natural_height']) == (a['width'], a['height']), 'Browser decoded different image dimensions')
                require(d['displayed_width'] > 0 and d['displayed_height'] > 0, 'Image was not rendered')
        value = {'task_id': task_id, 'input_hash': task['input_hash'], 'session_hash': session['session_hash'],
                 'submitted_at': datetime.now(timezone.utc).isoformat(), 'response': response}
        value['record_hash'] = fingerprint(value)
        write_new(directory / 'responses' / (task_id + '.json'), value)
        return value


def human_report(study_root, sessions_root, *, output=None):
    study = Study(study_root)
    root = Path(sessions_root)
    private = study.private()
    probes = {i: decode_probe(r) for i, r in private['probes'].items()}
    selected = [p for p in probes.values() if p.problem_id in study.plan['human_mothers']]
    responses, audits, adjudications = defaultdict(list), defaultdict(list), defaultdict(list)
    evidence, people, ambiguities = {}, {}, []
    display_audits = {'submissions': 0, 'with_browser_dimensions': 0}
    for path in sorted(root.glob('*/session.json')):
        session = _session(study, path.parent)
        participant = session['participant']
        require(participant not in people, 'Duplicate participant or conflicting human roles/cohorts')
        people[participant] = {'mode': session['mode'], 'cohort': session['cohort']}
        evidence[str(path.resolve())] = fingerprint(session)
        for task in session['tasks']:
            file = path.parent / 'responses' / (task['task_id'] + '.json')
            if not file.exists():
                continue
            record = read_json(file)
            require(record['record_hash'] == fingerprint({k: v for k, v in record.items() if k != 'record_hash'}), 'Human response record changed')
            require((record['session_hash'], record['input_hash'], record['task_id']) == (session['session_hash'], task['input_hash'], task['task_id']), 'Human response/assignment mismatch')
            evidence[str(file.resolve())] = fingerprint(record)
            row = study.by_key[task['instance_id'], task['condition']]
            response = record['response']
            display_audits['submissions'] += 1
            display_audits['with_browser_dimensions'] += int('display' in response)
            require(response.get('human_confirmed') is True, 'Unconfirmed human evidence')
            if session['mode'] == 'answer':
                score = score_response(probes[row['instance_id']], response['text'], condition=row['score_condition'])
                responses[row['instance_id'], row['score_condition']].append(score)
                if response['ambiguous']:
                    ambiguities.append({'task_id': task['task_id'], 'participant': participant, 'notes': response['notes']})
            else:
                require(response['decision'] != 'accept' or all(response['checks'].get(k) is True for k in AUDIT_CHECKS), 'Human approval contradicts checklist')
                (adjudications if session['mode'] == 'adjudicate' else audits)[task['task_id']].append(response)
    summaries, counts = {}, {}
    for condition in ('raw', 'gold'):
        ps = [p for p in selected if p.probe_id in study.plan['groups']['main'].get(condition, [])]
        if not ps:
            summaries[condition] = None
            continue
        pool = Pool(tuple(ps))
        def metric(name, eligible):
            values = {}
            for p in eligible:
                scores = responses[p.probe_id, condition]
                values[p.probe_id] = sum(getattr(s, name) for s in scores) / len(scores) if scores else None
            return Pool(tuple(eligible)).estimate(values)
        summaries[condition] = {'BAcc': metric('B', ps)}
        if condition == 'raw':
            joint = [p for p in ps if p.joint_eligible]
            summaries[condition].update(CAcc=metric('C', joint), JAcc=metric('J', joint))
        counts[condition] = {p.probe_id: len(responses[p.probe_id, condition]) for p in ps}
    paired_ps = tuple(p for p in selected if p.probe_id in study.plan['groups']['main'].get('gold', []))
    paired = None
    if paired_ps:
        pool = Pool(paired_ps)
        estimates = {}
        for condition in ('raw', 'gold'):
            values = {p.probe_id: (sum(s.B for s in responses[p.probe_id, condition]) / len(responses[p.probe_id, condition])
                      if responses[p.probe_id, condition] else None) for p in paired_ps}
            estimates[condition] = pool.estimate(values)
        raw, gold = estimates['raw'], estimates['gold']
        paired = {**estimates, 'delta_BAcc_gold_raw': gold['value'] - raw['value']
                  if raw['value'] is not None and gold['value'] is not None else None,
                  'delta_bounds': {'lower': gold['lower'] - raw['upper'], 'upper': gold['upper'] - raw['lower']},
                  'assignment': 'same probes, different counterbalanced humans; human means within probe first'}
    expected_audits = {fingerprint([r['instance_id'], r['condition'], r['input_hash']]) for r in study.plan['requests']
                       if r['kind'] == 'probe' and r['problem_id'] in study.plan['human_mothers']
                       and r['variant'] in {'main', 'permutation_reference', 'no_image_no_geometry'}}
    decisions = {}
    for tid in sorted(expected_audits):
        answers = adjudications.get(tid) or audits.get(tid, [])
        statuses = {a['decision'] for a in answers}
        decisions[tid] = 'accept' if statuses == {'accept'} else 'missing' if not answers else 'unresolved'
    agreement = {'pairs': 0, 'agreeing_pairs': 0}
    for scores in responses.values():
        for i, a in enumerate(scores):
            for b in scores[i + 1:]:
                agreement['pairs'] += 1
                agreement['agreeing_pairs'] += int(a.canonical_binding is not None and a.canonical_binding == b.canonical_binding)
    agreement['rate'] = agreement['agreeing_pairs'] / agreement['pairs'] if agreement['pairs'] else None
    result = {'schema_version': 'physalign_human_report_v1', 'plan_hash': study.plan['plan_hash'],
              'participants': people, 'summaries': summaries, 'paired': paired, 'responses_per_probe': counts,
              'display_audit_coverage': display_audits,
              'canonical_agreement': agreement, 'ambiguity_reports': ambiguities,
              'audit_decisions': decisions, 'adjudicated_tasks': sorted(adjudications), 'evidence': evidence,
              'complete_answer_coverage': bool(counts) and all(n > 0 for c in counts.values() for n in c.values()),
              'audit_complete_and_accepted': bool(decisions) and all(v == 'accept' for v in decisions.values())}
    if output:
        atomic_json(Path(output), result)
    return result


def seal_study(study_root, sessions_root):
    study = Study(study_root)
    report = human_report(study_root, sessions_root)
    require(report['complete_answer_coverage'], 'Real same-interface human Raw/Gold response coverage is incomplete')
    require(report['audit_complete_and_accepted'], 'Interface audit has missing or unresolved tasks')
    # Explicit ambiguity reports require an adjudication session even if an initial auditor accepted.
    if report['ambiguity_reports']:
        require(all(r['task_id'] in report['adjudicated_tasks'] for r in report['ambiguity_reports']), 'Every human ambiguity report requires task-specific adjudication')
    seal = {'schema_version': 'physalign_study_seal_v1', 'plan_hash': study.plan['plan_hash'],
            'human_report': report, 'sealed_at': datetime.now(timezone.utc).isoformat()}
    seal['seal_hash'] = fingerprint(seal)
    write_new(study.root / 'seal.json', seal)
    return seal


HTML = r'''<!doctype html><meta charset="utf-8"><title>PhysAlign human interface</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1200px;padding:0 20px;background:#f5f6f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border:1px solid #ddd}img{display:block;max-width:100%;height:auto;border:1px solid #ccc;cursor:zoom-in}textarea{width:100%;min-height:130px;font:16px monospace}button{padding:12px 24px;margin:16px 0}label{display:block;margin:10px 0}.asset{margin:24px 0}#notice{color:#953800}</style>
<h1>PhysAlign</h1><p id="progress"></p><div id="task"></div><div id="form"></div><p id="notice"></p>
<script>
const token=new URLSearchParams(location.search).get('token');let current;
async function api(path,options={}){const r=await fetch(path+'?token='+encodeURIComponent(token),options);const v=await r.json();if(!r.ok)throw Error(v.error);return v}
function el(tag,text,parent){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;(parent||document.querySelector('#task')).append(e);return e}
async function next(){try{const data=await api('/task');current=data.task;document.querySelector('#task').replaceChildren();document.querySelector('#form').replaceChildren();document.querySelector('#progress').textContent=`已提交 ${data.completed} / ${data.total}`;if(!current){el('p','本次任务已完成。');return}
el('h2','System');el('pre',current.system);el('h2','User');el('pre',current.user);for(const image of current.images){const div=el('div');div.className='asset';el('p',`Image asset ID: ${image.asset_id} · ${image.width} × ${image.height}`,div);const im=el('img',undefined,div);im.src=image.url;im.alt=image.asset_id;im.dataset.asset=image.asset_id;im.onclick=()=>{im.style.maxWidth=im.style.maxWidth==='none'?'100%':'none'}}
const f=document.querySelector('#form');if(current.mode==='answer'){el('h2','按题目 Output format 作答',f);const out=el('textarea',undefined,f);out.id='answer';const a=el('label','此题存在歧义或界面问题 ',f);const box=el('input',undefined,a);box.type='checkbox';box.id='ambiguous'}else{el('h2','审核参考（不向盲测参与者展示）',f);el('pre',JSON.stringify(current.private_reference,null,2),f);if(current.prior_reviews)el('pre',JSON.stringify(current.prior_reviews,null,2),f);for(const name of current.audit_checks){const l=el('label',name+' ',f);const c=el('input',undefined,l);c.type='checkbox';c.dataset.check=name}const select=el('select',undefined,f);select.id='decision';for(const name of ['revise','accept','reject']){const o=el('option',name,select);o.value=name}}
el('p','备注 / 歧义说明 / 审核理由',f);const notes=el('textarea',undefined,f);notes.id='notes';const h=el('label','我确认这是本人实际完成的作答或审核。 ',f);const cb=el('input',undefined,h);cb.type='checkbox';cb.id='confirmed';const button=el('button','提交并进入下一题（提交后不可修改）',f);button.onclick=send;
}catch(e){document.querySelector('#notice').textContent=e.message}}
async function send(){try{const response={human_confirmed:document.querySelector('#confirmed').checked,notes:document.querySelector('#notes').value,display:[...document.querySelectorAll('img[data-asset]')].map(e=>({asset_id:e.dataset.asset,natural_width:e.naturalWidth,natural_height:e.naturalHeight,displayed_width:e.getBoundingClientRect().width,displayed_height:e.getBoundingClientRect().height}))};if(current.mode==='answer'){response.text=document.querySelector('#answer').value;response.ambiguous=document.querySelector('#ambiguous').checked}else{response.decision=document.querySelector('#decision').value;response.checks=Object.fromEntries([...document.querySelectorAll('[data-check]')].map(e=>[e.dataset.check,e.checked]))}await api('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:current.task_id,response})});document.querySelector('#notice').textContent='';await next()}catch(e){document.querySelector('#notice').textContent=e.message}}
next();</script>'''


def serve_session(study_root, session_directory, *, port=8765):
    study, directory = Study(study_root), Path(session_directory)
    session = _session(study, directory)
    token = secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not put the access token in logs.
        def send(self, code, body, mime='application/json; charset=utf-8'):
            data = body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)
        def authorized(self):
            return parse_qs(urlparse(self.path).query).get('token') == [token]
        def do_GET(self):
            if not self.authorized():
                return self.send(403, canonical({'error': 'Invalid session token'}))
            route = urlparse(self.path).path
            if route == '/':
                return self.send(200, HTML, 'text/html; charset=utf-8')
            if route == '/task':
                pending = [t for t in session['tasks'] if not (directory / 'responses' / (t['task_id'] + '.json')).exists()]
                task = public_task(study, session, pending[0], directory.parent) if pending else None
                return self.send(200, canonical({'task': task, 'completed': len(session['tasks']) - len(pending), 'total': len(session['tasks'])}))
            self.send(404, canonical({'error': 'Unknown route'}))
        def do_POST(self):
            if not self.authorized() or urlparse(self.path).path != '/submit':
                return self.send(403, canonical({'error': 'Invalid route/token'}))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                require(0 < length <= 1024 * 1024, 'Invalid submission size')
                value = loads(self.rfile.read(length).decode('utf-8'))
                submit(study, directory, value['task_id'], value['response'])
                self.send(200, canonical({'saved': True}))
            except (ValueError, KeyError, OSError) as exc:
                self.send(400, canonical({'error': str(exc)}))
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'http://127.0.0.1:{server.server_port}/?token={token}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
