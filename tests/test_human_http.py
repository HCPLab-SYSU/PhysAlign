"""Real loopback HTTP exercise with temporary synthetic participant records."""

from http.server import ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json
import tempfile
import unittest
from unittest.mock import patch

from test_evaluation import native_bundle
from physalign.human import create_session, serve_session
from physalign.study import prepare_study


class HumanHTTPTests(unittest.TestCase):
    def test_blind_http_exposes_only_assigned_input_and_saves_first_submission(self):
        with tempfile.TemporaryDirectory(prefix='physalign-http-fixture-') as directory:
            root = Path(directory)
            bundle = native_bundle(root / 'bundle')
            study = root / 'study'
            prepare_study(bundle, study, split='development', bootstrap_resamples=0)
            session = create_session(study, root / 'sessions', participant='synthetic-fixture')
            servers, urls = Queue(), Queue()
            def build_server(address, handler):
                server = ThreadingHTTPServer(address, handler)
                servers.put(server)
                return server
            with patch('physalign.human.ThreadingHTTPServer', side_effect=build_server), \
                 patch('physalign.human.print', side_effect=lambda value, **kw: urls.put(value), create=True):
                thread = Thread(target=serve_session, args=(study, session), kwargs={'port': 0}, daemon=True)
                thread.start()
                server = servers.get(timeout=15)
                try:
                    address = urlparse(urls.get(timeout=15))
                    base = f'http://{address.netloc}'
                    token = '?' + address.query
                    def get(route):
                        with urlopen(base + route + token, timeout=10) as response:
                            return json.load(response)
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + '/task', timeout=10)
                    self.assertEqual(error.exception.code, 403)
                    with self.assertRaises(HTTPError) as error:
                        get('/private.json')
                    self.assertEqual(error.exception.code, 404)
                    value = get('/task')
                    task = value['task']
                    self.assertEqual(task['mode'], 'answer')
                    self.assertNotIn('private_reference', task)
                    self.assertNotIn('object-a', json.dumps(task))
                    self.assertTrue(all(i['url'].startswith('data:image/') for i in task['images']))
                    body = json.dumps({'task_id': task['task_id'], 'response': {
                        'text': '{}', 'human_confirmed': True, 'ambiguous': False,
                        'notes': 'Synthetic HTTP fixture; not actual participant data'}}).encode()
                    request = Request(base + '/submit' + token, data=body, headers={'Content-Type': 'application/json'})
                    with urlopen(request, timeout=10) as response:
                        self.assertTrue(json.load(response)['saved'])
                    self.assertEqual(get('/task')['completed'], 1)
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request, timeout=10)
                    self.assertEqual(error.exception.code, 400)
                finally:
                    server.shutdown()
                    thread.join(timeout=15)
                    self.assertFalse(thread.is_alive())
