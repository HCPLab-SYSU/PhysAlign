"""Synthetic HTTP fixtures only; no credentials or paid network requests."""
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from physalign.adapters import InfrastructureError, ModelRequest
from physalign.dataset import ImageData
from physalign.storage import canonical, read_json
from server_eval.api_adapter import APIAdapter, NoRedirect, ROOT, load_environment


@contextmanager
def environment():
    with patch.dict(os.environ, {'PHYSALIGN_GPT_API_KEY': 'fixture-gpt-credential',
                               'PHYSALIGN_GEMINI_API_KEY': 'fixture-gemini-credential',
                               'PHYSALIGN_ENV_FILE': str(ROOT / 'tmp/nonexistent-test-env')}, clear=False):
        yield


def config(name='gpt-6-astra-high'):
    return read_json(ROOT / 'configs/api' / (name + '.json'))


def envelope(text='{"owner":"E1"}', **extra):
    return {'id': 'fixture-call', 'model': 'gpt-6-astra', 'choices': [
        {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
        'usage': {'prompt_tokens': 50, 'completion_tokens': 12}, **extra}


def request(adapter):
    return ModelRequest('fixture-request', 'Exact system text', 'Exact public user text', (
        ImageData('R1', 'image/png', 1, 1, 'fixture-hash-1', b'first-image-bytes'),
        ImageData('E9', 'image/png', 1, 1, 'fixture-hash-2', b'second-image-bytes')),
        canonical(adapter.info.manifest()['settings']))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = environment()
        self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)
        self.adapter = APIAdapter(config())

    def test_public_text_all_images_and_order_are_exact(self):
        req = request(self.adapter)
        payload = self.adapter.payload(req)
        self.assertEqual(payload['messages'][0], {'role': 'system', 'content': req.system})
        content = payload['messages'][1]['content']
        self.assertEqual(content[0]['text'], req.user)
        self.assertEqual([content[i]['text'] for i in (1, 3)], ['R1', 'E9'])
        self.assertEqual([content[i]['image_url']['url'] for i in (2, 4)], list(req.image_data_urls()))
        self.assertNotIn('previous_response_id', payload)
        self.assertNotIn('tools', payload)
        self.assertNotIn('response_format', payload)

    def test_high_effort_and_token_parameter_are_not_silently_downgraded(self):
        payload = self.adapter.payload(request(self.adapter))
        self.assertEqual(payload['model'], 'gpt-6-astra')
        self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertEqual(payload['max_completion_tokens'], 16384)
        self.assertNotIn('temperature', payload)
        self.assertNotIn('max_tokens', payload)
        self.assertFalse(payload['stream'])

    def test_gemini_uses_its_exact_config_and_provider_default_reasoning(self):
        adapter = APIAdapter(config('gemini-3.8-flash'))
        payload = adapter.payload(request(adapter))
        self.assertEqual(payload['model'], 'gemini-3.8-flash')
        self.assertEqual(payload['max_tokens'], 8192)
        self.assertEqual(payload['temperature'], 0)
        self.assertNotIn('reasoning_effort', payload)

    def test_fresh_call_contains_no_previous_reply(self):
        with patch.object(self.adapter, '_json_request', return_value=envelope()) as http:
            self.adapter.generate(request(self.adapter))
            self.adapter.generate(request(self.adapter))
        self.assertEqual(http.call_args_list[0], http.call_args_list[1])

    def test_first_final_text_is_preserved_with_provider_usage_and_version(self):
        original = '```json\n{"owner":"E9"}\n```\n'
        payload = envelope(original)
        payload['choices'][0]['message']['reasoning_content'] = 'A separate provider field'
        with patch.object(self.adapter, '_json_request', return_value=payload):
            response = self.adapter.generate(request(self.adapter))
        self.assertEqual(response.text, original)
        self.assertEqual(response.returned_model_version, 'gpt-6-astra')
        self.assertEqual(json.loads(response.usage_json)['input_tokens'], 50)
        self.assertEqual(json.loads(response.generation_json)['provider_response'], payload)

    def test_refusal_empty_malformed_and_truncated_text_are_served(self):
        for text, reason in [('', 'length'), ('no thanks', 'content_filter'), ('{"own', 'length'), ('not json', 'stop')]:
            payload = envelope(text)
            payload['choices'][0]['finish_reason'] = reason
            with self.subTest(text=text), patch.object(self.adapter, '_json_request', return_value=payload):
                self.assertEqual(self.adapter.generate(request(self.adapter)).text, text)
        payload = envelope(None)
        payload['choices'][0]['message']['refusal'] = 'Refusal exactly as received'
        with patch.object(self.adapter, '_json_request', return_value=payload):
            self.assertEqual(self.adapter.generate(request(self.adapter)).text, 'Refusal exactly as received')

    def test_hidden_reasoning_does_not_become_the_final_answer(self):
        payload = envelope(None)
        payload['choices'][0]['message']['reasoning_content'] = '{"owner":"E1"}'
        with patch.object(self.adapter, '_json_request', return_value=payload):
            self.assertEqual(self.adapter.generate(request(self.adapter)).text, '')

    def test_auth_is_sent_only_to_configured_host_and_not_in_metadata(self):
        stream = BytesIO(canonical(envelope()).encode())
        with patch.object(self.adapter._opener, 'open', return_value=stream) as opened:
            self.adapter.generate(request(self.adapter))
        req = opened.call_args.args[0]
        self.assertEqual(req.full_url, 'https://api.example.invalid/v1/chat/completions')
        self.assertEqual(req.get_header('Authorization'), 'Bearer fixture-gpt-credential')
        self.assertNotIn('fixture-gpt-credential', canonical(self.adapter.info.manifest()))
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.test'))

    def test_error_bodies_and_echoed_credentials_do_not_enter_logs(self):
        error = HTTPError('https://example.test', 401, 'fixture-gpt-credential', {}, BytesIO(b'fixture-gpt-credential'))
        with patch.object(self.adapter._opener, 'open', side_effect=error), self.assertRaises(RuntimeError) as context:
            self.adapter.generate(request(self.adapter))
        self.assertNotIn('fixture-gpt-credential', str(context.exception))
        with patch.object(self.adapter._opener, 'open', return_value=BytesIO(canonical(envelope('fixture-gpt-credential')).encode())), \
             self.assertRaises(InfrastructureError) as context:
            self.adapter.generate(request(self.adapter))
        self.assertFalse(context.exception.retryable)

    def test_rate_limit_only_is_retryable_and_no_transport_retry_occurs(self):
        for status in (429, 500, 502, 503, 307):
            error = HTTPError('https://example.test', status, 'secret error body', {}, BytesIO(b'secret'))
            with self.subTest(status=status), patch.object(self.adapter._opener, 'open', side_effect=error) as opened, \
                 patch('server_eval.api_adapter.time.sleep') as sleep, self.assertRaises(InfrastructureError) as caught:
                self.adapter.generate(request(self.adapter))
            self.assertEqual(opened.call_count, 1)
            self.assertEqual(caught.exception.retryable, status == 429)
            self.assertEqual(sleep.call_count, int(status == 429))

    def test_timeout_and_ambiguous_transport_are_terminal(self):
        for error in (TimeoutError(), URLError('response status unknown')):
            with patch.object(self.adapter._opener, 'open', side_effect=error), self.assertRaises(InfrastructureError) as caught:
                self.adapter.generate(request(self.adapter))
            self.assertFalse(caught.exception.retryable)

    def test_invalid_envelopes_never_trigger_answer_retries(self):
        for result in ({'choices': []}, {'choices': [{}, {}]}, envelope({'bad': 'shape'}), {'choices': [{}]}):
            with patch.object(self.adapter, '_json_request', return_value=result), self.assertRaises(InfrastructureError) as caught:
                self.adapter.generate(request(self.adapter))
            self.assertFalse(caught.exception.retryable)

    def test_config_guards_and_request_settings_mismatch(self):
        for key, value in [('base_url', 'http://example.test/v1'), ('base_url', 'https://user:pass@example.test/v1'),
                           ('base_url', 'https://example.test/v1?key=secret'), ('temperature', 0),
                           ('max_output_tokens', 0), ('reasoning_effort', 'none'), ('token_limit_field', 'unknown')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                APIAdapter({**config(), key: value})
        with self.assertRaises(ValueError):
            APIAdapter({**config(), 'api_key': 'forbidden'})
        req = request(self.adapter)
        with self.assertRaises(ValueError):
            self.adapter.payload(ModelRequest(req.request_id, req.system, req.user, req.images, '{}'))

    def test_environment_file_is_data_and_does_not_override_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env.local'
            path.write_text('# comment\nPHYSALIGN_GPT_API_KEY=other-value\nPHYSALIGN_TEST_API_KEY="$(not-executed)"\n', encoding='utf-8')
            load_environment(path)
            self.assertEqual(os.environ['PHYSALIGN_GPT_API_KEY'], 'fixture-gpt-credential')
            self.assertEqual(os.environ['PHYSALIGN_TEST_API_KEY'], '$(not-executed)')
            path.write_text('MALICIOUS_SCRIPT=do-stuff\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_environment(path)


if __name__ == '__main__':
    unittest.main()
