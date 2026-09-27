"""Stateless OpenAI-compatible multimodal transport, outside the frozen core.

Only public ModelRequest data cross this boundary. No SDK retries, response
repair, image resizing, tools, or conversation persistence are used.
"""
from __future__ import annotations

from http.client import HTTPException
import math
import os
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from physalign.adapters import AdapterInfo, InfrastructureError, ModelResponse, _no_credentials
from physalign.dataset import require
from physalign.storage import canonical, loads

ROOT = Path(__file__).resolve().parents[1]


def load_environment(path=None):
    """Read a small KEY=value file as data. Existing environment wins."""
    path = Path(path or os.environ.get('PHYSALIGN_ENV_FILE', ROOT / '.env.local'))
    if not path.is_file():
        return
    entries = {}
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        require(separator and re.fullmatch(r'PHYSALIGN_[A-Z0-9_]+_API_KEY', key),
                f'Invalid credential-file entry at line {number}; only PHYSALIGN_*_API_KEY is allowed')
        require(key not in entries, f'Duplicate credential variable at line {number}')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        require('\n' not in value and '\r' not in value, 'Invalid credential value')
        entries[key] = value
    for key, value in entries.items():
        if value:
            os.environ.setdefault(key, value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class APIAdapter:
    def __init__(self, config):
        _no_credentials(config)
        required = {'model_id', 'base_url', 'api_key_env', 'max_output_tokens', 'token_limit_field',
                    'reasoning_effort', 'temperature', 'image_detail', 'timeout_seconds', 'retry_delay_seconds'}
        require(isinstance(config, dict) and set(config) == required, 'API configuration fields are missing or unknown')
        config = loads(canonical(config))
        url = urlsplit(config['base_url'])
        require(url.scheme == 'https' and url.hostname and not url.username and not url.password
                and not url.query and not url.fragment, 'API base URL must be HTTPS without credentials/query/fragment')
        require(url.path.rstrip('/') == '/v1', 'API base URL must end in /v1')
        require(isinstance(config['model_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:/-]*', config['model_id']), 'Invalid model ID')
        require(isinstance(config['api_key_env'], str) and re.fullmatch(r'PHYSALIGN_[A-Z0-9_]+_API_KEY', config['api_key_env']), 'Invalid credential variable')
        require(type(config['max_output_tokens']) is int and 1 <= config['max_output_tokens'] <= 128000, 'Invalid output budget')
        require(config['token_limit_field'] in {'max_tokens', 'max_completion_tokens'}, 'Invalid API token-limit field')
        require(config['reasoning_effort'] in {None, 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}, 'Invalid reasoning effort')
        temperature = config['temperature']
        require(temperature is None or type(temperature) in (int, float) and math.isfinite(temperature) and 0 <= temperature <= 2, 'Invalid temperature')
        if config['model_id'].startswith('gpt-6-astra'):
            require(temperature is None, 'GPT-6 Astra does not support temperature')
            require(config['reasoning_effort'] not in (None, 'none', 'minimal'), 'Set an explicit supported Astra reasoning effort')
        require(config['image_detail'] in {'auto', 'low', 'high'}, 'Invalid image detail')
        for key, upper in [('timeout_seconds', 3600), ('retry_delay_seconds', 60)]:
            require(type(config[key]) in (int, float) and math.isfinite(config[key]) and 0 < config[key] <= upper, f'Invalid {key}')
        load_environment()
        credential = os.environ.get(config['api_key_env'], '')
        require(bool(credential) and not any(c.isspace() for c in credential), f'Missing/invalid environment variable: {config["api_key_env"]}')
        self.config, self._credential = config, credential
        self._opener = build_opener(NoRedirect())
        settings = {'api': 'chat_completions', 'base_url': config['base_url'].rstrip('/'),
                    'max_output_tokens': config['max_output_tokens'], 'token_limit_field': config['token_limit_field'],
                    'reasoning_effort': config['reasoning_effort'], 'temperature': temperature,
                    'stream': False, 'n': 1, 'context': 'fresh_stateless_completion',
                    'timeout_seconds': config['timeout_seconds'], 'retry_delay_seconds': config['retry_delay_seconds'],
                    'transport_retries': 0, 'tools': 'none', 'reasoning_budget_inclusion': 'provider_defined'}
        preprocessing = {'transport_version': 'asset_id_before_image_v1', 'image_detail': config['image_detail'],
                         'image_handling': 'all_original_attachment_bytes_in_source_order',
                         'local_resizing': False, 'local_text_truncation': False,
                         'provider_image_processing': 'opaque_provider_managed',
                         'provider_tokenizer_preflight': 'not_available'}
        self.info = AdapterInfo('openai-compatible-api', config['model_id'], None, '1',
                                canonical(settings), canonical(preprocessing), True)

    def payload(self, request):
        require(loads(request.settings_json) == self.info.manifest()['settings'], 'Request settings differ from frozen API adapter')
        content = [{'type': 'text', 'text': request.user}]
        for asset, data_url in zip(request.images, request.image_data_urls()):
            content.extend([{'type': 'text', 'text': asset.asset_id},
                            {'type': 'image_url', 'image_url': {'url': data_url, 'detail': self.config['image_detail']}}])
        payload = {'model': self.config['model_id'], 'messages': [
            {'role': 'system', 'content': request.system}, {'role': 'user', 'content': content}],
            self.config['token_limit_field']: self.config['max_output_tokens'], 'stream': False, 'n': 1}
        for key in ('reasoning_effort', 'temperature'):
            if self.config[key] is not None:
                payload[key] = self.config[key]
        return payload

    def _json_request(self, suffix, payload=None):
        data = None if payload is None else canonical(payload).encode('utf-8')
        request = Request(self.config['base_url'].rstrip('/') + suffix, data=data,
                          headers={'Authorization': 'Bearer ' + self._credential, 'Content-Type': 'application/json',
                                   'Accept': 'application/json', 'User-Agent': 'PhysAlign/1.0'})
        try:
            with self._opener.open(request, timeout=self.config['timeout_seconds']) as response:
                body = response.read()
            # Keep the provider response for an audit. It cannot carry our credential.
            text = body.decode('utf-8')
            require(self._credential not in text, 'Provider echoed a credential; response was withheld')
            return loads(text)
        except HTTPError as error:
            status = error.code
            error.close()
            # Never print provider error bodies: gateways may echo Authorization.
            if status in (400, 401, 403, 404, 405, 422):
                raise RuntimeError(f'API HTTP {status}; check endpoint, model access, balance and frozen request parameters') from None
            if status == 429:
                time.sleep(self.config['retry_delay_seconds'])
            raise InfrastructureError(f'api_http_{status}', retryable=status == 429) from None
        except (TimeoutError, socket.timeout):
            raise InfrastructureError('api_timeout_response_unknown', retryable=False) from None
        except URLError:
            raise InfrastructureError('api_transport_response_unknown', retryable=False) from None
        except (OSError, HTTPException, UnicodeError, ValueError, RecursionError):
            raise InfrastructureError('api_unreadable_response', retryable=False) from None

    def models(self):
        result = self._json_request('/models')
        require(isinstance(result, dict) and isinstance(result.get('data'), list), 'Invalid models-list envelope')
        return [row['id'] for row in result['data'] if isinstance(row, dict) and isinstance(row.get('id'), str)]

    def generate(self, request):
        result = self._json_request('/chat/completions', self.payload(request))
        try:
            require(isinstance(result, dict) and isinstance(result.get('choices'), list) and len(result['choices']) == 1, 'Invalid completion envelope')
            choice = result['choices'][0]
            message = choice['message']
            require(isinstance(message, dict) and message.get('role', 'assistant') == 'assistant', 'Invalid assistant message')
            content = message.get('content')
            refusal = message.get('refusal')
            if isinstance(content, str):
                text = content
            elif content is None:
                # Refusal, output-budget exhaustion and empty final text are served outcomes.
                text = refusal if isinstance(refusal, str) else ''
            elif isinstance(content, list) and all(isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str) for p in content):
                text = ''.join(p['text'] for p in content)
            else:
                raise ValueError('Unsupported assistant content')
            usage = result.get('usage') or {}
            require(isinstance(usage, dict), 'Invalid usage')
            usage = {**usage, 'input_tokens': usage.get('prompt_tokens', usage.get('input_tokens')),
                     'output_tokens': usage.get('completion_tokens', usage.get('output_tokens')),
                     'input_images': len(request.images)}
            return ModelResponse(text, choice.get('finish_reason'), canonical(usage), result.get('model'), result.get('id'),
                                 canonical({'provider_response': result, 'requested_model': self.config['model_id'],
                                            'requested_reasoning_effort': self.config['reasoning_effort']}))
        except (KeyError, TypeError, ValueError, RecursionError):
            raise InfrastructureError('api_invalid_completion_envelope', retryable=False) from None


def create_adapter(config):
    return APIAdapter(config)
