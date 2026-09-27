"""Provider-independent model boundary. It carries public text and image BYTES.

Adapters must perform a stateless completion per request, with no conversation
history or previous-response ID. They may retain model weights, not dialogue.
Only InfrastructureError may be retried by the runner.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .dataset import ImageData, require
from .storage import canonical, loads, read_json


def _no_credentials(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in {"api_key", "access_token", "auth_token", "authorization", "password", "secret"},
                    "Credentials must not be stored in model metadata/config; use the adapter's environment")
            _no_credentials(child)
    elif isinstance(value, list):
        for child in value:
            _no_credentials(child)


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    model_id: str
    revision: str | None
    implementation_version: str
    settings_json: str
    preprocessing_json: str
    scientific_run: bool = True

    def __post_init__(self):
        require(all(isinstance(v, str) and v for v in (self.name, self.model_id, self.implementation_version)), "Adapter identity must be explicit")
        require(self.revision is None or isinstance(self.revision, str), "Invalid model revision")
        require(type(self.scientific_run) is bool, "Invalid scientific-run flag")
        for text in (self.settings_json, self.preprocessing_json):
            value = loads(text)
            require(isinstance(value, dict), "Settings and preprocessing must be JSON objects")
            canonical(value)
            _no_credentials(value)

    def manifest(self) -> dict:
        return {"name": self.name, "model_id": self.model_id, "revision": self.revision,
                "implementation_version": self.implementation_version,
                "settings": loads(self.settings_json), "preprocessing": loads(self.preprocessing_json),
                "scientific_run": self.scientific_run}


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    system: str
    user: str
    images: tuple[ImageData, ...]
    settings_json: str

    def public_snapshot(self) -> dict:
        return {"request_id": self.request_id, "messages": {"system": self.system, "user": self.user},
                "images": [{"asset_id": a.asset_id, "mime_type": a.mime_type, "sha256": a.sha256,
                            "width": a.width, "height": a.height, "byte_count": len(a.data)} for a in self.images],
                "settings": loads(self.settings_json), "context": "fresh_stateless_completion"}

    def image_data_urls(self) -> tuple[str, ...]:
        return tuple("data:" + a.mime_type + ";base64," + base64.b64encode(a.data).decode("ascii") for a in self.images)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    finish_reason: str | None = None
    usage_json: str = "{}"
    returned_model_version: str | None = None
    provider_request_id: str | None = None
    generation_json: str = "{}"

    def __post_init__(self):
        require(isinstance(self.text, str), "Adapter must preserve the raw response string, including empty responses")
        usage = loads(self.usage_json)
        require(isinstance(usage, dict), "Usage must be a JSON object")
        canonical(usage)
        require(isinstance(loads(self.generation_json), dict), "Generation trace must be a JSON object")
        canonical(loads(self.generation_json))
        for value in (self.finish_reason, self.returned_model_version, self.provider_request_id):
            require(value is None or isinstance(value, str), "Response metadata must be strings or null")

    def record(self) -> dict:
        return {"text": self.text, "finish_reason": self.finish_reason, "usage": loads(self.usage_json),
                "returned_model_version": self.returned_model_version, "provider_request_id": self.provider_request_id,
                "generation": loads(self.generation_json)}


class InfrastructureError(Exception):
    """The adapter certifies that NO usable response was received.

    Use a stable nonsecret code, not a request body/URL containing credentials.
    Refusal, malformed JSON, empty model text and output-budget truncation must
    return ModelResponse and must NEVER raise this exception.
    """
    def __init__(self, code: str, *, retryable: bool = True):
        require(isinstance(code, str) and 0 < len(code) <= 100 and all(c.isalnum() or c in "_.-" for c in code), "Infrastructure error needs a short nonsecret code")
        require(type(retryable) is bool, "Invalid retryability")
        super().__init__(code)
        self.code, self.retryable = code, retryable


class ModelAdapter(Protocol):
    @property
    def info(self) -> AdapterInfo: ...
    def generate(self, request: ModelRequest) -> ModelResponse: ...


class SmokeAdapter:
    """Offline transport check; its all-empty answers are NOT model results."""
    info = AdapterInfo("smoke", "not-a-model", None, "1", "{}",
                       '{"image_handling":"verify_encoded_bytes_only"}', False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        require(all(a.data for a in request.images), "Images were not loaded")
        return ModelResponse("{}", "smoke_test", returned_model_version="not-a-model")


class ReplayAdapter:
    """Replay previously recorded PUBLIC-request snapshots by their fingerprint.

    This validates plumbing without model/API integration. Replay is explicitly
    non-scientific; offline scoring of a real run reads that run's original log.
    """
    def __init__(self, path: str | Path):
        from .storage import file_hash
        self.records = read_json(Path(path))
        require(isinstance(self.records, dict), "Replay file must map content fingerprints to responses")
        self.info = AdapterInfo("replay", "offline-replay", file_hash(Path(path)), "1", "{}", "{}", False)

    def generate(self, request: ModelRequest) -> ModelResponse:
        from .storage import fingerprint
        snapshot = request.public_snapshot()
        snapshot.pop("request_id")
        key = fingerprint(snapshot)
        require(key in self.records, "Replay has no response for this exact public request")
        response = self.records[key]
        return ModelResponse(response["text"], response.get("finish_reason"), canonical(response.get("usage", {})),
                             response.get("returned_model_version"), response.get("provider_request_id"),
                             canonical(response.get("generation", {})))
