from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from physgraph_api_client import (  # noqa: E402
    PhysGraphAPIError,
    PhysGraphAPIClient,
    PhysGraphAPIResponseError,
    ExclusiveRunLock,
    extract_json_object,
    image_data_url,
    load_api_key,
    normalize_base_url,
)


class FakeCompletions:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), finish_reason="stop")],
            model="test-multimodal-model",
            usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            _request_id="request-test",
        )


class PhysGraphAPIClientTests(unittest.TestCase):
    def test_key_is_required_from_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(PhysGraphAPIError):
                load_api_key()

    def test_key_is_read_without_modification(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "temporary-test-key"}, clear=True):
            self.assertEqual(load_api_key(), "temporary-test-key")

    def test_base_url_requires_https_except_localhost(self) -> None:
        self.assertEqual(normalize_base_url("https://example.com/v1/"), "https://example.com/v1")
        self.assertEqual(normalize_base_url("http://localhost:8000/v1"), "http://localhost:8000/v1")
        with self.assertRaises(ValueError):
            normalize_base_url("http://example.com/v1")
        with self.assertRaises(ValueError):
            normalize_base_url("https://secret@localhost/v1")
        with self.assertRaises(ValueError):
            normalize_base_url("https://example.com/v1?api_key=secret")

    def test_json_response_fence_is_tolerated(self) -> None:
        self.assertEqual(extract_json_object("```json\n{\"ok\":true}\n```"), {"ok": True})

    def test_image_is_encoded_as_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "x.png"
            image.write_bytes(b"PNG")
            self.assertEqual(image_data_url(image), "data:image/png;base64,UE5H")

    def make_fake_client(self, text: str) -> tuple[PhysGraphAPIClient, FakeCompletions]:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "temporary-test-key"}, clear=True):
            client = PhysGraphAPIClient(base_url="https://example.com/v1", transport_retries=0)
        completions = FakeCompletions(text)
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        return client, completions

    def test_chat_uses_plain_text_when_there_are_no_images(self) -> None:
        client, completions = self.make_fake_client('{"ok":true}')
        document, result = client.request_json(
            model="test-multimodal-model",
            reasoning_effort="medium",
            system_prompt="system",
            user_prompt="user",
        )
        self.assertEqual(document, {"ok": True})
        self.assertEqual(result.usage["total_tokens"], 13)
        self.assertEqual(completions.kwargs["messages"][1]["content"], "user")

    def test_chat_embeds_images_as_data_urls(self) -> None:
        client, completions = self.make_fake_client('{"ok":true}')
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "x.png"
            image.write_bytes(b"PNG")
            client.request_json(
                model="test-multimodal-model",
                reasoning_effort="medium",
                system_prompt="system",
                user_prompt="user",
                image_paths=[image],
                image_detail="high",
            )
        content = completions.kwargs["messages"][1]["content"]
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,UE5H")
        self.assertEqual(content[1]["image_url"]["detail"], "high")

    def test_bad_json_preserves_usage_for_audit(self) -> None:
        client, _ = self.make_fake_client("not-json")
        with self.assertRaises(PhysGraphAPIResponseError) as raised:
            client.request_json(
                model="test-multimodal-model",
                reasoning_effort="medium",
                system_prompt="system",
                user_prompt="user",
            )
        self.assertEqual(raised.exception.result.request_id, "request-test")
        self.assertEqual(raised.exception.result.usage["total_tokens"], 13)

    def test_run_lock_blocks_a_second_worker_and_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.lock"
            first = ExclusiveRunLock(path)
            second = ExclusiveRunLock(path)
            first.acquire()
            try:
                with self.assertRaises(PhysGraphAPIError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
