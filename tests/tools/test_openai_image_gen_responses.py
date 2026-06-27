from __future__ import annotations

import base64
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path


_ONE_BY_ONE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def test_openai_provider_uses_responses_payload_for_high_quality(monkeypatch, tmp_path):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        openai_image,
        "_resolve_credentials",
        lambda: ("sk-test", "https://example.test/v1", "OPENAI_API_KEY"),
    )
    monkeypatch.setattr(
        openai_image,
        "_resolve_model",
        lambda: ("gpt-image-2-high", {"quality": "high"}),
    )

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse({
            "output": [{
                "type": "image_generation_call",
                "result": _ONE_BY_ONE_PNG,
                "quality": "high",
                "size": "1536x1024",
                "width": 1536,
                "height": 1024,
            }]
        })

    monkeypatch.setattr(openai_image.urllib.request, "urlopen", fake_urlopen)

    result = OpenAIImageGenProvider().generate("draw a cat", aspect_ratio="landscape")

    assert result["success"] is True
    assert result["model"] == "gpt-image-2-high"
    assert result["quality"] == "high"
    assert result["actual_quality"] == "high"
    assert captured["url"] == "https://example.test/v1/responses"
    assert captured["timeout"] == 420
    assert captured["payload"]["model"] == "gpt-image-2"
    assert captured["payload"]["quality"] == "high"
    assert captured["payload"]["size"] == "1536x1024"
    assert captured["payload"]["input"][0]["content"] == [
        {"type": "input_text", "text": "draw a cat"}
    ]
    assert Path(result["image"]).exists()


def test_openai_provider_embeds_source_images_as_responses_input_images(monkeypatch, tmp_path):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        openai_image,
        "_resolve_credentials",
        lambda: ("sk-test", "https://example.test/v1", "OPENAI_API_KEY"),
    )
    monkeypatch.setattr(
        openai_image,
        "_resolve_model",
        lambda: ("gpt-image-2-medium", {"quality": "medium"}),
    )
    monkeypatch.setattr(
        openai_image,
        "_load_image_bytes",
        lambda ref: (b"fake-image-bytes", "source.png"),
    )

    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse({
            "output": [{
                "type": "image_generation_call",
                "result": _ONE_BY_ONE_PNG,
                "quality": "medium",
            }]
        })

    monkeypatch.setattr(openai_image.urllib.request, "urlopen", fake_urlopen)

    result = OpenAIImageGenProvider().generate(
        "redraw this",
        image_url="/tmp/source.png",
        reference_image_urls=["/tmp/ref.png"],
    )

    assert result["success"] is True
    assert result["modality"] == "image"
    content = captured["payload"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "redraw this"}
    assert [item["type"] for item in content[1:]] == ["input_image", "input_image"]
    expected = "data:image/png;base64," + base64.b64encode(b"fake-image-bytes").decode("ascii")
    assert content[1]["image_url"] == expected
    assert content[2]["image_url"] == expected


def test_openai_provider_surfaces_responses_http_errors(monkeypatch, tmp_path):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        openai_image,
        "_resolve_credentials",
        lambda: ("sk-test", "https://example.test/v1", "OPENAI_API_KEY"),
    )

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":{"message":"temporarily unavailable"}}'),
        )

    monkeypatch.setattr(openai_image.urllib.request, "urlopen", fake_urlopen)

    result = OpenAIImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "HTTP 503" in result["error"]
    assert "temporarily unavailable" in result["error"]


def test_openai_provider_reports_empty_responses_payload(monkeypatch, tmp_path):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        openai_image,
        "_resolve_credentials",
        lambda: ("sk-test", "https://example.test/v1", "OPENAI_API_KEY"),
    )
    monkeypatch.setattr(
        openai_image.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse({"output": []}),
    )

    result = OpenAIImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "empty_response"
    assert "no image_generation_call" in result["error"]
