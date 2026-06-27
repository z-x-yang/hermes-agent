from __future__ import annotations

import base64
import io
import json
import sys
import types
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


def test_responses_api_false_string_overrides_gptcodex_heuristic(monkeypatch):
    import plugins.image_gen.openai as openai_image

    monkeypatch.setattr(
        openai_image,
        "_openai_subconfig",
        lambda: {"use_responses_api": "false"},
    )

    assert openai_image._use_responses_api("https://gptcodex.top/v1") is False


def test_responses_api_defaults_to_gptcodex_heuristic(monkeypatch):
    import plugins.image_gen.openai as openai_image

    monkeypatch.setattr(openai_image, "_openai_subconfig", lambda: {})

    assert openai_image._use_responses_api("https://gptcodex.top/v1") is True
    assert openai_image._use_responses_api("https://api.openai.com/v1") is False


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
    monkeypatch.setattr(openai_image, "_use_responses_api", lambda base_url: True)

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
                "size": "1536x864",
                "width": 1536,
                "height": 864,
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
    assert captured["payload"]["size"] == "1536x864"
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
    monkeypatch.setattr(openai_image, "_use_responses_api", lambda base_url: True)
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
    monkeypatch.setattr(openai_image, "_use_responses_api", lambda base_url: True)

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
    monkeypatch.setattr(openai_image, "_use_responses_api", lambda base_url: True)
    monkeypatch.setattr(
        openai_image.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse({"output": []}),
    )

    result = OpenAIImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "empty_response"
    assert "no image_generation_call" in result["error"]


def test_route_without_model_uses_global_openai_model(monkeypatch):
    import plugins.image_gen.openai as openai_image

    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    monkeypatch.setattr(
        openai_image,
        "_load_openai_config",
        lambda: {
            "model": "gpt-image-2-low",
            "openai": {"model": "gpt-image-2-high"},
        },
    )

    model, meta = openai_image._resolve_route_model(
        {
            "name": "openai-official",
            "provider": "openai",
            "openai": {"base_url": "", "api_key_env": "OPENAI_API_KEY"},
        }
    )

    assert model == "gpt-image-2-high"
    assert meta["quality"] == "high"


def test_openai_provider_falls_back_from_gptcodex_responses_to_official_openai(
    monkeypatch, tmp_path, caplog
):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GPTCODEX_API_KEY", "sk-gptcodex")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(
        openai_image,
        "_load_openai_config",
        lambda: {
            "provider": "openai",
            "model": "gpt-image-2-high",
            "fallbacks": [
                {
                    "name": "gptcodex",
                    "provider": "openai",
                    "model": "gpt-image-2-high",
                    "openai": {
                        "base_url": "https://gptcodex.top/v1",
                        "api_key_env": "GPTCODEX_API_KEY",
                        "use_responses_api": True,
                    },
                },
                {
                    "name": "openai-official",
                    "provider": "openai",
                    "model": "gpt-image-2-high",
                    "openai": {
                        "base_url": "",
                        "api_key_env": "OPENAI_API_KEY",
                        "use_responses_api": False,
                    },
                },
            ],
        },
    )

    captured = {"responses_urls": [], "openai_clients": [], "openai_payloads": []}

    def fake_urlopen(request, timeout):
        captured["responses_urls"].append(request.full_url)
        raise urllib.error.HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":{"message":"gptcodex overloaded"}}'),
        )

    class _FakeImage:
        b64_json = _ONE_BY_ONE_PNG
        revised_prompt = "official revised prompt"

    class _FakeImages:
        def generate(self, **payload):
            captured["openai_payloads"].append(payload)
            return types.SimpleNamespace(data=[_FakeImage()])

    class _FakeOpenAIClient:
        def __init__(self, **kwargs):
            captured["openai_clients"].append(kwargs)
            self.images = _FakeImages()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=_FakeOpenAIClient),
    )
    monkeypatch.setattr(openai_image.urllib.request, "urlopen", fake_urlopen)

    with caplog.at_level("WARNING", logger=openai_image.__name__):
        result = OpenAIImageGenProvider().generate("draw a cat", aspect_ratio="landscape")

    assert result["success"] is True
    assert result["provider_route"] == "openai-official"
    assert result["fallback_used"] is True
    assert result["attempted_routes"] == ["gptcodex", "openai-official"]
    assert result["route_failures"] == [
        {
            "route": "gptcodex",
            "error_type": "api_error",
            "error": 'OpenAI Responses image generation failed: HTTP 503: {"error":{"message":"gptcodex overloaded"}}',
        }
    ]
    assert result["model"] == "gpt-image-2-high"
    assert result["quality"] == "high"
    assert result["revised_prompt"] == "official revised prompt"
    assert Path(result["image"]).exists()
    assert captured["responses_urls"] == ["https://gptcodex.top/v1/responses"]
    assert captured["openai_clients"] == [{"api_key": "sk-openai"}]
    assert captured["openai_payloads"][0]["quality"] == "high"
    assert "OpenAI image generation route gptcodex failed" in caplog.text
    assert "falling back to route openai-official" in caplog.text


def test_openai_provider_all_route_failures_include_accumulated_details(monkeypatch, tmp_path):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SECOND_ROUTE_KEY", "sk-second")
    monkeypatch.delenv("MISSING_ROUTE_KEY", raising=False)
    monkeypatch.setattr(
        openai_image,
        "_load_openai_config",
        lambda: {
            "provider": "openai",
            "model": "gpt-image-2-medium",
            "fallbacks": [
                {
                    "name": "missing-key-route",
                    "provider": "openai",
                    "model": "gpt-image-2-medium",
                    "openai": {"api_key_env": "MISSING_ROUTE_KEY"},
                },
                {
                    "name": "empty-route",
                    "provider": "openai",
                    "model": "gpt-image-2-medium",
                    "openai": {
                        "base_url": "https://example.test/v1",
                        "api_key_env": "SECOND_ROUTE_KEY",
                        "use_responses_api": True,
                    },
                },
            ],
        },
    )
    monkeypatch.setattr(
        openai_image.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeHTTPResponse({"output": []}),
    )

    result = OpenAIImageGenProvider().generate("draw a cat")

    assert result["success"] is False
    assert result["error_type"] == "empty_response"
    assert result["attempted_routes"] == ["missing-key-route", "empty-route"]
    assert result["route_failures"] == [
        {
            "route": "missing-key-route",
            "error_type": "auth_required",
            "error": (
                "MISSING_ROUTE_KEY not set. Run `hermes tools` → Image "
                "Generation → OpenAI to configure, or set "
                "image_gen.openai.api_key_env in config.yaml."
            ),
        },
        {
            "route": "empty-route",
            "error_type": "empty_response",
            "error": "OpenAI Responses payload contained no image_generation_call result",
        },
    ]
    assert "All OpenAI image generation routes failed" in result["error"]


def test_openai_provider_does_not_fallback_for_source_image_load_errors(monkeypatch, tmp_path):
    from plugins.image_gen.openai import OpenAIImageGenProvider
    import plugins.image_gen.openai as openai_image

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("GPTCODEX_API_KEY", "sk-gptcodex")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setattr(
        openai_image,
        "_load_openai_config",
        lambda: {
            "provider": "openai",
            "model": "gpt-image-2-high",
            "fallbacks": [
                {
                    "name": "gptcodex",
                    "provider": "openai",
                    "model": "gpt-image-2-high",
                    "openai": {
                        "base_url": "https://gptcodex.top/v1",
                        "api_key_env": "GPTCODEX_API_KEY",
                        "use_responses_api": True,
                    },
                },
                {
                    "name": "openai-official",
                    "provider": "openai",
                    "model": "gpt-image-2-high",
                    "openai": {
                        "base_url": "",
                        "api_key_env": "OPENAI_API_KEY",
                        "use_responses_api": False,
                    },
                },
            ],
        },
    )
    monkeypatch.setattr(
        openai_image,
        "_load_image_bytes",
        lambda ref: (_ for _ in ()).throw(FileNotFoundError(ref)),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fallback route should not be attempted after local source image error")

    monkeypatch.setattr(openai_image.urllib.request, "urlopen", fail_if_called)
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(OpenAI=fail_if_called),
    )

    result = OpenAIImageGenProvider().generate("redraw this", image_url="/missing.png")

    assert result["success"] is False
    assert result["error_type"] == "io_error"
    assert result["provider_route"] == "gptcodex"
    assert result["fallback_used"] is False
    assert result["attempted_routes"] == ["gptcodex"]
