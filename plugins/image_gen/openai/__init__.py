"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three hit the same underlying API model (``gpt-image-2``) through the
Responses API with a different ``quality`` parameter. Output is read from the
``image_generation_call.result`` base64 field → saved under
``$HERMES_HOME/cache/images/``.

Selection precedence (first hit wins):

1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts / tests)
2. ``image_gen.openai.model`` in ``config.yaml``
3. ``image_gen.model`` in ``config.yaml`` (when it's one of our tier IDs)
4. :data:`DEFAULT_MODEL` — ``gpt-image-2-medium``
"""

from __future__ import annotations

import logging
import os
import base64
import json
import mimetypes
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# All three IDs resolve to the same underlying API model with a different
# ``quality`` setting. ``api_model`` is what gets sent to OpenAI;
# ``quality`` is the knob that changes generation time and output fidelity.

API_MODEL = "gpt-image-2"

_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-2-low": {
        "display": "GPT Image 2 (Low)",
        "speed": "~15s",
        "strengths": "Fast iteration, lowest cost",
        "quality": "low",
    },
    "gpt-image-2-medium": {
        "display": "GPT Image 2 (Medium)",
        "speed": "~40s",
        "strengths": "Balanced — default",
        "quality": "medium",
    },
    "gpt-image-2-high": {
        "display": "GPT Image 2 (High)",
        "speed": "~2min",
        "strengths": "Highest fidelity, strongest prompt adherence",
        "quality": "high",
    },
}

DEFAULT_MODEL = "gpt-image-2-medium"

_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


def _load_openai_config() -> Dict[str, Any]:
    """Read ``image_gen`` from config.yaml (returns {} on any failure)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _openai_subconfig() -> Dict[str, Any]:
    cfg = _load_openai_config()
    sub = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    return sub if isinstance(sub, dict) else {}


def _resolve_credentials() -> Tuple[Optional[str], Optional[str], str]:
    """Return ``(api_key, base_url, api_key_label)`` for the OpenAI-compatible backend.

    Defaults preserve the stock OpenAI plugin behavior.  Advanced users can bind
    image generation to an OpenAI-compatible gateway without clobbering their
    global ``OPENAI_API_KEY`` by setting::

        image_gen:
          openai:
            api_key_env: GPTCODEX_API_KEY
            base_url: https://gptcodex.top/v1
    """
    sub = _openai_subconfig()
    key_env = str(sub.get("api_key_env") or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
    inline_key = str(sub.get("api_key") or "").strip()
    api_key = inline_key or os.environ.get(key_env, "").strip()
    base_url = str(sub.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")).strip()
    return (api_key or None, base_url or None, key_env)


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which tier to use and return ``(model_id, meta)``."""
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate: Optional[str] = None
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            candidate = value
    if candidate is None:
        top = cfg.get("model")
        if isinstance(top, str) and top in _MODELS:
            candidate = top

    if candidate is not None:
        return candidate, _MODELS[candidate]

    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Source-image loading (for image-to-image / edit)
# ---------------------------------------------------------------------------


def _load_image_bytes(ref: str) -> Tuple[bytes, str]:
    """Load image bytes from a URL or local file path.

    Returns ``(data, filename)``. Raises on any network / IO error so the
    caller can surface a clean error_response.
    """
    ref = ref.strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://")):
        with urllib.request.urlopen(ref, timeout=60) as resp:
            data = resp.read()
        name = ref.split("?", 1)[0].rsplit("/", 1)[-1] or "image.png"
        return data, name
    if lower.startswith("data:"):
        import base64

        header, _, b64 = ref.partition(",")
        ext = "png"
        if "image/" in header:
            ext = header.split("image/", 1)[1].split(";", 1)[0] or "png"
        return base64.b64decode(b64), f"image.{ext}"
    # Local file path — enforce the shared credential-read guard before reading.
    from agent.file_safety import raise_if_read_blocked

    raise_if_read_blocked(ref)
    with open(ref, "rb") as fh:
        data = fh.read()
    name = os.path.basename(ref) or "image.png"
    return data, name


def _image_ref_to_data_url(ref: str) -> str:
    """Return a data URL suitable for Responses API ``input_image`` items."""
    data, filename = _load_image_bytes(ref)
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _responses_endpoint(base_url: Optional[str]) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    return f"{base}/responses"


def _build_responses_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    sources: List[str],
) -> Dict[str, Any]:
    """Build the OpenAI-compatible Responses request body.

    The gptcodex image route that works in practice accepts the image model as
    the top-level Responses model and returns an ``image_generation_call`` item.
    Keep the payload intentionally small: the older ``/images`` endpoints and
    the explicit ``tools=[image_generation]`` shape have both been observed to
    fail or downgrade quality on that gateway.
    """
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for ref in sources:
        content.append({"type": "input_image", "image_url": _image_ref_to_data_url(ref)})
    return {
        "model": API_MODEL,
        "input": [{"role": "user", "content": content}],
        "size": size,
        "quality": quality,
    }


def _extract_responses_image(value: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Return ``(b64, metadata)`` from a Responses image_generation payload."""
    if isinstance(value, dict):
        if value.get("type") == "image_generation_call":
            result = value.get("result")
            if isinstance(result, str) and result:
                return result, {
                    "actual_quality": value.get("quality"),
                    "actual_size": value.get("size"),
                    "width": value.get("width"),
                    "height": value.get("height"),
                    "revised_prompt": value.get("revised_prompt"),
                }
        for child in value.values():
            found = _extract_responses_image(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _extract_responses_image(child)
            if found:
                return found
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI-compatible ``/responses`` image-generation backend — gpt-image-2."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        api_key, _, _ = _resolve_credentials()
        return bool(api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model_id,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": "varies",
            }
            for model_id, meta in _MODELS.items()
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OpenAI",
            "badge": "paid",
            "tag": "gpt-image-2 at low/medium/high quality tiers — text-to-image & image editing",
            "env_vars": [
                {
                    "key": "OPENAI_API_KEY",
                    "prompt": "OpenAI API key",
                    "url": "https://platform.openai.com/api-keys",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        # gpt-image-2 supports image-conditioned generation via Responses API
        # input_image items. Keep the public surface as image-to-image/editing.
        return {"modalities": ["text", "image"], "max_reference_images": 16}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        api_key, base_url, api_key_label = _resolve_credentials()
        if not api_key:
            return error_response(
                error=(
                    f"{api_key_label} not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or set "
                    "image_gen.openai.api_key_env / image_gen.openai.api_key "
                    "in config.yaml."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        tier_id, meta = _resolve_model()
        size = _SIZES.get(aspect, _SIZES["square"])

        # Collect source images (primary + references) for image-to-image.
        sources: List[str] = []
        if isinstance(image_url, str) and image_url.strip():
            sources.append(image_url.strip())
        for ref in (normalize_reference_images(reference_image_urls) or []):
            sources.append(ref)
        sources = sources[:16]  # gpt-image-2 input_image cap
        is_edit = bool(sources)
        modality = "image" if is_edit else "text"

        try:
            payload = _build_responses_payload(
                prompt=prompt,
                size=size,
                quality=meta["quality"],
                sources=sources,
            )
        except Exception as exc:
            return error_response(
                error=f"Could not load source image for image generation: {exc}",
                error_type="io_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        request = urllib.request.Request(
            _responses_endpoint(base_url),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=420) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read(1000).decode("utf-8", errors="replace")
            logger.debug("OpenAI Responses image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI Responses image generation failed: HTTP {exc.code}: {body}",
                error_type="api_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.debug("OpenAI Responses image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI Responses image generation failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extracted = _extract_responses_image(response_payload)
        if not extracted:
            return error_response(
                error="OpenAI Responses payload contained no image_generation_call result",
                error_type="empty_response",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        b64, response_meta = extracted
        try:
            saved_path = save_b64_image(b64, prefix=f"openai_{tier_id}")
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="openai",
                model=tier_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        image_ref = str(saved_path)

        extra: Dict[str, Any] = {"size": size, "quality": meta["quality"]}
        for key, value in response_meta.items():
            if value is not None:
                extra[key] = value

        return success_response(
            image=image_ref,
            model=tier_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            modality=modality,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
