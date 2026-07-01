"""OpenAI image generation backend.

Exposes OpenAI's ``gpt-image-2`` model at three quality tiers as an
:class:`ImageGenProvider` implementation. The tiers are implemented as
three virtual model IDs so the ``hermes tools`` model picker and the
``image_gen.model`` config key behave like any other multi-model backend:

    gpt-image-2-low     ~15s   fastest, good for iteration
    gpt-image-2-medium  ~40s   default — balanced
    gpt-image-2-high    ~2min  slowest, highest fidelity

All three use the same underlying API model (``gpt-image-2``) with a
different ``quality`` parameter. Stock OpenAI uses the Images API. GPTCodex
compatibility mode uses the observed-working Responses API shape and reads
``image_generation_call.result`` base64 output. Images are saved under
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
    save_url_image,
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
RESPONSES_TEXT_MODEL = "gpt-5.5"
RESPONSES_IMAGE_INSTRUCTIONS = (
    "You are a tool runner. Pass the user prompt to image_generation VERBATIM. "
    "DO NOT rewrite, expand, polish, or revise it in any way. "
    "Use the exact text the user gave."
)

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
    # True 16:9 landscape. The public tool description promises landscape
    # output is wide; OpenAI's Images API accepts 1536x864 for gpt-image-2.
    "landscape": "1536x864",
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
    api_key = os.environ.get(key_env, "").strip()
    base_url = str(sub.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")).strip()
    return (api_key or None, base_url or None, key_env)


def _resolve_route_credentials(route: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    """Return credentials for one explicit fallback route."""
    raw_sub = route.get("openai")
    sub: Dict[str, Any] = raw_sub if isinstance(raw_sub, dict) else {}
    key_env = str(sub.get("api_key_env") or "OPENAI_API_KEY").strip() or "OPENAI_API_KEY"
    api_key = os.environ.get(key_env, "").strip()
    if "base_url" in sub:
        base_url = str(sub.get("base_url") or "").strip()
    else:
        base_url = str(os.environ.get("OPENAI_BASE_URL", "")).strip()
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


def _resolve_route_model(route: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Resolve the model tier for a configured route."""
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _MODELS:
        return env_override, _MODELS[env_override]

    openai_cfg = route.get("openai") if isinstance(route.get("openai"), dict) else {}
    if isinstance(openai_cfg, dict):
        value = openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            return value, _MODELS[value]
    value = route.get("model")
    if isinstance(value, str) and value in _MODELS:
        return value, _MODELS[value]

    cfg = _load_openai_config()
    global_openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    if isinstance(global_openai_cfg, dict):
        value = global_openai_cfg.get("model")
        if isinstance(value, str) and value in _MODELS:
            return value, _MODELS[value]
    value = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(value, str) and value in _MODELS:
        return value, _MODELS[value]
    return DEFAULT_MODEL, _MODELS[DEFAULT_MODEL]


def _configured_routes() -> List[Dict[str, Any]]:
    """Return explicit OpenAI fallback routes from ``image_gen.fallbacks``."""
    cfg = _load_openai_config()
    raw = cfg.get("fallbacks") if isinstance(cfg, dict) else None
    if not isinstance(raw, list):
        return []

    routes: List[Dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "openai").strip().lower()
        if provider != "openai":
            logger.warning(
                "Ignoring non-OpenAI image generation fallback route at position %s: %s",
                index,
                provider,
            )
            continue
        routes.append(item)
    return routes


def _route_name(route: Dict[str, Any], index: int) -> str:
    name = route.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"openai-route-{index}"


def _use_responses_api_for_route(route: Dict[str, Any], base_url: Optional[str]) -> bool:
    sub = route.get("openai") if isinstance(route.get("openai"), dict) else {}
    if isinstance(sub, dict):
        configured = sub.get("use_responses_api")
    else:
        configured = None
    if isinstance(configured, bool):
        return configured
    if isinstance(configured, str):
        value = configured.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return "gptcodex.top" in (base_url or "").lower()


_FALLBACK_ERROR_TYPES = {"api_error", "empty_response", "auth_required", "missing_dependency"}


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


def _use_responses_api(base_url: Optional[str]) -> bool:
    """Return true for gateways that need the observed GPTCodex Responses shape."""
    sub = _openai_subconfig()
    configured = sub.get("use_responses_api")
    if isinstance(configured, bool):
        return configured
    if isinstance(configured, str):
        value = configured.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return "gptcodex.top" in (base_url or "").lower()


def _resolve_responses_text_model(route: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the text model that should call the Responses image tool."""
    candidates: List[Any] = []
    if route is not None:
        sub = route.get("openai") if isinstance(route.get("openai"), dict) else {}
        if isinstance(sub, dict):
            candidates.extend([sub.get("responses_text_model"), sub.get("text_model")])
    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    if isinstance(openai_cfg, dict):
        candidates.extend([openai_cfg.get("responses_text_model"), openai_cfg.get("text_model")])
    candidates.append(cfg.get("responses_text_model") if isinstance(cfg, dict) else None)
    candidates.append(os.environ.get("OPENAI_IMAGE_RESPONSES_TEXT_MODEL"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return RESPONSES_TEXT_MODEL


def _build_responses_payload(
    *,
    prompt: str,
    size: str,
    quality: str,
    sources: List[str],
    text_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Responses image-generation-tool request body.

    GPTCodex-compatible image routes require a text model to invoke the
    ``image_generation`` tool over streaming Responses. Sending ``gpt-image-2``
    as the top-level Responses model can complete with only text output and no
    image result on that relay.
    """
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for ref in sources:
        content.append({"type": "input_image", "image_url": _image_ref_to_data_url(ref)})
    action = "edit" if sources else "generate"
    tool: Dict[str, Any] = {
        "type": "image_generation",
        "model": API_MODEL,
        "action": action,
        "size": size,
        "quality": quality,
        "output_format": "png",
        "partial_images": 0,
        "background": "auto",
        "moderation": "low",
    }
    return {
        "model": (text_model or RESPONSES_TEXT_MODEL).strip() or RESPONSES_TEXT_MODEL,
        "input": [{"role": "user", "content": content}],
        "tools": [tool],
        "tool_choice": {"type": "image_generation"},
        "reasoning": {"effort": "low"},
        "store": False,
        "stream": True,
        "instructions": RESPONSES_IMAGE_INSTRUCTIONS,
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


def _extract_responses_image_from_text(raw: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Extract a Responses image result from JSON or SSE text."""
    try:
        return _extract_responses_image(json.loads(raw))
    except Exception:
        pass

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except Exception:
            continue
        found = _extract_responses_image(event)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI GPT Image backend with GPTCodex Responses compatibility."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        routes = _configured_routes()
        if routes:
            return any(bool(_resolve_route_credentials(route)[0]) for route in routes)
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
        # gpt-image-2 supports image-to-image/editing via the stock Images API;
        # GPTCodex compatibility mode maps the same public surface to Responses
        # input_image items.
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

        routes = _configured_routes()
        if not routes:
            return self._generate_once(
                prompt,
                aspect,
                image_url=image_url,
                reference_image_urls=reference_image_urls,
                **kwargs,
            )

        attempted_routes: List[str] = []
        route_failures: List[Dict[str, str]] = []
        last_result: Optional[Dict[str, Any]] = None

        for index, route in enumerate(routes, start=1):
            name = _route_name(route, index)
            attempted_routes.append(name)
            result = self._generate_once(
                prompt,
                aspect,
                image_url=image_url,
                reference_image_urls=reference_image_urls,
                route=route,
                **kwargs,
            )
            result["provider_route"] = name
            result["fallback_used"] = bool(route_failures)
            result["attempted_routes"] = list(attempted_routes)
            if route_failures:
                result["route_failures"] = list(route_failures)

            if result.get("success"):
                if route_failures:
                    result["route_failures"] = list(route_failures)
                    logger.warning(
                        "OpenAI image generation succeeded on fallback route %s after failures: %s",
                        name,
                        route_failures,
                    )
                return result

            error_type = str(result.get("error_type") or "provider_error")
            if error_type not in _FALLBACK_ERROR_TYPES:
                return result

            failure = {
                "route": name,
                "error_type": error_type,
                "error": str(result.get("error") or ""),
            }
            route_failures.append(failure)
            result["route_failures"] = list(route_failures)
            last_result = result

            if index < len(routes):
                next_name = _route_name(routes[index], index + 1)
                logger.warning(
                    "OpenAI image generation route %s failed (%s): %s; falling back to route %s",
                    name,
                    error_type,
                    failure["error"],
                    next_name,
                )
            else:
                logger.warning(
                    "OpenAI image generation route %s failed (%s): %s; no fallback routes remain",
                    name,
                    error_type,
                    failure["error"],
                )

        final = last_result or error_response(
            error="All OpenAI image generation routes failed",
            error_type="provider_error",
            provider="openai",
            prompt=prompt,
            aspect_ratio=aspect,
        )
        if route_failures:
            failure_summary = "; ".join(
                f"{failure['route']} ({failure['error_type']}): {failure['error']}"
                for failure in route_failures
            )
            final["error"] = f"All OpenAI image generation routes failed. Route failures: {failure_summary}"
        else:
            final["error"] = f"All OpenAI image generation routes failed. Last error: {final.get('error', '')}"
        final["provider_route"] = attempted_routes[-1] if attempted_routes else ""
        final["fallback_used"] = len(attempted_routes) > 1
        final["attempted_routes"] = list(attempted_routes)
        final["route_failures"] = list(route_failures)
        return final

    def _generate_once(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        route: Optional[Dict[str, Any]] = None,
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

        if route is None:
            api_key, base_url, api_key_label = _resolve_credentials()
        else:
            api_key, base_url, api_key_label = _resolve_route_credentials(route)
        if not api_key:
            return error_response(
                error=(
                    f"{api_key_label} not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or set "
                    "image_gen.openai.api_key_env in config.yaml."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        if route is None:
            tier_id, meta = _resolve_model()
        else:
            tier_id, meta = _resolve_route_model(route)
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

        use_responses = (
            _use_responses_api(base_url)
            if route is None
            else _use_responses_api_for_route(route, base_url)
        )
        if use_responses:
            try:
                payload = _build_responses_payload(
                    prompt=prompt,
                    size=size,
                    quality=meta["quality"],
                    sources=sources,
                    text_model=_resolve_responses_text_model(route),
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
                    "Accept": "text/event-stream",
                },
            )

            try:
                with urllib.request.urlopen(request, timeout=420) as response:
                    response_text = response.read().decode("utf-8", errors="replace")
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

            extracted = _extract_responses_image_from_text(response_text)
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
        else:
            try:
                import openai
            except ImportError:
                return error_response(
                    error="openai Python package not installed (pip install openai)",
                    error_type="missing_dependency",
                    provider="openai",
                    aspect_ratio=aspect,
                )

            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)

            if is_edit:
                # images.edit() expects file-like objects. Download/read each
                # source into a named BytesIO so the SDK sends correct multipart.
                import io

                try:
                    files = []
                    for ref in sources:
                        data, fname = _load_image_bytes(ref)
                        bio = io.BytesIO(data)
                        bio.name = fname
                        files.append(bio)
                except Exception as exc:
                    return error_response(
                        error=f"Could not load source image for editing: {exc}",
                        error_type="io_error",
                        provider="openai",
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )

                try:
                    response = client.images.edit(
                        model=API_MODEL,
                        image=files if len(files) > 1 else files[0],
                        prompt=prompt,
                        size=size,  # type: ignore[arg-type]  # _SIZES values are valid gpt-image sizes
                        quality=meta["quality"],
                        n=1,
                    )
                except Exception as exc:
                    logger.debug("OpenAI image edit failed", exc_info=True)
                    return error_response(
                        error=f"OpenAI image editing failed: {exc}",
                        error_type="api_error",
                        provider="openai",
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
            else:
                # gpt-image-2 returns b64_json unconditionally and REJECTS
                # ``response_format`` as an unknown parameter. Don't send it.
                payload: Dict[str, Any] = {
                    "model": API_MODEL,
                    "prompt": prompt,
                    "size": size,
                    "n": 1,
                    "quality": meta["quality"],
                }

                try:
                    response = client.images.generate(**payload)
                except Exception as exc:
                    logger.debug("OpenAI image generation failed", exc_info=True)
                    return error_response(
                        error=f"OpenAI image generation failed: {exc}",
                        error_type="api_error",
                        provider="openai",
                        model=tier_id,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )

            data = getattr(response, "data", None) or []
            if not data:
                return error_response(
                    error="OpenAI returned no image data",
                    error_type="empty_response",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            first = data[0]
            b64 = getattr(first, "b64_json", None)
            url = getattr(first, "url", None)
            revised_prompt = getattr(first, "revised_prompt", None)

            if b64:
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
            elif url:
                # Defensive — gpt-image-2 returns b64 today, but OpenAI's API
                # has previously returned URLs. Cache the bytes locally so the
                # gateway never tries to fetch an ephemeral / signed URL after
                # it expires — same rationale as the xAI provider (#26942).
                try:
                    saved_path = save_url_image(url, prefix=f"openai_{tier_id}")
                except Exception as exc:
                    logger.warning(
                        "OpenAI image URL %s could not be cached (%s); falling back to bare URL.",
                        url,
                        exc,
                    )
                    image_ref = url
                else:
                    image_ref = str(saved_path)
            else:
                return error_response(
                    error="OpenAI response contained neither b64_json nor URL",
                    error_type="empty_response",
                    provider="openai",
                    model=tier_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            extra = {"size": size, "quality": meta["quality"]}
            if revised_prompt:
                extra["revised_prompt"] = revised_prompt

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
