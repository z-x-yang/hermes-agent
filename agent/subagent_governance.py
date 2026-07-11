"""Canonical, byte-preserving governance snapshots for delegated agents."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn

from hermes_cli.profiles import get_active_profile_name
from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class GovernanceSource:
    label: str
    path: Path
    text: str
    byte_length: int
    sha256: str


@dataclass(frozen=True)
class GovernanceSnapshot:
    profile_id: str
    profile_home: Path
    soul: GovernanceSource
    memory: GovernanceSource
    user: GovernanceSource
    fingerprint: str
    total_bytes: int


@dataclass(frozen=True)
class GovernanceRequestFit:
    serialized_utf8_bytes: int
    input_token_upper_bound: int
    output_reserve_tokens: int
    context_limit_tokens: int


class GovernancePreflightError(RuntimeError):
    """Deterministic fail-closed denial before a governed child backend call."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


_INTERNAL_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"timeout", "__bedrock_region__", "__bedrock_converse__"}
)
_VERIFIABLE_API_MODES: Final[frozenset[str]] = frozenset(
    {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse"}
)
_OUTPUT_CAP_FIELDS: Final[tuple[str, ...]] = (
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
)


def _deny(code: str) -> NoReturn:
    raise GovernancePreflightError(code)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def assert_governance_request_fits(
    agent: Any, api_kwargs: dict[str, Any]
) -> GovernanceRequestFit | None:
    """Prove a governed child's final provider request fits this attempt.

    The caller supplies the post-transport, post-request-middleware payload.
    Request/governance bodies are never logged or retained in diagnostics.
    """
    governance = getattr(agent, "_governance_diagnostics", None)
    if not isinstance(governance, dict) or not governance.get("fingerprint"):
        return None
    if getattr(agent, "api_mode", None) not in _VERIFIABLE_API_MODES:
        _deny("governance_transport_unverifiable")

    provider_request = {
        key: value
        for key, value in api_kwargs.items()
        if key not in _INTERNAL_REQUEST_FIELDS
    }
    try:
        serialized = json.dumps(
            provider_request,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        _deny("governance_transport_unverifiable")

    output_reserve = None
    for field in _OUTPUT_CAP_FIELDS:
        if field in provider_request:
            output_reserve = _positive_int(provider_request[field])
            break
    if output_reserve is None and not any(
        field in provider_request for field in _OUTPUT_CAP_FIELDS
    ):
        output_reserve = _positive_int(getattr(agent, "max_tokens", None))
    if output_reserve is None:
        _deny("governance_transport_unverifiable")

    compressor = getattr(agent, "context_compressor", None)
    compressor_limit = _positive_int(getattr(compressor, "context_length", None))
    if compressor_limit is None:
        _deny("governance_transport_unverifiable")

    from agent.model_metadata import get_verified_model_context_length

    route_proof = getattr(agent, "_governance_context_limit_proof", None)
    if (
        isinstance(route_proof, dict)
        and route_proof.get("model") == getattr(agent, "model", None)
        and route_proof.get("provider") == getattr(agent, "provider", None)
        and route_proof.get("base_url") == getattr(agent, "base_url", None)
        and route_proof.get("api_mode") == getattr(agent, "api_mode", None)
    ):
        proof = route_proof.get("limit")
    else:
        api_key = getattr(agent, "api_key", "")
        proof = get_verified_model_context_length(
            str(getattr(agent, "model", "") or ""),
            base_url=str(getattr(agent, "base_url", "") or ""),
            api_key=api_key if isinstance(api_key, str) else "",
            config_context_length=getattr(agent, "_config_context_length", None),
            provider=str(getattr(agent, "provider", "") or ""),
            custom_providers=getattr(agent, "_custom_providers", None),
        )
    if proof is None or _positive_int(proof.tokens) != compressor_limit:
        _deny("governance_transport_unverifiable")

    serialized_bytes = len(serialized)
    input_upper_bound = serialized_bytes + 2_048
    fit = GovernanceRequestFit(
        serialized_utf8_bytes=serialized_bytes,
        input_token_upper_bound=input_upper_bound,
        output_reserve_tokens=output_reserve,
        context_limit_tokens=compressor_limit,
    )
    request_fingerprint = hashlib.sha256(serialized).hexdigest()
    agent._governance_request_fit_diagnostics = {
        "serialized_utf8_bytes": serialized_bytes,
        "input_token_upper_bound": input_upper_bound,
        "output_reserve_tokens": output_reserve,
        "context_limit_tokens": compressor_limit,
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
        "api_mode": str(getattr(agent, "api_mode", "") or ""),
        "context_limit_source": proof.source,
        "request_fingerprint": request_fingerprint,
        "governance_fingerprint": str(governance["fingerprint"]),
    }
    if input_upper_bound + output_reserve > compressor_limit:
        _deny("governance_context_too_large")
    return fit


class GovernanceSnapshotError(RuntimeError):
    """Raised when a complete, stable governance snapshot cannot be loaded."""


@dataclass(frozen=True)
class _StableRead:
    raw: bytes
    identity: tuple[int, int, int, int] | None


_SOURCE_PATHS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("SOUL.md", ("SOUL.md",)),
    ("MEMORY.md", ("memories", "MEMORY.md")),
    ("USER.md", ("memories", "USER.md")),
)


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _stat_path(path: Path) -> os.stat_result:
    try:
        return path.stat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise GovernanceSnapshotError(
            f"failed to stat governance source {path}: errno={exc.errno}"
        ) from exc


def _stable_read_once(path: Path) -> _StableRead:
    """Read one source while proving path/file identity remained stable."""
    try:
        before = _stat_path(path)
    except FileNotFoundError:
        try:
            _stat_path(path)
        except FileNotFoundError:
            return _StableRead(raw=b"", identity=None)
        raise GovernanceSnapshotError(f"{path} changed during read: appeared")

    try:
        with path.open("rb") as source_file:
            raw = source_file.read()
            opened = os.fstat(source_file.fileno())
    except FileNotFoundError as exc:
        raise GovernanceSnapshotError(
            f"{path} changed during read: disappeared before open"
        ) from exc
    except OSError as exc:
        raise GovernanceSnapshotError(
            f"failed to read governance source {path}: errno={exc.errno}"
        ) from exc

    try:
        after = _stat_path(path)
    except FileNotFoundError as exc:
        raise GovernanceSnapshotError(
            f"{path} changed during read: disappeared after open"
        ) from exc

    before_identity = _stat_identity(before)
    opened_identity = _stat_identity(opened)
    after_identity = _stat_identity(after)
    if not (
        before_identity == opened_identity == after_identity
        and len(raw) == opened.st_size
    ):
        raise GovernanceSnapshotError(
            f"{path} changed during read: "
            f"before={before_identity}, opened={opened_identity}, "
            f"after={after_identity}, bytes_read={len(raw)}"
        )
    return _StableRead(raw=raw, identity=after_identity)


def _validate_complete_attempt(
    reads: tuple[tuple[Path, _StableRead], ...],
) -> None:
    """Prove every source still has the state tied to this attempt's bytes."""
    for path, read in reads:
        try:
            current = _stat_path(path)
        except FileNotFoundError:
            if read.identity is None:
                continue
            raise GovernanceSnapshotError(
                f"{path} changed during snapshot attempt: disappeared"
            ) from None

        current_identity = _stat_identity(current)
        if read.identity is None:
            raise GovernanceSnapshotError(
                f"{path} changed during snapshot attempt: appeared"
            )
        if current_identity != read.identity:
            raise GovernanceSnapshotError(
                f"{path} changed during snapshot attempt: "
                f"read={read.identity}, final={current_identity}"
            )


def _make_source(label: str, path: Path, raw: bytes) -> GovernanceSource:
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise GovernanceSnapshotError(
            f"governance source {path} is not strict UTF-8: "
            f"byte_length={len(raw)}, sha256={digest}"
        ) from None
    return GovernanceSource(
        label=label,
        path=path,
        text=text,
        byte_length=len(raw),
        sha256=digest,
    )


def _snapshot_fingerprint(
    profile_id: str, sources: tuple[GovernanceSource, ...]
) -> str:
    metadata = {
        "profile_id": profile_id,
        "sources": [
            {
                "path": str(source.path),
                "byte_length": source.byte_length,
                "sha256": source.sha256,
            }
            for source in sources
        ],
    }
    encoded = json.dumps(
        metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_governance_snapshot(
    *,
    profile_home: Path | None = None,
    profile_id: str | None = None,
    retry_limit: int = 1,
) -> GovernanceSnapshot:
    """Load one canonical three-file governance snapshot.

    Any unstable or failed attempt is discarded in full. Governance contents are
    never stripped, scanned, replaced, truncated, deduplicated, or logged.
    """
    if retry_limit < 0:
        raise ValueError("retry_limit must be non-negative")

    canonical_home = Path(
        get_hermes_home() if profile_home is None else profile_home
    ).expanduser().resolve()
    canonical_profile_id = (
        get_active_profile_name() if profile_id is None else profile_id
    )
    source_specs = tuple(
        (label, canonical_home.joinpath(*relative_path).resolve())
        for label, relative_path in _SOURCE_PATHS
    )

    last_error: GovernanceSnapshotError | None = None
    for _attempt in range(retry_limit + 1):
        try:
            stable_reads = tuple(
                (path, _stable_read_once(path)) for _label, path in source_specs
            )
            sources = tuple(
                _make_source(label, path, read.raw)
                for (label, path), (_read_path, read) in zip(
                    source_specs, stable_reads
                )
            )
            _validate_complete_attempt(stable_reads)
        except GovernanceSnapshotError as exc:
            last_error = exc
            continue

        soul, memory, user = sources
        return GovernanceSnapshot(
            profile_id=canonical_profile_id,
            profile_home=canonical_home,
            soul=soul,
            memory=memory,
            user=user,
            fingerprint=_snapshot_fingerprint(canonical_profile_id, sources),
            total_bytes=sum(source.byte_length for source in sources),
        )

    assert last_error is not None
    raise last_error
