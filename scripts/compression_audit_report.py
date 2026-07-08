#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"event": "invalid_json", "raw_len": len(line)})
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            rows.append({"event": "invalid_json", "raw_len": len(line)})
    return rows


def _home(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    return Path.home() / ".hermes"


def _fallback_reason(row: dict[str, Any]) -> str | None:
    call = row.get("summary_call") or {}
    if not isinstance(call, dict):
        return None
    reason = call.get("fallback_reason")
    if reason:
        return str(reason)
    fallback_from = call.get("fallback_from") or {}
    if isinstance(fallback_from, dict):
        reason = fallback_from.get("fallback_reason")
        if reason:
            return str(reason)
    return None


def _summary_call(row: dict[str, Any]) -> dict[str, Any]:
    call = row.get("summary_call") or {}
    return call if isinstance(call, dict) else {}


def _summary_cache(row: dict[str, Any]) -> dict[str, Any]:
    cache = _summary_call(row).get("cache") or {}
    return cache if isinstance(cache, dict) else {}


def build_report(rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> str:
    compactions = [row for row in rows if row.get("event") == "context_compression"]
    modes = Counter(_summary_call(row).get("mode", "unknown") for row in compactions)
    fallbacks = Counter(reason for row in compactions if (reason := _fallback_reason(row)))
    cache_reported = 0
    hit_rates: list[float] = []
    read_tokens = 0
    write_tokens = 0
    total_saved = 0

    for row in compactions:
        cache = _summary_cache(row)
        if cache.get("reported"):
            cache_reported += 1
        if cache.get("hit_rate_estimate") is not None:
            hit_rates.append(float(cache["hit_rate_estimate"]))
        if cache.get("read_tokens") is not None:
            read_tokens += int(cache["read_tokens"])
        if cache.get("write_tokens") is not None:
            write_tokens += int(cache["write_tokens"])
        tokens = row.get("tokens") or {}
        if isinstance(tokens, dict) and tokens.get("saved_estimate") is not None:
            total_saved += int(tokens["saved_estimate"])
        elif isinstance(tokens, dict) and tokens.get("before_estimate") is not None and tokens.get("after_estimate") is not None:
            total_saved += int(tokens["before_estimate"]) - int(tokens["after_estimate"])

    mode_text = ", ".join(
        f"{mode}: {count}" for mode, count in sorted(modes.items())
    ) or "none"
    fallback_text = ", ".join(
        f"{reason}: {count}" for reason, count in sorted(fallbacks.items())
    ) or "none"
    lines = [
        "Compression audit report",
        f"records: {len(compactions)}",
        "modes: " + mode_text,
        f"cache reported: {cache_reported}/{len(compactions)}",
        f"cache read tokens: {read_tokens}",
        f"cache write tokens: {write_tokens}",
        "median cache hit rate: " + (f"{median(hit_rates):.4f}" if hit_rates else "n/a"),
        "fallback reasons: " + fallback_text,
        f"estimated token savings: {total_saved}",
        f"summary samples: {len(samples)}",
    ]
    return "\n".join(lines)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    session: str | None,
    compression_id: str | None,
) -> list[dict[str, Any]]:
    if session:
        rows = [row for row in rows if row.get("session_id") == session]
    if compression_id:
        rows = [row for row in rows if row.get("compression_id") == compression_id]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Read Hermes compression audit logs.")
    parser.add_argument("--home", default=None, help="Hermes home directory; defaults to ~/.hermes")
    parser.add_argument("--last", type=int, default=20, help="Number of recent context_compression records to summarize")
    parser.add_argument("--session", default=None, help="Filter by Hermes session id")
    parser.add_argument("--compression-id", default=None, help="Filter by compression_id")
    parser.add_argument("--show-summary", action="store_true", help="Print redacted summary excerpts from the samples sidecar")
    args = parser.parse_args()

    home = _home(args.home)
    rows = _read_jsonl(home / "logs" / "compression_audit.jsonl")
    samples = _read_jsonl(home / "logs" / "compression_summary_samples.jsonl")
    rows = _filter_rows(rows, session=args.session, compression_id=args.compression_id)
    samples = _filter_rows(samples, session=args.session, compression_id=args.compression_id)

    if args.last and not args.compression_id:
        compactions = [row for row in rows if row.get("event") == "context_compression"]
        selected_ids = {row.get("compression_id") for row in compactions[-args.last:]}
        rows = [
            row for row in rows
            if row.get("event") != "context_compression" or row.get("compression_id") in selected_ids
        ]
        samples = [row for row in samples if row.get("compression_id") in selected_ids]

    print(build_report(rows, samples))
    if args.show_summary:
        sample_by_id = {row.get("compression_id"): row for row in samples}
        for row in rows:
            if row.get("event") != "context_compression":
                continue
            sample = sample_by_id.get(row.get("compression_id"))
            if not sample:
                continue
            print("\n--- summary sample", row.get("compression_id"), "---")
            print(sample.get("summary_excerpt") or "")
            print(
                "section_check:",
                json.dumps(sample.get("section_check") or {}, ensure_ascii=False, sort_keys=True),
            )
            print("quality_flags:", json.dumps(sample.get("quality_flags") or [], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
