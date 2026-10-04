#!/usr/bin/env python3
"""Reduce MAX benchmark JSON files to a compact learning summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _histogram(result: dict[str, Any], name: str) -> dict[str, Any]:
    return result.get("server_metrics", {}).get("histograms", {}).get(name, {})


def _counter(result: dict[str, Any], name: str) -> float | None:
    value = result.get("server_metrics", {}).get("counters", {}).get(name)
    return float(value) if value is not None else None


def _round(value: Any) -> float | None:
    return round(float(value), 3) if value is not None else None


def _accounting(result: dict[str, Any]) -> dict[str, Any]:
    records = result.get("request_records", [])
    offered = len(records) or int(result.get("num_prompts", 0))
    completed = sum(bool(row.get("success")) for row in records)
    cancelled = sum(bool(row.get("cancelled")) for row in records)
    errors = [
        str(row.get("error", "")).lower()
        for row in records
        if not row.get("success")
    ]
    rejected = sum("429" in error for error in errors)
    timed_out = sum(
        "timeout" in error or "timed out" in error for error in errors
    )
    unclassified = max(
        0, offered - completed - cancelled - rejected - timed_out
    )
    accepted = offered if completed == offered else None
    return {
        "offered": offered,
        "accepted": accepted,
        "completed": completed,
        "rejected_429": rejected,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "unclassified_failures": unclassified,
        "accepted_note": (
            "exact: every offered request completed"
            if accepted is not None
            else "null: add server-side admission evidence for failed runs"
        ),
    }


def _summarize(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    duration = float(result["duration"])
    server_output_tokens = _counter(result, "maxserve_num_output_tokens")
    aggregate_output_tps = (
        server_output_tokens / duration
        if server_output_tokens is not None and duration > 0
        else None
    )
    return {
        "max_concurrency": result.get("max_concurrency"),
        "accounting": _accounting(result),
        "duration_s": _round(duration),
        "request_throughput_per_s": _round(result.get("request_throughput")),
        "server_output_tokens": _round(server_output_tokens),
        "aggregate_output_tokens_per_s": _round(aggregate_output_tps),
        "latency_ms": {
            "ttft_p50": _round(result.get("median_ttft_ms")),
            "ttft_p95": _round(result.get("p95_ttft_ms")),
            "tpot_p50": _round(result.get("median_tpot_ms")),
            "tpot_p95": _round(result.get("p95_tpot_ms")),
            "itl_p50": _round(result.get("median_itl_ms")),
            "itl_p95": _round(result.get("p95_itl_ms")),
        },
        "server": {
            "input_tokens": _round(
                _counter(result, "maxserve_num_input_tokens")
            ),
            "cache_hit_tokens": _round(
                _counter(result, 'maxserve_cache_hits_tokens{tier="g0"}')
            ),
            "cache_miss_tokens": _round(
                _counter(result, "maxserve_cache_misses_tokens")
            ),
            "mean_ce_batch_size": _round(
                _histogram(result, 'maxserve_batch_size{batch_type="CE"}').get(
                    "mean"
                )
            ),
            "mean_tg_batch_size": _round(
                _histogram(result, 'maxserve_batch_size{batch_type="TG"}').get(
                    "mean"
                )
            ),
            "mean_kv_used_percent": _round(
                _histogram(result, "maxserve_cache_used_kv_pct_percent").get(
                    "mean"
                )
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    runs = sorted(
        (_summarize(path) for path in args.results),
        key=lambda row: int(row["max_concurrency"]),
    )
    output = {
        "evidence": "CPU-RUN",
        "scope": "semantic smoke test; not GPU capacity data",
        "runs": runs,
    }
    print(
        json.dumps(output, indent=None if args.compact else 2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
