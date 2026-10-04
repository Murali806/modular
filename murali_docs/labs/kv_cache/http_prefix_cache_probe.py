#!/usr/bin/env python3
"""Measure shared-prefix behavior through a running MAX Serve endpoint."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_METRICS_URL = "http://127.0.0.1:18001/metrics"
DEFAULT_MODEL = "modularai/SmolLM-135M-Instruct-FP32"


def _post(base_url: str, path: str, payload: dict[str, Any]) -> bytes:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120.0) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return response.read()


def _reset_prefix_cache(base_url: str) -> int:
    request = urllib.request.Request(
        f"{base_url}/reset_prefix_cache", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        error.read()
        return error.code


def _metrics(metrics_url: str) -> str:
    with urllib.request.urlopen(metrics_url, timeout=10.0) as response:
        return response.read().decode("utf-8")


def _counter(metrics: str, name: str) -> float:
    match = re.search(
        rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", metrics, re.MULTILINE
    )
    return float(match.group(1)) if match else 0.0


def _cache_counters(metrics_url: str) -> tuple[float, float]:
    metrics = _metrics(metrics_url)
    return (
        _counter(metrics, 'maxserve_cache_hits_tokens_total{tier="g0"}'),
        _counter(metrics, "maxserve_cache_misses_tokens_total"),
    )


def _stream_completion(
    base_url: str, model: str, prompt: str
) -> dict[str, float | int]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    start = time.perf_counter()
    first_token_ms: float | None = None
    prompt_tokens: int | None = None
    with urllib.request.urlopen(request, timeout=120.0) as response:
        if response.status != 200:
            raise RuntimeError(f"completion returned HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line.removeprefix("data: "))
            if event.get("choices") and first_token_ms is None:
                first_token_ms = (time.perf_counter() - start) * 1000
            usage = event.get("usage")
            if usage:
                prompt_tokens = int(usage["prompt_tokens"])

    if first_token_ms is None or prompt_tokens is None:
        raise RuntimeError("stream omitted first-token timing or usage")
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_ms": round(first_token_ms, 3),
        "e2e_ms": round((time.perf_counter() - start) * 1000, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be positive")

    reset_status = _reset_prefix_cache(args.base_url)
    if args.mode == "enabled" and reset_status != 200:
        raise RuntimeError(
            f"expected enabled prefix cache, reset returned {reset_status}"
        )

    before_hits, before_misses = _cache_counters(args.metrics_url)
    trials: list[dict[str, Any]] = []
    for trial in range(args.trials):
        shared = f"trial {trial} " + "alpha beta gamma delta " * 38
        seed = _stream_completion(
            args.base_url, args.model, shared + "first suffix"
        )
        reused = _stream_completion(
            args.base_url, args.model, shared + "second suffix"
        )
        if seed["prompt_tokens"] != reused["prompt_tokens"]:
            raise RuntimeError("seed/shared prompts produced different lengths")
        trials.append({"trial": trial, "seed": seed, "shared": reused})

    after_hits, after_misses = _cache_counters(args.metrics_url)
    hit_delta = int(after_hits - before_hits)
    miss_delta = int(after_misses - before_misses)
    if args.mode == "enabled" and hit_delta <= 0:
        raise RuntimeError("enabled run produced no device-cache hits")
    if args.mode == "disabled" and hit_delta != 0:
        raise RuntimeError("disabled run unexpectedly produced cache hits")

    result = {
        "evidence": "CPU-RUN",
        "mode": args.mode,
        "trials": args.trials,
        "reset_status": reset_status,
        "prompt_tokens_per_request": trials[0]["seed"]["prompt_tokens"],
        "cache_counter_delta": {
            "g0_hit_tokens": hit_delta,
            "miss_tokens": miss_delta,
        },
        "median_ms": {
            "seed_ttft": round(
                statistics.median(row["seed"]["ttft_ms"] for row in trials),
                3,
            ),
            "shared_ttft": round(
                statistics.median(
                    row["shared"]["ttft_ms"] for row in trials
                ),
                3,
            ),
            "seed_e2e": round(
                statistics.median(row["seed"]["e2e_ms"] for row in trials),
                3,
            ),
            "shared_e2e": round(
                statistics.median(row["shared"]["e2e_ms"] for row in trials),
                3,
            ),
        },
        "samples": trials,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
