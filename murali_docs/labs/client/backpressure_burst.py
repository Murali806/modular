#!/usr/bin/env python3
"""Send a synchronized burst to a bounded MAX Serve instance."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_METRICS_URL = "http://127.0.0.1:18001/metrics"
DEFAULT_MODEL = "modularai/SmolLM-135M-Instruct-FP32"
DEFAULT_PROMPT_IDS = [
    1,
    4093,
    198,
    5820,
    827,
    4683,
    30,
    2,
    198,
    1,
    520,
    9531,
    198,
]
GAUGES = (
    "maxserve_num_requests_awaiting_admission",
    "maxserve_num_requests_queued",
    "maxserve_num_requests_running",
    "maxserve_num_responses_buffered",
)


def _get(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except urllib.error.URLError:
        return 0, b""


def _metrics(url: str) -> dict[str, float]:
    status, body = _get(url)
    if status != 200:
        return {}
    values: dict[str, float] = {}
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, raw_value = line.rpartition(" ")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        for gauge in GAUGES:
            if key == gauge:
                values[gauge] = value
        if key.startswith("maxserve_request_count_total{"):
            if 'code="429"' in key and 'path="/v1/chat/completions"' in key:
                values["chat_429_total"] = value
    return values


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return round(ordered[index], 2)


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min_ms": round(min(values), 2) if values else None,
        "p50_ms": round(statistics.median(values), 2) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 2) if values else None,
    }


def _send_one(
    index: int,
    barrier: threading.Barrier,
    url: str,
    payload: bytes,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    barrier.wait()
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            headers = dict(response.headers)
            body = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        headers = dict(error.headers)
        body = error.read()
    except Exception as error:
        return {
            "index": index,
            "status": "transport_error",
            "latency_ms": round((time.monotonic() - start) * 1000, 2),
            "error": f"{type(error).__name__}: {error}",
        }

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError:
        decoded = {}
    return {
        "index": index,
        "status": status,
        "latency_ms": round((time.monotonic() - start) * 1000, 2),
        "retry_after": headers.get("Retry-After") or headers.get("retry-after"),
        "request_id": headers.get("X-Request-ID")
        or headers.get("x-request-id"),
        "error_type": decoded.get("error", {}).get("type"),
        "error_code": decoded.get("error", {}).get("code"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--pause-worker-pid",
        type=int,
        help="SIGSTOP this lab worker, then always SIGCONT it before exit.",
    )
    parser.add_argument("--pause-seconds", type=float, default=8.0)
    parser.add_argument("--require-429", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    health_before, _ = _get(f"{args.base_url}/health")
    metrics_before = _metrics(args.metrics_url)
    samples: list[dict[str, float]] = []
    stop_sampling = threading.Event()

    def sample_metrics() -> None:
        while not stop_sampling.wait(0.1):
            sample = _metrics(args.metrics_url)
            if sample:
                samples.append(sample)

    sampler = threading.Thread(target=sample_metrics, daemon=True)
    sampler.start()

    payload = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "ignored"}],
            "prompt_tokens": DEFAULT_PROMPT_IDS,
            "temperature": 0,
            "top_k": 1,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
            "stream": False,
        }
    ).encode("utf-8")
    barrier = threading.Barrier(args.requests + 1)
    resume_timer: threading.Timer | None = None

    worker_paused = False
    try:
        if args.pause_worker_pid is not None:
            os.kill(args.pause_worker_pid, signal.SIGSTOP)
            worker_paused = True
        with ThreadPoolExecutor(max_workers=args.requests) as executor:
            futures = [
                executor.submit(
                    _send_one,
                    index,
                    barrier,
                    f"{args.base_url}/v1/chat/completions",
                    payload,
                    args.timeout,
                )
                for index in range(args.requests)
            ]
            barrier.wait()
            if args.pause_worker_pid is not None:
                resume_timer = threading.Timer(
                    args.pause_seconds,
                    lambda: os.kill(args.pause_worker_pid, signal.SIGCONT),
                )
                resume_timer.start()
            rows = [future.result() for future in as_completed(futures)]
    finally:
        if resume_timer is not None:
            resume_timer.cancel()
        if worker_paused:
            with contextlib.suppress(ProcessLookupError):
                assert args.pause_worker_pid is not None
                os.kill(args.pause_worker_pid, signal.SIGCONT)
        stop_sampling.set()
        sampler.join(timeout=2.0)

    metrics_after = _metrics(args.metrics_url)
    health_after, _ = _get(f"{args.base_url}/health")
    rows.sort(key=lambda item: item["index"])
    status_counts = Counter(str(item["status"]) for item in rows)
    accepted = [
        float(item["latency_ms"]) for item in rows if item["status"] == 200
    ]
    rejected = [
        float(item["latency_ms"]) for item in rows if item["status"] == 429
    ]
    peak_gauges = {
        gauge: max((sample.get(gauge, 0.0) for sample in samples), default=0.0)
        for gauge in GAUGES
    }
    before_429 = metrics_before.get("chat_429_total", 0.0)
    after_429 = metrics_after.get("chat_429_total", 0.0)

    result = {
        "evidence": "CPU-RUN",
        "configuration": {
            "requests": args.requests,
            "max_tokens": args.max_tokens,
            "worker_pid_paused": args.pause_worker_pid,
            "worker_pause_seconds": (
                args.pause_seconds if args.pause_worker_pid is not None else 0
            ),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "accepted_latency": _latency_summary(accepted),
        "rejected_latency": _latency_summary(rejected),
        "rejection_contract": {
            "retry_after_values": sorted(
                {
                    value
                    for item in rows
                    if item["status"] == 429
                    if (value := item.get("retry_after")) is not None
                }
            ),
            "error_types": sorted(
                {
                    value
                    for item in rows
                    if item["status"] == 429
                    if (value := item.get("error_type")) is not None
                }
            ),
            "error_codes": sorted(
                {
                    value
                    for item in rows
                    if item["status"] == 429
                    if (value := item.get("error_code")) is not None
                }
            ),
        },
        "metrics": {
            "samples": len(samples),
            "peak_gauges": peak_gauges,
            "chat_429_counter_delta": after_429 - before_429,
        },
        "health": {"before": health_before, "after": health_after},
        "requests": rows,
    }
    print(
        json.dumps(result, indent=None if args.compact else 2, sort_keys=True)
    )

    if args.require_429 and not rejected:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
