#!/usr/bin/env python3
"""Dependency-free external client for the MAX Serve learning labs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:18000"
DEFAULT_MODEL = "modularai/SmolLM-135M-Instruct-FP32"


def request(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 120.0,
) -> tuple[int, dict[str, str], bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def print_response(
    title: str, status: int, headers: dict[str, str], body: bytes
) -> None:
    print(f"\n== {title} ==")
    print(f"status: {status}")
    request_id = headers.get("x-request-id") or headers.get("X-Request-ID")
    if request_id:
        print(f"x-request-id: {request_id}")
    if not body:
        print("body: <empty>")
        return
    text = body.decode("utf-8", errors="replace")
    try:
        print(json.dumps(json.loads(text), indent=2, sort_keys=True))
    except json.JSONDecodeError:
        print(text)


def chat_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "stream": stream,
        "ignore_eos": ignore_eos,
    }


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    disconnect_after: int | None,
    signal_pid: int | None,
    signal_after: int,
    ignore_eos: bool,
) -> None:
    payload = chat_payload(
        model, prompt, max_tokens, stream=True, ignore_eos=ignore_eos
    )
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print("\n== streaming chat: raw SSE ==")
    start = time.monotonic()
    frame_count = 0
    with urllib.request.urlopen(req, timeout=120.0) as response:
        print(f"status: {response.status}")
        print(f"x-request-id: {response.headers.get('x-request-id')}")
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            elapsed_ms = (time.monotonic() - start) * 1000
            print(f"{elapsed_ms:9.2f} ms | {line}")
            if line.startswith("data: "):
                frame_count += 1
                if signal_pid is not None and frame_count == signal_after:
                    os.kill(signal_pid, signal.SIGTERM)
                    print(f"sent SIGTERM to pid {signal_pid}")
            if disconnect_after is not None and frame_count >= disconnect_after:
                print(f"disconnecting after {frame_count} data frame(s)")
                return


def selected_modes(mode: str) -> Iterable[str]:
    if mode == "all":
        return ("health", "models", "chat", "invalid", "stream")
    return (mode,)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "health", "models", "chat", "invalid", "stream"),
        default="all",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Reply with exactly three words.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--disconnect-after", type=int)
    parser.add_argument("--signal-pid", type=int)
    parser.add_argument("--signal-after", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()

    for mode in selected_modes(args.mode):
        if mode == "health":
            print_response("health", *request(args.base_url, "/health"))
        elif mode == "models":
            print_response("models", *request(args.base_url, "/v1/models"))
        elif mode == "chat":
            print_response(
                "non-streaming chat",
                *request(
                    args.base_url,
                    "/v1/chat/completions",
                    chat_payload(
                        args.model,
                        args.prompt,
                        args.max_tokens,
                        stream=False,
                        ignore_eos=args.ignore_eos,
                    ),
                ),
            )
        elif mode == "invalid":
            print_response(
                "validation error",
                *request(
                    args.base_url,
                    "/v1/chat/completions",
                    {"model": args.model, "messages": "not-a-message-list"},
                ),
            )
        elif mode == "stream":
            stream_chat(
                args.base_url,
                args.model,
                args.prompt,
                args.max_tokens,
                args.disconnect_after,
                args.signal_pid,
                args.signal_after,
                args.ignore_eos,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
