#!/usr/bin/env python3
"""Produce deterministic CPU evidence for MAX continuous batching."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.modeling.types import RequestID
from tests.serve.scheduler.common import create_paged_scheduler


def _context(label: str, prompt: int, output: int) -> TextContext:
    return TextContext(
        request_id=RequestID(label),
        max_length=prompt + output,
        tokens=TokenBuffer(np.arange(prompt, dtype=np.int64)),
    )


def _request_state(context: TextContext) -> dict[str, Any]:
    return {
        "processed": context.tokens.processed_length,
        "active": context.tokens.active_length,
        "generated": context.tokens.generated_length,
        "total": len(context.tokens),
        "done": context.is_done,
    }


def _drain_responses(scheduler: Any) -> list[dict[str, Any]]:
    rows = []
    while not scheduler.response_queue.empty():
        for request_id, result in scheduler.response_queue.get_nowait().items():
            rows.append(
                {
                    "request": str(request_id),
                    "tokens": result.result.tokens if result.result else [],
                    "done": result.is_done,
                }
            )
    return rows


def _iteration(
    scheduler: Any,
    contexts: dict[str, TextContext],
    number: int,
) -> dict[str, Any]:
    scheduler._retrieve_pending_requests()
    constructor = scheduler.batch_constructor
    before = {
        "ce": [str(item) for item in constructor.all_ce_reqs],
        "tg": [str(item) for item in constructor.all_tg_reqs],
    }
    preemptions_before = constructor.total_preemption_count
    inputs = constructor.construct_batch()
    selected = [
        {
            "request": str(context.request_id),
            "kind": "CE" if context.tokens.generated_length == 0 else "TG",
            "active_tokens": context.tokens.active_length,
            "processed_tokens": context.tokens.processed_length,
        }
        for context in inputs.flat_batch
        if not context._is_padding_ctx
    ]
    per_replica = [
        [
            str(context.request_id)
            for context in batch
            if not context._is_padding_ctx
        ]
        for batch in inputs.batches
    ]
    terminated = scheduler._schedule(inputs) if inputs else 0
    block_counts = [
        {
            "replica": replica,
            "free": constructor.kv_cache.block_count(replica).free,
            "used": constructor.kv_cache.block_count(replica).used,
            "total": constructor.kv_cache.block_count(replica).total,
        }
        for replica in range(constructor.num_replicas)
    ]
    return {
        "iteration": number,
        "before": before,
        "batch_type": inputs.batch_type.value,
        "batch_size": inputs.batch_size,
        "input_tokens": inputs.input_tokens,
        "context_tokens": inputs.context_tokens,
        "per_replica": per_replica,
        "selected": selected,
        "preempted": constructor.total_preemption_count - preemptions_before,
        "terminated": terminated,
        "responses": _drain_responses(scheduler),
        "after": {
            "ce": [str(item) for item in constructor.all_ce_reqs],
            "tg": [str(item) for item in constructor.all_tg_reqs],
            "requests": {
                label: _request_state(context)
                for label, context in contexts.items()
            },
            "kv_blocks": block_counts,
        },
    }


def _mixed_probe() -> dict[str, Any]:
    scheduler, queue = create_paged_scheduler(
        max_seq_len=32,
        num_blocks=128,
        max_batch_size=3,
        page_size=1,
        target_tokens_per_batch_ce=8,
        enable_chunked_prefill=True,
        enable_in_flight_batching=True,
    )
    contexts = {
        "A": _context("A", prompt=4, output=4),
        "B": _context("B", prompt=3, output=3),
        "C": _context("C", prompt=5, output=3),
    }
    queue.put_nowait(contexts["A"])
    queue.put_nowait(contexts["B"])
    rows = [_iteration(scheduler, contexts, 1)]
    queue.put_nowait(contexts["C"])
    for number in range(2, 7):
        rows.append(_iteration(scheduler, contexts, number))
    return {
        "configuration": {
            "max_batch_size": 3,
            "active_token_budget": 8,
            "page_size": 1,
            "in_flight_batching": True,
        },
        "iterations": rows,
    }


def _chunk_probe() -> dict[str, Any]:
    scheduler, queue = create_paged_scheduler(
        max_seq_len=32,
        num_blocks=64,
        max_batch_size=2,
        page_size=1,
        target_tokens_per_batch_ce=4,
        enable_chunked_prefill=True,
    )
    contexts = {"LONG": _context("LONG", prompt=11, output=2)}
    queue.put_nowait(contexts["LONG"])
    return {
        "configuration": {
            "prompt_tokens": 11,
            "output_tokens": 2,
            "active_token_budget": 4,
            "page_size": 1,
        },
        "iterations": [
            _iteration(scheduler, contexts, number) for number in range(1, 6)
        ],
    }


def _preemption_probe() -> dict[str, Any]:
    scheduler, queue = create_paged_scheduler(
        max_seq_len=32,
        num_blocks=5,
        max_batch_size=8,
        page_size=2,
        target_tokens_per_batch_ce=16,
        enable_chunked_prefill=False,
    )
    contexts = {
        "P": _context("P", prompt=3, output=7),
        "Q": _context("Q", prompt=3, output=7),
    }
    queue.put_nowait(contexts["P"])
    queue.put_nowait(contexts["Q"])
    return {
        "configuration": {
            "requests": 2,
            "tokens_per_request": 10,
            "kv_capacity_tokens": 10,
            "page_size": 2,
        },
        "iterations": [
            _iteration(scheduler, contexts, number) for number in range(1, 12)
        ],
    }


def _data_parallel_probe() -> dict[str, Any]:
    scheduler, queue = create_paged_scheduler(
        max_seq_len=32,
        num_blocks=64,
        max_batch_size=2,
        page_size=1,
        target_tokens_per_batch_ce=8,
        enable_chunked_prefill=True,
        dp=2,
    )
    contexts = {
        label: _context(label, prompt=2 + index, output=2)
        for index, label in enumerate(("D0", "D1", "D2", "D3"))
    }
    for context in contexts.values():
        queue.put_nowait(context)
    scheduler._retrieve_pending_requests()
    constructor = scheduler.batch_constructor
    placement = [
        {
            "replica": replica_idx,
            "ce": [str(item) for item in replica.ce_reqs],
            "tg": [str(item) for item in replica.tg_reqs],
        }
        for replica_idx, replica in enumerate(constructor.replicas)
    ]
    rows = [_iteration(scheduler, contexts, number) for number in range(1, 4)]
    return {
        "configuration": {
            "data_parallel_degree": 2,
            "max_batch_size_per_replica": 2,
        },
        "initial_least_loaded_placement": placement,
        "iterations": rows,
        "padding": "not configured by the deterministic helper; source-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = {
        "evidence": "CPU-RUN",
        "mixed_prefill_decode": _mixed_probe(),
        "chunked_prefill": _chunk_probe(),
        "kv_preemption": _preemption_probe(),
        "data_parallel": _data_parallel_probe(),
    }
    print(
        json.dumps(result, indent=None if args.compact else 2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
