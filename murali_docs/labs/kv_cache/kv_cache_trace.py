#!/usr/bin/env python3
"""Produce deterministic CPU evidence for paged KV and prefix caching."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.kv_cache.connectors.null_connector import NullConnector
from max.pipelines.kv_cache.paged_kv_cache.block_manager import (
    BlockManager,
    compute_block_hashes,
)
from max.pipelines.kv_cache.paged_kv_cache.block_pool import BlockPool
from max.pipelines.modeling.types import RequestID


BLOCK_SIZE = 8


def _hash_label(value: bytes | None) -> str | None:
    return value.hex()[:12] if value is not None else None


def _queue_order(pool: BlockPool) -> list[int]:
    order: list[int] = []
    block = pool.free_block_queue.free_list_head
    while block is not None:
        order.append(block.bid)
        block = block.next_free_block
    return order


def _pool_snapshot(pool: BlockPool, event: str) -> dict[str, Any]:
    return {
        "event": event,
        "free_lru_order": _queue_order(pool),
        "blocks": [
            {
                "id": block.bid,
                "ref_count": block.ref_cnt,
                "hash": _hash_label(block.block_hash),
                "free": block.bid in pool.free_blocks,
            }
            for block in pool.pool
        ],
        "prefix_cache": {
            _hash_label(block_hash): block.bid
            for block_hash, block in pool.prefix_cache.items()
        },
    }


def _block_pool_lifecycle() -> dict[str, Any]:
    pool = BlockPool(total_num_blocks=2, enable_runtime_checks=True)
    rows = [_pool_snapshot(pool, "initial")]

    block_a, evicted = pool.alloc_block()
    assert evicted is None
    rows.append(_pool_snapshot(pool, "allocate block A"))

    hash_a = b"A" * 8
    pool.commit_into_prefix_cache(hash_a, block_a)
    rows.append(_pool_snapshot(pool, "commit A"))

    pool.free_block(block_a)
    rows.append(_pool_snapshot(pool, "release A: cached + evictable"))

    pool.touch(block_a)
    rows.append(_pool_snapshot(pool, "prefix hit A: touch + pin"))
    pool.free_block(block_a)

    block_b, evicted = pool.alloc_block()
    assert evicted is None and block_b.bid != block_a.bid
    hash_b = b"B" * 8
    pool.commit_into_prefix_cache(hash_b, block_b)
    pool.free_block(block_b)
    rows.append(_pool_snapshot(pool, "commit and release B"))

    recycled, evicted = pool.alloc_block()
    assert recycled.bid == block_a.bid and evicted == hash_a
    rows.append(_pool_snapshot(pool, "allocate LRU: evict A"))

    pool.assert_runtime_invariants(active_bids=[recycled.bid])
    return {
        "total_blocks": pool.total_num_blocks,
        "evicted_hash": _hash_label(evicted),
        "events": rows,
    }


def _stub_context(tokens: np.ndarray, label: str) -> TextContext:
    value = SimpleNamespace(
        request_id=RequestID(label),
        tokens=tokens,
        cache_salt=None,
        dkv_cache_hint=None,
    )
    return cast(TextContext, value)


def _manager(enable_prefix_caching: bool = True) -> BlockManager:
    return BlockManager(
        total_num_blocks=16,
        block_size=BLOCK_SIZE,
        connector=NullConnector(),
        enable_prefix_caching=enable_prefix_caching,
        enable_runtime_checks=False,
    )


def _hashes(manager: BlockManager, context: TextContext) -> list[bytes]:
    return compute_block_hashes(
        context,
        [],
        manager.block_size,
        manager.kv_hash_algo,
        manager.kv_hash_seed,
    )


def _seed(manager: BlockManager, hashes: list[bytes]) -> None:
    for block_hash in hashes:
        block, evicted = manager.device_block_pool.alloc_block()
        assert evicted is None
        manager.device_block_pool.commit_into_prefix_cache(block_hash, block)


def _prefix_probe() -> dict[str, Any]:
    context = _stub_context(np.arange(33, dtype=np.int32), "hash-chain")

    contiguous = _manager()
    chain = _hashes(contiguous, context)
    _seed(contiguous, chain[:3])
    free_before = contiguous.device_block_pool.num_free_blocks
    hits_first = contiguous.count_cached_prefix_blocks(chain)
    hits_second = contiguous.count_cached_prefix_blocks(chain)

    gap = _manager()
    _seed(gap, [chain[0], chain[2]])
    gap_hits = gap.count_cached_prefix_blocks(chain)

    disabled = _manager(enable_prefix_caching=False)
    disabled_hits = disabled.count_cached_prefix_blocks(chain)

    return {
        "tokens": len(context.tokens),
        "last_token_reserved": True,
        "hashable_tokens": len(context.tokens) - 1,
        "block_size": BLOCK_SIZE,
        "hash_chain": [_hash_label(value) for value in chain],
        "contiguous_seeded_blocks": [0, 1, 2],
        "contiguous_hit_blocks": hits_first.total_blocks,
        "read_only_repeat_hit_blocks": hits_second.total_blocks,
        "read_only_free_blocks_unchanged": (
            free_before == contiguous.device_block_pool.num_free_blocks
        ),
        "gap_seeded_blocks": [0, 2],
        "gap_hit_blocks": gap_hits.total_blocks,
        "disabled_hit_blocks": disabled_hits.total_blocks,
    }


def _reuse_probe() -> dict[str, Any]:
    manager = _manager()
    context = TextContext(
        request_id=RequestID("reuse"),
        max_length=20,
        tokens=TokenBuffer(np.arange(17, dtype=np.int64)),
    )
    manager.claim(context)
    manager.compute_hashes_for_request(context)
    hashes = list(manager.req_to_hashes[context.request_id])
    _seed(manager, hashes)

    skip_amount, event = manager.reuse_blocks_from_prefix_cache(context)
    result = {
        "prompt_tokens": 17,
        "cached_full_blocks": len(hashes),
        "skipped_tokens": skip_amount,
        "processed_tokens": context.tokens.processed_length,
        "active_tokens": context.tokens.active_length,
        "cached_prefix_length": context.cached_prefix_length,
        "transfer_complete": event.is_complete(),
        "device_blocks_served": manager.metrics.device_blocks_served,
    }
    manager.release(context)
    return result


def _capacity_model() -> dict[str, Any]:
    layers = 30
    kv_heads = 3
    head_dim = 64
    dtype_bytes = 4
    page_size = 128
    bytes_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    bytes_per_page = bytes_per_token * page_size
    budget = 1024**3
    pages = budget // bytes_per_page
    return {
        "model": "modularai/SmolLM-135M-Instruct-FP32",
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "page_size_tokens": page_size,
        "bytes_per_token": bytes_per_token,
        "bytes_per_page": bytes_per_page,
        "mib_per_page": bytes_per_page / 1024**2,
        "one_gib_budget": {
            "whole_pages": pages,
            "token_slots": pages * page_size,
            "unallocated_bytes": budget - pages * bytes_per_page,
        },
        "scope": "ideal KV payload only; excludes allocator and runtime reserve",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = {
        "evidence": "CPU-RUN",
        "capacity": _capacity_model(),
        "block_pool_lifecycle": _block_pool_lifecycle(),
        "prefix_hash_and_lookup": _prefix_probe(),
        "prefix_reuse": _reuse_probe(),
    }
    print(
        json.dumps(result, indent=None if args.compact else 2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
