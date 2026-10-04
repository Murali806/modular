#!/usr/bin/env python3
"""Show how prefill and decode contexts become one ragged token buffer."""

from __future__ import annotations

import json

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.lib.interfaces.batch_processor import (
    build_single_replica_ragged_token_arrays,
)
from max.pipelines.modeling.types import RequestID


def _context(
    request_id: str, prompt: list[int], generated: int | None = None
) -> TextContext:
    context = TextContext(
        request_id=RequestID(request_id),
        max_length=len(prompt) + 8,
        tokens=TokenBuffer(np.array(prompt, dtype=np.int64)),
    )
    if generated is not None:
        context.tokens.advance_with_token(generated)
    return context


def main() -> int:
    contexts = [
        _context("prefill", [10, 11, 12, 13]),
        _context("decode-a", [20, 21, 22], generated=120),
        _context("decode-b", [30, 31], generated=130),
    ]
    tokens, offsets = build_single_replica_ragged_token_arrays(contexts)
    result = {
        "evidence": "CPU-RUN",
        "contexts": [
            {
                "request": str(context.request_id),
                "processed": context.tokens.processed_length,
                "active": context.tokens.active.tolist(),
                "active_length": context.tokens.active_length,
                "generated_length": context.tokens.generated_length,
            }
            for context in contexts
        ],
        "ragged_tokens": {
            "dtype": str(tokens.dtype),
            "shape": list(tokens.shape),
            "values": tokens.tolist(),
        },
        "row_offsets": {
            "dtype": str(offsets.dtype),
            "shape": list(offsets.shape),
            "values": offsets.tolist(),
        },
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
