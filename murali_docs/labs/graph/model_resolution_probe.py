#!/usr/bin/env python3
"""Inspect a cached Llama checkpoint without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

from max.pipelines.architectures.llama3.arch import llama_arch

DEFAULT_SNAPSHOT = Path(
    "/local/mnt/workspace/.murali_hf/hub/"
    "models--modularai--SmolLM-135M-Instruct-FP32/snapshots/"
    "1ade67aacf72511c94c55529056f7222c1c0b586"
)


def _name(value: Any) -> str:
    return getattr(value, "__name__", str(value))


def _safetensor_keys(path: Path) -> list[str]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    return [key for key in header if key != "__metadata__"]


def _adapt(name: str) -> str:
    return name.replace("model.", "").replace("g_idx", "perm_idx")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "snapshot", nargs="?", type=Path, default=DEFAULT_SNAPSHOT
    )
    args = parser.parse_args()

    config = json.loads((args.snapshot / "config.json").read_text())
    keys = _safetensor_keys(args.snapshot / "model.safetensors")
    examples = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    ]
    missing = [key for key in examples if key not in keys]
    if missing:
        raise ValueError(f"Expected checkpoint keys are missing: {missing}")

    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[
        config["torch_dtype"]
    ]
    kv_bytes_per_token = (
        2
        * config["num_hidden_layers"]
        * config["num_key_value_heads"]
        * config["head_dim"]
        * dtype_bytes
    )
    result = {
        "evidence": "CPU-RUN + SOURCE",
        "snapshot": str(args.snapshot),
        "architecture_from_config": config["architectures"],
        "registry": {
            "name": llama_arch.name,
            "task": str(llama_arch.task),
            "pipeline_model": _name(llama_arch.pipeline_model),
            "tokenizer": _name(llama_arch.tokenizer),
            "context_type": _name(llama_arch.context_type),
            "batching": _name(llama_arch.batching),
            "memory_planner": _name(llama_arch.memory_planner),
            "default_weights_format": str(llama_arch.default_weights_format),
        },
        "model_geometry": {
            key: config[key]
            for key in (
                "torch_dtype",
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "max_position_embeddings",
                "rope_theta",
            )
        },
        "kv_bytes_per_token_unsharded": kv_bytes_per_token,
        "checkpoint_tensor_count": len(keys),
        "weight_name_examples": [
            {"checkpoint": key, "max": _adapt(key)} for key in examples
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
