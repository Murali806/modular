#!/usr/bin/env python3
"""Trace one chat request from OpenAI JSON to the scheduler IPC boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from max.pipelines.context import TextContext
from max.pipelines.context.sampling_params import (
    SamplingParamsGenerationConfigDefaults,
)
from max.pipelines.lib.tokenizer import TextTokenizer
from max.pipelines.modeling.types import RequestID, TextGenerationRequest
from max.pipelines.sampling import ToolCallPolicy
from max.serve.router import openai_routes
from max.serve.schemas.openai import CreateChatCompletionRequest
from max.serve.worker_interface._zmq_queue import ZmqConfig
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerProxy
from pydantic import ValidationError
from transformers import AutoConfig, GenerationConfig


DEFAULT_MODEL = "modularai/SmolLM-135M-Instruct-FP32"
DEFAULT_REVISION = "1ade67aacf72511c94c55529056f7222c1c0b586"
STANDARD_ROLES = frozenset(
    {"developer", "system", "user", "assistant", "tool", "function"}
)


class _Request:
    def __init__(self, payload: dict[str, Any], request_id: str) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.state = SimpleNamespace(
            request_id=request_id,
            request_timer=SimpleNamespace(start_ns=1),
        )
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(
                    max_media_bytes=0,
                    use_client_cache_salt=False,
                )
            )
        )
        self.url = SimpleNamespace(path="/v1/chat/completions")
        self.headers: dict[str, str] = {}

    async def body(self) -> bytes:
        return self._body


class _WritableQueue(asyncio.Queue[Any]):
    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        return True


def _generation_defaults(
    generation_config: GenerationConfig,
) -> SamplingParamsGenerationConfigDefaults:
    allowed = SamplingParamsGenerationConfigDefaults.__dataclass_fields__
    values = {
        key: value
        for key, value in generation_config.to_diff_dict().items()
        if key in allowed
    }
    return SamplingParamsGenerationConfigDefaults(**values)


def _pipeline_config(
    model_config: SimpleNamespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model_config,
        runtime=SimpleNamespace(
            temperature=1.0,
            top_k=-1,
            thinking_temperature=None,
            enable_overlap_scheduler=False,
            allow_unsupported_logprobs=False,
            tool_parser=None,
            reasoning_parser=None,
            emit_reasoning_content=False,
            allow_extra_request_fields=False,
        ),
        sampling=SimpleNamespace(
            tool_call_policy=(
                ToolCallPolicy.DEFAULT_STRICT_FALSE_AND_BEST_EFFORT
            ),
            enable_structured_output=False,
            structured_output_backend="xgrammar",
        ),
    )


async def _route_to_text_generation_request(
    payload: dict[str, Any],
    request_id: str,
    tokenizer: TextTokenizer,
    pipeline_config: SimpleNamespace,
) -> TextGenerationRequest:
    captured: dict[str, TextGenerationRequest] = {}
    pipeline = SimpleNamespace(
        tokenizer=tokenizer,
        model_name=payload["model"],
    )

    class _CaptureResponseGenerator:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def complete(
            self, requests: list[TextGenerationRequest]
        ) -> TextGenerationRequest:
            captured["request"] = requests[0]
            return requests[0]

    with (
        patch.object(openai_routes, "get_pipeline", return_value=pipeline),
        patch.object(
            openai_routes,
            "get_app_pipeline_config",
            return_value=pipeline_config,
        ),
        patch.object(openai_routes, "get_tool_parser", return_value=None),
        patch.object(
            openai_routes,
            "OpenAIChatResponseGenerator",
            _CaptureResponseGenerator,
        ),
        patch.object(
            openai_routes,
            "_record_request_feature_metrics",
            lambda *args, **kwargs: None,
        ),
    ):
        result = await openai_routes.openai_create_chat_completion(
            _Request(payload, request_id)
        )

    assert result is captured["request"]
    return captured["request"]


def _context_view(context: TextContext) -> dict[str, Any]:
    return {
        "request_id": str(context.request_id),
        "status": context.status.name,
        "prompt_token_ids": context.tokens.prompt.tolist(),
        "prompt_length": context.tokens.prompt_length,
        "generated_length": context.tokens.generated_length,
        "processed_length": context.tokens.processed_length,
        "active_length": context.tokens.active_length,
        "active_token_ids": context.tokens.active.tolist(),
        "max_length": context.max_length,
        "token_dtype": str(context.tokens.array.dtype),
        "token_buffer_writeable": bool(context.tokens.array.flags.writeable),
        "temperature": context.sampling_params.temperature,
        "top_k": context.sampling_params.top_k,
        "max_new_tokens": context.sampling_params.max_new_tokens,
        "stop_strings": list(context.eos_tracker.eos_stop_strings),
        "stop_token_sequences": [
            list(sequence) for sequence in context.eos_tracker.eos_sequences
        ],
        "eos_token_ids": sorted(context.eos_tracker.eos_token_ids),
        "scheduler_lane": (
            "CE/prefill"
            if context.tokens.generated_length == 0
            else "TG/decode"
        ),
    }


def _request_view(request: TextGenerationRequest) -> dict[str, Any]:
    frozen = False
    try:
        request.model_name = "mutation-must-fail"  # type: ignore[misc]
    except FrozenInstanceError:
        frozen = True

    return {
        "class": type(request).__name__,
        "frozen": frozen,
        "request_id": str(request.request_id),
        "model_name": request.model_name,
        "messages": [message.model_dump() for message in request.messages],
        "prompt": request.prompt,
        "temperature": request.sampling_params.temperature,
        "top_k": request.sampling_params.top_k,
        "max_new_tokens": request.sampling_params.max_new_tokens,
        "stop": request.sampling_params.stop,
        "stop_token_ids": request.sampling_params.stop_token_ids,
        "request_path": request.request_path,
    }


def _zmq_round_trip(context: TextContext) -> tuple[TextContext, dict[str, Any]]:
    sender, receiver = ZmqConfig[TextContext](TextContext).pair()
    try:
        time.sleep(0.05)
        sender.put(context)
        deadline = time.monotonic() + 2.0
        while True:
            try:
                received = receiver.get_nowait()
                break
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for local ZMQ round trip")
                time.sleep(0.01)
    finally:
        sender.close()
        receiver.close()

    return received, {
        "transport": "ZMQ IPC + msgpack/numpy",
        "new_python_object": received is not context,
        "token_values_equal": bool(
            np.array_equal(received.tokens.all, context.tokens.all)
        ),
        "token_storage_shared": bool(
            np.shares_memory(received.tokens.array, context.tokens.array)
        ),
    }


async def _trace_case(
    name: str,
    payload: dict[str, Any],
    tokenizer: TextTokenizer,
    pipeline_config: SimpleNamespace,
) -> tuple[dict[str, Any], TextContext]:
    request_id = f"phase3-{name}"
    validated = CreateChatCompletionRequest.model_validate(payload)
    token_request = await _route_to_text_generation_request(
        payload, request_id, tokenizer, pipeline_config
    )
    rendered_prompt, rendered_ids = (
        await tokenizer._generate_prompt_and_token_ids(
            token_request.prompt,
            token_request.messages,
            token_request.tools,
            **(token_request.chat_template_options or {}),
        )
    )
    context = await tokenizer.new_context(token_request)
    scheduler_context, ipc = _zmq_round_trip(context)

    assert rendered_ids.tolist() == context.tokens.prompt.tolist()
    assert context.tokens.prompt.tolist() == scheduler_context.tokens.prompt.tolist()

    return (
        {
            "raw_openai_json": payload,
            "validated_schema": {
                "class": type(validated).__name__,
                "message_count": len(validated.messages),
                "temperature": validated.temperature,
                "top_k": validated.top_k,
                "stop": validated.stop,
                "max_tokens": validated.max_tokens,
            },
            "text_generation_request": _request_view(token_request),
            "rendered_prompt": rendered_prompt,
            "prompt_tokens": {
                "ids": rendered_ids.tolist(),
                "count": len(rendered_ids),
                "dtype_before_context": str(rendered_ids.dtype),
            },
            "api_process_context": _context_view(context),
            "ipc": ipc,
            "first_scheduler_visible_context": _context_view(
                scheduler_context
            ),
        },
        context,
    )


async def _disconnect_probe(context: TextContext) -> dict[str, Any]:
    request_queue = _WritableQueue()
    response_queue = _WritableQueue()
    cancel_queue = _WritableQueue()
    proxy = ZmqModelWorkerProxy(
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
    )

    response_stream = await proxy.stream(context.request_id, context)
    scheduler_context = await request_queue.get()

    async def consume() -> None:
        async for _outputs, _batch_id in response_stream:
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass

    cancelled_ids = await asyncio.wait_for(cancel_queue.get(), timeout=1.0)
    return {
        "submitted_request_id": str(scheduler_context.request_id),
        "consumer_state": "cancelled",
        "cancel_queue_request_ids": [str(item) for item in cancelled_ids],
        "api_pending_output_queue_removed": (
            context.request_id not in proxy.pending_out_queues
        ),
    }


def _validation_probe(model: str) -> dict[str, Any]:
    bad_shape = {"model": model, "messages": "not-a-list"}
    unknown_field = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "temperatur": 0,
    }

    def error_types(payload: dict[str, Any]) -> list[str]:
        try:
            CreateChatCompletionRequest.model_validate(payload)
        except ValidationError as error:
            return [str(item["type"]) for item in error.errors()]
        raise AssertionError("invalid payload unexpectedly validated")

    return {
        "messages_string": error_types(bad_shape),
        "unknown_top_level_field": error_types(unknown_field),
    }


async def _semantic_validation_probe(
    model: str,
    tokenizer: TextTokenizer,
    pipeline_config: SimpleNamespace,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 3.0,
        "max_tokens": 1,
        "stream": False,
    }
    try:
        await _route_to_text_generation_request(
            payload,
            "phase3-invalid-temperature",
            tokenizer,
            pipeline_config,
        )
    except HTTPException as error:
        return {"status": error.status_code, "detail": error.detail}
    raise AssertionError("out-of-range temperature unexpectedly succeeded")


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    hf_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    generation_config = GenerationConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model_config = SimpleNamespace(
        huggingface_config=hf_config,
        generation_config=generation_config,
        sampling_params_defaults=_generation_defaults(generation_config),
    )
    pipeline_config = _pipeline_config(model_config)
    tokenizer = TextTokenizer(
        args.model,
        SimpleNamespace(
            tokenizer_impl=None,
            draft_model=None,
            model=model_config,
        ),
        revision=args.revision,
        max_length=args.max_length,
    )

    base = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Name two colors."},
        ],
        "temperature": 0.7,
        "top_k": 50,
        "max_tokens": 4,
        "seed": 7,
        "stream": False,
        "return_token_ids": True,
    }

    baseline, baseline_context = await _trace_case(
        "baseline", base, tokenizer, pipeline_config
    )
    temperature_zero, _ = await _trace_case(
        "temperature-zero",
        base | {"temperature": 0.0},
        tokenizer,
        pipeline_config,
    )
    stop_sequence, _ = await _trace_case(
        "stop-sequence",
        base | {"temperature": 0.0, "stop": " two"},
        tokenizer,
        pipeline_config,
    )

    return {
        "evidence": "CPU-RUN",
        "model": args.model,
        "revision": args.revision,
        "baseline": baseline,
        "comparisons": {
            "temperature_zero": {
                "wire_temperature": temperature_zero["raw_openai_json"][
                    "temperature"
                ],
                "wire_top_k": temperature_zero["raw_openai_json"]["top_k"],
                "resolved_temperature": temperature_zero[
                    "text_generation_request"
                ]["temperature"],
                "resolved_top_k": temperature_zero[
                    "text_generation_request"
                ]["top_k"],
            },
            "stop_sequence": {
                "wire_stop": stop_sequence["raw_openai_json"]["stop"],
                "request_stop": stop_sequence["text_generation_request"][
                    "stop"
                ],
                "context_stop_strings": stop_sequence["api_process_context"][
                    "stop_strings"
                ],
                "context_stop_token_sequences": stop_sequence[
                    "api_process_context"
                ]["stop_token_sequences"],
            },
            "client_disconnect": await _disconnect_probe(baseline_context),
        },
        "validation": {
            "schema": _validation_probe(args.model),
            "semantic_temperature": await _semantic_validation_probe(
                args.model, tokenizer, pipeline_config
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face network access when the model is not cached.",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    args.local_files_only = not args.allow_download

    result = asyncio.run(main_async(args))
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
