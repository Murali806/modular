#!/usr/bin/env python3
"""Trace MAX Serve IPC, admission, queue caps, and cancellation."""

from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace
from typing import Any, Generic, TypeVar, cast

import msgspec
import numpy as np
from max.pipelines.context import BaseContext, TextGenerationOutput
from max.pipelines.context.status import GenerationStatus
from max.pipelines.modeling.types import (
    RequestID,
    msgpack_numpy_oob_decoder,
    msgpack_numpy_oob_encoder,
)
from max.serve.api_server import _request_queue_full_exception_handler
from max.serve.scheduler_result import SchedulerResult
from max.serve.worker_interface import RequestQueueFull
from max.serve.worker_interface._zmq_queue import ZmqConfig
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerProxy
from tests.serve.scheduler.common import (
    create_paged_scheduler,
    create_text_context,
)

_T = TypeVar("_T")


class TraceContext(msgspec.Struct):
    request_id: RequestID
    label: str


class _AsyncPushQueue(Generic[_T]):
    def __init__(self, *, writable: bool = True) -> None:
        self.items: asyncio.Queue[_T] = asyncio.Queue()
        self.is_writable = writable
        self.put_calls = 0

    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        return self.is_writable

    async def put(self, item: _T) -> None:
        self.put_calls += 1
        await self.items.put(item)

    def put_nowait(self, item: _T) -> None:
        self.items.put_nowait(item)


class _AsyncPullQueue(Generic[_T]):
    def __init__(self) -> None:
        self.items: asyncio.Queue[_T] = asyncio.Queue()

    async def get(self) -> _T:
        return await self.items.get()


class _Transport:
    def __init__(self) -> None:
        response_type = dict[RequestID, SchedulerResult[TextGenerationOutput]]
        self.request_config = ZmqConfig[TraceContext](TraceContext)
        self.response_config = ZmqConfig(response_type)
        self.cancel_config = ZmqConfig[list[RequestID]](list[RequestID])

        self.request_push = self.request_config.async_push()
        self.request_pull = self.request_config.async_pull()
        self.response_push = self.response_config.async_push()
        self.response_pull = self.response_config.async_pull()
        self.cancel_push = self.cancel_config.async_push()
        self.cancel_pull = self.cancel_config.async_pull()
        self.proxy = ZmqModelWorkerProxy(
            self.request_push,
            self.response_pull,
            self.cancel_push,
        )
        self.response_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _Transport:
        self.response_task = asyncio.create_task(self.proxy.response_worker())
        await self.proxy.wait_until_connected(timeout_s=1.0)
        assert await self.response_push.writable(timeout_s=1.0)
        assert await self.cancel_push.writable(timeout_s=1.0)
        return self

    async def __aexit__(self, *args: object) -> None:
        assert self.response_task is not None
        self.response_task.cancel()
        try:
            await self.response_task
        except asyncio.CancelledError:
            pass
        for socket in (
            self.request_push,
            self.request_pull,
            self.response_push,
            self.response_pull,
            self.cancel_push,
            self.cancel_pull,
        ):
            socket.close()


def _output(
    request_id: RequestID,
    token: int,
    *,
    done: bool,
) -> TextGenerationOutput:
    return TextGenerationOutput(
        request_id=request_id,
        tokens=[token],
        final_status=(
            GenerationStatus.END_OF_SEQUENCE
            if done
            else GenerationStatus.ACTIVE
        ),
    )


async def _collect(
    stream: Any,
) -> list[dict[str, Any]]:
    chunks = []
    async for outputs, batch_id in stream:
        chunks.append(
            {
                "tokens": [
                    token for output in outputs for token in output.tokens
                ],
                "batch_id": batch_id,
                "done": outputs[-1].is_done,
            }
        )
    return chunks


async def _normal_routing_probe() -> dict[str, Any]:
    async with _Transport() as transport:
        request_a = TraceContext(RequestID("route-A"), "alpha")
        request_b = TraceContext(RequestID("route-B"), "beta")
        stream_a = await transport.proxy.stream(request_a.request_id, request_a)
        stream_b = await transport.proxy.stream(request_b.request_id, request_b)

        worker_received = [
            str((await transport.request_pull.get()).request_id),
            str((await transport.request_pull.get()).request_id),
        ]
        collect_a = asyncio.create_task(_collect(stream_a))
        collect_b = asyncio.create_task(_collect(stream_b))

        # Deliberately return B before A. The proxy must demultiplex by ID.
        await transport.response_push.put(
            {
                request_b.request_id: SchedulerResult.create(
                    _output(request_b.request_id, 202, done=True),
                    batch_id=7,
                )
            }
        )
        await transport.response_push.put(
            {
                request_a.request_id: SchedulerResult.create(
                    _output(request_a.request_id, 101, done=False),
                    batch_id=8,
                )
            }
        )
        await transport.response_push.put(
            {
                request_a.request_id: SchedulerResult.create(
                    _output(request_a.request_id, 102, done=True),
                    batch_id=9,
                )
            }
        )

        return {
            "worker_received": worker_received,
            "worker_response_order": ["route-B", "route-A", "route-A"],
            "client_A": await asyncio.wait_for(collect_a, timeout=1.0),
            "client_B": await asyncio.wait_for(collect_b, timeout=1.0),
            "pending_output_ids_after_completion": [
                str(item) for item in transport.proxy.pending_out_queues
            ],
        }


async def _cancel_before_worker_dequeue_probe() -> dict[str, Any]:
    async with _Transport() as transport:
        request = TraceContext(RequestID("cancel-before-dequeue"), "queued")
        stream = await transport.proxy.stream(request.request_id, request)
        consumer = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        consumer.cancel()
        try:
            await consumer
        except asyncio.CancelledError:
            pass

        queued_request = await asyncio.wait_for(
            transport.request_pull.get(), timeout=1.0
        )
        cancelled_ids = await asyncio.wait_for(
            transport.cancel_pull.get(), timeout=1.0
        )
        return {
            "request_still_crossed_request_queue": str(
                queued_request.request_id
            ),
            "cancel_crossed_separate_queue": [
                str(item) for item in cancelled_ids
            ],
            "api_pending_output_removed": (
                request.request_id not in transport.proxy.pending_out_queues
            ),
        }


async def _cancel_during_decode_transport_probe() -> dict[str, Any]:
    async with _Transport() as transport:
        request = TraceContext(RequestID("cancel-during-decode"), "active")
        stream = await transport.proxy.stream(request.request_id, request)
        worker_request = await asyncio.wait_for(
            transport.request_pull.get(), timeout=1.0
        )

        first_chunk_task = asyncio.create_task(anext(stream))
        await transport.response_push.put(
            {
                request.request_id: SchedulerResult.create(
                    _output(request.request_id, 303, done=False),
                    batch_id=12,
                )
            }
        )
        first_outputs, first_batch_id = await asyncio.wait_for(
            first_chunk_task, timeout=1.0
        )

        blocked_next_chunk = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        blocked_next_chunk.cancel()
        try:
            await blocked_next_chunk
        except asyncio.CancelledError:
            pass

        cancelled_ids = await asyncio.wait_for(
            transport.cancel_pull.get(), timeout=1.0
        )
        return {
            "worker_received": str(worker_request.request_id),
            "first_token": first_outputs[0].tokens,
            "first_batch_id": first_batch_id,
            "cancel_queue_request_ids": [str(item) for item in cancelled_ids],
            "api_pending_output_removed": (
                request.request_id not in transport.proxy.pending_out_queues
            ),
        }


async def _full_queue_probe() -> dict[str, Any]:
    request_queue: _AsyncPushQueue[BaseContext] = _AsyncPushQueue(
        writable=False
    )
    response_queue: _AsyncPullQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = _AsyncPullQueue()
    cancel_queue: _AsyncPushQueue[list[RequestID]] = _AsyncPushQueue()
    proxy = ZmqModelWorkerProxy(
        request_queue,
        response_queue,
        cancel_queue,
    )
    request_id = RequestID("queue-full")
    context = cast(BaseContext, TraceContext(request_id, "rejected"))

    error: RequestQueueFull | None = None
    try:
        await proxy.stream(request_id, context)
    except RequestQueueFull as caught:
        error = caught
    assert error is not None

    request = SimpleNamespace(state=SimpleNamespace(request_id=str(request_id)))
    response = await _request_queue_full_exception_handler(request, error)
    body = json.loads(bytes(response.body))
    return {
        "exception": type(error).__name__,
        "request_put_calls": request_queue.put_calls,
        "pending_output_ids": [str(item) for item in proxy.pending_out_queues],
        "http_status": response.status_code,
        "retry_after": response.headers.get("retry-after"),
        "error_type": body["error"]["type"],
        "error_code": body["error"]["code"],
    }


def _codec_probe() -> dict[str, Any]:
    encoder = msgpack_numpy_oob_encoder()
    decoder = msgpack_numpy_oob_decoder(np.ndarray)

    def encode(size: int) -> dict[str, Any]:
        source = np.arange(size, dtype=np.int64)
        frames = encoder(source)
        decoded = decoder(frames)
        return {
            "elements": size,
            "bytes": source.nbytes,
            "frames": len(frames),
            "decoded_equal": bool(np.array_equal(source, decoded)),
            "decoded_writeable": bool(decoded.flags.writeable),
            "codec_view_shares_source": bool(np.shares_memory(source, decoded)),
        }

    return {
        "inline": encode(22),
        "out_of_band_threshold": encode(8192),
    }


def _scheduler_pending_cap_probe() -> dict[str, Any]:
    scheduler, request_queue = create_paged_scheduler(
        max_batch_size=8,
        max_pending_requests=2,
    )
    contexts = [create_text_context(10, 15) for _ in range(5)]
    for context in contexts:
        request_queue.put_nowait(context)

    snapshots = []

    def snapshot(step: str) -> None:
        snapshots.append(
            {
                "step": step,
                "scheduler_pending": len(
                    scheduler.batch_constructor.all_ce_reqs
                ),
                "request_queue_remaining": request_queue.qsize(),
            }
        )

    scheduler._retrieve_pending_requests()
    snapshot("first drain with M=2")
    scheduler._retrieve_pending_requests()
    snapshot("second drain while M=2 is full")
    scheduler.max_pending_requests = 5
    scheduler._retrieve_pending_requests()
    snapshot("raise M to 5 and drain")
    return {"snapshots": snapshots}


def _drain_scheduler_responses(scheduler: Any) -> list[dict[str, Any]]:
    responses = []
    while not scheduler.response_queue.empty():
        response_dict = scheduler.response_queue.get_nowait()
        for request_id, result in response_dict.items():
            responses.append(
                {
                    "request_id": str(request_id),
                    "done": result.is_done,
                    "tokens": result.result.tokens if result.result else [],
                    "cancelled": result.is_done and result.result is None,
                }
            )
    return responses


def _scheduler_cancellation_probe() -> dict[str, Any]:
    during, during_queue = create_paged_scheduler(
        max_batch_size=1,
        max_pending_requests=1,
        max_seq_len=64,
        num_blocks=10,
        page_size=8,
        enable_chunked_prefill=False,
    )
    active = create_text_context(8, 20)
    during_queue.put_nowait(active)
    during.run_iteration()
    first_responses = _drain_scheduler_responses(during)
    generated_before_cancel = active.tokens.generated_length

    during.cancel_queue.put_nowait([active.request_id])
    during.run_iteration()
    cancellation_responses = _drain_scheduler_responses(during)

    before, before_queue = create_paged_scheduler(
        max_batch_size=1,
        max_pending_requests=1,
        max_seq_len=64,
        num_blocks=10,
        page_size=8,
        enable_chunked_prefill=False,
    )
    blocker = create_text_context(8, 20)
    not_yet_visible = create_text_context(8, 20)
    before_queue.put_nowait(blocker)
    before_queue.put_nowait(not_yet_visible)
    before._retrieve_pending_requests()
    before.cancel_queue.put_nowait([not_yet_visible.request_id])
    before.run_iteration()
    cancel_queue_drained = before.cancel_queue.empty()
    victim_still_in_request_queue = before_queue.qsize() == 1
    before.run_iteration()

    return {
        "during_decode": {
            "generated_before_cancel": generated_before_cancel,
            "generated_after_cancel_iteration": active.tokens.generated_length,
            "scheduler_contains_after_cancel": (
                during.batch_constructor.contains(active.request_id)
            ),
            "responses_before_cancel": first_responses,
            "responses_in_cancel_iteration": cancellation_responses,
        },
        "before_scheduler_visibility": {
            "cancel_queue_drained_while_request_waited_behind_M": (
                cancel_queue_drained
            ),
            "request_still_in_zmq_side_queue_after_cancel_check": (
                victim_still_in_request_queue
            ),
            "request_admitted_on_next_iteration": (
                before.batch_constructor.contains(not_yet_visible.request_id)
            ),
            "interpretation": (
                "text scheduler does not remember an unknown cancellation"
            ),
        },
    }


async def main_async() -> dict[str, Any]:
    return {
        "evidence": "CPU-RUN",
        "codec": _codec_probe(),
        "normal_response_routing": await _normal_routing_probe(),
        "cancel_after_handoff_before_worker_dequeue": (
            await _cancel_before_worker_dequeue_probe()
        ),
        "cancel_during_decode_transport": (
            await _cancel_during_decode_transport_probe()
        ),
        "full_queue_to_http_429": await _full_queue_probe(),
        "scheduler_pending_cap": _scheduler_pending_cap_probe(),
        "scheduler_cancellation": _scheduler_cancellation_probe(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(main_async())
    print(
        json.dumps(result, indent=None if args.compact else 2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
