# Complete Code Walk: `model_worker.stream(...)`

Sources:
[`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L436),
[`zmq_interface.py`](../../../max/python/max/serve/worker_interface/zmq_interface.py#L105)

## API-Side Call

```python
# Hand the request off to the model worker. Awaiting the submit
# performs the handoff (e.g. the zmq put), so a failure here — for
# example a dead worker — raises before the generator is returned,
# letting the caller respond with an HTTP error before streaming
# headers are sent.
response_stream = await self.model_worker.stream(
    context.request_id, context
)
```

```text
context.request_id + TextContext
             |
             | await model_worker.stream(...)
             v
      worker request queue handoff
             |
             v
AsyncGenerator[(list[TextGenerationOutput], batch_id)]
```

## Admission And Request Queue Push

```python
async def stream(
    self, req_id: RequestID, data: BaseContextType
) -> AsyncGenerator[tuple[list[PipelineOutputType], int | None], None]:
    # WALKTHROUGH COMMENT (not in source):
    # The source docstring defines immediate bounded-queue rejection,
    # duplicate-ID failure, registration rollback, and ordered output.

    if req_id in self.pending_out_queues:
        raise RuntimeError(
            f"Detected multiple requests with `req_id` set to {req_id}. "
            "This WILL lead to unexpected behavior! "
            "Please ensure that the `req_id` is unique for each request."
        )

    # Admission gate. Probe writability and push under a lock so the probe
    # is authoritative.
    async with self._admission_lock:
        if not await self.request_queue.writable():
            raise RequestQueueFull(
                f"Model worker request queue is full; rejecting {req_id}."
            )

        out_queue: asyncio.Queue[
            tuple[float, SchedulerResult[PipelineOutputType]]
        ] = asyncio.Queue()
        self.pending_out_queues[req_id] = out_queue
        try:
            await self.request_queue.put(data)
        except BaseException:
            # Submission failed before any response streamed; roll back the
            # registration and cancel so the worker drops partial state.
            del self.pending_out_queues[req_id]
            with contextlib.suppress(Exception):
                self.cancel(req_id)
            raise

    return self._drain_responses(req_id, out_queue)
```

```text
stream(req_id, data)
       |
       +-- req_id already registered? --> RuntimeError
       |
       | acquire _admission_lock
       v
request_queue.writable()?
       |
   +---+---+
   |       |
  no      yes
   |       |
   |       +-- create out_queue
   |       +-- pending_out_queues[req_id] = out_queue
   |       +-- await request_queue.put(data)
   |               |
   |               +-- failure --> remove registration + cancel + raise
   |               +-- success
   |                       |
   v                       v
RequestQueueFull     return _drain_responses(...)
```

## Worker Responses Routed To Per-Request Queue

```python
async def response_worker(self) -> None:
    """Awaits responses from the model worker and routes them to pending output queues."""
    while True:
        response_dict = await self.response_queue.get()
        for request_id, response in response_dict.items():
            if request_id in self.pending_out_queues:
                await self.pending_out_queues[request_id].put(
                    (time.monotonic(), response)
                )
```

```text
model-worker response_queue
          |
          | response_dict[request_id]
          v
pending_out_queues[request_id]
          |
          | enqueue timestamp + SchedulerResult
          v
_drain_responses consumer for that HTTP request
```

## Response Drain Generator

```python
async def _drain_responses(
    self,
    req_id: RequestID,
    queue: asyncio.Queue[tuple[float, SchedulerResult[PipelineOutputType]]],
) -> AsyncGenerator[tuple[list[PipelineOutputType], int | None], None]:
    try:
        while True:
            enqueue_s, item = await queue.get()
            METRICS.response_queue_time(
                (time.monotonic() - enqueue_s) * 1000
            )

            if item.result is None:
                if item.error is not None:
                    raise InputError(item.error)
                break

            outputs = [item.result]
            batch_id = item.batch_id
            should_stop = item.is_done

            while True:
                try:
                    _, item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if item.result is None:
                    if item.error is not None:
                        raise InputError(item.error)
                    should_stop = True
                    break

                outputs.append(item.result)
                if item.batch_id is not None:
                    batch_id = item.batch_id
                if item.is_done:
                    should_stop = True
                    break

            yield outputs, batch_id

            if should_stop:
                break
    except BaseException:
        with contextlib.suppress(Exception):
            self.cancel(req_id)
        raise
    finally:
        del self.pending_out_queues[req_id]
```

```text
await queue.get()
      |
      +-- result=None, error set -----> raise InputError
      +-- result=None, no error ------> finish stream
      +-- result present
              |
              | drain currently buffered items with get_nowait()
              v
        outputs: ordered list
        batch_id: latest non-None batch ID
        should_stop: any item is_done/terminal
              |
              | yield outputs, batch_id
              v
        continue or stop
              |
              v
finally: delete pending_out_queues[req_id]
```

## Cancellation Path

```python
def cancel(self, req_id: RequestID) -> None:
    # WALKTHROUGH COMMENT (not in source): enqueue cancellation without waiting.
    self.cancel_queue.put_nowait([req_id])
```

```text
client disconnect / generator error / submit rollback
                    |
                    | cancel(req_id)
                    v
cancel_queue.put_nowait([req_id])
                    |
                    v
scheduler releases request state
```

## Complete Admission And Response Values

```python
req_id = RequestID("req-chat-001")

prompt_token_ids = [
    128000,
    128006,
    9125,
    128007,
    271,
    2675,
    527,
    264,
    64694,
    18328,
    13,
    128009,
    128006,
    882,
    128007,
    271,
    849,
    21435,
    14736,
    304,
    832,
    11914,
    13,
    128009,
    128006,
    78191,
    128007,
    271,
]

data_snapshot = {
    "request_id": "req-chat-001",
    "tokens.prompt": prompt_token_ids,
    "tokens.generated": [],
    "tokens.active": prompt_token_ids,
    "max_length": 60,
    "vocab_size": 128256,
    "status": "ACTIVE",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
}

state_before_admission = {
    "pending_out_queues": {},
    "request_queue.writable()": True,
}

state_after_registration = {
    "pending_out_queues": {"req-chat-001": []},
    "request_queue": [data_snapshot],
}

worker_response_dicts = [
    {
        "req-chat-001": {
            "result": {
                "tokens": [48870, 6636],
                "decoded_tokens": "KV cache",
                "status": "ACTIVE",
                "is_done": False,
            },
            "batch_id": 41,
            "is_done": False,
            "error": None,
        }
    },
    {
        "req-chat-001": {
            "result": {
                "tokens": [304, 13],
                "decoded_tokens": " stores.",
                "status": "ACTIVE",
                "is_done": False,
            },
            "batch_id": 42,
            "is_done": False,
            "error": None,
        }
    },
    {
        "req-chat-001": {
            "result": {
                "tokens": [128009],
                "decoded_tokens": "",
                "status": "END_OF_SEQUENCE",
                "is_done": True,
            },
            "batch_id": 43,
            "is_done": True,
            "error": None,
        }
    },
]

drain_yields = [
    {
        "outputs": [
            {
                "tokens": [48870, 6636],
                "decoded_tokens": "KV cache",
                "status": "ACTIVE",
                "is_done": False,
            }
        ],
        "batch_id": 41,
    },
    {
        "outputs": [
            {
                "tokens": [304, 13],
                "decoded_tokens": " stores.",
                "status": "ACTIVE",
                "is_done": False,
            }
        ],
        "batch_id": 42,
    },
    {
        "outputs": [
            {
                "tokens": [128009],
                "decoded_tokens": "",
                "status": "END_OF_SEQUENCE",
                "is_done": True,
            }
        ],
        "batch_id": 43,
    },
]

state_after_finally = {
    "pending_out_queues": {},
    "cancel_queue": [],
}
```
