# `self.model_worker.stream(context.request_id, context)`

Sources:

- Call site in
  [`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L436)
- Abstract interface in
  [`worker_interface/__init__.py`](../../../max/python/max/serve/worker_interface/__init__.py#L100)
- ZMQ implementation in
  [`zmq_interface.py`](../../../max/python/max/serve/worker_interface/zmq_interface.py#L105)
- Scheduler request drain in
  [`text_generation_scheduler.py`](../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L185)

## One-Line Purpose

`self.model_worker.stream(context.request_id, context)` submits the prepared
`TextContext` to the model worker and returns an async generator that yields
the worker's generated outputs for that request.

```python
response_stream = await self.model_worker.stream(
    context.request_id,
    context,
)
```

## Big Picture

```text
API / pipeline process
  |
  | has TextContext from tokenizer.new_context(...)
  v
await model_worker.stream(request_id, context)
  |
  +-- check worker request queue has room
  +-- create one output queue for this request_id
  +-- put TextContext onto worker request queue
  |
  v
response_stream
  |
  +-- later yields worker responses for this request
```

Important split:

```text
await model_worker.stream(...)
  -> performs request handoff
  -> returns response_stream

async for outputs, batch_id in response_stream
  -> receives generated output chunks later
```

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `context.request_id` | Unique request ID used to route worker responses back to this API request. |
| Input | `context` | `TextContext` containing prompt tokens, generation state, EOS tracker, grammar, routing metadata, and sampling settings. |
| Internal queue | `request_queue` | API-to-worker queue where the `TextContext` is submitted. |
| Internal queue | `pending_out_queues[request_id]` | Per-request output queue used to collect responses from the worker. |
| Output | `response_stream` | Async generator yielding `(outputs, batch_id)` pairs. |
| Error | `RequestQueueFull` | Raised if the bounded worker request queue has no room. |
| Error | `RuntimeError` | Raised if the same request ID is already registered. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
context.request_id                     check duplicate request ID              branch decision
  RequestID("req-123")               ------------------------------->          not already pending

request_queue                          check if worker queue writable          admission decision
  bounded queue                      ------------------------------->          writable: yes

request_id                             create response queue                   pending route
  "req-123"                          ------------------------------->          pending_out_queues["req-123"]

context                                submit to worker request queue          worker can receive request
  TextContext(...)                    ------------------------------->          request_queue.put(context)

stream(...) return value               caller stores stream                    response stream
  _drain_responses(...)              ------------------------------->          response_stream
```

Later, after the scheduler/model produces output:

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
worker response_queue                  response_worker routes by request_id    per-request queue item
  {"req-123": SchedulerResult(...)}  ------------------------------->          pending_out_queues["req-123"].put(...)

per-request queue                      _drain_responses(...)                   pipeline output batch
  SchedulerResult(...)               ------------------------------->          ([TextGenerationOutput(...)], batch_id)

response_stream                        LLM pipeline consumes                   TokenGeneratorOutput
  outputs, batch_id                  ------------------------------->          decoded route-facing chunk
```

## Lifecycle View

```text
+------------------------------------------------------------------------------------------------------+
| model_worker.stream(request_id, context)                                                             |
|------------------------------------------------------------------------------------------------------|
| START                                                                                                |
|   |                                                                                                  |
|   v                                                                                                  |
| request_id already in pending_out_queues?                                                            |
|   |                                                                                                  |
|   +-- YES                                                                                            |
|   |     |                                                                                            |
|   |     v                                                                                            |
|   |   raise RuntimeError                                                                             |
|   |                                                                                                  |
|   +-- NO                                                                                             |
|         |                                                                                            |
|         v                                                                                            |
|       acquire _admission_lock                                                                        |
|         |                                                                                            |
|         v                                                                                            |
|       request_queue.writable()?                                                                      |
|         |                                                                                            |
|         +-- NO                                                                                       |
|         |     |                                                                                      |
|         |     v                                                                                      |
|         |   raise RequestQueueFull                                                                   |
|         |                                                                                             |
|         +-- YES                                                                                      |
|               |                                                                                      |
|               v                                                                                      |
|             create out_queue                                                                         |
|               |                                                                                      |
|               v                                                                                      |
|             pending_out_queues[request_id] = out_queue                                               |
|               |                                                                                      |
|               v                                                                                      |
|             request_queue.put(context)                                                               |
|               |                                                                                      |
|               v                                                                                      |
|             return _drain_responses(request_id, out_queue)                                           |
|------------------------------------------------------------------------------------------------------|
| END HANDOFF                                                                                          |
+------------------------------------------------------------------------------------------------------+
```

## Sequence View

```text
LLM Pipeline                         ZMQ ModelWorkerProxy                Model Worker / Scheduler
------------                         --------------------                ------------------------
context ready
  |
  | await stream(req_id, context)
  |---------------------------------> check duplicate req_id
                                      check request_queue.writable()
                                      create pending_out_queue
                                      request_queue.put(context)
  |<--------------------------------- return response_stream
  |
  | note_awaiting_admission(-1)
  |
  | async for outputs, batch_id
  |   in response_stream:
  |                                                                        drain request_queue
  |                                                                        enqueue TextContext
  |                                                                        run scheduler/model
  |                                                                        response_queue.put(result)
                                      response_worker reads response_queue
                                      routes result to pending_out_queue
  |<--------------------------------- _drain_responses yields outputs
  |
  v
decode / format / yield API chunk
```

## Why `await` Matters Here

`await` waits for the handoff step to finish, not for the whole model response.

```text
await model_worker.stream(...)
  |
  +-- completes after context is accepted into request_queue
  |
  +-- raises before streaming begins if handoff fails
  |
  v
response_stream
  |
  +-- later produces token outputs over time
```

That is why `note_awaiting_admission(-1)` is called immediately after the
`await` succeeds: at that point the request has been admitted to the worker
side, even though generation has not finished.

## Request Queue Full Path

```text
request_queue.writable()
  |
  +-- true
  |     |
  |     v
  |   request is accepted into worker queue
  |
  +-- false
        |
        v
      raise RequestQueueFull
        |
        v
      API can reject request before response streaming starts
```

This protects the worker from accepting more queued requests than configured by
the bounded queue size.

## Response Routing

The worker returns results through a shared response queue. The proxy routes
each result to the correct API request by `request_id`.

```text
shared response_queue
  |
  v
response_worker()
  |
  +-- reads {request_id: SchedulerResult}
  |
  +-- finds pending_out_queues[request_id]
  |
  v
per-request out_queue
  |
  v
_drain_responses(...)
  |
  v
response_stream yields to LLM pipeline
```

## Q&A

<details>
<summary><strong>Q1. Does <code>stream(...)</code> run the model directly?</strong></summary>

No. It submits the `TextContext` to the worker request queue.

```text
stream(...)
  -> queue handoff
  -> returns response_stream

scheduler/model
  -> drains request queue later
  -> builds batches
  -> runs model
  -> sends SchedulerResult responses
```

</details>

<details>
<summary><strong>Q2. Why pass both <code>request_id</code> and <code>context</code>?</strong></summary>

`context` is the work item. `request_id` is the routing key.

```text
context
  -> what the worker should generate from

request_id
  -> where the worker response should be delivered back
```

The response worker receives shared worker responses and uses the request ID to
put each response into the matching per-request output queue.

</details>

<details>
<summary><strong>Q3. Why create <code>pending_out_queues[request_id]</code> before putting the context?</strong></summary>

The output route must exist before the worker can produce a response.

```text
create out_queue first
  |
  v
submit context to worker
  |
  v
worker response has somewhere to go
```

If `request_queue.put(context)` fails, the code removes that pending output
queue and sends cancellation best-effort, so stale request state is not left
behind.

</details>

<details>
<summary><strong>Q4. What does <code>batch_id</code> mean in the returned stream?</strong></summary>

The returned async generator yields:

```text
(outputs, batch_id)
```

`outputs` is a non-empty list of pipeline outputs for this request. `batch_id`
is the scheduler's forward-pass counter for the batch that produced those
outputs. Upstream code can use it to correlate tracing across the API process
and the worker process.

</details>

<details>
<summary><strong>Q5. What happens if the client disconnects while streaming?</strong></summary>

If the API stops consuming the response stream, `_drain_responses(...)` catches
the interruption and asks the worker to cancel the request.

```text
client disconnect / stream abandoned
  |
  v
_drain_responses(...) exits with exception
  |
  v
self.cancel(request_id)
  |
  v
cancel_queue.put_nowait([request_id])
```

</details>

## Where It Goes Next

After `stream(...)` accepts the `TextContext`, the scheduler drains the worker
request queue:

```text
request_queue
  |
  v
TextGenerationScheduler._retrieve_pending_requests()
  |
  v
batch_constructor.enqueue_new_request(context)
  |
  v
scheduler batching and model execution
```

## Short Version

```text
self.model_worker.stream(request_id, context)
  -> verifies request ID is unique
  -> checks worker request queue capacity
  -> registers a per-request output queue
  -> submits TextContext to the worker queue
  -> returns an async response stream
```
