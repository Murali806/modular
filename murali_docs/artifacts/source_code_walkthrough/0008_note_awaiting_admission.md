# `self.model_worker.note_awaiting_admission(...)`

Sources:

- Call site in
  [`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L357)
- Counter update in
  [`worker_interface/__init__.py`](../../../max/python/max/serve/worker_interface/__init__.py#L75)
- Periodic histogram sample in
  [`zmq_interface.py`](../../../max/python/max/serve/worker_interface/zmq_interface.py#L260)
- Metric emission in
  [`metrics.py`](../../../max/python/max/serve/telemetry/metrics.py#L1132)

## One-Line Purpose

`self.model_worker.note_awaiting_admission(...)` tracks how many accepted API
requests are still waiting on the API side before they are handed to the model
worker.

```python
self.model_worker.note_awaiting_admission(1)
...
response_stream = await self.model_worker.stream(context.request_id, context)
...
self.model_worker.note_awaiting_admission(-1)
```

## Big Picture

```text
HTTP request accepted
  |
  v
note_awaiting_admission(+1)
  |
  | request is still API-side
  | examples: tokenization, TextContext creation, pre-submit work
  v
model_worker.stream(request_id, context)
  |
  | handoff succeeded
  v
note_awaiting_admission(-1)
  |
  v
request is now inside worker/scheduler path
```

This is not admission control by itself. It is metric bookkeeping that tells
operators where requests are waiting.

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `delta` | `+1` when request starts waiting for worker admission; `-1` when it stops waiting. |
| Internal state | `_awaiting_admission_count` | Running count on the `ModelWorkerProxy` instance. |
| Metric output | `maxserve.num_requests_awaiting_admission` | Live up/down counter updated immediately. |
| Metric output | `maxserve.requests_awaiting_admission` | Periodic histogram sample of the running count. |
| Function return | `None` | It records state/metrics only. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
new HTTP request                       mark API-side backlog                   count increases
  request_id="req-123"               ------------------------------->          _awaiting_admission_count += 1

delta                                  emit live metric update                 metric event
  +1                                 ------------------------------->          num_requests_awaiting_admission +1

request                                tokenizer.new_context(...)              TextContext
  TextGenerationRequest              ------------------------------->          context for scheduler/worker

context                                model_worker.stream(...)                handoff result
  request_id + TextContext           ------------------------------->          response_stream

delta                                  mark handoff complete                   count decreases
  -1                                ------------------------------->           _awaiting_admission_count -= 1
```

## Lifecycle View

```text
+------------------------------------------------------------------------------------------------------+
| A SINGLE REQUEST                                                                                     |
|------------------------------------------------------------------------------------------------------|
| API route accepted request                                                                           |
|   |                                                                                                  |
|   v                                                                                                  |
| note_awaiting_admission(+1)                                                                          |
|   |                                                                                                  |
|   | Count means: "accepted, but not yet submitted to model worker"                                    |
|   |                                                                                                  |
|   +-- tokenizer.new_context(request)                                                                 |
|   |     builds TextContext                                                                           |
|   |                                                                                                  |
|   +-- create_buffered_detokenizer(...)                                                               |
|   |     prepares streaming decode helpers                                                            |
|   |                                                                                                  |
|   v                                                                                                  |
| await model_worker.stream(context.request_id, context)                                               |
|   |                                                                                                  |
|   +-- SUCCESS                                                                                        |
|   |     |                                                                                            |
|   |     v                                                                                            |
|   |   note_awaiting_admission(-1)                                                                    |
|   |     request has crossed into worker/scheduler path                                                |
|   |                                                                                                  |
|   +-- FAILURE before handoff                                                                         |
|         |                                                                                            |
|         v                                                                                            |
|       except block calls note_awaiting_admission(-1)                                                 |
|       counter is balanced before error propagates                                                    |
|------------------------------------------------------------------------------------------------------|
| END REQUEST ADMISSION TRACKING                                                                       |
+------------------------------------------------------------------------------------------------------+
```

## What `await` Means Here

`await` is a **Python** keyword used with async code. It is not Mojo-specific.

In this line:

```python
response_stream = await self.model_worker.stream(
    context.request_id,
    context,
)
```

the code calls an async Python method and pauses this request's coroutine until
the worker handoff finishes.

```text
Python coroutine: next_token_chunk(...)
  |
  v
await self.model_worker.stream(...)
  |
  +-- send/submit TextContext to model worker
  |
  +-- while waiting, event loop can run other Python tasks
  |
  v
response_stream is ready OR an exception is raised
```

Side-by-side:

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
context.request_id                     await async stream handoff              response_stream
  RequestID("req-123")               ------------------------------->          async stream of worker outputs

context                                if handoff fails                        exception
  TextContext(...)                    ------------------------------->          caught by except block
```

Important meaning for this metric:

```text
before await finishes
  -> request is still counted as awaiting admission

after await succeeds
  -> model_worker.stream(...) accepted the request
  -> note_awaiting_admission(-1)

if await raises
  -> request was not handed off successfully
  -> except block still calls note_awaiting_admission(-1)
```

## Why The `try` / `except` Matters

```text
note_awaiting_admission(+1)
  |
  +-- tokenization succeeds
  |     |
  |     v
  |   stream(...) succeeds
  |     |
  |     v
  |   note_awaiting_admission(-1)
  |
  +-- tokenization fails OR stream(...) submit fails
        |
        v
      except BaseException
        |
        v
      note_awaiting_admission(-1)
```

The important invariant:

```text
every +1 must have exactly one matching -1
```

Without the failure-path decrement, the metric would show a request still
waiting even after that request already failed.

## What The Metric Means

Important wording:

```text
"awaiting admission"
  =
"accepted by API server, but not yet admitted into model worker/scheduler"
```

So tokenization and API-side preprocessing are **inside** this measured window,
not before it.

```text
high awaiting-admission count
  |
  +-- likely pressure before completed worker handoff
  |     examples:
  |       - slow tokenization
  |           happens after note_awaiting_admission(+1)
  |           and before model_worker.stream(...)
  |
  |       - API-side preprocessing backlog
  |           also happens after API accept
  |           and before worker admission completes
  |
  |       - worker submit path blocked or failing
              this is the handoff boundary itself
              the request is not considered admitted until stream(...) succeeds
  |
  +-- not the same as scheduler queue depth
        scheduler queue starts after model_worker.stream(...) accepts context
```

Timeline:

```text
API server accepts request
  |
  v
note_awaiting_admission(+1)
  |
  +-- tokenization
  +-- API-side preprocessing
  +-- submit TextContext to worker via model_worker.stream(...)
  |
  v
note_awaiting_admission(-1)
  |
  v
worker/scheduler owns the request
```

## Metric Flow

```text
note_awaiting_admission(delta)
  |
  +-- update local running count
  |     _awaiting_admission_count += delta
  |
  +-- emit live metric
        METRICS.reqs_awaiting_admission(delta)
          |
          v
        maxserve.num_requests_awaiting_admission
```

For ZMQ worker interfaces, a background metrics task also samples the current
running count:

```text
_metrics_worker loop
  |
  +-- sleep
  |
  +-- read _awaiting_admission_count
  |
  v
METRICS.requests_awaiting_admission_dist(count)
  |
  v
maxserve.requests_awaiting_admission histogram
```

## Q&A

<details>
<summary><strong>Q1. What does "awaiting admission" mean here?</strong></summary>

It means the request has already reached the API server, but it has not yet
been handed to the model worker.

```text
accepted by API server
  |
  +-- awaiting admission
        |
        +-- tokenization / TextContext creation
        +-- pre-submit setup
        +-- waiting for model_worker.stream(...) handoff
```

Once `model_worker.stream(context.request_id, context)` succeeds, the request
is no longer counted as awaiting admission.

</details>

<details>
<summary><strong>Q2. Is this the same as model batching or scheduling?</strong></summary>

No. This metric is before the worker/scheduler accepts the request.

```text
API-side backlog                         worker/scheduler backlog
----------------                         ------------------------
note_awaiting_admission(+1)              after model_worker.stream(...)
tokenization                             scheduler queues
TextContext creation                     batching
pre-submit handoff                       model execution
```

So a high value here points to API-side pressure, not necessarily GPU or
scheduler pressure.

</details>

<details>
<summary><strong>Q3. Why use <code>+1</code> and <code>-1</code> instead of setting a number directly?</strong></summary>

Each request owns one increment and one decrement.

```text
request A starts waiting -> +1
request B starts waiting -> +1
request A handed off     -> -1
request B handed off     -> -1
```

That pattern works naturally with concurrent requests because each request only
adjusts its own lifecycle.

</details>

<details>
<summary><strong>Q4. Is <code>await</code> a Mojo keyword or a Python keyword?</strong></summary>

Here it is a **Python** keyword.

`llm.py` is Python code. `await` is part of Python's `async` / `await` model:

```text
async function
  |
  +-- calls another async function
        |
        v
      await waits for that async operation to complete
```

It does not mean "run Mojo code." It means "pause this Python coroutine until
the async operation returns or raises."

In this specific line:

```text
await model_worker.stream(...)
  |
  +-- returns response_stream if the worker handoff succeeds
  |
  +-- raises if the handoff fails
```

</details>

## Where It Goes Next

After the `-1` decrement, the request is no longer considered API-side backlog.
The next important handoff is:

```text
model_worker.stream(context.request_id, context)
  |
  v
worker IPC boundary
  |
  v
scheduler receives TextContext
```

## Short Version

```text
note_awaiting_admission(+1)
  -> request accepted, not yet handed to worker

note_awaiting_admission(-1)
  -> handoff completed or failed, stop counting it as API-side backlog

metric meaning
  -> "How many requests are stuck before worker admission?"
```
