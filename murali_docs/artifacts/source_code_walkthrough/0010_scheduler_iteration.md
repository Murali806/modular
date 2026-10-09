# Scheduler Iteration - `run_iteration(...)`

Sources:

- Scheduler loop in
  [`text_generation_scheduler.py`](../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L205)
- Pending request drain in
  [`text_generation_scheduler.py`](../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L161)
- Batch execution in
  [`text_generation_scheduler.py`](../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L299)
- `SchedulerProgress` in
  [`base.py`](../../../max/python/max/serve/scheduler/base.py#L24)

## One-Line Purpose

`run_iteration(...)` is one pass of the text-generation scheduler loop: drain
new worker-queue requests, build the next batch, execute it, publish results,
handle cancellations, and report whether useful work happened.

## Big Picture

```text
worker request_queue
  |
  v
_retrieve_pending_requests()
  |
  v
batch_constructor.construct_batch()
  |
  v
_schedule(inputs)
  |
  v
response_queue.put_nowait(...)
  |
  v
API response_worker routes result by request_id
```

This file is the control loop. It does not perform transformer math. It decides
when available `TextContext` objects become executable model work.

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input queue | `request_queue` | Worker-side queue containing admitted `TextContext` objects. |
| Component | `batch_constructor` | Owns pending CE/TG requests and builds `TextGenerationInputs`. |
| Component | `pipeline` | Executes the chosen batch and returns per-request outputs. |
| Output queue | `response_queue` | Sends `SchedulerResult` objects back to the API process. |
| Return | `SchedulerProgress` | `MADE_PROGRESS` or `NO_PROGRESS` for the serving loop. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
request_queue                          drain available contexts               new contexts
  [ctx-A, ctx-B]                     ------------------------------->          [ctx-A, ctx-B]

new contexts                           enqueue into batch constructor          pending scheduler state
  [ctx-A, ctx-B]                     ------------------------------->          ce_reqs / tg_reqs updated

pending scheduler state                construct_batch()                      TextGenerationInputs
  ce_reqs + tg_reqs                  ------------------------------->          replica batches

TextGenerationInputs                   _schedule(inputs)                      terminated count
  flat_batch=[ctx-A, ctx-B]          ------------------------------->          num_terminated_reqs

pipeline outputs                       response_queue.put_nowait(...)          API-visible worker result
  {ctx-A: output}                    ------------------------------->          SchedulerResult
```

## Runtime Flow

```text
+------------------------------------------------------------------------------------------------------+
| run_iteration()                                                                                      |
|------------------------------------------------------------------------------------------------------|
| START                                                                                                |
|   |                                                                                                  |
|   v                                                                                                  |
| _retrieve_pending_requests()                                                                         |
|   |                                                                                                  |
|   | pulls admitted TextContext objects from worker request_queue                                      |
|   v                                                                                                  |
| batch_constructor.construct_batch()                                                                  |
|   |                                                                                                  |
|   | returns TextGenerationInputs, possibly empty                                                      |
|   v                                                                                                  |
| take_grammar_failed()                                                                                |
|   |                                                                                                  |
|   | failed grammar requests are sent back as SchedulerResult.failed                                   |
|   v                                                                                                  |
| any inputs OR empty-batch support OR pending overlap outputs?                                        |
|   |                                                                                                  |
|   +-- NO                                                                                             |
|   |     |                                                                                            |
|   |     v                                                                                            |
|   |   return SchedulerProgress.NO_PROGRESS                                                           |
|   |                                                                                                  |
|   +-- YES                                                                                            |
|         |                                                                                            |
|         v                                                                                            |
|       _schedule(inputs)                                                                              |
|         |                                                                                            |
|         v                                                                                            |
|       log scheduler / KV / batch metrics                                                             |
|         |                                                                                            |
|         v                                                                                            |
|       process cancellation queue                                                                     |
|         |                                                                                            |
|         v                                                                                            |
|       return SchedulerProgress.MADE_PROGRESS                                                         |
|------------------------------------------------------------------------------------------------------|
| END ITERATION                                                                                        |
+------------------------------------------------------------------------------------------------------+
```

## Landmark Map

```text
_retrieve_pending_requests()
  |
  +-- respects max_pending_requests cap
  +-- drains request_queue
  +-- calls batch_constructor.enqueue_new_request(context)
  +-- starts prefill tracing span, if tracing is enabled

batch_constructor.construct_batch()
  |
  +-- chooses CE/prefill and TG/decode work
  +-- returns TextGenerationInputs

_schedule(inputs)
  |
  +-- assigns batch_id
  +-- calls pipeline.execute(inputs)
  +-- pushes SchedulerResult objects to response_queue

SchedulerProgress
  |
  +-- MADE_PROGRESS: this iteration did useful work
  +-- NO_PROGRESS: nothing runnable was available
```

## Q&A

<details>
<summary><strong>Q1. Why can an iteration return <code>NO_PROGRESS</code>?</strong></summary>

Because the scheduler may wake up when there are no new requests and no active
batch work.

```text
no new contexts
  +
no constructed inputs
  +
no pending overlap outputs
  =
NO_PROGRESS
```

The outer serving loop can then sleep/back off instead of busy-spinning.

</details>

<details>
<summary><strong>Q2. Where do worker responses get created?</strong></summary>

Inside `_schedule(inputs)`, after the pipeline executes. The scheduler wraps
per-request outputs as `SchedulerResult` objects and puts them onto
`response_queue`.

```text
pipeline.execute(inputs)
  |
  v
per-request TextGenerationOutput
  |
  v
SchedulerResult
  |
  v
response_queue
```

</details>

## Where It Goes Next

`run_iteration(...)` delegates request packing to the batch constructor:

```text
batch_constructor.construct_batch()
  |
  v
TextGenerationInputs
  |
  v
pipeline.execute(inputs)
```

## Short Version

```text
run_iteration()
  -> drain admitted requests
  -> construct next batch
  -> execute batch
  -> publish results
  -> process cancellations
  -> report MADE_PROGRESS or NO_PROGRESS
```
