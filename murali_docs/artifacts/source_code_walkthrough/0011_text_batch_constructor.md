# Batch Construction - `TextBatchConstructor`

Sources:

- Request admission in
  [`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L677)
- `_admit_request(...)` in
  [`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L703)
- `_bind_request(...)` in
  [`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L725)
- `construct_batch(...)` in
  [`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1845)

## One-Line Purpose

The batch constructor turns admitted `TextContext` objects into concrete
`TextGenerationInputs`: per-replica lists of requests that can run in the next
model step.

## Big Picture

```text
TextContext from scheduler
  |
  v
enqueue_new_request(ctx)
  |
  +-- grammar not ready? hold in _grammar_pending
  |
  +-- DP CE balancing? hold in _ce_pending
  |
  +-- otherwise bind to replica
        |
        +-- generated_length == 0 -> CE / prefill queue
        +-- generated_length > 0  -> TG / decode queue
  |
  v
construct_batch()
  |
  v
TextGenerationInputs
```

CE means context encoding / prefill. TG means token generation / decode.

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `TextContext` | A request that has prompt tokens and generation state. |
| Pending state | `_grammar_pending` | Requests waiting for constrained-decoding grammar readiness. |
| Pending state | `_ce_pending` | Unbound data-parallel CE requests waiting for replica planning. |
| Replica queues | `replica.ce_reqs` | Requests that still need prompt/prefix processing. |
| Replica queues | `replica.tg_reqs` | Requests already generating one token at a time. |
| Output | `TextGenerationInputs` | Replica batches consumed by the pipeline. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
new request                            enqueue_new_request(ctx)                admission branch
  ctx.generated_length = 0           ------------------------------->          fresh CE request

grammar state                          submit/check grammar build              branch decision
  response_format=json_schema        ------------------------------->          ready or pending

DP CE balancing                        _admit_request(ctx)                     pending or bound
  enabled, replica_idx=None          ------------------------------->          _ce_pending[request_id]

replica choice                         _bind_request(ctx, replica_idx)         replica queue
  replica_idx=1                      ------------------------------->          replica[1].ce_reqs

replica queues                         construct_batch()                       TextGenerationInputs
  ce_reqs + tg_reqs                  ------------------------------->          batches=[[ctx-A], [ctx-B]]
```

## Admission Flow

```text
+------------------------------------------------------------------------------------------------------+
| enqueue_new_request(ctx)                                                                             |
|------------------------------------------------------------------------------------------------------|
| START                                                                                                |
|   |                                                                                                  |
|   v                                                                                                  |
| grammar gate exists AND ctx has generated_length == 0?                                               |
|   |                                                                                                  |
|   +-- YES                                                                                            |
|   |     |                                                                                            |
|   |     v                                                                                            |
|   |   submit grammar build                                                                           |
|   |     |                                                                                            |
|   |     +-- not ready -> _grammar_pending[request_id] = ctx -> RETURN                                |
|   |     |                                                                                            |
|   |     +-- ready but install error -> fail request -> RETURN                                        |
|   |                                                                                                  |
|   +-- NO or grammar ready                                                                            |
|         |                                                                                            |
|         v                                                                                            |
|       _admit_request(ctx, replica_idx)                                                               |
|------------------------------------------------------------------------------------------------------|
| END ENQUEUE                                                                                          |
+------------------------------------------------------------------------------------------------------+
```

## Replica Binding Flow

```text
_admit_request(ctx, replica_idx)
  |
  +-- DP CE balancing enabled
  |   AND caller did not pin replica
  |   AND this is fresh prefill?
  |     |
  |     v
  |   _ce_pending[request_id] = weighted request
  |
  +-- otherwise
        |
        v
      choose replica_idx
        |
        v
      _bind_request(ctx, replica_idx)
        |
        +-- generated_length == 0 -> replica.ce_reqs
        |
        +-- generated_length > 0  -> replica.tg_reqs
```

## Construct Batch Flow

```text
construct_batch()
  |
  +-- poll completed KV transfers
  |
  +-- readmit requests whose KV onload completed
  |
  +-- promote grammar-ready requests
  |
  +-- update prefill scheduling interval
  |
  +-- plan data-parallel CE placement
  |
  +-- choose CE/TG priority strategy
  |
  +-- construct one batch per replica
  |
  +-- optionally pad short DP replicas
  |
  v
TextGenerationInputs(batches=[...])
```

## Q&A

<details>
<summary><strong>Q1. Why separate CE and TG queues?</strong></summary>

Because prefill and decode have different shapes and costs.

```text
CE / prefill
  -> consumes prompt tokens
  -> can be long/chunked
  -> creates KV cache

TG / decode
  -> consumes latest active token(s)
  -> usually one generated step at a time
  -> extends KV cache
```

The batch constructor balances these two kinds of work each scheduler pass.

</details>

<details>
<summary><strong>Q2. Why can grammar delay admission?</strong></summary>

Structured output may need a matcher/grammar before the first token is sampled.

```text
request requires JSON grammar
  |
  +-- grammar ready
  |     -> request can enter CE/TG queues
  |
  +-- grammar not ready
        -> hold in _grammar_pending
```

This prevents the model from starting generation before constrained decoding
knows which next tokens are valid.

</details>

## Where It Goes Next

The scheduler sends the constructed batch into pipeline execution:

```text
TextGenerationInputs
  |
  v
TextGenerationPipeline.execute(inputs)
```

## Short Version

```text
TextBatchConstructor
  -> admits grammar-ready requests
  -> chooses replica placement
  -> separates prefill and decode work
  -> respects KV / DP / scheduling constraints
  -> returns TextGenerationInputs
```
