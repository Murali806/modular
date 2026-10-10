# Complete Code Walk: `TextBatchConstructor` Request Admission

Source:
[`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L677)

<strong><em><a href="0011_A_2_text_batch_constructor_batch_packing.md"><span style="color:#0b63ce">Continue to batch packing: 0011_A_2_text_batch_constructor_batch_packing.md</span></a></em></strong>.

## Grammar Gate

```python
def submit_grammar_build(self, ctx: TextContext) -> None:
    # WALKTHROUGH COMMENT (not in source): start an idempotent async build.
    if self._grammar_gate is not None:
        self._grammar_gate.submit(ctx)


def release_grammar_build(self, request_id: RequestID) -> None:
    # WALKTHROUGH COMMENT (not in source): release a pending build if present.
    if self._grammar_gate is not None:
        self._grammar_gate.release(request_id)
```

## New Request Entry

```python
def enqueue_new_request(
    self, ctx: TextContext, replica_idx: int | None = None
) -> None:
    # WALKTHROUGH COMMENT (not in source): grammar readiness gates admission.
    # Decode-side requests (generated_length != 0) built their matcher at
    # prefill admission.
    if self._grammar_gate is not None and ctx.tokens.generated_length == 0:
        self._grammar_gate.submit(ctx)
        if not self._grammar_gate.is_ready(ctx):
            self._grammar_pending[ctx.request_id] = _GrammarPendingRequest(
                ctx=ctx, replica_idx=replica_idx
            )
            return
        error = self._grammar_gate.install_ready(ctx)
        if error is not None:
            self._fail_grammar_request(ctx, error)
            return

    self._admit_request(ctx, replica_idx)
```

```text
TextContext
    |
    | grammar gate exists and generated_length == 0?
    v
+---+---+
|       |
no      yes
|       |
|       | submit(ctx)
|       | matcher ready?
|       +-- no --> _grammar_pending[request_id] = ctx; return
|       +-- yes --> install_ready(ctx)
|                    +-- error --> fail request; return
|                    +-- success
+-------+
    |
    v
_admit_request(ctx, replica_idx)
```

## DP Pool Or Immediate Replica Binding

```python
def _admit_request(self, ctx: TextContext, replica_idx: int | None) -> None:
    """Admits a grammar-ready request into the DP pool or a replica queue."""
    if (
        self._dp_ce_balance_enabled
        and replica_idx is None
        and ctx.tokens.generated_length == 0
    ):
        self._ce_pending[ctx.request_id] = _PendingCERequest(
            ctx=ctx, weights=self._post_cache_weights(ctx)
        )
        self._ce_arrival[ctx.request_id] = time.monotonic()
        return

    if replica_idx is None:
        replica_idx = self.get_next_replica_idx()
    self._bind_request(ctx, replica_idx)
```

```text
grammar-ready context
        |
        | DP CE balancing enabled + no pinned replica + fresh prefill?
        v
    +---+---+
    |       |
   yes      no
    |       |
    |       +-- replica_idx missing --> round-robin/get_next_replica_idx()
    |       +-- _bind_request(ctx, replica_idx)
    |
    +--> _ce_pending[request_id]
         store per-replica post-cache weights and arrival time
```

## Bind To CE Or TG Queue

```python
def _bind_request(
    self,
    ctx: TextContext,
    replica_idx: int,
    *,
    ce_weight: int | None = None,
) -> None:
    # WALKTHROUGH COMMENT (not in source): bind to CE or TG queue by progress.
    replica = self.replicas[replica_idx]

    if ctx.tokens.generated_length == 0:
        if self._dp_ce_balance_enabled and ce_weight is None:
            ce_weight = self._post_cache_weights(ctx)[replica_idx]
        replica.ce_reqs[ctx.request_id] = ctx
    else:
        ce_weight = None
        replica.tg_reqs[ctx.request_id] = ctx

    self._bound_requests[ctx.request_id] = _BoundRequest(
        ctx, replica_idx, ce_weight
    )
    self._request_id_to_lora_name[ctx.request_id] = (
        ctx.model_name
        if self._lora_manager and is_lora(ctx, self._lora_manager)
        else None
    )
```

```text
ctx.tokens.generated_length
        |
    +---+---+
    |       |
   == 0    > 0
    |       |
    v       v
replica.ce_reqs   replica.tg_reqs
(prefill)         (decode)
    |       |
    +---+---+
        |
        +-- _bound_requests records replica and CE weight
        +-- _request_id_to_lora_name records adapter selection
```

<strong><em><a href="0011_A_2_text_batch_constructor_batch_packing.md"><span style="color:#0b63ce">Continue to batch packing: 0011_A_2_text_batch_constructor_batch_packing.md</span></a></em></strong>.
