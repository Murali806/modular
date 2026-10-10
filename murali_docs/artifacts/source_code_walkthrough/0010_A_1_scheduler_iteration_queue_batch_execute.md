# Complete Code Walk: Scheduler `run_iteration(...)`

Source:
[`text_generation_scheduler.py`](../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L161)

## Drain Newly Admitted Contexts

```python
@traced
def _retrieve_pending_requests(self) -> None:
    max_items = self.max_items_per_drain
    if self.max_pending_requests is not None:
        # Cap M: only pull enough to keep the pending (CE/prefill) queue at
        # or below max_pending_requests. Anything beyond that stays in the
        # request queue, backing it up so the API can shed load.
        available = self.max_pending_requests - len(
            self.batch_constructor.all_ce_reqs
        )
        if available <= 0:
            return
        max_items = min(max_items, available)

    with Tracer("drain_queue"):
        items = drain_queue(
            self.request_queue,
            max_items=max_items,
        )

    with Tracer(f"adding_to_batch_constructor: {len(items)} items"):
        tracing_enabled = _tracing_enabled()
        for context in items:
            self.batch_constructor.enqueue_new_request(context)
            if tracing_enabled:
                self._prefill_spans[context.request_id] = (
                    _tracer.start_span(
                        "max.phase.prefill",
                        context=_parent_trace_context(context),
                        attributes={
                            "max.request_id": str(context.request_id)
                        },
                    )
                )
```

```text
worker request_queue
       |
       | max_pending_requests capacity check
       v
drain_queue(..., max_items)
       |
       v
list[TextContext]
       |
       | enqueue_new_request(context)
       v
batch_constructor pending/replica queues
```

## Complete Iteration Control Flow

```python
@traced
def run_iteration(self) -> SchedulerProgress:
    # WALKTHROUGH COMMENT (not in source): one scheduler control-loop pass.
    t0 = time.monotonic()
    self._retrieve_pending_requests()

    inputs = self.batch_constructor.construct_batch()
    t1 = time.monotonic()
    batch_creation_time_s = t1 - t0

    # Failed requests were never admitted and are already released.
    for failed_id, error in self.batch_constructor.take_grammar_failed():
        self.response_queue.put_nowait(
            {failed_id: SchedulerResult.failed(error)}
        )
        if failed_id in self._prefill_spans:
            self._prefill_spans.pop(failed_id).end()

    has_pending_outputs = (
        isinstance(self.pipeline, OverlapTextGenerationPipeline)
        and self.pipeline.has_pending_outputs()
    )
    if not (inputs or self.support_empty_batches or has_pending_outputs):
        return SchedulerProgress.NO_PROGRESS

    is_overlap_active = bool(
        getattr(self.pipeline, "overlap_active", False)
    )

    t0 = time.monotonic()
    if len(inputs.flat_batch) > 0:
        with Tracer(f"_schedule({inputs})"):
            num_terminated_reqs = self._schedule(inputs)
    else:
        num_terminated_reqs = self._schedule(inputs)
    t1 = time.monotonic()
    batch_execution_time_s = t1 - t0

    completed_batch_stats = (
        self.pipeline.take_completed_batch_stats()
        if hasattr(self.pipeline, "take_completed_batch_stats")
        else None
    )
    self.scheduler_logger.log_metrics(
        sch_config=self.scheduler_config,
        inputs=inputs,
        kv_cache=self.batch_constructor.kv_cache,
        batch_creation_time_s=batch_creation_time_s,
        batch_execution_time_s=batch_execution_time_s,
        num_pending_reqs=len(self.batch_constructor.all_ce_reqs),
        num_terminated_reqs=num_terminated_reqs,
        total_preemption_count=self.batch_constructor.total_preemption_count,
        batch_spec_decode_metrics=self.pipeline.batch_spec_decode_metrics()
        if hasattr(self.pipeline, "batch_spec_decode_metrics")
        else None,
        batch_vision_metrics=self.pipeline.batch_vision_metrics()
        if hasattr(self.pipeline, "batch_vision_metrics")
        else None,
        batch_video_metrics=self.pipeline.batch_video_metrics()
        if hasattr(self.pipeline, "batch_video_metrics")
        else None,
        overlap_active=is_overlap_active,
        completed_batch_stats=completed_batch_stats,
    )

    for cancelled_id in get_cancelled_reqs(self.cancel_queue):
        if self.batch_constructor.contains(cancelled_id):
            self.batch_constructor.release_request(cancelled_id)
            self.response_queue.put_nowait(
                {cancelled_id: SchedulerResult.cancelled()}
            )
        if cancelled_id in self._prefill_spans:
            self._prefill_spans.pop(cancelled_id).end()
        if cancelled_id in self._decode_spans:
            self._decode_spans.pop(cancelled_id).end()

    return SchedulerProgress.MADE_PROGRESS
```

```text
run_iteration
     |
     +-- _retrieve_pending_requests()
     +-- construct_batch()
     +-- publish grammar-build failures
     |
     +-- no inputs, no empty-batch support, no overlap output?
     |       +-- yes --> NO_PROGRESS
     |
     +-- _schedule(inputs)
     +-- log scheduler/KV/batch metrics
     +-- drain cancellations and release state
     |
     v
MADE_PROGRESS
```

## Execute, Advance, Release, Publish

```python
def _schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
    tracing_enabled = _tracing_enabled()
    batch_id = self._batch_counter
    self._batch_counter += 1

    ce_ids_before = (
        {
            ctx.request_id
            for batch, replica in zip(
                inputs.batches, self.batch_constructor.replicas, strict=True
            )
            for ctx in batch
            if not ctx._is_padding_ctx
            and ctx.request_id not in replica.tg_reqs
        }
        if tracing_enabled
        else None
    )

    batch_span: otel_trace.Span = otel_trace.INVALID_SPAN
    if tracing_enabled and batch_spans_enabled():
        assert ce_ids_before is not None
        batch_span = _tracer.start_span(
            "max.batch",
            attributes={
                "max.batch_id": batch_id,
                "max.ce_count": len(ce_ids_before),
                "max.tg_count": inputs.batch_size - len(ce_ids_before),
            },
        )

    try:
        _request_context.set_batch_id(batch_id)
        batch_id_token = (
            _batch_id_ctx.set(batch_id) if self._tracing else None
        )
        try:
            responses = self.pipeline.execute(inputs)
        finally:
            if batch_id_token is not None:
                _batch_id_ctx.reset(batch_id_token)
            _request_context.clear_batch_id()

        responses = {
            req_id: response
            for req_id, response in responses.items()
            if self.batch_constructor.contains(req_id)
        }

        self.batch_constructor.advance_requests(inputs)

        num_terminated_requests = 0
        for request_id, response in responses.items():
            if response.is_done:
                self.batch_constructor.release_request(request_id)
                num_terminated_requests += 1
                if request_id in self._decode_spans:
                    self._decode_spans.pop(request_id).end()

        if responses:
            self.response_queue.put_nowait(
                {
                    req_id: SchedulerResult.create(response, batch_id)
                    for req_id, response in responses.items()
                }
            )

        batch_span.set_attribute(
            "max.terminated_count", num_terminated_requests
        )
        return num_terminated_requests
    finally:
        batch_span.end()
```

```text
TextGenerationInputs
       |
       | pipeline.execute(inputs)
       v
responses by request_id
       |
       +-- remove responses for already-released requests
       +-- advance_requests(inputs): CE/prefill -> TG/decode progress
       +-- response.is_done --> release_request(request_id)
       +-- wrap each response in SchedulerResult(batch_id)
       |
       v
response_queue -> API process
```
