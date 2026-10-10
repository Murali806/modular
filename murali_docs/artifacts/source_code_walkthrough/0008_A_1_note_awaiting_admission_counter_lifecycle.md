# Complete Code Walk: `note_awaiting_admission(...)`

Sources:
[`llm.py`](../../../max/python/max/serve/pipelines/llm.py#L357),
[`worker_interface/__init__.py`](../../../max/python/max/serve/worker_interface/__init__.py#L75),
[`zmq_interface.py`](../../../max/python/max/serve/worker_interface/zmq_interface.py#L260),
[`metrics.py`](../../../max/python/max/serve/telemetry/metrics.py#L1132)

## Call-Site Lifecycle

```python
# Count this request as awaiting admission to the model worker: it has
# been accepted by the API server but is still API-side (tokenization /
# pre-submit). Decremented once the handoff to the worker succeeds (or
# fails) below, so a persistently high gauge points at an API-server
# backlog rather than the scheduler queue.
self.model_worker.note_awaiting_admission(1)

try:
    with record_ms(METRICS.input_time):
        context = await self.tokenizer.new_context(request)

    # WALKTHROUGH COMMENT (not in source):
    # Other API-side preparation occurs here before the worker handoff.
    response_stream = await self.model_worker.stream(
        context.request_id, context
    )
except BaseException:
    # Balance the awaiting-admission counter if we never reached a
    # successful handoff (tokenization failed or the submit raised).
    self.model_worker.note_awaiting_admission(-1)
    raise

# Handoff succeeded: the request is no longer awaiting admission.
self.model_worker.note_awaiting_admission(-1)
```

```text
API accepts request
       |
       | note_awaiting_admission(+1)
       v
count increases
       |
       | tokenizer.new_context + API-side preparation + worker submit
       v
  +----+----------------------------+
  |                                 |
failure before successful handoff   successful worker handoff
  |                                 |
  | note_awaiting_admission(-1)     | note_awaiting_admission(-1)
  v                                 v
count balanced; raise               count balanced; stream responses
```

## Counter Update

```python
class ModelWorkerProxy(ABC, Generic[BaseContextType, PipelineOutputType]):
    # Running count of requests accepted by the API server but not yet handed
    # off to this worker (the ingress backlog: tokenization / pre-submit).
    _awaiting_admission_count: int = 0

    def note_awaiting_admission(self, delta: int) -> None:
        """Adjust the ingress backlog (API-side, not yet handed to the worker).

        Call with ``1`` when a request is accepted by the API server (before
        tokenization) and ``-1`` once it is handed off to the worker. Updates
        both the live ``maxserve.num_requests_awaiting_admission`` up/down
        counter and the running count that implementations sample into the
        ``maxserve.requests_awaiting_admission`` histogram.
        """
        self._awaiting_admission_count += delta
        METRICS.reqs_awaiting_admission(delta)
```

```text
delta
  |
  +-- updates in-process count: self._awaiting_admission_count += delta
  |
  +-- emits live telemetry: METRICS.reqs_awaiting_admission(delta)
```

## Live Metric Emission

```python
def reqs_awaiting_admission(self, value: int) -> None:
    """Adjust the count of API-side requests not yet handed to the worker.

    ``maxserve.num_requests_awaiting_admission`` is an up/down counter:
    call with ``1`` when a request is accepted by the API server (before
    tokenization) and ``-1`` just before it is enqueued to the model
    worker. A persistently high value means requests are backing up in the
    API server (e.g. tokenization) rather than in the scheduler queue.
    """
    self.client.send_measurement(
        MaxMeasurement(
            "maxserve.num_requests_awaiting_admission",
            value,
            self.extra_attributes,
        ),
    )
```

## Periodic Distribution Sample

```python
async def _metrics_worker(self) -> None:
    """Periodically samples backlog gauges and histograms."""
    while True:
        await asyncio.sleep(_BACKLOG_SAMPLE_INTERVAL_S)
        backlog = self.egress_backlog()
        METRICS.responses_buffered(backlog)
        METRICS.responses_buffered_dist(backlog)
        METRICS.requests_awaiting_admission_dist(
            self._awaiting_admission_count
        )
```

```text
_awaiting_admission_count
       |
       | sampled every _BACKLOG_SAMPLE_INTERVAL_S
       v
requests_awaiting_admission_dist(value)

Live counter: records each +1/-1 transition.
Histogram: records periodic snapshots of backlog depth.
```
