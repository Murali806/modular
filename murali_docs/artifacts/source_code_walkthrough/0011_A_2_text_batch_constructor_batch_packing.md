# Complete Code Walk: `TextBatchConstructor.construct_batch(...)`

Source:
[`text_batch_constructor.py`](../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L1845)

<strong><em><a href="0011_A_1_text_batch_constructor_request_admission.md"><span style="color:#0b63ce">Return to request admission: 0011_A_1_text_batch_constructor_request_admission.md</span></a></em></strong>.

## Per-Replica Packing Policy

```python
def _construct_replica_batch(
    self, replica_idx: int, priority_override: RequestType | None = None
) -> ReplicaBatch:
    # WALKTHROUGH COMMENT (not in source): pack one replica under its budget.
    ce_capacity: int | None = None
    if (
        self._ce_step_quota is not None
        and self._ce_step_quota[replica_idx] > 0
        and not self.scheduler_config.enable_in_flight_batching
    ):
        ce_capacity = self._ce_step_quota[replica_idx]

    batch = ReplicaBatch(
        batch={},
        token_budget=self._create_new_token_budget(ce_capacity),
    )

    priority = (
        priority_override
        if priority_override is not None
        else self._identify_priority(replica_idx)
    )

    match priority:
        case RequestType.CE:
            self._add_ce_requests(batch, replica_idx)

            if len(batch) == 0 and priority_override is None:
                self._add_tg_requests(batch, replica_idx)
            else:
                pending_tg_count = len(self.replicas[replica_idx].tg_reqs)
                if len(batch) > 0 and pending_tg_count > 0:
                    METRICS.di_ce_preempted_tg_iteration_count()
                    METRICS.di_ce_preempted_tg_pending_count(
                        pending_tg_count
                    )

        case RequestType.TG:
            self._add_tg_requests(batch, replica_idx)

            if (
                self.scheduler_config.enable_in_flight_batching
                and len(batch) > 0
                and priority_override is None
                and self._ce_backfill_open
            ):
                self._add_ce_requests(batch, replica_idx)

        case None:
            pass

    return batch
```

```text
replica queues + token budget
          |
          | priority = CE / TG / None
          v
    +-----+-------------------------------+
    |                                     |
   CE                                    TG
    |                                     |
    | add CE requests                     | add TG requests
    | no CE admitted?                     | in-flight batching and CE backfill open?
    +-- yes --> fallback to TG             +-- yes --> add CE requests too
    |
   None --> empty batch
```

## Full Multi-Replica Construction

```python
def construct_batch(self) -> TextGenerationInputs[TextContext]:
    # WALKTHROUGH COMMENT (not in source): coordinate all replica batches.
    if self.kv_cache is not None:
        self.kv_cache.poll_transfers()

    self._readmit_completed_onloads()
    self._promote_grammar_ready_requests()

    self._prefill_interval_open = self._is_prefill_interval_open()
    interval = self.scheduler_config.prefill_schedule_interval
    phase = self._prefill_interval_phase + 1
    self._prefill_interval_phase = phase % interval

    self._plan_ce_step()
    self._ce_backfill_open = self._should_backfill_ce()

    priority_override = None
    replica_priorities: set[RequestType | None] | list[RequestType | None]
    match self.batch_scheduling_strategy:
        case BatchSchedulingStrategy.DECODE_FIRST:
            replica_priorities = {
                self._identify_priority(idx)
                for idx in range(self.num_replicas)
            }
            if RequestType.TG in replica_priorities:
                priority_override = RequestType.TG
        case BatchSchedulingStrategy.PREFILL_FIRST:
            replica_priorities = {
                self._identify_priority(idx)
                for idx in range(self.num_replicas)
            }
            if RequestType.CE in replica_priorities:
                priority_override = RequestType.CE
        case BatchSchedulingStrategy.BALANCED:
            replica_priorities = [
                self._identify_priority(idx)
                for idx in range(self.num_replicas)
            ]
            ce_count = replica_priorities.count(RequestType.CE)
            tg_count = replica_priorities.count(RequestType.TG)
            if ce_count > tg_count:
                priority_override = RequestType.CE
            else:
                priority_override = RequestType.TG

    with METRICS.transaction():
        batches_per_replica = [
            self._construct_replica_batch(
                replica_idx, priority_override=priority_override
            )
            for replica_idx in range(self.num_replicas)
        ]

    inputs = TextGenerationInputs[TextContext](
        batches=[
            list(batch.batch.values()) for batch in batches_per_replica
        ],
    )

    if self._dp_padder is not None:
        inputs, info = self._dp_padder.pad_batch(inputs)
        self._current_dp_padding = info

    return inputs
```

```text
queued CE/TG requests across replicas
              |
              +-- settle KV transfers/onloads
              +-- promote grammar-ready requests
              +-- update prefill cadence
              +-- plan data-parallel CE work
              +-- choose group priority override
              |
              v
_construct_replica_batch(replica 0..N-1)
              |
              v
batches_per_replica: list[ReplicaBatch]
              |
              | extract ordered context values
              v
TextGenerationInputs(batches=[...])
              |
              | optional DP padding for graph capture
              v
pipeline-ready TextGenerationInputs
```

## Output Shape

```text
TextGenerationInputs
|
+-- batches[0] -> [ctx_A, ctx_B, ...]  replica 0
+-- batches[1] -> [ctx_C, ...]         replica 1
+-- ...
|
+-- flat_batch -> concatenated view consumed by pipeline execution
```
