# Complete Code Walk: `Llama3BatchProcessor`

Source:
[`batch_processor.py`](../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L38)

## Ragged Token And Offset Staging

```python
def _stage_ragged_token_inputs(
    self, context_batch: Sequence[TextContext]
) -> tuple[Buffer, Buffer, Buffer]:
    # WALKTHROUGH COMMENT (not in source): stage packed tokens and boundaries.
    batch_size = len(context_batch)
    total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)

    with self._stager.stage() as staging:
        host_tokens, (device_tokens,) = staging.get(
            RAGGED_INPUT_TOKENS, (total_seq_len,)
        )
        host_row_offsets, (device_row_offsets,) = staging.get(
            RAGGED_INPUT_ROW_OFFSETS, (batch_size + 1,)
        )
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=host_row_offsets.to_numpy(),
        )
        if context_batch:
            np.concatenate(
                [ctx.tokens.active for ctx in context_batch],
                out=host_tokens.to_numpy(),
            )
    return device_tokens, device_row_offsets, host_row_offsets
```

```text
context A active tokens: [a0, a1, a2]
context B active tokens: [b0, b1]

device_tokens:
[a0, a1, a2, b0, b1]

device_row_offsets:
[0, 3, 5]

row 0 = device_tokens[0:3]
row 1 = device_tokens[3:5]
```

## Initial Model Input Preparation

```python
def prepare_initial_token_inputs(
    self,
    replica_batches: Sequence[Sequence[TextContext]],
    kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
    return_n_logits: int = 1,
) -> InputsT:

    dp = self.runtime.pipeline_config.model.data_parallel_degree
    if len(replica_batches) != dp:
        raise ValueError(
            "Number of replica batches must match data parallel degree"
        )

    context_batch = flatten2d(replica_batches)

    device_tokens, device_row_offsets, _ = self._stage_ragged_token_inputs(
        context_batch
    )

    return_n_logits_tensor = Buffer.from_numpy(
        np.array([return_n_logits], dtype=np.int64)
    )

    if dp > 1:
        data_parallel_splits = Buffer.from_numpy(
            compute_data_parallel_splits(replica_batches)
        )
    else:
        data_parallel_splits = None

    return self._make_inputs(
        tokens=device_tokens,
        input_row_offsets=device_row_offsets,
        return_n_logits=return_n_logits_tensor,
        signal_buffers=list(self.runtime.signal_buffers),
        kv_cache_inputs=kv_cache_inputs,
        data_parallel_splits=data_parallel_splits,
    )
```

```text
replica_batches
      |
      +-- validate len == data_parallel_degree
      +-- flatten2d ------------------------> context_batch
      +-- stage active tokens -------------> device_tokens
      +-- cumulative active lengths -------> device_row_offsets
      +-- return_n_logits ------------------> scalar Buffer
      +-- DP > 1 ---------------------------> data_parallel_splits
      +-- runtime --------------------------> signal_buffers
      +-- KV manager -----------------------> kv_cache_inputs
      |
      v
_make_inputs(...)
```

## Concrete Llama Input Object

```python
class Llama3BatchProcessor(Llama3BatchProcessorBase["Llama3Inputs"]):
    """Ragged batching for models whose inputs are :class:`Llama3Inputs`."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        signal_buffers: list[Buffer],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None,
        data_parallel_splits: Buffer | None,
    ) -> Llama3Inputs:
        from .model import Llama3Inputs

        return Llama3Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            signal_buffers=signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
            data_parallel_splits=data_parallel_splits,
        )
```

```text
Llama3Inputs
    |
    +-- tokens: packed active token IDs
    +-- input_row_offsets: request boundaries in tokens
    +-- return_n_logits: number of trailing logits requested
    +-- signal_buffers: multi-device synchronization buffers
    +-- kv_cache_inputs: KV block/runtime buffers
    +-- data_parallel_splits: per-replica split metadata or None
```

## Graph Output Adaptation

```python
def process_outputs(
    self, outputs: Sequence[Buffer | object]
) -> ModelOutputs:
    return process_ragged_kv_outputs(
        outputs,
        return_logits=self.runtime.return_logits,
        return_hidden_states=self.runtime.return_hidden_states,
    )
```

```text
compiled graph raw output sequence
              |
              | process_ragged_kv_outputs(...)
              v
ModelOutputs
    +-- logits / next-token logits
    +-- optional logit offsets
    +-- optional hidden states
```
