# Llama Input Staging - `Llama3BatchProcessor`

Sources:

- Ragged token staging in
  [`batch_processor.py`](../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L57)
- `prepare_initial_token_inputs(...)` in
  [`batch_processor.py`](../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L93)
- `_make_inputs(...)` in
  [`batch_processor.py`](../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L166)
- `process_outputs(...)` in
  [`batch_processor.py`](../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L153)

<strong><em><a href="0013_A_1_llama_input_staging_ragged_buffers.md"><span style="color:#0b63ce">See complete code walkthrough: 0013_A_1_llama_input_staging_ragged_buffers.md</span></a></em></strong>.

## One-Line Purpose

`Llama3BatchProcessor` converts scheduled `TextContext` batches into the exact
`Llama3Inputs` buffers expected by the compiled Llama graph.

## Big Picture

```text
replica batches of TextContext
  |
  v
flatten contexts
  |
  v
stage ragged token buffer + row offsets
  |
  v
attach return_n_logits, signal buffers, KV-cache inputs, DP splits
  |
  v
Llama3Inputs
```

It is a buffer adapter. It does not run attention, MLP, sampling, or model math.

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `replica_batches` | Per-replica lists of scheduled `TextContext` objects. |
| Input | `kv_cache_inputs` | Runtime KV-cache buffers prepared by the KV manager. |
| Input | `return_n_logits` | How many final logits rows the graph should return. |
| Intermediate | `tokens` | Packed active token IDs for all contexts. |
| Intermediate | `input_row_offsets` | Ragged boundaries into the packed token buffer. |
| Output | `Llama3Inputs` | Model input object consumed by `LlamaModelBase.execute(...)`. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
replica_batches                        flatten2d(...)                         context_batch
  [[ctx-A], [ctx-B]]                 ------------------------------->          [ctx-A, ctx-B]

context_batch                          concatenate active tokens              tokens buffer
  A active=[a0,a1,a2]                ------------------------------->          [a0,a1,a2,b0,b1]
  B active=[b0,b1]

context active lengths                 np.cumsum([0, 3, 2])                   row offsets
  [3, 2]                             ------------------------------->          [0, 3, 5]

return_n_logits                        Buffer.from_numpy(...)                  return_n_logits buffer
  1                                  ------------------------------->          Buffer([1])

replica_batches                        compute_data_parallel_splits(...)       DP splits or None
  dp=1                               ------------------------------->          None

all prepared fields                    _make_inputs(...)                       Llama3Inputs
  tokens + offsets + KV              ------------------------------->          Llama3Inputs(...)
```

## Ragged Batch Visual

```text
Prompt A active tokens: [a0 a1 a2]
Prompt B active tokens: [b0 b1]

packed tokens:
  [a0 a1 a2 b0 b1]

input_row_offsets:
  [0, 3, 5]

Meaning:
  row 0 reads tokens[0:3] -> A
  row 1 reads tokens[3:5] -> B
```

Ragged batching avoids padding every request to the same sequence length.

## Staging Flow

```text
prepare_initial_token_inputs(replica_batches, kv_cache_inputs)
  |
  +-- verify len(replica_batches) == data_parallel_degree
  |
  +-- context_batch = flatten2d(replica_batches)
  |
  +-- _stage_ragged_token_inputs(context_batch)
  |     |
  |     +-- allocate host/device token buffers
  |     +-- allocate host/device row-offset buffers
  |     +-- write cumulative row offsets
  |     +-- concatenate ctx.tokens.active into host token buffer
  |
  +-- build return_n_logits tensor
  |
  +-- if dp > 1, build data_parallel_splits
  |
  v
_make_inputs(...)
  |
  v
Llama3Inputs(...)
```

## Output Processing

```text
raw graph outputs
  |
  v
process_outputs(...)
  |
  v
process_ragged_kv_outputs(...)
  |
  v
ModelOutputs(logits, next_token_logits, logit_offsets, ...)
```

The batch processor also translates graph outputs back into the common
`ModelOutputs` shape used by the generic text-generation pipeline.

## Q&A

<details>
<summary><strong>Q1. Why does Llama need <code>input_row_offsets</code>?</strong></summary>

Because the graph receives one packed token buffer for multiple requests.

```text
tokens = [a0, a1, a2, b0, b1]
offsets = [0, 3, 5]

request A -> tokens[0:3]
request B -> tokens[3:5]
```

Without offsets, the model would not know where one request ends and the next
request begins.

</details>

<details>
<summary><strong>Q2. Why does this stage create device buffers?</strong></summary>

The compiled graph runs on device-side buffers. The batch processor writes data
from Python/host-side context objects into buffers the executable can consume.

```text
TextContext tokens
  |
  v
host staging buffer
  |
  v
device Buffer
  |
  v
compiled graph input
```

</details>

## Where It Goes Next

The pipeline passes `Llama3Inputs` to the model wrapper:

```text
Llama3Inputs
  |
  v
LlamaModelBase.execute(model_inputs)
```

## Short Version

```text
Llama3BatchProcessor
  -> flatten replica batches
  -> pack active token IDs
  -> build ragged row offsets
  -> attach KV / signal / DP inputs
  -> return Llama3Inputs
```
