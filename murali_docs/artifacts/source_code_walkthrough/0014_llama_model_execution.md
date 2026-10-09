# Llama Model Execution - `model.py`

Sources:

- `Llama3Inputs.buffers` in
  [`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L65)
- Runtime `execute(...)` in
  [`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L183)
- `_create_model_config(...)` in
  [`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L191)
- `_build_graph_for_compile(...)` in
  [`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L205)
- `Llama3Model.__init__(...)` in
  [`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L316)

## One-Line Purpose

`model.py` owns the Llama pipeline model wrapper: at runtime it calls the
compiled executable with `Llama3Inputs.buffers`; at load/compile time it builds
the Llama graph topology and model config.

## Big Picture

```text
Runtime path
  |
  v
Llama3Inputs
  |
  v
model_inputs.buffers
  |
  v
self.model.execute(*buffers)
  |
  v
raw graph outputs
  |
  v
batch_processor.process_outputs(...)
  |
  v
ModelOutputs
```

```text
Compile/load path
  |
  v
pipeline config + HF config + weights
  |
  v
Llama3Config
  |
  v
single-device OR tensor-parallel OR data-parallel graph
  |
  v
compiled executable model
```

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Runtime input | `Llama3Inputs` | Tokens, row offsets, return-logit control, signal buffers, KV-cache inputs, optional DP splits. |
| Runtime action | `model_inputs.buffers` | Converts named fields into positional buffers in graph input order. |
| Runtime action | `self.model.execute(...)` | Calls the compiled MAX executable. |
| Runtime output | `ModelOutputs` | Common output object with logits/next-token logits/offsets. |
| Compile input | `state_dict` | Adapted model weights. |
| Compile output | `Graph` + weights registry | Graph ready for compilation/loading. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
model_inputs                           assert Llama3Inputs                     validated runtime input
  Llama3Inputs(...)                  ------------------------------->          OK

model_inputs                           require KV cache inputs                 validated KV state
  kv_cache_inputs != None            ------------------------------->          OK

Llama3Inputs fields                    model_inputs.buffers                    positional graph args
  tokens, offsets, KV                ------------------------------->          (tokens, offsets, return_n_logits, ...)

positional graph args                  self.model.execute(*buffers)            raw graph outputs
  Buffer tuple                       ------------------------------->          Sequence[Buffer | object]

raw graph outputs                      batch_processor.process_outputs(...)    ModelOutputs
  logits buffers                     ------------------------------->          ModelOutputs(...)
```

## `Llama3Inputs.buffers`

This property defines the positional argument order passed into the compiled
graph.

```text
Llama3Inputs
  |
  +-- tokens
  +-- input_row_offsets
  +-- return_n_logits
  +-- signal_buffers OR data_parallel_splits
  +-- flattened KV-cache inputs
  |
  v
tuple[Buffer, ...]
  |
  v
self.model.execute(*model_inputs.buffers)
```

Two shapes exist:

```text
without data_parallel_splits
  (tokens,
   input_row_offsets,
   return_n_logits,
   *signal_buffers,
   *kv_cache_inputs)

with data_parallel_splits
  (tokens,
   input_row_offsets,
   return_n_logits,
   splits_tensor,
   *kv_cache_inputs)
```

## Runtime Execute Flow

```text
+------------------------------------------------------------------------------------------------------+
| LlamaModelBase.execute(model_inputs)                                                                 |
|------------------------------------------------------------------------------------------------------|
| START                                                                                                |
|   |                                                                                                  |
|   v                                                                                                  |
| assert isinstance(model_inputs, Llama3Inputs)                                                        |
|   |                                                                                                  |
|   v                                                                                                  |
| assert model_inputs.kv_cache_inputs is not None                                                      |
|   |                                                                                                  |
|   v                                                                                                  |
| self.model.execute(*model_inputs.buffers)                                                            |
|   |                                                                                                  |
|   | compiled MAX graph runs transformer computation                                                   |
|   v                                                                                                  |
| raw graph outputs                                                                                    |
|   |                                                                                                  |
|   v                                                                                                  |
| batch_processor.process_outputs(model_outputs)                                                       |
|   |                                                                                                  |
|   v                                                                                                  |
| ModelOutputs                                                                                         |
|------------------------------------------------------------------------------------------------------|
| END RUNTIME EXECUTE                                                                                  |
+------------------------------------------------------------------------------------------------------+
```

## Compile-Time Graph Choice

```text
_build_graph_for_compile(...)
  |
  +-- data_parallel_degree > 1?
  |     |
  |     v
  |   create_data_parallel_graph(...)
  |
  +-- else len(devices) > 1?
  |     |
  |     v
  |   _build_tensor_parallel_graph_for_compile(...)
  |     |
  |     +-- DistributedLlama3(model_config)
  |     +-- Graph(..., input_types=dist_model.input_types(...))
  |
  +-- else
        |
        v
      _build_single_device_graph_for_compile(...)
        |
        +-- Llama3(model_config)
        +-- Graph("llama3", input_types=single_model.input_types(...))
```

## Q&A

<details>
<summary><strong>Q1. Does <code>execute(...)</code> build the model graph?</strong></summary>

No. Runtime `execute(...)` calls an already compiled executable.

```text
compile/load time
  -> create config
  -> build graph
  -> compile executable

runtime
  -> pass buffers to executable
  -> receive output buffers
```

</details>

<details>
<summary><strong>Q2. Why use <code>model_inputs.buffers</code> instead of passing named fields?</strong></summary>

The compiled graph expects positional inputs in a fixed order.

```text
named Python object
  Llama3Inputs(tokens=..., input_row_offsets=..., ...)

buffers property
  -> ordered tuple of Buffer objects

compiled graph
  -> receives positional arguments
```

The property is the contract between the Python input object and the compiled
graph signature.

</details>

<details>
<summary><strong>Q3. Why assert KV-cache inputs exist?</strong></summary>

Text generation uses KV cache state to avoid recomputing all previous tokens on
every decode step.

```text
prompt/decode context
  |
  +-- current active tokens
  +-- previous key/value cache
  |
  v
next-token forward pass
```

For this runtime path, missing KV inputs means the batch was not prepared for
the compiled generation graph correctly.

</details>

## Where It Goes Next

`ModelOutputs` returns to the generic text-generation pipeline:

```text
ModelOutputs
  |
  v
logits_for_sampling(...)
  |
  v
sampler chooses generated token IDs
  |
  v
TextGenerationOutput
```

## Short Version

```text
Llama model.py
  -> compile/load: create config and graph
  -> runtime: call compiled graph with ordered buffers
  -> convert graph outputs into ModelOutputs
```
