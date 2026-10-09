# Pipeline Execution - `TextGenerationPipeline.execute(...)`

Sources:

- `prepare_batch(...)` in
  [`text_generation.py`](../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L399)
- `execute(...)` in
  [`text_generation.py`](../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L515)
- `_execute(...)` in
  [`text_generation.py`](../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L527)
- `_launch_forward_pass(...)` in
  [`text_generation.py`](../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L655)

## One-Line Purpose

`execute(inputs)` runs one generation step for the scheduled batch: prepare
model inputs, run the model forward pass, sample next tokens, update contexts,
and return `TextGenerationOutput` objects.

## Big Picture

```text
TextGenerationInputs
  |
  v
prepare_batch(...)
  |
  v
model_inputs + flat_batch + optional grammar bitmask
  |
  v
_launch_forward_pass(...)
  |
  v
ModelOutputs(logits / next_token_logits / offsets)
  |
  v
FusedSamplingProcessor + logits processors
  |
  v
new token IDs
  |
  v
update_context_and_prepare_responses(...)
  |
  v
TextGenerationOutput per request
```

The model produces logits. The sampler chooses the actual next token.

## Inputs And Output

| Side | Name | Meaning |
| --- | --- | --- |
| Input | `TextGenerationInputs` | Per-replica batches of `TextContext` objects. |
| Prepared | `model_inputs` | Architecture-specific model buffers. For Llama this becomes `Llama3Inputs`. |
| Prepared | `flat_batch` | All contexts flattened across replicas. |
| Prepared | `bitmask` | Optional structured-output mask for constrained decoding. |
| Model output | `ModelOutputs` | Raw logits and related output buffers from the model. |
| Final output | `PipelineOutputsDict[TextGenerationOutput]` | Per-request generated token results. |

## Worked Example

```text
INPUT                                  ACTION                                  OUTPUT
-----------------------------------    -----------------------------------     -----------------------------------
replica batches                        prepare_batch(...)                      flat context batch
  [[ctx-A], [ctx-B]]                 ------------------------------->          [ctx-A, ctx-B]

flat context batch                     initialize structured bitmask           optional bitmask
  ctx-B has JSON schema              ------------------------------->          bitmask for ctx-B

replica batches                        prepare_initial_token_inputs(...)       model inputs
  Llama text batch                   ------------------------------->          Llama3Inputs(...)

model inputs                           _launch_forward_pass(...)               model outputs
  Llama3Inputs(...)                  ------------------------------->          logits / next_token_logits

model outputs                          sample_next_token                       generated token IDs
  logits + sampling params           ------------------------------->          [token-A, token-B]

generated token IDs                    update contexts/responses               response dict
  [token-A, token-B]                 ------------------------------->          {req-A: TextGenerationOutput(...)}
```

## Execution Flow

```text
+------------------------------------------------------------------------------------------------------+
| execute(inputs)                                                                                      |
|------------------------------------------------------------------------------------------------------|
| START                                                                                                |
|   |                                                                                                  |
|   v                                                                                                  |
| request validator                                                                                    |
|   |                                                                                                  |
|   v                                                                                                  |
| _execute(inputs)                                                                                     |
|   |                                                                                                  |
|   +-- prepare_batch(inputs.batches)                                                                  |
|   |     -> model_inputs, bitmask, flat_batch                                                         |
|   |                                                                                                  |
|   +-- if flat_batch is non-empty                                                                     |
|   |     -> choose sampler with or without bitmask                                                     |
|   |     -> build FusedSamplingProcessor                                                              |
|   |                                                                                                  |
|   +-- _launch_forward_pass(model_inputs, flat_batch)                                                 |
|   |     -> _pipeline_model.execute(model_inputs)                                                      |
|   |                                                                                                  |
|   +-- validate variable-logit output shape                                                           |
|   |                                                                                                  |
|   +-- if flat_batch is empty                                                                         |
|   |     -> return {}                                                                                  |
|   |                                                                                                  |
|   +-- logits_for_sampling(...)                                                                       |
|   |                                                                                                  |
|   +-- apply_logits_processors(...)                                                                   |
|   |                                                                                                  |
|   +-- copy generated tokens device -> host                                                           |
|   |                                                                                                  |
|   +-- update_context_and_prepare_responses(...)                                                      |
|   |                                                                                                  |
|   +-- kv_manager.step(ctx) for each context                                                          |
|------------------------------------------------------------------------------------------------------|
| END EXECUTE                                                                                          |
+------------------------------------------------------------------------------------------------------+
```

## Landmark Map

```text
prepare_batch(inputs.batches)
  |
  +-- flatten replica batches
  +-- initialize structured-output bitmask, if needed
  +-- update grammar state
  +-- build KV-cache runtime inputs
  +-- call architecture-specific prepare_initial_token_inputs(...)

FusedSamplingProcessor(...)
  |
  +-- knows request sampling params
  +-- owns generated token buffers
  +-- can apply grammar bitmask

_launch_forward_pass(...)
  |
  +-- calls architecture-specific model execute(...)
  +-- logs batch dimensions on exception

logits_for_sampling(...)
  |
  +-- selects the logits rows that correspond to next-token sampling

apply_logits_processors(...)
  |
  +-- applies request-specific controls before token selection
```

## Q&A

<details>
<summary><strong>Q1. Why is sampling separate from the model forward pass?</strong></summary>

The model computes token scores. The sampler chooses one token using request
rules.

```text
model forward
  -> logits: scores for possible next tokens

sampler
  -> temperature / top_p / grammar / logit processors
  -> selected next token ID
```

This lets different requests in the same batch use different sampling behavior.

</details>

<details>
<summary><strong>Q2. Why copy generated tokens from device to host?</strong></summary>

The sampler output may live on the GPU or sampler device. The scheduler and API
response path need normal host-visible token IDs.

```text
generated_tokens device buffer
  |
  v
staging Buffer
  |
  v
numpy token IDs on host
  |
  v
TextGenerationOutput
```

</details>

## Where It Goes Next

For Llama, `prepare_batch(...)` calls the Llama batch processor to construct
`Llama3Inputs`:

```text
prepare_initial_token_inputs(...)
  |
  v
Llama3Inputs
  |
  v
Llama model execute(...)
```

## Short Version

```text
execute(inputs)
  -> prepare model buffers
  -> run model forward pass
  -> sample next token
  -> update contexts
  -> return per-request TextGenerationOutput
```
