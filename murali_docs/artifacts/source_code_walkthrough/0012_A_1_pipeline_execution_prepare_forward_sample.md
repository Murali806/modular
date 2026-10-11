# Complete Code Walk: `TextGenerationPipeline.execute(...)`

Source:
[`text_generation.py`](../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L399)

## Batch Preparation

```python
@traced
def prepare_batch(
    self,
    batches: list[list[TextGenerationContextType]],
) -> tuple[
    Any,
    npt.NDArray[np.int32] | None,
    list[TextGenerationContextType],
]:
    replica_batches: list[list[TextGenerationContextType]] = [
        self._maybe_sort_loras(batch) for batch in batches
    ]
    flat_batch = flatten2d(replica_batches)

    # Initialize a bitmask for structured output.
    bitmask = self.initialize_bitmask(flat_batch)

    for i, context in enumerate(flat_batch):
        if bitmask is not None:
            self.update_for_structured_output(context, bitmask, i)

    # Retrieve the KV Cache Inputs.
    kv_cache_inputs = self._kv_manager.runtime_inputs(replica_batches)

    if self.batch_info_output_fname is not None:
        self._record_batch_info(flat_batch)

    model_inputs = self._pipeline_model.prepare_initial_token_inputs(
        replica_batches=replica_batches,
        kv_cache_inputs=kv_cache_inputs,
    )

    if self._encoder_cache is not None:
        assert isinstance(self._pipeline_model, SupportsVisionEncoding)
        vision_result = self._encoder_cache.run_vision_encode(
            self._pipeline_model,
            as_vision_context_batches(replica_batches),
            self._devices,
        )
        self._encoder_cache.finalize_vision_inputs(
            self._pipeline_model,
            model_inputs,
            self._devices,
            vision_result,
        )

    return (model_inputs, bitmask, flat_batch)
```

```text
inputs.batches
     |
     +-- optional LoRA sorting per replica
     +-- flatten replicas ------------------> flat_batch
     +-- allocate/update grammar bitmask ---> bitmask | None
     +-- KV manager runtime inputs ---------> kv_cache_inputs
     +-- architecture batch processor ------> model_inputs
     +-- optional vision encoder cache -----> enriched model_inputs
     |
     v
(model_inputs, bitmask, flat_batch)
```

## Public Execute Wrapper

```python
@traced
def execute(
    self,
    inputs: TextGenerationInputs[TextGenerationContextType],
) -> PipelineOutputsDict[TextGenerationOutput]:
    # WALKTHROUGH COMMENT (not in source): validate then execute one step.
    with self._request_validator:
        return self._execute(inputs)
```

## Forward Pass And Sampling

```python
def _execute(
    self,
    inputs: TextGenerationInputs[TextGenerationContextType],
) -> PipelineOutputsDict[TextGenerationOutput]:
    model_inputs, bitmask, flat_batch = self.prepare_batch(inputs.batches)

    batch_processors: list[BatchLogitsProcessor] = []
    if len(flat_batch) > 0:
        sampler: Model
        if bitmask is not None:
            assert self._sampler_with_bitmask is not None, (
                "Sampler must be built with bitmask sampling"
            )
            sampler = self._sampler_with_bitmask
        else:
            assert self._sampler_without_bitmask is not None
            sampler = self._sampler_without_bitmask

        with Tracer("FusedSamplingProcessor"):
            sampling_processor = FusedSamplingProcessor(
                sampler=sampler,
                pipeline_config=self._pipeline_config,
                context_batch=flat_batch,
                device=self._sampler_device,
                pinned_new_tokens=self._pinned_new_tokens,
                identity_logit_offsets=self._identity_logit_offsets,
                bitmask=bitmask,
                vocab_size=self.vocab_size,
            )

        batch_processors.append(sampling_processor)

    curr_step_inputs = model_inputs
    batch_log_probabilities: list[list[LogProbabilities | None]] = []

    model_outputs = self._launch_forward_pass(
        curr_step_inputs, flat_batch, step=0
    )

    if (
        self._pipeline_config.sampling.enable_variable_logits
        and model_outputs.logit_offsets is None
    ):
        raise ValueError(
            "Model must return logit_offsets when enable_variable_logits is True."
        )

    if len(flat_batch) > 0:
        with Tracer("sample_next_token"):
            sample_logits, sample_offsets = (
                sampling_processor.logits_for_sampling(
                    logits=model_outputs.logits,
                    next_token_logits=model_outputs.next_token_logits,
                    logit_offsets=model_outputs.logit_offsets,
                )
            )
            apply_logits_processors(
                context_batch=flat_batch,
                batch_logits=sample_logits,
                batch_logit_offsets=sample_offsets,
                batch_processors=batch_processors,
            )
            new_tokens = sampling_processor.new_tokens
            assert new_tokens is not None

        if inputs.enable_log_probs:
            with Tracer("compute_log_probabilities"):
                try:
                    batch_log_probabilities.append(
                        self._pipeline_model.compute_log_probabilities(
                            self.session,
                            curr_step_inputs,
                            model_outputs,
                            new_tokens,
                            inputs.batch_top_log_probs,
                            inputs.batch_echo,
                        )
                    )
                except NotImplementedError:
                    logger.warning(
                        "Unable to compute log probabilities for"
                        f" {self._pipeline_config.model.model_path}"
                    )
                    batch_log_probabilities.append(
                        [None for _ in flat_batch]
                    )

    if len(flat_batch) == 0:
        return {}
```

```text
model_inputs
    |
    | _pipeline_model.execute(...)
    v
ModelOutputs(logits, next_token_logits, offsets, ...)
    |
    | logits_for_sampling(...)
    v
sample_logits + sample_offsets
    |
    | apply_logits_processors(...)
    |   sampling params + grammar bitmask + request processors
    v
sampling_processor.new_tokens
```

## Device-To-Host Copy And Context Update

```python
sampler_device = self._sampler_device
with Tracer("d2h_generated_tokens"):
    generated_tokens_device = sampling_processor.generated_tokens
    generated_tokens_host = Buffer(
        shape=generated_tokens_device.shape,
        dtype=generated_tokens_device.dtype,
        device=sampler_device,
        usage=Usage.STAGING,
    )
    generated_tokens_host.inplace_copy_from(generated_tokens_device)
    generated_tokens_np = generated_tokens_host.to_numpy()

res = update_context_and_prepare_responses(
    generated_tokens_np,
    flat_batch,
    batch_log_probabilities=batch_log_probabilities,
    enable_log_probs=inputs.enable_log_probs,
)

for ctx in inputs.flat_batch:
    self._kv_manager.step(ctx)

return res
```

```text
generated_tokens on sampler device
          |
          | copy into staging Buffer
          v
generated_tokens_np on host
          |
          | update_context_and_prepare_responses(...)
          +-- append/commit generated token to each context
          +-- update EOS/status/logprob response data
          v
dict[RequestID, TextGenerationOutput]
          |
          | kv_manager.step(ctx)
          v
KV lengths synchronized with updated contexts
```

## Architecture-Specific Forward Call

```python
def _launch_forward_pass(
    self,
    curr_step_inputs: ModelInputs,
    flat_batch: list[TextGenerationContextType],
    step: int,
) -> ModelOutputs:
    with Tracer(f"forward_pass_step_{step}"):
        try:
            model_outputs = self._pipeline_model.execute(
                model_inputs=curr_step_inputs
            )
            return model_outputs
        except Exception:
            batch_size = len(flat_batch)
            cache_tokens = sum(
                ctx.tokens.processed_length for ctx in flat_batch
            )
            input_tokens = sum(
                ctx.tokens.active_length for ctx in flat_batch
            )
            logger.error(
                "Encountered an exception while executing batch: "
                f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}"
            )
            raise
```

## Complete Pipeline Variable Values

```python
# Reduced illustrative vocabulary keeps every logit visible. Production Llama
# uses a much larger vocabulary, but the code path and array relationships are
# identical.
illustrative_vocab = {
    0: "<bos>",
    1: "KV",
    2: " cache",
    3: " stores",
    4: " keys",
    5: " values",
    6: ".",
    7: "<eos>",
}

context = {
    "request_id": "req-chat-001",
    "tokens.prompt": [0, 1, 2],
    "tokens.generated": [],
    "tokens.active": [0, 1, 2],
    "tokens.processed_length": 0,
    "tokens.active_length": 3,
    "json_schema": None,
    "grammar": None,
    "sampling_params": {
        "temperature": 0.2,
        "top_p": 0.9,
    },
}

inputs = {
    "batches": [[context]],
    "flat_batch": [context],
    "enable_log_probs": False,
    "batch_top_log_probs": [0],
    "batch_echo": [False],
}

prepare_batch_output = {
    "model_inputs": {
        "tokens": [0, 1, 2],
        "input_row_offsets": [0, 3],
        "return_n_logits": [1],
        "kv_cache_inputs": {
            "blocks": [0],
            "cache_lengths": [0],
        },
    },
    "bitmask": None,
    "flat_batch": [context],
}

model_outputs = {
    "logits": [
        [-8.0, -4.0, 1.5, 3.2, 0.7, 0.5, -1.0, -3.0]
    ],
    "next_token_logits": [
        [-8.0, -4.0, 1.5, 3.2, 0.7, 0.5, -1.0, -3.0]
    ],
    "logit_offsets": None,
}

sampling_values = {
    "sample_logits": [
        [-8.0, -4.0, 1.5, 3.2, 0.7, 0.5, -1.0, -3.0]
    ],
    "sample_offsets": None,
    "generated_tokens_device": [3],
    "generated_tokens_host": [3],
    "decoded_generated_token": " stores",
}

response_output = {
    "req-chat-001": {
        "tokens": [3],
        "decoded_tokens": " stores",
        "status": "ACTIVE",
        "is_done": False,
        "log_probabilities": None,
    }
}

context_after_update = {
    "request_id": "req-chat-001",
    "tokens.prompt": [0, 1, 2],
    "tokens.generated": [3],
    "tokens.processed_length": 3,
    "tokens.active": [3],
    "status": "ACTIVE",
}
```
