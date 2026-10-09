# Source Code Walkthrough

This walkthrough maps the Phase 01-10 notes to the main MAX source paths. Read
the code by request lifecycle and ownership boundary, not by repository folder.

## 1. Start With The Big Folders

- [`max/python/max/serve/`](../../../max/python/max/serve/): HTTP server, request routing, workers, and scheduler.
- [`max/python/max/pipelines/`](../../../max/python/max/pipelines/): tokenizer, contexts, model pipeline, and Llama architecture.
- [`max/python/max/engine/`](../../../max/python/max/engine/): graph compile/init/load and MEF boundary.
- [`max/python/max/nn/`](../../../max/python/max/nn/): Python graph ops such as attention and linear.
- [`max/kernels/src/`](../../../max/kernels/src/): Mojo kernels and graph-compiler custom op registrations.
- [`max/mojo/max/gpu/`](../../../max/mojo/max/gpu/): lower-level GPU and device APIs.

## 2. Follow One Request Top-Down

Use this path as the main traversal for a single OpenAI-compatible text request:

<details>
<summary><strong>1. HTTP/OpenAI Ingress</strong> - <a href="../../../max/python/max/serve/router/openai_routes.py#L2217"><code>openai_routes.py</code></a></summary>

**Source Role**

This is the HTTP boundary for `/v1/chat/completions`. The route receives the
FastAPI `Request`, gets the request ID from middleware state, parses the JSON
body into `CreateChatCompletionRequest`, and selects the serving pipeline for
the requested model.

At this level the code is intentionally model-agnostic. It validates and
normalizes OpenAI-compatible request fields, parses chat messages and media,
looks up tokenizer/parser capabilities, and prepares the request for the
pipeline layer.

**Code Landmarks**

- `_parse_openai_request_body(...)`: converts raw HTTP body into a typed
  schema. <strong><em><a href="0001_parse_openai_request_body.md"><span style="color:#0b63ce">See focused note: 0001_parse_openai_request_body.md</span></a></em></strong>.
- `get_pipeline(request, completion_request.model)`: chooses the pipeline for
  the requested model name. <strong><em><a href="0002_get_pipeline.md"><span style="color:#0b63ce">See focused note: 0002_get_pipeline.md</span></a></em></strong>.
- `openai_parse_chat_completion_request(...)`: normalizes messages, images,
  videos, roles, and content wrapping. <strong><em><a href="0003_openai_parse_chat_completion_request.md"><span style="color:#0b63ce">See focused note: 0003_openai_parse_chat_completion_request.md</span></a></em></strong>.
- `TextGenerationRequest(...)`: converts the OpenAI request into MAX's internal
  request object. <strong><em><a href="0004_TextGenerationRequest.md"><span style="color:#0b63ce">See focused note: 0004_TextGenerationRequest.md</span></a></em></strong>.
- Streaming versus non-streaming response branches: chooses the HTTP response
  shape before tokens are produced. <strong><em><a href="0005_streaming_vs_non_streaming.md"><span style="color:#0b63ce">See focused note: 0005_streaming_vs_non_streaming.md</span></a></em></strong>.

**Indented Flow**

```text
Client HTTP JSON
  -> FastAPI route: openai_create_chat_completion(...)
  -> CreateChatCompletionRequest
  -> parsed messages/images/videos/tools/response_format
  -> TextGenerationRequest
  -> LLM pipeline
```

```text
openai_create_chat_completion(request)
  |
  +-- request.state.request_id
  +-- _parse_openai_request_body(..., CreateChatCompletionRequest)
  +-- get_pipeline(..., completion_request.model)
  +-- openai_parse_chat_completion_request(...)
  +-- _convert_chat_completion_tools_to_token_generator_tools(...)
  +-- _create_response_format(...)
  +-- TextGenerationRequest(...)
  +-- token_generator.next_token_chunk(...)
```

The route is mostly a translator. It turns OpenAI-shaped API data into MAX's
internal text-generation request while preserving API-level behavior such as
streaming, tools, structured output, logprobs, stop controls, and media inputs.

**Hands Off To**

After validation, the route constructs a text-generation request and calls into
the LLM pipeline. From here onward, the request is no longer just HTTP JSON; it
is a typed serving request that can be tokenized and scheduled.

</details>

<details>
<summary><strong>2. LLM Pipeline Handoff</strong> - <a href="../../../max/python/max/serve/pipelines/llm.py#L294"><code>llm.py</code></a></summary>

**Source Role**

This file bridges API-level text-generation requests to the model worker. The
important method is `next_token_chunk(...)`: it tokenizes the request, creates a
`TextContext`, submits that context to the worker, and returns an async stream
of generated token chunks.

This is also where request-level output handling starts. It tracks time to first
token, decode time, reasoning-parser state, stop sequences, buffered
detokenization, and admission metrics.

**Code Landmarks**

- `self.tokenizer.new_context(request)`: creates the scheduler context from the
  user request. <strong><em><a href="0006_tokenizer_new_context.md"><span style="color:#0b63ce">See focused note: 0006_tokenizer_new_context.md</span></a></em></strong>.
- `create_buffered_detokenizer(...)`: prevents broken UTF-8 output when tokens
  split multi-byte characters. <strong><em><a href="0007_create_buffered_detokenizer.md"><span style="color:#0b63ce">See focused note: 0007_create_buffered_detokenizer.md</span></a></em></strong>.
- `self.model_worker.note_awaiting_admission(...)`: marks API-side backlog
  before and after handoff. <strong><em><a href="0008_note_awaiting_admission.md"><span style="color:#0b63ce">See focused note: 0008_note_awaiting_admission.md</span></a></em></strong>.
- `self.model_worker.stream(context.request_id, context)`: submits the request
  to the worker and receives the response stream.

**Indented Flow**

```text
TextGenerationRequest
  -> tokenizer.new_context(...)
  -> TextContext
  -> model_worker.stream(...)
  -> async stream of scheduler outputs
  -> decoded TokenGeneratorOutput chunks
```

```text
next_token_chunk(request)
  |
  +-- start TTFT / decode timers
  +-- maybe create reasoning parser
  +-- note_awaiting_admission(+1)
  +-- tokenizer.new_context(request)
  +-- create content/reasoning buffered detokenizers
  +-- await model_worker.stream(request_id, context)
  +-- for each worker response:
        +-- detokenize
        +-- apply reasoning parsing / stop handling
        +-- yield TokenGeneratorOutput
```

This file is the API-side generator. It hides worker response details from the
OpenAI route and emits chunks in the form the route can serialize as JSON or
server-sent events.

**Hands Off To**

The handoff to `model_worker.stream(...)` crosses from API/pipeline code into
worker IPC. If this handoff fails, the route can still return an HTTP error
before streaming headers are committed.

</details>

<details>
<summary><strong>3. Worker IPC Boundary</strong> - <a href="../../../max/python/max/serve/worker_interface/zmq_interface.py#L105"><code>zmq_interface.py</code></a></summary>

**Source Role**

This file owns the API-process to model-worker communication path. The
`stream(...)` method registers a per-request output queue, performs the
admission check, pushes the context onto the worker request queue, and exposes
an async generator for scheduler responses.

The key behavior is backpressure. The request queue is bounded. If it is not
writable at admission time, the request is rejected immediately instead of
blocking after partial response state has been created.

**Code Landmarks**

- `wait_until_connected(...)`: startup-time handshake so runtime failures mean
  queue pressure, not a still-connecting worker.
- `pending_out_queues`: per-request response queues in the API process.
- `_admission_lock`: serializes the writability check and push.
- `RequestQueueFull`: the failure that maps to load shedding.

**Indented Flow**

```text
API process
  pending_out_queues[request_id] <----- worker responses
       |
       v
  request_queue.put(TextContext) -----> model worker process
```

```text
stream(req_id, context)
  |
  +-- duplicate req_id? -> RuntimeError
  |
  +-- acquire _admission_lock
        |
        +-- request_queue.writable()?
              |
              +-- no  -> RequestQueueFull -> HTTP load shedding
              |
              +-- yes -> create out_queue
                       -> pending_out_queues[req_id] = out_queue
                       -> request_queue.put(context)
                       -> return _drain_responses(...)
```

The important detail is ordering: the output queue is registered only when the
request can actually be pushed to the worker. If the push fails, registration is
rolled back so there is no orphaned per-request state.

**Hands Off To**

Once the context reaches the worker side, the scheduler drains pending requests
and decides when each request participates in a prefill or decode batch.

</details>

<details>
<summary><strong>4. Scheduler Iteration</strong> - <a href="../../../max/python/max/serve/scheduler/text_generation_scheduler.py#L205"><code>text_generation_scheduler.py</code></a></summary>

**Source Role**

This file runs the continuous batching loop. `run_iteration(...)` drains new
requests, asks the batch constructor to form the next executable batch, calls
the pipeline, and pushes results back to the response queue.

Read this file as the control loop for token generation. It does not implement
the model math. It decides which contexts run in the next step and accounts for
progress, failures, telemetry, and empty-batch behavior.

**Code Landmarks**

- `_retrieve_pending_requests()`: pulls newly admitted contexts from the worker
  queue.
- `batch_constructor.construct_batch()`: converts queued contexts into model
  work.
- `_schedule(inputs)`: executes the selected batch through the pipeline.
- `SchedulerProgress`: tells the serving loop whether useful work happened.

**Indented Flow**

```text
worker request queue
  -> _retrieve_pending_requests()
  -> batch_constructor.construct_batch()
  -> _schedule(inputs)
  -> response_queue.put_nowait(...)
```

```text
run_iteration()
  |
  +-- drain new requests into batch constructor
  +-- construct next batch
  +-- fail requests whose grammar build failed
  +-- if no inputs and no pending overlap work:
  |     return NO_PROGRESS
  +-- _schedule(inputs)
  +-- log scheduler / KV / batch metrics
  +-- process cancellations
  +-- return MADE_PROGRESS
```

The scheduler loop is intentionally repetitive. Every pass either makes a small
piece of progress for some set of requests or reports that no useful work was
available.

**Hands Off To**

The scheduler delegates packing policy to the batch constructor. That is where
request queues, token budgets, chunked prefill, decode slots, and replica
placement become concrete batch inputs.

</details>

<details>
<summary><strong>5. Batch Construction</strong> - <a href="../../../max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py#L677"><code>text_batch_constructor.py</code></a></summary>

**Source Role**

This file owns admission into continuous batching. `enqueue_new_request(...)`
adds a `TextContext` to the right pending area, handles grammar readiness, and
then binds the request to a replica or a data-parallel pending pool.

The batch constructor is where serving policy becomes a concrete workload. It
balances prefill and decode work, tracks budgets, chooses candidate requests,
and produces the `TextGenerationInputs` consumed by the pipeline.

**Code Landmarks**

- `submit_grammar_build(...)`: starts constrained-decoding grammar work early
  when needed.
- `enqueue_new_request(...)`: first admission point for a new `TextContext`.
- `_admit_request(...)`: decides whether to place the request immediately or
  defer it for data-parallel balancing.
- `_bind_request(...)`: attaches a request to a replica queue.
- `construct_batch()`: the main method to follow after request admission.

**Indented Flow**

```text
TextContext
  -> grammar pending?      -> wait until matcher is ready
  -> DP CE pending pool?   -> later bind to best replica
  -> replica.ce_reqs       -> prompt/prefill work
  -> replica.tg_reqs       -> decode work
  -> TextGenerationInputs
```

```text
enqueue_new_request(ctx)
  |
  +-- if grammar required and not ready:
  |     _grammar_pending[request_id] = ctx
  |
  +-- else _admit_request(ctx)
          |
          +-- DP CE balancing enabled and fresh prefill?
          |     _ce_pending[request_id] = weighted request
          |
          +-- otherwise:
                replica_idx = caller supplied or round-robin
                _bind_request(ctx, replica_idx)
                  |
                  +-- generated_length == 0 -> replica.ce_reqs
                  +-- generated_length > 0  -> replica.tg_reqs
```

The file separates "can this request enter the scheduler?" from "which batch
should it run in?" Grammar readiness, data-parallel balancing, prefill chunks,
decode requests, and KV-cache pressure all influence the final batch.

**Hands Off To**

After batch construction, the scheduler has a batch of contexts. The pipeline
then turns those contexts into model buffers, runs the graph, samples tokens,
and returns per-request outputs.

</details>

<details>
<summary><strong>6. Pipeline Execution</strong> - <a href="../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L515"><code>text_generation.py</code></a></summary>

**Source Role**

This file is the high-level token-generation execution path. `execute(...)`
accepts `TextGenerationInputs`, prepares a model batch, launches the model
forward pass, runs the sampler, and converts raw model outputs into
`TextGenerationOutput` objects.

This is where model execution and sampling meet. The model produces logits; the
sampling processor applies sampling rules, grammar bitmasks, logit processors,
and token selection.

**Code Landmarks**

- `prepare_batch(inputs.batches)`: creates model inputs and the flat context
  batch.
- `FusedSamplingProcessor(...)`: prepares the sampling path for the current
  batch.
- `_launch_forward_pass(...)`: calls the architecture-specific model.
- `logits_for_sampling(...)`: chooses which logits should be sampled.
- `apply_logits_processors(...)`: applies request-specific sampling controls.

**Indented Flow**

```text
TextGenerationInputs
  -> prepare_batch(...)
  -> model_inputs + flat_batch + optional grammar bitmask
  -> _pipeline_model.execute(...)
  -> logits
  -> sampler/logits processors
  -> generated token IDs
  -> TextGenerationOutput per request
```

```text
execute(inputs)
  |
  +-- _execute(inputs)
        |
        +-- prepare_batch(inputs.batches)
        +-- build FusedSamplingProcessor if flat_batch is non-empty
        +-- _launch_forward_pass(model_inputs, flat_batch)
        +-- choose logits_for_sampling(...)
        +-- apply_logits_processors(...)
        +-- sampler produces generated_tokens
        +-- copy generated tokens device -> host
        +-- update_context_and_prepare_responses(...)
        +-- kv_manager.step(ctx)
```

The forward pass and sampler are distinct stages. The model decides token
probabilities; the sampler decides the actual next token under request-specific
sampling controls.

**Hands Off To**

`prepare_batch(...)` delegates architecture-specific input construction to the
Llama batch processor. That is where token IDs, row offsets, return-logit
buffers, KV-cache inputs, and signal buffers are assembled.

</details>

<details>
<summary><strong>7. Llama Input Staging</strong> - <a href="../../../max/python/max/pipelines/architectures/llama3/batch_processor.py#L163"><code>batch_processor.py</code></a></summary>

**Source Role**

This file converts scheduled `TextContext` objects into the exact `Llama3Inputs`
object expected by the compiled Llama graph. `Llama3BatchProcessor._make_inputs`
packages token buffers, row offsets, return-logit controls, signal buffers,
KV-cache inputs, and optional data-parallel split metadata.

The batch processor is a shape and buffer adapter. It does not perform the
transformer computation. Its job is to make the runtime batch match the graph
signature that was compiled earlier.

**Code Landmarks**

- `Llama3BatchProcessor`: the normal Llama text-generation batch processor.
- `_make_inputs(...)`: constructs `Llama3Inputs`.
- `process_outputs(...)`: converts raw graph outputs into `ModelOutputs`.
- EP/data-parallel subclasses: show how extra communication buffers are added
  for more complex deployments.

**Indented Flow**

```text
replica batches of TextContext
  -> flatten contexts
  -> stage token IDs on device
  -> build ragged row offsets
  -> build return_n_logits
  -> attach KV-cache inputs
  -> attach signal / DP / EP buffers when needed
  -> Llama3Inputs
```

```text
Prompt A tokens: [a0 a1 a2]
Prompt B tokens: [b0 b1]

tokens:
  [a0 a1 a2 b0 b1]

input_row_offsets:
  [0, 3, 5]

Meaning:
  row 0 reads tokens[0:3]
  row 1 reads tokens[3:5]
```

Llama uses ragged batching because requests have different active lengths. The
row-offset buffer tells the graph where each request's token span begins and
ends inside the packed token buffer.

**Hands Off To**

The result of input staging is a `Llama3Inputs` object. The Llama model wrapper
passes its buffers to the compiled executable model.

</details>

<details>
<summary><strong>8. Llama Model Execution</strong> - <a href="../../../max/python/max/pipelines/architectures/llama3/model.py#L183"><code>model.py</code></a></summary>

**Source Role**

This file owns the Llama pipeline model wrapper. The `execute(...)` method
asserts that it received `Llama3Inputs`, requires KV-cache inputs, calls the
compiled model with `model_inputs.buffers`, and asks the batch processor to
translate raw outputs back into `ModelOutputs`.

The same file also contains the graph-build path. During load/compile, it
creates the `Llama3Config`, finalizes it from Hugging Face config plus adapted
weights, and builds the single-device, tensor-parallel, or data-parallel graph.

**Code Landmarks**

- `Llama3Inputs.buffers`: defines the exact positional argument order passed to
  the compiled model.
- `execute(...)`: runtime hot path into the compiled model.
- `_create_model_config(...)`: initializes and finalizes `Llama3Config`.
- `_build_graph_for_compile(...)`: chooses single-device, tensor-parallel, or
  data-parallel graph construction.
- `Llama3Model.__init__(...)`: wires pipeline config, session, devices,
  KV-cache config, weights, adapter, and memory plan into the shared base class.

**Indented Flow**

```text
Runtime path:
  Llama3Inputs
    -> model_inputs.buffers
    -> self.model.execute(...)
    -> raw graph outputs
    -> batch_processor.process_outputs(...)
    -> ModelOutputs

Compile/load path:
  pipeline config + HF config + weights
    -> Llama3Config
    -> choose graph topology
    -> Graph(...)
    -> compiled executable self.model
```

```text
Llama3Inputs
  |
  +-- tokens
  +-- input_row_offsets
  +-- return_n_logits
  +-- signal_buffers or data_parallel_splits
  +-- flattened KV-cache inputs
  |
  v
self.model.execute(*model_inputs.buffers)
  |
  v
compiled MAX graph
  |
  v
raw output buffers
  |
  v
batch_processor.process_outputs(...)
  |
  v
ModelOutputs(logits, next_token_logits, logit_offsets, ...)
```

At runtime this file is deliberately thin. The heavy transformer computation is
already compiled; `execute(...)` mainly checks the input type, passes buffers in
the graph's expected order, and converts the returned buffers into the common
pipeline output structure.

```text
_build_graph_for_compile(...)
  |
  +-- data_parallel_degree > 1?
  |     -> create_data_parallel_graph(...)
  |
  +-- multiple devices?
  |     -> _build_tensor_parallel_graph_for_compile(...)
  |          -> DistributedLlama3(model_config)
  |          -> Graph(..., input_types=dist_model.input_types(...))
  |
  +-- otherwise
        -> _build_single_device_graph_for_compile(...)
             -> Llama3(model_config)
             -> Graph("llama3", input_types=single_model.input_types(...))
```

The same Python file therefore answers two different questions:

- Runtime: "How do I call the already compiled executable for this batch?"
- Compile time: "Which graph should be built for this model and device layout?"

**Hands Off To**

From here there are two directions:

- Runtime direction: compiled `self.model.execute(...)` returns logits and
  sampler inputs to the pipeline.
- Compile direction: graph construction enters `llama3.py`, then
  `engine/api.py`, then the Python graph ops and Mojo kernels described in
  later sections.

</details>

The mental model is:

```text
HTTP JSON
  -> OpenAI request schema
  -> tokenizer/context
  -> worker queue
  -> scheduler request
  -> scheduler batch
  -> model inputs
  -> graph execution
  -> sampled token
  -> response stream
```

## 3. Read Startup And Model Resolution

Use this path for understanding how `max serve` turns a model ID into runnable
pipeline components:

1. Registry lookup:
   [`registry.py`](../../../max/python/max/pipelines/lib/registry.py#L335)

2. Architecture lookup records:
   [`arch_lookup.py`](../../../max/python/max/pipelines/lib/arch_lookup.py)

3. Llama registration:
   [`arch.py`](../../../max/python/max/pipelines/architectures/llama3/arch.py#L28)

4. Llama config and finalization:
   [`model_config.py`](../../../max/python/max/pipelines/architectures/llama3/model_config.py#L108)

5. Weight-name conversion:
   [`weight_adapters.py`](../../../max/python/max/pipelines/architectures/llama3/weight_adapters.py)

6. Graph model selection:
   [`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L306)

7. Shared graph load template:
   [`pipeline_model.py`](../../../max/python/max/pipelines/lib/interfaces/pipeline_model.py#L936)

The mental model is:

```text
HF config.json
  -> architectures[0]
  -> PipelineRegistry
  -> llama_arch
  -> Llama3Config
  -> weight adapter
  -> tokenizer + model class + memory planner
  -> pipeline factory
```

## 4. Cross The Compiler Boundary

This is where Python graph construction becomes a compiled executable model:

1. Llama graph body:
   [`llama3.py`](../../../max/python/max/pipelines/architectures/llama3/llama3.py#L75)

2. Engine load:
   [`api.py`](../../../max/python/max/engine/api.py#L768)

3. MEF reuse path:
   [`api.py`](../../../max/python/max/engine/api.py#L883)

4. Model initialization:
   [`api.py`](../../../max/python/max/engine/api.py#L1146)

5. Native public contract:
   [`engine.pyi`](../../../max/python/max/_core/engine.pyi)

The mental model is:

```text
Llama3 Python module
  -> Graph(input_types)
  -> symbolic forward
  -> graph.output
  -> compile or import MEF
  -> init weights/resources
  -> executable Model
```

## 5. Follow Python Ops Into Mojo

For attention and KV-cache execution:

1. Python attention op:
   [`attention_with_rope.py`](../../../max/python/max/nn/attention/attention_with_rope.py#L53)

2. Python custom op bridge:
   [`kernels.py`](../../../max/python/max/nn/kernels.py)

3. Mojo graph-compiler registration:
   [`attention.mojo`](../../../max/kernels/src/graph_compiler/builtin_kernels/attention.mojo)

4. Mojo argument adapter:
   [`kernels.mojo`](../../../max/kernels/src/graph_compiler/builtin_kernels/kernels.mojo)

5. Ragged KV attention and matmul dispatch:
   [`kv_cache_ragged.mojo`](../../../max/kernels/src/nn/kv_cache_ragged.mojo#L103)

6. GPU MHA selection:
   [`mha.mojo`](../../../max/kernels/src/nn/attention/gpu/mha.mojo)

7. Linear graph op:
   [`linear.py`](../../../max/python/max/nn/linear.py)

8. Matmul kernels:
   [`matmul`](../../../max/kernels/src/linalg/matmul)

9. Device APIs:
   [`max.gpu`](../../../max/mojo/max/gpu)

The mental model is:

```text
Python graph op
  -> custom op symbol
  -> Mojo graph-compiler registration
  -> argument adaptation
  -> CPU/GPU dispatch
  -> selected kernel implementation
  -> hardware launch
```

## 6. Useful Search Commands

```bash
rg -n "class TextGenerationScheduler|def run_iteration" max/python/max/serve
rg -n "class TextGenerationPipeline|def execute" max/python/max/pipelines
rg -n "class Llama3Model|class Llama3|def execute" max/python/max/pipelines/architectures/llama3
rg -n "retrieve_factory|SupportedArchitecture|llama_arch" max/python/max/pipelines
rg -n "compile_reusing_mefs|init_all|def load" max/python/max/engine/api.py
rg -n "AttentionWithRope|custom|ragged|paged" max/python/max/nn max/kernels/src
```

## 7. Best Reading Order

1. Read Phase 01-04 source paths first to understand request ingress,
   validation, IPC, and response routing.
2. Read Phase 05-06 paths next to understand continuous batching and one
   scheduler iteration.
3. Read Phase 07 paths after that to understand how the selected model,
   tokenizer, config, weights, and memory planner are resolved.
4. Read Phase 08-09 paths to understand graph construction, compile/init, MEF
   reuse, and the Graph API versus ModuleV3 split.
5. Read Phase 10 paths last to understand how Python graph ops cross into
   Mojo kernels and hardware dispatch.

The shortest complete traversal is:

```text
openai_routes.py
  -> llm.py
  -> zmq_interface.py
  -> text_generation_scheduler.py
  -> text_batch_constructor.py
  -> text_generation.py
  -> llama3/batch_processor.py
  -> llama3/model.py
  -> llama3/llama3.py
  -> engine/api.py
  -> max/nn attention or linear
  -> max/kernels Mojo registrations
  -> selected Mojo kernel
```
