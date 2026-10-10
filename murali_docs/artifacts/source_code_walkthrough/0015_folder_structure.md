# MAX Serve Folder Structure

## One-Line Purpose

This note explains where the MAX Serve request-flow code lives, where the
supporting runtime code lives, and where the walkthrough notes are stored.

The tree is intentionally focused on the path followed by one
OpenAI-compatible text request. It is not a complete listing of every file in
the repository.

## The Two Folder Trees

There are two different structures to keep separate:

```text
Repository implementation tree
  max/
    python/max/serve/       HTTP, workers, scheduler, serving policy
    python/max/pipelines/   tokenization, contexts, model pipelines, architectures
    python/max/engine/      graph/runtime integration
    python/max/nn/          Python neural-network building blocks
    kernels/src/            lower-level Mojo kernels and graph compiler support
    mojo/max/gpu/           lower-level GPU and device APIs

Walkthrough documentation tree
  murali_docs/artifacts/source_code_walkthrough/
    0000_...md              main reading path and source-file index
    0001_...md - 0014_...md focused explanations of individual steps
    0015_...md              this folder-structure guide
    AGENTS.md               rules for maintaining this documentation folder
```

The first tree contains executable implementation code. The second tree
contains explanations of that implementation. A note in the second tree
links back to source code in the first tree; it does not replace the source
file.

## Repository-Level Tree

```text
modular/
|
+-- max/
|   |
|   +-- python/
|   |   |
|   |   +-- max/
|   |       |
|   |       +-- serve/
|   |       +-- pipelines/
|   |       +-- engine/
|   |       +-- nn/
|   |       +-- kv_cache/
|   |       +-- driver/
|   |       +-- graph/
|   |
|   +-- kernels/
|   |   |
|   |   +-- src/
|   |       +-- pipeline/
|   |       +-- kv_cache/
|   |       +-- nn/
|   |       +-- graph_compiler/
|   |       +-- quantization/
|   |       +-- comm/
|   |       +-- linalg/
|   |
|   +-- mojo/
|       |
|       +-- max/
|           +-- gpu/
|           +-- runtime/
|           +-- algorithm/
|           +-- benchmark/
|
+-- murali_docs/
    |
    +-- artifacts/
        |
        +-- source_code_walkthrough/
```

### `max/`

`max/` is the implementation area for the MAX platform. It contains both
high-level Python orchestration and lower-level compiled/runtime components.

```text
max/
|
+-- python/max/
|     Python APIs and orchestration
|
+-- kernels/
|     performance-critical kernel and graph-compiler sources
|
+-- mojo/max/
      lower-level Mojo APIs, including GPU and runtime support
```

For the request walkthrough, the Python tree is the primary entry point. The
kernel and Mojo trees explain what happens below the Python model wrapper when
the graph executes.

### `max/python/max/serve/`

This folder owns the serving system around the model. It receives HTTP
requests, converts them into internal request objects, queues them, schedules
them, and sends generated results back to the client.

```text
max/python/max/serve/
|
+-- router/
|     HTTP routes and OpenAI-compatible request/response behavior
|
+-- pipelines/
|     serving-level pipeline handoff, including LLM request streaming
|
+-- worker_interface/
|     API-process to model-worker communication
|
+-- scheduler/
|   |
|   +-- text_generation_scheduler.py
|   |     continuous-batching control loop
|   |
|   +-- batch_constructor/
|         queues requests and constructs executable batches
|
+-- schemas/
|     typed serving request/response schemas and related data models
|
+-- parser/
|     parsing helpers, including output/reasoning-related behavior
|
+-- telemetry/
|     metrics, tracing, and serving measurements
|
+-- dependencies/
|     serving dependency integration
|
+-- recordreplay/
|     request/response recording and replay support
|
+-- media/
|     media-related serving support
|
+-- mocks/
      test and development doubles
```

The main request path crosses these serving subfolders in this order:

```text
router
  -> pipelines
  -> worker_interface
  -> scheduler
  -> scheduler/batch_constructor
```

The route does not directly call the model graph. It hands a typed request to
the LLM pipeline; the pipeline crosses the worker boundary; the worker's
scheduler decides when the request runs.

### `router/`

The router is the HTTP/OpenAI ingress boundary.

```text
max/python/max/serve/router/
|
+-- openai_routes.py
      /v1/chat/completions route
      request-body validation
      model-pipeline lookup
      OpenAI-to-MAX request conversion
      streaming/non-streaming response selection
```

The router understands HTTP concepts such as `Request`, JSON bodies, headers,
request IDs, and HTTP response types. It should not need to know the internal
details of a Llama transformer layer.

### `pipelines/`

The serving pipeline is the bridge between a normalized API request and the
worker that executes generation.

```text
max/python/max/serve/pipelines/
|
+-- llm.py
      create TextContext through the tokenizer
      create buffered detokenizers
      submit work to the model worker
      consume worker outputs
      yield token-generation chunks
```

This is where API-side lifecycle work and worker submission meet. It is still
serving orchestration, not the transformer implementation itself.

### `worker_interface/`

The worker interface owns the process boundary between API code and model
execution code.

```text
max/python/max/serve/worker_interface/
|
+-- zmq_interface.py
      bounded request queue
      per-request output queues
      admission lock
      worker connection handshake
      async response draining
```

The important boundary is:

```text
API process                                      model-worker process
-----------                                      --------------------
TextContext  -- bounded request queue ------->   scheduler
response queue <---- worker responses --------  pipeline/model
```

This folder is where backpressure and load shedding become visible. A full
request queue can reject a request before model execution starts.

### `scheduler/`

The scheduler controls continuous batching. It decides which admitted
contexts participate in the next generation iteration.

```text
max/python/max/serve/scheduler/
|
+-- text_generation_scheduler.py
|     retrieve pending requests
|     ask for a batch
|     execute the selected batch
|     publish responses and progress
|
+-- batch_constructor/
      grammar readiness
      prefill/decode queues
      replica binding
      token and KV-cache budgets
      TextGenerationInputs construction
```

The scheduler is a control layer. It does not implement attention or compute
the next-token logits. It selects work and delegates execution downstream.

### `max/python/max/pipelines/`

This folder contains model-pipeline code. It is broader than serving: the
same pipeline concepts can support generation, embeddings, vision, and other
model tasks.

```text
max/python/max/pipelines/
|
+-- lib/
|     shared pipeline abstractions and pipeline variants
|
+-- architectures/
|     model-specific implementations
|     example: architectures/llama3/
|
+-- architectures/llama3/
      batch_processor.py
      model.py
      tokenizer/model-specific helpers
```

The walkthrough follows this path after scheduling:

```text
serve/scheduler
  -> pipelines/lib/pipeline_variants/text_generation.py
  -> pipelines/architectures/llama3/batch_processor.py
  -> pipelines/architectures/llama3/model.py
```

### `pipelines/lib/`

Shared pipeline behavior lives here. For text generation, the important
variant is:

```text
max/python/max/pipelines/lib/pipeline_variants/text_generation.py
```

Its job is to prepare the scheduled batch, invoke the architecture-specific
model, process logits, apply sampling and grammar constraints, and produce
per-request outputs.

The variant knows the generation workflow, but it does not need to contain
every architecture-specific tensor-shaping detail.

### `pipelines/architectures/llama3/`

This folder contains Llama 3-specific runtime behavior.

```text
max/python/max/pipelines/architectures/llama3/
|
+-- batch_processor.py
|     TextContext -> Llama3Inputs
|     token packing and row offsets
|     KV-cache and signal-buffer attachment
|     graph-output -> ModelOutputs conversion
|
+-- model.py
      Llama3Inputs -> compiled model execution
      graph-output handoff to batch processor
      Llama configuration and graph construction
```

The batch processor is an adapter. The model wrapper is the execution owner.
Neither should be confused with the scheduler: the scheduler chooses the
batch, while these files encode that batch for the Llama graph and run it.

### `engine/`

```text
max/python/max/engine/
```

This area supports graph compilation, initialization, loading, and runtime
integration. It is usually behind the model wrapper rather than directly on
the HTTP request path.

Conceptually:

```text
model wrapper
  -> engine/runtime integration
  -> compiled executable graph
```

The Llama model's `self.model.execute(...)` call is the point where the
high-level pipeline reaches the compiled execution machinery.

### `nn/`

```text
max/python/max/nn/
|
+-- attention/
+-- transformer/
+-- layer/
+-- norm/
+-- moe/
+-- sampling/
+-- kv_cache/
+-- parallel/
+-- comm/
```

This folder contains reusable Python-level neural-network components and
runtime building blocks. It supports model graph construction and execution;
it is not the HTTP serving layer.

Examples of responsibility include attention, transformer layers, sampling,
KV-cache handling, normalization, and parallelism support.

### `max/kernels/src/`

```text
max/kernels/src/
|
+-- pipeline/
+-- kv_cache/
+-- nn/
+-- graph_compiler/
+-- quantization/
+-- comm/
+-- linalg/
+-- structured_kernels/
```

This is a lower-level performance area. It contains kernels and graph
compiler support used beneath the Python APIs. A request normally reaches this
layer indirectly through the compiled model graph.

```text
Python serving and model code
  -> graph/runtime API
  -> compiled graph
  -> kernels/src implementation
  -> CPU/GPU execution
```

### `max/mojo/max/gpu/`

```text
max/mojo/max/gpu/
|
+-- compute/
+-- host/
+-- memory/
+-- primitives/
+-- sync/
```

This area provides lower-level Mojo GPU and device facilities. It is further
below the serving request path than `max/python/max/serve/` and is consulted
when the compiled runtime needs device operations, memory operations,
synchronization, or GPU primitives.

## Walkthrough Documentation Tree

```text
murali_docs/artifacts/source_code_walkthrough/
|
+-- AGENTS.md
|     folder-local rules for style, links, validation, and git completion
|
+-- 0000_source_code_walkthrough.md
|     main index and one top-level collapsible section per major source file
|
+-- 0001_parse_openai_request_body.md
|     raw HTTP JSON -> CreateChatCompletionRequest
|
+-- 0002_get_pipeline.md
|     model name -> selected serving pipeline
|
+-- 0002_get_pipeline_0a_PipelineTokenizer.md
|     PipelineTokenizer capabilities used during pipeline selection
|
+-- 0003_openai_parse_chat_completion_request.md
|     OpenAI messages/media -> normalized chat request data
|
+-- 0004_TextGenerationRequest.md
|     normalized API data -> MAX TextGenerationRequest
|
+-- 0005_streaming_vs_non_streaming.md
|     response mode -> HTTP response shape
|
+-- 0006_tokenizer_new_context.md
|     TextGenerationRequest -> TextContext
|
+-- 0007_create_buffered_detokenizer.md
|     token IDs -> safely buffered text fragments
|
+-- 0008_note_awaiting_admission.md
|     API-side awaiting-admission metric lifecycle
|
+-- 0009_model_worker_stream.md
|     TextContext -> worker queue -> async response stream
|
+-- 0010_scheduler_iteration.md
|     worker queue -> one scheduler iteration
|
+-- 0011_text_batch_constructor.md
|     admitted contexts -> prefill/decode batch inputs
|
+-- 0012_pipeline_execution.md
|     TextGenerationInputs -> logits -> sampled token IDs
|
+-- 0013_llama_input_staging.md
|     TextContext batch -> packed Llama3Inputs buffers
|
+-- 0014_llama_model_execution.md
|     Llama3Inputs -> compiled Llama graph -> ModelOutputs
|
+-- 0015_folder_structure.md
      this guide to the implementation and documentation trees
```

## How The Note Numbers Relate To Runtime

The numbers are a reading sequence, not a claim that every helper executes in
exact numeric order.

```text
0000  index and main source path
  |
  +-- 0001-0005  HTTP ingress and API request preparation
  |
  +-- 0006-0009  context creation, detokenization, and worker handoff
  |
  +-- 0010-0011  scheduling and batch construction
  |
  +-- 0012-0014  pipeline, Llama input staging, and model execution
  |
  +-- 0015  structure reference for navigating all of the above
```

The runtime handoff can be summarized as:

```text
HTTP JSON
  -> CreateChatCompletionRequest
  -> TextGenerationRequest
  -> TextContext
  -> worker queue
  -> scheduler iteration
  -> TextGenerationInputs
  -> Llama3Inputs
  -> compiled model
  -> token IDs and response outputs
```

## How To Navigate The Repository

Use the appropriate layer for the question:

```text
Question                                             Start here
--------------------------------------------------   ----------------------------------------------
How does HTTP JSON become a typed request?          max/python/max/serve/router/openai_routes.py
How is the model selected?                          max/python/max/serve/router/openai_routes.py
How does the request reach the worker?              max/python/max/serve/worker_interface/zmq_interface.py
Who chooses the next batch?                         max/python/max/serve/scheduler/
How are token IDs packed for Llama?                 max/python/max/pipelines/architectures/llama3/batch_processor.py
How is the compiled model called?                   max/python/max/pipelines/architectures/llama3/model.py
Where are reusable graph components?                max/python/max/nn/
Where are low-level kernels?                        max/kernels/src/
Where are the walkthrough explanations?             murali_docs/artifacts/source_code_walkthrough/
```

## Short Version

```text
serve/                 receives, validates, routes, queues, and schedules
pipelines/              prepares model work and owns architecture code
engine/                 connects model code to compiled execution
nn/                     provides reusable neural-network components
kernels/src/            implements lower-level performance kernels
mojo/max/gpu/           provides lower-level GPU/device facilities
source_code_walkthrough explains how these layers connect for one request
```
