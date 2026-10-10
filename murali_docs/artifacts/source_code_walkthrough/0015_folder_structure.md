# MAX Serve Folder Structure

The tree below is the primary explanation. Read the text after each `#` as
the purpose of that folder or file. The implementation tree shows executable
MAX code; the documentation branch shows the notes that explain it.

```text
modular/                                                        # repository root
|
+-- max/                                                        # MAX platform implementation
|   |
|   +-- python/                                                  # Python orchestration and APIs
|   |   |
|   |   +-- max/                                                 # Python package named `max`
|   |       |
|   |       +-- serve/                                           # HTTP serving and request lifecycle
|   |       |   |
|   |       |   +-- router/                                     # OpenAI-compatible HTTP boundary
|   |       |   |   +-- openai_routes.py                        # receives JSON; validates; routes model
|   |       |   |                                              # creates TextGenerationRequest and
|   |       |   |                                              # chooses streaming/non-streaming response
|   |       |   |
|   |       |   +-- pipelines/                                  # serving-level request handoff
|   |       |   |   +-- llm.py                                   # request -> TextContext -> worker stream
|   |       |   |                                              # tokenizes; detokenizes; yields chunks
|   |       |   |
|   |       |   +-- worker_interface/                           # API process <-> model worker boundary
|   |       |   |   +-- zmq_interface.py                         # bounded queue and response routing
|   |       |   |                                              # admission check; backpressure; IPC
|   |       |   |
|   |       |   +-- scheduler/                                  # continuous-batching control layer
|   |       |   |   +-- text_generation_scheduler.py             # repeats scheduler iterations
|   |       |   |   |                                            # retrieves; batches; executes; responds
|   |       |   |   +-- batch_constructor/                       # turns queued contexts into batches
|   |       |   |       +-- text_batch_constructor.py             # grammar readiness; prefill/decode
|   |       |   |                                                # queues; replica binding; budgets
|   |       |   |
|   |       |   +-- schemas/                                    # typed serving request/response objects
|   |       |   +-- parser/                                     # parsing and output helper behavior
|   |       |   +-- telemetry/                                  # metrics, tracing, and measurements
|   |       |   +-- dependencies/                               # serving dependency integration
|   |       |   +-- recordreplay/                               # request/response recording and replay
|   |       |   +-- media/                                      # serving support for media inputs
|   |       |   +-- mocks/                                      # test and development doubles
|   |       |
|   |       +-- pipelines/                                      # model-pipeline implementation
|   |       |   |
|   |       |   +-- lib/                                       # shared pipeline behavior
|   |       |   |   +-- pipeline_variants/
|   |       |   |       +-- text_generation.py                    # batch -> model -> logits -> tokens
|   |       |   |                                                # applies sampling and grammar rules
|   |       |   |
|   |       |   +-- architectures/                              # model-specific implementations
|   |       |       +-- llama3/                                 # Llama 3 runtime implementation
|   |       |           +-- batch_processor.py                    # TextContext batch -> Llama3Inputs
|   |       |           |                                        # packs tokens; row offsets; KV cache
|   |       |           +-- model.py                              # Llama3Inputs -> compiled model
|   |       |                                                # graph execution -> ModelOutputs
|   |       |
|   |       +-- engine/                                        # graph compile/load/runtime integration
|   |       |                                                   # model wrappers reach compiled graphs here
|   |       |
|   |       +-- nn/                                             # reusable Python neural-network pieces
|   |       |   +-- attention/                                  # attention building blocks
|   |       |   +-- transformer/                                # transformer components
|   |       |   +-- layer/                                      # reusable neural-network layers
|   |       |   +-- norm/                                       # normalization components
|   |       |   +-- moe/                                        # mixture-of-experts components
|   |       |   +-- sampling/                                   # sampling-related components
|   |       |   +-- kv_cache/                                   # KV-cache support
|   |       |   +-- parallel/                                   # parallel execution support
|   |       |   +-- comm/                                       # communication components
|   |       |
|   |       +-- kv_cache/                                       # shared KV-cache APIs and management
|   |       +-- driver/                                         # device and runtime driver integration
|   |       +-- graph/                                          # graph-level Python APIs and helpers
|   |
|   +-- kernels/                                                # lower-level performance implementation
|   |   |
|   |   +-- src/                                                 # kernel and graph-compiler sources
|   |       +-- pipeline/                                       # pipeline-oriented kernels
|   |       +-- kv_cache/                                       # KV-cache kernels and operations
|   |       +-- nn/                                             # neural-network kernels
|   |       +-- graph_compiler/                                 # compiler primitives and registrations
|   |       +-- quantization/                                   # quantized computation support
|   |       +-- comm/                                           # device/distributed communication
|   |       +-- linalg/                                         # linear algebra and matmul support
|   |       +-- structured_kernels/                             # structured/high-level kernel support
|   |
|   |       # Python model/serving code normally reaches this layer through
|   |       # the compiled graph; request JSON does not call kernels directly.
|   |
|   +-- mojo/                                                    # lower-level Mojo implementation
|       |
|       +-- max/                                                 # Mojo MAX package
|           +-- gpu/                                             # GPU/device facilities
|           |   +-- compute/                                    # GPU compute operations
|           |   +-- host/                                       # host-side GPU integration
|           |   +-- memory/                                     # device memory operations
|           |   +-- primitives/                                 # low-level GPU primitives
|           |   +-- sync/                                       # synchronization facilities
|           +-- runtime/                                        # lower-level runtime facilities
|           +-- algorithm/                                      # lower-level algorithms
|           +-- benchmark/                                      # benchmark support
|                                                               # Python and compiled code eventually
|                                                               # use these facilities for device work.
|
+-- murali_docs/                                                # documentation created for this study
    |
    +-- artifacts/                                               # generated/working documentation artifacts
        |
        +-- source_code_walkthrough/                             # one-request learning path
            |
            +-- AGENTS.md                                       # local rules for this documentation folder
            |                                                    # style; links; validation; git workflow
            |
            +-- 0000_source_code_walkthrough.md                 # main index and top-level source path
            |                                                    # one collapsible section per major file
            |
            +-- 0001_parse_openai_request_body.md                # raw HTTP body -> typed Pydantic request
            +-- 0002_get_pipeline.md                             # model name -> selected pipeline
            +-- 0002_get_pipeline_0a_PipelineTokenizer.md        # tokenizer capability contract
            +-- 0003_openai_parse_chat_completion_request.md     # OpenAI chat data -> normalized messages
            +-- 0004_TextGenerationRequest.md                     # normalized API data -> internal request
            +-- 0005_streaming_vs_non_streaming.md               # response mode -> HTTP response shape
            +-- 0006_tokenizer_new_context.md                     # internal request -> TextContext
            +-- 0007_create_buffered_detokenizer.md               # token IDs -> safe text fragments
            +-- 0008_note_awaiting_admission.md                   # admission metric before/after handoff
            +-- 0009_model_worker_stream.md                       # context -> worker queue -> response stream
            +-- 0010_scheduler_iteration.md                       # worker queue -> scheduler iteration
            +-- 0011_text_batch_constructor.md                    # contexts -> prefill/decode batch
            +-- 0012_pipeline_execution.md                        # batch -> logits -> sampled token IDs
            +-- 0013_llama_input_staging.md                       # contexts -> packed Llama3Inputs buffers
            +-- 0014_llama_model_execution.md                     # Llama3Inputs -> compiled Llama model
            +-- 0015_folder_structure.md                          # this tree and its explanations
```

## The Request Path Inside The Tree

```text
client HTTP JSON
  |
  +-- max/python/max/serve/router/openai_routes.py
  |     validate and normalize API data
  |     output: TextGenerationRequest
  |
  +-- max/python/max/serve/pipelines/llm.py
  |     tokenize and create TextContext
  |     output: worker submission request
  |
  +-- max/python/max/serve/worker_interface/zmq_interface.py
  |     cross API-process/model-worker boundary
  |     output: async worker response stream
  |
  +-- max/python/max/serve/scheduler/
  |     select pending contexts for continuous batching
  |     output: TextGenerationInputs
  |
  +-- max/python/max/pipelines/lib/pipeline_variants/text_generation.py
  |     prepare batch; run model; sample tokens
  |     output: generated token IDs and per-request outputs
  |
  +-- max/python/max/pipelines/architectures/llama3/batch_processor.py
  |     pack variable-length requests into graph-shaped buffers
  |     output: Llama3Inputs
  |
  +-- max/python/max/pipelines/architectures/llama3/model.py
        call compiled Llama graph
        output: ModelOutputs -> scheduler -> worker -> API response
```

## How To Read The Branches

```text
serve/                 # request ownership: HTTP, routing, queues, scheduling
  |
  +-- router/           # what came from the client?
  +-- pipelines/        # how does API work reach the worker?
  +-- worker_interface/ # how does work cross the process boundary?
  +-- scheduler/        # when and with what other requests does it run?

pipelines/              # model-work ownership: batch preparation and execution
  |
  +-- lib/              # what is common to text generation?
  +-- architectures/    # what is specific to Llama or another model?

engine/, nn/            # reusable graph/runtime building blocks
kernels/src/, mojo/     # lower-level compiled and device implementation

source_code_walkthrough/ # explanations that mirror the runtime path above
```

## Short Version

```text
HTTP request
  -> serve/router
  -> serve/pipelines
  -> serve/worker_interface
  -> serve/scheduler
  -> pipelines/lib
  -> pipelines/architectures/llama3
  -> engine / nn / kernels / mojo runtime
  -> generated response
```
