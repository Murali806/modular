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
|   |       |   |   +-- openai.py                               # OpenAI-compatible request schemas
|   |       |   |   +-- openai-docs.yaml                        # schema/documentation metadata
|   |       |   |   +-- README.md                               # schema package notes
|   |       |   |   +-- __init__.py                             # package exports
|   |       |   |
|   |       |   +-- parser/                                     # parsing and output helper behavior
|   |       |   |   +-- json_utils.py                           # JSON parsing/serialization helpers
|   |       |   |   +-- llama_tool_parser.py                    # Llama tool-call parsing
|   |       |   |   +-- tool_call_normalization.py              # normalizes tool-call structures
|   |       |   |
|   |       |   +-- telemetry/                                  # metrics, tracing, and measurements
|   |       |   |   +-- metrics.py                              # metric definitions/recording
|   |       |   |   +-- _metrics_catalog.py                     # catalog of known metrics
|   |       |   |   +-- _trace_context.py                       # trace context propagation
|   |       |   |   +-- asyncio_controller.py                   # async runtime telemetry control
|   |       |   |   +-- stopwatch.py                            # elapsed-time measurement
|   |       |   |   +-- process_controller.py                   # process-level telemetry control
|   |       |   |
|   |       |   +-- dependencies/                               # serving dependency integration
|   |       |   |   +-- request_parsing.py                       # request parsing dependency helpers
|   |       |   |
|   |       |   +-- recordreplay/                               # request/response recording and replay
|   |       |   |   +-- middleware.py                            # intercepts requests/responses
|   |       |   |   +-- replay.py                                # replays recorded traffic
|   |       |   |   +-- jsonl.py                                 # JSON Lines persistence
|   |       |   |   +-- schema.py                                # record/replay data schema
|   |       |   |   +-- interfaces.py                            # record/replay contracts
|   |       |   |
|   |       |   +-- media/                                      # serving support for media inputs
|   |       |   |   +-- audio.py                                 # audio input handling
|   |       |   |   +-- generated_media.py                       # generated media representations
|   |       |   |
|   |       |   +-- mocks/                                      # test and development doubles
|   |       |       +-- mock_api_requests.py                      # mock API request objects
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
|   |       |   +-- api.py                                     # engine-facing Python API
|   |       |   +-- mlrt.py                                    # ML runtime integration
|   |       |   +-- _precompiled_mefs.py                       # precompiled graph artifacts
|   |       |   +-- _compilation_stats.py                      # compilation measurements
|   |       |   +-- __init__.py                                # package exports
|   |       |
|   |       +-- nn/                                             # reusable Python neural-network pieces
|   |       |   +-- attention/                                  # attention building blocks
|   |       |   |   +-- multihead_attention.py                  # standard multi-head attention
|   |       |   |   +-- ragged_attention.py                     # variable-length attention
|   |       |   |   +-- attention_with_rope.py                  # attention with rotary embeddings
|   |       |   |   +-- multi_latent_attention.py               # multi-latent attention
|   |       |   |   +-- mask_config.py                          # attention-mask configuration
|   |       |   |
|   |       |   +-- transformer/                                # transformer components
|   |       |   |   +-- transformer.py                          # transformer block structure
|   |       |   |   +-- distributed_transformer.py              # distributed transformer support
|   |       |   |
|   |       |   +-- layer/                                      # reusable neural-network layers
|   |       |   |   +-- layer.py                                # base layer behavior
|   |       |   |   +-- layer_list.py                           # ordered layer collection
|   |       |   |
|   |       |   +-- norm/                                       # normalization components
|   |       |   |   +-- rms_norm.py                             # RMS normalization
|   |       |   |   +-- layer_norm.py                           # layer normalization
|   |       |   |   +-- group_norm.py                           # group normalization
|   |       |   |
|   |       |   +-- moe/                                        # mixture-of-experts components
|   |       |   |   +-- moe.py                                   # MoE routing/expert structure
|   |       |   |   +-- stacked_moe.py                           # stacked expert implementation
|   |       |   |   +-- expert_parallel.py                       # expert parallelism
|   |       |   |   +-- sigmoid_router.py                        # sigmoid router
|   |       |   |   +-- quant_strategy.py                        # quantization strategy
|   |       |   |
|   |       |   +-- sampling/                                   # sampling-related components
|   |       |   |   +-- min_p.py                                 # min-p sampling support
|   |       |   |   +-- rejection_sampler.py                     # rejection sampling support
|   |       |   |
|   |       |   +-- kv_cache/                                   # KV-cache support
|   |       |   |   +-- manager.py                               # cache lifecycle/management
|   |       |   |   +-- cache_params.py                          # cache configuration
|   |       |   |   +-- input_types.py                           # cache input types
|   |       |   |   +-- metrics.py                               # cache metrics
|   |       |   |   +-- data_parallelism_utils.py                # cache/data-parallel helpers
|   |       |   |
|   |       |   +-- parallel/                                   # parallel execution support
|   |       |   |   +-- numpy_parallel_ops.py                    # parallel operations helper
|   |       |   |
|   |       |   +-- comm/                                       # communication components
|   |       |       +-- allreduce.py                             # all-reduce operation
|   |       |       +-- ep/                                      # expert-parallel communication
|   |       |
|   |       +-- kv_cache/                                       # shared KV-cache APIs and management
|   |       |   +-- __init__.py                                 # package exports
|   |       |
|   |       +-- driver/                                         # device and runtime driver integration
|   |       |   +-- driver.py                                   # driver-facing operations
|   |       |   +-- buffer.py                                   # device buffer abstraction
|   |       |   +-- _hazard.py                                  # memory hazard handling
|   |       |
|   |       +-- graph/                                          # graph-level Python APIs and helpers
|   |           +-- graph.py                                    # graph construction API
|   |           +-- value.py                                    # graph value representation
|   |           +-- type.py                                     # graph type representation
|   |           +-- shape.py                                    # shape representation
|   |           +-- dim.py                                      # dimension representation
|   |           +-- weight.py                                   # graph weight representation
|   |           +-- buffer_utils.py                             # graph buffer helpers
|   |           +-- quantization.py                             # graph quantization helpers
|   |           +-- ops/                                        # graph operation definitions
|   |           +-- weights/                                    # graph weight utilities
|   |
|   +-- kernels/                                                # lower-level performance implementation
|   |   |
|   |   +-- src/                                                 # kernel and graph-compiler sources
|   |       +-- pipeline/                                       # pipeline-oriented kernels
|   |       |   +-- compiler.mojo                              # pipeline compilation support
|   |       |   +-- pipeline_dsl.mojo                           # pipeline DSL
|   |       |   +-- program.mojo                                # pipeline program model
|   |       |   +-- schedulers.mojo                             # pipeline scheduling
|   |       |   +-- types.mojo                                  # pipeline types
|   |       |
|   |       +-- kv_cache/                                       # KV-cache kernels and operations
|   |       |   +-- paged_sparse_kv_index_remap.mojo             # KV index remapping
|   |       |   +-- types.mojo                                  # KV-cache kernel types
|   |       |
|   |       +-- nn/                                             # neural-network kernels
|   |       |   +-- attention/                                  # attention kernel family
|   |       |   +-- sampling/                                   # sampling kernel family
|   |       |   +-- activations.mojo                            # activation functions
|   |       |   +-- softmax.mojo                                # softmax operation
|   |       |   +-- rope.mojo                                   # rotary position embedding
|   |       |   +-- kv_cache.mojo                               # KV-cache operations
|   |       |   +-- topk.mojo                                   # top-k selection
|   |       |   +-- toppminp.mojo                               # top-p/min-p sampling
|   |       |
|   |       +-- graph_compiler/                                 # compiler primitives and registrations
|   |       |   +-- builtin_kernels/                            # built-in kernel registrations
|   |       |   +-- builtin_primitives/                          # built-in compiler primitives
|   |       |   +-- extensibility/                               # compiler extension points
|   |       |
|   |       +-- quantization/                                   # quantized computation support
|   |       |   +-- qmatmul.mojo                                # quantized matrix multiplication
|   |       |   +-- qmatmul_gpu.mojo                             # GPU quantized matmul
|   |       |   +-- per_channel_grouped_4bit.mojo                # grouped 4-bit quantization
|   |       |
|   |       +-- comm/                                           # device/distributed communication
|   |       |   +-- allreduce.mojo                               # all-reduce
|   |       |   +-- allgather.mojo                               # all-gather
|   |       |   +-- reducescatter.mojo                           # reduce-scatter
|   |       |   +-- device_collective.mojo                       # collective operations
|   |       |   +-- sync.mojo                                    # communication synchronization
|   |       |
|   |       +-- linalg/                                         # linear algebra and matmul support
|   |       |   +-- matmul/                                      # matrix multiplication family
|   |       |   +-- arch/                                        # architecture-specific implementations
|   |       |   +-- bmm.mojo                                    # batched matrix multiplication
|   |       |   +-- gemv.mojo                                   # matrix-vector multiplication
|   |       |   +-- grouped_matmul.mojo                          # grouped matrix multiplication
|   |       |
|   |       +-- structured_kernels/                             # structured/high-level kernel support
|   |           +-- pipeline.mojo                                # structured kernel pipeline
|   |           +-- pipeline_backend.mojo                         # backend integration
|   |           +-- tile_types.mojo                              # tile representations
|   |           +-- barriers.mojo                                # synchronization barriers
|   |
|   |       # Python model/serving code normally reaches this layer through
|   |       # the compiled graph; request JSON does not call kernels directly.
|   |
|   +-- mojo/                                                    # lower-level Mojo implementation
|       |
|       +-- max/                                                 # Mojo MAX package
|           +-- gpu/                                             # GPU/device facilities
|           |   +-- compute/                                    # GPU compute operations
|           |   |   +-- tensor_ops.mojo                          # tensor operations
|           |   |   +-- mma.mojo                                 # matrix multiply-accumulate
|           |   |   +-- arch/                                    # architecture-specific GPU code
|           |   |   +-- mma_util.mojo                            # MMA helpers
|           |   |
|           |   +-- host/                                       # host-side GPU integration
|           |   |   +-- device_context.mojo                      # device context
|           |   |   +-- device_graph.mojo                         # device graph support
|           |   |   +-- compile.mojo                              # GPU compilation support
|           |   |   +-- device_attribute.mojo                     # device attributes
|           |   |   +-- info.mojo                                 # device information
|           |   |   +-- nvidia/                                  # NVIDIA-specific host code
|           |   |
|           |   +-- memory/                                     # device memory operations
|           |   |   +-- memory.mojo                               # memory APIs
|           |   |   +-- masked_load_apple.mojo                    # Apple masked load
|           |   |
|           |   +-- primitives/                                 # low-level GPU primitives
|           |   |   +-- block.mojo                                # block identity/control
|           |   |   +-- cluster.mojo                              # cluster operations
|           |   |   +-- grid_controls.mojo                        # grid controls
|           |   |   +-- id.mojo                                   # thread/block IDs
|           |   |   +-- warp.mojo                                 # warp operations
|           |   |
|           |   +-- sync/                                       # synchronization facilities
|           |       +-- sync.mojo                                 # synchronization APIs
|           |       +-- semaphore.mojo                            # semaphore support
|           |
|           +-- runtime/                                        # lower-level runtime facilities
|           |   +-- async_value.mojo                              # asynchronous values
|           |   +-- asyncrt.mojo                                  # async runtime integration
|           |   +-- tracing.mojo                                  # runtime tracing
|           |
|           +-- algorithm/                                      # lower-level algorithms
|           |   +-- functional.mojo                               # functional algorithms
|           |   +-- memory.mojo                                   # memory algorithms
|           |   +-- reduction.mojo                                # reductions
|           |   +-- backend/                                      # backend implementations
|           |
|           +-- benchmark/                                      # benchmark support
|               +-- bencher.mojo                                  # benchmark harness
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
