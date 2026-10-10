# MAX Serve Folder Structure

The tree below is the primary explanation. Read the text after each `#` as
the purpose of that folder or file. The implementation tree shows executable
MAX code; the documentation branch shows the notes that explain it.

## Expandable Root-Level Map

<details>
<summary><strong>bazel/ and bazel*</strong> - build rules, toolchains, generated outputs, and test logs</summary>

```text
bazel/
|
+-- internal/          # repository build macros and toolchain integration
+-- pip/               # Python dependency rules
+-- lint/              # build and source lint wrappers
+-- docs/              # Bazel usage documentation
bazelw                 # Bazel wrapper command
bazel-bin/             # generated binary-output view
bazel-out/             # generated configured-output view
bazel-testlogs/        # generated test-log view
```

```text
bazel/internal/cc-toolchain/tools/
  +-- builtin_module_map.bzl
  +-- linker-driver.sh
  +-- multi-platform-clang.sh
bazel/pip/pycross/
  +-- dependency.py
  +-- download.py
  +-- generate.py
bazel/lint/
  +-- buildifier_wrapper.py
  +-- ruff_wrapper.py
  +-- rumdl_wrapper.py
```
</details>

<details>
<summary><strong>Init/</strong> - process initialization and development signal handling</summary>

```text
Init/
|
+-- include/Init/      # public initialization headers
+-- lib/               # initialization implementation
+-- integration-test/  # integration tests
+-- unittests/         # unit tests
```

```text
Init/include/Init/
  +-- Init.h
  +-- DevelopmentSignalHandler.h
Init/lib/
  +-- Init.cpp
  +-- DevelopmentSignalHandler.cpp
Init/unittests/
  +-- DevelopmentSignalHandlerTests.cpp
```
</details>

<details>
<summary><strong>AsyncRT/</strong> - asynchronous values, queues, timers, allocators, and runtime support</summary>

```text
AsyncRT/
|
+-- include/AsyncRT/   # public runtime/compiler-support headers
+-- lib/               # runtime implementations
+-- docs/              # runtime documentation
+-- benchmarks/        # runtime benchmarks
+-- test/              # integration tests
+-- unittests/         # runtime unit tests
+-- tools/             # runtime diagnostic tools
```

```text
AsyncRT/include/AsyncRT/Runtime/
  +-- AsyncValue.h
  +-- WorkQueue.h
  +-- TimerHeap.h
  +-- Globals/RuntimeGlobal.h
AsyncRT/lib/Runtime/
  +-- AsyncValue.cpp
  +-- ThreadPoolWorkQueue.cpp
  +-- TimerHeap.cpp
AsyncRT/unittests/
  +-- AsyncValueTest.cpp
  +-- WorkQueueTest.cpp
```
</details>

<details>
<summary><strong>Cache/</strong> - blob caching, cached transforms, cache telemetry, and cache-manager tools</summary>

```text
Cache/
|
+-- include/Cache/     # cache APIs and key helpers
+-- lib/               # cache implementations
+-- tools/cache-mgr/   # cache-manager executable
+-- test/cache-mgr/    # cache-manager fixtures/tests
+-- unittests/         # cache unit tests
```

```text
Cache/include/Cache/
  +-- BlobCache.h
  +-- CachedTransform.h
  +-- Support/Keys.h
Cache/lib/
  +-- BlobCache.cpp
  +-- CachedTransform.cpp
Cache/test/cache-mgr/Inputs/
  +-- empty.txt
  +-- some_file.txt
```
</details>

<details>
<summary><strong>Config/</strong> - shared configuration and version support</summary>

```text
Config/
|
+-- include/Config/    # public configuration/version headers
+-- lib/               # configuration implementation
+-- BUILD.bazel        # Config build target
```

```text
Config/include/Config/Version.h
Config/include/GeneratedVersion.h.tmpl
Config/lib/Version.cpp
```
</details>

<details>
<summary><strong>Support/</strong> - shared C++ data structures, diagnostics, ML, telemetry, threading, and tools</summary>

```text
Support/
|
+-- include/Support/   # public shared APIs
+-- lib/               # implementations
+-- tools/             # support executables
+-- test/              # integration tests
+-- unittests/         # unit tests
+-- benchmarks/        # benchmarks
```

```text
Support/include/Support/
  +-- ADT/DenseStringMap.h
  +-- Compiler/Bytecode.h
  +-- ML/DType.h
  +-- ML/TensorShape.h
  +-- Telemetry/Telemetry.h
  +-- Threading/HWInfo.h
Support/lib/
  +-- Compiler/DiagnosticHandler.cpp
  +-- ML/TensorShape.cpp
  +-- Telemetry/TelemetryContext.cpp
Support/tools/system-info/system-info.cpp
Support/unittests/Log/LogTest.cpp
```
</details>

<details>
<summary><strong>Mojo/</strong> - Mojo language, standard library, compiler, tools, examples, and tests</summary>

```text
Mojo/
|
+-- stdlib/            # Mojo standard library and tests
+-- lib/               # compiler, dialect, parser, runtime libraries
+-- tools/             # Mojo CLI, formatter, docs, REPL, and compiler tools
+-- python/            # Python interoperability package
+-- docs/              # language/API documentation
+-- examples/          # examples
+-- test/              # compiler/language tests
+-- unittests/         # tooling/compiler unit tests
```

```text
Mojo/stdlib/std/collections/string/
  +-- string.mojo
  +-- format.mojo
  +-- _parsing_numbers/parsing_floats.mojo
Mojo/stdlib/std/testing/prop/strategy/
  +-- list_strategy.mojo
  +-- string_strategy.mojo
Mojo/lib/MojoParser/
  +-- Lexer.cpp
  +-- ParserExprs.cpp
Mojo/tools/mojo/
  +-- Format/mojo-format.cpp
  +-- Doc/mojo-doc.cpp
  +-- Run/mojo-run.cpp
```
</details>

<details>
<summary><strong>tools/</strong> - repository-level developer and build helpers</summary>

```text
tools/
|
+-- build_defs/cc/     # C/C++ build definitions
+-- bazel              # repository Bazel helper
```

```text
tools/build_defs/cc/
  +-- BUILD.bazel
  +-- link_hack.bzl
+-- bazel
```
</details>

<details>
<summary><strong>utils/</strong> - local setup and operational scripts</summary>

```text
utils/
|
+-- local_transformers_setup/  # local transformer setup/cleanup
+-- setup-gpu-clock.sh         # GPU clock utility
```

```text
utils/local_transformers_setup/
  +-- README.md
  +-- setup_local_transformers.sh
  +-- cleanup_local_transformers.sh
+-- setup-gpu-clock.sh
```
</details>

<details>
<summary><strong>docs/, Licenses/, and root configuration</strong> - repository guidance, legal files, and project configuration</summary>

```text
docs/                 # repository-wide documentation
Licenses/             # third-party license materials
README.md             # repository orientation
CONTRIBUTING.md       # contribution workflow
pyproject.toml        # Python project/tool configuration
```

```text
+-- docs/                 # documentation files and subfolders
+-- Licenses/             # license files
+-- README.md
+-- CONTRIBUTING.md
+-- pyproject.toml
```
</details>

<details>
<summary><strong>murali_docs/</strong> - this source-code walkthrough and related study artifacts</summary>

```text
murali_docs/
|
+-- artifacts/
    +-- source_code_walkthrough/  # numbered request-flow notes and this guide
```

```text
murali_docs/artifacts/source_code_walkthrough/
  +-- AGENTS.md
  +-- 0000_source_code_walkthrough.md
  +-- 0001_parse_openai_request_body.md
  +-- 0002_get_pipeline.md
  +-- 0003_openai_parse_chat_completion_request.md
  +-- 0004_TextGenerationRequest.md
  +-- 0005_streaming_vs_non_streaming.md
  +-- 0006_tokenizer_new_context.md
  +-- 0007_create_buffered_detokenizer.md
  +-- 0008_note_awaiting_admission.md
  +-- 0009_model_worker_stream.md
  +-- 0010_scheduler_iteration.md
  +-- 0011_text_batch_constructor.md
  +-- 0012_pipeline_execution.md
  +-- 0013_llama_input_staging.md
  +-- 0014_llama_model_execution.md
  +-- 0015_folder_structure.md
```
</details>

<details>
<summary><strong>max/</strong> - expand to view the complete recursive MAX implementation tree</summary>

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
|   |       |           +-- ep_config.py                          # expert-parallel configuration
|   |       |           +-- ep_kernels.py                         # expert-parallel kernel interface
|   |       |           +-- ep_manager.py                         # expert-parallel lifecycle/coordination
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
|   |           |   +-- matmul.py                               # matrix multiplication graph op
|   |           |   +-- reduction.py                             # reduction graph ops
|   |           |   +-- reshape.py / transpose.py                # shape/layout graph ops
|   |           |   +-- gather.py / scatter.py                   # indexed data movement ops
|   |           |   +-- allreduce.py / allgather.py               # distributed graph ops
|   |           |   +-- quantized.py                             # quantized graph ops
|   |           |   +-- validation.py / support.py                # op validation/support checks
|   |           |                                                # many additional operation modules
|   |           +-- weights/                                    # graph weight utilities
|   |               +-- load.py                                  # general weight loading
|   |               +-- format.py                                # weight format handling
|   |               +-- weights.py                               # weight representation/utilities
|   |               +-- load_safetensors.py                      # Safetensors loading
|   |               +-- load_gguf.py                             # GGUF loading
|   |               +-- _gguf_reader.py                          # GGUF reader internals
|   |               +-- loader_wrappers.py                       # loader adapters/wrappers
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
|   |       |   |   +-- cpu/                                   # CPU attention kernels
|   |       |   |   |   +-- mha.mojo                            # CPU multi-head attention
|   |       |   |   +-- gpu/                                   # GPU attention kernels
|   |       |   |       +-- mha.mojo                            # GPU multi-head attention
|   |       |   |       +-- mla.mojo                            # multi-latent attention
|   |       |   |       +-- mha_cross.mojo                      # cross-attention
|   |       |   |       +-- sparse_indexer_common.mojo           # sparse-index helpers
|   |       |   |       +-- nvidia/                             # NVIDIA variants
|   |       |   |       |   +-- common.mojo                       # shared NVIDIA attention helpers
|   |       |   |       |   +-- mha_tile_scheduler.mojo           # MHA tile scheduling
|   |       |   |       |   +-- sm90/                             # NVIDIA SM90 implementation
|   |       |   |       |   |   +-- attention.mojo                  # SM90 attention path
|   |       |   |       |   |   +-- mha.mojo                        # SM90 multi-head attention
|   |       |   |       |   +-- sm100/                            # NVIDIA SM100 implementation
|   |       |   |       |       +-- attention.mojo                  # SM100 attention core
|   |       |   |       |       +-- dispatch.mojo                   # SM100 dispatch selection
|   |       |   |       |       +-- kernel.mojo                     # SM100 kernel entry points
|   |       |   |       |       +-- attention_utils.mojo            # attention utilities
|   |       |   |       |       +-- mla_prefill.mojo                 # MLA prefill path
|   |       |   |       |       +-- mla_decode_dispatch.mojo         # MLA decode dispatch
|   |       |   |       |       +-- mla_decode_combine.mojo          # MLA decode combination
|   |       |   |       |       +-- softmax_warp.mojo                # warp-level softmax
|   |       |   |       |       +-- sm100/mha_depth512/              # depth-512 MHA specialization
|   |       |   |       |
|   |       |   |       +-- amd_rdna/                           # AMD RDNA variants
|   |       |   |       |   +-- attention.mojo                     # AMD attention entry point
|   |       |   |       |   +-- buffers.mojo                       # AMD attention buffers
|   |       |   |       |   +-- config.mojo                        # AMD attention configuration
|   |       |   |       |   +-- mha_prefill.mojo                   # AMD MHA prefill
|   |       |   |       |   +-- mha_decode.mojo                    # AMD MHA decode
|   |       |   |       |   +-- mma.mojo                           # AMD matrix multiply-accumulate
|   |       |   |       |   +-- softmax.mojo                        # AMD softmax
|   |       |   |       |   +-- utils.mojo                          # AMD attention helpers
|   |       |   |       |
|   |       |   |       +-- apple/                              # Apple GPU variants
|   |       |       |       +-- fa_prefill.mojo                    # Apple flash-attention prefill
|   |       |       |       +-- naive_fa_decode.mojo                # Apple fallback decode
|   |       |       |       +-- DESIGN.md                          # Apple attention design notes
|   |       |       |       +-- MODEL_ENABLEMENT.md                 # model enablement notes
|   |       |   +-- sampling/                                   # sampling kernel family
|   |       |       +-- sampling.mojo                           # sampling implementation
|   |       |       +-- coop_row.mojo                           # cooperative row processing
|   |       |       +-- topk_fi.mojo                            # top-k selection
|   |       |       +-- topk_fi_cluster.mojo                    # clustered top-k selection
|   |       |   +-- activations.mojo                            # activation functions
|   |       |   +-- softmax.mojo                                # softmax operation
|   |       |   +-- rope.mojo                                   # rotary position embedding
|   |       |   +-- kv_cache.mojo                               # KV-cache operations
|   |       |   +-- topk.mojo                                   # top-k selection
|   |       |   +-- toppminp.mojo                               # top-p/min-p sampling
|   |       |
|   |       +-- graph_compiler/                                 # compiler primitives and registrations
|   |       |   +-- builtin_kernels/                            # built-in kernel registrations
|   |       |   |   +-- kernels.mojo                            # built-in kernel registry
|   |       |   |   +-- attention.mojo                          # built-in attention kernels
|   |       |   |   +-- linalg.mojo                             # built-in linear algebra kernels
|   |       |   |   +-- kv_cache.mojo                           # built-in KV-cache kernels
|   |       |   |   +-- quantization.mojo                       # built-in quantization kernels
|   |       |   |   +-- logprobs.mojo                           # built-in log-probability kernels
|   |       |   +-- builtin_primitives/                          # built-in compiler primitives
|   |       |   |   +-- primitives.mojo                          # primitive definitions
|   |       |   |   +-- buffer_plan.mojo                         # buffer planning primitives
|   |       |   +-- extensibility/                               # compiler extension points
|   |       |       +-- decorators.mojo                          # extension decorators
|   |       |       +-- operation_traits.mojo                     # operation trait definitions
|   |       |       +-- tensor_arg_traits.mojo                    # tensor argument traits
|   |       |       +-- managed_tensor_slice.mojo                 # managed tensor slices
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
|   |       |   |   +-- cpu/                                    # CPU matmul implementations
|   |       |   |   |   +-- default.mojo                         # default CPU matmul
|   |       |   |   |   +-- neon.mojo                            # ARM NEON matmul
|   |       |   |   |   +-- vnni.mojo                            # VNNI matmul
|   |       |   |   +-- gpu/                                    # GPU matmul implementations
|   |       |   |   |   +-- tile_scheduler.mojo                  # GPU tile scheduling
|   |       |   |   |   +-- tile_scheduler_splitk.mojo            # split-K scheduling
|   |       |   |   +-- vendor/                                 # vendor BLAS/matmul paths
|   |       |   |       +-- blas.mojo                            # BLAS integration
|   |       |   |       +-- matmul.mojo                          # vendor matmul integration
|   |       |   +-- arch/                                        # architecture-specific implementations
|   |       |       +-- cpu/                                    # CPU architecture paths
|   |       |       |   +-- neon_intrinsics.mojo                 # NEON intrinsics
|   |       |       |   +-- vnni_intrinsics.mojo                 # VNNI intrinsics
|   |       |       +-- amd/                                    # AMD architecture paths
|   |       |       |   +-- block_scaled_mma.mojo                # AMD block-scaled MMA
|   |       |       +-- apple/                                  # Apple architecture paths
|   |       |       |   +-- mma.mojo                             # Apple MMA
|   |       |       +-- sm100/                                  # NVIDIA SM100 paths
|   |       |           +-- mma.mojo                             # SM100 MMA
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
|           |   |       +-- mma_nvidia.mojo                       # NVIDIA MMA
|           |   |       +-- mma_nvidia_sm100.mojo                 # NVIDIA SM100 MMA
|           |   |       +-- mma_amd.mojo                           # AMD MMA
|           |   |       +-- mma_amd_rdna.mojo                      # AMD RDNA MMA
|           |   |       +-- mma_apple.mojo                         # Apple MMA
|           |   |       +-- tcgen05.mojo                           # tensor-core generation support
|           |   |   +-- mma_util.mojo                            # MMA helpers
|           |   |
|           |   +-- host/                                       # host-side GPU integration
|           |   |   +-- device_context.mojo                      # device context
|           |   |   +-- device_graph.mojo                         # device graph support
|           |   |   +-- compile.mojo                              # GPU compilation support
|           |   |   +-- device_attribute.mojo                     # device attributes
|           |   |   +-- info.mojo                                 # device information
|           |   |   +-- nvidia/                                  # NVIDIA-specific host code
|           |   |       +-- tma.mojo                               # NVIDIA Tensor Memory Accelerator support
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
|           |       +-- cpu/                                      # CPU algorithm implementations
|           |       |   +-- elementwise.mojo                       # CPU elementwise algorithms
|           |       |   +-- parallelize.mojo                       # CPU parallelization
|           |       |   +-- reduction.mojo                         # CPU reductions
|           |       |   +-- stencil.mojo                           # CPU stencil algorithms
|           |       +-- gpu/                                      # GPU algorithm implementations
|           |           +-- elementwise.mojo                       # GPU elementwise algorithms
|           |           +-- reduction.mojo                         # GPU reductions
|           |           +-- stencil.mojo                           # GPU stencil algorithms
|           |
|           +-- benchmark/                                      # benchmark support
|               +-- bencher.mojo                                  # benchmark harness
|                                                               # Python and compiled code eventually
|                                                               # use these facilities for device work.
|
```
</details>

## Other Root-Level Folders

The `max/` tree above is the request-flow focus. The following branches are
separate repository-level siblings that provide build, language, runtime,
cache, configuration, support, and developer-tool infrastructure. All of the
root-level branches are kept in individual expandable trees below.

## Individual Root Folder Trees

<!-- The previous consolidated non-MAX tree is hidden because each root folder
     now has its own expandable tree above. -->
<!--
```text
modular/
|
+-- bazel/                                                       # repository build rules and tooling
|   +-- BUILD.bazel                                               # Bazel package definition
|   +-- api.bzl                                                    # shared Bazel APIs/rules
|   +-- config.bzl                                                 # build configuration helpers
|   +-- internal/                                                  # internal build macros and tools
|   +-- lint/                                                      # build and lint integration
|   +-- pip/                                                       # Python dependency integration
|   +-- docs/                                                      # Bazel usage documentation
|
+-- bazelw                                                        # checked-in Bazel wrapper script
+-- BUILD.bazel                                                    # root Bazel package entry point
+-- MODULE.bazel                                                   # Bazel module declarations
+-- MODULE.bazel.lock                                              # resolved Bazel dependency versions
+-- REPO.bazel                                                      # repository-level Bazel configuration
+-- bazel-bin -> generated binaries link                            # Bazel output/bin view
+-- bazel-out -> generated output link                              # configured/intermediate outputs
+-- bazel-testlogs -> generated test-log link                        # test results and logs
+-- bazel-modular -> Bazel execution-root link                       # active Bazel workspace view
+-- build/                                                         # local build resources and logs
|
+-- Init/                                                          # process initialization support
|   +-- include/Init/                                               # public initialization headers
|   +-- lib/                                                        # initialization implementation
|   |   +-- Init.cpp                                                # initialization behavior
|   |   +-- DevelopmentSignalHandler.cpp                            # development signal handling
|   +-- integration-test/python/                                    # Python integration tests
|   +-- unittests/                                                   # initialization unit tests
|
+-- AsyncRT/                                                       # asynchronous runtime library
|   +-- include/AsyncRT/                                             # public async runtime headers
|   +-- lib/                                                        # runtime implementation libraries
|   |   +-- Runtime/                                                # async runtime implementation
|   |   +-- CompilerSupport/                                        # compiler/runtime support
|   |   +-- JemallocPreload/                                        # allocator preload support
|   |   +-- Support/                                                # shared async support
|   +-- docs/                                                       # AsyncRT API/concept documentation
|   +-- benchmarks/                                                 # async runtime benchmarks
|   +-- test/                                                       # runtime tests
|   +-- unittests/                                                  # runtime unit tests
|   +-- tools/                                                      # runtime diagnostic tools
|
+-- Cache/                                                         # caching library and cache-manager support
|   +-- include/Cache/                                               # public cache headers
|   +-- lib/                                                        # cache implementation
|   |   +-- BlobCache.cpp                                            # blob cache behavior
|   |   +-- CachedTransform.cpp                                      # cached transformation behavior
|   |   +-- CacheTelemetryContext.cpp                                # cache telemetry context
|   +-- docs/                                                       # cache documentation
|   +-- tools/cache-mgr/                                             # cache-manager tools
|   +-- test/cache-mgr/                                              # cache-manager tests
|   +-- unittests/                                                   # cache unit tests
|
+-- Config/                                                        # configuration and version support
|   +-- include/Config/                                               # public configuration headers
|   |   +-- GeneratedVersion.h.tmpl                                  # generated version template
|   +-- lib/                                                         # configuration implementation
|       +-- Version.cpp                                               # version implementation
|
+-- Support/                                                       # shared C++ platform support library
|   +-- include/Support/                                              # public support headers
|   +-- lib/                                                         # reusable support implementation
|   |   +-- diagnostics/error handling                               # diagnostics infrastructure
|   |   +-- filesystem/process/threading                             # platform services
|   |   +-- Log/Metrics/Telemetry                                    # observability support
|   |   +-- DeviceSpecs/CPUCache                                    # system/device information
|   +-- docs/                                                        # support API documentation
|   +-- test/                                                        # support integration tests
|   +-- unittests/                                                   # support unit tests
|   +-- tools/                                                       # support command-line tools
|   +-- benchmarks/                                                  # support benchmarks
|
+-- Mojo/                                                          # Mojo language and platform source
|   +-- stdlib/                                                      # Mojo standard library
|   +-- lib/                                                         # compiler, dialect, and runtime libraries
|   +-- include/                                                     # public Mojo/compiler headers
|   +-- docs/                                                        # Mojo language/API documentation
|   +-- examples/                                                    # learning and feature examples
|   +-- integration-test/                                            # integration tests
|   +-- test/                                                        # compiler/language/tool tests
|   +-- unittests/                                                   # compiler/tooling unit tests
|   +-- tools/                                                       # Mojo developer tools
|   +-- proposals/                                                   # language/library proposals
|   +-- python/                                                      # Python interoperability support
|
+-- tools/                                                         # repository-level developer tools
|   +-- build_defs/                                                   # shared build definitions
|   |   +-- cc/                                                       # C/C++ build definitions
|   +-- bazel                                                         # repository Bazel helper
|
+-- utils/                                                         # small operational/setup utilities
|   +-- local_transformers_setup/                                    # local transformer environment setup
|   |   +-- setup_local_transformers.sh                               # setup script
|   |   +-- cleanup_local_transformers.sh                             # cleanup script
|   |   +-- README.md                                                  # setup instructions
|   +-- setup-gpu-clock.sh                                             # GPU clock setup utility
|
+-- docs/                                                          # repository-wide documentation
+-- Licenses/                                                       # third-party license materials
+-- README.md                                                       # repository orientation
+-- CONTRIBUTING.md                                                # contribution workflow
+-- pyproject.toml                                                  # Python project/tool configuration
|
|   # deeper Bazel, runtime, cache, Support, Mojo, tools, and utils branches
+-- bazel/
|   +-- internal/                                                  # implementation of repository build rules
|   |   +-- modular_cc_library.bzl                                  # C/C++ library macro
|   |   +-- modular_cc_binary.bzl                                   # C/C++ binary macro
|   |   +-- modular_cc_test.bzl                                     # C/C++ test macro
|   |   +-- modular_py_library.bzl                                  # Python library macro
|   |   +-- modular_py_binary.bzl                                   # Python binary macro
|   |   +-- modular_py_test.bzl                                     # Python test macro
|   |   +-- mojo_library.bzl                                        # Mojo library macro
|   |   +-- mojo_binary.bzl                                         # Mojo binary macro
|   |   +-- mojo_test.bzl                                           # Mojo test macro
|   |   +-- mojo_toolchain.bzl                                      # Mojo toolchain configuration
|   |   +-- mef.bzl                                                  # compiled MEF integration
|   |   +-- precompile_mefs_plugin.py                               # precompiled MEF support
|   |   +-- pytest_runner.py                                        # Python test execution
|   |   +-- llvm-lit/                                               # lit test integration
|   |       +-- lit_shim.py                                         # lit entry shim
|   |       +-- modular_test_format.py                              # Modular test format
|   +-- pip/                                                         # Python dependency rules
|   |   +-- pycross/                                                 # Python package/dependency generation
|   |   |   +-- dependency.py                                        # dependency model
|   |   |   +-- generate.py                                          # dependency generation
|   |   |   +-- download.py                                          # package download support
|   |   +-- pydeps/                                                  # Python dependency analysis
|   |   +-- requirements/                                            # locked requirements and uv rules
|   +-- lint/                                                        # repository lint wrappers
|       +-- buildifier_wrapper.py                                   # Bazel formatting wrapper
|       +-- ruff_wrapper.py                                         # Python lint wrapper
|       +-- rumdl_wrapper.py                                        # Markdown lint wrapper
|       +-- shellcheck_wrapper.py                                   # shell lint wrapper
|
+-- Init/
|   +-- include/Init/
|   |   +-- Init.h                                                   # initialization API
|   |   +-- DevelopmentSignalHandler.h                              # signal-handler API
|   +-- lib/
|   |   +-- Init.cpp                                                 # initialization implementation
|   |   +-- DevelopmentSignalHandler.cpp                             # signal handling implementation
|   +-- unittests/
|       +-- DevelopmentSignalHandlerTests.cpp                       # signal-handler tests
|
+-- AsyncRT/
|   +-- include/AsyncRT/
|   |   +-- Runtime/                                                  # public asynchronous runtime API
|   |   |   +-- AsyncValue.h                                          # asynchronous value state
|   |   |   +-- AsyncValueRef.h                                       # reference to async value
|   |   |   +-- WorkQueue.h                                           # work queue API
|   |   |   +-- TimerHeap.h                                           # timer scheduling API
|   |   |   +-- CPUDevice.h                                           # CPU device abstraction
|   |   |   +-- Allocator.h                                           # runtime allocation API
|   |   +-- CompilerSupport/                                          # compiler-facing async support
|   |   |   +-- Context.h                                             # compiler/runtime context
|   |   |   +-- LLVMThreadPool.h                                      # LLVM thread-pool integration
|   |   +-- Support/                                                  # queues, locks, diagnostics, and helpers
|   |       +-- ConcurrentMPMCQueue.h                                 # concurrent queue
|   |       +-- Semaphore.h                                           # semaphore API
|   |       +-- Diagnostic.h                                          # async diagnostics
|   +-- lib/
|   |   +-- Runtime/                                                  # runtime implementation
|   |   |   +-- AsyncValue.cpp                                        # async value implementation
|   |   |   +-- WorkQueue implementations                             # worker queue implementations
|   |   |   +-- TimerHeap.cpp                                         # timer implementation
|   |   |   +-- CPUDevice.cpp                                         # CPU device implementation
|   |   +-- CompilerSupport/                                          # compiler support implementation
|   |   +-- JemallocPreload/                                          # allocator preload implementation
|   |   +-- Support/                                                  # support implementation
|   +-- tools/                                                        # crash/runtime diagnostic tools
|   |   +-- crash-report-path-info/                                   # crash report path tool
|   |   +-- crash-test-dummy/                                        # crash testing executable
|   +-- unittests/                                                    # allocator, queue, timer, and device tests
|
+-- Cache/
|   +-- include/Cache/
|   |   +-- BlobCache.h                                               # blob cache API
|   |   +-- CachedTransform.h                                         # cached transform API
|   |   +-- CacheTelemetryContext.h                                   # cache telemetry API
|   |   +-- Support/Keys.h                                            # cache key helpers
|   +-- lib/                                                          # cache implementations
|   |   +-- BlobCache.cpp                                              # blob cache implementation
|   |   +-- CachedTransform.cpp                                        # transform cache implementation
|   |   +-- CacheTelemetryContext.cpp                                  # cache telemetry implementation
|   +-- tools/cache-mgr/cache-mgr.cpp                                  # cache-manager command-line tool
|   +-- unittests/                                                     # cache behavior tests
|
+-- Config/
|   +-- include/Config/Version.h                                      # public version API
|   +-- include/GeneratedVersion.h.tmpl                                # generated version template
|   +-- lib/Version.cpp                                                # version implementation
|
+-- Support/
|   +-- include/Support/                                               # public shared support APIs
|   |   +-- ADT/                                                       # data structures and ownership helpers
|   |   +-- Compiler/                                                  # compiler/MLIR support types
|   |   +-- Diagnostics/                                               # diagnostic formatting/handling
|   |   +-- Driver/                                                    # command-line driver support
|   |   +-- Filesystem/                                                # filesystem paths and disk usage
|   |   +-- ML/                                                        # machine-learning data types/utilities
|   |   +-- Telemetry/                                                 # metrics and telemetry exporters
|   |   +-- Threading/                                                 # threading and synchronization helpers
|   |   +-- Error.h / ErrorOr.h                                        # error result types
|   |   +-- Buffer.h / Context.h                                      # shared buffer/context APIs
|   +-- lib/                                                           # implementations matching public headers
|   |   +-- ADT/                                                       # ADT implementations
|   |   +-- Compiler/                                                  # compiler support implementations
|   |   +-- Diagnostics/                                               # diagnostic implementations
|   |   +-- ML/                                                        # ML utility implementations
|   |   +-- Telemetry/                                                 # telemetry exporters/context
|   |   +-- Threading/                                                 # thread and signal helpers
|   |   +-- Log.cpp / Metrics.cpp                                      # logging and metrics
|   |   +-- Process.cpp / FileSystemExtras.cpp                         # process/filesystem support
|   +-- tools/                                                         # support-specific executables
|   |   +-- build-info/                                                # build information tool
|   |   +-- compare-timings/                                           # timing comparison tool
|   |   +-- driver-tblgen/                                             # driver table generator
|   |   +-- system-info/                                               # system information tool
|   +-- unittests/                                                     # unit tests grouped by support domain
|
+-- Mojo/
|   +-- stdlib/
|   |   +-- std/                                                       # Mojo standard modules
|   |   |   +-- algorithm/                                             # algorithms
|   |   |   +-- collections/                                           # collections
|   |   |   +-- math/                                                  # math APIs
|   |   |   +-- memory/                                                # memory APIs
|   |   |   +-- runtime/                                               # runtime APIs
|   |   |   +-- testing/                                               # testing APIs
|   |   |   +-- gpu/ and _gpu/                                         # GPU-facing standard modules
|   |   +-- test/                                                      # standard-library tests
|   |   +-- benchmarks/                                                # standard-library benchmarks
|   |   +-- scripts/                                                   # table/data generation scripts
|   +-- lib/                                                           # Mojo compiler and runtime libraries
|   |   +-- Compiler/                                                  # compiler implementation
|   |   +-- Elaborator/                                                # semantic elaboration
|   |   +-- ExecutionEngine/                                           # execution engine
|   |   +-- Interpreter/                                               # interpreter
|   |   +-- MojoParser/                                                # parser
|   |   +-- MojoTooling/                                               # tooling APIs
|   |   +-- CompilerRT/                                                # compiler runtime bridge
|   |   +-- *Dialect/                                                  # compiler dialect implementations
|   +-- tools/                                                         # compiler and language tools
|   |   +-- mojo/                                                       # Mojo driver/tool
|   |   +-- kgen/                                                       # kernel generation tools
|   |   +-- compilation-server/                                         # compilation server
|   |   +-- mojo-lsp-server/                                            # language-server implementation
|   |   +-- kgen-opt/ and kgen-translate/                               # KGEN tooling
|   +-- python/mojo/                                                    # Python interoperability package
|   +-- examples/                                                       # language/GPU/interop examples
|   +-- docs/                                                           # language and standard-library docs
|   +-- test/ and unittests/                                            # compiler and language tests
|
+-- tools/                                                             # repository helper tools
|   +-- build_defs/cc/link_hack.bzl                                     # C/C++ build helper
|   +-- bazel                                                           # repository Bazel helper
|
+-- utils/
    +-- local_transformers_setup/                                      # local model environment scripts
    |   +-- setup_local_transformers.sh                                 # setup
    |   +-- cleanup_local_transformers.sh                               # cleanup
    |   +-- README.md                                                    # instructions
    +-- setup-gpu-clock.sh                                               # GPU clock utility
|
|   # further nested implementation branches
+-- bazel/
|   +-- internal/
|   |   +-- cc-toolchain/
|   |   |   +-- args/                                               # compiler argument definitions
|   |   |   |   +-- interface-libraries/                            # interface-library arguments
|   |   |   |   +-- modular/                                        # Modular compiler arguments
|   |   |   +-- features/features.bzl                               # toolchain feature declarations
|   |   |   +-- tools/                                              # compiler/linker wrappers
|   |   |       +-- builtin_module_map.bzl                          # builtin module map
|   |   |       +-- linker-driver.sh                                # linker driver wrapper
|   |   |       +-- multi-platform-clang.sh                         # platform clang wrapper
|   |   +-- pip/pycross/
|   |       +-- dependency.py                                       # Python dependency model
|   |       +-- download.py                                         # package download logic
|   |       +-- generate.py                                          # dependency generation
|   |       +-- package.py                                           # Python package model
|   |       +-- render.py                                            # BUILD/file rendering
|   |       +-- test_*.py                                            # pycross unit tests
|
+-- AsyncRT/
|   +-- include/AsyncRT/Runtime/
|   |   +-- AsyncValue.h                                             # async value declaration
|   |   +-- AsyncValueRef.h                                          # async value reference
|   |   +-- WorkQueue.h                                              # work queue interface
|   |   +-- TimerHeap.h                                              # timer heap interface
|   |   +-- Globals/
|   |       +-- Globals.h                                            # global runtime declarations
|   |       +-- RuntimeGlobal.h                                      # runtime-global state
|   |       +-- VirtualDeviceGlobals.h                               # virtual-device state
|   +-- lib/Runtime/
|   |   +-- AsyncValue.cpp                                           # async value implementation
|   |   +-- ThreadPoolWorkQueue.cpp                                  # thread-pool queue
|   |   +-- SingleThreadWorkQueue.cpp                                # single-thread queue
|   |   +-- DelegateThreadPoolWorkQueue.cpp                          # delegated queue
|   |   +-- TimerHeap.cpp                                             # timer implementation
|   |   +-- Globals/
|   |       +-- Globals.cpp                                           # global runtime implementation
|   |       +-- RuntimeGlobal.cpp                                     # runtime-global implementation
|   +-- lib/Support/
|   |   +-- Semaphore.cpp                                             # semaphore implementation
|   |   +-- ThreadAffinity.cpp                                         # thread affinity support
|   |   +-- Location.cpp                                               # source-location support
|   +-- unittests/
|       +-- AsyncValueTest.cpp                                        # async value tests
|       +-- WorkQueueTest.cpp                                          # work queue tests
|       +-- TimerHeapTest.cpp                                          # timer tests
|
+-- Cache/
|   +-- include/Cache/
|   |   +-- Support/Keys.h                                             # cache-key implementation helpers
|   |   +-- BlobCache.h                                                # blob cache contract
|   +-- tools/cache-mgr/
|   |   +-- cache-mgr.cpp                                               # cache manager executable
|   |   +-- BUILD.bazel                                                 # cache manager target
|   +-- test/cache-mgr/
|   |   +-- Inputs/                                                      # test input fixtures
|   |   |   +-- empty.txt
|   |   |   +-- some_file.txt
|   |   +-- cache-mgr.test                                                 # cache manager test definition
|
+-- Config/
|   +-- include/Config/Version.h                                       # version query API
|   +-- lib/Version.cpp                                                 # version query implementation
|   +-- BUILD.bazel                                                     # Config build target
|
+-- Support/
|   +-- include/Support/Telemetry/
|   |   +-- Telemetry.h                                                  # telemetry API
|   |   +-- Logs.h                                                       # telemetry logging API
|   |   +-- Instruments.h                                                 # instrumentation API
|   |   +-- Exporters/
|   |       +-- FileLogExporter.h                                         # log exporter API
|   |       +-- FileMetricExporter.h                                      # metric exporter API
|   +-- lib/Telemetry/
|   |   +-- TelemetryContext.cpp                                          # telemetry context implementation
|   |   +-- FileLogExporter.cpp                                            # file log exporter
|   |   +-- FileMetricExporter.cpp                                         # file metric exporter
|   +-- include/Support/Compiler/
|   |   +-- Bytecode.h                                                     # compiler bytecode API
|   |   +-- DiagnosticHandler.h                                            # diagnostic handler API
|   |   +-- ErrorTree.h                                                    # compiler error tree
|   |   +-- MLIRToString.h                                                 # MLIR conversion helpers
|   +-- lib/Compiler/
|   |   +-- BytecodeReaderWriter.cpp                                       # bytecode serialization
|   |   +-- DiagnosticHandler.cpp                                          # diagnostic implementation
|   |   +-- ErrorTree.cpp                                                   # error tree implementation
|   +-- unittests/Log/
|       +-- LogTest.cpp                                                    # basic logging tests
|       +-- LogJSONOutputTest.cpp                                          # JSON logging tests
|       +-- RequestLogTest.cpp                                             # request-log tests
|
+-- Mojo/
|   +-- stdlib/std/collections/
|   |   +-- array.mojo                                                    # array collection
|   |   +-- dict.mojo                                                     # dictionary collection
|   |   +-- list.mojo                                                     # list collection
|   |   +-- set.mojo                                                      # set collection
|   |   +-- string/
|   |       +-- string.mojo                                                # string type
|   |       +-- format.mojo                                                # string formatting
|   |       +-- _utf8.mojo                                                 # UTF-8 support
|   +-- std/runtime/
|   |   +-- _asyncrt.mojo                                                  # AsyncRT standard-library bridge
|   +-- lib/Compiler/
|   |   +-- KGENCompiler.cpp                                                # KGEN compiler integration
|   |   +-- ObjectCompiler/                                                 # object-code compiler layer
|   |   |   +-- LLVMAccessorHelper.cpp                                     # LLVM accessor support
|   |   |   +-- MCLinker.cpp                                                # machine-code linker support
|   |   +-- Pipeline/
|   |       +-- Pipeline.cpp                                                # compiler pipeline implementation
|   |       +-- Pipeline.h                                                  # compiler pipeline API
|   +-- lib/ExecutionEngine/
|   |   +-- ExecutionEngine.cpp                                             # execution engine
|   |   +-- JIT/
|   |       +-- MaterializationLayer.cpp                                    # JIT materialization
|   |       +-- StaticArchiveLayer.cpp                                       # static archive layer
|   +-- python/mojo/
|       +-- importer.py                                                      # Python-to-Mojo import support
|       +-- run.py                                                           # Python-facing Mojo runner
|       +-- notebook.py                                                       # notebook integration
|
+-- Mojo/tools/mojo/
    +-- mojo.cpp                                                             # Mojo tool entry point
    +-- Build/mojo-build.cpp                                                  # build subcommand
    +-- Debug/mojo-debug.cpp                                                  # debug subcommand
    +-- Demangle/mojo-demangle.cpp                                            # demangle subcommand
    +-- Doc/mojo-doc.cpp                                                      # documentation subcommand
    +-- Format/mojo-format.cpp                                                # formatter subcommand
    +-- Precompile/mojo-precompile.cpp                                        # precompile subcommand
    +-- REPL/mojo-repl.cpp                                                    # REPL subcommand
    +-- Run/mojo-run.cpp                                                      # run subcommand
|
|   # additional nested implementation branches
+-- bazel/
|   +-- internal/cc-toolchain/
|   |   +-- args/
|   |   |   +-- BUILD.bazel                                           # compiler-argument targets
|   |   |   +-- interface-libraries/                                  # interface-library argument set
|   |   |   +-- modular/                                              # Modular argument set
|   |   +-- features/
|   |   |   +-- features.bzl                                           # declared toolchain features
|   |   +-- tools/
|   |       +-- builtin_module_map.bzl                                 # builtin module mapping
|   |       +-- linker-driver.sh                                       # linker invocation wrapper
|   |       +-- multi-platform-clang.sh                                # platform-aware clang wrapper
|   +-- internal/llvm-lit/
|       +-- lit_shim.py                                                # test-runner entry point
|       +-- modular_test_format.py                                     # Modular test discovery/format
|       +-- lit.common.cfg.py                                          # common lit configuration
|       +-- validate_lit_features.py                                   # lit feature validation
|
+-- AsyncRT/
|   +-- include/AsyncRT/Support/
|   |   +-- ConcurrentQueue.h                                           # concurrent queue interface
|   |   +-- ConcurrentMPMCQueue.h                                       # multi-producer/multi-consumer queue
|   |   +-- LockFreeRingBuffer.h                                        # lock-free ring buffer
|   |   +-- Semaphore.h                                                 # semaphore interface
|   |   +-- ThreadAffinity.h                                            # thread placement interface
|   |   +-- Diagnostic.h                                                 # runtime diagnostic interface
|   +-- include/AsyncRT/CompilerSupport/
|   |   +-- Context.h                                                    # compiler/runtime context
|   |   +-- LLVMThreadPool.h                                             # LLVM thread-pool bridge
|   |   +-- MLIRLocationDecoder.h                                        # MLIR location decoding
|   +-- lib/Runtime/Globals/
|   |   +-- Globals.cpp                                                   # runtime global state
|   |   +-- RuntimeGlobal.cpp                                             # runtime-global lifecycle
|   |   +-- VirtualDeviceGlobals.cpp                                     # virtual-device globals
|   +-- lib/Runtime/
|       +-- MallocAllocator.cpp                                          # malloc allocator
|       +-- TCMallocAllocator.cpp                                        # TCMalloc allocator
|       +-- ThreadPoolWorkQueue.cpp                                      # thread-pool work queue
|       +-- SingleThreadWorkQueue.cpp                                    # single-thread work queue
|
+-- Cache/
|   +-- include/Cache/Support/
|   |   +-- Keys.h                                                        # key construction/helpers
|   +-- lib/
|   |   +-- BlobCache.cpp                                                 # blob cache storage behavior
|   |   +-- CachedTransform.cpp                                           # transform result caching
|   |   +-- CacheTelemetryContext.cpp                                     # cache metrics/context
|   +-- test/cache-mgr/Inputs/
|       +-- empty.txt                                                     # empty-input fixture
|       +-- some_file.txt                                                 # normal-input fixture
|       +-- some_file_windows.txt                                         # Windows-path fixture
|
+-- Support/
|   +-- include/Support/Threading/
|   |   +-- Atomics.h                                                      # atomic helpers
|   |   +-- HWInfo.h                                                       # hardware/threading information
|   |   +-- Shared.h                                                       # shared threading utilities
|   |   +-- SpinWaiter.h                                                   # spin-wait implementation API
|   |   +-- ThreadLocalCache.h                                             # thread-local cache API
|   +-- include/Support/ML/
|   |   +-- DType.h                                                        # data-type definitions
|   |   +-- TensorBase.h                                                    # tensor base abstraction
|   |   +-- TensorShape.h                                                   # tensor shape
|   |   +-- TensorSpec.h                                                    # tensor specification
|   |   +-- RangeUtils.h                                                    # range utilities
|   +-- include/Support/Diagnostics/
|   |   +-- FormatScopedDiagnosticHandler.h                                # scoped diagnostic formatting
|   +-- lib/Telemetry/
|       +-- TelemetryContext.cpp                                           # telemetry state/context
|       +-- FileLogExporter.cpp                                             # log-file exporter
|       +-- FileMetricExporter.cpp                                          # metric-file exporter
|
+-- Config/
|   +-- include/Config/Version.h                                            # public version declaration
|   +-- lib/Version.cpp                                                      # version implementation
|   +-- BUILD.bazel                                                          # build target definition
|
+-- Mojo/
|   +-- stdlib/std/algorithm/
|   |   +-- functional.mojo                                                  # functional algorithms
|   |   +-- backend/
|   |       +-- tile.mojo                                                     # tile backend
|   |       +-- unswitch.mojo                                                 # unswitch backend
|   |       +-- vectorize.mojo                                                 # vectorization backend
|   +-- stdlib/std/collections/string/
|   |   +-- string.mojo                                                       # string implementation
|   |   +-- format.mojo                                                       # string formatting
|   |   +-- _utf8.mojo                                                        # UTF-8 operations
|   |   +-- codepoint.mojo                                                    # Unicode code points
|   +-- stdlib/std/testing/
|   |   +-- testing.mojo                                                      # testing primitives
|   |   +-- suite.mojo                                                        # test-suite support
|   |   +-- assert_aborts.mojo                                                # expected-abort assertions
|   |   +-- prop/
|   |       +-- random.mojo                                                    # property-test random values
|   |       +-- runner.mojo                                                    # property-test runner
|   +-- lib/MojoParser/
|   |   +-- Lexer.cpp                                                          # lexical analysis
|   |   +-- ParserBase.cpp                                                     # parser base implementation
|   |   +-- ParserExprs.cpp                                                    # expression parsing
|   |   +-- ParserStmts.cpp                                                    # statement parsing
|   |   +-- IREmitter.cpp                                                      # parser-to-IR emission
|   +-- lib/MojoTooling/
|   |   +-- CodeComplete.cpp                                                    # completion support
|   |   +-- DocGen.cpp                                                         # documentation generation
|   |   +-- ParserDriver.cpp                                                    # parser driver API
|   |   +-- TypeExtractionUtils.cpp                                            # type extraction helpers
|   +-- lib/ToolCommon/
|   |   +-- CLOptions.cpp                                                       # common command-line options
|   |   +-- PassRegistry.cpp                                                    # pass registry
|   |   +-- PipelineTiming.cpp                                                  # pipeline timing
|   |   +-- InitAllDialects/                                                    # dialect initialization
|   |   +-- TranslationRegistry/                                                # translation registration
|   +-- tools/mojo/
|       +-- Build/
|       |   +-- mojo-build.cpp                                                  # build command
|       +-- Run/
|       |   +-- mojo-run.cpp                                                    # run command
|       +-- REPL/
|       |   +-- mojo-repl.cpp                                                    # REPL command
|       +-- Format/
|       |   +-- mojo-format.cpp                                                  # formatter command
|       +-- Doc/
|           +-- mojo-doc.cpp                                                     # documentation command
|
|   # deepest concrete implementation branches
+-- bazel/
|   +-- internal/cc-toolchain/
|       +-- args/
|       |   +-- interface-libraries/
|       |   |   +-- BUILD.bazel                                      # interface-library target definitions
|       |   +-- modular/
|       |       +-- BUILD.bazel                                      # Modular compiler-argument targets
|       +-- features/
|       |   +-- features.bzl                                         # feature declarations and transitions
|       +-- tools/
|           +-- builtin_module_map.bzl                               # builtin module map generation
|           +-- linker-driver.sh                                     # linker command adaptation
|           +-- multi-platform-clang.sh                              # platform-specific compiler selection
|
+-- AsyncRT/
|   +-- include/AsyncRT/Runtime/Globals/
|   |   +-- Globals.h                                                  # global declarations
|   |   +-- RuntimeGlobal.h                                            # runtime-global API
|   |   +-- VirtualDeviceGlobals.h                                     # virtual-device global API
|   +-- lib/Runtime/Globals/
|       +-- Globals.cpp                                                # global definitions
|       +-- RuntimeGlobal.cpp                                          # runtime-global implementation
|       +-- VirtualDeviceGlobals.cpp                                   # virtual-device implementation
|
+-- Cache/
|   +-- test/cache-mgr/
|       +-- Inputs/
|       |   +-- empty.txt                                               # empty input case
|       |   +-- some_file.txt                                           # ordinary input case
|       |   +-- some_file_windows.txt                                   # Windows-path input case
|       +-- cache-mgr.test                                               # test command/expectation file
|
+-- Support/
|   +-- include/Support/DebugInfoDialect/
|   |   +-- IR/
|   |   |   +-- DebugInfoDialect.h                                      # debug-info dialect declaration
|   |   |   +-- DebugInfoOps.h                                          # debug-info operations
|   |   |   +-- DebugInfoTypes.h                                        # debug-info types
|   |   |   +-- DIBuilder.h                                             # debug-info builder
|   |   |   +-- DebugInfo.td                                             # tablegen definitions
|   |   +-- DebugInfoToLLVM/
|   |   |   +-- DebugInfoToLLVM.h                                       # conversion-to-LLVM API
|   |   +-- Transforms/
|   |       +-- Conversion.h                                            # conversion pass declarations
|   |       +-- StripDebugInfo.h                                         # debug-info stripping pass
|   +-- include/Support/Telemetry/Exporters/
|   |   +-- FileLogExporter.h                                           # file log exporter API
|   |   +-- FileMetricExporter.h                                        # file metric exporter API
|   +-- lib/Telemetry/
|       +-- TelemetryContext.cpp                                        # telemetry context implementation
|       +-- FileLogExporter.cpp                                          # file log exporter implementation
|       +-- FileMetricExporter.cpp                                       # file metric exporter implementation
|
+-- Mojo/
|   +-- stdlib/std/collections/string/
|   |   +-- _parsing_numbers/
|   |   |   +-- constants.mojo                                          # numeric parsing constants
|   |   |   +-- parsing_floats.mojo                                     # floating-point parsing
|   |   |   +-- parsing_integers.mojo                                   # integer parsing
|   |   +-- string.mojo                                                  # string type and operations
|   |   +-- string_span.mojo                                             # non-owning string span
|   |   +-- format.mojo                                                  # formatting operations
|   |   +-- iterators.mojo                                               # string iterators
|   |   +-- codepoint.mojo                                               # Unicode code-point operations
|   |   +-- _utf8.mojo                                                   # UTF-8 internals
|   +-- stdlib/std/testing/prop/
|   |   +-- strategy/
|   |   |   +-- list_strategy.mojo                                       # list property strategy
|   |   |   +-- simd_strategy.mojo                                       # SIMD property strategy
|   |   |   +-- string_strategy.mojo                                     # string property strategy
|   |   +-- random.mojo                                                  # property-test random generation
|   |   +-- runner.mojo                                                  # property-test execution
|   +-- stdlib/std/algorithm/backend/
|   |   +-- cpu/
|   |   |   +-- map.mojo                                                  # CPU map implementation
|   |   +-- tile.mojo                                                     # tile backend
|   |   +-- unswitch.mojo                                                 # unswitch backend
|   |   +-- vectorize.mojo                                                # vectorization backend
|   +-- lib/Compiler/ObjectCompiler/
|   |   +-- LLVM/
|   |   |   +-- Bitcode/                                                 # LLVM bitcode compatibility layers
|   |   |   |   +-- 17/                                                    # LLVM 17 support
|   |   |   |   +-- 19/                                                    # LLVM 19 support
|   |   |   |   +-- 21/                                                    # LLVM 21 support
|   |   |   +-- Transforms/
|   |   |       +-- LLVMIRDowngradePass.cpp                               # LLVM IR downgrade pass
|   |   |       +-- PointerRewriter.cpp                                   # pointer rewriting pass
|   |   |       +-- SetFunctionAttributes.cpp                             # function attribute pass
|   |   +-- Target/Host/
|   |   |   +-- HostBackend.cpp                                            # host backend implementation
|   |   |   +-- HostBackend.h                                              # host backend API
|   |   +-- MCLinker.cpp                                                    # machine-code linker support
|   +-- lib/ToolCommon/
|       +-- InitAllDialects/
|       |   +-- InitAllDialects.cpp                                        # dialect registration
|       |   +-- IndexInterpreterInterface.cpp                             # interpreter interface registration
|       +-- TranslationRegistry/
|           +-- TranslationRegistry.cpp                                   # translation registration
|
+-- Mojo/tools/mojo/
    +-- Format/
    |   +-- FormatDescription.td                                           # formatter command description
    |   +-- FormatOptions.td                                                # formatter options
    |   +-- mojo-format.cpp                                                  # formatter implementation
    +-- Doc/
    |   +-- DocDescription.td                                                # documentation command description
    |   +-- DocOptions.td                                                     # documentation options
    |   +-- mojo-doc.cpp                                                      # documentation implementation
    +-- Build/
    |   +-- BuildDescription.td                                              # build command description
    |   +-- BuildOptions.td                                                   # build options
    |   +-- mojo-build.cpp                                                     # build implementation
    +-- Run/
        +-- RunDescription.td                                                 # run command description
        +-- RunOptions.td                                                      # run options
        +-- mojo-run.cpp                                                       # run implementation
|
+-- murali_docs/                                                # documentation created for this study
    +-- artifacts/source_code_walkthrough/                       # walkthrough documentation
        +-- AGENTS.md                                            # local documentation rules
        +-- 0000_source_code_walkthrough.md                      # main index
        +-- 0001_parse_openai_request_body.md                     # HTTP body -> typed request
        +-- 0002_get_pipeline.md                                  # model name -> pipeline
        +-- 0003_openai_parse_chat_completion_request.md           # chat normalization
        +-- 0004_TextGenerationRequest.md                          # API data -> internal request
        +-- 0005_streaming_vs_non_streaming.md                    # response mode selection
        +-- 0006_tokenizer_new_context.md                          # request -> TextContext
        +-- 0007_create_buffered_detokenizer.md                    # tokens -> text fragments
        +-- 0008_note_awaiting_admission.md                        # admission metric lifecycle
        +-- 0009_model_worker_stream.md                            # context -> worker stream
        +-- 0010_scheduler_iteration.md                            # queue -> scheduler iteration
        +-- 0011_text_batch_constructor.md                         # contexts -> batch
        +-- 0012_pipeline_execution.md                             # batch -> sampled tokens
        +-- 0013_llama_input_staging.md                            # contexts -> Llama3Inputs
        +-- 0014_llama_model_execution.md                          # inputs -> compiled model
        +-- 0015_folder_structure.md                               # this folder guide
|
|   # end of the single root-level folder tree
```

-->

Relationship to the MAX Serve path

max/python/max/serve
  -> uses bazel/ and bazelw for build/test orchestration
  -> uses MAX engine/NN/kernel/Mojo code under max/
  -> may use shared Cache/, Config/, Support/, Init/, and AsyncRT/ libraries
  -> is developed with tools/, utils/, and repository-wide docs/configuration

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
