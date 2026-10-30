# Llama / MAX / GPU Masterclass Course Map

Revision: `e9818b79ea73b498ec4b438a24ecd8d853307d0d`

Total: **586 PowerPoint slides**, **142 interactive source scenes**, **71 source units**, and **41 unique source files**.

Each source unit becomes a contract scene and implementation scene. Each scene becomes four progressive PowerPoint builds with three real On Click groups and one five-state HTML laboratory. Evidence is labeled `SOURCE`, `BOUNDARY`, `COMPILE-ONLY`, or `GPU-LAB`. Qualcomm is documented only as a boundary; this course does not claim a supported MAX Qualcomm GPU backend.

## Tree

- Part 1: Graph Construction and Finalization (11 units, 22 scenes, 91 slides)
    01. Graph object construction
        File: `max/python/max/graph/graph.py`
        Function/anchor: `class Graph:`
        Role: Creates the MLIR module, graph operation, entry block, symbolic arguments, chain state, and kernel library for one trace.
        Evidence: `SOURCE`
        Math: `G = (V, E, types, devices)`
    02. Graph constructor body
        File: `max/python/max/graph/graph.py`
        Function/anchor: `def __init__(         self,`
        Role: Turns declared input types into symbolic Values while preserving names and device placement.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    03. Kernel library loading
        File: `max/python/max/graph/graph.py`
        Function/anchor: `def load_paths(self, custom_extensions`
        Role: Loads built-in and custom Mojo extension paths, then seeds graph compiler declarations.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    04. Per-operation verification
        File: `max/python/max/graph/graph.py`
        Function/anchor: `def _verify_op(self, op`
        Role: Verifies each newly emitted operation and converts MLIR diagnostics into a Python GraphError close to the source call.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    05. Device metadata population
        File: `max/python/max/graph/graph.py`
        Function/anchor: `def _populate_device_info_mapping`
        Role: Collects graph device references and writes concrete target metadata used during compilation.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    06. Graph.output finalization
        File: `max/python/max/graph/graph.py`
        Function/anchor: `def output(self, *outputs`
        Role: Flattens output trees, fixes the function signature, emits mo.output, records names, and verifies the complete graph.
        Evidence: `SOURCE`
        Math: `f : (T_0,...,T_n) -> (R_0,...,R_m)`
    07. Llama config initialization
        File: `max/python/max/pipelines/architectures/llama3/model_config.py`
        Function/anchor: `def initialize(`
        Role: Maps Hugging Face configuration, devices, dtype, RoPE, attention, and cache policy into a Llama3Config.
        Evidence: `SOURCE`
        Math: `d_h = d_model / n_heads`
    08. Llama config finalization
        File: `max/python/max/pipelines/architectures/llama3/model_config.py`
        Function/anchor: `def finalize(`
        Role: Resolves deferred model dimensions and validates weight-dependent invariants before graph creation.
        Evidence: `SOURCE`
        Math: `P ~= Vd + L(4d^2 + 3d d_ff)`
    09. Llama symbolic trace
        File: `max/python/max/pipelines/architectures/llama3/llama3.py`
        Function/anchor: `def trace(`
        Role: Builds symbolic input leaves, invokes the Llama module with graph Values, and hands returned Values to Graph.output().
        Evidence: `SOURCE`
        Math: `logits = LMHead(Transformer(tokens, KV))`
    10. GraphSpec trace recipe
        File: `max/python/max/pipelines/lib/graph_specs.py`
        Function/anchor: `def trace(self, module`
        Role: Packages module, graph name, input tree, weights, extensions, and graph grouping into a repeatable trace operation.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    11. Compile and initialize graphs
        File: `max/python/max/pipelines/lib/graph_specs.py`
        Function/anchor: `def compile_and_init_graphs(`
        Role: Traces graph specs, compiles the finalized modules, merges weight registries, and initializes executable Models.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`

- Part 2: Graph Compiler Transformations (11 units, 22 scenes, 91 slides)
    01. Top-level graph compiler stages
        File: `GraphCompiler/lib/GraphCompiler.cpp`
        Function/anchor: `static void gcPipelineImpl`
        Role: Builds named pass stages and preserves dump, cache, timing, and pre-JIT behavior around the lowering pipeline.
        Evidence: `SOURCE`
        Math: `IR_(k+1) = Pass_k(IR_k)`
    02. MO preprocessing pipeline
        File: `GraphCompiler/lib/GraphCompiler.cpp`
        Function/anchor: `void M::MO::buildMoPipeline`
        Role: Runs graph-level shape, layout, constant, and canonical simplifications before kernel-oriented lowering.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    03. MAP optimization pipeline
        File: `GraphCompiler/lib/GraphCompiler.cpp`
        Function/anchor: `void M::MO::buildMapPipeline`
        Role: Orders kernel fusion, structured-body outlining, buffer allocation, and conversion toward generated routines.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    04. MOGG routine pipeline
        File: `GraphCompiler/lib/GraphCompiler.cpp`
        Function/anchor: `void M::MO::buildMoggPipeline`
        Role: Outlines executable routines and lowers graph-generation abstractions toward primitive device programs.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    05. MGP runtime pipeline
        File: `GraphCompiler/lib/GraphCompiler.cpp`
        Function/anchor: `void M::MO::buildMgpPipeline`
        Role: Performs buffer reuse, memory planning, dependency lowering, asynchronous routine shaping, and executable finalization.
        Evidence: `SOURCE`
        Math: `peak = max_t sum_(b live at t) bytes(b)`
    06. MO to MAP conversion
        File: `GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Applies legality rules and rewrite patterns that replace high-level model operations with MAP kernel abstractions.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    07. MAP kernel fusion
        File: `GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Combines compatible producer and consumer bodies to reduce intermediate materialization and launch boundaries.
        Evidence: `SOURCE`
        Math: `bytes_saved ~= read(tmp) + write(tmp)`
    08. Reduction fusion
        File: `GraphCompiler/lib/MAPDialect/Transforms/ReductionFuser.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Fuses elementwise producers and consumers around rowwise reductions while preserving reshape and ordering barriers.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    09. MAP to MOGG conversion
        File: `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Turns kernel-oriented MAP operations into generated graph routines with explicit callable structure.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    10. MOGG to MGP conversion
        File: `GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp`
        Function/anchor: `MOGGToMGPPass::runOnOperation`
        Role: Lowers generated routines into runtime primitives, explicit buffers, device work, and dependency-bearing operations.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    11. MEF serialization boundary
        File: `GraphCompiler/lib/GraphCompiler.cpp`
        Function/anchor: `convertMGPToMEF`
        Role: Converts one finalized MGP top-level graph into the runtime-loadable MEF artifact consumed by GraphRT.
        Evidence: `SOURCE + BOUNDARY`
        Math: `MEF = serialize(finalized MGP)`

- Part 3: Kernel Development and Optimization (13 units, 26 scenes, 107 slides)
    01. Kernel extensibility registration
        File: `max/kernels/src/graph_compiler/extensibility/decorators.mojo`
        Function/anchor: `def register(`
        Role: Registers a Mojo implementation against operation name, device type, API, architecture, and model specificity.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    02. GPU matmul registration
        File: `max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo`
        Function/anchor: `@extensibility.register("mo.matmul", type="gpu")`
        Role: Matches mo.matmul on GPU and dispatches shape, dtype, layout, and target information into the matmul implementation family.
        Evidence: `SOURCE`
        Math: `C_(M,N) = A_(M,K) B_(K,N)`
    03. Generic GPU matmul kernel
        File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
        Function/anchor: `def matmul_kernel[`
        Role: Maps output tiles to GPU workgroups and composes loading, MMA accumulation, and epilogue behavior.
        Evidence: `SOURCE`
        Math: `FLOPs = 2 M N K`
    04. Naive matmul baseline
        File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
        Function/anchor: `def matmul_kernel_naive[`
        Role: Provides a correctness-oriented baseline whose redundant global traffic motivates tiling and data reuse.
        Evidence: `SOURCE`
        Math: `AI_naive ~= 2K / (2K b + b)`
    05. Coalesced global-memory access
        File: `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
        Function/anchor: `def coalescing_matrix_multiplication[`
        Role: Remaps adjacent threads onto adjacent memory locations so a warp or wave can combine scalar requests into fewer global-memory transactions.
        Evidence: `SOURCE + GPU-LAB`
        Math: `transactions ~= requested_bytes / transaction_width`
    06. Asynchronous DRAM to SRAM copy
        File: `max/kernels/src/layout/tile_io.mojo`
        Function/anchor: `def copy_dram_to_sram_async[`
        Role: Stages tiles from device memory to shared memory asynchronously so transfer can overlap independent computation.
        Evidence: `SOURCE + GPU-LAB`
        Math: `T_stage ~= max(T_copy, T_compute) when overlap holds`
    07. Ldmatrix asynchronous path
        File: `max/kernels/src/layout/tile_async_copy.mojo`
        Function/anchor: `def copy_dram_to_sram_async_ldmatrix[`
        Role: Uses target-aware tiled transfers and matrix-load-friendly shared layouts for tensor-core consumption.
        Evidence: `SOURCE + GPU-LAB`
        Math: `state_(n+1) = transform_n(state_n)`
    08. Register and vector tiling
        File: `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
        Function/anchor: `def block_tiled_vectorized_matrix_multiplication[`
        Role: Combines two-dimensional block tiling with vectorized movement and register accumulators to reduce instruction count and repeated memory traffic.
        Evidence: `SOURCE + GPU-LAB`
        Math: `AI_tiled grows with reuse while occupancy falls with registers/thread`
    09. Persistent SM90 matmul
        File: `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernel_persistent.mojo`
        Function/anchor: `def run_persistent[`
        Role: Keeps workgroups resident and pulls multiple output tiles through a scheduler to amortize launch and setup costs.
        Evidence: `SOURCE + GPU-LAB`
        Math: `tiles_per_CTA > 1`
    10. AMD ping-pong pipeline
        File: `max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo`
        Function/anchor: `def _run_impl[`
        Role: Alternates producer and consumer phases across staged buffers to hide global-memory latency behind MFMA work.
        Evidence: `SOURCE + GPU-LAB`
        Math: `state_(n+1) = transform_n(state_n)`
    11. Flash-attention dispatch
        File: `max/kernels/src/graph_compiler/builtin_kernels/attention.mojo`
        Function/anchor: `def _dispatch_flash_attention[`
        Role: Selects a fused attention kernel using dtype, head dimensions, masking, cache layout, and target capability.
        Evidence: `SOURCE`
        Math: `softmax(QK^T/sqrt(d_h))V`
    12. Paged KV cache store kernel
        File: `max/kernels/src/graph_compiler/builtin_kernels/kv_cache.mojo`
        Function/anchor: `@extensibility.register("mo.kv_cache.store.paged.ragged")`
        Role: Stores ragged K/V rows into page-indexed cache blocks using request row offsets and cache metadata.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    13. Arithmetic intensity and roofline
        File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
        Function/anchor: `def _matmul_gpu_impl[`
        Role: Uses matrix shape and target properties to select an implementation; evaluate each candidate against traffic, reuse, and resource pressure.
        Evidence: `SOURCE + GPU-LAB`
        Math: `Perf <= min(PeakFLOPs, AI * Bandwidth); AI = FLOPs / bytes`

- Part 4: Driver, Offload, Memory, and Backends (14 units, 28 scenes, 115 slides)
    01. Compile an offloaded Mojo function
        File: `max/driver/src/max_license/_hal/context.mojo`
        Function/anchor: `def _compile_inner[`
        Role: Invokes kgen.compile_offload for the selected device context and turns a Mojo function plus compile-time parameters into a loadable bundle.
        Evidence: `SOURCE + COMPILE-ONLY`
        Math: `state_(n+1) = transform_n(state_n)`
    02. Load a compiled bundle
        File: `max/driver/src/max_license/_hal/context.mojo`
        Function/anchor: `def load_bundle[`
        Role: Loads device code into the driver context and records module ownership for later symbol resolution.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    03. Resolve a device function
        File: `max/driver/src/max_license/_hal/context.mojo`
        Function/anchor: `def load_function(`
        Role: Looks up a compiled symbol, checks the cache, and wraps its native handle as a launchable Function.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    04. Enqueue a kernel function
        File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
        Function/anchor: `def enqueue_function[`
        Role: Packs DevicePassable arguments and launch dimensions, then submits the function to the stream without forcing immediate host synchronization.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    05. Allocate a device buffer
        File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
        Function/anchor: `def enqueue_create_buffer[`
        Role: Requests device storage through the context and returns an owning buffer with explicit element count and byte size.
        Evidence: `SOURCE + BOUNDARY`
        Math: `bytes = count * sizeof(dtype)`
    06. Host to device copy
        File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
        Function/anchor: `def enqueue_copy[`
        Role: Selects a copy overload from host or pinned storage to device storage and appends it to the ordered stream.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    07. Queue synchronization
        File: `max/driver/src/max_license/_hal/queue.mojo`
        Function/anchor: `def synchronize(self)`
        Role: Waits for previously submitted queue work and converts native driver errors into HAL errors at an explicit synchronization boundary.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    08. CUDA HAL plugin
        File: `max/driver/src/max_license/_hal/_plugin/cuda/cuda_plugin.mojo`
        Function/anchor: `struct CudaHalPlugin`
        Role: Binds the common MAX HAL interfaces to CUDA-specific device, context, stream, function, event, and memory implementations.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    09. HIP HAL plugin
        File: `max/driver/src/max_license/_hal/_plugin/hip/hip_plugin.mojo`
        Function/anchor: `struct HipHalPlugin`
        Role: Binds the common MAX HAL interfaces to HIP-specific device, context, stream, function, event, and memory implementations.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    10. Metal HAL plugin
        File: `max/driver/src/max_license/_hal/_plugin/metal/metal_plugin.mojo`
        Function/anchor: `struct MetalHalPlugin`
        Role: Binds the common MAX HAL interfaces to Metal-specific device, command queue, function, event, and memory implementations on Apple GPUs.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    11. NVIDIA NVPTX target traits
        File: `Mojo/lib/Target/max_license/NVPTX/NVPTXTraits.cpp`
        Function/anchor: `NVPTX`
        Role: Defines NVIDIA target capabilities and code-object handling used when Mojo offload targets PTX-compatible devices.
        Evidence: `SOURCE + COMPILE-ONLY`
        Math: `state_(n+1) = transform_n(state_n)`
    12. AMD GPU target traits
        File: `Mojo/lib/Target/max_license/AMDGPU/AMDGPUTraits.cpp`
        Function/anchor: `AMDGPU`
        Role: Defines AMD target capabilities and amdgcn code-object handling used by the HIP/ROCm path.
        Evidence: `SOURCE + COMPILE-ONLY`
        Math: `state_(n+1) = transform_n(state_n)`
    13. AMD target cost model
        File: `max/kernels/src/linalg/matmul/gpu/amd/amd_target.mojo`
        Function/anchor: `MI355`
        Role: Maps AMD GPU profiles to wave, memory, and instruction characteristics consumed by dispatch and tuning logic.
        Evidence: `SOURCE + GPU-LAB`
        Math: `state_(n+1) = transform_n(state_n)`
    14. Qualcomm and Hexagon boundary
        File: `MLRT/include/MLRT/Driver/DeviceContext/DeviceContext.h`
        Function/anchor: `Hexagon`
        Role: Documents that some backends do not expose a CUDA-style stream model; this tree does not expose an Adreno GPU kernel backend equivalent to CUDA or HIP.
        Evidence: `SOURCE + BOUNDARY`
        Math: `backend support != inferred GPU implementation`

- Part 5: MAX Serve Scheduling and Contexts (11 units, 22 scenes, 91 slides)
    01. Scheduler iteration
        File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
        Function/anchor: `def run_iteration(self)`
        Role: Runs one scheduling cycle: retrieve work, construct a batch, execute the pipeline, update contexts, and report progress.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    02. Retrieve pending requests
        File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
        Function/anchor: `def _retrieve_pending_requests`
        Role: Moves newly arrived requests into scheduler-owned structures while preserving cancellation and capacity behavior.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    03. Schedule and execute
        File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
        Function/anchor: `def _schedule(`
        Role: Coordinates batch construction with model execution and handles empty, blocked, and completed scheduling outcomes.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    04. Admit a text request
        File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
        Function/anchor: `def _admit_request`
        Role: Checks replica and budget constraints before moving a TextContext from pending state into managed scheduling state.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    05. Bind request resources
        File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
        Function/anchor: `def _bind_request`
        Role: Claims KV/cache resources and associates request state with a replica before it may enter CE or TG batches.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    06. Add context-encoding work
        File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
        Function/anchor: `def _add_ce_requests`
        Role: Selects prefill requests and token chunks subject to context budgets, cache availability, and fairness policy.
        Evidence: `SOURCE`
        Math: `sum CE_chunk_i <= CE_budget`
    07. Plan chunked prefill
        File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
        Function/anchor: `def _plan_ce_step`
        Role: Computes how much context each request can encode this iteration without violating total and active token limits.
        Evidence: `SOURCE`
        Math: `chunk_i = min(remaining_i, per_req_cap, budget_left)`
    08. Add token-generation work
        File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
        Function/anchor: `def _add_tg_requests`
        Role: Adds decode requests, generally one active token per context, while respecting cache and replica constraints.
        Evidence: `SOURCE`
        Math: `TG_tokens ~= number of active decode contexts`
    09. Construct a replica batch
        File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
        Function/anchor: `def _construct_replica_batch`
        Role: Combines planned CE and TG rows into one replica-local batch and materializes model-facing metadata.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    10. Update TextContext
        File: `max/python/max/pipelines/context/context.py`
        Function/anchor: `def update(         self,`
        Role: Appends generated tokens, advances active ranges, updates log probabilities, and evaluates stopping conditions.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    11. Paged KV claim and release
        File: `max/python/max/pipelines/kv_cache/paged_kv_cache/cache_manager.py`
        Function/anchor: `def claim(self, ctx`
        Role: Claims page-backed KV capacity for a context and releases ownership when the request completes or is evicted.
        Evidence: `SOURCE`
        Math: `KV_bytes ~= 2 L n_kv d_h tokens bytes_per_elem`

- Part 6: Source Atlas, Debugging, and Invariants (11 units, 22 scenes, 91 slides)
    01. CompiledModel versus Model
        File: `max/python/max/engine/api.py`
        Function/anchor: `class CompiledModel`
        Role: Separates reusable compiled artifact state from initialized executable state with bound persistent weights and runtime resources.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    02. Asynchronous compilation
        File: `max/python/max/engine/api.py`
        Function/anchor: `def compile_async(`
        Role: Passes finalized Graph modules and compilation configuration to the native engine and returns asynchronous compiled handles.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    03. Initialize all graphs
        File: `max/python/max/engine/api.py`
        Function/anchor: `def init_all(`
        Role: Validates the merged weight registry and asks the native engine to instantiate executable graph models.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    04. Execute an initialized model
        File: `max/python/max/engine/api.py`
        Function/anchor: `def _Model_execute`
        Role: Validates contiguous driver-backed inputs, crosses into native execution, and returns device-aware output Buffers.
        Evidence: `SOURCE + BOUNDARY`
        Math: `state_(n+1) = transform_n(state_n)`
    05. Llama model execute wrapper
        File: `max/python/max/pipelines/architectures/llama3/model.py`
        Function/anchor: `def execute(self, model_inputs`
        Role: Maps Llama-specific model input buffers to the initialized graph Model and adapts the returned logits and cache outputs.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    06. Initial token inputs
        File: `max/python/max/pipelines/architectures/llama3/model.py`
        Function/anchor: `def prepare_initial_token_inputs`
        Role: Builds token and row-offset buffers required by the ragged Llama graph before execution begins.
        Evidence: `SOURCE + BOUNDARY`
        Math: `row_i = tokens[offset_i : offset_(i+1)]`
    07. Attach Python stack traces
        File: `GraphCompiler/lib/MGPDialect/Transforms/AttachPythonStackTraces.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Carries Python-origin locations into lowered runtime operations so failures can be related back to model source.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    08. Simple buffer reuse
        File: `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Reuses compatible allocations whose live ranges do not overlap while preserving the non-aliasing contract of distinct mo.buffer values.
        Evidence: `SOURCE`
        Math: `reuse(a,b) only if live(a) intersect live(b) = empty`
    09. Dynamic memory planning
        File: `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Builds runtime allocation plans for dynamic shapes and defers allocation decisions that require execution-time sizes.
        Evidence: `SOURCE`
        Math: `state_(n+1) = transform_n(state_n)`
    10. Emit Mojo kernels
        File: `GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Serializes outlined generated-kernel bodies as Mojo source consumed by the offload compilation step.
        Evidence: `SOURCE + COMPILE-ONLY`
        Math: `state_(n+1) = transform_n(state_n)`
    11. JIT compile kernels
        File: `GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp`
        Function/anchor: `void runOnOperation() override`
        Role: Compiles generated Mojo kernels for selected targets, installs resulting artifacts, and removes intermediate declarations as the pipeline advances.
        Evidence: `SOURCE + COMPILE-ONLY`
        Math: `state_(n+1) = transform_n(state_n)`
