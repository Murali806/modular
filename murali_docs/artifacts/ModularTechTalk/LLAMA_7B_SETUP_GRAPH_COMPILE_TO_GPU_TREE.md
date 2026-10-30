# Llama 7B Setup: Graph Compile to GPU Call-Flow Tree

Evidence: `SOURCE` unless a branch is marked `BOUNDARY` or `GPU-LAB`.

Source revision: `90ede769dad97c0b1edb3b245b7968f635d77c1b`

Scope: built-in `LlamaForCausalLM`, single-device Graph API path. Tensor
parallel, data parallel, device-graph capture, and ModuleV3 are named where
they branch, but are not expanded as the primary path.

This document starts at `max serve`, follows model/config/weight setup through
MLIR graph finalization and native compilation, then follows initialization,
host/device movement, runtime execution, Mojo kernels, and the GPU driver.

## 0. Read The Stack Before The Tree

```text
max serve process
+-- Python control plane
|   +-- CLI, registry, pipeline, model/config, batching
|   +-- Builds symbolic MAX Graph objects and MLIR mo.graph operations
|
+-- MAX native engine                                    [BOUNDARY]
|   +-- Compiles finalized MLIR into CompiledModels / MEF
|   +-- Initializes ExecutableModel objects with weight buffers
|   +-- Executes models through native TensorMap/runtime APIs
|
+-- Mojo kernels and MAX driver HAL
|   +-- Select/specialize kernels for device type/API/architecture
|   +-- Own Device, Context, Buffer, Queue, Stream abstractions
|   +-- Enqueue asynchronous copies and kernel launches
|
+-- Vendor driver and GPU                                [BOUNDARY]
    +-- CUDA/HIP/Metal driver calls, command streams, modules, events
    +-- H2D/D2H/D2D transfers and GPU grid execution
```

Do not merge these three meanings of "finalize":

```text
Llama3Config.finalize()
    -> resolves model dimensions and policy from HF config + real weights

Graph.output()
    -> finalizes the mo.graph signature, terminator, result names, verification

InferenceSession.compile*()
    -> hands the finalized MLIR module to the native graph compiler
```

## 1. End-To-End Object And Call Tree

```text
max serve --model <llama-7b-checkpoint>
|
+-- [1] Resolve serving configuration and architecture
|   +-- serve_api_server_and_model_worker(settings, pipeline_args)
|   |   +-- PipelineConfig.from_args(...)
|   |   +-- PIPELINE_REGISTRY.retrieve_factory(...)
|   |   |   +-- retrieve_architecture("LlamaForCausalLM")
|   |   |   +-- choose Llama3Model + tokenizer + weight adapter
|   |   |   +-- compute MemoryPlan
|   |   +-- fastapi_app(...)
|   |   +-- lifespan(...) starts the model worker/pipeline
|   |
|   +-- Objects created
|       +-- PipelineConfig: resolved runtime/model/serve policy
|       +-- SupportedArchitecture: declarative Llama registration
|       +-- RetrievedFactory: model factory + tokenizer + memory plan
|
+-- [2] Create concrete devices, engine session, checkpoint reader
|   +-- TextGenerationPipeline.__init__(...)
|   |   +-- load_devices(DeviceSpec...)
|   |   |   +-- load_device(DeviceSpec.accelerator(0))
|   |   |       +-- Accelerator(0)                          [BOUNDARY]
|   |   +-- InferenceSession(devices=[gpu0])
|   |   |   +-- _InferenceSession(...)                     [BOUNDARY]
|   |   +-- pipeline_config.configure_session(session)
|   |   +-- load_weights(...)
|   |   +-- Llama3Model(...)
|   |
|   +-- Object distinction
|       +-- DeviceSpec: CLI/config request such as gpu:0
|       +-- Device: concrete runtime accelerator object
|       +-- DeviceRef: symbolic placement embedded in graph tensor types
|
+-- [3] Resolve host-side Llama state
|   +-- LlamaModelBase.__init__(...)
|   |   +-- GraphPipelineModelWithKVCache.__init__(...)
|   |   +-- self.load_model(session)
|   |       +-- load_graphs(session)
|   |           +-- _load_state_dict()
|   |           +-- _create_model_config(state_dict)
|   |           |   +-- Llama3Config.initialize(...)
|   |           |   +-- Llama3Config.finalize(...)
|   |           +-- _init_distributed_runtime(...)
|   |           +-- graph_specs(model_config)
|   |           +-- compile_and_init_graphs(session, specs)
|   |
|   +-- Objects created/resolved
|       +-- state_dict: adapted host weight objects keyed by MAX names
|       +-- Llama3Config: dimensions, dtype, RoPE, KV, devices, policy
|       +-- GraphSpec: module + graph name + symbolic input types + weights
|
+-- [4] Construct and trace the symbolic decoder graph
|   +-- LlamaModelBase._single_device_spec(config)
|   |   +-- single_model = Llama3(config)
|   |   +-- _load_weights(single_model, state_dict)
|   |   +-- GraphSpec(single_model, "llama3", input_tree, weights=...)
|   |
|   +-- GraphSpec.trace()
|       +-- Llama3.trace(*input_types, name="llama3")
|           +-- define forward(tokens, row_offsets, return_n_logits, *kv)
|           +-- Graph(name, forward, input_types, custom_extensions, ...)
|               +-- create builtin.ModuleOp
|               +-- create mo.GraphOp with symbolic block arguments
|               +-- _populate_device_info_mapping()
|               +-- _import_kernels(...)
|               +-- forward(*graph.inputs)
|               |   +-- unflatten KV cache inputs
|               |   +-- Llama3.__call__(...)
|               |       +-- token embedding
|               |       +-- 32 transformer blocks for a typical 7B model
|               |       +-- RMSNorm + output projection
|               +-- Graph.output(outputs)
|
+-- [5] Finalize and verify mo.graph
|   +-- Graph.output(...)
|       +-- tree.flatten(outputs)
|       +-- convert MAX Values to MLIR values/types
|       +-- create functionType and kgen generator signature
|       +-- add mo.OutputOp terminator
|       +-- set argument_names and result_names
|       +-- clear temporary subgraph state
|       +-- self._mlir_op.verify()
|           +-- finalized, verified MLIR module
|
+-- [6] Compile finalized MLIR
|   +-- compile_and_init_graphs(...)
|   |   +-- session.compile_reusing_mefs(traced_graph)
|   |       +-- InferenceSession.compile_async(...)
|   |           +-- seed kernel declarations from KernelLibrary analysis
|   |           +-- InferenceSession._compile_module(module, ...)
|   |               +-- acquire process-global _COMPILATION_LOCK
|   |               +-- self._impl.compile(module.mlir_module._CAPIPtr, ...)
|   |                   +-- native Engine::Context::compileModel(...) [BOUNDARY]
|   |                       +-- graph compiler lowering/optimization [BOUNDARY]
|   |                       +-- kernel selection/specialization       [BOUNDARY]
|   |                       +-- build CompiledModels / MEF artifact   [BOUNDARY]
|   +-- CompiledModel
|       +-- has executable artifact metadata
|       +-- does not yet bind weights or device-memory allocations
|
+-- [7] Initialize executable model and place persistent state
|   +-- compile_and_init_graphs(...)
|       +-- collect GraphSpec.weights_registry()
|       +-- session.init_all(compiled, weights_registry=weights)
|           +-- validate weight names, device type, contiguity
|           +-- self._impl._load_all(compiled, weights_registry) [BOUNDARY]
|               +-- Engine::Context::initModel(...)              [BOUNDARY]
|               +-- resolve DeviceRef -> ConfiguredDevice        [BOUNDARY]
|               +-- allocate persistent device storage           [BOUNDARY]
|               +-- bind/copy weight data as required            [BOUNDARY]
|               +-- return native Model / ExecutableModel
|       +-- CompiledCallable(engine_model=Model, compiled=CompiledModel)
|
+-- [8] Stage one request and execute on GPU
    +-- Llama3BatchProcessor.prepare_initial_token_inputs(...)
    |   +-- build ragged token array and row offsets on host
    |   +-- Buffer.from_numpy(tokens_np)
    |   +-- Buffer.to(device0)                         H2D
    |   +-- attach KV cache buffers already owned by KV manager
    |
    +-- LlamaModelBase.execute(model_inputs)
    |   +-- self.model.execute(*model_inputs.buffers)
    |       +-- _Model_execute(...)
    |           +-- validate contiguous inputs
    |           +-- DLPack input -> Buffer.from_dlpack(...), if needed
    |           +-- self._execute_device_tensors(input_buffers)   [BOUNDARY]
    |               +-- executeFromDriver(...)                    [BOUNDARY]
    |               +-- Engine::Context::execute(...)             [BOUNDARY]
    |               +-- native runtime schedules compiled kernels [BOUNDARY]
    |               +-- Queue/Stream enqueue copies and launches
    |               +-- vendor driver submits work to GPU         [BOUNDARY]
    |               +-- GPU kernels update activations and KV pages
    |               +-- return output Buffer handles
    |
    +-- sampling output
        +-- generated token Buffer is device-resident
        +-- staging Buffer(..., usage=Usage.STAGING)
        +-- generated_tokens_host.inplace_copy_from(device)       D2H
        +-- generated_tokens_host.to_numpy()
```

## 2. Llama 7B Math That Becomes Graph Operations

```text
Typical Llama 7B dimensions
+-- layers L = 32
+-- hidden width d = 4096
+-- attention heads h = 32
+-- head width d_h = d / h = 128
+-- feed-forward width d_ff approximately 11008
+-- vocabulary V approximately 32000 for Llama 2 7B
```

Per decoder block:

```text
x_l
|
+-- a_l = RMSNorm(x_l)
|   +-- rms(x) = sqrt((1/d) * sum_i x_i^2 + epsilon)
|   +-- RMSNorm(x) = gamma elementwise-multiply x / rms(x)
|
+-- Q = a_l W_Q, K = a_l W_K, V = a_l W_V
|   +-- reshape Q,K,V to [tokens, heads, d_h]
|   +-- apply rotary position transform to Q and K
|
+-- attention
|   +-- scores = (Q K^T) / sqrt(d_h) + causal_mask
|   +-- P = softmax(scores)
|   +-- A = P V
|   +-- u_l = x_l + A W_O
|
+-- m_l = RMSNorm(u_l)
+-- gated MLP
|   +-- g = SiLU(m_l W_gate)
|   +-- v = m_l W_up
|   +-- MLP(m_l) = (g elementwise-multiply v) W_down
|
+-- x_(l+1) = u_l + MLP(m_l)
```

The Graph API records the tensor operations and symbolic types. It does not
execute this math during tracing. The native compiler later decides fusion,
layout, scheduling, and concrete kernel implementations.

Approximate dense parameter count, ignoring small norms/biases:

```text
embedding/output       approximately V*d
attention per layer    approximately 4*d^2
gated MLP per layer    approximately 3*d*d_ff

P approximately V*d + L*(4*d^2 + 3*d*d_ff)
  approximately 32000*4096 + 32*(4*4096^2 + 3*4096*11008)
  approximately 6.7 billion parameters

FP16/BF16 weight bytes approximately 2*P approximately 13.4 GB
```

This estimate explains why compilation and initialization are separate:
compiled code can be reusable while weight/device allocations are large and
device-specific.

## 3. Graph Recording And Finalization Tree

```text
Graph.__init__(name, forward, input_types, ...)
+-- create/reuse builtin.ModuleOp
+-- create mo.GraphOp
|   +-- input_types -> MLIR block argument types
|   +-- symbolic dimensions -> kgen parameter declarations
+-- create host/device chain values
|   +-- chain encodes ordering dependencies; it is not a tensor
+-- attach mo.device_info_mapping
|   +-- CPU entry
|   +-- Accelerator entry when visible
|       +-- label, API, architecture, model
+-- import kernel library paths
+-- enter CURRENT_GRAPH context
+-- call forward(symbolic Value objects)
|   +-- every ops.* call appends an MLIR operation
|   +-- every op may be locally verified by Graph._verify_op()
+-- call Graph.output(...)
    +-- freeze output tree shape
    +-- write callable function signature
    +-- add mo.OutputOp terminator
    +-- name arguments/results for engine execution
    +-- verify the complete mo.graph
```

Example symbolic shape reasoning:

```text
tokens             : uint32[T] on gpu:0
input_row_offsets  : uint32[B+1] on gpu:0
return_n_logits    : int64[1] on host or resolved graph placement
KV cache inputs    : opaque BufferType / tensor leaves per cache strategy
logits             : float[B_or_T, V]

For rows with lengths [3, 2]:
tokens  = [t0, t1, t2, t3, t4]
offsets = [0, 3, 5]
row r occupies tokens[offsets[r] : offsets[r+1]]
```

## 4. Compilation Tree: What Is Source-Proven And What Is Opaque

```text
Python InferenceSession.compile_async(Graph)
+-- capture Graph.module
+-- capture expected weight placement from graph._weights
+-- append custom extension paths
+-- KernelLibrary._analysis.seed_kernel_decls(mlir_module)
+-- _compile_module(...)
    +-- acquire _COMPILATION_LOCK
    +-- self._impl.compile(MLIR C-API pointer, extensions, pipeline, flags)
        |
        +-- Engine::Context::compileModel(CompilationRequest)       [SOURCE]
        +-- CompiledModels::create(...)                            [SOURCE]
        |
        +-- exact optimization/lowering pass pipeline              [BOUNDARY]
        +-- exact target code generation details                   [BOUNDARY]
        +-- exact CUDA/HIP module creation timing                  [BOUNDARY]
        |
        +-- AsyncValue<CompiledModels>
```

Visible native data model:

```text
CompilationRequest
+-- ModelSource: MLIR module, path, or serialized model
+-- pipelineName
+-- passthrough DriverConfig values
    +-- DeviceConfig
    +-- KernelLibraryDriverConfig
    +-- TileBasedFusionDriverConfig

CompiledModels
+-- SourceModel
+-- MOFContext / framework context
+-- UninitializedModel
    +-- compiled model buffer / MEF
    +-- list<M::Model>
    +-- MEFFile handle
```

## 5. Mojo Kernel Selection Tree

```text
MAX graph operation, for example mo.matmul
+-- Graph records operation + dtype/shape/layout/device metadata
+-- compiler queries built-in/custom kernel declarations             [BOUNDARY]
+-- extensibility registration can specialize by
|   +-- operation name: "mo.matmul"
|   +-- device type: cpu or gpu
|   +-- API: cuda, hip, metal, ...
|   +-- architecture: for example sm_100a
|   +-- model: for example NVIDIA B200
+-- selected Mojo implementation
|   +-- compiled/specialized for the target device
|   +-- may call vendor libraries or use native GPU instructions
+-- compiled device function/bundle
    +-- loaded by runtime Context
    +-- launched through Queue/Stream
```

Representative built-in registrations:

```text
max/kernels/src/graph_compiler/builtin_kernels/
+-- linalg.mojo
|   +-- @extensibility.register("mo.matmul", type="cpu")
|   +-- @extensibility.register("mo.matmul", type="gpu")
|   +-- matmul/SwiGLU/quantized variants
+-- attention.mojo
|   +-- masked flash-attention registrations
|   +-- paged/ragged attention registrations
+-- kernels.mojo
|   +-- RoPE and RoPE-store registrations
+-- kv_cache.mojo
    +-- paged KV-cache and fused normalization registrations
```

Mojo's role is kernel/runtime implementation and specialization. Python
objects do not move into a Mojo GPU function. Python hands graph metadata and
buffer handles across native bindings; compiled functions consume device
addresses and scalar metadata.

## 6. Initialization And Weight Placement Tree

```text
session.init_all(compiled, weights_registry)
+-- host checks
|   +-- every expected weight name exists
|   +-- expected CPU/GPU placement agrees with DLPack device type
|   +-- each value is contiguous
+-- self._impl._load_all(...)                              [BOUNDARY]
    +-- Engine::Context::initModel(...)
    +-- resolve symbolic DeviceRef values
    +-- allocate persistent runtime/device state
    +-- bind weight registry entries to compiled model slots
    +-- perform required copies/conversions/layout materialization
    +-- return ExecutableModel / Python Model wrapper
```

Ownership model:

```text
checkpoint files
    -> Weights loader
        -> host DLPack-compatible values / WeightData
            -> weights_registry[name]
                -> native initialization                         [BOUNDARY]
                    -> persistent device allocations
                        -> executable Model owns/retains runtime state
```

`CompiledModel` is reusable because it is independent of these allocations.
`Model` is executable because initialization has bound the artifact to weights
and runtime devices.

## 7. Runtime Buffer Movement Tree

```text
Input H2D
+-- NumPy tokens/offsets in CPU memory
+-- Buffer.from_numpy(array)
|   +-- require/produce contiguous array
|   +-- Buffer.from_dlpack(array)
|   +-- usually wraps host storage without an extra copy
+-- host_buffer.to(Accelerator(0))                         [BOUNDARY]
    +-- allocate device Buffer
    +-- enqueue host-to-device copy
    +-- preserve stream/hazard dependency

Model execution
+-- Model.execute(*buffers)
+-- _Model_execute(...)
|   +-- reject non-contiguous inputs
|   +-- convert external DLPack arrays to Buffer
+-- _execute_device_tensors(...)                           [BOUNDARY]
    +-- executeFromDriver(...)
    +-- runtime TensorMap/AsyncValue scheduling
    +-- enqueue compiled kernel sequence

Output D2H
+-- generated_tokens_device
+-- Buffer(..., usage=Usage.STAGING)
+-- host_staging.inplace_copy_from(device_buffer)
|   +-- same element count and dtype required
|   +-- copy implementation is native
+-- host_staging.to_numpy()
    +-- waits for tracked asynchronous copy before CPU reads
```

Memory spaces:

```text
CPU pageable RAM          CPU pinned/staging RAM             GPU VRAM
+----------------+       +----------------------+           +----------------+
| NumPy arrays   | ----> | async DMA-friendly   | --PCIe--> | token buffers  |
| HF weights     |       | host transfer buffer | <-------- | logits/KV      |
+----------------+       +----------------------+           +----------------+
       host code                 driver/runtime                    kernels
```

The exact use of pinned memory, copy engines, and overlap depends on the path,
buffer `Usage`, hardware, and runtime. Treat performance behavior as `GPU-LAB`
until measured on the target GPU.

## 8. MAX Driver HAL Object Tree

```text
Device[DeviceSpec]
+-- device handle
+-- Driver / RawDriver ownership
+-- id and device properties

Context[DeviceSpec]
+-- context handle
+-- owns/borrows Device
+-- function cache
+-- load_bundle(assembly)
+-- load_function(bundle, name)
+-- compile/load device functions

Buffer[DeviceSpec]
+-- MemoryHandle
+-- byte_size
+-- is_host_pinned
+-- owning Context

Queue[DeviceSpec]
+-- queue handle
+-- context
+-- enqueue kernel launch configuration
+-- enqueue events/copies

Stream[DeviceSpec]
+-- Queue wrapper
+-- chain event
+-- copy_to_device(...)
+-- copy_from_device(...)
+-- copy_intra_device(...)
+-- copy_inter_device(...)
```

Asynchronous copy overloads visible in Mojo:

```text
DeviceContext.enqueue_copy(...)
+-- host pointer -> DeviceBuffer
|   +-- Stream.copy_to_device(...)
+-- DeviceBuffer -> host pointer
|   +-- Stream.copy_from_device(...)
+-- DeviceBuffer -> DeviceBuffer on same device
|   +-- Stream.copy_intra_device(...)
+-- DeviceBuffer -> DeviceBuffer on peer device
|   +-- validate same driver
|   +-- Stream.copy_inter_device(...)
+-- HostBuffer -> DeviceBuffer
|   +-- size check
|   +-- Stream.copy_to_device(...)
+-- DeviceBuffer -> HostBuffer
    +-- size check
    +-- Stream.copy_from_device(...)
```

The vendor calls behind `RawDriver` are a boundary. The NVIDIA helper file
defines CUDA handle types such as `CUcontext`, `CUstream`, `CUmodule`, and
`CUevent`, but the full submission implementation is not asserted here.

## 9. One Decode Step Timeline

```text
CPU/Python          native runtime         copy/compute stream       GPU
    |                     |                        |                   |
    | pack tokens         |                        |                   |
    | Buffer.to(gpu) ---->| enqueue H2D ---------->| DMA ------------->|
    | Model.execute ----->| resolve dependencies   |                   |
    |                     | enqueue kernels ------>| launch ---------->|
    |                     |                        |    RMSNorm/QKV     |
    |                     |                        |    RoPE/attention  |
    |                     |                        |    MLP/KV update   |
    |                     | output Buffer handles |<------------------|
    | stage token D2H --->| enqueue D2H ---------->| DMA ------------->|
    | to_numpy waits      |                        |                   |
    |<--------------------| completion             |                   |
    | update request      |                        |                   |
```

The diagram gives dependency order, not measured overlap or latency.

## 10. Files, Classes, Functions, And Their Exact Role

```text
modular/
+-- max/python/max/
|   +-- _entrypoints/cli/serve/
|   |   +-- serve_api_and_model_worker.py
|   |       +-- serve_api_server_and_model_worker()
|   |       |   +-- Converts CLI settings into PipelineConfig and retrieves the
|   |       |       Llama pipeline factory before FastAPI/model-worker startup.
|   |       +-- _exit_on_signal()
|   |       +-- local serve()
|   |
|   +-- pipelines/lib/
|   |   +-- registry.py
|   |   |   +-- class PipelineRegistry
|   |   |   +-- retrieve_factory()
|   |   |   +-- retrieve_architecture()
|   |   |   +-- architecture_for_config()
|   |   |   +-- Maps `LlamaForCausalLM` to the concrete implementation,
|   |   |       tokenizer, adapters, and memory plan.
|   |   +-- pipeline_variants/text_generation.py
|   |   |   +-- class TextGenerationPipeline
|   |   |   +-- __init__()
|   |   |   +-- Builds Device objects, InferenceSession, weights, model, KV
|   |   |       manager, sampler, and request-time generation pipeline.
|   |   +-- graph_specs.py
|   |   |   +-- class GraphSpec
|   |   |   +-- GraphSpec.trace()
|   |   |   +-- GraphSpec.weights_registry()
|   |   |   +-- compile_and_init_graphs()
|   |   |   +-- Converts a model module into finalized Graph(s), calls compile,
|   |   |       merges weight registries, calls init_all, and returns callables.
|   |   +-- interfaces/batch_processor.py
|   |       +-- prepare_initial_token_inputs()
|   |       +-- build_single_replica_ragged_token_arrays()
|   |       +-- Shows the request H2D path: Buffer.from_numpy(...).to(device0).
|   |
|   +-- pipelines/architectures/llama3/
|   |   +-- arch.py
|   |   |   +-- object llama_arch: SupportedArchitecture
|   |   |   +-- Declares architecture names, encodings, model class, tokenizer,
|   |   |       and weight adapters used by the registry.
|   |   +-- model.py
|   |   |   +-- class LlamaModelBase
|   |   |   +-- LlamaModelBase.__init__()
|   |   |   +-- load_model()
|   |   |   +-- load_graphs()
|   |   |   +-- graph_specs()
|   |   |   +-- _single_device_spec()
|   |   |   +-- _tensor_parallel_spec()
|   |   |   +-- _data_parallel_spec()
|   |   |   +-- execute()
|   |   |   +-- Main model setup coordinator from host weights to executable
|   |   |       decoder Model.
|   |   +-- model_config.py
|   |   |   +-- class Llama3Config
|   |   |   +-- Llama3Config.initialize()
|   |   |   +-- Llama3Config.finalize()
|   |   |   +-- construct_kv_params()
|   |   |   +-- create_rope_embedding()
|   |   |   +-- Resolves architecture dimensions and attention/KV/RoPE policy;
|   |   |       this is model-config finalization, not graph finalization.
|   |   +-- llama3.py
|   |   |   +-- class Llama3
|   |   |   +-- Llama3.__init__()
|   |   |   +-- Llama3.trace()
|   |   |   +-- Llama3.input_tree()
|   |   |   +-- Llama3.input_types()
|   |   |   +-- Defines the symbolic forward and creates Graph(...), causing
|   |   |       transformer operations to be recorded into MLIR.
|   |   +-- batch_processor.py
|   |   |   +-- class Llama3BatchProcessor
|   |   |   +-- prepare_initial_token_inputs()
|   |   |   +-- _stage_ragged_token_inputs()
|   |   |   +-- _make_inputs()
|   |   |   +-- Packs request tokens, row offsets, KV inputs, and signals into
|   |   |       concrete Buffer objects expected by the compiled graph.
|   |   +-- weight_adapters.py
|   |       +-- convert_safetensor_state_dict()
|   |       +-- convert_gguf_state_dict()
|   |       +-- Renames/transforms checkpoint entries to MAX module weight names.
|   |
|   +-- graph/
|   |   +-- graph.py
|   |       +-- class Module
|   |       +-- class KernelLibrary
|   |       +-- KernelLibrary.load_paths()
|   |       +-- class Graph
|   |       +-- Graph.__init__()
|   |       +-- Graph._populate_device_info_mapping()
|   |       +-- Graph._verify_op()
|   |       +-- Graph.output()
|   |       +-- Graph._import_kernels()
|   |       +-- Owns the MLIR module/mo.graph operation, records symbolic ops,
|   |           attaches kernels/device metadata, finalizes, and verifies.
|   |
|   +-- engine/
|   |   +-- api.py
|   |       +-- class CompiledModel
|   |       +-- class InferenceSession
|   |       +-- InferenceSession.compile_reusing_mefs()
|   |       +-- InferenceSession.compile_async()
|   |       +-- InferenceSession.compile()
|   |       +-- InferenceSession._compile_module()
|   |       +-- InferenceSession.init()
|   |       +-- InferenceSession.init_all()
|   |       +-- _Model_execute()
|   |       +-- Python API boundary for native compilation, initialization, and
|   |           execution. Imports Model and _InferenceSession from max._core.
|   |
|   +-- driver/
|       +-- driver.py
|       |   +-- class DeviceSpec
|       |   +-- load_device()
|       |   +-- load_devices()
|       |   +-- scan_available_devices()
|       |   +-- Turns requested cpu/gpu IDs into concrete native Device objects.
|       +-- buffer.py
|           +-- _from_numpy()
|           +-- _from_dlpack()
|           +-- inplace_copy_from()
|           +-- _to_numpy()
|           +-- Adds Python validation/convenience around the native Buffer,
|               while Buffer.to() and underlying copies remain native methods.
|
+-- max/include/max/max_license/API/c/
|   +-- EngineContext.h
|   |   +-- class Engine::Context
|   |   +-- Context::compileModel()
|   |   +-- Context::initModel()
|   |   +-- Context::execute()
|   |   +-- Context::resolveDevice()
|   |   +-- Visible native contract connecting compiler, executable model,
|   |       runtime TensorMap, and configured devices.
|   +-- CompiledModels.h
|   |   +-- class CompilationRequest
|   |   +-- struct UninitializedModel
|   |   +-- struct CompiledModels
|   |   +-- CompiledModels::create()
|   |   +-- Represents the device-independent compiled artifact before init.
|   +-- ModelExecution.h
|       +-- struct ExecResult
|       +-- validateInputSpec()
|       +-- extractDriverResults()
|       +-- executeFromDriver()
|       +-- Bridges device Buffer references into/out of native model execution.
|
+-- max/kernels/src/graph_compiler/
|   +-- extensibility/decorators.mojo
|   |   +-- register()
|   |   +-- Defines operation/device/API/architecture/model-specific kernel
|   |       registration used by custom and built-in Mojo implementations.
|   +-- builtin_kernels/
|       +-- linalg.mojo
|       |   +-- mo.matmul CPU/GPU registrations and variants.
|       +-- attention.mojo
|       |   +-- flash, masked, paged, and ragged attention registrations.
|       +-- kernels.mojo
|       |   +-- RoPE and other core tensor kernel registrations.
|       +-- kv_cache.mojo
|           +-- Paged KV-cache and fused cache operation registrations.
|
+-- max/driver/src/max_license/
    +-- _hal/
    |   +-- device.mojo
    |   |   +-- struct Device[DeviceSpec]
    |   +-- context.mojo
    |   |   +-- struct Context[DeviceSpec]
    |   |   +-- load_bundle()
    |   |   +-- load_function()
    |   +-- buffer.mojo
    |   |   +-- struct Buffer[DeviceSpec]
    |   |   +-- struct BufferView
    |   +-- queue.mojo
    |   |   +-- struct Queue[DeviceSpec]
    |   |   +-- launch/copy/event queue methods
    |   +-- stream.mojo
    |       +-- struct Stream[DeviceSpec]
    |       +-- copy_to_device()/copy_from_device()
    |       +-- copy_intra_device()/copy_inter_device()
    +-- _device_context_hal/
        +-- __init__.mojo
        |   +-- struct DeviceContext
        |   +-- enqueue_function()
        |   +-- enqueue_copy() overloads
        |   +-- struct DeviceBuffer
        |   +-- struct HostBuffer
        |   +-- High-level Mojo host API that enqueues asynchronous work through
        |       Stream and the selected HAL plugin.
        +-- _nvidia_cuda.mojo
            +-- CUcontext/CUstream/CUmodule/CUevent handle declarations
            +-- Thin NVIDIA-specific helper boundary; vendor driver behavior is
                not inferred beyond the source-visible declarations.
```

## 11. Debug Checkpoints By Phase

```text
Wrong architecture/model class
+-- inspect PipelineConfig.main_architecture_name
+-- inspect PIPELINE_REGISTRY.retrieve_architecture()

Wrong dimensions or missing weights
+-- inspect Llama3Config.finalize()
+-- inspect adapted state_dict names and shapes

Graph construction failure
+-- print Graph / MLIR near Graph.output()
+-- enable max-debug.source-tracebacks
+-- distinguish per-op Graph._verify_op() from whole-graph verification

Compilation failure
+-- failure is after self._impl.compile(...)
+-- inspect custom extension paths and device metadata
+-- do not debug this as a checkpoint-loading issue without evidence

Initialization failure
+-- inspect weights_registry names, DLPack device types, contiguity
+-- inspect memory plan and actual accelerator availability

Execution failure
+-- inspect concrete Buffer dtype/shape/device against model input metadata
+-- inspect H2D completion/hazard ordering and KV-cache inputs

Performance issue                                                    [GPU-LAB]
+-- measure compile and init separately
+-- profile H2D, kernels, D2H, and synchronization separately
+-- inspect kernel selection and GPU occupancy on the target accelerator
```

## 12. Core Invariants

```text
1. DeviceRef is symbolic; Device is concrete.
2. Config finalization is not graph finalization.
3. Graph.output() finalizes and verifies mo.graph; it does not compile it.
4. CompiledModel has code/artifact state; Model has initialized runtime state.
5. Host arrays become executable inputs only after they are Buffer objects on
   the devices expected by the graph.
6. Mojo kernels consume compiled arguments/device addresses, not Python objects.
7. Queue/Stream work is asynchronous until a dependency or host read requires
   completion.
8. Exact native compiler passes and vendor driver calls remain BOUNDARY unless
   their implementation is directly visible or measured.
```
