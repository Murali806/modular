"""Annotated, source-linked call trees for the add-one lesson.

The strings are portable trees as well as the HTML's authoring source. Source
markers become real offline links; no compiler or inference is run here.
"""
from __future__ import annotations

import hashlib
import html
import re

import markdown

from compiler_card_conversions import render_card_conversion
from compiler_session_calltrees import PANELS as SESSION_CALL_STACK_PANELS
import compiler_custom_call_examples as custom_examples
from compiler_session_values import EXPLANATION as SESSION_VALUES_EXPLANATION

GRAPH = "max/python/max/graph/graph.py"
API = "max/python/max/engine/api.py"
CORE = "max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp"
CONTEXT = "max/lib/max_license/API/c/EngineContext.cpp"
COMPILED = "max/lib/max_license/API/c/CompiledModels.cpp"
MODEL = "GraphCompiler/lib/MEFModel/MEFModel.cpp"

# Full unchanged files are embedded in Source Explorer and validated against
# the same revision as the animation. Existing compiler/kernel refs are reused.
SOURCES = [
    ("ct_session_members", "max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h", "84:99", "lines", 0, "Native session owns compile config, runtime config, engine context and devices"),
    ("ct_request_members", "max/include/max/max_license/API/c/CompiledModels.h", "65:166", "lines", 0, "CompilationRequest defaults, config cloning and kernel-library path entries"),
    ("ct_max_context_members", "Support/include/Support/Context.h", "28:88", "lines", 0, "Reference-counted MAX context stores runtime reference and typed service objects"),
    ("ct_session_extension", CORE, "600:613", "lines", 0, "Validate a compiled Mojo extension path and add it to compile settings"),
    ("ct_execute_py", API, "_Model_execute", "python", 0, "Python Model.execute converts supported inputs and invokes native execution"),
    ("ct_execute_install", API, "287:296", "lines", 0, "Install Python execute implementation on the native Model wrapper"),
    ("ct_execute_binding", "max/python/max/_core/max_license/modules/engine.cpp", "565:567", "lines", 0, "Nanobind tensor execution entry point"),
    ("ct_execute_specs", CORE, "420:450", "lines", 0, "Validate runtime input count, dtype and shape against compiled specifications"),
    ("ct_execute_device", CORE, "372:419", "lines", 0, "Validate the input device and pinned-host exception"),
    ("ct_execute_native", CORE, "452:519", "lines", 0, "Native execution validates buffers and prepares AsyncValue inputs"),
    ("ct_execute_prepared", CORE, "521:539", "lines", 0, "Release GIL for runtime execution and extract Python buffer results"),
    ("ct_execute_input", "GraphCompiler/lib/Driver/Device/DeviceExecution.cpp", "112:126", "lines", 0, "Retain the input buffer in a ready Tensor AsyncValue"),
    ("ct_execute_driver", "max/lib/max_license/API/c/ModelExecution.cpp", "193:214", "lines", 0, "Driver path invokes the initialized executable model"),
    ("ct_execute_callable", MODEL, "363:385", "lines", 0, "Assemble retained runtime state and inputs; invoke compiled execute callable"),
    ("ct_execute_results", "max/lib/max_license/API/c/ModelExecution.cpp", "101:188", "lines", 0, "Await execution results, discard ordering chain and marshal output buffers"),
    ("ct_execute_py_results", CORE, "56:80", "lines", 0, "Preserve output aliases while returning Python Buffer wrappers"),
    ("ct_graph_callsite", "max/examples/custom_ops/addition.py", "33:33", "lines", 0, "Exact example call site that constructs Graph('addition')"),
    ("ct_graph_location", GRAPH, "_location", "python", 0, "Create source/profile Location metadata for graph IR"),
    ("ct_graph_frame_loc", "max/python/max/_core/max_license/modules/graph.cpp", "MlirLocation frame_loc(MlirContext ctx, nb::handle frame)", "cpp", 0, "Wrap Python stack summaries in an MLIR opaque source Location"),
    ("ct_graph_context", "max/python/max/_mlir_context.py", "in_default_mlir_context", "python", 0, "Graph constructor's MLIR-context decorator"),
    ("ct_graph_context_state", "max/python/max/_mlir_context.py", "26:67", "lines", 0, "Shared default MLIR context and thread-local active-context guard"),
    ("ct_graph_inputs", GRAPH, "934:950", "lines", 0, "Graph.inputs wraps its MLIR block arguments as symbolic values"),
    ("ct_graph_context_manager", GRAPH, "1134:1150", "lines", 0, "Graph context manager sets and restores Graph.current"),
    ("ct_mlir_thread_pool", "max/python/max/_mlir_context.py", "MLIRThreadPoolExecutor", "python", 0, "Executor that activates the default MLIR context for worker tasks"),
    ("ct_capi_export", "max/python/max/_core/max_license/_core.cpp", "606:611", "lines", 0, "MAX OpState._CAPIPtr exports the existing operation in an MLIR capsule"),
    ("ct_mlir_reexport", "max/python/max/mlir/__init__.py", "19:44", "lines", 0, "MAX re-exports MLIR Python Operation and other IR binding objects"),
    ("ct_llvm_pin", "bazel/public-patches/llvm_source.bzl", "1:8", "lines", 0, "Repository pin for the upstream LLVM/MLIR source snapshot"),
    ("ct_builtin_module", "max/python/max/_core/max_license/modules/dialects/builtin.cpp", "62:70", "lines", 0, "ModuleOp's native constructor and body binding"),
    ("ct_block_end", "max/python/max/_core/max_license/_core.cpp", "408:412", "lines", 0, "Block.end creates a native block-and-iterator insertion point"),
    ("ct_op_builder", "max/python/max/_core/max_license/_core.cpp", "612:617", "lines", 0, "OpBuilder's nanobind constructor creates the native MLIR builder"),
    ("ct_tensor_to_mlir", "max/python/max/graph/type.py", "TensorType.to_mlir", "python", 0, "Convert the Python tensor description to a native MO tensor type"),
    ("ct_filter_layout", "max/python/max/graph/type.py", "FilterLayout", "python", 0, "Convolution filter layout enum and native layout attribute conversion"),
    ("ct_conv_layout", "max/python/max/graph/ops/conv.py", "conv2d", "python", 0, "Convolution filter layout meanings and attachment to tensor metadata"),
    ("ct_layout_read", "GraphCompiler/lib/MODialect/MOTypes.cpp", "LayoutAttr TensorType::getLayoutAttr()", "cpp", 0, "Read layout from native tensor metadata"),
    ("ct_layout_packing", "GraphCompiler/lib/MODialect/PackingShapeFunctions.cpp", "M::MO::materializePackConvFilterShape(InvokeShapeFuncOp invoke,", "cpp", 0, "Filter layout selects which dimension is the output-channel count"),
    ("ct_tensor_description", "max/python/max/graph/type.py", "358:407", "lines", 0, "Symbolic TensorType contract and constructor fields"),
    ("ct_shape_to_mlir", "max/python/max/graph/shape.py", "Shape.to_mlir", "python", 0, "Convert dimension descriptions into a MOSH shape attribute"),
    ("ct_static_dim_to_mlir", "max/python/max/graph/dim.py", "StaticDim.to_mlir", "python", 0, "Static sizes become typed dimension attributes"),
    ("ct_symbolic_dim_to_mlir", "max/python/max/graph/dim.py", "SymbolicDim.to_mlir", "python", 0, "Symbolic sizes become named KGEN parameter-reference attributes"),
    ("ct_device_ref_to_mlir", "max/python/max/graph/type.py", "DeviceRef.to_mlir", "python", 0, "Convert symbolic device kind and index to a device attribute"),
    ("ct_native_tensor_type", "GraphCompiler/lib/MODialect/MOTypes.cpp", "219:329", "lines", 0, "Native MO tensor type printing and context-inferred type builders"),
    ("ct_mo_dialect", "GraphCompiler/include/GraphCompiler/MODialect/MODialect.td", "25:51", "lines", 0, "MO: MAX's frontend graph dialect"),
    ("ct_graph_op_build", "GraphCompiler/lib/MODialect/MOOps.cpp", "void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,", "cpp", 0, "GraphOp builds its symbol, signatures, region and input block arguments"),
    ("ct_graph_op_definition", "GraphCompiler/include/GraphCompiler/MODialect/MOOps.td", "211:297", "lines", 0, "GraphOp's module parent, graph region, attributes and output terminator"),
    ("ct_graph_module", GRAPH, "562:637", "lines", 0, "Module wrapper and shared encoder/decoder graph example"),
    ("ct_elementwise_builder", "max/python/max/graph/ops/elementwise.py", "", "file", 0, "Python add/mul record relaxed graph operations"),
    ("ct_rmo_dialect", "GraphCompiler/include/GraphCompiler/RMODialect/RMODialect.td", "", "file", 0, "Relaxed operations lower to strictly typed MO"),
    ("ct_mo_map_add_test", "GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-add-static.mlir", "", "file", 0, "Compiler test: tensor addition becomes elementwise iteration, loads, call and store"),
    ("ct_mo_map_opaque_test", "GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels.mlir", "", "file", 0, "Compiler test: registered custom ops lower to opaque MAP iteration and calls"),
    ("ct_device_chain_map", GRAPH, "_DeviceChainMap", "python", 0, "Lazy per-device dependency chain map"),
    ("ct_update_chain", GRAPH, "Graph._update_chain", "python", 0, "Set the host orchestration chain"),
    ("ct_is_chain", GRAPH, "_is_chain_value", "python", 0, "Recognize an internal chain by its native type"),
    ("ct_generated_op", GRAPH, "Graph._add_op_generated", "python", 0, "Insert and verify a generated op in the current graph block"),
    ("ct_add_weight", GRAPH, "Graph.add_weight", "python", 0, "Register symbolic external weights and reuse existing declarations"),
    ("ct_add_subgraph", GRAPH, "Graph.add_subgraph", "python", 0, "Subgraph construction passes an explicit trailing chain input"),
    ("ct_chain_definition", "GraphCompiler/include/GraphCompiler/MODialect/MOOps.td", "4795:4821", "lines", 0, "ChainCreateOp dependency semantics and operands"),
    ("ct_graph_enter", GRAPH, "Graph._enter", "python", 0, "Activate and reset the graph construction ContextVar"),
    ("ct_graph_current", GRAPH, "Graph.current", "python", 0, "Resolve the active graph from the construction context"),
    ("ct_insert_result", GRAPH, "Graph._add_op_get_op_with_results", "python", 0, "Insert an MO operation and wrap its symbolic results"),
    ("ct_verify_op", GRAPH, "Graph._verify_op", "python", 0, "Verify an inserted operation's IR structure"),
    ("ct_graph_device", GRAPH, "Graph._populate_device_info_mapping", "python", 0, "Graph's device-to-target metadata"),
    ("ct_graph_extensions", GRAPH, "_resolved_custom_extensions", "python", 0, "Explicit extensions and default overlays"),
    ("ct_graph_library", GRAPH, "KernelLibrary.__init__", "python", 0, "KernelLibrary creates native analysis"),
    ("ct_graph_import", GRAPH, "Graph._import_kernels", "python", 0, "Graph records resolved kernel package paths"),
    ("ct_graph_verify", GRAPH, "KernelLibrary.verify_custom_op", "python", 0, "Custom-op verification and analysis reuse"),
    ("ct_mojo_run", "Mojo/python/mojo/run.py", "subprocess_run_mojo", "python", 0, "Select and invoke the Mojo compiler process"),
    ("ct_mojo_env", "Mojo/python/mojo/run.py", "_sdk_default_env", "python", 0, "Resolve bundled Mojo executable and SDK paths"),
    ("ct_mojo_package_format", "Mojo/docs/compiler/MojoCompilerWalkthrough.md", "612:673", "lines", 0, "Documented precompile stages, LIT package shape and MLIR bytecode"),
    ("ct_precompile", "Mojo/tools/mojo/Precompile/mojo-precompile.cpp", "static int precompile(const State &subcommandState)", "cpp", 0, "mojo precompile: parse, build, check, serialize"),
    ("ct_parser", "Mojo/lib/MojoParser/EntryPoint.cpp", "LIT::importMojoPackage(ContextRef context,", "cpp", 1, "Parse a directory as a Mojo package"),
    ("ct_resolver", "Mojo/lib/MojoParser/DeclResolver.cpp", "void DeclResolver::resolveAllReferencedFrom(ASTDecl &decl)", "cpp", 0, "Resolve package modules and referenced dependencies"),
    ("ct_package_build", "Mojo/lib/Precompile/Precompile.cpp", "ErrorOr<OwningOpRef<ModuleOp>> M::KGEN::buildPrecompiledPackage(", "cpp", 0, "Clone post-parse package IR for serialization"),
    ("ct_check_lit", "Mojo/lib/Compiler/KGENCompiler.cpp", "LogicalResult KGENCompiler::runCheckLITPipeline(ModuleOp module)", "cpp", 0, "Run semantic and lifetime checks on parsed package IR"),
    ("ct_package_write", "Mojo/lib/Support/MojoPrecompiledFile.cpp", "LogicalResult M::KGEN::writePrecompiledFile(Operation *op,\n                                            MojoPrecompiledFileVersion &mojoVer,", "cpp", 0, "MPKG V2 header and zstd-compressed MLIR bytecode"),
    ("ct_analysis_verify", "max/python/max/_core/max_license/lib/Graph/Bindings.cpp", "void M::Graph::Python::Analysis::verifyCustomOp(MlirOperation op)", "cpp", 0, "Native custom-op validation against MojoRegistry"),
    ("ct_validation_registry", "max/python/max/_core/max_license/lib/Graph/Bindings.cpp", "180:210", "lines", 0, "Lazily create declaration module and Mojo registry from package paths"),
    ("ct_validation_lookup", "GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp", "48:73", "lines", 0, "Index kernel declarations by registration string"),
    ("ct_validation_device", "GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp", "157:210", "lines", 0, "Resolve device-compatible registration candidates"),
    ("ct_validation_native", "GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp", "634:688", "lines", 0, "Custom-op verifier: registration, device, signature and supplied parameters"),
    ("ct_validation_mapping", "GraphCompiler/lib/Conversion/OperandInference.cpp", "240:344", "lines", 0, "Map input operands first, then destination-passing results; reject unmatched values"),
    ("ct_validation_params", "GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp", "426:496", "lines", 0, "Collect compile-time declarations and check supplied parameter names and types"),
    ("ct_analysis_seed", "max/python/max/_core/max_license/lib/Graph/Bindings.cpp", "void M::Graph::Python::Analysis::seedKernelDecls(MlirOperation target)", "cpp", 0, "Seed kernel declarations into the caller's graph module"),
    ("ct_session", API, "InferenceSession.__init__", "python", 0, "Python InferenceSession constructor"),
    ("ct_session_binding", "max/python/max/_core/max_license/modules/engine.cpp", "787:803", "lines", 0, "Nanobind session constructor and refcount-sharing init binding"),
    ("ct_compile_binding", "max/python/max/_core/max_license/modules/engine.cpp", "824:844", "lines", 0, "In-memory compile binding and native-session keep-alive"),
    ("ct_session_native", CORE, "M::Core::InferenceSession::InferenceSession(", "cpp", 0, "Native session retains devices, configurations and Engine::Context"),
    ("ct_compile_config", CORE, "CompilationRequest M::Core::InferenceSession::populateCompileConfig(", "cpp", 0, "Clone compilation settings and append extension paths"),
    ("ct_context_create", CONTEXT, "Engine::Context::create(const M::Engine::Config &config)", "cpp", 1, "Get/create the MAX context and validate device labels"),
    ("ct_cpu_options", CONTEXT, "getOrCreateCPUDeviceOptions(const Engine::Config &config,", "cpp", 1, "CPU runtime options and automatic thread selection"),
    ("ct_context_ctor", CONTEXT, "Engine::Context::Context(const M::Engine::Config &runtimeConfig,", "cpp", 0, "Engine context retains runtime config, MAX context and CPUDevice"),
    ("ct_global_context", "Init/lib/Init.cpp", "ErrorOr<ContextRef> Init::getOrCreateContext(StringRef programName,", "cpp", 0, "Mutex-protected MAX-context reuse"),
    ("ct_global_create", "Init/lib/Init.cpp", "createContextImpl(StringRef programName,", "cpp", 1, "MAX-context creation and AsyncRT CPUDevice setup"),
    ("ct_compile", API, "InferenceSession.compile", "python", 0, "Synchronous compile waits and raises on failure"),
    ("ct_compile_module", API, "InferenceSession._compile_module", "python", 0, "Process-global compilation lock and async release callback"),
    ("ct_compile_native", CORE, "M::Core::InferenceSession::compile(\n    const nb::capsule &model,", "cpp", 0, "Native Graph-capsule compilation overload"),
    ("ct_context_compile", CONTEXT, "Engine::Context::compileModel(CompilationRequest config) const", "cpp", 1, "Schedule compilation on the host CPU runtime"),
    ("ct_context_compile_sync", CONTEXT, "Engine::Context::compileModelSync(CompilationRequest config) const", "cpp", 1, "Compile with session device configuration"),
    ("ct_compiled_create", COMPILED, "CompiledModels::create(M::ContextRef context,", "cpp", 1, "Create the framework compiler context and compiled artifact"),
    ("ct_framework_compile", "GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp", "M::ErrorOr<M::BufferRef> MOFContext::compileToBinary(void *source,", "cpp", 0, "Clone graph module, select cache route and compile the working IR"),
    ("ct_convert_mef", "GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp", "static ErrorOr<BufferRef> convertMOToMEF(ModuleOp moduleOp,", "cpp", 0, "Run graph passes and export MEF bytes"),
    ("ct_run_gc", "GraphCompiler/lib/GraphCompiler.cpp", "LogicalResult M::MO::runGCPipeline(ModuleOp moduleOp,", "cpp", 0, "Build and run the MLIR graph pass manager"),
    ("ct_emit_archive", "Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp", "ErrorOr<BufferRef> ObjectCompiler::emitArchive(OwningOpRef<ModuleOp> module,", "cpp", 0, "Emit a compiled native archive from specialized kernel IR"),
    ("ct_compiled_read", COMPILED, "CompiledModels::read(BufferRef buffer, std::unique_ptr<MOFContext> frameworkCtx,", "cpp", 1, "Read prepared bytes into uninitialized model descriptors"),
    ("ct_mef_open", COMPILED, "loadModelsFromMEF(BufferRef compiledModelBuffer)", "cpp", 1, "Open MEF and retain its bytes/descriptors"),
    ("ct_mef_models", MODEL, "ErrorOr<ModelList> M::loadMGPMEFFile(RCRef<MEFFile> mefFile)", "cpp", 0, "Install runtime primitives and find model descriptors"),
    ("ct_init", API, "InferenceSession.init", "python", 0, "Initialize exactly one model"),
    ("ct_load_all", CORE, "M::Core::InferenceSession::loadAll(", "cpp", 1, "Await compilation, initialize each model and retain ownership"),
    ("ct_weight_dict", CORE, "buildWeightsFromDict(const nb::dict &weightsDict)", "cpp", 2, "Create a shared WeightsRegistry and managed DLPack tensor list"),
    ("ct_load", CORE, "M::Core::InferenceSession::load(const M::Model *model,", "cpp", 1, "Await native model initialization while releasing the GIL"),
    ("ct_context_init", CONTEXT, "Engine::Context::initModel(const M::Model *model,", "cpp", 1, "Run initializer on a CPU worker with the proper thread-local runtime"),
    ("ct_executable_create", CONTEXT, "createExecutableModel(const M::Model *model, CPUDevice &cpuDevice,", "cpp", 1, "Map devices, set up a model and await setup readiness"),
    ("ct_model_setup", MODEL, "Model::setup(CPUDevice &cpuDevice, const WeightsRegistry *weightsRegistry,", "cpp", 1, "Invoke the compiled init callable and retain returned state"),
    ("ct_device_map", "GraphCompiler/lib/Driver/Device/DeviceMap.cpp", "M::GraphCompiler::Driver::mapDeviceRefsToDevices(", "cpp", 1, "Synchronize driver contexts and resolve model DeviceRefs"),
    ("ct_module_load", "GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp", "mgp_module_load_cpu(Chain inChain, StateContext &theContext,", "cpp", 1, "Load compiled host routines into runtime state"),
    ("ct_kernel_load", "GraphCompiler/lib/Support/CompiledKernelsContext.cpp", "ErrorOrSuccess CompiledKernelsContext::load(ArrayRef<uint8_t> binary,", "cpp", 0, "Load/reuse a host archive and resolve callable symbols"),
    ("ct_wait_setup", MODEL, "ErrorOrSuccess ExecutableModel::waitForModelSetup()", "cpp", 0, "Await initialization chain, context, weights and extra arguments"),
    ("ct_model_handle", CORE, "194:206", "lines", 0, "ModelHandle retains native session, artifact and executable"),
]


DEFAULT_CONTEXT_EXPLANATION = """**Yes. `_DEFAULT_MLIR_CONTEXT` is a module-level global variable holding an `mlir.Context` object.** MAX creates it when this module is imported:

```python
_DEFAULT_MLIR_CONTEXT = mlir.Context()
```

The same default object is reused within that Python interpreter. See [_mlir_context.py:26](sources/max/python/max/_mlir_context.py#L26).

An **MLIR context** provides the shared infrastructure for representing compiler IR:

- **Dialects:** definitions of operations and their types.
- **Types:** shared representations such as `f32` and tensor types.
- **Attributes:** metadata attached to operations and types.

Multiple graphs can use the same context while keeping their own operations and graph structure. ([mlir.llvm.org](https://mlir.llvm.org/doxygen/classmlir_1_1MLIRContext.html))

These three names have different roles: one is a context **object**, one is the **per-thread active-context lookup**, and one is a **thread-pool executor** that scopes tasks with a context.

| Name | What it is | The question it answers | Lifetime / behavior |
|---|---|---|---|
| [`_DEFAULT_MLIR_CONTEXT`](sources/max/python/max/_mlir_context.py#L26) | One module-level `mlir.Context` object in this Python interpreter. | “Which context should MAX use as its fallback?” | Created at module import. MAX enters it on the importing thread and exits it at interpreter shutdown. Worker tasks can temporarily enter this same object. |
| [`mlir.Context.current`](sources/max/python/max/_mlir_context.py#L27) | A thread-local property; it is not another context object or a module global. | “Which context, if any, is active on **this OS thread right now**?” | Can be `None`, `_DEFAULT_MLIR_CONTEXT`, or another context. One thread's current value is not automatically copied to another thread. |
| [`MLIRThreadPoolExecutor`](sources/max/python/max/_mlir_context.py#L89) | MAX's subclass of Python's `ThreadPoolExecutor`. It is an executor, not a context. | “How should tasks submitted to this pool get a usable MLIR context?” | Its `submit()` wraps each task with `call_with_default_mlir_context`. On the worker thread, that wrapper keeps an already-current context or temporarily enters `_DEFAULT_MLIR_CONTEXT` if the worker has none. It does not create a context per task or copy the submitting thread's current context. |

The implementation and its thread-local note are in [_mlir_context.py:26–67](sources/max/python/max/_mlir_context.py#L26); the executor wrapper is in [_mlir_context.py:89](sources/max/python/max/_mlir_context.py#L89).

```text
Shared object: _DEFAULT_MLIR_CONTEXT
Main thread: module import enters it → mlir.Context.current on main points to this object
Worker task: MLIRThreadPoolExecutor.submit → call_with_default_mlir_context(fn)
  If current on that worker is None → temporarily enter the shared default → run fn → leave it
  If current on that worker is already set → preserve it → run fn
```

So “current” is the thread's active-context slot; “default” is the shared object that MAX can put in that slot when it is empty; the executor runs tasks on worker threads and applies that fallback around each submitted task. The executor is not a per-context mutex or semaphore. The conditional behavior is implemented in [ensure_default_mlir_context:55](sources/max/python/max/_mlir_context.py#L55) and [call_with_default_mlir_context:44](sources/max/python/max/_mlir_context.py#L44). **Entering a context makes it active for IR-building APIs; entering alone does not compile or execute the graph.** ([MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/))

For the `Graph(...)` call tree, the decorator applies the same rule directly around construction:

```text
Graph(...)
└─ @in_default_mlir_context
   ├─ This thread already has an active context
   │  └─ Keep that context; call Graph.__init__
   └─ This thread has no active context
      └─ Temporarily enter _DEFAULT_MLIR_CONTEXT
         └─ Call Graph.__init__
```

### A.0.1 · What does `@in_default_mlir_context` mean?

It is **Python decorator syntax**. `in_default_mlir_context` is a Python function; `@` applies it to the `__init__` function when the class body is created. It is not a Python attribute on `Graph` or on a graph instance.

This syntax:

```python
@in_default_mlir_context
def __init__(self, ...):
    ...
```

is equivalent in effect to:

```python
def __init__(self, ...):
    ...

__init__ = in_default_mlir_context(__init__)
```

The decorator returns a wrapper. When `Graph(...)` calls the wrapped constructor, the wrapper enters `ensure_default_mlir_context()`, calls the original constructor, and leaves the context manager when construction returns or raises. If this thread already has an active MLIR context, the helper keeps it instead of entering another one. See the [decorator implementation](sources/max/python/max/_mlir_context.py#L70).

### A.0.2 · Why mention a background thread?

A **background thread** (bg-thread) is another OS thread in the same Python process, often a worker created by a thread pool. It is not another process. `mlir.Context.current` is thread-local: the main thread's active-context setting is not automatically copied to a worker thread. MAX's source states this directly in [`ensure_default_mlir_context`](sources/max/python/max/_mlir_context.py#L54).

For example, a caller may construct a graph from a worker:

```python
from concurrent.futures import ThreadPoolExecutor

def make_graph():
    return Graph("addition", input_types=[...], forward=...)

with ThreadPoolExecutor() as workers:
    graph = workers.submit(make_graph).result()
```

When `Graph.__init__` runs on that worker, its decorator notices whether that worker has a current MLIR context. If not, it temporarily enters `_DEFAULT_MLIR_CONTEXT` so `builtin.ModuleOp`, `mo.GraphOp`, and related MLIR bindings run with the intended context. The scope ends with the constructor call. MAX also provides an [`MLIRThreadPoolExecutor`](sources/max/python/max/_mlir_context.py#L89) that wraps submitted tasks with the same context guard. The module comments say this scoped worker handling also avoids nanobind GIL/TLS teardown crashes ([context setup and comments](sources/max/python/max/_mlir_context.py#L26)).
"""


MOJO_PRECOMPILE_EXPLANATION = """### A.1.1 · Is `mojo` an executable? Where is it?

**Yes.** Here, `mojo` is the Mojo compiler's command-line driver executable. Python starts it as a separate child process with `subprocess.run`; `mojo` is not a Python class and is not just a shell alias. The actual argument vector is:

```text
[resolved-mojo-executable, "precompile", <kernels-directory>, "-o", <staging-.mojoc-path>]
```

There is **no single absolute path for every install**. The [SDK environment setup](sources/Mojo/python/mojo/run.py#L22) chooses `<get_package_root()>/bin/mojo` when MAX can find its installed package root. An ambient `MODULAR_MOJO_MAX_DRIVER_PATH` overrides that default. If the package root is unavailable (the helper calls this the Bazel case) and there is no override, [the launcher falls back to `shutil.which("mojo")`](sources/Mojo/python/mojo/run.py#L57), which searches `PATH`.

For a wheel install, the bundled path is therefore typically shaped like `<venv>/lib/pythonX.Y/site-packages/max/bin/mojo`; in a Bazel or source-run environment, it can be a runfiles/toolchain path or whatever `PATH` resolves. To print the path selected by the installed `mojo` Python helper in the current environment:

```bash
python -c 'from mojo.run import _mojo_env; import shutil; e = _mojo_env(); print(e.get("MODULAR_MOJO_MAX_DRIVER_PATH") or shutil.which("mojo"))'
```

The helper verifies that the resolved path exists, prepends it to the `precompile` arguments, and waits for that compiler process to finish before publishing P1.

### A.1.2 · What does the `add_one` MLIR bytecode look like?

First, distinguish the **file bytes** from a **readable MLIR listing**. The `.mojoc` file is an MPKG V2 package header plus a zstd-compressed MLIR bytecode payload; literal byte values are binary and do not display as a function listing. During this precompile step the payload is a `lit.package` containing post-parse **LIT-dialect** IR, including function bodies after semantic/lifetime checks. It has not yet been specialized for a concrete `target`/`width`, lowered through the kernel JIT, or emitted as CPU/GPU machine code. See the [repository's Mojo package walkthrough](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L612) and the [package writer](sources/Mojo/lib/Support/MojoPrecompiledFile.cpp#L128).

The exact source expression is in [`add_one.mojo`](sources/max/examples/custom_ops/kernels/add_one.mojo#L32). `execute` has a compile-time `target`; its helper has a compile-time `width`:

```mojo
def elementwise_add_one[
    width: Int
](idx: Coord) {var} -> SIMD[x.dtype, width]:
    return x.load[width](idx) + 1

foreach[target=target](elementwise_add_one, output, ctx)
```

The package's readable structure is conceptually:

```text
lit.package @kernels
  lit.struct.decl @AddOne
    lit.fn "execute[target: StaticString]"  [body retained]
      helper "elementwise_add_one[width: Int]"  [body retained]
        load x[idx] as SIMD → add 1 → return SIMD
      foreach[target=target](elementwise_add_one, output, ctx)
```

This is a **structural sketch, not a captured compiler dump**: the repository documents `lit.package`, `lit.struct.decl`, and `lit.fn` as the package IR shape, while the arrows summarize the source-level math rather than claiming exact LIT operation spellings. The actual `.mojoc` payload is binary. A literal byte dump or exact decoded `add_one` IR would require running this repository revision's compiler and capturing its output. See the [documented package shape](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L647).
"""


GRAPH_LOCATION_EXPLANATION = """**SOURCE · loc is an MLIR Location: metadata describing where an IR operation was created.** The Python variable <code>loc</code> holds the Location returned by <code>_location()</code>. The active MLIR Context owns its interned metadata; source locations help diagnostics and source-aware debugging. See [the helper](sources/max/python/max/graph/graph.py#L438) and [the native frame binding](sources/max/python/max/_core/max_license/modules/graph.cpp#L162).

**Inputs → decisions → output, in source order**

```text
Graph.__init__: with _location() as loc [LINK:graph_init:851]
└─ _location(ignore_frames=1, attach_profile_scopes=True) [LINK:ct_graph_location:438]
   ├─ Require an active mlir.Context.current [LINK:ct_graph_location:442]
   │  └─ absent → raise RuntimeError [LINK:ct_graph_location:443]
   ├─ Read the current source_tracebacks setting [LINK:ct_graph_location:449]
   │  ├─ disabled → Location.unknown() [LINK:ct_graph_location:450]
   │  └─ enabled → traceback.extract_stack() [LINK:ct_graph_location:457]
   │     ├─ Keep frame summaries; trim _location + ignore_frames callers [LINK:ct_graph_location:457]
   │     ├─ nonempty → _graph.frame_loc(current_context, tb) [LINK:ct_graph_location:459]
   │     │  └─ PythonFrameAttr::getOpaque → MLIR Location [LINK:ct_graph_frame_loc:165]
   │     └─ empty → Location.unknown() [LINK:ct_graph_location:461]
   ├─ If attach_profile_scopes: wrap active name/color scopes [LINK:ct_graph_location:464]
   │  └─ _graph.profile_scope_location(context, name, location, color) [LINK:ct_graph_location:472]
   └─ Return location → the variable loc [LINK:ct_graph_location:475]
```

**What <code>with</code> does next**

The Location context manager makes <code>loc</code> current for the block. MAX also passes that same object explicitly to [<code>builtin.ModuleOp(location=loc)</code>](sources/max/python/max/graph/graph.py#L856) and [<code>_mo.GraphOp(builder, loc, ...)</code>](sources/max/python/max/graph/graph.py#L860). The native [GraphOp builder](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) attaches its operation location to every input block argument. Exiting <code>with</code> restores the surrounding current Location; the created IR retains its attached metadata. ([MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/))

**Add-one example · a Python stack becomes IR provenance**

The graph is requested at [addition.py:33](sources/max/examples/custom_ops/addition.py#L33). With source tracebacks enabled, the helper collects the stack and removes its final two frames with the default <code>ignore_frames=1</code>:

~~~text
Before trimming, illustrative frame summaries:
  ... → addition.py:33 → ... → Graph.__init__:851 → _location:457

After trimming:
  ... → addition.py:33 → ...

frame_loc(context, remaining summaries) → loc
  loc → builtin.ModuleOp
  loc → mo.GraphOp("addition") → symbolic input argument X
~~~

The ellipses allow surrounding Python/decorator frames. This is a source-derived example of retained provenance, rather than an executed stack capture or an exact printed MLIR dump. The native wrapper stores the summaries in an opaque MLIR source Location. Profiling scopes can wrap either the stack-derived or unknown Location, even when source tracebacks are disabled. See [frame_loc](sources/max/python/max/_core/max_license/modules/graph.cpp#L162) and [scope wrapping](sources/max/python/max/graph/graph.py#L464).
"""


GRAPH_MODULE_SETUP_EXPLANATION = """**SOURCE · self._module holds the selected builtin.ModuleOp: the IR container for this graph.** A public Python <code>Module</code> wrapper exposes that operation through <code>module.mlir_module</code>. Its body can hold several named graph definitions, compiled together. See [Module's contract](sources/max/python/max/graph/graph.py#L562).

**Inputs → branch → retained module, in source order**

```text
Graph.__init__(..., module=module): inside with _location() as loc [LINK:graph_init:851]
└─ self._module = module.mlir_module if module else builtin.ModuleOp(location=loc) [LINK:graph_init:855]
   ├─ module supplied → read its mlir_module attribute [LINK:ct_graph_module:619]
   │  └─ retain the same typed builtin.ModuleOp as self._module [LINK:ct_graph_module:637]
   └─ module omitted → builtin.ModuleOp(location=loc) [LINK:graph_init:856]
      └─ nanobind ModuleOp constructor(location, name=None) [LINK:ct_builtin_module:62]
         └─ mlir::ModuleOp::create(loc, name) [LINK:ct_builtin_module:65]
            └─ new builtin.module → body region → initially empty body block [LINK:ct_builtin_module:68]
```

<code>loc</code> comes from the preceding <code>_location()</code> call. The fresh-module branch passes it to the native constructor. The reuse branch keeps the existing module and its attached location; the upcoming graph definition receives the current <code>loc</code> separately. See [the conditional assignment and GraphOp call](sources/max/python/max/graph/graph.py#L855).

**Add-one · no module argument**

The [example's Graph call](sources/max/examples/custom_ops/addition.py#L33) omits <code>module=</code>. At this row, the newly created module's body is empty. The next two source actions position the builder and insert the graph definition:

~~~text
At this row                         After GraphOp insertion
self._module                        self._module
  builtin.module                      builtin.module
    body region                         body region
      body block: []                      body block
                                            mo.graph @addition
                                              graph region / block
                                                symbolic argument X
~~~

These are schematic IR structures before forward records the computation. The module body contains graph definitions; the graph's own body contains its computation. See [GraphOp insertion](sources/max/python/max/graph/graph.py#L860) and [native graph-body creation](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L847).

**Reuse example · encoder and decoder share one compilation unit**

Excerpt of the [Module documentation example](sources/max/python/max/graph/graph.py#L588); graph-output code is omitted here:

~~~python
module = Module()
with Graph("encoder", input_types=[], module=module) as encoder:
    ...  # record encoder output
with Graph("decoder", input_types=[], module=module) as decoder:
    ...  # record decoder output
~~~

~~~text
module.mlir_module ────────┐
encoder._module ──────────┼── same builtin.module
decoder._module ──────────┘     body block
                                 mo.graph @encoder
                                 mo.graph @decoder
~~~

Both Graph objects retain the shared enclosing operation and their own graph definitions. Constructing the decoder appends a definition to the existing module; it preserves the encoder definition. [Module's documented example](sources/max/python/max/graph/graph.py#L609) hands this unit to <code>InferenceSession.load_all(module)</code>.
"""


GRAPH_OP_BUILDER_EXPLANATION = """**SOURCE · OpBuilder is a native MLIR operation builder, positioned after the module body's existing operations.** The Python constructor creates a C++ <code>mlir::OpBuilder</code>; its insertion point identifies a block and a position within that block. See [the MAX call site](sources/max/python/max/graph/graph.py#L858) and [the native constructor](sources/max/python/max/_core/max_license/_core.cpp#L612).

**Evaluate the expression left to right**

```text
builder = OpBuilder(self._module.body.end) [LINK:graph_init:858]
├─ self._module → selected builtin.ModuleOp [LINK:graph_init:855]
├─ .body → native ModuleOp body binding [LINK:ct_builtin_module:68]
│  └─ &self.getBodyRegion().front() → the module body block [LINK:ct_builtin_module:69]
├─ .end → native Block.end binding [LINK:ct_block_end:408]
│  ├─ cast the Python block view to mlir::Block* [LINK:ct_block_end:410]
│  └─ OpBuilder::InsertPoint(block, block->end()) [LINK:ct_block_end:411]
└─ OpBuilder(insert_point) → nanobind constructor [LINK:ct_op_builder:612]
   └─ std::make_unique<mlir::OpBuilder>(ip.getBlock(), ip.getPoint()) [LINK:ct_op_builder:614]
      └─ OUT: builder ready to insert operations in this module block [LINK:ct_op_builder:615]
```

<code>body.end</code> is a position after the last operation, including the empty-block case. Creating the builder sets this position; the next [<code>_mo.GraphOp(builder, loc, ...)</code>](sources/max/python/max/graph/graph.py#L860) call creates the graph at that position. The native [GraphOp builder](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820) fills the graph signature, region and input block arguments.

**Fresh add-one module · the first insertion**

~~~text
Before builder creation          Builder ready                 After GraphOp call
module body: []                  module body: []               module body:
                                ↑ insertion point at end        mo.graph @addition
~~~

**Shared module · append the decoder after the encoder**

~~~text
Before decoder construction         After decoder GraphOp insertion
module body                          module body
  mo.graph @encoder                    mo.graph @encoder
  ↑ end insertion point                mo.graph @decoder
~~~

The arrow marks the position after <code>@encoder</code>. The selected block is the outer module body, so the new graph is a sibling of the encoder. The graph's computation is recorded later in its own body through [<code>with self</code> and the forward callback](sources/max/python/max/graph/graph.py#L921).

**Input → output:** the existing module's body block and its end iterator become a native builder stored in the local Python variable <code>builder</code>. That builder is passed immediately to GraphOp construction. These diagrams describe symbolic IR construction, with no compilation or tensor execution at this step.
"""


GRAPH_TYPE_CONVERSION_EXPLANATION = """**SOURCE · t.to_mlir() turns a Python type description into the native MO tensor type used by the graph's IR.** In add-one, <code>t</code> is the caller's <code>TensorType</code> describing float32 elements, shape <code>[5, 10]</code>, and a selected device. The result is a <code>mo.TensorType</code> handle. See [the symbolic-type contract](sources/max/python/max/graph/type.py#L358) and [the conversion method](sources/max/python/max/graph/type.py#L430).

**Inputs → conversions → native type, in source order**

```text
Graph.__init__: input_types=[t.to_mlir() for t in input_types] [LINK:graph_init:864]
└─ TensorType.to_mlir(self=t) [LINK:ct_tensor_to_mlir:430]
   ├─ metadata = []; optionally append the layout attribute [LINK:ct_tensor_to_mlir:436]
   │  └─ add-one has _layout=None → empty metadata [LINK:ct_tensor_description:404]
   └─ return mo.TensorType(...) [LINK:ct_tensor_to_mlir:439]
      ├─ argument 1: self.shape.to_mlir() [LINK:ct_tensor_to_mlir:440]
      │  └─ Shape.to_mlir() → mosh.ShapeAttr(dim attributes, mosh.ShapeType()) [LINK:ct_shape_to_mlir:36]
      │     └─ for 5 and 10: StaticDim.to_mlir() [LINK:ct_static_dim_to_mlir:562]
      │        └─ signed i64 IntegerAttr → kgen.CastFromBuiltinAttr [LINK:ct_static_dim_to_mlir:570] [CALLER:ct_shape_to_mlir|39]
      ├─ argument 2: self.dtype → existing DType.float32 value [LINK:ct_tensor_to_mlir:441] [CALLER:ct_tensor_to_mlir|441]
      ├─ argument 3: self.device.to_mlir() [LINK:ct_tensor_to_mlir:442]
      │  └─ DeviceRef.to_mlir() → m.DeviceRefAttr(kind string, device id) [LINK:ct_device_ref_to_mlir:181] [CALLER:ct_tensor_to_mlir|442]
      └─ keyword metadata: builtin.DictionaryAttr(metadata) [LINK:ct_tensor_to_mlir:443] [CALLER:ct_tensor_to_mlir|443]
```

The method is called on each Python input type. For this example there is one input, so GraphOp receives a one-element list of native types. The native MO type builders derive their MLIR Context from the shape attribute and retain shape, dtype, device reference and metadata as the tensor type's fields. See [the context-inferred builder](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L324) and [the base type builder](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L304).

**Add-one example · before and after**

Illustrative Python using the same float32 <code>[5, 10]</code> input declaration as [addition.py](sources/max/examples/custom_ops/addition.py#L51), with CPU selected explicitly:

~~~python
t = TensorType(DType.float32, shape=[5, 10], device=DeviceRef.CPU())
ir_type = t.to_mlir()
input_types_for_graph_op = [ir_type]
~~~

| Python description | Native type field |
| --- | --- |
| <code>shape=[5, 10]</code> | <code>mosh.ShapeAttr</code> containing typed attributes for 5 and 10 |
| <code>dtype=DType.float32</code> | dtype field; displayed as <code>f32</code> in MLIR assembly |
| <code>device=DeviceRef.CPU()</code> | <code>m.DeviceRefAttr("cpu", 0)</code> |
| <code>_layout=None</code> | empty <code>builtin.DictionaryAttr</code> metadata |

~~~text
Python type description t
  TensorType(float32, [5, 10], cpu:0)
                ↓ t.to_mlir()
Native mo.TensorType handle ir_type
  shape + dtype + device reference + metadata
                ↓ passed in input_types=[ir_type]
mo.GraphOp("addition")
  input block argument X has this type
~~~

**Schematic IR view of that typed input** — this is an illustration, not a captured compiler dump:

~~~mlir
// CPU: the default device can be omitted in printed tensor types.
mo.graph @addition(%x: !mo.tensor<[5, 10], f32>) -> () {
  // Construction snapshot: forward/output have not been recorded yet.
}

// If the same description uses DeviceRef.GPU(0), the device field changes:
// %x: !mo.tensor<[5, 10], f32, gpu:0>
~~~

The [native tensor-type printer](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L219) prints these fields; [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) gives each input block argument its declared type. This operation converts type metadata: tensor numbers enter later through execution buffers, while <code>t</code> remains the Python description. Conversion itself does not emit a <code>.mojoc</code> package or run graph-compilation passes.

**1 · Shape: what happens to the numbers 5 and 10?**

The Python sizes describe the number of rows and columns. The compiler needs typed attributes for those sizes, not raw Python integers. [StaticDim.to_mlir](sources/max/python/max/graph/dim.py#L570) constructs each attribute; [Shape.to_mlir](sources/max/python/max/graph/shape.py#L39) collects them.

~~~text
StaticDim(5)                                   StaticDim(10)
  ↓ self.dim = 5                                 ↓ self.dim = 10
IntegerType(64, signed) = si64                   IntegerType(64, signed) = si64
  ↓                                              ↓
IntegerAttr(si64, 5)                            IntegerAttr(si64, 10)
  ↓                                              ↓
kgen.CastFromBuiltinAttr(the typed 5)           kgen.CastFromBuiltinAttr(the typed 10)
  └──────────────────────┬───────────────────────┘
                         ↓
mosh.ShapeAttr([dimension-expression 5, dimension-expression 10], mosh.ShapeType())
                         ↓
shape field of mo.TensorType → printed shape [5, 10]
~~~

**Read it as:** <code>si64</code> means a signed 64-bit integer <em>type</em>; <code>IntegerAttr(si64, 5)</code> is an immutable typed constant describing size 5; <code>CastFromBuiltinAttr</code> wraps that builtin constant for KGEN's dimension-expression representation. It does not cast the tensor's float32 elements to integers. This structural illustration is not the exact serialized bytecode or printed attribute spelling. A tensor with 5 rows still has float32 elements.

**2 · Dtype: why is self.dtype passed directly?**

~~~python
# Illustrative values already stored in the Python type description:
t.dtype                 # DType.float32
# TensorType.to_mlir supplies this existing value as argument 2:
mo.TensorType(shape_attr, t.dtype, device_attr, metadata=metadata_attr)
~~~

[The conversion](sources/max/python/max/graph/type.py#L441) passes the dtype value already chosen by the caller. The [native type builder](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L304) uses it for the element type; the [printer](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L219) displays float32 as <code>f32</code>. For <code>[5, 10]</code>, this describes 50 float32 elements when a tensor is supplied later. Choosing <code>DType.int32</code> instead would describe integer elements; this method does not convert any actual tensor data.

| Field | Meaning in this example |
| --- | --- |
| <code>IntegerAttr(si64, 5)</code> | Row-count metadata: the size is 5 |
| <code>DType.float32</code> / <code>f32</code> | Element datatype: each tensor value is float32 |

**3 · Device: what becomes cpu:0 or gpu:0?**

[DeviceRef.to_mlir](sources/max/python/max/graph/type.py#L181) takes the symbolic device kind and index:

~~~text
DeviceRef.CPU()   → self.device_type="cpu", self.id=0 → m.DeviceRefAttr("cpu", 0)
DeviceRef.GPU(0)  → self.device_type="gpu", self.id=0 → m.DeviceRefAttr("gpu", 0)
DeviceRef.GPU(1)  → self.device_type="gpu", self.id=1 → m.DeviceRefAttr("gpu", 1)
~~~

The attribute labels the graph tensor's device. A <code>gpu:0</code> label is not a GPU buffer or a driver handle; forming it does not allocate or transfer tensor data. The same <code>[5, 10]</code> / <code>f32</code> type can therefore carry either a CPU or GPU reference. The tensor-type printer omits the default <code>cpu:0</code> field and prints a nondefault GPU field.

**4 · Metadata: a concrete convolution layout example**

**Add-one:** <code>_layout=None → metadata=[] → builtin.DictionaryAttr([])</code>. There is no convolution-filter layout attached to its input. The optional layout path is useful for convolution filters; [FilterLayout](sources/max/python/max/graph/type.py#L38) describes their memory layout.

~~~python
# Illustrative internal type description for a convolution filter:
filter_type = TensorType(
    DType.float32, [3, 3, 16, 32], device=DeviceRef.CPU(),
    _layout=FilterLayout.RSCF,
)
# to_mlir() builds:
layout_attr = FilterLayout.RSCF.to_mlir()  # mo.LayoutAttr(format_str="RSCF")
metadata = [NamedAttribute("layout", layout_attr)]
metadata_attr = builtin.DictionaryAttr(metadata)
# metadata_attr is argument metadata=... of mo.TensorType(...)
~~~

[conv2d's documented order](sources/max/python/max/graph/ops/conv.py#L44) is <code>R, S, C, F = height, width, input channels / groups, output channels</code>. With groups=1, this example is a 3×3 filter with 16 input channels and 32 output channels. The <code>layout</code> key tells compiler operations how to interpret the axes, separately from element dtype and device.

~~~text
Python _layout=FilterLayout.RSCF
  → mo.LayoutAttr("RSCF")
  → DictionaryAttr {layout: the RSCF attribute}
  → metadata field on native mo.TensorType
  → TensorType::getLayoutAttr() reads metadata["layout"]
  → filter-packing shape calculation reads the format:
       RSCF: F is last axis → [3, 3, 16, 32] → F = 32
       FCRS: F is first axis → [32, 16, 3, 3] → F = 32
~~~

The [native metadata reader](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L454) obtains this key; the [filter-packing shape function](sources/GraphCompiler/lib/MODialect/PackingShapeFunctions.cpp#L279) uses the layout to choose the F axis. [conv2d](sources/max/python/max/graph/ops/conv.py#L141) attaches the requested filter layout through <code>filter._with_layout(filter_layout)</code>. Merely attaching metadata does not transpose a buffer. RSCF and FCRS examples describe different axis orders; a real reorder requires the appropriate operation.

**Small extension · a symbolic dimension**

For <code>TensorType(DType.float32, ["batch", 10], device=DeviceRef.CPU())</code>, the shape remains rank two. Instead of fixing the first size, [SymbolicDim.to_mlir](sources/max/python/max/graph/dim.py#L286) creates <code>kgen.ParamDeclRefAttr("batch", si64)</code>. The second dimension still follows the static-size conversion above. The graph later declares its named dimension parameters from these types.
"""


GRAPH_NATIVE_BUILDER_EXPLANATION = """**SOURCE · GraphOp creates one region and one body block; each input adds an argument to that same block.** The region is created by the builder, not passed in as existing region information. See [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820).

**Native calls in source order**

```text
Graph.__init__ → mo.GraphOp(builder, loc, name, input_types, result_types=[]) [LINK:graph_init:860]
└─ GraphOp::build(builder, state, name, inputTypes, resultTypes, isSubgraph) [LINK:ct_graph_op_build:820]
   ├─ Set symbol name: "addition" [LINK:ct_graph_op_build:825] [CALLER:ct_graph_op_build|825]
   ├─ builder.getFunctionType(inputTypes, resultTypes) → ordered input/output types [LINK:ct_graph_op_build:831] [CALLER:ct_graph_op_build|831]
   ├─ Wrap function type in KGEN::FuncTypeGeneratorType; attach signature and functionType [LINK:ct_graph_op_build:832] [CALLER:ct_graph_op_build|832]
   ├─ Collect named dimension parameters from input/result shapes [LINK:ct_graph_op_build:839] [CALLER:ct_graph_op_build|839]
   ├─ state.addRegion() → new empty body region owned by GraphOp [LINK:ct_graph_op_build:846] [CALLER:ct_graph_op_build|846]
   ├─ bodyRegion->emplaceBlock() → ONE block inside that region [LINK:ct_graph_op_build:847] [CALLER:ct_graph_op_build|847]
   └─ for each input type: body.addArgument(input, state.location) [LINK:ct_graph_op_build:848]
      └─ append a typed argument to the SAME block; attach loc [LINK:ct_graph_op_build:849] [CALLER:ct_graph_op_build|849]
```

**Where does the region come from?** <code>OperationState</code> is the operation being assembled. <code>state.addRegion()</code> creates its nested body container; <code>emplaceBlock()</code> creates the block that will hold the computation. The later forward callback inserts operations into that body. Input types provide the arguments' types; they do not provide a region or a block per input.

~~~text
Example with TWO user inputs (schematic, not an executed dump):
inputTypes = [float32[5,10], float32[5,10]]
                         ↓
builtin.module
└─ mo.graph @pair                 ← graph operation
   └─ body region                ← created ONCE by addRegion()
      └─ entry block             ← created ONCE by emplaceBlock()
         ├─ block argument %x : float32[5,10]  ← first addArgument()
         ├─ block argument %y : float32[5,10]  ← second addArgument()
         └─ operations inserted later: use %x and %y in this block
~~~

**Why one argument per input?** Each argument gives the computation a separate symbolic value for an input position. For a two-input addition, the body needs both <code>%x</code> and <code>%y</code>. Arguments are values, not blocks. A block contains an ordered sequence of operations; splitting the inputs into two blocks would not represent the simple two-input computation shown here. In the actual add-one example, there is only one declared tensor input, so the native builder initially adds one tensor argument.

**Why keep a signature?** It is the ordered type contract for entering and leaving the graph. It tells the compiler the number and types of inputs/results, including shapes, dtypes and devices carried by those types. The builder stores a function type and its KGEN generator signature; the signature is metadata about the computation, while the body contains the computation itself.

~~~text
Add-one lifecycle (T = !mo.tensor<[5, 10], f32>):
Shell construction:    inputTypes=[T], resultTypes=[] → function type (T) → ()
Forward recording:     %x : T → mo.custom "add_one" → %result : T
Graph.output(result):  update output contract        → function type (T) → (T)
                       attach generator signature + mo.output terminator
~~~

[Graph.output](sources/max/python/max/graph/graph.py#L1455) constructs the function type from the body-argument types and output-value types, then updates <code>signature</code> and <code>functionType</code> before inserting the output operation. The empty result list is a construction-stage snapshot, not the final add-one output contract.

~~~mlir
// Schematic two-input computation, after forward/output recording.
// Block arguments are printed in the graph header; ^entry is implicit here.
mo.graph @pair(%x: !mo.tensor<[5, 10], f32>,
               %y: !mo.tensor<[5, 10], f32>) -> !mo.tensor<[5, 10], f32> {
  // Recorded operations consume %x and %y and produce %sum.
  // The final mo.output returns %sum using the result type in the signature.
}
~~~

These examples illustrate IR structure and contracts, not executed tensor arithmetic. The source-linked tree above shows the exact native construction calls.
"""


GRAPH_SYMBOLIC_PARAMS_EXPLANATION = """**SOURCE · Declare named shape parameters, not extra tensor inputs.** The constructor first collects <code>SymbolicDim</code> names from input shapes, then declares those names on GraphOp. See [discovery](sources/max/python/max/graph/graph.py#L840) and [declaration](sources/max/python/max/graph/graph.py#L868).

```text
Graph.__init__(input_types=...) [LINK:graph_init:840]
├─ scan TensorType/BufferType shapes; collect only SymbolicDim.name [LINK:graph_init:843] [CALLER:graph_init|840]
├─ dict.fromkeys(names) → unique names in first-seen order [LINK:graph_init:840] [CALLER:graph_init|840]
├─ si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True)) [LINK:graph_init:868] [CALLER:graph_init|868]
└─ op.input_parameters = ParamDeclArrayAttr([ParamDeclAttr(p, si64) for p in self._params]) [LINK:graph_init:870] [CALLER:graph_init|870]
```

~~~text
Static add-one input:        TensorType(float32, [5, 10])
  discovered names          []
  self._params              {}
  op.input_parameters       empty ParamDeclArrayAttr
  tensor block arguments    one: %x : float32[5,10]

Illustrative dynamic input: TensorType(float32, ["batch", 10])
  discovered names          ["batch"]
  self._params              {"batch": None}
  op.input_parameters       [ParamDeclAttr("batch", si64)]
  tensor block arguments    one: %x : float32[batch,10]
~~~

**Input → output:** a named shape description becomes a typed parameter declaration; the actual size is not chosen by this constructor. If a later input tensor has shape <code>[8, 10]</code>, 8 is the concrete batch-size example. The declaration itself is not a second tensor or a data allocation. The KGEN <code>si64</code> here is a one-lane signed 64-bit parameter type; it does not change float32 tensor elements.

[SymbolicDim.to_mlir](sources/max/python/max/graph/dim.py#L286) creates the corresponding named parameter reference in the tensor shape. Declaring <code>batch</code> gives that reference a name/type definition. For two tensor types <code>["batch", 10]</code> and <code>["batch", 20]</code>, <code>dict.fromkeys</code> declares <code>batch</code> once while both shapes refer to it. Static sizes 5 and 10 remain dimension constants.
"""

GRAPH_RETAIN_BODY_EXPLANATION = """**SOURCE · `_CAPICreate(op._CAPIPtr)` bridges two Python binding families around the same native GraphOp.** `op` is MAX's `_core` GraphOp wrapper. `mlir.Operation` is the MLIR Python Operation class [re-exported by MAX](sources/max/python/max/mlir/__init__.py#L19). The bridge creates a Python wrapper, not another graph, region, block, or IR copy.

**CAPI = C Application Programming Interface.** MLIR exposes C-compatible handles and functions around its C++ implementation. `MlirOperation` is a small opaque handle containing `void *ptr`; it is not the serialized graph or its tensor data. A Python capsule carries that pointer with a type-specific name so the receiving binding can extract the correct kind of handle. [Opaque structs](sources/external/llvm-project/mlir/include/mlir-c/IR.h#L45), [capsule contract](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L88).

**Exact constructor line, expanded in evaluation order**

```text
self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr) [LINK:graph_init:874]
├─ Evaluate op._CAPIPtr first: MAX _core property getter [LINK:ct_capi_export:607]
│  ├─ self.getOperation() → native mlir::Operation * for the existing mo.graph [LINK:ct_capi_export:609] [CALLER:ct_capi_export|609]
│  ├─ wrap(native pointer) → MlirOperation{ptr = same pointer} [LINK:capi_wrap:25] [CALLER:ct_capi_export|609]
│  └─ mlirPythonOperationToCapsule(handle) → PyCapsule_New(handle.ptr, operation capsule name, NULL) [LINK:capi_capsule:330] [CALLER:ct_capi_export|609]
├─ Call mlir.Operation._CAPICreate(capsule) [LINK:capi_factory_bind:4486]
│  └─ PyOperation::createFromCapsule(capsule) [LINK:capi_factory:1207]
│     ├─ mlirPythonCapsuleToOperation → PyCapsule_GetPointer(capsule, expected operation name) [LINK:capi_capsule:338] [CALLER:capi_factory|1208]
│     ├─ null extracted handle? throw nb::python_error() [LINK:capi_factory:1209] [CALLER:capi_factory|1209]
│     ├─ mlirOperationGetContext(rawOperation) → wrap(unwrap(op)->getContext()) [LINK:capi_context_get:695] [CALLER:capi_factory|1211]
│     └─ forOperation(PyMlirContext::forContext(rawCtxt), rawOperation) [LINK:capi_factory:1212]
│        ├─ createInstance → allocate Python PyOperation wrapper around the same handle [LINK:capi_wrapper_create:948] [CALLER:capi_wrapper_create|964]
│        └─ releaseObject() → return Python object; not release/destroy native IR [LINK:capi_factory:1213] [CALLER:capi_factory|1212]
├─ assign returned Python wrapper to self._mlir_op [LINK:graph_init:874] [CALLER:graph_init|874]
├─ self._current_block = self._mlir_op.regions[0].blocks[0] [LINK:graph_init:875] [CALLER:graph_init|875]
└─ self._graph_body = self._current_block [LINK:graph_init:876] [CALLER:graph_init|876]
```

**A small pointer example** — `P` is a symbolic address label, not a measured address:

~~~text
Native IR already exists:
MODULE1: builtin.module
└─ OP1: mo.graph @addition at address P
   └─ region[0] → block[0] → input %x

MAX _core GraphOp wrapper op ──→ OP1 at P
           ↓ op._CAPIPtr
PyCapsule {name=<qualified operation name>, pointer=P, destructor=NULL}
           ↓ mlir.Operation._CAPICreate(capsule)
MLIR Python PyOperation wrapper ──→ OP1 at the SAME P
           ↓ assigned to G1._mlir_op
G1._current_block and G1._graph_body ──→ the SAME block[0]

Native graph count: 1 throughout. Binding wrappers differ; graph identity does not.
~~~

The property name `_CAPIPtr` is defined by [MLIR's interop header](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L97). Its value is a capsule, not a Python integer address. `_CAPICreate` is a bound static factory method, [mapped to createFromCapsule](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L4486). These are Python extension calls in the same process; no subprocess, HTTP request, compiler invocation or machine-code lowering occurs here.

**What the C API looks like internally** — these short structural examples expand the actual macros/helpers:

~~~cpp
// mlir-c/IR.h exposes an opaque C handle:
struct MlirOperation { void *ptr; };
// CAPI/Wrap.h + CAPI/IR.h generate pointer adapters:
MlirOperation wrap(mlir::Operation *p) { return MlirOperation{p}; }
mlir::Operation *unwrap(MlirOperation h) {
  return static_cast<mlir::Operation *>(h.ptr);
}
// Python Interop.h carries that same pointer across binding families:
PyCapsule_New(h.ptr, MLIR_PYTHON_CAPSULE_OPERATION, NULL);
PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_OPERATION);
~~~

[wrap/unwrap macro](sources/external/llvm-project/mlir/include/mlir/CAPI/Wrap.h#L25), [Operation specialization](sources/external/llvm-project/mlir/include/mlir/CAPI/IR.h#L29), [capsule export/import](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L330). The qualified capsule name includes the binding package prefix; the source defines its suffix as `ir.Operation._CAPIPtr`. The producer and consumer must agree on that name. A wrong-kind capsule fails extraction and raises a Python error; a plain Python integer is not a correctly named capsule. The name check does not prove pointer liveness; the native operation must still be valid.

**Lifetime and identity**

| Item | What it retains or owns here |
| --- | --- |
| `MODULE1` / enclosing module | The attached native GraphOp and its region/block |
| MAX `_core` wrapper `op` | A binding view of the existing native GraphOp |
| `_CAPIPtr` capsule | Pointer + name; the export supplies no destructor or keep-alive owner |
| MLIR `PyOperation` wrapper | The C API operation handle plus a Python MLIR-context reference |
| `G1._mlir_op` | A Python reference to the returned wrapper |

**The capsule itself does not keep the native operation alive.** Keep the enclosing module/graph alive while using the wrappers. This constructor already retains `self._module`. [createFromCapsule](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1207) does not clone, erase, detach, or rename the GraphOp. [PyOperation's destructor](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L921) distinguishes an attached operation from a detached operation owned by Python. Do not generalize the lifetime rules of this attached GraphOp bridge to other factories such as Context/Module: the interop header gives those factories their own ownership contract.

**What the retained body is used for next**

`self._mlir_op.regions[0].blocks[0]` retrieves the existing graph entry block through the new wrapper. Both `_current_block` and `_graph_body` initially reference it. [_add_op_generated](sources/max/python/max/graph/graph.py#L1258) inserts later generated operations at the current block's end; [Graph.inputs](sources/max/python/max/graph/graph.py#L942) wraps the body arguments as symbolic Python values. None of those lookups executes tensor arithmetic.

**Evidence:** MAX export and constructor sources are pinned to the page's MAX revision. The complete upstream MLIR files embedded in Source Explorer are separately pinned to LLVM `6941494e2ae550d753728ba13454311b344d82cf`, the [LLVM_COMMIT selected by that MAX revision](sources/bazel/public-patches/llvm_source.bzl#L5). Source Explorer labels LLVM spans with their dependency revision. These are source-derived implementation examples, not proof that a particular installed wheel was rebuilt or executed.
"""


GRAPH_DEVICE_INFO_EXPLANATION = """**SOURCE · Add target-description metadata to the enclosing MLIR module.** This is separate from the device label on each tensor type and from the session's runtime devices. See [_populate_device_info_mapping](sources/max/python/max/graph/graph.py#L1366).

```text
Graph.__init__ → self._populate_device_info_mapping() [LINK:graph_init:877]
└─ Graph._populate_device_info_mapping() [LINK:ct_graph_device:1366]
   ├─ module = self._mlir_op.block.owner → enclosing builtin.module [LINK:ct_graph_device:1368] [CALLER:ct_graph_device|1368]
   ├─ existing mo.device_info_mapping? return immediately [LINK:ct_graph_device:1369] [CALLER:ct_graph_device|1369]
   ├─ devices = [CPU()]; if accelerator_count() > 0, append Accelerator() [LINK:ct_graph_device:1371] [CALLER:ct_graph_device|1371]
   ├─ read architecture_name and model_name; each falls back to "unknown" on error [LINK:ct_graph_device:1377] [CALLER:ct_graph_device|1375]
   ├─ _DeviceInfoAttr(label, api, arch, model, tile_based_fusion=False) [LINK:ct_graph_device:1385] [CALLER:ct_graph_device|1385]
   └─ entries keyed by dev.label → module attribute mo.device_info_mapping = mlir.DictAttr.get(entries) [LINK:ct_graph_device:1393] [CALLER:ct_graph_device|1393]
```

~~~text
Illustrative mapping, with placeholder values rather than measured hardware:
builtin.module attributes
└─ mo.device_info_mapping
   ├─ "cpu:0" → {label, api, arch="<host architecture>", model="<host model>",
   │             tile_based_fusion=false}
   └─ "gpu:0" → {label, api, arch="<detected architecture>", model="<GPU model>",
                 tile_based_fusion=false}   ← only if an accelerator is detected
~~~

| Item | What it describes |
| --- | --- |
| Tensor <code>DeviceRefAttr("gpu", 0)</code> | This tensor's symbolic device reference |
| Module <code>mo.device_info_mapping</code> | Device labels mapped to target-description attributes |
| <code>InferenceSession(devices=[device])</code> | Runtime devices selected for that session |

**Important source limit:** this helper starts with CPU and adds one default <code>Accelerator()</code> when any accelerator is available; it does not enumerate every GPU in a multi-GPU machine. A supplied shared module with an existing mapping takes the early return. This method records metadata; it does not compile a kernel or submit graph execution.
"""

GRAPH_TRACKING_EXPLANATION = """**SOURCE · Initialize two separate kinds of graph bookkeeping: symbolic weights and dependency chains.** See [constructor initialization](sources/max/python/max/graph/graph.py#L879).

```text
Graph.__init__ [LINK:graph_init:879]
├─ self._weights = {} → initially no named weight declarations [LINK:graph_init:879] [CALLER:graph_init|879]
├─ self._has_chain_input = False → no adopted incoming chain yet [LINK:graph_init:880] [CALLER:graph_init|880]
└─ self.device_chains = _DeviceChainMap(self) [LINK:graph_init:881]
   ├─ OrderedDict.__init__() → initially empty mapping [LINK:ct_device_chain_map:127] [CALLER:ct_device_chain_map|127]
   └─ retain self._graph = graph for later lazy insertion [LINK:ct_device_chain_map:129] [CALLER:ct_device_chain_map|129]
```

~~~text
Immediately after these lines:
_weights          = {}
_has_chain_input  = False
device_chains     = {}        ← next constructor step seeds the host chain
~~~

**Weights example:** add-one needs no model weights, so its registry stays empty. A different graph could register a symbolic weight named <code>projection.weight</code>. [Graph.add_weight](sources/max/python/max/graph/graph.py#L1619) checks the name, records a <code>mo.constant_external</code> declaration, and stores <code>_GraphWeight(weight, weight_tensor)</code> under that name. Reusing the same weight object returns an existing declaration; a different object with the same name raises an error. This Python map describes graph weights; it is not the runtime <code>WeightsRegistry</code> populated during model initialization.

**Lazy chains example (after the next step has seeded CPU):**

~~~text
Stored map:                  {cpu:0 → CH0}
Read device_chains[gpu:0]:    missing key → _DeviceChainMap.__getitem__()
Normal add-one graph:        _has_chain_input=False
                             → reuse current CPU chain as the seed
Stored map becomes:          {cpu:0 → CH0, gpu:0 → CH0}
~~~

[The lazy lookup](sources/max/python/max/graph/graph.py#L135) registers a missing device on access. In a graph with incoming chains at its top-level body, it can instead add a chain block argument. Merely constructing the map creates no GPU chain, thread, queue, lock or GPU allocation. The map holds symbolic ordering values for later graph operations.
"""

GRAPH_READY_CHAIN_EXPLANATION = """**SOURCE · Create an initially ready IR chain with no input dependencies.** A chain is an internal value used to express ordering of side effects. It is not tensor data, a mutex or a semaphore.

```text
Graph.__init__: create always-ready chain [LINK:graph_init:886]
├─ _add_op_generated(ChainCreateOp, result=ChainType(), inputs=[], attach_profile_scopes=False) [LINK:graph_init:887]
│  ├─ builder at self._current_block.end → insert and verify the op [LINK:ct_generated_op:1258] [CALLER:ct_generated_op|1258]
│  └─ return wrappers for op.results; constructor selects result [0] [LINK:ct_generated_op:1277] [CALLER:graph_init|886]
├─ _ChainValue(result[0]) → self._always_ready_chain = CH0 [LINK:graph_init:886] [CALLER:graph_init|886]
└─ _update_chain(CH0) → device_chains[DeviceRef.CPU()] = CH0 [LINK:ct_update_chain:1064] [CALLER:graph_init|894]
```

~~~text
IR dependencies (schematic):
[] ──→ mo.chain.create ──→ CH0 : !mo.chain
                          ├─ _always_ready_chain retains CH0
                          └─ device_chains[cpu:0] initially points to CH0

Later side-effect ordering example:
CH0 → side effect A → CH1 → side effect B → CH2
      device_chains[cpu:0] can advance to the latest chain
      _always_ready_chain remains CH0
~~~

**Why ready?** [ChainCreateOp's definition](sources/GraphCompiler/include/GraphCompiler/MODialect/MOOps.td#L4804) says its result depends on all input values. Here <code>inputs=[]</code>, so this initial chain has no producer dependencies to wait for. Graph construction records that property; it does not perform the side effects now.

**Why keep CH0 separately?** The [constructor comment](sources/max/python/max/graph/graph.py#L883) says it is never advanced by the graph and is used for operations safe to schedule without per-device ordering constraints, such as host-to-device staging transfers. The mutable host-chain map can advance independently. An always-ready dependency does not make a transfer itself complete or guarantee hardware timing.

**General ChainCreate example:** with <code>inputs=[CH1, tensor_result]</code>, the produced chain depends on both producers. This expresses a join of IR dependencies. The constructor's empty-input version is the initial seed. <code>attach_profile_scopes=False</code> avoids attaching an active profiling scope to this bookkeeping operation.
"""

GRAPH_TRAILING_CHAIN_EXPLANATION = """**SOURCE · If the final declared input is a chain, adopt it as the host ordering input.** The constructor checks the last block argument's type; it does not add a trailing chain to add-one.

```text
Graph.__init__: examine graph-body arguments [LINK:graph_init:896]
├─ no arguments? skip the check [LINK:graph_init:896] [CALLER:graph_init|896]
├─ wrap self._graph_body.arguments[-1] as a native Value [LINK:graph_init:897] [CALLER:graph_init|897]
├─ _is_chain_value(value) → isinstance(value.type, mo.ChainType) [LINK:ct_is_chain:119] [CALLER:graph_init|900]
└─ if True: _has_chain_input=True; convert with _ChainValue.from_mlir and _update_chain [LINK:graph_init:901] [CALLER:graph_init|901]
```

~~~text
Actual add-one:
body arguments = [%x : float32[5,10]]
last argument  = %x → tensor type → not a chain
_has_chain_input stays False
device_chains[cpu:0] stays CH0 (the new empty-input chain)

Subgraph-style example:
body arguments = [%x : float32[5,10], %incoming : !mo.chain]
last argument  = %incoming → chain type → adopt it
_has_chain_input becomes True
device_chains[cpu:0] = %incoming
_always_ready_chain still holds CH0
~~~

**Where can an incoming chain come from?** [Graph.add_subgraph](sources/max/python/max/graph/graph.py#L1027) constructs the child with <code>input_types=[*input_types, _ChainType()]</code>. The extra block argument lets the child carry an ordering dependency from its caller. The child graph's construction recognizes that chain; construction does not execute the caller or child.

**Why separate it from tensor inputs?** [Graph.inputs](sources/max/python/max/graph/graph.py#L943) excludes internal trailing chain values before exposing user inputs to the forward callback. In the simple child example, the lambda gets symbolic <code>x</code>, not the extra chain. Per-device chain bookkeeping can extend this pattern; the host chain is seeded from the explicit trailing chain instead of adding a duplicate host-chain argument.
"""


GRAPH_FORWARD_CONTEXT_EXPLANATION = """**SOURCE · `with self` enters Python's context-manager protocol and makes G1 the active graph.** It does not execute a compiled model. [Constructor call](sources/max/python/max/graph/graph.py#L921).

```text
with self [LINK:graph_init:921]
├─ Graph.__enter__ → push self._enter() context manager and enter it [LINK:ct_graph_context_manager:1135] [CALLER:graph_init|921]
└─ Graph._enter() [LINK:ct_graph_enter:1142]
   ├─ ensure_default_mlir_context() → make MLIR construction context available on this thread [LINK:ct_graph_enter:1144] [CALLER:ct_graph_enter|1144]
   ├─ token = CURRENT_GRAPH.set(self) → active graph is G1 [LINK:ct_graph_enter:1145] [CALLER:ct_graph_enter|1145]
   ├─ yield self → constructor runs its forward-building body [LINK:ct_graph_enter:1147] [CALLER:ct_graph_enter|1147]
   └─ finally CURRENT_GRAPH.reset(token) → restore earlier graph context [LINK:ct_graph_enter:1149] [CALLER:ct_graph_enter|1149]
```

~~~text
Before with G1:   CURRENT_GRAPH = previous graph or unset
Inside with G1:   CURRENT_GRAPH = G1 → ops.custom finds G1
After with G1:    previous graph or unset is restored
~~~

`Graph.current` is a Python class property backed by a ContextVar, not a newly constructed graph. [Its getter](sources/max/python/max/graph/graph.py#L1215) reads `CURRENT_GRAPH`; an unset value raises `LookupError("No graph found")`. The active graph selects the destination for operations. The MLIR Context instead owns/types IR objects. Nested graph contexts restore the previous graph when they exit, including exceptional exits.
"""

GRAPH_FORWARD_INPUTS_EXPLANATION = """**SOURCE · Wrap the graph's declared block arguments as Python symbolic values.** No matrix data is supplied at this step.

```text
forward(*self.inputs, *args, **kwargs) [LINK:graph_init:922]
└─ cached Graph.inputs property [LINK:ct_graph_inputs:935]
   ├─ body_args = self._graph_body.arguments [LINK:ct_graph_inputs:942] [CALLER:ct_graph_inputs|942]
   ├─ omit internal trailing chain values when the graph has chain inputs [LINK:ct_graph_inputs:943] [CALLER:ct_graph_inputs|943]
   └─ Value.from_mlir(_Value._from_cmlir(arg)) for each argument → tuple of symbolic values [LINK:ct_graph_inputs:949] [CALLER:ct_graph_inputs|949]
```

~~~text
G1 body:  %arg0 : !mo.tensor<[5,10], f32>
                         ↓ native value wrapped by Python Value.from_mlir
self.inputs = (TensorValue referring to %arg0,)
                         ↓ tuple expanded as forward(*self.inputs)
lambda parameter x refers to that same symbolic input
~~~

**Example:** `x.dtype` is float32 and `x.tensor.shape` describes `[5,10]`. There are still no 50 matrix values in `x`. The tuple has one element because add-one declares one user input; two declared tensor inputs would give two symbolic arguments to a two-parameter callback. The `cached_property` retains the result of this input-wrapper lookup.
"""

GRAPH_FORWARD_CALLBACK_EXPLANATION = """**SOURCE · The constructor calls the Python lambda immediately with symbolic x.** This executes graph-building Python, not the add-one Mojo kernel. [Lambda](sources/max/examples/custom_ops/addition.py#L37), [call](sources/max/python/max/graph/graph.py#L922).

```text
Graph.__init__ → result = forward(*self.inputs, *args, **kwargs) [LINK:graph_init:922]
└─ addition.py forward=lambda x: (...) [LINK:entry:37]
   ├─ prepare ops.custom arguments: name, device, values=[x] [LINK:entry:39] [CALLER:entry|38]
   ├─ TensorType(dtype=x.dtype, shape=x.tensor.shape, device=DeviceRef.from_device(device)) [LINK:entry:43] [CALLER:entry|43]
   └─ call ops.custom(...)[0].tensor → one symbolic result returned to constructor [LINK:entry:49] [CALLER:entry|38]
```

~~~text
INPUT x: symbolic float32[5,10], selected device
  → output Python TensorType: float32[5,10], same selected device
  → out_types=[that type] passed to ops.custom
  → type.to_mlir(): native MO result type
  → future result %r : float32[5,10]
~~~

The output TensorType is a declaration of the result contract. Copying x's shape/dtype does not allocate an output tensor or add 1 to its elements. `ops.custom` uses the declared types to construct a symbolic call. The registered Mojo implementation becomes executable through later compilation and initialization; real data enters through model execution.
"""

GRAPH_CUSTOM_ATTRIBUTES_EXPLANATION = """**SOURCE · Convert the Python call's name and device into attributes, and its output descriptions into native types.** [ops.custom](sources/max/python/max/graph/ops/custom.py#L69).

```text
ops.custom("add_one", device, values=[x], out_types=[T]) [LINK:custom_api:69]
├─ graph = Graph.current; context = default_mlir_context() [LINK:custom_api:94] [CALLER:custom_api|94]
├─ symbol_attr = StringAttr.get(name, context) → "add_one" [LINK:custom_api:96] [CALLER:custom_api|96]
├─ device = DeviceRef.from_device(device) [LINK:custom_api:97] [CALLER:custom_api|97]
├─ reject buffers/opaque values for this functional custom-op path; normalize inputs [LINK:custom_api:99] [CALLER:custom_api|99]
├─ [t.to_mlir() for t in out_types] → declared native result types [LINK:custom_api:110] [CALLER:custom_api|109]
└─ after insertion: custom_op.device = bridge(device.to_mlir()) [LINK:custom_api:122] [CALLER:custom_api|122]
```

~~~text
Python name "add_one"      → MLIR StringAttr → custom op symbol
Python selected device    → DeviceRef      → DeviceRefAttr → custom op device
Python out_types=[T]      → [T.to_mlir()]   → one native MO result type
Python parameters=None   → no extra kernel-parameter dictionary in add-one
~~~

The string names the kernel registered by the Mojo decorator; it does not embed the Python lambda or Mojo function body in the MO call. The device attribute is assigned after operation insertion and before kernel verification. Optional `parameters` are attached only when supplied; this example supplies none.
"""

GRAPH_CUSTOM_INSERT_EXPLANATION = """**SOURCE · Insert a new MO call operation into the active graph body, preserving the input value reference.** Python x remains a wrapper for the existing block argument.

```text
ops.custom → graph._add_op_get_op_with_results(mo.custom, result_types, values, symbol=...) [LINK:custom_api:109]
└─ Graph._add_op_get_op_with_results [LINK:ct_insert_result:1286]
   ├─ unwrap Python Value/Type/Attribute handles to MLIR binding handles [LINK:ct_insert_result:1297] [CALLER:ct_insert_result|1297]
   ├─ choose insertion point _ip or mlir.InsertionPoint(self._body) [LINK:ct_insert_result:1312] [CALLER:ct_insert_result|1312]
   ├─ enter insertion point + _location(); call mo.custom with converted arguments [LINK:ct_insert_result:1320] [CALLER:ct_insert_result|1320]
   ├─ find newly inserted operation and verify its IR structure [LINK:ct_insert_result:1328] [CALLER:ct_insert_result|1328]
   └─ return symbolic result wrappers and staged_op [LINK:ct_insert_result:1364] [CALLER:ct_insert_result|1364]
```

~~~text
Before: G1 body has %arg0 : float32[5,10] and chain seed CH0
             ↓ insert mo.custom using %arg0 as its operand
After:  G1 body retains %arg0 and CH0
        + custom call CUST1:
            symbol = "add_one"
            operand = %arg0
            result = %r : float32[5,10]
~~~

This is an operation-structure illustration, not an exact assembly dump. The input reference is connected to a new result-producing operation; the input data is not copied or computed. The generic builder verifies the inserted operation's IR structure before `ops.custom` finishes device/parameter attachment and explicitly verifies it against the loaded kernel library.
"""

GRAPH_CUSTOM_VERIFY_EXPLANATION = """**SOURCE · Check the symbolic MO call against the kernel definitions loaded from P1.** The existing Mojo package supplies the registered implementation contract.

```text
ops.custom → graph._kernel_library.verify_custom_op(custom_op) [LINK:custom_api:127]
└─ KernelLibrary.verify_custom_op [LINK:ct_graph_verify:307]
   ├─ library paths → frozenset cache key [LINK:ct_graph_verify:315] [CALLER:ct_graph_verify|315]
   ├─ cached Analysis exists? reuse it and verify this operation [LINK:ct_graph_verify:319] [CALLER:ct_graph_verify|319]
   └─ otherwise verify with current Analysis; cache it when paths exist [LINK:ct_graph_verify:322] [CALLER:ct_graph_verify|322]
```

~~~text
P1 (.mojoc): @extensibility.register("add_one") → AddOne.execute
                                           ↘ registered contract
MO call CUST1: "add_one", operand %x, result type T, device
                                           ↗ validation
valid symbolic call → Python result wrappers may be returned
invalid contract    → exception; caller does not receive a successful result
~~~

[Native Analysis::verifyCustomOp](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L69) bridges the operation to compiler analysis. P1 remains an implementation package; validation does not inline its Mojo body as MO arithmetic or run the kernel. Add-one supplies output types explicitly; its optional unimplemented shape function is not needed to infer these declared outputs.
"""

GRAPH_CUSTOM_RESULT_EXPLANATION = """**SOURCE · Return a Python list of wrappers for the custom operation's symbolic results.** Add-one has one declared result type, so the list contains one result.

```text
Graph._add_op_get_op_with_results [LINK:ct_insert_result:1356]
├─ native result Value → Value.from_mlir(_Value._from_cmlir(result)) [LINK:ct_insert_result:1358] [CALLER:ct_insert_result|1358]
└─ return results, staged_op → ops.custom returns results [LINK:custom_api:129] [CALLER:ct_insert_result|1364]
   └─ addition lambda selects results[0].tensor [LINK:entry:49] [CALLER:entry|38]
```

~~~text
MO result %r : float32[5,10]
    ↑ referenced by Python TensorValue wrapper
results = [wrapper(%r)]
    → results[0]          selects that wrapper
    → results[0].tensor   exposes the TensorValue interface
    → constructor receives symbolic result, not 50 computed numbers
~~~

The return is a list because `out_types` can declare several results; each result has its own native value and Python wrapper. `.tensor` does not execute or transfer data. `Graph.output(result)` records which symbolic result will be returned when the compiled model is eventually run.
"""

GRAPH_FORWARD_CALLBACK_EXPLANATION += custom_examples.FORWARD
GRAPH_CUSTOM_ATTRIBUTES_EXPLANATION += custom_examples.ATTRIBUTES
GRAPH_CUSTOM_INSERT_EXPLANATION += custom_examples.INSERT + custom_examples.ATTRIBUTES
GRAPH_CUSTOM_RESULT_EXPLANATION += custom_examples.RESULTS

GRAPH_FORWARD_OUTPUT_EXPLANATION = """**SOURCE · Finish the symbolic graph and restore the previous active graph.** [Constructor continuation](sources/max/python/max/graph/graph.py#L932).

```text
Graph.__init__ → Graph.output(result) [LINK:graph_init:932]
├─ flatten result structure and collect native output values/types [LINK:graph_output:1433] [CALLER:graph_output|1433]
├─ construct function type (body input types) → (output types) and generator signature [LINK:graph_output:1458] [CALLER:graph_output|1458]
├─ insert mo.OutputOp, record argument/result names and verify completed G1 [LINK:graph_output:1467] [CALLER:graph_output|1467]
└─ leave with self → __exit__ → CURRENT_GRAPH.reset(token) [LINK:ct_graph_enter:1149] [CALLER:ct_graph_context_manager|1138]
```

~~~text
Shell:     (%x : T) → ()
Forward:   %x → CUST1("add_one") → %r : T
Output:    signature becomes (T) → (T); mo.output returns %r
Exit:      previous Graph.current restored; populated G1 retained
Next:      session.compile(G1), then init, then execute with actual data
~~~

The output operation terminates the graph body and establishes its result contract. Verification checks the completed symbolic IR; no matrix arithmetic has happened during this recording sequence.
"""

GRAPH_OBJECTS_EXPLANATION = """**`builtin.ModuleOp` is the outer MLIR container; `mo.GraphOp(\"addition\")` is the named computation inside it.** Python’s `Graph(...)` constructor creates and retains both. See the [construction code](sources/max/python/max/graph/graph.py#L851).

**Source order inside this constructor:**

```text
@in_default_mlir_context
→ create empty builtin.ModuleOp + mo.GraphOp scaffold
→ import kernels/; prepare P1 (.mojoc)
→ forward(symbolic X); record mo.custom("add_one")
→ Graph.output(Y); return completed graph G1
```

The reader opens with the **package-preparation phase A.1**, then follows the completed **MO graph G1 in A.2/A.3**. The empty scaffold precedes package loading in the source; the custom operation is recorded after loading. See [kernel import](sources/max/python/max/graph/graph.py#L909) and [forward recording](sources/max/python/max/graph/graph.py#L918).

`Op` means **operation**. `builtin` is MLIR’s builtin dialect; `mo` is MAX’s graph dialect, defined in [MODialect.td](sources/GraphCompiler/include/GraphCompiler/MODialect/MODialect.td#L25). ([MLIR builtin dialect](https://mlir.llvm.org/docs/Dialects/Builtin/))

**1. `builtin.ModuleOp`: the container**

It represents an MLIR operation normally printed as `module { ... }`. Its body contains other operations—including your graph definition. MLIR organizes this nesting as **operation → region → block → operations**. ([MLIR operation structure](https://mlir.llvm.org/docs/LangRef/))

A **module is a compilation unit that groups definitions**. MAX’s Python `Module` wrapper owns the enclosing `builtin.module`; each `Graph` contributes a named computation inside it. The module’s body gives the compiler one place to find and process those definitions. See [Module’s documented role](sources/max/python/max/graph/graph.py#L562).

```text
Graph.__init__(..., module=maybe_module) [LINK:graph_init:851]
├─ caller supplied a Module wrapper
│  └─ reuse module.mlir_module → self._module [LINK:graph_init:855]
│     └─ Module exposes/stores the typed builtin.ModuleOp [LINK:ct_graph_module:637]
└─ no module supplied
   └─ builtin.ModuleOp(location=loc) [LINK:graph_init:856]
      └─ nanobind ModuleOp constructor [LINK:ct_builtin_module:62]
         └─ mlir::ModuleOp::create(loc, name=None) [LINK:ct_builtin_module:65]
            └─ builtin.module → body region → body block
               └─ initially empty; GraphOp will be inserted here
```

The native constructor and `.body` binding are in [builtin.cpp](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62).

When you supply `Graph(..., module=existing_module)`, MAX reuses that module. This allows several named graphs to share one module—for example, encoder and decoder graphs. See [the Python `Module` wrapper](sources/max/python/max/graph/graph.py#L562).

**For the addition example · immediately after GraphOp insertion**

At this instant, GraphOp has one typed input and no result types yet (<code>result_types=[]</code>). It has created the graph region and its entry block; <code>%arg0</code> is the symbolic input X, not a tensor buffer containing numbers. This is a temporary construction snapshot before the chain, custom operation, and <code>mo.output</code> are added. Location annotations are omitted from this schematic:

~~~mlir
module {
  mo.graph @addition(%arg0: !mo.tensor<[5, 10], f32>) -> () {
    // Entry block contains %arg0; no computation or mo.output yet.
  }
}
~~~

**Structure:** the outer module body contains the <code>mo.graph @addition</code> operation; that graph owns one region and one block, whose argument list contains <code>%arg0</code>. The selected-device mapping is attached separately immediately afterward by <code>Graph._populate_device_info_mapping()</code> ([graph.py](sources/max/python/max/graph/graph.py#L877)).

**1.1 Example · two graphs in one module**

**Illustrative Python using the repository’s APIs.** Both graphs take two float32 vectors of length four:

```python
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, ops

device = CPU()
t = TensorType(DType.float32, shape=[4],
               device=DeviceRef.from_device(device))
unit = Module()

g_add = Graph("add", module=unit, input_types=[t, t],
              forward=lambda a, b: ops.add(a, b))
g_mul = Graph("mul", module=unit, input_types=[t, t],
              forward=lambda a, b: ops.mul(a, b))
```

`unit` owns one IR module containing **two graph definitions**. `g_add` and `g_mul` retain their respective graph operations. Calls to `ops.add` and `ops.mul` first use **RMO builders** for relaxed shape handling; RMO later lowers to strict MO. See [add’s builder](sources/max/python/max/graph/ops/elementwise.py#L73), [mul’s builder](sources/max/python/max/graph/ops/elementwise.py#L353), and [the RMO dialect](sources/GraphCompiler/include/GraphCompiler/RMODialect/RMODialect.td#L27).

**Recorded module structure before compilation:**

```text
unit.mlir_module → builtin.module
├─ g_add → mo.graph @add → relaxed rmo.add computation
└─ g_mul → mo.graph @mul → relaxed rmo.mul computation
```

The Python calls build symbolic definitions. **C.1** follows their compilation through strict MO and MAP; **D.1** initializes the resulting named models.

**2. `mo.GraphOp(\"addition\")`: the computation definition**

Here, **`\"addition\"` names the graph**. The operation that performs your computation is separately referenced as **`\"add_one\"`** in [addition.py](sources/max/examples/custom_ops/addition.py#L33).

The actual constructor takes more than the name:

```python
op = _mo.GraphOp(
    builder,
    loc,
    name=\"addition\",
    input_types=[t.to_mlir() for t in input_types],
    result_types=[],
)
```

Its native construction expands as follows:

```text
Graph.__init__ [LINK:graph_init:851]
└─ builder = OpBuilder(self._module.body.end) [LINK:graph_init:858]
   ├─ evaluate self._module.body
   │  └─ ModuleOp.body binding returns the module body block [LINK:ct_builtin_module:68]
   ├─ evaluate .end
   │  └─ choose an insertion point after the block's current top-level operations
   ├─ construct the MLIR Python OpBuilder at that insertion point
   │  └─ underlying native mlir::OpBuilder is positioned in this module block
   └─ pass builder to _mo.GraphOp(builder, loc, ...) [LINK:graph_init:860]
      └─ C++ GraphOp::build(OpBuilder &, OperationState, ...) [LINK:ct_graph_op_build:820]
         ├─ sym_name = addition → identifies this graph in the module
         ├─ function type → input float32[5,10]; outputs initially empty
         ├─ KGEN signature + symbolic-dimension parameter attributes
         └─ create body region → entry block → typed symbolic input X
```

The builder positions the new graph at the end of the module body, so adding a graph to a shared module leaves earlier top-level graphs in place. The MLIR Python binding supplies the OpBuilder and block insertion-point API; the MAX call site is [graph.py:858](sources/max/python/max/graph/graph.py#L858), and the module-body binding is in [builtin.cpp:68–70](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L68).

These steps come directly from [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820). The operation’s definition requires a module parent, a single graph-body block, and a terminating `mo.output`; see [MOOps.td](sources/GraphCompiler/include/GraphCompiler/MODialect/MOOps.td#L211).

**3. How the initially empty graph becomes your addition graph**

The constructor calls `forward(...)` with symbolic inputs. Your `ops.custom(name=\"add_one\", ...)` records the computation and produces a symbolic output. Then `Graph.output(...)` establishes the output signature and inserts `mo.output`. See [forward construction](sources/max/python/max/graph/graph.py#L918) and [output finalization](sources/max/python/max/graph/graph.py#L1458).

The resulting structure is **schematic**, with internal chain and metadata operations omitted:

```text
Python Graph object
│
├─ _module → builtin.module
│  └─ body region / block
│     └─ mo.graph @addition
│        │
│        ├─ input signature:  float32[5,10]
│        ├─ output signature: float32[5,10]
│        │
│        └─ body region / block
│           ├─ X = symbolic input
│           ├─ custom computation \"add_one\"(X) → symbolic Y
│           └─ mo.output(Y)
│
└─ _mlir_op → another wrapper around that same mo.graph
```

The `_mlir_op` assignment wraps the existing operation through its C API pointer; it does not create another graph. See [graph.py](sources/max/python/max/graph/graph.py#L874).

**3.1 Your addition example · Python call → MO operation**

**Equivalent explicit construction** for `addition.py`: `Y_type` is its declared float32 `[5,10]` tensor type and `mojo_kernels` is its `kernels/` path. The original example uses a `forward` callback to record the same custom operation:

```python
with Graph("addition", input_types=[Y_type],
           custom_extensions=[mojo_kernels]) as graph:
    X = graph.inputs[0]
    y = ops.custom(name="add_one",
                   device=DeviceRef.from_device(device),
                   values=[X], out_types=[Y_type])[0].tensor
    graph.output(y)
```

**Schematic CPU MO view** of the recorded computation, with extra metadata omitted:

```mlir
module {
  mo.graph @addition(%x: !mo.tensor<[5, 10], f32>)
      -> !mo.tensor<[5, 10], f32> {
    %y = mo.custom [symbol = "add_one",
                   device = #M.device_ref<"cpu", 0>](%x)
        : (!mo.tensor<[5, 10], f32>)
          -> !mo.tensor<[5, 10], f32>
    mo.output %y : !mo.tensor<[5, 10], f32>
  }
}
```

| Python construction | Change in the IR |
|---|---|
| `Graph("addition", input_types=[...])` | Insert `mo.graph @addition`; create symbolic `%x`. |
| `ops.custom(name="add_one", ...)` | Insert `mo.custom`; attach kernel name and device; create symbolic `%y`. |
| `graph.output(y)` | Set the graph’s result signature and insert `mo.output %y`. |

The **Python builder runs to create IR nodes**; the kernel’s numerical work happens later. This does not translate the entire `addition.py` program into MLIR. See [custom-op construction](sources/max/python/max/graph/ops/custom.py#L114), [Graph.output](sources/max/python/max/graph/graph.py#L1458), and [the original example](sources/max/examples/custom_ops/addition.py#L33).

**Actual numbers arrive later**, through `model.execute(x)`. For example, an input element `2.5` produces `3.5` in this example. Compilation, initialization, buffer creation, and execution occur after graph construction in [addition.py](sources/max/examples/custom_ops/addition.py#L67).

The prepared **P1 `.mojoc` package** supplies kernel implementation bodies. **G1** contains the symbolic MO graph. In **C.1**, compilation rewrites a clone **C1** through MO → MAP; G1 remains with the caller. See [package loading](sources/max/python/max/graph/graph.py#L269).
"""


MODULE_COMPILATION_EXPLANATION = """**Shared-module example · compilation belongs to C.1**

After **B** has created `session` (S1), compile the `unit` built in A.2:

```python
compiled_unit = session.compile(unit)
```

This accepts a `Module` containing multiple graphs and returns one compiled artifact. See [compile’s Module input](sources/max/python/max/engine/api.py#L1062).

**Schematic MO view after RMO → MO.** Extra chains and device metadata are omitted. The graph bodies use the spelling in the repository’s [two-graph compiler test](sources/GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir#L14):

```mlir
module {
  mo.graph @add(%a: !mo.tensor<[4], f32>,
                %b: !mo.tensor<[4], f32>)
      -> !mo.tensor<[4], f32> {
    %r = mo.add(%a, %b) : !mo.tensor<[4], f32>
    mo.output %r : !mo.tensor<[4], f32>
  }
  mo.graph @mul(%a: !mo.tensor<[4], f32>,
                %b: !mo.tensor<[4], f32>)
      -> !mo.tensor<[4], f32> {
    %r = mo.mul(%a, %b) : !mo.tensor<[4], f32>
    mo.output %r : !mo.tensor<[4], f32>
  }
}
```

| Python / IR item | Meaning in this example |
|---|---|
| `unit` / outer `module { ... }` | Owns both definitions as one compilation unit. |
| `g_add` / `mo.graph @add` | Describes elementwise `A + B`. |
| `g_mul` / `mo.graph @mul` | Describes elementwise `A * B`. |
| `%a`, `%b`, `%r` | Symbolic values connected by data dependencies. |
| `!mo.tensor<[4], f32>` | Tensor type: four float32 elements; the numbers arrive at execution. |
"""


GRAPH_LOWERING_EXPLANATION = """**G1 → compiler clone C1 → MO → MAP**

`G1` is the caller-owned Python `Graph`; its `builtin.module` contains the `mo.graph` operation. `session.compile(G1)` gives the compiler a working clone, **C1**. MAX runs graph passes on C1, leaving G1 available to the caller.

```text
G1 / C1 at the MO boundary
module {
  mo.graph @addition(%x: !mo.tensor<[5, 10], f32>) -> !mo.tensor<[5, 10], f32> {
    %y = mo.custom [symbol = "add_one"](%x)
         : (!mo.tensor<[5, 10], f32>) -> !mo.tensor<[5, 10], f32>
    mo.output %y : !mo.tensor<[5, 10], f32>
  }
}

MOToMAP · registered custom-op pattern · schematic for add_one
map.iter.opaque
├─ inputs (X); outputs (Y)
└─ map.call.opaque @...AddOne
   └─ kernelName = "add_one"
```

**Custom-op evidence:** the [MOToMAP custom-kernel test](sources/GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels.mlir#L30) shows the same registered opaque-op form: `mo.custom` becomes `map.iter.opaque`, which contains `map.call.opaque`. Its test-only symbol is `map_lower_one_fused_output`; the drawing applies that lowering pattern to `add_one` and is schematic, not a captured MAX dump. The package P1 supplies the registered `AddOne` implementation; its source uses `foreach` to define the elementwise work ([AddOne](sources/max/examples/custom_ops/kernels/add_one.mojo#L32)).

**Built-in-op comparison:** Python `ops.add(A, B)` initially records relaxed RMO. After RMO → MO, the [elementwise-add test](sources/GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-add-static.mlir#L6) shows `mo.add` becoming `map.iter.elementwise` with indexed loads, `map.call.elementwise @...Add`, and a store. That is a distinct MAP lowering shape from the `add_one` opaque custom call.

The enclosing module groups definitions while compiler passes rewrite C1's contents. The [graph-stage test](sources/GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir#L28) later expects `mo.graph @add` and `mo.graph @mul` to become `mgp.model @add` and `mgp.model @mul` in the runtime-plan representation.
"""


MODULE_INITIALIZATION_EXPLANATION = """**Shared-module example · runtime initialization belongs to D.1**

Use the compiled artifact from C.1:

```python
models = session.init_all(compiled_unit)
add_model = models["add"]
mul_model = models["mul"]
# Supply input buffers to add_model.execute(A, B) or mul_model.execute(A, B) later.
```

`init_all` returns a model dictionary keyed by each graph’s `sym_name`; the compiled artifact is ready for execution after successful runtime initialization. See [init_all’s named results](sources/max/python/max/engine/api.py#L1160). The single-graph addition example uses `session.init(CM1)` to obtain one M1.
"""


STAGES = [
    dict(
        id="graph", letter="A", title="kernels/ → package P1 → recorded graph G1 (MO)", short="kernels/ → P1 → G1 (MO)",
        call='graph = Graph("addition", forward=..., input_types=..., custom_extensions=[mojo_kernels])',
        source="entry", line=33,
        contracts=[
            ("Input", 'kernels/__init__.mojo + referenced kernels/*.mojo; name="addition"; symbolic X: float32[5,10]'),
            ("Output", 'P1: one precompiled .mojoc package; G1: verified caller-owned MO graph using P1'),
            ("Example", 'parse/check package → P1; forward records custom("add_one", [X]) → symbolic Y; MO → MAP is in C'),
        ],
        note="Package preparation is shown first. Graph.__init__ creates its empty scaffold earlier and records the custom operation after P1 is ready; A.2 shows that source order. C performs MO → MAP on clone C1.",
        tree="""[SOURCE:ct_graph_import|_import_kernels(...): prepare package P1] [CALLER:graph_init|910]
├─ [SOURCE:package_load|KernelLibrary.load_paths([kernels/])]
│  └─ [SOURCE:package_builder|_build_mojo_source_package(kernels/)]
│     ├─ [SOURCE:kernel_package_root|kernels/__init__.mojo] + referenced child modules, including add_one.mojo
│     ├─ separate compiler process → **Mojo executable** + **mojo precompile kernels/ -o staging.mojoc**
│     │  └─ parse → resolve → semantic/lifetime checks → MLIR bytecode
│     └─ atomic publication → **P1: one .mojoc package**
├─ existing .mojoc path → load that package directly
└─ package phase complete → [CARD:toolchain-call-graph-package|A.1 · parser, checks, file format and subprocess]

[SOURCE:ct_graph_callsite|2 · Record and finish G1 with P1 available]
├─ [SOURCE:ct_graph_context|@in_default_mlir_context] → activate an MLIR context on this thread → [EXPLANATION:context]
├─ reuse the constructor's empty [EXPLANATION:graph_objects] scaffold
│  └─ input **X: float32[5,10]** + device metadata + ordering chains
├─ forward(symbolic X)
│  └─ [SOURCE:custom_api|ops.custom(name="add_one", values=[X], out_types=[Y_type], device=DeviceRef.from_device(device))]
│     ├─ insert **mo.custom** → symbolic Y
│     └─ [SOURCE:ct_graph_verify|verify_custom_op] → verify against **P1**
├─ [SOURCE:graph_output|Graph.output(Y)] → mo.output + output signature + graph verification
└─ return **G1: caller-owned MO graph** → [CARD:toolchain-call-graph-validation|A.3 · registration and validation]
   Next: create **S1** (B); compile a clone **C1: MO → MAP → MOGG → MGP** (C).
""",
        details=[
            dict(id="context", title="A.0 · What is the default MLIR context? Is it a global variable?",
                 explanation=DEFAULT_CONTEXT_EXPLANATION,
                 tree_callers=["ct_graph_context"]),
            dict(id="package", title="A.1 · Parse kernels/*.mojo → package P1 (.mojoc)", tree="""[SOURCE:package_builder|_build_mojo_source_package(kernels_dir)]  [parent Python process]
├─ input: absolute directory with [SOURCE:kernel_package_root|kernels/__init__.mojo]
├─ hash = MD5(absolute path string) → a stable filename, not a source-content hash
├─ final: <tempdir>/.modular_<uid>/mojo_pkg/mojo_pkg_<hash>.mojoc
├─ staging: .staging.<pid>/mojo_pkg_<hash>.mojoc
├─ [SOURCE:ct_mojo_run|subprocess_run_mojo(["precompile", kernels_dir, "-o", staging], check=True)]
│  ├─ [SOURCE:ct_mojo_env|SDK default: <MAX package root>/bin/mojo; ambient path override]
│  ├─ MODULAR_MOJO_MAX_DRIVER_PATH → otherwise shutil.which("mojo") on PATH
│  └─ subprocess.run(...) → Python waits for the **separate compiler process**
│     └─ [SOURCE:ct_precompile|mojo precompile: precompile(State)]
│        ├─ validate source package / names; create compiler context; open output
│        ├─ invokeMojoParser(...)
│        │  └─ [SOURCE:ct_parser|LIT::importMojoPackage(kernels_dir, "kernels", ...)]
│        │     └─ [SOURCE:ct_resolver|DeclResolver::resolveAllReferencedFrom(package)]
│        │        ├─ parse the package + child modules, including [SOURCE:reference_add_one|add_one.mojo]
│        │        └─ resolve referenced dependency declarations and bodies
│        ├─ [SOURCE:ct_package_build|buildPrecompiledPackage(...)] → serializable post-parse IR clone
│        ├─ [SOURCE:ct_check_lit|KGENCompiler::runCheckLITPipeline(parsed_module)] → semantic/lifetime checks
│        └─ [SOURCE:ct_package_write|writePrecompiledFile(built_package, stream)]
│           └─ LIT lit.package MLIR bytecode → zstd payload + **MPKG V2** header → **staging .mojoc**
│
├─ compiler succeeds → os.replace(staging, final) → atomic publication
├─ finally → remove staging directory
└─ return final Path → **P1**; **addition.py stays a separate Python program**
   The source-directory branch invokes precompile again; it has no exists() skip.
""", explanation=MOJO_PRECOMPILE_EXPLANATION),
            dict(id="location", title='_location() · What is loc?',
                 explanation=GRAPH_LOCATION_EXPLANATION,
                 tree_callers=["ct_graph_location"]),
            dict(id="module-setup", title='MLIR Module · Reuse or create the IR container',
                 explanation=GRAPH_MODULE_SETUP_EXPLANATION,
                 tree_callers=["graph_init"]),
            dict(id="op-builder", title='OpBuilder · Insert at module.body.end',
                 explanation=GRAPH_OP_BUILDER_EXPLANATION,
                 tree_callers=["graph_init"]),
            dict(id="type-conversion", title='to_mlir() · Python input type → MLIR tensor type',
                 explanation=GRAPH_TYPE_CONVERSION_EXPLANATION,
                 tree_callers=["ct_tensor_to_mlir"]),
            dict(id="native-builder", title="GraphOp · Region, block arguments and signature",
                 explanation=GRAPH_NATIVE_BUILDER_EXPLANATION,
                 tree_callers=["ct_graph_op_build"]),
            dict(id="symbolic-params", title="Shape parameters · Declare symbolic dimensions",
                 explanation=GRAPH_SYMBOLIC_PARAMS_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="retain-body", title="Graph handles · Retain the operation and body block",
                 explanation=GRAPH_RETAIN_BODY_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="device-info", title="Device metadata · Attach the module mapping",
                 explanation=GRAPH_DEVICE_INFO_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="tracking", title="Weights and chains · Initialize graph bookkeeping",
                 explanation=GRAPH_TRACKING_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="ready-chain", title="Ordering chain · Create the always-ready seed",
                 explanation=GRAPH_READY_CHAIN_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="trailing-chain", title="Chain input · Recognize and adopt an incoming dependency",
                 explanation=GRAPH_TRAILING_CHAIN_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="forward-context", title="with self · Activate graph construction", explanation=GRAPH_FORWARD_CONTEXT_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="forward-inputs", title="self.inputs · Block argument → symbolic TensorValue", explanation=GRAPH_FORWARD_INPUTS_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="forward-callback", title="forward(x) · Execute Python to record IR", explanation=GRAPH_FORWARD_CALLBACK_EXPLANATION, tree_callers=['graph_init', 'graph_init']),
            dict(id="custom-attributes", title="ops.custom · Name, device and output attributes", explanation=GRAPH_CUSTOM_ATTRIBUTES_EXPLANATION, tree_callers=['graph_init', 'graph_init']),
            dict(id="custom-insert", title="mo.custom · Insert the call into G1", explanation=GRAPH_CUSTOM_INSERT_EXPLANATION, tree_callers=['graph_init', 'graph_init', 'graph_init']),
            dict(id="custom-verify", title="Kernel validation · MO call ↔ Mojo package P1", explanation=GRAPH_CUSTOM_VERIFY_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="custom-result", title="Result wrappers · %r → [TensorValue] → lambda result", explanation=GRAPH_CUSTOM_RESULT_EXPLANATION, tree_callers=['graph_init', 'graph_init']),
            dict(id="forward-output", title="Finish forward · Output contract and context exit", explanation=GRAPH_FORWARD_OUTPUT_EXPLANATION, tree_callers=["graph_init"]),
            dict(id="objects", title='A.2 · ModuleOp + GraphOp → record graph G1',
                 explanation=GRAPH_OBJECTS_EXPLANATION,
                 tree_callers=["graph_init", "ct_graph_module", "ct_graph_op_build", "graph_init"]),
            dict(id="validation", title="A.3 · Custom-op registration → verified graph G1", tree="""[SOURCE:custom_api|ops.custom("add_one", values=[X], out_types=[Y_type], device)]
├─ explicit out_types → output dtype/shape/device is already supplied
│  └─ this path bypasses AddOne's optional unimplemented shape function
├─ insert mo.custom with name="add_one" → link to [SOURCE:registration|@extensibility.register("add_one") / AddOne]
├─ [SOURCE:ct_graph_verify|KernelLibrary.verify_custom_op(operation)]
│  ├─ lookup reusable native Analysis by frozenset(package_paths)
│  └─ [SOURCE:ct_analysis_verify|Analysis::verifyCustomOp(operation)]
│     └─ MojoRegistry → check registered kernel, device, arguments and results
├─ return symbolic custom result → callback returns its .tensor
└─ [SOURCE:graph_output|Graph.output(Y)]
   ├─ flatten result tree; infer output types/names; write graph signature
   ├─ record mo.output(Y, required chains)
   └─ verify G1 → **Y describes a future output; it contains no computed numbers**
"""),
        ],
    ),
    dict(
        id="session", letter="B", title="InferenceSession(devices=[device]) → S1", short="InferenceSession(...) → S1",
        call="session = InferenceSession(devices=[device])", source="entry", line=62,
        contracts=[
            ("Input", 'existing Device handle(s); num_threads=None; session custom_extensions=None'),
            ("Output", 'S1: Python session → native session → Engine::Context + host CPU runtime'),
            ("Example", 'CPU → [CPU]; GPU → [Accelerator, CPU]. G1 keeps its own package P1.'),
        ],
        tree="""[SOURCE:ct_session|InferenceSession.__init__(devices=[device])] → **S1**
├─ normalize devices → deduplicate while preserving first occurrence
├─ append CPU() if absent → host runtime is present for **both target routes**
├─ optional import/export MEF policy → neither selected in addition.py
├─ session extensions = []; num_threads or 0 → automatic runtime choice
└─ [SOURCE:ct_session_binding|_InferenceSession(final_devices, [], 0)]  [nanobind → C++]
   └─ [SOURCE:ct_session_native|Core::InferenceSession::InferenceSession(...)]
      ├─ [SOURCE:ct_compile_config|populateCompileConfig([])] → session compilation settings
      ├─ retain Device handles; runtimeConfig.addDevice(DeviceSpec, driver context)
      ├─ runtimeConfig → numThreads=0; API language="python"
      └─ [SOURCE:ct_context_create|Engine::Context::create(runtimeConfig)]
         ├─ [SOURCE:ct_cpu_options|getOrCreateCPUDeviceOptions(config)] → host scheduling options
         ├─ [SOURCE:ct_global_context|Init::getOrCreateContext("max-engine", options)]
         │  ├─ matching global context exists → retain it
         │  └─ otherwise → [SOURCE:ct_global_create|createContextImpl(...)] → MAX context + CPUDevice
         ├─ [SOURCE:ct_context_ctor|Engine::Context(config, maxContext)] → retain runtime references
         └─ native session owns this Engine::Context → **S1 ready for compile/init** → [CARD:toolchain-call-session-ownership|B.1]
""",
        details=[dict(id="ownership", title="B.1 · Devices → native context and host runtime → S1", tree="""[SOURCE:ct_session|Constructor input → normalized output]
├─ [CPU()]                  → [CPU()]
├─ [Accelerator()]          → [Accelerator(), CPU()]
└─ [device, device, CPU()]  → [device, CPU()]  (if device is an accelerator)

**S1** [Python InferenceSession]
└─ _impl → [SOURCE:ct_session_native|Core::InferenceSession] [C++ object in the same host process]
   ├─ devices → retained existing driver handles / contexts
   ├─ sessionCompileConfig → compiler settings / optional extension paths
   └─ runtimeContext → [SOURCE:ct_context_ctor|Engine::Context] [owned native object]
      ├─ config → runtime settings / device descriptions
      ├─ maxContext → [SOURCE:ct_global_context|shared MAX ContextRef]
      │  └─ creation/reuse protected by **std::mutex**
      └─ cpuDevice → AsyncRT host runtime / worker scheduling

Thread policy → [SOURCE:ct_cpu_options|num_threads=None becomes 0: automatic selection]
└─ context/runtime can be reused → do not infer a new private pool per S1
Constructor result → **S1** configuration + ownership; G1 and its prepared P1 are supplied later to compile.
In this example, session custom_extensions=None → package preparation remains in A.
""", explanation=SESSION_VALUES_EXPLANATION )],
    ),
    dict(
        id="compile", letter="C", title="session.compile(G1) → compiled artifact CM1", short="compile(G1) → CM1",
        call="compiled = session.compile(graph)", source="entry", line=67,
        contracts=[
            ("Input", 'S1 + G1 module + package P1 + graph target metadata; expected_weights={}'),
            ("Output", 'CM1: CompiledModel with resolved native artifact, MEF bytes and uninitialized model descriptor(s)'),
            ("Example", 'name="addition"; one X/Y float32[5,10]; AddOne specializes for CPU or GPU'),
        ],
        tree="""[SOURCE:ct_compile|InferenceSession.compile(G1)]  [Python caller]
├─ [SOURCE:compile_async|compile_async(G1)]
│  ├─ module = G1.module; extensions += G1.kernel_libraries_paths → **P1**
│  ├─ expected_weights = {} → add-one has no external weight bindings
│  ├─ [SOURCE:ct_analysis_seed|Analysis::seedKernelDecls(module)] → add reusable declarations to caller module
│  ├─ [SOURCE:ct_compile_module|_compile_module(module, extensions, tile_based_fusion=False)]
│  │  ├─ acquire **process-global compilation lock**
│  │  ├─ [SOURCE:ct_compile_binding|_impl.compile(module._CAPIPtr, extensions, pipeline_name, False)]
│  │  │  └─ [SOURCE:ct_compile_native|Core::InferenceSession::compile(capsule, ...)]
│  │  │     ├─ CompilationRequest → source module pointer + graph format + options
│  │  │     └─ [SOURCE:ct_context_compile|Engine::Context::compileModel(config)]
│  │  │        ├─ allocate pending AsyncValue<CompiledModels>
│  │  │        └─ addTask(cpuDevice, ...) → **compilation on a host runtime worker**
│  │  │           └─ [SOURCE:ct_context_compile_sync|compileModelSync(config)]
│  │  │              └─ [SOURCE:ct_compiled_create|CompiledModels::create(context, compileCfg, runtimeCfg)]
│  │  │                 └─ [SOURCE:frontend|MOFContext::compileToBinary(source, format)]
│  │  │                    ├─ clone module → compiler working graph **C1**
│  │  │                    ├─ graph-artifact cache hit → reuse prepared MEF bytes
│  │  │                    └─ miss/disabled → **MO → MAP → MOGG → MGP** → [CARD:toolchain-call-compile-passes|C.1]
│  │  │                       → kernel JIT / LLVM / targets [CARD:toolchain-call-compile-kernels|C.2] → MEF [CARD:toolchain-call-compile-mef|C.3]
│  │  └─ done callback → release compilation lock when the task resolves
│  └─ wrap pending handle → [SOURCE:artifact|CompiledModel(handle, expected_weights={})] → **CM1**
│
├─ handle.wait() → caller waits for completion; exception() → raise on failure
└─ return **resolved CM1** → graph names=("addition",); executable state comes from init
   **G1 remains with the caller**; native lowering transforms the clone **C1**.
""",
        details=[
            dict(id="passes", title="C.1 · Clone G1 → MO → MAP → MOGG → MGP", tree="""[SOURCE:frontend|MOFContext::compileToBinary] → owns the compilation route for **C1**
└─ cache miss → [SOURCE:ct_convert_mef|convertMOToMEF(C1, options)]
   └─ [SOURCE:ct_run_gc|MO::runGCPipeline(C1, options)]
      ├─ create mlir::PassManager; gcPipelineImpl builds passes; pm.run(C1) executes them
      ├─ [SOURCE:mo_pipeline|buildMoPipeline] → **MO = Modular Operations**
      │  ├─ graph/tensors/shapes/device placement/custom op contracts
      │  └─ seeded kernel declarations → skip redundant package import
      ├─ [SOURCE:mo_to_map|MOToMAPPass] → **MAP = Modular Access Primitives**
      │  └─ expose accesses and computations for fusion / implementation decisions
      ├─ [SOURCE:map_pipeline|buildMapPipeline] → optimize access patterns / eligible fusion
      ├─ [SOURCE:map_to_mogg|MAPToMOGGPass] → **MOGG = Modular Operation Generator Graph**
      │  └─ MojoRegistry + MAPToMOGGState → select implementation / glue calls
      ├─ [SOURCE:mogg_pipeline|buildMoggPipeline] → refine Mojo-oriented functions and calls
      ├─ [SOURCE:mogg_to_mgp|MOGGToMGPPass] → **MGP = Modular Glue Primitives**
      │  └─ MOGraphLowering → init/execute regions, buffers, state slots and chains
      ├─ [SOURCE:mgp_pipeline|buildMgpPipeline]
      │  ├─ [SOURCE:device_finalization|FinalizeDeviceGraphsPass] → finalize marked device graphs when present
      │  └─ [SOURCE:emit_mojo|EmitMojo] → emitted source for native routines / package imports
      ├─ [SOURCE:jit_entry|JitCompileKernelsPass] → compile/reuse native routines (C.2)
      └─ [SOURCE:jit_pipeline|buildJitPipeline]
         ├─ [SOURCE:runtime_calls|LowerCompiledRoutinesPass] → ModuleLoad / ExecuteGeneric
         └─ inline/clean chains → OutlineModelsToCallables → **initialization + execution plan**
   Dialect names and roles → [SOURCE:graph_compiler_design|GraphCompiler.md]

[SOURCE:map_to_mogg|Add-one ancestry] [schematic representation; not a captured compiler dump]
└─ mo.custom("add_one", X) → implementation-oriented accesses/call
   → selected AddOne routine → buffer/state execution plan → compiled callable
""", explanation="[Open the detailed C.1 compiler route · examples and why for every stage](max_graph_compiler_pipeline_examples.html)\n\n" + MODULE_COMPILATION_EXPLANATION + "\n\n" + GRAPH_LOWERING_EXPLANATION),
            dict(id="kernels", title="C.2 · Kernel specialization → LLVM → target code", tree="""[SOURCE:jit_entry|compileMojoToBinary(C1, emitted Mojo, ...)]
├─ emitted routine + **P1** → parser/checks/LIT lowering → separate kernel IR **KIR1**
├─ compiled-kernel lookup → [SOURCE:kernel_cache|checkCache(state, ...)]
│  ├─ hit → [SOURCE:kernel_hits|handleCacheHits] → reuse compatible native archives
│  └─ miss → [SOURCE:kernel_misses|handleCacheMisses]
│     ├─ [SOURCE:kgen_elaboration|KGENCompiler::runElaborationPipeline]
│     │  ├─ [SOURCE:kgen_elaborate_build|populateElaborateModulePasses] → concrete parameter specialization
│     │  ├─ [SOURCE:kgen_post_elaboration|buildPostElaborationPipeline] → concrete IR optimization
│     │  └─ offloads encountered → [SOURCE:offload_targets|compileOffloads] → group/slice IR by TargetInfo
│     └─ [SOURCE:ct_emit_archive|ObjectCompiler::emitArchive] → native archive with callable symbols
│        └─ [SOURCE:object_lower_to_llvm|ObjectCompiler::lowerAllFuncsToLLVM]
│           ├─ [SOURCE:kgen_llvm_passes|buildLowerToLLVMPipeline] → KGEN/POP/control flow → **LLVM dialect (MLIR)**
│           └─ [SOURCE:llvm_translation|MLIR-to-LLVM translation] → llvm::Module → **LLVM1 (LLVM IR)**
│
├─ target options guide compilation → [SOURCE:backend_registry|TargetBackendRegistry::getBackend] uses the triple
│  ├─ host target → [SOURCE:host_object|HostBackend] → CPU object / host routines
│  ├─ nvptx64-nvidia-cuda, e.g. sm_90a → [SOURCE:nvptx_backend|NVPTXBackend] → PTX → cubin
│  └─ amdgcn-amd-amdhsa, e.g. gfx942 → [SOURCE:amdgpu_backend|AMDGPUBackend] → GPU object → ELF image
│     Target examples are source examples, not detected hardware on this host.
├─ [SOURCE:llvm_target_machine|LLVM TargetRegistry + TargetMachine] → [SOURCE:llvm_code_gen|in-process LLVM code generation]
└─ native compiled resources + **C1 runtime plan** → continue MEF preparation (C.3)

**JIT = just-in-time compilation** → required specialized code is prepared as the program runs.
└─ here: addition.py → session.compile(G1) invokes compiler APIs in the host process
   → AddOne's dtype/layout/target specialize; compiled code can be reused for later inputs.
   **P1 precompilation is the earlier subprocess**; kernel JIT is a later compilation phase.
"""),
            dict(id="mef", title="C.3 · MEF preparation → compiled artifact CM1", tree="""[SOURCE:ct_convert_mef|convertMOToMEF(C1)]  [after graph passes + native routine JIT]
└─ [SOURCE:serialize|MO::convertMGPToMEF(lowered_module)]
   ├─ clone for serialization → prepare device-specific runtime primitive names
   ├─ translateMLIRToMEF → serialized graph callables / attrs / native resources
   └─ WriteableBuffer → **prepared MEF bytes in memory**

[SOURCE:ct_compiled_create|CompiledModels::create(...)] receives those bytes
└─ [SOURCE:ct_compiled_read|CompiledModels::read(buffer, frameworkCtx, sourceModel)]
   └─ [SOURCE:ct_mef_open|loadModelsFromMEF(buffer)]
      ├─ MEFFile::open → validate/open the prepared container
      ├─ [SOURCE:ct_mef_models|loadMGPMEFFile(mef)] → register runtime primitives; read Model descriptors
      └─ retain buffer + MEF handle + model descriptors → native **CompiledModels**
         → resolve async handle → Python **CM1: CompiledModel**

Cache routes → [SOURCE:artifact_cache|MOFContext graph-artifact cache]
├─ artifact hit → skip graph lowering / kernel JIT; read retained MEF bytes/descriptors
├─ kernel hit only → reuse routine code; still lower/prepare the graph's runtime plan
└─ cold path → both compilation phases run

On disk → [SOURCE:artifact|CM1.export_mef(path)] explicitly exports the compiled bytes
└─ session.compile(G1) itself returns CM1; it does not require a user-named .mef file.
"""),
        ],
    ),
    dict(
        id="init", letter="D", title="session.init(CM1) → executable model M1", short="init(CM1) → M1",
        call="model = session.init(compiled)", source="entry", line=68,
        contracts=[
            ("Input", 'S1 + resolved CM1 + real device contexts; weights_registry omitted → {}'),
            ("Output", 'M1: initialized ModelHandle retaining executable state, CM1 and native S1'),
            ("Example", 'init_all → {"addition": M1}; init → M1. The input Buffer is supplied later.'),
        ],
        tree="""[SOURCE:ct_init|InferenceSession.init(CM1)]
├─ [SOURCE:init_all|init_all(CM1, weights_registry=None)]
│  ├─ registry = {}; expected_weights = {} → no weight-binding validation failures → [CARD:toolchain-call-init-gates|D.2]
│  ├─ [SOURCE:ct_session_binding|_impl._load_all(compiled_handle, registry)] → share artifact handle
│  │  └─ [SOURCE:ct_load_all|Core::InferenceSession::loadAll(handle, weightsDict)]
│  │     ├─ [SOURCE:ct_weight_dict|buildWeightsFromDict({})] → empty native registry + managed-tensor list
│  │     ├─ await compiled handle (release GIL while waiting) → descriptor "addition"
│  │     └─ for each model descriptor → [SOURCE:ct_load|Core::InferenceSession::load(model, registry)]
│  │        └─ [SOURCE:ct_context_init|Engine::Context::initModel(model, registry)]
│  │           └─ addTask(cpuDevice, ...) → initializer runs on a host runtime worker
│  │              └─ [SOURCE:ct_executable_create|createExecutableModel(model, cpuDevice, config, registry)]
│  │                 ├─ DeviceMap from session DeviceSpecs + retained driver contexts
│  │                 ├─ [SOURCE:ct_model_setup|Model::setup(cpuDevice, registry, deviceMap)] → [CARD:toolchain-call-init-resources|D.1]
│  │                 │  ├─ [SOURCE:ct_device_map|mapDeviceRefsToDevices] → synchronize/resolve contexts
│  │                 │  ├─ initArgs = registry pointer + model device contexts
│  │                 │  ├─ initFn.invoke(initArgs) → run **compiled initialization callable**
│  │                 │  └─ construct ExecutableModel from chain/context/registry/extra args
│  │                 └─ [SOURCE:ct_wait_setup|waitForModelSetup()] → initialization results must be ready
│  │        → [SOURCE:ct_model_handle|ModelHandle(..., native_session_ref, executable, registry, index)]
│  │          → cache input/output devices → **M1**
│  └─ name-index results → **{"addition": M1}**
├─ exactly one model? yes → return sole model **M1**
└─ next source step → numeric input allocation, **model.execute(x)** (outside init)
""",
        details=[
            dict(id="resources", title="D.1 · Load runtime resources → ready executable M1", tree="""[SOURCE:ct_model_setup|Model::setup] → compiled initFn.invoke(registry + devices)
├─ [SOURCE:ct_device_map|mapDeviceRefsToDevices]
│  ├─ synchronize supplied driver contexts before first resource use
│  ├─ CPU DeviceRef → create legacy runtime CPU context
│  └─ GPU DeviceRef → reuse supplied context; fallback create by device id if absent
├─ CUDA/HIP contexts → ensure configured stream count before capture (default: 2)
├─ invoke the compiled init region → runtime/state initialization and native routine loading
│  └─ [SOURCE:runtime_calls|ModuleLoad planned by LowerCompiledRoutinesPass]
│     └─ [SOURCE:ct_module_load|mgp_module_load_cpu] → CompiledKernelsContext in a runtime state slot
│        └─ [SOURCE:ct_kernel_load|CompiledKernelsContext::load(archive_bytes, symbols)] [DETAIL:max_compiled_kernels_context_load_and_driver.html|Loader deep dive · all steps + GPU driver]
│           ├─ ordinary route: archive hash → execution engine addIfAbsent<StaticArchiveLayer>
│           └─ lookup named symbols → **resident native callable pointers**
│              (debug/profiler route can link/load a shared library)
├─ initializer results → [chain, MGP context, weight-registry reference, extra state/device args]
├─ [SOURCE:ct_wait_setup|waitForModelSetup] → await every required value; propagate errors
└─ executable **M1** → reuse retained state and routines for fresh execution inputs

Boundaries
├─ **.mojoc P1**: portable post-parse package for later specialization
├─ **CM1**: compiled bytes + descriptors; still needs runtime setup
└─ **M1**: ready executable handle; numeric X arrives at execute, not at constructor/init
""", explanation=MODULE_INITIALIZATION_EXPLANATION),
            dict(id="gates", title="D.2 · Initialization gates, weights and retained ownership", tree="""[SOURCE:init_all|init_all initialization gates]
├─ add-one: expected weights {} + supplied registry {} → pass
├─ weighted graph: missing declared name → ValueError before native initialization
├─ supplied weight: device type must match; array must be contiguous
├─ actual device/context/setup failure → native initialization raises
├─ virtual-device mode → CompileOnlyModel stand-in; execute refuses hardware work
└─ [SOURCE:ct_init|init] requires exactly one result; multi-model artifacts use init_all

**M1** [native ModelHandle exposed as the Python Model]
├─ loadedModel → initialized ExecutableModel and persistent execution state
├─ compiledModels → shared **CM1** native artifact / model descriptors / bytes
├─ inferenceSession → retained [SOURCE:ct_model_handle|native **S1** reference]
│  └─ Engine::Context → MAX context / CPU runtime / session device settings
└─ weightsRegistry + managedTensors → retained bindings (empty in add-one)

Repeated initialization → [SOURCE:ct_session_binding|_load_all uses compiled.copy()]
└─ same CM1 can initialize multiple handles; each handle retains what it needs.
"""),
        ],
    ),
]

EXAMPLE = """[SOURCE:entry|addition.py: execution happens after all four preparation calls]
├─ x_values = np.random.uniform(size=(5,10)).astype(np.float32) → host array
├─ x = Buffer.from_numpy(x_values).to(device) → **B1: input Buffer**
├─ result = model.execute(x)[0] → **O1: output Buffer**
└─ result.to(CPU()).to_numpy() → CPU-readable numbers; expected = x_values + 1

**Illustrative values in the same [5,10] shape** [not an executed MAX result]
├─ input row 0:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
├─ output row 0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
└─ **G1 / S1 / CM1 / M1** were prepared without these numbers; B1/O1 hold per-call data.
"""


MARKER = re.compile(r"\[SOURCE:([a-z_]+)\|((?:[^\[\]]|\[[^\[\]]*\])+)\]")
CARD_MARKER = re.compile(r"\[CARD:([a-z0-9_-]+)\|([^\]]+)\]")
CALLER_MARKER = re.compile(r"\[CALLER:([a-z_]+)\|(\d+)\]")
TREE_BRANCH = re.compile(r"^(?P<prefix>(?:│  |   )*)(?:├─|└─)\s*(?P<body>.*)$")
LINE_SOURCE_MARKER = re.compile(r"\[LINK:([a-z_]+)(?::(\d+))?(?:\|([^\]]+))?\]")
TREE_MARKER = re.compile(
    MARKER.pattern + "|" + CALLER_MARKER.pattern + "|" + LINE_SOURCE_MARKER.pattern
)


GRAPH_CONSTRUCTOR_CALL_STACK_TREE = """caller: graph = Graph(...) [LINK:entry:33]
│
├─ Evaluate the arguments [LINK:entry:33]
│  ├─ Create input TensorType [LINK:entry:51]
│  │    IN:  dtype=float32, shape=[5, 10], device=DeviceRef(device) [LINK:entry:52]
│  │    OUT: a type description; it contains no tensor data [LINK:entry:51]
│  ├─ Create the forward lambda [LINK:entry:37]
│  │    The lambda is captured here; its body runs later in __init__. [LINK:entry:37]
│  └─ Set custom_extensions=[mojo_kernels] [LINK:entry:58]
│
└─ Graph.__init__("addition", ...) [LINK:graph_init:819]
   ├─ Enter the default MLIR context [CARD:context] [LINK:graph_init:819]
   │    Makes the MLIR context available while constructing graph IR. [LINK:ct_graph_context_state:55]
   │
   ├─ Create the graph shell [LINK:graph_init:840]
   │  ├─ Collect named symbolic dimensions from TensorType/BufferType inputs [LINK:graph_init:840]
   │  │    Scan each input shape and keep only SymbolicDim names [LINK:graph_init:842]
   │  │    IN:  float32 input shape [5, 10] [LINK:entry:54]
   │  │    OUT: _params = {}; both dimensions are static, so neither becomes a runtime parameter [LINK:graph_init:846]
   │  ├─ Initialize graph-construction bookkeeping [LINK:graph_init:847]
   │  │    _context_state = []; _data_dependent_dims = 0; op verification enabled [LINK:graph_init:849]
   │  ├─ Enter _location() and choose the top-level module [CARD:toolchain-call-graph-location|_location() · What is loc?] [LINK:graph_init:851]
   │  │    => it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module. [LINK:ct_graph_location:457]
   │  │  ├─ Reuse module.mlir_module when module= is supplied; otherwise create builtin.ModuleOp(location=loc) [CARD:toolchain-call-graph-module-setup|MLIR Module Reuse or create] [LINK:graph_init:855] [LINK:ct_builtin_module:65]
   │  │  │    **MLIR.module is IR container** [LINK:ct_graph_module:562]
   │  │  └─ Place OpBuilder at module.body.end [CARD:toolchain-call-graph-op-builder|OpBuilder · Insertion point] [LINK:graph_init:858]
   │  ├─ Insert mo.GraphOp(builder, loc, name="addition", input_types=..., result_types=[]) [LINK:graph_init:860]
   │  │  ├─ Arguments passed to GraphOp, in source order [LINK:graph_init:861]
   │  │  │  ├─ builder (positional): the preceding OpBuilder(self._module.body.end); selects where this graph is inserted [LINK:graph_init:861]
   │  │  │  ├─ loc (positional): returned by with _location() as loc; supplies this GraphOp's source/profile metadata [LINK:graph_init:862]
   │  │  │  ├─ name=name (keyword): "addition", supplied by the caller's Graph("addition", ...) [LINK:entry:34] [LINK:graph_init:863]
   │  │  │  ├─ input_types=[t.to_mlir() for t in input_types] (keyword): convert the caller's Python input types to MLIR types [LINK:graph_init:864]
   │  │  │  │  ├─ caller supplies one TensorType(dtype, shape=[rows, columns], device=DeviceRef.from_device(device)) [LINK:entry:51]
   │  │  │  │  └─ example values: float32, shape [5, 10], and the selected CPU or accelerator; this describes symbolic X [LINK:entry:27] [LINK:entry:28] [LINK:entry:29] [LINK:entry:31]
   │  │  │  └─ result_types=[] (keyword): an empty list supplied by this constructor; Graph.output later records the output types [LINK:graph_init:865]
   │  │  ├─ Convert each Python input type with to_mlir() [CARD:toolchain-call-graph-type-conversion|to_mlir() · Input type conversion] [LINK:ct_tensor_to_mlir:430] [CALLER:graph_init|864]
   │  │  ├─ Native GraphOp builder records the signature, region and one block argument per input [CARD:toolchain-call-graph-native-builder|GraphOp · Region, block and signature] [LINK:ct_graph_op_build:831|C++:L831] [LINK:ct_graph_op_build:848|C++:L848]
   │  │  └─ OUT: block argument X is a symbolic input, not tensor data [LINK:ct_graph_op_build:849|C++:L849]
   │  ├─ Declare runtime input parameters for discovered symbolic dimensions [CARD:toolchain-call-graph-symbolic-params|Shape parameters · Symbolic dimensions] [LINK:graph_init:868]
   │  │    Here _params is empty, so the GraphOp receives no dynamic-dimension declarations [LINK:graph_init:870]
   │  ├─ Retain the GraphOp and its body block for later op insertion [CARD:toolchain-call-graph-retain-body|Graph handles · Operation and body] [LINK:graph_init:874]
   │  ├─ Attach module device-info mapping: CPU and any detected accelerator metadata [CARD:toolchain-call-graph-device-info|Device metadata · Module mapping] [LINK:graph_init:877] [LINK:ct_graph_device]
   │  ├─ Initialize weight and chain tracking [CARD:toolchain-call-graph-tracking|Weights and chains · Bookkeeping] [LINK:graph_init:879]
   │  │    _weights = {}; _has_chain_input = False; create the lazy per-device chain map [LINK:graph_init:881]
   │  ├─ Insert ChainCreateOp and store its result as the always-ready chain [CARD:toolchain-call-graph-ready-chain|Ordering chain · Always-ready seed] [LINK:graph_init:886] [LINK:graph_init:888]
   │  │    Seed the initial CPU ordering chain; this is an internal token, not a user tensor [LINK:graph_init:894]
   │  └─ Check whether the graph inputs include a trailing chain argument [CARD:toolchain-call-graph-trailing-chain|Chain input · Recognize and adopt] [LINK:graph_init:896]
   │       This example has only X, so no chain input is present; if one exists, adopt it [LINK:graph_init:900] [LINK:graph_init:904]
   │
   ├─ _import_kernels(...): prepare package P1 [LINK:ct_graph_import:1703|PY:L1703] [CALLER:graph_init|910]
   │  ├─ KernelLibrary.load_paths([kernels/]) [LINK:package_load:269|PY:L269] [CALLER:ct_graph_import|1704]
   │  │  └─ _build_mojo_source_package(kernels/) [LINK:package_builder:141|PY:L141] [CALLER:package_load|289]
   │  │     ├─ kernels/__init__.mojo [LINK:kernel_package_root:1|DOC:L1] + referenced child modules, including add_one.mojo [CALLER:ct_precompile|291]
   │  │     ├─ separate compiler process → Mojo executable [LINK:ct_mojo_run:70|PY:L70] + mojo precompile kernels/ -o staging.mojoc [LINK:package_builder:169|PY:L169] [CALLER:package_builder|178]
   │  │     │  └─ parse [LINK:ct_parser:392|C++:L392] → resolve [LINK:ct_resolver:1756|C++:L1756] → semantic/lifetime checks [LINK:ct_check_lit:1119|C++:L1119] → MLIR bytecode [LINK:ct_package_write:137|C++:L137] [CALLER:ct_precompile|290]
   │  │     └─ atomic publication → P1: one .mojoc package [LINK:package_builder:179|PY:L179] [CALLER:package_load|289]
   │  ├─ existing .mojoc path → load that package directly [LINK:package_load:286|PY:L286] [CALLER:ct_graph_import|1704]
   │  └─ package phase complete → [CARD:toolchain-call-graph-package|A.1 · parser, checks, file format and subprocess] [LINK:ct_graph_import:1713|PY:L1713] [CALLER:graph_init|910]
   │
   └─ Build forward inside `with self` [CARD:toolchain-call-graph-forward-context|with self · Record forward] [LINK:graph_init:921]
      ├─ __enter__ → _enter [CARD:toolchain-call-graph-forward-context|with self · Activate G1] [LINK:ct_graph_context_manager:1134]
      │    IN:  this Graph [LINK:ct_graph_context_manager:1135]
      │    OUT: Graph.current is set, so ops.custom can find it [LINK:ct_graph_context_manager:1145]
      │
      ├─ Read self.inputs [CARD:toolchain-call-graph-forward-inputs|self.inputs · Symbolic values] [LINK:ct_graph_inputs:935]
      │    IN:  graph body’s declared input block argument [LINK:ct_graph_inputs:943]
      │    OUT: one symbolic TensorValue, passed to the lambda as x [LINK:ct_graph_inputs:949]
      │
      ├─ Run forward(x) [CARD:toolchain-call-graph-forward-callback|forward(x) · Python records IR] [LINK:graph_init:922]
      │  ├─ Create output TensorType [CARD:toolchain-call-graph-forward-callback|Output type · Declare the result] [LINK:entry:43]
      │  │    IN:  x.dtype, x.tensor.shape, device [LINK:entry:44]
      │  │    OUT: float32 [5, 10] on the selected device [LINK:entry:43]
      │  │
      │  └─ ops.custom(name="add_one", values=[x], out_types=[...]) [CARD:toolchain-call-graph-custom-insert|mo.custom · Insert the call] [LINK:custom_api:69]
      │     ├─ Read Graph.current [CARD:toolchain-call-graph-custom-attributes|Call attributes · Active graph] [LINK:custom_api:94]
      │     ├─ Convert the name and device to MLIR attributes [CARD:toolchain-call-graph-custom-attributes|Call attributes · Name and device] [LINK:custom_api:95]
      │     ├─ Insert an MLIR `mo.custom` op into the graph body [CARD:toolchain-call-graph-custom-insert|mo.custom · Insert into G1] [LINK:custom_api:109]
      │     │    IN:  x and the declared output type [LINK:custom_api:109]
      │     │    OUT: one symbolic custom-op result [LINK:custom_api:109]
      │     ├─ Verify the op against the loaded kernel library [CARD:toolchain-call-graph-custom-verify|Kernel validation · P1 and MO] [LINK:custom_api:127]
      │     └─ Return [TensorValue] [CARD:toolchain-call-graph-custom-result|Result wrappers · Return symbolic values] [LINK:custom_api:129]
      │
      ├─ Select `[0].tensor` [CARD:toolchain-call-graph-custom-result|Result wrapper · Select the result] [LINK:entry:49]
      │    IN:  the one-element result list [LINK:entry:49]
      │    OUT: the TensorValue returned by the lambda [LINK:entry:49]
      │
      ├─ Call Graph.output(result) [CARD:toolchain-call-graph-forward-output|Finish · Output and exit] [LINK:graph_init:932]
      │  ├─ Record the single output [LINK:graph_output:1433]
      │  ├─ Add an output op and graph signature [LINK:graph_output:1458]
      │  └─ Verify the completed graph [LINK:graph_output:1489]
      │
      └─ __exit__ resets Graph.current [CARD:toolchain-call-graph-forward-output|Finish · Restore context] [LINK:ct_graph_context_manager:1138]
           OUT: the graph-building context is closed [LINK:ct_graph_context_manager:1149]

     Result of `Graph(...)`: a populated Graph object [LINK:entry:59]"""


def render_addition_calltrees(refs, revision, *, cards_with_callstack=False):
    used_refs = set()

    def link(key, label, line=None, *, caller=False, tree_id=None, call_stack=False):
        used_refs.add(key)
        r = refs[key]
        line = r["line"] if line is None else line
        if not r["start"] <= line <= r["end"]:
            raise ValueError(f"Call-tree line outside source definition: {key}:{line}")
        classes = "call-source qna-source-path"
        if caller:
            classes += " calltree-caller-link"
        elif call_stack:
            classes += " call-stack-source-link call-stack-source-link--path"
        caller_attrs = (f' data-caller-source="true" data-caller-tree="{html.escape(tree_id or "", quote=True)}" '
                        f'aria-label="Open caller source at {html.escape(r["file"], quote=True)}:L{line}"'
                        if caller else "")
        title = ("Caller source: " if caller else "") + f'{r["file"]}:{line}'
        return (f'<a class="{classes}" data-ref="{key}" data-line="{line}"{caller_attrs} '
                f'href="../sources/{html.escape(r["file"])}#L{line}" '
                f'title="{html.escape(title, quote=True)}">{html.escape(label)}'
                f' <small>[{html.escape(r["alias"])}:L{line}]</small></a>')

    def compact_source_link(key, line, label="↗", *, compact=True):
        used_refs.add(key)
        r = refs[key]
        if not r["start"] <= line <= r["end"]:
            raise ValueError(f"Call-stack source line outside source definition: {key}:{line}")
        compact_class = "call-stack-source-link--compact" if compact else "call-stack-source-link--path"
        return (
            f'<a class="call-source qna-source-path call-stack-source-link {compact_class}" '
            f'data-ref="{html.escape(key, quote=True)}" data-line="{line}" '
            f'href="../sources/{html.escape(r["file"])}#L{line}" '
            f'title="Open source at {html.escape(r["file"], quote=True)}:L{line}" '
            f'aria-label="Open source at {html.escape(r["file"], quote=True)}:L{line}">'
            f'{html.escape(label)}</a>'
        )

    def caller_annotated(text, fallback_key, fallback_line):
        """Add caller links to terminal branches and count explicit root links."""
        lines = text.rstrip().splitlines()
        explicit_caller_count = sum(len(CALLER_MARKER.findall(line)) for line in lines)
        roots = {0: (fallback_key, fallback_line)}
        nodes = []
        group = 0
        for index, line in enumerate(lines):
            match = TREE_BRANCH.match(line)
            if match:
                nodes.append(dict(index=index, group=group,
                                  depth=len(match["prefix"]) // 3,
                                  line=line))
            elif line.strip() and not line[0].isspace():
                group += 1
                markers = list(MARKER.finditer(line))
                roots[group] = ((markers[-1][1], fallback_line if markers[-1][1] == fallback_key else None) if markers
                                else (fallback_key, fallback_line))

        grouped = {}
        for node in nodes:
            grouped.setdefault(node["group"], []).append(node)

        additions = {}
        for group_id, group_nodes in grouped.items():
            stack = []
            for position, node in enumerate(group_nodes):
                while stack and stack[-1]["depth"] >= node["depth"]:
                    stack.pop()
                caller = stack[-1]["source"] if stack else roots[group_id]
                if not caller or caller[0] not in refs:
                    caller = (fallback_key, fallback_line)

                next_node = group_nodes[position + 1] if position + 1 < len(group_nodes) else None
                is_leaf = next_node is None or next_node["depth"] <= node["depth"]
                own_markers = list(MARKER.finditer(node["line"]))
                source = ((own_markers[-1][1], None) if own_markers else caller)

                if is_leaf and not CALLER_MARKER.search(node["line"]):
                    key, line = caller
                    line = refs[key]["line"] if line is None else line
                    if not refs[key]["start"] <= line <= refs[key]["end"]:
                        raise ValueError(f"Caller link outside source definition: {key}:{line}")
                    additions[node["index"]] = (key, line)
                stack.append(dict(depth=node["depth"], source=source))

        for index, (key, line) in additions.items():
            lines[index] += f" [CALLER:{key}|{line}]"
        return "\n".join(lines), len(additions) + explicit_caller_count

    caller_link_counts = {}

    def decorated(text, tree_id):
        # Escape prose and links separately. Fenced-code Markdown would escape
        # anchors and expose their markup instead of opening Source Explorer.
        result, last = [], 0
        for match in TREE_MARKER.finditer(text):
            result.append(html.escape(text[last:match.start()]))
            caller_match = CALLER_MARKER.fullmatch(match[0])
            if caller_match:
                result.append(link(caller_match[1], "↗ caller", int(caller_match[2]),
                                   caller=True, tree_id=tree_id))
            else:
                line_source_match = LINE_SOURCE_MARKER.fullmatch(match[0])
                if line_source_match:
                    key = line_source_match[1]
                    line = refs[key]["line"] if line_source_match[2] is None else int(line_source_match[2])
                    label = line_source_match[3]
                    if label is None:
                        result.append(compact_source_link(key, line))
                    elif re.fullmatch(r"(?:[A-Z]+|C\+\+):L\d+", label):
                        result.append(compact_source_link(key, line, f"[{label}]"))
                    else:
                        result.append(link(key, label, line, call_stack=tree_id == "graph-constructor-copy"))
                else:
                    source_match = MARKER.fullmatch(match[0])
                    label = source_match[2].replace("**", "")
                    result.append(link(
                        source_match[1], label,
                        call_stack=tree_id == "graph-constructor-copy",
                    ))
            last = match.end()
        result.append(html.escape(text[last:]))
        rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", "".join(result))
        code_tick = chr(96)
        rendered = re.sub(
            code_tick + r"([^" + code_tick + r"]+)" + code_tick,
            lambda match: '<span class="call-stack-inline-code">'
            + match[1]
            + "</span>",
            rendered,
        )
        rendered = CARD_MARKER.sub(
            lambda m: f'<a class="call-note-link" href="#{m[1]}" '
                      f'data-call-explanation="{m[1]}">{html.escape(m[2])}</a>', rendered,
        )
        rendered = re.sub(
            r"\[DETAIL:([a-z0-9_]+\.html(?:#[a-z0-9_-]+)?)\|([^\]]+)\]",
            lambda match: '<a class="call-note-link" href="' + match[1]
            + '">' + html.escape(match[2]) + '</a>', rendered,
        )
        return rendered.replace(
            "[EXPLANATION:context]",
            '<a class="call-note-link" href="#toolchain-call-graph-context" '
            'data-call-explanation="toolchain-call-graph-context">'
            'A.0 · default MLIR context</a>',
        ).replace(
            "[EXPLANATION:graph_objects]",
            '<a class="call-note-link" href="#toolchain-call-graph-objects" '
            'data-call-explanation="toolchain-call-graph-objects" '
            'title="Open the ModuleOp and GraphOp explanation">'
            'builtin.ModuleOp + mo.GraphOp(&quot;addition&quot;)</a>',
        )

    def explanation(text, tree_prefix, label, fallback_key, fallback_line, tree_callers=()):
        anchors = {}
        tree_widgets = {}
        tree_index = 0

        def render_calltree(match):
            nonlocal tree_index
            code = match[2]
            if not any(TREE_BRANCH.match(line) for line in code.splitlines()):
                return match[0]
            tree_index += 1
            key = (tree_callers[tree_index - 1]
                   if tree_index <= len(tree_callers) else fallback_key)
            line = (fallback_line if key == fallback_key and fallback_line is not None
                    else refs[key]["line"])
            token = f"EXPLANATION_CALLTREE_{tree_index}_TOKEN"
            tree_widgets[token] = tree(
                code, f"{label} · nested tree {tree_index}",
                f"{tree_prefix}-diagram-{tree_index}", key, line,
            )
            return f"\n\n{token}\n\n"

        def source_anchor(match):
            file, line = match[2], int(match[3])
            candidates = [key for key, r in refs.items()
                          if r["file"] == file and r["start"] <= line <= r["end"]]
            if not candidates:
                raise ValueError(f"Explanation source line has no definition span: {file}:{line}")
            key = min(candidates, key=lambda k: (refs[k]["end"] - refs[k]["start"], k))
            r = refs[key]
            used_refs.add(key)
            token = f"EXPLANATION_SOURCE_{len(anchors)}_TOKEN"
            label = markdown.markdown(match[1]).removeprefix("<p>").removesuffix("</p>")
            anchors[token] = (
                f'<a class="qna-source-path" data-ref="{key}" data-line="{line}" '
                f'href="../sources/{html.escape(r["file"])}#L{line}">'
                f'{label}</a>'
            )
            return token

        text = re.sub(r"```([^\n]*)\n(.*?)\n```", render_calltree, text, flags=re.S)
        if tree_callers and tree_index != len(tree_callers):
            raise ValueError(
                f"Expected {len(tree_callers)} caller-source trees in {tree_prefix}, found {tree_index}"
            )
        text = re.sub(
            r"\[([^\]]+)\]\(sources/([^\s)]+)#L(\d+)\)",
            source_anchor, text,
        )
        rendered = markdown.markdown(text, extensions=["tables", "fenced_code"])
        for token, anchor in anchors.items():
            rendered = rendered.replace(token, anchor)
        for token, widget in tree_widgets.items():
            rendered = rendered.replace(f"<p>{token}</p>", widget).replace(token, widget)
        rendered = re.sub(r"(<table>.*?</table>)",
                          r'<div class="qna-table-scroll" tabindex="0">\1</div>',
                          rendered, flags=re.S)
        return '<div class="call-explanation">' + rendered + '</div>'

    def tree(text, label, tree_id, fallback_key, fallback_line):
        marked, leaf_count = caller_annotated(text, fallback_key, fallback_line)
        caller_link_counts[tree_id] = leaf_count
        return (f'<div class="call-tree-scroll" tabindex="0" role="region" '
                f'data-call-tree-id="{html.escape(tree_id, quote=True)}" '
                f'aria-label="{html.escape(label)}; scroll horizontally for long branches">'
                '<pre class="call-tree"><code>' + decorated(marked, tree_id) + '</code></pre></div>')

    def portable(text, fallback_key, fallback_line):
        text, _ = caller_annotated(text, fallback_key, fallback_line)

        def caller_reference(match):
            r = refs[match[1]]
            return f'↗ caller [{r["file"]}:L{match[2]}]'

        def reference(match):
            r = refs[match[1]]
            return f'{match[2].replace("**", "")} [{r["file"]}:L{r["line"]}]'
        text = re.sub(r"\[DETAIL:([^|]+)\|([^\]]+)\]", lambda m: m[2] + " → " + m[1], text)
        return (CARD_MARKER.sub(lambda m: m[2], MARKER.sub(reference, CALLER_MARKER.sub(caller_reference, text))).replace("**", "")
                .replace("[EXPLANATION:context]", "Full explanation · is this a global variable? → A.0")
                .replace("[EXPLANATION:graph_objects]", 'builtin.ModuleOp + mo.GraphOp("addition")')
                .rstrip())

    def graph_constructor_call_stack_panel():
        rendered_lines = []
        source_link_count = 0
        caller_link_count = 0
        for raw_line in GRAPH_CONSTRUCTOR_CALL_STACK_TREE.splitlines():
            caller_link_count += len(CALLER_MARKER.findall(raw_line))
            markers = list(LINE_SOURCE_MARKER.finditer(raw_line))
            if not markers:
                rendered_lines.append(html.escape(raw_line))
                continue
            visible_line = raw_line
            context_card_marker = "[CARD:context]"
            open_context_card = context_card_marker in visible_line
            visible_line = visible_line.replace(context_card_marker, "").rstrip()
            source_link_count += len(markers)
            formatted_line = decorated(visible_line, "graph-constructor-copy")
            if open_context_card:
                formatted_line += (
                    ' <a class="call-note-link call-stack-context-card-link" '
                    'href="#toolchain-call-graph-context" '
                    'data-call-explanation="toolchain-call-graph-context" '
                    'title="Open the A.0 default MLIR context explanation">'
                    'A.0 · default MLIR context</a>'
                )
            copy_line = LINE_SOURCE_MARKER.sub(
                lambda match: (
                    f"[{match[3]}]"
                    if match[3] and re.fullmatch(r"(?:[A-Z]+|C\+\+):L\d+", match[3])
                    else match[3] or "↗"
                ),
                visible_line,
            )
            copy_line = CALLER_MARKER.sub(
                lambda match: f'↗ caller [{refs[match[1]]["alias"]}:L{match[2]}]',
                copy_line,
            )
            copy_line = CARD_MARKER.sub(lambda match: match[2], copy_line)
            rendered_lines.append(
                '<span class="call-stack-copy-line" '
                f'data-copy-line="{html.escape(copy_line, quote=True)}">'
                f'<span class="call-stack-line-text">{formatted_line}</span>'
                '</span>'
            )

        caller_link_counts["graph-constructor-copy"] = caller_link_count
        example_link = compact_source_link(
            "entry", 33, "addition.py (max/examples/custom_ops/addition.py:24)",
            compact=False,
        )
        source_link_count += 1
        return (
            '<article class="call-stack-copy">'
            '<h2>What happens when Graph(...) runs</h2>'
            '<p>In this example, Graph(...) builds a symbolic graph immediately. '
            'Its forward lambda runs inside the constructor, but it receives a '
            'symbolic input value—not the matrix data that will be supplied later '
            'to model.execute. The matching example is in '
            + example_link
            + '.</p>'
            '<div class="call-tree-scroll call-stack-copy-tree" id="graph-constructor-copy" '
            'data-call-tree-id="graph-constructor-copy" '
            'tabindex="0" role="region" '
            'aria-label="What happens when Graph runs; scroll horizontally for long branches">'
            '<pre class="call-tree"><code>'
            + "\n".join(rendered_lines)
            + '</code></pre></div></article>',
            source_link_count,
        )

    support_cards = []
    body = ['<div class="addition-calltrees" data-calltrees="addition">',
            '<p class="calltree-scope"><strong>SOURCE</strong> · '
            f'Revision <code>{revision[:12]}</code> · '
            'Follow P1 → G1, then S1 → CM1 → M1. Click numbered details to open a card; '
            'each terminal branch ends with a ↗ caller link to its enclosing source.</p>']
    reading_order = (
        "Reading order: A.1 kernels/*.mojo → P1 (.mojoc); "
        "A.2 P1 + Graph(...) → G1 (MO); "
        "B InferenceSession(...) → S1; "
        "C compile(G1) → C1: MO → MAP → MOGG → MGP → CM1; "
        "D init(CM1) → M1."
    )
    body.append('<p class="call-lifecycle"><strong>Reading order</strong> · '
                '<b>A.1</b> <code>kernels/*.mojo → P1 (.mojoc)</code> → '
                '<b>A.2</b> <code>P1 + Graph(...) → G1 (MO)</code> → '
                '<b>B</b> <code>InferenceSession(...) → S1</code> → '
                '<b>C</b> <code>compile(G1) → C1: MO → MAP → MOGG → MGP → CM1</code> → '
                '<b>D</b> <code>init(CM1) → M1</code>.</p>')
    companion = ["# Add-one: Graph, session, compile and init call trees", "",
                 f"SOURCE · repository revision {revision}. Numerical values are illustrative; no MAX execution.",
                 "", reading_order, ""]
    for stage in STAGES:
        title = stage["letter"] + " · " + stage["title"]
        body.append(f'<article class="call-stage" id="toolchain-call-{stage["id"]}" data-call-stage="{stage["id"]}">'
                    f'<h3><span class="call-stage-letter">{stage["letter"]}</span>{html.escape(stage["title"])}</h3>'
                    '<p class="callsite">' + link(stage["source"], stage["call"], stage["line"]) + '</p>'
                    '<dl class="call-contracts">')
        companion.extend(["## " + title, "", stage["call"], ""])
        for label, value in stage["contracts"]:
            body.append(f'<div><dt>{label}</dt><dd>{html.escape(value)}</dd></div>')
            companion.append(label.upper() + ": " + value)
        body.append('</dl>')
        if "note" in stage:
            body.append('<p class="call-reading-note">' + html.escape(stage["note"]) + '</p>')
            companion.extend(["", stage["note"]])
        body.append(tree(stage["tree"], title, stage["id"], stage["source"], stage["line"]))
        companion.extend(["", "```text", portable(stage["tree"], stage["source"], stage["line"]), "```", ""])
        for detail in stage["details"]:
            detail_tree_id = f'{stage["id"]}-{detail["id"]}'
            content = (tree(detail["tree"], detail["title"], detail_tree_id,
                            stage["source"], stage["line"]) if "tree" in detail else "")
            if "explanation" in detail:
                content += explanation(
                    detail["explanation"], detail_tree_id, detail["title"],
                    stage["source"], stage["line"], detail.get("tree_callers", ()),
                )
            animation_key = detail["id"] if stage["id"] == "graph" else stage["id"] + "-" + detail["id"]
            animation_source = detail.get("tree_callers", [stage["source"]])[0]
            content = render_card_conversion(animation_key, animation_source, refs) + content
            card_markup = (f'<details class="call-branch" data-call-card '
                        f'id="toolchain-call-{stage["id"]}-{detail["id"]}" '
                        f'data-explanation-title="{html.escape(detail["title"])}">'
                        f'<summary>{html.escape(detail["title"])}</summary>'
                        '<div class="call-card-content">' + content + '</div></details>')
            body.append(card_markup)
            support_cards.append(card_markup)
            companion.extend(["### " + detail["title"], ""])
            if "tree" in detail:
                companion.extend(["```text", portable(detail["tree"], stage["source"], stage["line"]), "```", ""])
            if "explanation" in detail:
                companion.extend([detail["explanation"].rstrip(), ""])
        body.append('</article>')
    example_launcher = ('<div class="call-card-launchers call-example"><button type="button" hidden '
                'class="call-card-launch" id="toolchain-call-example-launch" data-call-card-launch '
                'data-call-explanation="toolchain-call-example">'
                '<span class="call-card-number">E</span><span>Input buffers → execution → result</span>'
                '<span class="call-card-glyph" aria-hidden="true">↗</span></button></div>')
    body.append(example_launcher)
    example_card = ('<details class="call-branch call-example" data-call-card '
                'id="toolchain-call-example" data-explanation-title="E · Input buffers → execution → result"><summary>'
                'E · Connect the prepared objects to an input and a result</summary>'
                '<div class="call-card-content">'
                + render_card_conversion("example", "entry", refs)
                + tree(EXAMPLE, "Add-one execution and illustrative values", "example", "entry", 71) + '</div></details>')
    body.append(example_card + '</div>')
    support_cards.append(example_card)
    explanation_card = ('<aside class="call-explanation-card" id="call-explanation-card" hidden '
                'role="dialog" aria-modal="false" aria-labelledby="call-explanation-title">'
                '<div class="call-explanation-card-header"><h2 id="call-explanation-title">'
                'Default MLIR context</h2><button type="button" data-close-call-explanation '
                'aria-label="Close explanation">Close ×</button></div>'
                '<div class="call-explanation-card-body"></div></aside>')
    body.append(explanation_card)
    companion.extend(["## E · Input and output example", "", "```text", portable(EXAMPLE, "entry", 71), "```", ""])
    text = "\n".join(companion)
    graph_call_stack_html, graph_call_stack_source_links = graph_constructor_call_stack_panel()
    for panel in SESSION_CALL_STACK_PANELS:
        panel_html = ('<article class="call-stack-copy session-call-stack" '
                      f'id="{panel["id"]}-section"><h2>{html.escape(panel["title"])}</h2>'
                      '<p class="callsite">' + link("entry", panel["call"], panel["line"]) + '</p>'
                      '<p>' + html.escape(panel["intro"]) + '</p>'
                      + ('<pre class="example-code"><code>' + html.escape(panel["code"]) + '</code></pre>' if panel.get("code") else '')
                      + tree(panel["tree"], panel["title"], panel["id"], "entry", panel["line"])
                      + '</article>')
        graph_call_stack_html += panel_html
    if cards_with_callstack:
        graph_call_stack_html += (example_launcher
            + '<div class="call-stack-card-storage">' + ''.join(support_cards) + '</div>'
            + explanation_card)
        for stage in STAGES:
            caller_link_counts.pop(stage['id'], None)
    return "".join(body), text, dict(
        stages=[stage["id"] for stage in STAGES],
        branches=sum(len(stage["details"]) for stage in STAGES) + 1,
        cards=[f'toolchain-call-{stage["id"]}-{detail["id"]}'
               for stage in STAGES for detail in stage["details"]] + ["toolchain-call-example"],
        reading_order=["kernels/*.mojo → P1 (.mojoc)", "P1 → recorded G1 (MO)",
                       "devices → S1", "S1 + G1 → clone C1: MO → MAP → MOGG → MGP → CM1",
                       "S1 + CM1 → initialized M1"],
        references=sorted(used_refs),
        caller_link_counts=caller_link_counts,
        caller_links=sum(caller_link_counts.values()),
        graph_constructor_source_links=graph_call_stack_source_links,
        separate_call_stack_sections=["graph-constructor-copy", *[p["id"] for p in SESSION_CALL_STACK_PANELS]],
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    ), graph_call_stack_html
