"""Annotated, source-linked call trees for the add-one Text reader.

The strings are portable trees as well as the HTML's authoring source. Source
markers become real offline links; no compiler or inference is run here.
"""
from __future__ import annotations

import hashlib
import html
import re

import markdown

GRAPH = "max/python/max/graph/graph.py"
API = "max/python/max/engine/api.py"
CORE = "max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp"
CONTEXT = "max/lib/max_license/API/c/EngineContext.cpp"
COMPILED = "max/lib/max_license/API/c/CompiledModels.cpp"
MODEL = "GraphCompiler/lib/MEFModel/MEFModel.cpp"

# Full unchanged files are embedded in Source Explorer and validated against
# the same revision as the animation. Existing compiler/kernel refs are reused.
SOURCES = [
    ("ct_graph_context", "max/python/max/_mlir_context.py", "in_default_mlir_context", "python", 0, "Graph constructor's MLIR-context decorator"),
    ("ct_graph_context_state", "max/python/max/_mlir_context.py", "26:67", "lines", 0, "Shared default MLIR context and thread-local active-context guard"),
    ("ct_graph_device", GRAPH, "Graph._populate_device_info_mapping", "python", 0, "Graph's device-to-target metadata"),
    ("ct_graph_extensions", GRAPH, "_resolved_custom_extensions", "python", 0, "Explicit extensions and default overlays"),
    ("ct_graph_library", GRAPH, "KernelLibrary.__init__", "python", 0, "KernelLibrary creates native analysis"),
    ("ct_graph_import", GRAPH, "Graph._import_kernels", "python", 0, "Graph records resolved kernel package paths"),
    ("ct_graph_verify", GRAPH, "KernelLibrary.verify_custom_op", "python", 0, "Custom-op verification and analysis reuse"),
    ("ct_mojo_run", "Mojo/python/mojo/run.py", "subprocess_run_mojo", "python", 0, "Select and invoke the Mojo compiler process"),
    ("ct_precompile", "Mojo/tools/mojo/Precompile/mojo-precompile.cpp", "static int precompile(const State &subcommandState)", "cpp", 0, "mojo precompile: parse, build, check, serialize"),
    ("ct_parser", "Mojo/lib/MojoParser/EntryPoint.cpp", "LIT::importMojoPackage(ContextRef context,", "cpp", 1, "Parse a directory as a Mojo package"),
    ("ct_resolver", "Mojo/lib/MojoParser/DeclResolver.cpp", "void DeclResolver::resolveAllReferencedFrom(ASTDecl &decl)", "cpp", 0, "Resolve package modules and referenced dependencies"),
    ("ct_package_build", "Mojo/lib/Precompile/Precompile.cpp", "ErrorOr<OwningOpRef<ModuleOp>> M::KGEN::buildPrecompiledPackage(", "cpp", 0, "Clone post-parse package IR for serialization"),
    ("ct_check_lit", "Mojo/lib/Compiler/KGENCompiler.cpp", "LogicalResult KGENCompiler::runCheckLITPipeline(ModuleOp module)", "cpp", 0, "Run semantic and lifetime checks on parsed package IR"),
    ("ct_package_write", "Mojo/lib/Support/MojoPrecompiledFile.cpp", "LogicalResult M::KGEN::writePrecompiledFile(Operation *op,\n                                            MojoPrecompiledFileVersion &mojoVer,", "cpp", 0, "MPKG V2 header and zstd-compressed MLIR bytecode"),
    ("ct_analysis_verify", "max/python/max/_core/max_license/lib/Graph/Bindings.cpp", "void M::Graph::Python::Analysis::verifyCustomOp(MlirOperation op)", "cpp", 0, "Native custom-op validation against MojoRegistry"),
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

There are **two distinct pieces of state**:

| Name | Scope | Purpose |
|---|---|---|
| `_DEFAULT_MLIR_CONTEXT` | Module global | Holds the shared default context object. |
| `mlir.Context.current` | Thread local | Identifies the context currently active on this thread. |

The source explicitly documents this distinction in [_mlir_context.py:27](sources/max/python/max/_mlir_context.py#L27).

Your call-tree line is more precisely:

```text
Graph(...)
└─ @in_default_mlir_context
   ├─ This thread already has an active context
   │  └─ Keep that context; call Graph.__init__
   └─ This thread has no active context
      └─ Temporarily enter _DEFAULT_MLIR_CONTEXT
         └─ Call Graph.__init__
```

That conditional behavior is implemented in [ensure_default_mlir_context:55](sources/max/python/max/_mlir_context.py#L55). **“Enter” makes a context active for IR-building APIs; entering alone does not compile or execute the graph.** ([mlir.llvm.org](https://mlir.llvm.org/docs/Bindings/Python/))
"""


STAGES = [
    dict(
        id="graph", letter="A", title="Graph(...) → G1 + package P1", short="Graph(...) → G1 + P1",
        call='graph = Graph("addition", forward=..., input_types=..., custom_extensions=[mojo_kernels])',
        source="entry", line=33,
        contracts=[
            ("Input", 'name="addition"; symbolic X: float32[5,10] @ device; forward callback; kernels/'),
            ("Output", 'G1: Python Graph + mo.graph module; P1: resolved .mojoc package path'),
            ("Example", 'forward records custom("add_one", [X]) → symbolic Y: float32[5,10]'),
        ],
        tree="""[SOURCE:graph_init|Graph.__init__(name, forward, input_types, custom_extensions)]
├─ [SOURCE:ct_graph_context|in_default_mlir_context] → ensure this thread has an active MLIR context
│  └─ [EXPLANATION:context]
│
├─ create builtin.ModuleOp + mo.GraphOp("addition") → **G1**
│  ├─ input TensorType → MLIR block argument: **X: float32[5,10]**
│  ├─ _params = {} → this shape has no symbolic dimensions
│  ├─ [SOURCE:ct_graph_device|_populate_device_info_mapping()] → CPU/GPU target metadata
│  └─ _weights = {}; device chains + always-ready chain → ordering tokens
│
├─ [SOURCE:ct_graph_library|KernelLibrary()] → native kernel Analysis
├─ [SOURCE:ct_graph_extensions|_resolved_custom_extensions([kernels/])] → paths + default overlays
├─ [SOURCE:ct_graph_import|_import_kernels(paths)]
│  └─ [SOURCE:package_load|KernelLibrary.load_paths(paths)]
│     ├─ source directory + __init__.mojo → [SOURCE:package_builder|_build_mojo_source_package(kernels/)]
│     │  └─ **mojo precompile kernels/ -o staging.mojoc** → publish **P1**
│     └─ existing .mojoc → add directly; skip the source-package subprocess
│
├─ with G1 → forward(*self.inputs) runs once with **symbolic X**
│  ├─ [SOURCE:custom_api|ops.custom(name="add_one", device, values=[X], out_types=[...])]
│  │  ├─ record mo.custom → registration name + target + result type
│  │  └─ [SOURCE:ct_graph_verify|KernelLibrary.verify_custom_op(op)] → validate Mojo contract
│  └─ callback returns the custom result's .tensor → **symbolic Y**
├─ [SOURCE:graph_output|Graph.output(Y)] → mo.output + signature + whole-graph verification
└─ return **G1** → recorded computation; **P1** available for later specialization
""",
        details=[
            dict(id="context", title="A.0 · What is the default MLIR context? Is it a global variable?",
                 explanation=DEFAULT_CONTEXT_EXPLANATION),
            dict(id="package", title="A.1 · Expand the .mojo → .mojoc compiler subprocess", tree="""[SOURCE:package_builder|_build_mojo_source_package(kernels_dir)]  [parent Python process]
├─ input: absolute directory with [SOURCE:kernel_package_root|kernels/__init__.mojo]
├─ hash = MD5(absolute path string) → a stable filename, not a source-content hash
├─ final: <tempdir>/.modular_<uid>/mojo_pkg/mojo_pkg_<hash>.mojoc
├─ staging: .staging.<pid>/mojo_pkg_<hash>.mojoc
├─ [SOURCE:ct_mojo_run|subprocess_run_mojo(["precompile", kernels_dir, "-o", staging], check=True)]
│  ├─ executable: MODULAR_MOJO_MAX_DRIVER_PATH, otherwise mojo on PATH
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
│           └─ MLIR bytecode → zstd payload + **MPKG V2** header → **staging .mojoc**
│
├─ compiler succeeds → os.replace(staging, final) → atomic publication
├─ finally → remove staging directory
└─ return final Path → **P1**; **addition.py stays a separate Python program**
   The source-directory branch invokes precompile again; it has no exists() skip.
"""),
            dict(id="validation", title="A.2 · Expand registration, validation and Graph.output", tree="""[SOURCE:custom_api|ops.custom("add_one", values=[X], out_types=[Y_type], device)]
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
         └─ native session owns this Engine::Context → **S1 ready for compile/init**
""",
        details=[dict(id="ownership", title="B.1 · Expand ownership, threads and constructor examples", tree="""[SOURCE:ct_session|Constructor input → normalized output]
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
Constructor result → configuration + ownership; G1 is supplied later to compile.
""" )],
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
│  │  │                    └─ miss/disabled → **MO → MAP → MOGG → MGP → kernel JIT → MEF**
│  │  └─ done callback → release compilation lock when the task resolves
│  └─ wrap pending handle → [SOURCE:artifact|CompiledModel(handle, expected_weights={})] → **CM1**
│
├─ handle.wait() → caller waits for completion; exception() → raise on failure
└─ return **resolved CM1** → graph names=("addition",); executable state comes from init
   **G1 remains with the caller**; native lowering transforms the clone **C1**.
""",
        details=[
            dict(id="passes", title="C.1 · Expand MO → MAP → MOGG → MGP and their actual owners", tree="""[SOURCE:frontend|MOFContext::compileToBinary] → owns the compilation route for **C1**
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

**Add-one ancestry** [schematic representation; not a captured compiler dump]
└─ mo.custom("add_one", X) → implementation-oriented accesses/call
   → selected AddOne routine → buffer/state execution plan → compiled callable
"""),
            dict(id="kernels", title="C.2 · Expand specialization, LLVM translation, targets and JIT", tree="""[SOURCE:jit_entry|compileMojoToBinary(C1, emitted Mojo, ...)]
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
            dict(id="mef", title="C.3 · Expand MEF preparation, compiled descriptors and cache boundaries", tree="""[SOURCE:ct_convert_mef|convertMOToMEF(C1)]  [after graph passes + native routine JIT]
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
│  ├─ registry = {}; expected_weights = {} → no weight-binding validation failures
│  ├─ [SOURCE:ct_session_binding|_impl._load_all(compiled_handle, registry)] → share artifact handle
│  │  └─ [SOURCE:ct_load_all|Core::InferenceSession::loadAll(handle, weightsDict)]
│  │     ├─ [SOURCE:ct_weight_dict|buildWeightsFromDict({})] → empty native registry + managed-tensor list
│  │     ├─ await compiled handle (release GIL while waiting) → descriptor "addition"
│  │     └─ for each model descriptor → [SOURCE:ct_load|Core::InferenceSession::load(model, registry)]
│  │        └─ [SOURCE:ct_context_init|Engine::Context::initModel(model, registry)]
│  │           └─ addTask(cpuDevice, ...) → initializer runs on a host runtime worker
│  │              └─ [SOURCE:ct_executable_create|createExecutableModel(model, cpuDevice, config, registry)]
│  │                 ├─ DeviceMap from session DeviceSpecs + retained driver contexts
│  │                 ├─ [SOURCE:ct_model_setup|Model::setup(cpuDevice, registry, deviceMap)]
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
            dict(id="resources", title="D.1 · Expand initialization resources, loading and readiness", tree="""[SOURCE:ct_model_setup|Model::setup] → compiled initFn.invoke(registry + devices)
├─ [SOURCE:ct_device_map|mapDeviceRefsToDevices]
│  ├─ synchronize supplied driver contexts before first resource use
│  ├─ CPU DeviceRef → create legacy runtime CPU context
│  └─ GPU DeviceRef → reuse supplied context; fallback create by device id if absent
├─ CUDA/HIP contexts → ensure configured stream count before capture (default: 2)
├─ invoke the compiled init region → runtime/state initialization and native routine loading
│  └─ [SOURCE:runtime_calls|ModuleLoad planned by LowerCompiledRoutinesPass]
│     └─ [SOURCE:ct_module_load|mgp_module_load_cpu] → CompiledKernelsContext in a runtime state slot
│        └─ [SOURCE:ct_kernel_load|CompiledKernelsContext::load(archive_bytes, symbols)]
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
"""),
            dict(id="gates", title="D.2 · Expand initialization gates and retained ownership", tree="""[SOURCE:init_all|init_all initialization gates]
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


def render_addition_calltrees(refs, revision):
    used_refs = set()

    def link(key, label, line=None):
        used_refs.add(key)
        r = refs[key]
        line = r["line"] if line is None else line
        if not r["start"] <= line <= r["end"]:
            raise ValueError(f"Call-tree line outside source definition: {key}:{line}")
        return (f'<a class="call-source qna-source-path" data-ref="{key}" data-line="{line}" '
                f'href="../sources/{html.escape(r["file"])}#L{line}" '
                f'title="{html.escape(r["file"])}:{line}">{html.escape(label)}'
                f' <small>[{html.escape(r["alias"])}:L{line}]</small></a>')

    def decorated(text):
        # Escape prose and links separately. Fenced-code Markdown would escape
        # anchors and expose their markup instead of opening Source Explorer.
        result, last = [], 0
        for match in MARKER.finditer(text):
            result.append(html.escape(text[last:match.start()]))
            label = match[2].replace("**", "")
            result.append(link(match[1], label))
            last = match.end()
        result.append(html.escape(text[last:]))
        rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", "".join(result))
        return rendered.replace(
            "[EXPLANATION:context]",
            '<a class="call-note-link" href="#toolchain-call-graph-context">'
            'Full explanation · is this a global variable?</a>',
        )

    def explanation(text):
        anchors = {}

        def source_anchor(match):
            key = "ct_graph_context_state"
            line = int(match[2])
            r = refs[key]
            if not r["start"] <= line <= r["end"]:
                raise ValueError(f"Context explanation line outside source span: {line}")
            used_refs.add(key)
            token = f"CONTEXT_SOURCE_{len(anchors)}_TOKEN"
            anchors[token] = (
                f'<a class="qna-source-path" data-ref="{key}" data-line="{line}" '
                f'href="../sources/{html.escape(r["file"])}#L{line}">'
                f'{html.escape(match[1])}</a>'
            )
            return token

        text = re.sub(
            r"\[([^\]]+)\]\(sources/max/python/max/_mlir_context\.py#L(\d+)\)",
            source_anchor, text,
        )
        rendered = markdown.markdown(text, extensions=["tables", "fenced_code"])
        for token, anchor in anchors.items():
            rendered = rendered.replace(token, anchor)
        rendered = re.sub(r"(<table>.*?</table>)",
                          r'<div class="qna-table-scroll" tabindex="0">\1</div>',
                          rendered, flags=re.S)
        return '<div class="call-explanation">' + rendered + '</div>'

    def tree(text, label):
        return (f'<div class="call-tree-scroll" tabindex="0" role="region" '
                f'aria-label="{html.escape(label)}; scroll horizontally for long branches">'
                '<pre class="call-tree"><code>' + decorated(text.rstrip()) + '</code></pre></div>')

    def portable(text):
        def reference(match):
            r = refs[match[1]]
            return f'{match[2].replace("**", "")} [{r["file"]}:L{r["line"]}]'
        return (MARKER.sub(reference, text).replace("**", "")
                .replace("[EXPLANATION:context]", "Full explanation · is this a global variable? → A.0")
                .rstrip())

    body = ['<div class="addition-calltrees" data-calltrees="addition">',
            '<p class="calltree-scope"><strong>SOURCE</strong> · '
            f'Revision <code>{revision[:12]}</code> · '
            'Click a function to open its unchanged source. Expand a branch for compiler internals.</p>',
            '<nav class="calltree-nav" aria-label="Add-one call trees">']
    for stage in STAGES:
        body.append(f'<a href="#toolchain-call-{stage["id"]}"><span>{stage["letter"]}</span>'
                    f'<strong>{html.escape(stage["short"])}</strong></a>')
    body.append('</nav>')
    companion = ["# Add-one: Graph, session, compile and init call trees", "",
                 f"SOURCE · repository revision {revision}. Numerical values are illustrative; no MAX execution.", ""]
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
        body.append('</dl>' + tree(stage["tree"], title))
        companion.extend(["", "```text", portable(stage["tree"]), "```", ""])
        for detail in stage["details"]:
            content = (explanation(detail["explanation"]) if "explanation" in detail
                       else tree(detail["tree"], detail["title"]))
            body.append(f'<details class="call-branch" id="toolchain-call-{stage["id"]}-{detail["id"]}">'
                        f'<summary>{html.escape(detail["title"])}</summary>'
                        + content + '</details>')
            companion.extend(["### " + detail["title"], ""])
            if "explanation" in detail:
                companion.extend([detail["explanation"].rstrip(), ""])
            else:
                companion.extend(["```text", portable(detail["tree"]), "```", ""])
        body.append('</article>')
    body.append('<details class="call-branch call-example" id="toolchain-call-example"><summary>'
                'E · Connect the prepared objects to an input and a result</summary>'
                + tree(EXAMPLE, "Add-one execution and illustrative values") + '</details></div>')
    companion.extend(["## E · Input and output example", "", "```text", portable(EXAMPLE), "```", ""])
    text = "\n".join(companion)
    return "".join(body), text, dict(
        stages=[stage["id"] for stage in STAGES],
        branches=sum(len(stage["details"]) for stage in STAGES) + 1,
        references=sorted(used_refs),
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
