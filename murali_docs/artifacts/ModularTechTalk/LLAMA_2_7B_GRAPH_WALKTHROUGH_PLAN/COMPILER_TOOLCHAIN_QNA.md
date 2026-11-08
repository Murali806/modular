# Add-one: Graph, session, compile and init call trees

SOURCE · repository revision 610f7111836d9a950d45ba52ec9b3e0ed589fe3f. Numerical values are illustrative; no MAX execution.

## A · Graph(...) → G1 + package P1

graph = Graph("addition", forward=..., input_types=..., custom_extensions=[mojo_kernels])

INPUT: name="addition"; symbolic X: float32[5,10] @ device; forward callback; kernels/
OUTPUT: G1: Python Graph + mo.graph module; P1: resolved .mojoc package path
EXAMPLE: forward records custom("add_one", [X]) → symbolic Y: float32[5,10]

```text
Graph.__init__(name, forward, input_types, custom_extensions) [max/python/max/graph/graph.py:L819]
├─ in_default_mlir_context [max/python/max/_mlir_context.py:L70] → enter the default MLIR context
├─ create builtin.ModuleOp + mo.GraphOp("addition") → G1
│  ├─ input TensorType → MLIR block argument: X: float32[5,10]
│  ├─ _params = {} → this shape has no symbolic dimensions
│  ├─ _populate_device_info_mapping() [max/python/max/graph/graph.py:L1366] → CPU/GPU target metadata
│  └─ _weights = {}; device chains + always-ready chain → ordering tokens
│
├─ KernelLibrary() [max/python/max/graph/graph.py:L245] → native kernel Analysis
├─ _resolved_custom_extensions([kernels/]) [max/python/max/graph/graph.py:L363] → paths + default overlays
├─ _import_kernels(paths) [max/python/max/graph/graph.py:L1703]
│  └─ KernelLibrary.load_paths(paths) [max/python/max/graph/graph.py:L269]
│     ├─ source directory + __init__.mojo → _build_mojo_source_package(kernels/) [Mojo/python/mojo/paths.py:L141]
│     │  └─ mojo precompile kernels/ -o staging.mojoc → publish P1
│     └─ existing .mojoc → add directly; skip the source-package subprocess
│
├─ with G1 → forward(*self.inputs) runs once with symbolic X
│  ├─ ops.custom(name="add_one", device, values=[X], out_types=[...]) [max/python/max/graph/ops/custom.py:L69]
│  │  ├─ record mo.custom → registration name + target + result type
│  │  └─ KernelLibrary.verify_custom_op(op) [max/python/max/graph/graph.py:L307] → validate Mojo contract
│  └─ callback returns the custom result's .tensor → symbolic Y
├─ Graph.output(Y) [max/python/max/graph/graph.py:L1396] → mo.output + signature + whole-graph verification
└─ return G1 → recorded computation; P1 available for later specialization
```

### A.1 · Expand the .mojo → .mojoc compiler subprocess

```text
_build_mojo_source_package(kernels_dir) [Mojo/python/mojo/paths.py:L141]  [parent Python process]
├─ input: absolute directory with kernels/__init__.mojo [max/examples/custom_ops/kernels/__init__.mojo:L1]
├─ hash = MD5(absolute path string) → a stable filename, not a source-content hash
├─ final: <tempdir>/.modular_<uid>/mojo_pkg/mojo_pkg_<hash>.mojoc
├─ staging: .staging.<pid>/mojo_pkg_<hash>.mojoc
├─ subprocess_run_mojo(["precompile", kernels_dir, "-o", staging], check=True) [Mojo/python/mojo/run.py:L57]
│  ├─ executable: MODULAR_MOJO_MAX_DRIVER_PATH, otherwise mojo on PATH
│  └─ subprocess.run(...) → Python waits for the separate compiler process
│     └─ mojo precompile: precompile(State) [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209]
│        ├─ validate source package / names; create compiler context; open output
│        ├─ invokeMojoParser(...)
│        │  └─ LIT::importMojoPackage(kernels_dir, "kernels", ...) [Mojo/lib/MojoParser/EntryPoint.cpp:L391]
│        │     └─ DeclResolver::resolveAllReferencedFrom(package) [Mojo/lib/MojoParser/DeclResolver.cpp:L1756]
│        │        ├─ parse the package + child modules, including add_one.mojo [max/examples/custom_ops/kernels/add_one.mojo:L1]
│        │        └─ resolve referenced dependency declarations and bodies
│        ├─ buildPrecompiledPackage(...) [Mojo/lib/Precompile/Precompile.cpp:L146] → serializable post-parse IR clone
│        ├─ KGENCompiler::runCheckLITPipeline(parsed_module) [Mojo/lib/Compiler/KGENCompiler.cpp:L1119] → semantic/lifetime checks
│        └─ writePrecompiledFile(built_package, stream) [Mojo/lib/Support/MojoPrecompiledFile.cpp:L128]
│           └─ MLIR bytecode → zstd payload + MPKG V2 header → staging .mojoc
│
├─ compiler succeeds → os.replace(staging, final) → atomic publication
├─ finally → remove staging directory
└─ return final Path → P1; addition.py stays a separate Python program
   The source-directory branch invokes precompile again; it has no exists() skip.
```

### A.2 · Expand registration, validation and Graph.output

```text
ops.custom("add_one", values=[X], out_types=[Y_type], device) [max/python/max/graph/ops/custom.py:L69]
├─ explicit out_types → output dtype/shape/device is already supplied
│  └─ this path bypasses AddOne's optional unimplemented shape function
├─ insert mo.custom with name="add_one" → link to @extensibility.register("add_one") / AddOne [max/examples/custom_ops/kernels/add_one.mojo:L32]
├─ KernelLibrary.verify_custom_op(operation) [max/python/max/graph/graph.py:L307]
│  ├─ lookup reusable native Analysis by frozenset(package_paths)
│  └─ Analysis::verifyCustomOp(operation) [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L69]
│     └─ MojoRegistry → check registered kernel, device, arguments and results
├─ return symbolic custom result → callback returns its .tensor
└─ Graph.output(Y) [max/python/max/graph/graph.py:L1396]
   ├─ flatten result tree; infer output types/names; write graph signature
   ├─ record mo.output(Y, required chains)
   └─ verify G1 → Y describes a future output; it contains no computed numbers
```

## B · InferenceSession(devices=[device]) → S1

session = InferenceSession(devices=[device])

INPUT: existing Device handle(s); num_threads=None; session custom_extensions=None
OUTPUT: S1: Python session → native session → Engine::Context + host CPU runtime
EXAMPLE: CPU → [CPU]; GPU → [Accelerator, CPU]. G1 keeps its own package P1.

```text
InferenceSession.__init__(devices=[device]) [max/python/max/engine/api.py:L664] → S1
├─ normalize devices → deduplicate while preserving first occurrence
├─ append CPU() if absent → host runtime is present for both target routes
├─ optional import/export MEF policy → neither selected in addition.py
├─ session extensions = []; num_threads or 0 → automatic runtime choice
└─ _InferenceSession(final_devices, [], 0) [max/python/max/_core/max_license/modules/engine.cpp:L787]  [nanobind → C++]
   └─ Core::InferenceSession::InferenceSession(...) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541]
      ├─ populateCompileConfig([]) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L615] → session compilation settings
      ├─ retain Device handles; runtimeConfig.addDevice(DeviceSpec, driver context)
      ├─ runtimeConfig → numThreads=0; API language="python"
      └─ Engine::Context::create(runtimeConfig) [max/lib/max_license/API/c/EngineContext.cpp:L169]
         ├─ getOrCreateCPUDeviceOptions(config) [max/lib/max_license/API/c/EngineContext.cpp:L50] → host scheduling options
         ├─ Init::getOrCreateContext("max-engine", options) [Init/lib/Init.cpp:L116]
         │  ├─ matching global context exists → retain it
         │  └─ otherwise → createContextImpl(...) [Init/lib/Init.cpp:L40] → MAX context + CPUDevice
         ├─ Engine::Context(config, maxContext) [max/lib/max_license/API/c/EngineContext.cpp:L214] → retain runtime references
         └─ native session owns this Engine::Context → S1 ready for compile/init
```

### B.1 · Expand ownership, threads and constructor examples

```text
Constructor input → normalized output [max/python/max/engine/api.py:L664]
├─ [CPU()]                  → [CPU()]
├─ [Accelerator()]          → [Accelerator(), CPU()]
└─ [device, device, CPU()]  → [device, CPU()]  (if device is an accelerator)

S1 [Python InferenceSession]
└─ _impl → Core::InferenceSession [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L541] [C++ object in the same host process]
   ├─ devices → retained existing driver handles / contexts
   ├─ sessionCompileConfig → compiler settings / optional extension paths
   └─ runtimeContext → Engine::Context [max/lib/max_license/API/c/EngineContext.cpp:L214] [owned native object]
      ├─ config → runtime settings / device descriptions
      ├─ maxContext → shared MAX ContextRef [Init/lib/Init.cpp:L116]
      │  └─ creation/reuse protected by std::mutex
      └─ cpuDevice → AsyncRT host runtime / worker scheduling

Thread policy → num_threads=None becomes 0: automatic selection [max/lib/max_license/API/c/EngineContext.cpp:L50]
└─ context/runtime can be reused → do not infer a new private pool per S1
Constructor result → configuration + ownership; G1 is supplied later to compile.
```

## C · session.compile(G1) → compiled artifact CM1

compiled = session.compile(graph)

INPUT: S1 + G1 module + package P1 + graph target metadata; expected_weights={}
OUTPUT: CM1: CompiledModel with resolved native artifact, MEF bytes and uninitialized model descriptor(s)
EXAMPLE: name="addition"; one X/Y float32[5,10]; AddOne specializes for CPU or GPU

```text
InferenceSession.compile(G1) [max/python/max/engine/api.py:L1062]  [Python caller]
├─ compile_async(G1) [max/python/max/engine/api.py:L969]
│  ├─ module = G1.module; extensions += G1.kernel_libraries_paths → P1
│  ├─ expected_weights = {} → add-one has no external weight bindings
│  ├─ Analysis::seedKernelDecls(module) [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L75] → add reusable declarations to caller module
│  ├─ _compile_module(module, extensions, tile_based_fusion=False) [max/python/max/engine/api.py:L1276]
│  │  ├─ acquire process-global compilation lock
│  │  ├─ _impl.compile(module._CAPIPtr, extensions, pipeline_name, False) [max/python/max/_core/max_license/modules/engine.cpp:L824]
│  │  │  └─ Core::InferenceSession::compile(capsule, ...) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L654]
│  │  │     ├─ CompilationRequest → source module pointer + graph format + options
│  │  │     └─ Engine::Context::compileModel(config) [max/lib/max_license/API/c/EngineContext.cpp:L225]
│  │  │        ├─ allocate pending AsyncValue<CompiledModels>
│  │  │        └─ addTask(cpuDevice, ...) → compilation on a host runtime worker
│  │  │           └─ compileModelSync(config) [max/lib/max_license/API/c/EngineContext.cpp:L269]
│  │  │              └─ CompiledModels::create(context, compileCfg, runtimeCfg) [max/lib/max_license/API/c/CompiledModels.cpp:L100]
│  │  │                 └─ MOFContext::compileToBinary(source, format) [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
│  │  │                    ├─ clone module → compiler working graph C1
│  │  │                    ├─ graph-artifact cache hit → reuse prepared MEF bytes
│  │  │                    └─ miss/disabled → MO → MAP → MOGG → MGP → kernel JIT → MEF
│  │  └─ done callback → release compilation lock when the task resolves
│  └─ wrap pending handle → CompiledModel(handle, expected_weights={}) [max/python/max/engine/api.py:L414] → CM1
│
├─ handle.wait() → caller waits for completion; exception() → raise on failure
└─ return resolved CM1 → graph names=("addition",); executable state comes from init
   G1 remains with the caller; native lowering transforms the clone C1.
```

### C.1 · Expand MO → MAP → MOGG → MGP and their actual owners

```text
MOFContext::compileToBinary [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377] → owns the compilation route for C1
└─ cache miss → convertMOToMEF(C1, options) [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356]
   └─ MO::runGCPipeline(C1, options) [GraphCompiler/lib/GraphCompiler.cpp:L638]
      ├─ create mlir::PassManager; gcPipelineImpl builds passes; pm.run(C1) executes them
      ├─ buildMoPipeline [GraphCompiler/lib/GraphCompiler.cpp:L219] → MO = Modular Operations
      │  ├─ graph/tensors/shapes/device placement/custom op contracts
      │  └─ seeded kernel declarations → skip redundant package import
      ├─ MOToMAPPass [GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L764] → MAP = Modular Access Primitives
      │  └─ expose accesses and computations for fusion / implementation decisions
      ├─ buildMapPipeline [GraphCompiler/lib/GraphCompiler.cpp:L338] → optimize access patterns / eligible fusion
      ├─ MAPToMOGGPass [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L53] → MOGG = Modular Operation Generator Graph
      │  └─ MojoRegistry + MAPToMOGGState → select implementation / glue calls
      ├─ buildMoggPipeline [GraphCompiler/lib/GraphCompiler.cpp:L370] → refine Mojo-oriented functions and calls
      ├─ MOGGToMGPPass [GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:L2406] → MGP = Modular Glue Primitives
      │  └─ MOGraphLowering → init/execute regions, buffers, state slots and chains
      ├─ buildMgpPipeline [GraphCompiler/lib/GraphCompiler.cpp:L382]
      │  ├─ FinalizeDeviceGraphsPass [GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:L375] → finalize marked device graphs when present
      │  └─ EmitMojo [GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:L147] → emitted source for native routines / package imports
      ├─ JitCompileKernelsPass [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529] → compile/reuse native routines (C.2)
      └─ buildJitPipeline [GraphCompiler/lib/GraphCompiler.cpp:L75]
         ├─ LowerCompiledRoutinesPass [GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:L102] → ModuleLoad / ExecuteGeneric
         └─ inline/clean chains → OutlineModelsToCallables → initialization + execution plan
   Dialect names and roles → GraphCompiler.md [GraphCompiler/docs/GraphCompiler.md:L1]

Add-one ancestry [schematic representation; not a captured compiler dump]
└─ mo.custom("add_one", X) → implementation-oriented accesses/call
   → selected AddOne routine → buffer/state execution plan → compiled callable
```

### C.2 · Expand specialization, LLVM translation, targets and JIT

```text
compileMojoToBinary(C1, emitted Mojo, ...) [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529]
├─ emitted routine + P1 → parser/checks/LIT lowering → separate kernel IR KIR1
├─ compiled-kernel lookup → checkCache(state, ...) [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L213]
│  ├─ hit → handleCacheHits [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L507] → reuse compatible native archives
│  └─ miss → handleCacheMisses [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L435]
│     ├─ KGENCompiler::runElaborationPipeline [Mojo/lib/Compiler/KGENCompiler.cpp:L1140]
│     │  ├─ populateElaborateModulePasses [Mojo/lib/Compiler/KGENCompiler.cpp:L967] → concrete parameter specialization
│     │  ├─ buildPostElaborationPipeline [Mojo/lib/Compiler/Pipeline/Pipeline.cpp:L154] → concrete IR optimization
│     │  └─ offloads encountered → compileOffloads [Mojo/lib/Compiler/KGENCompiler.cpp:L516] → group/slice IR by TargetInfo
│     └─ ObjectCompiler::emitArchive [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L1023] → native archive with callable symbols
│        └─ ObjectCompiler::lowerAllFuncsToLLVM [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L816]
│           ├─ buildLowerToLLVMPipeline [Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp:L26] → KGEN/POP/control flow → LLVM dialect (MLIR)
│           └─ MLIR-to-LLVM translation [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L796] → llvm::Module → LLVM1 (LLVM IR)
│
├─ target options guide compilation → TargetBackendRegistry::getBackend [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L114] uses the triple
│  ├─ host target → HostBackend [Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:L54] → CPU object / host routines
│  ├─ nvptx64-nvidia-cuda, e.g. sm_90a → NVPTXBackend [Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:L1] → PTX → cubin
│  └─ amdgcn-amd-amdhsa, e.g. gfx942 → AMDGPUBackend [Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:L77] → GPU object → ELF image
│     Target examples are source examples, not detected hardware on this host.
├─ LLVM TargetRegistry + TargetMachine [Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:L63] → in-process LLVM code generation [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L277]
└─ native compiled resources + C1 runtime plan → continue MEF preparation (C.3)

JIT = just-in-time compilation → required specialized code is prepared as the program runs.
└─ here: addition.py → session.compile(G1) invokes compiler APIs in the host process
   → AddOne's dtype/layout/target specialize; compiled code can be reused for later inputs.
   P1 precompilation is the earlier subprocess; kernel JIT is a later compilation phase.
```

### C.3 · Expand MEF preparation, compiled descriptors and cache boundaries

```text
convertMOToMEF(C1) [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356]  [after graph passes + native routine JIT]
└─ MO::convertMGPToMEF(lowered_module) [GraphCompiler/lib/GraphCompiler.cpp:L692]
   ├─ clone for serialization → prepare device-specific runtime primitive names
   ├─ translateMLIRToMEF → serialized graph callables / attrs / native resources
   └─ WriteableBuffer → prepared MEF bytes in memory

CompiledModels::create(...) [max/lib/max_license/API/c/CompiledModels.cpp:L100] receives those bytes
└─ CompiledModels::read(buffer, frameworkCtx, sourceModel) [max/lib/max_license/API/c/CompiledModels.cpp:L80]
   └─ loadModelsFromMEF(buffer) [max/lib/max_license/API/c/CompiledModels.cpp:L29]
      ├─ MEFFile::open → validate/open the prepared container
      ├─ loadMGPMEFFile(mef) [GraphCompiler/lib/MEFModel/MEFModel.cpp:L901] → register runtime primitives; read Model descriptors
      └─ retain buffer + MEF handle + model descriptors → native CompiledModels
         → resolve async handle → Python CM1: CompiledModel

Cache routes → MOFContext graph-artifact cache [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377]
├─ artifact hit → skip graph lowering / kernel JIT; read retained MEF bytes/descriptors
├─ kernel hit only → reuse routine code; still lower/prepare the graph's runtime plan
└─ cold path → both compilation phases run

On disk → CM1.export_mef(path) [max/python/max/engine/api.py:L414] explicitly exports the compiled bytes
└─ session.compile(G1) itself returns CM1; it does not require a user-named .mef file.
```

## D · session.init(CM1) → executable model M1

model = session.init(compiled)

INPUT: S1 + resolved CM1 + real device contexts; weights_registry omitted → {}
OUTPUT: M1: initialized ModelHandle retaining executable state, CM1 and native S1
EXAMPLE: init_all → {"addition": M1}; init → M1. The input Buffer is supplied later.

```text
InferenceSession.init(CM1) [max/python/max/engine/api.py:L1126]
├─ init_all(CM1, weights_registry=None) [max/python/max/engine/api.py:L1160]
│  ├─ registry = {}; expected_weights = {} → no weight-binding validation failures
│  ├─ _impl._load_all(compiled_handle, registry) [max/python/max/_core/max_license/modules/engine.cpp:L787] → share artifact handle
│  │  └─ Core::InferenceSession::loadAll(handle, weightsDict) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L289]
│  │     ├─ buildWeightsFromDict({}) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L214] → empty native registry + managed-tensor list
│  │     ├─ await compiled handle (release GIL while waiting) → descriptor "addition"
│  │     └─ for each model descriptor → Core::InferenceSession::load(model, registry) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L275]
│  │        └─ Engine::Context::initModel(model, registry) [max/lib/max_license/API/c/EngineContext.cpp:L288]
│  │           └─ addTask(cpuDevice, ...) → initializer runs on a host runtime worker
│  │              └─ createExecutableModel(model, cpuDevice, config, registry) [max/lib/max_license/API/c/EngineContext.cpp:L79]
│  │                 ├─ DeviceMap from session DeviceSpecs + retained driver contexts
│  │                 ├─ Model::setup(cpuDevice, registry, deviceMap) [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184]
│  │                 │  ├─ mapDeviceRefsToDevices [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24] → synchronize/resolve contexts
│  │                 │  ├─ initArgs = registry pointer + model device contexts
│  │                 │  ├─ initFn.invoke(initArgs) → run compiled initialization callable
│  │                 │  └─ construct ExecutableModel from chain/context/registry/extra args
│  │                 └─ waitForModelSetup() [GraphCompiler/lib/MEFModel/MEFModel.cpp:L387] → initialization results must be ready
│  │        → ModelHandle(..., native_session_ref, executable, registry, index) [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L194]
│  │          → cache input/output devices → M1
│  └─ name-index results → {"addition": M1}
├─ exactly one model? yes → return sole model M1
└─ next source step → numeric input allocation, model.execute(x) (outside init)
```

### D.1 · Expand initialization resources, loading and readiness

```text
Model::setup [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184] → compiled initFn.invoke(registry + devices)
├─ mapDeviceRefsToDevices [GraphCompiler/lib/Driver/Device/DeviceMap.cpp:L24]
│  ├─ synchronize supplied driver contexts before first resource use
│  ├─ CPU DeviceRef → create legacy runtime CPU context
│  └─ GPU DeviceRef → reuse supplied context; fallback create by device id if absent
├─ CUDA/HIP contexts → ensure configured stream count before capture (default: 2)
├─ invoke the compiled init region → runtime/state initialization and native routine loading
│  └─ ModuleLoad planned by LowerCompiledRoutinesPass [GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:L102]
│     └─ mgp_module_load_cpu [GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp:L55] → CompiledKernelsContext in a runtime state slot
│        └─ CompiledKernelsContext::load(archive_bytes, symbols) [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L85]
│           ├─ ordinary route: archive hash → execution engine addIfAbsent<StaticArchiveLayer>
│           └─ lookup named symbols → resident native callable pointers
│              (debug/profiler route can link/load a shared library)
├─ initializer results → [chain, MGP context, weight-registry reference, extra state/device args]
├─ waitForModelSetup [GraphCompiler/lib/MEFModel/MEFModel.cpp:L387] → await every required value; propagate errors
└─ executable M1 → reuse retained state and routines for fresh execution inputs

Boundaries
├─ .mojoc P1: portable post-parse package for later specialization
├─ CM1: compiled bytes + descriptors; still needs runtime setup
└─ M1: ready executable handle; numeric X arrives at execute, not at constructor/init
```

### D.2 · Expand initialization gates and retained ownership

```text
init_all initialization gates [max/python/max/engine/api.py:L1160]
├─ add-one: expected weights {} + supplied registry {} → pass
├─ weighted graph: missing declared name → ValueError before native initialization
├─ supplied weight: device type must match; array must be contiguous
├─ actual device/context/setup failure → native initialization raises
├─ virtual-device mode → CompileOnlyModel stand-in; execute refuses hardware work
└─ init [max/python/max/engine/api.py:L1126] requires exactly one result; multi-model artifacts use init_all

M1 [native ModelHandle exposed as the Python Model]
├─ loadedModel → initialized ExecutableModel and persistent execution state
├─ compiledModels → shared CM1 native artifact / model descriptors / bytes
├─ inferenceSession → retained native S1 reference [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L194]
│  └─ Engine::Context → MAX context / CPU runtime / session device settings
└─ weightsRegistry + managedTensors → retained bindings (empty in add-one)

Repeated initialization → _load_all uses compiled.copy() [max/python/max/_core/max_license/modules/engine.cpp:L787]
└─ same CM1 can initialize multiple handles; each handle retains what it needs.
```

## E · Input and output example

```text
addition.py: execution happens after all four preparation calls [max/examples/custom_ops/addition.py:L1]
├─ x_values = np.random.uniform(size=(5,10)).astype(np.float32) → host array
├─ x = Buffer.from_numpy(x_values).to(device) → B1: input Buffer
├─ result = model.execute(x)[0] → O1: output Buffer
└─ result.to(CPU()).to_numpy() → CPU-readable numbers; expected = x_values + 1

Illustrative values in the same [5,10] shape [not an executed MAX result]
├─ input row 0:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
├─ output row 0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
└─ G1 / S1 / CM1 / M1 were prepared without these numbers; B1/O1 hold per-call data.
```


---

# From Bazel’s build graph to a selected device backend

## 1. Previous answer — unchanged

For ./bazelw build //max/examples/custom_ops:addition, add_one.mojo is runtime source data. Bazel does not compile
  that custom kernel into CPU or GPU machine code in this target. The Python example precompiles the kernel package
  later, when it runs and constructs its Graph.

  ### 1.1 What the addition build produces

  Bazel analysis identifies the addition target as a py_binary. Its resolved files include:

  • bazel-out/k8-dbg/bin/max/examples/custom_ops/addition — the Python launcher.
  • max/examples/custom_ops/addition.py — the Python entry-point source, available through runfiles.
  • bazel-out/k8-dbg/bin/max/examples/custom_ops/addition.mypy_stdout — an auxiliary output; I haven’t inspected its
    contents.

  The macro supplies :kernel_sources as data, so the matching kernels/*.mojo files are available at runtime. They are
  not compiled into kernels.mojoc as part of the :addition target. A separate :kernels target produces that package;
  shared Mojo dependencies such as layout, extensibility, and std have their own .mojoc outputs. See the BUILD file
  (max/examples/custom_ops/BUILD.bazel:9) and example macro (max/examples/custom_ops/custom_op_example.bzl:12).

  These paths came from Bazel analysis. I started the actual build, but interrupted its cold dependency build before
  it finished, so I can’t report a successful build result.

  ### 1.2 Where the Mojo dependencies come from

  They are Bazel targets and generated package files, not Python classes or objects.

   Label in the example macro    Target it aliases to                       Analyzed .mojoc output
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   //max:layout                  //max/kernels/src/layout:layout            bazel-out/k8-dbg/bin/max/kernels/src/
                                                                            layout/layout.mojoc
  ────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────────────
   //max:extensibility           //max/kernels/src/graph_compiler/          bazel-out/k8-dbg/bin/max/kernels/src/
                                 extensibility:extensibility                graph_compiler/extensibility/
                                                                            extensibility.mojoc
  ────────────────────────────  ─────────────────────────────────────────  ──────────────────────────────────────────
   @mojo//:std                   //Mojo/stdlib/std:std                      bazel-out/k8-dbg/bin/Mojo/stdlib/std/
                                                                            std.mojoc

  The //max:* aliases are created by max_aliases() in max/BUILD.bazel (max/BUILD.bazel) using the mappings in
  mojo_aliases.bzl (bazel/mojo_aliases.bzl:14). The source packages are declared in their respective BUILD files:
  layout (max/kernels/src/layout/BUILD.bazel), extensibility (max/kernels/src/graph_compiler/extensibility/
  BUILD.bazel), and stdlib (Mojo/stdlib/std/BUILD.bazel).

  A .mojoc is a precompiled Mojo package: it stores a lit.package operation serialized as MLIR bytecode, including
  declarations and bodies in the package’s post-parse IR. It has not yet been elaborated for concrete parameter
  values or turned into target machine code. The repository’s Mojo compiler walkthrough (Mojo/docs/compiler/
  MojoCompilerWalkthrough.md:599) describes this format; the official Mojo docs also describe .mojoc as a precompiled
  package for importing into other Mojo code. (mojolang.org)

  ### 1.3.1 How BUILD.bazel connects the targets

  The :kernel_sources filegroup is used in two distinct ways:

  This Mermaid diagram exceeds the terminal renderer's size limits.
  flowchart LR
      BUILD["BUILD.bazel"] --> MACRO["custom_op_example_py_binary(name = addition)"]
      MACRO --> PY["py_binary :addition"]
      ADDPY["addition.py · srcs"] --> PY
      SOURCES["filegroup :kernel_sources<br/>glob kernels/*.mojo"] -->|"runtime data / runfiles"| PY
      MOJODEPS["layout · extensibility · std<br/>mojo_deps"] --> ENV["Mojo runtime environment"]
      ENV --> PY

      MACRO --> TESTMACRO["modular_run_binary_test"]
      TESTMACRO --> TEST["binary_test :addition.example-test"]
      TEST -->|"runs this binary when tested"| PY

      SOURCES -->|"srcs"| LIB["mojo_library :kernels"]
      LIB -->|"dependency"| BENCH["mojo_binary :benchmarks"]

  There is no :addition → :kernels dependency edge. The Python target receives the Mojo sources as runtime data.
  Separately, :kernels precompiles them, and :benchmarks depends on that library. The rules and edges are in
  BUILD.bazel (max/examples/custom_ops/BUILD.bazel:9).

  #### 1.3.1.1 What each BUILD construct means

   Construct                      What it is and how this file uses it
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   mojo_binary                    A Bazel rule for a Mojo executable. Here, :benchmarks is a Mojo program that
                                  depends on :kernels. The local wrapper forwards to the upstream Mojo rule.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   mojo_library                   A Bazel rule that precompiles Mojo sources into a .mojoc package. Here, :kernels
                                  uses :kernel_sources as srcs. The local wrapper also adds documentation targets.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   custom_op_example_py_binary    A local Starlark macro, not a Bazel rule. It expands an example declaration into a
                                  Python binary and its automatic example test.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   filegroup                      A Bazel target that groups files. :kernel_sources collects kernels/*.mojo; the
                                  filegroup itself does not compile them.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   load(...)                      A Starlark directive that imports symbols from a .bzl file into the BUILD file’s
                                  scope. It does not create a build target.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   package(...)                   Package-level metadata. Here, default_visibility makes targets private unless they
                                  specify otherwise; it does not create a Python or Mojo package artifact.
  ─────────────────────────────  ────────────────────────────────────────────────────────────────────────────────────
   target_compatible_with         A rule attribute containing platform constraints. It can make a target
                                  incompatible with selected OS or hardware configurations; it does not select a
                                  compiler backend at runtime.

  The request listed mojo_binary twice; it is covered once above.

  #### 1.3.1.2 How they work together here

  • load makes mojo_binary, mojo_library, requirement, and the local Python-example macro available to BUILD.bazel.
  • filegroup(:kernel_sources) collects the .mojo files.
  • custom_op_example_py_binary(name = "addition") creates :addition with addition.py as its source, the kernel
    filegroup as run-time data, Python dependencies, and Mojo dependencies.

  • mojo_library(:kernels) compiles the same source group into a separate .mojoc package.
  • mojo_binary(:benchmarks) consumes :kernels.
  • package(default_visibility = ...) supplies the package’s default visibility. target_compatible_with, when
    provided, constrains the compatibility of the generated targets.

  ### 1.3.2 How the macro expands

  The custom_op_example_py_binary macro in custom_op_example.bzl (max/examples/custom_ops/custom_op_example.bzl:5)
  calls two other macros:

  This Mermaid diagram exceeds the terminal renderer's size limits.
  flowchart TD
      CALL["BUILD call:<br/>custom_op_example_py_binary(name = addition)"]
      CALL -->|"creates"| MPY["modular_py_binary(name = addition)"]
      MPY --> ENV["mojo_test_environment<br/>with layout, extensibility, std"]
      MPY --> PY["py_binary :addition<br/>addition.py + Python deps<br/>kernel sources in data/runfiles"]
      MPY --> HELPERS["Supporting targets<br/>Mojo deps, venv, debug, pydeps"]

      CALL -->|"also creates"| RUNTEST["modular_run_binary_test<br/>name = addition.example-test"]
      RUNTEST --> BT["binary_test :addition.example-test"]
      BT -->|"symlinks executable and merges runfiles"| PY

  modular_py_binary prepares the Mojo environment and then invokes py_binary; it also defines supporting targets such
  as the virtual environment and debug entry point. See modular_py_binary.bzl (bazel/internal/
  modular_py_binary.bzl:53).

  modular_run_binary_test creates a sibling test target named :addition.example-test. It configures the test
  environment and calls binary_test. That rule symlinks the binary’s executable and merges its runfiles. See
  modular_run_binary_test.bzl (bazel/internal/modular_run_binary_test.bzl:6) and binary_test.bzl (bazel/internal/
  binary_test.bzl:3).

  So:

  • ./bazelw build //max/examples/custom_ops:addition builds the Python target and its dependencies; it does not run
    addition.py.

  • ./bazelw run //max/examples/custom_ops:addition runs the Python example.
  • ./bazelw test //max/examples/custom_ops:addition.example-test runs the generated test target.

### Portable build and macro diagrams

```text
BUILD.bazel
  +-- custom_op_example_py_binary(name="addition")
  |     +-- modular_py_binary -> :addition Python launcher / runfiles
  |     |     +-- addition.py + Python dependencies
  |     |     +-- :kernel_sources -> raw kernels/*.mojo runtime data
  |     |     +-- mojo_test_environment -> shared .mojoc packages / tools
  |     +-- modular_run_binary_test -> :addition.example-test
  |           +-- binary_test -> runs :addition with its runfiles
  +-- :kernel_sources -> mojo_library :kernels -> kernels.mojoc
        +-- mojo_binary :benchmarks consumes the separate :kernels library

No :addition -> :kernels dependency edge.
```

### 2. Graph(custom_extensions=[kernels_dir]) packages the Mojo package in kernels/; it does not package
  addition.py

  At runtime, Graph passes the extension path to KernelLibrary.load_paths. If the path is a Mojo source package
  directory, MAX calls _build_mojo_source_package, which invokes mojo precompile and writes a .mojoc file in a
  temporary package cache. addition.py remains a separate Python program. See graph.py (max/python/max/graph/
  graph.py:269) and paths.py (Mojo/python/mojo/paths.py:141).

  For this example, the extension path is kernels/, a Mojo package rooted at kernels/__init__.mojo. Its source
  modules, including add_one.mojo, are part of that package. addition.py is not. The runtime package is cached under
  a path shaped like /tmp/.modular_<uid>/mojo_pkg/mojo_pkg_<hash>.mojoc.


### 2.1 Start the Python application {#toolchain-q2-1}

**2.1.1 Build and run are distinct actions.** The Python launcher resolves `addition.py`, its Python dependencies, the raw kernel source files and the Mojo runtime environment. The application begins when the launcher runs.

~~~bash
# From the modular/ repository root:
./bazelw build //max/examples/custom_ops:addition
./bazelw run //max/examples/custom_ops:addition

# The generated sibling test also runs the example:
./bazelw test //max/examples/custom_ops:addition.example-test
~~~

**2.1.2 The binary's environment supplies the native tools.** `modular_py_binary` creates a `mojo_test_environment`, collects transitive `MojoInfo` packages and expands compiler, import-path, shared-library and linker variables. The configured source-built Mojo tool is `//Mojo/tools/mojo`; the LLD dependency is `@llvm-project//lld:lld`.

| Runtime variable | What the Python/native compiler can locate |
| --- | --- |
| `MODULAR_MOJO_MAX_DRIVER_PATH` | Mojo compiler driver |
| `MODULAR_MOJO_MAX_IMPORT_PATH` | Directories containing the precompiled Mojo dependencies |
| `MODULAR_MOJO_MAX_LLD_PATH` | Configured LLD executable |
| `MODULAR_MOJO_MAX_LINKER_DRIVER` | Host C++ linker driver |
| `MODULAR_MOJO_MAX_SHARED_LIBS` / `...SYSTEM_LIBS` | Native libraries and system-link settings |

These are launch settings derived from the build toolchain. The kernel's LLVM target is selected later from target information. [Python binary rule stages the Mojo runtime environment (bazel/internal/modular_py_binary.bzl:53)](sources/bazel/internal/modular_py_binary.bzl#L53) · [Mojo toolchain and native runtime files collected for execution (bazel/internal/mojo_test_environment.bzl:78)](sources/bazel/internal/mojo_test_environment.bzl#L78) · [Mojo compiler, LLD and C++ linker settings exposed to the process (bazel/internal/mojo_test_environment.bzl:178)](sources/bazel/internal/mojo_test_environment.bzl#L178) · [Source-built Mojo compiler and LLVM LLD toolchain (bazel/internal/BUILD.bazel:185)](sources/bazel/internal/BUILD.bazel#L185)

### 2.2 Prepare the source package at Graph construction {#toolchain-q2-2}

**2.2.1 Call path.** `Graph(custom_extensions=[mojo_kernels])` uses `KernelLibrary.load_paths`. A source-package directory is passed to `mojo.paths._build_mojo_source_package`; an existing `.mojoc` path can be added directly.

**2.2.2 Actual subprocess arguments.** The helper constructs the following argument list; the configured Mojo driver is supplied by `subprocess_run_mojo`:

~~~text
mojo precompile <absolute kernels/ source path> -o <private staging package path>
~~~

The compiler receives the **Mojo source package directory**. `addition.py` remains the Python entry point. The package includes its modules such as `add_one.mojo`; this does not mean every packaged function is emitted as machine code.

**2.2.3 Actual output naming and publication.** The helper hashes the absolute source path and constructs:

~~~text
<tempdir>/.modular_<uid>/mojo_pkg/
    .staging.<pid>/mojo_pkg_<path_hash>.mojoc   # private compiler output
    mojo_pkg_<path_hash>.mojoc                 # published path
~~~

It runs precompilation into the staging directory, then `os.replace` atomically publishes the complete package. The private file uses the final basename because the compiler embeds the module name. This prevents another process from observing a half-written package.

**2.2.4 This hash identifies a source path.** It is not a hash of the kernel's contents, selected GPU architecture or final machine code. The helper shown here invokes precompilation; the presence of that pathname alone is not proof of a skipped compile. [Import source or precompiled Mojo packages (max/python/max/graph/graph.py:269)](sources/max/python/max/graph/graph.py#L269) · [Actual mojo precompile command and atomic package publication (Mojo/python/mojo/paths.py:141)](sources/Mojo/python/mojo/paths.py#L141) · [Custom kernel package root (max/examples/custom_ops/kernels/__init__.mojo:1)](sources/max/examples/custom_ops/kernels/__init__.mojo#L1)

### 2.3 What is inside .mojoc? {#toolchain-q2-3}

**2.3.1 Representation.** A precompiled Mojo package serializes a `lit.package` operation as MLIR bytecode. It retains declarations, implementation bodies and package/dependency information in post-parse IR. The compiler walkthrough describes semantic/lifetime checking during package preparation and later loading of those bodies.

**2.3.2 Why bodies remain.** `AddOne.execute` still has a compile-time `target`; its nested function still has a compile-time `width`. Concrete instantiation needs those bodies later. This package is useful compiler input rather than a final CPU/GPU image.

**2.3.3 Two .mojoc routes exist.** The standalone `:kernels` Bazel library and the runtime source-package helper can precompile the same source group into different output locations. The shared `layout`, `extensibility` and `std` packages are separate dependencies. The dependency edges in point 1 determine which packages Bazel builds.

~~~text
addition.py ── Python program; Graph builder and execute calls
kernels/    ── Mojo source package
  __init__.mojo
  add_one.mojo
  other declared .mojo modules
         │ mojo precompile
         ▼
P1 · .mojoc ── post-parse MLIR bytecode, declarations and bodies
         │ import + concrete specialization later
         ▼
target-specific routines / device images
~~~

[Mojo package bytecode and deferred specialization (Mojo/docs/compiler/MojoCompilerWalkthrough.md:569)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L569) · [Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1) · [Complete custom-op BUILD graph (max/examples/custom_ops/BUILD.bazel:1)](sources/max/examples/custom_ops/BUILD.bazel#L1)

### 2.4 Register implementations and record G1 {#toolchain-q2-4}

**2.4.1 Name linkage.** Python records `ops.custom(name="add_one", ...)`. Mojo declares `@extensibility.register("add_one")` on `AddOne`. The matching string connects the graph operation to its registered implementation.

**2.4.2 Input/output contract.** The example records float32 tensors of shape `[5,10]`, the chosen `DeviceRef` and an explicit output `TensorType`. Its custom shape function raises `NotImplemented`; the explicit output contract makes that optional function unnecessary in this example.

**2.4.3 Recording is symbolic.** G1 stores operations, types, shapes, device placement and def-use relationships. The example creates its random input values later. `Graph.output` completes/verifies the graph output contract.

**2.4.4 Compiler declarations can be seeded once.** The Graph branch of `InferenceSession.compile_async` seeds the MLIR module from the Graph's kernel analysis before native compilation. The compiler recognizes the mapping attribute and can avoid repeating package import. [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Create and immediately verify a custom graph operation (max/python/max/graph/ops/custom.py:69)](sources/max/python/max/graph/ops/custom.py#L69) · [Create MLIR graph structure and optionally trace a Python callable. (max/python/max/graph/graph.py:819)](sources/max/python/max/graph/graph.py#L819) · [Finish the graph's output contract and verify the completed graph. (max/python/max/graph/graph.py:1396)](sources/max/python/max/graph/graph.py#L1396) · [Prepare a Graph, Module, or artifact path for asynchronous native compilation. (max/python/max/engine/api.py:969)](sources/max/python/max/engine/api.py#L969)

### 3. InferenceSession.compile runs MAX’s graph compiler pipeline

  The MAX graph compilation path uses MLIR pass managers in the GraphCompiler library. The source builds the pipeline
  through MO, MAP, MOGG, and MGP stages, then can proceed to JIT compilation. These are compiler-library passes
  invoked by the runtime path; they are not a sequence of standalone shell commands such as mlir-opt. See
  GraphCompiler.cpp (GraphCompiler/lib/GraphCompiler.cpp:199).

  Conceptually:

  Python Graph + custom package
    → MAX C++ / MLIR GraphCompiler
    → MO → MAP → MOGG → MGP
    → device-graph finalization and kernel JIT


### 3.1 Which objects own and run the transformations? {#toolchain-q3-1}

**3.1.1 Entry.** The Python `InferenceSession` passes the Graph's module to the native engine. `MOFContext::compileToBinary` clones the MLIR module and gathers `GCPipelineOptions`. G1 remains the caller's graph; C1 is the compiler-owned clone.

**3.1.2 Execution mechanism.** Native `mlir::PassManager`/`mlir::OpPassManager` objects run passes assembled by `M::MO::buildGCPipeline` and `gcPipelineImpl`. The dialect classes provide operations/types; the conversion pass objects rewrite the module.

**3.1.3 Relation to MAX Serve.** These are MAX/Mojo compiler objects reached through the engine API. A Serve pipeline can call the same engine while initializing a model. Its API server, worker scheduler and batch constructor do not implement the MO/MAP/MOGG/MGP conversions. The four Python examples call the engine directly.

| Layer / object | Responsibility |
| --- | --- |
| Python `Graph`, `Module`, `KernelLibrary` | Symbolic graph and imported kernel declarations |
| Python `InferenceSession` + native engine/context | Compilation request and selected-device runtime context |
| Native `MOFContext` | Clone, compiler configuration, artifact lookup and conversion |
| `GCPipelineOptions`, MLIR pass managers | Configure and execute the ordered compiler transformations |
| `MOToMAPPass`, `MAPToMOGGPass`, `MOGGToMGPPass` | Perform the three named dialect conversions |
| `KGENCompiler`, `ObjectCompiler` | Specialize kernel implementations and emit code |

[Prepare a Graph, Module, or artifact path for asynchronous native compilation. (max/python/max/engine/api.py:969)](sources/max/python/max/engine/api.py#L969) · [Clone the graph module, configure lowering, and consult the native MEF cache. (GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:377)](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)

### 3.2 MO — Modular Operations {#toolchain-q3-2}

**3.2.1 Format and class.** `M::MO::MODialect` is the framework-level graph dialect. Operations carry typed tensor values and attributes. The source dialect definition calls it the frontend input to the graph compiler.

**3.2.2 Work at this level.** `buildMoPipeline` schedules shape/layout inference, symbolic optimization, constant folding when enabled, cleanup, package/kernel verification and device validation. These passes increase the information needed for implementation selection.

**3.2.3 Add-one teaching view.** C1 still describes “apply the registered add_one operation to float32 X[5,10] on the selected device; return Y with the declared type.” It contains tensor contracts and symbolic relationships rather than random X values.

**3.2.4 Artifact distinction.** MO is an in-memory representation. A `.mlir` file can be a diagnostic dump of it. It is not a mandatory file produced by `bazelw build :addition`. [MO dialect definition (GraphCompiler/include/GraphCompiler/MODialect/MODialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MODialect/MODialect.td#L1) · [Prepare and optimize high-level graph operations before MAP lowering. (GraphCompiler/lib/GraphCompiler.cpp:219)](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [Dialect names and current MAP fusion design (GraphCompiler/docs/GraphCompiler.md:1)](sources/GraphCompiler/docs/GraphCompiler.md#L1)

### 3.3 MO → MAP — Modular Access Primitives {#toolchain-q3-3}

#### 3.3.1 The conversion pass {#toolchain-q3-3-1}

`gcPipelineImpl` schedules `MO::createMarkStableTensorsPass()`, then `MO::createMOToMAP()`. `MOToMAPPass::runOnOperation` creates a `MOGG::MojoRegistry`, collects lowerable operations and calls `lowerOp`. Unsupported ranks or failed lowering signal pass failure.

The result exposes implementation-oriented accesses and kernel structure. It lets later passes reason about how computation loads values and stores outputs. [MOToMAPPass walks and lowers graph operations (GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:764)](sources/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp#L764) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)

#### 3.3.2 MAP's operation families and fusion {#toolchain-q3-3-2}

| MAP family | Information made visible |
| --- | --- |
| `map.iter.opaque` | A hand-written Mojo kernel and a store epilogue/fusion interface |
| `map.iter.elementwise` | Per-index loads, elementwise calls and stores |
| `map.load`, `map.store`, `map.call.elementwise` | The data accesses and arithmetic used by an explicit body |

`buildMapPipeline` schedules `MAP::createFuseKernels`, optionally `createFuseMutableLoads`, accumulation-buffer allocation and structured-body outlining. Fusion eligibility depends on implementation interfaces and data relationships.

In the epilogue example, `AddOneElementwise(ElementwiseUnaryOp)` explicitly supplies a fusible `x + 1` body beside matmul and multiplication. The simple `AddOne.execute` supplies its hand-written `foreach` routine. The animations show these structures schematically; they are not captured MAP dumps or proof of a particular fusion result. [MAP dialect and access-operation families (GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPDialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPDialect.td#L1) · [Fuse eligible structured computations and outline kernel bodies. (GraphCompiler/lib/GraphCompiler.cpp:338)](sources/GraphCompiler/lib/GraphCompiler.cpp#L338) · [Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1)

### 3.4 MAP → MOGG — Modular Operation Generator Graph {#toolchain-q3-4}

**3.4.1 Conversion owner.** `MAP::createMAPToMOGG(options)` constructs `MAPToMOGGPass`. Its pass creates `MojoRegistry` and `MAPToMOGGState` objects to lower outlined opaque kernels and shape-function invocations.

**3.4.2 New representation.** MOGG expresses generator/kernel choices, calls, parameter information and implementation interfaces. The graph now has code-generation-oriented routine boundaries while keeping its input/output dependencies.

**3.4.3 Finishing passes.** `buildMoggPipeline` legalizes rowwise kernels, conforms call operands, outlines routines and adds trace wrapping. These are compiler-generated call interfaces around the authored implementation.

**3.4.4 Current pipeline evidence.** The pipeline performs advanced kernel fusion in MAP before this conversion. Some MOGG dialect prose contains older fusion motivation; the ordered pipeline and the current GraphCompiler design document establish the current placement of that work. [MAPToMOGGPass uses MojoRegistry and MAPToMOGGState (GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:53)](sources/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp#L53) · [Legalize kernel calls and outline routines. (GraphCompiler/lib/GraphCompiler.cpp:370)](sources/GraphCompiler/lib/GraphCompiler.cpp#L370) · [MOGG dialect definition (GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGDialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGDialect.td#L1) · [Dialect names and current MAP fusion design (GraphCompiler/docs/GraphCompiler.md:1)](sources/GraphCompiler/docs/GraphCompiler.md#L1)

### 3.5 MOGG → MGP — Modular Glue Primitives {#toolchain-q3-5}

**3.5.1 Conversion owner.** `MGP::createMOGGToMGP(options)` constructs `MOGGToMGPPass`. `runOnOperation` uses `SymbolTableAnalysis`, orders graphs by their call dependencies and invokes `MOGraphLowering(...).run()`. A dependency cycle stops the pass.

The representation becomes runtime glue: initialized context/state, buffer references, calls, allocation and execution dependencies.

#### 3.5.2 MGP objects and memory/dependency planning {#toolchain-q3-5-2}

| MGP representation | Runtime meaning recorded by the compiler |
| --- | --- |
| `mgp.model` / `MGP::ModelOp` | Model initialization and execution invokables |
| `!mgp.context` | `MGP::Context` with reusable `StateSlot` values |
| `!mgp.buffer` | Byte storage represented by an MLRT `BufferRef` |
| `!mgp.tensor`, tensor spec | Shape/dtype metadata associated with buffer storage |
| Chains and call operands | Dependencies and values used to coordinate runtime work |

`buildMgpPipeline` runs buffer reuse, dynamic memory planning, deferred allocations, model-argument validation, chain simplification, slot globalization and primitive lowering. A slot in this plan is a compiler/runtime resource; it is not a CUDA thread, an IPC queue seat or a physical GPU register.

For add-one, the teaching view records X's input reference, an output resource and the routine/dependency needed to produce Y. Numerical X/Y contents arrive during execution. [MOGGToMGPPass and dependency-ordered MOGraphLowering (GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2406)](sources/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp#L2406) · [MGP model, context, buffers and state slots (GraphCompiler/include/GraphCompiler/MGPDialect/Dialect/MGPDialect.td:1)](sources/GraphCompiler/include/GraphCompiler/MGPDialect/Dialect/MGPDialect.td#L1) · [Plan runtime resources, lower primitives, and emit Mojo routines. (GraphCompiler/lib/GraphCompiler.cpp:382)](sources/GraphCompiler/lib/GraphCompiler.cpp#L382)

### 3.6 Device-graph finalization {#toolchain-q3-6}

**3.6.1 Ordered preparation.** The MGP pipeline stages device graphs, partitions collectives, computes/simplifies dependencies, generates build stubs and schedules `FinalizeDeviceGraphsPass`.

**3.6.2 Actual guard.** The finalization pass reads `ModelOp` and returns immediately when `getIsDeviceGraph()` is false. For a device-graph model, it finalizes the `DeviceGraphCollectiveOp` operations in its execute block. Scheduling this pass does not establish that every ordinary example has a nonempty device graph.

**3.6.3 Milestones.** Python `Graph.output` verifies the symbolic output contract; this later pass handles compiler device-graph structure; model initialization loads prepared runtime resources; execution submits work. The animation keeps these milestones separate. [Plan runtime resources, lower primitives, and emit Mojo routines. (GraphCompiler/lib/GraphCompiler.cpp:382)](sources/GraphCompiler/lib/GraphCompiler.cpp#L382) · [Device-graph finalization guard and collective audit (GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:375)](sources/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp#L375)

### 3.7 Emit routines, then enter kernel JIT {#toolchain-q3-7}

**3.7.1 Generated source.** `MOGG::createEmitMojo` imports the required standard/custom packages and emits Mojo routine scaffolding for the lowered graph. The emitted source is a compiler resource consumed by the JIT path. The precompiled custom package supplies the authored bodies that the scaffolding references.

**3.7.2 Separate compiler modules.** C1 retains the MGP runtime plan. The JIT entry creates a separate parsed Mojo/KGEN module, KIR1. Rewriting KIR1 toward LLVM does not turn the MGP buffer/dependency plan into GPU assembly.

**3.7.3 Pipeline tail.** On the normal JIT route, `gcPipelineImpl` adds `createJitCompileKernels`, strips kernel declarations and calls `buildJitPipeline`. Final runtime-call lowering and MEF preparation follow compiled-kernel work.



~~~mermaid
flowchart LR
    G1["Python Graph G1 · retained"] -->|clone| C1["C1 · MO"]
    C1 -->|MOToMAPPass| MAP["C1 · MAP"]
    MAP -->|MAPToMOGGPass| MOGG["C1 · MOGG"]
    MOGG -->|MOGGToMGPPass| MGP["C1 · MGP plan"]
    MGP --> FINAL["Conditional device-graph finalization"]
    FINAL --> KIR["EmitMojo + JIT · separate KIR1"]
    KIR --> CODE["compiled host / device routines"]
    CODE --> CALLS["runtime call lowering + MEF bytes"]
    CALLS --> M1["session.init · executable M1"]
~~~

~~~text
G1 retained
  └─ clone C1: MO → MAP → MOGG → MGP
                                 ├─ buffer / dependency / init / execute plan
                                 └─ EmitMojo → separate KIR1 → target code
                         runtime calls + code resources → CM1 → init → M1
~~~

[EmitMojo imports packages and generates routine source (GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:147)](sources/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp#L147) · [MAX kernel JIT entry: parse, LIT lowering, compiler objects and cache routes (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:529)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494) · [Lower compiled loads/calls to runtime operations (GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:102)](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [Finalize runtime chains and outline callables (GraphCompiler/lib/GraphCompiler.cpp:75)](sources/GraphCompiler/lib/GraphCompiler.cpp#L75)

### 4. Mojo specializes kernels and lowers them through its compiler pipeline

  The Mojo compiler’s KGEN pipeline performs specialization and lowering. The LLVM dialect is still MLIR; later, the
  compiler translates that representation to LLVM IR and uses the target backend to emit the target artifact. For
  MAX’s JIT path, this happens through compiler APIs in the process rather than a separate mojo build command for
  each kernel.

  The mojo precompile subprocess above is a separate package-preparation step: it creates .mojoc from source before
  the graph’s compilation pipeline consumes that package.


### 4.1 What JIT means in this example {#toolchain-q4-1}

**4.1.1 Definition.** JIT means **just-in-time compilation**: the running application prepares native code from compiler representations for the required implementation/target. The timing can be explicit preparation before an inference call.

**4.1.2 Concrete MAX path.** In `addition.py`, the order is `Graph(...)`, `InferenceSession(...)`, `session.compile(graph)`, `session.init(compiled)`, creation of numerical X, then `model.execute(x)`. Kernel JIT is reached during graph compilation. This example does not depend on first-execute lazy compilation.

**4.1.3 Actual entry and objects.** `JitCompileKernelsPass::compileMojoToBinary` retrieves target information, parses generated Mojo, runs LIT-to-KGEN work, creates a `KGENCompiler` and `ObjectCompiler`, and stores them with the MGP/KGEN modules in `CompilationState`.

**4.1.4 Miss/hit behavior.** `handleCacheMisses` calls `compiler->runElaborationPipeline`, then `objectCompiler->emitArchive`. It records a compiled-load resource. `handleCacheHits` loads compatible compiled archives. The surrounding graph/runtime plan still needs preparation. A graph-artifact hit is a separate, broader cache route.

~~~text
Running Python process
  session.compile(G1)
    native GraphCompiler passes on C1
      EmitMojo
        compileMojoToBinary
          parse → LIT-to-KGEN → compiled-kernel lookup
          miss: KGENCompiler.runElaborationPipeline
                ObjectCompiler.emitArchive → compiled resources
          hit:  load compatible compiled resources
      runtime-call lowering → prepared CM1
  session.init(CM1) → M1
  model.execute(Buffer X) → result Buffer Y
~~~

[Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [MAX kernel JIT entry: parse, LIT lowering, compiler objects and cache routes (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:529)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [Elaborate and compile cache-miss routines (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:435)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L435) · [Load compatible compiled-kernel archives (GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:507)](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L507) · [Native graph-artifact cache hit and miss (GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:377)](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377)

### 4.2 Which code performs specialization? {#toolchain-q4-2}

**4.2.1 Compiler pipeline.** `Mojo/lib/Compiler/KGENCompiler.cpp` implements `KGENCompiler::runElaborationPipeline`. It configures an MLIR pass manager, supplies target information and invokes `populateElaborateModulePasses`. That builder calls `buildElaborateModulePipeline` and `buildPostElaborationPipeline`.

The elaboration machinery instantiates generator functions/types with concrete compile-time values. The compiler walkthrough identifies `Mojo/lib/Elaborator/Elaborator.cpp` and its IR evaluators; the KGEN compiler supplies the `compileOffloads` callback used when specialization encounters device offloads.

#### 4.2.2 Add-one specialization example {#toolchain-q4-2-2}

| Authored parameter / operation | Concrete information used during preparation |
| --- | --- |
| `AddOne.execute[target]` | `"cpu"` or `"gpu"` implementation route |
| `InputTensor[dtype=output.dtype, rank=output.rank, ...]` | Output/input dtype and rank; float32, rank 2 in this example |
| `elementwise_add_one[width]` | A width selected by the implementation and target |
| `SIMD[x.dtype, width]` | Concrete scalar/vector type |
| `x.load[width](idx) + 1` | Typed load and native arithmetic |
| `foreach[target=target]` | The chosen target's iteration/dispatch machinery |

The animation's small SIMD width is illustrative. The full graph has 50 elements; a source walkthrough alone does not establish a captured vector width, launch count or instruction sequence.

**4.2.3 IR result.** Parametric `kgen.generator`/`kgen.struct.generator` forms become concrete `kgen.func`/`kgen.struct.instance` forms. Compile-time parameter expressions are evaluated; runtime tensor values remain runtime inputs. [KGENCompiler runs the elaboration pass manager (Mojo/lib/Compiler/KGENCompiler.cpp:1140)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) · [Elaboration and post-elaboration pipeline builders (Mojo/lib/Compiler/KGENCompiler.cpp:967)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L967) · [Mojo compile-time specialization (Mojo/docs/compiler/MojoCompilerWalkthrough.md:428)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L428) · [Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1)

### 4.3 Post-elaboration optimization {#toolchain-q4-3}

**4.3.1 Actual code.** `Mojo/lib/Compiler/Pipeline/Pipeline.cpp` implements `buildPostElaborationPipeline`. It applies floating-point policy, verifies concrete IR, updates materialized compile-time storage, then invokes the first and late optimization pipelines.

**4.3.2 Purpose.** This stage simplifies concrete code, lowers argument/calling conventions and enables optimizations such as scalar replacement, promotion and inlining. It operates on the specialized implementation; it does not invent the tiled matmul algorithm authored in the example or execute the tensor's arithmetic in Python. [Concrete-IR verification, materialization and optimization (Mojo/lib/Compiler/Pipeline/Pipeline.cpp:154)](sources/Mojo/lib/Compiler/Pipeline/Pipeline.cpp#L154) · [Mojo compile-time specialization (Mojo/docs/compiler/MojoCompilerWalkthrough.md:428)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L428)

### 4.4 Which code lowers to LLVM dialect? {#toolchain-q4-4}

**4.4.1 Caller.** `ObjectCompiler::lowerAllFuncsToLLVM` configures a pass manager and calls `buildLowerToLLVMPipeline`.

**4.4.2 Exact builder.** `Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp` schedules:

| Pass / action | Representation change |
| --- | --- |
| `createLowerKGENToLLVM` | KGEN functions/types toward LLVM-compatible forms |
| Runtime-closure lowering and POP legalization | Make closures and primitive operations lowerable |
| `createLowerGlobalPOPToLLVM`, `createLowerPOPToLLVM` | Primitive operations and globals into LLVM dialect |
| `createLowerControlFlow` | Lower remaining control-flow forms |
| Cast reconciliation, suspension-point lowering, cleanup | Finish legal LLVM-compatible MLIR |
| `createTargetSpecificLLVMLowering` | Late target-specific MLIR lowering |
| Debug-info lowering | Translate debug representation for LLVM export |

**4.4.3 Format.** LLVM dialect is still **MLIR**: MLIR operations, types, regions and an MLIR context. For example the conceptual SIMD float32 type can become an LLVM-compatible vector representation. A `.pre-llvm-ir.mlir` diagnostic dump is still MLIR. [MLIR lowering, optional dumps and LLVM-module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:816)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816) · [Exact KGEN / POP / control-flow to LLVM-dialect passes (Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp:26)](sources/Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp#L26) · [KGEN / POP → LLVM dialect (Mojo/docs/compiler/MojoCompilerWalkthrough.md:542)](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L542)

### 4.5 Which code translates to LLVM IR? {#toolchain-q4-5}

**4.5.1 Exact API.** The file-local `translateModuleToLLVMIR` helper in `ObjectCompiler.cpp` invokes `mlir::translateModuleToLLVMIR(module, ctx, moduleName)`. It produces a separate `std::unique_ptr<llvm::Module>` using an LLVM context.

**4.5.2 Boundary between IR frameworks.** KIR1 before translation is an MLIR module containing LLVM-dialect operations. LLVM1 after translation is an `llvm::Module`. Textual LLVM IR can be exported as `.ll`; LLVM bitcode can be exported as `.bc`. These exports are optional views of compiler data.

**4.5.3 Code generation.** LLVM optimization and `runLlcPasses` use the selected `llvm::TargetMachine`. The helper creates/runs LLVM code-generation passes in the process. The C++ callback name `runLlc` does not establish that a standalone `llc` executable is launched for each kernel.



~~~mermaid
flowchart LR
    PKG["P1 · .mojoc / post-parse MLIR"] --> KGEN["KIR1 · parametric KGEN"]
    KGEN -->|KGENCompiler + elaborator| CONCRETE["concrete KGEN / POP"]
    CONCRETE -->|buildLowerToLLVMPipeline| DIALECT["LLVM dialect · MLIR"]
    DIALECT -->|mlir::translateModuleToLLVMIR| LLVM["LLVM1 · llvm::Module"]
    LLVM -->|LLVM target codegen| BYTES["object / PTX bytes"]
~~~

~~~text
post-parse package + generated routines
  → parametric KGEN → elaboration → concrete KGEN / POP
  → LLVM dialect (MLIR module)
  → translation → LLVM IR (llvm::Module)
  → target machine / backend → compiled bytes
~~~

[MLIR → LLVM module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:796)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L796) · [MLIR lowering, optional dumps and LLVM-module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:816)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816) · [In-process LLVM machine-code pass manager (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:277)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L277)

### 5. TargetBackendRegistry is a compiler-side C++ registry

  It is declared in TargetBackend.h (Mojo/include/Mojo/Compiler/Target/TargetBackend.h:338) and implemented in
  TargetBackend.cpp (Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:108). It holds registered
  TargetBackend implementations and looks one up using an LLVM target triple.

  Backends register themselves during static initialization. Depending on the build, these include the host backend
  and available device backends such as NVPTX and AMDGPU; other backends can also be present. It is a registry of
  backend implementation objects in the compiler, not a Python dictionary of devices.


### 5.1 Where is the registry, and what does it contain? {#toolchain-q5-1}

**5.1.1 Declaration.** `Mojo/include/Mojo/Compiler/Target/TargetBackend.h` declares `M::KGEN::TargetBackendRegistry`. Its private storage is `std::vector<std::unique_ptr<TargetBackend>> Backends`: owned C++ implementation objects.

**5.1.2 Initialization.** A `RegisterTargetBackend<BackendT>` static object constructs a backend and calls `TargetBackendRegistry::get().add(...)`. Host, NVPTX and AMDGPU source files contain these registrations. Available implementations depend on which backends the compiler build includes.

**5.1.3 Lookup.** `Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp` iterates the registered implementations and calls `resolve(triple)`. It prefers an explicitly resolved backend over a direct self-match. A missing supported implementation produces an error. This is compiler dispatch over target triples.

| Compiler implementation | Source / registration evidence |
| --- | --- |
| `HostBackend` | `Target/Host/HostBackend.h` and `.cpp` |
| `NVPTXBackend` | `Target/max_license/NVPTX/NVPTXBackend.cpp` |
| `AMDGPUBackend` | `Target/max_license/AMDGPU/AMDGPUBackend.cpp` |
| Other compiled-in backends | Resolved by their registered implementation; this table focuses on the requested CPU/NVIDIA/AMD routes |

[Owned backend vector and static registration helper (Mojo/include/Mojo/Compiler/Target/TargetBackend.h:339)](sources/Mojo/include/Mojo/Compiler/Target/TargetBackend.h#L339) · [TargetBackendRegistry resolves the LLVM triple (Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:114)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp#L114) · [Host backend interface and target resolution (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.h:1)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.h#L1) · [Object emission and linking (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:54)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp#L54) · [NVIDIA backend registration, PTX and CUDA compilation (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:1)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L1) · [AMDGPU target and ELF linking (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:77)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77)

### 5.2 How is the backend selected? {#toolchain-q5-2}

**5.2.1 Target record.** `TargetInfoAttr` describes a triple, architecture, features, ABI, relocation model and data layout. `compileOffloads` groups device work by target, copies those fields to `CompilationOptions` and compiles a module sliced for that target.

**5.2.2 Mojo backend selection.** `ObjectCompiler::create` calls `TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple))`. The backend is resolved before its lowering/emission hooks are used.

**5.2.3 LLVM machine selection.** `defaultCreateTargetMachine` then uses the separate LLVM `TargetRegistry::lookupTarget` and `createTargetMachine(triple, targetCpu, targetFeatures, ...)`. ABI, relocation, code model and optimization settings also feed machine construction. The Mojo backend controls the artifact route; the LLVM target machine supplies the target's instruction/code-generation machinery.

| Illustrative target record | Selected backend / role |
| --- | --- |
| `x86_64-unknown-linux-gnu` + a compatible CPU/features | `HostBackend` for Linux host code |
| `nvptx64-nvidia-cuda` + `sm_90a` | `NVPTXBackend` for an NVIDIA device image |
| `amdgcn-amd-amdhsa` + `gfx942` | `AMDGPUBackend` for an AMD device code object |

The NVIDIA/AMD records come from the repository's cross-codegen test. The animation does not report that those devices are present or that the test was executed. [Per-target triple, architecture and backend lookup (Mojo/lib/Compiler/KGENCompiler.cpp:213)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L213) · [Compile each offload for its selected target triple (Mojo/lib/Compiler/KGENCompiler.cpp:516)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L516) · [Select target backend (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:93)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L93) · [LLVM TargetRegistry and triple / CPU / features / ABI (Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp:63)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp#L63) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1)

### 5.3 NVIDIA: LLVM NVPTX → PTX → cubin {#toolchain-q5-3}

#### 5.3.1 Emit PTX {#toolchain-q5-3-1}

With the `nvptx64-nvidia-cuda` triple and a selected SM architecture, `NVPTXBackend::emitObject` calls `ctx.runLlc(..., createObjectFile=false)`. The LLVM NVPTX backend emits **PTX text** into a byte buffer. PTX is the virtual device instruction representation consumed by NVIDIA's compilation tooling.

#### 5.3.2 Compile PTX for the same SM {#toolchain-q5-3-2}

The backend reads `ctx.tm.getTargetCPU()`, verifies that it starts with `sm_`, creates `Driver::CompilationDevice` with CUDA API and that architecture, then calls `compileFunction(ptx, debugLevel, optimizationLevel)`. The returned image is a **cubin**, containing native device code for that target.

The source explicitly ties cubin preparation to the SM used to emit PTX. Its Linux branch documents the NVPTXCompiler API and a configurable compiler-path override. The source boundary is the `CompilationDevice` call; it does not require every run to launch `nvcc` or a standalone `ptxas` process.

**5.3.3 Concrete source example.** The cross-codegen test requests `default-device-target-triple=nvptx64-nvidia-cuda` and `default-device-target-arch=sm_90a`, then inspects a generated `.ptx` diagnostic file for the LLVM NVPTX marker. This is a compiler-test specification, not a measured inference run.

~~~text
Mojo/lib/Compiler/ObjectCompiler/
  ObjectCompiler.cpp
  Target/TargetBackend.cpp
  Target/max_license/NVPTX/NVPTXBackend.cpp

TargetInfo(nvptx64-nvidia-cuda, sm_90a, features)
  → NVPTXBackend + LLVM TargetMachine
  → PTX bytes → CUDA CompilationDevice.compileFunction
  → cubin bytes → later module loading / kernel launch
~~~

[LLVM PTX → CUDA CompilationDevice → cubin (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:152)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L152) · [NVIDIA backend registration, PTX and CUDA compilation (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:1)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L1) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1)

### 5.4 AMD: LLVM AMDGPU → object → linked ELF / hsaco {#toolchain-q5-4}

**5.4.1 Target and implementation.** `AMDGPUBackend` inherits the host backend's object-emission mechanism and supplies AMD target traits, offload behavior, shared-memory address-space information and device-library/link hooks. A target such as `amdgcn-amd-amdhsa` + `gfx942` selects this route.

**5.4.2 Device-library preparation.** Its `linkRuntimeLibraries` constructs a `ROCDLTargetAttr` and uses `AMDGPUModuleLinker` for required ROCm bitcode. The shown ASAN case requests OCKL/OCML and `/opt/rocm/amdgcn/bitcode/asanrtl.bc`; those ASAN libraries are conditional rather than universal inputs for add-one.

**5.4.3 Machine code and link.** LLVM AMDGPU code generation emits target object bytes. The link step resolves the device object into a loadable ELF code object, commonly called an `hsaco`. The AMD backend adds `--no-undefined` to catch unresolved symbols during the link.

**5.4.4 Concrete source example.** The repository test selects `default-device-target-triple=amdgcn-amd-amdhsa`, `default-device-target-arch=gfx942` and checks an `.amdgcn` assembly diagnostic for its target marker.

~~~text
Mojo/lib/Compiler/ObjectCompiler/
  ObjectCompiler.cpp
  Target/TargetBackend.cpp
  Target/Host/HostBackend.cpp
  Target/max_license/AMDGPU/AMDGPUBackend.cpp

TargetInfo(amdgcn-amd-amdhsa, gfx942, features)
  → AMDGPUBackend + LLVM TargetMachine
  → required device bitcode linked into LLVM module
  → AMDGPU target .o bytes → device ELF link
  → loadable code object / hsaco → later device launch
~~~

[AMDGPU target and ELF linking (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:77)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77) · [Object emission and linking (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:54)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp#L54) · [In-process LLVM machine-code pass manager (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:277)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L277) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1)

### 5.5 CPU route and file-format comparison {#toolchain-q5-5}

`HostBackend::emitObject` calls `ctx.runLlc(..., createObjectFile=true)` and `ctx.linkObject`. Host objects are linked into loadable native code. The JIT archive path also uses its MC-linking flow; the backend's `createArchive` hook itself is marked unwired in this revision.

| Representation | What it contains / when it appears |
| --- | --- |
| `.mojo` | Source code |
| `.mojoc` | Precompiled post-parse Mojo package / MLIR bytecode |
| MO / MAP / MOGG / MGP | In-memory MAX graph dialect stages; optional `.mlir` dumps |
| LLVM dialect | MLIR representation suitable for LLVM translation |
| LLVM IR / `.ll` / `.bc` | LLVM module / optional text or bitcode export |
| CPU `.o` → native linked code | Host instruction bytes and link resources |
| PTX → cubin | NVIDIA virtual instruction text → native device image |
| AMDGPU `.o` → ELF / hsaco | AMD device object → loadable device image |
| MEF / compiled artifact | Runtime plan and code/resources prepared for initialization |

Object emission does not require writing an assembly file first. Text assembly is an optional output; NVIDIA's PTX route explicitly uses a textual intermediate. The file sheets in the animation represent in-memory compiler data unless a dump/export is named. [Object emission and linking (Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp:54)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/Host/HostBackend.cpp#L54) · [LLVM PTX → CUDA CompilationDevice → cubin (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp:152)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L152) · [AMDGPU target and ELF linking (Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp:77)](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77) · [CompiledModel (max/python/max/engine/api.py:414)](sources/max/python/max/engine/api.py#L414)

### 6. The selected graph device determines the kernel’s target

  In addition.py, the example chooses CPU() when no accelerator is available and otherwise uses an Accelerator(). It
  assigns that device to the graph and session. The graph’s device target information then guides kernel compilation
  and backend selection.

  The Python process, MAX runtime, and device driver remain host-side. For a CPU target the host backend emits host
  code; for a GPU target a device backend emits GPU-target code, which the host runtime can submit with its buffers
  to the device. The example’s Mojo kernel has a compile-time target parameter and passes it to foreach, so its
  implementation specializes for the chosen target.


### 6.1 Select graph placement {#toolchain-q6-1}

**6.1.1 Add-one source choice.** The unchanged example uses:

~~~python
device = CPU() if accelerator_count() == 0 else Accelerator()
~~~

It supplies `DeviceRef.from_device(device)` to the custom operation and tensor types, then constructs `InferenceSession(devices=[device])`. The compiler's MO device-validation/configuration work supplies the corresponding target information.

**6.1.2 One selected arithmetic device.** This script applies add-one on its chosen CPU or accelerator. A GPU route still has a CPU application and host launch/control routines. A graph can describe multiple placements; this example's single operation has one explicit placement. [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Prepare and optimize high-level graph operations before MAP lowering. (GraphCompiler/lib/GraphCompiler.cpp:219)](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)

### 6.2 Separate host routines from device offloads {#toolchain-q6-2}

**6.2.1 Authored dispatch.** `AddOne.execute[target]` passes its compile-time target to `foreach`. CPU iteration and GPU dispatch are target-specific implementations of that operation.

**6.2.2 Compiler mechanism.** Mojo elaboration can encounter device offload compilation. `compileOffloads` receives a map keyed by `TargetInfoAttr`, slices the relevant generators into a target-specific module, applies that group's options and compiles the group. The host module retains control/launch code and references to the resulting device image.

**6.2.3 Why the Python program stays on the host.** The compiler only sends selected native routine IR to a device backend. Python constructs the graph and supplies buffers; native host routines interact with device APIs. Device image bytes contain the selected kernels, with the target's ABI, address spaces and entry metadata.



~~~mermaid
flowchart TD
    APP["CPU process · Python + MAX"] --> HOST["host routine IR / HostBackend"]
    HOST --> LAUNCH["host driver launch + Buffer references"]
    HOST -->|offload generator + TargetInfo| DEV["separate device IR"]
    DEV --> NV["nvptx64 / sm → PTX → cubin"]
    DEV --> AMD["amdgcn / gfx → object → ELF"]
    NV --> IMAGE["resident compatible device image"]
    AMD --> IMAGE
    IMAGE --> GPU["physical GPU executes kernels"]
    LAUNCH --> GPU
~~~

~~~text
CPU process: Python → MAX host routines → driver API
                    │
                    ├─ host code / runtime state stays on CPU
                    └─ selected offload IR → GPU backend → device image
                                            │
host launch(kernel handle, device Buffers) ───┴─→ GPU execution
~~~

[Original AddOne and AddOneElementwise definitions (max/examples/custom_ops/kernels/add_one.mojo:1)](sources/max/examples/custom_ops/kernels/add_one.mojo#L1) · [Compile each offload for its selected target triple (Mojo/lib/Compiler/KGENCompiler.cpp:516)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L516) · [Per-target triple, architecture and backend lookup (Mojo/lib/Compiler/KGENCompiler.cpp:213)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L213) · [Driver execution (max/lib/max_license/API/c/ModelExecution.cpp:192)](sources/max/lib/max_license/API/c/ModelExecution.cpp#L192)

### 6.3 Finish the runtime call interface {#toolchain-q6-3}

**6.3.1 Prepared placeholders.** Kernel compilation creates compiled-load resources and compiled routine calls in the lowered module.

**6.3.2 Lowering owner.** `LowerCompiledRoutinesPass` converts these into module loading and runtime execution/call operations such as `ModuleLoad` and `ExecuteGeneric`. `buildJitPipeline` then inlines/cleans up MGP calls, simplifies chains and outlines model callables.

**6.3.3 Resulting contract.** Initialization knows which modules/callables to load. Each execution knows which input references and dependencies feed those callables. This is the graph's runtime interface around compiled routines. [Lower compiled loads/calls to runtime operations (GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:102)](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [Finalize runtime chains and outline callables (GraphCompiler/lib/GraphCompiler.cpp:75)](sources/GraphCompiler/lib/GraphCompiler.cpp#L75)

### 6.4 Prepare the compiled artifact / MEF bytes {#toolchain-q6-4}

**6.4.1 Artifact owner.** The engine API exposes a `CompiledModel` wrapper for prepared compilation. Its runtime plan, code resources and metadata belong to CM1; G1 remains the symbolic graph.

**6.4.2 In-memory preparation.** The example assigns `compiled = session.compile(graph)`. It does not explicitly export a `.mef` disk file. A prepared byte buffer is not yet the initialized executable model.

**6.4.3 Data separation.** There are no model weights to bind in these four tensor-input examples. Their numerical arrays are passed to execution. [CompiledModel (max/python/max/engine/api.py:414)](sources/max/python/max/engine/api.py#L414) · [Serialize MEF (GraphCompiler/lib/GraphCompiler.cpp:692)](sources/GraphCompiler/lib/GraphCompiler.cpp#L692) · [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1)

### 6.5 Initialize M1 {#toolchain-q6-5}

**6.5.1 Actual call.** `model = session.init(compiled)` initializes the prepared artifact with the session's runtime/device context.

**6.5.2 Runtime consequences.** Module-load and callable initialization resolve resident code/resources and executable handles. M1 can then accept input Buffers. CM1's compiled bytes, M1's initialized handles and S1's session context are separate identities in the scene.

**6.5.3 Reuse.** The initialized model can be retained for repeated executions with fresh inputs. The matmul example's helper specifically compiles/initializes a new model per helper invocation while its main function reuses the session; the diagram does not change that source ordering. [Validate weight buffers and initialize models in a compiled artifact. (max/python/max/engine/api.py:1160)](sources/max/python/max/engine/api.py#L1160) · [Lower compiled loads/calls to runtime operations (GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:102)](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [Complete Python example (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1)

### 6.6 Supply buffers and execute {#toolchain-q6-6}

**6.6.1 Input creation.** For add-one, `Buffer.from_numpy(x_values).to(device)` supplies numerical storage to the selected device. Wrapping/copying data and compilation are distinct operations.

**6.6.2 Actual call.** `model.execute(x)` hands Buffer references to native execution. MLRT's model/executor dependency machinery coordinates ready operations and result values. This is native graph-runtime work, independent of Serve's request batching scheduler.

**6.6.3 Host/device action.** A CPU routine operates on CPU storage. A GPU routine's host launch references resident device code and accelerator Buffers. Actual driver queues, scheduling overlap and elapsed time are beyond this browser teaching model. [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Model.execute (max/python/max/engine/api.py:118)](sources/max/python/max/engine/api.py#L118) · [Native input preparation (max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:451)](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L451) · [Driver execution (max/lib/max_license/API/c/ModelExecution.cpp:192)](sources/max/lib/max_license/API/c/ModelExecution.cpp#L192)

### 6.7 Obtain result readiness and host-readable values {#toolchain-q6-7}

**6.7.1 Readiness.** The runtime returns/coordinates result Buffers through dependency/readiness state. The executor tracks operands that are not ready and triggers dependent work as their values become available.

**6.7.2 Add-one read.** The Python source calls `result.to(CPU())`, then `result.to_numpy()`. GPU values must become CPU-readable through that explicit operation. A returned Buffer handle by itself is not a copied NumPy array.

**6.7.3 Verification limits in the originals.** Addition prints result and expected values without a numerical equality assertion. The epilogue and matmul scripts contain numerical checks. The attention script prints output Buffers without an explicit host copy/parity assertion. The browser's small arithmetic checks verify the teaching model rather than execution of MAX on a GPU. [Ready primitive worklist (MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp:559)](sources/MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp#L559) · [Await / extract outputs (max/lib/max_license/API/c/ModelExecution.cpp:100)](sources/max/lib/max_license/API/c/ModelExecution.cpp#L100) · [Add-one Python example used in the QnA (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1) · [Complete Python example (max/examples/custom_ops/addition.py:1)](sources/max/examples/custom_ops/addition.py#L1)

### 7. No standalone mlir-opt command is part of the example path

  The example BUILD rules do not declare mlir-opt as a step for compiling add_one.mojo. The graph compiler uses in-
  process MLIR pipelines, while package preparation invokes mojo precompile. The repository has compiler utilities
  for development and debugging, but they are separate from building or running this custom-op example.


### 7.1 Which tool runs at each boundary? {#toolchain-q7-1}

| Step | Tool / implementation actually responsible | Invocation boundary |
| --- | --- | --- |
| Build dependency graph | `./bazelw` → Bazel and declared rules | Build orchestration |
| Shared Mojo packages | Configured Mojo compiler through `mojo_library` | Bazel build action producing `.mojoc` |
| Runtime source-package preparation | `mojo.paths._build_mojo_source_package` → `mojo precompile` | Explicit subprocess in the Python application's package-preparation path |
| MAX MO/MAP/MOGG/MGP transformations | GraphCompiler C++ library + MLIR pass managers | Native compiler APIs in the application process |
| Kernel elaboration | `KGENCompiler` + elaboration pipeline/evaluators | Native compiler APIs and their work queues |
| KGEN/POP → LLVM dialect | `ObjectCompiler` + `buildLowerToLLVMPipeline` | MLIR passes in the compiler |
| LLVM dialect → LLVM IR | `mlir::translateModuleToLLVMIR` | MLIR export API producing `llvm::Module` |
| Target code generation | LLVM `TargetMachine`, `runLlcPasses`, selected backend | LLVM code-generation libraries |
| NVIDIA PTX preparation | CUDA `CompilationDevice::compileFunction` | Driver/compiler implementation boundary; configurable tooling |
| Host / AMD object linking | Object-compiler/link hooks and configured link tools | Link step; precise invocation follows target/tool configuration |
| Initialization / inference | MAX native runtime and device API | Loaded artifact + ready input/result resources |

`mo-opt`, `kgen-opt` and `mlir-opt` are useful compiler development tools. Their existence does not make them required subprocesses in `:addition`. Likewise, `mojo build` is a standalone program-build interface; MAX kernel JIT invokes the corresponding compiler machinery through native APIs.

[Custom-op binary: source data, Mojo libraries and Python dependencies (max/examples/custom_ops/custom_op_example.bzl:12)](sources/max/examples/custom_ops/custom_op_example.bzl#L12) · [Bazel Mojo library precompiles source into a .mojoc package (bazel/internal/mojo_library.bzl:35)](sources/bazel/internal/mojo_library.bzl#L35) · [Actual mojo precompile command and atomic package publication (Mojo/python/mojo/paths.py:141)](sources/Mojo/python/mojo/paths.py#L141) · [Build the ordered sequence of representation conversions and compiler stages. (GraphCompiler/lib/GraphCompiler.cpp:494)](sources/GraphCompiler/lib/GraphCompiler.cpp#L494) · [KGENCompiler runs the elaboration pass manager (Mojo/lib/Compiler/KGENCompiler.cpp:1140)](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) · [Exact KGEN / POP / control-flow to LLVM-dialect passes (Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp:26)](sources/Mojo/lib/Compiler/ObjectCompiler/KGENToLLVMPipeline.cpp#L26) · [MLIR → LLVM module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:796)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L796) · [In-process LLVM machine-code pass manager (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:277)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L277)

### 7.2 How can each representation be inspected? {#toolchain-q7-2}

**7.2.1 Read the actual definitions first.** Source cards and the offline Source Explorer expose complete unchanged source files, the selected function/pass span, its path and pinned revision.

**7.2.2 Diagnostic files are optional.** Graph-stage dumps can expose MO, MAP, MOGG and MGP views. ObjectCompiler's temporary-output policy can write pre-LLVM-dialect/pre-LLVM-IR `.mlir` views and LLVM/native diagnostics. These are separate from the ordinary example's required build outputs.

**7.2.3 Target examples are source tests.** `GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir` supplies the explicit NVIDIA/AMD triples and architectures shown in point 5. `gc-pipeline-dump.mlir` specifies stage-dump checks. Their `RUN` lines describe diagnostic compiler-test invocations; they have not been executed as part of this HTML update.

**7.2.4 Evidence categories.**

| Evidence shown here | What it establishes |
| --- | --- |
| Pinned repository source | Call ordering, pass construction, target selection and representation ownership |
| Inspected compiler-test source | Expected cross-target/dump behavior specified by the test |
| Deterministic browser state and arithmetic | The animation's transitions, controls and small numerical illustrations |
| Actual MAX / GPU execution | Requires running the real example and recording its outputs on the selected device; no such execution is claimed by this page |

The previous point 1 explicitly records an interrupted cold Bazel build. This update preserves that statement and does not replace it with a successful build claim.

[Source test for graph-stage diagnostic dumps (GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir:1)](sources/GraphCompiler/test/mo-opt/gc-pipeline-dump.mlir#L1) · [Source test: NVIDIA sm_90a and AMD gfx942 cross-codegen (GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir:1)](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1) · [MLIR lowering, optional dumps and LLVM-module translation (Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:816)](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816)
