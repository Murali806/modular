# Fused attention: source-ordered call stacks and mechanisms

SOURCE · revision 610f7111836d9a950d45ba52ec9b3e0ed589fe3f. Schematic examples; no MAX/GPU execution.

```text
caller: with Graph("fused_attention", input_types=[TQ,TK,TV], custom_extensions=[kernels/]) as graph ↗ [PY:L41] ↗ caller [PY:L41]
├─ Evaluate caller arguments before Graph.__init__ ↗ [PY:L43]
│  ├─ dtype=DType.float32; device=CPU() or Accelerator() ↗ [PY:L26] ↗ [PY:L29] ↗ caller [PY:L29]
│  ├─ CPU: N=D=8, BN=BD=4; GPU: N=D=32, BN=16, BD=8 ↗ [PY:L30] Sizes · Actual CPU/GPU contracts ↗ caller [PY:L30]
│  └─ TQ, TK, TV: TensorType(dtype, shape=[N,D], device=DeviceRef.from_device(device)) ↗ [PY:L45] ↗ caller [PY:L45]
│     OUT: three type descriptions; no NumPy matrices or Buffer objects yet ↗ [PY:L44] ↗ caller [PY:L43]
│
├─ Graph.__init__(name="fused_attention", input_types=[TQ,TK,TV], forward=None) ↗ [PY:L820]
│  ├─ @in_default_mlir_context → ensure an active MLIR context on this thread ↗ [PY:L70] A.0 · default MLIR context ↗ caller [PY:L819]
│  ├─ Discover SymbolicDim names; initialize _params and context state ↗ [PY:L840]
│  │  └─ Shapes are static → _params={}; _data_dependent_dims=0; _should_verify_ops=True ↗ [PY:L847] ↗ caller [PY:L840]
│  ├─ Enter _location() and choose the top-level module ↗ [PY:L851] _location() · What is loc?
│  │    it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module. ↗ [PY:L851] ↗ caller [PY:L851]
│  │  ├─ Reuse module.mlir_module if supplied; otherwise builtin.ModuleOp(location=loc) ↗ [PY:L856] MLIR Module Reuse or create ↗ caller [PY:L856]
│  │  │    **MLIR.module is IR container** ↗ [C++:L62] ↗ caller [PY:L856]
│  │  └─ Place OpBuilder at self._module.body.end ↗ [PY:L858] OpBuilder · Insertion point ↗ caller [PY:L858]
│  ├─ Insert mo.GraphOp(builder, loc, name, input_types, result_types=[]) ↗ [PY:L860]
│  │  ├─ builder comes from module.body.end; loc from _location(); name from caller L42 ↗ [PY:L861] ↗ [PY:L42] ↗ caller [PY:L860]
│  │  ├─ Convert three Python input types with t.to_mlir() ↗ [PY:L864] to_mlir() · Q/K/V type fields ↗ caller [PY:L864]
│  │  └─ Native builder creates a signature, one region, ONE block, THREE block arguments ↗ [C++:L831] ↗ [C++:L846] GraphOp · Signature and body ↗ caller [PY:L860]
│  │     OUT: %q:TQ, %k:TK, %v:TV in the same entry block; results initially empty ↗ [C++:L849] ↗ caller [C++:L849]
│  ├─ Declare runtime input parameters for symbolic dimensions ↗ [PY:L870] Shape names · Graph parameters
│  │  └─ _params={} → empty ParamDeclArrayAttr; BN/BD will be custom-op parameters later ↗ [PY:L871] ↗ caller [PY:L870]
│  ├─ Retain operation and body handles: _CAPICreate(op._CAPIPtr), regions[0].blocks[0] ↗ [PY:L874] Graph handles · CAPI and body ↗ caller [PY:L874]
│  ├─ Attach module device-info mapping ↗ [PY:L877] Device metadata · Targets ↗ caller [PY:L877]
│  ├─ Initialize _weights={}, _has_chain_input=False, lazy device_chains ↗ [PY:L879] Graph bookkeeping · Weights and chains ↗ caller [PY:L881]
│  ├─ Insert ChainCreateOp(inputs=[]), retain always-ready CH0, seed CPU ordering ↗ [PY:L885] Chain · Ordering token ↗ caller [PY:L886]
│  ├─ Inspect last argument %v: tensor, not a trailing chain → retain CH0 ↗ [PY:L896] Last input · Tensor or chain? ↗ caller [PY:L900]
│  ├─ Set up the custom-kernel library ↗ [PY:L906]
│  │    self._kernel_library = kernel_library or KernelLibrary() ↗ [PY:L909] ↗ caller [PY:L909]
│  │    self._import_kernels(_resolved_custom_extensions(custom_extensions)) ↗ [PY:L910] ↗ caller [PY:L910]
│  │    Resolving defaults also makes backend kernel overlays reachable when explicit paths are absent ↗ [PY:L363] ↗ caller [PY:L910]
│  │  └─ _import_kernels(...): prepare package P1 ↗ [PY:L1703] ↗ caller [PY:L910]
│  │     ├─ KernelLibrary.load_paths([kernels/]) ↗ [PY:L269]
│  │     │  └─ _build_mojo_source_package(kernels/) ↗ [PY:L141]
│  │     │     ├─ kernels/__init__.mojo + referenced child modules, including fused_attention.mojo ↗ [DOC:L1] ↗ caller [PY:L289]
│  │     │     ├─ Separate process: Mojo executable + precompile kernels/ -o <private staging path> ↗ [PY:L169] P1 · Executable, parsing and bytecode
│  │     │     │  └─ parse → resolve → semantic/lifetime checks → MLIR bytecode ↗ [C++:L209] ↗ [C++:L1119] ↗ caller [PY:L178]
│  │     │     └─ os.replace(staging_path, package_path) → atomically published P1 .mojoc ↗ [PY:L179] ↗ caller [PY:L179]
│  │     ├─ Existing .mojoc path → add_path directly ↗ [PY:L286] ↗ caller [PY:L1704]
│  │     └─ Store resolved library paths on the graph op ↗ [PY:L1706] ↗ caller [PY:L1704]
│  └─ forward=None → no lambda is executed in the constructor ↗ [PY:L918] ↗ caller [PY:L41]
│     OUT: graph shell G1 and P1 available; the with-body below records its computation ↗ [PY:L55] ↗ caller [PY:L41]
│
└─ Python enters the with-body after the constructor returns ↗ [PY:L55]
   ├─ Graph.__enter__ → _enter → set Graph.current=G1 ↗ [PY:L1134] with Graph · Active recording scope ↗ caller [PY:L55]
   ├─ q, k, v, *_ = graph.inputs ↗ [PY:L56] Inputs · Three symbolic wrappers
   │  └─ Wrap body.arguments → q→%arg0, k→%arg1, v→%arg2; no numeric data ↗ [PY:L943] ↗ caller [PY:L56]
   ├─ Declare one output TensorType(dtype, [N,D], device) ↗ [PY:L63] Output type · Future O ↗ caller [PY:L62]
   ├─ results = ops.custom(name="modular_ops::fused_attention_custom", ...) ↗ [PY:L57] mo.custom · Detailed three-input call
   │  ├─ Resolve Graph.current; name → StringAttr; normalize DeviceRef ↗ [PY:L94] Attributes · Name, device, BN/BD ↗ caller [PY:L57]
   │  ├─ values=[q,k,v] → existing %arg0,%arg1,%arg2; out_types=[TO] → native result type ↗ [PY:L105] ↗ caller [PY:L61]
   │  ├─ Insert mo.custom in G1.body; wrap its one symbolic result %o ↗ [PY:L109] ↗ caller [PY:L109]
   │  ├─ Attach parameters AFTER insertion: BD/BN → index attributes → dictionary ↗ [PY:L114] ↗ [PY:L49] ↗ caller [PY:L60]
   │  ├─ Attach DeviceRefAttr via MLIR Attribute capsule ↗ [PY:L122] ↗ caller [PY:L122]
   │  ├─ Verify call against loaded P1; resolve FusedAttention.execute and map types/parameters ↗ [PY:L127] Kernel validation · P1 and MO ↗ caller [PY:L127]
   │  └─ Return results=[TensorValue(%o: float32[N,D], selected device)] ↗ [PY:L129] Result wrappers · One symbolic output ↗ caller [PY:L57]
   ├─ graph.output(*results) → mo.output, result signature, graph verification ↗ [PY:L68] Graph.output · Complete G1 ↗ caller [PY:L68]
   └─ Graph.__exit__ restores prior Graph.current ↗ [PY:L1138] ↗ caller [PY:L55]
      OUT: verified G1: (TQ,TK,TV) → (TO); compilation and execution are still later ↗ [PY:L68] ↗ caller [PY:L68]
```

Source links: [max/examples/custom_ops/fused_attention.py:L41](sources/max/examples/custom_ops/fused_attention.py#L41) · [max/examples/custom_ops/fused_attention.py:L43](sources/max/examples/custom_ops/fused_attention.py#L43) · [max/examples/custom_ops/fused_attention.py:L26](sources/max/examples/custom_ops/fused_attention.py#L26) · [max/examples/custom_ops/fused_attention.py:L29](sources/max/examples/custom_ops/fused_attention.py#L29) · [max/examples/custom_ops/fused_attention.py:L30](sources/max/examples/custom_ops/fused_attention.py#L30) · [max/examples/custom_ops/fused_attention.py:L30](sources/max/examples/custom_ops/fused_attention.py#L30) · [max/examples/custom_ops/fused_attention.py:L45](sources/max/examples/custom_ops/fused_attention.py#L45) · [max/examples/custom_ops/fused_attention.py:L44](sources/max/examples/custom_ops/fused_attention.py#L44) · [max/python/max/graph/graph.py:L820](sources/max/python/max/graph/graph.py#L820) · [max/python/max/_mlir_context.py:L70](sources/max/python/max/_mlir_context.py#L70) · [max/python/max/graph/graph.py:L819](sources/max/python/max/graph/graph.py#L819) · [max/python/max/graph/graph.py:L840](sources/max/python/max/graph/graph.py#L840) · [max/python/max/graph/graph.py:L847](sources/max/python/max/graph/graph.py#L847) · [max/python/max/graph/graph.py:L851](sources/max/python/max/graph/graph.py#L851) · [max/python/max/graph/graph.py:L856](sources/max/python/max/graph/graph.py#L856) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L62](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62) · [max/python/max/graph/graph.py:L858](sources/max/python/max/graph/graph.py#L858) · [max/python/max/graph/graph.py:L860](sources/max/python/max/graph/graph.py#L860) · [max/python/max/graph/graph.py:L861](sources/max/python/max/graph/graph.py#L861) · [max/examples/custom_ops/fused_attention.py:L42](sources/max/examples/custom_ops/fused_attention.py#L42) · [max/python/max/graph/graph.py:L864](sources/max/python/max/graph/graph.py#L864) · [GraphCompiler/lib/MODialect/MOOps.cpp:L831](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L831) · [GraphCompiler/lib/MODialect/MOOps.cpp:L846](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L846) · [GraphCompiler/lib/MODialect/MOOps.cpp:L849](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) · [max/python/max/graph/graph.py:L870](sources/max/python/max/graph/graph.py#L870) · [max/python/max/graph/graph.py:L871](sources/max/python/max/graph/graph.py#L871) · [max/python/max/graph/graph.py:L874](sources/max/python/max/graph/graph.py#L874) · [max/python/max/graph/graph.py:L877](sources/max/python/max/graph/graph.py#L877) · [max/python/max/graph/graph.py:L879](sources/max/python/max/graph/graph.py#L879) · [max/python/max/graph/graph.py:L881](sources/max/python/max/graph/graph.py#L881) · [max/python/max/graph/graph.py:L885](sources/max/python/max/graph/graph.py#L885) · [max/python/max/graph/graph.py:L886](sources/max/python/max/graph/graph.py#L886) · [max/python/max/graph/graph.py:L896](sources/max/python/max/graph/graph.py#L896) · [max/python/max/graph/graph.py:L900](sources/max/python/max/graph/graph.py#L900) · [max/python/max/graph/graph.py:L906](sources/max/python/max/graph/graph.py#L906) · [max/python/max/graph/graph.py:L909](sources/max/python/max/graph/graph.py#L909) · [max/python/max/graph/graph.py:L910](sources/max/python/max/graph/graph.py#L910) · [max/python/max/graph/graph.py:L363](sources/max/python/max/graph/graph.py#L363) · [max/python/max/graph/graph.py:L1703](sources/max/python/max/graph/graph.py#L1703) · [max/python/max/graph/graph.py:L269](sources/max/python/max/graph/graph.py#L269) · [Mojo/python/mojo/paths.py:L141](sources/Mojo/python/mojo/paths.py#L141) · [max/examples/custom_ops/kernels/__init__.mojo:L1](sources/max/examples/custom_ops/kernels/__init__.mojo#L1) · [max/python/max/graph/graph.py:L289](sources/max/python/max/graph/graph.py#L289) · [Mojo/python/mojo/paths.py:L169](sources/Mojo/python/mojo/paths.py#L169) · [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209](sources/Mojo/tools/mojo/Precompile/mojo-precompile.cpp#L209) · [Mojo/lib/Compiler/KGENCompiler.cpp:L1119](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1119) · [Mojo/python/mojo/paths.py:L178](sources/Mojo/python/mojo/paths.py#L178) · [Mojo/python/mojo/paths.py:L179](sources/Mojo/python/mojo/paths.py#L179) · [max/python/max/graph/graph.py:L286](sources/max/python/max/graph/graph.py#L286) · [max/python/max/graph/graph.py:L1704](sources/max/python/max/graph/graph.py#L1704) · [max/python/max/graph/graph.py:L1706](sources/max/python/max/graph/graph.py#L1706) · [max/python/max/graph/graph.py:L918](sources/max/python/max/graph/graph.py#L918) · [max/examples/custom_ops/fused_attention.py:L55](sources/max/examples/custom_ops/fused_attention.py#L55) · [max/python/max/graph/graph.py:L1134](sources/max/python/max/graph/graph.py#L1134) · [max/examples/custom_ops/fused_attention.py:L56](sources/max/examples/custom_ops/fused_attention.py#L56) · [max/python/max/graph/graph.py:L943](sources/max/python/max/graph/graph.py#L943) · [max/examples/custom_ops/fused_attention.py:L63](sources/max/examples/custom_ops/fused_attention.py#L63) · [max/examples/custom_ops/fused_attention.py:L62](sources/max/examples/custom_ops/fused_attention.py#L62) · [max/examples/custom_ops/fused_attention.py:L57](sources/max/examples/custom_ops/fused_attention.py#L57) · [max/python/max/graph/ops/custom.py:L94](sources/max/python/max/graph/ops/custom.py#L94) · [max/python/max/graph/ops/custom.py:L105](sources/max/python/max/graph/ops/custom.py#L105) · [max/examples/custom_ops/fused_attention.py:L61](sources/max/examples/custom_ops/fused_attention.py#L61) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/python/max/graph/ops/custom.py:L114](sources/max/python/max/graph/ops/custom.py#L114) · [max/python/max/graph/ops/custom.py:L49](sources/max/python/max/graph/ops/custom.py#L49) · [max/examples/custom_ops/fused_attention.py:L60](sources/max/examples/custom_ops/fused_attention.py#L60) · [max/python/max/graph/ops/custom.py:L122](sources/max/python/max/graph/ops/custom.py#L122) · [max/python/max/graph/ops/custom.py:L127](sources/max/python/max/graph/ops/custom.py#L127) · [max/python/max/graph/ops/custom.py:L129](sources/max/python/max/graph/ops/custom.py#L129) · [max/examples/custom_ops/fused_attention.py:L68](sources/max/examples/custom_ops/fused_attention.py#L68) · [max/python/max/graph/graph.py:L1138](sources/max/python/max/graph/graph.py#L1138)

## InferenceSession constructor · configuration and context → S1

```text
caller: session = InferenceSession(devices=[device]) ↗ [PY:L71]
└─ InferenceSession.__init__(devices=[device], num_threads=None, custom_extensions=None) ↗ [PY:L664]
   ├─ Optional MEF import/export options: neither supplied in this example ↗ [PY:L673] ↗ caller [PY:L71]
   ├─ Build final_devices and seen_devices in input order ↗ [PY:L689]
   │  ├─ Keep the first occurrence of each device; skip duplicates ↗ [PY:L691] ↗ caller [PY:L689]
   │  └─ Create host_cpu = CPU(); append it only if not already present ↗ [PY:L695] ↗ caller [PY:L695]
   │     Example: [CPU()] → [CPU()]; [Accelerator()] → [Accelerator(), CPU()] ↗ [PY:L695] ↗ caller [PY:L695]
   │
   ├─ custom_extensions_final = []; process session extensions only if supplied ↗ [PY:L700] ↗ caller [PY:L700]
   │  This call supplies none; G1's P1 package paths are provided later to compile ↗ [PY:L71] ↗ caller [PY:L700]
   ├─ self._impl = _InferenceSession(final_devices, [], num_threads or 0) ↗ [PY:L707]
   │  └─ nanobind constructor → Core::InferenceSession ↗ [C++:L787]
   │     ├─ populateCompileConfig(customExtensions) → sessionCompileConfig ↗ [C++:L546]
   │     │  ├─ clone existing config; append any supplied extension paths ↗ [C++:L618] ↗ caller [C++:L546]
   │     │  ├─ Original call: customExtensions=[] → modelSource=unset, pipelineName=None, passthroughConfigs=[] ↗ [C++:L162] ↗ caller [C++:L546]
   │     │  └─ Extension illustration: existing /tmp/addon.mojoc → KernelLibraryDriverConfig(path) entry ↗ [C++:L607] ↗ caller [C++:L621]
   │     │     The path must name a valid existing compiled Mojo package; fused_attention.py supplies P1 later via G1 ↗ [C++:L602] ↗ caller [PY:L1022]
   │     ├─ runtimeConfig.setNumThreads(0) → automatic thread policy ↗ [C++:L549] ↗ caller [C++:L549]
   │     ├─ retain driver Device handles and register DeviceSpec + driver context ↗ [C++:L551] ↗ caller [C++:L553]
   │     ├─ runtimeConfig.setAPILanguage("python") ↗ [C++:L557] ↗ caller [C++:L557]
   │     ├─ Engine::Context::create(runtimeConfig) ↗ [C++:L559]
   │     │  ├─ use configured MAX context if present ↗ [C++:L175] ↗ caller [C++:L175]
   │     │  ├─ otherwise getOrCreateCPUDeviceOptions(config, profileFilename) ↗ [C++:L180]
   │     │  │  └─ derive host AsyncRT options; num_threads=0 does not prescribe a private pool ↗ [C++:L50] ↗ caller [C++:L180]
   │     │  ├─ Init::getOrCreateContext("max-engine", options) ↗ [C++:L184]
   │     │  │  └─ reuse or create MAX context under a mutex ↗ [C++:L116] ↗ caller [C++:L184]
   │     │  ├─ Engine::Context(config, maxContext.copy()) → retain runtime configuration/context ↗ [C++:L189] ↗ caller [C++:L189]
   │     │  └─ validate registered device labels; return context or error ↗ [C++:L200] ↗ caller [C++:L200]
   │     ├─ retain runtimeContext; propagate context-creation errors ↗ [C++:L560] ↗ caller [C++:L559]
   │     ├─ enable Python stack callback; conditionally start configured profiler ↗ [C++:L569] ↗ caller [C++:L569]
   │     └─ Example runtimeContext=E1; E1.maxContext=ContextRef(CX1), E1.cpuDevice=HRT1 ↗ [C++:L217] ↗ caller [C++:L189]
   │        ├─ CX1 is reference-counted M::Context, not the Graph MLIR Context ↗ [C++:L30] ↗ caller [C++:L184]
   │        ├─ CX1 runtime reference → HRT1; typed services → Config, Init::Options, telemetry ↗ [C++:L69] ↗ caller [C++:L80]
   │        ├─ No existing global context → create CX1; matching Init::Options → reuse same CX1 ↗ [C++:L120] ↗ caller [C++:L184]
   │        └─ Example second session owns E2, but E2.maxContext can reference the same CX1/HRT1 ↗ [C++:L217] ↗ caller [C++:L126]
   │
   ├─ Register async-safe faulthandler for SIGUSR2; tolerate unavailable/already registered handler ↗ [PY:L715] ↗ caller [PY:L715]
   ├─ Apply configured Mojo log/assert levels and optional kernel/quantization flags ↗ [PY:L727] ↗ caller [PY:L727]
   ├─ If uninitialized-read checking is configured: validate mode, set allocator poison and compiler define ↗ [PY:L757] ↗ caller [PY:L757]
   └─ Return S1: session owns configuration and native runtime references ↗ [PY:L707] Session card · devices, context and ownership ↗ caller [PY:L71]
      S1.num_threads=None; S1._impl=N1 (native session) ↗ [PY:L686] ↗ caller [PY:L707]
      N1.devices=[CPU0]; N1.sessionCompileConfig=CFG1(empty source/paths at construction) ↗ [C++:L95] ↗ caller [C++:L546]
      N1.runtimeConfig: numThreads=0, apiLanguage=python, deviceSpec/driverContext pairs=[CPU0] ↗ [C++:L549] ↗ caller [C++:L553]
      N1.runtimeContext → E1 → maxContext=ContextRef(CX1), cpuDevice=HRT1, config=runtimeConfig copy ↗ [C++:L217] ↗ caller [C++:L563]
      GPU illustration: N1.devices=[GPU0,CPU0]; same host-side ownership structure ↗ [PY:L695] ↗ caller [C++:L551]
      S1 is ready for compile/init; no CM1 or M1 has been produced yet ↗ [PY:L74] ↗ caller [PY:L71]
```

Source links: [max/examples/custom_ops/fused_attention.py:L71](sources/max/examples/custom_ops/fused_attention.py#L71) · [max/python/max/engine/api.py:L664](sources/max/python/max/engine/api.py#L664) · [max/python/max/engine/api.py:L673](sources/max/python/max/engine/api.py#L673) · [max/python/max/engine/api.py:L689](sources/max/python/max/engine/api.py#L689) · [max/python/max/engine/api.py:L691](sources/max/python/max/engine/api.py#L691) · [max/python/max/engine/api.py:L695](sources/max/python/max/engine/api.py#L695) · [max/python/max/engine/api.py:L700](sources/max/python/max/engine/api.py#L700) · [max/python/max/engine/api.py:L707](sources/max/python/max/engine/api.py#L707) · [max/python/max/_core/max_license/modules/engine.cpp:L787](sources/max/python/max/_core/max_license/modules/engine.cpp#L787) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L546](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L546) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L618](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L618) · [max/include/max/max_license/API/c/CompiledModels.h:L162](sources/max/include/max/max_license/API/c/CompiledModels.h#L162) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L607](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L607) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L621](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L621) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L602](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L602) · [max/python/max/engine/api.py:L1022](sources/max/python/max/engine/api.py#L1022) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L549](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L549) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L551](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L551) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L553](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L553) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L557](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L557) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L559](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L559) · [max/lib/max_license/API/c/EngineContext.cpp:L175](sources/max/lib/max_license/API/c/EngineContext.cpp#L175) · [max/lib/max_license/API/c/EngineContext.cpp:L180](sources/max/lib/max_license/API/c/EngineContext.cpp#L180) · [max/lib/max_license/API/c/EngineContext.cpp:L50](sources/max/lib/max_license/API/c/EngineContext.cpp#L50) · [max/lib/max_license/API/c/EngineContext.cpp:L184](sources/max/lib/max_license/API/c/EngineContext.cpp#L184) · [Init/lib/Init.cpp:L116](sources/Init/lib/Init.cpp#L116) · [max/lib/max_license/API/c/EngineContext.cpp:L189](sources/max/lib/max_license/API/c/EngineContext.cpp#L189) · [max/lib/max_license/API/c/EngineContext.cpp:L200](sources/max/lib/max_license/API/c/EngineContext.cpp#L200) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L560](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L560) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L569](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L569) · [max/lib/max_license/API/c/EngineContext.cpp:L217](sources/max/lib/max_license/API/c/EngineContext.cpp#L217) · [Support/include/Support/Context.h:L30](sources/Support/include/Support/Context.h#L30) · [Init/lib/Init.cpp:L69](sources/Init/lib/Init.cpp#L69) · [Init/lib/Init.cpp:L80](sources/Init/lib/Init.cpp#L80) · [Init/lib/Init.cpp:L120](sources/Init/lib/Init.cpp#L120) · [Init/lib/Init.cpp:L126](sources/Init/lib/Init.cpp#L126) · [max/python/max/engine/api.py:L715](sources/max/python/max/engine/api.py#L715) · [max/python/max/engine/api.py:L727](sources/max/python/max/engine/api.py#L727) · [max/python/max/engine/api.py:L757](sources/max/python/max/engine/api.py#L757) · [max/python/max/engine/api.py:L686](sources/max/python/max/engine/api.py#L686) · [max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h:L95](sources/max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h#L95) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L563](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L563) · [max/examples/custom_ops/fused_attention.py:L74](sources/max/examples/custom_ops/fused_attention.py#L74)

## session.compile(graph) · S1 + G1 → CM1

```text
caller: compiled = session.compile(graph) ↗ [PY:L74]
└─ InferenceSession.compile(model=G1, custom_extensions=None, tile_based_fusion=False) ↗ [PY:L1062]
   ├─ Enter compile phase recording; call self.compile_async(...) ↗ [PY:L1103]
   │  └─ InferenceSession.compile_async(G1) ↗ [PY:L969]
   │     ├─ Initialize optional extension list, module=None, expected_weights=None ↗ [PY:L1007] ↗ caller [PY:L1007]
   │     ├─ Select Graph branch; module = G1.module ↗ [PY:L1020] ↗ caller [PY:L1020]
   │     ├─ Append processed G1.kernel_libraries_paths → includes prepared P1 ↗ [PY:L1022] ↗ caller [PY:L1022]
   │     ├─ Capture expected weight devices from G1._weights; fused attention produces {} (Q/K/V are call inputs) ↗ [PY:L1028] ↗ caller [PY:L1028]
   │     ├─ Seed kernel declarations and opaque-type mapping from G1's KernelLibrary into the module ↗ [PY:L1042]
   │     │  └─ native Analysis::seedKernelDecls → reuse previously loaded package analysis ↗ [C++:L75] ↗ caller [PY:L1042]
   │     ├─ self._compile_module(module, extensions, tile_based_fusion=False) ↗ [PY:L1044]
   │     │  ├─ Acquire process-global _COMPILATION_LOCK ↗ [PY:L1293] ↗ caller [PY:L1044]
   │     │  ├─ self._impl.compile(module.mlir_module._CAPIPtr, extensions, pipeline_name, False) ↗ [PY:L1295]
   │     │  │  └─ nanobind capsule overload → native Core::InferenceSession::compile ↗ [C++:L824]
   │     │  │     ├─ clone session compile config; append extensions and record pipeline/tile settings ↗ [C++:L658] ↗ caller [C++:L658]
   │     │  │     ├─ ModelSource{source=model.data(), fileFormat=MAX graph format} ↗ [C++:L670] ↗ caller [C++:L670]
   │     │  │     └─ compileModel(runtimeContext, config) → pending AsyncValue ↗ [C++:L676]
   │     │  │        └─ Engine::Context::compileModel → allocate result; addTask on host CPU runtime ↗ [C++:L227]
   │     │  │           └─ Worker task: compileModelSync(config) ↗ [C++:L232]
   │     │  │              ├─ Use session runtime device specifications ↗ [C++:L271] ↗ caller [C++:L271]
   │     │  │              └─ CompiledModels::create(maxContext, config, runtimeConfig) ↗ [C++:L273]
   │     │  │                 ├─ setupFrameworkContext → graph compiler frontend context ↗ [C++:L106] ↗ caller [C++:L106]
   │     │  │                 ├─ frameworkCtx->compileToBinary(opaque module pointer, format) ↗ [C++:L111]
   │     │  │                 │  └─ MOFContext::compileToBinary ↗ [C++:L377]
   │     │  │                 │     ├─ Clone module IR → compiler working module C1 ↗ [C++:L381] ↗ caller [C++:L381]
   │     │  │                 │     │  G1 remains retained; declaration seeding above is distinct from lowering C1 ↗ [C++:L381] ↗ caller [PY:L1042]
   │     │  │                 │     ├─ Gather pipeline options; detect prepopulated declarations; evaluate graph-artifact cache ↗ [C++:L397] ↗ caller [C++:L397]
   │     │  │                 │     ├─ Cache hit → reuse prepared artifact; miss/disabled → convertMOToMEF(C1) ↗ [C++:L414] ↗ caller [C++:L414]
   │     │  │                 │     └─ Cold path: runGCPipeline → MO → MAP → MOGG → MGP ↗ [C++:L361] Compiler card · dialect transformations ↗ caller [C++:L361]
   │     │  │                 │        Kernel specialization/JIT and target code emission are pipeline work ↗ [C++:L638] Kernel card · Mojo, LLVM and target code ↗ caller [C++:L361]
   │     │  │                 │        Export MGP/code to MEF bytes; pre-jit stop produces an error instead ↗ [C++:L372] Artifact card · prepared MEF bytes ↗ caller [C++:L372]
   │     │  │                 └─ CompiledModels::read(bytes, frameworkCtx, sourceModel) → model descriptors ↗ [C++:L118] ↗ caller [C++:L118]
   │     │  ├─ On synchronous setup failure: release lock and raise compile message ↗ [PY:L1301] ↗ caller [PY:L1301]
   │     │  └─ Register done callback → release lock when compilation resolves; return pending handle ↗ [PY:L1305] ↗ caller [PY:L1305]
   │     ├─ CompiledModel(handle, expected_weights={}) → wrapper CM1 ↗ [PY:L1055] ↗ caller [PY:L1055]
   │     └─ Capture top-level graph names, including fused_attention; return CM1 to compile ↗ [PY:L1059] ↗ caller [PY:L1059]
   │
   ├─ Back in compile: handle = CM1._compiled; assert AsyncValue ↗ [PY:L1109] ↗ caller [PY:L1109]
   ├─ handle.wait() → block caller until compile finishes ↗ [PY:L1112] ↗ caller [PY:L1112]
   ├─ handle.exception(): failure raises; configured pre-jit stop raises CompilationStopped ↗ [PY:L1113] ↗ caller [PY:L1113]
   └─ Return resolved CM1 on success ↗ [PY:L1115] ↗ caller [PY:L74]
      CM1 holds compiled model descriptors/code; it is not an initialized Model and cannot yet execute ↗ [PY:L1073] ↗ caller [PY:L74]
```

Source links: [max/examples/custom_ops/fused_attention.py:L74](sources/max/examples/custom_ops/fused_attention.py#L74) · [max/python/max/engine/api.py:L1062](sources/max/python/max/engine/api.py#L1062) · [max/python/max/engine/api.py:L1103](sources/max/python/max/engine/api.py#L1103) · [max/python/max/engine/api.py:L969](sources/max/python/max/engine/api.py#L969) · [max/python/max/engine/api.py:L1007](sources/max/python/max/engine/api.py#L1007) · [max/python/max/engine/api.py:L1020](sources/max/python/max/engine/api.py#L1020) · [max/python/max/engine/api.py:L1022](sources/max/python/max/engine/api.py#L1022) · [max/python/max/engine/api.py:L1028](sources/max/python/max/engine/api.py#L1028) · [max/python/max/engine/api.py:L1042](sources/max/python/max/engine/api.py#L1042) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L75](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L75) · [max/python/max/engine/api.py:L1044](sources/max/python/max/engine/api.py#L1044) · [max/python/max/engine/api.py:L1293](sources/max/python/max/engine/api.py#L1293) · [max/python/max/engine/api.py:L1295](sources/max/python/max/engine/api.py#L1295) · [max/python/max/_core/max_license/modules/engine.cpp:L824](sources/max/python/max/_core/max_license/modules/engine.cpp#L824) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L658](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L658) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L670](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L670) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L676](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L676) · [max/lib/max_license/API/c/EngineContext.cpp:L227](sources/max/lib/max_license/API/c/EngineContext.cpp#L227) · [max/lib/max_license/API/c/EngineContext.cpp:L232](sources/max/lib/max_license/API/c/EngineContext.cpp#L232) · [max/lib/max_license/API/c/EngineContext.cpp:L271](sources/max/lib/max_license/API/c/EngineContext.cpp#L271) · [max/lib/max_license/API/c/EngineContext.cpp:L273](sources/max/lib/max_license/API/c/EngineContext.cpp#L273) · [max/lib/max_license/API/c/CompiledModels.cpp:L106](sources/max/lib/max_license/API/c/CompiledModels.cpp#L106) · [max/lib/max_license/API/c/CompiledModels.cpp:L111](sources/max/lib/max_license/API/c/CompiledModels.cpp#L111) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L381](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L381) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L397](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L397) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L414](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L414) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L361](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L361) · [GraphCompiler/lib/GraphCompiler.cpp:L638](sources/GraphCompiler/lib/GraphCompiler.cpp#L638) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L372](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L372) · [max/lib/max_license/API/c/CompiledModels.cpp:L118](sources/max/lib/max_license/API/c/CompiledModels.cpp#L118) · [max/python/max/engine/api.py:L1301](sources/max/python/max/engine/api.py#L1301) · [max/python/max/engine/api.py:L1305](sources/max/python/max/engine/api.py#L1305) · [max/python/max/engine/api.py:L1055](sources/max/python/max/engine/api.py#L1055) · [max/python/max/engine/api.py:L1059](sources/max/python/max/engine/api.py#L1059) · [max/python/max/engine/api.py:L1109](sources/max/python/max/engine/api.py#L1109) · [max/python/max/engine/api.py:L1112](sources/max/python/max/engine/api.py#L1112) · [max/python/max/engine/api.py:L1113](sources/max/python/max/engine/api.py#L1113) · [max/python/max/engine/api.py:L1115](sources/max/python/max/engine/api.py#L1115) · [max/python/max/engine/api.py:L1073](sources/max/python/max/engine/api.py#L1073)

## session.init(compiled) · S1 + CM1 → M1

```text
caller: model = session.init(compiled) ↗ [PY:L75]
└─ InferenceSession.init(compiled=CM1, weights_registry=None) ↗ [PY:L1126]
   ├─ self.init_all(CM1, weights_registry=None) ↗ [PY:L1151]
   │  └─ InferenceSession.init_all ↗ [PY:L1160]
   │     ├─ Virtual-device mode? Return named CompileOnlyModel stand-ins; real-device path continues ↗ [PY:L1183] ↗ caller [PY:L1183]
   │     ├─ weights_registry_real = weights_registry or {} ↗ [PY:L1202] ↗ caller [PY:L1202]
   │     ├─ Validate required weight names and host/device class against CM1._expected_weights ↗ [PY:L1209] ↗ caller [PY:L1209]
   │     │  Attention: expected_weights={} and registry={} → nothing missing ↗ [PY:L1029] Initialization card · bindings and readiness ↗ caller [PY:L1209]
   │     ├─ Check each supplied weight is contiguous; invalid input raises before native load ↗ [PY:L1233] ↗ caller [PY:L1233]
   │     ├─ If artifact holds resolved _CompiledModels directly: cache an async wrapper for reuse ↗ [PY:L1239] ↗ caller [PY:L1239]
   │     ├─ self._impl._load_all(CM1._compiled, {}) within init phase recording ↗ [PY:L1244]
   │     │  └─ nanobind _load_all → session.loadAll(compiled.copy(), weightsDict) ↗ [C++:L798]
   │     │     └─ Core::InferenceSession::loadAll ↗ [C++:L290]
   │     │        ├─ buildWeightsFromDict → shared WeightsRegistry + managed tensor owners ↗ [C++:L293]
   │     │        │  └─ Empty dictionary produces no external weight entries for fused attention ↗ [C++:L214] ↗ caller [C++:L293]
   │     │        ├─ Release Python GIL while awaiting compiled models ↗ [C++:L299] ↗ caller [C++:L299]
   │     │        ├─ Enumerate model descriptors; fused_attention has one model ↗ [C++:L309]
   │     │        │  ├─ load(model, sharedWeightsRegistry) ↗ [C++:L311]
   │     │        │  │  └─ Core::InferenceSession::load ↗ [C++:L276]
   │     │        │  │     ├─ runtimeContext->initModel(model, weightsRegistry) ↗ [C++:L278]
   │     │        │  │     │  └─ Engine::Context::initModel → addTask on CPU runtime worker ↗ [C++:L297]
   │     │        │  │     │     Worker context supplies AsyncRT CPUDevice in thread-local state ↗ [C++:L294] ↗ caller [C++:L297]
   │     │        │  │     │     └─ createExecutableModel(model, CPUDevice, runtimeConfig, weights) ↗ [C++:L299]
   │     │        │  │     │        ├─ Build DeviceMap from session device specifications and driver contexts ↗ [C++:L83] ↗ caller [C++:L83]
   │     │        │  │     │        ├─ model->setup(cpuDevice, weightsRegistry, deviceMap) ↗ [C++:L86]
   │     │        │  │     │        │  └─ Model::setup ↗ [C++:L185]
   │     │        │  │     │        │     ├─ Check CPU runtime; create initializer arguments including borrowed weight registry ↗ [C++:L187] ↗ caller [C++:L187]
   │     │        │  │     │        │     ├─ mapDeviceRefsToDevices → resolve model device references to driver contexts ↗ [C++:L197] ↗ caller [C++:L197]
   │     │        │  │     │        │     ├─ Ensure configured device stream count before capture ↗ [C++:L210] ↗ caller [C++:L210]
   │     │        │  │     │        │     ├─ initFn->invoke(initArgs, cpuDevice) → compiled initialization callable ↗ [C++:L220] Runtime card · loaded routines and resources ↗ caller [C++:L220]
   │     │        │  │     │        │     │  Initializer prepares runtime state/handles; it is not inference on user inputs Q/K/V ↗ [C++:L220] ↗ caller [C++:L220]
   │     │        │  │     │        │     └─ Construct ExecutableModel(chain, mgpContext, weightsRegistry, extraArgs) ↗ [C++:L237] ↗ caller [C++:L237]
   │     │        │  │     │        └─ waitForModelSetup → await chain, MGP context, registry and extra state; propagate errors ↗ [C++:L387] ↗ caller [C++:L91]
   │     │        │  │     └─ Release GIL while awaiting initialization; return executable async handle ↗ [C++:L281] ↗ caller [C++:L281]
   │     │        │  ├─ ModelHandle retains compiled artifact, native session, executable and weight owners ↗ [C++:L312] ↗ caller [C++:L312]
   │     │        │  └─ Cache Python input/output device vectors; append returned handle ↗ [C++:L316] ↗ caller [C++:L316]
   │     │        └─ Return vector of initialized Model handles to Python ↗ [C++:L320] ↗ caller [C++:L320]
   │     ├─ Map model names to returned wrappers → {"fused_attention": M1} ↗ [PY:L1247] ↗ caller [PY:L1247]
   │     └─ Reject duplicate model names; return dictionary ↗ [PY:L1248] ↗ caller [PY:L1248]
   │
   ├─ Back in init: require exactly one model, otherwise raise and advise init_all ↗ [PY:L1152] ↗ caller [PY:L1152]
   └─ Return next(iter(models.values())) → M1 ↗ [PY:L1158] ↗ caller [PY:L75]
      Next, outside this call: Buffer.from_numpy(...).to(device), then M1.execute(Q,K,V) ↗ [PY:L78] ↗ [PY:L82] ↗ caller [PY:L75]
```

Source links: [max/examples/custom_ops/fused_attention.py:L75](sources/max/examples/custom_ops/fused_attention.py#L75) · [max/python/max/engine/api.py:L1126](sources/max/python/max/engine/api.py#L1126) · [max/python/max/engine/api.py:L1151](sources/max/python/max/engine/api.py#L1151) · [max/python/max/engine/api.py:L1160](sources/max/python/max/engine/api.py#L1160) · [max/python/max/engine/api.py:L1183](sources/max/python/max/engine/api.py#L1183) · [max/python/max/engine/api.py:L1202](sources/max/python/max/engine/api.py#L1202) · [max/python/max/engine/api.py:L1209](sources/max/python/max/engine/api.py#L1209) · [max/python/max/engine/api.py:L1029](sources/max/python/max/engine/api.py#L1029) · [max/python/max/engine/api.py:L1233](sources/max/python/max/engine/api.py#L1233) · [max/python/max/engine/api.py:L1239](sources/max/python/max/engine/api.py#L1239) · [max/python/max/engine/api.py:L1244](sources/max/python/max/engine/api.py#L1244) · [max/python/max/_core/max_license/modules/engine.cpp:L798](sources/max/python/max/_core/max_license/modules/engine.cpp#L798) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L290](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L290) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L293](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L293) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L214](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L214) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L299](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L299) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L309](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L309) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L311](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L311) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L276](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L276) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L278](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L278) · [max/lib/max_license/API/c/EngineContext.cpp:L297](sources/max/lib/max_license/API/c/EngineContext.cpp#L297) · [max/lib/max_license/API/c/EngineContext.cpp:L294](sources/max/lib/max_license/API/c/EngineContext.cpp#L294) · [max/lib/max_license/API/c/EngineContext.cpp:L299](sources/max/lib/max_license/API/c/EngineContext.cpp#L299) · [max/lib/max_license/API/c/EngineContext.cpp:L83](sources/max/lib/max_license/API/c/EngineContext.cpp#L83) · [max/lib/max_license/API/c/EngineContext.cpp:L86](sources/max/lib/max_license/API/c/EngineContext.cpp#L86) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L185](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L185) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L187](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L187) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L197](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L197) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L210](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L210) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L220](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L220) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L237](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L237) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L387](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L387) · [max/lib/max_license/API/c/EngineContext.cpp:L91](sources/max/lib/max_license/API/c/EngineContext.cpp#L91) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L281](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L281) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L312](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L312) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L316](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L316) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L320](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L320) · [max/python/max/engine/api.py:L1247](sources/max/python/max/engine/api.py#L1247) · [max/python/max/engine/api.py:L1248](sources/max/python/max/engine/api.py#L1248) · [max/python/max/engine/api.py:L1152](sources/max/python/max/engine/api.py#L1152) · [max/python/max/engine/api.py:L1158](sources/max/python/max/engine/api.py#L1158) · [max/examples/custom_ops/fused_attention.py:L78](sources/max/examples/custom_ops/fused_attention.py#L78) · [max/examples/custom_ops/fused_attention.py:L82](sources/max/examples/custom_ops/fused_attention.py#L82)

## model.execute(Q, K, V) · M1 + three Buffers → [O1]

```text
caller: output = model.execute(Q, K, V) ↗ [PY:L82] ↗ caller [PY:L82]
├─ Before execute: np.random.seed(123); create three real NumPy matrices ↗ [PY:L77]
│  ├─ Buffer.from_numpy(Q_array).to(device) → BQ1 ↗ [PY:L78] ↗ caller [PY:L78]
│  ├─ Buffer.from_numpy(K_array).to(device) → BK1 ↗ [PY:L79] ↗ caller [PY:L79]
│  └─ Buffer.from_numpy(V_array).to(device) → BV1 ↗ [PY:L80] ↗ caller [PY:L80]
│     CPU: each has 8×8×4=256 bytes; GPU: each has 32×32×4=4096 bytes ↗ [PY:L78] ↗ caller [PY:L78]
│
└─ Model.execute is installed as _Model_execute; @traced records the call ↗ [PY:L291] ↗ [PY:L118]
   ├─ Iterate positional inputs in Q,K,V order ↗ [PY:L122]
   │  ├─ Validate contiguity; existing Buffer inputs remain Buffers ↗ [PY:L124] ↗ caller [PY:L124]
   │  ├─ Alternative DLPack/scalar objects use their conversion branches ↗ [PY:L130] ↗ caller [PY:L130]
   │  └─ input_impls=[BQ1,BK1,BV1]; unsupported types raise before native execution ↗ [PY:L136] ↗ [PY:L144] ↗ caller [PY:L144]
   └─ self._execute_device_tensors(input_impls) ↗ [PY:L145]
      └─ nanobind → ModelHandle::executeDeviceTensors([BQ1,BK1,BV1]) ↗ [C++:L565]
         ├─ No loadedModel? empty result; valid initialized M1 continues ↗ [C++:L457] ↗ caller [C++:L457]
         ├─ validateInputSpecList → count=3, shapes/dtype match compiled signature ↗ [C++:L461]
         │  └─ Expected CPU example: three float32[8,8]; wrong count or static size raises ↗ [C++:L438] ↗ caller [C++:L461]
         ├─ runtimeCtx=S1.native.runtimeContext; create input AsyncValue list ↗ [C++:L464] ↗ caller [C++:L464]
         ├─ For each of BQ1,BK1,BV1 ↗ [C++:L475]
         │  ├─ verifyInputDevice against corresponding model input DeviceRef ↗ [C++:L476] ↗ caller [C++:L476]
         │  ├─ getInputStorage → buffer owner, pointer and byte size ↗ [C++:L479] ↗ caller [C++:L479]
         │  └─ prepareInput(cpuDevice, storage.buf, tensorSpec) ↗ [C++:L484]
         │     └─ Driver::prepareInput → retain storage in ready Tensor AsyncValue ↗ [C++:L112]
         │        ├─ Example GPU input: symbolic pointer pQ, bytes=4096, spec=f32[32,32] ↗ [C++:L116] ↗ caller [C++:L484]
         │        └─ TensorBufferRef shares same storage; no new transfer in this wrapper step ↗ [C++:L121] ↗ caller [C++:L121]
         └─ executePreparedDeviceTensors(runtimeCtx, inputValues) ↗ [C++:L518]
            ├─ Release Python GIL while running native execution ↗ [C++:L529] ↗ caller [C++:L529]
            ├─ Driver::executeFromDriver(context, loadedModel.copy(), inputValues) ↗ [C++:L530]
            │  ├─ Await initialized executable; errors stop here ↗ [C++:L204] ↗ caller [C++:L204]
            │  ├─ model->execute(inputs) → ExecutableModel::execute ↗ [C++:L212]
            │  │  ├─ passedArgs=[chain,MGP state,weights,extraArgs,...Q/K/V AsyncValues] ↗ [C++:L373] ↗ caller [C++:L378]
            │  │  ├─ Set current AsyncRT CPUDevice for host routines ↗ [C++:L382] ↗ caller [C++:L382]
            │  │  └─ getExecuteFn()->invoke(passedArgs,cpuDevice) ↗ [C++:L384] E · Buffers, execution and output
            │  │     ├─ Compiled host routine reaches specialized FusedAttention.execute ↗ [DOC:L74] ↗ caller [C++:L384]
            │  │     ├─ CPU target → fused_attention_cpu[BN=4,BD=4] ↗ [DOC:L101] ↗ caller [DOC:L101]
            │  │     └─ GPU target → fused_attention_gpu[BN=16,BD=8] → enqueue_function ↗ [DOC:L105] Kernel · CPU/GPU dispatch ↗ caller [DOC:L105]
            │  │        └─ Kernel computes tiled QKᵀ → online softmax → PV → O ↗ [DOC:L403] Online softmax · Two tiles ↗ caller [DOC:L403]
            │  └─ extractDriverResults(context,executable,result AsyncValues) ↗ [C++:L213]
            │     ├─ Resolve output device from the session config ↗ [C++:L125] ↗ caller [C++:L125]
            │     ├─ awaitOrError(results) → result readiness or diagnostic ↗ [C++:L137] ↗ caller [C++:L137]
            │     ├─ Drop first native result: internal ordering chain ↗ [C++:L159] ↗ caller [C++:L159]
            │     └─ Tensor result → DeviceBuffer + spec + selected device ↗ [C++:L166] ↗ caller [C++:L178]
            └─ extractPyBufferResults → output=[O1] ↗ [C++:L538]
               ├─ Preserve storage/aliases; return a Python Buffer wrapper ↗ [C++:L70] ↗ caller [C++:L538]
               └─ Caller print(output) prints the list; no explicit to(CPU()) or parity assertion ↗ [PY:L83] ↗ caller [PY:L83]
                  OUT: one float32[N,D] Buffer; S1/M1 and resident code remain available ↗ [C++:L79] ↗ caller [PY:L82]
```

Source links: [max/examples/custom_ops/fused_attention.py:L82](sources/max/examples/custom_ops/fused_attention.py#L82) · [max/examples/custom_ops/fused_attention.py:L77](sources/max/examples/custom_ops/fused_attention.py#L77) · [max/examples/custom_ops/fused_attention.py:L78](sources/max/examples/custom_ops/fused_attention.py#L78) · [max/examples/custom_ops/fused_attention.py:L79](sources/max/examples/custom_ops/fused_attention.py#L79) · [max/examples/custom_ops/fused_attention.py:L80](sources/max/examples/custom_ops/fused_attention.py#L80) · [max/python/max/engine/api.py:L291](sources/max/python/max/engine/api.py#L291) · [max/python/max/engine/api.py:L118](sources/max/python/max/engine/api.py#L118) · [max/python/max/engine/api.py:L122](sources/max/python/max/engine/api.py#L122) · [max/python/max/engine/api.py:L124](sources/max/python/max/engine/api.py#L124) · [max/python/max/engine/api.py:L130](sources/max/python/max/engine/api.py#L130) · [max/python/max/engine/api.py:L136](sources/max/python/max/engine/api.py#L136) · [max/python/max/engine/api.py:L144](sources/max/python/max/engine/api.py#L144) · [max/python/max/engine/api.py:L145](sources/max/python/max/engine/api.py#L145) · [max/python/max/_core/max_license/modules/engine.cpp:L565](sources/max/python/max/_core/max_license/modules/engine.cpp#L565) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L457](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L457) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L461](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L461) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L438](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L438) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L464](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L464) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L475](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L475) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L476](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L476) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L479](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L479) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L484](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L484) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L112](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L112) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L116](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L116) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L121](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L121) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L518](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L518) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L529](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L529) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L530](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L530) · [max/lib/max_license/API/c/ModelExecution.cpp:L204](sources/max/lib/max_license/API/c/ModelExecution.cpp#L204) · [max/lib/max_license/API/c/ModelExecution.cpp:L212](sources/max/lib/max_license/API/c/ModelExecution.cpp#L212) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L373](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L373) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L378](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L378) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L382](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L382) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L384](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L384) · [max/examples/custom_ops/kernels/fused_attention.mojo:L74](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L74) · [max/examples/custom_ops/kernels/fused_attention.mojo:L101](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) · [max/examples/custom_ops/kernels/fused_attention.mojo:L105](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L105) · [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/lib/max_license/API/c/ModelExecution.cpp:L213](sources/max/lib/max_license/API/c/ModelExecution.cpp#L213) · [max/lib/max_license/API/c/ModelExecution.cpp:L125](sources/max/lib/max_license/API/c/ModelExecution.cpp#L125) · [max/lib/max_license/API/c/ModelExecution.cpp:L137](sources/max/lib/max_license/API/c/ModelExecution.cpp#L137) · [max/lib/max_license/API/c/ModelExecution.cpp:L159](sources/max/lib/max_license/API/c/ModelExecution.cpp#L159) · [max/lib/max_license/API/c/ModelExecution.cpp:L166](sources/max/lib/max_license/API/c/ModelExecution.cpp#L166) · [max/lib/max_license/API/c/ModelExecution.cpp:L178](sources/max/lib/max_license/API/c/ModelExecution.cpp#L178) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L538](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L538) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L70](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L70) · [max/examples/custom_ops/fused_attention.py:L83](sources/max/examples/custom_ops/fused_attention.py#L83) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L79](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L79)

## Kernel · Dispatch, tile arithmetic and storage

```text
caller: compiled FusedAttention.execute(output, query, key, value, ctx) ↗ [DOC:L74]
├─ Infer dtype=float32, rank=2 from tensor types; explicit BN/BD; target from device ↗ [DOC:L75] Dispatch · Specialization parameters
│  ├─ CPU: [BN=4,BD=4,target="cpu"]; GPU: [BN=16,BD=8,target="gpu"] ↗ [PY:L33] ↗ [PY:L38] ↗ caller [PY:L60]
│  └─ comptime assert rank==2; convert output/Q/K/V to tile tensor views ↗ [DOC:L88] ↗ caller [DOC:L91]
├─ CPU route: fused_attention_cpu[4,4](Q,K,V,O) ↗ [DOC:L101]
│  ├─ For each query tile [4,8] and output-column tile [4,4] ↗ [MOJO:L174] ↗ caller [DOC:L101]
│  ├─ Initialize running state m,l,O; visit two key/value tiles of 4 rows ↗ [MOJO:L185] ↗ caller [MOJO:L178]
│  └─ Same stable normalized-output recurrence; final O tile store ↗ [MOJO:L230] Online softmax · Recurrence ↗ caller [MOJO:L198]
└─ GPU route: fused_attention_gpu[16,8](ctx,Q,K,V,O) ↗ [DOC:L105]
   ├─ Specialize fused_attention_kernel for dtypes/layouts and BN/BD ↗ [DOC:L391] ↗ caller [DOC:L391]
   ├─ ctx.enqueue_function[kernel_func](Q,K,V,O,grid_dim=(4,2),block_dim=32) ↗ [DOC:L403] Launch · Blocks and driver handoff ↗ caller [DOC:L403]
   └─ Each block selects query rows by block_idx.y and output columns by block_idx.x ↗ [DOC:L333]
      ├─ Q_tile=[16,32]; retained m/l=[16], O_i=[16,8] ↗ [DOC:L335] ↗ caller [MOJO:L333]
      ├─ BN_1=8; for four key/value tiles ↗ [DOC:L341] Storage · Tiles and retained state
      │  ├─ K_tile=[8,32], V_tile=[8,8]; S=Q_tile×K_tileᵀ=[16,8] ↗ [DOC:L344] ↗ caller [DOC:L346]
      │  ├─ matmul["gpu",transpose_b=True]: fragments → MMA → shared output ↗ [DOC:L276] MMA · Loads, accumulation and barrier ↗ caller [DOC:L346]
      │  ├─ m_new=max(m_old,row_max(S)) ↗ [DOC:L348] ↗ caller [DOC:L349]
      │  ├─ alpha=exp(m_old-m_new); E=exp(S-m_new); l_new=alpha*l_old+sum(E) ↗ [DOC:L352] ↗ caller [MOJO:L355]
      │  ├─ Normalize E/=l_new; rescale old O*=l_old/l_new*alpha ↗ [DOC:L366] ↗ caller [DOC:L367]
      │  └─ PV=E×V_tile; O+=PV; keep m_new and l_new for next tile ↗ [DOC:L374] ↗ caller [DOC:L375]
      └─ Store O_i into O.tile[BN,BD](block_idx.y,block_idx.x) once after loop ↗ [DOC:L378] ↗ caller [DOC:L378]
```

Source links: [max/examples/custom_ops/kernels/fused_attention.mojo:L74](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L74) · [max/examples/custom_ops/kernels/fused_attention.mojo:L75](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L75) · [max/examples/custom_ops/fused_attention.py:L33](sources/max/examples/custom_ops/fused_attention.py#L33) · [max/examples/custom_ops/fused_attention.py:L38](sources/max/examples/custom_ops/fused_attention.py#L38) · [max/examples/custom_ops/fused_attention.py:L60](sources/max/examples/custom_ops/fused_attention.py#L60) · [max/examples/custom_ops/kernels/fused_attention.mojo:L88](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L88) · [max/examples/custom_ops/kernels/fused_attention.mojo:L91](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L91) · [max/examples/custom_ops/kernels/fused_attention.mojo:L101](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) · [max/examples/custom_ops/kernels/fused_attention.mojo:L174](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L174) · [max/examples/custom_ops/kernels/fused_attention.mojo:L185](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L185) · [max/examples/custom_ops/kernels/fused_attention.mojo:L178](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L178) · [max/examples/custom_ops/kernels/fused_attention.mojo:L230](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L230) · [max/examples/custom_ops/kernels/fused_attention.mojo:L198](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L198) · [max/examples/custom_ops/kernels/fused_attention.mojo:L105](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L105) · [max/examples/custom_ops/kernels/fused_attention.mojo:L391](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L391) · [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/examples/custom_ops/kernels/fused_attention.mojo:L333](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) · [max/examples/custom_ops/kernels/fused_attention.mojo:L335](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L335) · [max/examples/custom_ops/kernels/fused_attention.mojo:L333](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) · [max/examples/custom_ops/kernels/fused_attention.mojo:L341](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L341) · [max/examples/custom_ops/kernels/fused_attention.mojo:L344](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L344) · [max/examples/custom_ops/kernels/fused_attention.mojo:L346](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L346) · [max/examples/custom_ops/kernels/fused_attention.mojo:L276](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L276) · [max/examples/custom_ops/kernels/fused_attention.mojo:L348](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L348) · [max/examples/custom_ops/kernels/fused_attention.mojo:L349](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L349) · [max/examples/custom_ops/kernels/fused_attention.mojo:L352](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L352) · [max/examples/custom_ops/kernels/fused_attention.mojo:L355](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L355) · [max/examples/custom_ops/kernels/fused_attention.mojo:L366](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L366) · [max/examples/custom_ops/kernels/fused_attention.mojo:L367](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L367) · [max/examples/custom_ops/kernels/fused_attention.mojo:L374](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L374) · [max/examples/custom_ops/kernels/fused_attention.mojo:L375](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L375) · [max/examples/custom_ops/kernels/fused_attention.mojo:L378](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L378)

### Sizes · Actual CPU/GPU contracts

**Q asks, K supplies the vectors to compare, and V supplies the content to mix.** For each query row i, the scores are dot products Q[i,:]·K[j,:]. Softmax across j makes weights; O[i,:] is their weighted sum of V rows.

The actual [caller](sources/max/examples/custom_ops/fused_attention.py#L29) selects CPU [8,8] or GPU [32,32]. Its Q/K/V have the same type but represent different matrices. **N is a row count; D is a feature count.** BN and BD subdivide work, not the number of graph arguments.

On GPU, BN=16 and BD=8 divide O into eight [16,8] tiles. The independent key tile BN_1=8 means four key iterations per output block. CPU uses BN=BD=4 and two key iterations. The source computes **unscaled rank-2 attention**; heads, causal masking, dropout and 1/√D scaling are absent.

```text
caller: main() chooses sizes and type arguments ↗ [PY:L29]
├─ accelerator_count() selects CPU() or Accelerator() ↗ [PY:L29]
├─ CPU: N=D=8, BD=BN=4 ↗ [PY:L31] ↗ caller [PY:L30]
├─ GPU: N=D=32, BD=8, BN=16 ↗ [PY:L36] ↗ caller [PY:L35]
├─ Q type ← dtype + [N,D] + DeviceRef(device) ↗ [PY:L45] ↗ caller [PY:L44]
├─ K type ← same contract, different input slot ↗ [PY:L48] ↗ caller [PY:L47]
├─ V type ← same contract, different input slot ↗ [PY:L51] ↗ caller [PY:L50]
└─ TO ← declared [N,D]; corresponds to softmax(QKᵀ)V ↗ [PY:L63] ↗ caller [PY:L62]
```

Source links: [max/examples/custom_ops/fused_attention.py:L29](sources/max/examples/custom_ops/fused_attention.py#L29) · [max/examples/custom_ops/fused_attention.py:L31](sources/max/examples/custom_ops/fused_attention.py#L31) · [max/examples/custom_ops/fused_attention.py:L30](sources/max/examples/custom_ops/fused_attention.py#L30) · [max/examples/custom_ops/fused_attention.py:L36](sources/max/examples/custom_ops/fused_attention.py#L36) · [max/examples/custom_ops/fused_attention.py:L35](sources/max/examples/custom_ops/fused_attention.py#L35) · [max/examples/custom_ops/fused_attention.py:L45](sources/max/examples/custom_ops/fused_attention.py#L45) · [max/examples/custom_ops/fused_attention.py:L44](sources/max/examples/custom_ops/fused_attention.py#L44) · [max/examples/custom_ops/fused_attention.py:L48](sources/max/examples/custom_ops/fused_attention.py#L48) · [max/examples/custom_ops/fused_attention.py:L47](sources/max/examples/custom_ops/fused_attention.py#L47) · [max/examples/custom_ops/fused_attention.py:L51](sources/max/examples/custom_ops/fused_attention.py#L51) · [max/examples/custom_ops/fused_attention.py:L50](sources/max/examples/custom_ops/fused_attention.py#L50) · [max/examples/custom_ops/fused_attention.py:L63](sources/max/examples/custom_ops/fused_attention.py#L63) · [max/examples/custom_ops/fused_attention.py:L62](sources/max/examples/custom_ops/fused_attention.py#L62)

### A.0 · default MLIR context

**The shared object and the active scope are different things.** [_DEFAULT_MLIR_CONTEXT](sources/max/python/max/_mlir_context.py#L26) is one module-level instance. `mlir.Context.current` reads the top of this thread's entered-context stack. A new worker can see no current context even while main has MCX1 entered.

An MLIR Context is a compiler environment: dialects define operation/type names; interned types and attributes can be reused in that environment. It is **not a tensor store, graph container, or Engine::Context**.

`@in_default_mlir_context` replaces the constructor callable with a wrapper using `ensure_default_mlir_context()`. [MLIRThreadPoolExecutor](sources/max/python/max/_mlir_context.py#L89) also wraps submitted tasks. Both preserve an existing current context and enter MCX1 only when needed. Use the comparison selector to follow that branch; it is not a separate default instance per worker.

```text
caller: @in_default_mlir_context wraps Graph.__init__ ↗ [PY:L819]
├─ shared _DEFAULT_MLIR_CONTEXT = mlir.Context() ↗ [PY:L26]
├─ main enters MCX1; shutdown registers matching exit ↗ [PY:L32] ↗ caller [PY:L32]
├─ wrapper(*args) enters ensure_default_mlir_context() ↗ [PY:L83]
│  ├─ read mlir.Context.current on THIS thread ↗ [PY:L63]
│  ├─ None → with MCX1; yield to constructor ↗ [PY:L64] ↗ caller [PY:L83]
│  └─ active context → yield without replacement ↗ [PY:L67] ↗ caller [PY:L83]
├─ fn(*args) constructs Graph IR ↗ [PY:L84] ↗ caller [PY:L819]
└─ worker route: MLIRThreadPoolExecutor.submit ↗ [PY:L90]
   └─ submit(call_with_default_mlir_context, fn, ...) ↗ [PY:L97] ↗ caller [PY:L97]
```

Source links: [max/python/max/graph/graph.py:L819](sources/max/python/max/graph/graph.py#L819) · [max/python/max/_mlir_context.py:L26](sources/max/python/max/_mlir_context.py#L26) · [max/python/max/_mlir_context.py:L32](sources/max/python/max/_mlir_context.py#L32) · [max/python/max/_mlir_context.py:L83](sources/max/python/max/_mlir_context.py#L83) · [max/python/max/_mlir_context.py:L63](sources/max/python/max/_mlir_context.py#L63) · [max/python/max/_mlir_context.py:L64](sources/max/python/max/_mlir_context.py#L64) · [max/python/max/_mlir_context.py:L67](sources/max/python/max/_mlir_context.py#L67) · [max/python/max/_mlir_context.py:L84](sources/max/python/max/_mlir_context.py#L84) · [max/python/max/_mlir_context.py:L90](sources/max/python/max/_mlir_context.py#L90) · [max/python/max/_mlir_context.py:L97](sources/max/python/max/_mlir_context.py#L97)

### _location() · What is loc?

**loc is provenance metadata.** it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module.

[_location()](sources/max/python/max/graph/graph.py#L438) checks the current context, reads the live capture flag, extracts Python stack summaries when enabled, and can wrap profiling labels. In this caller an illustrative source stack ends at `main → fused_attention.py:41 → Graph(...)`. The native binding makes an **opaque Location carrying summaries**; it need not print as a simple filename-location string.

The same LOC1 is explicitly passed to the module and graph. [The native GraphOp builder](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) adds it to each input block argument. After the with-scope closes, the operations still carry it. A disabled capture path uses unknown; no fake traceback should be displayed as captured evidence.

```text
caller: with _location() as loc ↗ [PY:L851]
└─ _location(ignore_frames=1, attach_profile_scopes=True) ↗ [PY:L438]
   ├─ current context absent → RuntimeError ↗ [PY:L442] ↗ caller [PY:L851]
   ├─ capture disabled → Location.unknown() ↗ [PY:L449] ↗ caller [PY:L851]
   ├─ enabled → traceback.extract_stack() → trimmed summaries ↗ [PY:L457]
   │  └─ _graph.frame_loc(context,tb) → opaque Location ↗ [C++:L166] ↗ caller [PY:L459]
   ├─ wrap active profile scopes, if any ↗ [PY:L470] ↗ caller [PY:L471]
   ├─ ModuleOp(location=loc) stores LOC1 ↗ [PY:L856] ↗ caller [PY:L856]
   └─ GraphOp receives loc; body arguments use state.location ↗ [C++:L849] ↗ caller [PY:L862]
```

Source links: [max/python/max/graph/graph.py:L851](sources/max/python/max/graph/graph.py#L851) · [max/python/max/graph/graph.py:L438](sources/max/python/max/graph/graph.py#L438) · [max/python/max/graph/graph.py:L442](sources/max/python/max/graph/graph.py#L442) · [max/python/max/graph/graph.py:L449](sources/max/python/max/graph/graph.py#L449) · [max/python/max/graph/graph.py:L457](sources/max/python/max/graph/graph.py#L457) · [max/python/max/_core/max_license/modules/graph.cpp:L166](sources/max/python/max/_core/max_license/modules/graph.cpp#L166) · [max/python/max/graph/graph.py:L459](sources/max/python/max/graph/graph.py#L459) · [max/python/max/graph/graph.py:L470](sources/max/python/max/graph/graph.py#L470) · [max/python/max/graph/graph.py:L471](sources/max/python/max/graph/graph.py#L471) · [max/python/max/graph/graph.py:L856](sources/max/python/max/graph/graph.py#L856) · [GraphCompiler/lib/MODialect/MOOps.cpp:L849](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) · [max/python/max/graph/graph.py:L862](sources/max/python/max/graph/graph.py#L862)

### MLIR Module Reuse or create

**MLIR.module is IR container.** `builtin.module` is a native MLIR operation grouping definitions in a compilation unit. It owns a body region containing a body block; that block contains named graph operations.

The actual caller supplies no Module, so [ModuleOp::create](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L65) creates a fresh container. The reuse comparison keeps the same native container and any `@encoder` sibling. Each graph has its own region and arguments. **The graph's computation is nested inside G1, not directly in Bmodule.** The animation builds this nesting one level at a time.

```text
caller: self._module = module.mlir_module if module else builtin.ModuleOp(location=loc) ↗ [PY:L856]
├─ reuse branch: read Module.mlir_module → same enclosing operation ↗ [PY:L637] ↗ caller [PY:L856]
└─ creation branch: builtin.ModuleOp native binding ↗ [C++:L62]
   ├─ resolve supplied/current loc ↗ [C++:L64] ↗ caller [PY:L856]
   ├─ mlir::ModuleOp::create(loc) → MODULE1 ↗ [C++:L65] ↗ caller [PY:L856]
   ├─ body binding → getBodyRegion().front() → Bmodule ↗ [C++:L69] ↗ caller [PY:L858]
   └─ GraphOp inserted into Bmodule at end ↗ [PY:L860] ↗ caller [PY:L860]
```

Source links: [max/python/max/graph/graph.py:L856](sources/max/python/max/graph/graph.py#L856) · [max/python/max/graph/graph.py:L637](sources/max/python/max/graph/graph.py#L637) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L62](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L64](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L64) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L65](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L65) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L69](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L69) · [max/python/max/graph/graph.py:L858](sources/max/python/max/graph/graph.py#L858) · [max/python/max/graph/graph.py:L860](sources/max/python/max/graph/graph.py#L860)

### OpBuilder · Insertion point

**An insertion point is a destination block plus a position.** The [native binding](sources/max/python/max/_core/max_license/_core.cpp#L408) makes IP1 from `Bmodule.end`; another binding constructs a C++ OpBuilder from it.

GraphOp is inserted at IP1. Once its B0 is retained, [the graph insertion helper](sources/max/python/max/graph/graph.py#L1312) uses a different MLIR InsertionPoint inside B0 to record custom/output operations. The animated cursor shows both lists. A cursor is compiler bookkeeping, not a runtime instruction pointer or tensor memory address.

```text
caller: builder = OpBuilder(self._module.body.end) ↗ [PY:L858]
├─ ModuleOp.body → native body block Bmodule ↗ [C++:L69] ↗ caller [PY:L858]
├─ Block.end → InsertPoint(block,block.end()) ↗ [C++:L411] ↗ caller [PY:L858]
├─ OpBuilder(ip.getBlock(),ip.getPoint()) ↗ [C++:L614] ↗ caller [PY:L858]
├─ _mo.GraphOp(builder,...) inserts G1 into Bmodule ↗ [PY:L860] ↗ caller [PY:L860]
└─ later: _add_op_get_op_with_results ↗ [PY:L1284]
   ├─ ip = supplied _ip or InsertionPoint(self._body) ↗ [PY:L1312] ↗ caller [PY:L109]
   ├─ with ip,_location(): create graph-body op ↗ [PY:L1317] ↗ caller [PY:L109]
   └─ find inserted op and wrap result %o ↗ [PY:L1323] ↗ caller [PY:L1320]
```

Source links: [max/python/max/graph/graph.py:L858](sources/max/python/max/graph/graph.py#L858) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L69](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L69) · [max/python/max/_core/max_license/_core.cpp:L411](sources/max/python/max/_core/max_license/_core.cpp#L411) · [max/python/max/_core/max_license/_core.cpp:L614](sources/max/python/max/_core/max_license/_core.cpp#L614) · [max/python/max/graph/graph.py:L860](sources/max/python/max/graph/graph.py#L860) · [max/python/max/graph/graph.py:L1284](sources/max/python/max/graph/graph.py#L1284) · [max/python/max/graph/graph.py:L1312](sources/max/python/max/graph/graph.py#L1312) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/python/max/graph/graph.py:L1317](sources/max/python/max/graph/graph.py#L1317) · [max/python/max/graph/graph.py:L1323](sources/max/python/max/graph/graph.py#L1323) · [max/python/max/graph/graph.py:L1320](sources/max/python/max/graph/graph.py#L1320)

### to_mlir() · Q/K/V type fields

**32 as a shape dimension and float32 as an element dtype have different roles.** StaticDim(32) becomes a signed-i64 integer attribute, then a KGEN dimension-expression attribute. A pair of those forms MOSH ShapeAttr([32,32]); it describes dimensions rather than storing 1,024 float values.

`DType.float32` says a later element such as 1.5 is floating point. `DeviceRefAttr(gpu,0)` records placement. The [native type assembly](sources/max/python/max/graph/type.py#L439) combines these with a metadata dictionary.

**Attention's metadata is empty.** A [filter layout](sources/max/python/max/graph/type.py#L437) comparison could carry an HWIO/OIHW layout used by convolution dimension interpretation. That metadata is separate from the row-major TileTensor and TensorCore fragment layouts used later by this kernel.

```text
caller: input_types=[t.to_mlir() for t in input_types] ↗ [PY:L864]
└─ TensorType.to_mlir() ↗ [PY:L430]
   ├─ metadata=[]; if _layout then append layout attribute ↗ [PY:L436] ↗ caller [PY:L864]
   ├─ shape.to_mlir() ↗ [PY:L36]
   │  └─ StaticDim(32).to_mlir → IntegerAttr → CastFromBuiltinAttr ↗ [PY:L571] ↗ caller [PY:L39]
   ├─ self.dtype → DType.float32 ↗ [PY:L441] ↗ caller [PY:L439]
   ├─ device.to_mlir → DeviceRefAttr(gpu,0) ↗ [PY:L183] ↗ caller [PY:L442]
   └─ mo.TensorType(shape,dtype,device,DictionaryAttr(metadata)) ↗ [PY:L439] ↗ caller [PY:L864]
```

Source links: [max/python/max/graph/graph.py:L864](sources/max/python/max/graph/graph.py#L864) · [max/python/max/graph/type.py:L430](sources/max/python/max/graph/type.py#L430) · [max/python/max/graph/type.py:L436](sources/max/python/max/graph/type.py#L436) · [max/python/max/graph/shape.py:L36](sources/max/python/max/graph/shape.py#L36) · [max/python/max/graph/dim.py:L571](sources/max/python/max/graph/dim.py#L571) · [max/python/max/graph/shape.py:L39](sources/max/python/max/graph/shape.py#L39) · [max/python/max/graph/type.py:L441](sources/max/python/max/graph/type.py#L441) · [max/python/max/graph/type.py:L439](sources/max/python/max/graph/type.py#L439) · [max/python/max/graph/type.py:L183](sources/max/python/max/graph/type.py#L183) · [max/python/max/graph/type.py:L442](sources/max/python/max/graph/type.py#L442)

### GraphOp · Signature and body

**A region groups blocks; a block contains arguments and operations.** [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820) itself creates R0, then one B0, then adds one argument for each of the three input types. Python passes types and a location, not region data.

The function type initially says `(TQ,TK,TV)→()`. Both a functionType attribute and KGEN generator-signature attribute are attached. These form a caller/callee contract, not an execution result. `Graph.output` later declares one TO result. Runtime validation then compares three real input Buffers with this compiled contract.

The animation grows the region/block/argument structure incrementally. **%q, %k and %v share B0 but have distinct SSA identities.**

```text
caller: _mo.GraphOp(builder,loc,name,input_types,result_types=[]) ↗ [PY:L860]
└─ GraphOp::build(builder,state,name,inputTypes,resultTypes,isSubgraph) ↗ [C++:L820]
   ├─ attach sym_name; optional subgraph flag ↗ [C++:L825] ↗ caller [PY:L860]
   ├─ FunctionType=(TQ,TK,TV)→() ↗ [C++:L831] ↗ caller [PY:L864]
   ├─ generator signature + functionType attributes ↗ [C++:L832] ↗ caller [PY:L860]
   ├─ getParamDeclsFromShapes → input/result declarations ↗ [C++:L839] ↗ caller [PY:L860]
   ├─ state.addRegion() → R0 ↗ [C++:L846] ↗ caller [PY:L860]
   ├─ R0.emplaceBlock() → B0 ↗ [C++:L847] ↗ caller [PY:L860]
   └─ for input in inputTypes: B0.addArgument(input,state.location) ↗ [C++:L849] ↗ caller [PY:L864]
```

Source links: [max/python/max/graph/graph.py:L860](sources/max/python/max/graph/graph.py#L860) · [GraphCompiler/lib/MODialect/MOOps.cpp:L820](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820) · [GraphCompiler/lib/MODialect/MOOps.cpp:L825](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L825) · [GraphCompiler/lib/MODialect/MOOps.cpp:L831](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L831) · [max/python/max/graph/graph.py:L864](sources/max/python/max/graph/graph.py#L864) · [GraphCompiler/lib/MODialect/MOOps.cpp:L832](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L832) · [GraphCompiler/lib/MODialect/MOOps.cpp:L839](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L839) · [GraphCompiler/lib/MODialect/MOOps.cpp:L846](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L846) · [GraphCompiler/lib/MODialect/MOOps.cpp:L847](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L847) · [GraphCompiler/lib/MODialect/MOOps.cpp:L849](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849)

### Shape names · Graph parameters

**Static size 32 is already known. A symbolic name needs a binding.** The constructor visits all input shapes and deduplicates only `SymbolicDim.name` entries. Here `_params={}`, so its GraphOp declaration array is empty.

The comparison uses `["seq",32]` for Q/K/V and discovers one shared `seq`. That name constrains compatible input dimensions. **BN=16 and BD=8 remain compile-time custom-call parameters**; neither is an additional tensor argument or an entry in `_params`.

```text
caller: Graph.__init__ discovers dimensions before making G1 ↗ [PY:L840]
├─ walk input_types → t.shape → only SymbolicDim entries ↗ [PY:L843]
├─ dict.fromkeys(dim.name...) → deduplicate names; initial values=None ↗ [PY:L840] ↗ caller [PY:L840]
├─ create GraphOp, then assign input_param_decls ↗ [PY:L870]
│  └─ actual _params={} → ParamDeclArrayAttr([]) ↗ [PY:L871] ↗ caller [PY:L870]
└─ later, custom(parameters={BD:8,BN:16}) attaches a DIFFERENT dictionary ↗ [PY:L60] ↗ [PY:L113] ↗ caller [PY:L57]
```

Source links: [max/python/max/graph/graph.py:L840](sources/max/python/max/graph/graph.py#L840) · [max/python/max/graph/graph.py:L843](sources/max/python/max/graph/graph.py#L843) · [max/python/max/graph/graph.py:L870](sources/max/python/max/graph/graph.py#L870) · [max/python/max/graph/graph.py:L871](sources/max/python/max/graph/graph.py#L871) · [max/examples/custom_ops/fused_attention.py:L60](sources/max/examples/custom_ops/fused_attention.py#L60) · [max/python/max/graph/ops/custom.py:L113](sources/max/python/max/graph/ops/custom.py#L113) · [max/examples/custom_ops/fused_attention.py:L57](sources/max/examples/custom_ops/fused_attention.py#L57)

### Graph handles · CAPI and body

**CAPI is the MLIR C API pointer bridge between two Python binding implementations.** `_CAPIPtr` exports an operation-specific PyCapsule; `_CAPICreate` imports it into MLIR's Python wrapper. The illustrative pointer P stays identical through every adapter.

The capsule is a temporary adapter, **not an owner that independently keeps the whole graph alive**. Native IR/context ownership and Python operation references still matter. `regions[0].blocks[0]` follows OP1's existing region and block; `_current_block` and `_graph_body` initially alias B0. No package bytecode, native machine code or numerical tensor is produced here.

```text
caller: self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr) ↗ [PY:L874]
├─ MAX _CAPIPtr property → wrap(existing Operation*) ↗ [C++:L607]
│  └─ operation-named PyCapsule stores existing pointer P ↗ [LLVM:L331] ↗ caller [C++:L608]
├─ MLIR Python createFromCapsule → capsuleToOperation ↗ [LLVM:L1207]
│  ├─ extract MlirOperation {ptr=P} ↗ [LLVM:L339] ↗ caller [LLVM:L1208]
│  ├─ mlirOperationGetContext → existing context ↗ [LLVM:L695] ↗ caller [LLVM:L1211]
│  └─ PyOperation::forOperation(context,op) → wrapper for OP1 ↗ [LLVM:L1212] ↗ caller [PY:L874]
└─ _current_block = regions[0].blocks[0]; _graph_body = _current_block ↗ [PY:L875]
   └─ both Python references retain existing B0 ↗ [PY:L876] ↗ caller [PY:L875]
```

Source links: [max/python/max/graph/graph.py:L874](sources/max/python/max/graph/graph.py#L874) · [max/python/max/_core/max_license/_core.cpp:L607](sources/max/python/max/_core/max_license/_core.cpp#L607) · [external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h:L331](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L331) · [max/python/max/_core/max_license/_core.cpp:L608](sources/max/python/max/_core/max_license/_core.cpp#L608) · [external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp:L1207](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1207) · [external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h:L339](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L339) · [external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp:L1208](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1208) · [external/llvm-project/mlir/lib/CAPI/IR/IR.cpp:L695](sources/external/llvm-project/mlir/lib/CAPI/IR/IR.cpp#L695) · [external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp:L1211](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1211) · [external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp:L1212](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1212) · [max/python/max/graph/graph.py:L875](sources/max/python/max/graph/graph.py#L875) · [max/python/max/graph/graph.py:L876](sources/max/python/max/graph/graph.py#L876)

### Device metadata · Targets

**Placement, target descriptions, and runtime handles are separate.** `gpu:0` on a TensorType identifies placement. The module's `mo.device_info_mapping` adds API/architecture/model information for compiler decisions. The later session retains actual driver device/context references.

The helper always adds CPU metadata and optionally one detected accelerator. Architecture/model query failures become `"unknown"`; an already-present mapping is retained. Selecting a GPU here does not execute a kernel or move Q/K/V data.

```text
caller: self._populate_device_info_mapping() ↗ [PY:L877]
└─ _populate_device_info_mapping ↗ [PY:L1366]
   ├─ module ← self._mlir_op.block.owner ↗ [PY:L1368]
   ├─ mapping already exists → return ↗ [PY:L1369] ↗ caller [PY:L877]
   ├─ devices=[CPU()]; append Accelerator() when detected ↗ [PY:L1371]
   ├─ query architecture/model; each failure → unknown ↗ [PY:L1377] ↗ caller [PY:L877]
   ├─ _DeviceInfoAttr(label,api,arch,model,tile_based_fusion=False) ↗ [PY:L1384]
   └─ Attribute CAPI bridge → DictAttr(entries) on module ↗ [PY:L1392] ↗ caller [PY:L877]
```

Source links: [max/python/max/graph/graph.py:L877](sources/max/python/max/graph/graph.py#L877) · [max/python/max/graph/graph.py:L1366](sources/max/python/max/graph/graph.py#L1366) · [max/python/max/graph/graph.py:L1368](sources/max/python/max/graph/graph.py#L1368) · [max/python/max/graph/graph.py:L1369](sources/max/python/max/graph/graph.py#L1369) · [max/python/max/graph/graph.py:L1371](sources/max/python/max/graph/graph.py#L1371) · [max/python/max/graph/graph.py:L1377](sources/max/python/max/graph/graph.py#L1377) · [max/python/max/graph/graph.py:L1384](sources/max/python/max/graph/graph.py#L1384) · [max/python/max/graph/graph.py:L1392](sources/max/python/max/graph/graph.py#L1392)

### Graph bookkeeping · Weights and chains

**Weights describe persistent model parameters; chains describe dependencies.** `_weights={}` means this graph declares no external parameters to bind at init. It does not mean that Q/K/V will be zero.

The lazy map retains G1 and stores chain references by DeviceRef. With no explicit chain input, a first lookup for another device can use the existing CPU chain as its seed. Ordered operations may later replace a map entry with a newer token. **The dictionary is not GPU memory and a chain is not a CUDA/HIP stream handle.**

```text
caller: constructor starts graph bookkeeping ↗ [PY:L879]
├─ _weights={} ↗ [PY:L879] ↗ caller [PY:L879]
├─ _has_chain_input=False ↗ [PY:L880] ↗ caller [PY:L880]
└─ device_chains=_DeviceChainMap(self) ↗ [PY:L881]
   ├─ OrderedDict initialization; retain self._graph ↗ [PY:L126] ↗ caller [PY:L881]
   └─ later __getitem__(new_device) ↗ [PY:L134]
      ├─ explicit chain-input mode at graph body → add chain block argument ↗ [PY:L137] ↗ caller [PY:L136]
      └─ otherwise → use existing CPU chain as seed ↗ [PY:L148] ↗ caller [PY:L136]
```

Source links: [max/python/max/graph/graph.py:L879](sources/max/python/max/graph/graph.py#L879) · [max/python/max/graph/graph.py:L880](sources/max/python/max/graph/graph.py#L880) · [max/python/max/graph/graph.py:L881](sources/max/python/max/graph/graph.py#L881) · [max/python/max/graph/graph.py:L126](sources/max/python/max/graph/graph.py#L126) · [max/python/max/graph/graph.py:L134](sources/max/python/max/graph/graph.py#L134) · [max/python/max/graph/graph.py:L137](sources/max/python/max/graph/graph.py#L137) · [max/python/max/graph/graph.py:L136](sources/max/python/max/graph/graph.py#L136) · [max/python/max/graph/graph.py:L148](sources/max/python/max/graph/graph.py#L148)

### Chain · Ordering token

**A chain is an ordering token.** Its meaning is that dependency-producing work has completed. `ChainCreateOp(inputs=[])` has no earlier work to wait for, so it provides the always-ready seed.

The graph still has exactly three user arguments. CH0 is produced inside B0 and stored in `_always_ready_chain` and the CPU chain map. Runtime execution also uses chains to order resource work; that later chain is not a numerical fourth attention tensor.

```text
caller: _always_ready_chain = _ChainValue(_add_op_generated(...)[0]) ↗ [PY:L886]
├─ _mo.ChainCreateOp(inputs=[]) inserted in B0 ↗ [PY:L887] ↗ caller [PY:L886]
├─ wrap its one result as _ChainValue ↗ [PY:L886] ↗ caller [PY:L886]
└─ _update_chain(_always_ready_chain, DeviceRef.CPU()) ↗ [PY:L894]
   └─ device_chains[DeviceRef.CPU()]=chain ↗ [PY:L1064] ↗ caller [PY:L894]
```

Source links: [max/python/max/graph/graph.py:L886](sources/max/python/max/graph/graph.py#L886) · [max/python/max/graph/graph.py:L887](sources/max/python/max/graph/graph.py#L887) · [max/python/max/graph/graph.py:L894](sources/max/python/max/graph/graph.py#L894) · [max/python/max/graph/graph.py:L1064](sources/max/python/max/graph/graph.py#L1064)

### Last input · Tensor or chain?

**Attention's final argument is V, a TensorType.** Therefore `_has_chain_input` stays False and `Graph.inputs` returns all three tensor arguments. The constructor does not consume V as a dependency.

The comparison illustrates a parent-supplied trailing ChainType argument. That branch adopts the chain, marks internal chain-input mode, and excludes internal chains from the public inputs. Both cases keep **one entry block with several arguments**.

```text
caller: inspect input_args[-1] ↗ [PY:L896]
├─ no arguments → skip inspection ↗ [PY:L896] ↗ caller [PY:L896]
├─ last argument type is TensorType (V here) → leave CH0 ↗ [PY:L900] ↗ caller [PY:L896]
└─ comparison: ChainType → adopt chain and set _has_chain_input=True ↗ [PY:L900]
   ├─ _update_chain(chain) ↗ [PY:L902] ↗ caller [PY:L900]
   └─ Graph.inputs excludes trailing chain_count arguments ↗ [PY:L943] ↗ caller [PY:L943]
```

Source links: [max/python/max/graph/graph.py:L896](sources/max/python/max/graph/graph.py#L896) · [max/python/max/graph/graph.py:L900](sources/max/python/max/graph/graph.py#L900) · [max/python/max/graph/graph.py:L902](sources/max/python/max/graph/graph.py#L902) · [max/python/max/graph/graph.py:L943](sources/max/python/max/graph/graph.py#L943)

### P1 · Executable, parsing and bytecode

**`mojo` is a compiler-driver executable.** `subprocess_run_mojo` selects `MODULAR_MOJO_MAX_DRIVER_PATH` or a PATH lookup. SDK defaults can point to package `bin/mojo`; Bazel config supplies its runtime driver. The exact installed absolute path is environment-specific.

Cold preparation parses the source package, resolves dependencies, checks parametric IR, writes a private precompiled file, then publishes it with `os.replace`. **P1 stores package IR as versioned compressed MLIR bytecode**, not attention machine code. Shared dependencies supply imported declarations/bodies; Python caller code is separate.

The existing-.mojoc selector skips the subprocess. A source directory is precompiled even though its deterministic destination may already exist; compiler internals may cache work separately. Concrete dtype, rank, BN/BD, layouts and device target are specialized later during graph compilation.

```text
caller: self._import_kernels(_resolved_custom_extensions(custom_extensions)) ↗ [PY:L910]
├─ _resolved_custom_extensions combines explicit paths and backend overlays ↗ [PY:L363] ↗ caller [PY:L910]
└─ _import_kernels ↗ [PY:L1703]
   ├─ KernelLibrary.load_paths(paths) ↗ [PY:L269] ↗ caller [PY:L1704]
   │  ├─ existing .mojoc → add_path directly ↗ [PY:L286] ↗ caller [PY:L1704]
   │  └─ source directory → _build_mojo_source_package ↗ [PY:L141]
   │     ├─ deterministic destination keyed by source path and user ID ↗ [PY:L147] ↗ caller [PY:L289]
   │     ├─ separate process: subprocess_run_mojo(['precompile',...]) ↗ [PY:L178]
   │     │  ├─ executable override or shutil.which('mojo') ↗ [PY:L70] ↗ caller [PY:L178]
   │     │  ├─ parse → declaration resolution → package checks ↗ [C++:L391] ↗ [C++:L1756] ↗ [C++:L1119] ↗ caller [PY:L178]
   │     │  └─ header/compression + MLIR bytecode writer ↗ [C++:L137] ↗ caller [PY:L178]
   │     └─ os.replace → published P1 ↗ [PY:L179] ↗ caller [PY:L289]
   └─ attach loaded library paths to GraphOp.library ↗ [PY:L1706] ↗ caller [PY:L910]
```

Source links: [max/python/max/graph/graph.py:L910](sources/max/python/max/graph/graph.py#L910) · [max/python/max/graph/graph.py:L363](sources/max/python/max/graph/graph.py#L363) · [max/python/max/graph/graph.py:L1703](sources/max/python/max/graph/graph.py#L1703) · [max/python/max/graph/graph.py:L269](sources/max/python/max/graph/graph.py#L269) · [max/python/max/graph/graph.py:L1704](sources/max/python/max/graph/graph.py#L1704) · [max/python/max/graph/graph.py:L286](sources/max/python/max/graph/graph.py#L286) · [Mojo/python/mojo/paths.py:L141](sources/Mojo/python/mojo/paths.py#L141) · [Mojo/python/mojo/paths.py:L147](sources/Mojo/python/mojo/paths.py#L147) · [max/python/max/graph/graph.py:L289](sources/max/python/max/graph/graph.py#L289) · [Mojo/python/mojo/paths.py:L178](sources/Mojo/python/mojo/paths.py#L178) · [Mojo/python/mojo/run.py:L70](sources/Mojo/python/mojo/run.py#L70) · [Mojo/lib/MojoParser/EntryPoint.cpp:L391](sources/Mojo/lib/MojoParser/EntryPoint.cpp#L391) · [Mojo/lib/MojoParser/DeclResolver.cpp:L1756](sources/Mojo/lib/MojoParser/DeclResolver.cpp#L1756) · [Mojo/lib/Compiler/KGENCompiler.cpp:L1119](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1119) · [Mojo/lib/Support/MojoPrecompiledFile.cpp:L137](sources/Mojo/lib/Support/MojoPrecompiledFile.cpp#L137) · [Mojo/python/mojo/paths.py:L179](sources/Mojo/python/mojo/paths.py#L179) · [max/python/max/graph/graph.py:L1706](sources/max/python/max/graph/graph.py#L1706)

### with Graph · Active recording scope

**There are two scopes: the compiler Context and the current Graph.** The MLIR Context supplies the compiler environment. `CURRENT_GRAPH` is a Python ContextVar pointing to the graph receiving operations.

`__enter__` opens `_enter`; `_enter` saves a reset token, sets G1, and yields to the caller. `__exit__` closes it, resetting the previous graph even if construction fails. This differs from running a compiled model: the with-body only records symbolic operations.

```text
caller: with Graph(...) as graph ↗ [PY:L41]
├─ constructor returns G1; Python calls __enter__ ↗ [PY:L1134]
│  └─ _enter generator context manager ↗ [PY:L1142]
│     ├─ ensure_default_mlir_context() ↗ [PY:L1143] ↗ caller [PY:L1135]
│     ├─ token=CURRENT_GRAPH.set(self) ↗ [PY:L1144] ↗ caller [PY:L55]
│     └─ yield self → with-body records ops ↗ [PY:L1146] ↗ caller [PY:L56]
└─ __exit__ → context.__exit__ → finally reset(token) ↗ [PY:L1138]
   └─ CURRENT_GRAPH.reset(token) restores prior graph ↗ [PY:L1149] ↗ caller [PY:L55]
```

Source links: [max/examples/custom_ops/fused_attention.py:L41](sources/max/examples/custom_ops/fused_attention.py#L41) · [max/python/max/graph/graph.py:L1134](sources/max/python/max/graph/graph.py#L1134) · [max/python/max/graph/graph.py:L1142](sources/max/python/max/graph/graph.py#L1142) · [max/python/max/graph/graph.py:L1143](sources/max/python/max/graph/graph.py#L1143) · [max/python/max/graph/graph.py:L1135](sources/max/python/max/graph/graph.py#L1135) · [max/python/max/graph/graph.py:L1144](sources/max/python/max/graph/graph.py#L1144) · [max/examples/custom_ops/fused_attention.py:L55](sources/max/examples/custom_ops/fused_attention.py#L55) · [max/python/max/graph/graph.py:L1146](sources/max/python/max/graph/graph.py#L1146) · [max/examples/custom_ops/fused_attention.py:L56](sources/max/examples/custom_ops/fused_attention.py#L56) · [max/python/max/graph/graph.py:L1138](sources/max/python/max/graph/graph.py#L1138) · [max/python/max/graph/graph.py:L1149](sources/max/python/max/graph/graph.py#L1149)

### Inputs · Three symbolic wrappers

**q is an IR reference, not Q's matrix data.** `Graph.inputs` walks the one entry block's arguments, omits internal chains when needed, and wraps each native value with the Graph API's typed wrapper.

Here q→%arg0, k→%arg1, v→%arg2 all carry float32[32,32]@gpu:0. The actual BQ/BK/BV storage is created at caller lines 78–80 and bound by position at execute. **The three equal types do not mean the three input matrices have equal values.**

```text
caller: q,k,v,*_ = graph.inputs ↗ [PY:L56]
└─ Graph.inputs property ↗ [PY:L934]
   ├─ read _graph_body.arguments ↗ [PY:L943]
   ├─ chain_count=0 here → keep all three ↗ [PY:L944] ↗ caller [PY:L56]
   ├─ each arg → _Value._from_cmlir(arg) → Value.from_mlir ↗ [PY:L949] ↗ caller [PY:L56]
   └─ return tuple(q,k,v); no tensor-data allocation ↗ [PY:L948] ↗ caller [PY:L56]
```

Source links: [max/examples/custom_ops/fused_attention.py:L56](sources/max/examples/custom_ops/fused_attention.py#L56) · [max/python/max/graph/graph.py:L934](sources/max/python/max/graph/graph.py#L934) · [max/python/max/graph/graph.py:L943](sources/max/python/max/graph/graph.py#L943) · [max/python/max/graph/graph.py:L944](sources/max/python/max/graph/graph.py#L944) · [max/python/max/graph/graph.py:L949](sources/max/python/max/graph/graph.py#L949) · [max/python/max/graph/graph.py:L948](sources/max/python/max/graph/graph.py#L948)

### Output type · Future O

**Actual GPU values: `TensorType(DType.float32,[32,32],DeviceRef("gpu",0))`.** CPU uses `[8,8]@cpu:0`. The caller explicitly declares one output because each query row produces a weighted sum of V rows.

The equation is `[N,D]×[D,N]→[N,N]` scores, then `[N,N]×[N,D]→[N,D]` output. **TO is a symbolic contract.** It will become a runtime Buffer allocation only when the initialized model executes with real inputs.

```text
caller: out_types=[TensorType(dtype,[N,D],DeviceRef.from_device(device))] ↗ [PY:L63]
├─ dtype ← DType.float32 ↗ [PY:L26] ↗ caller [PY:L63]
├─ shape ← actual N/D branch ↗ [PY:L36] ↗ caller [PY:L63]
├─ device ← caller's selected device ↗ [PY:L29] ↗ caller [PY:L65]
└─ ops.custom converts TO with to_mlir ↗ [PY:L110]
   └─ native result type for mo.custom; no Buffer yet ↗ [PY:L439] ↗ caller [PY:L110]
```

Source links: [max/examples/custom_ops/fused_attention.py:L63](sources/max/examples/custom_ops/fused_attention.py#L63) · [max/examples/custom_ops/fused_attention.py:L26](sources/max/examples/custom_ops/fused_attention.py#L26) · [max/examples/custom_ops/fused_attention.py:L36](sources/max/examples/custom_ops/fused_attention.py#L36) · [max/examples/custom_ops/fused_attention.py:L29](sources/max/examples/custom_ops/fused_attention.py#L29) · [max/examples/custom_ops/fused_attention.py:L65](sources/max/examples/custom_ops/fused_attention.py#L65) · [max/python/max/graph/ops/custom.py:L110](sources/max/python/max/graph/ops/custom.py#L110) · [max/python/max/graph/type.py:L439](sources/max/python/max/graph/type.py#L439)

### Attributes · Name, device, BN/BD

**Example values are `symbol="modular_ops::fused_attention_custom"`, `device=(gpu,0)`, `BD=8:index`, `BN=16:index`.** Names match the Mojo registration, device kind/id select placement, and index attributes carry compile-time tile choices.

The source obtains the symbol/device first, inserts the op, then attaches parameters and the bridged device attribute before custom-kernel verification. CPU follows the same mechanism with `(cpu,0)` and BD=BN=4. **An index-valued size parameter is not a float32 tensor element.**

```text
caller: ops.custom(name,device,parameters,values,out_types) ↗ [PY:L57]
├─ Graph.current; default_mlir_context() ↗ [PY:L94]
├─ StringAttr.get(name,context) ↗ [PY:L96] ↗ caller [PY:L58]
├─ DeviceRef.from_device(device) ↗ [PY:L97] ↗ caller [PY:L59]
├─ insert the custom op before assigning its remaining attributes ↗ [PY:L109]
├─ parameters dictionary ↗ [PY:L113]
│  └─ Python int → IntegerAttr(index,value) ↗ [PY:L49] ↗ caller [PY:L116]
└─ device.to_mlir capsule → mlir.Attribute._CAPICreate → custom_op.device ↗ [PY:L122]
   └─ native DeviceRefAttr('gpu',0) here ↗ [PY:L181] ↗ caller [PY:L123]
```

Source links: [max/examples/custom_ops/fused_attention.py:L57](sources/max/examples/custom_ops/fused_attention.py#L57) · [max/python/max/graph/ops/custom.py:L94](sources/max/python/max/graph/ops/custom.py#L94) · [max/python/max/graph/ops/custom.py:L96](sources/max/python/max/graph/ops/custom.py#L96) · [max/examples/custom_ops/fused_attention.py:L58](sources/max/examples/custom_ops/fused_attention.py#L58) · [max/python/max/graph/ops/custom.py:L97](sources/max/python/max/graph/ops/custom.py#L97) · [max/examples/custom_ops/fused_attention.py:L59](sources/max/examples/custom_ops/fused_attention.py#L59) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/python/max/graph/ops/custom.py:L113](sources/max/python/max/graph/ops/custom.py#L113) · [max/python/max/graph/ops/custom.py:L49](sources/max/python/max/graph/ops/custom.py#L49) · [max/python/max/graph/ops/custom.py:L116](sources/max/python/max/graph/ops/custom.py#L116) · [max/python/max/graph/ops/custom.py:L122](sources/max/python/max/graph/ops/custom.py#L122) · [max/python/max/graph/type.py:L181](sources/max/python/max/graph/type.py#L181) · [max/python/max/graph/ops/custom.py:L123](sources/max/python/max/graph/ops/custom.py#L123)

### mo.custom · Detailed three-input call

**Three existing operands enter one custom op; one symbolic result leaves it.** The helper unwraps Python values/types/attributes into MLIR binding handles, activates an insertion point at B0.end, creates the operation, identifies it, verifies its structure and wraps its results.

```mlir
// Schematic MO, not a compiler dump or exact type printer syntax.
mo.graph @fused_attention(%q:T, %k:T, %v:T) -> T {
  %ch = mo.chain.create()
  %o = mo.custom(%q,%k,%v) {symbol="modular_ops::fused_attention_custom",
       parameters={BD=8:index,BN=16:index}, device=(gpu,0)} : T
  // graph.output will insert mo.output %o later
}
```

CUSTOM1 remains an opaque call to the attention implementation; it does not become an exposed graph of scores/exp/divide here. Generic structure verification happens during insertion. Package-specific verification runs after the remaining call attributes are attached.

```text
caller: results=ops.custom(...) ↗ [PY:L57]
├─ reject buffer/opaque in-place values; normalize tensor-like operands ↗ [PY:L99] ↗ caller [PY:L57]
└─ graph._add_op_get_op_with_results(mo.custom,[TO],values,symbol=...) ↗ [PY:L109]
   ├─ unwrap lists recursively; Values/Types/Attributes through CAPI ↗ [PY:L1293] ↗ caller [PY:L109]
   ├─ ip=InsertionPoint(self._body) ↗ [PY:L1312]
   ├─ with ip,_location(): op(*unwrapped_args,**kwargs) ↗ [PY:L1317] ↗ [PY:L1320]
   ├─ last_operation or prev_operation → staged_op ↗ [PY:L1323] ↗ caller [PY:L1320]
   ├─ _verify_op(staged_op) → structural verifier ↗ [PY:L1329] ↗ [PY:L1186] ↗ caller [PY:L1329]
   ├─ set output parameter declarations; wrap results ↗ [PY:L1351] ↗ [PY:L1360] ↗ caller [PY:L109]
   └─ return results, staged_op → attach params/device → custom verifier ↗ [PY:L113] ↗ [PY:L127] ↗ caller [PY:L57]
```

Source links: [max/examples/custom_ops/fused_attention.py:L57](sources/max/examples/custom_ops/fused_attention.py#L57) · [max/python/max/graph/ops/custom.py:L99](sources/max/python/max/graph/ops/custom.py#L99) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/python/max/graph/graph.py:L1293](sources/max/python/max/graph/graph.py#L1293) · [max/python/max/graph/graph.py:L1312](sources/max/python/max/graph/graph.py#L1312) · [max/python/max/graph/graph.py:L1317](sources/max/python/max/graph/graph.py#L1317) · [max/python/max/graph/graph.py:L1320](sources/max/python/max/graph/graph.py#L1320) · [max/python/max/graph/graph.py:L1323](sources/max/python/max/graph/graph.py#L1323) · [max/python/max/graph/graph.py:L1329](sources/max/python/max/graph/graph.py#L1329) · [max/python/max/graph/graph.py:L1186](sources/max/python/max/graph/graph.py#L1186) · [max/python/max/graph/graph.py:L1351](sources/max/python/max/graph/graph.py#L1351) · [max/python/max/graph/graph.py:L1360](sources/max/python/max/graph/graph.py#L1360) · [max/python/max/graph/ops/custom.py:L113](sources/max/python/max/graph/ops/custom.py#L113) · [max/python/max/graph/ops/custom.py:L127](sources/max/python/max/graph/ops/custom.py#L127)

### Kernel validation · P1 and MO

**The verifier matches a symbolic call to a declared kernel interface.** The registry maps the registration name to declarations; device resolution selects the suitable declaration. Operand inference then matches three MO operands to query/key/value and one result to the mutable output destination.

Python first selects analysis by the loaded package-path set. A cache miss lazily imports declaration packages and builds the registry; a hit reuses them, **while still verifying the new call**. Successful analysis is cached after native verification. This is distinct from source precompile and from native code-cache reuse.

`ctx` is a generated runtime argument, not a fourth tensor. dtype/rank/target can be inferred; Python supplies BN/BD explicitly. Parameter checks validate names and native attribute types. The comparison adds an invalid BX name to show where verification fails. **A passed verifier does not prove numeric correctness or GPU performance.**

```text
caller: graph._kernel_library.verify_custom_op(custom_op) ↗ [PY:L127]
├─ KernelLibrary.verify_custom_op: paths → path-set key → analysis lookup ↗ [PY:L314] ↗ [PY:L316] ↗ [PY:L317] ↗ caller [PY:L127]
│  ├─ cache hit → retain cached analysis; verify this call ↗ [PY:L319] ↗ [PY:L320] ↗ caller [PY:L127]
│  ├─ miss → verify current analysis; cache only after success ↗ [PY:L322] ↗ [PY:L323] ↗ caller [PY:L127]
│  └─ no paths → current native analysis verifier directly ↗ [PY:L325] ↗ caller [PY:L127]
├─ native Analysis::verifyCustomOp → ensureRegistry → ensureModule ↗ [C++:L70] ↗ [C++:L181]
│  ├─ module absent → create declaration module; import paths + built-ins ↗ [C++:L191] ↗ [C++:L199] ↗ caller [C++:L181]
│  └─ registry absent → MojoRegistry(module); otherwise reuse existing ↗ [C++:L183] ↗ [C++:L184] ↗ caller [C++:L70]
└─ MOGG::verifyCustomOp(registry,op) ↗ [C++:L634]
   ├─ get operation name; containsKernel gate ↗ [C++:L636] ↗ caller [C++:L70]
   ├─ getDeviceRefForOp; getResolvedKernel ↗ [C++:L650] ↗ caller [C++:L70]
   ├─ ensure execute exists ↗ [C++:L672] ↗ caller [C++:L70]
   ├─ mapMOOpToKernelImplementation ↗ [C++:L336]
   │  ├─ inputs %q/%k/%v → query/key/value ↗ [C++:L315] ↗ caller [C++:L677]
   │  └─ result %o → mutable output destination ↗ [C++:L324] ↗ caller [C++:L677]
   └─ validateCustomOpParameters ↗ [C++:L427]
      ├─ merge declared struct/function/shape parameters ↗ [C++:L451] ↗ [C++:L459] ↗ caller [C++:L685]
      ├─ supplied name missing → error ↗ [C++:L480] ↗ caller [C++:L685]
      └─ native attribute type compatible → success ↗ [C++:L492] ↗ [C++:L629] ↗ caller [C++:L685]
```

Source links: [max/python/max/graph/ops/custom.py:L127](sources/max/python/max/graph/ops/custom.py#L127) · [max/python/max/graph/graph.py:L314](sources/max/python/max/graph/graph.py#L314) · [max/python/max/graph/graph.py:L316](sources/max/python/max/graph/graph.py#L316) · [max/python/max/graph/graph.py:L317](sources/max/python/max/graph/graph.py#L317) · [max/python/max/graph/graph.py:L319](sources/max/python/max/graph/graph.py#L319) · [max/python/max/graph/graph.py:L320](sources/max/python/max/graph/graph.py#L320) · [max/python/max/graph/graph.py:L322](sources/max/python/max/graph/graph.py#L322) · [max/python/max/graph/graph.py:L323](sources/max/python/max/graph/graph.py#L323) · [max/python/max/graph/graph.py:L325](sources/max/python/max/graph/graph.py#L325) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L70](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L70) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L181](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L181) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L191](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L191) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L199](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L199) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L183](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L183) · [max/python/max/_core/max_license/lib/Graph/Bindings.cpp:L184](sources/max/python/max/_core/max_license/lib/Graph/Bindings.cpp#L184) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L634](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L634) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L636](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L636) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L650](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L650) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L672](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L672) · [GraphCompiler/lib/Conversion/OperandInference.cpp:L336](sources/GraphCompiler/lib/Conversion/OperandInference.cpp#L336) · [GraphCompiler/lib/Conversion/OperandInference.cpp:L315](sources/GraphCompiler/lib/Conversion/OperandInference.cpp#L315) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L677](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L677) · [GraphCompiler/lib/Conversion/OperandInference.cpp:L324](sources/GraphCompiler/lib/Conversion/OperandInference.cpp#L324) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L427](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L427) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L451](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L451) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L459](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L459) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L685](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L685) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L480](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L480) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L492](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L492) · [GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp:L629](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L629)

### Result wrappers · One symbolic output

**Example list: `[TensorValue(%o : float32[32,32] @ gpu:0)]`.** `%o` names one SSA result owned by CUSTOM1. The wrapper exposes `dtype=DType.float32`, `shape=[32,32]`, `device=gpu:0`, and empty explicit metadata.

It has no 4096-byte output allocation yet. The caller does `graph.output(*results)`; later `model.execute(Q,K,V)` returns a different Python list containing a real Buffer. Equal-looking result lists at graph and runtime stages contain **different kinds of objects**.

```text
caller: ops.custom returns results ↗ [PY:L129]
├─ native builder result %o ↗ [PY:L1355]
├─ _Value._from_cmlir → Value.from_mlir → TensorValue(%o) ↗ [PY:L1360] ↗ caller [PY:L109]
└─ Python results list → graph.output(*results) ↗ [PY:L68]
   └─ symbolic output recorded now; runtime Buffer returned later ↗ [PY:L82] ↗ caller [PY:L68]
```

Source links: [max/python/max/graph/ops/custom.py:L129](sources/max/python/max/graph/ops/custom.py#L129) · [max/python/max/graph/graph.py:L1355](sources/max/python/max/graph/graph.py#L1355) · [max/python/max/graph/graph.py:L1360](sources/max/python/max/graph/graph.py#L1360) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/examples/custom_ops/fused_attention.py:L68](sources/max/examples/custom_ops/fused_attention.py#L68) · [max/examples/custom_ops/fused_attention.py:L82](sources/max/examples/custom_ops/fused_attention.py#L82)

### Graph.output · Complete G1

**Before output: `(TQ,TK,TV)→()`; afterward: `(TQ,TK,TV)→(TO)`.** The Graph API flattens output leaves, bridges native types, updates both function and generator signatures, inserts `mo.output %o`, attaches engine-facing names and verifies the whole graph.

```text
MODULE1 / G1 / R0 / B0(%q:TQ,%k:TK,%v:TV)
  CH0 = chain.create()
  %o = custom(%q,%k,%v) : TO
  output %o
public interface: input0,input1,input2 → output0
```

This creates a complete **symbolic** computation; no numeric Q/K/V or target instructions were executed in these recording steps.

```text
caller: graph.output(*results) ↗ [PY:L68]
├─ flatten output structure and normalize Value wrappers ↗ [PY:L1433] ↗ caller [PY:L68]
├─ optional internal chains; native value/type handles ↗ [PY:L1449] ↗ caller [PY:L68]
├─ FunctionType(inputs,outputs); attach generator and functionType ↗ [PY:L1459] ↗ caller [PY:L68]
├─ _add_op_generated(OutputOp,mlir_values,params=[]) ↗ [PY:L1469] ↗ caller [PY:L68]
├─ argument_names/input0…; result_names/output0 ↗ [PY:L1479] ↗ caller [PY:L68]
└─ verify completed G1, then with-scope exit ↗ [PY:L1492] ↗ [PY:L1138] ↗ caller [PY:L68]
```

Source links: [max/examples/custom_ops/fused_attention.py:L68](sources/max/examples/custom_ops/fused_attention.py#L68) · [max/python/max/graph/graph.py:L1433](sources/max/python/max/graph/graph.py#L1433) · [max/python/max/graph/graph.py:L1449](sources/max/python/max/graph/graph.py#L1449) · [max/python/max/graph/graph.py:L1459](sources/max/python/max/graph/graph.py#L1459) · [max/python/max/graph/graph.py:L1469](sources/max/python/max/graph/graph.py#L1469) · [max/python/max/graph/graph.py:L1479](sources/max/python/max/graph/graph.py#L1479) · [max/python/max/graph/graph.py:L1492](sources/max/python/max/graph/graph.py#L1492) · [max/python/max/graph/graph.py:L1138](sources/max/python/max/graph/graph.py#L1138)

### B.1 · Session configuration and MAX Context

**S1 combines compiler configuration and runtime ownership.** Python deduplicates devices, appends CPU if needed, processes optional extensions, and calls the native constructor. Native code clones `sessionCompileConfig`, fills `runtimeConfig` with device specs/context references and thread/API settings, then retains an Engine::Context.

| Retained value | Concrete teaching example | Meaning |
|---|---|---|
| `devices` | `[GPU0,CPU0]` | Driver object references |
| `sessionCompileConfig` | base kernel paths; supplied extensions=[] | Settings for later compilation |
| optional extension | `/tmp/.../P1.mojoc` | Existing Mojo package path, appended if explicitly supplied |
| `runtimeConfig` | threads=0, language=python, devices/specs/ctx | Runtime setup policy |
| `runtimeContext` | EngineContext1→MAXCTX1 and CPUDevice1 | Native runtime services |
| shared MAX Context | typed service store; CPUDevice and telemetry | Shared infrastructure, not graph IR |

**Actual constructor extensions are empty.** P1 was loaded by G1; compile later picks up G1's library paths. Engine context creation can acquire a shared MAX Context or retain one supplied by configuration. This is distinct from the module-level default MLIR Context.

```text
caller: session=InferenceSession(devices=[device]) ↗ [PY:L71]
├─ Python: unique devices; ensure CPU; process optional extension objects ↗ [PY:L688] ↗ caller [PY:L71]
└─ native _InferenceSession(final_devices,extensions,num_threads or 0) ↗ [PY:L707]
   ├─ populateCompileConfig → clone config; append valid .mojoc paths ↗ [C++:L546] ↗ [C++:L615] ↗ caller [PY:L707]
   ├─ runtimeConfig.setNumThreads; retain specs/device contexts; language=python ↗ [C++:L549] ↗ caller [PY:L707]
   ├─ Engine::Context::create ↗ [C++:L169]
   │  ├─ supplied MAX Context → retain ↗ [C++:L175] ↗ caller [C++:L559]
   │  ├─ otherwise Init::getOrCreateContext with CPU runtime options ↗ [C++:L184] ↗ caller [C++:L559]
   │  └─ context wrapper stores config/MAXContext/CPUDevice reference ↗ [C++:L214] ↗ caller [C++:L189]
   └─ retain runtimeContext; stack callback; configured profiling branch ↗ [C++:L563] ↗ [C++:L569] ↗ caller [PY:L71]
```

Source links: [max/examples/custom_ops/fused_attention.py:L71](sources/max/examples/custom_ops/fused_attention.py#L71) · [max/python/max/engine/api.py:L688](sources/max/python/max/engine/api.py#L688) · [max/python/max/engine/api.py:L707](sources/max/python/max/engine/api.py#L707) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L546](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L546) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L615](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L615) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L549](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L549) · [max/lib/max_license/API/c/EngineContext.cpp:L169](sources/max/lib/max_license/API/c/EngineContext.cpp#L169) · [max/lib/max_license/API/c/EngineContext.cpp:L175](sources/max/lib/max_license/API/c/EngineContext.cpp#L175) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L559](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L559) · [max/lib/max_license/API/c/EngineContext.cpp:L184](sources/max/lib/max_license/API/c/EngineContext.cpp#L184) · [max/lib/max_license/API/c/EngineContext.cpp:L214](sources/max/lib/max_license/API/c/EngineContext.cpp#L214) · [max/lib/max_license/API/c/EngineContext.cpp:L189](sources/max/lib/max_license/API/c/EngineContext.cpp#L189) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L563](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L563) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L569](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L569)

### C.1 · MO → MAP → MOGG → MGP

**Graph compilation transforms a clone C1 into an executable plan.** It first obtains P1 paths/seeded declarations, creates the compiler clone, then runs an MLIR PassManager built by GraphCompiler.

| Stage | Concrete question answered | Attention example |
|---|---|---|
| MO: Modular Operations | What graph/type/device/custom contract? | Three f32[N,D] inputs, one output, symbol and BN/BD |
| MAP: Modular Access Primitives | What accesses/implementation boundary? | Custom routine remains opaque; supported built-ins expose views/loads/computations |
| MOGG: Modular Operation Generator Graph | What Mojo routine and arguments? | Declaration resolution; output/query/key/value/context glue |
| MGP: Modular Glue Primitives | What resource/dependency lifecycle? | Init resources, per-call buffers, state slots/chains, invoke/return |
| JIT/runtime lowering | What code resource and callable plans? | Archive, ModuleLoad, ExecuteGeneric, initFn/executeFn |

MO contracts include shape/dtype/device and registered operand/result/parameter compatibility. CSE/canonicalization can simplify eligible pure duplicates or identities; these passes do not inspect the numerical data in this custom call. MAP fusion examples are explicitly **built-in comparisons**, not claims that GraphCompiler decomposes this opaque attention kernel.

MojoRegistry indexes native declarations. MAPToMOGGState carries one operation's conversion context/mappings and emits implementation/glue calls. MGP schedules lifecycle work so repeated execute calls can reuse initialized code/resources with fresh inputs.

For the shared in-depth pass examples, open [the compiler pipeline lesson](max_graph_compiler_pipeline_examples.html). All snippets here are schematics, not captured IR dumps.

```text
caller: compiled=session.compile(graph) ↗ [PY:L74]
├─ compile_async: Graph.module, library paths, expected_weights, seed declarations ↗ [PY:L1021] ↗ [PY:L1042] ↗ caller [PY:L1104]
├─ _compile_module → native request → CPU worker compile ↗ [PY:L1292] ↗ [PY:L1294] ↗ [C++:L229] ↗ caller [PY:L1044]
└─ MOFContext::compileToBinary ↗ [C++:L377]
   ├─ clone module → C1 ↗ [C++:L381] ↗ caller [C++:L273]
   └─ cache miss: convertMOToMEF → MO::runGCPipeline ↗ [C++:L356] ↗ [C++:L638]
      ├─ gcPipelineImpl creates passes; pm.run executes them ↗ [C++:L653] ↗ [C++:L654] ↗ caller [C++:L361]
      ├─ buildMoPipeline: declarations/parameters, canonicalization, shapes/devices ↗ [C++:L219] ↗ caller [C++:L574]
      ├─ MOToMAPPass → access primitives / opaque boundary ↗ [C++:L764] ↗ caller [C++:L581]
      ├─ buildMapPipeline → eligible fusion, view motion, outlining ↗ [C++:L338] ↗ caller [C++:L587]
      ├─ MAPToMOGGPass → MojoRegistry + per-op MAPToMOGGState ↗ [C++:L57] ↗ [C++:L66] ↗ [C++:L68] ↗ caller [C++:L597]
      ├─ buildMoggPipeline → conform arguments / outline routines ↗ [C++:L370] ↗ caller [C++:L602]
      ├─ MOGGToMGPPass → lifecycle regions/buffers/chains ↗ [C++:L2406] ↗ caller [C++:L611]
      ├─ buildMgpPipeline → lifetime/memory/dependency plans + emitted Mojo ↗ [C++:L382] ↗ caller [C++:L616]
      ├─ JitCompileKernels → code resources ↗ [C++:L529] ↗ caller [C++:L618]
      └─ buildJitPipeline → ModuleLoad/ExecuteGeneric, init+execute callables ↗ [C++:L75] ↗ caller [C++:L620]
```

Source links: [max/examples/custom_ops/fused_attention.py:L74](sources/max/examples/custom_ops/fused_attention.py#L74) · [max/python/max/engine/api.py:L1021](sources/max/python/max/engine/api.py#L1021) · [max/python/max/engine/api.py:L1042](sources/max/python/max/engine/api.py#L1042) · [max/python/max/engine/api.py:L1104](sources/max/python/max/engine/api.py#L1104) · [max/python/max/engine/api.py:L1292](sources/max/python/max/engine/api.py#L1292) · [max/python/max/engine/api.py:L1294](sources/max/python/max/engine/api.py#L1294) · [max/lib/max_license/API/c/EngineContext.cpp:L229](sources/max/lib/max_license/API/c/EngineContext.cpp#L229) · [max/python/max/engine/api.py:L1044](sources/max/python/max/engine/api.py#L1044) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L377](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L381](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L381) · [max/lib/max_license/API/c/EngineContext.cpp:L273](sources/max/lib/max_license/API/c/EngineContext.cpp#L273) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356) · [GraphCompiler/lib/GraphCompiler.cpp:L638](sources/GraphCompiler/lib/GraphCompiler.cpp#L638) · [GraphCompiler/lib/GraphCompiler.cpp:L653](sources/GraphCompiler/lib/GraphCompiler.cpp#L653) · [GraphCompiler/lib/GraphCompiler.cpp:L654](sources/GraphCompiler/lib/GraphCompiler.cpp#L654) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L361](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L361) · [GraphCompiler/lib/GraphCompiler.cpp:L219](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [GraphCompiler/lib/GraphCompiler.cpp:L574](sources/GraphCompiler/lib/GraphCompiler.cpp#L574) · [GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L764](sources/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp#L764) · [GraphCompiler/lib/GraphCompiler.cpp:L581](sources/GraphCompiler/lib/GraphCompiler.cpp#L581) · [GraphCompiler/lib/GraphCompiler.cpp:L338](sources/GraphCompiler/lib/GraphCompiler.cpp#L338) · [GraphCompiler/lib/GraphCompiler.cpp:L587](sources/GraphCompiler/lib/GraphCompiler.cpp#L587) · [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L57](sources/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp#L57) · [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L66](sources/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp#L66) · [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L68](sources/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp#L68) · [GraphCompiler/lib/GraphCompiler.cpp:L597](sources/GraphCompiler/lib/GraphCompiler.cpp#L597) · [GraphCompiler/lib/GraphCompiler.cpp:L370](sources/GraphCompiler/lib/GraphCompiler.cpp#L370) · [GraphCompiler/lib/GraphCompiler.cpp:L602](sources/GraphCompiler/lib/GraphCompiler.cpp#L602) · [GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:L2406](sources/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp#L2406) · [GraphCompiler/lib/GraphCompiler.cpp:L611](sources/GraphCompiler/lib/GraphCompiler.cpp#L611) · [GraphCompiler/lib/GraphCompiler.cpp:L382](sources/GraphCompiler/lib/GraphCompiler.cpp#L382) · [GraphCompiler/lib/GraphCompiler.cpp:L616](sources/GraphCompiler/lib/GraphCompiler.cpp#L616) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [GraphCompiler/lib/GraphCompiler.cpp:L618](sources/GraphCompiler/lib/GraphCompiler.cpp#L618) · [GraphCompiler/lib/GraphCompiler.cpp:L75](sources/GraphCompiler/lib/GraphCompiler.cpp#L75) · [GraphCompiler/lib/GraphCompiler.cpp:L620](sources/GraphCompiler/lib/GraphCompiler.cpp#L620)

### C.2 · Mojo specialization, LLVM and target backend

**The graph compiler invokes the Mojo compiler library for JIT routines.** JIT means code is compiled when the application needs a compiled model, instead of being fully built ahead of time into the Python example target. `session.compile(G1)` is the example JIT path; subsequent execute calls reuse M1's initialized code.

Library generation first creates parametric KGEN generators. The host ObjectCompiler selects its CPU backend **before** routine-cache lookup. A cache miss elaborates the generators. During elaboration, device offload requests are grouped and sliced into a separate target module with its own ObjectCompiler/backend. Device lowering/emission returns a GPU payload; host lowering then builds the native archive containing the host routines and payload references.

Both ordinary CPU and NVPTX/AMDGPU paths use LLVM-dialect lowering followed by LLVM IR translation. These are two distinct representations: MLIR `llvm.*` operations, then `llvm::Module`. Architecture/features configure target generation; the triple selects the backend implementation. Host and device modules do not share a single CPU/GPU machine-code target.

| Target illustration | Triple family | Code ownership |
|---|---|---|
| Host CPU | host triple (e.g. x86_64 Linux) | Python/MAX/driver and generated host routines |
| NVIDIA GPU | nvptx64-nvidia-cuda, example sm_80 | Separate device target payload |
| AMD GPU | amdgcn-amd-amdhsa, example gfx942 | Separate device target payload |

Architectures are example values, not detected hardware. The code-cache selector reuses native routine resources. An existing .mojoc path skips source precompilation; a deterministic source-package output path alone is not a cache-hit check. **No per-kernel `mojo build` or standalone `mlir-opt` shell sequence is needed in this JIT route.**

```text
caller: JitCompileKernelsPass compiles emitted routines ↗ [C++:L529]
├─ target options: host triple plus accelerator architecture ↗ [C++:L154] ↗ caller [C++:L534]
├─ parseMojoText with P1 imports; stamp target info ↗ [C++:L547] ↗ [C++:L556] ↗ caller [C++:L547]
├─ KGENCompiler → runLITToKGEN/library generation ↗ [C++:L566] ↗ [C++:L567] ↗ caller [C++:L567]
├─ host ObjectCompiler::create(isJit=true) → select host backend now ↗ [C++:L146] ↗ caller [C++:L575]
├─ code-cache lookup; hits later attach existing archive resources ↗ [C++:L248] ↗ [C++:L507] ↗ caller [C++:L586]
└─ cache miss → handleCacheMisses ↗ [C++:L436]
   ├─ KGENCompiler::runElaborationPipeline ↗ [C++:L1140] ↗ caller [C++:L440]
   │  └─ discovered GPU offloads → group/slice by TargetInfo ↗ [C++:L523] ↗ [C++:L610]
   │     ├─ device triple/arch/features options; GPU ObjectCompiler/backend ↗ [C++:L535] ↗ [C++:L146] ↗ caller [C++:L720]
   │     ├─ device KGEN → LLVM dialect → LLVM IR ↗ [C++:L835] ↗ [C++:L807] ↗ caller [C++:L831]
   │     └─ emit device kernel payloads keyed by kernel ID ↗ [C++:L831] ↗ caller [C++:L831]
   ├─ host emitArchive → LLVM dialect lowering + LLVM IR translation ↗ [C++:L1023] ↗ [C++:L835] ↗ [C++:L807] ↗ caller [C++:L457]
   ├─ attach host archive and callable symbol names to init resource ↗ [C++:L493] ↗ caller [C++:L493]
   └─ later LowerCompiledRoutines → ModuleLoad / ExecuteGeneric ↗ [C++:L102] ↗ caller [C++:L76]
```

Source links: [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [GraphCompiler/lib/MOGGDialect/Support/MojoStubCompiler.cpp:L154](sources/GraphCompiler/lib/MOGGDialect/Support/MojoStubCompiler.cpp#L154) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L534](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L534) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L547](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L547) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L556](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L556) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L566](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L566) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L567](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L567) · [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L146](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L146) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L575](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L575) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L248](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L248) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L507](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L507) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L586](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L586) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L436](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L436) · [Mojo/lib/Compiler/KGENCompiler.cpp:L1140](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L440](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L440) · [Mojo/lib/Compiler/KGENCompiler.cpp:L523](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L523) · [Mojo/lib/Compiler/KGENCompiler.cpp:L610](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L610) · [Mojo/lib/Compiler/KGENCompiler.cpp:L535](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L535) · [Mojo/lib/Compiler/KGENCompiler.cpp:L720](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L720) · [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L835](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L835) · [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L807](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L807) · [Mojo/lib/Compiler/KGENCompiler.cpp:L831](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L831) · [Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp:L1023](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L1023) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L457](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L457) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L493](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L493) · [GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp:L102](sources/GraphCompiler/lib/MOGGDialect/Transforms/LowerCompiledRoutines.cpp#L102) · [GraphCompiler/lib/GraphCompiler.cpp:L76](sources/GraphCompiler/lib/GraphCompiler.cpp#L76)

### C.3 · Compiled artifact CM1 and MEF

**CM1 is a compiled artifact, not the initialized executable model.** It retains the native compiled handle plus expected-weight metadata and graph names. This custom example has expected_weights={} because its three matrices are per-call inputs.

MEF packages the model interface, code/resources and callable plans. .mojoc instead packages parametric Mojo declarations. Synchronous compile waits for its native AsyncValue and raises on error. Init must still resolve contexts, resource symbols, runtime state and initialization dependencies.

```text
caller: compiled=session.compile(graph) ↗ [PY:L74]
├─ convertMOToMEF → executable container ↗ [C++:L356] ↗ caller [C++:L414]
├─ CompiledModels opens MEF bytes and enumerates models ↗ [C++:L29] ↗ [C++:L901] ↗ caller [C++:L100]
├─ CPU worker resolves CompiledModels AsyncValue ↗ [C++:L247] ↗ caller [C++:L229]
└─ Python handle.wait; exception gate; return CM1 ↗ [PY:L1112] ↗ [PY:L1113] ↗ [PY:L1115]
   └─ next call is session.init(CM1), before model.execute ↗ [PY:L75] ↗ caller [PY:L74]
```

Source links: [max/examples/custom_ops/fused_attention.py:L74](sources/max/examples/custom_ops/fused_attention.py#L74) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356) · [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L414](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L414) · [max/lib/max_license/API/c/CompiledModels.cpp:L29](sources/max/lib/max_license/API/c/CompiledModels.cpp#L29) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L901](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L901) · [max/lib/max_license/API/c/CompiledModels.cpp:L100](sources/max/lib/max_license/API/c/CompiledModels.cpp#L100) · [max/lib/max_license/API/c/EngineContext.cpp:L247](sources/max/lib/max_license/API/c/EngineContext.cpp#L247) · [max/lib/max_license/API/c/EngineContext.cpp:L229](sources/max/lib/max_license/API/c/EngineContext.cpp#L229) · [max/python/max/engine/api.py:L1112](sources/max/python/max/engine/api.py#L1112) · [max/python/max/engine/api.py:L1113](sources/max/python/max/engine/api.py#L1113) · [max/python/max/engine/api.py:L1115](sources/max/python/max/engine/api.py#L1115) · [max/examples/custom_ops/fused_attention.py:L75](sources/max/examples/custom_ops/fused_attention.py#L75)

### D.1 · Model setup, host code and runtime state

**Model setup runs a compiled init plan and retains its resource results.** The CPU worker supplies AsyncRT's current CPUDevice in thread-local state; setup primitives such as `MGP_RT_BufferBorrowV2` require it. GPU targeting does not remove this host requirement.

`Model::setup` maps DeviceRefs, prepares stream capacity, invokes initFn and checks immediate errors. The first returned values are the initialization chain, MGP context and weight reference; remaining values become extraArgs. `ExecutableModel` retains these along with the model description and CPUDevice reference. **The worker then waits for all setup results before publishing M1.** Deferred setup errors also stop initialization.

`CompiledKernelsContext::load` hashes already-compiled archive bytes, loads them once in an execution-engine dylib and resolves named host symbols. **It loads host callable pointers; GPU handles/tensor pointers are different objects.** Device module/function/launch work is reached by later host execution. Open [the native loader lesson](max_compiled_kernels_context_load_and_driver.html) for archive linking and driver details.

```text
caller: model=session.init(compiled) ↗ [PY:L75]
├─ init → init_all: create real empty registry; validate bindings ↗ [PY:L1151] ↗ [PY:L1202] ↗ [PY:L1209] ↗ caller [PY:L75]
├─ native loadAll: managed weights; compiled-descriptor await; one model ↗ [C++:L293] ↗ [C++:L300] ↗ [C++:L309] ↗ caller [PY:L1244]
└─ Engine::Context::initModel ↗ [C++:L288]
   ├─ allocate result AsyncValue; addTask(cpuDevice,...) ↗ [C++:L297]
   │  └─ worker provides current CPUDevice TLS for setup primitives ↗ [C++:L294] ↗ caller [C++:L297]
   └─ createExecutableModel → Model::setup ↗ [C++:L79] ↗ [C++:L184]
      ├─ borrowed weights reference + DeviceRef mapping ↗ [C++:L193] ↗ [C++:L197] ↗ caller [C++:L86]
      ├─ ensure device stream count, default2 ↗ [C++:L212] ↗ caller [C++:L86]
      ├─ initFn.invoke ↗ [C++:L220]
      │  └─ ModuleLoad → CompiledKernelsContext::load(archive,symbols) ↗ [C++:L55] ↗ [C++:L85] ↗ caller [C++:L220]
      │     ├─ archive hash → addIfAbsent<StaticArchiveLayer> ↗ [C++:L91] ↗ [C++:L160] ↗ caller [C++:L160]
      │     └─ engine.lookup → resident host pointers; addKernel ↗ [C++:L166] ↗ [C++:L169] ↗ caller [C++:L166]
      ├─ construct ExecutableModel(chain,MGPContext,weights,extraArgs) ↗ [C++:L237] ↗ caller [C++:L86]
      ├─ waitForModelSetup → await fixed/extra results; propagate errors ↗ [C++:L388] ↗ caller [C++:L91]
      └─ worker publishes ready executable → retained ModelHandle → Python M1 ↗ [C++:L304] ↗ [C++:L312] ↗ [PY:L1158] ↗ caller [PY:L1244]
```

Source links: [max/examples/custom_ops/fused_attention.py:L75](sources/max/examples/custom_ops/fused_attention.py#L75) · [max/python/max/engine/api.py:L1151](sources/max/python/max/engine/api.py#L1151) · [max/python/max/engine/api.py:L1202](sources/max/python/max/engine/api.py#L1202) · [max/python/max/engine/api.py:L1209](sources/max/python/max/engine/api.py#L1209) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L293](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L293) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L300](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L300) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L309](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L309) · [max/python/max/engine/api.py:L1244](sources/max/python/max/engine/api.py#L1244) · [max/lib/max_license/API/c/EngineContext.cpp:L288](sources/max/lib/max_license/API/c/EngineContext.cpp#L288) · [max/lib/max_license/API/c/EngineContext.cpp:L297](sources/max/lib/max_license/API/c/EngineContext.cpp#L297) · [max/lib/max_license/API/c/EngineContext.cpp:L294](sources/max/lib/max_license/API/c/EngineContext.cpp#L294) · [max/lib/max_license/API/c/EngineContext.cpp:L79](sources/max/lib/max_license/API/c/EngineContext.cpp#L79) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L184](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L184) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L193](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L193) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L197](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L197) · [max/lib/max_license/API/c/EngineContext.cpp:L86](sources/max/lib/max_license/API/c/EngineContext.cpp#L86) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L212](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L212) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L220](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L220) · [GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp:L55](sources/GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp#L55) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L85](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L85) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L91](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L91) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L160](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L160) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L166](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L166) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L169](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L169) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L237](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L237) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L388](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L388) · [max/lib/max_license/API/c/EngineContext.cpp:L91](sources/max/lib/max_license/API/c/EngineContext.cpp#L91) · [max/lib/max_license/API/c/EngineContext.cpp:L304](sources/max/lib/max_license/API/c/EngineContext.cpp#L304) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L312](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L312) · [max/python/max/engine/api.py:L1158](sources/max/python/max/engine/api.py#L1158)

### D.2 · Initialization and readiness gates

**Three checks serve different purposes:** weight binding validates persistent parameters; setup completion produces the executable owner; initialization-chain readiness validates that resource work has finished successfully.

Empty bindings are valid for this custom graph because Q/K/V are invocation inputs. `Model::setup` first constructs an owner of init results; **createExecutableModel then waits for fixed and extra results and checks errors before the worker publishes the executable.** Native load awaits that task before Python init returns M1. A later executeFromDriver await acquires this already-checked executable; the initial resource wait is not deferred to execute. The failure selector shows a deferred setup error, not a failed GPU run.

```text
caller: session.init(CM1), then model.execute(Q,K,V) ↗ [PY:L75] ↗ [PY:L82]
├─ init_all checks weight bindings and loads models ↗ [PY:L1209] ↗ [PY:L1244] ↗ caller [PY:L1151]
├─ worker constructs ExecutableModel from init results ↗ [C++:L237] ↗ caller [C++:L86]
├─ waitForModelSetup awaits/checks chain,context,weights,extraArgs ↗ [C++:L388] ↗ [C++:L390] ↗ caller [C++:L91]
├─ CPU task publishes checked executable or diagnostic ↗ [C++:L302] ↗ [C++:L304] ↗ caller [C++:L299]
├─ native load awaits init result before returning M1 ↗ [C++:L282] ↗ caller [C++:L311]
├─ later executeFromDriver awaits loaded executable ↗ [C++:L204] ↗ caller [C++:L530]
└─ ready executable invokes executeFn with retained state + fresh input refs ↗ [C++:L384] ↗ caller [C++:L212]
```

Source links: [max/examples/custom_ops/fused_attention.py:L75](sources/max/examples/custom_ops/fused_attention.py#L75) · [max/examples/custom_ops/fused_attention.py:L82](sources/max/examples/custom_ops/fused_attention.py#L82) · [max/python/max/engine/api.py:L1209](sources/max/python/max/engine/api.py#L1209) · [max/python/max/engine/api.py:L1244](sources/max/python/max/engine/api.py#L1244) · [max/python/max/engine/api.py:L1151](sources/max/python/max/engine/api.py#L1151) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L237](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L237) · [max/lib/max_license/API/c/EngineContext.cpp:L86](sources/max/lib/max_license/API/c/EngineContext.cpp#L86) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L388](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L388) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L390](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L390) · [max/lib/max_license/API/c/EngineContext.cpp:L91](sources/max/lib/max_license/API/c/EngineContext.cpp#L91) · [max/lib/max_license/API/c/EngineContext.cpp:L302](sources/max/lib/max_license/API/c/EngineContext.cpp#L302) · [max/lib/max_license/API/c/EngineContext.cpp:L304](sources/max/lib/max_license/API/c/EngineContext.cpp#L304) · [max/lib/max_license/API/c/EngineContext.cpp:L299](sources/max/lib/max_license/API/c/EngineContext.cpp#L299) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L282](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L282) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L311](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L311) · [max/lib/max_license/API/c/ModelExecution.cpp:L204](sources/max/lib/max_license/API/c/ModelExecution.cpp#L204) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L530](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L530) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L384](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L384) · [max/lib/max_license/API/c/ModelExecution.cpp:L212](sources/max/lib/max_license/API/c/ModelExecution.cpp#L212)

### E · Buffers, execution and output

**Graph q/k/v were symbolic. Execute BQ1/BK1/BV1 hold actual float32 elements.** The script creates seeded random NumPy matrices, wraps them as Buffers and explicitly places them with `.to(device)`.

Python/native validation checks contiguity, type/count/static dimensions and devices. `prepareInput` retains original storage in runtime Tensor AsyncValues; the compiled callable receives implicit state/dependency arguments plus those input refs. Host code dispatches the specialized CPU/GPU implementation, runtime checks completion, and result extraction drops the internal chain.

Actual GPU shapes are [32,32] with 4096 bytes each; CPU [8,8] with 256 bytes. Addresses pQ/pK/pV and handles are examples. The returned value is **a list `[Buffer]`**, not a TensorValue or printed numerical matrix. The script has no explicit final CPU transfer or parity assertion.

```text
caller: output=model.execute(Q,K,V) ↗ [PY:L82]
├─ Buffer.from_numpy(...).to(device) before invocation ↗ [PY:L78] ↗ [PY:L79] ↗ [PY:L80] ↗ caller [PY:L82]
├─ _Model_execute: contiguity, supported types, ordered Buffer refs ↗ [PY:L118] ↗ caller [PY:L82]
└─ _execute_device_tensors → native ModelHandle ↗ [C++:L565]
   ├─ input-spec/device validation; prepareInput(storage,spec) ↗ [C++:L461] ↗ [C++:L484] ↗ caller [PY:L145]
   │  └─ retained TensorBufferRef shares original storage ↗ [C++:L121] ↗ caller [C++:L484]
   ├─ release GIL; executeFromDriver awaits M1 ↗ [C++:L529] ↗ [C++:L204] ↗ caller [C++:L518]
   ├─ ExecutableModel::execute assembles implicit+user arguments ↗ [C++:L373]
   │  └─ executeFn.invoke → host routine → GPU enqueue or CPU function ↗ [C++:L384] ↗ [DOC:L101] ↗ [DOC:L403] ↗ caller [C++:L212]
   ├─ await results/errors; drop internal first chain ↗ [C++:L137] ↗ [C++:L159] ↗ caller [C++:L213]
   └─ extractPyBufferResults → [BO1]; caller print(output) ↗ [C++:L538] ↗ [PY:L83] ↗ caller [PY:L82]
```

Source links: [max/examples/custom_ops/fused_attention.py:L82](sources/max/examples/custom_ops/fused_attention.py#L82) · [max/examples/custom_ops/fused_attention.py:L78](sources/max/examples/custom_ops/fused_attention.py#L78) · [max/examples/custom_ops/fused_attention.py:L79](sources/max/examples/custom_ops/fused_attention.py#L79) · [max/examples/custom_ops/fused_attention.py:L80](sources/max/examples/custom_ops/fused_attention.py#L80) · [max/python/max/engine/api.py:L118](sources/max/python/max/engine/api.py#L118) · [max/python/max/_core/max_license/modules/engine.cpp:L565](sources/max/python/max/_core/max_license/modules/engine.cpp#L565) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L461](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L461) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L484](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L484) · [max/python/max/engine/api.py:L145](sources/max/python/max/engine/api.py#L145) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L121](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L121) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L529](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L529) · [max/lib/max_license/API/c/ModelExecution.cpp:L204](sources/max/lib/max_license/API/c/ModelExecution.cpp#L204) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L518](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L518) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L373](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L373) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L384](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L384) · [max/examples/custom_ops/kernels/fused_attention.mojo:L101](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) · [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/lib/max_license/API/c/ModelExecution.cpp:L212](sources/max/lib/max_license/API/c/ModelExecution.cpp#L212) · [max/lib/max_license/API/c/ModelExecution.cpp:L137](sources/max/lib/max_license/API/c/ModelExecution.cpp#L137) · [max/lib/max_license/API/c/ModelExecution.cpp:L159](sources/max/lib/max_license/API/c/ModelExecution.cpp#L159) · [max/lib/max_license/API/c/ModelExecution.cpp:L213](sources/max/lib/max_license/API/c/ModelExecution.cpp#L213) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L538](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L538) · [max/examples/custom_ops/fused_attention.py:L83](sources/max/examples/custom_ops/fused_attention.py#L83)

### Kernel · Inferred parameters and CPU/GPU dispatch

**The execute interface has four tensor roles, but only three Graph inputs.** Output is a mutable destination corresponding to the Graph result. Query/key/value are the three input operands. ctx is supplied by native runtime glue.

dtype/rank are inferred; BN/BD come from custom-op parameters; the selected graph target determines cpu/gpu specialization. The `comptime` rank check and branch are compile-time operations. Views retain the same storage with layout/origin metadata.

CPU calls its helper directly with BN=BD=4. GPU uses BN=16,BD=8 and a host DeviceContext wrapper that submits a separate device kernel. **A GPU graph still needs CPU orchestration and runtime services.**

```text
caller: generated routine reaches FusedAttention.execute ↗ [DOC:L74]
├─ @register name; dtype/rank inferred; explicit BN/BD; target specialized ↗ [DOC:L67] ↗ [DOC:L75] ↗ caller [PY:L60]
├─ comptime assert rank==2 ↗ [DOC:L88] ↗ caller [DOC:L74]
├─ query/key/value/output → TileTensor views over existing storage ↗ [DOC:L91] ↗ [DOC:L93] ↗ [DOC:L95] ↗ [DOC:L97] ↗ caller [DOC:L74]
└─ comptime target branch ↗ [DOC:L99]
   ├─ CPU → fused_attention_cpu[4,4](Q,K,V,O) ↗ [DOC:L101] ↗ caller [DOC:L101]
   └─ GPU → fused_attention_gpu[16,8](ctx,Q,K,V,O) ↗ [DOC:L105] ↗ caller [DOC:L105]
```

Source links: [max/examples/custom_ops/kernels/fused_attention.mojo:L74](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L74) · [max/examples/custom_ops/kernels/fused_attention.mojo:L67](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L67) · [max/examples/custom_ops/kernels/fused_attention.mojo:L75](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L75) · [max/examples/custom_ops/fused_attention.py:L60](sources/max/examples/custom_ops/fused_attention.py#L60) · [max/examples/custom_ops/kernels/fused_attention.mojo:L88](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L88) · [max/examples/custom_ops/kernels/fused_attention.mojo:L91](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L91) · [max/examples/custom_ops/kernels/fused_attention.mojo:L93](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L93) · [max/examples/custom_ops/kernels/fused_attention.mojo:L95](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L95) · [max/examples/custom_ops/kernels/fused_attention.mojo:L97](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L97) · [max/examples/custom_ops/kernels/fused_attention.mojo:L99](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L99) · [max/examples/custom_ops/kernels/fused_attention.mojo:L101](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) · [max/examples/custom_ops/kernels/fused_attention.mojo:L105](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L105)

### Launch · Function handles, argument slots and output tiles

**grid.x=4 partitions output columns; grid.y=2 partitions query rows.** Each block writes a [16,8] O tile, computes [16,8] scores for four successive key tiles, and sees all 32 keys. x does not partition the softmax denominator.

| Object | Example | Kind |
|---|---|---|
| Host routine | HOST_FN1 | CPU callable pointer |
| Device function | DEVICE_FN1 in MODULE_GPU1 | CUDA/HIP driver handle |
| Q buffer | pQ,4096 bytes | Device tensor storage address |
| Argument slot | encoded TileTensor pointer/layout | Host ABI slot submitted to driver |
| Stream | CUstream/HIP stream | Device work queue handle |

The source route checks dimensions, obtains a checked function, encodes DevicePassable arguments, then calls native runtime enqueue. NVIDIA uses `cuLaunchKernelEx`; AMD uses `hipModuleLaunchKernel`. Select block(1,1) to see rows16–31/columns8–15 and how K/V tile views change. Physical scheduling and exact residency/timing remain unmeasured.

```text
caller: ctx.enqueue_function[kernel_func](Q,K,V,O,grid_dim=(4,2),block_dim=32) ↗ [DOC:L403]
├─ specialize kernel function with dtypes/layouts/BN/BD ↗ [DOC:L391] ↗ caller [DOC:L403]
├─ DeviceContext enqueue mixin: check dimensions and compile_function ↗ [DOC:L162] ↗ [DOC:L171] ↗ caller [DOC:L403]
│  └─ _load_function → native loadFunctionWithDebugName ↗ [DOC:L2994] ↗ [C++:L504] ↗ caller [DOC:L171]
├─ checked argument pack ↗ [DOC:L3503]
│  ├─ validate target argument types/offsets ↗ [DOC:L3536] ↗ caller [DOC:L179]
│  ├─ _to_device_type writes aligned slots ↗ [DOC:L3639] ↗ caller [DOC:L179]
│  └─ _enqueue_packed_checked → ctx.enqueue ↗ [DOC:L2947] ↗ caller [DOC:L3664]
├─ native enqueueFunctionDirect ↗ [C++:L542]
│  ├─ NVIDIA → CUDADeviceFunction::cudaCallKernelDirect → cuLaunchKernelEx ↗ [C++:L1573] ↗ [C++:L1610] ↗ caller [C++:L554]
│  └─ AMD → HIPDeviceFunction::hipCallKernelDirect → hipModuleLaunchKernel ↗ [C++:L1365] ↗ [C++:L1382] ↗ caller [C++:L554]
└─ block(y,x): Q rows[y*16:(y+1)*16]; O cols[x*8:(x+1)*8] ↗ [DOC:L333]
   ├─ four key tiles: all K rows; matching V rows and owned columns ↗ [DOC:L344] ↗ caller [DOC:L343]
   └─ store owned O tile after the loop ↗ [DOC:L378] ↗ caller [DOC:L403]
```

Source links: [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/examples/custom_ops/kernels/fused_attention.mojo:L391](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L391) · [max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo:L162](sources/max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo#L162) · [max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo:L171](sources/max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo#L171) · [max/mojo/max/gpu/host/_legacy/device_context.mojo:L2994](sources/max/mojo/max/gpu/host/_legacy/device_context.mojo#L2994) · [MLRT/lib/Driver/DeviceContext/MojoBindings.cpp:L504](sources/MLRT/lib/Driver/DeviceContext/MojoBindings.cpp#L504) · [max/mojo/max/gpu/host/_legacy/device_context.mojo:L3503](sources/max/mojo/max/gpu/host/_legacy/device_context.mojo#L3503) · [max/mojo/max/gpu/host/_legacy/device_context.mojo:L3536](sources/max/mojo/max/gpu/host/_legacy/device_context.mojo#L3536) · [max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo:L179](sources/max/mojo/max/gpu/host/_legacy/_device_context_extras.mojo#L179) · [max/mojo/max/gpu/host/_legacy/device_context.mojo:L3639](sources/max/mojo/max/gpu/host/_legacy/device_context.mojo#L3639) · [max/mojo/max/gpu/host/_legacy/device_context.mojo:L2947](sources/max/mojo/max/gpu/host/_legacy/device_context.mojo#L2947) · [max/mojo/max/gpu/host/_legacy/device_context.mojo:L3664](sources/max/mojo/max/gpu/host/_legacy/device_context.mojo#L3664) · [MLRT/lib/Driver/DeviceContext/MojoBindings.cpp:L542](sources/MLRT/lib/Driver/DeviceContext/MojoBindings.cpp#L542) · [MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp:L1573](sources/MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp#L1573) · [MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp:L1610](sources/MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.cpp#L1610) · [MLRT/lib/Driver/DeviceContext/MojoBindings.cpp:L554](sources/MLRT/lib/Driver/DeviceContext/MojoBindings.cpp#L554) · [MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp:L1365](sources/MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp#L1365) · [MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp:L1382](sources/MLRT/lib/Driver/DeviceContext/HIP/HIPDeviceContext.cpp#L1382) · [max/examples/custom_ops/kernels/fused_attention.mojo:L333](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) · [max/examples/custom_ops/kernels/fused_attention.mojo:L344](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L344) · [max/examples/custom_ops/kernels/fused_attention.mojo:L343](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L343) · [max/examples/custom_ops/kernels/fused_attention.mojo:L378](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L378)

### MMA · Operand fragments, accumulation and barrier

**QKᵀ has four BK=8 reduction slices because D=32. PV has one because the current key tile has eight rows.** The TensorCore descriptor determines fragment layout/load/MMA/store operations; a fragment is the lane-distributed representation of part of a matrix.

The animation follows one illustrative output cell: Q values=1 across features and key groups=1,2,0.5,−1 yield partial sums 8,16,4,−8 and final score20. This is a teaching input, not the seeded random caller matrix. The PV selector uses eight weights0.125 and value components1 to produce1.

c_reg is explicitly .LOCAL fragment storage. out_sram is explicitly .SHARED staging. After storing fragments, a block barrier precedes the result copy. These declarations **do not establish measured register allocation, spills or physical timing**. The source does not implement FA3-style asynchronous overlap here.

```text
caller: S=matmul['gpu',transpose_b=True](Q_tile,K_tile) ↗ [DOC:L346]
├─ M/N/K; out_sram .SHARED; BK=8 divisibility check ↗ [DOC:L264] ↗ [DOC:L268] ↗ caller [DOC:L346]
├─ TensorCore descriptor; c_reg .LOCAL fragment layout filled0 ↗ [DOC:L276] ↗ [DOC:L278] ↗ caller [DOC:L346]
├─ rhs_typed view rebind for compatible load dtype ↗ [DOC:L284] ↗ caller [DOC:L346]
├─ for k_i in K//BK ↗ [DOC:L295]
│  ├─ load_a current lhs tile ↗ [DOC:L296] ↗ caller [DOC:L295]
│  ├─ load_b; transposed K load uses [N,BK] orientation ↗ [DOC:L298] ↗ [DOC:L301] ↗ caller [DOC:L295]
│  ├─ d_reg=mma_op(a_reg,b_reg,c_reg) ↗ [DOC:L305] ↗ caller [DOC:L295]
│  └─ c_reg.copy_from(d_reg) ↗ [DOC:L306] ↗ caller [DOC:L295]
├─ store_d(out_sram,c_reg); barrier(); res.copy_from(out_sram) ↗ [DOC:L307] ↗ [DOC:L309] ↗ [DOC:L310] ↗ caller [DOC:L346]
└─ later PV=matmul['gpu'](E,V_tile), with K=8 ↗ [DOC:L374] ↗ caller [DOC:L374]
```

Source links: [max/examples/custom_ops/kernels/fused_attention.mojo:L346](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L346) · [max/examples/custom_ops/kernels/fused_attention.mojo:L264](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L264) · [max/examples/custom_ops/kernels/fused_attention.mojo:L268](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L268) · [max/examples/custom_ops/kernels/fused_attention.mojo:L276](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L276) · [max/examples/custom_ops/kernels/fused_attention.mojo:L278](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L278) · [max/examples/custom_ops/kernels/fused_attention.mojo:L284](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L284) · [max/examples/custom_ops/kernels/fused_attention.mojo:L295](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L295) · [max/examples/custom_ops/kernels/fused_attention.mojo:L296](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L296) · [max/examples/custom_ops/kernels/fused_attention.mojo:L298](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L298) · [max/examples/custom_ops/kernels/fused_attention.mojo:L301](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L301) · [max/examples/custom_ops/kernels/fused_attention.mojo:L305](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L305) · [max/examples/custom_ops/kernels/fused_attention.mojo:L306](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L306) · [max/examples/custom_ops/kernels/fused_attention.mojo:L307](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L307) · [max/examples/custom_ops/kernels/fused_attention.mojo:L309](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L309) · [max/examples/custom_ops/kernels/fused_attention.mojo:L310](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L310) · [max/examples/custom_ops/kernels/fused_attention.mojo:L374](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L374)

### Storage · Full tensors, temporary tiles and retained state

**Global tensor storage, logical views, retained row state and temporary arithmetic tiles have different lifetimes.** Q/K/V/O are full [32,32] buffers. Each GPU block views Q[16,32], current K[8,32] and matching V[8,8], retaining m/l[16] and O_i[16,8].

S/E/PV are [16,8] current-iteration tiles. E is reused as shifted scores, exponentials and normalized weights. Earlier keys survive through m/l/O_i, so the fused route has no persistent full N×N score/weight allocation.

**Source address spaces:** matmul c_reg=.LOCAL and out_sram=.SHARED. The general `stack_allocation` helper defaults to .GENERIC; m/l/O_i/E/PV logical stack arrays do not automatically prove hardware-register residency or absence of spills. Logical byte counts illustrate tensor size, not measured GPU memory/resource utilization.

```text
caller: fused_attention_kernel receives Q/K/V/O views ↗ [MOJO:L313]
├─ Q_tile selected once; initialize m₁,l₁,O_i ↗ [DOC:L333] ↗ [DOC:L335] ↗ caller [DOC:L403]
│  └─ stack_allocation default address_space=.GENERIC ↗ [DOC:L3942] ↗ caller [DOC:L335]
├─ four key iterations with BN_1=8 ↗ [DOC:L341] ↗ [DOC:L343]
│  ├─ K[8,32],V[8,8] views ↗ [DOC:L344] ↗ caller [DOC:L343]
│  ├─ S=QKᵀ[16,8] ↗ [DOC:L346] ↗ caller [DOC:L343]
│  ├─ E[16,8]: score copy → exp → normalized weights ↗ [DOC:L358] ↗ [DOC:L366] ↗ caller [DOC:L343]
│  ├─ temporary PV[16,8] → O_i; retain m/l ↗ [DOC:L374] ↗ [DOC:L376] ↗ caller [DOC:L343]
│  └─ only m/l/O_i survive as the next iteration's recurrence state ↗ [DOC:L377] ↗ caller [DOC:L343]
└─ final O tile copy after loop ↗ [DOC:L378] ↗ caller [DOC:L403]
```

Source links: [max/examples/custom_ops/kernels/fused_attention.mojo:L313](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L313) · [max/examples/custom_ops/kernels/fused_attention.mojo:L333](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) · [max/examples/custom_ops/kernels/fused_attention.mojo:L335](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L335) · [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/kernels/src/layout/tile_tensor.mojo:L3942](sources/max/kernels/src/layout/tile_tensor.mojo#L3942) · [max/examples/custom_ops/kernels/fused_attention.mojo:L341](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L341) · [max/examples/custom_ops/kernels/fused_attention.mojo:L343](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L343) · [max/examples/custom_ops/kernels/fused_attention.mojo:L344](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L344) · [max/examples/custom_ops/kernels/fused_attention.mojo:L346](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L346) · [max/examples/custom_ops/kernels/fused_attention.mojo:L358](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L358) · [max/examples/custom_ops/kernels/fused_attention.mojo:L366](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L366) · [max/examples/custom_ops/kernels/fused_attention.mojo:L374](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L374) · [max/examples/custom_ops/kernels/fused_attention.mojo:L376](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L376) · [max/examples/custom_ops/kernels/fused_attention.mojo:L377](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L377) · [max/examples/custom_ops/kernels/fused_attention.mojo:L378](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L378)

### Online softmax · Every recurrence operation

**Scores decide weights; values decide the content being mixed.** For query0=[1,0], K rows=[1,0],[0,1],[3,0],[2,1] yield scores1,0,3,2. V rows=[1,0],[0,2],[2,1],[1,3]. Query1=[0,1] yields scores0,1,0,1 and demonstrates an unchanged maximum.

```text
new_m = max(old_m, row_max(S))
alpha = exp(old_m-new_m)
E = exp(S-new_m)
new_l = alpha*old_l + row_sum(E)
P_tile = E/new_l
new_O = old_O * alpha*old_l/new_l + P_tile @ V_tile
```

**Why rescale?** Old_O is already normalized by old_l on the old maximum scale. Multiplying by old_l recovers its numerator; dividing by new_l and multiplying by alpha moves it to the new scale. Current PV uses that same new denominator. Adding them preserves the contribution of every earlier key without keeping its scores.

The animation follows the source's actual copy/subtract/exp/multiply/divide order, including l₂ and m₁_delta scratch. It hides unavailable values until computed. Pink shows prior-key mass/output; green shows current-key mass/output. Softmax reduces across keys **per query**, not across query rows. The final check is browser arithmetic, not a MAX/GPU run. Mathematical −∞ represents the source's finite MIN sentinel.

```text
caller: fused_attention_kernel key loop ↗ [DOC:L343]
├─ Q tile; initialize m₁=MIN,l₁=0,O_i=0 ↗ [DOC:L333] ↗ [DOC:L335] ↗ caller [DOC:L403]
├─ current K/V tile; S=matmul(Q_tile,K_tileᵀ) ↗ [DOC:L344] ↗ [DOC:L346] ↗ caller [DOC:L343]
├─ m₂=row_max(S); m₂=max(m₂,m₁) ↗ [DOC:L348] ↗ [DOC:L349] ↗ caller [DOC:L343]
├─ denominator scratch l₂ ↗ [DOC:L352]
│  ├─ copy m₁; subtract m₂; exp; multiply old l₁ ↗ [DOC:L353] ↗ [DOC:L354] ↗ [DOC:L355] ↗ [DOC:L356] ↗ caller [DOC:L343]
│  ├─ E=copy(S); E-=m₂; E.exp() ↗ [DOC:L358] ↗ [DOC:L359] ↗ [DOC:L360] ↗ caller [DOC:L343]
│  └─ row_sum(E); l₂+=sum → cumulative denominator ↗ [DOC:L362] ↗ [DOC:L363] ↗ caller [DOC:L343]
├─ E/=l₂ → current normalized weights ↗ [DOC:L366] ↗ caller [DOC:L343]
├─ rescale old O_i: *=old l₁; /=new l₂ ↗ [DOC:L367] ↗ [DOC:L368] ↗ caller [DOC:L343]
├─ m₁_delta=copy(m₁); -=m₂; exp; O_i*=delta ↗ [DOC:L370] ↗ [DOC:L371] ↗ [DOC:L372] ↗ [DOC:L373] ↗ caller [DOC:L343]
├─ PV=matmul(E,V_tile); O_i+=PV ↗ [DOC:L374] ↗ [DOC:L375] ↗ caller [DOC:L343]
├─ retain m₁←m₂ and l₁←l₂ ↗ [DOC:L376] ↗ [DOC:L377] ↗ caller [DOC:L343]
└─ store final O_i after every key tile has contributed ↗ [DOC:L378] ↗ caller [DOC:L403]
```

Source links: [max/examples/custom_ops/kernels/fused_attention.mojo:L343](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L343) · [max/examples/custom_ops/kernels/fused_attention.mojo:L333](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) · [max/examples/custom_ops/kernels/fused_attention.mojo:L335](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L335) · [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/examples/custom_ops/kernels/fused_attention.mojo:L344](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L344) · [max/examples/custom_ops/kernels/fused_attention.mojo:L346](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L346) · [max/examples/custom_ops/kernels/fused_attention.mojo:L348](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L348) · [max/examples/custom_ops/kernels/fused_attention.mojo:L349](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L349) · [max/examples/custom_ops/kernels/fused_attention.mojo:L352](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L352) · [max/examples/custom_ops/kernels/fused_attention.mojo:L353](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L353) · [max/examples/custom_ops/kernels/fused_attention.mojo:L354](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L354) · [max/examples/custom_ops/kernels/fused_attention.mojo:L355](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L355) · [max/examples/custom_ops/kernels/fused_attention.mojo:L356](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L356) · [max/examples/custom_ops/kernels/fused_attention.mojo:L358](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L358) · [max/examples/custom_ops/kernels/fused_attention.mojo:L359](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L359) · [max/examples/custom_ops/kernels/fused_attention.mojo:L360](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L360) · [max/examples/custom_ops/kernels/fused_attention.mojo:L362](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L362) · [max/examples/custom_ops/kernels/fused_attention.mojo:L363](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L363) · [max/examples/custom_ops/kernels/fused_attention.mojo:L366](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L366) · [max/examples/custom_ops/kernels/fused_attention.mojo:L367](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L367) · [max/examples/custom_ops/kernels/fused_attention.mojo:L368](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L368) · [max/examples/custom_ops/kernels/fused_attention.mojo:L370](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L370) · [max/examples/custom_ops/kernels/fused_attention.mojo:L371](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L371) · [max/examples/custom_ops/kernels/fused_attention.mojo:L372](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L372) · [max/examples/custom_ops/kernels/fused_attention.mojo:L373](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L373) · [max/examples/custom_ops/kernels/fused_attention.mojo:L374](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L374) · [max/examples/custom_ops/kernels/fused_attention.mojo:L375](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L375) · [max/examples/custom_ops/kernels/fused_attention.mojo:L376](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L376) · [max/examples/custom_ops/kernels/fused_attention.mojo:L377](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L377) · [max/examples/custom_ops/kernels/fused_attention.mojo:L378](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L378)
