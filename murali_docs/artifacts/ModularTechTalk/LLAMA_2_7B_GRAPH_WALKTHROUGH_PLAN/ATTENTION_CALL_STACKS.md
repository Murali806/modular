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
├─ Graph.__init__(name="fused_attention", input_types=[TQ,TK,TV], forward=None) ↗ [PY:L819]
│  ├─ @in_default_mlir_context → ensure an active MLIR context on this thread ↗ [PY:L70] A.0 · default MLIR context ↗ caller [PY:L818]
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
│  │     OUT: %q:TQ, %k:TK, %v:TV in the same entry block; results initially empty ↗ [C++:L848] ↗ caller [C++:L848]
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
   ├─ Graph.__enter__ → _enter → set Graph.current=G1 ↗ [PY:L1137] with Graph · Active recording scope ↗ caller [PY:L55]
   ├─ q, k, v, *_ = graph.inputs ↗ [PY:L56] Inputs · Three symbolic wrappers
   │  └─ Wrap body.arguments → q→%arg0, k→%arg1, v→%arg2; no numeric data ↗ [PY:L943] ↗ caller [PY:L56]
   ├─ Declare one output TensorType(dtype, [N,D], device) ↗ [PY:L63] Output type · Future O ↗ caller [PY:L62]
   ├─ results = ops.custom(name="modular_ops::fused_attention_custom", ...) ↗ [PY:L57] mo.custom · Detailed three-input call
   │  ├─ Resolve Graph.current; name → StringAttr; normalize DeviceRef ↗ [PY:L94] Attributes · Name, device, BN/BD ↗ caller [PY:L57]
   │  ├─ values=[q,k,v] → existing %arg0,%arg1,%arg2; out_types=[TO] → native result type ↗ [PY:L105] ↗ caller [PY:L61]
   │  ├─ Insert mo.custom in G1.body; wrap its one symbolic result %o ↗ [PY:L109] ↗ caller [PY:L109]
   │  ├─ Attach parameters AFTER insertion: BD/BN → index attributes → dictionary ↗ [PY:L113] ↗ [PY:L48] ↗ caller [PY:L60]
   │  ├─ Attach DeviceRefAttr via MLIR Attribute capsule ↗ [PY:L122] ↗ caller [PY:L122]
   │  ├─ Verify call against loaded P1; resolve FusedAttention.execute and map types/parameters ↗ [PY:L127] Kernel validation · P1 and MO ↗ caller [PY:L127]
   │  └─ Return results=[TensorValue(%o: float32[N,D], selected device)] ↗ [PY:L129] Result wrappers · One symbolic output ↗ caller [PY:L57]
   ├─ graph.output(*results) → mo.output, result signature, graph verification ↗ [PY:L68] Graph.output · Complete G1 ↗ caller [PY:L68]
   └─ Graph.__exit__ restores prior Graph.current ↗ [PY:L1145] ↗ caller [PY:L55]
      OUT: verified G1: (TQ,TK,TV) → (TO); compilation and execution are still later ↗ [PY:L68] ↗ caller [PY:L68]
```

Source links: [max/examples/custom_ops/fused_attention.py:L41](sources/max/examples/custom_ops/fused_attention.py#L41) · [max/examples/custom_ops/fused_attention.py:L43](sources/max/examples/custom_ops/fused_attention.py#L43) · [max/examples/custom_ops/fused_attention.py:L26](sources/max/examples/custom_ops/fused_attention.py#L26) · [max/examples/custom_ops/fused_attention.py:L29](sources/max/examples/custom_ops/fused_attention.py#L29) · [max/examples/custom_ops/fused_attention.py:L30](sources/max/examples/custom_ops/fused_attention.py#L30) · [max/examples/custom_ops/fused_attention.py:L30](sources/max/examples/custom_ops/fused_attention.py#L30) · [max/examples/custom_ops/fused_attention.py:L45](sources/max/examples/custom_ops/fused_attention.py#L45) · [max/examples/custom_ops/fused_attention.py:L44](sources/max/examples/custom_ops/fused_attention.py#L44) · [max/python/max/graph/graph.py:L819](sources/max/python/max/graph/graph.py#L819) · [max/python/max/_mlir_context.py:L70](sources/max/python/max/_mlir_context.py#L70) · [max/python/max/graph/graph.py:L818](sources/max/python/max/graph/graph.py#L818) · [max/python/max/graph/graph.py:L840](sources/max/python/max/graph/graph.py#L840) · [max/python/max/graph/graph.py:L847](sources/max/python/max/graph/graph.py#L847) · [max/python/max/graph/graph.py:L851](sources/max/python/max/graph/graph.py#L851) · [max/python/max/graph/graph.py:L856](sources/max/python/max/graph/graph.py#L856) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L62](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62) · [max/python/max/graph/graph.py:L858](sources/max/python/max/graph/graph.py#L858) · [max/python/max/graph/graph.py:L860](sources/max/python/max/graph/graph.py#L860) · [max/python/max/graph/graph.py:L861](sources/max/python/max/graph/graph.py#L861) · [max/examples/custom_ops/fused_attention.py:L42](sources/max/examples/custom_ops/fused_attention.py#L42) · [max/python/max/graph/graph.py:L864](sources/max/python/max/graph/graph.py#L864) · [GraphCompiler/lib/MODialect/MOOps.cpp:L831](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L831) · [GraphCompiler/lib/MODialect/MOOps.cpp:L846](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L846) · [GraphCompiler/lib/MODialect/MOOps.cpp:L848](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L848) · [max/python/max/graph/graph.py:L870](sources/max/python/max/graph/graph.py#L870) · [max/python/max/graph/graph.py:L871](sources/max/python/max/graph/graph.py#L871) · [max/python/max/graph/graph.py:L874](sources/max/python/max/graph/graph.py#L874) · [max/python/max/graph/graph.py:L877](sources/max/python/max/graph/graph.py#L877) · [max/python/max/graph/graph.py:L879](sources/max/python/max/graph/graph.py#L879) · [max/python/max/graph/graph.py:L881](sources/max/python/max/graph/graph.py#L881) · [max/python/max/graph/graph.py:L885](sources/max/python/max/graph/graph.py#L885) · [max/python/max/graph/graph.py:L886](sources/max/python/max/graph/graph.py#L886) · [max/python/max/graph/graph.py:L896](sources/max/python/max/graph/graph.py#L896) · [max/python/max/graph/graph.py:L900](sources/max/python/max/graph/graph.py#L900) · [max/python/max/graph/graph.py:L906](sources/max/python/max/graph/graph.py#L906) · [max/python/max/graph/graph.py:L909](sources/max/python/max/graph/graph.py#L909) · [max/python/max/graph/graph.py:L910](sources/max/python/max/graph/graph.py#L910) · [max/python/max/graph/graph.py:L363](sources/max/python/max/graph/graph.py#L363) · [max/python/max/graph/graph.py:L1703](sources/max/python/max/graph/graph.py#L1703) · [max/python/max/graph/graph.py:L269](sources/max/python/max/graph/graph.py#L269) · [Mojo/python/mojo/paths.py:L141](sources/Mojo/python/mojo/paths.py#L141) · [max/examples/custom_ops/kernels/__init__.mojo:L1](sources/max/examples/custom_ops/kernels/__init__.mojo#L1) · [max/python/max/graph/graph.py:L289](sources/max/python/max/graph/graph.py#L289) · [Mojo/python/mojo/paths.py:L169](sources/Mojo/python/mojo/paths.py#L169) · [Mojo/tools/mojo/Precompile/mojo-precompile.cpp:L209](sources/Mojo/tools/mojo/Precompile/mojo-precompile.cpp#L209) · [Mojo/lib/Compiler/KGENCompiler.cpp:L1119](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1119) · [Mojo/python/mojo/paths.py:L178](sources/Mojo/python/mojo/paths.py#L178) · [Mojo/python/mojo/paths.py:L179](sources/Mojo/python/mojo/paths.py#L179) · [max/python/max/graph/graph.py:L286](sources/max/python/max/graph/graph.py#L286) · [max/python/max/graph/graph.py:L1704](sources/max/python/max/graph/graph.py#L1704) · [max/python/max/graph/graph.py:L1706](sources/max/python/max/graph/graph.py#L1706) · [max/python/max/graph/graph.py:L918](sources/max/python/max/graph/graph.py#L918) · [max/examples/custom_ops/fused_attention.py:L55](sources/max/examples/custom_ops/fused_attention.py#L55) · [max/python/max/graph/graph.py:L1137](sources/max/python/max/graph/graph.py#L1137) · [max/examples/custom_ops/fused_attention.py:L56](sources/max/examples/custom_ops/fused_attention.py#L56) · [max/python/max/graph/graph.py:L943](sources/max/python/max/graph/graph.py#L943) · [max/examples/custom_ops/fused_attention.py:L63](sources/max/examples/custom_ops/fused_attention.py#L63) · [max/examples/custom_ops/fused_attention.py:L62](sources/max/examples/custom_ops/fused_attention.py#L62) · [max/examples/custom_ops/fused_attention.py:L57](sources/max/examples/custom_ops/fused_attention.py#L57) · [max/python/max/graph/ops/custom.py:L94](sources/max/python/max/graph/ops/custom.py#L94) · [max/python/max/graph/ops/custom.py:L105](sources/max/python/max/graph/ops/custom.py#L105) · [max/examples/custom_ops/fused_attention.py:L61](sources/max/examples/custom_ops/fused_attention.py#L61) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/python/max/graph/ops/custom.py:L113](sources/max/python/max/graph/ops/custom.py#L113) · [max/python/max/graph/ops/custom.py:L48](sources/max/python/max/graph/ops/custom.py#L48) · [max/examples/custom_ops/fused_attention.py:L60](sources/max/examples/custom_ops/fused_attention.py#L60) · [max/python/max/graph/ops/custom.py:L122](sources/max/python/max/graph/ops/custom.py#L122) · [max/python/max/graph/ops/custom.py:L127](sources/max/python/max/graph/ops/custom.py#L127) · [max/python/max/graph/ops/custom.py:L129](sources/max/python/max/graph/ops/custom.py#L129) · [max/examples/custom_ops/fused_attention.py:L68](sources/max/examples/custom_ops/fused_attention.py#L68) · [max/python/max/graph/graph.py:L1145](sources/max/python/max/graph/graph.py#L1145)

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
         ├─ validateInputSpecList → count=3, shapes/dtype match compiled signature ↗ [C++:L462]
         │  └─ Expected CPU example: three float32[8,8]; wrong count or static size raises ↗ [C++:L438] ↗ caller [C++:L462]
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

Source links: [max/examples/custom_ops/fused_attention.py:L82](sources/max/examples/custom_ops/fused_attention.py#L82) · [max/examples/custom_ops/fused_attention.py:L77](sources/max/examples/custom_ops/fused_attention.py#L77) · [max/examples/custom_ops/fused_attention.py:L78](sources/max/examples/custom_ops/fused_attention.py#L78) · [max/examples/custom_ops/fused_attention.py:L79](sources/max/examples/custom_ops/fused_attention.py#L79) · [max/examples/custom_ops/fused_attention.py:L80](sources/max/examples/custom_ops/fused_attention.py#L80) · [max/python/max/engine/api.py:L291](sources/max/python/max/engine/api.py#L291) · [max/python/max/engine/api.py:L118](sources/max/python/max/engine/api.py#L118) · [max/python/max/engine/api.py:L122](sources/max/python/max/engine/api.py#L122) · [max/python/max/engine/api.py:L124](sources/max/python/max/engine/api.py#L124) · [max/python/max/engine/api.py:L130](sources/max/python/max/engine/api.py#L130) · [max/python/max/engine/api.py:L136](sources/max/python/max/engine/api.py#L136) · [max/python/max/engine/api.py:L144](sources/max/python/max/engine/api.py#L144) · [max/python/max/engine/api.py:L145](sources/max/python/max/engine/api.py#L145) · [max/python/max/_core/max_license/modules/engine.cpp:L565](sources/max/python/max/_core/max_license/modules/engine.cpp#L565) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L457](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L457) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L462](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L462) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L438](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L438) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L464](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L464) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L475](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L475) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L476](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L476) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L479](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L479) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L484](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L484) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L112](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L112) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L116](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L116) · [GraphCompiler/lib/Driver/Device/DeviceExecution.cpp:L121](sources/GraphCompiler/lib/Driver/Device/DeviceExecution.cpp#L121) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L518](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L518) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L529](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L529) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L530](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L530) · [max/lib/max_license/API/c/ModelExecution.cpp:L204](sources/max/lib/max_license/API/c/ModelExecution.cpp#L204) · [max/lib/max_license/API/c/ModelExecution.cpp:L212](sources/max/lib/max_license/API/c/ModelExecution.cpp#L212) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L373](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L373) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L378](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L378) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L382](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L382) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L384](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L384) · [max/examples/custom_ops/kernels/fused_attention.mojo:L74](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L74) · [max/examples/custom_ops/kernels/fused_attention.mojo:L101](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) · [max/examples/custom_ops/kernels/fused_attention.mojo:L105](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L105) · [max/examples/custom_ops/kernels/fused_attention.mojo:L403](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L403) · [max/lib/max_license/API/c/ModelExecution.cpp:L213](sources/max/lib/max_license/API/c/ModelExecution.cpp#L213) · [max/lib/max_license/API/c/ModelExecution.cpp:L125](sources/max/lib/max_license/API/c/ModelExecution.cpp#L125) · [max/lib/max_license/API/c/ModelExecution.cpp:L137](sources/max/lib/max_license/API/c/ModelExecution.cpp#L137) · [max/lib/max_license/API/c/ModelExecution.cpp:L159](sources/max/lib/max_license/API/c/ModelExecution.cpp#L159) · [max/lib/max_license/API/c/ModelExecution.cpp:L166](sources/max/lib/max_license/API/c/ModelExecution.cpp#L166) · [max/lib/max_license/API/c/ModelExecution.cpp:L178](sources/max/lib/max_license/API/c/ModelExecution.cpp#L178) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L538](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L538) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L70](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L70) · [max/examples/custom_ops/fused_attention.py:L83](sources/max/examples/custom_ops/fused_attention.py#L83) · [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp:L79](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L79)

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

**The caller selects concrete sizes before constructing G1.** These are the values in [fused_attention.py](sources/max/examples/custom_ops/fused_attention.py#L30).

| Target | Q, K, V and O | Query BN | Output BD | Key rows per iteration |
|---|---|---|---|---|
| CPU | float32 [8,8] | 4 | 4 | 4 |
| GPU | float32 [32,32] | 16 | 8 | BN_1=8 |

**BN and BD are explicit compile-time parameters on the custom call.** They are not graph input tensors or symbolic dimension parameters. See [parameter attachment](sources/max/python/max/graph/ops/custom.py#L113).



### A.0 · default MLIR context

**The default is a module-level shared MLIR Context object. Current context is active state on this thread.** A context owns dialects and interned types/attributes; it does not contain the numerical Q/K/V matrices.

| Name | Example value | Role |
|---|---|---|
| `_DEFAULT_MLIR_CONTEXT` | context object MCX1 | Shared fallback in this interpreter |
| `mlir.Context.current` | MCX1 on main; None on a new worker | Per-thread active-context lookup |
| `MLIRThreadPoolExecutor` | executor EX1 | Wraps submitted tasks with the context guard |

The Python line `@in_default_mlir_context` is a **decorator**: it wraps the constructor. On a background thread, `call_with_default_mlir_context` preserves an existing current context or temporarily enters the shared default. It restores the previous scope afterward. See [the context module](sources/max/python/max/_mlir_context.py#L26) and [executor](sources/max/python/max/_mlir_context.py#L89).

**Background-thread example:** main submits `Graph(...)` construction to a worker whose current context is None. The guard enters MCX1 on that worker while it constructs IR. This does not copy main's active-context state or allocate another default context.



### _location() · What is loc?

**loc is an MLIR Location: provenance metadata attached to IR operations and input arguments.**

it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module.

[The helper](sources/max/python/max/graph/graph.py#L438) returns `Location.unknown()` with source-traceback capture disabled. When enabled, it derives metadata from the Python stack. An active `Graph.profile_scope()` can also contribute profiling-scope metadata.

**Example stack:** `fused_attention.py:87 → main() → fused_attention.py:41 → Graph(...)`. The native [frame binding](sources/max/python/max/_core/max_license/modules/graph.cpp#L162) wraps summaries in an opaque location. A readable label such as `main / fused_attention.py:41` illustrates provenance; it is not a captured dump.

Inside `with _location() as loc`, the binding scopes this Location as current. MAX also passes it explicitly to [ModuleOp and GraphOp](sources/max/python/max/graph/graph.py#L856); [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L848) attaches it to each input argument. **Locations remain attached after the with block exits.** loc is metadata, not the MLIR Context or a tensor address.



### MLIR Module Reuse or create

**MLIR.module is IR container.** The native `builtin.ModuleOp` normally prints as `module { ... }`. It has an operation → body region → body block → contained operations structure.

**Creation:** `module=None` calls [builtin.ModuleOp(location=loc)](sources/max/python/max/graph/graph.py#L856), whose [native constructor](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62) calls `mlir::ModuleOp::create`. The body initially has no graph definitions.

**Reuse:** `module=existing_module` retains `existing_module.mlir_module`. An existing `@encoder` definition remains a sibling of the newly inserted `@fused_attention`; no IR is copied by choosing this branch. See [the public Module wrapper](sources/max/python/max/graph/graph.py#L562).

```mlir
// Schematic nesting after GraphOp insertion
module {
  mo.graph @fused_attention(%q:T, %k:T, %v:T) {
    // one region, one block, three block arguments
  }
}
```

**The module groups definitions; the GraphOp owns the computation's body.**

```text
caller: self._module = module.mlir_module if module else builtin.ModuleOp(location=loc) ↗ [PY:L856]
├─ reuse: Module.mlir_module → existing builtin.ModuleOp ↗ [PY:L581] ↗ caller [PY:L856]
└─ create: native ModuleOp constructor ↗ [C++:L62]
   └─ mlir::ModuleOp::create(loc) → operation with body region/block ↗ [C++:L65] ↗ caller [PY:L856]
```

Source links: [max/python/max/graph/graph.py:L856](sources/max/python/max/graph/graph.py#L856) · [max/python/max/graph/graph.py:L581](sources/max/python/max/graph/graph.py#L581) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L62](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L62) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L65](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L65)

### OpBuilder · Insertion point

**An insertion point is a block plus a position in that block.** [body.end](sources/max/python/max/_core/max_license/_core.cpp#L408) creates an end iterator, and the [native OpBuilder constructor](sources/max/python/max/_core/max_license/_core.cpp#L612) receives it.

| Before | Builder action | After |
|---|---|---|
| module body `[encoder]` | insert at end | `[encoder, fused_attention]` |
| graph body `[%chain]` | later insert custom call | `[%chain, %o=custom(%q,%k,%v)]` |

The first builder inserts the graph into its module. [Graph._add_op_get_op_with_results](sources/max/python/max/graph/graph.py#L1284) uses the retained graph body for later call insertion. **These are different insertion blocks.**

```text
caller: OpBuilder(self._module.body.end) ↗ [PY:L858]
├─ ModuleOp.body → existing module-body block ↗ [C++:L67] ↗ caller [PY:L858]
├─ Block.end → InsertionPoint(block, block.end()) ↗ [C++:L408] ↗ caller [PY:L858]
└─ native OpBuilder(block, iterator) → insertion cursor ↗ [C++:L612] ↗ caller [PY:L858]
```

Source links: [max/python/max/graph/graph.py:L858](sources/max/python/max/graph/graph.py#L858) · [max/python/max/_core/max_license/modules/dialects/builtin.cpp:L67](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L67) · [max/python/max/_core/max_license/_core.cpp:L408](sources/max/python/max/_core/max_license/_core.cpp#L408) · [max/python/max/_core/max_license/_core.cpp:L612](sources/max/python/max/_core/max_license/_core.cpp#L612)

### to_mlir() · Q/K/V type fields

**to_mlir converts a Python type description into native MO type fields. It does not convert tensor data.** [TensorType.to_mlir](sources/max/python/max/graph/type.py#L430) constructs `mo.TensorType(shape_attr, dtype, device_attr, metadata=...)`.

| Field | Concrete example | Native meaning |
|---|---|---|
| Shape | GPU `[32,32]` | Two statically known sizes |
| Element dtype | `DType.float32` | Each future element is a 32-bit float |
| Device | `DeviceRef("gpu",0)` | Symbolic placement on gpu:0 |
| Metadata | `{}` here | No explicit layout constraint |

**A size attribute is not an element.** For a static size 32, [StaticDim.to_mlir](sources/max/python/max/graph/dim.py#L562) creates a signed i64 integer attribute, then wraps it with `kgen.CastFromBuiltinAttr`. This says “dimension length is 32”; the data elements are still float32. [Shape.to_mlir](sources/max/python/max/graph/shape.py#L36) collects dimension expressions in `mosh.ShapeAttr`.

**Layout comparison:** a convolution filter may carry `metadata={"layout": ...}` such as HWIO or OIHW. Native [TensorType::getLayoutAttr](sources/GraphCompiler/lib/MODialect/MOTypes.cpp#L454) reads that metadata to interpret dimensions. **Attention leaves this dictionary empty**; it does not attach a filter layout. Logical shape is separate from target-specific fragment/storage layout.

```text
caller: [t.to_mlir() for t in input_types] ↗ [PY:L864]
└─ TensorType.to_mlir → mo.TensorType(...) ↗ [PY:L430]
   ├─ self.shape.to_mlir() → ShapeAttr(dim attributes, ShapeType()) ↗ [PY:L36]
   │  └─ each StaticDim(32) → signed i64 IntegerAttr(32) → CastFromBuiltinAttr ↗ [PY:L562] ↗ caller [PY:L39]
   ├─ self.dtype=DType.float32 → existing element dtype value ↗ [PY:L441] ↗ caller [PY:L439]
   ├─ self.device.to_mlir() → m.DeviceRefAttr("gpu",0) ↗ [PY:L181] ↗ caller [PY:L442]
   └─ builtin.DictionaryAttr(metadata={}) → no explicit filter layout here ↗ [PY:L443] ↗ caller [PY:L443]
```

Source links: [max/python/max/graph/graph.py:L864](sources/max/python/max/graph/graph.py#L864) · [max/python/max/graph/type.py:L430](sources/max/python/max/graph/type.py#L430) · [max/python/max/graph/shape.py:L36](sources/max/python/max/graph/shape.py#L36) · [max/python/max/graph/dim.py:L562](sources/max/python/max/graph/dim.py#L562) · [max/python/max/graph/shape.py:L39](sources/max/python/max/graph/shape.py#L39) · [max/python/max/graph/type.py:L441](sources/max/python/max/graph/type.py#L441) · [max/python/max/graph/type.py:L439](sources/max/python/max/graph/type.py#L439) · [max/python/max/graph/type.py:L181](sources/max/python/max/graph/type.py#L181) · [max/python/max/graph/type.py:L442](sources/max/python/max/graph/type.py#L442) · [max/python/max/graph/type.py:L443](sources/max/python/max/graph/type.py#L443)

### GraphOp · Signature and body

**One graph region contains ONE entry block with THREE input arguments.** The caller provides input types, not an existing region. [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L831) records a function type, creates the region/block, then adds one block argument for each input type.

```text
MODULE1: builtin.module
└─ G1: mo.graph @fused_attention
   ├─ signature: (TQ,TK,TV) → () initially
   └─ region R0
      └─ block B0(%q:TQ,%k:TK,%v:TV)
         └─ operations will be inserted here
```

**Why a signature?** It declares the caller/callee contract: count, type, shape and device of inputs and outputs. During output finalization, it becomes `(TQ,TK,TV) → (TO)`. During execute, the runtime checks three real buffers against this contract. A block argument is an SSA value at entry, not a separately allocated block or Buffer.



### Shape names · Graph parameters

**All attention dimensions here are static: _params={}.** The constructor's [discovery](sources/max/python/max/graph/graph.py#L840) and [declaration](sources/max/python/max/graph/graph.py#L870) therefore attach an empty parameter array.

**Comparison:** three input shapes `["seq",32]` would discover and deduplicate `seq`; the graph's input parameter declaration would carry one signed-i64 size name. That is different from `parameters={"BN":16,"BD":8}` on the custom op. BN/BD choose kernel tile sizes; they are not additional Q/K/V inputs.



### Graph handles · CAPI and body

**_CAPICreate(op._CAPIPtr) creates another wrapper around the SAME native GraphOp.** CAPI means the MLIR C API used between MAX's nanobind wrapper and MLIR's Python bindings. There is no graph clone or serialized IR here.

```text
MAX GraphOp wrapper op
  → _CAPIPtr: named PyCapsule(pointer=P)
  → MlirOperation {ptr=P}
  → MLIR Python PyOperation(pointer=P, context reference)
  → existing region R0 / block B0
```

The [MAX export](sources/max/python/max/_core/max_license/_core.cpp#L606) wraps the existing operation. The pinned upstream [capsule adapter](sources/external/llvm-project/mlir/include/mlir-c/Bindings/Python/Interop.h#L326) and [factory](sources/external/llvm-project/mlir/lib/Bindings/Python/IRCore.cpp#L1207) import the handle and recover its context. Native address P is an illustrative identity, not a measured address.

[Graph.__init__](sources/max/python/max/graph/graph.py#L874) then retains `_mlir_op`, `_current_block=regions[0].blocks[0]` and `_graph_body`. **All refer to the already-created graph/body.** G1 is cloned later by compilation, not by this pointer bridge.



### Device metadata · Targets

**Tensor placement, module target metadata, and session device handles have different roles.** [The mapping helper](sources/max/python/max/graph/graph.py#L1366) stores CPU information and, when present, one default accelerator's API/architecture/model description on the enclosing module.

| Object | Example | Consumer |
|---|---|---|
| Tensor DeviceRef | `gpu:0` | Placement and type contracts |
| Module device-info entry | GPU API/arch/model | Compiler target decisions |
| Session device list | `[GPU0,CPU0]` | Runtime driver contexts |

If the module already has this mapping, the helper returns. Missing architecture/model queries use unknown. **This metadata step neither compiles nor executes a GPU kernel.**



### Graph bookkeeping · Weights and chains

**_weights={} records no external weight declarations. Q/K/V are runtime inputs.** [Initialization](sources/max/python/max/graph/graph.py#L879) also creates `_has_chain_input=False` and a lazy `_DeviceChainMap`.

The [chain map](sources/max/python/max/graph/graph.py#L123) retains G1; later device lookups can create dependency seeds. These references order effects. They are separate from matrix storage and from the session's runtime worker or stream.



### Chain · Ordering token

**CH0 is an always-ready dependency token with no producer inputs.** [ChainCreateOp](sources/max/python/max/graph/graph.py#L885) is inserted into G1's body, wrapped as `_always_ready_chain`, then used to seed CPU ordering.

```text
CH0 = mo.chain.create(inputs=[])
device_chains[cpu:0] → CH0
user inputs → %q,%k,%v
```

CH0 is internal bookkeeping, not a fourth input tensor. Later ordered operations may advance per-device chains; the always-ready seed itself remains available.



### Last input · Tensor or chain?

**The constructor checks the last existing block argument. In attention it is %v, a tensor.** [The native-type check](sources/max/python/max/graph/graph.py#L896) therefore leaves `_has_chain_input=False` and keeps the initial CPU chain.

**Subgraph comparison:** if a parent passes a trailing chain argument, the child adopts it and excludes internal chain arguments when exposing `Graph.inputs`. It does not create one chain for each Q/K/V matrix.



### P1 · Executable, parsing and bytecode

**mojo is an executable compiler driver launched in a separate OS process.** [subprocess_run_mojo](sources/Mojo/python/mojo/run.py#L57) chooses `MODULAR_MOJO_MAX_DRIVER_PATH` or falls back to `shutil.which("mojo")`, then invokes `[mojo]+args`.

| Environment | Path selection |
|---|---|
| MAX SDK/wheel | package root `bin/mojo`, unless overridden |
| Bazel runfiles | configured driver path from runtime toolchain |
| PATH fallback | the actual path returned by `shutil.which` |

The exact installed absolute executable path is environment-specific; this document has not run it. See [SDK defaults](sources/Mojo/python/mojo/run.py#L22) and [Bazel exports](sources/bazel/internal/mojo_test_environment.bzl#L178).

**Source package input:** `kernels/__init__.mojo` and referenced modules, including `fused_attention.mojo`. The Python entry file is not compiled into P1. [The helper](sources/Mojo/python/mojo/paths.py#L141) calls `mojo precompile <kernels dir> -o <private staging dir>/mojo_pkg_<path hash>.mojoc` and atomically publishes it. The staging filename uses the final basename.

**Format:** a precompiled parametric Mojo package, not CPU/GPU machine code. [The writer](sources/Mojo/lib/Support/MojoPrecompiledFile.cpp#L137) stores version/header information and compressed MLIR bytecode. A text rendering of a `lit.package` is an inspection view, not the binary bytecode itself. Parsing, declaration resolution and semantic/lifetime checks are separate from later concrete kernel specialization. No full compiler dump is claimed here.



### with Graph · Active recording scope

**Attention uses a with-body, not a forward lambda.** After the constructor finishes importing P1, Python invokes [Graph.__enter__](sources/max/python/max/graph/graph.py#L1137), which activates the MLIR context and sets the graph construction ContextVar.

`Graph.current` resolves G1 while [ops.custom](sources/max/python/max/graph/ops/custom.py#L94) runs. Exit restores the prior graph; this is a scoped graph reference, not the module-level default MLIR Context.



### Inputs · Three symbolic wrappers

**q, k, v refer to G1's three block arguments.** [Graph.inputs](sources/max/python/max/graph/graph.py#L943) wraps native SSA values as Python `TensorValue`s.

| Python | Native | Example type |
|---|---|---|
| q | %arg0 | float32 [32,32] gpu:0 |
| k | %arg1 | float32 [32,32] gpu:0 |
| v | %arg2 | float32 [32,32] gpu:0 |

These lowercase values contain shape/type/device information, not numerical matrices. Uppercase `Q/K/V` Buffers are created after init and supplied to execute.



### Output type · Future O

**The declared output type is float32 [N,D] on the selected device.** The [caller](sources/max/examples/custom_ops/fused_attention.py#L63) explicitly constructs it from `dtype`, `N`, `D`, and `DeviceRef.from_device(device)`.

**GPU example:** `out_types=[TensorType(DType.float32,[32,32],gpu:0)]`. **CPU example:** `[8,8]` on cpu:0. This describes the custom call's single future result; no output Buffer is allocated by declaring the type.



### Attributes · Name, device, BN/BD

**The registration string must match the namespaced Mojo registration exactly.** [ops.custom](sources/max/python/max/graph/ops/custom.py#L96) creates a StringAttr, and the [registered struct](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L66) supplies the definition.

| Python value | MO attribute/value example |
|---|---|
| name | `StringAttr("modular_ops::fused_attention_custom")` |
| GPU device | `DeviceRefAttr("gpu",0)` |
| GPU parameters | `BD=8:index, BN=16:index` |
| CPU parameters | `BD=4:index, BN=4:index` |

**Integer parameters use MLIR IndexType, not tensor float32.** [_parameter_attribute](sources/max/python/max/graph/ops/custom.py#L48) performs that conversion. [The parameters dictionary](sources/max/python/max/graph/ops/custom.py#L113) and [device attribute](sources/max/python/max/graph/ops/custom.py#L122) are attached after operation insertion, before validation.



### mo.custom · Detailed three-input call

**Python records one opaque attention call; it does not run softmax or translate arbitrary Python into Mojo.** [The insertion helper](sources/max/python/max/graph/graph.py#L1284) unwraps operands, converts result types, inserts the operation in G1's body, and creates symbolic result wrappers.

```mlir
// Schematic MO: full printed dialect syntax may differ
module {
  mo.graph @fused_attention(%q:T, %k:T, %v:T) -> T {
    %ch = mo.chain.create()
    %o = mo.custom(%q,%k,%v) {
      symbol = "modular_ops::fused_attention_custom",
      parameters = {BD=8:index, BN=16:index}, device=gpu:0
    } : (T,T,T) -> T
    mo.output %o
  }
}
// T describes float32 [32,32] on gpu:0; no matrix data here.
```

The Mojo algorithm stays in P1 until specialization/JIT. **One result type produces one symbolic `%o`, even though the call has three inputs.**

```text
caller: results = ops.custom(...) ↗ [PY:L57]
└─ custom(name,device,values=[q,k,v],out_types=[TO],parameters={BD:8,BN:16}) ↗ [PY:L69]
   ├─ Graph.current → G1; StringAttr.get(name,context) ↗ [PY:L94] ↗ caller [PY:L96]
   ├─ normalize TensorValue operands; convert output TO with to_mlir ↗ [PY:L105] ↗ caller [PY:L110]
   ├─ Graph._add_op_get_op_with_results(mo.custom,[TO],values,symbol=...) ↗ [PY:L109]
   │  └─ native operation CUST1 + result %o; wrappers refer to that result ↗ [PY:L1284] ↗ caller [PY:L109]
   ├─ _parameter_attribute(8/16) → IndexType IntegerAttr; DictAttr attaches parameters ↗ [PY:L48] ↗ caller [PY:L116]
   ├─ device.to_mlir → Attribute._CAPICreate → custom_op.device ↗ [PY:L122] ↗ caller [PY:L122]
   ├─ KernelLibrary.verify_custom_op(custom_op) ↗ [PY:L127] ↗ caller [PY:L127]
   └─ return [TensorValue(%o:TO)] ↗ [PY:L129] ↗ caller [PY:L57]
```

Source links: [max/examples/custom_ops/fused_attention.py:L57](sources/max/examples/custom_ops/fused_attention.py#L57) · [max/python/max/graph/ops/custom.py:L69](sources/max/python/max/graph/ops/custom.py#L69) · [max/python/max/graph/ops/custom.py:L94](sources/max/python/max/graph/ops/custom.py#L94) · [max/python/max/graph/ops/custom.py:L96](sources/max/python/max/graph/ops/custom.py#L96) · [max/python/max/graph/ops/custom.py:L105](sources/max/python/max/graph/ops/custom.py#L105) · [max/python/max/graph/ops/custom.py:L110](sources/max/python/max/graph/ops/custom.py#L110) · [max/python/max/graph/ops/custom.py:L109](sources/max/python/max/graph/ops/custom.py#L109) · [max/python/max/graph/graph.py:L1284](sources/max/python/max/graph/graph.py#L1284) · [max/python/max/graph/ops/custom.py:L48](sources/max/python/max/graph/ops/custom.py#L48) · [max/python/max/graph/ops/custom.py:L116](sources/max/python/max/graph/ops/custom.py#L116) · [max/python/max/graph/ops/custom.py:L122](sources/max/python/max/graph/ops/custom.py#L122) · [max/python/max/graph/ops/custom.py:L127](sources/max/python/max/graph/ops/custom.py#L127) · [max/python/max/graph/ops/custom.py:L129](sources/max/python/max/graph/ops/custom.py#L129)

### Kernel validation · P1 and MO

**Validation connects a symbolic call to a registered Mojo contract. It does not execute the kernel.** The animation shows registry lookup, device selection, input/result role mapping and parameter checks.

| Graph value | Destination-passing Mojo role |
|---|---|
| %q | query |
| %k | key |
| %v | value |
| %o (call result) | output |

[MojoRegistry](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L48) indexes declarations by registration string. [Operand inference](sources/GraphCompiler/lib/Conversion/OperandInference.cpp#L240) maps graph inputs and result destinations, including the output-first Mojo signature. Dtype/rank are inferred; target follows the device; BN/BD are checked as supplied integer parameters. See [native verifier](sources/GraphCompiler/lib/MOGGDialect/Analysis/MojoRegistry.cpp#L634).

**Failure example:** `name="fused_attention_typo"` has no matching entry. An extra input, mismatched rank or an unknown parameter also fails validation rather than returning a successful wrapper. `rank==2` is additionally asserted by the specialized Mojo implementation.



### Result wrappers · One symbolic output

**The returned value is a list with ONE TensorValue wrapper.** [return results](sources/max/python/max/graph/ops/custom.py#L129) returns the wrapper created during operation insertion.

```text
results = [TensorValue(native=%o,
                       dtype=float32,
                       shape=[32,32], device=gpu:0)]
graph.output(*results) → graph.output(results[0])
```

The fields above describe the wrapper conceptually; they are not a dump of private Python members. `%o` is owned by CUST1 in G1. The later `output=[Buffer O1]` from execute contains actual storage and is a different list.



### Graph.output · Complete G1

**graph.output(*results) finishes and verifies G1's symbolic contract.** It records the result, adds the output terminator, and updates the declared result signature. See [the call](sources/max/examples/custom_ops/fused_attention.py#L68) and [Graph.output](sources/max/python/max/graph/graph.py#L1396).

```text
before: (TQ,TK,TV) → ()    body contains custom result %o
after:  (TQ,TK,TV) → (TO)  body ends with output(%o)
```

This graph-building finalization is distinct from later MGP device-graph finalization, kernel JIT, and runtime model initialization.



### Session · Configurations, contexts and devices

**S1 owns settings and native runtime references as well as device handles.** [The constructor](sources/max/python/max/engine/api.py#L664) normalizes devices and calls the native session.

| Illustrative member | Original-call value |
|---|---|
| S1.num_threads | None |
| S1._impl | native N1 |
| N1.devices | [CPU0] or [GPU0,CPU0] |
| N1.sessionCompileConfig | CFG1; no model source or extension paths yet |
| N1.runtimeConfig | threads=0 (automatic), API=python, registered device/context pairs |
| N1.runtimeContext | Engine::Context E1 → shared MAX ContextRef CX1 → AsyncRT CPUDevice HRT1 |

The [native session](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L546) clones compilation settings and adds supplied session extensions. This caller supplies none: **P1 is attached to G1 and passed later by compile_async**. An extension example `/tmp/addon.mojoc` is an existing compiled package path, not a Python class.

[Context creation/reuse](sources/max/lib/max_license/API/c/EngineContext.cpp#L184) can reuse a matching global runtime CX1 across sessions. **CX1 is MAX's runtime/services context, not Graph's MLIR context.**



### Compiler · MO → MAP → MOGG → MGP

**Compiler work transforms clone C1 while G1 remains retained.** [MOFContext::compileToBinary](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L381) clones the module and checks the graph-artifact cache.

| Step | Attention example | Why it is needed |
|---|---|---|
| MO | three typed tensors → namespaced custom call → one output | Verify graph/shape/device contracts |
| MAP | opaque iteration with Q/K/V accesses, custom call, O destination | Expose graph-level accesses and implementation boundaries |
| MOGG | bound FusedAttention.execute routine and mapped tensor arguments | Connect registry implementation to executable glue |
| MGP | init/execute regions, buffers, state slots, chains | Express runtime ownership/readiness and calls |
| JIT + final lowering | compiled routine archive; ModuleLoad/ExecuteGeneric; outlined callables | Make the plan callable by init/execute |

**Opaque custom boundary:** MAP does not see or fuse the softmax recurrence inside FusedAttention. Its tiling is Mojo code. Graph canonicalization may remove redundant graph structure when eligible; this example does not prove the custom operation is algebraically simplified.

The [pipeline builder](sources/GraphCompiler/lib/GraphCompiler.cpp#L199) installs MLIR passes and runs them in-process. The detailed pipeline supplement below explains why each pass is needed and labels schematic examples.

```text
caller: convertMOToMEF(C1,options) ↗ [C++:L356]
└─ MO::runGCPipeline(C1,options) → PassManager.run ↗ [C++:L638]
   ├─ buildMoPipeline → verify/canonicalize MO contracts ↗ [C++:L219] ↗ caller [C++:L638]
   ├─ MOToMAPPass → opaque accesses/call for custom attention ↗ [C++:L764] ↗ caller [C++:L638]
   ├─ buildMapPipeline → access-pattern optimization; opaque call retained ↗ [C++:L338] ↗ caller [C++:L638]
   ├─ MAPToMOGGPass → MojoRegistry + MAPToMOGGState ↗ [C++:L53] ↗ caller [C++:L638]
   ├─ buildMoggPipeline → generated functions/calls ↗ [C++:L370] ↗ caller [C++:L638]
   ├─ MOGGToMGPPass → MOGraphLowering init/execute state ↗ [C++:L2406] ↗ caller [C++:L638]
   ├─ buildMgpPipeline → guarded device finalization + EmitMojo ↗ [C++:L382] ↗ caller [C++:L638]
   ├─ JitCompileKernelsPass → compiled native routines or kernel-cache reuse ↗ [C++:L529] ↗ caller [C++:L638]
   └─ buildJitPipeline → ModuleLoad/ExecuteGeneric + outlined callables ↗ [C++:L75] ↗ caller [C++:L638]
      Detailed pipeline · transformations and examples
```

Source links: [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp:L356](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356) · [GraphCompiler/lib/GraphCompiler.cpp:L638](sources/GraphCompiler/lib/GraphCompiler.cpp#L638) · [GraphCompiler/lib/GraphCompiler.cpp:L219](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:L764](sources/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp#L764) · [GraphCompiler/lib/GraphCompiler.cpp:L338](sources/GraphCompiler/lib/GraphCompiler.cpp#L338) · [GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:L53](sources/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp#L53) · [GraphCompiler/lib/GraphCompiler.cpp:L370](sources/GraphCompiler/lib/GraphCompiler.cpp#L370) · [GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:L2406](sources/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp#L2406) · [GraphCompiler/lib/GraphCompiler.cpp:L382](sources/GraphCompiler/lib/GraphCompiler.cpp#L382) · [GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp:L529](sources/GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp#L529) · [GraphCompiler/lib/GraphCompiler.cpp:L75](sources/GraphCompiler/lib/GraphCompiler.cpp#L75)

### Kernel · Mojo, LLVM and target artifacts

**JIT means just-in-time compilation: code generation requested by the running MAX process.** This example calls `session.compile(graph)` before supplying numeric buffers. The kernel code is prepared there (or reused from a cache), rather than running `mojo build` for every matrix execution.

```text
P1 parametric FusedAttention.execute
 → dtype=float32,rank=2,BN=16,BD=8,target=gpu
 → KGEN elaboration/specialization
 → MLIR LLVM dialect (still MLIR)
 → LLVM1: llvm::Module (LLVM IR)
 → host object + selected device code
```

[KGENCompiler](sources/Mojo/lib/Compiler/KGENCompiler.cpp#L1140) runs elaboration. [ObjectCompiler](sources/Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp#L816) lowers and translates into LLVM IR. The compiler-side [TargetBackendRegistry](sources/Mojo/lib/Compiler/ObjectCompiler/Target/TargetBackend.cpp#L114) selects a registered backend by target triple; architecture/features configure target codegen.

**NVIDIA example:** `nvptx64-nvidia-cuda`, illustrative arch `sm_90a`, [NVPTX backend](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/NVPTX/NVPTXBackend.cpp#L1) → PTX and device binary route. **AMD example:** `amdgcn-amd-amdhsa`, illustrative arch `gfx942`, [AMDGPU backend](sources/Mojo/lib/Compiler/ObjectCompiler/Target/max_license/AMDGPU/AMDGPUBackend.cpp#L77) → AMDGPU object/ELF route. The [cross-target source test](sources/GraphCompiler/test/mo-opt/cross-compile-gpu-arch.mlir#L1) supplies both architecture examples.

The host Python/MAX runtime and compiled host glue stay on CPU. Device kernels have separate offload targets. **P1 .mojoc preparation is a distinct earlier subprocess.** Architecture examples are not GPU detection or execution results.



### Artifact · Prepared CM1 bytes

**CM1 is a compiled artifact, not an initialized executable Model.** [convertMOToMEF](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356) runs graph/JIT work and exports the runtime plan and code bytes.

```text
CM1 ingredients: MGP plan + model descriptors + compiled routine archive
init(CM1): load symbols / initialize resources / create M1
execute(Q,K,V): invoke M1 with actual data
```

Graph-artifact cache reuse and kernel-code cache reuse are separate. They skip different work; neither makes the Python graph a machine-code file or computes Q/K/V output during construction.



### Initialization · Loaded routines and resources

**Initialization turns descriptors and compiled bytes into retained runtime state and callable handles.** [Model::setup](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L185) builds initializer arguments, resolves devices, invokes the initializer, and constructs `ExecutableModel(chain,mgpContext,weightsRegistry,extraArgs)`.

| Retained value | Example role |
|---|---|
| chain | asynchronous setup readiness |
| mgpContext | runtime state/module/function slots |
| weightsRegistry | empty external weights here |
| extraArgs | device and initialization state needed by execute |

**Why a CPU worker?** [Engine::Context::initModel](sources/max/lib/max_license/API/c/EngineContext.cpp#L297) schedules setup on the CPU runtime so the task has the correct AsyncRT CPUDevice association. Setup may depend on asynchronous resources; the caller waits for readiness instead of pretending initialization is already complete.

**Host archive loading is separate from driver launch.** `CompiledKernelsContext::load` hashes an archive, installs/reuses its layer, and resolves named native callable pointers. Later host routine execution reaches GPU module/function loading and launch. Use the directly linked loader supplement for that deeper tree and animation.

```text
caller: initFn->invoke(initArgs,cpuDevice) ↗ [C++:L220]
├─ mgp_module_load_cpu → runtime module slot ↗ [C++:L55]
│  └─ CompiledKernelsContext::load(archive_bytes,symbols) ↗ [C++:L85]
│     ├─ hash archive → addIfAbsent<StaticArchiveLayer> ↗ [C++:L91] ↗ [C++:L160] ↗ caller [C++:L160]
│     └─ lookup symbols → resident native callable pointers ↗ [C++:L165] ↗ caller [C++:L165]
│        Archive loading → native symbols → device driver · detailed animation
└─ ExecutableModel(chain,mgpContext,weightsRegistry,extraArgs) ↗ [C++:L237]
   └─ waitForModelSetup → readiness/error checks before M1 is returned ↗ [C++:L387] ↗ caller [C++:L237]
```

Source links: [GraphCompiler/lib/MEFModel/MEFModel.cpp:L220](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L220) · [GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp:L55](sources/GraphCompiler/lib/MGPDialect/Primitives/MGPCPUPrimitives.cpp#L55) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L85](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L85) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L91](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L91) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L160](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L160) · [GraphCompiler/lib/Support/CompiledKernelsContext.cpp:L165](sources/GraphCompiler/lib/Support/CompiledKernelsContext.cpp#L165) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L237](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L237) · [GraphCompiler/lib/MEFModel/MEFModel.cpp:L387](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L387)

### Initialization · Bindings and readiness

**The attention model has no external weights; Q/K/V are inputs supplied at execute.** [init_all](sources/max/python/max/engine/api.py#L1202) gets `weights_registry_real={}` and checks the expected weight names/devices: also `{}` here.

The real-device route waits for compiled and initialization AsyncValues. Virtual-device mode returns CompileOnlyModel stand-ins that cannot execute. A successful compile is therefore distinct from a ready M1.



### E · Buffers, execution and output

**Uppercase Q/K/V become real Buffers after compile and init.** The [original call](sources/max/examples/custom_ops/fused_attention.py#L82) passes three inputs and receives `output=[O1]`.

| Phase | Inputs | Result |
|---|---|---|
| Graph recording | q/k/v symbolic SSA wrappers | symbolic %o |
| compile | G1/P1/types/parameters | CM1 prepared artifact |
| init | CM1/S1 contexts | M1 callable state |
| execute | numeric Q/K/V Buffers | one numeric O1 Buffer |

**Tiny teaching row:** Q=[1,0], scores=[1,0,3,2]. Two tiles update m/l/O; the second maximum grows 1→3. The online-softmax card shows actual browser numbers at every update.

The original source prints the output Buffer list. It has **no explicit host copy and no numerical parity assertion**. Browser reference agreement is an independent tiny arithmetic illustration, not a measured MAX/GPU result.



### Dispatch · CPU/GPU specialization

**The graph device becomes the compile-time target for the registered routine.** [FusedAttention.execute](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L74) receives output/query/key/value/context in destination-passing form.

```text
dtype=float32 and rank=2 ← tensor types
BN/BD ← Python custom parameters
target="cpu" or "gpu" ← graph device
ctx ← runtime device context
```

The [CPU branch](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L101) invokes native CPU loops. The [GPU branch](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L105) calls the enqueue wrapper. Specialization removes the unselected compile-time branch; NVIDIA/AMD selection is a lower target-backend decision.



### Launch · Blocks and driver handoff

**GPU grid=(4,2), block=32 for this caller's sizes.** [The wrapper](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L391) specializes the kernel and passes Q/K/V/O with geometry to `ctx.enqueue_function`.

| Coordinate | Ownership |
|---|---|
| block_idx.y ∈ [0,1] | one group of 16 query rows |
| block_idx.x ∈ [0,3] | one group of 8 output columns |
| eight blocks | cover O [32,32] as 16×8 tiles |
| key loop t ∈ [0,3] | visits four groups of 8 keys |

Host launch metadata, native callable address, GPU function handle and tensor addresses are different values. Driver loading/submission/readiness is described in the linked loader lesson; this page does not run a GPU or claim a measured schedule.



### MMA · Loads, accumulation and barrier

**S is computed from operand fragments, not a saved full attention matrix.** [GPU matmul](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L234) creates a TensorCore helper with BK=8, a LOCAL accumulator, and SHARED output storage.

```text
Q_tile [16,32] × K_tileᵀ [32,8] → S [16,8]
features split into four BK=8 slices
load_a / load_b → mma_op(A,B,C) → retained C accumulator
store_d(out_sram,C) → barrier() → res.copy_from(out_sram)
```

The same helper is used for E×V in the output update. **barrier() is a block synchronization point for the shared temporary; it is not the global model result-readiness gate.** Exact hardware instructions/residency depend on the target backend and are not measured here.



### Storage · Current tiles and retained state

**The fused loop keeps one score/weight tile and per-query running state.** Source [tile selection](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L333) and [updates](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L376) establish the logical shapes.

| GPU block object | Logical shape | Lifetime |
|---|---|---|
| Q tile view | [16,32] | all key iterations for this block |
| K/V views | [8,32] / [8,8] | current key tile |
| S/E temporaries | [16,8] | current key tile |
| running m/l | [16] each | retained between key tiles |
| normalized O_i | [16,8] | retained, then stored once |

Q/K/V tile selection creates views. **The source does not allocate a persistent full N×N score matrix inside this loop.** Logical storage counts do not prove physical register/shared/local memory placement or total peak allocations.



### Online softmax · Two tiles and rescaling

**Each key tile updates one stable max, denominator, and normalized output per query row.** The source computes an unscaled, rank-2 `softmax(QKᵀ)V`: no heads, batch, mask, dropout, or 1/√D scale.

```text
m_new = max(m_old, max(S_tile))
alpha = exp(m_old - m_new)
E = exp(S_tile - m_new)
l_new = alpha*l_old + sum(E)
O_new = O_old * alpha*l_old/l_new + (E/l_new) @ V_tile
```

**Why rescale?** When a later maximum grows, earlier exponentials were measured against an older maximum. Multiplying their retained mass by alpha expresses them relative to the new maximum. Multiplying old normalized O by `alpha*l_old/l_new` also adjusts for the new denominator.

**Tiny values:** Q=[1,0]; K=[[1,0],[0,1],[3,0],[2,1]]; V=[[1,0],[0,2],[2,1],[1,3]]. Tile scores are [1,0], then [3,2]. The animation computes all terms and checks against an independent stable full-softmax result. Teaching tile size is 2; source GPU key size is 8.

Source initializes m with the dtype's minimum value; the teaching model uses mathematical −∞ for readability. See [max](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L348), [denominator](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L352), and [output update/store](sources/max/examples/custom_ops/kernels/fused_attention.mojo#L366).
