"""Compilation/runtime ownership; all values are source-based teaching examples."""
from compiler_attention_lessons import NAME, Story, T_GPU, lesson, objects


def runtime_lessons():
    cards=[]
    def add(*a,**kw):
        cards.append(lesson(*a,**kw))

    nodes=objects(("devices","final_devices","driver object references","Python S1"),
                  ("compile","sessionCompileConfig","CompilationRequest","native S1"),
                  ("runtime","runtimeConfig","Engine::Config","native S1"),
                  ("max","MAXCTX1","shared MAX services","ContextRef"),
                  ("cpu","CPUDevice / workers","AsyncRT runtime","MAXCTX1"),
                  ("engine","runtimeContext","Engine::Context","native S1"),
                  ("session","S1","Python/native session","caller"))
    def session(reuse=False):
        s=Story(nodes,label="Comparison · shared MAX Context reused" if reuse else "First session · create MAX services",view="ownership")
        s.step("Deduplicate requested devices","ct_session",693,"devices=[GPU0]","append unique devices in input order","[GPU0]",
               "Device handles are retained; duplicates do not require duplicate runtime entries.",patch={"devices":"[GPU0]"})
        s.step("Ensure the host CPU is available","ct_session",697,"seen_devices lacks CPU0","append CPU()","[GPU0,CPU0]",
               "Host orchestration, compilation, resource setup and driver calls still need CPU services.",patch={"devices":"[GPU0,CPU0] · ordered references"})
        s.step("Build the native session with no explicit extensions","ct_session",707,"devices; extensions=[]; num_threads=None","_InferenceSession(...,num_threads or 0)","native constructor inputs",
               "The actual example gives P1 to Graph. The session constructor itself receives no P1 path.",patch={"session":"S1._impl · native constructor", "compile":"base CompilationRequest; custom extension list=[]"})
        s.step("Clone the session compile configuration","ct_session_native",546,"current sessionCompileConfig + []","populateCompileConfig","sessionCompileConfig retained",
               "A supplied existing .mojoc path would append a kernel-library path; Python classes or raw tensors are not extension entries.",patch={"compile":"CompilationRequest{kernelLibraryPaths=base defaults; supplied=[]}"})
        s.step("Retain runtime device specifications and contexts","ct_session_native",551,"[GPU0,CPU0]","runtimeConfig.addDevice({shared DeviceSpec,device.ctx})","device refs in runtime config",
               "The symbolic graph DeviceRefs are resolved against these runtime handles later.",patch={"runtime":"numThreads=0 (use available threads); apiLanguage=python\ndevices=[{GPU spec,ctxGPU},{CPU spec,ctxCPU}]"})
        s.step("Create the Engine context wrapper","ct_context_create",169,"runtimeConfig","Engine::Context::create(config)","choose supplied/shared MAX Context",
               "Engine::Context is the runtime wrapper. It is not the MLIR Context or Graph object.",patch={"engine":"pending Engine::Context(config,ContextRef)"})
        s.step("Acquire shared MAX services","ct_context_create",175 if reuse else 184,"supplied/reused ContextRef" if reuse else "Init options + CPUDeviceOptions",
               "retain supplied context" if reuse else "Init::getOrCreateContext('max-engine',options)","MAXCTX1",
               "The process-level MAX Context can outlive one session; reference retention avoids recreating shared services.",patch={"max":"MAXCTX1 · reused ContextRef" if reuse else "MAXCTX1 · Init-managed context; typed service store"})
        s.step("Bind the CPU runtime service","ct_context_ctor",217,"MAXCTX1 + runtimeConfig","cpuDevice=*maxContext->get<CPUDevice>()","CPUDevice1; worker facility",
               "Later compile/init tasks use this runtime service. It is separate from the caller's CPU Tensor DeviceRef.",patch={"cpu":"CPUDevice1 · queues/workers/allocator services", "engine":"EngineContext1{maxContext→MAXCTX1,config,cpuDevice→CPUDevice1}"})
        s.step("Attach API telemetry and validate labels","ct_context_create",190,"EngineContext1; configured device labels","emplaceIfMissing telemetry; accept cpu/gpu labels","runtime context accepted",
               "A label outside the supported closed set is rejected; this is not numerical kernel validation.",patch={"max":"MAXCTX1: CPUDevice1, API telemetry(language=python), registered services"})
        s.step("Retain native runtimeContext and enable diagnostics","ct_session_native",563,"successful EngineContext1","unique_ptr context; Python stack callback; optional configured profiler","S1 ready for compile/init",
               "Profiling starts only when configured. Constructing S1 does not compile G1 or allocate its output.",patch={"session":"S1{_impl.devices=[GPU0,CPU0],sessionCompileConfig,runtimeConfig,runtimeContext→EngineContext1}"})
        return s
    add("session-ownership","B.1 · Session configuration and MAX Context","ownership","Is the constructor only about devices, and what does S1 actually retain?",
        "devices=[device]; optional extensions/threads", "S1 with compile config, runtime config and retained native runtime services",
        """**S1 combines compiler configuration and runtime ownership.** Python deduplicates devices, appends CPU if needed, processes optional extensions, and calls the native constructor. Native code clones `sessionCompileConfig`, fills `runtimeConfig` with device specs/context references and thread/API settings, then retains an Engine::Context.

| Retained value | Concrete teaching example | Meaning |
|---|---|---|
| `devices` | `[GPU0,CPU0]` | Driver object references |
| `sessionCompileConfig` | base kernel paths; supplied extensions=[] | Settings for later compilation |
| optional extension | `/tmp/.../P1.mojoc` | Existing Mojo package path, appended if explicitly supplied |
| `runtimeConfig` | threads=0, language=python, devices/specs/ctx | Runtime setup policy |
| `runtimeContext` | EngineContext1→MAXCTX1 and CPUDevice1 | Native runtime services |
| shared MAX Context | typed service store; CPUDevice and telemetry | Shared infrastructure, not graph IR |

**Actual constructor extensions are empty.** P1 was loaded by G1; compile later picks up G1's library paths. Engine context creation can acquire a shared MAX Context or retain one supplied by configuration. This is distinct from the module-level default MLIR Context.""",
        """caller: session=InferenceSession(devices=[device]) [LINK:entry:71]
├─ Python: unique devices; ensure CPU; process optional extension objects [LINK:ct_session:688] [CALLER:entry|71]
└─ native _InferenceSession(final_devices,extensions,num_threads or 0) [LINK:ct_session:707]
   ├─ populateCompileConfig → clone config; append valid .mojoc paths [LINK:ct_session_native:546] [LINK:ct_compile_config:615] [CALLER:ct_session|707]
   ├─ runtimeConfig.setNumThreads; retain specs/device contexts; language=python [LINK:ct_session_native:549] [CALLER:ct_session|707]
   ├─ Engine::Context::create [LINK:ct_context_create:169]
   │  ├─ supplied MAX Context → retain [LINK:ct_context_create:175] [CALLER:ct_session_native|559]
   │  ├─ otherwise Init::getOrCreateContext with CPU runtime options [LINK:ct_context_create:184] [CALLER:ct_session_native|559]
   │  └─ context wrapper stores config/MAXContext/CPUDevice reference [LINK:ct_context_ctor:214] [CALLER:ct_context_create|189]
   └─ retain runtimeContext; stack callback; configured profiling branch [LINK:ct_session_native:563] [LINK:ct_session_native:569] [CALLER:entry|71]""",
        session(),alternatives={"reuse":session(True)},source="entry",line=71)

    nodes=objects(("original","G1 / MODULE1","retained symbolic IR","Python caller"),
                  ("request","CompilationRequest","options + P1 paths","session compiler"),
                  ("clone","C1","changing compiler IR","MOFContext"),
                  ("registry","MojoRegistry / lowering state","declarations + mappings","pass pipeline"),
                  ("routine","native routine plan","MOGG functions","compiler"),
                  ("plan","init / execute plan","MGP primitives","compiler"))
    s=Story(nodes,view="compiler")
    s.step("Collect the graph's libraries and expected weights","compile_async",1021,"G1 with P1 and _weights={}","model.module; extend library paths; expected_weights={}","request sees P1",
           "The graph's custom package reaches compilation even though it was not a session-constructor extension.",patch={"original":"G1:(TQ,TK,TV)→TO; MODULE1 retained", "request":"P1 + defaults; expected_weights={}"},stage="API")
    s.step("Seed kernel declarations into the module","compile_async",1042,"G1._kernel_library analysis","seed_kernel_decls(module.mlir_module)","declarations + opaque mapping available",
           "This avoids repeating expensive package import when the compiler clone already contains seeded declarations.",patch={"registry":"FusedAttention declaration + P1 references"},stage="API")
    s.step("Acquire setup lock and schedule native compilation","ct_compile_module",1292,"module capsule; options","_impl.compile; done callback releases compilation lock","AsyncValue for compilation",
           "The lock serializes module compilation setup; synchronous compile waits for the asynchronous native result.",patch={"request":"native request scheduled on CPU runtime worker"},stage="API")
    s.step("Clone the module for transformation","ct_framework_compile",381,"MODULE1 opaque pointer","ModuleOp::getFromOpaquePointer(source).clone()","C1 compiler clone",
           "Passes can rewrite C1 while the caller retains the symbolic G1. Dense resources may share context-owned backing.",patch={"clone":"C1: MO graph + seeded declarations; distinct operation tree"},stage="MO")
    s.step("Build and execute the pass manager","ct_run_gc",653,"C1 + GCPipelineOptions","gcPipelineImpl(pm,opts); pm.run(C1)","ordered transformation pipeline",
           "These are C++ compiler-library passes in-process, not shell invocations of mlir-opt.",patch={"request":"mlir::PassManager · options and declared paths"},stage="MO")
    s.step("Verify and simplify the MO contract","attention_gc",219,"C1 MO custom graph","buildMoPipeline: parameter/device checks; CSE/canonicalization; optional folds","legal/simplified MO",
           "MO means Modular Operations: graph, tensor, shape/device and custom-call contracts. The actual custom kernel remains opaque.",patch={"clone":"MO: Q/K/V:T → custom(symbol,BD=8,BN=16,gpu:0) → TO\nvalid parameters/device, complete signature"},stage="MO")
    s.step("Understand simplification through an eligible comparison","attention_gc",241,"comparison: repeated pure expression or identity view","CSE merges equal pure ops; canonicalizer folds eligible identities","one shared value instead of duplicate work",
           "Example: two identical pure shape calculations can share one result. This does not imply runtime QK/softmax are folded inside this opaque custom op.",patch={"clone":"Actual: opaque attention call retained\nComparison: shape_expr(n)+duplicate → one reusable SSA value"},stage="MO")
    s.step("Lower MO to access/implementation primitives","mo_to_map",764,"legal MO graph","MOToMAPPass","MAP representation for supported operations",
           "MAP means Modular Access Primitives. Built-in elementwise bodies expose accesses/computation; opaque custom routines keep their declaration-backed implementation boundary.",patch={"clone":"MAP: view/access roles + opaque attention implementation call\ncomparison: y[i]=a[i]+b[i] exposes load/load/add/store"},stage="MAP")
    s.step("Optimize eligible MAP accesses and fusion","map_pipeline",338,"MAP bodies/accesses","FuseKernels; CSE/canonicalize; view motion; outline structured bodies","optimized/outlined implementations",
           "Comparison: y=a+b then z=y*c can share one elementwise body and avoid a temporary. This opaque attention routine is already fused internally, not newly fused from its QK/softmax steps here.",patch={"clone":"MAP optimized · opaque attention boundary retained\ncomparison fused body: load a,b,c → (a+b)*c → store z"},stage="MAP")
    s.step("Use declarations and per-op lowering state","map_to_mogg",57,"MojoRegistry(module); MAP IterOpaqueOp","MAPToMOGGState(registry,op).lowerOp(op)","Mojo-oriented function/call structure",
           "The registry indexes kernel/primitive/shape-function declarations. State tracks conversion of this operation's accesses/captures/arguments into implementation calls.",patch={"registry":f"kernelByName[{NAME}]→FusedAttention\nstate: current op + operand/capture/implementation mappings", "clone":"MOGG · Modular Operation Generator Graph", "routine":"routine arguments: output,Q,K,V,ctx; parameters dtype/rank/BD/BN/target"},stage="MOGG")
    s.step("Refine and outline Mojo routines","mogg_pipeline",370,"MOGG functions/calls","ConformCallOperands; OutlineRoutines; tracing; CSE","stable routine interfaces",
           "Glue supplies output destinations, context and ordering/state arguments around the implementation; Python is not calling Mojo directly at runtime.",patch={"routine":"outlined host entry stub → FusedAttention.execute(output,Q,K,V,ctx)"},stage="MOGG")
    s.step("Lower graph lifecycle to MGP","mogg_to_mgp",2406,"MOGG graph/routines","MOGGToMGPPass / MOGraphLowering","init and execute regions",
           "MGP means Modular Glue Primitives: buffer/state allocation, async dependencies and lifecycle actions that make routines executable.",patch={"clone":"MGP model @fused_attention", "plan":"init: contexts/resources/state slots\nexecute: validate/bind inputs → allocate O → invoke routine → output chain+O"},stage="MGP")
    s.step("Optimize lifecycle work and emit Mojo","mgp_pipeline",382,"MGP model + routine stubs","simplify chains; plan/reuse buffers; finalize device graphs when marked; EmitMojo","Mojo source/native routine definitions",
           "Invariant resource work belongs in init; per-call input/output work belongs in execute. Finalization only acts on marked device graphs present in IR.",patch={"routine":"emitted Mojo for glue/native routines; imports P1", "plan":"init/execute plan with explicit dependencies and state"},stage="MGP")
    s.step("JIT compile routines and lower their runtime calls","jit_entry",529,"emitted routines + P1 + target","JitCompileKernels; buildJitPipeline","ModuleLoad / ExecuteGeneric; callable plans",
           "Code becomes an archive resource; runtime plans become named initialization/execution callables. The detailed JIT card follows compiler/backend transformations.",patch={"routine":"compiled archive resource Rcode1", "plan":"initFn + executeFn; ModuleLoad and ExecuteGeneric primitives"},stage="JIT")
    add("compile-passes","C.1 · MO → MAP → MOGG → MGP","compiler","What changes at each representation, why is it needed, and who performs it?",
        "G1, P1, compile options", "C1 lowered to code resources plus init/execute plans; retained G1",
        """**Graph compilation transforms a clone C1 into an executable plan.** It first obtains P1 paths/seeded declarations, creates the compiler clone, then runs an MLIR PassManager built by GraphCompiler.

| Stage | Concrete question answered | Attention example |
|---|---|---|
| MO: Modular Operations | What graph/type/device/custom contract? | Three f32[N,D] inputs, one output, symbol and BN/BD |
| MAP: Modular Access Primitives | What accesses/implementation boundary? | Custom routine remains opaque; supported built-ins expose views/loads/computations |
| MOGG: Modular Operation Generator Graph | What Mojo routine and arguments? | Declaration resolution; output/query/key/value/context glue |
| MGP: Modular Glue Primitives | What resource/dependency lifecycle? | Init resources, per-call buffers, state slots/chains, invoke/return |
| JIT/runtime lowering | What code resource and callable plans? | Archive, ModuleLoad, ExecuteGeneric, initFn/executeFn |

MO contracts include shape/dtype/device and registered operand/result/parameter compatibility. CSE/canonicalization can simplify eligible pure duplicates or identities; these passes do not inspect the numerical data in this custom call. MAP fusion examples are explicitly **built-in comparisons**, not claims that GraphCompiler decomposes this opaque attention kernel.

MojoRegistry indexes native declarations. MAPToMOGGState carries one operation's conversion context/mappings and emits implementation/glue calls. MGP schedules lifecycle work so repeated execute calls can reuse initialized code/resources with fresh inputs.

For the shared in-depth pass examples, open [the compiler pipeline lesson](max_graph_compiler_pipeline_examples.html). All snippets here are schematics, not captured IR dumps.""",
        """caller: compiled=session.compile(graph) [LINK:entry:74]
├─ compile_async: Graph.module, library paths, expected_weights, seed declarations [LINK:compile_async:1021] [LINK:compile_async:1042] [CALLER:ct_compile|1104]
├─ _compile_module → native request → CPU worker compile [LINK:ct_compile_module:1292] [LINK:ct_compile_module:1294] [LINK:ct_context_compile:229] [CALLER:compile_async|1044]
└─ MOFContext::compileToBinary [LINK:ct_framework_compile:377]
   ├─ clone module → C1 [LINK:ct_framework_compile:381] [CALLER:ct_context_compile_sync|273]
   └─ cache miss: convertMOToMEF → MO::runGCPipeline [LINK:ct_convert_mef:356] [LINK:ct_run_gc:638]
      ├─ gcPipelineImpl creates passes; pm.run executes them [LINK:ct_run_gc:653] [LINK:ct_run_gc:654] [CALLER:ct_convert_mef|361]
      ├─ buildMoPipeline: declarations/parameters, canonicalization, shapes/devices [LINK:attention_gc:219] [CALLER:attention_gc|574]
      ├─ MOToMAPPass → access primitives / opaque boundary [LINK:mo_to_map:764] [CALLER:attention_gc|581]
      ├─ buildMapPipeline → eligible fusion, view motion, outlining [LINK:map_pipeline:338] [CALLER:attention_gc|587]
      ├─ MAPToMOGGPass → MojoRegistry + per-op MAPToMOGGState [LINK:map_to_mogg:57] [LINK:map_to_mogg:66] [LINK:map_to_mogg:68] [CALLER:attention_gc|597]
      ├─ buildMoggPipeline → conform arguments / outline routines [LINK:mogg_pipeline:370] [CALLER:attention_gc|602]
      ├─ MOGGToMGPPass → lifecycle regions/buffers/chains [LINK:mogg_to_mgp:2406] [CALLER:attention_gc|611]
      ├─ buildMgpPipeline → lifetime/memory/dependency plans + emitted Mojo [LINK:mgp_pipeline:382] [CALLER:attention_gc|616]
      ├─ JitCompileKernels → code resources [LINK:jit_entry:529] [CALLER:attention_gc|618]
      └─ buildJitPipeline → ModuleLoad/ExecuteGeneric, init+execute callables [LINK:jit_pipeline:75] [CALLER:attention_gc|620]""",
        s,source="entry",line=74)

    nodes=objects(("source","generated Mojo + P1","parametric declarations","GraphCompiler"),
                  ("params","specialization values","dtype/rank/tiles/target","KGEN compiler"),
                  ("kgen","specialized/lowered IR","MLIR KGEN","Mojo compiler"),
                  ("llvmDialect","LLVM dialect module","MLIR operations","ObjectCompiler"),
                  ("llvmIR","LLVM IR module","llvm::Module","LLVM backend"),
                  ("target","TargetInfo / backend","target description/implementation","ObjectCompiler"),
                  ("deviceIR","GPU offload module","separate device representation","offload compiler"),
                  ("deviceCode","GPU kernel payload","target artifact bytes","host routine resource"),
                  ("archive","native artifact","host archive + device code","compiled model resources"))
    def kernels(amd=False,cache=False):
        s=Story(nodes,label="Kernel-code cache reuse" if cache else "AMD target illustration" if amd else "NVIDIA target illustration",view="compiler-target")
        s.step("Choose host target plus accelerator architecture","attention_jit_target",154,"host triple/CPU/features; accelerator override or probe","getTargetInfoFor(host...,acceleratorArch)","host target description plus selected GPU architecture",
               "An accelerator override changes the GPU description, while the ordinary host triple remains the CPU triple.",patch={"target":"host: CPU triple/arch/features\naccelerator: gfx942 (illustrative)" if amd else "host: CPU triple/arch/features\naccelerator: sm_80 (illustrative)"},stage="target setup")
        s.step("Read emitted routine source and imports","jit_entry",547,"MOGG emitted Mojo; P1 paths","parseMojoText(ctx,maxContext,opts)","parsed parametric Mojo module",
               "This uses compiler APIs in-process. Source-package precompile earlier was a different subprocess.",patch={"source":"generated host routine + FusedAttention package P1"},stage="Mojo")
        s.step("Make concrete parameter/target information","jit_entry",566,"Q/K/V type and graph device","KGENCompiler; runLITToKGEN; setTargetInfo","dtype=f32,rank=2,BN=16,BD=8,target=gpu",
               "Library generation creates KGEN generators. Full elaboration follows a routine-cache miss; these input specialization values are not yet target machine code.",patch={"params":"f32,rank2,BN16,BD8,gpu · input specialization values", "kgen":"parametric KGEN generators + concrete call arguments"},stage="library IR")
        s.step("Create the host object compiler and resolve its backend","backend_select",146,"ordinary opts.targetTriple=host CPU triple","ObjectCompiler::create → TargetBackendRegistry.lookup(host triple)","host backend selected up front",
               "The compiler fails here if this build has no backend for the host triple. GPU code will receive a separate offload compiler.",patch={"target":"HostBackend for CPU triple; accelerator description retained separately"},stage="target setup")
        s.step("Check routine code-cache entries","kernel_cache",248,"KGEN generators, call arguments and target options","semantic-hash cache lookup","cached symbols" if cache else "cache-miss symbol worklist",
               "A code-cache hit avoids new code generation. This is distinct from loading a supplied .mojoc file or reusing declaration analysis.",patch={"archive":"cached resource matches routines/target" if cache else "pending artifact · misses require codegen"},stage="cache")
        if not cache:
            s.step("Elaborate cache-miss routines","attention_jit_misses",440,"parametric KGEN + concrete dtype/rank/tiles/target","compiler.runElaborationPipeline(...)","specialized host routines and discovered offloads",
                   "JIT means code is compiled during session.compile, before numerical execution. The GPU specialization eliminates the CPU branch and exposes its device-function compilation requests.",patch={"kgen":"elaborated host glue; GPU function offload requests discovered"},stage="elaborate")
            s.step("Group and slice GPU offloads by target","attention_offloads",610,"offload requests, exported symbols and target grouping","produceStandaloneModule; stamp the group's TargetInfo","separate GPU module D1",
                   "The device module contains the GPU entry and reachable dependencies. The retained host routine still contains orchestration and the device payload/load interface.",patch={"deviceIR":"D1: fused_attention_kernel + dependencies; GPU target, distinct from host module"},stage="device backend")
            s.step("Select the GPU backend from its target triple","attention_offloads",720,"per-group target triple/arch/features","separate ObjectCompiler::create; registry lookup for that triple","AMDGPU backend" if amd else "NVPTX backend",
                   "The same registry API selects a different implementation because this compiler's options contain the device triple.",patch={"target":"HostBackend retained\nD1: amdgcn-amd-amdhsa / gfx942 → AMDGPU" if amd else "HostBackend retained\nD1: nvptx64-nvidia-cuda / sm_80 → NVPTX"},stage="device backend")
            s.step("Lower the device module through MLIR LLVM operations","attention_host_lower",835,"elaborated D1 + GPU backend","buildLowerToLLVMPipeline; mgr.run(D1)","device LLVM dialect module",
                   "The common lowering library runs for the selected device target. Its llvm.* operations still belong to MLIR.",patch={"deviceIR":"D1: MLIR LLVM dialect · target intrinsics, loads, arithmetic, control flow"},stage="device LLVM")
            s.step("Translate device LLVM dialect to LLVM IR","llvm_translation",807,"D1 LLVM dialect","mlir::translateModuleToLLVMIR(D1,...)","device llvm::Module",
                   "This changes representation APIs; it does not copy tensor data or load the GPU driver.",patch={"deviceIR":"D1_LLVM: llvm::Module for GPU triple"},stage="device LLVM")
            s.step("Emit the device kernel payload","attention_offloads",831,"D1 + requested emission kinds","compiler.emitOffloadKernels(module,kernelEmissionKinds)","GPU payload associated with kernel ID",
                   "NVPTX and AMDGPU use their own target emission paths. PTX/CUDA images or AMD objects/code objects are artifact examples; this browser does not compile them.",patch={"deviceCode":"GPU payload bytes + kernel ID → returned to host elaboration"},stage="device artifact")
            s.step("Lower host routines through MLIR's LLVM dialect","attention_host_lower",835,"elaborated host module with device payload references","buildLowerToLLVMPipeline; mgr.run(host module)","host llvm.* operations in MLIR",
                   "LLVM dialect is still MLIR. It models low-level arithmetic/control/memory operations; it is not yet llvm::Module.",patch={"llvmDialect":"host MLIR: llvm.func / llvm.load / llvm.call / arithmetic"},stage="host LLVM dialect")
            s.step("Translate host MLIR to an LLVM IR module","llvm_translation",807,"host MLIR LLVM dialect","translateModuleToLLVMIR","host llvm::Module",
                   "This translation crosses representation APIs. LLVM's target code generator consumes LLVM IR.",patch={"llvmIR":"host LLVM IR functions/instructions · llvm::Module"},stage="host LLVM IR")
            s.step("Emit target artifacts and the host archive","ct_emit_archive",1023,"LLVM IR + selected target backend","emit code; collect host objects/device payloads into resource","native code resource Rcode1",
                   "NVIDIA device payload and AMD device payload are separate target artifacts; the browser does not compile or execute them.",patch={"archive":"Rcode1: host archive + AMD device payload" if amd else "Rcode1: host archive + NVIDIA device payload"},stage="host artifact")
        else:
            s.step("Reuse compiled routine resources","kernel_hits",507,"hit symbols/artifact references","attach previously compiled routines","Rcode1 reused; no new LLVM lowering",
                   "Output tensors are still new per-call buffers; reusing code does not reuse the Q/K/V data. Native callable addresses are resolved later during initialization.",patch={"archive":"Rcode1 reused · matching archive bytes and symbol names"},stage="host artifact")
        s.step("Lower compiled routines to runtime primitives","attention_runtime_lower",102,"Rcode1 + callable symbols","LowerCompiledRoutines","ModuleLoad in init; ExecuteGeneric in execute",
               "Runtime initialization loads/link-resolves host code. Execution later reaches device-module/driver calls.",patch={"archive":"Rcode1 + named symbols; ModuleLoad/ExecuteGeneric plan"},stage="runtime plan")
        return s
    add("compile-kernels","C.2 · Mojo specialization, LLVM and target backend","compiler-target","Which compiler code specializes kernels, translates LLVM dialect, and selects CPU/NVIDIA/AMD output?",
        "Generated routines + P1; concrete types/tiles/target", "Compiled code resource or matching cached routines",
        """**The graph compiler invokes the Mojo compiler library for JIT routines.** JIT means code is compiled when the application needs a compiled model, instead of being fully built ahead of time into the Python example target. `session.compile(G1)` is the example JIT path; subsequent execute calls reuse M1's initialized code.

Library generation first creates parametric KGEN generators. The host ObjectCompiler selects its CPU backend **before** routine-cache lookup. A cache miss elaborates the generators. During elaboration, device offload requests are grouped and sliced into a separate target module with its own ObjectCompiler/backend. Device lowering/emission returns a GPU payload; host lowering then builds the native archive containing the host routines and payload references.

Both ordinary CPU and NVPTX/AMDGPU paths use LLVM-dialect lowering followed by LLVM IR translation. These are two distinct representations: MLIR `llvm.*` operations, then `llvm::Module`. Architecture/features configure target generation; the triple selects the backend implementation. Host and device modules do not share a single CPU/GPU machine-code target.

| Target illustration | Triple family | Code ownership |
|---|---|---|
| Host CPU | host triple (e.g. x86_64 Linux) | Python/MAX/driver and generated host routines |
| NVIDIA GPU | nvptx64-nvidia-cuda, example sm_80 | Separate device target payload |
| AMD GPU | amdgcn-amd-amdhsa, example gfx942 | Separate device target payload |

Architectures are example values, not detected hardware. The code-cache selector reuses native routine resources. An existing .mojoc path skips source precompilation; a deterministic source-package output path alone is not a cache-hit check. **No per-kernel `mojo build` or standalone `mlir-opt` shell sequence is needed in this JIT route.**""",
        """caller: JitCompileKernelsPass compiles emitted routines [LINK:jit_entry:529]
├─ target options: host triple plus accelerator architecture [LINK:attention_jit_target:154] [CALLER:jit_entry|534]
├─ parseMojoText with P1 imports; stamp target info [LINK:jit_entry:547] [LINK:jit_entry:556] [CALLER:jit_entry|547]
├─ KGENCompiler → runLITToKGEN/library generation [LINK:jit_entry:566] [LINK:jit_entry:567] [CALLER:jit_entry|567]
├─ host ObjectCompiler::create(isJit=true) → select host backend now [LINK:backend_select:146] [CALLER:jit_entry|575]
├─ code-cache lookup; hits later attach existing archive resources [LINK:kernel_cache:248] [LINK:kernel_hits:507] [CALLER:jit_entry|586]
└─ cache miss → handleCacheMisses [LINK:attention_jit_misses:436]
   ├─ KGENCompiler::runElaborationPipeline [LINK:kgen_elaboration:1140] [CALLER:attention_jit_misses|440]
   │  └─ discovered GPU offloads → group/slice by TargetInfo [LINK:attention_offloads:523] [LINK:attention_offloads:610]
   │     ├─ device triple/arch/features options; GPU ObjectCompiler/backend [LINK:attention_offloads:535] [LINK:backend_select:146] [CALLER:attention_offloads|720]
   │     ├─ device KGEN → LLVM dialect → LLVM IR [LINK:attention_host_lower:835] [LINK:llvm_translation:807] [CALLER:attention_offloads|831]
   │     └─ emit device kernel payloads keyed by kernel ID [LINK:attention_offloads:831] [CALLER:attention_offloads|831]
   ├─ host emitArchive → LLVM dialect lowering + LLVM IR translation [LINK:ct_emit_archive:1023] [LINK:attention_host_lower:835] [LINK:llvm_translation:807] [CALLER:attention_jit_misses|457]
   ├─ attach host archive and callable symbol names to init resource [LINK:attention_jit_misses:493] [CALLER:attention_jit_misses|493]
   └─ later LowerCompiledRoutines → ModuleLoad / ExecuteGeneric [LINK:attention_runtime_lower:102] [CALLER:jit_pipeline|76]""",
        kernels(),alternatives={"amd":kernels(True),"hit":kernels(cache=True)},source="jit_entry",line=529)

    nodes=objects(("ir","C1","lowered compiler clone","MOFContext"),
                  ("code","Rcode1","compiled archive resource","MEF container"),
                  ("mef","MEF bytes","serialized model executable format","compiled artifact"),
                  ("models","model table","native metadata/interfaces","MEF reader"),
                  ("async","compile AsyncValue","pending/ready/error","CPU runtime"),
                  ("python","CM1","CompiledModel wrapper","caller"))
    s=Story(nodes,view="ownership")
    s.step("Finish lowered graph and code resources","ct_convert_mef",356,"C1 + Rcode1 + init/execute callables","convertMOToMEF","serializable executable model",
           "This packages code and plans, not random Q/K/V matrices.",patch={"ir":"C1 · finalized model/callable plan", "code":"Rcode1 · archive/symbol resource"})
    s.step("Serialize the executable container","ct_convert_mef",371,"lowered model/resources","convertMGPToMEF","MEF bytes",
           "MEF is the compiled artifact container. It is different from .mojoc package IR.",patch={"mef":"MEF bytes · code/resources/model metadata"})
    s.step("Open the compiled model metadata","ct_mef_open",29,"MEF BufferRef","MEF model reader","native models/table",
           "Opening the artifact does not bind external weights, instantiate device state or run the attention kernel.",patch={"models":"fused_attention: inputs TQ/TK/TV; output TO; initFn/executeFn"})
    s.step("Resolve the asynchronous compilation result","ct_context_compile",247,"CompiledModels or compilation error","emplace result; done callback releases lock","ready CompiledModels",
           "The CPU worker reports failure through the same AsyncValue path.",patch={"async":"ready · native CompiledModels"})
    s.step("Synchronous compile waits for the native result","ct_compile",1112,"compiled._compiled AsyncValue","handle.wait()","native compilation is complete",
           "The Python API is synchronous here even though native compilation was scheduled asynchronously.",patch={"python":"CM1{_compiled=ready handle,_graph_names=('fused_attention',),expected_weights={}}"})
    s.step("Check the exception gate and return CM1","ct_compile",1115,"handle.exception() is None","return compiled","CM1 returned on success",
           "An error follows the exception branch instead of returning an executable artifact.",patch={"async":"ready · checked; no compilation error"})
    s.step("Require a separate init before execute","ct_init",1126,"CM1","session.init(CM1)","next phase binds runtime resources",
           "CompiledModel is not executable Model M1. It can be retained/exported independently of initialized device allocations.",patch={"async":"compile complete · model setup still pending"})
    add("compile-mef","C.3 · Compiled artifact CM1 and MEF","ownership","What does compile return, and what still needs initialization?",
        "Lowered C1, native code resources and plans", "Ready CM1/MEF metadata; no initialized M1 yet",
        """**CM1 is a compiled artifact, not the initialized executable model.** It retains the native compiled handle plus expected-weight metadata and graph names. This custom example has expected_weights={} because its three matrices are per-call inputs.

MEF packages the model interface, code/resources and callable plans. .mojoc instead packages parametric Mojo declarations. Synchronous compile waits for its native AsyncValue and raises on error. Init must still resolve contexts, resource symbols, runtime state and initialization dependencies.""",
        """caller: compiled=session.compile(graph) [LINK:entry:74]
├─ convertMOToMEF → executable container [LINK:ct_convert_mef:356] [CALLER:ct_framework_compile|414]
├─ CompiledModels opens MEF bytes and enumerates models [LINK:ct_mef_open:29] [LINK:ct_mef_models:901] [CALLER:ct_compiled_create|100]
├─ CPU worker resolves CompiledModels AsyncValue [LINK:ct_context_compile:247] [CALLER:ct_context_compile|229]
└─ Python handle.wait; exception gate; return CM1 [LINK:ct_compile:1112] [LINK:ct_compile:1113] [LINK:ct_compile:1115]
   └─ next call is session.init(CM1), before model.execute [LINK:entry:75] [CALLER:entry|74]""",
        s,source="entry",line=74)

    nodes=objects(("weights","weight bindings","registry reference","init caller/native wrapper"),
                  ("task","init task","CPU runtime work","EngineContext1"),
                  ("tls","current CPUDevice","thread-local state","AsyncRT worker"),
                  ("devices","DeviceRef→context mapping","runtime handles","Model::setup"),
                  ("code","Rcode1 host symbols","resident callable addresses","CompiledKernelsContext"),
                  ("state","MGP state / extraArgs","runtime resources","ExecutableModel"),
                  ("model","M1","initialized executable owner","Python Model"))
    s=Story(nodes,view="ownership")
    s.step("Create the real-device weight registry","init_all",1202,"CM1.expected_weights={}; weights_registry omitted","weights_registry_real = weights_registry or {}","WR1={}",
           "Q/K/V are not weights and are not supplied to init.",patch={"weights":"WR1={} · real empty registry"})
    s.step("Check expected persistent weight bindings","init_all",1209,"expected_weights={}; WR1={}","validate required names/devices; check supplied tensors for contiguity","no missing or invalid weights",
           "Both maps are empty for this example. In a weighted graph, these checks run before any native model setup.",patch={"weights":"WR1={} · validation accepted"})
    s.step("Enter native load_all with the compiled handle","init_all",1244,"CM1._compiled; WR1","self._impl._load_all(compiled._compiled, weights_registry_real)","native model-loading route",
           "Resolved artifacts may first be wrapped as an AsyncValue for reuse; this call loads all model descriptors.",patch={"task":"native loadAll · waiting for compiled descriptors"})
    s.step("Retain the managed weight owner and await compiled metadata","ct_load_all",293,"WR1 dictionary; CompiledModels handle","buildWeightsFromDict; await compiledModels; enumerate descriptors","one descriptor: fused_attention",
           "An empty registry still has an owner that outlives the borrowed pointer passed to the init callable.",patch={"weights":"shared WeightsRegistry WR1={} + managedTensor owners=[]"})
    s.step("Request initialization of the model","ct_context_init",297,"MEF Model description; WR1","allocate AsyncValue; addTask(cpuDevice,...)","CPU init task pending",
           "Setup may invoke runtime primitives that require the AsyncRT CPUDevice on the current worker thread.",patch={"task":"init(Model fused_attention,WR1) queued"})
    s.step("Run on a CPU runtime worker with valid TLS","ct_context_init",294,"CPUDevice1 worker","worker supplies current AsyncRT CPUDevice","TLS→CPUDevice1",
           "MGP_RT_BufferBorrowV2, used during setup, requires this current CPUDevice; this is why a worker is needed even for GPU tensor targets.",patch={"tls":"worker.currentCPUDevice=CPUDevice1", "task":"running Model::setup"})
    s.step("Map model DeviceRefs to configured handles","ct_model_setup",197,"model devices + session DeviceMap","mapDeviceRefsToDevices","init arguments for selected contexts",
           "A graph's gpu:0 label must resolve to a retained runtime device/context.",patch={"devices":"cpu:0→CPU runtime; gpu:0→ctxGPU (when GPU graph)"})
    s.step("Prepare device stream capacity","ct_model_setup",212,"device contexts; stream-count default2","ensureStreamCountForDevices","base+side stream capacity ready",
           "Streams are created before capture begins; actual stream scheduling remains a runtime boundary.",patch={"devices":"ctxGPU · configured stream capacity (default 2)"})
    s.step("Invoke the compiled initialization callable","ct_model_setup",220,"initArgs=[borrowed WR1 ref, device contexts...]","initFn.invoke(initArgs,cpuDevice)","chain, MGP state, weights and extra resources",
           "The init plan prepares resources once, enabling repeated execute calls to bind fresh inputs.",patch={"state":"pending [chain,MGPContext,weights,extraArgs…]"})
    s.step("Load an archive as a host code resource","ct_kernel_load",85,"archive bytes + symbol names","CompiledKernelsContext::load","host execution-engine resource",
           "This loader links already compiled code. It does not parse .mojo or compile a new GPU kernel.",patch={"code":"Rcode1 bytes → archive hash/H1 dylib"})
    s.step("Add or reuse the archive by its hash","ct_kernel_load",160,"H1 archive bytes","engine.addIfAbsent<StaticArchiveLayer>(H1,buf)","one execution-engine dylib for this archive",
           "Repeated initialization can reuse the same resident code. This ordinary route does not run the debug/profiler shared-library linker.",patch={"code":"H1 → resident host archive; reused when already present"})
    s.step("Resolve each named callable","ct_kernel_load",166,"H1 + kernelNames","engine.lookup(H1,kernelName); addKernel(name,index,functionPointer)","host callable pointers stored",
           "Pointer addresses here are symbolic examples. GPU device module/function handles are obtained later by host routines reaching device APIs.",patch={"code":"symbol→HOST_FN1 (resident native callable); code resource retained"})
    s.step("Construct the executable owner","ct_model_setup",237,"init results[0..2] + extra results","ExecutableModel(chain,mgpContext,weights,extraArgs,model,cpuDevice)","M1 native executable",
           "It retains initialization dependencies, state/resources and underlying model/callable references, not a copy of Q/K/V.",patch={"state":"MGPContext1 + WR1 reference + resource extraArgs", "model":"ExecutableModel1{chain=CHinit,mgpContext,weights,extraArgs,model,cpuDevice}"})
    s.step("Await all deferred setup resources on the worker","ct_wait_setup",388,"CHinit, MGPContext1, weights and extraArgs","AsyncRT::await(fixed results); await(extraArgs); inspect errors","all init results ready or diagnostic",
           "createExecutableModel performs this wait before returning the owner. It is not postponed until the first execute call.",patch={"state":"CHinit, MGPContext1, WR1 and extraArgs · ready; checked"})
    s.step("Publish the ready executable to the caller","ct_context_init",304,"successful createExecutableModel result","result.emplace(executableModel)","ready init AsyncValue",
           "The outer load releases the Python GIL while awaiting this result; failures publish an error instead.",patch={"task":"worker done; setup AsyncValue ready"})
    s.step("Retain the native model wrapper and owners","ct_load_all",312,"compiled handle, native session, executable, shared weights","ModelHandle(...); cache input/output device vectors","M1 native handle",
           "These references preserve the descriptor, callable code and resource lifetimes for repeated execution.",patch={"model":"ModelHandle{compiledModels,session,ExecutableModel1,WR1,managedTensors,index=0}"})
    s.step("Name the returned Python model","init_all",1247,"native vector with M1","result={m.name:m}; reject duplicate names","{'fused_attention':M1}",
           "init_all can return several named models. This compiled example has exactly one.",patch={"task":"native setup finished; named model mapping available"})
    s.step("Return the one-model Python wrapper","ct_init",1158,"models={'fused_attention':M1}","verify one model; return first value","model=M1",
           "Init's successful wrapper is separate from CM1; execute still honors initialization readiness.",patch={"model":"Python Model M1 → native model handle / initialized executable"})
    add("init-resources","D.1 · Model setup, host code and runtime state","ownership","What does initialization construct internally, and why run setup on a CPU worker?",
        "CM1; empty WR1; runtime contexts", "M1 with init chain, MGP state, retained code/weights/extraArgs",
        """**Model setup runs a compiled init plan and retains its resource results.** The CPU worker supplies AsyncRT's current CPUDevice in thread-local state; setup primitives such as `MGP_RT_BufferBorrowV2` require it. GPU targeting does not remove this host requirement.

`Model::setup` maps DeviceRefs, prepares stream capacity, invokes initFn and checks immediate errors. The first returned values are the initialization chain, MGP context and weight reference; remaining values become extraArgs. `ExecutableModel` retains these along with the model description and CPUDevice reference. **The worker then waits for all setup results before publishing M1.** Deferred setup errors also stop initialization.

`CompiledKernelsContext::load` hashes already-compiled archive bytes, loads them once in an execution-engine dylib and resolves named host symbols. **It loads host callable pointers; GPU handles/tensor pointers are different objects.** Device module/function/launch work is reached by later host execution. Open [the native loader lesson](max_compiled_kernels_context_load_and_driver.html) for archive linking and driver details.""",
        """caller: model=session.init(compiled) [LINK:entry:75]
├─ init → init_all: create real empty registry; validate bindings [LINK:ct_init:1151] [LINK:init_all:1202] [LINK:init_all:1209] [CALLER:entry|75]
├─ native loadAll: managed weights; compiled-descriptor await; one model [LINK:ct_load_all:293] [LINK:ct_load_all:300] [LINK:ct_load_all:309] [CALLER:init_all|1244]
└─ Engine::Context::initModel [LINK:ct_context_init:288]
   ├─ allocate result AsyncValue; addTask(cpuDevice,...) [LINK:ct_context_init:297]
   │  └─ worker provides current CPUDevice TLS for setup primitives [LINK:ct_context_init:294] [CALLER:ct_context_init|297]
   └─ createExecutableModel → Model::setup [LINK:ct_executable_create:79] [LINK:ct_model_setup:184]
      ├─ borrowed weights reference + DeviceRef mapping [LINK:ct_model_setup:193] [LINK:ct_model_setup:197] [CALLER:ct_executable_create|86]
      ├─ ensure device stream count, default2 [LINK:ct_model_setup:212] [CALLER:ct_executable_create|86]
      ├─ initFn.invoke [LINK:ct_model_setup:220]
      │  └─ ModuleLoad → CompiledKernelsContext::load(archive,symbols) [LINK:ct_module_load:55] [LINK:ct_kernel_load:85] [CALLER:ct_model_setup|220]
      │     ├─ archive hash → addIfAbsent<StaticArchiveLayer> [LINK:ct_kernel_load:91] [LINK:ct_kernel_load:160] [CALLER:ct_kernel_load|160]
      │     └─ engine.lookup → resident host pointers; addKernel [LINK:ct_kernel_load:166] [LINK:ct_kernel_load:169] [CALLER:ct_kernel_load|166]
      ├─ construct ExecutableModel(chain,MGPContext,weights,extraArgs) [LINK:ct_model_setup:237] [CALLER:ct_executable_create|86]
      ├─ waitForModelSetup → await fixed/extra results; propagate errors [LINK:ct_wait_setup:388] [CALLER:ct_executable_create|91]
      └─ worker publishes ready executable → retained ModelHandle → Python M1 [LINK:ct_context_init:304] [LINK:ct_load_all:312] [LINK:ct_init:1158] [CALLER:init_all|1244]""",
        s,source="entry",line=75)

    nodes=objects(("artifact","CM1","compiled model metadata","caller"),
                  ("weights","weight gate","expected/provided maps","init_all"),
                  ("setup","setup result","AsyncValue<ExecutableModel>","CPU worker"),
                  ("chain","CHinit","async dependency","ExecutableModel"),
                  ("error","diagnostic gate","ready/error","runtime"),
                  ("execute","execute admission","callable invocation","M1"))
    def gates(error=False):
        s=Story(nodes,label="Comparison · setup failure" if error else "Actual · readiness succeeds",view="gates")
        s.step("Check expected weight bindings","init_all",1209,"CM1 expected_weights={}; supplied={}","init_all validates maps","empty real bindings accepted",
               "This example needs no model-weight tensors. Virtual/stand-in weights used for compile-only workflows do not authorize execution.",patch={"artifact":"CM1: graph_names=('fused_attention',)","weights":"expected={} supplied={} · PASS"})
        s.step("Track the asynchronous setup result","ct_context_init",292,"Model description + contexts","allocate result and schedule setup","pending setup AsyncValue",
               "Task completion and init-chain completion are separate states to await.",patch={"setup":"pending · queued on CPU runtime"})
        s.step("Construct the owner of init results","ct_model_setup",237,"initFn results: chain,context,weights,extraArgs","construct ExecutableModel","owner exists; deferred resources may still be pending",
               "An owner is not yet the successfully initialized model returned by Python init.",patch={"setup":"ExecutableModel constructed inside worker", "chain":"CHinit pending", "error":"immediate setup results accepted"})
        s.step("Wait for initialization dependencies inside setup","ct_wait_setup",388,"CHinit, MGPContext, weights and extraArgs","await fixed results and extraArgs","resources resolved",
               "createExecutableModel calls waitForModelSetup before publishing the executable. Execution is not the first place that checks setup readiness.",patch={"chain":"CHinit error" if error else "CHinit ready", "error":"deferred context/resource error (comparison)" if error else "deferred resources ready"})
        s.step("Inspect deferred errors before returning the owner","ct_wait_setup",390,"resolved init results","check chain/context/weights/extraArgs error states","diagnostic" if error else "success",
               "A failed deferred resource propagates through createExecutableModel and prevents publication of a successful executable.",patch={"setup":"error diagnostic" if error else "setup complete", "error":"ERROR: deferred initialization resource failed (comparison)" if error else "no deferred initialization error"})
        s.step("Publish the checked setup outcome","ct_context_init",302 if error else 304,"createExecutableModel result","setUnknownLocError" if error else "result.emplace(executableModel)","error" if error else "ExecutableModel ready",
               "Only the fully checked successful route can produce the Model returned by init.",patch={"setup":"failed init AsyncValue" if error else "ready init AsyncValue"})
        if not error:
            s.step("Await the initialization task before returning M1","ct_load",282,"init AsyncValue","Engine::Context::await with Python GIL released","init returns M1",
                   "The Python init call completes after worker setup and deferred-resource checks, not while those checks remain pending.",patch={"setup":"M1 returned with checked setup resources"})
            s.step("Await the executable before invocation","ct_execute_driver",204,"loadedModel AsyncValue","awaitOrError(model)","M1 acquired",
                   "Execution cannot dereference a pending or failed executable result.",patch={"execute":"M1 acquired; initialization chain must be honored"})
            s.step("Invoke only after prerequisites are satisfied","ct_execute_callable",384,"ready resources + fresh input AsyncValues","executeFn.invoke","execution result chain and output",
                   "Admission/readiness is host control. The attention arithmetic is in the selected kernel target.",patch={"execute":"invoke allowed"})
        else:
            s.step("Initialization raises before M1 is returned","ct_load",282,"failed init AsyncValue","Engine::Context::await returns error; load throws","no model available to execute",
                   "The caller's session.init fails, so the later model.execute line is not reached.",patch={"execute":"init failed; M1 not returned; no driver launch"})
        return s
    add("init-gates","D.2 · Initialization and readiness gates","gates","What must be ready before execute, and where are errors surfaced?",
        "Expected weights, setup AsyncValue, initialization chain", "Ready executable/resources or a propagated diagnostic",
        """**Three checks serve different purposes:** weight binding validates persistent parameters; setup completion produces the executable owner; initialization-chain readiness validates that resource work has finished successfully.

Empty bindings are valid for this custom graph because Q/K/V are invocation inputs. `Model::setup` first constructs an owner of init results; **createExecutableModel then waits for fixed and extra results and checks errors before the worker publishes the executable.** Native load awaits that task before Python init returns M1. A later executeFromDriver await acquires this already-checked executable; the initial resource wait is not deferred to execute. The failure selector shows a deferred setup error, not a failed GPU run.""",
        """caller: session.init(CM1), then model.execute(Q,K,V) [LINK:entry:75] [LINK:entry:82]
├─ init_all checks weight bindings and loads models [LINK:init_all:1209] [LINK:init_all:1244] [CALLER:ct_init|1151]
├─ worker constructs ExecutableModel from init results [LINK:ct_model_setup:237] [CALLER:ct_executable_create|86]
├─ waitForModelSetup awaits/checks chain,context,weights,extraArgs [LINK:ct_wait_setup:388] [LINK:ct_wait_setup:390] [CALLER:ct_executable_create|91]
├─ CPU task publishes checked executable or diagnostic [LINK:ct_context_init:302] [LINK:ct_context_init:304] [CALLER:ct_context_init|299]
├─ native load awaits init result before returning M1 [LINK:ct_load:282] [CALLER:ct_load_all|311]
├─ later executeFromDriver awaits loaded executable [LINK:ct_execute_driver:204] [CALLER:ct_execute_prepared|530]
└─ ready executable invokes executeFn with retained state + fresh input refs [LINK:ct_execute_callable:384] [CALLER:ct_execute_driver|212]""",
        gates(),alternatives={"error":gates(True)},source="entry",line=75)

    nodes=objects(("model","M1 / resident code","retained executable","session runtime"),
                  ("q","BQ1","real Buffer storage","caller/runtime refs"),
                  ("k","BK1","real Buffer storage","caller/runtime refs"),
                  ("v","BV1","real Buffer storage","caller/runtime refs"),
                  ("args","native invocation arguments","AsyncValue refs","execute plan"),
                  ("device","GPU context/function/stream","driver handles","host runtime"),
                  ("output","BO1","real output Buffer","returned list"),
                  ("chain","result ordering chain","async dependency","native result list"))
    s=Story(nodes,view="buffers")
    s.step("Retain the initialized model and code","entry",75,"S1 + CM1","init returns M1","M1 resources available",
           "The same executable can run again with fresh Q/K/V buffers.",patch={"model":"M1 → init state + resident host code/device payload"})
    for key,line in [("q",78),("k",79),("v",80)]:
        s.step(f"Create numerical {key.upper()} storage","entry",line,"seed123 NumPy random float32[N,D]","Buffer.from_numpy(...).to(device)",f"B{key.upper()}1:[32,32] gpu:0,4096 bytes",
               "These are actual tensor elements. .to(device) is the explicit transfer/device placement step, before execute.",patch={key:f"B{key.upper()}1 · ptr=p{key.upper()} · f32[32,32] · 4096 bytes @gpu:0"})
    s.step("Validate contiguity and retain input order","ct_execute_py",124,"[BQ1,BK1,BV1]","Buffer inputs retained directly; no DLPack/scalar conversion needed","input_impls=[BQ1,BK1,BV1]",
           "The wrapper checks supported inputs but does not copy them to CPU.",patch={"args":"ordered Buffer refs [BQ1,BK1,BV1]"})
    s.step("Validate the compiled input interface","ct_execute_native",461,"3 actual Buffers + TQ/TK/TV","validateInputSpecList; verifyInputDevice","count/dtype/static shape/device accepted",
           "Same shape on all three ports still requires correct positional binding Q,K,V.",patch={"args":"three compatible input specs/devices · validated"})
    s.step("Create ready tensor wrappers sharing storage","ct_execute_native",484,"buffer owners,pointers,byte sizes","Driver::prepareInput → TensorBufferRef/AsyncValue","ready Q/K/V AsyncValues",
           "This creates runtime wrappers retaining original storage. It is not the earlier Python→MO conversion and does not perform another transfer.",patch={"args":"AQ→BQ1(pQ); AK→BK1(pK); AV→BV1(pV) · shared storage"})
    s.step("Release the GIL and acquire the executable","ct_execute_prepared",529,"M1 + ready input refs","executeFromDriver; await initialized executable","native execution route",
           "Native host work can proceed without holding the Python GIL.",patch={"model":"M1 acquired; init dependencies checked"})
    s.step("Assemble retained state plus fresh input arguments","ct_execute_callable",373,"M1 state and AQ/AK/AV","passedArgs=[chain,MGPContext,weights,extraArgs,…inputs]","executeFn invocation arguments",
           "The runtime ABI includes implicit resource/dependency arguments that are not user tensor ports.",patch={"args":"[CHinit,MGPContext1,WR1,extraArgs…,AQ,AK,AV]"})
    s.step("Invoke the compiled host execute callable","ct_execute_callable",384,"passedArgs; current CPUDevice1","getExecuteFn()->invoke","host routine supplies output + context",
           "Host orchestration allocates/binds the output destination and calls the specialized implementation.",patch={"output":"BO1:f32[32,32]@gpu:0 · output destination", "device":"ctxGPU + resident target payload"})
    s.step("Reach the selected GPU driver route","attention_launch",403,"kernel specialization; Q/K/V/O tensor views","ctx.enqueue_function(kernel_func,...)","grid=(4,2),block=32 launch",
           "Host code obtains/uses device function/stream handles; device code computes tiled attention. Vendor API behavior is a source boundary, not measured here.",patch={"device":"device module/function handles + stream · host submit", "chain":"pending device work dependency"})
    s.step("Complete tiled attention and satisfy readiness","attention_output",378,"per-block final O_i tiles","store O tiles; runtime completion dependency","BO1 elements ready",
           "The browser's tiny reference demonstrates arithmetic only; actual GPU completion/timing was not measured.",patch={"output":"BO1 populated by selected kernel", "chain":"native results=[CHdone,BO1]; readiness satisfied"})
    s.step("Extract user results and keep their device","ct_execute_results",159,"native [chain,Tensor result]","drop first chain; construct DeviceBuffer/spec","user tensor result BO1",
           "The internal chain is omitted from the Python list. BO1 remains on the selected device.",patch={"chain":"CHdone omitted from user results", "output":"Buffer BO1:f32[32,32]@gpu:0"})
    s.step("Return the list exactly as the script uses it","ct_execute_prepared",538,"driver results","extractPyBufferResults → output=[BO1]; print(output)","one-element Python Buffer list",
           "The original script does not select [0], perform .to(CPU()), print full numeric values or assert parity.",patch={"output":"output=[BO1] · caller prints list", "model":"M1/code/state retained for next invocation"})
    add("example","E · Buffers, execution and output","buffers","How do real Q/K/V values enter M1, and what comes back?",
        "M1; three contiguous real Buffers on the selected device", "[BO1] on that device; code/state retained",
        """**Graph q/k/v were symbolic. Execute BQ1/BK1/BV1 hold actual float32 elements.** The script creates seeded random NumPy matrices, wraps them as Buffers and explicitly places them with `.to(device)`.

Python/native validation checks contiguity, type/count/static dimensions and devices. `prepareInput` retains original storage in runtime Tensor AsyncValues; the compiled callable receives implicit state/dependency arguments plus those input refs. Host code dispatches the specialized CPU/GPU implementation, runtime checks completion, and result extraction drops the internal chain.

Actual GPU shapes are [32,32] with 4096 bytes each; CPU [8,8] with 256 bytes. Addresses pQ/pK/pV and handles are examples. The returned value is **a list `[Buffer]`**, not a TensorValue or printed numerical matrix. The script has no explicit final CPU transfer or parity assertion.""",
        """caller: output=model.execute(Q,K,V) [LINK:entry:82]
├─ Buffer.from_numpy(...).to(device) before invocation [LINK:entry:78] [LINK:entry:79] [LINK:entry:80] [CALLER:entry|82]
├─ _Model_execute: contiguity, supported types, ordered Buffer refs [LINK:ct_execute_py:118] [CALLER:entry|82]
└─ _execute_device_tensors → native ModelHandle [LINK:ct_execute_binding:565]
   ├─ input-spec/device validation; prepareInput(storage,spec) [LINK:ct_execute_native:461] [LINK:ct_execute_native:484] [CALLER:ct_execute_py|145]
   │  └─ retained TensorBufferRef shares original storage [LINK:ct_execute_input:121] [CALLER:ct_execute_native|484]
   ├─ release GIL; executeFromDriver awaits M1 [LINK:ct_execute_prepared:529] [LINK:ct_execute_driver:204] [CALLER:ct_execute_native|518]
   ├─ ExecutableModel::execute assembles implicit+user arguments [LINK:ct_execute_callable:373]
   │  └─ executeFn.invoke → host routine → GPU enqueue or CPU function [LINK:ct_execute_callable:384] [LINK:attention_registration:101] [LINK:attention_launch:403] [CALLER:ct_execute_driver|212]
   ├─ await results/errors; drop internal first chain [LINK:ct_execute_results:137] [LINK:ct_execute_results:159] [CALLER:ct_execute_driver|213]
   └─ extractPyBufferResults → [BO1]; caller print(output) [LINK:ct_execute_prepared:538] [LINK:entry:83] [CALLER:entry|82]""",
        s,source="entry",line=82)
    return cards
