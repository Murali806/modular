# MAX engine: graph compilation, Mojo, MEF and inference

[Interactive animation](link_htmls/max_engine_graph_compilation_mef_and_inference.html).

SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Built-in Llama Graph API route plus engine artifact alternatives. Logical source order; no MAX model compilation or inference was run to produce this page.

## 01 · Session & graph

Create the session and trace the built-in Llama Graph API route. Input types are symbolic; forward records operations. Graph.output defines/verifies outputs. Session, graph, weight values and runtime inputs are different objects. Direct read(MEF) can bypass graph rebuilding.

```text
GraphSpec(module, input_types) → Llama3.trace → Graph(forward)
  input metadata → symbolic operations → Graph.output → verified IR
InferenceSession(devices) → native runtime / compilation context
Alternative: read(existing.mef) → CompiledModel, without a new Graph
```

## 02 · Graph compiler

Native Graph compilation is scheduled on a CPU runtime worker under a process-wide lock. Lower a clone through MO, MAP, MOGG and MGP. Configured native cache hits bypass lowering. These frames group compiler roles inside the application, not separate OS processes.

```text
compile(Graph) → compile_async → acquire _COMPILATION_LOCK
  capsule → CompilationRequest → CPU runtime task → clone MLIR
  native cache HIT → MEF; MISS → MO → MAP → MOGG → MGP
  MGP → emitted Mojo → JIT → runtime calls → MEF
  completion callback releases lock; compile() waits
```

## 03 · Mojo kernels

The real mo.add registration shows parameterized Mojo implementation. MAX emits/compiles routines; the Mojo compiler performs semantic/lifetime checks, parametric optimization, elaboration and target lowering. This chapter combines a source example with documented compiler architecture, rather than claiming a recorded Add or Llama compilation.

```text
@extensibility.register("mo.add") → Mojo Add implementation
MGP → Mojo routines → kernel JIT / runtime-call lowering
Mojo: parse / types → semantic / lifetime checks → LIT lowering
  → KGEN optimization → elaboration → concrete functions
  → post-elaboration optimization → LLVM / target code
An op can be fused; this is not one GPU launch per op.
```

## 04 · MEF preparation

Cold compilation serializes MEF. Native model-IR caching and Python MefStore are distinct. Export can occur before runtime initialization. GraphSpec-based import still traces; direct read does not need a new Graph. A declined import key compiles normally, while a missing accepted artifact is an error. Pre-JIT stopping creates no MEF.

```text
Cold: lowered MGP → prepare primitives → serialize MEF → CompiledModel
Native cache: matching module/options → cached MEF
Store export: compile → claim_export → export_mef → manifest
Store import: trace → claim_import → compile(path) → deserialize
Direct: read(MEF) → deserialize (no new Graph)
Declined key → compile; missing accepted key → MissingArtifactError
```

## 05 · Initialize weights

Bind actual named weight buffers after compilation. Graph metadata supports name/device-category validation; contiguity is checked on real initialization. Native setup binds contexts and waits for setup dependencies. Virtual mode creates explicit compile-only stand-ins if names are known, and rejects path artifacts with unknown names.

```text
CompiledModel + weights_registry(name → buffer)
  → expected-name/device checks when captured → contiguity checks
  → _load_all → Model.setup(CPU, registry, device map)
  → await setup chain/context → Model keyed by sym_name
Virtual: known names → CompileOnlyModel; execute refuses
Virtual: path/read artifact with unknown names → ValueError
```

## 06 · Run inference

Per-call Buffers are validated and marshaled as AsyncValues. Native execution releases the GIL, passes persistent context/weights with inputs, and invokes MEF. Primitive readiness activates users or installs callbacks for pending values. Result extraction awaits runtime results and returns Buffers. Target-level synchronization and transfers are backend-specific.

```text
Model.execute(inputs) → Buffer / DLPack / scalar conversion
  → validate dtype/shape/device → prepare AsyncValues
  → release GIL → ExecutableModel.execute
  fixed chain/context/weights + inputs → MEF invocation
  readiness counts / worklist → primitive dispatch
  pending result → andThenSync → downstream users
  await runtime results → drop chain → return output Buffers
```

## Object / role atlas

| Component | Owner / type | Role |
| --- | --- | --- |
| InferenceSession | Python / object → native session | Selects devices, runtime thread settings, extensions and artifact-store policy. Adds the host CPU if absent. A session is neither a model nor a separate OS process. |
| CPU / Accelerator handles | Driver / device handles / hardware context | Device handles identify configured targets. Compilation may use virtual devices; execution requires real device support. CPU is included in the session even when an accelerator is selected. |
| GraphSpec + Llama module | Python / recipe / module objects | The built-in Llama Graph API route supplies a module and symbolic input types. GraphSpec.trace delegates to Llama3.trace. ModuleV3 takes a different staging route. |
| Symbolic input contract | Graph / tensor metadata | Input types describe dtype, shape (including symbolic dimensions), and placement. They are descriptions rather than per-inference tensor values. |
| Graph / MLIR module | Graph / Python object + native IR | Graph owns symbolic graph structure. Forward runs during construction to record operations; it does not compute model outputs on a GPU. |
| KernelLibrary declarations | Graph / compiler metadata | The graph imports kernel declarations. compile_async seeds those declarations into the module so native graph compilation can skip repeating package loading. |
| Symbolic graph operations | Graph / IR operations / dependencies | Embedding, decoder blocks and projections become graph IR on the Llama trace route. The Add example in chapter 03 demonstrates a registration mechanism, not the full Llama kernel schedule. |
| Graph.output contract | Graph / signature / terminator / verification | Flattens outputs, adds dependency chains, sets signatures and mo.output, names the interface, then verifies MLIR. This finalizes the symbolic contract before native compilation. |
| _COMPILATION_LOCK | Python / process-wide threading lock | Serializes Graph/Module native compilation in this process. Acquire before native scheduling; release on setup failure or via the compile completion callback. It is not an IPC semaphore. |
| Pending compilation result | Engine / asynchronous result / synchronization | compile_async returns a CompiledModel backed by a pending native result. compile waits; export or initialization can also resolve pending compilation. A handle does not itself execute the model. |
| Runtime CPU worker pool | AsyncRT / threads in the application process | Native compileModel schedules a CPU runtime task. Native initialization uses a CPU worker continuation. No separate compiler process or fixed thread count is implied. |
| Compiler-owned MLIR clone | Compiler / native IR ownership | The frontend clones the input module before lowering, resolves compiler settings and checks whether declarations are already seeded. The caller graph and this transformed clone are distinct. |
| Native model-IR → MEF cache | Compiler / optional disk cache | If enabled and available, the frontend derives a cache key from the module/options/version and returns a matching MEF before lowering. This cache is separate from Python MefStore. |
| MO stage | Compiler / high-level graph passes | Imports definitions when needed; performs configured shape/symbolic optimization, layout inference and target validation. Pass scheduling does not establish which exact Llama transformations occur. |
| MAP stage | Compiler / fusion / kernel-body transforms | Fuses eligible structured computations, allocates accumulating buffers and outlines kernel bodies. Several graph operations may share a kernel; this animation makes no kernel-count claim. |
| MOGG stage | Compiler / kernel / routine IR | Legalizes calls and operands, outlines routines, then transitions into the runtime-model representation. These are compiler representations, not OS components. |
| MGP stage | Compiler / allocation / dependency planning | Plans buffer reuse and allocations, lowers primitives and dependencies, handles configured device-graph work, merges equivalent routines and emits Mojo. Device-graph passes are distinct from Graph.output. |
| Mojo Add registration | Mojo / registered kernel implementation | The real Add struct is registered as mo.add. Its elementwise overloads use dtype/SIMD-width or tile-layout parameters. Registration provides an implementation building block; it does not imply one launch per graph op. |
| Emitted Mojo routines | Compiler / generated routine representation | MGP emits Mojo routines before the enclosing graph pipeline schedules kernel JIT and final runtime-call lowering. The handwritten Add registration and emitted model routines have different roles. |
| Mojo parse / semantic checks | Mojo / compiler phase | The compiler walkthrough describes parsing/name/type resolution, semantic control-flow and lifetime checks, then LIT lowering. This is a compiler architecture view, not a captured compile trace. |
| KGEN pre-elaboration | Mojo / parametric compiler IR | Pre-elaboration optimization reduces parametric IR before specialization. Compile-time generators retain parameters until elaboration. |
| Elaboration / concrete functions | Mojo / compile-time specialization | Elaboration resolves generators and compile-time parameters into concrete functions/types. dtype, SIMD width and tile layout are illustrated by Add; this page does not choose actual target specializations. |
| LLVM / target lowering | Mojo / compiler backend | Concrete functions undergo lowering/optimization and target code generation. CPU and GPU targets have distinct backend requirements. A target binary is code; no inference input has run yet. |
| Compiled kernel code | Compiler / compiled routines / target binaries | JIT-compiled routines and final runtime calls contribute to the lowered executable artifact. Fusion and routine reuse mean graph-op counts, routine counts and runtime launch counts differ. |
| MEF bytes / file | Artifact / serialized executable format | convertMGPToMEF prepares device primitives and serializes the lowered model. MEF can include constant resources. External weights, initialization state and per-call inputs remain separate concepts. |
| Python MefStore | Python / artifact selection policy + locked metadata | Explicit session store overrides the process-wide default. Graph name/signature select the artifact; optional verify_body checks its fingerprint. The store lock protects its metadata. Declined keys compile normally; missing accepted artifacts raise. |
| Export manifest | Artifact / disk metadata | Export records key, graph name/signature/body fingerprint, writes MEF and persists the manifest. Exporting and importing cannot both be requested on the same session. |
| CompiledModel / CompiledModels | Engine / uninitialized artifact handle | Wraps compiled/deserialized native models. It cannot execute. It can be initialized repeatedly. read(MEF) bypasses rebuilding the Python Graph; a GraphSpec reuse route has already traced. |
| Named weight registry | Python / mapping of actual weight buffers | init_all checks expected weight names and CPU/device category when Graph metadata is available, then contiguity. read/path-compiled artifacts have no Python expected-weight metadata. Weights are distinct from input buffers. |
| Native model setup | Engine / runtime initialization | Model.setup receives CPU runtime, weights registry and device-context map. createExecutableModel waits for setup before returning a usable executable. Compilation alone does not initialize this device state. |
| Persistent executable context | Runtime / AsyncValue state / references | ExecutableModel retains chain, MGP context, weights registry and extra initialization/device arguments. Execution passes these fixed values alongside per-call inputs. |
| Initialized Model / ExecutableModel | Engine / Python/native executable objects | Normal init_all returns Models keyed by sym_name. These hold runnable native models. Virtual mode yields CompileOnlyModel stand-ins when graph names are known; they explicitly refuse execution. |
| Per-call input buffers | Driver / tensor storage / DLPack handles | Model.execute accepts Buffer, supported DLPack arrays or scalar inputs. Checks contiguity, wraps DLPack or constructs scalar Buffers, then prepares native inputs. These values differ from symbolic input types. |
| Native input validation | Engine / checks before executor entry | Native input handling checks input count, shape/dtype contract and device placement, extracts buffer storage and marshals AsyncValues. Invalid inputs can fail before graph execution. |
| AsyncValue register file | AsyncRT / readiness / atomics / reference counts | MEF registers hold values that can be pending, ready or errors. Ready values activate users immediately; pending results install readiness callbacks. This is dependency synchronization rather than a MAX Serve request queue. |
| MEF invocation / executor | Runtime / per-invocation native execution state | Invocation decodes register/primitive tables and allocates executor state. Runnable work follows operand readiness. Execution can dispatch asynchronous work; the executor is not a model-worker process. |
| Ready primitive worklist | Runtime / dependency-driven dispatch | The worklist decrements atomic not-ready counts. A primitive runs when its readiness gate clears. Nonstrict primitives can receive indirect values; readiness and errors govern downstream work. |
| Device runtime / driver | Driver / native device context / dispatch boundary | Execution uses configured device contexts and device-appropriate primitives. Exact launch streams, driver calls and physical allocation depend on the target; this animation does not measure them. |
| CPU / GPU computation | Hardware / hardware · BOUNDARY for measurements | Target code executes on real configured hardware. GPU kernel overlap, duration, occupancy and memory traffic require runtime evidence. Animated movement is a teaching aid. |
| Returned output buffers | Driver / device-associated results | extractDriverResults awaits runtime result values, reports errors, drops the internal chain and wraps tensor results into device buffers; Python returns Buffer objects. A return does not mean a universal device-wide barrier or host copy. |

## Source event sequence

- **Create InferenceSession**: The session selects runtime devices and compilation policy. Host CPU is added if absent. [InferenceSession](sources/max/python/max/engine/api.py#L664)
- **Use a GraphSpec recipe** (trace route): Module and symbolic input types describe the Llama computation, before per-call buffers exist. [Flatten the input-type tree and dispatch to the module's trace method.](sources/max/python/max/pipelines/lib/graph_specs.py#L114) · [Adapt flat graph inputs back into the Llama forward contract.](sources/max/python/max/pipelines/architectures/llama3/llama3.py#L273)
- **Graph records forward symbolically** (trace route): The Graph constructor creates MLIR and runs the forward callback to record operations. [Create MLIR graph structure and optionally trace a Python callable.](sources/max/python/max/graph/graph.py#L819) · [Adapt flat graph inputs back into the Llama forward contract.](sources/max/python/max/pipelines/architectures/llama3/llama3.py#L273)
- **Graph.output verifies the contract** (trace route): Set output signatures, chains, mo.output and names; verify MLIR. No numerical inference ran. [Finish the graph's output contract and verify the completed graph.](sources/max/python/max/graph/graph.py#L1396)
- **Direct MEF read needs no new Graph** (read route): This route starts from an existing artifact. It bypasses rebuilding/tracing the Python Graph. [read(MEF)](sources/max/python/max/engine/api.py#L537)
- **Choose an existing artifact route** (artifact-route route): Explicit import chooses an artifact after trace; read starts from bytes. Graph compiler lowering is skipped. [Select normal compilation, configured MEF import, or MEF export.](sources/max/python/max/engine/api.py#L897) · [read(MEF)](sources/max/python/max/engine/api.py#L537) · [claim_import](sources/max/python/max/engine/_precompiled_mefs.py#L431)
- **Import store declines this key** (decline route): claim_import returns None. The Graph follows normal compilation rather than the import path. [Select normal compilation, configured MEF import, or MEF export.](sources/max/python/max/engine/api.py#L897) · [claim_import](sources/max/python/max/engine/_precompiled_mefs.py#L431)
- **Enter compile_async from compile** (lower-route route): Prepare Graph weight-device expectations, extension libraries and seeded kernel declarations. Native scheduling comes next. [Prepare a Graph, Module, or artifact path for asynchronous native compilation.](sources/max/python/max/engine/api.py#L969) · [Wait for asynchronous compilation and surface errors to the caller.](sources/max/python/max/engine/api.py#L1062)
- **Acquire the process-wide compilation lock** (lower-route route): Graph/Module compilation stays protected until the native completion callback releases this lock. [Cross the Python/C++ boundary with the MLIR module capsule.](sources/max/python/max/engine/api.py#L1276)
- **Schedule native compile on the CPU pool** (lower-route route): The capsule becomes a CompilationRequest. compileModel adds a CPU runtime task and returns its async result. [Turn the Python module capsule into a native CompilationRequest.](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L654) · [Schedule compilation on a CPU worker and resolve an asynchronous result.](sources/max/lib/max_license/API/c/EngineContext.cpp#L225) · [Supply device configuration and create native compiled-model objects.](sources/max/lib/max_license/API/c/EngineContext.cpp#L269)
- **Return an async-backed CompiledModel handle** (lower-route route): compile_async wraps the native result. It remains pending in this illustration while native work continues; compile waits for completion. [Prepare a Graph, Module, or artifact path for asynchronous native compilation.](sources/max/python/max/engine/api.py#L969) · [Wait for asynchronous compilation and surface errors to the caller.](sources/max/python/max/engine/api.py#L1062)
- **Clone the input module / resolve options** (lower-route route): Lowering owns a clone. Seeded kernel declarations prevent repeated package imports. [Clone the graph module, configure lowering, and consult the native MEF cache.](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [Prepare a Graph, Module, or artifact path for asynchronous native compilation.](sources/max/python/max/engine/api.py#L969)
- **Native cache hit returns MEF early** (cache route): Assume native model-IR caching is enabled and a matching artifact exists. Compiler passes are bypassed. [Clone the graph module, configure lowering, and consult the native MEF cache.](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377)
- **MO: shapes, layouts and target legality** (lower route): Configured high-level passes prepare graph IR. This is a pass schedule, not measured Llama fusion. [Prepare and optimize high-level graph operations before MAP lowering.](sources/GraphCompiler/lib/GraphCompiler.cpp#L219) · [Construct the pass manager and run the configured compiler pipeline.](sources/GraphCompiler/lib/GraphCompiler.cpp#L638) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)
- **MAP: eligible fusion and outlined bodies** (lower route): Structured computations can fuse. Accumulating buffers and outlined bodies prepare kernel work. [Fuse eligible structured computations and outline kernel bodies.](sources/GraphCompiler/lib/GraphCompiler.cpp#L338) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)
- **MOGG: legalize calls and outline routines** (lower route): Prepare kernel calls and operands, then lower into the runtime-model representation. [Legalize kernel calls and outline routines.](sources/GraphCompiler/lib/GraphCompiler.cpp#L370) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)
- **MGP: plan buffers, dependencies and primitives** (lower route): Reuse/allocation planning, configured device-graph work and primitive lowering precede Mojo emission. [Plan runtime resources, lower primitives, and emit Mojo routines.](sources/GraphCompiler/lib/GraphCompiler.cpp#L382) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)
- **Kernel compilation is bypassed on this route** (skip-jit route): Cached/imported/read artifacts already contain compiled content. No new model kernel JIT is shown. [Clone the graph module, configure lowering, and consult the native MEF cache.](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [Compile in-memory source or read a model artifact from a path.](sources/max/lib/max_license/API/c/CompiledModels.cpp#L100) · [read(MEF)](sources/max/python/max/engine/api.py#L537)
- **Find the real mo.add → Mojo Add registration** (lower route): Add provides parameterized SIMD and tile overloads. This source example illustrates registration rather than a specific Llama launch. [Mojo mo.add registration](sources/max/kernels/src/graph_compiler/builtin_kernels/elementwise.mojo#L107)
- **MGP emits Mojo routines** (lower route): The enclosing MAX pipeline schedules JIT after Mojo emission, then final runtime-call lowering. [Plan runtime resources, lower primitives, and emit Mojo routines.](sources/GraphCompiler/lib/GraphCompiler.cpp#L382) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)
- **Mojo compiler: parse / types / lifetime checks** (jit route): The documented compiler architecture checks source/semantics and lowers LIT. Detailed JIT plumbing can vary by entry point. [Mojo compiler architecture](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L58)
- **Optimize parametric KGEN** (jit route): Reduce and simplify parametric IR before specialization. [Mojo compiler architecture](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L58)
- **Elaborate compile-time parameters** (jit route): Resolve generators into concrete functions/types. Add illustrates dtype, width and layout parameters. [Mojo compiler architecture](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L58) · [Mojo mo.add registration](sources/max/kernels/src/graph_compiler/builtin_kernels/elementwise.mojo#L107)
- **Lower / optimize into target code** (jit route): Concrete functions pass through post-elaboration and target lowering. CPU/GPU backends differ. [Mojo compiler architecture](sources/Mojo/docs/compiler/MojoCompilerWalkthrough.md#L58) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494)
- **Pre-JIT inspection stops before artifact creation** (prejit route): Deliberate stop after Mojo emission. convertMOToMEF returns an error; no MEF is serialized/cached and no model can execute. [Run graph compilation, then serialize the lowered model to MEF.](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356) · [Build the ordered sequence of representation conversions and compiler stages.](sources/GraphCompiler/lib/GraphCompiler.cpp#L494) · [Wait for asynchronous compilation and surface errors to the caller.](sources/max/python/max/engine/api.py#L1062)
- **Serialize lowered MGP into MEF** (jit route): Prepare device-specific primitives, translate to MEF and copy serialized resources into the artifact buffer. [Serialize MEF](sources/GraphCompiler/lib/GraphCompiler.cpp#L692) · [Run graph compilation, then serialize the lowered model to MEF.](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356)
- **Resolve cached MEF bytes** (cache route): Native cache supplied the buffer before graph/Mojo lowering. It still represents an uninitialized artifact. [Clone the graph module, configure lowering, and consult the native MEF cache.](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377) · [Compile in-memory source or read a model artifact from a path.](sources/max/lib/max_license/API/c/CompiledModels.cpp#L100)
- **Import a matched Graph artifact** (import route): Graph trace already occurred. claim_import returns a path; compile(path) deserializes the MEF rather than lowering new graph IR. [claim_import](sources/max/python/max/engine/_precompiled_mefs.py#L431) · [Select normal compilation, configured MEF import, or MEF export.](sources/max/python/max/engine/api.py#L897) · [Compile in-memory source or read a model artifact from a path.](sources/max/lib/max_license/API/c/CompiledModels.cpp#L100)
- **Missing accepted artifact raises** (missing route): An accepted key with no recorded artifact raises MissingArtifactError. It does not silently fall back to compilation. [claim_import](sources/max/python/max/engine/_precompiled_mefs.py#L431)
- **Read MEF directly into CompiledModel** (read route): read(path or file-like bytes) creates an uninitialized artifact. No new symbolic Graph or native graph lowering is required. [read(MEF)](sources/max/python/max/engine/api.py#L537) · [Compile in-memory source or read a model artifact from a path.](sources/max/lib/max_license/API/c/CompiledModels.cpp#L100)
- **Compilation result resolves / compile waits** (lower-success route): Graph compilation completion releases the lock. compile waits on the native result. The compiled artifact still cannot execute. [Wait for asynchronous compilation and surface errors to the caller.](sources/max/python/max/engine/api.py#L1062) · [Cross the Python/C++ boundary with the MLIR module capsule.](sources/max/python/max/engine/api.py#L1276) · [CompiledModel](sources/max/python/max/engine/api.py#L414)
- **Existing artifact is ready for initialization** (existing-success route): A deserialized artifact holds compiled models without runtime weight binding. [read(MEF)](sources/max/python/max/engine/api.py#L537) · [CompiledModel](sources/max/python/max/engine/api.py#L414) · [Compile in-memory source or read a model artifact from a path.](sources/max/lib/max_license/API/c/CompiledModels.cpp#L100)
- **Claim export identity and write MEF** (export route): Export before initialization. The store records signature/body fingerprint; export_mef writes the artifact and write_manifest persists metadata. [claim_export](sources/max/python/max/engine/_precompiled_mefs.py#L378) · [export_mef](sources/max/python/max/engine/api.py#L449) · [write_manifest](sources/max/python/max/engine/_precompiled_mefs.py#L531)
- **Pass actual named weight buffers to init_all** (success route): Graph-based compilation retains expected weight metadata. Direct read / path compilation lack that Python metadata; native artifact contracts still apply. [Validate weight buffers and initialize models in a compiled artifact.](sources/max/python/max/engine/api.py#L1160) · [Trace graph recipes, compile artifacts, then initialize their weights.](sources/max/python/max/pipelines/lib/graph_specs.py#L183)
- **Reject a noncontiguous registered weight** (bad-layout route): Normal-device initialization checks each registered weight and raises ValueError for this deliberately invalid layout. [Validate weight buffers and initialize models in a compiled artifact.](sources/max/python/max/engine/api.py#L1160)
- **Virtual mode returns compile-only stand-ins** (virtual-known route): Graph names are known for Graph/Module compilation. init_all returns CompileOnlyModel values rather than executable models. [Validate weight buffers and initialize models in a compiled artifact.](sources/max/python/max/engine/api.py#L1160) · [CompileOnlyModel](sources/max/python/max/engine/api.py#L473)
- **Virtual initialization rejects unknown graph names** (virtual-unknown route): Path/read artifacts carry no Python graph-name list here. Virtual mode raises instead of pretending initialization succeeded. [Validate weight buffers and initialize models in a compiled artifact.](sources/max/python/max/engine/api.py#L1160) · [read(MEF)](sources/max/python/max/engine/api.py#L537) · [Prepare a Graph, Module, or artifact path for asynchronous native compilation.](sources/max/python/max/engine/api.py#L969)
- **Validate / bind weights on real devices** (real route): Check captured names and CPU/device category when available; always check registered buffer contiguity on this normal path. [Validate weight buffers and initialize models in a compiled artifact.](sources/max/python/max/engine/api.py#L1160)
- **Native setup binds runtime/device context** (real route): A CPU worker continuation calls model.setup with registry and device map, then awaits setup dependencies. [Create executable model instances once compilation resolves.](sources/max/lib/max_license/API/c/EngineContext.cpp#L310) · [Create executable model](sources/max/lib/max_license/API/c/EngineContext.cpp#L79) · [Setup dependencies](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L387)
- **Return initialized Model keyed by sym_name** (real route): Setup finished. The Model retains its executable context and can now accept per-call inputs. [Validate weight buffers and initialize models in a compiled artifact.](sources/max/python/max/engine/api.py#L1160) · [Create executable model](sources/max/lib/max_license/API/c/EngineContext.cpp#L79) · [ExecutableModel.execute](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L362)
- **CompileOnlyModel refuses execute** (virtual-known route): Virtual compilation can inspect/export code, but the stand-in cannot execute or allocate device-dependent state. [CompileOnlyModel](sources/max/python/max/engine/api.py#L473)
- **Convert per-call inputs to Buffers** (real route): Accept Buffer, supported DLPack arrays or scalars after contiguity checks. Inputs are values, unlike symbolic input types. [Model.execute](sources/max/python/max/engine/api.py#L118)
- **Validate shape / dtype / device, marshal AsyncValues** (real route): Native handling validates input spec and placement, extracts storage and prepares runtime values. [Input shape / dtype checks](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L418) · [Native input preparation](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L451)
- **Release GIL / invoke the executable** (real route): Native execution releases the Python GIL. Fixed chain/context/weights and extra setup arguments accompany each call’s inputs. [Release GIL / execute](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L520) · [Driver execution](sources/max/lib/max_license/API/c/ModelExecution.cpp#L192) · [ExecutableModel.execute](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L362)
- **Decode registers / primitive tables** (real route): MEF invocation allocates per-call executor state and starts primitives whose dependency gates permit execution. [MEF invocation](sources/MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp#L809)
- **Ready operands activate primitive work** (real route): Atomic not-ready counts drop. A ready primitive gets its implementation and arguments; nonstrict primitives can take indirect values. [Ready primitive worklist](sources/MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp#L559)
- **Device-appropriate work computes results** (real route): Primitives use target code and configured device context. Arrow movement illustrates dispatch, not a measured GPU stream or timeline. [Serialize MEF](sources/GraphCompiler/lib/GraphCompiler.cpp#L692) · [ExecutableModel.execute](sources/GraphCompiler/lib/MEFModel/MEFModel.cpp#L362) · [Driver execution](sources/max/lib/max_license/API/c/ModelExecution.cpp#L192)
- **Pending result installs a readiness callback** (real route): For a pending AsyncValue, andThenSync retains the executor and processes users when the value becomes ready. [Result readiness callbacks](sources/MLRT/lib/GraphRT/MEFExecutor/MEFInvokable.cpp#L698)
- **Await runtime results / return output Buffers** (real route): Driver result extraction awaits values, propagates errors, drops the internal chain, preserves device placement and wraps returned tensors. [Await / extract outputs](sources/max/lib/max_license/API/c/ModelExecution.cpp#L100) · [Release GIL / execute](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L520) · [Model.execute](sources/max/python/max/engine/api.py#L118)
