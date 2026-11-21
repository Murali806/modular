## QnA 1 · What does the final MGP stage change?

**MGP optimizes the runtime model plan and prepares its routines for code generation.** The compiler can now reason about storage, dependency chains and work shared across executions. The stage ends by emitting Mojo source for the later kernel compiler. It operates on IR1; G1 remains the caller's original graph. [[gc:382]] [[gc:484]] [[frontend:377]]

**Example links** map the earlier Q&A illustrations and the worked examples to their native checks, rewrites and handoffs. Links marked “later” describe a subsequent stage; a listed rewrite still requires its matching conditions.

```calltree
gcPipelineImpl → buildMgpPipeline [[gc:616]] [[gc:382]]
├─ CSE → canonicalizer, nested on each ModelOp [[gc:387]]
│  ├─ [↗ QnA 1 · entry values — four 4096-byte tensor roles, ordinary graph mode](#qna-mgp-1-illustration-1)
│  └─ TensorSpecGetDimOp::fold: static attr / create operand / no fold [[mgp-ops:484]]
│     ├─ [↗ QnA 2 · static query example — extents 32,32; payload arithmetic 4096](#qna-mgp-2-illustration-1)
│     └─ [↗ QnA 9 · Example 1 — constant, dynamic-operand forwarding or retained query](#qna-mgp-example-spec-folding)
├─ SimplifyChainsPass::runOnOperation [[gc:389]]
│  ├─ flatten chain creates; remove dependencies implied by strict producers [SimplifyChainsPass.cpp:103](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimplifyChainsPass.cpp:103)
│  └─ update operands, then run chain canonicalization patterns [SimplifyChainsPass.cpp:165](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimplifyChainsPass.cpp:165)
│     ├─ [↗ QnA 2 · chain examples — remove duplicate/implied merge dependencies](#qna-mgp-2-illustration-2)
│     └─ [↗ QnA 9 · Example 2 — retain required dependencies and side effects](#qna-mgp-example-chain-simplification)
├─ ConcatInPlace → ExecInvariantCodeMotion [[gc:390]] [[gc:391]]
│  ├─ fixed point: depends-on-args / trivial-only / mutated-by-unliftable [ExecInvariantCodeMotionPass.cpp:693](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:693)
│  ├─ clone pure producers when both blocks need them [ExecInvariantCodeMotionPass.cpp:769](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:769)
│  └─ move eligible work to init; cache buffers/indices for execute [[invariant:822]] [[invariant:846]]
│     ├─ [↗ QnA 4 · invariant example — init caches W; execute reads its model slot](#qna-mgp-4-illustration-1)
│     └─ [↗ QnA 9 · Example 4 — lift eligible work; inputs/mutation keep other work in execute](#qna-mgp-example-invariant-cache)
├─ SimpleBufferReuse, AFTER invariant motion [[gc:393]] [[gc:394]]
│  ├─ group allocation ops by size SSA value, device and alignment [[reuse:112]]
│  ├─ reject shared-operation uses or unproved lifetime order [[reuse:77]] [[reuse:157]]
│  └─ replace later allocation uses with candidate; erase allocation [[reuse:169]]
│     ├─ [↗ QnA 3 · reuse example — A/B/C 12288 bytes → A/B 8192 bytes](#qna-mgp-3-illustration-1)
│     └─ [↗ QnA 9 · Example 3 — safe lifetime order; shared users reject reuse](#qna-mgp-example-buffer-reuse)
├─ DynamicMemoryPlanning → DeferAllocs [[gc:396]]
│  ├─ group eligible execute-block GPU allocations; compute canShare [DynamicMemoryPlanningPass.cpp:111](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:111)
│  ├─ create buffer.plan, one pool allocation and bulk_slice [DynamicMemoryPlanningPass.cpp:199](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:199)
│  │  └─ [↗ QnA 9 · Example 5 — GPU pool with planned slices, lifetime constraints](#qna-mgp-example-gpu-memory-plan)
│  └─ unique writer → move allocation before writer and add its guards [DeferAllocsPass.cpp:61](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DeferAllocsPass.cpp:61)
│     └─ [↗ QnA 9 · Example 6 — defer storage to writer guards without changing values](#qna-mgp-example-defer-allocations)
├─ VerifyModelArguments → PredicateAssertDependencies → SimplifyChains [[gc:398]]
│  └─ model canonicalizer → CSE [[gc:401]]
├─ AttachPythonStackTraces BEFORE fusion [[gc:405]]
├─ GlobalizeSlotIndices → model canonicalizer → CSE [[gc:407]]
├─ LowerPrimitives(kernelLibraryPaths,minimumCpuAlignment) [[gc:411]] [[gc:414]]
│  └─ eligible runtime primitives become code-generating async regions [LowerPrimitives.cpp:153](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/LowerPrimitives.cpp:153)
├─ Device-graph passes scheduled for all models; marker gates actual staging [[gc:417]] [[stage-device:197]]
│  ├─ [↗ QnA 5 · marker comparison — gpu:0 alone does not enable capture](#qna-mgp-5-illustration-1)
│  ├─ [↗ QnA 9 · Example 9 — marked staging/build/execute rewrites; unmarked attention skips](#qna-mgp-example-device-graph-staging-finalization)
│  ├─ Stage → PartitionCollective → ComputeDependencies [[gc:417]]
│  ├─ canonicalize → SimplifyDependencies → canonicalize → CSE [[gc:420]]
│  └─ GenerateBuildGraphStubs → canonicalize → Finalize → reconcile casts [[gc:424]] [[finalize-device:381]]
├─ SequenceFusion: duplicable first, then one-user nonduplicable producers [[gc:432]]
│  ├─ compatibility checks include access flags and device-context types [SequenceFusion.cpp:109](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/SequenceFusion.cpp:109)
│  └─ canonicalize → SimplifyChains → canonicalize → CSE [[gc:433]]
├─ SimplifyExtendLifetimes [[gc:438]]
├─ OutlineAsyncRoutines [[gc:440]]
│  ├─ new module FuncOp uses async BODY argument types, ABI=MGP [OutlineAsyncRoutines.cpp:72](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp:72)
│  ├─ takeBody moves the region into that function; register unique symbol [OutlineAsyncRoutines.cpp:91](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp:91)
│  └─ replace routine with compiled.call, preserving operands/results/device [OutlineAsyncRoutines.cpp:102](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp:102)
│     └─ [↗ QnA 9 · Example 7 — move async body into exported MGP-ABI function](#qna-mgp-example-outlining-and-async-abi)
├─ LookupConstantResources → LiftBufferCacheSlots [[gc:441]] [[gc:445]]
│  └─ cache-slot literals become caller operands to enable helper merging [EmitMojo.cpp:118](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:118)
├─ CanonicalizeFuncsForMerge → canonicalizer → CSE → MergeIdenticalFuncs [[gc:449]] [[gc:454]]
│  └─ RaiseBackIndexConstant → FixupOverhangs → helper canonicalizer → SymbolDCE [[gc:456]] [[gc:463]]
├─ optional profile scopes, controlled by enableProfileScopeTracing [[gc:467]]
├─ AddTryExceptBlocks AFTER final raising operations are known [[gc:476]]
│  └─ MGP ABI failures publish async error results; MOGG ABI can re-raise [AddTryExceptBlocks.cpp:44](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/AddTryExceptBlocks.cpp:44)
├─ EmitMojo(options) → optional DumpMojo [[gc:478]] [[gc:484]] [[gc:486]]
│  ├─ registry + EmitMojoState; built-in/custom/package-submodule imports [[emit:152]] [[emit:158]] [[emit:172]]
│  ├─ FuncOp::emitMojo selects direct MOGG ABI or exported MGP ABI [[emit:193]] [[mogg-ops:1319]]
│  │  ├─ [↗ QnA 7 · ABI example — direct pointer helper versus runtime pointer-to-slots entry](#qna-mgp-7-illustration-1)
│  │  └─ helper body → lowerOpToMojo [[mogg-ops:1462]]
│  │     ├─ ShapeCreate → Coord; TensorCreate → typed row-major views [[mogg-ops:1482]] [[mogg-ops:1504]]
│  │     │  └─ [↗ QnA 6 · emission example — Coord(32,32), strides (32,1), typed views/parameters](#qna-mgp-6-illustration-1)
│  │     └─ CallOpaque → outputs first; integer BN/BD; target from device label [[mogg-ops:1886]] [[emit-state:540]] [[emit-state:582]]
│  │        └─ [↗ QnA 6 · emitted call example — execute(BN=16,BD=8,target="gpu")](#qna-mgp-6-illustration-2)
│  ├─ append deferred struct definitions; optional .mojo dump [[emit:204]] [[emit:223]]
│  ├─ allocate/copy text bytes into resource "mojo"; update existing resource [[emit:228]] [[emit:241]]
│  │  ├─ [↗ QnA 8 · resource before/after — helpers become source bytes; plan remains IR](#qna-mgp-8-illustration-1)
│  │  └─ [↗ QnA 9 · Example 8 — attention shapes/parameters → Mojo text → resource/load](#qna-mgp-example-attention-mojo-resource)
│  ├─ collect each model's called symbols; insert compiled.load in init [[emit:255]]
│  └─ erase emitted functions and consumed device-info mapping [[emit:276]] [[emit:284]]
└─ normal JIT route continues AFTER MGP/EmitMojo [[gc:618]] [[frontend:371]]
   ├─ JitCompileKernels: compile Mojo/P1, or reuse eligible native-code cache [[jit:547]] [[jit:575]]
   ├─ LowerCompiledRoutines: replace compiled load/call with runtime primitives [[lower-routines:102]]
   └─ convertMGPToMEF: clone MGP plan; collect dense resources [[gc:692]] [[gc:697]]
      ├─ PrepareDevicePrimitives; run pass manager on this serialization clone [[gc:700]] [[gc:702]]
      └─ translateMLIRToMEF → ConcatenationTree for program/resources [[gc:706]]
         ├─ empty serialization result → diagnostic and error [[gc:707]]
         └─ allocate WriteableBuffer; traverse and copy serialized bytes [[gc:711]] [[gc:713]]
```

This ordering has a concrete reason: execution-invariant motion runs before buffer reuse because reuse can prevent hoisting. Helper cleanup precedes merging so redundant operations do not make otherwise identical functions look different. [[gc:393]] [[gc:451]]

**The tree describes scheduled native passes in source order.** `buildMgpPipeline` constructs the sequence; the pass bodies act when `pm.run(IR1)` reaches them. An enabled pass can leave a particular model unchanged when its matching conditions fail. [Pass execution](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:638)

| Option or condition | Native effect in this stage |
| --- | --- |
| `kernelLibraryPaths` | Supplies primitive lowering and generated package imports |
| `minimumCpuAlignment` | Passed into primitive lowering |
| `enableOpTraceComments` | Controls comments attached by `EmitMojoState` |
| `enableComputeLambdas` | Selects emission behavior through `EmitMojoState` |
| `moggDebugDumpFusedTensorSpecs` | Enables the emitter's fused-spec debug option |
| `enableProfileScopeTracing` | Schedules profile-scope insertion before exception wrapping |
| `profileScopeTracingTraceUnlabelled` | Forwarded to the profile-scope pass |
| `dumpMojoText` | Adds `DumpMojo` after emission; the separate IR-output-directory mechanism can also write a file |
| `isDeviceGraph` marker | Enables device-graph staging/finalization branches; GPU placement alone does not |
| `skipJit` in the enclosing pipeline | Stops after emitted-source preparation rather than scheduling kernel JIT |

The options are copied in `gcPipelineImpl` and forwarded at the actual pass-construction sites. `skipJit` surrounds the later JIT passes, not the MGP transformations or `EmitMojo` itself. [Option copies](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:543) [Emitter/profile options](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:467) [JIT branch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:614)

<span id="qna-mgp-1-illustration-1"></span>

```state
Attention values entering this stage
  model          = @fused_attention
  inputs         = Q, K, V : f32[32,32], gpu:0
  output         = O       : f32[32,32], gpu:0
  payload bytes  = 32 × 32 × 4 = 4096 per tensor
  custom params  = BD=8, BN=16
  helper         = @F_attention       schematic symbol
  graph mode     = ordinary graph     is_device_graph=False

Passes may rewrite wrappers, guards and storage plans.
The user interface remains THREE input tensors and ONE output tensor.
```

The page's small before/after examples show eligible transformations. Exact final helper names, allocation placement and slot counts require an actual pipeline dump; they are not measured here. [[entry:36]] [[entry:57]] [[graph:831]]

## QnA 2 · Show canonicalization and chain simplification with actual values.

**A static dimension query can become an integer constant.** `TensorSpecGetDimOp::fold` reads the static dimension from the spec type and returns an `IntegerAttr(index,dim)`. For a dynamic dimension created from an SSA operand, it can instead forward that operand. [[mgp-ops:484]]

<span id="qna-mgp-2-illustration-1"></span>

```transform
Static attention spec: <[32,32], f32>

BEFORE                             AFTER folding
  get_dim(%spec, axis=0)              32 : index
  get_dim(%spec, axis=1)              32 : index

Illustrative byte-count arithmetic
  32 × 32 × 4                        4096 bytes

After constants are materialized, eligible CSE can share identical
constant definitions. Q/K/V/O still identify different tensor storage.
```

**TEST · inspected expectation:** the chain tests eliminate duplicate dependencies, flatten nested merges and remove dependencies already implied by another chain. A chain is a completion dependency; simplifying its graph does not evaluate attention. [[test-chains:11]] [[test-chains:108]] [[test-chains:134]]

<span id="qna-mgp-2-illustration-2"></span>

```transform
BEFORE                             AFTER simplifying dependencies
  c1 = merge(c0)                      use c0
  c2 = merge(c0,c0)                   use c0
  c3 = merge(merge(a,b),c)            merge(a,b,c)

If b is produced by work already dependent on a:
  merge(a,b)                          use b

The dependency is preserved; the redundant merge node is removed.
```

This does not authorize removing an actual kernel call whose returned chain is unused. Compiled calls can read and write pointed-to buffers; their side effects must remain. [[mogg-td:444]]

## QnA 3 · Can equal-sized buffers be reused? Give a numerical example.

**Equal size is only one condition.** `SimpleBufferReusePass` groups allocations by the same size SSA value, device and alignment. It then checks that every use of the later allocation executes after every use of the earlier candidate. A shared use by one operation explicitly rejects aliasing. [[reuse:29]] [[reuse:77]] [[reuse:157]]

```calltree
SimpleBufferReusePass::runOnOperation [[reuse:95]]
├─ group buffer.alloc operations by (size SSA value, device, alignment) [[reuse:112]]
├─ collect uses and obtain dependency ordering [[reuse:120]] [[reuse:131]]
├─ reject buffers used by the same operation [[reuse:157]]
└─ safe later use → replace uses with candidate buffer; erase allocation [[reuse:169]]
```

**TEST · inspected expectation:** a serial three-kernel example reuses the first temporary for the third temporary. The middle temporary remains separate. A companion case retains all three allocations when sizes differ. [[test-reuse:3]] [[test-reuse:29]] [[test-reuse:53]]

<span id="qna-mgp-3-illustration-1"></span>

```transform
Comparison graph · three serial kernels, 4096-byte temporaries

BEFORE
  k0: input → A
  k1: A     → B
  k2: B     → C
  allocations: A + B + C = 3 × 4096 = 12288 bytes

AFTER eligible reuse
  k0: input → A
  k1: A     → B
  k2: B     → A       former A contents are no longer needed
  allocations: A + B = 2 × 4096 = 8192 bytes

Illustrative allocation capacity saving: 4096 bytes.
This is not a measurement of GPU peak memory or this attention graph.
```

For the single attention call, Q/K/V/O are used by the same computation. Their equal 4096-byte payloads do not permit the reuse transformation above. Q/K/V are also caller inputs, rather than three fresh temporary allocations created by this pass. [[reuse:157]] [[lower-state:808]] [[entry:78]]

## QnA 4 · What can move from execute to init, and what becomes a context slot?

Execution-invariant motion can lift eligible work out of the repeated execute region. When later execution still needs a lifted buffer, the pass stores it through the init context and replaces execute-side uses with a cached-buffer lookup. It also supports cached indices, allocating a slot by increasing `numSlots`. [[invariant:80]] [[invariant:822]]

<span id="qna-mgp-4-illustration-1"></span>

```transform
Comparison · eligible invariant buffer, schematic

BEFORE
  init:     context(numSlots=0)
  execute:  build buffer W from invariant inputs → use W

AFTER eligible lift + cache
  init:     build W → cache W in model-state slot s
  execute:  get_cached(context, slot=s) → use W

Example slot allocation
  next slot s = 0
  context slot count: 0 → 1
```

The slot contains runtime model state when init later executes; the compiler is currently creating the operations that manage it. This example illustrates the mechanism, not a claim that the attention graph has one final slot. Its attention result depends on per-call Q/K/V values, so that computation is not an invariant result that can be precomputed during init. [[invariant:846]] [[entry:57]] [[entry:78]]

## QnA 5 · Does gpu:0 automatically enable device-graph capture?

**No. Device placement and device-graph mode are separate settings.** `Graph.__init__` defaults `is_device_graph=False`; it adds the graph marker only when requested. The fused-attention caller omits this option. [[graph:831]] [[graph:915]] [[entry:41]]

<span id="qna-mgp-5-illustration-1"></span>

```state
This fused-attention example
  tensor placement      = gpu:0
  graph is_device_graph = False
  StageDeviceGraphs     → no staging for this unmarked model
  FinalizeDeviceGraphs  → returns for this unmarked model
  EmitMojo              → still emits the ordinary runtime routines

Comparison: a marked model
  stage device operations
    → partition collectives
    → build dependency/stub information
    → finalize collective graph operations
```

The guard is explicit in both native passes. GPU execution still occurs later through the ordinary generated routines and device context; using a GPU does not imply that this compile has constructed a captured device graph. [[stage-device:197]] [[finalize-device:381]] [[gc:417]]

## QnA 6 · How does EmitMojo turn shapes, tensors and parameters into code?

The emitter creates a `MojoRegistry` and `EmitMojoState`, writes built-in and P1 package imports, then asks each module-level `mogg.func` to emit its body. Values in the MLIR operations become names or expressions in the Mojo text builder. [[emit:152]] [[emit:172]] [[emit:193]]

```calltree
EmitMojo::emitMojoText() [[emit:147]]
├─ import P1 by package filename stem; import registered kernel submodules [[emit:172]] [[emit:182]]
├─ FuncOp::emitMojo → select MOGG or MGP ABI emitter [[mogg-ops:1319]]
│  └─ helper body operations → lowerOpToMojo [[mogg-ops:1462]]
│     ├─ ShapeCreateOp → Coord with compile-time Idx entries [[mogg-ops:1482]] [[emit-state:618]]
│     ├─ TensorCreateOp → row-major layout + tensor-view constructor [[mogg-ops:1504]]
│     └─ CallOpaqueOp::emitMojo [[mogg-ops:1886]]
│        ├─ select registered execute declaration [[mogg-ops:1887]]
│        ├─ prepare OUTPUTS first, then inputs, then context [[mogg-ops:1896]] [[mogg-ops:1915]]
│        ├─ BD/BN IntegerAttrs → integer parameter literals [[emit-state:540]]
│        ├─ target → surrounding helper's device kind "gpu" [[emit-state:582]]
│        └─ emit the fully resolved execute call [[mogg-ops:1937]]
└─ builder text + deferred struct definitions → mojoText [[emit:204]]
```

<span id="qna-mgp-6-illustration-1"></span>

```transform
Shape
  [32,32] → Coord containing Idx[32], Idx[32]
  row-major element strides = [32,1]
  element [1,2] offset = 1×32 + 2 = 34 elements = 136 bytes

Tensor views
  qptr,kptr,vptr → f32, rank=2, Input, mut=false
  optr          → f32, rank=2, Output, mut=true
  storage addresses remain runtime arguments

Compile-time call parameters
  BD = IntegerAttr(index,8)  → 8
  BN = IntegerAttr(index,16) → 16
  helper device "gpu"       → target="gpu"

Helper arguments: qptr,kptr,vptr,optr,ctx
Kernel execute arguments: output,query,key,value,ctx
```

The output-first kernel call follows the registered `OutputTensor` signature. The pointer helper's input-first argument order and the kernel's output-first argument order are different interfaces; the emitter constructs the correct mapping. [[kernel:78]] [[kernel:82]] [[mogg-ops:1896]]

**Illustrative emitted call**, with module qualification, variable names and view setup abbreviated:

<span id="qna-mgp-6-illustration-2"></span>

```mojo
FusedAttention.execute[BN=16, BD=8, target="gpu"](
    output, query, key, value, ctx
)
```

The infer-only dtype/rank parameters are resolved from the tensor argument types by Mojo. They are not extra Q/K/V buffers or runtime integer arguments. BD/BN specialize the implementation; the tensor dimensions stay `[32,32]`. [[params:328]] [[kernel:75]] [[kernel:78]]

## QnA 7 · Why are there two helper ABIs, and what does the runtime receive?

`FuncOp::emitMojo` distinguishes the direct **MOGG ABI** helper from the **MGP ABI** entry obtained from an asynchronous routine. The direct helper accepts pointers/dimensions/context. The runtime-facing entry takes a pointer to argument slots and handles async-value ownership and return slots. [[mogg-ops:1326]] [[mogg-ops:1334]] [[mogg-ops:1420]]

<span id="qna-mgp-7-illustration-1"></span>

```state
Schematic call relationship
  runtime entry, ABI=MGP
  └─ pointer to slots
     ├─ take ownership of async input slots
     ├─ unpack handles; preserve their lifetime
     ├─ call direct helper @F_attention, ABI=MOGG
     │  └─ four tensor pointers + device context
     │     └─ build views; call FusedAttention.execute
     └─ publish returned chain/value slots

No dynamic-dimension argument is needed for static [32,32].
Any final runtime slot list also reflects later wrapper/primitive fusion.
```

The MGP entry is exported and non-raising; preceding passes arrange error handling. The MOGG helper can raise and is not the exported pointer-to-slots entry. This makes the low-level runtime boundary explicit without claiming that the earlier five direct arguments are the complete final runtime slot array. [[mogg-ops:1347]] [[mogg-ops:1377]] [[mogg-ops:1453]] [[gc:476]]

## QnA 8 · What replaces the helper IR, and when do machine code and MEF appear?

**The emitted text becomes an MLIR resource named `mojo`.** The pass stores its bytes in a `DenseResourceElementsAttr`, collects the symbols referenced by each model's `mogg.compiled.call` operations and inserts a `mogg.compiled.load` at the start of that model's init block. It then erases the emitted `mogg.func` operations. [[emit:228]] [[emit:241]] [[emit:255]] [[emit:276]]

<span id="qna-mgp-8-illustration-1"></span>

```transform
BEFORE EmitMojo
  IR1 module
  ├─ mgp.model @fused_attention
  │  ├─ init plan
  │  └─ execute plan → compiled.call @entry
  └─ mogg.func helpers containing code-generation operations

AFTER EmitMojo, schematic
  IR1 module
  ├─ resource "mojo" = bytes of emitted Mojo source
  └─ mgp.model @fused_attention
     ├─ init: compiled.load(symbols=[@entry], resource="mojo")
     └─ execute: compiled.call @entry(...)

The helper code is now text in the resource.
The model's runtime plan is still represented in IR.
```

Despite the operation name `compiled.load`, its resource at this point holds source text for the next compiler stage. EmitMojo can optionally dump a `.mojo` file, but the in-memory resource is the normal handoff. It also removes the module's device-info mapping once emission has consumed it. [[emit:223]] [[emit:284]]

```calltree
After the MGP pipeline [[gc:616]]
├─ JitCompileKernels → parse emitted Mojo with package imports [[gc:618]] [[jit:547]]
│  └─ lower/compile or reuse eligible native-code cache entries [[jit:567]] [[jit:575]]
├─ LowerCompiledRoutines → runtime load/execute operations [[lower-routines:102]]
└─ convertMGPToMEF → serialize the runtime program and resources [[gc:692]]

C1 completion comes after these later compilation/serialization steps.
Calling EmitMojo alone has not generated a PTX or cubin artifact.
```

The existing GPU artifact / ABI / driver deep dive continues from that later boundary. The caller creates Q/K/V numeric arrays and GPU buffers after compilation and model initialization; these emitted expressions are not those arrays. [[entry:74]] [[entry:78]]

## QnA 9 · How do MGP passes simplify shapes and chains, reuse and plan buffers, move invariant work, defer allocations, emit routine interfaces and prepare marked models for device-graph construction? Show nine worked transformations with source code, test entry points and the actual fused-attention route.

**SOURCE + TEST · Each example identifies its input representation and native implementation.** The before/after IR is schematic unless explicitly quoted from a test. `TEST` means the repository's supplied input and expected output were inspected; no compiler or GPU workload was executed. A `RUN` line gives an independent test pipeline, and its `CHECK` comments constrain printed IR or emitted source. These test files are not callees in the production compiler.

<details class="qna-cell" id="qna-mgp-example-spec-folding" markdown="1">
<summary>Example 1 · When does a tensor dimension become a constant, an existing SSA value, or a retained runtime query?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · Canonicalization acts on tensor-spec metadata.** The attention output spec knows both dimensions, so querying either extent can fold to `32 : index`. This leaves the underlying tensor data and storage handles intact. [Dimension folder](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/MGPOps.cpp:484)

```transform
BEFORE · schematic static attention metadata
  %spec = tensor_spec.create <32x32xf32>           # No dynamic operands.
  %n = tensor_spec.get_dim[0] %spec               # Query row extent.
  %d = tensor_spec.get_dim[1] %spec               # Query column extent.
  %bytes = %n × %d × 4                            # f32 payload arithmetic.
  %buf = buffer.alloc(%guard,%bytes,%gpu,A)         # One output allocation.

AFTER · fold/canonicalize, then eligible CSE
  %c32 = index.constant 32                        # Shared extent value.
  %bytes = index.constant 4096                    # Constant payload size.
  %buf = buffer.alloc(%guard,%bytes,%gpu,A)         # Same storage role.
  %spec = tensor_spec.create <32x32xf32>           # Still describes output.

BEFORE · dynamic extent explicitly constructed
  %spec = tensor_spec.create(%batch) : <?x4xf32>   # One runtime dimension.
  %n = tensor_spec.get_dim[0] %spec               # Query what create received.
  %d = tensor_spec.get_dim[1] %spec               # Query static second axis.

AFTER
  use %batch wherever %n was used                 # Forward the same SSA value.
  use 4 : index wherever %d was used              # Static type component.
```

The implementation first validates the axis, then distinguishes the static type component from the defining `TensorSpecCreateOp`. These are the exact two successful return paths. [Static and dynamic cases](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/MGPOps.cpp:489)

```cpp
if (!specType.isDynamicDim(getAxis())) {
  auto indexTy = IndexType::get(getContext());
  uint64_t dim = specType.getDims()[getAxis()].value();
  return IntegerAttr::get(indexTy, dim);
}
if (auto specCreate = dyn_cast_or_null<TensorSpecCreateOp>(specProducer)) {
  return specCreate.getRuntimeDims()[specType.getDynamicDimIndex(getAxis())];
}
```

| Input to the query | Fold result | Example |
| --- | --- | --- |
| `spec<32x32xf32>`, axis 0 | Integer attribute 32 | `32×32×4 = 4096` bytes |
| <code>create(%batch):spec&lt;?x4xf32&gt;</code>, axis 0 | Existing `%batch` operand | `batch=5` means 80 bytes |
| Function argument <code>spec&lt;?x4xf32&gt;</code>, axis 0 | No fold from this rule | Runtime source must supply the extent |
| <code>spec&lt;?x64x?xf32&gt;</code>, axis 4 | Invalid IR: axis ≥ rank | Diagnostic before a successful transformation |

The folder returns no replacement for a dynamic argument spec whose creator is unavailable. Knowing rank two alone does not supply its batch length. Returning an SSA value also does not freeze its value across invocations. [No-fold return](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/MGPOps.cpp:494)

**TEST · The diagnostic input checks the contract this fold assumes.** `mgp-errors.mlir` feeds MGP operations directly to `mo-opt --split-input-file --verify-diagnostics --allow-unregistered-dialect`, without a transformation pass. Its `expected-error` for `get_dim[4]` on rank three matches `TensorSpecGetDimOp::verify`. The same file supplies only one dynamic operand for <code>&lt;?x64x?xf32&gt;</code> and expects a two-operands-required error from `TensorSpecCreateOp::verify`. These negative tests verify legal inputs; they are not evidence that the static fold was executed here. [Axis diagnostic input](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Dialect/mgp-errors.mlir:35) [Create diagnostic input](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Dialect/mgp-errors.mlir:10) [Native verifiers](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/MGPOps.cpp:397) [Axis verifier](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/MGPOps.cpp:475)

For attention, these metadata simplifications do not evaluate `softmax(QKᵀ)V` or merge the Q/K/V/O buffers merely because they share the same type. [[entry:57]] [[lower-state:808]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-chain-simplification" markdown="1">
<summary>Example 2 · How can a chain merge disappear while every required dependency remains?</summary>
<div class="qna-answer" markdown="1">

**TEST · `mgp-simplify-chains.mlir` already contains MGP models and GRT chains.** Its `RUN` applies the async-dataflow test analysis, `--mgp-simplify-chains` and `--cse` with unregistered stand-in operations. It checks graph structure, not their execution. [Test pipeline](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simplify-chains.mlir:1)

```transform
BEFORE · schematic subset of @chains
  %c1 = grt.chain.create(%entry)                  # Singleton dependency.
  %c2 = grt.chain.create(%c1,%c1)                  # Duplicate dependency.
  %a  = workA(%c2)                               # A waits for entry.
  %m  = grt.chain.create(%entry,%a)               # A already implies entry.
  %b  = workB(%m)                                # B needs A's completion.
  mef.output %b                                  # Publish completion of B.

AFTER
  %a  = workA(%entry)                            # Same A dependency.
  %b  = workB(%a)                                # Same B dependency.
  mef.output %b                                  # Same completion contract.

Another eligible form:
  merge(merge(%a,%b),%c) → merge(%a,%b,%c)         # Flatten a merge tree.
  merge(%a,%a) → %a                              # Remove repeated operand.

Skipped comparison:
  %a = independentA()                            # No dependency on b.
  %b = independentB()                            # No dependency on a.
  merge(%a,%b) stays a two-input dependency       # Both completions required.
```

The implementation flattens `grt.chain.create` inputs into unique primitive dependencies. It then walks each producer's **gating operands** and removes a candidate dependency only if another retained dependency implies it. Strict producers contribute their operands; control flow contributes only its strict extra operands; a non-strict operation contributes none. These conditions prevent a mere textual operand from being treated as a proven wait. [Flattening and gating rules](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimplifyChainsPass.cpp:53)

```cpp
if (isa<MEF::NonStrictInterface>(op))
  return ValueRange{};
return op->getOperands();
```

After computing the reduced sets, the pass changes each chain operation's operands, then invokes its canonicalization patterns. That second step removes singleton/empty/redundant structure exposed by the analysis. [Rewriting phase](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimplifyChainsPass.cpp:165)

```cpp
for (auto &[chainOp, values] : cache)
  chainOp.setInputChains(values);

RewritePatternSet patterns(&getContext());
GRT::ChainCreateOp::getCanonicalizationPatterns(patterns, &getContext());
```

The `@chains` checks require `some.op1` to consume the entry chain and its successors to consume the first operation's chain. The consecutive-merge test expects three independent chains to remain in one merge; the uniquing test expects the original producer's chain at the output. Thus the test covers both redundant edges and dependencies that must stay. [Transitive redundancy checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simplify-chains.mlir:10) [Independent inputs retained](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simplify-chains.mlir:99) [Duplicate elimination](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simplify-chains.mlir:121)

**Attention connection:** a redundant `merge(entry,entry)` around a static tensor view may simplify. The async attention call still reads Q/K/V and writes O. Removing a redundant merge leaves that call and its output guard in place; it is not numerical constant folding or proof that a GPU launch has completed. [[mogg-ops:751]] [[mogg-ops:999]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-buffer-reuse" markdown="1">
<summary>Example 3 · Why can the third temporary reuse the first, but the middle temporary cannot?</summary>
<div class="qna-answer" markdown="1">

**TEST · `mgp-simple-buffer-reuse.mlir` enters as a model with three allocations.** Its `RUN` applies only `--mgp-simple-buffer-reuse` to each split input, accepting unregistered `kernel_execute` placeholders. The placeholders encode buffer use and chain order; they are not arithmetic kernels run by this test. [Test entry and topology](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simple-buffer-reuse.mlir:1)

```transform
BEFORE · illustrative f32[32,32] temporaries, 4096 bytes each
  %A = alloc(%entry,%size,%gpu,align=A0)           # First temporary.
  %B = alloc(%entry,%size,%gpu,align=A0)           # Second temporary.
  %C = alloc(%entry,%size,%gpu,align=A0)           # Third temporary.
  %c0 = add_one(%entry,%X,%A)                     # A = X+1.
  %c1 = times_two(%c0,%A,%B)                      # B = 2A; final read of old A.
  %c2 = minus_three(%c1,%B,%C)                    # C = B-3.
  %Y = tensor.create(%c2,%C,%spec)                # Return C's contents.

AFTER · eligible reuse
  %A = alloc(%entry,%size,%gpu,align=A0)           # Retained allocation.
  %B = alloc(%entry,%size,%gpu,align=A0)           # Still needed beside A.
  %c0 = add_one(%entry,%X,%A)                     # Write generation 1 of A.
  %c1 = times_two(%c0,%A,%B)                      # Read generation 1 of A.
  %c2 = minus_three(%c1,%B,%A)                    # Write generation 2 of A.
  %Y = tensor.create(%c2,%A,%spec)                # Rewired result buffer.

Illustrative first four elements:
  X=[1,2,3,4] → A=[2,3,4,5] → B=[4,6,8,10] → Y=[1,3,5,7]
  allocation capacity: 12288 → 8192 bytes         # Arithmetic, not measured peak.
```

The reuse key compares the **same SSA size value**, device attribute and alignment. Earlier CSE may make duplicate size constants share an SSA definition; this pass itself does not prove arbitrary arithmetic expressions equal. [Reuse key](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp:29)

```cpp
bool operator==(const ReuseKey &other) const {
  return size == other.size && device == other.device &&
         alignment == other.alignment;
}
```

For A and C, every C use follows every A use through `c1`. A and B share the `times_two` operation, and B and C share `minus_three`, so those pairs cannot alias. The implementation rejects any shared operation first, then requires every current use to execute after every candidate use. On success it rewrites **all** users and erases the redundant allocation. [Same-operation rejection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp:157) [Dependency proof](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp:77)

```cpp
if (canSafelyReuseBuffer(currentUses, candidateUses, depTree)) {
  currentAlloc->replaceAllUsesWith(candidateAlloc);
  currentAlloc->erase();
```

The test checks exactly two remaining allocations, then requires the third `kernel_execute` to write `ALLOC0`. Its negative case uses a different size SSA expression and requires all three allocations. Its raw allocation-size arithmetic is a test fixture; the correctly sized 4096-byte tensor example above is a separate illustration. [Positive rewiring CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simple-buffer-reuse.mlir:29) [Different-size rejection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-simple-buffer-reuse.mlir:53)

| Candidate pair | Outcome | Reason |
| --- | --- | --- |
| A / C in the serial example | Reuse eligible | Compatible key; disjoint ordered uses |
| A / B | Rejected | One operation reads A while writing B |
| Same capacity, different device or alignment | Rejected | Different `ReuseKey` |
| A / C with an additional late read of A | Rejected unless all-use ordering is proved | Old A contents remain live |
| Attention Q/K/V / O | This example's reuse does not apply | Inputs are caller storage; all participate in the same call |

The attention output is one new allocation. The equal tensor payload sizes alone provide no candidate temporary whose entire lifetime has ended before O is written. [[lower-state:808]] [[mogg-ops:789]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-invariant-cache" markdown="1">
<summary>Example 4 · How does invariant work move into init, and why do aliases or runtime inputs keep work in execute?</summary>
<div class="qna-answer" markdown="1">

**TEST · `mgp-exec-invariant-code-motion.mlir` supplies MGP init/execute blocks.** Its `RUN` applies `--mgp-exec-invariant-code-motion --canonicalize --cse`, independently of graph lowering. `@liftable_op` retrieves an external 37632-byte weight named `foo` during execute; init starts with two context slots. [Input and checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-exec-invariant-code-motion.mlir:1)

```transform
BEFORE · schematic @liftable_op
  init:
    %ctx = context.create 2                       # Slots 0,1 already reserved.
    %ready = runtime.create(%initChain,%ctx[1])   # Existing initialization.
    output %ready,%ctx,%weights,%cpu              # State passed to execute.
  execute:
    %name = string.create "foo"                   # Fixed registry key.
    %W = buffer.constant.external(%entry,%weights[%name,37632,align=16])
                                                    # Invariant retrieval.
    output %entry,%W                              # Buffer needed every call.

AFTER · eligible motion
  init:
    %ctx = context.create 3                       # One new buffer-cache slot.
    %ready = runtime.create(%initChain,%ctx[1])   # Retained initialization.
    %W = buffer.constant.external(%ready,%weights["foo",37632,align=16])
                                                    # Original work moved here.
    %cached = buffer.cache(%cacheGuard,%ctx[2],%W) # Retain buffer in model state.
    output merge(%ready,%cached),%ctx,%weights,%cpu # Init waits for cache setup.
  execute:
    %W = buffer.get_cached(%entry,%ctx[2])        # Read same initialized value.
    output %entry,%W                              # Same result contract.
```

The analyses repeatedly exclude work that depends on per-call inputs, work trivial to recompute when only unliftable users need it, and buffers mutated by work remaining in execute. They then decide which pure producers need a clone in both blocks. This is a fixed-point dataflow decision rather than a rule that every constant-shaped allocation must move. [Analysis order](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:693)

For a lifted buffer still used in execute, the transformation creates a cache operation in init and a matching lookup in execute. It replaces only uses that stay in execute, then moves the original producer before init's terminator. [Buffer rewrite](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:822)

```cpp
auto newBuffer = bufferBuilder->getCached(
    executeState->builder, op->getLoc(), executeState->getContext(),
    executeState->getRootChain(), currBufferSlot,
    oldBuffer.getType());
oldBuffer.replaceUsesWithIf(newBuffer, isOpOperandUnliftable);
```

`MGPBufferBuilder::cache` takes the current slot count as the new slot index, then increments the count. That is the native reason for `slot=2` and `numSlots:2→3` in this test; neither is a pointer address. [Slot allocation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp:329)

```cpp
uint64_t bufferSlot = contextCreateOp.getNumSlots();
contextCreateOp.setNumSlots(bufferSlot + 1);
```

The positive `CHECK-DAG` lines require init's `context.create 3`, external lookup and `buffer.cache …[2]`; the execute checks require `buffer.get_cached`. `@constant_external_mutable` supplies `has_alias true` and expects the lookup to remain after the execute boundary. The native early return below encodes that rejection because outside code may mutate the externally aliased storage. [Positive and negative CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-exec-invariant-code-motion.mlir:7) [Alias guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:201)

```cpp
if (auto externalConstOp = dyn_cast<BufferConstantExternalOp>(op))
  if (externalConstOp.getDHasAlias())
    return false;
```

A companion case returns a chain plus a dimension value equal to four. It expects the routine to move to init and an `int.get_cached` in execute. An async region with an additional tensor result remains in execute because this lifting path has no cache representation for that result. [Dimension-cache test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-exec-invariant-code-motion.mlir:206) [Uncacheable-result test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-exec-invariant-code-motion.mlir:236) [Result-type guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:165)

**Actual attention:** Q/K/V arrive afresh through execute arguments, so the attention computation is input-dependent. Its fixed 4096-byte output size does not make the output contents invariant, and mutation analysis also matters when considering its storage. The test's three slots describe that fixture, not attention's final context layout. [[entry:78]] [[entry:82]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-gpu-memory-plan" markdown="1">
<summary>Example 5 · How do several GPU allocations become one pool with planned slices?</summary>
<div class="qna-answer" markdown="1">

**TEST · `mgp-dynamic-memory-planning.mlir` supplies allocation/use chains directly.** Its `RUN` applies `--mgp-dynamic-memory-planning` to split MGP inputs, allowing the stand-in `"use"` operation. It does not run the preceding simple-reuse pass, so `@basic_memory_planning` intentionally still contains two same-size allocations. [Test pipeline](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-dynamic-memory-planning.mlir:1)

The native planner examines top-level execute allocations on non-CPU devices. It requires known, nonempty uses that the dependency tree can order and a size computable from allowed metadata/arithmetic operations. It then groups candidates per device and requires at least two. [Eligibility and grouping](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:111)

```cpp
if (allocOp.getDevice().getLabel() == kCPULabel)
  continue;
```

```transform
BEFORE · illustrative gpu:0 plan, sizes known from this call's metadata
  %A = alloc(%entry,%sizeA,%gpu,align=16)          # sizeA=16 bytes, f32[4].
  %B = alloc(%entry,%sizeB,%gpu,align=16)          # sizeB=48 bytes, f32[12].
  %cA = use_A(%entry,%A)                          # Last use of A.
  %cB = use_B(%cA,%B)                             # First use of B follows A.
  output %cB                                     # Preserve work completion.

AFTER · schematic MGP
  %high,%offA,%offB = buffer.plan(%sizeA,%sizeB)   # Compute layout when sizes known.
    alignments=[16,16], canShare=[[0,1],[1,0]]     # Disjoint-use permission.
  %pool = alloc(%entry,%high,%gpu,align=16)        # One owning allocation.
  %A,%B = buffer.bulk_slice(%pool,%gpu,
                 offsets=[%offA,%offB],sizes=[%sizeA,%sizeB])
                                                    # Two planned buffer handles.
  %cA = use_A(%entry,%A)                          # Same operation and contents.
  %cB = use_B(%cA,%B)                             # Same dependency.
  output %cB                                     # Same completion.

One permitted layout for sizeA=16,sizeB=48:
  offsets=[0,0], capacity=48 bytes                # Reuse range after A ends.
  symbolic pool base B; both slices start at B    # No observed address claimed.
```

The transformation computes `canShare[i][j]` by checking whether **all** uses of one allocation follow all uses of the other, in either direction. It emits `BufferPlanOp`, uses its high-watermark result for the pool size, and constructs one `BufferBulkSliceOp`. It replaces each original allocation's users with the corresponding slice result and erases the old allocation. [Sharing proof](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:172) [Pool/slice construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:199)

```cpp
allocs[i].alloc->replaceAllUsesWith(
    ValueRange{bulkSliceOp.getOutBuffers()[i]});
allocsToDelete.push_back(allocs[i].alloc);
```

The numeric offsets above are an illustrative legal layout, not an inspected runtime result. If A has another use after B, sharing becomes illegal: `canShare=[[0,0],[0,0]]`. A separate layout such as `offA=0, offB=16, capacity=64` preserves both live ranges. A tensor view of A does not end A's lifetime; consumers reached through `tensor.create → tensor.extract.buffer` still count. [Overlapping-view test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-dynamic-memory-planning.mlir:139)

| Repository case | Relevant expected output |
| --- | --- |
| `@basic_memory_planning` | One `buffer.plan`, one allocation, one two-result `bulk_slice`; no individual `buffer.slice` |
| `@cpu_allocations_ignored` | GPU pool created; CPU allocation remains |
| `@single_allocation_not_planned` | `CHECK-NOT: mgp.buffer.plan` |
| `@different_alignments` | Alignment list `[16,32,64]`; pool alignment 64 |
| `@tensor_view_extends_lifetime` | `canShare = [0,0,0,0]` |

The pool alignment is the least common multiple of candidate alignments. Each device is planned separately, and a size produced by unsupported data-dependent work is skipped by `isTriviallyComputable`. [Test expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-dynamic-memory-planning.mlir:12) [Skipped cases](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-dynamic-memory-planning.mlir:30) [Alignment calculation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:164) [Size guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp:49)

**Actual attention:** Q/K/V are external inputs and its isolated custom call creates one O allocation. Those four tensor roles do not constitute the two-or-more compiler-created allocations required for this pool example. This pass also does not discover or pool temporary storage inside the authored attention kernel. [[lower-state:808]] [[mogg-ops:789]] [[kernel:105]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-defer-allocations" markdown="1">
<summary>Example 6 · How does deferring an allocation change when storage is needed while preserving its writer and values?</summary>
<div class="qna-answer" markdown="1">

**TEST · `mgp-defer-allocs.mlir` starts with a fully formed MGP plan.** Its `RUN` uses canonicalization, `--mgp-defer-allocs`, chain simplification and CSE. The fixture's `"mul"` / `"sub"` / `"add"` operations are unregistered stand-ins whose `tempInOutSig="iiio"` marks the final operand as output. The file includes an explanatory MO graph in comments; the actual parsed input is `mgp.model @basic`. [RUN and explanatory graph](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Dialect/mgp-defer-allocs.mlir:1) [Actual MGP input](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Dialect/mgp-defer-allocs.mlir:34)

```transform
BEFORE · schematic f32[8] computation, 32-byte outputs
  %A = alloc(%entry,32,%cpu)                      # Mul destination.
  %c1 = mul(%entry,%X,%Ten,%A)                    # A = X*10.
  %B = alloc(%entry,32,%cpu)                      # Allocation only waits for entry.
  %c2 = sub(%c1,%A,%Ten,%B)                       # B = A-10.
  %C = alloc(%entry,32,%cpu)                      # Also only waits for entry.
  %c3 = add(%c2,%B,%A,%C)                         # C = B+A.
  %Y = tensor.create(%c3,%C,%spec)                # Result uses C after its writer.

AFTER · defer + chain simplification
  %A = alloc(%entry,32,%cpu)                      # First writer has no later guard.
  %c1 = mul(%entry,%X,%Ten,%A)                    # Same first writer.
  %B = alloc(%c1,32,%cpu)                         # Wait for producer of B's input.
  %c2 = sub(%c1,%A,%Ten,%B)                       # Same subtraction.
  %C = alloc(%c2,32,%cpu)                         # Wait for producer of C's inputs.
  %c3 = add(%c2,%B,%A,%C)                         # Same addition.
  %Y = tensor.create(%c3,%C,%spec)                # Same result dependency.

Illustrative arithmetic matching the labels:
  X=[1,2,3,4,5,6,7,8]                             # Eight input values.
  A=[10,20,30,40,50,60,70,80]                      # Multiply by ten.
  B=[0,10,20,30,40,50,60,70]                       # Subtract ten.
  Y=[10,30,50,70,90,110,130,150]                   # Add B and A.
```

The key change is the allocation's **dependency operand**, together with its IR position. Moving the printed line alone would not encode readiness in an asynchronous execution plan. The implementation requires a unique mutator, moves the allocation just before that operation, gathers the writer's input chains and calls `alsoWaitForChains` to preserve/add dependencies. [Defer implementation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DeferAllocsPass.cpp:53)

```cpp
OpOperand *mutator = getUniqueMutator(allocOp.getOutBuffer());
if (!mutator)
  return;

allocOp->moveBefore(mutator->getOwner());
movedAllocOps.insert(allocOp);
```

The `@basic` checks require B's allocation to consume the mul chain and immediately precede `sub`; C's allocation must consume the sub chain and immediately precede `add`. They preserve all three allocation roles; this is different work from reuse or pooling. [Expected moved allocations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Dialect/mgp-defer-allocs.mlir:18)

The `@divergent` case adds other input chains. Its checks require merged guards so those dependencies survive the rewrite. If no unique mutator can be established, the early return above leaves the allocation unchanged. The visited-operation set also prevents repeatedly moving an allocation when one writer handles several output buffers. [Extra dependency checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Dialect/mgp-defer-allocs.mlir:69) [Repeated-writer guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/DeferAllocsPass.cpp:50)

**Attention connection:** O has a writing routine, so its allocation can be considered using that routine's guards. This is a change to allocation timing in the runtime plan; no new attention operation or CPU/GPU transfer is introduced. The mathematical output stays `softmax(QKᵀ)V`. [[mogg-ops:798]] [[mogg-ops:999]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-outlining-and-async-abi" markdown="1">
<summary>Example 7 · How does an asynchronous region become a callable symbol and a pointer-to-slots runtime entry?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · OutlineAsyncRoutines moves code ownership from a model region into a module-level function.** The operation's outside operands have ordinary MGP types; its body arguments have `!mogg.async<T>` types. The new MGP-ABI function uses the **body** argument types because it takes over that body verbatim. [Outlining implementation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp:67)

```transform
BEFORE · schematic isolated ReLU region
  %c1 = routine.async(%entry,%Xbuf,%Ybuf,%cpu)     # Four ordinary operands.
    ^body(ac,ax,ay,actx)                          # Four async-wrapped arguments.
      xptr = rebind(unpack(ax))                   # Input raw pointer.
      yptr = rebind(unpack(ay))                   # Output raw pointer.
      ctx  = rebind(unpack(actx))                 # Device context handle.
      call @F_relu(xptr,yptr,ctx)                 # Direct MOGG ABI helper.
      ready = async.ready                        # Success result.
      output ready                               # One chain result.

AFTER · model IR
  %c1 = compiled.call @F_relu_async(%entry,%Xbuf,%Ybuf,%cpu)
                                                    # Same operands/result/device.

AFTER · module sibling
  mogg.func @F_relu_async(ac,ax,ay,actx), ABI=MGP  # Async argument types retained.
    original region body                         # takeBody moves its block.
  mogg.func @F_relu(xptr,yptr,ctx), ABI=MOGG        # Direct helper remains separate.

Illustrative X=[-2,0,3,-1] → Y=[0,0,3,0]           # Computation preserved.
```

The native constructor labels the new function `ABIKind::MGP`, uses `takeBody`, registers its unique name in the symbol table, constructs `CompiledCallOp` and replaces the region's result users with the call's results. [Body ownership transfer](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp:91) [Call/result replacement](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineAsyncRoutines.cpp:102)

```cpp
stubOp.getBody().takeBody(kernelOp.getBody());
```

```cpp
kernelOp->replaceAllUsesWith(callOp);
kernelOp.erase();
```

When `FuncOp::emitMojo` later sees `ABI=MGP`, it writes an exported, non-raising function with one pointer-to-slots argument. It takes ownership of every async input slot, including unused chain slots, and keeps the owning handles alive through the compiled function. Return-value slots follow input/DPS slots. [ABI selection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1319) [Slot ownership and return offsets](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1354)

| Slot in this isolated four-operand example | Role |
| --- | --- |
| 0 | Async input chain |
| 1 | Async reference to X buffer |
| 2 | Async reference to Y destination buffer |
| 3 | Async device-context value |
| 4 | Destination slot for returned chain |

The emitter calculates result slot indices with `i + numArguments`. Thus the one chain result uses slot 4 in this example; Y is already supplied as a destination buffer and is not an additional by-value result. The slots carry handles/references, not an inline copy of all tensor elements. [Result-slot mapping](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1384)

```cpp
uint64_t numArguments = getBodyBlock().getNumArguments();
for (uint64_t i = 0; i < getNumResults(); ++i) {
  state.mojoReturnValues.emplace_back(
      runtimeArgument.pointerOffsetGet(i + numArguments));
}
```

**TEST · ReLU tests the two exception contracts.** `infra-add-try-except-blocks.mlir` begins as an MO ReLU graph, runs graph/kernel lowering, outlines async routines, then applies `--mogg-add-try-except-blocks`. The outer async function is checked for `mogg.try` with no `mogg.raise` in the checked interval. The direct `@__relu` helper is checked for a try/trace body and a re-raise in its exception block. [Test RUN and CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-add-try-except-blocks.mlir:1)

The implementation wraps MOGG-ABI bodies with a re-raise, but a raising MGP-ABI body produces `AsyncErrorOp` values for its outputs. A non-raising MGP body skips that wrapping. This allows the runtime-facing exported routine to keep its non-raising ABI while reporting failure through async results. [Native exception branches](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/AddTryExceptBlocks.cpp:44)

```cpp
if (kernelOp.getAbi() != MOGG::ABIKind::MGP || !kernelOp.isThrows())
  return;
```

**Fusion boundary:** earlier `SequenceFusion` can combine compatible async bodies before this outlining, subject to fusion flags, duplication policy and compatible context types. Its two source/destination passes handle duplicable and nonduplicable producers separately; mismatched contexts reject the merge. Such a merged wrapper can contain several helper calls, so one wrapper does not establish one GPU launch. [Compatibility](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/SequenceFusion.cpp:96) [Pass order](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/SequenceFusion.cpp:204)

For attention's isolated six-operand region, the same arithmetic would place its chain result at slot 6. Later primitive fusion and resource operands can change the final entry signature. The direct attention helper still receives its five handles, and that distinction survives outlining. [[mogg-ops:878]] [[mogg-ops:1420]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-attention-mojo-resource" markdown="1">
<summary>Example 8 · How do attention's shapes and parameters become Mojo text, and how is that text attached to the model?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · Actual attention route, schematic names.** `@F_attention` carries four tensor pointers plus its GPU context. Its shape/create/call operations describe `f32[32,32]` views and the existing `FusedAttention.execute` implementation. The emitter writes source expressions for those operations without reading Q/K/V values. [[mogg-ops:1420]] [[mogg-ops:1482]] [[entry:57]]

```transform
BEFORE · schematic direct helper
  mogg.func @F_attention(qptr,kptr,vptr,optr,ctx)  # Five direct handles.
    shape = shape.create <[32,32]>               # Statically known extents.
    Q = tensor.create(qptr,shape,Input,false,f32)  # Read-only query view.
    K = tensor.create(kptr,shape,Input,false,f32)  # Read-only key view.
    V = tensor.create(vptr,shape,Input,false,f32)  # Read-only value view.
    O = tensor.create(optr,shape,Output,true,f32)  # Writable destination view.
    call.opaque FusedAttention(inputs=[Q,K,V],outputs=[O],context=ctx)
      parameters={BN:16,BD:8}                     # Explicit compile-time values.

AFTER · source-expression mapping, not an actual captured dump
  [32,32] → Coord(Idx[32],Idx[32])                 # Static extent encoding.
  Coord → row_major(Coord)                       # Element strides [32,1].
  qptr/kptr/vptr → input tensor constructors       # Same runtime storage.
  optr → output tensor constructor               # Same writable storage.
  {BN:16,BD:8} → parameter literals 16,8           # No runtime BN/BD buffers.
  helper.device="gpu" → target="gpu"              # Select authored GPU branch.
  execute arguments → [O,Q,K,V,ctx]               # Output-first registered ABI.
```

`ShapeCreateOp::emitMojo` delegates to `getShapeAsCoordMojoValue`. Static dimensions use compile-time `Idx` entries; dynamic dimensions use the corresponding runtime operand as `Scalar[DType.int]`. `TensorCreateOp::emitMojo` then builds the row-major layout and passes dtype/rank/mutability/IO/alignment to the registered tensor-construction primitive. [Shape emission](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1482) [Coordinate components](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/EmitMojoState.cpp:618) [Tensor emission](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1504)

```cpp
MojoValue coord =
    state.getShapeAsCoordMojoValue(getShape(), getDynamicDims());
state.setMojoValueFor(getResult(), state.builder.var(coord));
```

The resulting view selects element `[1,2]` at `1×32+2 = 34` elements, or 136 bytes from its own tensor's base pointer. Writing `Bq+136` and `Bo+136` is symbolic address arithmetic for distinct Q/O storage, not observed allocator addresses. The 4096-byte tensor payload remains unchanged by creating this layout.

`CallOpaqueOp::emitMojo` obtains the registered execute declaration and explicitly collects outputs before inputs. Contexts follow for this ordinary execute-style kernel. Its parameter inference turns the IntegerAttrs into literals and derives `target` from the surrounding helper's device kind. [Output-first call assembly](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1886) [Integer/target parameter inference](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/EmitMojoState.cpp:534)

```cpp
SmallVector<MojoValue> kernelOperands;
for (Value out : getOutputs())
  kernelOperands.push_back(state.getMojoValueFor(out));
for (Value in : getInputs())
  kernelOperands.push_back(state.getMojoValueFor(in));
```

**TEST · The emitter test supplies a smaller direct helper.** `codegen-map-abi-stub.mlir` starts as MO, imports the test kernel package, lowers to MOGG, conforms operands and outlines the direct helper, then invokes `--mogg-emit-mojo`. It deliberately stops before MOGG→MGP and async outlining, disables verify-each for that intermediate container, and checks a dumped `graph.mojo` file. This isolates the same direct-helper emitter used in the full pipeline; it does not exercise the MGP wrapper or kernel JIT. [Exact RUN and test scope](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:1)

These unchanged CHECK lines pin the static shape/layout expressions and output-first call. Bracketed capture names are FileCheck syntax, not literal generated variable names. [Emitter expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:24)

```text
// CHECK:   var [[PSHAPE:.*]] = layout.Coord(layout.Idx[2], layout.Idx[2])
// CHECK:   var [[PLAYOUT:.*]] = layout.row_major([[PSHAPE]])
// CHECK:   {{.*}}.MapLowerOnePlainOutput.execute[target = "cpu"]([[POUT]], [[PIN]], [[PC]])
```

The same file checks `Scalar[DType.int]` for a dynamic `batch` extent. For the real attention registration, dtype/rank are infer-only parameters, while BN/BD/target are explicit specialization inputs; its body dispatches to the authored CPU or GPU implementation. Using `[32,32]` does not generate an MO matmul/softmax graph that this emitter must rediscover. [Dynamic shape expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:35) [Attention execute signature/branch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/examples/custom_ops/kernels/fused_attention.mojo:73)

**The module changes after source emission succeeds.** `EmitMojo` emits built-in imports, the filename stems of extension packages, and registered kernel submodules. It visits module-level functions, combines text with deferred struct definitions, then copies those bytes into the `mojo` resource. A preexisting resource is updated instead of receiving a second name. [Import and function emission](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:147) [Resource creation/update](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:228)

```transform
BEFORE · schematic module ownership
  mgp.model @fused_attention
    execute → compiled.call @F_attention_async    # Runtime call still in IR.
  mogg.func @F_attention_async, ABI=MGP            # Runtime-slot entry body.
  mogg.func @F_attention, ABI=MOGG                 # Tensor-view/helper body.

AFTER EmitMojo
  resource "mojo" = emitted UTF-8 source bytes     # Source text retained by IR.
  mgp.model @fused_attention
    init → compiled.load(["F_attention_async"],resource="mojo")
                                                    # Names required by this model.
    execute → compiled.call @F_attention_async    # Same call plan.
  emitted mogg.func operations erased             # Their code now lives in text.

NEXT, when JIT is enabled
  JitCompileKernels(resource)                     # Parse/specialize/compile source.
  later runtime-call lowering and MEF serialization # Produce the final C1 artifact.
```

The symbol set is collected from that model's `CompiledCallOp` operations. The pass skips `compiled.load` insertion if that set is empty. A helper-emission failure signals pass failure before this successful text handoff. The device-info mapping is removed after emission consumes it. [Called-symbol guard and load creation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:255) [Failure branch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:193) [Helper/mapping cleanup](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp:276)

For a concrete mathematical check on attention, set Q=K=0 and every column in V row j to j, for j=0…31. Uniform softmax weights give every O element `(0+…+31)/32 = 15.5` both before and after the representation changes. These values illustrate preserved computation; no such runtime inputs, calculated output, PTX or cubin are produced by `EmitMojo` itself. [[entry:78]] [[kernel:88]] [[gc:614]]

</div>
</details>

<details class="qna-cell" id="qna-mgp-example-device-graph-staging-finalization" markdown="1">
<summary>Example 9 · How does a marked model acquire staging, graph-building and graph-execution operations, and why does ordinary attention skip these rewrites?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · The model marker selects this path.** The ordinary attention caller places its tensors on `gpu:0` but omits `is_device_graph`, whose default is false. A comparison model explicitly carrying `device_graph` reaches the staging implementation. These native guards are in different passes; both consult the model attribute. [Attention construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/examples/custom_ops/fused_attention.py:41) [Graph default](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/graph/graph.py:831) [Staging guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/StageDeviceGraphsPass.cpp:194) [Finalization guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:379)

```cpp
if (model.getIsDeviceGraph())
  stageDeviceGraphOps(&getContext(), model);
```

```cpp
if (!model.getIsDeviceGraph())
  return;
```

**TEST · The first checkpoint starts with MGP async routines.** `mgp-stage-device-graphs.mlir` runs `mo-opt --mgp-stage-device-graphs --allow-unregistered-dialect --split-input-file | FileCheck`. Its marked `@dg` input contains one routine extracting an input buffer and another rebuilding a tensor view from that buffer. It checks ownership and operand wiring; these routines do not perform attention. [RUN and staging CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-stage-device-graphs.mlir:1) [Input routine bodies](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-stage-device-graphs.mlir:25)

```transform
BEFORE · schematic @dg, marked device_graph
  execute(%entry,%ctx,%gpu,%X:f32[2,3])           # X is one GPU tensor handle.
    %A = routine.async(%X) { extract buffer }     # Existing async body A.
    %Y = routine.async(%A) { make f32[2,3] view }  # Existing async body B.
    mef.output %entry,%Y                         # Y escapes the two routines.

AFTER StageDeviceGraphs · schematic
  execute(%entry,%ctx,%gpu,%X:f32[2,3])           # Same model interface.
    %Y = device_graph.staging inputs(%X)         # One isolated enclosing op.
      ^body(%builder,%xin:f32[2,3])               # Builder + bare tensor argument.
        %A = routine.async(%xin) { extract buffer } # Original body moved here.
        %B = routine.async(%A) { make f32[2,3] view } # A→B edge stays internal.
        device_graph.staging.yield %B            # Expose the escaping tensor.
    mef.output %entry,%Y                         # Consumer rewired to staging.

Illustrative payload: X=[[1,2,3],[4,5,6]]          # 2×3×4 = 24 bytes of f32.
  output view still names the same six elements  # No arithmetic kernel added.
  element [1,2] lies at base + 20 bytes           # Symbolic offset, not a trace.
```

The pass gathers the execute block's async routines in block order, includes stable-address creation when its producer is staged, and collects values defined outside that set as explicit inputs. It then **moves** those operations into the new body and remaps only their external operands. Escaping uses become results of the staging operation; a later canonicalizer can deduplicate repeated yields. [Selection and free-value collection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/StageDeviceGraphsPass.cpp:88) [Move, remap and yield construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/StageDeviceGraphsPass.cpp:145)

```cpp
for (Operation *op : stagedOps)
  op->moveBefore(builder.getBlock(), builder.getBlock()->end());

for (Operation *op : stagedOps)
  for (OpOperand &operand : op->getOpOperands())
    if (Value mapped = freeMap.lookupOrNull(operand.get()))
      operand.set(mapped);
```

The `CHECK` lines require a bare tensor argument beside the builder, A consuming that argument, B consuming A, and the model output consuming the staging result. The `CHECK-NOT: mogg.async.unpack` before A rules out an extra unpack at the new boundary; the existing routines retain their own async pack/unpack operations. A separate guard case converts `mgp.guard(chain,X)` to a forwarding async region before staging, so a guard between staged producers does not leave a use-def cycle outside the new region. [Boundary checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-stage-device-graphs.mlir:13) [Guard test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-stage-device-graphs.mlir:50) [Native guard rewrite](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/StageDeviceGraphsPass.cpp:38)

**The second checkpoint starts later in the pipeline.** Production runs collective partitioning, dependency computation/simplification and build-graph stub generation between staging and finalization. The independent `mgp-finalize-device-graphs.mlir` test starts with an already prepared `mgp.device_graph.collective`; its `RUN` applies only `--mgp-finalize-device-graphs` with split inputs and unregistered test operations. The following chain example corresponds to its `@dg_chain` fixture, not a claimed end-to-end dump of the staging fixture above. [Production order](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:417) [Finalization RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-finalize-device-graphs.mlir:1) [Chain fixture](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-finalize-device-graphs.mlir:62)

```transform
BEFORE FinalizeDeviceGraphs · schematic @dg_chain
  %done,%Y = device_graph.collective(%builder)    # Marked model; gpu:0 region.
    %R = test.identity(%X:f32[2,3])               # Stand-in tensor producer.
    %c = test.make_chain()                       # Stand-in host completion.
    collective.yield %c,%R                       # Old interface includes chain.
  mef.output %done,%Y                            # Both collective results escape.

AFTER · schematic, inside the model's execute block
  %G = routine.async(%ctx,%gpu,%X)               # Three inputs; one graph result.
    unpack async inputs → ctx,gpu,X              # Builder wrapper owns handles.
    %cache = device_graph.cache(ctx)             # Model-associated graph cache.
    %g = device_graph.collective.create(ctx,cache,gpu,inputs=[X])
                                                  # New build-time region.
      ^graph(%builder,%modelCtx,%xin:f32[2,3])    # Builder, context, tensor.
        %R = test.identity(%xin)                 # Original operation cloned.
        %c = test.make_chain()                   # Cloned; no longer yielded.
        device_graph.yield %R                    # Graph signature excludes chain.
    mogg.output pack(%g)                        # Publish graph handle.
  %done,%Y = device_graph.collective.execute(%G,inputs=[%X])
                                                  # Fresh completion + tensor.
  mef.output %done,%Y                            # Old users now consume execute.

Graph type: (f32[2,3] on gpu:0) → (f32[2,3] on gpu:0)
  six input elements → six identity output elements # No chain in graph I/O.
```

Finalization gathers non-chain captures and yields, resolves each graph device to an execute-block context, and constructs the async build wrapper. Inside that wrapper it creates the cache lookup and `DeviceGraphCollectiveCreateOp`. It **clones** each old region into the create operation, mapping the old builder and captured values to the new arguments, and omits chain values from its yields. Unlike staging's move, these clones belong to the replacement create operation; the original collective is erased after its users are rewired. [Region interface](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:148) [Wrapper and create construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:264) [Clone and remap](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:300)

The create operation's graph-body block receives a builder, a model context and the region's tensor inputs. Its graph type describes only the explicit tensor inputs and yielded outputs; the builder and context are construction parameters. [Block arguments and graph signature](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/MGPDeviceGraphOps.cpp:1576)

```cpp
Value executeChain = executeOp.getChain();
unsigned outputIdx = 0;
for (Value result : collectiveOp.getResults()) {
  if (isChain(result))
    result.replaceAllUsesWith(executeChain);
  else
    result.replaceAllUsesWith(executeOp.getOutputs()[outputIdx++]);
}
```

This rewrite preserves the host completion contract: each old collective chain result is replaced by the new execute operation's chain, while tensor results map in order. The test checks a chain-free `!mgp.device_graph` signature and requires `mef.output` to use the execute operation's fresh chain and tensor result. Its unregistered `test.identity` and `test.make_chain` operations are structural placeholders, so the six-element identity above is illustrative arithmetic, not a numerical test result. [Native result replacement](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:346) [Signature and result CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-finalize-device-graphs.mlir:68)

| Condition | Native outcome and inspected evidence |
| --- | --- |
| Marked model with eligible execute routines | Staging encloses the routines; the first fixture checks its inputs, body and results |
| Marked model with no staged operations | Staging returns without creating an empty region |
| Unmarked GPU model | Staging leaves its routine in execute; finalization returns at the model guard |
| Prepared collective whose device has no execute context | Finalization reports an error instead of selecting a different device |

The staging test's `@not_dg` checks that `mogg.routine.async` remains and `mgp.device_graph.staging` does not appear. The finalization test deliberately supplies an unmarked staging input and checks that it stays unchanged without a collective create operation. The diagnostic test applies the finalization pass with `--verify-diagnostics` and expects the missing-device-context error, matching `collectDeviceContexts`. [Empty staged set](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/StageDeviceGraphsPass.cpp:113) [Unmarked staging test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-stage-device-graphs.mlir:107) [Unmarked finalization test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-finalize-device-graphs.mlir:39) [Missing-context test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MGPDialect/Transforms/mgp-finalize-device-graphs-errors.mlir:31) [Native context validation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp:205)

**Actual attention connection:** the default unmarked `fused_attention` model follows the ordinary async routine, outlining and emission path described in Examples 7–8. The marked fixtures above explain what the compiler prepares when the marker is present. Creating these IR operations schedules later graph construction and execution; source inspection here establishes no completed capture, graph replay, launch count or timing measurement. [[entry:41]] [[gc:417]] [[gc:440]]

</div>
</details>
