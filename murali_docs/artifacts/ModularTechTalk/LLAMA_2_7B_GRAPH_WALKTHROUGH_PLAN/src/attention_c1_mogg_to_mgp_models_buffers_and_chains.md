## QnA 1 · What is the main change at MOGG→MGP?

**The graph becomes a runtime model plan with two regions: init and execute.** Its tensors become runtime tensor/storage representations, and its implementation call gains explicit buffers and dependency handling. These are IR construction steps during C1 compilation; runtime allocation and execution happen when that plan is used later. [[mogg-to-mgp:296]] [[mgp-ops:1141]] [[entry:74]]

```transform
BEFORE · inside compiler clone IR1
  mo.graph @fused_attention : (T0,T0,T0) → T0
    mogg.routine → outlined helper @F_attention
    mo.output %o

AFTER · schematic model plan
  mgp.model @fused_attention
  ├─ init region → one block
  │  ├─ create initial chain
  │  ├─ mgp.context.create numSlots=0 initially
  │  └─ return chain, model context, weights registry, devices
  └─ execute region → one block
     ├─ receive model state + Q/K/V tensors
     ├─ extract Q/K/V storage handles
     ├─ plan O storage
     ├─ mogg.routine.async → call @F_attention
     ├─ create result tensor from O storage, guarded by call's chain
     └─ mef.output current chain, result tensor

A module-level @F_attention helper still accompanies the model.
```

The new model takes the graph's symbol name. The lowering removes the old graph in IR1 after block lowering and isolation fixup. G1's original `mo.graph` remains in its separate source module. [[mogg-to-mgp:393]] [[mogg-to-mgp:1974]] [[frontend:377]]

## QnA 2 · What is the expanded native lowering call stack?

**SOURCE · The pass consumes graphs and already-outlined routines in IR1.** `gcPipelineImpl` copies `minimumCpuAlignment` and, when supplied, `compiledFrameworkLabel` into `MOGGToMGPOptions`, then schedules the conversion. `runGCPipeline` later calls `pm.run(moduleOp)`; that is when the following native bodies rewrite the module. [Options and scheduling](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:538) [[gc:611]] [[gc:654]]

```calltree
gcPipelineImpl → MOGGToMGPPass::runOnOperation [[gc:611]] [[mogg-to-mgp:2406]]
├─ obtain the module's cached SymbolTable; retain it through lowering [[mogg-to-mgp:2409]]
├─ worklist + graphDependencies + completedGraphs [[mogg-to-mgp:2414]]
│  ├─ gather each MO::CallOp callee; this is graph-call ordering [[mogg-to-mgp:2423]]
│  ├─ unmet dependencies → put caller back on the worklist [[mogg-to-mgp:2445]]
│  └─ full sweep without progress → diagnostic + signalPassFailure [[mogg-to-mgp:2433]]
├─ MOGraphLowering(graph, frameworkLabel, minimumCpuAlignment, symbolTable).run() [[mogg-to-mgp:2451]]
│  ├─ createModelOp() [[mogg-to-mgp:1950]]
│  │  ├─ collect used devices; add host and sort host first [[mogg-to-mgp:217]]
│  │  ├─ register per-device type builders BEFORE converting signature types [[mogg-to-mgp:303]]
│  │  ├─ convert inputs/results; preserve original argument locations [[mogg-to-mgp:308]]
│  │  ├─ carry names + device indices + subgraph/device-graph flags [[mogg-to-mgp:328]] [[mogg-to-mgp:379]]
│  │  ├─ ModelOp::create → init block + execute block, including implicit args [[mgp-ops:1141]]
│  │  ├─ hand graph's symbol name to model in the cached symbol table [[mogg-to-mgp:393]]
│  │  ├─ init builder: chain.create + context.create 0 [[mogg-to-mgp:403]]
│  │  ├─ execute builder: arg0=root/current chain; arg1=model context [[mogg-to-mgp:412]]
│  │  └─ setupBufferBuildersForAllDevices(): CPU first, then GPU [[mogg-to-mgp:258]] [[mogg-to-mgp:431]]
│  │     └─ setupForModel reserves slots; CPU inserts runtime.create [BufferBuilder.cpp:29](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp:29)
│  ├─ lowerArg(original arg, execute input) [[mogg-to-mgp:1953]]
│  │  ├─ static shape → direct value map; no dynamic-dimension operands [LoweringState.cpp:180](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/LoweringState.cpp:180)
│  │  └─ symbolic shape → extract dims, bind params, guard repeated-param checks [LoweringState.cpp:191](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/LoweringState.cpp:191)
│  ├─ lowerBlock → levelwiseTopologicalSort → lowerOp for every op [[mogg-to-mgp:1758]]
│  │  ├─ reshape/rebind/constant/transfer/call/control flow → dedicated lowerer [[mogg-to-mgp:1717]]
│  │  ├─ LowerableToMGPInterface → RoutineOp::lower → lowerToAsync [[mogg-to-mgp:1749]] [[mogg-ops:723]]
│  │  │  ├─ validate pointer/output/context block-argument count [[mogg-ops:734]]
│  │  │  ├─ use explicit op chain when present; merge input producer guards [[mogg-ops:751]] [[lower-state:753]]
│  │  │  ├─ tensor/buffer operands → extract storage; mutable inputs get m [[mogg-ops:769]] [[lower-state:808]]
│  │  │  ├─ createOutputStorage: spec + byte size + aligned allocation [[mogg-ops:789]] [[lower-state:826]]
│  │  │  ├─ materialize named dynamic dims; anonymous/missing name fails here [[mogg-ops:803]]
│  │  │  ├─ append one context, or validate multiple unique device contexts [[mogg-ops:825]]
│  │  │  ├─ construct AsyncRoutineOp + one async body arg per operand [[mogg-ops:865]] [[mogg-ops:878]]
│  │  │  ├─ unpack args 1..N; rebind storage/dim/context to direct handles [[mogg-ops:894]]
│  │  │  ├─ IRMapping: original body arguments → handles; clone stub call [[mogg-ops:913]]
│  │  │  ├─ embedded buffer constants → MGP resources carried by region [[mogg-ops:920]]
│  │  │  ├─ emit ready chain; pack shape-function/opaque results if present [[mogg-ops:951]]
│  │  │  └─ register tensor guards + explicit output chain; bind returned dims [[mogg-ops:980]] [[mogg-ops:999]]
│  │  └─ MO::OutputOp → init state tuple + execute chain/results [[mogg-to-mgp:460]]
│  └─ fixupIsolatedFromAbove; erase temporary model and old graph [[mogg-to-mgp:1974]]
├─ deferSideStreamJoins: move side-stream joins to first consumer when present [[mogg-to-mgp:2346]]
└─ nested AnonymizeDimNames pass; failure propagates to parent pass [[mogg-to-mgp:2462]]
```

This example contains one top-level graph and no graph-to-graph calls, so the dependency worklist has one graph to lower. The general implementation also handles subgraphs and detects cycles. [[entry:41]] [[mogg-to-mgp:2423]] [[mogg-to-mgp:2433]]

| State owned by the lowering | Concrete attention value | What happens to it |
| --- | --- | --- |
| Original graph arguments in IR1 | Q, K, V of type T0 | Keys in the lowering map; map to execute arguments 5, 6, 7 |
| Init / execute `OpBuilder` | Insertion points in different blocks | Insert runtime-plan operations without executing them |
| Parameter bindings | No dynamic dimensions in T0 | The dynamic-dimension branch is skipped |
| Buffer/type builders | CPU and GPU entries | Construct device-specific types, storage and context operations |
| New `AsyncRoutineOp` | Six operands, one chain result | Owns a newly constructed region with six async arguments |
| Module symbol table | `@fused_attention` and direct helper symbol | Model replaces graph's name; helper remains a module sibling |

The final isolation repair can create a replacement model, so the first `ModelOp` created inside `createModelOp` is an intermediate IR owner. G1 is outside this rewrite. [Lowering state and model replacement](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:1947) [[frontend:377]]

## QnA 3 · What are the model's exact argument positions for this example?

The device collector always includes the host and sorts it first. It discovers `gpu:0` from the attention types/device attributes, giving `[cpu:0,gpu:0]`. These are model-plan device entries; their ordering need not match the Python session's original device-list order. [[mogg-to-mgp:217]] [[mogg-to-mgp:238]]

```state
Model name = "fused_attention"
Devices = [cpu:0, gpu:0]
Argument names = ["input0","input1","input2"]
Result names = ["output0"]
Input device indices = [1,1,1]
Result device indices = [1]

INIT block arguments: THREE implicit arguments
  0  weights registry
  1  CPU device context
  2  GPU device context

EXECUTE block arguments: EIGHT total
  0  entry chain             : !mef.chain
  1  model state context     : !mgp.context
  2  weights registry
  3  CPU device context
  4  GPU device context
  5  Q runtime tensor
  6  K runtime tensor
  7  V runtime tensor

User-visible tensor inputs remain THREE.
```

Model builders prepend implicit state/device arguments, then append the converted explicit inputs. Argument names come from Graph.output metadata and are carried into the model. The weights registry argument exists even though this graph's expected weights are `{}`. [[mgp-ops:951]] [[mgp-ops:978]] [[mgp-ops:1147]] [[graph:1479]] [[mogg-to-mgp:328]]

`!mgp.context` is per-model runtime state storage. Its creation operation starts with `numSlots=0`; later passes may assign cache slots. It is separate from MLIRContext, shared MAXContext and the GPU device context argument. Do not read the initial zero as a guaranteed final slot count. [[mogg-to-mgp:406]] [[invariant:80]]

**The current conversion already increments that count.** `setupBufferBuildersForAllDevices` invokes `setupForModel` for CPU and GPU. Each call reserves one slot; the CPU call also inserts `mgp.runtime.create`. Starting at zero, this example therefore reserves slots 0 and 1 and reaches two slots during setup, before later cache-slot allocation. The GPU reservation does not create a second `mgp.runtime.create` in this helper. [Device setup order](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:258) [Slot increment and host-only runtime creation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp:29)

## QnA 4 · How does T0 turn into a tensor spec and an output buffer?

The output spec retains **shape `[32,32]` and dtype `float32`**. Its payload size is `32 × 32 × 4 = 4096` bytes. Buffer creation records this size, a device-specific alignment and the GPU context. The allocation op describes future runtime work. [[lower-state:826]] [[lower-state:871]] [[mgp-types:413]] [[mgp-types:423]]

```transform
MO/MOGG value
  %o : T0 = float32[32,32] on gpu:0

MGP storage description, schematic
  %specO = tensor_spec.create(shape=[32,32], dtype=f32)
  %bytes = index.constant 4096
  %bufO  = buffer.alloc(%guard, %bytes, %gpuContext, alignment)
  %chA   = async_attention(..., output=%bufO, context=%gpuContext)
  %out   = tensor.create(%chA, %bufO, %specO)

loweredValues[%o] = %out
chainGuard[%out]  = %chA
```

Q/K/V input tensors already represent caller-provided storage at execution time. The lowering extracts their buffer handles and specs; it does not insert three copies just to call the helper. Their mathematical payload sizes are also 4096 bytes each, but the caller creates those buffers later in the example. [[lower-state:808]] [[entry:78]]

The compiler chooses alignment through the driver helper. The diagram leaves it as `alignment` rather than inventing a host-dependent allocator value. Neither 4096 nor a pointer type is an actual observed GPU allocation address. [[lower-state:881]]

## QnA 5 · What goes into the asynchronous routine, and why is there a chain?

For this static, non-mutating tensor call, the async routine receives a leading guard chain, three input buffers, one output buffer and the GPU context. Its body wraps those operands as async values. The helper itself still takes five direct handles, so the chain does not become an extra argument to `FusedAttention.execute`. [[mogg-ops:763]] [[mogg-ops:788]] [[mogg-ops:878]]

```state
mogg.routine.async operands: SIX
  0  %guard       input dependency chain
  1  %bufQ        input buffer
  2  %bufK        input buffer
  3  %bufV        input buffer
  4  %bufO        output buffer
  5  %gpuContext  context

inOutSignature = "iiiio."
  i = input; o = output; . = neither buffer input nor output

body arguments: async wrappers for these SAME six operands
  skip body argument 0 when preparing the helper call
  unpack + rebind arguments 1..5
    → qptr, kptr, vptr, optr, ctx
    → mogg.call @F_attention(qptr,kptr,vptr,optr,ctx)

region result: one !mef.chain
  no dynamic shape result; no by-value opaque result
```

Input guards order the call after producers that supply its buffers. Output tensor construction is guarded by the region's chain. This is compiler IR describing dependency relationships, not a thread waiting on the GPU while C1 is being built. [[mogg-ops:751]] [[lower-state:818]] [[mogg-ops:999]]

**TEST · inspected expectation:** the conversion test checks async buffer unpacking, rebinding to `!mogg.pointer`, forwarding the helper call, and producing an async ready chain. Its smaller one-input case has the same mechanism. [[test-mgp:10]] [[test-mgp:25]]

## QnA 6 · How are results connected? Is the output chain always the attention chain?

**The result tensor's guard and the model's current execute chain are separately tracked.** For this chainless custom call, `createOutputs(regionResultChain, storage)` registers the tensor against the call's chain. The output lowering separately emits the current execute chain and the lowered tensor. [[mogg-ops:999]] [[lower-state:891]] [[mogg-to-mgp:472]]

```state
%chA   = asynchronous attention routine's chain result
%out   = tensor(%bufO, %specO), guarded by %chA

MO output:   mo.output %o
MGP output:  mef.output %currentExecuteChain, %out

%currentExecuteChain need not be spelled %chA.
The tensor dependency still records that %out depends on the attention call.
```

This avoids drawing an unsupported update to the ambient chain for a chainless operation. Later chain simplification and routine fusion may change the exact graph of guards, so the diagram names roles rather than claiming a final compiler dump. [[mogg-ops:1001]] [[gc:389]] [[gc:432]]

The init terminator returns the initialization chain, model state context, weights registry and device contexts. Those are the implicit values used by the execute entry. [[mogg-to-mgp:461]] [[mgp-ops:964]]

## QnA 7 · Which optimizations become possible now?

MGP exposes facts the earlier tensor graph did not: storage size, buffer lifetimes, ordering edges and the separation between one-time init work and per-call execute work. The following stage can fold static spec queries, simplify chains and reuse eligible buffers. [[gc:387]] [[gc:391]] [[gc:394]]

```transform
Example enabled by the new static spec
  get_dim(spec<[32,32],f32>, axis=0) → 32 : index
  get_dim(spec<[32,32],f32>, axis=1) → 32 : index
  logical payload byte count        → 4096

Example boundary
  Q/K/V/O are all used by the same attention call.
  Equal sizes do not authorize making them alias each other.
```

The dimension fold is implemented in `TensorSpecGetDimOp::fold`. Buffer reuse additionally requires compatible size/device/alignment and safe dependency ordering; buffers used by the same operation are rejected as reuse candidates. The next page follows those passes and Mojo emission. [[mgp-ops:484]] [[reuse:29]] [[reuse:77]] [[reuse:157]]

## QnA 8 · How does MOGG→MGP change the blocks, operands, storage and chains in fused attention, dynamic shapes, mutable writes, reshapes, transfers, external weights and subgraph calls? Show worked before/after examples and explain how each test reaches the native lowering.

**SOURCE + TEST · These seven examples follow the conversion itself.** Teaching IR is explicitly schematic; `TEST` means inspected input and expected text, without executing the compiler test. A test's `RUN` command parses that file into its own module and schedules named passes. Production lowering operates on IR1 and does not call a `.mlir` file. `CHECK-LABEL` selects the expected function/model, `CHECK` matches subsequent text, `CHECK-NEXT` requires the next output line, and `CHECK-NOT` excludes a pattern within its check interval.

<details class="qna-cell" id="qna-mogg-mgp-example-attention-storage" markdown="1">
<summary>Example 1 · How does the actual three-input attention call acquire one output buffer and an asynchronous body?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · Actual attention route, schematic IR.** Use T0 = `float32[32,32]` on `gpu:0`, `BN=16` and `BD=8`. The direct helper has Q/K/V/O pointer arguments and one context argument. MOGG→MGP keeps that helper and constructs a runtime-plan wrapper around its call. [[entry:36]] [[entry:57]] [[mogg-ops:734]]

```transform
BEFORE · graph body in IR1
  %o = mogg.routine(%q,%k,%v) : (T0,T0,T0) → T0  # Tensor-level inputs/result.
    ^body(qptr,kptr,vptr,optr,ctx)                 # Five direct handles.
      mogg.call @F_attention(qptr,kptr,vptr,optr,ctx) # Existing outlined call.
  mo.output %o                                    # One graph result.

AFTER · inside new model.execute
  %bQ = tensor.extract.buffer %Q                  # Borrow caller Q storage.
  %bK = tensor.extract.buffer %K                  # Borrow caller K storage.
  %bV = tensor.extract.buffer %V                  # Borrow caller V storage.
  %sO = tensor_spec.create <32x32xf32>             # Result shape/dtype metadata.
  %nO = index.constant 4096                       # 32 × 32 × 4 bytes.
  %bO = buffer.alloc %guard,%nO,%gpu,align=A       # Describe future O storage.
  %cA = mogg.routine.async(%guard,%bQ,%bK,%bV,%bO,%gpu)
    ^body(async_chain,async_Q,async_K,async_V,async_O,async_gpu)
      qptr = rebind(unpack(async_Q))              # MGP buffer → pointer handle.
      kptr = rebind(unpack(async_K))              # Same for K.
      vptr = rebind(unpack(async_V))              # Same for V.
      optr = rebind(unpack(async_O))              # Writable output handle.
      ctx  = rebind(unpack(async_gpu))            # Context handle.
      mogg.call @F_attention(qptr,kptr,vptr,optr,ctx) # Preserve computation.
      ready = mogg.async.ready                   # Region's completion value.
      mogg.output ready                          # One chain result.
  %O = tensor.create %cA,%bO,%sO                  # Tensor guarded by the call.
  mef.output %currentExecuteChain,%O             # Model output tuple.
```

The fresh allocation comes from `createOutputStorage`. The native code extracts existing input storage, but creates output storage before assembling the async operands. Chain results and by-value opaque results are excluded from this allocation path. [Input extraction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/LoweringState.cpp:808) [Output allocation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/LoweringState.cpp:775)

```cpp
for (const MGP::MOOutputStorage &out : storage) {
  operands.push_back(out.buffer);
  semantics.push_back(InOutSignatureAttr::kOut);
}
```

This loop adds exactly one `o` entry here. Combined with the leading chain, three tensor inputs and context, the signature is `iiiio.`. The body receives six async arguments, while `i + 1` below deliberately skips the chain when mapping to the five original helper arguments. [Operand construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:798) [Unpack loop](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:893)

```cpp
Value unpacked = MOGG::AsyncUnpackOp::create(
    builder, loc, regionBody->getArgument(i + 1));
```

| Boundary | Before | After |
| --- | --- | --- |
| Top-level executable object | `mo.graph` with one body | `mgp.model` with init and execute blocks |
| User inputs | Three T0 values | Three runtime tensors after five implicit execute arguments |
| Attention output | One SSA tensor | Allocation operation + spec + guarded tensor |
| Direct helper body | Five handles | Same five-handle call, cloned through `IRMapping` |
| Runtime wrapper | Absent | Six async operands and one chain result |

**TEST · A smaller static custom call checks the same mechanics.** `map-kernel-region.mlir` supplies an MO graph with one `f32[2,2]` input/output. Its `RUN` imports the test package, lowers MO→MAP→MOGG, conforms/ outlines the direct routine, then runs `--mogg-to-mgp`. The `@one_plain` checks require buffer unpack, pointer/context rebind, direct `mogg.call` and `async.ready`; `iio.` means chain/input/output/context. The helper's shape remains `[2,2]`. The test kernel's body is `pass`, so these checks establish ABI construction, not numerical output. [RUN and static CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOGGToMGP/map-kernel-region.mlir:1) [Test kernel declaration](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3566)

**Illustrative arithmetic for the real attention computation:** let every Q/K element be zero and let row j of V contain `j` in every column, for j = 0…31. Every score is zero, each softmax weight is `1/32`, and every output element is `(0+…+31)/32 = 15.5`. The representation change retains the same inputs and helper that compute this expression later; the compiler does not calculate or store this illustrative output during conversion. [[kernel:88]] [[entry:78]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-mgp-example-dynamic-dimensions" markdown="1">
<summary>Example 2 · Where does a runtime batch dimension come from, and why does its symbolic name disappear from the helper ABI?</summary>
<div class="qna-answer" markdown="1">

**TEST · `@dyn` enters as MO with `f32[batch,4]`.** The test uses the same `RUN` pipeline as Example 1. At the conversion boundary its outlined helper therefore has an additional `!mogg.dim<batch>` argument between the two pointers and context. `CHECK` expects an anonymous `!mogg.dim` after conversion, and the `NONAME-NOT` check prohibits a surviving named helper dimension anywhere in the module. [Dynamic input and checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOGGToMGP/map-kernel-region.mlir:45)

```transform
BEFORE · schematic MOGG boundary
  X : f32[batch,4]                                # One caller-supplied tensor.
  Y = routine(X) : f32[batch,4]                    # Same custom operation.
    call @F(xptr,yptr,batch_dim,ctx)                # Named dim in helper ABI.

AFTER · schematic model.execute
  %sx = tensor.extract.tensor_spec %X              # Runtime shape source.
  %n  = tensor_spec.get_dim[0] %sx                 # batch length as index.
  %nd = dimension(%n)                             # KGEN dimension value.
  %sy = tensor_spec.create(%n) : <?x4xf32>          # One dynamic extent.
  %bytes = %n × 4 × 4                             # Runtime output byte count.
  %by = buffer.alloc(%guard,%bytes,%cpu,A)          # Size follows this call.
  %cy = routine.async(%guard,%bx,%by,%nd,%cpu)      # Signature "iio..".
    call @F(xptr,yptr,anonymous_dim,ctx)            # Still passes the same n.
  %Y = tensor.create(%cy,%by,%sy)                  # Runtime result shape [n,4].

Illustrative shape values:
  batch=5 → 20 f32 elements → 80-byte payload       # One possible invocation.
  batch=8 → 32 f32 elements → 128-byte payload      # Another invocation.
```

`lowerArg` extracts the dynamic dimensions from the model input and binds each source parameter to a lowered value. If another top-level input uses the same parameter, it creates an equality assertion against the earlier binding and guards that input; subgraphs inherit the caller's check. Thus `X:[5,4]` and another input declared `[batch,4]` with actual shape `[8,4]` do not silently choose a new batch value. [Parameter binding and repeated-dimension guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Interfaces/LoweringState.cpp:191)

```cpp
auto existingValue = params.lookupLowered(parentRegion, inDimParam);
auto loweredValue = indexToDimension(builder, loc, outDim);
```

`RoutineOp::lowerToAsync` later looks up `batch` through `lowerExpression`, appends that value as a non-buffer operand, and rebinds its async-unpacked form to an anonymous helper dimension. [Dimension materialization](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:803) [Anonymous rebind](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:900)

```cpp
if (isa<MOGG::DimType>(handleType))
  handleType = MOGG::DimType::get(ctx, /*name=*/{});
```

The final `AnonymizeDimNames` pass applies matching changes to helper signatures and their bodies. Removing the spelling `batch` permits structurally identical routines named with `seq` to be compared later without losing the actual dimension operand. [Final nested pass](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2462)

**Guard and skipped cases:** a missing name on an incoming dimension argument is an error because the value cannot be looked up. Static attention has no dimension arguments to materialize, and its size stays 4096. The test's no-op registration supplies a signature exercise; the 80/128-byte arithmetic describes required capacity, not generated test values. [Missing-name diagnostic](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:809) [Test registration](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3566)

</div>
</details>

<details class="qna-cell" id="qna-mogg-mgp-example-mutable-write-order" markdown="1">
<summary>Example 3 · How do two writes to the same mutable buffer retain their order?</summary>
<div class="qna-answer" markdown="1">

**TEST · `@two_writes_chained` starts with MO buffer stores.** The `RUN` command includes `--map-fuse-mutable-loads` before MAP outlining and MOGG lowering; its final `--mogg-to-mgp` lowers the resulting buffer-writing routines. Its separate `@mutable_store` case checks `inOutSignature = <"iim.">`. [Test pipeline and mutable signature](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOGGToMGP/map-kernel-region-mutable.mlir:1)

```transform
BEFORE · schematic form of @two_writes_chained
  a = [1,2,3,4] : si32[4]                         # Illustrative first input.
  b = [9,8,7,6] : si32[4]                         # Illustrative second input.
  %buf = mo.buffer.create : si32[4]               # One mutable 16-byte buffer.
  %c1 = mo.mutable.store(%buf,%a,%c0)              # First write.
  %c2 = mo.mutable.store(%buf,%b,%c1)              # Second waits for first.
  mo.output %buf,%c2                             # Return buffer and write chain.

AFTER · schematic dependency structure
  %c1 = routine.async(%guard1,%aBuffer,%buf,%cpu)  # "iim.": buf is mutated.
  %g2 = grt.chain.create(%entry,%c1)              # Retain explicit store edge.
  %c2 = routine.async(%g2,%bBuffer,%buf,%cpu)      # Second write sees same buf.
  mef.output %ambient,%bufTensor,%c2              # Implicit + explicit chains.

Logical contents after both writes: [9,8,7,6]      # Order matters.
```

The pointer type inside the old helper is type-erased. Mutability therefore comes from the original `!mo.buffer` operand, using this exact branch in `lowerToAsync`. Marking it `m` lets downstream dependency and lifetime analysis see the write. [Mutable operand handling](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:769)

```cpp
bool isMutable = inputIdx < inputs.size() &&
                 isa<MO::BufferType>(inputs[inputIdx].getType());
semantics.push_back(isMutable ? InOutSignatureAttr::kMut
                              : InOutSignatureAttr::kIn);
```

The routine's explicit chain wins over the ambient chain before producer guards are merged. These statements are what preserve `c1 → c2`. [Explicit chain selection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:754)

```cpp
if (Value opChain = getChain()) {
  Value lowered = loweringState.lookupExecuteLoweredValue(opChain);
  ASSERT_STREAM(lowered, << "no lowered value for mogg.routine input chain: "
                         << opChain);
  inChain = lowered;
}
inChain = st.getGuardChain(inChain);
```

The test captures the first async result as `STORE1`, checks `grt.chain.create(%arg0, STORE1)`, then requires that merged value as the second routine's leading operand. This checks a concrete SSA edge; textual adjacency of the two calls would be insufficient. The conversion also maps the original explicit output-chain result to the region's result so `mo.output %buf,%c2` can lower. [Store-order CHECKs](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOGGToMGP/map-kernel-region-mutable.mlir:24) [Output-chain mapping](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1001)

For attention, Q/K/V are `!mo.tensor` inputs and receive `i`, while the newly allocated O receives `o`. It has no analogous mutable-store chain result. This is why the attention diagram keeps its tensor guard separate from the ambient execute chain. [[mogg-ops:781]] [[mogg-ops:798]] [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-mgp-example-reshape-view" markdown="1">
<summary>Example 4 · How can reshape produce a new tensor value without allocating or copying its elements?</summary>
<div class="qna-answer" markdown="1">

**TEST · `reshape.mlir` enters directly at `--mogg-to-mgp`.** Its `@static_reshape` graph already contains `mo.static.reshape`; neither Python reshape inference nor a matmul kernel runs in this test. Despite the operation's name, its result dimensions are symbolic parameters obtained from another graph argument. [RUN and source graph](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/reshape.mlir:1) [Input operation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/reshape.mlir:31)

```transform
Illustrative valid dimensions: D4=2,D5=6,D6=3,D7=4

BEFORE
  X : f32[2,6] = [[0,1,2,3,4,5],[6,7,8,9,10,11]] # Twelve elements.
  ShapeDonor : f32[3,4]                           # Only its extents are used.
  Y = mo.static.reshape(X) : f32[3,4]             # Same element order.
  mo.output Y                                    # Return reshaped tensor.

AFTER · schematic MGP
  %sx = tensor.extract.tensor_spec %X             # Bind X dimension params.
  %sd = tensor.extract.tensor_spec %ShapeDonor    # Bind result dimension params.
  %rows = tensor_spec.get_dim[0] %sd              # 3 at this invocation.
  %cols = tensor_spec.get_dim[1] %sd              # 4 at this invocation.
  %bx = tensor.extract.buffer %X                 # Existing storage handle.
  %sy = tensor_spec.create(%rows,%cols)           # New logical [3,4] spec.
  %Y  = tensor.create(%guard,%bx,%sy)             # Same storage, new tensor.
  mef.output %current,%Y                         # Return new tensor handle.

Y = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]              # No data rearrangement.
X[1,0] and Y[1,2] both select element 6             # Byte offset 6 × 4 = 24.
```

The native dispatch selects `lowerReshapeImpl` for `MO::Static::ReshapeOp`. That implementation obtains the input's buffer and creates a new spec from the result type, then reuses the buffer here. [Dispatch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:1728) [Complete reshape lowerer](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:931)

```cpp
Value inChain = opLoweringState.getGuardChain(chain);
opLoweringState.createOutputTensor(output, inChain, buffer, *tensorSpecOr);
```

No output-buffer allocation call occurs in that lowerer. The new tensor is guarded by the input's producer chains, so changing the view does not allow an early consumer of incomplete data. Distinct logical tensor views can describe one storage handle; this does not invent aliasing between independent mutable MO buffers. [Reshape guard and tensor creation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:934)

The `CHECK` block names the extracted buffer `BUF` and requires that same value in `mgp.tensor.create` with `NEW_SPEC`. It also checks the dimension-to-index conversions rather than treating the symbols D6/D7 as constants. The multi-reshape case expects nested `grt.chain.create` guards, which the following MGP cleanup can simplify. [Buffer/spec checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/reshape.mlir:22) [Repeated reshape guards](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/reshape.mlir:36)

**Boundary:** this example assumes the earlier reshape contract already established compatible element counts. MOGG→MGP does not repair a `[2,6] → [3,5]` request. The actual attention graph has no such extra reshape; its helper receives the original Q/K/V views. [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-mgp-example-device-to-host-transfer" markdown="1">
<summary>Example 5 · Why does an explicit device-to-host transfer need a copy, a synchronization chain and a lifetime hold?</summary>
<div class="qna-answer" markdown="1">

**TEST · `transfer.mlir` supplies `mo.transfer` directly to `--mogg-to-mgp`.** It describes a GPU tensor of five `si64` values returning on CPU. The expected sequence is `buffer.device_to_host → sync → extend.lifetime → tensor.create`; the last three checks use `CHECK-NEXT`. [Transfer test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/transfer.mlir:1)

```transform
BEFORE · schematic values for the test's type
  Xgpu : si64[5] = [10,20,30,40,50]               # 5 × 8 = 40-byte payload.
  H,%c1 = mo.transfer[%c0] Xgpu to cpu:0          # Explicit graph movement.
  mo.output H                                    # Host result.

AFTER · schematic MGP execute operations
  %db = tensor.extract.buffer %Xgpu              # Source GPU storage.
  %s  = tensor.extract.tensor_spec %Xgpu         # Shape/dtype stay [5],si64.
  %hb = buffer.alloc(%guard,40,%cpu,A)            # Fresh host destination.
  %copy = buffer.device_to_host(%guard,%db,%hb,%gpu) # Describe the copy.
  %done = sync(%copy,%modelContext,%gpu)          # Describe copy completion.
  extend.lifetime(%done,%hb)                     # Keep host destination alive.
  %H = tensor.create(%done,%hb,%s)                # Host read guarded by done.
  mef.output %current,%H                         # Model output.

After runtime completion: H=[10,20,30,40,50]       # Same values, new location.
```

`lower(MO::TransferOp)` first requires explicit source and destination devices. It extracts the input buffer/spec, merges the explicit transfer chain with value guards, allocates output storage using that same spec, and dispatches `lowerTransferImpl`. Failure to select a supported transfer path emits an error. [Transfer validation and rewrite](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:1086)

The device-to-host buffer builder creates the copy and sync operations, then explicitly retains the destination. It returns `syncChain`, which becomes the output tensor's guard. [Native device-to-host builder](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp:178)

```cpp
Value syncChain =
    MGP::SyncOp::create(builder, loc, copyChain, context, deviceContext,
                        /*bDevice*/ device);
MGP::ExtendLifetimeOp::create(builder, loc, syncChain, ValueRange{dstBuffer});

// The host cannot read dstBuffer until the d2h copy completes.
return syncChain;
```

| Direction | Retained buffer in this builder | Returned dependency |
| --- | --- | --- |
| Device→host | Destination host buffer through `syncChain` | `syncChain`, because CPU readers need completed copy |
| Host→device | Source host buffer through `syncChain` | `copyChain`, allowing subsequent same-context device work to remain stream-ordered |

The host→device branch deliberately returns the copy chain while retaining the host source until synchronization. The table states the current builder behavior; it does not turn every GPU operation into a host wait. [Host-to-device branch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Dialect/BufferBuilder.cpp:148)

**Actual attention:** its graph input and output types are all on GPU, so this `mo.transfer` comparison does not add a CPU copy to its model. The caller creates input buffers with `.to(device)`, then calls `model.execute(Q,K,V)` and `print(output)`; it contains no explicit output `.to("cpu")` call. [[entry:78]] [[entry:82]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-mgp-example-external-weights" markdown="1">
<summary>Example 6 · How does an external weight become a registry lookup, and when is a GPU copy omitted?</summary>
<div class="qna-answer" markdown="1">

**TEST · `constant-external-to-primitives.mlir` enters directly at this conversion.** Both graphs return GPU tensors named `foo` (`128xf32`) and `bar` (`256xf32`); they differ in the location of the externally supplied weight storage. Its `RUN` is `mo-opt %s --mogg-to-mgp | FileCheck %s`. [External-weight test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/constant-external-to-primitives.mlir:1)

```transform
BEFORE · CPU-resident external weight, schematic
  W = mo.constant.external(name="foo",device=cpu:0,align=4)
      : f32[128] on gpu:0                         # Storage source differs from result device.

AFTER · execute plan at this stage
  %name = mgp.string.create "foo"                 # Registry key.
  %host = buffer.constant.external(%chain,weights[%name,512,align=4])
                                                    # 128 × 4 bytes supplied by caller.
  %gpuB = buffer.alloc(%chain,512,%gpu,align=4)    # Destination allocation operation.
  %copy = buffer.host_to_device(%chain,%host,%gpuB,%gpu) # Transfer operation.
  %spec = tensor_spec.create <128xf32>            # Tensor metadata.
  %W = tensor.create(%copy,%gpuB,%spec)           # GPU-facing weight value.

BEFORE · GPU-resident external weight, schematic
  W = mo.constant.external(name="foo",device=gpu:0,align=4)
      : f32[128] on gpu:0                         # Storage already at requested device.

AFTER
  %name = mgp.string.create "foo"                 # Same registry key.
  %gpuB = buffer.constant.external(%chain,weights[%name,512,align=4],gpu:0)
                                                    # Reuse registered device storage.
  %W = tensor.create(%chain,%gpuB,%spec)          # No H2D copy inserted here.
```

`lower(ConstantExternalOp)` calculates byte size from dtype and element count, preserves `hasAlias` / placeholder metadata and builds the string key. The CPU branch delegates to `constHostBufferToTarget`; the device branch checks that operation and result devices match. [External-constant lowerer](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:850)

```cpp
DeviceRefAttr resultDevice = MO::getDeviceOrDefault(opResult.getType());
if (resultDevice != opDevice)
  return constExternalOp->emitOpError(
      Twine("expected op device ") + opDevice.getLabel() +
      " and result device " + resultDevice.getLabel() + " to match");
```

This branch prevents describing a `gpu:1` weight buffer as `gpu:0` storage without an actual transfer. In the CPU-source case, `constHostBufferToTarget` allocates 512 bytes for `foo` and 1024 bytes for `bar`, then creates host-to-device operations; the values themselves remain external. For illustrative `foo=[1,2,…,128]`, either valid route yields those same 128 values in the GPU tensor. [CPU-source transfer construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:809)

The first test block uses `CHECK-DAG` to find registry strings and allocations, then checks host-to-device copies. The second explicitly checks `CHECK-NOT: mgp.buffer.host_to_device` after each device-resident lookup. Both check that `bar` retains its alias flag. These are expected IR properties, not downloaded weights or observed transfers. [CPU-source expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/constant-external-to-primitives.mlir:3) [GPU-resident expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/constant-external-to-primitives.mlir:21)

At this boundary the lookup/copy operations are constructed in execute. The next stage may hoist eligible invariant work into init; `has_alias=true` is one reason that later hoisting is rejected. Attention's expected weight map is empty, so this example explains the purpose of the implicit weights-registry argument without adding weights to Q/K/V. [[entry:74]] [Hoisting alias guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MGPDialect/Transforms/ExecInvariantCodeMotionPass.cpp:201)

</div>
</details>

<details class="qna-cell" id="qna-mogg-mgp-example-subgraph-calls" markdown="1">
<summary>Example 7 · Why are callees lowered first, and how does a graph call forward model state and its result chain?</summary>
<div class="qna-answer" markdown="1">

**TEST · `call-graph.mlir` begins with two MO graphs.** Its pipeline imports kernels, lowers MO→MAP→MOGG and then applies `--mogg-to-mgp`. The subgraph contains ReLU; the caller contains `mo.call`. These are named graph calls, distinct from the already-outlined `mogg.call` used by an individual kernel helper. [Test input and RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/call-graph.mlir:1)

```transform
BEFORE · schematic graph pair
  @call_subgraph(X:f32[4]) isSubgraph              # Separate graph symbol.
    Y = relu(X)                                  # Elementwise max(X,0).
    mo.output Y                                  # One tensor result.
  @test_call(X:f32[4])
    Y = mo.call @call_subgraph(X)                 # Only explicit tensor operand.
    mo.output Y                                  # Forward subgraph result.

AFTER · schematic model pair
  mgp.model @call_subgraph subgraph                # Callee model already exists.
    init(...) → state tuple                      # Implicit init state.
    execute(chain,context,weights,cpu,X) → chain,Y # Implicit arguments included.
  mgp.model @test_call
    %c1,%Y = mgp.call @call_subgraph(%c0,%ctx,%weights,%cpu,%X)
                                                    # Forward state and caller tensor.
    mef.output %c1,%Y                            # Ambient chain becomes call result.

Illustrative X = [-2,0,3,-1] → Y = [0,0,3,0]        # Same ReLU before and after.
```

The pass scans each graph's `MO::CallOp` dependencies and postpones a caller until every callee name is in `completedGraphs`. This is necessary because `lower(MO::CallOp)` looks up an **MGP model** with the callee's name. A full worklist sweep without progress diagnoses `"cycle detected in graph ops"`; it does not recursively compile a cyclic call graph forever. [Dependency worklist](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2414) [Callee-model lookup](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2283)

The native lowerer prepends chain, model context and weights registry, then appends device contexts in the **callee's** device order and finally the explicit operands. It builds the result list with a leading chain. This code then rewires result users and advances the ambient chain. [Call operand/result assembly](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2288)

```cpp
for (unsigned i = 0; i < callOp.getNumResults(); ++i) {
  loweringState.emplaceExecuteLoweredValue(callOp.getResult(i),
                                           mgpCallOp.getResult(i + 1));
  loweringState.guardValueWithChain(mgpCallOp.getResult(i + 1), callOutChain);
}
loweringState.setCurrentExecuteChain(callOutChain);
```

The test's `CHECK-SAME` fixes the five execute arguments: chain/context/weights/CPU/X. Its `CHECK-NEXT` requires `mgp.call` with all five and then `mef.output` with both results from that call. This checks the offset `i+1` and ambient-chain behavior directly. [Expected call and terminator](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToPrimitives/call-graph.mlir:12)

**Rejected comparison:** a graph pair `A → B → A` cannot satisfy the callee-first worklist; the source diagnostic stops this conversion. **Actual attention:** only one graph exists, and `@F_attention` is a module helper rather than another `MO::GraphOp`. The graph dependency list is empty, and its chainless routine does not take this graph-call branch that advances `currentExecuteChain`. [Cycle branch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp:2430) [[entry:41]] [[mogg-ops:999]]

</div>
</details>
