## QnA 1 · What changes from MAP to MOGG for this call?

**The outer tensor interface is retained, while the routine body receives pointers, shapes and a context handle.** For this static example, its body has **five arguments: Q pointer, K pointer, V pointer, output pointer, device context**. There are no dynamic-dimension arguments. [[map-lowering:928]] [[map-lowering:942]] [[map-lowering:963]] [[map-lowering:996]]

<span id="qna-map-to-mogg-1-illustration-1"></span>

```state
BEFORE · map.iter.opaque
  outer operands: Q:T0, K:T0, V:T0
  outer result:   O:T0
  body arguments: Q:T0, K:T0, V:T0, output:T0

AFTER · mogg.routine
  outer operands: Q:T0, K:T0, V:T0
  outer result:   O:T0
  body arguments:
    0  %qptr : !mogg.pointer
    1  %kptr : !mogg.pointer
    2  %vptr : !mogg.pointer
    3  %optr : !mogg.pointer
    4  %ctx  : !mogg.context

T0 = float32[32,32] on gpu:0
```

`!mogg.pointer` is a compiler type for a future data-pointer argument. It is not an observed address. Likewise `%ctx` represents the future kernel device context; it is not the compiler's MLIR context or the shared MAXContext. Later MOGG→MGP supplies it from the model's device arguments. [[mogg-ops:825]] [[mogg-ops:894]]

## QnA 2 · What is the expanded MAP→MOGG call stack?

**SOURCE · the pass rewrites IR1, the compiler-owned clone.** `gcPipelineImpl` copies `opts.enableComputeLambdas` into the MAP→MOGG pass options; the default is `true`. Scheduling the pass only installs this recipe. Its `runOnOperation()` executes when the pass manager runs IR1. The option affects eligible output epilogues; the ordinary attention output has no such epilogue. [[gc:595]] [[gc:654]] [MAPToMOGG.td:28](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.td:28)

**Example links** map the earlier Q&A illustrations and the worked examples to their native checks, rewrites and handoffs. Links marked “later” describe a subsequent stage; a listed rewrite still requires its matching conditions.

```calltree
gcPipelineImpl → add MAPToMOGG(mapToMoggOpts) [[gc:595]] [[gc:597]]
└─ MAPToMOGGPass::runOnOperation() on IR1's module [[map-to-mogg:56]]
   ├─ MojoRegistry registry(module)                         // declaration lookup for this pass [[map-to-mogg:57]]
   ├─ walk → worklist of outer map.iter.opaque operations  // skip their nested regions for collection [[map-to-mogg:60]]
   │
   ├─ for each iterator: fresh MAPToMOGGState(registry,op)  // builder + SSA/dimension/context maps [[map-to-mogg:66]]
   │  ├─ copy enableComputeLambdas into this state [[map-to-mogg:67]]
   │  └─ state.lowerOp(op) [[map-state:21]]
   │     ├─ no LowerableToMOGGInterface → emit error → fail pass [[map-state:23]] [[map-to-mogg:68]]
   │     └─ IterOpaqueOp::lowerToMOGG(state) [[map-lowering:907]]
   │        ├─ read inputs, result types, input/output chain and full device ref [[map-lowering:910]]
   │        ├─ reuse fusion provenance; create a leaf only when absent [[map-lowering:921]]
   │        ├─ create mogg.routine with same outer operands/results [[map-lowering:928]]
   │        │  ├─ [↗ QnA 1 · boundary example — three tensor inputs, five local handles](#qna-map-to-mogg-1-illustration-1)
   │        │  ├─ [↗ QnA 7 · Example 1 — Q/K/V/O pointers and the required GPU context](#qna-map-to-mogg-example-attention-pointers)
   │        │  └─ copy read/write effects and dispatch; preserve gpu:0 [[map-lowering:930]]
   │        ├─ collect dynamic dimensions from attributes/types throughout body [[map-lowering:939]]
   │        │  └─ deduplicate by attribute in first-seen order; static SIMD attrs skipped [[map-lowering:41]]
   │        │     ├─ [↗ QnA 4 · dimension example — (32,32) adds zero dims; shared N adds one](#qna-map-to-mogg-4-illustration-1)
   │        │     └─ [↗ QnA 7 · Example 2 — collect dynamic and hidden view dimensions](#qna-map-to-mogg-example-dynamic-dimensions)
   │        ├─ create NEW body block and its arguments [[map-lowering:942]]
   │        │  ├─ tensor/buffer input → pointer; opaque input → its own type [[map-lowering:945]]
   │        │  ├─ ordinary tensor result → output pointer; chain/opaque result → none [[map-lowering:954]]
   │        │  ├─ one !mogg.dim argument per collected dynamic symbol [[map-lowering:963]]
   │        │  └─ context arguments: one, or unique-device count for DeviceContextArray [[map-lowering:975]]
   │        ├─ seed state.mapping with OLD body args → NEW representations [[map-lowering:1002]]
   │        │  ├─ opaque input → forward its new body argument directly [[map-lowering:1005]]
   │        │  ├─ tensor/buffer input → buildTensor(Input, mut = isBuffer) [[map-lowering:1011]]
   │        │  └─ tensor output → buildTensor(Output, mut = true) [[map-lowering:1018]]
   │        │     ├─ buildShape: constants in attr, dynamic dims as SSA operands [[map-lowering:79]]
   │        │     └─ TensorCreateOp: pointer, shape, dtype, device label, io, mut, stable [[map-lowering:102]]
   │        │        └─ [↗ QnA 3 · value-map example — reconstruct input/output views from pointers](#qna-map-to-mogg-3-illustration-1)
   │        ├─ lowerRegionBody in source order, excluding final map.yield [[map-lowering:1024]] [[map-state:28]]
   │        │  ├─ map.call.opaque → CallOpaqueOp::lowerToMOGG [[map-lowering:645]]
   │        │  │  ├─ lookup mapped input/output values and optional opaque result type [[map-lowering:646]]
   │        │  │  ├─ compute hasEpilogueFusion from output producers [[map-lowering:657]]
   │        │  │  ├─ declaration lookup → append contexts only when required [[map-lowering:665]]
   │        │  │  └─ create call with callee, parameters, flags; map its by-value result [[map-lowering:675]]
   │        │  │     ├─ [↗ QnA 5 · parameter example — preserve BN/BD; typed operands support later inference](#qna-map-to-mogg-5-metadata-example)
   │        │  │     └─ [↗ QnA 5 · later emitted call — output-first execute order; this pass prepares fields](#qna-map-to-mogg-5-illustration-1)
   │        │  ├─ stride view → tensor.create.slice/broadcast/reshape/transpose [[map-lowering:692]]
   │        │  │  ├─ TransposeStrideOp → TensorCreateTransposeOp: shape + permutation [[map-lowering:777]]
   │        │  │  │  └─ [↗ QnA 7 · Example 3 — transpose view and loop indices](#qna-map-to-mogg-example-transpose-view)
   │        │  │  ├─ BroadcastStrideOp → TensorCreateBroadcastOp: output shape [[map-lowering:722]]
   │        │  │  │  └─ [↗ QnA 7 · Example 4 — 64 input elements → 512 logical output elements](#qna-map-to-mogg-example-broadcast-view)
   │        │  │  └─ SliceStrideOp → TensorCreateSliceOp: offset + conditional static bound hints [[map-lowering:693]]
   │        │  │     └─ [↗ QnA 7 · Example 5 — slice bounds, offsets and hint guards](#qna-map-to-mogg-example-slice-offsets)
   │        │  ├─ elementwise body → call.foreach; remap captures then index [[map-lowering:793]]
   │        │  ├─ rowwise body/reduction → call.rowwise / rowwise.reduce [[map-lowering:841]]
   │        │  └─ epilogue → classify lambda; lower stores or yield computed value [[map-lowering:473]]
   │        │     └─ [↗ QnA 7 · Example 6 — lambda returns scalar or retains the store](#qna-map-to-mogg-example-compute-epilogue)
   │        ├─ by-value opaque yield → mogg.output; DPS output needs no value return [[map-lowering:1028]]
   │        │  └─ [↗ QnA 7 · Example 8 — opaque value crosses routine boundary by value](#qna-map-to-mogg-example-opaque-values)
   │        └─ replace outer SSA uses with routine results; erase old iterator/body [[map-lowering:1038]]
   │           └─ [↗ QnA 6 · handoff example — routine, reconstructed views and registered call](#qna-map-to-mogg-6-illustration-1)
   │
   └─ post-order walk of remaining mo.invoke_shape_func operations [[map-to-mogg:73]]
      └─ fresh state → lowerInvokeShapeFuncToMOGG [[map-lowering:126]]
         ├─ missing registered shape function → diagnostic → fail pass [[map-lowering:134]]
         ├─ create host routine returning !mogg.shape_func_result<rank> [[map-lowering:145]]
         ├─ input pointers, unique dims, context; reconstruct input views [[map-lowering:158]]
         ├─ params = output_rank + normalized kgenParams; preserve throwing flag [[map-lowering:177]]
         └─ call.shape → mogg.output; erase old invocation [[map-lowering:187]]
            └─ [↗ QnA 7 · Example 7 — host shape helper returns dimensions by value](#qna-map-to-mogg-example-shape-function)
```

The declaration was already selected during MO→MAP. Here the callee-symbol lookup supplies the signature needed for call construction. A fresh lowering state maps old body Values to their new representations. [[mo-to-map:705]] [[mo-to-map:241]] [[map-lowering:1002]]

| State owned by this lowering | Attention value | Why the lifetime matters |
| --- | --- | --- |
| pass-local `MojoRegistry` | P1's imported `FusedAttention` declaration | shared for symbol lookup while the pass handles all routines |
| per-iterator `IRMapping` | `%qi→%Q`, `%ki→%K`, `%vi→%V`, `%oi→%O` | body operands refer to the new IR objects before the old region is erased |
| per-iterator `dimArgs` | empty for four `[32,32]` tensors | a symbolic shape would add body arguments, without creating user tensor inputs |
| per-iterator `contextArgs` | one `%ctx` for `gpu:0` | the later runtime lowering supplies the actual device handle |
| surviving module IR | new routine, views and call | remains after the temporary C++ lowering state is destroyed |

The physical block-argument order is **inputs → output destinations → dimensions → contexts**. The custom printer can display its labeled `context (...)` group before `dims (...)`; that presentation does not reorder the underlying argument list. [MAPToMOGGState.h:31](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/Interfaces/MAPToMOGGState.h:31) [[map-lowering:942]] [infra-outline-routines.mlir:57](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir:57)

**TEST · forwarding-order cross-check.** That outlining test starts with MO custom calls, imports the test package, runs MO→MAP, accumulating-buffer setup and MAP→MOGG, then adds `--mogg-outline-routines`. Its dynamic case checks that the graph-local forwarding call and new helper both use pointer, pointer, dimension, context order. The native outlining code copies the body argument types in that order. [infra-outline-routines.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir:1) [[outline:49]] [[outline:124]]


## QnA 3 · Show the actual shape, tensor and value mappings.

The following is **source-derived schematic IR**, with shortened attribute spelling and package names. The pass initially calls `buildShape` for each tensor; a later cleanup can share identical shape operations. [[map-lowering:102]]

<span id="qna-map-to-mogg-3-illustration-1"></span>

```transform
%o = mogg.routine(%q:T0,%k:T0,%v:T0) -> T0
  body(%qptr,%kptr,%vptr,%optr : !mogg.pointer,
       %ctx : !mogg.context) {

    %sq = mogg.shape.create() {shape=[32,32]}
    %Q  = mogg.tensor.create(%qptr,%sq)
          {elementType=f32, device="gpu", io=Input, mut=false}

    %sk = mogg.shape.create() {shape=[32,32]}
    %K  = mogg.tensor.create(%kptr,%sk)
          {elementType=f32, device="gpu", io=Input, mut=false}

    %sv = mogg.shape.create() {shape=[32,32]}
    %V  = mogg.tensor.create(%vptr,%sv)
          {elementType=f32, device="gpu", io=Input, mut=false}

    %so = mogg.shape.create() {shape=[32,32]}
    %O  = mogg.tensor.create(%optr,%so)
          {elementType=f32, device="gpu", io=Output, mut=true}

    mogg.call.opaque @PKG.fused_attention.FusedAttention
      inputs(%Q,%K,%V) outputs(%O) context(%ctx)
      parameters={BD=8,BN=16}
  } device=gpu:0
```

| Old MAP body value | `state.mapping` target | Meaning |
| --- | --- | --- |
| `%qi` | `%Q` | read-only query view over `%qptr` |
| `%ki` | `%K` | read-only key view over `%kptr` |
| `%vi` | `%V` | read-only value view over `%vptr` |
| `%oi` | `%O` | writable output view over `%optr` |

The mapping is populated for inputs and destinations before lowering the nested call. The **full** `gpu:0` ref remains on the routine; the reconstructed tensor operations record the kind label `"gpu"`. [[map-lowering:916]] [[map-lowering:1011]] [[map-lowering:1018]]

**TEST · inspected expectation:** the plain-opaque conversion test checks pointer arguments, two shape constructions, input `mut=false`, output `mut=true`, and the trailing context. The diagram above extends that one-input test shape to this three-input attention signature using the same implementation. [[test-map-mogg:12]]

## QnA 4 · Why no dimension arguments for [32,32]? What if N were dynamic?

The collector treats a concrete `KGEN::SIMDAttr` dimension as static. All eight dimension entries across Q/K/V/O are the constant 32, so none needs a runtime dimension argument. This scalar dimension attribute is separate from the tensors' float32 element type. [[map-lowering:41]] [[map-lowering:48]]

<span id="qna-map-to-mogg-4-illustration-1"></span>

```state
This example: Q,K,V,O each [32,32]
  unique dynamic dimensions = {}
  body argument count = 4 pointers + 0 dims + 1 context = 5

Comparison: Q,K,V,O each [N,32], using the SAME symbolic N
  unique dynamic dimensions = {N}
  body argument count = 4 pointers + 1 dim + 1 context = 6
  additional argument: %n : !mogg.dim<N>
  shape construction: mogg.shape.create(%n) {shape=[dyn,32]}
```

The dynamic shape attribute uses a common `dyn` spelling; the operand `%n` retains the actual dimension value. If N later equals 32, the view describes 1,024 elements, but its dimension path is still different from a compile-time constant `[32,32]`. Distinct symbolic names are not merged solely because they may happen to have equal runtime values. [[map-lowering:76]] [[map-lowering:87]] [[map-lowering:963]]

## QnA 5 · Where do BN, BD, dtype, rank and target get their values?

**The phases are different.** MAP→MOGG preserves the explicit parameter dictionary and prepares typed call operands; it does not turn BN/BD into tensor inputs. [[map-lowering:675]] [[params:63]]

<span id="qna-map-to-mogg-5-metadata-example"></span>

| Item | Value | Where it comes from / when consumed |
| --- | --- | --- |
| BN | `16` | Python caller → custom dict → MAP/MOGG call attribute |
| BD | `8` | Python caller → custom dict → MAP/MOGG call attribute |
| dtype | `float32` | tensor element type; inferred by Mojo from the call argument types |
| rank | `2` | tensor shape rank; declared before `//` in `execute` |
| target | `"gpu"` | later emitter reads the outlined function's device label |
| ctx | future GPU context handle | routine context argument → later runtime device argument |

Sources: example arguments, infer-only parameter handling, emitter default inference and context wiring. [[entry:38]] [[kernel:74]] [[params:328]] [[emit-state:582]] [[mogg-ops:831]]

The eventual implementation call can be understood as this **abbreviated Mojo call**, with module qualification and setup omitted:

<span id="qna-map-to-mogg-5-illustration-1"></span>

```mojo
FusedAttention.execute[BN=16, BD=8, target="gpu"](
    output, query, key, value, ctx
)
```

Notice the implementation's **output-first** order. `mogg.call.opaque` keeps separate input/output lists, and the emitter puts outputs first when it writes Mojo. Dtype and rank are inferred; explicitly supplying those infer-only parameters is unnecessary. [[mogg-ops:1892]] [[mogg-ops:1918]] [[kernel:77]]

## QnA 6 · What is ready for the MOGG refinement stage?

<span id="qna-map-to-mogg-6-illustration-1"></span>

```state
IR1
├─ mo.graph @fused_attention
│  └─ mogg.routine : (T0,T0,T0) → T0
│     ├─ body: 4 pointer arguments + 1 context argument
│     ├─ typed Q/K/V/O view construction
│     └─ mogg.call.opaque to P1 declaration, BN=16, BD=8
└─ kernel declarations retained for later conformance and emission

G1 Python members: retained
Runtime output allocation: still to be planned
Next: conform call operands → outline this body → trace/cleanup
```

These are code-generation operations, not PTX or machine instructions. The next stage gives the body a callable symbol while retaining the routine's tensor boundary for MOGG→MGP. [[gc:370]] [[outline:14]] [[gc:618]]


## QnA 7 · How do concrete MAP bodies become MOGG calls, views and by-value results while preserving their computation?

**SOURCE · eight worked transformations.** The before/after IR below is schematic: names and type spellings are shortened, and each pointer names a future buffer reference. Numerical values and byte offsets are illustrative arithmetic, not captured execution. **TEST** paragraphs describe inspected repository inputs and expectations. A test's `RUN` command parses that file and invokes passes; production lowering does not call the `.mlir` file. `CHECK` matches required output, `CHECK-NEXT` also requires adjacency, and `CHECK-NOT` excludes a match within its check interval.

<details class="qna-cell" id="qna-map-to-mogg-example-attention-pointers" markdown="1">
<summary>Example 1 · How does the fused-attention call acquire four pointers and a device context without changing Q, K, V or O?</summary>
<div class="qna-answer" markdown="1">

Take the actual registration with Q/K/V/O all `float32[32,32]`, on `gpu:0`, and `BN=16, BD=8`. For an illustrative input, set every Q and K element to zero and `V[j,d]=j`. Every attention score is zero, so each row's 32 softmax weights are `1/32`; every output element is `(0+1+...+31)/32 = 15.5`. This is the same unscaled attention expression before and after lowering; these arrays are not supplied to this compiler pass. [[entry:38]] [[entry:57]] [[kernel:82]]

```transform
BEFORE · schematic MAP body inside IR1
%o = map.iter.opaque inputs(%qi=%q,%ki=%k,%vi=%v)  # Bind three tensor inputs.
                    outputs(%oi:T0) {             # Describe one tensor destination.
  map.call.opaque @P1.FusedAttention               # Preserve the selected implementation.
    inputs(%qi,%ki,%vi) outputs(%oi)                # Supply the same Q/K/V/O roles.
    params={BN=16,BD=8}                            # Keep compile-time tile parameters.
} device=gpu:0                                    # Retain the concrete graph device.
mo.output %o                                      # Return O at the graph boundary.

T0 = float32[32,32]                                # 1,024 f32 elements per tensor.
```

`IterOpaqueOp::lowerToMOGG` creates the outer replacement with the old result types, inputs and input chain, then copies memory effects and dispatch. Thus a result-producing kernel and a side-effect-only kernel keep their respective graph contracts. The following is an unchanged implementation excerpt. [MAPOpsLowerableToMOGG.cpp:928](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:928)

```cpp
auto kernel = MOGG::RoutineOp::create(builder, loc, resultTypes, inputs,
                                      inputChain, fusionInfo);
kernel.setDeviceRef(state.device);
```

Inside that new routine, four tensor-like positions produce four `!mogg.pointer` arguments. `buildTensor` then records shape, dtype, input/output role and mutability around each pointer. There is no dynamic symbol to add between those pointers and `%ctx`. [MAPOpsLowerableToMOGG.cpp:942](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:942) [buildTensor:102](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:102)

```transform
AFTER · schematic MOGG body, before CSE
%o = mogg.routine(%q,%k,%v) -> T0 {                 # Keep the same tensor-level I/O.
  body(%qp,%kp,%vp,%op: pointer, %ctx: context)     # Five fresh body arguments.
  %sq = shape.create([32,32])                      # Describe Q's two static extents.
  %Q = tensor.create(%qp,%sq,f32,Input,false)       # Read-only Q view.
  %sk = shape.create([32,32])                      # Describe K independently for now.
  %K = tensor.create(%kp,%sk,f32,Input,false)       # Read-only K view.
  %sv = shape.create([32,32])                      # Describe V independently for now.
  %V = tensor.create(%vp,%sv,f32,Input,false)       # Read-only V view.
  %so = shape.create([32,32])                      # Describe O's output extent.
  %O = tensor.create(%op,%so,f32,Output,true)       # Writable O destination view.
  mogg.call.opaque @P1.FusedAttention              # Call the same registration.
    inputs(%Q,%K,%V) outputs(%O) context(%ctx)      # Forward four views and context.
    params={BN=16,BD=8}                            # Keep BN/BD as call attributes.
} device=gpu:0                                    # Full device reference stays here.
mo.output %o                                      # Same graph result, future values 15.5.
```

The new body owns its shape and view operations; `state.mapping` rewires the old `%qi/%ki/%vi/%oi` body values to `%Q/%K/%V/%O`. After the nested call has been lowered, the pass redirects users of the old iterator results and erases the old iterator, including its region. For an illustrative row-major f32 layout, Q's element `[2,3]` remains at byte offset `(2*32+3)*4 = 268` from Q's eventual data pointer. The type describes `4096` payload bytes per tensor; this pass does not allocate those bytes. [MAPOpsLowerableToMOGG.cpp:1002](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:1002) [MAPOpsLowerableToMOGG.cpp:1038](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:1038)

**The context guard is signature-driven.** This excerpt forwards contexts for a declarative launch or an `execute` signature ending in a context argument. [CallOpaqueOp::lowerToMOGG:668](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:668)

```cpp
if (const auto &fn = adaptor.executeFunction;
    adaptor.isDeclarative() ||
    (fn && !fn->inputArguments.empty() &&
     (fn->inputArguments.back().isContextType() ||
      fn->inputArguments.back().isVariadicContextType())))
  contexts = state.contextArgs;
```

| Signature branch | Routine body | Implementation call |
| --- | --- | --- |
| attention declares `DeviceContext` | four pointers + one context | `%ctx` is forwarded |
| host-only comparison declares no context | its body still has the ordinary trailing context argument | call omits `context(...)` |
| distributed comparison declares `DeviceContextArray` | one context per unique device, at least one | contexts remain separate here; the next conformance pass packs them |

**TEST · plain tensor contract.** [plain-opaque.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/plain-opaque.mlir:1) enters at MO, imports the testing package, runs `--mo-to-map`, then `--map-to-mogg`. Its one-input `[2,2]` case checks input/output pointers, two shape constructions, `mut=false/true`, and the context passed to `MapLowerOnePlainOutput`; its two-output case checks two destination pointers. `CHECK-NOT: hasEpilogueFusion` checks the printer's omitted false default. Those are structural fixtures: their registered bodies are empty, so they establish no output arithmetic. [plain-opaque.mlir:14](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/plain-opaque.mlir:14) [plain-opaque.mlir:31](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/plain-opaque.mlir:31) [testing_kernels.mojo:3566](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3566)

**TEST · matched and skipped context forwarding.** [no-context-opaque.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/no-context-opaque.mlir:1) uses the same three-pass route. `@no_context` checks that the call has no `context(` suffix, while `@with_context` checks the exact body context forwarded to the call. The test deliberately distinguishes call spelling `context(` from the routine header's `context (`. [no-context-opaque.mlir:4](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/no-context-opaque.mlir:4)

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-dynamic-dimensions" markdown="1">
<summary>Example 2 · How does a shared dynamic dimension become one body argument, including dimensions hidden inside a view?</summary>
<div class="qna-answer" markdown="1">

Use the test's shape `[batch,4]` for both input and output. If `batch=3`, each tensor describes 12 f32 elements; if `batch=5`, it describes 20. These are alternative runtime extents of the same compiled shape contract. The example's fixture leaves tensor computation unspecified; the transformation preserves its input, output and selected call. [dyn-opaque.mlir:22](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/dyn-opaque.mlir:22)

```transform
BEFORE · schematic MAP
%y = map.iter.opaque inputs(%xi=%x:[batch,4])       # Bind one dynamic tensor.
                    outputs(%yo:[batch,4]) {       # Reuse the same symbolic batch.
  map.call.opaque @OnePlain inputs(%xi) outputs(%yo) # Keep the declared kernel call.
}

COLLECTION
[batch,4] input  → insert batch                   # First occurrence creates one map entry.
[batch,4] output → reuse batch                    # Attribute identity deduplicates it.
4               → skip                          # Concrete dimension needs no SSA argument.
```

The collector inserts each non-static dimension into a `MapVector`. Later, `buildShape` stores a common `dyn` placeholder in the shape attribute but connects the actual dimension through an SSA operand. These are implementation excerpts, showing the collection and reconstruction at their respective call sites. [dynamicDimWalker:48](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:48) [buildShape:87](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:87)

```cpp
for (Attribute dim : shape.getValues())
  if (isDynamicDim(dim))
    state.dimArgs.insert({dim, BlockArgument()});
```

```cpp
if (isDynamicDim(dim)) {
  dims.push_back(state.dimArgs.lookup(dim));
  canonicalDims.push_back(canonicalDyn);
} else {
  canonicalDims.push_back(dim);
}
```

```transform
AFTER · schematic MOGG
%y = mogg.routine(%x:[batch,4]) -> [batch,4] {      # Retain tensor-level types.
  body(%xp:pointer,%yp:pointer,%b:dim<batch>,%ctx)  # Two pointers + one dim + context.
  %sx = shape.create(%b) {shape=[dyn,4]}           # Use batch for the input extent.
  %X = tensor.create(%xp,%sx,f32,Input,false)      # Same future input storage.
  %sy = shape.create(%b) {shape=[dyn,4]}           # Same dimension SSA operand.
  %Y = tensor.create(%yp,%sy,f32,Output,true)      # Separate output destination.
  call.opaque @OnePlain inputs(%X) outputs(%Y) context(%ctx) # Preserve the call.
}
```

CSE may later share `%sx` and `%sy` because both their attributes **and operands** match. For input `[M,4]` and output `[N,4]`, the two shape attributes can both print `[dyn,4]`, but `%m` and `%n` are distinct operands. Runtime coincidence `M=N=3` is not enough for this pass to merge them. [MAPOpsLowerableToMOGG.cpp:76](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:76)

**TEST · shared symbol.** [dyn-opaque.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/dyn-opaque.mlir:1) starts with MO and runs package import → MO→MAP → MAP→MOGG. Its checks name one `!mogg.dim<batch>` and require that same captured SSA name in both `mogg.shape.create` operations. They check dimension wiring, not a measured choice of batch size. [dyn-opaque.mlir:13](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/dyn-opaque.mlir:13)

**A dimension can also occur only in a nested view.** The collector walks nested operation attributes and operand/result types, so it includes the axis reduced away from a kernel's output. [collectDynamicDims:65](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:65) [IterOpaqueOp::lowerToMOGG:937](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:937)

```transform
Comparison before lowering
%x : [D0,D1,D2,D3]                               # Four input dimensions.
%view = transpose(%x,%runtimePerm)                # View shape [D8,D9,D10,D11].
%out = reduce_add(%view,axis=3)                   # Output [D8,D9,D10,1].

After collecting the entire body
%viewShape = shape.create(%d8,%d9,%d10,%d11)       # D11 survives as a dimension value.
call.rowwise shape(%viewShape) axis(3)             # Reduction still knows its row length.

Illustrative permutation [2,0,3,1]
input [2,3,4,5] → view [4,2,5,3] → output [4,2,5,1] # Reduced extent remains 3.
```

**TEST · internal reduced axis.** [transpose-reduce-dynamic-dim.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-reduce-dynamic-dim.mlir:1) starts at MO, runs shape inference and the full graph pipeline with constant folding disabled, and checks the `mogg-pre-mgp` dump. Its check requires all four `%d8…%d11` shape operands feeding the rowwise call after tracing. This regression checks compile-side shape plumbing; the file explicitly records a separate unresolved strided-view/prologue value issue, so it supplies no runtime numerical proof. [transpose-reduce-dynamic-dim.mlir:6](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-reduce-dynamic-dim.mlir:6) [transpose-reduce-dynamic-dim.mlir:20](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-reduce-dynamic-dim.mlir:20)

The actual attention call has only static `[32,32]` types and no graph-visible transpose/reduction body; its dimension map remains empty. [[entry:57]] [[map-lowering:939]]

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-transpose-view" markdown="1">
<summary>Example 3 · How does a transposed MAP view become a MOGG view and a loop that writes the same matrix?</summary>
<div class="qna-answer" markdown="1">

Choose the test shape `X:[2,3]`, permutation `[1,0]`, and illustrative values `X=[[1,2,3],[4,5,6]]`. The required result is `[[1,4],[2,5],[3,6]]`, shape `[3,2]`. MAP view motion has already hoisted the transpose view out of the elementwise body before this conversion runs. [transpose-stride.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-stride.mlir:1)

```transform
BEFORE · schematic MAP after view motion
%tv = map.index.transpose.stride(%X,%perm)        # View shape [3,2], permutation [1,0].
map.iter.elementwise.body shape=[3,2] captures(%O,%tv) { # Iterate the output domain.
  %v = map.load %tv[%i]                          # Read X at swapped coordinates.
  map.store %O[%i] = %v                         # Write one output element.
}
```

`TransposeStrideOp::lowerToMOGG` reads the view signature, element type and known permutation, then creates the replacement using the already mapped input and permutation operands. It maps the old view result to the new result. [MAPOpsLowerableToMOGG.cpp:777](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:777)

```cpp
auto createTranspose = MOGG::TensorCreateTransposeOp::create(
    state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
    getSpec().getPerm(), TypeAttr::get(elementType),
    state.builder.getBoolAttr(viewsMutableStorage(getInput())),
    state.mapping.lookup(getInput()),
    state.mapping.lookup(getPermutations()));
state.mapping.map(getResult(), createTranspose.getResult());
```

```transform
AFTER · schematic MOGG, enclosing pointer reconstruction omitted
%TV = mogg.tensor.create.transpose(%X,%PERM)      # New view over the same input reference.
      outShape=[3,2], staticPermutations=[1,0]    # Preserve view shape and known mapping.
%loop = mogg.shape.create([3,2])                  # Extents come from the loop operation.
mogg.call.foreach shape(%loop) captures(%O,%TV) { # Capture output and hoisted view.
  %v = mogg.tensor.load %tvArg[%i]               # Same logical element as the MAP load.
  mogg.tensor.store %oArg[%i] = %v               # Same output coordinate and value.
}
```

The foreach lowering creates a new nested block with one argument per capture followed by the index argument. `state.mapping` connects the old capture/index arguments to these new arguments before it visits the load and store. `LoadOp` produces a `!mogg.simd<f32>` value; `StoreOp` consumes that mapped value. [IterElementwiseBodyOp::lowerToMOGG:824](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:824) [LoadOp::lowerToMOGG:223](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:223) [StoreOp::lowerToMOGG:232](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:232)

| Illustrative coordinate | Input read | Output written |
| --- | --- | --- |
| `%i=[0,1]` | `X[1,0]=4` | `O[0,1]=4` |
| `%i=[1,0]` | `X[0,1]=2`, source byte offset `4` | `O[1,0]=2`, output byte offset `8` |
| `%i=[2,1]` | `X[1,2]=6` | `O[2,1]=6` |

These offsets assume dense row-major f32 buffers with independent symbolic base pointers. The view operation describes indexing. This standalone transpose's foreach still has to write its output later; the compiler conversion itself does not move the six elements.

**TEST · pass entry and checks.** [transpose-stride.mlir:36](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-stride.mlir:36) supplies an MO transpose and a constant permutation. Its `RUN` pipeline performs import → MO→MAP → structured-body outlining → loop-invariant view motion → iterator canonicalization → MAP→MOGG. `CHECK-NEXT` verifies pointer reconstruction, the transpose view, the `[3,2]` foreach domain and the exact load/store sequence. The earlier canonicalizer removes dead original captures after view motion; MAP→MOGG lowers the surviving capture list. The second case checks permutation `[2,0,1]` from `[2,3,4]` to `[4,2,3]`. [transpose-stride.mlir:18](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-stride.mlir:18) [transpose-stride.mlir:48](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/transpose-stride.mlir:48)

Attention's internal `QKᵀ` remains inside its authored implementation. This test's graph-visible `mo.transpose` does not describe a separate transpose introduced around the attention custom call. [[kernel:91]] [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-broadcast-view" markdown="1">
<summary>Example 4 · How can a broadcast loop cover 512 output elements while reading only a 64-element input view?</summary>
<div class="qna-answer" markdown="1">

The test broadcasts `X:[1,64]` to `O:[8,64]`. Choose illustrative values `X[0,c]=c`. Every output row must contain `0…63`; for example, `O[7,5]=5`. The broadcast view and the loop domain both describe `[8,64]`, while the original input still describes only 64 elements. [broadcast-stride.mlir:30](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/broadcast-stride.mlir:30)

```transform
BEFORE · schematic MAP after view motion
%bv = map.index.broadcast.stride(%X) out=[8,64]   # Broadcast the size-one first axis.
map.iter.elementwise.body shape=[8,64] captures(%O,%bv) { # Use the output domain.
  %v = map.load %bv[%row,%col]                    # Read X[0,col] for each row.
  map.store %O[%row,%col] = %v                    # Materialize the requested output.
}
```

The native rewrite records `signature.getOut()` on the view and keeps the input reference. It does not attach a second copy of the input shape to that view. [BroadcastStrideOp::lowerToMOGG:721](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:721)

```cpp
auto createBroadcast = MOGG::TensorCreateBroadcastOp::create(
    state.builder, getLoc(), MOGG::TensorType::get(ctx), signature.getOut(),
    TypeAttr::get(elementType),
    state.builder.getBoolAttr(viewsMutableStorage(getInput())),
    state.mapping.lookup(getInput()));
state.mapping.map(getResult(), createBroadcast.getResult());
```

`IterElementwiseBodyOp` explicitly obtains the iteration shape from the MAP loop's `$shape`, because a broadcast input can have smaller extents. [MAPOpsLowerableToMOGG.cpp:793](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:793)

```transform
AFTER · schematic MOGG
%BV = mogg.tensor.create.broadcast(%X) outShape=[8,64] # Wrap the existing input view.
%loop = mogg.shape.create([8,64])                 # Preserve 8×64 iteration extents.
mogg.call.foreach shape(%loop) captures(%O,%BV) { # Forward two view references.
  %v = mogg.tensor.load %broadcastArg[%i]         # Repeated rows use the same input row.
  mogg.tensor.store %outputArg[%i] = %v           # Store 512 logical output elements.
}
```

For illustrative row-major f32 addressing, all eight output positions `[r,5]` read input byte offset `5*4=20`. Output `[7,5]` is stored at `(7*64+5)*4=1812` from the output base. No 512-element intermediate input buffer is created by `tensor.create.broadcast`; the enclosing routine still exposes its separate 512-element output contract.

**TEST · two broadcast forms.** [broadcast-stride.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/broadcast-stride.mlir:1) enters at MO and uses the same import, MO→MAP, outlining, view-motion, iterator-cleanup and MAP→MOGG sequence as the transpose test. `@broadcastSameRank` requires a `[1,64]` input view, `[8,64]` output view, broadcast view, and a foreach capturing only output plus broadcast. `@broadcastRankExpanding` starts with `[64]` and checks the same output view shape; the input's own rank remains available through its producer. [broadcast-stride.mlir:14](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/broadcast-stride.mlir:14) [broadcast-stride.mlir:42](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/broadcast-stride.mlir:42)

The matcher here is the already-existing `map.index.broadcast.stride` operation's lowering interface. This stage does not derive a broadcast from the attention name or its `BN`/`BD` values. [[map-state:21]] [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-slice-offsets" markdown="1">
<summary>Example 5 · How are slice offsets preserved, and why does a negative start on a dynamic axis lose its static hint?</summary>
<div class="qna-answer" markdown="1">

Use `X:[10,64]`, `start=[1,32]`, `stop=[4,64]`, `step=[1,1]`; the output shape is `[3,32]`. With illustrative values `X[r,c]=100*r+c`, output `[2,5]` is `X[3,37]=337`. [slice-stride.mlir:37](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/slice-stride.mlir:37)

```transform
BEFORE · schematic MAP after view motion
%sv = map.index.slice.stride(%X,%start,%stop,%step) # Preserve runtime bounds as operands.
      in=[10,64], out=[3,32], starts=[1,32], steps=[1,1] # Retain known static hints.
map.iter.elementwise.body shape=[3,32] captures(%O,%sv) { # Loop over slice coordinates.
  %v = map.load %sv[%i]                          # Read X[i0+1,i1+32].
  map.store %O[%i] = %v                         # Write the same element to O.
}
```

The native implementation first copies the static-start hints. It replaces a hint only when that start is negative **and** its corresponding input extent is dynamic. [SliceStrideOp::lowerToMOGG:699](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:699)

```cpp
SmallVector<int64_t> inDims = signature.getIn().getIntDims();
SmallVector<int64_t> staticStart =
    llvm::to_vector(getSpec().getStarts().asArrayRef());
for (size_t i = 0; i < staticStart.size(); ++i)
  if (staticStart[i] < 0 && ShapedType::isDynamic(inDims[i]))
    staticStart[i] = ShapedType::kDynamic;
```

It then creates `mogg.tensor.create.slice` with the output shape, static hints, mapped input and all three bounds tensors. Thus removing a static hint does not remove the runtime start value. [MAPOpsLowerableToMOGG.cpp:709](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:709)

```transform
AFTER · static example, schematic MOGG
%SV = mogg.tensor.create.slice(%X,%START,%STOP,%STEP) # Keep the bounds references.
      outShape=[3,32], staticStart=[1,32], staticStep=[1,1] # Both starts stay static.
%loop = mogg.shape.create([3,32])                 # Use the slice output extents.
mogg.call.foreach shape(%loop) captures(%O,%SV) { # Use the hoisted slice view.
  %v = mogg.tensor.load %sliceArg[%i]             # Same X[i0+1,i1+32] access.
  mogg.tensor.store %outArg[%i] = %v              # Preserve the output coordinate.
}
```

Illustrative f32 addressing for `%i=[2,5]` is source offset `(3*64+37)*4=916` bytes and output offset `(2*32+5)*4=276` bytes. The output remains `337` at that logical position.

| Input-axis extent and start | Static hint after this guard | Runtime interpretation |
| --- | --- | --- |
| known `10`, normalized start `1` | `1` | offset is fixed |
| symbolic `N`, start `-8` | dynamic sentinel | needs the actual N and the runtime start operand |
| symbolic `N`, start `2` | `2` | already an absolute start; no N needed |

```transform
Dynamic negative-start comparison, step=1 and runtime stop=N
X:[N,64], start=[-8,0]                            # Request the last eight rows.
  → slice view staticStart=[dynamic,0]            # Defer the extent-dependent offset.
N=20 → resolved first row 12 → output [8,64]      # 20-8 = 12.
N=32 → resolved first row 24 → output [8,64]      # 32-8 = 24.
```

**TEST · static and dynamic guards.** [slice-stride.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/slice-stride.mlir:1) starts at MO and runs import → MO→MAP → structured outlining → view motion → iterator canonicalization → MAP→MOGG. Its static checks require the `[3,32]` view and load/store loop; its runtime-bounds case checks that the view still receives four operands. The negative-start case explicitly expects the printed sentinel `-9223372036854775808`, while the positive-start dynamic case expects `2`. These expectations directly exercise the condition above. [slice-stride.mlir:15](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/slice-stride.mlir:15) [slice-stride.mlir:50](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/slice-stride.mlir:50) [slice-stride.mlir:72](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/slice-stride.mlir:72) [slice-stride.mlir:91](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/slice-stride.mlir:91)

This view-lowering comparison does not add a graph-level slice around fused attention. Its actual Q/K/V operands continue directly to the ordinary opaque call. [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-compute-epilogue" markdown="1">
<summary>Example 6 · When can an output epilogue yield a value for the kernel to store, and when must it retain its own store?</summary>
<div class="qna-answer" markdown="1">

Use the test's visible graph `O = matmul(A,B) + Z`, with all tensors `f32[8,8]` on `gpu:0`. For illustrative values `A=I`, `B=2I`, `Z=all_ones`, the required O has diagonal `3` and off-diagonal `1`. The earlier MAP fusion pass has placed the add in the matmul's output epilogue; MAP→MOGG chooses how to represent that already-fused body. [compute-fused-opaque.mlir:23](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/compute-fused-opaque.mlir:23)

```transform
BEFORE · schematic MAP epilogue after fusion and accumulating-buffer setup
%epi = map.iter.epilogue(%value:f32,%i) captures(%O,%Z) { # Kernel supplies one result value.
  %z = map.load %Z[%i]                          # Read the residual at that index.
  %sum = map.call.elementwise @Add(%value,%z)    # Compute matmul value + residual.
  map.store %O[%i] = %sum                       # Store once at the unchanged index.
}
map.call.opaque @Matmul inputs(%A,%B) outputs(%epi) # Output declaration permits compute fusion.
```

`classifyEpilogueLambdaKind` uses the consuming call's actual declaration. It requires a single opaque-call output use and an eligible compute-output slot. Multiple stores or a transformed store index force the store form; unsupported tile-output slots report an error. The final checks require the option to be enabled and equal incoming/outgoing element types. [MAPOpsLowerableToMOGG.cpp:345](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:345) [MAPOpsLowerableToMOGG.cpp:380](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:380)

```cpp
if (!state.enableComputeLambdas)
  return Kind::OutputLegacyForCompute;
Type inType = epilogueOp.getBody().getArgument(0).getType();
Type outType = cast<MO::TensorType>(result.getType()).getElementType();
if (inType != outType)
  return Kind::OutputLegacyForCompute;

return Kind::Compute;
```

In the matched case, `IterEpilogueOp::lowerToMOGG` lowers the arithmetic but skips the store to the designated output capture. It remembers that store's mapped value and creates `mogg.yield` with it. The output capture and output pointer remain so the emitter can bind the functor to the destination. [MAPOpsLowerableToMOGG.cpp:521](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:521) [MAPOpsLowerableToMOGG.cpp:553](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:553)

```transform
AFTER · enabled, eligible compute epilogue
%epi = mogg.tensor.create.epilogue(%value:simd<f32>,%i) # Preserve value and index inputs.
       captures(%O,%Z), computeOutputArg=true {       # Retain destination metadata.
  %z = mogg.tensor.load %Z[%i]                      # Same residual read.
  %sum = mogg.call.elementwise @Add(%value,%z)       # Same addition.
  mogg.yield %sum                                   # Kernel performs the final store.
}
call.opaque @Matmul inputs(%A,%B) outputs(%epi)       # Same matmul implementation.

AFTER · enable-compute-lambdas=false
%epi = mogg.tensor.create.epilogue(%value:simd<f32>,%i) # Same output contract.
       captures(%O,%Z), computeOutputArg=true {       # Slot still declares compute capability.
  %z = mogg.tensor.load %Z[%i]                      # Same residual read.
  %sum = mogg.call.elementwise @Add(%value,%z)       # Same addition.
  mogg.tensor.store %O[%i] = %sum                   # Functor owns the store instead.
}
```

At an illustrative diagonal element both routes calculate `2+1=3`; elsewhere both calculate `0+1=1`. `computeOutputArg=true` records the kernel argument's contract even when the body uses the store fallback. It does not by itself prove that the body ends in a yield. [MAPOpsLowerableToMOGG.cpp:508](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:508)

| Guard | Matched case | Skipped or rejected case |
| --- | --- | --- |
| eligible consumer contract | compute-output slot | ordinary store-output slot keeps store form |
| output stores | one destination | multiple destinations keep stores |
| index mapping | store at incoming index | reshape/index transform keeps store ownership |
| dtype | incoming f32 → outgoing f32 | incoming f32 → outgoing si32 keeps cast and store |
| option | enabled | disabled uses store fallback |
| tile fused-output contract | SIMD case supported here | tile case reports unsupported MAP epilogue fusion |

**TEST · two commands for the same graph.** [compute-fused-opaque.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/compute-fused-opaque.mlir:1) starts with MO, imports kernels, lowers to MAP, fuses kernels, allocates accumulating buffers, then runs MAP→MOGG. The first command checks `load → Add → yield`; the second sets `enable-compute-lambdas=false` and the `NOCOMPUTE` checks require `load → Add → store`. Its `@mixed_dtype` case checks the retained store after an f32-to-si32 cast. Its `@plain_store` case checks the store-only declaration's omitted false `computeOutputArg` default. [compute-fused-opaque.mlir:9](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/compute-fused-opaque.mlir:9) [compute-fused-opaque.mlir:18](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/compute-fused-opaque.mlir:18) [compute-fused-opaque.mlir:38](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/compute-fused-opaque.mlir:38) [compute-fused-opaque.mlir:56](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/compute-fused-opaque.mlir:56)

The actual attention registration declares an ordinary `OutputTensor`. Its implementation's name includes “fused”, but that name does not satisfy this compute-output contract or create a MAP epilogue. [[kernel:82]]

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-shape-function" markdown="1">
<summary>Example 7 · How does a shape-function invocation become a host routine that returns dimensions by value?</summary>
<div class="qna-answer" markdown="1">

The repository test supplies a shape-only gather invocation with input shape `[1,2,3]`, indices shape `[4,5]`, `axis=-2`, and four declared output dimensions. For the ordinary gather shape rule, axis `-2` in rank 3 means axis 1; replacing that input axis with the indices shape gives `[1,4,5,3]`. The test then broadcasts scalar integer `1` into that shape, describing 60 ones. It does not perform a data gather; its indices tensor's f32 element type is used only as a source of shape metadata in this fixture. [shape-func-kgen-params.mlir:11](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/shape-func-kgen-params.mlir:11) [gather_shape_fn:2096](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/kernels/src/graph_compiler/builtin_kernels/gather_scatter.mojo:2096) [gather_shape:1579](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/kernels/src/nn/gather_scatter.mojo:1579)

```transform
BEFORE · schematic MO operation remaining among the MAP work
mo.invoke_shape_func "mo.gather"(%input:[1,2,3],%indices:[4,5]) # Ask for four extents.
  outputParams=[D4,D5,D6,D7], kgenParams={axis=-2}              # Preserve axis as a parameter.
%one = constant(1) : scalar<si32>                            # Test's data value is one.
%out = broadcast(%one) : [D4,D5,D6,D7]                       # Use the returned shape.
```

The MAP→MOGG pass handles these invocations in a second walk after its opaque-iterator worklist. `lowerInvokeShapeFuncToMOGG` looks up the registered shape function, creates a routine with result type `!mogg.shape_func_result<4>` and copies the output parameter declarations. Missing registration produces a diagnostic; the pass cannot invent the shape computation. [MAPToMOGG.cpp:73](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp:73) [MAPOpsLowerableToMOGG.cpp:134](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:134)

The call's parameter dictionary is built from the output rank **and** normalized original `kgenParams`. Forwarding only rank would lose the axis argument required by the implementation. [MAPOpsLowerableToMOGG.cpp:179](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:179)

```cpp
SmallVector<NamedAttribute> shapeParams{
    builder.getNamedAttr(MOGG::kMoggOutputRankParameterName,
                         builder.getIndexAttr(static_cast<int64_t>(rank)))};
if (DictionaryAttr kgenParams = shapeFuncOp.getKgenParamsAttr())
  for (NamedAttribute param : kgenParams)
    shapeParams.push_back(builder.getNamedAttr(
        param.getName(), normalizeKernelParameter(param.getValue())));
```

```transform
AFTER · schematic shape routine
%shapeResult = mogg.routine(%input,%indices) -> shape_func_result<4> # Return a small shape value.
  outputParams=[D4,D5,D6,D7] {                                  # Retain the dimension bindings.
  body(%inputPtr:pointer,%indicesPtr:pointer,%ctx:context)       # No output data pointer.
  %si = shape.create([1,2,3])                                  # Input extents are static here.
  %I = tensor.create(%inputPtr,%si,f32,Input,false)               # Rebuild the input view.
  %sj = shape.create([4,5])                                    # Indices' own two extents.
  %J = tensor.create(%indicesPtr,%sj,f32,Input,false)             # Rebuild its view.
  %s = mogg.call.shape @gather_shape_fn(%I,%J)                  # Call the registered shape helper.
       params={axis=-2,output_rank=4}, throws                  # Preserve declared parameters/raising behavior.
  mogg.output %s                                              # Return dimensions by value.
}
%out = broadcast(constant(1)) : [D4,D5,D6,D7]                   # Same dependent tensor computation.
```

The routine body receives a context for the common routine boundary; `CallShapeOp` receives only its declared data operands. Input pointers and any input dynamic dimensions are arguments, while the four *new output* dimension declarations remain attached to the routine. The later runtime lowering materializes those shape values for dependent work. [MAPOpsLowerableToMOGG.cpp:145](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:145) [MAPOpsLowerableToMOGG.cpp:158](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:158) [MAPOpsLowerableToMOGG.cpp:187](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:187)

**TEST · parameter preservation.** [shape-func-kgen-params.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/shape-func-kgen-params.mlir:1) enters at MO with an already-written `mo.invoke_shape_func`, imports builtin declarations, runs MO→MAP and MAP→MOGG, then checks printed IR. `CHECK` finds `mogg.call.shape …gather_shape_fn`; `CHECK-SAME` requires `[axis = -2 : index, output_rank = 4 : index]` on that call. The test proves that those parameters survive conversion; `[1,4,5,3]` and 60 ones above are illustrative shape arithmetic, not a captured result. [shape-func-kgen-params.mlir:8](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/shape-func-kgen-params.mlir:8)

Static attention declares `[32,32]` directly and has no pending output-dimension declarations requiring this path. [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-map-to-mogg-example-opaque-values" markdown="1">
<summary>Example 8 · Why does an opaque value cross the routine boundary differently from an ordinary tensor result?</summary>
<div class="qna-answer" markdown="1">

Use the test's `SIMDPair<S0=4,S1=8>` value. Its registered producer constructs two integer vectors, `[0,1,2,3]` and `[4,5,6,7,8,9,10,11]`. Its consumer writes those two vectors into a 12-element output tensor. These values follow from the inspected kernel body; no kernel was executed for this page. [testing_kernels.mojo:2996](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:2996) [testing_kernels.mojo:3009](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3009)

```transform
BEFORE · schematic MAP producer
%pair = map.iter.opaque inputs() -> Pair {        # Return a non-tensor aggregate.
  %p = map.call.opaque @MakeSimdPair               # Registration returns the aggregate itself.
       params={P0=4,P1=8} -> Pair                 # Sizes remain type/call parameters.
  map.yield %p                                   # Yield the by-value result.
}

AFTER · schematic MOGG producer
%pair = mogg.routine() -> Pair {                  # Same outer opaque result type.
  body(%ctx:context)                             # No input or output data pointer.
  %p = mogg.call.opaque @MakeSimdPair              # No context on this host-only execute call.
       params={P0=4,P1=8} -> Pair                 # Preserve both vector widths.
  mogg.output %p                                 # Forward the actual opaque SSA result.
}
```

The ordinary-output loop skips `MO::OpaqueType` results, so it creates no destination pointer for `%pair`. The nested call retains an actual result type, maps its result into `state.mapping`, and the enclosing conversion writes `mogg.output` for the non-destination value. [MAPOpsLowerableToMOGG.cpp:954](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:954) [MAPOpsLowerableToMOGG.cpp:653](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:653) [MAPOpsLowerableToMOGG.cpp:1028](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:1028)

```cpp
if (hasOutputChain && i + 1 == e)
  break;
if (isa<MO::OpaqueType>(resultTypes[i]))
  continue;
outputArgs[i] = addBodyArg(resultTypes[i]);
```

For an opaque *input*, the conversion adds a block argument with the original opaque type and maps it directly. The consumer's ordinary tensor output still gets a pointer and writable view. [MAPOpsLowerableToMOGG.cpp:945](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:945) [MAPOpsLowerableToMOGG.cpp:1005](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:1005)

```transform
BEFORE · schematic MAP consumer
%out = map.iter.opaque inputs(%p=%pair:Pair) outputs(%o:si32[12]) { # Mixed value/tensor boundary.
  map.call.opaque @ParameterizedOpaqueType inputs(%p) outputs(%o)  # Store both vector fields.
}

AFTER · schematic MOGG consumer
%out = mogg.routine(%pair:Pair) -> si32[12] {      # Keep the graph's tensor result.
  body(%p:Pair,%op:pointer,%ctx:context)          # Opaque argument stays typed as Pair.
  %shape = shape.create([12])                    # Only the tensor output needs a shape.
  %O = tensor.create(%op,%shape,si32,Output,true) # Construct its destination view.
  mogg.call.opaque @ParameterizedOpaqueType      # Preserve the selected consumer.
    inputs(%p:Pair) outputs(%O) params={P0=4}     # Same opaque data and output location.
}

Future consumer writes, derived from the registered body
O[0:4] = [0,1,2,3]                               # First vector occupies four elements.
O[4:12] = [4,5,6,7,8,9,10,11]                   # Second vector occupies eight elements.
```

**TEST · producer and consumer contracts.** [opaque-values.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/opaque-values.mlir:1) starts at MO, imports the test package, rewrites public opaque names to their registered types, then runs MO→MAP and MAP→MOGG. `@opaque_result` checks that no `outputs (` destination group exists and that the call result is forwarded by `mogg.output`. `@opaque_input` checks the typed opaque block argument beside the tensor output pointer and its shape/view reconstruction. [opaque-values.mlir:19](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/opaque-values.mlir:19) [opaque-values.mlir:33](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/opaque-values.mlir:33)

“Opaque” in `map.call.opaque` identifies a call whose implementation is not expressed as a graph-visible elementwise body. It does not imply that its tensor arguments have `!mo.opaque` types. The actual attention call uses four ordinary tensor views and therefore retains the pointer-based path in Example 1. [[kernel:82]] [[map-lowering:643]]

</div>
</details>
