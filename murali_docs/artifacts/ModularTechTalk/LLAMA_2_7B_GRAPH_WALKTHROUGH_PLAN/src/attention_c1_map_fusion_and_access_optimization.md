## QnA 1 · What can MAP optimize in this particular attention graph?

**There is one opaque attention computation, with ordinary tensor inputs and output.** Its internal attention loops remain inside P1. MAP can simplify the surrounding representation, but it has no separate graph-level matmul, softmax or PV producer/consumer pair to fuse here. [[entry:57]] [[kernel:82]] [[mo-to-map:201]] [[mo-to-map:215]]

<span id="qna-map-1-illustration-1"></span>

```state
IR1's attention path at MAP entry
  Q ─┐
  K ─┼──► map.iter.opaque ──► O ──► mo.output
  V ─┘       └─ call P1.FusedAttention
                BN=16, BD=8, device=gpu:0

Top-level attention iterator count: 1
Ordinary input roles: Q, K, V
Ordinary output role: O
Scalar attention-body instructions exposed to MAP: none
```

Identical **types** do not make `%q`, `%k` and `%v` identical **values**. All three are used, so operand cleanup cannot merge them simply because each is `float32[32,32]`. [[entry:43]] [[entry:61]] [[map-fold:29]]

## QnA 2 · What is the expanded optimization pipeline?

**Example links** map the earlier Q&A illustrations and the worked examples to their native checks, rewrites and handoffs. Links marked “later” describe a subsequent stage; a listed rewrite still requires its matching conditions.

```calltree
gcPipelineImpl → buildMapPipeline [[gc:585]] [[gc:338]]
├─ mapOpts.fuseMutableLoadStore copied from compilation options [[gc:533]]
├─ graph canonicalizer → CSE [[gc:340]]
│  └─ [↗ QnA 1 · attention entry — one opaque iterator; Q/K/V remain distinct](#qna-map-1-illustration-1)
├─ FuseKernelsPass::runOnOperation [[gc:343]] [[map-fuse:37]]
│  ├─ construct fusePm for mo.graph; schedule one ordered fusion round [[map-fuse:46]]
│  │  ├─ FuseElementwise: pure elementwise producer + consumer [[map-fuse:47]]
│  │  │  └─ [fuseInput](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:437) → [findEligibleProducer](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:315) → [spliceIntoForeach](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:185) → [spliceProducer](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:394)
│  │  │     ├─ [↗ QnA 7 · Example 3 — multi-use duplication budget: matched/skipped; shared fuseInput guard](#qna-map-opt-example-duplication-budget)
│  │  │     ├─ [↗ QnA 3 · Add/Mul before/after — intermediate element becomes a local scalar](#qna-map-3-illustration-1)
│  │  │     ├─ [↗ QnA 3 · ownership example — two iterator groups → one](#qna-map-3-illustration-2)
│  │  │     └─ [↗ QnA 7 · Example 2 — splice producer arithmetic into consumer body](#qna-map-opt-example-elementwise-splice)
│  │  ├─ FuseViews: producer computes through a view [[map-fuse:48]]
│  │  │  └─ fuseInput; pure view may duplicate, compute-containing view has cost guards
│  │  │     └─ findEligibleProducer + multi-use cost/large-operand guards [InputFusion.cpp:315](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:315)
│  │  ├─ FusePrologue: elementwise producer enters an opaque input callback [[map-fuse:49]]
│  │  │  └─ fuseInput → [findExistingPrologueCapture](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:122) → [spliceIntoClosure](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:206)
│  │  │     └─ [↗ QnA 7 · Example 4 — gather receives Add computation in its prologue](#qna-map-opt-example-gather-prologue-fusion)
│  │  ├─ FuseReductions: findProducer/fuseProducer, then findConsumer/fuseConsumer [[map-fuse:50]]
│  │  ├─ FuseEpilogue: consumer reads at the original loop index [[map-fuse:51]]
│  │  │  └─ [fuseOutput](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:406) → [findFusibleEdge](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:125) → [fuseOutputOnce](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:249) → erase consumer
│  │  │     └─ [↗ QnA 7 · Example 5 — Matmul/Add output callback; retain guarded boundary](#qna-map-opt-example-matmul-epilogue-fusion)
│  │  ├─ FuseReshapeEpilogue: reshape read plus compute, with the same output-fusion driver [[map-fuse:52]]
│  │  └─ FuseHorizontal: combine eligible siblings after producer/consumer fusion [[map-fuse:55]]
│  ├─ construct cleanupPm containing CSE only [[map-fuse:60]]
│  └─ do/while on this graph [[map-fuse:75]]
│     ├─ before = count outer IterOpInterface operations; skip nested iterator bodies [[map-fuse:63]]
│     ├─ runPipeline(fusePm, graphOp); propagate pass failure [[map-fuse:78]]
│     ├─ fused = before - kernelCount(); record statistic [[map-fuse:80]]
│     ├─ runPipeline(cleanupPm, graphOp) [[map-fuse:82]]
│     └─ repeat only while fused > 0 [[map-fuse:84]]
│        └─ [↗ QnA 7 · Example 1 — attention fixed point: 1 → 1, fused=0](#qna-map-opt-example-attention-fixed-point)
├─ if fuseMutableLoadStore: FuseMutableLoads [[gc:344]]
│  ├─ split eligible multi-user mutable loads; recollect MO::MutableLoadOp candidates
│  └─ compatible same-block kernel → replace tensor snapshot input by buffer + ordering
├─ canonicalizer → CSE [[gc:346]]
├─ AllocateAccumulatingBuffers → [assignIntermediateTensors](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:133) [[gc:351]]
│  ├─ existing intermediate / no direct store target → skip
│  ├─ dtype changes + accumulating kernel → [addScratchResult](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:89); reconnect results
│  │  └─ [↗ QnA 7 · Example 7 — scratch storage retains accumulation dtype](#qna-map-opt-example-accumulation-storage)
│  └─ otherwise → point intermediateTensor at the existing store target
├─ RaiseBackContiguousViews → [matchViewChain](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:51) → [raiseViewChain](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:134) [[gc:356]]
│  └─ pure reshape/contiguous slice chain → mo.static.reshape / mogg.slice
│     └─ [↗ QnA 7 · Example 8 — raise contiguous views outside scalar iteration](#qna-map-opt-example-contiguous-view-raising)
├─ OutlineStructuredKernelBodies → [outline](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:37) each structured iterator [[gc:358]]
│  └─ new map.iter.opaque → move original body into a capturing body closure
│     └─ [↗ QnA 7 · Example 9 — structured body gains an opaque owner](#qna-map-opt-example-structured-body-outlining)
├─ canonicalize iterator inputs/captures, including constants/dead slots [[gc:361]]
│  ├─ [↗ QnA 4 · duplicate operand example — same SSA input shares one slot](#qna-map-4-illustration-1)
│  ├─ [↗ QnA 4 · attention comparison — equal types do not merge Q/K/V](#qna-map-4-illustration-2)
│  └─ [↗ QnA 7 · Example 6 — deduplicate inputs and remove dead captures](#qna-map-opt-example-input-capture-cleanup)
├─ LoopInvariantViewMotion [[gc:362]]
│  └─ [collectIndexTransformChain](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:42) → [hoistablePrefixLen](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:76) → [hoistChain](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:88)
│     └─ create strided views outside closure; load through newly captured view
│        ├─ [↗ QnA 5 · 2×3 transpose example — addressing moves outside the loop](#qna-map-5-illustration-1)
│        └─ [↗ QnA 7 · Example 10 — hoist load views; retain store-feeding transforms](#qna-map-opt-example-view-motion-store-guard)
└─ CSE → canonicalizer: unify duplicate views before duplicate captures [[gc:366]]
   └─ [↗ QnA 6 · handoff example — one attention call, no exposed QK/softmax/PV body](#qna-map-6-illustration-1)
```

In a one-iterator attention path, there is no second iterator to fuse. The round can inspect candidates and stop with `before=1`, `after=1`, `fused=0`. This is a source-derived expectation for this path, not a measured compiler dump. The source terminates its loop using that exact count comparison. [[map-fuse:77]] [[map-fuse:84]]

**SOURCE · native definitions behind the branches.** The stage builder schedules pass objects on `mo.graph`, except body outlining which is scheduled at module scope. During `pm.run(IR1)`, the fuser creates and runs its own nested pass managers on each graph. Temporary rewriters, topology analyses, vectors and value maps belong to a pass invocation; the created/moved operations, regions and block arguments remain owned by IR1. [Pass-manager execution](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:654) [[gc:338]] [[map-fuse:46]]

| Tree branch | Concrete native entry and handoff |
| --- | --- |
| Elementwise / view / prologue fusion | [FuseElementwisePass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ElementwiseFuser.cpp:64), [FuseViewsPass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ViewFuser.cpp:39) and [FuseProloguePass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/PrologueFuser.cpp:40) supply different predicates to [fuseInput](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:437). It visits consumers in reverse topological order within their own blocks. |
| Reduction fusion | [FuseReductionsPass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ReductionFuser.cpp:302) gathers rowwise iterators, repeatedly consumes eligible producers, then eligible consumers. |
| Epilogue / reshape-epilogue fusion | [FuseEpiloguePass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/EpilogueFuser.cpp:38) and [FuseReshapeEpiloguePass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ReshapeEpilogueFuser.cpp:39) call [fuseOutput](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:406). It visits producers before consumers and splices consumer arithmetic into the producer's epilogue. |
| Horizontal fusion | [fuse](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/HorizontalFuser.cpp:406) constructs a replacement with merged inputs and concatenated output types, retaining separate destinations. |
| Mutable loads | [FuseMutableLoads::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/FuseMutableLoads.cpp:486) chooses between rewiring an existing buffer input, adding chains, or merging with existing chains. [maybeGetCompatibleKernel](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/FuseMutableLoads.cpp:45) rejects cross-block users and an opaque consumer that also takes the mutable buffer. |
| Accumulation | [assignIntermediateTensors](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:133) identifies each epilogue's storage source; a replacement iterator can gain a scratch result. |
| Contiguous views / outlining | [raiseViewChain](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:134) removes eligible pure view iterators. [outline](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:37) wraps remaining structured iterators and moves their bodies. |
| View motion | [loopInvariantViewMotionInClosure](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:118) rewrites load addresses, then erases unused index transforms. A final invariant check permits only store-feeding index transforms to remain. |

`MapPipelineOptions` has one field here: `fuseMutableLoadStore`, defaulting to true. It gates the extra mutable-load pass; `createFuseKernels()` is scheduled unconditionally by this builder. The elementwise pass separately defaults its large-operand threshold to `4096` elements and duplication cap to `64`. The shared driver applies that cap to **duplication of multi-use producers**; moving a sole-use producer does not consume the cap. [MAP options](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/Passes.h:407) [Elementwise options](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MAPDialect/Transforms/MAPTransforms.td:96) [Duplication guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:462)

After this builder's passes, `gcPipelineImpl` may add NaN checks, then an optional `map-pre-mogg` dump. Check injection follows fusion so a check cannot fuse into the kernel being checked. A dump is a scheduled inspection step, not another compilation representation. [[gc:585]] [[gc:588]] [[gc:568]]

## QnA 3 · Show a real fusion pattern with numerical values.

**TEST · inspected repository expectation:** the `basic_add_mul` test starts with `(X + Y) * Z` over eight elements. It expects one MAP elementwise iterator containing both registered scalar calls. This is an eligible comparison graph, not a rewrite of the attention custom call. [[test-map-fusion:3]] [[test-map-fusion:13]]

<span id="qna-map-3-illustration-1"></span>

```transform
BEFORE
  iterator A: T[i] = X[i] + Y[i]       i = 0..7
  iterator B: O[i] = T[i] * Z[i]       i = 0..7

AFTER
  one iterator:
    x = load X[i]
    y = load Y[i]
    t = Add(x,y)                       ← local scalar value
    z = load Z[i]
    o = Mul(t,z)
    store O[i] = o

Example lane i=0: X=2, Y=3, Z=4
  before: T=5; O=20
  after:  t=5; O=20
```

The intermediate tensor's eight stores and eight reloads can be removed from this representation. At four bytes per float32, that is **64 bytes of illustrative intermediate traffic**; it excludes all X/Y/Z/O accesses and says nothing about measured cache or bus traffic. The test checks the combined loads, calls and final store. [[test-map-fusion:17]]

<span id="qna-map-3-illustration-2"></span>

```state
Kernel groups: 2 → 1
Intermediate tensor T: materialized edge → local scalar computation
Final values: preserved
Fusion provenance: records both mo.add and mo.mul
```

The native pass keeps provenance in `fusionInfo`; fusion is more than changing a kernel count. It rewires the producer's computation into the consumer. The test also covers a diamond where three uses of the same intermediate at one index share its computation. [[test-map-fusion:23]] [[test-map-fusion:55]] [[test-map-fusion:71]]

## QnA 4 · What does canonicalization do to operands and captured values?

**TEST · repeated input:** a MAP body that receives the *same SSA value* twice can have one incoming slot while its call still uses that slot twice. This does not change the callee from an Add into a unary operation. [[test-map-dedup:20]]

<span id="qna-map-4-illustration-1"></span>

```transform
BEFORE
  map.iter.opaque inputs(%x as %a, %x as %b)
    call Add inputs(%a,%b) outputs(%out)

AFTER
  map.iter.opaque inputs(%x as %a)
    call Add inputs(%a,%a) outputs(%out)

If x[0]=3: Add still receives (3,3) and writes 6.
Outer incoming slots: 2 → 1; call's mathematical inputs: still 2.
```

For a capturing closure, the implementation identifies duplicate captured **Values**, redirects uses of duplicate block arguments, then erases those arguments. Dead captures can also be removed. [[map-fold:40]] [[map-fold:56]] [[map-fold:69]]

<span id="qna-map-4-illustration-2"></span>

```state
Attention: [%q,%k,%v] → [%q,%k,%v]
Reason: three distinct live SSA values, despite the same shape and dtype.

Eligible duplicate case: [%x,%x] → [%x]
Reason: both slots refer to the very same SSA value.
```

The repository also checks that deduplicating prologue captures leaves a now-unused load, which the canonicalizer removes. This illustrates why simplifications are applied to a fixed point rather than once per original operation. [[test-map-dedup:49]]

## QnA 5 · How does view motion simplify addressing? Use a 2×3 tensor.

**TEST · inspected transpose-view example:** a per-element index permutation is moved to a strided view outside the loop body. The loop then loads with its own index. The final output store remains. [[test-view:3]] [[test-view:16]]

<span id="qna-map-5-illustration-1"></span>

```transform
X = [[0,1,2],
     [3,4,5]]
Shape [2,3], row-major element strides [3,1]

BEFORE · inside each iteration of output shape [3,2]
  input_index = transpose_index([i,j], permutation=[1,0])
  value = load X[input_index]         ← X[j,i]
  store O[i,j] = value

AFTER
  XT = strided_view(X, shape=[3,2], strides=[1,3])
  inside the loop:
    value = load XT[i,j]
    store O[i,j] = value

At output [2,1]: input [1,2]
  offset = 1*3 + 2*1 = 5 elements = 20 bytes for float32
  strided-view offset = 2*1 + 1*3 = 5 elements
  value = 5 in either representation
```

The hoisted view is an addressing description; it does not require copying the input into a separate transposed allocation. The pass removes repeated index transformation work where the source pattern permits it. Its actual IR uses `map.index.transpose.stride`. [[test-view:20]] [[test-view:25]]

For attention, Kᵀ is part of the authored kernel computation. This MAP view example does not imply that a graph-level K transpose exists or gets hoisted for `FusedAttention`. [[entry:61]] [[kernel:346]]

## QnA 6 · What is the outgoing MAP representation, and why run this stage anyway?

<span id="qna-map-6-illustration-1"></span>

```state
Fused-attention path leaving MAP, schematic
└─ map.iter.opaque
   ├─ 3 live input tensors: Q, K, V
   ├─ 1 output tensor destination: O
   ├─ plain registered call: P1.FusedAttention
   └─ compile-time dictionary: {BD=8, BN=16}

Next: map.iter.opaque → mogg.routine
      tensor body arguments → pointer body arguments
      shape/tensor reconstruction + mogg.call.opaque
```

The same pipeline handles graphs containing elementwise operations, views, reductions and fusion-capable interfaces. This custom op is one case within that pipeline. A pass may establish or simplify a representation without reducing the operation count for every input graph. [[map-fuse:40]] [[gc:358]] [[map-to-mogg:19]]

Allocating accumulation buffers is specifically for applicable fused output epilogues, including mixed-precision cases. It should not be described as creating this attention kernel's score matrix or choosing BN/BD. Those choices came from the caller and implementation. [[gc:349]] [[entry:38]] [[kernel:78]]

## QnA 7 · How do MAP fusion and access optimizations change concrete operations, block arguments, captures and storage requirements? Show worked attention, elementwise, prologue, epilogue, canonicalization, accumulation, view and outlining examples, with native rewrites, numerical checks, matching conditions and skipped cases.

**SOURCE / TEST · ten worked examples.** The before/after IR is schematic, with short tensor types and symbol names. Native excerpts are implementation statements; numerical arrays and offsets are illustrative. Each test description identifies whether its input is MO, structured MAP, or already outlined MAP, then names the `RUN` passes and the relevant expected output. The `.mlir` files are independent test inputs and expected text, not functions called by production lowering. No compiler tests, MAX execution or GPU workloads were run for this reader.

<details class="qna-cell" id="qna-map-opt-example-attention-fixed-point" markdown="1">
<summary>Example 1 · What does the complete MAP stage do to the single opaque attention call?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · actual attention route.** Assume the live computation is the one custom call from the example, with three distinct Q/K/V values, a single output and `BD=8`, `BN=16`. MO→MAP has created ordinary tensor argument paths and no fusion closures. Shape equality does not supply a producer edge, a scalar body, or an identity between Q, K and V. [[entry:57]] [[kernel:82]] [[mo-to-map:201]] [[mo-to-map:215]]

```transform
BEFORE · MAP entry, schematic; T=f32[32,32] on gpu:0
  graph inputs %q:T, %k:T, %v:T                   // distinct live graph arguments
  %o = map.iter.opaque inputs(%q,%k,%v) outputs(%dst:T) {
    map.call.opaque @PKG.FusedAttention           // P1's registered implementation
      [BD=8, BN=16] inputs(%q,%k,%v) outputs(%dst) // ordinary tensor interfaces
  }
  mo.output %o                                   // only consumer of the output

AFTER · same live computation through buildMapPipeline
  graph inputs %q:T, %k:T, %v:T                   // still distinct graph arguments
  %o = map.iter.opaque inputs(%q,%k,%v) outputs(%dst:T) {
    map.call.opaque @PKG.FusedAttention           // same selected implementation
      [BD=8, BN=16] inputs(%q,%k,%v) outputs(%dst) // same tensor computation
  }
  mo.output %o                                   // same graph result
```

The fusion driver's termination condition uses the count of outer iterator operations:

```cpp
unsigned before = kernelCount();
if (failed(runPipeline(fusePm, graphOp)))
  return signalPassFailure();
fused = before - kernelCount();
numKernelsFused += fused;
```

Its iterator walk returns `WalkResult::skip()` at each outer iterator, so nested callback iterators are not counted as separate kernel groups. Here the source-derived count is `before=1`, `after=1`, `fused=0`, and the do/while stops after that round. This is a control-flow explanation, not an observed compilation trace. [[map-fuse:63]] [[map-fuse:77]] [[map-fuse:84]]

| Pass | Why the attention computation remains in this form |
| --- | --- |
| Input/prologue fusion | No elementwise producer or input prologue exists. |
| Output fusion | The attention destination has no epilogue; `mo.output` is not an elementwise consumer. |
| Mutable-load fusion | This graph fragment has no `mo.mutable.load`. |
| Accumulation | The pass finds no `map.iter.epilogue` to assign an intermediate tensor. |
| Contiguous-view raising | There is no structured view iterator to match. |
| Outlining | `map.iter.opaque` is already the outer kernel form; only structured iterators enter `outline`. |
| View motion | There is no capturing closure with an index-transform/load chain. |
| Operand cleanup | Q/K/V are distinct and all used. |

These outcomes follow from the native candidate selection in [findEligibleProducer](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:315), [findFusibleEdge](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:125), [accumulation's epilogue walk](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:133), [outlining's structured worklist](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:90), and [view motion's closure walk](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:159). [[map-fold:295]]

For an illustrative data check, zero Q gives equal attention weights; if every V entry is `2`, every output entry remains `2`. The stage keeps the same call, inputs and parameters, so it keeps that computation. The next examples use other graph structures that satisfy the transformation predicates. [[kernel:90]] [[kernel:99]] [[mo-to-map:238]]

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-elementwise-splice" markdown="1">
<summary>Example 2 · How are an Add producer and Mul consumer spliced into one scalar body?</summary>
<div class="qna-answer" markdown="1">

**TEST · MO input, then an isolated MAP subpass.** `basic_pointwise_fusion.mlir` supplies `(X+Y)*Z` over `[8]` f32. Its `RUN` imports kernels, runs `--mo-to-map`, then `--map-fuse-elementwise`. The test therefore constructs the two structured MAP iterators before exercising the elementwise fuser; it does not start from hand-authored MAP or run every later MAP optimization. Its checks require one iterator with three inputs, two registered scalar calls, one final store, and provenance from both MO operations. [[test-map-fusion:1]] [[test-map-fusion:3]] [[test-map-fusion:13]]

```transform
Illustrative X=[2,2,2,2,2,2,2,2], Y=[3,3,3,3,3,3,3,3], Z=[4,4,4,4,4,4,4,4]

BEFORE · schematic MAP after MO→MAP
  %t = iterator inputs(X,Y) output(T) index(i) {    // producer, shape [8]
    x = map.load X[i]                              // first scalar operand
    y = map.load Y[i]                              // second scalar operand
    t = map.call.elementwise Add(i,x,y)            // t=5
    map.store T[i] = t                             // materialized intermediate edge
  }
  %o = iterator inputs(%t,Z) output(O) index(j) {   // consumer, shape [8]
    t = map.load T[j]                              // read producer tensor
    z = map.load Z[j]                              // other consumer operand
    o = map.call.elementwise Mul(j,t,z)            // o=20
    map.store O[j] = o                             // final store
  }
  mo.output %o                                    // all eight values are 20

AFTER
  %o = iterator inputs(X,Y,Z) output(O) index(j) {  // one replacement computation group
    x = map.load X[j]                              // producer loads mapped to consumer index
    y = map.load Y[j]                              // same element correspondence
    t = map.call.elementwise Add(j,x,y)            // local SSA scalar, t=5
    z = map.load Z[j]                              // consumer's remaining load
    o = map.call.elementwise Mul(j,t,z)            // use t directly, o=20
    map.store O[j] = o                             // final destination retained
  }
  mo.output %o                                    // same tensor values
```

`FuseElementwisePass` accepts a pair only when both are elementwise iterators without view transformations. The shared input-fusion driver prefers a sole-use producer, verifies the block/device/chain constraints and finds the consumer load site. The core splice clones the producer region, extracts its stored scalar and removes its former destination:

```cpp
rewriter.cloneRegionBefore(src.getBody(), dstRegion, dstRegion.begin());
Block &srcClone = dstRegion.front();

StoreOp storeOp = getUniqueStore(srcClone);
Value fusedVal = storeOp.getValue();
rewriter.eraseOp(storeOp);
srcClone.eraseArgument(numSrcInputs);
```

It then maps the cloned input arguments to the new consumer input slots and the cloned index to the load's index, inlines the block before that load, replaces the load by `fusedVal`, and removes the obsolete intermediate input slot. Once the producer result is unused, the driver erases its old iterator. [Elementwise predicate](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ElementwiseFuser.cpp:64) [Eligibility](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:315) [Splice and rewiring](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:408) [Producer deletion](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:471)

| Changed structure | Result |
| --- | --- |
| Consumer inputs `(T,Z)` | `(X,Y,Z)` |
| Producer store and consumer load of T | Removed; their value connection becomes scalar SSA |
| Top-level iterators | `2 → 1` |
| External graph inputs/output | Unchanged |
| Fusion metadata | Records Add and Mul ancestry |

The intermediate represents eight f32 writes and eight reads, or `8×4×2=64` bytes of illustrative tensor traffic removed from this representation. This excludes all X/Y/Z/O accesses and is not a measured bandwidth figure. [[test-map-fusion:17]] [[test-map-fusion:23]]

**Same-index sharing:** the file's `@fusion_diamond` reads one Add result through `Mul(t,t)` and `ReLU(t)`. At a lane with `X=2,Y=3`, `t=5`, so `t*t+ReLU(t)=30`. The expected body computes Add once. `mergeRedundantLoads` keys loads by `(block,index)` before cloning so three same-index reads share one scalar producer computation. [[test-map-fusion:55]] [[test-map-fusion:71]] [Load sharing implementation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:167)

**Skipped case:** this elementwise wrapper rejects a view-bearing producer or consumer; `FuseViews` has a separate predicate. The shared driver also refuses to move a producer across a block boundary or past a prohibited reshape read. [Pure-elementwise predicate](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ElementwiseFuser.cpp:67) [Block and reshape guards](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:329)

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-duplication-budget" markdown="1">
<summary>Example 3 · Why can a fan-out Add stay separate under one option value and be copied into both consumers under another?</summary>
<div class="qna-answer" markdown="1">

**TEST · the elementwise subpass's cost guard.** `large_operand_dup.mlir` runs import → MO→MAP → `map-fuse-elementwise` twice. The first `RUN` uses the default threshold; the second sets `large-tensor-element-threshold=16384` and checks `HIGHTHRESH` expectations. Both consume the same graph with two GPU inputs `[4096,2]`, an Add, and two consumers, Tanh and ReLU. Each input has `8192` elements. [Both RUN commands](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/ElementwiseFusion/large_operand_dup.mlir:1) [Input graph](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/ElementwiseFusion/large_operand_dup.mlir:6)

```transform
BEFORE · schematic, all tensors f32[4096,2] on gpu:0
  T = iterator Add(X,Y)                       // one producer with two users
  U = iterator Tanh(T)                        // first consumer
  V = iterator ReLU(T)                        // second consumer
  mo.output U,V                              // both results escape

AFTER default threshold=4096 · duplication skipped
  T = iterator Add(X,Y)                       // two 8192-element inputs count as large
  U = iterator Tanh(T)                        // reads materialized T
  V = iterator ReLU(T)                        // also reads T
  mo.output U,V                              // producer remains shared

AFTER threshold=16384 · duplication matched
  U = iterator { t=Add(X[i],Y[i]); store U[i]=Tanh(t) } // Add copied into first body
  V = iterator { t=Add(X[i],Y[i]); store V[i]=ReLU(t) } // Add copied into second body
  mo.output U,V                              // materialized T no longer needed
```

The decisive implementation counts expensive input tensors:

```cpp
std::optional<int64_t> numElements = tensorType.getNumElements();
return !numElements.has_value() || *numElements > largeTensorElementThreshold;
```

`isCheapToDuplicate` returns false after the second large input, and `config.canDuplicate` additionally rejects host kernels. With threshold `4096`, both 8192-element inputs are large; with `16384`, neither is large. The decision is about additional input reads caused by copying a producer's computation into multiple consumers. [Size classification](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ElementwiseFuser.cpp:38) [Large-input count](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ElementwiseFuser.cpp:49) [Host guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/ElementwiseFuser.cpp:73)

For illustrative values `X[i]=-1`, `Y[i]=1`, each version computes `T[i]=0`, `U[i]=tanh(0)=0` and `V[i]=ReLU(0)=0`. Input tensor storage is not cloned by the compiler; the duplicated objects are scalar computations inside two kernel bodies. The test's default `CHECK` requires one Add producer and two readers; `HIGHTHRESH` requires Add in both consumer bodies. [Default expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/ElementwiseFusion/large_operand_dup.mlir:13) [Changed-threshold expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/ElementwiseFusion/large_operand_dup.mlir:24)

**Another rejection:** every user must be able to absorb the duplicated computation. `duplicationAllowed` rejects a producer whose result also reaches `mo.output`, because that tensor must still be materialized. The file's `@unaryExpEscapingOutput` checks that only one Exp remains even though a unary producer has only one large input. [All-users guard](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:303) [Escaping-output test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/ElementwiseFusion/large_operand_dup.mlir:53)

These expected iterator counts stop at the isolated elementwise pass. The full MAP fusion round also has horizontal fusion, which can combine eligible siblings later. A rejection here therefore identifies this particular rewrite decision. [[map-fuse:47]] [[map-fuse:55]]

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-gather-prologue-fusion" markdown="1">
<summary>Example 4 · How does gather(Add(A,B), I) compute the Add inside gather's input callback?</summary>
<div class="qna-answer" markdown="1">

**TEST · full fusion round after MO lowering.** `prologue_fusion.mlir` imports `%kernel_api_test_lib`, runs `--mo-to-map`, then `--map-fuse-kernels`. In `@add_gather`, A/B have shape `[3,3]`, I has shape `[2,2]`, and gather's axis-0 output is `[2,2,3]`. The checks require Add's two loads and scalar call inside gather's data prologue, no standalone elementwise iterator, and `operandIndices=[0]` in fusion provenance. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/PrologueFusion/prologue_fusion.mlir:1) [Input and expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/PrologueFusion/prologue_fusion.mlir:3)

```transform
Illustrative A=[[0,1,2],[3,4,5],[6,7,8]], B=all 10, I=[[2,0],[1,2]]

BEFORE · schematic MAP
  %t = elementwise Add(A,B)                        // materialize f32[3,3] values 10…18
  %o = opaque inputs(%t,I) outputs(O) {
    %pd = prologue(index d) captures(T) {           // gather's data callback
      v = map.load T[d]                             // read materialized Add result
      map.yield v                                  // scalar value for gather
    }
    %pi = prologue(index k) captures(I) {           // indices callback
      n = map.load I[k]                             // obtain requested row number
      map.yield n                                  // pass row number to gather
    }
    %eo = epilogue(value v,index j) captures(O) {
      map.store O[j] = v                            // gather output destination
    }
    map.call.opaque Gather[axis=0] inputs(%pd,%pi) outputs(%eo) // unchanged gather logic
  }

AFTER
  %o = opaque inputs(I,A,B) outputs(O) {             // old T slot removed, A/B appended
    %pd = prologue(index d) captures(A,B) {           // rebuilt callback dependencies
      a = map.load A[d]                             // producer's first load
      b = map.load B[d]                             // producer's second load
      t = map.call.elementwise Add(d,a,b)           // compute only the requested element
      map.yield t                                  // replace old load result
    }
    %pi = prologue(index k) captures(I) {           // indices callback unchanged
      n = map.load I[k]                             // same selected row
      map.yield n                                  // same row number
    }
    %eo = epilogue(value v,index j) captures(O) {
      map.store O[j] = v                            // same output coordinate
    }
    map.call.opaque Gather[axis=0] inputs(%pd,%pi) outputs(%eo) // use fused data callback
  }
  mo.output %o                                     // [[[16,17,18],[10,11,12]],
                                                   //  [[13,14,15],[16,17,18]]]
```

`findExistingPrologueCapture` requires the opaque input argument to have one use in a prologue and the corresponding capture argument to have one load use. This gives the rewriter a precise scalar replacement site. `spliceIntoClosure` clones the producer body, deletes its store/output argument, appends A/B as opaque inputs and prologue captures, then performs this rewiring:

```cpp
rewriter.inlineBlockBefore(&srcClone, oldLoad, fusedArgs);

oldLoad.getResult().replaceAllUsesWith(fusedVal);
oldLoad.erase();
closure.eraseCaptureAt(capture.captureIdx);
dst.eraseInputsAt(argIdx);
```

The cloned Add receives the old load's index. At output `[0,0,2]`, gather requests data `[2,2]`, so the prologue computes `8+10=18`. Before fusion it read the same `18` from T. The opaque body's input slots change from two `(T,I)` to three `(I,A,B)`; its output destination stays O. The data prologue changes from one captured tensor to two. [Capture matcher](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:122) [Native splice](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:206) [Final rewiring](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/InputFusion.cpp:247)

**Skipped cases have different causes.** `@no_fuse_multi_use` also returns T, so the prologue fuser's `canDuplicate=false` and the all-users guard keep the producer. `@no_fuse_across_blocks` places the producer outside an `mo.if` branch and its gather inside; the block guard prevents moving it. `@no_fuse_plain_input` feeds Matmul's ordinary input, for which no prologue load site exists. Each test explicitly checks the surviving elementwise producer. [No-duplication policy](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/PrologueFuser.cpp:48) [Multi-use test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/PrologueFusion/prologue_fusion.mlir:178) [Block test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/PrologueFusion/prologue_fusion.mlir:153) [Plain-input test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/PrologueFusion/prologue_fusion.mlir:213)

The same plain-input limitation applies to the authored attention call's Q/K/V interfaces. This example establishes a gather route with declared callback support. [[kernel:83]] [[mo-to-map:201]]

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-matmul-epilogue-fusion" markdown="1">
<summary>Example 5 · How does Matmul(A,B)+Z become arithmetic in a producer's output callback?</summary>
<div class="qna-answer" markdown="1">

**TEST · MO input and the isolated epilogue pass.** `epilogue_fusion.mlir` imports kernels, runs `--mo-to-map`, then `--map-fuse-epilogue`. Its `@matmul_add` supplies `[8,8]` f32 tensors and checks one opaque iterator whose epilogue loads Z, adds the incoming Matmul scalar and stores the final result. The registered Matmul call still receives A/B and writes through the epilogue. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/EpilogueFusion/epilogue_fusion.mlir:1) [Input and CHECK sequence](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/EpilogueFusion/epilogue_fusion.mlir:3)

```text
Smaller illustrative matrices, all f32[2,2]:
  A=[[1,2],[3,4]], B=[[5,6],[7,8]], Z=[[1,1],[1,1]]
  T=A@B=[[19,22],[43,50]]; O=T+Z=[[20,23],[44,51]]
```

```transform
BEFORE · schematic MAP
  %t = opaque inputs(A,B) outputs(T) {
    %ep = epilogue(value m,index i) captures(T) {    // Matmul's initial output callback
      map.store T[i] = m                            // materialize product element
    }
    map.call.opaque Matmul inputs(A,B) outputs(%ep)  // registered producer
  }
  %o = elementwise inputs(%t,Z) outputs(O) index(i) {
    m = map.load T[i]                               // read product element
    z = map.load Z[i]                               // read added tensor element
    r = map.call.elementwise Add(i,m,z)             // postprocess product
    map.store O[i] = r                              // write final result
  }
  mo.output %o                                     // final graph output

AFTER
  %o = opaque inputs(A,B,Z) outputs(O) {
    %ep = epilogue(value m,index i) captures(O,Z) {  // m arrives directly from Matmul
      z = map.load Z[i]                             // new epilogue capture supplies Z
      r = map.call.elementwise Add(i,m,z)           // same scalar postprocessing
      map.store O[i] = r                            // final destination replaces T
    }
    map.call.opaque Matmul inputs(A,B) outputs(%ep)  // original Matmul implementation
  }
  mo.output %o                                     // same [[20,23],[44,51]]
```

`findFusibleEdge` requires a single-use result, an existing epilogue, a same-block elementwise consumer, a supported output-store structure, and no dependence cycle. The epilogue wrapper also checks that the consumer reads the producer at its untransformed loop index. Once matched, `fuseOutputOnce` replaces every load of the fused input by the scalar already arriving at the epilogue:

```cpp
for (Operation *user : llvm::to_vector(fusionArg.getUsers())) {
  auto load = cast<LoadOp>(user);
  load.getResult().replaceAllUsesWith(val);
  load.erase();
}
```

Here `val` is the Matmul value `m`, not a new load from T. Other consumer inputs become new opaque inputs and epilogue captures. The old producer store is removed, the consumer body is moved into the epilogue, and a store of `finalVal` is rebuilt at the destination. [Matcher](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:125) [Untransformed-index predicate](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:39) [Load substitution](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:288) [Capture insertion](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:298) [Body move and final stores](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:367)

For `[0,1]`, Matmul produces `1×6+2×8=22`. Before fusion the Add reloads 22; afterward its argument is the incoming scalar 22. It adds `Z[0,1]=1` and stores `23` in both forms. The driver redirects the consumer's result users to the producer's revised result and erases the old consumer iterator, while preserving both operations in fusion provenance. [Result replacement](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:374) [Provenance and deletion](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:436)

**Skipped case:** `@no_fuse_plain_output` keeps a plain-output opaque producer followed by Add because there is no epilogue. That guard also applies to P1's ordinary attention output. `@no_fuse_multi_use` has two consumers of one result; the isolated epilogue pass leaves them separate. Later horizontal fusion in the complete pipeline can change eligibility, so that test does not establish the final whole-pipeline count. [Plain-output expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/EpilogueFusion/epilogue_fusion.mlir:163) [Multi-use expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/EpilogueFusion/epilogue_fusion.mlir:151) [[kernel:82]] [[map-fuse:55]]

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-input-capture-cleanup" markdown="1">
<summary>Example 6 · How can two input slots collapse while a call still receives two mathematical operands?</summary>
<div class="qna-answer" markdown="1">

**TEST · input is already MAP.** `kernel_operands_basic.mlir` runs `mo-opt %s --canonicalize | FileCheck %s`. The supplied graph contains `map.iter.opaque` and declaration stubs; no MO→MAP pass is needed. `@dedupes_repeated_input` passes the same `%x` twice. Its expected iterator has one incoming slot, while the nested call still lists that slot twice. [[test-map-dedup:1]] [[test-map-dedup:20]]

```transform
BEFORE · schematic, X:f32[2,3,4]
  %r = map.iter.opaque                           // owns two redundant input slots
    inputs(%x as %a, %x as %b) outputs(%out) {
      map.call.opaque @kernels.add               // two mathematical call operands
        inputs(%a,%b) outputs(%out)
    }
  mo.output %r                                  // same graph output

AFTER
  %r = map.iter.opaque                           // owns one input slot
    inputs(%x as %a) outputs(%out) {
      map.call.opaque @kernels.add               // still passes two arguments
        inputs(%a,%a) outputs(%out)
    }
  mo.output %r                                  // result computation unchanged
```

The native helper keys its `seen` map by `Value`. It replaces uses of each duplicate block argument, then removes duplicate operand slots in reverse order so later indices do not shift before they are processed:

```cpp
for (unsigned idx : duplicates)
  rewriter.replaceAllUsesWith(inputArgs[idx], seen.lookup(getInputs()[idx]));
rewriter.modifyOpInPlace(getOperation(), [&] {
  for (unsigned idx : llvm::reverse(duplicates))
    eraseInputsAt(idx, 1);
});
```

At an illustrative lane with `x=3`, the call arguments remain `(3,3)`; an Add contract still produces `6`. The outer graph still has its original `%x` argument. The changing object is the iterator's input list and the corresponding argument list of its owned block. Same shape, dtype or even equal numerical contents do not establish the SSA identity required by this map. [[map-fold:295]] [[test-map-dedup:23]]

**Captures are a second boundary.** The file's `@dedupes_prologue_captures` starts with one opaque input but two captures of that same input inside a prologue. Its first load is dead; only the second feeds `map.yield`. The canonicalizer deduplicates captures and removes the unused load:

```transform
BEFORE · schematic prologue body
  prologue(index i) captures(c0=A,c1=A) {         // two captures of the same Value
    v0 = map.load c0[i]                          // unused scalar result
    v1 = map.load c1[i]                          // live scalar result
    map.yield v1                                // callback returns A[i]
  }

AFTER · canonicalizer fixed point
  prologue(index i) captures(c=A) {               // one capture slot and block argument
    v = map.load c[i]                            // surviving load
    map.yield v                                 // same returned scalar
  }
```

`DedupeCapturingClosureCaptures` performs the analogous Value-based replacement. `DropDeadCapturingClosureCaptures` then erases a capture if its shadowing argument has no uses. The test's `CHECK-NEXT` sequence requires one capture, one load and its yield; this is not a claim that deduplication itself performs arithmetic CSE. [[map-fold:40]] [[map-fold:56]] [[map-fold:69]] [[test-map-dedup:49]]

**Skipped cases:** distinct live Q/K/V values cannot deduplicate. A live capture cannot be dropped. The outer-input cleanup separately preserves chain-typed scheduling inputs even if a body does not consume them. `@keeps_prologue_captured_input` checks that a tensor remains an opaque input when its prologue uses it. [[map-fold:164]] [[test-map-dedup:33]]

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-accumulation-storage" markdown="1">
<summary>Example 7 · Why can a dtype-changing epilogue add a scratch result without adding a graph output?</summary>
<div class="qna-answer" markdown="1">

**TEST · MO → fusion → accumulation assignment.** `mixed-precision-matmul.mlir` imports test kernels, runs `--mo-to-map`, `--map-fuse-kernels`, and `--map-allocate-accumulating-buffers`. Its `@matmul_cast` computes Matmul in f32 and casts the result to si32, all at shape `[2,2]`. The checks require two iterator outputs, si32 and f32, and an epilogue `intermediate` operand of f32. The graph continues to return only the cast result. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/AllocateAccumulatingBuffers/mixed-precision-matmul.mlir:1) [Input and expected scratch](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/AllocateAccumulatingBuffers/mixed-precision-matmul.mlir:9)

```transform
BEFORE this pass · schematic, cast already fused
  %o = map.iter.opaque inputs(A:f32[2,2],B:f32[2,2])
    outputs(%oi:si32[2,2]) {                       // one external result destination
    %ep = epilogue(value m:f32,index i) captures(%oi) {
      n = map.call.elementwise Cast(i,m) : si32     // convert after producing m
      map.store %oi[i] = n                         // final store has integer dtype
    } intermediate=absent
    map.call.opaque Matmul inputs(A,B) outputs(%ep) // kernel still produces f32
  }
  mo.output %o                                    // one si32 graph result

AFTER
  %o,%scratch = map.iter.opaque inputs(A,B)
    outputs(%oi:si32[2,2],%tmp:f32[2,2]) {          // extra internal destination
    %ep = epilogue(value m:f32,index i)
      intermediate %tmp captures(%oi) {            // producer accumulation source is f32
      n = map.call.elementwise Cast(i,m) : si32     // same postprocessing
      map.store %oi[i] = n                         // final output stays si32
    }
    map.call.opaque Matmul inputs(A,B) outputs(%ep) // same f32 producer contract
  }
  mo.output %o                                    // scratch is not a new user output
```

`findStoreTarget` searches direct stores in the epilogue body and prefers a destination whose element type matches the incoming scalar. The assignment path creates scratch only when the chosen output type differs **and** `containsMatmulOrConvOrConcat(kernel)` identifies an accumulating kernel:

```cpp
Type scratchType = resultTy.cloneWith(*inDType);
BlockArgument scratchArg;
IterOpaqueOp newKernel =
    addScratchResult(kernel, scratchType, scratchArg);
epi.getIntermediateTensorMutable().assign(scratchArg);
return assignIntermediateTensors(newKernel);
```

An operation's result list cannot grow in place. `addScratchResult` creates a replacement iterator, copies device/effects/dispatch/provenance, moves the old region with `takeBody`, appends a scratch output block argument, redirects old result users and erases the old iterator. The recursive assignment resumes on the replacement and skips the already assigned epilogue. [Target selection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:66) [Predicate and assignment](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:142) [Iterator replacement](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:89)

For illustrative `A=[[1,2],[3,4]]`, `B=[[5,6],[7,8]]`, the f32 products are `[[19,22],[43,50]]`, and the final si32 values are the same integers. Both tensor types occupy four bytes per element here; scratch is needed to preserve the accumulation **type**, even though this test's two element widths happen to match. The added f32 `[2,2]` result describes 16 bytes of logical storage. This pass records that storage requirement in IR; it does not allocate a device address. [Scratch type construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:146)

**Matched alternative, no extra scratch:** `@matmul_add_and_cast` has two final destinations, one f32 and one si32, after horizontal/output fusion. `findStoreTarget` can use the existing f32 destination as `intermediateTensor`, so no third result is added. The sibling test reverses the store order and still expects the f32 destination: selection follows dtype, not the first store. [Two-output expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/AllocateAccumulatingBuffers/mixed-precision-matmul.mlir:26) [Reversed-order expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/AllocateAccumulatingBuffers/mixed-precision-matmul.mlir:52)

An already assigned intermediate is skipped, a same-dtype store normally becomes the intermediate directly, and an attention opaque with no epilogue has nothing to process. None of these branches introduces the attention score matrix or changes BN/BD. [Skip and direct-target paths](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/AllocateAccumulatingBuffers.cpp:133) [[kernel:82]]

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-contiguous-view-raising" markdown="1">
<summary>Example 8 · Why can a pure slice or reshape become a standalone view again before outlining?</summary>
<div class="qna-answer" markdown="1">

**TEST · the pass sees the output of MO→MAP.** `raise-back-contiguous-views.mlir` supplies MO slice/reshape operations, imports built-ins, lowers them with `--mo-to-map`, then runs `--map-raise-back-contiguous-views`. The matched slice keeps complete trailing rows: `[10,64]` sliced by starts `[1,0]`, stops `[4,64]`, steps `[1,1]` gives `[3,64]`. `CHECK-NOT` excludes the former iterator and `CHECK` requires scalar bounds plus `mogg.slice`. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/RaiseBackContiguousViews/raise-back-contiguous-views.mlir:1) [Contiguous input and expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/RaiseBackContiguousViews/raise-back-contiguous-views.mlir:13)

```transform
BEFORE this pass · schematic, old bound constants omitted
  %s = map.iter.elementwise<[3,64]>                 // pure view iterator
    inputs(X,start,stop,step) outputs(S) index(i) {
      j = map.index.slice<starts=[1,0],stops=[4,64],steps=[1,1]>(i,start,stop,step)
                                                    // [r,c] → [r+1,c]
      v = map.load X[j]                             // the body's only load
      map.store S[i] = v                            // unchanged loaded value
    }
  mo.output %s                                     // shape [3,64]

AFTER
  %start0 = mo.constant 1 : si64[]                  // scalar first-row bound
  %stop0  = mo.constant 4 : si64[]                  // scalar exclusive stop
  %s = mogg.slice(X,%start0,%stop0) : f32[3,64]      // standalone contiguous slice view
  mo.output %s                                     // same selected rows
```

`matchViewChain` accepts only a body consisting of slice/reshape index transformations, one input load, and its output store. For a slice, `matchContiguousSliceDim` requires known dimensions, unchanged extents outside the first non-unit input dimension, static bounds on that dimension and step 1. A matched chain is visited in reverse because its index operations run from output coordinates back to input, while data views are built from input toward output. [Pure-view matcher](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:51) [Contiguity checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:103)

The slice branch creates scalar bounds and then the view operation:

```cpp
cur = MOGG::SliceOp::create(builder, op.getLoc(), outType, cur, startK,
                            stopK)
          ->getResult(0);
```

After rebuilding the view chain, it redirects users and erases the iterator. For illustrative row-major `X[r,c]=100*r+c`, the output element `[0,7]` is `X[1,7]=107`. Its offset from the original input base is `1×64+7=71` elements. The slice begins 64 elements into X, so the view-relative offset is `7`; `64+7=71` preserves the address. The offsets describe the zero-copy view contract, not a measured pointer. [View construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:161) [Replacement](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:178)

**Reshape companion:** the same test's `[4,6] → [24]` pure reshape is raised back to `mo.static.reshape`. The original MO operation was replaced by MO→MAP; this pass creates a new MO reshape with the same view contract. It does not restart the earlier MO pipeline. [Reshape test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/RaiseBackContiguousViews/raise-back-contiguous-views.mlir:83) [Reshape construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/RaiseBackContiguousViews.cpp:147)

```transform
BEFORE · pure flatten, schematic
  %r = iterator<[24]> {                             // structured view of X:f32[4,6]
    j = map.index.reshape<[4,6]->[24]>(i)          // [k] → [k/6,k%6]
    v = map.load X[j]                               // preserve linear element k
    map.store R[i] = v                              // no arithmetic
  }
AFTER
  %r = mo.static.reshape(X) : f32[4,6] -> f32[24]   // same 24 elements in order
```

**Skipped case:** slicing `[10,64]` to `[3,32]` with starts `[1,32]` changes both dimensions, leaving gaps between selected row tails. The test expects it to remain an elementwise view iterator under this pass. Dynamic bounds and a broadcast are also kept. Additional arithmetic in the body fails the pure-view matcher. Being skipped here does not imply that a later general strided-view route must copy the data. [Noncontiguous case](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/RaiseBackContiguousViews/raise-back-contiguous-views.mlir:44) [Dynamic bounds](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/RaiseBackContiguousViews/raise-back-contiguous-views.mlir:61) [Broadcast case](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/RaiseBackContiguousViews/raise-back-contiguous-views.mlir:123)

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-structured-body-outlining" markdown="1">
<summary>Example 9 · How does outlining move a scalar body into an opaque container while preserving its operations?</summary>
<div class="qna-answer" markdown="1">

**TEST · input already contains a structured MAP iterator.** `outline-structured-kernel-bodies.mlir` runs only `--map-outline-structured-kernel-bodies`. It supplies a `[256,256]` Add iterator plus a declaration stub needed to verify its scalar call. The checks require an enclosing `map.iter.opaque`, a `map.iter.elementwise.body`, captures for X/Y/O, and the same two loads, Add and store in that body. [RUN and declaration](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/outline-structured-kernel-bodies.mlir:1) [Input and expected body](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/outline-structured-kernel-bodies.mlir:13)

```transform
BEFORE · schematic, use smaller X/Y:f32[2,3] for reading
  %r = map.iter.elementwise<[2,3]>                 // structured outer iterator
    inputs(%x as %a,%y as %b) outputs(%o) index(%i) {
      %av = map.load %a[%i]                        // original scalar operation 1
      %bv = map.load %b[%i]                        // original scalar operation 2
      %v = map.call.elementwise Add(%i,%av,%bv)    // original scalar operation 3
      map.store %o[%i] = %v                        // original scalar operation 4
    }
  mo.output %r                                    // graph result

AFTER
  %r2 = map.iter.opaque                           // new outer kernel container
    inputs(%x as %xi,%y as %yi) outputs(%oi) {
    map.iter.elementwise.body<[2,3]>               // new capturing closure shell
      index(%j) captures(%a=%xi,%b=%yi,%o=%oi) {
        %av = map.load %a[%j]                      // same operation, new index use
        %bv = map.load %b[%j]                      // same operation
        %v = map.call.elementwise Add(%j,%av,%bv)  // same scalar call
        map.store %o[%j] = %v                      // same final store
      }
    }
  mo.output %r2                                   // redirected outer result use
```

`outline` copies the old iterator's input/result types, chain, device, effects, dispatch and provenance into a new opaque operation. Its new body arguments become closure captures. `IterElementwiseOp::createBodyClosure` constructs `IterElementwiseBodyOp`; the reduction counterpart constructs `IterRowwiseBodyOp`. [Container creation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:37) [Elementwise closure factory](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOps.cpp:794) [Rowwise closure factory](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOps.cpp:1128)

The implementation moves the old region; it does not clone its scalar operations:

```cpp
Region &region = body.getBody();
region.takeBody(loop->getRegion(0));
Block &movedBody = region.front();
BlockArgument newIndexArg =
    movedBody.insertArgument(0u, oldIndexArg.getType(), loc);
oldIndexArg.replaceAllUsesWith(newIndexArg);
movedBody.eraseArgument(movedBody.getNumArguments() - 1);
```

The old block argument order is `[X,Y,O,index]`; the closure needs `[index,X,Y,O]`. Inserting a new index argument at position 0 and replacing the old index's uses accomplishes that reordering. The data arguments move with the block and now correspond to explicit captures. The old outer iterator is erased after its result uses move to the new opaque result. [Body transfer and index repair](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:69)

For illustrative `X[1,2]=4`, `Y[1,2]=5`, the same two loads and Add still produce `9`. The outer iterator count stays one; the number of ownership boundaries grows from one iterator body to an opaque body containing a capturing closure. An opaque container produced this way retains a visible elementwise body, while the attention container holds a registered `map.call.opaque`. Those two containers have different contents even though their outer operation name matches. [Body creation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:65) [[mo-to-map:238]]

**Skipped case:** the test's `@alreadyOpaque` expects its existing opaque operation unchanged. The native worklist only collects `StructuredIterOpInterface`, which explains why the attention iterator does not gain an extra wrapper here. [Already-opaque test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/outline-structured-kernel-bodies.mlir:73) [Worklist filter](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutlineStructuredKernelBodies.cpp:90)

</div>
</details>

<details class="qna-cell" id="qna-map-opt-example-view-motion-store-guard" markdown="1">
<summary>Example 10 · How does view motion hoist a load's addressing transform while retaining transforms needed by stores?</summary>
<div class="qna-answer" markdown="1">

**TEST · input is already outlined MAP.** `loop-invariant-view-motion-transpose.mlir` supplies an opaque iterator containing an elementwise body. Its `RUN` is `builtin.module(mo.graph(map-livm, map.iter.opaque(canonicalize,cse)))`: view motion runs on the graph, then cleanup runs within opaque containers. For `@transposeSwap`, `CHECK-NEXT` requires a transpose stride view before the body, a new capture of that view, and a load indexed by the body's original index. `CHECK-NOT` excludes the old in-body transpose index operation. [[test-view:1]] [[test-view:16]] [[test-view:28]]

```transform
Input X=[[0,1,2],[3,4,5]] : f32[2,3]              // row-major strides [3,1]

BEFORE · schematic, inside map.iter.opaque
  body<[3,2]> index(i) captures(X,perm,O) {
    j = map.index.transpose<perm=[1,0]>(i,perm)    // [row,col] → [col,row]
    v = map.load X[j]                             // read original input tensor
    map.store O[i] = v                            // write output-space coordinate
  }

AFTER · hoisted op remains in the enclosing opaque body
  XT = map.index.transpose.stride<perm=[1,0]>(X,perm) // shape [3,2], strides [1,3]
  body<[3,2]> index(i) captures(O,XT) {
    v = map.load XT[i]                            // same source element via a view
    map.store O[i] = v                            // final output store retained
  }
```

`collectIndexTransformChain` walks backward from a load's index, closest to the load first. `hoistablePrefixLen` keeps only the part that does not feed a store index. `hoistChain` creates strided views before the closure, adds the final view as a capture, then changes the load's tensor and index operands:

```cpp
Value tensor = outerTensor;
for (IndexTransformOpInterface transform : chain)
  tensor = transform.createStridedView(builder, tensor, outerValues);
BlockArgument newCaptureArg = closure.addCapture(tensor);

IndexTransformOpInterface outermost = chain.back();
loadOp.getTensorMutable().assign(newCaptureArg);
loadOp.getIndexMutable().assign(outermost.getIndex());
```

For output `[2,1]`, the old input coordinate is `[1,2]`, offset `1×3+2=5`. The view uses offset `2×1+1×3=5`, also `20` bytes from X's base for f32. Both loads produce `5`. XT is an addressing view, while the remaining store still describes writing O. The pass erases unused index transformations; subsequent canonicalization drops captures that are now dead. [Chain collection](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:42) [Hoist implementation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:88) [Dead-transform cleanup](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:144) [[test-view:23]]

**A store-related guard can stop only part of a chain.** In the independent `broadcast-atop-store-reshape` test, the loop shape is `[2,4]`, its output is flat `[8]`, and input C has shape `[1]`. A reshape index reaches the store, so it stays inside the body; the load-only broadcast above that reshape can still be hoisted. The test runs the same `map-livm` and opaque-cleanup pipeline. [RUN and expected partial hoist](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/LoopInvariantViewMotion/loop-invariant-view-motion-broadcast-atop-store-reshape.mlir:1)

```transform
BEFORE · schematic, C=[9], O has shape [8]
  body<[2,4]> index(i) captures(C,O) {
    r = map.index.reshape<[8]->[2,4]>(i)          // [row,col] → [4*row+col]
    b = map.index.broadcast<[1]->[8]>(r)         // [flat] → [0]
    v = map.load C[b]                            // always load 9
    map.store O[r] = v                           // store needs the reshape index
  }

AFTER
  CB = map.index.broadcast.stride<[1]->[8]>(C)    // one-element input exposed with stride 0
  body<[2,4]> index(i) captures(CB,O) {
    r = map.index.reshape<[8]->[2,4]>(i)          // retained because it feeds the store
    v = map.load CB[r]                           // hoisted view still supplies 9
    map.store O[r] = v                           // output positions 0…7 all become 9
  }
```

At loop index `[1,2]`, `r=6`; before rewriting `b=0`, afterward the view offset is `6×0=0`. Both store `9` at O[6]. `transitivelyFeedsStore` follows only index-transform users, so a load's value subsequently being stored does not itself block load-side hoisting. The final pass invariant allows surviving index transforms only if they feed stores. [Store-use analysis](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:55) [Hoistable prefix](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:76) [Final invariant](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/LoopInvariantViewMotion.cpp:171) [Expected retained reshape and store](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MAPDialect/Transforms/LoopInvariantViewMotion/loop-invariant-view-motion-broadcast-atop-store-reshape.mlir:23)

The production pipeline runs CSE before its final canonicalizer so two newly hoisted identical views can become one Value before duplicate captures are removed. This final ordering explains why view motion, CSE and capture cleanup are separate steps. P1's internal K transpose is outside this MAP access representation. [[gc:363]] [[map-fold:40]] [[kernel:346]]

</div>
</details>
