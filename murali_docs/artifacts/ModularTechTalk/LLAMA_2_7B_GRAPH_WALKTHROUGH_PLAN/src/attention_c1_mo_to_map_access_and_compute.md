## QnA 1 · What does MO→MAP change for fused attention?

**One `mo.custom` becomes one `map.iter.opaque` containing a `map.call.opaque`.** The surrounding `mo.graph` still has three tensor arguments and one tensor result. The new nested region makes the implementation's inputs and output destination explicit. [[mo-to-map:159]] [[mo-to-map:179]] [[mo-to-map:238]]

<span id="qna-mo-to-map-1-illustration-1"></span>

```transform
BEFORE · IR1 graph body
  %o = mo.custom(%q, %k, %v)
       symbol="modular_ops::fused_attention_custom"
       parameters={BD=8, BN=16}, device=gpu:0
  mo.output %o

AFTER · source-derived schematic, same external contract
  %o_map = map.iter.opaque
      inputs  (%q as %qi:T0, %k as %ki:T0, %v as %vi:T0)
      outputs (%oi:T0) {
    map.call.opaque @PKG.fused_attention.FusedAttention
      parameters={BD=8, BN=16}
      inputs(%qi, %ki, %vi) outputs(%oi)
      kernelName="modular_ops::fused_attention_custom"
  } device=gpu:0, dispatch=deferred
  mo.output %o_map

T0 = float32[32,32] on gpu:0
PKG = illustrative name for the imported P1 package
```

The actual fully qualified symbol comes from the imported declaration; `PKG` above is a readable stand-in. The pass rewires users to the new results and erases the old operation **in IR1**. This is replacement of native IR objects, not changing a Python `TensorValue` into a MAP wrapper. [[mo-to-map:241]] [[mo-to-map:255]] [[frontend:377]]

## QnA 2 · Which native calls construct this region?

**Example links** map the earlier Q&A illustrations and the worked examples to their native checks, rewrites and handoffs. Links marked “later” describe a subsequent stage; a listed rewrite still requires its matching conditions.

```calltree
gcPipelineImpl [[gc:578]]
├─ buildStage: schedule MarkStableTensors, then MOToMAP [[gc:578]] [[gc:580]] [[gc:581]]
│  └─ optional stage dump is scheduled after these passes [[gc:568]]
└─ when the pass manager runs IR1: MOToMAPPass::runOnOperation() [[mo-to-map:767]]
   ├─ MojoRegistry registry(IR1)                   ← resolves imported declarations [[mo-to-map:768]]
   ├─ walk IR1 before rewriting; save native Operation* in worklist [[mo-to-map:770]]
   │  ├─ MO tensor/buffer rank exceeds kMaxRank → diagnostic + interrupt [[mo-to-map:773]]
   │  └─ shouldLower(op) [[mo-to-map:63]]
   │     ├─ preserved MO op / InvokeShapeFunc / other dialect → leave for later [[mo-to-map:64]]
   │     └─ eligible MO operation → append to worklist [[mo-to-map:780]]
   └─ for each saved operation: lowerOp(op, registry) [[mo-to-map:787]]
      ├─ getMojoKernelAdaptor(registry, op) [[mo-to-map:706]] [[registry-support:67]]
      │  ├─ custom.symbol; otherwise operation name → registration key [[registry-support:43]]
      │  ├─ registration + device → resolved kernel declaration [[registry-support:76]]
      │  └─ lookup/adaptor error → return null success; leave this op [[mo-to-map:708]]
      ├─ MutableStore → lowerMutableStoreToForeach; chain result, buffer store [[mo-to-map:711]]
      │  └─ [↗ QnA 7 · Example 9 — store to buffer; return an ordering chain](#qna-mo-to-map-example-mutable-store-chain)
      ├─ reduction registration → lowerReductionToRowwise [[mo-to-map:718]] [[mo-to-map:646]]
      │  └─ [↗ QnA 7 · Example 8 — rowwise iterator and nested reduction fold](#qna-mo-to-map-example-rowwise-reduction)
      ├─ elementwise registration → lowerElementwiseToForeach [[mo-to-map:725]] [[mo-to-map:265]]
      │  ├─ [↗ QnA 5 · scalar Add example — A(1,2)=4 + B(1,2)=5 → C(1,2)=9](#qna-mo-to-map-5-illustration-1)
      │  └─ [↗ QnA 7 · Example 4 — result-shaped loop, scalar loads/call/store](#qna-mo-to-map-example-elementwise-add)
      └─ ordinary execute registration [[mo-to-map:731]]
         ├─ require executeFunction; map operands/results to declaration roles [[mo-to-map:736]]
         │  └─ bad mapping → diagnostic + failed pass [[mo-to-map:738]]
         ├─ Slice → lowerSliceToForeach: output index → input index [[mo-to-map:747]] [[mo-to-map:389]]
         │  └─ [↗ QnA 7 · Example 7 — normalized bounds map output indices to input](#qna-mo-to-map-example-slice-bounds)
         ├─ BroadcastTo → lowerBroadcastToForeach: project output index [[mo-to-map:749]] [[mo-to-map:486]]
         │  └─ [↗ QnA 7 · Example 6 — project broadcast indices; reshape preserves linear position](#qna-mo-to-map-example-broadcast-reshape-ranks)
         ├─ Reshape → lowerReshapeToForeach: preserve linear position [[mo-to-map:751]] [[mo-to-map:538]]
         │  └─ [↗ QnA 7 · Example 6 — reshape index rank changes, element order stays](#qna-mo-to-map-example-broadcast-reshape-ranks)
         ├─ Transpose → lowerTransposeToForeach: permute coordinates [[mo-to-map:753]] [[mo-to-map:589]]
         │  └─ [↗ QnA 7 · Example 5 — coordinate permutation and immediate dispatch](#qna-mo-to-map-example-transpose-index)
         └─ attention and other ordinary calls → lowerToOpaque [[mo-to-map:756]] [[mo-to-map:135]]
            ├─ inputFusible/outputFusible from mapped argument roles [[mo-to-map:137]]
            ├─ split trailing input chain from non-chain operands [[mo-to-map:152]]
            ├─ create iterator with original result types and inputs [[mo-to-map:159]]
            │  └─ [↗ QnA 8 · what the opaque iterator owns and how it exposes the tensor interface](#qna-mo-to-map-8-iter-opaque-call-opaque)
            ├─ attach device, fusion leaf, read/write effects, dispatch [[mo-to-map:171]]
            ├─ create one block: non-chain inputs, then DPS outputs [[mo-to-map:179]]
            │  ├─ [↗ QnA 3 · ownership example — three graph inputs, four local body arguments](#qna-mo-to-map-3-illustration-1)
            │  ├─ [↗ QnA 7 · Example 1 — Q/K/V plus the output destination](#qna-mo-to-map-example-attention-destinations)
            │  └─ !mo.opaque by-value result → no output block argument [[mo-to-map:189]]
            ├─ plain input → block argument; fused input → load prologue [[mo-to-map:201]] [[mo-to-map:210]]
            │  └─ [↗ QnA 7 · Example 3 — fused gather inputs receive load prologues](#qna-mo-to-map-example-gather-prologues)
            ├─ plain output → block argument; fused output → store epilogue [[mo-to-map:215]] [[mo-to-map:224]]
            │  └─ [↗ QnA 7 · Example 2 — ordinary output versus fused store epilogue](#qna-mo-to-map-example-fused-output-contract)
            ├─ inferParameters; infer opaque-type parameters when present [[mo-to-map:229]] [[mo-to-map:232]]
            ├─ create CallOpaqueOp with resolved symbol + parameters + I/O [[mo-to-map:238]]
            │  ├─ [↗ QnA 4 · metadata example — BD=8, BN=16, resolved symbol and device](#qna-mo-to-map-4-metadata-example)
            │  └─ [↗ QnA 8 · what the nested opaque call records and how it differs from the iterator](#qna-mo-to-map-8-iter-opaque-call-opaque)
            ├─ preserve equal-length variadic-group marker [[mo-to-map:248]]
            ├─ by-value opaque result only → map.yield [[mo-to-map:251]]
            └─ replace all old result uses; erase old op from IR1 [[mo-to-map:255]]
               ├─ [↗ QnA 1 · before/after attention — mo.custom → iterator and nested call](#qna-mo-to-map-1-illustration-1)
               └─ [↗ QnA 6 · handoff example — typed MAP contract; pointers/storage come later](#qna-mo-to-map-6-illustration-1)
```

This kernel is an ordinary `execute` registration, not an elementwise or reduction registration. It therefore follows `lowerToOpaque`, rather than the branches that create a scalar elementwise loop or a row reduction body. [[kernel:74]] [[mo-to-map:717]] [[mo-to-map:724]] [[mo-to-map:756]]

**SOURCE · scheduling and ownership.** `buildStage` adds pass objects; `runGCPipeline` later calls `pm.run(moduleOp)`. The worklist holds pointers into the compiler's IR1 module. The newly created iterator owns its region and block; that block owns its arguments and nested operations. The temporary registry/adaptor/mapping objects help select and describe a declaration. The new call retains a symbol reference and attributes in IR1. [Pipeline execution](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:654) [[mo-to-map:768]] [[mo-to-map:180]] [[mo-to-map:238]]

| Decision | Matched path | Skipped or rejected path |
| --- | --- | --- |
| Stable-tensor marking before MO→MAP | A device graph, a registered `build_graph` entry, and a stable-tensor role cause `mo.stable.create` to be inserted on that operand. | An ordinary graph, missing `build_graph`, or ordinary tensor role needs no marker. This does not add a fourth attention input. |
| Registered implementation | Imported declaration determines the elementwise/reduction/opaque branch. | An adaptor lookup error returns a successful null result from `lowerOp`; this pass leaves the MO op in place. |
| Fused I/O role | A tensor role marked fusible gets a load/store closure. | An ordinary tensor passes directly; an unsupported fused non-tensor role emits an error. |
| Dispatch | `stampAsyncLaunch` records `immediate` for operations selected by `kernelRequiresSyncLaunch`, including transpose. | Other operations, including this custom call, receive `deferred`; no operation executes while this field is set. |

The marking guards and mutation are in [MarkStableTensorsPass::runOnOperation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/MarkStableTensorsPass.cpp:47) and [StableCreateOp construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/MarkStableTensorsPass.cpp:128). The other branches are in `lowerOp`, `lowerToOpaque`, and the dispatch helper. [[mo-to-map:705]] [[mo-to-map:205]] [[mo-to-map:219]] [[mo-utils:27]]

The `FailureOr<Operation *>` return deserves care: a null **successful** value also follows successful elementwise/view/reduction replacement. The pass tests `failed(...)`, not whether the pointer is null. An opaque replacement returns the new iterator pointer. Thus a null result alone cannot tell a caller whether an operation changed. [[mo-to-map:714]] [[mo-to-map:721]] [[mo-to-map:728]] [[mo-to-map:761]] [[mo-to-map:257]] [[mo-to-map:788]]

## QnA 3 · Why are there now four body arguments? Did G1 gain an input?

**The four arguments belong to a new nested MAP body.** Three shadow inputs; the fourth denotes where this implementation writes the output. The outer graph's caller still supplies only Q, K and V. [[mo-to-map:179]] [[entry:61]]

<span id="qna-mo-to-map-3-illustration-1"></span>

```state
IR1 : builtin.module
└─ mo.graph @fused_attention
   └─ graph body block                  ← still THREE user inputs
      ├─ %q : T0
      ├─ %k : T0
      ├─ %v : T0
      └─ %o_map = map.iter.opaque
         └─ body region → ONE block     ← FOUR local arguments
            ├─ %qi : T0  ← Q input
            ├─ %ki : T0  ← K input
            ├─ %vi : T0  ← V input
            ├─ %oi : T0  ← output destination
            └─ map.call.opaque inputs(%qi,%ki,%vi) outputs(%oi)

Outer result %o_map and inner output destination %oi have different roles.
No fourth user tensor is added to G1.inputs.
```

This is **destination-passing style**: the implementation writes a supplied output destination. In this pass it is an IR argument; actual output storage is introduced by later runtime lowering. The nested call does not return a new tensor-valued SSA result for this case. [[mo-to-map:238]] [[mogg-ops:788]]

A `!mo.opaque` by-value result has a different branch: it has no destination block argument and uses `map.yield`. That branch does not apply to the tensor result here. [[mo-to-map:189]] [[mo-to-map:251]]

## QnA 4 · What metadata is copied, inferred or preserved?

<span id="qna-mo-to-map-4-metadata-example"></span>

| Field | Example value after lowering | Source |
| --- | --- | --- |
| Inputs / result types | three T0 inputs, one T0 result | [[mo-to-map:157]] |
| Device | `#M.device_ref<"gpu",0>` on the enclosing iterator | [[mo-to-map:171]] |
| Dispatch | `deferred` for this custom op | [[mo-to-map:127]] [[mo-utils:27]] |
| Provenance | a fusion leaf identifying the original custom operation | [[mo-to-map:172]] |
| Read/write flags | copied from the source operation's declared effects | [[mo-to-map:119]] [[mo-to-map:174]] |
| Parameter dictionary | `BD=8`, `BN=16` | [[params:63]] |
| Callee | fully resolved P1 declaration symbol | [[mo-to-map:241]] |
| Registered name | `modular_ops::fused_attention_custom` | [[mo-to-map:243]] [[entry:58]] |

The custom-op parameter normalizer leaves ordinary integer attributes as they are. It does **not** insert graph constants for BN/BD or add a runtime `rank` input. The eventual `target="gpu"` parameter is supplied during Mojo emission; dtype/rank can be inferred by Mojo from typed tensor arguments. [[params:32]] [[params:48]] [[emit-state:582]] [[params:328]]

The original custom call has no explicit input/output chain. `stampAsyncLaunch` still records how its lowered routine should be dispatched; runtime dependency handling is added later. [[entry:57]] [[mo-to-map:152]] [[mo-to-map:177]] [[mogg-ops:751]]

## QnA 5 · When does MAP expose scalar loads and stores? Give an example.

For an **elementwise** registered operation, the other MOToMAP branch exposes scalar computation indexed over the tensor. The original attention registration does not use that branch. [[mo-to-map:260]] [[mo-to-map:724]]

<span id="qna-mo-to-map-5-illustration-1"></span>

```transform
ILLUSTRATIVE elementwise operation, separate from attention
BEFORE: %c = mo.add(%a,%b) : float32[2,3]

AFTER: map.iter.elementwise over [2,3]
       └─ body argument %idx : !map.index<2>
          ├─ %av = map.load A[%idx]
          ├─ %bv = map.load B[%idx]
          ├─ %cv = registered scalar Add(%av,%bv)
          └─ map.store C[%idx] = %cv

At index [1,2]: A[1,2]=4, B[1,2]=5 → C[1,2]=9.
The loop describes work; creating the IR does not run those loads.
```

The implementation explicitly creates one load per tensor input, a scalar registered call and a store. These bodies give later fusion passes something they can safely combine. [[mo-to-map:260]]

For an opaque implementation, the declaration can instead request **FusedInputTensor** or **FusedOutputTensor**. MOToMAP then inserts load prologues or store epilogues. Their callbacks let compatible neighboring computations participate in fusion. The ordinary `InputTensor`/`OutputTensor` roles in this attention example select the direct-argument paths. [[mo-to-map:210]] [[mo-to-map:224]] [[kernel:82]]

**TEST · inspected expectation:** the repository's plain-output test expects `map.call.opaque` to write the raw output argument and explicitly rejects an epilogue. Its fused-output comparison expects a `map.iter.epilogue`. [[test-mo-map:41]] [[test-mo-map:20]]

## QnA 6 · What does the next stage receive?

<span id="qna-mo-to-map-6-illustration-1"></span>

```state
MAP optimization input, attention path
├─ one map.iter.opaque
│  ├─ inputs: Q, K, V
│  ├─ output destination: O
│  ├─ call: P1's FusedAttention declaration
│  └─ params: BN=16, BD=8
└─ outer mo.output consumes the iterator result

No scalar QK/softmax/PV body has been exposed to MAP.
No data pointers or GPU allocations exist merely from constructing these ops.
```

The enclosing module may contain several dialects at this boundary: `mo.graph` and its tensor types coexist with MAP operations and MOGG declaration metadata. MO→MAP is a transformation of selected operations, not a requirement that every `mo.*` object disappear at once. [[mo-to-map:780]] [[test-mo-map:41]] [[gc:585]]

## QnA 7 · How does MO→MAP rewrite concrete attention, fused-tensor, elementwise, view, reduction and mutable-store operations? For each example, show the input values and shapes, before/after IR, changed operands and blocks, native implementation, and the test or guard that establishes the result.

**SOURCE / TEST · nine worked examples.** The IR below is schematic: declarations, type repetitions and generated symbol prefixes are shortened. Native C++ excerpts retain their implementation statements. Numerical values are illustrative contract checks; the compiler constructs these operations without evaluating these arrays. The cited `.mlir` files supply independent test graphs to `mo-opt`. Their `RUN` commands identify the passes under test, and `CHECK` comments describe expected printed IR; none of these tests was executed for this reader.

<details class="qna-cell" id="qna-mo-to-map-example-attention-destinations" markdown="1">
<summary>Example 1 · How do three attention inputs become four local block arguments without changing the graph signature?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · actual attention route.** The accelerator configuration supplies Q, K and V with shape `[32,32]`, `float32`, and parameters `BD=8`, `BN=16`. Its registered `execute` signature requests three ordinary `InputTensor` values and one ordinary `OutputTensor`. That signature determines the absence of prologue/epilogue closures. [[entry:35]] [[entry:57]] [[kernel:74]] [[kernel:82]]

```transform
BEFORE · IR1, schematic
  graph inputs %q, %k, %v : f32[32,32] on gpu:0   // three caller-supplied tensors
  %o = mo.custom(%q,%k,%v)                       // one tensor-valued operation
       symbol="modular_ops::fused_attention_custom"
       parameters={BD=8, BN=16}, device=gpu:0
  mo.output %o                                  // one externally visible result

AFTER · same graph signature
  %r = map.iter.opaque inputs(%q,%k,%v)           // replacement op owns a region
       outputs(%oi : f32[32,32]) {               // one local output destination
    ^body(%qi,%ki,%vi,%oi):                      // all four are new block arguments
      map.call.opaque @PKG.FusedAttention         // selected declaration, same function
        [BD=8, BN=16] inputs(%qi,%ki,%vi)
        outputs(%oi)                            // writes into the local destination
  } device=gpu:0, dispatch=deferred
  mo.output %r                                  // old %o use rewired to %r
```

`lowerToOpaque` first creates the enclosing result-bearing operation, then its body block. This excerpt explains why an output destination is a block argument while the outer output is an SSA result:

```cpp
Block *body = builder.createBlock(&opaque.getBody());
SmallVector<BlockArgument> ins;
for (Value operand : nonChainOperands)
  ins.push_back(body->addArgument(operand.getType(), loc));
```

The following result-type loop appends ordinary output arguments; it skips chains and by-value `!mo.opaque` results. For attention it appends exactly one tensor argument, `%oi`. The input mapping selects `%qi/%ki/%vi` directly, and the output mapping selects `%oi` directly. [[mo-to-map:179]] [[mo-to-map:186]] [[mo-to-map:201]] [[mo-to-map:215]]

The final mutation is small but essential:

```cpp
op->replaceAllUsesWith(opaque.getResults());
op->erase();
```

This redirects `mo.output` and any other user before deleting the old native operation. It preserves the result type while changing the producer object. The original Python graph G1 remains outside this IR1 rewrite. [[mo-to-map:255]] [[frontend:377]]

| Object | Before | After |
| --- | --- | --- |
| Outer graph arguments | `%q,%k,%v` | Same three arguments |
| Computation result | `%o` from `mo.custom` | `%r` from `map.iter.opaque` |
| Nested body | None on this custom op | One region, one block, four arguments |
| Nested call result | Not applicable | No tensor SSA return; `%oi` is a destination |
| Parameters | `BD=8`, `BN=16` attributes | Same normalized integer attributes |

For an illustrative arithmetic check, take every Q entry as zero, finite K entries, and `V[j,d]=j` for `j=0…31`. The unscaled attention contract gives equal weights `1/32`, so every output feature is `(0+…+31)/32 = 15.5`. Both IR forms name the same implementation and input tensors; this stage neither evaluates the average nor expands softmax. The authored attention arithmetic remains in the registered implementation. [[kernel:90]] [[kernel:99]] [[mo-to-map:238]]

**Guard and comparison.** A registered elementwise custom kernel would take the earlier elementwise branch even though its MO operation is also named `mo.custom`. Conversely, no imported declaration means `getMojoKernelAdaptor` fails and `lowerOp` leaves the op untouched at this stage. **TEST:** `opaque-kernels.mlir` line 1 imports signature-only kernels and runs `--mo-to-map`; its `@one_plain` expectation forbids an epilogue. Line 2 omits the import, and the `DISABLED` checks require surviving `mo.custom`. Those tests validate the generic ordinary-output route and registry dependence; they do not run this attention kernel. [[mo-to-map:708]] [[mo-to-map:725]] [[test-mo-map:1]] [[test-mo-map:2]] [[test-mo-map:37]] [[test-mo-map:95]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-fused-output-contract" markdown="1">
<summary>Example 2 · Why does a fused output gain a store callback while an ordinary output remains a direct argument?</summary>
<div class="qna-answer" markdown="1">

**TEST · output-role comparison.** The supplied test input `@two_partial` is a custom operation with one `[2,2]` f32 input and two `[2,2]` f32 results. The selected signature marks only the first output as fused. Its `RUN` imports `%kernel_api_test_lib` before `--mo-to-map`; `CHECK` requires one epilogue, `CHECK-NOT` excludes a second, and the call receives `(epilogue, raw-output)` in result order. These are signature-only kernels, so the test establishes structure rather than a particular numerical function. [[test-mo-map:1]] [[test-mo-map:76]] [[test-mo-map:80]] [[test-mo-map:87]]

```transform
BEFORE · schematic test operation
  %u,%v = mo.custom @map_lower_partial_fused_outputs(%a) // two tensor results
  mo.output %u,%v                                      // expose both results

AFTER · both result destinations remain explicit
  %r0,%r1 = map.iter.opaque inputs(%a as %ai)           // one input block argument
    outputs(%o0:f32[2,2], %o1:f32[2,2]) {              // two destination arguments
    %e0 = map.iter.epilogue(%value:f32, %i:index<2>)    // receives one produced scalar
      capturing(%cap=%o0) {                           // local reference to output 0
        map.store %cap[%i] = %value                   // identity store callback
      }
    map.call.opaque @MapLowerPartialFusedOutputs      // same registered implementation
      inputs(%ai) outputs(%e0,%o1)                    // fused first, ordinary second
  }
  mo.output %r0,%r1                                  // same external two-result contract
```

The native decision reads the declaration mapping, not the operation name or tensor size:

```cpp
for (const auto &[idx, arg] : mapping.resultMapping)
  outputFusible[idx] = arg.isFusedTensorType();
```

The output loop passes an ordinary output straight through. For a fusible tensor output it instead pushes `buildStoreEpilogue(...).getResult()` into `callOutputs`. The epilogue builder creates block arguments in the order **scalar value, index, captured destination**, then inserts the store. [[mo-to-map:141]] [[mo-to-map:215]] [[mo-to-map:224]] [[mo-to-map:75]]

```cpp
BlockArgument val = body->addArgument(outType.getElementType(), loc);
BlockArgument idx = body->addArgument(indexType, loc);
BlockArgument cap = body->addArgument(out.getType(), loc);
builder.setInsertionPointToStart(body);
MAP::StoreOp::create(builder, loc, /*tensor=*/cap, /*index=*/idx,
                     /*value=*/val);
```

For a concrete callback invocation, if the implementation supplies `value=7` at index `[1,0]`, the new closure records `O0[1,0]=7`. If it writes `11` into ordinary destination `O1[1,0]`, that remains `11`. The rewrite adds no arithmetic to either result. Output 0 gains a place where a later pass can insert compatible arithmetic before storing. [[mo-to-map:89]] [[test-mo-map:84]]

**Skipped case:** `@one_plain` has the same `[2,2]` shape but `OutputTensor` semantics, so no epilogue is created. **Rejected case:** if a mapped fused output is not a tensor, `lowerToOpaque` emits “output fusion is only supported for tensor results.” At creation time `intermediateTensor` is null; the later accumulation pass assigns its storage source after fusion. These details keep an epilogue closure distinct from an allocated buffer. [[test-mo-map:37]] [[mo-to-map:219]] [[mo-to-map:80]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-gather-prologues" markdown="1">
<summary>Example 3 · How do fused input roles turn gather's data and indices into load callbacks?</summary>
<div class="qna-answer" markdown="1">

**TEST · an opaque gather with exposed access interfaces.** `opaque-kernels-prologue.mlir` starts from `mo.gather`, imports the test kernel package, then runs `--mo-to-map`. `@gather` has data `[3,3]` f32, indices `[2,2]` si32, axis `0`, and output `[2,2,3]` f32. Its checks require two input prologues, a rank-3 output epilogue, and `axis=0` on the registered call. [RUN and expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels-prologue.mlir:1) [Test graph](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels-prologue.mlir:18)

```text
Illustrative data D = [[0,1,2], [3,4,5], [6,7,8]]   // three source rows
Illustrative indices I = [[2,0], [1,2]]             // four requested row numbers
Expected O = [[[6,7,8], [0,1,2]],                   // I[0,0] and I[0,1]
              [[3,4,5], [6,7,8]]]                  // I[1,0] and I[1,1]
```

```transform
BEFORE · schematic
  %o = mo.gather(%data,%indices) axis=0             // whole-tensor gather operation
  mo.output %o                                    // f32[2,2,3]

AFTER
  %r = map.iter.opaque inputs(%data,%indices)       // outer data dependencies
    outputs(%out:f32[2,2,3]) {                     // local result destination
    %pd = prologue(%di:index<2>) captures(%data) {  // data load interface
      %d = map.load %data[%di]                     // one requested data element
      map.yield %d                                // return scalar to gather
    }
    %pi = prologue(%ii:index<2>) captures(%indices) {// indices load interface
      %n = map.load %indices[%ii]                  // one requested integer index
      map.yield %n                                // return row number to gather
    }
    %eo = epilogue(%v:f32,%oi:index<3>) captures(%out) {
      map.store %out[%oi] = %v                     // retain gather's output coordinate
    }
    map.call.opaque @Gather[axis=0]                // gather's control remains opaque
      inputs(%pd,%pi) outputs(%eo)                 // supply three callbacks
  }
  mo.output %r                                    // result shape is unchanged
```

`buildLoadPrologue` constructs a fresh region whose arguments are **requested index, captured tensor**. The defining statements for its computation are:

```cpp
Value loaded = MAP::LoadOp::create(builder, loc, inType.getElementType(),
                                   /*tensor=*/cap, /*index=*/idx);
MAP::YieldOp::create(builder, loc, loaded);
```

The enclosing `lowerToOpaque` input loop calls this builder only for fused tensor roles. It preserves each input's own rank and scalar element type: the data callback yields f32, the indices callback yields si32. The output callback separately uses rank 3. At output coordinate `[0,0,2]`, the indices callback supplies `I[0,0]=2`, the data callback supplies `D[2,2]=8`, and the output callback stores `8`. These are illustrative callback values, not operations run by the pass. [[mo-to-map:98]] [[mo-to-map:108]] [[mo-to-map:111]] [[mo-to-map:210]]

## QnA 8 · What are `map.iter.opaque` and `map.call.opaque`, what do they do, and how do they support fusion? {#qna-mo-to-map-8-iter-opaque-call-opaque}

**`map.iter.opaque` describes a kernel’s enclosing region and tensor interface. `map.call.opaque` identifies the Mojo implementation invoked inside that region.** Together, they let the compiler work with the kernel’s inputs, outputs and surrounding access paths while retaining its handwritten computation in Mojo.

| Operation | Main responsibility |
|---|---|
| `map.iter.opaque` | Own the body region, expose tensor inputs/results, and carry device, dispatch and memory-effect metadata |
| `map.call.opaque` | Record the selected kernel symbol, compile-time parameters, inputs and output destinations |

These roles are defined in the [iterator definition](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPOps.td:50) and [call definition](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPOps.td:263).

**Start with your fused-attention graph.** Before MO→MAP, the relevant operation is schematically:

```transform
%o = mo.custom(%q, %k, %v)
     symbol     = "modular_ops::fused_attention_custom"
     parameters = {BD=8, BN=16}
     device     = gpu:0

mo.output %o

%q, %k, %v, %o : float32[32,32] on gpu:0
```

This says: apply the registered custom operation to Q, K and V, producing one tensor. MO→MAP replaces that operation with an enclosing iterator and a nested call:

```transform
mo.graph @fused_attention
│
├─ graph input arguments
│  ├─ %q : T0
│  ├─ %k : T0
│  └─ %v : T0
│
└─ graph body — relevant operations
   │
   ├─ %o_map = map.iter.opaque
   │  ├─ inputs: %q, %k, %v
   │  ├─ result: %o_map : T0
   │  ├─ device: gpu:0
   │  └─ body region → one block
   │     │
   │     ├─ %qi : T0 ← local Q input
   │     ├─ %ki : T0 ← local K input
   │     ├─ %vi : T0 ← local V input
   │     ├─ %oi : T0 ← local output destination
   │     │
   │     └─ map.call.opaque
   │        ├─ callee: @PKG.fused_attention.FusedAttention
   │        ├─ parameters: {BD=8, BN=16}
   │        ├─ inputs: %qi, %ki, %vi
   │        └─ outputs: %oi
   │
   └─ mo.output %o_map

T0 = float32[32,32] on gpu:0
PKG = illustrative prefix for the imported P1 package
```

The pass constructs these native IR objects, redirects users of the original result to `%o_map`, and erases the original `mo.custom` from IR1. See [lowerToOpaque()](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:135).

**Why introduce `%qi`, `%ki` and `%vi`?** The iterator’s region has its own local arguments. Each corresponds to an incoming tensor:

```transform
Outside the region             Inside the region

%q ──────────────────────────► %qi
%k ──────────────────────────► %ki
%v ──────────────────────────► %vi

Three incoming tensors         Three local names for those inputs
```

These are different SSA values in different scopes. Introducing local arguments does not copy the tensor contents or add new user inputs. The constructor creates block arguments with the corresponding input types. See [body construction](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:179).

**The fourth local argument, `%oi`, describes where the kernel writes its result.** This is *destination-passing style*.

Your Mojo implementation accepts an output tensor alongside its three input tensors:

```transform
Mojo execute interface, schematically:

execute(
    output,
    query,
    key,
    value,
    context
)
```

MAP records the corresponding groups as:

```transform
map.call.opaque
    inputs (%qi, %ki, %vi)
    outputs(%oi)

Mapping to Mojo parameters:
    %qi → query
    %ki → key
    %vi → value
    %oi → output
```

The actual signature is at [FusedAttention.execute](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/examples/custom_ops/kernels/fused_attention.mojo:74).

| Value | Meaning |
|---|---|
| `%q`, `%k`, `%v` | Inputs available in the surrounding graph |
| `%qi`, `%ki`, `%vi` | Corresponding inputs available inside the iterator |
| `%oi` | Destination argument the nested call will write |
| `%o_map` | Result exposed by the enclosing iterator to subsequent graph operations |

For this tensor-result case, the nested call writes `%oi`; it does not produce another tensor-valued SSA result. The enclosing iterator exposes `%o_map`. At MAP construction time, `%oi` is a typed IR argument describing the destination—the construction shown above creates IR, rather than allocating and filling a GPU array.

**The nested call also preserves the selected implementation.** Its fields answer concrete compiler questions:

| Field | Attention example | Meaning |
|---|---|---|
| `callee` | `@PKG.fused_attention.FusedAttention` | Which imported kernel declaration implements the operation? |
| `kernelName` | `"modular_ops::fused_attention_custom"` | Under which operation name was it registered? |
| `parameters` | `{BD=8, BN=16}` | Which compile-time choices apply to this call? |
| `inputs` | `%qi, %ki, %vi` | Which input values does it consume? |
| `outputs` | `%oi` | Where does it write its tensor result? |

The registration name and qualified callee have different purposes: multiple implementation structs can register under one operation name. MAP therefore preserves both. See the [call’s attributes and description](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPOps.td:269).

**“Opaque” describes the computation boundary.** MAP retains the kernel’s declared interface while the internal attention algorithm remains in its Mojo implementation:

```transform
Visible to MAP                     Implemented inside Mojo
┌──────────────────────────┐       ┌──────────────────────────┐
│ Inputs: Q, K, V          │       │ Attention algorithm      │
│ Output destination: O    │──────►│ Internal loops and tiles │
│ Parameters: BD, BN       │       │ CPU/GPU implementation   │
│ Selected kernel symbol   │       │ Writes output O          │
└──────────────────────────┘       └──────────────────────────┘
```

Q/K/V/O still have tensor types. The name `iter` denotes this kernel-region form; the handwritten kernel supplies its internal iteration. For ordinary registered execution kernels, MO→MAP selects this opaque form, while other branches expose elementwise or rowwise computation. See [the lowering branches](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:705).

**A major benefit of the enclosing region is that eligible kernels can expose input-load and output-store callbacks.** MAP calls these a *prologue* and an *epilogue*:

```transform
map.iter.opaque
│
├─ prologue: describe how an input element is obtained
│
├─ map.call.opaque: retain the handwritten kernel computation
│
└─ epilogue: describe how an output element is transformed and stored
```

A prologue receives an index and yields an input value. An epilogue receives a computed value and index, then transforms or stores it. These interfaces are defined in the [callback operations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MAPDialect/Dialect/MAPOps.td:180).

**For an illustrative output-fusion example, consider matmul followed by bias addition.** Assume the selected kernel supports the required output callback and the fusion conditions permit the transformation.

```transform
A = [[1,2],           B = [[5,6],
     [3,4]]                [7,8]]

bias = [10,20]

R = matmul(A,B)       # row-by-column products
  = [[19,22],
     [43,50]]

Y = R + bias          # add bias[n] to each element in column n
  = [[29,42],
     [53,70]]
```

A separate output path could materialize and reread R:

```transform
Matmul computes 22 at index (0,1)
                │
                ▼
Store R[0,1] = 22
                │
                ▼
Add operation reads R[0,1] and bias[1]
                │
                ▼
Store Y[0,1] = 22 + 20 = 42
```

An eligible fused epilogue instead receives the computed value directly:

```transform
Matmul computes 22 at index (0,1)
                │
                ▼
Epilogue(value=22, index=(0,1))
├─ load bias[1] → 20
├─ add → 42
└─ store Y[0,1] = 42
```

The surrounding store path changes while the handwritten matmul computation remains in its implementation. This can avoid the separate intermediate store/read on that fused edge. The source checks conditions such as an existing epilogue and a suitable consumer before performing the rewrite. See [fusion checks](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:125) and [the rewrite](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Transforms/OutputFusion.cpp:249).

**An input prologue provides the corresponding capability on the read side.** For an illustrative kernel supporting a fused input callback:

```transform
Original expression:
    shifted_A = A + 1
    R = matmul(shifted_A, B)

Example A element:
    A[0,1] = 2
    shifted_A[0,1] = 3
```

The eligible input path can supply that transformed value when requested:

```transform
Kernel requests input at index (0,1)
                │
                ▼
Prologue(index=(0,1))
├─ load A[0,1] → 2
├─ add 1 → 3
└─ yield 3 to the kernel
```

This illustrates what a supported callback enables; it does not establish that every matmul implementation accepts such fusion. The MO→MAP constructor consults the mapped kernel argument roles before creating load/store callbacks. See [fusible argument handling](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp:137).

**Your current attention call uses plain tensor arguments and a plain output destination.** Its nested body therefore contains the direct call shown earlier. The prologue/epilogue examples explain capabilities of the representation, rather than transformations automatically applied to this attention kernel.

During the following compiler stage, the two levels lower together:

```transform
MAP                                  MOGG
map.iter.opaque ────────────────────► mogg.routine
└─ map.call.opaque ─────────────────► └─ mogg.call.opaque
```

The routine body introduces pointer arguments and reconstructs typed tensor descriptions. Call lowering also supplies the device context when the selected declaration requires one—as `FusedAttention.execute` does. These are further compiler transformations toward the eventual Mojo invocation; attention values are computed later during execution. See [routine lowering](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:907) and [call/context lowering](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:645).

**Matched and skipped inputs can coexist.** In `@partial_fused`, only input 0 has a fused role. The test expects one prologue and passes input 1 directly to the call. A plain attention input takes that same direct branch. A fused non-tensor input is rejected before a load closure is constructed. [Partial-input test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/opaque-kernels-prologue.mlir:39) [[mo-to-map:201]] [[mo-to-map:205]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-elementwise-add" markdown="1">
<summary>Example 4 · How does tensor addition become one scalar call per tensor index?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · different registration branch from attention.** `adaptor.isElementwise()` sends the operation to `lowerElementwiseToForeach`. The builder requires exactly one tensor result, takes the loop shape from that result type, and creates arguments `[inputs…, output, index]`. It does not wrap this scalar computation in `map.call.opaque`. [[mo-to-map:724]] [[mo-to-map:267]] [[mo-to-map:279]] [[mo-to-map:293]]

```transform
Inputs A=[[1,2,3],[4,5,6]], B=[[10,20,30],[40,50,60]]  // both f32[2,3]

BEFORE · schematic
  %c = mo.add(%a,%b) : f32[2,3]                     // elementwise tensor contract
  mo.output %c                                     // one tensor result

AFTER
  %r = map.iter.elementwise<[2,3]>                  // six logical output coordinates
    inputs(%a as %ai, %b as %bi) outputs(%ci)        // two inputs plus one destination
    index(%i:index<2>) {                           // one coordinate tuple per element
      %av = map.load %ai[%i]                       // A[i]
      %bv = map.load %bi[%i]                       // B[i]
      %cv = map.call.elementwise @Add(%i,%av,%bv)   // scalar Add, returning f32
      map.store %ci[%i] = %cv                      // C[i]
    }
  mo.output %r                                     // [[11,22,33],[44,55,66]]
```

The native loop inserts one `map.load` for each MO operand. The call consumes those loaded scalar values and the index, then the store writes the one output argument:

```cpp
Value scalar = MAP::CallElementwiseOp::create(
    bodyBuilder, loc, elemTy, kernelSymbol, params, idxArg, scalarLoads);

BlockArgument outputArg =
    body->getArgument(static_cast<unsigned>(op->getNumOperands()));
MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, scalar);
```

At index `[1,2]`, the values are `6+60=66`. The operation graph changes from one MO operation to one iterator with four body arguments and four body operations. No memory address is fixed at this point. The source uses each input tensor's element type for its load and the result tensor's element type for the scalar result, so dtype-changing elementwise registrations can use the same builder. [[mo-to-map:303]] [[mo-to-map:317]]

**TEST · direct entry at MO.** `elementwise-add-static.mlir` supplies `[8,16,128]` operands, imports built-ins, and runs `--mo-to-map`. Lines 18–22 require the result-shaped iterator, two loads at the same captured index, `.Add`, and a store. `CHECK-NOT` excludes an opaque iterator, and the end-of-line check confirms the default deferred dispatch is omitted from printing. The `[2,3]` arrays above are a smaller illustration of that same lowering. [[test-elementwise:1]] [[test-elementwise:6]] [[test-elementwise:16]]

**Guard:** zero or multiple results, or a non-tensor result, cause explicit diagnostics in this elementwise builder. Broadcasting is represented through the separate broadcast view branch below; this example's two operands already have equal shapes. Attention's ordinary `execute` declaration does not match `isElementwise()`. [[mo-to-map:267]] [[mo-to-map:271]] [[mo-to-map:749]] [[kernel:74]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-transpose-index" markdown="1">
<summary>Example 5 · How does a transpose expose its index mapping and retain its launch policy?</summary>
<div class="qna-answer" markdown="1">

**TEST · standalone transpose, already supplied as MO.** `transpose-view.mlir` imports built-ins and runs `--mo-to-map`; its `@transposeSwap` input is exactly `[2,3] → [3,2]` with permutation `[1,0]`. `CHECK` expects a result-shaped elementwise iterator, `dispatch(immediate)`, an index transpose, a load at the transformed index and a store at the original index. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/transpose-view.mlir:1) [Input and expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/transpose-view.mlir:11)

```transform
Input X=[[0,1,2],[3,4,5]]                           // row-major f32[2,3]

BEFORE · schematic
  %perm = mo.constant [1,0] : si64[2]              // output-axis to input-axis mapping
  %t = mo.transpose(%x,%perm) : f32[3,2]           // tensor-level transpose
  mo.output %t                                    // [[0,3],[1,4],[2,5]]

AFTER
  %perm = mo.constant [1,0] : si64[2]              // remains an explicit input value
  %r = map.iter.elementwise<[3,2]> dispatch(immediate)
    inputs(%x as %xi,%perm as %pi) outputs(%o)      // two input arguments, one destination
    index(%i:index<2>) {                           // coordinate in the output
      %j = map.index.transpose<[2,3]->[3,2],perm=[1,0]>(%i,%pi)
                                                   // [i0,i1] maps to [i1,i0]
      %v = map.load %xi[%j]                        // load original input coordinate
      map.store %o[%i] = %v                        // write corresponding output coordinate
    }
  mo.output %r                                    // same elements, transposed shape
```

The implementation creates the index op and wires its result only to the input load:

```cpp
Value innerIdx = MAP::TransposeIndexOp::create(
    bodyBuilder, loc, MAP::IndexType::get(ctx, rank), spec, idxArg, permArg);
Value loaded = MAP::LoadOp::create(
    bodyBuilder, loc, outputTy.getElementType(), inputArg, innerIdx);
MAP::StoreOp::create(bodyBuilder, loc, outputArg, idxArg, loaded);
```

At output `[2,1]`, input `[1,2]` has row-major offset `1×3+2=5` elements, or `20` bytes from the input base for f32. The new output has its own logical offset `2×2+1=5`; equal offsets here are incidental to this coordinate. At output `[1,0]`, input `[0,1]` has offset `1`, while the output offset is `2`. This is an index rewrite; later passes decide whether a strided view can replace part of the representation. [[mo-to-map:633]]

**Static versus dynamic branch.** A constant permutation is copied into `TransposeSpecAttr`; otherwise the spec contains `?` entries and the permutation tensor remains the source of the runtime values. The test's `@transposeDynamicPerm` checks that dynamic permutation still takes the view-index route. A rank-changing transpose is rejected. `stampAsyncLaunch` selects immediate dispatch because the source operation is `MO::TransposeOp`. [Dynamic test](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/transpose-view.mlir:63) [[mo-to-map:598]] [[mo-to-map:602]] [[mo-to-map:619]] [[mo-utils:27]]

This separate graph transpose is not the Kᵀ calculation inside P1's fused-attention implementation. The actual attention graph has no `mo.transpose` edge for this pass to inspect. [[entry:57]] [[kernel:346]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-broadcast-reshape-ranks" markdown="1">
<summary>Example 6 · How do broadcast and reshape change index rank while preserving their tensor contracts?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · both loops visit output coordinates.** `lowerBroadcastToForeach` and `lowerReshapeToForeach` create a loop index with the **output rank**, then an index-transform result with the **input rank**. The view signature stores both shapes. Broadcast projects repeated coordinates; reshape preserves a row-major linear position. [[mo-to-map:486]] [[mo-to-map:518]] [[mo-to-map:525]] [[mo-to-map:538]] [[mo-to-map:575]]

```transform
BROADCAST · schematic, B=[10,20,30] : f32[3]
  BEFORE %o = mo.static.broadcast_to(%b) : f32[2,3] // repeat the input vector twice
  AFTER  %o = map.iter.elementwise<[2,3]>           // six output positions
           inputs(%b) outputs(%out) index(%i:index<2>) {
             %j = map.index.broadcast<[3]->[2,3]>(%i) // [row,col] → [col]
             %v = map.load %b[%j]                     // rank-1 input access
             map.store %out[%i] = %v                  // rank-2 output access
           }
  OUTPUT [[10,20,30],[10,20,30]]                    // B[2]=30 supplies both rows

RESHAPE · schematic, X=[[0,1,2],[3,4,5]] : f32[2,3]
  BEFORE %o = mo.static.reshape(%x) : f32[6]        // flatten while preserving order
  AFTER  %o = map.iter.elementwise<[6]>             // six rank-1 output positions
           inputs(%x) outputs(%out) index(%i:index<1>) {
             %j = map.index.reshape<[2,3]->[6]>(%i) // [k] → [k/3,k%3]
             %v = map.load %x[%j]                   // rank-2 input access
             map.store %out[%i] = %v                // rank-1 output access
           }
  OUTPUT [0,1,2,3,4,5]                            // position 4 reads X[1,1]=4
```

These are the two native index-construction statements:

```cpp
Value innerIdx = MAP::BroadcastIndexOp::create(
    bodyBuilder, loc, MAP::IndexType::get(ctx, inRank), spec, idxArg);
```

```cpp
Value innerIdx = MAP::ReshapeIndexOp::create(
    bodyBuilder, loc, MAP::IndexType::get(ctx, inRank), spec, idxArg);
```

Each is followed by a load at `innerIdx` and a store at `idxArg`. Each iterator has one tensor input, one destination and one index argument; it does not acquire runtime dimension tensors merely because the two ranks differ. In the reshape example, index `[4]` and index `[1,1]` both refer to linear position `4`, or `16` bytes from a row-major f32 base. [[mo-to-map:513]] [[mo-to-map:525]] [[mo-to-map:563]] [[mo-to-map:575]]

**TEST · two independent MO inputs.** `broadcast-view.mlir` runs import + `--mo-to-map` on `[64] → [8,64]`; lines 28–39 check a rank-2 loop and rank-1 input index. `reshape-view.mlir` runs the same passes on `[4,6] → [24]`; lines 10–21 check a rank-1 loop and a reshape index used by the input load. These tests enter before MO→MAP and stop before later view fusion or contiguous-view raising. [Broadcast RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/broadcast-view.mlir:1) [Broadcast expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/broadcast-view.mlir:28) [Reshape RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/reshape-view.mlir:1) [Reshape expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/reshape-view.mlir:10)

**Guard:** the broadcast builder rejects input rank greater than output rank. Reshape permits rank changes in either direction and assumes a valid reshape contract; it does not infer `-1` here. A rank-0 scalar `9` broadcast to `[4,7]` therefore denotes 28 values equal to 9: every rank-2 output coordinate projects to the scalar's empty input index. That is illustrative broadcast arithmetic, not a captured test dump. [[mo-to-map:495]] [[mo-to-map:536]] [[mo-to-map:525]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-slice-bounds" markdown="1">
<summary>Example 7 · How are slice bounds normalized and connected to the generated load?</summary>
<div class="qna-answer" markdown="1">

**TEST · reverse slice with known bounds.** `slice-view.mlir` imports built-ins and runs `--mo-to-map`. `@sliceReverseStep` supplies an input `[10,64]`, start `[-1,32]`, stop `[-7,64]`, step `[-2,1]`, and output `[3,32]`. Its expected MAP spec contains normalized starts `[9,32]`, stops `[3,64]`, and unchanged steps `[-2,1]`. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/slice-view.mlir:1) [Reverse-slice input and CHECK](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/slice-view.mlir:70)

```transform
BEFORE · schematic, X[r,c]=100*r+c on f32[10,64]
  %start = mo.constant [-1,32] : si64[2]           // Python-style negative first bound
  %stop  = mo.constant [-7,64] : si64[2]           // exclusive stop
  %step  = mo.constant [-2,1] : si64[2]            // reverse rows, forward columns
  %o = mo.slice(%x,%start,%stop,%step) : f32[3,32]  // rows 9,7,5; columns 32…63
  mo.output %o                                    // retain selected elements

AFTER
  %o = map.iter.elementwise<[3,32]>                // iterate 96 output coordinates
    inputs(%x,%start,%stop,%step) outputs(%out) index(%i:index<2>) {
      %j = map.index.slice<starts=[9,32],stops=[3,64],steps=[-2,1]>
        (%i,%start,%stop,%step)                    // [r,c] → [9-2*r,32+c]
      %v = map.load %x[%j]                         // selected original element
      map.store %out[%i] = %v                      // compact output coordinate
    }
  mo.output %o                                    // same selected data
```

The normalization uses the static input dimension and the sign of the static step:

```cpp
int64_t lo = steps[i] > 0 ? 0 : -1;
int64_t hi = steps[i] > 0 ? dim : dim - 1;
auto normalize = [&](int64_t v) {
  if (ShapedType::isDynamic(v))
    return v;
  return std::clamp<int64_t>(v < 0 ? v + dim : v, lo, hi);
};
starts[i] = normalize(starts[i]);
stops[i] = normalize(stops[i]);
```

For the first dimension, `dim=10`: `-1+10=9`, `-7+10=3`; the reverse walk visits `9,7,5` and stops before `3`. At output `[2,0]`, the input coordinate is `[5,32]`, the illustrative value is `532`, and the input offset is `5×64+32=352` elements (`1408` bytes for f32). The four input operands remain; normalized literals enrich the spec rather than replacing those SSA operands. [[mo-to-map:410]] [[mo-to-map:424]] [[mo-to-map:457]] [[mo-to-map:471]]

**Matched and skipped normalization.** `boundsOrDynamic` treats each complete start/stop/step tensor independently. A graph-argument step yields `steps=[?,?]` even when starts/stops are constants; when the step or dimension is dynamic, the normalization loop skips that dimension. The test `@sliceNonConstantStep` checks this mixed case. Rank changes are rejected, and `SliceSpecAttr::verify` runs before the spec is constructed. [Mixed-bound CHECK](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/slice-view.mlir:49) [[mo-to-map:398]] [[mo-to-map:402]] [[mo-to-map:418]] [[mo-to-map:435]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-rowwise-reduction" markdown="1">
<summary>Example 8 · How does a reduction registration become a rowwise iterator with a nested fold?</summary>
<div class="qna-answer" markdown="1">

**SOURCE · monoid registration selects this branch.** `lowerOp` tests `isReduction()` before `isElementwise()` or the ordinary `execute` path. `lowerReductionToRowwise` derives the reduction axis from the input and result shapes, creates one rowwise iterator, then a nested `map.iter.reduce` that names the registered monoid. [[mo-to-map:717]] [[mo-to-map:646]] [[mo-to-map:662]] [[mo-to-map:682]]

```transform
Input X=[[1,2,3],[4,5,6]] : f32[2,3]               // two rows to reduce

BEFORE · schematic
  %s = mo.reduce.add(%x) : f32[2,3] -> f32[2,1]    // axis 1 shrinks to size 1
  mo.output %s                                    // [[6],[15]]

AFTER
  %s = map.iter.rowwise<[2,3], axis=1>             // reduction domain stays [2,3]
    inputs(%x as %xi) outputs(%out:f32[2,1])
    row(%row:index<2>) {                           // identifies the destination row
      %sum = map.iter.reduce @ReduceSum            // registered reduction semantics
        index(%i:index<2>) -> f32 {                // index ranges over this row's inputs
          %v = map.load %xi[%i]                    // one input element
          map.yield %v                            // feed the reduction monoid
        }
      map.store %out[%row] = %sum                  // one scalar result per row
    }
  mo.output %s                                    // same rank-preserving result
```

The fold body has one index argument and yields a loaded input element. The surrounding body holds the input, output destination(s), and row index. Scalar output types come from the element types of the original tensor results:

```cpp
SmallVector<Type> foldTypes;
for (Type resultTy : op->getResultTypes())
  foldTypes.push_back(cast<MO::TensorType>(resultTy).getElementType());
```

After constructing the reduction op, the builder pairs each destination with its corresponding returned field:

```cpp
for (auto [outputArg, field] :
     llvm::zip_equal(outputArgs, reduceOp.getResults()))
  MAP::StoreOp::create(bodyBuilder, loc, outputArg, rowArg, field);
```

For the illustrative first row, the sum is `1+2+3=6`; for the second it is `15`. The builder records the monoid symbol rather than spelling a sequential addition tree, so this representation does not promise a particular floating-point reduction order. [[mo-to-map:677]] [[mo-to-map:685]] [[mo-to-map:695]]

**TEST · direct MO reduction input.** `reductions.mlir` imports `%kernel_api_test_lib` and runs `--mo-to-map`. `@sumInnerAxis` supplies `[8,128] → [8,1]`; its consecutive `CHECK-NEXT` lines require `axis=1`, a nested `ReduceSum`, load/yield, and one row store. The same file's `@argMax` supplies two tensor results and expects fold fields `(bf16,si64)` with two stores, preserving value and index outputs separately. [RUN and sum expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/reductions.mlir:1) [ArgMax expectation](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/reductions.mlir:44)

**Guard:** an invalid axis inferred from the type contract returns an error. An ordinary opaque registration does not enter this branch merely because its authored implementation performs reductions. P1's attention softmax therefore remains inside P1. [[mo-to-map:648]] [[mo-to-map:718]] [[kernel:74]]

</div>
</details>

<details class="qna-cell" id="qna-mo-to-map-example-mutable-store-chain" markdown="1">
<summary>Example 9 · Why does mutable store produce a chain instead of an output tensor destination?</summary>
<div class="qna-answer" markdown="1">

**TEST · mutation has a different contract.** `elementwise-kernels.mlir` imports the test kernel package and runs only `--mo-to-map`. `@mutable_store` supplies a buffer `[2,2]`, two tensors `[2,2]`, a chain creator, an add and a store. Its checks require two separate elementwise iterators; the store iterator has `outputs ()`, a chain result, an identity `MutableStore` scalar call, a buffer store and `hasWriteEffect=true`. [RUN](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-kernels.mlir:1) [Input and expectations](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-kernels.mlir:113)

```transform
Inputs A=[[1,2],[3,4]], B=[[10,20],[30,40]]        // both f32[2,2]
Existing BUF has shape [2,2]                     // destination already belongs to caller

BEFORE · schematic
  %c0 = mo.chain.create()                        // order token, not tensor data
  %sum = mo.add(%a,%b)                           // [[11,22],[33,44]]
  %c1 = mo.mutable.store(%buf,%sum,%c0)           // update BUF; result is !mo.chain
  mo.output                                     // no tensor returned by this graph

AFTER
  %sum = map.iter.elementwise<[2,2]> {           // independent Add iterator
    %av = map.load %a[%i]                        // load A at iterator index
    %bv = map.load %b[%i]                        // load B at iterator index
    %v = map.call.elementwise @Add(%i,%av,%bv)  // sum one pair
    map.store %sum_destination[%i] = %v         // produce temporary sum tensor
  }
  %c1 = map.iter.elementwise<[2,2]>              // mutation iterator, schematic header
    inputs(%sum,%buf) outputs() index(%j:index<2>) chain(%c0) -> !mo.chain {
      %v = map.load %sum[%j]                     // read previously computed sum
      %same = map.call.elementwise @MutableStore(%j,%v) // scalar identity
      map.store %buf[%j] = %same                 // write into supplied mutable buffer
    } hasWriteEffect=true
  mo.output                                     // mutation is retained through its effects
```

`lowerOp` recognizes `MO::MutableStoreOp` before generic elementwise lowering. The ordinary elementwise builder requires a tensor result, which this operation does not have. The specialized builder instead selects the buffer's tensor shape and explicitly constructs a chain-result iterator:

```cpp
/*resultTypes=*/SmallVector<Type>{MO::ChainType::get(ctx)},
/*shape=*/shapeAttr,
/*inputs=*/ValueRange{op.getInTensor(), op.getInBuffer()},
/*chain=*/op.getInChain(),
```

These are arguments to `IterElementwiseOp::create`. The body has only `[tensor value, buffer, index]`; it receives neither an output-tensor argument nor a chain argument. The store targets `bufferArg`, while users of the old chain result are redirected to the new iterator result. [[mo-to-map:711]] [[mo-to-map:340]] [[mo-to-map:347]] [[mo-to-map:359]] [[mo-to-map:379]]

The illustrative final buffer is `[[11,22],[33,44]]`. MO→MAP still represents the sum tensor between the two iterators; later MAP fusion may combine eligible work while maintaining chain order. The test stops before that later optimization, which explains its expectation of two iterators. Fused attention has ordinary input/output tensors and no explicit chain in its custom-call signature, so it takes the earlier attention example's route. [Two-iterator CHECK](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MOToMAP/elementwise-kernels.mlir:115) [[entry:57]]

</div>
</details>
