## QnA 1 · What work happens in the MOGG stage?

**MOGG turns the implementation body into a well-formed callable helper.** This stage conforms call operands, outlines the routine body, adds a trace region and cleans up the resulting module. The tensor-level `mogg.routine` container remains for runtime lowering. [[gc:370]] [[outline:14]]

```calltree
gcPipelineImpl → buildMoggPipeline [[gc:600]] [[gc:370]]
├─ LegalizeRowwiseKernels [[gc:372]]
│  └─ prepare rowwise calls when present; none in this opaque attention body [[rowwise:14]]
├─ ConformCallOperands [[gc:373]]
│  ├─ lookup callee declaration and its execute signature [[conform:259]]
│  └─ pack scalar / variadic / trait arguments only as needed [[conform:41]]
├─ OutlineRoutines [[gc:374]]
│  ├─ make a module-level mogg.func and register its unique symbol [[outline:95]] [[outline:105]]
│  ├─ move body block + code into that function [[outline:109]]
│  └─ rebuild routine body as arguments forwarded by mogg.call [[outline:124]] [[outline:133]]
├─ WrapFuncsWithTrace [[gc:375]]
│  └─ move call dispatch into mogg.with_trace [[trace:48]] [[trace:65]]
└─ module canonicalizer → CSE [[gc:378]]
   └─ cleanup can reach the outlined functions beside mo.graph
```

The pass order is significant: outlining first makes a reusable helper boundary; cleanup must operate on the module because that helper is a sibling of the graph. [[gc:377]]

## QnA 2 · Does ConformCallOperands change Q, K, V or the output here?

The registered signature uses three ordinary `InputTensor` arguments, one ordinary `OutputTensor`, and one `DeviceContext`. It does not request scalar, variadic or trait-bound tensor arguments. The existing ordinary tensor operands therefore pass through this packing step. [[kernel:82]] [[conform:134]]

**SOURCE · native work inside the five scheduled steps.** `buildMoggPipeline` currently ignores its `MoggPipelineOptions` argument and schedules the same pass sequence. The rowwise legalization pass has its own default cost threshold of 16; that threshold classifies its reduction body and has no effect on this opaque attention call. [[gc:370]] [MOGGPasses.td:226](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGPasses.td:226)

**Example links** map the earlier Q&A illustrations and the worked examples to their native checks, rewrites and handoffs. Links marked “later” describe a subsequent stage; a listed rewrite still requires its matching conditions.

```calltree
gcPipelineImpl → buildMoggPipeline(pm,moggOpts) [[gc:600]] [[gc:370]]
├─ LegalizeRowwiseKernelsPass::runOnOperation [[rowwise:315]]
│  ├─ selectPrimaryInput: find first fold's element load; replace matching loads [[rowwise:72]]
│  ├─ bindMonoids: validate result fields/dtypes; record params and accumulator dtype [[rowwise:128]]
│  ├─ groupRowStores: move stores at the row index into rowwise.emit [[rowwise:188]]
│  └─ setLaunchParams: validate stores, set phase/cost/column attributes [[rowwise:222]]
│     ├─ [↗ QnA 7 · Example 9 — sum-of-squares rowwise phases, guards and errors](#qna-mogg-example-rowwise-reduction)
│     └─ this attention body has no call.rowwise → no rowwise mutation [[rowwise:260]]
│
├─ ConformCallOperandsPass::runOnOperation on the module [[conform:369]]
│  ├─ build MojoRegistry; walk calls post-order because replacements erase them [[conform:370]]
│  ├─ CallOpaqueOp → packCall [[conform:258]]
│  │  ├─ lookup decl; missing decl errors, no execute signature returns unchanged [[conform:259]]
│  │  ├─ copy execute adaptor and output list for local conformance [[conform:270]]
│  │  ├─ optional selectUnfusedExecute [[conform:273]]
│  │  │  └─ requires advertised overload, no computed epilogue, single-use pass-through outputs [[conform:234]]
│  │  │     └─ [↗ QnA 7 · Example 6 — unfused overload requires pass-through outputs and guards](#qna-mogg-example-unfused-overload)
│  │  ├─ packAgainstSignature → mapToKernelImplementation [[conform:279]] [[conform:155]]
│  │  │  ├─ operand/result mappings; launch context is handled separately [[operand-inference:310]]
│  │  │  ├─ read-only parameter fed by mutable view → cloneWithMut(false) [[conform:170]]
│  │  │  │  └─ [↗ QnA 7 · Example 5 — repair read permission with cloneWithMut(false)](#qna-mogg-example-mutable-input-repair)
│  │  │  └─ packOperands(inputs), then packOperands(outputs) [[conform:188]]
│  │  │     ├─ scalar → scalar.create; variadic run → tuple.create [[conform:83]] [[conform:91]]
│  │  │     │  ├─ [↗ QnA 5 · scalar/variadic examples — scalar payload 5.0 and tuple(A,B,C)](#qna-mogg-5-illustration-1)
│  │  │     │  ├─ [↗ QnA 7 · Example 2 — rank-zero tensor becomes a scalar argument](#qna-mogg-example-scalar-packing)
│  │  │     │  └─ [↗ QnA 7 · Example 3 — consecutive tensor operands become one tuple](#qna-mogg-example-variadic-packing)
│  │  │     ├─ ordinary trait → trait.create; fused trait retains its producer [[conform:61]]
│  │  │     │  ├─ [↗ QnA 5 · conformance comparison — scalar, variadic, trait and ordinary tensor roles](#qna-mogg-5-metadata-example)
│  │  │     │  └─ [↗ QnA 7 · Example 4 — trait/TensorTuple packing preserves tensor references](#qna-mogg-example-trait-and-tensor-tuple)
│  │  │     └─ ordinary tensor → same SSA value [[conform:134]]
│  │  ├─ DeviceContextArray signature → pack contexts into one tuple [[conform:293]]
│  │  ├─ changed=false → keep the existing call (attention takes this branch) [[conform:305]]
│  │  │  └─ [↗ QnA 2 · attention before/after — no packing, unchanged call fields](#qna-mogg-2-illustration-1)
│  │  └─ otherwise create replacement; retain callee/params/fusion flag/results [[conform:311]]
│  │     └─ rewire uses, erase old call, erase dropped epilogues/dead pure producers [[conform:317]]
│  ├─ CallShapeOp → packShapeCall; no output operand, result leaves by value [[conform:338]]
│  └─ any mapping failure → interrupt walk and signal pass failure [[conform:378]]
│
├─ OutlineRoutinesPass::runOnOperation [[outline:142]]
│  ├─ cached top-level SymbolTable; collect outer routines without recursing [[outline:146]]
│  └─ outlineKernel(builder,symtab,routine) [[outline:44]]
│     ├─ capture body argument types, nested constants and return operand types [[outline:49]]
│     ├─ function args = body args + trailing constant pointers [[outline:62]]
│     │  ├─ [↗ QnA 3 · helper signature example — four pointers + context; no tensor return](#qna-mogg-3-illustration-2)
│     │  └─ [↗ QnA 7 · Example 7 — constant resource gains a trailing helper pointer](#qna-mogg-example-outline-constant-resource)
│     ├─ preserve provenance; derive helper name; insert beside graph [[outline:79]]
│     ├─ create FuncOp with ABI=MOGG and device label; unique via symtab.insert [[outline:95]]
│     ├─ takeBody: move OLD block and all operations into helper [[outline:109]]
│     │  ├─ [↗ QnA 3 · ownership before/after — move body; rebuild forwarding call](#qna-mogg-3-illustration-1)
│     │  └─ [↗ QnA 7 · Example 1 — attention helper beside graph, retained tensor boundary](#qna-mogg-example-attention-helper)
│     ├─ constants' uses → new helper args; create empty output only if needed [[outline:114]]
│     └─ create NEW routine block; move constants back; call helper with fresh args [[outline:124]]
│
├─ WrapFuncsWithTrace::runOnOperation on each mogg.func [[trace:34]]
│  ├─ ABI != MOGG → return; no dispatch after setup → return [[trace:38]] [[trace:57]]
│  ├─ collect from first CallLike op onward, excluding terminator/existing trace [[trace:48]]
│  └─ move dispatch into with_trace; yield original return values to outer output [[trace:65]]
│     ├─ [↗ QnA 4 · tracing example — setup stays outside; dispatch enters with_trace](#qna-mogg-4-illustration-1)
│     └─ [↗ QnA 7 · Example 8 — returned shape crosses trace; cleanup/CSE follows](#qna-mogg-example-trace-return-and-cleanup)
│
└─ module canonicalizer → module CSE, including helper siblings of mo.graph [[gc:377]]
   ├─ [↗ QnA 4 · cleanup example — share equivalent shapes, keep distinct tensor pointers](#qna-mogg-4-illustration-2)
   └─ [↗ QnA 6 · handoff example — graph routine + module helper, storage comes next](#qna-mogg-6-illustration-1)
```

`packCall` copies the declaration's function adaptor before changing output contracts; it does not rewrite P1's registered API. The SSA call is replaced only when conformance changes something. Outlining subsequently **moves** its body objects into a helper and creates new forwarding objects in the graph-local routine. [[conform:270]] [[conform:305]] [[outline:109]] [[outline:124]]


<span id="qna-mogg-2-illustration-1"></span>

```state
Attention call before conformance
  inputs  = [%Q,%K,%V]       all ordinary read-only tensor views
  outputs = [%O]            ordinary writable output view
  context = [%ctx]          single device context
  params  = {BD=8, BN=16}

Attention call after conformance
  inputs  = [%Q,%K,%V]
  outputs = [%O]
  context = [%ctx]
  params  = {BD=8, BN=16}
```

The pass still checks the declaration mapping. When no packing or overload adjustment is needed, it returns without replacing the call. Existing types and shape metadata remain available to the subsequent emitter. [[conform:259]] [[conform:305]]

The implementation also has an unfused-overload path for eligible fused-output kernels. This attention output is already ordinary, so no pass-through epilogue needs removal here. [[conform:273]] [[kernel:82]]

## QnA 3 · Show exactly what outlining moves and what it leaves behind.

**Outlining creates a named function from an existing region body.** For readability, `@F_attention` below names the new helper; the real symbol is derived from fusion provenance and uniqued by the module symbol table. [[outline:83]] [[outline:105]]

<span id="qna-mogg-3-illustration-1"></span>

```transform
BEFORE · one graph-local routine body
IR1 : builtin.module
└─ mo.graph @fused_attention
   └─ mogg.routine(%q,%k,%v) → %o:T0
      └─ body(%qptr,%kptr,%vptr,%optr,%ctx)
         ├─ shape/tensor constructors
         └─ mogg.call.opaque → P1.FusedAttention

AFTER · tensor boundary and helper are separate
IR1 : builtin.module
├─ mo.graph @fused_attention
│  └─ mogg.routine(%q,%k,%v) → %o:T0
│     └─ NEW body(%qptr,%kptr,%vptr,%optr,%ctx)
│        └─ mogg.call @F_attention(%qptr,%kptr,%vptr,%optr,%ctx)
└─ mogg.func @F_attention          abi=MOGG, device="gpu"
   └─ MOVED body(%qptr,%kptr,%vptr,%optr,%ctx)
      ├─ shape/tensor constructors
      ├─ mogg.call.opaque → P1.FusedAttention
      └─ mogg.output              no by-value tensor return
```

The implementation literally transfers the body using `takeBody`. It then creates fresh arguments for the retained routine's forwarding call. This move changes ownership of IR blocks; it does not copy Q/K/V data. [[outline:109]] [[outline:124]]

<span id="qna-mogg-3-illustration-2"></span>

```state
F_attention signature
  (!mogg.pointer, !mogg.pointer, !mogg.pointer,
   !mogg.pointer, !mogg.context) → ()

Meaning
  Q pointer, K pointer, V pointer, O pointer, GPU context
  O is written through its pointer; no tensor object is returned by this helper.

Enclosing routine signature remains (T0,T0,T0) → T0.
G1's original Python graph remains separate from all these IR1 objects.
```

The source obtains the helper's signature from the routine body's argument types. For an ordinary output tensor, it adds no by-value return. The actual kernel implementation also puts its `OutputTensor` in the argument list. [[outline:49]] [[outline:65]] [[kernel:82]]

**TEST · inspected expectation:** `infra-outline-routines` checks the graph-local forwarding call and the module-level private MOGG helper with tensor reconstruction inside it. [[test-outline:14]]

## QnA 4 · How do tracing, canonicalization and CSE transform the helper?

Trace wrapping leaves setup before the first call outside the trace and moves the operations from that call onward into a `mogg.with_trace` region. This records an instrumentation boundary; it is not a GPU execution event during compilation. [[trace:44]] [[trace:65]]

<span id="qna-mogg-4-illustration-1"></span>

```transform
BEFORE tracing
  shape/tensor setup
  call FusedAttention
  mogg.output

AFTER tracing
  shape/tensor setup
  mogg.with_trace {
    call FusedAttention
    mogg.yield
  }
  mogg.output
```

For the static attention body, four calls to `buildShape([32,32])` initially create equivalent shape descriptions. CSE can share their eligible shape-producing operations. The Q/K/V/O tensor views remain distinct because they reference different pointers and output mutability differs. [[map-lowering:102]] [[gc:378]]

<span id="qna-mogg-4-illustration-2"></span>

```transform
BEFORE cleanup                         AFTER eligible CSE
  sq = shape.create([32,32])              s = shape.create([32,32])
  sk = shape.create([32,32])              Q = tensor(qptr,s,Input,false)
  sv = shape.create([32,32])              K = tensor(kptr,s,Input,false)
  so = shape.create([32,32])              V = tensor(vptr,s,Input,false)
  Q = tensor(qptr,sq,Input,false)         O = tensor(optr,s,Output,true)
  K = tensor(kptr,sk,Input,false)
  V = tensor(vptr,sv,Input,false)
  O = tensor(optr,so,Output,true)

Illustration of an eligible cleanup, not a captured dump.
Shared shape description ≠ shared tensor storage.
```

Identical computation with side effects is not automatically redundant. In particular, later `mogg.compiled.call` operations may read/write their pointed-to buffers, so unused results or equal operands do not authorize erasing those calls. [[mogg-td:444]]

## QnA 5 · Give examples where operand conformance really does add operations.

These are **comparison signatures**, showing what the same pass supports. They are not changes to the attention registration. [[conform:41]]

<span id="qna-mogg-5-metadata-example"></span>

| Callee requires | Incoming representation | Transformation |
| --- | --- | --- |
| scalar argument | one tensor representing a scalar | `mogg.scalar.create(%tensor)` |
| variadic tensor argument | several consecutive tensor operands | `mogg.tuple.create(%a,%b,...)` |
| non-fused tensor trait | plain tensor view | `mogg.trait.create` with the declared contract |
| ordinary `InputTensor` | ordinary tensor view | keep the operand |

<span id="qna-mogg-5-illustration-1"></span>

```transform
Example scalar input, conceptual values
  rank-0 tensor<[],f32> with scalar payload 5.0
    → scalar.create
    → scalar argument 5.0 expected by the callee

Example variadic input
  tensor handles [%A,%B,%C]
    → tuple.create(%A,%B,%C)
    → one aggregate argument preserving the three tensor references
```

The generated code later converts the scalar tensor through the registered scalar primitive. Tuple packing describes a call argument aggregate; it does not imply concatenation of the tensor elements. [[mogg-ops:1944]] [[mogg-ops:1960]]

## QnA 6 · What reaches MOGG→MGP?

<span id="qna-mogg-6-illustration-1"></span>

```state
IR1
├─ mo.graph @fused_attention : (T0,T0,T0) → T0
│  ├─ one mogg.routine with device=gpu:0
│  │  └─ call @F_attention through 4 pointers + context
│  └─ mo.output %o
└─ mogg.func @F_attention
   ├─ ABI = MOGG
   ├─ device kind = "gpu"
   ├─ tensor construction from [32,32], f32
   └─ registered implementation call, BN=16, BD=8

No GPU binary or initialized Model has been produced at this boundary.
Next: map tensors to storage and build the runtime model's two regions.
```

MOGG→MGP consumes the retained routine and its outlined helper. Later EmitMojo writes helper source, and JIT compilation turns that source and P1 implementations into native artifacts. Keeping those steps separate explains why creating `@F_attention` is not yet machine-code generation. [[mogg-ops:716]] [[gc:611]] [[gc:484]] [[gc:619]]


## QnA 7 · How do operand conformance, outlining, tracing and rowwise legalization change concrete calls while retaining their data and ownership contracts?

**SOURCE · nine worked examples.** The IR diagrams are schematic, with shortened symbols and attribute spellings. Numeric inputs are illustrative; no MAX, Mojo, GPU or compiler-test workload was executed. **TEST** paragraphs explain where independently supplied `.mlir` inputs enter a test pipeline and what the `CHECK` directives expect. The production compiler does not call these files. A test that checks emitted source establishes an emission expectation, not a measured numerical result.

<details class="qna-cell" id="qna-mogg-example-attention-helper" markdown="1">
<summary>Example 1 · Why is attention's call unchanged by packing, yet moved into a new named helper by outlining?</summary>
<div class="qna-answer" markdown="1">

The actual attention call has three ordinary `InputTensor` operands Q/K/V, one ordinary `OutputTensor` O, and one `DeviceContext`. Each tensor is `f32[32,32]`; `BN=16,BD=8` remain call parameters. With the illustrative inputs Q=K=0 and `V[j,d]=j`, O still represents 32 rows filled with `15.5`. Conformance and outlining modify the calling structure around that same computation. [[kernel:74]] [[kernel:82]] [[entry:57]]

```transform
BEFORE · schematic MOGG inside IR1
mo.graph @fused_attention(%q,%k,%v) -> T0 {             # Retain the graph's tensor boundary.
  %o = mogg.routine(%q,%k,%v) -> T0 {             # Bind the future output destination.
    OLD body(%qp,%kp,%vp,%op:pointer,%ctx:context) # Five existing body arguments.
    shape/tensor constructors → %Q,%K,%V,%O       # Rebuild four typed tensor views.
    call.opaque @FusedAttention                   # Same P1 implementation.
      inputs(%Q,%K,%V) outputs(%O) context(%ctx)   # Ordinary signature already matches.
      params={BN=16,BD=8}                        # Tile parameters stay compile-time data.
  }
  mo.output %o                                   # Return one graph tensor.
}
```

`packAgainstSignature` still maps those operands to the declaration, but `packOperands` returns each ordinary tensor directly. No scalar, tuple, trait or mutable-input repair is needed; the final guard keeps the existing `CallOpaqueOp` object. [ConformCallOperands.cpp:134](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:134) [ConformCallOperands.cpp:279](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:279)

```cpp
if (!changed)
  return success();
```

Outlining then creates a module-level `mogg.func`, derives its parameter types from the routine body, registers its name and **moves** the body into it. This is the central ownership operation. [OutlineRoutines.cpp:95](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:95) [OutlineRoutines.cpp:109](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:109)

```cpp
stub.getBody().takeBody(kernel.getBody());
Block &stubBody = stub.getBody().front();
```

```transform
AFTER · schematic MOGG before tracing
mo.graph @fused_attention(%q,%k,%v) -> T0 {             # Original tensor contract remains.
  %o = mogg.routine(%q,%k,%v) -> T0 {             # Same routine operation and outer uses.
    NEW body(%qp2,%kp2,%vp2,%op2:pointer,%ctx2)    # Fresh forwarding arguments.
    mogg.call @F_attention(%qp2,%kp2,%vp2,%op2,%ctx2) # Forward all five references.
  }
  mo.output %o                                   # Same graph result.
}
mogg.func @F_attention abi=MOGG device="gpu" {    # New module sibling of the graph.
  MOVED body(%qp,%kp,%vp,%op:pointer,%ctx)        # Existing block and arguments move here.
  shape/tensor constructors → %Q,%K,%V,%O         # Existing setup operations move with it.
  call.opaque @FusedAttention                     # Existing call object moves with the block.
    inputs(%Q,%K,%V) outputs(%O) context(%ctx)     # Same operand identities within that block.
    params={BN=16,BD=8}                          # Same parameters.
  mogg.output                                    # No by-value result for this tensor output.
}
```

| Object | Conformance | Outlining |
| --- | --- | --- |
| P1 declaration | read for matching | retained |
| `CallOpaqueOp` in attention | retained unchanged | moved with its block |
| routine's original block/arguments | retained | owned by new helper |
| routine's replacement block/call | absent | newly created |
| Q/K/V/O payload buffers | not evaluated by this pass | future pointers are forwarded |
| G1's Python graph | separate from IR1 | separate from IR1 |

`@F_attention` is a teaching name. The implementation derives a preferred name from fusion provenance, then `symtab.insert(stub)` returns a unique symbol when names collide. The new call uses that returned symbol. Two routines with the same preferred name are not merged by this outlining pass. [OutlineRoutines.cpp:83](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:83) [OutlineRoutines.cpp:105](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:105) [OutlineRoutines.cpp:133](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:133)

**TEST · helper structure.** [infra-outline-routines.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir:1) starts with MO custom calls and runs import → MO→MAP → accumulating-buffer setup → MAP→MOGG → outlining. Its plain `[2,2]` case checks that the graph-local body contains `mogg.call`, that `mogg.call.opaque` appears inside the new private helper, and that the helper has `abi=MOGG` and a terminator. Its dynamic case checks pointer, pointer, dimension, context in the forwarded argument list and helper signature. These tests exercise outlining separately; attention's no-packing branch is established by the signature and C++ guard above. [infra-outline-routines.mlir:14](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir:14) [infra-outline-routines.mlir:50](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir:50)

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-scalar-packing" markdown="1">
<summary>Example 2 · How does a rank-zero tensor carrying 5 become a scalar call operand without evaluating the kernel?</summary>
<div class="qna-answer" markdown="1">

Use the test signature `(output: OutputTensor, a: InputTensor, s: Scalar[dtype], ctx)`. Choose `a:[4]=[1,2,3,4]` and a **rank-zero** f32 tensor `s:[]` containing `5.0`. The packing pass changes the representation of `s` at the call boundary. The fixture's kernel body is empty, so no result such as `a+5` is implied. [codegen-map-abi-stub.mlir:51](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:51) [testing_kernels.mojo:3713](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3713)

```transform
BEFORE · schematic MOGG
%A = tensor.create(%ap,shape=[4],f32,Input,false)  # Four data elements in the ordinary input.
%S = tensor.create(%sp,shape=[],f32,Input,false)   # One scalar value, rank zero.
%O = tensor.create(%op,shape=[4],f32,Output,true)  # Preserve the output destination.
call.opaque @MapLowerScalar inputs(%A,%S) outputs(%O) context(%ctx) # S is still a tensor view.
```

Operand inference identifies the second input as a scalar parameter. `packOperands` creates one `ScalarCreateOp`, sets `changed=true`, and advances to the next input. [ConformCallOperands.cpp:83](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:83)

```cpp
if (arg.isScalarType()) {
  packed.push_back(ScalarCreateOp::create(
      builder, loc, ScalarType::get(ctx), operands[i]));
  changed = true;
  ++i;
  continue;
}
```

```transform
AFTER · schematic MOGG
%A,%S,%O retain their existing producers           # No tensor contents change here.
%scalar = mogg.scalar.create(%S)                   # New scalar representation of S's payload.
call.opaque @MapLowerScalar                        # New call operation, same callee.
  inputs(%A,%scalar) outputs(%O) context(%ctx)      # Only the second input changes representation.
```

The old call is erased after replacement. Its callee, parameters, result types and epilogue-fusion flag are retained. Later `ScalarCreateOp::emitMojo` emits a call to the registered `mogg.as_scalar` primitive; conformance itself only builds an IR operation and does not read `%sp`. [ConformCallOperands.cpp:311](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:311) [ScalarCreateOp::emitMojo:1944](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1944)

| Candidate original MO operand | Signature check |
| --- | --- |
| immutable `tensor<[],f32>` on CPU | scalar-compatible |
| immutable `tensor<[1],f32>` on CPU | rank 1; rejected as a scalar argument |
| rank-zero tensor on GPU | rejected by the host-scalar requirement |
| already lowered `!mogg.tensor` | rank/device metadata is no longer in this handle type; upstream validation supplies that contract |

The rank and host-device checks occur in `matchConvertedMOTypeToKernelDeclType`. At the MOGG boundary the same helper recognizes an opaque codegen handle whose rank/device were checked earlier; it does not infer rank zero from the number of tensor handles. [OperandInference.cpp:81](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/OperandInference.cpp:81)

**TEST · direct conformance input.** [package.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/package.mlir:1) starts directly with `mogg.call.opaque` in `func.func`, imports its declaration, and runs `--mogg-conform-call-operands`. `@scalar` checks a new `mogg.scalar.create` whose captured result is the second call input of type `!mogg.scalar`. The test bypasses MO→MAP/MAP→MOGG deliberately to isolate this packing rule. [package.mlir:10](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/package.mlir:10)

**TEST · emission connection.** [codegen-map-abi-stub.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:1) instead starts at MO and runs through MAP→MOGG, conformance, outlining and Mojo emission. Its scalar checks require the emitted `mogg_as_scalar` result to be passed to `MapLowerScalar.execute`. It reads generated source using FileCheck; it does not execute that kernel. The abbreviated pipeline uses `-verify-each=false` because it stops after emission erases the helper while the routine still names it. [codegen-map-abi-stub.mlir:15](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:15) [codegen-map-abi-stub.mlir:45](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-stub.mlir:45)

Attention has no scalar argument in `execute`; BN and BD remain parameter attributes and do not enter this packing rule. [[kernel:78]] [[kernel:82]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-variadic-packing" markdown="1">
<summary>Example 3 · How are several tensor operands packed into one variadic argument while preserving their separate buffers?</summary>
<div class="qna-answer" markdown="1">

Use two immutable input views `A:[4]=[1,2,3,4]` and `B:[4]=[10,20,30,40]`, with a signature declaring one `InputVariadicTensors` parameter. The flat call has two tensor operands; the declaration says both belong to the same parameter. Their numerical contents remain separate. This fixture has an empty body, so its output computation is unspecified. [testing_kernels.mojo:3728](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3728)

```transform
BEFORE · schematic MOGG
call.opaque @MapLowerVariadic                     # Declaration expects one tensor aggregate.
  inputs(%A,%B) outputs(%O) context(%ctx)          # Flat input list still has two handles.

OPERAND MAPPING
flat input 0 → declaration position "inputs"      # First member of this variadic run.
flat input 1 → declaration position "inputs"      # Same declared position, same run.
```

`packOperands` records the declaration position and collects consecutive operands only while their mapping refers to that same variadic parameter. It then creates one tuple operation. This is the native run-boundary check. [ConformCallOperands.cpp:91](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:91)

```cpp
unsigned pos = arg.sourceDescriptor->position;
SmallVector<Value> run;
while (i < e) {
  auto runIt = mapping.find(i);
  if (runIt == mapping.end() || !runIt->second.isVariadicTensorType() ||
      runIt->second.sourceDescriptor->position != pos)
    break;
  run.push_back(operands[i]);
  ++i;
}
```

```transform
AFTER · schematic MOGG
%inputs = mogg.tuple.create(%A,%B) variadic_abi=plain # Aggregate two existing references.
call.opaque @MapLowerVariadic                       # Rebuild the call with the same symbol.
  inputs(%inputs) outputs(%O) context(%ctx)          # One aggregate input, same output/context.

Logical aggregate
member 0 → A pointer + shape [4] → [1,2,3,4]       # Retain A's storage identity.
member 1 → B pointer + shape [4] → [10,20,30,40]   # Retain B's storage identity.
```

The later emitter builds runtime pointer/shape tuples and compile-time tensor specifications. There is no concatenation of the eight input elements in this packing operation. A real `mo.concat` still performs its declared concatenation through its implementation; packing only prepares the argument aggregate it receives. [MOGGOps.cpp:1963](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:1963)

| Declared parameter kind | Tuple's recorded ABI | Information retained |
| --- | --- | --- |
| ordinary variadic tensors | `plain` | per-member data references and shapes |
| stable variadic tensors | `stable` | slot-backed member convention |
| fused input/output variadics | `fused_input` / `fused_output` | member fusion convention |
| trait `TensorTuple` | `tensor_tuple` | per-member trait adapters; worked in Example 4 |

The choice comes from the declaration's adaptor, not from guessing based on a producer's mutability. [ConformCallOperands.cpp:108](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:108)

**Guard against ambiguous groups.** If one input list binds multiple variadic parameters, operand inference requires `hasSameVariadicOperandSize`; otherwise the flat list supplies no group boundaries. With the marker, the remaining variadic operand count must divide evenly between groups. For four operands and two equal groups the group size is two; five operands fails that condition. Launch arguments such as the context are skipped during this tensor mapping. [OperandInference.cpp:116](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/OperandInference.cpp:116) [OperandInference.cpp:166](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/OperandInference.cpp:166)

A `DeviceContextArray` signature uses a separate packing branch. For illustrative contexts `%c0` and `%c1` associated with two devices, it builds one tuple and replaces `context(%c0,%c1)` with `context(%contexts)`. Their order comes from the earlier MAP→MOGG unique-device collection. Ordinary attention declares one context and skips this branch. [ConformCallOperands.cpp:290](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:290) [MAPOpsLowerableToMOGG.cpp:966](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:966)

**TEST · direct packing and flag preservation.** [package.mlir:19](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/package.mlir:19) starts with two MOGG tensor inputs, imports the test kernel, runs only conformance and checks one tuple plus a one-input call. Its third case starts with `hasEpilogueFusion=true` and checks that the rebuilt call retains it. That flag is needed after packing obscures the original output producer. [package.mlir:28](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/package.mlir:28)

**TEST · ABI reaches emission.** [variadic-abi.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/variadic-abi.mlir:1) starts at MO, performs fusion and MAP→MOGG, then explicitly runs conformance before outlining, MOGG→MGP and emission. Its checks require the emitted fused-input, fused-output and plain aggregate types. The concat input has lengths `2,3,4` and output length `9`; the check is for its argument ABI, not an executed concatenation result. [variadic-abi.mlir:18](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/variadic-abi.mlir:18) [variadic-abi.mlir:28](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/variadic-abi.mlir:28) [variadic-abi.mlir:36](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/variadic-abi.mlir:36)

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-trait-and-tensor-tuple" markdown="1">
<summary>Example 4 · How do ordinary and heterogeneous tensor arguments acquire the declared trait contracts?</summary>
<div class="qna-answer" markdown="1">

`TraitNegate` declares input/output arguments through `TileTensorable` traits. For illustrative `X:[4]=[1,-2,3,-4]`, its body writes `O=[-1,2,-3,4]`. Conformance prepares the trait-compatible argument representation while preserving these element and storage roles. [testing_kernels.mojo:328](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:328) [testing_kernels.mojo:351](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:351)

```transform
BEFORE · schematic MOGG
%X = tensor.create(%xp,[4],f32,Input,false)       # Plain read-only tensor view.
%O = tensor.create(%op,[4],f32,Output,true)       # Plain writable tensor view.
call.opaque @TraitNegate inputs(%X) outputs(%O) context(%ctx) # Declaration requests trait arguments.

AFTER · schematic MOGG
%Xarg = mogg.trait.create(%X)                     # New input adapter over the same view.
  contract={role=input,fused=false,access=none,capabilities=none}, device="cpu"
%Oarg = mogg.trait.create(%O)                     # New output adapter over the same destination.
  contract={role=output,fused=false,access=none,capabilities=none}, device="cpu"
call.opaque @TraitNegate inputs(%Xarg) outputs(%Oarg) context(%ctx) # Same negation computation.
```

The helper `asTraitArg` skips already-fused contracts, otherwise builds `TraitCreateOp` using the operand's own MOGG kind and the enclosing routine's device label. A mutable input can remain `!mogg.buffer`; ordinary input/output tensors remain `!mogg.tensor`. The adapter's IR type alone is therefore insufficient to explain the contract—the attribute carries that information. [ConformCallOperands.cpp:61](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:61)

```cpp
if (contract.fused)
  return operand;
changed = true;
// Keep the operand's own kind: a mutable-input trait argument comes in as
// a `!mogg.buffer`, everything else as a `!mogg.tensor`.
return TraitCreateOp::create(builder, loc, operand.getType(), operand,
                             builder.getStringAttr(device.getLabel()),
                             tensorArgContractToAttr(ctx, contract));
```

**TEST · ordinary traits.** [conform-trait-args.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-args.mlir:1) enters at MO, imports the test package, runs MO→MAP → MAP→MOGG → conformance, and checks both role-specific adapters feeding `TraitNegate`. The original input/output tensor constructors retain their respective `mut=false/true` attributes. [conform-trait-args.mlir:15](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-args.mlir:15)

**Heterogeneous members keep their own shapes and dtypes.** The `TraitTensorTupleAdd` test supplies bias `f32[1]`, A `si32[2]`, B `f64[3]`, with outputs `f32[2]` and `si32[3]`. Choose illustrative bias `[10]`, A `[1,2]`, B `[3,4,5]`; the registered body casts each member to its corresponding output dtype and adds the bias, yielding `[11,12]` and `[13,14,15]`. [conform-trait-tensor-tuple.mlir:42](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-tensor-tuple.mlir:42) [testing_kernels.mojo:871](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:871)

```transform
BEFORE · flat call lists
call @TraitTensorTupleAdd inputs(%bias,%A,%B) outputs(%O0,%O1) # Three input/two output handles.

AFTER · schematic trait and tuple packing
%barg = trait.create(%bias,role=input)            # Ordinary bias parameter remains separate.
%aarg = trait.create(%A,role=input)               # Preserve A's si32[2] contract.
%bmember = trait.create(%B,role=input)            # Preserve B's f64[3] contract.
%ins = tuple.create(%aarg,%bmember) abi=tensor_tuple # One heterogeneous input aggregate.
%o0arg = trait.create(%O0,role=output)             # Preserve O0's f32[2] destination.
%o1arg = trait.create(%O1,role=output)             # Preserve O1's si32[3] destination.
%outs = tuple.create(%o0arg,%o1arg) abi=tensor_tuple # One heterogeneous output aggregate.
call @TraitTensorTupleAdd inputs(%barg,%ins) outputs(%outs) # Same paired computations.
```

This adds five trait adapters and two tuples; the routine's three input pointers, two output pointers and context stay as its boundary arguments. In the variadic branch, the compiler applies `asTraitArg` to each member before constructing the `tensor_tuple` aggregate. [ConformCallOperands.cpp:109](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:109)

**TEST · tuple member contracts.** [conform-trait-tensor-tuple.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-tensor-tuple.mlir:1) uses `MAX_GC_USE_ADV_FUSION=1` and first stops after conformance, checking each adapter, both tuple ABIs and the reordered call lists. Its second command continues through outlining, MOGG→MGP and emission, where `MOJO` checks require `TensorTuple` construction and positional output/bias/input passing. These are two independent printed-IR/source checks of the same declaration contract. [conform-trait-tensor-tuple.mlir:21](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-tensor-tuple.mlir:21) [conform-trait-tensor-tuple.mlir:31](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-tensor-tuple.mlir:31)

**Skipped branch: a fused argument already has a producer contract.** Its prologue/epilogue owns the fusion functor; the trait contract is stamped there during MAP→MOGG. `asTraitArg` therefore returns the operand unchanged when `contract.fused=true`. [MAPOpsLowerableToMOGG.cpp:408](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:408) [ConformCallOperands.cpp:63](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:63)

**TEST · earlier producer stamp.** [conform-trait-args-fused.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-args-fused.mlir:1) starts at MO and stops at MAP→MOGG. It checks that a fused-input prologue carries `role=input, fused=true, access=load`. This test does not run conformance; it establishes the producer metadata that the later source guard preserves. The actual attention signature uses ordinary `InputTensor`/`OutputTensor`, so neither the ordinary trait wrapping nor fused-trait branches apply to it. [conform-trait-args-fused.mlir:17](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/conform-trait-args-fused.mlir:17) [[kernel:82]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-mutable-input-repair" markdown="1">
<summary>Example 5 · How can a read-only kernel argument use a mutable buffer without changing the destination's write permission?</summary>
<div class="qna-answer" markdown="1">

The test's graph has distinct `dst` and `srcbuf`, both `f32[2,2]`. MAP fusion has absorbed a mutable load of `srcbuf` into a call whose `dst` parameter is mutable and whose `src` parameter is read-only. MAP→MOGG initially reconstructs both buffer inputs as mutable `!mogg.buffer` views. [fused-mutable-load-opaque.mlir:8](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/fused-mutable-load-opaque.mlir:8)

```transform
BEFORE · schematic MOGG after MAP→MOGG
%D = tensor.create(%dstp,[2,2],f32,Input,true) : buffer # Kernel is allowed to write dst.
%Smut = tensor.create(%srcp,[2,2],f32,Input,true) : buffer # Storage originally arrived as a buffer.
call @MapLowerInplaceCopy inputs(%D,%Smut) context(%ctx) # Second parameter only reads.

Outer routine keeps its dependency chain and read/write effects. # Preserve ordering of buffer access.
```

`packAgainstSignature` enables mutable-to-immutable input matching for opaque kernel calls. For a mapped non-mutable parameter, it requires a `!mogg.buffer` operand whose producer implements `TensorCreateInterface` and is marked mutable. It clones that producer with `mut=false` and rewires just this call input. [ConformCallOperands.cpp:170](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:170)

```cpp
Value operand = reconciledInputs[idx];
if (!isa<BufferType>(operand.getType()))
  continue;
auto producer = operand.getDefiningOp<TensorCreateInterface>();
if (!producer || !producer.getMut())
  continue;
builder.setInsertionPointAfter(producer);
reconciledInputs[idx] =
    producer.cloneWithMut(builder, /*mut=*/false)->getResult(0);
reconciled = true;
```

```transform
AFTER · schematic MOGG immediately after conformance
%D = tensor.create(%dstp,[2,2],f32,Input,true) : buffer # Retain the mutable destination view.
%Smut = tensor.create(%srcp,[2,2],f32,Input,true) : buffer # Old view may become dead.
%Sread = tensor.create(%srcp,[2,2],f32,Input,false) : tensor # New read-only view, same pointer.
call @MapLowerInplaceCopy inputs(%D,%Sread) context(%ctx) # Match each parameter independently.

Later cleanup may erase %Smut if no users remain.       # Producer cleanup does not free src storage.
```

For illustrative source contents `[[1,2],[3,4]]`, `%Sread` still refers to those same four f32 elements; it describes 16 payload bytes beginning at the same symbolic `%srcp`. The compiler neither copies them nor makes `dst` immutable. This signature-only testing kernel has an empty body, so the example deliberately makes no claim about values written to `dst`. [testing_kernels.mojo:3616](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3616)

| Guard | Action |
| --- | --- |
| parameter is actually mutable | keep `%D` as a buffer |
| argument is already an ordinary immutable tensor | keep it |
| read-only parameter, mutable tensor-creation producer | clone producer with `mut=false` and replace this use |
| immutable value proposed for a mutable output | no reverse widening; signature mismatch remains an error |

Input matching permits only the mutable-buffer-to-read-only-view conversion. Output matching calls the same matcher with `allowMutToImmut=false`. Existing dependency chains continue to order access to the original storage. [OperandInference.cpp:57](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/OperandInference.cpp:57) [OperandInference.cpp:322](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/Conversion/OperandInference.cpp:322)

**TEST · before and after are checked by different commands.** [fused-mutable-load-opaque.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/fused-mutable-load-opaque.mlir:1) starts at MO and runs package import → MO→MAP → the MAP pipeline → MAP→MOGG. Its `CHECK` expectations require two mutable views and a `buffer,buffer` call. The second `RUN` adds conformance; `CONFORM` checks require a new `mut=false` tensor and a `buffer,tensor` call. This pairs the exact producer cloning branch with the changed call types. [fused-mutable-load-opaque.mlir:18](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/fused-mutable-load-opaque.mlir:18)

Attention's Q/K/V are ordinary immutable tensor inputs already, so this repair leaves them alone. [[kernel:83]]

</div>
</details>


<details class="qna-cell" id="qna-mogg-example-unfused-overload" markdown="1">
<summary>Example 6 · When can pass-through output epilogues be removed to select an unfused execute overload?</summary>
<div class="qna-answer" markdown="1">

A kernel can advertise both a fused-output `execute` and an ordinary-output overload. Suppose its result must be written unchanged to each destination: the output epilogues only forward `(value,index)` to a captured output. Passing the ordinary output views directly can express that same destination behavior. `packCall` attempts this only when the declaration advertises the overload and the call-wide `hasEpilogueFusion` flag is false. [ConformCallOperands.cpp:270](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:270)

```cpp
bool selectedUnfused = adaptor.supportsUnfusedExecute &&
                       !call.getHasEpilogueFusion() &&
                       selectUnfusedExecute(fn, outputs, droppedEpilogues);
```

```transform
BEFORE · schematic two-output call with pass-through epilogues
%ez = tensor.create.epilogue(%v,%i) captures(%Z) { # One destination capture.
  tensor.store %Z[%i] = %v                        # Write the supplied value unchanged.
}
%ew = tensor.create.epilogue(%v,%i) captures(%W) { # Second independent destination.
  tensor.store %W[%i] = %v                        # Another pass-through store.
}
call @K inputs(%X) outputs(%ez,%ew)               # Same kernel symbol has two execute forms.
  hasEpilogueFusion=false                         # No output arithmetic was fused.
```

`selectUnfusedExecute` inspects **every** fused output before mutating the output list. Each must be a `TensorCreateEpilogueOp`, have one use, and not recompute the value. Any failure keeps the fused form for the whole call. [ConformCallOperands.cpp:237](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:237)

```cpp
auto epilogue = dyn_cast<TensorCreateEpilogueOp>(producer.getOperation());
if (!epilogue || !out.hasOneUse() || epilogue.recomputesValue())
  return false;
passThroughs.emplace_back(i, epilogue);
```

The producer's `recomputesValue()` uses a strict pass-through test: the body must contain exactly one operation, either a yield of the incoming value or a store of that value at the incoming index into the first capture. Even unchanged data with extra index-transform operations does not satisfy that simple-body test. [MOGGOps.cpp:3439](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/MOGGOps.cpp:3439)

On success, the output operand becomes the epilogue's first capture, its store target. The compiler strips fusion from the **local copy** of each output contract and packs those ordinary outputs. The registered declaration is retained. [ConformCallOperands.cpp:249](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:249)

```transform
AFTER · schematic successful selection
%Xarg = trait.create(%X,ordinary-input-contract)   # Adapt the unchanged input if trait-bound.
%Zarg = trait.create(%Z,ordinary-output-contract)  # Wrap the original Z destination if trait-bound.
%Warg = trait.create(%W,ordinary-output-contract)  # Wrap the original W destination if trait-bound.
call @K inputs(%Xarg) outputs(%Zarg,%Warg)         # Same callee symbol; plain operand types choose overload.
remove %ez and %ew                                # Drop the two pass-through regions.
remove their unused pure pointer/setup producers # Erase only when no live use remains.

Value-preserving illustration for an overload pair with equal contracts
X=[1,2,3,4], Z=X+1, W=X-1                         # Both overloads implement these same expressions.
Z=[2,3,4,5], W=[0,1,2,3] before and after          # Only destination passing changes.
```

The arithmetic pair above is a teaching contract for equivalent overloads. The repository's test kernel deliberately adds a diagnostic output, discussed below. The cleanup only erases an epilogue operand's defining operation when it has no uses and is memory-effect-free. It does not erase the tensor storage behind a surviving output view. [ConformCallOperands.cpp:320](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/ConformCallOperands.cpp:320)

```transform
SKIPPED · one output actually transforms the value
%ew = tensor.create.epilogue(%v,%i) captures(%W,%X) { # Fuse work into the W output.
  %x = tensor.load %X[%i]                            # Read X at the same element.
  %sum = call.elementwise @Add(%v,%x)               # Change the value to store.
  tensor.store %W[%i] = %sum                        # Preserve the fused addition.
}
call @K outputs(%ez,%ew) hasEpilogueFusion=true      # Keep both outputs in fused form.
```

| Condition | Outcome |
| --- | --- |
| advertised overload + all fused outputs are private pass-throughs | replace those epilogues by ordinary destinations |
| any output recomputes a value | keep the fused call |
| a fused output has another user | keep the fused call |
| a fused output is a different producer kind | keep the fused call |
| no output epilogues are present | selector returns false; nothing to remove |

**TEST · declaration and selected output types.** [codegen-unfused-execute.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-unfused-execute.mlir:1) first imports the test package and checks `supports_unfused_execute` in the declaration. Its second route starts at MO, performs MAP fusion/setup, MAP→MOGG, conformance, outlining, MOGG→MGP and emission. `@unfused` checks two dense output adapters and excludes store-fusion wrappers; `@fused_other` adds `%w2=mo.add(%w,%x)` and checks two store-fusion wrappers. This tests the call-wide, all-or-nothing branch. [codegen-unfused-execute.mlir:18](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-unfused-execute.mlir:18) [codegen-unfused-execute.mlir:31](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-unfused-execute.mlir:31)

The fixture intentionally sets `z=x+1` in both overloads, but writes `w=0` in the plain overload and `w=1` in the fused overload to identify the selected branch. For illustrative `x=[1,2,3,4]`, `z=[2,3,4,5]` in both cases; the second graph's fused `%w2=w+x` is `[2,3,4,5]`. Those marker outputs are deliberately different, so this fixture should not be described as proof of numerical equivalence between arbitrary overload bodies. Its test checks emitted wrapper selection. [testing_kernels.mojo:3988](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:3988) [testing_kernels.mojo:4013](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:4013) [testing_kernels.mojo:4041](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/kernels/testing_kernels.mojo:4041)

Attention's output is already ordinary and supplies no pass-through output epilogue for this selector to remove. [[kernel:82]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-outline-constant-resource" markdown="1">
<summary>Example 7 · Why does outlining move a constant's pointer back to the call site and add a helper argument for it?</summary>
<div class="qna-answer" markdown="1">

Consider a routine that copies a constant matrix `C=[[1,2],[3,4]]` into its `f32[2,2]` output. The routine has no tensor inputs. MAP→MOGG has created `mogg.buffer.constant` for the embedded bytes and an input view over that pointer. Its output remains a distinct destination. [ConstantOp::lowerToMOGG:203](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:203)

```transform
BEFORE · schematic MOGG routine
%o = mogg.routine() -> f32[2,2] {                 # Graph has no user tensor inputs.
  body(%op:pointer,%ctx:context)                  # Output destination and context only.
  %osh = shape.create([2,2])                      # Describe the output dimensions.
  %O = tensor.create(%op,%osh,f32,Output,true)     # Writable output view.
  %cp = mogg.buffer.constant [1,2,3,4] : f32[2,2] # Embedded payload's pointer-producing op.
  %csh = shape.create([2,2])                      # Describe the constant dimensions.
  %C = tensor.create(%cp,%csh,f32,Input,false)     # Read-only constant view.
  %lsh = shape.create([2,2])                      # Describe the loop dimensions.
  call.foreach shape(%lsh) captures(%C,%O) {      # Copy the four logical elements later.
    %v = tensor.load %cArg[%i]                    # Read the constant at the loop index.
    tensor.store %oArg[%i] = %v                  # Store the same value to output.
  }
}
```

`outlineKernel` collects nested `BufferConstantOp`s and adds one trailing `!mogg.pointer` to the helper signature for each constant. Resources later enter through the runtime-facing MGP ABI; the pointer-based MOGG helper does not have that resource table. [OutlineRoutines.cpp:55](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:55)

```cpp
SmallVector<MOGG::BufferConstantOp> constants;
body.walk([&](MOGG::BufferConstantOp cst) { constants.push_back(cst); });

SmallVector<Type> stubArgTypes(argTypes);
stubArgTypes.append(constants.size(), MOGG::PointerType::get(ctx));
```

After `takeBody`, it rewires uses of each constant pointer inside the moved body to the newly added helper argument. It then moves the constant operation itself back into the routine's new call-site block and forwards its pointer. [OutlineRoutines.cpp:114](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:114) [OutlineRoutines.cpp:129](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:129)

```cpp
for (MOGG::BufferConstantOp cst : constants)
  cst.getPointer().replaceAllUsesWith(
      stubBody.addArgument(MOGG::PointerType::get(ctx), cst.getLoc()));
```

```transform
AFTER · schematic MOGG
%o = mogg.routine() -> f32[2,2] {                  # Still no user tensor input.
  NEW body(%op2:pointer,%ctx2:context)             # Same routine boundary types.
  %cp = mogg.buffer.constant [1,2,3,4] : f32[2,2] # Existing resource operation moved back here.
  mogg.call @F_copy(%op2,%ctx2,%cp)                # Constant pointer is a trailing helper argument.
}
mogg.func @F_copy(%op:pointer,%ctx:context,%cArg:pointer) { # New three-argument helper.
  MOVED %osh = shape.create([2,2])                # Output setup moves with the original block.
  MOVED %O = tensor.create(%op,%osh,f32,Output,true) # Same output view.
  MOVED %csh = shape.create([2,2])                 # Constant shape operation also moves.
  MOVED %C = tensor.create(%cArg,%csh,f32,Input,false) # Only pointer operand is redirected.
  MOVED %lsh = shape.create([2,2])                 # Loop domain operation also moves.
  MOVED call.foreach shape(%lsh) captures(%C,%O) { # Same copying computation.
    %v = tensor.load %cView[%i]                   # Still reads 1,2,3,4 from C.
    tensor.store %oView[%i] = %v                 # Future O still becomes [[1,2],[3,4]].
  }
  mogg.output                                     # Tensor output is written through %op.
}
```

The constant's 16 payload bytes remain embedded data. There is one constant operation at the call site, not a second copied constant in the helper. A helper argument was added; no input was added to the public graph. CPU alignment and runtime constant-buffer materialization belong to later MOGG→MGP work. [OutlineRoutines.cpp:55](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:55) [MAPOpsLowerableToMOGG.cpp:200](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MAPDialect/Dialect/MAPOpsLowerableToMOGG.cpp:200)

**TEST · input boundary for this source-derived outlining example.** [constant.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/constant.mlir:1) already supplies MAP IR and runs import → MAP→MOGG. Its checks require no input pointer group, one output pointer, `mogg.buffer.constant` containing `1,2,3,4`, a read-only constant view, and a foreach that copies it to the output. That test stops **before outlining**; the subsequent movement shown above is established by `outlineKernel`, not by an unexecuted expected-output claim about that test. [constant.mlir:12](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/constant.mlir:12) [constant.mlir:25](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/constant.mlir:25)

The plain/fused/dynamic tests in [infra-outline-routines.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-outline-routines.mlir:1) separately exercise body movement and forwarding. They do not assert this constant-specific branch. Attention's routine consumes three graph inputs and has no such embedded Q/K/V data constant. [[entry:57]]

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-trace-return-and-cleanup" markdown="1">
<summary>Example 8 · How does tracing retain a helper's returned shape, and which setup operations can later be shared?</summary>
<div class="qna-answer" markdown="1">

Ordinary attention returns no value from its helper because O is written through a pointer. A shape helper instead ends in `mogg.output %shapeValue`. Outlining obtains helper result types from that terminator's operands, so it can preserve a `!mogg.shape_func_result<4>` return or an opaque by-value result as well. [OutlineRoutines.cpp:65](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/OutlineRoutines.cpp:65)

```cpp
auto bodyOutput =
    dyn_cast<MOGG::OutputOp>(body.empty() ? nullptr : &body.back());
SmallVector<Type> stubResultTypes;
if (bodyOutput)
  llvm::append_range(stubResultTypes, bodyOutput.getOperandTypes());
```

Use the earlier gather-shape example: input `[1,2,3]`, indices `[4,5]`, axis `-2`, illustrative returned dimensions `[1,4,5,3]`. The trace wrapper must forward those four dimension values intact. [shape-func-kgen-params.mlir:11](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/shape-func-kgen-params.mlir:11)

```transform
BEFORE · schematic outlined MOGG helper
func @F_shape(%ip:pointer,%jp:pointer,%ctx:context) -> shape_result<4> {
  %I = tensor.create(%ip,shape=[1,2,3],Input,false) # Initial view construction.
  %J = tensor.create(%jp,shape=[4,5],Input,false)   # Second input view construction.
  %s = call.shape @gather_shape_fn(%I,%J)          # First CallLike operation.
       params={axis=-2,output_rank=4}             # Same shape-function arguments.
  mogg.output %s                                 # Return its dimension tuple by value.
}
```

`WrapFuncsWithTrace` skips non-MOGG-ABI functions. For a MOGG helper, it collects operations from the first `CallLike` onward, excluding the terminator and any existing `WithTraceOp`. Setup before that first call stays outside; operations after it move with the dispatch. If the collected list is empty, the function is unchanged. [WrapFuncsWithTrace.cpp:38](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/WrapFuncsWithTrace.cpp:38) [WrapFuncsWithTrace.cpp:48](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/WrapFuncsWithTrace.cpp:48)

The new trace operation has result types matching the original terminator operands. `mogg.yield` forwards those operands from the nested region, and the outer terminator is rewired to the trace results. This unchanged excerpt implements the final forwarding step. [WrapFuncsWithTrace.cpp:65](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/WrapFuncsWithTrace.cpp:65)

```cpp
builder.setInsertionPointToEnd(&tracedBlock);
MOGG::YieldOp::create(builder, loc, terminator->getOperands());
terminator->setOperands(withTrace.getOutputs());
```

```transform
AFTER · schematic MOGG helper
func @F_shape(%ip:pointer,%jp:pointer,%ctx:context) -> shape_result<4> {
  %I = tensor.create(%ip,shape=[1,2,3],Input,false) # Same setup before trace boundary.
  %J = tensor.create(%jp,shape=[4,5],Input,false)   # Same setup and input references.
  %traced = mogg.with_trace -> shape_result<4> {   # New result-bearing instrumentation region.
    %s = call.shape @gather_shape_fn(%I,%J)        # Original call moves here.
         params={axis=-2,output_rank=4}           # Same shape computation.
    mogg.yield %s                                # Forward [1,4,5,3] out of the trace.
  }
  mogg.output %traced                             # Forward the same four values to the caller.
}
```

**TEST · pass boundaries.** The referenced [shape-func-kgen-params.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/Conversion/MAPToMOGG/shape-func-kgen-params.mlir:1) stops after MAP→MOGG and checks axis/rank forwarding; its input above supplies the shape routine for this source-derived tracing example. For tracing itself, [infra-wrap-funcs-with-trace.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-wrap-funcs-with-trace.mlir:1) starts with MO addition, runs MAP elementwise formation, MAP→MOGG, outlining and trace wrapping. It checks `with_trace → call.foreach → yield`, followed by outer `mogg.output`. Its constant vector `[1,2,3,4]` added to itself would yield `[2,4,6,8]`; the test checks the instrumentation structure. [infra-wrap-funcs-with-trace.mlir:6](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/infra-wrap-funcs-with-trace.mlir:6)

**TEST · returning through nested wrappers.** [codegen-map-abi-trace-try-shape-emit-mojo.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-trace-try-shape-emit-mojo.mlir:1) starts with an MO custom call whose shape function can raise, then runs the pre-JIT graph pipeline with Mojo text dumping. Its checks require a returned `IndexList` to pass from the shape-function call through the trace's temporary, through the later try/except wrapper's temporary, and finally through the function return. That expected emitted source checks forwarding across both regions; it supplies no runtime trace sample. [codegen-map-abi-trace-try-shape-emit-mojo.mlir:7](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/codegen-map-abi-trace-try-shape-emit-mojo.mlir:7)

**The subsequent module cleanup can share initial shape descriptions.** `mogg.shape.create` is a pure operation. For the actual attention helper, four static `[32,32]` constructors with no operands are eligible for CSE; distinct Q/K/V/O tensor pointers still keep their tensor views distinct. The module scope matters because the helper is a sibling of `mo.graph`. [MOGGOps.td:1082](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGOps.td:1082) [GraphCompiler.cpp:377](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:377)

```transform
BEFORE eligible cleanup
%sq = shape.create([32,32])                      # Same pure shape expression.
%sk = shape.create([32,32])                      # Duplicate expression.
%Q = tensor.create(%qp,%sq,Input,false)          # Distinct data pointer Q.
%K = tensor.create(%kp,%sk,Input,false)          # Distinct data pointer K.

AFTER eligible cleanup
%s = shape.create([32,32])                       # One shared shape description.
%Q = tensor.create(%qp,%s,Input,false)           # Retain Q's pointer.
%K = tensor.create(%kp,%s,Input,false)           # Retain K's pointer.

Skipped comparison
shape.create(%n,[dyn,32]) and shape.create(%m,[dyn,32]) # Different SSA dimension operands remain distinct.
```

These CSE diagrams describe eligibility, not a captured compiler dump. A function containing no call is also skipped by trace wrapping; an MGP-ABI function uses another instrumentation path. Neither branch changes the attention computation or samples a GPU event during compilation. [WrapFuncsWithTrace.cpp:35](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/WrapFuncsWithTrace.cpp:35) [WrapFuncsWithTrace.cpp:57](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/WrapFuncsWithTrace.cpp:57)

</div>
</details>

<details class="qna-cell" id="qna-mogg-example-rowwise-reduction" markdown="1">
<summary>Example 9 · What does rowwise legalization add to a sum-of-squares reduction, and what malformed bodies does it reject?</summary>
<div class="qna-answer" markdown="1">

This pass runs first in the MOGG stage. Use the test's visible graph `Y = reduce_add(cast_f32(X) * cast_f32(X)) + B`, with `X:bf16[8,128]`, `B:f32[8,1]` and `Y:f32[8,1]`. For an illustrative first row `[1,2,3,0,…,0]` and bias `5`, the required first output is `1+4+9+5=19`. Set the other rows and biases to zero to make the remaining seven outputs zero. [LegalizeRowwiseKernels/legalize.mlir:19](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/legalize.mlir:19)

```transform
BEFORE · schematic MOGG after MAP→MOGG
call.rowwise shape=[8,128], axis=1, captures(%X,%B,%O) { # Rowwise body lacks legalization metadata.
  %acc = rowwise.reduce @ReduceSum(%i) -> simd<f32> {   # Fold over one row's 128 elements.
    %x = tensor.load %xArg[%i] -> simd<bf16>            # Read the primary input at the element index.
    %a = call.elementwise @Cast(%x) -> simd<f32>       # Accumulate using f32 values.
    %sq = call.elementwise @Mul(%a,%a)                 # Square each element.
    mogg.yield %sq                                    # Contribute to the sum.
  }
  %b = tensor.load %bArg[%row]                         # Read one bias per row.
  %y = call.elementwise @Add(%acc,%b)                  # Add bias after reduction.
  tensor.store %oArg[%row] = %y                       # Write the row's one output value.
}
```

The umbrella pass schedules four subpasses in order: select primary input, bind monoids, group row stores, then set launch parameters. `selectPrimaryInput` chooses the first capture loaded by the first fold at its own element index. It replaces matching loads in all folds with `RowwisePrimaryOp` and records the capture number and dtype. [LegalizeRowwiseKernels.cpp:315](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:315) [LegalizeRowwiseKernels.cpp:78](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:78)

```cpp
auto value =
    RowwisePrimaryOp::create(builder, load.getLoc(), load.getType());
dtype = value.getType().getElementType();
load.replaceAllUsesWith(value.getResult());
load.erase();
```

Here that selects capture 0, X, with `dtype=bf16`. One fold reads it, so `is_cached` remains false. `bindMonoid` uses `ReduceSum`'s declared output field to record `fields=["acc"]` and parameter `dtype=f32`, while the widest yielded contribution type gives the rowwise accumulator dtype `f32`. [LegalizeRowwiseKernels.cpp:113](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:113) [LegalizeRowwiseKernels.cpp:128](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:128) [LegalizeRowwiseKernels.cpp:170](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:170)

```transform
AFTER · schematic legalized MOGG
call.rowwise shape=[8,128], axis=1, captures(%X,%B,%O)  # Same tensors and reduction domain.
  attrs={primary=0,dtype=bf16,accum=f32,cols=128,         # Record input/accumulator/extent metadata.
         num_phases=1,dtype_size=0,simd_monoid=@ReduceSum} {
  %acc = rowwise.reduce @ReduceSum(%i) -> simd<f32>       # Same sum operation.
    fields=["acc"], params={dtype=f32} {
    %x = mogg.rowwise.primary -> simd<bf16>              # Read the selected primary stream.
    %a = call.elementwise @Cast(%x) -> simd<f32>         # Same conversion.
    %sq = call.elementwise @Mul(%a,%a)                   # Same square.
    mogg.yield %sq                                      # Same contributions 1,4,9,0,….
  }
  %b = tensor.load %bArg[%row]                           # Same per-row bias.
  %y = call.elementwise @Add(%acc,%b)                    # First illustrative row remains 14+5.
  mogg.rowwise.emit {                                    # New region groups row-result writes.
    tensor.store %oArg[%row] = %y                       # Existing store moves into this region.
  }
}
```

`groupRowStores` moves stores whose index is the row argument into one `RowwiseEmitOp`; it does not change their values or create an output buffer. `setLaunchParams` then rejects any store outside such an emit region, sets the supported reduction phase metadata, and reads the known axis extent to set `cols=128`. [LegalizeRowwiseKernels.cpp:188](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:188) [LegalizeRowwiseKernels.cpp:222](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:222)

The cost flag is a source-defined heuristic: an expensive elementwise operation contributes 16, an ordinary one contributes 1, and the default threshold is 16. The cast and multiply in this fold contribute 2; the add after the fold is not counted as per-element work. A fold containing `exp` reaches the threshold. These values classify a body for code generation; they are not measured cycle counts or speedups. [LegalizeRowwiseKernels.cpp:204](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:204) [LegalizeRowwiseKernels.cpp:243](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:243) [MOGGPasses.td:235](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/include/GraphCompiler/MOGGDialect/MOGGPasses.td:235)

**TEST · successful legalization.** [LegalizeRowwiseKernels/legalize.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/legalize.mlir:1) starts at MO, runs import → MO→MAP → kernel fusion → structured-body outlining → MAP→MOGG → rowwise legalization. Its `CHECK-NEXT` sequence requires the primary op, cast, multiply, yield, bound monoid fields, post-fold bias/add, and the grouped output store. The second graph checks `computationally_expensive` for a sum of exponentials. [LegalizeRowwiseKernels/legalize.mlir:5](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/legalize.mlir:5) [LegalizeRowwiseKernels/legalize.mlir:27](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/legalize.mlir:27)

| Malformed input at legalization | Native guard / outcome |
| --- | --- |
| rowwise body has no fold | `selectPrimaryInput` reports `has no fold` |
| first fold loads only at the row index, not its element index | no eligible primary input; reports an error |
| ArgMax declares an si32 index result against a fixed si64 field | monoid result dtype validation fails |
| one shared monoid dtype parameter is bound to f32 and f16 | conflicting parameter binding fails |
| fold writes every element outside `rowwise.emit` | unsupported body shape; launch-parameter step fails |

**TEST · rejected bodies.** [LegalizeRowwiseKernels/errors.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/errors.mlir:1) directly supplies MOGG calls and declarations, splits the file into independent cases, and runs `--mogg-legalize-rowwise-kernels -verify-diagnostics`. Its `expected-error` annotations check the diagnostics above rather than a successful output transformation. The guards live in primary selection, monoid binding and launch validation. [errors.mlir:3](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/errors.mlir:3) [errors.mlir:23](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/errors.mlir:23) [errors.mlir:46](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/errors.mlir:46) [errors.mlir:70](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/errors.mlir:70) [errors.mlir:109](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MOGGDialect/Transforms/LegalizeRowwiseKernels/errors.mlir:109) [LegalizeRowwiseKernels.cpp:72](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:72) [LegalizeRowwiseKernels.cpp:128](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:128) [LegalizeRowwiseKernels.cpp:230](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MOGGDialect/Transforms/LegalizeRowwiseKernels.cpp:230)

The actual fused-attention routine contains an ordinary opaque call. Its online-softmax reductions live inside the authored Mojo implementation, so this MOGG rowwise pass finds no `call.rowwise` to legalize in that routine. [[kernel:99]] [[map-lowering:675]]

</div>
</details>
