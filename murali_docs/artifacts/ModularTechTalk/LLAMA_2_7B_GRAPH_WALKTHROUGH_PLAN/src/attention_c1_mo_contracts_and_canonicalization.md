## QnA 1 · What enters the MO stage, and which object changes?

**MO describes the computation using tensor values, types and operation contracts.** For this example, the recorded computation is one custom call followed by the graph output. Q, K and V have known types; their numerical arrays are created only after `compile` and `init`. [[entry:57]] [[entry:68]] [[entry:74]] [[entry:78]]

```state
G1                         caller's Python Graph
└─ module                  retained source module
   └─ mo.graph @fused_attention
      ├─ inputs: Q, K, V : T0
      ├─ OP1 : mo.custom → O : T0
      └─ mo.output O

IR1 = native clone of G1.module
└─ mo.graph @fused_attention       ← MO passes rewrite this copy
   ├─ arguments: %q, %k, %v : T0
   ├─ custom symbol = "modular_ops::fused_attention_custom"
   ├─ parameters = {BD = 8 : index, BN = 16 : index}
   ├─ device = gpu:0
   └─ declared result: T0

T0 = !mo.tensor<[32,32],f32,gpu:0>  ← compact type notation
C1 = CompiledModel being prepared; IR1 is its compilation work
```

Python seeds P1's declarations before the native module clone. The compiler then rewrites IR1; these rewrites do not replace `G1._mlir_op`, `G1._graph_body` or its TensorValue wrappers. Any unused initial chain in the clone may be cleaned up, so the diagram concentrates on the live attention path. [[api:1042]] [[frontend:377]] [[gc:334]] [[graph:887]]

## QnA 2 · At gcPipelineImpl → buildMoPipeline, what do the shape/fold/fusion/device settings copied into MoPipelineOptions actually control? Explain what gates and pass objects are and when the transformations execute. Give multiple examples beyond fused attention, covering symbolic reshape, shape arithmetic and materialization, full/partial constant folding, branches and external constants, LayerNorm and matmul/bias fusion (including rejected matches), mutable load/store cleanup, target metadata and dtype validation, canonicalization and CSE. For every example, show the complete before/after code, concrete values and half-line comments explaining every operation. Clarify that tensor<[], si64> is a scalar and whether broadcasting scalar=9 to [N+1,M+2], with N=3 and M=5, produces a 4×7 tensor filled with 9. Alongside each source reference, explain what the test checks and how the linked Python/C++ implementation achieves that transformation; interleave these code explanations within their corresponding examples rather than putting them in a separate section. Retain the complete combined answer and all 16 examples. {#qna-mo-2-examples}

Here is the combined explanation, with **what each transformation does**, **before/after examples**, and **short comments explaining each operation**.

These examples use simplified IR/pseudocode. During C1 compilation, the transformations operate on `IR1`, the cloned compilation module.

A **gate** is a condition such as `if (opts.constantFold)` that decides whether to schedule a pass. A **pass object** implements a compiler transformation or analysis. `pm.addPass(...)` schedules it; `pm.run(IR1)` executes the configured pipeline. ([mlir.llvm.org](https://mlir.llvm.org/docs/PassManagement/))

The options copied at [GraphCompiler.cpp:508](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:508) control these choices:

| Setting | Effect in the MO pipeline |
|---|---|
| `useShapeInference` | Adds shape propagation, cleanup, and parameter verification. |
| `constantFold` | Adds constant-folding passes at several stages. |
| `usePatterns` | Adds pattern fusion, followed by shape propagation and cleanup. |
| `fuseJitKernels` | Enables removal of redundant mutable loads/stores. |
| Device label and ID | Configure the target device for pattern fusion. |
| Target triple and architecture | Configure device validation; that pass is scheduled unconditionally. |

The scheduling is implemented in [buildMoPipeline()](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:219). An enabled pass still needs suitable operations and matching conditions before it can rewrite anything.

**How that code works:** `gcPipelineImpl()` copies the compilation options into `MoPipelineOptions`. `buildMoPipeline()` then reads those fields and creates the appropriate pass objects:

```text
useShapeInference=True → schedule shape propagation and cleanup.
constantFold=True      → schedule constant-subgraph evaluation.
usePatterns=True       → schedule operation-pattern matching and rewriting.
fuseJitKernels=True    → schedule redundant mutable load/store removal.
target triple/arch    → configure the always-scheduled device validator.
```

Finally, [runGCPipeline():638](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:638) calls `pm.run(moduleOp)`, which invokes the scheduled implementations. Copying a flag or scheduling a pass does not perform the transformation.

The linked `.mlir` files below contain **test inputs and expected compiler output**. Their C++ and Python implementations explain how the transformations happen. I inspected these sources and expectations; I did not run the tests.

The test directives mean:

| Directive | Meaning |
|---|---|
| `RUN:` | Gives the test command, often `mo-opt … \| FileCheck …`. |
| `CHECK-LABEL:` | Identifies the graph/function being checked. |
| `CHECK:` | Requires a matching line later in the output. |
| `CHECK-SAME:` | Requires a match on the same line as the preceding match. |
| `CHECK-NEXT:` | Requires a match on the immediately following line. |
| `CHECK-NOT:` | Prohibits a match between the surrounding positive checks. |
| `CHECK-DAG:` | Allows the specified matches in any order. |

See the [FileCheck documentation](https://llvm.org/docs/CommandGuide/FileCheck.html). The implementation flows added below are simplified; their links point to the actual source.

First, a correction to the tensor notation:

```text
tensor<[], si64>       # Known rank-0 shape: one scalar integer.
tensor<[2], si64>      # Rank-1 vector containing two integers.
tensor<[N,M], si64>    # Rank-2 tensor; axis lengths are N and M.
```

Thus, `[]` means **scalar shape**, rather than unknown shape. Also, a shape vector containing `[4,7]` has its own tensor shape `[2]`: it contains two numbers describing another tensor’s dimensions.

With **shape inference**, the compiler determines dimensions and relationships between dimensions. Those dimensions can remain symbolic.

**1. Flattening a symbolic tensor.** Suppose the input has shape `[B,4]`:

```text
# BEFORE

x : tensor<[B,4], si32>           # B rows, four columns; B×4 elements.
y = reshape(x,[-1])               # Flatten; -1 requests an inferred output length.
y : tensor<[Dout], si32>          # The output length is initially represented by Dout.

# AFTER

y = mo.static.reshape(x)          # Preserve elements; use the inferred result shape.
y : tensor<[mul_no_wrap(B,4)], si32> # Output length uses the non-wrapping product B×4.
```

For example:

```text
B = 5  → input [5,4] → output [20]
B = 8  → input [8,4] → output [32]
```

**What changes:** the result type carries `B*4` directly. The reshape preserves the tensor elements; it changes their logical shape.

The checked output uses this dimension expression in [shape-inference-e2e.mlir:786](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/shape-inference-e2e.mlir:786).

**How the code achieves this:** for the actual MAX Python path, **Python resolves `-1` during graph construction**, before the native compiler receives the reshape. The before/after example above is schematic.

In [reshape.py:78](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/graph/ops/reshape.py:78), the implementation permits one `-1`, then calculates:

```text
total = product(x.shape)                   # B×4 input elements.
known = product(requested dims except -1)  # Empty product = 1.
missing = total // known                   # (B×4)//1 = B×4.
replace -1 with missing                    # Requested shape becomes [B×4].
record rmo.ReshapeOp                       # Result shape is already resolved.
```

The `_product()` helper starts multiplication at `Dim(1)`, which explains why the empty product is `1`.

Then [LowerReshape:710](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/RMODialect/Transforms/LegalizeRMOOps.cpp:710) performs this replacement:

```text
rmo.ReshapeOp
  → MO::Static::ReshapeOp(
        existing result type,   # Retains the resolved [B×4] shape.
        existing input)         # Retains the same tensor elements.
```

The referenced `.mlir` test already contains a static reshape with `mul_no_wrap(D0,4)`; it checks symbolic result/signature handling. Native [materializeReshapeShape():1192](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/ShapeFunctions.cpp:1192) verifies equal element counts and explicitly rejects a target shape that still contains literal `-1`.

**2. Moving shape arithmetic into a broadcast’s result type.**

```text
# BEFORE: N=3, M=5, scalar=9

dummy : tensor<[N,M], si64>       # Rank-2 input: three rows, five columns.
scalar : tensor<[], si64>         # Rank-0 input containing one integer, 9.

s0 = shape_of(dummy)              # Shape vector: values [3,5], vector shape [2].
c  = constant([1,2])              # Embed the known two-element vector [1,2].
s1 = add(s0,c)                    # Elementwise addition: [3+1,5+2] = [4,7].
y  = broadcast_to(scalar,s1)      # Produce a logical 4×7 tensor filled with 9.

# AFTER

y = mo.static.broadcast_to(scalar) # Target dimensions come from the result type.
y : tensor<[N+1,M+2], si64>        # Result shape [4,7]; all 28 elements are 9.
```

Your broadcasting interpretation is correct: the scalar value applies to every logical output element. Scalar broadcasting is the simplest broadcasting case. ([numpy.org](https://numpy.org/doc/stable/user/basics.broadcasting.html))

```text
Logical result:

[
  [9,9,9,9,9,9,9],
  [9,9,9,9,9,9,9],
  [9,9,9,9,9,9,9],
  [9,9,9,9,9,9,9]
]
```

**What changes:** the explicit shape-tensor arithmetic disappears from the checked result. Its calculation becomes expressions in the output type. The broadcast still produces tensor values when the model runs.

For this operation, `static` allows symbolic target dimensions such as `N+1`; they need not all be literal numbers. See [StaticizeOperations/add.mlir:28](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/StaticizeOperations/add.mlir:28).

**How the code achieves this:** [tryStaticizeImpl():349](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/PropagateShapesPass.cpp:349) recognizes `mo.shape_of(dummy)` and reads the shape from `dummy`’s type:

```text
shape_of(dummy) → ShapeAttr([N,M])  # Uses dimension metadata.
constant([1,2]) → ShapeAttr([1,2])  # Uses the embedded integers.
```

[AddOp::tryStaticize():210](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Staticization.cpp:210) invokes an elementwise helper. That helper takes corresponding entries from the two vectors and constructs KGEN addition expressions:

```text
index 0: KGEN.Add(N,1) → N+1
index 1: KGEN.Add(M,2) → M+2

result: ShapeAttr([N+1,M+2])
```

The broadcast declares its **target-shape operand** as data-dependent in [ShapeMaterialization.cpp:75](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/ShapeMaterialization.cpp:75). Shape inference therefore needs that operand’s dimension **values**, such as `[4,7]`. For the scalar input, it needs only the shape `[]`; the scalar’s value can remain unknown.

[PropagateShapesPass.cpp:883](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/PropagateShapesPass.cpp:883) supplies these attributes to the shape implementation, which resolves the output dimensions.

Once the output parameters are resolved, [BroadcastToOp::canonicalize():455](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Canonicalize.cpp:455) replaces the operation with `Static::BroadcastToOp`, passing only the scalar input and the resolved result type. Cleanup removes the now-unused shape calculations.

The test’s `CHECK-NEXT` lines require that static broadcast and its output, with the dimension expressions recorded in the type.

**3. Removing a tensor dependency when only its shape is needed.**

```text
# BEFORE

x : tensor<[B,64], f32>           # Rank-2 tensor with symbolic batch dimension B.
shape = mo.shape_of(x)            # Return the dimension values [B,64].

# AFTER

s = mosh.param.to_value<[B,64]>   # Build a shape value from dimension parameters.
shape = mo.shape.to_tensor(s)     # Convert that shape value into an integer tensor.
```

The dependency changes:

```text
BEFORE: x ──► shape_of ──► shape consumer

AFTER:  shape parameters ──► shape materialization ──► shape consumer
```

**What changes:** the shape calculation no longer has an explicit SSA dependency on `x`. Equivalent information comes from its type and dimension parameters.

This replacement introduces two operations in place of one. Its purpose is to improve the dependency structure. The implementation explains this directly in [PropagateShapesPass.cpp:1163](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/PropagateShapesPass.cpp:1163).

**How that implementation works:** the code immediately below the comment performs the replacement:

```text
inputShape = op.input.type.shape           # Obtain [B,64] from the type.
shConst = MOSH::ParamToValueOp(inputShape)  # Produce a shape value.
toTensor = Shape::ToTensorOp(shConst)      # Produce the integer shape tensor.
replace old shape_of result uses          # Consumers now use toTensor.
erase old shape_of operation              # Remove its explicit input edge.
```

The original shape-tensor result type is passed to `ToTensorOp`, so consumers retain the expected type. The shape dimensions still come from the graph’s parameters, but the shape calculation no longer consumes `x` as a tensor operand.

With **constant folding**, the compiler precomputes eligible expressions whose inputs are known constants and replaces their results with constant data.

**4. Folding an entire constant arithmetic sequence.**

```text
# BEFORE

a = constant([0,1])               # Embed a known two-element integer vector.
b = constant([6,3])               # Embed another known integer vector.
m = mul(a,b)                      # Elementwise multiplication: [0,3].
d = sub(m,a)                      # Elementwise subtraction: [0,2].
y = add(d,m)                      # Elementwise addition: [0,5].

# AFTER

y = constant([0,5])               # Embed the precomputed result.
```

**What changes:** `mul`, `sub`, and `add` disappear. One constant operation stores their final result. The result retains shape `[2]` and dtype `si32`.

This exact arithmetic example appears in [constant-fold-mogg.mlir:108](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/constant-fold-mogg.mlir:108).

**How the code achieves this:** [constantSubgraphs():432](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:432) starts from embedded constants and grows connected operation groups when all their operands are constant.

For this example:

```text
a,b are constants → admit mul.
m,a are constants → admit sub.
d,m are constants → admit add.
```

[ConstantFoldPass::runOnOperation():815](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:815) then follows this sequence:

```text
constantSubgraphs(...)     # Discover eligible constant computations.
policy->shouldFold(...)   # Apply the configured folding policy.
cloneSubgraphs(...)       # Copy them into temporary graphs.
lowerSubgraphs(...)       # Compile those graphs for host CPU execution.
executeModel(...)         # Calculate their results during compilation.
replaceConstantOps(...)   # Install constants and redirect existing uses.
```

This pass actually computes the eligible tensor expressions through a temporary executable. [lowerSubgraphs():757](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:757) selects host execution and disables recursive constant folding.

Finally, [replaceConstantOps():669](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:669) creates the `[0,5]` constant with the existing result type, redirects uses, and erases the old arithmetic. Internal operations are erased in reverse dependency order.

The test expects the constant immediately followed by `mo.output`, demonstrating the intended disappearance of the arithmetic sequence.

**5. Precomputing a weight transpose while retaining runtime matmul.**

```text
# BEFORE

W = constant([[1,2,3],[4,5,6]])    # Known matrix with shape [2,3].
WT = transpose(W,[1,0])           # Swap row/column axes; result shape [3,2].
y = matmul(runtime_x,WT)          # Matrix multiplication: row–column dot products.

# AFTER

WT = constant([[1,4],[2,5],[3,6]]) # Store the already-transposed weight values.
y = matmul(runtime_x,WT)          # Runtime input still requires matrix multiplication.
```

For example:

```text
runtime_x = [[2,4,6]]

First output  = 2×1 + 4×2 + 6×3 = 28
Second output = 2×4 + 4×5 + 6×6 = 64

y = [[28,64]]
```

**What changes:** the transpose becomes precomputed constant contents. The matmul remains because `runtime_x` arrives during execution.

The repository checks constant broadcasts and a transpose folding while `mo.matmul` remains in [constant-fold.mlir:17](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/constant-fold.mlir:17).

**How the code achieves this:** the same constant-subgraph discovery inspects each operand’s defining operation in [ConstantFold.cpp:478](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:478).

```text
W and permutation are known → transpose belongs to a constant subgraph.
runtime_x is a block argument → it has no defining operation.
matmul consumes runtime_x    → matmul is classified as nonconstant.
```

The transposed weight is a boundary result: it is produced by the constant subgraph and consumed by the surviving matmul. Replacement redirects that matmul operand to the new constant.

The referenced test actually uses broadcasted `-1.0` weights. Its checks require a `[3,2]` constant, followed by matmul and output. The varied matrix values above illustrate the same transpose transformation.

**6. Folding computations inside branches without selecting a branch.**

```text
# BEFORE

c = constant([0,3])               # Known vector shared by both branches.
if runtime_condition:            # Select a branch using a runtime Boolean.
    y = mul(c,c)                  # Square each element: [0,9].
else:
    y = add(c,c)                  # Double each element: [0,6].

# AFTER

if runtime_condition:            # The runtime branch decision remains.
    y = constant([0,9])           # First branch uses its precomputed result.
else:
    y = constant([0,6])           # Second branch uses its precomputed result.
```

**What changes:** both branch computations become constants. The conditional remains because its condition is unknown during compilation.

See [constant-fold-mogg.mlir:53](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/constant-fold-mogg.mlir:53).

**How the code achieves this:** [isMaybeConstantFoldable():174](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:174) rejects operations that own regions:

```text
mo.if owns branch regions → do not evaluate the whole conditional.
```

However, constant discovery walks nested operations. It can find the constant and its arithmetic users inside the branches:

```text
then branch: mul([0,3],[0,3]) → constant [0,9].
else branch: add([0,3],[0,3]) → constant [0,6].
```

Replacement redirects the branches’ `mo.yield` operands to those constants. The runtime condition and `mo.if` remain.

The relevant checks require both constant results, the conditional, and the corresponding yields. The schematic example places constants inside the branches; their precise placement in compiler output can differ while the branch results remain equivalent.

**7. An external constant can prevent this folding route.**

```text
# BEFORE

W = mo.constant.external("W")     # Reference external data; contents aren't embedded here.
two = constant(2)                 # Embed the known scalar multiplier.
y = mul(W,two)                    # Multiply each weight element by two.

# AFTER

W = mo.constant.external("W")     # The external-data reference remains.
two = constant(2)                 # The multiplier remains known.
y = mul(W,two)                    # This pass cannot precompute unavailable W contents.
```

**What changes:** the dependent multiplication remains in this tested route. A constant reference alone does not supply the data needed to evaluate it.

See [constant-fold-external.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/constant-fold-external.mlir:1). Unresolved parameters and folding policies can also prevent folding, as illustrated in [constant-fold.mlir:4](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/constant-fold.mlir:4).

**How the code enforces this:** [ConstantFold.cpp:181](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ConstantFold.cpp:181) explicitly excludes `ConstantExternalOp`:

```text
operation is ConstantExternalOp → classify it as nonconstant.
mul depends on that operation   → classify mul as nonconstant.
```

Although `two` is known, the “all operands constant” condition fails for `mul`.

The nearby code also scans operand types, result types, and attributes for `ParamDeclRefAttr`. Remaining parameter references prevent this folding route. That explains the other linked test’s instruction to resolve the shape parameters first.

After discovery, `policy->shouldFold(...)` can further filter candidate subgraphs. Eligibility and the configured folding policy both affect whether evaluation occurs.

With **fusion**, the compiler replaces a matched operation sequence with a combined representation. Enabling a fusion pass gives it an opportunity to inspect the graph; its individual matching conditions still apply.

**8. Recognizing LayerNorm from arithmetic.**

Suppose `x` has shape `[4,3,2]`, and `gamma` and `beta` each have shape `[2]`:

```text
# BEFORE

eps = constant(0.00001)           # Small positive value for numerical stability.
mu = reduce_mean(x,last_axis,
                 keep_dims=True) # Average each pair; result shape [4,3,1].
d = x-broadcast_to(mu,[4,3,2])     # Repeat each mean within its pair, then subtract.
var = reduce_mean(d*d,last_axis,
                  keep_dims=True) # Mean squared deviations: variance.
inv = rsqrt(var+eps)              # Compute 1/sqrt(variance+epsilon).
norm = d*broadcast_to(inv,[4,3,2]) # Normalize each pair using its inverse deviation.
y = norm*gamma+beta               # Scale and shift; channel vectors broadcast.

# AFTER

y = mo.reduce.layer_norm(
    x,gamma,beta,eps)             # Replace matched arithmetic with one LayerNorm operation.
```

**What changes:** the matched reductions, broadcasts, and elementwise arithmetic become a dedicated LayerNorm operation. The result shape remains `[4,3,2]`; numerical normalization still happens for the runtime input.

The test runs shape inference, symbolic optimization, and pattern fusion together in [layernorm.mlir:1](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/layernorm.mlir:1).

**How the code recognizes this sequence:** [LayerNormPattern:996](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:996) starts at the final addition and matches backward through its defining operations.

Its nested matchers capture:

```text
final add                         → beta.
preceding multiply                → gamma and normalized values.
subtract / square / reductions    → input and variance calculation.
sqrt or rsqrt path                → epsilon and normalization factor.
```

`m_Op<...>` checks operation kinds. `m_Any(&input)` captures an existing SSA value. Commutative matchers allow operands of additions and multiplications to appear in either order.

The implementation checks that both reductions form **means over the last axis**. For `[4,3,2]`, the reduced extent is `2`:

```text
reduce_add(...)/2       → recognized as a mean.
reduce_add(...)*0.5     → also recognized as a mean.
```

This specific matcher requires **rank-3 input and CPU placement**.

At [Fusion.cpp:1154](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:1154), `replaceOpWithNewOp<MO::LayerNormOp>` retains the original result types and supplies the captured `input`, `gamma`, `beta`, and `eps`. Uses of the final addition become uses of LayerNorm’s result; unused arithmetic is removed during rewriting and cleanup.

The test’s earlier shape inference and symbolic optimization prepare operation forms that the matcher recognizes.

**9. Combining matrix multiplication and bias addition.**

```text
# BEFORE
# X=[256,1536], W=[8192,1536], bias=[8192].
# Inputs/output are BF16 on gpu:0.

m = matmul(X,W,transpose_b=True)  # Multiply X by Wᵀ; result shape [256,8192].
b = broadcast_to(bias,[256,8192]) # Repeat the bias vector across 256 rows.
y = add(m,b)                      # Add the corresponding bias to every element.

# AFTER

y = mo.composite.matmul_add(
    X,W,bias)                    # Combined matmul/bias operation; transpose choice retained.
```

**What changes:** the matched matmul/add sequence becomes a composite operation. The explicit bias broadcast disappears; the composite receives the original `[8192]` vector.

See [matmul-add-pattern-fusion.mlir:32](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/PatternFusion/matmul-add-pattern-fusion.mlir:32).

The test’s `CHECK-NOT` prohibits the explicit bias broadcast, and its positive checks require the composite operation and its output.

**This particular pattern requires** BF16 inputs/output, GPU architecture `sm_100a`, static `N` and `K`, and suitable use/layout conditions. These checks appear in [Fusion.cpp:4440](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:4440).

**How those checks and the rewrite work:**

```text
X dtype, W dtype, matmul result dtype → must be BF16.
result dimension N = 8192            → must be static.
input dimension K = 1536             → must be static.
```

The matcher identifies one add operand as the matmul result and the other as the bias/residual. It checks device, architecture, use counts, and surrounding operations.

For a suitable rank-1 bias broadcast, this assignment selects the original vector:

```text
residual = broadcastOp.getInput()  # Select bias[8192] before its broadcast.
```

At [Fusion.cpp:4554](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:4554), the implementation creates the composite from the existing matmul operands and selected bias:

```text
create FusedMatmulAddOp(
    X, W, bias,
    transpose_b=True)            # Preserve the original transpose choice.

replace final add with its result
```

The old matmul and broadcast become unused and can be erased.

**10. Skipping fusion when the matmul result has multiple users.**

```text
# BEFORE

m = matmul(X,W)                   # Produce the matrix multiplication result.
y = add(m,residual)               # First use of m: elementwise addition.
output(m,y)                       # Second use of m: return it separately.

# AFTER

m = matmul(X,W)                   # Retain the separately needed matmul result.
y = add(m,residual)               # This particular fusion does not apply.
output(m,y)                       # Both graph results remain.
```

**What changes:** this pattern leaves the sequence unchanged because it requires the matmul result to have one use.

The negative case is checked in [matmul-add-pattern-fusion.mlir:44](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/PatternFusion/matmul-add-pattern-fusion.mlir:44).

**How the code declines the transformation:** the decisive guard is at [Fusion.cpp:4505](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:4505):

```cpp
if (!matmulOp.getResult().hasOneUse()) {
  return rewriter.notifyMatchFailure(addOp,
                                     "matmul result has multiple uses");
}
```

Here, `m` is used by both `add` and `output`, so `hasOneUse()` is false.

`notifyMatchFailure` declines this particular rewrite. It does not report a compilation failure. No composite replacement is created, explaining the test’s expected surviving matmul and add.

**11. Removing an intermediate buffer store and reload.**

This explains the cleanup enabled by `fuseJitKernels`. Here, `load`/`store` abbreviate `mo.mutable.load`/`mo.mutable.store`, and `ch*` are ordering tokens.

```text
# BEFORE

v0,ch1 = load(buffer,ch0)         # Read the buffer after dependency ch0.
v1 = add(v0,rhs)                  # Compute the first elementwise addition.
ch2 = store(buffer,v1,ch1)        # Write v1; return an updated ordering token.
v2,ch3 = load(buffer,ch2)         # Read back the value just written.
v3 = add(v2,rhs)                  # Compute the second elementwise addition.
ch4 = store(buffer,v3,ch3)        # Write the final result.

# AFTER

v0,ch1 = load(buffer,ch0)         # Retain the initial buffer read.
v1 = add(v0,rhs)                  # Compute the first addition.
v3 = add(v1,rhs)                  # Consume v1 directly, eliminating the reload.
ch4 = store(buffer,v3,ch1)        # Keep the final write; reconnect its dependency.
```

**What changes:** the intermediate store/reload disappear. The second addition consumes the first addition’s result directly. Both additions remain at this stage.

For a scalar analogy, starting with buffer value `2` and `rhs=3`, both versions finish with `8`.

The transformation requires the same buffer and appropriate exclusive chain uses. See [elide-redundant-mutable-ops.mlir:16](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/Transforms/elide-redundant-mutable-ops.mlir:16) and [ElideRedundantMutableOps.cpp:36](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ElideRedundantMutableOps.cpp:36).

**How the code reconnects values and chains:** `EliminateRedundantLoads` checks that the reload’s input chain comes from a store, has one use, and refers to the same buffer.

It then forwards both results:

```text
replace v2 with v1    # The reload would return the value just stored.
replace ch3 with ch2  # Remove the reload's ordering step.
erase the reload
```

This exposes two consecutive stores in the chain. `EliminateRedundantStores` checks exclusive chain use and the same buffer, then:

```text
replace ch2 with ch1  # Bypass the intermediate store's ordering token.
erase the intermediate store
```

The final store now consumes `ch1`, and the second addition consumes `v1` directly. Greedy pattern application lets the first rewrite expose the opportunity for the second. Neither pattern removes the additions.

Fusion settings have different scopes. Disabling `usePatterns` skips the MO pattern-fusion stage; MAP’s `FuseKernels` is scheduled independently in [buildMapPipeline()](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/GraphCompiler.cpp:338).

In that function, MAP’s fusion pass is added separately to the pass manager. Consequently, the MO pattern-fusion flag governs the earlier MO stage rather than every fusion opportunity in the compilation pipeline.

With **device settings**, the compiler receives target information that influences validation and applicable rewrites. The result can be a metadata change, a different optimization decision, or a compilation error.

**12. Attaching CPU target metadata to a graph.**

```text
triple = "x86_64-unknown-linux"    # Target platform: x86-64, unknown vendor, Linux.
arch = "x86_64"                   # Target architecture identifier.

# BEFORE

mo.graph @example(...)           # Graph initially lacks target_info.

# AFTER, schematically

target = TargetInfoAttr(triple,arch) # Package target settings as compiler metadata.
mo.graph @example(...) attributes {
    target_info = target         # Attach the metadata to the graph operation.
}
```

**What changes:** the graph operation receives target metadata. Its body operations can remain identical.

The validator attaches this information when both target fields are supplied. If no target information is available, it returns without performing its dtype checks. See [ValidateDevicesPass.cpp:126](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ValidateDevicesPass.cpp:126).

**How that code attaches the metadata:**

```text
construct TargetInfo(triple,arch)          # Runtime/compiler target description.
fromRuntimeTargetInfo(MLIR context, info)  # Convert it into an MLIR attribute.
graphOp.setDeviceTargetInfo(attribute)     # Attach it to the graph operation.
```

The setter in [MOOps.cpp:1011](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/MOOps.cpp:1011) calls `setAttr(...)` with the `target_info` attribute name.

That is the metadata change shown in the example. If neither new nor existing target information is available, the validator returns before examining dtypes.

**13. Rejecting an unsupported device/dtype combination.**

```text
# BEFORE

x : tensor<[128], f32, cpu:0>     # CPU tensor containing 128 float32 values.
y = cast(x,dtype=f8e5m2)          # Request conversion to the specified 8-bit float format.

# AFTER VALIDATION

ERROR: unsupported dtype on cpu:0 # This CPU validator rejects the requested combination.
```

**What changes:** compilation fails at validation. The pass does not replace the requested cast with a different dtype.

This behavior is checked in [f8e5m2-cpu-errors.mlir:5](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/test/mo-opt/MODialect/ValidateDevicesPass/f8e5m2-cpu-errors.mlir:5).

**How the code finds the invalid dtype:** [ValidateDevicesPass.cpp:80](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ValidateDevicesPass.cpp:80) examines graph arguments and operation results whose types are MO tensor types.

```text
x's declared dtype = f32       → accepted.
cast result dtype = f8e5m2     → checked against the target architecture.
CPU + float width below 32    → rejected unless an explicit exception applies.
```

When the cast result fails that check, the implementation emits an error on the operation and propagates failure to `signalPassFailure()`. These decisions use declared types; evaluating the tensor values is unnecessary.

This checkout’s validator has these additional rules:

| Dtype | x86-64 target | AArch64 target |
|---|---|---|
| `f32` | Allowed | Allowed |
| `bf16` | Allowed | Allowed |
| `f16` | Allowed | Rejected |
| `f8e5m2` | Rejected | Rejected |

These describe this pass’s validation rules. See [ValidateDevicesPass.cpp:45](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/ValidateDevicesPass.cpp:45).

The helper first accepts BF16 for x86-64/AArch64 and F16 for x86-64. It then rejects other floating types narrower than 32 bits on those CPU architectures. The order of these conditions produces the table above.

**14. Using architecture to determine whether fusion applies.**

Assume the other requirements of the BF16 matmul/add pattern are satisfied:

```text
# SAME INITIAL GRAPH

m = matmul(X,W)                   # Produce a matrix multiplication result.
y = add(m,residual)               # Add a matching residual tensor.

# WITH module device metadata arch="sm_100a"

y = mo.composite.matmul_add(
    X,W,residual)                # This architecture permits the particular fusion.

# WITH module device metadata arch="sm_90"

m = matmul(X,W)                   # The architecture check skips this fusion.
y = add(m,residual)               # The separate addition remains.
```

**What changes:** the available rewrite depends on the GPU architecture. This pattern obtains that architecture from the module’s `mo.device_info_mapping`.

See [Fusion.cpp:4460](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Patterns/Fusion.cpp:4460).

**How the code obtains and checks the architecture:** [PatternFusion.cpp:97](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/GraphCompiler/lib/MODialect/Transforms/PatternFusion.cpp:97) creates `DeviceInfoMapping` from the enclosing module and supplies it to the matmul/add patterns.

The matcher follows:

```text
add result type → DeviceRef(gpu,0).
DeviceRef       → module device-info entry.
device-info     → architecture string.

arch == "sm_100a" → continue checking the fusion.
arch == "sm_90"   → notifyMatchFailure; retain separate operations.
```

This lookup uses the module’s `mo.device_info_mapping`. The `target_info` attribute from example 12 is a separate source of information used by the device validator.

The **canonicalizer and CSE** around these passes perform smaller cleanups.

**15. Simplifying an integer identity.**

```text
# BEFORE

x : i32                          # An integer value, possibly unknown at compile time.
zero = constant(0 : i32)          # Embed the integer zero with the matching type.
y = arith.addi(x,zero)            # Compute integer x+0.

# AFTER

y = x                            # Reuse x directly; eliminate the unnecessary addition.
```

**What changes:** the addition disappears without needing to know `x`’s value. This illustrates the local simplifications performed by canonicalization. ([mlir.llvm.org](https://mlir.llvm.org/docs/Canonicalization/))

**How the code achieves this:** the upstream implementation is `arith::AddIOp::fold()` in [ArithOps.cpp](https://mlir.llvm.org/doxygen/ArithOps_8cpp_source.html). Its relevant code is:

```cpp
if (matchPattern(adaptor.getRhs(), m_Zero()))
  return getLhs();
```

`m_Zero()` recognizes the constant right operand. `getLhs()` returns the existing SSA value `x` as the folded result. The rewrite driver redirects uses of the addition’s result to `x` and removes the redundant addition. Knowing `x`’s numerical value is unnecessary.

The [canonicalizer implementation](https://mlir.llvm.org/doxygen/Canonicalizer_8cpp_source.html) gathers dialect/operation canonicalization patterns and applies greedy rewriting and folding. An unused zero constant can also disappear during cleanup.

**16. Eliminating equivalent constants with CSE.**

```text
# BEFORE

a = constant(3)                   # Create a constant integer value.
b = constant(3)                   # Create an equivalent constant of the same type.
output(a,b)                       # Return both values.

# AFTER

a = constant(3)                   # Retain one constant operation.
output(a,a)                       # Both outputs reference that same value.
```

**What changes:** duplicate computations share one result. Two graph outputs remain, but they reference the same constant value. MLIR’s pass documentation shows this kind of CSE transformation. ([mlir.llvm.org](https://mlir.llvm.org/docs/PassManagement/))

**How the code achieves this:** upstream [CSE::runOnOperation()](https://mlir.llvm.org/doxygen/CSE_8cpp_source.html) obtains dominance information and calls the [common-subexpression elimination utility](https://mlir.llvm.org/doxygen/Utils_2CSE_8cpp_source.html).

The utility maintains a scoped table of previously encountered operations. It compares operation structure using `OperationEquivalence`, including the types, attributes, and operands relevant to equivalence. Different SSA result names do not make otherwise equivalent constants different.

For this example:

```text
visit a = constant(3)  # Record this operation as an available expression.
visit b = constant(3)  # Find an equivalent available operation: a.
redirect b's uses to a # output(a,b) becomes output(a,a).
erase b               # Remove the duplicate operation.
```

`replaceUsesAndDelete()` performs the use replacement and removal, with dominance handling to ensure the retained result is available where it is used. The two graph outputs remain; both reference the retained constant’s result.

Canonicalization and CSE recur around the main transformations because a rewrite can expose further simplifications or duplicate computations. ([mlir.llvm.org](https://mlir.llvm.org/docs/Canonicalization/))

## QnA 3 · What is the expanded MO pass call stack? {#qna-mo-2}

**Building the pipeline records work; running it performs the rewrites.** `buildMoPipeline` calls `addPass` / `addNestedPass<GraphOp>`. The native pass implementations below enter their `runOnOperation()` bodies only when `pm.run(IR1)` reaches them. A GraphOp pass works inside a graph; a module pass can also inspect declaration symbols and other graphs. [[gc:219]] [[gc:241]] [[gc:638]] [[gc:654]]

The tree follows the route that actually runs the graph compiler. **EXAMPLE** rows describe comparison graphs or possible rewrites; **ATTENTION** rows describe the recorded Q/K/V custom-call structure. They are source walkthroughs, not captured pass dumps.

```calltree
MOFContext::compileToBinary → clone source module as IR1 [[frontend:377]]
├─ ModuleOp::getFromOpaquePointer(source).clone() creates new operations/regions/SSA values [[frontend:381]]
│  ├─ G1 keeps its source GraphOp and body; IR1 owns the compilation copy
│  └─ clone stays in the same MLIRContext; register the shared MAXContext there [[frontend:384]]
├─ GCPipelineOptions → overrideLoweringOptions → detect seeded declaration marker [[frontend:397]]
│  └─ ATTENTION: Python seeded P1; kernelDeclsPrePopulated=true [[api:1042]] [[frontend:402]]
└─ convertMOToMEF(IR1) [[frontend:356]]
   └─ MO::runGCPipeline(IR1, options) [[gc:638]]
      ├─ gcPipelineImpl → buildMoPipeline [[gc:574]] [[gc:219]]
      │  ├─ copy shape/fold/fusion/device options into MoPipelineOptions [[gc:508]]
      │  │  └─ these gates choose which pass objects get added; no tensor computation here
      │  ├─ if declarations were NOT seeded: ImportPackages → VerifyKernels [[gc:227]]
      │  │  ├─ import package declarations so registration names can be resolved
      │  │  ├─ VerifyKernelsPass::runOnOperation → MojoRegistry → visit KernelDeclOps [[verify-kernels:979]]
      │  │  │  ├─ check declared entry points and compatible execute/shape/build_graph roles [[verify-kernels:998]] [[verify-kernels:1021]]
      │  │  │  └─ EXAMPLE: a declaration missing its required entry point fails here
      │  │  └─ ATTENTION: skip this initial pair; already seeded does not mean skip all later checks
      │  ├─ RewriteOpaqueTypes → VerifyParameters [[gc:233]] [[gc:234]]
      │  │  ├─ RewriteOpaqueTypesPass reads mogg.opaque_type_mapping [[rewrite-opaque:34]]
      │  │  │  ├─ EXAMPLE mapping: "State" → "package.State"; rewrite !mo.opaque<"State">
      │  │  │  ├─ recursively replace matching types/attributes; consume the mapping marker [[rewrite-opaque:77]] [[rewrite-opaque:87]]
      │  │  │  └─ ATTENTION: Q/K/V/O are TensorTypes; "opaque custom call" does not mean OpaqueType
      │  │  └─ VerifyParametersPass builds a ParameterUseDefGraph for each declaration region [[verify-params:182]] [[verify-params:203]]
      │  │     ├─ verify parameter references, definitions and their scopes
      │  │     ├─ EXAMPLE: a type uses N with no valid declaration/definition → failure
      │  │     └─ simplifyParameters also permits substitution; otherwise preserve analyses [[verify-params:232]] [[verify-params:242]]
      │  ├─ optional tensor debug printing; legalize RMO operations [[gc:236]] [[gc:239]]
      │  │  ├─ debugPrintAllTensors gates instrumentation; it is not a request to print Q now
      │  │  ├─ LegalizeRMOOps registers analogue/composite/shape patterns and applies them greedily [[rmo-legalize:1203]] [[rmo-legalize:1244]]
      │  │  │  └─ EXAMPLE: an RMO arithmetic analogue becomes MO operations with explicit shape handling
      │  │  └─ ATTENTION: the authored call is already mo.custom, with no RMO arithmetic body
      │  ├─ graph CSE → canonicalizer → CanonicalizeChainSequences [[gc:241]] [[gc:243]]
      │  │  ├─ CSE shares eligible equivalent producers; consumers reference the retained SSA value
      │  │  │  └─ EXAMPLE: two identical shape constants → one producer used twice
      │  │  ├─ canonicalizer invokes operation folds/patterns [[mo-fold:419]] [[mo-fold:1252]]
      │  │  │  └─ EXAMPLE: cast(%q):T0→T0 → %q; redirect the custom call's input
      │  │  └─ CanonicalizeBufferOpSequences::matchAndRewrite examines mutable chains [[mo-chains:115]]
      │  │     ├─ collect write effects; reject overlaps, region crossings and unsafe aliases [[mo-chains:45]] [[mo-chains:70]] [[mo-chains:132]]
      │  │     ├─ independent A→B: make A/B depend on the earlier chain; join their results [[mo-chains:138]]
      │  │     └─ ATTENTION: this immutable custom call has no such mutable-operation pair
      │  ├─ graph canonicalizer → CSE [[gc:245]]
      │  │  └─ clean up values exposed by the chain rewrite; see QnA 5 for SSA examples
      │  ├─ optional shape inference and constant folding [[gc:256]] [[gc:261]]
      │  │  ├─ useShapeInference → buildShapeInferencePipeline [[gc:657]]
      │  │  │  ├─ canonicalizer → PropagateShapes → CSE → canonicalizer → VerifyParameters
      │  │  │  ├─ PropagateShapesPass → propagateShape → shapeInferOp [[mo-propagate:1206]] [[mo-propagate:881]]
      │  │  │  │  ├─ ordinary inputs contribute type shapes; data-dependent inputs must be staticizable [[mo-propagate:892]]
      │  │  │  │  └─ substitute/solve parameter expressions and update affected types [[mo-propagate:1075]]
      │  │  │  └─ EXAMPLE: (N+4)-4 → N; ATTENTION: [32,32] has no unresolved dimension names
      │  │  └─ constantFold → ConstantFoldPass::runOnOperation [[mo-constant-fold:815]]
      │  │     ├─ constantSubgraphs: follow constants; a block argument stops constant propagation [[mo-constant-fold:445]] [[mo-constant-fold:480]]
      │  │     ├─ policy selects candidates; clone them into temporary constant_fold_subgraphs [[mo-constant-fold:823]] [[mo-constant-fold:836]]
      │  │     ├─ lowerSubgraphs: run a host pipeline with constantFold=false to prevent recursion [[mo-constant-fold:757]] [[mo-constant-fold:776]]
      │  │     ├─ executeModel on CPU → await results → replaceConstantOps in IR1 [[mo-constant-fold:343]] [[mo-constant-fold:669]]
      │  │     └─ ATTENTION: %q/%k/%v are block arguments; shapes alone cannot supply O's numbers
      │  ├─ hoist constant subgraphs → CSE [[gc:265]]
      │  │  ├─ buildCallGraphAndSortGraphs → processGraph, callees before callers [[mo-hoist:112]] [[mo-hoist:127]]
      │  │  ├─ find inputs supplied as the same SSA value at every call site [[mo-hoist:47]]
      │  │  ├─ clone invariant-derived ops into callers; pass boundary values as new callee arguments [[mo-hoist:528]]
      │  │  ├─ EXAMPLE: @helper transposes shared %w → compute transpose in caller, pass %wt
      │  │  └─ ATTENTION: no nested mo.call subgraph here; the Mojo kernel is not a callee MO graph
      │  ├─ SymbolicOptimization → canonicalizer → CSE [[gc:273]]
      │  │  ├─ populate algebraic/shape patterns; apply top-down greedy rewrites [[mo-symbolic:49]] [[mo-symbolic:60]]
      │  │  ├─ core → forward mathematical rewrites → core → backward rewrites → core [[mo-symbolic:94]]
      │  │  ├─ EXAMPLE: sub(x,x) → broadcast(0) under the fast-math pattern gate [[mo-algebra:2254]]
      │  │  └─ ATTENTION: no graph-visible sub/matmul/softmax arithmetic inside mo.custom
      │  ├─ merge shape materializations; add fallback shape functions [[gc:277]]
      │  │  ├─ MergeDuplicateShapeMaterializations keys on operation name + operands [[mo-merge-shapes:112]]
      │  │  │  ├─ require the candidate to define its referenced parameters [[mo-merge-shapes:103]]
      │  │  │  └─ EXAMPLE: duplicate broadcasts define D1 and D2 → D2 remaps to D1; erase duplicate [[mo-merge-shapes:118]]
      │  │  ├─ AddFallbackShapeFunctions visits operations still declaring output dimensions [[mo-fallback:158]]
      │  │  │  ├─ EXAMPLE: data-dependent custom output needs a registered shape function [[mo-fallback:225]]
      │  │  │  └─ ATTENTION: static output [32,32], output parameter declarations=[] → skip
      │  │  └─ SymbolicOptimization → canonicalizer → CSE runs again after this shape work [[gc:279]]
      │  ├─ optional constant folding; infer layouts [[gc:283]] [[gc:287]]
      │  │  ├─ retry enabled constant folding after newly exposed constants
      │  │  └─ InferLayoutPass: seed constraints → propagate → assign inferred layouts [[mo-layouts:204]]
      │  │     ├─ EXAMPLE: matmul B gets KN; 2-D convolution input/filter get NHWC/FCRS [[mo-layouts:90]] [[mo-layouts:137]]
      │  │     ├─ clone affected tensor types with layout metadata; synchronize graph metadata [[mo-layouts:196]]
      │  │     └─ ATTENTION: no matmul/conv seed exposed by this custom call
      │  ├─ optional PatternFusion + shape propagation + cleanup [[gc:288]]
      │  │  ├─ usePatterns → PatternFusionPass::runOnOperation [[mo-patterns:40]]
      │  │  │  ├─ collect enabled fusion patterns with target/MHA options; rewrite matches [[mo-patterns:81]]
      │  │  │  ├─ eligible example: matmul + residual add → mo.composite.matmul_add [[test-mo-fusion:6]]
      │  │  │  └─ ATTENTION: one named custom call; no separate QK/softmax/PV pattern to match
      │  │  ├─ InvokeShapePackingFunctions resolves supported packing helpers [[mo-pack-shapes:154]]
      │  │  │  ├─ known dimensions → KGEN parameter declarations with those integer values [[mo-pack-shapes:179]]
      │  │  │  └─ sanity mode retains a comparison call/assertion; otherwise erase the resolved invocation [[mo-pack-shapes:187]]
      │  │  └─ PropagateShapes → graph canonicalizer → CSE [[gc:303]]
      │  ├─ ValidateDevices → canonicalizer → CSE [[gc:312]] [[gc:314]]
      │  │  ├─ read graph target; explicit triple+arch can supply it; absent target → return [[mo-devices:126]]
      │  │  ├─ scan tensor arguments/results for this target's supported dtypes [[mo-devices:80]]
      │  │  ├─ EXAMPLE: f8e5m2 on x86_64 CPU is rejected; f32 passes this dtype guard [[mo-devices:46]]
      │  │  └─ this is an IR/target check, not CUDA device initialization
      │  ├─ parallel-region cleanup; optional mutable-op cleanup [[gc:320]] [[gc:329]]
      │  │  ├─ CreateParallelFromUnbundle wraps structurally identical per-device consumers [[mo-parallel:16]] [[mo-parallel:313]]
      │  │  ├─ canonicalize bundle(unbundle(X)) → X; MergeAdjacentParallels joins eligible bodies [[mo-merge-parallel:16]]
      │  │  │  └─ EXAMPLE: two per-device add consumers become one parallel body after its producer
      │  │  ├─ optional ConstantFold runs again after parallel cleanup [[gc:325]]
      │  │  ├─ fuseJitKernels → ElideRedundantMutableOps [[mo-mutable:121]]
      │  │  │  └─ same-buffer store→load with exclusive chain: forward stored tensor; erase load [[mo-mutable:49]]
      │  │  └─ ATTENTION: one GPU, immutable inputs; no unbundle/parallel/store-load pattern here
      │  └─ hoist parameter expressions → canonicalizer → RemoveUnusedOps [[gc:333]]
      │     ├─ HoistParameterExpressions aliases complex dimension expressions per region [[mo-hoist-params:242]]
      │     │  ├─ EXAMPLE: [N*2,32] → declare p=N*2 → [p,32], with rebinds at boundaries [[mo-hoist-params:76]] [[mo-hoist-params:161]]
      │     │  └─ ATTENTION: [32,32] is already literal; BN/BD are separate custom-op attributes
      │     ├─ canonicalizer removes redundant rebinds and other local simplifications
      │     └─ RemoveUnusedOps has a restricted safe-to-erase list [[mo-remove:31]] [[mo-remove:37]]
      │        └─ unused AllReduceSum/DistributedBroadcast/DistributedScatter qualify; arbitrary side effects do not
      └─ pm.run(IR1) executes the configured sequence [[gc:654]]
         ├─ pass failure propagates to "Graph compilation failed" [[frontend:361]]
         └─ on success, this same run continues through MAP/MOGG/MGP; MO is its first stage
```

**For this Graph API route, seeded declarations skip the initial import and VerifyKernels pair.** The earlier custom-call validation used G1's library analysis; later lowering again resolves the declaration and maps operands. A constant-folding submodule, if one is needed, has its own lowering path. [[frontend:397]] [[gc:224]] [[mo-to-map:705]] [[mo-constant-fold:757]]

| Gate | What it controls | Value established here |
| --- | --- | --- |
| `kernelDeclsPrePopulated` | initial ImportPackages / VerifyKernels pair | `true` on this seeded Graph API route |
| `useShapeInference` | initial shape subpipeline | copied from effective GC options |
| `constantFold` | three scheduled opportunities to fold constant subgraphs | copied from effective GC options |
| `symbolicFfastMath` / `symbolicNoUndefinedMath` | particular algebraic assumptions | copied into symbolic-pattern options |
| `usePatterns` | fusion, packing-shape evaluation and following cleanup | copied from effective GC options |
| `fuseJitKernels` | mutable load/store cleanup | copied from effective GC options |

The option declarations contain defaults; the frontend also applies overrides. **A default in the header is not a recorded runtime flag value.** Every example below states the condition needed for its rewrite. [[mo-options:295]] [[mo-options:303]] [[mo-options:340]] [[frontend:398]] [[gc:508]]

## QnA 4 · What do graph and custom-call checks actually validate? {#qna-mo-3}

**There are several contracts, checked at different times.** Graph construction already validates the recorded call and output. MO compilation checks the cloned IR and its parameters; the unseeded route additionally verifies imported declarations. Later Mojo specialization checks constraints inside the chosen implementation. [[custom:127]] [[graph:1492]] [[verify-params:203]] [[verify-kernels:979]] [[kernel:88]]

```calltree
Validation lifecycle
├─ ops.custom(...) while recording G1
│  └─ KernelLibrary.verify_custom_op → native verifyCustomOp(registry, OP1) [[custom:127]] [[registry:634]]
│     ├─ registration/device → chosen declaration
│     ├─ map operand/result roles
│     └─ validate supplied parameter names and attribute kinds
├─ graph.output(%o)
│  └─ verify graph/output structure and parameter bindings [[graph:1492]] [[mo-ops:1082]]
└─ C1 compilation of IR1
   ├─ initial VerifyKernels only on the unseeded route: checks declarations [[gc:224]]
   ├─ VerifyParameters checks the IR's parameter use/definition graph [[verify-params:203]]
   ├─ ValidateDevices checks target/dtype compatibility when target info exists [[mo-devices:134]]
   └─ later Mojo specialization evaluates the execute body's rank assertion [[kernel:88]]
```

| Checked item | Fused-attention value | What a problem looks like |
| --- | --- | --- |
| Graph result contract | one `float32[32,32]` result | terminator result type disagrees with the graph signature |
| Registration | `modular_ops::fused_attention_custom` | unregistered name |
| Device selection | `gpu:0` | no compatible implementation for device metadata |
| Input roles | `%q → query`, `%k → key`, `%v → value` | missing input or incompatible operand role |
| Output role | `%o → OutputTensor` | wrong result count or role |
| Supplied parameters | `BN=16`, `BD=8` as integer attributes | `BNN` is unknown; a string for `BN` has the wrong attribute kind |

Sources for the table: graph output verification, kernel resolution, operand mapping and parameter validation. [[mo-ops:1117]] [[registry:650]] [[registry:677]] [[registry:477]] [[registry:492]] [[kernel:74]]

**Example 3.1 · a graph result must match its terminator.**

`OutputOp::verify` first pairs result-parameter declarations with their bindings, substitutes those bindings into the declared result types, then calls `checkOperandTypes`. Shape, dtype and device are part of a tensor type. [[mo-ops:1085]] [[mo-ops:1103]] [[mo-ops:1117]]

```transform
PASS — this example
  graph result types = [float32[32,32] on gpu:0]
  result parameters = []; mo.output parameter bindings = []
  mo.output operand = %o : float32[32,32] on gpu:0
  expected type == actual type

FAIL — inconsistent IR, changing only the declared result
  graph result types = [float32[32,16] on gpu:0]
  mo.output operand = %o : float32[32,32] on gpu:0
  expected [32,16] != actual [32,32]

Also inconsistent:
  expected float16[32,32] vs actual float32[32,32]
  expected cpu:0 placement vs actual gpu:0 placement
```

These failures diagnose a malformed contract; the verifier does not reshape, cast or transfer `%o` to repair it. Python's normal `Graph.output()` derives the signature from the supplied values, which is why the unmodified example agrees. [[graph:1449]] [[graph:1459]] [[mo-ops:1117]]

**Example 3.2 · registration lookup and positional operand mapping.**

```transform
PASS
  name = "modular_ops::fused_attention_custom"
  device = gpu:0
  inputs 0/1/2 = %q/%k/%v → query/key/value
  result 0     = %o       → output: OutputTensor
  ctx: DeviceContext     → launch argument, no fourth user tensor

FAIL: name = "modular_ops::fused_attention_typo"
  registry.containsKernel(name) = false → no registered kernel

FAIL: operands = [%q,%k]
  query and key match; value has no supplied operand

FAIL: operands = [%q,%k,%v,%extra]
  the extra tensor has no corresponding kernel input
```

The mapper skips launch arguments, matches tensor roles by position, then detects missing or extra operands. Supplying `[%q,%v,%k]` with identical types can pass these structural checks: the compiler cannot infer that the user meant to exchange the semantic roles back. [[operand-inference:116]] [[operand-inference:202]] [[operand-inference:244]] [[operand-inference:253]] [[registry:642]]

**Example 3.3 · integer metadata versus a misspelled or mistyped parameter.**

The native validator builds a name-to-declaration map, finds each supplied name and checks its attribute kind. Mojo `Int` parameters here are checked through the KGEN index-type branch. [[registry:470]] [[registry:477]] [[registry:492]] [[kernel:78]]

```transform
PASS
  {BN = 16 : index, BD = 8 : index}
  BN → declared Int → IntegerAttr → accepted by this check
  BD → declared Int → IntegerAttr → accepted by this check

FAIL: {BNN = 16 : index, BD = 8 : index}
  lookup("BNN") → no declaration with this name

FAIL: {BN = "16", BD = 8 : index}
  lookup("BN") succeeds; StringAttr is not IntegerAttr

IR state on success:
  OP1.parameters retains the two integer attributes
  G1/IR1 still have three user tensor arguments
  no mo.constant(16) or mo.constant(8) is added by this validator
```

Accepting an integer attribute proves its kind, not every numerical precondition of a tiling algorithm. Those requirements belong to the implementation and later specialization. [[registry:492]] [[kernel:78]] [[kernel:105]]

**Example 3.4 · shape names need declarations and bindings.**

```transform
PASS — static fused attention
  input parameter declarations = []
  result parameter declarations = []
  Q/K/V/O shapes = [32,32]
  no dimension reference needs resolution

PASS — illustrative symbolic graph
  graph declares input dimension N
  input %x : [N,32]
  graph declares output dimension M
  output binding M=N; output value has shape [N,32]
  substitute M=N → declared result and returned value agree

FAIL — malformed symbolic graph
  a type refers to N, but the relevant scope supplies no definition
  OR graph declares result parameter M but mo.output has zero bindings
```

The first failure concerns the parameter use/definition graph; the second fails the output binding-count check before type comparison. Naming `N` does not assign it the value 32. [[verify-params:202]] [[mo-ops:984]] [[mo-ops:1087]]

**Example 3.5 · a kernel declaration must describe a usable implementation.**

```state
Unseeded compilation path
└─ imported KernelDeclOp
   ├─ valid execute entry → inspect its operand/result roles
   ├─ optional shape function → must return a shape
   └─ optional shape/build_graph inputs → must agree with execute's roles

Comparison failure:
  shape function declared, but no shape result → declaration verification fails

This attention route:
  declarations already seeded → the initial VerifyKernels pass is skipped
  earlier OP1 validation and subsequent IR/lowering checks still matter
```

`declVerifyShapeFunctionReturnsShape` allows an absent shape function; it rejects a present function with no shape result. Static output dimensions do not require every custom kernel to provide a shape function. [[verify-kernels:46]] [[verify-kernels:1021]] [[mo-fallback:158]] [[gc:224]]

**Example 3.6 · target/dtype validation is narrower than GPU readiness.**

| Comparison input | Result of this pass |
| --- | --- |
| tensor dtype `f32`, valid target information | passes the shown architecture/dtype restrictions |
| `f8e5m2` tensor, x86_64 CPU target | rejected |
| no graph target and no complete target override | this pass returns without a dtype scan |

**TEST · inspected expectation:** the CPU test contains a cast to `f8e5m2` and expects a diagnostic naming `cpu:0`. It is not a GPU support test and was not executed for this walkthrough. The implementation neither creates a device context nor queries CUDA. [[test-mo-device:3]] [[test-mo-device:7]] [[mo-devices:46]] [[mo-devices:126]]

**Example 3.7 · rank 2 is a later implementation assertion.**

```transform
Fused-attention types:
  Q/K/V/O = float32[32,32] → rank=2
  execute specialization: assert rank == 2 → satisfied

Comparison:
  Q/K/V/O = float32[1,32,32] → rank=3
  tensor-role matching alone does not prove rank==2
  execute specialization: assert rank == 2 → fails
```

The native role checker does not run attention on sample numbers. It can catch an absent `value` tensor; it cannot establish numerical correctness or validate the contents of runtime Q/K/V buffers. [[operand-inference:202]] [[registry:677]] [[kernel:88]] [[entry:78]]

## QnA 5 · What are canonicalization, CSE and constant folding? Show values. {#qna-mo-4}

**A rewrite changes compiler objects and their uses. It need not calculate a tensor.** The word “fold” covers both a small operation hook and the larger MO constant-subgraph pass. [[mo-fold:419]] [[mo-constant-fold:815]]

| Mechanism | What it does | Concrete change |
| --- | --- | --- |
| operation `fold` hook | returns an existing value or a constant attribute when its rule applies | identity cast result → original input |
| canonicalization pattern | rewrites operations into a simpler form | consecutive reshapes → one reshape |
| CSE | shares eligible equivalent producers in a valid scope | two constant results → one shared result |
| MO ConstantFold pass | selects, compiles and executes constant-only subgraphs on the host | computed tensor expression → constant data in IR |

Sources: local folds/patterns, the scheduled CSE pass and the host constant-subgraph evaluator. [[mo-fold:419]] [[mo-fold:1597]] [[gc:241]] [[mo-constant-fold:757]] [[mo-constant-fold:343]]

The added mini-graphs below are **illustrative comparisons** unless marked TEST. Operation counts cover only the displayed operations.

**Example 4.1 · identity cast rewires the custom call.**

```transform
T0 = float32[32,32] on gpu:0

BEFORE
  %q1 = mo.cast(%q) : T0 → T0
  %o  = mo.custom(%q1, %k, %v) : T0
  mo.output %o

RULE
  inTy == resTy → CastOp::fold returns %q

AFTER
  %o = mo.custom(%q, %k, %v) : T0
  mo.output %o

SSA change: OP1.operand[0] points to %q; %q1's cast disappears.
Shown operation count: 3 → 2.
If q[0,0] were 1.25, that value would still be 1.25.
```

{{excerpt:mo-fold:419:423}}

`float32[32,32] → float16[32,32]` is not an identity: the complete types differ, so this particular return-input rule cannot apply. A real conversion or a separate constant-conversion rule must account for the dtype change. [[mo-fold:420]] [[mo-fold:427]]

**Example 4.2 · rebind and reshape cleanup can expose each other.**

```transform
Same-type rebind:
  BEFORE: %r = mo.rebind(%q) : T0 → T0
          consumer(%r)
  AFTER:  consumer(%q)
  condition: no new output parameter declaration

Round-trip static reshape:
  BEFORE: %flat = mo.static.reshape(%q) : [32,32] → [1024]
          %back = mo.static.reshape(%flat) : [1024] → [32,32]
          %o = mo.custom(%back,%k,%v)

  STEP 1: bypass the preceding reshape:
          %back = mo.static.reshape(%q) : [32,32] → [32,32]

  STEP 2: equal input/output types → replace %back with %q

  AFTER:  %o = mo.custom(%q,%k,%v)
          %flat is dead if it had no other users

Payload order is unchanged: 32×32 = 1024 elements.
With no other uses, the two added reshape operations disappear.
```

The implementation explicitly forwards through a preceding static reshape, then its fold can return the input on type equality. A rebind that changes layout or unresolved shape information does not satisfy the simple same-type rule. [[mo-fold:1252]] [[mo-fold:1591]] [[mo-fold:1597]]

**Example 4.3 · duplicate constants share a producer, not tensor arguments.**

```transform
BEFORE
  %a = mo.constant([32,32]) : si64[2] on cpu:0
  %b = mo.constant([32,32]) : si64[2] on cpu:0
  consumer1(%a)
  consumer2(%b)

AFTER SHARING
  %a = mo.constant([32,32]) : si64[2] on cpu:0
  consumer1(%a)
  consumer2(%a)

Two eligible identical producers → one producer with two uses.

COUNTEREXAMPLES
  constant([32,16]) has different data → not this duplicate
  constant([32,32]) as si32[2] has a different type
  %q and %k have the same T0 type but are separate inputs → not duplicates
```

CSE can perform this sharing; constant uniquing in the canonicalizer may expose the same result earlier. Native attribute uniquing and removing a producer operation are different changes: two operations can reference one attribute before either operation is erased. The original attention graph stores its dimensions in tensor types and needs no pair of these shape-constant operations. [[gc:241]] [[mo-fold:658]] [[entry:43]]

**Example 4.4 · simplify a chain expression, or relax a dependency safely.**

```transform
Local ChainCreateOp fold:
  %same = mo.chain.create(%chA)       → %chA
  %dup  = mo.chain.create(%chA,%chA)  → %chA
  %ch0  = mo.chain.create()          → no replacement from this fold

Separate CanonicalizeChainSequences pattern:
  BEFORE: c0 → A(writes buffer X) → cA → B(writes buffer Y) → cB → C

  AFTER, when the accesses are independent:
          c0 → A(writes X) → cA ──┐
          c0 → B(writes Y) → cB ──┴─ join(cA,cB) → C

If X and Y overlap as mutable operands, keep the ordering.
The rewrite changes dependency edges; no GPU command runs here.
```

The local fold deduplicates operands and forwards a sole chain. The separate chain-sequence pass checks write effects, mutable operands, opaque alias risks, multi-device rendezvous and region boundaries before changing dependencies. An unused initial chain may disappear through other cleanup; the zero-input fold itself returns no replacement. [[mo-fold:470]] [[mo-fold:478]] [[mo-chains:70]] [[mo-chains:132]] [[mo-chains:138]]

**Example 4.5 · real constant-subgraph evaluation, with a variable input left live.**

**TEST · inspected expectation:** `@multiple_internal_ops` builds a matrix from a constant `-1.0`, then uses it in a matmul with a graph argument. The expected output keeps the matmul and replaces its constant operand subgraph. [[test-mo-constant:17]]

```transform
BEFORE — values from the repository test, simplified syntax
  %x : float32[1,3]                         ← runtime graph argument
  %c = constant(-1.0)
  %b = broadcast(%c) → [[-1,-1,-1],[-1,-1,-1]]   shape [2,3]
  %p = constant([1,0])                     ← transpose permutation
  %w = transpose(%b,%p)                    shape [3,2]
  %y = matmul(%x,%w)                       shape [1,2]

AFTER — expected constant boundary
  %w = constant([[-1,-1],[-1,-1],[-1,-1]])  shape [3,2]
  %y = matmul(%x,%w)
  output %y

Known during folding: all six values of %w.
Unknown during folding: %x, therefore %y.

Illustrative later input: x=[[1,2,3]] → y=[[-6,-6]].
That arithmetic is an explanation, not an executed result.
```

```calltree
ConstantFoldPass::runOnOperation [[mo-constant-fold:815]]
├─ constantSubgraphs: collect constant-derived operations [[mo-constant-fold:445]]
│  └─ %x has no defining operation → block-argument boundary, so matmul remains [[mo-constant-fold:480]]
├─ policy->shouldFold selects acceptable candidates [[mo-constant-fold:836]]
├─ cloneSubgraphs → temporary module; lowerSubgraphs compiles for host execution [[mo-constant-fold:840]] [[mo-constant-fold:789]]
│  └─ constantFold=false in this nested pipeline prevents recursive folding [[mo-constant-fold:776]]
├─ executeModel → setup CPU model → execute → await ready results [[mo-constant-fold:345]] [[mo-constant-fold:372]]
├─ replaceConstantOps: create constant results, redirect uses, erase internal operations [[mo-constant-fold:688]]
└─ destroy temporary module; canonicalize IR1 before another iteration [[mo-constant-fold:899]]
```

This pass can allocate host buffers and execute constant work during compilation. That does not imply the caller's attention computation has run. Candidates with parameterized shapes, transfers or unsupported results are filtered, and the folding policy can decline a candidate even when its values are known. [[mo-constant-fold:174]] [[mo-constant-fold:836]]

**Example 4.6 · algebraic simplification needs its stated assumptions.**

```transform
Source pattern under enableFastMath:
  BEFORE: %z = mo.sub(%x,%x)
  AFTER:  %zero = constant(0)
          %z = broadcast(%zero) to x's shape

Finite example: x=[3,-2,0] → [0,0,0].
No values of x have to be read to apply the algebraic rule.

IEEE counterexample without the finite-math assumption:
  NaN - NaN is NaN, not zero.

With this fast-math pattern disabled, do not attribute x-x→0 to it.
```

The pattern is registered inside `if (enableFastMath)`. The inspected elementwise test enables the symbolic math options explicitly and checks for the scalar-zero/broadcast replacement. This is an algebraic rewrite, unlike the CPU evaluation in Example 4.5. [[mo-algebra:2254]] [[test-mo-symbolic:1]] [[test-mo-symbolic:62]]

**Example 4.7 · store-to-load forwarding preserves the stored value and chain.**

```transform
BEFORE — same buffer, store chain used only by this load
  %cs = mutable.store(buffer X, tensor %x, %c0)
  (%y,%cl) = mutable.load(buffer X, %cs)
  consumer(%y,%cl)

AFTER
  %cs = mutable.store(buffer X, tensor %x, %c0)
  consumer(%x,%cs)

%y → %x; %cl → %cs; erase the load.
For x=[2,4], the forwarded tensor is still [2,4].

Different buffer, or another user of %cs → this rule declines.
```

`EliminateRedundantLoads` checks the unique chain use and identical buffer operand before replacing both results. The MO pipeline enables this cleanup under `fuseJitKernels`; the authored attention call has no mutable store/load pair. [[mo-mutable:49]] [[mo-mutable:61]] [[mo-mutable:65]] [[gc:329]]

**What is actually known for fused attention?**

| Known at this stage | Still supplied at runtime |
| --- | --- |
| shapes `[32,32]`, rank 2, dtype f32, device gpu:0 | the 1,024 Q values |
| `BN=16`, `BD=8`, registration name | the 1,024 K values and 1,024 V values |
| 1,024 elements, 4,096 logical payload bytes per tensor | the 1,024 output numbers |

Q/K/V are arguments; the Python arrays are created after `compile` and `init`. Knowing that a tensor holds 1,024 float32 values cannot supply those values to a constant folder. [[entry:43]] [[entry:60]] [[entry:74]] [[entry:78]] [[mo-constant-fold:480]]

## QnA 6 · Do shape inference and pattern fusion discover FlashAttention here? {#qna-mo-5}

**The user already selected a fused-attention implementation.** The graph records `mo.custom`; the QK, softmax and PV computations are inside the Mojo implementation. Graph-level patterns do not see three separate MO operations representing those computations. [[entry:57]] [[kernel:67]] [[kernel:90]]

```state
Visible to MO graph passes
  inputs Q/K/V → mo.custom(symbol, BN=16, BD=8) → output O

Inside the selected package implementation
  FusedAttention.execute → fused_attention_gpu[BN, BD](...)

These are different transformation scopes.
```

The shape pipeline canonicalizes, propagates shapes, eliminates duplicate expressions and verifies parameter declarations. It reasons about dimensions separately from tensor contents. The following examples show where that distinction permits a rewrite and where it leaves work for runtime. [[gc:657]] [[gc:663]] [[mo-propagate:881]]

**Example 5.1 · extract a known shape without reading Q's numbers.**

```transform
Teaching addition to the attention graph:
  %q : float32[32,32] on gpu:0
  %s = mo.shape_of(%q) : si64[2]

Canonicalization:
  inputTy.getShape() = [32,32]
  %shape = mosh.param.to_value <[32,32]>
  %s = mo.shape.to_tensor(%shape)
  → %s = mo.constant([32,32]) : si64[2]

Known: s[0]=32, s[1]=32, element count=32×32=1024.
Unknown: q[0,0], q[0,1], …, q[31,31].
```

`ShapeOfOp::canonicalize` obtains the shape from the input type and creates the shape-value conversion. `Shape::ToTensorOp::canonicalize` materializes an integer constant when that shape is constant. This does not convert Q's float32 data into integers. The original custom call needs no explicit `mo.shape_of` at this point. [[mo-fold:1337]] [[mo-fold:1396]] [[entry:57]]

**Example 5.2 · symbolic inference can simplify an expression without finding N.**

**TEST · inspected expectation:** `@param_expr_dce` simplifies a reshape whose dimension is `(D0+4)-4` back to its input. Renaming the input dimension to N makes the value flow easier to read. [[test-mo-shapes:50]]

```transform
BEFORE
  graph input dimension: N
  %x : float32[N]
  D2 = N + 4
  D3 = D2 - 4
  %r = static.reshape(%x) : [N] → [D3]
  output %r, binding result dimension to D3

SUBSTITUTION
  D3 = (N+4)-4 = N
  reshape input/output types now agree

AFTER
  output %x : float32[N]
  redundant dimension aliases and reshape are gone

If N later equals 32:  (32+4)-4 = 32.
If N later equals 64:  (64+4)-4 = 64.
Compilation need not choose either value to simplify the expression.
```

The propagator substitutes and folds parameter expressions, updates attributes/types, and simplifies graph result metadata. In the actual attention graph, N and D were already Python integers equal to 32 before the TensorTypes were recorded. [[mo-propagate:1075]] [[mo-propagate:1149]] [[entry:36]]

**Example 5.3 · a constant reshape target versus a runtime shape tensor.**

```transform
Case A — target entries known
  input shape = [32,32]
  new_shape = constant([16,64])
  output shape = [16,64]
  element-count check: 32×32 = 16×64 = 1024
  shape computation can become parameter/static information

Case B — target entries unknown
  input shape = [32,32]
  new_shape is a graph argument of type si64[2]
  that type says "two integers", not "the integers 16 and 64"
  output dimensions R0/R1 still need values

Two possible valid runtime targets:
  new_shape=[16,64] → R0=16, R1=64
  new_shape=[8,128] → R0=8, R1=128
```

`shapeInferOp` tries to staticize inputs whose **values** determine shape. If such an input cannot be staticized, it leaves the operation for further handling. `AddFallbackShapeFunctions` supplies helpers for supported operations that still declare output dimensions. The fixed output type of this attention call avoids that unresolved reshape case. [[mo-propagate:883]] [[mo-propagate:895]] [[mo-fallback:158]] [[mo-fallback:194]]

**Example 5.4 · merge dimension definitions that ordinary CSE cannot equate.**

**TEST · inspected expectation:** the duplicate-materialization test contains two broadcasts with identical operands, but fresh result dimension names D1 and D2. The dedicated pass remaps the names as well as the SSA results. [[test-mo-shape-merge:3]]

```transform
BEFORE
  %shape = shape_of(%selector : bool[N])
  %a = broadcast_to(%scalar,%shape), defines D1 → tensor[D1]
  %b = broadcast_to(%scalar,%shape), defines D2 → tensor[D2]
  users contain both D1 and D2 references

DEDUPLICATION
  key = (operation name, the same %scalar and %shape operands)
  verify each candidate defines the parameter references it owns
  retain %a; parameter mapping D2 → D1
  rewrite uses of %b → %a; remap types/attributes; erase %b

AFTER
  one broadcast producer and one dimension-definition family
  following rebind/CSE cleanup can simplify the consumers

Example runtime length N=32:
  scalar 7 broadcasts to 32 copies of 7 in either branch.
  The pass shares the producer without assuming N was known as 32.
```

Different declaration names initially make the operations' parameterized forms unequal. This pass builds a parameter use/definition graph, records the correspondence, then calls `remapTypesAndAttrs`. If operands differ or parameter ownership does not meet its test, it does not use this merge. [[mo-merge-shapes:77]] [[mo-merge-shapes:103]] [[mo-merge-shapes:112]] [[mo-merge-shapes:127]]

**Example 5.5 · a data-dependent custom output needs a shape function.**

This is an **illustrative custom kernel**, `demo::nonzero_indices`, whose output contains the indices of nonzero input elements. It is not the fused-attention registration.

```transform
Same input type, two different possible output lengths:
  x : int32[5]

  x=[0,7,0,-2,9] → indices=[1,3,4] → output shape [3]
  x=[0,0,0,0,0]  → indices=[]      → output shape [0]

The type int32[5] alone cannot determine result dimension K.

Fallback path, with a matching registered shape function:
  input x
  → shape-function invocation defines K
  → custom output described as int32[K]
  → execution can use the resolved result extent

Without that shape-function registration:
  custom op still declares its own data-dependent output dimensions
  → AddFallbackShapeFunctions reports an error
```

The pass first skips calls whose output parameter declarations are empty. For a remaining custom call, it requires the registered shape function, forwards matching compile-time parameters and includes the output index when building the helper invocation. Static attention has output `[32,32]` and no such pending output dimension declarations. [[mo-fallback:158]] [[mo-fallback:225]] [[mo-fallback:559]] [[graph:1351]]

**Example 5.6 · layout inference attaches meaning to axes.**

```transform
Comparison graph with an explicit matmul:
  A : float32[32,32]
  B : float32[32,32], no explicit layout
  C = mo.matmul(A,B)

InferLayouts:
  seed constraint on B → layout KN
  B's type → float32[32,32] with layout=KN
  propagate compatible layout constraints; synchronize graph metadata

Meaning: B's two axes represent reduction K and output N.
This pass sets type metadata; it does not itself transpose B's data.

Actual attention:
  Q/K/V → mo.custom
  no visible mo.matmul / mo.conv seed
  this seeding logic does not infer KN from the name "fused_attention"
```

For another comparison, the convolution branch assigns `NHWC` to a rank-4 input such as `[1,8,8,16]` and `FCRS` to a filter such as `[32,16,3,3]`. Existing explicit layouts are handled separately; incompatible newly inferred constraints produce an error. [[mo-layouts:34]] [[mo-layouts:59]] [[mo-layouts:100]] [[mo-layouts:137]] [[mo-layouts:196]]

**Example 5.7 · fusion needs visible operations and satisfied guards.**

**TEST · inspected expectations:** the matmul/add fusion tests supply CUDA `sm_100a` device metadata and explicit eligible graph shapes. They check replacement by `mo.composite.matmul_add`; this walkthrough did not run those tests. [[test-mo-fusion:1]] [[test-mo-fusion:4]]

```transform
Eligible residual example from the test:
  A = bf16[65536,1536]
  B = bf16[1536,1536], layout KN
  R = bf16[65536,1536]

  BEFORE: %m = matmul(A,B)
          %y = add(%m,R)
          output %y

  AFTER:  %y = composite.matmul_add(A,B,R)
          output %y

Eligible bias example from the test:
  A = bf16[256,1536]
  B = bf16[8192,1536], layout NK, transpose_b=true
  bias = bf16[8192]

  BEFORE: matmul → add(broadcast(bias)) → output
  AFTER:  composite.matmul_add(A,B,bias) → output [256,8192]
```

The inspected negative tests also matter:

| Same broad “matmul + add” shape | Expected behavior in these tests |
| --- | --- |
| matmul result used by add **and** returned separately | retain matmul and add |
| residual comes from another elementwise op | decline this composite pattern |
| another elementwise op follows the add | decline this composite pattern to leave later epilogue opportunities |

The examples and exclusions are explicit test expectations, not universal claims about every hardware target or fusion pass. [[test-mo-fusion:6]] [[test-mo-fusion:32]] [[test-mo-fusion:44]] [[test-mo-fusion:77]] [[test-mo-fusion:89]]

```state
Why this does not discover the example's FlashAttention:
  graph-visible: mo.custom("modular_ops::fused_attention_custom", Q,K,V)
  selected body: FusedAttention.execute → fused_attention_gpu[BN=16,BD=8]

MO PatternFusion has no separate matmul→softmax→matmul chain to replace.
BN=16 and BD=8 remain attributes selecting the authored implementation.
Later compilation specializes and lowers that Mojo implementation.
```

The pass builds a target-aware pattern set and applies matches greedily; a successful run can leave a custom computation unchanged. Packing-shape helpers and another shape-propagation cleanup follow the fusion pass when `usePatterns` is enabled. [[mo-patterns:81]] [[gc:288]] [[mo-pack-shapes:154]] [[kernel:105]]

## QnA 7 · What leaves MO, and what remains for later stages? {#qna-mo-6}

**MO hands onward a typed computation, with whatever simplifications its enabled passes proved.** It does not replace the attention custom call with measured output data. These exit examples distinguish the actual attention structure from comparison graphs that offer more work to the optimizer. [[gc:574]] [[gc:581]] [[entry:78]]

**Example 6.1 · the live fused-attention contract remains recognizable.**

```state
IR1 after MO cleanup, schematic live computation
└─ mo.graph @fused_attention : (T0,T0,T0) → T0
   ├─ %q, %k, %v : T0
   ├─ %o = mo.custom(%q,%k,%v)
   │  ├─ symbol = "modular_ops::fused_attention_custom"
   │  ├─ device = gpu:0
   │  └─ parameters = {BD=8, BN=16}
   └─ mo.output %o

G1's Python member values: retained
IR1: simplified typed graph + declarations
P1: implementation package available to the compiler
Next: MOToMAP constructs the access/compute representation
```

This is the live data path, not an exact textual compiler dump. It omits declaration symbols and any surviving setup operations. The example supplies no redundant casts, nested MO calls, parallel bundles or mutable store/load sequence for the comparisons above. Cleanup of an unused initial chain does not add or remove a user tensor input. [[entry:41]] [[entry:57]] [[graph:887]]

| Object or field | State across MO |
| --- | --- |
| `G1._mlir_op`, `G1._graph_body`, original TensorValue wrappers | continue to refer to the source graph |
| IR1's graph body | passes can replace operations, operands, result types and attributes here |
| live attention operands | three values: Q, K, V, each T0 |
| live attention result | one value: O, also T0 |
| custom registration / explicit parameters | `modular_ops::fused_attention_custom`, `BN=16`, `BD=8` |
| declaration-name mapping marker | consumed by RewriteOpaqueTypes; declaration symbols remain available |
| Python input arrays / attention output numbers | not provided by this compiler stage |

The operation clone separates G1's recording objects from IR1. The opaque-type rewrite consumes its mapping attribute, while MOToMAP later resolves the custom implementation using the declarations. [[frontend:381]] [[rewrite-opaque:87]] [[mo-to-map:705]] [[entry:78]]

**Example 6.2 · several cleanup rules compose around an attention call.**

```transform
Comparison IR1 BEFORE
  %qc   = cast(%q)                  T0 → T0
  %flat = static.reshape(%qc)       [32,32] → [1024]
  %back = static.reshape(%flat)     [1024] → [32,32]
  %r    = rebind(%back)             T0 → T0
  %o    = custom(%r,%k,%v)          result T0
  output %o

1. cast fold:             %qc → %q
2. same-type rebind:      %r → %back
3. consecutive reshape:  %back now reshapes %q directly to T0
4. identity reshape:     %back → %q
5. dead reshape cleanup: remove unused %flat

AFTER
  %o = custom(%q,%k,%v) : T0
  output %o

Shown operations: 6 → 2, if the removed intermediates have no other uses.
Graph signature: (T0,T0,T0) → T0 throughout.
The output still depends on all three runtime inputs.
```

This is why repeated canonicalization is useful: rewiring an earlier value exposes equality to a later rule. The actual attention source already starts without these four extra operations. [[mo-fold:419]] [[mo-fold:1252]] [[mo-fold:1591]] [[mo-fold:1597]] [[entry:57]]

**Example 6.3 · only the constant part of a graph disappears.**

```transform
Comparison graph entering MO:
  constant(-1) → broadcast → transpose → %w : float32[3,2]
  runtime %x : float32[1,3] ────────────┐
                                      └─ matmul(%x,%w) → %y

Leaving after the constant-subgraph example in QnA 5:
  %w = constant([[-1,-1],[-1,-1],[-1,-1]])
  runtime %x ──► matmul(%x,%w) ──► %y

Example runtime possibilities:
  x=[[1,2,3]] → y=[[-6,-6]]
  x=[[0,0,0]] → y=[[0,0]]

The compiler can store the six known weight values.
It cannot pick either result before the runtime x is supplied.
```

This is the boundary checked by the repository's constant-folding example: the final matmul still consumes a graph argument. Likewise, the attention graph's static dimensions and BN/BD do not make its three input tensors constant. [[test-mo-constant:17]] [[mo-constant-fold:480]] [[entry:78]]

**Example 6.4 · an unresolved dimension can leave MO as a named expression.**

```transform
Illustrative local tensor type before the final hoisting pass:
  %v : float32[N*2,32]
  N is an input dimension; no fixed numerical value is known

HoistParameterExpressions:
  declare param_expr_1 = N*2
  rewrite the local type to float32[param_expr_1,32]
  insert/rewrite rebinds where the surrounding types need to stay compatible

Possible later values:
  N=16 → param_expr_1=32 → local shape [32,32]
  N=32 → param_expr_1=64 → local shape [64,32]

The generated parameter name is an alias for the expression.
It does not add a user tensor argument or choose a value for N.

Actual attention local types:
  [32,32] → no complex dimension expression needs this alias
```

The pass reuses existing expression aliases per region where possible, otherwise creates a fresh name, replaces references and repairs type boundaries with rebinds. The spelling `param_expr_1` is illustrative; the real suffix comes from the graph's parameter counter. [[mo-hoist-params:76]] [[mo-hoist-params:161]] [[mo-hoist-params:227]] [[mo-ops:996]]

**Example 6.5 · verification failure stops the handoff.**

```state
Comparison: output expects [32,16] but returns a [32,32] tensor
└─ output/IR verification reports the mismatch
   └─ a failed compilation path does not produce a successful C1 artifact

During the MO pipeline:
└─ a pass signals failure
   └─ pm.run(IR1) returns failure
      └─ convertMOToMEF returns "Graph compilation failed" with diagnostics
```

The output mismatch can already be caught during graph construction. If a failure arises in the native pipeline, the frontend propagates it instead of treating a partially rewritten IR1 as a completed model. [[graph:1492]] [[mo-ops:1117]] [[gc:654]] [[frontend:361]]

**The next representation answers different questions.**

| Handoff | Work still ahead for this attention call |
| --- | --- |
| final MO marking → MOToMAP | mark stable tensors where applicable; form explicit access/compute structure |
| MAP → MOGG | select the registered implementation and construct its call/arguments |
| MOGG → MGP | form model init/execute regions, storage roles and dependency chains |
| Mojo emission / kernel compilation | specialize the authored implementation and produce native code |
| later model execution | supply Q/K/V buffers and calculate output values |

The optional `mo-pre-map` stage dump is placed after `buildMoPipeline`, before stable-tensor marking and MOToMAP. No such dump was captured here. Although the MO constant folder may internally compile and run a **separate constant model**, the main attention artifact still has the later compilation/runtime stages above. [[gc:568]] [[gc:574]] [[gc:580]] [[gc:597]] [[gc:611]] [[gc:618]] [[mo-constant-fold:757]] [[entry:82]]
