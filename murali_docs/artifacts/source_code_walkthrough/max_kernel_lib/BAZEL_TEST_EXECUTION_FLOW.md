<details>
<summary><strong>Overview Through Test Phase: expand or collapse the complete runtime sequence</strong></summary>

<details>
<summary><strong>One-Screen Flow Through Test Phase</strong></summary>

**Evidence:** `SOURCE`

## One-Screen Flow

```text
run.sh
  |
  v
./bazelw test ...:add_one_demo_test
  |
  +-- BUILD ---------------------------------------------------------+
  |   create Python launcher + runfiles + Mojo/MAX environment       |
  |                                                                 |
  +-- RUN -----------------------------------------------------------+
      test wrapper
        -> add_one_demo.py::main()
        -> Graph(...)
             -> load kernels/add_one.mojo
             -> create symbolic graph
        -> session.compile(graph)
        -> model.execute(input_tensor)
        -> demo_add_one kernel
        -> np.testing.assert_allclose(...)
        -> PASSED or FAILED
```

> `:kernels` is a separate build-check target. This test loads the raw
> `kernels/` directory through `custom_extensions`.

## Command Start

```text
cd "${repo_root}"
        |
        v
exec ./bazelw ...
        |
        +-- replaces the shell process
        +-- starts the repository-selected Bazel
```

## Build Phase

```text
//.../max_kernel_lib:add_one_demo_test
                  |
                  v
       modular_run_binary_test
                  |
                  v
            :add_one_demo
                  |
       +----------+-------------------+------------------+
       |                              |                  |
 add_one_demo.py              kernels/*.mojo       declared deps
 Python entry file             runtime data        MAX + NumPy + Mojo
```

```text
Bazel action required?
        |
   +----+----+
   |         |
  NO        YES
   |         |
cache hit   create launcher, runfiles, environment
   |         |
   +----+----+
        |
        v
test executable ready
```

Python remains Python source. Bazel assembles the executable environment; it
does not turn `add_one_demo.py` into a native machine-code binary.

## Test Phase

```text
BAZEL        PYTHON             GRAPH/MAX          MOJO           NUMPY
  |             |                  |                 |               |
  | run test    |                  |                 |               |
  |-----------> | main()           |                 |               |
  |             |----------------> | Graph(...)      |               |
  |             |                  |---------------> | load/register |
  |             |----------------> | compile graph   |               |
  |             |----------------> | execute         | add one       |
  |             |                  | <---------------|               |
  |             |--------------------------------------------------> | compare
  | <-----------| exit 0 or nonzero|                 |               |
```

</details>

<details>
<summary><strong>Step 1: Find the Mojo kernel directory</strong></summary>

```text
__file__
  = .../max_kernel_lib/add_one_demo.py
                 |
                 | Path(__file__).parent / "kernels"
                 v
mojo_kernels
  = .../max_kernel_lib/kernels/
```

```python
mojo_kernels = Path(__file__).parent / "kernels"
```

</details>

<details>
<summary><strong>Step 2: Create the hardcoded NumPy input</strong></summary>

```text
input_values: float32[2, 4]

+-----+-----+-----+-----+
| 0.0 | 1.5 | 2.0 | 3.5 |
+-----+-----+-----+-----+
| 4.0 | 5.5 | 6.0 | 7.5 |
+-----+-----+-----+-----+

rows = 2       columns = 4       dtype = float32
```

</details>

<details>
<summary><strong>Step 3: Select CPU or accelerator</strong></summary>

```text
accelerator_count() == 0 ?
          |
     +----+----+
    YES        NO
     |          |
   CPU()   Accelerator()
     |          |
     +----+-----+
          |
          v
device -> DeviceRef.from_device(device) -> device_ref
```

`device` is concrete runtime hardware. `device_ref` is its symbolic graph
identity, such as `cpu:0` or `gpu:0`.

</details>

<details>
<summary><strong>Step 4: Construct The MAX Graph</strong></summary>

<details>
<summary><strong>Exact Source And Sections 4.1-4.8</strong></summary>

Exact source: [add_one_demo.py](add_one_demo.py#L27)

```python
graph = Graph(
    "add_one_demo",
    forward=lambda x: (
        # x is the symbolic input created from input_types below.
        ops.custom(
            name="demo_add_one",
            # Execute the custom operation on the selected CPU/GPU.
            device=device_ref,
            values=[x],
            out_types=[
                # The output resides on device_ref; no input copy occurs here.
                TensorType(
                    dtype=x.dtype,
                    shape=x.tensor.shape,
                    device=device_ref,
                )
            ],
        )[0].tensor  # Return the first symbolic output as a TensorValue.
    ),
    # model.execute() must receive its input buffer on device_ref.
    input_types=[
        TensorType(
            dtype,
            shape=[rows, columns],
            device=device_ref,
        ),
    ],
    # Load and register demo_add_one from the Mojo source directory.
    custom_extensions=[mojo_kernels],
)
```

### 4.1 What Is `Graph`?

```text
Python Graph object
        |
        | records operations and tensor edges
        v
internal MLIR/MAX graph
        |
        v
compiled executable model
```

`Graph` is a Python class, not a Mojo struct. Source:
[graph.py](../../../../max/python/max/graph/graph.py#L682).

The kernel implementation `DemoAddOne` is a Mojo struct:
[add_one.mojo](kernels/add_one.mojo#L22).

### 4.2 `forward`, `lambda`, And `x`

```text
input_types has one TensorType
              |
              v
self.inputs = [symbolic TensorValue]
              |
              | forward(*self.inputs)
              v
lambda x: ops.custom(...)[0].tensor
       ^
       +-- x is the symbolic graph input
```

Equivalent Python:

```python
def forward(x):
    outputs = ops.custom(...)
    return outputs[0].tensor
```

### 4.3 Symbolic `x` Versus Runtime Data

```text
GRAPH CONSTRUCTION                     MODEL EXECUTION
------------------                     ---------------
x: symbolic TensorValue                input_tensor: real Buffer
dtype  = float32                              |
shape  = [2, 4]                               | model.execute(...)
device = device_ref                           v
        |                               binds to graph input x
        +---------- no array data ------------+
```

### 4.4 Custom-Op Input

```text
symbolic x
    |
    | values=[x]
    v
ops.custom(name="demo_add_one")
    |
    | runtime binding
    v
DemoAddOne.execute(..., x: InputTensor, ...)
```

### 4.5 Output Contract

```text
INPUT x                        DECLARED OUTPUT
--------------------------     --------------------------
dtype  = x.dtype          ---> dtype  = x.dtype
shape  = x.tensor.shape   ---> shape  = x.tensor.shape
device = device_ref       ---> device = device_ref
```

`TensorType` describes the output. It does not allocate or calculate it.

### 4.6 `ops.custom(...)[0].tensor`

```text
out_types = [TensorType(...)]
                 |
                 v
ops.custom(...) returns [output_value]
                         |
                         | [0]
                         v
                    output_value
                         |
                         | .tensor
                         v
                    TensorValue
```

Source: [`ops.custom()`](../../../../max/python/max/graph/ops/custom.py#L58).

### 4.7 Device Placement

```text
device_ref in TensorType       -> expected tensor location
device_ref in ops.custom       -> custom-op execution target
Buffer(...).to(device)         -> actual data movement
```

`device_ref` declares placement; `.to(device)` moves the buffer.

### 4.8 Custom Extension Registration

```text
custom_extensions=[kernels/]
             |
             v
compile Mojo source package
             |
             v
find @extensibility.register("demo_add_one")
             |
             v
verify ops.custom(name="demo_add_one")
             |
             v
attach kernel library to graph
```

Without `custom_extensions`, the graph cannot find the `demo_add_one`
implementation.

</details>

<details>
<summary><strong>Graph Constructor Implementation: Step By Step</strong></summary>

**Evidence:** `SOURCE`

Source: [`Graph.__init__`](../../../../max/python/max/graph/graph.py#L793)

```text
Graph(...)
|
+-- 1. Save graph identity and placement policy
|
|   +-- self.name = name
|   |   |
|   |   +-- Stores the graph identifier
|   |   +-- name = "add_one_demo"
|   |   +-- It becomes the graph/model symbol, approximately:
|   |       mo.graph @add_one_demo
|   |
|   +-- self.strict_device_placement = strict_device_placement
|       |
|       +-- Stores the policy for handling operations that cannot run on
|       |   the tensor's current device
|       |
|       +-- Device Placement
|       |   |
|       |   +-- Describes where a tensor resides and where an operation executes
|       |   |
|       |   +-- TensorType(..., device=DeviceRef.GPU(0))
|       |   |   +-- Tensor data must reside on GPU:0
|       |   |
|       |   +-- ops.custom(..., device=DeviceRef.GPU(0))
|       |       +-- Custom operation executes on GPU:0
|       |
|       +-- Strict Placement Policy
|           |
|           +-- Suppose x is on GPU, but an operation only has a CPU implementation
|           |
|           |   x on GPU
|           |      |
|           |      | CPU-only operation
|           |      v
|           |   Requires GPU -> CPU transfer
|           |
|           +-- strict_device_placement controls how MAX reports that transfer
|           |
|           |   from max.graph import DevicePlacementPolicy
|           |
|           |   Graph(
|           |       ...,
|           |       strict_device_placement=DevicePlacementPolicy.Warn,
|           |   )
|           |
|           +-- Policy behavior
|           |   +-- Ignore: allow the operation-specific transfer without warning
|           |   +-- Warn: allow the transfer and emit a warning; default behavior
|           |
|           +-- Internally, a CPU-only operation may perform:
|           |
|           |   _check_device_placement("ops.some_cpu_only_op")
|           |   cpu_x = ops.transfer_to(x, DeviceRef.CPU())
|           |
|           +-- With Error, _check_device_placement() raises before the transfer
|           |
|           +-- Important
|           |   +-- strict_device_placement does not select the device
|           |   +-- strict_device_placement does not copy runtime buffers itself
|           |   +-- It does not automatically fix arbitrary CPU/GPU mismatches
|           |
|           +-- Device selection is expressed separately through:
|               +-- TensorType.device
|               +-- operation device=
|               +-- explicit ops.transfer_to() calls
|
+-- 2. Optional path-based loading
|   |
|   |   if path is not None:
|   |       self._load_mlir(path)
|   |       return
|   |
|   +-- If path is supplied, load an existing graph
|   +-- Otherwise, construct a new graph
|
+-- 3. Collect symbolic dimensions
|   |
|   |   self._params = dict.fromkeys(
|   |       dim.name
|   |       for t in input_types
|   |       if isinstance(t, TensorType | BufferType)
|   |       for dim in t.shape
|   |       if isinstance(dim, SymbolicDim)
|   |   )
|   |
|   +-- Example:
|       shape = ["batch", 1024]
|                  |
|                  +-- self._params = {"batch": None}
|
+-- 4. Initialize graph-construction state
|   |
|   |   self._context_state = []
|   |   self._data_dependent_dims = 0
|   |   self._should_verify_ops = True
|   |
|   +-- Context stack starts empty
|   +-- Operation verification is enabled
|
+-- 5. Create or reuse an MLIR builtin.module
|   |
|   |   self._module = (
|   |       module.mlir_module
|   |       if module
|   |       else builtin.ModuleOp(location=loc)
|   |   )
|   |
|   +-- module not supplied
|   |   +-- Create a new builtin.module
|   |
|   +-- module supplied
|       +-- Reuse it for subgraphs or multiple models
|
+-- 6. Create an MLIR builder
|   |
|   |   builder = OpBuilder(self._module.body.end)
|   |
|   +-- New operations are inserted at the end of the module
|
+-- 7. Create mo.graph @add_one_demo
|   |
|   |   op = _mo.GraphOp(
|   |       builder,
|   |       loc,
|   |       name=name,
|   |       input_types=[t.to_mlir() for t in input_types],
|   |       result_types=[],
|   |   )
|   |
|   +-- Converts every Python TensorType to an MLIR type
|   |
|   +-- For this example:
|       TensorType(float32, [2,4], device_ref)
|                 |
|                 v
|       symbolic MLIR block argument %input0
|
+-- 8. Attach symbolic shape parameters
|   |
|   |   op.input_parameters = kgen.ParamDeclArrayAttr(
|   |       [
|   |           kgen.ParamDeclAttr(p, si64)
|   |           for p in self._params
|   |       ]
|   |   )
|   |
|   +-- Empty for the fixed shape [2,4]
|
+-- 9. Save the graph operation and graph body
|   |
|   |   self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr)
|   |   self._current_block = self._mlir_op.regions[0].blocks[0]
|   |   self._graph_body = self._current_block
|   |
|   +-- _mlir_op
|   |   +-- mo.graph @add_one_demo
|   |
|   +-- _graph_body
|       +-- Block containing mo.custom and mo.output
|
+-- 10. Attach device information
|   |
|   |   self._populate_device_info_mapping()
|   |
|   +-- Adds module attribute:
|       mo.device_info_mapping
|           +-- CPU information
|           +-- Accelerator information, when available
|
+-- 11. Initialize weight and execution-chain state
|   |
|   |   self._weights = {}
|   |   self._has_chain_input = False
|   |   self.device_chains = _DeviceChainMap(self)
|   |
|   +-- _weights
|   |   +-- Empty because add-one has no model weights
|   |
|   +-- device_chains
|       +-- Tracks operation ordering per device
|
+-- 12. Create the always-ready chain
|   |
|   |   self._always_ready_chain = _ChainValue(
|   |       self._add_op_generated(
|   |           _mo.ChainCreateOp,
|   |           result=_mo.ChainType(),
|   |           inputs=[],
|   |           attach_profile_scopes=False,
|   |       )[0]
|   |   )
|   |
|   |   self._update_chain(self._always_ready_chain)
|   |
|   +-- Used for operations without earlier dependencies
|   +-- Example: graph-level host/device staging transfers
|
+-- 13. Create and populate KernelLibrary
|   |
|   |   self._kernel_library = kernel_library or KernelLibrary()
|   |
|   |   self._import_kernels(
|   |       _resolved_custom_extensions(custom_extensions)
|   |   )
|   |
|   +-- custom_extensions contains kernels/
|   |
|   +-- _import_kernels performs:
|       |
|       |   self._kernel_library.load_paths(paths)
|       |
|       +-- Mojo source directory detected
|       +-- Source package precompiled into .mojoc
|       +-- demo_add_one registration discovered
|
+-- 14. Initialize subgraph registry
|   |
|   |   self._subgraphs = {}
|   |
|   +-- Empty for this example
|
+-- 15. Optionally mark this as a device graph
|   |
|   |   if is_device_graph:
|   |       op.is_device_graph = builtin.UnitAttr()
|   |
|   +-- Not enabled in the add-one example
|
+-- 16. If forward was provided, enter Graph.current context
|   |
|   |   if forward is not None:
|   |       with self:
|   |           ...
|   |
|   +-- with self sets:
|       CURRENT_GRAPH = this Graph object
|       |
|       +-- ops.custom(...) can now find the active graph
|
+-- 17. Create Python wrappers for symbolic inputs
|   |
|   |   @property
|   |   def inputs(self):
|   |       return tuple(
|   |           Value.from_mlir(_Value._from_cmlir(arg))
|   |           for arg in self._graph_body.arguments
|   |       )
|   |
|   +-- MLIR %input0
|       |
|       v
|       Python symbolic Value x
|
+-- 18. Invoke forward with symbolic inputs
|   |
|   |   result = forward(*self.inputs, *args, **kwargs)
|   |
|   +-- For this example:
|       |
|       |   result = forward(x)
|       |
|       |   forward=lambda x: (
|       |       ops.custom(
|       |           name="demo_add_one",
|       |           values=[x],
|       |           ...
|       |       )[0].tensor
|       |   )
|       |
|       +-- ops.custom inserts:
|           %result = mo.custom "demo_add_one"(%input0)
|
+-- 19. Normalize forward's return value
|   |
|   |   outputs = (
|   |       ()
|   |       if result is None
|   |       else (result,)
|   |       if not isinstance(result, Iterable)
|   |       else result
|   |   )
|   |
|   +-- No result    -> ()
|   +-- One result   -> (result,)
|   +-- Many results -> result iterable
|
+-- 20. Finalize the graph outputs
    |
    |   self.output(*outputs)
    |
    +-- Inserts:
    |   mo.output %result
    |
    +-- Builds input/output function type
    +-- Adds argument_names = ["input0"]
    +-- Adds result_names = ["output0"]
    +-- Verifies the complete mo.graph

The resulting structure is conceptually:

builtin.module {
    mo.graph @add_one_demo(%input0: float32[2,4]) {
        %chain = mo.chain.create
        %result = mo.custom "demo_add_one"(%input0)
        mo.output %result
    }
}

The constructor stops here:

Graph(...)                 Build symbolic graph
session.compile(graph)     Compile symbolic graph
session.init(compiled)     Initialize runtime model
model.execute(buffer)      Execute on real data

It does not allocate the runtime input buffer, copy it to a GPU, or execute
demo_add_one.
```

</details>

<details>
<summary><strong>Q&amp;A: What Does <code>Graph</code> Create And Manage?</strong></summary>

**Evidence:** `SOURCE`

```text
Python Graph object
|
+-- MAX-specific Python class
|   +-- Implemented in max/python/max/graph/graph.py
|   +-- Builds a symbolic computation
|   +-- Does not contain runtime tensor data
|   +-- Does not execute the computation
|
+-- MLIR builtin.module
|   |
|   +-- Top-level compilation container
|   +-- Similar to a source file or compilation unit
|   +-- Can contain multiple mo.graph operations
|   |
|   +-- Module-level device information
|   |   +-- mo.device_info_mapping
|   |       +-- Device label: cpu:0, gpu:0, ...
|   |       +-- Device API
|   |       +-- Architecture
|   |       +-- Device model
|   |
|   +-- mo.graph @add_one_demo
|   |   |
|   |   +-- Represents one model entry point
|   |   +-- Similar to a function inside a compilation unit
|   |   |
|   |   +-- Graph attributes
|   |   |   +-- name = "add_one_demo"
|   |   |   +-- input/output function signature
|   |   |   +-- argument_names = ["input0"]
|   |   |   +-- result_names = ["output0"]
|   |   |   +-- custom-kernel library paths
|   |   |
|   |   +-- Symbolic input block argument: %input0
|   |   |   +-- dtype = float32
|   |   |   +-- shape = [2, 4]
|   |   |   +-- device = device_ref
|   |   |   +-- Exposed to forward() as symbolic x
|   |   |
|   |   +-- Device dependency chains
|   |   |   +-- Preserve execution order
|   |   |   +-- Track CPU operations
|   |   |   +-- Track GPU:0 and GPU:1 independently
|   |   |   +-- Coordinate transfers and kernel execution
|   |   |
|   |   +-- Always-ready chain
|   |   |   +-- Used for work without an earlier dependency
|   |   |
|   |   +-- mo.custom "demo_add_one"
|   |   |   +-- Inserted by ops.custom(...)
|   |   |   +-- Input = symbolic %input0
|   |   |   +-- Execution device = device_ref
|   |   |   +-- Output = symbolic %result
|   |   |   +-- Matches:
|   |   |       @extensibility.register("demo_add_one")
|   |   |
|   |   +-- mo.output %result
|   |       +-- Marks %result as the graph output
|   |       +-- Finalizes the graph signature
|   |       +-- Triggers complete graph verification
|   |
|   +-- Optional additional top-level graphs
|   |   +-- mo.graph @encoder
|   |   +-- mo.graph @decoder
|   |
|   +-- Optional subgraphs
|       |
|       +-- Created using:
|       |
|       |   with graph.add_subgraph(
|       |       "add_one_block",
|       |       input_types=[input_type],
|       |   ) as sub:
|       |       x = sub.inputs[0].tensor
|       |       sub.output(ops.elementwise.add(x, one))
|       |
|       +-- Creates another Python Graph object
|       |   +-- Passes module=parent.module
|       |       +-- Parent and subgraph share one MLIR module
|       |
|       +-- Creates another mo.graph operation
|       |   +-- mo.graph @add_one_block
|       |   +-- Marked with is_subgraph
|       |   +-- Not loaded as an independent runtime Model
|       |   +-- Used as a reusable callable definition
|       |
|       +-- Called from the parent using:
|       |
|       |   result = ops.call(sub, parent_input)
|       |
|       +-- ops.call inserts:
|       |
|       |   %result = mo.call @add_one_block(%parent_input)
|       |
|       +-- Argument handling
|       |   +-- Checks argument count
|       |   +-- Arguments must match subgraph input types
|       |   +-- Returns symbolic subgraph outputs
|       |
|       +-- Device-chain handling
|       |   +-- Chain arguments are added automatically
|       |   +-- Parent chains are passed into the subgraph
|       |   +-- Updated chains are returned to the parent
|       |
|       +-- Weight-prefix support
|       |   +-- One definition can represent repeated layers
|       |   +-- Call 0 can use prefix="layers.0."
|       |   +-- Call 1 can use prefix="layers.1."
|       |
|       +-- Benefit
|       |   +-- Compiler processes repeated definition once
|       |
|       +-- Trade-offs
|           +-- Fusion cannot normally cross subgraph boundaries
|           +-- Internal allocations may not be shared outside
|
+-- Graph-construction context
|   |
|   +-- Uses CURRENT_GRAPH
|   |   +-- Python ContextVar containing the active Graph
|   |
|   +-- Entering:
|   |
|   |   with graph:
|   |       ...
|   |
|   |   performs:
|   |
|   |   token = CURRENT_GRAPH.set(graph)
|   |
|   +-- Leaving:
|   |
|   |   CURRENT_GRAPH.reset(token)
|   |   +-- Restores the previous graph
|   |
|   +-- Tracks current MLIR block
|   |   +-- Normally the body of mo.graph
|   |   +-- Can change for conditions and loops
|   |   +-- Restored after leaving nested blocks
|   |
|   +-- Tracks current insertion point
|   |   +-- New operations are appended to the active block
|   |
|   +-- Tracks block-local state
|   |   +-- Device chains are saved and restored
|   |   +-- Weight-deduplication state is saved and restored
|   |
|   +-- Graph constructor flow
|       |
|       +-- Step 1: Create symbolic inputs
|       |
|       |   input_types=[
|       |       TensorType(float32, [2,4], device_ref)
|       |   ]
|       |
|       |   creates:
|       |       symbolic block argument %input0
|       |
|       +-- Step 2: Call forward()
|       |
|       |   result = forward(*self.inputs)
|       |
|       |   Effectively:
|       |       result = forward(%input0)
|       |
|       +-- Step 3: Execute Python lambda
|       |
|       |   forward=lambda x: ops.custom(...)[0].tensor
|       |
|       |   Important:
|       |   +-- Lambda runs during graph construction
|       |   +-- x is symbolic
|       |   +-- No numerical calculation occurs yet
|       |
|       +-- Step 4: ops.custom(...) deep dive
|       |   +-- Reads Graph.current
|       |   +-- Converts name to MLIR symbol attribute
|       |   +-- Converts DeviceRef to MLIR device attribute
|       |   +-- Converts values=[x] to MLIR operands
|       |   +-- Converts out_types to MLIR result types
|       |   +-- Creates and inserts mo.custom
|       |   +-- Adds compile-time parameters
|       |   +-- Verifies against KernelLibrary
|       |   +-- Returns symbolic Python Value wrappers
|       |
|       +-- Step 5: Record output
|
|           graph.output(result)
|               +-- Creates mo.output
|               +-- Updates graph signature
|               +-- Assigns input/output names
|               +-- Verifies completed graph
|
+-- Operation builder
|   |
|   +-- Example Python expression
|   |
|   |   result = ops.custom(
|   |       name="demo_add_one",
|   |       values=[x],
|   |       out_types=[output_type],
|   |       device=device_ref,
|   |   )
|   |
|   +-- Converts Python wrappers to MLIR SSA values
|   |
|   |   Python:
|   |       x = TensorValue(_mlir_value=%input0)
|   |
|   |   Builder unwraps:
|   |       x -> %input0
|   |
|   |   MLIR receives:
|   |       mo.custom(%input0)
|   |
|   +-- Inserts operation into graph body
|   |
|   |   Before:
|   |
|   |   mo.graph @add_one_demo(%input0) {
|   |   }
|   |
|   |   After:
|   |
|   |   mo.graph @add_one_demo(%input0) {
|   |       %result = mo.custom "demo_add_one"(%input0)
|   |   }
|   |
|   +-- Converts MLIR results to Python wrappers
|   |
|   |   MLIR:
|   |       %result
|   |
|   |   Python:
|   |       Value(_mlir_value=%result)
|   |
|   |   Tensor interface:
|   |       ops.custom(...)[0].tensor
|   |
|   |   This permits additional symbolic operations:
|   |       y = ops.add(result, another_tensor)
|   |
|   +-- Records source locations
|       +-- Enable:
|       |   Graph.debug.source_tracebacks = True
|       +-- MLIR operation stores Python call-stack information
|       +-- Errors can point to:
|           add_one_demo.py:31
|           ops.custom(...)
|
+-- Operation verifier
|   |
|   +-- Input count
|   |   +-- Correct: values=[x]
|   |   +-- Incorrect: values=[]
|   |       +-- Kernel expects an input, so verification fails
|   |
|   +-- Input types
|   |   +-- Checks dtype
|   |   +-- Checks rank
|   |   +-- Checks tensor/buffer category
|   |   +-- Checks compatibility with registered signature
|   |
|   +-- Output count
|   |   +-- Correct: out_types=[TensorType(...)]
|   |   +-- Incorrect: out_types=[]
|   |       +-- DemoAddOne produces one output
|   |
|   +-- Output types
|   |   +-- Expected: float32[2,4]
|   |   +-- Incompatible dtype/rank can fail verification
|   |
|   +-- Device placement
|   |   +-- x.device = gpu:0
|   |   +-- custom-op device = gpu:0
|   |   +-- output device = gpu:0
|   |
|   +-- Required attributes
|   |   +-- Non-empty operation symbol
|   |   +-- Device attribute
|   |   +-- Result types
|   |   +-- Valid parameter types
|   |
|   +-- Kernel existence
|
|       Python:
|           name="demo_add_one"
|
|       Must match:
|           @extensibility.register("demo_add_one")
|
|       An unknown name causes graph construction to fail
|
+-- KernelLibrary
|   +-- Receives custom_extensions=[mojo_kernels]
|   +-- Recognizes kernels/ through __init__.mojo
|   +-- Precompiles source into a .mojoc package
|   +-- Discovers registered custom operations
|   +-- Verifies custom-op signatures
|   +-- Supplies declarations and paths to compilation
|
+-- Weight registry
|   |
|   +-- Used for learned/static model parameters
|   |
|   +-- Example:
|   |
|   |   weight = Weight(
|   |       "linear.weight",
|   |       DType.float32,
|   |       [1024, 1024],
|   |       device=DeviceRef.GPU(0),
|   |   )
|   |
|   +-- First use creates:
|   |
|   |   %weight = mo.constant.external "linear.weight"
|   |
|   +-- Actual data is supplied during initialization:
|   |
|   |   session.init(
|   |       compiled,
|   |       weights_registry={
|   |           "linear.weight": actual_array
|   |       },
|   |   )
|   |
|   +-- Registry entry
|   |   +-- _weights["linear.weight"]
|   |       +-- Original Weight object
|   |       +-- Symbolic TensorValue
|   |
|   +-- Duplicate prevention
|   |   +-- Same Weight object used twice
|   |   |   +-- Reuses existing symbolic value
|   |   +-- Different Weight object with same name
|   |       +-- Raises ValueError
|   |
|   +-- Add-one example
|       +-- No weights, so registry is empty
|
+-- Symbolic-dimension registry
|   +-- Tracks dimensions such as "batch"
|   +-- Example shape: ["batch", 1024]
|   +-- Adds "batch" to graph parameters
|   +-- Add-one uses fixed [2,4], so registry is empty
|
+-- Device-placement state
|   |
|   +-- Can one graph contain multiple DeviceRefs?
|   |   +-- Yes
|   |       +-- CPU:0
|   |       +-- GPU:0
|   |       +-- GPU:1
|   |
|   +-- Example multi-device graph
|   |
|   |   cpu_x = graph input on CPU:0
|   |       |
|   |       +-- Programmer explicitly writes:
|   |       |
|   |       |   gpu_x = cpu_x.to(DeviceRef.GPU(0))
|   |       |
|   |       +-- During graph construction
|   |       |   +-- Python executes TensorValue.to(...)
|   |       |       +-- Graph inserts:
|   |       |           mo.transfer CPU:0 -> GPU:0
|   |       |
|   |       +-- During compilation
|   |       |   +-- Compiler sees the existing mo.transfer
|   |       |   +-- Lowers it to a runtime transfer
|   |       |   +-- May optimize or reschedule it
|   |       |
|   |       +-- During model.execute()
|   |           +-- Runtime performs the actual memory copy
|   |
|   +-- Does compilation automatically insert the transfer?
|   |   |
|   |   +-- No
|   |       +-- Compiler does not insert Python code:
|   |       |
|   |       |   cpu_x.to(DeviceRef.GPU(0))
|   |       |
|   |       +-- Programmer normally inserts it explicitly
|   |       +-- Missing placement can cause verification,
|   |           compilation or operation-specific errors
|   |
|   +-- Exception: operation-specific implicit transfers
|   |   +-- Some MAX operations only support CPU execution
|   |   +-- Their Python implementation may internally call:
|   |   |
|   |   |   ops.transfer_to(x, DeviceRef.CPU())
|   |   |
|   |   +-- Controlled by strict_device_placement
|   |       +-- Ignore: transfer without warning
|   |       +-- Warn: transfer and emit warning
|   |       +-- Error: reject graph construction
|   |
|   +-- Records where operations execute
|   |
|   |   ops.custom(
|   |       device=DeviceRef.GPU(0),
|   |       ...
|   |   )
|   |
|   |   records:
|   |       demo_add_one executes on GPU:0
|   |
|   +-- Records where results reside
|   |
|   |   TensorType(
|   |       dtype=DType.float32,
|   |       shape=[2,4],
|   |       device=DeviceRef.GPU(0),
|   |   )
|   |
|   |   records:
|   |       Result resides on GPU:0 at runtime
|   |
|   +-- DeviceRef itself does not copy data
|   |   |
|   |   +-- DeviceRef is symbolic graph metadata
|   |   |
|   |   +-- Graph-level transfer
|   |   |   |
|   |   |   |   gpu_x = cpu_x.to(DeviceRef.GPU(0))
|   |   |   |
|   |   |   +-- Records mo.transfer
|   |   |       +-- Copy occurs during model execution
|   |   |
|   |   +-- Runtime Buffer transfer
|   |       |
|   |       |   gpu_buffer = cpu_buffer.to(Accelerator())
|   |       |
|   |       +-- Copies an actual buffer before model.execute()
|   |
|   +-- Current add-one example
|
|       Graph input is declared on device_ref:
|
|       input_types=[
|           TensorType(..., device=device_ref)
|       ]
|
|       Python prepares the real buffer:
|
|       input_tensor = Buffer.from_numpy(input_values).to(device)
|
|       Therefore:
|       +-- Input arrives on the required CPU/GPU
|       +-- No graph-level mo.transfer is needed
|       +-- demo_add_one can execute directly
|
+-- Profiling and debugging state
|   +-- Source traceback locations
|   +-- Profile-scope labels
|   +-- Captured MLIR diagnostics
|
+-- Compilation and execution handoff
    |
    +-- session.compile(graph)
        +-- Reads graph.module
        +-- Gets the complete MLIR builtin.module
        +-- Adds custom-kernel declarations
        +-- Passes module to native MAX
        +-- Returns CompiledModel
            |
            +-- session.init(compiled)
                +-- Creates executable Model
                    |
                    +-- model.execute(input_tensor)
                        +-- Executes transfers and kernels
                            on real runtime buffers
```

</details>

</details>

<details>
<summary><strong>Step 5: Supply the custom Mojo sources</strong></summary>

```text
mojo_kernels Path
      |
      v
custom_extensions=[mojo_kernels]
      |
      v
kernels/add_one.mojo available to Graph
```

</details>

<details>
<summary><strong>Step 6: Compile the complete graph</strong></summary>

```text
symbolic Graph + registered custom kernel
                    |
                    | session.compile(graph)
                    v
                 compiled
```

The Mojo source package is loaded during `Graph(...)`; this step compiles the
complete graph.

</details>

<details>
<summary><strong>Step 7: Initialize and execute</strong></summary>

```text
compiled
   | session.init(compiled)
   v
 model
   | model.execute(input_tensor)
   v
[result Buffer]
```

</details>

<details>
<summary><strong>Step 8: Execute the Mojo add-one kernel</strong></summary>

```text
INPUT                              OUTPUT
+-----+-----+-----+-----+          +-----+-----+-----+-----+
| 0.0 | 1.5 | 2.0 | 3.5 |          | 1.0 | 2.5 | 3.0 | 4.5 |
+-----+-----+-----+-----+   +1     +-----+-----+-----+-----+
| 4.0 | 5.5 | 6.0 | 7.5 | ------> | 5.0 | 6.5 | 7.0 | 8.5 |
+-----+-----+-----+-----+          +-----+-----+-----+-----+
```

</details>

<details>
<summary><strong>Step 9: Verify the result with NumPy</strong></summary>

```text
result_values -------------------+
                                  | np.testing.assert_allclose
expected = input_values + 1.0 ---+
                                  |
                           +------+------+
                           |             |
                         equal       mismatch
                           |             |
                          pass      AssertionError
```

</details>

<details>
<summary><strong>Step 10: Print the values</strong></summary>

```text
stdout
  +-- Device
  +-- Input
  +-- Graph result
  +-- Expected
```

</details>

</details>

<details>
<summary><strong>Pass Or Fail Through Flag Roles</strong></summary>

## Pass Or Fail

```text
main() completed?
       |
  +----+----+
 YES       NO
  |         |
exit 0   exception / assertion / compile failure
  |         |
PASSED    nonzero exit
            |
          FAILED
```

`--test_output=all` prints the program output and failure details.

## Flag Roles

| Flag | Effect |
|---|---|
| `--output_user_root=...` | Bazel state and generated outputs |
| `--disk_cache=...` | Reuse cached build actions |
| `--repo_contents_cache=` | Disable repository-content cache |
| `--config=prebuilt-mojo` | Use prebuilt Mojo configuration |
| `--config=disable-lint` | Disable lint configuration |
| `--aspects=` | Clear configured aspects |
| `--output_groups=default` | Request normal outputs |
| `--enable_runfiles=true` | Enable runtime file trees |
| `--build_runfile_links=true` | Create runfile links |
| `--remote_download_outputs=all` | Download all remote outputs |

</details>
