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
```

<details>
<summary><strong>Step 4 Deep Dive: Graph-Construction State</strong></summary>

**Evidence:** `SOURCE`

```text
+-- 4. Initialize graph-construction state
    |
    +-- What does "graph construction" mean?
    |   |
    |   +-- Python executes forward() using symbolic values
    |   +-- ops.custom(), ops.add(), etc. create MLIR operations
    |   +-- Operations are appended to the active mo.graph body
    |   +-- Types, devices and dependencies are verified
    |   +-- No real tensor calculation occurs during this phase
    |   |
    |   |   Python forward(symbolic x)
    |   |               |
    |   |               | ops.custom(...)
    |   |               v
    |   |       insert symbolic mo.custom
    |   |               |
    |   |               v
    |   |         MLIR/MAX graph
    |
    +-- self._context_state = []
    |   |
    |   +-- A Python list used as a stack of active graph contexts
    |   |
    |   +-- "Stack starts empty" means:
    |   |   +-- No with graph: block has been entered yet
    |   |   +-- This graph is not currently Graph.current
    |   |   +-- No previous graph context needs to be restored
    |   |
    |   +-- It is NOT:
    |   |   +-- A GPU execution stack
    |   |   +-- A tensor stack
    |   |   +-- The Python function-call stack
    |   |   +-- A list of MLIR operations
    |   |
    |   +-- Entering with graph:
    |   |
    |   |   def __enter__(self):
    |   |       self._context_state.append(
    |   |           state := self._enter()
    |   |       )
    |   |       return state.__enter__()
    |   |
    |   |   State change:
    |   |
    |   |   [] -> [context_1]
    |   |
    |   |   _enter() performs:
    |   |
    |   |   token = CURRENT_GRAPH.set(self)
    |   |
    |   |   Therefore:
    |   |
    |   |   Graph.current
    |   |       |
    |   |       v
    |   |   returns this graph
    |   |
    |   |   ops.custom(...)
    |   |       |
    |   |       | reads Graph.current
    |   |       v
    |   |   inserts mo.custom into this graph
    |   |
    |   +-- Leaving with graph:
    |   |
    |   |   def __exit__(self, *exc):
    |   |       self._context_state.pop().__exit__(*exc)
    |   |
    |   |   CURRENT_GRAPH.reset(token)
    |   |
    |   |   State change:
    |   |
    |   |   [context_1] -> []
    |   |
    |   +-- Why use a stack?
    |       |
    |       +-- Nested graph contexts exit in reverse order
    |       |
    |       |   with graph:          # context_1
    |       |       with graph:      # context_2
    |       |           ops.custom(...)
    |       |
    |       +-- State evolution:
    |           +-- Start:         []
    |           +-- Enter outer:   [context_1]
    |           +-- Enter inner:   [context_1, context_2]
    |           +-- Exit inner:    [context_1]
    |           +-- Exit outer:    []
    |
    +-- self._data_dependent_dims = 0
    |   |
    |   +-- Counter for dimensions whose size depends on runtime data
    |   |
    |   +-- Example:
    |   |
    |   |   input = [0, 5, 0, 8]
    |   |              |
    |   |              | nonzero(input)
    |   |              v
    |   |   output = [1, 3]
    |   |
    |   |   Output length depends on input values, not only input shape
    |   |
    |   +-- MAX assigns stable internal ordinals:
    |   |   +-- First data-dependent dimension  -> 0
    |   |   +-- Second data-dependent dimension -> 1
    |   |
    |   +-- Internal update:
    |   |
    |   |   index = self._data_dependent_dims
    |   |   self._data_dependent_dims = index + 1
    |   |   return index
    |   |
    |   +-- Add-one output shape is fixed [2,4]
    |       +-- Counter remains 0
    |
    +-- self._should_verify_ops = True
        |
        +-- Verify each MLIR operation as it is inserted
        |
        |   def _verify_op(self, op):
        |       if self._should_verify_ops:
        |           op.verify()
        |
        +-- Can detect invalid:
        |   +-- Input types
        |   +-- Output types
        |   +-- Result count
        |   +-- Required attributes
        |   +-- MLIR operation structure
        |
        +-- Verification may be temporarily paused:
        |
        |   with graph._pause_verification():
        |       # Construct temporarily incomplete operation state
        |
        +-- graph.output(...) later verifies the complete graph
        |
        +-- Verification checks symbolic structure
            +-- It does not execute the operation
            +-- It does not inspect runtime tensor values

Constructor-time state evolution:

Before forward():                 _context_state = []
Enter with self:                  _context_state = [context_1]
Execute forward(symbolic x):      ops.custom inserted into graph
Call self.output(...):            graph finalized and verified
Leave with self:                  _context_state = []
```

</details>

```text
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
```

<details>
<summary><strong>Step 5 Deep Dive: Create A New <code>builtin.module</code></strong></summary>

**Evidence:** `SOURCE`

```text
+-- module not supplied
    |
    +-- Execute: builtin.ModuleOp(location=loc)
    |
    +-- What is created?
    |   |
    |   +-- One empty MLIR builtin.module operation
    |   +-- The top-level container for this compilation unit
    |   +-- NOT a normal Python module
    |   +-- NOT a compiled executable or runtime model
    |
    +-- Initial conceptual structure
    |   |
    |   |   builtin.module {
    |   |       // Empty module body block
    |   |   }
    |   |
    |   +-- The following constructor step inserts mo.graph here
    |
    +-- Important parts created by ModuleOp
    |   |
    |   +-- builtin.module operation
    |   +-- One body region
    |   +-- One body block
    |   +-- A symbol-table container for named operations such as mo.graph
    |
    +-- Python references
    |   |
    |   +-- self._module
    |   |   +-- Typed Python/nanobind wrapper for the MLIR ModuleOp
    |   |
    |   +-- self._module.body
    |   |   +-- The block that holds top-level operations
    |   |
    |   +-- self._module.body.end
    |       +-- Position immediately after the last top-level operation
    |
    +-- location=loc
    |   |
    |   +-- Attaches an MLIR source location for diagnostics
    |   +-- Does not mean a filesystem output directory
    |
    +-- For add_one_demo
    |   |
    |   |   self._module = builtin.ModuleOp(location=loc)
    |   |                         |
    |   |                         v
    |   |   builtin.module {
    |   |       // mo.graph @add_one_demo is inserted next
    |   |   }
    |   |
    |   +-- Later module-level data includes:
    |       +-- mo.graph @add_one_demo
    |       +-- mo.device_info_mapping
    |       +-- Any subgraph mo.graph operations
    |
    +-- This step does NOT
        +-- Compile MLIR
        +-- Allocate tensor buffers
        +-- Load runtime input values
        +-- Execute the Mojo kernel
```

</details>

```text
+-- 6. Create an MLIR builder
|   |
|   |   builder = OpBuilder(self._module.body.end)
|   |
|   +-- New operations are inserted at the end of the module
|
```

<details>
<summary><strong>Step 6 Deep Dive: <code>OpBuilder(self._module.body.end)</code></strong></summary>

**Evidence:** `SOURCE`

```text
+-- builder = OpBuilder(self._module.body.end)
    |
    +-- OpBuilder
    |   |
    |   +-- Imported from max._core
    |   +-- Python binding for an MLIR operation builder
    |   +-- Holds an insertion point used while creating an operation
    |   +-- NOT an operation in the final graph
    |   +-- NOT the MAX compiler or runtime executor
    |
    +-- self._module.body.end
    |   |
    |   +-- self._module       -> builtin.module
    |   +-- .body              -> module's top-level block
    |   +-- .end               -> position after its last operation
    |
    +-- Empty-module insertion point
    |   |
    |   |   builtin.module {
    |   |       <builder insertion point>
    |   |   }
    |
    +-- The builder is consumed by the next statement
    |   |
    |   |   op = _mo.GraphOp(
    |   |       builder,
    |   |       loc,
    |   |       name="add_one_demo",
    |   |       ...
    |   |   )
    |   |
    |   +-- _mo.GraphOp creates the mo.graph operation
    |   +-- builder inserts that operation at module.body.end
    |
    +-- Structure before and after
    |   |
    |   |   BEFORE                         AFTER
    |   |
    |   |   builtin.module {               builtin.module {
    |   |       <insert here>                  mo.graph @add_one_demo {...}
    |   |   }                              }
    |
    +-- If an existing module is reused
    |   |
    |   |   builtin.module {
    |   |       mo.graph @existing_graph {...}
    |   |       <builder insertion point>
    |   |   }
    |   |
    |   +-- The new mo.graph is appended after existing top-level operations
    |
    +-- Scope of this builder
        |
        +-- This builder inserts the top-level mo.graph into builtin.module
        +-- mo.custom and mo.output are inserted later into mo.graph's body
        +-- Creating the builder does not compile or execute anything
```

</details>

```text
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

<details>
<summary><strong>Step 6.1: InferenceSession: MAX Python Class, Constructor, And Usage</strong></summary>

**Evidence:** `SOURCE`

Source paths:

- Class: `max/python/max/engine/api.py:546`
- Constructor: `max/python/max/engine/api.py:650`
- Public re-export: `max/python/max/engine/__init__.py:25`
- Native implementation import: `max/python/max/engine/api.py:32`

```text
InferenceSession
|
+-- Is it a MAX-defined Python class?
|   |
|   +-- YES
|   +-- Public import: from max.engine import InferenceSession
|   +-- Python implementation: max/python/max/engine/api.py
|   +-- It is not a generic Python or machine-learning class
|   |
|   +-- Python wrapper
|       |
|       +-- Validates and normalizes user configuration
|       +-- Creates the native MAX session in self._impl
|       +-- Exposes compile(), init(), load(), and runtime controls
|
+-- Why is it called a session?
|   |
|   +-- It is the long-lived MAX compiler/runtime environment
|   +-- It owns the selected device set and runtime configuration
|   +-- Graphs are compiled and initialized through this environment
|   +-- It is NOT the Graph, CompiledModel, executable Model, or input Buffer
|
+-- Python and native layers
    |
    |   from max._core.engine import InferenceSession as _InferenceSession
    |
    |   class InferenceSession:
    |       ...
    |       self._impl = _InferenceSession(...)
    |
    +-- InferenceSession -> public MAX Python API
    +-- _InferenceSession -> native MAX engine binding
```

Constructor signature and operational core:

```python
def __init__(
    self,
    devices: Iterable[Device] = (),
    num_threads: int | None = None,
    *,
    custom_extensions: CustomExtensionsType | None = None,
    precompiled_mefs: str | Path | Iterable[str | Path] | None = None,
    export_mefs: str | Path | None = None,
) -> None:
    # A session cannot import and export precompiled artifacts simultaneously.
    if precompiled_mefs is not None and export_mefs is not None:
        raise ValueError(...)

    # Configure optional compiled-artifact reuse or export.
    self._mef_store: MefStore | None = None
    if precompiled_mefs is not None:
        self._mef_store = MefStore.for_import(precompiled_mefs)
    elif export_mefs is not None:
        self._mef_store = MefStore.for_export(export_mefs)

    self.num_threads = num_threads

    # Remove duplicate devices while preserving their order.
    final_devices: list[Device] = []
    seen_devices: set[Device] = set()
    for device in devices:
        if device not in seen_devices:
            final_devices.append(device)
            seen_devices.add(device)

    # The host CPU is always available for orchestration.
    host_cpu = CPU()
    if host_cpu not in seen_devices:
        final_devices.append(host_cpu)
        seen_devices.add(host_cpu)

    # Convert Python custom-extension inputs for the native API.
    custom_extensions_final = []
    if custom_extensions:
        custom_extensions_final = _process_custom_extensions_objects(
            custom_extensions
        )

    # Construct the native MAX compiler/runtime session.
    self._impl = _InferenceSession(
        final_devices,
        custom_extensions_final,
        num_threads or 0,
    )

    # The remaining constructor code installs crash-stack handling and applies
    # MAX debug, logging, assertion, and uninitialized-read configuration.
```

```text
InferenceSession(devices=[device])
|
+-- 1. Validate MEF import/export options
|
+-- 2. Store num_threads
|
+-- 3. Deduplicate requested devices
|   |
|   +-- [GPU:0, GPU:0] -> [GPU:0]
|
+-- 4. Add the host CPU when absent
|   |
|   +-- devices=[Accelerator(0)]
|       -> final_devices=[Accelerator(0), CPU()]
|
+-- 5. Process session-level custom extensions
|
+-- 6. Create the native engine session
|   |
|   +-- self._impl = _InferenceSession(...)
|
+-- 7. Apply diagnostics and debug configuration
|
+-- Constructor returns the Python session object
    |
    +-- No graph has been compiled yet
    +-- No executable Model has been initialized yet
    +-- No input tensor has been copied or executed yet
```

Constructor arguments:

```text
devices
  Requested CPU/accelerator devices. MAX automatically adds the host CPU.

num_threads
  Host execution-thread count. None becomes 0 for native automatic selection.

custom_extensions
  Session-level Mojo custom-operation packages or source inputs.

precompiled_mefs
  Reuse previously compiled MEF artifacts instead of compiling matching graphs.

export_mefs
  Write compiled graph artifacts for reuse by a later session.

precompiled_mefs and export_mefs are mutually exclusive.
```

Usage 1: this add-one example, using the recommended split workflow:

```python
device = Accelerator() if accelerator_count() > 0 else CPU()

session = InferenceSession(devices=[device])
compiled = session.compile(graph)       # Graph -> CompiledModel
model = session.init(compiled)          # CompiledModel -> executable Model
result = model.execute(input_tensor)[0] # Execute with a real Buffer
```

```text
session
  |
  +-- compile(graph) -> compiled artifact
  |
  +-- init(compiled) -> executable model bound to runtime resources
  |
  +-- model.execute(input_tensor) -> result buffer
```

Usage 2: CPU-only session:

```python
# With no explicit devices, the constructor still adds CPU().
session = InferenceSession()
compiled = session.compile(graph)
model = session.init(compiled)
```

Usage 3: accelerator session; CPU is still added automatically:

```python
gpu = Accelerator(0)
session = InferenceSession(devices=[gpu])

# Conceptual native device list:
# [Accelerator(0), CPU()]
```

Usage 4: convenience API:

```python
session = InferenceSession(devices=[device])
model = session.load(graph)  # Roughly compile(graph) + init(compiled)
```

New code should prefer separate `compile()` and `init()` calls because the
compiled artifact can then be reused or exported independently.

</details>

<details>
<summary><strong>Step 6.2: <code>InferenceSession.compile()</code>: Implementation</strong></summary>

**Evidence:** `SOURCE`

Source paths:

- `compile()`: `max/python/max/engine/api.py:1048`
- `compile_async()`: `max/python/max/engine/api.py:955`
- Native module handoff: `max/python/max/engine/api.py:1262`
- `CompiledModel`: `max/python/max/engine/api.py:400`

```text
session.compile(graph)
|
+-- 1. Record compilation timing
|
+-- 2. Call compile_async(graph)
|   |
|   +-- Normalize explicit custom_extensions
|   |
|   +-- Detect input kind
|   |   +-- Path/string -> native path compilation
|   |   +-- Graph       -> use graph.module
|   |   +-- Module      -> compile the supplied MLIR module
|   |   +-- Other       -> RuntimeError
|   |
|   +-- Graph-specific preparation
|   |   +-- Add graph.kernel_libraries_paths
|   |   +-- Capture expected weight devices
|   |   +-- Seed custom-kernel declarations into the MLIR module
|   |   +-- Record top-level mo.graph names
|   |
|   +-- Dispatch compilation
|       +-- Path/string -> self._impl.compile(path, ...)
|       +-- Graph/Module -> _compile_module(...)
|           +-- Acquire process-wide compilation lock
|           +-- Pass the MLIR module pointer to native MAX
|           +-- Return an asynchronous native handle
|           +-- Release the lock when compilation finishes
|
+-- 3. Wait for the asynchronous handle
|
+-- 4. Check the compilation result
|   +-- Success -> return CompiledModel
|   +-- Failure -> raise the compiler error with MAX context
|
+-- Returned CompiledModel
    +-- Contains a compiled artifact
    +-- Is not directly executable
    +-- Must be passed to session.init(...)
```

`compile()` is the blocking wrapper:

```python
def compile(
    self,
    model: str | Path | Module | Graph,
    *,
    custom_extensions: CustomExtensionsType | None = None,
    tile_based_fusion: bool = False,
) -> CompiledModel:
    with _record_phase("compile_seconds"):
        # Schedule compilation and receive a CompiledModel containing a
        # pending native asynchronous handle.
        compiled = self.compile_async(
            model,
            custom_extensions=custom_extensions,
            tile_based_fusion=tile_based_fusion,
        )
        handle = compiled._compiled
        assert isinstance(handle, _AsyncValue)

        # Make compile() synchronous.
        handle.wait()

    exception = handle.exception()
    if exception is None:
        return compiled

    # Special requested stop after emitting pre-JIT output.
    if self.debug.pre_jit and "pre-jit" in str(exception):
        raise CompilationStopped(str(exception)) from exception

    # Add MAX graph diagnostic guidance for Graph/Module failures.
    if isinstance(model, (Graph, Module)):
        raise RuntimeError(self._compile_failure_message()) from exception

    raise exception
```

For a `Graph`, `compile_async()` performs this preparation:

```python
module = model.module

# Add the custom-kernel libraries collected while Graph(...) was built.
custom_extensions_final.extend(
    _process_custom_extensions_objects(model.kernel_libraries_paths)
)

# Remember expected weight placement for session.init() validation.
if hasattr(model, "_weights"):
    expected_weights = {
        name: weight.value.device
        for name, weight in model._weights.items()
    }

# Insert known custom-kernel declarations and type mappings into the module.
kernel_library = getattr(model, "_kernel_library", None)
if kernel_library is not None:
    kernel_library._analysis.seed_kernel_decls(module.mlir_module)

# Start native compilation of the prepared MLIR module.
handle = self._compile_module(
    module,
    custom_extensions_final,
    tile_based_fusion,
)

compiled = CompiledModel(
    compiled=handle,
    expected_weights=expected_weights,
)
compiled._graph_names = tuple(module.top_level_graph_names())
return compiled
```

`_compile_module()` performs the native handoff:

```python
_COMPILATION_LOCK.acquire()
try:
    handle = self._impl.compile(
        module.mlir_module._CAPIPtr,
        custom_extensions_final,
        _derive_pipeline_name(module),
        tile_based_fusion,
    )
except Exception as error:
    _COMPILATION_LOCK.release()
    raise RuntimeError(self._compile_failure_message()) from error

# Native compilation is asynchronous. Release the global lock when it resolves.
handle.add_done_callback(lambda _: _COMPILATION_LOCK.release())
return handle
```

```text
Python Graph
    |
    | graph.module
    v
MLIR builtin.module
    |
    | seed demo_add_one declaration
    | add custom-kernel library paths
    v
module.mlir_module._CAPIPtr
    |
    | self._impl.compile(...)
    v
Native MAX compiler pipeline
    |
    | lower + optimize + generate compiled artifact
    v
_AsyncValue[_CompiledModels]
    |
    | handle.wait()
    v
CompiledModel
```

For this add-one example:

```text
session.compile(graph)
|
+-- graph is a Graph instance
+-- module contains mo.graph @add_one_demo
+-- KernelLibrary supplies demo_add_one declarations
+-- Weight registry is empty
+-- Native MAX compiles the complete graph
+-- compile() waits for completion
+-- Returns CompiledModel
```

Important boundary:

```text
compile(graph)
  +-- Compiles symbolic MLIR and custom-kernel references
  +-- Does not bind runtime weights
  +-- Does not allocate the input Buffer
  +-- Does not execute demo_add_one

init(compiled)
  +-- Creates the executable runtime Model

model.execute(input_tensor)
  +-- Executes with real runtime buffers
```

</details>

<details>
<summary><strong>Step 6.3: <code>_compile_module()</code>: Native Compilation Handoff</strong></summary>

**Evidence:** `SOURCE`

Source paths:

- `_compile_module()`: `max/python/max/engine/api.py:1262`
- `_derive_pipeline_name()`: `max/python/max/engine/api.py:351`
- Global compilation lock: `max/python/max/engine/api.py:94`

Call site:

```python
# Start native compilation of the prepared MLIR module.
handle = self._compile_module(
    module,
    custom_extensions_final,
    tile_based_fusion,
)
```

Argument meanings:

```text
_compile_module(
    module,
    custom_extensions_final,
    tile_based_fusion,
)
|
+-- module
|   +-- max.graph.Module Python wrapper
|   +-- Owns the prepared MLIR builtin.module
|   +-- For this example, contains mo.graph @add_one_demo
|
+-- custom_extensions_final
|   +-- Processed custom-kernel paths passed to native MAX
|   +-- Includes paths collected from graph.kernel_libraries_paths
|   +-- Makes demo_add_one available to the compiler
|
+-- tile_based_fusion
    +-- False in this example
    +-- False -> use the normal graph compilation mode
    +-- True  -> request the tile-based programming model
```

Exact Python implementation with comments:

```python
def _compile_module(
    self,
    module: Module,
    custom_extensions_final: list[CustomExtensionType],
    tile_based_fusion: bool = False,
) -> _AsyncValue[_CompiledModels]:
    # Only one Graph/Module compilation may be active process-wide.
    _COMPILATION_LOCK.acquire()

    try:
        # Enter the compiled/native MAX engine.
        handle = self._impl.compile(
            # Opaque native pointer to the MLIR builtin.module.
            module.mlir_module._CAPIPtr,

            # Processed .mojoc/.mojo custom-extension paths.
            custom_extensions_final,

            # Diagnostic name derived from top-level mo.graph symbols.
            _derive_pipeline_name(module),

            # Select normal or tile-based graph compilation.
            tile_based_fusion,
        )
    except Exception as error:
        # Native compilation could not even be scheduled.
        _COMPILATION_LOCK.release()
        raise RuntimeError(self._compile_failure_message()) from error

    # Native compilation continues asynchronously. The runtime worker that
    # resolves the handle releases the lock, even when compilation fails.
    handle.add_done_callback(lambda _: _COMPILATION_LOCK.release())
    return handle
```

Detailed execution flow:

```text
Python thread
|
+-- _COMPILATION_LOCK.acquire()
|   |
|   +-- Prevent another Graph/Module compilation from starting
|   +-- The lock is shared across every InferenceSession in this process
|
+-- Prepare native compile arguments
|   |
|   +-- MLIR pointer
|   |   module.mlir_module._CAPIPtr
|   |
|   +-- Custom extensions
|   |   custom_extensions_final
|   |
|   +-- Pipeline name
|   |   _derive_pipeline_name(module)
|   |
|   +-- Compilation mode
|       tile_based_fusion
|
+-- self._impl.compile(...)
|   |
|   +-- Crosses from Python into the compiled max._core engine
|   +-- Schedules native MAX graph compilation
|   +-- Returns _AsyncValue[_CompiledModels]
|
+-- Register completion callback
|   |
|   +-- Success or failure resolves the handle
|   +-- Runtime worker calls _COMPILATION_LOCK.release()
|
+-- Return handle to compile_async()
    |
    +-- compile_async() wraps it in CompiledModel
    +-- compile() calls handle.wait()
```

Why `_CAPIPtr` is passed:

```text
module                         module.mlir_module._CAPIPtr
Python max.graph.Module        Opaque native MLIR operation pointer
          |                                  |
          | wrapper access                   | native API argument
          v                                  v
Typed MLIR ModuleOp ----------> Native MAX compiler
```

`_CAPIPtr` is not a text serialization or Python-level copy. It exposes the
underlying native MLIR object to the compiled engine; the native binding/runtime
must retain everything required by the asynchronous compilation.

Pipeline-name derivation:

```python
def _derive_pipeline_name(module: Module) -> str:
    return "+".join(module.top_level_graph_names())
```

```text
One top-level graph:
  ["add_one_demo"] -> "add_one_demo"

Multiple top-level graphs:
  ["encoder", "decoder"] -> "encoder+decoder"
```

The pipeline name identifies the compilation in diagnostics. It does not rename
the `mo.graph` operations or choose the target CPU/GPU.

Asynchronous timeline:

```text
Python compile()          Native/runtime worker          Global lock
      |                           |                         |
      | acquire -----------------------------------------> HELD
      |                           |                         |
      | self._impl.compile() ---->| schedule compilation   |
      |<---- async handle --------|                         |
      | add_done_callback()       |                         |
      | return handle             | compile MLIR/kernels    |
      | handle.wait()             |                         |
      |                           | finish success/failure  |
      |                           | release callback ------> FREE
      |<---- handle resolves -----|                         |
      | inspect exception         |                         |
      v                           v                         v
 return CompiledModel
```

Two failure paths:

```text
Synchronous setup failure
  self._impl.compile(...) raises immediately
      -> release lock in except block
      -> raise RuntimeError with MAX diagnostic guidance

Asynchronous compilation failure
  native worker stores exception in handle
      -> completion callback releases lock
      -> compile() unblocks from handle.wait()
      -> compile() reads handle.exception() and raises
```

Values for this example:

```text
module
  -> builtin.module containing mo.graph @add_one_demo

custom_extensions_final
  -> processed kernel-library paths containing demo_add_one

_derive_pipeline_name(module)
  -> "add_one_demo"

tile_based_fusion
  -> False

return value
  -> pending _AsyncValue[_CompiledModels]
```

Important boundary:

```text
_compile_module()
  +-- Starts native compilation
  +-- Controls compilation serialization and lifetime handoff
  +-- Returns an asynchronous handle
  +-- Does not wait for completion itself
  +-- Does not create runtime input buffers
  +-- Does not execute the compiled model
```

</details>

<details>
<summary><strong>Step 6.4: How The Mojo Kernel Becomes CPU Or GPU Machine Code</strong></summary>

**Question:** At `self._impl.compile(...)`, how do our Mojo kernels get converted to CPU machine code and GPU machine code?

`self._impl.compile(...)` is the boundary where Python hands the prepared MLIR
module and custom-extension paths to the native MAX compiler. The source for
that native `max._core` implementation is not present in this checkout, but the
code before and after that boundary shows this call flow:

```text
InferenceSession.compile(graph)
    |
    v
compile_async()
    |
    +-- Seed declarations for kernels used by mo.custom
    +-- Add KernelLibrary/custom-extension paths
    v
_compile_module()
    |
    v
self._impl.compile(module.mlir_module._CAPIPtr, ...)
    |
    +-- Enter native MAX graph compiler
    +-- Resolve mo.custom "demo_add_one"
    +-- Specialize DemoAddOne.execute[target=...]
    v
+---------------------------+-----------------------------+
| CPU placement             | GPU placement               |
+---------------------------+-----------------------------+
| target = "cpu"            | target = "gpu"              |
| foreach[target="cpu"]     | foreach[target="gpu"]      |
| _elementwise_impl_cpu     | _elementwise_impl_gpu       |
| sync_parallelize          | grid-stride or SM100 CLC    |
| SIMD vectorize            | ctx.enqueue_function(...)   |
| host LLVM backend         | GPU LLVM backend            |
| x86-64/AArch64 code       | NVIDIA or AMD device code   |
+---------------------------+-----------------------------+
```

The custom kernel is registered here:

```mojo
@extensibility.register("demo_add_one")
struct DemoAddOne:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        ...
        foreach[target=target](add_one, output, ctx)
```

`target` is a compile-time Mojo parameter. MAX specializes the same generic
kernel implementation for the graph's selected device:

```text
CPU graph/device
    -> DemoAddOne.execute[target="cpu"]
    -> CPU implementation of foreach
    -> native host machine code

GPU graph/device
    -> DemoAddOne.execute[target="gpu"]
    -> GPU implementation of foreach
    -> accelerator device code plus a host-side launcher
```

The CPU path is:

```text
foreach[target="cpu"]
    |
    v
_elementwise_impl_cpu
    |
    +-- Divide work among CPU workers with sync_parallelize
    +-- Vectorize inner work with CPU SIMD
    v
Mojo/MLIR lowering
    |
    v
LLVM host target backend
    |
    v
x86-64 or AArch64 machine code
```

The GPU path is:

```text
foreach[target="gpu"]
    |
    v
_elementwise_impl_gpu
    |
    +-- Choose block size and grid size
    +-- Build a GPU kernel closure
    +-- Emit ctx.enqueue_function(...)
    v
DeviceFunction / _compile_code / compile_info
    |
    v
kgen.compile_offload
    |
    v
GPU-oriented Mojo/MLIR -> LLVM IR -> target GPU backend
    |
    +-- NVIDIA: NVPTX/PTX and a loadable CUDA module image
    +-- AMD: AMDGPU code object/HSACO
```

The `.mojoc` file is a precompiled Mojo package. It makes the registered Mojo
definitions available to the MAX compiler, but it is not necessarily the final
CPU or GPU machine-code image. Final specialization and target code generation
happen when MAX compiles the graph for the selected device.

MAX does not necessarily generate both CPU and GPU versions on every compile.
It compiles the specialization required by the graph placement and available
target configuration.

Relevant source paths:

- `murali_docs/artifacts/source_code_walkthrough/max_kernel_lib/kernels/add_one.mojo:24`
- `max/mojo/max/algorithm/functional.mojo:362`
- `max/mojo/max/algorithm/backend/cpu/elementwise.mojo:33`
- `max/mojo/max/algorithm/backend/gpu/elementwise.mojo:830`
- `max/mojo/max/gpu/host/_device_context_extras.mojo:123`
- `max/mojo/max/gpu/host/compile.mojo:36`
- `Mojo/stdlib/std/compile/compile.mojo:210`

</details>

<details>
<summary><strong>Step 6.5: End-To-End Call Flow From <code>_compile_module()</code> To GPU Code</strong></summary>

**End-To-End Call Flow**

Starting from the Python code:

```python
def _compile_module(
    self,
    module: Module,
    custom_extensions_final: list[CustomExtensionType],
    tile_based_fusion: bool = False,
) -> _AsyncValue[_CompiledModels]:
    _COMPILATION_LOCK.acquire()

    try:
        handle = self._impl.compile(
            module.mlir_module._CAPIPtr,
            custom_extensions_final,
            _derive_pipeline_name(module),
            tile_based_fusion,
        )
```

the complete GPU path is:

```text
Python InferenceSession._compile_module()
|
+-- module.mlir_module._CAPIPtr
|   +-- Native pointer to builtin.module
|   +-- Contains mo.graph @add_one_demo
|   +-- Contains mo.custom "demo_add_one"
|
+-- custom_extensions_final
|   +-- Contains the compiled custom Mojo package/library path
|
v
self._impl.compile(...)
|
+-- Crosses from Python into compiled max._core
+-- Runs the native MAX graph compiler
+-- Resolves mo.custom "demo_add_one" through KernelLibrary
+-- Selects/specializes the kernel for the operation's device
v
DemoAddOne.execute[target="gpu"]
|
+-- target is a compile-time StaticString
+-- add_one is captured as the elementwise body
v
foreach[target="gpu"](add_one, output, ctx)
|
v
elementwise[target="gpu"]
|
v
_elementwise_impl(...)
|
+-- CPU target -> _elementwise_impl_cpu
+-- GPU target -> _elementwise_impl_gpu
v
_elementwise_impl_gpu
|
+-- Implemented in:
|   max/mojo/max/algorithm/backend/gpu/elementwise.mojo:830
|
+-- Reads compile-time GPU information
|   +-- warp/wavefront size
|   +-- register count
|   +-- SM/CU count
|   +-- threads per multiprocessor
|
+-- Chooses launch strategy
|   +-- normal grid-stride kernel
|   +-- SM100 CLC work-stealing kernel when enabled/suitable
|
+-- Chooses block size and number of blocks
v
ctx.enqueue_function(...)
|
+-- Visible at elementwise.mojo:378, :563, and :814
+-- Captures the generated GPU kernel closure
+-- Supplies grid dimensions, block dimensions, arguments, and shared memory
v
DeviceFunction.__init__
|
+-- Calls _compile_code[func, ...]
v
max.gpu.host.compile._compile_code
|
+-- Implemented in max/mojo/max/gpu/host/compile.mojo:36
+-- Selects CompilationTarget.current_accelerator()
+-- Calls compile_info[...] with that target
v
std.compile.compile_info
|
+-- Materializes kgen.compile_offload
+-- Records the requested target, emission kind, and compiler options
v
Mojo compiler parametric elaboration
|
+-- Finds kgen.compile_offload operations
+-- Groups them by TargetInfoAttr
+-- Compiles each offload group for its requested target
+-- Replaces the compile_offload result with generated code bytes
v
ObjectCompiler::emitOffloadKernels
|
+-- Splits offload kernels as required
+-- Lowers the GPU MLIR module to LLVM IR
+-- Creates the target-specific LLVM backend from the target triple
v
LLVM TargetRegistry::lookupTarget(targetTriple)
|
+-- NVIDIA target triple/features -> NVPTX backend
+-- AMD target triple/features    -> AMDGPU backend
v
TargetMachine::createTargetMachine(...)
|
+-- Uses target CPU/GPU architecture and feature strings
+-- Runs target optimization and assembly/object emission
v
+-----------------------------+--------------------------------+
| NVIDIA                      | AMD                            |
+-----------------------------+--------------------------------+
| NVPTX LLVM backend          | AMDGPU LLVM backend            |
| emits PTX/object image      | emits AMD GPU code object      |
| driver may JIT PTX          | HSACO/code object is loadable  |
| module loaded by CUDA       | module loaded by HIP           |
+-----------------------------+--------------------------------+
|
v
Generated device-code bytes are returned to the Mojo/MAX host artifact
|
v
AsyncRT_DeviceContext_loadFunction(...)
|
+-- Loads the module image through the active accelerator runtime
+-- Looks up the generated kernel function
+-- Returns an opaque DeviceFunction handle
v
AsyncRT_DeviceContext_enqueueFunctionDirect(...)
|
+-- Passes function handle
+-- Passes grid and block dimensions
+-- Passes kernel argument pointers
+-- Enqueues work on the device stream
v
GPU executes demo_add_one
```

`_elementwise_impl_gpu` is implemented in
`max/mojo/max/algorithm/backend/gpu/elementwise.mojo:830`. It is not inside the
closed-source `max._core` Python extension. The native graph compiler reaches
this Mojo implementation after it resolves `demo_add_one` and specializes its
`execute[target]` function for a GPU target.

The backend is not selected by a Python `if nvidia` statement. The selected
accelerator contributes a `CompilationTarget` containing a target triple,
architecture, features, and compile options. The Mojo offload compiler carries
that target into `ObjectCompiler`; LLVM then selects the registered backend by
the target triple:

```cpp
const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
    llvm::Triple(options.targetTriple), errorMessage);

target->createTargetMachine(
    llvm::Triple(options.targetTriple),
    options.targetCpu,
    options.targetFeatures,
    ...
);
```

Conceptually:

```text
NVIDIA GPU selected
    -> NVIDIA target triple + SM architecture/features
    -> LLVM NVPTX target
    -> PTX and/or CUDA-loadable module image

AMD GPU selected
    -> AMDGPU target triple + GFX architecture/features
    -> LLVM AMDGPU target
    -> AMD GPU code object/HSACO
```

The generated GPU host launcher conceptually performs:

```text
load GPU module image
    |
get kernel function handle
    |
set/pass kernel arguments
    |
launch grid x block on a stream
    |
return while device work remains asynchronous
```

There is no explicit application-level copy into GPU instruction RAM. MAX hands
the generated module image to the CUDA or HIP driver. The driver decides where
executable code resides and how instruction caches are populated.

Source boundary: the Python-to-native call and Mojo/LLVM offload machinery are
visible in this checkout. The internal implementation of
`self._impl.compile(...)` itself is supplied by the compiled `max._core` engine,
so its exact internal C++ call sequence cannot be copied from repository source.

</details>

<details>
<summary><strong>Step 6.6: NVIDIA And AMD Driver APIs Used To Load And Launch GPU Code</strong></summary>

For this MAX build, `libAsyncRTMojoBindings.so` dynamically loads:

```text
NVIDIA
    libcuda.so.1

AMD
    libamdhip64.so
```

It resolves the driver functions with `dlopen()` / `dlsym()`.

The NVIDIA APIs present in the library are:

```text
cuModuleLoadDataEx
cuModuleGetFunction
cuLaunchKernel
cuLaunchKernelEx
cuModuleUnload
```

The call flow is:

```text
Mojo DeviceFunction construction
    |
    v
external_call["AsyncRT_DeviceContext_loadFunction"](...)
    |
    v
AsyncRT CUDADeviceContext::loadFunction(...)
    |
    +-- cuModuleLoadDataEx(...)
    |      +-- Input: PTX, CUBIN, or another CUDA module image in host memory
    |      +-- Output: CUmodule
    |
    +-- cuModuleGetFunction(...)
           +-- Input: CUmodule + generated kernel symbol name
           +-- Output: CUfunction

Mojo ctx.enqueue_function(...)
    |
    v
external_call["AsyncRT_DeviceContext_enqueueFunctionDirect"](...)
    |
    v
cuLaunchKernel(...) or cuLaunchKernelEx(...)
    |
    +-- CUfunction
    +-- grid dimensions
    +-- block dimensions
    +-- dynamic shared-memory size
    +-- CUDA stream
    +-- kernel argument pointers
```

`cuModuleLoadDataEx()` is the API that gives the generated code image to the
NVIDIA driver. If the image contains PTX, the driver can JIT-compile it for the
installed GPU. If it contains compatible binary code, the driver can load that
code directly. `cuModuleGetFunction()` obtains a launchable function handle, and
`cuLaunchKernel()` or `cuLaunchKernelEx()` enqueues it.

The AMD APIs present in the library are:

```text
hipModuleLoadData
hipModuleLoadDataEx
hipModuleGetFunction
hipModuleLaunchKernel
hipModuleUnload
```

The AMD call flow is:

```text
Mojo DeviceFunction construction
    |
    v
external_call["AsyncRT_DeviceContext_loadFunction"](...)
    |
    v
AsyncRT HIPDeviceContext::loadFunction(...)
    |
    +-- hipModuleLoadData(...) or hipModuleLoadDataEx(...)
    |      +-- Input: AMD GPU code object/HSACO image in host memory
    |      +-- Output: hipModule_t
    |
    +-- hipModuleGetFunction(...)
           +-- Input: hipModule_t + generated kernel symbol name
           +-- Output: hipFunction_t

Mojo ctx.enqueue_function(...)
    |
    v
external_call["AsyncRT_DeviceContext_enqueueFunctionDirect"](...)
    |
    v
hipModuleLaunchKernel(...)
    |
    +-- hipFunction_t
    +-- grid dimensions
    +-- block dimensions
    +-- dynamic shared-memory size
    +-- HIP stream
    +-- kernel argument pointers
```

There is no CUDA or HIP API named "copy this kernel into instruction RAM."
The module-load API is the software-visible boundary:

```text
MAX/AsyncRT owns
    +-- Generated code bytes in host memory
    +-- Module and function handles
    +-- Launch dimensions and arguments

CUDA/HIP driver owns
    +-- Validation/JIT/linking required for the installed GPU
    +-- Device virtual-address mappings for executable code
    +-- Uploading or paging code to device-accessible memory
    +-- Instruction-cache filling and eviction
    +-- Scheduling the launched kernel
```

Therefore, "load compiled binary to device IRAM" should be read as:

```text
NVIDIA:
    cuModuleLoadDataEx -> cuModuleGetFunction -> cuLaunchKernel[Ex]

AMD:
    hipModuleLoadData[Ex] -> hipModuleGetFunction -> hipModuleLaunchKernel
```

The GPU fetches instructions when the kernel is scheduled. Hardware and the
vendor driver manage the physical code placement and instruction caches; MAX
does not manually address or fill an IRAM region.

</details>

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

<details>
<summary><strong>QnA 7.1: Deep Dive Into <code>session.init(compiled)</code> Through The GPU Driver Layer</strong></summary>

**Purpose**

`session.init(compiled)` converts the reusable compiled artifact into a runtime `Model` instance bound to the session’s devices, weights, and persistent state.

```text
compile() -> machine code + metadata
init()    -> runtime model + weights + persistent allocations
execute() -> per-call input/output processing and kernel launches
```

**Python Call Flow**

Source: [api.py](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/engine/api.py:1112)

```text
session.init(compiled)
|
+-- Calls init_all(compiled)
|
+-- Convert None weights to {}
|
+-- Validate every required weight exists
|
+-- Validate weight device
|   +-- CPU weight expected -> DLPack CPU tensor required
|   +-- GPU weight expected -> DLPack GPU tensor required
|
+-- Verify weights are contiguous
|
+-- Ensure compiled artifact has a session-owned AsyncValue
|   |
|   +-- _wrap_compiled(...)
|
+-- Enter native MAX:
|   |
|   +-- self._impl._load_all(
|           compiled._compiled,
|           weights_registry_real,
|       )
|
+-- Return one max._core.engine.Model
```

The key native boundary is:

```python
models = self._impl._load_all(
    compiled._compiled,
    weights_registry_real,
)
```

Its interface is declared in [engine.pyi](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/_core/engine.pyi:545).

**Native Initialization**

Binary symbols show this call flow:

```text
max._core InferenceSession._load_all
|
+-- Convert Python/DLPack weights into native WeightsRegistry
|
+-- Select each Model inside CompiledModels
|
+-- M::Engine::Context::initModel(
|       M::Model const*,
|       M::RCRef<M::WeightsRegistry>
|   )
|
+-- Schedule model setup through AsyncRT
|
+-- Produce:
|   AsyncValueRef<unique_ptr<ExecutableModel>>
|
+-- Wrap it in Python max._core.engine.Model
```

The native setup is asynchronous. This is documented by [model.h](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/include/max/c/model.h:125):

```text
M_initModel(...)
    -> returns M_AsyncModel
    -> model setup may still be running
```

`model.execute()` can be called immediately because execution is chained after that asynchronous setup dependency.

Initialization errors can therefore surface at the next synchronization point, commonly `model.execute()`.

**What Setup Does**

Depending on the graph, the generated model setup/init entry point can:

```text
Model setup
|
+-- Bind external weights by name
|
+-- Retain DLPack ownership/lifetime references
|
+-- Allocate persistent buffers
|   +-- model weights
|   +-- mutable graph state
|   +-- buffers created by ops.buffer_create(...)
|
+-- Copy weights to their required device
|
+-- Repack or transform weights
|   +-- quantized weight repacking
|   +-- layout conversion
|   +-- sharding
|
+-- Initialize persistent values
|
+-- Run setup-time CPU/GPU kernels when required
|
+-- Construct ExecutableModel
    +-- execution entry point
    +-- input/output metadata
    +-- device contexts
    +-- persistent values
    +-- setup dependency chain
```

It does not provide normal graph inputs or execute the forward graph.

**GPU Driver Path**

When model setup requires GPU memory or setup kernels:

```text
Executable model setup
|
v
Generated Mojo/MAX host code
|
v
AsyncRT device operations
|
+-- Allocate GPU memory
|   NVIDIA: cuMemAllocAsync / cuMemAlloc_v2
|   AMD:    hipMallocAsync / hipMalloc
|
+-- Copy weights host -> GPU
|   NVIDIA: cuMemcpyHtoDAsync_v2
|   AMD:    hipMemcpyHtoDAsync
|
+-- Load a setup kernel, if needed
|   NVIDIA:
|     cuModuleLoadDataEx
|     cuModuleGetFunction
|
|   AMD:
|     hipModuleLoadData[Ex]
|     hipModuleGetFunction
|
+-- Launch the setup kernel
    NVIDIA: cuLaunchKernel / cuLaunchKernelEx
    AMD:    hipModuleLaunchKernel
```

The Mojo-side bridge for loading a device function is visible in [_device_context_extras.mojo](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/mojo/max/gpu/host/_device_context_extras.mojo:157).

**Driver Layers**

```text
MAX / Mojo generated host code
|
v
libAsyncRTMojoBindings.so
|
+-- NVIDIA -> libcuda.so.1
|              |
|              v
|          CUDA user-mode driver
|              |
|              v
|          ioctl/system calls
|              |
|              v
|          NVIDIA kernel drivers
|          /dev/nvidiactl
|          /dev/nvidia0
|          /dev/nvidia-uvm
|
+-- AMD -> libamdhip64.so
               |
               v
           HIP/ROCr user-space runtime
               |
               v
           ioctl/system calls
               |
               v
           amdgpu/KFD kernel drivers
           /dev/kfd
           /dev/dri/renderD*
```

The kernel driver manages GPU virtual-memory mappings, DMA copies, command queues, scheduling, and communication with GPU firmware.

**Your Add-One Example**

Your graph has no weights or persistent buffers:

```python
model = session.init(compiled)
```

Therefore initialization is approximately:

```text
Resolve compiled add_one_demo artifact
    |
Create asynchronous ExecutableModel
    |
Attach existing CPU/GPU device context
    |
Store input/output metadata
    |
Return Model
```

It normally does not launch `demo_add_one` during `init()`.

The important GPU operations occur later:

```text
Buffer.from_numpy(input_values).to(device)
    -> GPU allocation
    -> host-to-device copy

model.execute(input_tensor)
    -> load demo_add_one GPU module/function if needed
    -> cuLaunchKernel[Ex] or hipModuleLaunchKernel
    -> execute add-one on GPU
```

So `init()` creates and optionally initializes the runtime model; `execute()` launches your actual add-one forward kernel.

</details>

<details>
<summary><strong>QnA 7.2: Deep Dive Into <code>model.execute(input_tensor)[0]</code> Through The GPU Driver Layer</strong></summary>

`model.execute(input_tensor)[0]` has two separate operations:

```python
outputs = model.execute(input_tensor)  # Execute; returns list[Buffer]
result = outputs[0]                    # Select first graph output
```

`[0]` does not copy or execute anything. It only indexes the returned Python list.

**Complete Call Flow**
```text
Python
|
+-- model.execute(input_tensor)
|   +-- _Model_execute(...)
|       +-- Verify input is contiguous
|       +-- Detect that input_tensor is already a MAX Buffer
|       +-- Keep the existing Buffer; no Python-side copy
|       +-- self._execute_device_tensors([input_tensor])
|
+-- max._core native Python binding
|   +-- GraphCompiler::Driver::prepareInput(...)
|   |   +-- Obtain DeviceBuffer handle
|   |   +-- Validate dtype, shape and device
|   |   +-- Prepare the native input value
|   |
|   +-- GraphCompiler::Driver::executeFromDriver(...)
|       +-- ExecutableModel::execute(...)
|
+-- Compiled add_one_demo graph
|   +-- Bind input GPU buffer to %input0
|   +-- Allocate/reuse output GPU buffer
|   +-- Invoke mo.custom "demo_add_one"
|
+-- Compiled Mojo custom operation
|   +-- DemoAddOne.execute[target="gpu"](...)
|       +-- foreach[target="gpu"](...)
|           +-- _elementwise_impl_gpu(...)
|               +-- Select block size
|               +-- Calculate grid size
|               +-- Create specialized GPU kernel
|               +-- ctx.enqueue_function(...)
|
+-- MAX DeviceContext / AsyncRT
|   +-- Load function, normally on first use or cache miss
|   |   +-- AsyncRT_DeviceContext_loadFunction(...)
|   +-- Enqueue kernel
|       +-- AsyncRT_DeviceContext_enqueueFunctionDirect(...)
|
+-- Vendor user-mode GPU driver
|   +-- NVIDIA
|   |   +-- libcuda.so.1
|   |   +-- cuModuleLoadDataEx(...)
|   |   +-- cuModuleGetFunction(...)
|   |   +-- cuLaunchKernel(...) / cuLaunchKernelEx(...)
|   |
|   +-- AMD
|       +-- libamdhip64.so
|       +-- hipModuleLoadDataEx(...)
|       +-- hipModuleGetFunction(...)
|       +-- hipModuleLaunchKernel(...)
|
+-- Linux kernel GPU driver
|   +-- NVIDIA: /dev/nvidiactl, /dev/nvidia0, /dev/nvidia-uvm
|   +-- AMD: /dev/kfd, /dev/dri/renderD*
|
+-- GPU hardware
|   +-- Command processor receives launch
|   +-- Scheduler dispatches thread blocks
|   +-- Threads load input values from GPU global memory
|   +-- Execute x + 1
|   +-- Store values into output GPU global memory
|
+-- MAX returns list[Buffer]
    +-- [0] selects the first output Buffer
```

### 1. Python Wrapper

Implemented in [api.py](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/python/max/engine/api.py:120):

```python
if isinstance(arg, Buffer):
    buffer = arg

input_impls.append(buffer)
return self._execute_device_tensors(input_impls)
```

Because your input was created as:

```python
input_tensor = Buffer.from_numpy(input_values).to(device)
```

it is already a `Buffer` on `device`. Python passes the native buffer handle, not the numerical contents.

```text
Python input_tensor variable
    |
    +-- shape, dtype, device and ownership metadata
    +-- native DeviceBuffer handle
            |
            +-- points to values in GPU global memory
```

There should be no second host-to-device copy when the buffer is already on the model's expected GPU.

### 2. Native MAX Execution

`_execute_device_tensors()` enters the compiled `max._core` extension. The native binaries expose:

```text
GraphCompiler::Driver::prepareInput(...)
GraphCompiler::Driver::executeFromDriver(...)
ExecutableModel::execute(...)
GraphCompiler::Driver::extractDriverResults(...)
```

Conceptually:

```cpp
prepared_input = prepareInput(input_tensor.device_buffer);
executable_model.execute(prepared_input);
outputs = extractDriverResults(...);
```

The compiled executable already contains the host-side execution plan for `add_one_demo`.

### 3. Custom Mojo Operation

Your implementation is [add_one.mojo](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/murali_docs/artifacts/source_code_walkthrough/max_kernel_lib/kernels/add_one.mojo:21):

```mojo
@extensibility.register("demo_add_one")
struct DemoAddOne:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor,
        x: InputTensor[...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def add_one[width: Int](idx: Coord) -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[target=target](add_one, output, ctx)
```

At runtime, `x` and `output` contain GPU buffer addresses and tensor-layout information. `ctx` represents the GPU execution context and default queue/stream.

### 4. GPU Launch

`foreach` dispatches to `_elementwise_impl_gpu` in [elementwise.mojo](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/mojo/max/algorithm/backend/gpu/elementwise.mojo:830).

The launch eventually calls:

```mojo
ctx.enqueue_function(
    kernel,
    grid_dim=num_blocks,
    block_dim=block_size,
)
```

That reaches [device_context.mojo](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/max/mojo/max/gpu/host/device_context.mojo:4209):

```mojo
external_call[
    "AsyncRT_DeviceContext_enqueueFunctionDirect", _CString[]
](...)
```

The arguments include:

```text
GPU function handle
input/output pointer arguments
grid dimensions
block dimensions
dynamic shared-memory size
launch attributes
device stream/queue
```

### 5. User-Mode And Kernel Drivers

For NVIDIA, AsyncRT dynamically uses `libcuda.so.1`:

```text
cuModuleLoadDataEx    Load CUBIN/PTX into the CUDA context
cuModuleGetFunction  Obtain the kernel handle
cuLaunchKernelEx     Submit the GPU launch
```

For AMD, it uses `libamdhip64.so`:

```text
hipModuleLoadDataEx
hipModuleGetFunction
hipModuleLaunchKernel
```

The user-mode driver communicates with the Linux kernel driver through private driver interfaces, typically `ioctl`, mapped command queues and doorbells.

The kernel driver manages:

```text
GPU virtual-memory mappings
command queues
DMA operations
process isolation
scheduling
firmware communication
completion events
```

The GPU command processor then dispatches blocks to SMs/CUs. Hardware instruction caches are populated by the GPU automatically.

### 6. Returning The Result

The low-level GPU launch is asynchronous. MAX/AsyncRT records the dependency between the output buffer and the GPU work.

```python
result = model.execute(input_tensor)[0]
```

`result` is still a device-resident `Buffer`. No GPU-to-CPU copy is caused by `[0]`.

The copy and required synchronization happen here:

```python
result_values = result.to(CPU()).to_numpy()
```

```text
GPU output buffer
    |
    +-- wait for add-one kernel dependency
    +-- device-to-host transfer
    +-- expose completed CPU memory as NumPy
```

</details>

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
