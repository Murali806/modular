<details>
<summary><strong>Overview Through Test Phase: expand or collapse the complete runtime sequence</strong></summary>

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

```text
Graph(
  name="add_one_demo",
  forward=<symbolic computation>,
  input_types=[float32[2,4] on device_ref],
  custom_extensions=[kernels/],
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
