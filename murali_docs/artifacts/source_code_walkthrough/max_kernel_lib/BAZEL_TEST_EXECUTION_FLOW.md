`bazel test` automatically performs two major operations:

```text
1. Build the test and all dependencies
2. Run the resulting test executable
```

**Command Start**

```sh
cd "${repo_root}"
```

Moves to the repository root because `./bazelw` and Bazel labels are resolved from there.

```sh
exec ./bazelw ...
```

`exec` replaces the current shell process with `bazelw`. `bazelw` then launches the repository’s expected Bazel version.

**Build Phase**

Bazel loads:

[BUILD.bazel](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/murali_docs/artifacts/source_code_walkthrough/max_kernel_lib/BUILD.bazel:1)

The requested target is:

```text
//murali_docs/artifacts/source_code_walkthrough/max_kernel_lib:add_one_demo_test
```

Bazel discovers this dependency chain:

```text
:add_one_demo_test
    depends on :add_one_demo
        depends on add_one_demo.py
        depends on kernels/add_one.mojo as runtime data
        depends on MAX Python packages
        depends on NumPy
        depends on Mojo packages and runtime tools
```

Specifically:

```python
modular_run_binary_test(
    name = "add_one_demo_test",
    binary = "add_one_demo",
)
```

means `:add_one_demo_test` cannot be built until `:add_one_demo` is available.

Then:

```python
modular_py_binary(
    name = "add_one_demo",
    srcs = ["add_one_demo.py"],
    ...
)
```

causes Bazel to:

1. Prepare a Python executable launcher.
2. Locate the declared Python dependencies.
3. Prepare the Mojo compiler/runtime environment.
4. Place `kernels/add_one.mojo` into the runfiles directory.
5. Create the test wrapper and runfiles links.
6. Reuse anything already available in the disk cache.

Python itself is not compiled into native machine code. Bazel prepares an executable launcher, Python sources, libraries, tools, environment variables, and runtime files.

Also, the custom `add_one.mojo` kernel is not compiled through the separate `:kernels` target here. `Graph(...)` loads the source directory through `custom_extensions`, builds it into a precompiled Mojo package, and registers the custom operation before the full graph is compiled.

If any required build action fails, Bazel stops and does not run the test.

<details>
<summary><strong>Test Phase</strong></summary>

After the build succeeds, Bazel starts the `:add_one_demo_test` test process.

The test wrapper launches:

```text
:add_one_demo
    -> Python launcher
        -> add_one_demo.py
```

Python reaches:

```python
if __name__ == "__main__":
    main()
```

Execution inside `main()` is:

1. Find the Mojo kernel directory:

```python
mojo_kernels = Path(__file__).parent / "kernels"
```

2. Create the hardcoded NumPy input.

3. Select CPU or accelerator.

4. Construct the MAX graph containing:

```python
ops.custom(name="demo_add_one", ...)
```

5. Supply the custom Mojo sources:

```python
custom_extensions=[mojo_kernels]
```

6. Compile the graph:

```python
compiled = session.compile(graph)
```

At this point MAX compiles the complete graph. The custom Mojo source package was already loaded and registered while `Graph(...)` was being constructed through `custom_extensions=[mojo_kernels]`.

7. Initialize and execute the compiled graph:

```python
model = session.init(compiled)
result = model.execute(input_tensor)[0]
```

8. The Mojo kernel adds `1` to every element.

9. NumPy verifies the result:

```python
np.testing.assert_allclose(result_values, expected)
```

10. The program prints the input, result, and expected values.

</details>

**Pass Or Fail**

If `main()` completes normally:

```text
Python exit code = 0
Bazel result = PASSED
```

If compilation, execution, or assertion fails:

```text
Python exit code != 0
Bazel result = FAILED
```

`--test_output=all` tells Bazel to display all printed output.

**Flag Roles**

```text
--output_user_root=...
```

Stores Bazel’s working state and generated output inside the selected directory.

```text
--disk_cache=...
```

Caches build results so unchanged work can be reused.

```text
--repo_contents_cache=
```

Disables Bazel’s repository-content cache.

```text
--config=prebuilt-mojo
```

Uses the repository configuration for prebuilt Mojo tooling.

```text
--config=disable-lint
```

Disables lint-related build configuration.

```text
--aspects=
--output_groups=default
```

Clears configured aspects and requests normal build outputs.

```text
--enable_runfiles=true
--build_runfile_links=true
```

Creates the runtime file tree needed by the Python executable.

```text
--remote_download_outputs=all
```

Downloads all remote build outputs when remote caching/execution is configured.

The condensed flow is:

```text
bazel test
  -> load and analyze BUILD.bazel
  -> build :add_one_demo and its dependencies
  -> create :add_one_demo_test wrapper
  -> run add_one_demo.py
  -> construct the graph and load add_one.mojo
  -> compile the complete MAX graph
  -> execute kernel
  -> verify result
  -> report PASSED or FAILED
```

## Step 4: Construct The MAX Graph

**Evidence:** `SOURCE`

### 1. `Graph`

```python
graph = Graph(...)
```

`Graph` is a **Python class**, defined in [graph.py](../../../../max/python/max/graph/graph.py#L682).

It is not a Python struct or Mojo struct. It constructs an internal MLIR/MAX computation graph.

The Mojo struct in this example is `DemoAddOne` in [add_one.mojo](kernels/add_one.mojo#L22).

### 2. `forward` And `lambda`

```python
forward=lambda x: (...)
```

`forward` tells MAX how graph inputs are transformed into graph outputs.

`lambda` is an anonymous Python function. This:

```python
forward=lambda x: ops.custom(...)[0].tensor
```

is equivalent to:

```python
def forward(x):
    outputs = ops.custom(...)
    return outputs[0].tensor
```

The function is called while constructing the graph. It describes operations; it does not process the NumPy input immediately.

### 3. Where Does `x` Come From?

It comes from:

```python
input_types=[
    TensorType(...)
]
```

MAX creates one symbolic graph input for each `TensorType`. Internally, the constructor effectively calls:

```python
result = forward(*self.inputs)
```

Because there is one input type:

```text
self.inputs = [symbolic_input]
                     |
                     v
               lambda x
```

Therefore, `x` is a symbolic `TensorValue` describing:

```text
dtype  = float32
shape  = [2, 4]
device = cpu:0 or gpu:0
```

It is not yet the actual NumPy array.

Later:

```python
model.execute(input_tensor)
```

binds the real `input_tensor` buffer to symbolic graph input `x`.

### 4. `values=[x]`

```python
values=[x]
```

Passes symbolic input `x` as the first argument to the custom operation.

At runtime, that corresponds to the Mojo kernel argument:

```mojo
x: InputTensor[...]
```

The flow is:

```text
input_tensor
    |
    | model.execute(input_tensor)
    v
symbolic graph input x
    |
    | values=[x]
    v
DemoAddOne.execute(..., x, ...)
```

### 5. `out_types`

```python
out_types=[
    TensorType(
        dtype=x.dtype,
        shape=x.tensor.shape,
        device=device_ref,
    )
]
```

This declares the expected custom-operation output:

```text
same dtype as x
same shape as x
located on device_ref
```

It does not allocate or calculate the output at this point. It gives the graph compiler the output type information it needs.

### 6. `ops.custom(...)[0].tensor`

[`ops.custom()`](../../../../max/python/max/graph/ops/custom.py#L58) returns a list of symbolic output values:

```python
list[Value]
```

There is one `out_types` entry, so there is one output:

```text
ops.custom(...)           -> [output_value]
ops.custom(...)[0]        -> output_value
ops.custom(...)[0].tensor -> output as TensorValue
```

`.tensor` verifies that the selected output is a tensor and returns its `TensorValue` representation. It does not extract concrete tensor data.

### 7. `device=device_ref`

`device_ref` is a symbolic device identifier such as:

```text
cpu:0
```

or:

```text
gpu:0
```

In the input `TensorType`, it declares where the graph expects that input tensor to reside.

In `ops.custom`, it assigns the custom operation to that device. It also becomes the Mojo kernel's compile-time `target` parameter.

It does not move data by itself. This line performs the actual transfer:

```python
input_tensor = Buffer.from_numpy(input_values).to(device)
```

### 8. `custom_extensions`

```python
custom_extensions=[mojo_kernels]
```

`mojo_kernels` points to:

```text
max_kernel_lib/kernels/
```

That directory contains `add_one.mojo`, which registers:

```mojo
@extensibility.register("demo_add_one")
```

The registration name matches:

```python
ops.custom(name="demo_add_one", ...)
```

During `Graph(...)` construction, MAX:

```text
loads the kernels/ directory
    -> compiles the Mojo source package
    -> discovers registered "demo_add_one"
    -> verifies ops.custom(name="demo_add_one")
```

Then `session.compile(graph)` compiles the complete MAX graph using that custom kernel.

Without `custom_extensions=[mojo_kernels]`, MAX would not know where the implementation of `"demo_add_one"` is located.
