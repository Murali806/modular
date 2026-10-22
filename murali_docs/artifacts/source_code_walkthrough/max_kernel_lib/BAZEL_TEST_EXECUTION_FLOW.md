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

Also, the custom `add_one.mojo` kernel is not compiled through the separate `:kernels` target here. It is supplied as source and compiled later by MAX during program execution.

If any required build action fails, Bazel stops and does not run the test.

**Test Phase**

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

At this point MAX invokes the Mojo compiler and compiles/registers [`add_one.mojo`](/prj/adsp-platform/mur/Murali/aisw/Qgen/Confluence_Readings/Modular/max_serve/modular/murali_docs/artifacts/source_code_walkthrough/max_kernel_lib/kernels/add_one.mojo:14).

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
  -> compile MAX graph and add_one.mojo at runtime
  -> execute kernel
  -> verify result
  -> report PASSED or FAILED
```
