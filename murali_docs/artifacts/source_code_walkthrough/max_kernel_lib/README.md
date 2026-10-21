# MAX Kernel Library Demo

Evidence: `SOURCE`, `TEST`, `CPU-RUN`

This folder contains the complete source and test for a minimal MAX custom
kernel. The Mojo op adds `1` to every input element. Python loads the op into a
MAX Graph, compiles it with `InferenceSession`, executes it, and checks the
hardcoded result.

```text
add_one_demo.py
      |
      | ops.custom(name="demo_add_one")
      v
kernels/add_one.mojo
      |
      | extensibility.foreach
      v
CPU or available accelerator
```

## Run From This Folder

From the repository root:

```sh
cd murali_docs/artifacts/source_code_walkthrough/max_kernel_lib
./run.sh
```

`run.sh` keeps Bazel output and cache files under
`/local/mnt/workspace/max-kernel-lib-bazel-$USER`, rather than creating large
generated files inside this learning-artifact folder. Override the location
with `MAX_KERNEL_LIB_BAZEL_ROOT` when needed.

## Build Only

While standing in this folder:

```sh
../../../../bazelw \
  --output_user_root=/local/mnt/workspace/max-kernel-lib-bazel-$USER/output-user-root \
  build \
  --disk_cache=/local/mnt/workspace/max-kernel-lib-bazel-$USER/disk-cache \
  --config=prebuilt-mojo \
  //murali_docs/artifacts/source_code_walkthrough/max_kernel_lib:kernels
```
