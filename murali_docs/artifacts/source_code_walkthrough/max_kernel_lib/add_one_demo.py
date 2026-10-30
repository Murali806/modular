from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def main() -> None:
    mojo_kernels = Path(__file__).parent / "kernels"

    input_values = np.array(
        [
            [0.0, 1.5, 2.0, 3.5],
            [4.0, 5.5, 6.0, 7.5],
        ],
        dtype=np.float32,
    )
    rows, columns = input_values.shape
    dtype = DType.float32

    device = CPU() if accelerator_count() == 0 else Accelerator()
    # DeviceRef records symbolic CPU/GPU placement; it does not move data.
    device_ref = DeviceRef.from_device(device)

    # Graph is a Python class implemented by MAX in
    # max/python/max/graph/graph.py.
    # So Graph is a Python frontend class specifically implemented by the
    # MAX framework to construct its internal MLIR-based computation graph.
    # Graph records a symbolic MAX computation.
    # Internally, it creates an MLIR module and a MAX mo.graph operation with
    # symbolic inputs, operations, and outputs.

    graph = Graph(
        "add_one_demo",
        forward=lambda x: (
            # x is the symbolic input created from input_types below.
            ops.custom(
                name="demo_add_one",
                # Execute the custom operation on the selected CPU/GPU.
                device=device_ref,  # Expected tensor location.
                values=[x],
                out_types=[
                    # The output resides on device_ref; no input copy occurs here.
                    TensorType(
                        dtype=x.dtype,
                        shape=x.tensor.shape,
                        device=device_ref,  # Expected tensor location.
                    )
                ],
            )[0].tensor  # Return the first symbolic output as a TensorValue.
        ),
        # model.execute() must receive its input buffer on device_ref.
        input_types=[
            TensorType(
                dtype,
                shape=[rows, columns],  # Expected input shape.
                device=device_ref,
            ),
        ],
        # Load and register demo_add_one from the Mojo source directory.
        custom_extensions=[mojo_kernels],

        #      |
        #      v
        # custom_extensions=[mojo_kernels], tells the MAX graph:
        #
        #  > “This directory contains Mojo implementations of custom operations used by this graph.”
        #
        #  Here, mojo_kernels points to:
        #
        #  murali_docs/artifacts/source_code_walkthrough/max_kernel_lib/kernels
        #
        #  ### Processing Flow
        #
        #  custom_extensions=[kernels/]
        #              |
        #              v
        #  MAX finds kernels/__init__.mojo
        #              |
        #              v
        #  Recognizes kernels/ as a Mojo package
        #              |
        #              v
        #  Runs approximately:
        #  mojo precompile kernels/ -o <temporary-file>.mojoc
        #              |
        #              v
        #  Finds registered operation:
        #  @extensibility.register("demo_add_one")
        #              |
        #              v
        #  Matches it with:
        #  ops.custom(name="demo_add_one", ...)
        #              |
        #              v
        #  Makes the implementation available to session.compile(graph)
        #
        #  The registration is in murali_docs/artifacts/source_code_walkthrough/max_kernel_lib/kernels/add_one.mojo:21:
        #
        #  @extensibility.register("demo_add_one")
        #  struct DemoAddOne:
        #      @staticmethod
        #      def execute(...):
        #          ...
        #
        #  The Python graph refers to that registered name:
        #
        #  ops.custom(
        #      name="demo_add_one",
        #      ...
        #  )
        #
        #  The names must match exactly:
        #
        #  Python graph                         Mojo package
        #
        #  ops.custom(                         @extensibility.register(
        #      name="demo_add_one"                 "demo_add_one"
        #  )                                   )
        #
        #  custom_extensions does not:
        #
        #  - Execute the kernel immediately.
        #  - Copy tensors to the GPU.
        #  - Pass the kernels/ directory to the running GPU kernel.
        #
        #  It supplies the Mojo package to MAX while constructing and compiling the graph. Later:
        #
        #  model.execute(input_tensor)
        #
        #  executes the already-compiled demo_add_one implementation.
        #
        #  Without custom_extensions=[mojo_kernels], MAX sees the symbolic request:
        #
        #  ops.custom(name="demo_add_one", ...)
        #
        #  but cannot find its Mojo implementation, so graph verification or compilation fails.
        #

    )

    # InferenceSession is imported from max.engine. It configures the selected
    # CPU or GPU and manages the environment in which graphs are compiled and
    # initialized. Even when only an accelerator is supplied, MAX also includes
    # the host CPU.
    session = InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)

    # .to(device) performs the host-to-GPU copy when a GPU is selected.
    # On an accelerator, .to(device) is equivalent to .to(Accelerator()).
    input_tensor = Buffer.from_numpy(input_values).to(device)

    #  input_tensor     (<--- on Host/Python)
    #      |
    #      v
    #  Buffer handle
    #  - device = Accelerator(0)
    #  - shape = [2, 4]
    #  - dtype = float32
    #  - native allocation handle
    #      |
    #      v
    #  GPU global memory
    #  [actual tensor values]
    #
    #  After:
    #
    #  input_tensor = Buffer.from_numpy(input_values).to(device)
    #
    #  - The Python input_tensor variable stays in CPU/host code.
    #  - Its metadata and native handle are host-managed.
    #  - The actual numerical data resides in GPU global memory when device is an Accelerator.
    #  - model.execute(input_tensor) passes that native device-buffer handle to MAX.
    #  - MAX extracts the GPU address and supplies it to the Mojo kernel.
    #  - The Mojo kernel reads and writes the GPU allocation, not the Python variable.
    #
    #  This is the same general model as CUDA:
    #
    #  float* device_ptr;              // Host variable
    #  cudaMalloc(&device_ptr, size);  // Points to GPU memory
    #
    #  The pointer variable is on the host, while the memory it represents is on the GPU. MAX’s Buffer wraps this pointer
    #  plus shape, dtype, device, queue, ownership, and lifetime information.

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Buffer)

    # Copy a GPU result back to CPU before exposing it as a NumPy array.
    result_values = result.to(CPU()).to_numpy()
    expected = input_values + 1.0
    np.testing.assert_allclose(result_values, expected)

    print("Device:")
    print(device)
    print()
    print("Input:")
    print(input_values)
    print()
    print("Graph result:")
    print(result_values)
    print()
    print("Expected:")
    print(expected)


if __name__ == "__main__":
    main()
