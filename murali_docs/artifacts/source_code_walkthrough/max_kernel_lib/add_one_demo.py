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

    session = InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)

    # .to(device) performs the host-to-GPU copy when a GPU is selected.
    input_tensor = Buffer.from_numpy(input_values).to(device)
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
