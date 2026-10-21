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
    device_ref = DeviceRef.from_device(device)

    graph = Graph(
        "add_one_demo",
        forward=lambda x: (
            ops.custom(
                name="demo_add_one",
                device=device_ref,
                values=[x],
                out_types=[
                    TensorType(
                        dtype=x.dtype,
                        shape=x.tensor.shape,
                        device=device_ref,
                    )
                ],
            )[0].tensor
        ),
        input_types=[
            TensorType(
                dtype,
                shape=[rows, columns],
                device=device_ref,
            ),
        ],
        custom_extensions=[mojo_kernels],
    )

    session = InferenceSession(devices=[device])
    compiled = session.compile(graph)
    model = session.init(compiled)

    input_tensor = Buffer.from_numpy(input_values).to(device)
    result = model.execute(input_tensor)[0]
    assert isinstance(result, Buffer)

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
