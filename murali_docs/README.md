# MAX Internals Learning Lab

```text
external Python
      |
      v
OpenAI HTTP/SSE -> MAX Serve -> scheduler -> model graph -> MAX Engine
                                                       |
                          +----------------------------+------------------+
                          |                                               |
                          v                                               v
                    CPU execution                                GPU compilation
                    [CPU-RUN]                                    [COMPILE-ONLY]
                                                                        |
                                                                        v
                                                               GPU measurement
                                                                  [GPU-LAB]
```

```mermaid
flowchart LR
    C[External Python] --> A[OpenAI HTTP + SSE]
    A --> S[MAX Serve]
    S --> Q[Scheduler]
    Q --> G[Model graph]
    G --> E[MAX Engine]
    E --> CPU[CPU execution]
    E --> VC[CUDA/HIP virtual-device compile]
    VC --> GPU[Future GPU runtime lab]
```

## Progress

| Phase | Artifact                                                     | Evidence                            |
|-------|--------------------------------------------------------------|-------------------------------------|
| 0     | [Environment](artifacts/00_environment.md)                   | `CPU-RUN`, `COMPILE-ONLY`, `SOURCE` |
| 1     | [External contract](artifacts/01_external_contract.md)       | `CPU-RUN`, `SOURCE`                 |
| 2     | [Startup topology](artifacts/02_startup_topology.md)         | `CPU-RUN`, `SOURCE`, `BOUNDARY`     |
| 3     | [Request object trace](artifacts/03_request_object_trace.md) | `CPU-RUN`, `SOURCE`, `TEST`         |
| 4     | [IPC and backpressure](artifacts/04_ipc_backpressure.md)     | `CPU-RUN`, `SOURCE`, `TEST`         |
| 5     | [Scheduler trace](artifacts/05_scheduler_trace.md)           | `CPU-RUN`, `SOURCE`, `TEST`         |
| 6     | [Batch to token](artifacts/06_batch_to_token.md)             | `CPU-RUN`, `SOURCE`, `COMPILE-ONLY` |
| 7     | [Model resolution](artifacts/07_model_resolution.md)         | `CPU-RUN`, `SOURCE`, `BOUNDARY`     |
| 8-14  | [Master plan](Plan_max_internals_learning.md)                | queued                              |

## Reproduce

```bash
# Terminal 1: source-built server
HF_HOME=/local/mnt/workspace/.murali_hf \
MAX_SERVE_METRICS_ENDPOINT_PORT=18001 \
./bazelw --output_user_root=/local/mnt/workspace/.murali_bazel/user run \
  --config=prebuilt-mojo \
  --disk_cache=/local/mnt/workspace/.murali_bazel/disk \
  --repository_cache=/local/mnt/workspace/.murali_bazel/repo \
  //max/python/max/_entrypoints:pipelines -- \
  serve --model modularai/SmolLM-135M-Instruct-FP32 \
  --devices cpu --quantization-encoding float32 \
  --max-length 256 --max-batch-size 4 \
  --host 127.0.0.1 --port 18000 --allow-cold-interpreter-cache

# Terminal 2: dependency-free external client
python3 murali_docs/labs/client/max_serve_client.py --mode all

# Route -> prompt -> tokens -> context -> local ZMQ round trip
murali_docs/labs/tracing/run_request_object_trace.sh

# IPC framing -> response routing -> caps -> cancellation
murali_docs/labs/tracing/run_ipc_backpressure_trace.sh

# Continuous batching, chunking, preemption, and DP placement
murali_docs/labs/scheduler/run_scheduler_trace.sh

# CE/TG contexts -> ragged token and row-offset buffers
murali_docs/labs/graph/run_ragged_batch_probe.sh

# HF config -> registry components + adapted weight names
murali_docs/labs/graph/run_model_resolution_probe.sh
```

Evidence was captured through repository revision `c44d2c99b8` on 2026-10-04.
