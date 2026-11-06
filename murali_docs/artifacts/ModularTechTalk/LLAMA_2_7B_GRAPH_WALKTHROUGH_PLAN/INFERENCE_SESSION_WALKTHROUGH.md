# InferenceSession: devices, runtime and compilation

[Open the animations](link_htmls/max_inference_session_devices_runtime_and_compilation.html).

SOURCE revision: 610f7111836d9a950d45ba52ec9b3e0ed589fe3f. Model context: meta-llama/Llama-2-7b-hf.

All steps are source-derived teaching states. No model inference, compilation, or GPU measurement was run.

## 01 · Inside the constructor

Follow the exact call at ENGINE:L664. Device handles enter; an engine session comes back.

```text
SOURCE · illustrative values; logical call order
MODEL WORKER PROCESS
  TextGenerationPipeline.__init__
    -> InferenceSession(devices=[gpu0, gpu0], num_threads=None)
       -> check explicit MEF import/export policy
       -> final_devices: [gpu0, cpu]  (first occurrences + host CPU)
       -> _InferenceSession(final_devices, extensions, 0)
          -> sessionCompileConfig
          -> runtimeConfig: thread setting + DeviceSpec/context pairs
          -> owned Engine::Context
             -> shared/reused MAX context + CPUDevice
    <- session; store pipeline.session
  configure_session(session) -> model loading follows
At constructor return: decoder artifacts = 0; initialized decoders = 0.
Underlying device contexts and CPU runtime resources can be shared.
```

Watch the device list normalize, then watch compilation settings and the runtime context appear. The host CPU is included even when the supplied list only contains a GPU.

[Pipeline constructs and retains session](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L275) · [InferenceSession.__init__](sources/max/python/max/engine/api.py#L664) · [Native InferenceSession constructor](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L541) · [Engine::Context::create](sources/max/lib/max_license/API/c/EngineContext.cpp#L169)

## 02 · From a graph to a runnable model

A session, a compiled artifact, and an initialized model become ready at different points.

```text
SOURCE · ordinary real-device, ungrouped GraphSpec route
Llama model -> GraphSpec.trace() -> Graph
                           |
                           v
                    existing session S1
                     /             \
          compile(graph)         matching MEF store
          native compiler        compile(artifact_path) -> read MEF
                     \             /
                       CompiledModel
                            |
GraphSpec.weights_registry() -> init_all(artifact, weights)
                            | required weight missing -> ValueError
                            v
                    native loadAll / initModel
                            |
                  executable ModelHandle -> retains native S1
Returned decoder -> Llama model / compiled callable
All arrows represent logical source order, not captured GPU timing.
```

Tracing creates the graph. Compilation produces an artifact. Initialization binds real weights and returns the executable engine model. Each object has its own role.

[Llama load_graphs](sources/max/python/max/pipelines/architectures/llama3/model.py#L254) · [Trace, compile and initialize graphs](sources/max/python/max/pipelines/lib/graph_specs.py#L183) · [Compile or reuse a MEF](sources/max/python/max/engine/api.py#L897) · [Validate weights and initialize](sources/max/python/max/engine/api.py#L1160)

## 03 · Change the batch. Reuse the model.

Prefill and decode change the inputs while the session and initialized decoder remain available.

```text
SOURCE + BOUNDARY · illustrative token IDs, placements and event timing
MODEL WORKER PROCESS                            OS / HARDWARE
  Pipeline.prepare_batch()
      prefill: [17 42 6 | 9 21], offsets [0,3,5]
      decode : [88 | 31],        offsets [0,1,2]
          |
  LlamaModelBase.execute -> existing engine Model M1
          |                    |
          |             ModelHandle -> native S1 -> Engine::Context
          |                    |
          |             executeFromDriver ------> driver -> GPU work
          |                    |                             |
          |             output Buffers <--------------------+
          v                    |
  batch output processing / configured sampling
  next model call reuses S1 and M1
Buffer metadata controls placement; an output Buffer can stay on device.
Setup counts are fixture counts, not performance measurements.
```

The Python pipeline prepares Buffers. Llama's engine Model calls the native runtime through its retained session. Output logits are processed and sampled per sequence. A/B tiles depict sequence views; KV tile counts illustrate token entries, not actual cache-page allocation.

[Prepare model inputs and KV references](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L465) · [Llama executes its engine Model](sources/max/python/max/pipelines/architectures/llama3/model.py#L201) · [Native execution using the session context](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L521) · [Wrap native results as Python Buffers](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L55) · [Prepare, execute and sample](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L594)

## 04 · Who keeps the session alive?

Follow native-session ownership after the Python wrapper or a model is released.

```text
SOURCE · minimal ownership example; only pictured owners are counted
Python InferenceSession._impl -----------+
decoder ModelHandle.inferenceSession ---+--> native session S1
sampler ModelHandle.inferenceSession ---+        |
                                                +-- compile/runtime config
                                                +-- retained Device handles
                                                +-- owned Engine::Context
                                                       |
                                                       +-- MAX context / CPUDevice
Drop wrapper -> model references still retain S1
Drop decoder -> sampler can still retain S1
Drop sampler -> with no other references/work, native destruction is eligible
Shared underlying contexts/driver resources may remain.
ModelHandle member order keeps the session alive through model cleanup.
```

Native ModelHandles retain S1. Its context must outlive model state that uses the runtime to release buffers. This example shows a wrapper, a decoder and a sampler.

[self._impl = _InferenceSession](sources/max/python/max/engine/api.py#L707) · [Native loadAll creates ModelHandles](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L289) · [Native session fields](sources/max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h#L50) · [ModelHandle ownership and lifetime](sources/max/python/max/_core/max_license/include/_core/Engine/InferenceSession.h#L112) · [Engine::Context::create](sources/max/lib/max_license/API/c/EngineContext.cpp#L169)

## Build and verify

```sh
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_inference_session.cjs
# Set NODE_PATH to an available Playwright installation when necessary.
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_inference_session_browser.cjs
```
