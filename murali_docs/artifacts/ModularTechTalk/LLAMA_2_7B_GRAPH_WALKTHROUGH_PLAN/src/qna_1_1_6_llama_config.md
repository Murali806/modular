The registry's Llama declaration selects
[`Llama3Config`](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L51)
for MAX's architecture-specific model config. This is MAX's class; it is
separate from the Transformers config returned by `AutoConfig`.

**Call-flow tree — `initialize()` has setup uses; `finalize()` waits for the
loaded weight state:**

- **Memory planning, before the model factory runs:**
  - [`PipelineRegistry.retrieve_factory()`](../../../../max/python/max/pipelines/lib/registry.py#L711)
    asks [`MemoryEstimator.plan()`](../../../../max/python/max/pipelines/lib/memory_estimation.py#L147)
    to estimate the model's memory needs before serving.
    - [`MemoryEstimator.plan()`](../../../../max/python/max/pipelines/lib/memory_estimation.py#L147)
      calls [`arch.config.initialize(...)`](../../../../max/python/max/pipelines/lib/memory_estimation.py#L189)
      with the pipeline/model config and sequence-length bound.
      - [`Llama3Config.initialize()`](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L224)
        reads the Hugging Face config and MAX settings to establish model
        dimensions and runtime choices needed for the plan. This call has no
        loaded model state dictionary to inspect.
- **Model construction, when the worker executes its factory:**
  - [`ModelWorker.run()`](../../../../max/python/max/serve/pipelines/model_worker.py#L340)
    invokes [`model_factory()`](../../../../max/python/max/serve/pipelines/model_worker.py#L444),
    which enters [`TextGenerationPipeline.__init__()`](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L208)
    and resolves weight paths inside that initializer before constructing the model.
    - The pipeline constructs [`Llama3Model`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L334).
      Its [`LlamaModelBase.__init__()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L153)
      runs through [`PipelineModelWithKVCache.__init__()`](../../../../max/python/max/pipelines/lib/interfaces/pipeline_model.py#L1017)
      to [`PipelineModel.__init__()`](../../../../max/python/max/pipelines/lib/interfaces/pipeline_model.py#L553).
      - The base initializer calls
        [`model_config_cls.initialize()`](../../../../max/python/max/pipelines/lib/interfaces/pipeline_model.py#L581).
        Llama sets `model_config_cls` to `Llama3Config` in
        [`LlamaModelBase`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L136),
        so this is another config-only initialization point.
- **Weight-aware config finalization, before graph specs are compiled:**
  - [`LlamaModelBase.load_model()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L234)
    enters [`LlamaModelBase.load_graphs()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L255).
    - [`LlamaModelBase._load_state_dict()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L283)
      obtains the model's state dictionary through
      [`PipelineModelWithKVCache._load_state_dict()`](../../../../max/python/max/pipelines/lib/interfaces/pipeline_model.py#L1090).
    - [`LlamaModelBase._create_model_config()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L228)
      delegates to [`LlamaModelBase._finalized_config()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L210).
      - `_finalized_config()` calls
        [`Llama3Config.initialize()`](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L224)
        to create the config object, then
        [`Llama3Config.finalize()`](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L397)
        with the Hugging Face config and loaded state dictionary.
        - [`Llama3Config.finalize()`](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L397)
          normalizes weight-name prefixes and derives weight-dependent
          settings such as quantization details and normalization dtype. It
          also uses checkpoint metadata such as tied-embedding settings.
    - With that config finalized,
      [`LlamaModelBase.graph_specs()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L287)
      chooses the single-device, tensor-parallel, or data-parallel graph spec.
      [`compile_and_init_graphs(...)`](../../../../max/python/max/pipelines/lib/graph_specs.py#L183)
      is called at this [Llama load site](../../../../max/python/max/pipelines/architectures/llama3/model.py#L274)
      to build, compile, and initialize the selected graph(s).

Here `Llama3Config.finalize()` means **finish the model's Python configuration
using its weights**. It is not `Graph.output()` finalization and not the
compiler's device-graph finalize pass. Those graph/compiler stages happen
after this config is ready.
