**Call-flow tree — when the large checkpoint weight files are resolved:**

- [`serve_api_server_and_model_worker()`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60)
  calls [`PipelineRegistry.retrieve_factory()`](../../../../max/python/max/pipelines/lib/registry.py#L711)
  after building the `PipelineConfig`.
  - [`retrieve_factory()`](../../../../max/python/max/pipelines/lib/registry.py#L711)
    resolves the registered architecture and memory plan, constructs the
    tokenizer, then packages a deferred pipeline constructor with
    [`functools.partial(...)`](../../../../max/python/max/pipelines/lib/registry.py#L992).
    It returns that factory; it has not called the factory or constructed the
    7B model yet.
  - [`ServingTokenGeneratorSettings`](../../../../max/python/max/serve/api_server.py#L103)
    holds the factory and resolved pipeline config until serving startup.
    [`lifespan(...)`](../../../../max/python/max/serve/api_server.py#L118)
    calls [`start_model_worker(...)`](../../../../max/python/max/serve/pipelines/model_worker.py#L760)
    at this [`lifespan` call site](../../../../max/python/max/serve/api_server.py#L178);
    it starts the worker process and waits for it to report ready.
    - [`ModelWorker.run()`](../../../../max/python/max/serve/pipelines/model_worker.py#L340)
      invokes [`model_factory()`](../../../../max/python/max/serve/pipelines/model_worker.py#L444).
      This is when the deferred pipeline constructor runs, during server
      startup and before the API begins serving requests.
      - [`TextGenerationPipeline.__init__()`](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L208)
        sets up the session and asks the model config for local weight paths.
        - [`MAXModelConfig.resolved_weight_paths()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1563)
          uses a direct local path for a local checkpoint; for online
          Hugging Face weights it calls
          [`download_weight_files()`](../../../../max/python/max/pipelines/weights/hf_utils.py#L239)
          at this [MAX call site](../../../../max/python/max/pipelines/lib/config/model_config.py#L1594).
          The downloader checks direct paths and the Hugging Face cache first,
          then fetches missing files from the requested repo and revision.
          With `header_only_weights`, MAX takes a separate header-only path.
        - With the resolved paths,
          [`load_weights(weight_paths)`](../../../../max/python/max/graph/weights/load.py#L24)
          returns a format-specific `Weights` provider; the pipeline invokes it
          at this [call site](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L310).
          The Llama model consumes that provider in
          [`LlamaModelBase.load_graphs()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L255)
          and [`LlamaModelBase._load_state_dict()`](../../../../max/python/max/pipelines/architectures/llama3/model.py#L283),
          followed by configuration finalization and graph construction in
          **QnA 1.1.6**.

The timing distinction is: Hugging Face config, tokenizer, and optional
generation files can be accessed while the CLI prepares the pipeline; the
large tensor files are resolved when the worker executes the model factory.
For a warm cache or local checkpoint, `download_weight_files()` may perform no
network download at all.
