**Call-flow tree — when the Llama mapping is recorded, imported, and used:**

- Click option loading calls
  [`WithLazyPipelineOptions._add_options()`](../../../../max/python/max/_entrypoints/pipelines.py#L81),
  whose CLI-config import can reach `max.pipelines` before `cli_serve()`.
  On the first package import,
  [`max.pipelines.__init__`](../../../../max/python/max/pipelines/__init__.py#L80)
  calls [`register_all_models()`](../../../../max/python/max/pipelines/architectures/__init__.py#L58).
  - The architecture table includes
    [`LlamaForCausalLM → .llama3 / llama_arch`](../../../../max/python/max/pipelines/architectures/__init__.py#L269).
    These three values mean: use `LlamaForCausalLM` as the lookup key; when
    needed, import `.llama3` relative to the architecture package; then read
    its `llama_arch` declaration.
  - [`PipelineRegistry.register_lazy()`](../../../../max/python/max/pipelines/lib/registry.py#L415)
    delegates that mapping to
    [`ArchLookup.register_lazy()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L767).
    This appends the module, symbol, and package tuple to the in-memory
    `_lazy_architectures["LlamaForCausalLM"]` table. It records *how to import*
    the implementation; it does not import the Llama module, instantiate a
    `Llama3Model`, or load model weights. Deferring imports lets MAX register
    many supported architectures without importing every implementation at
    startup.
- **First lookup of `LlamaForCausalLM` materializes that record:**
  - With no task specified,
    [`PipelineRegistry.retrieve_pipeline_task()`](../../../../max/python/max/pipelines/lib/registry.py#L1047)
    needs the architecture's registered task. It calls
    [`PipelineRegistry._materialize_lazy()`](../../../../max/python/max/pipelines/lib/registry.py#L452)
    before reading the task table.
  - With a task already specified,
    [`PipelineConfig.from_args()`](../../../../max/python/max/pipelines/lib/config/config.py#L1562)
    calls `find_architecture()`, then `ArchLookup.find()` and `resolve()`.
    This can materialize the pending entry before `retrieve_factory()`.
    Later, `retrieve_factory()` selects through
    [`PipelineRegistry.architecture_for_config()`](../../../../max/python/max/pipelines/lib/registry.py#L496),
    [`PipelineRegistry.retrieve_architecture()`](../../../../max/python/max/pipelines/lib/registry.py#L471),
    [`ArchLookup.find()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L947),
    and [`ArchLookup.resolve()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L930).
    The pending lazy record is materialized if that lookup needs it.
  - [`ArchLookup.materialize(name)`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L804)
    removes the pending record, calls
    [`importlib.import_module()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L815)
    for `.llama3`, and retrieves the `llama_arch` symbol from that module.
    - [`ArchLookup.register(llama_arch)`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L705)
      puts the resulting `SupportedArchitecture` declaration into the
      in-memory name and `(name, task)` lookup tables. Future lookups can reuse
      that registered object instead of importing it again.
    - [`llama_arch`](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L28)
      describes the implementation classes and metadata MAX needs, including
      [`Llama3Model`, `Llama3Config`, and `TextTokenizer`](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L42).
      It is an architecture description, not a model instance or a database
      of checkpoint tensors.
- **The selected declaration is turned into a serving factory:**
  - [`PipelineRegistry.retrieve_factory()`](../../../../max/python/max/pipelines/lib/registry.py#L711)
    uses the architecture declaration to select the tokenizer, model pipeline
    class, weight adapter, and memory plan. It creates a deferred factory; the
    model object is not constructed at this point.
  - [`ServingTokenGeneratorSettings`](../../../../max/python/max/serve/api_server.py#L103)
    packages that factory with the resolved `PipelineConfig`, tokenizer, task,
    and memory plan. The handler passes this object to
    [`fastapi_app(settings, pipeline_settings)`](../../../../max/python/max/serve/api_server.py#L346)
    alongside server-level settings.
    - [`Settings`](../../../../max/python/max/serve/config.py#L93)
      contains server behavior such as exposed APIs, host, port, request
      limits, and queue limits. It comes from server flags and `MAX_SERVE_*`
      environment settings.
    - `ServingTokenGeneratorSettings` contains the *resolved model-serving
      inputs*: which factory to call, which config/tokenizer/task to use, and
      which memory plan was selected. `fastapi_app()` uses the server settings
      to configure routes and middleware, while its lifespan setup retains the
      serving settings to start and connect the model worker.
  - [`fastapi_config(app, server_settings=settings)`](../../../../max/python/max/serve/api_server.py#L482)
    converts the already-built FastAPI app and server settings into Uvicorn's
    `Config`. The same `Settings` object is passed again because this is a
    different job: `fastapi_app()` configures the ASGI application, while
    `fastapi_config()` reads host, port, and timeout values for the HTTP
    server.
  - [`Server(config)`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L134)
    constructs Uvicorn's server object from that runtime config; construction
    does not start listening. The handler's `serve()` coroutine enters
    [`lifespan(...)`](../../../../max/python/max/serve/api_server.py#L118)
    first, then calls [`server.serve(...)`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L147).
    During lifespan startup,
    [`start_model_worker(...)`](../../../../max/python/max/serve/pipelines/model_worker.py#L760)
    waits for the worker to become ready. In the worker,
    [`ModelWorker.run()`](../../../../max/python/max/serve/pipelines/model_worker.py#L340)
    invokes [`model_factory()`](../../../../max/python/max/serve/pipelines/model_worker.py#L444).

**Why the guide says weight downloading happens later:** “Later” means after
the CLI has resolved the config, registry entry, tokenizer, and model factory.
Some earlier metadata assets can already be read or fetched: `AutoConfig`
loads `config.json`, the tokenizer may load vocabulary files, and generation
defaults may load `generation_config.json`. The large model weight files are
resolved when the worker invokes the factory and constructs
[`TextGenerationPipeline`](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L208).
Its [`resolved_weight_paths()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1563)
calls [`download_weight_files()`](../../../../max/python/max/pipelines/weights/hf_utils.py#L239)
for an online repo; that checks direct paths and the Hugging Face cache before
downloading missing files. In the normal server path the worker reports ready
before `server.serve()` begins accepting HTTP traffic, so weight loading is a
startup step, not work deferred until the first request. The full weight path
is traced in **QnA 1.1.5**.
