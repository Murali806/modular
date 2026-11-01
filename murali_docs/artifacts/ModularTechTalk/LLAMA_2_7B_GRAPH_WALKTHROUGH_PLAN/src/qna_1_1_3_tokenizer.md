**Call-flow tree — from the resolved config to `AutoTokenizer`:**

- [`serve_api_server_and_model_worker()`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60)
  first builds [`PipelineConfig.from_args()`](../../../../max/python/max/pipelines/lib/config/config.py#L1483),
  then calls [`PipelineRegistry.retrieve_factory()`](../../../../max/python/max/pipelines/lib/registry.py#L711)
  to prepare the model-serving components.
  - [`PipelineRegistry.architecture_for_config()`](../../../../max/python/max/pipelines/lib/registry.py#L496)
    resolves the checkpoint architecture name and task to the registered Llama
    declaration.
    - [`PipelineRegistry.retrieve_architecture()`](../../../../max/python/max/pipelines/lib/registry.py#L471)
      delegates the architecture lookup to
      [`ArchLookup.find()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L947)
      and [`ArchLookup.resolve()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L930).
  - [`llama_arch`](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L28)
    declares [`TextTokenizer`](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L42)
    as this architecture's tokenizer implementation.
    - At the selected-architecture call site,
      [`arch.tokenizer(...)`](../../../../max/python/max/pipelines/lib/registry.py#L957)
      constructs the tokenizer with the model path, Hugging Face revision,
      maximum length, and `trust_remote_code` setting.
      - [`TextTokenizer.__init__()`](../../../../max/python/max/pipelines/lib/tokenizer.py#L618)
        creates MAX's tokenizer wrapper and delegates checkpoint tokenizer
        loading to [`AutoTokenizer.from_pretrained(...)`](../../../../max/python/max/pipelines/lib/tokenizer.py#L633).
        That Transformers call reads tokenizer files such as the vocabulary,
        tokenization rules, and special-token metadata. It uses the local Hub
        cache when possible and can fetch missing tokenizer assets at this
        preparation stage.
  - `retrieve_factory()` returns the tokenizer together with a deferred model
    factory and a memory plan. The serving handler stores those in
    [`ServingTokenGeneratorSettings`](../../../../max/python/max/serve/api_server.py#L103);
    the worker later uses this tokenizer wrapper to encode prompts and decode
    generated token IDs.

Tokenizer loading is separate from model-weight loading and graph creation.
The tokenizer may be fetched before the API server is assembled; the model
factory loads the large tensor files during worker startup, as traced in
**QnA 1.1.5**.
