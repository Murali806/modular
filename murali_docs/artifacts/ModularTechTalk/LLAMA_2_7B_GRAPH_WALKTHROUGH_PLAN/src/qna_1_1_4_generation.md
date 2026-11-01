**Call-flow tree — optional Hugging Face generation defaults to MAX sampling:**

- [`PipelineConfig.from_args()`](../../../../max/python/max/pipelines/lib/config/config.py#L1483)
  constructs the serving config and its main
  [`MAXModelConfig.from_pipeline_args()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1155).
  - [`_build_model_config()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L155)
    constructs the model config from the model path and resolved repository
    metadata.
    - During construction,
      [`MAXModelConfig._populate_generation_config()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1087)
      asks for the optional Hugging Face generation defaults when a model path
      is present.
      - [`MAXModelConfig._make_generation_config()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1377)
        calls [`GenerationConfig.from_pretrained(...)`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1390)
        with the checkpoint repo, revision, and `trust_remote_code` setting.
        This reads an optional `generation_config.json`; the result is a
        Transformers `GenerationConfig`, not MAX's `Llama3Config`.
        - If the file is absent or loading raises, MAX logs at debug level
          and returns a default `GenerationConfig` from this same method.
  - The model config exposes the loaded values as sampling defaults. During
    `PipelineConfig.from_args()`,
    [`SamplingConfig.from_generation_config_sampling_defaults()`](../../../../max/python/max/pipelines/lib/config/config.py#L1532)
    combines supported defaults with the sampling fields the user explicitly
    set. The result determines the defaults the serving pipeline can honor.

This config is constructed before the model factory loads the large weight
files. A task-inference prepass may create a temporary `MAXModelConfig` first;
each model-config instance can make its own generation-config load attempt,
while the Hugging Face Hub cache may satisfy repeat file reads locally.
