**Example commands — specify the task, or let MAX infer it:**

```bash
max serve --model meta-llama/Llama-2-7b-hf --task text_generation
max serve --model meta-llama/Llama-2-7b-hf
```

Assume the effective configuration does not set `task` in a config file. The
default `--headless` and `--cascade` settings choose the standard API-server
path shown below.

**Call-flow tree — CLI arguments to Hugging Face `AutoConfig`:**

- [`@main.command(name="serve", cls=WithLazyPipelineOptions)`](../../../../max/python/max/_entrypoints/pipelines.py#L349)
  registers `max serve` as a Click command whose callback is
  [`cli_serve()`](../../../../max/python/max/_entrypoints/pipelines.py#L433).
  - [`WithLazyPipelineOptions._ensure_options_loaded()`](../../../../max/python/max/_entrypoints/pipelines.py#L86)
    adds the model and pipeline options just before Click parses the command.
    - [`pipeline_config_options()`](../../../../max/python/max/_entrypoints/cli/config.py#L443)
      generates those options from MAX's configuration fields.
      - [`get_normalized_flag_names()`](../../../../max/python/max/_entrypoints/cli/config.py#L204)
        maps the `PipelineArgs.model_path` field to the user-facing
        `--model` flag (and also accepts `--model-path`). Thus the example
        value becomes `config_kwargs["model_path"]` in `cli_serve()`.
- [`cli_serve(..., **config_kwargs)`](../../../../max/python/max/_entrypoints/pipelines.py#L433)
  is the handler Click invokes after parsing the command. It builds server
  [`Settings`](../../../../max/python/max/serve/config.py#L93) from server
  flags such as `--host` and `--port`, then turns the model-related keyword
  arguments into pipeline arguments.
  - [`PipelineArgs.from_flat_kwargs(**config_kwargs)`](../../../../max/python/max/pipelines/lib/pipeline_args.py#L460)
    constructs the immutable `PipelineArgs` input. CLI options such as
    `--model`, `--task`, and `--max-length` arrive as keyword fields; when
    `--config-file` is used, its YAML values are merged with the CLI values,
    with explicit CLI values winning for a field.
    For these commands, it also calls
    [`ModelManifest.from_model_path()`](../../../../max/python/max/pipelines/lib/model_manifest.py#L383).
    The single-model probe builds a temporary model config and can load HF
    config and generation defaults **before task inference**, in both examples.
  - [`PipelineArgs.task`](../../../../max/python/max/pipelines/lib/pipeline_args.py#L153)
    defaults to `PipelineTask.UNDEFINED`. It stays undefined when neither
    the CLI nor the merged config file specifies a task. That effective value
    is the condition for task inference; merely omitting `--task` is not
    enough if the config file supplies one.
- [`serve_api_server_and_model_worker(settings, pipeline_args)`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60)
  is called for the standard server path. `--headless` uses a worker-only
  route, and `--cascade` selects a separate server path.
  - **Task inference branch — runs only when `pipeline_args.task` is
    `PipelineTask.UNDEFINED`:**
    - [`PipelineArgs.main_architecture_name`](../../../../max/python/max/pipelines/lib/pipeline_args.py#L440)
      asks the model config for the checkpoint's first `architectures` entry;
      for the usual Llama 2 7B Hugging Face config this is
      `LlamaForCausalLM`.
      - [`MAXModelConfig.from_pipeline_args()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1155)
        constructs the model metadata needed to inspect the checkpoint.
        - [`_build_model_config()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L155)
          prepares the repository handle and Hugging Face config.
          - [`MAXModelConfig._populate_hf_config()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1067)
            loads a config when the repository has a loadable config file.
            - [`load_huggingface_config()`](../../../../max/python/max/pipelines/lib/_hf_config.py#L90)
              checks MAX's process-wide config cache before asking
              Transformers to load the config.
              - [`AutoConfig.from_pretrained(...)`](../../../../max/python/max/pipelines/lib/_hf_config.py#L121)
                is the Transformers call site. It reads checkpoint metadata
                such as `model_type` and `architectures`; it returns a
                Transformers config object, not MAX's `Llama3Config`.
    - [`PipelineRegistry.retrieve_pipeline_task(architecture_name)`](../../../../max/python/max/pipelines/lib/registry.py#L1047)
      maps that architecture name to its registered MAX task. The Llama
      causal-LM registration resolves to text generation, and the handler
      continues with a copy of `PipelineArgs` containing that task.
  - **Explicit-task branch:** if `--task text_generation` (or a task in the
    config file) already populated `PipelineArgs.task`, the `UNDEFINED` test
    is false and the architecture-based task-inference prepass is skipped.
    The checkpoint config is still needed by the full pipeline-config step.
  - [`PipelineConfig.from_args(pipeline_args)`](../../../../max/python/max/pipelines/lib/config/config.py#L1483)
    builds the resolved, validated config used by architecture selection and
    serving. Its model config also needs the Hugging Face config. On the
    inference branch, and also with an explicit task, the HF loader can reuse
    the earlier argument-probe result from its process cache. Configuration
    construction itself performs architecture lookup before `retrieve_factory()`.

The inference branch only discovers the task from checkpoint metadata. It
does not construct the Llama model or load the checkpoint's large weight
files; those steps are traced in **QnA 1.1.5**.
