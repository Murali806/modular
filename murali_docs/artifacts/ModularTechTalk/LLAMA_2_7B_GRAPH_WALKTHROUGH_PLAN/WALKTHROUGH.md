# Llama 2 7B: from graph to executable

Evidence: `SOURCE` unless marked otherwise. Source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Read [the interactive guide](index.html) for selectable diagrams and code-line highlighting.
This companion contains the same six chapters and complete expandable source functions.

Code is from the pinned checkout. State/buffer examples are illustrations; repository tests are inspected, not run.
No Llama inference or compiler dump is presented as a measured result.

## 1. Select and configure Llama

**The checkpoint supplies a model configuration; MAX uses that configuration
to build a decoder graph.** The `llama3` name identifies MAX's implementation
package. Its registration maps the
`LlamaForCausalLM` architecture name to `Llama3Model`.

<details id="qna-1"><summary><strong>QnA 1</strong> — Package name, Hugging Face checkpoint, and architecture registration</summary>

### 1. Where does the package name `llama3` come from?

**It is the name of an implementation package in MAX's source tree:**
`max.pipelines.architectures.llama3`. The built-in registration explicitly
connects the checkpoint architecture name to that package.

`SOURCE` — the exact entry in
[`register_all_models()`](../../../../max/python/max/pipelines/architectures/__init__.py#L269):

```python
_LazyArch("LlamaForCausalLM", ".llama3", "llama_arch"),
```

| Entry | Meaning |
|---|---|
| `"LlamaForCausalLM"` | Architecture name used for lookup |
| `".llama3"` | Package to import, relative to `max.pipelines.architectures` |
| `"llama_arch"` | Registration object to retrieve from the imported package |

[`llama_arch`](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L28)
selects the `Llama3Model` class and the `Llama3Config` class. The package keeps
its name while
[`Llama3Config` reads the checkpoint dimensions](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L344):
hidden size, attention and KV heads, layer count, intermediate size, and
vocabulary size. This is why the Llama 2 7B walkthrough enters a directory
named `llama3`. The architecture match selects a candidate implementation;
compatible weights, encoding, and settings are still required.

### 2. What comes from Hugging Face, and how does MAX match it to registered objects?

**Hugging Face supplies checkpoint files. MAX supplies the implementation
classes and their registration.** Files are fetched as needed; the small
configuration can be inspected before downloading the large weight files.

| Checkpoint asset | What it contains | Use in the built-in MAX path |
|---|---|---|
| `config.json` | Architecture name, dimensions, attention/RoPE settings, and other model metadata | Select and configure an implementation |
| Weight files, commonly `.safetensors` | Learned tensors for embeddings, attention, MLPs, and normalization | Supply model parameters |
| Weight index, when present | Tensor-name-to-shard mapping | Describe which file contains each tensor; MAX's selected weight loader determines how files are opened |
| Tokenizer files, such as `tokenizer.json` or `tokenizer.model`, plus tokenizer settings | Vocabulary, tokenization rules, and special tokens | Convert text to token IDs and back |
| Optional `generation_config.json` | Generation defaults | Supply defaults when available; the config loader has a fallback |

`BOUNDARY` — a typical Hugging Face Hub cache layout is shown below. This is
an illustration, not an inspection of a downloaded Llama checkpoint. The
cache root is configurable; snapshots are keyed by checkpoint revision.

```text
~/.cache/huggingface/hub/
└── models--meta-llama--Llama-2-7b-hf/
    ├── refs/
    ├── blobs/
    └── snapshots/<checkpoint-commit>/
        ├── config.json
        ├── <selected weight files>
        └── <tokenizer and other fetched assets>
```

| Entry | What it means |
|---|---|
| `~/.cache/huggingface/hub/` | Default Hub cache root; `HF_HOME` or `HF_HUB_CACHE` can change it. |
| `models--meta-llama--Llama-2-7b-hf/` | Cache namespace for the `meta-llama/Llama-2-7b-hf` repository. |
| `refs/` | Cached branch or tag names, such as `main`, mapped to the commit each ref resolved to. |
| `blobs/` | The downloaded file contents; different revision snapshots can reuse the same blob. |
| `snapshots/<checkpoint-commit>/` | The file view for one exact repository commit; entries normally point to the shared blobs. |
| `config.json` | Model metadata, including `model_type`, `architectures`, and dimension settings. |
| `<selected weight files>` | Learned tensors in the chosen format; large checkpoints can split them into shards. |
| `<tokenizer and other fetched assets>` | Tokenizer data/settings and optional files such as `generation_config.json`. |

These are cache roles: a snapshot does not mean MAX copied or downloaded every
repository file. MAX/Hugging Face resolve the requested revision and fetch the
configuration, selected weights, tokenizer files, or generation settings as
their loaders need them.

`SOURCE` —
[`download_weight_files()`](../../../../max/python/max/pipelines/weights/hf_utils.py#L239)
checks direct local paths, then the Hugging Face cache, then downloads as
needed. Configuration uses
[`AutoConfig.from_pretrained()`](../../../../max/python/max/pipelines/lib/_hf_config.py#L90),
tokenization uses
[`AutoTokenizer.from_pretrained()`](../../../../max/python/max/pipelines/lib/tokenizer.py#L633),
and generation defaults use
[`GenerationConfig.from_pretrained()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1378).

The `AutoConfig` call raises a broader question: how does `max serve` move
from checkpoint metadata and registry selection through tokenizer and weight
loading to MAX's finalized `Llama3Config`? Expand nested **QnA 1.1**; its
**QnA 1.1.x** cells trace each path.

<details id="qna-1-1"><summary><strong>QnA 1.1 · Checkpoint-to-model call paths</strong> — How does max serve prepare a Llama checkpoint and model?</summary>

**`AutoConfig` here is Hugging Face Transformers; it is not MAX's
`Llama3Config`.** For a normal Llama 2 7B Transformers checkpoint,
`config.json` contains `model_type: "llama"`. Hugging Face's
`AutoConfig.from_pretrained()` reads that file and selects the Transformers
config class, normally `transformers.LlamaConfig`. MAX then uses the returned
metadata to build its own architecture-specific config.

`Llama3Config.initialize()` combines Hugging Face fields with MAX serving
settings. `Llama3Config.finalize()` uses loaded weight metadata to resolve
weight-dependent choices. They are separate from the Transformers config
object:

| Call or class | Defined by | Role in Llama 2 7B serving |
|---|---|---|
| [`AutoConfig.from_pretrained(...)`](../../../../max/python/max/pipelines/lib/_hf_config.py#L121) | Hugging Face Transformers, called by MAX | Read `config.json` and return a Transformers `LlamaConfig` with dimensions and RoPE settings. The link opens MAX's exact call site. |
| [`Llama3Config.initialize(...)`](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L224) | MAX | Combine checkpoint metadata with devices, dtype, sequence length, quantization encoding, and KV-cache settings. |
| [`Llama3Config.finalize(...)`](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L397) | MAX | Inspect the loaded weight state and finish settings such as quantization, normalization, tied embeddings, and projection layout. |

The checkpoint's `model_type` selects a Transformers config class; its
`architectures[0]` (normally `"LlamaForCausalLM"`) selects a MAX architecture
registration. `llama3` is MAX's implementation package name, not a package
name supplied by this Llama 2 checkpoint.

In the HTML guide, each linked MAX function opens **Source Explorer** at its
definition and highlights that line. Links for Transformers calls open MAX's
invocation site because the Transformers implementation is outside this
repository. Expand a numbered cell below to follow each serving path.

Use **Call stack tree ↗** in this QnA 1.1 header to open all six text trees in
a separate window/tab, with component labels, object handoffs and source links.
The same trees are available as [plain text](CALL_STACKS.txt).

<details><summary>Startup map · SOURCE</summary>

```text
SOURCE: startup order for the standard Llama 2 7B Graph API route

MAIN / API PROCESS               HF HUB / DISK CACHE          MODEL WORKER PROCESS
------------------------------   -------------------------    ------------------------------
Click loads options
  -> register lazy import map                                 (not started)
CLI builds PipelineArgs
  -> checkpoint probe ---------> config / generation files
  -> infer task if UNDEFINED
  -> build PipelineConfig
Registry prepares factory
  -> memory plan
  -> tokenizer ----------------> tokenizer assets
  -> deferred factory                                        (still not started)
FastAPI app -> Uvicorn Config
  -> Server object
Enter MAX lifespan
  -> telemetry + ZMQ interface
  -> spawn worker -----------------------------------------> reconstruct config; own HF cache
  |                                                          invoke model_factory()
  |                              weight files <------------- resolve paths / fetch misses
  |                                                          Llama3Config initialize/finalize
  |                                                          trace -> compile -> init graphs
  |                                                          warmup + queues + scheduler
  <- readiness Event --------------------------------------- alive.set()
  -> API proxy <============= ZeroMQ connection ============> worker queues
  -> yield lifespan
server.serve(): accept HTTP                                   scheduler loop

After startup: request/cancel -> worker; results -> API proxy -> HTTP client.
Cache hits may avoid downloads. Arrows show order/data flow, not measured durations.
```

</details>

<details id="qna-1-1-1"><summary><strong>QnA 1.1.1 · HF config loading</strong> — Who calls load_huggingface_config and AutoConfig?</summary>

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

</details>

<details id="qna-1-1-2"><summary><strong>QnA 1.1.2 · Architecture registration</strong> — When does MAX register and import the Llama implementation?</summary>

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

</details>

<details id="qna-1-1-3"><summary><strong>QnA 1.1.3 · Tokenizer loading</strong> — How does max serve reach AutoTokenizer?</summary>

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

</details>

<details id="qna-1-1-4"><summary><strong>QnA 1.1.4 · Generation defaults</strong> — How does GenerationConfig reach MAX sampling settings?</summary>

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
files. The argument manifest probe can create a temporary `MAXModelConfig`
even before task inference; an inference prepass may create another instance.
Each model-config instance can make its own generation-config load attempt,
while the Hugging Face Hub cache may satisfy repeat file reads locally.

</details>

<details id="qna-1-1-5"><summary><strong>QnA 1.1.5 · Weight files</strong> — When are weight paths resolved, downloaded, and loaded?</summary>

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

</details>

<details id="qna-1-1-6"><summary><strong>QnA 1.1.6 · Llama3Config lifecycle</strong> — When are initialize and finalize called?</summary>

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

</details>

</details>

The tokenizer and generation configuration are separate inputs to serving:

- [`AutoTokenizer.from_pretrained()`](../../../../max/python/max/pipelines/lib/tokenizer.py#L633)
  creates the Transformers tokenizer delegate used by MAX's `TextTokenizer`.
  It loads the vocabulary and tokenization rules so prompts become token IDs
  and generated IDs become text. It does not create the decoder graph.
- [`GenerationConfig.from_pretrained()`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1377)
  reads optional generation defaults such as `max_length`, `temperature`, and
  `top_p`. MAX maps supported defaults into its sampling configuration;
  explicit request sampling settings can take precedence. If loading the
  generation config fails, this path logs at debug level and uses a default
  Hugging Face `GenerationConfig`.

**The architecture “database” is a set of Python dictionaries in process
memory**, owned by
[`ArchLookup`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L649).
The global `PIPELINE_REGISTRY` shares that lookup object.

| Storage | Key → value | Lifetime / purpose |
|---|---|---|
| Hugging Face disk cache | Repository + revision + filename → local file | Reuse downloaded assets across runs |
| `_lazy_architectures` | Architecture name → import instructions | Describe how to load a built-in registration |
| `architectures` | Architecture name → `SupportedArchitecture` object | Reuse the imported description within the process |
| `_architectures_by_task` | Architecture name + task → description | Disambiguate architectures that share a name |

`SOURCE` — selection for this Graph API walkthrough:

```text
Checkpoint config
  architectures[0] = "LlamaForCausalLM"
                 |
                 v
MAX architecture lookup
  .llama3 package -> llama_arch registration
                 |
                 v
Implementation classes
  Llama3Model + Llama3Config + tokenizer + weight adapters
                 |
                 v
Checkpoint settings + weights + MAX serving settings
                 |
                 v
Configured model instance -> graph construction and compilation
```

MAX reads the key in
[`MAXModelConfig.architecture_name`](../../../../max/python/max/pipelines/lib/config/model_config.py#L1330)
and performs selection through
[`architecture_for_config()`](../../../../max/python/max/pipelines/lib/registry.py#L496)
and [`ArchLookup.find()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L947).
The Graph API path uses `prefer_module_v3=False`. The `example_repo_ids` in
`llama_arch` are examples; the lookup key here is the architecture name.
Checkpoint dimensions then become values in the selected config class.

### 3. When are architectures registered, and when is the model created?

**The built-in mappings ship with MAX and are registered when Python imports
`max.pipelines`. The matching architecture is imported on its first lookup.**
Creating a model instance happens later.

| Stage | Trigger | What exists afterward |
|---|---|---|
| Create lookup objects | Import the lookup and registry modules | `ARCH_LOOKUP` and `PIPELINE_REGISTRY` share initially empty architecture tables |
| Record built-in mappings | `max.pipelines.__init__` calls `register_all_models()` | Deferred mappings such as `LlamaForCausalLM → .llama3 / llama_arch` |
| Import the selected architecture | First lookup of `LlamaForCausalLM` | `materialize()` imports the package, reads `llama_arch`, and registers its `SupportedArchitecture` description |
| Construct the model | The serving pipeline calls the selected `pipeline_model` class | A `Llama3Model` instance receives configuration, weights, session, and devices; the graph-loading path follows |

`SOURCE` — follow these exact locations:

1. [`ARCH_LOOKUP = ArchLookup()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L1003)
   and [`PIPELINE_REGISTRY = PipelineRegistry([], arch_lookup=ARCH_LOOKUP)`](../../../../max/python/max/pipelines/lib/registry.py#L1127)
   create the shared lookup infrastructure.
2. [`max.pipelines` calls `register_all_models()`](../../../../max/python/max/pipelines/__init__.py#L80).
   Its [registration loop](../../../../max/python/max/pipelines/architectures/__init__.py#L436)
   records module, symbol, and package. A guard prevents repeated population.
3. [`ArchLookup.materialize()`](../../../../max/python/max/pipelines/lib/arch_lookup.py#L804)
   calls `importlib.import_module()`, gets the declaration, and registers it.
4. [`TextGenerationPipeline` constructs the selected model](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L304)
   using configuration and loaded weights.

A registration stores class references and model-loading metadata. It does
not hold a loaded 7B model. Downloading another compatible checkpoint can reuse
the same registration with different settings and tensors. Registration is
separate from the graph finalization and compilation explained below.

</details>


This guide uses **Llama 2 7B**, one device, and the Graph API path. The working
checkpoint name is `meta-llama/Llama-2-7b-hf`; a checkpoint revision and actual
hardware run have not been captured. Hidden size, heads, vocabulary, and layer
count are read from the checkpoint configuration rather than inferred from
the name “7B.” The Graph API is the default when `prefer_module_v3=False`.

`SOURCE` — the initial serving route is
[`serve_api_server_and_model_worker()`](../../../../max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py#L60)
→ registry factory → model construction in
[`TextGenerationPipeline`](../../../../max/python/max/pipelines/lib/pipeline_variants/text_generation.py#L274).
The model constructor calls `load_model()`, which selects the decoder returned
by `load_graphs()` on this route.

```text
HF architecture + checkpoint settings + MAX serving settings
                     |
                     v
Llama3Model -> state dictionary -> resolved Llama3Config
                     |
                     v
single-device GraphSpec -> trace -> compile -> initialize
```

| Object | What it contains | When it matters |
|---|---|---|
| `Llama3Config` | Architecture dimensions, dtype, devices, RoPE, KV parameters, output policy | Module construction and tracing |
| `GraphSpec` | Module, graph name, input-type tree, weight registry, optional grouping | Recipe for tracing and compilation |
| `Graph` | Symbolic operations and an MLIR module | Finalization and compilation |
| `CompiledModel` | Compiled artifact handle and expected weight metadata | Initialization and artifact reuse |
| Engine `Model` | Initialized executable state | Execution on concrete buffers |

Expand the registration, then follow the model-loading method. The full
functions include alternative branches so you can see exactly where the
single-device choice is made.

<a id="code-registration"></a>

<details><summary><strong>llama_arch</strong> — 33 complete lines · SOURCE</summary>

Map the checkpoint's architecture name to the Llama pipeline implementation.

Original: [max/python/max/pipelines/architectures/llama3/arch.py](../../../../max/python/max/pipelines/architectures/llama3/arch.py#L28); lines 28–60.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/arch.py).

```python
llama_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    example_repo_ids=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "modularai/Llama-3.1-8B-Instruct-GGUF",
    ],
    default_encoding=Llama3Config.DEFAULT_ENCODING,
    supported_encodings=Llama3Config.SUPPORTED_ENCODINGS,
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=Llama3Config,
    batching=Llama3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
    cascade_pipeline_factory=CommonTextGenPipeline,
    # An EAGLE checkpoint names the Llama architecture of its one decoder layer.
    draft_architectures={
        "eagle": {"LlamaForCausalLM": "LlamaForCausalLMEagle"}
    },
    supports_spec_decode_mixed_batches=True,
)
```

| Source lines | Explanation |
|---|---|
| 29–29 | **Architecture key** — Lookup uses the architecture name. The example repository IDs are examples, while dimensions come from the selected checkpoint. |
| 41–48 | **Implementation and adapters** — The registration supplies the pipeline model, tokenizer, supported encodings, and weight adapters. |

</details>


<a id="code-load"></a>

<details><summary><strong>LlamaModelBase.load_graphs</strong> — 28 complete lines · SOURCE</summary>

Coordinate model settings, graph recipes, compilation, and initialization.

Original: [max/python/max/pipelines/architectures/llama3/model.py](../../../../max/python/max/pipelines/architectures/llama3/model.py#L254); lines 254–281.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/model.py).

```python
    @traced
    def load_graphs(self, session: InferenceSession) -> dict[str, Model]:
        """Compiles and initializes the graphs :meth:`graph_specs` returns.

        Sets up the distributed runtime with ``_init_distributed_runtime``
        first, since the graphs depend on it, and wires the batch processor
        with ``_wire_batch_processor`` after.

        Args:
            session: The session that compiles and initializes the graphs.

        Returns:
            The initialized models, keyed by role.
        """
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        self._init_distributed_runtime(session, model_config)
        with CompilationTimer("model") as timer:
            specs = self.graph_specs(model_config)
            timer.mark_build_complete()
            self.compiled_modules = compile_and_init_graphs(session, specs)
        models = {
            role: compiled.engine_model
            for role, compiled in self.compiled_modules.items()
        }
        self._maybe_release_host_weights(*models.values())
        self._wire_batch_processor(models[GraphRole.DECODER], model_config)
        return models
```

| Source lines | Explanation |
|---|---|
| 268–270 | **Resolve the model** — Read/adapt weights, resolve config, and initialize any distributed runtime required by the selected graph. |
| 272–274 | **Trace, compile, initialize** — The helper called here owns tracing as well as compilation and initialization; a GraphSpec is still a recipe. |
| 279–281 | **Finish pipeline wiring** — After initialized models exist, eligible host weights can be released and the batch processor is wired. |

</details>


<a id="code-choose"></a>

<details><summary><strong>LlamaModelBase.graph_specs</strong> — 11 complete lines · SOURCE</summary>

Choose the decoder implementation from parallelism and device count.

Original: [max/python/max/pipelines/architectures/llama3/model.py](../../../../max/python/max/pipelines/architectures/llama3/model.py#L287); lines 287–297.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/model.py).

```python
    def graph_specs(self, model_config: Any) -> dict[str, GraphSpec]:
        """Returns the decoder: data-parallel when the config asks for it,
        else tensor-parallel across several devices, else single-device."""
        assert isinstance(model_config, Llama3Config)
        if model_config.data_parallel_degree > 1:
            spec = self._data_parallel_spec(model_config)
        elif len(self.devices) > 1:
            spec = self._tensor_parallel_spec(model_config)
        else:
            spec = self._single_device_spec(model_config)
        return {GraphRole.DECODER: spec}
```

| Source lines | Explanation |
|---|---|
| 291–297 | **Ordered selection** — Data parallel takes priority, then multiple-device tensor parallel, then the single-device route used here. |

</details>


### Configuration finalization

`Llama3Config.finalize()` resolves settings that depend on checkpoint metadata:
quantization, normalization, tied embeddings, stacked projections, and output
policy. This mutates a Python configuration object. The symbolic graph has
not been finalized by this call.

<a id="code-config_call"></a>

<details><summary><strong>LlamaModelBase._finalized_config</strong> — 18 complete lines · SOURCE</summary>

Initialize model settings and resolve weight-dependent configuration.

Original: [max/python/max/pipelines/architectures/llama3/model.py](../../../../max/python/max/pipelines/architectures/llama3/model.py#L209); lines 209–226.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/model.py).

```python
    @classmethod
    def _finalized_config(
        cls, context: ModelLoadContext, state_dict: dict[str, Any]
    ) -> Llama3Config:
        config = Llama3Config.initialize(
            context.pipeline_config, max_seq_len=context.max_seq_len
        )
        config.finalize(
            huggingface_config=context.huggingface_config,
            state_dict=state_dict,
            norm_method=cls.norm_method,
            attention_bias=cls.attention_bias,
            return_logits=context.return_logits,
            return_hidden_states=context.return_hidden_states,
        )
        if context.target_layer_ids:
            config.target_layer_ids = list(context.target_layer_ids)
        return config
```

| Source lines | Explanation |
|---|---|
| 213–223 | **Two configuration steps** — Initialize reads pipeline/HF settings; finalize receives checkpoint state plus output and normalization policy. |

</details>


<a id="code-config_finalize"></a>

<details><summary><strong>Llama3Config.finalize</strong> — 78 complete lines · SOURCE</summary>

Resolve quantization, projection layout, tied weights, and output policy.

Original: [max/python/max/pipelines/architectures/llama3/model_config.py](../../../../max/python/max/pipelines/architectures/llama3/model_config.py#L397); lines 397–474.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/model_config.py).

```python
    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",
        attention_bias: bool = False,
    ) -> None:
        """Define parameters that can't be determined just from the pipeline config."""

        # Normalize the LLM state dict so downstream introspection sees canonical
        # Llama-style keys (no "language_model." or "model." prefix). This keeps
        # float8 parsing and feature detection resilient to pack variations.
        def _strip_prefix(s: str, prefix: str) -> str:
            return s.removeprefix(prefix)

        has_lm_prefix = any(k.startswith("language_model.") for k in state_dict)
        has_model_prefix = any(k.startswith("model.") for k in state_dict)

        if has_lm_prefix:
            normalized_state_dict: dict[str, WeightData] = {
                _strip_prefix(k, "language_model."): v
                for k, v in state_dict.items()
                if k.startswith("language_model.")
            }
        elif has_model_prefix:
            normalized_state_dict = {
                _strip_prefix(k, "model."): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }
        else:
            normalized_state_dict = dict(state_dict)

        quant_config = self._parse_quant_config(
            huggingface_config, normalized_state_dict
        )

        # Determine norm_dtype.
        # Note: due to automatic weight dtype casting, norm dtype is not always
        # correct. To avoid any issue, only set norm_dtype for float8 models
        # for now.
        norm_dtype = None
        if "layers.0.input_layernorm.weight" in normalized_state_dict:
            norm_dtype = normalized_state_dict[
                "layers.0.input_layernorm.weight"
            ].dtype

        # When tie_word_embeddings=True, the embedding weights are shared with
        # the output weights.
        if "tie_word_embeddings" in huggingface_config:
            tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            tie_word_embeddings = (
                getattr(huggingface_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in normalized_state_dict
            )

        rms_norm_eps = None
        if norm_method == "rms_norm":
            rms_norm_eps = huggingface_config.rms_norm_eps

        self.norm_method = norm_method
        self.norm_dtype = norm_dtype
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.quant_config = quant_config
        self.stacked_mlp = (
            "layers.0.mlp.gate_up_proj.weight" in normalized_state_dict
        )
        self.stacked_qkv = (
            "layers.0.self_attn.qkv_proj.weight" in normalized_state_dict
        )

        self.attention_bias = attention_bias
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states
```

| Source lines | Explanation |
|---|---|
| 414–430 | **Normalize names for inspection** — Use canonical Llama-style keys when inspecting checkpoint metadata. This local normalized mapping supports subsequent feature detection. |
| 432–444 | **Inspect metadata** — Quantization and normalization information are derived from config and weights before graph modules are constructed. |
| 460–474 | **Commit resolved settings** — The final fields include tied embeddings, stacked MLP/QKV projections, attention bias, and output policy. |

</details>


<details class="self-check"><summary>Check your understanding: does the name Llama3Model hard-code a Llama 3 8B graph?</summary><p>No. The registration selects this implementation, while the configuration supplies dimensions and model options. A specific checkpoint still needs compatible weights and settings; this source route is not a claim of a completed checkpoint run.</p></details>

## 2. Record the computation

**Tracing calls the Python model on symbolic graph inputs.** The model's tensor
operations append IR operations; they do not consume an actual prompt and
generate tokens at this stage. Python construction and control flow still
execute: the configured layer loop expands the decoder structure into the
graph.

`_single_device_spec()` constructs the module, loads its weights, and captures
its input contract. Weight values can exist before compilation. The later
initialization step binds the named registry to the compiled engine model;
these are different meanings of “loading weights.”

<a id="code-single_spec"></a>

<details><summary><strong>LlamaModelBase._single_device_spec</strong> — 9 complete lines · SOURCE</summary>

Package the Llama module, input contract, and weight registry into a recipe.

Original: [max/python/max/pipelines/architectures/llama3/model.py](../../../../max/python/max/pipelines/architectures/llama3/model.py#L316); lines 316–324.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/model.py).

```python
    def _single_device_spec(self, model_config: Llama3Config) -> GraphSpec:
        single_model = Llama3(model_config)
        _load_weights(single_model, self.state_dict)
        return GraphSpec(
            single_model,
            "llama3",
            self._bind_graph_inputs(single_model.input_tree(self.kv_params)),
            weights=single_model.state_dict(),
        )
```

| Source lines | Explanation |
|---|---|
| 317–324 | **Module then recipe** — The module is constructed and weights are loaded before the named input tree and registry are packaged in GraphSpec. |

</details>


### One block connects the model to the graph

`SOURCE` — the following diagram follows the ordinary residual path in
`TransformerBlock.__call__()`. `T` is the packed token count and `H` is the
configured hidden dimension. This is a shape/dataflow illustration.

```text
tokens [T] -> embedding -> hidden states x [T,H]
                               |
                 +-------------+------------------+
                 |                                |
                 v                                |
             RMSNorm -> attention(Q/K/V, RoPE, KV) |
                 |                                |
                 +---------------------- add <----+
                                          |
                                          h [T,H]
                                          |
                         +----------------+----------------+
                         |                                 |
                         v                                 |
                     RMSNorm -> MLP ---------------------- add
                                                           |
                                                     next block

final block -> selected hidden-state rows -> norm -> LM head -> logits
```

The config chooses attention/linear/MLP implementations and the number of
blocks. `TransformerBlock` exposes both residual paths. The full layer loop
and logits postprocessing show how those blocks become a complete decoder.

<a id="code-llama_init"></a>

<details><summary><strong>Llama3.__init__</strong> — 153 complete lines · SOURCE</summary>

Assemble the Llama decoder layers from the resolved configuration.

Original: [max/python/max/pipelines/architectures/llama3/llama3.py](../../../../max/python/max/pipelines/architectures/llama3/llama3.py#L92); lines 92–244.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/llama3.py).

```python
    def __init__(self, config: Llama3Config) -> None:
        assert len(config.devices) == 1
        self.config = config
        rope = create_rope_embedding(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved_rope_weights=config.interleaved_rope_weights,
            rope_scaling_params=config.rope_scaling_params,
            longrope_scaling_params=config.longrope_scaling_params,
            device=config.devices[0],
        )

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.hidden_size,
                config.norm_dtype or config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,  # disable Gemma3-style scaling
            )
        else:
            create_norm = functools.partial(
                ConstantLayerNorm,
                config.hidden_size,
                config.devices[0],
                config.norm_dtype or config.dtype,
            )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinear, quantization_config=config.quantization_config
            )
        else:
            linear_cls = functools.partial(
                Linear, quant_config=config.quant_config
            )
        if config.stacked_mlp and config.quant_config:
            raise ValueError(
                "StackedMLP and scaled quantization are not compatible"
            )
        mlp_cls = (
            StackedMLP
            if config.stacked_mlp
            else functools.partial(MLP, quant_config=config.quant_config)
        )
        attention_cls: Callable[..., AttentionWithRope]
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            assert config.quantization_config is not None
            assert not config.attention_bias, (
                "Attention bias is not supported for GPTQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        elif config.model_quantization_encoding is not None:
            assert not config.attention_bias, (
                "Attention bias is not supported for GGUFQAttentionWithRope."
            )
            attention_cls = functools.partial(
                GGUFQAttentionWithRope,
                quantization_encoding=config.model_quantization_encoding,
                scale=config.attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                AttentionWithRope,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
                has_bias=config.attention_bias,
                quant_config=config.quant_config,
            )

        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_output_dtype = config.quant_config.embedding_output_dtype
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
            quantization_encoding=embedding_output_quantization,
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
            embedding_multiplier=config.embedding_multiplier,
            logits_scaling=config.logits_scaling,
            target_layer_ids=config.target_layer_ids,
        )
```

| Source lines | Explanation |
|---|---|
| 95–104 | **Position encoding** — RoPE configuration includes head count, hidden size, sequence limit, and the target device. |
| 128–175 | **Encoding-specific modules** — Linear, attention, and MLP implementations are chosen from quantization and stacked-projection settings. |
| 177–202 | **Repeated decoder blocks** — Each block receives attention, MLP, normalization, and KV parameters. The layer count comes from config. |
| 204–228 | **Input/output projections** — Embedding and language-model head dimensions follow config. The output weight can be shared with the embedding. |

</details>


<a id="code-block"></a>

<details><summary><strong>TransformerBlock.__call__</strong> — 28 complete lines · SOURCE</summary>

Record the attention and MLP residual paths for one decoder block.

Original: [max/python/max/nn/transformer/transformer.py](../../../../max/python/max/nn/transformer/transformer.py#L293); lines 293–320.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/nn/transformer/transformer.py).

```python
    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        residual_multiplier = ops.constant(
            self.residual_multiplier, x.dtype, device=x.device
        )
        attn_out = self.self_attn(
            layer_idx,
            self.input_layernorm(x),
            kv_collection,
            freqs_cis=freqs_cis,
            input_row_offsets=input_row_offsets,
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out
        mlp = self.mlp(self.post_attention_layernorm(h))
        if self.residual_multiplier != 1.0:
            mlp = mlp * residual_multiplier

        return h + mlp
```

| Source lines | Explanation |
|---|---|
| 304–310 | **Attention path** — Normalize the incoming hidden states and call attention with layer index, KV collection, RoPE data, and sequence boundaries. |
| 315–320 | **Two residual additions** — The attention output joins x; the normalized result goes through MLP and joins h. Optional multipliers are explicit branches. |

</details>


<a id="code-transform_layers"></a>

<details><summary><strong>Transformer._process_hidden_states</strong> — 39 complete lines · SOURCE</summary>

Trace the configured layer sequence, then process logits.

Original: [max/python/max/nn/transformer/transformer.py](../../../../max/python/max/nn/transformer/transformer.py#L617); lines 617–655.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/nn/transformer/transformer.py).

```python
    def _process_hidden_states(
        self,
        h: TensorValue,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> DecoderOutputs:
        freqs_cis = self.rope.freqs_cis

        capture_layer_set: set[int] | None = None
        capture_hidden_states: list[list[TensorValue]] | None = None
        if (
            self.target_layer_ids is not None
            and self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
        ):
            capture_layer_set = set(self.target_layer_ids)
            capture_hidden_states = []

        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
            if (
                capture_layer_set is not None
                and capture_hidden_states is not None
                and idx in capture_layer_set
            ):
                capture_hidden_states.append([h])

        return self._postprocess_logits(
            h,
            input_row_offsets,
            return_n_logits,
            capture_hidden_states=capture_hidden_states,
        )
```

| Source lines | Explanation |
|---|---|
| 635–642 | **Python expands the layer structure** — Each call uses symbolic tensors while tracing. Layer index constants and KV/row-offset inputs are attached to the recorded computation. |
| 650–655 | **Decoder output policy** — The final hidden states are passed to the common logits/output-tree construction path. |

</details>


<a id="code-logits"></a>

<details><summary><strong>logits_postprocess</strong> — 85 complete lines · SOURCE</summary>

Select output positions, normalize them, and project to vocabulary logits.

Original: [max/python/max/nn/transformer/transformer.py](../../../../max/python/max/nn/transformer/transformer.py#L413); lines 413–497.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/nn/transformer/transformer.py).

```python
def logits_postprocess(
    h: TensorValue,
    input_row_offsets: TensorValue,
    return_n_logits: TensorValue,
    norm: Callable[[TensorValue], TensorValue],
    lm_head: Callable[[TensorValue], TensorValue],
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
    logits_scaling: float = 1.0,
    capture_hidden_states: list[list[TensorValue]] | None = None,
) -> DecoderOutputs:
    """Performs common logits postprocessing for single-device models.

    Handles last-token gathering, logits computation, optional logits
    scaling, and hidden states return.

    Args:
        h: The hidden states from the final transformer layer.
        input_row_offsets: The row offsets for ragged batching.
        return_n_logits: The number of logits to return per sequence.
        norm: The normalization function applied before the language model head.
        lm_head: The language model head projection.
        return_logits: Which logits to return.
        return_hidden_states: Which hidden states to return. Defaults to
            ``ReturnHiddenStates.NONE``.
        logits_scaling: A divisor applied to logits after projection. Logits
            are divided by this value before returning. Defaults to ``1.0``
            (no scaling).
        capture_hidden_states: For ``SELECTED_LAYERS`` mode, the
            per-layer captured hidden states gathered during the layer loop.
            See :func:`extract_hs`.

    Returns:
        The named outputs; see :func:`logits_outputs`.
    """
    last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
    last_logits = ops.cast(lm_head(norm(last_h)), DType.float32)
    logits = None
    offsets = None

    if return_logits == ReturnLogits.VARIABLE:
        return_n_logits_range = ops.range(
            return_n_logits[0],
            0,
            -1,
            out_dim="return_n_logits_range",
            device=h.device,
            dtype=DType.int64,
        )
        offsets = (
            ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
        )
        last_indices = ops.reshape(offsets, shape=(-1,))
        last_tokens = ops.gather(h, last_indices, axis=0)
        logits = ops.cast(lm_head(norm(last_tokens)), DType.float32)
        offsets = ops.range(
            0,
            TensorValue(last_indices.shape[0]) + return_n_logits[0],
            return_n_logits[0],
            out_dim="logit_offsets",
            device=h.device,
            dtype=DType.int64,
        )
    elif return_logits == ReturnLogits.ALL:
        logits = ops.cast(lm_head(norm(h)), DType.float32)
        offsets = input_row_offsets

    if logits_scaling != 1.0:
        last_logits = last_logits / logits_scaling
        if logits is not None:
            logits = logits / logits_scaling

    return logits_outputs(
        last_logits,
        logits,
        offsets,
        return_hidden_states,
        extract_hs(
            return_hidden_states=return_hidden_states,
            last_token_hs_distributed=[last_h],
            all_hs_distributed=[h],
            normalizer=[norm],
            capture_hidden_states=capture_hidden_states,
        ),
    )
```

| Source lines | Explanation |
|---|---|
| 448–451 | **Last-token positions** — Row offsets identify each sequence's final token. The model gathers those rows before normalization and the language-model head. |
| 453–481 | **Static output-policy branches** — The configured policy selects extra variable-count or all-token logits. return_n_logits supplies values used by the variable path. |
| 485–490 | **Named outputs** — Logits, offsets, and optional hidden states form the output tree later flattened by Graph.output. |

</details>


### Symbolic inputs describe a call boundary

| Input | Declared type / placement | Meaning |
|---|---|---|
| `tokens` | `int64[total_seq_len]`, selected device | All tokens packed into one sequence buffer |
| `row_offsets` | `uint32[input_row_offsets_len]`, selected device | Boundaries between sequences; conventionally `B+1` entries for `B` sequences |
| `return_n_logits` | `int64[return_n_logits]`, CPU | Count input used by the configured variable-logits path |
| KV subtree | Types from `kv_params.get_symbolic_inputs()` | Cache buffers and metadata consumed by attention |

Those are symbolic type declarations. Actual buffer values, their concrete
lengths, and the cache contents arrive during execution. The graph's fixed
architecture dimensions and its variable batch/token dimensions serve
different purposes.

<a id="code-ragged"></a>

<details><summary><strong>ragged_batch_inputs</strong> — 61 complete lines · SOURCE</summary>

Declare symbolic batch tensor types and placements.

Original: [max/python/max/pipelines/lib/interfaces/batch_processor.py](../../../../max/python/max/pipelines/lib/interfaces/batch_processor.py#L539); lines 539–599.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/lib/interfaces/batch_processor.py).

```python
def ragged_batch_inputs(
    device_ref: DeviceRef,
    fields: Sequence[str] = ("tokens", "row_offsets", "return_n_logits"),
    *,
    data_parallel_degree: int = 1,
    hidden_states: tuple[DType, int] | None = None,
    row_offsets_devices: Sequence[DeviceRef] | None = None,
) -> RaggedBatch:
    """Returns the named per-batch graph inputs, in the order ``fields`` gives.

    Args:
        device_ref: The device that holds the token and row-offset tensors.
        fields: The inputs to include, in graph order. Any of ``tokens``,
            ``row_offsets``, ``host_row_offsets``, ``return_n_logits``,
            ``data_parallel_splits`` and ``hidden_states``.
        data_parallel_degree: The number of replicas, which sizes
            ``data_parallel_splits``.
        hidden_states: The ``(dtype, hidden_size)`` of the EAGLE draft's
            hidden-state input. Required when ``fields`` names
            ``hidden_states``.
        row_offsets_devices: The devices that each take their own
            ``row_offsets``, for a graph that does not broadcast them.
            ``row_offsets`` is then a tuple with one type per device.

    Returns:
        The input types keyed by field name.
    """
    available: dict[str, TensorType | tuple[TensorType, ...]] = {
        "tokens": TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        ),
        "row_offsets": TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        ),
        "host_row_offsets": TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        ),
        "return_n_logits": TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        ),
        "data_parallel_splits": TensorType(
            DType.int64,
            shape=[data_parallel_degree + 1],
            device=DeviceRef.CPU(),
        ),
    }
    if row_offsets_devices is not None:
        available["row_offsets"] = tuple(
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            )
            for device in row_offsets_devices
        )
    if hidden_states is not None:
        dtype, hidden_size = hidden_states
        available["hidden_states"] = TensorType(
            dtype, shape=["total_seq_len", hidden_size], device=device_ref
        )
    return RaggedBatch((name, available[name]) for name in fields)
```

| Source lines | Explanation |
|---|---|
| 567–572 | **Packed tokens and boundaries** — Tokens are int64; row offsets are uint32. Both use the selected device, with symbolic lengths. |
| 578–580 | **Host-side count input** — This count tensor is int64 on CPU; its symbolic length is declared independently from the token count. |

</details>


### Follow the trace into Graph

```text
GraphSpec.trace()
  -> flatten named input types
  -> Llama3.trace()
     -> define forward adapter for flat inputs and KV reconstruction
     -> Graph(name, forward, input_types=...)
        -> create MLIR module and mo.graph input arguments
        -> forward(*graph.inputs)
        -> Graph.output(returned_values)
```

`Llama3.trace()` reconstructs the nested KV collection from flat graph inputs.
`Graph.__init__()` creates the graph container and its initially empty result
list. Passing `forward` causes automatic tracing and output finalization.
The graph body can initially contain RMO operations; early compiler processing
legalizes them. The container name `mo.graph` does not mean every initial
operation belongs to the MO dialect.

<a id="code-spec_trace"></a>

<details><summary><strong>GraphSpec.trace</strong> — 20 complete lines · SOURCE</summary>

Flatten the input-type tree and dispatch to the module's trace method.

Original: [max/python/max/pipelines/lib/graph_specs.py](../../../../max/python/max/pipelines/lib/graph_specs.py#L114); lines 114–133.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/lib/graph_specs.py).

```python
    def trace(self, module: GraphModule | None = None) -> Graph:
        """Traces the graph, into ``module`` if given.

        Raises:
            ValueError: If ``module`` is given for a ModuleV3 graph.
        """
        if isinstance(self.module, ModuleV3):
            if module is not None:
                raise ValueError(
                    f"ModuleV3 graph {self.name!r} cannot trace into a shared"
                    " module; give its spec no group"
                )
            return self._staged[0].graph
        return self.module.trace(
            *tree.leaves(self.input_types),
            name=self.name,
            module=module,
            custom_extensions=self.custom_extensions,
            is_device_graph=self.is_device_graph,
        )
```

| Source lines | Explanation |
|---|---|
| 120–127 | **Variant boundary** — The ModuleV3 branch stages differently. The Graph API route continues to self.module.trace below. |
| 127–133 | **Preserve recipe settings** — The graph name, module container, extensions, and device-graph flag are passed with the flattened type leaves. |

</details>


<a id="code-llama_trace"></a>

<details><summary><strong>Llama3.trace</strong> — 53 complete lines · SOURCE</summary>

Adapt flat graph inputs back into the Llama forward contract.

Original: [max/python/max/pipelines/architectures/llama3/llama3.py](../../../../max/python/max/pipelines/architectures/llama3/llama3.py#L273); lines 273–325.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/llama3.py).

```python
    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_tree`'s leaves.

        Adapts the flat graph inputs to :meth:`__call__`: unflattens the KV
        cache inputs into the device's KV collection.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree` without a hidden state input.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """

        def forward(
            tokens: Value[Any],
            input_row_offsets: Value[Any],
            return_n_logits: Value[Any],
            *kv_inputs: Value[Any],
        ) -> DecoderOutputs:
            kv_collections = tree.leaves(
                self.kv_params.unflatten_kv_inputs(iter(kv_inputs)),
                leaf=KVCacheInputsPerDevice,
            )
            return self(
                tokens.tensor,
                kv_collections[0],
                return_n_logits.tensor,
                input_row_offsets.tensor,
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )
```

| Source lines | Explanation |
|---|---|
| 307–316 | **Recover KV structure** — The graph boundary is flat; this adapter reconstructs the per-device KV collection required by the model. |
| 318–325 | **Trace through the constructor** — Passing forward to Graph causes the constructor to invoke it on symbolic graph inputs and finalize its returned outputs. |

</details>


<a id="code-graph_init"></a>

<details><summary><strong>Graph.__init__</strong> — 114 complete lines · SOURCE</summary>

Create MLIR graph structure and optionally trace a Python callable.

Original: [max/python/max/graph/graph.py](../../../../max/python/max/graph/graph.py#L819); lines 819–932.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/graph/graph.py).

```python
    @in_default_mlir_context
    def __init__(
        self,
        name: str,
        forward: Callable[..., GraphOutputs | None] | None = None,
        input_types: Iterable[Type[Any]] = (),
        path: Path | None = None,
        *args,
        custom_extensions: Iterable[Path] = [],
        kernel_library: KernelLibrary | None = None,
        module: Module | None = None,
        strict_device_placement: DevicePlacementPolicy = DevicePlacementPolicy.Warn,
        is_device_graph: bool = False,
        **kwargs,
    ) -> None:
        self.name = name
        self.strict_device_placement = strict_device_placement
        if path is not None:
            self._load_mlir(path)
            return

        self._params = dict.fromkeys(
            dim.name
            for t in input_types
            if isinstance(t, TensorType | BufferType)
            for dim in t.shape
            if isinstance(dim, SymbolicDim)
        )
        self._context_state = []
        self._data_dependent_dims = 0
        self._should_verify_ops = True

        with _location() as loc:
            # The top-level module op is stored as a typed nanobind
            # :class:`builtin.ModuleOp`. Either reuse the one supplied via
            # ``module=`` or create a fresh one.
            self._module = (
                module.mlir_module if module else builtin.ModuleOp(location=loc)
            )
            builder = OpBuilder(self._module.body.end)

            op = _mo.GraphOp(
                builder,
                loc,
                name=name,
                input_types=[t.to_mlir() for t in input_types],
                result_types=[],
            )

            si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True))
            # TODO(MAXPLAT-306): Type annotations are wrong here
            op.input_parameters = kgen.ParamDeclArrayAttr(
                [kgen.ParamDeclAttr(p, si64) for p in self._params]
            )

            self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr)
            self._current_block = self._mlir_op.regions[0].blocks[0]
            self._graph_body = self._current_block
            self._populate_device_info_mapping()

        self._weights = {}
        self._has_chain_input = False
        self.device_chains = _DeviceChainMap(self)

        # Create an always-ready chain that is never advanced by the graph.
        # Use it for operations that are safe to schedule without per-device
        # ordering constraints (e.g., host→device transfers for staging).
        self._always_ready_chain = _ChainValue(
            self._add_op_generated(
                _mo.ChainCreateOp,
                result=_mo.ChainType(),
                inputs=[],
                attach_profile_scopes=False,
            )[0]
        )
        self._update_chain(self._always_ready_chain)

        if self._graph_body.arguments:
            mlir_maybe_chain_value = _Value._from_cmlir(
                self._graph_body.arguments[-1]
            )
            if _is_chain_value(mlir_maybe_chain_value):
                self._has_chain_input = True
                self._update_chain(
                    _ChainValue.from_mlir(mlir_maybe_chain_value)
                )

        # Resolving through `_resolved_custom_extensions` is what lets a graph
        # built without `custom_extensions` still reach a backend's kernel
        # overlays (see `default_custom_extensions`).
        self._kernel_library = kernel_library or KernelLibrary()
        self._import_kernels(_resolved_custom_extensions(custom_extensions))

        self._subgraphs = {}

        # If we're building a device graph, annotate the graph appropriately.
        if is_device_graph:
            op.is_device_graph = builtin.UnitAttr()

        if forward is not None:
            # If the forward method was passed stage the graph directly in the
            # constructor.
            with self:
                result = forward(*self.inputs, *args, **kwargs)
                # Forward returns None, one value, a tree such as a tuple or a
                # named dict, or any other iterable of values.
                if result is None:
                    self.output()
                elif tree.is_node(result):
                    self.output(result)
                elif isinstance(result, Iterable):
                    self.output(*result)
                else:
                    self.output(result)
```

| Source lines | Explanation |
|---|---|
| 840–846 | **Symbolic dimensions** — Names from symbolic tensor/buffer dimensions become graph parameters. |
| 855–866 | **MLIR ownership** — Use a supplied module or create one, then create mo.graph with input types and initially empty result types. |
| 910–916 | **Kernel declarations and graph mode** — Import the kernel library and mark the graph if explicit device-graph construction was requested. |
| 918–932 | **Automatic finalization** — Run the callable inside the active graph context. Its return form determines how output is invoked. |

</details>


<details class="self-check"><summary>Check your understanding: what is executing when the Python layer loop runs during tracing?</summary><p>Python is constructing the symbolic computation. Graph tensor operations record dataflow on symbolic values. Runtime token buffers are supplied to the initialized engine model later. Some host-side setup work can occur during construction; tracing is not equivalent to running Llama inference.</p></details>

## 3. Finalize the graph

**`Graph.output()` completes the graph's output contract and verifies its IR.**
The Llama path reaches it automatically from the `Graph` constructor after
the symbolic forward returns. There is no separate Python `Graph.finalize()`
call on this route.

### Three different objects can be “finalized”

| Name | Object | Meaning |
|---|---|---|
| `Llama3Config.finalize()` | Python model configuration | Resolve checkpoint-dependent settings before tracing |
| `Graph.output()` | Symbolic graph | Install output types, signature, terminator, and names; verify graph IR |
| `FinalizeDeviceGraphsPass` | Marked MGP device-graph model | Finish device-graph collective construction during native compilation |

Select each step below, then open its matching implementation. The state
panels describe the source logic; they are not captured IR from a model run.

```text
Forward outputs -> flatten tree -> normalize Values + conditional chains
                -> functionType + generator signature
                -> mo.output terminator
                -> argument_names + result_names
                -> verify completed graph

Graph.__exit__ restores context. Native compilation is a subsequent call.
```

`SOURCE` — conceptual illustration; not an executed graph dump.


<a id="code-graph_output"></a>

<details><summary><strong>Graph.output</strong> — 103 complete lines · SOURCE</summary>

Finish the graph's output contract and verify the completed graph.

Original: [max/python/max/graph/graph.py](../../../../max/python/max/graph/graph.py#L1396); lines 1396–1498.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/graph/graph.py).

```python
    def output(self, *outputs: GraphOutputs) -> None:
        """Sets the output values of the graph and finalizes construction.

        Call this once after building all ops. The graph can't be executed
        until ``output()`` has been called. Subsequent calls to
        :attr:`output_types` read back the types of the values passed here.

        For example, build a graph that doubles its input and set the
        output:

        .. code-block:: python

            from max.dtype import DType
            from max.graph import DeviceRef, Graph, ops
            from max.graph.type import TensorType

            input_type = TensorType(DType.float32, [4], DeviceRef.CPU())

            with Graph("double", input_types=[input_type]) as graph:
                x = graph.inputs[0].tensor
                two = ops.constant(2.0, DType.float32, device=DeviceRef.CPU())
                graph.output(ops.elementwise.mul(x, two))

        Pass one named tree, such as a dict of logits and hidden states, to
        name each output by its path in it; :attr:`output_tree` then returns
        the output types in that structure.

        Args:
            outputs: The output values of the graph, or a single tree whose
                leaves they are. Each value may be a :class:`Value` or any
                :class:`~max.graph.TensorValueLike`.
        """
        root = (
            outputs[0]
            if len(outputs) == 1 and tree.is_node(outputs[0])
            else outputs
        )
        leaves, self._output_structure = tree.flatten(root)
        if any(isinstance(leaf, str) for leaf in leaves):
            raise TypeError(
                f"graph outputs {leaves} include a string, as when a named"
                " tree is unpacked into its keys: pass the tree itself, as"
                " graph.output(outputs)"
            )
        outputs = tuple(
            o if isinstance(o, Value) else TensorValue(o) for o in leaves
        )
        outputs = cast(tuple[Value[Any], ...], outputs)
        # mo.output doesn't support infer_type
        graph_body_args = self._graph_body.arguments
        output_values: list[Value[Any]] = list(outputs)
        if self._has_chain_input:
            output_values = self.device_chains.pack(output_values)
        mlir_values: list[_Value[Any]] = [v._mlir_value for v in output_values]

        # We have a type mismatch now, these are MLIR types
        # Convert from max._core.Type to mlir.Type using CAPI bridge
        output_types = [
            mlir.Type._CAPICreate(value.type._CAPIPtr)  # type: ignore[attr-defined]
            for value in mlir_values
        ]

        # Need to set some more stuff.
        function_type = mlir.FunctionType.get(
            graph_body_args.types,
            output_types,
        )
        signature = mlir.Type.parse(f"!kgen.generator<{function_type}>")
        self._mlir_op.attributes["signature"] = mlir.TypeAttr.get(signature)
        self._mlir_op.attributes["functionType"] = mlir.TypeAttr.get(
            function_type
        )

        self._add_op_generated(
            _mo.OutputOp,
            mlir_values,
            _kgen.ParameterExprArrayAttr([]),
            attach_profile_scopes=False,
        )

        # Set the result_names metadata on the staged op, which is needed by
        # the engine for execution.
        # Note that result_names here needs to match kMgpModelResultNames.
        input_names = [f'"input{i}"' for i in range(len(self.inputs))]
        self._mlir_op.attributes["argument_names"] = mlir.Attribute.parse(
            f"[{', '.join(input_names)}]"
        )
        output_names = [f'"output{i}"' for i in range(len(output_types))]
        self._mlir_op.attributes["result_names"] = mlir.Attribute.parse(
            f"[{', '.join(output_names)}]"
        )

        self._subgraphs = {}
        # Outputting means the graph is complete. Verify the entire graph.
        try:
            with self._capturing_mlir_diagnostics():
                assert self._mlir_op.verify()
        except Exception as e:
            print(self)
            raise ValueError(
                "Graph failed to verify. Please file an issue. This should be"
                " impossible." + f"\n{e}"
            ) from None
```

| Source lines | Explanation |
|---|---|
| 1428–1439 | **1 · Flatten outputs** — Retain the Python output structure; flatten its leaves. A string leaf is rejected because unpacking a dict into its keys is usually a mistake. |
| 1440–1449 | **2 · Normalize values and chains** — Convert leaves to Graph Values and include final chain values when the graph has chain inputs. Chains preserve dependency ordering. |
| 1459–1467 | **3 · Set both signatures** — The actual argument and result types define functionType and the kgen generator signature. Symbolic dimensions can remain symbolic. |
| 1469–1474 | **4 · Append the terminator** — mo.output identifies which SSA values leave the graph. This records IR; it does not launch inference. |
| 1479–1486 | **5 · Name the interface** — Ordinal input/result names are written for the engine. The Python output tree is retained separately. |
| 1488–1498 | **6 · Verify the graph** — Clear temporary bookkeeping and invoke MLIR graph verification. Failure reports diagnostics. Numerical correctness and target compilation remain separate checks. |

</details>


### What the completed contract provides

- **Output structure:** Python retains a tree describing semantic groups such
  as logits and hidden states, while the engine boundary uses flat values.
- **Types:** `functionType` and `signature` describe actual graph argument and
  result types. Symbolic dimensions are allowed to remain symbolic.
- **Termination:** `mo.output` identifies the SSA values returned by the graph.
- **Names:** engine metadata uses ordinal names such as `input0` and `output0`.
- **Verification:** `_mlir_op.verify()` checks the completed graph. Compiler
  legality, weight-buffer validation, and numerical correctness require
  their corresponding later checks.

### Context exit has a separate job

The `with graph:` context chooses where operations are recorded.
`Graph.__exit__()` unwinds that context through `_enter()` and restores the
previous `CURRENT_GRAPH`. Compilation starts through a later session call.

<a id="code-graph_exit"></a>

<details><summary><strong>Graph.__exit__</strong> — 2 complete lines · SOURCE</summary>

Exit the active graph-construction context.

Original: [max/python/max/graph/graph.py](../../../../max/python/max/graph/graph.py#L1138); lines 1138–1139.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/graph/graph.py).

```python
    def __exit__(self, *exc) -> None:
        self._context_state.pop().__exit__(*exc)
```

| Source lines | Explanation |
|---|---|
| 1139–1139 | **Context restoration** — This delegates context exit; output signature construction and native compilation occur elsewhere. |

</details>


<a id="code-graph_enter"></a>

<details><summary><strong>Graph._enter</strong> — 9 complete lines · SOURCE</summary>

Set and restore the active graph while Python operations are recorded.

Original: [max/python/max/graph/graph.py](../../../../max/python/max/graph/graph.py#L1141); lines 1141–1149.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/graph/graph.py).

```python
    @contextlib.contextmanager
    def _enter(self) -> Generator[Graph]:
        # Body of ``with graph:`` creates MLIR ops; ensure context on bg threads.
        with ensure_default_mlir_context():
            token = CURRENT_GRAPH.set(self)
            try:
                yield self
            finally:
                CURRENT_GRAPH.reset(token)
```

| Source lines | Explanation |
|---|---|
| 1145–1149 | **Scoped graph state** — The finally block restores the previous graph even when construction raises an exception. |

</details>


### A small, complete example from the repository

The existing test below exposes finalization without Llama's many layers:
two symbolic inputs → add → output → metadata checks. **`TEST` means the test
definition was inspected; it was not executed in this session.**

<a id="code-graph_test"></a>

<details><summary><strong>test_add_op</strong> — 13 complete lines · TEST</summary>

Read a complete existing test that builds an add graph and checks interface metadata.

Original: [max/tests/tests/graph/test_graph.py](../../../../max/tests/tests/graph/test_graph.py#L135); lines 135–147.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/tests/tests/graph/test_graph.py).

```python
def test_add_op() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    input_type = TensorType(
        dtype=DType.float32, shape=["batch", "channels"], device=DeviceRef.CPU()
    )
    with Graph("add", input_types=(input_type, input_type)) as graph:
        lhs, rhs = graph.inputs
        elemwise_sum = ops.add(lhs, rhs)
        graph.output(elemwise_sum)

        # Check that the arg/result name attributes were added.
        assert "argument_names = " in str(graph._mlir_op)
        assert "result_names = " in str(graph._mlir_op)
```

| Source lines | Explanation |
|---|---|
| 143–147 | **Finalization in a small graph** — This explicit output call establishes the result contract; the following assertions inspect its names. |

</details>


<details class="self-check"><summary>Check your understanding: if Graph.output() succeeds, can that Graph already execute a prompt?</summary><p>Its IR contract has been completed and verified. The serving path must still compile or resolve an artifact, initialize it with weights and runtime state, and call the resulting engine Model with concrete buffers.</p></details>

## 4. Cross into the compiler

**The compiled object is produced by the session, after the graph is traced.**
`compile_and_init_graphs()` makes the order explicit: trace → compile or
reuse → gather named weights → initialize. It also retains the input/output
tree information needed by the serving wrapper.

<a id="code-compile_graphs"></a>

<details><summary><strong>compile_and_init_graphs</strong> — 105 complete lines · SOURCE</summary>

Trace graph recipes, compile artifacts, then initialize their weights.

Original: [max/python/max/pipelines/lib/graph_specs.py](../../../../max/python/max/pipelines/lib/graph_specs.py#L183); lines 183–287.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/lib/graph_specs.py).

```python
def compile_and_init_graphs(
    session: InferenceSession, specs: Mapping[str, GraphSpec]
) -> dict[str, CompiledCallable[..., Any]]:
    """Compiles and initializes each role's graph.

    Roles that share a group trace into one :class:`~max.graph.Module`,
    compile once, and initialize once with the weights of all of them; each
    role's callable then holds the same compiled model. Each compiles through
    :meth:`~max.engine.InferenceSession.compile_reusing_mefs`, so an ungrouped
    role reuses the session's precompiled or exported MEFs. A group compiles
    every time, as any module of several graphs does. Inside
    :func:`compile_roles_separately`, groups are ignored.

    Each callable is called with one argument per input type, positionally,
    or with the named input tree as its one argument, and returns the graph's
    outputs in the tree its ``forward`` returned. It keeps no weight registry,
    so the host weights can be released after initialization. A ModuleV3
    role's callable instead takes and returns what the module's ``forward``
    does, adds the signal buffers a multi-device graph takes, and keeps its
    weight registry.

    Args:
        session: The session that compiles and initializes the graphs.
        specs: The graphs to compile, keyed by role.

    Returns:
        Each role's compiled, initialized graph.

    Raises:
        ValueError: If two roles in one group share a graph name, or bind
            different values to one weight name.
    """
    separately = _compile_roles_separately.get()
    groups: dict[str, list[str]] = {}
    for role, spec in specs.items():
        # Ungrouped roles compile alone, so each gets its own key.
        key = (
            spec.group
            if spec.group is not None and not separately
            else f"\0{role}"
        )
        groups.setdefault(key, []).append(role)

    callables: dict[str, CompiledCallable[..., Any]] = {}
    for members in groups.values():
        names = [specs[role].name for role in members]
        if len(set(names)) != len(names):
            raise ValueError(
                f"roles {members} share a group but not distinct graph names:"
                f" {names}"
            )
        traced_graph: Graph | GraphModule
        if len(members) == 1:
            graphs = {members[0]: specs[members[0]].trace()}
            traced_graph = graphs[members[0]]
            extensions = None
        else:
            traced_graph = GraphModule()
            graphs = {role: specs[role].trace(traced_graph) for role in members}
            # A graph carries its own extensions; a module of graphs does not.
            extensions = [
                path
                for role in members
                for path in specs[role].custom_extensions
            ]
        output_trees = {
            role: graph.output_tree for role, graph in graphs.items()
        }
        # Not session.load_all, which drops the compiled model a callable
        # keeps to export its MEF.
        compiled = session.compile_reusing_mefs(
            traced_graph, custom_extensions=extensions
        )

        weights: dict[str, DLPackArray] = {}
        for role in members:
            for name, value in specs[role].weights_registry().items():
                if weights.setdefault(name, value) is not value:
                    raise ValueError(
                        f"weight {name!r} has two different values"
                    )
        by_name = session.init_all(compiled, weights_registry=weights)
        for role in members:
            if len(by_name) == 1:
                # The session can hand back a cached MEF compiled from an
                # identical graph under another name, so don't look it up by
                # this spec's name.
                (model,) = by_name.values()
            else:
                model = by_name[specs[role].name]
            spec = specs[role]
            if isinstance(spec.module, ModuleV3):
                # The trace's own signature, so the callable takes and
                # returns what ``forward`` does and adds any signal buffers.
                staged, registry = spec._staged
                callables[role] = CompiledCallable.from_engine_model(
                    staged, compiled, model, registry
                )
            else:
                callables[role] = _callable(
                    compiled, model, spec.input_types, output_trees[role]
                )
                _check_inputs(role, callables[role].in_tree)
                _check_outputs(role, callables[role].out_tree)
    return callables
```

| Source lines | Explanation |
|---|---|
| 235–247 | **Trace each compilation group** — The ordinary Llama decoder has one spec. Multiple grouped roles can instead share a graph Module. |
| 253–255 | **Compilation boundary** — The completed graph or module is compiled or resolved through configured artifact reuse. |
| 257–264 | **Bind after compilation** — Collect weights by name, reject conflicting values, and initialize all models in the compiled artifact. |

</details>


### Follow the payload, not just the function names

| Boundary | Payload | Result |
|---|---|---|
| Pipeline → Python session | Completed Graph/Module, extensions | `CompiledModel` handle |
| Python → C++ session | MLIR module capsule, pipeline name, options | Native asynchronous compilation result |
| C++ session → engine | `CompilationRequest` | CPU worker scheduled to process it |
| Engine → frontend | Source pointer, format, settings | Compiled MEF buffer or cached MEF |
| Frontend → compiler | Cloned MLIR module and pipeline options | Lowered module, then MEF serialization |
| MEF reader → session | Parsed compiled-model objects | Artifact available for initialization |

```text
Pipeline      Python session    C++ session    Engine worker   CompiledModels   MOF frontend   Compiler
   |                |                |              |                |               |             |
   | compile(graph) |                |              |                |               |             |
   |--------------->| capsule        |              |                |               |             |
   |                |--------------->| request      |                |               |             |
   |                |                |------------->| create         |               |             |
   |                |                |              |--------------->| compile       |             |
   |                |                |              |                |-------------->| run passes  |
   |                |                |              |                |               |------------>|
   |                |                |              |                |               |<------------|
   |                |                |              |                |<--------------| MEF buffer  |
   |                |                |              |<---------------| read artifact |             |
   |                |<----------------------------------------------- compiled result              |
   |<---------------| wait complete  |              |                |               |             |
   | init_all(weights)              |              |                |               |             |
   |--------------->| native initialization -> executable models    |               |             |

GraphSpec.trace / Graph.output finish before this cold compilation sequence.
The engine schedules the compile on a CPU worker; arrows show call/result order.
```


The native implementations are present in this checkout. The Python/C++
crossing is a visible language boundary, not an unexplained stop in the
source trail.

### Python session: dispatch, handoff, and wait

`compile_reusing_mefs()` may resolve a stored artifact. Otherwise
`compile_async()` selects a path or in-memory graph. For a Graph, it collects
expected weight-device metadata and seeds kernel declarations before
`_compile_module()` passes the MLIR capsule into C++.

The process-wide compilation lock spans the asynchronous compile. A completion
callback releases it. The synchronous `compile()` waits on the result and
surfaces errors before returning the artifact.

<a id="code-compile_reuse"></a>

<details><summary><strong>InferenceSession.compile_reusing_mefs</strong> — 71 complete lines · SOURCE</summary>

Select normal compilation, configured MEF import, or MEF export.

Original: [max/python/max/engine/api.py](../../../../max/python/max/engine/api.py#L897); lines 897–967.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/engine/api.py).

```python
    def compile_reusing_mefs(
        self,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
        tile_based_fusion: bool = False,
    ) -> CompiledModel:
        """Compiles ``model``, or reuses an artifact if the session has a store.

        The compiling half of :meth:`load_all`, split out for callers that trace
        a graph and initialize it themselves rather than handing it over -- a
        :class:`~max.experimental.nn.Module`, for one, which keeps the compiled
        artifact and the input/output plumbing it derived alongside it. Without
        this they would call :meth:`compile` and silently bypass the session's
        ``precompiled_mefs``/``export_mefs``.

        Behaves exactly like :meth:`compile` on a session with no store at all,
        and only :class:`~max.graph.Graph` models participate: the store
        identifies an artifact by a graph's name and signature. A session
        constructed with neither consults the process-wide store installed with
        :attr:`default_mef_store`, if one is installed; an explicit
        ``precompiled_mefs``/``export_mefs`` always wins.

        Args:
            model: As :meth:`compile`.
            custom_extensions: As :meth:`compile`. Ignored when reusing an
                artifact, which is already compiled.
            tile_based_fusion: As :meth:`compile`. Ignored when reusing an
                artifact.

        Returns:
            The compiled artifact, ready for :meth:`init` or :meth:`init_all`.

        Raises:
            RuntimeError: If reusing and no recorded artifact matches
                ``model``'s name and signature.
        """
        # See `_precompiled_mefs` for why a caller would want this: it lets the
        # compile happen somewhere that holds no accelerator.
        store = (
            self._mef_store
            if self._mef_store is not None
            else InferenceSession.default_mef_store
        )
        if store is None or not isinstance(model, Graph):
            return self.compile(
                model,
                custom_extensions=custom_extensions,
                tile_based_fusion=tile_based_fusion,
            )

        if not store.exporting:
            artifact = store.claim_import(model)
            # A store with nothing to say about this graph declines it, and
            # the compile happens here as it would with no store at all.
            if artifact is not None:
                return self.compile(artifact)
            return self.compile(
                model,
                custom_extensions=custom_extensions,
                tile_based_fusion=tile_based_fusion,
            )

        compiled = self.compile(
            model,
            custom_extensions=custom_extensions,
            tile_based_fusion=tile_based_fusion,
        )
        compiled.export_mef(store.claim_export(model))
        store.write_manifest()
        return compiled
```

| Source lines | Explanation |
|---|---|
| 936–948 | **Choose the store** — Use the explicit session store if present, otherwise the default. Non-Graph modules take the regular compile route. |
| 948–958 | **Import or decline** — A matching artifact path is passed to compile. A store that declines this graph falls back to compiling the graph. |
| 965–965 | **Export mode** — Write the compiled artifact under the store's claimed export location. |

</details>


<a id="code-compile_async"></a>

<details><summary><strong>InferenceSession.compile_async</strong> — 92 complete lines · SOURCE</summary>

Prepare a Graph, Module, or artifact path for asynchronous native compilation.

Original: [max/python/max/engine/api.py](../../../../max/python/max/engine/api.py#L969); lines 969–1060.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/engine/api.py).

```python
    def compile_async(
        self,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
        tile_based_fusion: bool = False,
    ) -> CompiledModel:
        """Compiles a model without blocking on the compilation finishing.

        Returns as soon as the compile is scheduled; the returned
        :class:`CompiledModel` wraps a pending compilation that runs on the
        runtime's worker pool. Compilation errors are not raised here — they
        surface when the artifact is awaited, for example by :meth:`init`,
        :meth:`init_all`, or :meth:`CompiledModel.export_mef`. Use
        :meth:`compile` for the synchronous variant that blocks and raises.

        Compiles are serialized process-wide: if another compile is in
        flight (from any session), this call blocks until it resolves
        before scheduling this one.

        Args:
            model: A :class:`Graph` instance, a :class:`max.graph.Module`
                containing one or more ``mo.graph`` ops, or the path to a
                saved model file (for example, a ``.mef`` file).

            custom_extensions: The extensions to load for the model.
                Supports paths to ``.mojoc`` custom ops.

            tile_based_fusion: When ``True``, compile the graph under the tile-based
                programming model, in which the graph compiler selects
                tile-based kernels (operating on ``TileTensor`` values instead
                of SIMD). Only applies when ``model`` is a :class:`Graph` or
                :class:`max.graph.Module`; ignored for precompiled ``.mef``
                paths. Defaults to ``False``.

        Returns:
            A :class:`CompiledModel` artifact wrapping the pending compilation.
        """
        custom_extensions_final: list[CustomExtensionType] = []
        if custom_extensions is not None:
            custom_extensions_final = _process_custom_extensions_objects(
                custom_extensions
            )

        # Track the MLIR module if we have one so we can enumerate graph
        # names and capture expected-weight metadata for init-time validation.
        module: Module | None = None
        expected_weights: dict[str, Any] | None = None

        if isinstance(model, Path | str):
            handle = self._impl.compile(model, custom_extensions_final)
        elif isinstance(model, Graph):
            module = model.module
            custom_extensions_final.extend(
                _process_custom_extensions_objects(model.kernel_libraries_paths)
            )

            # TODO: if the model has been loaded from a serialized MLIR
            # file, we don't have the _weights attribute available to us
            if hasattr(model, "_weights"):
                expected_weights = {
                    name: weight.value.device
                    for name, weight in model._weights.items()
                }

            # Seed the model module with kernel decls + the opaque-type
            # mapping from the graph's KernelLibrary. The GC pipeline
            # detects the mapping attribute and skips
            # `mogg-import-packages`, so the expensive package-loading
            # step (run once at KernelLibrary construction) doesn't
            # repeat on every compile.
            kernel_library = getattr(model, "_kernel_library", None)
            if kernel_library is not None:
                kernel_library._analysis.seed_kernel_decls(module.mlir_module)

            handle = self._compile_module(
                module, custom_extensions_final, tile_based_fusion
            )
        elif isinstance(model, Module):
            module = model
            handle = self._compile_module(
                module, custom_extensions_final, tile_based_fusion
            )
        else:
            raise RuntimeError("The model is not a valid path or module.")

        compiled = CompiledModel(
            compiled=handle, expected_weights=expected_weights
        )
        if module is not None:
            compiled._graph_names = tuple(module.top_level_graph_names())
        return compiled
```

| Source lines | Explanation |
|---|---|
| 1018–1024 | **Source-kind dispatch** — Paths and in-memory Graphs follow different branches. A Graph contributes its MLIR module and extension libraries. |
| 1029–1032 | **Retain initialization expectations** — The graph's weight declarations supply expected device metadata, checked when weight buffers are bound. |
| 1040–1046 | **Seed declarations** — Reuse kernel declarations already loaded by the Graph's KernelLibrary before the native module handoff. |
| 1055–1060 | **Return an artifact handle** — The handle carries the asynchronous result and graph names. compile() below waits for completion. |

</details>


<a id="code-compile_module"></a>

<details><summary><strong>InferenceSession._compile_module</strong> — 30 complete lines · SOURCE</summary>

Cross the Python/C++ boundary with the MLIR module capsule.

Original: [max/python/max/engine/api.py](../../../../max/python/max/engine/api.py#L1276); lines 1276–1305.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/engine/api.py).

```python
    def _compile_module(
        self,
        module: Module,
        custom_extensions_final: list[CustomExtensionType],
        tile_based_fusion: bool = False,
    ) -> _AsyncValue[_CompiledModels]:
        """Compiles an MLIR module under the process-global compilation lock.

        Compilation itself is asynchronous; a compile failure surfaces when
        the returned value is awaited (see :meth:`compile`).

        Raises:
            RuntimeError: If synchronous compile setup fails; the message
                points at the ``max-debug.source-tracebacks`` config key
                for richer diagnostics.
        """
        _COMPILATION_LOCK.acquire()
        try:
            handle = self._impl.compile(
                module.mlir_module._CAPIPtr,
                custom_extensions_final,
                _derive_pipeline_name(module),
                tile_based_fusion,
            )
        except Exception as e:
            _COMPILATION_LOCK.release()
            raise RuntimeError(self._compile_failure_message()) from e
        # Released from a runtime worker thread when the compile resolves.
        handle.add_done_callback(lambda _: _COMPILATION_LOCK.release())
        return handle
```

| Source lines | Explanation |
|---|---|
| 1292–1302 | **Serialize native compilation** — The process-wide lock is acquired before scheduling. Synchronous setup errors release it immediately. |
| 1295–1298 | **Payload** — The capsule identifies the MLIR module; extensions, a pipeline name, and the fusion option accompany it. |
| 1304–1304 | **Asynchronous lifetime** — Completion releases the lock through a callback. Returning a pending handle does not end the protected compile interval. |

</details>


<a id="code-compile_wait"></a>

<details><summary><strong>InferenceSession.compile</strong> — 63 complete lines · SOURCE</summary>

Wait for asynchronous compilation and surface errors to the caller.

Original: [max/python/max/engine/api.py](../../../../max/python/max/engine/api.py#L1062); lines 1062–1124.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/engine/api.py).

```python
    def compile(
        self,
        model: str | Path | Module | Graph,
        *,
        custom_extensions: CustomExtensionsType | None = None,
        tile_based_fusion: bool = False,
    ) -> CompiledModel:
        """Compiles a model without binding weights or device memory.

        Use this when you want to separate compilation from initialization, for
        example to populate a compile cache ahead of time, including in
        cross-compilation scenarios where the target device may not be
        attached. The returned :class:`CompiledModel` requires initialization
        before execution. Pass it to :meth:`init` or :meth:`init_all` to
        produce an executable :class:`Model`.

        Blocks until compilation finishes and raises on failure. Use
        :meth:`compile_async` to schedule compilation without blocking.

        Args:
            model: A :class:`Graph` instance, a :class:`max.graph.Module`
                containing one or more ``mo.graph`` ops, or the path to a
                saved model file (for example, a ``.mef`` file).

            custom_extensions: The extensions to load for the model.
                Supports paths to ``.mojoc`` custom ops.

            tile_based_fusion: When ``True``, compile the graph under the tile-based
                programming model, in which the graph compiler selects
                tile-based kernels (operating on ``TileTensor`` values instead
                of SIMD). Only applies when ``model`` is a :class:`Graph` or
                :class:`max.graph.Module`; ignored for precompiled ``.mef``
                paths. Defaults to ``False``.

        Returns:
            A :class:`CompiledModel` artifact ready to be initialized.

        Raises:
            RuntimeError: If the path provided is invalid or compilation
                fails.
        """
        with _record_phase("compile_seconds"):
            compiled = self.compile_async(
                model,
                custom_extensions=custom_extensions,
                tile_based_fusion=tile_based_fusion,
            )
            handle = compiled._compiled
            assert isinstance(handle, _AsyncValue)
            # Synchronously complete the compilation and raise errors.
            handle.wait()
        exception = handle.exception()
        if exception is None:
            return compiled
        if self.debug.pre_jit and "pre-jit" in str(exception):
            raise CompilationStopped(str(exception)) from exception
        # compile_async surfaces the compile failure here rather than from the
        # compile call, so the Graph/Module wrapping that _compile_module
        # applies to synchronous setup errors is repeated here for the async
        # failure.
        if isinstance(model, (Graph, Module)):
            raise RuntimeError(self._compile_failure_message()) from exception
        raise exception
```

| Source lines | Explanation |
|---|---|
| 1103–1124 | **Synchronous completion** — Schedule compilation, wait on its result, and distinguish requested pre-JIT stopping from a compile failure. |

</details>


### C++ session and engine

The C++ session turns the capsule into a `CompilationRequest` with an opaque
source pointer and format. `Engine::Context::compileModel()` schedules a CPU
task. That task calls `compileModelSync()`, which supplies device configuration
and calls `CompiledModels::create()`.

<a id="code-native_session"></a>

<details><summary><strong>Core::InferenceSession::compile</strong> — 25 complete lines · SOURCE</summary>

Turn the Python module capsule into a native CompilationRequest.

Original: [max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp](../../../../max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp#L654); lines 654–678.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/_core/max_license/lib/Engine/InferenceSession.cpp).

```cpp
AsyncRT::AsyncValueRef<CompiledModels> M::Core::InferenceSession::compile(
    const nb::capsule &model,
    std::vector<std::filesystem::path> &&customExtensions,
    std::string &&pipelineName, bool tileBasedFusion) {
  CompilationRequest compileConfig =
      populateCompileConfig(std::move(customExtensions));
  // Set pipeline name. Tracked for telemetry.
  compileConfig.setPipelineName(std::move(pipelineName));
  // Opt into the tile-based programming model if requested.
  compileConfig.setTileBasedFusion(tileBasedFusion);
  // When the format is Max Graph, model is a opaque pointer of graph.
  // So we cast it to a pointer representation and pass it along.
  // The C API will open appropriate dylib for the format and pass the
  // pointer to the library. We have to make sure the pointer is passed down
  // to the library which created the pointer. This is to ensure mlir type
  // id invariants are not broken.
  compileConfig.setModelSource(ModelSource{
      .source = model.data(),
      .fileFormat = frameworkToFileFormat(),
  });
  // Compile async, so it's safe to hold the gil here
  AsyncValueRef<CompiledModels> asyncModel =
      compileModel(*runtimeContext, std::move(compileConfig));
  return asyncModel;
}
```

| Source lines | Explanation |
|---|---|
| 670–673 | **Opaque source and format** — The frontend must interpret the pointer using the matching MLIR implementation and file format. |
| 676–677 | **Schedule engine compilation** — Hand the populated request to the engine and return its asynchronous result. |

</details>


<a id="code-engine_compile"></a>

<details><summary><strong>Engine::Context::compileModel</strong> — 27 complete lines · SOURCE</summary>

Schedule compilation on a CPU worker and resolve an asynchronous result.

Original: [max/lib/max_license/API/c/EngineContext.cpp](../../../../max/lib/max_license/API/c/EngineContext.cpp#L225); lines 225–251.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/lib/max_license/API/c/EngineContext.cpp).

```cpp
AsyncValueRef<CompiledModels>
Engine::Context::compileModel(CompilationRequest config) const {
  auto compiledModelAsync = AsyncValueRef<CompiledModels>::allocate(cpuDevice);

  addTask(cpuDevice, [config = std::move(config),
                      compiledModelAsync = compiledModelAsync.copy(),
                      this]() mutable {
    auto compiledModelOr = compileModelSync(std::move(config));

    auto &telemetryCtx = *maxContext->get<GML::MaxEngineTelemetryContext>();

    if (compiledModelOr.isError()) {
      Error compileError = compiledModelOr.takeError();
      // FIXME: This potentially can contain absolute file paths in user
      // machine. We might need to scrub that.
      Telemetry::emitCompileInfo("compile.failure", telemetryCtx,
                                 compileError.copy());
      return setUnknownLocError(std::move(compiledModelAsync),
                                std::move(compileError));
    }

    Telemetry::emitCompileInfo("compile.success", telemetryCtx, {});
    std::move(compiledModelAsync).emplace(compiledModelOr.takeValue());
  });

  return compiledModelAsync;
}
```

| Source lines | Explanation |
|---|---|
| 229–233 | **Worker boundary** — The captured request is processed by compileModelSync on the CPU worker; error and telemetry handling resolve the returned handle. |

</details>


<a id="code-engine_sync"></a>

<details><summary><strong>Engine::Context::compileModelSync</strong> — 14 complete lines · SOURCE</summary>

Supply device configuration and create native compiled-model objects.

Original: [max/lib/max_license/API/c/EngineContext.cpp](../../../../max/lib/max_license/API/c/EngineContext.cpp#L269); lines 269–282.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/lib/max_license/API/c/EngineContext.cpp).

```cpp
ErrorOr<CompiledModels>
Engine::Context::compileModelSync(CompilationRequest config) const {
  config.setDevices(this->config.getDevices());
  auto modelOr =
      CompiledModels::create(maxContext.copy(), config, this->config);
  if (modelOr.isError())
    return modelOr.takeError();

  if (auto pipelineName = config.getPipelineName(); pipelineName)
    telemetryEmitPipelineName(*pipelineName, maxContext);
  telemetryEmitModelFormat(modelOr.get().getSourceModel().format(), maxContext);

  return modelOr.takeValue();
}
```

| Source lines | Explanation |
|---|---|
| 271–274 | **Target context** — Pass configured devices into the compilation request, then dispatch to CompiledModels::create. |

</details>


### Frontend and compiled artifact

For a graph source, `CompiledModels::create()` constructs the MOF frontend,
calls `compileToBinary()`, and reads its returned MEF. For an already compiled
MEF path, it reads the artifact directly.

The frontend clones the source module before applying compiler transformations,
resolves options, and checks the native MEF cache. A miss enters
`convertMOToMEF()`: run the compiler, then serialize MGP to MEF.

<a id="code-compiled_create"></a>

<details><summary><strong>CompiledModels::create</strong> — 62 complete lines · SOURCE</summary>

Compile in-memory source or read a model artifact from a path.

Original: [max/lib/max_license/API/c/CompiledModels.cpp](../../../../max/lib/max_license/API/c/CompiledModels.cpp#L100); lines 100–161.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/lib/max_license/API/c/CompiledModels.cpp).

```cpp
ErrorOr<CompiledModels>
CompiledModels::create(M::ContextRef context, const CompilationRequest &compCfg,
                       const M::Engine::Config &rtCfg) {
  // If model is from an opaque source with a known format.
  if (auto modelSourceOr = compCfg.getModelSource(); modelSourceOr) {
    auto modelSource = modelSourceOr.value();
    auto frameworkCtxOr = setupFrameworkContext(context.copy(), compCfg, rtCfg);
    if (frameworkCtxOr.isError())
      return frameworkCtxOr.takeError();
    std::unique_ptr<MOFContext> &frameworkCtx = *frameworkCtxOr;

    auto outputBufferOr = frameworkCtx->compileToBinary(modelSource.source,
                                                        modelSource.fileFormat);
    if (outputBufferOr.isError())
      return outputBufferOr.takeError();

    BufferRef outputBuffer = outputBufferOr.takeValue();

    return read(std::move(outputBuffer), std::move(frameworkCtx),
                SourceModel(modelSource));
  } else if (auto modelPathOr = compCfg.getModelPath(); modelPathOr) {
    // Capture runtime configuration which is also needed by the compile-time
    // configuration, since that's currently the only way to convey options
    // down into the driver's M_compileToBinary.
    auto modelPath = modelPathOr.value();

    ErrorOr<ModelContext> modelCtxOr =
        loadModelFramework(modelPath, context.copy());
    if (modelCtxOr.isError())
      return modelCtxOr.takeError();
    auto modelCtx = modelCtxOr.takeValue();
    // Convey compilation options to the driver.
    ErrorOrSuccess errorOr =
        setModelAndFramework(context.copy(), modelCtx, compCfg, rtCfg);
    if (errorOr.isError())
      return errorOr.takeError();

    SourceModel &sourceModel = modelCtx.model;
    M::BufferRef modelBuffer = modelCtx.buffer.copy();
    std::unique_ptr<MOFContext> &frameworkCtx = modelCtx.frameworkCtx;

    // If it's already compiled use that directly.
    if (modelCtx.model.format() == FileFormat::kMEFFile) {
      return read(modelBuffer.copy(), std::move(frameworkCtx), modelCtx.model);
    }

    // Actually compile the model.
    auto outputBufferOr =
        frameworkCtx->compileToBinary(modelBuffer.copy(), sourceModel.format());
    if (outputBufferOr.isError())
      return outputBufferOr.takeError();

    BufferRef outputBuffer = outputBufferOr.takeValue();
    return read(std::move(outputBuffer), std::move(frameworkCtx),
                modelCtx.model);
  }
  auto error = Error("Model path or source is not set in compilation config");
  auto *ctx = context->get<GML::MaxEngineTelemetryContext>();
  GML::emitErrorTelemetry<GML::MaxEngineErrorCode::UserInputError>(
      *ctx, error.copy());
  return error;
}
```

| Source lines | Explanation |
|---|---|
| 104–120 | **Graph-source branch** — A source pointer selects the MOF frontend, which produces a MEF buffer. read then builds compiled-model objects from that buffer. |
| 142–144 | **Already compiled path** — A MEF path is read directly. Entering an API named compile does not imply compiler passes run. |

</details>


<a id="code-frontend"></a>

<details><summary><strong>MOFContext::compileToBinary(void*, FileFormat)</strong> — 83 complete lines · SOURCE</summary>

Clone the graph module, configure lowering, and consult the native MEF cache.

Original: [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp](../../../../GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L377); lines 377–459.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp).

```cpp
M::ErrorOr<M::BufferRef> MOFContext::compileToBinary(void *source,
                                                     FileFormat format) {
  // Cheap: weights are dense_resource attributes owned by the MLIRContext.
  OwningOpRef<ModuleOp> moduleOp =
      ModuleOp::getFromOpaquePointer(source).clone();

  auto &maxContext = impl->maxContext;
  MLIRContext *mlirCtx = moduleOp->getContext();
  registerContext(*mlirCtx, maxContext);

  auto telemetryContext = maxContext->get<GML::MaxEngineTelemetryContext>();

  // Must precede the telemetry diagnostic handler.
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, mlirCtx);

  GML::MaxEngineTelemetryScope<GML::MaxEngineErrorCode::GraphLoweringError>
      loweringDiagScope(mlirCtx, telemetryContext);

  // Gather all the options from all the sources.
  MO::GCPipelineOptions gcPipelineOptions;
  overrideLoweringOptions(gcPipelineOptions);

  // Set iff Python `Analysis::seed_kernel_decls` already seeded kernel decls,
  // so mogg-import-packages can be skipped.
  if ((*moduleOp)->hasAttr(MOGG::kOpaqueTypeMappingAttrName))
    gcPipelineOptions.kernelDeclsPrePopulated = true;

  // Before the cache key, so tile and non-tile compiles never collide.
  if (impl->mofConfig && impl->mofConfig->tileBasedFusion)
    enableTileBasedFusion(*moduleOp);

  bool enableMEFMetadata = impl->getEnableMEFMetadata();

  auto baseCacheDirOr = M::getMaxCacheDir();

  if (!isModelIRCacheEnabled() || !baseCacheDirOr)
    return convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);

  const std::filesystem::path &baseCacheDir = *baseCacheDirOr;
  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      (baseCacheDir / "mof" / "mef").string(),
      std::string(M::getModularVersionString()));

  if (mefBackendOr.isError()) {
    return convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);
  }

  auto keyHashOr = computeMEFCacheKey(*moduleOp, gcPipelineOptions, maxContext);
  if (keyHashOr.isError())
    return keyHashOr.takeError();
  std::string keyHash = keyHashOr.takeValue();

  auto mefCache = RCRef<Cache::TransformCache>::create(*mefBackendOr);
  auto maybeMefOr = mefCache->findKeyedSync(keyHash);
  if (maybeMefOr.isError())
    return maybeMefOr.takeError();
  std::optional<BufferRef> maybeMef = maybeMefOr.takeValue();
  if (maybeMef) {
    auto onHitLambda = Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
        "mof::M_compileToBinaryFromSource");
    onHitLambda(*moduleOp);

    return std::move(*maybeMef);
  }

  auto onMissLambda = Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
      "mof::M_compileToBinaryFromSource",
      "mof.compiletobinaryfromsource.cache.miss.time",
      {{"pipeline", "mof::M_compileToBinaryFromSource"}});
  onMissLambda(*moduleOp);

  auto outBufOr =
      convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);
  if (outBufOr.isError())
    return outBufOr.takeError();
  BufferRef outputBuffer = outBufOr.takeValue();

  // Insertion failure is ignored.
  (void)mefCache->insertKeyedSync(keyHash, outputBuffer.copy());

  return outputBuffer;
}
```

| Source lines | Explanation |
|---|---|
| 381–381 | **Compiler-owned module** — Lowering transforms a clone of the incoming module. Weight resources are owned by its MLIR context. |
| 398–407 | **Resolve compiler options** — Collect configuration and detect pre-seeded kernel declarations before deriving cache behavior. |
| 435–441 | **Cache-hit return** — When model-IR caching is enabled and a matching artifact exists, return it before convertMOToMEF runs. |
| 449–458 | **Cache miss** — Compile/serialize, then attempt to insert the resulting buffer in the cache. |

</details>


<a id="code-convert_mef"></a>

<details><summary><strong>convertMOToMEF</strong> — 20 complete lines · SOURCE</summary>

Run graph compilation, then serialize the lowered model to MEF.

Original: [GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp](../../../../GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp#L356); lines 356–375.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp).

```cpp
static ErrorOr<BufferRef> convertMOToMEF(ModuleOp moduleOp,
                                         const MO::GCPipelineOptions &options,
                                         bool enableMEFMetadata) {
  bool const dumpStages = !getMaxIRDumpDir().empty();
  GraphApiDiagnosticHandler diagnostics(moduleOp.getContext());
  if (failed(MO::runGCPipeline(moduleOp, options, dumpStages)))
    return Error(Twine("Graph compilation failed:\n") +
                 diagnostics.formatMessage());

  // Nothing to export without the JIT; erroring also keeps the half-lowered
  // result out of the MEF cache. emit-mojo already wrote the Mojo.
  if (options.preJit)
    return Error("compilation stopped after emitting Mojo, as requested by "
                 "pre-jit; the emitted Mojo is in ir-output-dir");

  auto bufOr = MO::convertMGPToMEF(moduleOp, enableMEFMetadata);
  if (bufOr)
    return bufOr.takeError();
  return bufOr.takeValue();
}
```

| Source lines | Explanation |
|---|---|
| 361–363 | **Execute compiler passes** — A failed pipeline returns graph diagnostics rather than an artifact. |
| 367–369 | **Intentional early stop** — Pre-JIT inspection ends after Mojo emission; the partially lowered result is not serialized to the MEF cache. |
| 371–374 | **Artifact boundary** — Serialize the successfully lowered module and return its MEF buffer. |

</details>


<details class="self-check"><summary>Check your understanding: does every call to a method named compile run compiler passes?</summary><p>No. The configured MEF import path can read an artifact, and the native frontend can return a cached MEF. Both paths ultimately provide compiled-model objects for initialization.</p></details>

## 5. Transform compiler representations

**The graph passes through several representations because different
decisions need different information.** High-level algebra and shapes are
useful early; structured iteration and memory access support fusion; runtime
models expose allocation, lifetime, and execution requirements.

The stage explorer describes the pass schedule proven by this checkout. It
does not assert a particular fused Llama kernel or a GPU speedup. Such claims
need a concrete graph/target dump and, for speed, a measured run.

```text
Graph IR -> MO preparation -> MOToMAP -> MAP fusion/outlining
         -> MAPToMOGG -> MOGG routines -> MOGGToMGP
         -> MGP resource planning + Mojo emission
         -> kernel JIT + runtime-call lowering -> MEF serialization

Pass scheduling is source-proven. A specific fused kernel needs a concrete dump.
```

`SOURCE` — conceptual illustration; not an executed graph dump.


### MO: prepare graph operations

`buildMoPipeline()` loads missing kernel declarations, legalizes RMO operations,
and performs canonicalization and common-subexpression elimination. Options
control shape inference, constant folding, and named pattern fusion. Layout
inference and device validation prepare the graph for subsequent lowering.

This is distinct from `Graph.output()` verification: a structurally valid
graph still needs target-aware compilation and lowering.

<a id="code-mo_pipeline"></a>

<details><summary><strong>buildMoPipeline</strong> — 118 complete lines · SOURCE</summary>

Prepare and optimize high-level graph operations before MAP lowering.

Original: [GraphCompiler/lib/GraphCompiler.cpp](../../../../GraphCompiler/lib/GraphCompiler.cpp#L219); lines 219–336.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/GraphCompiler.cpp).

```cpp
void M::MO::buildMoPipeline(mlir::OpPassManager &pm,
                            const MoPipelineOptions &opts) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  // Skipped when the caller already seeded decls and
  // `mogg.opaque_type_mapping` via `MOGG::copyKernelDecls`. RewriteOpaqueTypes
  // still runs so freshly-cloned `!mo.opaque<"simple">` uses get rewritten.
  if (!opts.kernelDeclsPrePopulated) {
    pm.addPass(MOGG::createImportPackages(
        {.mojoPkgPaths = kernelLibraryPaths, .litMlirPaths = {}}));
    pm.addPass(MOGG::createVerifyKernels());
  }

  pm.addPass(MOGG::createRewriteOpaqueTypes());
  pm.addPass(KGEN::createVerifyParameters());

  if (opts.debugPrintAllTensors)
    pm.addPass(MO::createDebugPrintAllTensorsPass());

  pm.addPass(RMO::createLegalizeRMOOps());

  pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createCanonicalizeChainSequences());

  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  std::string_view compiledFrameworkLabel;
  if (opts.compiledFrameworkLabel.hasValue())
    compiledFrameworkLabel = opts.compiledFrameworkLabel;

  std::optional<std::string> cacheDir;
  if (opts.cacheDir.hasValue())
    cacheDir = opts.cacheDir;

  if (opts.useShapeInference)
    buildShapeInferencePipeline(pm);

  // TODO(GEX-886): Remove symbolic optimization's dependency on constant
  // folding.
  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  pm.addPass(MO::createHoistConstantSubgraphsIntoParent());
  pm.addNestedPass<MO::GraphOp>(mlir::createCSEPass());

  SymbolicOptimizationOptions symbolicOptimizationOpts;
  symbolicOptimizationOpts.symbolicFfastMath = opts.symbolicFfastMath;
  symbolicOptimizationOpts.symbolicNoUndefinedMath =
      opts.symbolicNoUndefinedMath;

  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addPass(MO::createMergeDuplicateShapeMaterializations());
  pm.addPass(MO::createAddFallbackShapeFunctions());
  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  pm.addPass(MO::createInferLayouts());
  if (opts.usePatterns) {
    MOGG::PatternFusionOptions patternFusionOpts;
    patternFusionOpts.mhaAllowUnsafeBCast =
        opts.patternFusionMHAAllowUnsafeBCasts;
    patternFusionOpts.mhaAllowInternalReshape =
        opts.patternFusionMHAAllowInternalReshape;
    patternFusionOpts.enableMHAPattern = opts.patternFusionMHAEnable;
    patternFusionOpts.targetDevice = opts.defaultDeviceLabel;
    patternFusionOpts.targetDeviceId = opts.defaultDeviceId;
    pm.addPass(MOGG::createPatternFusion(patternFusionOpts));

    InvokeShapePackingFunctionsOptions shapeFuncOptions;
    shapeFuncOptions.sanityCheckShapeFuncs = opts.sanityCheckShapeFuncs;

    pm.addPass(MO::createInvokeShapePackingFunctions(shapeFuncOptions));
    pm.addPass(MO::createPropagateShapes());
    pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  }

  MO::ValidateDevicesPassOptions validateDevicesPassOpt;
  validateDevicesPassOpt.defaultDeviceTargetTriple =
      opts.defaultDeviceTargetTriple;
  validateDevicesPassOpt.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
  pm.addPass(MO::createValidateDevicesPass(validateDevicesPassOpt));

  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  // Absorb per-device consumers of an mo.tensor.unbundle back into the
  // producing mo.parallel (canonicalize collapses bundle(unbundle(X)) to X) so
  // epilogue fusion sees a single kernel body.
  pm.addPass(MO::createCreateParallelFromUnbundle());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createMergeAdjacentParallels());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  if (opts.fuseJitKernels) {
    // Without fusion this can add extraneous copies.
    pm.addPass(MO::createElideRedundantMutableOps());
  }

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createRemoveUnusedOps());
}
```

| Source lines | Explanation |
|---|---|
| 228–231 | **Kernel definitions** — Import/verify packages only when declarations were not already seeded by the caller. |
| 256–263 | **Shape and symbolic work** — Shape inference and constant folding are configured independently; symbolic optimization and cleanup run in this stage. |
| 287–297 | **Layouts and optional patterns** — Infer layouts and optionally run named pattern fusion before validating device information. |
| 312–312 | **Target legality** — Check device information before the final cleanup and unused-operation removal. |

</details>


### MAP: expose and combine structured computation

After `MOToMAP`, structured kernels make iteration and access relationships
available to the compiler. MAP fusion combines eligible work; later passes
handle accumulating buffers and views before outlining kernel bodies.

A Llama residual addition is visible in `TransformerBlock.__call__()`. Whether
it becomes a separate runtime kernel, merges into an epilogue, or participates
in another transformation depends on the concrete graph, dtype, target, and
eligible patterns. Reading `createFuseKernels()` in the schedule proves that
the fusion pass is invoked, not which final kernels it produces.

<a id="code-map_pipeline"></a>

<details><summary><strong>buildMapPipeline</strong> — 31 complete lines · SOURCE</summary>

Fuse eligible structured computations and outline kernel bodies.

Original: [GraphCompiler/lib/GraphCompiler.cpp](../../../../GraphCompiler/lib/GraphCompiler.cpp#L338); lines 338–368.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/GraphCompiler.cpp).

```cpp
void M::MO::buildMapPipeline(mlir::OpPassManager &pm,
                             const MapPipelineOptions &opts) {
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addNestedPass<GraphOp>(MAP::createFuseKernels());
  if (opts.fuseMutableLoadStore)
    pm.addNestedPass<GraphOp>(MAP::createFuseMutableLoads());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  // After fusion settles, give every fused-output epilogue a real store pointer
  // (plus a pre-cast accumulation buffer for mixed-precision accumulation).
  pm.addNestedPass<GraphOp>(MAP::createAllocateAccumulatingBuffers());

  // An unfused reshape has no MAPToMOGG lowering (its hoisted stride op isn't
  // LowerableToMOGGInterface yet), and an unfused contiguous slice can take the
  // zero-copy path. Raise both back before outlining commits to MAP.
  pm.addNestedPass<GraphOp>(MAP::createRaiseBackContiguousViews());

  pm.addPass(MAP::createOutlineStructuredKernelBodies());
  // Uniques `map.iter.opaque` inputs and captures so the lowered kernel's
  // parameter list is minimal.
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(MAP::createLoopInvariantViewMotion());
  // CSE before canonicalize: two loads hoisting the same view create identical
  // stride ops; CSE must unify them so DedupeCapturingClosureCaptures can fold
  // the duplicate captures.
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
}
```

| Source lines | Explanation |
|---|---|
| 343–345 | **Structural fusion** — Fusion is a transformation of eligible graph structure. The schedule alone does not prove which Llama operations fuse. |
| 351–351 | **Output storage** — Provide accumulating buffers after fusion, including mixed-precision handling. |
| 358–367 | **Outline bodies** — Create structured kernel bodies after view handling, then simplify their captures and invariant views. |

</details>


### MOGG: prepare kernel routines

`MAPToMOGG` lowers structured bodies into the next representation.
`buildMoggPipeline()` legalizes rowwise kernels, conforms call operands,
outlines routines, and wraps functions for tracing. The module is then ready
for `MOGGToMGP`.

<a id="code-mogg_pipeline"></a>

<details><summary><strong>buildMoggPipeline</strong> — 11 complete lines · SOURCE</summary>

Legalize kernel calls and outline routines.

Original: [GraphCompiler/lib/GraphCompiler.cpp](../../../../GraphCompiler/lib/GraphCompiler.cpp#L370); lines 370–380.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/GraphCompiler.cpp).

```cpp
void M::MO::buildMoggPipeline(mlir::OpPassManager &pm,
                              const MoggPipelineOptions &) {
  pm.addPass(MOGG::createLegalizeRowwiseKernels());
  pm.addPass(MOGG::createConformCallOperands());
  pm.addPass(MOGG::createOutlineRoutines());
  pm.addPass(MOGG::createWrapFuncsWithTrace());

  // On the module, not GraphOp: the outlined stubs are its siblings.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}
```

| Source lines | Explanation |
|---|---|
| 372–379 | **Call preparation** — Legalize rowwise kernels, conform operands, outline routines, and add tracing wrappers before module cleanup. |

</details>


### MGP: plan runtime work and emit Mojo

This stage reasons about model execution resources: buffer reuse, allocation
planning and deferral, model arguments, dependency chains, and runtime
primitives. It also handles marked device graphs and merges equivalent
routines before emitting Mojo.

Graph-level activation/temporary-buffer planning is a different responsibility
from the serving KV-cache manager's request/page policy. Both concern memory,
but their objects and lifetimes differ.

<a id="code-mgp_pipeline"></a>

<details><summary><strong>buildMgpPipeline</strong> — 107 complete lines · SOURCE</summary>

Plan runtime resources, lower primitives, and emit Mojo routines.

Original: [GraphCompiler/lib/GraphCompiler.cpp](../../../../GraphCompiler/lib/GraphCompiler.cpp#L382); lines 382–488.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/GraphCompiler.cpp).

```cpp
void M::MO::buildMgpPipeline(mlir::OpPassManager &pm,
                             const MgpPipelineOptions &opts) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(MGP::createConcatInPlacePass());
  pm.addPass(MGP::createExecInvariantCodeMotionPass());

  // After ExecInvariantCodeMotion: buffer reuse can block hoisting.
  pm.addPass(MGP::createSimpleBufferReusePass());

  pm.addPass(MGP::createDynamicMemoryPlanningPass());
  pm.addPass(MGP::createDeferAllocsPass());
  pm.addPass(MGP::createVerifyModelArgumentsPass());
  pm.addPass(MGP::createPredicateAssertDependenciesPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  // Before any fusion.
  pm.addPass(MGP::createAttachPythonStackTraces());

  pm.addPass(MGP::createGlobalizeSlotIndicesPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  MGP::LowerPrimitivesOptions lowerPrimOpts;
  lowerPrimOpts.kernelLibraryPaths = kernelLibraryPaths;
  lowerPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  pm.addPass(MGP::createLowerPrimitives(lowerPrimOpts));

  {
    pm.addPass(MGP::createStageDeviceGraphsPass());
    pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());
    pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());
    pm.addPass(MGP::createGenerateBuildGraphStubsPass());
    // Drop the dead launch context chain before finalization audits the
    // builder bridge's live users.
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addPass(MGP::createFinalizeDeviceGraphsPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  pm.addPass(MOGG::createSequenceFusion());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  pm.addPass(MGP::createSimplifyExtendLifetimesPass());

  pm.addPass(MOGG::createOutlineAsyncRoutines());
  pm.addPass(MOGG::createLookupConstantResources());

  // Lifting slot-index constants into stub arguments lets kernels differing
  // only in those indices merge below.
  pm.addPass(MOGG::createLiftBufferCacheSlots());

  // The emitted Mojo reads neither device ids nor symbolic dim names, so
  // erasing them lets per-device and per-layer copies merge below.
  pm.addPass(MOGG::createCanonicalizeFuncsForMerge());

  // Stubs that differ only by a redundant op merge once it is gone.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MOGG::createMergeIdenticalFuncs());

  // index.constant emits fewer lines of Mojo than its lowered form.
  pm.addPass(MGP::createRaiseBackIndexConstant());

  pm.addPass(MGP::createFixupOverhangsPass());
  pm.addNestedPass<MOGG::FuncOp>(mlir::createCanonicalizerPass());

  // Drop stubs orphaned by MergeIdenticalFuncs.
  pm.addPass(mlir::createSymbolDCEPass());

  // Before try/except wrapping: the begin/end Trace calls raise, and the
  // MGP-ABI entry point must not.
  if (opts.enableProfileScopeTracing) {
    MOGG::AddProfileScopeTracingOptions profileScopeOptions;
    profileScopeOptions.profileScopeTracingTraceUnlabelled =
        opts.profileScopeTracingTraceUnlabelled;
    pm.addPass(MOGG::createAddProfileScopeTracing(profileScopeOptions));
  }

  // Last, so it sees the final raising set; try/except around non-raising
  // code warns in the Mojo compiler.
  pm.addPass(MOGG::createAddTryExceptBlocks());

  MOGG::EmitMojoOptions emitMojoOptions;
  emitMojoOptions.kernelLibraryPaths = kernelLibraryPaths;
  emitMojoOptions.enableOpTraceComments = opts.enableOpTraceComments;
  emitMojoOptions.enableComputeLambdas = opts.enableComputeLambdas;
  emitMojoOptions.moggDebugDumpFusedTensorSpecs =
      opts.moggDebugDumpFusedTensorSpecs;
  pm.addPass(MOGG::createEmitMojo(emitMojoOptions));

  if (opts.dumpMojoText)
    pm.addPass(MOGG::createDumpMojo());
}
```

| Source lines | Explanation |
|---|---|
| 394–400 | **Memory planning** — Buffer reuse, allocation planning, deferral, and argument validation operate on the runtime-model representation. |
| 417–429 | **Device-graph path** — Staging, dependency computation, build stubs, and finalization are separate from the frontend Graph.output operation. |
| 454–454 | **Deduplicate routines** — Earlier canonicalization makes equivalent routines eligible for merging. Layer count is not necessarily the count of distinct generated routines. |
| 484–484 | **Emit source for JIT** — Mojo emission is at the end of this builder. The enclosing pipeline schedules kernel JIT afterward. |

</details>


### JIT and MEF: finish the artifact

`gcPipelineImpl()` connects the builders with explicit conversions. After
MGP processing and Mojo emission, it schedules `JitCompileKernels`, strips
kernel declarations, and finishes JIT lowering. `runGCPipeline()` constructs
the pass manager and calls `pm.run()`; adding passes alone does not execute
them.

MEF serialization occurs in the frontend after successful pipeline execution.
A MEF-backed `CompiledModel` is a compiled artifact. It is not yet the
weight-initialized engine `Model` used for token execution.

<a id="code-gc_pipeline"></a>

<details><summary><strong>gcPipelineImpl</strong> — 131 complete lines · SOURCE</summary>

Build the ordered sequence of representation conversions and compiler stages.

Original: [GraphCompiler/lib/GraphCompiler.cpp](../../../../GraphCompiler/lib/GraphCompiler.cpp#L494); lines 494–624.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/GraphCompiler.cpp).

```cpp
static void gcPipelineImpl(mlir::OpPassManager &pm,
                           const GCPipelineOptions &opts, bool skipJit,
                           bool dumpIntermediates) {

  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  // Filled by the first ("mo") dump from the non-subgraph model names while
  // the GraphOps still exist; later dumps reuse it.
  auto dumpBaseName = std::make_shared<std::string>();

  if (dumpIntermediates)
    pm.addPass(MO::createDumpStagePass("mo", dumpBaseName));

  MoPipelineOptions moOpts;
  moOpts.debugPrintAllTensors = opts.debugPrintAllTensors;
  moOpts.useShapeInference = opts.useShapeInference;
  moOpts.constantFold = opts.constantFold;
  if (opts.compiledFrameworkLabel.hasValue())
    moOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel.getValue();
  if (opts.cacheDir.hasValue())
    moOpts.cacheDir = opts.cacheDir.getValue();
  moOpts.symbolicFfastMath = opts.symbolicFfastMath;
  moOpts.symbolicNoUndefinedMath = opts.symbolicNoUndefinedMath;
  moOpts.kernelLibraryPaths = kernelLibraryPaths;
  moOpts.kernelDeclsPrePopulated = opts.kernelDeclsPrePopulated;
  moOpts.usePatterns = opts.usePatterns;
  moOpts.patternFusionMHAEnable = opts.patternFusionMHAEnable;
  moOpts.patternFusionMHAAllowUnsafeBCasts =
      opts.patternFusionMHAAllowUnsafeBCasts;
  moOpts.patternFusionMHAAllowInternalReshape =
      opts.patternFusionMHAAllowInternalReshape;
  moOpts.sanityCheckShapeFuncs = opts.sanityCheckShapeFuncs;
  moOpts.defaultDeviceLabel = opts.defaultDeviceLabel;
  moOpts.defaultDeviceId = opts.defaultDeviceId;
  moOpts.defaultDeviceTargetTriple = opts.defaultDeviceTargetTriple;
  moOpts.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
  moOpts.fuseJitKernels = opts.fuseJitKernels;

  MapPipelineOptions mapOpts;
  mapOpts.fuseMutableLoadStore = opts.fuseMutableLoadStore;

  MoggPipelineOptions moggOpts;

  MGP::MOGGToMGPOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;

  MgpPipelineOptions mgpOpts;
  mgpOpts.kernelLibraryPaths = kernelLibraryPaths;
  mgpOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  mgpOpts.enableOpTraceComments = opts.enableOpTraceComments;
  mgpOpts.enableComputeLambdas = opts.enableComputeLambdas;
  mgpOpts.moggDebugDumpFusedTensorSpecs = opts.moggDebugDumpFusedTensorSpecs;
  mgpOpts.dumpMojoText = opts.dumpMojoText;
  mgpOpts.enableProfileScopeTracing = opts.enableProfileScopeTracing;
  mgpOpts.profileScopeTracingTraceUnlabelled =
      opts.profileScopeTracingTraceUnlabelled;

  MOGG::JitCompileKernelsOptions jitOpts;
  if (!skipJit) {
    jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
    jitOpts.saveTempPrefix = opts.saveTempPrefix;
    jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
    jitOpts.kernelLibraryPaths = kernelLibraryPaths;
    jitOpts.timeMOGGCompile = opts.timeMOGGCompile;
    jitOpts.debugKGENStubs = opts.debugKGENStubs;
    // Cross-compile for the requested GPU instead of probing the host.
    jitOpts.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
    jitOpts.defaultDeviceTargetTriple = opts.defaultDeviceTargetTriple;
    jitOpts.compileForHostExecution = opts.compileForHostExecution;
  }

  auto buildStage = [&](auto &&build, const std::string &dumpSuffix) {
    build(pm);
    if (dumpIntermediates && !dumpSuffix.empty())
      pm.addPass(MO::createDumpStagePass(dumpSuffix, dumpBaseName));
  };

  buildStage([&](auto &p) { buildMoPipeline(p, moOpts); }, "mo-pre-map");

  // Stable-tensor marking must see MO ops, so it runs as the last MO-level
  // pass before the dialect is lowered away.
  buildStage(
      [&](auto &p) {
        p.addPass(MO::createMarkStableTensorsPass());
        p.addPass(MO::createMOToMAP());
      },
      "mo-to-map");

  buildStage(
      [&](auto &p) {
        buildMapPipeline(p, mapOpts);
        // After MAP fusion so a check never fuses into the kernel it checks;
        // before map-to-mogg so the injected checks compile with the rest.
        if (opts.nanCheck)
          p.addPass(MO::createNanCheckPass(opts.nanCheckStride));
      },
      "map-pre-mogg");

  MAP::MAPToMOGGOptions mapToMoggOpts;
  mapToMoggOpts.enableComputeLambdas = opts.enableComputeLambdas;
  buildStage([&](auto &p) { p.addPass(MAP::createMAPToMOGG(mapToMoggOpts)); },
             "map-to-mogg");

  buildStage(
      [&](auto &p) {
        buildMoggPipeline(p, moggOpts);
        // OpaqueLoc tracebacks can't be printed, so convert them for the
        // dump. Dump-only: this strips the OpaqueLocs buildMgpPipeline's
        // assert handling reads.
        if (dumpIntermediates)
          p.addPass(MGP::createAttachPythonStackTraces());
      },
      "mogg-pre-mgp");

  buildStage([&](auto &p) { p.addPass(MGP::createMOGGToMGP(moToPrimOpts)); },
             "mogg-to-mgp");

  buildStage(
      [&](auto &p) {
        buildMgpPipeline(p, mgpOpts);
        if (!skipJit) {
          p.addPass(MOGG::createJitCompileKernels(jitOpts));
          p.addPass(MOGG::createStripKernelDecls());
          buildJitPipeline(p);
        }
      },
      "mgp");
}
```

| Source lines | Explanation |
|---|---|
| 574–574 | **High-level graph stage** — The mo-pre-map dump follows MO processing; subsequent stage names distinguish conversions from stage-local transforms. |
| 581–583 | **Representation transition** — Stable-tensor marking precedes MO-to-MAP conversion. |
| 602–608 | **Kernel representation** — MOGG preparation precedes conversion into the MGP runtime model. |
| 555–565 | **JIT configuration** — JIT target and compilation options are only configured when JIT is enabled. |
| 618–620 | **Compile kernels** — After MGP/Mojo emission, run JIT compilation, strip kernel declarations, and finish runtime-call lowering. |

</details>


<a id="code-run_pipeline"></a>

<details><summary><strong>runGCPipeline</strong> — 18 complete lines · SOURCE</summary>

Construct the pass manager and run the configured compiler pipeline.

Original: [GraphCompiler/lib/GraphCompiler.cpp](../../../../GraphCompiler/lib/GraphCompiler.cpp#L638); lines 638–655.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/GraphCompiler.cpp).

```cpp
LogicalResult M::MO::runGCPipeline(ModuleOp moduleOp,
                                   const GCPipelineOptions &opts,
                                   bool dumpIntermediates) {
  auto pm = mlir::PassManager::on<ModuleOp>(
      moduleOp.getContext(), mlir::OpPassManager::Nesting::Implicit);

  if (!opts.mlirTimingOutDir.empty()) {
    auto enabledTiming = M::configureMLIRPassTimingJSONOutput(
        pm, opts.mlirTimingOutDir.getValue(), "GC Pipeline");
    if (enabledTiming.isError()) {
      llvm::errs() << enabledTiming.getError() << "\n";
      return failure();
    }
  }

  gcPipelineImpl(pm, opts, /*skipJit=*/opts.preJit, dumpIntermediates);
  return pm.run(moduleOp);
}
```

| Source lines | Explanation |
|---|---|
| 653–654 | **Build then execute** — Adding passes describes the schedule. pm.run applies that schedule to the module and returns success or failure. |

</details>


| Observation point | What to inspect | Evidence needed |
|---|---|---|
| Before native compile | Graph signature, output types, devices, named weight declarations | Source or a graph-building run |
| Stage boundaries | `mo-pre-map`, `mo-to-map`, `map-pre-mogg`, `map-to-mogg`, `mogg-pre-mgp`, `mogg-to-mgp`, `mgp` | Actual IR dumps for the chosen graph/target |
| After Mojo emission | Generated routines; `pre_jit` can intentionally stop before JIT | Source-matched compile inspection |
| After MEF creation | Artifact and model interfaces | Compile result |
| Runtime performance | Kernel launches, allocation/copy behavior, latency | GPU run with workload and target metadata |

`pre_jit` deliberately ends before a runnable artifact can be serialized. The
frontend reports that requested stop; it is different from claiming the
entire graph compiled successfully.

<details class="self-check"><summary>Check your understanding: does one transformer layer imply one GPU kernel?</summary><p>No. A layer contains many operations. Compiler fusion, outlining, and routine reuse determine a different structure, while kernels can launch their own target-specific work. The final kernel set must be inspected for the concrete graph and target.</p></details>

## 6. Initialize, execute, and reuse

**Initialization combines the compiled artifact with weight buffers and
runtime state. Execution supplies the current batch's inputs to that instance.**
The Python session checks expected weight names and CPU/device category when
that metadata is available, checks buffer contiguity, and enters the native
load path. The engine's continuation constructs executable models on a CPU
runtime worker.

The full `init_all()` also contains a virtual-device branch that returns
compile-only stand-ins. This chapter's executable model path assumes the
required real target device is available.

<a id="code-init_all"></a>

<details><summary><strong>InferenceSession.init_all</strong> — 94 complete lines · SOURCE</summary>

Validate weight buffers and initialize models in a compiled artifact.

Original: [max/python/max/engine/api.py](../../../../max/python/max/engine/api.py#L1160); lines 1160–1253.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/engine/api.py).

```python
    def init_all(
        self,
        compiled: CompiledModel,
        *,
        weights_registry: Mapping[str, DLPackArray] | None = None,
    ) -> dict[str, Model]:
        """Initializes all models in a compiled artifact for execution.

        Use this to complete the second half of a
        :meth:`compile`/:meth:`init_all` pair. Returns one :class:`Model` per
        top-level graph in the artifact,
        keyed by ``sym_name``.

        Args:
            compiled: The compiled artifact returned by :meth:`compile`.

            weights_registry: A mapping from model weight names to their
                values. See :meth:`init` for details.

        Returns:
            A mapping from each model's ``sym_name`` to its initialized
            :class:`Model`, ready to execute.
        """
        if is_virtual_device_mode():
            # Virtual device mode can't actually initialize the model, but
            # users (eg. cross compilation, benchmarking) want it to not fail.
            # Return one stand-in per top-level graph in the artifact so callers
            # that key by graph name still work; the stand-in refuses anything
            # that would need the device rather than pretending to have one.
            if not compiled._graph_names:
                raise ValueError(
                    "Cannot initialize a path-compiled artifact in "
                    "virtual-device mode: graph names are unknown without "
                    "an MLIR module to inspect. Initialize on a real device "
                    "instead, or compile from a Graph/Module."
                )
            # Deliberately not a Model: the point is that it cannot execute.
            return {
                name: cast(Model, CompileOnlyModel(name, compiled))
                for name in compiled._graph_names
            }

        weights_registry_real: Mapping[str, DLPackArray] = (
            weights_registry or {}
        )

        # Validate the registry against the expected weights captured at
        # compile time (only available when the source was a Graph with a
        # `_weights` attribute).
        if compiled._expected_weights is not None:
            for (
                weight_name,
                expected_device,
            ) in compiled._expected_weights.items():
                if weight_name not in weights_registry_real:
                    raise ValueError(
                        f"Weight '{weight_name}' is not in the weights registry."
                    )

                registered_weight = weights_registry_real[weight_name]
                if (
                    expected_device is None
                    or expected_device.device_type.value == "cpu"
                ) != (
                    # 1 is the value of DLDeviceType::kDLCPU
                    registered_weight.__dlpack_device__()[0] == 1
                ):
                    raise ValueError(
                        f"Mismatch in device type for weight '{weight_name}'. Expected {expected_device} but weight is {registered_weight}"
                    )

        for weight_name, weight_value in weights_registry_real.items():
            try:
                _raise_if_not_contiguous(weight_value)
            except ValueError as e:
                raise ValueError(
                    f"Weight '{weight_name}' is not contiguous: {str(e)}"
                ) from e

        if isinstance(compiled._compiled, _CompiledModels):
            # Wrapping consumes the resolved artifact; cache the async
            # handle so a later init can reuse it.
            compiled._compiled = self._impl._wrap_compiled(compiled._compiled)
        with _record_phase("init_seconds"):
            models = self._impl._load_all(
                compiled._compiled, weights_registry_real
            )
        result = {m.name: m for m in models}
        if len(result) != len(models):
            raise RuntimeError(
                "Compiled artifact contains models with duplicate sym_names; "
                f"got {[m.name for m in models]}"
            )
        return result
```

| Source lines | Explanation |
|---|---|
| 1209–1228 | **Weight contract** — Check names and expected CPU/device category captured from the graph. These checks occur after graph finalization and compilation. |
| 1233–1237 | **Buffer layout check** — Non-contiguous registered weights are rejected with the offending weight's name. |
| 1244–1252 | **Native initialization** — Bind the registry through the native load path; return models keyed by their names. |

</details>


<a id="code-native_init"></a>

<details><summary><strong>Engine::Context::initModel(CompiledModels)</strong> — 31 complete lines · SOURCE</summary>

Create executable model instances once compilation resolves.

Original: [max/lib/max_license/API/c/EngineContext.cpp](../../../../max/lib/max_license/API/c/EngineContext.cpp#L310); lines 310–340.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/lib/max_license/API/c/EngineContext.cpp).

```cpp
AsyncValueRef<ExecutableModels>
Engine::Context::initModel(AsyncValueRef<CompiledModels> compiledModels,
                           const WeightsRegistryRef &weightsRegistry) const {
  auto result = AsyncValueRef<ExecutableModels>::allocate(cpuDevice);

  // Use andThenAsync to ensure createExecutableModels runs on a cpuDevice
  // worker thread.  The caller may be a foreign thread (e.g. a Python
  // ThreadPoolExecutor thread) that has no AsyncRT cpuDevice set in its
  // thread-local storage; MGP_RT_BufferBorrowV2 (called during Model::setup)
  // requires a valid current cpuDevice.
  std::move(compiledModels)
      .andThenAsync(
          [result = result.copy(), weights = weightsRegistry.copy(),
           this](AsyncValueRef<CompiledModels> &&compiledModelsRef) mutable {
            if (compiledModelsRef.isError())
              return setUnknownLocError(
                  std::move(result),
                  compiledModelsRef.takeDiagnostic().getMessage().copy());

            auto executableModelsOr =
                createExecutableModels(compiledModelsRef.get(), cpuDevice,
                                       getConfig(), weights.getPointer());
            if (executableModelsOr.isError())
              return setUnknownLocError(std::move(result),
                                        executableModelsOr.takeError());

            std::move(result).emplace(executableModelsOr.takeValue());
          });

  return result;
}
```

| Source lines | Explanation |
|---|---|
| 321–331 | **Initialize on a runtime worker** — The continuation uses the compiled artifact and captured weight registry on the CPU runtime worker. |
| 330–336 | **Executable state** — Create the executable models with runtime configuration and actual weight buffers, then resolve the result. |

</details>


### Per-batch execution is a small call boundary

`LlamaModelBase.execute()` receives already prepared buffers, calls the
initialized model, and delegates interpretation of its output buffers to the
batch processor. The graph-building and compiler methods are not called in
this method.

<a id="code-execute"></a>

<details><summary><strong>LlamaModelBase.execute</strong> — 7 complete lines · SOURCE</summary>

Execute an initialized decoder on the current batch's buffers.

Original: [max/python/max/pipelines/architectures/llama3/model.py](../../../../max/python/max/pipelines/architectures/llama3/model.py#L201); lines 201–207.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/max/python/max/pipelines/architectures/llama3/model.py).

```python
    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Llama3Inputs)
        assert model_inputs.kv_cache_inputs is not None
        model_outputs = self.model.execute(*model_inputs.buffers)

        assert self.batch_processor is not None
        return self.batch_processor.process_outputs(model_outputs)
```

| Source lines | Explanation |
|---|---|
| 204–207 | **Per-batch call** — Prepared token/KV buffers enter the existing model. The batch processor interprets its output buffers. |

</details>


### Prefill and decode change the inputs

Use the two-sequence illustration below to see what changes at the graph
boundary. Token IDs are invented examples; they are not captured model output.
The logical KV history grows between steps, while the runtime supplies the
appropriate KV buffers/metadata through the declared input subtree.

```text
Two illustrative sequences; token IDs are examples, not a captured Llama run.

PREFILL: tokens [17, 42, 6 | 9, 21]   row_offsets [0, 3, 5]
DECODE:  tokens [88       | 31   ]   row_offsets [0, 1, 2]

types: tokens=int64[T], offsets=uint32[B+1], return_n_logits=int64[R] on CPU
KV leaves: buffers/metadata declared by kv_params; carry the cache state.
```

`SOURCE` — conceptual illustration; not an executed graph dump.


Changing concrete token values or lengths within the supported symbolic
contract does not itself alter the Python layer definitions. Changes to
architecture, dtype, devices, output policy, or specialization settings need
the corresponding compatible graph/artifact. This is not a universal promise
that every possible shape change avoids compilation.

### Cached compilation still has a trace in this route

`compile_and_init_graphs()` calls `GraphSpec.trace()` before
`compile_reusing_mefs()`. Therefore a configured MEF import or native cache hit
can skip graph compiler work after tracing has already happened. Other MAX
artifact-loading APIs may enter at a different point.

```text
COLD:       trace -> frontend -> passes -> serialize -> read MEF -> init
MEF IMPORT: trace -> Python store -> read MEF path -> init
NATIVE HIT: trace -> frontend -> MEF cache hit -> read MEF -> init

In this GraphSpec route, all three paths have already traced the graph.
```

`SOURCE` — conceptual illustration; not an executed graph dump.


### Device-graph finalization is another mechanism

The ordinary `GraphSpec` defaults `is_device_graph=False`.
`FinalizeDeviceGraphsPass` runs within MGP processing but returns immediately
for an unmarked model. For a marked model, it finalizes device-graph
collectives after the staging/build-stub pipeline. That work is separate from
the output signature and `mo.output` installed by Python.

CUDA graph capture/replay also concerns replaying device work. It should not
be used as another name for constructing or finalizing a MAX `Graph`.

<a id="code-device_finalize"></a>

<details><summary><strong>FinalizeDeviceGraphsPass</strong> — 15 complete lines · SOURCE</summary>

Finalize collectives only for an MGP model marked as a device graph.

Original: [GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp](../../../../GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp#L375); lines 375–389.
Revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. [Complete file](sources/GraphCompiler/lib/MGPDialect/Transforms/FinalizeDeviceGraphsPass.cpp).

```cpp
class FinalizeDeviceGraphsPass
    : public impl::FinalizeDeviceGraphsPassBase<FinalizeDeviceGraphsPass> {
  using Base::Base;

  void runOnOperation() override {
    ModelOp model = getOperation();
    if (!model.getIsDeviceGraph())
      return;

    for (DeviceGraphCollectiveOp collectiveOp : llvm::to_vector(
             model.getExecuteBlock()->getOps<DeviceGraphCollectiveOp>()))
      if (failed(finalizeCollectiveOp(collectiveOp, model)))
        return signalPassFailure();
  }
};
```

| Source lines | Explanation |
|---|---|
| 381–382 | **Explicit gate** — An unmarked model returns immediately. The pass's name refers to device-graph collectives, not Python graph output finalization. |
| 384–387 | **Finalize each collective** — Failures from collective finalization signal a compiler pass failure. |

</details>


### Diagnose the boundary first

| Symptom | First code to inspect | Question |
|---|---|---|
| Graph output/verification error | `Graph.output()` | Are the output values, chains, and signature valid? |
| Native compilation failure | `_compile_module()` → frontend → compiler diagnostics | Which representation, target rule, or lowering failed? |
| Weight initialization failure | `InferenceSession.init_all()` | Are names, device categories, and buffer layouts compatible? |
| Incorrect inference values | Model execution and a reference implementation | Does numerical behavior match for the same inputs/configuration? |
| Unexpected startup time | Trace, compilation, artifact reuse, and initialization separately | Which phase actually ran, and was an artifact reused? |

The [source map](SOURCE_MAP.md) is the compact reading route. The
[source manifest](source_manifest.json) records every included file and
function range. The [research review](EXPLANATION_RESEARCH.md) explains the
teaching methods used to structure this guide.

<details class="self-check"><summary>Check your understanding: what is the complete lifecycle in one sentence?</summary><p>MAX resolves the Llama configuration, records its symbolic computation, finalizes the graph's interface, compiles or resolves an artifact, initializes executable state with weights, and executes that model on prepared batch buffers.</p></details>

