# Llama 2 7B / MAX serve call stacks

[Interactive text-tree window](call_stack.html).

SOURCE revision: 610f7111836d9a950d45ba52ec9b3e0ed589fe3f

## QnA 1.1.1 · CLI arguments -> Hugging Face model configuration

```text
MAIN / API PROCESS | Start with either command shown above.
Click dispatches a registered command; it is an external framework callback.
|
+-- WithLazyPipelineOptions.parse_args() [CLI:L117] [Click wrapper in MAX]
|   +-- WithLazyPipelineOptions._ensure_options_loaded() [CLI:L86]
|       +-- WithLazyPipelineOptions._add_options() [CLI:L80]
|           |   Importing CLI config can import max.pipelines: lazy registration is in QnA 1.1.2.
|           +-- pipeline_config_options() [FLAGS:L443]
|               +-- get_normalized_flag_names() [FLAGS:L204]
|                   result: --model / --model-path -> internal model_path field.
|
+-- cli_serve(..., **config_kwargs) [CLI:L349] [MAX CLI]  Click registration [CLI:L349]
    |   input: parsed Click kwargs; --host/--port are server arguments.
    |          --model, --task, --max-length, etc. arrive through config_kwargs.
    +-- Settings(**setting_kwargs) [SETTINGS:L93] [server policy]  call site [CLI:L490]
    |   returns: settings : Settings (host, port, API/queue/timeouts policy).
    +-- PipelineArgs.from_flat_kwargs() [ARGS:L459] [MAX configuration]  call site [CLI:L500]
    |   |   input example: model_path="meta-llama/Llama-2-7b-hf".
    |   |   Merge --config-file YAML when supplied; explicit CLI fields take precedence.
    |   |   output: immutable pipeline_args : PipelineArgs.
    |   +-- ModelManifest.from_model_path() [MANIFEST:L382] [checkpoint discovery]  call site [ARGS:L621]
    |       |   For these commands: detect a single-model checkpoint; build a temporary "main" entry.
    |       +-- _build_model_config() [MODEL_CFG:L155]  call site [MANIFEST:L428]
    |           +-- MAXModelConfig._populate_repo_handles() [MODEL_CFG:L1055]
    |           |   creates: HuggingFaceRepo handles carrying repo identity, revision and trust policy.
    |           +-- MAXModelConfig._populate_hf_config() [MODEL_CFG:L1067]
    |           |   +-- load_huggingface_config(repo) [HF_CFG:L90]  call site [MODEL_CFG:L1083]
    |           |       |   input: model repo handle. Cache key belongs to this Python process.
    |           |       +-- IF repo already in _config_cache -> return cached config object.
    |           |       +-- ELSE AutoConfig.from_pretrained(repo.repo_id, **kwargs) [HF_CFG:L121] [Transformers; MAX call site]
    |           |           reads: config.json from Hub/cache; missing assets can be fetched.
    |           |           returns: normally transformers.LlamaConfig for model_type="llama".
    |           |           stores: _config_cache[repo] = returned config.
    |           +-- MAXModelConfig._populate_generation_config() [MODEL_CFG:L1087] -> generation defaults; expanded in QnA 1.1.4.
    |       result: manifest contains "main"; this probe is not retained as a manifest override.
    |               Its HF config can remain in the shared process cache.
    +-- serve_api_server_and_model_worker() [SERVE:L60] [MAX serving orchestration]  call site [CLI:L564]
        +-- IF effective task == UNDEFINED:  branch test [SERVE:L83]
        |   |   Example: command WITHOUT --task, and no config-file task.
        |   +-- PipelineArgs.main_architecture_name [property] [ARGS:L439]
        |   |   +-- MAXModelConfig.from_pipeline_args() [MODEL_CFG:L1154] -> _build_model_config() [MODEL_CFG:L155]
        |   |       reads architectures[0] -> usually "LlamaForCausalLM"; may reuse cached HF config.
        |   +-- PipelineRegistry.retrieve_pipeline_task() [REGISTRY:L1047] [MAX registry]
        |       returns TEXT_GENERATION; handler creates an args copy with the resolved task.
        +-- ELSE: --task text_generation already supplied -> skip ONLY the inference branch.
        |   The checkpoint probe above and full configuration below still run.
        +-- PipelineConfig.from_args() [CONFIG:L1482]  call site [SERVE:L96]
            +-- MAXModelConfig.from_pipeline_args() [MODEL_CFG:L1154] -> _build_model_config() [MODEL_CFG:L155]
            |   creates: serving models["main"] : MAXModelConfig; may reuse the HF process cache.
            +-- find_architecture() [LOOKUP:L1012]  call site [CONFIG:L1562]
            |   Need the architecture while resolving denoising-cache defaults, even on this text path.
            |   With explicit task, this can be the FIRST materialization; see QnA 1.1.2.
            +-- resolve remaining model/runtime/sampling policies -> pipeline_config : PipelineConfig.

OBJECT DISTINCTION: Transformers LlamaConfig = checkpoint metadata.
                   MAX Llama3Config = architecture/runtime config, built later (QnA 1.1.6).
```

## QnA 1.1.2 · Lazy architecture registration -> API, worker and message queues

```text
MAIN / API PROCESS | Three connected phases; indentation shows calls within each phase.
|
+-- A. First import of max.pipelines  package import site [PACKAGE:L80]
|   |   Can happen during Click option loading, before cli_serve; earlier imports may reach it first.
|   +-- register_all_models() [ARCHS:L58] [MAX architecture catalog]
|       |   shipped row: ("LlamaForCausalLM", ".llama3", "llama_arch").
|       +-- PipelineRegistry.register_lazy() [REGISTRY:L415]
|           +-- ArchLookup.register_lazy() [LOOKUP:L767]
|               stores: _lazy_architectures[name] += [(module, symbol, package)].
|               example: name="LlamaForCausalLM"; package="max.pipelines.architectures".
|               result: an in-memory import recipe; no Llama model instance or 7B tensors.
|
+-- B. First lookup materializes that import recipe [fresh standard process]
|   +-- WITHOUT explicit task: PipelineRegistry.retrieve_pipeline_task() [REGISTRY:L1047] -> PipelineRegistry._materialize_lazy() [REGISTRY:L452]
|   +-- WITH explicit task: PipelineConfig.from_args() [CONFIG:L1482] -> find_architecture() [LOOKUP:L1012]
|   |   +-- ArchLookup.find() [LOOKUP:L947] -> ArchLookup.resolve() [LOOKUP:L930]
|   +-- EITHER pending path reaches ArchLookup.materialize(name) [LOOKUP:L804]
|       +-- importlib.import_module(module, package=package) [LOOKUP:L815] [Python import system; MAX call site]
|       |   imports .llama3 relative to max.pipelines.architectures; reads llama_arch.
|       +-- ArchLookup.register(llama_arch) [LOOKUP:L705]
|           input: llama_arch : SupportedArchitecture [LLAMA_ARCH:L28]
|           fields: pipeline_model=Llama3Model; config=Llama3Config; tokenizer=TextTokenizer.
|           before: pending import tuple. After: declaration in name and (name, task) tables.
|           later lookups reuse it; class references are not a checkpoint database.
|
+-- C. serve_api_server_and_model_worker() [SERVE:L60] continues with pipeline_config
    +-- PipelineRegistry.retrieve_factory() [REGISTRY:L711]  call site [SERVE:L97]
    |   +-- PipelineRegistry.architecture_for_config() [REGISTRY:L496] -> PipelineRegistry.retrieve_architecture() [REGISTRY:L471]
    |   |   +-- ArchLookup.find() [LOOKUP:L947] -> ArchLookup.resolve() [LOOKUP:L930] -> selected SupportedArchitecture.
    |   +-- MemoryEstimator.plan() [MEMORY:L146] -> memory_plan : MemoryPlan; config initialization in QnA 1.1.6.
    |   +-- tokenizer construction -> tokenizer : TextTokenizer; expanded in QnA 1.1.3.
    |   +-- functools.partial(pipeline_class, **factory_kwargs) [REGISTRY:L994] [deferred Python callable]
    |       binds TextGenerationPipeline + config, model class, tokenizer, adapters and memory plan.
    |       returns RetrievedPipeline(tokenizer, factory, memory_plan); factory has NOT executed.
    +-- ServingTokenGeneratorSettings(...) [API:L102]  call site [SERVE:L116]
    |   creates pipeline_settings: factory + pipeline_config + tokenizer + task + memory_plan.
    +-- fastapi_app(settings, pipeline_settings) [API:L346] [MAX constructs FastAPI app]  call site [SERVE:L128]
    |   settings -> HTTP routes, middleware and limits; pipeline_settings -> model-serving context.
    |   returns app : FastAPI (ASGI application). Creating the app does not start HTTP listening.
    +-- fastapi_config(app, server_settings=settings) [API:L482] [MAX -> Uvicorn configuration]  call site [SERVE:L129]
    |   reuses settings for host, port, keep-alive and shutdown timeouts.
    |   returns config : uvicorn.Config; lifespan="off" because MAX enters lifespan explicitly.
    +-- Server(config) [SERVE:L134] [Uvicorn constructor; MAX call site]
    |   creates the server controller holding config/app; listening starts later.
    +-- uvloop.run(serve()) [SERVE:L176] [uvloop executes MAX's serve coroutine]
        +-- async with lifespan() [API:L117] [MAX lifecycle]  call site [SERVE:L144]
        |   +-- set up media storage and telemetry; these live with the serving context.
        |   +-- ZmqModelWorkerInterface(task, context_type, request_queue_size) [ZMQ:L298] [MAX / ZeroMQ IPC]  call site [API:L165]
        |   |   creates model_worker_interface: request, response and cancellation queue configs.
        |   |   request_queue_size comes from settings.max_queue_size (backpressure limit).
        |   +-- async with start_model_worker() [WORKER:L759]
        |   |   +-- proc.start(ModelWorker(), alive, factory, config, ...) [WORKER:L801] [multiprocessing "spawn"; separate Python process]
        |   |   |   => WORKER runs QnA 1.1.5 / 1.1.6, including factory, model and graph initialization.
        |   |   +-- await proc.ready(alive, timeout=...) [WORKER:L815]
        |   |   |   MAIN waits; worker sets the Event only after initialization and scheduler setup.
        |   |   +-- ZmqModelWorkerInterface.model_worker_proxy() [ZMQ:L330]  call site [WORKER:L823]
        |   |   |   yields model_worker : ZmqModelWorkerProxy on the API side.
        |   |   +-- await ZmqModelWorkerProxy.wait_until_connected() [ZMQ:L87]  call site [WORKER:L827]
        |   |       waits for request PUSH/PULL handshake, then yields proxy to lifespan.
        |   +-- TokenGeneratorPipeline(tokenizer=..., model_worker=...) [API:L204] [MAX API frontend; MAX construction site]
        |   |   owns tokenizer + worker proxy; links HTTP handling to model-worker communication.
        |   +-- yield -> startup complete; serve coroutine resumes.
        +-- await server.serve(sockets=sockets) [SERVE:L147] [Uvicorn; MAX call site]
            accepts HTTP only after successful worker readiness + IPC connection.

MESSAGE DIRECTIONS AFTER STARTUP [ZMQ = ZeroMQ messaging, not a checkpoint database]
API proxy: request PUSH  --------> WORKER queues: request PULL -> scheduler
API proxy: cancel  PUSH  --------> WORKER queues: cancel  PULL
API proxy: response PULL <-------- WORKER queues: response PUSH <- scheduler results
HTTP request handling, proxy activity and worker execution use different async/process stacks.
```

## QnA 1.1.3 · Tokenizer assets -> TextTokenizer object -> serving factory

```text
MAIN / API PROCESS | Config already resolved; no model-factory invocation yet.
|
+-- serve_api_server_and_model_worker() [SERVE:L60]
    +-- PipelineRegistry.retrieve_factory() [REGISTRY:L711]
        |   input: pipeline_config + TEXT_GENERATION task.
        +-- PipelineRegistry.architecture_for_config() [REGISTRY:L496] -> PipelineRegistry.retrieve_architecture() [REGISTRY:L471]
        |   +-- ArchLookup.find() [LOOKUP:L947] -> ArchLookup.resolve() [LOOKUP:L930]
        |       returns arch: llama_arch : SupportedArchitecture [LLAMA_ARCH:L28], with tokenizer=TextTokenizer.
        +-- MemoryEstimator.plan() [MEMORY:L146] -> memory_plan.planned_max_length becomes tokenizer's max_length.
        +-- arch.tokenizer(...) [REGISTRY:L957] [dynamic class selection]
        |   +-- TextTokenizer.__init__(model_path, pipeline_config, ...) [TOKENIZER:L618] [MAX wrapper]
        |       |   input: model_path, revision, max_length, trust_remote_code, pipeline_config.
        |       +-- AutoTokenizer.from_pretrained(model_path, ...) [TOKENIZER:L633] [Transformers; MAX call site]
        |           reads: vocabulary/tokenization rules, tokenizer config and special tokens.
        |           storage: reuse Hub disk cache or fetch missing tokenizer assets.
        |           returns: Transformers tokenizer, retained as the MAX wrapper's delegate.
        +-- functools.partial(pipeline_class, **factory_kwargs) [REGISTRY:L994]
            binds this tokenizer together with model/config/adapter/memory-plan arguments.
            returns RetrievedPipeline: tokenizer object + deferred factory + memory_plan.

OBJECT HANDOFF
tokenizer : TextTokenizer -> pipeline_settings.tokenizer -> API TokenGeneratorPipeline
                         -> bound factory -> worker's TextGenerationPipeline
Role: encode prompt text / interpret token IDs; no 7B tensor allocation in AutoTokenizer.
Timing: tokenizer assets can load before FastAPI/Uvicorn assembly and before worker spawn.
```

## QnA 1.1.4 · GenerationConfig defaults -> MAX sampling settings

```text
MAIN / API PROCESS | Same helper can run for several MAXModelConfig instances.
|
+-- Possible callers of _build_model_config() [MODEL_CFG:L155]
|   +-- argument probe: PipelineArgs.from_flat_kwargs() [ARGS:L459] -> ModelManifest.from_model_path() [MANIFEST:L382]
|   +-- task inference, if UNDEFINED: PipelineArgs.main_architecture_name [property] [ARGS:L439] -> MAXModelConfig.from_pipeline_args() [MODEL_CFG:L1154]
|   +-- full serving config: PipelineConfig.from_args() [CONFIG:L1482] -> MAXModelConfig.from_pipeline_args() [MODEL_CFG:L1154]
|
+-- _build_model_config() [MODEL_CFG:L155] [MAX configuration]
    +-- MAXModelConfig._populate_generation_config() [MODEL_CFG:L1087]
        |   Skip work if generation state is already populated or no model path is supplied.
        +-- MAXModelConfig._make_generation_config() [MODEL_CFG:L1377]
            +-- TRY GenerationConfig.from_pretrained(repo_id, **kwargs) [MODEL_CFG:L1390] [Transformers; MAX call site]
            |   input: repo ID + revision + trust setting + optional subfolder.
            |   reads optional generation_config.json via Hub/cache.
            |   returns generation defaults; examples include temperature, top_p and token IDs.
            +-- EXCEPT loading error: debug log + GenerationConfig() [fallback] [MODEL_CFG:L1400]
                returns a default Transformers GenerationConfig object.

CONTINUE in PipelineConfig.from_args() [CONFIG:L1482]
+-- MAXModelConfig.sampling_params_defaults [property] [MODEL_CFG:L1402]
|   selects only fields supported by MAX's sampling-defaults schema.
+-- SamplingConfig.from_generation_config_sampling_defaults(...) [CONFIG:L1532] [MAX merge; call site]
    combines supported checkpoint defaults with explicitly set sampling fields.
    returns the SamplingConfig used by the serving pipeline.

WORKER PROCESS | Deserialization has its own entry path, not the API's live Python stack.
+-- MAXModelConfig.__setstate__() [MODEL_CFG:L1024]
    +-- rebuild repo handles -> populate HF config -> MAXModelConfig._populate_generation_config() [MODEL_CFG:L1087]
        The worker has its own HF-config process cache; disk assets may already be cached.

OBJECT DISTINCTION
GenerationConfig = generation defaults. Transformers LlamaConfig = model metadata.
MAX Llama3Config = architecture/runtime configuration. None of these is a weight tensor file.
```

## QnA 1.1.5 · Worker startup -> weight paths -> loaded model -> readiness

```text
HANDOFF FROM MAIN: deferred factory in pipeline_settings -> start_model_worker -> spawn.
This is a NEW PROCESS, not a synchronous child frame of the API's Python stack.
|
+-- During spawn deserialization: MAXModelConfig.__setstate__() [MODEL_CFG:L1024]
|   rebuilds repo/HF/generation config state in the worker.
|
+-- ModelWorker.__call__() [WORKER:L680] [MAX model worker]
    +-- uvloop executes ModelWorker.run() [WORKER:L338]
        +-- model_factory() [WORKER:L444] [the previously deferred functools.partial; MAX call site]
        |   +-- TextGenerationPipeline.__init__() [TEXT:L208] [MAX text pipeline]
        |       |   input: pipeline_config, Llama3Model class, tokenizer, adapters, memory_plan.
        |       |   creates devices/session, configures runtime and builds the model before returning.
        |       +-- InferenceSession(devices=..., ...) [ENGINE:L664] [MAX engine session]
        |       |   owns runtime/compilation context for the selected devices.
        |       +-- MAXModelConfig.resolved_weight_paths() [MODEL_CFG:L1563]  call site [TEXT:L293]
        |       |   +-- local repository -> use local checkpoint paths.
        |       |   +-- online repository -> download_weight_files() [WEIGHTS:L239]  call site [MODEL_CFG:L1594]
        |       |       |   input: weights repo ID, selected filenames, revision.
        |       |       +-- IF all filenames are direct local paths -> return them.
        |       |       +-- ELSE huggingface_hub.try_to_load_from_cache(...) [WEIGHTS:L285] [Hugging Face Hub; MAX call site]
        |       |       |   if every selected file is cached -> return cache paths; no transfer.
        |       |       +-- ELSE thread_map schedules per-file download callbacks
        |       |           +-- _hf_hub_download_with_retry() [WEIGHTS:L218]
        |       |               +-- huggingface_hub.hf_hub_download(**kwargs) [WEIGHTS:L226] [Hugging Face Hub; MAX call site]
        |       |                   cache/fetch selected files; retry the documented incomplete-file race.
        |       |   returns weight_paths : list[Path]. Header-only weights use a separate branch.
        |       +-- load_weights(weight_paths) [LOAD:L24]  call site [TEXT:L310]
        |       |   returns a format-specific Weights provider; tensor consumption follows model loading.
        |       +-- Llama3Model.__init__() [MODEL:L334] [selected pipeline_model class]  call site [TEXT:L305]
        |       |   +-- LlamaModelBase.__init__() [MODEL:L153]
        |       |       load state -> finalize config -> trace/compile/init; expanded in QnA 1.1.6.
        |       +-- finish KV-cache manager / sampler setup -> pipeline : TextGenerationPipeline.
        +-- configured warmup / graph-capture preparation completes.
        +-- ZmqModelWorkerInterface.model_worker_queues() [ZMQ:L320] [MAX / ZeroMQ IPC]  call site [WORKER:L554]
        |   yields WorkerQueues: request PULL, response PUSH, cancel PULL endpoints.
        +-- load_scheduler() [SCHED:L80]  call site [WORKER:L558]
        |   input: initialized pipeline + pipeline_config + Settings + worker queues + memory_plan.
        |   output: scheduler object that batches and executes admitted work.
        +-- scheduler loop:
            +-- alive.set() [WORKER:L650] [readiness Event, then heartbeat on subsequent iterations]
            |   allows MAIN's proc.ready wait to finish; the IPC connection wait is still required.
            +-- scheduler.run_iteration() [WORKER:L671] [scheduler call site]
                manages pending work; backs off when no progress is possible.

TIMING: metadata/tokenizer downloads can occur earlier; large weight resolution is in this factory.
        Factory + warmup + scheduler setup precede readiness. HTTP listening follows MAIN's waits.
```

## QnA 1.1.6 · Llama3Config initialization -> weight-aware finalization -> graph compilation

```text
THREE INITIALIZATION USES | Each creates a config for a different consumer.
The finalization below mutates a Python model-config object; it is not graph compilation.
|
+-- A. MAIN / API: plan memory before spawning the model worker
|   +-- PipelineRegistry.retrieve_factory() [REGISTRY:L711]
|       +-- MemoryEstimator.plan() [MEMORY:L146]  call site [REGISTRY:L795]
|           +-- Llama3Config.initialize() [LLAMA_CFG:L222] [arch.config = Llama3Config]  call site [MEMORY:L189]
|               input: HF fields + MAX device/dtype/KV/length settings.
|               output: arch_config for the memory planner; no loaded model state_dict here.
|
+-- B. WORKER: construct the pipeline model
    +-- ModelWorker.run() [WORKER:L338] -> model_factory() [WORKER:L444] -> TextGenerationPipeline.__init__() [TEXT:L208]
        +-- Llama3Model.__init__() [MODEL:L334] -> LlamaModelBase.__init__() [MODEL:L153]
            +-- PipelineModelWithKVCache.__init__() [BASE_MODEL:L1017]
            |   +-- PipelineModel.__init__() [BASE_MODEL:L553]
            |       +-- Llama3Config.initialize() [LLAMA_CFG:L222] [model_config_cls = Llama3Config]  call site [BASE_MODEL:L581]
            |           creates self.arch_config using the model's runtime sequence-length bound.
            +-- LlamaModelBase.load_model() [MODEL:L233]
                +-- LlamaModelBase.load_graphs() [MODEL:L254] [standard Llama3Model GraphSpec route]
                    +-- LlamaModelBase._load_state_dict() [MODEL:L283]
                    |   +-- PipelineModelWithKVCache._load_state_dict() [BASE_MODEL:L1090]
                    |       parses the Weights provider with the selected adapter -> state_dict.
                    +-- LlamaModelBase._create_model_config() [MODEL:L228]
                    |   +-- LlamaModelBase._finalized_config() [MODEL:L209]
                    |       +-- C. Llama3Config.initialize() [LLAMA_CFG:L222]  call site [MODEL:L213]
                    |       |   creates a fresh config using ModelLoadContext's pipeline/length settings.
                    |       +-- Llama3Config.finalize() [LLAMA_CFG:L397]  call site [MODEL:L216]
                    |           input: HF config + loaded state_dict + norm/bias/output policies.
                    |           work: normalize weight names for inspection; resolve quantization,
                    |                 normalization dtype, tied embeddings and projection choices.
                    |           effect: finishes this config in place; _finalized_config returns it.
                    +-- initialize distributed runtime as required by the selected device layout.
                    +-- LlamaModelBase.graph_specs() [MODEL:L287]
                    |   +-- one-device path: LlamaModelBase._single_device_spec() [MODEL:L316]
                    |       constructs the Llama module, loads its state, and returns a decoder GraphSpec.
                    |       GraphSpec describes module + input types + weight registry; not executable yet.
                    +-- compile_and_init_graphs() [SPECS:L183] [MAX graph / engine boundary]  call site [MODEL:L274]
                        +-- GraphSpec.trace() [SPECS:L114]
                        |   +-- Llama3.trace() [LLAMA:L273]
                        |       +-- Graph.__init__() [GRAPH:L819]
                        |           +-- forward(...) [Llama3.trace callback] [LLAMA:L301] -> record the Llama operations symbolically.
                        |           +-- Graph.output() [GRAPH:L1396] -> finish and verify the graph output contract.
                        +-- InferenceSession.compile_reusing_mefs() [ENGINE:L897]
                        |   compile or reuse a compatible MEF-backed artifact (subject to session policy).
                        +-- InferenceSession.init_all() [ENGINE:L1160]
                            bind the weight registry -> executable engine models / compiled callables.

RETURN CHAIN: initialized decoder -> self.model on Llama3Model -> TextGenerationPipeline -> worker.
             Worker completes warmup/scheduler setup, signals ready; MAIN can then finish startup.

KEEP THESE THREE OPERATIONS SEPARATE
Llama3Config.finalize : derive weight-aware Python configuration before graph construction.
Graph.output         : finish/verify symbolic graph outputs and effects.
Compiler/init path   : lower or reuse an artifact, then bind weights and executable state.
```

## Source map

- CLI: [max/python/max/_entrypoints/pipelines.py](sources/max/python/max/_entrypoints/pipelines.py)
- FLAGS: [max/python/max/_entrypoints/cli/config.py](sources/max/python/max/_entrypoints/cli/config.py)
- SERVE: [max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py](sources/max/python/max/_entrypoints/cli/serve/serve_api_and_model_worker.py)
- ARGS: [max/python/max/pipelines/lib/pipeline_args.py](sources/max/python/max/pipelines/lib/pipeline_args.py)
- MANIFEST: [max/python/max/pipelines/lib/model_manifest.py](sources/max/python/max/pipelines/lib/model_manifest.py)
- CONFIG: [max/python/max/pipelines/lib/config/config.py](sources/max/python/max/pipelines/lib/config/config.py)
- MODEL_CFG: [max/python/max/pipelines/lib/config/model_config.py](sources/max/python/max/pipelines/lib/config/model_config.py)
- HF_CFG: [max/python/max/pipelines/lib/_hf_config.py](sources/max/python/max/pipelines/lib/_hf_config.py)
- PACKAGE: [max/python/max/pipelines/__init__.py](sources/max/python/max/pipelines/__init__.py)
- ARCHS: [max/python/max/pipelines/architectures/__init__.py](sources/max/python/max/pipelines/architectures/__init__.py)
- REGISTRY: [max/python/max/pipelines/lib/registry.py](sources/max/python/max/pipelines/lib/registry.py)
- LOOKUP: [max/python/max/pipelines/lib/arch_lookup.py](sources/max/python/max/pipelines/lib/arch_lookup.py)
- LLAMA_ARCH: [max/python/max/pipelines/architectures/llama3/arch.py](sources/max/python/max/pipelines/architectures/llama3/arch.py)
- API: [max/python/max/serve/api_server.py](sources/max/python/max/serve/api_server.py)
- SETTINGS: [max/python/max/serve/config.py](sources/max/python/max/serve/config.py)
- WORKER: [max/python/max/serve/pipelines/model_worker.py](sources/max/python/max/serve/pipelines/model_worker.py)
- ZMQ: [max/python/max/serve/worker_interface/zmq_interface.py](sources/max/python/max/serve/worker_interface/zmq_interface.py)
- SCHED: [max/python/max/serve/scheduler/__init__.py](sources/max/python/max/serve/scheduler/__init__.py)
- TOKENIZER: [max/python/max/pipelines/lib/tokenizer.py](sources/max/python/max/pipelines/lib/tokenizer.py)
- MEMORY: [max/python/max/pipelines/lib/memory_estimation.py](sources/max/python/max/pipelines/lib/memory_estimation.py)
- TEXT: [max/python/max/pipelines/lib/pipeline_variants/text_generation.py](sources/max/python/max/pipelines/lib/pipeline_variants/text_generation.py)
- WEIGHTS: [max/python/max/pipelines/weights/hf_utils.py](sources/max/python/max/pipelines/weights/hf_utils.py)
- LOAD: [max/python/max/graph/weights/load.py](sources/max/python/max/graph/weights/load.py)
- MODEL: [max/python/max/pipelines/architectures/llama3/model.py](sources/max/python/max/pipelines/architectures/llama3/model.py)
- BASE_MODEL: [max/python/max/pipelines/lib/interfaces/pipeline_model.py](sources/max/python/max/pipelines/lib/interfaces/pipeline_model.py)
- LLAMA_CFG: [max/python/max/pipelines/architectures/llama3/model_config.py](sources/max/python/max/pipelines/architectures/llama3/model_config.py)
- LLAMA: [max/python/max/pipelines/architectures/llama3/llama3.py](sources/max/python/max/pipelines/architectures/llama3/llama3.py)
- SPECS: [max/python/max/pipelines/lib/graph_specs.py](sources/max/python/max/pipelines/lib/graph_specs.py)
- GRAPH: [max/python/max/graph/graph.py](sources/max/python/max/graph/graph.py)
- ENGINE: [max/python/max/engine/api.py](sources/max/python/max/engine/api.py)
