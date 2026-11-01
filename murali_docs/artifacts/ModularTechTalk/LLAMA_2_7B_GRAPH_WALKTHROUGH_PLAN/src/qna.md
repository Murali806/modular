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

{{nested:qna-1-1}}

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
