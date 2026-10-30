# MAX Model Bring-Up Workflow

Evidence: `SOURCE`

`model-bringup-workflow.mdx` explains how to make a model with a new architecture run on MAX.

This is different from serving a fine-tuned model on an already-supported architecture.

```text
Supported architecture + new weights
        |
        v
Usually only configure the weights
```

```text
New or different architecture
        |
        v
Implement and register the architecture
        |
        v
This is model bring-up
```

## Complete workflow

```text
Hugging Face checkpoint
        |
        +-- config.json
        |       |
        |       v
        |   architecture name
        |
        +-- model weights
        |       |
        |       v
        |   checkpoint parameter names
        |
        v
Choose closest MAX reference architecture
        |
        v
Map configuration fields
        |
        v
Translate checkpoint weight names
        |
        v
Connect the KV cache
        |
        v
Override or extend model modules
        |
        v
Register SupportedArchitecture
        |
        v
Compile and serve
        |
        v
Compare results with PyTorch
```

The goal is:

```text
Hugging Face model
        |
        v
MAX architecture package
        |
        v
MAX compiled model
        |
        v
Production inference
```

## What is model bring-up?

A model architecture is the computation performed by the model.

```text
Architecture
    |
    +-- embedding
    +-- attention
    +-- normalization
    +-- feed-forward network
    +-- residual connections
    +-- output projection
```

Model bring-up means translating the original implementation into MAX:

```text
PyTorch/Hugging Face model
        |
        | translate
        v
MAX nn.Module implementation
        |
        | compile
        v
MAX runtime model
```

You need model bring-up when the model changes something structural, such as:

```text
- a new attention mechanism;
- a new normalization layout;
- a new MoE routing system;
- a different KV-cache structure;
- a new positional encoding;
- a different tensor-parallel layout.
```

## Standard architecture directory

A typical MAX architecture package looks like this:

```text
your_model/
├── layers/
├── __init__.py
├── arch.py
├── batch_processor.py
├── model_config.py
├── model.py
├── your_model.py
└── weight_adapters.py
```

Visual responsibility map:

```text
your_model/
      |
      +-- model_config.py
      |       Reads config.json
      |       Produces typed MAX configuration
      |
      +-- weight_adapters.py
      |       Renames/transforms checkpoint weights
      |
      +-- your_model.py
      |       Defines the model's module tree
      |
      +-- layers/
      |       Defines custom building blocks
      |
      +-- model.py
      |       Connects the module to the MAX pipeline
      |
      +-- batch_processor.py
      |       Prepares scheduled batch inputs
      |
      +-- arch.py
      |       Registers the architecture with MAX
      |
      +-- __init__.py
              Exposes ARCHITECTURES
```

## What each file does

### `layers/`

Contains custom components:

```text
layers/
├── custom_attention.py
├── custom_mlp.py
└── custom_norm.py
```

Each component is usually a `max.experimental.nn.Module`.

```text
custom_attention.py
        |
        v
CustomAttention(Module)
        |
        v
Tensor -> Tensor
```

### `your_model.py`

Defines the root model and connects the layers:

```text
YourModel
    |
    +-- embedding
    +-- transformer layers
    |       |
    |       +-- attention
    |       +-- normalization
    |       +-- MLP
    |
    +-- final normalization
    +-- output projection
```

### `model_config.py`

Translates Hugging Face configuration into the fields MAX modules expect.

```text
Hugging Face config.json
        |
        v
YourConfig.initialize()
        |
        v
Typed MAX configuration
        |
        v
MAX modules read config.hidden_size,
config.num_layers, config.kv_params, ...
```

### `weight_adapters.py`

Translates checkpoint names into MAX module names.

```text
Checkpoint name
    |
    | rename/reshape/cast
    v
MAX parameter name
```

### `model.py`

Connects the architecture to the pipeline system.

It usually:

```text
- selects the configuration class;
- creates the root module;
- places it on CPU/GPU/device mesh;
- connects KV-cache parameters;
- invokes module compilation.
```

### `arch.py`

Registers the architecture:

```text
MAX registry
    |
    v
architecture name
    |
    +-- config class
    +-- pipeline model
    +-- weight adapter
    +-- tokenizer
    +-- supported encodings
    +-- memory planner
    +-- batch processor
```

### `__init__.py`

Exposes the registry entry:

```python
from .arch import your_model_arch

ARCHITECTURES = [your_model_arch]
```

Without this export, MAX cannot discover the architecture package.

## Step 1: Choose a reference architecture

Most transformer models are variations of existing families.

```text
Your model
    |
    +-- standard dense transformer?
    |       -> Llama 3 reference
    |
    +-- sliding-window + full attention?
    |       -> OLMo 3 reference
    |
    +-- local/global attention + QK norm?
    |       -> Gemma 3 reference
    |
    +-- fused qkv_proj weights?
    |       -> Phi 3 reference
    |
    +-- sparse MoE?
    |       -> GPT-OSS or DeepSeek reference
    |
    +-- MLA?
    |       -> DeepSeek reference
    |
    +-- Mamba + attention?
            -> Nemotron-H reference
```

The recommended strategy is:

```text
Find the closest working MAX architecture
        |
        v
Reuse everything that matches
        |
        v
Override only the differences
```

This lets you inherit:

```text
- tested attention modules;
- KV-cache wiring;
- optimized kernels;
- batching behavior;
- quantization support;
- distributed execution patterns.
```

## Step 2: Write MAX modules

MAX modules use `max.experimental.nn.Module`.

Conceptually:

```python
class YourModel(Module):
    def __init__(self, config, kv_params):
        self.embed_tokens = Embedding(...)
        self.layers = ModuleList(...)
        self.norm = RMSNorm(...)
        self.lm_head = Linear(...)

    def forward(self, tokens, ...):
        x = self.embed_tokens(tokens)
        x = self.layers(x, ...)
        x = self.norm(x)
        return self.lm_head(x)
```

Visual flow:

```text
tokens
   |
   v
Embedding
   |
   v
Transformer layer 0
   |
   v
Transformer layer 1
   |
   v
...
   |
   v
Final normalization
   |
   v
LM head
   |
   v
logits
```

Modules use Tensor values:

```text
Tensor input
      |
      v
forward()
      |
      v
Tensor output
```

Operations normally come from:

```python
from max.experimental import functional as F
```

For example:

```python
x = F.gelu(x)
x = x @ weight
```

## Step 3: Understand the pipeline loading sequence

`ModuleV3PipelineModelWithKVCache.load_model()` performs the main setup:

```text
load_model()
    |
    +-- 1. Load checkpoint
    |
    +-- 2. Apply weight adapter
    |
    +-- 3. Create model config
    |
    +-- 4. Prepare/cast/rewrite state dict
    |
    +-- 5. Initialize distributed runtime
    |
    +-- 6. Build module inside F.lazy()
    |
    +-- 7. Compile module with weights
    |
    v
Compiled serving model
```

More detailed:

```text
checkpoint
    |
    v
raw state_dict
    |
    | weight adapter
    v
MAX-compatible state_dict
    |
    | config initialization/finalization
    v
YourConfig
    |
    | instantiate module
    v
YourModel(config, kv_params)
    |
    | compile(input_types, weights=state_dict)
    v
compiled MAX model
```

The module is created under `F.lazy()`:

```python
with F.lazy():
    model = YourModel(config, kv_params)
```

This means:

```text
Construct module structure
        |
        v
Do not unnecessarily initialize/execute every tensor
        |
        v
Load the real checkpoint weights
        |
        v
Compile the prepared module
```

## Step 4: Map configuration fields

Suppose the Hugging Face configuration contains:

```json
{
  "architectures": ["YourModelForCausalLM"],
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_hidden_layers": 32,
  "vocab_size": 256000,
  "logit_scale": 0.0625
}
```

The MAX configuration translates these values:

```text
hidden_size              -> config.hidden_size
num_attention_heads     -> config.num_attention_heads
num_key_value_heads     -> config.num_key_value_heads
num_hidden_layers       -> config.num_hidden_layers
vocab_size               -> config.vocab_size
logit_scale              -> config.logits_scaling
```

The example has a representation difference:

```text
Hugging Face:
logit_scale = 0.0625

MAX expects:
logits_scaling = 1 / 0.0625 = 16.0
```

The config adapter performs that conversion:

```text
raw config field
      |
      v
interpret model convention
      |
      v
MAX config field
```

Usually, extend an existing configuration:

```python
@dataclass(kw_only=True)
class YourConfig(Llama3Config):
    @classmethod
    def initialize(...):
        config = super().initialize(...)
        config.logits_scaling = 1.0 / logit_scale
        return config
```

Calling `super().initialize()` is important:

```text
Llama3Config.initialize()
        |
        +-- maps fields already understood by Llama 3
        |
        v
YourConfig.initialize()
        |
        +-- maps your model's differences
```

## Step 5: Connect the pipeline model to the config

Your pipeline model tells MAX which configuration class to use:

```python
class YourModel(Llama3Model):
    model_config_cls = YourConfig
    config_class = YourConfig
```

Visual relationship:

```text
config.json
    |
    v
YourConfig
    |
    +-- model dimensions
    +-- normalization settings
    +-- logits behavior
    +-- KV-cache parameters
    |
    v
YourModel
    |
    v
YourRootModule
```

`model_config_cls` is especially important for sizing the KV cache before the first request.

## Step 6: Map checkpoint weights

The module has expected parameter names:

```text
language_model.layers.0.self_attn.q_proj.weight
language_model.layers.0.self_attn.k_proj.weight
language_model.layers.0.self_attn.v_proj.weight
```

The checkpoint may use different names:

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
```

The adapter maps them:

```text
Hugging Face checkpoint
model.layers.0.self_attn.q_proj.weight
                |
                | rename
                v
MAX module
language_model.layers.0.self_attn.q_proj.weight
```

To discover the names expected by MAX:

```python
with F.lazy():
    model = YourRootModule(config, kv_params)

expected = {name for name, _ in model.parameters}
```

Then compare:

```text
expected MAX names
        versus
adapted checkpoint names
```

Compilation checks the result:

```text
Missing MAX parameter
        |
        v
KeyError

Wrong shape or dtype
        |
        v
ValueError

Checkpoint entry used by nobody
        |
        v
Ignored by compile()
```

Example:

```text
MAX expects:
layers.0.self_attn.q_proj.weight -> shape [4096, 4096]

Adapter provides:
layers.0.self_attn.q_proj.weight -> shape [4096, 4096]
                                      |
                                      v
                                   accepted
```

If the checkpoint contains:

```text
layers.0.self_attn.q_proj.weight -> shape [4096, 2048]
```

then:

```text
shape mismatch -> ValueError
```

Adapters may perform:

```text
- renaming;
- dtype conversion;
- tensor reshaping;
- layout conversion;
- dropping unused checkpoint entries;
- combining or splitting fused weights.
```

## Step 7: Connect the KV cache

Autoregressive generation repeatedly predicts the next token.

Without a KV cache:

```text
Request token 1 -> recompute token 1
Request token 2 -> recompute tokens 1 and 2
Request token 3 -> recompute tokens 1, 2, and 3
```

This repeats work.

With a KV cache:

```text
Token 1 -> calculate K/V and store them
Token 2 -> reuse token 1 K/V, calculate token 2 K/V
Token 3 -> reuse token 1 and 2 K/V, calculate token 3 K/V
```

Visual flow:

```text
Previous tokens
      |
      v
Key/value tensors
      |
      | store
      v
Paged KV cache
      |
      | read during next attention operation
      v
Attention output
```

MAX pre-allocates the cache before serving:

```text
KV-cache size depends on:

number of KV heads
        +
head dimension
        +
number of transformer layers
        +
maximum sequence length
        +
cache dtype
        |
        v
required memory
```

Important distinction:

```text
num_attention_heads
        |
        +-- total attention query heads

num_key_value_heads
        |
        +-- heads actually stored in the KV cache
```

For grouped-query attention:

```text
32 query heads
8 KV heads
```

MAX sizes the cache using the `8` KV heads, not `32` query heads.

### KV cache parameters

The config provides:

```text
kv_params
    |
    +-- number of KV heads
    +-- head dimension
    +-- number of layers
    +-- cache layout
    +-- cache dtype
    +-- device distribution
```

The configuration typically constructs these from:

```python
num_key_value_heads
head_dim
num_hidden_layers
devices
cache_dtype
```

### Passing the cache to the module

The compiled module receives KV inputs after the token inputs:

```text
forward(
    tokens,
    return_n_logits,
    input_row_offsets,
    *variadic_args
)
```

The flat inputs are reconstructed:

```text
flat KV inputs
      |
      v
kv_params.unflatten_...
      |
      v
PagedCacheValues
      |
      v
attention layers
```

Attention layers then use kernels to:

```text
- apply rotary position encoding;
- store new keys and values;
- read prior keys and values;
- run flash attention.
```

## Step 8: Override versus extend

There are two different kinds of changes.

### Small behavioral difference: override

If your model is almost identical to Llama 3:

```python
class YourModel(Llama3Model):
    attention_bias = True
```

Visual:

```text
Llama3Model
    |
    +-- inherited attention
    +-- inherited config handling
    +-- inherited KV cache
    +-- inherited compilation
    |
    +-- changed attention_bias
```

Use this when the structure remains the same.

### Structural difference: extend

If the computation graph is different, create or replace modules.

#### Llama pre-norm

```text
x
|
v
LayerNorm
|
v
Attention
|
+-- residual add with x
```

Equivalent form:

```python
h = x + self.self_attn(self.input_layernorm(x), ...)
```

#### OLMo-style post-norm

```text
x
|
v
Attention
|
v
LayerNorm
|
+-- residual add with x
```

Equivalent form:

```python
h = x + self.post_attention_layernorm(
    self.self_attn(x, ...)
)
```

#### Parallel attention and MLP

```text
             +--> Attention --+
             |                |
x -> LayerNorm                 +--> residual add
             |                |
             +--> MLP --------+
```

Equivalent form:

```python
normed = self.input_layernorm(x)
x = x + self.self_attn(normed, ...) + self.mlp(normed)
```

This is a structural change, so it belongs in the model module rather than only in configuration.

## Step 9: Implement custom layers

If MAX already provides the required building block, reuse it:

```text
RMSNorm
LayerNorm
Linear
Embedding
MLP
rotary embedding
attention kernels
```

If the model needs something new:

```text
layers/
├── custom_attention.py
├── custom_router.py
└── custom_norm.py
```

Each custom layer becomes a module:

```text
custom layer
      |
      v
Module.forward(...)
      |
      v
Tensor output
```

If an operation is written using lower-level `TensorValue` inputs, wrap it with `F.functional` so the module can use Tensor values:

```text
Tensor
  |
  v
F.functional wrapper
  |
  v
TensorValue-based graph operation
  |
  v
Tensor
```

## Step 10: Multi-GPU execution

For one device:

```text
model -> GPU:0
```

For multiple devices, create a `DeviceMesh`:

```text
GPU:0   GPU:1   GPU:2   GPU:3
  \       |       |       /
       DeviceMesh
```

Conceptual code:

```python
mesh = DeviceMesh(
    tuple(self.devices),
    (len(self.devices),),
    ("tp",),
)

with default_device(mesh):
    model = Gemma3(config, kv_params)
```

The mesh defines how tensors and weights are distributed.

Common parallel layers:

```text
ColumnParallelLinear
RowParallelLinear
VocabParallelEmbedding
```

Visual example:

```text
Large linear weight
        |
        | split across devices
        v
GPU:0   GPU:1   GPU:2   GPU:3
```

Collective operations combine partial results:

```text
F.allreduce_sum()
F.allgather()
F.reduce_scatter()
F.transfer_to()
```

After implementing multi-GPU support, register:

```python
multi_gpu_supported = True
```

## Step 11: Register the architecture

`arch.py` connects all pieces:

```python
your_model_arch = SupportedArchitecture(
    name="YourModelForCausalLM",
    pipeline_model=YourModel,
    config=YourConfig,
    weight_adapters={...},
    tokenizer=TextTokenizer,
    memory_planner=PagedMemoryPlanner,
    batching=Llama3ModuleV3BatchProcessor,
)
```

Registry flow:

```text
config.json
    |
    | architectures[0]
    v
"YourModelForCausalLM"
    |
    | must match exactly
    v
SupportedArchitecture.name
    |
    v
MAX selects:
    +-- YourConfig
    +-- YourModel
    +-- weight adapter
    +-- tokenizer
    +-- batch processor
    +-- memory planner
```

The names must match:

```text
config.json:
architectures[0] = "YourModelForCausalLM"

arch.py:
name = "YourModelForCausalLM"
```

Finally expose it:

```python
from .arch import your_model_arch

ARCHITECTURES = [your_model_arch]
```

MAX discovers the package through this list.

## Step 12: Validate the model

Model bring-up is not complete merely because the model loads.

Validation flow:

```text
MAX model loads
      |
      v
MAX model generates text
      |
      v
Compare MAX logits with PyTorch
      |
      +-- match
      |     |
      |     v
      |  test full accuracy
      |
      +-- mismatch
            |
            v
      compare layer by layer
```

Numerical validation:

```text
same input tokens
       |
       +------------------+
       |                  |
       v                  v
PyTorch reference      MAX model
       |                  |
       v                  v
reference logits       MAX logits
       \                  /
        \                /
         +-- compare ---+
```

If the final logits differ, compare intermediate layers:

```text
embedding
    |
    v
layer 0
    |
    v
layer 1
    |
    v
layer 2
    |
    v
final logits
```

Find the first divergence:

```text
PyTorch: layer 0 matches
MAX:     layer 0 matches

PyTorch: layer 1 = A
MAX:     layer 1 = B
                  ^
                  |
          likely source of bug
```

After numerical parity:

```text
logits match
    |
    v
generation works
    |
    v
benchmark task accuracy
    |
    v
production readiness
```

## Final mental model

```text
1. Find the closest reference architecture
              |
              v
2. Map config.json into MAX config
              |
              v
3. Build the MAX module tree
              |
              v
4. Rename/transform checkpoint weights
              |
              v
5. Configure the KV cache
              |
              v
6. Override or extend structural differences
              |
              v
7. Add multi-GPU support if needed
              |
              v
8. Register SupportedArchitecture
              |
              v
9. Compile and serve
              |
              v
10. Compare against the original model
```

In one sentence:

> Model bring-up is the process of translating a model's configuration, modules, weights, cache behavior, and device strategy into a registered MAX architecture that produces the same results as the original implementation.
