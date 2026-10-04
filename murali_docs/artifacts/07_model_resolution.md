# Phase 7: Model Registry, Configuration, and Weights

## Resolution Chain

`SOURCE + CPU-RUN`

```text
HF repo/config.json
  architectures[0] = LlamaForCausalLM
               |
               v
PipelineRegistry / ArchitectureLookup
               |
               v
llama_arch
  task       Text generation       tokenizer  TextTokenizer
  context    TextContext           model      Llama3Model
  batching   Llama3BatchProcessor  memory     PagedMemoryPlanner
  weights    safetensors -> Llama adapter
```

```mermaid
sequenceDiagram
    autonumber
    participant CLI as max serve
    participant HF as HF config/weights
    participant R as PipelineRegistry
    participant A as llama_arch
    participant W as Weight adapter
    participant C as Llama3Config
    participant F as Pipeline factory

    CLI->>HF: read config.json
    HF-->>CLI: architectures=[LlamaForCausalLM]
    CLI->>R: retrieve architecture + task
    R->>A: materialize registered entry
    A-->>R: tokenizer/context/model/batcher/planner
    CLI->>HF: inspect weight format
    CLI->>W: select safetensors adapter
    W-->>CLI: MAX state-dict names/data
    CLI->>C: initialize + finalize from HF geometry
    C-->>F: resolved architecture config
    F-->>CLI: text-generation pipeline components
```

## Two-Model Resolution

| Property          | Model A                                | Model B                                      |
|-------------------|----------------------------------------|----------------------------------------------|
| ID                | `modularai/SmolLM-135M-Instruct-FP32`  | `meta-llama/Llama-3.1-8B-Instruct`           |
| Evidence          | `CPU-RUN` cached snapshot              | `SOURCE + BOUNDARY`; gated checkpoint absent |
| HF architecture   | `LlamaForCausalLM`                     | `LlamaForCausalLM`                           |
| registry entry    | `llama_arch`                           | `llama_arch`                                 |
| task              | `TEXT_GENERATION`                      | `TEXT_GENERATION`                            |
| tokenizer/context | `TextTokenizer` / `TextContext`        | same                                         |
| model/batcher     | `Llama3Model` / `Llama3BatchProcessor` | same                                         |
| memory planner    | `PagedMemoryPlanner`                   | same                                         |
| weight route      | safetensors adapter                    | safetensors adapter                          |
| execution lane    | CPU float32                            | future GPU BF16/FP16 lane                    |

```text
same server + scheduler + pipeline shell
                    |
          +---------+---------+
          |                   |
          v                   v
 Model A config/weights   Model B config/weights
 30 small blocks         32 large blocks
 CPU mechanics           GPU production study
```

```mermaid
flowchart TD
    O[OpenAI + scheduler + text pipeline] --> R[Llama registry entry]
    R --> A[Model A config + FP32 weights]
    R --> B[Model B config + BF16/FP16 weights]
    A --> CPU[CPU-RUN]
    B --> GPU[GPU-LAB]
```

## Model A Geometry

`CPU-RUN`

| Field              |   Value | Drives                           |
|--------------------|--------:|----------------------------------|
| layers             |      30 | transformer repetition, KV depth |
| hidden size        |     576 | residual/linear shapes           |
| intermediate size  |    1536 | MLP shapes                       |
| Q heads / KV heads |   9 / 3 | GQA ratio `3:1`                  |
| head dimension     |      64 | attention vector width           |
| vocabulary         |  49,152 | embedding/logit width            |
| max positions      |   2,048 | configured context ceiling       |
| RoPE theta         |  10,000 | rotary frequencies               |
| weight dtype       | float32 | storage and kernel dtype         |

```text
one unsharded KV token
  K + V
  = 2 x layers x KV heads x head_dim x dtype bytes
  = 2 x 30 x 3 x 64 x 4
  = 46,080 bytes = 45 KiB/token
```

```mermaid
flowchart LR
    L[30 layers] --> K[KV bytes/token]
    H[3 KV heads] --> K
    D[64 values/head] --> K
    T[FP32: 4 bytes] --> K
    K --> V[46,080 bytes]
```

## Model B Geometry Boundary

`BOUNDARY + GPU-LAB`

The gated `meta-llama` snapshot is not cached. A public Modular GGUF mirror
reports the canonical Llama 3.1 8B geometry: 32 layers, hidden 4096, 32 Q
heads, 8 KV heads, vocabulary 128,256, and context 131,072. Confirm the exact
gated revision and encoding on the GPU host before measurement.

```text
local source proves implementation selection
        |
        v
GPU host resolves exact model revision + dtype + weights
        |
        v
memory plan -> compile -> initialize -> execute/profile
```

```mermaid
stateDiagram-v2
    [*] --> SourceResolved
    SourceResolved --> CheckpointNeeded
    CheckpointNeeded --> RevisionPinned: authorized GPU host
    RevisionPinned --> ConfigResolved
    ConfigResolved --> Compiled
    Compiled --> Measured
```

## Weight-Name Adaptation

`CPU-RUN + SOURCE`: 272 tensor headers inspected; payloads were not loaded.

```text
checkpoint key                                 MAX state-dict key
model.embed_tokens.weight                  -> embed_tokens.weight
model.layers.0.input_layernorm.weight      -> layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight     -> layers.0.self_attn.q_proj.weight
model.layers.0.mlp.gate_proj.weight        -> layers.0.mlp.gate_proj.weight
model.norm.weight                          -> norm.weight

rule 1: replace "model." with ""
rule 2: replace "g_idx" with "perm_idx" for GPTQ
```

```mermaid
flowchart LR
    ST[Safetensor header name] --> M1[Remove model prefix]
    M1 --> M2[Map g_idx to perm_idx]
    M2 --> Q{GPTQ?}
    Q -->|yes| QA[argsort/select/cast rules]
    Q -->|no| D[Dtype cast if configured]
    QA --> SD[MAX state dict]
    D --> SD
    SD --> G[Graph weight registry]
```

## Configuration Fan-Out

`SOURCE`

```text
HF config
  +-> graph dimensions: hidden, intermediate, layers, vocab
  +-> attention: Q/KV heads, head_dim, RoPE
  +-> KV geometry: layers x KV heads x head_dim x dtype
  +-> scheduler limits: max sequence after memory planning
  +-> kernel specialization: dtype/head_dim/target/layout
```

```mermaid
flowchart TD
    H[HF config] --> G[Graph shapes]
    H --> A[Attention + RoPE]
    H --> K[KV parameters]
    H --> M[Memory plan]
    H --> D[Kernel specialization]
    W[CLI encoding/device overrides] --> C[Llama3Config.finalize]
    H --> C
    C --> G
    C --> A
    C --> K
    C --> M
    C --> D
```

## Reproduce

```bash
murali_docs/labs/graph/run_model_resolution_probe.sh \
  > /tmp/phase7_resolution.json
```

## Source Pins

| Concern               | Source                                                                                         |
|-----------------------|------------------------------------------------------------------------------------------------|
| registry lookup       | [`registry.py`](../../max/python/max/pipelines/lib/registry.py)                                |
| architecture records  | [`arch_lookup.py`](../../max/python/max/pipelines/lib/arch_lookup.py)                          |
| Llama registration    | [`arch.py`](../../max/python/max/pipelines/architectures/llama3/arch.py)                       |
| config finalization   | [`model_config.py`](../../max/python/max/pipelines/architectures/llama3/model_config.py)       |
| weight conversion     | [`weight_adapters.py`](../../max/python/max/pipelines/architectures/llama3/weight_adapters.py) |
| graph model selection | [`model.py`](../../max/python/max/pipelines/architectures/llama3/model.py)                     |
