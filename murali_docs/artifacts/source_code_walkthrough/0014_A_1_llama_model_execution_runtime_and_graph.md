# Complete Code Walk: Llama Model Runtime And Graph

Source:
[`model.py`](../../../max/python/max/pipelines/architectures/llama3/model.py#L45)

## `Llama3Inputs.buffers`

```python
@dataclass
class Llama3Inputs(ModelInputs):
    tokens: Buffer
    input_row_offsets: Buffer
    signal_buffers: list[Buffer]
    return_n_logits: Buffer
    data_parallel_splits: Buffer | Sequence[Sequence[int]] | None = None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        if self.data_parallel_splits is not None:
            if isinstance(self.data_parallel_splits, Buffer):
                splits_tensor = self.data_parallel_splits
            else:
                splits_array = np.concatenate(
                    [
                        np.array(split, dtype=np.int64)
                        for split in self.data_parallel_splits
                    ]
                )
                splits_tensor = Buffer.from_numpy(splits_array).to(
                    self.tokens.device
                )
            return (
                self.tokens,
                self.input_row_offsets,
                self.return_n_logits,
                splits_tensor,
                *(
                    tree.leaves(self.kv_cache_inputs)
                    if self.kv_cache_inputs is not None
                    else ()
                ),
            )

        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
        )
```

```text
data_parallel_splits present?
        |
    +---+---+
    |       |
   yes      no
    |       |
    |       +--> tokens
    |            row_offsets
    |            return_n_logits
    |            signal_buffers
    |            flattened KV inputs
    |
    +--> tokens
         row_offsets
         return_n_logits
         splits_tensor
         flattened KV inputs
```

## Runtime Hot Path

```python
def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
    assert isinstance(model_inputs, Llama3Inputs)
    assert model_inputs.kv_cache_inputs is not None
    model_outputs = self.model.execute(*model_inputs.buffers)

    assert self.batch_processor is not None
    return self.batch_processor.process_outputs(model_outputs)
```

```text
Llama3Inputs
     |
     | model_inputs.buffers
     v
ordered tuple[Buffer, ...]
     |
     | compiled self.model.execute(*buffers)
     v
raw graph outputs
     |
     | batch_processor.process_outputs(...)
     v
ModelOutputs consumed by text-generation sampler
```

## Model Configuration

```python
def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
    model_config = Llama3Config.initialize(
        self.pipeline_config, max_seq_len=self.max_seq_len
    )
    model_config.finalize(
        huggingface_config=self.huggingface_config,
        state_dict=state_dict,
        norm_method=self.norm_method,
        attention_bias=self.attention_bias,
        return_logits=self.return_logits,
        return_hidden_states=self.return_hidden_states,
    )
    return model_config
```

```text
pipeline config + max sequence length
             |
             | Llama3Config.initialize(...)
             v
initial model_config
             |
             | finalize(HF config, weights, norm, outputs)
             v
graph-ready Llama3Config
```

## Parallelism Dispatch During Graph Build

```python
def _build_graph_for_compile(
    self,
    session: InferenceSession,
    state_dict: dict[str, Any],
    model_config: Any,
) -> tuple[Graph, dict[str, Any]]:
    del session
    assert isinstance(model_config, Llama3Config)
    if model_config.data_parallel_degree > 1:
        return create_data_parallel_graph(
            model_config, self.kv_params, state_dict
        )
    if len(self.devices) > 1:
        return self._build_tensor_parallel_graph_for_compile(
            state_dict, model_config
        )
    return self._build_single_device_graph_for_compile(
        state_dict, model_config
    )
```

```text
model_config.data_parallel_degree > 1?
        |
    +---+---+
    |       |
   yes      no
    |       |
    v       | len(devices) > 1?
DP graph    +-- yes --> tensor-parallel graph
            +-- no  --> single-device graph
```

## Single-Device Graph Construction

```python
def _build_single_device_graph_for_compile(
    self,
    state_dict: dict[str, Any],
    model_config: Any,
) -> tuple[Graph, dict[str, Any]]:
    assert isinstance(model_config, Llama3Config)
    single_model: Llama3 = Llama3(model_config)

    single_model.load_state_dict(
        state_dict,
        override_quantization_encoding=True,
        weight_alignment=1,
        strict=False,  # TODO(MODELS-550) `rope_freqs.weight` not used
    )
    weights_registry = single_model.state_dict()

    with Graph(
        "llama3",
        input_types=single_model.input_types(self.kv_params),
    ) as graph:
        (
            tokens,
            input_row_offsets,
            return_n_logits,
            *rest,
        ) = graph.inputs
        kv_collections = self._unflatten_kv_inputs(rest)
        outputs = single_model(
            tokens.tensor,
            kv_collections[0],
            return_n_logits.tensor,
            input_row_offsets.tensor,
        )
        graph.output(*outputs)
        return graph, weights_registry
```

```text
state_dict + Llama3Config
        |
        +-- instantiate Llama3
        +-- load_state_dict(...)
        +-- declare Graph input types
        +-- split graph.inputs into tokens/offsets/logit-count/KV
        +-- call single_model(...)
        +-- graph.output(*outputs)
        v
(Graph, weights_registry)
```

## Pipeline Model Initialization

```python
class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    config_class: type[Llama3Config] = Llama3Config
    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
```

```text
Llama3Model.__init__
        |
        | LlamaModelBase.__init__
        v
GraphPipelineModelWithKVCache initialization
        |
        +-- prepare state_dict/config
        +-- build and compile selected graph
        +-- load weights
        +-- create batch processor/KV integration
        v
runtime-ready compiled model wrapper
```

## Complete Runtime Buffer And Output Values

```python
model_inputs = {
    "tokens": {
        "shape": [5],
        "dtype": "int64",
        "device": "gpu:0",
        "values": [101, 102, 103, 201, 202],
    },
    "input_row_offsets": {
        "shape": [3],
        "dtype": "uint32",
        "device": "gpu:0",
        "values": [0, 3, 5],
    },
    "return_n_logits": {
        "shape": [1],
        "dtype": "int64",
        "device": "cpu",
        "values": [1],
    },
    "data_parallel_splits": {
        "shape": [3],
        "dtype": "int64",
        "device": "gpu:0",
        "values": [0, 1, 2],
    },
    "kv_cache_inputs": {
        "block_ids": {
            "shape": [2, 1],
            "dtype": "int32",
            "device": "gpu:0",
            "values": [[7], [11]],
        },
        "cache_lengths": {
            "shape": [2, 1],
            "dtype": "uint32",
            "device": "gpu:0",
            "values": [[0], [4]],
        },
    },
}

model_inputs_buffers = (
    [101, 102, 103, 201, 202],
    [0, 3, 5],
    [1],
    [0, 1, 2],
    [[7], [11]],
    [[0], [4]],
)

compiled_model_execute_output = (
    [
        [-2.0, 0.5, 3.0, 1.0],
        [-1.5, 2.5, 0.25, -0.5],
    ],
    [0, 1, 2],
)

processed_model_outputs = {
    "logits": [
        [-2.0, 0.5, 3.0, 1.0],
        [-1.5, 2.5, 0.25, -0.5],
    ],
    "next_token_logits": None,
    "logit_offsets": [0, 1, 2],
    "hidden_states": None,
}

graph_build_values = {
    "data_parallel_degree": 2,
    "len(devices)": 2,
    "selected_branch": "create_data_parallel_graph",
    "graph_name": "llama3",
    "return_logits": "LAST_TOKEN",
    "return_hidden_states": "NONE",
    "norm_method": "rms_norm",
    "attention_bias": False,
}
```
