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

{{startup-map}}

{{nested:qna-1-1-1}}

{{nested:qna-1-1-2}}

{{nested:qna-1-1-3}}

{{nested:qna-1-1-4}}

{{nested:qna-1-1-5}}

{{nested:qna-1-1-6}}
