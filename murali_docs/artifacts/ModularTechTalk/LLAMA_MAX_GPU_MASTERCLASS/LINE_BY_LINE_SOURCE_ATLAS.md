# Line-by-Line Source Atlas

Revision: `e9818b79ea73b498ec4b438a24ecd8d853307d0d`

This atlas explains every source line displayed in the 142 interactive scenes. The first window captures each callable's contract; the second selects a high-signal implementation block nearby.

## Part 1: Graph Construction and Finalization

### Graph object construction I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `705-712`
- Function/anchor: `class Graph:`
- Evidence: `SOURCE`
- Mathematical model: `G = (V, E, types, devices)`

- `L705` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L706` `class Graph:`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L707` `"""Represents a single MAX graph.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L708` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L709` `A :class:\`Graph\` defines a model's computation. You build a graph by`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L710` `composing operations that describe how input tensors are transformed into`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L711` `outputs. Unlike imperative code that executes operations, a :class:\`Graph\``
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L712` `captures the data flow between operations, which allows MAX to optimize and`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Graph object construction II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `736-743`
- Function/anchor: `class Graph:`
- Evidence: `SOURCE`
- Mathematical model: `G = (V, E, types, devices)`

- `L736` `) as graph:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L737` `x = graph.inputs[0].tensor`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L738` `y = x @ W + b`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L739` `graph.output(y)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L740` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L741` `**Use the graph constructor:** Pass a callable as the \`\`forward\`\` argument.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L742` `The graph automatically passes the input :class:\`TensorValue\` to the`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L743` `callable and records the return value as the graph output. Under the hood,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Graph constructor body I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `819-826`
- Function/anchor: `def __init__(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L819` `@in_default_mlir_context`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L820` `def __init__(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L821` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L822` `name: str,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L823` `forward: Callable[..., GraphOutputs | None] | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L824` `input_types: Iterable[Type[Any]] = (),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L825` `path: Path | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L826` `*args,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Graph constructor body II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `909-916`
- Function/anchor: `def __init__(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L909` `self._kernel_library = kernel_library or KernelLibrary()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L910` `self._import_kernels(_resolved_custom_extensions(custom_extensions))`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L911` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L912` `self._subgraphs = {}`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L913` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L914` `# If we're building a device graph, annotate the graph appropriately.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L915` `if is_device_graph:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L916` `op.is_device_graph = builtin.UnitAttr()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Kernel library loading I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `268-275`
- Function/anchor: `def load_paths(self, custom_extensions`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L268` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L269` `def load_paths(self, custom_extensions: Iterable[Path]) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L270` `"""Loads custom operations from provided library paths.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L271` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L272` `Performs additional "smart" library loading logic for custom operation`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L273` `libraries in additional formats. The loading logic supports the`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L274` `following formats:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L275` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Kernel library loading II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `316-323`
- Function/anchor: `def load_paths(self, custom_extensions`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L316` `cache_key = frozenset(str(p) for p in paths)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L317` `cached = KernelLibrary._analysis_cache.get(cache_key)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L318` `if cached is not None:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L319` `self._analysis = cached`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L320` `self._analysis.verify_custom_op(custom_op)`
  This validates the current object or IR contract and surfaces diagnostics close to the source that created it.
- `L321` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L322` `self._analysis.verify_custom_op(custom_op)`
  This validates the current object or IR contract and surfaces diagnostics close to the source that created it.
- `L323` `KernelLibrary._analysis_cache[cache_key] = self._analysis`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Per-operation verification I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `1185-1192`
- Function/anchor: `def _verify_op(self, op`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1185` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1186` `def _verify_op(self, op: mlir.Operation | mlir.OpView | Operation) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1187` `if self._should_verify_ops:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1188` `with self._capturing_mlir_diagnostics():`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1189` `op.verify()`
  This validates the current object or IR contract and surfaces diagnostics close to the source that created it.
- `L1190` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1191` `@contextlib.contextmanager`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L1192` `def _capturing_mlir_diagnostics(self):  # noqa: ANN202`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### Per-operation verification II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `1366-1373`
- Function/anchor: `def _verify_op(self, op`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1366` `def _populate_device_info_mapping(self) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1367` `"""Attaches mo.device_info_mapping to the module if not already present."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1368` `module = self._mlir_op.block.owner`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1369` `if _DEVICE_INFO_MAPPING_ATTR_NAME in module.attributes:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1370` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1371` `devices: list[Device] = [CPU()]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1372` `if accelerator_count() > 0:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1373` `devices.append(Accelerator())`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Device metadata population I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `1365-1372`
- Function/anchor: `def _populate_device_info_mapping`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1365` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1366` `def _populate_device_info_mapping(self) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1367` `"""Attaches mo.device_info_mapping to the module if not already present."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1368` `module = self._mlir_op.block.owner`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1369` `if _DEVICE_INFO_MAPPING_ATTR_NAME in module.attributes:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1370` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1371` `devices: list[Device] = [CPU()]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1372` `if accelerator_count() > 0:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### Device metadata population II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `1538-1545`
- Function/anchor: `def _populate_device_info_mapping`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1538` `Raises:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1539` `TypeError: If the graph has not yet been terminated by a call to`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1540` `:meth:\`output()\`.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1541` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1542` `types = self.output_types`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1543` `if self._output_structure is None:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1544` `return tuple(types)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1545` `return tree.unflatten(self._output_structure, types)`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Graph.output finalization I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `1395-1402`
- Function/anchor: `def output(self, *outputs`
- Evidence: `SOURCE`
- Mathematical model: `f : (T_0,...,T_n) -> (R_0,...,R_m)`

- `L1395` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1396` `def output(self, *outputs: GraphOutputs) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1397` `"""Sets the output values of the graph and finalizes construction.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1398` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1399` `Call this once after building all ops. The graph can't be executed`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1400` `until \`\`output()\`\` has been called. Subsequent calls to`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1401` `:attr:\`output_types\` read back the types of the values passed here.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1402` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Graph.output finalization II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `1538-1545`
- Function/anchor: `def output(self, *outputs`
- Evidence: `SOURCE`
- Mathematical model: `f : (T_0,...,T_n) -> (R_0,...,R_m)`

- `L1538` `Raises:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1539` `TypeError: If the graph has not yet been terminated by a call to`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1540` `:meth:\`output()\`.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1541` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1542` `types = self.output_types`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1543` `if self._output_structure is None:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1544` `return tuple(types)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1545` `return tree.unflatten(self._output_structure, types)`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Llama config initialization I - contract

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `223-230`
- Function/anchor: `def initialize(`
- Evidence: `SOURCE`
- Mathematical model: `d_h = d_model / n_heads`

- `L223` `@classmethod`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L224` `def initialize(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L225` `cls,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L226` `pipeline_config: PipelineConfig,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L227` `model_config: MAXModelConfig | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L228` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L229` `max_seq_len: int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L230` `) -> Self:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Llama config initialization II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `255-262`
- Function/anchor: `def initialize(`
- Evidence: `SOURCE`
- Mathematical model: `d_h = d_model / n_heads`

- `L255` `model_config = model_config or pipeline_config.model`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L256` `kv_cache_config = model_config.kv_cache`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L257` `quantization_encoding = _select_quantization_encoding(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L258` `model_config, cls.DEFAULT_ENCODING`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L259` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L260` `dtype = supported_encoding_dtype(quantization_encoding)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L261` `cache_dtype = cache_dtype_for_encoding(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L262` `quantization_encoding, model_config.kv_cache.kv_cache_format`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Llama config finalization I - contract

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `396-403`
- Function/anchor: `def finalize(`
- Evidence: `SOURCE`
- Mathematical model: `P ~= Vd + L(4d^2 + 3d d_ff)`

- `L396` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L397` `def finalize(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L398` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L399` `huggingface_config: AutoConfig,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L400` `state_dict: dict[str, WeightData],`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L401` `return_logits: ReturnLogits,`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L402` `return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L403` `norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Llama config finalization II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `420-427`
- Function/anchor: `def finalize(`
- Evidence: `SOURCE`
- Mathematical model: `P ~= Vd + L(4d^2 + 3d d_ff)`

- `L420` `for k, v in state_dict.items()`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L421` `if k.startswith("language_model.")`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L422` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L423` `elif has_model_prefix:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L424` `normalized_state_dict = {`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L425` `_strip_prefix(k, "model."): v`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L426` `for k, v in state_dict.items()`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L427` `if k.startswith("model.")`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### Llama symbolic trace I - contract

- File: `max/python/max/pipelines/architectures/llama3/llama3.py`
- Source range: `272-279`
- Function/anchor: `def trace(`
- Evidence: `SOURCE`
- Mathematical model: `logits = LMHead(Transformer(tokens, KV))`

- `L272` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L273` `def trace(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L274` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L275` `*input_types: Type[Any],`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L276` `name: str | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L277` `module: GraphModule | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L278` `custom_extensions: Iterable[Path] = (),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L279` `is_device_graph: bool = False,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Llama symbolic trace II - implementation

- File: `max/python/max/pipelines/architectures/llama3/llama3.py`
- Source range: `289-296`
- Function/anchor: `def trace(`
- Evidence: `SOURCE`
- Mathematical model: `logits = LMHead(Transformer(tokens, KV))`

- `L289` `name: The name of the graph. Defaults to the class's qualified`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L290` `name.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L291` `module: The MLIR module to add the graph to. Defaults to a new`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L292` `module.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L293` `custom_extensions: Paths to custom kernel libraries the graph`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L294` `uses.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L295` `is_device_graph: Whether to build a device graph.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L296` ``
  This blank line separates logical source blocks; it performs no runtime work.

### GraphSpec trace recipe I - contract

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `113-120`
- Function/anchor: `def trace(self, module`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L113` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L114` `def trace(self, module: GraphModule | None = None) -> Graph:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L115` `"""Traces the graph, into \`\`module\`\` if given.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L116` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L117` `Raises:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L118` `ValueError: If \`\`module\`\` is given for a ModuleV3 graph.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L119` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L120` `if isinstance(self.module, ModuleV3):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### GraphSpec trace recipe II - implementation

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `194-201`
- Function/anchor: `def trace(self, module`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L194` `:func:\`compile_roles_separately\`, groups are ignored.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L195` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L196` `Each callable is called with one argument per input type, positionally,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L197` `or with the named input tree as its one argument, and returns the graph's`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L198` `outputs in the tree its \`\`forward\`\` returned. It keeps no weight registry,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L199` `so the host weights can be released after initialization. A ModuleV3`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L200` `role's callable instead takes and returns what the module's \`\`forward\`\``
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L201` `does, adds the signal buffers a multi-device graph takes, and keeps its`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Compile and initialize graphs I - contract

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `182-189`
- Function/anchor: `def compile_and_init_graphs(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L182` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L183` `def compile_and_init_graphs(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L184` `session: InferenceSession, specs: Mapping[str, GraphSpec]`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L185` `) -> dict[str, CompiledCallable[..., Any]]:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L186` `"""Compiles and initializes each role's graph.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L187` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L188` `Roles that share a group trace into one :class:\`~max.graph.Module\`,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L189` `compile once, and initialize once with the weights of all of them; each`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Compile and initialize graphs II - implementation

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `194-201`
- Function/anchor: `def compile_and_init_graphs(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L194` `:func:\`compile_roles_separately\`, groups are ignored.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L195` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L196` `Each callable is called with one argument per input type, positionally,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L197` `or with the named input tree as its one argument, and returns the graph's`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L198` `outputs in the tree its \`\`forward\`\` returned. It keeps no weight registry,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L199` `so the host weights can be released after initialization. A ModuleV3`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L200` `role's callable instead takes and returns what the module's \`\`forward\`\``
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L201` `does, adds the signal buffers a multi-device graph takes, and keeps its`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

## Part 2: Graph Compiler Transformations

### Top-level graph compiler stages I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `86-93`
- Function/anchor: `static void gcPipelineImpl`
- Evidence: `SOURCE`
- Mathematical model: `IR_(k+1) = Pass_k(IR_k)`

- `L86` `namespace {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L87` `static void gcPipelineImpl(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L88` `const GCPipelineOptions &opts, bool skipJit,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L89` `bool dumpIntermediates = false);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L90` `} // namespace`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L91` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L92` `void M::MO::registerGCPipeline() {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L93` `mlir::PassPipelineRegistration<GCPipelineOptions>(`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Top-level graph compiler stages II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `227-234`
- Function/anchor: `static void gcPipelineImpl`
- Evidence: `SOURCE`
- Mathematical model: `IR_(k+1) = Pass_k(IR_k)`

- `L227` `if (!opts.kernelDeclsPrePopulated) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L228` `pm.addPass(MOGG::createImportPackages(`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L229` `{.mojoPkgPaths = kernelLibraryPaths, .litMlirPaths = {}}));`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L230` `pm.addPass(MOGG::createVerifyKernels());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L231` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L232` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L233` `pm.addPass(MOGG::createRewriteOpaqueTypes());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L234` `pm.addPass(KGEN::createVerifyParameters());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.

### MO preprocessing pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `218-225`
- Function/anchor: `void M::MO::buildMoPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L218` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L219` `void M::MO::buildMoPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L220` `const MoPipelineOptions &opts) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L221` `auto kernelLibraryPaths =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L222` `llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L223` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L224` `// Skipped when the caller already seeded decls and`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L225` `// \`mogg.opaque_type_mapping\` via \`MOGG::copyKernelDecls\`. RewriteOpaqueTypes`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### MO preprocessing pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `393-400`
- Function/anchor: `void M::MO::buildMoPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L393` `// After ExecInvariantCodeMotion: buffer reuse can block hoisting.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L394` `pm.addPass(MGP::createSimpleBufferReusePass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L395` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L396` `pm.addPass(MGP::createDynamicMemoryPlanningPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L397` `pm.addPass(MGP::createDeferAllocsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L398` `pm.addPass(MGP::createVerifyModelArgumentsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L399` `pm.addPass(MGP::createPredicateAssertDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L400` `pm.addPass(MGP::createSimplifyChainsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.

### MAP optimization pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `337-344`
- Function/anchor: `void M::MO::buildMapPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L337` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L338` `void M::MO::buildMapPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L339` `const MapPipelineOptions &opts) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L340` `pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L341` `pm.addNestedPass<GraphOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L342` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L343` `pm.addNestedPass<GraphOp>(MAP::createFuseKernels());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L344` `if (opts.fuseMutableLoadStore)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### MAP optimization pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `417-424`
- Function/anchor: `void M::MO::buildMapPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L417` `pm.addPass(MGP::createStageDeviceGraphsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L418` `pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L419` `pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L420` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L421` `pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L422` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L423` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L424` `pm.addPass(MGP::createGenerateBuildGraphStubsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.

### MOGG routine pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `369-376`
- Function/anchor: `void M::MO::buildMoggPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L369` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L370` `void M::MO::buildMoggPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L371` `const MoggPipelineOptions &) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L372` `pm.addPass(MOGG::createLegalizeRowwiseKernels());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L373` `pm.addPass(MOGG::createConformCallOperands());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L374` `pm.addPass(MOGG::createOutlineRoutines());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L375` `pm.addPass(MOGG::createWrapFuncsWithTrace());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L376` ``
  This blank line separates logical source blocks; it performs no runtime work.

### MOGG routine pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `417-424`
- Function/anchor: `void M::MO::buildMoggPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L417` `pm.addPass(MGP::createStageDeviceGraphsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L418` `pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L419` `pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L420` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L421` `pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L422` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L423` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L424` `pm.addPass(MGP::createGenerateBuildGraphStubsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.

### MGP runtime pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `381-388`
- Function/anchor: `void M::MO::buildMgpPipeline`
- Evidence: `SOURCE`
- Mathematical model: `peak = max_t sum_(b live at t) bytes(b)`

- `L381` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L382` `void M::MO::buildMgpPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L383` `const MgpPipelineOptions &opts) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L384` `auto kernelLibraryPaths =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L385` `llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L386` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L387` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L388` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### MGP runtime pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `417-424`
- Function/anchor: `void M::MO::buildMgpPipeline`
- Evidence: `SOURCE`
- Mathematical model: `peak = max_t sum_(b live at t) bytes(b)`

- `L417` `pm.addPass(MGP::createStageDeviceGraphsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L418` `pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L419` `pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L420` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L421` `pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L422` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L423` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L424` `pm.addPass(MGP::createGenerateBuildGraphStubsPass());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.

### MO to MAP conversion I - contract

- File: `GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp`
- Source range: `766-773`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L766` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L767` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L768` `MOGG::MojoRegistry registry(getOperation());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L769` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L770` `SmallVector<Operation *> worklist;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L771` `WalkResult walkResult = getOperation()->walk([&](Operation *op) {`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L772` `Dialect *dialect = op->getDialect();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L773` `if (dialect && isa<MO::MODialect>(dialect) && !hasLowerableRank(op)) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### MO to MAP conversion II - implementation

- File: `GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp`
- Source range: `782-789`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L782` `return WalkResult::advance();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L783` `});`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L784` `if (walkResult.wasInterrupted())`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L785` `return signalPassFailure();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L786` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L787` `for (Operation *op : worklist)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L788` `if (failed(lowerOp(op, registry)))`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L789` `return signalPassFailure();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### MAP kernel fusion I - contract

- File: `GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp`
- Source range: `36-43`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `bytes_saved ~= read(tmp) + write(tmp)`

- `L36` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L37` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L38` `MO::GraphOp graphOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L39` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L40` `// One fusion round: input fusion before output fusion, so a`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L41` `// view/elementwise producer fuses into a foreach consumer before that`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L42` `// consumer is output-fused into an epilogue -- which is what lets a view`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L43` `// reach an epilogue without fusing directly into a closure. Prologue runs`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### MAP kernel fusion II - implementation

- File: `GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp`
- Source range: `46-53`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `bytes_saved ~= read(tmp) + write(tmp)`

- `L46` `mlir::OpPassManager fusePm(MO::GraphOp::getOperationName());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L47` `fusePm.addPass(createFuseElementwise());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L48` `fusePm.addPass(createFuseViews());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L49` `fusePm.addPass(createFusePrologue());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L50` `fusePm.addPass(createFuseReductions());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L51` `fusePm.addPass(createFuseEpilogue());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L52` `fusePm.addPass(createFuseReshapeEpilogue());`
  This fixes transformation order. The next pass consumes invariants produced by the pass registered here.
- `L53` `// Last, so siblings merge only once the producer-consumer fusers have`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### Reduction fusion I - contract

- File: `GraphCompiler/lib/MAPDialect/Transforms/ReductionFuser.cpp`
- Source range: `301-308`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L301` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L302` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L303` `MO::GraphOp graphOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L304` `IRRewriter rewriter(graphOp.getContext());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L305` `SmallVector<IterRowwiseOp> rowwiseOps;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L306` `graphOp.getBody().walk([&](IterRowwiseOp op) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L307` `rowwiseOps.push_back(op);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L308` `return WalkResult::skip();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Reduction fusion II - implementation

- File: `GraphCompiler/lib/MAPDialect/Transforms/ReductionFuser.cpp`
- Source range: `315-322`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L315` `while (IterElementwiseOp src = findProducer(dst)) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L316` `fuseProducer(dst, src, rewriter);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L317` `rewriter.eraseOp(src);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L318` `++numKernelsFused;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L319` `changed = true;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L320` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L321` `if (IterElementwiseOp consumer = findConsumer(dst)) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L322` `dst = fuseConsumer(dst, consumer, rewriter);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### MAP to MOGG conversion I - contract

- File: `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`
- Source range: `55-62`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L55` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L56` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L57` `MOGG::MojoRegistry registry(getOperation());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L58` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L59` `SmallVector<MAP::IterOpaqueOp> worklist;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L60` `getOperation()->walk([&](MAP::IterOpaqueOp op) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L61` `worklist.push_back(op);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L62` `return WalkResult::skip();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### MAP to MOGG conversion II - implementation

- File: `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`
- Source range: `74-81`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L74` `MAPToMOGGState state(registry, op);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L75` `state.enableComputeLambdas = enableComputeLambdas;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L76` `return failed(lowerInvokeShapeFuncToMOGG(op, state))`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L77` `? WalkResult::interrupt()`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L78` `: WalkResult::advance();`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L79` `});`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L80` `if (result.wasInterrupted())`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L81` `return signalPassFailure();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### MOGG to MGP conversion I - contract

- File: `GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp`
- Source range: `2405-2412`
- Function/anchor: `MOGGToMGPPass::runOnOperation`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L2405` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L2406` `void MOGGToMGPPass::runOnOperation() {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L2407` `ModuleOp theModule = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2408` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L2409` `// Kernel stubs created during lowering are registered through the cached`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L2410` `// \`SymbolTableAnalysis\`, which the pass manager never invalidates.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L2411` `SymbolTable &moduleSymbolTable =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2412` `getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### MOGG to MGP conversion II - implementation

- File: `GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp`
- Source range: `2429-2436`
- Function/anchor: `MOGGToMGPPass::runOnOperation`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L2429` `size_t lastProcessed = 0;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2430` `// Process graphs in dependency order.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L2431` `while (!worklist.empty()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L2432` `// A full pass with no progress means a cycle.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L2433` `if (lastProcessed > worklist.size()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L2434` `theModule.emitError("cycle detected in graph ops");`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L2435` `return signalPassFailure();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L2436` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### MEF serialization boundary I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `691-698`
- Function/anchor: `convertMGPToMEF`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `MEF = serialize(finalized MGP)`

- `L691` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L692` `ErrorOr<BufferRef> M::MO::convertMGPToMEF(ModuleOp module,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L693` `bool enableMEFMetadata) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L694` `OwningOpRef<ModuleOp> moduleToRewrite = module.clone();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L695` `// Device-generic ops get renamed with their device label so GraphRT can`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L696` `// dispatch to a device-appropriate primitive.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L697` `BlobMap resources = getDenseResources(*moduleToRewrite);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L698` `mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### MEF serialization boundary II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `701-708`
- Function/anchor: `convertMGPToMEF`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `MEF = serialize(finalized MGP)`

- `L701` `FormatScopedDiagnosticHandler diagnostics(moduleToRewrite->getContext());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L702` `if (failed(pm.run(*moduleToRewrite)))`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L703` `return Error(Twine("Failed to prepare MGP model for conversion to MEF. ") +`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L704` `diagnostics.formatMessage());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L705` `ConcatenationTree result =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L706` `translateMLIRToMEF(*moduleToRewrite, enableMEFMetadata);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L707` `if (result.getSize() == 0) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L708` `return Error(Twine("Failed to convert model to MEF. ") +`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

## Part 3: Kernel Development and Optimization

### Kernel extensibility registration I - contract

- File: `max/kernels/src/graph_compiler/extensibility/decorators.mojo`
- Source range: `45-52`
- Function/anchor: `def register(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L45` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L46` `def register(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L47` `name: StaticString,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L48` `type: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L49` `api: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L50` `arch: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L51` `model: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L52` `):`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Kernel extensibility registration II - implementation

- File: `max/kernels/src/graph_compiler/extensibility/decorators.mojo`
- Source range: `66-73`
- Function/anchor: `def register(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L66` `Fields are ordered from least to most specific: \`type\` < \`api\` < \`arch\` <`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L67` `\`model\`. Among all matching candidates, the most specific registration wins:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L68` `a registration that sets \`arch\` is more specific than one that sets only`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L69` `\`type\` and \`api\`, because \`arch\` sits higher in the hierarchy.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L70` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L71` `If a user-defined kernel and a built-in kernel are equally specific for a`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L72` `device, the user-defined kernel takes precedence.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L73` `Registering the same op with identical device fields more than once,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### GPU matmul registration I - contract

- File: `max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo`
- Source range: `350-357`
- Function/anchor: `@extensibility.register("mo.matmul", type="gpu")`
- Evidence: `SOURCE`
- Mathematical model: `C_(M,N) = A_(M,K) B_(K,N)`

- `L350` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L351` `@extensibility.register("mo.matmul", type="gpu")`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L352` `struct MatmulGPU:`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L353` `"""Registers the \`mo.matmul\` graph op for GPU devices."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L354` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L355` `@staticmethod`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L356` `def execute[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L357` `transpose_b: Bool,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### GPU matmul registration II - implementation

- File: `max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo`
- Source range: `385-392`
- Function/anchor: `@extensibility.register("mo.matmul", type="gpu")`
- Evidence: `SOURCE`
- Mathematical model: `C_(M,N) = A_(M,K) B_(K,N)`

- `L385` `](coords: Coord, val: SIMD[_dtype, _width]) {var c} -> SIMD[`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L386` `_dtype, _width`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L387` `]:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L388` `comptime assert (`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L389` `coords.rank == 2`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L390` `), "the matmul epilogue contract is rank 2"`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L391` `return rebind[SIMD[_dtype, _width]](`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L392` `c._fused_compute_output_lambda[element_alignment=alignment](`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Generic GPU matmul kernel I - contract

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `106-113`
- Function/anchor: `def matmul_kernel[`
- Evidence: `SOURCE`
- Mathematical model: `FLOPs = 2 M N K`

- `L106` `@__name(t"matmul_kernel_{c_type}_{a_type}_{b_type}_{tile_size}")`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L107` `def matmul_kernel[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L108` `c_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L109` `a_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L110` `b_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L111` `tile_size: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L112` `s_type: DType = get_accum_type[c_type](),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L113` `](`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Generic GPU matmul kernel II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `207-214`
- Function/anchor: `def matmul_kernel[`
- Evidence: `SOURCE`
- Mathematical model: `FLOPs = 2 M N K`

- `L207` `# If K is not multiple of tile_size, the last tile contains less than`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L208` `# tile_size elements. The thread block needs to take addition bound check`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L209` `# when loading elements into shared memory.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L210` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L211` `# Load A tile into shared memory.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L212` `var a_val: Scalar[a_type]`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L213` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L214` `comptime if not full_tile:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### Naive matmul baseline I - contract

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `253-260`
- Function/anchor: `def matmul_kernel_naive[`
- Evidence: `SOURCE`
- Mathematical model: `AI_naive ~= 2K / (2K b + b)`

- `L253` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L254` `def matmul_kernel_naive[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L255` `c_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L256` `a_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L257` `b_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L258` `c_layout_type: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L259` `a_layout_type: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L260` `b_layout_type: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Naive matmul baseline II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `406-413`
- Function/anchor: `def matmul_kernel_naive[`
- Evidence: `SOURCE`
- Mathematical model: `AI_naive ~= 2K / (2K b + b)`

- `L406` `else:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L407` `for i in range(_k):`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L408` `accum += (`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L409` `a.load[width=1]((x, i)).cast[s_type]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L410` `* b.load[width=1]((i, y)).cast[s_type]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L411` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L412` `comptime if has_epilogue_fn:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L413` `epilogue_fn[c_type, 1, alignment=1]((x, y), accum.cast[c_type]())`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Coalesced global-memory access I - contract

- File: `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- Source range: `169-176`
- Function/anchor: `def coalescing_matrix_multiplication[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `transactions ~= requested_bytes / transaction_width`

- `L169` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L170` `def coalescing_matrix_multiplication[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L171` `dtype: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L172` `a_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L173` `b_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L174` `c_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L175` `BM: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L176` `BN: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Coalesced global-memory access II - implementation

- File: `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- Source range: `316-323`
- Function/anchor: `def coalescing_matrix_multiplication[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `transactions ~= requested_bytes / transaction_width`

- `L316` `# Asynchronously copy tiles of A and B from global memory to shared`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L317` `# memory.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L318` `GenericToSharedAsyncTileCopier[load_a_layout]().copy(a_smem, a_tile)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L319` `GenericToSharedAsyncTileCopier[load_b_layout]().copy(b_smem, b_tile)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L320` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L321` `# Commit the issued async copies as a group and wait for them to`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L322` `# complete before any thread reads from shared memory.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L323` `async_copy_commit_group()`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### Asynchronous DRAM to SRAM copy I - contract

- File: `max/kernels/src/layout/tile_io.mojo`
- Source range: `1245-1252`
- Function/anchor: `def copy_dram_to_sram_async[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `T_stage ~= max(T_copy, T_compute) when overlap holds`

- `L1245` `@inline(.nodebug)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L1246` `def copy_dram_to_sram_async[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1247` `thread_layout: Layout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1248` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1249` `swizzle: Optional[Swizzle] = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1250` `masked: Bool = False,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1251` `eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1252` `num_threads: Int = thread_layout.size(),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Asynchronous DRAM to SRAM copy II - implementation

- File: `max/kernels/src/layout/tile_io.mojo`
- Source range: `1258-1265`
- Function/anchor: `def copy_dram_to_sram_async[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `T_stage ~= max(T_copy, T_compute) when overlap holds`

- `L1258` `):`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1259` `"""Asynchronously copies a tile from DRAM (generic memory) to SRAM (shared).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1260` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1261` `Delegates to \`GenericToSharedAsyncTileCopier\`, which issues NVIDIA`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1262` `\`cp.async\` instructions (falling back to synchronous loads/stores on AMD`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1263` `and Apple GPUs). The copy is asynchronous: callers must commit it via`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1264` `\`async_copy_commit_group()\` and synchronize via \`async_copy_wait_all()\``
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1265` `or \`async_copy_wait_group()\` before reading the destination tile.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Ldmatrix asynchronous path I - contract

- File: `max/kernels/src/layout/tile_async_copy.mojo`
- Source range: `24-31`
- Function/anchor: `def copy_dram_to_sram_async_ldmatrix[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L24` `@always_inline`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L25` `def copy_dram_to_sram_async_ldmatrix[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L26` `thread_layout: Layout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L27` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L28` `simd_size: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L29` `swizzle: Bool,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L30` `masked: Bool = False,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L31` `num_threads: Int = thread_layout.size(),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Ldmatrix asynchronous path II - implementation

- File: `max/kernels/src/layout/tile_async_copy.mojo`
- Source range: `55-62`
- Function/anchor: `def copy_dram_to_sram_async_ldmatrix[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L55` `\`thread_layout.size()\` do not copy.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L56` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L57` `Args:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L58` `dst: Destination tile in shared memory, with a static row stride.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L59` `src: Source tile in global memory.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L60` `num_valid_rows: Valid rows of \`src\`; only consulted when \`masked\`.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L61` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L62` `comptime if is_nvidia_gpu():`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### Register and vector tiling I - contract

- File: `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- Source range: `570-577`
- Function/anchor: `def block_tiled_vectorized_matrix_multiplication[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `AI_tiled grows with reuse while occupancy falls with registers/thread`

- `L570` `@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L571` `def block_tiled_vectorized_matrix_multiplication[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L572` `dtype: DType,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L573` `a_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L574` `b_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L575` `c_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L576` `BM: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L577` `BN: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Register and vector tiling II - implementation

- File: `max/examples/custom_ops/kernels/matrix_multiplication.mojo`
- Source range: `672-679`
- Function/anchor: `def block_tiled_vectorized_matrix_multiplication[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `AI_tiled grows with reuse while occupancy falls with registers/thread`

- `L672` `comptime for v in range(simd_width):`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L673` `a_smem[inner_row_a, inner_col_a * simd_width + v] = rebind[`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L674` `a_smem.ElementType`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L675` `](a_load[v])`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L676` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L677` `# Asynchronously load the tile of B into shared memory using`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L678` `# vectorized memory access.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L679` `GenericToSharedAsyncTileCopier[load_b_layout]().copy(`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Persistent SM90 matmul I - contract

- File: `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernel_persistent.mojo`
- Source range: `57-64`
- Function/anchor: `def run_persistent[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `tiles_per_CTA > 1`

- `L57` `@__llvm_arg_metadata(c_tma_op, \`nvvm.grid_constant\`)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L58` `def run_persistent[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L59` `a_tile_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L60` `b_tile_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L61` `c_tile_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L62` `a_desc_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L63` `b_desc_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L64` `c_desc_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Persistent SM90 matmul II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernel_persistent.mojo`
- Source range: `104-111`
- Function/anchor: `def run_persistent[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `tiles_per_CTA > 1`

- `L104` `var pipeline = smem.create_pipeline()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L105` `var barrier_handler = Self.TMABarrier(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L106` `pipeline, Self.num_consumer, Self.cluster_size`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L107` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L108` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L109` `# Create TileLoaderTMA loaders`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L110` `var a_loader, b_loader = Self.build_tma_loaders(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L111` `a_tma_op, b_tma_op, rank_m, rank_n`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### AMD ping-pong pipeline I - contract

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo`
- Source range: `424-431`
- Function/anchor: `def _run_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L424` `@inline(.always)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L425` `def _run_impl[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L426` `a_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L427` `b_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L428` `c_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L429` `a_engine: TensorEngine,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L430` `b_engine: TensorEngine,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L431` `c_engine: TensorEngine,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### AMD ping-pong pipeline II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo`
- Source range: `480-487`
- Function/anchor: `def _run_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L480` `# === SMEM: double-buffered half-tiles ===`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L481` `# A: 2 stages x 2 warp_m groups`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L482` `comptime a_half_layout = row_major[half_BM, BK]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L483` `var a_s0_g0 = stack_allocation[Self.in_type, address_space=.SHARED](`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L484` `a_half_layout`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L485` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L486` `var a_s0_g1 = stack_allocation[Self.in_type, address_space=.SHARED](`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L487` `a_half_layout`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Flash-attention dispatch I - contract

- File: `max/kernels/src/graph_compiler/builtin_kernels/attention.mojo`
- Source range: `1139-1146`
- Function/anchor: `def _dispatch_flash_attention[`
- Evidence: `SOURCE`
- Mathematical model: `softmax(QK^T/sqrt(d_h))V`

- `L1139` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1140` `def _dispatch_flash_attention[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1141` `mask_t: MHAMask`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1142` `](mask: mask_t) raises {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1143` `var output_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1144` `var q_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1145` `var k_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1146` `var v_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Flash-attention dispatch II - implementation

- File: `max/kernels/src/graph_compiler/builtin_kernels/attention.mojo`
- Source range: `1283-1290`
- Function/anchor: `def _dispatch_flash_attention[`
- Evidence: `SOURCE`
- Mathematical model: `softmax(QK^T/sqrt(d_h))V`

- `L1283` `var output_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1284` `var q_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1285` `var k_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1286` `var v_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1287` `imm,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1288` `}:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1289` `flash_attention_ragged[](`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1290` `output_buffer,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Paged KV cache store kernel I - contract

- File: `max/kernels/src/graph_compiler/builtin_kernels/kv_cache.mojo`
- Source range: `76-83`
- Function/anchor: `@extensibility.register("mo.kv_cache.store.paged.ragged")`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L76` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L77` `@extensibility.register("mo.kv_cache.store.paged.ragged")`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L78` `struct Struct_kv_cache_store_paged:`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L79` `"""Registers the \`mo.kv_cache.store.paged.ragged\` graph op with the graph compiler.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L80` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L81` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L82` `@inline(.always)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L83` `@staticmethod`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.

### Paged KV cache store kernel II - implementation

- File: `max/kernels/src/graph_compiler/builtin_kernels/kv_cache.mojo`
- Source range: `192-199`
- Function/anchor: `@extensibility.register("mo.kv_cache.store.paged.ragged")`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L192` `@extensibility.register("mo.kv_cache.store_k_scales.paged.ragged")`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L193` `struct Struct_kv_cache_store_k_scales_paged:`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L194` `"""Registers the \`mo.kv_cache.store_k_scales.paged.ragged\` graph op with the graph compiler.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L195` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L196` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L197` `@inline(.always)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L198` `@staticmethod`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L199` `def execute[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### Arithmetic intensity and roofline I - contract

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `803-810`
- Function/anchor: `def _matmul_gpu_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `Perf <= min(PeakFLOPs, AI * Bandwidth); AI = FLOPs / bytes`

- `L803` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L804` `def _matmul_gpu_impl[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L805` `EpilogueFnType: ElementwiseEpilogueFn,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L806` `ComputeFnType: ElementwiseComputeFn,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L807` `//,`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L808` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L809` `use_tensor_core: Bool,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L810` `transpose_b: Bool,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Arithmetic intensity and roofline II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `971-978`
- Function/anchor: `def _matmul_gpu_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `Perf <= min(PeakFLOPs, AI * Bandwidth); AI = FLOPs / bytes`

- `L971` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L972` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L973` `# m > 1 (not m >= 64): the kernel already handles a partial M tile`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L974` `# (per-simdgroup \`_bounded_load\`/\`_bounded_store\` + the row_base>=M`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L975` `# early return), so 1 < m < 64 (concurrent-decode batch widths) is`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L976` `# correct here. Routing it off the naive per-row fallback onto this`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L977` `# co-batched GEMM is 6-27x faster at real Llama decode shapes`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L978` `# (microbench, M5 Max). m == 1 stays on the gemv path: a rank-1`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

## Part 4: Driver, Offload, Memory, and Backends

### Compile an offloaded Mojo function I - contract

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `315-322`
- Function/anchor: `def _compile_inner[`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L315` `@inline(.always)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L316` `def _compile_inner[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L317` `fn_type: TrivialRegisterPassable,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L318` `func: fn_type,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L319` `](self) -> CompiledFunctionInfo[fn_type, func, Self._device_spec_target,]:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L320` `comptime target = Self._device_spec_target`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L321` `comptime emission_kind_id = _get_emission_kind_id[`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L322` `"object"`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Compile an offloaded Mojo function II - implementation

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `465-472`
- Function/anchor: `def _compile_inner[`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L465` `the plugin never frees it, freeing the buffer releases only the`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L466` `plugin's bookkeeping, and the caller must keep the underlying`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L467` `allocation alive for the buffer's lifetime. With \`owning=True\``
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L468` `the buffer frees the region through the plugin's normal path,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L469` `which is only valid for an address that came from this plugin's`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L470` `own allocator (e.g. one released with \`unwrap_memory\`).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L471` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L472` `return Buffer[Self.device_spec](`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Load a compiled bundle I - contract

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `358-365`
- Function/anchor: `def load_bundle[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L358` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L359` `def load_bundle[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L360` `asm_origin: ImmOrigin`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L361` `](`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L362` `self, asm: StringSlice[origin=asm_origin]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L363` `) raises HALError -> RuntimeBundle:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L364` `"""Loads a runtime bundle from pre-compiled binary bytes."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L365` `# Each plugin expects a specific file_type string. PTX text is`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### Load a compiled bundle II - implementation

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `465-472`
- Function/anchor: `def load_bundle[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L465` `the plugin never frees it, freeing the buffer releases only the`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L466` `plugin's bookkeeping, and the caller must keep the underlying`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L467` `allocation alive for the buffer's lifetime. With \`owning=True\``
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L468` `the buffer frees the region through the plugin's normal path,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L469` `which is only valid for an address that came from this plugin's`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L470` `own allocator (e.g. one released with \`unwrap_memory\`).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L471` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L472` `return Buffer[Self.device_spec](`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Resolve a device function I - contract

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `544-551`
- Function/anchor: `def load_function(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L544` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L545` `def load_function(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L546` `self, bundle: RuntimeBundle, var name: String`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L547` `) raises HALError -> FunctionHandle:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L548` `return self._raw[].load_function(self._handle, bundle._handle, name)`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L549` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L550` `def set_function_attribute(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L551` `self, func: FunctionHandle, attribute: Int32, value: Int32`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Resolve a device function II - implementation

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `632-639`
- Function/anchor: `def load_function(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L632` `"""Inserts \`entry\` under \`key\`, re-checking for a load race; acquires`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L633` `and releases the cache lock.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L634` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L635` `If another thread cached \`key\` while the caller was loading, discards`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L636` `the duplicate, unloading its function BEFORE its \`entry\` (and so`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L637` `the bundle that defines the function) is destroyed, and returns the`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L638` `cached entry's handle. Otherwise inserts \`entry\` and returns None."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L639` `self._function_cache_lock_acquire()`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Enqueue a kernel function I - contract

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `967-974`
- Function/anchor: `def enqueue_function[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L967` `@inline(.always)`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L968` `def enqueue_function[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L969` `declared_arg_types: TypeList[Trait=AnyType, ...],`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L970` `ExecutionConfigType: GridBlockExecutionConfig,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L971` `//,`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L972` `func: def(*args: *declared_arg_types) thin -> None,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L973` `*actual_arg_types: DevicePassable,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L974` `](`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Enqueue a kernel function II - implementation

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1091-1098`
- Function/anchor: `def enqueue_function[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1091` `location: Optional[SourceLocation] = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1092` `) raises:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1093` `"""Enqueues a pre-compiled checked function for execution on this device.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1094` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1095` `This overload requires a \`DeviceFunction\` that was compiled with`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1096` `type checking enabled (via \`compile_function\`). The function`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1097` `will verify that the argument types match the declared types at`
  This validates the current object or IR contract and surfaces diagnostics close to the source that created it.
- `L1098` `compile time.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Allocate a device buffer I - contract

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1470-1477`
- Function/anchor: `def enqueue_create_buffer[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `bytes = count * sizeof(dtype)`

- `L1470` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1471` `def enqueue_create_buffer[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1472` `dtype: DType`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1473` `](self, size: Int) raises -> DeviceBuffer[dtype]:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1474` `"""Enqueues a buffer creation using the \`DeviceBuffer\` constructor.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1475` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1476` `For GPU devices, the space is allocated in the device's global memory.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1477` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Allocate a device buffer II - implementation

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1519-1526`
- Function/anchor: `def enqueue_create_buffer[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `bytes = count * sizeof(dtype)`

- `L1519` `This function allocates memory on the host that is accessible by the device.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1520` `The memory is page-locked (pinned) for efficient data transfer between host and device.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1521` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1522` `Pinned memory is guaranteed to remain resident in the host's RAM, not be`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1523` `paged/swapped out to disk. Memory allocated normally (for example, using`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1524` `[\`alloc()\`](/docs/std/memory/unsafe_pointer/alloc/))`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1525` `is pageable—individual pages of memory can be moved to secondary storage`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1526` `(disk/SSD) when main memory fills up.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Host to device copy I - contract

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1561-1568`
- Function/anchor: `def enqueue_copy[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1561` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1562` `def enqueue_copy[`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1563` `dtype: DType`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1564` `](`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1565` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1566` `dst_ptr: Pointer[mut=True, Scalar[dtype], _],`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1567` `src_ptr: Pointer[Scalar[dtype], _],`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1568` `size: Int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Host to device copy II - implementation

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1673-1680`
- Function/anchor: `def enqueue_copy[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1673` `var src_view = src_buf._inner[]._buffer.view()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1674` `if src_context[].get_device_id() == self._device[].id:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1675` `self._stream[].copy_intra_device(dst_view, src_view)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1676` `else:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1677` `# This plugin would misread another driver's context handle.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1678` `if not (src_context[]._raw is self._context[]._raw):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1679` `raise Error(`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1680` `"enqueue_copy source is on another driver; cross-device"`
  This enqueues asynchronous work. Host completion is not implied until a dependency or explicit synchronization requires it.

### Queue synchronization I - contract

- File: `max/driver/src/max_license/_hal/queue.mojo`
- Source range: `404-411`
- Function/anchor: `def synchronize(self)`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L404` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L405` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L406` `def synchronize(self) raises HALError:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L407` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L408` `Totally ordered with respect to other operations within this queue`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L409` `if backed by a stream.`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L410` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L411` `self._raw[].synchronize_queue(self._handle)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Queue synchronization II - implementation

- File: `max/driver/src/max_license/_hal/queue.mojo`
- Source range: `412-419`
- Function/anchor: `def synchronize(self)`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L412` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L413` `def native_handle(self) raises HALError -> OptionalReg[UInt64]:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L414` `"""Returns the backend stream/queue handle, or None if the queue has`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L415` `none (a device with no OS-level stream object, e.g. CPU)."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L416` `return self._raw[].get_optional_queue_property["native_handle", UInt64](`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L417` `self._handle`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L418` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L419` ``
  This blank line separates logical source blocks; it performs no runtime work.

### CUDA HAL plugin I - contract

- File: `max/driver/src/max_license/_hal/_plugin/cuda/cuda_plugin.mojo`
- Source range: `21-28`
- Function/anchor: `struct CudaHalPlugin`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L21` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L22` `struct CudaHalPlugin(HalPlugin):`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L23` `"""\`PluginHooks\` implementation for NVIDIA CUDA backends.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L24` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L25` `Every hook is left at its \`PluginHooks\` default.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L26` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L27` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L28` `comptime name: __mlir_type.\`!kgen.string\` = _get_kgen_string["cuda"]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### CUDA HAL plugin II - implementation

- File: `max/driver/src/max_license/_hal/_plugin/cuda/cuda_plugin.mojo`
- Source range: `31-38`
- Function/anchor: `struct CudaHalPlugin`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L31` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L32` `struct CudaHalPluginHooks[target: CompilationTarget](HalPluginHooks):`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L33` `comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L34` `device.spec.name`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L35` `]`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L36` `"""Returns the execution configuration for the given device."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L37` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L38` `comptime extensions = CUDAHALExtensions`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### HIP HAL plugin I - contract

- File: `max/driver/src/max_license/_hal/_plugin/hip/hip_plugin.mojo`
- Source range: `21-28`
- Function/anchor: `struct HipHalPlugin`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L21` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L22` `struct HipHalPlugin(HalPlugin):`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L23` `"""\`PluginHooks\` implementation for AMD HIP backends.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L24` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L25` `Every hook is left at its \`PluginHooks\` default.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L26` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L27` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L28` `comptime name: __mlir_type.\`!kgen.string\` = _get_kgen_string["hip"]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### HIP HAL plugin II - implementation

- File: `max/driver/src/max_license/_hal/_plugin/hip/hip_plugin.mojo`
- Source range: `31-38`
- Function/anchor: `struct HipHalPlugin`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L31` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L32` `struct HipHalPluginHooks[target: CompilationTarget](HalPluginHooks):`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L33` `comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L34` `device.spec.name`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L35` `]`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L36` `"""Returns the execution configuration for the given device."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L37` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L38` `comptime extensions = HIPHALExtensions`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Metal HAL plugin I - contract

- File: `max/driver/src/max_license/_hal/_plugin/metal/metal_plugin.mojo`
- Source range: `21-28`
- Function/anchor: `struct MetalHalPlugin`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L21` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L22` `struct MetalHalPlugin(HalPlugin):`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L23` `"""\`PluginHooks\` implementation for Apple Metal backends.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L24` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L25` `Every hook is left at its \`PluginHooks\` default.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L26` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L27` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L28` `comptime name: __mlir_type.\`!kgen.string\` = _get_kgen_string["metal"]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Metal HAL plugin II - implementation

- File: `max/driver/src/max_license/_hal/_plugin/metal/metal_plugin.mojo`
- Source range: `31-38`
- Function/anchor: `struct MetalHalPlugin`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L31` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L32` `struct MetalHalPluginHooks[target: CompilationTarget](HalPluginHooks):`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L33` `comptime ExecutionConfigType[device: DeviceRef] = ExecutionConfigForTarget[`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L34` `device.spec.name`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L35` `]`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L36` `"""Returns the execution configuration for the given device."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L37` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L38` `comptime extensions = MetalHALExtensions`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### NVIDIA NVPTX target traits I - contract

- File: `Mojo/lib/Target/max_license/NVPTX/NVPTXTraits.cpp`
- Source range: `13-20`
- Function/anchor: `NVPTX`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L13` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L14` `#include "Target/max_license/NVPTX/NVPTXTraits.h"`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L15` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L16` `#include "llvm/Support/NVPTXAddrSpace.h"`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L17` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L18` `namespace M::KGEN {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L19` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L20` `const NVPTXTraits &NVPTXTraits::get() {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### NVIDIA NVPTX target traits II - implementation

- File: `Mojo/lib/Target/max_license/NVPTX/NVPTXTraits.cpp`
- Source range: `22-29`
- Function/anchor: `NVPTX`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L22` `return instance;`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L23` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L24` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L25` `std::optional<unsigned>`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L26` `NVPTXTraits::requiredStackAllocationAddressSpace() const {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L27` `return llvm::NVPTXAS::AddressSpace::ADDRESS_SPACE_LOCAL;`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L28` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L29` ``
  This blank line separates logical source blocks; it performs no runtime work.

### AMD GPU target traits I - contract

- File: `Mojo/lib/Target/max_license/AMDGPU/AMDGPUTraits.cpp`
- Source range: `13-20`
- Function/anchor: `AMDGPU`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L13` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L14` `#include "Target/max_license/AMDGPU/AMDGPUTraits.h"`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L15` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L16` `#include "llvm/Support/AMDGPUAddrSpace.h"`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L17` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L18` `namespace M::KGEN {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L19` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L20` `const AMDGPUTraits &AMDGPUTraits::get() {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### AMD GPU target traits II - implementation

- File: `Mojo/lib/Target/max_license/AMDGPU/AMDGPUTraits.cpp`
- Source range: `22-29`
- Function/anchor: `AMDGPU`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L22` `return instance;`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L23` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L24` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L25` `std::optional<unsigned>`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L26` `AMDGPUTraits::requiredStackAllocationAddressSpace() const {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L27` `return llvm::AMDGPUAS::PRIVATE_ADDRESS;`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L28` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L29` ``
  This blank line separates logical source blocks; it performs no runtime work.

### AMD target cost model I - contract

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_target.mojo`
- Source range: `15-22`
- Function/anchor: `MI355`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L15` `Platform-specific hardware descriptions for AMD CDNA/RDNA GPUs. Provides:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L16` `- MI355X cost model: per-op latencies, resources, and roles`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L17` `- MI355X pipeline configurations: double-buffer and single-buffer`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L18` `- MI355X target profiles: unified cost model + pipeline config`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L19` `- AMD schedule_group_barrier hints for instruction interleaving`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L20` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L21` `The generic framework (schedule_framework.mojo) has no AMD-specific`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L22` `references. All hardware knowledge lives here.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### AMD target cost model II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_target.mojo`
- Source range: `177-184`
- Function/anchor: `MI355`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L177` `vm_per_load_a: \`vmcnt\` ops per channel-A global load (defaults to 4).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L178` `vm_per_load_b: \`vmcnt\` ops per channel-B global load (defaults to 4).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L179` `max_globals: Maximum global loads per block, 0 means unlimited`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L180` `(defaults to 1).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L181` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L182` `return TargetProfile(`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L183` `cost_model=mi355x_cost_model(),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L184` `pipeline=mi355x_double_buffer(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Qualcomm and Hexagon boundary I - contract

- File: `MLRT/include/MLRT/Driver/DeviceContext/DeviceContext.h`
- Source range: `1128-1135`
- Function/anchor: `Hexagon`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `backend support != inferred GPU implementation`

- `L1128` `/// Returns the stream pointer for use in the DLPack protocol, or nullopt if`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1129` `/// this context has no CUDA-style stream model (e.g. Metal, Hexagon).`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1130` `///`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1131` `/// Override this in a DeviceContext subclass to expose a stream pointer`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1132` `/// through DLPack without modifying the core DLPack bridge code.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1133` `virtual std::optional<intptr_t> getDLPackStream() const {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1134` `return std::nullopt;`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1135` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Qualcomm and Hexagon boundary II - implementation

- File: `MLRT/include/MLRT/Driver/DeviceContext/DeviceContext.h`
- Source range: `1152-1159`
- Function/anchor: `Hexagon`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `backend support != inferred GPU implementation`

- `L1152` `// Note: streams created in this manner are NOT counted in numStreams(),`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1153` `// which only counts internally-managed streams.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1154` `virtual ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1155` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1156` `// Returns a view of this context whose current stream is \`streamId\`. The view`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1157` `// shares this context's full stream set, driver context, and device memory`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1158` `// pool; only the current-stream selector differs. Backends without a`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1159` `// multi-stream model return this context, so all work runs on its single`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

## Part 5: MAX Serve Scheduling and Contexts

### Scheduler iteration I - contract

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `304-311`
- Function/anchor: `def run_iteration(self)`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L304` `@traced`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L305` `def run_iteration(self) -> SchedulerProgress:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L306` `"""The Scheduler routine that creates batches and schedules them on GPU`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L307` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L308` `Returns:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L309` `SchedulerProgress: Indicates whether work was performed in this iteration.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L310` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L311` `# Drain the request queue and add to CE requests`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### Scheduler iteration II - implementation

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `332-339`
- Function/anchor: `def run_iteration(self)`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L332` `and self.pipeline.has_pending_outputs()`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L333` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L334` `if not (inputs or self.support_empty_batches or has_pending_outputs):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L335` `self.scheduler_logger.log_idle(`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L336` `self.batch_constructor.kv_cache,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L337` `self.scheduler_config.data_parallel_degree,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L338` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L339` `return SchedulerProgress.NO_PROGRESS`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Retrieve pending requests I - contract

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `244-251`
- Function/anchor: `def _retrieve_pending_requests`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L244` `@traced`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L245` `def _retrieve_pending_requests(self) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L246` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L247` `Retrieves pending requests from the request queue.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L248` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L249` `This method drains the request queue synchronously and processes`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L250` `any contexts that are available.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L251` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Retrieve pending requests II - implementation

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `332-339`
- Function/anchor: `def _retrieve_pending_requests`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L332` `and self.pipeline.has_pending_outputs()`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L333` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L334` `if not (inputs or self.support_empty_batches or has_pending_outputs):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L335` `self.scheduler_logger.log_idle(`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L336` `self.batch_constructor.kv_cache,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L337` `self.scheduler_config.data_parallel_degree,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L338` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L339` `return SchedulerProgress.NO_PROGRESS`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Schedule and execute I - contract

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `406-413`
- Function/anchor: `def _schedule(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L406` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L407` `def _schedule(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L408` `self, inputs: TextGenerationInputs[TextContext], lagged: bool = False`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L409` `) -> int:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L410` `"""Returns the number of terminated requests."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L411` `batch_id = self._batch_counter`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L412` `self._batch_counter += 1`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L413` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Schedule and execute II - implementation

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `584-591`
- Function/anchor: `def _schedule(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L584` `# Build DP batch padder when DP > 1 with device graph capture.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L585` `kv_manager = pipeline.kv_manager`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L586` `dp_padder: DPBatchPadder | None = None`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L587` `assert pipeline_config.model is not None`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L588` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L589` `scheduler_config.data_parallel_degree > 1`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L590` `and pipeline_config.runtime.device_graph_capture`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L591` `):`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Admit a text request I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `783-790`
- Function/anchor: `def _admit_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L783` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L784` `def _admit_request(self, ctx: TextContext, replica_idx: int | None) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L785` `"""Admits a grammar-ready request into the DP pool or a replica queue."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L786` `# DP-balanced CE deferral: fresh CE requests enter an unbound pool and`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L787` `# are bound to a replica by the per-step planner, which prices them by`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L788` `# estimated post-prefix-cache length. Caller-pinned requests and`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L789` `# already-generating requests bypass the pool.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L790` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### Admit a text request II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `928-935`
- Function/anchor: `def _admit_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L928` `return ctx.tokens.active_length`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L929` `return bound.ce_weight`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L930` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L931` `def _post_cache_weights(self, ctx: TextContext) -> list[int]:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L932` `"""Estimates per-replica post-prefix-cache CE length for a new request.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L933` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L934` `Probes the prefix caches read-only (device tier per replica, shared`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L935` `host/disk tiers once) and subtracts the contiguous cached prefix from`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Bind request resources I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `805-812`
- Function/anchor: `def _bind_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L805` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L806` `def _bind_request(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L807` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L808` `ctx: TextContext,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L809` `replica_idx: int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L810` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L811` `ce_weight: int | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L812` `) -> None:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Bind request resources II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `928-935`
- Function/anchor: `def _bind_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L928` `return ctx.tokens.active_length`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L929` `return bound.ce_weight`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L930` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L931` `def _post_cache_weights(self, ctx: TextContext) -> list[int]:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L932` `"""Estimates per-replica post-prefix-cache CE length for a new request.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L933` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L934` `Probes the prefix caches read-only (device tier per replica, shared`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L935` `host/disk tiers once) and subtracts the contiguous cached prefix from`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Add context-encoding work I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1460-1467`
- Function/anchor: `def _add_ce_requests`
- Evidence: `SOURCE`
- Mathematical model: `sum CE_chunk_i <= CE_budget`

- `L1460` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1461` `def _add_ce_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1462` `# Deferred by the DP CE balancer this iteration (also covers paths`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1463` `# that bypass _identify_priority, e.g. in-flight batching).`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1464` `if replica_idx in self._ce_deferred_replicas:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1465` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1466` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1467` `# Off-cadence step. Also covers the in-flight batching backfill,`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### Add context-encoding work II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1529-1536`
- Function/anchor: `def _add_ce_requests`
- Evidence: `SOURCE`
- Mathematical model: `sum CE_chunk_i <= CE_budget`

- `L1529` `self._return_to_request_queue(ctx, replica_idx)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1530` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1531` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1532` `# Check if the request fits in memory`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1533` `if self.kv_cache is not None:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1534` `# Claim the request if needed.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1535` `if not self.kv_cache.contains(ctx):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1536` `self.kv_cache.claim(ctx, replica_idx=replica_idx)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Plan chunked prefill I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1939-1946`
- Function/anchor: `def _plan_ce_step`
- Evidence: `SOURCE`
- Mathematical model: `chunk_i = min(remaining_i, per_req_cap, budget_left)`

- `L1939` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1940` `def _plan_ce_step(self) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1941` `"""Plans this iteration's CE work across DP replicas.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1942` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1943` `Prices CE work in post-prefix-cache tokens, against each replica's`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1944` `admission seats (\`\`_ce_admission_seats\`\`) so a replica is only ever`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1945` `planned work admission will take, and greedily assembles the most`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1946` `balanced CE step it can, deferring the rest:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Plan chunked prefill II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `2048-2055`
- Function/anchor: `def _plan_ce_step`
- Evidence: `SOURCE`
- Mathematical model: `chunk_i = min(remaining_i, per_req_cap, budget_left)`

- `L2048` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L2049` `step_load = list(floor)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2050` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L2051` `def _occupancy(loads: list[int]) -> float:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L2052` `max_load = max(loads)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2053` `if max_load == 0:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L2054` `return 1.0`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L2055` `return sum(loads) / len(loads) / max_load`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Add token-generation work I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1684-1691`
- Function/anchor: `def _add_tg_requests`
- Evidence: `SOURCE`
- Mathematical model: `TG_tokens ~= number of active decode contexts`

- `L1684` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1685` `def _add_tg_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1686` `replica_requests = self.replicas[replica_idx]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1687` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1688` `# Add based on the oldest request, respecting KV cache limits and token budgets.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1689` `candidate_ids = deque(replica_requests.tg_reqs.keys())`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1690` `max_batch_size = self.scheduler_config.max_batch_size`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1691` `while len(batch) < max_batch_size and len(candidate_ids) > 0:`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.

### Add token-generation work II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1851-1858`
- Function/anchor: `def _add_tg_requests`
- Evidence: `SOURCE`
- Mathematical model: `TG_tokens ~= number of active decode contexts`

- `L1851` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1852` `interval = self.scheduler_config.prefill_schedule_interval`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1853` `if interval <= 1:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1854` `return True`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1855` `# No decode work anywhere, so no lockstep stall to protect.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1856` `if not any(replica.tg_reqs for replica in self.replicas):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1857` `return True`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1858` `return self._prefill_interval_phase == 0`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Construct a replica batch I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1859-1866`
- Function/anchor: `def _construct_replica_batch`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1859` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1860` `def _construct_replica_batch(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1861` `self, replica_idx: int, priority_override: RequestType | None = None`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1862` `) -> ReplicaBatch:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1863` `"""Constructs a batch for a single replica.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1864` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1865` `Args:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1866` `replica_idx: The index of the replica to construct a batch for.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Construct a replica batch II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1980-1987`
- Function/anchor: `def _construct_replica_batch`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1980` `self._ce_deferred_replicas.clear()`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1981` `self._ce_step_quota = None`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1982` `if not self._dp_ce_balance_enabled:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1983` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L1984` `if not self._ce_pending and not any(`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1985` `replica.ce_reqs for replica in self.replicas`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1986` `):`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1987` `return`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Update TextContext I - contract

- File: `max/python/max/pipelines/context/context.py`
- Source range: `1031-1038`
- Function/anchor: `def update(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1031` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1032` `def update(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1033` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1034` `new_token: int,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1035` `log_probabilities: LogProbabilities | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1036` `) -> None:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1037` `"""Advance both token buffer and FSM state.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1038` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Update TextContext II - implementation

- File: `max/python/max/pipelines/context/context.py`
- Source range: `1061-1068`
- Function/anchor: `def update(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1061` `if self._has_pending_future_token():`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1062` `raise ValueError("Cannot have multiple future tokens.")`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1063` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1064` `if self.matcher is not None:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1065` `# For structured output, only advance the token buffer.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1066` `# The FSM cannot accept placeholder tokens and will be`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1067` `# advanced with the real token in sync_and_process_outputs().`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1068` `self.advance_token_buffer(FUTURE_TOKEN)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Paged KV claim and release I - contract

- File: `max/python/max/pipelines/kv_cache/paged_kv_cache/cache_manager.py`
- Source range: `678-685`
- Function/anchor: `def claim(self, ctx`
- Evidence: `SOURCE`
- Mathematical model: `KV_bytes ~= 2 L n_kv d_h tokens bytes_per_elem`

- `L678` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L679` `def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L680` `"""Pins a request to one replica, which owns it until it is released."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L681` `self._block_manager.claim(ctx, replica_idx)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L682` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L683` `def step(self, ctx: TextContext) -> None:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L684` `"""Commits the request's newly written tokens into the prefix cache."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L685` `self._block_manager.step(ctx)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Paged KV claim and release II - implementation

- File: `max/python/max/pipelines/kv_cache/paged_kv_cache/cache_manager.py`
- Source range: `732-739`
- Function/anchor: `def claim(self, ctx`
- Evidence: `SOURCE`
- Mathematical model: `KV_bytes ~= 2 L n_kv d_h tokens bytes_per_elem`

- `L732` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L733` `def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L734` `"""Returns the replica's KV buffer (single leaf or tree).`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L735` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L736` `HACK: this exists only for the transfer engine; callers flatten via`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L737` `:attr:\`KVCacheBufferInterface.all_buffers\`.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L738` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L739` `return self._kv_buffers[replica_idx]`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

## Part 6: Source Atlas, Debugging, and Invariants

### CompiledModel versus Model I - contract

- File: `max/python/max/engine/api.py`
- Source range: `413-420`
- Function/anchor: `class CompiledModel`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L413` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L414` `class CompiledModel:`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L415` `"""A compiled model artifact, ready for initialization with weights.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L416` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L417` `Returned by :meth:\`InferenceSession.compile\`. Pass it to`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L418` `:meth:\`InferenceSession.init\` or :meth:\`InferenceSession.init_all\` to`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L419` `produce executable :class:\`Model\` instances.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L420` ``
  This blank line separates logical source blocks; it performs no runtime work.

### CompiledModel versus Model II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `502-509`
- Function/anchor: `class CompiledModel`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L502` `f"{what} needs the device that graph {self.name!r} was compiled "`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L503` `"for, and it was compiled under a virtual device, which stands in "`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L504` `"for hardware that is not attached. Initialize the compiled "`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L505` `"artifact on a real device to execute it."`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L506` `)`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L507` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L508` `def execute(self, *args: object, **kwargs: object) -> NoReturn:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L509` `"""Raises: a compile-only stand-in has nothing to execute on."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Asynchronous compilation I - contract

- File: `max/python/max/engine/api.py`
- Source range: `968-975`
- Function/anchor: `def compile_async(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L968` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L969` `def compile_async(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L970` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L971` `model: str | Path | Module | Graph,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L972` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L973` `custom_extensions: CustomExtensionsType | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L974` `tile_based_fusion: bool = False,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L975` `) -> CompiledModel:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Asynchronous compilation II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `994-1001`
- Function/anchor: `def compile_async(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L994` `custom_extensions: The extensions to load for the model.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L995` `Supports paths to \`\`.mojoc\`\` custom ops.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L996` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L997` `tile_based_fusion: When \`\`True\`\`, compile the graph under the tile-based`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L998` `programming model, in which the graph compiler selects`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L999` `tile-based kernels (operating on \`\`TileTensor\`\` values instead`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1000` `of SIMD). Only applies when \`\`model\`\` is a :class:\`Graph\` or`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1001` `:class:\`max.graph.Module\`; ignored for precompiled \`\`.mef\`\``
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Initialize all graphs I - contract

- File: `max/python/max/engine/api.py`
- Source range: `1159-1166`
- Function/anchor: `def init_all(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1159` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L1160` `def init_all(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L1161` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1162` `compiled: CompiledModel,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1163` `*,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1164` `weights_registry: Mapping[str, DLPackArray] | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1165` `) -> dict[str, Model]:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1166` `"""Initializes all models in a compiled artifact for execution.`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Initialize all graphs II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `1182-1189`
- Function/anchor: `def init_all(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1182` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L1183` `if is_virtual_device_mode():`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L1184` `# Virtual device mode can't actually initialize the model, but`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1185` `# users (eg. cross compilation, benchmarking) want it to not fail.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1186` `# Return one stand-in per top-level graph in the artifact so callers`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1187` `# that key by graph name still work; the stand-in refuses anything`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1188` `# that would need the device rather than pretending to have one.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L1189` `if not compiled._graph_names:`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.

### Execute an initialized model I - contract

- File: `max/python/max/engine/api.py`
- Source range: `118-125`
- Function/anchor: `def _Model_execute`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L118` `@traced`
  The annotation attaches compile-time registration or dispatch metadata to the declaration that follows.
- `L119` `def _Model_execute(self: Model, *args: InputType) -> list[Buffer]:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L120` `# Original tensor-only execution path`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L121` `input_impls: list[Buffer] = []`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L122` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L123` `for idx, arg in enumerate(args):`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L124` `_raise_if_not_contiguous(arg)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L125` ``
  This blank line separates logical source blocks; it performs no runtime work.

### Execute an initialized model II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `178-185`
- Function/anchor: `def _Model_execute`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L178` `raise TypeError("graph_keys must be an int or sequence of ints.")`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L179` `if isinstance(graph_keys, int):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L180` `return [_normalize_graph_key(graph_keys)]`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L181` `if isinstance(graph_keys, (str, bytes)):`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L182` `raise TypeError("graph_keys must be a sequence of ints.")`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L183` `normalized: list[int] = []`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L184` `for graph_key in graph_keys:`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L185` `normalized.append(_normalize_graph_key(graph_key))`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Llama model execute wrapper I - contract

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `200-207`
- Function/anchor: `def execute(self, model_inputs`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L200` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L201` `def execute(self, model_inputs: ModelInputs) -> ModelOutputs:`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L202` `assert isinstance(model_inputs, Llama3Inputs)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L203` `assert model_inputs.kv_cache_inputs is not None`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L204` `model_outputs = self.model.execute(*model_inputs.buffers)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L205` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L206` `assert self.batch_processor is not None`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L207` `return self.batch_processor.process_outputs(model_outputs)`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Llama model execute wrapper II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `245-252`
- Function/anchor: `def execute(self, model_inputs`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L245` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L246` `# TODO(kathywu): Drop once every subclass implements graph_specs.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L247` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L248` `type(self)._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L249` `is not GraphPipelineModelWithKVCache._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L250` `):`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L251` `return super().load_model(session)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L252` `return self.load_graphs(session)[GraphRole.DECODER]`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Initial token inputs I - contract

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `185-192`
- Function/anchor: `def prepare_initial_token_inputs`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `row_i = tokens[offset_i : offset_(i+1)]`

- `L185` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L186` `def prepare_initial_token_inputs(`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L187` `self,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L188` `replica_batches: Sequence[Sequence[TextContext]],`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L189` `kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L190` `return_n_logits: int = 1,`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L191` `) -> Llama3Inputs:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L192` `"""Delegates to the batch processor and narrows to \`\`Llama3Inputs\`\`."""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### Initial token inputs II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `245-252`
- Function/anchor: `def prepare_initial_token_inputs`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `row_i = tokens[offset_i : offset_(i+1)]`

- `L245` `"""`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L246` `# TODO(kathywu): Drop once every subclass implements graph_specs.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L247` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
- `L248` `type(self)._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L249` `is not GraphPipelineModelWithKVCache._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L250` `):`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L251` `return super().load_model(session)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L252` `return self.load_graphs(session)[GraphRole.DECODER]`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Attach Python stack traces I - contract

- File: `GraphCompiler/lib/MGPDialect/Transforms/AttachPythonStackTraces.cpp`
- Source range: `54-61`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L54` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L55` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L56` `mlir::MLIRContext *ctx = &getContext();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L57` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L58` `getOperation()->walk([&ctx](Operation *op) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L59` `Location loc = op->getLoc();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L60` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L61` `// Only the op's immediate OpaqueLoc applies, not ones nested in a`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### Attach Python stack traces II - implementation

- File: `GraphCompiler/lib/MGPDialect/Transforms/AttachPythonStackTraces.cpp`
- Source range: `63-70`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L63` `if (auto assertOp = dyn_cast<AssertOp>(op)) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L64` `if (auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc)) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L65` `llvm::StringRef stackTrace = extractStackTrace(opaqueLoc);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L66` `if (!stackTrace.empty()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L67` `std::string fullMsg = "\nSource Traceback:\n";`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L68` `llvm::raw_string_ostream ss(fullMsg);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L69` `ss << stackTrace << '\n';`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L70` `StringAttr msg = assertOp.getMessageAttr();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Simple buffer reuse I - contract

- File: `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`
- Source range: `94-101`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `reuse(a,b) only if live(a) intersect live(b) = empty`

- `L94` `public:`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L95` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L96` `ModelOp modelOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L97` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L98` `auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L99` `int reuseCount = 0;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L100` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L101` `// Erasing allocs invalidates the analysis, but the next pass only queries`
  The source comment records an invariant, intent, target distinction, or implementation boundary.

### Simple buffer reuse II - implementation

- File: `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`
- Source range: `111-118`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `reuse(a,b) only if live(a) intersect live(b) = empty`

- `L111` `for (auto allocOp : block->getOps<MGP::BufferAllocOp>()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L112` `ReuseKey key = {allocOp.getRuntimeSize(), allocOp.getDevice(),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L113` `allocOp.getRawAlign()};`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L114` `allocGroups[key].push_back(allocOp);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L115` `}`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L116` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L117` `if (allocGroups.empty())`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L118` `return WalkResult::advance();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.

### Dynamic memory planning I - contract

- File: `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
- Source range: `84-91`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L84` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L85` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L86` `ModelOp modelOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L87` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L88` `auto getDeviceContext = [&modelOp](DeviceRefAttr dev,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L89` `ArrayRef<BlockArgument> devices) {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L90` `for (auto [deviceRefAttr, deviceContext] :`
  This loop expands repeated work while preserving the ordering and dependency rules of the surrounding operation.
- `L91` `llvm::zip_equal(modelOp.getDevices(), devices))`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### Dynamic memory planning II - implementation

- File: `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
- Source range: `111-118`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L111` `for (auto allocOp : execState.execute->getOps<MGP::BufferAllocOp>()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L112` `// TODO(GEX-557): Plan CPU allocations too.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L113` `if (allocOp.getDevice().getLabel() == kCPULabel)`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L114` `continue;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L115` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L116` `ErrorOr<SmallVector<Operation *>> allocUses = getAllocUses(allocOp);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L117` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L118` `if (allocUses.isError() || allocUses->empty())`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### Emit Mojo kernels I - contract

- File: `GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp`
- Source range: `140-147`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L140` `using LiftBufferCacheSlotsBase::LiftBufferCacheSlotsBase;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L141` `void runOnOperation() override { liftBufferCacheSlots(getOperation()); }`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L142` `};`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L143` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L144` `struct EmitMojo : public impl::EmitMojoBase<EmitMojo> {`
  This type creates an ownership and behavior boundary for state used by later compiler or runtime calls.
- `L145` `using EmitMojoBase::EmitMojoBase;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L146` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L147` `void emitMojoText() {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### Emit Mojo kernels II - implementation

- File: `GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp`
- Source range: `206-213`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L206` `// Compute a base file name for the dump from the names of non-subgraph`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L207` `// ModelOps (user-visible model names) joined by "+", or "graph" when there`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L208` `// are none. Mirrors the \`<names>.mo.mlir\` stem pipeline_to_mo_graph.py`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L209` `// writes, so the .mojo dump lands beside the source graph's stages.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L210` `std::string baseFileName;`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L211` `for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L212` `if (!modelOp.getIsSubgraph()) {`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L213` `if (!baseFileName.empty())`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.

### JIT compile kernels I - contract

- File: `GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp`
- Source range: `55-62`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L55` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L56` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L57` `ModuleOp module = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L58` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L59` `KGEN::CompilerTimeTraceScope traceScope("jit-compile-kernels",`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L60` `module->getName().getStringRef());`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L61` ``
  This blank line separates logical source blocks; it performs no runtime work.
- `L62` `initLLVM();`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.

### JIT compile kernels II - implementation

- File: `GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp`
- Source range: `187-194`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L187` `// cross-compiled target could hit a host-warmed artifact, or a`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L188` `// host-compiled fold kernel could poison the cross target's key.`
  The source comment records an invariant, intent, target distinction, or implementation boundary.
- `L189` `auto hostTargetOr = MOGG::getHostTargetInfoAttr(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L190` `ctx, defaultDeviceTargetArch, defaultDeviceTargetTriple,`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L191` `compileForHostExecution);`
  This statement invokes or composes the next operation in the local dataflow; its arguments identify the state crossing the boundary.
- `L192` `if (hostTargetOr.isError())`
  This declaration defines a callable contract. Its parameters and return type determine which state can cross the boundary.
- `L193` `return kgenModuleOp->emitError() << hostTargetOr.takeError();`
  The returned object becomes state consumed by the caller or the next compiler/runtime stage.
- `L194` `if (auto error = MOGG::graphCompilerCacheKey(`
  This branch selects behavior from runtime or compile-time facts such as shape, target, device, or scheduler state.
