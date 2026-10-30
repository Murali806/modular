# Line-by-Line Source Atlas

Revision: `eab325f81bdbcf0195c511a56510f65c37540798`

This atlas explains every source line displayed by the 132 animated scenes. The first window captures each callable's contract; the second selects a high-signal implementation block nearby. Follow the file link for the complete source body.

## Part 1: Graph Construction and Finalization

### Graph object construction I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `705-712`
- Function/anchor: `class Graph:`
- Evidence: `SOURCE`
- Mathematical model: `G = (V, E, types, devices)`

- `L705` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L706` `class Graph:`
  This type creates an ownership and behavior boundary for the state used by later calls.
- `L707` `"""Represents a single MAX graph.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L708` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L709` `A :class:\`Graph\` defines a model's computation. You build a graph by`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L710` `composing operations that describe how input tensors are transformed into`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L711` `outputs. Unlike imperative code that executes operations, a :class:\`Graph\``
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L712` `captures the data flow between operations, which allows MAX to optimize and`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Graph object construction II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `736-743`
- Function/anchor: `class Graph:`
- Evidence: `SOURCE`
- Mathematical model: `G = (V, E, types, devices)`

- `L736` `) as graph:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L737` `x = graph.inputs[0].tensor`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L738` `y = x @ W + b`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L739` `graph.output(y)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L740` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L741` `**Use the graph constructor:** Pass a callable as the \`\`forward\`\` argument.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L742` `The graph automatically passes the input :class:\`TensorValue\` to the`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L743` `callable and records the return value as the graph output. Under the hood,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Graph constructor body I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `819-826`
- Function/anchor: `def __init__(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L819` `@in_default_mlir_context`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L820` `def __init__(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L821` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L822` `name: str,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L823` `forward: Callable[..., GraphOutputs | None] | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L824` `input_types: Iterable[Type[Any]] = (),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L825` `path: Path | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L826` `*args,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Graph constructor body II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `909-916`
- Function/anchor: `def __init__(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L909` `self._kernel_library = kernel_library or KernelLibrary()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L910` `self._import_kernels(_resolved_custom_extensions(custom_extensions))`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L911` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L912` `self._subgraphs = {}`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L913` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L914` `# If we're building a device graph, annotate the graph appropriately.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L915` `if is_device_graph:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L916` `op.is_device_graph = builtin.UnitAttr()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Kernel library loading I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `268-275`
- Function/anchor: `def load_paths(self, custom_extensions`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L268` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L269` `def load_paths(self, custom_extensions: Iterable[Path]) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L270` `"""Loads custom operations from provided library paths.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L271` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L272` `Performs additional "smart" library loading logic for custom operation`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L273` `libraries in additional formats. The loading logic supports the`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L274` `following formats:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L275` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

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
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L319` `self._analysis = cached`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L320` `self._analysis.verify_custom_op(custom_op)`
  This validates the current IR or object contract and surfaces diagnostics close to the originating source operation.
- `L321` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L322` `self._analysis.verify_custom_op(custom_op)`
  This validates the current IR or object contract and surfaces diagnostics close to the originating source operation.
- `L323` `KernelLibrary._analysis_cache[cache_key] = self._analysis`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Per-operation verification I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `1185-1192`
- Function/anchor: `def _verify_op(self, op`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1185` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1186` `def _verify_op(self, op: mlir.Operation | mlir.OpView | Operation) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1187` `if self._should_verify_ops:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1188` `with self._capturing_mlir_diagnostics():`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1189` `op.verify()`
  This validates the current IR or object contract and surfaces diagnostics close to the originating source operation.
- `L1190` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1191` `@contextlib.contextmanager`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L1192` `def _capturing_mlir_diagnostics(self):  # noqa: ANN202`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### Per-operation verification II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `1366-1373`
- Function/anchor: `def _verify_op(self, op`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1366` `def _populate_device_info_mapping(self) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1367` `"""Attaches mo.device_info_mapping to the module if not already present."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1368` `module = self._mlir_op.block.owner`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1369` `if _DEVICE_INFO_MAPPING_ATTR_NAME in module.attributes:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1370` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1371` `devices: list[Device] = [CPU()]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1372` `if accelerator_count() > 0:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1373` `devices.append(Accelerator())`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Device metadata population I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `1365-1372`
- Function/anchor: `def _populate_device_info_mapping`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1365` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1366` `def _populate_device_info_mapping(self) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1367` `"""Attaches mo.device_info_mapping to the module if not already present."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1368` `module = self._mlir_op.block.owner`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1369` `if _DEVICE_INFO_MAPPING_ATTR_NAME in module.attributes:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1370` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1371` `devices: list[Device] = [CPU()]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1372` `if accelerator_count() > 0:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### Device metadata population II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `1538-1545`
- Function/anchor: `def _populate_device_info_mapping`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1538` `Raises:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1539` `TypeError: If the graph has not yet been terminated by a call to`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1540` `:meth:\`output()\`.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1541` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1542` `types = self.output_types`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1543` `if self._output_structure is None:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1544` `return tuple(types)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1545` `return tree.unflatten(self._output_structure, types)`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Graph.output finalization I - contract

- File: `max/python/max/graph/graph.py`
- Source range: `1395-1402`
- Function/anchor: `def output(self, *outputs`
- Evidence: `SOURCE`
- Mathematical model: `f : (T_0,...,T_n) -> (R_0,...,R_m)`

- `L1395` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1396` `def output(self, *outputs: GraphOutputs) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1397` `"""Sets the output values of the graph and finalizes construction.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1398` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1399` `Call this once after building all ops. The graph can't be executed`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1400` `until \`\`output()\`\` has been called. Subsequent calls to`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1401` `:attr:\`output_types\` read back the types of the values passed here.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1402` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### Graph.output finalization II - implementation

- File: `max/python/max/graph/graph.py`
- Source range: `1538-1545`
- Function/anchor: `def output(self, *outputs`
- Evidence: `SOURCE`
- Mathematical model: `f : (T_0,...,T_n) -> (R_0,...,R_m)`

- `L1538` `Raises:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1539` `TypeError: If the graph has not yet been terminated by a call to`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1540` `:meth:\`output()\`.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1541` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1542` `types = self.output_types`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1543` `if self._output_structure is None:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1544` `return tuple(types)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1545` `return tree.unflatten(self._output_structure, types)`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Llama config initialization I - contract

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `223-230`
- Function/anchor: `def initialize(`
- Evidence: `SOURCE`
- Mathematical model: `d_h = d_model / n_heads`

- `L223` `@classmethod`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L224` `def initialize(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L225` `cls,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L226` `pipeline_config: PipelineConfig,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L227` `model_config: MAXModelConfig | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L228` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L229` `max_seq_len: int,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L230` `) -> Self:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

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
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L259` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L260` `dtype = supported_encoding_dtype(quantization_encoding)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L261` `cache_dtype = cache_dtype_for_encoding(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L262` `quantization_encoding, model_config.kv_cache.kv_cache_format`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Llama config finalization I - contract

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `396-403`
- Function/anchor: `def finalize(`
- Evidence: `SOURCE`
- Mathematical model: `P ~= Vd + L(4d^2 + 3d d_ff)`

- `L396` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L397` `def finalize(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L398` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L399` `huggingface_config: AutoConfig,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L400` `state_dict: dict[str, WeightData],`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L401` `return_logits: ReturnLogits,`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L402` `return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L403` `norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Llama config finalization II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model_config.py`
- Source range: `420-427`
- Function/anchor: `def finalize(`
- Evidence: `SOURCE`
- Mathematical model: `P ~= Vd + L(4d^2 + 3d d_ff)`

- `L420` `for k, v in state_dict.items()`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.
- `L421` `if k.startswith("language_model.")`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L422` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L423` `elif has_model_prefix:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L424` `normalized_state_dict = {`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L425` `_strip_prefix(k, "model."): v`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L426` `for k, v in state_dict.items()`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.
- `L427` `if k.startswith("model.")`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### Llama symbolic trace I - contract

- File: `max/python/max/pipelines/architectures/llama3/llama3.py`
- Source range: `272-279`
- Function/anchor: `def trace(`
- Evidence: `SOURCE`
- Mathematical model: `logits = LMHead(Transformer(tokens, KV))`

- `L272` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L273` `def trace(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L274` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L275` `*input_types: Type[Any],`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
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
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L290` `name.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L291` `module: The MLIR module to add the graph to. Defaults to a new`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L292` `module.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L293` `custom_extensions: Paths to custom kernel libraries the graph`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L294` `uses.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L295` `is_device_graph: Whether to build a device graph.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L296` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### GraphSpec trace recipe I - contract

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `113-120`
- Function/anchor: `def trace(self, module`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L113` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L114` `def trace(self, module: GraphModule | None = None) -> Graph:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L115` `"""Traces the graph, into \`\`module\`\` if given.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L116` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L117` `Raises:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L118` `ValueError: If \`\`module\`\` is given for a ModuleV3 graph.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L119` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L120` `if isinstance(self.module, ModuleV3):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### GraphSpec trace recipe II - implementation

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `194-201`
- Function/anchor: `def trace(self, module`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L194` `:func:\`compile_roles_separately\`, groups are ignored.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L195` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L196` `Each callable is called with one argument per input type, positionally,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L197` `or with the named input tree as its one argument, and returns the graph's`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L198` `outputs in the tree its \`\`forward\`\` returned. It keeps no weight registry,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L199` `so the host weights can be released after initialization. A ModuleV3`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L200` `role's callable instead takes and returns what the module's \`\`forward\`\``
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L201` `does, adds the signal buffers a multi-device graph takes, and keeps its`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Compile and initialize graphs I - contract

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `182-189`
- Function/anchor: `def compile_and_init_graphs(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L182` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L183` `def compile_and_init_graphs(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L184` `session: InferenceSession, specs: Mapping[str, GraphSpec]`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L185` `) -> dict[str, CompiledCallable[..., Any]]:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L186` `"""Compiles and initializes each role's graph.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L187` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L188` `Roles that share a group trace into one :class:\`~max.graph.Module\`,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L189` `compile once, and initialize once with the weights of all of them; each`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Compile and initialize graphs II - implementation

- File: `max/python/max/pipelines/lib/graph_specs.py`
- Source range: `194-201`
- Function/anchor: `def compile_and_init_graphs(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L194` `:func:\`compile_roles_separately\`, groups are ignored.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L195` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L196` `Each callable is called with one argument per input type, positionally,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L197` `or with the named input tree as its one argument, and returns the graph's`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L198` `outputs in the tree its \`\`forward\`\` returned. It keeps no weight registry,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L199` `so the host weights can be released after initialization. A ModuleV3`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L200` `role's callable instead takes and returns what the module's \`\`forward\`\``
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L201` `does, adds the signal buffers a multi-device graph takes, and keeps its`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

## Part 2: Graph Compiler Transformations

### Top-level graph compiler stages I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `86-93`
- Function/anchor: `static void gcPipelineImpl`
- Evidence: `SOURCE`
- Mathematical model: `IR_(k+1) = Pass_k(IR_k)`

- `L86` `namespace {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L87` `static void gcPipelineImpl(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L88` `const GCPipelineOptions &opts, bool skipJit,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L89` `bool dumpIntermediates = false);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L90` `} // namespace`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L91` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L92` `void M::MO::registerGCPipeline() {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L93` `mlir::PassPipelineRegistration<GCPipelineOptions>(`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Top-level graph compiler stages II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `227-234`
- Function/anchor: `static void gcPipelineImpl`
- Evidence: `SOURCE`
- Mathematical model: `IR_(k+1) = Pass_k(IR_k)`

- `L227` `if (!opts.kernelDeclsPrePopulated) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L228` `pm.addPass(MOGG::createImportPackages(`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L229` `{.mojoPkgPaths = kernelLibraryPaths, .litMlirPaths = {}}));`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L230` `pm.addPass(MOGG::createVerifyKernels());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L231` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L232` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L233` `pm.addPass(MOGG::createRewriteOpaqueTypes());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L234` `pm.addPass(KGEN::createVerifyParameters());`
  This line fixes transformation order; downstream passes consume the invariants produced here.

### MO preprocessing pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `218-225`
- Function/anchor: `void M::MO::buildMoPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L218` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L219` `void M::MO::buildMoPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L220` `const MoPipelineOptions &opts) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L221` `auto kernelLibraryPaths =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L222` `llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L223` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L224` `// Skipped when the caller already seeded decls and`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L225` `// \`mogg.opaque_type_mapping\` via \`MOGG::copyKernelDecls\`. RewriteOpaqueTypes`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### MO preprocessing pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `393-400`
- Function/anchor: `void M::MO::buildMoPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L393` `// After ExecInvariantCodeMotion: buffer reuse can block hoisting.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L394` `pm.addPass(MGP::createSimpleBufferReusePass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L395` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L396` `pm.addPass(MGP::createDynamicMemoryPlanningPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L397` `pm.addPass(MGP::createDeferAllocsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L398` `pm.addPass(MGP::createVerifyModelArgumentsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L399` `pm.addPass(MGP::createPredicateAssertDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L400` `pm.addPass(MGP::createSimplifyChainsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.

### MAP optimization pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `337-344`
- Function/anchor: `void M::MO::buildMapPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L337` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L338` `void M::MO::buildMapPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L339` `const MapPipelineOptions &opts) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L340` `pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L341` `pm.addNestedPass<GraphOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L342` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L343` `pm.addNestedPass<GraphOp>(MAP::createFuseKernels());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L344` `if (opts.fuseMutableLoadStore)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### MAP optimization pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `417-424`
- Function/anchor: `void M::MO::buildMapPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L417` `pm.addPass(MGP::createStageDeviceGraphsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L418` `pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L419` `pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L420` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L421` `pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L422` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L423` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L424` `pm.addPass(MGP::createGenerateBuildGraphStubsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.

### MOGG routine pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `369-376`
- Function/anchor: `void M::MO::buildMoggPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L369` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L370` `void M::MO::buildMoggPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L371` `const MoggPipelineOptions &) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L372` `pm.addPass(MOGG::createLegalizeRowwiseKernels());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L373` `pm.addPass(MOGG::createConformCallOperands());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L374` `pm.addPass(MOGG::createOutlineRoutines());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L375` `pm.addPass(MOGG::createWrapFuncsWithTrace());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L376` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### MOGG routine pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `417-424`
- Function/anchor: `void M::MO::buildMoggPipeline`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L417` `pm.addPass(MGP::createStageDeviceGraphsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L418` `pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L419` `pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L420` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L421` `pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L422` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L423` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L424` `pm.addPass(MGP::createGenerateBuildGraphStubsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.

### MGP runtime pipeline I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `381-388`
- Function/anchor: `void M::MO::buildMgpPipeline`
- Evidence: `SOURCE`
- Mathematical model: `peak = max_t sum_(b live at t) bytes(b)`

- `L381` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L382` `void M::MO::buildMgpPipeline(mlir::OpPassManager &pm,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L383` `const MgpPipelineOptions &opts) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L384` `auto kernelLibraryPaths =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L385` `llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L386` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L387` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L388` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### MGP runtime pipeline II - implementation

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `417-424`
- Function/anchor: `void M::MO::buildMgpPipeline`
- Evidence: `SOURCE`
- Mathematical model: `peak = max_t sum_(b live at t) bytes(b)`

- `L417` `pm.addPass(MGP::createStageDeviceGraphsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L418` `pm.addPass(MGP::createPartitionDeviceGraphCollectivePass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L419` `pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L420` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L421` `pm.addPass(MGP::createSimplifyDeviceGraphDependenciesPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L422` `pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L423` `pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L424` `pm.addPass(MGP::createGenerateBuildGraphStubsPass());`
  This line fixes transformation order; downstream passes consume the invariants produced here.

### MO to MAP conversion I - contract

- File: `GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp`
- Source range: `766-773`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L766` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L767` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L768` `MOGG::MojoRegistry registry(getOperation());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L769` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L770` `SmallVector<Operation *> worklist;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L771` `WalkResult walkResult = getOperation()->walk([&](Operation *op) {`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L772` `Dialect *dialect = op->getDialect();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L773` `if (dialect && isa<MO::MODialect>(dialect) && !hasLowerableRank(op)) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### MO to MAP conversion II - implementation

- File: `GraphCompiler/lib/Conversion/MOToMAP/MOToMAP.cpp`
- Source range: `782-789`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L782` `return WalkResult::advance();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L783` `});`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L784` `if (walkResult.wasInterrupted())`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L785` `return signalPassFailure();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L786` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L787` `for (Operation *op : worklist)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L788` `if (failed(lowerOp(op, registry)))`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L789` `return signalPassFailure();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### MAP kernel fusion I - contract

- File: `GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp`
- Source range: `36-43`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `bytes_saved ~= read(tmp) + write(tmp)`

- `L36` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L37` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L38` `MO::GraphOp graphOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L39` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L40` `// One fusion round: input fusion before output fusion, so a`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L41` `// view/elementwise producer fuses into a foreach consumer before that`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L42` `// consumer is output-fused into an epilogue -- which is what lets a view`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L43` `// reach an epilogue without fusing directly into a closure. Prologue runs`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### MAP kernel fusion II - implementation

- File: `GraphCompiler/lib/MAPDialect/Transforms/FuseKernels.cpp`
- Source range: `46-53`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `bytes_saved ~= read(tmp) + write(tmp)`

- `L46` `mlir::OpPassManager fusePm(MO::GraphOp::getOperationName());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L47` `fusePm.addPass(createFuseElementwise());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L48` `fusePm.addPass(createFuseViews());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L49` `fusePm.addPass(createFusePrologue());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L50` `fusePm.addPass(createFuseReductions());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L51` `fusePm.addPass(createFuseEpilogue());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L52` `fusePm.addPass(createFuseReshapeEpilogue());`
  This line fixes transformation order; downstream passes consume the invariants produced here.
- `L53` `// Last, so siblings merge only once the producer-consumer fusers have`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### Reduction fusion I - contract

- File: `GraphCompiler/lib/MAPDialect/Transforms/ReductionFuser.cpp`
- Source range: `301-308`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L301` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L302` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L303` `MO::GraphOp graphOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L304` `IRRewriter rewriter(graphOp.getContext());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L305` `SmallVector<IterRowwiseOp> rowwiseOps;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L306` `graphOp.getBody().walk([&](IterRowwiseOp op) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L307` `rowwiseOps.push_back(op);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L308` `return WalkResult::skip();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Reduction fusion II - implementation

- File: `GraphCompiler/lib/MAPDialect/Transforms/ReductionFuser.cpp`
- Source range: `315-322`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L315` `while (IterElementwiseOp src = findProducer(dst)) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L316` `fuseProducer(dst, src, rewriter);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L317` `rewriter.eraseOp(src);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L318` `++numKernelsFused;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L319` `changed = true;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L320` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L321` `if (IterElementwiseOp consumer = findConsumer(dst)) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L322` `dst = fuseConsumer(dst, consumer, rewriter);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### MAP to MOGG conversion I - contract

- File: `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`
- Source range: `55-62`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L55` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L56` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L57` `MOGG::MojoRegistry registry(getOperation());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L58` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L59` `SmallVector<MAP::IterOpaqueOp> worklist;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L60` `getOperation()->walk([&](MAP::IterOpaqueOp op) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L61` `worklist.push_back(op);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L62` `return WalkResult::skip();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### MAP to MOGG conversion II - implementation

- File: `GraphCompiler/lib/Conversion/MAPToMOGG/MAPToMOGG.cpp`
- Source range: `74-81`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L74` `MAPToMOGGState state(registry, op);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L75` `state.enableComputeLambdas = enableComputeLambdas;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L76` `return failed(lowerInvokeShapeFuncToMOGG(op, state))`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L77` `? WalkResult::interrupt()`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L78` `: WalkResult::advance();`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L79` `});`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L80` `if (result.wasInterrupted())`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L81` `return signalPassFailure();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### MOGG to MGP conversion I - contract

- File: `GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp`
- Source range: `2405-2412`
- Function/anchor: `MOGGToMGPPass::runOnOperation`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L2405` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L2406` `void MOGGToMGPPass::runOnOperation() {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L2407` `ModuleOp theModule = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2408` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L2409` `// Kernel stubs created during lowering are registered through the cached`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L2410` `// \`SymbolTableAnalysis\`, which the pass manager never invalidates.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L2411` `SymbolTable &moduleSymbolTable =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2412` `getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### MOGG to MGP conversion II - implementation

- File: `GraphCompiler/lib/Conversion/MOGGToMGP/MOGGToMGP.cpp`
- Source range: `2429-2436`
- Function/anchor: `MOGGToMGPPass::runOnOperation`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L2429` `size_t lastProcessed = 0;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2430` `// Process graphs in dependency order.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L2431` `while (!worklist.empty()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L2432` `// A full pass with no progress means a cycle.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L2433` `if (lastProcessed > worklist.size()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L2434` `theModule.emitError("cycle detected in graph ops");`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L2435` `return signalPassFailure();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L2436` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### MEF serialization boundary I - contract

- File: `GraphCompiler/lib/GraphCompiler.cpp`
- Source range: `691-698`
- Function/anchor: `convertMGPToMEF`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `MEF = serialize(finalized MGP)`

- `L691` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L692` `ErrorOr<BufferRef> M::MO::convertMGPToMEF(ModuleOp module,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L693` `bool enableMEFMetadata) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L694` `OwningOpRef<ModuleOp> moduleToRewrite = module.clone();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L695` `// Device-generic ops get renamed with their device label so GraphRT can`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L696` `// dispatch to a device-appropriate primitive.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
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
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L702` `if (failed(pm.run(*moduleToRewrite)))`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L703` `return Error(Twine("Failed to prepare MGP model for conversion to MEF. ") +`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L704` `diagnostics.formatMessage());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L705` `ConcatenationTree result =`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L706` `translateMLIRToMEF(*moduleToRewrite, enableMEFMetadata);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L707` `if (result.getSize() == 0) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L708` `return Error(Twine("Failed to convert model to MEF. ") +`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

## Part 3: Kernel Development and Optimization

### Kernel extensibility registration I - contract

- File: `max/kernels/src/graph_compiler/extensibility/decorators.mojo`
- Source range: `45-52`
- Function/anchor: `def register(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L45` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L46` `def register(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L47` `name: StaticString,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L48` `type: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L49` `api: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L50` `arch: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L51` `model: StaticString = "",`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L52` `):`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Kernel extensibility registration II - implementation

- File: `max/kernels/src/graph_compiler/extensibility/decorators.mojo`
- Source range: `66-73`
- Function/anchor: `def register(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L66` `Fields are ordered from least to most specific: \`type\` < \`api\` < \`arch\` <`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L67` `\`model\`. Among all matching candidates, the most specific registration wins:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L68` `a registration that sets \`arch\` is more specific than one that sets only`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L69` `\`type\` and \`api\`, because \`arch\` sits higher in the hierarchy.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L70` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L71` `If a user-defined kernel and a built-in kernel are equally specific for a`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L72` `device, the user-defined kernel takes precedence.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L73` `Registering the same op with identical device fields more than once,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### GPU matmul registration I - contract

- File: `max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo`
- Source range: `350-357`
- Function/anchor: `@extensibility.register("mo.matmul", type="gpu")`
- Evidence: `SOURCE`
- Mathematical model: `C_(M,N) = A_(M,K) B_(K,N)`

- `L350` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L351` `@extensibility.register("mo.matmul", type="gpu")`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L352` `struct MatmulGPU:`
  This type creates an ownership and behavior boundary for the state used by later calls.
- `L353` `"""Registers the \`mo.matmul\` graph op for GPU devices."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L354` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L355` `@staticmethod`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L356` `def execute[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L357` `transpose_b: Bool,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### GPU matmul registration II - implementation

- File: `max/kernels/src/graph_compiler/builtin_kernels/linalg.mojo`
- Source range: `385-392`
- Function/anchor: `@extensibility.register("mo.matmul", type="gpu")`
- Evidence: `SOURCE`
- Mathematical model: `C_(M,N) = A_(M,K) B_(K,N)`

- `L385` `](coords: Coord, val: SIMD[_dtype, _width]) {var c} -> SIMD[`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L386` `_dtype, _width`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L387` `]:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L388` `comptime assert (`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L389` `coords.rank == 2`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L390` `), "the matmul epilogue contract is rank 2"`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L391` `return rebind[SIMD[_dtype, _width]](`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L392` `c._fused_compute_output_lambda[element_alignment=alignment](`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Generic GPU matmul kernel I - contract

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `106-113`
- Function/anchor: `def matmul_kernel[`
- Evidence: `SOURCE`
- Mathematical model: `FLOPs = 2 M N K`

- `L106` `@__name(t"matmul_kernel_{c_type}_{a_type}_{b_type}_{tile_size}")`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L107` `def matmul_kernel[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L108` `c_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L109` `a_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L110` `b_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L111` `tile_size: Int,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L112` `s_type: DType = get_accum_type[c_type](),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L113` `](`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Generic GPU matmul kernel II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `207-214`
- Function/anchor: `def matmul_kernel[`
- Evidence: `SOURCE`
- Mathematical model: `FLOPs = 2 M N K`

- `L207` `# If K is not multiple of tile_size, the last tile contains less than`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L208` `# tile_size elements. The thread block needs to take addition bound check`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L209` `# when loading elements into shared memory.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L210` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L211` `# Load A tile into shared memory.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L212` `var a_val: Scalar[a_type]`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L213` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L214` `comptime if not full_tile:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### Naive matmul baseline I - contract

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `253-260`
- Function/anchor: `def matmul_kernel_naive[`
- Evidence: `SOURCE`
- Mathematical model: `AI_naive ~= 2K / (2K b + b)`

- `L253` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L254` `def matmul_kernel_naive[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L255` `c_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L256` `a_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L257` `b_type: DType,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L258` `c_layout_type: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L259` `a_layout_type: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L260` `b_layout_type: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Naive matmul baseline II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `406-413`
- Function/anchor: `def matmul_kernel_naive[`
- Evidence: `SOURCE`
- Mathematical model: `AI_naive ~= 2K / (2K b + b)`

- `L406` `else:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L407` `for i in range(_k):`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.
- `L408` `accum += (`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L409` `a.load[width=1]((x, i)).cast[s_type]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L410` `* b.load[width=1]((i, y)).cast[s_type]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L411` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L412` `comptime if has_epilogue_fn:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L413` `epilogue_fn[c_type, 1, alignment=1]((x, y), accum.cast[c_type]())`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Asynchronous DRAM to SRAM copy I - contract

- File: `max/kernels/src/layout/tile_io.mojo`
- Source range: `1245-1252`
- Function/anchor: `def copy_dram_to_sram_async[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `T_stage ~= max(T_copy, T_compute) when overlap holds`

- `L1245` `@inline(.nodebug)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L1246` `def copy_dram_to_sram_async[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1247` `thread_layout: Layout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1248` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
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
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1259` `"""Asynchronously copies a tile from DRAM (generic memory) to SRAM (shared).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1260` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1261` `Delegates to \`GenericToSharedAsyncTileCopier\`, which issues NVIDIA`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1262` `\`cp.async\` instructions (falling back to synchronous loads/stores on AMD`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1263` `and Apple GPUs). The copy is asynchronous: callers must commit it via`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1264` `\`async_copy_commit_group()\` and synchronize via \`async_copy_wait_all()\``
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1265` `or \`async_copy_wait_group()\` before reading the destination tile.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Ldmatrix asynchronous path I - contract

- File: `max/kernels/src/layout/tile_async_copy.mojo`
- Source range: `24-31`
- Function/anchor: `def copy_dram_to_sram_async_ldmatrix[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L24` `@always_inline`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L25` `def copy_dram_to_sram_async_ldmatrix[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L26` `thread_layout: Layout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L27` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L28` `simd_size: Int,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L29` `swizzle: Bool,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
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
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L56` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L57` `Args:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L58` `dst: Destination tile in shared memory, with a static row stride.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L59` `src: Source tile in global memory.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L60` `num_valid_rows: Valid rows of \`src\`; only consulted when \`masked\`.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L61` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L62` `comptime if is_nvidia_gpu():`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### Persistent SM90 matmul I - contract

- File: `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernel_persistent.mojo`
- Source range: `57-64`
- Function/anchor: `def run_persistent[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `tiles_per_CTA > 1`

- `L57` `@__llvm_arg_metadata(c_tma_op, \`nvvm.grid_constant\`)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L58` `def run_persistent[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L59` `a_tile_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L60` `b_tile_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L61` `c_tile_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L62` `a_desc_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L63` `b_desc_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L64` `c_desc_shape: Coord,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

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
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L107` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L108` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L109` `# Create TileLoaderTMA loaders`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L110` `var a_loader, b_loader = Self.build_tma_loaders(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L111` `a_tma_op, b_tma_op, rank_m, rank_n`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### AMD ping-pong pipeline I - contract

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo`
- Source range: `424-431`
- Function/anchor: `def _run_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L424` `@inline(.always)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L425` `def _run_impl[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L426` `a_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L427` `b_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L428` `c_layout: TensorLayout,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L429` `a_engine: TensorEngine,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L430` `b_engine: TensorEngine,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L431` `c_engine: TensorEngine,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### AMD ping-pong pipeline II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_ping_pong_matmul.mojo`
- Source range: `480-487`
- Function/anchor: `def _run_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L480` `# === SMEM: double-buffered half-tiles ===`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L481` `# A: 2 stages x 2 warp_m groups`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L482` `comptime a_half_layout = row_major[half_BM, BK]()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L483` `var a_s0_g0 = stack_allocation[Self.in_type, address_space=.SHARED](`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L484` `a_half_layout`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L485` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L486` `var a_s0_g1 = stack_allocation[Self.in_type, address_space=.SHARED](`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L487` `a_half_layout`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Flash-attention dispatch I - contract

- File: `max/kernels/src/graph_compiler/builtin_kernels/attention.mojo`
- Source range: `1139-1146`
- Function/anchor: `def _dispatch_flash_attention[`
- Evidence: `SOURCE`
- Mathematical model: `softmax(QK^T/sqrt(d_h))V`

- `L1139` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1140` `def _dispatch_flash_attention[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1141` `mask_t: MHAMask`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1142` `](mask: mask_t) raises {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1143` `var output_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1144` `var q_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1145` `var k_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1146` `var v_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Flash-attention dispatch II - implementation

- File: `max/kernels/src/graph_compiler/builtin_kernels/attention.mojo`
- Source range: `1283-1290`
- Function/anchor: `def _dispatch_flash_attention[`
- Evidence: `SOURCE`
- Mathematical model: `softmax(QK^T/sqrt(d_h))V`

- `L1283` `var output_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1284` `var q_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1285` `var k_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1286` `var v_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1287` `imm,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1288` `}:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1289` `flash_attention_ragged[](`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1290` `output_buffer,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Paged KV cache store kernel I - contract

- File: `max/kernels/src/graph_compiler/builtin_kernels/kv_cache.mojo`
- Source range: `76-83`
- Function/anchor: `@extensibility.register("mo.kv_cache.store.paged.ragged")`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L76` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L77` `@extensibility.register("mo.kv_cache.store.paged.ragged")`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L78` `struct Struct_kv_cache_store_paged:`
  This type creates an ownership and behavior boundary for the state used by later calls.
- `L79` `"""Registers the \`mo.kv_cache.store.paged.ragged\` graph op with the graph compiler.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L80` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L81` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L82` `@inline(.always)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L83` `@staticmethod`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.

### Paged KV cache store kernel II - implementation

- File: `max/kernels/src/graph_compiler/builtin_kernels/kv_cache.mojo`
- Source range: `192-199`
- Function/anchor: `@extensibility.register("mo.kv_cache.store.paged.ragged")`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L192` `@extensibility.register("mo.kv_cache.store_k_scales.paged.ragged")`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L193` `struct Struct_kv_cache_store_k_scales_paged:`
  This type creates an ownership and behavior boundary for the state used by later calls.
- `L194` `"""Registers the \`mo.kv_cache.store_k_scales.paged.ragged\` graph op with the graph compiler.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L195` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L196` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L197` `@inline(.always)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L198` `@staticmethod`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L199` `def execute[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### Arithmetic intensity and roofline I - contract

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `803-810`
- Function/anchor: `def _matmul_gpu_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `Perf <= min(PeakFLOPs, AI * Bandwidth); AI = FLOPs / bytes`

- `L803` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L804` `def _matmul_gpu_impl[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L805` `EpilogueFnType: ElementwiseEpilogueFn,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L806` `ComputeFnType: ElementwiseComputeFn,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L807` `//,`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L808` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L809` `use_tensor_core: Bool,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L810` `transpose_b: Bool,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Arithmetic intensity and roofline II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/__init__.mojo`
- Source range: `971-978`
- Function/anchor: `def _matmul_gpu_impl[`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `Perf <= min(PeakFLOPs, AI * Bandwidth); AI = FLOPs / bytes`

- `L971` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L972` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L973` `# m > 1 (not m >= 64): the kernel already handles a partial M tile`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L974` `# (per-simdgroup \`_bounded_load\`/\`_bounded_store\` + the row_base>=M`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L975` `# early return), so 1 < m < 64 (concurrent-decode batch widths) is`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L976` `# correct here. Routing it off the naive per-row fallback onto this`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L977` `# co-batched GEMM is 6-27x faster at real Llama decode shapes`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L978` `# (microbench, M5 Max). m == 1 stays on the gemv path: a rank-1`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

## Part 4: Driver, Offload, Memory, and Backends

### Compile an offloaded Mojo function I - contract

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `315-322`
- Function/anchor: `def _compile_inner[`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L315` `@inline(.always)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L316` `def _compile_inner[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L317` `fn_type: TrivialRegisterPassable,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L318` `func: fn_type,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L319` `](self) -> CompiledFunctionInfo[fn_type, func, Self._device_spec_target,]:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L320` `comptime target = Self._device_spec_target`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L321` `comptime emission_kind_id = _get_emission_kind_id[`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L322` `"object"`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Compile an offloaded Mojo function II - implementation

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `465-472`
- Function/anchor: `def _compile_inner[`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L465` `the plugin never frees it, freeing the buffer releases only the`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L466` `plugin's bookkeeping, and the caller must keep the underlying`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L467` `allocation alive for the buffer's lifetime. With \`owning=True\``
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L468` `the buffer frees the region through the plugin's normal path,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L469` `which is only valid for an address that came from this plugin's`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L470` `own allocator (e.g. one released with \`unwrap_memory\`).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L471` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L472` `return Buffer[Self.device_spec](`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Load a compiled bundle I - contract

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `358-365`
- Function/anchor: `def load_bundle[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L358` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L359` `def load_bundle[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L360` `asm_origin: ImmOrigin`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L361` `](`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L362` `self, asm: StringSlice[origin=asm_origin]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L363` `) raises HALError -> RuntimeBundle:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L364` `"""Loads a runtime bundle from pre-compiled binary bytes."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L365` `# Each plugin expects a specific file_type string. PTX text is`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### Load a compiled bundle II - implementation

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `465-472`
- Function/anchor: `def load_bundle[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L465` `the plugin never frees it, freeing the buffer releases only the`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L466` `plugin's bookkeeping, and the caller must keep the underlying`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L467` `allocation alive for the buffer's lifetime. With \`owning=True\``
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L468` `the buffer frees the region through the plugin's normal path,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L469` `which is only valid for an address that came from this plugin's`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L470` `own allocator (e.g. one released with \`unwrap_memory\`).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L471` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L472` `return Buffer[Self.device_spec](`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Resolve a device function I - contract

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `544-551`
- Function/anchor: `def load_function(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L544` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L545` `def load_function(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L546` `self, bundle: RuntimeBundle, var name: String`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L547` `) raises HALError -> FunctionHandle:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L548` `return self._raw[].load_function(self._handle, bundle._handle, name)`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L549` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L550` `def set_function_attribute(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L551` `self, func: FunctionHandle, attribute: Int32, value: Int32`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Resolve a device function II - implementation

- File: `max/driver/src/max_license/_hal/context.mojo`
- Source range: `632-639`
- Function/anchor: `def load_function(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L632` `"""Inserts \`entry\` under \`key\`, re-checking for a load race; acquires`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L633` `and releases the cache lock.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L634` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L635` `If another thread cached \`key\` while the caller was loading, discards`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L636` `the duplicate, unloading its function BEFORE its \`entry\` (and so`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L637` `the bundle that defines the function) is destroyed, and returns the`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L638` `cached entry's handle. Otherwise inserts \`entry\` and returns None."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L639` `self._function_cache_lock_acquire()`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Enqueue a kernel function I - contract

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `967-974`
- Function/anchor: `def enqueue_function[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L967` `@inline(.always)`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L968` `def enqueue_function[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L969` `declared_arg_types: TypeList[Trait=AnyType, ...],`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L970` `ExecutionConfigType: GridBlockExecutionConfig,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L971` `//,`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L972` `func: def(*args: *declared_arg_types) thin -> None,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L973` `*actual_arg_types: DevicePassable,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L974` `](`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Enqueue a kernel function II - implementation

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1091-1098`
- Function/anchor: `def enqueue_function[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1091` `location: Optional[SourceLocation] = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1092` `) raises:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1093` `"""Enqueues a pre-compiled checked function for execution on this device.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1094` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1095` `This overload requires a \`DeviceFunction\` that was compiled with`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1096` `type checking enabled (via \`compile_function\`). The function`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1097` `will verify that the argument types match the declared types at`
  This validates the current IR or object contract and surfaces diagnostics close to the originating source operation.
- `L1098` `compile time.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Allocate a device buffer I - contract

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1470-1477`
- Function/anchor: `def enqueue_create_buffer[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `bytes = count * sizeof(dtype)`

- `L1470` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1471` `def enqueue_create_buffer[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1472` `dtype: DType`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1473` `](self, size: Int) raises -> DeviceBuffer[dtype]:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1474` `"""Enqueues a buffer creation using the \`DeviceBuffer\` constructor.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1475` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1476` `For GPU devices, the space is allocated in the device's global memory.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1477` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### Allocate a device buffer II - implementation

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1519-1526`
- Function/anchor: `def enqueue_create_buffer[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `bytes = count * sizeof(dtype)`

- `L1519` `This function allocates memory on the host that is accessible by the device.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1520` `The memory is page-locked (pinned) for efficient data transfer between host and device.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1521` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1522` `Pinned memory is guaranteed to remain resident in the host's RAM, not be`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1523` `paged/swapped out to disk. Memory allocated normally (for example, using`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1524` `[\`alloc()\`](/docs/std/memory/unsafe_pointer/alloc/))`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1525` `is pageable—individual pages of memory can be moved to secondary storage`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1526` `(disk/SSD) when main memory fills up.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Host to device copy I - contract

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1561-1568`
- Function/anchor: `def enqueue_copy[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1561` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1562` `def enqueue_copy[`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1563` `dtype: DType`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1564` `](`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1565` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1566` `dst_ptr: Pointer[mut=True, Scalar[dtype], _],`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1567` `src_ptr: Pointer[Scalar[dtype], _],`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1568` `size: Int,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Host to device copy II - implementation

- File: `max/driver/src/max_license/_device_context_hal/__init__.mojo`
- Source range: `1673-1680`
- Function/anchor: `def enqueue_copy[`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1673` `var src_view = src_buf._inner[]._buffer.view()`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1674` `if src_context[].get_device_id() == self._device[].id:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1675` `self._stream[].copy_intra_device(dst_view, src_view)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1676` `else:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1677` `# This plugin would misread another driver's context handle.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1678` `if not (src_context[]._raw is self._context[]._raw):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1679` `raise Error(`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1680` `"enqueue_copy source is on another driver; cross-device"`
  This enqueues asynchronous device work; host completion is not implied until an explicit synchronization or dependency requires it.

### Queue synchronization I - contract

- File: `max/driver/src/max_license/_hal/queue.mojo`
- Source range: `404-411`
- Function/anchor: `def synchronize(self)`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L404` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L405` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L406` `def synchronize(self) raises HALError:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L407` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L408` `Totally ordered with respect to other operations within this queue`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L409` `if backed by a stream.`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L410` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L411` `self._raw[].synchronize_queue(self._handle)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Queue synchronization II - implementation

- File: `max/driver/src/max_license/_hal/queue.mojo`
- Source range: `412-419`
- Function/anchor: `def synchronize(self)`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L412` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L413` `def native_handle(self) raises HALError -> OptionalReg[UInt64]:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L414` `"""Returns the backend stream/queue handle, or None if the queue has`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L415` `none (a device with no OS-level stream object, e.g. CPU)."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L416` `return self._raw[].get_optional_queue_property["native_handle", UInt64](`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L417` `self._handle`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L418` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L419` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### NVIDIA NVPTX target traits I - contract

- File: `Mojo/lib/Target/max_license/NVPTX/NVPTXTraits.cpp`
- Source range: `13-20`
- Function/anchor: `NVPTX`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L13` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L14` `#include "Target/max_license/NVPTX/NVPTXTraits.h"`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L15` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L16` `#include "llvm/Support/NVPTXAddrSpace.h"`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L17` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L18` `namespace M::KGEN {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L19` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L20` `const NVPTXTraits &NVPTXTraits::get() {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### NVIDIA NVPTX target traits II - implementation

- File: `Mojo/lib/Target/max_license/NVPTX/NVPTXTraits.cpp`
- Source range: `22-29`
- Function/anchor: `NVPTX`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L22` `return instance;`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L23` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L24` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L25` `std::optional<unsigned>`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L26` `NVPTXTraits::requiredStackAllocationAddressSpace() const {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L27` `return llvm::NVPTXAS::AddressSpace::ADDRESS_SPACE_LOCAL;`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L28` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L29` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### AMD GPU target traits I - contract

- File: `Mojo/lib/Target/max_license/AMDGPU/AMDGPUTraits.cpp`
- Source range: `13-20`
- Function/anchor: `AMDGPU`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L13` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L14` `#include "Target/max_license/AMDGPU/AMDGPUTraits.h"`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L15` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L16` `#include "llvm/Support/AMDGPUAddrSpace.h"`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L17` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L18` `namespace M::KGEN {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L19` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L20` `const AMDGPUTraits &AMDGPUTraits::get() {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### AMD GPU target traits II - implementation

- File: `Mojo/lib/Target/max_license/AMDGPU/AMDGPUTraits.cpp`
- Source range: `22-29`
- Function/anchor: `AMDGPU`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L22` `return instance;`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L23` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L24` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L25` `std::optional<unsigned>`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L26` `AMDGPUTraits::requiredStackAllocationAddressSpace() const {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L27` `return llvm::AMDGPUAS::PRIVATE_ADDRESS;`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L28` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L29` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### AMD target cost model I - contract

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_target.mojo`
- Source range: `15-22`
- Function/anchor: `MI355`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L15` `Platform-specific hardware descriptions for AMD CDNA/RDNA GPUs. Provides:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L16` `- MI355X cost model: per-op latencies, resources, and roles`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L17` `- MI355X pipeline configurations: double-buffer and single-buffer`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L18` `- MI355X target profiles: unified cost model + pipeline config`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L19` `- AMD schedule_group_barrier hints for instruction interleaving`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L20` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L21` `The generic framework (schedule_framework.mojo) has no AMD-specific`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L22` `references. All hardware knowledge lives here.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### AMD target cost model II - implementation

- File: `max/kernels/src/linalg/matmul/gpu/amd/amd_target.mojo`
- Source range: `177-184`
- Function/anchor: `MI355`
- Evidence: `SOURCE + GPU-LAB`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L177` `vm_per_load_a: \`vmcnt\` ops per channel-A global load (defaults to 4).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L178` `vm_per_load_b: \`vmcnt\` ops per channel-B global load (defaults to 4).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L179` `max_globals: Maximum global loads per block, 0 means unlimited`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L180` `(defaults to 1).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L181` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L182` `return TargetProfile(`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
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
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1129` `/// this context has no CUDA-style stream model (e.g. Metal, Hexagon).`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1130` `///`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1131` `/// Override this in a DeviceContext subclass to expose a stream pointer`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1132` `/// through DLPack without modifying the core DLPack bridge code.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1133` `virtual std::optional<intptr_t> getDLPackStream() const {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1134` `return std::nullopt;`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1135` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Qualcomm and Hexagon boundary II - implementation

- File: `MLRT/include/MLRT/Driver/DeviceContext/DeviceContext.h`
- Source range: `1152-1159`
- Function/anchor: `Hexagon`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `backend support != inferred GPU implementation`

- `L1152` `// Note: streams created in this manner are NOT counted in numStreams(),`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1153` `// which only counts internally-managed streams.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1154` `virtual ErrorOr<DeviceStreamRef> createExternalStream(void *externalStream);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1155` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1156` `// Returns a view of this context whose current stream is \`streamId\`. The view`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1157` `// shares this context's full stream set, driver context, and device memory`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1158` `// pool; only the current-stream selector differs. Backends without a`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1159` `// multi-stream model return this context, so all work runs on its single`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

## Part 5: MAX Serve Scheduling and Contexts

### Scheduler iteration I - contract

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `304-311`
- Function/anchor: `def run_iteration(self)`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L304` `@traced`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L305` `def run_iteration(self) -> SchedulerProgress:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L306` `"""The Scheduler routine that creates batches and schedules them on GPU`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L307` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L308` `Returns:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L309` `SchedulerProgress: Indicates whether work was performed in this iteration.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L310` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L311` `# Drain the request queue and add to CE requests`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### Scheduler iteration II - implementation

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `332-339`
- Function/anchor: `def run_iteration(self)`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L332` `and self.pipeline.has_pending_outputs()`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L333` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L334` `if not (inputs or self.support_empty_batches or has_pending_outputs):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L335` `self.scheduler_logger.log_idle(`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L336` `self.batch_constructor.kv_cache,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L337` `self.scheduler_config.data_parallel_degree,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L338` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L339` `return SchedulerProgress.NO_PROGRESS`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Retrieve pending requests I - contract

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `244-251`
- Function/anchor: `def _retrieve_pending_requests`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L244` `@traced`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L245` `def _retrieve_pending_requests(self) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L246` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L247` `Retrieves pending requests from the request queue.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L248` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L249` `This method drains the request queue synchronously and processes`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L250` `any contexts that are available.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L251` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### Retrieve pending requests II - implementation

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `332-339`
- Function/anchor: `def _retrieve_pending_requests`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L332` `and self.pipeline.has_pending_outputs()`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L333` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L334` `if not (inputs or self.support_empty_batches or has_pending_outputs):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L335` `self.scheduler_logger.log_idle(`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L336` `self.batch_constructor.kv_cache,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L337` `self.scheduler_config.data_parallel_degree,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L338` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L339` `return SchedulerProgress.NO_PROGRESS`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Schedule and execute I - contract

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `406-413`
- Function/anchor: `def _schedule(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L406` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L407` `def _schedule(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L408` `self, inputs: TextGenerationInputs[TextContext], lagged: bool = False`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L409` `) -> int:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L410` `"""Returns the number of terminated requests."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L411` `batch_id = self._batch_counter`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L412` `self._batch_counter += 1`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L413` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### Schedule and execute II - implementation

- File: `max/python/max/serve/scheduler/text_generation_scheduler.py`
- Source range: `584-591`
- Function/anchor: `def _schedule(`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L584` `# Build DP batch padder when DP > 1 with device graph capture.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L585` `kv_manager = pipeline.kv_manager`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L586` `dp_padder: DPBatchPadder | None = None`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L587` `assert pipeline_config.model is not None`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L588` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L589` `scheduler_config.data_parallel_degree > 1`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L590` `and pipeline_config.runtime.device_graph_capture`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L591` `):`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Admit a text request I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `783-790`
- Function/anchor: `def _admit_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L783` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L784` `def _admit_request(self, ctx: TextContext, replica_idx: int | None) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L785` `"""Admits a grammar-ready request into the DP pool or a replica queue."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L786` `# DP-balanced CE deferral: fresh CE requests enter an unbound pool and`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L787` `# are bound to a replica by the per-step planner, which prices them by`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L788` `# estimated post-prefix-cache length. Caller-pinned requests and`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L789` `# already-generating requests bypass the pool.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L790` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### Admit a text request II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `928-935`
- Function/anchor: `def _admit_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L928` `return ctx.tokens.active_length`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L929` `return bound.ce_weight`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L930` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L931` `def _post_cache_weights(self, ctx: TextContext) -> list[int]:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L932` `"""Estimates per-replica post-prefix-cache CE length for a new request.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L933` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L934` `Probes the prefix caches read-only (device tier per replica, shared`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L935` `host/disk tiers once) and subtracts the contiguous cached prefix from`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Bind request resources I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `805-812`
- Function/anchor: `def _bind_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L805` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L806` `def _bind_request(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L807` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L808` `ctx: TextContext,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L809` `replica_idx: int,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L810` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L811` `ce_weight: int | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L812` `) -> None:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Bind request resources II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `928-935`
- Function/anchor: `def _bind_request`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L928` `return ctx.tokens.active_length`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L929` `return bound.ce_weight`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L930` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L931` `def _post_cache_weights(self, ctx: TextContext) -> list[int]:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L932` `"""Estimates per-replica post-prefix-cache CE length for a new request.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L933` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L934` `Probes the prefix caches read-only (device tier per replica, shared`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L935` `host/disk tiers once) and subtracts the contiguous cached prefix from`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Add context-encoding work I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1460-1467`
- Function/anchor: `def _add_ce_requests`
- Evidence: `SOURCE`
- Mathematical model: `sum CE_chunk_i <= CE_budget`

- `L1460` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1461` `def _add_ce_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1462` `# Deferred by the DP CE balancer this iteration (also covers paths`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1463` `# that bypass _identify_priority, e.g. in-flight batching).`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1464` `if replica_idx in self._ce_deferred_replicas:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1465` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1466` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1467` `# Off-cadence step. Also covers the in-flight batching backfill,`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### Add context-encoding work II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1529-1536`
- Function/anchor: `def _add_ce_requests`
- Evidence: `SOURCE`
- Mathematical model: `sum CE_chunk_i <= CE_budget`

- `L1529` `self._return_to_request_queue(ctx, replica_idx)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1530` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1531` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1532` `# Check if the request fits in memory`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1533` `if self.kv_cache is not None:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1534` `# Claim the request if needed.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1535` `if not self.kv_cache.contains(ctx):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1536` `self.kv_cache.claim(ctx, replica_idx=replica_idx)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Plan chunked prefill I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1939-1946`
- Function/anchor: `def _plan_ce_step`
- Evidence: `SOURCE`
- Mathematical model: `chunk_i = min(remaining_i, per_req_cap, budget_left)`

- `L1939` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1940` `def _plan_ce_step(self) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1941` `"""Plans this iteration's CE work across DP replicas.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1942` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1943` `Prices CE work in post-prefix-cache tokens, against each replica's`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1944` `admission seats (\`\`_ce_admission_seats\`\`) so a replica is only ever`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1945` `planned work admission will take, and greedily assembles the most`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1946` `balanced CE step it can, deferring the rest:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Plan chunked prefill II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `2048-2055`
- Function/anchor: `def _plan_ce_step`
- Evidence: `SOURCE`
- Mathematical model: `chunk_i = min(remaining_i, per_req_cap, budget_left)`

- `L2048` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L2049` `step_load = list(floor)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2050` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L2051` `def _occupancy(loads: list[int]) -> float:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L2052` `max_load = max(loads)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L2053` `if max_load == 0:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L2054` `return 1.0`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L2055` `return sum(loads) / len(loads) / max_load`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Add token-generation work I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1684-1691`
- Function/anchor: `def _add_tg_requests`
- Evidence: `SOURCE`
- Mathematical model: `TG_tokens ~= number of active decode contexts`

- `L1684` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1685` `def _add_tg_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1686` `replica_requests = self.replicas[replica_idx]`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1687` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1688` `# Add based on the oldest request, respecting KV cache limits and token budgets.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1689` `candidate_ids = deque(replica_requests.tg_reqs.keys())`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1690` `max_batch_size = self.scheduler_config.max_batch_size`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1691` `while len(batch) < max_batch_size and len(candidate_ids) > 0:`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.

### Add token-generation work II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1851-1858`
- Function/anchor: `def _add_tg_requests`
- Evidence: `SOURCE`
- Mathematical model: `TG_tokens ~= number of active decode contexts`

- `L1851` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1852` `interval = self.scheduler_config.prefill_schedule_interval`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1853` `if interval <= 1:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1854` `return True`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1855` `# No decode work anywhere, so no lockstep stall to protect.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1856` `if not any(replica.tg_reqs for replica in self.replicas):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1857` `return True`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1858` `return self._prefill_interval_phase == 0`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Construct a replica batch I - contract

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1859-1866`
- Function/anchor: `def _construct_replica_batch`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1859` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1860` `def _construct_replica_batch(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1861` `self, replica_idx: int, priority_override: RequestType | None = None`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1862` `) -> ReplicaBatch:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1863` `"""Constructs a batch for a single replica.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1864` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1865` `Args:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1866` `replica_idx: The index of the replica to construct a batch for.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Construct a replica batch II - implementation

- File: `max/python/max/serve/scheduler/batch_constructor/text_batch_constructor.py`
- Source range: `1980-1987`
- Function/anchor: `def _construct_replica_batch`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1980` `self._ce_deferred_replicas.clear()`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1981` `self._ce_step_quota = None`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1982` `if not self._dp_ce_balance_enabled:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1983` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L1984` `if not self._ce_pending and not any(`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1985` `replica.ce_reqs for replica in self.replicas`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1986` `):`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1987` `return`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Update TextContext I - contract

- File: `max/python/max/pipelines/context/context.py`
- Source range: `1031-1038`
- Function/anchor: `def update(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1031` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1032` `def update(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1033` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1034` `new_token: int,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1035` `log_probabilities: LogProbabilities | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1036` `) -> None:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1037` `"""Advance both token buffer and FSM state.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1038` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### Update TextContext II - implementation

- File: `max/python/max/pipelines/context/context.py`
- Source range: `1061-1068`
- Function/anchor: `def update(         self,`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1061` `if self._has_pending_future_token():`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1062` `raise ValueError("Cannot have multiple future tokens.")`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1063` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1064` `if self.matcher is not None:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1065` `# For structured output, only advance the token buffer.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1066` `# The FSM cannot accept placeholder tokens and will be`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1067` `# advanced with the real token in sync_and_process_outputs().`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1068` `self.advance_token_buffer(FUTURE_TOKEN)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Paged KV claim and release I - contract

- File: `max/python/max/pipelines/kv_cache/paged_kv_cache/cache_manager.py`
- Source range: `678-685`
- Function/anchor: `def claim(self, ctx`
- Evidence: `SOURCE`
- Mathematical model: `KV_bytes ~= 2 L n_kv d_h tokens bytes_per_elem`

- `L678` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L679` `def claim(self, ctx: TextContext, replica_idx: int = 0) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L680` `"""Pins a request to one replica, which owns it until it is released."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L681` `self._block_manager.claim(ctx, replica_idx)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L682` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L683` `def step(self, ctx: TextContext) -> None:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L684` `"""Commits the request's newly written tokens into the prefix cache."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L685` `self._block_manager.step(ctx)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Paged KV claim and release II - implementation

- File: `max/python/max/pipelines/kv_cache/paged_kv_cache/cache_manager.py`
- Source range: `732-739`
- Function/anchor: `def claim(self, ctx`
- Evidence: `SOURCE`
- Mathematical model: `KV_bytes ~= 2 L n_kv d_h tokens bytes_per_elem`

- `L732` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L733` `def get_device_buffer(self, replica_idx: int) -> KVCacheBufferInterface:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L734` `"""Returns the replica's KV buffer (single leaf or tree).`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L735` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L736` `HACK: this exists only for the transfer engine; callers flatten via`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L737` `:attr:\`KVCacheBufferInterface.all_buffers\`.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L738` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L739` `return self._kv_buffers[replica_idx]`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

## Part 6: Source Atlas, Debugging, and Invariants

### CompiledModel versus Model I - contract

- File: `max/python/max/engine/api.py`
- Source range: `413-420`
- Function/anchor: `class CompiledModel`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L413` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L414` `class CompiledModel:`
  This type creates an ownership and behavior boundary for the state used by later calls.
- `L415` `"""A compiled model artifact, ready for initialization with weights.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L416` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L417` `Returned by :meth:\`InferenceSession.compile\`. Pass it to`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L418` `:meth:\`InferenceSession.init\` or :meth:\`InferenceSession.init_all\` to`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L419` `produce executable :class:\`Model\` instances.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L420` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### CompiledModel versus Model II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `502-509`
- Function/anchor: `class CompiledModel`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L502` `f"{what} needs the device that graph {self.name!r} was compiled "`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L503` `"for, and it was compiled under a virtual device, which stands in "`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L504` `"for hardware that is not attached. Initialize the compiled "`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L505` `"artifact on a real device to execute it."`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L506` `)`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L507` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L508` `def execute(self, *args: object, **kwargs: object) -> NoReturn:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L509` `"""Raises: a compile-only stand-in has nothing to execute on."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Asynchronous compilation I - contract

- File: `max/python/max/engine/api.py`
- Source range: `968-975`
- Function/anchor: `def compile_async(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L968` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L969` `def compile_async(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L970` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L971` `model: str | Path | Module | Graph,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L972` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L973` `custom_extensions: CustomExtensionsType | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L974` `tile_based_fusion: bool = False,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L975` `) -> CompiledModel:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Asynchronous compilation II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `994-1001`
- Function/anchor: `def compile_async(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L994` `custom_extensions: The extensions to load for the model.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L995` `Supports paths to \`\`.mojoc\`\` custom ops.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L996` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L997` `tile_based_fusion: When \`\`True\`\`, compile the graph under the tile-based`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L998` `programming model, in which the graph compiler selects`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L999` `tile-based kernels (operating on \`\`TileTensor\`\` values instead`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1000` `of SIMD). Only applies when \`\`model\`\` is a :class:\`Graph\` or`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1001` `:class:\`max.graph.Module\`; ignored for precompiled \`\`.mef\`\``
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Initialize all graphs I - contract

- File: `max/python/max/engine/api.py`
- Source range: `1159-1166`
- Function/anchor: `def init_all(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1159` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L1160` `def init_all(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L1161` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1162` `compiled: CompiledModel,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1163` `*,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1164` `weights_registry: Mapping[str, DLPackArray] | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L1165` `) -> dict[str, Model]:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1166` `"""Initializes all models in a compiled artifact for execution.`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Initialize all graphs II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `1182-1189`
- Function/anchor: `def init_all(`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L1182` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L1183` `if is_virtual_device_mode():`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L1184` `# Virtual device mode can't actually initialize the model, but`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1185` `# users (eg. cross compilation, benchmarking) want it to not fail.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1186` `# Return one stand-in per top-level graph in the artifact so callers`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1187` `# that key by graph name still work; the stand-in refuses anything`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1188` `# that would need the device rather than pretending to have one.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L1189` `if not compiled._graph_names:`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.

### Execute an initialized model I - contract

- File: `max/python/max/engine/api.py`
- Source range: `118-125`
- Function/anchor: `def _Model_execute`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L118` `@traced`
  A decorator or annotation attaches compile-time registration and dispatch metadata to the declaration that follows.
- `L119` `def _Model_execute(self: Model, *args: InputType) -> list[Buffer]:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L120` `# Original tensor-only execution path`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L121` `input_impls: list[Buffer] = []`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L122` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L123` `for idx, arg in enumerate(args):`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.
- `L124` `_raise_if_not_contiguous(arg)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L125` ``
  A visual separator keeps the source block readable; no runtime action occurs here.

### Execute an initialized model II - implementation

- File: `max/python/max/engine/api.py`
- Source range: `178-185`
- Function/anchor: `def _Model_execute`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L178` `raise TypeError("graph_keys must be an int or sequence of ints.")`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L179` `if isinstance(graph_keys, int):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L180` `return [_normalize_graph_key(graph_keys)]`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L181` `if isinstance(graph_keys, (str, bytes)):`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L182` `raise TypeError("graph_keys must be a sequence of ints.")`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L183` `normalized: list[int] = []`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L184` `for graph_key in graph_keys:`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.
- `L185` `normalized.append(_normalize_graph_key(graph_key))`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Llama model execute wrapper I - contract

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `200-207`
- Function/anchor: `def execute(self, model_inputs`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L200` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L201` `def execute(self, model_inputs: ModelInputs) -> ModelOutputs:`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L202` `assert isinstance(model_inputs, Llama3Inputs)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L203` `assert model_inputs.kv_cache_inputs is not None`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L204` `model_outputs = self.model.execute(*model_inputs.buffers)`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L205` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L206` `assert self.batch_processor is not None`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L207` `return self.batch_processor.process_outputs(model_outputs)`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Llama model execute wrapper II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `245-252`
- Function/anchor: `def execute(self, model_inputs`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L245` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L246` `# TODO(kathywu): Drop once every subclass implements graph_specs.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L247` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L248` `type(self)._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L249` `is not GraphPipelineModelWithKVCache._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L250` `):`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L251` `return super().load_model(session)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L252` `return self.load_graphs(session)[GraphRole.DECODER]`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Initial token inputs I - contract

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `185-192`
- Function/anchor: `def prepare_initial_token_inputs`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `row_i = tokens[offset_i : offset_(i+1)]`

- `L185` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L186` `def prepare_initial_token_inputs(`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L187` `self,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L188` `replica_batches: Sequence[Sequence[TextContext]],`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L189` `kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L190` `return_n_logits: int = 1,`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L191` `) -> Llama3Inputs:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L192` `"""Delegates to the batch processor and narrows to \`\`Llama3Inputs\`\`."""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### Initial token inputs II - implementation

- File: `max/python/max/pipelines/architectures/llama3/model.py`
- Source range: `245-252`
- Function/anchor: `def prepare_initial_token_inputs`
- Evidence: `SOURCE + BOUNDARY`
- Mathematical model: `row_i = tokens[offset_i : offset_(i+1)]`

- `L245` `"""`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L246` `# TODO(kathywu): Drop once every subclass implements graph_specs.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L247` `if (`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
- `L248` `type(self)._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L249` `is not GraphPipelineModelWithKVCache._build_graph_for_compile`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L250` `):`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L251` `return super().load_model(session)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L252` `return self.load_graphs(session)[GraphRole.DECODER]`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Attach Python stack traces I - contract

- File: `GraphCompiler/lib/MGPDialect/Transforms/AttachPythonStackTraces.cpp`
- Source range: `54-61`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L54` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L55` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L56` `mlir::MLIRContext *ctx = &getContext();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L57` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L58` `getOperation()->walk([&ctx](Operation *op) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L59` `Location loc = op->getLoc();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L60` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L61` `// Only the op's immediate OpaqueLoc applies, not ones nested in a`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### Attach Python stack traces II - implementation

- File: `GraphCompiler/lib/MGPDialect/Transforms/AttachPythonStackTraces.cpp`
- Source range: `63-70`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L63` `if (auto assertOp = dyn_cast<AssertOp>(op)) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L64` `if (auto opaqueLoc = mlir::dyn_cast<mlir::OpaqueLoc>(loc)) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L65` `llvm::StringRef stackTrace = extractStackTrace(opaqueLoc);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L66` `if (!stackTrace.empty()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L67` `std::string fullMsg = "\nSource Traceback:\n";`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L68` `llvm::raw_string_ostream ss(fullMsg);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L69` `ss << stackTrace << '\n';`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L70` `StringAttr msg = assertOp.getMessageAttr();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.

### Simple buffer reuse I - contract

- File: `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`
- Source range: `94-101`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `reuse(a,b) only if live(a) intersect live(b) = empty`

- `L94` `public:`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L95` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L96` `ModelOp modelOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L97` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L98` `auto &analysis = getAnalysis<BufferLifetimeAnalysisPass>();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L99` `int reuseCount = 0;`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L100` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L101` `// Erasing allocs invalidates the analysis, but the next pass only queries`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.

### Simple buffer reuse II - implementation

- File: `GraphCompiler/lib/MGPDialect/Transforms/SimpleBufferReusePass.cpp`
- Source range: `111-118`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `reuse(a,b) only if live(a) intersect live(b) = empty`

- `L111` `for (auto allocOp : block->getOps<MGP::BufferAllocOp>()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L112` `ReuseKey key = {allocOp.getRuntimeSize(), allocOp.getDevice(),`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L113` `allocOp.getRawAlign()};`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L114` `allocGroups[key].push_back(allocOp);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L115` `}`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L116` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L117` `if (allocGroups.empty())`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L118` `return WalkResult::advance();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.

### Dynamic memory planning I - contract

- File: `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
- Source range: `84-91`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L84` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L85` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L86` `ModelOp modelOp = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L87` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L88` `auto getDeviceContext = [&modelOp](DeviceRefAttr dev,`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L89` `ArrayRef<BlockArgument> devices) {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L90` `for (auto [deviceRefAttr, deviceContext] :`
  This loop expands repeated work while preserving the surrounding ordering and dependency rules.
- `L91` `llvm::zip_equal(modelOp.getDevices(), devices))`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### Dynamic memory planning II - implementation

- File: `GraphCompiler/lib/MGPDialect/Transforms/DynamicMemoryPlanningPass.cpp`
- Source range: `111-118`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L111` `for (auto allocOp : execState.execute->getOps<MGP::BufferAllocOp>()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L112` `// TODO(GEX-557): Plan CPU allocations too.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L113` `if (allocOp.getDevice().getLabel() == kCPULabel)`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L114` `continue;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L115` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L116` `ErrorOr<SmallVector<Operation *>> allocUses = getAllocUses(allocOp);`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L117` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L118` `if (allocUses.isError() || allocUses->empty())`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### Emit Mojo kernels I - contract

- File: `GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp`
- Source range: `140-147`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L140` `using LiftBufferCacheSlotsBase::LiftBufferCacheSlotsBase;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L141` `void runOnOperation() override { liftBufferCacheSlots(getOperation()); }`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L142` `};`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L143` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L144` `struct EmitMojo : public impl::EmitMojoBase<EmitMojo> {`
  This type creates an ownership and behavior boundary for the state used by later calls.
- `L145` `using EmitMojoBase::EmitMojoBase;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L146` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L147` `void emitMojoText() {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### Emit Mojo kernels II - implementation

- File: `GraphCompiler/lib/MOGGDialect/Transforms/EmitMojo.cpp`
- Source range: `206-213`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L206` `// Compute a base file name for the dump from the names of non-subgraph`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L207` `// ModelOps (user-visible model names) joined by "+", or "graph" when there`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L208` `// are none. Mirrors the \`<names>.mo.mlir\` stem pipeline_to_mo_graph.py`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L209` `// writes, so the .mojo dump lands beside the source graph's stages.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L210` `std::string baseFileName;`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L211` `for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L212` `if (!modelOp.getIsSubgraph()) {`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L213` `if (!baseFileName.empty())`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.

### JIT compile kernels I - contract

- File: `GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp`
- Source range: `55-62`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L55` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L56` `void runOnOperation() override {`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L57` `ModuleOp module = getOperation();`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L58` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L59` `KGEN::CompilerTimeTraceScope traceScope("jit-compile-kernels",`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L60` `module->getName().getStringRef());`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L61` ``
  A visual separator keeps the source block readable; no runtime action occurs here.
- `L62` `initLLVM();`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.

### JIT compile kernels II - implementation

- File: `GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp`
- Source range: `187-194`
- Function/anchor: `void runOnOperation() override`
- Evidence: `SOURCE + COMPILE-ONLY`
- Mathematical model: `state_(n+1) = transform_n(state_n)`

- `L187` `// cross-compiled target could hit a host-warmed artifact, or a`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L188` `// host-compiled fold kernel could poison the cross target's key.`
  The source comment records an invariant, intent, or boundary that constrains the implementation below.
- `L189` `auto hostTargetOr = MOGG::getHostTargetInfoAttr(`
  This binds derived state or materializes an intermediate object that later lines transform, validate, or submit.
- `L190` `ctx, defaultDeviceTargetArch, defaultDeviceTargetTriple,`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L191` `compileForHostExecution);`
  This statement invokes or composes the next operation in the local dataflow; inspect its arguments to see which state crosses the boundary.
- `L192` `if (hostTargetOr.isError())`
  This declaration defines a callable boundary; parameters and return type form the contract used by the next layer.
- `L193` `return kgenModuleOp->emitError() << hostTargetOr.takeError();`
  The return value crosses this function boundary and becomes state consumed by the caller or the next compiler/runtime stage.
- `L194` `if (auto error = MOGG::graphCompilerCacheKey(`
  This branch selects behavior from runtime or compile-time facts such as shape, device, target, or scheduling state.
