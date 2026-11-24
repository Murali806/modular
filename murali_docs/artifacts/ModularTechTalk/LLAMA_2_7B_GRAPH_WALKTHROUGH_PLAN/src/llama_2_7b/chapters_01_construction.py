from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('session','S1: session, devices and runtime context',
 'Follow the constructor without treating a device description, CUDA context, MAX context and MLIR context as the same object.',
 'Model and construction',[
 A('PY','Python caller','thread / object','TextGenerationPipeline','Constructs S1 before decoder tracing and keeps it for auxiliary graphs.'),
 A('DEV','Device wrappers','references','Python driver objects','Pair target descriptions with retained driver contexts.'),
 A('S1','S1 · native session','configuration + owner','Python InferenceSession','Holds devices, compile configuration and engine context.'),
 A('CTX','MAX context','shared context','Init / engine','Holds runtime facilities including the CPUDevice.'),
 A('RT','CPUDevice / AsyncRT','scheduler','process-wide shared runtime','Owns task queues and worker execution facilities; existing options can constrain later session hints.'),
 A('IR','MLIR context','compiler context','graph/compiler subsystem','Uniqued types, attributes, dialects and compiler workers; not the GPU context.')],[
 E('Normalize constructor arguments','Reject simultaneous MEF import/export options, materialize the device iterable, remove duplicate devices and retain their order.',{'PY':B('constructing S1','devices=[GPU(0)]','num_threads=None'),'DEV':B('normalized','GPU(0)')},[R('engine','final_devices: list[Device]'),R('engine','pass at most one of precompiled_mefs')]),
 E('Ensure a host CPU device exists','CPU() is appended if absent. This does not move model weights to the CPU; host shape values and runtime work need a host device context.',{'DEV':B('complete device set','GPU(0)','CPU()')},[R('engine','host_cpu = CPU()')]),
 E('Construct the native session','The binding receives normalized devices, processed custom extensions and num_threads or zero. It retains DeviceSpec and driver context references.',{'S1':B('native configuration',('s1','runtimeConfig + compilation config'))},[R('engine','self._impl = _InferenceSession('),R('session_native','runtimeConfig.setNumThreads(numThreads)')],edges=[('PY','S1'),('DEV','S1')]),
 E('Acquire the MAX engine context','Engine::Context::create uses an explicit MAX context when provided, otherwise Init::getOrCreateContext. It validates recognized CPU/GPU device labels.',{'CTX':B('shared facilities',('ctx','MAX context / telemetry / CPUDevice ref'))},[R('engine_context','Engine::Context::create'),R('engine_context','Init::getOrCreateContext')],edges=[('S1','CTX')]),
 E('Use the process CPU runtime','The Python CPU driver route uses the shared CPUDevice. Main-thread non-donation is configured on that path; an await from another thread must follow its actual branch.',{'RT':B('available scheduler',('rt','CPUDevice: queues + workers'),facts=['Session count ≠ thread-pool count'])},[R('device_native','withMainWillNotDonate'),R('engine_context','cpuDevice(*maxContext->get<CPUDevice>())')],edges=[('CTX','RT')]),
 E('Apply session policies before model construction','Pipeline configuration sets compiler/runtime policies. The later graph creation enters the MLIR context separately.',{'S1':B('configured for models',('s1','session policy + device references')),'IR':B('separate compiler facility','MLIR context for later G1')},[R('pipeline','configure_session(session)'),R('mlir','def in_default_mlir_context(')]),
 E('Return S1 without inventing M1','No decoder graph, decoder weights binding or decoder execution has occurred merely because the session constructor returned.',{'PY':B('S1 retained',('s1-ref','pipeline.session → S1')),'S1':B('ready for compile/init',('s1','retained engine context'))},[R('pipeline','self.session = session'),R('pipeline','self._pipeline_model = pipeline_model(')],edges=[('PY','S1')])],[
 Q('The constructor contract, argument by argument','''
| Input | Python action | Native consequence |
|---|---|---|
| `devices` | preserve first occurrence; add CPU if absent | retain device specs and context handles |
| `num_threads` | store optional value; pass `num_threads or 0` | runtime configuration hint; inspect actual shared runtime creation |
| `custom_extensions` | normalize extension objects if supplied | session compilation extensions; distinct from GraphSpec extensions |
| `precompiled_mefs` | create import MefStore | graph compilation can reuse artifacts by store policy |
| `export_mefs` | create export MefStore | generated compiled artifacts may be saved |

The two MEF options are mutually exclusive. The store captures its position through loaded graphs, which matters because a serving pipeline has decoder and sampler roles. An imported file is compiled-code evidence, not an initialized model with valid live weights.
''',[R('engine','def __init__(',occurrence=3),R('engine','self._mef_store: MefStore | None')],open=True),
 Q('Four contexts and three kinds of parallelism','''
```text
S1 → engine/MAX context → AsyncRT CPUDevice → host task queues/workers
G1 / IR1 → MLIRContext → dialect/type/attribute uniquing + compiler pool
GPU(0) wrapper → Driver DeviceContext → CUDA stream/module/event resources
HTTP/event loop → serving coordination → model worker requests
```

The MLIR worker pool performs compiler work. AsyncRT workers perform host runtime work and ready continuations. CUDA grid/block threads execute device programs. A Python GIL release enables native concurrency; it is not itself any of these pools.

`num_threads=0` is the native sentinel used when Python receives no positive override. To determine the effective AsyncRT worker count, follow CPUDevice creation options and whether a process-wide runtime already exists. Do not multiply a guessed thread count by the number of sessions, or interpret `num_threads` as GPU threads per block.
''',[R('session_native','runtimeConfig.setNumThreads(numThreads)'),R('device_native','getOrCreateCPUDevice('),R('mlir_native','ownedThreadPool')],open=True),
 Q('Device spec versus context, discovery versus execution','''
`DeviceRef("gpu", 0)` is a symbolic placement identity used by graph types. A runtime `GPU(0)` wrapper additionally has a retained driver context. A DeviceSpec supplies target properties for validation and compilation; the context supplies operations such as allocation, stream submission, module loading and events.

Virtual-device support supplies compile-time target identities without establishing physical GPU execution or memory allocation. Enumeration, target IDs, CUDA ordinals and hardware partitions are separate concepts. For NVIDIA, NVML-based discovery and CUDA execution have different APIs; a MIG-visible device need not mean an entire physical card. These details belong to device discovery, not a layer-specific Llama compiler transformation. The Example 04 discovery companions remain linked in the coverage map for their native branch detail.

This course chooses a real single GPU only as its downstream source scenario, without claiming one is available on this documentation host.
''',[R('types','class DeviceRef'),R('virtual','VirtualDevice'),R('device_native','DeviceContextPair'),R('cuda','cuModuleLoadDataEx')]),
 Q('Failures and retention before the graph exists','''
An unsupported device label, incompatible MEF options, failed runtime context creation or invalid configuration can stop startup here. A Python wrapper is not sufficient evidence that CUDA module loading, memory allocation or graph compilation succeeded.

Native session configuration retains device context references. Later initialized models also retain the session and compiled artifact needed for execution. Releasing one Python name does not necessarily destroy shared runtime facilities. Cleanup must respect pending work; CPUDevice shutdown drains its queue before invalidating runtime state.
''',[R('engine_context','unrecognized device'),R('session_native','loadAll('),R('cpu','CPUDevice::~CPUDevice')])],'''
TextGenerationPipeline.__init__
└─ InferenceSession.__init__
   ├─ MEF option validation / MefStore
   ├─ unique devices + CPU()
   ├─ _InferenceSession(final_devices, extensions, num_threads or 0)
   │  ├─ retain DeviceSpec + DeviceContextRef
   │  └─ Engine::Context::create
   │     ├─ supplied MAX context OR Init::getOrCreateContext
   │     └─ shared CPUDevice reference → AsyncRT
   └─ Python diagnostics / runtime configuration
      (decoder G1 is constructed later)
''',[R('engine','final_devices: list[Device]'),R('session_native','InferenceSession::InferenceSession('),R('engine_context','Engine::Context::create')])

C('graph_scaffold','G1: MLIR context, module, graph and block',
 'The actual objects created before one Llama layer is traced, including shape parameters, placement and ordering chains.',
 'Model and construction',[
 A('CTX','MLIRContext','compiler context','MAX Python context utility','Owns dialect registration and interned types/attributes; wrappers reference this context.'),
 A('MOD','ModuleOp','IR container','Graph or GraphModule','Owns the graph operation and eventually other symbols.'),
 A('G1','G1 · GraphOp','symbolic graph','ModuleOp','Has parameters, region, body block and result signature.'),
 A('B0','B0 · body block','IR block','GraphOp region','Owns block arguments and recorded tensor/custom operations.'),
 A('P1','Kernel library','declarations','Graph','Resolves built-ins and any applicable extension overlays.'),
 A('CHAIN','Device chains','symbolic SSA values','Graph','Track operation ordering during construction, including an always-ready chain.')],[
 E('Enter the default MLIR context','The constructor decorator ensures a context is active. Nested uses reuse the appropriate context scope rather than making a fresh runtime for each operation.',{'CTX':B('current compiler context','dialects','types / attributes','diagnostics')},[R('mlir','def ensure_default_mlir_context('),R('graph','@in_default_mlir_context')]),
 E('Collect symbolic shape names','Graph scans TensorType/BufferType input shapes and builds named parameters. T, batch size and page count remain symbolic; 4096 and 128 are constants.',{'G1':B('parameter contract','symbolic input dimensions','si64 parameter declarations')},[R('graph','self._params = dict.fromkeys('),R('graph','si64 = kgen.SIMDType')]),
 E('Create or reuse the owning module','With no supplied GraphModule, builtin.ModuleOp is created. Its body hosts an OpBuilder positioned for the GraphOp insertion.',{'MOD':B('module allocated',('module','ModuleOp → body'))},[R('graph','self._module = ('),R('graph','builder = OpBuilder(')],edges=[('CTX','MOD')]),
 E('Convert input types and create GraphOp','Each input type converts dtype, shape and DeviceRef into native attributes/types. GraphOp initially has no declared result types.',{'G1':B('GraphOp inserted',('g1','GraphOp(name="llama3")'),'input types → MLIR')},[R('graph','op = _mo.GraphOp('),R('types','def to_mlir(self) -> mo.TensorType:')],edges=[('MOD','G1')]),
 E('Use the same native operation through its wrapper','The CAPI bridge creates the Python MLIR view of that operation. The first region’s first block becomes B0; block arguments are symbolic inputs.',{'B0':B('body block available',('b0','%tokens %offsets %return_n %kv…'))},[R('graph','mlir.Operation._CAPICreate(op._CAPIPtr)'),R('graph','self._current_block =')],edges=[('G1','B0')]),
 E('Initialize weights and symbolic ordering','The weight registry starts empty. Graph creates its per-device chain map and an always-ready chain for work safe without advancing ordinary device ordering.',{'CHAIN':B('symbolic ordering initialized','device_chains','always-ready mo chain'),'G1':B('construction state',('g1','GraphOp + empty weight registry'))},[R('graph','self._weights = {}'),R('graph','self._always_ready_chain =')],edges=[('G1','CHAIN')]),
 E('Resolve the implementation library','KernelLibrary plus resolved extensions makes operation contracts available. Built-in Llama GraphSpec has no explicit custom-source package.',{'P1':B('declarations available',('p1','kernel declarations + opaque types'))},[R('graph','self._kernel_library = kernel_library or KernelLibrary()'),R('graph','self._import_kernels(')],edges=[('P1','B0')]),
 E('Trace forward under the graph context','With self activates insertion and graph state; forward consumes symbolic inputs. Output() records results and finalization verifies the graph boundary.',{'B0':B('recording → finalized','embedding → 32 blocks → logits'),'G1':B('complete symbolic graph',('g1','inputs + weights + ops + output tree'))},[R('graph','result = forward(*self.inputs'),R('graph','def __enter__('),R('graph','def __exit__(')],edges=[('B0','G1')])],[
 Q('Object layout, ownership and the CAPI bridge','''
```text
Python Graph G1
 ├─ _module ──owns──> builtin.ModuleOp
 │                    └─ module body → mo.GraphOp
 │                                      └─ region[0] → block B0
 │                                                     ├─ arguments
 │                                                     ├─ recorded ops
 │                                                     └─ outputs
 ├─ _mlir_op ──view──> same native GraphOp via CAPI pointer
 ├─ _weights: name → typed Weight
 └─ chains / kernel library / context stack
```

Module, region, block, operation and SSA value are different entities. A block argument is supplied by an enclosing call boundary; it has no producer operation inside B0. An operation result is produced by an operation inside the block. Both are graph values, not numerical NumPy arrays.

`_CAPICreate(op._CAPIPtr)` bridges wrappers over the same native operation; it is not the compiler’s IR1 clone. The clone happens at native compilation. The MLIR context owns uniqued metadata and dialect facilities; ownership of the module/operations is a separate lifetime.
''',[R('graph','mlir.Operation._CAPICreate(op._CAPIPtr)'),R('mlir_native','StorageUniquer typeUniquer')],open=True),
 Q('Types: dtype, dimensions and device placement','''
For a BF16 illustration, a hidden value is `TensorType(bfloat16, [T,4096], GPU(0))`. Dtype describes elements; the shape parameter `T` uses integer shape arithmetic. Graph’s parameter declarations use a one-lane signed 64-bit KGEN SIMD type. That internal SIMD type does **not** make every activation an integer tensor or imply GPU vector width 1.

Shape attributes can contain concrete dimensions and symbolic parameter expressions. The shape/APE representation lets the compiler express relationships such as flattened products and inferred reshape dimensions without an actual tensor payload. A `DeviceRefAttr` records placement in a type/operation contract; converting it does not allocate or copy memory.

| Entity | Example | Runtime meaning |
|---|---|---|
| dtype | BF16 activation / FP32 logits | element interpretation |
| dimension | `4096`, `T`, `B+1` | size or expression to resolve |
| placement | GPU 0 / CPU 0 | where values are expected or produced |
| BufferType | mutable KV storage | mutation/alias contract, unlike a fresh tensor result |
''',[R('types','def to_mlir(self) -> mo.TensorType:'),R('types','def to_mlir(self) -> mo.BufferType:'),R('types','def to_mlir(self) -> m.DeviceRefAttr:'),R('graph','si64 = kgen.SIMDType')],open=True),
 Q('Context managers: enter, insert, restore','''
The MLIR-context decorator and Graph’s own context manager solve different problems. The first makes native type/operation creation legal in an active compiler context. The second directs graph-building calls into the right graph/block and manages graph-local construction state.

`Llama3.trace` supplies a `forward` closure, so Graph invokes it inside `with self:`. This differs from Example 04’s explicit external with-block. On exiting a nested graph scope, the previous construction context must be restored. Escaping an exception is a failure to finish tracing, not an initialized partial decoder.

The default Python compiler context, native MLIR thread pool and thread-local current graph should not be confused with AsyncRT’s CPUDevice TLS. Compiling a model and executing one have different context requirements.
''',[R('graph','if forward is not None:'),R('graph','def __enter__('),R('graph','def __exit__('),R('mlir','def in_default_mlir_context(')]),
 Q('Chains are symbolic ordering, not GPU completion','''
Graph tracks per-device chains to preserve side-effect ordering. The always-ready graph chain deliberately avoids advancing those ordinary chains for safe independent operations. Its name does not mean a CUDA event has signaled or that all device work is complete.

Later lowering converts dependencies to executable runtime readiness relationships. At execution, an available AsyncValue carrying a Chain is a runtime object; a driver event is a device-completion mechanism. The graph chain, runtime chain and CUDA event belong to three stages of the same dependency story, not one shared object.
''',[R('graph','self.device_chains ='),R('graph','self._always_ready_chain ='),R('async','bool isValueAvailable()')]),
 Q('MLIRContext internals and cost boundaries','''
Native MLIRContext maintains dialect registries, operation information, diagnostic facilities and distinct type, attribute and affine uniquers. Interning lets equal metadata share identity inside that context. Threading settings determine whether an owned LLVM thread pool is created or an external pool is attached; `setThreadPool` has its own preconditions.

These allocations are host compiler metadata. They do not reserve a 7B-weight GPU buffer or execute transformer arithmetic. Shapes may later be folded or materialized, but creating an integer dimension attribute alone is not a device kernel launch.
''',[R('mlir_native','MLIRContext::MLIRContext'),R('mlir_native','StorageUniquer attributeUniquer'),R('mlir_native','void MLIRContext::setThreadPool')])],'''
Llama3.trace → Graph(name, forward, input_types, ...)
└─ @in_default_mlir_context
   ├─ collect SymbolicDim names → si64 parameters
   ├─ ModuleOp → OpBuilder → mo.GraphOp(input_types.to_mlir())
   ├─ CAPI wrapper → region[0].block[0] → B0
   ├─ weight registry + device chains + always-ready chain
   ├─ KernelLibrary + resolved extensions → P1 declarations
   └─ with G1:
      ├─ forward(*G1.inputs) → record model operations
      └─ output(result tree) → graph boundary / finalization
''',[R('graph','self._params = dict.fromkeys('),R('graph','result = forward(*self.inputs'),R('llama','def trace(')])

C('decoder_trace','Trace all 32 decoder layers and bind weights',
 'Follow names, module construction, GraphSpec, the Python layer loop and output selection without hiding a layer behind a generic model box.',
 'Model and construction',[
 A('HF','Checkpoint state','host tensors','weight loader / adapter','Provides named checkpoint tensors, subject to the accepted encoding route.'),
 A('CFG','Model configuration','resolved object','Llama3Model','Initialization and finalization choose concrete architecture and KV policies.'),
 A('MOD','Llama3 module','Python object tree','GraphSpec','Owns embedding, 32 blocks, norms and output layer definitions.'),
 A('G1','G1 body','symbolic IR','Graph','Records each layer’s operations and weight references.'),
 A('OUT','Output tree','schema + SSA values','GraphSpec / callable','Preserves named decoder outputs through compile/init.')],[
 E('Adapt checkpoint names','For the safetensors route, the adapter removes the model. prefix. Names must still match the MAX module hierarchy and shape expectations.',{'HF':B('adapted state dict','model.layers.0… → layers.0…','lm_head.weight retained')},[R('adapter','model.'),R('model','_load_state_dict(')]),
 E('Finalize model configuration','load_graphs finalizes config using the loaded state and devices. Config finalization is distinct from Graph finalization and runtime model setup.',{'CFG':B('finalized','dimensions / encoding / KV policy')},[R('model','def load_graphs('),R('config','def finalize(')]),
 E('Construct the single-device module','Build Llama3(config), load weights, derive the named input tree and create the decoder GraphSpec named llama3.',{'MOD':B('module tree','Embedding','32 × TransformerBlock','final norm + Linear'),'OUT':B('GraphSpec','role=decoder; name=llama3')},[R('model','def _single_device_spec('),R('llama','embedding_layer = Embedding(')],edges=[('HF','MOD'),('CFG','MOD')]),
 E('Flatten inputs, then restore KV structure','The forward closure accepts token IDs, offsets, return_n_logits and flat KV leaves. It unflattens KV into the device’s collection before invoking the transformer.',{'G1':B('input boundary','%tokens %offsets %return_n','KVCacheInputsPerDevice')},[R('llama','def forward('),R('llama','self.kv_params.unflatten_kv_inputs')],edges=[('MOD','G1')]),
 E('Emit block 0','The layer loop supplies a CPU uint32 layer index and the same structured cache collection. QKV/RoPE/attention refer to layer 0 storage.',{'G1':B('layer 0 recorded','embedding [T,4096]','norm → QKV → RoPE/store → MHA → O','residual → norm → gate/up → down → residual')},[R('transformer','for idx, layer in enumerate(self.layers)'),R('attention','qkv = self.qkv_proj(x)')]),
 E('Emit blocks 1 through 31','Python repeats construction with distinct layer modules and indices. Repeated shapes permit downstream code reuse, but each layer has distinct weights and KV addresses.',{'G1':B('32 blocks recorded','X0 → X1 → … → X32','32 sets of weights','KV layer index 0…31')},[R('transformer','for idx, layer in enumerate(self.layers)')]),
 E('Preserve output structure','Last-token selection, final norm and head yield FP32 logits. Graph output_tree records structured outputs so the compiled callable can reconstruct them.',{'OUT':B('decoder outputs','logits / optional output fields','saved output_tree')},[R('transformer','input_row_offsets[1:] - 1'),R('spec','output_trees = {')],edges=[('G1','OUT')]),
 E('Compile and initialize, then release eligible host ownership','GraphSpec merges weight registries and init_all binds them. load_graphs retains compiled callables and releases host state only when its registry/lifetime rules permit.',{'MOD':B('callable replaces construction work','compiled_modules[decoder]','engine model M1'),'HF':B('conditional release','host state may be released after initialization')},[R('spec','weights_registry=weights'),R('model','def load_graphs(')])],[
 Q('Names, payloads and the weight registry','''
The HF model identifier, architecture string `LlamaForCausalLM`, MAX Python package `llama3`, graph name `llama3` and GraphRole `decoder` name different things. Shared implementation naming does not change the checkpoint into a Llama 3 model.

The state adapter translates checkpoint names, while module loading checks/assigns named weights. Graph construction records typed weight references; it does not inline billions of elements as one giant textual constant. GraphSpec supplies the actual registry used at init. Native initialization retains managed tensor/DLPack ownership as necessary.

If grouped roles share a weight name, `compile_and_init_graphs` requires the same value object for that name. This prevents an ambiguous registry. The primary single-decoder spec is ungrouped; grouping behavior must not be silently substituted for it.
''',[R('adapter','model.'),R('spec','weights.setdefault(name, value)'),R('session_native','DLPack')],open=True),
 Q('A detailed unquantized block ledger','''
| Order | Operation | Shape / storage implication |
|---|---|---|
| 1 | input RMSNorm | consumes `[T,4096]` and scale `[4096]` |
| 2 | QKV projection | output widths 4096+4096+4096; stacking policy can combine projections |
| 3 | RoPE, split, KV store | Q returned; rotated K and V committed to paged cache |
| 4 | paged causal attention | Q `[T,32,128]`; reads this layer’s retained K/V |
| 5 | flatten heads + O projection | `[T,4096]` |
| 6 | first residual add | consumes original layer input |
| 7 | post-attention RMSNorm | `[T,4096]` |
| 8 | gate/up projection | compatible stacked result `[T,22016]` |
| 9 | split, SiLU, multiply | `[T,11008]` |
| 10 | down projection | `[T,4096]` |
| 11 | second residual add | becomes the next layer’s residual stream |

This is a semantic operation ledger, not a measured launch count. Compiler fusion may combine eligible work; library matmuls/attention may contain several launches or specialized routines. The source’s explicit paged-attention custom op does not require the compiler to rediscover dense attention from a generic softmax pattern.
''',[R('transformer','class TransformerBlock'),R('attention','def __call__('),R('linear','class MLP')],open=True),
 Q('Why 32 layers do not imply 32 unrelated compiled programs','''
Tracing emits the repeated layer computation with distinct weights and layer-index values. Static dtype, head dimension and layout can match between layers, while runtime addresses differ. Later stub canonicalization and identical-function merging can share generated code where legal; they do not merge all weight buffers or make KV layers alias.

The graph source exposes a Python loop. That alone does not establish whether the final backend emits an unrolled machine-code sequence, shared host stubs, runtime loops inside kernels, or one device graph. Those decisions need the generated representation and selected target/options. This course follows the pass machinery without inventing a model-specific binary dump.
''',[R('transformer','for idx, layer in enumerate(self.layers)'),R('compiler','createCanonicalizeFuncsForMerge'),R('compiler','createMergeIdenticalFuncs')]),
 Q('GraphSpec and callable shape trees','''
The input tree names batch and KV groups. Its leaves define positional engine inputs; the wrapper remembers the tree definition. `output_tree` preserves the return structure. After `session.init_all`, the helper handles a single returned model without assuming that an imported/cached graph kept the requested graph name.

This matters for serving: preparation code binds the correct kind of input, while the low-level engine still receives a flat positional list. A correct-looking tensor in the wrong position can violate a different rank/dtype/device contract. The input boundary lesson enumerates those leaves explicitly.
''',[R('spec','if len(by_name) == 1:'),R('spec','_check_inputs(role'),R('llama','def input_tree(')])],'''
Llama3Model.load_graphs
├─ _load_state_dict → adapter / dtype-specific handling
├─ finalized Llama3Config + KV parameters
├─ graph_specs → _single_device_spec
│  ├─ Llama3(config) → 32 TransformerBlocks
│  ├─ load weights → weights_registry
│  └─ GraphSpec(name="llama3", input tree, decoder role)
└─ compile_and_init_graphs
   ├─ spec.trace → Graph(forward)
   │  └─ unflatten KV → Transformer.__call__ → emit 32 layers → outputs
   ├─ compile_reusing_mefs → C1
   ├─ init_all(C1, weights) → M1
   └─ CompiledCallable preserves input/output trees
''',[R('model','def load_graphs('),R('model','def _single_device_spec('),R('spec','def compile_and_init_graphs(')])
