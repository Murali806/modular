"""The remainder of graph construction and recording, with individual mechanisms."""
from compiler_attention_lessons import NAME, Story, T_GPU, lesson, objects


def recording_lessons():
    cards = []

    def add(*a, **kw):
        cards.append(lesson(*a, **kw))

    nodes = objects(("shapes", "input shapes", "Python dimensions", "TQ/TK/TV"),
                    ("names", "discovered names", "deduplicated set", "constructor"),
                    ("params", "G1._params", "dimension-name dictionary", "Graph"),
                    ("decls", "input parameter declarations", "MLIR attributes", "G1"),
                    ("tiles", "BN / BD", "custom-call parameters", "mo.custom"))
    def dimensions(dynamic=False):
        s = Story(nodes, label="Comparison · symbolic seq" if dynamic else "Actual · static dimensions", view="parameters")
        s.step("Walk all three declared shapes", "graph_init", 840, "TQ/TK/TV.shape", "visit each dimension", '["seq",32] × 3' if dynamic else '[32,32] × 3',
               "Only SymbolicDim instances contribute runtime parameter names.", patch={"shapes": '["seq",32] × 3' if dynamic else '[32,32] × 3'})
        s.step("Select only symbolic names", "graph_init", 845, "dimensions", "keep dim.name only when dim is SymbolicDim", '{"seq"}' if dynamic else '{}',
               "Three uses of the same name need one binding, not three unrelated sizes.", patch={"names": '{"seq"}' if dynamic else '{}'})
        s.step("Initialize dimension-name bookkeeping", "graph_init", 840, "selected names", "dict.fromkeys(names)", "_params={'seq':None}" if dynamic else '_params={}',
               "dict.fromkeys deduplicates names and initializes values to None. Native dimension expressions come from shape conversion; this statement does not create them.", patch={"params": "{'seq':None}" if dynamic else '{}'})
        s.step("Declare those parameters on the GraphOp", "graph_init", 870, "_params", "ParamDeclArrayAttr([ParamDeclAttr(...)])", '[seq:i64]' if dynamic else '[]',
               "The compiler sees which runtime dimension names may occur in the graph.", patch={"decls": '[seq:i64]' if dynamic else '[]'})
        s.step("Keep kernel tile parameters on their own call", "entry", 60, "BN=16; BD=8", "ops.custom(parameters={...})", '{BN:16 index, BD:8 index}',
               "Tile parameters specialize code; they do not add graph tensor inputs or symbolic sequence names.", patch={"tiles": '{BN:16, BD:8} · separate mo.custom dictionary'})
        return s
    add("graph-symbolic-params", "Shape names · Graph parameters", "parameters",
        "Why are _params empty, and how would a symbolic sequence length differ from BN/BD?", "Three input shapes", "G1.input parameter declarations; separate tile parameters",
        """**Static size 32 is already known. A symbolic name needs a binding.** The constructor visits all input shapes and deduplicates only `SymbolicDim.name` entries. Here `_params={}`, so its GraphOp declaration array is empty.

The comparison uses `[\"seq\",32]` for Q/K/V and discovers one shared `seq`. That name constrains compatible input dimensions. **BN=16 and BD=8 remain compile-time custom-call parameters**; neither is an additional tensor argument or an entry in `_params`.""",
        """caller: Graph.__init__ discovers dimensions before making G1 [LINK:graph_init:840]
├─ walk input_types → t.shape → only SymbolicDim entries [LINK:graph_init:843]
├─ dict.fromkeys(dim.name...) → deduplicate names; initial values=None [LINK:graph_init:840] [CALLER:graph_init|840]
├─ create GraphOp, then assign input_param_decls [LINK:graph_init:870]
│  └─ actual _params={} → ParamDeclArrayAttr([]) [LINK:graph_init:871] [CALLER:graph_init|870]
└─ later, custom(parameters={BD:8,BN:16}) attaches a DIFFERENT dictionary [LINK:entry:60] [LINK:custom_api:113] [CALLER:entry|57]""",
        dimensions(), alternatives={"symbolic": dimensions(True)}, source="graph_init", line=840)

    nodes = objects(("max", "op · MAX wrapper", "Python native wrapper", "Graph constructor"),
                    ("capsule", "named PyCapsule", "C API adapter", "temporary Python handle"),
                    ("c", "MlirOperation", "C struct {ptr}", "MLIR adapter"),
                    ("context", "MCX1", "compiler context reference", "PyOperation"),
                    ("py", "self._mlir_op", "MLIR Python wrapper", "G1"),
                    ("region", "R0", "existing region", "native G1"),
                    ("body", "B0", "existing entry block", "R0"))
    s = Story(nodes, view="capi")
    s.step("Start with the already-inserted native graph", "graph_init", 860, "Module body and GraphOp builder", "create native GraphOp once", "MAX op wrapper → OP1",
           "OP1 already owns R0/B0 with the three input arguments.", patch={"max": "op → OP1 · example pointer P"}, pointer="P")
    s.step("Export its C API pointer adapter", "ct_capi_export", 607, "op._CAPIPtr", "wrap(existing Operation*) → MlirOperation", "{ptr=P}",
           "wrap/unwrap translates the pointer representation; it does not serialize or clone IR.", patch={"c": "MlirOperation{ptr=P}"})
    s.step("Put the pointer in a named capsule", "capi_capsule", 331, "MlirOperation{ptr=P}", "PyCapsule_New(pointer, operation capsule name, ...)", "capsule → P",
           "The name distinguishes an operation pointer from a type/context pointer. Acquiring a capsule alone does not extend the underlying IR lifetime.", patch={"capsule": "operation-named capsule → P"})
    s.step("Import through MLIR Python's factory", "capi_factory", 1207, "mlir.Operation._CAPICreate(capsule)", "mlirPythonCapsuleToOperation(capsule)", "same MlirOperation{ptr=P}",
           "The factory checks/extracts the operation-specific capsule payload.", patch={"c": "imported MlirOperation{ptr=P} · unchanged identity"})
    s.step("Recover the native operation's context", "capi_factory", 1211, "OP1 pointer P", "mlirOperationGetContext(op) → PyMlirContext.forContext", "MCX1 reference",
           "The wrapper must stay associated with the context that owns native types and attributes.", patch={"context": "MCX1 · recovered from OP1"})
    s.step("Create or recover a Python operation wrapper", "capi_factory", 1212, "MCX1 + P", "PyOperation::forOperation(context, op)", "self._mlir_op → OP1",
           "A different wrapper may refer to the same operation. G1 is cloned only in the later compile route.", patch={"py": "PyOperation → OP1 · pointer P"})
    s.step("Follow its existing region", "graph_init", 875, "self._mlir_op", "regions[0]", "R0",
           "Indexing the operation exposes existing nested IR, not a new region.", patch={"region": "R0 · OP1.regions[0]"})
    s.step("Retain the same body block twice", "graph_init", 876, "R0.blocks[0]", "_current_block = block; _graph_body = block", "both handles → B0(%q,%k,%v)",
           "Later insertion uses _current_block; inputs and graph finalization use _graph_body.", patch={"body": "B0 · _current_block and _graph_body"}, aliases=["_current_block → B0", "_graph_body → B0"])
    add("graph-retain-body", "Graph handles · CAPI and body", "capi", "What passes across CAPI, and why is this not a graph copy?",
        "MAX wrapper for OP1 in MODULE1", "MLIR wrapper for the same OP1; retained R0/B0 handles",
        """**CAPI is the MLIR C API pointer bridge between two Python binding implementations.** `_CAPIPtr` exports an operation-specific PyCapsule; `_CAPICreate` imports it into MLIR's Python wrapper. The illustrative pointer P stays identical through every adapter.

The capsule is a temporary adapter, **not an owner that independently keeps the whole graph alive**. Native IR/context ownership and Python operation references still matter. `regions[0].blocks[0]` follows OP1's existing region and block; `_current_block` and `_graph_body` initially alias B0. No package bytecode, native machine code or numerical tensor is produced here.""",
        """caller: self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr) [LINK:graph_init:874]
├─ MAX _CAPIPtr property → wrap(existing Operation*) [LINK:ct_capi_export:607]
│  └─ operation-named PyCapsule stores existing pointer P [LINK:capi_capsule:331] [CALLER:ct_capi_export|608]
├─ MLIR Python createFromCapsule → capsuleToOperation [LINK:capi_factory:1207]
│  ├─ extract MlirOperation {ptr=P} [LINK:capi_capsule:339] [CALLER:capi_factory|1208]
│  ├─ mlirOperationGetContext → existing context [LINK:capi_context_get:695] [CALLER:capi_factory|1211]
│  └─ PyOperation::forOperation(context,op) → wrapper for OP1 [LINK:capi_factory:1212] [CALLER:graph_init|874]
└─ _current_block = regions[0].blocks[0]; _graph_body = _current_block [LINK:graph_init:875]
   └─ both Python references retain existing B0 [LINK:graph_init:876] [CALLER:graph_init|875]""",
        s, source="graph_init", line=874)

    nodes = objects(("tensor", "Tensor DeviceRef", "symbolic placement", "TQ/TK/TV"),
                    ("module", "MODULE1", "IR container", "Graph"),
                    ("cpu", "cpu:0 metadata", "API/arch/model attributes", "module mapping"),
                    ("gpu", "gpu:0 metadata", "API/arch/model attributes", "module mapping"),
                    ("dict", "mo.device_info_mapping", "native dictionary", "MODULE1"))
    def device_info(present=False):
        s = Story(nodes, label="Comparison · mapping already present" if present else "Actual · populate mapping", view="device-map")
        s.step("Get the enclosing module", "ct_graph_device", 1368, "G1._mlir_op.block.owner", "follow owning block → module operation", "MODULE1",
               "Device target descriptions are attached to the module shared by its graphs.", patch={"module": "MODULE1", "tensor": "gpu:0 · placement only"})
        s.step("Check for a prior mapping", "ct_graph_device", 1369, "MODULE1.attributes", "if mo.device_info_mapping exists: return", "reuse existing map" if present else "map absent → collect devices",
               "Constructing a second graph in the same module need not query/attach the same mapping again.", patch={"dict": "existing map retained" if present else None})
        if present:
            s.step("Retain target descriptions", "ct_graph_device", 1370, "existing mapping", "return without replacement", "no metadata re-query",
                   "This is metadata reuse, not a compile-cache or a resident GPU-code check.", patch={"module": "MODULE1 · existing mapping unchanged"})
        else:
            s.step("Always describe the host", "ct_graph_device", 1371, "CPU()", "start devices=[CPU()]", "CPU descriptor candidate",
                   "Host routines are needed even when tensors are placed on a GPU.", patch={"cpu": "cpu:0 · driver metadata query"})
            s.step("Add a detected default accelerator", "ct_graph_device", 1373, "accelerator_count()>0", "devices.append(Accelerator())", "GPU descriptor candidate",
                   "The helper describes the default accelerator; it does not enumerate a new graph input per device.", patch={"gpu": "gpu:0 · detected accelerator"})
            s.step("Read architecture and model with fallbacks", "ct_graph_device", 1377, "driver device properties", "architecture_name / model_name; exceptions → unknown", "API / arch / model strings",
                   "Example names are illustrative, not this CPU host's measured GPU inventory.", patch={"cpu": "label=cpu:0 · api=host · arch/model queried", "gpu": "label=gpu:0 · api=CUDA or HIP · arch/model queried"})
            s.step("Build native entries and attach the dictionary", "ct_graph_device", 1384, "label, api, arch, model, tile_based_fusion=False", "_DeviceInfoAttr → Attribute capsule → DictAttr", "MODULE1.mo.device_info_mapping",
                   "The compiler combines placement with target descriptions later; this step emits no target code.", patch={"dict": "cpu:0 → host descriptor; gpu:0 → accelerator descriptor"})
        return s
    add("graph-device-info", "Device metadata · Targets", "device-map", "How does a device label become useful compiler metadata?",
        "Enclosing module, CPU and optional accelerator properties", "Module attribute map; no runtime context created",
        """**Placement, target descriptions, and runtime handles are separate.** `gpu:0` on a TensorType identifies placement. The module's `mo.device_info_mapping` adds API/architecture/model information for compiler decisions. The later session retains actual driver device/context references.

The helper always adds CPU metadata and optionally one detected accelerator. Architecture/model query failures become `\"unknown\"`; an already-present mapping is retained. Selecting a GPU here does not execute a kernel or move Q/K/V data.""",
        """caller: self._populate_device_info_mapping() [LINK:graph_init:877]
└─ _populate_device_info_mapping [LINK:ct_graph_device:1366]
   ├─ module ← self._mlir_op.block.owner [LINK:ct_graph_device:1368]
   ├─ mapping already exists → return [LINK:ct_graph_device:1369] [CALLER:graph_init|877]
   ├─ devices=[CPU()]; append Accelerator() when detected [LINK:ct_graph_device:1371]
   ├─ query architecture/model; each failure → unknown [LINK:ct_graph_device:1377] [CALLER:graph_init|877]
   ├─ _DeviceInfoAttr(label,api,arch,model,tile_based_fusion=False) [LINK:ct_graph_device:1384]
   └─ Attribute CAPI bridge → DictAttr(entries) on module [LINK:ct_graph_device:1392] [CALLER:graph_init|877]""",
        device_info(), alternatives={"existing": device_info(True)}, source="graph_init", line=877)

    nodes = objects(("weights", "G1._weights", "declaration dictionary", "Graph"),
                    ("flag", "_has_chain_input", "boolean", "Graph"),
                    ("map", "device_chains", "lazy ordered dictionary", "Graph"),
                    ("cpu", "cpu:0 chain", "dependency reference", "chain map"),
                    ("gpu", "gpu:0 chain", "dependency reference", "chain map"))
    s=Story(nodes,view="chains")
    s.step("Start an empty weight registry", "graph_init",879,"new Graph", "_weights={}", "no external weights",
           "Q/K/V are invocation inputs, not named model weights.",patch={"weights":"{} · no Q/K/V entries"})
    s.step("Start without an explicit chain input", "graph_init",880,"declared tensor inputs", "_has_chain_input=False", "ordinary graph-input mode",
           "A later last-argument check may adopt an incoming chain in subgraph cases.",patch={"flag":"False"})
    s.step("Create a lazy map that retains G1", "ct_device_chain_map",126,"_DeviceChainMap(self)", "OrderedDict init; self._graph=G1", "device_chains={} → G1",
           "Lazy entries give ordered operations a dependency reference for each device.",patch={"map":"{} · _graph→G1"})
    s.step("Seed the CPU entry after CH0 is built", "graph_init",894,"always-ready CH0", "device_chains[DeviceRef.CPU()]=CH0", "cpu:0 → CH0",
           "New device chains can be seeded from this host entry.",patch={"cpu":"CH0", "map":"{cpu:0→CH0}"})
    s.step("A later first device lookup can share the seed", "ct_device_chain_map",148,"device_chains[gpu:0] with no chain input", "missing device → existing CPU chain", "gpu:0 → CH0",
           "This comparison follows the lazy lookup; it does not create a GPU stream or synchronize matrices.",patch={"gpu":"CH0 · shared reference", "map":"{cpu:0→CH0,gpu:0→CH0} · illustrative later lookup"})
    add("graph-tracking","Graph bookkeeping · Weights and chains","chains","What is retained here before any attention arithmetic?",
        "G1", "Empty weight declarations, chain-input flag, lazy ordering map",
        """**Weights describe persistent model parameters; chains describe dependencies.** `_weights={}` means this graph declares no external parameters to bind at init. It does not mean that Q/K/V will be zero.

The lazy map retains G1 and stores chain references by DeviceRef. With no explicit chain input, a first lookup for another device can use the existing CPU chain as its seed. Ordered operations may later replace a map entry with a newer token. **The dictionary is not GPU memory and a chain is not a CUDA/HIP stream handle.**""",
        """caller: constructor starts graph bookkeeping [LINK:graph_init:879]
├─ _weights={} [LINK:graph_init:879] [CALLER:graph_init|879]
├─ _has_chain_input=False [LINK:graph_init:880] [CALLER:graph_init|880]
└─ device_chains=_DeviceChainMap(self) [LINK:graph_init:881]
   ├─ OrderedDict initialization; retain self._graph [LINK:ct_device_chain_map:126] [CALLER:graph_init|881]
   └─ later __getitem__(new_device) [LINK:ct_device_chain_map:134]
      ├─ explicit chain-input mode at graph body → add chain block argument [LINK:ct_device_chain_map:137] [CALLER:ct_device_chain_map|136]
      └─ otherwise → use existing CPU chain as seed [LINK:ct_device_chain_map:148] [CALLER:ct_device_chain_map|136]""",
        s,source="graph_init",line=879)

    nodes=objects(("inputs","producer inputs","chain operands","ChainCreateOp"),
                  ("body","B0.operations","IR sequence","G1"),
                  ("token","CH0","chain SSA result","ChainCreateOp"),
                  ("ready","readiness contract","dependency state","CH0"),
                  ("map","device_chains[cpu:0]","ordering reference","G1"))
    s=Story(nodes,view="chains")
    s.step("Select the graph body insertion site","graph_init",886,"B0(%q,%k,%v)","_add_op_generated(ChainCreateOp,inputs=[])","empty producer list",
           "A dependency seed should have nothing to wait on.",patch={"inputs":"[]", "body":"B0 · three tensor arguments, no ops"})
    s.step("Insert the chain operation","ct_generated_op",1258,"inputs=[]; B0.end","native ChainCreateOp builder","CH0=mo.chain.create()",
           "The token is an SSA result inside G1, not a new user argument.",patch={"body":"B0: CH0=mo.chain.create()", "token":"CH0 : !mo.chain"})
    s.step("Wrap its single result","graph_init",886,"native CH0 result","_ChainValue(results[0])","_always_ready_chain→CH0",
           "The Graph API retains a typed wrapper to reuse the dependency seed.",patch={"token":"_always_ready_chain → CH0"})
    s.step("Interpret a zero-input dependency","ct_chain_definition",4795,"CH0 has no prior producer","chain/create contract","always ready",
           "Ready means no pending dependency here; it does not assert all future GPU work has completed.",patch={"ready":"CH0 · no prerequisites"})
    s.step("Initialize CPU ordering from that result","graph_init",894,"DeviceRef.CPU(), CH0","_update_chain(CH0,CPU)","cpu:0 → CH0",
           "The per-device map gives later ordered operations a starting token.",patch={"map":"cpu:0 → CH0"})
    add("graph-ready-chain","Chain · Ordering token","chains","Why does a tensor graph contain a chain operation?",
        "G1 body; empty producer list", "Always-ready internal CH0 and CPU ordering reference",
        """**A chain is an ordering token.** Its meaning is that dependency-producing work has completed. `ChainCreateOp(inputs=[])` has no earlier work to wait for, so it provides the always-ready seed.

The graph still has exactly three user arguments. CH0 is produced inside B0 and stored in `_always_ready_chain` and the CPU chain map. Runtime execution also uses chains to order resource work; that later chain is not a numerical fourth attention tensor.""",
        """caller: _always_ready_chain = _ChainValue(_add_op_generated(...)[0]) [LINK:graph_init:886]
├─ _mo.ChainCreateOp(inputs=[]) inserted in B0 [LINK:graph_init:887] [CALLER:graph_init|886]
├─ wrap its one result as _ChainValue [LINK:graph_init:886] [CALLER:graph_init|886]
└─ _update_chain(_always_ready_chain, DeviceRef.CPU()) [LINK:graph_init:894]
   └─ device_chains[DeviceRef.CPU()]=chain [LINK:ct_update_chain:1064] [CALLER:graph_init|894]""",
        s,source="graph_init",line=886)

    nodes=objects(("args","B0 arguments","SSA input list","G1"),
                  ("last","last input type","native type","B0"),
                  ("flag","_has_chain_input","boolean","G1"),
                  ("chain","CPU chain","dependency reference","device_chains"),
                  ("public","Graph.inputs","public value wrappers","Python caller"))
    def trailing(adopt=False):
        s=Story(nodes,label="Comparison · incoming chain" if adopt else "Actual · final input is V",view="ports")
        s.step("Inspect the existing entry arguments","graph_init",896,"B0.arguments","check len(args)>0","three tensor arguments" if not adopt else "three tensors plus CH_parent",
               "The constructor checks a trailing chain; it does not create a separate block per input.",patch={"args":"[%q,%k,%v,CH_parent]" if adopt else "[%q,%k,%v]"})
        s.step("Read the last argument's type","graph_init",897,"arguments[-1]","last_arg.type","!mo.chain" if adopt else "TV=float32[32,32]@gpu:0",
               "Only the native ChainType triggers adoption.",patch={"last":"!mo.chain" if adopt else T_GPU})
        s.step("Choose the chain-input branch","graph_init",900,"last type","_is_chain_value(mlir_maybe_chain_value)","True" if adopt else "False",
               "Ordinary attention retains CH0; a subgraph can depend on parent work.",patch={"flag":"True" if adopt else "False"})
        s.step("Retain the chosen dependency","graph_init",902 if adopt else 900,"CH_parent" if adopt else "CH0","_update_chain(_ChainValue.from_mlir(...))" if adopt else "skip adoption; keep the already-created CPU seed","CPU chain→CH_parent" if adopt else "CPU chain→CH0",
               "An incoming chain changes internal ordering, not Q/K/V tensor values.",patch={"chain":"CH_parent" if adopt else "CH0"})
        s.step("Expose only user inputs","ct_graph_inputs",943,"B0.args; chain_count","exclude trailing internal chains when _has_chain_input","q,k,v wrappers",
               "Public Graph.inputs keeps user tensor ports distinct from dependency ports.",patch={"public":"(q→%q,k→%k,v→%v)"})
        return s
    add("graph-trailing-chain","Last input · Tensor or chain?","ports","Why inspect the last argument, and why is V kept as a user input?",
        "B0.arguments", "False flag and CH0 retained here; optional incoming dependency in subgraphs",
        """**Attention's final argument is V, a TensorType.** Therefore `_has_chain_input` stays False and `Graph.inputs` returns all three tensor arguments. The constructor does not consume V as a dependency.

The comparison illustrates a parent-supplied trailing ChainType argument. That branch adopts the chain, marks internal chain-input mode, and excludes internal chains from the public inputs. Both cases keep **one entry block with several arguments**.""",
        """caller: inspect input_args[-1] [LINK:graph_init:896]
├─ no arguments → skip inspection [LINK:graph_init:896] [CALLER:graph_init|896]
├─ last argument type is TensorType (V here) → leave CH0 [LINK:graph_init:900] [CALLER:graph_init|896]
└─ comparison: ChainType → adopt chain and set _has_chain_input=True [LINK:graph_init:900]
   ├─ _update_chain(chain) [LINK:graph_init:902] [CALLER:graph_init|900]
   └─ Graph.inputs excludes trailing chain_count arguments [LINK:ct_graph_inputs:943] [CALLER:ct_graph_inputs|943]""",
        trailing(),alternatives={"chain":trailing(True)},source="graph_init",line=896)

    nodes=objects(("path","extension path","filesystem directory","Python Graph caller"),
                  ("exe","mojo driver","OS executable","separate compiler process"),
                  ("ir","source package IR","parametric MLIR","Mojo compiler process"),
                  ("staging","private staging file","compressed bytecode file","temporary package builder"),
                  ("published","P1","cached .mojoc package","package cache"),
                  ("library","G1 kernel library","analysis/package references","Graph"))
    def package(direct=False):
        s=Story(nodes,label="Existing .mojoc supplied" if direct else "Source directory → precompile",view="package")
        s.step("Resolve extension paths and overlay defaults","graph_init",910,"custom_extensions=[kernels/]","_resolved_custom_extensions → _import_kernels","paths for package loading",
               "The explicit directory is joined with applicable defaults; Python entry code is not part of the Mojo package.",patch={"path":"P1.mojoc" if direct else "kernels/ · __init__.mojo + imported children"})
        s.step("Choose direct package or source-directory route","package_load",285,"each path","suffix==.mojoc? add_path : build source package","direct package" if direct else "source package builder",
               "P1 is parametric package IR; it is not a CPU/GPU binary.",patch={"library":"KernelLibrary · loading paths"})
        if not direct:
            s.step("Compute the deterministic package destination", "package_builder",147,"resolved source directory + user ID",
                   "md5(source absolute path); choose temp .modular_<uid>/mojo_pkg destination", "P1 output path",
                   "This helper does not check for an existing package-file cache hit: the source-directory route invokes precompile and replaces this path.",patch={"published":"deterministic P1 destination · publication pending"})
            s.step("Resolve an executable compiler driver","ct_mojo_run",70,"environment override or PATH","MODULAR_MOJO_MAX_DRIVER_PATH; otherwise shutil.which('mojo')","mojo executable path chosen by environment",
                   "bin/mojo is an executable, not a Python class. An exact installed absolute path is environment-dependent and was not executed for this reader.",patch={"exe":"configured SDK/Bazel bin/mojo or PATH result"})
            s.step("Launch precompile as a separate process","package_builder",178,"kernels/; private staging basename","mojo precompile kernels/ -o <staging>/mojo_pkg_<hash>.mojoc","compiler receives source package",
                   "This is package preparation before graph compilation, distinct from the later in-process kernel JIT.",patch={"exe":"OS child process: mojo precompile", "staging":"private file · not yet published"})
            s.step("Parse the package and referenced modules","ct_parser",391,"kernels/__init__.mojo and imported child modules","Mojo parser creates parametric declarations/bodies","lit.package IR",
                   "fused_attention.mojo is included through the package root; fused_attention.py remains a Python program.",patch={"ir":"lit.package · FusedAttention / execute / helpers · unresolved parameters"})
            s.step("Resolve declarations and imported names","ct_resolver",1756,"parsed module references","resolve types, functions and imports","linked package declarations",
                   "execute can refer to TileTensor, TensorCore, math operations and runtime APIs through dependencies.",patch={"ir":"resolved declarations + parametric bodies"})
            s.step("Run package semantic checks","ct_check_lit",1119,"resolved parametric IR","check LIT before packaging","checked package representation",
                   "This does not select BN/BD/dtype/target machine instructions for an individual launch.",patch={"ir":"checked parametric package IR"})
            s.step("Write versioned compressed bytecode","ct_package_write",137,"checked lit.package","Mojo precompiled writer → MLIR bytecode + header/compression","staging .mojoc bytes",
                   "A schematic textual lit.package is a reading view. Bytecode is binary, not printed LLVM IR.",patch={"staging":"version/header + compressed MLIR bytecode"})
            s.step("Atomically publish the completed file","package_builder",179,"private staging file","os.replace(staging_path,package_path)","P1 available at cache destination",
                   "Readers must not see a partially-written package. Publication occurs after the child succeeds.",patch={"published":"P1 · one complete .mojoc", "staging":"renamed/published"})
        s.step("Load package into kernel analysis","package_load",289 if not direct else 286,"P1 path","KernelLibrary.add_path(P1)","loaded package declarations",
               "The custom verifier will resolve registration names from these package declarations.",patch={"library":"P1 declarations available for analysis", "published":"P1 .mojoc"})
        s.step("Record library paths on the graph","ct_graph_import",1706,"KernelLibrary.paths","self._mlir_op.library=paths","G1 refers to P1",
               "Compilation later consumes the package; numerical Q/K/V still do not exist.",patch={"library":"G1.library → [P1,…defaults]"})
        return s
    add("graph-package","P1 · Executable, parsing and bytecode","package","Which process builds P1, what is inside it, and how does an existing package avoid parsing?",
        "Mojo source directory or existing .mojoc; configured compiler path", "Checked cached P1; package declarations and paths on G1",
        """**`mojo` is a compiler-driver executable.** `subprocess_run_mojo` selects `MODULAR_MOJO_MAX_DRIVER_PATH` or a PATH lookup. SDK defaults can point to package `bin/mojo`; Bazel config supplies its runtime driver. The exact installed absolute path is environment-specific.

Cold preparation parses the source package, resolves dependencies, checks parametric IR, writes a private precompiled file, then publishes it with `os.replace`. **P1 stores package IR as versioned compressed MLIR bytecode**, not attention machine code. Shared dependencies supply imported declarations/bodies; Python caller code is separate.

The existing-.mojoc selector skips the subprocess. A source directory is precompiled even though its deterministic destination may already exist; compiler internals may cache work separately. Concrete dtype, rank, BN/BD, layouts and device target are specialized later during graph compilation.""",
        """caller: self._import_kernels(_resolved_custom_extensions(custom_extensions)) [LINK:graph_init:910]
├─ _resolved_custom_extensions combines explicit paths and backend overlays [LINK:ct_graph_extensions:363] [CALLER:graph_init|910]
└─ _import_kernels [LINK:ct_graph_import:1703]
   ├─ KernelLibrary.load_paths(paths) [LINK:package_load:269] [CALLER:ct_graph_import|1704]
   │  ├─ existing .mojoc → add_path directly [LINK:package_load:286] [CALLER:ct_graph_import|1704]
   │  └─ source directory → _build_mojo_source_package [LINK:package_builder:141]
   │     ├─ deterministic destination keyed by source path and user ID [LINK:package_builder:147] [CALLER:package_load|289]
   │     ├─ separate process: subprocess_run_mojo(['precompile',...]) [LINK:package_builder:178]
   │     │  ├─ executable override or shutil.which('mojo') [LINK:ct_mojo_run:70] [CALLER:package_builder|178]
   │     │  ├─ parse → declaration resolution → package checks [LINK:ct_parser:391] [LINK:ct_resolver:1756] [LINK:ct_check_lit:1119] [CALLER:package_builder|178]
   │     │  └─ header/compression + MLIR bytecode writer [LINK:ct_package_write:137] [CALLER:package_builder|178]
   │     └─ os.replace → published P1 [LINK:package_builder:179] [CALLER:package_load|289]
   └─ attach loaded library paths to GraphOp.library [LINK:ct_graph_import:1706] [CALLER:graph_init|910]""",
        package(),alternatives={"direct":package(direct=True)},source="graph_init",line=910)

    nodes=objects(("context","active compiler context","thread-local scope","calling thread"),
                  ("prior","prior CURRENT_GRAPH","ContextVar state","caller scope"),
                  ("token","saved reset token","ContextVar token","context manager"),
                  ("current","Graph.current","scoped Graph reference","with-body"),
                  ("ops","ops.custom / graph.output","recording consumers","Python with-body"))
    s=Story(nodes,view="context")
    s.step("Constructor returns before entering the body","entry",55,"G1 shell and P1; forward=None","Python with invokes Graph.__enter__","recording scope about to open",
           "Attention has no forward lambda in __init__; the explicit with-body records its computation.",patch={"prior":"None or an outer graph · saved state"})
    s.step("Create the internal context manager","ct_graph_context_manager",1134,"self=G1","_context=self._enter(); _context.__enter__()","_enter generator advances",
           "The Graph object retains the context manager so __exit__ can close that exact scope.",patch={"token":"pending _enter token"})
    s.step("Ensure the thread has an MLIR context","ct_graph_enter",1143,"current compiler context","with ensure_default_mlir_context()","MCX1 active during recording",
           "Native type/op construction needs an active MLIR context as well as an active graph.",patch={"context":"MCX1 active"})
    s.step("Bind the graph ContextVar","ct_graph_enter",1144,"G1","token=CURRENT_GRAPH.set(G1)","Graph.current→G1",
           "ops.custom can discover its destination without a graph argument at every call.",patch={"token":"Treset · remembers prior Graph", "current":"G1"})
    s.step("Record the caller's operations","entry",57,"q,k,v; G1 current","ops.custom; graph.output","G1 gets custom/output operations",
           "This builds IR now; tensor numerical data arrives later at model.execute.",patch={"ops":"custom insertion/verification → output finalization"})
    s.step("Exit restores the prior Graph.current","ct_graph_enter",1149,"saved token Treset","CURRENT_GRAPH.reset(token)","prior Graph restored",
           "Reset also runs on exceptions, so a failed recording scope does not leak G1 into subsequent construction.",patch={"current":"prior value restored", "token":"used to reset ContextVar", "context":"entered context scope restored"})
    add("graph-forward-context","with Graph · Active recording scope","context","How do ops find G1, and what is restored on exit?",
        "G1 and calling thread", "Scoped CURRENT_GRAPH binding during the with-body",
        """**There are two scopes: the compiler Context and the current Graph.** The MLIR Context supplies the compiler environment. `CURRENT_GRAPH` is a Python ContextVar pointing to the graph receiving operations.

`__enter__` opens `_enter`; `_enter` saves a reset token, sets G1, and yields to the caller. `__exit__` closes it, resetting the previous graph even if construction fails. This differs from running a compiled model: the with-body only records symbolic operations.""",
        """caller: with Graph(...) as graph [LINK:entry:41]
├─ constructor returns G1; Python calls __enter__ [LINK:ct_graph_context_manager:1134]
│  └─ _enter generator context manager [LINK:ct_graph_enter:1142]
│     ├─ ensure_default_mlir_context() [LINK:ct_graph_enter:1143] [CALLER:ct_graph_context_manager|1135]
│     ├─ token=CURRENT_GRAPH.set(self) [LINK:ct_graph_enter:1144] [CALLER:entry|55]
│     └─ yield self → with-body records ops [LINK:ct_graph_enter:1146] [CALLER:entry|56]
└─ __exit__ → context.__exit__ → finally reset(token) [LINK:ct_graph_context_manager:1138]
   └─ CURRENT_GRAPH.reset(token) restores prior graph [LINK:ct_graph_enter:1149] [CALLER:entry|55]""",
        s,source="entry",line=41)

    nodes=objects(("args","B0.arguments","native SSA values","graph entry block"),
                  ("filtered","public arguments","native SSA list","Graph.inputs"),
                  ("q","q","TensorValue wrapper","Python caller"),
                  ("k","k","TensorValue wrapper","Python caller"),
                  ("v","v","TensorValue wrapper","Python caller"),
                  ("data","numerical storage","future Buffer inputs","execute call"))
    s=Story(nodes,view="ports")
    s.step("Read arguments already created by GraphOp","ct_graph_inputs",943,"G1._graph_body=B0","body_args=B0.arguments","[%arg0,%arg1,%arg2]",
           "These values come from the declared signature, not from NumPy.",patch={"args":"%arg0:TQ, %arg1:TK, %arg2:TV"})
    s.step("Compute the internal-chain exclusion count","ct_graph_inputs",944,"_has_chain_input=False","chain_count=0","all three user arguments retained",
           "No trailing chain was declared in the attention input signature.",patch={"filtered":"[%arg0,%arg1,%arg2] · chain_count=0"})
    for key,index in [("q",0),("k",1),("v",2)]:
        s.step(f"Wrap argument {index} as {key}","ct_graph_inputs",949,f"%arg{index}:{T_GPU}","_Value._from_cmlir → Value.from_mlir",f"{key}=TensorValue(%arg{index})",
               "The wrapper carries type and SSA identity; no float32 matrix elements are copied.",patch={key:f"TensorValue(%arg{index}) · shape=[32,32], dtype=f32, gpu:0"})
    s.step("Unpack the tuple in Q/K/V order","entry",56,"Graph.inputs tuple","q,k,v,*_ = graph.inputs","three wrappers; remainder=[]",
           "Argument order defines which real execute input will later occupy each port.",patch={"data":"not supplied yet · BQ/BK/BV created at L78–80"})
    add("graph-forward-inputs","Inputs · Three symbolic wrappers","ports","Where do q, k, v come from when no matrices exist yet?",
        "B0 entry arguments", "q/k/v TensorValue wrappers around the same native SSA arguments",
        """**q is an IR reference, not Q's matrix data.** `Graph.inputs` walks the one entry block's arguments, omits internal chains when needed, and wraps each native value with the Graph API's typed wrapper.

Here q→%arg0, k→%arg1, v→%arg2 all carry float32[32,32]@gpu:0. The actual BQ/BK/BV storage is created at caller lines 78–80 and bound by position at execute. **The three equal types do not mean the three input matrices have equal values.**""",
        """caller: q,k,v,*_ = graph.inputs [LINK:entry:56]
└─ Graph.inputs property [LINK:ct_graph_inputs:934]
   ├─ read _graph_body.arguments [LINK:ct_graph_inputs:943]
   ├─ chain_count=0 here → keep all three [LINK:ct_graph_inputs:944] [CALLER:entry|56]
   ├─ each arg → _Value._from_cmlir(arg) → Value.from_mlir [LINK:ct_graph_inputs:949] [CALLER:entry|56]
   └─ return tuple(q,k,v); no tensor-data allocation [LINK:ct_graph_inputs:948] [CALLER:entry|56]""",
        s,source="entry",line=56)

    nodes=objects(("q","q.type","symbolic Q contract","G1"),
                  ("equation","dimension equation","shape reasoning","attention operator"),
                  ("python","TO","Python TensorType","custom caller"),
                  ("native","native TO","MO TensorType","mo.custom result"),
                  ("allocation","future BO1","runtime allocation","execute plan"))
    s=Story(nodes,view="contracts")
    s.step("Keep symbolic input contracts","entry",56,"q,k,v wrappers","read declared types","Q/K/V=f32[32,32]@gpu:0",
           "The output is declared before any numerical dot product runs.",patch={"q":T_GPU})
    s.step("Determine rows from Q and columns from V","entry",63,"Q:[N,Dq]; K:[N,Dq]; V:[N,Dv]","QKᵀ:[N,N]; row-softmax(S)V:[N,Dv]","TO:[32,32] in this example",
           "Each query yields one weighted value vector. Here Dq=Dv=32.",patch={"equation":"[32,32]×[32,32]ᵀ → [32,32] weights\nweights×V[32,32] → O[32,32]"})
    s.step("Create the caller's output type description","entry",63,"DType.float32; [N,D]; DeviceRef(device)","TensorType(...) ","TO=f32[32,32]@gpu:0",
           "This is a promise to the custom-op verifier/compiler; it contains no output data.",patch={"python":T_GPU})
    s.step("Convert TO to a native result type","custom_api",110,"out_types=[TO]","[t.to_mlir() for t in out_types]","one native MO tensor result type",
           "mo.custom needs a native type for its SSA result because this API declares output types explicitly.",patch={"native":"MO TensorType(shape=[32,32],dtype=f32,device=gpu:0,metadata={})"})
    s.step("Reserve the allocation for later runtime work","ct_execute_callable",384,"compiled execution plan","future execute allocates/returns output storage","BO1 will hold 4096 bytes on GPU",
           "Declaring TO here neither allocates BO1 nor computes its elements.",patch={"allocation":"future BO1 · not allocated in Graph construction"})
    add("graph-output-type","Output type · Future O","contracts","What are the complete example output-type values, and why must the caller supply them?",
        "N/D, dtype and selected device", "TO; one mo.custom result type",
        """**Actual GPU values: `TensorType(DType.float32,[32,32],DeviceRef(\"gpu\",0))`.** CPU uses `[8,8]@cpu:0`. The caller explicitly declares one output because each query row produces a weighted sum of V rows.

The equation is `[N,D]×[D,N]→[N,N]` scores, then `[N,N]×[N,D]→[N,D]` output. **TO is a symbolic contract.** It will become a runtime Buffer allocation only when the initialized model executes with real inputs.""",
        """caller: out_types=[TensorType(dtype,[N,D],DeviceRef.from_device(device))] [LINK:entry:63]
├─ dtype ← DType.float32 [LINK:entry:26] [CALLER:entry|63]
├─ shape ← actual N/D branch [LINK:entry:36] [CALLER:entry|63]
├─ device ← caller's selected device [LINK:entry:29] [CALLER:entry|65]
└─ ops.custom converts TO with to_mlir [LINK:custom_api:110]
   └─ native result type for mo.custom; no Buffer yet [LINK:ct_tensor_to_mlir:439] [CALLER:custom_api|110]""",
        s,source="entry",line=63)

    nodes=objects(("current","G1","active graph","Graph.current"),
                  ("symbol","symbol attribute","StringAttr","mo.custom"),
                  ("device","device attribute","DeviceRefAttr","mo.custom"),
                  ("bd","BD parameter","index IntegerAttr","parameters dictionary"),
                  ("bn","BN parameter","index IntegerAttr","parameters dictionary"),
                  ("dict","parameters","DictAttr","mo.custom"))
    s=Story(nodes,view="attributes")
    s.step("Find the active destination graph","custom_api",94,"CURRENT_GRAPH binding","Graph.current","G1",
           "The current Graph tells this function where to insert its op.",patch={"current":"G1"})
    s.step("Intern the exact registration name","custom_api",96,NAME,"StringAttr.get(name,context)",f'"{NAME}"',
           "This is a registry key, not a Python function call or a filesystem path.",patch={"symbol":f'StringAttr("{NAME}")'})
    s.step("Normalize the driver device to a symbolic reference","custom_api",97,"Accelerator()","DeviceRef.from_device(device)","DeviceRef('gpu',0)",
           "The custom call stores kind/id; it does not store a raw CUDA pointer.",patch={"device":"DeviceRef('gpu',0) · Python descriptor"})
    s.step("Convert integer BD to an index attribute","attention_parameter_attr",49,"Python int BD=8","IntegerAttr.get(IndexType.get(),8)","8 : index",
           "BD is a compile-time tile width, distinct from tensor float32 elements.",patch={"bd":"8 : index"})
    s.step("Convert integer BN to an index attribute","attention_parameter_attr",49,"Python int BN=16","IntegerAttr.get(IndexType.get(),16)","16 : index",
           "BN is query rows per output block, not a symbolic Graph input parameter.",patch={"bn":"16 : index"})
    s.step("Attach the parameter dictionary after insertion","custom_api",113,"BD/BN native attributes","DictAttr.get({BD:...,BN:...})","custom_op.parameters={BD=8:index,BN=16:index}",
           "The registered execute declaration later checks these names/types and specializes them.",patch={"dict":"{BD=8:index,BN=16:index}"})
    s.step("Bridge and attach the native device attribute","custom_api",122,"device.to_mlir()._CAPIPtr","mlir.Attribute._CAPICreate","custom_op.device=DeviceRefAttr('gpu',0)",
           "This preserves device identity across MAX and MLIR Python bindings.",patch={"device":"native DeviceRefAttr(kind='gpu',id=0)"})
    add("graph-custom-attributes","Attributes · Name, device, BN/BD","attributes","What exact name/device/parameter values become MO attributes?",
        "Registration string; driver device; Python BD/BN integers", "StringAttr, DeviceRefAttr, index-valued DictAttr",
        """**Example values are `symbol=\"modular_ops::fused_attention_custom\"`, `device=(gpu,0)`, `BD=8:index`, `BN=16:index`.** Names match the Mojo registration, device kind/id select placement, and index attributes carry compile-time tile choices.

The source obtains the symbol/device first, inserts the op, then attaches parameters and the bridged device attribute before custom-kernel verification. CPU follows the same mechanism with `(cpu,0)` and BD=BN=4. **An index-valued size parameter is not a float32 tensor element.**""",
        """caller: ops.custom(name,device,parameters,values,out_types) [LINK:entry:57]
├─ Graph.current; default_mlir_context() [LINK:custom_api:94]
├─ StringAttr.get(name,context) [LINK:custom_api:96] [CALLER:entry|58]
├─ DeviceRef.from_device(device) [LINK:custom_api:97] [CALLER:entry|59]
├─ insert the custom op before assigning its remaining attributes [LINK:custom_api:109]
├─ parameters dictionary [LINK:custom_api:113]
│  └─ Python int → IntegerAttr(index,value) [LINK:attention_parameter_attr:49] [CALLER:custom_api|116]
└─ device.to_mlir capsule → mlir.Attribute._CAPICreate → custom_op.device [LINK:custom_api:122]
   └─ native DeviceRefAttr('gpu',0) here [LINK:ct_device_ref_to_mlir:181] [CALLER:custom_api|123]""",
        s,source="entry",line=57)

    nodes=objects(("args","Python inputs","TensorValue list","caller"),
                  ("operands","native operands","SSA values","mo.custom"),
                  ("type","native output type","MO TensorType","custom builder"),
                  ("body","B0.operations","ordered IR list","G1"),
                  ("op","CUSTOM1","native mo.custom","B0"),
                  ("result","%o","SSA result / wrapper","CUSTOM1"))
    s=Story(nodes,view="ir-insertion")
    s.step("Receive all three symbolic operands","custom_api",105,"values=[q,k,v]","normalize tensor-like values","three existing TensorValue wrappers",
           "This route rejects BufferValue/opaque in-place operands; real runtime Buffers are not graph-building inputs.",patch={"args":"[q→%arg0,k→%arg1,v→%arg2]", "body":"B0: [CH0=mo.chain.create()]"})
    s.step("Convert the explicit result type","custom_api",110,"out_types=[TO]","TO.to_mlir()","one native f32[32,32] result type",
           "The op builder must know its SSA result contract before creating the result.",patch={"type":T_GPU})
    s.step("Recursively unwrap the operand list","ct_insert_result",1299,"[q,k,v]","unwrap(list) → unwrap each Value → Value capsule","[%arg0,%arg1,%arg2]",
           "Native SSA identities survive the Python wrapper bridge; no tensor data is copied.",patch={"operands":"[%arg0,%arg1,%arg2] · unchanged identities"})
    s.step("Unwrap native types and symbol attributes","ct_insert_result",1301,"TO and StringAttr","type/attribute _CAPIPtr → MLIR factories","builder arguments ready",
           "The same adapter pattern works for values, types and attributes with different capsule kinds.",patch={"type":"native TO handle", "op":f"pending symbol={NAME}"})
    s.step("Choose insertion at B0's end","ct_insert_result",1312,"_ip=None; self._body=B0","mlir.InsertionPoint(B0)","IP2=B0.end",
           "This inserts inside the graph, not into the enclosing module body.",patch={"body":"B0: CH0 | IP2=end"},cursor="IP2")
    s.step("Enter insertion and provenance scopes","ct_insert_result",1317,"IP2 + current _location()","with ip,_location(...)","native builder context active",
           "Location explains source provenance; IP determines the block/position.",patch={"op":"builder ready · location attached"})
    s.step("Create CUSTOM1 and its result","ct_insert_result",1320,"TO; [%q,%k,%v]; symbol","mo.custom(...) inserts one operation","%o=CUSTOM1(%q,%k,%v)",
           "It records a call to registered kernel code. QKᵀ/softmax/PV are not computed during graph construction.",patch={"body":"B0: CH0, %o=mo.custom(%q,%k,%v)", "op":"CUSTOM1 · symbol + operand/result ports", "result":"%o : TO"})
    s.step("Find the staged op and run structural verification","ct_insert_result",1323,"IP2 has no reference operation","last_operation(B0); _verify_op(staged_op)","CUSTOM1 identified",
           "This structural check precedes the later package-specific custom verifier after attributes are attached.",patch={"op":"CUSTOM1 · native structure verified"})
    s.step("Wrap the SSA result for Graph API callers","ct_insert_result",1360,"CUSTOM1.results[0]=%o","Value.from_mlir(_Value._from_cmlir(%o))","results=[TensorValue(%o)]",
           "The caller gets a usable symbolic result and the function retains CUSTOM1 for attribute attachment.",patch={"result":"results=[TensorValue(%o)] · no allocation"})
    s.step("Complete attributes and verify the kernel contract","custom_api",127,"CUSTOM1 + BD/BN + device + P1","verify_custom_op(CUSTOM1)","registered call verified",
           "The detailed registry card explains name resolution and role/type mapping.",patch={"op":"CUSTOM1 · symbol,device,parameters,operands,TO verified"})
    add("graph-custom-insert","mo.custom · Detailed three-input call","ir-insertion","How does the Python call become an operation inside the graph body?",
        "G1/P1; q/k/v; TO; symbol/device/BD/BN", "CUSTOM1 in B0 and results=[TensorValue(%o)]",
        """**Three existing operands enter one custom op; one symbolic result leaves it.** The helper unwraps Python values/types/attributes into MLIR binding handles, activates an insertion point at B0.end, creates the operation, identifies it, verifies its structure and wraps its results.

```mlir
// Schematic MO, not a compiler dump or exact type printer syntax.
mo.graph @fused_attention(%q:T, %k:T, %v:T) -> T {
  %ch = mo.chain.create()
  %o = mo.custom(%q,%k,%v) {symbol="modular_ops::fused_attention_custom",
       parameters={BD=8:index,BN=16:index}, device=(gpu,0)} : T
  // graph.output will insert mo.output %o later
}
```

CUSTOM1 remains an opaque call to the attention implementation; it does not become an exposed graph of scores/exp/divide here. Generic structure verification happens during insertion. Package-specific verification runs after the remaining call attributes are attached.""",
        """caller: results=ops.custom(...) [LINK:entry:57]
├─ reject buffer/opaque in-place values; normalize tensor-like operands [LINK:custom_api:99] [CALLER:entry|57]
└─ graph._add_op_get_op_with_results(mo.custom,[TO],values,symbol=...) [LINK:custom_api:109]
   ├─ unwrap lists recursively; Values/Types/Attributes through CAPI [LINK:ct_insert_result:1293] [CALLER:custom_api|109]
   ├─ ip=InsertionPoint(self._body) [LINK:ct_insert_result:1312]
   ├─ with ip,_location(): op(*unwrapped_args,**kwargs) [LINK:ct_insert_result:1317] [LINK:ct_insert_result:1320]
   ├─ last_operation or prev_operation → staged_op [LINK:ct_insert_result:1323] [CALLER:ct_insert_result|1320]
   ├─ _verify_op(staged_op) → structural verifier [LINK:ct_insert_result:1329] [LINK:ct_verify_op:1186] [CALLER:ct_insert_result|1329]
   ├─ set output parameter declarations; wrap results [LINK:ct_insert_result:1351] [LINK:ct_insert_result:1360] [CALLER:custom_api|109]
   └─ return results, staged_op → attach params/device → custom verifier [LINK:custom_api:113] [LINK:custom_api:127] [CALLER:entry|57]""",
        s,source="entry",line=57)

    nodes=objects(("call","CUSTOM1","MO operation contract","G1"),
                  ("analysis","AnalysisA1 / cache key","Python/native declaration analysis","KernelLibrary"),
                  ("registry","MojoRegistry","declaration maps","KernelAnalysis"),
                  ("candidate","kernel declaration","registration + execute","P1"),
                  ("inputs","input-role mapping","type/mutability checks","operand inference"),
                  ("outputs","output-role mapping","destination-passing contract","operand inference"),
                  ("params","parameter map","names/types","custom verifier"),
                  ("gate","verification outcome","success/error","Python caller"))
    def verify(bad=False,cached=False):
        s=Story(nodes,label="Comparison · unknown parameter" if bad else "Reuse analysis · verify a fresh call" if cached else "First path analysis · valid contract",view="validation")
        s.step("Start after all call fields are attached","custom_api",127,"CUSTOM1 + P1 library","KernelLibrary.verify_custom_op","call contract available",
               "Kernel verification runs with symbol, device, operands, output types and explicit parameters present.",patch={"call":f"{NAME}\ninputs=Q/K/V:T; output=TO; gpu:0\nBD=8, BN=16"+(", BX=2" if bad else "")})
        s.step("Read the analysis library paths","ct_graph_verify",314,"self._analysis.library_paths","paths = self._analysis.library_paths","resolved package paths including P1",
               "The cache is scoped by loaded declaration paths, not numerical Q/K/V values.",patch={"analysis":"current AnalysisA1; paths=[P1,...default packages]"})
        s.step("Make a path-set cache key","ct_graph_verify",316,"nonempty library paths","frozenset(str(p) for p in paths)","PATHKEY1",
               "An immutable set makes the analysis lookup independent of path-list order.",patch={"analysis":"PATHKEY1=frozenset({P1,...defaults})"})
        s.step("Look for reusable declaration analysis","ct_graph_verify",317,"PATHKEY1","KernelLibrary._analysis_cache.get(PATHKEY1)","cached AnalysisA1" if cached else "no prior entry",
               "Reusing analysis does not skip validation of this new custom call or rerun source precompile.",patch={"analysis":"cache HIT → AnalysisA1 retained" if cached else "cache MISS → current AnalysisA1"})
        s.step("Validate through the chosen Analysis","ct_graph_verify",320 if cached else 322,"cached/current AnalysisA1 and CUSTOM1","self._analysis.verify_custom_op(custom_op)","native declaration verifier",
               "The cached branch first installs that analysis on the library; both branches call the native verifier.",patch={"analysis":"AnalysisA1 selected; verify CUSTOM1 now"})
        if not cached:
            s.step("Create the analysis declaration module lazily","attention_analysis_module",191,"AnalysisA1; no kernelDecls module yet","MAXG_createKernelDeclsModule(ctx)","DECLMODULE1",
                   "This declaration module is separate from G1's computation body and contains kernel interfaces for analysis.",patch={"registry":"DECLMODULE1 empty; native registry not built"})
            s.step("Import each package's declarations","attention_analysis_module",199,"DECLMODULE1; libraryPathsSet","MAXG_addKernelPackage(module,path)","P1 plus built-in kernel/primitive declarations",
                   "The first import also loads the built-ins. This imports declarations from prepared packages; it does not execute the attention kernel.",patch={"registry":"DECLMODULE1: P1 FusedAttention + built-in declarations"})
        s.step("Read loaded declaration maps","ct_validation_registry",183 if not cached else 184,"AnalysisA1 declaration module","build MojoRegistry(module)" if not cached else "return existing registry","registration → candidate declarations",
               "MojoRegistry is a compiler-side index of kernel/primitive/shape-function declarations, not a Python tensor dictionary.",patch={"registry":f"kernelByName['{NAME}']→[FusedAttention decl]\nprimitiveByName / shapeFuncByName also indexed"})
        s.step("Check that the registration exists","ct_validation_native",642,"operationName","containsKernel(operationName)","candidate present",
               "An unknown name fails before code lowering; it is not deferred to a missing GPU symbol at launch.",patch={"candidate":"FusedAttention declaration · found"})
        s.step("Resolve the device-compatible declaration","ct_validation_native",650,"registration + DeviceRef(gpu,0)","getResolvedKernel(name,device)","declaration with execute function",
               "Device/overlay resolution can distinguish available implementation declarations.",patch={"candidate":"FusedAttention.execute · selected declaration"})
        s.step("Require an execute implementation","ct_validation_native",672,"KernelDeclAdaptor selected declaration","check adaptor.executeFunction exists","execute interface available",
               "The attention registration is an execute kernel; the reduction-specific branch does not apply.",patch={"candidate":"FusedAttention.execute · interface accepted"})
        s.step("Map input zero to query","ct_validation_mapping",315,"MO %q: immutable tensor T","matchOperandsOrResultsKernelDecl<true>","%q→query",
               "Count, supported tensor type and mutability must agree with the declaration.",patch={"inputs":"0: %q→query (f32,rank2,immutable)"})
        s.step("Map input one to key","ct_validation_mapping",315,"MO %k: immutable tensor T","continue input mapping","%k→key",
               "The same shape/type does not make Q and K interchangeable; operand order fixes roles.",patch={"inputs":"0: %q→query; 1: %k→key"})
        s.step("Map input two to value","ct_validation_mapping",315,"MO %v: immutable tensor T","complete input mapping","%v→value",
               "ctx is supplied by generated runtime glue, not a fourth Graph input.",patch={"inputs":"0→query, 1→key, 2→value · three matched inputs"})
        s.step("Map the result to a mutable output destination","ct_validation_mapping",324,"MO result %o:TO","matchOperandsOrResultsKernelDecl<false> with no mut→immut widening","%o→output",
               "execute writes the provided output destination. It need not return a Mojo tensor by value.",patch={"outputs":"%o→mut output : f32 rank2\nctx→runtime argument, not user tensor"})
        s.step("Collect declared parameter names","attention_validation_parameters",451,"struct/function/shape declarations","union/deduplicate compile-time parameter declarations","dtype,rank,BN,BD,target,… inferred/declared",
               "Supplied names must exist in the merged declaration list; inferred parameters need not all be explicit in Python.",patch={"params":"declared names include BN,BD; dtype/rank/target infer from types/device"})
        s.step("Index each declared name and expected native type","attention_validation_parameters",473,"merged inputParams","paramDecls[maybeDemangleName(name)]={param,typeName}","native parameter lookup map",
               "Name lookup and native type compatibility are separate checks.",patch={"params":"paramDecls: BD→Int-compatible type; BN→Int-compatible type; inferred declarations…"})
        s.step("Lookup supplied BD and BN names","attention_validation_parameters",478,"BD=8:index; BN=16:index","paramDecls.find(namedAttr.getName())","BD/BN declarations found",
               "An undeclared supplied name follows the immediate error branch.",patch={"params":"BD and BN names found; type checks still required"})
        s.step("Validate integer attributes against native parameter types","attention_validation_parameters",492,"BD=8:index; BN=16:index and declared Int types","check expected type and IntegerAttr representation","BD/BN accepted",
               "This is declaration compatibility, not a numerical attention test or a benchmark.",patch={"params":"BD→Int/index attr8; BN→Int/index attr16 · accepted"})
        if bad:
            s.step("Reject an unknown supplied name","attention_validation_parameters",480,"BX=2:index","BX absent from paramDecls","error: parameter BX does not exist",
                   "The Python caller receives an error instead of a supposedly valid graph custom call.",patch={"gate":"ERROR · unknown BX; compilation stops"})
        else:
            s.step("Return success to the caller","attention_validation_parameters",629,"all mappings and parameters accepted","success()","results may be recorded as graph output",
                   "Verification establishes the symbolic contract; actual numeric computation still occurs at execute.",patch={"gate":"PASS · declaration/type/parameter contract verified"})
            if not cached:
                s.step("Cache the successful declaration analysis","ct_graph_verify",323,"PATHKEY1 and successful AnalysisA1","KernelLibrary._analysis_cache[PATHKEY1]=self._analysis","later calls can reuse AnalysisA1",
                       "Only analysis is reused; a later call's operands, result types and parameters are still verified.",patch={"analysis":"PATHKEY1→AnalysisA1 cached after successful verify"})
        return s
    add("graph-custom-verify","Kernel validation · P1 and MO","validation","How are registration lookup, input/output roles and compile-time parameters checked?",
        "CUSTOM1 and P1 declarations", "Matched declaration + operand/result roles + accepted attributes, or error",
        """**The verifier matches a symbolic call to a declared kernel interface.** The registry maps the registration name to declarations; device resolution selects the suitable declaration. Operand inference then matches three MO operands to query/key/value and one result to the mutable output destination.

Python first selects analysis by the loaded package-path set. A cache miss lazily imports declaration packages and builds the registry; a hit reuses them, **while still verifying the new call**. Successful analysis is cached after native verification. This is distinct from source precompile and from native code-cache reuse.

`ctx` is a generated runtime argument, not a fourth tensor. dtype/rank/target can be inferred; Python supplies BN/BD explicitly. Parameter checks validate names and native attribute types. The comparison adds an invalid BX name to show where verification fails. **A passed verifier does not prove numeric correctness or GPU performance.**""",
        """caller: graph._kernel_library.verify_custom_op(custom_op) [LINK:custom_api:127]
├─ KernelLibrary.verify_custom_op: paths → path-set key → analysis lookup [LINK:ct_graph_verify:314] [LINK:ct_graph_verify:316] [LINK:ct_graph_verify:317] [CALLER:custom_api|127]
│  ├─ cache hit → retain cached analysis; verify this call [LINK:ct_graph_verify:319] [LINK:ct_graph_verify:320] [CALLER:custom_api|127]
│  ├─ miss → verify current analysis; cache only after success [LINK:ct_graph_verify:322] [LINK:ct_graph_verify:323] [CALLER:custom_api|127]
│  └─ no paths → current native analysis verifier directly [LINK:ct_graph_verify:325] [CALLER:custom_api|127]
├─ native Analysis::verifyCustomOp → ensureRegistry → ensureModule [LINK:ct_analysis_verify:70] [LINK:ct_validation_registry:181]
│  ├─ module absent → create declaration module; import paths + built-ins [LINK:attention_analysis_module:191] [LINK:attention_analysis_module:199] [CALLER:ct_validation_registry|181]
│  └─ registry absent → MojoRegistry(module); otherwise reuse existing [LINK:ct_validation_registry:183] [LINK:ct_validation_registry:184] [CALLER:ct_analysis_verify|70]
└─ MOGG::verifyCustomOp(registry,op) [LINK:ct_validation_native:634]
   ├─ get operation name; containsKernel gate [LINK:ct_validation_native:636] [CALLER:ct_analysis_verify|70]
   ├─ getDeviceRefForOp; getResolvedKernel [LINK:ct_validation_native:650] [CALLER:ct_analysis_verify|70]
   ├─ ensure execute exists [LINK:ct_validation_native:672] [CALLER:ct_analysis_verify|70]
   ├─ mapMOOpToKernelImplementation [LINK:ct_validation_mapping:336]
   │  ├─ inputs %q/%k/%v → query/key/value [LINK:ct_validation_mapping:315] [CALLER:ct_validation_native|677]
   │  └─ result %o → mutable output destination [LINK:ct_validation_mapping:324] [CALLER:ct_validation_native|677]
   └─ validateCustomOpParameters [LINK:attention_validation_parameters:427]
      ├─ merge declared struct/function/shape parameters [LINK:attention_validation_parameters:451] [LINK:attention_validation_parameters:459] [CALLER:ct_validation_native|685]
      ├─ supplied name missing → error [LINK:attention_validation_parameters:480] [CALLER:ct_validation_native|685]
      └─ native attribute type compatible → success [LINK:attention_validation_parameters:492] [LINK:attention_validation_parameters:629] [CALLER:ct_validation_native|685]""",
        verify(),alternatives={"invalid":verify(True),"reuse":verify(cached=True)},source="custom_api",line=127)

    nodes=objects(("native","%o","native SSA value","CUSTOM1"),
                  ("core","MAX _Value","C API wrapper","Graph binding"),
                  ("wrapper","TensorValue","Graph API wrapper","results list"),
                  ("type","result contract","TensorType fields","wrapper"),
                  ("list","results","Python list","caller"),
                  ("buffer","BO1","future numeric storage","execute"))
    s=Story(nodes,view="ports")
    s.step("Start with one native result","ct_insert_result",1355,"CUSTOM1 produced one SSA value","read result handles","%o:TO",
           "The registered output destination is represented by this one symbolic result.",patch={"native":"%o · CUSTOM1.result[0]"})
    s.step("Bridge the result value handle","ct_insert_result",1360,"mlir.Value %o","_Value._from_cmlir(%o)","MAX _Value for %o",
           "This bridge preserves SSA identity just as the operation bridge preserves OP1.",patch={"core":"_Value→%o"})
    s.step("Select the Graph API wrapper from its type","ct_insert_result",1360,"native value typed as MO TensorType","Value.from_mlir","TensorValue(%o)",
           "The wrapper gives Python shape/dtype/device access without storing matrix elements.",patch={"wrapper":"TensorValue(%o)","type":"shape=[32,32],dtype=DType.float32,device=gpu:0,metadata={}"})
    s.step("Return a one-element list","custom_api",129,"wrapped results and verified CUSTOM1","return results","[TensorValue(%o)]",
           "The script keeps the list and expands it into graph.output; it does not select [0].tensor like add-one.",patch={"list":"results=[TensorValue(%o)]"})
    s.step("Record the result as a graph output","entry",68,"*results","graph.output(*results)","MO output contract completed",
           "This is still Graph construction, so the numeric output Buffer remains a future object.",patch={"buffer":"not allocated · future BO1:f32[32,32]@gpu:0"})
    add("graph-custom-result","Result wrappers · One symbolic output","ports","What exactly is inside Return [TensorValue], and how is it used?",
        "CUSTOM1.results=[%o]", "[TensorValue(%o)] with full type fields; no numerical output",
        """**Example list: `[TensorValue(%o : float32[32,32] @ gpu:0)]`.** `%o` names one SSA result owned by CUSTOM1. The wrapper exposes `dtype=DType.float32`, `shape=[32,32]`, `device=gpu:0`, and empty explicit metadata.

It has no 4096-byte output allocation yet. The caller does `graph.output(*results)`; later `model.execute(Q,K,V)` returns a different Python list containing a real Buffer. Equal-looking result lists at graph and runtime stages contain **different kinds of objects**.""",
        """caller: ops.custom returns results [LINK:custom_api:129]
├─ native builder result %o [LINK:ct_insert_result:1355]
├─ _Value._from_cmlir → Value.from_mlir → TensorValue(%o) [LINK:ct_insert_result:1360] [CALLER:custom_api|109]
└─ Python results list → graph.output(*results) [LINK:entry:68]
   └─ symbolic output recorded now; runtime Buffer returned later [LINK:entry:82] [CALLER:entry|68]""",
        s,source="custom_api",line=129)

    nodes=objects(("results","Python output structure","symbolic wrappers","caller"),
                  ("values","native output values","SSA list","Graph.output"),
                  ("signature","G1 signature","native function/generator type","GraphOp"),
                  ("body","B0.operations","MO body","G1"),
                  ("names","argument/result names","engine metadata","GraphOp"),
                  ("verified","graph verification","success gate","compiler IR"))
    s=Story(nodes,view="ir-insertion")
    s.step("Flatten the caller's result structure","attention_graph_output",1433,"graph.output(*[TensorValue(%o)])","tree.flatten(root)","one leaf %o",
           "Output structure is remembered for API handling; string leaves are rejected.",patch={"results":"[TensorValue(%o)] · one leaf"})
    s.step("Gather native output value handles","attention_graph_output",1449,"output wrappers; no chain input","[v._mlir_value for v in output_values]","[%o]",
           "Internal chains would be packed in chain-input graphs; attention has one tensor result.",patch={"values":"[%o:TO]"})
    s.step("Convert the output type handles","attention_graph_output",1453,"%o.type","mlir.Type._CAPICreate(value.type._CAPIPtr)","TO in MLIR Python binding",
           "Type bridges adapt handles, not data layouts or numeric elements.",patch={"signature":"pending (TQ,TK,TV) → (TO)"})
    s.step("Replace the initially-empty result signature","attention_graph_output",1459,"B0 argument types + [TO]","FunctionType.get; kgen.generator; attach signature/functionType","G1:(TQ,TK,TV)→(TO)",
           "Compiler and runtime now have a complete input/output contract.",patch={"signature":"functionType=(TQ,TK,TV)→(TO)\nsignature=kgen.generator<functionType>"})
    s.step("Insert the output terminator","attention_graph_output",1469,"native %o; output params=[]","_add_op_generated(_mo.OutputOp,...)","mo.output %o",
           "The terminator tells the compiler which SSA value leaves the graph.",patch={"body":"B0: CH0, %o=mo.custom(%q,%k,%v), mo.output %o"})
    s.step("Attach engine-facing argument and result names","attention_graph_output",1479,"3 user inputs; 1 output","argument_names/result_names attributes","input0/input1/input2; output0",
           "The engine uses stable public names when exposing the model interface.",patch={"names":"inputs=[input0,input1,input2]\nresults=[output0]"})
    s.step("Verify the complete graph","attention_graph_output",1492,"G1 body + signatures + terminator","self._mlir_op.verify()","verified G1 ready for compilation",
           "Whole-graph structural consistency is checked after output recording, not after real tensor execution.",patch={"verified":"PASS · complete symbolic G1"})
    s.step("Close the with-body recording scope","ct_graph_context_manager",1138,"verified G1","__exit__ restores prior CURRENT_GRAPH","retained G1/P1 survive",
           "Session creation, compiler passes, init and execute are distinct later calls.",patch={"results":"G1 retained; recording scope closed"})
    add("graph-forward-output","Graph.output · Complete G1","ir-insertion","How does a symbolic result finish the graph and its callable signature?",
        "results=[TensorValue(%o)]", "mo.output, completed signatures/names and verified G1",
        """**Before output: `(TQ,TK,TV)→()`; afterward: `(TQ,TK,TV)→(TO)`.** The Graph API flattens output leaves, bridges native types, updates both function and generator signatures, inserts `mo.output %o`, attaches engine-facing names and verifies the whole graph.

```text
MODULE1 / G1 / R0 / B0(%q:TQ,%k:TK,%v:TV)
  CH0 = chain.create()
  %o = custom(%q,%k,%v) : TO
  output %o
public interface: input0,input1,input2 → output0
```

This creates a complete **symbolic** computation; no numeric Q/K/V or target instructions were executed in these recording steps.""",
        """caller: graph.output(*results) [LINK:entry:68]
├─ flatten output structure and normalize Value wrappers [LINK:attention_graph_output:1433] [CALLER:entry|68]
├─ optional internal chains; native value/type handles [LINK:attention_graph_output:1449] [CALLER:entry|68]
├─ FunctionType(inputs,outputs); attach generator and functionType [LINK:attention_graph_output:1459] [CALLER:entry|68]
├─ _add_op_generated(OutputOp,mlir_values,params=[]) [LINK:attention_graph_output:1469] [CALLER:entry|68]
├─ argument_names/input0…; result_names/output0 [LINK:attention_graph_output:1479] [CALLER:entry|68]
└─ verify completed G1, then with-scope exit [LINK:attention_graph_output:1492] [LINK:ct_graph_context_manager:1138] [CALLER:entry|68]""",
        s,source="entry",line=68)
    return cards
