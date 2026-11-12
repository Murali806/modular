"""Graph constructor/recording cards: concrete native calls, fields and scopes."""
from compiler_attention_lessons import NAME, Story, T_CPU, T_GPU, lesson, objects


def graph_lessons():
    result = []
    def add(*args, **kw):
        result.append(lesson(*args, **kw))

    nodes = objects(("device", "D0 · selected device", "driver handle", "Python caller"),
                    ("sizes", "N / D", "static integers", "Python caller"),
                    ("tiles", "BN / BD", "compile-time integers", "custom call"),
                    ("q", "TQ", "Python type", "Graph arguments"),
                    ("k", "TK", "Python type", "Graph arguments"),
                    ("v", "TV", "Python type", "Graph arguments"),
                    ("output", "TO", "future output type", "custom call"))
    def sizes(cpu):
        n,d,bn,bd = (8,8,4,4) if cpu else (32,32,16,8)
        target = "cpu:0" if cpu else "gpu:0"
        s = Story(nodes, label="Actual CPU caller" if cpu else "Actual GPU caller", view="contracts")
        s.step("Choose the device before graph construction", "entry", 29,
               "accelerator_count() == 0" if cpu else "accelerator_count() > 0",
               "CPU() if none; otherwise Accelerator()", target,
               "The caller's selection is used consistently by types, session and Buffer.to(device).",
               patch={"device": target})
        s.step("Set sequence length and feature width", "entry", 31 if cpu else 36,
               "CPU branch" if cpu else "accelerator branch", "assign N and D",
               f"N={n}; D={d}", "N counts query/key rows; D is both the input feature width and output value width here.",
               patch={"sizes": f"N={n}, D={d}"})
        s.step("Set query and output-column tile sizes", "entry", 33 if cpu else 38,
               f"N=D={n}", "assign BD, then BN", f"BD={bd}; BN={bn}",
               "BN is rows per query tile; BD is output columns per block. GPU key rows use separate BN_1=8.",
               patch={"tiles": f"BN={bn}, BD={bd}; key rows={4 if cpu else 8}"})
        for key,line in (("q",45),("k",48),("v",51)):
            s.step(f"Declare {key.upper()}'s type", "entry", line, f"dtype=float32; shape=[{n},{d}]; device={target}",
                   "TensorType(dtype, shape, DeviceRef.from_device(device))", f"T{key.upper()}={T_CPU if cpu else T_GPU}",
                   "The three independent graph arguments share one type contract; none holds numerical matrix data.",
                   patch={key: T_CPU if cpu else T_GPU})
        s.step("Derive the output dimensions from the attention equation", "entry", 63,
               f"Q:[{n},{d}], Kᵀ:[{d},{n}], V:[{n},{d}]",
               f"[{n},{d}] @ [{d},{n}] → S:[{n},{n}]; row-softmax(S) @ V",
               f"O:[{n},{d}]", "Softmax normalizes across keys, independently for every query row.",
               patch={"output": T_CPU if cpu else T_GPU},
               equation=f"Q[{n},{d}] × Kᵀ[{d},{n}] → scores[{n},{n}]\nrow softmax → weights[{n},{n}]\nweights × V[{n},{d}] → O[{n},{d}]")
        return s
    add("graph-sizes", "Sizes · Actual CPU/GPU contracts", "contracts",
        "What do N, D, BN and BD mean, and why are there three equal-shaped inputs?",
        "CPU or accelerator availability; DType.float32", "TQ/TK/TV and one TO; BN/BD tile parameters",
        """**Q asks, K supplies the vectors to compare, and V supplies the content to mix.** For each query row i, the scores are dot products Q[i,:]·K[j,:]. Softmax across j makes weights; O[i,:] is their weighted sum of V rows.

The actual [caller](sources/max/examples/custom_ops/fused_attention.py#L29) selects CPU [8,8] or GPU [32,32]. Its Q/K/V have the same type but represent different matrices. **N is a row count; D is a feature count.** BN and BD subdivide work, not the number of graph arguments.

On GPU, BN=16 and BD=8 divide O into eight [16,8] tiles. The independent key tile BN_1=8 means four key iterations per output block. CPU uses BN=BD=4 and two key iterations. The source computes **unscaled rank-2 attention**; heads, causal masking, dropout and 1/√D scaling are absent.""",
        """caller: main() chooses sizes and type arguments [LINK:entry:29]
├─ accelerator_count() selects CPU() or Accelerator() [LINK:entry:29]
├─ CPU: N=D=8, BD=BN=4 [LINK:entry:31] [CALLER:entry|30]
├─ GPU: N=D=32, BD=8, BN=16 [LINK:entry:36] [CALLER:entry|35]
├─ Q type ← dtype + [N,D] + DeviceRef(device) [LINK:entry:45] [CALLER:entry|44]
├─ K type ← same contract, different input slot [LINK:entry:48] [CALLER:entry|47]
├─ V type ← same contract, different input slot [LINK:entry:51] [CALLER:entry|50]
└─ TO ← declared [N,D]; corresponds to softmax(QKᵀ)V [LINK:entry:63] [CALLER:entry|62]""",
        sizes(False), alternatives={"cpu": sizes(True)}, source="entry", line=29)

    nodes = objects(("default","MCX1 · shared default","MLIR Context","module _mlir_context"),
                    ("main","main.current","thread-local context stack","main thread"),
                    ("worker","worker.current","thread-local context stack","background thread"),
                    ("guard","decorator / scope","Python callable","Graph constructor"),
                    ("ir","G1 + types","MLIR objects","selected context"))
    def context(preserve=False):
        s=Story(nodes, label="Comparison · preserve an active context" if preserve else "Background-thread construction",view="context")
        s.step("Create the interpreter's shared default once", "ct_graph_context_state",26,
               "import max._mlir_context", "_DEFAULT_MLIR_CONTEXT = mlir.Context()", "MCX1",
               "Context owns dialect/type/attribute uniquing. It is not the MAX runtime context.",
               patch={"default":"MCX1 · module-level instance"})
        s.step("Enter the default on the main thread", "ct_graph_context_state",32,
               "MCX1", "MCX1.__enter__()", "main.current → MCX1",
               "Entering a context changes this thread's active context stack.",
               patch={"main":"[MCX1]"})
        s.step("A worker has its own context stack", "ct_mlir_thread_pool",97,
               "MLIRThreadPoolExecutor.submit(Graph, ...)", "submit call_with_default_mlir_context wrapper",
               "worker.current → MCX2" if preserve else "worker.current → None",
               "Thread-local active state is not inherited merely because the default object is shared.",
               patch={"worker":"[MCX2 · explicitly entered]" if preserve else "[] · current=None", "guard":"call_with_default_mlir_context"})
        s.step("The guard reads this thread's current context", "ct_graph_context_state",47,
               "worker current", "if context is None: with _DEFAULT_MLIR_CONTEXT",
               "preserve MCX2" if preserve else "temporarily enter MCX1 on worker",
               "It does not replace an already-active context.",
               patch={"worker":"[MCX2]" if preserve else "[MCX1]","guard":"scope active"})
        s.step("Graph's decorator establishes the same guard", "ct_graph_context",83,
               "@in_default_mlir_context", "wrapper → ensure_default_mlir_context → fn",
               "Graph.__init__ executes with an active MLIR context",
               "A decorator wraps a function; @ is not an instance attribute.",
               patch={"guard":"wrapper(ensure_default_mlir_context())"})
        s.step("Construct operations while context is active", "graph_init",860,
               "input types and builder", "create native GraphOp", "G1 / types / attrs use active context",
               "The native builder needs a valid context to construct MLIR types and operations.",
               patch={"ir":"G1 and interned type/attribute handles"})
        s.step("Restore the worker's previous scope", "ct_graph_context_state",64,
               "constructor returns", "leave with _DEFAULT_MLIR_CONTEXT if it was entered",
               "worker stays in MCX2" if preserve else "worker.current → None; MCX1 survives",
               "Scope restoration does not destroy retained graph IR or the shared default.",
               patch={"worker":"[MCX2]" if preserve else "[] · current=None","guard":"returned"})
        return s
    add("graph-context","A.0 · default MLIR context","context",
        "Which context is global, which state is thread-local, and what does the decorator do?",
        "Calling thread; shared MCX1; optional active MCX2","An active context during constructor/with-body; previous scope restored",
        """**The shared object and the active scope are different things.** [_DEFAULT_MLIR_CONTEXT](sources/max/python/max/_mlir_context.py#L26) is one module-level instance. `mlir.Context.current` reads the top of this thread's entered-context stack. A new worker can see no current context even while main has MCX1 entered.

An MLIR Context is a compiler environment: dialects define operation/type names; interned types and attributes can be reused in that environment. It is **not a tensor store, graph container, or Engine::Context**.

`@in_default_mlir_context` replaces the constructor callable with a wrapper using `ensure_default_mlir_context()`. [MLIRThreadPoolExecutor](sources/max/python/max/_mlir_context.py#L89) also wraps submitted tasks. Both preserve an existing current context and enter MCX1 only when needed. Use the comparison selector to follow that branch; it is not a separate default instance per worker.""",
        """caller: @in_default_mlir_context wraps Graph.__init__ [LINK:attention_ctor_decorator:819]
├─ shared _DEFAULT_MLIR_CONTEXT = mlir.Context() [LINK:ct_graph_context_state:26]
├─ main enters MCX1; shutdown registers matching exit [LINK:ct_graph_context_state:32] [CALLER:ct_graph_context_state|32]
├─ wrapper(*args) enters ensure_default_mlir_context() [LINK:ct_graph_context:83]
│  ├─ read mlir.Context.current on THIS thread [LINK:ct_graph_context_state:63]
│  ├─ None → with MCX1; yield to constructor [LINK:ct_graph_context_state:64] [CALLER:ct_graph_context|83]
│  └─ active context → yield without replacement [LINK:ct_graph_context_state:67] [CALLER:ct_graph_context|83]
├─ fn(*args) constructs Graph IR [LINK:ct_graph_context:84] [CALLER:attention_ctor_decorator|819]
└─ worker route: MLIRThreadPoolExecutor.submit [LINK:ct_mlir_thread_pool:90]
   └─ submit(call_with_default_mlir_context, fn, ...) [LINK:ct_mlir_thread_pool:97] [CALLER:ct_mlir_thread_pool|97]""",
        context(),alternatives={"active":context(True)},source="attention_ctor_decorator",line=819)

    nodes=objects(("thread","current context","thread-local scope","calling thread"),
                  ("stack","Python stack summaries","provenance metadata","_location"),
                  ("loc","LOC1","MLIR Location","context"),
                  ("module","MODULE1.location","attached attribute","module op"),
                  ("graph","G1 / input locations","attached provenance","GraphOp + B0 arguments"))
    def locations(trace=True):
        s=Story(nodes,label="Traceback capture enabled" if trace else "Actual branch · capture disabled",view="location")
        s.step("Require an active MLIR context", "ct_graph_location",442,"current context MCX1",
               "if not mlir.Context.current: raise", "Location construction can continue",
               "Location belongs to an MLIR context; missing context is an error.",
               patch={"thread":"MCX1 active"})
        s.step("Read the live traceback policy", "ct_graph_location",449,"source_tracebacks="+str(trace),
               "choose captured or unknown location", "extract stack" if trace else "Location.unknown()",
               "The live setting is read per operation, not cached at import time.",
               patch={"loc":None if trace else "LOC1=unknown"})
        if trace:
            s.step("Capture summaries instead of live frames", "ct_graph_location",457,
                   "main() → fused_attention.py:41 → Graph.__init__ → _location",
                   "traceback.extract_stack(); trim ignored/internal frames", "FrameSummary list",
                   "Summaries retain file/line/function information without keeping Python frame objects and closures alive.",
                   patch={"stack":"main at fused_attention.py:41 · illustrative stack summaries"})
            s.step("Build native opaque provenance", "ct_graph_frame_loc",166,"summaries + MCX1",
                   "_graph.frame_loc(context,tb)", "LOC1: opaque MLIR Location containing stack metadata",
                   "A readable label is not the literal printed representation of this opaque attribute.",
                   patch={"loc":"LOC1 · Python call-site provenance"})
        s.step("Optionally wrap active profiling scopes", "ct_graph_location",470,"LOC1 + active scope labels",
               "profile_scope_location(context,name,location,color)", "location with zero or more scope wrappers",
               "Profile scopes are independent of traceback capture; this example has none.",
               patch={"loc":"LOC1 · no active profile scopes in this caller"})
        s.step("Scope loc and create the module", "graph_init",851,"with _location() as loc",
               "builtin.ModuleOp(location=loc)", "MODULE1.location → LOC1",
               "The Location's with-scope makes it current; MAX also passes it explicitly.",
               patch={"module":"MODULE1 → LOC1"})
        s.step("Attach the same provenance to GraphOp and arguments", "ct_graph_op_build",849,
               "builder, loc, inputTypes", "body.addArgument(input, state.location)", "G1 and %q/%k/%v carry LOC1",
               "An error involving a graph argument can refer to its creation provenance.",
               patch={"graph":"G1/%q/%k/%v → LOC1"})
        s.step("Leaving with does not erase provenance", "graph_init",879,"Location scope exits",
               "continue constructor", "operations keep attached LOC1",
               "Location is retained metadata; it is not a pointer to a tensor or a device.",
               patch={"thread":"scope restored; graph metadata retained"})
        return s
    add("graph-location","_location() · What is loc?","location",
        "How does a Python call site become diagnostic metadata on native IR?",
        "Current MLIR context, traceback policy, optional profiling scopes","LOC1 attached to module, graph and input arguments",
        """**loc is provenance metadata.** it provides stack trace metadata for mo.GraphOp & builtin.ModuleOp(location=loc) which is MLIR module.

[_location()](sources/max/python/max/graph/graph.py#L438) checks the current context, reads the live capture flag, extracts Python stack summaries when enabled, and can wrap profiling labels. In this caller an illustrative source stack ends at `main → fused_attention.py:41 → Graph(...)`. The native binding makes an **opaque Location carrying summaries**; it need not print as a simple filename-location string.

The same LOC1 is explicitly passed to the module and graph. [The native GraphOp builder](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L849) adds it to each input block argument. After the with-scope closes, the operations still carry it. A disabled capture path uses unknown; no fake traceback should be displayed as captured evidence.""",
        """caller: with _location() as loc [LINK:graph_init:851]
└─ _location(ignore_frames=1, attach_profile_scopes=True) [LINK:ct_graph_location:438]
   ├─ current context absent → RuntimeError [LINK:ct_graph_location:442] [CALLER:graph_init|851]
   ├─ capture disabled → Location.unknown() [LINK:ct_graph_location:449] [CALLER:graph_init|851]
   ├─ enabled → traceback.extract_stack() → trimmed summaries [LINK:ct_graph_location:457]
   │  └─ _graph.frame_loc(context,tb) → opaque Location [LINK:ct_graph_frame_loc:166] [CALLER:ct_graph_location|459]
   ├─ wrap active profile scopes, if any [LINK:ct_graph_location:470] [CALLER:ct_graph_location|471]
   ├─ ModuleOp(location=loc) stores LOC1 [LINK:graph_init:856] [CALLER:graph_init|856]
   └─ GraphOp receives loc; body arguments use state.location [LINK:ct_graph_op_build:849] [CALLER:graph_init|862]""",
        locations(False),alternatives={"trace":locations()},source="graph_init",line=851)

    nodes=objects(("wrapper","existing Module wrapper","Python wrapper","caller"),
                  ("module","MODULE1","builtin.module operation","MLIR context"),
                  ("region","Rmodule","body region","MODULE1"),
                  ("block","Bmodule","body block","Rmodule"),
                  ("sibling","encoder","existing named graph","Bmodule"),
                  ("graph","G1","mo.graph operation","Bmodule"))
    def modules(reuse=False):
        s=Story(nodes,label="Comparison · reuse existing Module" if reuse else "Actual · create a fresh module",view="ir")
        s.step("Choose the constructor branch", "graph_init",856,"module=existing" if reuse else "module=None",
               "module.mlir_module if module else builtin.ModuleOp(location=loc)",
               "same native module" if reuse else "fresh module",
               "Reuse is a reference selection; it does not clone sibling definitions.",
               patch={"wrapper":"existing Module → MODULE1" if reuse else "None"})
        if reuse:
            s.step("Retain existing module and siblings", "ct_graph_module",637,"Module.mlir_module",
                   "read the mlir_module attribute stored by Module.__init__", "MODULE1 with @encoder preserved",
                   "Several named graphs can share a compilation-unit container.",
                   patch={"module":"existing builtin.module","region":"existing body region","block":"existing body block","sibling":"@encoder · preserved"},
                   ir="builtin.module MODULE1\n└─ body region Rmodule\n   └─ body block Bmodule\n      └─ mo.graph @encoder")
        else:
            s.step("Enter the native ModuleOp constructor", "ct_builtin_module",62,"loc=LOC1",
                   "mlir::ModuleOp::create(loc)", "builtin.module operation MODULE1",
                   "This native operation is an IR container, not a Python import module or a Mojo package.",
                   patch={"module":"builtin.module · loc=LOC1"})
            s.step("Expose its body region and block", "ct_builtin_module",69,"MODULE1",
                   "ModuleOp.body → &self.getBodyRegion().front()", "Rmodule → Bmodule; operations=[]",
                   "MLIR nesting is operation → region → block → contained operations.",
                   patch={"region":"body region Rmodule","block":"body block Bmodule · empty"},
                   ir="builtin.module MODULE1\n└─ body region Rmodule\n   └─ body block Bmodule\n      └─ no operations yet")
        s.step("Prepare the module-end insertion cursor", "graph_init",858,"Bmodule.end",
               "OpBuilder(self._module.body.end)", "builder points after existing definitions",
               "The graph definition belongs inside the module body block.",
               patch={"block":"Bmodule · insertion point=end"})
        s.step("Insert GraphOp as a child of the module", "graph_init",860,"builder + name + input types",
               "_mo.GraphOp(...)", "G1 is a new child; module identity remains MODULE1",
               "G1 owns a separate graph region/body; the module body does not contain the graph's input arguments directly.",
               patch={"graph":"mo.graph @fused_attention"},
               ir="builtin.module MODULE1\n└─ region Rmodule\n   └─ block Bmodule\n"+("      ├─ mo.graph @encoder · preserved\n" if reuse else "")+"      └─ G1: mo.graph @fused_attention\n         └─ R0 → B0(%q,%k,%v)")
        return s
    add("graph-module-setup","MLIR Module Reuse or create","ir",
        "What is the container, and where does GraphOp appear after insertion?",
        "module=None, loc=LOC1; or an existing Module wrapper","MODULE1, its body block, and a child graph G1",
        """**MLIR.module is IR container.** `builtin.module` is a native MLIR operation grouping definitions in a compilation unit. It owns a body region containing a body block; that block contains named graph operations.

The actual caller supplies no Module, so [ModuleOp::create](sources/max/python/max/_core/max_license/modules/dialects/builtin.cpp#L65) creates a fresh container. The reuse comparison keeps the same native container and any `@encoder` sibling. Each graph has its own region and arguments. **The graph's computation is nested inside G1, not directly in Bmodule.** The animation builds this nesting one level at a time.""",
        """caller: self._module = module.mlir_module if module else builtin.ModuleOp(location=loc) [LINK:graph_init:856]
├─ reuse branch: read Module.mlir_module → same enclosing operation [LINK:ct_graph_module:637] [CALLER:graph_init|856]
└─ creation branch: builtin.ModuleOp native binding [LINK:ct_builtin_module:62]
   ├─ resolve supplied/current loc [LINK:ct_builtin_module:64] [CALLER:graph_init|856]
   ├─ mlir::ModuleOp::create(loc) → MODULE1 [LINK:ct_builtin_module:65] [CALLER:graph_init|856]
   ├─ body binding → getBodyRegion().front() → Bmodule [LINK:ct_builtin_module:69] [CALLER:graph_init|858]
   └─ GraphOp inserted into Bmodule at end [LINK:graph_init:860] [CALLER:graph_init|860]""",
        modules(),alternatives={"reuse":modules(True)},source="graph_init",line=856)

    nodes=objects(("module","MODULE1.body","native block","module region"),
                  ("cursor","IP1","block + iterator","builder"),
                  ("builder","BUILDER1","mlir::OpBuilder","constructor"),
                  ("graph","G1","new operation","module body"),
                  ("body","B0","graph entry block","G1"),
                  ("call","CUST1","mo.custom","B0"))
    s=Story(nodes,view="insertion")
    s.step("Read the module's existing body block","ct_builtin_module",69,"MODULE1",
           "ModuleOp.body → &self.getBodyRegion().front()", "Bmodule · operations=[]",
           "A block is an ordered operation list; end is the position after its last operation.",
           patch={"module":"Bmodule []"},ir="Bmodule: [ |end ]")
    s.step("Create a block/iterator insertion point","ct_block_end",411,"Bmodule.end",
           "OpBuilder::InsertPoint(block, block->end())", "IP1=(Bmodule,end)",
           "The insertion point includes both the destination block and position.",
           patch={"cursor":"IP1=(Bmodule,end)"},ir="Bmodule: [ |IP1 ]")
    s.step("Construct the native builder from IP1","ct_op_builder",614,"IP1",
           "mlir::OpBuilder(ip.getBlock(), ip.getPoint())", "BUILDER1",
           "The builder creates new operations at the chosen position; it does not move all existing IR.",
           patch={"builder":"BUILDER1 → Bmodule/end"})
    s.step("Insert G1 in Bmodule","graph_init",860,"BUILDER1 + GraphOp arguments",
           "_mo.GraphOp(builder,loc,name,input_types,result_types)", "[ G1 |end ]",
           "Only the graph definition is inserted into this module block.",
           patch={"graph":"G1 @fused_attention"},ir="Bmodule: [ G1 |end ]\nG1.R0.B0: [ %q,%k,%v | empty operations ]")
    s.step("Retain B0 for later graph-body insertion","graph_init",875,"G1.regions[0].blocks[0]",
           "self._current_block = B0", "later operations target B0",
           "The module-end builder and the graph-body insertion point are different targets.",
           patch={"body":"B0 · current graph body"})
    s.step("Create a graph-body insertion scope","ct_insert_result",1312,"self._body=B0",
           "mlir.InsertionPoint(self._body); with ip,_location()", "IP2=(B0,end)",
           "Custom operations must consume B0 arguments inside their defining graph.",
           patch={"cursor":"IP2=(B0,end)"},ir="Bmodule: [ G1 ]\nG1.R0.B0: [ CH0 |IP2 ]")
    s.step("Record the custom call in the graph body","ct_insert_result",1320,"IP2 + %q,%k,%v",
           "op(*unwrapped_args,**unwrapped_kwargs)", "B0: [CH0, CUST1] ; result %o",
           "The native SSA operands point to existing inputs; the call creates one new result.",
           patch={"call":"%o=CUST1(%q,%k,%v)"},ir="Bmodule: [ G1 ]\nG1.R0.B0: [ CH0, %o=CUST1(%q,%k,%v) |end ]")
    add("graph-op-builder","OpBuilder · Insertion point","insertion",
        "Why are the module-end builder and the graph-body insertion point different?",
        "Existing body block and end iterator","G1 in Bmodule; CUST1 later in B0",
        """**An insertion point is a destination block plus a position.** The [native binding](sources/max/python/max/_core/max_license/_core.cpp#L408) makes IP1 from `Bmodule.end`; another binding constructs a C++ OpBuilder from it.

GraphOp is inserted at IP1. Once its B0 is retained, [the graph insertion helper](sources/max/python/max/graph/graph.py#L1312) uses a different MLIR InsertionPoint inside B0 to record custom/output operations. The animated cursor shows both lists. A cursor is compiler bookkeeping, not a runtime instruction pointer or tensor memory address.""",
        """caller: builder = OpBuilder(self._module.body.end) [LINK:graph_init:858]
├─ ModuleOp.body → native body block Bmodule [LINK:ct_builtin_module:69] [CALLER:graph_init|858]
├─ Block.end → InsertPoint(block,block.end()) [LINK:ct_block_end:411] [CALLER:graph_init|858]
├─ OpBuilder(ip.getBlock(),ip.getPoint()) [LINK:ct_op_builder:614] [CALLER:graph_init|858]
├─ _mo.GraphOp(builder,...) inserts G1 into Bmodule [LINK:graph_init:860] [CALLER:graph_init|860]
└─ later: _add_op_get_op_with_results [LINK:ct_insert_result:1284]
   ├─ ip = supplied _ip or InsertionPoint(self._body) [LINK:ct_insert_result:1312] [CALLER:custom_api|109]
   ├─ with ip,_location(): create graph-body op [LINK:ct_insert_result:1317] [CALLER:custom_api|109]
   └─ find inserted op and wrap result %o [LINK:ct_insert_result:1323] [CALLER:ct_insert_result|1320]""",
        s,source="graph_init",line=858)

    nodes=objects(("python","TQ · Python description","TensorType","caller"),
                  ("dim","DIM32","typed integer attribute","MLIR context"),
                  ("shape","SHAPE1","mosh.ShapeAttr","MLIR context"),
                  ("dtype","DTYPE1","element dtype","native type"),
                  ("device","DEVATTR1","m.DeviceRefAttr","native type"),
                  ("metadata","META1","DictionaryAttr","native type"),
                  ("native","T_NATIVE","mo.TensorType","MLIR context"))
    s=Story(nodes,view="type")
    s.step("Read the symbolic TensorType fields","ct_tensor_to_mlir",430,"TQ=(float32,[32,32],gpu:0,_layout=None)",
           "TensorType.to_mlir()", "construct fields for native MO tensor type",
           "This converts a type description, not a matrix or its bytes.",
           patch={"python":"float32; Shape(32,32); DeviceRef(gpu,0)"})
    s.step("Initialize optional Python metadata", "ct_tensor_to_mlir",436,"self._layout=None",
           "metadata=[]; skip layout append", "no Python layout entry",
           "Metadata setup happens before evaluating the native TensorType arguments.",patch={"metadata":"empty Python metadata list"})
    s.step("Convert each static size to a signed integer attribute","ct_static_dim_to_mlir",571,"StaticDim(32)",
           "IntegerAttr(si64,32) → kgen.CastFromBuiltinAttr", "DIM32 encodes the number 32 in the dimension-expression type",
           "The number is a shape constant. Signed i64 does not make the matrix elements integers.",
           patch={"dim":"32:si64 → cast to KGEN dimension expression"})
    s.step("Assemble the two-dimensional shape attribute","ct_shape_to_mlir",39,"DIM32,DIM32",
           "mosh.ShapeAttr([dim.to_mlir()...],mosh.ShapeType())", "SHAPE1=[32,32]",
           "Rank is two because the shape has two entries.",
           patch={"shape":"SHAPE1=[DIM32,DIM32]"})
    s.step("Keep the element dtype value","ct_tensor_to_mlir",441,"self.dtype=DType.float32",
           "pass dtype to mo.TensorType", "DTYPE1=float32",
           "dtype specifies how a later element is interpreted: e.g. 1.5 uses 32-bit floating-point storage.",
           patch={"dtype":"float32 · 4 bytes per later element"})
    s.step("Encode the device reference","ct_device_ref_to_mlir",183,"DeviceRef(kind=gpu,id=0)",
           "m.DeviceRefAttr(str(self.device_type),self.id)", "DEVATTR1=(gpu,0)",
           "This is logical placement metadata, not a CUDA/HIP device-context handle.",
           patch={"device":"DEVATTR1: kind=gpu, id=0"})
    s.step("Make the native metadata dictionary","ct_tensor_to_mlir",443,"_layout=None",
           "DictionaryAttr(metadata)", "META1={}",
           "No convolution filter-layout attribute is supplied here. Physical GPU fragment layouts are selected later.",
           patch={"metadata":"META1={} · no filter-layout entry"})
    s.step("Intern the native tensor type","ct_tensor_to_mlir",439,"SHAPE1,DTYPE1,DEVATTR1,META1",
           "mo.TensorType(shape,dtype,device,metadata=...)", "T_NATIVE=float32[32,32] @ gpu:0",
           "Equal field combinations can share an interned type; three graph values can have the same type.",
           patch={"native":T_GPU},
           mapping=[["shape","[32,32]","MOSH dimensions"],["dtype","float32","element interpretation"],["device","gpu:0","logical placement"],["metadata","{}","no filter layout"]])
    s.step("Convert Q, K and V types at GraphOp creation","graph_init",864,"input_types=[TQ,TK,TV]",
           "[t.to_mlir() for t in input_types]", "native input type list [T,T,T]",
           "Input positions remain distinct even if their native type handles compare equal.",
           patch={"python":"TQ/TK/TV → [T_NATIVE,T_NATIVE,T_NATIVE]"})
    add("graph-type-conversion","to_mlir() · Q/K/V type fields","type",
        "How do Python shape, dtype, device and metadata become one native type?",
        "TensorType(float32,[32,32],gpu:0,_layout=None)","An interned MO tensor type; no tensor data",
        """**32 as a shape dimension and float32 as an element dtype have different roles.** StaticDim(32) becomes a signed-i64 integer attribute, then a KGEN dimension-expression attribute. A pair of those forms MOSH ShapeAttr([32,32]); it describes dimensions rather than storing 1,024 float values.

`DType.float32` says a later element such as 1.5 is floating point. `DeviceRefAttr(gpu,0)` records placement. The [native type assembly](sources/max/python/max/graph/type.py#L439) combines these with a metadata dictionary.

**Attention's metadata is empty.** A [filter layout](sources/max/python/max/graph/type.py#L437) comparison could carry an HWIO/OIHW layout used by convolution dimension interpretation. That metadata is separate from the row-major TileTensor and TensorCore fragment layouts used later by this kernel.""",
        """caller: input_types=[t.to_mlir() for t in input_types] [LINK:graph_init:864]
└─ TensorType.to_mlir() [LINK:ct_tensor_to_mlir:430]
   ├─ metadata=[]; if _layout then append layout attribute [LINK:ct_tensor_to_mlir:436] [CALLER:graph_init|864]
   ├─ shape.to_mlir() [LINK:ct_shape_to_mlir:36]
   │  └─ StaticDim(32).to_mlir → IntegerAttr → CastFromBuiltinAttr [LINK:ct_static_dim_to_mlir:571] [CALLER:ct_shape_to_mlir|39]
   ├─ self.dtype → DType.float32 [LINK:ct_tensor_to_mlir:441] [CALLER:ct_tensor_to_mlir|439]
   ├─ device.to_mlir → DeviceRefAttr(gpu,0) [LINK:ct_device_ref_to_mlir:183] [CALLER:ct_tensor_to_mlir|442]
   └─ mo.TensorType(shape,dtype,device,DictionaryAttr(metadata)) [LINK:ct_tensor_to_mlir:439] [CALLER:graph_init|864]""",
        s,source="graph_init",line=864)

    nodes=objects(("name","sym_name","StringAttr","GraphOp state"),
                  ("types","TQ / TK / TV","native input types","GraphOp signature"),
                  ("fn","FUNCTION_TYPE1","FunctionType","GraphOp"),
                  ("sig","SIGNATURE1","KGEN generator type","GraphOp"),
                  ("params","shape declarations","ParamDecl arrays","GraphOp"),
                  ("region","R0","Region","G1"),
                  ("block","B0","Block","R0"),
                  ("args","%q / %k / %v","SSA block arguments","B0"))
    s=Story(nodes,view="ir")
    s.step("Record the graph symbol name","ct_graph_op_build",825,"name=fused_attention",
           "state.addAttribute(sym_name,builder.getStringAttr(name))", "G1.sym_name=fused_attention",
           "The compiled artifact uses the graph name to identify its model.",
           patch={"name":"@fused_attention","types":"[TQ,TK,TV] ; resultTypes=[]"})
    s.step("Create the caller/callee type contract","ct_graph_op_build",831,"inputTypes=[T,T,T], resultTypes=[]",
           "builder.getFunctionType(inputTypes,resultTypes)", "FUNCTION_TYPE1=(T,T,T)→()",
           "A signature records input/result count and native types before the body is populated.",
           patch={"fn":"(T,T,T) → ()"})
    s.step("Wrap and attach both signature attributes","ct_graph_op_build",832,"FUNCTION_TYPE1",
           "FuncTypeGeneratorType::get({},funcType); attach signature + functionType",
           "G1 has generator signature and ordinary function type",
           "KGEN uses the generator signature for parameterized type handling.",
           patch={"sig":"generator<(T,T,T)→()>"} )
    s.step("Discover native parameter declarations from shapes","ct_graph_op_build",839,"static input shapes",
           "getParamDeclsFromShapes(inputTypes/resultTypes)", "inputParameters=[]; resultParameters=[]",
           "A static dimension does not need a runtime symbolic name.",
           patch={"params":"empty input/result ParamDecl arrays"})
    s.step("Create the graph's new region","ct_graph_op_build",846,"OperationState without graph body",
           "state.addRegion()", "R0 owned by G1",
           "Region info is created by this builder; Python does not pass a preexisting region.",
           patch={"region":"R0 created"},ir="G1: mo.graph @fused_attention\n└─ region R0\n   └─ no blocks yet")
    s.step("Create ONE entry block in the region","ct_graph_op_build",847,"R0",
           "bodyRegion->emplaceBlock()", "B0 owned by R0",
           "One block can hold many arguments. There is not one block per input.",
           patch={"block":"B0 created"},ir="G1\n└─ R0\n   └─ B0() · empty operations")
    for i,name in enumerate(("q","k","v")):
        s.step("Append argument "+str(i)+": %"+name,"ct_graph_op_build",849,
               f"inputTypes[{i}]=T; state.location=LOC1", "body.addArgument(input,location)",
               f"B0 argument {i}=%{name}:T", "An argument is an SSA value available when execution begins, not allocated matrix storage.",
               patch={"args":", ".join("%"+v+":T" for v in ("q","k","v")[:i+1])},
               ir="G1\n└─ R0\n   └─ B0("+", ".join("%"+v+":T" for v in ("q","k","v")[:i+1])+")")
    add("graph-native-builder","GraphOp · Signature and body","ir",
        "Where does the region come from, and why are three arguments in one block?",
        "Builder, LOC1, name, three input types, empty result types","G1 with signature, R0 and B0(%q,%k,%v)",
        """**A region groups blocks; a block contains arguments and operations.** [GraphOp::build](sources/GraphCompiler/lib/MODialect/MOOps.cpp#L820) itself creates R0, then one B0, then adds one argument for each of the three input types. Python passes types and a location, not region data.

The function type initially says `(TQ,TK,TV)→()`. Both a functionType attribute and KGEN generator-signature attribute are attached. These form a caller/callee contract, not an execution result. `Graph.output` later declares one TO result. Runtime validation then compares three real input Buffers with this compiled contract.

The animation grows the region/block/argument structure incrementally. **%q, %k and %v share B0 but have distinct SSA identities.**""",
        """caller: _mo.GraphOp(builder,loc,name,input_types,result_types=[]) [LINK:graph_init:860]
└─ GraphOp::build(builder,state,name,inputTypes,resultTypes,isSubgraph) [LINK:ct_graph_op_build:820]
   ├─ attach sym_name; optional subgraph flag [LINK:ct_graph_op_build:825] [CALLER:graph_init|860]
   ├─ FunctionType=(TQ,TK,TV)→() [LINK:ct_graph_op_build:831] [CALLER:graph_init|864]
   ├─ generator signature + functionType attributes [LINK:ct_graph_op_build:832] [CALLER:graph_init|860]
   ├─ getParamDeclsFromShapes → input/result declarations [LINK:ct_graph_op_build:839] [CALLER:graph_init|860]
   ├─ state.addRegion() → R0 [LINK:ct_graph_op_build:846] [CALLER:graph_init|860]
   ├─ R0.emplaceBlock() → B0 [LINK:ct_graph_op_build:847] [CALLER:graph_init|860]
   └─ for input in inputTypes: B0.addArgument(input,state.location) [LINK:ct_graph_op_build:849] [CALLER:graph_init|864]""",
        s,source="graph_init",line=860)

    # Later cards use the same native source route but their own state/scene.
    result.extend(remaining_graph_lessons())
    return result


def remaining_graph_lessons():
    from compiler_attention_recording_lessons import recording_lessons
    return recording_lessons()
