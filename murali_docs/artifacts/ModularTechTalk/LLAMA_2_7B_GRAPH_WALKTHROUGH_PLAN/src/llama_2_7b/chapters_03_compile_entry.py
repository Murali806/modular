from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('compile_cache','Compilation entry, IR1 and the three cache levels',
 'What compile_reusing_mefs actually skips, what a native cache hit returns, and why C1 still needs initialization.',
 'Compiler transformations',[
 A('G1','G1 · caller graph','symbolic IR','GraphSpec / Python','Retains original graph while native compilation operates on a clone.'),
 A('STORE','Session MEF store','artifact policy','session or process default','Imports/exports artifacts using graph identity/signature policy.'),
 A('IR1','IR1 · compiler clone','mutable IR','native frontend','Carries imported declarations, options and graph transformations.'),
 A('CACHE','Native MEF cache','serialized bytes','compiler cache backend','Maps graph/options/environment identity to reusable compiled model bytes.'),
 A('JIT','Kernel caches','compiled routines','MOGG/Mojo compiler','Can reuse individual compiled code archives when whole-model compilation still proceeds.'),
 A('C1','C1 · CompiledModel','compiled artifact','compiled callable','Contains compiled resources and expected-weight contract, not a fully bound running model.')],[
 E('Trace once for this GraphSpec','compile_and_init_graphs preserves output_tree and calls compile_reusing_mefs on the traced decoder graph.',{'G1':B('ready',('g1','Graph("llama3") + weight contract'))},[R('spec','compiled = session.compile_reusing_mefs(')]),
 E('Choose the session store policy','An explicit store wins over the process default. No store or non-Graph input goes directly to compile; export compiles then records the artifact.',{'STORE':B('main route: no imported artifact','store absent / declines → compile')},[R('engine','def compile_reusing_mefs('),R('engine','artifact = store.claim_import(model)')],edges=[('G1','STORE')]),
 E('Seed declarations and capture expected weights','compile_async collects expected weight placement and seeds kernel declarations plus the opaque-type mapping from G1’s KernelLibrary.',{'G1':B('preserved caller graph',('g1','G1 + seeded declarations')),'IR1':B('about to clone','expected-weight metadata captured')},[R('engine','kernel_library._analysis.seed_kernel_decls('),R('engine','expected_weights = {')]),
 E('Clone the module for native compilation','MOFContext::compileToBinary clones the module. Register context, gather options and apply target/tile-mode choices before cache-key construction.',{'IR1':B('mutable compilation ownership',('ir1','IR1 clone + effective options'))},[R('frontend','ModuleOp::getFromOpaquePointer(source).clone()'),R('frontend','overrideLoweringOptions(gcPipelineOptions)')],transfer=['G1','clone operation structure','IR1'],edges=[('G1','IR1')]),
 E('Check the native model-IR cache','If enabled and usable, a keyed lookup can return MEF bytes. Disabled/unavailable caching takes the compile path; a miss runs convertMOToMEF.',{'CACHE':B('main route: miss','key(graph + options + environment)','no reusable MEF')},[R('frontend','computeMEFCacheKey('),R('frontend','mefCache->findKeyedSync(keyHash)')],edges=[('IR1','CACHE')]),
 E('Lower the whole graph; reuse kernels where possible','MO → MAP → MOGG → MGP builds the execution plan. Per-generator MOGG/Mojo cache hits can still avoid some kernel compilation.',{'IR1':B('lowered representations',('ir1','MO → MAP → MOGG → MGP')),'JIT':B('per-routine hit/miss','specialization cache lookups')},[R('compiler','gcPipelineImpl'),R('jit','checkCache(')],edges=[('IR1','JIT')]),
 E('Return completed C1, not M1','The synchronous compile wrapper waits on its compilation AsyncValue and raises errors. C1 is ready for init_all with actual weights.',{'C1':B('compiled',('c1','MEF + host/device resources')),'G1':B('still caller-owned',('g1','G1 source graph'))},[R('engine','handle.wait()'),R('engine','def init_all(')],edges=[('IR1','C1'),('JIT','C1')])],[
 Q('Three caches, three different skip boundaries','''
| Layer | Lookup unit | On a usable hit | Work still required |
|---|---|---|---|
| session/import MEF store | graph identity/signature policy | compile/import the selected artifact path | tracing/policy selection already occurred; init with weights; request execution |
| native model-IR → MEF cache | graph/options/environment key | return saved compiled MEF bytes | artifact loading, init, device resources, input execution |
| MOGG/Mojo routine caches | generator semantic identity and compilation environment | reuse compiled routine archive(s) | remaining graph lowering, misses, artifact assembly and init |

P1’s deterministic package location is not a fourth proven cache hit. Reusing one initialized M1 across tokens is runtime object reuse, not any of these compiler-cache lookups.

The import store can decline a graph and fall back to compile; a configured matching policy may also raise for missing/incompatible recorded artifacts. The exact store decision matters. A grouped multi-graph Module does not participate in the single-Graph store path in the same way.
''',[R('engine','def compile_reusing_mefs('),R('frontend','if (maybeMef)'),R('jit','checkCache(')],open=True),
 Q('Where G1 ends and mutable IR1 begins','''
Python captures graph names and expected weight devices, then seeds kernel declarations into the source module. Native `compileToBinary(void*, format)` clones the module and transforms that clone. The native comment identifies weights as context-owned dense-resource attributes, so cloning operation structure need not duplicate the full 7B weight payload.

Effective options are gathered before lowering and before relevant cache-key decisions. Seeded opaque mappings set `kernelDeclsPrePopulated`; the MO pipeline can skip package re-import/verification while still rewriting opaque types. This optimization is not permission to skip all graph validation.
''',[R('engine','expected_weights = {'),R('frontend','ModuleOp::getFromOpaquePointer(source).clone()'),R('frontend','kernelDeclsPrePopulated = true')],open=True),
 Q('Synchronous API, asynchronous native work','''
`compile_async` returns a CompiledModel wrapping pending native work. Its documented process-wide serialization can wait for a previous compile before scheduling the next. `compile` calls that API and waits; failures surface at the wait boundary. Engine::Context schedules a compile task which publishes success or an error into an AsyncValue.

The public type C1 does not tell you whether compilation has finished unless the API/wait state establishes it. Export/init also need to await a pending compilation. The chosen GraphSpec helper uses the synchronous path here; it is not a per-request just-in-time recompilation API.
''',[R('engine','def compile_async('),R('engine','handle.wait()'),R('engine_context','Engine::Context::compileModel(')]),
 Q('Failure and cache integrity','''
Invalid types/devices, unsupported implementation signatures and backend errors stop artifact creation. The pre-JIT debug route deliberately errors after emitting Mojo and before exporting a half-lowered MEF. Native model-cache insertion failure is ignored after successful compilation; a lookup/compilation failure has different semantics.

To verify a real model’s selected path, record effective encoding, target, source revision and compile options, then collect stage dumps or compiler logs with the supported debug configuration. This documentation does not claim those dumps exist; every IR representation shown here is explicitly schematic.
''',[R('frontend','if (options.preJit)'),R('frontend','Insertion failure is ignored.')])],'''
GraphSpec.trace → G1
└─ session.compile_reusing_mefs(G1)
   ├─ import store hit → compile(artifact path) → C1
   └─ compile(G1)
      ├─ compile_async → expected weights + seed kernel declarations
      ├─ native compileToBinary → clone IR1 → effective options
      │  ├─ native MEF cache hit → compiled bytes
      │  └─ convertMOToMEF → graph passes + routine caches → compiled bytes
      └─ wait compilation AsyncValue → C1
         └─ init_all(C1, registry) still required
''',[R('spec','compiled = session.compile_reusing_mefs('),R('engine','def compile_async('),R('frontend','MOFContext::compileToBinary(void *source')],scenarios=[dict(id='import',title='Imported MEF selected',steps=[
 E('Trace and select a matching imported artifact','The traced graph supplies the store’s identity/signature. Successful artifact selection skips this call’s Graph IR compilation.',{'G1':B('traced',('g1','decoder graph signature')),'STORE':B('artifact selected',('mef','precompiled decoder.mef'))},[R('engine','artifact = store.claim_import(model)')]),
 E('Import compiled bytes','compile(artifact) goes through the file/artifact loading path. No fresh MO→MAP model lowering is asserted for this route.',{'STORE':B('claimed'),'C1':B('loaded artifact',('mef','precompiled decoder.mef')),'IR1':B('graph compilation skipped'),'JIT':B('fresh model codegen skipped')},[R('engine','return self.compile(artifact)')],transfer=['STORE','compiled artifact import','C1']),
 E('Initialize with live weights and resources','Imported code still needs the current model’s weight binding, device resources and initialization contract before execution.',{'C1':B('ready for init',('mef','C1 imported artifact'))},[R('engine','def init_all(')])]),dict(id='native_hit',title='Native model-IR cache hit',steps=[
 E('Clone and key the graph','The native cache is checked after IR cloning, context registration and effective option resolution.',{'G1':B('retained',('g1','G1')),'IR1':B('cloned',('ir1','IR1 + options')),'CACHE':B('key computed',('bytes','cached MEF bytes'))},[R('frontend','computeMEFCacheKey(')]),
 E('Return cached MEF bytes','The hit returns the cached BufferRef instead of running convertMOToMEF. Per-kernel compilation is not visited on this successful full-model hit.',{'CACHE':B('hit'),'IR1':B('lowering skipped'),'JIT':B('not visited'),'C1':B('compiled artifact available',('bytes','MEF bytes → C1'))},[R('frontend','return std::move(*maybeMef)')],transfer=['CACHE','BufferRef','C1'])])])

C('mo_contracts','MO: contracts, shape inference and canonicalization',
 'The first compiler stage makes Llama operations well-formed and simplifies their relationships while preserving mutable KV semantics.',
 'Compiler transformations',[
 A('G1','Original G1','caller IR','Python Graph','Kept outside the mutable compiler work.'),
 A('IR1','IR1 · MO graph','mutable compiler IR','frontend/pass manager','Carries typed operations, placement, shape expressions and side effects.'),
 A('P1','Kernel declarations','operation contracts','KernelLibrary / packages','Describe parameter, tensor, context and mutability expectations.'),
 A('SHAPE','Shape expressions','compiler metadata','MO/MOSH/KGEN','Tracks static and symbolic dimensions and materialized shapes.'),
 A('ORDER','Chains / effects','dependency representation','graph compiler','Prevents incorrect reordering of mutable cache effects.'),
 A('NEXT','MAP boundary','next representation','pass manager','Receives stable-tensor marking and access/compute lowering.')],[
 E('Start from the cloned MO model','IR1 contains embedding, all 32 blocks, explicit rope/store and paged attention operations, weights, shapes and chains.',{'G1':B('preserved','G1'),'IR1':B('typed model graph',('ir1','MO: weights + decoder ops + output'))},[R('frontend','ModuleOp::getFromOpaquePointer(source).clone()')]),
 E('Import only if declarations were not prepopulated','Otherwise skip ImportPackages/VerifyKernels, but always run RewriteOpaqueTypes and VerifyParameters.',{'P1':B('contracts resolved','kernel declarations','opaque-type mapping'),'IR1':B('contracts checked',('ir1','MO + resolved opaque types'))},[R('compiler','if (!opts.kernelDeclsPrePopulated)'),R('compiler','createRewriteOpaqueTypes'),R('compiler','createVerifyParameters')],edges=[('P1','IR1')]),
 E('Canonicalize and simplify chains','CSE/canonicalization and chain sequence cleanup remove equivalent pure structure while preserving side-effect ordering.',{'ORDER':B('ordering retained','rope/store → cache readers'),'IR1':B('canonicalized',('ir1','deduplicated pure MO expressions'))},[R('compiler','createCanonicalizeChainSequences')],edges=[('IR1','ORDER')]),
 E('Infer/fold shapes under actual options','Shape inference and constant folding are conditional. T stays dynamic where the batch controls it; known model dimensions can simplify expressions.',{'SHAPE':B('shape relationships','hidden=4096; heads×dim=32×128','T unresolved until execution','B+1 offsets')},[R('compiler','if (opts.useShapeInference)'),R('compiler','if (opts.constantFold)')],edges=[('IR1','SHAPE')]),
 E('Hoist constants and merge shape materializations','Symbolic optimization, fallback shape functions and duplicate-materialization cleanup organize host/control computation without assuming token values are constants.',{'SHAPE':B('shared shape work','unique materializations','fallback shape functions as required')},[R('compiler','createHoistConstantSubgraphsIntoParent'),R('compiler','createMergeDuplicateShapeMaterializations'),R('compiler','createAddFallbackShapeFunctions')]),
 E('Infer layouts and conditionally run patterns','InferLayouts always appears here. Pattern fusion/shape packing is under usePatterns. The model already names paged attention explicitly.',{'IR1':B('layout / target contract',('ir1','MO + inferred layout / validated device'))},[R('compiler','createInferLayouts'),R('compiler','if (opts.usePatterns)'),R('compiler','createValidateDevicesPass')]),
 E('Finish MO and mark stable tensors','Parallel cleanup, optional redundant-mutability cleanup, hoisted parameters and unused-op removal precede MarkStableTensors and MOToMAP.',{'NEXT':B('ready for access lowering','stable tensors marked','MOToMAP next')},[R('compiler','createHoistParameterExpressions'),R('compiler','createMarkStableTensorsPass')],edges=[('IR1','NEXT')])],[
 Q('Complete MO pass ledger, in source order','''
| Sequence | Pass/action | Predicate or Llama consequence |
|---|---|---|
| 1 | ImportPackages; VerifyKernels | only if declarations were not prepopulated |
| 2 | RewriteOpaqueTypes; VerifyParameters | required type/parameter normalization |
| 3 | DebugPrintAllTensors | debug option only |
| 4 | LegalizeRMOOps; CSE/canonicalization; CanonicalizeChainSequences | normalize operations and ordering |
| 5 | shape inference pipeline | `useShapeInference` |
| 6 | ConstantFold | `constantFold` |
| 7 | HoistConstantSubgraphsIntoParent; CSE; SymbolicOptimization; canonicalization/CSE | constants and symbolic expressions |
| 8 | MergeDuplicateShapeMaterializations; AddFallbackShapeFunctions; SymbolicOptimization; cleanup | shared/required shape work |
| 9 | ConstantFold; InferLayouts | folding conditional; layout inference follows |
| 10 | PatternFusion; InvokeShapePackingFunctions; PropagateShapes; cleanup | `usePatterns` |
| 11 | ValidateDevices; cleanup | actual target/placement validity |
| 12 | CreateParallelFromUnbundle; cleanup; MergeAdjacentParallels | relevant parallel regions if present |
| 13 | ConstantFold; ElideRedundantMutableOps | respective `constantFold` / `fuseJitKernels` options |
| 14 | HoistParameterExpressions; canonicalization; RemoveUnusedOps | final MO cleanup |
| 15 | MarkStableTensors; MOToMAP | stage transition in gcPipelineImpl |

A scheduled pass can find nothing applicable. The table is the source pipeline, not a claim that each pass rewrote this model. Options controlling fast-math/no-undefined-math inform symbolic optimization; they are not silently selected here.
''',[R('compiler','void M::MO::buildMoPipeline'),R('compiler','createMarkStableTensorsPass')],open=True),
 Q('Trace two actual model operations through the contracts','''
**Residual add:** two `[T,4096]` values with compatible dtype/placement yield another `[T,4096]`. Shape propagation can carry T through without learning any token’s value. Pure duplicate shape expressions can be shared.

**RoPE/store and paged MHA:** typed custom-op declarations bind query/output ranks, mutable KV storage, uint32 offsets/layer index, host control values and DeviceContext. Cache mutation has observable effects beyond an ordinary fresh tensor output. Chains/effect information must survive normalization so the later reader sees newly written K/V.

The numerical tensor program and its effect/shape program coexist. Reducing redundant ordering edges must preserve required producer/consumer relationships. Neither CSE nor pattern fusion is a license to treat stateful KV writes as interchangeable pure math.
''',[R('transformer','h = x + attn_out'),R('kernels','mo.rope_split_store.ragged.paged'),R('mo_map','getReadWriteEffects')],open=True),
 Q('Shape functions, symbolic expressions and constants','''
Static model values can fold: 32×128=4096, compatible reshape products, fixed layer/head information. Batch quantities such as T, B, page counts and request lengths can remain symbolic or runtime-dependent. Shape functions cover operations whose result sizes cannot be established solely by generic propagation.

Hoisting/merging shape materializations avoids repeating equivalent control computation. It does not hoist request-specific token contents into model initialization. Weight-backed dense-resource data, scalar constants and dynamic activation tensors have different representation and lifetime rules.
''',[R('compiler','createMergeDuplicateShapeMaterializations'),R('compiler','createAddFallbackShapeFunctions'),R('graph','si64 = kgen.SIMDType')]),
 Q('Why an attention-pattern story would be misleading here','''
The Python model emits `mo.mha.ragged.paged` directly through `ops.inplace_custom`. The compiler already has a production attention boundary. A conditional generic attention pattern matcher may run over the graph, but this source route does not prove that it recognized `QKᵀ → softmax → V` and invented the paged operator.

The built-in implementation can contain internal fusion and hardware-specific kernels independently of graph-level MAP fusion. Keep high-level custom-op selection, graph fusion and low-level instruction optimization as separate transformations.
''',[R('kernels','op_name = "mo.mha.ragged.paged"'),R('compiler','createPatternFusion')])],'''
compileToBinary → IR1 (MO)
└─ buildMoPipeline(options)
   ├─ optional package import → opaque/parameter verification
   ├─ canonicalization / CSE / chains
   ├─ conditional shapes + folding → symbolic optimization
   ├─ shape materializations / fallback functions / layouts
   ├─ conditional patterns → validate devices
   └─ mutable/parameter/unused cleanup
      └─ MarkStableTensors → MOToMAP
''',[R('compiler','void M::MO::buildMoPipeline'),R('compiler','gcPipelineImpl')])

C('mo_to_map','MO → MAP: separate tensor access from computation',
 'Turn a residual expression and a paged attention call into explicit input access, kernel work and output storage contracts.',
 'Compiler transformations',[
 A('MO','MO operations','tensor semantics','IR1','Describe what model tensor operations compute and which effects they have.'),
 A('VIEW','Access/view program','index transformations','MAP regions','Describes loads, reshapes, slicing/broadcast indexing and captured tensors.'),
 A('COMPUTE','Compute region','scalar or opaque work','MAP iterator','Contains elementwise computation or a registered opaque implementation call.'),
 A('STORE','Output epilogue','store contract','MAP iterator','Captures output storage and index/value arguments.'),
 A('CHAIN','Effect dependency','symbolic chain','enclosing MAP operation','Threads mutation ordering outside the implementation’s data argument body.'),
 A('MAP','MAP iterator','transformed IR','IR1','Keeps access, computation, output storage, dispatch and effects available for fusion.')],[
 E('Choose the source operation','Track a residual add on [T,4096], and compare it with the opaque paged MHA call. Both are model operations; they lower through different forms.',{'MO':B('source semantics',('add','add(X,attention_out) [T,4096]'),('mha','mo.mha.ragged.paged'))},[R('transformer','h = x + attn_out'),R('kernels','op_name = "mo.mha.ragged.paged"')]),
 E('Check whether MOToMAP lowers this operation','Preserved operations and InvokeShapeFunc remain for other handling; rank limits also constrain eligibility.',{'MAP':B('lowering eligibility','shouldLower(op)','preserved / rank checks')},[R('mo_map','bool shouldLower(Operation *op)'),R('mo_map','static bool hasLowerableRank')]),
 E('Build load prologues','The access region captures the source tensor and accepts an index; MAP Load and Yield provide values to computation.',{'VIEW':B('loads made explicit',('loads','index i → load X[i], load A[i]'))},[R('mo_map','MAP::IterPrologueOp buildLoadPrologue'),R('mo_map','MAP::LoadOp::create')],edges=[('MO','VIEW')]),
 E('Represent the calculation','Elementwise residual addition exposes scalar computation. Paged attention stays an opaque implementation call with only declared fusible operand/result capabilities.',{'COMPUTE':B('compute represented',('add','scalar x+a'),('mha','opaque attention call + contract')),'MO':B('lowered source operations')},[R('mo_map','lowerToOpaque('),R('mo_map','isFusedTensorType()')],transfer=['MO','operation lowering','COMPUTE'],edges=[('VIEW','COMPUTE')]),
 E('Build a store epilogue','An output epilogue receives value and index, captures the destination tensor, and stores. Its intermediate accumulation tensor is filled later if needed.',{'STORE':B('output access explicit','(value,index,destination) → Store','intermediateTensor initially null')},[R('mo_map','MAP::IterEpilogueOp buildStoreEpilogue'),R('mo_map','intermediateTensor=*/Value()')],edges=[('COMPUTE','STORE')]),
 E('Carry side effects and dispatch','For a mutating custom op, the last chain operand/result is carried by map.iter.opaque outside its body. Read/write effects and immediate/deferred launch policy are stamped.',{'CHAIN':B('cache ordering preserved','inputChain → trailing chain result'),'MAP':B('effects and placement','read/write flags','device ref + dispatch mode')},[R('mo_map','A mutating op\'s chain'),R('mo_map','stampAsyncLaunch('),R('mo_map','getReadWriteEffects')],edges=[('CHAIN','MAP')]),
 E('Expose the whole access/compute boundary to fusion','MAP now has the regions and contracts needed for legal fusion. A view may alter indexing without copying tensor contents.',{'MAP':B('ready for MAP optimization','load/access → compute → store','opaque contracts + effects retained')},[R('compiler','void M::MO::buildMapPipeline')],edges=[('VIEW','MAP'),('STORE','MAP')])],[
 Q('Schematic before/after: a residual add','''
```text
MO semantic view:
    H[T,4096] = add(X[T,4096], A[T,4096])

MAP schematic view (not captured model IR):
    inputs / captures: X, A
    iteration index: i = (token, feature)
    prologue X: load X[i]
    prologue A: load A[i]
    compute: x + a
    epilogue: store H[i]
```

Splitting access from computation permits later fusion to replace a load with a producer or fold a view into indexing. The resulting physical allocation/launch decisions belong to later stages. A scalar expression exposed in MAP is not proof of a scalar-only GPU kernel; backend vectorization/tiling is downstream.
''',[R('mo_map','buildLoadPrologue'),R('mo_map','buildStoreEpilogue')],open=True),
 Q('Opaque production attention retains an implementation boundary','''
The pass uses kernel declarations to identify fusible input/output forms, builds a `map.iter.opaque`, preserves the operation’s device and fusion ancestry, and carries read/write effects. Its nested call does not make arbitrary production attention internals visible as elementwise MAP arithmetic.

For mutation, the chain is separated from non-chain operands and returned as the trailing dependency result. The implementation’s `execute` signature consumes tensor/control/context arguments rather than that compiler chain as an ordinary Q/K/V argument. Later runtime lowering converts the ordering relationship into execution dependencies.
''',[R('mo_map','lowerToOpaque('),R('mo_map','ValueRange nonChainOperands'),R('mo_map','opaque.setHasReadEffect')],open=True),
 Q('Views, rank limits and operation preservation','''
Reshape/slice/broadcast/transpose access relationships may become index/view transformations. Whether a zero-copy view survives or a real store is needed depends on contiguity, strides, consumers and implementation requirements. A transpose is not universally free, and a reshape is not universally a GPU launch.

`shouldLower` excludes operations marked preserved during kernel lowering and shape-function invocation, and checks representable tensor ranks. The runtime limit is `kMaxRank = 8`; this Llama's ordinary KV buffer has rank 6, Q has rank 3 and the residual stream has rank 2. Unknown rank passes the static rank predicate; that does not waive eventual runtime contract checks.

The operation walk explicitly diagnoses an MO tensor rank above the limit and signals pass failure. It does not quietly leave an invalid rank for MGP to assert on later. Remaining valid MO control/shape operations can be lowered by later stages. “MO → MAP” names a conversion phase, not a promise that every surviving operation immediately belongs to the MAP dialect.
''',[R('mo_map','bool shouldLower(Operation *op)'),R('mo_map','the graph cannot be lowered'),R('tensor_shape','constexpr size_t kMaxRank = 8'),R('compiler','createRaiseBackContiguousViews')]),
 Q('What remains invariant across this conversion','''
Input/output tensor semantics, static/symbolic shape relations, placement and required mutable ordering must survive. The access program may change representation, but the cache page written for request A/layer 5 must still be the page read by its corresponding attention. Runtime page IDs remain input values; lowering does not hard-code the illustrative physical page 7.

The stable-tensor pass runs immediately before conversion because it needs MO operations. Stability describes a compiler-recognized property, not a declaration that request-local activations or KV contents never change.
''',[R('compiler','createMarkStableTensorsPass'),R('mo_map','getReadWriteEffects')])],'''
MOToMAP.runOnOperation
├─ shouldLower: dialect / preserved-op / rank eligibility
├─ infer implementation operand / parameter mapping
├─ accesses → prologues / views / loads
├─ compute → elementwise body OR opaque implementation call
├─ outputs → store epilogues
└─ carry device, dispatch, effects and chain → MAP iterator
''',[R('mo_map','bool shouldLower(Operation *op)'),R('mo_map','lowerToOpaque('),R('mo_map','buildStoreEpilogue')])
