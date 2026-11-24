from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('map_optimization','MAP fusion: access, legality and materialization',
 'See what may fuse around the Llama MLP and residual stream, and why mutable KV and opaque kernels constrain the answer.',
 'Compiler transformations',[
 A('PROD','Producer access / compute','MAP region','IR1','May be inlined into an eligible consumer if effects, shapes and contracts allow it.'),
 A('TEMP','Intermediate tensor','logical value / possible buffer','compiler lifetime plan','A producer-consumer boundary may require storage or may disappear through legal fusion.'),
 A('CONS','Consumer kernel','MAP iterator','IR1','Receives fused prologue or epilogue work only through its supported contract.'),
 A('GUARD','Legality constraints','compiler analysis','fusers / contracts','Tracks shared uses, mutation, indexing and output capabilities.'),
 A('STORE','Output store','epilogue','MAP compiler','Gets an explicit destination and, when needed, an accumulation buffer.'),
 A('OUT','Optimized MAP','IR representation','pass manager','Outlines structured bodies, moves invariant views and deduplicates captures.')],[
 E('Start from candidate producer/consumer work','Use SiLU(gate) multiplied by up as the elementwise example. This is a schematic eligible chain; the actual chosen implementation determines fusion.',{'PROD':B('candidate','gate → SiLU'),'TEMP':B('logical intermediate',('tmp','SiLU(gate) [T,11008]')),'CONS':B('candidate consumer','multiply with up')},[R('linear','class MLP'),R('fusion','createFuseElementwise')]),
 E('Run input fusion before output fusion','Each round performs elementwise, views and prologue fusion before reductions and epilogues. This exposes longer access/compute chains to consumers.',{'PROD':B('eligible work inlined'),'CONS':B('schematic fused elementwise','SiLU(gate) × up in one access/compute region'),'TEMP':B('eligible boundary eliminated')},[R('fusion','fusePm.addPass(createFuseElementwise())'),R('fusion','fusePm.addPass(createFusePrologue())')],transfer=['PROD','legal producer inlining','CONS']),
 E('Preserve a residual’s other live use','A value feeding both the attention branch and a later residual add cannot simply lose its necessary storage or ownership because one consumer fused.',{'GUARD':B('retain required uses','X → norm/attention','X → residual add')},[R('transformer','h = x + attn_out'),R('fusion','Fuses MAP kernels')],edges=[('GUARD','CONS')]),
 E('Do not erase cache mutation boundaries blindly','Optional mutable-load fusion must preserve read/write ordering and declared mutability. Paged MHA is an opaque implementation boundary.',{'GUARD':B('mutation / opaque contracts','KV writer must precede dependent read','opaque input/output fusibility checked')},[R('compiler','if (opts.fuseMutableLoadStore)'),R('mutable_fusion','FuseMutableLoads')]),
 E('Repeat rounds until no kernel is eliminated','Horizontal fusion runs after the producer-consumer fusers. The driver compares iterator counts and performs CSE between rounds.',{'OUT':B('fusion fixed point','elementwise → views → prologue → reduction','epilogue → reshape epilogue → horizontal','repeat while fused > 0')},[R('fusion','unsigned before = kernelCount()'),R('fusion','while (fused > 0)')]),
 E('Assign real output storage','AllocateAccumulatingBuffers gives fused epilogues a store pointer and any required pre-cast accumulation storage. Logical fusion does not eliminate the final result.',{'STORE':B('materialization explicit','output destination','optional accumulation buffer')},[R('compiler','createAllocateAccumulatingBuffers')],edges=[('CONS','STORE')]),
 E('Normalize views and captures before MOGG','Raise back surviving contiguous views, outline structured bodies, move invariant views and run CSE/canonicalization. Shared shape/stride captures can collapse.',{'OUT':B('ready for implementation calls','outlined bodies','deduplicated shape/stride captures')},[R('compiler','createRaiseBackContiguousViews'),R('compiler','createOutlineStructuredKernelBodies'),R('compiler','createLoopInvariantViewMotion')],edges=[('STORE','OUT')])],[
 Q('The actual fusion round and stop condition','''
```text
repeat:
  FuseElementwise
  FuseViews
  FusePrologue
  FuseReductions
  FuseEpilogue
  FuseReshapeEpilogue
  FuseHorizontal
  CSE
until this round eliminates no MAP iterator
```

The implementation counts iterators before and after a round. A successful fusion erases a kernel region, so unchanged count terminates the loop. These are compiler regions, not measured CUDA launch counts. A single region may call an implementation that submits more than one device operation.

Input-side fusion precedes output-side fusion so view/elementwise chains can reach an epilogue through an eligible consumer. Prologue fusion runs after view/elementwise expansion; horizontal sibling fusion runs last in the round.
''',[R('fusion','void runOnOperation() override'),R('fusion','while (fused > 0)')],open=True),
 Q('Three different meanings of “fused” in this model','''
| Meaning | Llama example | Where selected |
|---|---|---|
| architecture-level combined operation | rope + split + KV store | Python attention module explicitly selects one custom op |
| compatible weight/projection stacking | gate/up projection; QKV projection policy | module/linear construction and tracing |
| graph compiler fusion | eligible elementwise/view/prologue/epilogue regions | MAP fusers using contracts and legality |

A fused production attention implementation also performs internal tiled work. Calling all four cases “the compiler fuses attention” hides the actual transformation boundary. For this Llama route, RoPE/store and paged MHA are already explicit operation choices before MAP optimization.
''',[R('attention','fuse=self._fuse_rope_and_store'),R('linear','def _can_used_fused_mlp('),R('fusion','createFuseElementwise')],open=True),
 Q('Storage is governed by use and contract, not a diagram arrow','''
An intermediate with no remaining external use can lose its materialization when legal fusion replaces the load/store boundary. A residual input with another live consumer still has to exist. An opaque implementation only accepts fused accesses where its declaration permits them. Side effects and mutable aliases further restrict reordering.

After fusion, output epilogues receive an actual store target; mixed-precision accumulation can need a separate pre-cast buffer. Some unfused contiguous views are raised back because the next lowering either lacks a MAP form or can preserve a zero-copy view. Other layouts may require a real copy.
''',[R('compiler','createAllocateAccumulatingBuffers'),R('compiler','createRaiseBackContiguousViews'),R('mo_map','isFusedTensorType()')]),
 Q('What can be asserted for Llama without a compile dump?','''
The source proves that these passes are scheduled and that this model emits compatible semantic families: residual adds, MLP activations, reshapes and explicit cache operations. It does not establish a precise count of fused regions, eliminated allocations, launches or registers for an unspecified GPU.

To substantiate a claim such as “layer 5’s down projection and residual add fuse,” compare the model’s pre/post MAP dump under recorded options. The current diagrams show the contract and eligible mechanism, not that unobserved result.
''',[R('compiler','void M::MO::buildMapPipeline'),R('compiler','createDumpStagePass')])],'''
buildMapPipeline
├─ canonicalize / CSE
├─ FuseKernels → repeated ordered fusion rounds
├─ optional FuseMutableLoads → cleanup
├─ AllocateAccumulatingBuffers
├─ RaiseBackContiguousViews
├─ OutlineStructuredKernelBodies
└─ canonicalize → LoopInvariantViewMotion → CSE → canonicalize
''',[R('compiler','void M::MO::buildMapPipeline'),R('fusion','void runOnOperation() override')])

C('map_to_mogg','MAP → MOGG: implementation calls and shape arguments',
 'Follow pointers, dynamic dimensions, context and captured views into typed kernel routines.',
 'Compiler transformations',[
 A('MAP','MAP iterator','access/compute IR','IR1','Owns fused accesses, nested compute and store regions.'),
 A('PTR','Pointer operands','runtime-address placeholders','outlined routine signature','Represent input/output storage without embedding actual request addresses.'),
 A('DIM','Dynamic dimensions','routine parameters','lowering state','One parameter per unique dynamic dimension; shared dimensions are deduplicated.'),
 A('DESC','Tensor descriptors','shape/layout + pointer','MOGG routine','Reconstruct typed tensor access wrappers from pointer and dimension arguments.'),
 A('CALL','Implementation calls','MOGG IR','MojoRegistry / lowering interfaces','Carry registered kernel calls, foreach bodies or epilogue accesses.'),
 A('CTX','Device context argument','runtime reference','routine caller','Supplies device operations at execution; not a Python module object.')],[
 E('Visit each top-level opaque iterator','MAPToMOGG constructs a MojoRegistry and a worklist of map.iter.opaque operations, skipping nested traversal during collection.',{'MAP':B('worklist',('iter','map.iter.opaque: paged attention'))},[R('map_mogg','MOGG::MojoRegistry registry('),R('map_mogg','SmallVector<MAP::IterOpaqueOp> worklist')]),
 E('Form the pointer boundary','Lowering introduces pointer arguments for captured buffers and results. Addresses remain runtime inputs, while dtype/rank/layout information defines their interpretation.',{'PTR':B('signature pointers','Q pointer','KV pointer + lookup/control pointers','output pointer')},[R('map_mogg','!mogg.pointer')],edges=[('MAP','PTR')]),
 E('Deduplicate dynamic dimensions','Each unique dynamic dimension becomes a !mogg.dim argument. The same T referenced by input and output is not independently recomputed for every descriptor.',{'DIM':B('signature dimensions','T / B / page-dependent dimensions','one argument per unique dimension')},[R('map_mogg','one `!mogg.dim` block argument')],edges=[('MAP','DIM')]),
 E('Rebuild shapes and tensor access objects','mogg.shape.create and mogg.tensor.create reconstruct the tensor descriptors. Access/epilogue closures retain captures and layout relationships.',{'DESC':B('typed access wrappers','shape + pointer → tensor descriptor','captured views / epilogue target')},[R('map_mogg','mogg.shape.create'),R('map_mogg','mogg.tensor.create.epilogue')],edges=[('PTR','DESC'),('DIM','DESC')]),
 E('Lower each nested computation through its interface','map.call.opaque becomes mogg.call.opaque; an elementwise body becomes a foreach call. Unsupported lowerable operations produce a diagnostic.',{'CALL':B('implementation-bound calls',('call','mo.mha.ragged.paged execute contract')),'CTX':B('context argument','DeviceContext appended after dimensions')},[R('map_mogg','mogg.call.opaque'),R('map_state','MAPToMOGG: unsupported op')],edges=[('DESC','CALL'),('CTX','CALL')]),
 E('Handle remaining shape-function invocations','The pass lowers MO InvokeShapeFunc after iterator conversion, in an order that tolerates erasing the source operation.',{'CALL':B('routines + shape support',('call','typed implementation call'),'shape function calls as required'),'MAP':B('converted')},[R('map_mogg','lowerInvokeShapeFuncToMOGG')]),
 E('Carry the contract to call conformance','The next stage reconciles this routine’s operands with the exact registered execute signature before Mojo emission.',{'DESC':B('ready for signature adaptation','pointer / shape / mutability contracts')},[R('compiler','createConformCallOperands')],edges=[('CALL','DESC')])],[
 Q('The boundary changed: tensor value to pointer/shape recipe','''
```text
MAP view: captured Q[T,32,128] + output epilogue
                         ↓ MAPToMOGG
routine inputs (schematic):
    Q_ptr, KV_ptr, metadata_ptrs, output_ptr,
    unique dynamic dimensions,
    device context
routine body:
    shape.create(...) → tensor.create(pointer, shape, layout)
    call.opaque(registered implementation, descriptors, controls)
```

The representation contains no actual address of a future request’s Q buffer. Static heads=32 and head_dim=128 can inform specialization, while T and actual page addresses arrive with execution. These pointer arguments belong to an internal routine representation; they are not yet the final native AsyncValue ABI or the CUDA launch parameter packet.
''',[R('map_mogg','!mogg.pointer'),R('map_mogg','one `!mogg.dim` block argument')],open=True),
 Q('Nested lowering dispatch and registry role','''
`MAPToMOGGState.lowerOp` requires `LowerableToMOGGInterface`; unsupported operations fail explicitly. The registry supplies kernel declaration information used by implementation calls. It is not a runtime registry containing a token batch’s data.

The top-level iterator walk collects a worklist before mutation. Shape-function invocations are handled separately because MOToMAP left that form untouched. `enableComputeLambdas` is passed into the lowering state; its existence is a configurable code-generation choice, not proof of a particular generated closure in this model.
''',[R('map_state','MAPToMOGGState::lowerOp'),R('map_mogg','state.enableComputeLambdas'),R('map_mogg','lowerInvokeShapeFuncToMOGG')],open=True),
 Q('Two model lineages through the conversion','''
| Model operation | MAP representation | MOGG responsibility |
|---|---|---|
| residual add | explicit loads / elementwise body / output store | foreach/computation call with reconstructed accesses |
| paged attention | opaque call, buffer captures, effect ordering | registered attention call with typed Q/KV/control/context operands |
| reshape around attention | view/index relation or raised-back view | preserve legal shape/stride interpretation |
| runtime shape function | left as MO invocation until this pass | callable shape computation using the required argument representation |

MAP fusion can alter the first column’s producer/consumer grouping; it cannot invent missing shape/stride or mutable-access contracts. The pointer/descriptor reconstruction must match the selected implementation’s expected ranks and accesses.
''',[R('map_mogg','mogg.call.foreach'),R('conform','packAgainstSignature')]),
 Q('Separate four kinds of parameter','''
Compile-time specialization includes dtype, target and static layout/head dimensions. Runtime shape arguments include T or batch-dependent extents. Runtime tensor/control arguments include buffer pointers, offsets, page IDs and scale values. A DeviceContext is a runtime facility reference used by the callable.

Confusing these categories leads to incorrect explanations such as “every decode recompiles for its token IDs” or “a compile-time device attribute already contains a CUDA stream.” Later specialization and ABI packing preserve these distinctions.
''',[R('builtin','def _execute_mha_ragged_paged_scalar_args['),R('map_mogg','one `!mogg.dim` block argument')])],'''
MAPToMOGG.runOnOperation
├─ MojoRegistry
├─ collect top-level map.iter.opaque worklist
│  └─ MAPToMOGGState.lowerOp via LowerableToMOGGInterface
│     ├─ pointer arguments + unique dynamic dimensions + context
│     ├─ shape/tensor reconstruction
│     └─ nested opaque / foreach / epilogue calls
└─ lower remaining MO::InvokeShapeFuncOp
''',[R('map_mogg','void runOnOperation() override'),R('map_state','MAPToMOGGState::lowerOp')])

C('mogg_conformance','MOGG: exact signatures, mutability and outlined routines',
 'Reconcile generated call operands with real implementation declarations before turning routines into an execution plan.',
 'Compiler transformations',[
 A('CALL','Incoming MOGG call','IR call','MAP lowering','Has candidate input/output operands and callee identity.'),
 A('DECL','Registered execute signature','type contract','MojoRegistry / P1','Defines scalars, tensors, variadics, mutable access and trait requirements.'),
 A('PACK','Adapted operands','IR values','ConformCallOperands','Wraps scalar/tuple/trait operands and reconciles supported access conversions.'),
 A('OUT','Output contract','plain/fused form','call conformance','Selects plain execute only if every relevant fused output is a pass-through epilogue.'),
 A('ROUT','Outlined routine','function IR','compiler module','Gives repeated/callable computation a stable symbol and explicit arguments.'),
 A('TRACE','Trace wrapper','diagnostic scope','optional tracing infrastructure','Associates runtime/compiler diagnostics with callable work.')],[
 E('Resolve the callee declaration','The call’s symbol must resolve to a kernel declaration in MojoRegistry. A missing declaration is a compiler error, not a runtime GPU fallback.',{'CALL':B('candidate call',('call','paged MHA call')),'DECL':B('implementation signature','Q rank3; KV mutable rank6','uint32 offsets/layer; FP32 scale; context')},[R('conform','registry.lookupKernelBySymName'),R('builtin','def _execute_mha_ragged_paged_scalar_args[')],edges=[('DECL','CALL')]),
 E('Map operands against the exact signature','mapToKernelImplementation builds the operand/result mapping with variadic-size and by-value-result rules.',{'PACK':B('mapped contract','inputs → declared argument positions','outputs → declared result positions')},[R('conform','packAgainstSignature('),R('conform','mapToKernelImplementation(')],edges=[('CALL','PACK')]),
 E('Reconcile supported mutable-to-immutable use','For an allowed read-only kernel input backed by a mutable tensor producer, recreate the descriptor with immutable access; do not copy the whole tensor.',{'PACK':B('access adapted','cloneWithMut(false) where legal','original producer can become dead')},[R('conform','producer.cloneWithMut(builder, /*mut=*/false)')]),
 E('Pack scalar, tuple and trait forms','The lowering constructs the exact argument wrappers the execute contract expects. Device information selects the appropriate access/alignment flavor.',{'PACK':B('signature-ready operands',('packed','typed scalar / tuple / trait operands'))},[R('conform','packOperands('),R('conform','enclosingKernelDevice')]),
 E('Choose a plain output only for a pass-through epilogue','If every fused output has the required one-use pass-through form, selectUnfusedExecute replaces it with its store target and strips the fusion contract.',{'OUT':B('conditional output adaptation','all pass-through → plain execute','otherwise preserve fused form')},[R('conform','selectUnfusedExecute('),R('conform','out.hasOneUse()')],edges=[('OUT','PACK')]),
 E('Outline and wrap callable work','After rowwise legalization and call conformance, the pipeline outlines routines, wraps functions with trace support and runs module cleanup.',{'ROUT':B('outlined symbols',('routine','routine + explicit arguments')),'TRACE':B('associated trace scope','WrapFuncsWithTrace')},[R('compiler','void M::MO::buildMoggPipeline'),R('compiler','createOutlineRoutines')],edges=[('PACK','ROUT'),('ROUT','TRACE')])],[
 Q('Pipeline and failure points','''
```text
LegalizeRowwiseKernels
  → ConformCallOperands
  → OutlineRoutines
  → WrapFuncsWithTrace
  → module canonicalization / CSE
```

Conformance requires a valid callee declaration and compatible operand mapping. A tensor with the right element count but wrong rank, access kind or scalar/variadic form is not automatically a valid call. Diagnostics at this stage identify contract failures before device execution.

Outlining gives a body a function boundary; it does not prove the final compiler emits a distinct host symbol for every Llama layer. Later identical-function merging and JIT caching can reuse implementations where their effective specialization is equal.
''',[R('compiler','void M::MO::buildMoggPipeline'),R('conform','ConformCallOperands: no kernel decl')],open=True),
 Q('Mutability adaptation preserves storage and access intent','''
The source explicitly handles a fused mutable-load operand reconstructed as a mutable tensor. A read-only kernel input can accept its storage when the mapping permits mutable-to-immutable conversion. To make generated types agree, conformance clones the descriptor producer with `mut=false`; dead-code elimination can remove the old producer.

This is an access-wrapper transformation, not an allocation/copy of every KV byte. It also does not turn a write-capable KV parameter into immutable storage when the registered implementation requires mutation. Each operand is matched against its own contract.
''',[R('conform','allowMutToImmut'),R('conform','producer.cloneWithMut(builder, /*mut=*/false)')],open=True),
 Q('Fused output versus a plain execute overload','''
MOToMAP initially gives outputs a store epilogue. If no useful epilogue fusion remains, the call may be better represented by the implementation’s plain execute overload. `selectUnfusedExecute` accepts this only when every relevant output is a one-use pass-through epilogue that does not recompute a value.

It uses each epilogue’s destination capture as the plain output and updates the declaration’s output contract. If any output fails the check, the call retains its fused form. This all-or-nothing condition prevents discarding real epilogue work.
''',[R('conform','selectUnfusedExecute('),R('conform','epilogue.recomputesValue()'),R('conform','stripOutputFusion')]),
 Q('What Llama’s concrete signature contributes','''
The attention signature distinguishes an output rank-3 tensor, Q rank-3 tensor, offsets, mutable rank-6 KV, page-stride/length/lookup controls, layer index, scale, dispatch metadata and context. Static parameters such as target/mask/dtype are not inserted as arbitrary extra top-level model inputs.

The outer MOGG routine can contain pointer/shape reconstruction and adapted wrappers around that call. The next MOGG→MGP phase makes model initialization, execution, resources and readiness explicit. Host machine-code ABI and CUDA launch ABI still come later.
''',[R('builtin','def _execute_mha_ragged_paged_scalar_args['),R('compiler','createMOGGToMGP')])],'''
buildMoggPipeline
├─ LegalizeRowwiseKernels
├─ ConformCallOperands
│  ├─ lookup KernelDecl → mapToKernelImplementation
│  ├─ supported mutable/immutable descriptor reconciliation
│  ├─ scalar / tuple / trait packing
│  └─ optional all-pass-through unfused-output selection
├─ OutlineRoutines
├─ WrapFuncsWithTrace
└─ module cleanup → MOGGToMGP
''',[R('compiler','void M::MO::buildMoggPipeline'),R('conform','packAgainstSignature(')])
