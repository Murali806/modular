"""Examples and why for each C.1 pass group; fragments are explicitly schematic."""
LESSONS = [
 dict(id='route', title='01 · PassManager and compiler clone C1', source='gc', line=638,
 why='The caller must retain G1 while compiler transformations rewrite a separate module. A pass manager provides ordered transformations, nested scopes, verification and failure propagation.',
 before=['G1: verified MO graph','P1: prepared implementation package'], after=['G1 retained by Python caller','C1: compiler clone','PassManager schedules and runs passes'],
 example='''G1.module retained
  → frontend clones module → C1
  → cache miss → convertMOToMEF(C1, options)
  → runGCPipeline → gcPipelineImpl adds passes
  → pm.run(C1) executes that pass list
  → successful lowered plan + compiled code → export MEF

Options are inputs, not measured stage timings.
Graph-artifact hit bypasses new graph lowering/JIT, but still needs initialization.''',
 detail='MOFContext::compileToBinary owns the frontend route. Constructing a pass list does not execute it. pm.run applies that list to C1; a failure stops the route before a usable artifact is returned. A dialect is an IR vocabulary, not a required new disk file or process.'),
 dict(id='mo',title='02 · buildMoPipeline: verify and simplify MO',source='gc',line=219,
 why='Preserve and check high-level intent before exposing implementation details: inputs/results, shapes, devices, parameter legality and kernel declarations must agree.',
 before=['C1 mo.graph @addition','custom symbol=add_one','X/Y: float32[5,10], cpu:0'],after=['Verified types/shapes/devices','Parameters checked; RMO legalized','Simplified MO; declarations retained'],
 example='''Graph contract: @addition(X:T) → Y:T
T = tensor(float32, shape=[5,10], device=cpu:0)
CUST1 contract: name=add_one, one tensor input, one declared output
P1 declaration: AddOne.execute(output, x, ctx), target parameter

Wrong kernel name / unsupported argument contract → diagnostic, not codegen
Shape [5,10] = 50 logical elements; no input values are present yet
CPU/GPU placement labels are metadata checked against supported targets

kernelDeclsPrePopulated=true:
  skip ImportPackages + VerifyKernels in this pass group
  still RewriteOpaqueTypes + VerifyParameters + other scheduled passes
false:
  import package declarations, verify kernels, then continue

SIMPLIFY MO · BEFORE / AFTER EXAMPLES
All fragments below are schematic; source tests are linked below.

A · Canonicalization: remove identity transpose
 before: %t = mo.transpose(%X, permutation=[0,1]); output %t
  after: output %X
For X=[[1,2],[3,4]], permutation [0,1] preserves every element.
Permutation [1,0] is a real transpose and cannot be removed this way.
Why: remove a no-op definition and expose the original value to later passes.
Pass: GraphOp canonicalizer uses the MO operation's folding rules.

B · CSE: share identical eligible definitions
 before: %a = constant([0,1], si64)
         %b = constant([0,1], si64)
         transpose(X,%a); transpose(Z,%b)
  after: %p = constant([0,1], si64)
         both consumers reference %p
Why: retain one equivalent IR definition instead of two.
Same constant contents/types in a scope where reuse dominates both uses;
this does not merge X and Z or permit merging arbitrary side effects.
Pass: Common Subexpression Elimination (CSE), scheduled on GraphOp.

C · Symbolic algebra: eliminate addition of zero when permitted
 before: %zero = constant([0,0,...], type=T)
         %Y = mo.add(%X,%zero); output %Y
  after: output %X
Example ordinary values: [2,3]+[0,0] = [2,3].
Source test enables symbolic-ffast-math=true and
symbolic-no-undefined-math=true; floating-point rewrites must obey policy.
Signed zero / exceptional values mean this is not an unconditional
floating-point identity under all semantics.
Nonzero comparison: [2,3]+[0,1] stays an addition → [2,4].
Why: avoid an unnecessary arithmetic operation and its constant input.
Pass: SymbolicOptimization, separate from evaluating constant-only graphs.

D · Duplicate shape materialization: remap parameters then merge
 before: broadcast_to(value,shape) defines D1 → A
         broadcast_to(same value,same shape) defines D2 → B
         rebind A to N; rebind B to N; select(mask,A,B)
  after: one broadcast defines D1; remap D2 references to D1
         equivalent rebinds can then fold/share during CSE/canonicalization
Example: scalar 7 broadcast to N=5 → [7,7,7,7,7] on either branch.
Why: eliminate repeated shape-dependent work and duplicate dimension names.
Plain CSE cannot initially merge definitions with distinct parameter refs.
Pass: MergeDuplicateShapeMaterializations checks name/operands and
parameter-definition ownership before remapping and erasing duplicates.

Original add-one:
  mo.custom add_one(X) remains the requested computation X+1.
  The examples above illustrate opportunities in other MO graphs;
  they do not claim AddOne's opaque Mojo body is rewritten as X+0.
Unused results also do not automatically make an operation safe to erase;
side effects and pass-specific eligibility still matter.''',
 detail='Python seeds reusable declarations before the frontend clone; it has already checked custom calls during recording. Skipping redundant import does not skip the whole MO pipeline. Other scheduled work includes CSE, canonicalization, optional shape inference/constant folding, layout inference, device validation and unused-op cleanup. Their effect depends on options and the graph; add-one has no matrix constants to evaluate at compile time.'),
 dict(id='map',title='03 · MOToMAP: accesses versus computation',source='opaque',line=38,
 why='Expose iteration, reads, writes and eligible computation boundaries so fusion can reason about producer/consumer relationships while preserving kernel contracts.',
 before=['MO tensor call / operation','Whole-tensor result contract'],after=['MAP iteration + operand mapping','Access: read/write index or raw output','Compute: registered call inside iteration'],
 example='''ADD-ONE ANCESTRY (schematic, advanced fusion route)
mo.custom add_one(X) → Y
  → map.iter.opaque inputs(X) outputs(Y) {
       map.call.opaque AddOne inputs(X) outputs(Y)
     }
Plain OutputTensor → raw output; no automatic fused store epilogue.
The implementation still performs x.load + 1 in Mojo.

ELEMENTWISE COMPARISON (source test: mo.add)
mo.add(X,Z) → Y
  → iteration over indices i
       a = map.load X[i]       ACCESS
       b = map.load Z[i]       ACCESS
       c = call Add(a,b)       COMPUTATION
       map.store Y[i] = c      ACCESS
At i=(1,2), X=3 and Z=4 would produce 7 at execution.
Those numbers illustrate behavior; compilation manipulates typed IR.''',
 detail='The opaque-kernels test proves plain outputs stay raw and fusible outputs gain map.iter.epilogue/map.store. The separate elementwise-add test proves loads, Add call and store. Applying the opaque pattern to add_one is a schematic ancestry example, not a captured add-one compiler dump. The test also has a disabled route where custom operations survive; this detailed MAP route is conditional.'),
 dict(id='map-opt',title='04 · buildMapPipeline: fuse and simplify accesses',source='fuse',line=35,
 why='Avoid materializing eligible intermediate tensors and repeated loop-invariant metadata; reduce redundant captures without changing the computation.',
 before=['Kernel A produces TMP','Kernel B reads TMP','Duplicate view/stride captures'],after=['Eligible combined kernel body','TMP access pair removed','Shared invariant views/captures'],
 example='''ELIGIBLE ELEMENTWISE COMPARISON
before: TMP[i] = X[i]+1; Y[i] = TMP[i]*2
  two iterations, intermediate TMP store + load
 after: a=load X[i]; b=(a+1)*2; store Y[i]=b
  one combined iteration, no TMP store/load
For a=3: before and after both produce 8.

PER ELEMENT: TMP store+load = 2 avoidable scalar accesses
For 50 elements: 100 logical intermediate accesses avoided
Teaching arithmetic; excludes X/Y accesses and measured memory traffic.

Loop-invariant view:
  repeated stride(X) captures → hoist shared stride computation
  CSE unifies identical stride ops → deduplicate closure captures''',
 detail='FuseKernels schedules elementwise, view, prologue, reduction, epilogue, reshape-epilogue and horizontal fusion. It repeats rounds until kernel count stops decreasing, with CSE between rounds. buildMapPipeline then allocates accumulating buffers, raises suitable contiguous views back, outlines structured bodies and moves invariant views. Eligibility depends on kernel signatures, effects and supported patterns. Plain add_one alone has no second consumer to fuse; do not promise a fused body for it.'),
 dict(id='mogg',title='05 · MAPToMOGG: registry + per-kernel state',source='state',line=32,
 why='Connect generic MAP structure to the actual registered implementation and build the pointer, shape and device-context interface needed to call it.',
 before=['MAP input/output block args','Registration string + device','Iteration body operations'],after=['MOGG routine with pointer args','Reconstructed tensor descriptions','Mapped operations + call glue'],
 example='''MojoRegistry R1 (schematic contents)
  kernelByName["add_one"] = [KernelDecl(AddOne, execute signature)]
  kernelBySymName[qualified_AddOne_symbol] = that declaration
  shapeFuncByName["add_one"] = shape-function declaration
  primitiveByName[...] = imported primitive declarations
  deviceInfoMapping[cpu:0] = host target metadata
Actual symbol mangling and declarations depend on package loading.

MAPToMOGGState K1 (rebuilt per map.iter.opaque)
  registry → R1; builder → current insertion point
  device=cpu:0; enableComputeLambdas=pass option
  mapping: MAP X argument → reconstructed input tensor
           MAP Y argument → reconstructed output tensor
  dimArgs={} for static [5,10]; contextArgs=[ctx0]
  indexShapeArgs={} when no dynamic reshape/slice index needs captures

Routine ABI: input_ptr, output_ptr, [dynamic dims], ctx
  pointers + shape/dtype descriptions → tensor handles
  call selected AddOne with (output, input, ctx), target from device
GLUE = mapping values, reconstructing types, passing contexts,
       creating calls, and preserving effects/ordering.''',
 detail='MAPToMOGGPass constructs MojoRegistry from the module, collects IterOpaqueOps, then creates a fresh MAPToMOGGState for each. IterOpaqueOp::lowerToMOGG creates a mogg.routine, pointer block arguments for tensor I/O, unique dynamic dimensions and context arguments. State.lowerOp dispatches each operation through its lowering interface. Registry selection filters device-compatible declarations and ranks user-defined/more-specific matches. This registry is a C++ index of declarations, not the LLVM backend registry, a Python dictionary, or a kernel binary cache.'),
 dict(id='mogg-opt',title='06 · buildMoggPipeline: callable interfaces',source='gc',line=370,
 why='Make the reconstructed calls conform to their registered signature and turn routine regions into source-emittable function definitions.',
 before=['MOGG routine regions','Calls with generic operands','Potential context tuples'],after=['Conformed call operand order','Outlined function + invocation','Tracing and cleanup'],
 example='''Before (schematic): routine(input_ptr, output_ptr, ctx)
  reconstruct input X and output Y
  call AddOne with mapped values

ConformCallOperands: match declaration order and parameter layout
  AddOne.execute(output=Y, x=X, ctx=ctx)
OutlineRoutines: routine body → named MOGG function K0
  graph retains an invocation referring to K0
WrapFuncsWithTrace: attach supported trace wrapping
canonicalization/CSE: simplify redundant structure

Distributed comparison: separate ctx0,ctx1 → tuple expected
by a DeviceContextArray signature, when that signature applies.''',
 detail='LegalizeRowwiseKernels is also scheduled, but plain add-one is not a rowwise reduction. Function outlining changes IR organization; it does not emit target instructions. Trace wrapping follows implementation policy; it is not evidence that this host recorded a profile.'),
 dict(id='mgp',title='07 · MOGGToMGP: init, execute, buffers and chains',source='mgp-convert',line=295,
 why='Turn a named tensor computation into a model with explicit runtime setup and per-call work, separating reusable resources from actual inputs and ordering dependencies.',
 before=['mo.graph tensor signature','MOGG implementation calls'],after=['mgp.model init region','execute region with buffers','state slots + dependency chains'],
 example='''mgp.model @addition (schematic structure)
  init:
    establish device runtime / buffer builders
    prepare shared context and persistent resources
    return initial chain + context + state
  execute(chain, context, weights, device state, X):
    acquire output storage Y for 50 float32 elements
    call K0 with X/Y/context after prerequisites
    return completion chain, Y

50 × 4 bytes = 200 payload bytes for one logical tensor
Actual allocation capacity/alignment may be larger.
slot s0 → runtime handle (illustration, not captured slot index)
chain CH1 → ordering dependency, not matrix data or a mutex.''',
 detail='MOGraphLowering creates an MGP ModelOp, converts input/output types using device-aware builders, initializes root execution chains and lowers graph operations in dependency order. Model metadata carries input/output names, types and devices forward. Some allocations/resources can later move or be reused; this structure does not guarantee one fresh output allocation per call. Add-one has no external model weights, though the runtime interface still carries a weights registry.'),
 dict(id='mgp-opt',title='08 · buildMgpPipeline: resource and dependency plan',source='gc',line=382,
 why='Make allocation lifetimes and execution dependencies explicit, then simplify them so runtime calls use safe resources and useful ordering rather than redundant bookkeeping.',
 before=['Buffers + init/execute plan','Chains and repeated setup','Temporary lifetimes'],after=['Hoisted invariant setup','Eligible buffer reuse / memory plan','Validated arguments and lifetimes'],
 example='''Invariant setup (schematic comparison)
  execute computes a shape/stride independent of X data
  → ExecInvariantCodeMotion may move it into init

Buffer reuse (eligible non-overlapping temporaries)
  A_TMP live during stage A, dead before stage B_TMP
  → reuse storage slot instead of retaining two live allocations
  Never reuse when lifetimes overlap or aliases require retention.

Simplify chains:
  ordering tokens used only to join already-covered dependencies
  → remove redundant dependency structure, keep real prerequisites

Plan order matters: invariant motion precedes SimpleBufferReuse.
DynamicMemoryPlanning, DeferAllocs and model-argument checks follow.''',
 detail='Other passes include predicate/assert dependencies, Python traceback attachment, globalizing slot indices and LowerPrimitives. These are compile-time rewrites of the runtime plan, not physical memory transfers or measured allocations. Example optimization opportunities are conditional; a one-call add-one graph may have little to change.'),
 dict(id='device-graphs',title='09 · FinalizeDeviceGraphs and EmitMojo',source='emit',line=147,
 why='Finalize explicitly marked device-graph build interfaces when present, then express surviving native routines as Mojo source the kernel compiler can compile.',
 before=['MGP plan + marked graph regions','MOGG function K0 + P1 refs'],after=['Finalized marked graph bridge','Generated source + package imports','Plan retains routine references'],
 example='''Marked GPU comparison (when device-graph regions exist)
  stage → partition collectives → compute/simplify dependencies
  → generate build-graph stubs → finalize builder bridge
  Not Graph.output; not recording a Chrome animation;
  not guaranteed CUDA capture for every add-one call.

EmitMojo (schematic source ingredients)
  imports: required runtime/stdlib helpers + P1 package/modules
  K0(output_ptr,input_ptr,ctx):
    reconstruct typed tensors from pointers and shape metadata
    invoke registered AddOne.execute with target information
    translate supported effects/errors through emitted glue

The generated source is typically in-memory compiler input.
Optional dumpMojoText exports a debugging view; no .mojo file is mandatory.''',
 detail='Before emission the pipeline outlines async routines, lifts buffer-cache slots, canonicalizes functions for merging, merges identical functions, removes dead symbols and wraps needed raising calls. EmitMojoState uses its registry and source builders to render MOGG FuncOps. Device-graph finalization applies to marked regions and validates their dependencies; it does not itself submit hardware work. Add-one without such regions simply has no marked graph to finalize.'),
 dict(id='jit',title='10 · JitCompileKernels: source to native routines',source='jit_entry',line=529,
 why='The model plan still refers to routines, not target executable bytes. JIT prepares or retrieves code for those routines so the later runtime interface can load and invoke them.',
 before=['Generated Mojo + P1','Target info + routine symbols'],after=['Compiled/reused code archive','Compiled load/call IR','Kernel symbol inventory'],
 example='''Cold kernel route:
  parse emitted Mojo → LIT → KGEN
  specialize parameters / elaborate concrete routines
  lower MLIR toward LLVM dialect → translate to LLVM IR
  selected backend emits target artifacts

CPU: host native code; GPU: host glue + device-target routine code
Example target: NVIDIA sm_90a / AMD gfx942 (source cross-target test)
Artifact forms/tool paths are explained in C.2 and the compiler workbench.

Compatible kernel cache hit → reuse native archive
  graph planning/interface work is still needed
Graph-artifact hit earlier → skips this fresh preparation path entirely.''',
 detail='JIT means code preparation during the running program, rather than requiring a prebuilt executable for every graph. This route invokes compiler APIs in process. It does not imply recompiling on every model.execute or necessarily lazy first-call compilation. The earlier mojo precompile subprocess prepared P1; it is a separate phase. This page illustrates code preparation, not a successful accelerator build/run on this host.'),
 dict(id='runtime-calls',title='11 · LowerCompiledRoutines: load and indexed execute',source='lower',line=102,
 why='Replace compiler-level routine references with operations whose runtime implementation knows which module and callable to use, and ensure initialization calls wait for the correct module load.',
 before=['CompiledLoad archive + symbols','CompiledCall callee=K0'],after=['ModuleLoad in init','ExecuteGeneric(slot,module,kernel)','Load-before-call dependencies'],
 example='''Illustrative symbol table:
  archive A0 contains [K0,K1]
  kernelModuleAndFnIndex[K0] = (module=0,kernel=0)
  kernelModuleAndFnIndex[K1] = (module=0,kernel=1)

init: CompiledLoad(A0) → mgp.module.load(runtimeSlot=s0,
                                       archive=A0,symbols=[K0,K1])
exec: CompiledCall K0(X,Y,ctx)
   → ExecuteGeneric(context,args, runtimeSlot=s0,
                    module_index=0,kernel_index=0)

Indices are illustrative table entries, not captured add-one IDs.
The actual ModuleLoad happens during runtime initialization,
not when this compiler pass creates its IR operation.''',
 detail='LowerCompiledRoutines builds the kernel-symbol-to-module/function-index mapping from CompiledLoad symbols. It rewrites both init and execute block calls, preserving init ordering and per-module load dependencies. The async flag follows the old dispatch kind; this is not an arbitrary new scheduling policy. GPU runtime behavior remains behind the compiled/driver interfaces, not a hardcoded ptxas command here.'),
 dict(id='callables',title='12 · buildJitPipeline: outline init/execute callables',source='gc',line=75,
 why='Seal the model into stable callable entry points and metadata so compiled bytes can become CM1, initialization can produce M1, and execution can pass real buffers.',
 before=['MGP model init/execute regions','Runtime calls + chains'],after=['MEF init callable + exec callable','Model descriptor + prepared bytes','CM1 ready for session.init'],
 example='''buildJitPipeline:
  LowerCompiledRoutines
  InlineMGPCalls → FixupOverhangs → CSE → SimplifyChains
  canonicalization → CSE → OutlineModelsToCallables

Before: mgp.model @addition {init {...}; execute {...}}
 After: init callable I0 + execute callable E0
        descriptor: name=addition, input/output specs,
                    device refs, callable references

Export → CM1 (prepared artifact, not initialized model)
session.init(CM1) → invokes I0 → M1
model.execute(B1) → invokes E0 → output Buffer O1
G1 and its Python wrappers remain caller-owned.''',
 detail='OutlineModelsToCallables uses per-model GraphContext state that starts unsealed with ModelOp and ends sealed with MEF initCallable/execCallable. It handles referenced subgraphs before sealing parents. Inlining/cleanup remove redundant plan structure but must preserve effects and real chain dependencies. This final plan boundary explains why compile, init and execute are distinct API calls.'),
]
