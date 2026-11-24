from content import actor as A, box as B, chapter as C, ref as R, section as Q, step as E

C('mgp_plan','MOGG → MGP: model phases, buffers and dependency plans',
 'Make initialization and repeated execution explicit; plan temporary storage without stealing persistent weights or KV pages.',
 'Compiler transformations',[
 A('MODEL','mgp.model','executable-plan IR','IR1','Takes over the graph symbol and carries device/input/output contracts.'),
 A('INIT','Init region','once-per-model work','mgp.model','Establishes context slots, devices and persistent resources.'),
 A('EXEC','Execute region','per-invocation plan','mgp.model','Consumes root chain/context and input buffers; produces results/dependencies.'),
 A('MEM','Buffer planner','allocation/lifetime analysis','MGP passes','Uses dependency-aware lifetimes and eligible dynamic sizes for scratch storage.'),
 A('PERSIST','Weights and KV','retained / borrowed resources','model + cache manager','Weight binding and request KV ownership survive execution-local scratch reuse.'),
 A('ASYNC','Async routines','outlined host work','MOGG/MGP lowering','Defines callable work and lifetimes for runtime scheduling.')],[
 E('Create an executable model representation','MOGraphLowering builds device type builders, converts signatures, preserves argument locations and creates mgp.model with the graph’s symbol.',{'MODEL':B('model representation','name=llama3','devices + argument/result metadata')},[R('mogg_mgp','MOGraphLowering::createModelOp()')]),
 E('Create initialization state','An init chain and mgp.context begin with zero slots; lowering adds resources as needed. Device buffer builders establish their setup here.',{'INIT':B('resource setup','init chain','mgp.context(numSlots=0 → grows)')},[R('mogg_mgp','loweringState.setInitChain('),R('mogg_mgp','/*numSlots=*/0')],edges=[('MODEL','INIT')]),
 E('Create the execution entry','Execute block argument 0 is the root chain; argument 1 is the MGP context. Model input tensor types become runtime-compatible descriptors/buffers.',{'EXEC':B('entry contract','root chain + context','tokens / offsets / KV inputs')},[R('mogg_mgp','setRootExecuteChain(executeBlock->getArgument(0))'),R('mogg_mgp','setExecuteMgpContext(executeBlock->getArgument(1))')],edges=[('MODEL','EXEC')]),
 E('Keep persistent and temporary ownership separate','Initialized weights and externally owned cache buffers are resources/inputs. Intermediate QKV, normalized activations and MLP outputs have execution lifetimes.',{'PERSIST':B('retained / borrowed','weights: model-owned references','KV: cache-manager-owned input'),'MEM':B('temporary candidates','QKV / norms / MLP intermediates')},[R('mogg_mgp','lower(MO::ConstantExternalOp'),R('memory','struct AllocInfo')],edges=[('PERSIST','EXEC')]),
 E('Hoist invariants before reuse','Execution-invariant code motion precedes simple buffer reuse because premature reuse can prevent hoisting. Dynamic memory planning then considers eligible device allocations.',{'MEM':B('planned scratch','ExecInvariantCodeMotion → SimpleBufferReuse','dynamic sizes + alignment + use sets')},[R('compiler','createExecInvariantCodeMotionPass'),R('compiler','createSimpleBufferReusePass'),R('memory','isTriviallyComputable')]),
 E('Respect lifetime dependencies','A temporary cannot be reused while a live consumer needs it. Dynamic planning consults the dependency tree, defers allocations and preserves required lifetimes.',{'MEM':B('reuse constrained by uses','X stays live through residual add','workspace reuse after dependency-safe last use')},[R('memory','BufferLifetimeAnalysisPass'),R('compiler','createDeferAllocsPass'),R('compiler','createSimplifyExtendLifetimesPass')],edges=[('EXEC','MEM')]),
 E('Lower primitives and outline asynchronous work','Global slot indices, primitive lowering, optional device-graph-related operations, sequence fusion, async outlining and resource lookup prepare generated callable stubs.',{'ASYNC':B('outlined work','async routine arguments + resources','slot indices / chains / error paths')},[R('compiler','createGlobalizeSlotIndicesPass'),R('compiler','createOutlineAsyncRoutines'),R('compiler','createLookupConstantResources')],edges=[('INIT','ASYNC'),('EXEC','ASYNC')]),
 E('Canonicalize equivalent stubs before emitting Mojo','Lift eligible buffer-cache slot constants into arguments; canonicalize and merge identical functions. Preserve data/address differences as runtime arguments.',{'ASYNC':B('ready for code generation','deduplicated callable bodies','try/except → EmitMojo')},[R('compiler','createLiftBufferCacheSlots'),R('compiler','createMergeIdenticalFuncs'),R('compiler','createAddTryExceptBlocks')])],[
 Q('The MGP model has distinct init and execute regions','''
```text
mgp.model @llama3
 ├─ init(device contexts, weights/resources)
 │   ├─ init chain
 │   ├─ mgp.context / resource slots
 │   └─ device-specific buffer/resource setup
 └─ execute(root chain, mgp.context, model inputs)
     ├─ tensor descriptors / runtime shapes
     ├─ allocation + primitive/call dependency plan
     └─ output chain + result buffers
```

The conversion establishes type builders before lowering signatures, preserves named/device-index metadata, and changes the module symbol table from GraphOp to ModelOp. It sets up buffer builders after init/execute contexts exist. These ordering details prevent type/device-dependent lowering from referring to nonexistent model state.
''',[R('mogg_mgp','MOGraphLowering::createModelOp()'),R('mogg_mgp','moduleSymbolTable.remove(graph)')],open=True),
 Q('MGP optimization ledger, in source order','''
| Phase | Scheduled work |
|---|---|
| early cleanup | CSE; canonicalization; SimplifyChains |
| model/storage normalization | ConcatInPlace; ExecInvariantCodeMotion; SimpleBufferReuse |
| memory plan | DynamicMemoryPlanning; DeferAllocs; VerifyModelArguments; PredicateAssertDependencies; chain/CSE/canonicalization cleanup |
| diagnostics / slots | AttachPythonStackTraces; GlobalizeSlotIndices; cleanup |
| primitive lowering | LowerPrimitives with package paths and minimum CPU alignment |
| device-graph machinery | StageDeviceGraphs; PartitionDeviceGraphCollective; Compute/SimplifyDeviceGraphDependencies; GenerateBuildGraphStubs; FinalizeDeviceGraphs; reconcile casts, with cleanup between |
| executable grouping | SequenceFusion; cleanup; SimplifyExtendLifetimes |
| host routine preparation | OutlineAsyncRoutines; LookupConstantResources; LiftBufferCacheSlots |
| deduplication | CanonicalizeFuncsForMerge; cleanup; MergeIdenticalFuncs; RaiseBackIndexConstant; FixupOverhangs; SymbolDCE |
| diagnostic/error wrapping | optional AddProfileScopeTracing; AddTryExceptBlocks |
| output | EmitMojo; optional DumpMojo |

The single-device GraphSpec defaults to `is_device_graph=False`. Scheduling device-graph passes does not prove that this decoder became one captured CUDA graph. Pass applicability depends on the IR and options. Collective-related passes may have no work on this single-device path.
''',[R('compiler','void M::MO::buildMgpPipeline'),R('spec','is_device_graph: bool = False')],open=True),
 Q('Dynamic allocation planning: the actual eligibility limits','''
The pass gathers BufferAlloc operations in the execute region and consults BufferLifetimeAnalysis’s dependency tree. Its source skips CPU allocations. It requires useful allocation uses and a runtime size computable from allowed block-argument arithmetic/tensor-spec operations; arbitrary data-dependent sizes are not silently included.

Allocation alignment and device grouping matter. A plan for temporary device storage is different from the serving parent’s MemoryPlan for the persistent KV budget. Both use the word “planning,” but they operate at different levels and times.

For the Llama residual stream, the input X remains required until its residual add consumes it. A sequential drawing is insufficient to prove storage can be overwritten: dependencies, aliases and asynchronous users must establish the last safe use.
''',[R('memory','isTriviallyComputable'),R('memory','Plan CPU allocations too'),R('memory','BufferLifetimeAnalysisPass')]),
 Q('Why layer-specific slot constants can obstruct code reuse','''
Two generated host stubs may perform the same work but refer to different buffer-cache slot numbers. LiftBufferCacheSlots promotes a constant only when all its uses are cached-buffer-slot operands; callers pass the old value as a new argument. After canonicalization, the function bodies can compare equal and merge.

The call sites still pass distinct slots/weights/layer data. Merging code does not mean layer 0 and layer 31 share the same parameter tensor or KV region. The source comment uses another model as an example; this reader does not claim a measured number of Llama stub merges.
''',[R('emit','allUsesAreCachedBufferSlot'),R('emit','void liftBufferCacheSlots('),R('compiler','createMergeIdenticalFuncs')]),
 Q('Chains become readiness; lifetime becomes retained ownership','''
The execute plan carries dependencies for inputs, effects and resources. Async outlining preserves captures needed until host work completes; generated native primitives later use reference-counted AsyncValue slots and retained execution context. This is how an apparently simple `M1.execute` reaches a dependency-driven runtime.

A compiler-deleted intermediate and a runtime buffer released after last use are different events. The compiler establishes the legal plan; the runtime enforces concrete ownership for a particular invocation.
''',[R('compiler','createOutlineAsyncRoutines'),R('primitive','CompiledKernelAsyncABIFn')])],'''
MOGGToMGP → MOGraphLowering.createModelOp
├─ device type builders → converted input/output signatures
├─ mgp.model replaces graph symbol
├─ init chain + MGP context + device buffer builders
└─ execute root chain/context + input buffers
   └─ buildMgpPipeline
      ├─ invariant motion → reuse → dynamic memory planning
      ├─ primitive / chain / slot lowering
      ├─ device-graph machinery where applicable
      ├─ async routines + resource lookup + stub merging
      └─ error wrapping → EmitMojo
''',[R('mogg_mgp','MOGraphLowering::createModelOp()'),R('compiler','void M::MO::buildMgpPipeline')])

C('mojo_codegen','Generated Mojo, specialization, KGEN and LLVM',
 'Separate the execution plan, generated host stubs, parametric implementation packages and device code generation.',
 'Compiler transformations',[
 A('PLAN','MGP execution plan','IR','graph compiler','Keeps model resources and runtime dependency structure.'),
 A('MOJO','Generated Mojo stubs','code resource','EmitMojo','Expresses typed calls, descriptor/shape construction, runtime helpers and error adaptation.'),
 A('KGEN','Specialized KGEN','compiler IR','Mojo compiler','Elaborates parameters and implementation code under the chosen target environment.'),
 A('CACHE','MOGG / Mojo caches','archive references','JIT compilation stage','Separately tracks generator identity and compiled archive availability.'),
 A('LLVM','LLVM/backend modules','host/device code IR','object compiler','Lowers/optimizes target code; CPU and GPU code have separate target roles.'),
 A('BIN','Compiled resources','archive / image bytes','C1 artifact','Contains reusable host objects and device code resources, not a token batch.')],[
 E('Emit host-callable Mojo from final routines','EmitMojo sees routines after conformance, async outlining, resource handling and try/except insertion. It is downstream of the graph memory/dependency plan.',{'PLAN':B('MGP plan retained','init / execute / resources'),'MOJO':B('generated text resource',('mojo','typed host stubs + calls'))},[R('compiler','createEmitMojo(emitMojoOptions)'),R('emit','EmitMojo')],edges=[('PLAN','MOJO')]),
 E('Build the compilation environment','The JIT stage gathers packages, target settings and generated routines. Its cache key includes environment attributes so incompatible targets do not share code accidentally.',{'KGEN':B('parametric program','built-in package code + generated entry stubs'),'CACHE':B('environment-keyed lookup','target / compile environment + semantic generator key')},[R('jit','buildCacheKey('),R('jit','kernelLibraryPaths')],edges=[('MOJO','KGEN')]),
 E('Check both cache layers','A MOGG entry points to a Mojo archive key. A MOGG hit with a missing/erroring Mojo archive is treated as a compilation miss.',{'CACHE':B('usable hit or genuine miss','MOGG hit + Mojo miss → compile','usable archive hit → retain')},[R('jit','MOGG hit but Mojo miss')]),
 E('Elaborate missing specializations','On misses, runElaborationPipeline resolves compile-time arguments and the selected implementation program. Static head/layout/dtype/target facts specialize code; token IDs do not.',{'KGEN':B('elaborated specializations',('kgen','resolved routines / types / target'))},[R('jit','runElaborationPipeline('),R('builtin','def _execute_mha_ragged_paged_scalar_args[')],edges=[('MOJO','KGEN')]),
 E('Lower KGEN function and closure structure','The ordered LLVM pipeline first converts fully elaborated functions/types, verifies the resulting functions, then lowers applicable runtime closures to a function pointer plus captured-state pointer.',{'KGEN':B('function structure lowered'),'LLVM':B('LLVM-dialect functions',('kgen','typed functions → llvm.func'),'exported ABI wrappers where required','closure = code pointer + state pointer')},[R('llvm_pipeline','pm.addPass(createLowerKGENToLLVM('),R('kgen_passes','def LowerRuntimeClosures')],transfer=['KGEN','function/type lowering','LLVM'],edges=[('KGEN','LLVM')]),
 E('Lower operations, control flow and suspension points','Legalize POP operands, lower global and function POP operations, form LLVM control-flow blocks, reconcile conversion casts and lower applicable suspension points. These are code-generation steps, not token sampling steps.',{'LLVM':B('low-level function bodies',('kgen','operations + CFG + explicit conversions'),'global calls/constants','resume state machine only if present')},[R('llvm_pipeline','createLegalizePOPOperations'),R('kgen_passes','def LowerControlFlow'),R('kgen_passes','def LowerSuspensionPoints')]),
 E('Run late cleanup, target lowering and debug lowering','Canonicalization with region simplification disabled and CSE precede target-contributed LLVM-dialect passes. DebugInfoToLLVM is last. A scheduled pass can leave this model’s functions unchanged.',{'LLVM':B('ready for LLVM IR translation',('kgen','canonicalized target LLVM dialect'),'late target passes if contributed','debug information lowered last')},[R('llvm_pipeline','GreedySimplifyRegionLevel::Disabled'),R('llvm_pipeline','createTargetSpecificLLVMLowering'),R('llvm_pipeline','DebugInfo::createDebugInfoToLLVM')]),
 E('Emit target objects into an archive','ObjectCompiler.emitArchive completes object/backend work. Host CPU code and GPU-target code retain distinct roles; the archive records bytes and requested symbols.',{'BIN':B('compiled archive',('archive','archive bytes + symbol list'))},[R('jit','state.objectCompiler->emitArchive('),R('object_compiler','emitArchive(')],edges=[('LLVM','BIN')]),
 E('Attach compiled loads to model initialization','Both hits and misses insert CompiledLoad operations into each model’s init block with symbols and archive bytes. Cache reuse still requires loading runtime resources.',{'PLAN':B('compiled resources referenced','init: CompiledLoad(archive,symbols)','execute: compiled routine calls')},[R('jit','MOGG::CompiledLoadOp::create('),R('jit','void handleCacheHits(')],edges=[('BIN','PLAN')]),
 E('Finish graph-level JIT lowering','Strip kernel declarations, lower compiled routines, inline MGP calls, clean chains/overhangs and outline models to callables before MEF translation.',{'PLAN':B('final callable plan','LowerCompiledRoutines → InlineMGPCalls','cleanup → OutlineModelsToCallables')},[R('compiler','createStripKernelDecls'),R('compiler','createLowerCompiledRoutines'),R('compiler','createOutlineModelsToCallables')])],[
 Q('Four programs coexist during compilation','''
| Representation | Responsibility | What it is not |
|---|---|---|
| G1 / MO tensor graph | model semantics and typed weights/inputs | machine code |
| MGP executable plan | init/execute, resources, buffers and readiness | merely a list of CUDA kernels |
| generated Mojo / elaborated KGEN | concrete callable interfaces and selected implementations | the original checkpoint |
| LLVM + target objects/images | low-level code generation for host/device | live tensor allocation or initialized driver handle |

Mojo emission is not a textual translation of every model layer into one user-visible Python-equivalent source file. It emits host routine machinery around the selected implementation and runtime contracts. Built-in P1 remains the origin of reusable implementation code.
''',[R('compiler','createEmitMojo(emitMojoOptions)'),R('jit','runElaborationPipeline('),R('jit','emitArchive(')],open=True),
 Q('Specialization inputs: what can change compiled code?','''
Relevant static inputs include implementation identity, target/triple/architecture, dtype, rank/layout, declared compile-time parameters and effective compiler options. Runtime buffer addresses, token IDs and ordinary symbolic batch extents remain execution inputs unless a particular compiler mechanism specializes them.

For Llama attention, 32 heads, head dimension 128, causal masking and the chosen activation/cache dtype constrain implementation forms. The GPU dispatch can still make runtime prefill/decode or partition decisions from lengths and prepared metadata. Static specialization and dynamic dispatch cooperate.

Changing requests from `[3,2]` prompt rows to `[1,1]` decode rows does not itself prove a full GraphSpec retrace or new C1 compile. The compiled model is designed to accept its symbolic shape contract.
''',[R('builtin','def _execute_mha_ragged_paged_scalar_args['),R('jit','buildCacheKey('),R('mha','def flash_attention_dispatch[')],open=True),
 Q('KGEN → LLVM: every scheduled lowering, in order','''
`buildLowerToLLVMPipeline` operates on the specialized program, downstream of the model graph transformations. **LLVM dialect** is still MLIR. Translation to LLVM IR and machine-code/backend emission follow it.

| Order | Pass / scope | Concrete transformation or constraint |
|---|---|---|
| 1 | LowerKGENToLLVM / module | Convert elaborated KGEN functions/types; establish internal versus exported linkage and C-compatible wrappers for exports. Carry configured global constructor/destructor names. |
| 2 | KGENVerifier / each LLVM function | Check the post-conversion function representation before further lowering. |
| 3 | LowerRuntimeClosures / module | Where closures exist, represent callable state as `{code pointer, captured-state pointer}`; invocation unpacks both and passes state plus dynamic arguments. |
| 4 | LegalizePOPOperations / each LLVM function | Adapt unsupported primitive operand/result types to a supported operation type with necessary conversions. The source’s FP8 illustration is not a claim that this unquantized Llama uses FP8. |
| 5 | LowerGlobalPOPToLLVM / module | Lower primitive operations with module-wide consequences: globals, external calls and applicable aligned allocation/free runtime declarations. |
| 6 | LowerPOPToLLVM / each LLVM function | Convert remaining elaborated primitive operations into LLVM-dialect operations. |
| 7 | LowerControlFlow / each LLVM function | Turn structured flow into LLVM/control-flow blocks. It follows function lowering because those CFG regions belong in `llvm.func`. |
| 8 | ReconcileUnrealizedCasts / each LLVM function | Reconcile conversion-boundary cast scaffolding. This is distinct from deleting numerically required BF16/FP32 casts. |
| 9 | LowerSuspensionPoints / each LLVM function | Lower suspension points in resume functions into a state machine **when present**. This does not prove that every decoder kernel is a coroutine. |
| 10 | Canonicalizer / each LLVM function | Simplify resulting operations; region simplification is explicitly disabled here because the source documents exponential behavior. |
| 11 | CSE / each LLVM function | Share eligible repeated expressions while preserving effects. |
| 12 | TargetSpecificLLVMLowering / module | Resolve the target lowering by triple and run its contributed late passes. No matching contribution means no work. |
| 13 | DebugInfoToLLVM / module | Lower debug information last; optimization level zero enables the stated variable-debug tradeoff. |

These passes lower the **selected implementations and generated host program**. They do not replace the 32-layer model with a new architecture, choose request token IDs, merge distinct layer weights, or establish GPU completion.
''',[R('llvm_pipeline','void KGEN::buildLowerToLLVMPipeline'),R('kgen_passes','def LowerKGENToLLVM'),R('kgen_passes','def LegalizePOPOperations'),R('kgen_passes','def TargetSpecificLLVMLowering')],open=True),
 Q('A descriptor and a captured callback across the lowering boundary','''
```text
symbolic model tensor: Q[T,32,128]
  → implementation operand: pointer + shape/stride/layout + dtype contract
  → elaborated function signature and concrete field types
  → LLVM-dialect fields / pointers / operations
  → target instructions using this invocation's runtime addresses

applicable runtime closure:
  captured values ──> retained state storage ──> state pointer ─┐
  lifted function ──> wrapper / unpacking code ──> code pointer ├─ closure pair
  invocation ──> extract both ──> code(state, dynamic args) ───┘
```

The source describes exported wrappers that split aggregate arguments/results and pass arrays by reference when the export convention requires it. It does not prove one universal flat signature for all internal routines. The MGP runtime independently selects `CompiledKernelAsyncABIFn` at its host entry point; the CUDA driver later consumes a separate device argument packet.

The closure lowering description explains how compiled language-level captures become data plus callable code. AsyncRT's C++ work-item captures and AsyncValue waiter storage have their own implementations. Their shared lifetime concern does not make them one representation.
''',[R('kgen_passes','def LowerKGENToLLVM'),R('kgen_passes','def LowerRuntimeClosures'),R('primitive','CompiledKernelAsyncABIFn')]),
 Q('Cache miss, partial hit and archive attachment','''
The MOGG cache associates semantic generator hashes with Mojo archive keys. A complete usable hit provides archived compiled symbols. A missing secondary archive converts an apparent first-level hit into a miss. Cache misses run elaboration and object emission, then record archive identity for future use.

Both routes attach `CompiledLoad` entries to init. Reusing code bytes skips compilation work but does not supply the current process’s resolved function pointers, device module handles or weight buffers. These are established during loading/initialization or a later generated first-use path.
''',[R('jit','MOGG hit but Mojo miss'),R('jit','handleCacheMisses('),R('jit','void handleCacheHits(')]),
 Q('The final graph passes still matter after machine code exists','''
After JitCompileKernels and StripKernelDecls, the source schedules LowerCompiledRoutines, InlineMGPCalls, FixupOverhangs, CSE, SimplifyChains, canonicalization/CSE and OutlineModelsToCallables. These connect compiled routines back into the runtime plan.

`convertMGPToMEF` clones the final module, prepares device primitives and translates the model representation into MEF. The public C1 wraps the resulting artifact; native execution does not interpret the original Python layer constructors on every token.
''',[R('compiler','createLowerCompiledRoutines'),R('compiler','convertMGPToMEF(')]),
 Q('Compiler-debug boundaries','''
The source supports stage dumping and a pre-JIT stop after Mojo emission. Those facilities can provide actual IR/code evidence in a future compatible model compile. This reader’s operation lineages deliberately use schematic notation and preserve option predicates.

No exact instruction count, register allocation, tensor-core selection, compiler-time speedup or final launch count is inferred from the existence of an optimization pass. Backend source and a generated artifact answer different questions.
''',[R('compiler','createDumpStagePass'),R('frontend','if (options.preJit)')])],'''
EmitMojo → generated code resource
└─ JitCompileKernels
   ├─ environment + generator cache keys
   ├─ usable cache hit → compiled archive
   └─ miss → elaborate KGEN → ObjectCompiler.emitArchive → LLVM/backends
      └─ CompiledLoad(archive,symbols) in init region
         └─ StripKernelDecls → final JIT graph pipeline
            └─ PrepareDevicePrimitives → MEF → C1
''',[R('jit','handleCacheMisses('),R('compiler','createStripKernelDecls'),R('compiler','convertMGPToMEF(')])

C('gpu_artifacts','Host objects, device images and two loading boundaries',
 'Keep A1/H1 and D1/KD1 separate: compiled bytes become different callable resources in the host and GPU contexts.',
 'Compiler transformations',[
 A('LLVM','Target code generation','compiler','Mojo object compiler','Produces host objects and device-target executable resources.'),
 A('A1','A1 · host object/archive','compiled bytes','C1 / compiled resource','Contains host code, symbols, relocations and referenced resources.'),
 A('H1','H1 · resolved host callable','host function pointer','CompiledKernelsContext / JIT','Entered by MGP’s runtime primitive with AsyncValue/resource ABI.'),
 A('D1','D1 · GPU executable image','compiled bytes','compiled resource','NVIDIA path can produce cubin from emitted PTX via a compilation device.'),
 A('KD1','KD1 · loaded GPU function','driver handle','retained CUDA module/context','Used for device launch; not the same address/ABI as H1.'),
 A('DATA','Weights / KV / activations','tensor allocations','model/cache/execution owners','Actual tensor data lives in separately owned buffers, not automatically in ELF .data.')],[
 E('Split host and device code roles','Generated host stubs coordinate descriptors, runtime calls and device submission. Selected GPU functions compile for a device target.',{'LLVM':B('target-specific compilation','host CPU module','NVIDIA GPU-target module')},[R('jit','state.objectCompiler->emitArchive('),R('nvptx','class NVPTXBackend')]),
 E('Emit host objects','The compiled archive contains symbols and relocatable host code. It is a resource in C1 before loading yields a callable pointer.',{'A1':B('prepared bytes',('a1','host object/archive A1'))},[R('jit','createArchiveAttr('),R('host_loader','CompiledKernelsContext::load')],edges=[('LLVM','A1')]),
 E('Produce NVIDIA executable bytes','The NVPTX backend emits PTX, checks the target architecture and uses a CUDA compilation device to produce cubin. A ptxas subprocess is not universal.',{'D1':B('prepared GPU image',('d1','PTX → compilation device → cubin'))},[R('nvptx','Create CompilationDevice'),R('nvptx','compileFunction(ptx')],edges=[('LLVM','D1')]),
 E('Load host symbols through the JIT','Normal loading uses the archive hash for the JIT namespace, adds the static archive if absent and looks up requested symbols. H1 becomes a host function pointer.',{'H1':B('host-callable',('h1','resolved H1 function pointer'))},[R('host_loader','addIfAbsent<KGEN::StaticArchiveLayer>'),R('host_loader','engine->lookup(dylibName, kernelName)')],edges=[('A1','H1')]),
 E('Load a GPU module and look up its function','At the generated device load site, CUDA cuModuleLoadDataEx consumes image bytes and cuModuleGetFunction returns KD1. This can occur on a cold first execution.',{'KD1':B('driver function handle',('kd1','CUmodule retained → CUfunction KD1'))},[R('cuda','cuModuleLoadDataEx'),R('cuda','cuModuleGetFunction')],edges=[('D1','KD1')]),
 E('Bind buffer addresses at launch time','M1 supplies weight/resource references and execution supplies activation/KV addresses. The kernel code stays separate from these payload allocations.',{'DATA':B('separate data ownership','W: initialized model weights','KV: cache pool','X/O: execution buffers')},[R('primitive','CompiledKernelAsyncABIFn'),R('cuda','cuLaunchKernelEx')],edges=[('H1','KD1'),('DATA','KD1')]),
 E('Retain code handles across warm execution','Later calls can reuse H1/KD1 and C1/M1 while arguments and request buffers change. Destroying the owning resources eventually unloads code safely.',{'H1':B('reused',('h1','same resolved callable')),'KD1':B('reused',('kd1','same retained module/function'))},[R('host_loader','addIfAbsent<KGEN::StaticArchiveLayer>'),R('cuda','cuModuleUnload')])],[
 Q('Artifact dictionary and address spaces','''
| Name | Representation | Where meaningful |
|---|---|---|
| P1 | parametric implementation package | compiler import/specialization |
| IR1 | mutable MLIR graph/compiler clone | host compiler memory |
| MEF / C1 | serialized compiled model and resource description | artifact loading/init |
| A1 | host object/archive bytes | host linker/JIT input |
| H1 | resolved host callable address | current host process |
| D1 | GPU image bytes, e.g. cubin | CUDA module-load input |
| KD1 | CUDA function handle | its module and CUDA context |
| W/KV/X | typed tensor buffers | driver-managed host/device memory |

Host object code/data/relocations and tensor payload memory have different owners. A loaded JIT symbol namespace does not itself imply a temporary shared library was written. A CUDA module handle is not a host-callable C++ function pointer.
''',[R('host_loader','CompiledKernelsContext::load'),R('cuda','cuModuleGetFunction')],open=True),
 Q('Normal host loading and the debug shared-library branch','''
Normal `CompiledKernelsContext::load` hashes the archive, calls the execution engine’s atomic add-if-absent path through StaticArchiveLayer, looks up each requested symbol and stores its function pointer. ORC/JITLink-style object loading handles host relocation/symbol resolution through the execution engine.

When `loadDebuginfo` is enabled, a separate branch writes objects, links a temporary `.so`, loads it with a dynamic-library search generator and exposes symbols for profilers. This conditional debug route does not justify teaching “every MAX kernel is dlopened from a temporary shared library.” Installed CUDA/HIP runtime libraries also have a separate origin from generated model objects.
''',[R('host_loader','if (loadDebuginfo)'),R('host_loader','addIfAbsent<KGEN::StaticArchiveLayer>'),R('host_loader','addKernel(kernelName, kernelIndex')],open=True),
 Q('NVIDIA backend: PTX → cubin is a compiler boundary','''
The NVPTX backend emits PTX through its LLVM code-generation path. It derives the cubin architecture from the same target machine so the PTX target and assembly target agree. It creates a CUDA CompilationDevice and calls compileFunction with the PTX and compile settings.

The source has an NVPTXCompiler API route and an environment-controlled alternative. It does not establish a mandatory external `ptxas` process for every model. The later CUDA driver image loader consumes the prepared executable bytes; do not relabel all module loading as fresh graph compilation.

AMD uses another target/backend and driver path. This page’s concrete CUDA handles and cubin terminology apply to the NVIDIA illustration only.
''',[R('nvptx','Create CompilationDevice'),R('nvptx','MODULAR_NVPTX_COMPILER_PATH'),R('nvptx','compileFunction(ptx')]),
 Q('When is D1 actually loaded?','''
Compiled bytes may already exist in C1, while GPU function handles are still absent. Initialization sets up model resources and host callables, but generated device-load/check sites can be reached through execution, including first use. The driver retains modules/functions for repeated work in their context.

This is why “compile completed,” “model initialization returned,” “device module loaded,” “kernel enqueued” and “kernel completed” are separate milestones. A precise runtime timeline would require instrumentation of the chosen model/target; this source walkthrough does not claim every module is eagerly uploaded during Python init.
''',[R('cuda','cuModuleLoadDataEx'),R('primitive','CompiledKernelAsyncABIFn'),R('engine_context','Engine::Context::initModel(')]),
 Q('Two ABI boundaries lead into the driver','''
The MGP primitive calls H1 through `CompiledKernelAsyncABIFn`, using addresses of AsyncValue reference slots and resource entries. Generated host code ultimately builds a device-layout argument packet whose fields include device pointers and scalar/control values. The CUDA driver consumes that second packet with launch configuration and KD1.

Do not substitute the separately declared two-argument compiled-kernel ABI for the primitive’s actual async-ABI call. The existence of multiple ABI declarations makes the selected call site authoritative. The driver lesson traces the second packet and completion gates in detail.
''',[R('primitive','CompiledKernelAsyncABIFn'),R('cuda','CUlaunchConfig'),R('cuda','cuLaunchKernelEx')])],'''
ObjectCompiler
├─ host object/archive A1
│  └─ normal JIT archive loading → symbol lookup → H1
│     └─ conditional debug branch: temporary shared library
└─ GPU target: PTX → CompilationDevice → cubin D1
   └─ generated load/check site
      ├─ cuModuleLoadDataEx → retained CUmodule
      └─ cuModuleGetFunction → KD1
         └─ cuLaunchKernelEx(KD1, launch configuration, device args)
''',[R('host_loader','CompiledKernelsContext::load'),R('nvptx','class NVPTXBackend'),R('cuda','cuLaunchKernelEx')])
