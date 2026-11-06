# Compiler workbench: research and teaching decisions

Expanded November 7, 2026 after the user approved deeper research and implementation.
[Open the compiler workbench](link_htmls/max_mlir_mojo_compiler_and_runtime.html).
The accepted [InferenceSession reader](link_htmls/max_inference_session_devices_runtime_and_compilation.html)
links to it. The parent and all four example readers now begin with a connected
operation journey; their focused scenes remain available as zooms.

## Research that changed the animation

Primary compiler documentation, compiler-author talks and original worked
tutorials were inspected. The web tool returned no usable content; direct HTTPS
retrieval succeeded. [Retrieval evidence](evidence/compiler_deep_research.json)
records URLs, dates and hashes without redistributing full articles or PDFs.
Animations, source access and fonts work offline; external readings are optional.

| Source | Mechanism learned | Applied visual decision / limit |
| --- | --- | --- |
| [Modular's LLVM 2023 presentation](https://www.modular.com/blog/mojo-llvm-2023) | Progressive semantic explanations and clickable animated slides | Retain the accepted interactive workbench; the hosted player is not a dependency |
| [VizMLIR](https://github.com/joepothiboot/vizmlir) | Following an operation through creation, fusion, lowering and erasure | Select source operation, loads, stores or plan; retain ancestry across changing representations |
| [mlir-interactive](https://github.com/mortbopet/mlir-interactive) | Inspecting intermediate representations and composing passes | Before/after panes and stage navigation; its Qt/GIF demonstration is inspiration, not MAX evidence |
| [LLVM: Life of an instruction](https://blog.llvm.org/2012/11/life-of-instruction-in-llvm.html) | An instruction changes form through backend work | Follow a typed operation into target operations and registers; this 2012 walkthrough is historical |
| [Nikita Popov: LLVM middle-end pipeline](https://www.npopov.com/2023/04/07/LLVM-middle-end-pipeline.html) | Grouped/repeated optimization pipelines | Stage views are conceptual groups; avoid a universal one-pass conveyor or invented exact pass list |
| [Jeremy Kun: MLIR lowering through LLVM](https://www.jeremykun.com/2023/11/01/mlir-lowering-through-llvm/) | A concrete worked conversion to lower-level IR and native code | Small operation fragments next to their predecessors; our fragments remain schematic |
| [Eli Bendersky: JIT compilation with LLVM](https://eli.thegreenplace.net/2017/adventures-in-jit-compilation-part-3-llvm/) | Generating native code and subsequently calling it | Preparation clock versus execution calls; this tutorial's older API is not MAX's implementation |
| [MLIR Toy 5](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) / [Toy 6](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/) | Partial conversion, mixed dialects, LLVM translation and execution | C1's graph/plan stays separate from KIR1's implementation and LLVM1's translated module |
| [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/) | Tensor values, buffer aliases and read/write conflicts | Epilogue intermediates, consumer edges, lifetime endings and logical traffic counters; no claim that MAX uses One-Shot Bufferize |
| [MLIR Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/) | Payload operations and transformation handles have different roles | Source ancestry selection changes the highlighted lineage and Next change; no claim that MAX uses Transform dialect handles |
| [LLVM Code Generator](https://llvm.org/docs/CodeGenerator.html) | Selection, virtual registers, register allocation and MC emission | Separate native instruction selection/register views; NVIDIA PTX registers remain virtual until device compilation |
| [LLVM ORCv2](https://llvm.org/docs/ORCv2.html) | Symbol materialization, lookup and optional lazy compilation | Separate module loading/callable resolution from calling a retained model; no invented ORC implementation for MAX |
| [Clang CrossCompilation](https://clang.llvm.org/docs/CrossCompilation.html) | Host, target triple/features, libraries and toolchain support differ | CPU, NVIDIA and AMD selections change artifacts; cross-codegen is separate from accelerator execution |
| [MLIR GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/) | Host/offload representation and target serialization | Show host routines and device code separately; generic GPU passes are not MAX's exact pipeline |
| [Mojo optimization pipeline, LLVM 2024](https://llvm.org/devmtg/2024-10/slides/techtalk/Weiwei-What-We-Learned-Building-Mojo-OptimizationPipeline.pdf) | Early structured-IR simplification, elaboration, LLVM-module splitting and parallel codegen | Do not imply a separate process per stage or a fixed pass count; historical performance figures are not reproduced as current measurements |
| [Mojo GPU compilation, LLVM 2025](https://llvm.org/devmtg/2025-10/slides/lightning_talks/chen_dakkak.pdf) | Host/GPU input, module extraction during elaboration, compiled offload incorporated into host code | HOST1 and GPU1 are distinct; displayed target branches are a zoom, not proof that all GPU codegen occurs after all host lowering |
| [NVIDIA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | cubin/SASS inspection through nvdisasm/cuobjdump | Disassembly branches from retained binary bytes; it is not a compulsory compilation stage |

ORC's lookup-triggered materialization and optional first-call laziness are
generic JIT concepts. The pinned MAX sources determine these pages' actual
compile/init/execute and cache routes. A JIT label does not mean every
`execute()` recompiles, nor that native code must be exported to disk first.

## Repository authority

SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
The [parent manifest](compiler_workbench_manifest.json) and
[example manifest](compiler_examples_manifest.json) contain exact symbols,
complete-definition spans and full-file hashes. Builders compare working-tree
bytes with the pinned Git revision before copying and embedding source files.

| Boundary / decision | Pinned repository authority |
| --- | --- |
| Package import and custom-op validation | `max/python/max/graph/graph.py`: KernelLibrary.load_paths / Graph.__init__; `graph/ops/custom.py`: custom |
| `.mojoc` retains source-level MLIR for specialization | `Mojo/docs/compiler/MojoCompilerWalkthrough.md`: packages/precompiled files |
| Caller graph retained, clone compiled, native graph-artifact cache | `GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp`: MOFContext::compileToBinary |
| MO → MAP → MOGG → MGP, later device finalization, EmitMojo, JIT, runtime calls | `GraphCompiler/lib/GraphCompiler.cpp` and linked dialect pipeline definitions |
| Compiled-kernel lookup, miss compilation and hit archives | `GraphCompiler/lib/MOGGDialect/Transforms/JitCompileKernels.cpp`: checkCache, handleCacheMisses, handleCacheHits |
| Compiled-load/call replacements and callable outlining | `MOGGDialect/Transforms/LowerCompiledRoutines.cpp`: runOnOperation; `GraphCompiler.cpp`: buildJitPipeline |
| Mojo semantic/lifetime checks, parametric KGEN/POP and elaboration | `Mojo/docs/compiler/MojoCompilerWalkthrough.md` |
| MLIR → LLVM module and backend selection | `Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp` |
| CPU object/assembly output and native linking | ObjectCompiler `Target/Host/HostBackend.cpp` |
| PTX → same-SM cubin via CUDA CompilationDevice | ObjectCompiler `Target/max_license/NVPTX/NVPTXBackend.cpp` |
| AMD object → linked ELF device code | ObjectCompiler `Target/max_license/AMDGPU/AMDGPUBackend.cpp` |
| Prepared bytes, init, module/callable handles, buffer results/readiness | Linked engine/runtime/driver definitions and original example call sites |

Graph-artifact and compiled-kernel caches are distinct. The teaching selector
uses compatible complete-hit routes: artifact hit bypasses graph lowering and
kernel compilation; kernel hit reuses code while retaining new graph/runtime
planning. Mixed kernel hit/miss sets and exact key contents are left in source.
Python MefStore/GraphSpec policy is a separate route, not the selected native
cache here.

## Continuous journey and focused experiments

The parent's 01 follows repository add-one from Python plus P1 package to G1,
C1 plan, KIR1 implementation, LLVM1 module, HOST1/GPU1, CM1 bytes, H1 handles
and M1. Two illustrative calls replace input/result buffers while preserving
the model/session/code. The six earlier scenes remain 02–07: graph contract,
graph lowering, Mojo specialization/ownership, target output, MEF bindings and
inference. Their fragment IDs are preserved.

Each example has the same connected 01 followed by graph/build, cold compile
and numerical kernel zooms. Shared choreography exposes source ancestry; each
example's unique mechanism comes from its original code:

| Example | Distinct experiment | Evidence / comparison boundary |
| --- | --- | --- |
| Add one | Typed load/add/store and new inputs using resident code | Original 5×10 float32 example; tiny groups/LLVM snippets are illustrations |
| Matmul epilogue | Accumulator → +1 → ×2 versus TMP1/TMP2; release intermediates and count accesses | Eligibility is source-backed. Tiny separate route counts 8 intermediate reads, 8 writes and 4 final writes; fusion counts 0/0/4. Input/cache/bus traffic is excluded |
| Tiled matmul | Authored naive/tiled compile-time selection; global/shared/register storage, copy wait, block barrier and reuse barrier | CPU chooses naive. GPU zeros output before +=. Full-K example, AMD tile 16 / otherwise 32. The helper creates buffers before graph and a new model per invocation |
| Online attention | Update m/l and rescale normalized O; reuse the same local S/E slots | Original unscaled rank-2 operation. Full-score storage is a hypothetical comparison, not the supplied kernel |

Attention's storage view counts score elements only: full N×N versus local
queryTile×keyTile. The original CPU example has 64 full-comparison versus 16
local scores; GPU has 1024 versus 128 (BN=16, key BN_1=8). These are shape
arithmetic, not measured memory usage. Q/K/V/O, exponent tiles and scratch are
excluded. The tiny one-row view uses four keys and two reusable score slots.

## Scope and validation

Source and IR views are folded sheets; code is a cartridge; CM1 is a cutaway
crate; buffers contain values; references are separately labeled handles.
Transforms occur in memory. Optional exports/disassembly retain their own
branches. The geometry, registers, command slots and motion timings are
schematic. There is no implied arbitrary-Python-to-Mojo translation.

No MAX/Mojo compilation, inference or accelerator measurement was performed.
This docs-only task used pinned source, independently checked browser arithmetic
and offline Chromium. Compiler inspection tools were not available on PATH.
[Validation](validation/COMPILER_JOURNEY_VALIDATION.md) records scoped commands,
checks and rendered review. Hosted demos were inspected as references; they
were not run as compiler evidence.
