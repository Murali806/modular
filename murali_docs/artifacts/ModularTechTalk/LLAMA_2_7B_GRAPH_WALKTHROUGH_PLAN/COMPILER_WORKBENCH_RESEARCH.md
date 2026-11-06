# Compiler workbench: research and design

Implemented November 6, 2026. [Open the six animations](link_htmls/max_mlir_mojo_compiler_and_runtime.html).
The accepted [InferenceSession reader](link_htmls/max_inference_session_devices_runtime_and_compilation.html)
retains its four lessons and links to the new workbench.

## What the external references contributed

The following primary pages/READMEs were fetched successfully during this task.
The browser's web fetch failed DNS resolution; ordinary HTTPS retrieval was used
to inspect the pages. No external script, hosted animation or font is needed to
open the delivered offline page.

| Reference | Verified content | Applied design |
| --- | --- | --- |
| [Modular LLVM 2023 presentation](https://www.modular.com/blog/mojo-llvm-2023) | The author describes an interactive presentation with animations and clickable links | Progressive semantic views; compact interactive source access |
| [mlir-interactive](https://github.com/mortbopet/mlir-interactive) | README describes a Qt UI for composition/execution of MLIR flows; links a demo GIF | Pass-by-pass views and retained intermediate representations |
| [VizMLIR](https://github.com/joepotibutr/vizmlir) | README describes following one source line/op through passes that create, fuse, lower or erase it | Operation provenance, retained graph versus compiler clone, visible DCE |
| [MLIR Toy, chapter 5](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/) | Partial lowering to lower-level dialects | Do not imply a module always contains exactly one dialect |
| [NVIDIA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | Documents cubin, cuobjdump and nvdisasm; SASS/ELF/PTX inspection | Binary-to-disassembly as an inspection branch |

Additional primary reading links are included in the HTML for MLIR LLVM
translation/GPU serialization and LLVM NVPTX/objdump. Generic tutorials explain
the mechanisms; they do not establish the exact MAX pass pipeline. The animated
slides/demo themselves were not independently executed as compiler evidence.

## Source authority

SOURCE revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
The [manifest](compiler_workbench_manifest.json) records 37 complete source spans
and hashes for 19 full original files. The builder compares each against Git's
pinned revision and the current repository before embedding it.

| Decision | Repository authority |
| --- | --- |
| Graph construction/output finalization | `max/python/max/graph/graph.py`, Graph constructor and Graph.output |
| Preserve caller module; lower a clone | `GraphCompiler/lib/FrameworkFrontend/FrameworkFrontend.cpp`, compileToBinary |
| MO/MAP/MOGG/MGP, symbol DCE, EmitMojo/JIT/runtime calls | `GraphCompiler/lib/GraphCompiler.cpp` |
| Mojo LIT checks, KGEN/POP, elaboration, LLVM | `Mojo/docs/compiler/MojoCompilerWalkthrough.md` |
| Concrete registered Add routine | `max/kernels/src/graph_compiler/builtin_kernels/elementwise.mojo` |
| Target registry and MLIR→LLVM translation | `Mojo/lib/Compiler/ObjectCompiler/ObjectCompiler.cpp` |
| Direct object emission/link or assembly | `Target/Host/HostBackend.cpp` under ObjectCompiler |
| PTX→same-SM CUDA CompilationDevice→cubin | `Target/max_license/NVPTX/NVPTXBackend.cpp` |
| AMD object/ELF link inheritance | `Target/max_license/AMDGPU/AMDGPUBackend.cpp` |
| MEF serialization, binding/init, execution/readiness | Existing engine lifecycle definitions, now linked from this workbench |

The stage-dump/cross-target repository tests were inspected, not executed.
The source does not require a saved file at every IR stage or universally spawn
`ptxas`. Graph.output finalization and later device-graph finalization are separate.

## Six focused scenes

| Lesson | Moving objects | Reader's experiment / visible consequence |
| --- | --- | --- |
| 01 Graph | Python sheet, G1, ADD1/MUL1, output signature, verified stamp | Trace only leaves the contract open; finalization seals it while hardware stays idle |
| 02 Dialects | G1 retained, C1 clone, operation/routine tiles, snapshot tray, symbol bin | Fusion changes grouping; MGP gains dependency/reference tiles; DCE removes a symbol |
| 03 Mojo | Source/IR sheets, WIDTH parameter, value lanes, B1/owner/borrow | SIMD grouping changes with identical outputs; conflicting borrow blocks, repaired lifetime permits write/move |
| 04 Target | LLVM module, backend/tool cabinet, PTX/object/binary, inspection sheet | CPU/NVIDIA/AMD produce different artifacts; text-only branch stops; tool/arch failures prevent binary creation |
| 05 MEF | Same code/plan actors enter a cutaway crate; export sheet; registry/model dock | Missing W_b keeps M1 unavailable; supplying it permits init independently of optional MEF export |
| 06 Execute | Resident S1/M1/K1/W1, new input/result cells, dispatch descriptor | Resident input avoids upload; explicit output copy is separate; another call reuses code/context |

The refreshed light blue-grey drafting surface uses folded sheets, binary pins,
tool handles, cutaway crates and buffer cells. Motion follows named resources;
it does not claim realistic hardware timing or byte-accurate physical geometry.

## Scope and verification

Python records a MAX model graph; the page does not promise compilation of
arbitrary Python into Mojo. Its `(A + 10) × 2` expression, IR previews, filenames,
fusion, bit patterns, lane counts, MEF ingredients and command slots are teaching
illustrations. The first graph view uses symbolic B/C for those constant inputs.
Native drivers and physical hardware activity are BOUNDARY illustrations.

CPU-RUN verification exercises the JavaScript state model, original-file
embedding and Chromium interactions/motion/layout. No MAX model compilation,
Mojo example compilation, inference, GPU run, benchmark or profiler capture was
performed. Commands/results are recorded in [validation](validation/COMPILER_WORKBENCH_VALIDATION.md).
