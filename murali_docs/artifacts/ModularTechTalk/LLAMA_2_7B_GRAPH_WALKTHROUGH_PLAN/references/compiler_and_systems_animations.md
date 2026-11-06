# Compiler and systems animations

Read this reference for Python/MAX graphs, MLIR, Mojo, native code generation,
MEF preparation, buffers, drivers or systems-language concepts. Use it with the
[shared animation recipe](animation_authoring.md). It captures the compiler
workbench requested on November 6, 2026; the lesson count and palette are an
example, not requirements for every future topic.

## Start from the accepted implementation

Open [the compiler workbench](../link_htmls/max_mlir_mojo_compiler_and_runtime.html).
Inspect these current authoring files before inventing a simpler replacement:

| Responsibility | Files |
| --- | --- |
| Source spans, owners, legend, scene geometry, isolated build | [compiler_workbench.py](../src/compiler_workbench.py) |
| Page and workbench theme | [compiler_workbench.html](../src/compiler_workbench.html), [compiler_workbench.css](../src/compiler_workbench.css) |
| Frames, prerequisites, interventions, target routes | [compiler_workbench_model.js](../src/compiler_workbench_model.js) |
| Persistent file/IR/code/handle/cell actors and motion | [compiler_workbench_visuals.js](../src/compiler_workbench_visuals.js) |
| Playback, seek, cards, target controls, original-source explorer | [compiler_workbench.js](../src/compiler_workbench.js) |
| Behavioral/source and rendered checks | [check_compiler_workbench.cjs](../src/check_compiler_workbench.cjs), [check_compiler_workbench_browser.cjs](../src/check_compiler_workbench_browser.cjs) |

The [research decisions](../COMPILER_WORKBENCH_RESEARCH.md) and
[source manifest](../compiler_workbench_manifest.json) identify the authority
for this implementation. Recheck routes against the requested revision.

## Research changes the animation, not just the prose

Use primary compiler documentation and the actual MAX/Mojo implementation.
Inspect a visual reference's real interaction before describing it as an
animation. A repository with a demo GIF, a pass editor and a blog hosting
animated slides provide different kinds of inspiration.

Useful patterns from the reviewed references:

- Modular's LLVM 2023 presentation: progressively reveal compiler semantics.
- `mlir-interactive`: compose passes and inspect intermediate representations.
- VizMLIR: follow an operation through creation, fusion, lowering and erasure.
- MLIR Toy: partial lowering can keep several dialects in the same module.

Record which visual decision each reference informed. Generic MLIR GPU examples
do not establish the precise MAX backend or its tool invocation.

## Representation lineage is the main actor

Draw a sheet containing operations/types, transform it through the workbench,
and preserve its identity or its relationship to a new representation. Keep
the caller-owned graph while the compiler-owned clone changes. Pair the current
IR view with its predecessor and show small operation tiles that group, gain
buffer/dependency operands, become calls, or move into a symbol-DCE bin.

Use file folds for IR/source views, a cartridge for code bytes, a cutaway crate
for serialized ingredients, a tool cabinet for backend choice, allocation cells
for data and separate handle cards for references. These forms should explain
what changed, not provide unrelated decorative motion. A light drafting/workbench
theme is an accepted variation when physical artifact shapes help this topic.

Keep these distinctions visible:

| Concept | Consequence to draw |
| --- | --- |
| Python model-building code | Records a MAX symbolic graph; do not imply arbitrary Python automatically becomes Mojo |
| MLIR dialect | Vocabulary in an IR module; several dialects can coexist |
| Lowering pass | Changes selected operations/types/structure; not necessarily a new disk file or OS process |
| LLVM dialect → LLVM IR | Translation creates a distinct LLVM module after MLIR lowering |
| Optional textual dump | A view/export of IR, not a compulsory saved file at every stage |
| Assembly/object/binary | Target-dependent outputs with distinct roles |
| Disassembly | Inspection of retained binary bytes, not a required compiler stage |
| Serialized MEF | Prepared artifact bytes, separate from initialized Model and registry bindings |

The workbench IR, bit patterns and MEF cutaway are explicitly illustrative. If
using actual dumps, retain their source/configuration and do not silently mix
illustrations with generated output.

## Finalization has several meanings

In the pinned source, `Graph.output` completes and verifies the symbolic graph's
output contract. MAX's later device-graph finalization occurs within MGP work.
Kernel JIT and final runtime-call lowering are later still. Give these different
milestones their own visible state; one “compile complete” light loses the
distinction the user wants to understand.

At the workbench revision, the MAX route is:

```text
caller G1 retained → compiler clone C1
MO → MAP → MOGG → MGP
MGP buffers/primitives/dependencies + device-graph finalization + symbol DCE
→ EmitMojo → kernel JIT → final runtime-call lowering
```

Do not turn an illustrative fusion switch into a documented MAX CLI option,
or assume that two graph operations produce exactly two kernels or launches.

## Explain Mojo through control over concrete resources

Avoid promising a blanket “C++ performance” result. Show why readable syntax
can compile to typed native operations, using one mechanism at a time:

- **Specialization:** keep WIDTH parametric through IR, elaborate concrete
  functions/types, then regroup the same values into native SIMD bundles.
  Keep outputs identical while grouping changes. Lane/instruction counts are
  schematic unless compiled or measured; SIMD lanes are not GPU thread IDs.
- **Ownership:** one B1 allocation, separate owner and borrow handles, dashed
  references, a conflicting access blocked by lifetime checking. Ending the
  borrow changes the program before compilation; it does not bypass the gate.
  A write updates cells in the same B1. Moving the owner transfers its handle,
  without an inherent copy of the underlying data.
- **Low-level layouts/pointers:** if requested, draw element placement and
  addressing explicitly. Separate safe typed access, an unsafe operation's
  programmer obligation and hardware synchronization.

The reviewed compiler walkthrough separates LIT semantic/lifetime checks,
parametric KGEN/POP, elaboration, concrete lowering and LLVM code generation.
Compile-time lifetime legality and runtime AsyncValue readiness solve different
problems. Use separate gates and labels.

Use an unchanged source implementation in the explorer. If creating runnable
Mojo examples, apply the available Mojo syntax/GPU skills and verify the code;
an illustrative source card is not proof of execution.

## Target choice must change the resulting artifacts

Show the target triple, architecture/features and selected backend. The CPU
compiler host and execution device are different roles. GPU cross-codegen can
avoid physical GPU execution, but still requires target/toolchain support.

Pinned example routes:

```text
CPU:    LLVM MC → relocatable .o → host linker → native .so
NVIDIA: LLVM NVPTX → PTX text → CUDA CompilationDevice → cubin
AMD:    LLVM AMDGPU → .o → ELF/device-library link → hsaco
```

Assembly text can be emitted instead of object output. Do not require a saved
`.s` before `.o`; LLVM MC can emit object bytes directly. PTX is virtual-ISA
text, not SASS. The NVIDIA implementation uses a CUDA compilation API/default
or configured override; do not invent a mandatory `ptxas` subprocess. Internal
compiler tools in repository documentation are not necessarily exposed by an
ordinary installed package.

Use a separate read-only inspection branch. NVIDIA cubin inspection uses
`nvdisasm`/`cuobjdump`; LLVM object inspection applies where supported. Keep
the binary actor present. Unsupported arch or unavailable compilation support
must stop the binary/initialization route visibly.

## Code stays resident while calls exchange data

After init, keep S1, M1, code and weights with stable IDs. New calls introduce
fresh input/result buffers and dispatch references. A queued descriptor refers
to code/buffers; it is not another copy of the code file.

Animate a data transfer only for an explicit copy route. Already-resident input
avoids upload. CPU reference views must not resemble host-to-GPU transfers.
Result readiness does not inherently mean a device-wide barrier or a host copy.
Keep runtime dependency readiness, driver submission and physical hardware
execution distinct; tiny command slots and device timing remain schematic.

Show MEF ingredients gathering, optional export, named weight bindings and the
init gate. A missing weight holds M1 unavailable until supplied. The crate is
a conceptual cutaway, not a specification of MEF's byte layout.

## Verify the visible mechanism

Use the isolated compiler builder when only these pages change:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_workbench.py --refresh-session
node src/check_compiler_workbench.cjs
NODE_PATH=<available-playwright-node_modules> node src/check_compiler_workbench_browser.cjs
```

`--refresh-session` rebuilds its accepted reader/link; omit it for compiler-only
changes. Use the full builder only when its broader rebuild is appropriate.
The legacy engine page was retired; its source map still provides pinned spans.

Review intermediate frames and the actual motion, as well as all target/error
routes. Verify the same actor DOM node moves, Pause freezes it, reset/seek are
deterministic, and resident code is reused. Fit text by actual rendered width,
not character-count guesses; short cards need fewer lines. Audit text on its
own paper/card backing and ensure later opaque actors do not cover labels.

Keep hover/focus/touch explanation cards, click-only circled L, stable legend
colors/types, offline full original files, reduced motion and narrow-screen
internal scene scrolling. Qualify browser teaching-model checks separately
from real compiler, MAX or accelerator execution.
