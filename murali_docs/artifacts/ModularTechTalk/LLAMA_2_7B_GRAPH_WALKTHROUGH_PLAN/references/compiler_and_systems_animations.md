# Compiler and systems animations

Read this reference for Python/MAX graphs, MLIR, Mojo, native code generation,
MEF preparation, buffers, drivers or systems-language concepts. Use it with the
[shared animation recipe](animation_authoring.md). It captures the compiler
workbench requested on November 6, 2026 and the researched operation journey
approved on November 7, 2026. Lesson counts and palettes are accepted examples,
not requirements for every future topic.

## Start from the accepted implementation

Open [the compiler workbench](../link_htmls/max_mlir_mojo_compiler_and_runtime.html).
Inspect these current authoring files before inventing a simpler replacement:

| Responsibility | Files |
| --- | --- |
| Connected operation journey: source/scene + deterministic state | [compiler_journey.py](../src/compiler_journey.py), [compiler_journey_model.js](../src/compiler_journey_model.js) |
| Journey ancestry selection, stable actors and live controls | [compiler_journey_visuals.js](../src/compiler_journey_visuals.js), [compiler_journey.js](../src/compiler_journey.js), [compiler_journey.css](../src/compiler_journey.css) |
| Source spans, owners, legend, scene geometry, isolated build | [compiler_workbench.py](../src/compiler_workbench.py) |
| Page and workbench theme | [compiler_workbench.html](../src/compiler_workbench.html), [compiler_workbench.css](../src/compiler_workbench.css) |
| Frames, prerequisites, interventions, target routes | [compiler_workbench_model.js](../src/compiler_workbench_model.js) |
| Persistent file/IR/code/handle/cell actors and motion | [compiler_workbench_visuals.js](../src/compiler_workbench_visuals.js) |
| Playback, seek, cards, target controls, original-source explorer | [compiler_workbench.js](../src/compiler_workbench.js) |
| Shared transient cards / offline full-source explorer | [compiler_reader_ui.js](../src/compiler_reader_ui.js) |
| Behavioral/source and rendered checks | [check_compiler_workbench.cjs](../src/check_compiler_workbench.cjs), [check_compiler_workbench_browser.cjs](../src/check_compiler_workbench_browser.cjs) |
| Shared journey routes and rendered behavior | [check_compiler_journey.cjs](../src/check_compiler_journey.cjs), [check_compiler_journey_browser.cjs](../src/check_compiler_journey_browser.cjs) |

The [research decisions](../COMPILER_WORKBENCH_RESEARCH.md) and
[source manifest](../compiler_workbench_manifest.json) identify the authority
for this implementation. Recheck routes against the requested revision.

For an example-driven walkthrough, use the four concrete readers and their
arithmetic/model family in [the example catalog](animation_examples.md#concrete-repository-examples-arithmetic-and-compiler-path).
They extend the workbench with the actual custom-op registration, parameters,
tile coordination and numerical transformations. Preserve their distinction
between source-proven APIs and illustrative optimized IR/hardware motion.

## Plan a connected explanation before the focused zooms

For a full compiler example, retain one operation's ancestry across preparation
and execution. The current parent has one connected 01 and six zooms; examples
have connected 01 plus three zooms. Keep fragment IDs when renumbering.

Use two kinds of continuity: a representation may retain its ID as it changes,
or a new representation may inherit provenance from an older one. A graph op
becoming multiple loads/arithmetic/stores does not mean those are one object.

| Identity | Visible role / lifetime |
| --- | --- |
| P1 | Parametric implementation package imported before custom-call construction |
| G1 | Caller-owned verified graph, retained while compilation rewrites its clone |
| C1 | Compiler-owned graph/execution plan; MGP survives later kernel lowering |
| KIR1 | Mojo implementation IR, including parametric and concrete forms |
| LLVM1 | Translated `llvm::Module`; distinct from MLIR's LLVM dialect |
| HOST1 / GPU1 | Native host routines / compiled device offload; not input tensors |
| CM1 | Prepared artifact bytes; separate from loaded modules and initialized model |
| H1 / M1 | Resolved callable/module handles / initialized executable in S1's context |
| B1 / JOB1 / O1 | Per-call input args, code/buffer references and result; fresh IDs next call |

Let readers select operation, load/store ancestry or graph/readiness plan.
Show the selection on the actual actors, include keyboard activation and use
Next change to skip unrelated steps. A caption selector that leaves the same
ambiguous highlights is insufficient. Before/after panes should include
operand/type/control-flow fragments at the chosen level, labeled schematic
unless they are captured dumps. Keep source ancestry collapsed but immediately
reachable through the original-file explorer.

The timeline needs a preparation/execution clock, stage navigation and visible
cache consequences. Repeat execution with retained context/model/code and new
inputs. The journey's second call is illustrative: do not silently rewrite an
example that constructs/compiles a fresh model per helper invocation.

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
- LLVM instruction/codegen references: follow typed values, selection,
  virtual registers, allocation and encoding into target bytes.
- Bufferization/transform references: separate tensor values, storage aliases,
  transformations and their payload; count the resources a comparison changes.
- JIT/ORC references: distinguish generation, symbol resolution/materialization,
  optional lazy first-call behavior and subsequent native calls.
- Mojo compiler-author talks: preserve structured information early and keep
  host/device offload artifacts separate; GPU extraction can occur in elaboration.

Record which visual decision each reference informed. Generic MLIR GPU examples
do not establish the precise MAX backend or its tool invocation. Read the
[research matrix](../COMPILER_WORKBENCH_RESEARCH.md) and choose the references
needed for the requested mechanism. Historical talks and blogs do not establish
current pass order, tool availability or measured performance.

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
| `.mojoc` package | MLIR bytecode / implementation bodies for deferred specialization, not final machine executable |
| Graph plan versus kernel IR | Keep C1's resources/dependencies visible while KIR1/LLVM1 changes |

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

Show `ModuleLoad` / `ExecuteGeneric` and callable outlining as finishing the
runtime interface after compiled-kernel work. Prepared bytes and loaded callable
handles remain distinct. Initialization is a readiness gate, not another graph
trace or a numerical inference call.

## Show exactly what each cache bypasses

Native graph-artifact lookup in `MOFContext::compileToBinary` differs from
compiled-kernel lookup in `JitCompileKernels`. The current selector uses compatible
complete-hit illustrations, with source links for the more complex policies:

| Route | Skip | Work that remains |
| --- | --- | --- |
| Cold | Nothing in the selected preparation path | Graph lowering, kernel specialization/codegen, runtime-call interface, artifact, init and calls |
| Kernel hit | Fresh compilation of cached routines | Graph/execution planning, archive loading, runtime-call interface, artifact and init |
| Graph artifact hit | New graph lowering and kernel compilation | Artifact retrieval, runtime/device initialization and calls |

Do not conflate these with Python MefStore/GraphSpec policies. JIT describes
when code is prepared, not a guarantee that MAX recompiles at the first or
every `execute`. Generic ORC lazy compilation explains a comparison only unless
the pinned MAX source establishes it.

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

## Each example needs its own resource mechanism

Use the original repository entry/registration/dispatch/arithmetic, not just a
renamed generic compile scene. The accepted four examples progress through:

- Addition: dtype/width, loads, native +1, stores and code reuse with new inputs.
- Epilogue: accumulator plus +1/×2 before store versus materialized TMP1/TMP2.
  Draw producer/consumer edges and lifetime endings. The tiny comparison counts
  intermediate scalar reads/writes; exclude A/B loads and measured bus traffic.
  Fusion is documented eligibility, not a captured optimization result.
- Matmul: authored compile-time naive/tiled selection, global → shared copies,
  commit/wait, block consumption barrier, register MAC and reuse barrier.
  Do not claim tiling was invented by a generic compiler pass. CPU uses naive;
  original GPU output is zeroed before +=; buffers precede graph construction.
- Attention: local scores, changing maximum, denominator and rescaled normalized
  output; release/reuse S/E slots while m/l/O remain. Compare full N×N score
  storage with queryTile×keyTile, explicitly excluding inputs/output and scratch.
  Reuse allocation IDs across key tiles; generation numbers are contents, not
  new physical slots. The full-score route is a hypothetical comparison.

The original attention example is unscaled rank 2 with no mask/dropout/head
axes. GPU BN=16 is the query tile, BN_1=8 is the key tile, and BD=8 is a feature
tile. Do not multiply the wrong dimensions when drawing score storage. Always
separate full source dimensions from the tiny arithmetic chosen for readability.

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
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_examples.py
node src/check_compiler_journey.cjs
NODE_PATH=<available-playwright-node_modules> node src/check_compiler_journey_browser.cjs
node src/check_compiler_workbench.cjs
NODE_PATH=<available-playwright-node_modules> node src/check_compiler_workbench_browser.cjs
node src/check_compiler_examples.cjs
PYTHONDONTWRITEBYTECODE=1 python3 src/check_compiler_examples.py
NODE_PATH=<available-playwright-node_modules> node src/check_compiler_examples_browser.cjs
```

`--refresh-session` rebuilds its accepted reader/link; omit it for compiler-only
changes. Use the full builder only when its broader rebuild is appropriate.
The legacy engine page was retired; its source map still provides pinned spans.
Rebuild parent and examples after shared journey edits. Use only the relevant
family checks for a narrower change; the list is the full shared-change scope.

Review intermediate frames and the actual motion, as well as all target/error
routes. Verify the same actor DOM node moves, Pause freezes it, reset/seek are
deterministic, and resident code is reused. Fit text by actual rendered width,
not character-count guesses; short cards need fewer lines. Audit text on its
own paper/card backing and ensure later opaque actors do not cover labels.

Keep hover/focus/touch explanation cards, click-only circled L, stable legend
colors/types, offline full original files, reduced motion and narrow-screen
internal scene scrolling. Qualify browser teaching-model checks separately
from real compiler, MAX or accelerator execution.
