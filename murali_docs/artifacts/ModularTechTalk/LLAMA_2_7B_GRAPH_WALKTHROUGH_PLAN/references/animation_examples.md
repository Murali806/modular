# Animation examples and reuse map

Choose the family that matches the requested mechanism. Paths are relative to
the skill/task directory. Read [the shared recipe](animation_authoring.md) for
the common design contract, and inspect the current authoring files alongside
the generated HTML. Existing validation/screenshot files show prior states;
they are not proof that a later working-tree change has been checked.

## Serve: movement, queues, selection and repeating work

Rendered references:

- [Driver, OS and scaling](../link_htmls/max_serve_os_driver_and_scaling.html)
- [Components and flow](../link_htmls/max_serve_os_components_and_flow.html)

Read [Serve requirements and source invariants](max_serve_os_animations.md)
when the lesson involves this domain. Its 01/02/03 opening applies to these
readers only. Preserve the existing stories and fragment links.

| Pattern to reuse | Specific files / entry points |
| --- | --- |
| Full-width panels, numbered cards, legend, source cards and colors | [serve_story.py](../src/serve_story.py), [serve_story.css](../src/serve_story.css), `FloatingCard` in [serve_story.js](../src/serve_story.js) |
| API roles sharing a loop, executor jobs, admission decisions and consequences | [serve_api.py](../src/serve_api.py), [serve_api_model.js](../src/serve_api_model.js), `ApiStory` in `serve_story.js` |
| Request/cancel/response slots, request sets, pass membership, KV ownership | [serve_worker.py](../src/serve_worker.py), [serve_worker_model.js](../src/serve_worker_model.js), [serve_worker.js](../src/serve_worker.js) |
| Candidate gates, seats, budgets, P/D inputs and pass history | [serve_batch.py](../src/serve_batch.py), [serve_batch_model.js](../src/serve_batch_model.js), [serve_batch.js](../src/serve_batch.js) |
| Busy, backpressure, slow-reader and replica consequences | [serve_story_model.js](../src/serve_story_model.js), `Story`, `renderSlots`, `renderLanes` and `renderHistory` in `serve_story.js` |
| Page assembly / reference atlas | [os_systems.py](../src/os_systems.py), [os_components.py](../src/os_components.py) |

Useful choreography: R1 enters an occupied IPC slot, drains into a worker-owned
request set, gains a per-pass selected reference, consumes visible budget/KV
capacity, produces output, and repeats while other requests wait. A full queue
changes the probe/send outcome. A slow reader can delay delivery after
generation has completed. Occupancy, request IDs and pass history agree with
the current event rather than merely flashing the active component.

Keep stored request contexts distinct from selected references. Keep admission
distinct from batch construction, context encoding from token generation,
generation completion from delivery, and freed per-request pages from retained
device allocations. A selected ID appearing twice does not create two requests.
Exact policies and guard order are in the Serve reference and pinned source.

For behavior changes, choose the relevant `src/check_serve_api.cjs`,
`src/check_serve_worker.cjs`, `src/check_serve_batch.cjs` or
`src/check_serve_stories.cjs`. For shared cards/layout/motion use
[check_serve_stories_browser.cjs](../src/check_serve_stories_browser.cjs), which
checks both readers, slots/reset, full width, distinct matching colors, source
cards, status backplates and the larger bold/tinted event styling. Use
`check_os_systems_browser.cjs`, `check_os_components_browser.cjs` or
`check_os_systems_lab.cjs` when their separate atlas/lab changes.

## InferenceSession: constructor, objects, buffers and lifetime

Rendered reference:
[max_inference_session_devices_runtime_and_compilation.html](../link_htmls/max_inference_session_devices_runtime_and_compilation.html).

- Builder/components/source targets: [inference_session.py](../src/inference_session.py)
- Page shell: [inference_session.html](../src/inference_session.html),
  [inference_session.css](../src/inference_session.css)
- State model: [inference_session_model.js](../src/inference_session_model.js)
- Controls/cards/source explorer: [inference_session.js](../src/inference_session.js)
- Persistent actor interpolation and target placements:
  [inference_session_visuals.js](../src/inference_session_visuals.js)
- Provenance: [inference_session_manifest.json](../inference_session_manifest.json)
- Checks: [check_inference_session.cjs](../src/check_inference_session.cjs),
  [check_inference_session_browser.cjs](../src/check_inference_session_browser.cjs)

Use `constructorFrames`, `compilationFrames`, `executionFrames` and
`lifetimeFrames` as the finite-timeline pattern. `InferenceSessionVisual`
keeps an actor map keyed by ID, samples the current
pose before retargeting, maps frames to resource slots, follows via paths and
supports reduced motion and freeze/resume. Inspect how the current visual
layer is included by the builder. If the checkout has only packet markers,
use the shared recipe and Serve request/slot renderers to construct persistent
object movement rather than treating the older renderer as the quality limit.

| Lesson | State worth making visible |
| --- | --- |
| Constructor | Argument device tiles, first-occurrence slots, duplicate merge, CPU addition, S1/native handle retention, compile configuration and runtime context creation |
| Compile/init | Symbolic Graph inputs, compile work, CompiledModel artifact, named registry bindings, missing-weight gate and initialized M1 |
| Repeat inference | Packed token and row-offset buffers for prefill/decode, input/completion slots, fresh results and the same S1/M1 retained across calls |
| Lifetime | Python-wrapper and native-ModelHandle reference edges, release-order branches and a session that survives while a model handle retains it |

Reuse the interventions as well as playback: supplying the missing W_b binding
opens initialization's gate; running another decode feeds sampled tokens into
fresh inputs while S1/M1 remain; independently releasing caller/decoder/sampler
owners detaches the corresponding references. These controls change the model
and visible resources. The device-work tiles represent schematic execution;
KV tiles represent token entries rather than page geometry, and A/B result
tiles are sequence views rather than a count of native output buffers.

Source invariants at the recorded revision:

- The constructor creates configuration/context and retains the selected
  devices. It does not compile the decoder or execute it.
- Device normalization preserves first occurrence and includes the host CPU.
  Mutual import/export options fail before native session construction.
- Python `InferenceSession`, native `Core::InferenceSession` and
  `Engine::Context` are different objects in the host process. `num_threads`
  does not prove a fresh private thread pool for every session.
- A Graph, uninitialized CompiledModel, weight provider/registry, initialized
  Model and per-call inputs/results occupy different lifecycle roles.
- Lifetime diagrams must follow the actual wrapper/native-handle retention;
  deleting a Python session name does not necessarily release the native
  session while model ownership remains.

Good contrast cases include repeated devices, CPU already listed, conflicting
MEF policy, missing binding and alternate release orders. GPU execution/timing
and illustrated token/reference counts keep their evidence qualifications.
See [recorded validation](../validation/INFERENCE_SESSION_VALIDATION.md) and
[constructor](../validation/inference_session_constructor.png),
[execution](../validation/inference_session_execution.png) and
[missing-binding](../validation/inference_session_missing_weight.png)
screenshots for the established layout. Inspect the
[device-work](../validation/inference_session_device_work.png),
[binding-repair](../validation/inference_session_repaired_weight.png) and
[owner-release](../validation/inference_session_lifetime.png) states for the
more detailed resource/ownership animation. Current source may contain later
improvements.

## Compiler workbench: transformations, artifacts and systems control

Rendered reference:
[max_mlir_mojo_compiler_and_runtime.html](../link_htmls/max_mlir_mojo_compiler_and_runtime.html).

Read [the compiler/systems reference](compiler_and_systems_animations.md) for
its research, design rules, source distinctions and exact builder/model/visual
files. Reuse the sheets, operation lineage, backend cabinet, cutaway crate,
owner/borrow handles and resident-code/per-call-buffer movement. Target choices,
text/binary/disassembly branches and prerequisite repairs change visible state.
The six lessons are the reference implementation, not a universal lesson order.

Its isolated builder is `src/compiler_workbench.py`; checks are
`src/check_compiler_workbench.cjs` and `src/check_compiler_workbench_browser.cjs`.
Use the scoped builder for this family when rebuilding every page would disturb
independent changes. The session reader has a visible link to the workbench.

### Concrete repository examples: arithmetic and compiler path

The workbench's `#examples` section links four standalone readers:

| Difficulty / reference | Actual Python / Mojo example | Observable mechanism |
| --- | --- | --- |
| [Simple: add one](../link_htmls/max_example_01_add_one_compilation_and_execution.html) | `custom_ops/addition.py`, `kernels/add_one.mojo` | Explicit output contract, registration-string linkage, typed `foreach` loads/results |
| [Medium: matmul epilogue](../link_htmls/max_example_02_matmul_fusible_epilogue.html) | `custom_ops/elementwise_addition.py`, `kernels/add_one.mojo` | Matmul accumulator with +1/×2 before store; compare separate temporary matrices |
| [Medium-hard: tiled matmul](../link_htmls/max_example_03_tiled_matmul_compiler_and_memory.html) | `custom_ops/matrix_multiplication.py`, `kernels/matrix_multiplication.mojo` | Compile-time algorithm, global/shared tile copies, copy readiness, block barriers and register accumulation |
| [Hard: online attention](../link_htmls/max_example_04_fused_attention_online_softmax.html) | `custom_ops/fused_attention.py`, `kernels/fused_attention.mojo` | Local scores, running maximum/denominator, rescaled normalized output and next-key-tile reuse |

Each reader has independent graph, compile/init and arithmetic entry points.
Reuse [compiler_examples.py](../src/compiler_examples.py),
[compiler_examples_model.js](../src/compiler_examples_model.js),
[compiler_examples_visuals.js](../src/compiler_examples_visuals.js),
[compiler_examples.js](../src/compiler_examples.js) and
[compiler_examples.html](../src/compiler_examples.html).
The visual subclass retains the workbench's actor map and interpolation;
[compiler_reader_ui.js](../src/compiler_reader_ui.js) shares transient cards
and the offline original-source explorer with the parent workbench.

For a new concrete example, preserve its real filename, shapes, parameters,
registration and call ordering. Label smaller numerical inputs separately.
Do not copy a generic lifecycle and merely rename it: add a scene showing the
example's arithmetic, resource changes or dependency guards. Keep running
kernel state inside its scope; input/device storage does not own attention's
local m/l/O accumulators.

At this revision, epilogue fusion is documented eligibility, not a captured
optimized dump. Tiled matmul uses full K tiles, zeros GPU output before `+=`,
and chooses BM/BN/BK=16 on AMD or 32 otherwise. Its helper compiles a new
model per invocation while main retains the session. Attention is unscaled
rank 2; GPU key tiles use BN_1=8 although query BN=16. Addition prints values
without asserting parity; attention prints Buffers without an explicit CPU
copy. All four call `init(compiled)` without a model-weight registry.

Build just these readers with `python3 src/compiler_examples.py`. Rebuild
`src/compiler_workbench.py` when changing the parent's progression cards.
Use [check_compiler_examples.cjs](../src/check_compiler_examples.cjs) for
independent tiny arithmetic and lifecycle invariants,
[check_compiler_examples_browser.cjs](../src/check_compiler_examples_browser.cjs)
for visible resource/motion/card behavior, and
[check_compiler_examples.py](../src/check_compiler_examples.py) for pinned
source bytes/spans and links. These checks do not execute MAX or Mojo.

### Legacy engine source family

The earlier engine HTML was retired on November 6, 2026. Its authoring/model
files and source manifest remain useful for MEF cache/reuse and runtime cases;
the new workbench is the rendered compiler reference. Do not regenerate the
retired page merely to inspect these semantics.

- Component/phase/event/source definitions: [engine_lifecycle.py](../src/engine_lifecycle.py)
- Shell and colors: [engine_lifecycle.html](../src/engine_lifecycle.html),
  [engine_lifecycle.css](../src/engine_lifecycle.css)
- Route filtering, terminal errors, replay/seek: [engine_lifecycle_model.js](../src/engine_lifecycle_model.js)
- Diagram, `drawEdge` port/gutter routing, cards and explorer:
  [engine_lifecycle.js](../src/engine_lifecycle.js)
- Provenance: [engine_lifecycle_manifest.json](../engine_lifecycle_manifest.json)
- Checks: [check_engine_lifecycle.cjs](../src/check_engine_lifecycle.cjs),
  [check_engine_lifecycle_browser.cjs](../src/check_engine_lifecycle_browser.cjs)
- Evidence: [ENGINE_LIFECYCLE.md](../validation/ENGINE_LIFECYCLE.md)

This source family separates session/Graph, graph compilation, Mojo, MEF, init and
inference into focused chapters. Reuse its explicit route-dependent states,
event/source mapping and ownership labels. For a new focused animation, combine
those semantics with the persistent actor/slot pattern from InferenceSession
when graph values, bindings, artifacts or runtime inputs can be drawn.

Preserve these distinctions when teaching the corresponding route:

- Symbolic Graph verification, compilation, MEF serialization/deserialization,
  weight binding/init and executable calls are separate stages. An
  uninitialized compiled artifact cannot infer; a virtual-device stand-in
  refuses execution.
- Native module/options MEF cache and Python `MefStore` are distinct. Direct
  `read(MEF)` bypasses a new Python Graph; GraphSpec reuse traces before its
  artifact lookup. A declined store key compiles; an accepted missing artifact
  key fails. Show each branch's actual downstream work and terminal gates.
- The pinned graph-lowering stages MO → MAP → MOGG → MGP and Mojo emission,
  JIT and runtime-call lowering describe compiler architecture. They are not
  measured timings or separate OS processes. Verify the stage names/order
  against source before applying them to another revision.
- Mojo Add registration is a real implementation building block. A graph
  operation count is not a GPU-kernel or device-launch count.
- The Python compile lock is a process-wide `threading.Lock` in this route.
  Native runtime/CPU scheduling requires its own evidence; do not invent
  per-phase thread IDs.
- Runtime dependencies use AsyncValue readiness and callbacks/worklists.
  The native path releases the GIL. Extracting results awaits runtime values;
  it does not establish a universal device-wide barrier or automatic host copy.

Useful cases: cold/cache, export/import/direct-read, declined/missing key,
pre-JIT branch, valid/bad-layout initialization and virtual refusal. Read the
actual source refs and model gates rather than inferring behavior from names.

## Memory: origins, arithmetic and owned capacity

Rendered reference:
[llama2_7b_memory_planning_and_memory_plan.html](../link_htmls/llama2_7b_memory_planning_and_memory_plan.html).

Read the memory-planning section in [agent.md](../agent.md). Authoring family:
[memory_planning.py](../src/memory_planning.py),
[memory_planning.html](../src/memory_planning.html),
[memory_planning.css](../src/memory_planning.css) and
[memory_planning.js](../src/memory_planning.js).
Use [check_memory_planning.cjs](../src/check_memory_planning.cjs).

Reuse the input-origin map, budget waterfall, page layout and parent/worker
handoff. Keep free bytes, provisional/capped/committed budgets, logical pages,
physical allocations and per-request ownership distinct. Preserve the frozen
MemoryPlan serialization boundary instead of showing shared Python identity
across spawned processes. Keep byte/page arithmetic assumptions explicit.

## Build and validation selection

From the repository root, build page changes with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
```

Run the matched `node src/check_<family>.cjs` from the task directory, then the
matched browser check for changed presentation/motion. Browser checks need
Playwright; discover an available installation rather than copying an old
`/tmp` path from a historical report. Finish page work with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py
```

For a new family, adapt checks around actual guards, ownership, reuse/failure
and the rendered resource changes. Do not copy another page's expected counts
or simply test that particular words exist. Use its real call-stack link.
Browser review covers desktop and narrow/touch layouts, intermediate motion,
cards/focus/Escape, matching colors, dynamic SVG backing, reset/pause/seek,
reduced motion, offline/no-JavaScript and print behavior as affected.

For instruction-only edits, validate skill frontmatter and changed local links
plus `git diff --check`. The linked validation reports document prior checks;
their combination/frame/contrast counts are not future quotas.
