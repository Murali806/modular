# Authoring animations with the accepted level of detail

Use this recipe for animation HTML in this artifact. The examples and exact
source files are in [animation_examples.md](animation_examples.md). The
[Serve reference](max_serve_os_animations.md) holds domain-specific Serve rules;
[agent.md](../agent.md) holds main-guide and memory-planning conventions.

These are local teaching requirements and implementation patterns, extracted
from the accepted pages and their current sources on November 6, 2026. The
current user request can revise the teaching design. Source behavior must be
rechecked when the pinned revision changes.

## 1. Turn the question into observable state

Start with the named component's entry conditions, inputs, actions, outputs,
owner and lifetime. Break a large subject into focused lessons whose scenes
answer a question through state changes. Select the lesson count from the
subject; only the two Serve readers have the fixed 01 API / 02 worker /
03 constructor opening.

Before implementing, make a compact transition plan:

| Moment | Actor/resource to draw | Visible consequence | Evidence |
| --- | --- | --- | --- |
| Input arrives | Named handle, request or buffer in an input slot | Slot becomes occupied; input labels remain readable | Entry/call-site span |
| Guard runs | Candidate beside a gate, budget or policy switch | Accept, wait, merge, reject or stop reason appears beside it | Actual condition and order |
| Handoff occurs | Labeled message, reference edge or data tiles | Destination gains the reference/data; source state changes only as warranted | Caller and callee |
| Work completes | Result/artifact/pass row | New object, readiness flag, counter and history agree | Return/completion path |
| Work repeats or owner releases | Same IDs, fresh per-call inputs or disappearing reference edge | Reuse and surviving ownership are visible | Loop/destructor path |

For every important step, identify what changes on the canvas beyond the event
sentence. When a step is purely a control decision, show the gate state and its
consequence, such as a retained waiting candidate or a disabled next stage.
This plan is a working aid, not an extra document the user must approve.

Select meaningful contrast cases supported by the code: cache hit versus cold
compile, enough versus insufficient capacity, duplicate versus unique devices,
ready versus waiting, valid versus missing bindings. A scenario control changes
the state/route/consequence; changing its caption alone adds little teaching
value. A single narrow mechanism does not need unrelated failure scenarios.

## 2. Draw owners, actors and resources separately

Maintain a component table with `id`, label, kind, owner, role, color, source
references and lifetime. Use it to keep diagram labels, legend and evidence
consistent. Python objects, native C++ objects, IR, files, OS processes,
threads, channels and hardware have different kinds even when connected.

Nest components inside the actual owner. Label boundary types directly, for
example `MODEL WORKER · OS PROCESS` enclosing a solid native-object boundary.
Compiler stages do not become separate processes merely because they occupy
separate columns. T1/T2/T3 are the accepted badges in the Serve API lesson
only; a runtime thread option elsewhere is an option, not evidence of three
dedicated threads or a private per-session pool. Use `BOUNDARY` for unresolved
native scheduling or driver behavior.

Draw the moving or changing subject as an actor with a stable identity:

- Requests: R1/R2 with lifecycle state, current owner and relevant token count.
- References: the referent ID plus an explicit reference edge or `ref(R1)`
  marker; the object stays with its owner when only a reference is copied.
- Buffers: labeled token/row-offset/weight tiles, shape or count and host/device
  location. New call inputs have fresh identities; a reused model keeps M1.
- Artifacts: a symbolic graph, compiled artifact and initialized model get
  distinct IDs/kinds. A result of compilation is created rather than implying
  the Python graph physically travels into a file.
- Resources: queue seats, batch seats, registry bindings, devices and KV pages
  have visible occupancy, ownership and capacity.

Preserve identity across steps. Retain the actor node keyed by ID and move it
between defined positions where movement represents a real handoff. When
rebuilding a slot grid is appropriate, preserve the ID-to-slot mapping and
interpolate the labeled actor over it. Avoid clearing and respawning all
actors each frame. Distinguish releasing a reference, destroying an object,
reusing capacity and transferring data.

A generic traveling dot can mark a control call or an unknown native boundary.
For an explained queue, binding, selection or buffer transfer, draw its named
payload and destination state as well. A progress bar can indicate illustrative
work in progress; it does not replace the resulting artifact/resource change.

Use an appropriate resource view rather than applying queue slots everywhere:

| Subject | Useful visual form |
| --- | --- |
| IPC/backpressure | Empty/occupied message slots, count/capacity, send/probe gate and draining |
| Batch selection | Candidate cursor, gate reasons, seats, input/context budgets, KV ownership and pass history |
| Constructor | Incoming argument/device slots, duplicate merges, retained native handles/config/context |
| Compilation/init | Symbolic inputs, phase states, artifact slot, named weight-binding slots and readiness gates |
| Inference | Input-buffer tiles, dependency readiness, submission/completion, per-call results and reused model |
| Lifetime | Wrapper and native-handle reference edges, surviving referents, release-order branches |
| Memory planning | Input-origin map, budget waterfall and owned/free/reserved page regions |

Tiny capacities, token IDs, durations and layout sizes are teaching settings.
Keep that qualification near the scene or its card. Neither depicted motion
nor a fixed message-slot count proves measured hardware timing or exact ZeroMQ
high-water-mark enforcement.

### Example: a constructor scene that explains the call

For a GPU-only argument list `[GPU0, GPU0]`, plan visible consequences such as:

| Transition | Canvas change |
| --- | --- |
| Arguments ready | Two labeled incoming references; normalized/config/context slots empty |
| Keep first device | First GPU0 reference occupies normalized slot 0 |
| Skip duplicate | Second GPU0 reference converges into the first-occurrence marker; distinct count stays 1 |
| Ensure host CPU | CPU reference appears in normalized slot 1; distinct count becomes 2 |
| Enter native constructor | Native S1 appears; registration slots gain retained references to the same devices |
| Populate configs/create context | CFG and CTX objects appear in their respective native ownership regions |
| Return | Pipeline retains S1; decoder/artifact count remains 0 and hardware shows no execution |

Check the current source for the exact option/config ordering and guard paths.
The scene shows references, not a physical relocation of GPU hardware. An
artifact-policy conflict follows its source early-exit point, leaving the
native S1/config/context slots empty. This same approach applies to a batch
candidate or MEF route: each decision changes a concrete visible state.

## 3. Model semantics before interpolating movement

Reuse the closest deterministic model family. Keep scenario options and domain
state independent from DOM operations and animation timing. Two existing
patterns work well: precomputed immutable frames for finite lifecycles, or a
stateful scheduler model with reproducible transitions and snapshots.

A useful frame/snapshot contract contains the current phase, event title,
source refs, created/retained objects, actors/placements, resource occupancy,
budgets/readiness, branch outcome, history, error/terminal state and step index.
Use only fields needed by the lesson. A model's `snapshot()` must not expose
mutable state that rendering can accidentally change.

Make guards match source order. Terminating failures stop downstream work;
waiting is distinguishable from rejection; successful reuse skips only the
work that its route actually bypasses. Counters, occupied slots, pass history,
event text and enabled controls all derive from the same semantic snapshot.

Map each snapshot to target positions separately from the transition logic.
Use stable owner/slot anchors and labeled paths. During playback, interpolate
from the actor's current pose to its new target with `requestAnimationFrame`.
Sample the current pose before a new transition to avoid teleporting during
rapid controls. Stagger only when it teaches ordering; do not suggest source
concurrency through decorative simultaneous movement.

Support Play/Pause, Step and Reset. Timeline lessons should also provide Back
or seek plus an index; looping simulations can retain their existing Step/
Replay controls. Scenario changes reset/recompute their model and clear stale
history, actors and errors. Reset must restore slots, references, counters,
bindings and labels as well as the moving marker.

Pause both semantic stepping and visual interpolation. Resume from the frozen
pose without charging hidden elapsed time. Reduced motion shows the same
states immediately; no autoplay. Pause offscreen, on document hiding, and when
the containing disclosure closes. Stop scheduling frames when idle. A scoped
read-only snapshot bridge lets browser checks compare visible state with the
model without exposing mutation hooks in the product UI.

## 4. Use the established article and card shell

Reuse the accepted light article shell: warm off-white background, restrained
green navigation, serif headings, compact sans-serif labels, monospace IDs and
controls, lightly tinted component boxes and thin colored outlines. Exact
palette values live in the matched CSS, not in a newly invented theme. The
main QnA guide keeps its separate dark-grey convention.

Give each lesson a full-width bordered animation panel. Keep the scene,
compact controls, current event and counters close together. Put long detail
in transient cards, the event transcript or collapsed source/reference cells.
Do not bring back the persistent right explanation column or shrink the canvas
to make room for paragraphs.

The numbered lesson badge lives inside the panel. Show its card on pointer
hover or keyboard focus, preserve it while the pointer/focus moves into the
card, and dismiss when both are left. Support touch toggle, Close and Escape.
Clamp the card to the viewport, allow internal scrolling, and restore focus
after keyboard dismissal without immediately reopening it.

The slightly larger circled **L** stays at the right edge of the sticky menu.
Its legend opens on click/tap/keyboard activation, not hover. Outside click,
Close or Escape dismisses it. Opening it must keep the current scene in place.
Each row includes a unique component swatch, role, kind/owner, applicable
IPC/lock/readiness method and source link. Uvicorn's Server can share one row
with Uvicorn because it is that server object; do not create a fictional server
process to satisfy a list of words.

Keep component colors stable across the page and legend. Requests/buffer IDs
and state labels must remain readable without relying on color alone. Use
separate lifecycle styling, shapes or explicit labels for ready/waiting/error.
Show real synchronization scope, such as a lock surrounding probe/send, and
its release before a later await; a KV resource owner is not a mutex.

Make the current action easy to notice: larger bold text and a tinted backing
fitted to its content. For SVG status plates, fit the backing to `getBBox()`
with padding, following existing helpers. Test changing/long/error messages
against the actual drawn background; readable CSS colors alone do not prove
SVG contrast or that labels fit.

Route handoff arrows between meaningful ports, using free gutters instead of
crossing text or unrelated objects. Label call, reference, IPC, submission and
completion edges distinctly. An event loop schedules ready tasks; it is not a
physical data-hop station. On narrow viewports, keep readable diagram sizing
and contain horizontal scrolling within the scene. Cards and document content
must stay inside the viewport.

## 5. Keep source evidence and offline reading intact

Pin technical evidence to the artifact's revision. Resolve exact symbols,
call sites, complete definition spans and file hashes through its existing
extractors/manifests. Use the appropriate evidence label from `murali_docs/AGENTS.md`.
Explanatory code is separate from complete unchanged source; a fragment called
“full function” is not an acceptable substitute.

Reuse the local Source Explorer: embedded full files, highlighted target
spans, search, copy/download where supported, and return focus to the launching
link. Record an unavailable implementation as `BOUNDARY`. Keep source refs
available from the current event and legend rather than overloading the canvas.

Deliver self-contained HTML with inline CSS/model/controller/data and offline
source content. Do not require runtime CDN fonts, libraries or fetched JSON.
Provide a portable text flow/state equivalent in a nearby disclosure. Native
disclosures keep the main explanation and sources useful without JavaScript;
print should expose useful diagrams/transcripts while hiding inactive controls.

## 6. Build and judge the actual explanation

Edit the matched files under `src/` and integrate new pages in `src/build.py`.
Use the commands and checks in [the catalog](animation_examples.md). Review the
rendered first, intermediate, blocked/error and repeated/released states as
relevant. For supplements, exercise the real link from the call stack.

The visual review is decisive: with cards closed, can the reader identify the
actors and owners, watch the handoff or state change, see why a candidate waits
or a phase stops, and see what survives the next pass/call/release? If removing
the changing paragraph leaves the same ambiguous flashing boxes, improve the
scene before calling it complete.

State checks validate invariants and route consequences. Browser checks also
need to validate the displayed actors/slots/references and meaningful movement,
plus the accepted cards, legend, controls and readability. Passing a model
test or taking a screenshot does not by itself establish animation quality.
Use existing checks proportionally; expand them when changed behavior creates
an unresolved concern. Historical validation counts are examples, not quotas.

For skill/reference-only changes, check metadata, local paths and the diff.
Do not build animations or run MAX workloads just to change these instructions.
