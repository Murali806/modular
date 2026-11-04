# Revised plan: concepts with complete, expandable MAX source

Model: **Llama 2 7B**. Route: built-in single-device Graph API decoder.
Source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

The [research review](EXPLANATION_RESEARCH.md) supports an HTML reading guide
with a Markdown companion. Each chapter pairs a concept and diagram with
complete, expandable source functions and a full-file view.

## Six chapters

| Chapter | Main teaching device | Complete source to expose |
|---|---|---|
| 1. Select and configure Llama | System overview, object contract table | Llama registration; model loading, graph selection, and config finalization |
| 2. Record the computation | One Llama block; packed-token and KV input diagram | Llama module construction and tracing; transformer block; Graph constructor |
| 3. Finalize the graph | Six selectable before/after states | Complete `Graph.output()`; context exit; a small existing graph test |
| 4. Cross into the compiler | Actor-complete sequence and payload labels | Compile orchestration, Python session, native session, engine context, frontend |
| 5. Transform representations | Compiler-stage explorer | MO/MAP/MOGG/MGP builders, pipeline execution, Mojo/JIT scheduling, MEF serialization call site |
| 6. Initialize, execute, reuse | Prefill/decode buffer example; cold/warm comparison | Weight initialization, Llama execution, MEF import/cache branches; device-graph finalization distinction |

The core narrative concentrates on finalization and compilation. Other model
variants appear where the source branches, with links to their original file
context. External framework behavior is used only as a teaching analogy.

## Required code-cell behavior

1. A collapsed summary names the function, its purpose, file, original line
   interval, and evidence category.
2. Expansion reveals the entire function, including decorators and docstrings.
3. Numbered explanations highlight the matching lines without rewriting code.
4. Copy returns the original function text, preserving whitespace.
5. Full-file view reveals imports and surrounding implementation; the original
   file can be downloaded from the local snapshot.
6. Deep links open the relevant cell. Keyboard navigation and native
   expand/collapse controls work without pointer-only interactions.
7. A source manifest records the revision, file SHA-256, and extracted ranges;
   validation compares each cell with its original file at that revision.

## Reading experience

```text
Chapter navigation
        |
        v
Concept + contract -> focused visual -> complete source cell -> self-check
                              |                    |
                        choose a step       highlight explanation
                                                   |
                                             full-file context
```

Use consistent colors for graph construction, compilation, and execution.
Provide static/ASCII fallbacks for important diagrams. State illustrations
describe source behavior; performance charts require recorded measurements.

Use dark grey page and panel backgrounds, with contrasting code, labels,
source links, diagram states, and controls. Keep a readable light print style.
Place one collapsed **QnA 1** cell directly after chapter 1's first paragraph;
it contains the three package/checkpoint/registration answers and opens the
matching complete files in the source explorer.
Keep the Hugging Face `AutoConfig` explanation inside its own nested disclosure
within QnA 1, beside the MAX `Llama3Config` mapping it clarifies.

## Deliverables

| Artifact | Purpose |
|---|---|
| `index.html` | Main interactive guide; source and interactions available offline |
| `WALKTHROUGH.md` | Complete six-chapter reading companion with expandable exact code |
| `EXPLANATION_RESEARCH.md` | Reviewed references and the selected teaching methods |
| `DESIGN_PLAN.md` | This design and its acceptance criteria |
| `SOURCE_MAP.md` | Compact route through repository files |
| `sources/` and source manifest | Exact local snapshots for code fidelity and full-file context |
| `src/` | Reproducible guide builder and focused validators |
| `validation/` | Link/source-fidelity/browser results and small UI screenshots |

All artifacts stay under this task folder.

## Build order and acceptance

- [x] Read hardware, framework, and LLM explanation references.
- [x] Compare presentation formats and define the code-cell contract.
- [x] Extract complete functions and snapshot their original files.
- [x] Build the graph-finalization chapter and its code interactions.
- [x] Complete the other five chapters using the same source-checked structure.
- [x] Generate the HTML and Markdown companion from shared content.
- [x] Verify every source snapshot and cell against the recorded revision.
- [x] Verify local links, code-cell expansion, deep links, copy/full-file views,
      compiler-stage navigation, and mobile layout in a browser.
- [x] Update the entry README with artifacts and observed validation results.
- [x] Add QnA 1 beside the opening `llama3` explanation and keep it synchronized
      with the Markdown companion.
- [x] Check the dark grey theme across interactive states, source search,
      keyboard focus, mobile widths, and print colors.
- [x] Make chapter navigation collapsible from the top-left toolbar, reclaim
      its reading space, and verify keyboard/mobile use and saved preference.

## Text-tree reader revision: 2026-11-01

The original guide was restored to its Git baseline before this replacement
was built. The reader lives in a separate window opened from the QnA 1.1
header; individual function cards and task-selection radios are absent.

- [x] Show both task command examples as text.
- [x] Keep the startup map in a collapsed disclosure.
- [x] Combine QnA 1.1.1–1.1.6 into one continuous startup call tree.
- [x] Explain component ownership, input/output objects and process boundaries.
- [x] Include API/worker ZeroMQ request, response and cancellation directions.
- [x] Link 127 verified source targets to complete original files.
- [x] Validate reader launch, exact text/source, search, copy, mobile and print.

## Option B: OS/GPU systems lab — 2026-11-04

The user selected an expanded standalone systems page linked from the
call-stack reader. Scope: Linux + NVIDIA, the pinned MAX single-instance
startup path, and an explicitly illustrative external replica fleet.

| Scene | Boundary and teaching goal |
|---|---|
| Startup/readiness | Fresh worker interpreter, session before weight loading, retained device state, readiness/IPC before HTTP serving. |
| Network arrival | NIC/DMA, IRQ/NAPI, TCP listen/connection buffers, FDs, epoll/libuv and ASGI callbacks. |
| API/worker handoff | Tokenizer default executor, effective completion limit, request IDs, local Unix IPC, N and pending CE M. |
| GPU execution | MAX token batching/KV ownership, native engine, user/kernel-driver boundaries, streams, copies and completion. |
| Streaming/pressure | Early HTTP status boundary, unbounded application outputs, transport flow control, slow readers and safe cancellation/reuse. |
| Fleet | External routing, unequal request work, warmup/readiness, fixed request owners, worker/API failure and fresh replacement. |

The lab starts paused at 1,000 clients / 100 seeded tokenized contexts /
four prefill members / 96 waiting. Finite N and M are chosen teaching inputs;
MAX defaults both to unbounded in this source. M counts active prefill as
well as waiting CE. Each added replica includes one independent GPU/model
allocation and receives only new traffic after a compressed warmup.

The pure simulator conserves generated tokens as delivered + buffered +
discarded, maintains one unfinished request per conceptual client, bounds
page reservations and delays cancellation page reuse until a live pass ends.
The controller updates the selected request's state without closing its
transition disclosure. Historical graph/event/terminal-record storage and
rendered request lists are bounded.

- [x] Review system-design articles and official OS/GPU references.
- [x] Build six native animated sequences with 48 detailed green step cards.
- [x] Add 22 component/object cards and six worked arithmetic cards.
- [x] Implement traffic, memory, admission, routing, readers and fleet controls.
- [x] Preserve the existing source call tree and earlier source payloads.
- [x] Verify 53 source targets, 11 complete definitions and 17 research extracts.
- [x] Verify 15 meaningful simulator invariants and scenario behaviors.
- [x] Verify offline/browser/touch/keyboard/reduced-motion/print presentation.

Deliverables: [systems HTML](link_htmls/max_serve_os_driver_and_scaling.html),
[portable companion](OS_SYSTEMS_GUIDE.md),
[source/evidence manifest](os_systems_manifest.json) and
[validation record](validation/OS_SYSTEMS_VALIDATION.md).

The [reader](max_serve_llama2_7b_startup_call_stack.html),
[text export](MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK.txt) and
[checkpoint record](CHECKPOINTS.md) are the deliverables for this revision.

The merged tree places each startup phase under its actual caller, keeps
the worker under a labeled process handoff, and removes repeated top-level
startup entries. Configuration calls that really recur remain visible.
Output filenames identify MAX Serve, Llama 2 7B and startup.

## Focused OS-component reader — 2026-11-05

The user's narrowed scope is component roles, interactions, state machines
and high-level flow within one MAX Serve instance.

- [x] Map user-space software, kernel resources and hardware to the supplied
      startup call stack.
- [x] Build three actor-complete flows with 26 numbered handoffs: startup,
      request/response and the native device/driver boundary.
- [x] Show server, request and OS-thread state machines with 19 teaching states
      and eight valid normal/failure/cancellation paths.
- [x] Keep ordinary Python tasks, Linux CPU scheduling and GPU completion
      separate in the diagrams and role explanations.
- [x] Reuse 32 pinned source targets and saved OS/device learning references.
- [x] Complete browser, touch, keyboard, print and reduced-motion verification.

Deliverables: [focused HTML](link_htmls/max_serve_os_components_and_flow.html),
[portable companion](OS_COMPONENTS_GUIDE.md),
[manifest](os_components_manifest.json) and
[validation](validation/OS_COMPONENTS_VALIDATION.md).

## Evidence boundary

`SOURCE` is sufficient for this code explanation; a measured Llama inference
run is a separate experiment. The available Python runfiles environment was
checked, but importing `max.graph.graph` failed because `max._core` is missing.
Therefore the guide must label conceptual state views and inspected tests
accurately, rather than present a simulated IR view as an executed MAX dump.
