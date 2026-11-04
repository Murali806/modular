# Llama 2 7B: graph finalization and compilation in MAX

**Start with [the interactive HTML guide](index.html).** It contains six
chapters, selectable diagrams, and complete expandable code cells on a dark
grey background. **QnA 1**, immediately after section 1's first paragraph,
explains the `llama3` name, Hugging Face checkpoint files and lookup, and
architecture registration timing. The
[Markdown companion](WALKTHROUGH.md) contains the same explanation and source.

| Artifact | What it provides |
|---|---|
| [Interactive guide](index.html) | Concept → diagram → exact code → line-linked explanation → full-file context |
| [MAX Serve startup call stack](max_serve_llama2_7b_startup_call_stack.html) | One continuous tree combining QnA 1.1.1–1.1.6, from CLI parsing to HTTP readiness |
| [Memory planning and MemoryPlan](link_htmls/llama2_7b_memory_planning_and_memory_plan.html) | Nested QnA cells with input origins, initialization calls and example objects, a budget waterfall, KV pages, parent/worker handoff and complete pinned source |
| [OS components, roles and flow](link_htmls/max_serve_os_components_and_flow.html) | Focused user-space/kernel/hardware map, three animated flows and server/request/OS-thread state machines for one serving instance |
| [Focused OS companion](OS_COMPONENTS_GUIDE.md) / [validation](validation/OS_COMPONENTS_VALIDATION.md) | Portable ownership/sequence/state diagrams, 32 source targets and browser results |
| [MAX Serve OS / GPU systems lab](link_htmls/max_serve_os_driver_and_scaling.html) | Six animated Linux/NVIDIA scenes, 48 detailed step cards, 22 object cards and a deterministic client/queue/KV/fleet simulator |
| [Portable systems guide](OS_SYSTEMS_GUIDE.md) / [systems manifest](os_systems_manifest.json) | Actor-complete ASCII sequences, example objects, exact source targets and research hashes |
| [Memory explanation design](MEMORY_PLANNING_DESIGN.md) / [validation](validation/MEMORY_PLANNING_VALIDATION.md) | Disclosure structure, diagram controls, source checks and browser results |
| [Text export](MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK.txt) / [Markdown export](MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK.md) | The combined Llama 2 7B startup call tree, commands and source-file aliases |
| [Markdown companion](WALKTHROUGH.md) | Complete reading guide with expandable source and portable diagrams |
| [Explanation research](EXPLANATION_RESEARCH.md) | Hardware/framework/LLM references plus 17 OS/GPU/system-design sources and the teaching methods selected from them |
| [Revised design plan](DESIGN_PLAN.md) | Chapter structure, code-cell contract, and completed acceptance checklist |
| [Source map](SOURCE_MAP.md) | Compact navigation through repository files and symbols |
| [Source manifest](source_manifest.json) | Revision, SHA-256 hashes, complete code ranges, and annotations |
| [Validation record](validation/VALIDATION.md) | Source-fidelity and browser checks, commands, environment, and limits |
| [Git checkpoints](CHECKPOINTS.md) | Baseline, replacement checkpoint and safe rollback |

## Why HTML is the primary format

HTML supports source expansion, line highlighting, full-file search, and
selectable graph/compiler states. Markdown provides a searchable, reviewable
companion. PowerPoint is suited to a later condensed presentation; it is less
useful for reading entire functions and following their callers.

The guide uses methods from the Illustrated Transformer, Harvard's Annotated
Transformer, Distill, PyTorch, JAX, NVIDIA, AMD, Google TPU, vLLM, and GPU
optimization articles. The [research review](EXPLANATION_RESEARCH.md) links
all twelve references and explains where each method is applied.

## What is complete

- **Six chapters:** model/config selection; symbolic construction; graph
  finalization; Python/C++ compilation; compiler stages; initialization and reuse.
- **39 expandable source cells**, containing **2,012 original source lines**.
  The cells retain complete definitions or assignments, including comments and
  docstrings. Annotations appear outside the code.
- **49 complete source files**, available in the source explorer and as local
  snapshots under `sources/`.
- Finalization and compiler-stage explorers, prefill/decode input illustrations,
  a native call sequence, and cold/import/cache-hit comparisons.
- Keyboard expansion, source copying, source search, deep links, chapter-level
  expansion, and mobile layout.
- A **Hide sidebar / Show sidebar** toggle in the top-left toolbar. Hiding the
  sidebar widens the guide; the choice is remembered when browser storage is
  available. The toggle also controls chapter navigation on mobile.
- A single expandable **QnA 1** cell containing the three checkpoint/package/
  registry answers, with links to complete original source files. Its
  checkpoint answer contains nested **QnA 1.1** with expandable **QnA 1.1.x**
  call-flow trees for CLI/config loading, lazy architecture registration,
  tokenizer and generation defaults, weight loading, and MAX's `Llama3Config`
  lifecycle. Each tree node includes its explanation and a source link. QnA
  1.1.2 also distinguishes API `Settings` from pipeline settings and clarifies
  when weight downloads occur.
- A **Call stack tree ↗** button in the **QnA 1.1 header** opens a separate
  window with one combined startup call tree covering all six QnA topics.
  It shows both CLI command examples, component
  ownership, typed objects, call/return effects, the worker boundary and ZeroMQ
  request/response/cancellation directions. Its 129 source targets link to
  complete original files and exact definitions or invocation lines.
- A collapsed **Startup map** in QnA 1.1 and in the separate reader. The
  reader also includes a source-alias map, text search, copy and download.
- Dark grey surfaces and checked text/control contrast, including selected
  diagram states, highlighted code, search matches, and a light print layout.
- In the memory page's **QnA 1.1.6.1.2**, a **Sequence diagram** cell follows
  the complete initialization source. Its call labels open source lines and
  its object labels reveal the Llama 2 7B example values, including BF16 dtype,
  32 layers, 128-wide heads and 128-token KV pages.
  Hover or focus any green step for a detailed logic card with inputs,
  ordered logic, example results and source/object links. **Hold card** keeps
  the selected step; Close or Escape dismisses it. Keyboard and touch access,
  plus a native cell containing all twelve explanations, are included.
- **QnA 1.1.6.1.5** has six worked examples showing how checkpoint length,
  batch count, input-token targets and resident KV pages determine the final
  limits. Each example includes inputs, calculations, source links and plan
  results, including the 4 GiB and 1 GiB cache cases.
  Hover or focus an **Example** link in the table to read the complete case
  in a card with **Hold card**, **Close** and source links. Touch taps also
  open cards; clicking, Enter or Space holds and enters the selected card.
  The separate six-panel list and its helper paragraph are removed from the
  screen view. Complete examples remain readable without JavaScript and
  when printing.
- The separate **OS / GPU systems lab** follows startup/readiness, NIC DMA and
  Linux TCP/epoll, tokenizer/executor and local Unix IPC, native GPU execution,
  SSE pressure/cancellation, and an external fleet deployment layer. Green
  step labels open persistent cards with inputs, ordered logic, results and
  source/reference links. Its six worked examples stay in table-triggered
  cards without a duplicate visible list.
- The scaling lab separates idle connections, API CPU work, IPC N, pending
  CE M, GPU membership, retained KV allocation and output buffers. Controls
  vary token lengths, arrival/client counts, finite admission, memory,
  reader speed and routing. Add/Crash/Replace demonstrate readiness and
  failure boundaries for independent one-GPU replicas. All rates are explicit
  teaching inputs; charts are simulation results.
- The focused **OS components & flow** page explains the roles and interaction
  of 15 components in one serving instance. Three animations contain 26
  handoffs: startup/readiness, one request/response, and the GPU driver/buffer
  boundary. Server, request and OS-thread diagrams contain 19 teaching states
  with eight selectable paths. Green labels open persistent role/transition
  cards; optional source links open the complete original files.

## Source scope

| Item | Value |
|---|---|
| Model family | Llama 2 7B, confirmed by the user |
| Working checkpoint name | `meta-llama/Llama-2-7b-hf`; checkpoint revision not captured |
| Memory example metadata | Official file sizes at `01c7f73d771dfac7d292323805ebc428287df4f9`; public-mirror config at `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` |
| Implementation route | Built-in `LlamaForCausalLM`, one device, Graph API |
| Repository revision | `610f7111836d9a950d45ba52ec9b3e0ed589fe3f` |
| Evidence | Repository source and inspected test definitions |
| Runtime status | No Llama inference or GPU measurements performed |

The three main operations are distinct: `Graph.output()` finishes and verifies
the graph's output contract; compilation produces a MEF-backed artifact;
initialization binds weights and executable state. Model configuration
finalization and device-graph finalization have their own explanations.

## Using the guide

1. Open `index.html` in a browser from this folder. It works with a direct
   `file://` URL; the guide has no external runtime dependencies.
2. Select a chapter or a diagram step. Use **Hide sidebar / Show sidebar** in
   the top-left toolbar to change the reading space.
3. Expand a source cell. Select an annotation to highlight its original lines.
4. Use **View whole file** for imports and surrounding context, or **Copy
   complete source** to copy the original definition.
5. Use the Markdown companion when reviewing or searching in an editor.
6. Open **QnA 1**, then select **Call stack tree ↗** in the QnA 1.1 header.
   The browser opens a separate window, or a tab if its window policy requires
   one. The guide stays in place. Follow the single tree from top to bottom,
   or use **Find in tree** to locate a function, object or component.
7. Select a function or a **call site** link to inspect exact source. Expand
   **Source map** for file aliases. Long text lines scroll inside the tree,
   preserving indentation on small screens.
8. Open **OS components & flow ↗** in the call-stack reader header. Inspect
   component roles, Step/Play through a high-level flow, or select a server,
   request or OS-thread state-machine path. Hover/focus/tap labels and use
   Hold/Close/Escape to control the explanation card.
9. Open **OS / GPU systems lab ↗** in the call-stack reader header. Follow
   the six scenes with Step/Play; hover, focus or tap green labels and object
   headers. Hold/Close/Escape control the persistent explanation card.
10. In the lab, choose a preset, Step or Play, and select a request to inspect
    its state and KV/output ownership. Input changes take effect with
    **Apply & restart**. Adding a replica introduces a two-second illustrative
    warmup and affects future routing; existing requests retain their owners.

## Rebuilding

From this task folder, using Python with the `markdown` package available:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py
```

`src/content.md` holds the explanation, and `src/qna.md` holds the three answers
in QnA 1; `src/cells.py` defines the source
selections and authored annotations. The builder extracts source directly
from the checkout and compares it with the pinned revision before generating
both outputs. `src/widgets.py`, `src/style.css`, and `src/app.js` define the
interactive presentation.

The separate reader is generated by `src/call_stack.py` from the authored
`src/call_stack_tree.txt`. `src/call_stack_refs.py` defines source targets;
the builder verifies full definition bounds and unique invocation text.
`src/startup_map.txt` supplies the collapsible ASCII startup map.
`src/call_stack.css` and `src/call_stack.js` provide the reader's presentation.

The output name `MAX_SERVE_LLAMA2_7B_STARTUP_CALL_STACK` identifies the
framework, model and phase covered: CLI/configuration, registry/tokenizer,
worker weight loading, graph compilation/initialization and HTTP readiness.
The main startup entry points appear once; genuine repeated configuration
calls and the separate worker process remain explicit.

`src/os_systems.py` builds the new standalone page from
`src/os_systems_content.py`, `src/os_systems.html`, `src/os_systems.css`,
`src/os_systems_lab.js` and `src/os_systems.js`. It reuses pinned source and
model provenance, adding eight source snapshots only to the systems manifest.
The earlier readers' source manifests/payloads remain unchanged.

`src/fetch_os_systems_evidence.py --refresh` deliberately refreshes the 17
public-reference extracts. Normal builds never fetch them. Simulator and
browser commands are in [the systems validation record](validation/OS_SYSTEMS_VALIDATION.md).

`src/os_components.py` builds the focused OS reader from its content, HTML,
CSS and browser controller. It reuses existing pinned snapshots and the
reviewed OS/device references. Its [manifest](os_components_manifest.json)
and [validation record](validation/OS_COMPONENTS_VALIDATION.md) record the
component, source and state-path checks.

This version starts from the restored baseline checkpoint. The rejected
radio-button prototype is not part of the guide.

Browser validation uses Playwright. The exact command used on this host is
recorded in [the validation record](validation/VALIDATION.md).

## Validation

Source fidelity, complete ranges, local links, all interactive states,
keyboard expansion, exact copying, full-file search, deep links, direct-file
opening, and desktop/mobile layouts passed their checks. The browser reported
no console errors. Screenshots are in `validation/`.

The local MAX Python runfiles environment lacks `max._core`, so the guide
uses accurately labeled source-derived illustrations and inspected tests.
It does not present those illustrations as captured compiler IR or measured
model execution.
