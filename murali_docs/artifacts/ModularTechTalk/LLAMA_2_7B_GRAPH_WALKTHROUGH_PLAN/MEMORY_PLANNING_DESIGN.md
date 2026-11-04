# Memory planning for MAX Serve / Llama 2 7B

Implemented page: [Memory planning and MemoryPlan](link_htmls/llama2_7b_memory_planning_and_memory_plan.html).
The [startup call stack](max_serve_llama2_7b_startup_call_stack.html) links it
immediately below the `MemoryEstimator.plan()` explanation, with continuous
connector-only spacer rows.

## Reading and disclosure plan

| Cell | Question | Main visual |
|---|---|---|
| QnA 1.1.6.1 | What does planning produce? | Inputs → pricing → MemoryPlan |
| QnA 1.1.6.1.1 | Where does each input originate? | CLI / HF / architecture / driver input map |
| QnA 1.1.6.1.2 | What does Llama3Config.initialize() create? | Model inputs → source-linked call sequence → example architecture and KV objects |
| QnA 1.1.6.1.3 | How is free memory converted into a cache budget? | Budget waterfall and memory partition |
| QnA 1.1.6.1.4 | How do dimensions, dtype and page size affect demand? | KV pages, null block and element-width controls |
| QnA 1.1.6.1.5 | How are batch and length limits resolved? | Construction → capacity fitting → effective plan, with six worked examples |
| QnA 1.1.6.1.6 | Who creates and consumes the plan? | API-parent / model-worker sequence |
| QnA 1.1.6.1.7 | What are all MemoryPlan fields? | Complete object example and consumer table |
| QnA 1.1.6.1.8 | Which source and checkpoint evidence supports this? | Complete source directory and pinned metadata |

The root and input map start open. Other QnAs, deeper explanations, portable
text diagrams and complete source definitions use native expandable cells.
Code starts collapsed. Deep links reveal the requested cell and its parents.
The contents sidebar can be hidden and restored.

After the three complete initialization source cells, an expandable sequence
cell traces twelve steps from `MemoryEstimator.plan()` to the returned
`Llama3Config`. Green call labels and arrows open the exact source lines;
amber object labels open six records for the pipeline, model, HF metadata,
cache config, `MHAKVCacheParams` and architecture config. Example fields use
the pinned Llama 2 7B metadata with an explicit BF16 policy, one device,
4096-token length and 128-token pages. A portable text sequence accompanies
the native SVG, which needs no renderer or network.

Each green step opens a detailed logic card on hover or focus. The card
shows its inputs, ordered logic, example result, source links and object
links. It remains open while the reader moves the pointer; **Hold card**
keeps its step selected across other hovers. **Release card** permits step
changes again. Close, Escape, an outside click or opening source dismisses
the card. Space on a focused step holds the card and moves keyboard focus
into it; Enter retains the source route. Touch taps open the card, whose
links open source. The card stays within the viewport and uses more vertical
space on phones. All twelve explanations are also available in a native
disclosure cell for offline reading without JavaScript and for printing.

The batch/length section gives each concrete example named inputs,
ordered calculations and resulting plan fields. Six complete worked examples
cover checkpoint length choices; explicit/default batches and prefill targets;
four-request versus 512-request page demand; 4 GiB and 1 GiB allocation caps;
the optional OOM-search guard; and page-aligned resident-token limits.
Links in the table open the relevant example card. The cases distinguish a
single-request length, a request-count ceiling and the pages resident across
the batch, including null-block overhead and per-request page rounding.

Hovering or focusing any of the table's seven **Example** links opens the
complete corresponding worked example in a card. The two Example 2 links
share one explanation. Cards retain every calculation, comparison table,
plan field and source link; nested comparisons start expanded. They share
the sequence cards' Hold/Release, Close, Escape, keyboard and touch behavior,
with one card visible at a time. Clicking, Enter or Space holds and enters
the selected card. Existing example fragment links also open the relevant
card. Wider example cards fit the tables on desktop; tables scroll inside
their containers on phones. The separate six-panel list and helper paragraph
are absent from the screen view. The complete example bodies supply the
cards and remain available as native disclosures without JavaScript and
for printing.

## Reader-controlled diagrams

- Input map: step through CLI/defaults, checkpoint metadata, MAX architecture
  facts and live device statistics.
- Budget waterfall: step through free bytes, utilization, static estimates,
  allocation cap, manager pricing and the returned plan.
- KV pages: change context length, candidate request count and unquantized
  element width; show null-block overhead, physical page capacity and errors.
- Parent/worker sequence: configuration and planning in the API parent,
  deferred factory, spawn, worker model construction, KV-manager setup,
  readiness, IPC connection and HTTP serving.

The three sequence controls offer back, step, play, pause and replay. They do
not autoplay; closing a containing section or hiding the document pauses them.
Reduced-motion settings remove transitions. Each main visual has a portable
text equivalent. No CDN, renderer or network is required to read the page.

## Evidence contract

MAX claims are `SOURCE` at revision
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. The builder verifies complete
snapshots against Git. The page has 100 source targets and 23 complete source
cells; annotations remain outside original code. Driver and Transformers
interfaces are identified as boundaries.

Official pinned Hub metadata supplies selected file sizes. The dimensions
come from a pinned public-mirror config because the official unauthenticated
config request returned HTTP 401. Both are stored in the [provenance record](evidence/llama2_7b_memory_planning_provenance.json).

Hardware memory is hypothetical. The numeric exercise assumes one plain
full-attention leaf, no quantized scales, no draft/vision cache and one device.
It distinguishes free memory, initial KV budget, committed cache budget and
the manager's page-aligned physical slab. Encoding compatibility is stated
explicitly; no model execution or GPU measurement is claimed.
