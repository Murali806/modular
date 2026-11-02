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
| QnA 1.1.6.1.2 | What does Llama3Config.initialize() create? | Model config + architecture declaration → architecture config |
| QnA 1.1.6.1.3 | How is free memory converted into a cache budget? | Budget waterfall and memory partition |
| QnA 1.1.6.1.4 | How do dimensions, dtype and page size affect demand? | KV pages, null block and element-width controls |
| QnA 1.1.6.1.5 | How are batch and length limits resolved? | Construction → capacity fitting → effective plan |
| QnA 1.1.6.1.6 | Who creates and consumes the plan? | API-parent / model-worker sequence |
| QnA 1.1.6.1.7 | What are all MemoryPlan fields? | Complete object example and consumer table |
| QnA 1.1.6.1.8 | Which source and checkpoint evidence supports this? | Complete source directory and pinned metadata |

The root and input map start open. Other QnAs, deeper explanations, portable
text diagrams and complete source definitions use native expandable cells.
Code starts collapsed. Deep links reveal the requested cell and its parents.
The contents sidebar can be hidden and restored.

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
snapshots against Git. The page has 85 source targets and 23 complete source
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
