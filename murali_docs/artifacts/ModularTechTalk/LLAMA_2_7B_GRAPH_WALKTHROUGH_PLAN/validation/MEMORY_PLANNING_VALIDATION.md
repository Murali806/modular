# Memory-planning page validation

Date: 2026-11-02. Result: **PASS**.

Artifact: [Llama 2 7B memory planning and MemoryPlan](../link_htmls/llama2_7b_memory_planning_and_memory_plan.html).
MAX source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
Model anchor: `max serve --model meta-llama/Llama-2-7b-hf`.
Device execution: none. These are documentation/browser checks.

## Source and navigation

- 49 complete snapshots match both the pinned Git revision and the working
  implementation, byte for byte.
- The new page has 85 verified source targets and 23 complete code cells,
  containing 1,534 original source lines. Definition bounds, decorators,
  comments, docstrings and copy text match the snapshots.
- Nine QnA cells include eight nested subjects. Deeper explanations, text
  diagrams and source definitions provide a third disclosure level.
- Local links and cross-page anchors pass. The new startup-tree link opens
  input origins in a separate window.
- The authored call-stack text is identical to its pre-edit backup after
  removing the three inserted connector/link rows. Existing prose and
  indentation were preserved.
- Builds read saved checkpoint metadata and need no network access.

Machine record: [source_checks.json](source_checks.json).
Complete source targets: [memory_planning_manifest.json](../memory_planning_manifest.json).

## Browser and visual checks

Chromium `141.0.7390.37`, using the existing Playwright installation.
Direct `file://` use; viewports 1440×1000, 390×844 and 320×844.

- Native nested expansion, keyboard control, deep links and the collapsible
  contents sidebar work. Complete code starts collapsed.
- Input, budget and process diagrams support back, step, play, pause and
  replay, with bounded steps and no autoplay. Closing a section pauses it.
  Reduced-motion settings remove transitions.
- Baseline page arithmetic, two-byte/four-byte elements, context rounding,
  null blocks, 1 GiB/4 GiB caps and the 512-request example pass. Cache budgets
  that cannot hold a null block plus one request page show an error.
- Source Explorer displays complete original files and highlights complete
  definitions plus entry/call-site lines. Copying, search, next/no-match
  results and Escape work.
- All expanded sections, tables, code scroll areas and the source dialog fit
  at 390px/320px without document-level horizontal overflow.
- Text/control contrast checks pass in all inspected states. The lowest
  inspected normal-text contrast is 5.624:1. This is a targeted color review,
  not a complete accessibility-conformance audit.
- With JavaScript disabled, native disclosures, the static page drawing,
  original code and local snapshot links remain usable.
- No browser console errors or external asset requests occurred.

Records: [browser checks](memory_planning_browser_checks.json) and
[contrast checks](memory_planning_contrast_checks.json). The final mobile
header adjustment also passed [title, header and navigation checks](memory_planning_header_checks.json).

Visuals inspected: [overview](memory_planning_overview.png),
[budget waterfall](memory_budget_waterfall.png),
[KV pages](llama2_kv_pages.png),
[parent/worker handoff](memory_plan_worker_handoff.png),
[source explorer](memory_planning_source_explorer.png),
[390px budget](memory_budget_mobile_390.png) and
[320px budget](memory_budget_mobile_320.png). Mobile viewport reviews:
[390px titles and controls](memory_planning_mobile_390.png) and
[320px titles and controls](memory_planning_mobile_320.png).

## Example boundaries

Official pinned Hub metadata supplies the selected safetensors file sizes.
The official unauthenticated config request returned HTTP 401; dimensions
come from the recorded pinned public-mirror config. The numerical exercise
uses a hypothetical 24 GiB free device, one unquantized full-attention cache
leaf, four requests and two-byte KV elements. It distinguishes byte budgets
from whole-block physical allocation. The page explains the pinned MAX
revision's encoding compatibility limits.

No MAX model was loaded, compiled, served or benchmarked. No GPU allocation
or memory measurement is presented as executed.

## Commands

From this task folder:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py
node --check src/memory_planning.js
node --check src/check_memory_planning.cjs
NODE_PATH=/tmp/max-v4-playwright/node_modules node src/check_memory_planning.cjs
git diff --check
```

The Node dependency is used only for validation. Reading the generated page
does not require Node, Python, Playwright or an internet connection.
