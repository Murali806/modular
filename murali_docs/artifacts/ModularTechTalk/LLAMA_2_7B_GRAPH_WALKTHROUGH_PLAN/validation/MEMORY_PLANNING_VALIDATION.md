# Memory-planning page validation

Date: 2026-11-04. Result: **PASS**.

Artifact: [Llama 2 7B memory planning and MemoryPlan](../link_htmls/llama2_7b_memory_planning_and_memory_plan.html).
MAX source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
Model anchor: `max serve --model meta-llama/Llama-2-7b-hf`.
Device execution: none. These are documentation/browser checks.

## Source and navigation

- 49 complete snapshots match both the pinned Git revision and the working
  implementation, byte for byte.
- The page has 100 verified source targets and 23 complete code cells,
  containing 1,534 original source lines. Definition bounds, decorators,
  comments, docstrings and copy text match the snapshots.
- Nine QnA cells include eight nested subjects. Deeper explanations, text
  diagrams, example objects and source definitions provide up to four
  disclosure levels.
- QnA 1.1.6.1.2 places a twelve-step initialization sequence immediately
  after the three complete source cells. All step links select their exact
  source lines; object links reveal six input/output records with example
  values. The BF16 example policy is distinct from the saved HF metadata's
  `torch_dtype="float16"`.
- Each of the twelve steps also has detailed inputs, ordered logic, example
  results and source/object links. The same content is available in the
  native **Read all twelve step explanations** cell. New source targets
  cover the main-model property and the attention-multiplier rule.
- QnA 1.1.6.1.5 has six worked examples, with seven table links to their
  inputs, ordered calculations, plan fields and source. The cases cover
  checkpoint bounds, batch defaults, input-token targets, full-context page
  demand, allocation caps, OOM-search conditions and resident-token rounding.
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
Direct `file://` use; viewports 1440×1000 and 1600×1100, plus 390×844 and
320×844.

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
- All twelve sequence source links, six object links and seven example
  links work. The sequence scrolls within its own container on phones.
  The six separate example panels and helper paragraph are absent from the
  screen view, including after **Expand explanations**. Their full content
  remains readable in the cards.
- Hover/focus cards remain open when the pointer leaves. Hold keeps a step
  selected across other hovers; Release restores step switching. Source
  navigation, outside-click dismissal, Close, Escape and keyboard focus
  return pass. Space holds a focused step's card and enters it; Enter opens
  the step's source.
- Touch taps open a card at 390px/320px, with its source links opening the
  correct definition. Cards and their text fit within the viewport; the
  card body scrolls while Close and Hold remain visible. The twelve native
  explanations remain readable with JavaScript disabled. Card contrast
  checks cover desktop and both touch widths in the focused sequence report.
- All seven table **Example** links open the correct complete worked-example
  card, with six distinct cases and both Example 2 links sharing one case.
  Card text and table counts match the stored bodies; hovering and clicking
  preserve the page's reading location. The nested 512-request comparison
  starts expanded in its card.
- Example-card Hold/Release, keyboard entry, source links and desktop clicks
  pass. Click, Enter and Space hold and enter the selected card. Existing
  example fragment links open cards on desktop and phones. Switching between
  example and sequence cards keeps one visible card. At both touch widths,
  cards fit the viewport, comparison tables scroll within their own
  containers, and source links
  select the exact code line. Native examples work without JavaScript;
  all six complete examples are visible and expanded for printing.
  Example-card contrast results are recorded in the worked-examples report.
- Text/control contrast checks pass in all inspected states. The lowest
  inspected normal-text contrast is 5.624:1. This is a targeted color review,
  not a complete accessibility-conformance audit.
- With JavaScript disabled, native disclosures, the static page drawing,
  original code and local snapshot links remain usable.
- No browser console errors or external asset requests occurred.

Records: [browser checks](memory_planning_browser_checks.json) and
[contrast checks](memory_planning_contrast_checks.json). The final mobile
header adjustment also passed [title, header and navigation checks](memory_planning_header_checks.json).
Focused records: [initialization sequence](initialize_sequence_checks.json)
and [worked examples](length_examples_checks.json).

Visuals inspected: [overview](memory_planning_overview.png),
[budget waterfall](memory_budget_waterfall.png),
[KV pages](llama2_kv_pages.png),
[parent/worker handoff](memory_plan_worker_handoff.png),
[source explorer](memory_planning_source_explorer.png),
[390px budget](memory_budget_mobile_390.png) and
[320px budget](memory_budget_mobile_320.png). Mobile viewport reviews:
[390px titles and controls](memory_planning_mobile_390.png) and
[320px titles and controls](memory_planning_mobile_320.png).
New visuals: [initialization sequence](initialize_sequence.png),
[390px native fallback example](length_examples_mobile_390.png) and
[320px native fallback example](length_examples_mobile_320.png).
Step cards: [desktop](initialize_step_card_desktop.png),
[390px touch](initialize_step_card_mobile_390.png) and
[320px touch](initialize_step_card_mobile_320.png).
Worked-example cards: [desktop](worked_example_card_desktop.png),
[390px touch](worked_example_card_mobile_390.png) and
[320px touch](worked_example_card_mobile_320.png).

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
node --check src/check_initialize_and_limits.cjs
NODE_PATH=/tmp/max-v4-playwright/node_modules node src/check_memory_planning.cjs
NODE_PATH=/tmp/max-v4-playwright/node_modules node src/check_initialize_and_limits.cjs
git diff --check
```

The Node dependency is used only for validation. Reading the generated page
does not require Node, Python, Playwright or an internet connection.
