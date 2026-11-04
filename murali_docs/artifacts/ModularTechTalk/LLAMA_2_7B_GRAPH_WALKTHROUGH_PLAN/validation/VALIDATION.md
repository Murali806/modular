# Validation record

Date: 2026-11-01. Revision:
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

Subject: Llama 2 7B / `meta-llama/Llama-2-7b-hf`, single-device Graph API.
Device execution: none. The checks below validate documentation, source
fidelity, and browser interactions; they do not run Llama inference.

## Source and document checks — PASS

Run: 2026-11-01, revision `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.

- 39 snapshots are byte-for-byte identical to the files at the recorded Git
  revision and to the current working source.
- 39 complete source selections contain 2,012 original lines. Python selections
  include decorators and complete AST definition bounds; C++ selections include
  complete balanced-brace definitions.
- Every displayed code line matches its original selection, including comments,
  docstrings, indentation, and source line numbers.
- HTML copy data and the Markdown companion contain the same original code.
- All authored annotation ranges lie inside their complete source selections.
- Local links, source line anchors, HTML IDs, and Python source syntax pass.
  The exact link count is recorded in `source_checks.json`.
  The six nested QnA cells each contain an expandable call-flow list with
  source-linked function names.
- The HTML has no external runtime assets.
- One combined startup tree matches its manifest and exported text.
  All 127 source targets have verified definition bounds or unique invocation
  text. Complete source shown in the reader matches the pinned snapshots.

Machine-readable result: [source_checks.json](source_checks.json).

## Browser checks — PASS

Chromium `141.0.7390.37`, driven with the existing Playwright installation.
Viewports: 1440×1000 desktop and 390×844 mobile.
The expanded QnA was also checked at 320×760. The separate call-tree reader
was checked at 390×844 and 320×844.

- Complete source cells start collapsed and expand from the keyboard.
- The top-left **Hide sidebar / Show sidebar** button works with pointer,
  touch, Enter, and Space. It updates its expanded state and leaves a visible
  control for reopening navigation. At 1440px, hiding the sidebar increased
  the main reading area from 1205px to 1440px.
- Sidebar preference survives a direct-file reload when storage is available.
  The toggle still works when storage access is blocked; with JavaScript
  disabled, navigation remains visible and the inactive toggle stays hidden.
- One cell named **QnA 1** appears immediately after the first paragraph of
  chapter 1. Its three answers, keyboard expansion/collapse, source links,
  and direct `#qna-1` link work.
- All six **QnA 1.1.x** cells expand and contain linked call-flow trees. A
  function link in QnA 1.1.2 opens the `api_server.py` source in Source Explorer
  and highlights the linked `fastapi_app()` definition line.
- The expanded QnA 1.1.2 tree has no horizontal overflow at 390px or 320px.
- Annotations highlight their actual source ranges.
- Copying `Graph.output()` returns its entire original text, checked through
  the browser clipboard.
- Full-file text matches its source, including original license headers.
- File switching, text search, next match, empty search results, and Escape
  closing work.
- Every finalization, compiler-stage, cache, and buffer state renders correctly.
- Previous/next controls respect their bounds; diagram links open matching code.
- Code deep links, chapter expansion, and collapse-all work.
- Direct `file://` opening works with embedded source and interactive controls.
- The mobile page, expanded source cell, and source dialog have no document-level
  horizontal overflow.
- No browser console errors or external requests were observed.
- Dark grey colors were checked for body text, tables, links, metadata,
  annotations, code, selected diagram tabs, search matches, and source controls.
- Rendered text meets the checked WCAG text thresholds (4.5:1 for normal text,
  3:1 for large text). The checker also evaluates control outlines, visible
  SVG lines, and keyboard focus at 3:1. These color checks are not a complete
  accessibility-conformance audit.
- The print stylesheet switches to light paper colors, expands disclosures,
  and restores their previous states afterward.

The replacement call-tree reader also passed these checks:

- The QnA 1.1 header button opens a separate page without toggling the parent
  disclosure. Both command examples appear as text; no radio inputs exist.
- The startup map starts collapsed and supports pointer/keyboard expansion.
- The single tree covers all six QnA topics and preserves its exact text,
  source aliases and indentation. Major startup entry points appear once,
  while all three Llama3Config initialization uses remain.
- Links across every QnA open full source, highlight definition/caller lines,
  and support file search and Escape. Source-map file links work too.
- Tree search, next match, no-match handling and exact text copying work.
- Worker factory invocation precedes readiness in the tree; configuration
  finalization precedes graph compilation. ZeroMQ directions are included.
- The mobile tree scrolls horizontally inside its own area; the page, source
  map and source dialog remain within 390px and 320px viewports.
- Printing uses readable paper colors and restores disclosure states.
- Direct `file://` use works with the renamed startup reader. With JavaScript disabled, the combined tree,
  expandable map and raw snapshot links remain available.

The final browser report contains 24 grouped checks and no console errors.

Machine-readable result: [browser_checks.json](browser_checks.json).
Color results by inspected state: [contrast_checks.json](contrast_checks.json).
Additional sidebar interaction review: [sidebar_checks.json](sidebar_checks.json),
covering desktop, 390px/320px mobile widths, persistence, and fallback behavior.

Visual inspection artifacts:

- [Desktop overview](overview.png)
- [Expanded code and annotations](code_expanded.png)
- [Mobile finalization explorer](mobile_finalization.png)
- [QnA 1 expanded](qna_expanded.png)
- [Mobile QnA 1](mobile_qna.png)
- [Source explorer and search](source_explorer.png)
- [Compiler explorer](compiler_dark.png)
- [Native compilation sequence](native_sequence_dark.png)
- [Sidebar collapsed on desktop](sidebar_collapsed.png)
- [Sidebar collapsed on mobile](mobile_sidebar_collapsed.png)
- [QnA 1.1 launch button](call_stack_launch.png)
- [Separate text-tree reader](call_stack_overview.png)
- [Collapsible startup map expanded](call_stack_startup_map.png)
- [Source opened from a tree](call_stack_source.png)
- [Text reader at 390px](call_stack_mobile_390.png)
- [Text reader at 320px](call_stack_mobile_320.png)

## Commands

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/build.py
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/validate.py
node --check murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/app.js
node --check murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/call_stack.js
NODE_PATH=/tmp/max-v4-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_browser.cjs
git diff --check
```

The Playwright dependency path is the existing host installation used for
validation. Opening the generated HTML does not require Python, Node, or
Playwright.

## MAX execution boundary

The available `.max+python+max+_entrypoints+pipelines.venv/bin/python` reports
Python 3.13.11. A read-only import check for `max.graph.graph` failed with
`ModuleNotFoundError: No module named 'max._core'`. No model compile, test
suite, inference request, or GPU benchmark is recorded as executed.

The state diagrams and token IDs are explicitly labeled illustrations. Blog
references inform teaching methods; MAX implementation claims are grounded
in the pinned local source.

## Memory-planning supplement — 2026-11-02

The [memory-planning validation record](MEMORY_PLANNING_VALIDATION.md) covers
the new linked page: 85 verified source targets, 23 complete code cells,
nested QnAs, reader-controlled diagrams, exact copying, direct-file use,
and 390px/320px layouts. Source/link and browser checks passed. Memory inputs
for the numerical exercise are explicitly hypothetical; no model execution
or GPU measurement was performed.

## OS / GPU systems supplement — 2026-11-04

The [systems validation record](OS_SYSTEMS_VALIDATION.md) covers the new
standalone animated page, 53 verified source targets, 17 research extracts,
15 deterministic simulator checks and 14 grouped browser checks. All 48 step
cards, 22 object cards, 11 source cells and 22 contrast states passed, including
offline, touch, keyboard, print and reduced-motion behavior. Rates and fleet
actions are explicitly illustrative.

## Focused OS-component supplement — 2026-11-05

The [focused validation record](OS_COMPONENTS_VALIDATION.md) covers component
roles and interactions in one serving instance, three animated flows and
server/request/OS-thread state machines. Source checks and 11 grouped browser
checks pass: 15 component cards, 26 steps, 19 state cards, eight paths, 32
source targets and 17 contrast states, including touch, print, keyboard,
offline and reduced-motion operation.
