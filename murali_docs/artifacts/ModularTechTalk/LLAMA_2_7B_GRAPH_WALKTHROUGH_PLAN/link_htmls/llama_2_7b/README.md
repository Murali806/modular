# Llama 2 7B source walkthrough

Open `index.html` in the generated `link_htmls/llama_2_7b/` directory. Start
with the course map, follow the source-linked call tree, and use **Step** to
inspect each object or buffer change. **Play** starts only when requested.
**Back**, **Reset**, the event slider and alternative routes are deterministic.

The 26 lessons cover construction, all 32 decoder layers, paged attention and
KV addressing, compiler transformations through machine-code artifacts,
initialization, AsyncRT, the driver, sampling and lifetime. `coverage.html`
maps every one of the 35 Example 04 companions to its Llama adaptation.

Source links open the exact line. The source window supports **Hold**,
**Expand**, dragging and resizing. A held window preserves its current
contents when another explanation opens. Focus a window header or its resize
handle and use arrow keys for keyboard movement; Shift uses a larger step.
Escape closes an unheld window. Use **Close** to dismiss a held window.

The attention calculator preserves the previous tile's maximum, denominator
and numerator. The KV calculator separates payload from page-rounded capacity;
its second view resolves logical positions into physical page/element/byte
offsets. These are small numerical teaching examples.

## Architecture, costs and numerical animation

`architecture.html` begins with a complete **Stage / Input / Output / Weight
and operation** table and an explicit KV lifetime diagram. Its companion
navigation opens:

- `architecture_01_prefill.html`: a 512-token prefill, with a per-stage calculator.
- `architecture_02_decode.html`: one new token after 512 cached positions,
  including the 4-to-5-page capacity change at page size 128.
- `architecture_math.html`: a computed miniature decoder with matrix
  multiplication microsteps, normalization, RoPE, causal softmax, residuals,
  gated MLP, an untied head and stable KV page identities across three passes.

Select a stage in either calculator for substituted MAC, transfer-byte,
bandwidth, arithmetic-intensity and live-buffer formulas. Editable inputs
include active/cached token counts, equal-length batch size, payload bytes,
page size, causal versus rectangular contraction counts, attention traffic
boundary, KV read multiplier and hypothetical timing/rate assumptions. Export
the exact numerical ledger as small JSON. Packed-byte options are payload
experiments; they do not establish a supported encoding or quantization costs.

The default contraction totals are 3,350,272,671,744 MACs for causal initial
prefill and 6,741,557,248 MACs for a decode at C=512. Elementwise operations,
reductions and sampling are described separately. The calculator distinguishes
stored capacity from bytes moved, and analytical roofline terms from measured
performance. The byte accounting and working-set schedule are explicit.

The animation uses invented deterministic weights (D=4, F=6, two heads, two
layers, six token IDs) and browser FP64 arithmetic. Choose a head or output
cell to inspect different dot products. The online attention view computes a
vector weighted-value result from tiles. These are educational numerical
checks, not Llama output parity or GPU measurements. The full transcript and
default tables remain readable without JavaScript.

## Scope and evidence

- **SOURCE:** MAX revision
  `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`. Complete source files are shared
  once across the lessons. The explorer records paths, provenance and SHA-256.
  One native MLIR file comes from the existing saved source snapshot; the
  others are checked against this Git revision.
- Model identity: `meta-llama/Llama-2-7b-hf`. The original official config
  request was gated; saved config evidence uses `NousResearch/Llama-2-7b-hf`
  at `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`. See the parent guide's
  `evidence/llama2_7b_public_mirror_config.json` and
  `evidence/llama2_7b_memory_planning_provenance.json`.
- The checkpoint metadata says FP16, but this source revision's architecture
  does not list FP16 as supported. The main route therefore assumes an
  explicitly compatible resolved, unquantized configuration on one device.
  The rejection branch is visible in the startup lesson. No automatic weight
  conversion or successful serving of the original checkpoint is claimed.
- Compiler diagrams show source-established mechanisms and conditional pass
  order. They are schematic, not captured Llama IR. NVIDIA illustrates the
  concrete driver boundary; exact target dispatch, launch geometry and
  hardware execution remain target-dependent.
- Browser/state/math checks validate this documentation. **No model build,
  inference run, CUDA execution, performance measurement or parity result is
  claimed.** Those would be separate GPU-lab evidence.

## Small files and offline use

Each generated file must be at most **100 KiB**. HTML lessons share the reader,
styles, window helper and source manifest. Complete sources are divided into
small local JavaScript chunks and loaded only when opened. There are no
embedded screenshots, weights, binaries, CDNs or runtime package downloads.
Git LFS is unnecessary.

Keep this whole directory together for offline reading, including `sources/`.
The lessons, animations, calculators and complete source explorer work from
`file://` URLs without a server. The Startup tree and original Example 04
links additionally need the surrounding documentation checkout.

With JavaScript disabled, the call trees, explanations and complete event
transcripts remain readable using native disclosure controls. The source
explorer also lists original source-file links, which need the surrounding
repository/snapshot directories. Print opens all explanations and transcripts.

## Rebuild and check

Authoring files live under `src/llama_2_7b/`. Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/llama_2_7b/build.py
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/llama_2_7b/verify.py
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/llama_2_7b/verify_model.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/llama_2_7b/verify_architecture.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/llama_2_7b/verify_browser.cjs
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/llama_2_7b/verify_architecture_browser.cjs
git diff --check
```

Building needs Python and the `markdown` package, Node.js for the shared
architecture cost formulas, the surrounding pinned
sources/Example 04 snapshots and Git history. The accepted shared window
helper is extracted byte-for-byte from `cde7f11ef5^`; the builder changes only
this supplement. It performs no MAX build or model download.

The browser check needs Node.js, Playwright and its Chromium browser. If
Playwright lives outside this repository, point `NODE_PATH` at its
`node_modules` directory. It opens real local files with network access
disabled, checks every lesson and alternative route, exercises playback and
window controls, and inspects mobile/fullscreen/reduced-motion/no-JavaScript
behavior. Temporary screenshots and the browser report go to `/tmp` by
default; they are not committed as large assets.

`verify.py` checks source hashes and anchors, coverage, local paths/fragments,
object identities and startup HTML/TXT agreement. `verify_model.cjs` checks
state reconstruction, exact KV/address arithmetic and the online-softmax
recurrence against an independent stable normalization. The final recorded
documentation evidence lives in the parent `evidence/` directory.

`verify_architecture.cjs` checks independently worked counts, page-boundary
arithmetic, nonduplicated fused accounting and an independent full-sequence
miniature-decoder oracle. It compares cached and recomputed forwards and
checks causality and online softmax. `verify_architecture_browser.cjs` exercises
every numerical microstep, visible KV slots, selected dot products, calculator
formulas/export, source windows, playback, mobile/no-JavaScript/print and
offline navigation. Reference reconciliation and this addition's validation
are recorded as `evidence/architecture_math_reference_notes.md` and
`evidence/architecture_math_validation.json` beside the parent guide.

## Source organization

| File | Responsibility |
|---|---|
| `chapters_*.py` | Focused lessons, explicit scene events and exact source needles |
| `catalog.py` | Pinned source catalog, hashes and line-anchor resolution |
| `tree_links.py` | Direct call-tree symbol links |
| `coverage.py` | Example 04 coverage ledger; missing companions fail the build |
| `model.js` | Pure state snapshots and numerical examples |
| `reader.js`, `labs.js`, `reader.css` | Shared controller, source windows and calculators |
| `architecture_pages.py` | Shape/operation tables, prefill/decode companions and math-page shell |
| `architecture_model.js`, `architecture_costs.js` | Pure cost arithmetic and its offline controls |
| `architecture_toy.js`, `architecture_math.js` | Independent numerical state model and matrix/KV playback |
| `architecture.css` | Scoped, responsive companion styling |
| `build.py` | Scoped HTML/assets/source-chunk generation and size limit |
| `verify*` | Reproducible static, numerical and browser checks |

Copied source files retain their original notices and licenses. Generated
source chunks are exact reconstructions, not edited source excerpts.
