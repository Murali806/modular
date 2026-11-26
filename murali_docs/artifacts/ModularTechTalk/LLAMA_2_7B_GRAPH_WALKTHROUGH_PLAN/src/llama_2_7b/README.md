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
- `architecture_math.html`: actual Llama tensor dimensions with selectable
  layer, head and absolute coordinates; full reductions, rotary pairs, causal
  attention and 128-slot KV pages across prefill and two decode passes.

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

The animation uses the actual D=4096, F=11008, 32 layers, 32 heads × 128
channels and 32000-token vocabulary. Prefill is T=512; decode is T=1 after
512 or 513 cached positions. Matrix overviews locate a bounded 3×5 viewport
inside each full tensor; coordinates are absolute. Select any output and
follow every term of its 4096-, 11008-, 128- or L-term reduction. Step size,
Back, seek, Finish operation, Next stage and all 32 layer selectors make
long reductions navigable. RoPE exposes split-half pairs c and c+64.

The numbers are deterministic generated operands, evaluated lazily in browser
FP64. Each operation is a separate numerical probe, not an end-to-end model
forward. The attention probes share their Q/K/V values across RoPE, cache,
scores, stable softmax and P·V, and preserve old positions across passes.
Independent tests cover full reductions and online softmax for all 128
channels. No real checkpoint values, output parity or GPU timing is claimed.

The cache grid shows every slot in four or five 128-token pages for the
selected layer. All 32 layer counters expose partially completed writes.
Inspect any slot, including unused capacity, and resolve exact K/V byte
offsets. A 2-byte payload assumption gives 256 MiB used at 512 completed
tokens, then 256.5 MiB used within 320 MiB assigned at 513. Physical page IDs
and paging size are teaching settings. The larger allocator pool is excluded.

Assets are shared and work offline. No large matrix, model weight or bitmap
is embedded. Source cards, a no-JavaScript transcript and print remain
available.

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

`verify_architecture.cjs` checks independently worked cost counts, literal
full-size reductions, split-half rotary pairs, stable attention for all 128
channels, per-layer KV occupancy and exact physical addresses.
`verify_architecture_browser.cjs` checks actual dimensions, last coordinates,
every operation across three passes, all 32 cache slices, bounded DOM size,
matrix navigation, retained values, calculator links, Play/Pause, cards,
mobile, no-JavaScript and print. See
`evidence/architecture_tensor_validation.json` beside the parent guide.
The older `architecture_math_validation.json` records the previous
miniature version; its forward-parity checks do not describe this explorer.

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
| `architecture_tensor.js`, `architecture_math.js` | Lazy full-size operation probes, coordinate viewports and KV playback |
| `architecture.css` | Scoped, responsive companion styling |
| `build.py` | Scoped HTML/assets/source-chunk generation and size limit |
| `verify*` | Reproducible static, numerical and browser checks |

Copied source files retain their original notices and licenses. Generated
source chunks are exact reconstructions, not edited source excerpts.
