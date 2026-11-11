# Compiled kernel loader reader

Validated November 11, 2026.

The add-one D.1 resource card links directly beside
`CompiledKernelsContext::load(archive_bytes, symbols)` to
`link_htmls/max_compiled_kernels_context_load_and_driver.html`.

MAX source revision: `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
LLVM source revision: `6941494e2ae550d753728ba13454311b344d82cf`, selected by
the same checkout's `bazel/public-patches/llvm_source.bzl:5`.
Authoring files are in `src/compiler_kernel_loader*`; the preserved previous
answer is `src/compiler_kernel_loader_answer.md`. Its SHA256 is recorded in
`compiler_kernel_loader_manifest.json`.

## Scope and evidence

- SOURCE: 34 embedded complete files: 25 repository files and nine unchanged
  upstream LLVM files. File hashes are verified against the pinned revisions
  and the [upstream manifest](../references/llvm_orc/6941494e2ae550d753728ba13454311b344d82cf/manifest.json).
- Teaching model: 68 conditional events covering cold/reuse/debug/error and
  CPU/CUDA/HIP routes, a connected source tree, 16 detailed sections and nine
  focused transformation views on one deterministic timeline.
- Values: illustrative symbol names, hashes, native addresses, sizes, opaque
  device handles, argument slots and tiny arithmetic; no compiler dump.
- SOURCE: pinned LLVM implementations establish object decoding, pruning,
  allocation, block-content copying, address publication, external resolution,
  fixups, memory finalization and symbol readiness. The ELF64/x86-64 cutaway
  and its Pointer64 example are illustrations of those mechanisms.
- BOUNDARY: vendor code residency, physical register/thread assignments,
  numerical execution and exact completion scheduling. No MAX/Mojo
  compilation, inference, GPU measurement, OS linking or vendor-driver
  execution was performed for this documentation.

[Research decisions and primary references](../KERNEL_LOADER_ANIMATION_RESEARCH.md)
connect each mechanism to its visible transformation. The whole-process
ownership map remains available below the focused views.

## Commands used

Run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/compiler_kernel_loader.py
node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_compiler_kernel_loader.cjs
NODE_PATH=/tmp/codex-inference-session-playwright/node_modules node murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_compiler_kernel_loader_browser.cjs
PYTHONDONTWRITEBYTECODE=1 python3 murali_docs/artifacts/ModularTechTalk/LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/src/check_compiler_examples.py
git diff --check
```

The Playwright installation above was verified in this environment; it is not
a product dependency. The generated HTML works directly from disk offline.

## Results

All scoped checks passed. The [state/source result](compiler_kernel_loader_model.json)
covers 24 scenarios, sequential table appends, partial errors, exact cache
skips and immutable/resettable state. It also checks the exact Pointer64
equation and eight-byte little-endian encoding, the pinned LLVM phase order,
the Resolved/Ready gate, separate numerical loads/adds/stores, explicit host
copying and fresh second-call argument pointers.

The [browser result](compiler_kernel_loader_browser.json) covers 12 route/target
combinations and verifies that text fits its backing at every intermediate
event. It checks working-to-executable memory, the relocation word, the empty
table while symbols are Resolved, two pointer indirections, value/output
updates, completion before copying and the fresh B2/O2 second call. It also
checks every bounded full-source link, the exact offline upstream fixup
statement, preserved text, connected/deeper trees, movable Hold/Expand/Close
cards, the real add-one D.1 link, Chrome fullscreen, persistent byte/register
motion, live Pause/Resume, 320/390px touch layout and no-JavaScript reading.
Logs contained zero page errors and zero external runtime requests.

Source checks for the four existing example readers passed. Saved frames
were reviewed for relocation bytes, launch-pointer arrows, separate working
values, explicit CPU copying and the fresh second-call output. They verify
browser teaching-model behavior; they do not establish MAX/GPU execution.
