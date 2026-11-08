# Four MAX repository example readers

Validated November 6, 2026 against SOURCE revision
`610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
The [compiler workbench](../link_htmls/max_mlir_mojo_compiler_and_runtime.html#examples)
links all four standalone readers. Each contains graph construction,
compile/init and numerical execution scenes, with original Python/Mojo sources.

| Reader | Source / arithmetic / important distinction |
| --- | --- |
| [01 · Simple](../link_htmls/max_example_01_add_one_compilation_and_execution.html) | `addition.py`: X+1; explicit output type avoids the optional unimplemented shape function. The script prints results rather than asserting parity. |
| [02 · Medium](../link_htmls/max_example_02_matmul_fusible_epilogue.html) | `elementwise_addition.py`: (A@B+1)×2. Fusion is eligible; the fused/separate comparison is illustrative, not a compiler dump or real flag. |
| [03 · Medium-hard](../link_htmls/max_example_03_tiled_matmul_compiler_and_memory.html) | `matrix_multiplication.py`: compile-time algorithm; GPU output memset, shared tiles, copy wait, block barriers and register accumulation. CPU uses naive. |
| [04 · Hard](../link_htmls/max_example_04_fused_attention_online_softmax.html) | `fused_attention.py`: unscaled rank-2 attention, running m/l/normalized O. GPU key BN_1=8 differs from query BN=16. The script prints Buffers without a host copy or parity assertion. |

## Evidence and results

- [Model/arithmetic checks](compiler_examples_model.json): 18 target/variant
  combinations. Independent matmul and full-softmax references agree with
  the reduced numerical scenes. Graph/artifact/executable states, copy/barrier
  ordering, CPU branches and immutable seek/reset snapshots pass.
- [Rendered checks](compiler_examples_browser.json): 430 complete/intermediate
  render audits, plus four scoped final legend audits. Actor text bounds,
  actual text/outline contrast, later-actor occlusion and document overflow
  pass. Visible output values agree with the model.
- Persistent shared-tile nodes visibly move. Pause freezes both their current
  pose and semantic stepping; Resume continues; offscreen scenes pause; Reset
  clears tile/result resources. No autoplay.
- All four readers work offline and at 320/390px with contained diagram
  scrolling. Hover/focus/touch card dismissal, click-only L, eleven matching
  colors, original-source selection/search/copy/download, no-JS and print
  reading pass.
- [Scoped source checks](compiler_examples_sources.json): four pages, twelve
  scenes, twenty complete pinned source files, 128 hashed source spans and
  196 local links. [Full artifact checks](compiler_examples_full_sources.json)
  also pass without changing the existing Serve source-check report.
- Parent workbench regression: ten model/source check groups and
  eleven browser groups / 72 render audits pass after the shared card/source
  UI extraction and progression links.
- Skill frontmatter validation and `git diff --check` pass. The authoring
  catalog points to the new example models and scoped checks.

## Commands

From `LLAMA_2_7B_GRAPH_WALKTHROUGH_PLAN/`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_examples.py
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_workbench.py
node src/check_compiler_examples.cjs
PYTHONDONTWRITEBYTECODE=1 python3 src/check_compiler_examples.py
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_compiler_examples_browser.cjs
node src/check_compiler_workbench.cjs
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_compiler_workbench_browser.cjs
PYTHONDONTWRITEBYTECODE=1 python3 src/validate.py --report validation/compiler_examples_full_sources.json
```

The `NODE_PATH` above records this host's available Playwright installation;
discover the installed location when replaying elsewhere. These CPU-RUN checks
execute JavaScript arithmetic and Chromium UI behavior. No MAX/Mojo compilation,
real example inference, GPU timing, benchmark or production fusion outcome was
measured. IR/artifact sheets and device motion remain illustrations.

## Representative states

- [Epilogue applies before store](compiler_example_epilogue_mechanism.png).
- [Copies ready while the block gate waits](compiler_example_matmul_mechanism.png).
- [Attention rescales previous O after a larger maximum](compiler_example_attention_mechanism.png).

## Add-one Text call trees · November 8, 2026

The [add-one Text reader](../link_htmls/max_example_01_add_one_compilation_and_execution.html#toolchain-call-graph)
now starts with four annotated call trees: Graph/package creation,
InferenceSession construction, compile/MEF preparation and executable
initialization. Each has input, output and example cards. Nine expandable
branches cover package serialization, registration, ownership, compiler
passes/JIT, runtime loading, gates and the later input/result buffers.
The complete earlier answers 1–7 remain in a collapsible reference and in
copy/download text.

Evidence remains pinned to `610f7111836d9a950d45ba52ec9b3e0ed589fe3f`.
[Source validation](compiler_examples_sources.json) passed for four pages,
74 complete source files, 413 hashed source spans and 1,968 local links.
[Browser validation](compiler_examples_browser.json) passed 12 groups and
904 render audits, including all nine branches at 320/390px, native/package
source links, preserved old question links and paused animation state.
[The final compiler-link check](addition_calltrees_browser.json) also passed
at 1600/320/390px and with JavaScript disabled for print. These checks inspect
documentation and browser behavior; no MAX/Mojo compilation or inference ran.

Scoped commands from this task directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 src/compiler_examples.py
PYTHONDONTWRITEBYTECODE=1 python3 src/check_compiler_examples.py
NODE_PATH=/tmp/max-os-playwright/node_modules node src/check_compiler_examples_browser.cjs
git diff --check -- .
```

Representative call trees:
[Graph](compiler_example_addition_calltree_graph.png),
[session](compiler_example_addition_calltree_session.png),
[compile](compiler_example_addition_calltree_compile.png),
[init](compiler_example_addition_calltree_init.png) and
[mobile](compiler_example_addition_calltree_mobile.png).
